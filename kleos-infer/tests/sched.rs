/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Scheduler and budget behaviour with a fake model (no weights). The
//! budget is process-global, so the tests that touch it hold `GLOBAL`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

use kleos_infer::budget::{self, ModelSlot};
use kleos_infer::sched::{with_priority, BatchFns, Batcher, Gate, Priority};

static GLOBAL: Mutex<()> = Mutex::new(());

/// A fake forward pass: each row's vector is its tokens summed, after a
/// delay proportional to the batch's padded tokens.
fn fake_forward(rows: &[&Vec<u32>], per_token: Duration, log: &Mutex<Vec<(usize, usize)>>, lane_of: &dyn Fn(&Vec<u32>) -> usize) -> Vec<Vec<f32>> {
    let l = rows.iter().map(|r| r.len()).max().unwrap_or(1);
    std::thread::sleep(per_token * (rows.len() * l) as u32);
    let lane = lane_of(rows[0]);
    log.lock().unwrap().push((lane, rows.len()));
    rows.iter().map(|r| vec![r.iter().map(|&x| x as f32).sum()]).collect()
}

#[test]
fn rows_come_back_in_order_and_batch_across_callers() {
    let b: Arc<Batcher<Vec<u32>>> = Arc::new(Batcher::new());
    let gate = Arc::new(Gate::new());
    let log = Arc::new(Mutex::new(vec![]));
    let n = 40;
    let bar = Arc::new(Barrier::new(n));
    let hs: Vec<_> = (0..n)
        .map(|c| {
            let (b, gate, log, bar) = (b.clone(), gate.clone(), log.clone(), bar.clone());
            std::thread::spawn(move || {
                let rows: Vec<Vec<u32>> = (0..3).map(|k| vec![(c * 10 + k) as u32; 4]).collect();
                let max_rows = |_l: usize| 64;
                let act = |_n: usize, _l: usize| 0;
                let fwd = |rs: &[&Vec<u32>]| Ok(fake_forward(rs, Duration::from_micros(200), &log, &|_| 0));
                let after = || {};
                let f = BatchFns { max_rows: &max_rows, act_bytes: &act, forward: &fwd, after: &after, gate: &gate };
                bar.wait();
                let out = b.run(rows, Priority::Interactive, &f).unwrap();
                for (k, v) in out.iter().enumerate() {
                    assert_eq!(v[0], ((c * 10 + k) * 4) as f32, "caller {c} row {k}");
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    let log = log.lock().unwrap();
    let rows: usize = log.iter().map(|(_, r)| r).sum();
    assert_eq!(rows, n * 3);
    assert!(log.len() < n, "expected batching across callers, got {} passes for {n} callers", log.len());
}

#[test]
fn interactive_goes_first_and_bulk_is_not_starved() {
    let b: Arc<Batcher<Vec<u32>>> = Arc::new(Batcher::new());
    let gate = Arc::new(Gate::new());
    let log = Arc::new(Mutex::new(vec![]));
    // bulk rows are tagged 1_000_000+, interactive rows below
    let lane_of = |r: &Vec<u32>| if r[0] >= 1_000_000 { 1 } else { 0 };
    let bulk = {
        let (b, gate, log) = (b.clone(), gate.clone(), log.clone());
        std::thread::spawn(move || {
            let rows: Vec<Vec<u32>> = (0..200).map(|k| vec![1_000_000 + k as u32; 32]).collect();
            let max_rows = |_l: usize| 8;
            let act = |_n: usize, _l: usize| 0;
            let fwd = |rs: &[&Vec<u32>]| Ok(fake_forward(rs, Duration::from_micros(50), &log, &lane_of));
            let after = || {};
            let f = BatchFns { max_rows: &max_rows, act_bytes: &act, forward: &fwd, after: &after, gate: &gate };
            let t = Instant::now();
            let out = b.run(rows, Priority::Bulk, &f).unwrap();
            assert_eq!(out.len(), 200);
            t.elapsed()
        })
    };
    std::thread::sleep(Duration::from_millis(30)); // bulk is running
    let lat = Arc::new(Mutex::new(vec![]));
    let hs: Vec<_> = (0..20)
        .map(|c| {
            let (b, gate, log, lat) = (b.clone(), gate.clone(), log.clone(), lat.clone());
            std::thread::spawn(move || {
                for i in 0..5 {
                    let max_rows = |_l: usize| 8;
                    let act = |_n: usize, _l: usize| 0;
                    let fwd = |rs: &[&Vec<u32>]| Ok(fake_forward(rs, Duration::from_micros(50), &log, &lane_of));
                    let after = || {};
                    let f = BatchFns { max_rows: &max_rows, act_bytes: &act, forward: &fwd, after: &after, gate: &gate };
                    let t = Instant::now();
                    b.run(vec![vec![(c * 100 + i) as u32; 8]], Priority::Interactive, &f).unwrap();
                    lat.lock().unwrap().push(t.elapsed());
                    std::thread::sleep(Duration::from_millis(5));
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    let bulk_time = bulk.join().unwrap();
    let mut lat = lat.lock().unwrap().clone();
    lat.sort();
    let p95 = lat[(lat.len() * 95) / 100];
    // one bulk slice is 8 rows × 32 tokens × 50 µs ≈ 13 ms: an interactive
    // call waits at most a slice plus an interactive batch or two
    assert!(p95 < Duration::from_millis(80), "interactive p95 {p95:?}");
    // bulk kept progressing while interactive calls kept coming
    let log = log.lock().unwrap();
    let bulk_passes = log.iter().filter(|(l, _)| *l == 1).count();
    assert!(bulk_passes >= 25, "bulk passes {bulk_passes}");
    assert!(bulk_time < Duration::from_secs(10));
}

#[test]
fn backpressure_bounds_queued_tokens() {
    // lanes of 64 tokens: a 1000-row bulk caller never has more than a lane's
    // worth queued, yet finishes
    let b: Arc<Batcher<Vec<u32>>> = Arc::new(Batcher::with_caps(64, 64));
    let gate = Gate::new();
    let peak_batch = AtomicUsize::new(0);
    let max_rows = |_l: usize| 1000;
    let act = |_n: usize, _l: usize| 0;
    let fwd = |rs: &[&Vec<u32>]| {
        peak_batch.fetch_max(rs.len(), Ordering::Relaxed);
        Ok(rs.iter().map(|r| vec![r[0] as f32]).collect())
    };
    let after = || {};
    let f = BatchFns { max_rows: &max_rows, act_bytes: &act, forward: &fwd, after: &after, gate: &gate };
    let rows: Vec<Vec<u32>> = (0..1000).map(|k| vec![k as u32; 8]).collect();
    let out = b.run(rows, Priority::Bulk, &f).unwrap();
    assert_eq!(out.len(), 1000);
    assert!(out.iter().enumerate().all(|(k, v)| v[0] == k as f32));
    // 64 tokens / 8 per row: never more than 8 rows queued, so no pass is bigger
    assert!(peak_batch.load(Ordering::Relaxed) <= 8, "peak batch {}", peak_batch.load(Ordering::Relaxed));
}

#[test]
fn activations_stay_within_the_budget() {
    let _g = GLOBAL.lock().unwrap();
    budget::set_memory_budget(100 << 20, Some(64 << 20));
    let gate_a = Arc::new(Gate::new());
    let gate_b = Arc::new(Gate::new());
    let peak = Arc::new(AtomicUsize::new(0));
    let hs: Vec<_> = (0..16)
        .map(|i| {
            let gate = if i % 2 == 0 { gate_a.clone() } else { gate_b.clone() };
            let peak = peak.clone();
            std::thread::spawn(move || {
                for _ in 0..10 {
                    let _run = gate.acquire(Priority::Interactive, 40 << 20);
                    peak.fetch_max(budget::stats().active, Ordering::Relaxed);
                    std::thread::sleep(Duration::from_millis(1));
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    // two models' gates, 40 MiB passes, 100 MiB budget: never both at once
    assert!(peak.load(Ordering::Relaxed) <= 100 << 20, "peak {}", peak.load(Ordering::Relaxed));
    assert_eq!(budget::stats().active, 0);
    budget::set_memory_budget(budget::DEFAULT_BUDGET, Some(budget::DEFAULT_ACTIVATION_CAP));
}

struct Fake(usize);

#[test]
fn idle_models_unload_reload_and_get_evicted() {
    let _g = GLOBAL.lock().unwrap();
    budget::set_memory_budget(1000 << 20, None);
    let loads = AtomicUsize::new(0);
    let a: ModelSlot<Fake> = ModelSlot::new("test:a");
    let b: ModelSlot<Fake> = ModelSlot::new("test:b");
    let w0 = budget::stats().weights;
    a.get_or_load(|| { loads.fetch_add(1, Ordering::Relaxed); Ok(Fake(600 << 20)) }, |f| f.0).unwrap();
    assert!(a.is_loaded());
    assert_eq!(budget::stats().weights, w0 + (600 << 20));
    // a second 600 MiB model does not fit next to the first: the idle one goes
    b.get_or_load(|| { loads.fetch_add(1, Ordering::Relaxed); Ok(Fake(600 << 20)) }, |f| f.0).unwrap();
    assert!(!a.is_loaded(), "the idle model should have been evicted");
    assert!(b.is_loaded());
    assert_eq!(budget::stats().weights, w0 + (600 << 20));
    // a model that is running is not evicted
    {
        let _hold = b.hold();
        a.get_or_load(|| { loads.fetch_add(1, Ordering::Relaxed); Ok(Fake(600 << 20)) }, |f| f.0).unwrap();
        assert!(b.is_loaded(), "a busy model must not be evicted");
    }
    // idle unloading, then a lazy reload
    let n = budget::reap_idle(0);
    assert!(n >= 2, "reaped {n}");
    assert!(!a.is_loaded() && !b.is_loaded());
    assert_eq!(budget::stats().weights, w0);
    a.get_or_load(|| { loads.fetch_add(1, Ordering::Relaxed); Ok(Fake(1 << 20)) }, |f| f.0).unwrap();
    assert!(a.is_loaded());
    assert_eq!(loads.load(Ordering::Relaxed), 4);
    drop(a);
    drop(b);
    assert_eq!(budget::stats().weights, w0);
    budget::set_memory_budget(budget::DEFAULT_BUDGET, None);
}

#[test]
fn shared_returns_one_instance_per_key() {
    let a = budget::shared("test-key", || Ok(Arc::new(Fake(1)))).unwrap();
    let b = budget::shared("test-key", || Ok(Arc::new(Fake(2)))).unwrap();
    assert!(Arc::ptr_eq(&a, &b));
    assert_eq!(b.0, 1);
    drop((a, b));
    let c = budget::shared("test-key", || Ok(Arc::new(Fake(3)))).unwrap();
    assert_eq!(c.0, 3, "a dropped instance is rebuilt");
}

#[test]
fn priority_rides_on_the_thread() {
    use kleos_infer::sched::current_priority;
    assert_eq!(current_priority(), Priority::Interactive);
    with_priority(Priority::Bulk, || {
        assert_eq!(current_priority(), Priority::Bulk);
        with_priority(Priority::Interactive, || assert_eq!(current_priority(), Priority::Interactive));
        assert_eq!(current_priority(), Priority::Bulk);
    });
    assert_eq!(current_priority(), Priority::Interactive);
    assert_eq!(Priority::parse("background").unwrap(), Priority::Bulk);
    assert!(Priority::parse("urgent").is_err());
}
