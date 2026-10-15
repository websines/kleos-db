/*
 * Copyright 2026, The kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Live-subscription overhead benchmark (UPGRADE_PLAN F1).
//!
//! `cargo run --release -p kleos --example livebench [writes] [paced_writes] [mem|sqlite]`
//!
//! mem backend by default (sqlite: a fresh file under `$TMPDIR`). Note that
//! the mem backend holds one store-wide lock for a transaction's whole life,
//! so a reader (the subscription worker included) excludes writers. Two relations, `hot` (written) and `cold` (never written),
//! each preloaded with 1000 rows. N subscriptions are point lookups
//! `?[v] := *REL{k: $k, v}` with a distinct `$k` each, so dedup does not
//! merge them: every one is its own group and its own evaluation.
//!
//! * Throughput: one thread issues `writes` single-row `:put hot` transactions
//!   as fast as it can (new keys, so no subscribed result changes; every
//!   subscription on `hot` is still re-evaluated). Reported: txns/s, writer
//!   thread CPU, process CPU (includes the subscription worker), worker
//!   re-evaluations, and the time until the worker had caught up.
//! * Latency: the same N subscriptions plus one probe subscription on
//!   `hot{k: 1_000_000}`, whose key sorts last, so the worker evaluates it
//!   after all N others. `paced_writes` writes to that key, one every 2 ms.
//!   Latency of write i = from the start of its `:put` call to the probe
//!   receiving the first delta whose value is >= i (its effect became
//!   visible; a write coalesced into a later delta waits for that delta).
//!   It includes the write itself; the write's median is printed too.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use kleos::{DataValue, DbInstance, ScriptMutability, SubscribeOptions, SubscriptionEvent};

#[repr(C)]
struct Timespec {
    sec: i64,
    nsec: i64,
}

extern "C" {
    fn clock_gettime(clk: i32, ts: *mut Timespec) -> i32;
}

#[cfg(target_os = "macos")]
const PROCESS_CPU: i32 = 12;
#[cfg(target_os = "macos")]
const THREAD_CPU: i32 = 16;
#[cfg(not(target_os = "macos"))]
const PROCESS_CPU: i32 = 2;
#[cfg(not(target_os = "macos"))]
const THREAD_CPU: i32 = 3;

fn cpu(clk: i32) -> Duration {
    let mut ts = Timespec { sec: 0, nsec: 0 };
    unsafe { clock_gettime(clk, &mut ts) };
    Duration::new(ts.sec as u64, ts.nsec as u32)
}

fn run(db: &DbInstance, script: &str, params: BTreeMap<String, DataValue>) {
    db.run_script(script, params, ScriptMutability::Mutable)
        .unwrap_or_else(|e| panic!("{e:?}\n{script}"));
}

fn setup() -> DbInstance {
    let db = match std::env::args().nth(3).as_deref() {
        Some("sqlite") => {
            let dir = std::env::temp_dir().join(format!(
                "livebench-{}-{}.db",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            DbInstance::new("sqlite", dir, "").unwrap()
        }
        _ => DbInstance::new("mem", "", "").unwrap(),
    };
    run(&db, ":create hot {k: Int => v: Int}", BTreeMap::new());
    run(&db, ":create cold {k: Int => v: Int}", BTreeMap::new());
    for rel in ["hot", "cold"] {
        run(
            &db,
            &format!("?[k, v] := k in int_range(1000), v = k :put {rel} {{k => v}}"),
            BTreeMap::new(),
        );
    }
    db
}

fn subscribe_n(db: &DbInstance, rel: &str, n: usize) -> Vec<kleos::Subscription> {
    (0..n)
        .map(|i| {
            let s = db
                .subscribe(
                    &format!("?[v] := *{rel}{{k: $k, v}}"),
                    BTreeMap::from([("k".to_string(), DataValue::from(i as i64))]),
                    SubscribeOptions::default(),
                )
                .unwrap();
            // Drain the Initial so queues do not matter.
            s.try_recv();
            s
        })
        .collect()
}

fn wait_idle(db: &DbInstance) -> Duration {
    let start = Instant::now();
    let mut last = db.live_stats().evaluations;
    let mut stable_since = Instant::now();
    loop {
        std::thread::sleep(Duration::from_millis(2));
        let now = db.live_stats().evaluations;
        if now != last {
            last = now;
            stable_since = Instant::now();
        } else if stable_since.elapsed() > Duration::from_millis(100) {
            return stable_since - start;
        }
    }
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn throughput(n: usize, rel: &str, writes: usize) -> (f64, String) {
    let db = setup();
    let _subs = subscribe_n(&db, rel, n);
    let evals0 = db.live_stats().evaluations;
    let p0 = cpu(PROCESS_CPU);
    let t0 = cpu(THREAD_CPU);
    let start = Instant::now();
    for i in 0..writes {
        run(
            &db,
            "?[k, v] <- [[$k, 1]] :put hot {k => v}",
            BTreeMap::from([("k".to_string(), DataValue::from(10_000 + i as i64))]),
        );
    }
    let wall = start.elapsed();
    let writer_cpu = cpu(THREAD_CPU) - t0;
    let catch_up = wait_idle(&db);
    let proc_cpu = cpu(PROCESS_CPU) - p0;
    let evals = db.live_stats().evaluations - evals0;
    let rounds = if n > 0 && rel == "hot" {
        format!("{:.0}", evals as f64 / n as f64)
    } else {
        "-".into()
    };
    let row = format!(
        "| {n:>4} | {rel:>4} | {:>8.0} | {:>7.1} | {:>8.1} | {:>8.1} | {evals:>8} | {rounds:>6} | {:>8.1} |",
        writes as f64 / wall.as_secs_f64(),
        wall.as_secs_f64() * 1e6 / writes as f64,
        writer_cpu.as_secs_f64() * 1e6 / writes as f64,
        proc_cpu.as_secs_f64() * 1e3,
        catch_up.as_secs_f64() * 1e3,
    );
    (writes as f64 / wall.as_secs_f64(), row)
}

fn latency(n: usize, rel: &str, writes: usize) -> (f64, String) {
    let db = setup();
    let _subs = subscribe_n(&db, rel, n);
    const PROBE: i64 = 1_000_000;
    run(
        &db,
        "?[k, v] <- [[$k, -1]] :put hot {k => v}",
        BTreeMap::from([("k".to_string(), DataValue::from(PROBE))]),
    );
    let probe = db
        .subscribe(
            "?[v] := *hot{k: $k, v}",
            BTreeMap::from([("k".to_string(), DataValue::from(PROBE))]),
            SubscribeOptions {
                queue_capacity: 1 << 16,
            },
        )
        .unwrap();
    probe.recv().unwrap();
    let base = Instant::now();
    let issued: Arc<Vec<AtomicU64>> = Arc::new((0..writes).map(|_| AtomicU64::new(0)).collect());
    let issued2 = issued.clone();
    let reader = std::thread::spawn(move || {
        let mut lat = vec![];
        let mut deltas = 0usize;
        let mut next_unseen = 0usize;
        while next_unseen < writes {
            match probe.recv_timeout(Duration::from_secs(20)) {
                Some(SubscriptionEvent::Delta { added, .. }) => {
                    let at = base.elapsed().as_nanos() as u64;
                    let v = added.rows[0][0].get_int().unwrap() as usize;
                    deltas += 1;
                    for t in &issued2[next_unseen..=v] {
                        lat.push((at - t.load(Ordering::Acquire)) as f64 / 1e3);
                    }
                    next_unseen = v + 1;
                }
                Some(e) => panic!("unexpected {e:?}"),
                None => break,
            }
        }
        (lat, deltas)
    });
    let mut write_us = vec![];
    for i in 0..writes {
        let t = base.elapsed().as_nanos() as u64;
        issued[i].store(t.max(1), Ordering::Release);
        let s = Instant::now();
        run(
            &db,
            "?[k, v] <- [[$k, $v]] :put hot {k => v}",
            BTreeMap::from([
                ("k".to_string(), DataValue::from(PROBE)),
                ("v".to_string(), DataValue::from(i as i64)),
            ]),
        );
        write_us.push(s.elapsed().as_secs_f64() * 1e6);
        std::thread::sleep(Duration::from_millis(2));
    }
    let (mut lat, deltas) = reader.join().unwrap();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    write_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let row = format!(
        "| {n:>4} | {rel:>4} | {:>9.0} | {:>9.0} | {:>9.0} | {:>5}/{:<5} | {:>7.0} |",
        pct(&lat, 0.5),
        pct(&lat, 0.99),
        lat.last().copied().unwrap_or(f64::NAN),
        deltas,
        lat.len(),
        pct(&write_us, 0.5),
    );
    // Lower p50 is better: score is its negation.
    (-pct(&lat, 0.5), row)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let writes: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let paced: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1_000);

    let backend = args.get(3).cloned().unwrap_or_else(|| "mem".into());
    println!("Backend: {backend}");
    println!("Throughput: {writes} single-row :put txns on `hot`, one writer thread (best of LIVEBENCH_REPS runs)");
    println!("|    N | subs |  txns/s  | µs/txn  | writer CPU µs/txn | proc CPU ms | re-evals | rounds | catch-up ms |");
    println!("|------|------|----------|---------|----------|----------|----------|--------|----------|");
    let reps: usize = std::env::var("LIVEBENCH_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let configs = [
        (0, "-"),
        (10, "cold"),
        (10, "hot"),
        (100, "cold"),
        (100, "hot"),
        (1000, "cold"),
        (1000, "hot"),
    ];
    // The machine is shared: run every config `reps` times, interleaved, and
    // keep each config's best run (highest txns/s, lowest p50 latency).
    let best = |f: &dyn Fn(usize, &str) -> (f64, String)| {
        let mut rows: Vec<Option<(f64, String)>> = vec![None; configs.len()];
        for _ in 0..reps {
            for (i, (n, rel)) in configs.iter().enumerate() {
                let r = f(*n, rel);
                if rows[i].as_ref().map_or(true, |b| r.0 > b.0) {
                    rows[i] = Some(r);
                }
            }
        }
        for r in rows {
            println!("{}", r.unwrap().1);
        }
    };
    best(&|n, rel| throughput(n, rel, writes));
    println!();
    println!("Latency: {paced} paced writes (2 ms apart) to the probe key; µs from :put call to probe delivery");
    println!("|    N | subs |   p50 µs  |   p99 µs  |   max µs  | deltas/writes | put p50 µs |");
    println!("|------|------|-----------|-----------|-----------|-------------|---------|");
    best(&|n, rel| latency(n, rel, paced));
}
