/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Many concurrent callers on the built-in EmbeddingGemma: memory (process
//! footprint, which includes Metal's shared buffers), throughput and the
//! latency of interactive calls.
//!
//! ```text
//! cargo +1.99.0 run --release -p kleos --features infer-metal --example concurrency_bench -- \
//!     embed <callers> <calls per caller> [metal|cpu]
//! cargo +1.99.0 run --release -p kleos --features infer-metal --example concurrency_bench -- \
//!     mixed <conversations> <turns per conversation> [metal|cpu] [bulk chunks, default 2000]
//! ```
//!
//! `embed`: every caller embeds one fresh query at a time (no cache hits).
//! `mixed`: a bulk stream of 300-word document chunks (16 per call, like a
//! large PDF being ingested), plus conversations doing `remember` + `recall`,
//! plus an agent-trace stream of short `remember`s. Interactive latency is the
//! conversations' `recall`s. Prints one JSON line.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use kleos::embed::{self, EmbedderSpec};
use kleos::memory::{InitOpts, RecallOpts, RememberOpts};
use kleos::DbInstance;

const WORDS: [&str; 48] = [
    "harbour", "invoice", "quarterly", "margin", "sister", "Lisbon", "contract", "clause", "pipeline", "budget",
    "meeting", "deadline", "customer", "refund", "server", "outage", "garden", "piano", "flight", "hotel",
    "doctor", "policy", "claim", "premium", "engine", "report", "revenue", "forecast", "trial", "dosage",
    "terminal", "crane", "cargo", "volume", "approach", "channel", "dredged", "depth", "night", "desk",
    "project", "review", "vendor", "license", "renewal", "notice", "termination", "party",
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn words(&mut self, n: usize) -> String {
        let mut s = String::with_capacity(n * 8);
        for i in 0..n {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(WORDS[(self.next() % WORDS.len() as u64) as usize]);
            if self.next() % 7 == 0 {
                s.push_str(&format!(" {}", self.next() % 10_000));
            }
        }
        s
    }
}

/// (current, lifetime max) physical footprint in bytes (macOS).
fn footprint() -> (u64, u64) {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut info: libc::rusage_info_v4 = std::mem::zeroed();
        let ok = libc::proc_pid_rusage(
            std::process::id() as i32,
            libc::RUSAGE_INFO_V4,
            &mut info as *mut _ as *mut libc::rusage_info_t,
        );
        if ok == 0 {
            return (info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint);
        }
    }
    (0, 0)
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

fn main() -> miette::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or_else(|| "embed".into());
    let callers: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let per: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let device = args.get(4).cloned().unwrap_or_else(|| "metal".into());
    let bulk_chunks: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(2000);

    embed::enable_embedder(
        "gemma",
        EmbedderSpec { model: Some("embeddinggemma".into()), precision: Some("q8".into()), device: Some(device.clone()), dim: None },
    )?;
    // load and warm the kernels outside the measurement
    embed::embed_texts("gemma", &["warm up the kernels once"])?;
    let (base, _) = footprint();

    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(base));
    let sampler = {
        let (stop, peak) = (stop.clone(), peak.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let (cur, _) = footprint();
                peak.fetch_max(cur, Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(50));
            }
        })
    };

    let lat = Arc::new(Mutex::new(Vec::<f64>::new()));
    let texts_done = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(AtomicUsize::new(0));
    let t0 = Instant::now();
    let mut extra = serde_json::Map::new();

    match mode.as_str() {
        "embed" => {
            let hs: Vec<_> = (0..callers)
                .map(|c| {
                    let (lat, done, errs) = (lat.clone(), texts_done.clone(), errors.clone());
                    std::thread::spawn(move || {
                        let mut r = Rng(0x9e37 + c as u64 * 7919);
                        for i in 0..per {
                            let q = format!("q{c}-{i} {}", r.words(12));
                            let t = Instant::now();
                            match embed::embed_texts("gemma_query", &[q.as_str()]) {
                                Ok(_) => {
                                    lat.lock().unwrap().push(t.elapsed().as_secs_f64() * 1000.0);
                                    done.fetch_add(1, Ordering::Relaxed);
                                }
                                Err(_) => {
                                    errs.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    })
                })
                .collect();
            for h in hs {
                let _ = h.join();
            }
        }
        "mixed" => {
            let db = DbInstance::new("mem", "", "")?;
            db.memory_init(InitOpts { embedder: Some("gemma".into()), ..Default::default() })?;
            let bulk_done = Arc::new(AtomicBool::new(false));
            let bulk_texts = Arc::new(AtomicUsize::new(0));
            let bulk = {
                let (bt, bd, errs) = (bulk_texts.clone(), bulk_done.clone(), errors.clone());
                std::thread::spawn(move || {
                    let mut r = Rng(42);
                    let mut i = 0;
                    while i < bulk_chunks {
                        let batch: Vec<String> =
                            (0..16.min(bulk_chunks - i)).map(|k| format!("p{} {}", i + k, r.words(300))).collect();
                        let refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
                        let r = kleos::sched::with_priority(kleos::sched::Priority::Bulk, || embed::embed_texts("gemma", &refs));
                        if r.is_err() {
                            errs.fetch_add(1, Ordering::Relaxed);
                        }
                        i += batch.len();
                        bt.fetch_add(batch.len(), Ordering::Relaxed);
                    }
                    bd.store(true, Ordering::Relaxed);
                })
            };
            let traces = {
                let (db, bd, errs) = (db.clone(), bulk_done.clone(), errors.clone());
                std::thread::spawn(move || {
                    let mut r = Rng(7);
                    let mut n = 0usize;
                    while !bd.load(Ordering::Relaxed) {
                        let t = format!("tool call {n}: fetched {} rows for {}", r.next() % 500, r.words(6));
                        let o = RememberOpts {
                            ns: Some("traces".into()),
                            split: Some("none".into()),
                            priority: Some("bulk".into()),
                            ..Default::default()
                        };
                        if db.remember(&t, o).is_err() {
                            errs.fetch_add(1, Ordering::Relaxed);
                        }
                        n += 1;
                    }
                    n
                })
            };
            let convs: Vec<_> = (0..callers)
                .map(|c| {
                    let (db, lat, done, errs) = (db.clone(), lat.clone(), texts_done.clone(), errors.clone());
                    std::thread::spawn(move || {
                        let mut r = Rng(1000 + c as u64);
                        let ns = format!("conv{c}");
                        for i in 0..per {
                            let said = format!("User {c} turn {i}: {}", r.words(18));
                            let o = RememberOpts { ns: Some(ns.clone()), ..Default::default() };
                            if db.remember(&said, o).is_err() {
                                errs.fetch_add(1, Ordering::Relaxed);
                            }
                            let q = format!("what did user {c} say about {}", r.words(3));
                            let t = Instant::now();
                            let o = RecallOpts { ns: Some(ns.clone()), k: Some(5), ..Default::default() };
                            match db.recall(&q, o) {
                                Ok(_) => {
                                    lat.lock().unwrap().push(t.elapsed().as_secs_f64() * 1000.0);
                                    done.fetch_add(2, Ordering::Relaxed);
                                }
                                Err(_) => {
                                    errs.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    })
                })
                .collect();
            let conv_t0 = Instant::now();
            for h in convs {
                let _ = h.join();
            }
            let conv_wall = conv_t0.elapsed().as_secs_f64();
            let bulk_t0 = Instant::now();
            let _ = bulk.join();
            let tr = traces.join().unwrap_or(0);
            extra.insert("conversations_wall_s".into(), conv_wall.into());
            extra.insert("bulk_tail_s".into(), bulk_t0.elapsed().as_secs_f64().into());
            extra.insert("bulk_chunks".into(), bulk_texts.load(Ordering::Relaxed).into());
            extra.insert("traces".into(), tr.into());
        }
        m => miette::bail!("unknown mode {m}"),
    }

    let wall = t0.elapsed().as_secs_f64();
    stop.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    let (cur, life) = footprint();
    let mut l = lat.lock().unwrap().clone();
    let mut out = serde_json::json!({
        "mode": mode, "callers": callers, "per_caller": per, "device": device,
        "wall_s": wall,
        "interactive_ops": l.len(),
        "interactive_p50_ms": pct(&mut l, 0.5), "interactive_p95_ms": pct(&mut l, 0.95),
        "interactive_max_ms": pct(&mut l, 1.0),
        "ops_per_s": texts_done.load(Ordering::Relaxed) as f64 / wall,
        "errors": errors.load(Ordering::Relaxed),
        "footprint_base_mb": base as f64 / 1e6,
        "footprint_peak_mb": peak.load(Ordering::Relaxed) as f64 / 1e6,
        "footprint_end_mb": cur as f64 / 1e6,
        "footprint_lifetime_max_mb": life as f64 / 1e6,
    });
    for (k, v) in extra {
        out[k] = v;
    }
    println!("{out}");
    Ok(())
}
