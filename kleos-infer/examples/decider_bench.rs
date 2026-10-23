/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Latency and memory of decider-nano on short memory-note
//! decisions, rendered with the kleos `supersedes` template (NLI, 3 options).
//!
//! ```text
//! cargo +1.99.0 run --release -p kleos-infer [--features metal|accelerate] --example decider_bench -- \
//!     <cpu|metal> <f32|f16|bf16|q8|q4> [--model DIR_OR_REPO] [--json]
//! ```
//!
//! Memory is the process **physical footprint** (what Activity Monitor and
//! `footprint` report, including Metal buffers and the allocator's cached
//! free blocks) and RSS, at: start, after load, after a batch of 16, and the
//! lifetime peak (= load peak, since load is the largest transient).
//! Run with `MallocLargeCache=0` to see what the code holds rather than what
//! libmalloc keeps cached (docs/specs/embedder-toggle.md).
//!
//! `weights_cached_pct` is how much of `model.safetensors` was in the OS page
//! cache before loading. `load_ms` only measures the loader when it is near
//! 100: a cold load is a plain disk read of the whole checkpoint (300 MB, at
//! whatever the disk delivers), the same for every precision. Under memory
//! pressure macOS drops the file from the cache within a minute or two, so
//! back-to-back runs can still be cold.

use std::time::Instant;

use kleos_infer::*;

/// `(phys_footprint, lifetime_max_phys_footprint, resident_size)` in MB.
#[cfg(target_os = "macos")]
fn mem() -> (f64, f64, f64) {
    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
    }
    // rusage_info_v4 as u64 words: [0..2) uuid, 8 resident_size,
    // 9 phys_footprint, 30 lifetime_max_phys_footprint
    let mut b = [0u64; 64];
    let r = unsafe { proc_pid_rusage(std::process::id() as i32, 4, b.as_mut_ptr()) };
    if r != 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mb = |x: u64| (x as f64 / 1048576.0).round();
    (mb(b[9]), mb(b[30]), mb(b[8]))
}

#[cfg(not(target_os = "macos"))]
fn mem() -> (f64, f64, f64) {
    (f64::NAN, f64::NAN, f64::NAN)
}

/// Percent of `path`'s pages resident in the OS page cache (`mincore`).
#[cfg(unix)]
fn cached_pct(path: &std::path::Path) -> f64 {
    extern "C" {
        fn getpagesize() -> i32;
        fn mincore(addr: *const std::ffi::c_void, len: usize, vec: *mut std::ffi::c_char) -> i32;
    }
    let Ok(f) = std::fs::File::open(path) else { return f64::NAN };
    // SAFETY: a read-only map whose pages are never touched, only queried.
    let Ok(m) = (unsafe { memmap2::Mmap::map(&f) }) else { return f64::NAN };
    let pages = m.len().div_ceil(unsafe { getpagesize() } as usize);
    let mut v = vec![0 as std::ffi::c_char; pages];
    if pages == 0 || unsafe { mincore(m.as_ptr().cast(), m.len(), v.as_mut_ptr()) } != 0 {
        return f64::NAN;
    }
    (v.iter().filter(|b| **b & 1 != 0).count() as f64 / pages as f64 * 100.0).round()
}

#[cfg(not(unix))]
fn cached_pct(_: &std::path::Path) -> f64 {
    f64::NAN
}

const PAIRS: [(&str, &str); 8] = [
    ("Tom lives in Berlin.", "Tom moved to Lisbon in March."),
    ("Maya works as a nurse at St. Mary's Hospital.", "Maya started a new job as a data analyst at Stripe."),
    ("Lena is a vegetarian.", "Lena started eating meat again this year."),
    ("The project deadline is June 30.", "The project deadline was pushed to August 15."),
    ("The user's landlord is called Mr. Petrov and the flat is on the fourth floor.", "The user is learning to bake sourdough bread at weekends."),
    ("Ravi drives a Honda Civic.", "Ravi's car is a Honda Civic."),
    ("The staging cluster has 32 GB of RAM and runs Debian 12 on three nodes.", "The staging cluster was upgraded to 64 GB of RAM last night."),
    ("Ana's rent is €1,250 per month.", "From next month Ana pays €1,320 a month in rent."),
];

fn req(i: usize) -> DecideRequest {
    let (a, b) = PAIRS[i % PAIRS.len()];
    DecideRequest {
        state: format!("Premise: {a}\nHypothesis: {b}"),
        question: "What is the relationship between the premise and the hypothesis?".into(),
        options: vec![
            "entailment: The hypothesis follows from the premise".into(),
            "neutral: The hypothesis might be true or false given the premise".into(),
            "contradiction: The hypothesis contradicts the premise".into(),
        ],
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let device = DevicePref::parse(args.get(1).map(String::as_str).unwrap_or("cpu"))?;
    let precision = Precision::parse(args.get(2).map(String::as_str).unwrap_or("f32"))?;
    let model = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| DECIDER_NANO_REPO.to_string());
    let m0 = mem();
    let cached = resolve_hf_file(&model, "model.safetensors")
        .map(|p| cached_pct(&p))
        .unwrap_or(f64::NAN);
    let m = DeciderNano::with_repo(
        &model,
        EngineConfig {
            device,
            precision,
            ..Default::default()
        },
    )?;
    let t = Instant::now();
    m.decide(&[req(0)], false)?;
    let load_ms = t.elapsed().as_secs_f64() * 1e3;
    let m_load = mem();
    let (dev, bytes) = m.loaded_info().unwrap();
    let reqs16: Vec<_> = (0..16).map(req).collect();
    m.decide(&reqs16, false)?;
    let m16 = mem();
    let lat = |n: usize, reps: usize| -> Result<Vec<f64>> {
        let reqs: Vec<_> = (0..n).map(req).collect();
        m.decide(&reqs, false)?;
        let mut v = vec![];
        for r in 0..reps {
            let reqs: Vec<_> = (r..r + n).map(req).collect();
            let t = Instant::now();
            m.decide(&reqs, false)?;
            v.push(t.elapsed().as_secs_f64() * 1e3);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(v)
    };
    let b1 = lat(1, 40)?;
    let b16 = lat(16, 10)?;
    let b64 = lat(64, 5)?;
    let m_end = mem();
    let p = |v: &[f64], q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    let out = serde_json::json!({
        "device": dev, "precision": precision.name(), "model": model,
        "weights_mb": (bytes as f64 / 1048576.0).round(),
        "load_ms": load_ms.round(), "weights_cached_pct": cached,
        "footprint_mb": {"start": m0.0, "loaded": m_load.0, "load_peak": m_load.1, "after_b16": m16.0, "end": m_end.0, "peak": m_end.1},
        "rss_mb": {"start": m0.2, "loaded": m_load.2, "after_b16": m16.2, "end": m_end.2},
        "b1_p50_ms": (p(&b1, 0.5) * 10.0).round() / 10.0, "b1_p90_ms": (p(&b1, 0.9) * 10.0).round() / 10.0,
        "b16_ms_per_decision": (p(&b16, 0.5) / 16.0 * 10.0).round() / 10.0,
        "b64_ms_per_decision": (p(&b64, 0.5) / 64.0 * 10.0).round() / 10.0,
    });
    println!("{out}");
    Ok(())
}
