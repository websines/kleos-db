/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Latency of tasksource-decider-nano on short memory-note decisions.
//!
//! `cargo +1.99.0 run --release -p kleos-infer [--features metal] --example decider_bench -- <cpu|metal> <f32|f16|bf16|q8|q4> [batch sizes, e.g. 1,4,16]`

use std::time::Instant;

use kleos_infer::*;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let device = DevicePref::parse(args.get(1).map(String::as_str).unwrap_or("cpu"))?;
    let precision = Precision::parse(args.get(2).map(String::as_str).unwrap_or("f32"))?;
    let m = DeciderNano::new(EngineConfig {
        device,
        precision,
        ..Default::default()
    })?;
    let pairs = [
        ("Tom lives in Berlin.", "Tom moved to Lisbon in March."),
        ("Maya works as a nurse at St. Mary's Hospital.", "Maya started a new job as a data analyst at Stripe."),
        ("Lena is a vegetarian.", "Lena started eating meat again this year."),
        ("The project deadline is June 30.", "The project deadline was pushed to August 15."),
    ];
    let req = |i: usize| {
        let (old, new) = pairs[i % pairs.len()];
        DecideRequest {
            state: old.to_string(),
            question: format!(
                "Classify the relationship between the old note and this new statement:\n{new}"
            ),
            options: vec!["Entailment".into(), "Contradiction".into(), "NotMentioned".into()],
        }
    };
    let t = Instant::now();
    m.decide(&[req(0)], false)?;
    let (dev, bytes) = m.loaded_info().unwrap();
    println!(
        "load+first: {:.0} ms on {dev}, {:.0} MB weights",
        t.elapsed().as_secs_f64() * 1e3,
        bytes as f64 / 1e6
    );
    let time = |n: usize, reps: usize| -> Result<f64> {
        let reqs: Vec<_> = (0..n).map(req).collect();
        m.decide(&reqs, false)?;
        let mut v = vec![];
        for _ in 0..reps {
            let t = Instant::now();
            m.decide(&reqs, false)?;
            v.push(t.elapsed().as_secs_f64() * 1e3);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(v[v.len() / 2])
    };
    if let Some(sizes) = args.get(3) {
        for n in sizes.split(',') {
            let n: usize = n.parse().unwrap();
            let t = time(n, 10)?;
            println!("batch {n}: {t:.1} ms ({:.1} ms/decision)", t / n as f64);
        }
        return Ok(());
    }
    let one = time(1, 30)?;
    let b16 = time(16, 10)?;
    println!(
        "{} {}: single request (3 options) median {one:.1} ms; batch of 16 median {b16:.1} ms ({:.1} ms/decision)",
        dev,
        precision.name(),
        b16 / 16.0
    );
    Ok(())
}
