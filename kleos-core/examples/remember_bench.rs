/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! `remember` latency with model-free stand-ins (a hashed bag-of-words
//! embedder and an always-neutral NLI decider), so the numbers are the
//! engine's: planning, the commit script, the version check.
//!
//! ```text
//! cargo run --release -p kleos --example remember_bench -- [n, default 300] [mem|sqlite] [dir]
//! ```

use std::sync::Arc;
use std::time::Instant;

use kleos::decide::{self, DecideAnswer, DecideRequest, Decider, Family};
use kleos::embed::{self, Embedder};
use kleos::memory::{InitOpts, RememberOpts};
use kleos::DbInstance;

struct Words;

impl Embedder for Words {
    fn embed(&self, texts: &[&str]) -> miette::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0f32; 64];
                for w in t.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2) {
                    let h = w
                        .to_lowercase()
                        .bytes()
                        .fold(1469598103934665603u64, |h, b| (h ^ b as u64).wrapping_mul(1099511628211));
                    v[(h % 64) as usize] += 1.0;
                }
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                v.iter().map(|x| x / n).collect()
            })
            .collect())
    }
    fn dim(&self) -> usize {
        64
    }
}

struct Neutral;

impl Decider for Neutral {
    fn model_id(&self) -> String {
        "bench-neutral".into()
    }
    fn weights_version(&self) -> Option<String> {
        Some("nano-tune-v1".to_string())
    }
    fn decide(&self, reqs: &[DecideRequest], _f: bool) -> miette::Result<Vec<DecideAnswer>> {
        Ok(reqs.iter().map(|_| DecideAnswer { logits: vec![0.0, 6.0, 0.0], features: vec![] }).collect())
    }
}

const NAMES: [&str; 12] = ["Anna", "Ben", "Chloe", "Dev", "Ema", "Finn", "Gita", "Hugo", "Ines", "Jon", "Kai", "Lea"];
const THINGS: [&str; 10] = ["car", "phone", "flat", "bike", "laptop", "garden", "office", "boat", "piano", "camera"];

fn turn(i: usize) -> String {
    let n = NAMES[i % NAMES.len()];
    let t = THINGS[(i / 12) % THINGS.len()];
    format!(
        "{n}: I bought a new {t} last week, it cost {} dollars. My sister thinks the {t} was a bad idea, but I love it.",
        100 + i
    )
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

fn main() -> miette::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(300);
    let engine = args.get(2).cloned().unwrap_or_else(|| "mem".into());
    let path = match engine.as_str() {
        "mem" => String::new(),
        _ => {
            let dir = args.get(3).cloned().unwrap_or_else(|| ".".into());
            let p = format!("{dir}/remember_bench_{}.db", std::process::id());
            let _ = std::fs::remove_file(&p);
            p
        }
    };
    embed::register_embedder("rb_words", Arc::new(Words))?;
    decide::register_decider("rb_neutral", Arc::new(Neutral), Family::Nano)?;
    let db = DbInstance::new(&engine, &path, "")?;
    db.memory_init(InitOpts {
        embedder: Some("rb_words".into()),
        decider: Some("rb_neutral".into()),
        ..Default::default()
    })?;
    let mut lat = vec![];
    let t0 = Instant::now();
    for i in 0..n {
        let t = Instant::now();
        db.remember(&turn(i), RememberOpts { ns: Some("bench".into()), ..Default::default() })?;
        lat.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let total = t0.elapsed().as_secs_f64();
    let (p50, p95) = (pct(&mut lat.clone(), 0.5), pct(&mut lat, 0.95));
    println!("engine={engine} n={n} remember p50 {p50:.2} ms  p95 {p95:.2} ms  total {total:.2} s");
    if !path.is_empty() {
        let _ = std::fs::remove_file(&path);
    }
    Ok(())
}
