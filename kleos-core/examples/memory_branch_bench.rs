/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Memory branches: fork cost against the parent's size, and recall cost
//! on a plain namespace against branches of it (docs/specs/memory-branches.md,
//! "Costs"). Model-free: a hashed bag-of-words embedder (64 dims) and no
//! decider, so the numbers are the engine's, not a model's.
//!
//! ```text
//! cargo run --release -p kleos --example memory_branch_bench -- [sizes, default 1000,10000] [sqlite dir]
//! ```

use std::sync::Arc;
use std::time::Instant;

use kleos::embed::{self, Embedder};
use kleos::memory::{ForkOpts, InitOpts, RecallOpts, RememberOpts};
use kleos::DbInstance;
use serde_json::json;

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

const NAMES: [&str; 12] = ["Anna", "Ben", "Chloe", "Dev", "Ema", "Finn", "Gita", "Hugo", "Ines", "Jon", "Kai", "Lea"];
const THINGS: [&str; 10] = ["car", "phone", "flat", "bike", "laptop", "garden", "office", "boat", "piano", "camera"];
const COLORS: [&str; 8] = ["red", "blue", "green", "black", "white", "grey", "yellow", "orange"];

fn fact(i: usize) -> String {
    format!(
        "{}'s {} number {} is {} and was bought in {}.",
        NAMES[i % NAMES.len()],
        THINGS[(i / 12) % THINGS.len()],
        i,
        COLORS[(i / 7) % COLORS.len()],
        1990 + i % 35
    )
}

fn p50(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn recall_p50(db: &DbInstance, ns: &str, queries: &[String]) -> f64 {
    let opts = |ns: &str| RecallOpts {
        ns: Some(ns.to_string()),
        k: Some(10),
        record_access: Some(false),
        ..Default::default()
    };
    for q in queries.iter().take(5) {
        db.recall(q, opts(ns)).unwrap();
    }
    p50(queries
        .iter()
        .map(|q| {
            let t = Instant::now();
            db.recall(q, opts(ns)).unwrap();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect())
}

fn main() -> miette::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let sizes: Vec<usize> = args
        .get(1)
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![1000, 10000]);
    let dir = args.get(2).cloned();
    embed::register_embedder("bench_words", Arc::new(Words))?;
    let mut out = vec![];
    for &n in &sizes {
        let (engine, path) = match &dir {
            Some(d) => {
                let p = format!("{d}/branch-bench-{n}.db");
                let _ = std::fs::remove_file(&p);
                ("sqlite", p)
            }
            None => ("mem", String::new()),
        };
        let db = DbInstance::new(engine, &path, "")?;
        db.memory_init(InitOpts { embedder: Some("bench_words".into()), ..Default::default() })?;
        let t = Instant::now();
        for i in 0..n {
            db.remember(&fact(i), RememberOpts { ns: Some("main".into()), split: Some("none".into()), ..Default::default() })?;
        }
        let load_s = t.elapsed().as_secs_f64();
        // fork cost: 20 forks of a namespace of n memories
        let mut fork_ms = vec![];
        for j in 0..20 {
            let t = Instant::now();
            db.memory_fork(ForkOpts { from: Some("main".into()), to: format!("f{j}"), at: None })?;
            fork_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        // a working branch: 5% own memories, 1% inherited ones changed (copied up)
        db.memory_fork(ForkOpts { from: Some("main".into()), to: "work".into(), at: None })?;
        for i in 0..n / 20 {
            db.remember(&format!("{} also owns a kite number {i}.", NAMES[i % NAMES.len()]), RememberOpts {
                ns: Some("work".into()),
                split: Some("none".into()),
                ..Default::default()
            })?;
        }
        let ids: Vec<String> = db
            .recall("car phone flat bike", RecallOpts { ns: Some("main".into()), k: (n / 100).max(1), diversity: Some(false), record_access: Some(false), ..Default::default() })?
            .into_iter()
            .map(|h| h.id)
            .collect();
        db.memory_set(kleos::memory::SetOpts { ns: Some("work".into()), ids: ids.clone(), importance: Some(0.8), ..Default::default() })?;
        db.memory_fork(ForkOpts { from: Some("work".into()), to: "deep1".into(), at: None })?;
        db.memory_fork(ForkOpts { from: Some("deep1".into()), to: "deep2".into(), at: None })?;
        let queries: Vec<String> = (0..60)
            .map(|i| format!("What colour is {}'s {}?", NAMES[(i * 5) % NAMES.len()], THINGS[(i * 3) % THINGS.len()]))
            .collect();
        let r = json!({
            "memories": n,
            "engine": engine,
            "load_s": (load_s * 10.0).round() / 10.0,
            "fork_ms_p50": (p50(fork_ms.clone()) * 1000.0).round() / 1000.0,
            "fork_ms_max": (fork_ms.iter().cloned().fold(0.0, f64::max) * 1000.0).round() / 1000.0,
            "recall_ms_p50": {
                "plain": (recall_p50(&db, "main", &queries) * 100.0).round() / 100.0,
                "fork_depth1": (recall_p50(&db, "f0", &queries) * 100.0).round() / 100.0,
                "work_depth1_5pct_own": (recall_p50(&db, "work", &queries) * 100.0).round() / 100.0,
                "depth3": (recall_p50(&db, "deep2", &queries) * 100.0).round() / 100.0,
            },
            "copied_up": ids.len(),
        });
        println!("{r}");
        out.push(r);
        drop(db);
        if let Some(_) = &dir {
            let _ = std::fs::remove_file(&path);
        }
    }
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
    Ok(())
}
