/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! kleos-engram spec    <config.json> <tokenizer.json> [--pad-id N]   fingerprint + spec summary (JSON)
//! kleos-engram rows    <config.json> <tokenizer.json> [--pad-id N] --text T [--last]
//!                                                                    rows read for a text (JSON)
//! kleos-engram parity  <tokenizer.json> <ref-dir>...                  compare against parity_ref.py dumps
//! kleos-engram bench   <tokenizer.json> <n-tokens>                    hashing throughput (demo config)

use std::path::Path;
use std::time::Instant;

use kleos_engram::{parity, CompressedTokenizer, EngramConfig, EngramSpec, NgramHashMapping, Semantics, SpecOverrides};
use serde_json::json;

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn load_spec(args: &[String]) -> EngramSpec {
    let ov = SpecOverrides {
        pad_id: flag(args, "--pad-id").map(|s| s.parse().expect("--pad-id")),
        ..Default::default()
    };
    EngramSpec::from_files(Path::new(&args[2]), Path::new(&args[3]), &ov).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2)
    })
}

fn load_ct(path: &str) -> CompressedTokenizer {
    let tok = tokenizers::Tokenizer::from_file(path).expect("tokenizer.json");
    let t = Instant::now();
    let ct = CompressedTokenizer::from_tokenizer(&tok).expect("compress");
    eprintln!("compressed tokenizer: {} -> {} ids in {:.2?}", ct.lookup.len(), ct.num_new_token, t.elapsed());
    ct
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("spec") => {
            let s = load_spec(&args);
            println!("{}", serde_json::to_string_pretty(&s.summary()).unwrap());
        }
        Some("rows") => {
            let s = load_spec(&args);
            let text = flag(&args, "--text").expect("--text");
            let (ids, hits) = s.rows_for_text(&text, false).unwrap();
            let last = args.iter().any(|a| a == "--last");
            let hits: Vec<_> = hits.into_iter().filter(|h| !last || h.pos + 1 == ids.len()).collect();
            println!("{}", json!({"fingerprint": s.fingerprint(), "ids": ids, "rows": hits}));
        }
        Some("parity") => {
            let built = load_ct(&args[2]);
            let mut all = true;
            for dir in &args[3..] {
                let r = parity::check_dir(Path::new(dir), Some(&built)).expect("reference dir");
                println!(
                    "{dir}:\n  compression built from tokenizer.json: {:?} (reference {:?}), {} entries differ -> {}\n  \
                     hashing: {} parameter mismatches, {} (case, layer) pairs, {} indices, {} mismatching -> {}",
                    r.built, r.reference, r.lookup_mismatch,
                    if r.compression_ok() { "PASS" } else { "DIFF" },
                    r.param_mismatch, r.cases, r.indices, r.bad,
                    if r.hashing_ok() { "PASS" } else { "FAIL" }
                );
                all &= r.hashing_ok();
            }
            std::process::exit(if all { 0 } else { 1 })
        }
        Some("bench") => {
            let cfg = EngramConfig {
                engram_vocab_size: vec![129280 * 5, 129280 * 5],
                max_ngram_size: 3,
                n_head_per_ngram: 8,
                layer_ids: vec![1, 15],
                pad_id: 2,
                seed: 0,
                semantics: Semantics::Demo,
            };
            let hm = NgramHashMapping::new(cfg, load_ct(&args[2]));
            let n: usize = args[3].parse().unwrap();
            // Deterministic pseudo-random raw ids (xorshift), no extra deps.
            let mut x = 0x9E3779B97F4A7C15u64;
            let vocab = hm.tokenizer.lookup.len() as u64;
            let ids: Vec<i64> = (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    (x % vocab) as i64
                })
                .collect();
            let mut out = Vec::new();
            let t = Instant::now();
            let mut sink = 0i64;
            for l in &hm.layers {
                hm.hash(l, &ids, &mut out);
                sink ^= out.iter().fold(0, |a, &b| a ^ b);
            }
            let dt = t.elapsed();
            println!(
                "hashed {n} tokens x {} layers in {dt:.2?}: {:.1} ns per token-layer (sink {sink})",
                hm.layers.len(),
                dt.as_nanos() as f64 / (n * hm.layers.len()) as f64
            );
        }
        _ => eprintln!(
            "usage: kleos-engram spec <config.json> <tokenizer.json> [--pad-id N]\n       \
             kleos-engram rows <config.json> <tokenizer.json> [--pad-id N] --text T [--last]\n       \
             kleos-engram parity <tokenizer.json> <ref-dir>...\n       kleos-engram bench <tokenizer.json> <n>"
        ),
    }
}
