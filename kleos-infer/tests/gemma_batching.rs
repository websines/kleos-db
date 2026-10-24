/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Rows batched across concurrent callers embed exactly as one at a time.
//! Ignored by default: needs `google/embeddinggemma-300m` in the HF cache.
//! `KLEOS_TEST_DEVICE=metal` runs it on the GPU (looser tolerance).

use std::sync::{Arc, Barrier};

use kleos_infer::{DevicePref, EngineConfig, GemmaModel, Precision, TextEmbedder, GEMMA_DOCUMENT_PROMPT};

#[test]
#[ignore = "needs google/embeddinggemma-300m in the HF cache"]
fn concurrent_batching_matches_one_at_a_time() {
    let metal = std::env::var("KLEOS_TEST_DEVICE").map(|d| d == "metal").unwrap_or(false);
    let engine = EngineConfig {
        device: if metal { DevicePref::Metal } else { DevicePref::Cpu },
        precision: if metal { Precision::F16 } else { Precision::F32 },
        ..Default::default()
    };
    let model = GemmaModel::new("google/embeddinggemma-300m", engine).unwrap();
    let e = Arc::new(model.embedder(GEMMA_DOCUMENT_PROMPT, None).unwrap());
    let texts: Vec<String> = (0..24)
        .map(|i| format!("Memory {i}: the harbour master moved the night desk to extension {} {}", 4400 + i, "and the cargo volume rose ".repeat(i % 5)))
        .collect();
    let single: Vec<Vec<f32>> = texts.iter().map(|t| e.embed(&[t.as_str()]).unwrap().remove(0)).collect();
    let bar = Arc::new(Barrier::new(texts.len()));
    let hs: Vec<_> = texts
        .iter()
        .cloned()
        .map(|t| {
            let (e, bar) = (e.clone(), bar.clone());
            std::thread::spawn(move || {
                bar.wait();
                e.embed(&[t.as_str()]).unwrap().remove(0)
            })
        })
        .collect();
    let batched: Vec<Vec<f32>> = hs.into_iter().map(|h| h.join().unwrap()).collect();
    let mut max = 0f32;
    for (a, b) in single.iter().zip(&batched) {
        for (x, y) in a.iter().zip(b) {
            max = max.max((x - y).abs());
        }
    }
    let tol = if metal { 2e-3 } else { 1e-5 };
    eprintln!("max |batched - single| = {max:e} (tolerance {tol:e})");
    assert!(max <= tol, "max diff {max}");
}
