/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! EmbeddingGemma 2 against sentence-transformers 6.1 (transformers 5.19,
//! f32, eager attention). The reference JSON comes from
//! `tests/fixtures/egemma2_ref.py`; point `KLEOS_EGEMMA2_REF` at it. Ignored
//! by default (needs the model in the HF cache and the reference file).
//! `KLEOS_TEST_DEVICE` (cpu|metal) and `KLEOS_TEST_PRECISION` (f32|f16|q8)
//! pick the run; the cosine floor is 0.9999 at f32, 0.999 otherwise.

use kleos_infer::{DevicePref, Egemma2Model, EngineConfig, Precision, TextEmbedder, EGEMMA2_DOCUMENT_PROMPT, EGEMMA2_QUERY_PROMPT, EGEMMA2_REPO};

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    d / (na.sqrt() * nb.sqrt()).max(1e-12)
}

#[test]
#[ignore = "needs google/embeddinggemma-2 and KLEOS_EGEMMA2_REF"]
fn egemma2_matches_sentence_transformers() {
    let path = std::env::var("KLEOS_EGEMMA2_REF").expect("KLEOS_EGEMMA2_REF");
    let r: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let texts: Vec<String> = r["texts"].as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect();
    let prec = std::env::var("KLEOS_TEST_PRECISION").unwrap_or_else(|_| "f32".into());
    let dev = std::env::var("KLEOS_TEST_DEVICE").unwrap_or_else(|_| "cpu".into());
    let mut engine = EngineConfig::default();
    engine.precision = Precision::parse(&prec).unwrap();
    engine.device = DevicePref::parse(&dev).unwrap();
    // the reference does not truncate; compare texts that fit 8192 tokens
    engine.max_seq_len = 8192;
    let model = Egemma2Model::new(EGEMMA2_REPO, engine).unwrap();
    let keep: Vec<usize> = (0..texts.len())
        .filter(|&i| model.token_ids(&format!("{EGEMMA2_QUERY_PROMPT}{}", texts[i])).unwrap().len() < 8192)
        .collect();
    let longest = keep.iter().map(|&i| model.token_ids(&format!("{EGEMMA2_QUERY_PROMPT}{}", texts[i])).unwrap().len()).max().unwrap();
    println!("{} of {} texts fit 8192 tokens (longest kept: {longest})", keep.len(), texts.len());
    // token ids as sentence-transformers fed them
    for (i, want) in r["ids_document_first6"].as_array().unwrap().iter().enumerate() {
        let want: Vec<u32> = want.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let got = model.token_ids(&format!("{EGEMMA2_DOCUMENT_PROMPT}{}", texts[i])).unwrap();
        assert_eq!(got, want, "token ids of text {i}");
    }
    let floor = if prec == "f32" { 0.9999 } else { 0.999 };
    for (kind, prompt) in [("document", EGEMMA2_DOCUMENT_PROMPT), ("query", EGEMMA2_QUERY_PROMPT)] {
        let e = model.embedder(prompt, None).unwrap();
        let refs: Vec<&str> = keep.iter().map(|&i| texts[i].as_str()).collect();
        let t0 = std::time::Instant::now();
        let got = e.embed(&refs).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        let mut worst = (1.0f64, 0usize);
        let mut sum = 0f64;
        for (j, g) in got.iter().enumerate() {
            let i = keep[j];
            let want: Vec<f32> = r[kind][i].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            let c = cos(g, &want);
            sum += c;
            if c < worst.0 {
                worst = (c, i);
            }
        }
        println!("{dev} {prec} {kind}: min cos {:.6} (text {}), mean {:.7}, {:.1}s for {} texts", worst.0, worst.1, sum / got.len() as f64, secs, got.len());
        assert!(worst.0 >= floor, "{kind}: text {} cos {} < {floor}", worst.1, worst.0);
    }
    println!("weights: {:?}", model.loaded_info());
}
