/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Parity of [`DeciderNano`] with the repository's own `decider.py` (PyTorch,
//! CPU, f32). The fixture `fixtures/decider_nano_ref.json` was produced by
//! `fixtures/decider_nano_ref.py`. Skips (with a notice) when the model is not
//! in the local HF cache. Run in release:
//! `cargo +1.99.0 test --release -p kleos-infer --test decider_nano_parity -- --nocapture`.
//! `KLEOS_TEST_DEVICE=metal` (with `--features metal`) runs it on the GPU.

use kleos_infer::*;
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/decider_nano_ref.json")).unwrap()
}

fn requests(fx: &Value) -> (Vec<DecideRequest>, Vec<Vec<f32>>) {
    let mut reqs = vec![];
    let mut want = vec![];
    for it in fx["items"].as_array().unwrap() {
        reqs.push(DecideRequest {
            state: it["state"].as_str().unwrap().to_string(),
            question: it["question"].as_str().unwrap().to_string(),
            options: it["options"]
                .as_array()
                .unwrap()
                .iter()
                .map(|o| o.as_str().unwrap().to_string())
                .collect(),
        });
        want.push(
            it["logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect(),
        );
    }
    (reqs, want)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

fn device() -> DevicePref {
    std::env::var("KLEOS_TEST_DEVICE")
        .ok()
        .map(|d| DevicePref::parse(&d).unwrap())
        .unwrap_or(DevicePref::Cpu)
}

#[test]
fn decider_nano_matches_reference() {
    if resolve_hf_file(DECIDER_NANO_REPO, "model.safetensors").is_err() {
        eprintln!("skip: {DECIDER_NANO_REPO} not in the HF cache (`hf download {DECIDER_NANO_REPO}`)");
        return;
    }
    let fx = fixture();
    let (reqs, want) = requests(&fx);
    // f32 must match tightly; quantised and half precisions are reported
    // and held to looser bounds (same answer on clear-cut items).
    // candle has no bf16 matmul on the CPU (f16 weights there, bf16 on
    // Metal), and its Accelerate backend has no f16 matmul either.
    let mut precisions = vec![(Precision::F32, 2e-3f32), (Precision::Q8, 0.25)];
    if device() != DevicePref::Cpu {
        precisions.push((Precision::BF16, 0.5));
    } else if !cfg!(feature = "accelerate") {
        precisions.push((Precision::F16, 0.5));
    }
    for (prec, tol) in precisions {
        let cfg = EngineConfig {
            device: device(),
            precision: prec,
            ..Default::default()
        };
        let m = DeciderNano::new(cfg).unwrap();
        assert_eq!(m.model_id(), DECIDER_NANO_REPO);
        assert!(!m.is_loaded());
        // batched (all items at once) and one at a time must agree
        let batched = m.decide(&reqs, false).unwrap();
        let single: Vec<DecideAnswer> = reqs
            .iter()
            .map(|r| m.decide(std::slice::from_ref(r), false).unwrap().remove(0))
            .collect();
        let mut max_diff = 0f32;
        let mut max_batch_diff = 0f32;
        let mut flips = 0;
        for ((b, s), w) in batched.iter().zip(&single).zip(&want) {
            assert_eq!(b.logits.len(), w.len());
            for ((x, y), z) in b.logits.iter().zip(&s.logits).zip(w) {
                max_diff = max_diff.max((x - z).abs());
                max_batch_diff = max_batch_diff.max((x - y).abs());
            }
            if argmax(&b.logits) != argmax(w) {
                flips += 1;
            }
        }
        eprintln!(
            "{}/{:?}: max |logit - ref| = {max_diff:.2e}, batched vs single = {max_batch_diff:.2e}, argmax flips = {flips}/{}",
            prec.name(),
            device(),
            want.len()
        );
        assert!(max_diff <= tol, "{} logits differ by {max_diff}", prec.name());
        assert!(max_batch_diff <= tol.max(1e-3));
        if prec == Precision::F32 {
            assert_eq!(flips, 0);
            // pooled option features (what the scorer reads)
            for f in fx["features"].as_array().unwrap() {
                let i = f["item"].as_u64().unwrap() as usize;
                let got = m.decide(std::slice::from_ref(&reqs[i]), true).unwrap().remove(0);
                let vecs = f["vectors"].as_array().unwrap();
                assert_eq!(got.features.len(), vecs.len());
                let mut d = 0f32;
                for (g, v) in got.features.iter().zip(vecs) {
                    let v = v.as_array().unwrap();
                    assert_eq!(g.len(), v.len());
                    assert_eq!(g.len(), 768);
                    for (a, b) in g.iter().zip(v) {
                        d = d.max((a - b.as_f64().unwrap() as f32).abs());
                    }
                }
                eprintln!("f32 features item {i}: max |diff| = {d:.2e}");
                assert!(d < 5e-3);
            }
        }
        assert!(m.is_loaded());
        m.unload();
        assert!(!m.is_loaded());
    }
}

#[test]
fn decider_nano_rejects_bad_requests() {
    if resolve_hf_file(DECIDER_NANO_REPO, "config.json").is_err() {
        eprintln!("skip: {DECIDER_NANO_REPO} not in the HF cache");
        return;
    }
    let m = DeciderNano::new(EngineConfig {
        device: DevicePref::Cpu,
        precision: Precision::Q8,
        ..Default::default()
    })
    .unwrap();
    assert!(m.decide(&[], false).unwrap().is_empty());
    let one = DecideRequest {
        state: "x".into(),
        question: "q".into(),
        options: vec!["only".into()],
    };
    assert!(m.decide(&[one], false).is_err());
    let huge = DecideRequest {
        state: "word ".repeat(9000),
        question: "q".into(),
        options: vec!["a".into(), "b".into()],
    };
    let e = m.decide(&[huge], false).unwrap_err().to_string();
    assert!(e.contains("not truncated"), "{e}");
}
