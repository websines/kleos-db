/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Parity of [`Gliner`] with gliner2 2.1.0 (PyTorch, CPU, f32), stage by
//! stage. The fixture `fixtures/gliner_ref.json` was produced by
//! `fixtures/gliner_ref.py` from the checkpoint in `KLEOS_GLINER_TEST_MODEL`
//! (kleos's fine-tune of `fastino/gliner2.5-base-v1`). Ignored by default
//! (needs that checkpoint); run in release:
//! `KLEOS_GLINER_TEST_MODEL=<dir> cargo +1.99.0 test --release -p kleos-infer --test gliner_parity -- --ignored --nocapture`.
//! `KLEOS_TEST_DEVICE=metal` (with `--features metal`) runs it on the GPU.

use kleos_infer::*;
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/gliner_ref.json")).unwrap()
}

fn model() -> Option<String> {
    std::env::var("KLEOS_GLINER_TEST_MODEL").ok().filter(|p| std::path::Path::new(p).is_dir())
}

fn device() -> DevicePref {
    std::env::var("KLEOS_TEST_DEVICE")
        .ok()
        .map(|d| DevicePref::parse(&d).unwrap())
        .unwrap_or(DevicePref::Cpu)
}

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

fn f32s2(v: &Value) -> Vec<Vec<f32>> {
    v.as_array().unwrap().iter().map(f32s).collect()
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
}

fn max_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len(), "row count");
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            assert_eq!(x.len(), y.len(), "column count");
            x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max)
        })
        .fold(0f32, f32::max)
}

/// Char offsets of the fixture (Python) as byte offsets of `text`.
fn byte_of(text: &str, ch: usize) -> usize {
    text.char_indices().nth(ch).map(|(b, _)| b).unwrap_or(text.len())
}

fn engine(p: Precision) -> EngineConfig {
    EngineConfig {
        device: device(),
        precision: p,
        ..Default::default()
    }
}

#[test]
fn log_buckets_match_transformers() {
    let fx = fixture();
    for (row, q) in [("bucket_row0", 0i64), ("bucket_row699", 699)] {
        let want: Vec<i64> = fx[row].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
        let got: Vec<i64> = (0..want.len() as i64).map(|k| log_bucket_position(q - k, 256, 512)).collect();
        assert_eq!(got, want, "{row}");
    }
}

#[test]
fn words_match_gliner2() {
    let fx = fixture();
    for it in fx["items"].as_array().unwrap() {
        let norm = it["normalized"].as_str().unwrap();
        assert_eq!(normalize_text(it["text"].as_str().unwrap()), norm);
        let got = split_words(norm);
        let want = it["words"].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "word count of {norm:?}");
        for (g, w) in got.iter().zip(want) {
            assert_eq!(g.0, w[0].as_str().unwrap());
            assert_eq!(g.1, byte_of(norm, w[1].as_u64().unwrap() as usize));
            assert_eq!(g.2, byte_of(norm, w[2].as_u64().unwrap() as usize));
        }
    }
}

#[test]
#[ignore = "needs the GLiNER checkpoint in KLEOS_GLINER_TEST_MODEL"]
fn gliner_matches_reference_f32() {
    let Some(dir) = model() else {
        eprintln!("skip: set KLEOS_GLINER_TEST_MODEL to the checkpoint directory");
        return;
    };
    let fx = fixture();
    let labels: Vec<String> = fx["labels"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
    let labels: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
    let floor = fx["floor"].as_f64().unwrap() as f32;
    let g = Gliner::with_repo(&dir, engine(Precision::F32)).unwrap();
    let mut worst = [0f32; 8];
    for it in fx["items"].as_array().unwrap() {
        let text = it["text"].as_str().unwrap();
        let norm = it["normalized"].as_str().unwrap();
        let st = g.debug_stages(text, &labels, floor).unwrap();
        let ids: Vec<u32> = usizes(&it["input_ids"]).iter().map(|&x| x as u32).collect();
        assert_eq!(st.ids, ids, "input ids of {text:?}");
        assert_eq!(st.word_pos, usizes(&it["text_positions"]), "word positions of {text:?}");
        assert_eq!(st.marker_pos, usizes(&it["marker_positions"]), "marker positions of {text:?}");
        let h16: Vec<Vec<f32>> = st.hidden.iter().map(|r| r[..16].to_vec()).collect();
        let d_h = max_diff(&h16, &f32s2(&it["hidden16"]));
        let norms: Vec<Vec<f32>> = st.hidden.iter().map(|r| vec![r.iter().map(|x| x * x).sum::<f32>().sqrt()]).collect();
        let want_norm: Vec<Vec<f32>> = f32s(&it["hidden_norm"]).into_iter().map(|x| vec![x]).collect();
        let d_n = max_diff(&norms, &want_norm);
        let b8: Vec<Vec<f32>> = st.boundary.iter().map(|r| r[..8].to_vec()).collect();
        let d_b = max_diff(&b8, &f32s2(&it["boundary8"]));
        let d_s = max_diff(&st.start_logits, &f32s2(&it["start_logits"]))
            .max(max_diff(&st.end_logits, &f32s2(&it["end_logits"])));
        let d_i = max_diff(&st.inside_prefix, &f32s2(&it["inside_prefix"]))
            .max(max_diff(&[st.inside_mean.clone()], &[f32s(&it["inside_mean"])]));
        let pool: Vec<(usize, usize)> = it["pool"].as_array().unwrap().iter().map(|p| {
            let u = usizes(p);
            (u[0], u[1])
        }).collect();
        assert_eq!(st.pool, pool, "candidate pool of {text:?}");
        let d_c = max_diff(&[st.pool_compat.clone()], &[f32s(&it["pool_compat"])]);
        let d_p = max_diff(&st.pair_logits, &f32s2(&it["pair_logits"]));
        let d_z = max_diff(&[st.null_logits.clone()], &[f32s(&it["null_logits"])]);
        for (k, d) in [d_h, d_n, d_b, d_s, d_i, d_c, d_p, d_z].into_iter().enumerate() {
            worst[k] = worst[k].max(d);
        }
        // decoded spans: same label, text, offsets and order; scores within 1e-4
        let mut want = vec![];
        for (li, l) in labels.iter().enumerate() {
            for e in it["entities"][*l].as_array().unwrap() {
                want.push((
                    li,
                    e["text"].as_str().unwrap().to_string(),
                    byte_of(norm, e["start"].as_u64().unwrap() as usize),
                    byte_of(norm, e["end"].as_u64().unwrap() as usize),
                    e["confidence"].as_f64().unwrap() as f32,
                ));
            }
        }
        let got: Vec<_> = st.spans.iter().map(|s| (s.label, s.text.clone(), s.start, s.end, s.score)).collect();
        assert_eq!(got.len(), want.len(), "span count of {text:?}: {got:?} vs {want:?}");
        for (a, b) in got.iter().zip(&want) {
            assert_eq!((a.0, &a.1, a.2, a.3), (b.0, &b.1, b.2, b.3), "span of {text:?}");
            assert!((a.4 - b.4).abs() < 1e-4, "score of {:?}: {} vs {}", a.1, a.4, b.4);
        }
    }
    eprintln!(
        "max |Δ|: hidden16 {:.2e}, hidden norm {:.2e}, boundary {:.2e}, start/end {:.2e}, inside {:.2e}, compat {:.2e}, pair logits {:.2e}, null {:.2e}",
        worst[0], worst[1], worst[2], worst[3], worst[4], worst[5], worst[6], worst[7]
    );
    assert!(worst[0] < 1e-4 && worst[2] < 1e-4, "encoder/boundary drift {worst:?}");
    assert!(worst[3] < 1e-3 && worst[6] < 1e-3, "logit drift {worst:?}");
}

#[test]
#[ignore = "needs the GLiNER checkpoint in KLEOS_GLINER_TEST_MODEL"]
fn gliner_reduced_precisions_keep_spans() {
    let Some(dir) = model() else {
        eprintln!("skip: set KLEOS_GLINER_TEST_MODEL to the checkpoint directory");
        return;
    };
    let fx = fixture();
    let labels: Vec<String> = fx["labels"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
    let labels: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
    let texts: Vec<&str> = fx["items"].as_array().unwrap().iter().map(|i| i["text"].as_str().unwrap()).collect();
    let reference = Gliner::with_repo(&dir, engine(Precision::F32)).unwrap().extract(&texts, &labels, 0.1).unwrap();
    let mut precisions = vec![Precision::Q8];
    if device() != DevicePref::Cpu || !cfg!(feature = "accelerate") {
        precisions.push(Precision::F16);
    }
    for p in precisions {
        let got = Gliner::with_repo(&dir, engine(p)).unwrap().extract(&texts, &labels, 0.1).unwrap();
        let mut dmax = 0f32;
        let mut flips = 0;
        for (a, b) in got.iter().zip(&reference) {
            // spans at or above 0.5 must agree; scores drift a little
            let strong = |v: &Vec<GlinerSpan>| {
                let mut s: Vec<(usize, String)> = v.iter().filter(|s| s.score >= 0.5).map(|s| (s.label, s.text.clone())).collect();
                s.sort();
                s
            };
            if strong(a) != strong(b) {
                flips += 1;
                eprintln!("{}: {:?} vs {:?}", p.name(), strong(a), strong(b));
            }
            for x in a {
                if let Some(y) = b.iter().find(|y| y.label == x.label && y.start == x.start && y.end == x.end) {
                    dmax = dmax.max((x.score - y.score).abs());
                }
            }
        }
        eprintln!("{}: max |Δp| {dmax:.4}, texts with a changed span ≥ 0.5: {flips}", p.name());
        assert_eq!(flips, 0, "{} changed spans", p.name());
    }
}
