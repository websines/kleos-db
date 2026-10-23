/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Model tests run only when the model is in the local HF cache (they print a
//! skip notice otherwise). The EmbeddingGemma test is `#[ignore]`d: it loads a
//! 1.2 GB checkpoint. Run it with `cargo test -p kleos-infer -- --ignored`.

use kleos_infer::*;

#[test]
fn precision_and_device_parse() {
    assert_eq!(Precision::parse("Q8").unwrap(), Precision::Q8);
    assert_eq!(Precision::parse("bf16").unwrap(), Precision::BF16);
    assert!(Precision::parse("q3").is_err());
    assert_eq!(DevicePref::parse("cpu").unwrap(), DevicePref::Cpu);
    assert!(DevicePref::parse("tpu").is_err());
}

#[test]
fn missing_model_is_a_clear_error() {
    // offline, so this test never touches the network (download is tested
    // in hub_download.rs)
    std::env::set_var("HF_HUB_OFFLINE", "1");
    let e = StaticEmbedder::new("nobody/no-such-model-kleos-test").err().unwrap();
    assert!(e.to_string().contains("hf download"), "{e}");
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn model2vec_static_embedder() {
    let Ok(m) = StaticEmbedder::new("minishlab/potion-base-8M") else {
        eprintln!("skip: minishlab/potion-base-8M not in the HF cache");
        return;
    };
    assert_eq!(m.dim(), 256);
    let v = m
        .embed(&["the cat sat on the mat", "a cat is sitting on a mat", "quarterly tax filing deadline", ""])
        .unwrap();
    assert_eq!(v.len(), 4);
    assert!((cos(&v[0], &v[0]) - 1.0).abs() < 1e-5);
    assert!(cos(&v[0], &v[1]) > cos(&v[0], &v[2]));
    assert!(v[3].iter().all(|x| *x == 0.0), "empty text -> zero vector");
}

#[test]
#[ignore]
fn embeddinggemma_q8_cpu() {
    let engine = EngineConfig {
        device: DevicePref::Cpu,
        precision: Precision::Q8,
        ..Default::default()
    };
    let Ok(model) = GemmaModel::new("google/embeddinggemma-300m", engine) else {
        eprintln!("skip: google/embeddinggemma-300m not in the HF cache");
        return;
    };
    let q = model.embedder(GEMMA_QUERY_PROMPT, None).unwrap();
    let d = model.embedder(GEMMA_DOCUMENT_PROMPT, Some(256)).unwrap();
    assert!(!model.is_loaded(), "weights load lazily");
    let qv = q.embed(&["Which planet is known as the red planet?"]).unwrap();
    assert!(model.is_loaded());
    let dv = d
        .embed(&["Mars is called the red planet.", "Venus is the hottest planet."])
        .unwrap();
    assert_eq!(qv[0].len(), 768);
    assert_eq!(dv[0].len(), 256);
    let q256: Vec<f32> = {
        let mut v = qv[0][..256].to_vec();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= n);
        v
    };
    assert!(cos(&q256, &dv[0]) > cos(&q256, &dv[1]));
    // batching must not change results (padding is masked)
    let alone = d.embed(&["Venus is the hottest planet."]).unwrap();
    assert!(cos(&alone[0], &dv[1]) > 0.9999);
    model.unload();
    assert!(!model.is_loaded());
}
