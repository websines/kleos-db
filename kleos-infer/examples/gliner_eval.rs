/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! GLiNER predictions for an evaluation set, in the prediction format of
//! `scripts/gliner/ft_eval.py`, and a speed probe.
//!
//! ```text
//! gliner_eval predict <model> <in.json> <out.json> <f32|f16|q8> <cpu|metal> [labels]
//! gliner_eval speed   <model> <in.json> <f32|f16|q8> <cpu|metal> [n]
//! ```
//!
//! `in.json` is a list of `{id, text}` records, optionally with
//! `claim_start`/`claim_end` (character offsets): the text is then a whole
//! message, read in 384-word windows, and only spans inside the claim are kept
//! (offsets relative to the claim).

use std::time::Instant;

use kleos_infer::*;
use serde_json::{json, Value};

const LABELS: [&str; 4] = ["person", "location", "organization", "relative"];

fn engine(p: &str, d: &str) -> EngineConfig {
    EngineConfig {
        device: DevicePref::parse(d).unwrap(),
        precision: Precision::parse(p).unwrap(),
        ..Default::default()
    }
}

fn char_of(text: &str, byte: usize) -> usize {
    text[..byte].chars().count()
}

fn byte_of(text: &str, ch: usize) -> usize {
    text.char_indices().nth(ch).map(|(b, _)| b).unwrap_or(text.len())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let items: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(&a[3]).unwrap()).unwrap();
    let texts: Vec<&str> = items.iter().map(|x| x["text"].as_str().unwrap()).collect();
    match a[1].as_str() {
        "predict" => {
            let g = Gliner::with_repo(&a[2], engine(&a[5], &a[6])).unwrap();
            let t0 = Instant::now();
            let msg = items.iter().any(|x| x.get("claim_start").is_some());
            let res = if msg {
                g.extract_windows(&texts, &LABELS, 0.1, 384, 64).unwrap()
            } else {
                g.extract(&texts, &LABELS, 0.1).unwrap()
            };
            let secs = t0.elapsed().as_secs_f64();
            let mut pred = vec![];
            for (x, spans) in items.iter().zip(&res) {
                let text = x["text"].as_str().unwrap();
                let (cs, ce) = match (x.get("claim_start"), x.get("claim_end")) {
                    (Some(s), Some(e)) => (
                        byte_of(text, s.as_u64().unwrap() as usize),
                        byte_of(text, e.as_u64().unwrap() as usize),
                    ),
                    _ => (0, text.len()),
                };
                let ents: Vec<Value> = spans
                    .iter()
                    .filter(|s| s.start >= cs && s.end <= ce)
                    .map(|s| {
                        json!({"label": LABELS[s.label], "text": s.text,
                               "start": char_of(text, s.start) - char_of(text, cs),
                               "end": char_of(text, s.end) - char_of(text, cs), "p": s.score})
                    })
                    .collect();
                pred.push(json!({"id": x["id"], "entities": ents, "kind": null, "about_speaker": null}));
            }
            let (dev, bytes) = g.loaded_info().unwrap();
            std::fs::write(
                &a[4],
                serde_json::to_string(&json!({"path": a[2], "device": dev, "precision": a[5], "weight_bytes": bytes,
                                               "secs": secs, "n": items.len(), "pred": pred}))
                .unwrap(),
            )
            .unwrap();
            eprintln!("{} items in {secs:.1}s on {dev} {} ({:.1} ms/item, weights {:.0} MB)", items.len(), a[5],
                      1000.0 * secs / items.len() as f64, bytes as f64 / 1e6);
        }
        "speed" => {
            let g = Gliner::with_repo(&a[2], engine(&a[4], &a[5])).unwrap();
            let n: usize = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(64).min(texts.len());
            g.extract(&texts[..8.min(n)], &LABELS, 0.1).unwrap(); // load + warm up
            for bs in [1usize, 8] {
                let t0 = Instant::now();
                for c in texts[..n].chunks(bs) {
                    g.extract(c, &LABELS, 0.1).unwrap();
                }
                let ms = 1000.0 * t0.elapsed().as_secs_f64() / n as f64;
                eprintln!("{} {} batch {bs}: {ms:.1} ms/claim", a[5], a[4]);
            }
            let (dev, bytes) = g.loaded_info().unwrap();
            eprintln!("weights on {dev}: {:.0} MB", bytes as f64 / 1e6);
        }
        _ => panic!("predict | speed"),
    }
}
