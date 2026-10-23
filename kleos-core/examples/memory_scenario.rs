/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Runs a scripted "user life" stream (scripts/memory-api/life_stream.py)
//! through the memory API with the real built-in models (decider-nano +
//! an embedder) and reports: recall accuracy on questions about the current
//! (and past) facts, right/wrong auto-supersedes against the gold labels,
//! duplicates, and remember/recall latency.
//!
//! ```text
//! cargo +1.99.0 run --release -p kleos --features infer-metal --example memory_scenario -- \
//!     <life_stream.json> <report.json> [embeddinggemma|potion-base-8M]
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use kleos::decide::{self, DeciderSpec};
use kleos::embed::{self, EmbedderSpec};
use kleos::memory::{InitOpts, RecallOpts, RememberOpts};
use kleos::DbInstance;
use serde_json::{json, Value};

const DAY: i64 = 86_400_000_000;
const T0: i64 = 1_767_225_600_000_000; // 2026-01-01T00:00:00Z

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn main() -> miette::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let data: Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).expect("read data")).expect("json");
    let out_path = args.get(2).cloned().unwrap_or_else(|| "life_stream_report.json".into());
    let model = args.get(3).cloned().unwrap_or_else(|| "embeddinggemma".into());

    let t = Instant::now();
    embed::enable_embedder("life", EmbedderSpec::model(model.as_str()))?;
    decide::enable_decider("nano", DeciderSpec::model("decider-nano"))?;
    // load both models before timing anything
    embed::embed_texts("life", &["warm up"])?;
    decide::decide_one("nano", decide::Template::Supersedes, "warm", "up")?;
    let load_s = t.elapsed().as_secs_f64();

    let db = DbInstance::new("mem", "", "")?;
    db.memory_init(InitOpts {
        embedder: Some("life".into()),
        decider: Some("nano".into()),
    })?;

    let turns = data["turns"].as_array().unwrap();
    let mut id_turn: BTreeMap<String, usize> = BTreeMap::new();
    let mut rem_ms = vec![];
    let mut sup_right = 0;
    let mut sup_wrong = vec![];
    let mut sup_missed = vec![];
    let mut dup_right = 0;
    let mut dup_wrong = vec![];
    let mut dup_missed = vec![];
    let mut tentative = 0;
    let mut log = vec![];
    for (i, tn) in turns.iter().enumerate() {
        let text = tn["text"].as_str().unwrap();
        let day = tn["day"].as_i64().unwrap();
        let t = Instant::now();
        let r = db.remember(
            text,
            RememberOpts {
                ns: Some("user".into()),
                at: Some(T0 + day * DAY),
                ..Default::default()
            },
        )?;
        rem_ms.push(t.elapsed().as_secs_f64() * 1e3);
        if let Some(id) = &r.id {
            id_turn.insert(id.clone(), i);
        }
        let gold_sup: BTreeSet<usize> = tn["supersedes"]
            .as_array()
            .map(|a| a.iter().map(|x| x.as_u64().unwrap() as usize).collect())
            .unwrap_or_default();
        let gold_dup = tn["duplicate_of"].as_u64().map(|x| x as usize);
        let got_sup: BTreeSet<usize> = r.superseded.iter().filter_map(|id| id_turn.get(id).copied()).collect();
        for j in &got_sup {
            if gold_sup.contains(j) {
                sup_right += 1;
            } else {
                sup_wrong.push(json!({"turn": i, "text": text, "superseded": turns[*j]["text"]}));
            }
        }
        for j in gold_sup.difference(&got_sup) {
            // a gold target that is itself no longer current (e.g. deduped) is not a miss
            let j_id = id_turn.iter().find(|(_, t)| **t == *j).map(|(k, _)| k.clone());
            sup_missed.push(json!({
                "turn": i, "text": text, "should_supersede": turns[*j]["text"],
                "stored": j_id.is_some(),
                "decisions": r.decisions.iter().filter(|d| Some(&d.candidate) == j_id.as_ref()).collect::<Vec<_>>(),
            }));
        }
        match (r.duplicate_of.as_ref().and_then(|id| id_turn.get(id)), gold_dup) {
            (Some(j), Some(g)) if *j == g => dup_right += 1,
            (Some(j), _) => dup_wrong.push(json!({"turn": i, "text": text, "duplicate_of": turns[*j]["text"]})),
            (None, Some(g)) => dup_missed.push(json!({"turn": i, "text": text, "of": turns[g]["text"],
                "tentative": r.tentative.iter().any(|d| id_turn.get(&d.candidate) == Some(&g))})),
            (None, None) => {}
        }
        tentative += r.tentative.len();
        log.push(json!({"turn": i, "text": text, "result": r}));
    }

    let qs = data["questions"].as_array().unwrap();
    let mut policies = vec![];
    let mut rec_ms = vec![];
    for hide in [false, true] {
    let mut correct = 0;
    let mut stale_top = 0;
    let mut stale_in_k = 0;
    let mut answers = vec![];
    for q in qs {
        let query = q["query"].as_str().unwrap();
        let as_of = q["as_of_day"].as_i64().map(|d| T0 + d * DAY);
        let t = Instant::now();
        let hits = db.recall(
            query,
            RecallOpts {
                ns: Some("user".into()),
                k: Some(3),
                as_of,
                hide_pending: hide,
                ..Default::default()
            },
        )?;
        if !hide {
            // the second pass hits the query-embedding cache
            rec_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let top = hits.first().map(|h| h.text.clone()).unwrap_or_default();
        let ok = q["expect"].as_str().unwrap().split('|').any(|e| top.contains(e));
        let stale: Vec<&str> = q["stale"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect();
        let is_stale = |t: &str| stale.iter().any(|s| t.contains(s));
        if ok {
            correct += 1;
        }
        if is_stale(&top) {
            stale_top += 1;
        }
        if hits.iter().any(|h| is_stale(&h.text)) {
            stale_in_k += 1;
        }
        answers.push(json!({"query": query, "as_of_day": q["as_of_day"], "ok": ok, "top": top,
            "hits": hits.iter().map(|h| json!([h.text, h.score, h.pending_superseded_by.len()])).collect::<Vec<_>>()}));
    }
    policies.push(json!({"hide_pending": hide, "questions": qs.len(), "correct_top1": correct,
        "accuracy": correct as f64 / qs.len() as f64, "stale_top1": stale_top, "stale_in_top3": stale_in_k,
        "answers": answers}));
    }

    let gold_sups: usize = turns.iter().map(|t| t["supersedes"].as_array().map_or(0, |a| a.len())).sum();
    let gold_dups = turns.iter().filter(|t| t["duplicate_of"].is_u64()).count();
    let report = json!({
        "embedder": model, "decider": "decider-nano (nano-tune-v1, q8)", "turns": turns.len(),
        "model_load_s": load_s,
        "recall": policies,
        "supersede": {"gold": gold_sups, "auto_right": sup_right, "auto_wrong": sup_wrong.len(),
                      "missed": sup_missed.len(), "wrong": sup_wrong, "missed_detail": sup_missed},
        "duplicate": {"gold": gold_dups, "right": dup_right, "wrong": dup_wrong.len(), "missed": dup_missed.len(),
                      "wrong_detail": dup_wrong, "missed_detail": dup_missed},
        "tentative_decisions": tentative,
        "latency_ms": {"remember_p50": pct(&mut rem_ms.clone(), 0.5), "remember_p90": pct(&mut rem_ms.clone(), 0.9),
                       "recall_p50": pct(&mut rec_ms.clone(), 0.5), "recall_p90": pct(&mut rec_ms.clone(), 0.9)},
        "log": log,
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&report).unwrap()).expect("write report");
    for p in report["recall"].as_array().unwrap() {
        println!(
            "recall (hide_pending {}): {}/{} top-1 correct, stale top-1 {}, stale in top-3 {}",
            p["hide_pending"], p["correct_top1"], p["questions"], p["stale_top1"], p["stale_in_top3"]
        );
    }
    println!(
        "auto-supersede right {sup_right} / wrong {} / missed {} of {gold_sups}; \
         duplicates right {dup_right} / wrong {} / missed {} of {gold_dups}; tentative {tentative}; \
         remember p50 {:.1} ms, recall p50 {:.1} ms; load {load_s:.1} s\nreport: {out_path}",
        sup_wrong.len(),
        sup_missed.len(),
        dup_wrong.len(),
        dup_missed.len(),
        pct(&mut rem_ms, 0.5),
        pct(&mut rec_ms, 0.5),
    );
    Ok(())
}
