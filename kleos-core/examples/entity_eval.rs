/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Entity-resolution evaluation (docs/specs/entity-resolution.md).
//!
//! * `--memory`: the memory-style mention set (examples/data/entity_mentions.jsonl),
//!   streamed into an empty relation with commit; adapt split for fitting,
//!   test split for reporting.
//! * `--dblp <dir>`: Leipzig DBLP-ACM (CC BY 4.0; not in the repo). ACM is
//!   the entity table, DBLP records are the mentions (dry run). A seeded 30%
//!   of the DBLP records is the adapt split; the rest is reported.
//!
//! Variants: `baseline` (exact normalized name, same type), `rules` (kleos
//! without ML: rules + LSH + FTS), `nano` (decider-nano, shipped
//! same_entity calibration), `nano+fit` (threshold and τ_auto fitted on the
//! adapt split with risk control). With an embedder (potion-base-8M) the
//! vector source joins the ML variants.
//!
//! ```text
//! cargo +1.99.0 run --release -p kleos --features infer-metal --example entity_eval -- \
//!     --memory --dblp .bench-data/entity/dblp-acm
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use kleos::decide::{self, fit_tau_auto, fit_threshold};
use kleos::entity::{entity_embed_text, normalize_name, EntityConfig, Mention, Outcome, Resolution, ResolveOptions};
use kleos::{DataValue, DbInstance, ScriptMutability};
use serde_json::Value;

fn run(db: &DbInstance, s: &str, params: BTreeMap<String, DataValue>) -> kleos::NamedRows {
    db.run_script(s, params, ScriptMutability::Mutable)
        .unwrap_or_else(|e| panic!("{s}: {e:?}"))
}

#[derive(Clone, Copy, PartialEq)]
enum Variant {
    Baseline,
    Rules,
    Nano,
    NanoFit,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Variant::Baseline => "baseline (exact name)",
            Variant::Rules => "kleos rules+LSH+FTS (no ML)",
            Variant::Nano => "kleos + decider-nano (shipped cal.)",
            Variant::NanoFit => "kleos + decider-nano (fitted on adapt)",
        }
    }
}

#[derive(Default, Debug, Clone)]
struct Report {
    n: usize,
    cand_recall: Option<(usize, usize)>,
    topm_recall: Option<(usize, usize)>,
    tp: usize,
    fp: usize,
    fn_: usize,
    auto: usize,
    auto_false: usize,
    secs: f64,
    model_pairs: usize,
}

impl Report {
    fn line(&self, name: &str) -> String {
        let p = self.tp as f64 / (self.tp + self.fp).max(1) as f64;
        let r = self.tp as f64 / (self.tp + self.fn_).max(1) as f64;
        let f1 = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
        let rec = |x: Option<(usize, usize)>| match x {
            Some((a, b)) => format!("{:.3} ({a}/{b})", a as f64 / b.max(1) as f64),
            None => "-".to_string(),
        };
        format!(
            "| {name} | {} | {} | {p:.3} | {r:.3} | {f1:.3} | {:.2}% ({}/{}) | {:.0} mentions/s | {} |",
            rec(self.cand_recall),
            rec(self.topm_recall),
            100.0 * self.auto_false as f64 / self.auto.max(1) as f64,
            self.auto_false,
            self.auto,
            self.n as f64 / self.secs.max(1e-9),
            self.model_pairs
        )
    }
}

const HEADER: &str = "| variant | candidate recall | top-M recall | P | R | F1 | Auto false merges | throughput | pairs to model |\n|---|---|---|---|---|---|---|---|---|";

// ------------------------------------------------------------------ memory set

struct MemItem {
    split: String,
    gold: String,
    m: Mention,
}

fn load_memory() -> Vec<MemItem> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data/entity_mentions.jsonl");
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Value = serde_json::from_str(l).unwrap();
            let s = |k: &str| v[k].as_str().unwrap().to_string();
            MemItem {
                split: s("split"),
                gold: s("gold"),
                m: Mention::new(s("name")).with_type(s("type")).with_context(s("context")),
            }
        })
        .collect()
}

fn memory_db(embedder: Option<&str>, dim: usize) -> DbInstance {
    let db = DbInstance::new("mem", "", "").unwrap();
    run(
        &db,
        &format!(
            ":create ent {{id: String => name: String, type: String? default null, aliases: [String]? default null, description: String? default null, emb: <F32; {dim}>? default null}}"
        ),
        Default::default(),
    );
    run(&db, "::lsh create ent:lsh {extractor: name, tokenizer: NGram(3, 3, false), filters: [Lowercase], n_perm: 64, target_threshold: 0.3, n_gram: 1}", Default::default());
    run(&db, "::fts create ent:fts {extractor: name, tokenizer: Simple, filters: [Lowercase]}", Default::default());
    if embedder.is_some() {
        run(&db, &format!("::hnsw create ent:vec {{dim: {dim}, m: 16, dtype: F32, fields: [emb], distance: Cosine, ef_construction: 64}}"), Default::default());
    }
    db
}

fn memory_cfg(embedder: Option<&str>, decider: Option<&str>) -> EntityConfig {
    let mut c = EntityConfig::new("ent");
    c.type_col = Some("type".into());
    c.aliases = Some("aliases".into());
    c.description = Some("description".into());
    c.lsh = Some("lsh".into());
    c.fts = Some("fts".into());
    if let Some(e) = embedder {
        c.embedding = Some("emb".into());
        c.hnsw = Some("vec".into());
        c.embedder = Some(e.into());
    }
    c.decider = decider.map(|d| d.to_string());
    c
}

/// Stream one split; returns the report and the decided (p, same?) pairs.
fn run_memory(
    items: &[&MemItem],
    v: Variant,
    embedder: Option<&str>,
    dim: usize,
    decider: Option<&str>,
    fit: Option<(f64, f64)>,
) -> (Report, Vec<(f64, bool)>) {
    let mut rep = Report {
        n: items.len(),
        ..Default::default()
    };
    // predicted cluster per mention index
    let mut cluster: Vec<usize> = (0..items.len()).collect();
    let mut pairs = vec![];
    let t = Instant::now();
    if v == Variant::Baseline {
        let mut seen: BTreeMap<(String, String), usize> = BTreeMap::new();
        for (i, it) in items.iter().enumerate() {
            let key = (normalize_name(&it.m.name), it.m.entity_type.clone().unwrap_or_default());
            if let Some(&j) = seen.get(&key) {
                cluster[i] = cluster[j];
                rep.auto += 1;
                if items[j].gold != it.gold {
                    rep.auto_false += 1;
                }
            } else {
                seen.insert(key, i);
            }
        }
    } else {
        let db = memory_db(embedder, dim);
        let mut cfg = memory_cfg(embedder, if v == Variant::Rules { None } else { decider });
        if v == Variant::Rules {
            cfg.embedder = None;
            cfg.hnsw = None;
        }
        if let Some((thr, tau)) = fit {
            cfg.threshold = Some(thr);
            cfg.tau_auto = Some(tau);
        }
        let ms: Vec<Mention> = items
            .iter()
            .enumerate()
            .map(|(i, it)| it.m.clone().with_id(format!("m{i}").as_str()))
            .collect();
        let res = db.resolve_entities(&cfg, &ms, ResolveOptions { commit: true }).unwrap();
        let idx = |d: &DataValue| -> usize { d.get_str().unwrap()[1..].parse().unwrap() };
        for (i, r) in res.iter().enumerate() {
            let has_prior = items[..i].iter().any(|x| x.gold == items[i].gold);
            if has_prior {
                let e = rep.cand_recall.get_or_insert((0, 0));
                e.1 += 1;
                if r.candidates.iter().any(|c| items[idx(&c.entity)].gold == items[i].gold) {
                    e.0 += 1;
                }
                let e = rep.topm_recall.get_or_insert((0, 0));
                e.1 += 1;
                if r.candidates.iter().take(cfg.max_candidates).any(|c| items[idx(&c.entity)].gold == items[i].gold) {
                    e.0 += 1;
                }
            }
            for c in &r.candidates {
                if let (Some(p), Some(by)) = (c.p, &c.decided_by) {
                    if by.contains("same_entity") {
                        rep.model_pairs += 1;
                        pairs.push((p, items[idx(&c.entity)].gold == items[i].gold));
                    }
                }
            }
            if r.outcome == Outcome::Matched {
                rep.auto += 1;
                let target = idx(r.entity.as_ref().unwrap());
                if items[target].gold != items[i].gold {
                    rep.auto_false += 1;
                    eprintln!("  false merge: {:?} -> {:?} ({}, p {:.3})", items[i].m.name, items[target].m.name, r.decided_by, r.p);
                }
            }
        }
        for i in 0..items.len() {
            let c = db.canonical_entity("ent", &DataValue::from(format!("m{i}").as_str())).unwrap();
            cluster[i] = idx(&c);
        }
    }
    rep.secs = t.elapsed().as_secs_f64();
    for i in 0..items.len() {
        for j in i + 1..items.len() {
            let same_gold = items[i].gold == items[j].gold;
            let same_pred = cluster[i] == cluster[j];
            match (same_pred, same_gold) {
                (true, true) => rep.tp += 1,
                (true, false) => rep.fp += 1,
                (false, true) => rep.fn_ += 1,
                _ => {}
            }
        }
    }
    (rep, pairs)
}

// ------------------------------------------------------------------ DBLP-ACM

struct Rec {
    id: String,
    title: String,
    authors: String,
    venue: String,
    year: String,
}

fn load_csv(path: &str) -> Vec<Rec> {
    let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("{path}"));
    let text = String::from_utf8_lossy(&bytes).to_string();
    let mut rdr = csv::Reader::from_reader(text.as_bytes());
    rdr.records()
        .map(|r| {
            let r = r.unwrap();
            Rec {
                id: r[0].to_string(),
                title: r[1].trim().to_string(),
                authors: r[2].to_string(),
                venue: r[3].to_string(),
                year: r[4].to_string(),
            }
        })
        .collect()
}

fn desc(r: &Rec) -> String {
    format!("{}; {} {}", r.authors, r.venue, r.year)
}

fn dblp_db(acm: &[Rec], embedder: Option<&str>, dim: usize) -> DbInstance {
    let db = DbInstance::new("mem", "", "").unwrap();
    run(
        &db,
        &format!(":create paper {{id: String => name: String, description: String? default null, emb: <F32; {dim}>? default null}}"),
        Default::default(),
    );
    let rows: Vec<DataValue> = acm
        .iter()
        .map(|r| DataValue::List(vec![r.id.as_str().into(), r.title.as_str().into(), desc(r).as_str().into()]))
        .collect();
    run(
        &db,
        "?[id, name, description] <- $rows :put paper {id => name, description}",
        [("rows".to_string(), DataValue::List(rows))].into_iter().collect(),
    );
    if let Some(e) = embedder {
        let texts: Vec<String> = acm.iter().map(|r| entity_embed_text(&r.title, None, Some(&desc(r)))).collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let vecs = kleos::embed::embed_texts(e, &refs).unwrap();
        let rows: Vec<DataValue> = acm
            .iter()
            .zip(vecs)
            .map(|(r, v)| {
                DataValue::List(vec![
                    r.id.as_str().into(),
                    DataValue::Vec(kleos::Vector::F32(ndarray::Array1::from(v.to_vec()))),
                ])
            })
            .collect();
        run(&db, "?[id, emb] <- $rows :update paper {id => emb}", [("rows".to_string(), DataValue::List(rows))].into_iter().collect());
        run(&db, &format!("::hnsw create paper:vec {{dim: {dim}, m: 16, dtype: F32, fields: [emb], distance: Cosine, ef_construction: 64}}"), Default::default());
    }
    run(&db, "::lsh create paper:lsh {extractor: name, tokenizer: NGram(3, 3, false), filters: [Lowercase], n_perm: 64, target_threshold: 0.4, n_gram: 1}", Default::default());
    run(&db, "::fts create paper:fts {extractor: name, tokenizer: Simple, filters: [Lowercase]}", Default::default());
    db
}

#[allow(clippy::too_many_arguments)]
fn run_dblp(
    db: &DbInstance,
    acm: &[Rec],
    dblp: &[&Rec],
    gold: &BTreeMap<String, String>,
    v: Variant,
    embedder: Option<&str>,
    decider: Option<&str>,
    fit: Option<(f64, f64)>,
) -> (Report, Vec<(f64, bool)>) {
    let mut rep = Report {
        n: dblp.len(),
        ..Default::default()
    };
    let n_gold = dblp.iter().filter(|r| gold.contains_key(&r.id)).count();
    let mut pairs = vec![];
    let t = Instant::now();
    let mut predicted: Vec<(String, String)> = vec![];
    if v == Variant::Baseline {
        let mut by: BTreeMap<String, Vec<&str>> = BTreeMap::new();
        for r in acm {
            by.entry(normalize_name(&r.title)).or_default().push(&r.id);
        }
        for r in dblp {
            if let Some(ids) = by.get(&normalize_name(&r.title)) {
                if ids.len() == 1 {
                    predicted.push((r.id.clone(), ids[0].to_string()));
                }
            }
        }
    } else {
        let mut cfg = EntityConfig::new("paper");
        cfg.description = Some("description".into());
        cfg.lsh = Some("lsh".into());
        cfg.fts = Some("fts".into());
        cfg.max_candidates = 3;
        if v != Variant::Rules {
            cfg.decider = decider.map(|d| d.to_string());
            if let Some(e) = embedder {
                cfg.embedding = Some("emb".into());
                cfg.hnsw = Some("vec".into());
                cfg.embedder = Some(e.into());
            }
        }
        if let Some((thr, tau)) = fit {
            cfg.threshold = Some(thr);
            cfg.tau_auto = Some(tau);
        }
        let ms: Vec<Mention> = dblp.iter().map(|r| Mention::new(r.title.as_str()).with_description(desc(r))).collect();
        let res: Vec<Resolution> = db.resolve_entities(&cfg, &ms, ResolveOptions::default()).unwrap();
        let mut cr = (0, 0);
        let mut tr = (0, 0);
        for (r, m) in res.iter().zip(dblp) {
            let g = gold.get(&m.id);
            if let Some(g) = g {
                cr.1 += 1;
                tr.1 += 1;
                if r.candidates.iter().any(|c| c.entity.get_str() == Some(g)) {
                    cr.0 += 1;
                }
                if r.candidates.iter().take(cfg.max_candidates).any(|c| c.entity.get_str() == Some(g)) {
                    tr.0 += 1;
                }
            }
            for c in &r.candidates {
                if let (Some(p), Some(by)) = (c.p, &c.decided_by) {
                    if by.contains("same_entity") {
                        rep.model_pairs += 1;
                        pairs.push((p, g.map(|g| c.entity.get_str() == Some(g)).unwrap_or(false)));
                    }
                }
            }
            if r.outcome == Outcome::Matched {
                predicted.push((m.id.clone(), r.entity.as_ref().unwrap().get_str().unwrap().to_string()));
            }
        }
        rep.cand_recall = Some(cr);
        rep.topm_recall = Some(tr);
    }
    rep.secs = t.elapsed().as_secs_f64();
    for (d, a) in &predicted {
        rep.auto += 1;
        if gold.get(d) == Some(a) {
            rep.tp += 1;
        } else {
            rep.fp += 1;
            rep.auto_false += 1;
        }
    }
    rep.fn_ = n_gold - rep.tp;
    (rep, pairs)
}

fn fit_from(pairs: &[(f64, bool)], alpha: f64) -> (f64, f64, String) {
    let s: Vec<f64> = pairs.iter().map(|x| x.0).collect();
    let l: Vec<bool> = pairs.iter().map(|x| x.1).collect();
    if s.len() < 4 || l.iter().all(|x| *x) || l.iter().all(|x| !*x) {
        return (0.5, 0.9, "too few adapt pairs: shipped defaults".into());
    }
    let thr = fit_threshold(&s, &l);
    let tf = fit_tau_auto(&s, &l, alpha, false);
    let note = format!(
        "fitted on {} adapt pairs ({} positive): threshold {thr:.3}, tau_auto {:.3} (α {alpha}, adapt coverage {:.1}%, false {:.1}%), AUROC {:.3}",
        s.len(),
        l.iter().filter(|x| **x).count(),
        tf.tau,
        100.0 * tf.coverage,
        100.0 * tf.false_rate,
        decide::auroc(&s, &l)
    );
    // the label threshold never sits above the auto threshold
    (thr.min(tf.tau), tf.tau, note)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let flag = |k: &str| args.iter().any(|a| a == k);
    decide::set_decision_cache_capacity(0);
    let no_ml = flag("--no-ml");
    let decider = if no_ml {
        None
    } else {
        match kleos::decide::enable_decider("nano", kleos::DeciderSpec::model("decider-nano")) {
            Ok(()) => Some("nano"),
            Err(e) => {
                eprintln!("decider unavailable ({e}); ML variants skipped");
                None
            }
        }
    };
    let (embedder, dim) = if no_ml || flag("--no-vector") {
        (None, 4)
    } else {
        match kleos::embed::enable_embedder("potion", kleos::EmbedderSpec::model("potion-base-8M")) {
            Ok(()) => {
                let d = kleos::embed::embed_texts("potion", &["probe"]).map(|v| v[0].len()).unwrap_or(256);
                (Some("potion"), d)
            }
            Err(e) => {
                eprintln!("embedder unavailable ({e}); vector source skipped");
                (None, 4)
            }
        }
    };
    let mut variants = vec![Variant::Baseline, Variant::Rules];
    if decider.is_some() {
        variants.push(Variant::Nano);
        variants.push(Variant::NanoFit);
    }

    if flag("--memory") {
        let all = load_memory();
        let adapt: Vec<&MemItem> = all.iter().filter(|x| x.split == "adapt").collect();
        let test: Vec<&MemItem> = all.iter().filter(|x| x.split == "test").collect();
        println!("\n## Memory-style mentions: {} adapt / {} test mentions, {} / {} entities\n",
            adapt.len(), test.len(),
            adapt.iter().map(|x| &x.gold).collect::<BTreeSet<_>>().len(),
            test.iter().map(|x| &x.gold).collect::<BTreeSet<_>>().len());
        let alpha = arg("--alpha").map(|a| a.parse().unwrap()).unwrap_or(0.02);
        let mut fit = None;
        if decider.is_some() {
            let (rep, pairs) = run_memory(&adapt, Variant::Nano, embedder, dim, decider, None);
            let (thr, tau, note) = fit_from(&pairs, alpha);
            println!("adapt (zero-shot, for the fit): {}", rep.line("nano on adapt"));
            println!("fit: {note}\n");
            fit = Some((thr, tau));
        }
        println!("{HEADER}");
        for v in &variants {
            let f = if *v == Variant::NanoFit { fit } else { None };
            let (rep, pairs) = run_memory(&test, *v, embedder, dim, decider, f);
            println!("{}", rep.line(v.name()));
            if *v == Variant::Nano && !pairs.is_empty() {
                let s: Vec<f64> = pairs.iter().map(|x| x.0).collect();
                let l: Vec<bool> = pairs.iter().map(|x| x.1).collect();
                eprintln!("  test same_entity AUROC over decided pairs: {:.3} ({} pairs)", decide::auroc(&s, &l), s.len());
            }
        }
    }

    if let Some(dir) = arg("--dblp") {
        let acm = load_csv(&format!("{dir}/ACM.csv"));
        let dblp_all = load_csv(&format!("{dir}/DBLP2.csv"));
        let mut gold = BTreeMap::new();
        let text = String::from_utf8_lossy(&std::fs::read(format!("{dir}/DBLP-ACM_perfectMapping.csv")).unwrap()).to_string();
        let mut rdr = csv::Reader::from_reader(text.as_bytes());
        for r in rdr.records() {
            let r = r.unwrap();
            gold.insert(r[0].to_string(), r[1].to_string());
        }
        // seeded split by a hash of the id: 30% adapt
        let h = |s: &str| s.bytes().fold(1469598103934665603u64, |a, b| (a ^ b as u64).wrapping_mul(1099511628211));
        let limit: usize = arg("--limit").map(|x| x.parse().unwrap()).unwrap_or(usize::MAX);
        let adapt: Vec<&Rec> = dblp_all.iter().filter(|r| h(&r.id) % 10 < 3).take(limit).collect();
        let test: Vec<&Rec> = dblp_all.iter().filter(|r| h(&r.id) % 10 >= 3).take(limit).collect();
        println!(
            "\n## DBLP-ACM: {} ACM entities; DBLP mentions {} adapt / {} test ({} / {} with a gold match)\n",
            acm.len(),
            adapt.len(),
            test.len(),
            adapt.iter().filter(|r| gold.contains_key(&r.id)).count(),
            test.iter().filter(|r| gold.contains_key(&r.id)).count()
        );
        let t = Instant::now();
        let db = dblp_db(&acm, embedder, dim);
        eprintln!("loaded and indexed ACM in {:.1}s", t.elapsed().as_secs_f64());
        let alpha = arg("--alpha").map(|a| a.parse().unwrap()).unwrap_or(0.01);
        let mut fit = None;
        if decider.is_some() {
            let (rep, pairs) = run_dblp(&db, &acm, &adapt, &gold, Variant::Nano, embedder, decider, None);
            let (thr, tau, note) = fit_from(&pairs, alpha);
            println!("adapt (shipped calibration, for the fit): {}", rep.line("nano on adapt"));
            println!("fit: {note}\n");
            fit = Some((thr, tau));
        }
        println!("{HEADER}");
        for v in &variants {
            let f = if *v == Variant::NanoFit { fit } else { None };
            let (rep, pairs) = run_dblp(&db, &acm, &test, &gold, *v, embedder, decider, f);
            println!("{}", rep.line(v.name()));
            if *v == Variant::Nano && !pairs.is_empty() {
                let s: Vec<f64> = pairs.iter().map(|x| x.0).collect();
                let l: Vec<bool> = pairs.iter().map(|x| x.1).collect();
                eprintln!("  test same_entity AUROC over decided pairs: {:.3} ({} pairs)", decide::auroc(&s, &l), s.len());
            }
        }
    }
}
