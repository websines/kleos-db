/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Evaluate every decide() template × backend (docs/specs/decide.md).
//!
//! Calibration (threshold, τ_auto, relation bias) and probes are fitted on
//! the 60 adapt items only and reported on the 42 + 24 eval items
//! (examples/data/decide_items.json). AUROC with stratified-bootstrap 95%
//! CIs, accuracy at the adapt threshold, auto-band coverage / false autos,
//! latency per decision.
//!
//! ```text
//! cargo +1.99.0 run --release -p kleos --features infer --example decide_eval -- \
//!     [--nano] [--device cpu|metal] [--precision f32|q8] \
//!     [--d1-fixture .bench-data/decide/select/d1_record.json] \
//!     [--vitaminc .bench-data/decide/vitaminc_sample.json] [--fit-defaults] [--out out.json]
//! ```
//!
//! Backends: `nano` runs tasksource-decider-nano through kleos-infer
//! (feature `infer`); `d1` replays recorded d1-omni answers (a fixture written
//! by scripts/infer-bench/decide/d1_record.py) until the Rust d1 port lands.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use kleos::decide::{
    self, accuracy_at, auroc, fit_calibration, fit_probe, fit_tau_auto, score_items, Calibration, DecideAnswer,
    DecideRequest, Decider, Family, ProbeOptions, Template,
};
use serde_json::{json, Value};

type Item = (String, String, String);

fn load_items(path: &str) -> (Vec<Item>, Vec<Item>, Vec<Item>) {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).expect(path)).unwrap();
    let get = |k: &str| -> Vec<Item> {
        v[k].as_array()
            .unwrap()
            .iter()
            .map(|r| {
                (
                    r[0].as_str().unwrap().to_string(),
                    r[1].as_str().unwrap().to_string(),
                    r[2].as_str().unwrap().to_string(),
                )
            })
            .collect()
    };
    (get("adapt"), get("eval"), get("entity"))
}

/// Replays recorded answers keyed on `state ␞ question`.
struct Recorded {
    id: String,
    answers: HashMap<String, (Vec<f32>, Vec<Vec<f32>>)>,
}

impl Decider for Recorded {
    fn model_id(&self) -> String {
        self.id.clone()
    }
    fn max_batch(&self) -> usize {
        64
    }
    fn decide(&self, reqs: &[DecideRequest], with_features: bool) -> miette::Result<Vec<DecideAnswer>> {
        reqs.iter()
            .map(|r| {
                let k = format!("{}\u{241e}{}", r.state, r.question);
                let (l, f) = self
                    .answers
                    .get(&k)
                    .ok_or_else(|| miette::miette!("no recorded answer for {k:?}"))?;
                if with_features && f.is_empty() {
                    miette::bail!("no recorded features for {k:?}");
                }
                Ok(DecideAnswer {
                    logits: l.clone(),
                    features: if with_features { f.clone() } else { vec![] },
                })
            })
            .collect()
    }
}

fn load_recorded(path: &str) -> (Recorded, Value) {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).expect(path)).unwrap();
    let mut answers = HashMap::new();
    for (k, a) in v["answers"].as_object().unwrap() {
        let logits = a["logits"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let feats = a
            .get("features")
            .and_then(|f| f.as_array())
            .map(|rows| {
                rows.iter()
                    .map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect())
                    .collect()
            })
            .unwrap_or_default();
        answers.insert(k.clone(), (logits, feats));
    }
    let id = v["model"].as_str().unwrap_or("d1").to_string();
    let vit = v.get("vitaminc300_summary").cloned().unwrap_or(Value::Null);
    (Recorded { id, answers }, vit)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Stratified bootstrap 95% CI of `stat(scores, labels)`.
fn boot(scores: &[f64], labels: &[bool], stat: &dyn Fn(&[f64], &[bool]) -> f64) -> [f64; 2] {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let pos: Vec<usize> = (0..labels.len()).filter(|i| labels[*i]).collect();
    let neg: Vec<usize> = (0..labels.len()).filter(|i| !labels[*i]).collect();
    let mut vals = vec![];
    for _ in 0..2000 {
        let mut s = vec![];
        let mut l = vec![];
        for g in [&pos, &neg] {
            for _ in 0..g.len() {
                let i = g[rng.below(g.len())];
                s.push(scores[i]);
                l.push(labels[i]);
            }
        }
        vals.push(stat(&s, &l));
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    [vals[(vals.len() as f64 * 0.025) as usize], vals[(vals.len() as f64 * 0.975) as usize]]
}

fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// Gold of a template on a memory item (labels update/contradiction/compatible/duplicate).
fn gold(t: Template, label: &str) -> usize {
    match t {
        Template::Supersedes | Template::Contradicts => matches!(label, "update" | "contradiction") as usize,
        Template::Duplicate => (label == "duplicate") as usize,
        Template::SameEntity => (label == "same") as usize,
        Template::Relation => decide::RELATION_LABELS.iter().position(|l| *l == label).unwrap(),
        Template::Important => 0,
    }
}

fn pairs(items: &[Item]) -> Vec<(&str, &str)> {
    items.iter().map(|(a, b, _)| (a.as_str(), b.as_str())).collect()
}

/// Evaluate one binary template; returns a JSON report and the fitted calibration.
fn eval_binary(d: &dyn Decider, f: Family, t: Template, adapt: &[Item], eval: &[Item], with_probe: bool) -> (Value, Option<Calibration>) {
    let raw = Calibration::default();
    let ye: Vec<bool> = eval.iter().map(|x| gold(t, &x.2) == 1).collect();
    let se = score_items(d, f, t, &pairs(eval), &raw, with_probe).unwrap();
    let pe: Vec<f64> = se.iter().map(|s| s.probs[1]).collect();
    let mut rep = json!({
        "auroc": r3(auroc(&pe, &ye)),
        "auroc_ci95": boot(&pe, &ye, &|s, l| auroc(s, l)).map(r3),
    });
    if t == Template::Supersedes {
        let lab: Vec<&str> = eval.iter().map(|x| x.2.as_str()).collect();
        for neg in ["compatible", "duplicate"] {
            let (s, l): (Vec<f64>, Vec<bool>) = pe
                .iter()
                .zip(&ye)
                .zip(&lab)
                .filter(|((_, y), l)| **y || **l == neg)
                .map(|((p, y), _)| (*p, *y))
                .unzip();
            rep[format!("auroc_vs_{neg}")] = json!(r3(auroc(&s, &l)));
        }
    }
    let mut cal_out = None;
    if !adapt.is_empty() {
        let ya: Vec<usize> = adapt.iter().map(|x| gold(t, &x.2)).collect();
        let sa = score_items(d, f, t, &pairs(adapt), &raw, with_probe).unwrap();
        let probs: Vec<Vec<f64>> = sa.iter().map(|s| s.probs.clone()).collect();
        let cal = fit_calibration(t, f, &probs, None, &ya, 0.02, false);
        let ybool: Vec<bool> = ya.iter().map(|y| *y == 1).collect();
        let pa: Vec<f64> = probs.iter().map(|p| p[1]).collect();
        let thr = cal.threshold;
        rep["adapt_auroc"] = json!(r3(auroc(&pa, &ybool)));
        rep["threshold"] = json!(r3(thr));
        rep["acc_at_threshold"] = json!(r3(accuracy_at(&pe, &ye, thr)));
        rep["acc_ci95"] = json!(boot(&pe, &ye, &|s, l| accuracy_at(s, l, thr)).map(r3));
        rep["tau_auto"] = json!(r3(cal.tau_auto));
        let auto: Vec<bool> = pe.iter().map(|p| *p >= cal.tau_auto).collect();
        rep["auto_coverage"] = json!(r3(auto.iter().filter(|a| **a).count() as f64 / pe.len() as f64));
        rep["auto_false"] = json!(auto.iter().zip(&ye).filter(|(a, y)| **a && !**y).count());
        if with_probe {
            let mut xs = vec![];
            let mut ys = vec![];
            for (s, y) in sa.iter().zip(&ya) {
                for x in &s.features {
                    xs.push(x.clone());
                    ys.push(*y);
                }
            }
            let probe = fit_probe(t, f, &d.model_id(), &xs, &ys, &ProbeOptions::default()).unwrap();
            let pp: Vec<f64> = se
                .iter()
                .map(|s| s.features.iter().map(|x| probe.predict(x).unwrap()[1]).sum::<f64>() / s.features.len() as f64)
                .collect();
            let auto: Vec<bool> = pp.iter().map(|p| *p >= probe.tau_auto).collect();
            rep["probe"] = json!({
                "C": probe.c, "adapt_cv_auroc": r3(probe.cv_score),
                "auroc": r3(auroc(&pp, &ye)), "auroc_ci95": boot(&pp, &ye, &|s, l| auroc(s, l)).map(r3),
                "threshold": r3(probe.threshold), "acc_at_threshold": r3(accuracy_at(&pp, &ye, probe.threshold)),
                "tau_auto": r3(probe.tau_auto),
                "auto_coverage": r3(auto.iter().filter(|a| **a).count() as f64 / pp.len() as f64),
                "auto_false": auto.iter().zip(&ye).filter(|(a, y)| **a && !**y).count(),
            });
        }
        cal_out = Some(cal);
    } else {
        rep["acc_at_0.5"] = json!(r3(accuracy_at(&pe, &ye, 0.5)));
    }
    (rep, cal_out)
}

fn argmax(p: &[f64]) -> usize {
    p.iter().enumerate().fold((0, f64::NEG_INFINITY), |a, (i, v)| if *v > a.1 { (i, *v) } else { a }).0
}

fn eval_relation(d: &dyn Decider, f: Family, adapt: &[Item], eval: &[Item], with_probe: bool) -> (Value, Calibration) {
    let t = Template::Relation;
    let raw = Calibration::default();
    let ya: Vec<usize> = adapt.iter().map(|x| gold(t, &x.2)).collect();
    let ye: Vec<usize> = eval.iter().map(|x| gold(t, &x.2)).collect();
    let sa = score_items(d, f, t, &pairs(adapt), &raw, with_probe).unwrap();
    let se = score_items(d, f, t, &pairs(eval), &raw, with_probe).unwrap();
    // 3-way: update and contradiction merged (the zero-shot readout has no contradiction)
    let m3 = |l: usize| if l == 1 { 0 } else { l };
    let acc3 = |pred: &[usize], g: &[usize]| pred.iter().zip(g).filter(|(p, g)| m3(**p) == m3(**g)).count() as f64 / g.len() as f64;
    let probs_a: Vec<Vec<f64>> = sa.iter().map(|s| s.probs.clone()).collect();
    let ops_a: Vec<Vec<f64>> = sa.iter().map(|s| s.option_probs.clone()).collect();
    let cal = fit_calibration(t, f, &probs_a, Some(&ops_a), &ya, 0.02, false);
    let spec = t.spec(f);
    let corrected = |op: &[f64]| -> Vec<f64> {
        let z: Vec<f64> = op.iter().enumerate().map(|(i, p)| (p + 1e-9).ln() + cal.bias.get(i).copied().unwrap_or(0.0)).collect();
        let q = decide::softmax(&z);
        let mut out = vec![0.0; 4];
        for (i, x) in q.into_iter().enumerate() {
            out[spec.label_of_option[i]] += x;
        }
        out
    };
    let pred0: Vec<usize> = se.iter().map(|s| argmax(&s.probs)).collect();
    let pred1: Vec<usize> = se.iter().map(|s| argmax(&corrected(&s.option_probs))).collect();
    let mut rep = json!({
        "acc3": r3(acc3(&pred0, &ye)),
        "bias_from_adapt": cal.bias,
        "acc3_with_bias": r3(acc3(&pred1, &ye)),
    });
    if with_probe {
        let mut xs = vec![];
        let mut ys = vec![];
        for (s, y) in sa.iter().zip(&ya) {
            xs.push(s.features[0].clone());
            ys.push(*y);
        }
        let probe = fit_probe(t, f, &d.model_id(), &xs, &ys, &ProbeOptions::default()).unwrap();
        let pp: Vec<usize> = se.iter().map(|s| argmax(&probe.predict(&s.features[0]).unwrap())).collect();
        let acc4 = pp.iter().zip(&ye).filter(|(p, g)| p == g).count() as f64 / ye.len() as f64;
        rep["probe"] = json!({"C": probe.c, "adapt_cv_acc": r3(probe.cv_score), "acc4": r3(acc4), "acc3": r3(acc3(&pp, &ye))});
    }
    (rep, cal)
}

fn latency(d: &dyn Decider, f: Family, eval: &[Item]) -> Value {
    let ps = pairs(eval);
    let cal = Calibration::default();
    score_items(d, f, Template::Supersedes, &ps[..4], &cal, false).unwrap(); // warm-up / load
    let mut lat = vec![];
    for p in &ps {
        let t = Instant::now();
        score_items(d, f, Template::Supersedes, std::slice::from_ref(p), &cal, false).unwrap();
        lat.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    lat.sort_by(|a, b| a.total_cmp(b));
    let t = Instant::now();
    score_items(d, f, Template::Supersedes, &ps, &cal, false).unwrap();
    let batched = t.elapsed().as_secs_f64() * 1000.0 / ps.len() as f64;
    json!({"b1_p50_ms": r3(lat[lat.len() / 2]), "b1_p90_ms": r3(lat[lat.len() * 9 / 10]), "batched_ms_per_decision": r3(batched)})
}

fn rss_mb() -> f64 {
    let pid = std::process::id().to_string();
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|kb| (kb / 1024.0).round())
        .unwrap_or(f64::NAN)
}

fn eval_backend(name: &str, d: &dyn Decider, f: Family, items: &(Vec<Item>, Vec<Item>, Vec<Item>), vit: Option<&Value>, real: bool) -> (Value, Vec<(Template, Calibration)>) {
    let (adapt, eval, entity) = items;
    let mut rep = json!({"backend": name, "model": d.model_id(), "family": f.name()});
    let mut cals = vec![];
    for t in [Template::Supersedes, Template::Contradicts, Template::Duplicate] {
        // the d1 fixture records features for the forward order only
        let (r, c) = eval_binary(d, f, t, adapt, eval, real || !t.symmetric());
        eprintln!("[{name}] {}: {r}", t.name());
        rep[t.name()] = r;
        cals.push((t, c.unwrap()));
    }
    let (r, _) = eval_binary(d, f, Template::SameEntity, &[], entity, false);
    eprintln!("[{name}] same_entity: {r}");
    rep["same_entity"] = r;
    let (r, c) = eval_relation(d, f, adapt, eval, true);
    eprintln!("[{name}] relation: {r}");
    rep["relation"] = r;
    cals.push((Template::Relation, c));
    if let Some(v) = vit {
        let te: Vec<Item> = v["test300"].as_array().unwrap().iter().map(|r| (r[0].as_str().unwrap().into(), r[1].as_str().unwrap().into(), r[2].as_str().unwrap().into())).collect();
        let tr: Vec<Item> = v["train100"].as_array().unwrap().iter().map(|r| (r[0].as_str().unwrap().into(), r[1].as_str().unwrap().into(), r[2].as_str().unwrap().into())).collect();
        let raw = Calibration::default();
        let se = score_items(d, f, Template::Supersedes, &pairs(&te), &raw, true).unwrap();
        let ye: Vec<bool> = te.iter().map(|x| x.2 == "REFUTES").collect();
        let pe: Vec<f64> = se.iter().map(|s| s.probs[1]).collect();
        let sa = score_items(d, f, Template::Supersedes, &pairs(&tr), &raw, true).unwrap();
        let ya: Vec<usize> = tr.iter().map(|x| (x.2 == "REFUTES") as usize).collect();
        let xs: Vec<Vec<f32>> = sa.iter().map(|s| s.features[0].clone()).collect();
        let probe = fit_probe(Template::Supersedes, f, &d.model_id(), &xs, &ya, &ProbeOptions::default()).unwrap();
        let pp: Vec<f64> = se.iter().map(|s| probe.predict(&s.features[0]).unwrap()[1]).collect();
        let r = json!({
            "zero_shot_auroc": r3(auroc(&pe, &ye)), "zero_shot_ci95": boot(&pe, &ye, &|s, l| auroc(s, l)).map(r3),
            "probe_train100_auroc": r3(auroc(&pp, &ye)), "probe_ci95": boot(&pp, &ye, &|s, l| auroc(s, l)).map(r3),
        });
        eprintln!("[{name}] vitaminc300: {r}");
        rep["vitaminc300_supersedes"] = r;
    }
    if real {
        rep["latency_supersedes"] = latency(d, f, eval);
        rep["rss_mb"] = json!(rss_mb());
        eprintln!("[{name}] latency: {} rss {} MB", rep["latency_supersedes"], rep["rss_mb"]);
    }
    let _ = fit_tau_auto; // re-exported helper, used through fit_calibration
    (rep, cals)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let flag = |k: &str| args.iter().any(|a| a == k);
    let items_path = arg("--items").unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data/decide_items.json").to_string());
    let items = load_items(&items_path);
    let vit: Option<Value> = arg("--vitaminc").map(|p| serde_json::from_str(&std::fs::read_to_string(&p).expect(&p)).unwrap());
    let mut reports = vec![];
    let mut fitted = vec![];
    if let Some(p) = arg("--d1-fixture") {
        let (rec, vsum) = load_recorded(&p);
        let rec: Arc<dyn Decider> = Arc::new(rec);
        let (mut r, c) = eval_backend("d1 (recorded, MPS f16)", &*rec, Family::D1, &items, None, false);
        r["vitaminc300_supersedes_python"] = vsum;
        reports.push(r);
        fitted.push((Family::D1, c));
    }
    #[cfg(feature = "infer")]
    if flag("--nano") {
        use kleos::decide::builtin::{DevicePref, EngineConfig, Precision};
        let mut eng = EngineConfig { precision: Precision::F32, ..EngineConfig::default() };
        if let Some(p) = arg("--precision") {
            eng.precision = Precision::parse(&p).unwrap();
        }
        if let Some(dv) = arg("--device") {
            eng.device = DevicePref::parse(&dv).unwrap();
        }
        let label = format!("decider-nano ({}, {})", arg("--device").unwrap_or("auto".into()), eng.precision.name());
        let nano = kleos_infer::DeciderNano::new(eng).unwrap();
        let rss0 = rss_mb();
        struct W(kleos_infer::DeciderNano);
        impl Decider for W {
            fn model_id(&self) -> String {
                kleos_infer::Decider::model_id(&self.0).to_string()
            }
            fn max_batch(&self) -> usize {
                kleos_infer::Decider::max_batch(&self.0)
            }
            fn decide(&self, reqs: &[DecideRequest], wf: bool) -> miette::Result<Vec<DecideAnswer>> {
                let r: Vec<_> = reqs.iter().map(|r| kleos_infer::DecideRequest { state: r.state.clone(), question: r.question.clone(), options: r.options.clone() }).collect();
                Ok(kleos_infer::Decider::decide(&self.0, &r, wf)
                    .map_err(|e| miette::miette!("{e}"))?
                    .into_iter()
                    .map(|a| DecideAnswer { logits: a.logits, features: a.features })
                    .collect())
            }
        }
        let w = W(nano);
        let (mut r, c) = eval_backend(&label, &w, Family::Nano, &items, vit.as_ref(), true);
        r["rss_before_load_mb"] = json!(rss0);
        reports.push(r);
        fitted.push((Family::Nano, c));
    }
    #[cfg(not(feature = "infer"))]
    if flag("--nano") {
        eprintln!("--nano needs --features infer");
    }
    if flag("--fit-defaults") {
        println!("// defaults fitted on the 60 adapt items (alpha = 0.02, empirical)");
        for (f, cals) in &fitted {
            for (t, c) in cals {
                println!(
                    "(Template::{:?}, Family::{:?}) => cal(&{:?}, {:.3}, {:.3}, ADAPT),",
                    t, f, c.bias, c.threshold, c.tau_auto
                );
            }
        }
    }
    let out = json!(reports);
    match arg("--out") {
        Some(p) => std::fs::write(&p, serde_json::to_string_pretty(&out).unwrap()).unwrap(),
        None => println!("{}", serde_json::to_string_pretty(&out).unwrap()),
    }
}
