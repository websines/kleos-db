/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! decide(): templates, calibration, symmetric scoring, probes, the
//! `::decider` ops, the `decide_*` functions and the `Decide` fixed rule
//! (docs/specs/decide.md), all against mock deciders. The registry is
//! process-wide: every test uses its own decider names and model ids.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use kleos::decide::{
    self, auroc, fit_bias, fit_probe, fit_tau_auto, fit_threshold, label_probs, wilson_upper,
    Calibration, ProbeOptions,
};
use kleos::{
    Band, DataValue, DbInstance, DecideAnswer, DecideRequest, Decider, Family, NamedRows,
    ScriptMutability, Template,
};

type Rule = dyn Fn(&DecideRequest) -> (Vec<f32>, Vec<Vec<f32>>) + Send + Sync;

/// A decider whose logits/features come from a closure; it records batches.
struct Mock {
    id: String,
    max_batch: usize,
    rule: Box<Rule>,
    calls: AtomicUsize,
    batch_sizes: Mutex<Vec<usize>>,
    seen: Mutex<Vec<DecideRequest>>,
    unloads: AtomicUsize,
}

impl Mock {
    fn new(id: &str, max_batch: usize, rule: impl Fn(&DecideRequest) -> (Vec<f32>, Vec<Vec<f32>>) + Send + Sync + 'static) -> Arc<Self> {
        Arc::new(Self {
            id: id.to_string(),
            max_batch,
            rule: Box::new(rule),
            calls: AtomicUsize::new(0),
            batch_sizes: Mutex::new(vec![]),
            seen: Mutex::new(vec![]),
            unloads: AtomicUsize::new(0),
        })
    }
}

impl Decider for Mock {
    fn model_id(&self) -> String {
        self.id.clone()
    }
    fn max_batch(&self) -> usize {
        self.max_batch
    }
    fn decide(&self, reqs: &[DecideRequest], with_features: bool) -> miette::Result<Vec<DecideAnswer>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.batch_sizes.lock().unwrap().push(reqs.len());
        assert!(reqs.len() <= self.max_batch, "batch above max_batch");
        self.seen.lock().unwrap().extend(reqs.iter().cloned());
        Ok(reqs
            .iter()
            .map(|r| {
                let (logits, features) = (self.rule)(r);
                assert_eq!(logits.len(), r.options.len());
                DecideAnswer {
                    logits,
                    features: if with_features { features } else { vec![] },
                }
            })
            .collect())
    }
    fn is_loaded(&self) -> Option<bool> {
        Some(true)
    }
    fn unload(&self) {
        self.unloads.fetch_add(1, Ordering::SeqCst);
    }
}

fn mem() -> DbInstance {
    DbInstance::new("mem", "", "").unwrap()
}

fn run(db: &DbInstance, s: &str) -> miette::Result<NamedRows> {
    db.run_script(s, Default::default(), ScriptMutability::Mutable)
}

/// The error's debug report with whitespace collapsed (miette wraps lines).
fn err(db: &DbInstance, s: &str) -> String {
    let e = format!("{:?}", run(db, s).expect_err(s));
    e.split(|c: char| c.is_whitespace() || c == '│')
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

// ------------------------------------------------------------------ templates

#[test]
fn golden_d1_supersedes() {
    let r = Template::Supersedes.render(Family::D1, "Tom lives in Berlin.", "Tom moved to Lisbon.");
    assert_eq!(r.state, "Tom lives in Berlin.");
    assert_eq!(
        r.question,
        "Classify the relationship between the old note and this new statement:\nTom moved to Lisbon."
    );
    assert_eq!(
        r.options,
        vec![
            "Entailment: The old note implies the new statement is true",
            "Contradiction: The old note and the new statement cannot both be true",
            "NotMentioned: The old note neither implies nor contradicts the new statement",
        ]
    );
    assert_eq!(Template::Supersedes.versioned(Family::D1), "supersedes@v1");
}

#[test]
fn golden_nano_templates() {
    let r = Template::Supersedes.render(Family::Nano, "A", "B");
    assert_eq!(r.state, "Premise: A\nHypothesis: B");
    assert_eq!(r.question, "What is the relationship between the premise and the hypothesis?");
    assert_eq!(
        r.options,
        vec![
            "entailment: The hypothesis follows from the premise",
            "neutral: The hypothesis might be true or false given the premise",
            "contradiction: The hypothesis contradicts the premise",
        ]
    );
    let e = Template::SameEntity.render(Family::Nano, "Paris (city)", "Paris Hilton");
    assert_eq!(e.state, "Mention A: Paris (city)\nMention B: Paris Hilton");
    assert_eq!(e.question, "Are mention A and mention B the same entity?");
    assert_eq!(
        e.options,
        vec![
            "same: both mentions refer to one and the same entity",
            "different: they are different entities that may share a name",
            "unclear: there is not enough information to tell",
        ]
    );
    assert_eq!(Template::SameEntity.versioned(Family::Nano), "same_entity@v2");
    let i = Template::Important.render(Family::D1, "The user is allergic to penicillin.", "ignored");
    assert_eq!(i.state, "The user is allergic to penicillin.");
    assert_eq!(i.question, "How worth remembering long-term is this note?");
    assert!(i.options[0].starts_with("lasting: "));
    // the no-bare-yes/no policy holds for every built-in template
    for t in Template::ALL {
        for f in [Family::D1, Family::Nano] {
            decide::validate_spec(t, &t.spec(f)).unwrap();
        }
    }
    // relation: ContractNLI framing on both families
    let rel = Template::Relation.render(Family::Nano, "old", "new");
    assert_eq!(rel, Template::Relation.render(Family::D1, "old", "new"));
    assert_eq!(Template::Relation.labels(), &["update", "contradiction", "compatible", "duplicate"]);
    // every template renders as many options as its readout maps
    for t in Template::ALL {
        for f in [Family::D1, Family::Nano] {
            let s = t.spec(f);
            assert_eq!(s.options.len(), s.label_of_option.len(), "{t:?} {f:?}");
            assert!(s.label_of_option.iter().all(|l| *l < t.labels().len()));
        }
    }
}

#[test]
fn template_parse_and_shape() {
    assert_eq!(Template::parse("same_entity").unwrap(), Template::SameEntity);
    assert!(format!("{:?}", Template::parse("whatever").unwrap_err()).contains("unknown decide template 'whatever'"));
    assert!(Template::Duplicate.symmetric() && Template::SameEntity.symmetric() && Template::Contradicts.symmetric());
    assert!(!Template::Supersedes.symmetric() && !Template::Relation.symmetric());
    assert_eq!(Template::Important.arity(), 1);
    assert_eq!(Family::parse("decider-nano").unwrap(), Family::Nano);
    assert_eq!(Family::parse("d1-omni").unwrap(), Family::D1);
}

// ------------------------------------------------------------------ calibration math

#[test]
fn label_probs_bias_temperature() {
    let spec = Template::Relation.spec(Family::D1); // Ent->dup, Contra->update, NotM->compat
    let cal = Calibration {
        bias: vec![0.0, 0.0, -0.5],
        temperature: 2.0,
        ..Default::default()
    };
    let p = label_probs(&spec, 4, &[2.0, 4.0, 2.0], &cal).unwrap();
    // z = [1, 2, 0.5]
    let e = [1f64.exp(), 2f64.exp(), 0.5f64.exp()];
    let s: f64 = e.iter().sum();
    assert!((p[0] - e[1] / s).abs() < 1e-12); // update
    assert_eq!(p[1], 0.0); // contradiction: no zero-shot option
    assert!((p[2] - e[2] / s).abs() < 1e-12); // compatible
    assert!((p[3] - e[0] / s).abs() < 1e-12); // duplicate
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(label_probs(&spec, 4, &[1.0, 2.0], &cal).is_err());
}

#[test]
fn bands_and_thresholds() {
    let cal = Calibration {
        threshold: 0.4,
        tau_auto: 0.8,
        ..Default::default()
    };
    assert_eq!(cal.band(0.85), Band::Auto);
    assert_eq!(cal.band(0.8), Band::Auto);
    assert_eq!(cal.band(0.5), Band::Tentative);
    assert_eq!(cal.band(0.39), Band::None);
    let d = decide::finish(Template::Supersedes, Family::D1, "m", vec![0.45, 0.55], &cal, "calibrated");
    assert_eq!((d.label.as_str(), d.band, d.confident), ("yes", Band::Tentative, false));
    assert_eq!(d.decided_by(), "m|supersedes@v1|calibrated");
    let d = decide::finish(Template::Supersedes, Family::D1, "m", vec![0.7, 0.3], &cal, "calibrated");
    assert_eq!((d.label.as_str(), d.band, d.confident), ("no", Band::None, true));
    let d = decide::finish(Template::Relation, Family::D1, "m", vec![0.1, 0.0, 0.05, 0.85], &cal, "x");
    assert_eq!((d.label.as_str(), d.band), ("duplicate", Band::Auto));
    assert!((d.p - 0.85).abs() < 1e-12);

    let s = [0.1, 0.2, 0.35, 0.6, 0.7, 0.9];
    let y = [false, false, true, false, true, true];
    assert_eq!(fit_threshold(&s, &y), 0.35);
    assert!((auroc(&s, &y) - 8.0 / 9.0).abs() < 1e-12);
    // risk control: scan from the top while the false share stays <= alpha
    let tf = fit_tau_auto(&s, &y, 0.0, false);
    assert_eq!(tf.tau, 0.7);
    assert!((tf.coverage - 2.0 / 6.0).abs() < 1e-12);
    let tf = fit_tau_auto(&s, &y, 0.34, false);
    assert_eq!(tf.tau, 0.35); // 1 false in the top 4 = 0.25
    // nothing qualifies: never auto
    assert_eq!(fit_tau_auto(&[0.9, 0.1], &[false, true], 0.02, false).tau, 1.01);
    // Wilson bound: 0 errors in 60 still allows ~6%
    let w = wilson_upper(0, 60);
    assert!((w - 0.0602).abs() < 1e-3, "{w}");
    assert_eq!(fit_tau_auto(&s, &y, 0.02, true).tau, 1.01);
}

#[test]
fn bias_fit_recovers_prior_shift() {
    // option 2 is over-predicted: its probability is inflated on every item
    let map = [3usize, 0, 2]; // Ent->dup, Contra->update, NotM->compat
    let mut probs = vec![];
    let mut gold = vec![];
    for i in 0..30 {
        let g = i % 3;
        let mut p = [0.2, 0.2, 0.6];
        p[g] += 0.3;
        let s: f64 = p.iter().sum();
        probs.push(p.iter().map(|x| x / s).collect::<Vec<_>>());
        gold.push(map[g]);
    }
    let b = fit_bias(&probs, &gold, &map);
    assert_eq!(b[0], 0.0);
    assert!(b[2] < 0.0, "{b:?}");
    let acc = |bias: &[f64]| {
        probs
            .iter()
            .zip(&gold)
            .filter(|(p, g)| {
                let am = (0..3)
                    .max_by(|&i, &j| ((p[i] + 1e-9f64).ln() + bias[i]).total_cmp(&((p[j] + 1e-9f64).ln() + bias[j])))
                    .unwrap();
                map[am] == **g
            })
            .count()
    };
    assert_eq!(acc(&b), 30);
    assert!(acc(&[0.0, 0.0, 0.0]) < 30);
}

// ------------------------------------------------------------------ probe

fn separable(n: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut xs = vec![];
    let mut ys = vec![];
    for i in 0..n {
        let y = i % 2;
        let s = if y == 1 { 1.0 } else { -1.0 };
        let noise = ((i * 7919) % 13) as f32 / 13.0 - 0.5;
        xs.push(vec![s + 0.3 * noise, noise, 0.5, -noise]);
        ys.push(y);
    }
    (xs, ys)
}

#[test]
fn probe_fits_separable_data() {
    let (xs, ys) = separable(40);
    let p = fit_probe(Template::Supersedes, Family::D1, "m", &xs, &ys, &ProbeOptions::default()).unwrap();
    assert_eq!(p.dim, 4);
    assert!(p.cv_score > 0.95, "{}", p.cv_score);
    for (x, y) in xs.iter().zip(&ys) {
        let q = p.predict(x).unwrap();
        assert_eq!((q[1] >= p.threshold) as usize, *y);
    }
    assert!(p.tau_auto <= 1.0 && p.coverage > 0.0);
    // constant column does not blow up standardisation
    assert_eq!(p.scale[2], 1.0);
    assert!(p.predict(&[0.0; 3]).is_err());
    // multi-class (relation)
    let mut xs = vec![];
    let mut ys = vec![];
    for i in 0..48 {
        let c = i % 4;
        let mut v = vec![0.0f32; 4];
        v[c] = 1.0 + ((i * 31) % 7) as f32 / 20.0;
        xs.push(v);
        ys.push(c);
    }
    let p = fit_probe(Template::Relation, Family::D1, "m", &xs, &ys, &ProbeOptions::default()).unwrap();
    assert!(p.cv_score > 0.95, "{}", p.cv_score);
    assert!(fit_probe(Template::Supersedes, Family::D1, "m", &xs[..3], &[0, 0, 1], &ProbeOptions::default()).is_err());
}

// ------------------------------------------------------------------ symmetric scoring

#[test]
fn symmetric_scoring_averages_both_orders() {
    // CNLI options; duplicate reads Entailment (index 0). Order (x, y)
    // where the state is "first" gives P(ent) = 0.8, the reverse 0.4.
    let m = Mock::new("sym-model", 8, |r| {
        let p: f32 = if r.state == "first" { 0.8 } else { 0.4 };
        let rest = (1.0 - p) / 2.0;
        (vec![p.ln(), rest.ln(), rest.ln()], vec![])
    });
    decide::register_decider("sym", m.clone(), Family::D1).unwrap();
    let cal = Calibration::default();
    let d = decide::decide_with(&*m, Family::D1, Template::Duplicate, &[("first", "second")], &cal, None).unwrap();
    assert!((d[0].p - 0.6).abs() < 1e-6, "{}", d[0].p);
    let seen = m.seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].state, "first");
    assert_eq!(seen[1].state, "second");
    assert!(seen[1].question.ends_with("\nfirst"));
    // supersedes is directional: one request
    m.seen.lock().unwrap().clear();
    decide::decide_with(&*m, Family::D1, Template::Supersedes, &[("first", "second")], &cal, None).unwrap();
    assert_eq!(m.seen.lock().unwrap().len(), 1);
    // and the result is order-invariant for symmetric templates
    let a = decide::decide("sym", Template::Duplicate, &[("first", "second")]).unwrap();
    let b = decide::decide("sym", Template::Duplicate, &[("second", "first")]).unwrap();
    assert!((a[0].p - b[0].p).abs() < 1e-12);
    decide::unregister_decider("sym");
}

// ------------------------------------------------------------------ CozoScript surface

#[test]
fn list_enable_disable_and_errors() {
    let m = Mock::new("toggle-model", 8, |r| (vec![0.0; r.options.len()], vec![]));
    decide::register_decider("tog", m.clone(), Family::Nano).unwrap();
    let db = mem();
    let rows = run(&db, "::decider list").unwrap();
    assert_eq!(
        rows.headers,
        vec!["name", "kind", "family", "model", "enabled", "loaded", "precision", "device"]
    );
    let row = rows.rows.iter().find(|r| r[0] == DataValue::from("tog")).unwrap();
    assert_eq!(row[1], DataValue::from("custom"));
    assert_eq!(row[2], DataValue::from("nano"));
    assert_eq!(row[3], DataValue::from("toggle-model"));
    assert_eq!(row[4], DataValue::Bool(true));

    let p = run(&db, "?[p] := p = decide_supersedes('tog', 'a', 'b')").unwrap();
    let pv = p.rows[0][0].get_float().unwrap();
    assert!((pv - 1.0 / 3.0).abs() < 1e-6);
    let l = run(&db, "?[l, b] := l = decide_label('tog', 'supersedes', 'a', 'b'), b = decide_band('tog', 'supersedes', 'a', 'b')").unwrap();
    // shipped nano calibration: threshold 0.246, never auto (tau 1.01)
    assert_eq!(l.rows[0][0], DataValue::from("yes"));
    assert_eq!(l.rows[0][1], DataValue::from("tentative"));
    let r = run(&db, "?[l] := l = decide_relation('tog', 'a', 'b')").unwrap();
    assert!(matches!(&r.rows[0][0], DataValue::Str(_)));
    let i = run(&db, "?[p] := p = decide_important('tog', 'a fact')").unwrap();
    assert!((i.rows[0][0].get_float().unwrap() - 0.5).abs() < 1e-6);
    // nulls pass through
    let n = run(&db, "?[p] := p = decide_supersedes('tog', null, 'b')").unwrap();
    assert_eq!(n.rows[0][0], DataValue::Null);

    assert_eq!(run(&db, "::decider disable tog").unwrap().rows[0][0], DataValue::from("OK"));
    assert_eq!(m.unloads.load(Ordering::SeqCst), 1);
    assert_eq!(run(&db, "::decider disable tog").unwrap().rows[0][0], DataValue::from("ALREADY_DISABLED"));
    let rows = run(&db, "::decider list").unwrap();
    let row = rows.rows.iter().find(|r| r[0] == DataValue::from("tog")).unwrap();
    assert_eq!(row[4], DataValue::Bool(false));
    let e = err(&db, "?[p] := p = decide_supersedes('tog', 'a', 'b')");
    let e = e.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(e.contains("decider 'tog' is disabled; turn it back on with `::decider enable tog`"), "{e}");
    assert_eq!(run(&db, "::decider enable tog").unwrap().rows[0][0], DataValue::from("OK"));
    run(&db, "?[p] := p = decide_supersedes('tog', 'a', 'b')").unwrap();

    // unknown decider / template / arity / options / read-only
    let e = err(&db, "?[p] := p = decide_supersedes('nope', 'a', 'b')");
    assert!(e.contains("no decider is enabled under the name 'nope'"), "{e}");
    assert!(e.contains("::decider enable nope {model: 'decider-nano'}"), "{e}");
    let e = err(&db, "?[p] := p = decide_p('tog', 'frobnicate', 'a', 'b')");
    assert!(e.contains("unknown decide template 'frobnicate'"), "{e}");
    let e = err(&db, "?[p] := p = decide_p('tog', 'important', 'a', 'b')");
    assert!(e.contains("takes 1 text argument(s), got 2"), "{e}");
    let e = err(&db, "::decider disable never_enabled");
    assert!(e.contains("no decider is enabled under the name 'never_enabled'"), "{e}");
    let e = err(&db, "::decider enable x {model: 'decider-nano', dim: 3}");
    assert!(e.contains("unknown option 'dim' for `::decider enable`"), "{e}");
    let e = format!(
        "{:?}",
        db.run_script("::decider disable tog", Default::default(), ScriptMutability::Immutable)
            .unwrap_err()
    );
    assert!(e.contains("cannot run in read-only mode"), "{e}");
    db.run_script("::decider list", Default::default(), ScriptMutability::Immutable)
        .unwrap();
    #[cfg(not(feature = "infer"))]
    {
        let e = err(&db, "::decider enable nano {model: 'decider-nano'}");
        assert!(e.contains("built without the `infer` cargo feature"), "{e}");
    }
    decide::unregister_decider("tog");
}

#[test]
fn decide_fixed_rule_batches() {
    let m = Mock::new("rule-model", 4, |r| {
        // Contradiction (index 1) wins when the new statement says "moved"
        let c = if r.question.contains("moved") { 3.0 } else { -3.0 };
        (vec![0.0, c, 0.0], vec![])
    });
    decide::register_decider("rule", m.clone(), Family::D1).unwrap();
    let db = mem();
    run(&db, ":create pairs {id: Int => old: String, new: String}").unwrap();
    let mut rows = vec![];
    for i in 0..10 {
        // distinct texts: identical pairs would be answered once (decision cache)
        let new = if i % 2 == 0 { format!("Tom moved to Lisbon (note {i})") } else { format!("Tom likes jazz (note {i})") };
        rows.push(format!("[{i}, 'Tom lives in Berlin', '{new}']"));
    }
    run(&db, &format!("?[id, old, new] <- [{}] :put pairs {{id => old, new}}", rows.join(", "))).unwrap();
    let out = run(
        &db,
        "p[id, a, b] := *pairs{id, old: a, new: b}
         ?[id, label, p, band, by] <~ Decide(p[id, a, b], model: 'rule', template: 'supersedes')",
    )
    .unwrap();
    assert_eq!(out.rows.len(), 10);
    // 10 requests in batches of at most 4 -> 3 calls
    assert_eq!(m.calls.load(Ordering::SeqCst), 3);
    assert_eq!(*m.batch_sizes.lock().unwrap(), vec![4, 4, 2]);
    for r in &out.rows {
        let id = r[0].get_int().unwrap();
        assert_eq!(r[1], DataValue::from(if id % 2 == 0 { "yes" } else { "no" }));
        assert!(matches!(&r[3], DataValue::Str(s) if ["auto", "tentative", "none"].contains(&s.as_str())));
        assert_eq!(r[4], DataValue::from("rule-model|supersedes@v1|calibrated"));
    }
    // recording decisions as rows: the documented audit pattern
    run(&db, ":create decisions {id: Int, template: String => label: String, p: Float, band: String, decided_by: String}").unwrap();
    run(
        &db,
        "p[id, a, b] := *pairs{id, old: a, new: b}
         d[id, label, p, band, by] <~ Decide(p[id, a, b], model: 'rule', template: 'supersedes')
         ?[id, template, label, p, band, decided_by] := d[id, label, p, band, decided_by], template = 'supersedes'
         :put decisions {id, template => label, p, band, decided_by}",
    )
    .unwrap();
    assert_eq!(run(&db, "?[count(id)] := *decisions{id}").unwrap().rows[0][0], DataValue::from(10));
    // the second run was answered by the pair-decision cache
    assert_eq!(m.calls.load(Ordering::SeqCst), 3);
    // wrong input arity / unknown template
    let e = err(&db, "p[id, a] := *pairs{id, old: a}\n?[a, b, c, d, e] <~ Decide(p[id, a], model: 'rule', template: 'supersedes')");
    assert!(e.contains("expects input rows [id, a, b]"), "{e}");
    let e = err(&db, "p[id, a, b] := *pairs{id, old: a, new: b}\n?[a, b, c, d, e] <~ Decide(p[id, a, b], model: 'rule', template: 'nah')");
    assert!(e.contains("unknown decide template 'nah'"), "{e}");
    decide::unregister_decider("rule");
}

// ------------------------------------------------------------------ fit -> persist -> reopen

fn probe_mock(id: &str) -> Arc<Mock> {
    // zero-shot logits are uninformative; the features carry the label
    Mock::new(id, 16, |r| {
        let pos = r.question.contains("moved") || r.question.contains("quit");
        let s = if pos { 1.0f32 } else { -1.0 };
        let h = r.question.len() as f32 % 7.0 / 7.0;
        let feats = (0..r.options.len()).map(|k| vec![s * (k as f32 + 1.0), h, 0.25]).collect();
        (vec![0.0; r.options.len()], feats)
    })
}

#[test]
fn fit_persist_reopen_probe_applied() {
    let dir = std::env::temp_dir().join(format!("kleos-decide-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("db.sqlite");
    let path = path.to_str().unwrap();
    let model = "probe-model-v1";
    decide::register_decider("pm", probe_mock(model), Family::D1).unwrap();
    {
        let db = DbInstance::new("sqlite", path, "").unwrap();
        run(&db, ":create lab {id: Int => old: String, new: String, label: Bool}").unwrap();
        let mut rows = vec![];
        for i in 0..20 {
            let (new, y) = match i % 4 {
                0 => (format!("Person {i} moved to city {i}"), true),
                1 => (format!("Person {i} quit job {i}"), true),
                2 => (format!("Person {i} likes tea {i}"), false),
                _ => (format!("Person {i} has a cat {i}"), false),
            };
            rows.push(format!("[{i}, 'Person {i} lives somewhere', '{new}', {y}]"));
        }
        run(&db, &format!("?[id, old, new, label] <- [{}] :put lab {{id => old, new, label}}", rows.join(", "))).unwrap();

        // before fitting: zero-shot, uninformative
        let d = db.decide("pm", "supersedes", "X lives in Y", "X moved to Z").unwrap();
        assert_eq!(d.via, "calibrated");
        assert!((d.p - 1.0 / 3.0).abs() < 1e-6);

        let r = run(&db, "::decider fit pm supersedes lab {alpha: 0.05}").unwrap();
        assert_eq!(r.headers[..3], ["model", "template", "kind"]);
        assert_eq!(r.rows[0][0], DataValue::from(model));
        assert_eq!(r.rows[0][2], DataValue::from("probe"));
        assert_eq!(r.rows[0][3], DataValue::from(20));
        assert!(r.rows[0][4].get_float().unwrap() > 0.95);

        let d = db.decide("pm", "supersedes", "X lives in Y", "X moved to Z").unwrap();
        assert_eq!((d.via.as_str(), d.label.as_str()), ("probe", "yes"));
        let d = db.decide("pm", "supersedes", "X lives in Y", "X likes jazz").unwrap();
        assert_eq!((d.via.as_str(), d.label.as_str()), ("probe", "no"));
        let f = run(&db, "::decider fitted").unwrap();
        assert!(f.rows.iter().any(|r| r[0] == DataValue::from(model) && r[2] == DataValue::from("probe")));
        // a calibration-only fit for another template, persisted too
        run(&db, ":create lab3 {id: Int => a: String, b: String, label: String}").unwrap();
        run(
            &db,
            "?[id, a, b, label] <- [[1, 'a', 'b', 'update'], [2, 'c', 'd', 'duplicate'], [3, 'e', 'f', 'compatible'], [4, 'g', 'h', 'contradiction'], [5, 'i', 'j', 'update']] :put lab3 {id => a, b, label}",
        )
        .unwrap();
        let r = run(&db, "::decider fit pm relation lab3 {probe: false}").unwrap();
        assert_eq!(r.rows[0][2], DataValue::from("calibration"));
        let e = err(&db, "::decider fit pm relation lab");
        assert!(e.contains("is not valid for relation"), "{e}");
    }
    // simulate a restart: process state gone, database file kept
    assert!(decide::remove_fitted(model, Template::Supersedes));
    assert!(decide::remove_fitted(model, Template::Relation));
    assert_eq!(decide::decide_one("pm", Template::Supersedes, "a", "b moved").unwrap().via, "calibrated");
    {
        let db = DbInstance::new("sqlite", path, "").unwrap();
        let d = db.decide("pm", "supersedes", "X lives in Y", "X moved to Z").unwrap();
        assert_eq!((d.via.as_str(), d.label.as_str()), ("probe", "yes"));
        assert_eq!(db.decide("pm", "relation", "a", "b").unwrap().via, "calibrated");
        // the fixed rule uses the probe too
        let out = run(
            &db,
            "p[id, a, b] <- [[1, 'X lives in Y', 'X quit job Q'], [2, 'X lives in Y', 'X has a cat']]
             ?[id, label, p, band, by] <~ Decide(p[id, a, b], model: 'pm', template: 'supersedes')",
        )
        .unwrap();
        assert_eq!(out.rows[0][1], DataValue::from("yes"));
        assert_eq!(out.rows[1][1], DataValue::from("no"));
        assert_eq!(out.rows[0][4], DataValue::from("probe-model-v1|supersedes@v1|probe"));
        // forget removes it from the process and the database
        assert_eq!(run(&db, "::decider forget pm supersedes").unwrap().rows[0][0], DataValue::from("OK"));
        assert_eq!(db.decide("pm", "supersedes", "a", "b moved").unwrap().via, "calibrated");
        assert_eq!(run(&db, "::decider forget pm supersedes").unwrap().rows[0][0], DataValue::from("NOT_FITTED"));
    }
    {
        let db = DbInstance::new("sqlite", path, "").unwrap();
        assert_eq!(db.decide("pm", "supersedes", "a", "b moved").unwrap().via, "calibrated");
        drop(db);
    }
    decide::unregister_decider("pm");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fit_errors() {
    let m = Mock::new("nofeat-model", 8, |r| (vec![0.0; r.options.len()], vec![]));
    decide::register_decider("nofeat", m, Family::D1).unwrap();
    let db = mem();
    run(&db, ":create l {id: Int => a: String, b: String, y: Bool}").unwrap();
    run(&db, "?[id, a, b, y] <- [[1,'a','b',true],[2,'c','d',false],[3,'e','f',true],[4,'g','h',false]] :put l {id => a, b, y}").unwrap();
    let e = err(&db, "::decider fit nofeat supersedes l");
    assert!(e.contains("returns no features"), "{e}");
    // a calibration needs no features
    run(&db, "::decider fit nofeat supersedes l {probe: false}").unwrap();
    let e = err(&db, "::decider fit nofeat supersedes missing_rel");
    assert!(e.to_lowercase().contains("missing_rel"), "{e}");
    let e = err(&db, "::decider fit nofeat supersedes l {alpha: 2}");
    assert!(e.contains("'alpha' must be a number in (0, 1)"), "{e}");
    let e = err(&db, "{::decider fit nofeat supersedes l}");
    assert!(e.contains("must run as a script of its own"), "{e}");
    decide::remove_fitted("nofeat-model", Template::Supersedes);
    decide::unregister_decider("nofeat");
}

#[test]
fn third_backend_registers_its_own_family() {
    use kleos::decide::{register_family, FamilyTemplates, TemplateSpec};
    fn spec(_t: Template) -> TemplateSpec {
        fn st(a: &str, b: &str) -> String {
            format!("A: {a}\nB: {b}")
        }
        fn q(_a: &str, _b: &str) -> String {
            "Is B a change of A?".to_string()
        }
        TemplateSpec {
            state: st,
            question: q,
            options: &["kept: A is still true after B", "changed: B replaces what A said"],
            label_of_option: &[0, 1],
        }
    }
    fn bare(_t: Template) -> TemplateSpec {
        fn st(a: &str, _b: &str) -> String {
            a.to_string()
        }
        TemplateSpec { state: st, question: st, options: &["false: no", "true: yes"], label_of_option: &[0, 1] }
    }
    fn cal(_t: Template) -> Calibration {
        Calibration { threshold: 0.6, tau_auto: 0.95, source: "custom".into(), ..Default::default() }
    }
    let fam = register_family("jev", FamilyTemplates { spec, version: 3, calibration: cal }).unwrap();
    assert_eq!(fam.name(), "jev");
    assert_eq!(Family::parse("JEV").unwrap(), fam);
    assert!(register_family("nano", FamilyTemplates { spec, version: 1, calibration: cal }).is_err());
    let e = format!("{:?}", register_family("bare", FamilyTemplates { spec: bare, version: 1, calibration: cal }).unwrap_err());
    assert!(e.contains("no bare yes/no answers"), "{e}");
    let r = Template::Supersedes.render(fam, "x", "y");
    assert_eq!((r.state.as_str(), r.question.as_str()), ("A: x\nB: y", "Is B a change of A?"));
    assert_eq!(Template::Supersedes.versioned(fam), "supersedes@v3");
    let m = Mock::new("jev-model", 4, |_| (vec![0.0, 0.5], vec![]));
    decide::register_decider("jev1", m, fam).unwrap();
    let d = decide::decide_one("jev1", Template::Supersedes, "x", "y").unwrap();
    assert_eq!((d.label.as_str(), d.via.as_str(), d.template_version), ("yes", "calibrated", 3));
    let db = mem();
    let rows = run(&db, "::decider list").unwrap();
    assert!(rows.rows.iter().any(|r| r[0] == DataValue::from("jev1") && r[2] == DataValue::from("jev")));
    decide::unregister_decider("jev1");
}

/// NLI logits that always say "entailment" (so supersedes p is low).
fn always_entail(_: &DecideRequest) -> (Vec<f32>, Vec<Vec<f32>>) {
    (vec![4.0, 0.0, -4.0], vec![vec![1.0]; 3])
}

#[test]
fn pair_decision_cache() {
    let m = Mock::new("cache-mock-v1", 8, always_entail);
    decide::register_decider("cachemock", m.clone(), Family::Nano).unwrap();
    let items = [("Ava likes tea.", "Ava drinks tea daily."), ("Ava likes tea.", "Ava owns a bike.")];
    let a = decide::decide("cachemock", Template::Supersedes, &items).unwrap();
    let calls = m.calls.load(Ordering::SeqCst);
    assert_eq!(m.seen.lock().unwrap().len(), 2);
    // same pairs again: answered from the cache, no model call
    let b = decide::decide("cachemock", Template::Supersedes, &items).unwrap();
    assert_eq!(a, b);
    assert_eq!(m.calls.load(Ordering::SeqCst), calls);
    // another template renders other requests (duplicate is symmetric: 2 orders x 2 items)
    decide::decide("cachemock", Template::Duplicate, &items).unwrap();
    assert_eq!(m.seen.lock().unwrap().len(), 2 + 4);
    // duplicates inside one batch go to the model once
    decide::decide("cachemock", Template::Contradicts, &[items[0], items[0], items[0]]).unwrap();
    assert_eq!(m.seen.lock().unwrap().len(), 6 + 2);
    // re-registering the model (weights may have changed) drops its entries
    decide::register_decider("cachemock", m.clone(), Family::Nano).unwrap();
    decide::decide("cachemock", Template::Supersedes, &items).unwrap();
    assert_eq!(m.seen.lock().unwrap().len(), 8 + 2);
    let (hits, _, _) = decide::decision_cache_stats();
    assert!(hits >= 2);
}

#[test]
fn value_rule_short_circuits_numbers_and_dates() {
    let m = Mock::new("rule-mock-v1", 8, always_entail);
    decide::register_decider("rulemock", m.clone(), Family::Nano).unwrap();
    // only the value changed: the rule decides "update", whatever the model says
    let d = decide::decide_one("rulemock", Template::Supersedes, "Ana's rent is €1,250 per month.", "Ana's rent is €1,320 per month.").unwrap();
    assert_eq!(d.label, "yes");
    assert_eq!(d.via, "rule");
    assert!(d.reason.as_deref().unwrap().contains("1250 -> 1320"), "{d:?}");
    let r = decide::decide_one("rulemock", Template::Relation, "The launch is scheduled for May 3.", "The launch is now scheduled for June 9.").unwrap();
    assert_eq!((r.label.as_str(), r.via.as_str()), ("update", "rule"));
    let dup = decide::decide_one("rulemock", Template::Duplicate, "Kai speaks two languages.", "Kai now speaks three languages.").unwrap();
    assert_eq!(dup.label, "no");
    // different events (date and amount differ) and value-free pairs: the model decides
    let e = decide::decide_one("rulemock", Template::Supersedes, "Tom ran 5 km on Monday.", "Tom ran 10 km on Tuesday.").unwrap();
    assert_ne!(e.via, "rule");
    let n = decide::decide_one("rulemock", Template::Supersedes, "Ava likes tea.", "Ava hates tea.").unwrap();
    assert_ne!(n.via, "rule");
    assert!(n.reason.is_none());
}

/// The real decider-nano through `::decider enable` (feature `infer`; skips
/// when the model is not in the Hugging Face cache).
#[cfg(feature = "infer")]
#[test]
fn real_decider_nano_end_to_end() {
    if kleos_infer::resolve_hf_file(kleos_infer::DECIDER_NANO_REPO, "model.safetensors").is_err() {
        eprintln!("skipping: tasksource/tasksource-decider-nano is not in the HF cache");
        return;
    }
    let db = mem();
    run(&db, "::decider enable realnano {model: 'decider-nano', precision: 'f32'}").unwrap();
    let up = db.decide("realnano", "supersedes", "Tom lives in Berlin.", "Tom moved to Lisbon in March and has settled in there.").unwrap();
    let compat = db.decide("realnano", "supersedes", "Tom lives in Berlin.", "Tom is learning to play the cello.").unwrap();
    assert!(up.p > compat.p, "{up:?} {compat:?}");
    // The zero-shot threshold (0.246, fitted on adapt) favours recall, so a
    // compatible addition can still be labelled "yes"; it must never be
    // auto-applied, and zero-shot supersedes never auto-applies at all.
    assert_ne!(up.band, Band::Auto);
    assert_ne!(compat.band, Band::Auto);
    let dup = db.decide("realnano", "duplicate", "Lena is a vegetarian.", "Lena doesn't eat meat.").unwrap();
    assert_eq!(dup.label, "yes");
    let rows = run(&db, "::decider list").unwrap();
    let row = rows.rows.iter().find(|r| r[0] == DataValue::from("realnano")).unwrap();
    assert_eq!(row[1], DataValue::from("decider-nano"));
    assert_eq!(row[5], DataValue::Bool(true));
    let e = err(&db, "::decider enable d1 {model: 'd1-omni'}");
    assert!(e.contains("d1-omni backend is not built into this kleos yet"), "{e}");
    // local weights: `path` (here the cached snapshot itself; a fine-tune in practice)
    let snap = kleos_infer::resolve_hf_file(kleos_infer::DECIDER_NANO_REPO, "model.safetensors").unwrap();
    let dir = snap.parent().unwrap().to_str().unwrap().to_string();
    run(&db, &format!("::decider enable localnano {{path: '{dir}', precision: 'f32'}}")).unwrap();
    let local = db.decide("localnano", "supersedes", "Tom lives in Berlin.", "Tom moved to Lisbon in March and has settled in there.").unwrap();
    assert!((local.p - up.p).abs() < 1e-6, "{local:?} {up:?}");
    assert_eq!(local.model, dir);
    let e = err(&db, "::decider enable badpath {path: '/nonexistent/nano'}");
    assert!(e.contains("not a decider-nano weights directory"), "{e}");
    assert_eq!(run(&db, "::decider disable localnano").unwrap().rows[0][0], DataValue::from("OK"));
    assert_eq!(run(&db, "::decider disable realnano").unwrap().rows[0][0], DataValue::from("OK"));
}
