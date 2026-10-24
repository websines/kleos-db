/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Zero-maintenance ontologies (docs/specs/ontology.md, "Defaults",
//! "Learning", "Maintenance"): the personal pack is on by default, the
//! ontology learns properties from untyped claims (auto and review modes),
//! corrects itself, every change can be undone, branches inherit it,
//! kinship has one source of truth, and a domain ontology's classes reach
//! the extractor as labels.

use std::sync::{Arc, Mutex, RwLock};

use kleos::decide::{self};
use kleos::embed::{self, Embedder};
use kleos::extract::{self, EntityExtractor, ExtractSpan};
use kleos::memory::{InitOpts, RememberOpts};
use kleos::{DbInstance, DecideAnswer, DecideRequest, Decider, Family};
use serde_json::{json, Value};

static REGISTRY: RwLock<()> = RwLock::new(());

const DAY: i64 = 86_400_000_000;
const T0: i64 = 1_767_225_600_000_000;

fn d(n: i64) -> i64 {
    T0 + n * DAY
}

/// Topic directions: claims about the same hobby point the same way, so
/// they cluster; everything else is spread by a word hash.
struct Topics;

const TOPICS: &[&[&str]] = &[&["aquarium", "fish", "tank"], &["chess", "elo", "rating"], &["sourdough", "starter", "loaf"]];

impl Embedder for Topics {
    fn embed(&self, texts: &[&str]) -> miette::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let low = t.to_lowercase();
                let mut v = vec![0f32; 512];
                for (i, ws) in TOPICS.iter().enumerate() {
                    if ws.iter().any(|w| low.contains(w)) {
                        v[i] += 5.0;
                    }
                }
                for w in low.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2) {
                    let h = w.bytes().fold(1469598103934665603u64, |h, b| (h ^ b as u64).wrapping_mul(1099511628211));
                    v[3 + (h % 509) as usize] += 0.3;
                }
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                v.iter().map(|x| x / n).collect()
            })
            .collect())
    }
    fn dim(&self) -> usize {
        512
    }
}

struct Neutral(String);

impl Decider for Neutral {
    fn model_id(&self) -> String {
        self.0.clone()
    }
    fn weights_version(&self) -> Option<String> {
        Some("nano-tune-v1".to_string())
    }
    fn decide(&self, reqs: &[DecideRequest], _f: bool) -> miette::Result<Vec<DecideAnswer>> {
        Ok(reqs.iter().map(|_| DecideAnswer { logits: vec![0.0, 6.0, 0.0], features: vec![] }).collect())
    }
}

struct Models {
    embedder: String,
    decider: String,
    _guard: std::sync::RwLockReadGuard<'static, ()>,
}

impl Models {
    fn new(tag: &str) -> Self {
        // automatic runs off: the tests call `learn` themselves
        std::env::set_var("KLEOS_ONTO_LEARN_EVERY", "0");
        let guard = REGISTRY.read().unwrap_or_else(|e| e.into_inner());
        let embedder = format!("oa_emb_{tag}");
        let decider = format!("oa_dec_{tag}");
        embed::register_embedder(&embedder, Arc::new(Topics)).unwrap();
        decide::register_decider(&decider, Arc::new(Neutral(format!("oa-{tag}"))), Family::Nano).unwrap();
        Models { embedder, decider, _guard: guard }
    }
    fn db(&self) -> DbInstance {
        let db = DbInstance::new("mem", "", "").unwrap();
        db.memory_init(InitOpts {
            embedder: Some(self.embedder.clone()),
            decider: Some(self.decider.clone()),
            ..Default::default()
        })
        .unwrap();
        db
    }
}

impl Drop for Models {
    fn drop(&mut self) {
        decide::unregister_decider(&self.decider);
        embed::unregister_embedder(&self.embedder);
    }
}

fn at(ns: &str, t: i64) -> RememberOpts {
    RememberOpts { ns: Some(ns.into()), at: Some(t), ..Default::default() }
}

fn onto(db: &DbInstance, op: &str, args: Value) -> Value {
    db.ontology_call(op, args).unwrap_or_else(|e| panic!("ontology {op}: {e:?}"))
}

/// Claims about an aquarium, with update cues, for one subject.
const AQUARIUM: &[&str] = &[
    "My aquarium has 12 fish.",
    "The aquarium now has 14 fish after the weekend.",
    "My aquarium tank is 200 litres.",
    "I added two fish to my aquarium, now 16 fish.",
    "The aquarium has 18 fish now.",
    "My aquarium now holds 20 fish.",
    "I cleaned my aquarium tank and it has 21 fish now.",
    "The aquarium now has 22 fish in total.",
    "My aquarium fish count is now 24.",
    "There are now 25 fish in my aquarium.",
];

#[test]
fn personal_pack_is_on_by_default_and_can_be_turned_off() {
    let m = Models::new("default");
    let db = m.db();
    // typed with no setup (the property itself needs a real embedder or
    // no embedder: cue typing; the stub's vectors type nothing)
    let b = db.remember("I live in Berlin.", at("u", d(1))).unwrap();
    assert_eq!(b.types.len(), 1, "{b:?}");
    assert_eq!(b.types[0].subject.as_deref(), Some("user"), "{b:?}");
    let s = db.remember("My sister Anna lives in Munich.", at("u", d(2))).unwrap();
    assert!(s.superseded.is_empty(), "{s:?}");
    // small talk is typed and never decided, with no setup at all
    let st = db.remember("Wow, that's amazing, Anna!", at("u", d(3))).unwrap();
    assert!(st.types.iter().all(|t| t.cls == "smalltalk"), "{st:?}");
    // off for the store
    db.memory_init(InitOpts { ontology: Some("none".into()), ..Default::default() }).unwrap();
    let r = db.remember("I live in Paris.", at("w", d(4))).unwrap();
    assert!(r.types.is_empty(), "{r:?}");
}

#[test]
fn learn_auto_adds_a_property_types_its_claims_and_undo_restores() {
    let m = Models::new("learn");
    let db = m.db();
    for (i, t) in AQUARIUM.iter().enumerate() {
        db.remember(t, at("u", d(i as i64))).unwrap();
    }
    // a few unrelated claims for the pool
    for (i, t) in ["I went to Rome with Anna.", "My car is a red Volvo.", "I read a book on Sunday."].iter().enumerate() {
        db.remember(t, at("u", d(20 + i as i64))).unwrap();
    }
    let out = onto(&db, "learn", json!({"ns": "u"}));
    let applied = out["applied"].as_array().unwrap();
    assert_eq!(applied.len(), 1, "{out:#}");
    let p = out["proposals"].as_array().unwrap().iter().find(|p| p["status"] == "active").unwrap();
    let name = p["body"]["prop"]["name"].as_str().unwrap().to_string();
    assert!(name.contains("aquarium") || name.contains("fish"), "{name}");
    // the (neutral) decider says the statements coexist: kept multi-valued,
    // the safe side (cue-only inference is covered by the unit tests)
    assert_eq!(p["body"]["prop"]["card"], "many", "{p:#}");
    assert!(p["body"]["shape"]["why"].as_str().unwrap().contains("decider"), "{p:#}");
    assert_eq!(p["body"]["prop"]["datatype"], "number", "{p:#}");
    // the learned layer is versioned and active
    let doc = onto(&db, "get", json!({"onto": "learned:u"}));
    assert!(doc["properties"].as_array().unwrap().iter().any(|x| x["name"] == name.as_str()), "{doc:#}");
    // the evidence claims carry the property now
    let typed = db
        .run_default(&format!("?[count(id)] := *kleos_mem_type{{ns: 'u', id, prop: '{name}'}}"))
        .unwrap()
        .rows[0][0]
        .get_int()
        .unwrap();
    assert!(typed >= 8, "{typed}");
    // history explains it; undo removes it as a new version
    let h = onto(&db, "history", json!({"ns": "u"}));
    let ch = h.as_array().unwrap().iter().find(|c| c["kind"] == "add_property").unwrap();
    assert!(ch["evidence"].as_array().unwrap().len() >= 8, "{ch:#}");
    onto(&db, "undo", json!({"ns": "u", "change": ch["id"]}));
    let doc = onto(&db, "get", json!({"onto": "learned:u"}));
    assert!(!doc["properties"].as_array().unwrap().iter().any(|x| x["name"] == name.as_str()), "{doc:#}");
    let h = onto(&db, "history", json!({"ns": "u"}));
    assert!(h.as_array().unwrap().iter().any(|c| c["id"] == ch["id"] && c["undone"] == true), "{h:#}");
    // a second learn does not re-propose what it already proposed
    let again = onto(&db, "learn", json!({"ns": "u"}));
    assert!(again["applied"].as_array().unwrap().is_empty(), "{again:#}");
}

#[test]
fn review_mode_proposes_and_approve_or_reject_decides() {
    let m = Models::new("review");
    let db = m.db();
    for (i, t) in AQUARIUM.iter().enumerate() {
        db.remember(t, at("r", d(i as i64))).unwrap();
    }
    let out = onto(&db, "learn", json!({"ns": "r", "mode": "review"}));
    assert!(out["applied"].as_array().unwrap().is_empty(), "{out:#}");
    let props = onto(&db, "proposals", json!({"ns": "r", "status": "proposed"}));
    let id = props.as_array().unwrap()[0]["id"].as_str().unwrap().to_string();
    assert!(db.ontology_call("get", json!({"onto": "learned:r"})).is_err());
    let a = onto(&db, "approve", json!({"ns": "r", "id": id}));
    assert_eq!(a["status"], "active");
    assert!(!onto(&db, "get", json!({"onto": "learned:r"}))["properties"].as_array().unwrap().is_empty());
    // a decided proposal cannot be decided again
    assert!(db.ontology_call("reject", json!({"ns": "r", "id": id})).is_err());
}

#[test]
fn maintenance_makes_a_contradicted_single_valued_property_multi_valued() {
    let m = Models::new("maint");
    let db = m.db();
    onto(
        &db,
        "define",
        json!({"onto": "hobby", "properties": [
            {"name": "favorite_color", "kind": "data", "card": "one", "label": "favorite color", "cues": ["favorite color", "favourite colour"]}
        ]}),
    );
    onto(&db, "activate", json!({"ns": "c", "ontos": ["personal", "hobby"]}));
    for (i, t) in [
        "My favorite color is red.",
        "My favorite color is blue.",
        "My favorite color is green.",
        "My sister's favorite color is pink.",
        "My sister's favorite color is teal.",
        "My sister's favorite color is gold.",
    ]
    .iter()
    .enumerate()
    {
        db.remember(t, at("c", d(i as i64))).unwrap();
    }
    // prototype-only types are not evidence: the same claims typed by a
    // prototype match alone change nothing
    let set_by = |by: &str| {
        db.run_script(
            "?[ns, id, by] := *kleos_mem_type{ns, id, prop}, ns = 'c', prop = 'favorite_color', by = $by \
             :update kleos_mem_type {ns, id => by}",
            [("by".to_string(), kleos::DataValue::from(by))].into_iter().collect(),
            kleos::ScriptMutability::Mutable,
        )
        .unwrap()
    };
    let orig = db
        .run_script(
            "?[by] := *kleos_mem_type{ns: 'c', prop: 'favorite_color', by} :limit 1",
            Default::default(),
            kleos::ScriptMutability::Immutable,
        )
        .unwrap();
    let orig_by = orig.rows[0][0].get_str().unwrap().to_string();
    set_by("proto");
    let none = onto(&db, "maintain", json!({"ns": "c", "dry_run": true}));
    assert!(!none["changes"].as_array().unwrap().iter().any(|c| c["kind"] == "cardinality"), "{none:#}");
    set_by(&orig_by);
    let dry = onto(&db, "maintain", json!({"ns": "c", "dry_run": true}));
    assert!(dry["changes"].as_array().unwrap().iter().any(|c| c["kind"] == "cardinality" && c["prop"] == "favorite_color"), "{dry:#}");
    let out = onto(&db, "maintain", json!({"ns": "c"}));
    assert_eq!(out["changes"].as_array().unwrap().len(), 1, "{out:#}");
    let doc = onto(&db, "get", json!({"onto": "learned:c"}));
    let p = doc["properties"].as_array().unwrap().iter().find(|x| x["name"] == "favorite_color").unwrap();
    assert_eq!(p["card"], "many", "{p:#}");
    // undo puts `one` back (the learned override goes away)
    let h = onto(&db, "history", json!({"ns": "c"}));
    onto(&db, "undo", json!({"ns": "c", "change": h[0]["id"]}));
    let doc = onto(&db, "get", json!({"onto": "learned:c"}));
    assert!(!doc["properties"].as_array().unwrap().iter().any(|x| x["name"] == "favorite_color"), "{doc:#}");
}

#[test]
fn a_branch_inherits_its_parents_ontology_and_facts_as_of_the_fork() {
    let m = Models::new("branch");
    let db = m.db();
    onto(&db, "define", json!({"onto": "pets", "properties": [{"name": "pet_name", "kind": "data", "card": "many", "label": "pet name", "cues": ["my dog", "my cat"]}]}));
    onto(&db, "activate", json!({"ns": "p", "ontos": ["personal", "pets"]}));
    db.remember("My sister Anna lives in Munich.", at("p", d(1))).unwrap();
    // the fork sees the parent as of day 2 (statements carry their own `at`)
    db.memory_call("fork", json!({"from": "p", "to": "b", "at": d(2)})).unwrap();
    // the parent learns something after the fork
    db.remember("My brother Tom lives in Oslo.", at("p", d(3))).unwrap();
    // the branch types with the parent's ontology
    let r = db.remember("My dog is called Rex.", at("b", d(4))).unwrap();
    let id = r.id.clone().unwrap();
    let o = db.run_default(&format!("?[onto] := *kleos_mem_type{{ns: 'b', id: '{id}', onto}}")).unwrap();
    assert_eq!(o.rows[0][0].get_str(), Some("personal+pets"), "{r:?}");
    let facts = onto(&db, "facts", json!({"ns": "b"}));
    let fs = facts.as_array().unwrap();
    assert!(fs.iter().any(|f| f["p"] == "sister_of" && f["inherited_from"] == "p"), "{facts:#}");
    assert!(!fs.iter().any(|f| f["p"] == "brother_of"), "{facts:#}");
    // inference reads the lineage too
    let inf = onto(&db, "infer", json!({"ns": "b"}));
    assert!(inf.as_array().unwrap().iter().any(|f| f["p"] == "sister_of"), "{inf:#}");
    // the branch retracts what it inherited: hidden in the branch only
    let out = onto(&db, "retract", json!({"ns": "b", "facts": [{"s": "anna", "p": "sister_of", "o": "user"}], "at": d(5)}));
    assert_eq!(out["inherited_hidden"], 1, "{out:#}");
    let fb = onto(&db, "facts", json!({"ns": "b"}));
    assert!(!fb.as_array().unwrap().iter().any(|f| f["p"] == "sister_of"), "{fb:#}");
    let ib = onto(&db, "infer", json!({"ns": "b"}));
    assert!(!ib.as_array().unwrap().iter().any(|f| f["s"] == "anna" && f["p"] == "sister_of"), "{ib:#}");
    let fp = onto(&db, "facts", json!({"ns": "p"}));
    assert!(fp.as_array().unwrap().iter().any(|f| f["p"] == "sister_of"), "the parent keeps it: {fp:#}");
}

/// Finds the listed spans; records the extra labels it was asked for.
struct Table(Vec<(&'static str, &'static str, f32)>, Arc<Mutex<Vec<String>>>);

impl EntityExtractor for Table {
    fn model_id(&self) -> String {
        "oa-table".into()
    }
    fn version(&self) -> String {
        "oa-table:v1".into()
    }
    fn labels(&self) -> Vec<String> {
        ["person", "location", "organization", "relative"].iter().map(|s| s.to_string()).collect()
    }
    fn extract(&self, texts: &[&str]) -> miette::Result<Vec<Vec<ExtractSpan>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut out = vec![];
                for (label, span, score) in &self.0 {
                    if let Some(a) = t.find(span) {
                        out.push(ExtractSpan { label: label.to_string(), start: a, end: a + span.len(), text: span.to_string(), score: *score });
                    }
                }
                out
            })
            .collect())
    }
    fn extract_labels(&self, texts: &[&str], extra: &[String]) -> miette::Result<Vec<Vec<ExtractSpan>>> {
        self.1.lock().unwrap().extend(extra.iter().cloned());
        self.extract(texts)
    }
}

#[test]
fn kinship_edges_have_one_source_of_truth() {
    let m = Models::new("kin");
    let db = m.db();
    let asked = Arc::new(Mutex::new(vec![]));
    extract::register_extractor(
        "oa_ext_kin",
        Arc::new(Table(vec![("person", "Tom", 0.95), ("relative", "his wife", 0.9)], asked)),
    )
    .unwrap();
    db.memory_init(InitOpts { extractor: Some("oa_ext_kin".into()), ..Default::default() }).unwrap();
    let r = db.remember("Tom said his wife moved to Rome.", at("k", d(1))).unwrap();
    assert!(!r.edges.is_empty(), "{r:?}");
    // the ontology holds the edge; kleos_mem_link does not duplicate it
    let links = db.run_default("?[count(src)] := *kleos_mem_link{ns: 'k', src, decided_by: 'gliner-rule'}").unwrap();
    assert_eq!(links.rows[0][0].get_int(), Some(0));
    let facts = onto(&db, "facts", json!({"ns": "k"}));
    assert!(
        facts.as_array().unwrap().iter().any(|f| f["p"] == "partner" && f["o"] == "tom" && f["by"] == "extractor:kin"),
        "{facts:#}"
    );
    extract::disable_extractor("oa_ext_kin").unwrap();
}

#[test]
fn a_domain_ontologys_classes_reach_the_extractor_as_labels() {
    let m = Models::new("labels");
    let db = m.db();
    let asked = Arc::new(Mutex::new(vec![]));
    extract::register_extractor("oa_ext_lab", Arc::new(Table(vec![("instrument", "AAPL", 0.9)], asked.clone()))).unwrap();
    db.memory_init(InitOpts { extractor: Some("oa_ext_lab".into()), ..Default::default() }).unwrap();
    onto(&db, "define", json!({"onto": "desk", "classes": [{"name": "Instrument"}, {"name": "Counterparty", "label": "counterparty"}]}));
    onto(&db, "activate", json!({"ns": "t", "ontos": ["personal", "desk"]}));
    let r = db.remember("Bought 5k AAPL from Citadel for the macro book.", at("t", d(1))).unwrap();
    let got = asked.lock().unwrap().clone();
    assert!(got.contains(&"instrument".to_string()) && got.contains(&"counterparty".to_string()), "{got:?}");
    // a domain span becomes an entity
    let names = db.run_default("?[name] := *kleos_mem_entity{ns: 't', name}").unwrap();
    assert!(names.rows.iter().any(|x| x[0].get_str() == Some("AAPL")), "{r:?} {names:?}");
    extract::disable_extractor("oa_ext_lab").unwrap();
}

#[test]
fn suggest_points_finance_notes_at_fibo() {
    let m = Models::new("suggest");
    let db = m.db();
    for (i, t) in [
        "The counterparty confirmed the swap notional.",
        "Settlement of the bond trade is T+2.",
        "Our exposure to that counterparty is too high.",
        "The equity portfolio hedge rolled today.",
        "The custodian rejected the trade settlement.",
        "We booked a new derivative trade.",
    ]
    .iter()
    .enumerate()
    {
        db.remember(t, at("f", d(i as i64))).unwrap();
    }
    let s = onto(&db, "suggest", json!({"ns": "f"}));
    assert!(s["packs"].as_array().unwrap().iter().any(|p| p["body"]["pack"] == "fibo"), "{s:#}");
    let props = onto(&db, "proposals", json!({"ns": "f", "status": "proposed"}));
    assert!(props.as_array().unwrap().iter().any(|p| p["kind"] == "pack"), "{props:#}");
}
