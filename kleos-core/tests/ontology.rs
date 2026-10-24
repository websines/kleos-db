/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Ontologies (docs/specs/ontology.md): the cardinality supersession policy
//! through `remember`, inference, validation, import/export, N-ary facts and
//! versioning. Typing here is cue-only (no embedder), with a stub decider
//! that answers "supersedes" for the listed pairs.

use std::sync::{Arc, RwLock};

use kleos::decide::{self};
use kleos::memory::{InitOpts, RememberOpts};
use kleos::{DbInstance, DecideAnswer, DecideRequest, Decider, Family, ScriptMutability};
use serde_json::{json, Value};

static REGISTRY: RwLock<()> = RwLock::new(());

const DAY: i64 = 86_400_000_000;
const T0: i64 = 1_767_225_600_000_000;

fn d(n: i64) -> i64 {
    T0 + n * DAY
}

/// Contradiction (→ supersedes yes) for the listed (old, new) pairs,
/// neutral otherwise.
struct Stub {
    id: String,
    contra: Vec<(&'static str, &'static str)>,
}

impl Decider for Stub {
    fn model_id(&self) -> String {
        self.id.clone()
    }
    fn weights_version(&self) -> Option<String> {
        Some("nano-tune-v1".to_string())
    }
    fn decide(&self, reqs: &[DecideRequest], _f: bool) -> miette::Result<Vec<DecideAnswer>> {
        Ok(reqs
            .iter()
            .map(|r| {
                let st = r.state.strip_prefix("Premise: ").unwrap_or(&r.state);
                let (a, b) = st.split_once("\nHypothesis: ").unwrap_or((st, ""));
                let logits = if self.contra.iter().any(|&(x, y)| x == a && y == b) {
                    vec![0.0, 0.0, 12.0]
                } else {
                    vec![0.0, 6.0, 0.0]
                };
                DecideAnswer { logits, features: vec![] }
            })
            .collect())
    }
}

struct Models {
    decider: String,
    _guard: std::sync::RwLockReadGuard<'static, ()>,
}

impl Models {
    fn new(tag: &str, contra: Vec<(&'static str, &'static str)>) -> Self {
        let guard = REGISTRY.read().unwrap_or_else(|e| e.into_inner());
        let decider = format!("onto_dec_{tag}");
        decide::register_decider(&decider, Arc::new(Stub { id: format!("onto-{tag}"), contra }), Family::Nano).unwrap();
        Models { decider, _guard: guard }
    }
    /// A store with the decider, and the personal ontology active on `ns`
    /// unless `onto` is false.
    fn db(&self, onto: bool) -> DbInstance {
        let db = DbInstance::new("mem", "", "").unwrap();
        db.memory_init(InitOpts {
            decider: Some(self.decider.clone()),
            // the personal pack is on by default: `none` turns it off
            ontology: Some(if onto { "personal" } else { "none" }.to_string()),
            ns: Some("u".into()),
            ..Default::default()
        })
        .unwrap();
        db
    }
}

impl Drop for Models {
    fn drop(&mut self) {
        decide::unregister_decider(&self.decider);
    }
}

fn at(t: i64) -> RememberOpts {
    RememberOpts { ns: Some("u".into()), at: Some(t), ..Default::default() }
}

/// Remember `old` then `new`; whether `new` superseded `old`.
fn supersedes(db: &DbInstance, old: &str, new: &str) -> (bool, kleos::memory::RememberResult) {
    let a = db.remember(old, at(d(1))).unwrap();
    let b = db.remember(new, at(d(2))).unwrap();
    (b.superseded.contains(a.id.as_ref().unwrap()), b)
}

const HIKE_OLD: &str = "Caroline: Looks like you had a great day!";
const HIKE_NEW: &str = "Caroline: Recently, I had a not-so-great experience on a hike.";
const BERLIN: &str = "I live in Berlin.";
const LISBON: &str = "I live in Lisbon now.";
const PIZZA: &str = "I love pizza.";
const SUSHI: &str = "I love sushi.";
const PARIS: &str = "I live in Paris.";
const SISTER: &str = "My sister lives in Paris.";

#[test]
fn small_talk_never_supersedes() {
    let m = Models::new("st", vec![(HIKE_OLD, HIKE_NEW)]);
    // without an ontology the stub's "yes" is applied (today's behaviour)
    assert!(supersedes(&m.db(false), HIKE_OLD, HIKE_NEW).0);
    let (sup, r) = supersedes(&m.db(true), HIKE_OLD, HIKE_NEW);
    assert!(!sup, "{r:?}");
    assert!(r.decisions.is_empty(), "small talk is never decided: {:?}", r.decisions);
    assert_eq!(r.types[0].cls, "fact");
}

#[test]
fn single_valued_property_supersedes() {
    let m = Models::new("one", vec![(BERLIN, LISBON)]);
    let (sup, r) = supersedes(&m.db(true), BERLIN, LISBON);
    assert!(sup, "{r:?}");
    assert_eq!(r.types[0].prop.as_deref(), Some("lives_in"));
    assert_eq!(r.types[0].subject.as_deref(), Some("user"));
}

#[test]
fn multi_valued_property_never_supersedes() {
    let m = Models::new("many", vec![(PIZZA, SUSHI)]);
    assert!(supersedes(&m.db(false), PIZZA, SUSHI).0);
    let (sup, r) = supersedes(&m.db(true), PIZZA, SUSHI);
    assert!(!sup, "{r:?}");
    assert_eq!(r.types[0].prop.as_deref(), Some("likes"));
    assert!(!r.ontology_gated.is_empty(), "{r:?}");
    // both stay current
    let db = m.db(true);
    db.remember(PIZZA, at(d(1))).unwrap();
    db.remember(SUSHI, at(d(2))).unwrap();
    let hits = db.recall("love", kleos::memory::RecallOpts { ns: Some("u".into()), ..Default::default() }).unwrap();
    assert_eq!(hits.len(), 2, "{hits:?}");
}

#[test]
fn different_subject_never_supersedes() {
    let m = Models::new("subj", vec![(PARIS, SISTER)]);
    assert!(supersedes(&m.db(false), PARIS, SISTER).0);
    let (sup, r) = supersedes(&m.db(true), PARIS, SISTER);
    assert!(!sup, "{r:?}");
    assert_eq!(r.types[0].subject.as_deref(), Some("user.sister"));
    assert!(!r.ontology_blocked.is_empty(), "{r:?}");
}

#[test]
fn policy_off_is_todays_behaviour() {
    let m = Models::new("off", vec![(PIZZA, SUSHI)]);
    let db = m.db(true);
    db.ontology_call("activate", json!({"ns": "u", "ontos": ["personal"], "policy": "off"})).unwrap();
    assert!(supersedes(&db, PIZZA, SUSHI).0);
}

fn infer(db: &DbInstance, ns: &str) -> Vec<(String, String, String, bool)> {
    let v = db.ontology_call("infer", json!({"ns": ns})).unwrap();
    v.as_array()
        .unwrap()
        .iter()
        .map(|f| {
            (
                f["s"].as_str().unwrap().to_string(),
                f["p"].as_str().unwrap().to_string(),
                f["o"].as_str().unwrap().to_string(),
                f["derived"].as_bool().unwrap(),
            )
        })
        .collect()
}

fn has(fs: &[(String, String, String, bool)], s: &str, p: &str, o: &str) -> bool {
    fs.iter().any(|f| f.0 == s && f.1 == p && f.2 == o)
}

#[test]
fn inference_symmetric_inverse_sub_property_chain() {
    let db = DbInstance::new("mem", "", "").unwrap();
    db.ontology_call("activate", json!({"ns": "k", "ontos": ["personal"]})).unwrap();
    db.ontology_call(
        "assert",
        json!({"ns": "k", "facts": [
            {"s": "anna", "p": "sister_of", "o": "mom"},
            {"s": "mom", "p": "parent_of", "o": "user"},
            {"s": "kreuzberg", "p": "located_in", "o": "berlin"},
            {"s": "berlin", "p": "located_in", "o": "germany"},
        ]}),
    )
    .unwrap();
    let fs = infer(&db, "k");
    // sub-property, then symmetric
    assert!(has(&fs, "anna", "sibling_of", "mom"));
    assert!(has(&fs, "mom", "sibling_of", "anna"));
    // inverse
    assert!(has(&fs, "user", "child_of", "mom"));
    // chain: sister of a parent
    assert!(has(&fs, "anna", "aunt_of", "user"));
    // transitive
    assert!(has(&fs, "kreuzberg", "located_in", "germany"));
    assert!(fs.iter().any(|f| f.0 == "anna" && f.1 == "sister_of" && !f.3));
    assert!(fs.iter().any(|f| f.0 == "anna" && f.1 == "aunt_of" && f.3));
}

#[test]
fn remember_writes_kinship_facts_and_types() {
    let m = Models::new("kin", vec![]);
    let db = m.db(true);
    let r = db.remember("My sister Anna lives in Munich.", at(d(1))).unwrap();
    assert_eq!(r.types[0].subject.as_deref(), Some("user.sister"));
    let fs = infer(&db, "u");
    assert!(has(&fs, "anna", "sister_of", "user"), "{fs:?}");
    assert!(has(&fs, "user", "sibling_of", "anna"), "{fs:?}");
    // the relation key is the same node as the name
    assert!(has(&fs, "user.sister", "sister_of", "user"), "{fs:?}");
    let n = db
        .run_script("?[id, prop] := *kleos_mem_type{ns: 'u', id, prop}", Default::default(), ScriptMutability::Immutable)
        .unwrap();
    assert_eq!(n.rows.len(), 1);
}

#[test]
fn validation_domain_range_datatype_cardinality() {
    let db = DbInstance::new("mem", "", "").unwrap();
    db.ontology_call("activate", json!({"ns": "v", "ontos": ["personal"]})).unwrap();
    let r = db
        .ontology_call(
            "assert",
            json!({"ns": "v",
                   "nodes": [{"node": "user", "class": "Person"}, {"node": "acme", "class": "Organization"}],
                   "facts": [{"s": "user", "p": "lives_in", "o": "acme"},
                             {"s": "user", "p": "age", "lit": {"value": "thirty"}}]}),
        )
        .unwrap();
    let rules: Vec<&str> = r["violations"].as_array().unwrap().iter().map(|v| v["rule"].as_str().unwrap()).collect();
    assert!(rules.contains(&"range"), "{r}");
    assert!(rules.contains(&"datatype"), "{r}");
    let e = db
        .ontology_call(
            "assert",
            json!({"ns": "v", "validate": "reject", "facts": [{"s": "user", "p": "lives_in", "o": "acme"}]}),
        )
        .unwrap_err();
    assert!(e.to_string().contains("rejected"), "{e}");
    let e = db.ontology_call("assert", json!({"ns": "v", "facts": [{"s": "user", "p": "flies_to", "o": "mars"}]})).unwrap();
    assert_eq!(e["violations"][0]["rule"], "property");
}

const TRADING: &str = r#"
@prefix : <https://acme.example/onto/trading#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
<https://acme.example/onto/trading> a owl:Ontology ; owl:versionInfo "1" .
:Event a owl:Class .
:Trade a owl:Class ; rdfs:subClassOf :Event ; skos:altLabel "fill"@en .
:Instrument a owl:Class .
:Organization a owl:Class .
:instrument a owl:ObjectProperty, owl:FunctionalProperty ; rdfs:domain :Trade ; rdfs:range :Instrument .
:price a owl:DatatypeProperty, owl:FunctionalProperty ; rdfs:domain :Trade ; rdfs:range xsd:decimal .
:quantity a owl:DatatypeProperty, owl:FunctionalProperty ; rdfs:domain :Trade ; rdfs:range xsd:integer .
:counterparty a owl:ObjectProperty, owl:FunctionalProperty ; rdfs:domain :Trade ; rdfs:range :Organization .
:isSubsidiaryOf a owl:ObjectProperty, owl:TransitiveProperty ; rdfs:domain :Organization ; rdfs:range :Organization .
:hasSubsidiary a owl:ObjectProperty ; owl:inverseOf :isSubsidiaryOf .
"#;

#[test]
fn domain_ontology_import_events_and_inference() {
    let db = DbInstance::new("mem", "", "").unwrap();
    let r = db.ontology_call("import", json!({"format": "turtle", "text": TRADING, "onto": "acme"})).unwrap();
    assert_eq!(r["properties"], 6, "{r}");
    db.ontology_call("activate", json!({"ns": "desk", "ontos": ["acme"]})).unwrap();
    // an N-ary fact: one event node, its roles, typed literals
    let r = db
        .ontology_call(
            "assert",
            json!({"ns": "desk", "validate": "reject",
                   "nodes": [{"node": "aapl", "class": "Instrument"}, {"node": "gs", "class": "Organization"}],
                   "event": {"id": "ev:t1", "class": "Trade", "roles": {
                       "instrument": "aapl", "counterparty": "gs",
                       "quantity": {"value": 500, "datatype": "xsd:integer"},
                       "price": {"value": 187.2, "datatype": "xsd:decimal", "unit": "USD"}}}}),
        )
        .unwrap();
    assert_eq!(r["event"], "ev:t1");
    // a corrected price ends the old one (single-valued per event)
    db.ontology_call(
        "assert",
        json!({"ns": "desk", "facts": [{"s": "ev:t1", "p": "price", "lit": {"value": 187.25, "datatype": "xsd:decimal", "unit": "USD"}}]}),
    )
    .unwrap();
    let f = db.ontology_call("facts", json!({"ns": "desk", "s": "ev:t1", "p": "price"})).unwrap();
    assert_eq!(f.as_array().unwrap().len(), 1, "{f}");
    assert_eq!(f[0]["o"], "187.25 USD");
    // the old value is still there in time travel
    db.ontology_call(
        "assert",
        json!({"ns": "desk", "facts": [
            {"s": "gs_bank", "p": "isSubsidiaryOf", "o": "gs"},
            {"s": "gs_uk", "p": "isSubsidiaryOf", "o": "gs_bank"}]}),
    )
    .unwrap();
    let fs = infer(&db, "desk");
    assert!(has(&fs, "gs_uk", "isSubsidiaryOf", "gs"));
    assert!(has(&fs, "gs", "hasSubsidiary", "gs_uk"));
    // export round trip
    let t = db.ontology_call("export", json!({"onto": "acme", "format": "turtle"})).unwrap();
    assert!(t["text"].as_str().unwrap().contains("owl:FunctionalProperty"));
    let j = db.ontology_call("export", json!({"onto": "acme"})).unwrap();
    assert_eq!(j["properties"].as_array().unwrap().len(), 6);
}

#[test]
fn ontologies_are_versioned_with_valid_time() {
    let db = DbInstance::new("mem", "", "").unwrap();
    db.ontology_call(
        "define",
        json!({"onto": "o", "version": "1", "properties": [{"name": "likes", "card": "many"}]}),
    )
    .unwrap();
    let t1 = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros() as i64;
    std::thread::sleep(std::time::Duration::from_millis(5));
    db.ontology_call(
        "define",
        json!({"onto": "o", "version": "2", "replace": true,
               "properties": [{"name": "likes", "card": "one"}, {"name": "owns"}]}),
    )
    .unwrap();
    let now = db.ontology_call("get", json!({"onto": "o"})).unwrap();
    assert_eq!(now["version"], "2");
    assert_eq!(now["properties"].as_array().unwrap().len(), 2);
    let then = db.ontology_call("get", json!({"onto": "o", "at": t1})).unwrap();
    assert_eq!(then["version"], "1");
    let likes: Vec<&Value> = then["properties"].as_array().unwrap().iter().filter(|p| p["name"] == "likes").collect();
    assert_eq!(likes[0]["card"], "many");
    db.ontology_call("define", json!({"onto": "o", "version": "3", "replace": true, "properties": [{"name": "owns"}]})).unwrap();
    let now = db.ontology_call("get", json!({"onto": "o"})).unwrap();
    assert_eq!(now["properties"].as_array().unwrap().len(), 1, "{now}");
}

#[test]
fn script_op_and_type() {
    let db = DbInstance::new("mem", "", "").unwrap();
    let r = db.run_script("::ontology list", Default::default(), ScriptMutability::Mutable).unwrap();
    assert!(!r.rows.is_empty());
    let t = db.ontology_call("type", json!({"text": "Melanie: Wow, fantastic, Caroline!"})).unwrap();
    assert_eq!(t["cls"], "smalltalk");
    let t = db.ontology_call("type", json!({"text": "I moved to Lisbon last month."})).unwrap();
    assert_eq!(t["prop"], "lives_in");
    let e = db.run_script("::ontology activate {ns: 'x', ontos: ['personal']}", Default::default(), ScriptMutability::Immutable);
    assert!(e.is_err());
}
