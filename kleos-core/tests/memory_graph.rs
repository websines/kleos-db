/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Graph recall (docs/specs/memory-api.md, "Graph recall"): memories reached
//! through inferred ontology facts, the personalized-PageRank re-rank, and
//! unit conversion of typed literals. No embedder: text search seeds the
//! graph.

use kleos::memory::{InitOpts, RecallOpts, RememberOpts};
use kleos::DbInstance;
use serde_json::json;

const DAY: i64 = 86_400_000_000;
const T0: i64 = 1_767_225_600_000_000;

fn db() -> DbInstance {
    let db = DbInstance::new("mem", "", "").unwrap();
    db.memory_init(InitOpts { ontology: Some("personal".into()), ns: Some("u".into()), ..Default::default() })
        .unwrap();
    db
}

fn rem(db: &DbInstance, text: &str, day: i64, ents: &[&str]) -> String {
    db.remember(
        text,
        RememberOpts {
            ns: Some("u".into()),
            at: Some(T0 + day * DAY),
            entities: ents.iter().map(|x| x.to_string()).collect(),
            ..Default::default()
        },
    )
    .unwrap()
    .id
    .unwrap()
}

fn recall(db: &DbInstance, q: &str, graph: &str) -> Vec<kleos::memory::RecallHit> {
    db.recall(
        q,
        RecallOpts {
            ns: Some("u".into()),
            k: Some(10),
            graph: Some(graph.into()),
            record_access: Some(false),
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn inferred_links_reach_relatives() {
    let db = db();
    rem(&db, "My sister Anna lives in Munich.", 1, &[]);
    rem(&db, "My brother Tom works at Siemens.", 2, &[]);
    let hike = rem(&db, "Anna loves hiking in the Alps.", 3, &[]);
    rem(&db, "I bought a new bike last week.", 4, &[]);
    // plain recall: "Anna loves hiking" shares no word with the question
    let plain = recall(&db, "Who are my relatives?", "off");
    assert!(!plain.iter().any(|h| h.id == hike), "{plain:?}");
    let hits = recall(&db, "Who are my relatives?", "inferred");
    let h = hits.iter().find(|h| h.id == hike).unwrap_or_else(|| panic!("{hits:?}"));
    assert_eq!(h.via.as_deref(), Some("inferred: relative_of"), "{h:?}");
    assert!(h.expanded);
    // the context block says why
    let ctx = db
        .recall_context(
            "Who are my relatives?",
            RecallOpts { ns: Some("u".into()), graph: Some("inferred".into()), record_access: Some(false), ..Default::default() },
        )
        .unwrap();
    assert!(ctx.text.contains("(via inferred: relative_of)"), "{}", ctx.text);
}

#[test]
fn graph_option_is_checked() {
    let db = db();
    rem(&db, "I live in Paris.", 1, &[]);
    let e = db
        .recall("Paris", RecallOpts { ns: Some("u".into()), graph: Some("walk".into()), ..Default::default() })
        .unwrap_err();
    assert!(format!("{e:?}").contains("graph must be"), "{e:?}");
}

#[test]
fn ppr_reaches_through_shared_entities() {
    let db = db();
    let a = rem(&db, "Anna adopted a puppy last spring.", 1, &["Anna", "Max"]);
    let b = rem(&db, "Max loves running on the beach.", 2, &["Max"]);
    rem(&db, "The weather was nice today.", 3, &[]);
    let plain = recall(&db, "Anna puppy", "off");
    assert!(plain.iter().any(|h| h.id == a));
    assert!(!plain.iter().any(|h| h.id == b), "{plain:?}");
    let hits = recall(&db, "Anna puppy", "ppr");
    assert_eq!(hits[0].id, a, "{hits:?}");
    let hb = hits.iter().find(|h| h.id == b).unwrap_or_else(|| panic!("{hits:?}"));
    assert_eq!(hb.via.as_deref(), Some("graph"));
    // the unrelated memory stays out
    assert!(!hits.iter().any(|h| h.text.contains("weather")), "{hits:?}");
}

#[test]
fn units_convert_validate_and_restate() {
    let db = DbInstance::new("mem", "", "").unwrap();
    db.ontology_call(
        "define",
        json!({"onto": "trip", "classes": [{"name": "Person"}],
               "properties": [{"name": "commute_distance", "kind": "data", "card": "one", "unit": "km"}]}),
    )
    .unwrap();
    db.ontology_call("activate", json!({"ns": "t", "ontos": ["trip"]})).unwrap();
    let assert = |v: f64, u: &str| {
        db.ontology_call("assert", json!({"ns": "t", "facts": [{"s": "user", "p": "commute_distance", "lit": {"value": v, "unit": u}}]}))
            .unwrap()
    };
    let r = assert(5.0, "km");
    assert_eq!(r["violations"].as_array().unwrap().len(), 0);
    // the same distance in metres: not a change
    let r = assert(5000.0, "m");
    assert_eq!(r["ended"], json!(0), "{r}");
    let cur = db.ontology_call("facts", json!({"ns": "t", "p": "commute_distance"})).unwrap();
    assert_eq!(cur.as_array().unwrap().len(), 1, "{cur}");
    // miles convert, kilograms do not
    let r = assert(4.0, "mi");
    assert_eq!(r["violations"].as_array().unwrap().len(), 0, "{r}");
    assert_eq!(r["ended"], json!(1), "{r}");
    let r = assert(3.0, "kg");
    assert_eq!(r["violations"][0]["rule"], json!("unit"), "{r}");
    let c = db.ontology_call("convert", json!({"value": 10, "from": "km", "to": "mi"})).unwrap();
    assert!((c["value"].as_f64().unwrap() - 6.2137).abs() < 1e-3, "{c}");
    assert!(db.ontology_call("convert", json!({"value": 10, "from": "EUR", "to": "USD"})).is_err());
}
