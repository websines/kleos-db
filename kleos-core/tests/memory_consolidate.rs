/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Consolidation (docs/specs/consolidation.md): entity profiles, session
//! summaries, rebuild on change, erase with their sources, and the durable
//! workflow. No embedder (word-overlap centrality), no LLM.

use kleos::memory::{ForgetOpts, InitOpts, RecallOpts, RememberOpts};
use kleos::{DbInstance, ScriptMutability};
use serde_json::json;

const DAY: i64 = 86_400_000_000;
const T0: i64 = 1_767_225_600_000_000;

fn db() -> DbInstance {
    let db = DbInstance::new("mem", "", "").unwrap();
    db.memory_init(InitOpts { ontology: Some("personal".into()), ns: Some("u".into()), ..Default::default() })
        .unwrap();
    db
}

fn rem(db: &DbInstance, text: &str, day: i64, source: &str) -> String {
    db.remember(
        text,
        RememberOpts { ns: Some("u".into()), at: Some(T0 + day * DAY), source: Some(source.into()), ..Default::default() },
    )
    .unwrap()
    .id
    .unwrap_or_default()
}

const CHAT: &[&str] = &[
    "Caroline adopted a dog named Max in May.",
    "Caroline works as a counselor at the youth center.",
    "Caroline volunteers at the youth center every weekend.",
    "Caroline is training for a half marathon in October.",
    "Caroline runs ten kilometres every Sunday morning.",
    "Caroline's sister Anna moved to Lisbon last year.",
    "Caroline painted a sunset over the lake for her mother.",
    "Caroline and Melanie went camping near the lake in June.",
    "Max the dog loves swimming in the lake.",
    "Caroline plans to adopt a second dog next year.",
];

fn rows(db: &DbInstance, q: &str) -> usize {
    db.run_script(q, Default::default(), ScriptMutability::Immutable).unwrap().rows.len()
}

#[test]
fn summaries_built_rebuilt_and_erased() {
    let db = db();
    let ids: Vec<String> = CHAT.iter().enumerate().map(|(i, t)| rem(&db, t, i as i64, &format!("chat-1#{i}"))).collect();
    rem(&db, "I like tea.", 20, "chat-2#0");
    let r = db.memory_call("consolidate", json!({"ns": "u", "min_claims": 8})).unwrap();
    assert_eq!(r["summaries"]["built"], json!(1), "{r}");
    assert_eq!(r["summaries"]["groups"], json!(1), "{r}");
    assert!(r["summaries"]["summary_tokens"].as_u64().unwrap() < r["summaries"]["source_tokens"].as_u64().unwrap(), "{r}");
    // the summary is a memory of kind `summary`, role `other`
    let hits = db
        .recall("Summary of chat-1", RecallOpts { ns: Some("u".into()), kind: Some("summary".into()), ..Default::default() })
        .unwrap();
    assert_eq!(hits.len(), 1, "{hits:?}");
    let sum = hits[0].clone();
    assert!(sum.text.starts_with("Summary of chat-1 ("), "{}", sum.text);
    assert_eq!(sum.role.as_deref(), Some("other"));
    // not a hit by default; the context block shows it for the conversation
    let plain = db.recall("Caroline dog", RecallOpts { ns: Some("u".into()), ..Default::default() }).unwrap();
    assert!(!plain.iter().any(|h| h.kind == "summary"), "{plain:?}");
    let ctx = db.recall_context("Caroline dog", RecallOpts { ns: Some("u".into()), ..Default::default() }).unwrap();
    assert!(ctx.text.contains("Conversation summaries") && ctx.text.contains("Summary of chat-1"), "{}", ctx.text);
    let off = db
        .recall_context("Caroline dog", RecallOpts { ns: Some("u".into()), summaries: Some(false), ..Default::default() })
        .unwrap();
    assert!(!off.text.contains("Summary of"), "{}", off.text);
    // unchanged sources: nothing to do
    let r = db.memory_call("consolidate", json!({"ns": "u", "min_claims": 8})).unwrap();
    assert_eq!(r["summaries"]["unchanged"], json!(1), "{r}");
    // a new source: rebuilt
    rem(&db, "Caroline bought a red kayak for the lake.", 12, "chat-1#10");
    let r = db.memory_call("consolidate", json!({"ns": "u", "min_claims": 8})).unwrap();
    assert_eq!(r["summaries"]["built"], json!(1), "{r}");
    // erasing a source erases the summary built from it
    db.forget(ForgetOpts { ns: Some("u".into()), id: Some(ids[0].clone()), mode: Some("erase".into()), ..Default::default() }).unwrap();
    assert_eq!(rows(&db, &format!("?[id] := *kleos_mem{{ns: 'u', id}}, id = '{}'", sum.id)), 0);
    assert_eq!(rows(&db, "?[sid] := *kleos_mem_summary{ns: 'u', sid}"), 0);
}

#[test]
fn profiles_in_the_context_block() {
    let db = db();
    rem(&db, "I live in Paris.", 1, "s#1");
    rem(&db, "My sister Anna lives in Munich.", 2, "s#2");
    rem(&db, "I work at Siemens as an engineer.", 3, "s#3");
    let r = db.memory_call("consolidate", json!({"ns": "u", "summaries": false})).unwrap();
    assert!(r["profiles"]["built"].as_u64().unwrap() >= 1, "{r}");
    let t = db
        .run_script("?[node, text] := *kleos_mem_profile{ns: 'u', node, text}", Default::default(), ScriptMutability::Immutable)
        .unwrap();
    let user = t.rows.iter().find(|r| r[0].get_str() == Some("user")).unwrap_or_else(|| panic!("{:?}", t.rows));
    let text = user[1].get_str().unwrap();
    assert!(text.starts_with("Profile: User — "), "{text}");
    assert!(text.contains("lives_in: I live in Paris"), "{text}");
    assert!(text.contains("← sister_of: Anna"), "{text}");
    let ctx = db.recall_context("Where do I live?", RecallOpts { ns: Some("u".into()), ..Default::default() }).unwrap();
    assert!(ctx.text.starts_with("Profiles (current facts"), "{}", ctx.text);
    assert!(!ctx.profiles.is_empty());
    let off = db
        .recall_context("Where do I live?", RecallOpts { ns: Some("u".into()), profiles: Some(false), ..Default::default() })
        .unwrap();
    assert!(!off.text.contains("Profiles"), "{}", off.text);
    // a moved user: the profile follows on the next run
    rem(&db, "I live in Lisbon now.", 5, "s#4");
    db.memory_call("consolidate", json!({"ns": "u", "summaries": false})).unwrap();
    let t = db
        .run_script("?[text] := *kleos_mem_profile{ns: 'u', node: 'user', text}", Default::default(), ScriptMutability::Immutable)
        .unwrap();
    let text = t.rows[0][0].get_str().unwrap();
    assert!(text.contains("Lisbon"), "{text}");
}

#[test]
fn consolidation_runs_as_a_workflow() {
    let db = db();
    let r = db.memory_call("consolidate", json!({"ns": "u", "every": 5, "min_claims": 8})).unwrap();
    assert_eq!(r["auto"]["every"], json!(5), "{r}");
    for (i, t) in CHAT.iter().enumerate() {
        rem(&db, t, i as i64, &format!("chat-9#{i}"));
    }
    // every 5 new memories started a run; a worker runs it
    for _ in 0..4 {
        let _ = db.flow_call("tick", json!({"max": 4, "schedules": false, "lease_ms": 2000}));
    }
    let l = db.flow_call("list", json!({"workflow": "memory.consolidate"})).unwrap();
    assert!(l.as_array().is_some_and(|a| !a.is_empty()), "{l}");
    assert_eq!(rows(&db, "?[sid] := *kleos_mem_summary{ns: 'u', sid}"), 1);
}
