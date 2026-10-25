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
    // every 5 new memories started a run; the store's in-process runner (or
    // this explicit tick, whichever claims it first) runs it
    let mut n = 0;
    for _ in 0..100 {
        let _ = db.flow_call("tick", json!({"max": 4, "schedules": false, "lease_ms": 2000}));
        n = rows(&db, "?[sid] := *kleos_mem_summary{ns: 'u', sid}");
        if n == 1 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let l = db.flow_call("list", json!({"workflow": "memory.consolidate"})).unwrap();
    assert!(l.as_array().is_some_and(|a| !a.is_empty()), "{l}");
    assert_eq!(n, 1);
}

/// Automatic consolidation runs in-process with no worker started by the
/// caller: after `every` new memories the store builds profiles on its own,
/// and the context block shows them.
#[test]
fn auto_consolidation_runs_in_process() {
    let db = db();
    db.memory_call("consolidate", json!({"ns": "u", "every": 3, "summaries": false})).unwrap();
    rem(&db, "I live in Paris.", 1, "a#1");
    rem(&db, "My sister Anna lives in Munich.", 2, "a#2");
    rem(&db, "I work at Siemens as an engineer.", 3, "a#3");
    let q = "?[node] := *kleos_mem_profile{ns: 'u', node}, node = 'user'";
    let mut built = false;
    for _ in 0..100 {
        if rows(&db, q) > 0 {
            built = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert!(built, "no profile built by the in-process consolidation runner");
    let ctx = db.recall_context("Where do I live?", RecallOpts { ns: Some("u".into()), ..Default::default() }).unwrap();
    assert!(ctx.text.starts_with("Profiles (current facts"), "{}", ctx.text);
}

#[test]
fn entity_profiles_from_extractor_entities() {
    let db = db();
    let ent = |text: &str, day: i64, src: &str, ents: &[&str]| {
        db.remember(
            text,
            RememberOpts {
                ns: Some("u".into()),
                at: Some(T0 + day * DAY),
                source: Some(src.into()),
                entities: ents.iter().map(|e| e.to_string()).collect(),
                ..Default::default()
            },
        )
        .unwrap();
    };
    ent("Max the dog loves swimming in the lake.", 1, "c#1", &["Max"]);
    ent("We took Max to the vet on Tuesday.", 2, "c#2", &["Max"]);
    ent("Max learned to fetch a frisbee.", 3, "c#3", &["Max"]);
    ent("Bella came over once.", 4, "c#4", &["Bella"]);
    let r = db.memory_call("consolidate", json!({"ns": "u", "summaries": false})).unwrap();
    assert!(r["profiles"]["nodes"].as_u64().unwrap() >= 1, "{r}");
    let t = db
        .run_script("?[node, text] := *kleos_mem_profile{ns: 'u', node, text}", Default::default(), ScriptMutability::Immutable)
        .unwrap();
    let max = t.rows.iter().find(|r| r[0].get_str() == Some("max")).unwrap_or_else(|| panic!("{:?}", t.rows));
    let text = max[1].get_str().unwrap();
    assert!(text.starts_with("Profile: Max"), "{text}");
    assert!(text.contains("frisbee"), "newest first: {text}");
    // one mention is not enough for a profile
    assert!(!t.rows.iter().any(|r| r[0].get_str() == Some("bella")), "{:?}", t.rows);
    // the profile reaches the context block of a hit that mentions Max
    let ctx = db.recall_context("What does Max like?", RecallOpts { ns: Some("u".into()), ..Default::default() }).unwrap();
    assert!(ctx.text.contains("Profile: Max"), "{}", ctx.text);
}

#[test]
fn summaries_leave_out_replaced_values() {
    let db = db();
    let lines = [
        "I work as a nurse at the city hospital.",
        "I volunteer at the animal shelter on weekends.",
        "I am training for the city marathon this autumn.",
        "I bake sourdough bread every Sunday morning.",
        "I started learning Portuguese with an online tutor.",
        "I repainted the kitchen walls a pale green colour.",
        "I planted tomatoes and basil on the balcony garden.",
        "I adopted a grey cat from the animal shelter.",
        "I work as a teacher at the primary school now.",
    ];
    let ids: Vec<String> = lines.iter().enumerate().map(|(i, t)| rem(&db, t, i as i64, &format!("chat-9#{i}"))).collect();
    // the newest claim may replace the first (a pending supersession, as a
    // decider in the uncertain band leaves it)
    db.run_script(
        "?[ns, new, cand, template, label, p, band, decided_by, reason, status, at] <- \
         [['u', $new, $old, 'supersedes', 'yes', 0.7, 'pending', 'test', null, 'pending', 0]] \
         :put kleos_mem_decision {ns, new, cand, template => label, p, band, decided_by, reason, status, at}",
        [("new".to_string(), kleos::DataValue::from(ids[8].as_str())), ("old".to_string(), kleos::DataValue::from(ids[0].as_str()))].into_iter().collect(),
        ScriptMutability::Mutable,
    )
    .unwrap();
    let r = db.memory_call("consolidate", json!({"ns": "u", "min_claims": 8, "max_summary": 8, "profiles": false})).unwrap();
    assert!(r["summaries"]["replaced_dropped"].as_u64().unwrap() >= 1, "{r}");
    let t = db
        .run_script("?[text] := *kleos_mem{ns: 'u', kind: 'summary', text @ 'NOW'}", Default::default(), ScriptMutability::Immutable)
        .unwrap();
    assert_eq!(t.rows.len(), 1, "{r}");
    for row in &t.rows {
        let s = row[0].get_str().unwrap();
        assert!(!s.contains("nurse"), "a replaced value in a summary: {s}");
    }
}

/// The abstractive (LLM) summary path against a live OpenAI-compatible
/// endpoint. Ignored by default; run with KLEOS_TEST_LLM_URL,
/// KLEOS_TEST_LLM_MODEL and the key in the variable KLEOS_TEST_LLM_KEY_ENV
/// names (the key itself is never printed or stored).
#[test]
#[ignore]
fn llm_summaries_live() {
    let (Ok(url), Ok(model), Ok(key_env)) =
        (std::env::var("KLEOS_TEST_LLM_URL"), std::env::var("KLEOS_TEST_LLM_MODEL"), std::env::var("KLEOS_TEST_LLM_KEY_ENV"))
    else {
        eprintln!("skipped: KLEOS_TEST_LLM_URL / _MODEL / _KEY_ENV not set");
        return;
    };
    let db = db();
    for (i, t) in CHAT.iter().enumerate() {
        rem(&db, t, i as i64, &format!("chat-1#{i}"));
    }
    let r = db
        .memory_call("consolidate", json!({"ns": "u", "min_claims": 8, "profiles": false, "llm": {"url": url, "model": model, "api_key_env": key_env, "params": {"reasoning_effort": "none"}}}))
        .unwrap();
    let s = &r["summaries"];
    assert_eq!(s["llm_calls"], json!(1), "{r}");
    assert_eq!(s["llm_errors"], json!(0), "{r}");
    assert!(s["method"].as_str().unwrap().starts_with("llm:"), "{r}");
    let t = db
        .run_script("?[text] := *kleos_mem{ns: 'u', kind: 'summary', text @ 'NOW'}", Default::default(), ScriptMutability::Immutable)
        .unwrap();
    let text = t.rows[0][0].get_str().unwrap().to_string();
    eprintln!("summary: {text}");
    assert!(text.starts_with("Summary of chat-1"), "{text}");
    assert!(!text.contains("<think>") && text.contains("Caroline"), "{text}");
}
