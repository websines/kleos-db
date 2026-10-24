/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Moving a memory store to another embedder (`::memory reembed`): the
//! store records the model behind its vectors and refuses to mix two; a
//! re-embed runs as a durable workflow, survives its process being killed
//! between batches, catches up memories remembered meanwhile, and switches
//! in one transaction. Fake embedders: two hashed bags of words with
//! different seeds (same width, different vectors).

use std::sync::{Arc, Mutex};

use kleos::embed::{self, Embedder};
use kleos::memory::{InitOpts, RecallOpts, RememberOpts};
use kleos::{DbInstance, ScriptMutability};
use serde_json::json;

static LOCK: Mutex<()> = Mutex::new(());

struct Words {
    seed: u64,
    id: &'static str,
}

impl Embedder for Words {
    fn embed(&self, texts: &[&str]) -> miette::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0f32; 32];
                for w in t.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2) {
                    let h = w.to_lowercase().bytes().fold(self.seed, |h, b| (h ^ b as u64).wrapping_mul(1099511628211));
                    v[(h % 32) as usize] += 1.0;
                }
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                v.iter().map(|x| x / n).collect()
            })
            .collect())
    }
    fn dim(&self) -> usize {
        32
    }
    fn model_id(&self) -> Option<String> {
        Some(self.id.to_string())
    }
}

const A: Words = Words { seed: 1469598103934665603, id: "words-a" };
const B: Words = Words { seed: 7, id: "words-b" };

fn register(name: &str, w: Words) {
    embed::register_embedder(name, Arc::new(w)).unwrap();
}

fn text(i: usize) -> String {
    format!("Memory number {i} mentions the {} and the {} near station {}.", ["river", "garden", "harbour", "market"][i % 4], ["violin", "bicycle", "lantern"][i % 3], i * 7)
}

/// Every stored vector equals `w` applied to its memory's text.
fn assert_vectors(db: &DbInstance, w: &Words) -> usize {
    let rows = db
        .run_script(
            "t[ns, id, choice(text)] := *kleos_mem{ns, id, text} ?[ns, id, text, vec] := t[ns, id, text], *kleos_mem_vec{ns, id, vec}",
            Default::default(),
            ScriptMutability::Immutable,
        )
        .unwrap()
        .rows;
    for r in &rows {
        let text = r[2].get_str().unwrap();
        let want = &w.embed(&[text]).unwrap()[0];
        let got: Vec<f32> = match &r[3] {
            kleos::DataValue::Vec(kleos::Vector::F32(a)) => a.iter().copied().collect(),
            v => panic!("vector expected, got {v:?}"),
        };
        let d: f32 = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).sum();
        assert!(d < 1e-4, "memory {:?} has a vector of another model (L1 {d})", r[1]);
    }
    rows.len()
}

fn config(db: &DbInstance, key: &str) -> Option<String> {
    db.run_script(
        &format!("?[v] := *kleos_mem_config{{key: '{key}', value: v}}"),
        Default::default(),
        ScriptMutability::Immutable,
    )
    .unwrap()
    .rows
    .first()
    .and_then(|r| r[0].get_str().map(|s| s.to_string()))
}

#[test]
fn reembed_moves_every_vector_and_switches_once() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register("re_a", A);
    register("re_b", B);
    let db = DbInstance::new("mem", "", "").unwrap();
    db.memory_init(InitOpts { embedder: Some("re_a".into()), ..Default::default() }).unwrap();
    for i in 0..70 {
        db.remember(&text(i), RememberOpts { ns: Some(format!("n{}", i % 3)), ..Default::default() }).unwrap();
    }
    assert_eq!(config(&db, "embedder_model").as_deref(), Some("words-a"));
    assert_eq!(assert_vectors(&db, &A), 70);
    let r = db.memory_call("reembed", json!({"embedder": "re_b", "batch": 16})).unwrap();
    assert_eq!(r["status"], "completed", "{r}");
    assert_eq!(r["output"], json!(70));
    assert_eq!(config(&db, "embedder").as_deref(), Some("re_b"));
    assert_eq!(config(&db, "embedder_model").as_deref(), Some("words-b"));
    assert_eq!(config(&db, "reembed"), None);
    assert_eq!(assert_vectors(&db, &B), 70);
    // recall and remember use the new model
    let hits = db.recall("harbour lantern", RecallOpts { ns: Some("n2".into()), ..Default::default() }).unwrap();
    assert!(!hits.is_empty());
    db.remember("A new memory about the harbour.", RememberOpts::default()).unwrap();
    assert_eq!(assert_vectors(&db, &B), 71);
    // again: nothing to do
    let r = db.memory_call("reembed", json!({"embedder": "re_b"})).unwrap();
    assert_eq!(r["unchanged"], json!(true), "{r}");
    embed::unregister_embedder("re_a");
    embed::unregister_embedder("re_b");
}

#[test]
fn store_refuses_vectors_of_another_model() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register("re_c", A);
    let db = DbInstance::new("mem", "", "").unwrap();
    db.memory_init(InitOpts { embedder: Some("re_c".into()), ..Default::default() }).unwrap();
    db.remember("I live in Paris.", RememberOpts::default()).unwrap();
    // the same name now points at another model
    embed::unregister_embedder("re_c");
    register("re_c", B);
    let e = db.remember("I live in Lisbon.", RememberOpts::default()).unwrap_err();
    assert!(format!("{e:?}").contains("never mixed"), "{e:?}");
    let e = db.recall("Paris", RecallOpts::default()).unwrap_err();
    assert!(format!("{e:?}").contains("reembed"), "{e:?}");
    // the fix the error names
    let r = db.memory_call("reembed", json!({"embedder": "re_c"})).unwrap();
    assert_eq!(r["status"], "completed", "{r}");
    db.remember("I live in Lisbon.", RememberOpts::default()).unwrap();
    assert_eq!(assert_vectors(&db, &B), 2);
    embed::unregister_embedder("re_c");
}

/// Child process of [`reembed_resumes_after_its_process_dies`]: starts the
/// re-embed, runs two batch steps, then dies without finishing.
#[test]
fn reembed_child() {
    let Ok(path) = std::env::var("KLEOS_REEMBED_DB") else { return };
    register("re_a", A);
    register("re_b", B);
    let db = DbInstance::new("sqlite", &path, "").unwrap();
    let r = db.memory_call("reembed", json!({"embedder": "re_b", "batch": 5, "wait": false})).unwrap();
    assert_eq!(r["status"], "running");
    // a worker with a short lease runs the batches; die once some are done
    std::thread::spawn(move || {
        let _ = db.flow_call("tick", json!({"max": 1, "schedules": false, "lease_ms": 2000}));
    });
    let probe = DbInstance::new("sqlite", &path, "").unwrap();
    let t = std::time::Instant::now();
    loop {
        let n = probe
            .run_script("?[count(id)] := *kleos_mem_vec_next{id}", Default::default(), ScriptMutability::Immutable)
            .ok()
            .and_then(|r| r.rows.first().and_then(|x| x[0].get_int()))
            .unwrap_or(0);
        if n >= 30 || t.elapsed().as_secs() > 60 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    std::process::abort();
}

#[test]
fn reembed_resumes_after_its_process_dies() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let base = std::env::temp_dir().join(format!("kleos-reembed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let path = base.join("mem.db");
    register("re_a", A);
    register("re_b", B);
    {
        let db = DbInstance::new("sqlite", &path, "").unwrap();
        db.memory_init(InitOpts { embedder: Some("re_a".into()), ..Default::default() }).unwrap();
        for i in 0..400 {
            db.remember(&text(i), RememberOpts::default()).unwrap();
        }
    }
    let st = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["reembed_child", "--exact", "--nocapture", "--test-threads", "1"])
        .env("KLEOS_REEMBED_DB", &path)
        .status()
        .unwrap();
    assert!(!st.success(), "the child should have died");
    let db = DbInstance::new("sqlite", &path, "").unwrap();
    // the child died mid-way: the store still serves the old model
    assert_eq!(config(&db, "embedder").as_deref(), Some("re_a"));
    assert!(config(&db, "reembed").is_some());
    let shadow = db
        .run_script("?[count(id)] := *kleos_mem_vec_next{id}", Default::default(), ScriptMutability::Immutable)
        .map(|r| r.rows[0][0].get_int().unwrap_or(0))
        .unwrap_or(0);
    println!("shadow rows after the kill: {shadow} of 400");
    // a memory remembered meanwhile, with the old model
    db.remember("Remembered while the migration was paused.", RememberOpts::default()).unwrap();
    // resume: the worker's lease has to expire first (default lease)
    let r = db
        .memory_call("reembed", json!({"embedder": "re_b", "batch": 25}))
        .unwrap();
    assert_eq!(r["status"], "completed", "{r}");
    assert_eq!(r["output"], json!(401));
    assert_eq!(assert_vectors(&db, &B), 401);
    assert_eq!(config(&db, "embedder_model").as_deref(), Some("words-b"));
    drop(db);
    embed::unregister_embedder("re_a");
    embed::unregister_embedder("re_b");
    let _ = std::fs::remove_dir_all(&base);
}
