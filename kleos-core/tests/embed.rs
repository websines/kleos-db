/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! embed(), the embedding cache and computed (auto-embed) columns
//! (UPGRADE_PLAN D3). The embedder here is a deterministic toy, so these tests
//! need no model. The registry is process-wide: every test uses its own names.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use kleos::embed::{embed_texts, embedding_cache_stats, register_embedder, Embedder};
use kleos::{DataValue, DbInstance, NamedRows, ScriptMutability, Vector};

/// Bag-of-letters embedder: dimension `dim`, counts letters a..z modulo dim,
/// L2-normalised. Counts model calls and texts seen.
struct Letters {
    dim: usize,
    calls: AtomicUsize,
    texts: AtomicUsize,
    max_batch: usize,
}

impl Letters {
    fn new(dim: usize, max_batch: usize) -> Arc<Self> {
        Arc::new(Self {
            dim,
            calls: AtomicUsize::new(0),
            texts: AtomicUsize::new(0),
            max_batch,
        })
    }
    fn vector(&self, t: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        for c in t.chars().filter(|c| c.is_ascii_alphabetic()) {
            v[(c.to_ascii_lowercase() as usize - 'a' as usize) % self.dim] += 1.0;
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            v.iter_mut().for_each(|x| *x /= n);
        }
        v
    }
}

impl Embedder for Letters {
    fn embed(&self, texts: &[&str]) -> miette::Result<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.texts.fetch_add(texts.len(), Ordering::SeqCst);
        assert!(texts.len() <= self.max_batch, "batch above max_batch");
        Ok(texts.iter().map(|t| self.vector(t)).collect())
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn max_batch(&self) -> usize {
        self.max_batch
    }
}

fn run(db: &DbInstance, script: &str) -> NamedRows {
    db.run_script(script, Default::default(), ScriptMutability::Mutable)
        .unwrap_or_else(|e| panic!("script failed: {e:?}\n--- script ---\n{script}"))
}

fn as_vec(v: &DataValue) -> Vec<f32> {
    match v {
        DataValue::Vec(Vector::F32(a)) => a.to_vec(),
        other => panic!("not an F32 vector: {other:?}"),
    }
}

#[test]
fn embed_function_returns_vectors() {
    let m = Letters::new(8, 32);
    register_embedder("t_fn", m.clone()).unwrap();
    let db = DbInstance::new("mem", "", "").unwrap();
    let rows = run(
        &db,
        "?[v, n, short] := v = embed('t_fn', 'abc'), n = embed('t_fn', null), \
         short = embed('t_fn', 'abcd', 2)",
    );
    assert_eq!(as_vec(&rows.rows[0][0]), m.vector("abc"));
    assert_eq!(rows.rows[0][1], DataValue::Null);
    let short = as_vec(&rows.rows[0][2]);
    assert_eq!(short.len(), 2);
    let norm: f32 = short.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "Matryoshka prefix is re-normalised");

    let err = db
        .run_default("?[v] := v = embed('no_such_model', 'x')")
        .unwrap_err();
    assert!(format!("{err:?}").contains("no_such_model"));
    let err = db.run_default("?[v] := v = embed('t_fn', 'x', 99)").unwrap_err();
    assert!(format!("{err:?}").contains("truncate"));
}

#[test]
fn cache_avoids_recomputing_and_dedups_within_a_batch() {
    let m = Letters::new(4, 32);
    register_embedder("t_cache", m.clone()).unwrap();
    let before = embedding_cache_stats();
    let out = embed_texts("t_cache", &["one", "two", "one"]).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out[0], out[2]);
    assert_eq!(m.texts.load(Ordering::SeqCst), 2, "duplicate text embedded once");
    embed_texts("t_cache", &["two", "one"]).unwrap();
    assert_eq!(m.texts.load(Ordering::SeqCst), 2, "second call served from cache");
    let after = embedding_cache_stats();
    assert!(after.hits >= before.hits + 2);

    // re-registering the name must not serve the old model's vectors
    let m2 = Letters::new(4, 32);
    register_embedder("t_cache", m2.clone()).unwrap();
    embed_texts("t_cache", &["one"]).unwrap();
    assert_eq!(m2.texts.load(Ordering::SeqCst), 1);
}

#[test]
fn computed_column_embeds_on_put_in_batches() {
    let m = Letters::new(8, 16);
    register_embedder("t_put", m.clone()).unwrap();
    let db = DbInstance::new("mem", "", "").unwrap();
    run(
        &db,
        ":create docs { id: Int => text: String, vec: <F32; 8> default embed('t_put', text) }",
    );
    let rows: Vec<String> = (0..100).map(|i| format!("[{i}, 'doc number {i} {}']", "x".repeat(i % 7))).collect();
    run(
        &db,
        &format!("?[id, text] <- [{}] :put docs {{ id => text }}", rows.join(",")),
    );
    // 100 distinct texts, max_batch 16 -> 7 model calls, not 100
    assert_eq!(m.texts.load(Ordering::SeqCst), 100);
    assert_eq!(m.calls.load(Ordering::SeqCst), 7);
    let got = run(&db, "?[text, vec] := *docs{id: 42, text, vec}");
    let text = got.rows[0][0].get_str().unwrap().to_string();
    assert_eq!(as_vec(&got.rows[0][1]), m.vector(&text));

    // an explicit vector wins over the computed default
    run(
        &db,
        "?[id, text, vec] <- [[1000, 'zzz', vec([1,0,0,0,0,0,0,0])]] :put docs { id => text, vec }",
    );
    let got = run(&db, "?[vec] := *docs{id: 1000, vec}");
    assert_eq!(as_vec(&got.rows[0][0])[0], 1.0);
}

#[test]
fn computed_column_follows_updates() {
    let m = Letters::new(8, 32);
    register_embedder("t_upd", m.clone()).unwrap();
    let db = DbInstance::new("mem", "", "").unwrap();
    run(
        &db,
        ":create notes { id: Int => text: String, tag: String default 'none', \
         vec: <F32; 8> default embed('t_upd', text) }",
    );
    run(&db, "?[id, text] <- [[1, 'aaaa']] :put notes { id => text }");
    run(&db, "?[id, text] <- [[1, 'bbbb']] :update notes { id => text }");
    let got = run(&db, "?[vec] := *notes{id: 1, vec}");
    assert_eq!(as_vec(&got.rows[0][0]), m.vector("bbbb"), "recomputed on :update");

    // an update that does not touch the source keeps the stored vector
    run(&db, "?[id, tag] <- [[1, 'x']] :update notes { id => tag }");
    let got = run(&db, "?[vec] := *notes{id: 1, vec}");
    assert_eq!(as_vec(&got.rows[0][0]), m.vector("bbbb"));
}

#[test]
fn computed_column_with_two_sources_rejects_partial_update() {
    let m = Letters::new(8, 32);
    register_embedder("t_two", m).unwrap();
    let db = DbInstance::new("mem", "", "").unwrap();
    run(
        &db,
        ":create items { id: Int => title: String, body: String, \
         vec: <F32; 8> default embed('t_two', concat(title, ' ', body)) }",
    );
    run(
        &db,
        "?[id, title, body] <- [[1, 'ab', 'cd']] :put items { id => title, body }",
    );
    let err = db
        .run_default("?[id, title] <- [[1, 'zz']] :update items { id => title }")
        .unwrap_err();
    assert!(format!("{err:?}").contains("body"), "{err:?}");
    let err = db
        .run_default("?[id, title] <- [[2, 'zz']] :put items { id => title }")
        .unwrap_err();
    assert!(format!("{err:?}").contains("body") || format!("{err:?}").contains("required"));
}

#[test]
fn hnsw_over_a_computed_column() {
    let m = Letters::new(8, 32);
    register_embedder("t_hnsw", m).unwrap();
    let db = DbInstance::new("mem", "", "").unwrap();
    run(
        &db,
        ":create mem { id: Int => text: String, vec: <F32; 8> default embed('t_hnsw', text) }",
    );
    run(
        &db,
        "::hnsw create mem:v { dim: 8, m: 16, dtype: F32, fields: [vec], distance: Cosine, ef_construction: 32 }",
    );
    run(
        &db,
        "?[id, text] <- [[1, 'aaaa'], [2, 'bbbb'], [3, 'cccc'], [4, 'abab']] :put mem { id => text }",
    );
    let got = run(
        &db,
        "?[id, d] := ~mem:v{ id | query: embed('t_hnsw', 'bbbbb'), k: 1, ef: 16, bind_distance: d }",
    );
    assert_eq!(got.rows[0][0].get_int(), Some(2));
}
