/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Runtime on/off switch for embedders: `::embedder list | enable | disable |
//! backfill`, `embed_or_null` and the Rust API (docs/specs/embedder-toggle.md).
//! The non-`infer` tests use a deterministic toy embedder registered from
//! Rust, then switch it with the same CozoScript ops a user would. The
//! registry is process-wide: every test uses its own names.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use kleos::embed::{disable_embedder, embedders, enable_embedder, register_embedder, Embedder};
use kleos::{DataValue, DbInstance, EmbedderSpec, NamedRows, ScriptMutability, Vector};

struct Letters {
    dim: usize,
    calls: AtomicUsize,
    texts: AtomicUsize,
    max_batch: usize,
    unloads: AtomicUsize,
}

impl Letters {
    fn new(dim: usize, max_batch: usize) -> Arc<Self> {
        Arc::new(Self {
            dim,
            calls: AtomicUsize::new(0),
            texts: AtomicUsize::new(0),
            max_batch,
            unloads: AtomicUsize::new(0),
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
    fn unload(&self) {
        self.unloads.fetch_add(1, Ordering::SeqCst);
    }
}

fn run(db: &DbInstance, script: &str) -> NamedRows {
    db.run_script(script, Default::default(), ScriptMutability::Mutable)
        .unwrap_or_else(|e| panic!("script failed: {e:?}\n--- script ---\n{script}"))
}

fn err(db: &DbInstance, script: &str) -> String {
    match db.run_default(script) {
        Ok(r) => panic!("expected an error from {script}, got {:?}", r.rows),
        // miette wraps long help lines; compare on single spaces
        Err(e) => format!("{e:?}").split_whitespace().collect::<Vec<_>>().join(" "),
    }
}

fn as_vec(v: &DataValue) -> Vec<f32> {
    match v {
        DataValue::Vec(Vector::F32(a)) => a.to_vec(),
        other => panic!("not an F32 vector: {other:?}"),
    }
}

/// `::embedder list` row of `name` as (kind, enabled, loaded).
fn listed(db: &DbInstance, name: &str) -> Option<(String, bool, DataValue)> {
    let rows = run(db, "::embedder list");
    assert_eq!(
        rows.headers,
        ["name", "kind", "model", "dim", "enabled", "loaded", "precision", "device"]
    );
    rows.rows.iter().find(|r| r[0].get_str() == Some(name)).map(|r| {
        (
            r[1].get_str().unwrap().to_string(),
            r[4].get_bool().unwrap(),
            r[5].clone(),
        )
    })
}

#[test]
fn list_disable_and_embed_errors_when_off() {
    let m = Letters::new(8, 32);
    register_embedder("tg_list", m.clone()).unwrap();
    let db = DbInstance::new("mem", "", "").unwrap();

    let (kind, enabled, _) = listed(&db, "tg_list").unwrap();
    assert_eq!((kind.as_str(), enabled), ("custom", true));
    let before = run(&db, "?[v] := v = embed('tg_list', 'hello world')");

    let out = run(&db, "::embedder disable tg_list");
    assert_eq!(out.rows[0][0].get_str(), Some("OK"));
    assert_eq!(m.unloads.load(Ordering::SeqCst), 1, "disable calls unload()");
    let (_, enabled, _) = listed(&db, "tg_list").unwrap();
    assert!(!enabled, "a disabled embedder stays listed, as disabled");
    assert!(embedders().iter().any(|e| e.name == "tg_list" && !e.enabled));

    let e = err(&db, "?[v] := v = embed('tg_list', 'hello world')");
    assert!(e.contains("tg_list") && e.contains("disabled"), "{e}");
    assert!(e.contains("::embedder enable tg_list"), "names the fix: {e}");
    // embed_or_null is the lenient twin
    let r = run(&db, "?[v] := v = embed_or_null('tg_list', 'hello world')");
    assert_eq!(r.rows[0][0], DataValue::Null);

    let out = run(&db, "::embedder disable tg_list");
    assert_eq!(out.rows[0][0].get_str(), Some("ALREADY_DISABLED"));
    let e = err(&db, "::embedder disable tg_never_registered");
    assert!(e.contains("tg_never_registered"), "{e}");
    let e = err(&db, "?[v] := v = embed('tg_never_registered', 'x')");
    assert!(e.contains("no embedder is enabled") && e.contains("::embedder enable"), "{e}");

    // read-only scripts may list but not switch
    let ro = db
        .run_script("::embedder enable tg_list", Default::default(), ScriptMutability::Immutable)
        .unwrap_err();
    assert!(format!("{ro:?}").contains("read-only"));
    db.run_script("::embedder list", Default::default(), ScriptMutability::Immutable)
        .unwrap();

    // re-enable with no options: the same embedder comes back
    run(&db, "::embedder enable tg_list");
    let (_, enabled, _) = listed(&db, "tg_list").unwrap();
    assert!(enabled);
    let after = run(&db, "?[v] := v = embed('tg_list', 'hello world')");
    assert_eq!(as_vec(&before.rows[0][0]), as_vec(&after.rows[0][0]));
    // the cache was dropped on disable, so this ran the model again
    assert_eq!(m.texts.load(Ordering::SeqCst), 2);
}

#[test]
fn rust_api_mirrors_the_ops() {
    let m = Letters::new(4, 32);
    register_embedder("tg_api", m.clone()).unwrap();
    let db = DbInstance::new("mem", "", "").unwrap();
    assert!(db.embedders().iter().any(|e| e.name == "tg_api" && e.enabled && e.dim == 4));
    assert!(db.disable_embedder("tg_api").unwrap());
    assert!(!disable_embedder("tg_api").unwrap());
    assert!(db.embedders().iter().any(|e| e.name == "tg_api" && !e.enabled));
    db.enable_embedder("tg_api", EmbedderSpec::default()).unwrap();
    assert!(kleos::embed::is_embedder_enabled("tg_api"));
    // enabling an already-enabled custom embedder with no spec is a no-op
    enable_embedder("tg_api", EmbedderSpec::default()).unwrap();
    assert!(kleos::embed::is_embedder_enabled("tg_api"));
}

#[test]
fn cli_embedder_argument() {
    let (n, s) = EmbedderSpec::parse_cli("gemma=embeddinggemma:int8").unwrap();
    assert_eq!(n, "gemma");
    assert_eq!(s.model.as_deref(), Some("embeddinggemma"));
    assert_eq!(s.precision.as_deref(), Some("int8"));
    assert_eq!((s.device, s.dim), (None, None));
    let (_, s) = EmbedderSpec::parse_cli("g=embeddinggemma::metal:256").unwrap();
    assert_eq!((s.precision, s.device.as_deref(), s.dim), (None, Some("metal"), Some(256)));
    let (_, s) = EmbedderSpec::parse_cli("p=minishlab/potion-base-8M").unwrap();
    assert_eq!(s, EmbedderSpec::model("minishlab/potion-base-8M"));
    assert!(EmbedderSpec::parse_cli("noequals").is_err());
    assert!(EmbedderSpec::parse_cli("g=a:b:c:notanumber").is_err());
}

#[test]
fn strict_put_fails_while_off() {
    let m = Letters::new(8, 32);
    register_embedder("tg_strict", m).unwrap();
    let db = DbInstance::new("mem", "", "").unwrap();
    run(
        &db,
        ":create s_docs { id: Int => text: String, vec: <F32; 8> default embed('tg_strict', text) }",
    );
    run(&db, "?[id, text] <- [[1, 'abc']] :put s_docs { id => text }");
    run(&db, "::embedder disable tg_strict");
    let e = err(&db, "?[id, text] <- [[2, 'def']] :put s_docs { id => text }");
    assert!(e.contains("tg_strict") && e.contains("disabled"), "{e}");
    // nothing was written
    let n = run(&db, "?[count(id)] := *s_docs{id}");
    assert_eq!(n.rows[0][0].get_int(), Some(1));
    // an explicit vector still works: only the computed default needs the model
    run(
        &db,
        "?[id, text, vec] <- [[3, 'x', vec([1,0,0,0,0,0,0,0])]] :put s_docs { id => text, vec }",
    );
    run(&db, "::embedder enable tg_strict");
    run(&db, "?[id, text] <- [[2, 'def']] :put s_docs { id => text }");
}

#[test]
fn lenient_null_then_backfill_and_hnsw() {
    let m = Letters::new(8, 16);
    register_embedder("tg_len", m.clone()).unwrap();
    let db = DbInstance::new("mem", "", "").unwrap();
    run(
        &db,
        ":create l_docs { id: Int => text: String?, tag: String default 'x', \
         vec: <F32; 8>? default embed_or_null('tg_len', text) }",
    );
    run(
        &db,
        "::hnsw create l_docs:v { dim: 8, m: 16, dtype: F32, fields: [vec], distance: Cosine, ef_construction: 32 }",
    );
    // on: embed_or_null behaves like embed
    run(&db, "?[id, text] <- [[1, 'aaaa']] :put l_docs { id => text }");
    let got = run(&db, "?[vec] := *l_docs{id: 1, vec}");
    assert_eq!(as_vec(&got.rows[0][0]), m.vector("aaaa"));

    run(&db, "::embedder disable tg_len");
    let texts = ["bbbb", "cccc", "abab", "dddd zzz", "bbbbbbbb c"];
    let mut rows: Vec<String> = texts
        .iter()
        .enumerate()
        .map(|(i, t)| format!("[{}, '{t}']", i + 2))
        .collect();
    rows.push("[100, null]".to_string());
    for i in 0..40 {
        rows.push(format!("[{}, 'filler {i} {}']", 200 + i, "q".repeat(i % 5)));
    }
    run(
        &db,
        &format!("?[id, text] <- [{}] :put l_docs {{ id => text }}", rows.join(", ")),
    );
    let nulls = run(&db, "?[count(id)] := *l_docs{id, vec}, is_null(vec)");
    assert_eq!(nulls.rows[0][0].get_int(), Some(46), "rows were written with null vectors");

    // backfill refuses while the embedder is still off
    let e = err(&db, "::embedder backfill l_docs");
    assert!(e.contains("tg_len") && e.contains("disabled"), "{e}");

    run(&db, "::embedder enable tg_len");
    let calls_before = m.calls.load(Ordering::SeqCst);
    let out = run(&db, "::embedder backfill l_docs");
    assert_eq!(out.headers, ["column", "filled", "still_null"]);
    assert_eq!(out.rows[0][0].get_str(), Some("vec"));
    assert_eq!(out.rows[0][1].get_int(), Some(45));
    assert_eq!(out.rows[0][2].get_int(), Some(1), "the null-text row stays null");
    // 45 texts at max_batch 16 -> 3 model calls (batched, not per row)
    assert_eq!(m.calls.load(Ordering::SeqCst) - calls_before, 3);
    for (i, t) in texts.iter().enumerate() {
        let got = run(&db, &format!("?[vec] := *l_docs{{id: {}, vec}}", i + 2));
        assert_eq!(as_vec(&got.rows[0][0]), m.vector(t), "row {t}");
    }
    // backfill left the other columns alone
    let tag = run(&db, "?[tag] := *l_docs{id: 3, tag}");
    assert_eq!(tag.rows[0][0].get_str(), Some("x"));

    // the HNSW index sees the backfilled vectors
    let got = run(
        &db,
        "?[id, d] := ~l_docs:v{ id | query: embed('tg_len', 'cccccc'), k: 1, ef: 32, bind_distance: d }",
    );
    assert_eq!(got.rows[0][0].get_int(), Some(3));
    let got = run(
        &db,
        "?[id] := ~l_docs:v{ id | query: embed('tg_len', 'dddd zzz'), k: 1, ef: 32 }",
    );
    assert_eq!(got.rows[0][0].get_int(), Some(5));

    // a second backfill has nothing to do
    let out = run(&db, "::embedder backfill l_docs");
    assert_eq!(out.rows[0][1].get_int(), Some(0));

    // Rust API
    let out = db.backfill_embeddings("l_docs").unwrap();
    assert_eq!(out.rows[0][1].get_int(), Some(0));
}

#[test]
fn lenient_needs_a_nullable_column() {
    let m = Letters::new(8, 16);
    register_embedder("tg_nn", m).unwrap();
    let db = DbInstance::new("mem", "", "").unwrap();
    run(
        &db,
        ":create nn_docs { id: Int => text: String, vec: <F32; 8> default embed_or_null('tg_nn', text) }",
    );
    run(&db, "::embedder disable tg_nn");
    let e = err(&db, "?[id, text] <- [[1, 'abc']] :put nn_docs { id => text }");
    assert!(e.to_lowercase().contains("null"), "{e}");
    let e = err(&db, "::embedder backfill no_such_relation_here");
    assert!(e.contains("no_such_relation_here"), "{e}");
}

#[test]
fn reenable_gives_identical_vectors_through_a_relation() {
    let m = Letters::new(8, 32);
    register_embedder("tg_same", m.clone()).unwrap();
    let db = DbInstance::new("mem", "", "").unwrap();
    run(
        &db,
        ":create same { id: Int => text: String, vec: <F32; 8> default embed('tg_same', text) }",
    );
    run(&db, "?[id, text] <- [[1, 'the quick brown fox']] :put same { id => text }");
    let first = run(&db, "?[vec] := *same{id: 1, vec}");
    run(&db, "::embedder disable tg_same");
    run(&db, "::embedder enable tg_same");
    run(&db, "?[id, text] <- [[2, 'the quick brown fox']] :put same { id => text }");
    let second = run(&db, "?[vec] := *same{id: 2, vec}");
    assert_eq!(as_vec(&first.rows[0][0]), as_vec(&second.rows[0][0]));
}

#[cfg(not(feature = "infer"))]
#[test]
fn builtin_enable_without_infer_names_the_feature() {
    let db = DbInstance::new("mem", "", "").unwrap();
    let e = err(&db, "::embedder enable tg_gemma {model: 'embeddinggemma', precision: 'int8'}");
    assert!(e.contains("infer") && e.contains("tg_gemma"), "{e}");
    let e = err(&db, "::embedder enable tg_gemma2");
    assert!(e.contains("infer"), "{e}");
    let e = err(&db, "::embedder enable tg_bad {color: 'red'}");
    assert!(e.contains("unknown option 'color'"), "{e}");
    // list works regardless
    run(&db, "::embedder list");
}

// ------------------------------------------------------------------ infer

#[cfg(feature = "infer")]
fn rss_kib() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(0)
}

/// Physical footprint (what Activity Monitor shows; includes Metal buffers).
#[cfg(feature = "infer")]
fn footprint() -> String {
    let out = std::process::Command::new("footprint")
        .args(["-p", &std::process::id().to_string()])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout)
            .lines()
            .find(|l| l.contains("Footprint:"))
            .map(|l| l.trim().to_string())
            .unwrap_or_default(),
        Err(_) => String::new(),
    }
}

#[cfg(feature = "infer")]
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Real models: enable -> embed -> disable -> enable, for EmbeddingGemma int8
/// and potion-base-8M. Needs the models in the HF cache. Prints RSS.
#[cfg(feature = "infer")]
#[test]
#[ignore]
fn real_models_toggle_cycle() {
    let db = DbInstance::new("mem", "", "").unwrap();
    let text = "Kleos is an embedded Datalog database for agent memory.";

    // --- gemma
    let rss0 = rss_kib();
    let fp0 = footprint();
    run(
        &db,
        "::embedder enable tg_gemma {model: 'embeddinggemma', precision: 'int8', device: 'auto'}",
    );
    let rows = run(&db, "::embedder list");
    let row = rows.rows.iter().find(|r| r[0].get_str() == Some("tg_gemma")).unwrap();
    assert_eq!(row[1].get_str(), Some("gemma"));
    assert_eq!(row[5], DataValue::Bool(false), "lazy: not loaded before first use");
    assert!(rows.rows.iter().any(|r| r[0].get_str() == Some("tg_gemma_query")));

    let v1 = as_vec(&run(&db, &format!("?[v] := v = embed('tg_gemma', '{text}')")).rows[0][0]);
    assert_eq!(v1.len(), 768);
    let rss1 = rss_kib();
    let fp1 = footprint();
    let rows = run(&db, "::embedder list");
    let row = rows.rows.iter().find(|r| r[0].get_str() == Some("tg_gemma")).unwrap();
    assert_eq!(row[5], DataValue::Bool(true));
    eprintln!("gemma device after load: {:?}, precision {:?}", row[7], row[6]);

    run(&db, "::embedder disable tg_gemma");
    let rss2 = rss_kib();
    let fp2 = footprint();
    assert!(!kleos::embed::is_embedder_enabled("tg_gemma"));
    assert!(!kleos::embed::is_embedder_enabled("tg_gemma_query"), "twin goes too");
    let e = err(&db, &format!("?[v] := v = embed('tg_gemma_query', '{text}')"));
    assert!(e.contains("tg_gemma_query"), "{e}");

    eprintln!(
        "RSS gemma int8: before enable {} MiB | after enable+embed {} MiB | after disable {} MiB",
        rss0 / 1024,
        rss1 / 1024,
        rss2 / 1024
    );
    eprintln!("footprint: before [{fp0}] | loaded [{fp1}] | disabled [{fp2}]");

    run(&db, "::embedder enable tg_gemma");
    let rows = run(&db, "::embedder list");
    let row = rows.rows.iter().find(|r| r[0].get_str() == Some("tg_gemma")).unwrap();
    assert_eq!(row[6].get_str(), Some("q8"), "re-enable keeps the previous spec");
    let v2 = as_vec(&run(&db, &format!("?[v] := v = embed('tg_gemma', '{text}')")).rows[0][0]);
    let d = max_abs_diff(&v1, &v2);
    eprintln!("gemma re-enable max |v1 - v2| = {d:e}");
    assert!(d < 1e-4, "re-enabled gemma gives the same vector (diff {d})");
    run(&db, "::embedder disable tg_gemma");
    eprintln!("RSS after second disable {} MiB", rss_kib() / 1024);

    // --- potion
    run(&db, "::embedder enable tg_potion {model: 'potion-base-8M'}");
    let p1 = as_vec(&run(&db, &format!("?[v] := v = embed('tg_potion', '{text}')")).rows[0][0]);
    run(&db, "::embedder disable tg_potion");
    let e = err(&db, &format!("?[v] := v = embed('tg_potion', '{text}')"));
    assert!(e.contains("disabled"), "{e}");
    run(&db, "::embedder enable tg_potion");
    let p2 = as_vec(&run(&db, &format!("?[v] := v = embed('tg_potion', '{text}')")).rows[0][0]);
    assert_eq!(p1, p2);
    run(&db, "::embedder enable tg_potion2 {model: 'minishlab/potion-base-8M'}");
    let p3 = as_vec(&run(&db, &format!("?[v] := v = embed('tg_potion2', '{text}')")).rows[0][0]);
    assert_eq!(p1, p3);
    let e = err(&db, "::embedder enable tg_bad {model: 'no-such-model'}");
    assert!(e.contains("no-such-model"), "{e}");
    let e = err(&db, "::embedder enable tg_bad {model: 'potion-base-8M', precision: 'int4'}");
    assert!(e.contains("f32"), "{e}");
}
