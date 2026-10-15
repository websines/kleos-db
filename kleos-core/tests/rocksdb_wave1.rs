/*
 * Copyright 2026, The kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Wave-1 vector features on the **rocksdb** backend: half-precision columns,
//! quantized HNSW (built over existing rows, i.e. through the non-blocking
//! SST-ingest build, then extended incrementally), persistence across a
//! reopen, and filtered search. The other wave-1 features run on rocksdb
//! through their own files: `live_subscriptions` (its `each` helper adds
//! rocksdb when compiled in) and `memory_supersession`
//! (`KLEOS_TEST_BACKEND=rocksdb`).
//!
//! Run with:
//! `cargo test -p kleos --features storage-rocksdb --test rocksdb_wave1`

#![cfg(feature = "storage-rocksdb")]

use std::collections::BTreeMap;

use kleos::{DbInstance, NamedRows, ScriptMutability};

const DIM: usize = 16;
const CATS: usize = 50;

fn run(db: &DbInstance, script: &str) -> NamedRows {
    db.run_script(script, BTreeMap::new(), ScriptMutability::Mutable)
        .unwrap_or_else(|e| panic!("script failed: {e:?}\n{script}"))
}

fn vectors(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
    };
    (0..n).map(|_| (0..DIM).map(|_| next()).collect()).collect()
}

/// Round to F16 the way the column stores it.
fn f16(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| half::f16::from_f32(*x).to_f32()).collect()
}

fn lit(v: &[f32]) -> String {
    format!(
        "vec([{}])",
        v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(", ")
    )
}

fn put_range(db: &DbInstance, pts: &[Vec<f32>], from: usize) {
    for (c, chunk) in pts[from..].chunks(200).enumerate() {
        let rows = chunk
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let id = from + c * 200 + i;
                format!("[{id}, {}, {}]", id % CATS, lit(v))
            })
            .collect::<Vec<_>>()
            .join(", ");
        run(
            db,
            &format!("?[id, cat, emb] <- [{rows}] :put pts {{id => cat, emb}}"),
        );
    }
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum();
    let nb: f32 = b.iter().map(|x| x * x).sum();
    1.0 - dot as f64 / ((na as f64) * (nb as f64)).sqrt()
}

/// Exact top-k ids over the stored (F16-rounded) vectors passing `keep`.
fn truth(pts: &[Vec<f32>], q: &[f32], k: usize, keep: impl Fn(usize) -> bool) -> Vec<i64> {
    let mut d: Vec<(f64, usize)> = pts
        .iter()
        .enumerate()
        .filter(|(i, _)| keep(*i))
        .map(|(i, p)| (cos(q, &f16(p)), i))
        .collect();
    d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    d.into_iter().take(k).map(|x| x.1 as i64).collect()
}

fn search(db: &DbInstance, q: &[f32], k: usize, ef: usize, extra: &str) -> Vec<(i64, i64)> {
    run(
        db,
        &format!(
            "?[id, cat, d] := ~pts:idx{{id, cat | query: {}, k: {k}, ef: {ef}, \
             bind_distance: d{extra}}} :order d",
            lit(q)
        ),
    )
    .rows
    .iter()
    .map(|r| (r[0].get_int().unwrap(), r[1].get_int().unwrap()))
    .collect()
}

fn recall(got: &[(i64, i64)], truth: &[i64]) -> f64 {
    let hits = got.iter().filter(|(id, _)| truth.contains(id)).count();
    hits as f64 / truth.len().max(1) as f64
}

fn code_rows(db: &DbInstance) -> i64 {
    run(db, "?[count(fr_id)] := *pts:idx{layer: 2, fr_id}")
        .rows
        .first()
        .and_then(|r| r[0].get_int())
        .unwrap_or(0)
}

/// F16 column + turbo4-quantized HNSW on rocksdb: the index is built over
/// existing rows (non-blocking SST-ingest build), extended by incremental
/// puts, survives a reopen, and answers unfiltered and filtered searches with
/// the expected recall.
#[test]
fn quantized_f16_hnsw_persists_and_searches_on_rocksdb() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db");
    let path = path.to_str().unwrap();
    let pts = vectors(3000, 11);
    let built = 2500;
    {
        let db = DbInstance::new("rocksdb", path, "").unwrap();
        run(&db, ":create pts { id: Int => cat: Int, emb: <F16; 16> }");
        put_range(&db, &pts[..built], 0);
        run(
            &db,
            "::hnsw create pts:idx { dim: 16, m: 12, dtype: F16, fields: [emb], \
             distance: Cosine, ef_construction: 64, quantize: turbo4 }",
        );
        assert_eq!(code_rows(&db), built as i64, "one code row per built vector");
        put_range(&db, &pts, built);
        assert_eq!(code_rows(&db), pts.len() as i64, "incremental puts write codes");
    }

    let db = DbInstance::new("rocksdb", path, "").unwrap();
    assert_eq!(code_rows(&db), pts.len() as i64, "codes persisted");
    let queries = vectors(20, 99);

    let mut r_all = 0.0;
    let mut r_filtered = 0.0;
    for q in &queries {
        let got = search(&db, q, 10, 64, "");
        assert_eq!(got.len(), 10);
        r_all += recall(&got, &truth(&pts, q, 10, |_| true));

        // 2% selectivity: the filter is applied inside the search (or by the
        // exact-scan fallback); every row passes, k rows come back.
        let got = search(&db, q, 10, 32, ", filter: cat == 7");
        assert_eq!(got.len(), 10, "filtered search under-filled");
        assert!(got.iter().all(|(_, c)| *c == 7), "{got:?}");
        r_filtered += recall(&got, &truth(&pts, q, 10, |i| i % CATS == 7));
    }
    let n = queries.len() as f64;
    assert!(r_all / n >= 0.9, "recall {}", r_all / n);
    assert!(r_filtered / n >= 0.9, "filtered recall {}", r_filtered / n);

    // Updates and deletes after the reopen keep the index consistent.
    run(&db, "?[id] <- [[0], [1], [2]] :rm pts {id}");
    run(
        &db,
        &format!("?[id, cat, emb] <- [[3, 7, {}]] :put pts {{id => cat, emb}}", lit(&pts[9])),
    );
    assert_eq!(code_rows(&db), pts.len() as i64 - 3);
    let got = search(&db, &pts[9], 2, 32, ", filter: cat == 7");
    let ids: Vec<i64> = got.iter().map(|x| x.0).collect();
    assert!(ids.contains(&3), "{got:?}");
    let got = search(&db, &pts[0], 5, 32, "");
    assert!(got.iter().all(|(id, _)| *id > 2), "deleted rows returned: {got:?}");
}

/// Unquantized F32 index on rocksdb with a selective filter that forces the
/// exact-scan fallback and a broad one that stays on the graph: both return
/// exactly the filtered top-k.
#[test]
fn filtered_search_graph_and_scan_paths_on_rocksdb() {
    let dir = tempfile::tempdir().unwrap();
    let db = DbInstance::new("rocksdb", dir.path().join("db").to_str().unwrap(), "").unwrap();
    run(&db, ":create pts { id: Int => cat: Int, emb: <F16; 16> }");
    run(
        &db,
        "::hnsw create pts:idx { dim: 16, m: 12, dtype: F16, fields: [emb], \
         distance: Cosine, ef_construction: 64 }",
    );
    let pts = vectors(4000, 5);
    put_range(&db, &pts, 0);
    for q in vectors(10, 77) {
        for (filter, keep) in [
            ("cat < 25", Box::new(|i: usize| i % CATS < 25) as Box<dyn Fn(usize) -> bool>),
            ("cat == 3", Box::new(|i: usize| i % CATS == 3)),
            ("cat == 3 && id < 400", Box::new(|i: usize| i % CATS == 3 && i < 400)),
        ] {
            let got = search(&db, &q, 5, 64, &format!(", filter: {filter}"));
            let t = truth(&pts, &q, 5, &keep);
            assert_eq!(got.len(), t.len(), "`{filter}` under-filled");
            assert!(got.iter().all(|(id, _)| keep(*id as usize)), "{got:?}");
            assert!(recall(&got, &t) >= 0.8, "`{filter}`: {got:?} vs {t:?}");
        }
    }
}
