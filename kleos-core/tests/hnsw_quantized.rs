/*
 * Copyright 2026, The kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Half-width vector element types (A3) and quantized HNSW traversal (A2).
//!
//! * `<F16; N>` / `<BF16; N>` columns round on write, read back as F32, and
//!   persist through every backend's value encoding;
//! * an index created with `quantize: ...` keeps one code row per vector,
//!   searches with code-estimated distances, and returns exactly rescored
//!   results whose recall matches the unquantized index;
//! * code rows follow inserts, updates, deletes, `::reindex` and the bulk
//!   build; unquantized indices are unchanged (no code rows).

use std::collections::BTreeMap;

use kleos::{DataValue, DbInstance, NamedRows, ScriptMutability, Vector};

fn run(db: &DbInstance, script: &str) -> NamedRows {
    db.run_script(script, BTreeMap::new(), ScriptMutability::Mutable)
        .unwrap_or_else(|e| panic!("script failed: {e:?}\n{script}"))
}

fn vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
}

fn vec_lit(v: &[f64]) -> String {
    format!(
        "vec([{}])",
        v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(", ")
    )
}

fn f32s(v: &DataValue) -> Vec<f32> {
    match v {
        DataValue::Vec(Vector::F32(a)) => a.to_vec(),
        other => panic!("expected an F32 vector, got {other:?}"),
    }
}

/// Exact distance as the index defines it (computed in f32 like the index).
fn exact(dist: &str, a: &[f64], b: &[f64]) -> f64 {
    let a: Vec<f32> = a.iter().map(|x| *x as f32).collect();
    let b: Vec<f32> = b.iter().map(|x| *x as f32).collect();
    let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    match dist {
        "L2" => a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() as f64,
        "Cosine" => {
            let na: f32 = a.iter().map(|x| x * x).sum();
            let nb: f32 = b.iter().map(|x| x * x).sum();
            1.0 - dot as f64 / ((na as f64) * (nb as f64)).sqrt()
        }
        _ => 1.0 - dot as f64,
    }
}

fn put_all(db: &DbInstance, pts: &[Vec<f64>]) {
    for (c, chunk) in pts.chunks(100).enumerate() {
        let rows = chunk
            .iter()
            .enumerate()
            .map(|(i, v)| format!("[{}, {}]", c * 100 + i, vec_lit(v)))
            .collect::<Vec<_>>()
            .join(", ");
        run(db, &format!("?[id, emb] <- [{rows}] :put pts {{id => emb}}"));
    }
}

fn code_rows(db: &DbInstance) -> i64 {
    let res = run(db, "?[count(fr_id)] := *pts:idx{layer: 2, fr_id}");
    res.rows
        .first()
        .and_then(|r| r[0].get_int())
        .unwrap_or(0)
}

fn index_opts(dim: usize, dist: &str, quant: &str) -> String {
    let q = if quant.is_empty() {
        String::new()
    } else {
        format!(", quantize: {quant}")
    };
    format!(
        "::hnsw create pts:idx {{ dim: {dim}, m: 12, dtype: F32, fields: [emb], \
         distance: {dist}, ef_construction: 64{q} }}"
    )
}

/// Mean recall@10 over `queries` (vs brute force) and the largest relative
/// error between a returned distance and the exact one.
fn recall_and_distance_error(
    db: &DbInstance,
    pts: &[Vec<f64>],
    queries: &[Vec<f64>],
    dist: &str,
) -> (f64, f64) {
    let mut recall = 0.0;
    let mut max_err: f64 = 0.0;
    for q in queries {
        let mut truth: Vec<(f64, usize)> = pts
            .iter()
            .enumerate()
            .map(|(i, p)| (exact(dist, q, p), i))
            .collect();
        truth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let truth: Vec<i64> = truth[..10].iter().map(|x| x.1 as i64).collect();
        let res = run(
            db,
            &format!(
                "?[id, d] := ~pts:idx{{id | query: {}, k: 10, ef: 64, bind_distance: d}}",
                vec_lit(q)
            ),
        );
        assert_eq!(res.rows.len(), 10);
        for row in &res.rows {
            let id = row[0].get_int().unwrap();
            let d = row[1].get_float().unwrap();
            let e = exact(dist, q, &pts[id as usize]);
            max_err = max_err.max((d - e).abs() / e.abs().max(1e-3));
            if truth.contains(&id) {
                recall += 0.1;
            }
        }
    }
    (recall / queries.len() as f64, max_err)
}

#[test]
fn half_width_columns_round_trip() {
    let db = DbInstance::new("mem", "", "").unwrap();
    run(
        &db,
        ":create h { id: Int => a: <F16; 3>, b: <BF16; 3>?, c: [<F16; 2>]? }",
    );
    let cols = run(&db, "::columns h");
    let types: Vec<String> = cols.rows.iter().map(|r| format!("{:?}", r)).collect();
    assert!(types.iter().any(|t| t.contains("<F16;3>")), "{types:?}");
    assert!(types.iter().any(|t| t.contains("<BF16;3>?")), "{types:?}");
    run(
        &db,
        "?[id, a, b, c] <- [[1, [1.0, 0.1, 65504.0], vec([1.0, 0.1, 3.0]), [[0.5, 0.3]]], \
                           [2, vec([2.0, -0.333, 1e-3], 'F64'), null, null]] \
         :put h {id => a, b, c}",
    );
    let r = run(&db, "?[id, a, b, c] := *h{id, a, b, c} :order id");
    let f16 = |x: f32| half::f16::from_f32(x).to_f32();
    let bf16 = |x: f32| half::bf16::from_f32(x).to_f32();
    assert_eq!(f32s(&r.rows[0][1]), vec![1.0, f16(0.1), 65504.0]);
    assert_eq!(f32s(&r.rows[0][2]), vec![1.0, bf16(0.1), 3.0]);
    match &r.rows[0][3] {
        DataValue::List(l) => assert_eq!(f32s(&l[0]), vec![0.5, f16(0.3)]),
        other => panic!("{other:?}"),
    }
    assert_eq!(f32s(&r.rows[1][1]), vec![2.0, f16(-0.333), f16(1e-3)]);
    assert_eq!(r.rows[1][2], DataValue::Null);
    // Values read back satisfy the column type, and vec() can produce them.
    run(
        &db,
        "?[id, a, b, c] := *h{id: 1, a, b, c}, id = 4 :put h {id => a, b, c}",
    );
    let again = run(&db, "?[a, b, c] := *h{id: 4, a, b, c}");
    assert_eq!(again.rows[0], r.rows[0][1..].to_vec());
    run(
        &db,
        "?[id, a, b, c] <- [[3, vec([1.5, 2.5, 3.5], 'F16'), vec([1, 2, 3], 'BF16'), null]] \
         :put h {id => a, b, c}",
    );
    let r = run(&db, "?[a] := *h{id: 3, a}");
    assert_eq!(f32s(&r.rows[0][0]), vec![1.5, 2.5, 3.5]);
    // Distance functions work on half-width columns.
    let r = run(&db, "?[d] := *h{id: 1, a}, d = l2_dist(a, vec([1.0, 0.1, 65504.0]))");
    assert!(r.rows[0][0].get_float().unwrap() < 1e-6);
}

#[test]
fn half_width_columns_persist_on_sqlite() {
    let dir = std::env::temp_dir().join(format!("kleos-f16-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("f16.db");
    let path = path.to_str().unwrap();
    {
        let db = DbInstance::new("sqlite", path, "").unwrap();
        run(&db, ":create h { id: Int => a: <F16; 4>, b: <F32; 4> }");
        run(
            &db,
            "?[id, a, b] <- [[1, [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]] :put h {id => a, b}",
        );
    }
    let db = DbInstance::new("sqlite", path, "").unwrap();
    let r = run(&db, "?[a, b] := *h{id: 1, a, b}");
    let f16 = |x: f32| half::f16::from_f32(x).to_f32();
    assert_eq!(f32s(&r.rows[0][0]), [0.1, 0.2, 0.3, 0.4].map(f16).to_vec());
    assert_eq!(f32s(&r.rows[0][1]), vec![0.1, 0.2, 0.3, 0.4]);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hnsw_over_f16_column() {
    let db = DbInstance::new("mem", "", "").unwrap();
    run(&db, ":create pts { id: Int => emb: <F16; 8> }");
    run(
        &db,
        "::hnsw create pts:idx { dim: 8, m: 8, dtype: F16, fields: [emb], \
         distance: Cosine, ef_construction: 32 }",
    );
    let pts = vectors(150, 8, 7);
    put_all(&db, &pts);
    for (i, p) in pts.iter().enumerate().step_by(10) {
        let r = run(
            &db,
            &format!("?[id] := ~pts:idx{{id | query: {}, k: 1, ef: 32}}", vec_lit(p)),
        );
        assert_eq!(r.rows, vec![vec![DataValue::from(i as i64)]]);
    }
    // dtype must match the column.
    let err = db.run_script(
        "::hnsw create pts:idx2 { dim: 8, m: 8, dtype: F32, fields: [emb], \
         distance: Cosine, ef_construction: 32 }",
        BTreeMap::new(),
        ScriptMutability::Mutable,
    );
    assert!(err.is_err());
}

#[test]
fn quantized_search_recall_and_exact_distances() {
    const N: usize = 600;
    let pts = vectors(N, 48, 11);
    let queries = vectors(40, 48, 12);
    for dist in ["Cosine", "L2", "IP"] {
        // Baseline: the unquantized index.
        let db = DbInstance::new("mem", "", "").unwrap();
        run(&db, ":create pts { id: Int => emb: <F32; 48> }");
        run(&db, &index_opts(48, dist, ""));
        put_all(&db, &pts);
        assert_eq!(code_rows(&db), 0, "unquantized index must not store codes");
        let (base_recall, _) = recall_and_distance_error(&db, &pts, &queries, dist);

        for quant in ["binary", "turbo2", "turbo4", "turbo8"] {
            let db = DbInstance::new("mem", "", "").unwrap();
            run(&db, ":create pts { id: Int => emb: <F32; 48> }");
            run(&db, &index_opts(48, dist, quant));
            put_all(&db, &pts);
            assert_eq!(code_rows(&db), N as i64, "{dist} {quant}");
            let (recall, err) = recall_and_distance_error(&db, &pts, &queries, dist);
            assert!(
                err < 1e-4,
                "{dist} {quant}: returned distances must be exact (rel err {err})"
            );
            // 1-bit codes are coarse at this low dimension (48, padded to 64);
            // the multi-bit codes track the exact index closely.
            let slack = if quant == "binary" { 0.12 } else { 0.05 };
            assert!(
                recall >= base_recall - slack,
                "{dist} {quant}: recall {recall} vs unquantized {base_recall}"
            );
        }
    }
}

#[test]
fn quantized_bulk_build_matches_incremental() {
    const N: usize = 500;
    let pts = vectors(N, 32, 21);
    let queries = vectors(30, 32, 22);
    let db = DbInstance::new("mem", "", "").unwrap();
    run(&db, ":create pts { id: Int => emb: <F32; 32> }");
    put_all(&db, &pts);
    // Index created over existing rows: the bulk build path.
    run(&db, &index_opts(32, "Cosine", "turbo4"));
    assert_eq!(code_rows(&db), N as i64);
    let (recall, err) = recall_and_distance_error(&db, &pts, &queries, "Cosine");
    assert!(err < 1e-4, "{err}");
    assert!(recall >= 0.9, "{recall}");
    let listed = run(&db, "::indices pts");
    assert!(format!("{:?}", listed.rows).contains("turbo4"));
}

#[test]
fn quantized_index_follows_updates_deletes_and_reindex() {
    const N: usize = 200;
    let pts = vectors(N, 16, 31);
    let db = DbInstance::new("mem", "", "").unwrap();
    run(&db, ":create pts { id: Int => emb: <F32; 16> }");
    run(&db, &index_opts(16, "L2", "binary"));
    put_all(&db, &pts);
    assert_eq!(code_rows(&db), N as i64);

    // Update: move every 7th point; the code must follow the new vector.
    let moved = vectors(N, 16, 32);
    for i in (0..N).step_by(7) {
        run(
            &db,
            &format!("?[id, emb] <- [[{i}, {}]] :put pts {{id => emb}}", vec_lit(&moved[i])),
        );
    }
    let current: Vec<Vec<f64>> = (0..N)
        .map(|i| if i % 7 == 0 { moved[i].clone() } else { pts[i].clone() })
        .collect();
    assert_eq!(code_rows(&db), N as i64);

    // Delete 60%.
    let doomed: Vec<usize> = (0..N).filter(|i| i % 5 != 0 || i % 50 == 0).collect();
    for &i in &doomed {
        run(&db, &format!("?[id] <- [[{i}]] :rm pts {{id}}"));
    }
    let survivors: Vec<usize> = (0..N).filter(|i| !doomed.contains(i)).collect();
    assert_eq!(code_rows(&db), survivors.len() as i64);

    let check = |db: &DbInstance| {
        for &i in &survivors {
            let r = run(
                db,
                &format!(
                    "?[id, d] := ~pts:idx{{id | query: {}, k: 1, ef: 64, bind_distance: d}}",
                    vec_lit(&current[i])
                ),
            );
            assert_eq!(r.rows[0][0], DataValue::from(i as i64), "survivor {i}");
            assert!(r.rows[0][1].get_float().unwrap() < 1e-6);
        }
    };
    check(&db);

    run(&db, "::reindex pts");
    assert_eq!(code_rows(&db), survivors.len() as i64);
    check(&db);
}

#[test]
fn quantize_option_is_validated() {
    let db = DbInstance::new("mem", "", "").unwrap();
    run(&db, ":create pts { id: Int => emb: <F32; 4> }");
    let err = db.run_script(
        "::hnsw create pts:idx { dim: 4, m: 6, dtype: F32, fields: [emb], \
         distance: L2, ef_construction: 16, quantize: turbo5 }",
        BTreeMap::new(),
        ScriptMutability::Mutable,
    );
    assert!(err.is_err());
    run(
        &db,
        "::hnsw create pts:idx { dim: 4, m: 6, dtype: F32, fields: [emb], \
         distance: L2, ef_construction: 16, quantize: 'turbo8', rescore: 3 }",
    );
    let listed = run(&db, "::indices pts");
    let s = format!("{:?}", listed.rows);
    assert!(s.contains("turbo8") && s.contains("rescore"), "{s}");
}
