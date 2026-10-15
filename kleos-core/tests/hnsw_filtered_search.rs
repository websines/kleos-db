/*
 * Copyright 2026, The kleos authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Filters applied inside HNSW search (kleos upgrade plan A1). The search
//! `filter` used to run after candidate retrieval, so a selective filter
//! returned fewer than `k` rows even when many rows matched. These tests check
//! that filtered searches return `k` rows whenever at least `k` rows match,
//! that every row honours the filter (and `radius`, index filters, bound
//! variables), and that the rows are the right ones (recall against an exact
//! scan).

use kleos::{DataValue, DbInstance, NamedRows, ScriptMutability, Vector};
use ndarray::Array1;
use std::collections::{BTreeMap, BTreeSet};

const DIM: usize = 8;

fn run(db: &DbInstance, s: &str, params: BTreeMap<String, DataValue>) -> NamedRows {
    db.run_script(s, params, ScriptMutability::Mutable)
        .unwrap_or_else(|e| panic!("script failed: {e:?}\n--- script ---\n{s}"))
}

/// Deterministic pseudo-random vectors (64-bit LCG).
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
    fn vector(&mut self) -> Vec<f32> {
        (0..DIM).map(|_| self.next_f32()).collect()
    }
}

fn vec_value(v: &[f32]) -> DataValue {
    DataValue::Vec(Vector::F32(Array1::from(v.to_vec())))
}

/// `n` rows `{id => cat: id % 1000, emb}` with an HNSW index over `emb`.
fn setup(n: usize, index_filter: Option<&str>) -> (DbInstance, Lcg) {
    let filter = index_filter
        .map(|f| format!(", filter: {f}"))
        .unwrap_or_default();
    setup_with(n, "F32", &filter)
}

/// `setup` with the vector element type and extra index options given.
fn setup_with(n: usize, elem: &str, extra_opts: &str) -> (DbInstance, Lcg) {
    let db = DbInstance::new("mem", "", "").unwrap();
    run(
        &db,
        &format!(":create pts {{ id: Int => cat: Int, emb: <{elem}; {DIM}> }}"),
        BTreeMap::new(),
    );
    let mut rng = Lcg(42);
    let rows: Vec<DataValue> = (0..n)
        .map(|i| {
            DataValue::List(vec![
                DataValue::from(i as i64),
                DataValue::from((i % 1000) as i64),
                vec_value(&rng.vector()),
            ])
        })
        .collect();
    run(
        &db,
        "?[id, cat, emb] <- $rows :put pts { id => cat, emb }",
        BTreeMap::from([("rows".to_string(), DataValue::List(rows))]),
    );
    run(
        &db,
        &format!(
            "::hnsw create pts:idx {{ dim: {DIM}, m: 16, dtype: {elem}, fields: [emb], \
             distance: Cosine, ef_construction: 64{extra_opts} }}"
        ),
        BTreeMap::new(),
    );
    (db, rng)
}

/// HNSW search; returns `(id, cat, dist)` rows.
fn search(db: &DbInstance, q: &[f32], k: usize, ef: usize, extra: &str) -> Vec<(i64, i64, f64)> {
    let script = format!(
        "?[id, cat, d] := ~pts:idx{{ id, cat | query: $q, k: {k}, ef: {ef}, \
         bind_distance: d{extra} }}"
    );
    let res = run(
        db,
        &script,
        BTreeMap::from([("q".to_string(), vec_value(q))]),
    );
    res.rows
        .iter()
        .map(|r| {
            (
                r[0].get_int().unwrap(),
                r[1].get_int().unwrap(),
                r[2].get_float().unwrap(),
            )
        })
        .collect()
}

/// Exact top-`k` ids among rows matching `cond` (a CozoScript condition over
/// `id`, `cat`, `d`).
fn exact(db: &DbInstance, q: &[f32], k: usize, cond: &str) -> Vec<i64> {
    let script = format!(
        "?[id, d] := *pts{{ id, cat, emb }}, d = cos_dist(emb, $q), {cond} \
         :order d :limit {k}"
    );
    let res = run(
        db,
        &script,
        BTreeMap::from([("q".to_string(), vec_value(q))]),
    );
    res.rows.iter().map(|r| r[0].get_int().unwrap()).collect()
}

fn recall(got: &[(i64, i64, f64)], truth: &[i64]) -> f64 {
    let got: BTreeSet<i64> = got.iter().map(|r| r.0).collect();
    truth.iter().filter(|id| got.contains(id)).count() as f64 / truth.len().max(1) as f64
}

#[test]
fn selective_filters_fill_k() {
    let (db, mut rng) = setup(20_000, None);
    let k = 10;
    // (search filter, exact condition, rows matching): 1% and 0.1% of 20k
    for (filter, cond, matching) in [
        ("cat < 10", "cat < 10", 200usize),
        ("cat == 7", "cat == 7", 20),
    ] {
        let mut total_recall = 0.0;
        let trials = 10;
        for _ in 0..trials {
            let q = rng.vector();
            let got = search(&db, &q, k, 32, &format!(", filter: {filter}"));
            assert!(matching >= k);
            assert_eq!(got.len(), k, "filter `{filter}` under-filled: {got:?}");
            for (id, cat, _) in &got {
                assert_eq!(*cat, id % 1000);
                let ok = if filter == "cat < 10" {
                    *cat < 10
                } else {
                    *cat == 7
                };
                assert!(ok, "row {id} (cat {cat}) violates `{filter}`");
            }
            total_recall += recall(&got, &exact(&db, &q, k, cond));
        }
        let mean = total_recall / trials as f64;
        assert!(mean >= 0.9, "filter `{filter}`: mean recall {mean}");
    }
}

#[test]
fn unfiltered_and_pass_all_filter_agree_with_exact() {
    let (db, mut rng) = setup(5_000, None);
    let k = 10;
    let (mut r_plain, mut r_true) = (0.0, 0.0);
    let trials = 10;
    for _ in 0..trials {
        let q = rng.vector();
        let truth = exact(&db, &q, k, "cat >= 0");
        let plain = search(&db, &q, k, 64, "");
        let all = search(&db, &q, k, 64, ", filter: cat >= 0");
        assert_eq!(plain.len(), k);
        assert_eq!(all.len(), k);
        r_plain += recall(&plain, &truth);
        r_true += recall(&all, &truth);
    }
    assert!(
        r_plain / trials as f64 >= 0.9,
        "unfiltered recall {r_plain}"
    );
    assert!(
        r_true / trials as f64 >= 0.9,
        "pass-all filter recall {r_true}"
    );
}

#[test]
fn filter_matching_nothing_or_few_rows() {
    let (db, mut rng) = setup(3_000, None);
    let q = rng.vector();
    assert!(search(&db, &q, 10, 32, ", filter: cat > 5000").is_empty());
    // cat == 3 matches exactly 3 rows (3, 1003, 2003): all of them
    let got = search(&db, &q, 10, 32, ", filter: cat == 3");
    let ids: BTreeSet<i64> = got.iter().map(|r| r.0).collect();
    assert_eq!(ids, BTreeSet::from([3, 1003, 2003]));
}

#[test]
fn filter_on_bound_distance_and_radius() {
    let (db, mut rng) = setup(5_000, None);
    let k = 10;
    for _ in 0..5 {
        let q = rng.vector();
        // the filter may read the bound distance variable
        let got = search(&db, &q, k, 32, ", filter: cat < 50 && d > 0.2");
        assert_eq!(got.len(), k);
        for (_, cat, d) in &got {
            assert!(*cat < 50 && *d > 0.2, "violates filter: {got:?}");
        }
        let truth = exact(&db, &q, k, "cat < 50, d > 0.2");
        assert!(recall(&got, &truth) >= 0.8, "{got:?} vs {truth:?}");

        // radius combines with the filter
        let radius = 0.6;
        let got = search(
            &db,
            &q,
            k,
            32,
            &format!(", radius: {radius}, filter: cat < 20"),
        );
        let truth = exact(&db, &q, k, &format!("cat < 20, d <= {radius}"));
        assert_eq!(got.len(), truth.len(), "{got:?} vs {truth:?}");
        for (_, cat, d) in &got {
            assert!(*cat < 20 && *d <= radius, "violates radius/filter: {got:?}");
        }
    }
}

#[test]
fn brute_force_fallback_respects_index_filter() {
    // Only even ids are indexed; a very selective search filter must never
    // return a row the index excludes, even on the exact-scan fallback.
    let (db, mut rng) = setup(20_000, Some("id % 2 == 0"));
    for _ in 0..5 {
        let q = rng.vector();
        // cat == 4 matches 20 rows, all even (id % 1000 == 4)
        let got = search(&db, &q, 10, 32, ", filter: cat == 4");
        assert_eq!(got.len(), 10);
        let truth = exact(&db, &q, 10, "cat == 4");
        assert!(recall(&got, &truth) >= 0.9, "{got:?} vs {truth:?}");
        // cat == 5 matches 20 rows, all odd: none are indexed
        assert!(search(&db, &q, 10, 32, ", filter: cat == 5").is_empty());
    }
}

#[test]
fn multi_vector_fields_with_filter() {
    let db = DbInstance::new("mem", "", "").unwrap();
    run(
        &db,
        ":create docs { id: Int => tag: Int, segs: [<F32; 2>] }",
        BTreeMap::new(),
    );
    // doc i: segment 0 at angle i, segment 1 at angle i + 0.5 (radians)
    let rows: Vec<String> = (0..400)
        .map(|i| {
            let a = i as f32 * 0.01;
            let b = a + 0.5;
            format!(
                "[{i}, {}, [[{}, {}], [{}, {}]]]",
                i % 100,
                a.cos(),
                a.sin(),
                b.cos(),
                b.sin()
            )
        })
        .collect();
    run(
        &db,
        &format!(
            "?[id, tag, segs] <- [{}] :put docs {{ id => tag, segs }}",
            rows.join(",")
        ),
        BTreeMap::new(),
    );
    run(
        &db,
        "::hnsw create docs:idx { dim: 2, m: 16, dtype: F32, fields: [segs], \
         distance: Cosine, ef_construction: 64 }",
        BTreeMap::new(),
    );
    let res = run(
        &db,
        "?[id, tag, sub, d] := ~docs:idx{ id, tag | query: vec([1.0, 0.0]), k: 4, ef: 16, \
         bind_field_idx: sub, bind_distance: d, filter: tag == 50 && sub == 1 }",
        BTreeMap::new(),
    );
    assert_eq!(res.rows.len(), 4, "{:?}", res.rows);
    for r in &res.rows {
        assert_eq!(r[1].get_int(), Some(50));
        assert_eq!(r[2].get_int(), Some(1));
    }
}

/// Filtered search composes with half-precision columns and quantized
/// (code-traversal) indexes: k rows at 1% and 0.1% selectivity, and the bound
/// distances are exact (equal to `cos_dist` on the stored vector).
#[test]
fn filters_with_half_precision_and_quantized_indexes() {
    for (elem, opts) in [("F16", ""), ("F32", ", quantize: turbo4")] {
        let (db, mut rng) = setup_with(20_000, elem, opts);
        for filter in ["cat < 10", "cat == 7"] {
            for _ in 0..3 {
                let q = rng.vector();
                let got = search(&db, &q, 10, 32, &format!(", filter: {filter}"));
                assert_eq!(got.len(), 10, "{elem}{opts} `{filter}` under-filled");
                let truth = exact(&db, &q, 10, filter);
                assert!(
                    recall(&got, &truth) >= 0.8,
                    "{elem}{opts} `{filter}`: {got:?} vs {truth:?}"
                );
                let ids: Vec<DataValue> = got.iter().map(|r| DataValue::from(r.0)).collect();
                let exact_d = run(
                    &db,
                    "?[id, d] := id in $ids, *pts{id, emb}, d = cos_dist(emb, $q)",
                    BTreeMap::from([
                        ("ids".to_string(), DataValue::List(ids)),
                        ("q".to_string(), vec_value(&q)),
                    ]),
                );
                let exact_d: BTreeMap<i64, f64> = exact_d
                    .rows
                    .iter()
                    .map(|r| (r[0].get_int().unwrap(), r[1].get_float().unwrap()))
                    .collect();
                for (id, _, d) in &got {
                    assert!(
                        (exact_d[id] - d).abs() < 1e-5,
                        "{elem}{opts}: distance of {id} is {d}, exact {}",
                        exact_d[id]
                    );
                }
            }
        }
    }
}
