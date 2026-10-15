/*
 * Copyright 2026, The kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Deleting vectors must not make the surviving ones unreachable.
//!
//! HNSW removal used to unlink the node and stop. Its former neighbours lost
//! any path that ran through it, so search silently skipped whole regions of
//! the graph: removing a hub of a three-node index hid a survivor ~2 times in
//! 3. Removal now reconnects the orphaned neighbours. Every survivor queried
//! with its own vector must find itself, after heavy churn and on every layer.

use std::collections::BTreeMap;

use kleos::{DataValue, DbInstance, NamedRows, ScriptMutability};

fn run(db: &DbInstance, script: &str) -> NamedRows {
    db.run_script(script, BTreeMap::new(), ScriptMutability::Mutable)
        .unwrap_or_else(|e| panic!("script failed: {e:?}\n{script}"))
}

/// Deterministic vectors in [0, 1)^4 (a 64-bit LCG), so failures reproduce.
fn vectors(n: usize, seed: u64) -> Vec<[f64; 4]> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n).map(|_| [next(), next(), next(), next()]).collect()
}

fn vec_lit(v: &[f64; 4]) -> String {
    format!("vec([{}, {}, {}, {}])", v[0], v[1], v[2], v[3])
}

fn unreachable_after_churn(seed: u64, delete_via_import: bool) -> Vec<usize> {
    const N: usize = 200;
    let pts = vectors(N, seed);
    let db = DbInstance::new("mem", "", "").unwrap();
    run(&db, ":create pts { id: Int => emb: <F32; 4> }");
    run(
        &db,
        "::hnsw create pts:idx { dim: 4, m: 6, dtype: F32, fields: [emb], \
         distance: L2, ef_construction: 32 }",
    );
    // Incremental inserts, so removal meets graphs built by the write path.
    for (i, v) in pts.iter().enumerate() {
        run(
            &db,
            &format!(
                "?[id, emb] <- [[{i}, {}]] :put pts {{id => emb}}",
                vec_lit(v)
            ),
        );
    }
    // Delete 60%: every id not divisible by 5, plus every tenth survivor.
    let doomed: Vec<usize> = (0..N).filter(|i| i % 5 != 0 || i % 50 == 0).collect();
    if delete_via_import {
        let rows = doomed
            .iter()
            .map(|&i| vec![DataValue::from(i as i64)])
            .collect();
        let mut data = BTreeMap::new();
        data.insert("-pts".to_string(), NamedRows::new(vec!["id".into()], rows));
        db.import_relations(data).unwrap();
    } else {
        for &i in &doomed {
            run(&db, &format!("?[id] <- [[{i}]] :rm pts {{id}}"));
        }
    }
    (0..N)
        .filter(|i| !doomed.contains(i))
        .filter(|&i| {
            let found = run(
                &db,
                &format!(
                    "?[id] := ~pts:idx{{id | query: {}, k: 1, ef: 64}}",
                    vec_lit(&pts[i])
                ),
            );
            found.rows != vec![vec![DataValue::from(i as i64)]]
        })
        .collect()
}

#[test]
fn survivors_stay_reachable_after_rm_churn() {
    for seed in [1, 2, 3] {
        assert_eq!(
            unreachable_after_churn(seed, false),
            Vec::<usize>::new(),
            "seed {seed}"
        );
    }
}

#[test]
fn survivors_stay_reachable_after_import_delete_churn() {
    for seed in [4, 5, 6] {
        assert_eq!(
            unreachable_after_churn(seed, true),
            Vec::<usize>::new(),
            "seed {seed}"
        );
    }
}
