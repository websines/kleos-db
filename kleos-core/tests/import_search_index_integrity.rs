/*
 * Copyright 2026, The kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Bulk import may leave rows *unindexed* (the documented fast path: warn, then
//! `::reindex`), but it must never leave HNSW / FTS / LSH entries that point at
//! rows it deleted or replaced. A dangling HNSW node turned every later vector
//! search into "Cannot find compound key for HNSW" (upstream CozoDB #232), and
//! stale FTS/LSH postings returned rows for text they no longer contain.

use std::collections::BTreeMap;

use kleos::{DataValue, DbInstance, NamedRows, ScriptMutability};

fn run(db: &DbInstance, script: &str) -> NamedRows {
    db.run_script(script, BTreeMap::new(), ScriptMutability::Mutable)
        .unwrap_or_else(|e| panic!("script failed: {e:?}\n{script}"))
}

fn ids(rows: &NamedRows) -> Vec<i64> {
    let mut v: Vec<i64> = rows.rows.iter().map(|r| r[0].get_int().unwrap()).collect();
    v.sort();
    v
}

fn import(db: &DbInstance, rel: &str, headers: &[&str], rows: Vec<Vec<DataValue>>) {
    let mut data = BTreeMap::new();
    data.insert(
        rel.to_string(),
        NamedRows::new(headers.iter().map(|h| h.to_string()).collect(), rows),
    );
    db.import_relations(data).unwrap();
}

fn vec2(x: f32, y: f32) -> DataValue {
    DataValue::Vec(kleos::Vector::F32(ndarray::Array1::from(vec![x, y])))
}

fn points_db() -> DbInstance {
    let db = DbInstance::new("mem", "", "").unwrap();
    run(&db, ":create pts { id: Int => emb: <F32; 2> }");
    run(
        &db,
        "?[id, emb] <- [[1, vec([1.0, 0.0])], [2, vec([0.0, 1.0])], [3, vec([1.0, 1.0])]] \
         :put pts {id => emb}",
    );
    run(
        &db,
        "::hnsw create pts:idx { dim: 2, m: 8, dtype: F32, fields: [emb], \
         distance: L2, ef_construction: 32 }",
    );
    db
}

const KNN: &str = "?[id] := ~pts:idx{id | query: vec([1.0, 0.0]), k: 10, ef: 32}";

#[test]
fn import_delete_keeps_vector_search_working() {
    let db = points_db();
    import(&db, "-pts", &["id"], vec![vec![DataValue::from(1)]]);

    assert_eq!(ids(&run(&db, KNN)), vec![2, 3]);

    // Writes traverse the graph too, so they must not trip over the deletion.
    run(
        &db,
        "?[id, emb] <- [[4, vec([0.5, 0.5])]] :put pts {id => emb}",
    );
    assert_eq!(ids(&run(&db, KNN)), vec![2, 3, 4]);
}

#[test]
fn import_overwrite_drops_the_stale_vector_until_reindex() {
    let db = points_db();
    import(
        &db,
        "pts",
        &["id", "emb"],
        vec![vec![DataValue::from(2), vec2(0.9, 0.0)]],
    );

    // The replaced row is unindexed (documented), not indexed under its old vector.
    assert_eq!(ids(&run(&db, KNN)), vec![1, 3]);
    run(&db, "::reindex pts");
    assert_eq!(ids(&run(&db, KNN)), vec![1, 2, 3]);
}

#[test]
fn import_reput_of_an_unchanged_row_keeps_it_indexed() {
    let db = points_db();
    import(
        &db,
        "pts",
        &["id", "emb"],
        vec![vec![DataValue::from(2), vec2(0.0, 1.0)]],
    );
    assert_eq!(ids(&run(&db, KNN)), vec![1, 2, 3]);
}

fn docs_db() -> DbInstance {
    let db = DbInstance::new("mem", "", "").unwrap();
    run(&db, ":create doc { id: Int => body: String }");
    run(
        &db,
        "?[id, body] <- [[1, 'apple pie'], [2, 'banana bread']] :put doc {id => body}",
    );
    run(
        &db,
        "::fts create doc:fts { extractor: body, tokenizer: Simple, filters: [Lowercase] }",
    );
    run(
        &db,
        "::lsh create doc:lsh { extractor: body, tokenizer: Simple, filters: [Lowercase], \
         n_perm: 64, target_threshold: 0.5, n_gram: 3 }",
    );
    db
}

fn fts(db: &DbInstance, q: &str) -> Vec<i64> {
    ids(&run(
        db,
        &format!("?[id] := ~doc:fts{{id | query: '{q}', k: 10}}"),
    ))
}

fn lsh(db: &DbInstance, q: &str) -> Vec<i64> {
    ids(&run(
        db,
        &format!("?[id] := ~doc:lsh{{id | query: '{q}', k: 10}}"),
    ))
}

#[test]
fn import_delete_removes_text_postings() {
    let db = docs_db();
    assert_eq!(fts(&db, "apple"), vec![1]);
    assert_eq!(lsh(&db, "apple pie"), vec![1]);

    import(&db, "-doc", &["id"], vec![vec![DataValue::from(1)]]);

    assert_eq!(fts(&db, "apple"), Vec::<i64>::new());
    assert_eq!(lsh(&db, "apple pie"), Vec::<i64>::new());
    assert_eq!(fts(&db, "banana"), vec![2]);
}

#[test]
fn import_overwrite_removes_stale_text_postings() {
    let db = docs_db();
    import(
        &db,
        "doc",
        &["id", "body"],
        vec![vec![DataValue::from(2), DataValue::from("cherry tart")]],
    );

    assert_eq!(fts(&db, "banana"), Vec::<i64>::new());
    assert_eq!(lsh(&db, "banana bread"), Vec::<i64>::new());
    run(&db, "::reindex doc");
    assert_eq!(fts(&db, "cherry"), vec![2]);
}

#[test]
fn backup_restore_over_a_row_removes_its_stale_postings() {
    let dir = tempfile::tempdir().unwrap();
    let backup = dir.path().join("backup.db");

    let src = DbInstance::new("mem", "", "").unwrap();
    run(&src, ":create doc { id: Int => body: String }");
    run(
        &src,
        "?[id, body] <- [[2, 'cherry tart']] :put doc {id => body}",
    );
    src.backup_db(backup.to_str().unwrap()).unwrap();

    let db = docs_db();
    db.import_from_backup(backup.to_str().unwrap(), &["doc".to_string()])
        .unwrap();

    assert_eq!(fts(&db, "banana"), Vec::<i64>::new());
    assert_eq!(lsh(&db, "banana bread"), Vec::<i64>::new());
    assert_eq!(fts(&db, "apple"), vec![1]);
}
