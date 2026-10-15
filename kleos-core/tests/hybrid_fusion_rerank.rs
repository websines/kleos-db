/*
 * Copyright 2026, Shan Rizvi (mnestic fork).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Weighted (convex) score fusion and the pluggable re-rank stage (kleos fork,
//! UPGRADE_PLAN C1/C2): the `WeightedFusion` and `Rerank` fixed rules, and
//! their `HybridSearch` options. sqlite backend per the fork's test convention.
#![allow(clippy::field_reassign_with_default)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use kleos::{
    DbInstance, Fusion, HybridSearch, MmrParams, NamedRows, RerankParams, Reranker, ScoreNorm,
    ScriptMutability, WeightedFusionParams,
};

fn run(db: &DbInstance, s: &str) -> NamedRows {
    db.run_script(s, BTreeMap::new(), ScriptMutability::Mutable)
        .unwrap_or_else(|e| panic!("script failed: {e:?}\n--- script ---\n{s}"))
}

fn run_err(db: &DbInstance, s: &str) -> String {
    match db.run_script(s, BTreeMap::new(), ScriptMutability::Mutable) {
        Ok(r) => panic!("expected an error, got {r:?}\n--- script ---\n{s}"),
        Err(e) => format!("{e:?}"),
    }
}

fn make_db() -> (DbInstance, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fusion.db");
    let db = DbInstance::new("sqlite", path.to_str().unwrap(), "").unwrap();
    run(
        &db,
        r#"
        ?[id, text, emb] <- [
            ['d1', 'the cat sat on the mat', [1.0, 0.0]],
            ['d2', 'a dog ran in the park',  [0.9, 0.1]],
            ['d3', 'cats and dogs are pets', [0.0, 1.0]],
            ['d4', 'the weather is nice',    [0.1, 0.9]]
        ]
        :create docs { id: String => text: String, emb: <F32; 2> }
    "#,
    );
    run(
        &db,
        r#"::hnsw create docs:vec {
            dim: 2, m: 50, dtype: F32, fields: [emb],
            distance: Cosine, ef_construction: 20
        }"#,
    );
    run(
        &db,
        r#"::fts create docs:fts {
            extractor: text, tokenizer: Simple,
            filters: [Lowercase, Stemmer('English'), Stopwords('en')]
        }"#,
    );
    (db, dir)
}

fn base_query() -> HybridSearch {
    let mut q = HybridSearch::default();
    q.relation = "docs".into();
    q.vector_index = Some("vec".into());
    q.query_vector = vec![1.0, 0.0];
    q.fts_index = Some("fts".into());
    q.query_text = "cat".into();
    q
}

fn scores(res: &NamedRows) -> BTreeMap<String, f64> {
    res.rows
        .iter()
        .map(|r| {
            (
                r[0].get_str().unwrap().to_string(),
                r[1].get_float().unwrap(),
            )
        })
        .collect()
}

fn ids(res: &NamedRows) -> Vec<String> {
    res.rows
        .iter()
        .map(|r| r[0].get_str().unwrap().to_string())
        .collect()
}

const LISTS: &str = r#"
    l[lid, item, score] <- [
        ['a', 'x', 10.0], ['a', 'y', 5.0], ['a', 'z', 0.0],
        ['b', 'y', 4.0],  ['b', 'w', 2.0]
    ]
"#;

#[test]
fn weighted_fusion_minmax_convex_combination() {
    let (db, _dir) = make_db();
    let res = run(
        &db,
        &format!(
            "{LISTS}\n?[item, s] <~ WeightedFusion(l[lid, item, score], \
             weights: [['a', 0.25], ['b', 0.75]])"
        ),
    );
    let s = scores(&res);
    // a: x=1, y=.5, z=0 ; b: y=1, w=0 ; missing ⇒ 0
    assert!((s["x"] - 0.25).abs() < 1e-12);
    assert!((s["y"] - (0.25 * 0.5 + 0.75)).abs() < 1e-12);
    assert!((s["z"] - 0.0).abs() < 1e-12);
    assert!((s["w"] - 0.0).abs() < 1e-12);

    // no weights ⇒ equal 1/n
    let s = scores(&run(
        &db,
        &format!("{LISTS}\n?[item, s] <~ WeightedFusion(l[lid, item, score])"),
    ));
    assert!((s["y"] - 0.75).abs() < 1e-12);

    // theoretical bm25 norm keeps the weakest hit above zero
    let s = scores(&run(
        &db,
        &format!(
            "{LISTS}\n?[item, s] <~ WeightedFusion(l[lid, item, score], \
             weights: [['a', 0.5], ['b', 0.5]], norm: 'bm25')"
        ),
    ));
    assert!((s["w"] - 0.25).abs() < 1e-12, "{s:?}");

    // per-list override: 'b' as a fixed range [0, 8]
    let s = scores(&run(
        &db,
        &format!(
            "{LISTS}\n?[item, s] <~ WeightedFusion(l[lid, item, score], \
             weights: [['a', 0.0], ['b', 1.0]], norms: [['b', ['range', 0.0, 8.0]]])"
        ),
    ));
    assert!((s["y"] - 0.5).abs() < 1e-12 && (s["w"] - 0.25).abs() < 1e-12, "{s:?}");
}

#[test]
fn weighted_fusion_detailed_and_errors() {
    let (db, _dir) = make_db();
    let res = run(
        &db,
        &format!(
            "{LISTS}\n?[item, s, lid, rank, raw] <~ WeightedFusion(l[lid, item, score], \
             weights: [['a', 0.5], ['b', 0.5]], detailed: true)"
        ),
    );
    // y appears in both lists: two rows, ranks 2 (in a) and 1 (in b)
    let y: Vec<_> = res
        .rows
        .iter()
        .filter(|r| r[0].get_str() == Some("y"))
        .map(|r| (r[2].get_str().unwrap().to_string(), r[3].get_float().unwrap()))
        .collect();
    assert_eq!(y, vec![("a".to_string(), 2.0), ("b".to_string(), 1.0)]);

    // a list without a weight is loud...
    let e = run_err(
        &db,
        &format!("{LISTS}\n?[item, s] <~ WeightedFusion(l[lid, item, score], weights: [['a', 1.0]])"),
    );
    assert!(e.contains("has no weight"), "{e}");
    // ...unless default_weight covers it
    let s = scores(&run(
        &db,
        &format!(
            "{LISTS}\n?[item, s] <~ WeightedFusion(l[lid, item, score], \
             weights: [['a', 1.0]], default_weight: 0.0)"
        ),
    ));
    assert!((s["x"] - 1.0).abs() < 1e-12);
    let e = run_err(
        &db,
        &format!("{LISTS}\n?[item, s] <~ WeightedFusion(l[lid, item, score], norm: 'cubic')"),
    );
    assert!(e.contains("unknown normalization"), "{e}");
    let e = run_err(
        &db,
        &format!(
            "{LISTS}\n?[item, s] <~ WeightedFusion(l[lid, item, score], weights: [['a', -1.0], ['b', 1.0]])"
        ),
    );
    assert!(e.contains(">= 0"), "{e}");
}

#[test]
fn hybrid_weighted_fusion_extremes_match_single_legs() {
    let (db, _dir) = make_db();
    // weight 1 on the vector leg ⇒ the vector leg's own order
    let mut q = base_query();
    q.fusion = Fusion::Weighted(
        WeightedFusionParams::new([("semantic", 1.0), ("text", 0.0)])
            .with_leg_norm("semantic", ScoreNorm::NegCosineDistance)
            .with_leg_norm("text", ScoreNorm::Bm25),
    );
    let res = db.hybrid_search(&q).unwrap();
    assert_eq!(ids(&res)[..2], ["d1", "d2"]);
    // cosine distance 0 ⇒ normalized similarity exactly 1
    assert!((res.rows[0][1].get_float().unwrap() - 1.0).abs() < 1e-6);

    // weight 1 on the text leg ⇒ the only FTS hit (d1, "mat") first, and the
    // non-matching docs tie at 0
    let mut q = base_query();
    q.query_text = "mat".into();
    q.fusion = Fusion::Weighted(WeightedFusionParams::new([("semantic", 0.0), ("text", 1.0)]));
    let res = db.hybrid_search(&q).unwrap();
    assert_eq!(ids(&res)[0], "d1");
    assert!((res.rows[0][1].get_float().unwrap() - 1.0).abs() < 1e-12);

    // detailed mode keeps the 5-column shape
    let mut q = base_query();
    q.fusion = Fusion::Weighted(WeightedFusionParams::default());
    q.detailed = true;
    let res = db.hybrid_search(&q).unwrap();
    assert_eq!(
        res.headers,
        vec!["id", "score", "list_id", "leg_rank", "leg_score"]
    );
}

/// Scores a doc by how many times `query`'s words occur in it; counts calls.
struct WordCount {
    calls: AtomicUsize,
    docs_seen: AtomicUsize,
}

impl Reranker for WordCount {
    fn rerank(&self, query: &str, docs: &[&str]) -> miette::Result<Vec<f32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.docs_seen.fetch_add(docs.len(), Ordering::SeqCst);
        Ok(docs
            .iter()
            .map(|d| {
                query
                    .split_whitespace()
                    .map(|w| d.matches(w).count() as f32)
                    .sum()
            })
            .collect())
    }
}

struct Broken;

impl Reranker for Broken {
    fn rerank(&self, _query: &str, docs: &[&str]) -> miette::Result<Vec<f32>> {
        Ok(vec![0.0; docs.len().saturating_sub(1)])
    }
}

#[test]
fn rerank_fixed_rule_from_cozoscript() {
    let (db, _dir) = make_db();
    let wc = Arc::new(WordCount {
        calls: AtomicUsize::new(0),
        docs_seen: AtomicUsize::new(0),
    });
    db.register_reranker("wc", wc.clone()).unwrap();
    assert!(db.register_reranker("wc", wc.clone()).is_err());

    // all rows, no prior score
    let res = run(
        &db,
        "c[id, text] := *docs{id, text}\n\
         ?[id, s] <~ Rerank(c[id, text], model: 'wc', query: 'the weather')\n\
         :order -s",
    );
    assert_eq!(ids(&res)[0], "d4"); // 'the' + 'weather'
    assert_eq!(scores(&res)["d1"], 2.0); // 'the' twice

    // top_n picks by prior score; batch_size splits the call
    let before = wc.calls.load(Ordering::SeqCst);
    let res = run(
        &db,
        "c[id, text, p] := *docs{id, text}, p = if(id == 'd4', 0.0, 1.0)\n\
         ?[id, s] <~ Rerank(c[id, text, p], model: 'wc', query: 'the weather', top_n: 3, batch_size: 2)",
    );
    assert_eq!(res.rows.len(), 3);
    assert!(!ids(&res).contains(&"d4".to_string()));
    assert_eq!(wc.calls.load(Ordering::SeqCst) - before, 2);

    // duplicate items (chunks) keep the max
    let res = run(
        &db,
        "c[id, text] <- [['a', 'cat'], ['a', 'cat cat'], ['b', 'dog']]\n\
         ?[id, s] <~ Rerank(c[id, text], model: 'wc', query: 'cat')",
    );
    assert_eq!(scores(&res)["a"], 2.0);

    // errors are loud
    let e = run_err(
        &db,
        "c[id, text] := *docs{id, text}\n?[id, s] <~ Rerank(c[id, text], model: 'nope', query: 'x')",
    );
    assert!(e.contains("no reranker registered under 'nope'"), "{e}");
    let e = run_err(
        &db,
        "c[id, text] := *docs{id, text}\n?[id, s] <~ Rerank(c[id, text], model: 'wc', query: 'x', top_n: 2)",
    );
    assert!(e.contains("no prior-score"), "{e}");
    db.register_reranker("broken", Arc::new(Broken)).unwrap();
    let e = run_err(
        &db,
        "c[id, text] := *docs{id, text}\n?[id, s] <~ Rerank(c[id, text], model: 'broken', query: 'x')",
    );
    assert!(e.contains("returned 3 scores for 4 documents"), "{e}");

    // the per-Db rule is a builtin; the model can be unregistered
    assert!(db.unregister_fixed_rule("Rerank").is_err());
    assert!(db.unregister_reranker("broken"));
    assert!(!db.unregister_reranker("broken"));
}

#[test]
fn hybrid_search_rerank_stage() {
    let (db, _dir) = make_db();
    db.register_reranker(
        "wc",
        Arc::new(WordCount {
            calls: AtomicUsize::new(0),
            docs_seen: AtomicUsize::new(0),
        }),
    )
    .unwrap();

    // Fused order puts d1 first (vector + "cat"); the reranker scoring
    // "weather" lifts d4 — which only the vector leg's tail retrieved — to the top.
    let mut q = base_query();
    let mut rr = RerankParams::new("wc");
    rr.query = Some("weather".into());
    rr.top_n = 4;
    q.rerank = Some(rr);
    let res = db.hybrid_search(&q).unwrap();
    assert_eq!(ids(&res)[0], "d4");
    assert_eq!(res.rows.len(), 4);

    // top_n bounds what is returned
    let mut q2 = q.clone();
    q2.rerank.as_mut().unwrap().top_n = 2;
    let res = db.hybrid_search(&q2).unwrap();
    assert_eq!(res.rows.len(), 2);

    // rerank composes with weighted fusion, detailed mode and MMR
    let mut q3 = q.clone();
    q3.fusion = Fusion::Weighted(WeightedFusionParams::default());
    q3.detailed = true;
    let res = db.hybrid_search(&q3).unwrap();
    assert_eq!(res.rows[0][0].get_str(), Some("d4"));
    assert_eq!(res.rows[0][1].get_float(), Some(1.0)); // the model's score

    let mut q4 = q.clone();
    q4.mmr = Some(MmrParams {
        lambda: 1.0,
        k: 2,
        embedding_col: "emb".into(),
    });
    let res = db.hybrid_search(&q4).unwrap();
    assert_eq!(ids(&res)[0], "d4");

    // unregistered model: a runtime error naming the model
    let mut q5 = base_query();
    q5.rerank = Some(RerankParams::new("missing"));
    let e = format!("{:?}", db.hybrid_search(&q5).unwrap_err());
    assert!(e.contains("'missing'"), "{e}");
}
