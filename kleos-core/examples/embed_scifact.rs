/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! End-to-end retrieval through kleos (UPGRADE_PLAN D3): auto-embed a BEIR
//! corpus on `:put` into an HNSW-indexed relation, then answer every test
//! query with `embed()` + HNSW search, and score nDCG@10.
//!
//! cargo run --release -p kleos --features infer-metal --example embed_scifact -- \
//!     <dir with docs.jsonl, queries.jsonl, meta.json> <gemma|potion-8M|potion-32M> [precision] [device] [dim]
//!
//! docs.jsonl / queries.jsonl: one JSON string per line; meta.json:
//! {"doc_ids": [...], "query_ids": [...], "qrels": {qid: {docid: grade}}}.

use std::collections::BTreeMap;
use std::time::Instant;

use kleos::embed::builtin::{
    register_embeddinggemma, register_model2vec, DevicePref, EngineConfig, Precision,
};
use kleos::embed::{embedding_cache_stats, registered_embedders};
use kleos::{DataValue, DbInstance, ScriptMutability};
use serde_json::Value;

fn lines(path: &str) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<String>(l).unwrap())
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = &a[1];
    let model = a[2].as_str();
    let precision = Precision::parse(a.get(3).map(|s| s.as_str()).unwrap_or("q8")).unwrap();
    let device = DevicePref::parse(a.get(4).map(|s| s.as_str()).unwrap_or("auto")).unwrap();
    let dim: Option<usize> = a.get(5).map(|s| s.parse().unwrap());
    let engine = EngineConfig {
        device,
        precision,
        ..Default::default()
    };
    let (doc_m, query_m) = match model {
        "gemma" => {
            register_embeddinggemma("m", engine, dim).unwrap();
            ("m", "m_query")
        }
        "potion-8M" => {
            register_model2vec("m", "minishlab/potion-base-8M").unwrap();
            ("m", "m")
        }
        "potion-32M" => {
            register_model2vec("m", "minishlab/potion-base-32M").unwrap();
            ("m", "m")
        }
        other => panic!("unknown model {other}"),
    };
    let d = registered_embedders()
        .into_iter()
        .find(|(n, _)| n == doc_m)
        .unwrap()
        .1;
    let docs = lines(&format!("{dir}/docs.jsonl"));
    let queries = lines(&format!("{dir}/queries.jsonl"));
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json")).unwrap()).unwrap();
    let doc_ids: Vec<&str> = meta["doc_ids"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let query_ids: Vec<&str> = meta["query_ids"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();

    let db = DbInstance::new("mem", "", "").unwrap();
    let run = |s: &str, p: BTreeMap<String, DataValue>| {
        db.run_script(s, p, ScriptMutability::Mutable)
            .unwrap_or_else(|e| panic!("{e:?}\n{s}"))
    };
    run(
        &format!(":create docs {{ id: Int => did: String, text: String, vec: <F32; {d}> default embed('{doc_m}', text) }}"),
        Default::default(),
    );
    run(
        &format!("::hnsw create docs:v {{ dim: {d}, m: 32, dtype: F32, fields: [vec], distance: Cosine, ef_construction: 200 }}"),
        Default::default(),
    );

    // warm the model (lazy load) outside the timed ingest
    let t = Instant::now();
    run(&format!("?[v] := v = embed('{doc_m}', 'warm up')"), Default::default());
    let load_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    for (c, chunk) in docs.chunks(512).enumerate() {
        let rows: Vec<DataValue> = chunk
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let id = c * 512 + i;
                DataValue::List(vec![
                    DataValue::from(id as i64),
                    DataValue::from(doc_ids[id]),
                    DataValue::from(text.as_str()),
                ])
            })
            .collect();
        run(
            "?[id, did, text] <- $rows :put docs { id => did, text }",
            BTreeMap::from([("rows".to_string(), DataValue::List(rows))]),
        );
    }
    let ingest_s = t.elapsed().as_secs_f64();

    let search = format!(
        "?[did, dist] := ~docs:v{{ did | query: embed('{query_m}', $q), k: 10, ef: 100, bind_distance: dist }} :order dist"
    );
    let mut ndcg_sum = 0.0;
    let t = Instant::now();
    for (qi, q) in queries.iter().enumerate() {
        let res = run(&search, BTreeMap::from([("q".to_string(), DataValue::from(q.as_str()))]));
        let ranked: Vec<String> = res.rows.iter().map(|r| r[0].get_str().unwrap().to_string()).collect();
        let rel = &meta["qrels"][query_ids[qi]];
        let gain = |doc: &str| rel.get(doc).and_then(|v| v.as_f64()).unwrap_or(0.0);
        let dcg: f64 = ranked.iter().enumerate().map(|(i, dd)| (2f64.powf(gain(dd)) - 1.0) / ((i + 2) as f64).log2()).sum();
        let mut ideal: Vec<f64> = rel.as_object().unwrap().values().map(|v| v.as_f64().unwrap()).collect();
        ideal.sort_by(|x, y| y.partial_cmp(x).unwrap());
        let idcg: f64 = ideal.iter().take(10).enumerate().map(|(i, g)| (2f64.powf(*g) - 1.0) / ((i + 2) as f64).log2()).sum();
        ndcg_sum += if idcg > 0.0 { dcg / idcg } else { 0.0 };
    }
    let query_s = t.elapsed().as_secs_f64();
    // repeat: every query embedding now comes from the cache
    let t = Instant::now();
    for q in &queries {
        run(&search, BTreeMap::from([("q".to_string(), DataValue::from(q.as_str()))]));
    }
    let cached_s = t.elapsed().as_secs_f64();
    let st = embedding_cache_stats();
    println!(
        "{{\"model\":\"{model}\",\"precision\":\"{}\",\"device\":\"{:?}\",\"dim\":{d},\"docs\":{},\"queries\":{},\"load_s\":{load_s:.2},\"ingest_s\":{ingest_s:.2},\"ingest_docs_per_s\":{:.1},\"ndcg@10\":{:.4},\"query_ms\":{:.2},\"query_ms_cached\":{:.2},\"cache_hits\":{},\"cache_misses\":{}}}",
        precision.name(), device, docs.len(), queries.len(),
        docs.len() as f64 / ingest_s, ndcg_sum / queries.len() as f64,
        query_s * 1000.0 / queries.len() as f64, cached_s * 1000.0 / queries.len() as f64,
        st.hits, st.misses
    );
}
