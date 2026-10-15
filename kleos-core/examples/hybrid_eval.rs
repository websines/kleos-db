/*
 * Copyright 2026, Shan Rizvi (mnestic fork).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! BEIR SciFact evaluation of kleos hybrid retrieval: BM25, vector, RRF,
//! weighted (convex) fusion, and each followed by a cross-encoder re-rank
//! served through the `Reranker` hook.
//!
//! ```text
//! cargo run --release --example hybrid_eval -- <data_dir>
//! ```
//!
//! `<data_dir>` holds what a preparation script wrote (see the fork's
//! UPGRADE_PLAN C1/C2 notes): `corpus.jsonl` (`{id,title,text}`),
//! `docs_emb.npy` / `queries_emb.npy` (f32, L2-normalised, row-aligned with
//! the corpus / `queries.tsv`), `queries.tsv` (`qid\tsplit\ttext`, split
//! `test`|`dev`) and `qrels.tsv` (`qid\tdocid\trel`).
//!
//! Re-ranking: the harness collects every (query, doc) pair any method sends
//! to the re-ranker, writes the ones missing from `<data_dir>/rerank_scores.tsv`
//! to `<data_dir>/rerank_candidates.tsv`, and runs `$RERANK_CMD` (a shell
//! command that appends `qid\tdocid\tscore` lines for them, computed by a real
//! cross-encoder) before evaluating. The scores are then served through a
//! `Reranker` registered on the database, so the `Rerank` stage runs
//! end-to-end inside `hybrid_search`; a pair the table lacks is a hard error.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use kleos::{
    DataValue, DbInstance, Fusion, HybridSearch, RerankParams, Reranker, ScoreNorm,
    ScriptMutability, Vector, WeightedFusionParams,
};
use miette::{bail, IntoDiagnostic, Result};

const K_LEG: usize = 100; // per-leg candidates
const TOP: usize = 100; // evaluated depth (Recall@100)
const RERANK_N: usize = 100; // re-ranked candidates

fn load_npy(path: &Path) -> Result<(usize, usize, Vec<f32>)> {
    let bytes = fs::read(path).into_diagnostic()?;
    if &bytes[..6] != b"\x93NUMPY" {
        bail!("{path:?} is not an .npy file");
    }
    let (hlen, off) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        _ => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
    };
    let header = std::str::from_utf8(&bytes[off..off + hlen]).into_diagnostic()?;
    if !header.contains("'<f4'") || header.contains("'fortran_order': True") {
        bail!("{path:?}: expected a C-order <f4 array, header {header}");
    }
    let shape = &header[header.find("'shape': (").unwrap() + 10..];
    let shape = &shape[..shape.find(')').unwrap()];
    let dims: Vec<usize> = shape
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let data: Vec<f32> = bytes[off + hlen..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(data.len(), dims[0] * dims[1]);
    Ok((dims[0], dims[1], data))
}

fn text_hash(s: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

struct Query {
    qid: String,
    split: String,
    text: String,
    fts: String,
    emb: Vec<f32>,
}

/// FTS expression for a natural-language query: its alphanumeric tokens,
/// OR-ed (`,`), deduplicated. The index's filters lowercase/stem/stopword them.
fn fts_expr(text: &str) -> String {
    let mut seen = HashSet::new();
    let toks: Vec<String> = text
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_ascii_lowercase())
        .filter(|t| seen.insert(t.clone()))
        .collect();
    toks.join(", ")
}

/// Serves precomputed cross-encoder scores. Keyed by (query text, hash of the
/// doc text the database hands over), so it exercises the real hook: the
/// `Rerank` rule passes the stored `text` column, not ids.
struct TableReranker {
    scores: HashMap<(String, u64), f32>,
    misses: std::sync::atomic::AtomicUsize,
}

impl Reranker for TableReranker {
    fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        docs.iter()
            .map(|d| match self.scores.get(&(query.to_string(), text_hash(d))) {
                Some(s) => Ok(*s),
                None => {
                    self.misses
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    bail!("no precomputed cross-encoder score for this (query, doc) pair")
                }
            })
            .collect()
    }
}

#[derive(Clone)]
enum Method {
    Bm25,
    Vector,
    Rrf,
    Weighted { alpha: f64, theory: bool },
}

impl Method {
    fn name(&self) -> String {
        match self {
            Method::Bm25 => "BM25".into(),
            Method::Vector => "vector".into(),
            Method::Rrf => "RRF (k=60)".into(),
            Method::Weighted { alpha, theory } => format!(
                "weighted {} (α_vec={alpha:.1})",
                if *theory { "theory-norm" } else { "min-max" }
            ),
        }
    }
}

fn build(m: &Method, q: &Query, rerank: bool) -> HybridSearch {
    let mut h = HybridSearch::default();
    h.relation = "docs".into();
    h.limit = TOP;
    let vec_leg = |h: &mut HybridSearch| {
        h.vector_index = Some("vec".into());
        h.query_vector = q.emb.clone();
        h.vector_k = K_LEG;
        h.ef = 200;
    };
    let fts_leg = |h: &mut HybridSearch| {
        h.fts_index = Some("fts".into());
        h.query_text = q.fts.clone();
        h.fts_k = K_LEG;
    };
    match m {
        Method::Bm25 => fts_leg(&mut h),
        Method::Vector => vec_leg(&mut h),
        Method::Rrf => {
            vec_leg(&mut h);
            fts_leg(&mut h);
        }
        Method::Weighted { alpha, theory } => {
            vec_leg(&mut h);
            fts_leg(&mut h);
            let mut p = WeightedFusionParams::new([("semantic", *alpha), ("text", 1.0 - alpha)]);
            if *theory {
                p = p
                    .with_leg_norm("semantic", ScoreNorm::NegCosineDistance)
                    .with_leg_norm("text", ScoreNorm::Bm25);
            }
            h.fusion = Fusion::Weighted(p);
        }
    }
    if rerank {
        let mut r = RerankParams::new("ce");
        r.top_n = RERANK_N;
        r.query = Some(q.text.clone());
        h.rerank = Some(r);
    }
    h
}

fn run_q(db: &DbInstance, h: &HybridSearch) -> Result<Vec<String>> {
    let res = db.hybrid_search(h)?;
    Ok(res
        .rows
        .iter()
        .map(|r| r[0].get_str().unwrap().to_string())
        .collect())
}

fn ndcg10(ranked: &[String], rel: &HashMap<String, u32>) -> f64 {
    let dcg: f64 = ranked
        .iter()
        .take(10)
        .enumerate()
        .map(|(i, d)| *rel.get(d).unwrap_or(&0) as f64 / ((i + 2) as f64).log2())
        .sum();
    let mut ideal: Vec<u32> = rel.values().copied().collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = ideal
        .iter()
        .take(10)
        .enumerate()
        .map(|(i, r)| *r as f64 / ((i + 2) as f64).log2())
        .sum();
    if idcg > 0.0 {
        dcg / idcg
    } else {
        0.0
    }
}

fn recall(ranked: &[String], rel: &HashMap<String, u32>, k: usize) -> f64 {
    let hit = ranked.iter().take(k).filter(|d| rel.contains_key(*d)).count();
    hit as f64 / rel.len() as f64
}

struct Eval {
    ndcg: f64,
    r100: f64,
    mean_ms: f64,
    p50_ms: f64,
}

fn evaluate(
    db: &DbInstance,
    m: &Method,
    qs: &[&Query],
    qrels: &HashMap<String, HashMap<String, u32>>,
    rerank: bool,
) -> Result<(Eval, Vec<Vec<String>>)> {
    let (mut n, mut r) = (0.0, 0.0);
    let mut lat = Vec::with_capacity(qs.len());
    let mut runs = Vec::with_capacity(qs.len());
    for q in qs {
        let h = build(m, q, rerank);
        let t = Instant::now();
        let ranked = run_q(db, &h)?;
        lat.push(t.elapsed().as_secs_f64() * 1e3);
        let rel = &qrels[&q.qid];
        n += ndcg10(&ranked, rel);
        r += recall(&ranked, rel, TOP);
        runs.push(ranked);
    }
    let nq = qs.len() as f64;
    let mean_ms = lat.iter().sum::<f64>() / nq;
    lat.sort_by(|a, b| a.total_cmp(b));
    Ok((
        Eval {
            ndcg: n / nq,
            r100: r / nq,
            mean_ms,
            p50_ms: lat[lat.len() / 2],
        },
        runs,
    ))
}

/// Latency of the two legs *without* fusion (their union), to isolate the
/// fusion operator's own cost.
fn legs_only_ms(db: &DbInstance, qs: &[&Query]) -> Result<f64> {
    let mut total = 0.0;
    for q in qs {
        let h = build(&Method::Rrf, q, false);
        let (script, params) = kleos::build_hybrid_query(&h)?;
        let cut = script.find("?[id, score] <~").unwrap();
        let script = format!("{}?[lid, id, score] := combined[lid, id, score]", &script[..cut]);
        let t = Instant::now();
        db.run_script(&script, params, ScriptMutability::Immutable)?;
        total += t.elapsed().as_secs_f64() * 1e3;
    }
    Ok(total / qs.len() as f64)
}

fn main() -> Result<()> {
    let dir = PathBuf::from(std::env::args().nth(1).expect("usage: hybrid_eval <data_dir>"));

    // ---- load data
    let mut doc_ids = Vec::new();
    let mut doc_texts = Vec::new();
    for line in BufReader::new(fs::File::open(dir.join("corpus.jsonl")).into_diagnostic()?).lines()
    {
        let v: serde_json::Value = serde_json::from_str(&line.into_diagnostic()?).into_diagnostic()?;
        let title = v["title"].as_str().unwrap_or("");
        let text = v["text"].as_str().unwrap_or("");
        doc_ids.push(v["id"].as_str().unwrap().to_string());
        doc_texts.push(format!("{title} {text}").trim().to_string());
    }
    let (nd, dim, demb) = load_npy(&dir.join("docs_emb.npy"))?;
    assert_eq!(nd, doc_ids.len());
    let (nq, qdim, qemb) = load_npy(&dir.join("queries_emb.npy"))?;
    assert_eq!(qdim, dim);
    let mut queries = Vec::new();
    for (i, line) in fs::read_to_string(dir.join("queries.tsv"))
        .into_diagnostic()?
        .lines()
        .enumerate()
    {
        let mut it = line.splitn(3, '\t');
        let (qid, split, text) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
        queries.push(Query {
            qid: qid.into(),
            split: split.into(),
            text: text.into(),
            fts: fts_expr(text),
            emb: qemb[i * dim..(i + 1) * dim].to_vec(),
        });
    }
    assert_eq!(nq, queries.len());
    let mut qrels: HashMap<String, HashMap<String, u32>> = HashMap::new();
    for line in fs::read_to_string(dir.join("qrels.tsv")).into_diagnostic()?.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        qrels
            .entry(f[0].into())
            .or_default()
            .insert(f[1].into(), f[2].parse().unwrap());
    }
    let test: Vec<&Query> = queries.iter().filter(|q| q.split == "test").collect();
    let dev: Vec<&Query> = queries.iter().filter(|q| q.split == "dev").collect();
    println!(
        "SciFact: {} docs (dim {dim}), {} test queries, {} dev queries",
        nd,
        test.len(),
        dev.len()
    );

    // ---- load into kleos: relation + HNSW (cosine) + FTS (BM25)
    let db = DbInstance::new("mem", "", "")?;
    db.run_script(
        &format!(":create docs {{ id: String => text: String, emb: <F32; {dim}> }}"),
        Default::default(),
        ScriptMutability::Mutable,
    )?;
    let t = Instant::now();
    for start in (0..nd).step_by(500) {
        let rows: Vec<DataValue> = (start..(start + 500).min(nd))
            .map(|i| {
                let v = ndarray::Array1::from(demb[i * dim..(i + 1) * dim].to_vec());
                DataValue::List(vec![
                    DataValue::from(doc_ids[i].as_str()),
                    DataValue::from(doc_texts[i].as_str()),
                    DataValue::Vec(Vector::F32(v)),
                ])
            })
            .collect();
        db.run_script(
            "?[id, text, emb] <- $rows :put docs { id => text, emb }",
            BTreeMap::from([("rows".to_string(), DataValue::List(rows))]),
            ScriptMutability::Mutable,
        )?;
    }
    db.run_script(
        &format!(
            "::hnsw create docs:vec {{ dim: {dim}, m: 32, dtype: F32, fields: [emb], \
             distance: Cosine, ef_construction: 200 }}"
        ),
        Default::default(),
        ScriptMutability::Mutable,
    )?;
    db.run_script(
        "::fts create docs:fts { extractor: text, tokenizer: Simple, \
         filters: [Lowercase, Stemmer('English'), Stopwords('en')] }",
        Default::default(),
        ScriptMutability::Mutable,
    )?;
    println!("loaded + indexed in {:.1}s", t.elapsed().as_secs_f64());

    // ---- weighted fusion: pick α on the dev split (never on test)
    let alphas: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
    let mut best: BTreeMap<bool, (f64, f64)> = BTreeMap::new(); // theory -> (α, ndcg)
    let mut sweep_rows = Vec::new();
    for &theory in &[false, true] {
        for &alpha in &alphas {
            let m = Method::Weighted { alpha, theory };
            let (e, _) = evaluate(&db, &m, &dev, &qrels, false)?;
            sweep_rows.push((theory, alpha, e.ndcg, e.r100));
            let b = best.entry(theory).or_insert((alpha, -1.0));
            if e.ndcg > b.1 {
                *b = (alpha, e.ndcg);
            }
        }
    }
    println!("\n## Weighted-fusion sweep on dev ({} train-split queries)\n", dev.len());
    println!("| α_vec | min-max nDCG@10 | min-max R@100 | theory nDCG@10 | theory R@100 |");
    println!("|---|---|---|---|---|");
    for &alpha in &alphas {
        let f = |th: bool| {
            sweep_rows
                .iter()
                .find(|r| r.0 == th && r.1 == alpha)
                .map(|r| (r.2, r.3))
                .unwrap()
        };
        let (a, b) = (f(false), f(true));
        println!(
            "| {alpha:.1} | {:.4} | {:.4} | {:.4} | {:.4} |",
            a.0, a.1, b.0, b.1
        );
    }
    let rrf_dev = evaluate(&db, &Method::Rrf, &dev, &qrels, false)?.0;
    println!("| RRF k=60 (ref) | {:.4} | {:.4} | | |", rrf_dev.ndcg, rrf_dev.r100);

    let methods = vec![
        Method::Bm25,
        Method::Vector,
        Method::Rrf,
        Method::Weighted {
            alpha: best[&false].0,
            theory: false,
        },
        Method::Weighted {
            alpha: best[&true].0,
            theory: true,
        },
    ];

    // warm-up (page cache, allocator) so the first method's latency is fair
    for q in test.iter().take(20) {
        run_q(&db, &build(&Method::Rrf, q, false))?;
    }

    // ---- first stage on test
    let mut first = Vec::new();
    let mut pairs: BTreeMap<String, HashSet<String>> = BTreeMap::new();
    for m in &methods {
        let (e, runs) = evaluate(&db, m, &test, &qrels, false)?;
        for (q, ranked) in test.iter().zip(&runs) {
            pairs
                .entry(q.qid.clone())
                .or_default()
                .extend(ranked.iter().take(RERANK_N).cloned());
        }
        first.push(e);
    }

    // ---- cross-encoder scores for every pair any method re-ranks
    let scores_path = dir.join("rerank_scores.tsv");
    let mut have: HashSet<(String, String)> = HashSet::new();
    if scores_path.exists() {
        for line in fs::read_to_string(&scores_path).into_diagnostic()?.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            have.insert((f[0].into(), f[1].into()));
        }
    }
    let missing: Vec<(String, String)> = pairs
        .iter()
        .flat_map(|(q, ds)| ds.iter().map(move |d| (q.clone(), d.clone())))
        .filter(|p| !have.contains(p))
        .collect();
    let total_pairs: usize = pairs.values().map(|s| s.len()).sum();
    println!("\nre-rank pairs: {total_pairs} ({} missing scores)", missing.len());
    if !missing.is_empty() {
        let cand = dir.join("rerank_candidates.tsv");
        let mut f = fs::File::create(&cand).into_diagnostic()?;
        for (q, d) in &missing {
            writeln!(f, "{q}\t{d}").into_diagnostic()?;
        }
        drop(f);
        let cmd = std::env::var("RERANK_CMD")
            .expect("missing cross-encoder scores: set RERANK_CMD to a command that scores rerank_candidates.tsv");
        let t = Instant::now();
        let st = std::process::Command::new("sh")
            .arg("-c")
            .arg(&cmd)
            .status()
            .into_diagnostic()?;
        if !st.success() {
            bail!("RERANK_CMD failed: {st}");
        }
        println!("cross-encoder scored {} pairs in {:.0}s", missing.len(), t.elapsed().as_secs_f64());
    }
    let doc_hash: HashMap<&str, u64> = doc_ids
        .iter()
        .zip(&doc_texts)
        .map(|(id, t)| (id.as_str(), text_hash(t)))
        .collect();
    let qtext: HashMap<&str, &str> = queries.iter().map(|q| (q.qid.as_str(), q.text.as_str())).collect();
    let mut table = HashMap::new();
    for line in fs::read_to_string(&scores_path).into_diagnostic()?.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        table.insert(
            (qtext[f[0]].to_string(), doc_hash[f[1]]),
            f[2].parse::<f32>().into_diagnostic()?,
        );
    }
    let reranker = Arc::new(TableReranker {
        scores: table,
        misses: Default::default(),
    });
    db.register_reranker("ce", reranker.clone())?;

    // ---- with re-rank
    let mut reranked = Vec::new();
    for m in &methods {
        reranked.push(evaluate(&db, m, &test, &qrels, true)?.0);
    }
    assert_eq!(
        reranker.misses.load(std::sync::atomic::Ordering::Relaxed),
        0
    );

    println!(
        "\n## SciFact test ({} queries); per-leg k={K_LEG}, re-rank top {RERANK_N}\n",
        test.len()
    );
    println!(
        "| method | nDCG@10 | R@100 | +rerank nDCG@10 | +rerank R@100 | ms mean / p50 | +rerank ms mean (table lookup) |"
    );
    println!("|---|---|---|---|---|---|---|");
    for ((m, a), b) in methods.iter().zip(&first).zip(&reranked) {
        println!(
            "| {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.2} / {:.2} | {:.2} |",
            m.name(),
            a.ndcg,
            a.r100,
            b.ndcg,
            b.r100,
            a.mean_ms,
            a.p50_ms,
            b.mean_ms
        );
    }

    // ---- fusion overhead: both legs unioned without a fusion operator vs
    // RRF vs weighted, same queries, interleaved repeats.
    let mut legs = 0.0;
    let mut rrf = 0.0;
    let mut wf = 0.0;
    let reps = 3;
    for _ in 0..reps {
        legs += legs_only_ms(&db, &test)?;
        rrf += evaluate(&db, &Method::Rrf, &test, &qrels, false)?.0.mean_ms;
        wf += evaluate(&db, &methods[4], &test, &qrels, false)?.0.mean_ms;
    }
    let (legs, rrf, wf) = (legs / reps as f64, rrf / reps as f64, wf / reps as f64);
    println!("\n## Fusion latency (mean per query, {reps} passes over test)\n");
    println!("| pipeline | ms | overhead vs legs-only |");
    println!("|---|---|---|");
    println!("| both legs, no fusion | {legs:.2} | — |");
    println!("| + RRF | {rrf:.2} | {:+.2} |", rrf - legs);
    println!("| + WeightedFusion (theory norms) | {wf:.2} | {:+.2} |", wf - legs);
    Ok(())
}
