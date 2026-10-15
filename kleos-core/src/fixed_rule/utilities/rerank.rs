/*
 * Copyright 2026, Shan Rizvi (mnestic fork).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Pluggable re-rank stage — a kleos fork addition for hybrid retrieval.
//!
//! First-stage recall (HNSW, BM25, graph legs, fusion) is cheap and broad; a
//! *cross-encoder* that reads the query and each candidate together is far
//! more precise but too slow to run over a corpus. The standard pipeline is
//! therefore "fuse, keep the top N, re-score those N with the cross-encoder".
//!
//! kleos ships **no model**: the host registers any scorer implementing
//! [`Reranker`] under a name ([`crate::Db::register_reranker`]) — a local
//! ONNX/candle/MLX cross-encoder, a remote rerank API, or a lookup table — and
//! scripts call it through the `Rerank` fixed rule:
//!
//! ```cozo
//! cand[id, text, score] := fused[id, score], *docs{ id, text }
//! ?[id, score] <~ Rerank(cand[id, text, score], model: 'bge', query: $q, top_n: 50)
//! ```
//!
//! Input: `[item, text]` or `[item, text, prior_score]`. With a prior score
//! and `top_n > 0`, only the `top_n` rows with the highest prior score are
//! sent to the model (the rest are dropped from the output). `text` must be a
//! string (`null` is treated as the empty string). If an item appears on
//! several rows (e.g. one row per chunk) each row is scored and the item keeps
//! its **maximum** score. Output: `[item, score]`, the model's score.
//!
//! Options: `model` (registered name, required), `query` (required),
//! `top_n` (default 0 = all rows), `batch_size` (default 0 = one call; a
//! positive value splits the call and checks for query cancellation between
//! batches).
//!
//! The model runs **inside** query evaluation, i.e. while the query's read
//! transaction is open. Keep `top_n` small; a slow remote model holds the
//! snapshot for its full latency.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::bail;

use crossbeam::sync::ShardedLock;
use miette::Result;
use smartstring::{LazyCompact, SmartString};

use crate::data::expr::Expr;
use crate::data::symb::Symbol;
use crate::data::value::DataValue;
use crate::fixed_rule::{FixedRule, FixedRulePayload};
use crate::parse::SourceSpan;
use crate::runtime::db::Poison;
use crate::runtime::temp_store::RegularTempStore;

/// A query–document relevance scorer (typically a cross-encoder) usable as a
/// re-rank stage. Register one with [`crate::Db::register_reranker`].
///
/// `rerank` receives the query and a batch of document texts and must return
/// exactly one **finite** score per document, in input order; higher means
/// more relevant. Scores need not be normalized. Implementations must be
/// thread-safe: concurrent queries may call the same reranker.
pub trait Reranker: Send + Sync {
    /// Score each of `docs` against `query`.
    fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>>;
}

/// The per-`Db` registry the `Rerank` fixed rule resolves `model:` against.
pub(crate) type RerankerRegistry = Arc<ShardedLock<BTreeMap<String, Arc<dyn Reranker>>>>;

/// Name under which the re-rank fixed rule is installed in every `Db`.
pub(crate) const RERANK_RULE_NAME: &str = "Rerank";

/// The `Rerank` fixed rule. Unlike the stateless builtins it holds a handle
/// on its `Db`'s reranker registry, so it is installed per `Db` at creation.
pub(crate) struct Rerank {
    pub(crate) registry: RerankerRegistry,
}

impl FixedRule for Rerank {
    fn run(
        &self,
        payload: FixedRulePayload<'_, '_>,
        out: &mut RegularTempStore,
        poison: Poison,
    ) -> Result<()> {
        let in_rel = payload.get_input(0)?;
        let model = payload.string_option("model", None)?;
        let query = payload.string_option("query", None)?;
        let top_n = payload.non_neg_integer_option("top_n", Some(0))?;
        let batch_size = payload.non_neg_integer_option("batch_size", Some(0))?;

        let found = self.registry.read().unwrap().get(model.as_str()).cloned();
        let reranker = match found {
            Some(r) => r,
            None => {
                let known: Vec<String> = self.registry.read().unwrap().keys().cloned().collect();
                bail!(
                    "Rerank: no reranker registered under '{model}' (registered: {known:?}); \
                     register one with Db::register_reranker"
                )
            }
        };

        // (item, text, prior)
        let mut rows: Vec<(DataValue, SmartString<LazyCompact>, Option<f64>)> = vec![];
        for tuple in in_rel.iter()? {
            let tuple = tuple?;
            if tuple.len() != 2 && tuple.len() != 3 {
                bail!(
                    "Rerank expects an input [item, text] or [item, text, prior_score], \
                     got a row of arity {}",
                    tuple.len()
                );
            }
            let mut it = tuple.into_iter();
            let item = it.next().unwrap();
            let text = match it.next().unwrap() {
                DataValue::Str(s) => s,
                DataValue::Null => SmartString::new(),
                v => bail!("Rerank: text (column 2) must be a string, got {v:?}"),
            };
            let prior = match it.next() {
                None => None,
                Some(v) => match v.get_float() {
                    Some(f) if f.is_finite() => Some(f),
                    _ => bail!("Rerank: prior score (column 3) must be a finite number, got {v:?}"),
                },
            };
            rows.push((item, text, prior));
            poison.check()?;
        }

        if top_n > 0 && rows.len() > top_n {
            if rows.iter().any(|r| r.2.is_none()) {
                bail!(
                    "Rerank: top_n = {top_n} but the input has {} rows and no prior-score \
                     column to choose them by; pass [item, text, prior_score]",
                    rows.len()
                );
            }
            // Highest prior first; ties broken by item for determinism.
            rows.sort_by(|a, b| b.2.unwrap().total_cmp(&a.2.unwrap()).then(a.0.cmp(&b.0)));
            rows.truncate(top_n);
        }
        if rows.is_empty() {
            return Ok(());
        }

        let chunk = if batch_size == 0 { rows.len() } else { batch_size };
        let mut best: BTreeMap<DataValue, f64> = BTreeMap::new();
        for batch in rows.chunks(chunk) {
            poison.check()?;
            let docs: Vec<&str> = batch.iter().map(|r| r.1.as_str()).collect();
            let scores = reranker.rerank(&query, &docs)?;
            if scores.len() != docs.len() {
                bail!(
                    "Rerank: reranker '{model}' returned {} scores for {} documents",
                    scores.len(),
                    docs.len()
                );
            }
            for ((item, _, _), s) in batch.iter().zip(scores) {
                if !s.is_finite() {
                    bail!("Rerank: reranker '{model}' returned a non-finite score {s}");
                }
                let s = s as f64;
                let e = best.entry(item.clone()).or_insert(s);
                if s > *e {
                    *e = s;
                }
            }
        }
        for (item, s) in best {
            out.put(vec![item, DataValue::from(s)]);
        }
        Ok(())
    }

    fn arity(
        &self,
        _options: &BTreeMap<SmartString<LazyCompact>, Expr>,
        _rule_head: &[Symbol],
        _span: SourceSpan,
    ) -> Result<usize> {
        Ok(2)
    }
}
