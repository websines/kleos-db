/*
 * Copyright 2026, The kleos authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Predicate-aware HNSW search (kleos fork, upgrade plan A1).
//!
//! The upstream search ran the bottom layer with `ef` candidates and applied the
//! search `filter` afterwards, so a selective filter silently returned fewer
//! than `k` rows. Here the filter (and `radius`) is applied *during* the
//! bottom-layer search:
//!
//! * Nodes failing the predicate still navigate (they enter the candidate
//!   queue) but never enter the result set. The search keeps expanding until
//!   `max(ef, k)` passing results are held and the closest unexpanded candidate
//!   is farther than the worst of them, or the reachable graph is exhausted —
//!   the adaptive "keep scanning" behaviour of pgvector 0.8 iterative scans,
//!   with the hnswlib filtered-search stopping rule.
//! * When the observed pass rate is low, a failing neighbour of the node being
//!   expanded is expanded on the spot (ACORN-1 style two-hop expansion), so the
//!   search reaches passing nodes behind it without queueing a failing node.
//! * When the projected remaining graph work exceeds a full scan of the base
//!   relation (selective filters), the search switches to exact brute force
//!   over the rows that pass the filter and are present in the index.

use super::{distance_is_farther, hnsw_row_key, CompoundKey, VectorCache};
use crate::bail;
use crate::data::expr::{eval_bytecode_pred, Bytecode};
use crate::data::program::HnswSearch;
use crate::data::tuple::Tuple;
use crate::data::value::Vector;
use crate::runtime::transact::SessionTx;
use crate::{DataValue, SourceSpan};
use miette::Result;
use ordered_float::OrderedFloat;
use priority_queue::PriorityQueue;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::{max, Reverse};

/// Below this observed pass rate a failing neighbour is expanded immediately
/// (two-hop) instead of being queued.
const TWO_HOP_PASS_RATE: f64 = 0.1;
/// Predicate evaluations before the observed pass rate drives any decision.
const MIN_SAMPLE: usize = 128;
/// Projected remaining graph visits below which brute force is never
/// considered (so cheap searches never pay for the size estimate).
const BRUTE_FORCE_MIN_VISITS: f64 = 1024.0;
/// Cost of one graph visit (fetch + distance + predicate + share of the
/// adjacency reads) relative to one row of a full base-relation scan.
const VISIT_COST: f64 = 3.0;
/// A level must hold this many nodes before its count is used to estimate the
/// index size.
const SIZE_SAMPLE_NODES: usize = 16;

/// The compiled search filter.
pub(super) struct SearchPredicate<'a> {
    pub(super) code: &'a [Bytecode],
    pub(super) span: SourceSpan,
    /// Whether the filter reads any of the bound outputs (`bind_field`,
    /// `bind_field_idx`, `bind_distance`, `bind_vector`). If not, it can be
    /// evaluated on the stored row alone.
    pub(super) needs_bindings: bool,
}

/// Bottom-layer search state.
struct FilteredState {
    /// passing results kept (the hnswlib `ef`)
    target: usize,
    visited: FxHashSet<CompoundKey>,
    /// min-queue of nodes to expand
    candidates: PriorityQueue<CompoundKey, Reverse<OrderedFloat<f64>>>,
    /// max-queue of passing nodes, at most `target`
    results: PriorityQueue<CompoundKey, OrderedFloat<f64>>,
    /// output tuples (row + bindings) of `results`
    result_tuples: FxHashMap<CompoundKey, Tuple>,
    evaluated: usize,
    passed: usize,
}

impl FilteredState {
    /// Whether a node at `dist` could still enter the results (and therefore
    /// is worth testing and queueing).
    fn admits(&self, dist: f64) -> bool {
        match self.worst() {
            Some(worst) => distance_is_farther(worst, dist),
            None => true,
        }
    }
    /// Distance of the worst kept result once the result set is full.
    fn worst(&self) -> Option<f64> {
        if self.results.len() < self.target {
            None
        } else {
            self.results.peek().map(|(_, d)| d.0)
        }
    }
    /// Laplace-smoothed observed pass rate.
    fn pass_rate(&self) -> f64 {
        (self.passed as f64 + 1.0) / (self.evaluated as f64 + 2.0)
    }
    fn add_result(&mut self, key: CompoundKey, dist: f64, tuple: Tuple) {
        self.result_tuples.insert(key.clone(), tuple);
        self.results.push(key, OrderedFloat(dist));
        if self.results.len() > self.target {
            if let Some((evicted, _)) = self.results.pop() {
                self.result_tuples.remove(&evicted);
            }
        }
    }
}

/// The vector a node refers to inside its stored row.
fn node_vector<'t>(tuple: &'t [DataValue], key: &CompoundKey) -> Result<&'t Vector> {
    let mut field = match tuple.get(key.1) {
        Some(f) => f,
        None => bail!("corrupt HNSW node: field {} out of range", key.1),
    };
    if key.2 >= 0 {
        match field {
            DataValue::List(l) if (key.2 as usize) < l.len() => field = &l[key.2 as usize],
            _ => bail!("Cannot interpret {} as list", field),
        }
    }
    match field {
        DataValue::Vec(v) => Ok(v),
        _ => bail!("Cannot interpret {} as vector", field),
    }
}

/// Append the requested output bindings to a stored row. The order must match
/// `HnswSearch::all_bindings()`.
pub(super) fn bind_result(
    config: &HnswSearch,
    key: &CompoundKey,
    mut tuple: Tuple,
    distance: f64,
) -> Result<Tuple> {
    let meta = &config.base_handle.metadata;
    if config.bind_field.is_some() {
        let field = if key.1 < meta.keys.len() {
            meta.keys[key.1].name.clone()
        } else {
            meta.non_keys[key.1 - meta.keys.len()].name.clone()
        };
        tuple.push(DataValue::Str(field));
    }
    if config.bind_field_idx.is_some() {
        tuple.push(if key.2 < 0 {
            DataValue::Null
        } else {
            DataValue::from(key.2 as i64)
        });
    }
    if config.bind_distance.is_some() {
        tuple.push(DataValue::from(distance));
    }
    if config.bind_vector.is_some() {
        let vec = if key.2 < 0 {
            tuple[key.1].clone()
        } else {
            match &tuple[key.1] {
                DataValue::List(v) => v[key.2 as usize].clone(),
                v => bail!("corrupted index value {:?}", v),
            }
        };
        tuple.push(vec);
    }
    Ok(tuple)
}

/// Pop a max-queue of results into ascending distance order, at most `k`.
fn into_output(
    mut results: PriorityQueue<CompoundKey, OrderedFloat<f64>>,
    mut tuples: FxHashMap<CompoundKey, Tuple>,
    k: usize,
) -> Vec<Tuple> {
    while results.len() > k {
        results.pop();
    }
    let mut ret = Vec::with_capacity(results.len());
    while let Some((key, _)) = results.pop() {
        if let Some(t) = tuples.remove(&key) {
            ret.push(t);
        }
    }
    ret.reverse();
    ret
}

// TEMP tuning knobs (removed before merge)
struct Tuning {
    two_hop_rate: f64,
    two_hop_mode: u8,
    visit_cost: f64,
}
fn tuning() -> Tuning {
    let get = |k: &str, d: f64| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(d)
    };
    Tuning {
        two_hop_rate: get("KLEOS_A1_TWO_HOP", TWO_HOP_PASS_RATE),
        two_hop_mode: get("KLEOS_A1_TWO_HOP_MODE", 1.0) as u8,
        visit_cost: get("KLEOS_A1_VISIT_COST", VISIT_COST),
    }
}

impl SessionTx<'_> {
    /// Bottom-layer search with the filter applied inside the search. `seeds`
    /// are the entry points found by the upper-layer descent; `top_level` is
    /// the entry level (used only to estimate the index size).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hnsw_knn_filtered(
        &self,
        q: &Vector,
        config: &HnswSearch,
        pred: &SearchPredicate<'_>,
        stack: &mut Vec<DataValue>,
        vec_cache: &VectorCache,
        seeds: Vec<CompoundKey>,
        top_level: i64,
    ) -> Result<Vec<Tuple>> {
        let mut st = FilteredState {
            target: max(config.ef, config.k).max(1),
            visited: FxHashSet::default(),
            candidates: PriorityQueue::new(),
            results: PriorityQueue::new(),
            result_tuples: FxHashMap::default(),
            evaluated: 0,
            passed: 0,
        };
        for s in &seeds {
            st.visited.insert(s.clone());
        }
        for (key, dist, _) in self.hnsw_filtered_visit(seeds, q, config, pred, stack, vec_cache, &mut st)? {
            st.candidates.push(key, Reverse(OrderedFloat(dist)));
        }

        let tun = tuning();
        let mut size_estimate: Option<f64> = None;
        while let Some((cand, Reverse(OrderedFloat(cand_dist)))) = st.candidates.pop() {
            if let Some(worst) = st.worst() {
                if distance_is_farther(cand_dist, worst) {
                    break;
                }
            }
            // Past the radius nothing can pass; stop once the frontier has
            // moved beyond it after an `ef`-sized exploration.
            if let Some(r) = config.radius {
                if cand_dist > r && st.visited.len() >= st.target {
                    break;
                }
            }
            // Selective filter: switch to brute force once the projected
            // remaining graph work costs more than scanning the relation.
            if st.results.len() < st.target && st.evaluated >= MIN_SAMPLE {
                let remaining = (st.target - st.results.len()) as f64 / st.pass_rate();
                if remaining >= BRUTE_FORCE_MIN_VISITS {
                    let size = match size_estimate {
                        Some(s) => s,
                        None => {
                            let s = self.hnsw_estimate_size(config, top_level)?;
                            size_estimate = Some(s);
                            s
                        }
                    };
                    if remaining * tun.visit_cost > size {
                        if std::env::var_os("KLEOS_A1_TRACE").is_some() {
                            eprintln!("A1TRACE brute visited={} evaluated={} passed={} size={size:.0}", st.visited.len(), st.evaluated, st.passed);
                        }
                        return self.hnsw_knn_brute_force(q, config, pred, stack, vec_cache);
                    }
                }
            }

            let two_hop = tun.two_hop_mode > 0
                && st.evaluated >= MIN_SAMPLE
                && st.pass_rate() < tun.two_hop_rate;
            let unvisited: Vec<CompoundKey> = self
                .hnsw_get_neighbours(&cand, 0, &config.idx_handle, false)?
                .filter_map(|(k, _)| st.visited.insert(k.clone()).then_some(k))
                .collect();
            let mut second_hop = vec![];
            for (key, dist, passed) in
                self.hnsw_filtered_visit(unvisited, q, config, pred, stack, vec_cache, &mut st)?
            {
                if two_hop && !passed && (tun.two_hop_mode == 1 || dist <= cand_dist) {
                    for (k2, _) in self.hnsw_get_neighbours(&key, 0, &config.idx_handle, false)? {
                        if st.visited.insert(k2.clone()) {
                            second_hop.push(k2);
                        }
                    }
                } else {
                    st.candidates.push(key, Reverse(OrderedFloat(dist)));
                }
            }
            if !second_hop.is_empty() {
                for (key, dist, _) in self
                    .hnsw_filtered_visit(second_hop, q, config, pred, stack, vec_cache, &mut st)?
                {
                    st.candidates.push(key, Reverse(OrderedFloat(dist)));
                }
            }
        }
        if std::env::var_os("KLEOS_A1_TRACE").is_some() {
            eprintln!("A1TRACE graph visited={} evaluated={} passed={} results={}", st.visited.len(), st.evaluated, st.passed, st.results.len());
        }
        Ok(into_output(st.results, st.result_tuples, config.k))
    }

    /// Fetch, measure and (if the node could still enter the results) test
    /// the given freshly visited nodes. Passing nodes are added to the
    /// results. Returns the admitted nodes as `(key, distance, passed)`.
    #[allow(clippy::too_many_arguments)]
    fn hnsw_filtered_visit(
        &self,
        keys: Vec<CompoundKey>,
        q: &Vector,
        config: &HnswSearch,
        pred: &SearchPredicate<'_>,
        stack: &mut Vec<DataValue>,
        vec_cache: &VectorCache,
        st: &mut FilteredState,
    ) -> Result<Vec<(CompoundKey, f64, bool)>> {
        if keys.is_empty() {
            return Ok(vec![]);
        }
        let key_slices: Vec<&[DataValue]> = keys.iter().map(|k| k.0.as_slice()).collect();
        let tuples = config.base_handle.get_batch(self, &key_slices)?;
        let mut ret = Vec::with_capacity(keys.len());
        for (key, tuple) in keys.into_iter().zip(tuples) {
            let tuple = match tuple {
                Some(t) => t,
                None => bail!("Cannot find compound key for HNSW: {:?}", key),
            };
            let dist = vec_cache.dist(q, node_vector(&tuple, &key)?);
            if !st.admits(dist) {
                continue;
            }
            st.evaluated += 1;
            let bound = if config.radius.is_some_and(|r| dist > r) {
                None
            } else {
                self.hnsw_test_row(config, pred, stack, &key, tuple, dist)?
            };
            let passed = bound.is_some();
            if let Some(bound) = bound {
                st.passed += 1;
                st.add_result(key.clone(), dist, bound);
            }
            ret.push((key, dist, passed));
        }
        Ok(ret)
    }

    /// Evaluate the filter on a stored row; on success return the output
    /// tuple (row + bindings).
    fn hnsw_test_row(
        &self,
        config: &HnswSearch,
        pred: &SearchPredicate<'_>,
        stack: &mut Vec<DataValue>,
        key: &CompoundKey,
        tuple: Tuple,
        dist: f64,
    ) -> Result<Option<Tuple>> {
        if pred.needs_bindings {
            let bound = bind_result(config, key, tuple, dist)?;
            Ok(eval_bytecode_pred(pred.code, &bound, stack, pred.span)?.then_some(bound))
        } else if eval_bytecode_pred(pred.code, &tuple, stack, pred.span)? {
            Ok(Some(bind_result(config, key, tuple, dist)?))
        } else {
            Ok(None)
        }
    }

    /// Exact k-NN over the rows of the base relation that pass the filter and
    /// are present in the index (the index may exclude rows by its own
    /// `filter`, or hold vectors of only some rows).
    fn hnsw_knn_brute_force(
        &self,
        q: &Vector,
        config: &HnswSearch,
        pred: &SearchPredicate<'_>,
        stack: &mut Vec<DataValue>,
        vec_cache: &VectorCache,
    ) -> Result<Vec<Tuple>> {
        let key_len = config.base_handle.metadata.keys.len();
        let k = config.k;
        let mut results: PriorityQueue<CompoundKey, OrderedFloat<f64>> = PriorityQueue::new();
        let mut tuples: FxHashMap<CompoundKey, Tuple> = FxHashMap::default();
        if k == 0 {
            return Ok(vec![]);
        }
        for row in config.base_handle.scan_all(self) {
            let row = row?;
            if !pred.needs_bindings && !eval_bytecode_pred(pred.code, &row, stack, pred.span)? {
                continue;
            }
            for &field in &config.manifest.vec_fields {
                let vectors: Vec<(i32, &Vector)> = match row.get(field) {
                    Some(DataValue::Vec(v)) => vec![(-1, v)],
                    Some(DataValue::List(l)) => l
                        .iter()
                        .enumerate()
                        .filter_map(|(i, v)| match v {
                            DataValue::Vec(v) => Some((i as i32, v)),
                            _ => None,
                        })
                        .collect(),
                    _ => continue,
                };
                for (sub_idx, v) in vectors {
                    if v.el_type() != config.manifest.dtype || v.len() != config.manifest.vec_dim
                    {
                        continue;
                    }
                    let dist = vec_cache.dist(q, v);
                    if config.radius.is_some_and(|r| dist > r) {
                        continue;
                    }
                    if results.len() >= k {
                        if let Some((_, worst)) = results.peek() {
                            if !distance_is_farther(worst.0, dist) {
                                continue;
                            }
                        }
                    }
                    let key: CompoundKey = (row[..key_len].to_vec(), field, sub_idx);
                    let self_row = hnsw_row_key(0, &key, &key);
                    if !config.idx_handle.exists(self, &self_row)? {
                        continue;
                    }
                    let out = if pred.needs_bindings {
                        match self.hnsw_test_row(config, pred, stack, &key, row.clone(), dist)? {
                            Some(t) => t,
                            None => continue,
                        }
                    } else {
                        bind_result(config, &key, row.clone(), dist)?
                    };
                    tuples.insert(key.clone(), out);
                    results.push(key, OrderedFloat(dist));
                    if results.len() > k {
                        if let Some((evicted, _)) = results.pop() {
                            tuples.remove(&evicted);
                        }
                    }
                }
            }
        }
        Ok(into_output(results, tuples, k))
    }

    /// Estimate the number of indexed vectors from the node count of the
    /// highest level holding at least `SIZE_SAMPLE_NODES` nodes: a node
    /// reaches depth `d` with probability `exp(-d / level_multiplier)`.
    fn hnsw_estimate_size(&self, config: &HnswSearch, top_level: i64) -> Result<f64> {
        let key_len = config.base_handle.metadata.keys.len();
        for level in top_level..=0 {
            let mut count = 0usize;
            for row in config
                .idx_handle
                .scan_prefix(self, &vec![DataValue::from(level)])
            {
                let row = row?;
                if row.len() >= 2 * key_len + 5
                    && row[1..key_len + 3] == row[key_len + 3..2 * key_len + 5]
                {
                    count += 1;
                }
            }
            if count >= SIZE_SAMPLE_NODES || level == 0 {
                let depth = (-level) as f64;
                return Ok(count as f64 * (depth / config.manifest.level_multiplier).exp());
            }
        }
        Ok(0.0)
    }
}
