/*
 * Copyright 2026, Shan Rizvi (mnestic fork).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Weighted (convex-combination) score fusion — a kleos fork addition for
//! hybrid retrieval, the score-based sibling of [`ReciprocalRankFusion`].
//!
//! RRF throws the scores away and fuses ranks. Bruch, Gai & Ingber, *An
//! Analysis of Fusion Functions for Hybrid Retrieval* (arXiv 2210.11934,
//! TOIS 2023) show that a **convex combination of normalized scores**
//! (`Σ_l w_l · φ_l(score)`) is at least as good as RRF in- and
//! out-of-domain, is far less sensitive to its one parameter, and needs only a
//! handful of labelled queries to tune the weight. This rule implements it.
//!
//! Input: a single relation `[list_id, item, score]` (the same shape RRF
//! takes). Scores must be **higher = better** within each list. Output:
//! `[item, fused_score]`; with `detailed: true` the long-format
//! `[item, fused_score, list_id, leg_rank, leg_score]` (identical columns to
//! RRF's detailed mode: `leg_rank` is the 1-based within-list rank by raw
//! score, `leg_score` the raw deduplicated best score).
//!
//! Options:
//! - `weights`: list of `[list_id, weight]` pairs, e.g.
//!   `[['semantic', 0.7], ['text', 0.3]]`. Weights must be finite and `>= 0`;
//!   they are used as given (pass weights summing to 1 for a true convex
//!   combination). Omitted entirely ⇒ every list present gets `1 / n_lists`.
//! - `default_weight`: weight for a list that `weights` does not name. Without
//!   it, an unnamed list is an error (a typo'd label must not silently drop a
//!   leg).
//! - `norm`: the default normalization φ for every list (default
//!   `'minmax'`). `norms`: per-list overrides, `[[list_id, norm], ...]`.
//!
//! Normalizations (the score-scale problem is the whole game for convex
//! fusion; the theoretical ones are the paper's "TMM" family):
//! - `'minmax'` — `(s − min) / (max − min)` over the list's observed scores
//!   (all-equal ⇒ 1).
//! - `'zscore'` — `(s − mean) / std` (zero spread ⇒ 0).
//! - `'none'` — raw scores.
//! - `'neg_cosine_distance'` — theoretical: the score is a *negated* cosine
//!   distance `−d`, `d ∈ [0, 2]` (what HNSW `bind_distance` gives for a cosine
//!   index, negated so higher is better — the hybrid builder's vector leg) ⇒
//!   `1 − d / 2`, i.e. `(1 + cos_sim) / 2`.
//! - `'cosine_similarity'` — theoretical: `s ∈ [−1, 1]` ⇒ `(s + 1) / 2`.
//! - `'bm25'` — theoretical min-max with BM25's lower bound 0:
//!   `s / max` (bounded to `[0, 1]`, but unlike min-max the weakest hit keeps
//!   its real distance from zero).
//! - `['range', lo, hi]` — known fixed range ⇒ `(s − lo) / (hi − lo)`, clamped
//!   to `[0, 1]`.
//! - `['tmm', lo]` — theoretical lower bound, observed upper bound ⇒
//!   `(s − lo) / (max − lo)`.
//!
//! An item absent from a list contributes that list's **floor**: `0` for the
//! bounded normalizations (the theoretical minimum — a non-retrieved item
//! scores at most the weakest retrieved one), and the list's smallest
//! normalized value for `'zscore'` / `'none'` (where 0 is not a minimum).
//!
//! ```cozo
//! ?[item, fused] <~ WeightedFusion(combined[lid, item, score],
//!     weights: [['semantic', 0.6], ['text', 0.4]],
//!     norm: 'minmax', norms: [['semantic', 'neg_cosine_distance'], ['text', 'bm25']])
//! ```
//!
//! [`ReciprocalRankFusion`]: super::ReciprocalRankFusion

use std::collections::BTreeMap;

use crate::bail;

use miette::Result;
use smartstring::{LazyCompact, SmartString};

use crate::data::expr::Expr;
use crate::data::symb::Symbol;
use crate::data::value::DataValue;
use crate::fixed_rule::{CannotDetermineArity, FixedRule, FixedRulePayload};
use crate::parse::SourceSpan;
use crate::runtime::db::Poison;
use crate::runtime::temp_store::RegularTempStore;

pub(crate) struct WeightedFusion;

/// A parsed normalization spec.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Norm {
    MinMax,
    ZScore,
    Raw,
    NegCosineDistance,
    CosineSimilarity,
    Bm25,
    Range(f64, f64),
    Tmm(f64),
}

fn parse_norm(v: &DataValue) -> Result<Norm> {
    let finite = |v: &DataValue, what: &str| -> Result<f64> {
        match v.get_float() {
            Some(f) if f.is_finite() => Ok(f),
            _ => bail!("WeightedFusion: {what} must be a finite number, got {v:?}"),
        }
    };
    Ok(match v {
        DataValue::Str(s) => match s.as_str() {
            "minmax" | "min_max" => Norm::MinMax,
            "zscore" | "z_score" => Norm::ZScore,
            "none" | "raw" => Norm::Raw,
            "neg_cosine_distance" => Norm::NegCosineDistance,
            "cosine_similarity" => Norm::CosineSimilarity,
            "bm25" => Norm::Bm25,
            other => bail!(
                "WeightedFusion: unknown normalization '{other}' (expected one of \
                 'minmax', 'zscore', 'none', 'neg_cosine_distance', 'cosine_similarity', \
                 'bm25', ['range', lo, hi], ['tmm', lo])"
            ),
        },
        DataValue::List(l) => match (l.first().and_then(|h| h.get_str()), l.len()) {
            (Some("range"), 3) => {
                let lo = finite(&l[1], "range lower bound")?;
                let hi = finite(&l[2], "range upper bound")?;
                if hi <= lo {
                    bail!("WeightedFusion: ['range', lo, hi] needs hi > lo, got [{lo}, {hi}]");
                }
                Norm::Range(lo, hi)
            }
            (Some("tmm"), 2) => Norm::Tmm(finite(&l[1], "tmm lower bound")?),
            _ => bail!(
                "WeightedFusion: a list normalization must be ['range', lo, hi] or \
                 ['tmm', lo], got {v:?}"
            ),
        },
        _ => bail!("WeightedFusion: a normalization must be a string or a list, got {v:?}"),
    })
}

/// Parse `[[list_id, value], ...]` into a map, rejecting duplicates.
fn parse_pairs(v: &DataValue, opt: &str) -> Result<Vec<(DataValue, DataValue)>> {
    let DataValue::List(l) = v else {
        bail!("WeightedFusion: option '{opt}' must be a list of [list_id, value] pairs, got {v:?}");
    };
    let mut out: Vec<(DataValue, DataValue)> = Vec::with_capacity(l.len());
    for pair in l {
        match pair {
            DataValue::List(p) if p.len() == 2 => {
                if out.iter().any(|(k, _)| *k == p[0]) {
                    bail!("WeightedFusion: option '{opt}' names list {:?} twice", p[0]);
                }
                out.push((p[0].clone(), p[1].clone()));
            }
            _ => bail!(
                "WeightedFusion: option '{opt}' must be a list of [list_id, value] pairs, \
                 got element {pair:?}"
            ),
        }
    }
    Ok(out)
}

/// Normalize one list's scores (index-aligned) under `norm`; returns the
/// normalized values and the list's floor (the contribution of an absent item).
fn normalize(norm: Norm, scores: &[f64]) -> (Vec<f64>, f64) {
    if scores.is_empty() {
        return (vec![], 0.0);
    }
    let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let normed: Vec<f64> = match norm {
        Norm::MinMax => {
            let span = max - min;
            scores
                .iter()
                .map(|s| if span > 0.0 { (s - min) / span } else { 1.0 })
                .collect()
        }
        Norm::ZScore => {
            let n = scores.len() as f64;
            let mean = scores.iter().sum::<f64>() / n;
            let var = scores.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / n;
            let std = var.sqrt();
            scores
                .iter()
                .map(|s| if std > 0.0 { (s - mean) / std } else { 0.0 })
                .collect()
        }
        Norm::Raw => scores.to_vec(),
        Norm::NegCosineDistance => scores
            .iter()
            .map(|s| (1.0 + s / 2.0).clamp(0.0, 1.0))
            .collect(),
        Norm::CosineSimilarity => scores
            .iter()
            .map(|s| ((s + 1.0) / 2.0).clamp(0.0, 1.0))
            .collect(),
        Norm::Bm25 => scores
            .iter()
            .map(|s| {
                if max > 0.0 {
                    (s / max).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
            .collect(),
        Norm::Range(lo, hi) => scores
            .iter()
            .map(|s| ((s - lo) / (hi - lo)).clamp(0.0, 1.0))
            .collect(),
        Norm::Tmm(lo) => scores
            .iter()
            .map(|s| {
                if max > lo {
                    ((s - lo) / (max - lo)).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
            .collect(),
    };
    let floor = match norm {
        Norm::ZScore | Norm::Raw => normed.iter().copied().fold(f64::INFINITY, f64::min),
        _ => 0.0,
    };
    (normed, floor)
}

impl FixedRule for WeightedFusion {
    fn run(
        &self,
        payload: FixedRulePayload<'_, '_>,
        out: &mut RegularTempStore,
        poison: Poison,
    ) -> Result<()> {
        let in_rel = payload.get_input(0)?;
        let detailed = payload.bool_option("detailed", Some(false))?;
        let opt = |name: &str| -> Result<Option<DataValue>> {
            match payload.manifest.options.get(name) {
                None => Ok(None),
                Some(ex) => Ok(Some(ex.clone().eval_to_const()?)),
            }
        };
        let default_norm = match opt("norm")? {
            None => Norm::MinMax,
            Some(v) => parse_norm(&v)?,
        };
        let mut leg_norms: BTreeMap<DataValue, Norm> = BTreeMap::new();
        if let Some(v) = opt("norms")? {
            for (lid, spec) in parse_pairs(&v, "norms")? {
                leg_norms.insert(lid, parse_norm(&spec)?);
            }
        }
        let explicit_weights: Option<BTreeMap<DataValue, f64>> = match opt("weights")? {
            None => None,
            Some(v) => {
                let mut m = BTreeMap::new();
                for (lid, w) in parse_pairs(&v, "weights")? {
                    let w = match w.get_float() {
                        Some(f) if f.is_finite() && f >= 0.0 => f,
                        _ => bail!(
                            "WeightedFusion: weight for list {lid:?} must be a finite number \
                             >= 0, got {w:?}"
                        ),
                    };
                    m.insert(lid, w);
                }
                Some(m)
            }
        };
        let default_weight = match opt("default_weight")? {
            None => None,
            Some(v) => match v.get_float() {
                Some(f) if f.is_finite() && f >= 0.0 => Some(f),
                _ => bail!("WeightedFusion: default_weight must be a finite number >= 0"),
            },
        };

        // Group (item, score) by list_id, keeping each item's best score.
        let mut lists: BTreeMap<DataValue, BTreeMap<DataValue, f64>> = BTreeMap::new();
        for tuple in in_rel.iter()? {
            let tuple = tuple?;
            if tuple.len() != 3 {
                bail!(
                    "WeightedFusion expects a 3-column input [list_id, item, score], \
                     got a row of arity {}",
                    tuple.len()
                );
            }
            let mut it = tuple.into_iter();
            let list_id = it.next().unwrap();
            let item = it.next().unwrap();
            let score = it.next().unwrap();
            let s = match score.get_float() {
                Some(f) if f.is_finite() => f,
                _ => bail!(
                    "WeightedFusion: score (column 3) must be a finite number, got {score:?}"
                ),
            };
            let best = lists.entry(list_id).or_default().entry(item).or_insert(s);
            if s > *best {
                *best = s;
            }
            poison.check()?;
        }

        // Resolve every list's weight up front (a missing weight is loud).
        let n_lists = lists.len();
        let mut weights: BTreeMap<DataValue, f64> = BTreeMap::new();
        for lid in lists.keys() {
            let w = match &explicit_weights {
                None => 1.0 / n_lists as f64,
                Some(m) => match m.get(lid).copied().or(default_weight) {
                    Some(w) => w,
                    None => bail!(
                        "WeightedFusion: list {lid:?} has no weight; name it in 'weights' \
                         or set 'default_weight'"
                    ),
                },
            };
            weights.insert(lid.clone(), w);
        }

        // fused(item) = Σ_l w_l·floor_l + Σ_{l ∋ item} w_l·(φ_l(s) − floor_l)
        let mut base = 0.0;
        let mut fused: BTreeMap<DataValue, f64> = BTreeMap::new();
        let mut contribs: BTreeMap<DataValue, Vec<(DataValue, f64, f64)>> = BTreeMap::new();
        for (lid, items) in lists {
            let w = weights[&lid];
            let norm = leg_norms.get(&lid).copied().unwrap_or(default_norm);
            let (ids, raw): (Vec<DataValue>, Vec<f64>) = items.into_iter().unzip();
            let (normed, floor) = normalize(norm, &raw);
            base += w * floor;
            // Within-list rank by raw score (descending, ties by item order),
            // only needed for the detailed output.
            let mut rank_of: Vec<usize> = vec![0; ids.len()];
            if detailed {
                let mut order: Vec<usize> = (0..ids.len()).collect();
                order.sort_by(|&a, &b| raw[b].total_cmp(&raw[a]));
                for (r, idx) in order.into_iter().enumerate() {
                    rank_of[idx] = r + 1;
                }
            }
            for (idx, item) in ids.into_iter().enumerate() {
                *fused.entry(item.clone()).or_insert(0.0) += w * (normed[idx] - floor);
                if detailed {
                    contribs.entry(item).or_default().push((
                        lid.clone(),
                        rank_of[idx] as f64,
                        raw[idx],
                    ));
                }
            }
            poison.check()?;
        }

        for (item, delta) in fused {
            let score = base + delta;
            if detailed {
                for (lid, leg_rank, leg_score) in contribs.remove(&item).unwrap_or_default() {
                    out.put(vec![
                        item.clone(),
                        DataValue::from(score),
                        lid,
                        DataValue::from(leg_rank),
                        DataValue::from(leg_score),
                    ]);
                }
            } else {
                out.put(vec![item, DataValue::from(score)]);
            }
        }
        Ok(())
    }

    fn arity(
        &self,
        options: &BTreeMap<SmartString<LazyCompact>, Expr>,
        _rule_head: &[Symbol],
        span: SourceSpan,
    ) -> Result<usize> {
        match options.get("detailed") {
            None
            | Some(Expr::Const {
                val: DataValue::Bool(false),
                ..
            }) => Ok(2),
            Some(Expr::Const {
                val: DataValue::Bool(true),
                ..
            }) => Ok(5),
            _ => bail!(CannotDetermineArity(
                "WeightedFusion".to_string(),
                "invalid option 'detailed' given, expect a constant boolean".to_string(),
                span
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theoretical_norms_map_known_ranges_to_unit_interval() {
        // negated cosine distance: -0 (identical) -> 1, -2 (opposite) -> 0
        let (n, f) = normalize(Norm::NegCosineDistance, &[0.0, -1.0, -2.0]);
        assert_eq!(n, vec![1.0, 0.5, 0.0]);
        assert_eq!(f, 0.0);
        // bm25: s / max, the weakest hit keeps its distance from zero
        let (n, _) = normalize(Norm::Bm25, &[4.0, 2.0, 1.0]);
        assert_eq!(n, vec![1.0, 0.5, 0.25]);
        // minmax: the weakest hit collapses to zero
        let (n, _) = normalize(Norm::MinMax, &[4.0, 2.0, 1.0]);
        assert_eq!(n, vec![1.0, 1.0 / 3.0, 0.0]);
        let (n, _) = normalize(Norm::Range(0.0, 10.0), &[5.0, 20.0, -1.0]);
        assert_eq!(n, vec![0.5, 1.0, 0.0]);
        let (n, _) = normalize(Norm::Tmm(1.0), &[3.0, 2.0]);
        assert_eq!(n, vec![1.0, 0.5]);
        // zscore floor is the list minimum, not 0
        let (n, f) = normalize(Norm::ZScore, &[1.0, 3.0]);
        assert_eq!(n, vec![-1.0, 1.0]);
        assert_eq!(f, -1.0);
    }
}
