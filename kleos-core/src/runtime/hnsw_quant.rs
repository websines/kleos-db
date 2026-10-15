/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Quantized HNSW traversal (kleos fork, A2): glue between the HNSW index and
//! the codec in [`crate::runtime::vec_quant`].
//!
//! An index created with `quantize: turbo4` (etc.) keeps one *code row* per
//! indexed vector in its index relation, at layer [`CODE_LAYER`]:
//!
//! ```text
//! key   [2, key.., field, sub_idx, key.., field, sub_idx]
//! value [0.0, Bytes(code), false]
//! ```
//!
//! Graph layers are `<= 0` and the entry-point canary lives at layer 1, so the
//! code rows never appear in neighbour scans or entry-point lookups. Search
//! fetches these small rows (e.g. 776 bytes for 1536-d `turbo4`, versus the
//! full base row) and estimates distances from them; the best candidates are
//! then rescored exactly against the base relation (see
//! `SessionTx::hnsw_rescore`). Graph construction (insert, repair, bulk build)
//! always uses exact distances.

use std::sync::Arc;

use miette::Result;
use ordered_float::OrderedFloat;
use priority_queue::PriorityQueue;
use rustc_hash::FxHashMap;

use crate::data::value::Vector;
use crate::parse::sys::HnswDistance;
use crate::runtime::hnsw::CompoundKey;
use crate::runtime::relation::{try_decode_val_only, RelationHandle};
use crate::runtime::transact::SessionTx;
use crate::runtime::vec_quant::{PreparedQuery, QuantCodec, VecQuant};
use crate::storage::StoreTx;
use crate::DataValue;

/// Index-relation layer holding the per-vector codes.
pub(crate) const CODE_LAYER: i64 = 2;

/// Default rescoring multiplier: the top `k * rescore` code-ranked candidates
/// are rescored exactly. Chosen from the codec evaluation on 1536-d OpenAI
/// embeddings (recall@10 of the rescored codes reaches >= 0.99 at these
/// multipliers; see CHANGELOG / UPGRADE_PLAN A2).
pub(crate) fn default_rescore(quant: VecQuant) -> usize {
    match quant {
        VecQuant::None => 1,
        VecQuant::Binary => 8,
        VecQuant::Turbo2 => 4,
        VecQuant::Turbo4 => 4,
        VecQuant::Turbo8 => 2,
    }
}

/// Key of the code row of `key`.
pub(crate) fn code_row_key(key: &CompoundKey) -> Vec<DataValue> {
    let mut out = Vec::with_capacity(key.0.len() * 2 + 5);
    out.push(DataValue::from(CODE_LAYER));
    for _ in 0..2 {
        out.extend_from_slice(&key.0);
        out.push(DataValue::from(key.1 as i64));
        out.push(DataValue::from(key.2 as i64));
    }
    out
}

fn as_f32(v: &Vector) -> std::borrow::Cow<'_, [f32]> {
    match v {
        Vector::F32(a) => match a.as_slice() {
            Some(s) => std::borrow::Cow::Borrowed(s),
            None => std::borrow::Cow::Owned(a.to_vec()),
        },
        Vector::F64(a) => std::borrow::Cow::Owned(a.iter().map(|x| *x as f32).collect()),
    }
}

/// Encode one vector.
pub(crate) fn encode_vector(codec: &QuantCodec, v: &Vector) -> Vec<u8> {
    codec.encode(&as_f32(v))
}

/// Encode many vectors on `threads` threads, preserving order.
pub(crate) fn encode_vectors(codec: &QuantCodec, vecs: &[&Vector], threads: usize) -> Vec<Vec<u8>> {
    let threads = threads.max(1);
    if threads == 1 || vecs.len() < 1024 {
        return vecs.iter().map(|v| encode_vector(codec, v)).collect();
    }
    let chunk = vecs.len().div_ceil(threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = vecs
            .chunks(chunk)
            .map(|part| s.spawn(move || part.iter().map(|v| encode_vector(codec, v)).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_default())
            .collect()
    })
}

/// Per-search quantized state held by the HNSW vector cache.
pub(crate) struct QuantState {
    codec: Arc<QuantCodec>,
    query: PreparedQuery,
    distance: HnswDistance,
    idx_handle: RelationHandle,
    codes: FxHashMap<CompoundKey, Vec<u8>>,
}

impl QuantState {
    /// Quantized state for a search over `idx_handle`, or `None` when the
    /// index is not quantized.
    pub(crate) fn for_search(
        quant: VecQuant,
        dim: usize,
        distance: HnswDistance,
        q: &Vector,
        idx_handle: &RelationHandle,
    ) -> Option<Self> {
        let codec = QuantCodec::for_quant(quant, dim)?;
        let query = codec.prepare(&as_f32(q));
        Some(Self {
            codec,
            query,
            distance,
            idx_handle: idx_handle.clone(),
            codes: FxHashMap::default(),
        })
    }

    pub(crate) fn has(&self, key: &CompoundKey) -> bool {
        self.codes.contains_key(key)
    }

    /// Estimated distance from the query to `key`, if its code is loaded.
    /// Same scale as the exact distances in `VectorCache::dist`.
    pub(crate) fn dist(&self, key: &CompoundKey) -> Option<f64> {
        let code = self.codes.get(key)?;
        let dot = self.codec.dot(&self.query, code) as f64;
        Some(match self.distance {
            HnswDistance::L2 => {
                let n = QuantCodec::code_norm(code) as f64;
                (self.query.norm_sq() as f64 + n * n - 2.0 * dot).max(0.0)
            }
            HnswDistance::Cosine => {
                let n = QuantCodec::code_norm(code) as f64;
                let denom = (self.query.norm_sq() as f64).sqrt() * n;
                if denom > 0.0 {
                    1.0 - dot / denom
                } else {
                    2.0
                }
            }
            HnswDistance::InnerProduct => 1.0 - dot,
        })
    }

    /// Load the codes of `keys` not yet loaded (one batched read). Returns the
    /// keys that have no code row (callers fall back to exact vectors).
    pub(crate) fn load(&mut self, keys: &[CompoundKey], tx: &SessionTx<'_>) -> Result<Vec<CompoundKey>> {
        let missing: Vec<&CompoundKey> = keys.iter().filter(|k| !self.codes.contains_key(*k)).collect();
        if missing.is_empty() {
            return Ok(vec![]);
        }
        let encoded: Vec<Vec<u8>> = missing
            .iter()
            .map(|k| {
                self.idx_handle
                    .encode_key_for_store(&code_row_key(k), Default::default())
            })
            .collect::<Result<_>>()?;
        let raw = if self.idx_handle.is_temp {
            tx.temp_store_tx.multi_get(&encoded, false)?
        } else {
            tx.store_tx.multi_get(&encoded, false)?
        };
        let mut absent = vec![];
        for ((key, enc), val) in missing.into_iter().zip(&encoded).zip(raw) {
            let code = match val {
                Some(val) => match try_decode_val_only(enc, &val)?.into_iter().nth(1) {
                    Some(DataValue::Bytes(b)) if b.len() == self.codec.code_len() => Some(b),
                    _ => None,
                },
                None => None,
            };
            match code {
                Some(b) => {
                    self.codes.insert(key.clone(), b);
                }
                None => absent.push(key.clone()),
            }
        }
        Ok(absent)
    }
}

impl<'a> SessionTx<'a> {
    fn hnsw_code_key_bytes(idx_table: &RelationHandle, key: &CompoundKey) -> Result<Vec<u8>> {
        idx_table.encode_key_for_store(&code_row_key(key), Default::default())
    }

    /// Write the code row of `key` (no-op for unquantized indices).
    pub(crate) fn hnsw_put_code(
        &mut self,
        quant: VecQuant,
        key: &CompoundKey,
        v: &Vector,
        idx_table: &RelationHandle,
    ) -> Result<()> {
        let Some(codec) = QuantCodec::for_quant(quant, v.len()) else {
            return Ok(());
        };
        self.hnsw_put_code_bytes(key, encode_vector(&codec, v), idx_table)
    }

    pub(crate) fn hnsw_put_code_bytes(
        &mut self,
        key: &CompoundKey,
        code: Vec<u8>,
        idx_table: &RelationHandle,
    ) -> Result<()> {
        let k = Self::hnsw_code_key_bytes(idx_table, key)?;
        let v = idx_table.encode_val_only_for_store(
            &[
                DataValue::from(0.0),
                DataValue::Bytes(code),
                DataValue::from(false),
            ],
            Default::default(),
        )?;
        if idx_table.is_temp {
            self.temp_store_tx.put(&k, &v)
        } else {
            self.store_tx.put(&k, &v)
        }
    }

    /// Delete the code row of `key`, if any.
    pub(crate) fn hnsw_del_code(&mut self, key: &CompoundKey, idx_table: &RelationHandle) -> Result<()> {
        let k = Self::hnsw_code_key_bytes(idx_table, key)?;
        if idx_table.is_temp {
            self.temp_store_tx.del(&k)
        } else {
            self.store_tx.del(&k)
        }
    }

    /// Replace the code-estimated distances in `found_nn` (a max-queue on
    /// distance) by exact ones for its `keep` closest entries, dropping the
    /// rest. `exact` computes the exact distance of a key (fetching its
    /// vector as needed).
    pub(crate) fn hnsw_rescore(
        found_nn: &mut PriorityQueue<CompoundKey, OrderedFloat<f64>>,
        keep: usize,
        mut exact: impl FnMut(&[CompoundKey]) -> Result<Vec<f64>>,
    ) -> Result<()> {
        let mut by_estimate: Vec<(CompoundKey, OrderedFloat<f64>)> =
            std::mem::replace(found_nn, PriorityQueue::new())
                .into_iter()
                .collect();
        by_estimate.sort_by_key(|(_, d)| *d);
        by_estimate.truncate(keep);
        let keys: Vec<CompoundKey> = by_estimate.into_iter().map(|(k, _)| k).collect();
        let dists = exact(&keys)?;
        for (k, d) in keys.into_iter().zip(dists) {
            found_nn.push(k, OrderedFloat(d));
        }
        Ok(())
    }
}
