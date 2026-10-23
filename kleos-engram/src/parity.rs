/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Parity against reference dumps written by `scripts/engram/parity_ref.py`
//! (and `overlay_edit.py` for the nanogpt model): `meta.json`, `lookup.u32`,
//! `cases.jsonl` in one directory per reference.

use std::path::Path;

use serde_json::Value;

use crate::{CompressedTokenizer, ConfigFields, EngramError, NgramHashMapping, Result};

/// Outcome for one reference directory.
#[derive(Clone, Debug, Default)]
pub struct ParityReport {
    /// Raw / compressed sizes of the table we built from tokenizer.json.
    pub built: (usize, u32),
    /// Raw / compressed sizes of the reference table.
    pub reference: (usize, u32),
    /// Entries of the compression table that differ (reported apart: the
    /// demo decodes through transformers, which is version dependent).
    pub lookup_mismatch: usize,
    /// Multiplier, prime, offset, pad-id or table-size mismatches.
    pub param_mismatch: usize,
    /// Cases x layers compared.
    pub cases: usize,
    /// Indices compared.
    pub indices: usize,
    /// (case, layer) pairs whose indices differ.
    pub bad: usize,
}

impl ParityReport {
    /// Hashing parity (given the reference's compression table).
    pub fn hashing_ok(&self) -> bool {
        self.param_mismatch == 0 && self.bad == 0 && self.indices > 0
    }
    /// Compression built from tokenizer.json equals the reference's.
    pub fn compression_ok(&self) -> bool {
        self.lookup_mismatch == 0 && self.built == self.reference
    }
}

fn read_json(p: &Path) -> Result<Value> {
    serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| EngramError(format!("{}: {e}", p.display())))?)
        .map_err(|e| EngramError(format!("{}: {e}", p.display())))
}

/// Compare one reference directory. `built` is the compression table built
/// from the model's tokenizer.json, if one is available.
pub fn check_dir(dir: &Path, built: Option<&CompressedTokenizer>) -> Result<ParityReport> {
    let meta = read_json(&dir.join("meta.json"))?;
    let fields = ConfigFields::from_model_config(&meta, &Default::default())?;
    let bytes = std::fs::read(dir.join("lookup.u32")).map_err(|e| EngramError(format!("lookup.u32: {e}")))?;
    let ref_lookup: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let ref_nnt = meta["num_new_token"].as_u64().unwrap_or(0) as u32;
    let mut r = ParityReport {
        reference: (ref_lookup.len(), ref_nnt),
        ..Default::default()
    };
    if let Some(b) = built {
        r.built = (b.lookup.len(), b.num_new_token);
        r.lookup_mismatch = ref_lookup.iter().zip(&b.lookup).filter(|(a, b)| a != b).count()
            + ref_lookup.len().abs_diff(b.lookup.len());
    }

    let hm = NgramHashMapping::new(fields.cfg, CompressedTokenizer::from_lookup(ref_lookup, Some(ref_nnt)));
    if meta["pad_id_compressed"].as_i64() != Some(hm.pad_id) {
        r.param_mismatch += 1;
    }
    let u64s = |v: &Value| -> Vec<u64> { v.as_array().map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default() };
    if let Some(offs) = meta.get("offsets") {
        for l in &hm.layers {
            r.param_mismatch += (u64s(&offs[l.layer_id.to_string()]) != l.offsets) as usize;
        }
    }
    if let Some(ne) = meta.get("num_embeddings") {
        r.param_mismatch += (u64s(ne) != hm.layers.iter().map(|l| l.total_rows).collect::<Vec<_>>()) as usize;
    }
    for l in &hm.layers {
        let key = l.layer_id.to_string();
        let rm: Vec<i64> = meta["multipliers"][&key].as_array().map(|a| a.iter().filter_map(Value::as_i64).collect()).unwrap_or_default();
        let rp: Vec<Vec<u64>> = meta["primes"][&key].as_array().map(|a| a.iter().map(u64s).collect()).unwrap_or_default();
        r.param_mismatch += (rm != l.multipliers) as usize + (rp != l.primes) as usize;
    }

    let cases = std::fs::read_to_string(dir.join("cases.jsonl")).map_err(|e| EngramError(format!("cases.jsonl: {e}")))?;
    let mut out = Vec::new();
    for line in cases.lines() {
        let c: Value = serde_json::from_str(line).map_err(|e| EngramError(format!("cases.jsonl: {e}")))?;
        let ids: Vec<i64> = c["ids"].as_array().map(|a| a.iter().filter_map(Value::as_i64).collect()).unwrap_or_default();
        for l in &hm.layers {
            hm.hash(l, &ids, &mut out);
            let want: Vec<i64> = c["hashes"][l.layer_id.to_string()]
                .as_array()
                .map(|rows| rows.iter().flat_map(|row| row.as_array().into_iter().flatten().filter_map(Value::as_i64)).collect())
                .unwrap_or_default();
            r.indices += want.len();
            r.cases += 1;
            r.bad += (want != out) as usize;
        }
    }
    Ok(r)
}
