/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! DeepSeek Engram addressing for kleos (docs/specs/engram-overlay.md).
//!
//! Engram addresses its tables only from token ids, so the full address of
//! every row a token reads is known before the model runs:
//!
//! 1. compress raw ids to canonical ids ([`CompressedTokenizer`]),
//! 2. for each order n in 2..=max_ngram, `mix = XOR_k (tok[t-k] * mult[layer][k])`
//!    over k in 0..n, with left padding by the compressed pad id,
//! 3. for each head j of that order, `idx = mix mod prime[layer][n][j]`
//!    (floor-mod), plus the (order, head) offset into the layer's table.
//!
//! The multipliers come from NumPy's `default_rng(seed + 10007 * layer)`
//! (SeedSequence -> PCG64 -> bounded Lemire draws), reproduced bit-exactly in
//! [`numpy_rng`]. The whole computation is parity-checked against
//! DeepSeek-V4.1-Flash `inference/engram.py`, `engram_demo_v1.py` and the
//! littlemex nanogpt `NgramHash` (see `tests/parity.rs`).
//!
//! An [`EngramSpec`] adds the row dimension and a **fingerprint**: a hash
//! over everything that determines addressing. A row id means nothing without
//! it, so kleos keys every overlay row by it and refuses to mix specs.

use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

mod hash;
pub mod numpy_rng;
#[cfg(feature = "tokenizer")]
pub mod parity;
pub mod safetensors;

pub use hash::{CompressedTokenizer, EngramConfig, LayerHash, NgramHashMapping, Semantics};

/// Errors of this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngramError(pub String);

impl fmt::Display for EngramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EngramError {}

/// Result alias.
pub type Result<T> = std::result::Result<T, EngramError>;

macro_rules! err {
    ($($t:tt)*) => { EngramError(format!($($t)*)) };
}

/// Values a model config may leave out (the nanogpt config has no pad id) or
/// that a caller wants to force.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecOverrides {
    /// Raw pad id.
    pub pad_id: Option<u32>,
    /// Multiplier seed.
    pub seed: Option<u64>,
    /// Edge semantics.
    pub semantics: Option<Semantics>,
}

/// What a model config says about Engram, before the token map is known.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigFields {
    /// Hashing configuration.
    pub cfg: EngramConfig,
    /// Dimension of one table row (one head's slice).
    pub row_dim: usize,
    /// `engram_num_embeddings`, when the config states it: checked against
    /// the computed primes.
    pub expect_rows: Option<Vec<u64>>,
    /// `engram_compressed_vocab_size`, when stated: checked against the
    /// token map.
    pub expect_compressed_vocab: Option<u32>,
}

fn get<'a>(v: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|k| v.get(*k)).filter(|x| !x.is_null())
}

fn as_u64(v: &Value, what: &str) -> Result<u64> {
    v.as_u64().ok_or_else(|| err!("Engram config field '{what}' must be a non-negative integer, got {v}"))
}

impl ConfigFields {
    /// Read the Engram fields of a model's `config.json`. Recognised layouts:
    /// HF DeepSeek-V4.1 (`text_config.engram_*`), the littlemex nanogpt
    /// checkpoint (`model.engram_*`), a top-level `engram_*` config, and the
    /// parity dumps of `parity_ref.py` (`config.{layer_ids, ...}`).
    pub fn from_model_config(root: &Value, ov: &SpecOverrides) -> Result<Self> {
        let has = |v: &Value| v.get("engram_layer_ids").is_some() || v.get("layer_ids").is_some();
        let c = ["text_config", "model", "config"]
            .iter()
            .filter_map(|k| root.get(*k))
            .find(|v| has(v))
            .or(if has(root) { Some(root) } else { None })
            .ok_or_else(|| err!("no Engram fields (engram_layer_ids) found in the model config"))?;

        let layer_ids: Vec<u64> = get(c, &["engram_layer_ids", "layer_ids"])
            .and_then(Value::as_array)
            .ok_or_else(|| err!("Engram config: engram_layer_ids must be a list"))?
            .iter()
            .map(|v| as_u64(v, "engram_layer_ids"))
            .collect::<Result<_>>()?;
        let max_ngram_size = as_u64(
            get(c, &["engram_max_ngram_size", "max_ngram_size"]).ok_or_else(|| err!("Engram config: missing engram_max_ngram_size"))?,
            "engram_max_ngram_size",
        )? as usize;
        let n_head = as_u64(
            get(c, &["engram_n_heads", "n_head_per_ngram"]).ok_or_else(|| err!("Engram config: missing engram_n_heads"))?,
            "engram_n_heads",
        )? as usize;
        if max_ngram_size < 2 || n_head == 0 || layer_ids.is_empty() {
            bail_cfg(max_ngram_size, n_head)?;
        }
        let vocab = get(c, &["engram_vocab_size", "engram_rows_per_head"])
            .ok_or_else(|| err!("Engram config: missing engram_vocab_size"))?;
        let engram_vocab_size: Vec<u64> = match vocab {
            Value::Array(a) => a.iter().map(|v| as_u64(v, "engram_vocab_size")).collect::<Result<_>>()?,
            v => vec![as_u64(v, "engram_vocab_size")?; max_ngram_size - 1],
        };
        if engram_vocab_size.len() != max_ngram_size - 1 || engram_vocab_size.iter().any(|&s| s < 2) {
            return Err(err!(
                "Engram config: engram_vocab_size needs {} entries of at least 2, got {engram_vocab_size:?}",
                max_ngram_size - 1
            ));
        }
        let row_dim = match get(c, &["engram_head_dim"]) {
            Some(v) => as_u64(v, "engram_head_dim")? as usize,
            None => match get(c, &["n_embed_per_ngram"]) {
                Some(v) => as_u64(v, "n_embed_per_ngram")? as usize / n_head,
                // unknown (parity dumps); EngramSpec::from_parts refuses 0
                None => 0,
            },
        };
        let pad_id = match ov.pad_id {
            Some(p) => p,
            None => as_u64(
                get(c, &["engram_pad_token_id", "engram_pad_id", "pad_id"]).ok_or_else(|| {
                    err!("Engram config has no pad id (engram_pad_token_id); pass it explicitly (the nanogpt checkpoint pads with <|endoftext|> = 50256)")
                })?,
                "engram_pad_token_id",
            )? as u32,
        };
        let seed = match ov.seed {
            Some(s) => s,
            None => get(c, &["engram_seed", "seed"]).map(|v| as_u64(v, "seed")).transpose()?.unwrap_or(0),
        };
        let semantics = match ov.semantics {
            Some(s) => s,
            None => match get(c, &["semantics", "engram_semantics"]).and_then(Value::as_str) {
                Some("demo") => Semantics::Demo,
                Some("v41") | None => {
                    // parity dumps of the demo carry no `semantics` and no offsets
                    if root.get("offsets").is_none() && root.get("config").is_some() {
                        Semantics::Demo
                    } else {
                        Semantics::V41
                    }
                }
                Some(other) => return Err(err!("unknown Engram semantics '{other}': expected v41 or demo")),
            },
        };
        let expect_rows = get(c, &["engram_num_embeddings"])
            .or_else(|| root.get("num_embeddings"))
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|v| as_u64(v, "engram_num_embeddings")).collect::<Result<Vec<_>>>())
            .transpose()?;
        let expect_compressed_vocab = get(c, &["engram_compressed_vocab_size"])
            .or_else(|| root.get("num_new_token"))
            .map(|v| as_u64(v, "engram_compressed_vocab_size").map(|x| x as u32))
            .transpose()?;
        Ok(Self {
            cfg: EngramConfig {
                engram_vocab_size,
                max_ngram_size,
                n_head_per_ngram: n_head,
                layer_ids,
                pad_id,
                seed,
                semantics,
            },
            row_dim,
            expect_rows,
            expect_compressed_vocab,
        })
    }
}

fn bail_cfg(max_ngram_size: usize, n_head: usize) -> Result<()> {
    Err(err!(
        "Engram config: need max_ngram_size >= 2, at least one head and one layer (got max_ngram_size {max_ngram_size}, heads {n_head})"
    ))
}

/// One table row read at one position: `(layer, row)` is the overlay key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowHit {
    /// Token position.
    pub pos: usize,
    /// Transformer layer id.
    pub layer: u64,
    /// N-gram order (2..=max_ngram).
    pub order: usize,
    /// Hash head within the order.
    pub head: usize,
    /// Row id in the layer's table (with the V4.1 offset; per-head index under demo semantics).
    pub row: i64,
}

/// A model's complete Engram addressing: hashing state, row dimension and
/// fingerprint.
#[derive(Clone)]
pub struct EngramSpec {
    /// Hashing state.
    pub mapping: NgramHashMapping,
    /// Dimension of one table row.
    pub row_dim: usize,
    fingerprint: String,
    #[cfg(feature = "tokenizer")]
    tokenizer: Option<std::sync::Arc<tokenizers::Tokenizer>>,
}

impl fmt::Debug for EngramSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EngramSpec")
            .field("fingerprint", &self.fingerprint)
            .field("cfg", &self.mapping.cfg)
            .field("row_dim", &self.row_dim)
            .finish()
    }
}

/// Version tag mixed into every fingerprint; bump on any encoding change.
pub const FINGERPRINT_VERSION: &str = "kleos-engram-fp/v1";

impl EngramSpec {
    /// Build from parts, checking the config's stated sizes against the
    /// computed ones (a mismatch means the spec drifted from the weights).
    pub fn from_parts(fields: ConfigFields, tokens: CompressedTokenizer) -> Result<Self> {
        if (fields.cfg.pad_id as usize) >= tokens.lookup.len() {
            return Err(err!(
                "Engram pad id {} is outside the token map ({} ids)",
                fields.cfg.pad_id,
                tokens.lookup.len()
            ));
        }
        if let Some(n) = fields.expect_compressed_vocab {
            if n != tokens.num_new_token {
                return Err(err!(
                    "compressed vocab mismatch: the config says {n}, the tokenizer gives {}",
                    tokens.num_new_token
                ));
            }
        }
        if fields.row_dim == 0 {
            return Err(err!("Engram row dimension unknown: the config has no engram_head_dim"));
        }
        let mapping = NgramHashMapping::new(fields.cfg, tokens);
        if let Some(rows) = &fields.expect_rows {
            let ours: Vec<u64> = mapping.layers.iter().map(|l| l.total_rows).collect();
            if rows != &ours {
                return Err(err!("table rows per layer mismatch: the config says {rows:?}, the primes give {ours:?}"));
            }
        }
        let fingerprint = fingerprint(&mapping, fields.row_dim);
        Ok(Self {
            mapping,
            row_dim: fields.row_dim,
            fingerprint,
            #[cfg(feature = "tokenizer")]
            tokenizer: None,
        })
    }

    /// Load from a model's `config.json` and `tokenizer.json`.
    #[cfg(feature = "tokenizer")]
    pub fn from_files(config_json: &Path, tokenizer_json: &Path, ov: &SpecOverrides) -> Result<Self> {
        let cfg: Value = serde_json::from_str(
            &std::fs::read_to_string(config_json).map_err(|e| err!("reading {}: {e}", config_json.display()))?,
        )
        .map_err(|e| err!("parsing {}: {e}", config_json.display()))?;
        let tok = tokenizers::Tokenizer::from_file(tokenizer_json)
            .map_err(|e| err!("loading {}: {e}", tokenizer_json.display()))?;
        Self::from_config_and_tokenizer(&cfg, tok, ov)
    }

    /// Load from a parsed `config.json` and a tokenizer.
    #[cfg(feature = "tokenizer")]
    pub fn from_config_and_tokenizer(cfg: &Value, tok: tokenizers::Tokenizer, ov: &SpecOverrides) -> Result<Self> {
        let fields = ConfigFields::from_model_config(cfg, ov)?;
        let tokens = CompressedTokenizer::from_tokenizer(&tok).map_err(|e| err!("building the token map: {e}"))?;
        let mut spec = Self::from_parts(fields, tokens)?;
        spec.tokenizer = Some(std::sync::Arc::new(tok));
        Ok(spec)
    }

    /// Load from a `config.json` path and a precomputed token map (e.g. GGUF
    /// `engram.token_map`); text hashing is then unavailable.
    pub fn from_config_and_token_map(config_json: &Path, token_map: Vec<u32>, ov: &SpecOverrides) -> Result<Self> {
        let cfg: Value = serde_json::from_str(
            &std::fs::read_to_string(config_json).map_err(|e| err!("reading {}: {e}", config_json.display()))?,
        )
        .map_err(|e| err!("parsing {}: {e}", config_json.display()))?;
        Self::from_parts(
            ConfigFields::from_model_config(&cfg, ov)?,
            CompressedTokenizer::from_lookup(token_map, None),
        )
    }

    /// The addressing fingerprint, `eg1:` + 32 hex digits.
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Engram layer ids.
    pub fn layer_ids(&self) -> Vec<u64> {
        self.mapping.layers.iter().map(|l| l.layer_id).collect()
    }

    /// Everything the fingerprint covers except the token map, as JSON
    /// (stored with a registered model; the token map is summarised by size).
    pub fn summary(&self) -> Value {
        let m = &self.mapping;
        json!({
            "fingerprint": self.fingerprint,
            "fingerprint_version": FINGERPRINT_VERSION,
            "semantics": m.cfg.semantics,
            "layer_ids": self.layer_ids(),
            "max_ngram_size": m.cfg.max_ngram_size,
            "n_head_per_ngram": m.cfg.n_head_per_ngram,
            "engram_vocab_size": m.cfg.engram_vocab_size,
            "row_dim": self.row_dim,
            "seed": m.cfg.seed,
            "pad_id": m.cfg.pad_id,
            "pad_id_compressed": m.pad_id,
            "raw_vocab": m.tokenizer.lookup.len(),
            "compressed_vocab": m.tokenizer.num_new_token,
            "table_rows": m.layers.iter().map(|l| l.total_rows).collect::<Vec<_>>(),
            "multipliers": m.layers.iter().map(|l| l.multipliers.clone()).collect::<Vec<_>>(),
            "primes": m.layers.iter().map(|l| l.primes.clone()).collect::<Vec<_>>(),
        })
    }

    /// Every row read at every position of a raw-id sequence, layer-major.
    pub fn rows_for_ids(&self, raw_ids: &[i64]) -> Vec<RowHit> {
        let m = &self.mapping;
        let h = m.n_columns();
        let heads = m.cfg.n_head_per_ngram;
        let compressed: Vec<i64> = raw_ids.iter().map(|&x| m.tokenizer.compress(x)).collect();
        let mut out = Vec::with_capacity(raw_ids.len() * h * m.layers.len());
        let mut buf = Vec::new();
        for l in &m.layers {
            m.hash_compressed(l, &compressed, &mut buf);
            for (i, &row) in buf.iter().enumerate() {
                let col = i % h;
                out.push(RowHit {
                    pos: i / h,
                    layer: l.layer_id,
                    order: 2 + col / heads,
                    head: col % heads,
                    row,
                });
            }
        }
        out
    }

    /// Tokenize `text` with the model's tokenizer.
    #[cfg(feature = "tokenizer")]
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i64>> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| err!("this Engram spec was built without a tokenizer; text cannot be hashed"))?;
        let enc = tok.encode(text, add_special_tokens).map_err(|e| err!("tokenizing: {e}"))?;
        Ok(enc.get_ids().iter().map(|&i| i as i64).collect())
    }

    /// Every row the model reads for `text`.
    #[cfg(feature = "tokenizer")]
    pub fn rows_for_text(&self, text: &str, add_special_tokens: bool) -> Result<(Vec<i64>, Vec<RowHit>)> {
        let ids = self.encode(text, add_special_tokens)?;
        let rows = self.rows_for_ids(&ids);
        Ok((ids, rows))
    }
}

/// The fingerprint: SHA-256 over a fixed little-endian encoding of the
/// semantics, token map, pad id, orders, heads, row dimension, and per layer
/// its id, multipliers, primes and offsets. Displayed as `eg1:` + the first
/// 16 bytes in hex.
pub fn fingerprint(m: &NgramHashMapping, row_dim: usize) -> String {
    let mut h = Sha256::new();
    h.update(FINGERPRINT_VERSION.as_bytes());
    h.update([0u8, matches!(m.cfg.semantics, Semantics::V41) as u8]);
    let u = |h: &mut Sha256, x: u64| h.update(x.to_le_bytes());
    u(&mut h, m.tokenizer.lookup.len() as u64);
    for &t in &m.tokenizer.lookup {
        h.update(t.to_le_bytes());
    }
    u(&mut h, m.tokenizer.num_new_token as u64);
    h.update(m.pad_id.to_le_bytes());
    u(&mut h, m.cfg.max_ngram_size as u64);
    u(&mut h, m.cfg.n_head_per_ngram as u64);
    u(&mut h, row_dim as u64);
    u(&mut h, m.layers.len() as u64);
    for l in &m.layers {
        u(&mut h, l.layer_id);
        for &x in &l.multipliers {
            h.update(x.to_le_bytes());
        }
        for &p in l.primes.iter().flatten() {
            u(&mut h, p);
        }
        for &o in &l.offsets {
            u(&mut h, o);
        }
    }
    let d = h.finalize();
    let hex: String = d[..16].iter().map(|b| format!("{b:02x}")).collect();
    format!("eg1:{hex}")
}

/// bf16 <-> f32 helpers for overlay vectors (round-to-nearest-even).
pub mod bf16 {
    /// f32 -> bf16 bits.
    pub fn from_f32(x: &[f32]) -> Vec<u16> {
        x.iter().map(|&v| half::bf16::from_f32(v).to_bits()).collect()
    }
    /// bf16 bits -> f32.
    pub fn to_f32(x: &[u16]) -> Vec<f32> {
        x.iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect()
    }
    /// bf16 bits -> little-endian bytes.
    pub fn to_le_bytes(x: &[u16]) -> Vec<u8> {
        x.iter().flat_map(|b| b.to_le_bytes()).collect()
    }
    /// Little-endian bytes -> bf16 bits (odd trailing byte ignored).
    pub fn from_le_bytes(b: &[u8]) -> Vec<u16> {
        b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(seed: u64, row_dim: usize) -> EngramSpec {
        // 40 raw ids folding to 20 compressed ids
        let lookup: Vec<u32> = (0..40).map(|i| i / 2).collect();
        let fields = ConfigFields {
            cfg: EngramConfig {
                engram_vocab_size: vec![101, 101],
                max_ngram_size: 3,
                n_head_per_ngram: 2,
                layer_ids: vec![1, 5],
                pad_id: 0,
                seed,
                semantics: Semantics::V41,
            },
            row_dim,
            expect_rows: None,
            expect_compressed_vocab: None,
        };
        EngramSpec::from_parts(fields, CompressedTokenizer::from_lookup(lookup, None)).unwrap()
    }

    #[test]
    fn v41_multipliers_and_primes_from_the_seed_alone() {
        // DeepSeek-V4.1-Flash: compressed vocab 99,092, seed 0, layers [1, 14],
        // orders 2..4, 8 heads, 16M rows per head (docs/research/engram.md §1).
        let cfg = EngramConfig {
            engram_vocab_size: vec![16_000_000; 3],
            max_ngram_size: 4,
            n_head_per_ngram: 8,
            layer_ids: vec![1, 14],
            pad_id: 2,
            seed: 0,
            semantics: Semantics::V41,
        };
        let m = NgramHashMapping::new(cfg, CompressedTokenizer::from_lookup(vec![0, 1, 2], Some(99_092)));
        assert_eq!(m.layers[0].multipliers, [76632096046245, 4839876093313, 35959672319349, 73987337458391]);
        assert_eq!(m.layers[1].multipliers, [67716810739261, 51510806800915, 30921347202721, 82619226485591]);
        assert_eq!(m.layers[0].total_rows, 384_006_168);
        assert_eq!(m.layers[1].total_rows, 384_016_682);
        assert_eq!(m.layers[0].primes[0][0], 16_000_057);
        assert_eq!(*m.layers[1].primes.last().unwrap().last().unwrap(), 16_000_889);
    }

    #[test]
    fn fingerprint_is_stable_and_sensitive() {
        let a = synthetic(0, 8);
        assert_eq!(a.fingerprint(), synthetic(0, 8).fingerprint());
        assert!(a.fingerprint().starts_with("eg1:") && a.fingerprint().len() == 36);
        assert_ne!(a.fingerprint(), synthetic(1, 8).fingerprint(), "seed changes multipliers");
        assert_ne!(a.fingerprint(), synthetic(0, 16).fingerprint(), "row dim");
        let mut f = ConfigFields::from_model_config(
            &json!({"engram_layer_ids": [1, 5], "engram_max_ngram_size": 3, "engram_n_heads": 2,
                    "engram_vocab_size": 101, "engram_head_dim": 8, "engram_pad_token_id": 0}),
            &Default::default(),
        )
        .unwrap();
        let mut lookup: Vec<u32> = (0..40).map(|i| i / 2).collect();
        assert_eq!(
            EngramSpec::from_parts(f.clone(), CompressedTokenizer::from_lookup(lookup.clone(), None)).unwrap().fingerprint(),
            a.fingerprint(),
            "config.json path and struct path agree"
        );
        lookup.swap(3, 5);
        assert_ne!(
            EngramSpec::from_parts(f.clone(), CompressedTokenizer::from_lookup(lookup, None)).unwrap().fingerprint(),
            a.fingerprint(),
            "token map"
        );
        f.expect_rows = Some(vec![1, 2]);
        assert!(EngramSpec::from_parts(f, CompressedTokenizer::from_lookup(vec![0; 40], None)).is_err());
    }

    #[test]
    fn rows_for_ids_layout() {
        let s = synthetic(0, 8);
        let hits = s.rows_for_ids(&[3, 7, 9]);
        // 2 layers x 3 positions x (2 orders x 2 heads)
        assert_eq!(hits.len(), 2 * 3 * 4);
        let l1: Vec<_> = hits.iter().filter(|h| h.layer == 1 && h.pos == 2).collect();
        assert_eq!(l1.iter().map(|h| (h.order, h.head)).collect::<Vec<_>>(), [(2, 0), (2, 1), (3, 0), (3, 1)]);
        let layer = s.mapping.layer(1).unwrap();
        for h in &l1 {
            let col = (h.order - 2) * 2 + h.head;
            let lo = layer.offsets[col] as i64;
            assert!(h.row >= lo && h.row < lo + layer.primes[h.order - 2][h.head] as i64);
        }
        // raw ids 6 and 7 compress alike, so they hash alike
        assert_eq!(
            s.rows_for_ids(&[3, 6]).iter().map(|h| h.row).collect::<Vec<_>>(),
            s.rows_for_ids(&[2, 7]).iter().map(|h| h.row).collect::<Vec<_>>()
        );
    }

    #[test]
    fn nanogpt_config_needs_a_pad_id() {
        let cfg = json!({"model": {"vocab_size": 50257, "engram_layer_ids": [1], "engram_max_ngram_size": 3,
                         "engram_n_heads": 8, "engram_rows_per_head": 65536, "engram_head_dim": 64}});
        assert!(ConfigFields::from_model_config(&cfg, &Default::default()).is_err());
        let f = ConfigFields::from_model_config(&cfg, &SpecOverrides { pad_id: Some(50256), ..Default::default() }).unwrap();
        assert_eq!(f.cfg.engram_vocab_size, vec![65536, 65536]);
        assert_eq!(f.row_dim, 64);
        assert_eq!(f.cfg.semantics, Semantics::V41);
    }
}
