/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Tokenizer compression and multi-head N-gram hashing, bit-exact with
//! DeepSeek-V4.1-Flash `inference/engram.py` and `engram_demo_v1.py`.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::numpy_rng;

/// Engram hashing configuration (the hashing-relevant part of a model config).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngramConfig {
    /// Base table size per order, index 0 = 2-grams. Length = max_ngram - 1.
    pub engram_vocab_size: Vec<u64>,
    /// Longest N-gram hashed (orders 2..=max_ngram_size).
    pub max_ngram_size: usize,
    /// Hash heads per order.
    pub n_head_per_ngram: usize,
    /// Transformer layers that carry an Engram table.
    pub layer_ids: Vec<u64>,
    /// Raw pad id; compressed before use.
    pub pad_id: u32,
    /// Seed of the multiplier generator (0 for V4.1-Flash).
    pub seed: u64,
    /// Edge semantics.
    pub semantics: Semantics,
}

/// The demo and the shipped DeepSeek-V4.1-Flash code agree on the hash but
/// differ at the edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Semantics {
    /// `engram_demo_v1.py`: per-head indices (no table offset); negative ids
    /// are hashed as values.
    Demo,
    /// V4.1-Flash `inference/engram.py`: indices include the per-(order, head)
    /// row offset into the layer's single table; a negative id is DEAD (image
    /// span): look-back stops there and blocked positions read the pad id.
    V41,
}

/// Raw token id -> canonical (compressed) id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedTokenizer {
    /// `lookup[raw_id]` = compressed id.
    pub lookup: Vec<u32>,
    /// Size of the compressed vocabulary.
    pub num_new_token: u32,
}

impl CompressedTokenizer {
    /// Use a precomputed table (e.g. GGUF `engram.token_map`). The size is
    /// `max + 1` when not given.
    pub fn from_lookup(lookup: Vec<u32>, num_new_token: Option<u32>) -> Self {
        let n = num_new_token.unwrap_or_else(|| lookup.iter().max().map_or(0, |m| m + 1));
        Self { lookup, num_new_token: n }
    }

    /// Build the table from a Hugging Face tokenizer, decoding each id with
    /// the raw backend (never through `transformers.decode`: its class swaps
    /// can change the decoder and silently change the table).
    #[cfg(feature = "tokenizer")]
    pub fn from_tokenizer(tok: &tokenizers::Tokenizer) -> tokenizers::Result<Self> {
        use tokenizers::normalizers::{
            replace::Replace, replace::ReplacePattern, strip::Strip, unicode::NFD, unicode::NFKC,
            utils::Sequence, Lowercase, NormalizerWrapper, StripAccents,
        };
        use tokenizers::{NormalizedString, Normalizer};

        const SENTINEL: &str = "\u{E000}";
        let norm = Sequence::new(vec![
            NormalizerWrapper::NFKC(NFKC),
            NormalizerWrapper::NFD(NFD),
            NormalizerWrapper::StripAccents(StripAccents),
            NormalizerWrapper::Lowercase(Lowercase),
            NormalizerWrapper::Replace(Replace::new(ReplacePattern::Regex(r"[ \t\r\n]+".into()), " ")?),
            NormalizerWrapper::Replace(Replace::new(ReplacePattern::Regex(r"^ $".into()), SENTINEL)?),
            NormalizerWrapper::StripNormalizer(Strip::new(true, true)),
            NormalizerWrapper::Replace(Replace::new(ReplacePattern::String(SENTINEL.into()), " ")?),
        ]);

        // len(tokenizer) in transformers = base vocab + added tokens.
        let vocab_size = tok.get_vocab_size(true) as u32;
        let mut key2new: HashMap<String, u32> = HashMap::new();
        let mut lookup = Vec::with_capacity(vocab_size as usize);
        for tid in 0..vocab_size {
            let text = tok.decode(&[tid], false)?;
            let key = if text.contains('\u{FFFD}') {
                tok.id_to_token(tid).unwrap_or_default()
            } else {
                let mut n = NormalizedString::from(text.as_str());
                norm.normalize(&mut n)?;
                let s = n.get();
                if s.is_empty() {
                    text
                } else {
                    s.to_owned()
                }
            };
            let next = key2new.len() as u32;
            lookup.push(*key2new.entry(key).or_insert(next));
        }
        Ok(Self {
            lookup,
            num_new_token: key2new.len() as u32,
        })
    }

    /// Negative ids pass through unchanged (DEAD tokens / demo semantics).
    /// Ids beyond the table also pass through, as negative "unknown" markers
    /// would; callers validate ids against the tokenizer.
    #[inline]
    pub fn compress(&self, id: i64) -> i64 {
        if id >= 0 {
            match self.lookup.get(id as usize) {
                Some(&c) => c as i64,
                None => -1,
            }
        } else {
            id
        }
    }
}

pub(crate) fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    if n % 3 == 0 {
        return n == 3;
    }
    let mut i = 5u64;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

/// Per-layer hashing state: multipliers and per-(order, head) prime table sizes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerHash {
    /// Transformer layer id.
    pub layer_id: u64,
    /// One odd multiplier per look-back k in 0..max_ngram.
    pub multipliers: Vec<i64>,
    /// `primes[order - 2][head]`.
    pub primes: Vec<Vec<u64>>,
    /// Row offset of each (order, head) table inside the layer's single
    /// concatenated embedding, flattened order-major.
    pub offsets: Vec<u64>,
    /// Rows of the layer's table (sum of the primes).
    pub total_rows: u64,
}

/// The full hashing state of a model: token map plus per-layer parameters.
#[derive(Clone, Debug)]
pub struct NgramHashMapping {
    /// Configuration it was built from.
    pub cfg: EngramConfig,
    /// Raw -> compressed ids.
    pub tokenizer: CompressedTokenizer,
    /// Compressed pad id.
    pub pad_id: i64,
    /// One entry per Engram layer, in `cfg.layer_ids` order.
    pub layers: Vec<LayerHash>,
}

impl NgramHashMapping {
    /// Compute multipliers, primes and offsets (they are derived, never stored).
    pub fn new(cfg: EngramConfig, tokenizer: CompressedTokenizer) -> Self {
        let pad_id = tokenizer.compress(cfg.pad_id as i64);
        let max_long = i64::MAX as u64;
        let m_max = max_long / (tokenizer.num_new_token.max(1) as u64);
        let half_bound = (m_max / 2).max(1);
        const PRIME_1: u64 = 10007;

        // The prime search shares one `seen` set across ALL layers and orders,
        // so later layers get later primes. Order matters: layer-major.
        let mut seen = HashSet::new();
        let mut layers = Vec::new();
        for &layer_id in &cfg.layer_ids {
            let mut rng = numpy_rng::Generator::from_seed(cfg.seed + PRIME_1 * layer_id);
            let multipliers: Vec<i64> = (0..cfg.max_ngram_size)
                .map(|_| rng.integers_i64(0, half_bound as i64) * 2 + 1)
                .collect();
            let mut primes = Vec::new();
            for n in 2..=cfg.max_ngram_size {
                let mut start = cfg.engram_vocab_size[n - 2] - 1;
                let mut heads = Vec::new();
                for _ in 0..cfg.n_head_per_ngram {
                    let mut c = start + 1;
                    while !(is_prime(c) && !seen.contains(&c)) {
                        c += 1;
                    }
                    seen.insert(c);
                    heads.push(c);
                    start = c;
                }
                primes.push(heads);
            }
            let mut offsets = Vec::new();
            let mut acc = 0u64;
            for p in primes.iter().flatten() {
                offsets.push(acc);
                acc += p;
            }
            layers.push(LayerHash {
                layer_id,
                multipliers,
                primes,
                offsets,
                total_rows: acc,
            });
        }
        Self {
            cfg,
            tokenizer,
            pad_id,
            layers,
        }
    }

    /// The hashing state of one layer.
    pub fn layer(&self, layer_id: u64) -> Option<&LayerHash> {
        self.layers.iter().find(|l| l.layer_id == layer_id)
    }

    /// Hash columns per position per layer: `(max_ngram - 1) * n_head`.
    pub fn n_columns(&self) -> usize {
        (self.cfg.max_ngram_size - 1) * self.cfg.n_head_per_ngram
    }

    /// Per-head table indices for one sequence: `out[t * H + h]`, with
    /// `H = n_columns()` and h order-major, as the reference code stacks them.
    /// `compressed` must already be compressed ids.
    pub fn hash_compressed(&self, layer: &LayerHash, compressed: &[i64], out: &mut Vec<i64>) {
        out.clear();
        out.reserve(compressed.len() * self.n_columns());
        let m = &layer.multipliers;
        let v41 = self.cfg.semantics == Semantics::V41;
        let mut toks = vec![0i64; self.cfg.max_ngram_size];
        for t in 0..compressed.len() {
            let mut blocked = false;
            for (k, slot) in toks.iter_mut().enumerate() {
                if v41 {
                    blocked |= t < k || compressed[t - k] < 0;
                    *slot = if blocked { self.pad_id } else { compressed[t - k] };
                } else {
                    *slot = if t >= k { compressed[t - k] } else { self.pad_id };
                }
            }
            let mut col = 0usize;
            // NumPy int64 arithmetic wraps; it cannot overflow for in-range ids,
            // but wrapping keeps us identical even for negative ids.
            let mut mix = toks[0].wrapping_mul(m[0]);
            for n in 2..=self.cfg.max_ngram_size {
                mix ^= toks[n - 1].wrapping_mul(m[n - 1]);
                for &p in &layer.primes[n - 2] {
                    let idx = mix.rem_euclid(p as i64); // numpy/torch % on int64 = floor mod
                    out.push(if v41 { idx + layer.offsets[col] as i64 } else { idx });
                    col += 1;
                }
            }
        }
    }

    /// Like [`hash_compressed`](Self::hash_compressed), from raw token ids.
    pub fn hash(&self, layer: &LayerHash, raw_ids: &[i64], out: &mut Vec<i64>) {
        let c: Vec<i64> = raw_ids.iter().map(|&x| self.tokenizer.compress(x)).collect();
        self.hash_compressed(layer, &c, out);
    }
}
