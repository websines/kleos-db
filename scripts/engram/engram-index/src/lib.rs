//! DeepSeek Engram index computation, bit-exact with `engram_demo_v1.py`.
//!
//! Engram addresses its tables only from token ids, so the full address of
//! every row a token reads is known before the model runs:
//!
//! 1. compress raw ids to canonical ids (`CompressedTokenizer`),
//! 2. for each order n in 2..=max_ngram, `mix = XOR_k (tok[t-k] * mult[layer][k])`
//!    over k in 0..n, with left padding by the compressed pad id,
//! 3. for each head j of that order, `idx = mix mod prime[layer][n][j]`
//!    (Python floor-mod), each head having its own distinct prime table size.
//!
//! The multipliers come from NumPy's `default_rng(seed + 10007 * layer)`
//! (SeedSequence -> PCG64 -> bounded Lemire draws), reproduced here.

use std::collections::HashMap;

pub mod numpy_rng;

/// Engram hashing configuration (the hashing-relevant part of `EngramConfig`).
#[derive(Clone, Debug)]
pub struct EngramConfig {
    /// Base table size per order, index 0 = 2-grams. Length = max_ngram - 1.
    pub engram_vocab_size: Vec<u64>,
    pub max_ngram_size: usize,
    pub n_head_per_ngram: usize,
    pub layer_ids: Vec<u64>,
    /// Raw pad id; compressed before use, as in the demo.
    pub pad_id: Option<u32>,
    pub seed: u64,
    pub semantics: Semantics,
}

/// The demo and the shipped DeepSeek-V4.1-Flash code agree on the hash but
/// differ at the edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Semantics {
    /// `engram_demo_v1.py`: per-head indices (no table offset); negative ids
    /// are hashed as values.
    Demo,
    /// V4.1-Flash `inference/engram.py`: indices include the per-(order, head)
    /// row offset into the layer table; a negative id is DEAD (image span):
    /// look-back stops there and blocked positions read the pad id.
    V41,
}

/// Raw token id -> canonical id, built from a HF tokenizer exactly as the demo does.
pub struct CompressedTokenizer {
    pub lookup: Vec<u32>,
    pub num_new_token: u32,
}

impl CompressedTokenizer {
    /// Use a precomputed table (e.g. GGUF `engram.token_map`).
    pub fn from_lookup(lookup: Vec<u32>, num_new_token: u32) -> Self {
        Self { lookup, num_new_token }
    }

    pub fn from_tokenizer(tok: &tokenizers::Tokenizer) -> tokenizers::Result<Self> {
        use tokenizers::normalizers::{
            replace::Replace, strip::Strip, unicode::NFD, unicode::NFKC, utils::Sequence,
            Lowercase, NormalizerWrapper, StripAccents,
        };
        use tokenizers::{NormalizedString, Normalizer};

        const SENTINEL: &str = "\u{E000}";
        let norm = Sequence::new(vec![
            NormalizerWrapper::NFKC(NFKC),
            NormalizerWrapper::NFD(NFD),
            NormalizerWrapper::StripAccents(StripAccents),
            NormalizerWrapper::Lowercase(Lowercase),
            NormalizerWrapper::Replace(Replace::new(
                tokenizers::normalizers::replace::ReplacePattern::Regex(r"[ \t\r\n]+".into()),
                " ",
            )?),
            NormalizerWrapper::Replace(Replace::new(
                tokenizers::normalizers::replace::ReplacePattern::Regex(r"^ $".into()),
                SENTINEL,
            )?),
            NormalizerWrapper::StripNormalizer(Strip::new(true, true)),
            NormalizerWrapper::Replace(Replace::new(
                tokenizers::normalizers::replace::ReplacePattern::String(SENTINEL.into()),
                " ",
            )?),
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
                if s.is_empty() { text } else { s.to_owned() }
            };
            let next = key2new.len() as u32;
            lookup.push(*key2new.entry(key).or_insert(next));
        }
        Ok(Self { lookup, num_new_token: key2new.len() as u32 })
    }

    /// Negative ids pass through unchanged, as in the demo.
    #[inline]
    pub fn compress(&self, id: i64) -> i64 {
        if id >= 0 { self.lookup[id as usize] as i64 } else { id }
    }
}

fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n % 2 == 0 { return n == 2; }
    if n % 3 == 0 { return n == 3; }
    let mut i = 5u64;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 { return false; }
        i += 6;
    }
    true
}

/// Per-layer hashing state: multipliers and per-(order, head) prime table sizes.
#[derive(Clone, Debug)]
pub struct LayerHash {
    pub layer_id: u64,
    pub multipliers: Vec<i64>,
    /// primes[order - 2][head]
    pub primes: Vec<Vec<u64>>,
    /// Row offset of each (order, head) table inside the layer's single
    /// concatenated embedding (MultiHeadEmbedding), flattened order-major.
    pub offsets: Vec<u64>,
    pub total_rows: u64,
}

pub struct NgramHashMapping {
    pub cfg: EngramConfig,
    pub tokenizer: CompressedTokenizer,
    pub pad_id: i64,
    pub layers: Vec<LayerHash>,
}

impl NgramHashMapping {
    pub fn new(cfg: EngramConfig, tokenizer: CompressedTokenizer) -> Self {
        let pad_id = cfg.pad_id.map(|p| tokenizer.lookup[p as usize] as i64).unwrap_or(0);
        let max_long = i64::MAX as u64;
        let m_max = max_long / tokenizer.num_new_token as u64;
        let half_bound = (m_max / 2).max(1);
        const PRIME_1: u64 = 10007;

        // The prime search shares one `seen` set across ALL layers and orders,
        // so later layers get later primes. Order matters: layer-major.
        let mut seen = std::collections::HashSet::new();
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
                    while !(is_prime(c) && !seen.contains(&c)) { c += 1; }
                    seen.insert(c);
                    heads.push(c);
                    start = c;
                }
                primes.push(heads);
            }
            let mut offsets = Vec::new();
            let mut acc = 0u64;
            for p in primes.iter().flatten() { offsets.push(acc); acc += p; }
            layers.push(LayerHash { layer_id, multipliers, primes, offsets, total_rows: acc });
        }
        Self { cfg, tokenizer, pad_id, layers }
    }

    pub fn layer(&self, layer_id: u64) -> &LayerHash {
        self.layers.iter().find(|l| l.layer_id == layer_id).expect("unknown layer")
    }

    /// Per-head table indices for one sequence: out[t * H + h], with
    /// H = (max_ngram - 1) * n_head and h order-major, as the demo stacks them.
    /// `compressed` must already be compressed ids.
    pub fn hash_compressed(&self, layer: &LayerHash, compressed: &[i64], out: &mut Vec<i64>) {
        let n_heads = (self.cfg.max_ngram_size - 1) * self.cfg.n_head_per_ngram;
        out.clear();
        out.reserve(compressed.len() * n_heads);
        let m = &layer.multipliers;
        let v41 = self.cfg.semantics == Semantics::V41;
        let mut toks = vec![0i64; self.cfg.max_ngram_size];
        for t in 0..compressed.len() {
            let mut blocked = false;
            for k in 0..self.cfg.max_ngram_size {
                if v41 {
                    blocked |= t < k || compressed[t - k] < 0;
                    toks[k] = if blocked { self.pad_id } else { compressed[t - k] };
                } else {
                    toks[k] = if t >= k { compressed[t - k] } else { self.pad_id };
                }
            }
            let tok = |k: usize| toks[k];
            let mut col = 0usize;
            // NumPy int64 arithmetic wraps; it cannot overflow for in-range ids,
            // but wrapping keeps us identical even for negative ids.
            let mut mix = tok(0).wrapping_mul(m[0]);
            for n in 2..=self.cfg.max_ngram_size {
                mix ^= tok(n - 1).wrapping_mul(m[n - 1]);
                for &p in &layer.primes[n - 2] {
                    let idx = mix.rem_euclid(p as i64); // numpy/torch % on int64 = floor mod
                    out.push(if v41 { idx + layer.offsets[col] as i64 } else { idx });
                    col += 1;
                }
            }
        }
    }

    pub fn hash(&self, layer: &LayerHash, raw_ids: &[i64], out: &mut Vec<i64>) {
        let c: Vec<i64> = raw_ids.iter().map(|&x| self.tokenizer.compress(x)).collect();
        self.hash_compressed(layer, &c, out);
    }
}
