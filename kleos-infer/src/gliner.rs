/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! GLiNER2.5 entity extraction (the `boundary` architecture of
//! [gliner2](https://github.com/fastino-ai/GLiNER2), Apache-2.0), entities
//! only: given a text and a list of labels, the spans of each label with a
//! confidence. Default checkpoint [`GLINER_REPO`] (`Mazino0/kleos-gliner`, kleos's
//! fine-tune of `fastino/gliner2.5-base-v1`, Apache-2.0, a DeBERTa-v3-base encoder); any checkpoint with the same
//! layout loads, e.g. a local fine-tune directory.
//!
//! Port of gliner2 2.1.0's inference path for an entity-only schema
//! (`docs/research/gliner.md`, "Architecture"):
//!
//! * **Input.** The text gets a final `.` unless it ends in `.`, `!` or `?`;
//!   it is split into words by gliner2's regex ([`split_words`]) and each
//!   word is lower-cased and tokenised on its own. The sequence is the schema
//!   `( [P] entities ( [E] label₁ [E] label₂ … ) )`, `[SEP_TEXT]`, then the
//!   words, with no `[CLS]`/`[SEP]`.
//! * **Encoder.** DeBERTa-v3 (disentangled attention, c2p + p2c with shared
//!   keys, 256 log-bucketed relative positions, LayerNorm'd relative
//!   embeddings), on the CPU or Metal at f32/f16/q8 weights with f32
//!   activations. A word is its first subword's state, a label its `[E]`
//!   marker's state.
//! * **Heads** (CPU, f32): boundary states between words (projections, two
//!   windowed attention blocks, a SwiGLU block), start/end/inside marginals
//!   per label, one shared candidate pool per text (top-32 start × top-32
//!   end boundaries, 8 reserved per label, deduplicated to 192), a FiLM
//!   scorer over the pool, and per-label abstention.
//! * **Decoding.** Spans with sigmoid ≥ the floor, then gliner2's `flat`
//!   overlap policy per label (maximum-total-score non-overlapping set,
//!   weighted interval scheduling with its tie rules); an abstaining label
//!   returns nothing.
//!
//! Offsets are byte offsets into the input text.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::budget::{self, ModelSlot};

use candle_core::{DType, Device, Tensor};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::device::device_name;
use crate::error::tok_err;
use crate::gemma::with_pool;
use crate::linear::{Embedding, Linear, Loader, TensorFile};
use crate::{hub, EngineConfig, Error, Result};

/// Hugging Face repo of the default GLiNER2.5 checkpoint: kleos's fine-tune
/// of [`GLINER_BASE_REPO`] for agent-memory entities (person, location,
/// organization, relative; Apache-2.0).
pub const GLINER_REPO: &str = "Mazino0/kleos-gliner";

/// The upstream GLiNER2.5 base checkpoint (Apache-2.0), for open label sets
/// such as domain ontologies.
pub const GLINER_BASE_REPO: &str = "fastino/gliner2.5-base-v1";

/// Masking sentinel of the boundary head (gliner2 `MASK_LOGIT`).
const MASK_LOGIT: f32 = -1.0e4;

/// Default cap on tokens per encoder pass.
pub const GLINER_BATCH_TOKENS: usize = 2048;
/// Bound on `rows × heads × length²` per pass (attention scores and the
/// relative-position index tensors).
const MAX_ATTN_ELEMS: usize = 1 << 25;

#[derive(Debug, Clone, Deserialize)]
struct EncoderConfig {
    hidden_size: usize,
    /// MLP width (only sizes the activation estimate).
    #[serde(default)]
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    layer_norm_eps: f64,
    #[serde(default = "neg1")]
    position_buckets: i64,
    max_position_embeddings: i64,
    #[serde(default = "neg1")]
    max_relative_positions: i64,
    #[serde(default)]
    relative_attention: bool,
    #[serde(default)]
    share_att_key: bool,
    #[serde(default)]
    pos_att_type: Vec<String>,
    #[serde(default)]
    position_biased_input: bool,
    #[serde(default)]
    conv_kernel_size: usize,
    #[serde(default)]
    embedding_size: Option<usize>,
    #[serde(default)]
    type_vocab_size: usize,
    #[serde(default)]
    norm_rel_ebd: String,
    #[serde(default)]
    pad_token_id: Option<u32>,
}

fn neg1() -> i64 {
    -1
}

#[derive(Debug, Clone, Deserialize)]
struct HeadConfig {
    boundary_dim: usize,
    boundary_attention_heads: usize,
    boundary_attention_layers: usize,
    boundary_attention_window: usize,
    boundary_refinement_layers: usize,
    pool_boundary_top_k: usize,
    pool_size: usize,
    min_pool_per_query: usize,
    content_dim: usize,
    candidate_pool: String,
    candidate_attention_layers: usize,
    query_attention_layers: usize,
    enable_span_content: bool,
    content_soft_max_pool: bool,
    use_inside_evidence: bool,
    enable_abstention: bool,
    abstention_threshold: f64,
    pair_temperature: f64,
    overlap_policy: String,
    adaptive_threshold: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct Config {
    architecture: String,
    token_pooling: String,
    encoder_config: EncoderConfig,
    boundary_head: HeadConfig,
}

impl Config {
    fn check(&self) -> Result<()> {
        let e = &self.encoder_config;
        let h = &self.boundary_head;
        let bad = |what: &str| Err(Error::Config(format!("unsupported GLiNER checkpoint: {what}")));
        if self.architecture != "boundary" {
            return bad("architecture is not 'boundary'");
        }
        if self.token_pooling != "first" {
            return bad("token_pooling is not 'first'");
        }
        if !e.relative_attention || !e.share_att_key || e.position_biased_input || e.conv_kernel_size > 0 {
            return bad("encoder is not DeBERTa-v3 (relative attention, shared keys, no conv)");
        }
        if e.type_vocab_size > 0 || e.embedding_size.is_some_and(|s| s != e.hidden_size) {
            return bad("token types or an embedding projection");
        }
        let mut att: Vec<&str> = e.pos_att_type.iter().map(|s| s.as_str()).collect();
        att.sort();
        if att != ["c2p", "p2c"] {
            return bad("pos_att_type is not c2p + p2c");
        }
        if !e.norm_rel_ebd.contains("layer_norm") {
            return bad("relative embeddings without LayerNorm");
        }
        if e.hidden_size % e.num_attention_heads != 0 || h.boundary_dim % h.boundary_attention_heads != 0 {
            return bad("hidden size not a multiple of the head count");
        }
        if h.candidate_pool != "shared"
            || h.candidate_attention_layers > 0
            || h.query_attention_layers > 0
            || !h.enable_span_content
            || h.content_soft_max_pool
            || !h.use_inside_evidence
            || h.adaptive_threshold
        {
            return bad("boundary head settings other than gliner2.5's (shared pool, no candidate/query attention, mean span content, inside evidence)");
        }
        if !matches!(h.overlap_policy.as_str(), "flat" | "disallow") {
            return bad("overlap policy is not 'flat'");
        }
        Ok(())
    }
    fn max_rel(&self) -> i64 {
        let e = &self.encoder_config;
        if e.max_relative_positions < 1 {
            e.max_position_embeddings
        } else {
            e.max_relative_positions
        }
    }
    /// `pos_ebd_size`: the attention span of the relative embeddings.
    fn att_span(&self) -> i64 {
        let e = &self.encoder_config;
        if e.position_buckets > 0 {
            e.position_buckets
        } else {
            self.max_rel()
        }
    }
}

/// DeBERTa's log-bucketed relative position (`make_log_bucket_position`),
/// in f32 as transformers computes it.
pub fn log_bucket_position(rel: i64, bucket_size: i64, max_position: i64) -> i64 {
    if bucket_size <= 0 || max_position <= 0 {
        return rel;
    }
    let mid = bucket_size / 2;
    let abs = if rel < mid && rel > -mid { mid - 1 } else { rel.abs() };
    if abs <= mid {
        return rel;
    }
    let base = (((max_position - 1) as f64 / mid as f64) as f32).ln();
    let log_pos = ((abs as f32 / mid as f32).ln() / base * (mid - 1) as f32).ceil() + mid as f32;
    (log_pos * rel.signum() as f32) as i64
}

// ---------------------------------------------------------------------------
// Words and schema

fn word_regex() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // gliner2 `WhitespaceTokenSplitter`: URLs, e-mails, @handles, word runs
        // joined by `-`/`_`, else one non-space character. A word character is
        // Python's Unicode `\w` (letters, numbers, `_`) plus combining marks
        // and the zero-width (non-)joiners.
        let w = r"[\p{L}\p{N}_\p{M}\x{200C}\x{200D}]";
        regex::Regex::new(&format!(
            r"(?i)(?:https?://[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{{2,}}|@[a-z0-9_]+|{w}+(?:[-_]{w}+)*|\S"
        ))
        .expect("word regex")
    })
}

/// gliner2's word split: `(lower-cased word, byte start, byte end)` per word.
pub fn split_words(text: &str) -> Vec<(String, usize, usize)> {
    word_regex()
        .find_iter(text)
        .map(|m| (m.as_str().to_lowercase(), m.start(), m.end()))
        .collect()
}

/// gliner2's text normalisation: a final `.` unless the text ends in `.`, `!`
/// or `?` (an empty text becomes `.`).
pub fn normalize_text(text: &str) -> String {
    if text.is_empty() {
        return ".".into();
    }
    if text.ends_with(['.', '!', '?']) {
        text.to_string()
    } else {
        format!("{text}.")
    }
}

/// The schema words of an entity-only schema with `labels`.
pub fn entity_schema_words(labels: &[&str]) -> Vec<String> {
    let mut w: Vec<String> = vec!["(".into(), "[P]".into(), "entities".into(), "(".into()];
    for l in labels {
        w.push("[E]".into());
        w.push((*l).to_string());
    }
    w.push(")".into());
    w.push(")".into());
    w
}

/// One encoder input: token ids, the first subword of each text word, and
/// the `[E]` marker of each label.
#[derive(Debug, Clone)]
pub struct GlinerInput {
    /// The normalised text the offsets index.
    pub text: String,
    /// Token ids.
    pub ids: Vec<u32>,
    /// Per text word: `(byte start, byte end)` in `text`.
    pub words: Vec<(usize, usize)>,
    /// Per text word: position of its first subword (clamped to the sequence).
    pub word_pos: Vec<usize>,
    /// Per label: position of its `[E]` marker.
    pub marker_pos: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Weights

/// A dense f32 CPU linear layer with bias (head weights).
struct Lin {
    wt: Tensor,
    b: Tensor,
}

impl Lin {
    fn load(st: &TensorFile, name: &str) -> Result<Self> {
        let w = st.load_f32(&format!("{name}.weight"))?;
        let b = st.load_f32(&format!("{name}.bias"))?;
        Ok(Self { wt: w.t()?.contiguous()?, b })
    }
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.matmul(&self.wt)?.broadcast_add(&self.b)?)
    }
}

struct Ln {
    w: Tensor,
    b: Tensor,
    eps: f32,
}

impl Ln {
    fn load(st: &TensorFile, name: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            w: st.load_f32(&format!("{name}.weight"))?,
            b: st.load_f32(&format!("{name}.bias"))?,
            eps,
        })
    }
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::ops::layer_norm(&x.contiguous()?, &self.w, &self.b, self.eps)?)
    }
}

struct AttnBlock {
    norm: Ln,
    qkv: Lin,
    out: Lin,
}

struct SwiGlu {
    norm: Ln,
    inp: Lin,
    out: Lin,
}

struct Heads {
    bos: Tensor,
    eos: Tensor,
    left: Lin,
    right: Lin,
    merge: Lin,
    norm: Ln,
    attn: Vec<AttnBlock>,
    refine: Vec<SwiGlu>,
    start_b: Lin,
    start_q: Lin,
    end_b: Lin,
    end_q: Lin,
    inside_t: Lin,
    inside_q: Lin,
    pool_start: Lin,
    pool_end: Lin,
    sc_start: Lin,
    sc_end: Lin,
    sc_len: Lin,
    sc_prior: Lin,
    sc_value: Lin,
    sc_value_norm: Ln,
    sc_content: Lin,
    sc_norm: Ln,
    sc_query: Lin,
    sc_film: Lin,
    sc_film0: Lin,
    sc_film3: Lin,
    null: Option<Lin>,
}

struct EncLayer {
    q: Linear,
    qb: Tensor,
    k: Linear,
    kb: Tensor,
    v: Linear,
    vb: Tensor,
    o: Linear,
    ob: Tensor,
    ln1w: Tensor,
    ln1b: Tensor,
    i: Linear,
    ib: Tensor,
    out: Linear,
    outb: Tensor,
    ln2w: Tensor,
    ln2b: Tensor,
    /// `key_proj(rel_emb)` as `[heads, head_dim, 2·span]` (c2p).
    pos_k_t: Tensor,
    /// `query_proj(rel_emb)` as `[heads, head_dim, 2·span]` (p2c).
    pos_q_t: Tensor,
}

struct Weights {
    embed: Embedding,
    emb_lnw: Tensor,
    emb_lnb: Tensor,
    layers: Vec<EncLayer>,
    heads: Heads,
    device: Device,
    tokenizer: Tokenizer,
    piece_ids: Mutex<HashMap<String, Vec<u32>>>,
    pad: u32,
    bytes: usize,
}

impl Weights {
    fn piece(&self, p: &str) -> Result<Vec<u32>> {
        if let Some(v) = self.piece_ids.lock().unwrap().get(p) {
            return Ok(v.clone());
        }
        let ids = self.tokenizer.encode(p, false).map_err(tok_err)?.get_ids().to_vec();
        let mut g = self.piece_ids.lock().unwrap();
        if g.len() < 4096 {
            g.insert(p.to_string(), ids.clone());
        }
        Ok(ids)
    }
}

/// A span found for one label.
#[derive(Debug, Clone, PartialEq)]
pub struct GlinerSpan {
    /// Index into the labels asked for.
    pub label: usize,
    /// Byte offsets into the input text.
    pub start: usize,
    /// Exclusive byte end.
    pub end: usize,
    /// The span's text (trimmed).
    pub text: String,
    /// Sigmoid confidence.
    pub score: f32,
}

/// Every intermediate of one text, for parity tests.
#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct GlinerStages {
    pub ids: Vec<u32>,
    pub word_pos: Vec<usize>,
    pub marker_pos: Vec<usize>,
    /// `[tokens][hidden]`.
    pub hidden: Vec<Vec<f32>>,
    /// `[words + 1][boundary_dim]`.
    pub boundary: Vec<Vec<f32>>,
    /// `[labels][words + 1]`.
    pub start_logits: Vec<Vec<f32>>,
    pub end_logits: Vec<Vec<f32>>,
    pub inside_prefix: Vec<Vec<f32>>,
    pub inside_mean: Vec<f32>,
    /// Valid pool candidates in pool order: `(start, end)` boundaries.
    pub pool: Vec<(usize, usize)>,
    pub pool_compat: Vec<f32>,
    /// `[candidate][label]`.
    pub pair_logits: Vec<Vec<f32>>,
    pub null_logits: Vec<f32>,
    pub spans: Vec<GlinerSpan>,
}

/// GLiNER2.5 entity extraction. Weights load lazily on the first call.
pub struct Gliner {
    repo: String,
    cfg: Config,
    engine: EngineConfig,
    /// Weights, device gate and budget bookkeeping.
    weights: ModelSlot<Weights>,
}

impl Gliner {
    /// `$KLEOS_GLINER_MODEL` (a repo id or a local checkpoint directory) when
    /// set, else [`GLINER_REPO`].
    pub fn new(engine: EngineConfig) -> Result<Self> {
        let repo = std::env::var("KLEOS_GLINER_MODEL")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| GLINER_REPO.to_string());
        Self::with_repo(&repo, engine)
    }

    /// A checkpoint by repo id or local directory. Fetches missing files into
    /// the HF cache and reads `config.json` now; weights load on first use.
    pub fn with_repo(repo: &str, engine: EngineConfig) -> Result<Self> {
        hub::ensure_hf_files(repo, &["config.json", "tokenizer.json", "model.safetensors"], &[])?;
        let cfg: Config = serde_json::from_slice(&std::fs::read(hub::resolve_hf_file(repo, "config.json")?)?)?;
        cfg.check()?;
        Ok(Self {
            repo: repo.to_string(),
            cfg,
            engine,
            weights: ModelSlot::new(format!("extractor:{repo}")),
        })
    }

    /// The repo id or directory the model was built from.
    pub fn model_id(&self) -> &str {
        &self.repo
    }

    /// The engine settings this model was built with.
    pub fn engine(&self) -> &EngineConfig {
        &self.engine
    }

    /// `(device, bytes of weights)` once loaded.
    pub fn loaded_info(&self) -> Option<(&'static str, usize)> {
        self.weights.peek().map(|w| (device_name(&w.device), w.bytes))
    }

    /// Whether the weights are in memory.
    pub fn is_loaded(&self) -> bool {
        self.weights.is_loaded()
    }

    /// Drop the weights (they reload on the next call).
    pub fn unload(&self) {
        self.weights.unload();
    }

    fn weights(&self) -> Result<Arc<Weights>> {
        self.weights.get_or_load(|| with_pool(|| self.load()), |w| w.bytes)
    }

    /// Activation bytes of one encoder pass over `n` rows of padded length
    /// `l` (DeBERTa's disentangled attention adds c2p and p2c maps).
    fn act_bytes(&self, n: usize, l: usize, dt: DType) -> usize {
        let e = &self.cfg.encoder_config;
        let inter = if e.intermediate_size > 0 { e.intermediate_size } else { 4 * e.hidden_size };
        budget::encoder_activation_bytes(n, l, e.hidden_size, inter, e.num_attention_heads, dt.size_in_bytes(), 4)
    }

    fn load(&self) -> Result<Weights> {
        let e = &self.cfg.encoder_config;
        let device = self.engine.device.device()?;
        let mut tokenizer =
            Tokenizer::from_file(hub::resolve_hf_file(&self.repo, "tokenizer.json")?).map_err(tok_err)?;
        tokenizer.with_truncation(None).map_err(tok_err)?;
        tokenizer.with_padding(None);
        for t in ["[P]", "[E]", "[SEP_TEXT]"] {
            if tokenizer.token_to_id(t).is_none() {
                return Err(Error::Config(format!("tokenizer has no {t} token")));
            }
        }
        let pad = e.pad_token_id.unwrap_or(0);
        let st = TensorFile::open(&hub::resolve_hf_file(&self.repo, "model.safetensors")?)?;
        let mut ld = Loader {
            st: &st,
            device: &device,
            precision: self.engine.precision,
            bytes: 0,
        };
        let eps = e.layer_norm_eps as f32;
        let p = "encoder.";
        let embed = ld.embedding(&format!("{p}embeddings.word_embeddings.weight"))?;
        let emb_lnw = ld.f32(&format!("{p}embeddings.LayerNorm.weight"))?;
        let emb_lnb = ld.f32(&format!("{p}embeddings.LayerNorm.bias"))?;
        // relative embeddings, LayerNorm'd once (`get_rel_embedding`)
        let span2 = (2 * self.cfg.att_span()) as usize;
        let rel = ld.f32(&format!("{p}encoder.rel_embeddings.weight"))?;
        let rel = rel.narrow(0, 0, span2.min(rel.dim(0)?))?;
        let rel = candle_nn::ops::layer_norm(
            &rel.contiguous()?,
            &ld.f32(&format!("{p}encoder.LayerNorm.weight"))?,
            &ld.f32(&format!("{p}encoder.LayerNorm.bias"))?,
            eps,
        )?;
        let (h, hd) = (e.num_attention_heads, e.hidden_size / e.num_attention_heads);
        let mut layers = Vec::with_capacity(e.num_hidden_layers);
        for i in 0..e.num_hidden_layers {
            let n = |s: &str| format!("{p}encoder.layer.{i}.{s}");
            let q = ld.linear(&n("attention.self.query_proj.weight"))?;
            let qb = ld.f32(&n("attention.self.query_proj.bias"))?;
            let k = ld.linear(&n("attention.self.key_proj.weight"))?;
            let kb = ld.f32(&n("attention.self.key_proj.bias"))?;
            // [2span, hidden] -> [h, hd, 2span]
            let pos_t = |lin: &Linear, b: &Tensor| -> Result<Tensor> {
                let y = lin.forward(&rel)?.broadcast_add(b)?;
                Ok(y.reshape((span2, h, hd))?.permute((1, 2, 0))?.contiguous()?)
            };
            let pos_k_t = pos_t(&k, &kb)?;
            let pos_q_t = pos_t(&q, &qb)?;
            ld.bytes += 2 * pos_k_t.elem_count() * 4;
            layers.push(EncLayer {
                v: ld.linear(&n("attention.self.value_proj.weight"))?,
                vb: ld.f32(&n("attention.self.value_proj.bias"))?,
                o: ld.linear(&n("attention.output.dense.weight"))?,
                ob: ld.f32(&n("attention.output.dense.bias"))?,
                ln1w: ld.f32(&n("attention.output.LayerNorm.weight"))?,
                ln1b: ld.f32(&n("attention.output.LayerNorm.bias"))?,
                i: ld.linear(&n("intermediate.dense.weight"))?,
                ib: ld.f32(&n("intermediate.dense.bias"))?,
                out: ld.linear(&n("output.dense.weight"))?,
                outb: ld.f32(&n("output.dense.bias"))?,
                ln2w: ld.f32(&n("output.LayerNorm.weight"))?,
                ln2b: ld.f32(&n("output.LayerNorm.bias"))?,
                q,
                qb,
                k,
                kb,
                pos_k_t,
                pos_q_t,
            });
        }
        let heads = self.load_heads(&st)?;
        let head_bytes = 4 * self.head_params(&heads);
        let bytes = ld.bytes + head_bytes;
        Ok(Weights {
            embed,
            emb_lnw,
            emb_lnb,
            layers,
            heads,
            device,
            tokenizer,
            piece_ids: Mutex::new(HashMap::new()),
            pad,
            bytes,
        })
    }

    fn head_params(&self, hd: &Heads) -> usize {
        let lins: Vec<&Lin> = vec![
            &hd.left, &hd.right, &hd.merge, &hd.start_b, &hd.start_q, &hd.end_b, &hd.end_q, &hd.inside_t,
            &hd.inside_q, &hd.pool_start, &hd.pool_end, &hd.sc_start, &hd.sc_end, &hd.sc_len, &hd.sc_prior,
            &hd.sc_value, &hd.sc_content, &hd.sc_query, &hd.sc_film, &hd.sc_film0, &hd.sc_film3,
        ];
        let mut n: usize = lins.iter().map(|l| l.wt.elem_count() + l.b.elem_count()).sum();
        for a in &hd.attn {
            n += a.qkv.wt.elem_count() + a.out.wt.elem_count();
        }
        for r in &hd.refine {
            n += r.inp.wt.elem_count() + r.out.wt.elem_count();
        }
        n + hd.bos.elem_count() * 2
    }

    fn load_heads(&self, st: &TensorFile) -> Result<Heads> {
        let hc = &self.cfg.boundary_head;
        let b = "boundary_head.";
        let l = |s: &str| Lin::load(st, &format!("{b}{s}"));
        let n = |s: &str| Ln::load(st, &format!("{b}{s}"), 1e-5);
        let mut attn = vec![];
        for i in 0..hc.boundary_attention_layers {
            attn.push(AttnBlock {
                norm: n(&format!("boundary_encoder.attention_blocks.{i}.norm"))?,
                qkv: l(&format!("boundary_encoder.attention_blocks.{i}.qkv_projection"))?,
                out: l(&format!("boundary_encoder.attention_blocks.{i}.output_projection"))?,
            });
        }
        let mut refine = vec![];
        for i in 0..hc.boundary_refinement_layers {
            refine.push(SwiGlu {
                norm: n(&format!("boundary_encoder.refinement_blocks.{i}.norm"))?,
                inp: l(&format!("boundary_encoder.refinement_blocks.{i}.input_projection"))?,
                out: l(&format!("boundary_encoder.refinement_blocks.{i}.output_projection"))?,
            });
        }
        Ok(Heads {
            bos: st.load_f32(&format!("{b}boundary_encoder.bos_state"))?,
            eos: st.load_f32(&format!("{b}boundary_encoder.eos_state"))?,
            left: l("boundary_encoder.left_projection")?,
            right: l("boundary_encoder.right_projection")?,
            merge: l("boundary_encoder.output_projection")?,
            norm: n("boundary_encoder.layer_norm")?,
            attn,
            refine,
            start_b: l("boundary_query_head.start_boundary_projection")?,
            start_q: l("boundary_query_head.start_query_projection")?,
            end_b: l("boundary_query_head.end_boundary_projection")?,
            end_q: l("boundary_query_head.end_query_projection")?,
            inside_t: l("boundary_query_head.inside_text_projection")?,
            inside_q: l("boundary_query_head.inside_query_projection")?,
            pool_start: l("shared_pool_builder.start_projection")?,
            pool_end: l("shared_pool_builder.end_projection")?,
            sc_start: l("shared_pool_scorer.start_projection")?,
            sc_end: l("shared_pool_scorer.end_projection")?,
            sc_len: l("shared_pool_scorer.length_projection")?,
            sc_prior: l("shared_pool_scorer.prior_projection")?,
            sc_value: l("shared_pool_scorer.content_pooler.value_projection")?,
            sc_value_norm: n("shared_pool_scorer.content_pooler.layer_norm")?,
            sc_content: l("shared_pool_scorer.content_projection")?,
            sc_norm: n("shared_pool_scorer.candidate_norm")?,
            sc_query: l("shared_pool_scorer.query_projection")?,
            sc_film: l("shared_pool_scorer.film")?,
            sc_film0: l("shared_pool_scorer.film_output.0")?,
            sc_film3: l("shared_pool_scorer.film_output.3")?,
            null: if hc.enable_abstention {
                Some(l("null_projection")?)
            } else {
                None
            },
        })
    }

    /// Build the encoder input of `text` for `labels`.
    fn input(&self, w: &Weights, text: &str, labels: &[&str]) -> Result<GlinerInput> {
        let norm = normalize_text(text);
        let mut words = split_words(&norm);
        let mut ids: Vec<u32> = vec![];
        let mut marker_pos = vec![];
        for (k, piece) in entity_schema_words(labels).iter().enumerate() {
            // `[E]` markers sit at schema indices 4, 6, 8, ...
            if k >= 4 && k % 2 == 0 && piece == "[E]" {
                marker_pos.push(ids.len());
            }
            ids.extend(w.piece(piece)?);
        }
        ids.extend(w.piece("[SEP_TEXT]")?);
        let budget = self.engine.max_seq_len.max(ids.len() + 1);
        let mut word_pos = Vec::with_capacity(words.len());
        let mut kept = 0;
        for (wd, _, _) in &words {
            let sub = w.piece(wd)?;
            if ids.len() + sub.len() > budget && kept > 0 {
                break;
            }
            word_pos.push(ids.len());
            ids.extend(sub);
            kept += 1;
        }
        words.truncate(kept);
        let last = ids.len().saturating_sub(1);
        for p in word_pos.iter_mut() {
            *p = (*p).min(last);
        }
        Ok(GlinerInput {
            text: norm,
            ids,
            words: words.iter().map(|(_, s, e)| (*s, *e)).collect(),
            word_pos,
            marker_pos,
        })
    }

    /// Entity spans of each text for `labels`: every span whose confidence is
    /// at least `floor` after gliner2's `flat` overlap resolution, per label,
    /// in descending confidence. Offsets index the text as given.
    pub fn extract(&self, texts: &[&str], labels: &[&str], floor: f32) -> Result<Vec<Vec<GlinerSpan>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        if labels.is_empty() {
            return Ok(vec![vec![]; texts.len()]);
        }
        with_pool(|| self.extract_inner(texts, labels, floor, None))
    }

    /// Like [`Gliner::extract`] for texts of any length: a text longer than
    /// `window` words is read in windows of `window` words that overlap by
    /// `overlap` (gliner2's long-text defaults are 384 and 64); the windows'
    /// spans are mapped back to the text and merged per label with the same
    /// `flat` resolution, so a span seen in two windows is kept once, with its
    /// higher score. Offsets index the text as given.
    pub fn extract_windows(
        &self,
        texts: &[&str],
        labels: &[&str],
        floor: f32,
        window: usize,
        overlap: usize,
    ) -> Result<Vec<Vec<GlinerSpan>>> {
        if window == 0 || overlap >= window {
            return Err(Error::Config("window must be > 0 and larger than overlap".into()));
        }
        // (text index, byte start of the window, window text)
        let mut chunks: Vec<(usize, usize, &str)> = vec![];
        for (ti, t) in texts.iter().enumerate() {
            let words = split_words(t);
            if words.len() <= window {
                chunks.push((ti, 0, t));
                continue;
            }
            let mut sw = 0;
            loop {
                let ew = (sw + window).min(words.len());
                let (a, b) = (words[sw].1, words[ew - 1].2);
                chunks.push((ti, a, &t[a..b]));
                if ew == words.len() {
                    break;
                }
                sw += window - overlap;
            }
        }
        let inputs: Vec<&str> = chunks.iter().map(|c| c.2).collect();
        let found = self.extract(&inputs, labels, floor)?;
        let mut per_text: Vec<Vec<GlinerSpan>> = vec![vec![]; texts.len()];
        let mut windows = vec![0usize; texts.len()];
        for ((ti, off, _), spans) in chunks.iter().zip(found) {
            windows[*ti] += 1;
            per_text[*ti].extend(spans.into_iter().map(|mut s| {
                s.start += off;
                s.end += off;
                s
            }));
        }
        Ok(per_text
            .into_iter()
            .zip(windows)
            .map(|(spans, nw)| {
                if nw <= 1 {
                    return spans;
                }
                let mut out = vec![];
                for q in 0..labels.len() {
                    let mine: Vec<&GlinerSpan> = spans.iter().filter(|s| s.label == q).collect();
                    let items: Vec<(f32, usize, usize)> = mine.iter().map(|s| (s.score, s.start, s.end)).collect();
                    for (p, a, b) in resolve_flat(&items) {
                        if let Some(s) = mine.iter().find(|s| s.start == a && s.end == b && s.score == p) {
                            out.push((*s).clone());
                        }
                    }
                }
                out
            })
            .collect())
    }

    /// Every intermediate of one text (parity tests).
    #[doc(hidden)]
    pub fn debug_stages(&self, text: &str, labels: &[&str], floor: f32) -> Result<GlinerStages> {
        let mut st = GlinerStages::default();
        with_pool(|| self.extract_inner(&[text], labels, floor, Some(&mut st)))?;
        Ok(st)
    }

    fn extract_inner(
        &self,
        texts: &[&str],
        labels: &[&str],
        floor: f32,
        mut stages: Option<&mut GlinerStages>,
    ) -> Result<Vec<Vec<GlinerSpan>>> {
        let w = self.weights()?;
        let inputs = texts.iter().map(|t| self.input(&w, t, labels)).collect::<Result<Vec<_>>>()?;
        let mut order: Vec<usize> = (0..inputs.len()).collect();
        order.sort_by_key(|&i| inputs[i].ids.len());
        let budget = self.batch_tokens();
        let heads = self.cfg.encoder_config.num_attention_heads;
        let mut out: Vec<Vec<GlinerSpan>> = vec![vec![]; inputs.len()];
        let dt = DType::F32; // conservative: f32 attention maps and LayerNorms
        let rows_cap = |l: usize| budget::rows_for(l, budget, |l| self.act_bytes(1, l, dt));
        let mut i = 0;
        while i < order.len() {
            let mut j = i + 1;
            while j < order.len() {
                let l = inputs[order[j]].ids.len();
                let n = j - i + 1;
                if n * l > budget || n * heads * l * l > MAX_ATTN_ELEMS || n > rows_cap(l) {
                    break;
                }
                j += 1;
            }
            let rows: Vec<&GlinerInput> = order[i..j].iter().map(|&k| &inputs[k]).collect();
            let l = rows.iter().map(|r| r.ids.len()).max().unwrap_or(1);
            // the device, at this thread's priority, with the pass's activations reserved
            let _run = self.weights.run(self.act_bytes(rows.len(), l, dt));
            let hidden = self.encode(&w, &rows)?;
            for (r, &k) in order[i..j].iter().enumerate() {
                let inp = &inputs[k];
                let hs = hidden.get(r)?.narrow(0, 0, inp.ids.len())?;
                let st = stages.as_deref_mut();
                let spans = self.head(&w, inp, &hs, labels.len(), floor, st)?;
                // offsets index the normalised text, which only appends a '.'
                let orig = texts[k];
                out[k] = spans
                    .into_iter()
                    .filter(|s| s.start < orig.len())
                    .map(|mut s| {
                        s.end = s.end.min(orig.len());
                        s.text = orig[s.start..s.end].trim().to_string();
                        s
                    })
                    .filter(|s| !s.text.is_empty())
                    .collect();
            }
            i = j;
        }
        w.device.synchronize()?;
        Ok(out)
    }

    /// Tokens per encoder pass: the engine's budget, capped at
    /// [`GLINER_BATCH_TOKENS`] (`KLEOS_GLINER_BATCH_TOKENS` overrides).
    fn batch_tokens(&self) -> usize {
        let cap = std::env::var("KLEOS_GLINER_BATCH_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(GLINER_BATCH_TOKENS);
        self.engine.max_batch_tokens.min(cap).max(1)
    }

    /// DeBERTa-v3 over a padded batch; last hidden states `[b, l, hidden]`
    /// on the CPU, f32.
    fn encode(&self, w: &Weights, rows: &[&GlinerInput]) -> Result<Tensor> {
        let e = &self.cfg.encoder_config;
        let dev = &w.device;
        let b = rows.len();
        let l = rows.iter().map(|r| r.ids.len()).max().unwrap_or(1);
        let hidden = e.hidden_size;
        let (h, hd) = (e.num_attention_heads, hidden / e.num_attention_heads);
        let eps = e.layer_norm_eps as f32;
        let mut ids = vec![w.pad; b * l];
        let mut keep = vec![0f32; b * l];
        let mut key_mask = vec![-1.0e9f32; b * l];
        for (r, row) in rows.iter().enumerate() {
            let n = row.ids.len();
            ids[r * l..r * l + n].copy_from_slice(&row.ids);
            for t in 0..n {
                keep[r * l + t] = 1.0;
                key_mask[r * l + t] = 0.0;
            }
        }
        // relative-position gather indices: c2p[i][j] = clamp(bucket(i - j) + span),
        // and its transpose for p2c (r_pos = relative_pos, -bucket(j - i) = bucket(i - j))
        let span = self.cfg.att_span();
        let (bs, mp) = (e.position_buckets, self.cfg.max_rel());
        let mut c2p = vec![0u32; l * l];
        let mut p2c_t = vec![0u32; l * l];
        for qi in 0..l {
            for kj in 0..l {
                let rel = log_bucket_position(qi as i64 - kj as i64, bs, mp);
                let pos = (rel + span).clamp(0, 2 * span - 1) as u32;
                c2p[qi * l + kj] = pos;
                p2c_t[kj * l + qi] = pos;
            }
        }
        // heads-first layout [h, b, l, ·]: the relative-position products are
        // then one batched matmul per head over all rows ([h, b·l, hd] × [h, hd, 2span])
        let c2p = Tensor::from_vec(c2p, (1, 1, l, l), dev)?.broadcast_as((h, b, l, l))?.contiguous()?;
        let p2c_t = Tensor::from_vec(p2c_t, (1, 1, l, l), dev)?.broadcast_as((h, b, l, l))?.contiguous()?;
        let ids_t = Tensor::from_vec(ids, (b, l), dev)?;
        let keep_t = Tensor::from_vec(keep, (b * l, 1), dev)?;
        let mask_t = Tensor::from_vec(key_mask, (1, b, 1, l), dev)?;
        let ln = |x: &Tensor, wt: &Tensor, bias: &Tensor| -> Result<Tensor> {
            Ok(candle_nn::ops::layer_norm(&x.contiguous()?, wt, bias, eps)?)
        };
        let lin = |m: &Linear, bias: &Tensor, x: &Tensor| -> Result<Tensor> { Ok(m.forward(x)?.broadcast_add(bias)?) };
        let mut x = w.embed.lookup(&ids_t)?.reshape((b * l, hidden))?;
        x = ln(&x, &w.emb_lnw, &w.emb_lnb)?.broadcast_mul(&keep_t)?;
        let scale = ((hd as f32) * 3.0).sqrt() as f64;
        let heads = |t: Tensor| -> Result<Tensor> { Ok(t.reshape((b, l, h, hd))?.permute((2, 0, 1, 3))?.contiguous()?) };
        let rel_att = |t: &Tensor, pos: &Tensor, idx: &Tensor| -> Result<Tensor> {
            let full = t.reshape((h, b * l, hd))?.matmul(pos)?.reshape((h, b, l, pos.dim(2)?))?;
            Ok(full.gather(idx, 3)?)
        };
        for layer in &w.layers {
            let q = heads(lin(&layer.q, &layer.qb, &x)?)?;
            let k = heads(lin(&layer.k, &layer.kb, &x)?)?;
            let v = heads(lin(&layer.v, &layer.vb, &x)?)?;
            let mut att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / scale)?;
            let c2p_att = rel_att(&q, &layer.pos_k_t, &c2p)?;
            let p2c_att = rel_att(&k, &layer.pos_q_t, &p2c_t)?.transpose(2, 3)?;
            att = (att + ((c2p_att + p2c_att)? / scale)?)?.broadcast_add(&mask_t)?;
            let probs = candle_nn::ops::softmax_last_dim(&att.contiguous()?)?;
            let ctx = probs.matmul(&v)?.permute((1, 2, 0, 3))?.reshape((b * l, hidden))?;
            let a = ln(&(lin(&layer.o, &layer.ob, &ctx)? + &x)?, &layer.ln1w, &layer.ln1b)?;
            let inter = lin(&layer.i, &layer.ib, &a)?.gelu_erf()?;
            x = ln(&(lin(&layer.out, &layer.outb, &inter)? + &a)?, &layer.ln2w, &layer.ln2b)?;
        }
        Ok(x.reshape((b, l, hidden))?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?)
    }

    /// The boundary head and decoding for one text, from its encoder states
    /// `hs` (`[tokens, hidden]`, CPU f32).
    fn head(
        &self,
        w: &Weights,
        inp: &GlinerInput,
        hs: &Tensor,
        nq: usize,
        floor: f32,
        stages: Option<&mut GlinerStages>,
    ) -> Result<Vec<GlinerSpan>> {
        let hc = &self.cfg.boundary_head;
        let hd = &w.heads;
        let cpu = Device::Cpu;
        let nl = inp.word_pos.len();
        let nb = nl + 1;
        let idx = |v: &[usize]| Tensor::from_vec(v.iter().map(|&p| p as u32).collect::<Vec<_>>(), v.len(), &cpu);
        let text = if nl > 0 {
            hs.index_select(&idx(&inp.word_pos)?, 0)?
        } else {
            Tensor::zeros((0, hs.dim(1)?), DType::F32, &cpu)?
        };
        let query = hs.index_select(&idx(&inp.marker_pos)?, 0)?;

        // boundary encoder: left(i) = word i-1 (BOS at 0), right(i) = word i (EOS at n)
        let left = Tensor::cat(&[&hd.bos.unsqueeze(0)?, &text], 0)?;
        let right = Tensor::cat(&[&text, &hd.eos.unsqueeze(0)?], 0)?;
        let mut bs = hd.merge.fwd(&Tensor::cat(&[&hd.left.fwd(&left)?, &hd.right.fwd(&right)?], 1)?)?;
        bs = hd.norm.fwd(&bs)?;
        let bd = hc.boundary_dim;
        let ah = hc.boundary_attention_heads;
        let ad = bd / ah;
        let window = hc.boundary_attention_window;
        if !hd.attn.is_empty() {
            let mut local = vec![f32::NEG_INFINITY; nb * nb];
            for a in 0..nb {
                for c in 0..nb {
                    if window == 0 || a.abs_diff(c) <= window {
                        local[a * nb + c] = 0.0;
                    }
                }
            }
            let local = Tensor::from_vec(local, (1, nb, nb), &cpu)?;
            for blk in &hd.attn {
                let qkv = blk.qkv.fwd(&blk.norm.fwd(&bs)?)?.reshape((nb, 3, ah, ad))?.permute((1, 2, 0, 3))?;
                let q = qkv.get(0)?.contiguous()?;
                let k = qkv.get(1)?.contiguous()?;
                let v = qkv.get(2)?.contiguous()?;
                let s = (q.matmul(&k.t()?)? / (ad as f64).sqrt())?.broadcast_add(&local)?;
                let p = candle_nn::ops::softmax_last_dim(&s)?;
                let o = p.matmul(&v)?.transpose(0, 1)?.reshape((nb, bd))?;
                bs = (bs + blk.out.fwd(&o)?)?;
            }
        }
        for blk in &hd.refine {
            let y = blk.inp.fwd(&blk.norm.fwd(&bs)?)?;
            let half = y.dim(1)? / 2;
            let upd = (y.narrow(1, 0, half)? * candle_nn::ops::silu(&y.narrow(1, half, half)?)?)?;
            bs = (bs + blk.out.fwd(&upd)?)?;
        }

        // marginals [Q, N]
        let inv = 1.0 / (bd as f64).sqrt();
        let dot = |a: &Tensor, b: &Tensor| -> Result<Vec<Vec<f32>>> { Ok((a.matmul(&b.t()?)? * inv)?.to_vec2::<f32>()?) };
        let start_l = dot(&hd.start_q.fwd(&query)?, &hd.start_b.fwd(&bs)?)?;
        let end_l = dot(&hd.end_q.fwd(&query)?, &hd.end_b.fwd(&bs)?)?;
        let inside = if nl > 0 {
            dot(&hd.inside_q.fwd(&query)?, &hd.inside_t.fwd(&text)?)?
        } else {
            vec![vec![]; nq]
        };
        let mut prefix = vec![vec![0f32; nb]; nq];
        let mut imean = vec![0f32; nq];
        for q in 0..nq {
            let sum: f32 = inside[q].iter().sum();
            let mean = sum / (nl.max(1) as f32);
            imean[q] = mean;
            let mut acc = 0f32;
            for t in 0..nl {
                acc += inside[q][t] - mean;
                prefix[q][t + 1] = acc;
            }
        }

        // shared pool
        let union = |m: &Vec<Vec<f32>>| -> Vec<f32> {
            (0..nb).map(|n| (0..nq).map(|q| m[q][n]).fold(MASK_LOGIT, f32::max)).collect()
        };
        let u_start = union(&start_l);
        let u_end = union(&end_l);
        let topk = |v: &[f32]| -> Vec<usize> {
            let mut o: Vec<usize> = (0..v.len()).collect();
            o.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
            o.truncate(hc.pool_boundary_top_k.min(v.len()));
            o
        };
        let starts = topk(&u_start);
        let ends = topk(&u_end);
        let p_start = hd.pool_start.fwd(&bs)?.to_vec2::<f32>()?;
        let p_end = hd.pool_end.fwd(&bs)?.to_vec2::<f32>()?;
        let compat = |s: usize, e: usize| -> f32 {
            let d: f32 = p_start[s].iter().zip(&p_end[e]).map(|(a, b)| a * b).sum();
            d / (bd as f32).sqrt()
        };
        let (ks, ke) = (starts.len(), ends.len());
        let mut pair_s = Vec::with_capacity(ks * ke);
        let mut pair_e = Vec::with_capacity(ks * ke);
        let mut pair_valid = Vec::with_capacity(ks * ke);
        let mut pair_compat = Vec::with_capacity(ks * ke);
        for &s in &starts {
            for &e in &ends {
                pair_s.push(s);
                pair_e.push(e);
                pair_valid.push(e > s);
                pair_compat.push(compat(s, e));
            }
        }
        let np = pair_s.len();
        let quota = hc.min_pool_per_query.min(np);
        let mut keys: Vec<usize> = vec![];
        let mut scores: Vec<f32> = vec![];
        let mut valid: Vec<bool> = vec![];
        if quota > 0 {
            for q in 0..nq {
                let pq: Vec<f32> = (0..np)
                    .map(|i| {
                        if pair_valid[i] {
                            start_l[q][pair_s[i]] + end_l[q][pair_e[i]] + pair_compat[i]
                        } else {
                            MASK_LOGIT
                        }
                    })
                    .collect();
                let mut o: Vec<usize> = (0..np).collect();
                o.sort_by(|&a, &b| pq[b].partial_cmp(&pq[a]).unwrap_or(std::cmp::Ordering::Equal));
                for (r, &i) in o.iter().take(quota).enumerate() {
                    keys.push(pair_s[i] * nb + pair_e[i]);
                    valid.push(pair_valid[i]);
                    scores.push(-MASK_LOGIT * 0.5 + (quota - r) as f32);
                }
            }
        }
        for i in 0..np {
            keys.push(pair_s[i] * nb + pair_e[i]);
            valid.push(pair_valid[i]);
            scores.push(pair_compat[i] + u_start[pair_s[i]] + u_end[pair_e[i]]);
        }
        let (sel_keys, sel_valid) = dedupe_pool(&keys, &scores, &valid, hc.pool_size, nb);
        let cands: Vec<(usize, usize)> = sel_keys
            .iter()
            .zip(&sel_valid)
            .filter(|(_, v)| **v)
            .map(|(k, _)| (k / nb, k % nb))
            .collect();
        let nc = cands.len();

        // pooled scorer over the valid candidates (invalid ones score MASK_LOGIT and never pass)
        let mut pair_logits = vec![vec![MASK_LOGIT; nq]; nc];
        let mut c_compat = vec![0f32; nc];
        if nc > 0 {
            let cs: Vec<usize> = cands.iter().map(|c| c.0).collect();
            let ce: Vec<usize> = cands.iter().map(|c| c.1).collect();
            for (c, (s, e)) in cands.iter().enumerate() {
                c_compat[c] = compat(*s, *e);
            }
            let start_rep = hd.sc_start.fwd(&bs)?.index_select(&idx(&cs)?, 0)?;
            let end_rep = hd.sc_end.fwd(&bs)?.index_select(&idx(&ce)?, 0)?;
            let tl = nl.max(1) as f32;
            let mut lf = Vec::with_capacity(nc * 3);
            for (s, e) in &cands {
                let len = (e - s).max(1) as f32;
                lf.extend([len.ln_1p(), len / tl, 1.0 / len.sqrt()]);
            }
            let lf = Tensor::from_vec(lf, (nc, 3), &cpu)?;
            let prior = Tensor::from_vec(c_compat.clone(), (nc, 1), &cpu)?;
            let mut cand = (((start_rep + end_rep)? + hd.sc_len.fwd(&lf)?)? + hd.sc_prior.fwd(&prior)?)?;
            // mean span content from word prefixes
            let cd = hc.content_dim;
            let vals = if nl > 0 {
                hd.sc_value.fwd(&text)?.to_vec2::<f32>()?
            } else {
                vec![]
            };
            let mut pre = vec![vec![0f32; cd]; nb];
            for t in 0..nl {
                for d in 0..cd {
                    pre[t + 1][d] = pre[t][d] + vals[t][d];
                }
            }
            let mut content = Vec::with_capacity(nc * cd);
            for (s, e) in &cands {
                let len = (e - s).max(1) as f32;
                for d in 0..cd {
                    content.push((pre[*e][d] - pre[*s][d]) / len);
                }
            }
            let content = hd.sc_value_norm.fwd(&Tensor::from_vec(content, (nc, cd), &cpu)?)?;
            cand = (cand + hd.sc_content.fwd(&content)?)?;
            cand = hd.sc_norm.fwd(&cand)?;
            let qp = hd.sc_query.fwd(&query)?;
            let pd = qp.dim(1)?;
            let base = (cand.matmul(&qp.t()?)? / (pd as f64).sqrt())?.to_vec2::<f32>()?;
            let film = hd.sc_film.fwd(&qp)?;
            let gamma = film.narrow(1, 0, pd)?;
            let beta = film.narrow(1, pd, pd)?;
            // conditioned[c, q] = cand[c] * (1 + gamma[q]) + beta[q]
            let cond = cand
                .unsqueeze(1)?
                .broadcast_mul(&(gamma + 1.0)?.unsqueeze(0)?)?
                .broadcast_add(&beta.unsqueeze(0)?)?
                .reshape((nc * nq, pd))?;
            let film_out = hd.sc_film3.fwd(&hd.sc_film0.fwd(&cond)?.gelu_erf()?)?.reshape((nc, nq))?.to_vec2::<f32>()?;
            for c in 0..nc {
                let (s, e) = cands[c];
                let width = (e - s).max(1) as f32;
                for q in 0..nq {
                    let mut z = base[c][q] + film_out[c][q];
                    z += start_l[q][s];
                    z += end_l[q][e];
                    let interval = prefix[q][e] - prefix[q][s] + imean[q] * (e - s) as f32;
                    z += interval / width.sqrt();
                    pair_logits[c][q] = z;
                }
            }
        }

        let null = match &hd.null {
            Some(n) => n.fwd(&query)?.flatten_all()?.to_vec1::<f32>()?,
            None => vec![],
        };

        // decode: per label, threshold then flat overlap resolution
        let temp = hc.pair_temperature as f32;
        let mut spans = vec![];
        for q in 0..nq {
            if let Some(z) = null.get(q) {
                if sigmoid(*z) as f64 > hc.abstention_threshold {
                    continue;
                }
            }
            let scored: Vec<(f32, usize, usize)> = (0..nc)
                .filter_map(|c| {
                    let p = sigmoid(pair_logits[c][q] / temp);
                    (p >= floor).then_some((p, cands[c].0, cands[c].1))
                })
                .collect();
            for (p, s, e) in resolve_flat(&scored) {
                if !(s < e && e <= nl) {
                    continue;
                }
                let (cs, ce) = (inp.words[s].0, inp.words[e - 1].1);
                let surface = inp.text[cs..ce].trim();
                if surface.is_empty() {
                    continue;
                }
                spans.push(GlinerSpan {
                    label: q,
                    start: cs,
                    end: ce,
                    text: surface.to_string(),
                    score: p,
                });
            }
        }

        if let Some(st) = stages {
            st.ids = inp.ids.clone();
            st.word_pos = inp.word_pos.clone();
            st.marker_pos = inp.marker_pos.clone();
            st.hidden = hs.to_vec2::<f32>()?;
            st.boundary = bs.to_vec2::<f32>()?;
            st.start_logits = start_l;
            st.end_logits = end_l;
            st.inside_prefix = prefix;
            st.inside_mean = imean;
            st.pool = cands;
            st.pool_compat = c_compat;
            st.pair_logits = pair_logits;
            st.null_logits = null;
            st.spans = spans.clone();
        }
        Ok(spans)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// gliner2 `_deduplicate_pool`: keep each key's highest-priority occurrence,
/// then the `capacity` best keys. Returns `(keys, valid)` of length
/// `capacity`, in priority order (padding is key 0, invalid).
fn dedupe_pool(keys: &[usize], scores: &[f32], valid: &[bool], capacity: usize, nb: usize) -> (Vec<usize>, Vec<bool>) {
    let invalid_key = nb * nb;
    let n = keys.len();
    let k: Vec<usize> = (0..n).map(|i| if valid[i] { keys[i] } else { invalid_key }).collect();
    let s: Vec<f32> = (0..n).map(|i| if valid[i] { scores[i] } else { MASK_LOGIT }).collect();
    // stable sort by score descending, then stable sort by key ascending
    let mut o: Vec<usize> = (0..n).collect();
    o.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal));
    o.sort_by_key(|&i| k[i]);
    let mut keep = vec![false; n];
    for (r, &i) in o.iter().enumerate() {
        let first = r == 0 || k[o[r - 1]] != k[i];
        keep[r] = valid[i] && first;
    }
    let pri: Vec<f32> = (0..n).map(|r| if keep[r] { s[o[r]] } else { MASK_LOGIT }).collect();
    let mut by = (0..n).collect::<Vec<usize>>();
    by.sort_by(|&a, &b| pri[b].partial_cmp(&pri[a]).unwrap_or(std::cmp::Ordering::Equal));
    by.truncate(capacity);
    let mut out_k: Vec<usize> = by.iter().map(|&r| k[o[r]]).collect();
    let mut out_v: Vec<bool> = by.iter().map(|&r| keep[r]).collect();
    out_k.resize(capacity, 0);
    out_v.resize(capacity, false);
    for (key, v) in out_k.iter_mut().zip(&out_v) {
        if !v {
            *key = 0;
        }
    }
    (out_k, out_v)
}

type RankKey = (f64, usize, usize, usize);

fn rank_cmp(a: &RankKey, b: &RankKey) -> std::cmp::Ordering {
    a.0.partial_cmp(&b.0)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.1.cmp(&b.1))
        .then(a.2.cmp(&b.2))
        .then(a.3.cmp(&b.3))
}

/// gliner2's `flat` overlap policy (`resolve_overlaps(..., "disallow")`): the
/// maximum-total-score non-overlapping set of half-open spans, by weighted
/// interval scheduling with its tie rules, ranked by descending score.
pub fn resolve_flat(items: &[(f32, usize, usize)]) -> Vec<(f32, usize, usize)> {
    if items.is_empty() {
        return vec![];
    }
    let rk = |i: usize| -> RankKey { (-(items[i].0 as f64), items[i].1, items[i].2, i) };
    let mut ranked: Vec<usize> = (0..items.len()).collect();
    ranked.sort_by(|&a, &b| rank_cmp(&rk(a), &rk(b)));
    let mut seen = std::collections::HashSet::new();
    let distinct: Vec<usize> = ranked.into_iter().filter(|&i| seen.insert((items[i].1, items[i].2))).collect();
    let mut by_end = distinct;
    by_end.sort_by(|&a, &b| {
        items[a]
            .2
            .cmp(&items[b].2)
            .then(items[a].1.cmp(&items[b].1))
            .then((-(items[a].0 as f64)).partial_cmp(&-(items[b].0 as f64)).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.cmp(&b))
    });
    let ends: Vec<usize> = by_end.iter().map(|&i| items[i].2).collect();
    let selection_key = |sel: &[usize]| -> Vec<RankKey> {
        let mut v: Vec<RankKey> = sel.iter().map(|&k| rk(by_end[k])).collect();
        v.sort_by(rank_cmp);
        v
    };
    let mut best: Vec<(f64, Vec<usize>)> = vec![(0.0, vec![])];
    for (index, &it) in by_end.iter().enumerate() {
        let start = items[it].1;
        // bisect_right(ends, start, 0, index) - 1
        let pred = ends[..index].partition_point(|&e| e <= start) as isize - 1;
        let (prev_score, prev_sel) = &best[(pred + 1) as usize];
        let mut with_sel = prev_sel.clone();
        with_sel.push(index);
        let with = (prev_score + items[it].0 as f64, with_sel);
        let without = best[index].clone();
        let pick = if with.0 > without.0 {
            with
        } else if with.0 < without.0 {
            without
        } else if with.1.len() > without.1.len() {
            with
        } else if with.1.len() < without.1.len() {
            without
        } else {
            let (a, b) = (selection_key(&with.1), selection_key(&without.1));
            let less = a
                .iter()
                .zip(&b)
                .map(|(x, y)| rank_cmp(x, y))
                .find(|o| *o != std::cmp::Ordering::Equal)
                .unwrap_or(a.len().cmp(&b.len()))
                == std::cmp::Ordering::Less;
            if less {
                with
            } else {
                without
            }
        };
        best.push(pick);
    }
    let mut sel: Vec<usize> = best.last().unwrap().1.iter().map(|&k| by_end[k]).collect();
    sel.sort_by(|&a, &b| rank_cmp(&rk(a), &rk(b)));
    sel.into_iter().map(|i| items[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_match_gliner2_splitter() {
        let w = split_words("Visit https://ex.com/a?q=1 or mail Jane.Doe@ex.org, cc @Johnny_B. Co-op e-mail_x Zoë!");
        let toks: Vec<&str> = w.iter().map(|x| x.0.as_str()).collect();
        assert_eq!(
            toks,
            ["visit", "https://ex.com/a?q=1", "or", "mail", "jane.doe@ex.org", ",", "cc", "@johnny_b", ".", "co-op", "e-mail_x", "zoë", "!"]
        );
        // byte offsets into the original text
        let t = "Zoë met José";
        let w = split_words(t);
        assert_eq!(&t[w[0].1..w[0].2], "Zoë");
        assert_eq!(&t[w[2].1..w[2].2], "José");
        // combining marks stay inside the word
        let t = "Cafe\u{301} au lait";
        assert_eq!(split_words(t)[0].0, "cafe\u{301}");
        assert!(split_words("").is_empty());
    }

    #[test]
    fn normalisation_and_schema() {
        assert_eq!(normalize_text(""), ".");
        assert_eq!(normalize_text("hi"), "hi.");
        assert_eq!(normalize_text("hi!"), "hi!");
        assert_eq!(
            entity_schema_words(&["person", "location"]),
            ["(", "[P]", "entities", "(", "[E]", "person", "[E]", "location", ")", ")"]
        );
    }

    #[test]
    fn log_buckets() {
        assert_eq!(log_bucket_position(0, 256, 512), 0);
        assert_eq!(log_bucket_position(127, 256, 512), 127);
        assert_eq!(log_bucket_position(-128, 256, 512), -128);
        assert_eq!(log_bucket_position(129, 256, 512), 129);
        assert_eq!(log_bucket_position(511, 256, 512), 255);
        assert_eq!(log_bucket_position(-700, 256, 512), -(log_bucket_position(700, 256, 512)));
    }

    #[test]
    fn flat_resolution() {
        // the higher-total disjoint pair beats the single overlapping span
        let r = resolve_flat(&[(0.6, 0, 3), (0.5, 0, 1), (0.5, 2, 3)]);
        assert_eq!(r, vec![(0.5, 0, 1), (0.5, 2, 3)]);
        // duplicates collapse to the best
        let r = resolve_flat(&[(0.3, 1, 2), (0.9, 1, 2)]);
        assert_eq!(r, vec![(0.9, 1, 2)]);
        // ranked by score
        let r = resolve_flat(&[(0.2, 0, 1), (0.8, 4, 5)]);
        assert_eq!(r, vec![(0.8, 4, 5), (0.2, 0, 1)]);
        assert!(resolve_flat(&[]).is_empty());
    }

    #[test]
    fn pool_dedupe_keeps_best_occurrence() {
        let (k, v) = dedupe_pool(&[5, 7, 5, 9], &[1.0, 3.0, 2.0, 0.5], &[true, true, true, false], 4, 10);
        assert_eq!(k, vec![7, 5, 0, 0]);
        assert_eq!(v, vec![true, true, false, false]);
    }
}
