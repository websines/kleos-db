/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! EmbeddingGemma 2 (`google/embeddinggemma-2`, Apache-2.0), text path only:
//! a Gemma 4 bidirectional encoder (24 layers, d 512) with projection-only
//! per-layer embeddings (PLE), a final 512 → 768 projection applied per
//! token, mean pooling and L2 normalisation, as in its sentence-transformers
//! config. The checkpoint also holds a vision and an audio tower; only the
//! `language_model.*` tensors are read (~271M parameters). Output is
//! Matryoshka-trained: a prefix of 512/256/128 dims, re-normalised, is a valid
//! smaller embedding.
//!
//! Differences from EmbeddingGemma (v1, [`crate::GemmaModel`]) that matter for
//! the port: RMSNorm scales by `w` (not `1 + w`); attention scaling is 1 (q and
//! k are RMS-normalised, v is RMS-normalised without a scale); the four full-
//! attention layers have head_dim 512 and one KV head, the sliding ones
//! head_dim 256 and two; the sliding window is `|i - j| <= 512`; every layer
//! ends with a PLE block and a learned scalar.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use serde::Deserialize;
use tokenizers::{Tokenizer, TruncationParams};

use crate::budget::{self, ModelSlot};
use crate::device::device_name;
use crate::sched::{current_priority, BatchFns, Batcher};
use crate::error::tok_err;
use crate::gemma::with_pool;
use crate::linear::{Embedding, Linear, Loader, TensorFile};
use crate::{hub, l2_normalize, EngineConfig, Error, Result, TextEmbedder};

/// Hugging Face repo of EmbeddingGemma 2 (Apache-2.0, not gated).
pub const EGEMMA2_REPO: &str = "google/embeddinggemma-2";
/// Prompt EmbeddingGemma 2 expects in front of search queries.
pub const EGEMMA2_QUERY_PROMPT: &str = "task: search result | query: ";
/// Prompt EmbeddingGemma 2 expects in front of documents.
pub const EGEMMA2_DOCUMENT_PROMPT: &str = "title: none | text: ";

#[derive(Debug, Clone, Deserialize)]
struct RopeParam {
    rope_theta: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct LayerOverride {
    head_dim: usize,
    num_key_value_heads: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct TextConfig {
    hidden_size: usize,
    hidden_size_per_layer_input: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    sliding_window: usize,
    layer_types: Vec<String>,
    #[serde(default)]
    per_layer_config: HashMap<String, LayerOverride>,
    rope_parameters: HashMap<String, RopeParam>,
    embedding_dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct Config {
    text_config: TextConfig,
}

struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    v_norm: Tensor,
    gate: Linear,
    up: Linear,
    down: Linear,
    input_norm: Tensor,
    post_attn_norm: Tensor,
    pre_ff_norm: Tensor,
    post_ff_norm: Tensor,
    ple_gate: Linear,
    ple_proj: Linear,
    ple_norm: Tensor,
    scalar: f64,
    sliding: bool,
    heads_kv: usize,
    head_dim: usize,
}

struct Weights {
    embed: Embedding,
    ple_proj: Linear,
    ple_norm: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    out_proj: Linear,
    device: Device,
    dtype: DType,
    tokenizer: Tokenizer,
    bytes: usize,
}

/// One loaded (or not yet loaded) EmbeddingGemma 2, shareable between
/// embedders with different prompts and output widths.
pub struct Egemma2Model {
    repo: String,
    cfg: TextConfig,
    engine: EngineConfig,
    /// Weights, device gate and budget bookkeeping.
    weights: ModelSlot<Weights>,
    /// Batches rows across concurrent callers.
    batcher: Batcher<Vec<u32>>,
}

impl Egemma2Model {
    /// Reads `config.json` of `repo` from the HF cache (downloaded if
    /// missing; the repo is not gated); weights load lazily.
    pub fn new(repo: &str, engine: EngineConfig) -> Result<Arc<Self>> {
        hub::ensure_hf_files(repo, &["config.json", "tokenizer.json", "model.safetensors"], &[])?;
        let cfg: Config = serde_json::from_slice(&std::fs::read(hub::resolve_hf_file(repo, "config.json")?)?)?;
        let cfg = cfg.text_config;
        if cfg.layer_types.len() != cfg.num_hidden_layers {
            return Err(Error::Config("layer_types does not match num_hidden_layers".into()));
        }
        for t in ["full_attention", "sliding_attention"] {
            if !cfg.rope_parameters.contains_key(t) {
                return Err(Error::Config(format!("rope_parameters lacks '{t}'")));
            }
        }
        Ok(Arc::new(Self {
            repo: repo.to_string(),
            cfg,
            engine,
            weights: ModelSlot::new(format!("embedder:{repo}")),
            batcher: Batcher::new(),
        }))
    }

    /// Full output width (768).
    pub fn full_dim(&self) -> usize {
        self.cfg.embedding_dim
    }

    /// The repo this model loads from.
    pub fn repo(&self) -> &str {
        &self.repo
    }

    /// An embedder over this model: `prompt` is prepended to every text,
    /// `dim` (≤ 768) truncates the Matryoshka output.
    pub fn embedder(self: &Arc<Self>, prompt: &str, dim: Option<usize>) -> Result<Egemma2Embedder> {
        let full = self.cfg.embedding_dim;
        let dim = dim.unwrap_or(full);
        if dim == 0 || dim > full {
            return Err(Error::Config(format!("output dim must be in 1..={full}")));
        }
        Ok(Egemma2Embedder {
            model: self.clone(),
            prompt: prompt.to_string(),
            dim,
        })
    }

    /// Drop the weights (they load again on next use).
    pub fn unload(&self) {
        self.weights.unload();
    }

    /// Whether the weights are in memory.
    pub fn is_loaded(&self) -> bool {
        self.weights.is_loaded()
    }

    /// `(device, bytes of weights)` once loaded.
    pub fn loaded_info(&self) -> Option<(&'static str, usize)> {
        self.weights.peek().map(|w| (device_name(&w.device), w.bytes))
    }

    fn weights(&self) -> Result<Arc<Weights>> {
        self.weights.get_or_load(|| self.load(), |w| w.bytes)
    }

    /// Activation bytes of one forward pass over `n` rows of padded length `l`:
    /// counted at the widest attention projection (4 heads × 512 on the full
    /// layers), plus the masks and the per-layer-embedding signal.
    fn act_bytes(&self, n: usize, l: usize, dt: DType) -> usize {
        let c = &self.cfg;
        let wide = c.hidden_size.max(c.num_attention_heads * 512);
        budget::encoder_activation_bytes(n, l, wide, 2048, c.num_attention_heads, dt.size_in_bytes(), 2)
            + 2 * 2 * n * l * l * 4
            + 2 * n * l * c.num_hidden_layers * c.hidden_size_per_layer_input * 4
    }

    fn layer_shape(&self, i: usize) -> (usize, usize) {
        match self.cfg.per_layer_config.get(&format!("{i:02}")).or_else(|| self.cfg.per_layer_config.get(&i.to_string())) {
            Some(o) => (o.num_key_value_heads, o.head_dim),
            None => (self.cfg.num_key_value_heads, self.cfg.head_dim),
        }
    }

    fn load(&self) -> Result<Weights> {
        let device = self.engine.device.device()?;
        let mut tokenizer = Tokenizer::from_file(hub::resolve_hf_file(&self.repo, "tokenizer.json")?).map_err(tok_err)?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: self.engine.max_seq_len,
                ..Default::default()
            }))
            .map_err(tok_err)?;
        tokenizer.with_padding(None);
        let st = TensorFile::open(&hub::resolve_hf_file(&self.repo, "model.safetensors")?)?;
        let mut ld = Loader {
            st: &st,
            device: &device,
            precision: self.engine.precision,
            bytes: 0,
        };
        let dtype = ld.act_dtype();
        let norm = |ld: &mut Loader, n: &str| -> Result<Tensor> { Ok(ld.f32(n)?.to_dtype(dtype)?) };
        let p0 = "language_model.";
        let embed = ld.embedding(&format!("{p0}embed_tokens.weight"))?;
        let ple_proj = ld.linear(&format!("{p0}ple.per_layer_model_projection.weight"))?;
        let ple_norm = norm(&mut ld, &format!("{p0}ple.per_layer_projection_norm.weight"))?;
        let mut layers = Vec::with_capacity(self.cfg.num_hidden_layers);
        for i in 0..self.cfg.num_hidden_layers {
            let p = |s: &str| format!("{p0}layers.{i}.{s}");
            let (heads_kv, head_dim) = self.layer_shape(i);
            let scalar = ld.f32(&p("layer_scalar"))?.flatten_all()?.to_vec1::<f32>()?[0] as f64;
            layers.push(Layer {
                q: ld.linear(&p("self_attn.q_proj.weight"))?,
                k: ld.linear(&p("self_attn.k_proj.weight"))?,
                v: ld.linear(&p("self_attn.v_proj.weight"))?,
                o: ld.linear(&p("self_attn.o_proj.weight"))?,
                q_norm: norm(&mut ld, &p("self_attn.q_norm.weight"))?,
                k_norm: norm(&mut ld, &p("self_attn.k_norm.weight"))?,
                v_norm: Tensor::ones(head_dim, dtype, &device)?,
                gate: ld.linear(&p("mlp.gate_proj.weight"))?,
                up: ld.linear(&p("mlp.up_proj.weight"))?,
                down: ld.linear(&p("mlp.down_proj.weight"))?,
                input_norm: norm(&mut ld, &p("input_layernorm.weight"))?,
                post_attn_norm: norm(&mut ld, &p("post_attention_layernorm.weight"))?,
                pre_ff_norm: norm(&mut ld, &p("pre_feedforward_layernorm.weight"))?,
                post_ff_norm: norm(&mut ld, &p("post_feedforward_layernorm.weight"))?,
                ple_gate: ld.linear(&p("ple_block.per_layer_input_gate.weight"))?,
                ple_proj: ld.linear(&p("ple_block.per_layer_projection.weight"))?,
                ple_norm: norm(&mut ld, &p("ple_block.post_per_layer_input_norm.weight"))?,
                scalar,
                sliding: self.cfg.layer_types[i] == "sliding_attention",
                heads_kv,
                head_dim,
            });
        }
        let norm_w = norm(&mut ld, &format!("{p0}norm.weight"))?;
        let out_proj = ld.linear(&format!("{p0}embedding_projection.weight"))?;
        Ok(Weights {
            embed,
            ple_proj,
            ple_norm,
            layers,
            norm: norm_w,
            out_proj,
            device: device.clone(),
            dtype,
            tokenizer,
            bytes: ld.bytes,
        })
    }

    /// Embeds full-width, normalised vectors.
    fn embed_full(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        with_pool(|| self.embed_full_inner(texts))
    }

    fn embed_full_inner(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let w = self.weights()?;
        let encs = w.tokenizer.encode_batch(texts.to_vec(), true).map_err(tok_err)?;
        let rows: Vec<Vec<u32>> = encs.into_iter().map(|e| e.get_ids().to_vec()).collect();
        // rows from concurrent callers share forward passes (sched.rs); each
        // pass fits the token budget and the activation cap (budget.rs)
        let _hold = self.weights.hold();
        let dt = w.dtype;
        let max_tokens = self.engine.max_batch_tokens.max(1);
        let max_rows = |l: usize| budget::rows_for(l, max_tokens, |l| self.act_bytes(1, l, dt));
        let act = |n: usize, l: usize| self.act_bytes(n, l, dt);
        let forward = |batch: &[&Vec<u32>]| self.forward(&w, batch.iter().map(|r| r.as_slice()).collect());
        let after = || {
            let _ = w.device.synchronize();
        };
        let fns = BatchFns { max_rows: &max_rows, act_bytes: &act, forward: &forward, after: &after, gate: self.weights.gate() };
        self.batcher.run(rows, current_priority(), &fns)
    }

    /// Token ids as this model feeds them (for tests).
    pub fn token_ids(&self, text: &str) -> Result<Vec<u32>> {
        let w = self.weights()?;
        Ok(w.tokenizer.encode(text, true).map_err(tok_err)?.get_ids().to_vec())
    }

    fn forward(&self, w: &Weights, ids: Vec<&[u32]>) -> Result<Vec<Vec<f32>>> {
        let cfg = &self.cfg;
        let dev = &w.device;
        let b = ids.len();
        let l = ids.iter().map(|x| x.len()).max().unwrap_or(1).max(1);
        let mut flat = vec![0u32; b * l];
        let mut valid = vec![0f32; b * l];
        for (r, row) in ids.iter().enumerate() {
            flat[r * l..r * l + row.len()].copy_from_slice(row);
            valid[r * l..r * l + row.len()].iter_mut().for_each(|x| *x = 1.0);
        }
        let ids_t = Tensor::from_vec(flat, (b, l), dev)?;
        let valid_t = Tensor::from_vec(valid, (b, l), dev)?;
        let neg = -3.0e4f32;
        let key_pad = valid_t.affine(1.0, -1.0)?.affine(-(neg as f64), 0.0)?.reshape((b, 1, 1, l))?;
        let full_mask = key_pad.broadcast_as((b, 1, l, l))?.contiguous()?;
        let sliding_mask = if l > cfg.sliding_window + 1 {
            let band: Vec<f32> = (0..l * l)
                .map(|k| {
                    let (i, j) = (k / l, k % l);
                    if i.abs_diff(j) <= cfg.sliding_window { 0.0 } else { neg }
                })
                .collect();
            full_mask.broadcast_add(&Tensor::from_vec(band, (1, 1, l, l), dev)?)?
        } else {
            full_mask.clone()
        };
        let dt = w.dtype;
        let eps = cfg.rms_norm_eps;
        let theta = |t: &str| cfg.rope_parameters[t].rope_theta;
        let mut ropes: HashMap<(bool, usize), (Tensor, Tensor)> = HashMap::new();
        for layer in &w.layers {
            let key = (layer.sliding, layer.head_dim);
            if let std::collections::hash_map::Entry::Vacant(e) = ropes.entry(key) {
                let t = if layer.sliding { theta("sliding_attention") } else { theta("full_attention") };
                e.insert(rope_tables(t, layer.head_dim, l, dev, dt)?);
            }
        }
        let x0 = (w.embed.lookup(&ids_t)? * (cfg.hidden_size as f64).sqrt())?.to_dtype(dt)?;
        // per-layer inputs [b, l, n_layers, d_ple]
        let nl = cfg.num_hidden_layers;
        let dp = cfg.hidden_size_per_layer_input;
        let ple = (w.ple_proj.forward(&x0)? * (cfg.hidden_size as f64).powf(-0.5))?.reshape((b, l, nl, dp))?;
        let ple = rms(&ple, &w.ple_norm, eps)?;
        let h = cfg.num_attention_heads;
        let mut x = x0;
        for (li, layer) in w.layers.iter().enumerate() {
            let (kvh, hd) = (layer.heads_kv, layer.head_dim);
            let mask = if layer.sliding { &sliding_mask } else { &full_mask };
            let (cos, sin) = &ropes[&(layer.sliding, hd)];
            let res = x.clone();
            let xn = rms(&x, &layer.input_norm, eps)?;
            let q = layer.q.forward(&xn)?.reshape((b, l, h, hd))?.transpose(1, 2)?;
            let k = layer.k.forward(&xn)?.reshape((b, l, kvh, hd))?.transpose(1, 2)?;
            let v = layer.v.forward(&xn)?.reshape((b, l, kvh, hd))?.transpose(1, 2)?;
            let q = rms(&q, &layer.q_norm, eps)?;
            let k = rms(&k, &layer.k_norm, eps)?;
            let v = rms(&v, &layer.v_norm, eps)?;
            let q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
            let k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;
            let a = if dev.is_metal() && hd <= 256 {
                let adt = if dt == DType::F32 { DType::F16 } else { dt };
                let m = mask.to_dtype(adt)?.broadcast_as((b, h, l, l))?;
                candle_nn::ops::sdpa(
                    &q.to_dtype(adt)?,
                    &k.to_dtype(adt)?,
                    &v.to_dtype(adt)?.contiguous()?,
                    Some(&m),
                    false,
                    1.0,
                    1.0,
                )?
                .to_dtype(dt)?
            } else {
                let (k, v) = if kvh == h {
                    (k, v.contiguous()?)
                } else {
                    let r = h / kvh;
                    (repeat_kv(&k, r)?, repeat_kv(&v, r)?)
                };
                let att = q
                    .contiguous()?
                    .matmul(&k.transpose(2, 3)?.contiguous()?)?
                    .to_dtype(DType::F32)?
                    .broadcast_add(mask)?;
                let att = candle_nn::ops::softmax_last_dim(&att)?.to_dtype(dt)?;
                att.matmul(&v.contiguous()?)?
            };
            let a = a.transpose(1, 2)?.reshape((b, l, h * hd))?;
            let a = layer.o.forward(&a)?;
            x = (res + rms(&a, &layer.post_attn_norm, eps)?)?;
            let res = x.clone();
            let xn = rms(&x, &layer.pre_ff_norm, eps)?;
            let g = layer.gate.forward(&xn)?.gelu()?;
            let u = layer.up.forward(&xn)?;
            let f = layer.down.forward(&(g * u)?)?;
            x = (res + rms(&f, &layer.post_ff_norm, eps)?)?;
            // PLE block: gate, multiply by this layer's slice, project back
            let pin = ple.narrow(2, li, 1)?.squeeze(2)?;
            let g = layer.ple_gate.forward(&x)?.gelu()?;
            let pp = layer.ple_proj.forward(&(g * pin)?)?;
            x = (&x + rms(&pp, &layer.ple_norm, eps)?)?;
            if layer.scalar != 1.0 {
                x = (x * layer.scalar)?;
            }
        }
        let x = rms(&x, &w.norm, eps)?;
        let x = w.out_proj.forward(&x)?.to_dtype(DType::F32)?;
        let m = valid_t.reshape((b, l, 1))?;
        let summed = x.broadcast_mul(&m)?.sum(1)?;
        let counts = m.sum(1)?.clamp(1.0, f64::MAX)?;
        let pooled = summed.broadcast_div(&counts)?;
        let rows: Vec<Vec<f32>> = pooled.to_vec2()?;
        dev.synchronize()?;
        Ok(rows)
    }
}

/// RMSNorm `x / rms(x) * w` (EmbeddingGemma 2 scales by `w` itself).
fn rms(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    Ok(candle_nn::ops::rms_norm(&x.contiguous()?, w, eps as f32)?)
}

fn repeat_kv(x: &Tensor, r: usize) -> Result<Tensor> {
    let (b, kvh, l, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?.broadcast_as((b, kvh, r, l, d))?.reshape((b, kvh * r, l, d))?)
}

fn rope_tables(theta: f64, head_dim: usize, l: usize, dev: &Device, dt: DType) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
        .collect();
    let mut cos = Vec::with_capacity(l * half);
    let mut sin = Vec::with_capacity(l * half);
    for p in 0..l {
        for f in &inv {
            let a = p as f32 * f;
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    Ok((
        Tensor::from_vec(cos, (l, half), dev)?.to_dtype(dt)?,
        Tensor::from_vec(sin, (l, half), dev)?.to_dtype(dt)?,
    ))
}

/// EmbeddingGemma 2 with a fixed prompt and output width.
pub struct Egemma2Embedder {
    model: Arc<Egemma2Model>,
    prompt: String,
    dim: usize,
}

impl Egemma2Embedder {
    /// Load `repo` (e.g. `google/embeddinggemma-2`) with the document prompt
    /// and full width.
    pub fn new(repo: &str, engine: EngineConfig) -> Result<Self> {
        Egemma2Model::new(repo, engine)?.embedder(EGEMMA2_DOCUMENT_PROMPT, None)
    }
    /// The shared model.
    pub fn model(&self) -> &Arc<Egemma2Model> {
        &self.model
    }
}

impl TextEmbedder for Egemma2Embedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let prompted: Vec<String> = texts.iter().map(|t| format!("{}{t}", self.prompt)).collect();
        let mut vecs = self.model.embed_full(&prompted)?;
        for v in &mut vecs {
            if self.dim < v.len() {
                v.truncate(self.dim);
            }
            l2_normalize(v);
        }
        Ok(vecs)
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn model_id(&self) -> String {
        format!(
            "{}:{}:{}:{}:{}",
            self.model.repo,
            self.model.engine.precision.name(),
            self.model.engine.max_seq_len,
            self.dim,
            self.prompt
        )
    }
    fn max_batch(&self) -> usize {
        64
    }
}
