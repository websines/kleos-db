/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! EmbeddingGemma (`google/embeddinggemma-300m`): a Gemma 3 text model run
//! with bidirectional attention, mean-pooled, then two bias-free dense
//! projections (768 → 3072 → 768) and L2 normalisation, as in its
//! sentence-transformers config. Output is Matryoshka-trained: a prefix of
//! 512/256/128 dims, re-normalised, is a valid smaller embedding.

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use serde::Deserialize;
use tokenizers::{Tokenizer, TruncationParams};

use crate::budget::{self, ModelSlot};
use crate::device::device_name;
use crate::sched::{current_priority, BatchFns, Batcher};
use crate::error::tok_err;
use crate::linear::{Embedding, Linear, Loader, TensorFile};
use crate::{hub, l2_normalize, EngineConfig, Error, Result, TextEmbedder};

/// Prompt EmbeddingGemma expects in front of search queries.
pub const GEMMA_QUERY_PROMPT: &str = "task: search result | query: ";
/// Prompt EmbeddingGemma expects in front of documents.
pub const GEMMA_DOCUMENT_PROMPT: &str = "title: none | text: ";

#[derive(Debug, Clone, Deserialize)]
struct Config {
    hidden_size: usize,
    /// MLP width (only sizes the activation estimate).
    #[serde(default)]
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    rope_local_base_freq: f64,
    sliding_window: usize,
    query_pre_attn_scalar: f64,
    layer_types: Vec<String>,
}

struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: Linear,
    up: Linear,
    down: Linear,
    input_norm: Tensor,
    post_attn_norm: Tensor,
    pre_ff_norm: Tensor,
    post_ff_norm: Tensor,
    sliding: bool,
}

struct Weights {
    embed: Embedding,
    layers: Vec<Layer>,
    norm: Tensor,
    dense1: Linear,
    dense2: Linear,
    device: Device,
    dtype: DType,
    tokenizer: Tokenizer,
    bytes: usize,
}

/// One loaded (or not yet loaded) EmbeddingGemma, shareable between
/// embedders with different prompts and output widths.
pub struct GemmaModel {
    repo: String,
    cfg: Config,
    engine: EngineConfig,
    /// Weights, device gate and budget bookkeeping.
    weights: ModelSlot<Weights>,
    /// Batches rows across concurrent callers.
    batcher: Batcher<Vec<u32>>,
}

impl GemmaModel {
    /// Reads `config.json` of `repo` from the HF cache (fetched if missing);
    /// weights load lazily.
    pub fn new(repo: &str, engine: EngineConfig) -> Result<Arc<Self>> {
        hub::ensure_hf_files(
            repo,
            &["config.json", "tokenizer.json", "model.safetensors", "2_Dense/model.safetensors", "3_Dense/model.safetensors"],
            &[],
        )?;
        let cfg: Config =
            serde_json::from_slice(&std::fs::read(hub::resolve_hf_file(repo, "config.json")?)?)?;
        if cfg.layer_types.len() != cfg.num_hidden_layers {
            return Err(Error::Config("layer_types does not match num_hidden_layers".into()));
        }
        Ok(Arc::new(Self {
            repo: repo.to_string(),
            cfg,
            engine,
            weights: ModelSlot::new(format!("embedder:{repo}")),
            batcher: Batcher::new(),
        }))
    }

    /// An embedder over this model: `prompt` is prepended to every text,
    /// `dim` (≤ 768) truncates the Matryoshka output.
    pub fn embedder(self: &Arc<Self>, prompt: &str, dim: Option<usize>) -> Result<GemmaEmbedder> {
        let full = self.cfg.hidden_size;
        let dim = dim.unwrap_or(full);
        if dim == 0 || dim > full {
            return Err(Error::Config(format!("output dim must be in 1..={full}")));
        }
        Ok(GemmaEmbedder {
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

    /// Activation bytes of one forward pass over `n` rows of padded length `l`.
    fn act_bytes(&self, n: usize, l: usize, dt: DType) -> usize {
        let c = &self.cfg;
        let inter = if c.intermediate_size > 0 { c.intermediate_size } else { 4 * c.hidden_size };
        // scores + probabilities, plus the two [b, 1, l, l] masks
        budget::encoder_activation_bytes(n, l, c.hidden_size, inter, c.num_attention_heads, dt.size_in_bytes(), 2)
            + 2 * 2 * n * l * l * 4
    }

    fn load(&self) -> Result<Weights> {
        let device = self.engine.device.device()?;
        let mut tokenizer =
            Tokenizer::from_file(hub::resolve_hf_file(&self.repo, "tokenizer.json")?)
                .map_err(tok_err)?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: self.engine.max_seq_len,
                ..Default::default()
            }))
            .map_err(tok_err)?;
        tokenizer.with_padding(None);
        let st = TensorFile::open(&hub::resolve_hf_file(&self.repo, "model.safetensors")?)?;
        let d1 = TensorFile::open(&hub::resolve_hf_file(&self.repo, "2_Dense/model.safetensors")?)?;
        let d2 = TensorFile::open(&hub::resolve_hf_file(&self.repo, "3_Dense/model.safetensors")?)?;
        let mut ld = Loader {
            st: &st,
            device: &device,
            precision: self.engine.precision,
            bytes: 0,
        };
        let dtype = ld.act_dtype();
        // Gemma RMSNorm scales by (1 + w): fold the 1 in once, in the
        // activation dtype, so the fused rms_norm kernel can be used.
        let norm_w = |t: Tensor| -> Result<Tensor> { Ok((t + 1.0)?.to_dtype(dtype)?) };
        let embed = ld.embedding("embed_tokens.weight")?;
        let mut layers = Vec::with_capacity(self.cfg.num_hidden_layers);
        for i in 0..self.cfg.num_hidden_layers {
            let p = |s: &str| format!("layers.{i}.{s}");
            layers.push(Layer {
                q: ld.linear(&p("self_attn.q_proj.weight"))?,
                k: ld.linear(&p("self_attn.k_proj.weight"))?,
                v: ld.linear(&p("self_attn.v_proj.weight"))?,
                o: ld.linear(&p("self_attn.o_proj.weight"))?,
                q_norm: norm_w(ld.f32(&p("self_attn.q_norm.weight"))?)?,
                k_norm: norm_w(ld.f32(&p("self_attn.k_norm.weight"))?)?,
                gate: ld.linear(&p("mlp.gate_proj.weight"))?,
                up: ld.linear(&p("mlp.up_proj.weight"))?,
                down: ld.linear(&p("mlp.down_proj.weight"))?,
                input_norm: norm_w(ld.f32(&p("input_layernorm.weight"))?)?,
                post_attn_norm: norm_w(ld.f32(&p("post_attention_layernorm.weight"))?)?,
                pre_ff_norm: norm_w(ld.f32(&p("pre_feedforward_layernorm.weight"))?)?,
                post_ff_norm: norm_w(ld.f32(&p("post_feedforward_layernorm.weight"))?)?,
                sliding: self.cfg.layer_types[i] == "sliding_attention",
            });
        }
        let norm = norm_w(ld.f32("norm.weight")?)?;
        let mut ld1 = Loader { st: &d1, device: &device, precision: self.engine.precision, bytes: 0 };
        let dense1 = ld1.linear("linear.weight")?;
        let mut ld2 = Loader { st: &d2, device: &device, precision: self.engine.precision, bytes: 0 };
        let dense2 = ld2.linear("linear.weight")?;
        Ok(Weights {
            embed,
            layers,
            norm,
            dense1,
            dense2,
            device: device.clone(),
            dtype,
            tokenizer,
            bytes: ld.bytes + ld1.bytes + ld2.bytes,
        })
    }

    /// Embeds full-width, normalised vectors.
    fn embed_full(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        with_pool(|| self.embed_full_inner(texts))
    }

    fn embed_full_inner(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let w = self.weights()?;
        let encs = w
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(tok_err)?;
        let rows: Vec<Vec<u32>> = encs.into_iter().map(|e| e.get_ids().to_vec()).collect();
        // rows from concurrent callers share forward passes (sched.rs); each
        // pass fits the token budget and the activation cap (budget.rs)
        let _hold = self.weights.hold();
        let dt = w.dtype;
        let max_tokens = self.engine.max_batch_tokens.max(1);
        let max_rows = |l: usize| budget::rows_for(l, max_tokens, |l| self.act_bytes(1, l, dt));
        let act = |n: usize, l: usize| self.act_bytes(n, l, dt);
        let forward = |batch: &[&Vec<u32>]| self.forward(&w, batch.iter().map(|r| r.as_slice()).collect());
        // let candle's Metal pool drop the last batch's buffers (it frees
        // unused ones only on a synchronize)
        let after = || {
            let _ = w.device.synchronize();
        };
        let fns = BatchFns { max_rows: &max_rows, act_bytes: &act, forward: &forward, after: &after, gate: self.weights.gate() };
        self.batcher.run(rows, current_priority(), &fns)
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
        let valid_t = Tensor::from_vec(valid.clone(), (b, l), dev)?;
        // additive key masks [b, 1, l(q), l(k)]: padding, plus |i-j| < window for sliding layers
        let neg = -3.0e4f32; // finite in f16 too; exp(-3e4) == 0
        let key_pad = valid_t
            .affine(1.0, -1.0)?
            .affine(-(neg as f64), 0.0)?
            .reshape((b, 1, 1, l))?;
        let full_mask = key_pad.broadcast_as((b, 1, l, l))?.contiguous()?;
        let sliding_mask = if l > cfg.sliding_window {
            let band: Vec<f32> = (0..l * l)
                .map(|k| {
                    let (i, j) = (k / l, k % l);
                    if i.abs_diff(j) < cfg.sliding_window { 0.0 } else { neg }
                })
                .collect();
            let band = Tensor::from_vec(band, (1, 1, l, l), dev)?;
            full_mask.broadcast_add(&band)?
        } else {
            full_mask.clone()
        };
        let dt = w.dtype;
        let (cos_g, sin_g) = rope_tables(cfg.rope_theta, cfg.head_dim, l, dev, dt)?;
        let (cos_l, sin_l) = rope_tables(cfg.rope_local_base_freq, cfg.head_dim, l, dev, dt)?;
        let eps = cfg.rms_norm_eps;
        let mut x = (w.embed.lookup(&ids_t)? * (cfg.hidden_size as f64).sqrt())?.to_dtype(dt)?;
        let (h, kvh, hd) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let scale = 1.0 / cfg.query_pre_attn_scalar.sqrt();
        for layer in &w.layers {
            let (mask, cos, sin) = if layer.sliding {
                (&sliding_mask, &cos_l, &sin_l)
            } else {
                (&full_mask, &cos_g, &sin_g)
            };
            let res = x.clone();
            let xn = rms(&x, &layer.input_norm, eps)?;
            let q = layer.q.forward(&xn)?.reshape((b, l, h, hd))?.transpose(1, 2)?;
            let k = layer.k.forward(&xn)?.reshape((b, l, kvh, hd))?.transpose(1, 2)?;
            let v = layer.v.forward(&xn)?.reshape((b, l, kvh, hd))?.transpose(1, 2)?;
            let q = rms(&q, &layer.q_norm, eps)?;
            let k = rms(&k, &layer.k_norm, eps)?;
            let q = candle_nn::rotary_emb::rope(&q, cos, sin)?;
            let k = candle_nn::rotary_emb::rope(&k, cos, sin)?;
            let a = if dev.is_metal() {
                // Fused attention kernel (handles GQA without tiling k/v).
                // candle's f32 kernel does not fit head_dim 256 in threadgroup
                // memory, so f32-activation modes attend in f16: q and k are
                // RMS-normalised, and the result is checked against f32 in tests.
                let adt = if dt == DType::F32 { DType::F16 } else { dt };
                let m = mask.to_dtype(adt)?.broadcast_as((b, h, l, l))?;
                candle_nn::ops::sdpa(
                    &q.to_dtype(adt)?,
                    &k.to_dtype(adt)?,
                    &v.to_dtype(adt)?.contiguous()?,
                    Some(&m),
                    false,
                    scale as f32,
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
                let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?
                    .to_dtype(DType::F32)?
                    .broadcast_add(mask)?;
                let att = candle_nn::ops::softmax_last_dim(&att)?.to_dtype(dt)?;
                att.matmul(&v)?
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
        }
        let x = rms(&x, &w.norm, eps)?.to_dtype(DType::F32)?;
        // mean over valid tokens
        let m = valid_t.reshape((b, l, 1))?;
        let summed = x.broadcast_mul(&m)?.sum(1)?;
        let counts = m.sum(1)?.clamp(1.0, f64::MAX)?;
        let pooled = summed.broadcast_div(&counts)?.to_dtype(dt)?;
        let y = w.dense2.forward(&w.dense1.forward(&pooled)?)?.to_dtype(DType::F32)?;
        let rows: Vec<Vec<f32>> = y.to_vec2()?;
        // Release this batch's GPU buffers: candle's Metal pool only drops
        // unreferenced buffers when the device is synchronised.
        dev.synchronize()?;
        Ok(rows)
    }
}

/// Run `f` inside an Objective-C autorelease pool. Metal objects that candle
/// creates (command buffers, encoders, buffer handles) can be autoreleased;
/// on a thread with no pool (any Rust thread) they are never released, which
/// kept every loaded model's GPU and CPU buffers alive after `unload()` in
/// debug builds (measured: +315 MB per load/unload cycle; release builds
/// happened to avoid the autorelease path).
#[cfg(feature = "metal")]
pub(crate) fn with_pool<R>(f: impl FnOnce() -> R) -> R {
    objc2::rc::autoreleasepool(|_| f())
}

#[cfg(not(feature = "metal"))]
pub(crate) fn with_pool<R>(f: impl FnOnce() -> R) -> R {
    f()
}

/// Gemma RMSNorm with the `(1 + w)` scale pre-folded into `w`.
fn rms(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    Ok(candle_nn::ops::rms_norm(&x.contiguous()?, w, eps as f32)?)
}

fn repeat_kv(x: &Tensor, r: usize) -> Result<Tensor> {
    let (b, kvh, l, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?
        .broadcast_as((b, kvh, r, l, d))?
        .reshape((b, kvh * r, l, d))?)
}

fn rope_tables(
    theta: f64,
    head_dim: usize,
    l: usize,
    dev: &Device,
    dt: DType,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64) as f32)
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

/// EmbeddingGemma with a fixed prompt and output width.
pub struct GemmaEmbedder {
    model: Arc<GemmaModel>,
    prompt: String,
    dim: usize,
}

impl GemmaEmbedder {
    /// Load `repo` (e.g. `google/embeddinggemma-300m`) with document prompt
    /// and full width. Use [`GemmaModel::embedder`] to share one model
    /// between a query and a document embedder.
    pub fn new(repo: &str, engine: EngineConfig) -> Result<Self> {
        GemmaModel::new(repo, engine)?.embedder(GEMMA_DOCUMENT_PROMPT, None)
    }
    /// The shared model.
    pub fn model(&self) -> &Arc<GemmaModel> {
        &self.model
    }
}

impl TextEmbedder for GemmaEmbedder {
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
