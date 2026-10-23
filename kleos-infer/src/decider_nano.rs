/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! `tasksource/tasksource-decider-nano` (Apache-2.0): a 150M joint
//! typed-decision cross-encoder on a ModernBERT / Ettin encoder
//! (`cross-encoder/ettin-reranker-150m-v1`), as a [`Decider`] backend.
//!
//! Port of the repository's `decider.py`, layout `shared`, pooling `opt`,
//! attention `cached` (its `decider_config.json`). One request is one row:
//!
//! ```text
//! [CLS] state [SEP] | Question: <question>\nOption: | ␠<opt 1> [SEP] | ␠<opt 2> [SEP] | ...
//! ```
//!
//! with a tree attention mask: the state attends to itself, the question
//! block to the state and itself, each option block to the state, the
//! question and itself (never to other options). Rotary positions are those
//! of the single `(state, "Question: q\nOption: o")` pair, so every option
//! block starts at the same position. Each option's logit is a linear head on
//! the mean of its block's final hidden states (option tokens plus its
//! `[SEP]`); the pooled vector is what [`DecideAnswer::features`] returns
//! (768 wide).
//!
//! ModernBERT: pre-LayerNorm (no biases anywhere), fused `Wqkv`, RoPE
//! (θ = 160 000 on every layer for this checkpoint), GeGLU MLP, global
//! attention on every third layer and a ±64-position window on the others
//! (`sliding_window = local_attention / 2`, inclusive, applied on rotary
//! positions as `decider.py` does).
//!
//! Inputs are never truncated: a request whose state + question + longest
//! option exceeds 8 192 tokens is an error, as in the reference.

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Tensor, D};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::decide::{DecideAnswer, DecideRequest, Decider};
use crate::device::device_name;
use crate::error::tok_err;
use crate::gemma::with_pool;
use crate::linear::{Embedding, Linear, Loader, TensorFile};
use crate::{hub, EngineConfig, Error, Precision, Result};

/// Hugging Face repo of the default decider-nano checkpoint.
pub const DECIDER_NANO_REPO: &str = "tasksource/tasksource-decider-nano";

/// Longest `(state, question, option)` pair the encoder reads.
const MAX_PAIR_TOKENS: usize = 8192;
/// Bound on `rows × length²` per forward pass (the dense masks), as in
/// `decider.py`.
const MAX_MASK_ELEMS: usize = 1 << 26;

#[derive(Debug, Clone, Deserialize)]
struct RopeParams {
    rope_theta: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct Config {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default = "default_eps")]
    norm_eps: f64,
    #[serde(default)]
    global_attn_every_n_layers: Option<usize>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    local_attention: usize,
    #[serde(default)]
    global_rope_theta: Option<f64>,
    #[serde(default)]
    local_rope_theta: Option<f64>,
    #[serde(default)]
    rope_parameters: Option<std::collections::HashMap<String, RopeParams>>,
    #[serde(default)]
    sep_token_id: Option<u32>,
    #[serde(default)]
    pad_token_id: Option<u32>,
}

fn default_eps() -> f64 {
    1e-5
}

impl Config {
    fn sliding(&self, i: usize) -> bool {
        match &self.layer_types {
            Some(t) => t.get(i).map(|s| s == "sliding_attention").unwrap_or(false),
            None => i % self.global_attn_every_n_layers.unwrap_or(3) != 0,
        }
    }
    fn theta(&self, sliding: bool) -> f64 {
        let key = if sliding { "sliding_attention" } else { "full_attention" };
        if let Some(p) = self.rope_parameters.as_ref().and_then(|m| m.get(key)) {
            return p.rope_theta;
        }
        if sliding {
            self.local_rope_theta.unwrap_or(10_000.0)
        } else {
            self.global_rope_theta.unwrap_or(160_000.0)
        }
    }
    fn window(&self) -> usize {
        self.local_attention / 2
    }
}

struct Layer {
    attn_norm: Option<Tensor>,
    wqkv: Linear,
    wo: Linear,
    mlp_norm: Tensor,
    wi: Linear,
    mlp_wo: Linear,
    sliding: bool,
}

struct Weights {
    embed: Embedding,
    embed_norm: Tensor,
    layers: Vec<Layer>,
    final_norm: Tensor,
    /// Scorer `[1, hidden]` and bias, f32.
    scorer_w: Tensor,
    scorer_b: f32,
    /// LayerNorm bias (all zeros: the checkpoint has none), activation dtype.
    zero_bias: Tensor,
    device: Device,
    dtype: DType,
    tokenizer: Tokenizer,
    sep: u32,
    pad: u32,
    bytes: usize,
}

/// tasksource-decider-nano as a [`Decider`]. Weights load lazily from the
/// local HF cache on the first [`Decider::decide`] call.
pub struct DeciderNano {
    repo: String,
    weights_version: String,
    cfg: Config,
    engine: EngineConfig,
    weights: Mutex<Option<Arc<Weights>>>,
    run: Mutex<()>,
}

/// One input row of the encoder: one tokenised request, or several packed
/// back to back (on the CPU) with a block-diagonal mask between them.
#[derive(Clone)]
struct Row {
    ids: Vec<u32>,
    pos: Vec<u32>,
    /// 0 state, 1 question, 2 option.
    kind: Vec<u8>,
    /// Option index within the row for option tokens.
    opt: Vec<u32>,
    /// Request index within the row (tokens of different requests never
    /// attend to each other).
    seg: Vec<u32>,
    /// Options per request in the row.
    n_opts: Vec<usize>,
}

impl Row {
    fn append(&mut self, other: &Row) {
        let o = self.n_opts.iter().sum::<usize>() as u32;
        let g = self.n_opts.len() as u32;
        self.ids.extend_from_slice(&other.ids);
        self.pos.extend_from_slice(&other.pos);
        self.kind.extend_from_slice(&other.kind);
        self.opt.extend(other.opt.iter().map(|x| x + o));
        self.seg.extend(other.seg.iter().map(|x| x + g));
        self.n_opts.extend_from_slice(&other.n_opts);
    }
}

/// On the CPU, short requests are packed into one row of up to this many
/// tokens: candle runs a batched matmul as one GEMM call (each with its own
/// thread fan-out) per batch × head, which made padded batches of short rows
/// slower than packed ones with its built-in GEMM (q8: 350 vs 195
/// ms/decision on a loaded machine).
const CPU_PACK_TOKENS: usize = 512;

/// Default cap on tokens per forward pass. Activations (and on Metal the
/// pooled buffers candle keeps for them) scale with it; decisions are short,
/// so a few thousand tokens already fill the GPU.
pub const DECIDER_BATCH_TOKENS: usize = 1024;

impl DeciderNano {
    /// The default checkpoint ([`DECIDER_NANO_REPO`]).
    pub fn new(cfg: EngineConfig) -> Result<Self> {
        Self::with_repo(DECIDER_NANO_REPO, cfg)
    }

    /// A checkpoint by repo id (or local directory). Reads `config.json`
    /// now; weights load on first use.
    pub fn with_repo(repo: &str, engine: EngineConfig) -> Result<Self> {
        let cfg: Config =
            serde_json::from_slice(&std::fs::read(hub::resolve_hf_file(repo, "config.json")?)?)?;
        if cfg.hidden_size % cfg.num_attention_heads != 0 {
            return Err(Error::Config("hidden_size is not a multiple of the head count".into()));
        }
        // `kleos_decider.json` next to the weights names the weights version
        // that shipped calibration defaults are keyed on (e.g. a fine-tune);
        // the published checkpoint is "base", any other directory "custom".
        let weights_version = hub::resolve_hf_file(repo, "kleos_decider.json")
            .ok()
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|v| v.get("weights_version").and_then(|x| x.as_str()).map(str::to_string))
            .unwrap_or_else(|| if repo == DECIDER_NANO_REPO { "base".into() } else { "custom".into() });
        Ok(Self {
            repo: repo.to_string(),
            weights_version,
            cfg,
            engine,
            weights: Mutex::new(None),
            run: Mutex::new(()),
        })
    }

    /// `(device, bytes of weights)` once loaded.
    pub fn loaded_info(&self) -> Option<(&'static str, usize)> {
        self.weights
            .lock()
            .unwrap()
            .as_ref()
            .map(|w| (device_name(&w.device), w.bytes))
    }

    /// The device the weights live on, once loaded.
    pub fn device_name(&self) -> Option<String> {
        self.loaded_info().map(|(d, _)| d.to_string())
    }

    /// Weights version: `base` for the published checkpoint, the
    /// `weights_version` of a `kleos_decider.json` beside local weights, else
    /// `custom`.
    pub fn weights_version(&self) -> &str {
        &self.weights_version
    }

    /// The engine settings this model was built with.
    pub fn engine(&self) -> &EngineConfig {
        &self.engine
    }

    fn weights(&self) -> Result<Arc<Weights>> {
        let mut g = self.weights.lock().unwrap();
        if let Some(w) = g.as_ref() {
            return Ok(w.clone());
        }
        let w = Arc::new(self.load()?);
        *g = Some(w.clone());
        Ok(w)
    }

    fn load(&self) -> Result<Weights> {
        let device = self.engine.device.device()?;
        let mut tokenizer =
            Tokenizer::from_file(hub::resolve_hf_file(&self.repo, "tokenizer.json")?)
                .map_err(tok_err)?;
        tokenizer.with_truncation(None).map_err(tok_err)?;
        tokenizer.with_padding(None);
        let sep = self
            .cfg
            .sep_token_id
            .or_else(|| tokenizer.token_to_id("[SEP]"))
            .ok_or_else(|| Error::Config("no [SEP] token".into()))?;
        let pad = self
            .cfg
            .pad_token_id
            .or_else(|| tokenizer.token_to_id("[PAD]"))
            .unwrap_or(0);
        let st = TensorFile::open(&hub::resolve_hf_file(&self.repo, "model.safetensors")?)?;
        let mut ld = Loader {
            st: &st,
            device: &device,
            precision: self.engine.precision,
            bytes: 0,
        };
        let dtype = ld.act_dtype();
        let p = "backbone.";
        let embed = ld.embedding(&format!("{p}embeddings.tok_embeddings.weight"))?;
        let norm = |ld: &mut Loader, n: &str| -> Result<Tensor> { Ok(ld.f32(n)?.to_dtype(dtype)?) };
        let embed_norm = norm(&mut ld, &format!("{p}embeddings.norm.weight"))?;
        let mut layers = Vec::with_capacity(self.cfg.num_hidden_layers);
        for i in 0..self.cfg.num_hidden_layers {
            let n = |s: &str| format!("{p}layers.{i}.{s}");
            layers.push(Layer {
                attn_norm: if i == 0 {
                    None
                } else {
                    Some(norm(&mut ld, &n("attn_norm.weight"))?)
                },
                wqkv: ld.linear(&n("attn.Wqkv.weight"))?,
                wo: ld.linear(&n("attn.Wo.weight"))?,
                mlp_norm: norm(&mut ld, &n("mlp_norm.weight"))?,
                wi: ld.linear(&n("mlp.Wi.weight"))?,
                mlp_wo: ld.linear(&n("mlp.Wo.weight"))?,
                sliding: self.cfg.sliding(i),
            });
        }
        let final_norm = norm(&mut ld, &format!("{p}final_norm.weight"))?;
        let scorer_w = ld.f32("scorer.weight")?;
        let scorer_b = ld.f32("scorer.bias")?.to_device(&Device::Cpu)?.to_vec1::<f32>()?[0];
        let zero_bias = Tensor::zeros(self.cfg.hidden_size, dtype, &device)?;
        Ok(Weights {
            embed,
            embed_norm,
            layers,
            final_norm,
            scorer_w,
            scorer_b,
            zero_bias,
            device: device.clone(),
            dtype,
            tokenizer,
            sep,
            pad,
            bytes: ld.bytes,
        })
    }

    fn tokenize(&self, w: &Weights, r: &DecideRequest) -> Result<Row> {
        if r.options.len() < 2 {
            return Err(Error::Config("a decision needs at least two options".into()));
        }
        let state = if r.state.is_empty() { "(empty)" } else { r.state.as_str() };
        let tok = &w.tokenizer;
        let p = tok.encode(state, true).map_err(tok_err)?.get_ids().to_vec();
        let q = tok
            .encode(format!("Question: {}\nOption:", r.question), false)
            .map_err(tok_err)?
            .get_ids()
            .to_vec();
        let mut ids = Vec::new();
        let mut pos = Vec::new();
        let mut kind = Vec::new();
        let mut opt = Vec::new();
        let (np, nq) = (p.len(), q.len());
        ids.extend_from_slice(&p);
        pos.extend(0..np as u32);
        kind.extend(std::iter::repeat(0u8).take(np));
        opt.extend(std::iter::repeat(0u32).take(np));
        ids.extend_from_slice(&q);
        pos.extend(np as u32..(np + nq) as u32);
        kind.extend(std::iter::repeat(1u8).take(nq));
        opt.extend(std::iter::repeat(0u32).take(nq));
        let mut longest = 0;
        for (k, o) in r.options.iter().enumerate() {
            let mut ob = tok
                .encode(format!(" {o}"), false)
                .map_err(tok_err)?
                .get_ids()
                .to_vec();
            ob.push(w.sep);
            let start = (np + nq) as u32;
            longest = longest.max(ob.len());
            pos.extend(start..start + ob.len() as u32);
            kind.extend(std::iter::repeat(2u8).take(ob.len()));
            opt.extend(std::iter::repeat(k as u32).take(ob.len()));
            ids.extend(ob);
        }
        if np + nq + longest > MAX_PAIR_TOKENS {
            return Err(Error::Config(format!(
                "a (state, question, option) pair needs {} tokens > {MAX_PAIR_TOKENS}; inputs are not truncated",
                np + nq + longest
            )));
        }
        let seg = vec![0u32; ids.len()];
        Ok(Row {
            ids,
            pos,
            kind,
            opt,
            seg,
            n_opts: vec![r.options.len()],
        })
    }

    fn decide_inner(&self, reqs: &[DecideRequest], with_features: bool) -> Result<Vec<DecideAnswer>> {
        let w = self.weights()?;
        let rows = reqs
            .iter()
            .map(|r| self.tokenize(&w, r))
            .collect::<Result<Vec<_>>>()?;
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].ids.len());
        // (row, the requests it holds, in row order)
        let mut lines: Vec<(Row, Vec<usize>)> = vec![];
        // Pack on the CPU when candle's own GEMM runs the matmuls (quantised
        // weights, or no Accelerate BLAS); padded rows otherwise (packing adds
        // attention work across requests that the mask then discards).
        let quant = matches!(self.engine.precision, Precision::Q8 | Precision::Q4);
        let pack = w.device.is_cpu() && (quant || !cfg!(feature = "accelerate"));
        for &k in &order {
            match lines.last_mut() {
                Some((row, idx))
                    if pack && row.ids.len() + rows[k].ids.len() <= CPU_PACK_TOKENS =>
                {
                    row.append(&rows[k]);
                    idx.push(k);
                }
                _ => lines.push((rows[k].clone(), vec![k])),
            }
        }
        lines.sort_by_key(|(r, _)| r.ids.len());
        let mut out: Vec<Option<DecideAnswer>> = vec![None; rows.len()];
        let budget = self.batch_tokens();
        let _run = self.run.lock().unwrap();
        let mut i = 0;
        while i < lines.len() {
            let mut j = i + 1;
            while j < lines.len() {
                let l = lines[j].0.ids.len();
                let n = j - i + 1;
                if n * l > budget || n * l * l > MAX_MASK_ELEMS {
                    break;
                }
                j += 1;
            }
            let batch: Vec<&Row> = lines[i..j].iter().map(|(r, _)| r).collect();
            let answers = self.forward(&w, &batch, with_features)?;
            let idx = lines[i..j].iter().flat_map(|(_, idx)| idx.iter().copied());
            for (k, a) in idx.zip(answers) {
                out[k] = Some(a);
            }
            i = j;
        }
        // Every intermediate of the forward passes is dropped by now; one more
        // sync lets candle's Metal allocator free its pooled activation
        // buffers (it drops unused ones only on synchronize), so the
        // footprint between calls is the weights, not the last batch's
        // working set.
        w.device.synchronize()?;
        Ok(out.into_iter().map(|a| a.unwrap()).collect())
    }

    /// Tokens per forward pass: the engine's budget, capped at
    /// [`DECIDER_BATCH_TOKENS`] (`KLEOS_DECIDER_BATCH_TOKENS` overrides).
    fn batch_tokens(&self) -> usize {
        let cap = std::env::var("KLEOS_DECIDER_BATCH_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DECIDER_BATCH_TOKENS);
        self.engine.max_batch_tokens.min(cap).max(1)
    }

    fn forward(&self, w: &Weights, rows: &[&Row], with_features: bool) -> Result<Vec<DecideAnswer>> {
        let cfg = &self.cfg;
        let dev = &w.device;
        let dt = w.dtype;
        let b = rows.len();
        let l = rows.iter().map(|r| r.ids.len()).max().unwrap_or(1);
        let (h, hd) = (cfg.num_attention_heads, cfg.hidden_size / cfg.num_attention_heads);
        let hidden = cfg.hidden_size;
        let window = cfg.window() as i64;

        let mut ids = vec![w.pad; b * l];
        let mut pos = vec![0u32; b * l];
        // additive masks [b, 1, l, l]: tree mask, and tree mask ∧ window
        let neg = -1.0e9f32;
        let mut full = vec![neg; b * l * l];
        let mut slide = vec![neg; b * l * l];
        for (r, row) in rows.iter().enumerate() {
            let n = row.ids.len();
            ids[r * l..r * l + n].copy_from_slice(&row.ids);
            pos[r * l..r * l + n].copy_from_slice(&row.pos);
            let base = r * l * l;
            for qi in 0..l {
                for kj in 0..l {
                    let allow = if qi >= n || kj >= n {
                        qi == kj
                    } else {
                        let (a, c) = (row.kind[qi], row.kind[kj]);
                        qi == kj
                            || (row.seg[qi] == row.seg[kj]
                                && (c == 0
                                    || (c == 1 && a >= 1)
                                    || (c == 2 && a == 2 && row.opt[qi] == row.opt[kj])))
                    };
                    if allow {
                        full[base + qi * l + kj] = 0.0;
                        let near = qi >= n
                            || kj >= n
                            || (row.pos[qi] as i64 - row.pos[kj] as i64).abs() <= window;
                        if near {
                            slide[base + qi * l + kj] = 0.0;
                        }
                    }
                }
            }
        }
        let ids_t = Tensor::from_vec(ids, (b, l), dev)?;
        let full_m = Tensor::from_vec(full, (b, 1, l, l), dev)?;
        let slide_m = Tensor::from_vec(slide, (b, 1, l, l), dev)?;
        let pos_t = Tensor::from_vec(pos, b * l, dev)?;
        let rope = |theta: f64| -> Result<(Tensor, Tensor)> {
            let half = hd / 2;
            let inv: Vec<f32> = (0..half)
                .map(|i| 1.0 / theta.powf(2.0 * i as f64 / hd as f64) as f32)
                .collect();
            let inv = Tensor::from_vec(inv, (1, half), dev)?;
            let p = pos_t.to_dtype(DType::F32)?.reshape((b * l, 1))?;
            let f = p.broadcast_mul(&inv)?;
            Ok((
                f.cos()?.reshape((b, l, half))?.to_dtype(dt)?.contiguous()?,
                f.sin()?.reshape((b, l, half))?.to_dtype(dt)?.contiguous()?,
            ))
        };
        let (cos_g, sin_g) = rope(cfg.theta(false))?;
        let (cos_l, sin_l) = rope(cfg.theta(true))?;
        let eps = cfg.norm_eps as f32;
        let ln = |x: &Tensor, wt: &Tensor| -> Result<Tensor> {
            Ok(candle_nn::ops::layer_norm(&x.contiguous()?, wt, &w.zero_bias, eps)?)
        };
        // hidden states kept 2-d `[b*l, hidden]`: one large GEMM per linear
        // layer instead of one per row
        let mut x = ln(
            &w.embed.lookup(&ids_t)?.to_dtype(dt)?.reshape((b * l, hidden))?,
            &w.embed_norm,
        )?;
        let scale = 1.0 / (hd as f64).sqrt();
        for layer in &w.layers {
            let (mask, cos, sin) = if layer.sliding {
                (&slide_m, &cos_l, &sin_l)
            } else {
                (&full_m, &cos_g, &sin_g)
            };
            let xn = match &layer.attn_norm {
                Some(n) => ln(&x, n)?,
                None => x.clone(),
            };
            let qkv = layer.wqkv.forward(&xn)?.reshape((b, l, 3, h, hd))?;
            let q = qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
            let k = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
            let v = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
            let q = candle_nn::rotary_emb::rope(&q, cos, sin)?;
            let k = candle_nn::rotary_emb::rope(&k, cos, sin)?;
            let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?
                .to_dtype(DType::F32)?
                .broadcast_add(mask)?;
            let att = candle_nn::ops::softmax_last_dim(&att)?.to_dtype(dt)?;
            let a = att.matmul(&v)?.transpose(1, 2)?.reshape((b * l, hidden))?;
            x = (x + layer.wo.forward(&a)?)?;
            let xn = ln(&x, &layer.mlp_norm)?;
            let hi = layer.wi.forward(&xn)?;
            let inter = hi.dim(D::Minus1)? / 2;
            let input = hi.narrow(D::Minus1, 0, inter)?;
            let gate = hi.narrow(D::Minus1, inter, inter)?;
            let m = (input.gelu_erf()? * gate)?;
            x = (x + layer.mlp_wo.forward(&m)?)?;
        }
        let x = ln(&x, &w.final_norm)?.to_dtype(DType::F32)?;
        // mean over each option block: a pooling matrix [n_options, b*l]
        let n_opts: usize = rows.iter().map(|r| r.n_opts.iter().sum::<usize>()).sum();
        let mut pool = vec![0f32; n_opts * b * l];
        let mut base = 0;
        for (r, row) in rows.iter().enumerate() {
            let row_opts: usize = row.n_opts.iter().sum();
            let mut counts = vec![0usize; row_opts];
            for t in 0..row.ids.len() {
                if row.kind[t] == 2 {
                    counts[row.opt[t] as usize] += 1;
                }
            }
            for t in 0..row.ids.len() {
                if row.kind[t] == 2 {
                    let o = row.opt[t] as usize;
                    pool[(base + o) * b * l + r * l + t] = 1.0 / counts[o] as f32;
                }
            }
            base += row_opts;
        }
        let pool = Tensor::from_vec(pool, (n_opts, b * l), dev)?;
        let pooled = pool.matmul(&x)?;
        let z = (pooled.matmul(&w.scorer_w.t()?)? + w.scorer_b as f64)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let feats: Vec<Vec<f32>> = if with_features {
            pooled.to_vec2()?
        } else {
            vec![]
        };
        dev.synchronize()?;
        let mut out = Vec::with_capacity(b);
        let mut base = 0;
        for &n in rows.iter().flat_map(|r| r.n_opts.iter()) {
            out.push(DecideAnswer {
                logits: z[base..base + n].to_vec(),
                features: if with_features {
                    feats[base..base + n].to_vec()
                } else {
                    vec![]
                },
            });
            base += n;
        }
        Ok(out)
    }
}

impl Decider for DeciderNano {
    fn model_id(&self) -> &str {
        &self.repo
    }
    fn max_batch(&self) -> usize {
        32
    }
    fn decide(&self, reqs: &[DecideRequest], with_features: bool) -> Result<Vec<DecideAnswer>> {
        if reqs.is_empty() {
            return Ok(vec![]);
        }
        with_pool(|| self.decide_inner(reqs, with_features))
    }
    fn is_loaded(&self) -> bool {
        self.weights.lock().unwrap().is_some()
    }
    fn unload(&self) {
        with_pool(|| *self.weights.lock().unwrap() = None);
    }
}
