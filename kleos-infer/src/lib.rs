/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Single-forward-pass inference for kleos (UPGRADE_PLAN D2).
//!
//! Encoders only: one forward pass per input, no autoregressive decoding, no
//! KV cache, no sampling. Today that means text embedders; cross-encoder
//! re-rankers and decision heads reuse the same pieces later.
//!
//! * Backend: [candle](https://github.com/huggingface/candle), CPU everywhere,
//!   Apple GPU with the `metal` feature (falls back to CPU when no GPU).
//! * Weights: f32 / f16 / bf16, or int8 / int4 block-quantised linear layers
//!   ([`Precision`]), quantised from the published f32/bf16 checkpoints at load.
//! * Models load lazily from the local Hugging Face cache on first use and can
//!   be dropped with `unload()`. Nothing is downloaded by this crate.
//! * Memory: activations are bounded by [`EngineConfig::max_batch_tokens`]
//!   (inputs are length-sorted and packed under that budget), and the GPU
//!   buffer pool is released after every batch, so steady-state memory is the
//!   weights plus one batch.
//! * Thread-safe: embedders are `Send + Sync`; forward passes on one model are
//!   serialised (they saturate the device anyway).
//!
//! Models:
//! * [`StaticEmbedder`]: Model2Vec static embeddings (e.g.
//!   `minishlab/potion-base-8M`), a table lookup and a mean, ~0.05 ms/text.
//! * [`GemmaEmbedder`]: `google/embeddinggemma-300m` (Gemma 3 bidirectional
//!   encoder, mean pooling, two dense projections, Matryoshka output).

mod device;
mod error;
mod gemma;
mod hub;
mod linear;
mod static_embed;

pub use device::DevicePref;
pub use error::{Error, Result};
pub use gemma::{GemmaEmbedder, GemmaModel, GEMMA_DOCUMENT_PROMPT, GEMMA_QUERY_PROMPT};
pub use hub::{hf_cache_dir, resolve_hf_file};
pub use static_embed::StaticEmbedder;

/// Weight precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// 32-bit floats (reference).
    F32,
    /// 16-bit floats.
    F16,
    /// bfloat16.
    BF16,
    /// int8 blocks of 32 with an f16 scale (GGML Q8_0) for every linear layer
    /// and the token table; activations in f32.
    Q8,
    /// int4 blocks of 32 (GGML Q4_0) for linear layers, Q8 token table.
    Q4,
}

impl Precision {
    /// Parse `f32`, `f16`, `bf16`, `q8`, `q4` (case-insensitive).
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "f32" | "fp32" => Precision::F32,
            "f16" | "fp16" => Precision::F16,
            "bf16" => Precision::BF16,
            "q8" | "q8_0" | "int8" => Precision::Q8,
            "q4" | "q4_0" | "int4" => Precision::Q4,
            other => return Err(Error::Config(format!("unknown precision '{other}'"))),
        })
    }
    /// Short name, as accepted by [`Precision::parse`].
    pub fn name(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F16 => "f16",
            Precision::BF16 => "bf16",
            Precision::Q8 => "q8",
            Precision::Q4 => "q4",
        }
    }
}

/// Engine settings shared by all models.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Where to run.
    pub device: DevicePref,
    /// Weight precision (ignored by static embedders, which are always f32).
    pub precision: Precision,
    /// Upper bound on `batch × padded length` per forward pass. Bounds the
    /// activation memory (attention scores grow with `batch × length²`).
    pub max_batch_tokens: usize,
    /// Inputs are truncated to this many tokens.
    pub max_seq_len: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            device: DevicePref::Auto,
            precision: Precision::Q8,
            max_batch_tokens: 8192,
            max_seq_len: 512,
        }
    }
}

/// A text embedder. One vector of length [`TextEmbedder::dim`] per input, in
/// input order.
pub trait TextEmbedder: Send + Sync {
    /// Embed a batch of texts.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    /// Output width.
    fn dim(&self) -> usize;
    /// Identity of everything that determines the output (repo, precision,
    /// prompt, output width); usable as a cache key.
    fn model_id(&self) -> String;
    /// Preferred number of texts per call.
    fn max_batch(&self) -> usize {
        64
    }
}

pub(crate) fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}
