/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Built-in embedders backed by `kleos-infer` (cargo feature `infer`). Model
//! files are read from the local Hugging Face cache (`HF_HOME`); fetch them
//! once with e.g. `hf download google/embeddinggemma-300m`.
//!
//! ```no_run
//! use kleos::embed::builtin::{register_embeddinggemma, register_model2vec, EngineConfig};
//! register_embeddinggemma("gemma", EngineConfig::default(), None).unwrap();
//! register_model2vec("potion", "minishlab/potion-base-8M").unwrap();
//! // CozoScript: embed('gemma', text) for documents, embed('gemma_query', q) for queries
//! ```

use std::sync::Arc;

use miette::{miette, Result};

pub use kleos_infer::{DevicePref, EngineConfig, Precision, TextEmbedder};
use kleos_infer::{GemmaModel, StaticEmbedder, GEMMA_DOCUMENT_PROMPT, GEMMA_QUERY_PROMPT};

use super::{register_embedder, Embedder};

struct Adapter<E>(E);

impl<E: TextEmbedder> Embedder for Adapter<E> {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.0.embed(texts).map_err(|e| miette!("{e}"))
    }
    fn dim(&self) -> usize {
        self.0.dim()
    }
    fn model_id(&self) -> Option<String> {
        Some(self.0.model_id())
    }
    fn max_batch(&self) -> usize {
        self.0.max_batch()
    }
}

/// Register any `kleos-infer` embedder under `name`.
pub fn register_text_embedder(name: &str, e: impl TextEmbedder + 'static) -> Result<()> {
    register_embedder(name, Arc::new(Adapter(e)))
}

/// Register `google/embeddinggemma-300m` as `name` (document prompt) and
/// `{name}_query` (query prompt), sharing one lazily loaded model. `dim`
/// truncates the Matryoshka output (768, 512, 256 or 128).
pub fn register_embeddinggemma(name: &str, engine: EngineConfig, dim: Option<usize>) -> Result<()> {
    let model = GemmaModel::new("google/embeddinggemma-300m", engine).map_err(|e| miette!("{e}"))?;
    let doc = model
        .embedder(GEMMA_DOCUMENT_PROMPT, dim)
        .map_err(|e| miette!("{e}"))?;
    let query = model
        .embedder(GEMMA_QUERY_PROMPT, dim)
        .map_err(|e| miette!("{e}"))?;
    register_text_embedder(name, doc)?;
    register_text_embedder(&format!("{name}_query"), query)
}

/// Register a Model2Vec static model (e.g. `minishlab/potion-base-8M`) as `name`.
pub fn register_model2vec(name: &str, repo: &str) -> Result<()> {
    register_text_embedder(name, StaticEmbedder::new(repo).map_err(|e| miette!("{e}"))?)
}
