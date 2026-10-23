/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Model2Vec static embeddings: tokenize, look up one row per token, average,
//! normalise. No neural network at inference, so it runs on the CPU in
//! microseconds and is the near-zero-cost fallback embedder.

use std::sync::OnceLock;

use safetensors::{Dtype, SafeTensors};
use tokenizers::Tokenizer;

use crate::error::tok_err;
use crate::{hub, l2_normalize, Error, Result, TextEmbedder};

struct Loaded {
    tokenizer: Tokenizer,
    table: Vec<f32>,
    rows: usize,
    unk: Option<u32>,
}

/// A Model2Vec model (e.g. `minishlab/potion-base-8M`), loaded on first use.
pub struct StaticEmbedder {
    repo: String,
    dim: usize,
    normalize: bool,
    max_tokens: usize,
    loaded: OnceLock<Loaded>,
}

impl StaticEmbedder {
    /// Reads `config.json` of `repo` (HF cache, fetched if missing) now;
    /// weights load on first use.
    pub fn new(repo: &str) -> Result<Self> {
        hub::ensure_hf_files(repo, &["config.json", "tokenizer.json", "model.safetensors"], &[])?;
        let cfg: serde_json::Value =
            serde_json::from_slice(&std::fs::read(hub::resolve_hf_file(repo, "config.json")?)?)?;
        let dim = cfg["hidden_dim"]
            .as_u64()
            .ok_or_else(|| Error::Config(format!("{repo}: config.json has no hidden_dim")))?
            as usize;
        Ok(Self {
            repo: repo.to_string(),
            dim,
            normalize: cfg["normalize"].as_bool().unwrap_or(true),
            max_tokens: 512,
            loaded: OnceLock::new(),
        })
    }

    /// Whether the embedding table is in memory.
    pub fn is_loaded(&self) -> bool {
        self.loaded.get().is_some()
    }

    fn load(&self) -> Result<&Loaded> {
        if let Some(l) = self.loaded.get() {
            return Ok(l);
        }
        let tokenizer = Tokenizer::from_file(hub::resolve_hf_file(&self.repo, "tokenizer.json")?)
            .map_err(tok_err)?;
        let bytes = std::fs::read(hub::resolve_hf_file(&self.repo, "model.safetensors")?)?;
        let st = SafeTensors::deserialize(&bytes).map_err(|e| Error::Config(e.to_string()))?;
        let t = st
            .tensor("embeddings")
            .map_err(|e| Error::Config(e.to_string()))?;
        if t.dtype() != Dtype::F32 || t.shape().len() != 2 || t.shape()[1] != self.dim {
            return Err(Error::Config(format!(
                "{}: expected an f32 [vocab, {}] 'embeddings' tensor, got {:?} {:?}",
                self.repo,
                self.dim,
                t.dtype(),
                t.shape()
            )));
        }
        let table: Vec<f32> = t
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let rows = t.shape()[0];
        let unk = tokenizer.token_to_id("[UNK]");
        let _ = self.loaded.set(Loaded {
            tokenizer,
            table,
            rows,
            unk,
        });
        Ok(self.loaded.get().unwrap())
    }
}

impl TextEmbedder for StaticEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let m = self.load()?;
        let encs = m
            .tokenizer
            .encode_batch(texts.to_vec(), false)
            .map_err(tok_err)?;
        let mut out = Vec::with_capacity(texts.len());
        for enc in encs {
            let mut v = vec![0f32; self.dim];
            let mut n = 0usize;
            for &id in enc
                .get_ids()
                .iter()
                .filter(|&&id| Some(id) != m.unk && (id as usize) < m.rows)
                .take(self.max_tokens)
            {
                let row = &m.table[id as usize * self.dim..(id as usize + 1) * self.dim];
                v.iter_mut().zip(row).for_each(|(a, b)| *a += b);
                n += 1;
            }
            if n > 0 {
                v.iter_mut().for_each(|x| *x /= n as f32);
            }
            if self.normalize {
                l2_normalize(&mut v);
            }
            out.push(v);
        }
        Ok(out)
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn model_id(&self) -> String {
        format!("model2vec:{}", self.repo)
    }
    fn max_batch(&self) -> usize {
        1024
    }
}
