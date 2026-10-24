/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/// Errors from model loading and inference.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A model file is not in the local Hugging Face cache.
    #[error("model file not found: {0}")]
    NotFound(String),
    /// Downloading a model file failed.
    #[error("model download failed: {0}")]
    Download(String),
    /// Bad configuration or model metadata.
    #[error("configuration error: {0}")]
    Config(String),
    /// Tensor backend failure.
    #[error("tensor backend: {0}")]
    Candle(#[from] candle_core::Error),
    /// Tokenizer failure.
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    /// I/O failure.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// A forward pass failed (reported to every caller batched with it).
    #[error("inference: {0}")]
    Model(String),
    /// JSON parse failure.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result alias.
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn tok_err(e: impl std::fmt::Display) -> Error {
    Error::Tokenizer(e.to_string())
}
