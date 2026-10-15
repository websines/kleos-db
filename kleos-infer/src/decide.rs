/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Decision models ("System One" style, UPGRADE_PLAN D5): one forward pass
//! scores a fixed set of options for a question about a state, instead of
//! generating text.
//!
//! This module is only the backend contract. Question templates, calibration
//! (thresholds, per-option bias, temperature) and the CozoScript surface live
//! in kleos-core, so every backend is driven the same way.

use crate::Result;

/// One question about one state.
#[derive(Debug, Clone, PartialEq)]
pub struct DecideRequest {
    /// What the question is about: free text, or JSON serialised by the caller.
    pub state: String,
    /// The question or instruction.
    pub question: String,
    /// The candidate answers, in order (2 or more). For a yes/no question pass
    /// the backend's own wording, e.g. `["false: ...", "true: ..."]`.
    pub options: Vec<String>,
}

/// The backend's answer to one [`DecideRequest`].
#[derive(Debug, Clone, PartialEq)]
pub struct DecideAnswer {
    /// Raw scores (pre-softmax, after the backend's own temperature if it has
    /// one), one per option, in request order.
    pub logits: Vec<f32>,
    /// Optional per-option feature vectors (e.g. the hidden state the decision
    /// head reads), for fitting a light probe on labelled examples. Empty when
    /// not requested or not supported.
    pub features: Vec<Vec<f32>>,
}

/// A loaded (or lazily loading) decision model.
pub trait Decider: Send + Sync {
    /// Model id, e.g. `LiquidAI/d1-omni-600M`.
    fn model_id(&self) -> &str;
    /// Largest batch the caller should send at once.
    fn max_batch(&self) -> usize {
        16
    }
    /// Score a batch of requests. `with_features` asks for
    /// [`DecideAnswer::features`]; backends that cannot provide them return
    /// empty vectors.
    fn decide(&self, reqs: &[DecideRequest], with_features: bool) -> Result<Vec<DecideAnswer>>;
    /// Whether weights are resident.
    fn is_loaded(&self) -> bool {
        true
    }
    /// Drop weights; the next call reloads them.
    fn unload(&self) {}
}

/// Softmax over one answer's logits.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}
