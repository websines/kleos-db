/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Entity extractors: models that find the names a text mentions, so
//! `remember` can link memories to entities without the caller passing them
//! (docs/specs/memory-api.md, "Automatic entities").
//!
//! An extractor has a fixed label set (the one its model was trained on, in
//! training order) and returns spans with raw scores; what counts as an
//! entity (thresholds, case rules, relatives, the speaker) is the memory
//! layer's policy (`memory_extract.rs`). The built-in extractor is GLiNER2.5
//! (`kleos-infer`, cargo feature `infer`):
//!
//! ```no_run
//! use kleos::extract::{enable_extractor, ExtractorSpec};
//! enable_extractor("gliner", ExtractorSpec::default()).unwrap();
//! // ::memory init {extractor: 'gliner'}
//! ```
//!
//! Like embedders and deciders, extractors are process-wide.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use lazy_static::lazy_static;
use miette::{miette, Result};
use serde_derive::{Deserialize, Serialize};

use crate::bail;

/// One span an extractor found.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractSpan {
    /// One of the extractor's labels.
    pub label: String,
    /// Byte offsets into the text.
    pub start: usize,
    /// Exclusive byte end.
    pub end: usize,
    /// The span's text.
    pub text: String,
    /// Raw model score (not a calibrated probability).
    pub score: f32,
}

/// A model that finds entity spans.
pub trait EntityExtractor: Send + Sync {
    /// Model id (repo or directory).
    fn model_id(&self) -> String;
    /// Identity of everything that determines the spans (model, weights,
    /// precision); stored with every extracted mention.
    fn version(&self) -> String;
    /// The labels, in the order the model was trained with.
    fn labels(&self) -> Vec<String>;
    /// Spans of each text whose score is at least the extractor's decode
    /// floor; texts of any length.
    fn extract(&self, texts: &[&str]) -> Result<Vec<Vec<ExtractSpan>>>;
    /// [`EntityExtractor::extract`] with `extra` labels after the trained
    /// ones (an open-label model such as GLiNER reads them zero-shot; e.g.
    /// the classes of a domain ontology). The default ignores `extra`.
    fn extract_labels(&self, texts: &[&str], extra: &[String]) -> Result<Vec<Vec<ExtractSpan>>> {
        let _ = extra;
        self.extract(texts)
    }
    /// Whether the weights are in memory (`None`: not applicable).
    fn is_loaded(&self) -> Option<bool> {
        None
    }
    /// Drop the weights.
    fn unload(&self) {}
}

/// What to enable.
///
/// * `model`: `gliner` (default; `$KLEOS_GLINER_MODEL` when set, else
///   `fastino/gliner2.5-base-v1`) or a Hugging Face repo id.
/// * `path`: a local checkpoint directory (GLiNER2.5 boundary layout), e.g.
///   a fine-tune.
/// * `precision`: `f16` (default), `f32`, `q8`.
/// * `device`: `auto` (default), `cpu`, `metal`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExtractorSpec {
    /// Model name or repo.
    pub model: Option<String>,
    /// Local checkpoint directory.
    pub path: Option<String>,
    /// Weight precision.
    pub precision: Option<String>,
    /// Device preference.
    pub device: Option<String>,
}

/// An extractor as listed by [`extractors`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractorStatus {
    /// Registered name.
    pub name: String,
    /// Model id.
    pub model: String,
    /// Version stored with its mentions.
    pub version: String,
    /// Labels in training order.
    pub labels: Vec<String>,
    /// Whether it is enabled (disabled ones keep their configuration).
    pub enabled: bool,
    /// Whether its weights are in memory.
    pub loaded: Option<bool>,
}

struct Registered {
    ex: Arc<dyn EntityExtractor>,
    spec: Option<ExtractorSpec>,
}

lazy_static! {
    static ref EXTRACTORS: RwLock<BTreeMap<String, Registered>> = RwLock::new(BTreeMap::new());
    static ref PARKED: RwLock<BTreeMap<String, Registered>> = RwLock::new(BTreeMap::new());
}

/// Register an extractor under `name` (replacing one of that name).
pub fn register_extractor(name: &str, ex: Arc<dyn EntityExtractor>) -> Result<()> {
    register(name, ex, None)
}

fn register(name: &str, ex: Arc<dyn EntityExtractor>, spec: Option<ExtractorSpec>) -> Result<()> {
    if name.is_empty() {
        bail!("an extractor name must not be empty");
    }
    let old = EXTRACTORS
        .write()
        .map_err(|_| miette!("extractor registry poisoned"))?
        .insert(name.to_string(), Registered { ex, spec });
    if let Ok(mut p) = PARKED.write() {
        p.remove(name);
    }
    if let Some(o) = old {
        o.ex.unload();
    }
    Ok(())
}

/// Turn an extractor on (process-wide). With an empty spec and a name
/// disabled earlier, the previous configuration comes back.
pub fn enable_extractor(name: &str, spec: ExtractorSpec) -> Result<()> {
    if spec == ExtractorSpec::default() {
        let parked = PARKED.write().map_err(|_| miette!("extractor registry poisoned"))?.remove(name);
        if let Some(p) = parked {
            return register(name, p.ex, p.spec);
        }
        if EXTRACTORS.read().map(|r| r.contains_key(name)).unwrap_or(false) {
            return Ok(());
        }
    }
    enable_builtin(name, spec)
}

#[cfg(feature = "infer")]
fn enable_builtin(name: &str, spec: ExtractorSpec) -> Result<()> {
    let ex = crate::extract_builtin::build(name, &spec)?;
    register(name, ex, Some(spec))
}

#[cfg(not(feature = "infer"))]
fn enable_builtin(name: &str, spec: ExtractorSpec) -> Result<()> {
    let _ = spec;
    bail!(
        "cannot enable extractor '{name}': kleos was built without the `infer` cargo feature, so it has no built-in extractors \
         (rebuild with `--features infer`, or register one from Rust with kleos::extract::register_extractor)"
    )
}

/// Turn an extractor off (unloading it). Returns false when it was already
/// disabled.
pub fn disable_extractor(name: &str) -> Result<bool> {
    let removed = EXTRACTORS.write().map_err(|_| miette!("extractor registry poisoned"))?.remove(name);
    match removed {
        Some(r) => {
            r.ex.unload();
            PARKED.write().map_err(|_| miette!("extractor registry poisoned"))?.insert(name.to_string(), r);
            Ok(true)
        }
        None if PARKED.read().map(|p| p.contains_key(name)).unwrap_or(false) => Ok(false),
        None => bail!("extractor '{name}' is not enabled"),
    }
}

/// The enabled extractor `name`.
pub fn get_extractor(name: &str) -> Option<Arc<dyn EntityExtractor>> {
    EXTRACTORS.read().ok()?.get(name).map(|r| r.ex.clone())
}

/// Every enabled and disabled extractor.
pub fn extractors() -> Vec<ExtractorStatus> {
    let mut out = vec![];
    for (reg, enabled) in [(&*EXTRACTORS, true), (&*PARKED, false)] {
        if let Ok(r) = reg.read() {
            for (name, x) in r.iter() {
                out.push(ExtractorStatus {
                    name: name.clone(),
                    model: x.ex.model_id(),
                    version: x.ex.version(),
                    labels: x.ex.labels(),
                    enabled,
                    loaded: x.ex.is_loaded(),
                });
            }
        }
    }
    out
}
