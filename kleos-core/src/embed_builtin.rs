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
//!
//! At runtime the same models are switched on and off with
//! `::embedder enable` / `::embedder disable` (see [`super::enable_embedder`]).

use std::sync::Arc;

use miette::{bail, miette, Result};

pub use kleos_infer::{DevicePref, EngineConfig, Precision, TextEmbedder};
use kleos_infer::{
    GemmaEmbedder, GemmaModel, StaticEmbedder, GEMMA_DOCUMENT_PROMPT, GEMMA_QUERY_PROMPT,
};

use super::{register_embedder, register_with_info, Embedder, EmbedderInfo, EmbedderSpec};

/// The EmbeddingGemma repo used by `model: 'embeddinggemma'`.
pub const EMBEDDINGGEMMA_REPO: &str = "google/embeddinggemma-300m";

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

struct GemmaAdapter(GemmaEmbedder);

impl Embedder for GemmaAdapter {
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
    fn is_loaded(&self) -> Option<bool> {
        Some(self.0.model().is_loaded())
    }
    fn device(&self) -> Option<String> {
        self.0.model().loaded_info().map(|(d, _)| d.to_string())
    }
    fn unload(&self) {
        self.0.model().unload()
    }
}

struct StaticAdapter(StaticEmbedder);

impl Embedder for StaticAdapter {
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
    fn is_loaded(&self) -> Option<bool> {
        Some(self.0.is_loaded())
    }
    fn device(&self) -> Option<String> {
        Some("cpu".to_string())
    }
}

/// Register any `kleos-infer` embedder under `name`.
pub fn register_text_embedder(name: &str, e: impl TextEmbedder + 'static) -> Result<()> {
    register_embedder(name, Arc::new(Adapter(e)))
}

fn device_name(d: DevicePref) -> &'static str {
    match d {
        DevicePref::Auto => "auto",
        DevicePref::Cpu => "cpu",
        DevicePref::Metal => "metal",
    }
}

fn register_gemma(
    name: &str,
    repo: &str,
    engine: EngineConfig,
    dim: Option<usize>,
    spec: EmbedderSpec,
) -> Result<()> {
    let model = GemmaModel::new(repo, engine.clone()).map_err(|e| miette!("{e}"))?;
    let doc = model
        .embedder(GEMMA_DOCUMENT_PROMPT, dim)
        .map_err(|e| miette!("{e}"))?;
    let query = model
        .embedder(GEMMA_QUERY_PROMPT, dim)
        .map_err(|e| miette!("{e}"))?;
    let qname = format!("{name}_query");
    let info = EmbedderInfo {
        kind: "gemma",
        model: repo.to_string(),
        precision: Some(engine.precision.name().to_string()),
        device_pref: Some(device_name(engine.device).to_string()),
        spec: Some(spec),
        twin: None,
        twin_of: None,
    };
    register_with_info(
        &qname,
        Arc::new(GemmaAdapter(query)),
        EmbedderInfo {
            twin_of: Some(name.to_string()),
            spec: None,
            ..info.clone()
        },
    )?;
    register_with_info(
        name,
        Arc::new(GemmaAdapter(doc)),
        EmbedderInfo {
            twin: Some(qname),
            ..info
        },
    )
}

/// Register `google/embeddinggemma-300m` as `name` (document prompt) and
/// `{name}_query` (query prompt), sharing one lazily loaded model. `dim`
/// truncates the Matryoshka output (768, 512, 256 or 128).
pub fn register_embeddinggemma(name: &str, engine: EngineConfig, dim: Option<usize>) -> Result<()> {
    let spec = EmbedderSpec {
        model: Some(EMBEDDINGGEMMA_REPO.to_string()),
        precision: Some(engine.precision.name().to_string()),
        device: Some(device_name(engine.device).to_string()),
        dim,
    };
    register_gemma(name, EMBEDDINGGEMMA_REPO, engine, dim, spec)
}

/// Register a Model2Vec static model (e.g. `minishlab/potion-base-8M`) as `name`.
pub fn register_model2vec(name: &str, repo: &str) -> Result<()> {
    let e = StaticEmbedder::new(repo).map_err(|e| miette!("{e}"))?;
    register_with_info(
        name,
        Arc::new(StaticAdapter(e)),
        EmbedderInfo {
            kind: "model2vec",
            model: repo.to_string(),
            precision: Some("f32".to_string()),
            device_pref: Some("cpu".to_string()),
            spec: Some(EmbedderSpec::model(repo)),
            twin: None,
            twin_of: None,
        },
    )
}

/// Which built-in model a spec's `model` names.
enum Builtin {
    Gemma(String),
    Model2Vec(String),
}

fn resolve_model(model: Option<&str>) -> Result<Builtin> {
    let m = model.unwrap_or("embeddinggemma").trim();
    let lower = m.to_ascii_lowercase();
    Ok(match lower.as_str() {
        "embeddinggemma" | "embeddinggemma-300m" | "gemma" => {
            Builtin::Gemma(EMBEDDINGGEMMA_REPO.to_string())
        }
        _ if lower.contains("embeddinggemma") && m.contains('/') => Builtin::Gemma(m.to_string()),
        _ if m.contains('/') => Builtin::Model2Vec(m.to_string()),
        _ if lower.starts_with("potion-") => Builtin::Model2Vec(format!("minishlab/{m}")),
        _ => {
            bail!(
                "unknown built-in embedder model '{m}': expected 'embeddinggemma', \
                 'potion-base-8M' or a Model2Vec repo such as 'minishlab/potion-base-8M'"
            );
        }
    })
}

/// Build and register the model `spec` describes (see [`EmbedderSpec`]).
pub(crate) fn enable_from_spec(name: &str, spec: EmbedderSpec) -> Result<()> {
    match resolve_model(spec.model.as_deref())? {
        Builtin::Gemma(repo) => {
            let mut engine = EngineConfig::default();
            if let Some(p) = &spec.precision {
                engine.precision = Precision::parse(p).map_err(|e| miette!("{e}"))?;
            }
            if let Some(d) = &spec.device {
                engine.device = DevicePref::parse(d).map_err(|e| miette!("{e}"))?;
            }
            if let Some(d) = spec.dim {
                if d == 0 || d > 768 {
                    bail!("embeddinggemma's dim must be in 1..=768 (768, 512, 256 or 128), got {d}");
                }
            }
            let dim = spec.dim.filter(|d| *d != 768);
            let stored = EmbedderSpec {
                model: Some(repo.clone()),
                precision: Some(engine.precision.name().to_string()),
                device: Some(device_name(engine.device).to_string()),
                dim: spec.dim,
            };
            register_gemma(name, &repo, engine, dim, stored)
        }
        Builtin::Model2Vec(repo) => {
            if let Some(p) = &spec.precision {
                if !p.eq_ignore_ascii_case("f32") {
                    bail!("Model2Vec models are always f32; drop `precision: '{p}'`");
                }
            }
            if let Some(d) = &spec.device {
                if !matches!(d.to_ascii_lowercase().as_str(), "auto" | "cpu") {
                    bail!("Model2Vec models run on the CPU; drop `device: '{d}'`");
                }
            }
            let e = StaticEmbedder::new(&repo).map_err(|e| miette!("{e}"))?;
            if let Some(d) = spec.dim {
                if d != e.dim() {
                    bail!("'{repo}' has a fixed width of {}, not {d}", e.dim());
                }
            }
            register_with_info(
                name,
                Arc::new(StaticAdapter(e)),
                EmbedderInfo {
                    kind: "model2vec",
                    model: repo.clone(),
                    precision: Some("f32".to_string()),
                    device_pref: Some("cpu".to_string()),
                    spec: Some(EmbedderSpec::model(repo)),
                    twin: None,
                    twin_of: None,
                },
            )
        }
    }
}
