/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Built-in embedders backed by `kleos-infer` (cargo feature `infer`). Model
//! files are read from the local Hugging Face cache (`HF_HOME`); fetch them
//! once with e.g. `hf download google/embeddinggemma-2`, or let enabling
//! the model download them (`HF_HUB_OFFLINE=1` forbids downloads). The
//! default embedder is EmbeddingGemma 2 (Apache-2.0, not gated); EmbeddingGemma
//! v1 (`embeddinggemma-300m`, gated: needs its licence accepted and `HF_TOKEN`)
//! stays selectable for stores whose vectors were made with it.
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
    Egemma2Embedder, Egemma2Model, GemmaEmbedder, GemmaModel, StaticEmbedder, EGEMMA2_DOCUMENT_PROMPT,
    EGEMMA2_QUERY_PROMPT, EGEMMA2_REPO, GEMMA_DOCUMENT_PROMPT, GEMMA_QUERY_PROMPT,
};

use super::{register_embedder, register_with_info, Embedder, EmbedderInfo, EmbedderSpec};

/// The EmbeddingGemma repo used by `model: 'embeddinggemma'` (and `gemma`):
/// EmbeddingGemma 2, Apache-2.0, not gated.
pub const EMBEDDINGGEMMA_REPO: &str = EGEMMA2_REPO;

/// EmbeddingGemma v1 (`model: 'embeddinggemma-300m'` or `gemma-v1`), gated
/// under the Gemma Terms of Use.
pub const EMBEDDINGGEMMA_V1_REPO: &str = "google/embeddinggemma-300m";

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

struct Gemma2Adapter(Egemma2Embedder);

impl Embedder for Gemma2Adapter {
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
    // one model object per (repo, engine) in the process: several stores or
    // enables share its weights, gate and batcher
    let key = format!(
        "{}|{repo}|{}|{:?}|{}|{}",
        if is_v2(repo) { "gemma2" } else { "gemma" },
        engine.precision.name(),
        engine.device,
        engine.max_seq_len,
        engine.max_batch_tokens
    );
    let (doc, query, kind): (Arc<dyn Embedder>, Arc<dyn Embedder>, &'static str) = if is_v2(repo) {
        let model = kleos_infer::budget::shared(&key, || Egemma2Model::new(repo, engine.clone()))
            .map_err(|e| miette!("{e}"))?;
        let doc = model.embedder(EGEMMA2_DOCUMENT_PROMPT, dim).map_err(|e| miette!("{e}"))?;
        let query = model.embedder(EGEMMA2_QUERY_PROMPT, dim).map_err(|e| miette!("{e}"))?;
        (Arc::new(Gemma2Adapter(doc)), Arc::new(Gemma2Adapter(query)), "gemma2")
    } else {
        let model = kleos_infer::budget::shared(&key, || GemmaModel::new(repo, engine.clone()))
            .map_err(|e| miette!("{e}"))?;
        let doc = model.embedder(GEMMA_DOCUMENT_PROMPT, dim).map_err(|e| miette!("{e}"))?;
        let query = model.embedder(GEMMA_QUERY_PROMPT, dim).map_err(|e| miette!("{e}"))?;
        (Arc::new(GemmaAdapter(doc)), Arc::new(GemmaAdapter(query)), "gemma")
    };
    let qname = format!("{name}_query");
    let info = EmbedderInfo {
        kind,
        model: repo.to_string(),
        precision: Some(engine.precision.name().to_string()),
        device_pref: Some(device_name(engine.device).to_string()),
        spec: Some(spec),
        twin: None,
        twin_of: None,
    };
    register_with_info(
        &qname,
        query,
        EmbedderInfo {
            twin_of: Some(name.to_string()),
            spec: None,
            ..info.clone()
        },
    )?;
    register_with_info(
        name,
        doc,
        EmbedderInfo {
            twin: Some(qname),
            ..info
        },
    )
}

/// Whether `repo` is an EmbeddingGemma 2 checkpoint (else v1).
fn is_v2(repo: &str) -> bool {
    let r = repo.to_ascii_lowercase();
    r.contains("embeddinggemma-2") || r.contains("embeddinggemma2")
}

/// Register EmbeddingGemma 2 (`google/embeddinggemma-2`, the default) as
/// `name` (document prompt) and `{name}_query` (query prompt), sharing one
/// lazily loaded model. `dim` truncates the Matryoshka output (768, 512, 256
/// or 128).
pub fn register_embeddinggemma(name: &str, engine: EngineConfig, dim: Option<usize>) -> Result<()> {
    register_gemma_repo(name, EMBEDDINGGEMMA_REPO, engine, dim)
}

/// Register EmbeddingGemma v1 (`google/embeddinggemma-300m`, gated) like
/// [`register_embeddinggemma`], for stores whose vectors were made with it.
pub fn register_embeddinggemma_v1(name: &str, engine: EngineConfig, dim: Option<usize>) -> Result<()> {
    register_gemma_repo(name, EMBEDDINGGEMMA_V1_REPO, engine, dim)
}

fn register_gemma_repo(name: &str, repo: &str, engine: EngineConfig, dim: Option<usize>) -> Result<()> {
    let spec = EmbedderSpec {
        model: Some(repo.to_string()),
        precision: Some(engine.precision.name().to_string()),
        device: Some(device_name(engine.device).to_string()),
        dim,
    };
    register_gemma(name, repo, engine, dim, spec)
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
        "embeddinggemma" | "gemma" | "embeddinggemma-2" | "embeddinggemma2" | "gemma2" => {
            Builtin::Gemma(EMBEDDINGGEMMA_REPO.to_string())
        }
        "embeddinggemma-300m" | "embeddinggemma-v1" | "embeddinggemma1" | "gemma-v1" | "gemma1" => {
            Builtin::Gemma(EMBEDDINGGEMMA_V1_REPO.to_string())
        }
        _ if lower.contains("embeddinggemma") && m.contains('/') => Builtin::Gemma(m.to_string()),
        _ if m.contains('/') => Builtin::Model2Vec(m.to_string()),
        _ if lower.starts_with("potion-") => Builtin::Model2Vec(format!("minishlab/{m}")),
        _ => {
            bail!(
                "unknown built-in embedder model '{m}': expected 'embeddinggemma' (EmbeddingGemma 2), \
                 'embeddinggemma-300m' (v1), 'potion-base-8M' or a Model2Vec repo such as 'minishlab/potion-base-8M'"
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
