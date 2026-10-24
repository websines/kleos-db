/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The built-in GLiNER2.5 extractor (`kleos-infer`, cargo feature `infer`).
//!
//! Prompted exactly as kleos's fine-tune was trained: the four bare labels
//! [`GLINER_LABELS`] in that order, no descriptions, gliner2's word splitter.
//! Texts longer than 384 words are read in 384-word windows overlapping by
//! 64 (gliner2's long-text defaults). Spans come back at the decode floor
//! [`GLINER_FLOOR`] with their raw scores; thresholds are the memory layer's.

use std::sync::Arc;

use miette::{miette, Result};

use kleos_infer::{DevicePref, EngineConfig, Gliner, Precision, GLINER_BASE_REPO, GLINER_REPO};

use crate::bail;
use crate::extract::{EntityExtractor, ExtractSpan, ExtractorSpec};

/// Labels of kleos's GLiNER fine-tune, in training order.
pub const GLINER_LABELS: [&str; 4] = ["person", "location", "organization", "relative"];
/// Spans scoring below this are not returned (thresholds sit above it).
pub const GLINER_FLOOR: f32 = 0.1;
/// Window and overlap, in words, for long texts.
pub const GLINER_WINDOW: (usize, usize) = (384, 64);

/// Default weight precision: f16 (docs/specs/memory-api.md, "Automatic
/// entities": f16 and q8 change no span on the gold set; f16 is the
/// reference-faithful default).
pub const GLINER_DEFAULT_PRECISION: Precision = Precision::F16;

struct GlinerExtractor {
    g: Arc<Gliner>,
}

impl EntityExtractor for GlinerExtractor {
    fn model_id(&self) -> String {
        self.g.model_id().to_string()
    }
    fn version(&self) -> String {
        format!("gliner2.5:{}:{}", self.g.model_id(), self.g.engine().precision.name())
    }
    fn labels(&self) -> Vec<String> {
        GLINER_LABELS.iter().map(|s| s.to_string()).collect()
    }
    fn extract(&self, texts: &[&str]) -> Result<Vec<Vec<ExtractSpan>>> {
        let found = self
            .g
            .extract_windows(texts, &GLINER_LABELS, GLINER_FLOOR, GLINER_WINDOW.0, GLINER_WINDOW.1)
            .map_err(|e| miette!("gliner: {e}"))?;
        Ok(found
            .into_iter()
            .map(|spans| {
                spans
                    .into_iter()
                    .map(|s| ExtractSpan {
                        label: GLINER_LABELS[s.label].to_string(),
                        start: s.start,
                        end: s.end,
                        text: s.text,
                        score: s.score,
                    })
                    .collect()
            })
            .collect())
    }
    fn extract_labels(&self, texts: &[&str], extra: &[String]) -> Result<Vec<Vec<ExtractSpan>>> {
        if extra.is_empty() {
            return self.extract(texts);
        }
        // the trained labels first, in training order, then the extra ones
        let mut labels: Vec<&str> = GLINER_LABELS.to_vec();
        labels.extend(extra.iter().map(|x| x.as_str()).filter(|x| !GLINER_LABELS.contains(x)));
        let found = self
            .g
            .extract_windows(texts, &labels, GLINER_FLOOR, GLINER_WINDOW.0, GLINER_WINDOW.1)
            .map_err(|e| miette!("gliner: {e}"))?;
        Ok(found
            .into_iter()
            .map(|spans| {
                spans
                    .into_iter()
                    .map(|s| ExtractSpan { label: labels[s.label].to_string(), start: s.start, end: s.end, text: s.text, score: s.score })
                    .collect()
            })
            .collect())
    }
    fn is_loaded(&self) -> Option<bool> {
        Some(self.g.is_loaded())
    }
    fn unload(&self) {
        self.g.unload()
    }
}

/// Build the extractor `spec` describes.
pub(crate) fn build(name: &str, spec: &ExtractorSpec) -> Result<Arc<dyn EntityExtractor>> {
    let mut engine = EngineConfig::default();
    if let Some(d) = &spec.device {
        engine.device = DevicePref::parse(d).map_err(|e| miette!("{e}"))?;
    }
    engine.precision = match &spec.precision {
        Some(p) => Precision::parse(p).map_err(|e| miette!("{e}"))?,
        None => GLINER_DEFAULT_PRECISION,
    };
    let repo = match (&spec.path, spec.model.as_deref().map(str::trim)) {
        (Some(p), _) => {
            let dir = std::path::Path::new(p);
            if !dir.join("model.safetensors").is_file() || !dir.join("config.json").is_file() {
                bail!("cannot enable extractor '{name}': '{p}' is not a GLiNER2.5 checkpoint directory (needs config.json, tokenizer.json and model.safetensors)");
            }
            p.clone()
        }
        (None, None | Some("gliner") | Some("gliner2.5")) => std::env::var("KLEOS_GLINER_MODEL")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| GLINER_REPO.to_string()),
        (None, Some("gliner-base" | "gliner2.5-base")) => GLINER_BASE_REPO.to_string(),
        (None, Some(m)) => m.to_string(),
    };
    let key = format!(
        "gliner|{repo}|{}|{:?}|{}|{}",
        engine.precision.name(),
        engine.device,
        engine.max_seq_len,
        engine.max_batch_tokens
    );
    let g = kleos_infer::budget::shared(&key, || Gliner::with_repo(&repo, engine).map(Arc::new))
        .map_err(|e| miette!("cannot enable extractor '{name}' ({repo}): {e}; download it once with `hf download {repo}`"))?;
    Ok(Arc::new(GlinerExtractor { g }))
}
