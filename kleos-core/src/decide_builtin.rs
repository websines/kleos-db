/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Built-in deciders backed by `kleos-infer` (cargo feature `infer`). Model
//! files live in the Hugging Face cache (`HF_HOME`). Enabling a model
//! downloads missing files there (the decider-nano default is about 300 MB);
//! set `HF_HUB_OFFLINE=1` to forbid that and pre-seed the cache with e.g.
//! `hf download Mazino0/kleos-decider-nano`.
//!
//! ```no_run
//! use kleos::decide::{enable_decider, DeciderSpec};
//! enable_decider("nano", DeciderSpec::model("decider-nano")).unwrap();
//! // CozoScript: decide_supersedes('nano', old, new)
//! ```

use std::sync::Arc;

use miette::{miette, Result};

use crate::bail;

pub use kleos_infer::{DevicePref, EngineConfig, Precision};
use kleos_infer::{DeciderNano, DECIDER_NANO_BASE_REPO, DECIDER_NANO_REPO};

use super::{register_with_info, DecideAnswer, DecideRequest, Decider, DeciderInfo, DeciderSpec, Family};

/// Reference precision of decider-nano (what calibrations were fitted at).
pub const DECIDER_NANO_PRECISION: Precision = Precision::F32;

/// Default weight precision of decider-nano for a device: int8 (Q8_0) on
/// both the CPU and the Apple GPU. Measured against f32 on 922 decisions
/// (docs/specs/decide.md, "Precision and memory"): 1 flip, every AUROC within
/// 0.002, a quarter of the f32 weight memory, and no slower than f16 on
/// Metal. f16 (1 flip, max |Δp| 0.008) is the higher-fidelity option at about
/// 130 MB more.
pub fn default_precision(device: DevicePref) -> Precision {
    let _ = device;
    Precision::Q8
}

/// Wraps any `kleos_infer::Decider`.
pub struct InferDecider<D> {
    inner: D,
    device: fn(&D) -> Option<String>,
    weights: fn(&D) -> Option<String>,
}

impl<D: kleos_infer::Decider> Decider for InferDecider<D> {
    fn model_id(&self) -> String {
        self.inner.model_id().to_string()
    }
    fn max_batch(&self) -> usize {
        self.inner.max_batch()
    }
    fn decide(&self, reqs: &[DecideRequest], with_features: bool) -> Result<Vec<DecideAnswer>> {
        let r: Vec<kleos_infer::DecideRequest> = reqs
            .iter()
            .map(|r| kleos_infer::DecideRequest {
                state: r.state.clone(),
                question: r.question.clone(),
                options: r.options.clone(),
            })
            .collect();
        Ok(self
            .inner
            .decide(&r, with_features)
            .map_err(|e| miette!("{e}"))?
            .into_iter()
            .map(|a| DecideAnswer {
                logits: a.logits,
                features: a.features,
            })
            .collect())
    }
    fn is_loaded(&self) -> Option<bool> {
        Some(self.inner.is_loaded())
    }
    fn device(&self) -> Option<String> {
        (self.device)(&self.inner)
    }
    fn unload(&self) {
        self.inner.unload()
    }
    fn weights_version(&self) -> Option<String> {
        (self.weights)(&self.inner)
    }
}

/// Register any `kleos-infer` decider under `name`, speaking `family`.
pub fn register_infer_decider<D: kleos_infer::Decider + 'static>(name: &str, d: D, family: Family) -> Result<()> {
    super::register_decider(
        name,
        Arc::new(InferDecider {
            inner: d,
            device: |_| None,
            weights: |_| None,
        }),
        family,
    )
}

fn device_name(d: DevicePref) -> &'static str {
    match d {
        DevicePref::Auto => "auto",
        DevicePref::Cpu => "cpu",
        DevicePref::Metal => "metal",
    }
}

enum Builtin {
    Nano(String),
    D1,
}

fn resolve(model: Option<&str>) -> Result<Builtin> {
    let m = model.unwrap_or("decider-nano").trim();
    Ok(match m.to_ascii_lowercase().as_str() {
        "decider-nano" | "nano" | "kleos-decider-nano" | "mazino0/kleos-decider-nano" => {
            Builtin::Nano(DECIDER_NANO_REPO.to_string())
        }
        "decider-nano-base" | "nano-base" | "tasksource-decider-nano" | "tasksource/tasksource-decider-nano" => {
            Builtin::Nano(DECIDER_NANO_BASE_REPO.to_string())
        }
        "d1" | "d1-omni" | "d1-omni-600m" | "liquidai/d1-omni-600m" => Builtin::D1,
        _ if m.starts_with('/') || m.starts_with('.') => Builtin::Nano(m.to_string()),
        other => bail!(
            "unknown decider model '{other}': built-in models are 'decider-nano' (Mazino0/kleos-decider-nano), 'decider-nano-base' (tasksource/tasksource-decider-nano) and 'd1-omni' (LiquidAI/d1-omni-600M)"
        ),
    })
}

/// Build and register the built-in model `spec` describes.
pub(crate) fn enable_from_spec(name: &str, spec: DeciderSpec) -> Result<()> {
    let mut engine = EngineConfig::default();
    if let Some(d) = &spec.device {
        engine.device = DevicePref::parse(d).map_err(|e| miette!("{e}"))?;
    }
    engine.precision = match &spec.precision {
        Some(p) => Precision::parse(p).map_err(|e| miette!("{e}"))?,
        None => default_precision(engine.device),
    };
    let target = match &spec.path {
        Some(path) => {
            if !matches!(resolve(spec.model.as_deref())?, Builtin::Nano(_)) {
                bail!("cannot enable decider '{name}': `path` holds decider-nano weights, but `model` names another model");
            }
            let dir = std::path::Path::new(path);
            if !dir.join("model.safetensors").is_file() || !dir.join("config.json").is_file() {
                bail!(
                    "cannot enable decider '{name}': '{path}' is not a decider-nano weights directory (needs config.json, tokenizer.json and model.safetensors)"
                );
            }
            Builtin::Nano(path.clone())
        }
        None => resolve(spec.model.as_deref())?,
    };
    match target {
        Builtin::Nano(repo) => {
            let key = format!(
                "decider-nano|{repo}|{}|{:?}|{}|{}",
                engine.precision.name(),
                engine.device,
                engine.max_seq_len,
                engine.max_batch_tokens
            );
            let d = kleos_infer::budget::shared(&key, || DeciderNano::with_repo(&repo, engine.clone()).map(Arc::new))
                .map_err(|e| {
                    miette!("cannot enable decider '{name}' ({repo}): {e}; download it once with `hf download {repo}`")
                })?;
            register_with_info(
                name,
                Arc::new(InferDecider {
                    inner: d,
                    device: |d: &Arc<DeciderNano>| d.device_name(),
                    weights: |d: &Arc<DeciderNano>| Some(d.weights_version().to_string()),
                }),
                DeciderInfo {
                    kind: "decider-nano",
                    family: Family::Nano,
                    model: repo.clone(),
                    precision: Some(engine.precision.name().to_string()),
                    device_pref: Some(device_name(engine.device).to_string()),
                    spec: Some(DeciderSpec {
                        path: spec.path.clone(),
                        model: if spec.path.is_some() { spec.model.clone() } else { Some(repo) },
                        precision: Some(engine.precision.name().to_string()),
                        device: Some(device_name(engine.device).to_string()),
                    }),
                },
            )
        }
        Builtin::D1 => bail!(
            "cannot enable decider '{name}': the d1-omni backend is not built into this kleos yet (kleos-infer d1 port pending); \
             use 'decider-nano', or register a d1 backend from Rust with kleos::decide::register_decider(name, d, Family::D1)"
        ),
    }
}
