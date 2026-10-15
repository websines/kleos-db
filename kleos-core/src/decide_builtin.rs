/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Built-in deciders backed by `kleos-infer` (cargo feature `infer`). Model
//! files are read from the local Hugging Face cache (`HF_HOME`); fetch them
//! once with e.g. `hf download tasksource/tasksource-decider-nano`.
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
use kleos_infer::{DeciderNano, DECIDER_NANO_REPO};

use super::{register_with_info, DecideAnswer, DecideRequest, Decider, DeciderInfo, DeciderSpec, Family};

/// Default weight precision of decider-nano (see docs/specs/decide.md).
pub const DECIDER_NANO_PRECISION: Precision = Precision::F32;

/// Wraps any `kleos_infer::Decider`.
pub struct InferDecider<D> {
    inner: D,
    device: fn(&D) -> Option<String>,
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
}

/// Register any `kleos-infer` decider under `name`, speaking `family`.
pub fn register_infer_decider<D: kleos_infer::Decider + 'static>(name: &str, d: D, family: Family) -> Result<()> {
    super::register_decider(
        name,
        Arc::new(InferDecider {
            inner: d,
            device: |_| None,
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
        "decider-nano" | "nano" | "tasksource-decider-nano" => Builtin::Nano(DECIDER_NANO_REPO.to_string()),
        "tasksource/tasksource-decider-nano" => Builtin::Nano(m.to_string()),
        "d1" | "d1-omni" | "d1-omni-600m" | "liquidai/d1-omni-600m" => Builtin::D1,
        _ if m.starts_with('/') || m.starts_with('.') => Builtin::Nano(m.to_string()),
        other => bail!(
            "unknown decider model '{other}': built-in models are 'decider-nano' (tasksource/tasksource-decider-nano) and 'd1-omni' (LiquidAI/d1-omni-600M)"
        ),
    })
}

/// Build and register the built-in model `spec` describes.
pub(crate) fn enable_from_spec(name: &str, spec: DeciderSpec) -> Result<()> {
    let mut engine = EngineConfig {
        precision: DECIDER_NANO_PRECISION,
        ..EngineConfig::default()
    };
    if let Some(p) = &spec.precision {
        engine.precision = Precision::parse(p).map_err(|e| miette!("{e}"))?;
    }
    if let Some(d) = &spec.device {
        engine.device = DevicePref::parse(d).map_err(|e| miette!("{e}"))?;
    }
    match resolve(spec.model.as_deref())? {
        Builtin::Nano(repo) => {
            let d = DeciderNano::with_repo(&repo, engine.clone()).map_err(|e| {
                miette!("cannot enable decider '{name}' ({repo}): {e}; download it once with `hf download {repo}`")
            })?;
            register_with_info(
                name,
                Arc::new(InferDecider {
                    inner: d,
                    device: |d: &DeciderNano| d.device_name(),
                }),
                DeciderInfo {
                    kind: "decider-nano",
                    family: Family::Nano,
                    model: repo.clone(),
                    precision: Some(engine.precision.name().to_string()),
                    device_pref: Some(device_name(engine.device).to_string()),
                    spec: Some(DeciderSpec {
                        model: Some(repo),
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
