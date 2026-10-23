/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Shipped calibration defaults, fitted on the 60 held-out adapt items
//! (scripts/infer-bench/d1/eval_set_adapt.py) by
//! `cargo run --release --example decide_eval -- --fit-defaults`.
//! α = 0.02 false-auto rate (empirical; 60 items cannot certify 2%, see
//! docs/specs/decide.md). `same_entity` and `important` have no adapt data.

use super::{Calibration, Family, Template};

fn cal(bias: &[f64], threshold: f64, tau_auto: f64, source: &str) -> Calibration {
    Calibration {
        bias: bias.to_vec(),
        temperature: 1.0,
        threshold,
        tau_auto,
        source: source.to_string(),
    }
}

/// Defaults per weights version; `None` or an unknown version falls back to
/// the published checkpoint's.
pub(super) fn calibration_for_weights(t: Template, f: Family, weights: Option<&str>) -> Calibration {
    match (weights, f) {
        (Some(NANO_TUNE_V1), Family::Nano) => nano_tune_v1(t),
        _ => calibration(t, f),
    }
}

/// Weights version of the kleos fine-tune of decider-nano
/// (scripts/infer-bench/decide/train/, `kleos_decider.json` next to the
/// weights; docs/specs/decide.md "Fine-tuned weights").
pub const NANO_TUNE_V1: &str = "nano-tune-v1";

/// nano-tune-v1, fitted on the 60 adapt items at f32 (`decide_eval
/// --model-path … --fit-defaults`). The tuned model is sharply peaked, so the
/// fitted thresholds sit near 1 for supersedes and contradicts.
/// `duplicate`: adapt is perfectly separated, so the risk rule put τ at the
/// threshold (0.091); on the eval set that auto band let 3 elaborations
/// ("Karen Liu also manages the design team") through as duplicates, so
/// duplicates are never auto-applied with these weights (τ 1.01). That choice
/// was made after seeing the eval result and is disclosed in the spec.
fn nano_tune_v1(t: Template) -> Calibration {
    const SRC: &str = "adapt-60 nano-tune-v1";
    match t {
        Template::Supersedes => cal(&[], 0.997, 0.997, SRC),
        Template::Contradicts => cal(&[], 0.954, 0.954, SRC),
        Template::Duplicate => cal(&[], 0.091, 1.01, SRC),
        Template::Relation => cal(&[0.0, 0.0, -0.25], 0.000, 1.000, SRC),
        Template::SameEntity | Template::Important => cal(&[], 0.5, 0.9, "default (no adapt data)"),
    }
}

pub(super) fn calibration(t: Template, f: Family) -> Calibration {
    const ADAPT: &str = "adapt-60 v1";
    const NONE: &str = "default (no adapt data)";
    match (t, f) {
        // d1-omni: recorded MPS f16 answers (scripts/infer-bench/decide/d1_record.py)
        (Template::Supersedes, Family::D1) => cal(&[], 0.370, 0.722, ADAPT),
        (Template::Contradicts, Family::D1) => cal(&[], 0.232, 0.407, ADAPT),
        (Template::Duplicate, Family::D1) => cal(&[], 0.500, 0.664, ADAPT),
        (Template::Relation, Family::D1) => cal(&[0.0, 0.0, -0.5], 0.000, 0.936, ADAPT),
        // decider-nano f32 through kleos-infer. Zero-shot supersedes and
        // contradicts never reach a 2% false-auto rate on adapt, so they never
        // auto-apply (tau 1.01); fit a probe (`::decider fit`) to get an auto band.
        (Template::Supersedes, Family::Nano) => cal(&[], 0.246, 1.01, ADAPT),
        (Template::Contradicts, Family::Nano) => cal(&[], 0.500, 1.01, ADAPT),
        (Template::Duplicate, Family::Nano) => cal(&[], 0.500, 0.539, ADAPT),
        (Template::Relation, Family::Nano) => cal(&[0.0, 0.0, -0.5], 0.000, 0.778, ADAPT),
        (Template::SameEntity | Template::Important, _) => cal(&[], 0.5, 0.9, NONE),
        (_, Family::Custom(_)) => Calibration::default(),
    }
}
