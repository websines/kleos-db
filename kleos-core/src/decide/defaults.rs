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

pub(super) fn calibration(t: Template, f: Family) -> Calibration {
    const ADAPT: &str = "adapt-60 v1";
    const NONE: &str = "default (no adapt data)";
    match (t, f) {
        (Template::Supersedes, Family::D1) => cal(&[], 0.351, 1.01, ADAPT),
        (Template::Contradicts, Family::D1) => cal(&[], 0.351, 1.01, ADAPT),
        (Template::Duplicate, Family::D1) => cal(&[], 0.5, 1.01, ADAPT),
        (Template::Relation, Family::D1) => cal(&[0.0, 0.0, -0.5], 0.0, 1.01, ADAPT),
        (Template::Supersedes, Family::Nano) => cal(&[], 0.5, 1.01, ADAPT),
        (Template::Contradicts, Family::Nano) => cal(&[], 0.5, 1.01, ADAPT),
        (Template::Duplicate, Family::Nano) => cal(&[], 0.5, 1.01, ADAPT),
        (Template::Relation, Family::Nano) => cal(&[], 0.0, 1.01, ADAPT),
        (Template::SameEntity | Template::Important, _) => cal(&[], 0.5, 0.9, NONE),
    }
}
