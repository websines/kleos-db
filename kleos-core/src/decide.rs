/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! `decide()`: small decision models attached to the database (kleos fork,
//! UPGRADE_PLAN D5; docs/specs/decide.md).
//!
//! A [`Decider`] scores a fixed set of options for one question about one
//! state in a single forward pass. This module turns that raw contract into
//! memory decisions that callers can act on:
//!
//! * **Fixed, versioned templates** ([`Template`]) per decision type and per
//!   backend dialect ([`Family`]). Small decision models are very sensitive
//!   to phrasing, so callers never write the question: they pick a template
//!   (`supersedes`, `same_entity`, `duplicate`, `contradicts`, `relation`,
//!   `important`) and pass the texts.
//! * **Pair-symmetric scoring**: for symmetric relations (`same_entity`,
//!   `duplicate`, `contradicts`) both orders are scored and averaged.
//! * **Calibration** ([`Calibration`]): per-option log-bias, a label
//!   threshold and a risk-controlled auto-apply threshold, with defaults
//!   fitted on the held-out adapt set and shipped here.
//! * **Bands** ([`Band`]): every [`Decision`] says whether it is safe to act
//!   on automatically (`auto`), should be applied as tentative or escalated
//!   (`tentative`), or calls for no action (`none`).
//! * **Probes** ([`Probe`]): an L2 logistic-regression head over the
//!   backend's decision features, fitted on labelled pairs and used in place
//!   of the zero-shot readout for that model and template.
//!
//! The registry of deciders is process-wide, like the embedder registry.
//! Fitted probes and calibrations are persisted per database (relation
//! `kleos_decider_calibration`) and loaded into the process-wide table when a
//! database opens.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, RwLock};

use lazy_static::lazy_static;
use miette::{miette, Result};
use serde_derive::{Deserialize, Serialize};

use crate::bail;
use crate::data::expr::Op;
use crate::data::value::DataValue;

// ------------------------------------------------------------------ contract

/// One question about one state (mirrors `kleos_infer::DecideRequest`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DecideRequest {
    /// What the question is about.
    pub state: String,
    /// The question or instruction.
    pub question: String,
    /// Candidate answers in order, in the backend's own wording.
    pub options: Vec<String>,
}

/// A backend's answer to one [`DecideRequest`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DecideAnswer {
    /// Raw scores, one per option (pre-softmax).
    pub logits: Vec<f32>,
    /// Optional per-option feature vectors (what the decision head reads).
    pub features: Vec<Vec<f32>>,
}

/// A decision model. Built-in ones come from `kleos-infer` (cargo feature
/// `infer`); anything else can be registered with [`register_decider`].
pub trait Decider: Send + Sync {
    /// Stable identity of the model and everything that changes its output.
    /// Probes and calibrations are keyed on it.
    fn model_id(&self) -> String;
    /// Largest batch per call.
    fn max_batch(&self) -> usize {
        16
    }
    /// Score a batch. `with_features` asks for [`DecideAnswer::features`].
    fn decide(&self, reqs: &[DecideRequest], with_features: bool) -> Result<Vec<DecideAnswer>>;
    /// Whether weights are resident (`None`: not tracked).
    fn is_loaded(&self) -> Option<bool> {
        None
    }
    /// Device the weights live on once loaded.
    fn device(&self) -> Option<String> {
        None
    }
    /// Release the weights.
    fn unload(&self) {}
}

/// Built-in deciders (cargo feature `infer`).
#[cfg(feature = "infer")]
#[path = "decide_builtin.rs"]
pub mod builtin;

// ------------------------------------------------------------------ templates

/// The prompt dialect a backend was trained on. Templates are validated per
/// family; a custom decider declares which family it speaks. A new backend
/// with its own prompts registers a family with [`register_family`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Family {
    /// LiquidAI d1-omni (Decision Index schema, option markers).
    D1,
    /// tasksource-decider-nano (cross-encoder, `Question: ..\nOption: ..`).
    Nano,
    /// A family registered with [`register_family`].
    Custom(&'static str),
}

/// The prompts of a registered family: a total function from template to
/// spec, the prompt version, and the default calibration per template.
#[derive(Clone, Copy)]
pub struct FamilyTemplates {
    /// The fixed prompt of each template.
    pub spec: fn(Template) -> TemplateSpec,
    /// Prompt version (bump on any prompt change).
    pub version: u32,
    /// Shipped calibration per template.
    pub calibration: fn(Template) -> Calibration,
}

lazy_static! {
    static ref FAMILIES: RwLock<BTreeMap<&'static str, FamilyTemplates>> = RwLock::new(BTreeMap::new());
}

/// Register (or replace) a template family for a new backend, e.g. a
/// llama.cpp decision model with its own validated prompts. Returns the
/// [`Family`] to pass to [`register_decider`].
pub fn register_family(name: &str, templates: FamilyTemplates) -> Result<Family> {
    let lower = name.trim().to_ascii_lowercase();
    if lower.is_empty() || matches!(lower.as_str(), "d1" | "nano") {
        bail!("family name '{name}' is empty or reserved");
    }
    let mut f = FAMILIES.write().map_err(|_| miette!("family registry poisoned"))?;
    let key: &'static str = match f.keys().find(|k| **k == lower) {
        Some(k) => k,
        None => Box::leak(lower.into_boxed_str()),
    };
    f.insert(key, templates);
    Ok(Family::Custom(key))
}

fn family_templates(name: &str) -> Option<FamilyTemplates> {
    FAMILIES.read().ok().and_then(|f| f.get(name).copied())
}

impl Family {
    /// `d1` / `nano` / the registered name.
    pub fn name(self) -> &'static str {
        match self {
            Family::D1 => "d1",
            Family::Nano => "nano",
            Family::Custom(n) => n,
        }
    }
    /// Parse `d1`, `d1-omni`, `nano`, `decider-nano`, or a registered family.
    pub fn parse(s: &str) -> Result<Self> {
        let l = s.trim().to_ascii_lowercase();
        Ok(match l.as_str() {
            "d1" | "d1-omni" | "d1-omni-600m" => Family::D1,
            "nano" | "decider-nano" | "tasksource-decider-nano" => Family::Nano,
            other => match FAMILIES.read().ok().and_then(|f| f.keys().find(|k| **k == other).copied()) {
                Some(k) => Family::Custom(k),
                None => bail!("unknown decider family '{other}': expected 'd1', 'nano' or a family registered with register_family"),
            },
        })
    }
}

/// A decision type. Each has fixed, versioned prompts per [`Family`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Template {
    /// `supersedes(old, new)`: P(the new note makes the old one no longer true).
    Supersedes,
    /// `same_entity(a, b)`: P(both mentions refer to one entity). Symmetric.
    SameEntity,
    /// `duplicate(a, b)`: P(both say the same thing). Symmetric.
    Duplicate,
    /// `contradicts(a, b)`: P(they cannot both be true). Symmetric.
    Contradicts,
    /// `relation(old, new)`: update / contradiction / compatible / duplicate.
    Relation,
    /// `important(text)`: P(worth remembering long-term). Unvalidated.
    Important,
}

/// Labels of the binary templates, in order: negative, positive.
pub const BINARY_LABELS: [&str; 2] = ["no", "yes"];
/// Labels of `relation`.
pub const RELATION_LABELS: [&str; 4] = ["update", "contradiction", "compatible", "duplicate"];

impl Template {
    /// Every template.
    pub const ALL: [Template; 6] = [
        Template::Supersedes,
        Template::SameEntity,
        Template::Duplicate,
        Template::Contradicts,
        Template::Relation,
        Template::Important,
    ];
    /// Name as used in CozoScript.
    pub fn name(self) -> &'static str {
        match self {
            Template::Supersedes => "supersedes",
            Template::SameEntity => "same_entity",
            Template::Duplicate => "duplicate",
            Template::Contradicts => "contradicts",
            Template::Relation => "relation",
            Template::Important => "important",
        }
    }
    /// Parse a template name.
    pub fn parse(s: &str) -> Result<Self> {
        let t = s.trim().to_ascii_lowercase();
        for x in Template::ALL {
            if x.name() == t {
                return Ok(x);
            }
        }
        #[derive(Debug, thiserror::Error, miette::Diagnostic)]
        #[error("unknown decide template '{0}'")]
        #[diagnostic(code(eval::decide_unknown_template))]
        #[diagnostic(help(
            "templates are fixed: supersedes, same_entity, duplicate, contradicts, relation, important"
        ))]
        struct UnknownTemplate(String);
        Err(UnknownTemplate(s.to_string()).into())
    }
    /// Number of texts it takes (1 for `important`, else 2).
    pub fn arity(self) -> usize {
        if self == Template::Important {
            1
        } else {
            2
        }
    }
    /// Whether both orders are scored and averaged.
    pub fn symmetric(self) -> bool {
        matches!(
            self,
            Template::SameEntity | Template::Duplicate | Template::Contradicts
        )
    }
    /// Output labels.
    pub fn labels(self) -> &'static [&'static str] {
        if self == Template::Relation {
            &RELATION_LABELS
        } else {
            &BINARY_LABELS
        }
    }
    /// Whether the output is yes/no.
    pub fn is_binary(self) -> bool {
        self != Template::Relation
    }
    /// Prompt version for `family`. Bumped whenever a prompt string changes;
    /// probes and calibrations fitted on another version are not applied.
    pub fn version(self, family: Family) -> u32 {
        match family {
            Family::Custom(n) => family_templates(n).map(|t| t.version).unwrap_or(0),
            _ => 1,
        }
    }
    /// `name@vN`.
    pub fn versioned(self, family: Family) -> String {
        format!("{}@v{}", self.name(), self.version(family))
    }
    /// The fixed prompt and readout for `family`.
    pub fn spec(self, family: Family) -> TemplateSpec {
        match family {
            Family::Custom(n) => match family_templates(n) {
                Some(t) => (t.spec)(self),
                None => panic!("template family '{n}' is not registered"),
            },
            _ => templates::spec(self, family),
        }
    }
    /// Render the request for `(a, b)` (`b` ignored by `important`).
    pub fn render(self, family: Family, a: &str, b: &str) -> DecideRequest {
        let s = self.spec(family);
        DecideRequest {
            state: (s.state)(a, b),
            question: (s.question)(a, b),
            options: s.options.iter().map(|o| o.to_string()).collect(),
        }
    }
}

/// A template's prompt for one family: how to build the state and question,
/// the option texts, and which output label each option's probability goes
/// to (several options may feed one label; a label no option feeds gets 0).
#[derive(Clone, Copy)]
pub struct TemplateSpec {
    /// State builder.
    pub state: fn(&str, &str) -> String,
    /// Question builder.
    pub question: fn(&str, &str) -> String,
    /// Option texts.
    pub options: &'static [&'static str],
    /// Option index -> label index.
    pub label_of_option: &'static [usize],
}

/// The validated prompts. d1 strings are the NLI / ContractNLI framings of
/// the d1-omni audit (scripts/infer-bench/d1/run_d1_v2.py, selected on the
/// adapt set); decider-nano strings follow its card's `Question/Option`
/// format and were selected on the same adapt set
/// (scripts/infer-bench/decide/nano_select.py).
pub mod templates {
    use super::{Family, Template, TemplateSpec};

    /// ContractNLI options, d1's rendering `Key: description`.
    pub const CNLI_OPTIONS: [&str; 3] = [
        "Entailment: The old note implies the new statement is true",
        "Contradiction: The old note and the new statement cannot both be true",
        "NotMentioned: The old note neither implies nor contradicts the new statement",
    ];
    /// ContractNLI question about `new` with the old note as the state.
    pub fn cnli_question(_old: &str, new: &str) -> String {
        format!("Classify the relationship between the old note and this new statement:\n{new}")
    }
    fn first(a: &str, _b: &str) -> String {
        a.to_string()
    }
    fn mentions(a: &str, b: &str) -> String {
        format!("Mention A: {a}\nMention B: {b}")
    }
    fn same_entity_q(_a: &str, _b: &str) -> String {
        "Are mention A and mention B the same entity?".to_string()
    }
    /// Same-entity options (noul rendering: `false: ..`, `true: ..`).
    pub const ENTITY_OPTIONS: [&str; 2] = [
        "false: they are different entities that may share a name",
        "true: both mentions refer to one and the same entity",
    ];
    fn important_q(_a: &str, _b: &str) -> String {
        "Is this an important, lasting fact worth remembering long-term?".to_string()
    }
    /// Importance options.
    pub const IMPORTANT_OPTIONS: [&str; 2] = [
        "false: no, it is trivial, temporary or small talk",
        "true: yes, it is an important, lasting fact",
    ];
    fn premise_hypothesis(a: &str, b: &str) -> String {
        format!("Premise: {a}\nHypothesis: {b}")
    }
    fn nli_q(_a: &str, _b: &str) -> String {
        "What is the relationship between the premise and the hypothesis?".to_string()
    }
    /// NLI options (decider-nano rendering `key: description`).
    pub const NLI_OPTIONS: [&str; 3] = [
        "entailment: The hypothesis follows from the premise",
        "neutral: The hypothesis might be true or false given the premise",
        "contradiction: The hypothesis contradicts the premise",
    ];

    /// The spec of `t` for `f`.
    pub fn spec(t: Template, f: Family) -> TemplateSpec {
        // option -> label index; binary labels are [no, yes], relation labels
        // [update, contradiction, compatible, duplicate].
        const CNLI_CONTRA: [usize; 3] = [0, 1, 0];
        const CNLI_ENTAIL: [usize; 3] = [1, 0, 0];
        const CNLI_REL: [usize; 3] = [3, 0, 2];
        const NLI_CONTRA: [usize; 3] = [0, 0, 1];
        const NLI_ENTAIL: [usize; 3] = [1, 0, 0];
        const NOUL: [usize; 2] = [0, 1];
        let cnli = |map: &'static [usize]| TemplateSpec {
            state: first,
            question: cnli_question,
            options: &CNLI_OPTIONS,
            label_of_option: map,
        };
        let nli = |map: &'static [usize]| TemplateSpec {
            state: premise_hypothesis,
            question: nli_q,
            options: &NLI_OPTIONS,
            label_of_option: map,
        };
        match (t, f) {
            // d1: the audit's ContractNLI framing (adapt AUROC 0.941)
            (Template::Supersedes | Template::Contradicts, Family::D1) => cnli(&CNLI_CONTRA),
            (Template::Duplicate, Family::D1) => cnli(&CNLI_ENTAIL),
            // nano: premise/hypothesis NLI (adapt AUROC 0.894 / 0.946 / 1.0)
            (Template::Supersedes | Template::Contradicts, _) => nli(&NLI_CONTRA),
            (Template::Duplicate, _) => nli(&NLI_ENTAIL),
            // both: ContractNLI mapped to 3 of the 4 relation labels
            // (Entailment -> duplicate, Contradiction -> update,
            // NotMentioned -> compatible); a probe separates contradiction
            (Template::Relation, _) => cnli(&CNLI_REL),
            (Template::SameEntity, _) => TemplateSpec {
                state: mentions,
                question: same_entity_q,
                options: &ENTITY_OPTIONS,
                label_of_option: &NOUL,
            },
            (Template::Important, _) => TemplateSpec {
                state: first,
                question: important_q,
                options: &IMPORTANT_OPTIONS,
                label_of_option: &NOUL,
            },
        }
    }
}

// ------------------------------------------------------------------ calibration

/// How confident a decision is, as an action policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Band {
    /// p ≥ `tau_auto`: safe to apply automatically (still reversible).
    Auto,
    /// `threshold` ≤ p < `tau_auto`: apply as tentative, or escalate.
    Tentative,
    /// p < `threshold`: take no action.
    None,
}

impl Band {
    /// `auto` / `tentative` / `none`.
    pub fn name(self) -> &'static str {
        match self {
            Band::Auto => "auto",
            Band::Tentative => "tentative",
            Band::None => "none",
        }
    }
}

/// Per-template, per-model calibration of the zero-shot readout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    /// Additive log-bias per option (prior correction); empty = none.
    pub bias: Vec<f64>,
    /// Logit temperature (1 = none).
    pub temperature: f64,
    /// Label threshold: binary → "yes" iff p ≥ threshold; relation → the
    /// argmax label is acted on (tentatively) iff its p ≥ threshold.
    pub threshold: f64,
    /// Auto-apply threshold (risk-controlled; > 1 means never auto).
    pub tau_auto: f64,
    /// Where these numbers come from.
    pub source: String,
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            bias: vec![],
            temperature: 1.0,
            threshold: 0.5,
            tau_auto: 0.9,
            source: "uncalibrated".to_string(),
        }
    }
}

impl Calibration {
    /// The shipped defaults for `template` on `family`, fitted on the 60
    /// held-out adapt items (examples/decide_eval.rs `--fit-defaults`).
    /// Templates without labelled adapt data (`same_entity`, `important`)
    /// get a neutral 0.5 threshold and a conservative 0.9 auto threshold.
    pub fn default_for(template: Template, family: Family) -> Self {
        match family {
            Family::Custom(n) => family_templates(n)
                .map(|t| (t.calibration)(template))
                .unwrap_or_default(),
            _ => defaults::calibration(template, family),
        }
    }
    /// Band of probability `p`.
    pub fn band(&self, p: f64) -> Band {
        if p >= self.tau_auto {
            Band::Auto
        } else if p >= self.threshold {
            Band::Tentative
        } else {
            Band::None
        }
    }
}

mod defaults;

/// Label probabilities from one answer: temperature and bias on the option
/// logits, softmax, then option mass summed into labels.
pub fn label_probs(spec: &TemplateSpec, n_labels: usize, logits: &[f32], cal: &Calibration) -> Result<Vec<f64>> {
    if logits.len() != spec.options.len() {
        bail!(
            "the decider returned {} scores for {} options",
            logits.len(),
            spec.options.len()
        );
    }
    let t = if cal.temperature > 0.0 { cal.temperature } else { 1.0 };
    let z: Vec<f64> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| l as f64 / t + cal.bias.get(i).copied().unwrap_or(0.0))
        .collect();
    let p = softmax(&z);
    let mut out = vec![0.0; n_labels];
    for (i, pi) in p.into_iter().enumerate() {
        out[spec.label_of_option[i]] += pi;
    }
    Ok(out)
}

/// Numerically stable softmax.
pub fn softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

// ------------------------------------------------------------------ decisions

/// The outcome of one decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    /// Template name.
    pub template: String,
    /// Prompt version.
    pub template_version: u32,
    /// The chosen label: `yes`/`no`, or a relation label.
    pub label: String,
    /// Binary: P(yes). Relation: P(label).
    pub p: f64,
    /// Every label's probability, in [`Template::labels`] order.
    pub probs: Vec<f64>,
    /// Action band.
    pub band: Band,
    /// `band != Tentative`: the answer is either safe to act on or safe to
    /// ignore. Escalate (or record as tentative) when false.
    pub confident: bool,
    /// Model id of the decider.
    pub model: String,
    /// `zero_shot`, `calibrated` or `probe`.
    pub via: String,
}

impl Decision {
    /// `model|template@vN|via`: one string to store next to a decision row.
    pub fn decided_by(&self) -> String {
        format!(
            "{}|{}@v{}|{}",
            self.model, self.template, self.template_version, self.via
        )
    }
}

/// Raw per-item output of a decider for one template: label probabilities
/// (calibrated, averaged over both orders for symmetric templates) and, when
/// requested, the probe feature vector of each order.
#[derive(Debug, Clone, PartialEq)]
pub struct Scored {
    /// Calibrated label probabilities.
    pub probs: Vec<f64>,
    /// Raw option probabilities (softmax of the logits, no calibration),
    /// averaged over orders; what [`fit_bias`] takes.
    pub option_probs: Vec<f64>,
    /// One concatenated feature vector per scored order (1 or 2).
    pub features: Vec<Vec<f32>>,
}

/// Render, batch and score `items` (`(a, b)`; `b` unused by `important`).
/// Symmetric templates score both orders. Requests are sent in chunks of
/// [`Decider::max_batch`].
pub fn score_items(
    decider: &dyn Decider,
    family: Family,
    template: Template,
    items: &[(&str, &str)],
    cal: &Calibration,
    with_features: bool,
) -> Result<Vec<Scored>> {
    let spec = template.spec(family);
    let orders = if template.symmetric() { 2 } else { 1 };
    let mut reqs = Vec::with_capacity(items.len() * orders);
    for (a, b) in items {
        reqs.push(template.render(family, a, b));
        if orders == 2 {
            reqs.push(template.render(family, b, a));
        }
    }
    let chunk = decider.max_batch().max(1);
    let mut answers = Vec::with_capacity(reqs.len());
    for c in reqs.chunks(chunk) {
        let got = decider.decide(c, with_features)?;
        if got.len() != c.len() {
            bail!(
                "decider '{}' returned {} answers for {} requests",
                decider.model_id(),
                got.len(),
                c.len()
            );
        }
        answers.extend(got);
    }
    let n_labels = template.labels().len();
    let mut out = Vec::with_capacity(items.len());
    for ans in answers.chunks(orders) {
        let mut probs = vec![0.0; n_labels];
        let mut option_probs = vec![0.0; spec.options.len()];
        let mut feats = vec![];
        for a in ans {
            let p = label_probs(&spec, n_labels, &a.logits, cal)?;
            for (acc, x) in probs.iter_mut().zip(p) {
                *acc += x / orders as f64;
            }
            let raw: Vec<f64> = a.logits.iter().map(|x| *x as f64).collect();
            for (acc, x) in option_probs.iter_mut().zip(softmax(&raw)) {
                *acc += x / orders as f64;
            }
            if with_features {
                if a.features.is_empty() {
                    bail!(
                        "decider '{}' returns no features, so a probe cannot be fitted or applied",
                        decider.model_id()
                    );
                }
                feats.push(a.features.concat());
            }
        }
        out.push(Scored {
            probs,
            option_probs,
            features: feats,
        });
    }
    Ok(out)
}

/// Turn label probabilities into a [`Decision`] with `cal`'s thresholds.
pub fn finish(template: Template, family: Family, model: &str, probs: Vec<f64>, cal: &Calibration, via: &str) -> Decision {
    let labels = template.labels();
    let (label, p, band) = if template.is_binary() {
        let p = probs[1];
        let label = if p >= cal.threshold { labels[1] } else { labels[0] };
        (label, p, cal.band(p))
    } else {
        let (i, p) = probs
            .iter()
            .copied()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |acc, (i, p)| if p > acc.1 { (i, p) } else { acc });
        (labels[i], p, cal.band(p))
    };
    Decision {
        template: template.name().to_string(),
        template_version: template.version(family),
        label: label.to_string(),
        p,
        probs,
        band,
        confident: band != Band::Tentative,
        model: model.to_string(),
        via: via.to_string(),
    }
}

/// Decide `items` with an explicit decider, calibration and optional probe.
/// This is the whole pipeline without any registry; the registry-based
/// [`decide`] looks the three up and calls it.
pub fn decide_with(
    decider: &dyn Decider,
    family: Family,
    template: Template,
    items: &[(&str, &str)],
    cal: &Calibration,
    probe: Option<&Probe>,
) -> Result<Vec<Decision>> {
    let model = decider.model_id();
    if let Some(probe) = probe {
        // the probe reads raw features; the zero-shot readout is not used
        let scored = score_items(decider, family, template, items, &Calibration::default(), true)?;
        let pc = probe.calibration();
        return scored
            .into_iter()
            .map(|s| {
                let k = s.features.len() as f64;
                let mut probs = vec![0.0; template.labels().len()];
                for f in &s.features {
                    for (acc, x) in probs.iter_mut().zip(probe.predict(f)?) {
                        *acc += x / k;
                    }
                }
                Ok(finish(template, family, &model, probs, &pc, "probe"))
            })
            .collect();
    }
    let via = if cal.source == "uncalibrated" {
        "zero_shot"
    } else {
        "calibrated"
    };
    Ok(score_items(decider, family, template, items, cal, false)?
        .into_iter()
        .map(|s| finish(template, family, &model, s.probs, cal, via))
        .collect())
}

// ------------------------------------------------------------------ fitting helpers

/// AUROC (ties count half). NaN when one class is empty.
pub fn auroc(scores: &[f64], labels: &[bool]) -> f64 {
    let pos: Vec<f64> = scores.iter().zip(labels).filter(|x| *x.1).map(|x| *x.0).collect();
    let neg: Vec<f64> = scores.iter().zip(labels).filter(|x| !*x.1).map(|x| *x.0).collect();
    if pos.is_empty() || neg.is_empty() {
        return f64::NAN;
    }
    let mut s = 0.0;
    for p in &pos {
        for n in &neg {
            s += if p > n {
                1.0
            } else if p == n {
                0.5
            } else {
                0.0
            };
        }
    }
    s / (pos.len() * neg.len()) as f64
}

/// Accuracy of `score >= t`.
pub fn accuracy_at(scores: &[f64], labels: &[bool], t: f64) -> f64 {
    let ok = scores
        .iter()
        .zip(labels)
        .filter(|(s, y)| (**s >= t) == **y)
        .count();
    ok as f64 / scores.len().max(1) as f64
}

/// The accuracy-maximising threshold (ties: closest to 0.5).
pub fn fit_threshold(scores: &[f64], labels: &[bool]) -> f64 {
    let mut cands: Vec<f64> = scores.to_vec();
    cands.push(1.01);
    cands.push(0.5);
    let mut best: (f64, f64) = (f64::NEG_INFINITY, 0.5);
    for t in cands {
        let a = accuracy_at(scores, labels, t);
        if a > best.0 + 1e-12 || ((a - best.0).abs() <= 1e-12 && (t - 0.5).abs() < (best.1 - 0.5).abs()) {
            best = (a, t);
        }
    }
    best.1
}

/// Upper end of the 95% Wilson interval for `k` errors out of `n`.
pub fn wilson_upper(k: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let z = 1.96f64;
    let n_f = n as f64;
    let ph = k as f64 / n_f;
    let den = 1.0 + z * z / n_f;
    let c = ph + z * z / (2.0 * n_f);
    let r = z * ((ph * (1.0 - ph) / n_f) + z * z / (4.0 * n_f * n_f)).sqrt();
    ((c + r) / den).min(1.0)
}

/// Result of [`fit_tau_auto`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TauFit {
    /// The auto-apply threshold (1.01 = never).
    pub tau: f64,
    /// Share of all items at or above it.
    pub coverage: f64,
    /// False-positive share among the items at or above it.
    pub false_rate: f64,
}

/// Risk-controlled auto threshold: the smallest τ such that, for every
/// τ' ≥ τ, the false-positive share among items scored ≥ τ' is ≤ `alpha`
/// (with `wilson`, its 95% Wilson upper bound). Monotone, so the guarantee
/// does not depend on where the scan stops.
pub fn fit_tau_auto(scores: &[f64], labels: &[bool], alpha: f64, wilson: bool) -> TauFit {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut best = TauFit {
        tau: 1.01,
        coverage: 0.0,
        false_rate: 0.0,
    };
    let (mut n, mut fp) = (0usize, 0usize);
    let mut i = 0;
    while i < idx.len() {
        // take all items tied at this score together
        let s = scores[idx[i]];
        while i < idx.len() && scores[idx[i]] == s {
            n += 1;
            if !labels[idx[i]] {
                fp += 1;
            }
            i += 1;
        }
        let rate = if wilson {
            wilson_upper(fp, n)
        } else {
            fp as f64 / n as f64
        };
        if rate > alpha {
            break;
        }
        best = TauFit {
            tau: s,
            coverage: n as f64 / scores.len() as f64,
            false_rate: fp as f64 / n as f64,
        };
    }
    best
}

/// Per-option log-bias maximising accuracy of the argmax label (prior
/// correction), by grid search over `[-3, 3]` with option 0 fixed at 0.
/// `option_probs[i]` are item i's option probabilities; `gold[i]` its label
/// index; `label_of_option` maps options to labels.
pub fn fit_bias(option_probs: &[Vec<f64>], gold: &[usize], label_of_option: &[usize]) -> Vec<f64> {
    let k = label_of_option.len();
    if k < 2 || option_probs.is_empty() {
        return vec![0.0; k];
    }
    let grid: Vec<f64> = (0..25).map(|i| -3.0 + 0.25 * i as f64).collect();
    let logs: Vec<Vec<f64>> = option_probs
        .iter()
        .map(|p| p.iter().map(|x| (x + 1e-9).ln()).collect())
        .collect();
    let acc = |bias: &[f64]| -> f64 {
        let mut ok = 0;
        for (l, g) in logs.iter().zip(gold) {
            let mut best = (0, f64::NEG_INFINITY);
            for j in 0..k {
                let v = l[j] + bias[j];
                if v > best.1 {
                    best = (j, v);
                }
            }
            if label_of_option[best.0] == *g {
                ok += 1;
            }
        }
        ok as f64 / logs.len() as f64
    };
    let mut best = (acc(&vec![0.0; k]), vec![0.0; k]);
    let mut cur = vec![0.0; k];
    // exhaustive for up to 3 free options (25^3), else coordinate passes
    fn rec(j: usize, k: usize, grid: &[f64], cur: &mut Vec<f64>, best: &mut (f64, Vec<f64>), acc: &dyn Fn(&[f64]) -> f64) {
        if j == k {
            let a = acc(cur);
            let norm = |b: &[f64]| b.iter().map(|x| x.abs()).sum::<f64>();
            if a > best.0 + 1e-12 || ((a - best.0).abs() <= 1e-12 && norm(cur) < norm(&best.1)) {
                *best = (a, cur.clone());
            }
            return;
        }
        for &g in grid {
            cur[j] = g;
            rec(j + 1, k, grid, cur, best, acc);
        }
        cur[j] = 0.0;
    }
    rec(1, k, &grid, &mut cur, &mut best, &acc);
    best.1
}

// ------------------------------------------------------------------ probe

/// A logistic-regression probe over decision features (softmax regression
/// with an L2 penalty; binary is two classes). Features are standardised
/// with the training mean and scale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Probe {
    /// Template name.
    pub template: String,
    /// Prompt version the features came from.
    pub template_version: u32,
    /// Model id the features came from.
    pub model: String,
    /// Class labels (the template's labels).
    pub classes: Vec<String>,
    /// Feature width.
    pub dim: usize,
    /// Training mean per feature.
    pub mean: Vec<f32>,
    /// Training scale per feature.
    pub scale: Vec<f32>,
    /// `classes × dim` weights.
    pub weights: Vec<Vec<f32>>,
    /// Per-class intercept.
    pub intercept: Vec<f32>,
    /// Inverse regularisation strength chosen by cross-validation.
    pub c: f64,
    /// Training items.
    pub n: usize,
    /// Cross-validated AUROC (binary) or accuracy (relation).
    pub cv_score: f64,
    /// Label threshold from cross-validated scores.
    pub threshold: f64,
    /// Auto threshold from cross-validated scores.
    pub tau_auto: f64,
    /// Auto coverage on the training items (cross-validated).
    pub coverage: f64,
    /// Seconds since the epoch.
    pub fitted_at: f64,
}

impl Probe {
    /// Class probabilities for one feature vector.
    pub fn predict(&self, x: &[f32]) -> Result<Vec<f64>> {
        if x.len() != self.dim {
            bail!(
                "probe for {} expects {} features, got {}",
                self.template,
                self.dim,
                x.len()
            );
        }
        let z: Vec<f64> = self
            .weights
            .iter()
            .zip(&self.intercept)
            .map(|(w, b)| {
                let mut s = *b as f64;
                for j in 0..self.dim {
                    s += w[j] as f64 * ((x[j] - self.mean[j]) / self.scale[j]) as f64;
                }
                s
            })
            .collect();
        Ok(softmax(&z))
    }
    /// The thresholds this probe was fitted with.
    pub fn calibration(&self) -> Calibration {
        Calibration {
            bias: vec![],
            temperature: 1.0,
            threshold: self.threshold,
            tau_auto: self.tau_auto,
            source: "probe".to_string(),
        }
    }
}

/// Options of [`fit_probe`].
#[derive(Debug, Clone)]
pub struct ProbeOptions {
    /// Candidate inverse regularisation strengths (chosen by CV).
    pub c_grid: Vec<f64>,
    /// Cross-validation folds.
    pub folds: usize,
    /// False-auto rate bound for `tau_auto`.
    pub alpha: f64,
    /// Bound the false-auto rate by its Wilson upper bound.
    pub wilson: bool,
    /// Gradient steps.
    pub iters: usize,
}

impl Default for ProbeOptions {
    fn default() -> Self {
        Self {
            c_grid: vec![0.01, 0.03, 0.1, 0.3, 1.0],
            folds: 5,
            alpha: 0.02,
            wilson: false,
            iters: 300,
        }
    }
}

struct Lr {
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
}

/// Fit softmax regression on standardised `x` (n × d) with penalty
/// `‖W‖² / (2·C·n)` on the mean cross-entropy, by Nesterov-accelerated
/// gradient descent with step 1/L.
fn fit_lr(x: &[Vec<f64>], y: &[usize], k: usize, c: f64, iters: usize) -> Lr {
    let n = x.len();
    let d = x.first().map(|r| r.len()).unwrap_or(0);
    let lam = 1.0 / (c * n as f64);
    // L ≤ λ_max(XᵀX)/n · 1/2 (softmax Hessian bound) + λ; power iteration
    let mut v = vec![1.0 / (d.max(1) as f64).sqrt(); d];
    let mut eig = 1.0;
    for _ in 0..30 {
        let xv: Vec<f64> = x.iter().map(|r| r.iter().zip(&v).map(|(a, b)| a * b).sum()).collect();
        let mut w = vec![0.0; d];
        for (r, s) in x.iter().zip(&xv) {
            for j in 0..d {
                w[j] += r[j] * s;
            }
        }
        let norm = w.iter().map(|a| a * a).sum::<f64>().sqrt().max(1e-12);
        eig = norm;
        v = w.into_iter().map(|a| a / norm).collect();
    }
    let lip = 0.5 * (eig + n as f64) / n as f64 + lam; // + n for the intercept column
    let step = 1.0 / lip;
    let mut w = vec![vec![0.0; d]; k];
    let mut b = vec![0.0; k];
    let (mut wp, mut bp) = (w.clone(), b.clone());
    for it in 0..iters {
        let mom = it as f64 / (it as f64 + 3.0);
        let wl: Vec<Vec<f64>> = w
            .iter()
            .zip(&wp)
            .map(|(a, p)| a.iter().zip(p).map(|(a, p)| a + mom * (a - p)).collect())
            .collect();
        let bl: Vec<f64> = b.iter().zip(&bp).map(|(a, p)| a + mom * (a - p)).collect();
        let mut gw = vec![vec![0.0; d]; k];
        let mut gb = vec![0.0; k];
        for (r, &yi) in x.iter().zip(y) {
            let z: Vec<f64> = (0..k)
                .map(|c| bl[c] + wl[c].iter().zip(r).map(|(a, b)| a * b).sum::<f64>())
                .collect();
            let p = softmax(&z);
            for c in 0..k {
                let g = p[c] - if c == yi { 1.0 } else { 0.0 };
                gb[c] += g / n as f64;
                for j in 0..d {
                    gw[c][j] += g * r[j] / n as f64;
                }
            }
        }
        wp = w;
        bp = b;
        w = wl
            .iter()
            .zip(&gw)
            .map(|(a, g)| a.iter().zip(g).map(|(a, g)| a - step * (g + lam * a)).collect())
            .collect();
        b = bl.iter().zip(&gb).map(|(a, g)| a - step * g).collect();
    }
    Lr { w, b }
}

fn lr_predict(m: &Lr, r: &[f64]) -> Vec<f64> {
    let z: Vec<f64> = m
        .w
        .iter()
        .zip(&m.b)
        .map(|(w, b)| b + w.iter().zip(r).map(|(a, b)| a * b).sum::<f64>())
        .collect();
    softmax(&z)
}

/// Standardise columns; returns (mean, scale).
fn standardise(xs: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    let d = xs[0].len();
    let n = xs.len() as f64;
    let mut mean = vec![0.0f64; d];
    for r in xs {
        for j in 0..d {
            mean[j] += r[j] as f64 / n;
        }
    }
    let mut var = vec![0.0f64; d];
    for r in xs {
        for j in 0..d {
            var[j] += (r[j] as f64 - mean[j]).powi(2) / n;
        }
    }
    let scale = var
        .into_iter()
        .map(|v| if v.sqrt() > 1e-6 { v.sqrt() as f32 } else { 1.0 })
        .collect();
    (mean.into_iter().map(|m| m as f32).collect(), scale)
}

/// Stratified fold of each item (deterministic).
fn folds_of(y: &[usize], folds: usize) -> Vec<usize> {
    let mut seen: HashMap<usize, usize> = HashMap::new();
    y.iter()
        .map(|c| {
            let e = seen.entry(*c).or_insert(0);
            let f = *e % folds;
            *e += 1;
            f
        })
        .collect()
}

/// Score of out-of-fold predictions: AUROC of class 1 when binary, else
/// accuracy.
fn cv_score(p: &[Vec<f64>], y: &[usize], k: usize) -> f64 {
    if k == 2 {
        let s: Vec<f64> = p.iter().map(|r| r[1]).collect();
        let l: Vec<bool> = y.iter().map(|c| *c == 1).collect();
        auroc(&s, &l)
    } else {
        let ok = p
            .iter()
            .zip(y)
            .filter(|(r, c)| {
                let am = r
                    .iter()
                    .enumerate()
                    .fold((0, f64::NEG_INFINITY), |a, (i, v)| if *v > a.1 { (i, *v) } else { a })
                    .0;
                am == **c
            })
            .count();
        ok as f64 / y.len() as f64
    }
}

/// Fit a probe for `template` on `model`'s features `xs` with label indices
/// `ys` (into [`Template::labels`]). C is chosen by stratified k-fold CV; the
/// label and auto thresholds are fitted on the out-of-fold predictions, so
/// they are not optimistic.
pub fn fit_probe(
    template: Template,
    family: Family,
    model: &str,
    xs: &[Vec<f32>],
    ys: &[usize],
    opts: &ProbeOptions,
) -> Result<Probe> {
    let k = template.labels().len();
    if xs.len() != ys.len() || xs.is_empty() {
        bail!("a probe needs as many labels as feature rows, and at least one");
    }
    let d = xs[0].len();
    if d == 0 || xs.iter().any(|r| r.len() != d) {
        bail!("probe features must all have the same non-zero width");
    }
    for c in 0..k {
        let cnt = ys.iter().filter(|y| **y == c).count();
        if template.is_binary() && cnt < 2 {
            bail!(
                "fitting a probe for {} needs at least 2 '{}' examples, got {cnt}",
                template.name(),
                template.labels()[c]
            );
        }
    }
    if ys.iter().any(|y| *y >= k) {
        bail!("label index out of range for {}", template.name());
    }
    let (mean, scale) = standardise(xs);
    let z: Vec<Vec<f64>> = xs
        .iter()
        .map(|r| (0..d).map(|j| ((r[j] - mean[j]) / scale[j]) as f64).collect())
        .collect();
    let folds = opts.folds.clamp(2, xs.len().max(2));
    let fold = folds_of(ys, folds);
    let mut best: Option<(f64, f64, Vec<Vec<f64>>)> = None;
    for &c in &opts.c_grid {
        let mut oof = vec![vec![0.0; k]; xs.len()];
        for f in 0..folds {
            let tr: Vec<usize> = (0..xs.len()).filter(|i| fold[*i] != f).collect();
            if tr.is_empty() {
                continue;
            }
            let m = fit_lr(
                &tr.iter().map(|i| z[*i].clone()).collect::<Vec<_>>(),
                &tr.iter().map(|i| ys[*i]).collect::<Vec<_>>(),
                k,
                c,
                opts.iters,
            );
            for i in (0..xs.len()).filter(|i| fold[*i] == f) {
                oof[i] = lr_predict(&m, &z[i]);
            }
        }
        let s = cv_score(&oof, ys, k);
        let s = if s.is_nan() { 0.0 } else { s };
        if best.as_ref().map(|b| s > b.0 + 1e-9).unwrap_or(true) {
            best = Some((s, c, oof));
        }
    }
    let (cv, c, oof) = best.ok_or_else(|| miette!("empty C grid"))?;
    let m = fit_lr(&z, ys, k, c, opts.iters);
    let (threshold, tf) = if k == 2 {
        let s: Vec<f64> = oof.iter().map(|r| r[1]).collect();
        let l: Vec<bool> = ys.iter().map(|c| *c == 1).collect();
        (fit_threshold(&s, &l), fit_tau_auto(&s, &l, opts.alpha, opts.wilson))
    } else {
        // relation: confidence of the argmax label vs whether it was right
        let (s, l): (Vec<f64>, Vec<bool>) = oof
            .iter()
            .zip(ys)
            .map(|(r, c)| {
                let (i, p) = r
                    .iter()
                    .enumerate()
                    .fold((0, f64::NEG_INFINITY), |a, (i, v)| if *v > a.1 { (i, *v) } else { a });
                (p, i == *c)
            })
            .unzip();
        (0.0, fit_tau_auto(&s, &l, opts.alpha, opts.wilson))
    };
    Ok(Probe {
        template: template.name().to_string(),
        template_version: template.version(family),
        model: model.to_string(),
        classes: template.labels().iter().map(|s| s.to_string()).collect(),
        dim: d,
        mean,
        scale,
        weights: m.w.iter().map(|r| r.iter().map(|x| *x as f32).collect()).collect(),
        intercept: m.b.iter().map(|x| *x as f32).collect(),
        c,
        n: xs.len(),
        cv_score: cv,
        threshold,
        tau_auto: tf.tau,
        coverage: tf.coverage,
        fitted_at: now(),
    })
}

/// Fit a zero-shot calibration from label probabilities (`probs`, as from
/// [`score_items`] with an uncalibrated [`Calibration`]) and gold label
/// indices. Binary: threshold by accuracy, τ_auto by risk control. Relation:
/// per-option log-bias needs option probabilities, so pass them in
/// `option_probs` (else the bias stays zero).
pub fn fit_calibration(
    template: Template,
    family: Family,
    probs: &[Vec<f64>],
    option_probs: Option<&[Vec<f64>]>,
    gold: &[usize],
    alpha: f64,
    wilson: bool,
) -> Calibration {
    if template.is_binary() {
        let s: Vec<f64> = probs.iter().map(|p| p[1]).collect();
        let l: Vec<bool> = gold.iter().map(|g| *g == 1).collect();
        let tf = fit_tau_auto(&s, &l, alpha, wilson);
        Calibration {
            bias: vec![],
            temperature: 1.0,
            threshold: fit_threshold(&s, &l),
            tau_auto: tf.tau,
            source: format!("fitted n={} alpha={alpha}", gold.len()),
        }
    } else {
        let spec = template.spec(family);
        let bias = option_probs
            .map(|op| fit_bias(op, gold, spec.label_of_option))
            .unwrap_or_default();
        // confidence of the corrected argmax vs correctness
        let (s, l): (Vec<f64>, Vec<bool>) = match option_probs {
            Some(op) => op
                .iter()
                .zip(gold)
                .map(|(p, g)| {
                    let z: Vec<f64> = p
                        .iter()
                        .enumerate()
                        .map(|(i, x)| (x + 1e-9).ln() + bias.get(i).copied().unwrap_or(0.0))
                        .collect();
                    let q = softmax(&z);
                    let mut lp = vec![0.0; template.labels().len()];
                    for (i, x) in q.into_iter().enumerate() {
                        lp[spec.label_of_option[i]] += x;
                    }
                    let (i, m) = lp
                        .iter()
                        .enumerate()
                        .fold((0, f64::NEG_INFINITY), |a, (i, v)| if *v > a.1 { (i, *v) } else { a });
                    (m, i == *g)
                })
                .unzip(),
            None => probs
                .iter()
                .zip(gold)
                .map(|(p, g)| {
                    let (i, m) = p
                        .iter()
                        .enumerate()
                        .fold((0, f64::NEG_INFINITY), |a, (i, v)| if *v > a.1 { (i, *v) } else { a });
                    (m, i == *g)
                })
                .unzip(),
        };
        let tf = fit_tau_auto(&s, &l, alpha, wilson);
        Calibration {
            bias,
            temperature: 1.0,
            threshold: 0.0,
            tau_auto: tf.tau,
            source: format!("fitted n={} alpha={alpha}", gold.len()),
        }
    }
}

/// Index of a gold label value in `template`'s labels. Binary templates
/// accept booleans, 0/1, `yes`/`no`, `true`/`false`; relation takes its
/// label names (`contradiction` also as `contradict`).
pub fn label_index(template: Template, v: &DataValue) -> Result<usize> {
    let bad = || {
        miette!(
            "label {v:?} is not valid for {}: expected {}",
            template.name(),
            if template.is_binary() {
                "a boolean, 0/1 or 'yes'/'no'".to_string()
            } else {
                RELATION_LABELS.join(", ")
            }
        )
    };
    if template.is_binary() {
        match v {
            DataValue::Bool(b) => Ok(*b as usize),
            DataValue::Num(_) => match v.get_int() {
                Some(0) => Ok(0),
                Some(1) => Ok(1),
                _ => Err(bad()),
            },
            DataValue::Str(s) => match s.to_ascii_lowercase().as_str() {
                "yes" | "true" | "1" => Ok(1),
                "no" | "false" | "0" => Ok(0),
                _ => Err(bad()),
            },
            _ => Err(bad()),
        }
    } else {
        match v {
            DataValue::Str(s) => {
                let s = s.to_ascii_lowercase();
                let s = if s == "contradict" { "contradiction".to_string() } else { s };
                RELATION_LABELS.iter().position(|l| *l == s).ok_or_else(bad)
            }
            _ => Err(bad()),
        }
    }
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ------------------------------------------------------------------ registry

/// What [`enable_decider`] (and `::decider enable`) should build. Every
/// field is optional:
///
/// * `model`: `decider-nano` (default; also `nano` or
///   `tasksource/tasksource-decider-nano`) or `d1-omni` (`d1`,
///   `LiquidAI/d1-omni-600M`).
/// * `precision`: `f32` (default for decider-nano), `f16`, `bf16`, `int8`,
///   `int4`.
/// * `device`: `auto` (default), `cpu`, `metal`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeciderSpec {
    /// Model name or Hugging Face repo.
    pub model: Option<String>,
    /// Weight precision.
    pub precision: Option<String>,
    /// Device preference.
    pub device: Option<String>,
}

impl DeciderSpec {
    /// A spec for `model` with defaults elsewhere.
    pub fn model(model: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
            ..Default::default()
        }
    }
    fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

/// One row of [`deciders`] / `::decider list`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeciderStatus {
    /// Registered name.
    pub name: String,
    /// `d1-omni`, `decider-nano` or `custom`.
    pub kind: String,
    /// Template dialect (`d1` / `nano`).
    pub family: String,
    /// Model id.
    pub model: String,
    /// Whether it can be called.
    pub enabled: bool,
    /// Whether weights are resident (`None`: not tracked).
    pub loaded: Option<bool>,
    /// Weight precision.
    pub precision: Option<String>,
    /// Device.
    pub device: Option<String>,
}

#[derive(Clone)]
pub(crate) struct DeciderInfo {
    pub(crate) kind: &'static str,
    pub(crate) family: Family,
    pub(crate) model: String,
    pub(crate) precision: Option<String>,
    pub(crate) device_pref: Option<String>,
    pub(crate) spec: Option<DeciderSpec>,
}

struct RegisteredDecider {
    decider: Arc<dyn Decider>,
    info: DeciderInfo,
}

struct ParkedDecider {
    info: DeciderInfo,
    custom: Option<Arc<dyn Decider>>,
}

/// A fitted per-(model, template) override: a probe or a calibration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Fitted {
    /// Logistic probe on features.
    Probe(Probe),
    /// Zero-shot calibration.
    Calibration {
        /// Template version it was fitted on.
        template_version: u32,
        /// The numbers.
        calibration: Calibration,
    },
}

impl Fitted {
    /// Template version the fit applies to.
    pub fn template_version(&self) -> u32 {
        match self {
            Fitted::Probe(p) => p.template_version,
            Fitted::Calibration {
                template_version, ..
            } => *template_version,
        }
    }
    /// `probe` / `calibration`.
    pub fn kind(&self) -> &'static str {
        match self {
            Fitted::Probe(_) => "probe",
            Fitted::Calibration { .. } => "calibration",
        }
    }
}

lazy_static! {
    static ref DECIDERS: RwLock<BTreeMap<String, RegisteredDecider>> = RwLock::new(BTreeMap::new());
    static ref PARKED: Mutex<BTreeMap<String, ParkedDecider>> = Mutex::new(BTreeMap::new());
    /// (model id, template name) -> fitted override.
    static ref FITTED: RwLock<HashMap<(String, String), Arc<Fitted>>> = RwLock::new(HashMap::new());
}

/// Register any [`Decider`] as `name`, speaking `family`'s templates. Listed
/// as kind `custom`; `::decider disable` parks it and `::decider enable name`
/// restores it.
pub fn register_decider(name: &str, decider: Arc<dyn Decider>, family: Family) -> Result<()> {
    let model = decider.model_id();
    register_with_info(
        name,
        decider,
        DeciderInfo {
            kind: "custom",
            family,
            model,
            precision: None,
            device_pref: None,
            spec: None,
        },
    )
}

pub(crate) fn register_with_info(name: &str, decider: Arc<dyn Decider>, info: DeciderInfo) -> Result<()> {
    if name.is_empty() {
        bail!("a decider name must not be empty");
    }
    let old = DECIDERS
        .write()
        .map_err(|_| miette!("decider registry poisoned"))?
        .insert(name.to_string(), RegisteredDecider { decider, info });
    if let Ok(mut p) = PARKED.lock() {
        p.remove(name);
    }
    if let Some(old) = old {
        old.decider.unload();
    }
    Ok(())
}

/// Turn a decider on (process-wide). With an empty spec and a name disabled
/// earlier, the previous configuration comes back; with an empty spec and an
/// enabled name nothing happens. Otherwise the built-in model `spec` names is
/// built (needs the `infer` cargo feature); weights load on first use.
pub fn enable_decider(name: &str, spec: DeciderSpec) -> Result<()> {
    if name.is_empty() {
        bail!("a decider name must not be empty");
    }
    if spec.is_empty() {
        let parked = PARKED
            .lock()
            .map_err(|_| miette!("decider registry poisoned"))?
            .remove(name);
        if let Some(p) = parked {
            let restored = match (&p.custom, &p.info.spec) {
                (Some(c), _) => register_with_info(name, c.clone(), p.info.clone()),
                (None, Some(s)) => enable_builtin(name, s.clone()),
                (None, None) => Err(miette!("decider '{name}' cannot be re-enabled")),
            };
            if restored.is_err() {
                if let Ok(mut d) = PARKED.lock() {
                    d.insert(name.to_string(), p);
                }
            }
            return restored;
        }
        if DECIDERS.read().map(|r| r.contains_key(name)).unwrap_or(false) {
            return Ok(());
        }
    }
    enable_builtin(name, spec)
}

#[cfg(feature = "infer")]
fn enable_builtin(name: &str, spec: DeciderSpec) -> Result<()> {
    builtin::enable_from_spec(name, spec)
}

#[cfg(not(feature = "infer"))]
fn enable_builtin(name: &str, spec: DeciderSpec) -> Result<()> {
    #[derive(Debug, thiserror::Error, miette::Diagnostic)]
    #[error(
        "cannot enable decider '{0}' (model '{1}'): kleos was built without the `infer` cargo feature, so it has no built-in deciders"
    )]
    #[diagnostic(code(eval::decider_feature_missing))]
    #[diagnostic(help(
        "rebuild kleos with `--features infer` (or `infer-metal` for the Apple GPU), or register a custom decider from Rust with kleos::decide::register_decider"
    ))]
    struct FeatureMissing(String, String);
    Err(FeatureMissing(
        name.to_string(),
        spec.model.unwrap_or_else(|| "decider-nano".to_string()),
    )
    .into())
}

/// Turn a decider off: unload it and park its configuration. Returns false
/// when it was already disabled; errors when it was never enabled.
pub fn disable_decider(name: &str) -> Result<bool> {
    let removed = DECIDERS
        .write()
        .map_err(|_| miette!("decider registry poisoned"))?
        .remove(name);
    match removed {
        Some(r) => {
            r.decider.unload();
            let custom = if r.info.spec.is_some() {
                None
            } else {
                Some(r.decider.clone())
            };
            let info = r.info.clone();
            drop(r);
            PARKED
                .lock()
                .map_err(|_| miette!("decider registry poisoned"))?
                .insert(name.to_string(), ParkedDecider { info, custom });
            crate::embed::release_freed_memory();
            Ok(true)
        }
        None => {
            if PARKED.lock().map(|p| p.contains_key(name)).unwrap_or(false) {
                Ok(false)
            } else {
                Err(not_found(name))
            }
        }
    }
}

/// Forget a decider entirely (no re-enable). Returns whether it existed.
pub fn unregister_decider(name: &str) -> bool {
    let parked = PARKED.lock().map(|mut p| p.remove(name).is_some()).unwrap_or(false);
    let reg = DECIDERS
        .write()
        .map(|mut r| r.remove(name).is_some())
        .unwrap_or(false);
    parked || reg
}

/// Every enabled and disabled decider, sorted by name.
pub fn deciders() -> Vec<DeciderStatus> {
    let mut out: BTreeMap<String, DeciderStatus> = BTreeMap::new();
    if let Ok(r) = DECIDERS.read() {
        for (name, reg) in r.iter() {
            out.insert(
                name.clone(),
                DeciderStatus {
                    name: name.clone(),
                    kind: reg.info.kind.to_string(),
                    family: reg.info.family.name().to_string(),
                    model: reg.info.model.clone(),
                    enabled: true,
                    loaded: reg.decider.is_loaded(),
                    precision: reg.info.precision.clone(),
                    device: reg.decider.device().or_else(|| reg.info.device_pref.clone()),
                },
            );
        }
    }
    if let Ok(p) = PARKED.lock() {
        for (name, pk) in p.iter() {
            out.entry(name.clone()).or_insert_with(|| DeciderStatus {
                name: name.clone(),
                kind: pk.info.kind.to_string(),
                family: pk.info.family.name().to_string(),
                model: pk.info.model.clone(),
                enabled: false,
                loaded: Some(false),
                precision: pk.info.precision.clone(),
                device: pk.info.device_pref.clone(),
            });
        }
    }
    out.into_values().collect()
}

fn not_found(name: &str) -> miette::Report {
    if PARKED.lock().map(|p| p.contains_key(name)).unwrap_or(false) {
        #[derive(Debug, thiserror::Error, miette::Diagnostic)]
        #[error("decider '{0}' is disabled; turn it back on with `::decider enable {0}`")]
        #[diagnostic(code(eval::decider_disabled))]
        #[diagnostic(help(
            "turn it back on with `::decider enable {0}` (Rust: kleos::decide::enable_decider)"
        ))]
        struct DeciderDisabled(String);
        DeciderDisabled(name.to_string()).into()
    } else {
        #[derive(Debug, thiserror::Error, miette::Diagnostic)]
        #[error(
            "no decider is enabled under the name '{0}'; enable one with `::decider enable {0} {{model: 'decider-nano'}}`"
        )]
        #[diagnostic(code(eval::decider_not_found))]
        #[diagnostic(help(
            "built-in models: 'decider-nano' (tasksource/tasksource-decider-nano), 'd1-omni' (LiquidAI/d1-omni-600M); `::decider list` shows what is registered"
        ))]
        struct DeciderNotFound(String);
        DeciderNotFound(name.to_string()).into()
    }
}

/// The decider registered as `name` and its family.
pub fn lookup_decider(name: &str) -> Result<(Arc<dyn Decider>, Family)> {
    let r = DECIDERS
        .read()
        .map_err(|_| miette!("decider registry poisoned"))?;
    match r.get(name) {
        Some(reg) => Ok((reg.decider.clone(), reg.info.family)),
        None => {
            drop(r);
            Err(not_found(name))
        }
    }
}

/// Install a fitted probe or calibration for `(model, template)`
/// (process-wide; replaces any previous one).
pub fn install_fitted(model: &str, template: Template, fitted: Fitted) {
    if let Ok(mut f) = FITTED.write() {
        f.insert((model.to_string(), template.name().to_string()), Arc::new(fitted));
    }
}

/// Remove the fitted override of `(model, template)`; returns whether one
/// existed.
pub fn remove_fitted(model: &str, template: Template) -> bool {
    FITTED
        .write()
        .map(|mut f| f.remove(&(model.to_string(), template.name().to_string())).is_some())
        .unwrap_or(false)
}

/// The fitted override of `(model, template)`, if any.
pub fn fitted_for(model: &str, template: Template) -> Option<Arc<Fitted>> {
    FITTED
        .read()
        .ok()
        .and_then(|f| f.get(&(model.to_string(), template.name().to_string())).cloned())
}

/// Every installed fit: `(model, template, fit)`.
pub fn fitted_all() -> Vec<(String, String, Arc<Fitted>)> {
    let mut v: Vec<_> = FITTED
        .read()
        .map(|f| f.iter().map(|((m, t), x)| (m.clone(), t.clone(), x.clone())).collect())
        .unwrap_or_default();
    v.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
    v
}

/// Decide `items` with the decider registered as `name`: its family's
/// template, the installed probe for (model, template) if any (and its
/// template version matches), else an installed calibration, else the
/// shipped default calibration.
pub fn decide(name: &str, template: Template, items: &[(&str, &str)]) -> Result<Vec<Decision>> {
    let (decider, family) = lookup_decider(name)?;
    let model = decider.model_id();
    let version = template.version(family);
    match fitted_for(&model, template) {
        Some(f) if f.template_version() == version => match &*f {
            Fitted::Probe(p) => decide_with(&*decider, family, template, items, &Calibration::default(), Some(p)),
            Fitted::Calibration { calibration, .. } => {
                decide_with(&*decider, family, template, items, calibration, None)
            }
        },
        _ => decide_with(
            &*decider,
            family,
            template,
            items,
            &Calibration::default_for(template, family),
            None,
        ),
    }
}

/// Decide one pair (`b` is ignored by `important`).
pub fn decide_one(name: &str, template: Template, a: &str, b: &str) -> Result<Decision> {
    Ok(decide(name, template, &[(a, b)])?.remove(0))
}

// ------------------------------------------------------------------ scalar functions

fn str_arg<'a>(fname: &str, what: &str, v: &'a DataValue) -> Result<Option<&'a str>> {
    match v {
        DataValue::Str(s) => Ok(Some(s.as_str())),
        DataValue::Null => Ok(None),
        v => bail!("'{fname}' requires {what} as a string, got {v:?}"),
    }
}

/// Shared body of the `decide_*` scalar functions. `fixed` is the template
/// of the per-template shortcuts; otherwise the template is argument 2.
/// Returns `None` when a text is null.
fn scalar_decide(fname: &str, args: &[DataValue], fixed: Option<Template>) -> Result<Option<Decision>> {
    let name = match &args[0] {
        DataValue::Str(s) => s.to_string(),
        v => bail!("'{fname}' requires the decider name as a string, got {v:?}"),
    };
    let (template, rest) = match fixed {
        Some(t) => (t, &args[1..]),
        None => {
            let t = match &args[1] {
                DataValue::Str(s) => Template::parse(s)?,
                v => bail!("'{fname}' requires the template name as a string, got {v:?}"),
            };
            (t, &args[2..])
        }
    };
    if rest.len() != template.arity() {
        bail!(
            "'{fname}' with template {} takes {} text argument(s), got {}",
            template.name(),
            template.arity(),
            rest.len()
        );
    }
    let Some(a) = str_arg(fname, "the first text", &rest[0])? else {
        return Ok(None);
    };
    let b = match rest.get(1) {
        Some(v) => match str_arg(fname, "the second text", v)? {
            Some(b) => b,
            None => return Ok(None),
        },
        None => "",
    };
    Ok(Some(decide_one(&name, template, a, b)?))
}

macro_rules! decide_fn {
    ($opname:ident, $name:literal, $fname:ident, $min:expr, $vararg:expr, $fixed:expr, $out:expr) => {
        pub(crate) const $opname: Op = Op {
            name: $name,
            min_arity: $min,
            vararg: $vararg,
            inner: $fname,
        };
        fn $fname(args: &[DataValue]) -> Result<DataValue> {
            let fname = &$name[3..].to_ascii_lowercase();
            Ok(match scalar_decide(fname, args, $fixed)? {
                None => DataValue::Null,
                Some(d) => $out(d),
            })
        }
    };
}

fn out_p(d: Decision) -> DataValue {
    DataValue::from(d.p)
}
fn out_label(d: Decision) -> DataValue {
    DataValue::from(d.label.as_str())
}
fn out_band(d: Decision) -> DataValue {
    DataValue::from(d.band.name())
}

decide_fn!(OP_DECIDE_P, "OP_DECIDE_P", op_decide_p, 3, true, None, out_p);
decide_fn!(OP_DECIDE_LABEL, "OP_DECIDE_LABEL", op_decide_label, 3, true, None, out_label);
decide_fn!(OP_DECIDE_BAND, "OP_DECIDE_BAND", op_decide_band, 3, true, None, out_band);
decide_fn!(OP_DECIDE_SUPERSEDES, "OP_DECIDE_SUPERSEDES", op_decide_supersedes, 3, false, Some(Template::Supersedes), out_p);
decide_fn!(OP_DECIDE_SAME_ENTITY, "OP_DECIDE_SAME_ENTITY", op_decide_same_entity, 3, false, Some(Template::SameEntity), out_p);
decide_fn!(OP_DECIDE_DUPLICATE, "OP_DECIDE_DUPLICATE", op_decide_duplicate, 3, false, Some(Template::Duplicate), out_p);
decide_fn!(OP_DECIDE_CONTRADICTS, "OP_DECIDE_CONTRADICTS", op_decide_contradicts, 3, false, Some(Template::Contradicts), out_p);
decide_fn!(OP_DECIDE_IMPORTANT, "OP_DECIDE_IMPORTANT", op_decide_important, 2, false, Some(Template::Important), out_p);
decide_fn!(OP_DECIDE_RELATION, "OP_DECIDE_RELATION", op_decide_relation, 3, false, Some(Template::Relation), out_label);
