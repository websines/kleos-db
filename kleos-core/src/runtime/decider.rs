/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! `::decider list | enable | disable | fit | fitted | forget` (kleos fork;
//! docs/specs/decide.md). The decider registry and the installed fits are
//! process-wide; fits are also persisted in this database's
//! `kleos_decider_fits` relation and re-installed when it opens.

use std::collections::BTreeMap;

use miette::{IntoDiagnostic, Result};

use crate::bail;
use crate::data::value::DataValue;
use crate::decide::{
    self, fit_calibration, fit_probe, label_index, score_items, Calibration, DeciderSpec,
    DeciderStatus, Decision, Fitted, ProbeOptions, Template,
};
use crate::parse::sys::{DeciderOp, FitOpts};
use crate::runtime::db::{Db, ScriptMutability};
use crate::storage::Storage;
use crate::NamedRows;

/// Relation holding this database's fitted probes and calibrations.
pub const DECIDER_FITS: &str = "kleos_decider_fits";

fn opt_str(s: &Option<String>) -> DataValue {
    match s {
        Some(s) => DataValue::from(s.as_str()),
        None => DataValue::Null,
    }
}

fn list_rows(list: Vec<DeciderStatus>) -> NamedRows {
    NamedRows::new(
        ["name", "kind", "family", "model", "enabled", "loaded", "precision", "device"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        list.into_iter()
            .map(|d| {
                vec![
                    DataValue::from(d.name.as_str()),
                    DataValue::from(d.kind.as_str()),
                    DataValue::from(d.family.as_str()),
                    DataValue::from(d.model.as_str()),
                    DataValue::Bool(d.enabled),
                    d.loaded.map(DataValue::Bool).unwrap_or(DataValue::Null),
                    opt_str(&d.precision),
                    opt_str(&d.device),
                ]
            })
            .collect(),
    )
}

fn fitted_rows() -> NamedRows {
    NamedRows::new(
        ["model", "template", "kind", "template_version", "n", "threshold", "tau_auto", "source"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        decide::fitted_all()
            .into_iter()
            .map(|(m, t, f)| {
                let (n, thr, tau, src) = match &*f {
                    Fitted::Probe(p) => (
                        DataValue::from(p.n as i64),
                        p.threshold,
                        p.tau_auto,
                        format!("probe C={} cv={:.3}", p.c, p.cv_score),
                    ),
                    Fitted::Calibration { calibration, .. } => (
                        DataValue::Null,
                        calibration.threshold,
                        calibration.tau_auto,
                        calibration.source.clone(),
                    ),
                };
                vec![
                    DataValue::from(m.as_str()),
                    DataValue::from(t.as_str()),
                    DataValue::from(f.kind()),
                    DataValue::from(f.template_version() as i64),
                    n,
                    DataValue::from(thr),
                    DataValue::from(tau),
                    DataValue::from(src.as_str()),
                ]
            })
            .collect(),
    )
}

fn status(s: &str) -> NamedRows {
    NamedRows::new(vec!["status".to_string()], vec![vec![DataValue::from(s)]])
}

/// What `::decider fit` / [`Db::fit_decider`] report.
#[derive(Debug, Clone, PartialEq)]
pub struct FitReport {
    /// Model id the fit is keyed on.
    pub model: String,
    /// Template name.
    pub template: String,
    /// `probe` or `calibration`.
    pub kind: String,
    /// Labelled items used.
    pub n: usize,
    /// Cross-validated AUROC / accuracy (probe), else NaN.
    pub cv_score: f64,
    /// Label threshold.
    pub threshold: f64,
    /// Auto threshold.
    pub tau_auto: f64,
    /// Share of the labelled items auto-decided at `tau_auto`.
    pub coverage: f64,
}

impl FitReport {
    fn rows(&self) -> NamedRows {
        NamedRows::new(
            ["model", "template", "kind", "n", "cv_score", "threshold", "tau_auto", "coverage"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            vec![vec![
                DataValue::from(self.model.as_str()),
                DataValue::from(self.template.as_str()),
                DataValue::from(self.kind.as_str()),
                DataValue::from(self.n as i64),
                if self.cv_score.is_nan() {
                    DataValue::Null
                } else {
                    DataValue::from(self.cv_score)
                },
                DataValue::from(self.threshold),
                DataValue::from(self.tau_auto),
                DataValue::from(self.coverage),
            ]],
        )
    }
}

impl<'s, S: Storage<'s>> Db<S> {
    /// Turn a built-in decider on (process-wide); see
    /// [`crate::decide::enable_decider`].
    pub fn enable_decider(&self, name: &str, spec: DeciderSpec) -> Result<()> {
        decide::enable_decider(name, spec)
    }
    /// Turn a decider off (process-wide).
    pub fn disable_decider(&self, name: &str) -> Result<bool> {
        decide::disable_decider(name)
    }
    /// Every enabled and disabled decider (process-wide).
    pub fn deciders(&self) -> Vec<DeciderStatus> {
        decide::deciders()
    }
    /// Decide one pair with decider `name` (`b` ignored by `important`).
    pub fn decide(&self, name: &str, template: &str, a: &str, b: &str) -> Result<Decision> {
        decide::decide_one(name, Template::parse(template)?, a, b)
    }
    /// Decide many pairs in batches.
    pub fn decide_batch(&self, name: &str, template: &str, items: &[(&str, &str)]) -> Result<Vec<Decision>> {
        decide::decide(name, Template::parse(template)?, items)
    }

    /// Run a `::decider` op. `in_tx`: on the imperative path, where a write
    /// transaction is open and `fit` / `forget` (which write) cannot run.
    pub(crate) fn run_decider_op(&'s self, op: &DeciderOp, read_only: bool, in_tx: bool) -> Result<NamedRows> {
        if read_only && !matches!(op, DeciderOp::List | DeciderOp::Fitted) {
            bail!("`::decider enable/disable/fit/forget` cannot run in read-only mode");
        }
        match op {
            DeciderOp::List => Ok(list_rows(decide::deciders())),
            DeciderOp::Fitted => Ok(fitted_rows()),
            DeciderOp::Enable(name, spec) => {
                decide::enable_decider(name, spec.clone())?;
                Ok(status("OK"))
            }
            DeciderOp::Disable(name) => Ok(status(if decide::disable_decider(name)? {
                "OK"
            } else {
                "ALREADY_DISABLED"
            })),
            DeciderOp::Fit {
                name,
                template,
                relation,
                opts,
            } => {
                if in_tx {
                    bail!("`::decider fit` must run as a script of its own, not inside an imperative block");
                }
                Ok(self.fit_decider_on(name, template, &relation.name, opts)?.rows())
            }
            DeciderOp::Forget(name, template) => {
                if in_tx {
                    bail!("`::decider forget` must run as a script of its own, not inside an imperative block");
                }
                Ok(status(if self.forget_decider_fit(name, template)? {
                    "OK"
                } else {
                    "NOT_FITTED"
                }))
            }
        }
    }

    /// Fit a logistic probe (or, with `probe = false`, a zero-shot
    /// calibration) for decider `name` and `template` on the labelled stored
    /// relation `relation`: its last column is the label, the
    /// `template.arity()` columns before it the texts. The fit is installed
    /// for the decider's model (process-wide) and persisted in
    /// `kleos_decider_fits`, so it is applied again after a restart.
    pub fn fit_decider(&'s self, name: &str, template: &str, relation: &str, probe: bool, alpha: f64) -> Result<FitReport> {
        self.fit_decider_on(
            name,
            template,
            relation,
            &FitOpts {
                probe,
                alpha,
                wilson: false,
            },
        )
    }

    fn fit_decider_on(&'s self, name: &str, template: &str, relation: &str, opts: &FitOpts) -> Result<FitReport> {
        let template = Template::parse(template)?;
        let (decider, family) = decide::lookup_decider(name)?;
        let model = decider.model_id();
        let handle = {
            let tx = self.transact()?;
            tx.get_relation(relation, false)?
        };
        let cols: Vec<String> = handle
            .metadata
            .keys
            .iter()
            .chain(handle.metadata.non_keys.iter())
            .map(|c| c.name.to_string())
            .collect();
        let need = template.arity() + 1;
        if cols.len() < need {
            bail!(
                "`::decider fit` with template {} needs a relation whose last {need} columns are [{}label]; {relation} has {} column(s)",
                template.name(),
                if need == 2 { "text, " } else { "a, b, " },
                cols.len()
            );
        }
        let use_cols = cols[cols.len() - need..].join(", ");
        let rows = self
            .run_script(
                &format!("?[{use_cols}] := *{relation}{{{use_cols}}}"),
                Default::default(),
                ScriptMutability::Immutable,
            )?
            .rows;
        let mut texts: Vec<(String, String)> = vec![];
        let mut gold = vec![];
        for r in rows {
            let t = |v: &DataValue| -> Result<String> {
                match v {
                    DataValue::Str(s) => Ok(s.to_string()),
                    v => bail!("`::decider fit`: text columns must be strings, got {v:?}"),
                }
            };
            let a = t(&r[0])?;
            let b = if need == 3 { t(&r[1])? } else { String::new() };
            gold.push(label_index(template, &r[need - 1])?);
            texts.push((a, b));
        }
        if texts.len() < 4 {
            bail!("`::decider fit` needs at least 4 labelled rows, {relation} has {}", texts.len());
        }
        let items: Vec<(&str, &str)> = texts.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        let scored = score_items(&*decider, family, template, &items, &Calibration::default(), opts.probe)?;
        let (fitted, report) = if opts.probe {
            // symmetric templates: each order is a training row
            let mut xs = vec![];
            let mut ys = vec![];
            for (s, g) in scored.iter().zip(&gold) {
                for f in &s.features {
                    xs.push(f.clone());
                    ys.push(*g);
                }
            }
            let p = fit_probe(
                template,
                family,
                &model,
                &xs,
                &ys,
                &ProbeOptions {
                    alpha: opts.alpha,
                    wilson: opts.wilson,
                    ..Default::default()
                },
            )?;
            let report = FitReport {
                model: model.clone(),
                template: template.name().to_string(),
                kind: "probe".to_string(),
                n: texts.len(),
                cv_score: p.cv_score,
                threshold: p.threshold,
                tau_auto: p.tau_auto,
                coverage: p.coverage,
            };
            (Fitted::Probe(p), report)
        } else {
            let probs: Vec<Vec<f64>> = scored.iter().map(|s| s.probs.clone()).collect();
            let ops: Vec<Vec<f64>> = scored.iter().map(|s| s.option_probs.clone()).collect();
            let cal = fit_calibration(template, family, &probs, Some(&ops), &gold, opts.alpha, opts.wilson);
            // coverage of the fitted auto threshold on the same items
            let mut auto = 0;
            for s in &scored {
                let d = decide::finish(template, family, &model, recalibrated(&s.option_probs, template, family, &cal), &cal, "calibrated");
                if d.band == decide::Band::Auto {
                    auto += 1;
                }
            }
            let report = FitReport {
                model: model.clone(),
                template: template.name().to_string(),
                kind: "calibration".to_string(),
                n: texts.len(),
                cv_score: f64::NAN,
                threshold: cal.threshold,
                tau_auto: cal.tau_auto,
                coverage: auto as f64 / texts.len() as f64,
            };
            (
                Fitted::Calibration {
                    template_version: template.version(family),
                    calibration: cal,
                },
                report,
            )
        };
        self.persist_fit(name, &model, template, &fitted)?;
        decide::install_fitted(&model, template, fitted);
        Ok(report)
    }

    fn persist_fit(&'s self, decider: &str, model: &str, template: Template, fitted: &Fitted) -> Result<()> {
        let exists = {
            let tx = self.transact()?;
            tx.get_relation(DECIDER_FITS, false).is_ok()
        };
        if !exists {
            self.run_script(
                &format!(
                    ":create {DECIDER_FITS} {{model: String, template: String => kind: String, \
                     template_version: Int, decider: String, payload: String, created_at: Float}}"
                ),
                Default::default(),
                ScriptMutability::Mutable,
            )?;
        }
        let payload = serde_json::to_string(fitted).into_diagnostic()?;
        let params = BTreeMap::from([
            ("model".to_string(), DataValue::from(model)),
            ("template".to_string(), DataValue::from(template.name())),
            ("kind".to_string(), DataValue::from(fitted.kind())),
            ("tv".to_string(), DataValue::from(fitted.template_version() as i64)),
            ("decider".to_string(), DataValue::from(decider)),
            ("payload".to_string(), DataValue::from(payload.as_str())),
        ]);
        self.run_script(
            &format!(
                "?[model, template, kind, template_version, decider, payload, created_at] <- \
                 [[$model, $template, $kind, $tv, $decider, $payload, now()]] :put {DECIDER_FITS}"
            ),
            params,
            ScriptMutability::Mutable,
        )?;
        Ok(())
    }

    /// Remove the fit of decider `name`'s model for `template`, from the
    /// process and from this database. Returns whether one existed.
    pub fn forget_decider_fit(&'s self, name: &str, template: &str) -> Result<bool> {
        let template = Template::parse(template)?;
        let (decider, _) = decide::lookup_decider(name)?;
        let model = decider.model_id();
        let had = decide::remove_fitted(&model, template);
        let exists = {
            let tx = self.transact()?;
            tx.get_relation(DECIDER_FITS, false).is_ok()
        };
        if exists {
            let params = BTreeMap::from([
                ("model".to_string(), DataValue::from(model.as_str())),
                ("template".to_string(), DataValue::from(template.name())),
            ]);
            self.run_script(
                &format!("?[model, template] <- [[$model, $template]] :rm {DECIDER_FITS}"),
                params,
                ScriptMutability::Mutable,
            )?;
        }
        Ok(had)
    }

    /// Install every fit persisted in this database (called on open; errors
    /// are ignored so a damaged row never prevents opening).
    pub(crate) fn load_decider_fits(&'s self) {
        let exists = match self.transact() {
            Ok(tx) => tx.get_relation(DECIDER_FITS, false).is_ok(),
            Err(_) => false,
        };
        if !exists {
            return;
        }
        let Ok(r) = self.run_script(
            &format!("?[model, template, payload] := *{DECIDER_FITS}{{model, template, payload}}"),
            Default::default(),
            ScriptMutability::Immutable,
        ) else {
            return;
        };
        for row in r.rows {
            let (DataValue::Str(m), DataValue::Str(t), DataValue::Str(p)) = (&row[0], &row[1], &row[2]) else {
                continue;
            };
            let (Ok(t), Ok(f)) = (Template::parse(t), serde_json::from_str::<Fitted>(p)) else {
                continue;
            };
            decide::install_fitted(m, t, f);
        }
    }
}

/// Label probabilities from raw option probabilities under `cal`'s bias.
fn recalibrated(option_probs: &[f64], template: Template, family: decide::Family, cal: &Calibration) -> Vec<f64> {
    let spec = template.spec(family);
    let z: Vec<f64> = option_probs
        .iter()
        .enumerate()
        .map(|(i, p)| (p + 1e-12).ln() + cal.bias.get(i).copied().unwrap_or(0.0))
        .collect();
    let q = decide::softmax(&z);
    let mut out = vec![0.0; template.labels().len()];
    for (i, x) in q.into_iter().enumerate() {
        out[spec.label_of_option[i]] += x;
    }
    out
}
