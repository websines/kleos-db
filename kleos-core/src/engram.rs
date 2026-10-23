/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Engram overlay store (kleos fork; docs/specs/engram-overlay.md, feature
//! `engram`).
//!
//! A model with DeepSeek-Engram-style N-gram memory reads, at every token,
//! a few rows of large hashed tables. kleos stores per-user / per-agent
//! **row edits** to those tables as versioned data:
//!
//! - `kleos_engram_models {name => fp, row_dim, layers, table_rows, spec, ...}`:
//!   registered models, keyed by name, carrying the addressing fingerprint;
//! - `kleos_engram_facts {fp, ns, fact, vt: Validity => text, target,
//!   phrasings, rows, decided_by, source, payload}`: one record per fact,
//!   with the rows it touched (provenance) and method state for replay;
//! - `kleos_engram_rows {fp, ns, layer, row, vt: Validity => vec, facts}`:
//!   replacement row values (bf16 bytes) and the facts each depends on.
//!
//! Valid time (`Validity`, as in E1 supersession) gives supersession and
//! time travel; erase deletes every version. A row id means nothing without
//! the model's fingerprint, so every read and write names it and a mismatch
//! is refused.

#[cfg(feature = "engram")]
pub use imp::*;

#[cfg(feature = "engram")]
pub use kleos_engram as addressing;

#[cfg(not(feature = "engram"))]
pub(crate) fn run_engram_op<'s, S: crate::Storage<'s>>(
    _db: &'s crate::Db<S>,
    _op: &crate::parse::sys::EngramOp,
    _read_only: bool,
    _in_tx: bool,
) -> miette::Result<crate::NamedRows> {
    #[derive(Debug, thiserror::Error, miette::Diagnostic)]
    #[error("`::engram` needs a kleos built with the `engram` feature")]
    #[diagnostic(code(eval::engram_feature_missing))]
    struct Missing;
    Err(Missing.into())
}

#[cfg(feature = "engram")]
mod imp {
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    use itertools::Itertools;
    use kleos_engram::safetensors::OverlayLayer;
    use kleos_engram::{bf16, EngramSpec, SpecOverrides};
    use lazy_static::lazy_static;
    use miette::{miette, Diagnostic, IntoDiagnostic, Result};
    use serde::{Deserialize, Serialize};
    use thiserror::Error;

    use crate::bail;
    use crate::data::json::JsonValue;
    use crate::data::value::DataValue;
    use crate::parse::sys::EngramOp;
    use crate::runtime::db::{Db, ScriptMutability};
    use crate::storage::Storage;
    use crate::NamedRows;

    /// Registered models.
    pub const ENGRAM_MODELS: &str = "kleos_engram_models";
    /// Fact records.
    pub const ENGRAM_FACTS: &str = "kleos_engram_facts";
    /// Overlay rows.
    pub const ENGRAM_ROWS: &str = "kleos_engram_rows";

    lazy_static! {
        /// Loaded specs (with tokenizers) by fingerprint, for `::engram rows`.
        static ref SPECS: Mutex<HashMap<String, Arc<EngramSpec>>> = Mutex::new(HashMap::new());
    }

    #[derive(Debug, Error, Diagnostic)]
    #[error("Engram fingerprint mismatch: model '{model}' is registered as {registered}, but the {what} is for {given}; refusing")]
    #[diagnostic(code(eval::engram_fingerprint_mismatch))]
    struct FingerprintMismatch {
        model: String,
        registered: String,
        given: String,
        what: &'static str,
    }

    #[derive(Debug, Error, Diagnostic)]
    #[error("no Engram model named '{0}' is registered here; use `::engram register {0} {{config: ..., tokenizer: ...}}`")]
    #[diagnostic(code(eval::engram_model_not_found))]
    struct ModelNotFound(String);

    /// A registered model.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct EngramModel {
        /// Name it was registered under.
        pub name: String,
        /// Addressing fingerprint (`eg1:...`).
        pub fingerprint: String,
        /// Row dimension.
        pub row_dim: usize,
        /// Engram layer ids.
        pub layers: Vec<u64>,
        /// Table rows per layer (0 when unknown).
        pub table_rows: Vec<u64>,
    }

    /// One fact of an edit batch: what was learned, from where, and which
    /// rows it touched.
    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct EngramFact {
        /// Fact id, unique within the namespace.
        pub id: String,
        /// The fact as text.
        #[serde(default)]
        pub text: String,
        /// Its value (e.g. the target continuation).
        #[serde(default)]
        pub target: String,
        /// Phrasings the edit was trained on.
        #[serde(default)]
        pub phrasings: Vec<String>,
        /// `(layer, row)` pairs this fact's phrasings read: the rows it touched.
        #[serde(default)]
        pub rows: Vec<(u64, i64)>,
        /// Who decided to remember it (`model|template@v|via`, a user, ...).
        #[serde(default)]
        pub decided_by: String,
        /// Where it came from (episode id, document, ...).
        #[serde(default)]
        pub source: String,
        /// Method state needed to replay the edit without the original
        /// inputs (e.g. its target delta), stored as f32.
        #[serde(default)]
        pub payload: Option<Vec<f32>>,
    }

    /// One row value of an edit batch.
    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct RowEdit {
        /// Engram layer id.
        pub layer: u64,
        /// Row id in the layer's table.
        pub row: i64,
        /// Replacement value (stored as bf16).
        pub vec: Vec<f32>,
        /// Facts this value depends on (at least one).
        pub facts: Vec<String>,
    }

    /// An edit batch for one namespace.
    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct EngramWrite {
        /// Namespace (user, agent, org ...).
        pub ns: String,
        /// Fingerprint of the model the edit was computed for; must match.
        pub fingerprint: String,
        /// Valid time of the write, µs since the epoch (default now).
        #[serde(default)]
        pub at: Option<i64>,
        /// Facts asserted (a known id gets a new version).
        #[serde(default)]
        pub facts: Vec<EngramFact>,
        /// Row values.
        #[serde(default)]
        pub rows: Vec<RowEdit>,
        /// Facts whose validity ends at `at` (supersession).
        #[serde(default)]
        pub retract: Vec<String>,
    }

    /// One materialised overlay row.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct OverlayRow {
        /// Engram layer id.
        pub layer: u64,
        /// Row id.
        pub row: i64,
        /// Namespace the value came from.
        pub ns: String,
        /// Facts it depends on.
        pub facts: Vec<String>,
        /// bf16 bits.
        pub vec: Vec<u16>,
    }

    /// A materialised overlay: what an engine reads instead of base rows.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Overlay {
        /// Model fingerprint.
        pub fingerprint: String,
        /// Row dimension.
        pub row_dim: usize,
        /// Valid time it was materialised at (µs).
        pub as_of: i64,
        /// Namespace stack, most general first.
        pub namespaces: Vec<String>,
        /// Rows sorted by (layer, row).
        pub rows: Vec<OverlayRow>,
    }

    #[derive(Clone, Debug)]
    struct FactRec {
        at: i64,
        text: String,
        target: String,
        phrasings: Vec<String>,
        rows: Vec<(u64, i64)>,
        decided_by: String,
        source: String,
        payload: Option<Vec<u8>>,
    }

    #[derive(Clone, Debug)]
    struct RowRec {
        vec: Vec<u8>,
        facts: Vec<String>,
    }

    fn now_us() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0)
    }

    fn s(v: &str) -> DataValue {
        DataValue::from(v)
    }

    fn strs(v: &[String]) -> DataValue {
        DataValue::List(v.iter().map(|x| s(x)).collect())
    }

    fn vld(ts: i64, assert: bool) -> DataValue {
        DataValue::List(vec![DataValue::from(ts), DataValue::Bool(assert)])
    }

    fn get_strs(v: &DataValue) -> Vec<String> {
        match v {
            DataValue::List(l) => l.iter().filter_map(|x| x.get_str().map(str::to_string)).collect(),
            _ => vec![],
        }
    }

    fn get_bytes(v: &DataValue) -> Option<Vec<u8>> {
        match v {
            DataValue::Bytes(b) => Some(b.clone()),
            _ => None,
        }
    }

    fn pairs(v: &DataValue) -> Vec<(u64, i64)> {
        match v {
            DataValue::List(l) => l
                .iter()
                .filter_map(|x| x.get_int())
                .tuples()
                .map(|(a, b)| (a as u64, b))
                .collect(),
            _ => vec![],
        }
    }

    fn flat_pairs(p: &[(u64, i64)]) -> DataValue {
        DataValue::List(p.iter().flat_map(|&(l, r)| [DataValue::from(l as i64), DataValue::from(r)]).collect())
    }

    fn f32_bytes(x: &[f32]) -> Vec<u8> {
        x.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn status_rows(cols: &[&str], rows: Vec<Vec<DataValue>>) -> NamedRows {
        NamedRows::new(cols.iter().map(|c| c.to_string()).collect(), rows)
    }

    fn rel_exists<'s, S: Storage<'s>>(db: &'s Db<S>, name: &str) -> Result<bool> {
        let tx = db.transact()?;
        Ok(tx.get_relation(name, false).is_ok())
    }

    fn ensure_relations<'s, S: Storage<'s>>(db: &'s Db<S>) -> Result<()> {
        let defs = [
            (
                ENGRAM_MODELS,
                "{name: String => fp: String, row_dim: Int, layers: [Int], table_rows: [Int], spec: Json, \
                 config_path: String?, tokenizer_path: String?, pad_id: Int?, created_at: Float}",
            ),
            (
                ENGRAM_FACTS,
                "{fp: String, ns: String, fact: String, vt: Validity => text: String, target: String, \
                 phrasings: [String], rows: [Int], decided_by: String, source: String, payload: Bytes?}",
            ),
            (ENGRAM_ROWS, "{fp: String, ns: String, layer: Int, row: Int, vt: Validity => vec: Bytes, facts: [String]}"),
        ];
        for (name, schema) in defs {
            if !rel_exists(db, name)? {
                db.run_script(&format!(":create {name} {schema}"), Default::default(), ScriptMutability::Mutable)?;
            }
        }
        Ok(())
    }

    fn query<'s, S: Storage<'s>>(db: &'s Db<S>, script: &str, params: BTreeMap<String, DataValue>) -> Result<Vec<Vec<DataValue>>> {
        Ok(db.run_script(script, params, ScriptMutability::Immutable)?.rows)
    }

    fn params<const N: usize>(kv: [(&str, DataValue); N]) -> BTreeMap<String, DataValue> {
        kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }

    /// Look up a registered model.
    pub fn model<'s, S: Storage<'s>>(db: &'s Db<S>, name: &str) -> Result<EngramModel> {
        if !rel_exists(db, ENGRAM_MODELS)? {
            return Err(ModelNotFound(name.to_string()).into());
        }
        let rows = query(
            db,
            &format!("?[fp, row_dim, layers, table_rows] := *{ENGRAM_MODELS}{{name: $name, fp, row_dim, layers, table_rows}}"),
            params([("name", s(name))]),
        )?;
        let r = rows.first().ok_or_else(|| ModelNotFound(name.to_string()))?;
        let ints = |v: &DataValue| -> Vec<u64> {
            match v {
                DataValue::List(l) => l.iter().filter_map(|x| x.get_int()).map(|x| x as u64).collect(),
                _ => vec![],
            }
        };
        Ok(EngramModel {
            name: name.to_string(),
            fingerprint: r[0].get_str().unwrap_or_default().to_string(),
            row_dim: r[1].get_int().unwrap_or(0) as usize,
            layers: ints(&r[2]),
            table_rows: ints(&r[3]),
        })
    }

    fn check_fp(m: &EngramModel, given: Option<&str>, what: &'static str) -> Result<()> {
        match given {
            Some(g) if g != m.fingerprint => Err(FingerprintMismatch {
                model: m.name.clone(),
                registered: m.fingerprint.clone(),
                given: g.to_string(),
                what,
            }
            .into()),
            _ => Ok(()),
        }
    }

    fn vt_parts(v: &DataValue) -> (i64, bool) {
        match v {
            DataValue::Validity(v) => (v.timestamp.0 .0, v.is_assert.0),
            _ => (0, false),
        }
    }

    fn facts_at<'s, S: Storage<'s>>(db: &'s Db<S>, fp: &str, ns: &str, t: i64) -> Result<BTreeMap<String, FactRec>> {
        if !rel_exists(db, ENGRAM_FACTS)? {
            return Ok(BTreeMap::new());
        }
        let p = params([("fp", s(fp)), ("ns", s(ns))]);
        let rows = query(
            db,
            &format!(
                "?[fact, text, target, phrasings, rows, decided_by, source, payload] := \
                 *{ENGRAM_FACTS}{{fp: $fp, ns: $ns, fact, text, target, phrasings, rows, decided_by, source, payload @ {t}}}"
            ),
            p.clone(),
        )?;
        // since = the latest assertion at or before t (the current version)
        let mut since: BTreeMap<String, i64> = BTreeMap::new();
        for r in query(db, &format!("?[fact, vt] := *{ENGRAM_FACTS}{{fp: $fp, ns: $ns, fact, vt}}"), p)? {
            let (ts, assert) = vt_parts(&r[1]);
            if assert && ts <= t {
                let e = since.entry(r[0].get_str().unwrap_or_default().to_string()).or_insert(ts);
                *e = (*e).max(ts);
            }
        }
        Ok(rows
            .into_iter()
            .map(|r| {
                let id = r[0].get_str().unwrap_or_default().to_string();
                (
                    id.clone(),
                    FactRec {
                        at: since.get(&id).copied().unwrap_or(0),
                        text: r[1].get_str().unwrap_or_default().to_string(),
                        target: r[2].get_str().unwrap_or_default().to_string(),
                        phrasings: get_strs(&r[3]),
                        rows: pairs(&r[4]),
                        decided_by: r[5].get_str().unwrap_or_default().to_string(),
                        source: r[6].get_str().unwrap_or_default().to_string(),
                        payload: get_bytes(&r[7]),
                    },
                )
            })
            .collect())
    }

    fn rows_at<'s, S: Storage<'s>>(db: &'s Db<S>, fp: &str, ns: &str, t: i64) -> Result<BTreeMap<(u64, i64), RowRec>> {
        if !rel_exists(db, ENGRAM_ROWS)? {
            return Ok(BTreeMap::new());
        }
        let rows = query(
            db,
            &format!("?[layer, row, vec, facts] := *{ENGRAM_ROWS}{{fp: $fp, ns: $ns, layer, row, vec, facts @ {t}}}"),
            params([("fp", s(fp)), ("ns", s(ns))]),
        )?;
        Ok(rows
            .into_iter()
            .map(|r| {
                (
                    (r[0].get_int().unwrap_or(0) as u64, r[1].get_int().unwrap_or(0)),
                    RowRec {
                        vec: get_bytes(&r[2]).unwrap_or_default(),
                        facts: get_strs(&r[3]),
                    },
                )
            })
            .collect())
    }

    fn validate_rows(m: &EngramModel, rows: &[RowEdit]) -> Result<()> {
        let mut seen = BTreeSet::new();
        for r in rows {
            let Some(li) = m.layers.iter().position(|&l| l == r.layer) else {
                bail!("Engram row ({}, {}): layer {} is not an Engram layer of '{}' ({:?})", r.layer, r.row, r.layer, m.name, m.layers);
            };
            let max = m.table_rows.get(li).copied().unwrap_or(0);
            if r.row < 0 || (max > 0 && r.row as u64 >= max) {
                bail!("Engram row ({}, {}) is outside layer {}'s table ({max} rows)", r.layer, r.row, r.layer);
            }
            if r.vec.len() != m.row_dim {
                bail!("Engram row ({}, {}) has {} values; '{}' rows have {}", r.layer, r.row, r.vec.len(), m.name, m.row_dim);
            }
            if r.vec.iter().any(|v| !v.is_finite()) {
                bail!("Engram row ({}, {}) has a non-finite value", r.layer, r.row);
            }
            if r.facts.is_empty() {
                bail!("Engram row ({}, {}) names no fact: every row value needs provenance", r.layer, r.row);
            }
            if !seen.insert((r.layer, r.row)) {
                bail!("Engram row ({}, {}) appears twice in one write", r.layer, r.row);
            }
        }
        Ok(())
    }

    fn row_put(fp: &str, ns: &str, r: &RowEdit, t: i64) -> Vec<DataValue> {
        vec![
            s(fp),
            s(ns),
            DataValue::from(r.layer as i64),
            DataValue::from(r.row),
            vld(t, true),
            DataValue::Bytes(bf16::to_le_bytes(&bf16::from_f32(&r.vec))),
            strs(&r.facts),
        ]
    }

    fn row_retract(fp: &str, ns: &str, k: (u64, i64), t: i64) -> Vec<DataValue> {
        vec![s(fp), s(ns), DataValue::from(k.0 as i64), DataValue::from(k.1), vld(t, false), DataValue::Bytes(vec![]), DataValue::List(vec![])]
    }

    fn fact_put(fp: &str, ns: &str, f: &EngramFact, t: i64) -> Vec<DataValue> {
        vec![
            s(fp),
            s(ns),
            s(&f.id),
            vld(t, true),
            s(&f.text),
            s(&f.target),
            strs(&f.phrasings),
            flat_pairs(&f.rows),
            s(&f.decided_by),
            s(&f.source),
            f.payload.as_ref().map_or(DataValue::Null, |p| DataValue::Bytes(f32_bytes(p))),
        ]
    }

    fn fact_retract(fp: &str, ns: &str, id: &str, f: &FactRec, t: i64) -> Vec<DataValue> {
        vec![
            s(fp),
            s(ns),
            s(id),
            vld(t, false),
            s(&f.text),
            s(&f.target),
            strs(&f.phrasings),
            flat_pairs(&f.rows),
            s(&f.decided_by),
            s(&f.source),
            f.payload.clone().map_or(DataValue::Null, DataValue::Bytes),
        ]
    }

    const FACT_COLS: &str = "fp, ns, fact, vt, text, target, phrasings, rows, decided_by, source, payload";
    const ROW_COLS: &str = "fp, ns, layer, row, vt, vec, facts";

    /// Run the puts / removals of one op as a single script (one transaction).
    fn commit<'s, S: Storage<'s>>(
        db: &'s Db<S>,
        fact_puts: Vec<Vec<DataValue>>,
        row_puts: Vec<Vec<DataValue>>,
        fact_rms: Vec<Vec<DataValue>>,
        row_rms: Vec<Vec<DataValue>>,
    ) -> Result<()> {
        let mut blocks = vec![];
        let mut p = BTreeMap::new();
        if !fact_rms.is_empty() {
            blocks.push(format!("{{?[fp, ns, fact, vt] <- $frm :rm {ENGRAM_FACTS} {{fp, ns, fact, vt}}}}"));
            p.insert("frm".to_string(), DataValue::List(fact_rms.into_iter().map(DataValue::List).collect()));
        }
        if !row_rms.is_empty() {
            blocks.push(format!("{{?[fp, ns, layer, row, vt] <- $rrm :rm {ENGRAM_ROWS} {{fp, ns, layer, row, vt}}}}"));
            p.insert("rrm".to_string(), DataValue::List(row_rms.into_iter().map(DataValue::List).collect()));
        }
        if !fact_puts.is_empty() {
            blocks.push(format!("{{?[{FACT_COLS}] <- $fput :put {ENGRAM_FACTS} {{{FACT_COLS}}}}}"));
            p.insert("fput".to_string(), DataValue::List(fact_puts.into_iter().map(DataValue::List).collect()));
        }
        if !row_puts.is_empty() {
            blocks.push(format!("{{?[{ROW_COLS}] <- $rput :put {ENGRAM_ROWS} {{{ROW_COLS}}}}}"));
            p.insert("rput".to_string(), DataValue::List(row_puts.into_iter().map(DataValue::List).collect()));
        }
        if blocks.is_empty() {
            return Ok(());
        }
        db.run_script(&blocks.join("\n"), p, ScriptMutability::Mutable)?;
        Ok(())
    }

    impl<'s, S: Storage<'s>> Db<S> {
        /// Register a model's Engram addressing under `name`. Re-registering
        /// the same name with the same fingerprint is a no-op; a different
        /// fingerprint is refused (the spec drifted). Returns the fingerprint.
        pub fn engram_register(&'s self, name: &str, spec: &EngramSpec, paths: Option<(&Path, &Path)>) -> Result<String> {
            let summary = spec.summary();
            let table_rows: Vec<u64> = spec.mapping.layers.iter().map(|l| l.total_rows).collect();
            let r = self.engram_register_raw(
                name,
                spec.fingerprint(),
                spec.row_dim,
                &spec.layer_ids(),
                &table_rows,
                summary,
                paths,
                Some(spec.mapping.cfg.pad_id),
            )?;
            SPECS.lock().unwrap().insert(spec.fingerprint().to_string(), Arc::new(spec.clone()));
            Ok(r)
        }

        /// Register a model by fingerprint alone (computed elsewhere, e.g. by
        /// the `kleos-engram` CLI). kleos then cannot hash text for it.
        #[allow(clippy::too_many_arguments)]
        pub fn engram_register_raw(
            &'s self,
            name: &str,
            fingerprint: &str,
            row_dim: usize,
            layers: &[u64],
            table_rows: &[u64],
            spec: serde_json::Value,
            paths: Option<(&Path, &Path)>,
            pad_id: Option<u32>,
        ) -> Result<String> {
            if !fingerprint.starts_with("eg1:") {
                bail!("Engram fingerprints look like eg1:<32 hex>, got '{fingerprint}'");
            }
            if row_dim == 0 || layers.is_empty() {
                bail!("an Engram model needs a positive row_dim and at least one layer");
            }
            ensure_relations(self)?;
            if let Ok(m) = model(self, name) {
                check_fp(&m, Some(fingerprint), "new registration")?;
                return Ok(m.fingerprint);
            }
            let ints = |v: &[u64]| DataValue::List(v.iter().map(|&x| DataValue::from(x as i64)).collect());
            let path = |p: Option<&Path>| p.map_or(DataValue::Null, |p| s(&p.to_string_lossy()));
            let p = params([
                ("name", s(name)),
                ("fp", s(fingerprint)),
                ("row_dim", DataValue::from(row_dim as i64)),
                ("layers", ints(layers)),
                ("table_rows", ints(table_rows)),
                ("spec", DataValue::Json(crate::JsonData(spec))),
                ("cp", path(paths.map(|p| p.0))),
                ("tp", path(paths.map(|p| p.1))),
                ("pad", pad_id.map_or(DataValue::Null, |x| DataValue::from(x as i64))),
            ]);
            self.run_script(
                &format!(
                    "?[name, fp, row_dim, layers, table_rows, spec, config_path, tokenizer_path, pad_id, created_at] <- \
                     [[$name, $fp, $row_dim, $layers, $table_rows, $spec, $cp, $tp, $pad, now()]] \
                     :put {ENGRAM_MODELS} {{name => fp, row_dim, layers, table_rows, spec, config_path, tokenizer_path, pad_id, created_at}}"
                ),
                p,
                ScriptMutability::Mutable,
            )?;
            Ok(fingerprint.to_string())
        }

        /// Registered models.
        pub fn engram_models(&'s self) -> Result<Vec<EngramModel>> {
            if !rel_exists(self, ENGRAM_MODELS)? {
                return Ok(vec![]);
            }
            let names = query(self, &format!("?[name] := *{ENGRAM_MODELS}{{name}}"), Default::default())?;
            names.iter().map(|r| model(self, r[0].get_str().unwrap_or_default())).collect()
        }

        /// Write an edit batch for `model` (see [`EngramWrite`]). Rows that
        /// depend on a retracted (or re-asserted) fact and are not rewritten in
        /// the batch are retracted at `at`, reverting to the base row; rows
        /// that still have other live facts are reported `needs_replay`.
        /// Output columns: `kind, id, action`.
        pub fn engram_write(&'s self, model_name: &str, w: &EngramWrite) -> Result<NamedRows> {
            let m = model(self, model_name)?;
            check_fp(&m, Some(&w.fingerprint), "edit")?;
            if w.ns.is_empty() {
                bail!("an Engram write needs a namespace (ns)");
            }
            validate_rows(&m, &w.rows)?;
            let t = w.at.unwrap_or_else(now_us);
            let fp = m.fingerprint.as_str();
            let live = facts_at(self, fp, &w.ns, t)?;
            let mut out = vec![];
            let new_ids: BTreeSet<&str> = w.facts.iter().map(|f| f.id.as_str()).collect();
            if new_ids.len() != w.facts.len() || w.facts.iter().any(|f| f.id.is_empty()) {
                bail!("Engram fact ids in one write must be non-empty and distinct");
            }
            for id in &w.retract {
                if new_ids.contains(id.as_str()) {
                    bail!("fact '{id}' is both asserted and retracted in one write");
                }
                if !live.contains_key(id) {
                    out.push(vec![s("fact"), s(id), s("not_current")]);
                }
            }
            // stale = facts whose earlier rows no longer hold after this write
            let stale: BTreeSet<&str> = w
                .retract
                .iter()
                .map(String::as_str)
                .chain(new_ids.iter().copied().filter(|id| live.contains_key(*id)))
                .collect();
            for r in &w.rows {
                for f in &r.facts {
                    let ok = new_ids.contains(f.as_str()) || (live.contains_key(f) && !w.retract.contains(f));
                    if !ok {
                        bail!("Engram row ({}, {}) depends on fact '{f}', which is neither in this write nor current in '{}'", r.layer, r.row, w.ns);
                    }
                }
            }
            let written: BTreeSet<(u64, i64)> = w.rows.iter().map(|r| (r.layer, r.row)).collect();
            let mut row_puts: Vec<Vec<DataValue>> = w.rows.iter().map(|r| row_put(fp, &w.ns, r, t)).collect();
            if !stale.is_empty() {
                for (k, rec) in rows_at(self, fp, &w.ns, t)? {
                    if written.contains(&k) || !rec.facts.iter().any(|f| stale.contains(f.as_str())) {
                        continue;
                    }
                    let others = rec.facts.iter().any(|f| !stale.contains(f.as_str()) && live.contains_key(f));
                    row_puts.push(row_retract(fp, &w.ns, k, t));
                    out.push(vec![s("row"), s(&format!("{}:{}", k.0, k.1)), s(if others { "needs_replay" } else { "reverted" })]);
                }
            }
            let mut fact_puts: Vec<Vec<DataValue>> = w.facts.iter().map(|f| fact_put(fp, &w.ns, f, t)).collect();
            for id in &w.retract {
                if let Some(rec) = live.get(id) {
                    fact_puts.push(fact_retract(fp, &w.ns, id, rec, t));
                    out.push(vec![s("fact"), s(id), s("retracted")]);
                }
            }
            for f in &w.facts {
                out.push(vec![s("fact"), s(&f.id), s(if live.contains_key(&f.id) { "updated" } else { "asserted" })]);
            }
            for r in &w.rows {
                out.push(vec![s("row"), s(&format!("{}:{}", r.layer, r.row)), s("written")]);
            }
            commit(self, fact_puts, row_puts, vec![], vec![])?;
            Ok(status_rows(&["kind", "id", "action"], out))
        }

        /// Supersede fact `old` with `new` in one batch at `at`: the old fact's
        /// validity ends, the new fact and its rows start. `as_of` an earlier
        /// time still materialises the old value.
        pub fn engram_supersede(&'s self, model_name: &str, old: &str, w: &EngramWrite) -> Result<NamedRows> {
            let mut w = w.clone();
            if !w.retract.iter().any(|r| r == old) {
                w.retract.push(old.to_string());
            }
            self.engram_write(model_name, &w)
        }

        /// Materialise the overlay of a namespace stack (most general first,
        /// e.g. `["org", "user", "agent"]`: a row in a later namespace
        /// overrides the same row in an earlier one) as of valid time `as_of`
        /// (µs, default now). `expect_fingerprint`, when given, must match.
        pub fn engram_materialize(
            &'s self,
            model_name: &str,
            namespaces: &[String],
            as_of: Option<i64>,
            expect_fingerprint: Option<&str>,
        ) -> Result<Overlay> {
            let m = model(self, model_name)?;
            check_fp(&m, expect_fingerprint, "overlay request")?;
            if namespaces.is_empty() {
                bail!("materialising an Engram overlay needs at least one namespace");
            }
            let t = as_of.unwrap_or_else(now_us);
            let mut merged: BTreeMap<(u64, i64), OverlayRow> = BTreeMap::new();
            if rel_exists(self, ENGRAM_ROWS)? {
                for ns in namespaces {
                    for (k, rec) in rows_at(self, &m.fingerprint, ns, t)? {
                        merged.insert(
                            k,
                            OverlayRow {
                                layer: k.0,
                                row: k.1,
                                ns: ns.clone(),
                                facts: rec.facts,
                                vec: bf16::from_le_bytes(&rec.vec),
                            },
                        );
                    }
                }
            }
            Ok(Overlay {
                fingerprint: m.fingerprint,
                row_dim: m.row_dim,
                as_of: t,
                namespaces: namespaces.to_vec(),
                rows: merged.into_values().collect(),
            })
        }

        /// Materialise and write one safetensors file per Engram layer into
        /// `dir` (`layer{L}.safetensors`: `row_ids` I64 [n], `values` BF16
        /// [n, row_dim], metadata with the fingerprint). Returns the paths.
        pub fn engram_export(
            &'s self,
            model_name: &str,
            namespaces: &[String],
            as_of: Option<i64>,
            dir: &Path,
            expect_fingerprint: Option<&str>,
        ) -> Result<Vec<(u64, PathBuf, usize)>> {
            let m = model(self, model_name)?;
            let ov = self.engram_materialize(model_name, namespaces, as_of, expect_fingerprint)?;
            std::fs::create_dir_all(dir).into_diagnostic()?;
            let mut out = vec![];
            for &layer in &m.layers {
                let rows: Vec<&OverlayRow> = ov.rows.iter().filter(|r| r.layer == layer).collect();
                let l = OverlayLayer {
                    layer,
                    row_dim: m.row_dim,
                    rows: rows.iter().map(|r| r.row).collect(),
                    values: rows.iter().flat_map(|r| r.vec.iter().copied()).collect(),
                };
                let meta = BTreeMap::from([
                    ("fingerprint".to_string(), m.fingerprint.clone()),
                    ("model".to_string(), m.name.clone()),
                    ("namespaces".to_string(), serde_json::to_string(namespaces).into_diagnostic()?),
                    ("as_of_us".to_string(), ov.as_of.to_string()),
                    ("mode".to_string(), "replace".to_string()),
                    ("writer".to_string(), format!("kleos {}", env!("CARGO_PKG_VERSION"))),
                ]);
                let path = dir.join(format!("layer{layer}.safetensors"));
                kleos_engram::safetensors::write_file(&path, &l, &meta).map_err(|e| miette!("{e}"))?;
                out.push((layer, path, l.rows.len()));
            }
            Ok(out)
        }

        /// Hard-erase a namespace: every version of every row and fact record
        /// under `model` (or under every model when `None`) is deleted. Returns
        /// (fact versions, row versions) deleted.
        pub fn engram_erase_namespace(&'s self, ns: &str, model_name: Option<&str>) -> Result<(usize, usize)> {
            if !rel_exists(self, ENGRAM_FACTS)? {
                return Ok((0, 0));
            }
            let fp = match model_name {
                Some(n) => Some(model(self, n)?.fingerprint),
                None => None,
            };
            let fpf = if fp.is_some() { ", fp == $fp" } else { "" };
            let p = params([("ns", s(ns)), ("fp", fp.as_deref().map_or(DataValue::Null, s))]);
            let facts = query(self, &format!("?[fp, ns, fact, vt] := *{ENGRAM_FACTS}{{fp, ns, fact, vt}}, ns == $ns{fpf}"), p.clone())?;
            let rows = query(self, &format!("?[fp, ns, layer, row, vt] := *{ENGRAM_ROWS}{{fp, ns, layer, row, vt}}, ns == $ns{fpf}"), p)?;
            let counts = (facts.len(), rows.len());
            commit(self, vec![], vec![], facts, rows)?;
            Ok(counts)
        }

        /// The replay plan for erasing one fact: each `(layer, row)` it
        /// touched that another current fact of the namespace also touches,
        /// with those facts. Only these rows need recomputing; rows touched by
        /// the erased fact alone simply revert to base.
        pub fn engram_erase_plan(&'s self, model_name: &str, ns: &str, fact: &str) -> Result<Vec<((u64, i64), Vec<String>)>> {
            let m = model(self, model_name)?;
            let t = now_us();
            let live = facts_at(self, &m.fingerprint, ns, t)?;
            let touched = self.touched_by(&m.fingerprint, ns, fact)?.0;
            let mut out = vec![];
            for k in touched {
                let others: Vec<String> =
                    live.iter().filter(|(id, f)| id.as_str() != fact && f.rows.contains(&k)).map(|(id, _)| id.clone()).collect();
                if !others.is_empty() {
                    out.push((k, others));
                }
            }
            Ok(out)
        }

        fn touched_by(&'s self, fp: &str, ns: &str, fact: &str) -> Result<(BTreeSet<(u64, i64)>, Vec<Vec<DataValue>>, Vec<Vec<DataValue>>)> {
            if !rel_exists(self, ENGRAM_FACTS)? {
                return Ok(Default::default());
            }
            let p = params([("fp", s(fp)), ("ns", s(ns)), ("fact", s(fact))]);
            let fvs = query(
                self,
                &format!("?[fp, ns, fact, vt, rows] := *{ENGRAM_FACTS}{{fp, ns, fact, vt, rows}}, fp == $fp, ns == $ns, fact == $fact"),
                p.clone(),
            )?;
            let rvs = query(
                self,
                &format!("?[fp, ns, layer, row, vt] := *{ENGRAM_ROWS}{{fp, ns, layer, row, vt, facts}}, fp == $fp, ns == $ns, is_in($fact, facts)"),
                p,
            )?;
            let mut touched: BTreeSet<(u64, i64)> = fvs.iter().flat_map(|r| pairs(&r[4])).collect();
            touched.extend(rvs.iter().map(|r| (r[2].get_int().unwrap_or(0) as u64, r[3].get_int().unwrap_or(0))));
            let fact_keys = fvs.into_iter().map(|mut r| {
                r.truncate(4);
                r
            });
            Ok((touched, fact_keys.collect(), rvs))
        }

        /// Hard-erase one fact by provenance replay. Every version of its
        /// record and every row version that depended on it is deleted.
        /// `replacements` are recomputed values for rows it touched, computed
        /// from the namespace's remaining facts (see [`engram_erase_plan`]);
        /// they must lie inside the touched set and must not depend on the
        /// erased fact, so rows it never touched stay bit-identical. Output
        /// rows: `(layer, row, action)` with `replayed`, `previous` (an
        /// earlier value without the fact is current again) or `reverted`
        /// (base row).
        ///
        /// [`engram_erase_plan`]: Db::engram_erase_plan
        pub fn engram_erase_fact(
            &'s self,
            model_name: &str,
            ns: &str,
            fact: &str,
            replacements: &[RowEdit],
            at: Option<i64>,
        ) -> Result<Vec<((u64, i64), &'static str)>> {
            let m = model(self, model_name)?;
            validate_rows(&m, replacements)?;
            let fp = m.fingerprint.clone();
            let (touched, fact_keys, row_keys) = self.touched_by(&fp, ns, fact)?;
            if fact_keys.is_empty() && row_keys.is_empty() {
                bail!("no Engram fact '{fact}' in namespace '{ns}' of '{model_name}'");
            }
            let t = at.unwrap_or_else(now_us);
            let live = facts_at(self, &fp, ns, t)?;
            for r in replacements {
                if !touched.contains(&(r.layer, r.row)) {
                    bail!("replacement row ({}, {}) was not touched by fact '{fact}': erase only rewrites touched rows", r.layer, r.row);
                }
                for f in &r.facts {
                    if f == fact || !live.contains_key(f) {
                        bail!("replacement row ({}, {}) depends on '{f}', which is erased or not current", r.layer, r.row);
                    }
                }
            }
            let puts = replacements.iter().map(|r| row_put(&fp, ns, r, t)).collect();
            commit(self, vec![], puts, fact_keys, row_keys)?;
            let after = rows_at(self, &fp, ns, t)?;
            let replaced: BTreeSet<(u64, i64)> = replacements.iter().map(|r| (r.layer, r.row)).collect();
            Ok(touched
                .into_iter()
                .map(|k| {
                    let a = if replaced.contains(&k) {
                        "replayed"
                    } else if after.contains_key(&k) {
                        "previous"
                    } else {
                        "reverted"
                    };
                    (k, a)
                })
                .collect())
        }

        /// Facts of a namespace current at `as_of` (default now): `(id, since
        /// µs, text, target, phrasings, rows touched, decided_by, source)`.
        pub fn engram_facts(&'s self, model_name: &str, ns: &str, as_of: Option<i64>) -> Result<Vec<(String, EngramFact, i64)>> {
            let m = model(self, model_name)?;
            if !rel_exists(self, ENGRAM_FACTS)? {
                return Ok(vec![]);
            }
            Ok(facts_at(self, &m.fingerprint, ns, as_of.unwrap_or_else(now_us))?
                .into_iter()
                .map(|(id, f)| {
                    let at = f.at;
                    (
                        id.clone(),
                        EngramFact {
                            id,
                            text: f.text,
                            target: f.target,
                            phrasings: f.phrasings,
                            rows: f.rows,
                            decided_by: f.decided_by,
                            source: f.source,
                            payload: f.payload.map(|b| b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()),
                        },
                        at,
                    )
                })
                .collect())
        }

        /// Every version of the fact records of a namespace (or one fact):
        /// `(fact, at µs, asserted, text, target, decided_by, source)`.
        pub fn engram_history(&'s self, model_name: &str, ns: &str, fact: Option<&str>) -> Result<NamedRows> {
            let m = model(self, model_name)?;
            let cols = ["fact", "at", "asserted", "text", "target", "decided_by", "source", "rows"];
            if !rel_exists(self, ENGRAM_FACTS)? {
                return Ok(status_rows(&cols, vec![]));
            }
            let ff = if fact.is_some() { ", fact == $fact" } else { "" };
            let mut rows = query(
                self,
                &format!(
                    "?[fact, vt, text, target, decided_by, source, n] := \
                     *{ENGRAM_FACTS}{{fp, ns, fact, vt, text, target, decided_by, source, rows}}, \
                     fp == $fp, ns == $ns{ff}, n = length(rows) / 2"
                ),
                params([("fp", s(&m.fingerprint)), ("ns", s(ns)), ("fact", fact.map_or(DataValue::Null, s))]),
            )?;
            for r in rows.iter_mut() {
                let (ts, assert) = vt_parts(&r[1]);
                r[1] = DataValue::from(ts);
                r.insert(2, DataValue::Bool(assert));
            }
            rows.sort_by(|a, b| (a[0].get_str(), a[1].get_int()).cmp(&(b[0].get_str(), b[1].get_int())));
            Ok(status_rows(&cols, rows))
        }

        /// Rows `text` reads under `model` (needs the model registered from
        /// config + tokenizer in this process or with stored paths).
        pub fn engram_rows_for_text(&'s self, model_name: &str, text: &str, last_only: bool) -> Result<Vec<kleos_engram::RowHit>> {
            let m = model(self, model_name)?;
            let spec = self.engram_spec(&m)?;
            let (ids, hits) = spec.rows_for_text(text, false).map_err(|e| miette!("{e}"))?;
            Ok(hits.into_iter().filter(|h| !last_only || h.pos + 1 == ids.len()).collect())
        }

        fn engram_spec(&'s self, m: &EngramModel) -> Result<Arc<EngramSpec>> {
            if let Some(s) = SPECS.lock().unwrap().get(&m.fingerprint) {
                return Ok(s.clone());
            }
            let r = query(
                self,
                &format!("?[cp, tp, pad] := *{ENGRAM_MODELS}{{name: $name, config_path: cp, tokenizer_path: tp, pad_id: pad}}"),
                params([("name", s(&m.name))]),
            )?;
            let r = r.first().ok_or_else(|| ModelNotFound(m.name.clone()))?;
            let (Some(cp), Some(tp)) = (r[0].get_str(), r[1].get_str()) else {
                bail!("Engram model '{}' was registered by fingerprint only; text cannot be hashed", m.name);
            };
            let ov = SpecOverrides { pad_id: r[2].get_int().map(|p| p as u32), ..Default::default() };
            let spec = EngramSpec::from_files(Path::new(cp), Path::new(tp), &ov).map_err(|e| miette!("{e}"))?;
            if spec.fingerprint() != m.fingerprint {
                return Err(FingerprintMismatch {
                    model: m.name.clone(),
                    registered: m.fingerprint.clone(),
                    given: spec.fingerprint().to_string(),
                    what: "model files on disk",
                }
                .into());
            }
            let spec = Arc::new(spec);
            SPECS.lock().unwrap().insert(m.fingerprint.clone(), spec.clone());
            Ok(spec)
        }
    }

    // ---- CozoScript surface -------------------------------------------------

    fn opt_json(op: &EngramOp, key: &str) -> Option<JsonValue> {
        op.opts.get(key).map(JsonValue::from)
    }

    fn opt_str(op: &EngramOp, key: &str) -> Result<Option<String>> {
        match op.opts.get(key) {
            None | Some(DataValue::Null) => Ok(None),
            Some(DataValue::Str(s)) => Ok(Some(s.to_string())),
            Some(v) => bail!("`::engram {}` option '{key}' must be a string, got {v:?}", op.verb),
        }
    }

    fn need_str(op: &EngramOp, key: &str) -> Result<String> {
        opt_str(op, key)?.ok_or_else(|| miette!("`::engram {}` needs option '{key}'", op.verb))
    }

    fn opt_int(op: &EngramOp, key: &str) -> Result<Option<i64>> {
        match op.opts.get(key) {
            None | Some(DataValue::Null) => Ok(None),
            Some(v) => v.get_int().map(Some).ok_or_else(|| miette!("`::engram {}` option '{key}' must be an integer (µs), got {v:?}", op.verb)),
        }
    }

    fn namespaces(op: &EngramOp) -> Result<Vec<String>> {
        match op.opts.get("ns") {
            Some(DataValue::Str(s)) => Ok(vec![s.to_string()]),
            Some(DataValue::List(l)) => l
                .iter()
                .map(|v| v.get_str().map(str::to_string).ok_or_else(|| miette!("`::engram {}`: ns must be strings", op.verb)))
                .collect(),
            _ => bail!("`::engram {}` needs option 'ns' (a namespace or a list, most general first)", op.verb),
        }
    }

    fn need_model(op: &EngramOp) -> Result<&str> {
        op.model.as_deref().ok_or_else(|| miette!("`::engram {}` needs a model name: `::engram {} <model> {{...}}`", op.verb, op.verb))
    }

    fn check_keys(op: &EngramOp, allowed: &[&str]) -> Result<()> {
        for k in op.opts.keys() {
            if !allowed.contains(&k.as_str()) {
                bail!("unknown option '{k}' for `::engram {}`: expected {}", op.verb, allowed.join(", "));
            }
        }
        Ok(())
    }

    fn parse_list<T: for<'de> Deserialize<'de>>(op: &EngramOp, key: &str) -> Result<Vec<T>> {
        match opt_json(op, key) {
            None | Some(JsonValue::Null) => Ok(vec![]),
            Some(v) => serde_json::from_value(v).map_err(|e| miette!("`::engram {}` option '{key}': {e}", op.verb)),
        }
    }

    fn row_out(rows: Vec<((u64, i64), &'static str)>) -> NamedRows {
        status_rows(
            &["layer", "row", "action"],
            rows.into_iter().map(|((l, r), a)| vec![DataValue::from(l as i64), DataValue::from(r), s(a)]).collect(),
        )
    }

    /// Run a `::engram` op (all its writes go through their own scripts, so
    /// they refuse to run inside an imperative block).
    pub(crate) fn run_engram_op<'s, S: Storage<'s>>(db: &'s Db<S>, op: &EngramOp, read_only: bool, in_tx: bool) -> Result<NamedRows> {
        let writes = matches!(op.verb.as_str(), "register" | "write" | "supersede" | "erase" | "erase_fact");
        if writes && read_only {
            bail!("`::engram {}` cannot run in read-only mode", op.verb);
        }
        if writes && in_tx {
            bail!("`::engram {}` must run as a script of its own, not inside an imperative block", op.verb);
        }
        match op.verb.as_str() {
            "models" => {
                let ms = db.engram_models()?;
                Ok(status_rows(
                    &["name", "fingerprint", "row_dim", "layers", "table_rows"],
                    ms.into_iter()
                        .map(|m| {
                            vec![
                                s(&m.name),
                                s(&m.fingerprint),
                                DataValue::from(m.row_dim as i64),
                                DataValue::List(m.layers.iter().map(|&l| DataValue::from(l as i64)).collect()),
                                DataValue::List(m.table_rows.iter().map(|&l| DataValue::from(l as i64)).collect()),
                            ]
                        })
                        .collect(),
                ))
            }
            "register" => {
                check_keys(op, &["config", "tokenizer", "pad_id", "fingerprint", "row_dim", "layers"])?;
                let name = need_model(op)?;
                let fp = if let Some(fpr) = opt_str(op, "fingerprint")? {
                    let row_dim = opt_int(op, "row_dim")?.ok_or_else(|| miette!("`::engram register` by fingerprint needs row_dim"))?;
                    let layers: Vec<u64> = parse_list(op, "layers")?;
                    db.engram_register_raw(name, &fpr, row_dim as usize, &layers, &vec![0; layers.len()], serde_json::json!({}), None, None)?
                } else {
                    let cp = PathBuf::from(need_str(op, "config")?);
                    let tp = PathBuf::from(need_str(op, "tokenizer")?);
                    let ov = SpecOverrides { pad_id: opt_int(op, "pad_id")?.map(|p| p as u32), ..Default::default() };
                    let spec = EngramSpec::from_files(&cp, &tp, &ov).map_err(|e| miette!("{e}"))?;
                    db.engram_register(name, &spec, Some((&cp, &tp)))?
                };
                Ok(status_rows(&["name", "fingerprint"], vec![vec![s(name), s(&fp)]]))
            }
            "write" | "supersede" => {
                let name = need_model(op)?;
                check_keys(op, &["ns", "fingerprint", "at", "facts", "rows", "retract", "fact"])?;
                let w = EngramWrite {
                    ns: need_str(op, "ns")?,
                    fingerprint: need_str(op, "fingerprint")?,
                    at: opt_int(op, "at")?,
                    facts: parse_list(op, "facts")?,
                    rows: parse_list(op, "rows")?,
                    retract: parse_list(op, "retract")?,
                };
                if op.verb == "supersede" {
                    db.engram_supersede(name, &need_str(op, "fact")?, &w)
                } else {
                    db.engram_write(name, &w)
                }
            }
            "materialize" => {
                check_keys(op, &["ns", "as_of", "fingerprint"])?;
                let ov = db.engram_materialize(need_model(op)?, &namespaces(op)?, opt_int(op, "as_of")?, opt_str(op, "fingerprint")?.as_deref())?;
                Ok(status_rows(
                    &["layer", "row", "ns", "facts", "vec"],
                    ov.rows
                        .into_iter()
                        .map(|r| {
                            vec![
                                DataValue::from(r.layer as i64),
                                DataValue::from(r.row),
                                s(&r.ns),
                                strs(&r.facts),
                                DataValue::List(bf16::to_f32(&r.vec).into_iter().map(|x| DataValue::from(x as f64)).collect()),
                            ]
                        })
                        .collect(),
                ))
            }
            "export" => {
                check_keys(op, &["ns", "as_of", "fingerprint", "dir"])?;
                let dir = PathBuf::from(need_str(op, "dir")?);
                let files = db.engram_export(need_model(op)?, &namespaces(op)?, opt_int(op, "as_of")?, &dir, opt_str(op, "fingerprint")?.as_deref())?;
                Ok(status_rows(
                    &["layer", "path", "rows"],
                    files
                        .into_iter()
                        .map(|(l, p, n)| vec![DataValue::from(l as i64), s(&p.to_string_lossy()), DataValue::from(n as i64)])
                        .collect(),
                ))
            }
            "erase" => {
                check_keys(op, &["ns"])?;
                let ns = need_str(op, "ns")?;
                let (f, r) = db.engram_erase_namespace(&ns, op.model.as_deref())?;
                Ok(status_rows(&["ns", "fact_versions", "row_versions"], vec![vec![s(&ns), DataValue::from(f as i64), DataValue::from(r as i64)]]))
            }
            "erase_plan" => {
                check_keys(op, &["ns", "fact"])?;
                let plan = db.engram_erase_plan(need_model(op)?, &need_str(op, "ns")?, &need_str(op, "fact")?)?;
                Ok(status_rows(
                    &["layer", "row", "other_facts"],
                    plan.into_iter().map(|((l, r), f)| vec![DataValue::from(l as i64), DataValue::from(r), strs(&f)]).collect(),
                ))
            }
            "erase_fact" => {
                check_keys(op, &["ns", "fact", "rows", "at"])?;
                let reps: Vec<RowEdit> = parse_list(op, "rows")?;
                Ok(row_out(db.engram_erase_fact(need_model(op)?, &need_str(op, "ns")?, &need_str(op, "fact")?, &reps, opt_int(op, "at")?)?))
            }
            "facts" | "list" => {
                check_keys(op, &["ns", "as_of"])?;
                let facts = db.engram_facts(need_model(op)?, &need_str(op, "ns")?, opt_int(op, "as_of")?)?;
                Ok(status_rows(
                    &["fact", "since", "text", "target", "phrasings", "rows", "decided_by", "source"],
                    facts
                        .into_iter()
                        .map(|(id, f, at)| {
                            vec![
                                s(&id),
                                DataValue::from(at),
                                s(&f.text),
                                s(&f.target),
                                strs(&f.phrasings),
                                DataValue::from(f.rows.len() as i64),
                                s(&f.decided_by),
                                s(&f.source),
                            ]
                        })
                        .collect(),
                ))
            }
            "history" => {
                check_keys(op, &["ns", "fact"])?;
                db.engram_history(need_model(op)?, &need_str(op, "ns")?, opt_str(op, "fact")?.as_deref())
            }
            "rows" => {
                check_keys(op, &["text", "last"])?;
                let last = matches!(op.opts.get("last"), Some(DataValue::Bool(true)));
                let hits = db.engram_rows_for_text(need_model(op)?, &need_str(op, "text")?, last)?;
                Ok(status_rows(
                    &["pos", "layer", "order", "head", "row"],
                    hits.into_iter()
                        .map(|h| {
                            vec![
                                DataValue::from(h.pos as i64),
                                DataValue::from(h.layer as i64),
                                DataValue::from(h.order as i64),
                                DataValue::from(h.head as i64),
                                DataValue::from(h.row),
                            ]
                        })
                        .collect(),
                ))
            }
            other => bail!(
                "unknown `::engram` op '{other}': expected models, register, write, supersede, materialize, export, \
                 erase, erase_plan, erase_fact, facts, history or rows"
            ),
        }
    }
}
