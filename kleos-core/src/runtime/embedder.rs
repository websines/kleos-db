/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! `::embedder list | enable | disable | backfill` (kleos fork;
//! docs/specs/embedder-toggle.md). The embedder registry is process-wide, so
//! list/enable/disable act on every open database in the process; backfill
//! writes one relation of this database.

use std::collections::BTreeSet;

use itertools::Itertools;
use miette::Result;

use crate::bail;
use crate::data::symb::Symbol;
use crate::data::value::DataValue;
use crate::embed::{self, EmbedderStatus};
use crate::parse::sys::EmbedderOp;
use crate::runtime::db::{Db, ScriptMutability};
use crate::storage::Storage;
use crate::NamedRows;

fn opt_str(s: &Option<String>) -> DataValue {
    match s {
        Some(s) => DataValue::from(s.as_str()),
        None => DataValue::Null,
    }
}

/// `::embedder list` rows.
pub(crate) fn list_rows(list: Vec<EmbedderStatus>) -> NamedRows {
    NamedRows::new(
        [
            "name",
            "kind",
            "model",
            "dim",
            "enabled",
            "loaded",
            "precision",
            "device",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        list.into_iter()
            .map(|e| {
                vec![
                    DataValue::from(e.name.as_str()),
                    DataValue::from(e.kind.as_str()),
                    DataValue::from(e.model.as_str()),
                    DataValue::from(e.dim as i64),
                    DataValue::Bool(e.enabled),
                    e.loaded.map(DataValue::Bool).unwrap_or(DataValue::Null),
                    opt_str(&e.precision),
                    opt_str(&e.device),
                ]
            })
            .collect(),
    )
}

fn status(s: &str) -> NamedRows {
    NamedRows::new(vec!["status".to_string()], vec![vec![DataValue::from(s)]])
}

impl<'s, S: Storage<'s>> Db<S> {
    /// Turn a built-in embedder on (process-wide); see
    /// [`crate::embed::enable_embedder`].
    pub fn enable_embedder(&self, name: &str, spec: embed::EmbedderSpec) -> Result<()> {
        embed::enable_embedder(name, spec)
    }
    /// Turn an embedder off (process-wide); see
    /// [`crate::embed::disable_embedder`].
    pub fn disable_embedder(&self, name: &str) -> Result<bool> {
        embed::disable_embedder(name)
    }
    /// Every enabled and disabled embedder (process-wide).
    pub fn embedders(&self) -> Vec<EmbedderStatus> {
        embed::embedders()
    }
    /// Run an `::embedder` op. `in_tx` is true on the imperative path, where a
    /// write transaction is already open and backfill cannot run.
    pub(crate) fn run_embedder_op(
        &'s self,
        op: &EmbedderOp,
        read_only: bool,
        in_tx: bool,
    ) -> Result<NamedRows> {
        if read_only && !matches!(op, EmbedderOp::List) {
            bail!("`::embedder enable/disable/backfill` cannot run in read-only mode");
        }
        match op {
            EmbedderOp::List => Ok(list_rows(embed::embedders())),
            EmbedderOp::Enable(name, spec) => {
                embed::enable_embedder(name, spec.clone())?;
                Ok(status("OK"))
            }
            EmbedderOp::Disable(name) => Ok(status(if embed::disable_embedder(name)? {
                "OK"
            } else {
                "ALREADY_DISABLED"
            })),
            EmbedderOp::Backfill(rel) => {
                if in_tx {
                    bail!("`::embedder backfill` must run as a script of its own, not inside an imperative block");
                }
                self.backfill_embeddings(&rel.name)
            }
        }
    }

    /// Fill the null values of every column of `relation` whose default calls
    /// `embed()` / `embed_or_null()`, by re-writing the column's source
    /// columns with `:update` (which recomputes the column in batches through
    /// the embedding cache and maintains its HNSW indexes). Errors, writing
    /// nothing, when an embedder those columns use is still off. Returns one
    /// row per such column: `column, filled, still_null` (`still_null`
    /// counts rows whose source text is itself null).
    pub fn backfill_embeddings(&'s self, relation: &str) -> Result<NamedRows> {
        let handle = {
            let tx = self.transact()?;
            tx.get_relation(relation, false)?
        };
        if handle.has_txtime() {
            bail!("`::embedder backfill` does not support TxTime relations ({relation})");
        }
        let keys = handle
            .metadata
            .keys
            .iter()
            .map(|c| c.name.to_string())
            .collect_vec();
        let key_set: BTreeSet<&str> = keys.iter().map(|s| s.as_str()).collect();
        let mut targets = vec![];
        for col in &handle.metadata.non_keys {
            let Some(expr) = &col.default_gen else {
                continue;
            };
            let models = embed::embed_models_in(expr);
            if models.is_empty() {
                continue;
            }
            let sources = expr
                .bindings()?
                .into_iter()
                .map(|s: Symbol| s.name.to_string())
                .collect_vec();
            if sources.is_empty() {
                continue;
            }
            for m in models {
                if let Ok(DataValue::Str(name)) = m.eval_to_const() {
                    if !embed::is_embedder_enabled(&name) {
                        // the same "disabled / not found" error embed() gives
                        embed::embed_texts(&name, &[])?;
                    }
                }
            }
            let value_sources = sources
                .iter()
                .filter(|s| !key_set.contains(s.as_str()))
                .cloned()
                .collect_vec();
            if value_sources.is_empty() {
                bail!(
                    "`::embedder backfill` cannot refill {relation}.{}: its default reads only key columns",
                    col.name
                );
            }
            targets.push((col.name.to_string(), value_sources));
        }
        if targets.is_empty() {
            bail!("relation {relation} has no column whose default calls embed() or embed_or_null()");
        }
        let mut rows = vec![];
        for (col, value_sources) in targets {
            let mut read_cols = keys.clone();
            read_cols.extend(value_sources.iter().cloned());
            read_cols.push(col.clone());
            let read_cols = read_cols.into_iter().unique().join(", ");
            let count = |db: &'s Self| -> Result<i64> {
                let r = db.run_script(
                    &format!(
                        "?[count({k})] := *{relation}{{{read_cols}}}, is_null({col})",
                        k = keys[0]
                    ),
                    Default::default(),
                    ScriptMutability::Immutable,
                )?;
                Ok(r.rows.first().and_then(|r| r[0].get_int()).unwrap_or(0))
            };
            let before = count(self)?;
            if before > 0 {
                let out_cols = keys
                    .iter()
                    .chain(value_sources.iter())
                    .join(", ");
                let script = format!(
                    "?[{out_cols}] := *{relation}{{{read_cols}}}, is_null({col}) \
                     :update {relation} {{{keys} => {vals}}}",
                    keys = keys.join(", "),
                    vals = value_sources.join(", ")
                );
                self.run_script(&script, Default::default(), ScriptMutability::Mutable)?;
            }
            let after = if before > 0 { count(self)? } else { 0 };
            rows.push(vec![
                DataValue::from(col.as_str()),
                DataValue::from(before - after),
                DataValue::from(after),
            ]);
        }
        Ok(NamedRows::new(
            vec![
                "column".to_string(),
                "filled".to_string(),
                "still_null".to_string(),
            ],
            rows,
        ))
    }
}
