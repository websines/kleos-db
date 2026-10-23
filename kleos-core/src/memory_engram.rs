/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The memory API's Engram hook (docs/specs/engram-overlay.md, "Memory
//! attachment"; feature `engram`).
//!
//! When a store is attached to a registered Engram model
//! (`::memory init {engram: 'nano'}`), every memory change that an overlay
//! must follow leaves a job in `kleos_engram_queue`, written by the same
//! script, so the same transaction, as the change itself:
//!
//! - `remember` that adds a memory: `write` (or `supersede` when it ended
//!   older memories, with their ids in `old`);
//! - `forget` retract: `retract`; `forget` erase: `erase`.
//!
//! A job's `qid` is the namespace version (`kleos_mem_ns`) its commit
//! produced, so jobs are numbered in commit order per namespace. The edits
//! themselves need the model (target computation by backprop), so a worker
//! outside the transaction applies the jobs through the `::engram` ops
//! (`scripts/engram/engram_worker.py`). Without an attachment, or in a build
//! without the feature, [`enqueue`] returns nothing and the memory API runs
//! exactly as before.

use crate::data::value::DataValue;
use crate::memory::MemConfig;

/// Engram jobs: `{ns, qid => op, model, mem_ids, old, text, at, meta, status,
/// attempts, error, created, updated}`.
pub const ENGRAM_QUEUE: &str = "kleos_engram_queue";

/// One memory change an attached overlay has to follow.
pub(crate) struct Job<'a> {
    /// `write`, `supersede`, `retract` or `erase`.
    pub op: &'static str,
    /// The memories the job is about (the new memory for write / supersede).
    pub mem_ids: &'a [String],
    /// Memories a supersede ended.
    pub old: &'a [String],
    /// The memory text (write / supersede).
    pub text: Option<&'a str>,
    /// Valid time of the change, µs: the memory's valid-from, or the
    /// retraction time.
    pub at: i64,
    /// Entity names given to `remember` (the worker's subject hint).
    pub entities: &'a [String],
}

/// Script fragment (one `{}` block, or two for erase) and its parameters
/// that enqueue `job`, to be placed in the memory change's own write script
/// **before** the block that bumps `kleos_mem_ns`: it numbers the job with
/// the version the bump produces. `None` when no Engram model is attached.
/// The script must bind `$ns`.
#[cfg(not(feature = "engram"))]
pub(crate) fn enqueue(_cfg: &MemConfig, _job: Job<'_>) -> Option<(String, Vec<(&'static str, DataValue)>)> {
    None
}

#[cfg(feature = "engram")]
pub(crate) fn enqueue(cfg: &MemConfig, job: Job<'_>) -> Option<(String, Vec<(&'static str, DataValue)>)> {
    let model = cfg.engram.as_ref()?;
    let strs = |v: &[String]| DataValue::List(v.iter().map(|x| DataValue::from(x.as_str())).collect());
    let mut script = String::new();
    if job.op == "erase" {
        // the queue must not keep what was erased: drop the text and entity
        // names of earlier jobs about these memories
        script.push_str(&format!(
            "{{?[ns, qid, text, meta] := *{ENGRAM_QUEUE}{{ns, qid, mem_ids, text: t}}, ns = $ns, \
               !is_null(t), length(intersection(mem_ids, $eq_ids)) > 0, text = null, meta = null \
              :update {ENGRAM_QUEUE} {{ns, qid => text, meta}}}}"
        ));
    }
    script.push_str(&format!(
        "{{cur[max(s)] := *{}{{ns: $ns, seq: s}} \
           cur[max(s)] := s = 0 \
           ?[ns, qid, op, model, mem_ids, old, text, at, meta, status, attempts, error, created, updated] := \
             cur[s0], ns = $ns, qid = s0 + 1, op = $eq_op, model = $eq_model, mem_ids = $eq_ids, old = $eq_old, \
             text = $eq_text, at = $eq_at, meta = $eq_meta, status = 'pending', attempts = 0, error = null, \
             created = $eq_now, updated = $eq_now \
          :put {ENGRAM_QUEUE} {{ns, qid => op, model, mem_ids, old, text, at, meta, status, attempts, error, created, updated}}}}",
        crate::memory::MEM_NS
    ));
    let meta = if job.entities.is_empty() {
        DataValue::Null
    } else {
        DataValue::Json(crate::JsonData(serde_json::json!({ "entities": job.entities })))
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    Some((
        script,
        vec![
            ("eq_op", DataValue::from(job.op)),
            ("eq_model", DataValue::from(model.as_str())),
            ("eq_ids", strs(job.mem_ids)),
            ("eq_old", strs(job.old)),
            ("eq_text", job.text.map_or(DataValue::Null, DataValue::from)),
            ("eq_at", DataValue::from(job.at)),
            ("eq_meta", meta),
            ("eq_now", DataValue::from(now)),
        ],
    ))
}

#[cfg(feature = "engram")]
pub(crate) use imp::*;

#[cfg(feature = "engram")]
mod imp {
    use std::collections::BTreeMap;

    use miette::Result;

    use super::ENGRAM_QUEUE;
    use crate::data::value::DataValue;
    use crate::memory::{MemConfig, MEM_CONFIG};
    use crate::runtime::db::{Db, NamedRows, ScriptMutability};
    use crate::storage::Storage;

    fn rel_exists<'s, S: Storage<'s>>(db: &'s Db<S>, name: &str) -> Result<bool> {
        let tx = db.transact()?;
        Ok(tx.get_relation(name, false).is_ok())
    }

    /// Attach the memory store to the registered Engram model `name` (an
    /// empty name detaches): checks the model, creates the job queue and
    /// records the attachment in the memory config.
    pub(crate) fn attach<'s, S: Storage<'s>>(db: &'s Db<S>, cfg: &mut MemConfig, name: &str) -> Result<()> {
        let name = name.trim();
        if !name.is_empty() {
            crate::engram::model(db, name)?;
            if !rel_exists(db, ENGRAM_QUEUE)? {
                db.run_script(
                    &format!(
                        ":create {ENGRAM_QUEUE} {{ns: String, qid: Int => op: String, model: String, mem_ids: [String], \
                         old: [String], text: String?, at: Int, meta: Json?, status: String, attempts: Int, \
                         error: String?, created: Int, updated: Int}}"
                    ),
                    Default::default(),
                    ScriptMutability::Mutable,
                )?;
            }
        }
        let value = if name.is_empty() { None } else { Some(name.to_string()) };
        if cfg.engram != value {
            db.run_script(
                &format!("?[key, value] <- [['engram', $v]] :put {MEM_CONFIG} {{key => value}}"),
                BTreeMap::from([("v".to_string(), value.as_deref().map_or(DataValue::Null, DataValue::from))]),
                ScriptMutability::Mutable,
            )?;
            cfg.engram = value;
        }
        Ok(())
    }

    /// Queue status per namespace (or one): `ns, pending, running, done,
    /// failed, last_qid, last_error`. With `jobs`, one row per job instead:
    /// `ns, qid, op, mem_ids, old, at, status, attempts, error, updated`.
    pub(crate) fn queue_status<'s, S: Storage<'s>>(db: &'s Db<S>, ns: Option<&str>, jobs: bool) -> Result<NamedRows> {
        let cols: &[&str] = if jobs {
            &["ns", "qid", "op", "mem_ids", "old", "at", "status", "attempts", "error", "updated"]
        } else {
            &["ns", "pending", "running", "done", "failed", "last_qid", "last_error"]
        };
        let cols: Vec<String> = cols.iter().map(|c| c.to_string()).collect();
        if !rel_exists(db, ENGRAM_QUEUE)? {
            return Ok(NamedRows::new(cols, vec![]));
        }
        let filter = if ns.is_some() { ", ns == $ns" } else { "" };
        let rows = db
            .run_script(
                &format!(
                    "?[ns, qid, op, mem_ids, old, at, status, attempts, error, updated] := \
                     *{ENGRAM_QUEUE}{{ns, qid, op, mem_ids, old, at, status, attempts, error, updated}}{filter}"
                ),
                BTreeMap::from([("ns".to_string(), ns.map_or(DataValue::Null, DataValue::from))]),
                ScriptMutability::Immutable,
            )?
            .rows;
        if jobs {
            return Ok(NamedRows::new(cols, rows));
        }
        // (pending, running, done, failed, last qid, (updated, last error))
        type Acc = (i64, i64, i64, i64, i64, Option<(i64, String)>);
        let mut by: BTreeMap<String, Acc> = BTreeMap::new();
        for r in rows {
            let a = by.entry(r[0].get_str().unwrap_or_default().to_string()).or_default();
            match r[6].get_str() {
                Some("pending") => a.0 += 1,
                Some("running") => a.1 += 1,
                Some("done") => a.2 += 1,
                Some("failed") => a.3 += 1,
                _ => {}
            }
            a.4 = a.4.max(r[1].get_int().unwrap_or(0));
            if let (Some(e), Some(t)) = (r[8].get_str(), r[9].get_int()) {
                if a.5.as_ref().map_or(true, |(u, _)| t >= *u) {
                    a.5 = Some((t, e.to_string()));
                }
            }
        }
        Ok(NamedRows::new(
            cols,
            by.into_iter()
                .map(|(ns, a)| {
                    vec![
                        DataValue::from(ns.as_str()),
                        DataValue::from(a.0),
                        DataValue::from(a.1),
                        DataValue::from(a.2),
                        DataValue::from(a.3),
                        DataValue::from(a.4),
                        a.5.map_or(DataValue::Null, |(_, e)| DataValue::from(e.as_str())),
                    ]
                })
                .collect(),
        ))
    }
}
