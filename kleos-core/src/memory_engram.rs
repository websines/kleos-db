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
//!
//! Each job is also a durable workflow (docs/specs/durable-workflows.md,
//! "Engram"): the same transaction starts `engram.apply` with id
//! `engram:<ns>:<qid>` on queue `engram`, in lane `engram:<ns>` ordered by
//! qid, so the worker gets claims, leases, retries with back-off, history
//! and status from the workflow engine. The job row stays the job's record.

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
/// Make sure what an attached store's commits write exists: the workflow
/// relations, for stores attached before jobs became workflows.
#[cfg(not(feature = "engram"))]
pub(crate) fn ready<'s, S: crate::storage::Storage<'s>>(_db: &'s crate::runtime::db::Db<S>, _cfg: &MemConfig) -> miette::Result<()> {
    Ok(())
}

#[cfg(feature = "engram")]
pub(crate) fn ready<'s, S: crate::storage::Storage<'s>>(db: &'s crate::runtime::db::Db<S>, cfg: &MemConfig) -> miette::Result<()> {
    if cfg.engram.is_some() {
        db.flow_ensure()?;
    }
    Ok(())
}

/// Blocks that start one `engram.apply` workflow per job, given rules that
/// define `q[qid]` (the qids the jobs take; they must be computed from the
/// namespace version before this commit's bump, like the job rows).
#[cfg(feature = "engram")]
fn flow_blocks(qrule: &str) -> String {
    use crate::flow::{FLOW_EVENT, FLOW_LANE, FLOW_TASK, FLOW_WF};
    let fw = format!("{qrule} fw[id, qid] := q[qid], id = concat('engram:', $ns, ':', to_string(qid))");
    format!(
        "{{{fw} \
           ?[id, name, version, mode, queue, status, input, output, error, parent, serial, serial_seq, token, seq, \
             owner, lease_until, wait, cancel, attempts, created, updated, finished, meta] := fw[id, qid], \
             name = 'engram.apply', version = 1, mode = 'code', queue = 'engram', status = 'pending', \
             input = json_object('ns', $ns, 'qid', qid), output = null, error = null, parent = null, \
             serial = concat('engram:', $ns), serial_seq = qid, token = 0, seq = 1, owner = null, \
             lease_until = null, wait = null, cancel = to_bool(0), attempts = 0, created = $eq_now, updated = $eq_now, \
             finished = null, meta = null \
          :put {FLOW_WF} {{id => name, version, mode, queue, status, input, output, error, parent, serial, serial_seq, \
             token, seq, owner, lease_until, wait, cancel, attempts, created, updated, finished, meta}}}} \
         {{{fw} \
           ?[wf, seq, type, key, name, data, at, token] := fw[wf, qid], seq = 1, type = 'started', key = null, \
             name = 'engram.apply', data = json_object('ns', $ns, 'qid', qid), at = $eq_now, token = 0 \
          :put {FLOW_EVENT} {{wf, seq => type, key, name, data, at, token}}}} \
         {{{fw} \
           ?[wf, queue, due, serial, serial_seq] := fw[wf, qid], queue = 'engram', due = $eq_now, \
             serial = concat('engram:', $ns), serial_seq = qid \
          :put {FLOW_TASK} {{wf => queue, due, serial, serial_seq}}}} \
         {{{fw} \
           ?[serial, serial_seq, wf] := fw[wf, qid], serial = concat('engram:', $ns), serial_seq = qid \
          :put {FLOW_LANE} {{serial, serial_seq => wf}}}}"
    )
}

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
    script.push_str(&flow_blocks(&format!(
        "cur[max(s)] := *{}{{ns: $ns, seq: s}} cur[max(s)] := s = 0 q[qid] := cur[s0], qid = s0 + 1",
        crate::memory::MEM_NS
    )));
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

/// Several jobs of one memory change (one per claim a `remember` adds), as
/// one script block numbered `qid = v + 1 .. v + n` from the version `v` the
/// change planned against. Returns the fragment, its parameters and `n`:
/// the caller must advance `kleos_mem_ns` by `n`, so the last qid is the
/// version its commit produces and qids stay unique and in commit order.
/// `None` when no Engram model is attached or there are no jobs.
#[cfg(not(feature = "engram"))]
pub(crate) fn enqueue_jobs(
    _cfg: &MemConfig,
    _jobs: &[Job<'_>],
) -> Option<(String, Vec<(&'static str, DataValue)>, usize)> {
    None
}

#[cfg(feature = "engram")]
pub(crate) fn enqueue_jobs(
    cfg: &MemConfig,
    jobs: &[Job<'_>],
) -> Option<(String, Vec<(&'static str, DataValue)>, usize)> {
    let model = cfg.engram.as_ref()?;
    if jobs.is_empty() {
        return None;
    }
    let strs = |v: &[String]| DataValue::List(v.iter().map(|x| DataValue::from(x.as_str())).collect());
    let rows: Vec<DataValue> = jobs
        .iter()
        .enumerate()
        .map(|(i, job)| {
            let meta = if job.entities.is_empty() {
                DataValue::Null
            } else {
                DataValue::Json(crate::JsonData(serde_json::json!({ "entities": job.entities })))
            };
            DataValue::List(vec![
                DataValue::from(i as i64 + 1),
                DataValue::from(job.op),
                strs(job.mem_ids),
                strs(job.old),
                job.text.map_or(DataValue::Null, DataValue::from),
                DataValue::from(job.at),
                meta,
            ])
        })
        .collect();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    let script = format!(
        "{{cur[max(s)] := *{}{{ns: $ns, seq: s}} \
           cur[max(s)] := s = 0 \
           j[i, op, mem_ids, old, text, at, meta] <- $eq_jobs \
           ?[ns, qid, op, model, mem_ids, old, text, at, meta, status, attempts, error, created, updated] := \
             cur[s0], j[i, op, mem_ids, old, text, at, meta], ns = $ns, qid = s0 + i, model = $eq_model, \
             status = 'pending', attempts = 0, error = null, created = $eq_now, updated = $eq_now \
          :put {ENGRAM_QUEUE} {{ns, qid => op, model, mem_ids, old, text, at, meta, status, attempts, error, created, updated}}}}",
        crate::memory::MEM_NS
    ) + flow_blocks(&format!(
        "cur[max(s)] := *{}{{ns: $ns, seq: s}} cur[max(s)] := s = 0 jn[i] <- $eq_idx q[qid] := cur[s0], jn[i], qid = s0 + i",
        crate::memory::MEM_NS
    ))
    .as_str();
    let idx = DataValue::List((1..=jobs.len()).map(|i| DataValue::List(vec![DataValue::from(i as i64)])).collect());
    Some((
        script,
        vec![
            ("eq_idx", idx),
            ("eq_jobs", DataValue::List(rows)),
            ("eq_model", DataValue::from(model.as_str())),
            ("eq_now", DataValue::from(now)),
        ],
        jobs.len(),
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
            db.flow_ensure()?;
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
