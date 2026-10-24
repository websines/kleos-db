/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Durable workflows (docs/specs/durable-workflows.md): the store, the ops
//! behind `::flow`, `Db::flow_call`, `POST /flow/:op` and the Python SDK, and
//! the engine actions a step can run atomically with its checkpoint.
//!
//! A workflow's progress is its append-only history (`kleos_flow_event`); a
//! worker claims a runnable workflow (`kleos_flow_task`) with a
//! compare-and-set that bumps its fencing token, replays it against the
//! history and records each new command. Every write a worker makes asserts
//! its token inside the same transaction, so a worker that lost its lease
//! cannot record anything, and a step's data writes (`query`, `remember`)
//! commit together with the step's event or not at all.

use std::collections::BTreeMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crossbeam::channel::bounded;
use miette::{bail, miette, IntoDiagnostic, Result};
use serde_derive::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};

use crate::data::value::DataValue;
use crate::runtime::db::{Db, NamedRows, ScriptMutability, TransactionPayload};
use crate::storage::Storage;
use crate::JsonData;

/// Workflow instances.
pub const FLOW_WF: &str = "kleos_flow_wf";
/// Append-only histories.
pub const FLOW_EVENT: &str = "kleos_flow_event";
/// Runnable work: a row per workflow that should run at `due`.
pub const FLOW_TASK: &str = "kleos_flow_task";
/// Pending durable timers (informational: `kleos_flow_task.due` is the
/// earliest of a waiting workflow's timers).
pub const FLOW_TIMER: &str = "kleos_flow_timer";
/// Signal inboxes.
pub const FLOW_SIGNAL: &str = "kleos_flow_signal";
/// Built-in step graphs, versioned.
pub const FLOW_DEF: &str = "kleos_flow_def";
/// Cron schedules.
pub const FLOW_SCHEDULE: &str = "kleos_flow_schedule";
/// Unfinished workflows of each lane (`serial`), by `serial_seq`.
pub const FLOW_LANE: &str = "kleos_flow_lane";

const DDL: &[(&str, &str)] = &[
    (
        FLOW_WF,
        "{id: String => name: String, version: Int, mode: String, queue: String, status: String, \
         input: Json?, output: Json?, error: String?, parent: String?, serial: String?, serial_seq: Int?, \
         token: Int, seq: Int, owner: String?, lease_until: Int?, wait: Json?, cancel: Bool, attempts: Int, \
         created: Int, updated: Int, finished: Int?, meta: Json?}",
    ),
    (
        FLOW_EVENT,
        "{wf: String, seq: Int => type: String, key: String?, name: String?, data: Json?, at: Int, token: Int}",
    ),
    (
        FLOW_TASK,
        "{wf: String => queue: String, due: Int, serial: String?, serial_seq: Int?}",
    ),
    (FLOW_TIMER, "{wf: String, key: String => due: Int, kind: String}"),
    (
        FLOW_SIGNAL,
        "{wf: String, sid: String => name: String, payload: Json?, at: Int, consumed: Int?}",
    ),
    (FLOW_DEF, "{name: String, version: Int => mode: String, graph: Json?, created: Int}"),
    (
        FLOW_SCHEDULE,
        "{name: String => workflow: String, version: Int?, cron: String, input: Json?, queue: String, \
         overlap: String, next_at: Int, last_id: String?, enabled: Bool, created: Int, updated: Int}",
    ),
    (FLOW_LANE, "{serial: String, serial_seq: Int => wf: String}"),
];

/// The workflow columns, in `:put` order.
const WF_COLS: &str = "id, name, version, mode, queue, status, input, output, error, parent, serial, serial_seq, \
                       token, seq, owner, lease_until, wait, cancel, attempts, created, updated, finished, meta";

/// Default lease of a claim.
pub(crate) const DEFAULT_LEASE_MS: i64 = 30_000;
/// Statuses after which nothing runs.
pub(crate) const FINISHED: [&str; 3] = ["completed", "failed", "canceled"];
/// Event types replay matches against command keys.
const COMMAND_TYPES: [&str; 11] = [
    "step",
    "step_failed",
    "timer",
    "timer_fired",
    "wait",
    "signal",
    "signal_timeout",
    "child",
    "child_done",
    "marker",
    "patch",
];

pub(crate) fn now_us() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

pub(crate) fn new_id(prefix: &str) -> String {
    format!("{prefix}_{:012x}{:06x}", now_us(), rand::random::<u32>() & 0xff_ffff)
}

fn sv(v: &str) -> DataValue {
    DataValue::from(v)
}

fn osv(v: &Option<String>) -> DataValue {
    v.as_deref().map(DataValue::from).unwrap_or(DataValue::Null)
}

/// A JSON value for a `Json?` column (`null` stays null).
pub(crate) fn jd(v: &JsonValue) -> DataValue {
    match v {
        JsonValue::Null => DataValue::Null,
        v => DataValue::Json(JsonData(v.clone())),
    }
}

fn jv(v: &DataValue) -> JsonValue {
    JsonValue::from(v)
}

fn params(p: Vec<(&str, DataValue)>) -> BTreeMap<String, DataValue> {
    p.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn is_contention(e: &miette::Report) -> bool {
    let m = format!("{e:?}").to_ascii_lowercase();
    ["busy", "locked", "conflict", "try again"].iter().any(|k| m.contains(k))
}

/// One event to append: `(type, key, name, data)`.
pub(crate) type Ev = (String, Option<String>, Option<String>, JsonValue);

pub(crate) fn ev(ty: &str, key: Option<&str>, name: Option<&str>, data: JsonValue) -> Ev {
    (ty.to_string(), key.map(str::to_string), name.map(str::to_string), data)
}

/// A recorded event.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Event {
    pub seq: i64,
    #[serde(rename = "type")]
    pub ty: String,
    pub key: Option<String>,
    pub name: Option<String>,
    pub data: JsonValue,
    pub at: i64,
    pub token: i64,
}

/// What a claim hands a worker.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Claimed {
    pub id: String,
    pub name: String,
    pub version: i64,
    pub mode: String,
    pub queue: String,
    pub input: JsonValue,
    pub token: i64,
    pub cancel: bool,
    pub now: i64,
    pub history: Vec<Event>,
}

/// The fencing check: the workflow's token is still `$fl_token`.
fn fence_q() -> String {
    format!("?[x] := *{FLOW_WF}{{id: $fl_id, token: t}}, t == $fl_token, x = 1 :assert some")
}

/// No cancel was requested.
fn not_canceled_q() -> String {
    format!("?[x] := *{FLOW_WF}{{id: $fl_id, cancel: c}}, c, x = 1 :assert none")
}

/// Append `$fl_evs` (rows `[i, type, key, name, data]`, i from 1) and
/// advance the workflow's `seq` by `$fl_n`: two queries.
fn append_q() -> [String; 2] {
    [
        // values come in as constant rows: a rule with many `x = <const>`
        // atoms costs CozoScript's planner far more than the write itself
        format!(
            "ev[i, wf, type, key, name, data, at, token] <- $fl_evs \
             ?[wf, seq, type, key, name, data, at, token] := *{FLOW_WF}{{id: $fl_id, seq: s0}}, \
               ev[i, wf, type, key, name, data, at, token], seq = s0 + i \
             :put {FLOW_EVENT} {{wf, seq => type, key, name, data, at, token}}"
        ),
        format!(
            "c[id, updated, n] <- [[$fl_id, $fl_now, $fl_n]] \
             ?[id, seq, updated] := c[id, updated, n], *{FLOW_WF}{{id, seq: s0}}, seq = s0 + n \
             :update {FLOW_WF} {{id => seq, updated}}"
        ),
    ]
}

fn block(q: &str) -> String {
    format!("{{{q}}}")
}

/// The fencing block, for a chained script.
fn fence() -> String {
    block(&fence_q())
}

/// The cancel check block, for a chained script.
fn not_canceled() -> String {
    block(&not_canceled_q())
}

fn append() -> String {
    append_q().iter().map(|q| block(q)).collect()
}

/// The queries of [`record_blocks`], one program each (for a multi-statement
/// transaction, which takes single programs).
pub(crate) fn record_parts(
    id: &str,
    token: i64,
    evs: &[Ev],
    check_cancel: bool,
) -> (Vec<String>, Vec<(&'static str, DataValue)>) {
    let mut parts = vec![fence_q()];
    if check_cancel {
        parts.push(not_canceled_q());
    }
    parts.extend(append_q());
    (parts, ev_params(id, token, now_us(), evs))
}

fn ev_params(id: &str, token: i64, now: i64, evs: &[Ev]) -> Vec<(&'static str, DataValue)> {
    let rows = evs
        .iter()
        .enumerate()
        .map(|(i, (ty, key, name, data))| {
            DataValue::List(vec![
                DataValue::from(i as i64 + 1),
                sv(id),
                sv(ty),
                osv(key),
                osv(name),
                jd(data),
                DataValue::from(now),
                DataValue::from(token),
            ])
        })
        .collect();
    vec![
        ("fl_id", sv(id)),
        ("fl_token", DataValue::from(token)),
        ("fl_now", DataValue::from(now)),
        ("fl_evs", DataValue::List(rows)),
        ("fl_n", DataValue::from(evs.len() as i64)),
    ]
}

/// Script blocks (and their parameters) that record `evs` for a worker
/// holding `token`, to run inside another write's transaction: the fencing
/// check, the cancel check (when `check_cancel`) and the append. Parameter
/// names start with `fl_`.
pub(crate) fn record_blocks(
    id: &str,
    token: i64,
    evs: &[Ev],
    check_cancel: bool,
) -> (String, Vec<(&'static str, DataValue)>) {
    let mut s = fence();
    if check_cancel {
        s.push_str(&not_canceled());
    }
    s.push_str(&append());
    (s, ev_params(id, token, now_us(), evs))
}

/// The row of a new workflow, for `:put {FLOW_WF} {WF_COLS}`.
#[allow(clippy::too_many_arguments)]
fn wf_row(
    id: &str,
    name: &str,
    version: i64,
    mode: &str,
    queue: &str,
    input: &JsonValue,
    parent: Option<&str>,
    serial: Option<&str>,
    serial_seq: Option<i64>,
    now: i64,
    meta: &JsonValue,
) -> DataValue {
    DataValue::List(vec![
        sv(id),
        sv(name),
        DataValue::from(version),
        sv(mode),
        sv(queue),
        sv("pending"),
        jd(input),
        DataValue::Null,
        DataValue::Null,
        parent.map(sv).unwrap_or(DataValue::Null),
        serial.map(sv).unwrap_or(DataValue::Null),
        serial_seq.map(DataValue::from).unwrap_or(DataValue::Null),
        DataValue::from(0),
        DataValue::from(1),
        DataValue::Null,
        DataValue::Null,
        DataValue::Null,
        DataValue::Bool(false),
        DataValue::from(0),
        DataValue::from(now),
        DataValue::from(now),
        DataValue::Null,
        jd(meta),
    ])
}

/// Blocks that create workflows from `$fl_new` (rows in [`WF_COLS`] order)
/// with their `started` events, tasks and lane rows ([`create_params`]).
fn create_blocks() -> String {
    format!(
        "{{?[{WF_COLS}] <- $fl_new :put {FLOW_WF} {{id => name, version, mode, queue, status, input, output, error, \
           parent, serial, serial_seq, token, seq, owner, lease_until, wait, cancel, attempts, created, updated, \
           finished, meta}}}} \
         {{?[wf, seq, type, key, name, data, at, token] <- $fl_new_ev \
          :put {FLOW_EVENT} {{wf, seq => type, key, name, data, at, token}}}} \
         {{?[wf, queue, due, serial, serial_seq] <- $fl_new_task :put {FLOW_TASK} {{wf => queue, due, serial, serial_seq}}}} \
         {{?[serial, serial_seq, wf] <- $fl_new_lane :put {FLOW_LANE} {{serial, serial_seq => wf}}}}"
    )
}

/// Parameters of [`create_blocks`] for new workflow rows ([`wf_row`]),
/// runnable at `due`.
fn create_params(rows: Vec<DataValue>, due: i64) -> Vec<(&'static str, DataValue)> {
    let (mut evs, mut tasks, mut lanes) = (vec![], vec![], vec![]);
    for r in &rows {
        let DataValue::List(c) = r else { continue };
        // id 0, name 1, queue 4, input 6, serial 10, serial_seq 11, created 19
        evs.push(DataValue::List(vec![
            c[0].clone(),
            DataValue::from(1),
            sv("started"),
            DataValue::Null,
            c[1].clone(),
            c[6].clone(),
            c[19].clone(),
            DataValue::from(0),
        ]));
        tasks.push(DataValue::List(vec![c[0].clone(), c[4].clone(), DataValue::from(due), c[10].clone(), c[11].clone()]));
        if c[10] != DataValue::Null {
            lanes.push(DataValue::List(vec![c[10].clone(), c[11].clone(), c[0].clone()]));
        }
    }
    vec![
        ("fl_new", DataValue::List(rows)),
        ("fl_new_ev", DataValue::List(evs)),
        ("fl_new_task", DataValue::List(tasks)),
        ("fl_new_lane", DataValue::List(lanes)),
    ]
}

/// Blocks that finish workflow `$fl_id` with `$fl_status`, `$fl_output`,
/// `$fl_error`: drop its task, timers and lane row, and wake a waiting
/// parent.
fn finish_blocks() -> String {
    format!(
        "{{?[id, status, output, error, lease_until, wait, finished, updated] <- \
             [[$fl_id, $fl_status, $fl_output, $fl_error, null, null, $fl_now, $fl_now]] \
          :update {FLOW_WF} {{id => status, output, error, lease_until, wait, finished, updated}}}} \
         {{?[wf] <- [[$fl_id]] :rm {FLOW_TASK} {{wf}}}} \
         {{?[wf, key] := *{FLOW_TIMER}{{wf, key}}, wf = $fl_id :rm {FLOW_TIMER} {{wf, key}}}} \
         {{?[serial, serial_seq] := *{FLOW_WF}{{id: $fl_id, serial, serial_seq}}, !is_null(serial) \
          :rm {FLOW_LANE} {{serial, serial_seq}}}} \
         {{?[wf, queue, due, serial, serial_seq] := *{FLOW_WF}{{id: $fl_id, parent: p}}, !is_null(p), \
             *{FLOW_WF}{{id: p, queue, serial, serial_seq, status: 'waiting'}}, wf = p, due = $fl_now \
          :put {FLOW_TASK} {{wf => queue, due, serial, serial_seq}}}}"
    )
}

// ------------------------------------------------------------------ options

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct StartOpts {
    pub workflow: String,
    pub input: JsonValue,
    pub id: Option<String>,
    pub version: Option<i64>,
    pub queue: Option<String>,
    pub serial: Option<String>,
    pub serial_seq: Option<i64>,
    pub run_at: Option<i64>,
    pub mode: Option<String>,
    pub meta: JsonValue,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DefineOpts {
    name: String,
    graph: JsonValue,
    version: Option<i64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct SignalOpts {
    id: String,
    name: String,
    payload: JsonValue,
    sid: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct CancelOpts {
    id: String,
    reason: Option<String>,
    children: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct IdOpts {
    id: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ListOpts {
    status: Option<String>,
    workflow: Option<String>,
    queue: Option<String>,
    parent: Option<String>,
    limit: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ClaimOpts {
    pub worker: Option<String>,
    pub queues: Option<Vec<String>>,
    pub modes: Option<Vec<String>>,
    pub workflows: Option<Vec<String>>,
    pub lanes: Option<Vec<String>>,
    pub lease_ms: Option<i64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct StepOpts {
    id: String,
    token: i64,
    key: String,
    name: Option<String>,
    action: Option<JsonValue>,
    result: Option<JsonValue>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RecordEv {
    #[serde(rename = "type")]
    ty: String,
    key: Option<String>,
    name: Option<String>,
    data: JsonValue,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RecordOpts {
    id: String,
    token: i64,
    events: Vec<RecordEv>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ConsumeOpts {
    id: String,
    token: i64,
    key: String,
    name: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ChildOpts {
    id: String,
    token: i64,
    key: String,
    workflow: String,
    input: JsonValue,
    child_id: Option<String>,
    version: Option<i64>,
    queue: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct TimerSpec {
    pub key: String,
    pub due: i64,
    pub kind: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct SuspendOpts {
    pub id: String,
    pub token: i64,
    pub timers: Vec<TimerSpec>,
    pub signals: Vec<String>,
    pub children: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct CompleteOpts {
    id: String,
    token: i64,
    output: JsonValue,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FailOpts {
    id: String,
    token: i64,
    error: String,
    #[serde(rename = "final")]
    fin: Option<bool>,
    retry_in_ms: Option<i64>,
    canceled: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct HeartbeatOpts {
    id: String,
    token: i64,
    lease_ms: Option<i64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct TickOpts {
    pub worker: Option<String>,
    pub queues: Option<Vec<String>>,
    pub max: Option<usize>,
    pub lease_ms: Option<i64>,
    pub schedules: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ScheduleOpts {
    name: String,
    cron: String,
    workflow: String,
    version: Option<i64>,
    input: JsonValue,
    queue: Option<String>,
    overlap: Option<String>,
    enabled: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct NameOpts {
    name: String,
}

fn from_args<T: serde::de::DeserializeOwned>(op: &str, args: JsonValue) -> Result<T> {
    let args = if args.is_null() { json!({}) } else { args };
    serde_json::from_value(args).map_err(|e| miette!("flow {op}: bad options: {e}"))
}

// ------------------------------------------------------------------ engine

impl<'s, S: Storage<'s>> Db<S> {
    fn fl_r(&'s self, script: &str, p: Vec<(&str, DataValue)>) -> Result<NamedRows> {
        self.run_script(script, params(p), ScriptMutability::Immutable)
    }

    /// A write, retried while it loses to concurrent writers (SQLite busy,
    /// optimistic conflicts).
    pub(crate) fn fl_w(&'s self, script: &str, p: Vec<(&str, DataValue)>) -> Result<NamedRows> {
        let p = params(p);
        let mut attempt = 0;
        loop {
            match self.run_script(script, p.clone(), ScriptMutability::Mutable) {
                Ok(r) => return Ok(r),
                Err(e) if attempt < 10 && is_contention(&e) => {
                    std::thread::sleep(Duration::from_millis(1 << attempt.min(6)));
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Create the flow relations that do not exist yet.
    pub(crate) fn flow_ensure(&'s self) -> Result<()> {
        let missing: Vec<&(&str, &str)> = {
            let tx = self.transact()?;
            DDL.iter().filter(|(n, _)| tx.get_relation(n, false).is_err()).collect()
        };
        for (name, cols) in missing {
            if let Err(e) = self.run_script(&format!(":create {name} {cols}"), Default::default(), ScriptMutability::Mutable) {
                // another process created it meanwhile
                let tx = self.transact()?;
                if tx.get_relation(name, false).is_err() {
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// The workflow row as JSON, or `None`.
    pub(crate) fn flow_wf(&'s self, id: &str) -> Result<Option<serde_json::Map<String, JsonValue>>> {
        let r = self.fl_r(&format!("?[{WF_COLS}] := *{FLOW_WF}{{{WF_COLS}}}, id = $id"), vec![("id", sv(id))])?;
        Ok(r.rows.first().map(|row| {
            r.headers
                .iter()
                .zip(row.iter())
                .map(|(h, v)| (h.clone(), jv(v)))
                .collect()
        }))
    }

    /// Why a worker write failed: `lease lost`, `canceled`, or the error.
    fn fl_diagnose(&'s self, id: &str, token: i64, e: miette::Report) -> miette::Report {
        match self.flow_wf(id) {
            Ok(Some(w)) => {
                if w.get("token").and_then(|t| t.as_i64()) != Some(token) {
                    miette!("lease lost: workflow '{id}' was claimed by another worker (token {token} is stale)")
                } else if w.get("cancel").and_then(|c| c.as_bool()) == Some(true) {
                    miette!("canceled: workflow '{id}' was canceled")
                } else if w
                    .get("status")
                    .and_then(|s| s.as_str())
                    .map_or(false, |s| FINISHED.contains(&s))
                {
                    miette!("lease lost: workflow '{id}' is already finished")
                } else {
                    e
                }
            }
            Ok(None) => miette!("no workflow '{id}'"),
            Err(_) => e,
        }
    }

    /// History of a workflow, in order.
    pub(crate) fn flow_history(&'s self, id: &str) -> Result<Vec<Event>> {
        let r = self.fl_r(
            &format!(
                "?[seq, type, key, name, data, at, token] := *{FLOW_EVENT}{{wf: $id, seq, type, key, name, data, at, token}} \
                 :order seq"
            ),
            vec![("id", sv(id))],
        )?;
        Ok(r.rows
            .into_iter()
            .map(|x| Event {
                seq: x[0].get_int().unwrap_or(0),
                ty: x[1].get_str().unwrap_or_default().to_string(),
                key: x[2].get_str().map(str::to_string),
                name: x[3].get_str().map(str::to_string),
                data: jv(&x[4]),
                at: x[5].get_int().unwrap_or(0),
                token: x[6].get_int().unwrap_or(0),
            })
            .collect())
    }

    /// The recorded command events of `key`.
    fn fl_key_events(&'s self, id: &str, key: &str) -> Result<Vec<Event>> {
        let r = self.fl_r(
            &format!(
                "?[seq, type, name, data, at, token] := *{FLOW_EVENT}{{wf: $id, seq, type, key, name, data, at, token}}, \
                 key == $key :order seq"
            ),
            vec![("id", sv(id)), ("key", sv(key))],
        )?;
        Ok(r.rows
            .into_iter()
            .map(|x| Event {
                seq: x[0].get_int().unwrap_or(0),
                ty: x[1].get_str().unwrap_or_default().to_string(),
                key: Some(key.to_string()),
                name: x[2].get_str().map(str::to_string),
                data: jv(&x[3]),
                at: x[4].get_int().unwrap_or(0),
                token: x[5].get_int().unwrap_or(0),
            })
            .collect())
    }

    /// Record `evs` for the worker holding `token` (fenced; refused after a
    /// cancel request when `check_cancel`).
    pub(crate) fn flow_record_evs(&'s self, id: &str, token: i64, evs: &[Ev], check_cancel: bool) -> Result<()> {
        if evs.is_empty() {
            return Ok(());
        }
        let (script, p) = record_blocks(id, token, evs, check_cancel);
        self.fl_w(&script, p).map(|_| ()).map_err(|e| self.fl_diagnose(id, token, e))
    }

    // ---------------------------------------------------------- client ops

    /// Start a workflow; an existing `id` returns that workflow instead.
    pub(crate) fn flow_start(&'s self, o: StartOpts) -> Result<JsonValue> {
        self.flow_ensure()?;
        if o.workflow.trim().is_empty() {
            bail!("flow start needs 'workflow'");
        }
        let id = o.id.clone().unwrap_or_else(|| new_id("wf"));
        if let Some(w) = self.flow_wf(&id)? {
            return Ok(json!({"id": id, "status": w["status"], "created": w["created"], "existing": true}));
        }
        let (mode, version) = self.flow_resolve_def(&o.workflow, o.version, o.mode.as_deref())?;
        let now = now_us();
        let queue = o.queue.clone().unwrap_or_else(|| "default".to_string());
        let serial_seq = o.serial.as_ref().map(|_| o.serial_seq.unwrap_or(now));
        let row = wf_row(
            &id,
            &o.workflow,
            version,
            &mode,
            &queue,
            &o.input,
            None,
            o.serial.as_deref(),
            serial_seq,
            now,
            &o.meta,
        );
        let script = format!("{{?[x] := *{FLOW_WF}{{id: $fl_nid}}, x = 1 :assert none}}{}", create_blocks());
        let mut p = vec![("fl_nid", sv(&id))];
        p.extend(create_params(vec![row], o.run_at.unwrap_or(now)));
        let res = self.fl_w(&script, p);
        match res {
            Ok(_) => Ok(json!({"id": id, "status": "pending", "created": now, "mode": mode, "version": version})),
            Err(e) => match self.flow_wf(&id)? {
                Some(w) => Ok(json!({"id": id, "status": w["status"], "created": w["created"], "existing": true})),
                None => Err(e),
            },
        }
    }

    /// `(mode, version)` of a workflow name: a registered graph is built-in
    /// (latest version unless given), anything else is code mode.
    fn flow_resolve_def(&'s self, name: &str, version: Option<i64>, mode: Option<&str>) -> Result<(String, i64)> {
        if let Some(m) = mode {
            if m != "code" && m != "builtin" {
                bail!("flow start: mode must be 'code' or 'builtin', got '{m}'");
            }
        }
        if mode == Some("code") {
            return Ok(("code".to_string(), version.unwrap_or(1)));
        }
        let r = self.fl_r(
            &format!("?[version] := *{FLOW_DEF}{{name: $n, version}}"),
            vec![("n", sv(name))],
        )?;
        let versions: Vec<i64> = r.rows.iter().filter_map(|x| x[0].get_int()).collect();
        match (versions.iter().max(), version) {
            (None, _) if mode == Some("builtin") => bail!("flow start: no built-in workflow '{name}' is defined"),
            (None, v) => Ok(("code".to_string(), v.unwrap_or(1))),
            (Some(_), Some(v)) if versions.contains(&v) => Ok(("builtin".to_string(), v)),
            (Some(_), Some(v)) => bail!("flow start: workflow '{name}' has no version {v}"),
            (Some(&m), None) => Ok(("builtin".to_string(), m)),
        }
    }

    fn flow_define(&'s self, o: DefineOpts) -> Result<JsonValue> {
        self.flow_ensure()?;
        if o.name.trim().is_empty() {
            bail!("flow define needs 'name'");
        }
        crate::flow_builtin::validate(&o.graph)?;
        let r = self.fl_r(
            &format!("?[max(version)] := *{FLOW_DEF}{{name: $n, version}}"),
            vec![("n", sv(&o.name))],
        )?;
        let latest = r.rows.first().and_then(|x| x[0].get_int()).unwrap_or(0);
        let version = o.version.unwrap_or(latest + 1);
        self.fl_w(
            &format!(
                "{{?[x] := *{FLOW_DEF}{{name: $n, version: $v}}, x = 1 :assert none}} \
                 {{?[name, version, mode, graph, created] <- [[$n, $v, 'builtin', $g, $now]] \
                  :put {FLOW_DEF} {{name, version => mode, graph, created}}}}"
            ),
            vec![
                ("n", sv(&o.name)),
                ("v", DataValue::from(version)),
                ("g", jd(&o.graph)),
                ("now", DataValue::from(now_us())),
            ],
        )
        .map_err(|e| miette!("flow define: version {version} of '{}' exists or could not be written: {e}", o.name))?;
        Ok(json!({"name": o.name, "version": version}))
    }

    /// The graph of a built-in workflow version.
    pub(crate) fn flow_graph(&'s self, name: &str, version: i64) -> Result<JsonValue> {
        let r = self.fl_r(
            &format!("?[graph] := *{FLOW_DEF}{{name: $n, version: $v, graph}}"),
            vec![("n", sv(name)), ("v", DataValue::from(version))],
        )?;
        r.rows
            .first()
            .map(|x| jv(&x[0]))
            .ok_or_else(|| miette!("no built-in workflow '{name}' version {version}"))
    }

    fn flow_signal(&'s self, o: SignalOpts) -> Result<JsonValue> {
        self.flow_ensure()?;
        let w = self.flow_wf(&o.id)?.ok_or_else(|| miette!("flow signal: no workflow '{}'", o.id))?;
        if o.name.trim().is_empty() {
            bail!("flow signal needs 'name'");
        }
        let st = w["status"].as_str().unwrap_or_default().to_string();
        if FINISHED.contains(&st.as_str()) {
            bail!("flow signal: workflow '{}' is {st}", o.id);
        }
        let sid = o.sid.clone().unwrap_or_else(|| new_id("sig"));
        let now = now_us();
        self.fl_w(
            &format!(
                "{{c[wf, sid, name, payload, at, consumed] <- [[$id, $sid, $name, $payload, $now, null]] \
                   ?[wf, sid, name, payload, at, consumed] := c[wf, sid, name, payload, at, consumed], \
                     not *{FLOW_SIGNAL}{{wf, sid}} \
                  :put {FLOW_SIGNAL} {{wf, sid => name, payload, at, consumed}}}} \
                 {{?[wf, queue, due, serial, serial_seq] := *{FLOW_WF}{{id: $id, queue, serial, serial_seq, status: 'waiting'}}, \
                   wf = $id, due = $now :put {FLOW_TASK} {{wf => queue, due, serial, serial_seq}}}}"
            ),
            vec![
                ("id", sv(&o.id)),
                ("sid", sv(&sid)),
                ("name", sv(&o.name)),
                ("payload", jd(&o.payload)),
                ("now", DataValue::from(now)),
            ],
        )?;
        Ok(json!({"id": o.id, "sid": sid}))
    }

    fn flow_cancel(&'s self, o: CancelOpts) -> Result<JsonValue> {
        self.flow_ensure()?;
        let reason = o.reason.clone().unwrap_or_else(|| "canceled".to_string());
        let mut attempt = 0;
        loop {
            let w = self.flow_wf(&o.id)?.ok_or_else(|| miette!("flow cancel: no workflow '{}'", o.id))?;
            let st = w["status"].as_str().unwrap_or_default().to_string();
            if FINISHED.contains(&st.as_str()) {
                break;
            }
            let token = w["token"].as_i64().unwrap_or(0);
            let res = if st == "running" {
                // the worker sees it at its next command
                let (mut script, p) =
                    record_blocks(&o.id, token, &[ev("cancel_requested", None, None, json!({"reason": reason}))], false);
                script.push_str(&format!(
                    "{{?[id, cancel] <- [[$fl_id, true]] :update {FLOW_WF} {{id => cancel}}}}"
                ));
                self.fl_w(&script, p)
            } else {
                let (mut script, mut p) =
                    record_blocks(&o.id, token, &[ev("canceled", None, None, json!({"reason": reason}))], false);
                script.push_str(&format!(
                    "{{?[x] := *{FLOW_WF}{{id: $fl_id, status}}, status == 'running', x = 1 :assert none}}"
                ));
                script.push_str(&finish_blocks());
                script.push_str(&format!(
                    "{{?[id, cancel] <- [[$fl_id, true]] :update {FLOW_WF} {{id => cancel}}}}"
                ));
                p.extend([
                    ("fl_status", sv("canceled")),
                    ("fl_output", DataValue::Null),
                    ("fl_error", sv(&reason)),
                ]);
                self.fl_w(&script, p)
            };
            match res {
                Ok(_) => break,
                // it moved under us (claimed, finished): look again
                Err(e) if attempt >= 5 => return Err(e),
                Err(_) => attempt += 1,
            }
        }
        if o.children.unwrap_or(true) {
            let r = self.fl_r(
                &format!("?[id, status] := *{FLOW_WF}{{id, parent, status}}, parent == $p"),
                vec![("p", sv(&o.id))],
            )?;
            for row in r.rows {
                let st = row[1].get_str().unwrap_or_default();
                if !FINISHED.contains(&st) {
                    self.flow_cancel(CancelOpts {
                        id: row[0].get_str().unwrap_or_default().to_string(),
                        reason: Some(reason.clone()),
                        children: Some(true),
                    })?;
                }
            }
        }
        let w = self.flow_wf(&o.id)?.unwrap_or_default();
        Ok(json!({"id": o.id, "status": w.get("status"), "cancel": w.get("cancel")}))
    }

    fn flow_status(&'s self, id: &str) -> Result<JsonValue> {
        self.flow_ensure()?;
        let mut w = self.flow_wf(id)?.ok_or_else(|| miette!("flow status: no workflow '{id}'"))?;
        let t = self.fl_r(
            &format!("?[key, due, kind] := *{FLOW_TIMER}{{wf: $id, key, due, kind}} :order due"),
            vec![("id", sv(id))],
        )?;
        w.insert(
            "timers".to_string(),
            JsonValue::Array(
                t.rows
                    .iter()
                    .map(|r| json!({"key": jv(&r[0]), "due": jv(&r[1]), "kind": jv(&r[2])}))
                    .collect(),
            ),
        );
        let task = self.fl_r(&format!("?[due] := *{FLOW_TASK}{{wf: $id, due}}"), vec![("id", sv(id))])?;
        w.insert("due".to_string(), task.rows.first().map(|r| jv(&r[0])).unwrap_or(JsonValue::Null));
        Ok(JsonValue::Object(w))
    }

    fn flow_list(&'s self, o: ListOpts) -> Result<JsonValue> {
        self.flow_ensure()?;
        let mut conds = vec![];
        let mut p = vec![];
        if let Some(s) = &o.status {
            conds.push("status == $status");
            p.push(("status", sv(s)));
        }
        if let Some(s) = &o.workflow {
            conds.push("name == $wfname");
            p.push(("wfname", sv(s)));
        }
        if let Some(s) = &o.queue {
            conds.push("queue == $queue");
            p.push(("queue", sv(s)));
        }
        if let Some(s) = &o.parent {
            conds.push("parent == $parent");
            p.push(("parent", sv(s)));
        }
        let filter = if conds.is_empty() { String::new() } else { format!(", {}", conds.join(", ")) };
        let cols = "id, name, version, mode, queue, status, parent, serial, serial_seq, attempts, error, created, updated, finished";
        let r = self.fl_r(
            &format!(
                "?[{cols}] := *{FLOW_WF}{{{cols}}}{filter} :order -created :limit {}",
                o.limit.unwrap_or(100)
            ),
            p,
        )?;
        Ok(JsonValue::Array(
            r.rows
                .iter()
                .map(|row| {
                    JsonValue::Object(r.headers.iter().zip(row).map(|(h, v)| (h.clone(), jv(v))).collect())
                })
                .collect(),
        ))
    }

    fn flow_stats(&'s self) -> Result<JsonValue> {
        self.flow_ensure()?;
        let now = now_us();
        let r = self.fl_r(
            &format!("?[queue, status, count(id)] := *{FLOW_WF}{{id, queue, status}}"),
            vec![],
        )?;
        let mut by: BTreeMap<String, serde_json::Map<String, JsonValue>> = BTreeMap::new();
        for row in r.rows {
            by.entry(row[0].get_str().unwrap_or_default().to_string())
                .or_default()
                .insert(row[1].get_str().unwrap_or_default().to_string(), jv(&row[2]));
        }
        let due = self.fl_r(
            &format!("?[min(due), count(wf)] := *{FLOW_TASK}{{wf, due}}, due <= $now"),
            vec![("now", DataValue::from(now))],
        )?;
        let stale = self.fl_r(
            &format!(
                "?[count(id)] := *{FLOW_WF}{{id, status: 'running', lease_until: l}}, !is_null(l), l < $now"
            ),
            vec![("now", DataValue::from(now))],
        )?;
        Ok(json!({
            "queues": by,
            "due_now": due.rows.first().map(|x| jv(&x[1])).unwrap_or(json!(0)),
            "oldest_due": due.rows.first().map(|x| jv(&x[0])).unwrap_or(JsonValue::Null),
            "running_past_lease": stale.rows.first().map(|x| jv(&x[0])).unwrap_or(json!(0)),
        }))
    }

    // ---------------------------------------------------------- worker ops

    /// Claim one runnable workflow, or `None`.
    pub(crate) fn flow_claim(&'s self, o: &ClaimOpts) -> Result<Option<Claimed>> {
        self.flow_ensure()?;
        let worker = o.worker.clone().unwrap_or_else(|| format!("w-{}", std::process::id()));
        let lease = o.lease_ms.unwrap_or(DEFAULT_LEASE_MS).max(1) * 1000;
        let queues: Vec<DataValue> = o
            .queues
            .clone()
            .unwrap_or_else(|| vec!["default".to_string()])
            .iter()
            .map(|q| sv(q))
            .collect();
        let modes: Vec<DataValue> = o
            .modes
            .clone()
            .unwrap_or_else(|| vec!["code".to_string(), "builtin".to_string()])
            .iter()
            .map(|q| sv(q))
            .collect();
        let (mut wf_filter, mut p) = match &o.workflows {
            Some(names) => (
                ", is_in(name, $names)".to_string(),
                vec![("names", DataValue::List(names.iter().map(|n| sv(n)).collect()))],
            ),
            None => (String::new(), vec![]),
        };
        if let Some(lanes) = &o.lanes {
            wf_filter.push_str(", is_in(serial, $lanes)");
            p.push(("lanes", DataValue::List(lanes.iter().map(|n| sv(n)).collect())));
        }
        let now = now_us();
        p.extend([
            ("now", DataValue::from(now)),
            ("queues", DataValue::List(queues)),
            ("modes", DataValue::List(modes)),
        ]);
        let cands = self.fl_r(
            &format!(
                "c[wf, due, token, status, serial, sseq] := *{FLOW_TASK}{{wf, queue, due, serial, serial_seq: sseq}}, \
                   due <= $now, is_in(queue, $queues), \
                   *{FLOW_WF}{{id: wf, token, status, mode, name}}, is_in(mode, $modes){wf_filter} \
                 blocked[wf] := c[wf, _, _, _, serial, sseq], !is_null(serial), \
                   *{FLOW_LANE}{{serial, serial_seq: s2}}, s2 < sseq \
                 ?[wf, due, token, status] := c[wf, due, token, status, _, _], not blocked[wf] \
                 :order due :limit 16"
            ),
            p,
        )?;
        for row in cands.rows {
            let id = row[0].get_str().unwrap_or_default().to_string();
            let token = row[2].get_int().unwrap_or(0);
            let lease_until = now + lease;
            let res = self.fl_w(
                &format!(
                    "{{?[x] := *{FLOW_TASK}{{wf: $id, due}}, due <= $now, x = 1 :assert some}} \
                     {{?[x] := *{FLOW_WF}{{id: $id, token: t}}, t == $token, x = 1 :assert some}} \
                     {{c[id, status, owner, lease_until, token, updated] <- [[$id, 'running', $worker, $lease, $token1, $now]] \
                       ?[id, status, owner, lease_until, token, attempts, updated] := \
                         c[id, status, owner, lease_until, token, updated], *{FLOW_WF}{{id, attempts: a}}, attempts = a + 1 \
                      :update {FLOW_WF} {{id => status, owner, lease_until, token, attempts, updated}}}} \
                     {{?[wf, due] <- [[$id, $lease]] :update {FLOW_TASK} {{wf => due}}}}"
                ),
                vec![
                    ("id", sv(&id)),
                    ("now", DataValue::from(now)),
                    ("token", DataValue::from(token)),
                    ("token1", DataValue::from(token + 1)),
                    ("worker", sv(&worker)),
                    ("lease", DataValue::from(lease_until)),
                ],
            );
            if res.is_err() {
                continue; // another worker took it
            }
            let w = self.flow_wf(&id)?.ok_or_else(|| miette!("claimed workflow '{id}' vanished"))?;
            return Ok(Some(Claimed {
                id: id.clone(),
                name: w["name"].as_str().unwrap_or_default().to_string(),
                version: w["version"].as_i64().unwrap_or(1),
                mode: w["mode"].as_str().unwrap_or_default().to_string(),
                queue: w["queue"].as_str().unwrap_or_default().to_string(),
                input: w.get("input").cloned().unwrap_or(JsonValue::Null),
                token: token + 1,
                cancel: w["cancel"].as_bool().unwrap_or(false),
                now,
                history: self.flow_history(&id)?,
            }));
        }
        Ok(None)
    }

    /// The result recorded for a step key, checking the command's name.
    fn fl_replayed(&'s self, id: &str, key: &str, name: Option<&str>) -> Result<Option<JsonValue>> {
        for e in self.fl_key_events(id, key)? {
            if e.ty == "step" {
                if name.is_some() && e.name.as_deref() != name {
                    bail!(
                        "non-determinism: workflow '{id}' step '{key}' was '{}' in the history, now '{}'",
                        e.name.unwrap_or_default(),
                        name.unwrap_or_default()
                    );
                }
                return Ok(Some(e.data));
            }
            if !matches!(e.ty.as_str(), "step_failed") {
                bail!("non-determinism: workflow '{id}' key '{key}' is a '{}' in the history, now a step", e.ty);
            }
        }
        Ok(None)
    }

    /// Run (or replay) step `key`: an engine `action` runs atomically with
    /// its event; a caller's `result` is recorded as is.
    pub(crate) fn flow_step(
        &'s self,
        id: &str,
        token: i64,
        key: &str,
        name: Option<&str>,
        action: Option<&JsonValue>,
        result: Option<&JsonValue>,
    ) -> Result<(JsonValue, bool)> {
        if let Some(v) = self.fl_replayed(id, key, name)? {
            return Ok((v, true));
        }
        let v = match (action, result) {
            (Some(a), _) => self.flow_action(id, token, key, name, a)?,
            (None, Some(r)) => {
                self.flow_record_evs(id, token, &[ev("step", Some(key), name, r.clone())], true)?;
                r.clone()
            }
            (None, None) => bail!("flow step needs 'action' or 'result'"),
        };
        Ok((v, false))
    }

    /// Execute an engine action and record its result under `key`, in one
    /// transaction where the action writes.
    fn flow_action(&'s self, id: &str, token: i64, key: &str, name: Option<&str>, action: &JsonValue) -> Result<JsonValue> {
        let obj = action.as_object().ok_or_else(|| miette!("flow step: 'action' must be an object"))?;
        if obj.len() != 1 {
            bail!("flow step: 'action' must have exactly one kind (query, remember, recall, embed, decide, http)");
        }
        let (kind, args) = obj.iter().next().unwrap();
        let name = name.unwrap_or(kind.as_str());
        let record = |v: JsonValue| -> Result<JsonValue> {
            self.flow_record_evs(id, token, &[ev("step", Some(key), Some(name), v.clone())], true)?;
            Ok(v)
        };
        match kind.as_str() {
            "query" => self.flow_action_query(id, token, key, name, args),
            "remember" => self.flow_action_remember(id, token, key, name, args),
            "recall" => record(self.memory_call("recall", args.clone())?),
            "embed" => {
                let texts = flatten_texts(args.get("texts").unwrap_or(&JsonValue::Null))?;
                let model = match args.get("embedder").and_then(|x| x.as_str()) {
                    Some(m) => m.to_string(),
                    None => crate::embed::embedders()
                        .into_iter()
                        .next()
                        .map(|e| e.name)
                        .ok_or_else(|| miette!("flow embed: no embedder is enabled"))?,
                };
                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                let vecs = crate::embed::embed_texts(&model, &refs)?;
                record(JsonValue::Array(
                    vecs.iter()
                        .map(|v| JsonValue::Array(v.iter().map(|x| json!(*x as f64)).collect()))
                        .collect(),
                ))
            }
            "decide" => {
                let template = args
                    .get("template")
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| miette!("flow decide needs 'template'"))?;
                let pairs = args
                    .get("pairs")
                    .and_then(|x| x.as_array())
                    .ok_or_else(|| miette!("flow decide needs 'pairs': [[a, b], ...]"))?;
                let items: Vec<(String, String)> = pairs
                    .iter()
                    .map(|p| {
                        let a = p.get(0).and_then(|x| x.as_str()).unwrap_or_default().to_string();
                        let b = p.get(1).and_then(|x| x.as_str()).unwrap_or_default().to_string();
                        (a, b)
                    })
                    .collect();
                let refs: Vec<(&str, &str)> = items.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
                let decider = match args.get("decider").and_then(|x| x.as_str()) {
                    Some(d) => d.to_string(),
                    None => crate::decide::deciders()
                        .into_iter()
                        .next()
                        .map(|d| d.name)
                        .ok_or_else(|| miette!("flow decide: no decider is enabled"))?,
                };
                let out = crate::decide::decide(&decider, crate::decide::Template::parse(template)?, &refs)?;
                record(serde_json::to_value(&out).into_diagnostic()?)
            }
            "http" => record(flow_http(args)?),
            other => bail!("flow step: unknown action '{other}' (query, remember, recall, embed, decide, http)"),
        }
    }

    /// `query {script, params}`: the script and the step's event in one
    /// transaction (a multi-statement transaction on a helper thread).
    fn flow_action_query(&'s self, id: &str, token: i64, key: &str, name: &str, args: &JsonValue) -> Result<JsonValue> {
        let script = args
            .get("script")
            .and_then(|x| x.as_str())
            .ok_or_else(|| miette!("flow query needs 'script'"))?
            .to_string();
        let qp: BTreeMap<String, DataValue> = match args.get("params") {
            None | Some(JsonValue::Null) => BTreeMap::new(),
            Some(JsonValue::Object(m)) => m.iter().map(|(k, v)| (k.clone(), DataValue::from(v))).collect(),
            Some(_) => bail!("flow query: 'params' must be an object"),
        };
        let rows_json = |rows: &NamedRows| {
            json!({"headers": rows.headers, "rows": rows.rows.iter().map(|r| r.iter().map(jv).collect::<Vec<_>>()).collect::<Vec<_>>()})
        };
        let mut attempt = 0;
        loop {
            let (pay_tx, pay_rx) = bounded(1);
            let (res_tx, res_rx) = bounded(1);
            let (script, qp) = (&script, &qp);
            let out: Result<JsonValue> = std::thread::scope(move |sc| {
                // `pay_tx` is owned here: any early return drops it, which ends
                // the transaction thread and rolls the transaction back
                let pay_tx = pay_tx;
                sc.spawn(move || self.run_multi_transaction(true, pay_rx, res_tx));
                let gone = || miette!("flow query: the transaction ended early");
                let run = |q: String, p: BTreeMap<String, DataValue>| -> Result<NamedRows> {
                    pay_tx.send(TransactionPayload::Query((q, p))).map_err(|_| gone())?;
                    res_rx.recv().map_err(|_| gone())?
                };
                // 1. the step's script
                let rows = run(script.clone(), qp.clone())?;
                let result = rows_json(&rows);
                // 2. its event, fenced, in the same transaction
                let (parts, p) = record_parts(id, token, &[ev("step", Some(key), Some(name), result.clone())], true);
                let p: BTreeMap<String, DataValue> = p.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
                for q in parts {
                    run(q, p.clone())?;
                }
                // 3. commit both
                pay_tx.send(TransactionPayload::Commit).map_err(|_| gone())?;
                res_rx.recv().map_err(|_| gone())??;
                Ok(result)
            });
            match out {
                Ok(v) => return Ok(v),
                Err(e) if attempt < 10 && is_contention(&e) => {
                    std::thread::sleep(Duration::from_millis(1 << attempt.min(6)));
                    attempt += 1;
                }
                Err(e) => return Err(self.fl_diagnose(id, token, e)),
            }
        }
    }

    /// `remember {text, …}`: the memory and the step's event commit in
    /// `remember`'s own transaction (`Db::remember_with`).
    fn flow_action_remember(&'s self, id: &str, token: i64, key: &str, name: &str, args: &JsonValue) -> Result<JsonValue> {
        let mut m = args.as_object().cloned().ok_or_else(|| miette!("flow remember: options must be an object"))?;
        let text = match m.remove("text") {
            Some(JsonValue::String(t)) => t,
            _ => bail!("flow remember needs 'text'"),
        };
        let opts: crate::memory::RememberOpts =
            serde_json::from_value(JsonValue::Object(m)).map_err(|e| miette!("flow remember: bad options: {e}"))?;
        let extra = |res: &crate::memory::RememberResult| -> (String, Vec<(&'static str, DataValue)>) {
            let v = serde_json::to_value(res).unwrap_or(JsonValue::Null);
            record_blocks(id, token, &[ev("step", Some(key), Some(name), v)], true)
        };
        let res = self
            .remember_with(&text, opts, Some(&extra))
            .map_err(|e| self.fl_diagnose(id, token, e))?;
        serde_json::to_value(res).into_diagnostic()
    }

    /// Record `events`, skipping those whose `(key, type)` already exist;
    /// returns each event's recorded data (the existing one wins).
    pub(crate) fn flow_record(&'s self, id: &str, token: i64, events: Vec<Ev>) -> Result<Vec<JsonValue>> {
        let mut out = vec![];
        let mut new = vec![];
        let mut existing: BTreeMap<String, Vec<Event>> = BTreeMap::new();
        for e in &events {
            if let Some(k) = &e.1 {
                if !existing.contains_key(k) {
                    existing.insert(k.clone(), self.fl_key_events(id, k)?);
                }
            }
        }
        for e in events {
            // one event per (key, type), except failures: one per attempt
            let found = e.1.as_ref().and_then(|k| existing.get(k)).and_then(|evs| {
                evs.iter().find(|x| {
                    x.ty == e.0 && (x.ty != "step_failed" || x.data.get("attempt") == e.3.get("attempt"))
                })
            });
            match found {
                Some(x) => {
                    if e.2.is_some() && x.name != e.2 {
                        bail!(
                            "non-determinism: workflow '{id}' key '{}' was '{}' in the history, now '{}'",
                            e.1.unwrap_or_default(),
                            x.name.clone().unwrap_or_default(),
                            e.2.unwrap_or_default()
                        );
                    }
                    out.push(x.data.clone())
                }
                None => {
                    out.push(e.3.clone());
                    new.push(e);
                }
            }
        }
        self.flow_record_evs(id, token, &new, true)?;
        Ok(out)
    }

    /// Take the oldest unconsumed signal `name` for command `key` (or the
    /// one it already took); `None` when there is none.
    pub(crate) fn flow_consume_signal(&'s self, id: &str, token: i64, key: &str, name: &str) -> Result<Option<JsonValue>> {
        for e in self.fl_key_events(id, key)? {
            if e.ty == "signal" {
                return Ok(Some(e.data.get("payload").cloned().unwrap_or(JsonValue::Null)));
            }
        }
        for _ in 0..5 {
            let r = self.fl_r(
                &format!(
                    "?[at, sid, payload] := *{FLOW_SIGNAL}{{wf: $id, sid, name, payload, at, consumed}}, \
                     name == $name, is_null(consumed) :order at, sid :limit 1"
                ),
                vec![("id", sv(id)), ("name", sv(name))],
            )?;
            let Some(row) = r.rows.first() else { return Ok(None) };
            let sid = row[1].get_str().unwrap_or_default().to_string();
            let payload = jv(&row[2]);
            let (mut script, mut p) = record_blocks(
                id,
                token,
                &[ev("signal", Some(key), Some(name), json!({"sid": sid, "payload": payload}))],
                true,
            );
            script.push_str(&format!(
                "{{?[x] := *{FLOW_SIGNAL}{{wf: $fl_id, sid: $fl_sid, consumed}}, is_null(consumed), x = 1 :assert some}} \
                 {{c[wf, sid] <- [[$fl_id, $fl_sid]] \
                   ?[wf, sid, consumed] := c[wf, sid], *{FLOW_WF}{{id: wf, seq: consumed}} \
                  :update {FLOW_SIGNAL} {{wf, sid => consumed}}}}"
            ));
            p.push(("fl_sid", sv(&sid)));
            match self.fl_w(&script, p) {
                Ok(_) => return Ok(Some(payload)),
                Err(e) => {
                    let d = self.fl_diagnose(id, token, e);
                    let m = d.to_string();
                    if m.starts_with("lease lost") || m.starts_with("canceled") {
                        return Err(d);
                    }
                    // the signal was taken meanwhile: look again
                }
            }
        }
        bail!("flow: could not consume signal '{name}' of workflow '{id}'")
    }

    /// Start child `workflow` for command `key` (or return the one it
    /// started): the child and the parent's event in one transaction.
    pub(crate) fn flow_start_child(
        &'s self,
        id: &str,
        token: i64,
        key: &str,
        workflow: &str,
        input: &JsonValue,
        child_id: Option<&str>,
        version: Option<i64>,
        queue: Option<&str>,
    ) -> Result<String> {
        for e in self.fl_key_events(id, key)? {
            if e.ty == "child" {
                if e.name.as_deref() != Some(workflow) {
                    bail!(
                        "non-determinism: workflow '{id}' key '{key}' started child '{}' in the history, now '{workflow}'",
                        e.name.unwrap_or_default()
                    );
                }
                return Ok(e.data.get("id").and_then(|x| x.as_str()).unwrap_or_default().to_string());
            }
        }
        let parent = self.flow_wf(id)?.ok_or_else(|| miette!("no workflow '{id}'"))?;
        let cid = child_id.map(str::to_string).unwrap_or_else(|| format!("{id}/{key}"));
        let (mode, version) = self.flow_resolve_def(workflow, version, None)?;
        let now = now_us();
        let queue = queue
            .map(str::to_string)
            .unwrap_or_else(|| parent["queue"].as_str().unwrap_or("default").to_string());
        let row = wf_row(&cid, workflow, version, &mode, &queue, input, Some(id), None, None, now, &JsonValue::Null);
        let (mut script, mut p) =
            record_blocks(id, token, &[ev("child", Some(key), Some(workflow), json!({"id": cid}))], true);
        script.push_str(&format!("{{?[x] := *{FLOW_WF}{{id: $fl_cid}}, x = 1 :assert none}}"));
        script.push_str(&create_blocks());
        p.push(("fl_cid", sv(&cid)));
        p.extend(create_params(vec![row], now));
        self.fl_w(&script, p).map_err(|e| self.fl_diagnose(id, token, e))?;
        Ok(cid)
    }

    /// Park the workflow until its earliest timer, a signal it waits for, or
    /// a child finishing (whichever is first; at once if one already
    /// happened).
    pub(crate) fn flow_suspend(&'s self, o: &SuspendOpts) -> Result<JsonValue> {
        let now = now_us();
        let timers: Vec<DataValue> = o
            .timers
            .iter()
            .map(|t| {
                DataValue::List(vec![
                    sv(&o.id),
                    sv(&t.key),
                    DataValue::from(t.due),
                    sv(t.kind.as_deref().unwrap_or("sleep")),
                ])
            })
            .collect();
        let min_timer = o.timers.iter().map(|t| t.due).min();
        let wait = json!({"timers": o.timers, "signals": o.signals, "children": o.children});
        let script = format!(
            "{}{} \
             {{?[wf, key] := *{FLOW_TIMER}{{wf, key}}, wf = $fl_id :rm {FLOW_TIMER} {{wf, key}}}} \
             {{?[wf, key, due, kind] <- $fl_timers :put {FLOW_TIMER} {{wf, key => due, kind}}}} \
             {{?[id, status, owner, lease_until, wait, updated] <- [[$fl_id, 'waiting', null, null, $fl_wait, $fl_now]] \
              :update {FLOW_WF} {{id => status, owner, lease_until, wait, updated}}}} \
             {{?[wf] <- [[$fl_id]] :rm {FLOW_TASK} {{wf}}}} \
             {{d[t] := *{FLOW_SIGNAL}{{wf: $fl_id, name, consumed}}, is_null(consumed), is_in(name, $fl_signals), t = $fl_now \
               d[t] := *{FLOW_WF}{{id: c, status}}, is_in(c, $fl_children), is_in(status, $fl_finished), t = $fl_now \
               d[t] := t = $fl_min_timer, !is_null(t) \
               m[min(t)] := d[t] \
               ?[wf, queue, due, serial, serial_seq] := m[due], !is_null(due), \
                 *{FLOW_WF}{{id: $fl_id, queue, serial, serial_seq}}, wf = $fl_id \
              :put {FLOW_TASK} {{wf => queue, due, serial, serial_seq}}}}",
            fence(),
            not_canceled(),
        );
        let p = vec![
            ("fl_id", sv(&o.id)),
            ("fl_token", DataValue::from(o.token)),
            ("fl_now", DataValue::from(now)),
            ("fl_timers", DataValue::List(timers)),
            ("fl_wait", jd(&wait)),
            ("fl_signals", DataValue::List(o.signals.iter().map(|s| sv(s)).collect())),
            ("fl_children", DataValue::List(o.children.iter().map(|s| sv(s)).collect())),
            (
                "fl_finished",
                DataValue::List(FINISHED.iter().map(|s| sv(s)).collect()),
            ),
            ("fl_min_timer", min_timer.map(DataValue::from).unwrap_or(DataValue::Null)),
        ];
        self.fl_w(&script, p).map_err(|e| self.fl_diagnose(&o.id, o.token, e))?;
        let due = self.fl_r(&format!("?[due] := *{FLOW_TASK}{{wf: $id, due}}"), vec![("id", sv(&o.id))])?;
        Ok(json!({"id": o.id, "status": "waiting", "due": due.rows.first().map(|r| jv(&r[0]))}))
    }

    /// Finish with `status` (`completed`, `failed`, `canceled`).
    pub(crate) fn flow_finish(
        &'s self,
        id: &str,
        token: i64,
        status: &str,
        output: &JsonValue,
        error: Option<&str>,
    ) -> Result<JsonValue> {
        let now = now_us();
        let evt = match status {
            "completed" => ev("completed", None, None, output.clone()),
            "canceled" => ev("canceled", None, None, json!({"reason": error})),
            _ => ev("failed", None, None, json!({"error": error})),
        };
        let (mut script, mut p) = record_blocks(id, token, &[evt], false);
        script.push_str(&finish_blocks());
        p.retain(|(k, _)| *k != "fl_now");
        p.extend([
            ("fl_now", DataValue::from(now)),
            ("fl_status", sv(status)),
            ("fl_output", jd(output)),
            ("fl_error", error.map(sv).unwrap_or(DataValue::Null)),
        ]);
        self.fl_w(&script, p).map_err(|e| self.fl_diagnose(id, token, e))?;
        Ok(json!({"id": id, "status": status}))
    }

    /// Continue-as-new (Temporal's): end this run and start a fresh history
    /// for the same workflow id with `input`, so a long-running loop's
    /// history stays bounded. The new run's `started` event records the
    /// run number; timers are dropped, buffered signals stay.
    pub(crate) fn flow_continue(&'s self, id: &str, token: i64, input: &JsonValue) -> Result<JsonValue> {
        let w = self.flow_wf(id)?.ok_or_else(|| miette!("no workflow '{id}'"))?;
        let runs = w["meta"]["runs"].as_i64().unwrap_or(1) + 1;
        let mut meta = if w["meta"].is_object() { w["meta"].clone() } else { json!({}) };
        meta["runs"] = json!(runs);
        let now = now_us();
        let script = format!(
            "{}{} \
             {{?[wf, seq] := *{FLOW_EVENT}{{wf, seq}}, wf = $fl_id :rm {FLOW_EVENT} {{wf, seq}}}} \
             {{?[wf, seq, type, key, name, data, at, token] <- [[$fl_id, 1, 'started', null, $fl_name, $fl_input, $fl_now, $fl_token]] \
              :put {FLOW_EVENT} {{wf, seq => type, key, name, data, at, token}}}} \
             {{?[wf, key] := *{FLOW_TIMER}{{wf, key}}, wf = $fl_id :rm {FLOW_TIMER} {{wf, key}}}} \
             {{?[id, status, input, seq, owner, lease_until, wait, attempts, meta, updated] <- \
                 [[$fl_id, 'pending', $fl_input, 1, null, null, null, 0, $fl_meta, $fl_now]] \
              :update {FLOW_WF} {{id => status, input, seq, owner, lease_until, wait, attempts, meta, updated}}}} \
             {{?[wf, queue, due, serial, serial_seq] := *{FLOW_WF}{{id: $fl_id, queue, serial, serial_seq}}, wf = $fl_id, due = $fl_now \
              :put {FLOW_TASK} {{wf => queue, due, serial, serial_seq}}}}",
            fence(),
            not_canceled()
        );
        self.fl_w(
            &script,
            vec![
                ("fl_id", sv(id)),
                ("fl_token", DataValue::from(token)),
                ("fl_name", w["name"].as_str().map(sv).unwrap_or(DataValue::Null)),
                ("fl_input", jd(input)),
                ("fl_meta", jd(&meta)),
                ("fl_now", DataValue::from(now)),
            ],
        )
        .map_err(|e| self.fl_diagnose(id, token, e))?;
        Ok(json!({"id": id, "status": "pending", "continued": true, "run": runs}))
    }

    fn flow_fail(&'s self, o: FailOpts) -> Result<JsonValue> {
        let w = self.flow_wf(&o.id)?.ok_or_else(|| miette!("no workflow '{}'", o.id))?;
        let canceled = o.canceled.unwrap_or(false) || w["cancel"].as_bool().unwrap_or(false);
        if canceled {
            return self.flow_finish(&o.id, o.token, "canceled", &JsonValue::Null, Some(&o.error));
        }
        if o.fin.unwrap_or(true) {
            return self.flow_finish(&o.id, o.token, "failed", &JsonValue::Null, Some(&o.error));
        }
        // a workflow-task error (non-determinism, a worker bug): park it and
        // retry later, so a fixed worker carries on
        let due = now_us() + o.retry_in_ms.unwrap_or(5_000).max(0) * 1000;
        let (mut script, mut p) =
            record_blocks(&o.id, o.token, &[ev("task_error", None, None, json!({"error": o.error}))], false);
        script.push_str(&format!(
            "{{?[id, status, owner, lease_until, error, updated] <- [[$fl_id, 'waiting', null, null, $fl_error, $fl_now]] \
              :update {FLOW_WF} {{id => status, owner, lease_until, error, updated}}}} \
             {{?[wf, due] <- [[$fl_id, $fl_due]] :update {FLOW_TASK} {{wf => due}}}}"
        ));
        p.extend([("fl_error", sv(&o.error)), ("fl_due", DataValue::from(due))]);
        self.fl_w(&script, p).map_err(|e| self.fl_diagnose(&o.id, o.token, e))?;
        Ok(json!({"id": o.id, "status": "waiting", "due": due}))
    }

    fn flow_heartbeat(&'s self, o: HeartbeatOpts) -> Result<JsonValue> {
        let lease_until = now_us() + o.lease_ms.unwrap_or(DEFAULT_LEASE_MS).max(1) * 1000;
        self.fl_w(
            &format!(
                "{}{{?[id, lease_until] <- [[$fl_id, $fl_lease]] :update {FLOW_WF} {{id => lease_until}}}} \
                 {{?[wf, due] <- [[$fl_id, $fl_lease]] :update {FLOW_TASK} {{wf => due}}}}",
                fence()
            ),
            vec![
                ("fl_id", sv(&o.id)),
                ("fl_token", DataValue::from(o.token)),
                ("fl_lease", DataValue::from(lease_until)),
            ],
        )
        .map_err(|e| self.fl_diagnose(&o.id, o.token, e))?;
        let w = self.flow_wf(&o.id)?.unwrap_or_default();
        Ok(json!({"id": o.id, "lease_until": lease_until, "cancel": w.get("cancel")}))
    }

    // ---------------------------------------------------------- schedules

    fn flow_schedule(&'s self, o: ScheduleOpts) -> Result<JsonValue> {
        self.flow_ensure()?;
        if o.name.trim().is_empty() || o.workflow.trim().is_empty() {
            bail!("flow schedule needs 'name', 'cron' and 'workflow'");
        }
        let cron = crate::flow_cron::Cron::parse(&o.cron)?;
        let overlap = o.overlap.clone().unwrap_or_else(|| "skip".to_string());
        if overlap != "skip" && overlap != "allow" {
            bail!("flow schedule: overlap must be 'skip' or 'allow'");
        }
        let now = now_us();
        let next = cron.next_after(now)?;
        let created = self
            .fl_r(&format!("?[created] := *{FLOW_SCHEDULE}{{name: $n, created}}"), vec![("n", sv(&o.name))])?
            .rows
            .first()
            .and_then(|r| r[0].get_int())
            .unwrap_or(now);
        self.fl_w(
            &format!(
                "?[name, workflow, version, cron, input, queue, overlap, next_at, last_id, enabled, created, updated] <- \
                 [[$n, $wf, $v, $cron, $input, $q, $ov, $next, null, $en, $created, $now]] \
                 :put {FLOW_SCHEDULE} {{name => workflow, version, cron, input, queue, overlap, next_at, last_id, enabled, created, updated}}"
            ),
            vec![
                ("n", sv(&o.name)),
                ("wf", sv(&o.workflow)),
                ("v", o.version.map(DataValue::from).unwrap_or(DataValue::Null)),
                ("cron", sv(&o.cron)),
                ("input", jd(&o.input)),
                ("q", sv(o.queue.as_deref().unwrap_or("default"))),
                ("ov", sv(&overlap)),
                ("next", DataValue::from(next)),
                ("en", DataValue::Bool(o.enabled.unwrap_or(true))),
                ("created", DataValue::from(created)),
                ("now", DataValue::from(now)),
            ],
        )?;
        Ok(json!({"name": o.name, "cron": o.cron, "workflow": o.workflow, "next_at": next}))
    }

    fn flow_schedules(&'s self) -> Result<JsonValue> {
        self.flow_ensure()?;
        let cols = "name, workflow, version, cron, input, queue, overlap, next_at, last_id, enabled, created, updated";
        let r = self.fl_r(&format!("?[{cols}] := *{FLOW_SCHEDULE}{{{cols}}}"), vec![])?;
        Ok(JsonValue::Array(
            r.rows
                .iter()
                .map(|row| JsonValue::Object(r.headers.iter().zip(row).map(|(h, v)| (h.clone(), jv(v))).collect()))
                .collect(),
        ))
    }

    /// Fire every enabled schedule whose time has come: start its workflow
    /// (id `sched:<name>:<time>`, so two workers start it once) and advance
    /// `next_at`, in one transaction. Returns the ids started.
    pub(crate) fn flow_fire_schedules(&'s self) -> Result<Vec<String>> {
        let now = now_us();
        let r = self.fl_r(
            &format!(
                "?[name, workflow, version, cron, input, queue, overlap, next_at, last_id] := \
                 *{FLOW_SCHEDULE}{{name, workflow, version, cron, input, queue, overlap, next_at, last_id, enabled}}, \
                 enabled, next_at <= $now"
            ),
            vec![("now", DataValue::from(now))],
        )?;
        let mut started = vec![];
        for row in r.rows {
            let name = row[0].get_str().unwrap_or_default().to_string();
            let workflow = row[1].get_str().unwrap_or_default().to_string();
            let version = row[2].get_int();
            let cron = crate::flow_cron::Cron::parse(row[3].get_str().unwrap_or_default())?;
            let input = jv(&row[4]);
            let queue = row[5].get_str().unwrap_or("default").to_string();
            let overlap = row[6].get_str().unwrap_or("skip").to_string();
            let at = row[7].get_int().unwrap_or(now);
            let last = row[8].get_str().map(str::to_string);
            // missed firings collapse into this one
            let next = cron.next_after(now.max(at))?;
            let busy = match (&last, overlap.as_str()) {
                (Some(l), "skip") => self
                    .flow_wf(l)?
                    .map_or(false, |w| !FINISHED.contains(&w["status"].as_str().unwrap_or("completed"))),
                _ => false,
            };
            let id = format!("sched:{name}:{at}");
            let mut script = format!(
                "{{?[x] := *{FLOW_SCHEDULE}{{name: $sn, next_at}}, next_at == $at, x = 1 :assert some}} \
                 {{?[name, next_at, last_id, updated] <- [[$sn, $next, $lid, $sat]] \
                  :update {FLOW_SCHEDULE} {{name => next_at, last_id, updated}}}}"
            );
            let mut p = vec![
                ("sn", sv(&name)),
                ("at", DataValue::from(at)),
                ("next", DataValue::from(next)),
                ("sat", DataValue::from(now)),
            ];
            if busy {
                p.push(("lid", osv(&last)));
            } else {
                let (mode, v) = self.flow_resolve_def(&workflow, version, None)?;
                let wrow = wf_row(&id, &workflow, v, &mode, &queue, &input, None, None, None, now, &json!({"schedule": name}));
                script.push_str(&format!("{{?[x] := *{FLOW_WF}{{id: $fl_nid}}, x = 1 :assert none}}"));
                script.push_str(&create_blocks());
                p.extend([("lid", sv(&id)), ("fl_nid", sv(&id))]);
                p.extend(create_params(vec![wrow], now));
            }
            if self.fl_w(&script, p).is_ok() && !busy {
                started.push(id);
            }
        }
        Ok(started)
    }

    /// Earliest due time of work in `queues` (tasks and schedules), if any.
    pub(crate) fn flow_next_due(&'s self, queues: &[String]) -> Result<Option<i64>> {
        self.flow_ensure()?;
        let q: Vec<DataValue> = queues.iter().map(|x| sv(x)).collect();
        let r = self.fl_r(
            &format!(
                "d[due] := *{FLOW_TASK}{{queue, due}}, is_in(queue, $q) \
                 d[due] := *{FLOW_SCHEDULE}{{next_at: due, enabled}}, enabled \
                 ?[min(due)] := d[due]"
            ),
            vec![("q", DataValue::List(q))],
        )?;
        Ok(r.rows.first().and_then(|x| x[0].get_int()))
    }

    // ---------------------------------------------------------- engram

    /// Start a workflow for every Engram queue row that is pending or
    /// running and has none (jobs queued before the queue moved onto
    /// workflows). Idempotent.
    fn flow_adopt_engram(&'s self) -> Result<JsonValue> {
        self.flow_ensure()?;
        let has_queue = {
            let tx = self.transact()?;
            tx.get_relation("kleos_engram_queue", false).is_ok()
        };
        if !has_queue {
            return Ok(json!({"adopted": 0}));
        }
        let r = self.fl_r(
            &format!(
                "?[ns, qid] := *kleos_engram_queue{{ns, qid, status}}, (status == 'pending' || status == 'running'), \
                 id = concat('engram:', ns, ':', to_string(qid)), not *{FLOW_WF}{{id}}"
            ),
            vec![],
        )?;
        let mut n = 0;
        for row in r.rows {
            let ns = row[0].get_str().unwrap_or_default().to_string();
            let qid = row[1].get_int().unwrap_or(0);
            let out = self.flow_start(StartOpts {
                workflow: "engram.apply".to_string(),
                input: json!({"ns": ns, "qid": qid}),
                id: Some(format!("engram:{ns}:{qid}")),
                queue: Some("engram".to_string()),
                serial: Some(format!("engram:{ns}")),
                serial_seq: Some(qid),
                mode: Some("code".to_string()),
                ..Default::default()
            })?;
            if out.get("existing").is_none() {
                n += 1;
            }
        }
        Ok(json!({"adopted": n}))
    }

    // ---------------------------------------------------------- dispatch

    /// One entry point for every surface that speaks JSON (HTTP, `::flow`,
    /// bindings). See docs/specs/durable-workflows.md, "Ops".
    pub fn flow_call(&'s self, op: &str, args: JsonValue) -> Result<JsonValue> {
        self.flow_ensure()?;
        Ok(match op {
            "define" => self.flow_define(from_args(op, args)?)?,
            "start" => self.flow_start(from_args(op, args)?)?,
            "signal" => self.flow_signal(from_args(op, args)?)?,
            "cancel" => self.flow_cancel(from_args(op, args)?)?,
            "status" => self.flow_status(&from_args::<IdOpts>(op, args)?.id)?,
            "list" => self.flow_list(from_args(op, args)?)?,
            "history" => serde_json::to_value(self.flow_history(&from_args::<IdOpts>(op, args)?.id)?).into_diagnostic()?,
            "stats" => self.flow_stats()?,
            "schedule" => self.flow_schedule(from_args(op, args)?)?,
            "unschedule" => {
                let o: NameOpts = from_args(op, args)?;
                let had = !self
                    .fl_r(&format!("?[name] := *{FLOW_SCHEDULE}{{name: $n}}, name = $n"), vec![("n", sv(&o.name))])?
                    .rows
                    .is_empty();
                self.fl_w(&format!("?[name] <- [[$n]] :rm {FLOW_SCHEDULE} {{name}}"), vec![("n", sv(&o.name))])?;
                json!({"name": o.name, "removed": had})
            }
            "schedules" => self.flow_schedules()?,
            "adopt_engram" => self.flow_adopt_engram()?,
            "claim" => match self.flow_claim(&from_args(op, args)?)? {
                Some(c) => serde_json::to_value(c).into_diagnostic()?,
                None => JsonValue::Null,
            },
            "step" => {
                let o: StepOpts = from_args(op, args)?;
                let (v, replayed) =
                    self.flow_step(&o.id, o.token, &o.key, o.name.as_deref(), o.action.as_ref(), o.result.as_ref())?;
                json!({"result": v, "replayed": replayed})
            }
            "record" => {
                let o: RecordOpts = from_args(op, args)?;
                for e in &o.events {
                    if !COMMAND_TYPES.contains(&e.ty.as_str()) {
                        bail!("flow record: '{}' is not a command event type", e.ty);
                    }
                }
                let evs = o.events.into_iter().map(|e| (e.ty, e.key, e.name, e.data)).collect();
                json!({"data": self.flow_record(&o.id, o.token, evs)?})
            }
            "consume_signal" => {
                let o: ConsumeOpts = from_args(op, args)?;
                match self.flow_consume_signal(&o.id, o.token, &o.key, &o.name)? {
                    Some(p) => json!({"received": true, "payload": p}),
                    None => json!({"received": false}),
                }
            }
            "start_child" => {
                let o: ChildOpts = from_args(op, args)?;
                let cid = self.flow_start_child(
                    &o.id,
                    o.token,
                    &o.key,
                    &o.workflow,
                    &o.input,
                    o.child_id.as_deref(),
                    o.version,
                    o.queue.as_deref(),
                )?;
                json!({"child": cid})
            }
            "suspend" => self.flow_suspend(&from_args(op, args)?)?,
            "complete" => {
                let o: CompleteOpts = from_args(op, args)?;
                self.flow_finish(&o.id, o.token, "completed", &o.output, None)?
            }
            "fail" => self.flow_fail(from_args(op, args)?)?,
            "continue_as_new" => {
                #[derive(Default, Deserialize)]
                #[serde(default, deny_unknown_fields)]
                struct C {
                    id: String,
                    token: i64,
                    input: JsonValue,
                }
                let o: C = from_args(op, args)?;
                self.flow_continue(&o.id, o.token, &o.input)?
            }
            "heartbeat" => self.flow_heartbeat(from_args(op, args)?)?,
            "tick" => self.flow_tick(from_args(op, args)?)?,
            "next_due" => {
                #[derive(Default, Deserialize)]
                #[serde(default, deny_unknown_fields)]
                struct Q {
                    queues: Option<Vec<String>>,
                }
                let o: Q = from_args(op, args)?;
                let queues = o.queues.unwrap_or_else(|| vec!["default".to_string()]);
                json!({"due": self.flow_next_due(&queues)?, "now": now_us()})
            }
            other => bail!(
                "unknown flow op '{other}': expected define, start, signal, cancel, status, list, history, stats, \
                 schedule, unschedule, schedules, adopt_engram, claim, step, record, consume_signal, start_child, \
                 suspend, complete, fail, continue_as_new, heartbeat, tick or next_due"
            ),
        })
    }

    /// Run due built-in workflows (and fire due schedules): the in-process
    /// executor. Returns `{ran, fired}`.
    pub fn flow_tick(&'s self, o: TickOpts) -> Result<JsonValue> {
        self.flow_ensure()?;
        let fired = if o.schedules.unwrap_or(true) { self.flow_fire_schedules()? } else { vec![] };
        let max = o.max.unwrap_or(64);
        let claim = ClaimOpts {
            worker: o.worker.clone(),
            queues: o.queues.clone(),
            modes: Some(vec!["builtin".to_string()]),
            workflows: None,
            lanes: None,
            lease_ms: o.lease_ms,
        };
        let mut ran = vec![];
        while ran.len() < max {
            let Some(c) = self.flow_claim(&claim)? else { break };
            let id = c.id.clone();
            let status = crate::flow_builtin::run(self, c)?;
            ran.push(json!({"id": id, "status": status}));
        }
        Ok(json!({"ran": ran, "fired": fired}))
    }
}

/// Texts for `embed`: a list of strings, or query rows of one column.
fn flatten_texts(v: &JsonValue) -> Result<Vec<String>> {
    let v = match v.get("rows") {
        Some(r) => r,
        None => v,
    };
    let arr = v.as_array().ok_or_else(|| miette!("flow embed: 'texts' must be a list"))?;
    arr.iter()
        .map(|x| match x {
            JsonValue::String(s) => Ok(s.clone()),
            JsonValue::Array(a) if a.len() == 1 && a[0].is_string() => Ok(a[0].as_str().unwrap().to_string()),
            other => bail!("flow embed: '{other}' is not a text"),
        })
        .collect()
}

#[cfg(feature = "flow-http")]
fn flow_http(args: &JsonValue) -> Result<JsonValue> {
    let url = args.get("url").and_then(|x| x.as_str()).ok_or_else(|| miette!("flow http needs 'url'"))?;
    let method = args.get("method").and_then(|x| x.as_str()).unwrap_or("GET").to_ascii_uppercase();
    let mut req = match method.as_str() {
        "GET" => minreq::get(url),
        "POST" => minreq::post(url),
        "PUT" => minreq::put(url),
        "DELETE" => minreq::delete(url),
        "PATCH" => minreq::patch(url),
        m => bail!("flow http: unsupported method {m}"),
    };
    if let Some(h) = args.get("headers").and_then(|x| x.as_object()) {
        for (k, v) in h {
            req = req.with_header(k, v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()));
        }
    }
    if let Some(b) = args.get("body") {
        req = match b {
            JsonValue::String(s) => req.with_body(s.as_str()),
            other => req.with_header("Content-Type", "application/json").with_body(other.to_string()),
        };
    }
    req = req.with_timeout(args.get("timeout_ms").and_then(|x| x.as_u64()).unwrap_or(30_000).div_ceil(1000));
    let resp = req.send().map_err(|e| miette!("flow http: {e}"))?;
    let body = resp.as_str().unwrap_or_default().to_string();
    let parsed = serde_json::from_str::<JsonValue>(&body).unwrap_or(JsonValue::String(body));
    if resp.status_code >= 400 {
        bail!("flow http: {} {} returned {}", method, url, resp.status_code);
    }
    Ok(json!({"status": resp.status_code, "body": parsed}))
}

#[cfg(not(feature = "flow-http"))]
fn flow_http(_args: &JsonValue) -> Result<JsonValue> {
    bail!("flow http steps need kleos built with the `flow-http` feature")
}

/// `::flow` result rows: a list of objects is one row per object (columns
/// in first-seen order), an object one row, anything else one `result`
/// cell. Nested objects stay JSON.
pub(crate) fn flow_rows(v: JsonValue) -> NamedRows {
    let cell = |x: Option<&JsonValue>| -> DataValue {
        match x {
            None | Some(JsonValue::Null) => DataValue::Null,
            Some(JsonValue::Array(a)) if a.iter().all(|y| !y.is_object()) => DataValue::from(x.unwrap()),
            Some(o @ (JsonValue::Object(_) | JsonValue::Array(_))) => DataValue::Json(JsonData(o.clone())),
            Some(y) => DataValue::from(y),
        }
    };
    match v {
        JsonValue::Array(items) if items.iter().all(|x| x.is_object()) => {
            let mut cols: Vec<String> = vec![];
            for it in &items {
                for k in it.as_object().unwrap().keys() {
                    if !cols.contains(k) {
                        cols.push(k.clone());
                    }
                }
            }
            let rows = items
                .iter()
                .map(|it| cols.iter().map(|c| cell(it.get(c))).collect())
                .collect();
            NamedRows::new(cols, rows)
        }
        JsonValue::Object(o) => {
            let cols: Vec<String> = o.keys().cloned().collect();
            let row = cols.iter().map(|c| cell(o.get(c))).collect();
            NamedRows::new(cols, vec![row])
        }
        JsonValue::Null => NamedRows::new(vec!["result".to_string()], vec![]),
        other => NamedRows::new(vec!["result".to_string()], vec![vec![cell(Some(&other))]]),
    }
}
