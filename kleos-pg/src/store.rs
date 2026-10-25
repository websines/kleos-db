/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The connector's state, kept inside kleos (docs/specs/pg-connector.md,
//! "State"). The checkpoint commits in the same kleos transaction as the
//! facts it covers, which is what makes resume exactly once.

use std::collections::BTreeMap;

use kleos::{DataValue, DbInstance, JsonData, NamedRows, ScriptMutability};
use miette::{miette, Result};
use serde_json::{json, Value};

use crate::mapping::Mapping;

pub const SOURCE: &str = "kleos_pg_source";
pub const CHECKPOINT: &str = "kleos_pg_checkpoint";
pub const TABLE: &str = "kleos_pg_table";
pub const MAPPING: &str = "kleos_pg_mapping";
pub const ROW: &str = "kleos_pg_row";
pub const ECHO: &str = "kleos_pg_echo";
pub const EVENT: &str = "kleos_pg_event";
pub const OFFSET: &str = "kleos_pg_offset";

const DDL: &[(&str, &str)] = &[
    (SOURCE, "{src: String => conf: Json, actor: Json, created: Int, updated: Int}"),
    (
        CHECKPOINT,
        "{src: String => lsn: Int, end_lsn: Int, commit_ts: Int, vt: Int, applied: Int, consistent: Int, slot: String, updated: Int}",
    ),
    (
        TABLE,
        "{src: String, tbl: String => state: String, skip_before: Int, rows: Int, info: Json?, error: String?, updated: Int}",
    ),
    (MAPPING, "{src: String, version: Int => mapping: Json, at: Int, reason: String, status: String}"),
    (ROW, "{src: String, tbl: String, key: String => vals: Json, mems: Json, lsn: Int, at: Int}"),
    (ECHO, "{src: String, lsn: Int => idem: String, wf: String?, step: String?, actor: String?, rows: Int, at: Int}"),
    (EVENT, "{src: String, at: Int, seq: Int => level: String, msg: String}"),
    (OFFSET, "{src: String, part: String => pos: Json, at: Int}"),
];

pub fn s(x: &str) -> DataValue {
    DataValue::from(x)
}

pub fn j(v: Value) -> DataValue {
    DataValue::Json(JsonData(v))
}

pub fn i(n: i64) -> DataValue {
    DataValue::from(n)
}

pub fn params(p: Vec<(&str, DataValue)>) -> BTreeMap<String, DataValue> {
    p.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

pub fn q(db: &DbInstance, script: &str, p: Vec<(&str, DataValue)>) -> Result<NamedRows> {
    db.run_script(script, params(p), ScriptMutability::Mutable)
}

pub fn read(db: &DbInstance, script: &str, p: Vec<(&str, DataValue)>) -> Result<NamedRows> {
    db.run_script(script, params(p), ScriptMutability::Immutable)
}

pub fn json_of(v: &DataValue) -> Value {
    match v {
        DataValue::Json(j) => j.0.clone(),
        DataValue::Null => Value::Null,
        other => serde_json::to_value(other).unwrap_or(Value::Null),
    }
}

pub fn str_of(v: &DataValue) -> Option<String> {
    v.get_str().map(|x| x.to_string())
}

/// Relations that exist.
pub fn relations(db: &DbInstance) -> Result<Vec<String>> {
    Ok(read(db, "::relations", vec![])?.rows.iter().filter_map(|r| str_of(&r[0])).collect())
}

/// Create the connector's relations, and the memory and ontology schema.
pub fn ensure(db: &DbInstance) -> Result<()> {
    let have = relations(db)?;
    for (name, cols) in DDL {
        if !have.iter().any(|h| h == name) {
            if let Err(e) = q(db, &format!(":create {name} {cols}"), vec![]) {
                if !relations(db)?.iter().any(|h| h == name) {
                    return Err(e);
                }
            }
        }
    }
    db.memory_init(kleos::memory::InitOpts::default())?;
    // the ontology relations (kleos_onto_*): `proposals` creates them
    db.ontology_call("proposals", json!({"ns": "default"}))?;
    Ok(())
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Checkpoint {
    /// Commit LSN of the last applied transaction.
    pub lsn: u64,
    /// Its end LSN (what was acknowledged).
    pub end_lsn: u64,
    pub commit_ts: i64,
    /// The last valid time written (monotonic per source).
    pub vt: i64,
    pub applied: i64,
    /// The slot's consistent point when kleos created it.
    pub consistent: u64,
    pub slot: String,
    pub updated: i64,
}

pub fn checkpoint(db: &DbInstance, src: &str) -> Result<Option<Checkpoint>> {
    let r = read(
        db,
        &format!("?[lsn, end_lsn, commit_ts, vt, applied, consistent, slot, updated] := *{CHECKPOINT}{{src: $src, lsn, end_lsn, commit_ts, vt, applied, consistent, slot, updated}}"),
        vec![("src", s(src))],
    )?;
    Ok(r.rows.first().map(|x| Checkpoint {
        lsn: x[0].get_int().unwrap_or(0) as u64,
        end_lsn: x[1].get_int().unwrap_or(0) as u64,
        commit_ts: x[2].get_int().unwrap_or(0),
        vt: x[3].get_int().unwrap_or(0),
        applied: x[4].get_int().unwrap_or(0),
        consistent: x[5].get_int().unwrap_or(0) as u64,
        slot: str_of(&x[6]).unwrap_or_default(),
        updated: x[7].get_int().unwrap_or(0),
    }))
}

pub fn put_checkpoint(db: &DbInstance, src: &str, c: &Checkpoint) -> Result<()> {
    q(
        db,
        &format!(
            "?[src, lsn, end_lsn, commit_ts, vt, applied, consistent, slot, updated] <- [[$src, $lsn, $end_lsn, $commit_ts, $vt, $applied, $consistent, $slot, $updated]] \
             :put {CHECKPOINT} {{src => lsn, end_lsn, commit_ts, vt, applied, consistent, slot, updated}}"
        ),
        vec![
            ("src", s(src)),
            ("lsn", i(c.lsn as i64)),
            ("end_lsn", i(c.end_lsn as i64)),
            ("commit_ts", i(c.commit_ts)),
            ("vt", i(c.vt)),
            ("applied", i(c.applied)),
            ("consistent", i(c.consistent as i64)),
            ("slot", s(&c.slot)),
            ("updated", i(now_us())),
        ],
    )?;
    Ok(())
}

pub fn delete_checkpoint(db: &DbInstance, src: &str) -> Result<()> {
    q(db, &format!("?[src] <- [[$src]] :rm {CHECKPOINT} {{src}}"), vec![("src", s(src))])?;
    Ok(())
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TableState {
    pub tbl: String,
    /// `copying`, `ready`, `resync` (requested), `skipped`.
    pub state: String,
    /// Stream changes of this table committed before this LSN are already
    /// in its snapshot.
    pub skip_before: u64,
    pub rows: i64,
    pub info: Option<Value>,
    pub error: Option<String>,
    pub updated: i64,
}

pub fn table_states(db: &DbInstance, src: &str) -> Result<BTreeMap<String, TableState>> {
    let r = read(
        db,
        &format!("?[tbl, state, skip_before, rows, info, error, updated] := *{TABLE}{{src: $src, tbl, state, skip_before, rows, info, error, updated}}"),
        vec![("src", s(src))],
    )?;
    Ok(r.rows
        .iter()
        .map(|x| {
            let t = TableState {
                tbl: str_of(&x[0]).unwrap_or_default(),
                state: str_of(&x[1]).unwrap_or_default(),
                skip_before: x[2].get_int().unwrap_or(0) as u64,
                rows: x[3].get_int().unwrap_or(0),
                info: match &x[4] {
                    DataValue::Null => None,
                    v => Some(json_of(v)),
                },
                error: str_of(&x[5]),
                updated: x[6].get_int().unwrap_or(0),
            };
            (t.tbl.clone(), t)
        })
        .collect())
}

pub fn put_table_state(db: &DbInstance, src: &str, t: &TableState) -> Result<()> {
    q(
        db,
        &format!(
            "?[src, tbl, state, skip_before, rows, info, error, updated] <- [[$src, $tbl, $state, $skip, $rows, $info, $error, $updated]] \
             :put {TABLE} {{src, tbl => state, skip_before, rows, info, error, updated}}"
        ),
        vec![
            ("src", s(src)),
            ("tbl", s(&t.tbl)),
            ("state", s(&t.state)),
            ("skip", i(t.skip_before as i64)),
            ("rows", i(t.rows)),
            ("info", t.info.clone().map(j).unwrap_or(DataValue::Null)),
            ("error", t.error.as_deref().map(s).unwrap_or(DataValue::Null)),
            ("updated", i(now_us())),
        ],
    )?;
    Ok(())
}

/// The latest mapping version: (version, mapping, status).
pub fn latest_mapping(db: &DbInstance, src: &str) -> Result<Option<(i64, Mapping, String)>> {
    let r = read(
        db,
        &format!("?[version, mapping, status] := *{MAPPING}{{src: $src, version, mapping, status}} :order -version :limit 1"),
        vec![("src", s(src))],
    )?;
    match r.rows.first() {
        None => Ok(None),
        Some(x) => {
            let m: Mapping = serde_json::from_value(json_of(&x[1])).map_err(|e| miette!("stored pg mapping: {e}"))?;
            Ok(Some((x[0].get_int().unwrap_or(0), m, str_of(&x[2]).unwrap_or_default())))
        }
    }
}

pub fn save_mapping(db: &DbInstance, src: &str, version: i64, m: &Mapping, reason: &str, status: &str) -> Result<()> {
    q(
        db,
        &format!(
            "?[src, version, mapping, at, reason, status] <- [[$src, $v, $m, $at, $reason, $status]] \
             :put {MAPPING} {{src, version => mapping, at, reason, status}}"
        ),
        vec![
            ("src", s(src)),
            ("v", i(version)),
            ("m", j(serde_json::to_value(m).unwrap_or(Value::Null))),
            ("at", i(now_us())),
            ("reason", s(reason)),
            ("status", s(status)),
        ],
    )?;
    Ok(())
}

/// An entry in the connector's event log (errors, DDL, resyncs, alerts).
pub fn log_event(db: &DbInstance, src: &str, level: &str, msg: &str) {
    static SEQ: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    match level {
        "error" => log::error!("pg {src}: {msg}"),
        "warn" => log::warn!("pg {src}: {msg}"),
        _ => log::info!("pg {src}: {msg}"),
    }
    let _ = q(
        db,
        &format!("?[src, at, seq, level, msg] <- [[$src, $at, $seq, $level, $msg]] :put {EVENT} {{src, at, seq => level, msg}}"),
        vec![("src", s(src)), ("at", i(now_us())), ("seq", i(seq)), ("level", s(level)), ("msg", s(msg))],
    );
}

pub fn events(db: &DbInstance, src: &str, limit: usize) -> Result<Vec<Value>> {
    let r = read(
        db,
        &format!("?[at, level, msg] := *{EVENT}{{src: $src, at, level, msg}} :order -at :limit {limit}"),
        vec![("src", s(src))],
    )?;
    Ok(r.rows.iter().map(|x| json!({"at": x[0].get_int(), "level": str_of(&x[1]), "msg": str_of(&x[2])})).collect())
}

pub fn put_source(db: &DbInstance, cfg: &crate::config::PgConfig) -> Result<()> {
    let created = read(db, &format!("?[c] := *{SOURCE}{{src: $src, created: c}}"), vec![("src", s(&cfg.source))])?
        .rows
        .first()
        .and_then(|r| r[0].get_int())
        .unwrap_or_else(now_us);
    q(
        db,
        &format!("?[src, conf, actor, created, updated] <- [[$src, $conf, $actor, $created, $updated]] :put {SOURCE} {{src => conf, actor, created, updated}}"),
        vec![
            ("src", s(&cfg.source)),
            ("conf", j(cfg.redacted())),
            (
                "actor",
                j(json!({"service": cfg.service(), "human": cfg.actor.human, "agent": cfg.actor.agent, "workspace": cfg.workspace, "scope": cfg.ns})),
            ),
            ("created", i(created)),
            ("updated", i(now_us())),
        ],
    )?;
    Ok(())
}

pub fn source_conf(db: &DbInstance, src: &str) -> Result<Option<Value>> {
    Ok(read(db, &format!("?[c] := *{SOURCE}{{src: $src, conf: c}}"), vec![("src", s(src))])?.rows.first().map(|r| json_of(&r[0])))
}

pub fn sources(db: &DbInstance) -> Result<Vec<String>> {
    if !relations(db)?.iter().any(|r| r == SOURCE) {
        return Ok(vec![]);
    }
    Ok(read(db, &format!("?[src] := *{SOURCE}{{src}}"), vec![])?.rows.iter().filter_map(|r| str_of(&r[0])).collect())
}

pub fn now_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}
