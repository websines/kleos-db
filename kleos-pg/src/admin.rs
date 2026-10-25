/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Server checks, publications, slots and the write-back ledger.

use miette::{bail, Result};
use serde::Serialize;
use tokio_postgres::Client;

use crate::config::PgConfig;
use crate::conn::{ident, lit, qident, simple_rows};

#[derive(Debug, Clone, Serialize)]
pub struct ServerInfo {
    pub version: i32,
    pub wal_level: String,
    /// `-1` is unbounded.
    pub max_slot_wal_keep_size: String,
    pub replication_role: bool,
    pub superuser: bool,
    pub in_recovery: bool,
    /// PG18+.
    pub idle_replication_slot_timeout: Option<String>,
}

pub async fn server_info(pg: &Client) -> Result<ServerInfo> {
    let version = crate::conn::server_version(pg).await?;
    let idle = if version >= 180000 { ", current_setting('idle_replication_slot_timeout', true)" } else { ", NULL" };
    let r = simple_rows(
        pg,
        &format!(
            "SELECT current_setting('wal_level'), current_setting('max_slot_wal_keep_size'), \
                    r.rolreplication OR r.rolsuper, r.rolsuper, pg_is_in_recovery(){idle} \
             FROM pg_roles r WHERE r.rolname = current_user"
        ),
    )
    .await?;
    let row = r.first().cloned().unwrap_or_default();
    let g = |i: usize| row.get(i).cloned().flatten().unwrap_or_default();
    Ok(ServerInfo {
        version,
        wal_level: g(0),
        max_slot_wal_keep_size: g(1),
        replication_role: g(2) == "t",
        superuser: g(3) == "t",
        in_recovery: g(4) == "t",
        idle_replication_slot_timeout: row.get(5).cloned().flatten(),
    })
}

/// Problems that stop replication, with what to do about them.
pub fn blockers(info: &ServerInfo) -> Vec<String> {
    let mut out = vec![];
    if info.wal_level != "logical" {
        out.push(format!(
            "wal_level is '{}': set wal_level = logical and restart (RDS/Aurora: rds.logical_replication = 1; Cloud SQL: cloudsql.logical_decoding = on; Azure: wal_level = logical)",
            info.wal_level
        ));
    }
    if !info.replication_role {
        out.push("the connecting role lacks REPLICATION: ALTER ROLE <role> WITH REPLICATION (RDS/Aurora: GRANT rds_replication TO <role>)".into());
    }
    if info.version < 140000 {
        out.push(format!("Postgres {} is not supported (14+; 15–18 officially)", info.version / 10000));
    }
    out
}

/// Advice that does not block (WAL retention, versions).
pub fn warnings(info: &ServerInfo) -> Vec<String> {
    let mut out = vec![];
    if info.max_slot_wal_keep_size == "-1" {
        out.push("max_slot_wal_keep_size is unbounded: a stalled connector's slot can fill the source's disk. Set it (e.g. 50GB); an invalidated slot is detected and resynced".into());
    }
    if info.version < 150000 {
        out.push("Postgres 14 is end of life on 2026-11-12; supported best effort".into());
    }
    if info.in_recovery && info.version < 160000 {
        out.push("logical decoding on a standby needs Postgres 16+".into());
    }
    out
}

/// Schemas named by `include` globs (for `FOR TABLES IN SCHEMA`).
fn include_schemas(cfg: &PgConfig) -> Result<Vec<String>> {
    let mut out = vec![];
    for g in &cfg.include {
        let schema = g.split('.').next().unwrap_or("");
        if schema.contains('*') || schema.is_empty() {
            bail!("publication_mode: schema needs include globs with literal schemas (public.*), got '{g}'");
        }
        if !out.contains(&schema.to_string()) {
            out.push(schema.to_string());
        }
    }
    Ok(out)
}

/// The `CREATE PUBLICATION` statement for `tables`.
pub fn publication_ddl(cfg: &PgConfig, tables: &[String], version: i32) -> Result<String> {
    let name = ident(&cfg.publication());
    let with = " WITH (publish = 'insert, update, delete, truncate', publish_via_partition_root = true)";
    Ok(match cfg.publication_mode.as_str() {
        "all" => format!("CREATE PUBLICATION {name} FOR ALL TABLES{with}"),
        "schema" => {
            if version < 150000 {
                bail!("publication_mode: schema needs Postgres 15+");
            }
            let schemas = include_schemas(cfg)?.iter().map(|s| ident(s)).collect::<Vec<_>>().join(", ");
            format!("CREATE PUBLICATION {name} FOR TABLES IN SCHEMA {schemas}{with}")
        }
        _ => {
            if tables.is_empty() {
                bail!("no tables match include {:?}", cfg.include);
            }
            format!("CREATE PUBLICATION {name} FOR TABLE {}{with}", tables.iter().map(|t| qident(t)).collect::<Vec<_>>().join(", "))
        }
    })
}

/// Tables of a publication (`schema.table`); `None` when it does not exist.
pub async fn publication_tables(pg: &Client, publication: &str) -> Result<Option<Vec<String>>> {
    let exists = simple_rows(pg, &format!("SELECT 1 FROM pg_publication WHERE pubname = {}", lit(publication))).await?;
    if exists.is_empty() {
        return Ok(None);
    }
    let rows = simple_rows(pg, &format!("SELECT schemaname || '.' || tablename FROM pg_publication_tables WHERE pubname = {}", lit(publication))).await?;
    Ok(Some(rows.into_iter().filter_map(|r| r[0].clone()).collect()))
}

/// Create the publication, or add missing tables to it (`tables` mode).
/// Returns notes for the operator.
pub async fn ensure_publication(pg: &Client, cfg: &PgConfig, tables: &[String], version: i32) -> Result<Vec<String>> {
    let mut notes = vec![];
    let pubname = cfg.publication();
    match publication_tables(pg, &pubname).await? {
        None => {
            let ddl = publication_ddl(cfg, tables, version)?;
            if !cfg.create_publication {
                bail!("publication {pubname} does not exist and create_publication is false; ask a DBA to run:\n  {ddl};");
            }
            pg.batch_execute(&ddl).await.map_err(|e| {
                miette::miette!("creating publication {pubname} failed ({}); ask a DBA to run:\n  {ddl};", crate::conn::pg_err(e))
            })?;
            notes.push(format!("created publication {pubname}"));
        }
        Some(have) => {
            if cfg.publication_mode == "tables" {
                let missing: Vec<&String> = tables.iter().filter(|t| !have.contains(t)).collect();
                if !missing.is_empty() {
                    let ddl = format!(
                        "ALTER PUBLICATION {} ADD TABLE {}",
                        ident(&pubname),
                        missing.iter().map(|t| qident(t)).collect::<Vec<_>>().join(", ")
                    );
                    if cfg.create_publication {
                        pg.batch_execute(&ddl).await.map_err(|e| {
                            miette::miette!("adding tables to {pubname} failed ({}); ask a DBA to run:\n  {ddl};", crate::conn::pg_err(e))
                        })?;
                        notes.push(format!("added {} table(s) to publication {pubname}", missing.len()));
                    } else {
                        notes.push(format!("tables missing from publication {pubname}; ask a DBA to run: {ddl};"));
                    }
                }
            }
        }
    }
    Ok(notes)
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SlotInfo {
    pub active: bool,
    pub wal_status: Option<String>,
    pub restart_lsn: Option<String>,
    pub confirmed_flush_lsn: Option<String>,
    /// WAL between the server's current position and `confirmed_flush_lsn`.
    pub lag_bytes: Option<i64>,
    /// WAL the slot holds back (from `restart_lsn`).
    pub retained_bytes: Option<i64>,
    pub safe_wal_size: Option<i64>,
    pub invalidation_reason: Option<String>,
    pub failover: Option<bool>,
    pub inactive_since: Option<String>,
    pub current_lsn: Option<String>,
}

impl SlotInfo {
    /// A slot Postgres invalidated (WAL removed, idle timeout, conflict):
    /// it cannot stream any more.
    pub fn invalid(&self) -> bool {
        self.wal_status.as_deref() == Some("lost") || self.invalidation_reason.is_some()
    }
}

pub async fn slot_info(pg: &Client, slot: &str, version: i32) -> Result<Option<SlotInfo>> {
    let extra = if version >= 170000 {
        "invalidation_reason, failover::text, inactive_since::text"
    } else if version >= 160000 {
        "CASE WHEN conflicting THEN 'conflict' END, NULL, NULL"
    } else {
        "NULL, NULL, NULL"
    };
    let cur = if version >= 100000 { "CASE WHEN pg_is_in_recovery() THEN pg_last_wal_replay_lsn() ELSE pg_current_wal_lsn() END" } else { "NULL" };
    let r = simple_rows(
        pg,
        &format!(
            "SELECT active, wal_status, restart_lsn::text, confirmed_flush_lsn::text, \
                    pg_wal_lsn_diff({cur}, confirmed_flush_lsn)::int8, pg_wal_lsn_diff({cur}, restart_lsn)::int8, \
                    safe_wal_size, {extra}, ({cur})::text \
             FROM pg_replication_slots WHERE slot_name = {}",
            lit(slot)
        ),
    )
    .await?;
    Ok(r.first().map(|x| {
        let g = |i: usize| x.get(i).cloned().flatten();
        SlotInfo {
            active: g(0).as_deref() == Some("t"),
            wal_status: g(1),
            restart_lsn: g(2),
            confirmed_flush_lsn: g(3),
            lag_bytes: g(4).and_then(|v| v.parse().ok()),
            retained_bytes: g(5).and_then(|v| v.parse().ok()),
            safe_wal_size: g(6).and_then(|v| v.parse().ok()),
            invalidation_reason: g(7),
            failover: g(8).map(|v| v == "true"),
            inactive_since: g(9),
            current_lsn: g(10),
        }
    }))
}

/// Drop a slot. Refused while a connector is streaming from it.
pub async fn drop_slot(pg: &Client, slot: &str) -> Result<bool> {
    let r = simple_rows(pg, &format!("SELECT active, active_pid FROM pg_replication_slots WHERE slot_name = {}", lit(slot))).await?;
    let Some(row) = r.first() else { return Ok(false) };
    if row[0].as_deref() == Some("t") {
        bail!("slot {slot} is in use (pid {}); stop the connector first", row[1].clone().unwrap_or_default());
    }
    simple_rows(pg, &format!("SELECT pg_drop_replication_slot({})", lit(slot))).await?;
    Ok(true)
}

pub fn ledger_ddl(cfg: &PgConfig) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS {} (idem_key text PRIMARY KEY, workflow text, step text, actor text, \
         result jsonb, committed_at timestamptz NOT NULL DEFAULT now())",
        qident(&cfg.writeback.ledger)
    )
}

/// Create the write-back ledger if it is missing (`Ok(false)`: it existed).
pub async fn ensure_ledger(pg: &Client, cfg: &PgConfig) -> Result<bool> {
    if crate::catalog::table_exists(pg, &cfg.writeback.ledger).await? {
        return Ok(false);
    }
    let ddl = ledger_ddl(cfg);
    pg.batch_execute(&ddl)
        .await
        .map_err(|e| miette::miette!("creating the write-back ledger failed ({}); ask a DBA to run:\n  {ddl};", crate::conn::pg_err(e)))?;
    Ok(true)
}
