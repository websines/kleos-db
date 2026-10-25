/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! `kleos pg …` and the Postgres MCP tools (kleos fork;
//! docs/specs/pg-connector.md). The connector itself is kleos-pg (feature
//! `pg-cdc`); without the feature the commands say so.

use clap::Subcommand;
use serde_json::{json, Value};

use crate::backend::{Area, Backend};
use crate::cli::ConnArgs;

/// Postgres coexistence commands.
#[derive(Subcommand, Debug)]
pub enum PgCmd {
    /// Check the server, create the publication (and write-back ledger), infer the mapping
    Init {
        #[command(flatten)]
        src: SourceArgs,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// Replication status: lag, slot, tables, alerts
    Status {
        #[command(flatten)]
        src: SourceArgs,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// Copy tables again (all mapped tables without --table)
    Resync {
        /// Table to resync, schema-qualified (repeatable)
        #[arg(long = "table")]
        tables: Vec<String>,
        #[command(flatten)]
        src: SourceArgs,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// Drop the replication slot (stop the connector first; the next start resyncs)
    DropSlot {
        #[arg(long)]
        yes: bool,
        #[command(flatten)]
        src: SourceArgs,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// The mapping and the ontology it defines
    Map {
        /// Re-infer from the live catalog and show the difference (dry run)
        #[arg(long)]
        infer: bool,
        /// Review mode: apply a proposed mapping version to the ontology
        #[arg(long)]
        approve: Option<i64>,
        #[command(flatten)]
        src: SourceArgs,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// Run the connector in the foreground (a standalone worker over an embedded store)
    Run {
        #[command(flatten)]
        src: SourceArgs,
        #[command(flatten)]
        conn: ConnArgs,
    },
    /// A live SQL query over Postgres tables (pg.<schema>.<table>) joined with kleos (kleos.facts, …)
    Query {
        sql: String,
        #[command(flatten)]
        src: SourceArgs,
        #[command(flatten)]
        conn: ConnArgs,
    },
}

#[derive(clap::Args, Clone, Debug, Default)]
pub struct SourceArgs {
    /// The connector's YAML config (embedded mode; a server uses its --pg-cdc configs)
    #[arg(long = "pg-config")]
    pub pg_config: Option<String>,
    /// Source name (default: the config's, or the only one)
    #[arg(long)]
    pub source: Option<String>,
}

impl PgCmd {
    pub fn conn(&self) -> &ConnArgs {
        match self {
            PgCmd::Init { conn, .. }
            | PgCmd::Status { conn, .. }
            | PgCmd::Resync { conn, .. }
            | PgCmd::DropSlot { conn, .. }
            | PgCmd::Map { conn, .. }
            | PgCmd::Run { conn, .. }
            | PgCmd::Query { conn, .. } => conn,
        }
    }
}

/// Load `--pg-config` (embedded mode) and return the source name.
fn source(src: &SourceArgs, backend: &Backend) -> Result<Option<String>, String> {
    let mut name = src.source.clone();
    if let Some(path) = &src.pg_config {
        if backend.is_remote() {
            return Err("--pg-config is for an embedded store; a server reads its own --pg-cdc configs".into());
        }
        #[cfg(feature = "pg-cdc")]
        {
            let cfg = kleos_pg::PgConfig::load(path).map_err(|e| e.to_string())?;
            name.get_or_insert(cfg.source.clone());
            kleos_pg::register_config(cfg);
        }
        #[cfg(not(feature = "pg-cdc"))]
        {
            let _ = path;
            return Err("the Postgres connector needs kleos built with the `pg-cdc` feature".into());
        }
    }
    Ok(name)
}

fn with_source(a: Value, name: &Option<String>) -> Value {
    let mut a = a;
    if let Some(n) = name {
        a["source"] = json!(n);
    }
    a
}

/// Runs `kleos pg …`; returns the result to print.
pub fn run(cmd: PgCmd, backend: &Backend) -> Result<Value, String> {
    match cmd {
        PgCmd::Init { src, .. } => {
            let n = source(&src, backend)?;
            backend.call(Area::Pg, "init", with_source(json!({}), &n))
        }
        PgCmd::Status { src, .. } => {
            let n = source(&src, backend)?;
            backend.call(Area::Pg, "status", with_source(json!({}), &n))
        }
        PgCmd::Resync { tables, src, .. } => {
            let n = source(&src, backend)?;
            let mut a = with_source(json!({}), &n);
            if !tables.is_empty() {
                a["tables"] = json!(tables);
            }
            backend.call(Area::Pg, "resync", a)
        }
        PgCmd::DropSlot { yes, src, .. } => {
            let n = source(&src, backend)?;
            if !yes {
                return Err("dropping the slot discards its position (the next start resyncs every table); pass --yes".into());
            }
            backend.call(Area::Pg, "drop_slot", with_source(json!({}), &n))
        }
        PgCmd::Map { infer, approve, src, .. } => {
            let n = source(&src, backend)?;
            match approve {
                Some(v) => backend.call(Area::Pg, "approve", with_source(json!({"version": v}), &n)),
                None => backend.call(Area::Pg, "map", with_source(json!({"infer": infer}), &n)),
            }
        }
        PgCmd::Query { sql, src, .. } => {
            let n = source(&src, backend)?;
            backend.call(Area::Pg, "query", with_source(json!({"sql": sql}), &n))
        }
        PgCmd::Run { src, .. } => run_worker(&src, backend),
    }
}

#[cfg(feature = "pg-cdc")]
fn run_worker(src: &SourceArgs, backend: &Backend) -> Result<Value, String> {
    let Backend::Embedded(db) = backend else {
        return Err("kleos pg run needs an embedded store (--db); a server runs connectors with --pg-cdc".into());
    };
    let path = src.pg_config.as_deref().ok_or("kleos pg run needs --pg-config")?;
    let cfg = kleos_pg::PgConfig::load(path).map_err(|e| e.to_string())?;
    let name = cfg.source.clone();
    #[cfg(feature = "flow-pg")]
    kleos_pg::writeback::register_flow_action().map_err(|e| e.to_string())?;
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().map_err(|e| e.to_string())?;
    let db = db.clone();
    rt.block_on(async move {
        let shared = kleos_pg::spawn(cfg, db);
        let s2 = shared.clone();
        let _ = ctrlc::set_handler(move || s2.stop());
        eprintln!("kleos pg: connector '{name}' running; Ctrl-C stops it");
        while shared.live.lock().unwrap().state != "stopped" {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
    });
    Ok(json!({"stopped": true}))
}

#[cfg(not(feature = "pg-cdc"))]
fn run_worker(_src: &SourceArgs, _backend: &Backend) -> Result<Value, String> {
    Err("the Postgres connector needs kleos built with the `pg-cdc` feature".into())
}

/// MCP tool definitions (feature `pg-cdc`): `pg_status`, and `pg_query`
/// when queries are allowed.
pub fn mcp_tools(allow_query: bool) -> Vec<Value> {
    if !cfg!(feature = "pg-cdc") {
        return vec![];
    }
    let mut out = vec![json!({
        "name": "pg_status",
        "title": "Postgres connector status",
        "description": "Replication status of the Postgres sources kleos mirrors: lag (bytes, seconds), slot state, last applied LSN, per-table sync state, alerts.",
        "inputSchema": {"type": "object", "properties": {"source": {"type": "string", "description": "Source name; all when omitted"}}},
        "annotations": {"readOnlyHint": true, "openWorldHint": false}
    })];
    if allow_query {
        out.push(json!({
            "name": "pg_query",
            "title": "Live Postgres query",
            "description": "Read-only SQL over live Postgres tables (pg.<schema>.<table>) joined with kleos facts (kleos.facts: s, p, o, value, datatype, by, valid_from; kleos.nodes; kleos.memories). Filters and limits are pushed down to Postgres.",
            "inputSchema": {"type": "object", "properties": {"sql": {"type": "string"}, "source": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["sql"]},
            "annotations": {"readOnlyHint": true, "openWorldHint": true}
        }));
    }
    out
}

pub fn is_mcp_tool(name: &str, allow_query: bool) -> bool {
    cfg!(feature = "pg-cdc") && (name == "pg_status" || (name == "pg_query" && allow_query))
}

/// Run a Postgres MCP tool.
pub fn mcp_call(backend: &Backend, name: &str, args: &Value) -> Result<Value, String> {
    let mut a = json!({});
    for k in ["source", "sql", "limit"] {
        if let Some(v) = args.get(k).filter(|v| !v.is_null()) {
            a[k] = v.clone();
        }
    }
    match name {
        "pg_status" => backend.call(Area::Pg, "status", a),
        "pg_query" => backend.call(Area::Pg, "query", a),
        other => Err(format!("unknown tool {other}")),
    }
}
