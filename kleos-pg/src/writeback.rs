/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Write-back (docs/specs/pg-connector.md, "Write-back"): an agent's write
//! to Postgres as a durable `::flow` step (`{"pg": {sql, params}}`, feature
//! `flow-pg`). The business statements and a row in the idempotency ledger
//! commit in one Postgres transaction (the DBOS pattern); a retried or
//! replayed step finds its ledger row and returns the stored result instead
//! of writing again, so the write happens once in effect.
//!
//! Guardrails: one statement kind per entry (INSERT, UPDATE, DELETE, MERGE,
//! WITH), parameters only, a statement timeout, and an allow-list checked
//! after the fact against the tables the transaction actually modified
//! (`pg_stat_xact_user_tables`), so a CTE cannot write elsewhere.

use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::sync::{Arc, Mutex, OnceLock};

use bytes::BytesMut;
use deadpool::managed::{Manager, Metrics, Pool, RecycleError, RecycleResult};
use miette::{bail, miette, Result};
use serde_json::{json, Value};
use tokio_postgres::types::{FromSql, Format, IsNull, ToSql, Type};
use tokio_postgres::{Client, Row};

use crate::config::PgConfig;
use crate::conn::{self, qident};

struct Mgr {
    cfg: tokio_postgres::Config,
}

impl Manager for Mgr {
    type Type = Client;
    type Error = String;

    async fn create(&self) -> std::result::Result<Client, String> {
        conn::connect(&self.cfg).await.map_err(|e| e.to_string())
    }

    async fn recycle(&self, obj: &mut Client, _m: &Metrics) -> RecycleResult<String> {
        if obj.is_closed() {
            return Err(RecycleError::message("connection closed"));
        }
        obj.simple_query("").await.map(|_| ()).map_err(|e| RecycleError::Backend(e.to_string()))
    }
}

fn pools() -> &'static Mutex<BTreeMap<String, Pool<Mgr>>> {
    static P: OnceLock<Mutex<BTreeMap<String, Pool<Mgr>>>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn pool(cfg: &PgConfig) -> Result<Pool<Mgr>> {
    let mut p = pools().lock().unwrap();
    // keyed by the connection too: a reconfigured source gets a new pool
    let key = format!("{}|{}", cfg.source, cfg.connection);
    if let Some(x) = p.get(&key) {
        return Ok(x.clone());
    }
    let mut pc = conn::parse(&cfg.connection)?;
    pc.application_name("kleos-pg-writeback");
    let pool = Pool::builder(Mgr { cfg: pc })
        .max_size(cfg.writeback.pool_size.max(1))
        .runtime(deadpool::Runtime::Tokio1)
        .build()
        .map_err(|e| miette!("write-back pool: {e}"))?;
    p.insert(key, pool.clone());
    Ok(pool)
}

/// A parameter sent in text format: Postgres parses it with the input
/// function of whatever type the statement expects.
#[derive(Debug)]
struct TextParam(Option<String>);

impl ToSql for TextParam {
    fn to_sql(&self, _ty: &Type, out: &mut BytesMut) -> std::result::Result<IsNull, Box<dyn StdError + Sync + Send>> {
        match &self.0 {
            None => Ok(IsNull::Yes),
            Some(s) => {
                out.extend_from_slice(s.as_bytes());
                Ok(IsNull::No)
            }
        }
    }
    fn accepts(_ty: &Type) -> bool {
        true
    }
    fn encode_format(&self, _ty: &Type) -> Format {
        Format::Text
    }
    tokio_postgres::types::to_sql_checked!();
}

fn text_param(v: &Value) -> TextParam {
    TextParam(match v {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        Value::Bool(b) => Some(if *b { "true".into() } else { "false".into() }),
        Value::Array(a) if a.iter().all(|x| !x.is_object() && !x.is_array()) => Some(format!(
            "{{{}}}",
            a.iter()
                .map(|x| match x {
                    Value::Null => "NULL".to_string(),
                    Value::String(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
                    o => o.to_string(),
                })
                .collect::<Vec<_>>()
                .join(",")
        )),
        other => Some(other.to_string()),
    })
}

/// Any column value, decoded from Postgres's binary format into JSON.
struct AnyValue(Value);

impl<'a> FromSql<'a> for AnyValue {
    fn from_sql(ty: &Type, raw: &'a [u8]) -> std::result::Result<Self, Box<dyn StdError + Sync + Send>> {
        let v = match *ty {
            Type::BOOL => json!(bool::from_sql(ty, raw)?),
            Type::INT2 => json!(i16::from_sql(ty, raw)?),
            Type::INT4 => json!(i32::from_sql(ty, raw)?),
            Type::INT8 => json!(i64::from_sql(ty, raw)?),
            Type::OID => json!(u32::from_sql(ty, raw)?),
            Type::FLOAT4 => json!(f32::from_sql(ty, raw)?),
            Type::FLOAT8 => json!(f64::from_sql(ty, raw)?),
            Type::NUMERIC => json!(numeric_text(raw)),
            Type::JSON | Type::JSONB => Value::from_sql(ty, raw)?,
            Type::TIMESTAMPTZ => json!(chrono::DateTime::<chrono::Utc>::from_sql(ty, raw)?.to_rfc3339()),
            Type::TIMESTAMP => json!(chrono::NaiveDateTime::from_sql(ty, raw)?.to_string()),
            Type::DATE => json!(chrono::NaiveDate::from_sql(ty, raw)?.to_string()),
            Type::UUID if raw.len() == 16 => {
                let h: String = raw.iter().map(|b| format!("{b:02x}")).collect();
                json!(format!("{}-{}-{}-{}-{}", &h[0..8], &h[8..12], &h[12..16], &h[16..20], &h[20..]))
            }
            _ => match std::str::from_utf8(raw) {
                // text-like types (text, varchar, enums, citext) are UTF-8 in binary too
                Ok(s) if matches!(*ty, Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME) || matches!(ty.kind(), tokio_postgres::types::Kind::Enum(_)) => json!(s),
                _ => json!(format!("<{} value; cast it to text in RETURNING>", ty.name())),
            },
        };
        Ok(AnyValue(v))
    }
    fn accepts(_ty: &Type) -> bool {
        true
    }
}

/// Postgres binary `numeric` to its decimal string.
fn numeric_text(raw: &[u8]) -> String {
    if raw.len() < 8 {
        return "NaN".into();
    }
    let rd = |i: usize| i16::from_be_bytes([raw[i], raw[i + 1]]);
    let (ndigits, weight, sign, dscale) = (rd(0) as usize, rd(2) as i32, rd(4) as u16, rd(6) as i32);
    if sign == 0xC000 {
        return "NaN".into();
    }
    let digits: Vec<i32> = (0..ndigits).map(|k| rd(8 + 2 * k) as i32).collect();
    let mut int_part = String::new();
    for w in 0..=weight.max(-1) {
        let d = digits.get(w as usize).copied().unwrap_or(0);
        if int_part.is_empty() {
            int_part = d.to_string();
        } else {
            int_part.push_str(&format!("{d:04}"));
        }
    }
    if int_part.is_empty() {
        int_part = "0".into();
    }
    let mut frac = String::new();
    let mut k = weight + 1;
    while (frac.len() as i32) < dscale {
        let d = if k < 0 { 0 } else { digits.get(k as usize).copied().unwrap_or(0) };
        frac.push_str(&format!("{d:04}"));
        k += 1;
    }
    frac.truncate(dscale.max(0) as usize);
    let s = if frac.is_empty() { int_part } else { format!("{int_part}.{frac}") };
    if sign == 0x4000 {
        format!("-{s}")
    } else {
        s
    }
}

fn rows_json(rows: &[Row]) -> Vec<Value> {
    rows.iter()
        .map(|r| {
            let mut m = serde_json::Map::new();
            for (i, c) in r.columns().iter().enumerate() {
                let v = r.try_get::<_, Option<AnyValue>>(i).ok().flatten().map(|a| a.0).unwrap_or(Value::Null);
                m.insert(c.name().to_string(), v);
            }
            Value::Object(m)
        })
        .collect()
}

/// The first SQL keyword, comments skipped.
fn first_keyword(sql: &str) -> String {
    let mut s = sql.trim_start();
    loop {
        if let Some(rest) = s.strip_prefix("--") {
            s = rest.split_once('\n').map(|x| x.1).unwrap_or("").trim_start();
        } else if let Some(rest) = s.strip_prefix("/*") {
            s = rest.split_once("*/").map(|x| x.1).unwrap_or("").trim_start();
        } else {
            break;
        }
    }
    s.split(|c: char| !c.is_ascii_alphabetic()).next().unwrap_or("").to_ascii_lowercase()
}

/// The statements of a write-back call: `{sql, params}` or
/// `{statements: [{sql, params}, …]}`.
fn statements(args: &Value) -> Result<Vec<(String, Vec<Value>)>> {
    let one = |v: &Value| -> Result<(String, Vec<Value>)> {
        let sql = v.get("sql").and_then(|x| x.as_str()).ok_or_else(|| miette!("pg write-back needs 'sql'"))?.to_string();
        let params = match v.get("params") {
            None | Some(Value::Null) => vec![],
            Some(Value::Array(a)) => a.clone(),
            Some(_) => bail!("pg write-back: 'params' must be a list"),
        };
        let kw = first_keyword(&sql);
        if !["insert", "update", "delete", "merge", "with"].contains(&kw.as_str()) {
            bail!("pg write-back: only INSERT, UPDATE, DELETE, MERGE or WITH statements, got '{kw}'");
        }
        Ok((sql, params))
    };
    match args.get("statements") {
        Some(Value::Array(list)) if !list.is_empty() => list.iter().map(one).collect(),
        Some(_) => bail!("pg write-back: 'statements' must be a non-empty list"),
        None => Ok(vec![one(args)?]),
    }
}

/// Run a write-back exactly once in effect under `idem_key`.
pub fn execute(cfg: &Arc<PgConfig>, idem_key: &str, workflow: Option<&str>, step: Option<&str>, args: &Value) -> Result<Value> {
    if cfg.writeback.allow.is_empty() {
        bail!("pg write-back is off for source '{}': list the tables flows may write in writeback.allow", cfg.source);
    }
    let stmts = statements(args)?;
    let pool = pool(cfg)?;
    let cfg = cfg.clone();
    let (idem, wf, st) = (idem_key.to_string(), workflow.map(|s| s.to_string()), step.map(|s| s.to_string()));
    conn::block_on(async move { run(pool, cfg, idem, wf, st, stmts).await })
}

async fn run(pool: Pool<Mgr>, cfg: Arc<PgConfig>, idem: String, wf: Option<String>, step: Option<String>, stmts: Vec<(String, Vec<Value>)>) -> Result<Value> {
    let mut client = pool.get().await.map_err(|e| miette!("write-back connection: {e}"))?;
    let tx = client.transaction().await.map_err(conn::pg_err)?;
    tx.batch_execute(&format!("SET LOCAL statement_timeout = {}", cfg.writeback.statement_timeout_ms)).await.map_err(conn::pg_err)?;
    // one attempt per key at a time; the ledger answers repeats
    tx.execute("SELECT pg_advisory_xact_lock(hashtextextended($1, 7))", &[&TextParam(Some(idem.clone()))])
        .await
        .map_err(conn::pg_err)?;
    let ledger = qident(&cfg.writeback.ledger);
    if let Some(row) = tx
        .query_opt(&format!("SELECT result FROM {ledger} WHERE idem_key = $1"), &[&TextParam(Some(idem.clone()))])
        .await
        .map_err(|e| match e.code() {
            Some(c) if c.code() == "42P01" => miette!(
                "the write-back ledger {} does not exist: run `kleos pg init` (or create it: {})",
                cfg.writeback.ledger,
                crate::admin::ledger_ddl(&cfg)
            ),
            _ => conn::pg_err(e),
        })?
    {
        let stored: Option<Value> = row.get(0);
        tx.commit().await.map_err(conn::pg_err)?;
        return Ok(json!({"idem_key": idem, "replayed": true, "result": stored}));
    }
    let mut results = vec![];
    for (sql, params) in &stmts {
        let ps: Vec<TextParam> = params.iter().map(text_param).collect();
        let refs: Vec<&(dyn ToSql + Sync)> = ps.iter().map(|p| p as &(dyn ToSql + Sync)).collect();
        let stmt = tx.prepare(sql).await.map_err(conn::pg_err)?;
        if stmt.columns().is_empty() {
            let n = tx.execute(&stmt, &refs).await.map_err(conn::pg_err)?;
            results.push(json!({"affected": n}));
        } else {
            let rows = tx.query(&stmt, &refs).await.map_err(conn::pg_err)?;
            results.push(json!({"affected": rows.len(), "rows": rows_json(&rows)}));
        }
    }
    // the allow-list, against what the transaction really modified
    let touched = tx
        .query(
            "SELECT schemaname || '.' || relname FROM pg_stat_xact_user_tables WHERE n_tup_ins + n_tup_upd + n_tup_del > 0",
            &[],
        )
        .await
        .map_err(conn::pg_err)?;
    for r in &touched {
        let t: String = r.get(0);
        if t != cfg.writeback.ledger && !cfg.writeback_allows(&t) {
            drop(tx); // rolls back
            bail!("pg write-back: table {t} is not in writeback.allow; the transaction was rolled back");
        }
    }
    let result = if results.len() == 1 { results.pop().unwrap() } else { Value::Array(results) };
    tx.execute(
        &format!("INSERT INTO {ledger} (idem_key, workflow, step, actor, result) VALUES ($1, $2, $3, $4, $5::text::jsonb)"),
        &[
            &TextParam(Some(idem.clone())),
            &TextParam(wf.clone()),
            &TextParam(step.clone()),
            &TextParam(Some(cfg.service())),
            &TextParam(Some(result.to_string())),
        ],
    )
    .await
    .map_err(conn::pg_err)?;
    if std::env::var("KLEOS_PG_WB_CRASH").as_deref() == Ok("before_commit") {
        std::process::abort();
    }
    tx.commit().await.map_err(conn::pg_err)?;
    if std::env::var("KLEOS_PG_WB_CRASH").as_deref() == Ok("after_commit") {
        // test hook: the Postgres write committed, the flow step was not recorded
        std::process::abort();
    }
    Ok(json!({"idem_key": idem, "replayed": false, "result": result}))
}

/// Register the `pg` flow action (feature `flow-pg` in kleos-bin):
/// `{"pg": {"source"?, "sql", "params"?, "statements"?, "idem_key"?}}`. The
/// idempotency key defaults to `<workflow>/<step key>`, the same on every
/// retry and replay of the step.
pub fn register_flow_action() -> Result<()> {
    kleos::flow::register_action(
        "pg",
        Arc::new(|ctx: &kleos::flow::ActionCtx, args: &Value| {
            let src = match args.get("source").and_then(|v| v.as_str()) {
                Some(s) => s.to_string(),
                None => crate::ops::only_source().ok_or_else(|| miette!("flow pg: 'source' is required when several sources are configured"))?,
            };
            let cfg = crate::ops::config(&src).ok_or_else(|| miette!("flow pg: no configuration for source '{src}' in this process"))?;
            let idem = args.get("idem_key").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| ctx.idem_key());
            execute(&cfg, &idem, Some(&ctx.workflow), Some(&ctx.key), args)
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_decoding() {
        // 12.50: ndigits 2, weight 0, sign 0, dscale 2, digits [12, 5000]
        let raw = [0u8, 2, 0, 0, 0, 0, 0, 2, 0, 12, 0x13, 0x88];
        assert_eq!(numeric_text(&raw), "12.50");
        // -10000: ndigits 1, weight 1, sign 0x4000, dscale 0, digits [1]
        let raw = [0u8, 1, 0, 1, 0x40, 0, 0, 0, 0, 1];
        assert_eq!(numeric_text(&raw), "-10000");
    }

    #[test]
    fn keywords() {
        assert_eq!(first_keyword("  -- note\n /* x */ INSERT INTO t"), "insert");
        assert!(statements(&json!({"sql": "DROP TABLE t"})).is_err());
        assert!(statements(&json!({"sql": "update t set a = $1", "params": [1]})).is_ok());
    }
}
