/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Debezium change events (docs/specs/pg-connector.md, "Debezium adapter"):
//! the JSON that Debezium Server's HTTP sink posts (with or without the
//! `{schema, payload}` envelope), mapped by the same writer as native CDC.
//! MySQL/MariaDB, MongoDB, SQL Server, Oracle and Postgres sources work
//! without a driver in kleos.
//!
//! - `c` insert, `u` update (supersedes changed columns), `d` delete
//!   (retracts), `r` snapshot read (asserted, flagged snapshot), `t`
//!   truncate; tombstones (null) are ignored;
//! - valid time is `source.ts_ms` (the commit time);
//! - each source partition (connector + server/db) keeps its last applied
//!   position in kleos (`kleos_pg_offset`), committed with the facts:
//!   a redelivered event at or before it is skipped.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use kleos::DbInstance;
use miette::{bail, Result};
use serde_json::{json, Value};

use crate::catalog::{Catalog, ColumnInfo, TableInfo, TypeKind};
use crate::change::{Change, Op, Row, Tx};
use crate::config::PgConfig;
use crate::mapping::{self, Mapping};
use crate::sink::{Sink, Stats};
use crate::store::{self, s};

struct DbzSource {
    sink: Sink,
    cat: Catalog,
    offsets: HashMap<String, Vec<Value>>,
}

fn sources() -> &'static Mutex<BTreeMap<String, Arc<Mutex<DbzSource>>>> {
    static S: OnceLock<Mutex<BTreeMap<String, Arc<Mutex<DbzSource>>>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn open(db: &DbInstance, src: &str) -> Result<Arc<Mutex<DbzSource>>> {
    if let Some(x) = sources().lock().unwrap().get(src) {
        return Ok(x.clone());
    }
    let cfg: Arc<PgConfig> = match crate::ops::config(src) {
        Some(c) => c,
        None => Arc::new(PgConfig::parse(&format!("source: {src}\nconnection: debezium\n"))?),
    };
    store::ensure(db)?;
    store::put_source(db, &cfg)?;
    let (mapping, cat) = match store::latest_mapping(db, src)? {
        Some((_, m, _)) => {
            let mut cat = Catalog::default();
            for (q, st) in store::table_states(db, src)? {
                if let Some(info) = st.info.and_then(|i| serde_json::from_value::<TableInfo>(i).ok()) {
                    cat.tables.insert(q, info);
                }
            }
            (m, cat)
        }
        None => (Mapping { source: src.to_string(), tables: BTreeMap::new() }, Catalog::default()),
    };
    let mapping = Arc::new(RwLock::new(mapping));
    let sink = Sink::new(db.clone(), cfg, mapping, Arc::new(Stats::default()))?;
    let mut offsets = HashMap::new();
    for x in store::read(db, &format!("?[part, pos] := *{}{{src: $src, part, pos}}", store::OFFSET), vec![("src", s(src))])?.rows {
        if let (Some(p), Value::Array(v)) = (store::str_of(&x[0]), store::json_of(&x[1])) {
            offsets.insert(p, v);
        }
    }
    let d = Arc::new(Mutex::new(DbzSource { sink, cat, offsets }));
    sources().lock().unwrap().insert(src.to_string(), d.clone());
    Ok(d)
}

/// The payload of an event (the envelope's, or the event itself); `None`
/// for a tombstone.
fn payload(ev: &Value) -> Option<(Value, Option<Value>, Option<Value>)> {
    let (v, key) = match ev.get("value") {
        Some(v) if ev.get("key").is_some() => (v.clone(), ev.get("key").cloned()),
        _ => (ev.clone(), None),
    };
    if v.is_null() {
        return None;
    }
    let schema = v.get("schema").cloned();
    let p = match v.get("payload") {
        Some(p) if schema.is_some() || v.get("op").is_none() => p.clone(),
        _ => v,
    };
    if p.is_null() {
        return None;
    }
    let key = key.map(|k| k.get("payload").cloned().unwrap_or(k));
    Some((p, schema, key))
}

/// Field schemas of `before`/`after` by name, from the envelope.
fn field_types(schema: &Option<Value>) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    let Some(sc) = schema else { return out };
    for f in sc.get("fields").and_then(|x| x.as_array()).into_iter().flatten() {
        if f.get("field").and_then(|x| x.as_str()) == Some("after") || f.get("field").and_then(|x| x.as_str()) == Some("before") {
            for c in f.get("fields").and_then(|x| x.as_array()).into_iter().flatten() {
                if let Some(n) = c.get("field").and_then(|x| x.as_str()) {
                    out.entry(n.to_string()).or_insert_with(|| c.clone());
                }
            }
        }
    }
    out
}

fn kind_of(fs: Option<&Value>, sample: &Value) -> TypeKind {
    if let Some(f) = fs {
        let name = f.get("name").and_then(|x| x.as_str()).unwrap_or("");
        let ty = f.get("type").and_then(|x| x.as_str()).unwrap_or("");
        return match name {
            "io.debezium.time.Date" => TypeKind::Date,
            "io.debezium.time.Timestamp" | "io.debezium.time.MicroTimestamp" | "io.debezium.time.NanoTimestamp" => TypeKind::Timestamp,
            "io.debezium.time.ZonedTimestamp" => TypeKind::TimestampTz,
            "io.debezium.time.Time" | "io.debezium.time.MicroTime" | "io.debezium.time.ZonedTime" => TypeKind::Time,
            "org.apache.kafka.connect.data.Decimal" | "io.debezium.data.VariableScaleDecimal" => TypeKind::Numeric,
            "io.debezium.data.Json" => TypeKind::Json,
            "io.debezium.data.Uuid" => TypeKind::Uuid,
            _ => match ty {
                "boolean" => TypeKind::Bool,
                "int8" | "int16" | "int32" | "int64" => TypeKind::Int,
                "float" | "float32" | "float64" | "double" => TypeKind::Float,
                "string" => TypeKind::Text,
                "bytes" => TypeKind::Bytea,
                "array" => TypeKind::Array,
                _ => TypeKind::Other,
            },
        };
    }
    match sample {
        Value::Bool(_) => TypeKind::Bool,
        Value::Number(n) if n.is_i64() || n.is_u64() => TypeKind::Int,
        Value::Number(_) => TypeKind::Float,
        Value::Array(_) => TypeKind::Array,
        Value::Object(_) => TypeKind::Json,
        _ => TypeKind::Text,
    }
}

/// A Debezium value as kleos keeps it (dates, timestamps and decimals
/// decoded when the envelope says what they are; Mongo extended JSON
/// unwrapped).
fn convert(fs: Option<&Value>, v: &Value) -> Value {
    if let Value::Object(m) = v {
        for k in ["$oid", "$numberLong", "$numberInt", "$numberDecimal", "$numberDouble"] {
            if let Some(x) = m.get(k) {
                return match (k, x.as_str().and_then(|s| s.parse::<f64>().ok())) {
                    ("$oid", _) => x.clone(),
                    (_, Some(f)) if f.fract() == 0.0 && f.abs() < 9e15 => json!(f as i64),
                    (_, Some(f)) => json!(f),
                    _ => x.clone(),
                };
            }
        }
        if let Some(d) = m.get("$date") {
            let ms = d.as_i64().or_else(|| d.get("$numberLong").and_then(|x| x.as_str()).and_then(|s| s.parse().ok()));
            if let Some(ms) = ms {
                return json!(chrono::DateTime::from_timestamp_millis(ms).map(|t| t.to_rfc3339()).unwrap_or_default());
            }
            return d.clone();
        }
    }
    let Some(f) = fs else { return v.clone() };
    let name = f.get("name").and_then(|x| x.as_str()).unwrap_or("");
    match (name, v) {
        ("io.debezium.time.Date", Value::Number(n)) => n
            .as_i64()
            .and_then(|d| chrono::NaiveDate::from_ymd_opt(1970, 1, 1).map(|e| e + chrono::Duration::days(d)))
            .map(|d| json!(d.to_string()))
            .unwrap_or(v.clone()),
        ("io.debezium.time.Timestamp", Value::Number(n)) => n
            .as_i64()
            .and_then(chrono::DateTime::from_timestamp_millis)
            .map(|t| json!(t.naive_utc().format("%Y-%m-%dT%H:%M:%S%.f").to_string()))
            .unwrap_or(v.clone()),
        ("io.debezium.time.MicroTimestamp", Value::Number(n)) => n
            .as_i64()
            .and_then(chrono::DateTime::from_timestamp_micros)
            .map(|t| json!(t.naive_utc().format("%Y-%m-%dT%H:%M:%S%.f").to_string()))
            .unwrap_or(v.clone()),
        ("org.apache.kafka.connect.data.Decimal", Value::String(b64)) => {
            let scale = f.get("parameters").and_then(|p| p.get("scale")).and_then(|x| x.as_str()).and_then(|x| x.parse::<u32>().ok()).unwrap_or(0);
            decode_decimal(b64, scale).map(|x| json!(x)).unwrap_or(v.clone())
        }
        ("io.debezium.data.Json", Value::String(t)) => serde_json::from_str(t).unwrap_or(v.clone()),
        _ => v.clone(),
    }
}

/// Kafka Connect `Decimal`: big-endian two's-complement unscaled bytes,
/// base64.
fn decode_decimal(b64: &str, scale: u32) -> Option<String> {
    let bytes = base64_decode(b64)?;
    if bytes.is_empty() || bytes.len() > 16 {
        return None;
    }
    let mut n: i128 = if bytes[0] & 0x80 != 0 { -1 } else { 0 };
    for b in bytes {
        n = (n << 8) | b as i128;
    }
    let neg = n < 0;
    let digits = n.unsigned_abs().to_string();
    let s = if scale == 0 {
        digits
    } else {
        let sc = scale as usize;
        let padded = format!("{:0>width$}", digits, width = sc + 1);
        format!("{}.{}", &padded[..padded.len() - sc], &padded[padded.len() - sc..])
    };
    Some(if neg { format!("-{s}") } else { s })
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = vec![];
    let mut buf = 0u32;
    let mut bits = 0;
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            b'\n' | b'\r' | b' ' => continue,
            _ => return None,
        } as u32;
        buf = (buf << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Some(out)
}

fn row_of(v: Option<&Value>, types: &BTreeMap<String, Value>) -> Option<Row> {
    let v = v?;
    let v = match v {
        // Mongo: `after` is an extended-JSON string
        Value::String(t) => serde_json::from_str::<Value>(t).ok()?,
        Value::Null => return None,
        other => other.clone(),
    };
    let Value::Object(m) = v else { return None };
    Some(m.iter().map(|(k, x)| (k.clone(), convert(types.get(k), x))).collect())
}

/// A partition's position as a comparable list (or `None`: no ordering
/// guarantee, applied as it comes).
fn position(connector: &str, src: &Value) -> Option<Vec<Value>> {
    let n = |k: &str| src.get(k).and_then(|x| x.as_i64().or_else(|| x.as_str().and_then(|s| s.parse().ok())));
    match connector {
        "postgresql" => Some(vec![json!(n("lsn")?)]),
        "mysql" | "mariadb" => {
            let file = src.get("file").and_then(|x| x.as_str())?;
            let seq: i64 = file.rsplit('.').next().and_then(|s| s.parse().ok()).unwrap_or(0);
            Some(vec![json!(seq), json!(n("pos")?), json!(n("row").unwrap_or(0))])
        }
        "sqlserver" => Some(vec![
            json!(src.get("commit_lsn").and_then(|x| x.as_str())?),
            json!(src.get("change_lsn").and_then(|x| x.as_str()).unwrap_or("")),
            json!(n("event_serial_no").unwrap_or(0)),
        ]),
        "mongodb" => Some(vec![json!(n("ts_ms")?), json!(n("ord").unwrap_or(0))]),
        "oracle" => Some(vec![json!(src.get("scn").and_then(|x| x.as_str()).and_then(|s| s.parse::<i64>().ok())?)]),
        _ => None,
    }
}

fn cmp_pos(a: &[Value], b: &[Value]) -> std::cmp::Ordering {
    for (x, y) in a.iter().zip(b) {
        let o = match (x, y) {
            (Value::Number(p), Value::Number(q)) => p.as_f64().partial_cmp(&q.as_f64()).unwrap_or(std::cmp::Ordering::Equal),
            (Value::String(p), Value::String(q)) => p.cmp(q),
            _ => std::cmp::Ordering::Equal,
        };
        if o != std::cmp::Ordering::Equal {
            return o;
        }
    }
    a.len().cmp(&b.len())
}

/// Ingest one request: an event, or a list of events. Applied in one kleos
/// transaction together with the partitions' new positions.
pub fn ingest(db: &DbInstance, src: &str, body: Value) -> Result<Value> {
    let events: Vec<Value> = match body {
        Value::Array(a) => a,
        Value::Null => bail!("debezium: no events"),
        other => vec![other],
    };
    let d = open(db, src)?;
    let mut d = d.lock().unwrap();
    let cfg = {
        let c = crate::ops::config(src);
        c.unwrap_or_else(|| Arc::new(PgConfig::parse(&format!("source: {src}\nconnection: debezium\n")).unwrap()))
    };
    let mut txs: Vec<Tx> = vec![];
    let mut last_tx_id: Option<String> = None;
    let mut new_pos: HashMap<String, Vec<Value>> = HashMap::new();
    let (mut applied, mut skipped, mut ignored) = (0usize, 0usize, 0usize);
    for ev in &events {
        let Some((p, schema, key)) = payload(ev) else {
            ignored += 1;
            continue;
        };
        let op = p.get("op").and_then(|x| x.as_str()).unwrap_or("");
        let source = p.get("source").cloned().unwrap_or(Value::Null);
        let connector = source.get("connector").and_then(|x| x.as_str()).unwrap_or("unknown").to_string();
        let ns_part = source.get("schema").or(source.get("db")).and_then(|x| x.as_str()).unwrap_or("public");
        let tname = source.get("table").or(source.get("collection")).and_then(|x| x.as_str());
        let Some(tname) = tname else {
            ignored += 1;
            continue;
        };
        if op == "m" || op.is_empty() {
            ignored += 1;
            continue;
        }
        let table = format!("{ns_part}.{tname}");
        // the default include (public.*) takes every table a Debezium source sends
        if cfg.include != vec!["public.*".to_string()] && !cfg.includes(&table) {
            ignored += 1;
            continue;
        }
        let part = format!("{connector}:{}", source.get("name").or(source.get("db")).and_then(|x| x.as_str()).unwrap_or(""));
        let snapshot = op == "r";
        let pos = position(&connector, &source);
        if let (Some(pos), false) = (&pos, snapshot) {
            let prev = new_pos.get(&part).or(d.offsets.get(&part));
            if prev.map_or(false, |pv| cmp_pos(pos, pv) != std::cmp::Ordering::Greater) {
                skipped += 1;
                continue;
            }
            new_pos.insert(part.clone(), pos.clone());
        }
        let types = field_types(&schema);
        let before = row_of(p.get("before"), &types);
        let mut after = row_of(p.get("after"), &types);
        let mut missing = vec![];
        // Mongo without full documents: the changed fields only
        if after.is_none() && op == "u" {
            if let Some(ud) = p.get("updateDescription") {
                let mut r = Row::new();
                if let Some(Value::String(t)) = ud.get("updatedFields") {
                    if let Ok(Value::Object(m)) = serde_json::from_str::<Value>(t) {
                        for (k, v) in m {
                            r.insert(k, convert(None, &v));
                        }
                    }
                }
                for f in ud.get("removedFields").and_then(|x| x.as_array()).into_iter().flatten() {
                    if let Some(f) = f.as_str() {
                        r.insert(f.to_string(), Value::Null);
                    }
                }
                if let Some(k) = key.as_ref().and_then(|k| k.get("id")).and_then(|x| x.as_str()) {
                    let id = serde_json::from_str::<Value>(k).map(|v| convert(None, &v)).unwrap_or(json!(k));
                    r.insert("_id".into(), id);
                }
                after = Some(r);
            }
        }
        // the table's catalog entry and mapping, grown from what events show
        let sample = after.clone().or(before.clone()).unwrap_or_default();
        let key_cols: Vec<String> = cfg
            .table_override(&table)
            .and_then(|o| o.key.clone())
            .or_else(|| key.as_ref().and_then(|k| k.as_object()).map(|m| m.keys().cloned().collect()))
            .or_else(|| ["id", "_id"].iter().find(|c| sample.contains_key(**c)).map(|c| vec![c.to_string()]))
            .unwrap_or_default();
        if op == "u" && after.as_ref().map_or(false, |a| a.len() < sample.len()) {
            missing = sample.keys().filter(|k| !after.as_ref().unwrap().contains_key(*k)).cloned().collect();
        }
        if !sample.is_empty() {
            let mut info = d.cat.tables.get(&table).cloned().unwrap_or_else(|| TableInfo {
                oid: 0,
                schema: ns_part.to_string(),
                name: tname.to_string(),
                comment: None,
                columns: vec![],
                pk: key_cols.clone(),
                uniques: vec![],
                fks: vec![],
                replica_identity: "d".into(),
                partitioned: false,
                rows_estimate: 0,
            });
            let mut grew = false;
            for (c, v) in &sample {
                if v.is_null() || info.columns.iter().any(|x| &x.name == c) {
                    continue;
                }
                let kind = kind_of(types.get(c), v);
                let avg = match v {
                    Value::String(t) => Some(t.len() as f64),
                    _ => None,
                };
                info.columns.push(ColumnInfo {
                    name: c.clone(),
                    type_name: types.get(c).and_then(|f| f.get("type")).and_then(|x| x.as_str()).unwrap_or("json").to_string(),
                    type_oid: 0,
                    atttypid: 0,
                    kind,
                    not_null: false,
                    comment: None,
                    avg_len: avg,
                    enum_labels: vec![],
                });
                grew = true;
            }
            if info.pk.is_empty() && !key_cols.is_empty() {
                info.pk = key_cols.clone();
                grew = true;
            }
            if grew {
                let tm = mapping::infer_table(&cfg, &d.cat, &info);
                d.cat.tables.insert(table.clone(), info.clone());
                d.sink.install_table(&tm, "debezium schema")?;
                store::put_table_state(
                    db,
                    src,
                    &store::TableState { tbl: table.clone(), state: "ready".into(), info: serde_json::to_value(&info).ok(), ..Default::default() },
                )?;
            }
        }
        let ts_ms = source.get("ts_ms").or(p.get("ts_ms")).and_then(|x| x.as_i64()).unwrap_or_else(|| store::now_us() / 1000);
        let change = match op {
            "c" => Change { table: table.clone(), op: Op::Insert, new: after, old: None, missing },
            "r" => Change { table: table.clone(), op: Op::Read, new: after, old: None, missing },
            "u" => Change { table: table.clone(), op: Op::Update, new: after, old: before, missing },
            "d" => Change { table: table.clone(), op: Op::Delete, new: None, old: before.or(key.as_ref().and_then(|k| k.as_object()).map(|m| m.iter().map(|(a, b)| (a.clone(), b.clone())).collect())), missing },
            "t" => Change { table: table.clone(), op: Op::Truncate, new: None, old: None, missing },
            other => {
                log::warn!("debezium: unknown op '{other}' ignored");
                ignored += 1;
                continue;
            }
        };
        // events of one source transaction stay one kleos transaction
        let tx_id = p
            .get("transaction")
            .and_then(|t| t.get("id"))
            .and_then(|x| x.as_str())
            .map(|s| s.to_string())
            .or_else(|| source.get("txId").map(|x| x.to_string()));
        let same = tx_id.is_some() && tx_id == last_tx_id && txs.last().map_or(false, |t| t.snapshot == snapshot);
        if same {
            txs.last_mut().unwrap().changes.push(change);
        } else {
            txs.push(Tx { commit_ts: ts_ms * 1000, snapshot, changes: vec![change], ..Default::default() });
        }
        last_tx_id = tx_id;
        applied += 1;
    }
    let offsets: Vec<(String, Value)> = new_pos.iter().map(|(k, v)| (k.clone(), Value::Array(v.clone()))).collect();
    let out = d.sink.apply(txs, offsets)?;
    for (k, v) in new_pos {
        d.offsets.insert(k, v);
    }
    Ok(json!({"source": src, "events": events.len(), "applied": applied, "skipped": skipped, "ignored": ignored,
              "rows": out.rows, "facts": out.facts, "memories": out.memories}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimals_and_positions() {
        // 12345 unscaled = 0x3039, scale 2 → 123.45
        assert_eq!(decode_decimal("MDk=", 2).as_deref(), Some("123.45"));
        assert_eq!(decode_decimal("/w==", 0).as_deref(), Some("-1"));
        let a = position("mysql", &json!({"file": "mysql-bin.000003", "pos": 100, "row": 0})).unwrap();
        let b = position("mysql", &json!({"file": "mysql-bin.000004", "pos": 4, "row": 0})).unwrap();
        assert_eq!(cmp_pos(&a, &b), std::cmp::Ordering::Less);
        assert_eq!(convert(None, &json!({"$oid": "abc"})), json!("abc"));
    }
}
