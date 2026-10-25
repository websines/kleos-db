/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The `pg` ops behind the CLI (`kleos pg …`), `POST /pg/:op` and the MCP
//! tools: init, status (lag, slot, tables), resync, drop-slot, map
//! (+ approve in review mode), write-back and live queries.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use kleos::DbInstance;
use miette::{bail, miette, Result};
use serde_json::{json, Value};

use crate::admin;
use crate::config::PgConfig;
use crate::mapping::{self, Mapping, TableMap};
use crate::store::{self, i, j, s};

/// Configurations known to this process (connectors started here, or a
/// `--config` given to the CLI), by source: the connection is needed for the
/// Postgres side of `status`, `drop_slot`, `init`, write-back and queries.
fn configs() -> &'static Mutex<BTreeMap<String, Arc<PgConfig>>> {
    static C: OnceLock<Mutex<BTreeMap<String, Arc<PgConfig>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(BTreeMap::new()))
}

pub fn register_config(cfg: PgConfig) -> Arc<PgConfig> {
    let a = Arc::new(cfg);
    configs().lock().unwrap().insert(a.source.clone(), a.clone());
    a
}

pub fn config(source: &str) -> Option<Arc<PgConfig>> {
    configs().lock().unwrap().get(source).cloned()
}

/// The source when exactly one is configured here.
pub fn only_source() -> Option<String> {
    let c = configs().lock().unwrap();
    (c.len() == 1).then(|| c.keys().next().cloned()).flatten()
}

fn source_arg(args: &Value) -> Result<String> {
    if let Some(s) = args.get("source").and_then(|v| v.as_str()) {
        return Ok(s.to_string());
    }
    let known: Vec<String> = configs().lock().unwrap().keys().cloned().collect();
    match known.as_slice() {
        [one] => Ok(one.clone()),
        [] => bail!("pg: 'source' is required (no connector configured in this process)"),
        _ => bail!("pg: 'source' is required (configured: {})", known.join(", ")),
    }
}

fn need_config(source: &str) -> Result<Arc<PgConfig>> {
    config(source).ok_or_else(|| miette!("pg: no configuration for source '{source}' in this process (pass --config, or run the server with --pg-cdc)"))
}

/// One entry point for every surface.
pub fn pg_call(db: &DbInstance, op: &str, args: Value) -> Result<Value> {
    match op {
        "status" => match args.get("source").and_then(|v| v.as_str()) {
            Some(src) => status(db, src),
            None => {
                let mut names: Vec<String> = store::sources(db)?;
                for k in configs().lock().unwrap().keys() {
                    if !names.contains(k) {
                        names.push(k.clone());
                    }
                }
                let mut out = vec![];
                for n in names {
                    out.push(status(db, &n)?);
                }
                Ok(Value::Array(out))
            }
        },
        "init" => init(db, &source_arg(&args)?),
        "resync" => {
            let src = source_arg(&args)?;
            let tables: Option<Vec<String>> = args.get("tables").and_then(|v| serde_json::from_value(v.clone()).ok());
            resync(db, &src, tables)
        }
        "drop_slot" | "drop-slot" => {
            let src = source_arg(&args)?;
            let cfg = need_config(&src)?;
            let slot = cfg.slot();
            let c2 = cfg.clone();
            let dropped = crate::conn::block_on(async move {
                let pg = crate::conn::connect(&crate::conn::parse(&c2.connection)?).await?;
                admin::drop_slot(&pg, &c2.slot()).await
            })?;
            store::log_event(db, &src, "warn", &format!("slot {slot} dropped by request; the next start recreates it and resyncs every table"));
            Ok(json!({"source": src, "slot": slot, "dropped": dropped}))
        }
        "map" => {
            let src = source_arg(&args)?;
            map(db, &src, args.get("infer").and_then(|v| v.as_bool()).unwrap_or(false))
        }
        "approve" => {
            let src = source_arg(&args)?;
            approve(db, &src, args.get("version").and_then(|v| v.as_i64()))
        }
        "writeback" => {
            let src = source_arg(&args)?;
            let cfg = need_config(&src)?;
            let idem = args.get("idem_key").and_then(|v| v.as_str()).ok_or_else(|| miette!("pg writeback needs 'idem_key'"))?.to_string();
            crate::writeback::execute(&cfg, &idem, None, None, &args)
        }
        "debezium" => {
            let src = source_arg(&args)?;
            let events = args.get("events").cloned().unwrap_or(Value::Null);
            crate::debezium::ingest(db, &src, events)
        }
        "query" => {
            let src = source_arg(&args)?;
            let cfg = need_config(&src)?;
            let sql = args.get("sql").and_then(|v| v.as_str()).ok_or_else(|| miette!("pg query needs 'sql'"))?;
            query(db, &cfg, sql, &args)
        }
        other => bail!("unknown pg op '{other}': expected init, status, resync, drop_slot, map, approve, writeback, debezium or query"),
    }
}

#[cfg(feature = "query")]
fn query(db: &DbInstance, cfg: &Arc<PgConfig>, sql: &str, args: &Value) -> Result<Value> {
    crate::query::run(db, cfg, sql, args)
}

#[cfg(not(feature = "query"))]
fn query(_db: &DbInstance, _cfg: &Arc<PgConfig>, _sql: &str, _args: &Value) -> Result<Value> {
    bail!("pg query needs kleos built with the `pg-query` feature (DataFusion live queries)")
}

/// The Postgres side of a source: server, slot and lag; `None` fields when
/// it cannot be reached.
async fn pg_side(cfg: Arc<PgConfig>) -> Result<(admin::ServerInfo, Option<admin::SlotInfo>, Option<Vec<String>>)> {
    let pg = crate::conn::connect(&crate::conn::parse(&cfg.connection)?).await?;
    let info = admin::server_info(&pg).await?;
    let slot = admin::slot_info(&pg, &cfg.slot(), info.version).await?;
    let pubt = admin::publication_tables(&pg, &cfg.publication()).await?;
    Ok((info, slot, pubt))
}

/// `status`: kleos's checkpoint and tables, the running connector's live
/// state, the slot and lag from Postgres, and alerts.
pub fn status(db: &DbInstance, src: &str) -> Result<Value> {
    let ckpt = store::checkpoint(db, src).unwrap_or(None);
    let tables = store::table_states(db, src).unwrap_or_default();
    let mapping = store::latest_mapping(db, src).unwrap_or(None);
    let events = store::events(db, src, 10).unwrap_or_default();
    let echoes = store::read(db, &format!("?[count(lsn)] := *{}{{src: $src, lsn}}", store::ECHO), vec![("src", s(src))])
        .ok()
        .and_then(|r| r.rows.first().and_then(|x| x[0].get_int()))
        .unwrap_or(0);
    let live = crate::source::registry().lock().unwrap().get(src).cloned();
    let live_v = live.as_ref().map(|l| {
        let st = &l.stats;
        use std::sync::atomic::Ordering::Relaxed;
        json!({
            "live": *l.live.lock().unwrap(),
            "applied": {"txs": st.txs.load(Relaxed), "rows": st.rows.load(Relaxed), "facts": st.facts.load(Relaxed),
                        "memories": st.memories.load(Relaxed), "skipped_replays": st.skipped.load(Relaxed), "echoes": st.echoes.load(Relaxed)}
        })
    });
    let cfg = config(src);
    let mut alerts: Vec<String> = vec![];
    let mut warnings: Vec<String> = vec![];
    let mut pg_v = Value::Null;
    let mut lag = json!({"bytes": null, "seconds": null});
    if let Some(cfg) = cfg.clone() {
        let c2 = cfg.clone();
        match crate::conn::block_on(pg_side(c2)) {
            Ok((info, slot, pubt)) => {
                warnings.extend(admin::warnings(&info));
                alerts.extend(admin::blockers(&info));
                let lag_bytes = slot.as_ref().and_then(|s| s.lag_bytes);
                let last_commit = live
                    .as_ref()
                    .and_then(|l| l.live.lock().unwrap().last_commit_ts)
                    .or(ckpt.as_ref().map(|c| c.commit_ts).filter(|t| *t > 0));
                // time lag: how old the last applied commit is, while WAL is pending
                let lag_secs = match (lag_bytes, last_commit) {
                    (Some(b), Some(t)) if b > 0 => Some(((store::now_us() - t) / 1_000_000).max(0)),
                    (Some(_), _) => Some(0),
                    _ => None,
                };
                lag = json!({"bytes": lag_bytes, "seconds": lag_secs});
                match &slot {
                    None => alerts.push(format!("slot {} does not exist (the connector creates it on start; a missing slot means a full resync)", cfg.slot())),
                    Some(si) => {
                        if si.invalid() {
                            alerts.push(format!("slot {} is invalidated ({}): the next start recreates it and resyncs", cfg.slot(), si.invalidation_reason.clone().or(si.wal_status.clone()).unwrap_or_default()));
                        }
                        if let Some(b) = si.lag_bytes {
                            if b > cfg.lag_alert.bytes {
                                alerts.push(format!("replication lag {b} bytes exceeds {}", cfg.lag_alert.bytes));
                            }
                        }
                        if si.wal_status.as_deref() == Some("unreserved") {
                            alerts.push("the slot's WAL is past max_slot_wal_keep_size and may be removed at the next checkpoint".into());
                        }
                        if !si.active && live.is_some() {
                            alerts.push("the slot is inactive although a connector runs here (reconnecting?)".into());
                        }
                    }
                }
                if let Some(s) = lag_secs {
                    if s > cfg.lag_alert.seconds {
                        alerts.push(format!("replication lag {s}s exceeds {}s", cfg.lag_alert.seconds));
                    }
                }
                pg_v = json!({"server": info, "slot": slot, "publication": cfg.publication(), "publication_tables": pubt});
            }
            Err(e) => alerts.push(format!("Postgres unreachable: {e}")),
        }
    }
    Ok(json!({
        "source": src,
        "slot_name": cfg.as_ref().map(|c| c.slot()),
        "ns": cfg.as_ref().map(|c| c.ns.clone()),
        "running": live.is_some(),
        "connector": live_v,
        "checkpoint": ckpt.map(|c| json!({"lsn": crate::conn::fmt_lsn(c.lsn), "end_lsn": crate::conn::fmt_lsn(c.end_lsn),
            "commit_ts": c.commit_ts, "applied_txs": c.applied, "consistent": crate::conn::fmt_lsn(c.consistent), "updated": c.updated})),
        "lag": lag,
        "tables": tables.values().map(|t| json!({"table": t.tbl, "state": t.state, "rows_copied": t.rows,
            "skip_before": crate::conn::fmt_lsn(t.skip_before), "error": t.error})).collect::<Vec<_>>(),
        "mapping": mapping.map(|(v, m, st)| json!({"version": v, "status": st, "tables": m.tables.len(),
            "skipped": m.tables.values().filter(|t| t.is_skip()).map(|t| t.table.clone()).collect::<Vec<_>>()})),
        "echoes": echoes,
        "postgres": pg_v,
        "alerts": alerts,
        "warnings": warnings,
        "events": events,
    }))
}

/// `init`: check the server, create the publication (and the write-back
/// ledger), infer the mapping and define the ontology. The slot is created
/// by the connector with its snapshot.
pub fn init(db: &DbInstance, src: &str) -> Result<Value> {
    let cfg = need_config(src)?;
    store::ensure(db)?;
    store::put_source(db, &cfg)?;
    let c2 = cfg.clone();
    let (info, cat, notes, ledger) = crate::conn::block_on(async move {
        let pg = crate::conn::connect(&crate::conn::parse(&c2.connection)?).await?;
        let info = admin::server_info(&pg).await?;
        let c3 = c2.clone();
        let cat = crate::catalog::read_catalog(&pg, info.version, &move |q| c3.includes(q), true).await?;
        let mut notes = vec![];
        let blockers = admin::blockers(&info);
        let mut ledger = None;
        if blockers.is_empty() {
            let m = mapping::infer(&c2, &cat);
            let mut tables: Vec<String> = m.tables.values().filter(|t| !t.is_skip()).map(|t| t.table.clone()).collect();
            if !c2.writeback.allow.is_empty() {
                match admin::ensure_ledger(&pg, &c2).await {
                    Ok(created) => ledger = Some(json!({"table": c2.writeback.ledger, "created": created})),
                    Err(e) => notes.push(format!("{e}")),
                }
            }
            if crate::catalog::table_exists(&pg, &c2.writeback.ledger).await.unwrap_or(false) {
                tables.push(c2.writeback.ledger.clone());
            }
            match admin::ensure_publication(&pg, &c2, &tables, info.version).await {
                Ok(n) => notes.extend(n),
                Err(e) => notes.push(format!("{e}")),
            }
        }
        Ok::<_, miette::Report>((info, cat, notes, ledger))
    })?;
    let blockers = admin::blockers(&info);
    let inferred = mapping::infer(&cfg, &cat);
    let (version, m) = sync_mapping(db, &cfg, inferred, "init")?;
    Ok(json!({
        "source": src,
        "ok": blockers.is_empty(),
        "server": info,
        "blockers": blockers,
        "warnings": admin::warnings(&info),
        "notes": notes,
        "ledger": ledger,
        "publication": cfg.publication(),
        "publication_ddl": admin::publication_ddl(&cfg, &m.tables.values().filter(|t| !t.is_skip()).map(|t| t.table.clone()).collect::<Vec<_>>(), info.version).ok(),
        "slot": cfg.slot(),
        "mapping_version": version,
        "tables": m.tables.values().map(|t| summary(t)).collect::<Vec<_>>(),
    }))
}

fn summary(t: &TableMap) -> Value {
    let shape = match &t.shape {
        mapping::Shape::Entity => "entity".to_string(),
        mapping::Shape::Edge { predicate, .. } => format!("edge {predicate}"),
        mapping::Shape::Skip { reason } => format!("skipped: {reason}"),
    };
    json!({
        "table": t.table, "class": t.class, "shape": shape, "key": t.key, "label": t.label, "row_memory": t.row_memory,
        "facts": t.columns.iter().filter(|c| c.kind == mapping::ColKind::Fact).map(|c| c.predicate.clone()).collect::<Vec<_>>(),
        "text_memories": t.columns.iter().filter(|c| c.kind == mapping::ColKind::Text).map(|c| c.name.clone()).collect::<Vec<_>>(),
        "edges": t.fks.iter().map(|f| json!({"predicate": f.predicate, "to": f.table})).collect::<Vec<_>>(),
        "dropped": t.columns.iter().filter(|c| c.kind == mapping::ColKind::Drop).map(|c| c.name.clone()).collect::<Vec<_>>(),
        "pii": t.columns.iter().filter(|c| c.pii).map(|c| c.name.clone()).collect::<Vec<_>>(),
    })
}

/// Mark tables for a resync (all mapped tables when `tables` is None). A
/// running connector picks the request up within seconds; otherwise the
/// next start does.
pub fn resync(db: &DbInstance, src: &str, tables: Option<Vec<String>>) -> Result<Value> {
    let states = store::table_states(db, src)?;
    let list: Vec<String> = match tables {
        Some(t) => t,
        None => store::latest_mapping(db, src)?.map(|(_, m, _)| m.tables.values().filter(|t| !t.is_skip()).map(|t| t.table.clone()).collect()).unwrap_or_default(),
    };
    for t in &list {
        let mut st = states.get(t).cloned().unwrap_or_else(|| store::TableState { tbl: t.clone(), ..Default::default() });
        st.state = "resync".into();
        store::put_table_state(db, src, &st)?;
    }
    store::log_event(db, src, "info", &format!("resync requested: {list:?}"));
    Ok(json!({"source": src, "resync": list}))
}

/// The current mapping and the ontology it defines; with `infer`, a dry
/// run of re-inference against the live catalog and its diff.
pub fn map(db: &DbInstance, src: &str, infer: bool) -> Result<Value> {
    let cur = store::latest_mapping(db, src)?;
    let mut out = json!({
        "source": src,
        "version": cur.as_ref().map(|c| c.0),
        "status": cur.as_ref().map(|c| c.2.clone()),
        "tables": cur.as_ref().map(|c| c.1.tables.values().map(summary).collect::<Vec<_>>()),
    });
    if let Some(cfg) = config(src) {
        if let Some((_, m, _)) = &cur {
            out["ontology"] = serde_json::to_value(mapping::ontology(&cfg, m)).unwrap_or(Value::Null);
        }
        if infer {
            let c2 = cfg.clone();
            let cat = crate::conn::block_on(async move {
                let pg = crate::conn::connect(&crate::conn::parse(&c2.connection)?).await?;
                let v = crate::conn::server_version(&pg).await?;
                let c3 = c2.clone();
                crate::catalog::read_catalog(&pg, v, &move |q| c3.includes(q), true).await
            })?;
            let new = mapping::infer(&cfg, &cat);
            let old = cur.map(|c| c.1).unwrap_or_default();
            out["diff"] = json!(mapping::diff(&old, &new).iter().map(|c| json!({"table": c.table, "kind": c.kind})).collect::<Vec<_>>());
        }
    }
    Ok(out)
}

/// Store `inferred` as the source's mapping unless it equals the latest;
/// returns the version in force and the mapping.
pub fn sync_mapping(db: &DbInstance, cfg: &PgConfig, inferred: Mapping, reason: &str) -> Result<(i64, Mapping)> {
    let latest = store::latest_mapping(db, &cfg.source)?;
    match latest {
        Some((v, m, _)) if m == inferred => {
            ensure_activation(db, cfg)?;
            Ok((v, m))
        }
        Some((_, m, _)) => {
            let v = record_mapping(db, cfg, &m, &inferred, reason)?;
            Ok((v, inferred))
        }
        None => {
            let empty = Mapping { source: cfg.source.clone(), tables: BTreeMap::new() };
            let v = record_mapping(db, cfg, &empty, &inferred, reason)?;
            Ok((v, inferred))
        }
    }
}

/// Install one table's mapping (new table, DDL, resync).
pub fn install_table(db: &DbInstance, cfg: &PgConfig, mapping: &Arc<RwLock<Mapping>>, tm: &TableMap, reason: &str) -> Result<()> {
    let old = mapping.read().unwrap().clone();
    if old.tables.get(&tm.table) == Some(tm) {
        return Ok(());
    }
    let mut new = old.clone();
    new.tables.insert(tm.table.clone(), tm.clone());
    record_mapping(db, cfg, &old, &new, reason)?;
    *mapping.write().unwrap() = new;
    Ok(())
}

/// A new mapping version: stored, and (unless review mode) the ontology
/// redefined with a change row per table in `kleos_onto_change`, so
/// `::ontology history` explains the evolution. Review mode files a
/// proposal instead; data keeps flowing with the new mapping either way.
pub fn record_mapping(db: &DbInstance, cfg: &PgConfig, old: &Mapping, new: &Mapping, reason: &str) -> Result<i64> {
    let latest = store::latest_mapping(db, &cfg.source)?;
    let changes = mapping::diff(old, new);
    if changes.is_empty() {
        if let Some((v, _, _)) = latest {
            return Ok(v);
        }
    }
    let version = latest.map(|l| l.0).unwrap_or(0) + 1;
    let review = cfg.ontology.review;
    store::save_mapping(db, &cfg.source, version, new, reason, if review { "proposed" } else { "applied" })?;
    if review {
        let body = json!({"source": cfg.source, "version": version, "reason": reason,
            "changes": changes.iter().map(|c| json!({"table": c.table, "kind": c.kind})).collect::<Vec<_>>(),
            "how": format!("kleos pg map --approve {version} (or POST /pg/approve {{source, version}})")});
        store::q(
            db,
            "?[ns, id, kind, body, evidence, score, status, created, decided] <- [[$ns, $id, 'pg_map', $body, [], 1.0, 'proposed', $at, null]] \
             :put kleos_onto_proposal {ns, id => kind, body, evidence, score, status, created, decided}",
            vec![("ns", s(&cfg.ns)), ("id", s(&format!("pr_pg_{}_{version}", cfg.source))), ("body", j(body)), ("at", i(store::now_us()))],
        )?;
        store::log_event(db, &cfg.source, "info", &format!("mapping v{version} proposed ({reason}): {} table change(s) await approval", changes.len()));
    } else {
        apply_ontology(db, cfg, old, new, version, reason)?;
    }
    Ok(version)
}

fn apply_ontology(db: &DbInstance, cfg: &PgConfig, old: &Mapping, new: &Mapping, version: i64, reason: &str) -> Result<()> {
    let mut doc = mapping::ontology(cfg, new);
    doc.version = format!("v{version}");
    let mut v = serde_json::to_value(&doc).map_err(|e| miette!("{e}"))?;
    v["replace"] = json!(true);
    db.ontology_call("define", v)?;
    let changes = mapping::diff(old, new);
    let at = store::now_us();
    let rows: Vec<kleos::DataValue> = changes
        .iter()
        .enumerate()
        .map(|(n, c)| {
            let before = serde_json::to_value(mapping::table_props(cfg, old, &c.table)).unwrap_or(json!([]));
            let after = serde_json::to_value(mapping::table_props(cfg, new, &c.table)).unwrap_or(json!([]));
            kleos::DataValue::List(vec![
                s(&cfg.ns),
                s(&format!("ch_pg_{}_{version}_{n}", cfg.source)),
                i(at),
                s("pg_map"),
                s(&cfg.onto()),
                s(&c.table),
                j(before),
                j(after),
                kleos::DataValue::List(vec![s(reason)]),
                s(&cfg.by()),
                kleos::DataValue::from(false),
            ])
        })
        .collect();
    if !rows.is_empty() {
        store::q(
            db,
            "?[ns, id, at, kind, onto, target, before, after, evidence, by, undone] <- $rows \
             :put kleos_onto_change {ns, id => at, kind, onto, target, before, after, evidence, by, undone}",
            vec![("rows", kleos::DataValue::List(rows))],
        )?;
    }
    ensure_activation(db, cfg)?;
    store::log_event(db, &cfg.source, "info", &format!("mapping v{version} applied ({reason}): {} table change(s)", changes.len()));
    Ok(())
}

/// Add `pg:<source>` to the namespace's active ontologies, keeping what is
/// active (the store default, `personal`, when nothing is).
pub fn ensure_activation(db: &DbInstance, cfg: &PgConfig) -> Result<()> {
    if !cfg.ontology.activate {
        return Ok(());
    }
    let onto = cfg.onto();
    if store::read(db, "?[o] := *kleos_onto{onto: o @ 'NOW'}, o = $o", vec![("o", s(&onto))])?.rows.is_empty() {
        return Ok(()); // not defined yet (review mode before the first approval)
    }
    let r = store::read(
        db,
        "?[ontos, policy, untyped] := *kleos_onto_ns{ns: $ns, ontos, policy, untyped @ 'NOW'}",
        vec![("ns", s(&cfg.ns))],
    )?;
    let (mut ontos, policy, untyped): (Vec<String>, Option<String>, Option<String>) = match r.rows.first() {
        Some(x) => (
            serde_json::from_value(store::json_of(&x[0])).unwrap_or_default(),
            store::str_of(&x[1]),
            store::str_of(&x[2]),
        ),
        None => {
            let d = store::read(db, "?[v] := *kleos_mem_config{key: 'onto_default', value: v}", vec![])?
                .rows
                .first()
                .and_then(|x| store::str_of(&x[0]));
            let base = match d.as_deref() {
                Some("none") | Some("off") | Some("") => vec![],
                Some(x) => vec![x.to_string()],
                None => vec!["personal".to_string()],
            };
            (base, None, None)
        }
    };
    if ontos.contains(&onto) {
        return Ok(());
    }
    ontos.push(onto);
    db.ontology_call("activate", json!({"ns": cfg.ns, "ontos": ontos, "policy": policy, "untyped": untyped}))?;
    Ok(())
}

/// Review mode: apply a proposed mapping version to the ontology.
pub fn approve(db: &DbInstance, src: &str, version: Option<i64>) -> Result<Value> {
    let cfg = need_config(src)?;
    let all = store::read(
        db,
        &format!("?[version, mapping, status] := *{}{{src: $src, version, mapping, status}} :order version", store::MAPPING),
        vec![("src", s(src))],
    )?;
    let mut last_applied = Mapping { source: src.to_string(), tables: BTreeMap::new() };
    let mut target: Option<(i64, Mapping)> = None;
    for x in &all.rows {
        let v = x[0].get_int().unwrap_or(0);
        let m: Mapping = serde_json::from_value(store::json_of(&x[1])).map_err(|e| miette!("{e}"))?;
        let st = store::str_of(&x[2]).unwrap_or_default();
        if st == "applied" {
            last_applied = m.clone();
        }
        if st == "proposed" && version.map_or(true, |w| w == v) {
            target = Some((v, m));
        }
    }
    let (v, m) = target.ok_or_else(|| miette!("pg approve: no proposed mapping version for '{src}'"))?;
    apply_ontology(db, &cfg, &last_applied, &m, v, "approved")?;
    store::save_mapping(db, src, v, &m, "approved", "applied")?;
    store::q(
        db,
        "?[ns, id, status, decided] <- [[$ns, $id, 'approved', $at]] :update kleos_onto_proposal {ns, id => status, decided}",
        vec![("ns", s(&cfg.ns)), ("id", s(&format!("pr_pg_{src}_{v}"))), ("at", i(store::now_us()))],
    )
    .ok();
    Ok(json!({"source": src, "approved": v}))
}
