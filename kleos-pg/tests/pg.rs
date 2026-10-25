/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! kleos-pg against a real Postgres (docs/specs/pg-connector.md, "Tests").
//! Set `KLEOS_PG_TEST_URL` to a superuser URL of a server with
//! `wal_level = logical` (e.g. `postgres://postgres@127.0.0.1:54317/postgres`);
//! without it the Postgres tests skip. Each test works in its own database.
//! The Debezium test needs no server.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use kleos::{DataValue, DbInstance, ScriptMutability};
use kleos_pg::{Connector, FailPoint, PgConfig};
use serde_json::{json, Value};

fn base_url() -> Option<String> {
    std::env::var("KLEOS_PG_TEST_URL").ok()
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap()
}

/// A fresh database `kt_<name>` (old slots and connections dropped); its URL.
fn fresh_db(rt: &tokio::runtime::Runtime, name: &str) -> String {
    let base = base_url().unwrap();
    let db = format!("kt_{name}");
    rt.block_on(async {
        let c = kleos_pg::conn::connect(&kleos_pg::conn::parse(&base).unwrap()).await.unwrap();
        c.batch_execute(&format!(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{db}' AND pid <> pg_backend_pid();
             SELECT pg_drop_replication_slot(slot_name) FROM pg_replication_slots WHERE database = '{db}' AND NOT active;"
        ))
        .await
        .unwrap();
        c.batch_execute(&format!("DROP DATABASE IF EXISTS {db}")).await.unwrap();
        c.batch_execute(&format!("CREATE DATABASE {db}")).await.unwrap();
    });
    match base.rsplit_once('/') {
        Some((head, tail)) => {
            let q = tail.split_once('?').map(|x| format!("?{}", x.1)).unwrap_or_default();
            format!("{head}/{db}{q}")
        }
        None => base,
    }
}

fn sql(rt: &tokio::runtime::Runtime, url: &str, script: &str) {
    let url = url.to_string();
    let script = script.to_string();
    rt.block_on(async move {
        let c = kleos_pg::conn::connect(&kleos_pg::conn::parse(&url).unwrap()).await.unwrap();
        c.batch_execute(&script).await.unwrap();
    });
}

fn sql_rows(rt: &tokio::runtime::Runtime, url: &str, q: &str) -> Vec<Vec<Option<String>>> {
    let url = url.to_string();
    let q = q.to_string();
    rt.block_on(async move {
        let c = kleos_pg::conn::connect(&kleos_pg::conn::parse(&url).unwrap()).await.unwrap();
        kleos_pg::conn::simple_rows(&c, &q).await.unwrap()
    })
}

fn cfg(url: &str, extra: &str) -> PgConfig {
    let batch = if extra.contains("batch:") { "" } else { "batch: {max_ms: 20}\n" };
    PgConfig::parse(&format!("source: crm\nconnection: {url}\n{batch}status_secs: 1\n{extra}")).unwrap()
}

fn q(db: &DbInstance, script: &str) -> Vec<Vec<DataValue>> {
    db.run_script(script, Default::default(), ScriptMutability::Immutable).unwrap().rows
}

fn wait(what: &str, secs: u64, mut f: impl FnMut() -> bool) {
    let t0 = Instant::now();
    while !f() {
        assert!(t0.elapsed() < Duration::from_secs(secs), "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Current facts `(p, o)` of an entity.
fn facts(db: &DbInstance, s: &str) -> Vec<(String, String)> {
    let r = db
        .run_script(
            "?[p, o] := *kleos_onto_fact{ns: 'default', s: $s, p, o, status @ 'NOW'}, status = 'current'",
            [("s".to_string(), DataValue::from(s))].into_iter().collect(),
            ScriptMutability::Immutable,
        )
        .unwrap();
    r.rows.iter().map(|x| (x[0].get_str().unwrap().to_string(), x[1].get_str().unwrap().to_string())).collect()
}

fn fact(db: &DbInstance, s: &str, p: &str) -> Option<String> {
    facts(db, s).into_iter().find(|(pp, _)| pp == p).map(|x| x.1)
}

fn rows_of(db: &DbInstance, table: &str) -> usize {
    q(db, &format!("?[count(key)] := *kleos_pg_row{{src: 'crm', tbl: '{table}', key}}"))
        .first()
        .and_then(|r| r[0].get_int())
        .unwrap_or(0) as usize
}

const SCHEMA: &str = "
CREATE TABLE customers (id serial PRIMARY KEY, name text NOT NULL, plan text, city text,
  notes text);
COMMENT ON TABLE customers IS 'Paying customers';
CREATE TABLE tags (id serial PRIMARY KEY, name text);
CREATE TABLE customer_tags (customer_id int REFERENCES customers(id), tag_id int REFERENCES tags(id),
  PRIMARY KEY (customer_id, tag_id));
CREATE TABLE orders (id serial PRIMARY KEY, customer_id int REFERENCES customers(id), amount numeric(10,2),
  status text);
CREATE TABLE audit (msg text);
INSERT INTO customers (name, plan, city, notes) VALUES
  ('Acme Corp', 'pro', 'Lisbon', 'Long-time customer since 2019; prefers email contact and quarterly invoices, renewal handled by Dana.'),
  ('Globex', 'free', 'Porto', 'Trial account opened after the October webinar; evaluating the analytics add-on with three seats.');
INSERT INTO tags (name) VALUES ('vip'), ('churn-risk');
INSERT INTO customer_tags VALUES (1, 1), (2, 2);
INSERT INTO orders (customer_id, amount, status) VALUES (1, 120.50, 'paid'), (2, 9.99, 'open');
";

fn start(rt: &tokio::runtime::Runtime, cfg: PgConfig, db: &DbInstance) -> std::sync::Arc<kleos_pg::source::Shared> {
    kleos_pg::register_config(cfg.clone());
    kleos_pg::store::ensure(db).unwrap();
    let c = Connector::new(cfg, db.clone());
    let shared = c.shared.clone();
    rt.spawn(async move { c.run().await });
    shared
}

fn ready(db: &DbInstance) -> bool {
    let r = q(db, "?[count(tbl)] := *kleos_pg_table{src: 'crm', tbl, state}, state = 'ready'");
    r.first().and_then(|x| x[0].get_int()).unwrap_or(0) >= 4
}

fn tmp(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("kleos-pg-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn snapshot_then_stream() {
    if base_url().is_none() {
        return;
    }
    let rt = rt();
    let url = fresh_db(&rt, "snap");
    sql(&rt, &url, SCHEMA);
    let db = DbInstance::new("mem", "", "").unwrap();
    let h = start(&rt, cfg(&url, ""), &db);
    wait("the snapshot", 60, || ready(&db));
    let acme = "pg:crm/public.customers/1";
    assert_eq!(fact(&db, acme, "customers.plan").as_deref(), Some("pro"));
    assert_eq!(fact(&db, acme, "customers.name").as_deref(), Some("Acme Corp"));
    // foreign key and join table are edges; the audit table (no key) is skipped
    assert_eq!(fact(&db, "pg:crm/public.orders/1", "orders.customer").as_deref(), Some(acme));
    assert_eq!(fact(&db, acme, "customer_tags").as_deref(), Some("pg:crm/public.tags/1"));
    assert_eq!(rows_of(&db, "public.audit"), 0);
    // snapshot facts are flagged as such
    let by = q(&db, "?[by] := *kleos_onto_fact{ns: 'default', s: 'pg:crm/public.customers/1', p: 'customers.plan', by @ 'NOW'}");
    assert_eq!(by[0][0].get_str(), Some("pg:crm:snapshot"));
    // the class and properties are in the ontology pg:crm, active on the namespace
    let onto = db.ontology_call("get", json!({"onto": "pg:crm"})).unwrap();
    assert!(onto["classes"].as_array().unwrap().iter().any(|c| c["name"] == "Customer"));
    assert!(onto["properties"].as_array().unwrap().iter().any(|p| p["name"] == "customers.plan" && p["card"] == "one"));
    // notes are memories of their own; rows have a summary memory; both recallable
    let hits = db.memory_call("recall", json!({"query": "quarterly invoices renewal", "k": 3})).unwrap();
    assert!(hits[0]["text"].as_str().unwrap().contains("quarterly invoices"), "{hits}");
    let hits = db.memory_call("recall", json!({"query": "Acme Corp plan", "k": 3})).unwrap();
    assert!(hits.as_array().unwrap().iter().any(|h| h["text"].as_str().unwrap().contains("plan: pro")), "{hits}");
    // streaming: a new customer and order, at their commit time
    let t0 = kleos_pg::store::now_us();
    sql(&rt, &url, "BEGIN; INSERT INTO customers (id, name, plan) VALUES (10, 'Initech', 'team'); INSERT INTO orders (customer_id, amount, status) VALUES (10, 42, 'paid'); COMMIT;");
    wait("the streamed insert", 30, || fact(&db, "pg:crm/public.orders/3", "orders.customer").is_some());
    assert_eq!(fact(&db, "pg:crm/public.customers/10", "customers.plan").as_deref(), Some("team"));
    let vt = q(&db, "?[vt] := *kleos_onto_fact{ns: 'default', s: 'pg:crm/public.customers/10', p: 'customers.plan', vt}");
    let ts = match &vt[0][0] {
        DataValue::Validity(v) => v.timestamp.0 .0,
        _ => 0,
    };
    assert!(ts >= t0 - 2_000_000, "valid time is the commit time");
    let st = kleos_pg::pg_call(&db, "status", json!({"source": "crm"})).unwrap();
    assert_eq!(st["running"], true);
    assert!(st["postgres"]["slot"]["active"] == true, "{st}");
    h.stop();
}

#[test]
fn update_supersedes_changed_facts() {
    if base_url().is_none() {
        return;
    }
    let rt = rt();
    let url = fresh_db(&rt, "upd");
    sql(&rt, &url, SCHEMA);
    let db = DbInstance::new("mem", "", "").unwrap();
    let h = start(&rt, cfg(&url, ""), &db);
    wait("the snapshot", 60, || ready(&db));
    let acme = "pg:crm/public.customers/1";
    sql(&rt, &url, "UPDATE customers SET plan = 'enterprise' WHERE id = 1");
    wait("the update", 30, || fact(&db, acme, "customers.plan").as_deref() == Some("enterprise"));
    // only the changed column: city keeps its single (snapshot) version
    let plans = q(&db, "?[o, vt] := *kleos_onto_fact{ns: 'default', s: 'pg:crm/public.customers/1', p: 'customers.plan', o, vt}");
    assert_eq!(plans.len(), 3, "pro asserted, pro retracted, enterprise asserted: {plans:?}");
    let cities = q(&db, "?[o, vt] := *kleos_onto_fact{ns: 'default', s: 'pg:crm/public.customers/1', p: 'customers.city', o, vt}");
    assert_eq!(cities.len(), 1);
    // the row memory is superseded: recall answers the current plan, the old one is linked
    let hits = db.memory_call("recall", json!({"query": "Acme Corp plan", "k": 5})).unwrap();
    let texts: Vec<&str> = hits.as_array().unwrap().iter().map(|h| h["text"].as_str().unwrap()).collect();
    assert!(texts.iter().any(|t| t.contains("plan: enterprise")) && !texts.iter().any(|t| t.contains("plan: pro")), "{texts:?}");
    let links = q(&db, "?[count(src)] := *kleos_mem_link{ns: 'default', src, rel: 'supersedes'}");
    assert!(links[0][0].get_int().unwrap() >= 1);
    // a foreign key moves: the edge is superseded too
    sql(&rt, &url, "UPDATE orders SET customer_id = 2 WHERE id = 1");
    wait("the edge update", 30, || fact(&db, "pg:crm/public.orders/1", "orders.customer").as_deref() == Some("pg:crm/public.customers/2"));
    // TOAST: a big unchanged notes column is not resent; the stored value stays
    let big = "x".repeat(20_000);
    sql(&rt, &url, &format!("UPDATE customers SET notes = '{big}' WHERE id = 2"));
    sql(&rt, &url, "UPDATE customers SET plan = 'pro' WHERE id = 2");
    wait("the toast update", 30, || fact(&db, "pg:crm/public.customers/2", "customers.plan").as_deref() == Some("pro"));
    let notes = q(&db, "?[vals] := *kleos_pg_row{src: 'crm', tbl: 'public.customers', key: '2', vals}");
    let v = match &notes[0][0] {
        DataValue::Json(j) => j.0.clone(),
        _ => Value::Null,
    };
    assert_eq!(v["notes"].as_str().map(|s| s.len()), Some(20_000), "the unchanged TOAST value was carried forward");
    h.stop();
}

#[test]
fn delete_retracts_entity() {
    if base_url().is_none() {
        return;
    }
    let rt = rt();
    let url = fresh_db(&rt, "del");
    sql(&rt, &url, SCHEMA);
    let db = DbInstance::new("mem", "", "").unwrap();
    let h = start(&rt, cfg(&url, ""), &db);
    wait("the snapshot", 60, || ready(&db));
    sql(&rt, &url, "DELETE FROM customer_tags WHERE customer_id = 2; DELETE FROM orders WHERE customer_id = 2; DELETE FROM customers WHERE id = 2");
    let globex = "pg:crm/public.customers/2";
    wait("the delete", 30, || facts(&db, globex).is_empty() && rows_of(&db, "public.customers") == 1);
    // history is kept: the facts are readable as of before the delete
    let all = q(&db, "?[count(o)] := *kleos_onto_fact{ns: 'default', s: 'pg:crm/public.customers/2', o}");
    assert!(all[0][0].get_int().unwrap() > 0);
    let node = q(&db, "?[class] := *kleos_onto_node{ns: 'default', node: 'pg:crm/public.customers/2', class @ 'NOW'}");
    assert!(node.is_empty(), "the node is retracted");
    let hits = db.memory_call("recall", json!({"query": "Globex trial webinar", "k": 5})).unwrap();
    assert!(!hits.as_array().unwrap().iter().any(|h| h["text"].as_str().unwrap().contains("Globex")), "{hits}");
    // truncate retracts every entity of a table
    sql(&rt, &url, "TRUNCATE tags CASCADE");
    wait("the truncate", 30, || rows_of(&db, "public.tags") == 0);
    h.stop();
}

#[test]
fn crash_resume_exactly_once() {
    if base_url().is_none() {
        return;
    }
    let rt = rt();
    let url = fresh_db(&rt, "crash");
    sql(&rt, &url, SCHEMA);
    let db = DbInstance::new("mem", "", "").unwrap();
    let h = start(&rt, cfg(&url, "batch: {max_changes: 10, max_ms: 20}"), &db);
    wait("the snapshot", 60, || ready(&db));
    // stop after committing to kleos but before acknowledging Postgres
    *h.failpoint.lock().unwrap() = Some(FailPoint::AfterCommit(2));
    for i in 0..20 {
        sql(&rt, &url, &format!("BEGIN; INSERT INTO orders (customer_id, amount, status) SELECT 1, g, 'paid' FROM generate_series({0}, {0} + 9) g; UPDATE customers SET city = 'c{i}' WHERE id = 1; COMMIT;", i * 10));
    }
    wait("the crash", 30, || h.live.lock().unwrap().restarts >= 1);
    *h.failpoint.lock().unwrap() = None;
    wait("the replay", 60, || rows_of(&db, "public.orders") == 202);
    wait("the last update", 30, || fact(&db, "pg:crm/public.customers/1", "customers.city").as_deref() == Some("c19"));
    // every transaction once: one current value, the city history has one version per update
    let cities = q(&db, "?[o, vt] := *kleos_onto_fact{ns: 'default', s: 'pg:crm/public.customers/1', p: 'customers.city', o, vt}");
    assert_eq!(cities.len(), 1 + 2 * 20, "snapshot assert + (retract, assert) per transaction: {}", cities.len());
    let ck = kleos_pg::store::checkpoint(&db, "crm").unwrap().unwrap();
    assert_eq!(ck.applied, 20);
    // the restart asks Postgres to start at kleos's checkpoint; a transaction
    // at or below it that still arrives is skipped by LSN
    let mapping = std::sync::Arc::new(std::sync::RwLock::new(kleos_pg::store::latest_mapping(&db, "crm").unwrap().unwrap().1));
    let mut sink = kleos_pg::sink::Sink::new(db.clone(), kleos_pg::ops::config("crm").unwrap(), mapping, Default::default()).unwrap();
    let replay = kleos_pg::change::Tx { lsn: ck.lsn, end_lsn: ck.end_lsn, commit_ts: ck.commit_ts, ..Default::default() };
    assert_eq!(sink.apply(vec![replay], vec![]).unwrap().skipped, 1);
    // a failure inside the kleos write: nothing of the batch stays, the retry applies it once
    *h.failpoint.lock().unwrap() = Some(FailPoint::BeforeCommit(1));
    sql(&rt, &url, "UPDATE customers SET city = 'after' WHERE id = 1");
    wait("the second crash", 30, || h.live.lock().unwrap().restarts >= 2);
    *h.failpoint.lock().unwrap() = None;
    wait("the retry", 60, || fact(&db, "pg:crm/public.customers/1", "customers.city").as_deref() == Some("after"));
    let ck = kleos_pg::store::checkpoint(&db, "crm").unwrap().unwrap();
    assert_eq!(ck.applied, 21);
    h.stop();
}

#[test]
fn ddl_change_remaps() {
    if base_url().is_none() {
        return;
    }
    let rt = rt();
    let url = fresh_db(&rt, "ddl");
    sql(&rt, &url, SCHEMA);
    let db = DbInstance::new("mem", "", "").unwrap();
    let h = start(&rt, cfg(&url, ""), &db);
    wait("the snapshot", 60, || ready(&db));
    let v0 = kleos_pg::store::latest_mapping(&db, "crm").unwrap().unwrap().0;
    sql(&rt, &url, "ALTER TABLE customers ADD COLUMN region text; UPDATE customers SET region = 'EU' WHERE id = 1");
    wait("the new column's fact", 30, || fact(&db, "pg:crm/public.customers/1", "customers.region").as_deref() == Some("EU"));
    let (v1, m, _) = kleos_pg::store::latest_mapping(&db, "crm").unwrap().unwrap();
    assert!(v1 > v0 && m.tables["public.customers"].col("region").is_some());
    let onto = db.ontology_call("get", json!({"onto": "pg:crm"})).unwrap();
    assert!(onto["properties"].as_array().unwrap().iter().any(|p| p["name"] == "customers.region"));
    let hist = db.ontology_call("history", json!({"ns": "default"})).unwrap();
    assert!(hist.to_string().contains("pg_map"), "mapping changes are in the ontology history");
    // a dropped column's facts end
    sql(&rt, &url, "ALTER TABLE customers DROP COLUMN plan; UPDATE customers SET city = 'Faro' WHERE id = 1");
    wait("the drop", 30, || fact(&db, "pg:crm/public.customers/1", "customers.city").as_deref() == Some("Faro"));
    assert_eq!(fact(&db, "pg:crm/public.customers/1", "customers.plan"), None);
    assert_eq!(fact(&db, "pg:crm/public.customers/2", "customers.plan"), None);
    // a new table is picked up (copied through a resync), then streams
    sql(&rt, &url, "CREATE TABLE invoices (id int PRIMARY KEY, customer_id int REFERENCES customers(id), total int); INSERT INTO invoices VALUES (1, 1, 100)");
    kleos_pg::pg_call(&db, "resync", json!({"source": "crm", "tables": ["public.invoices"]})).unwrap();
    // tables mode: the new table must be in the publication first
    sql(&rt, &url, "ALTER PUBLICATION kleos_crm ADD TABLE invoices");
    wait("the new table", 60, || fact(&db, "pg:crm/public.invoices/1", "invoices.total").as_deref() == Some("100"));
    sql(&rt, &url, "INSERT INTO invoices VALUES (2, 1, 7)");
    wait("its stream", 30, || fact(&db, "pg:crm/public.invoices/2", "invoices.total").as_deref() == Some("7"));
    assert_eq!(rows_of(&db, "public.invoices"), 2);
    h.stop();
}

#[test]
fn lost_slot_resyncs_and_reconciles() {
    if base_url().is_none() {
        return;
    }
    let rt = rt();
    let url = fresh_db(&rt, "lost");
    sql(&rt, &url, SCHEMA);
    let db = DbInstance::new("mem", "", "").unwrap();
    let h = start(&rt, cfg(&url, ""), &db);
    wait("the snapshot", 60, || ready(&db));
    h.stop();
    wait("the stop", 30, || h.live.lock().unwrap().state == "stopped");
    // while stopped: the slot is lost and rows change
    sql(&rt, &url, "SELECT pg_drop_replication_slot('kleos_crm'); DELETE FROM customer_tags WHERE customer_id = 2; DELETE FROM orders WHERE customer_id = 2; DELETE FROM customers WHERE id = 2; UPDATE customers SET plan = 'gold' WHERE id = 1");
    let h = start(&rt, cfg(&url, ""), &db);
    wait("the resync", 60, || fact(&db, "pg:crm/public.customers/1", "customers.plan").as_deref() == Some("gold"));
    wait("the reconcile", 30, || facts(&db, "pg:crm/public.customers/2").is_empty());
    let ev = kleos_pg::pg_call(&db, "status", json!({"source": "crm"})).unwrap();
    assert!(ev["events"].to_string().contains("gone"), "{ev}");
    h.stop();
}

/// Child process: start the write-back workflow and crash right after the
/// Postgres commit (before the flow records the step).
#[test]
#[ignore]
fn writeback_crash_child() {
    let (Ok(path), Ok(url)) = (std::env::var("KLEOS_PG_WB_DB"), std::env::var("KLEOS_PG_WB_URL")) else { return };
    let db = DbInstance::new("sqlite", &path, "").unwrap();
    kleos_pg::register_config(cfg(&url, "writeback: {allow: [public.tickets]}"));
    kleos_pg::writeback::register_flow_action().unwrap();
    // KLEOS_PG_WB_CRASH=after_commit aborts inside the step
    let _ = db.flow_call("tick", json!({"worker": "doomed", "lease_ms": 300}));
}

#[test]
fn writeback_crash_exactly_once() {
    if base_url().is_none() {
        return;
    }
    let rt = rt();
    let url = fresh_db(&rt, "wb");
    sql(&rt, &url, "CREATE TABLE tickets (id serial PRIMARY KEY, title text, customer text); CREATE TABLE other (id int)");
    let c = cfg(&url, "writeback: {allow: [public.tickets]}");
    kleos_pg::register_config(c.clone());
    let dir = tmp("wb");
    let path = dir.join("flow.db");
    let db = DbInstance::new("sqlite", &path, "").unwrap();
    kleos_pg::pg_call(&db, "init", json!({"source": "crm"})).unwrap();
    kleos_pg::writeback::register_flow_action().unwrap();
    db.flow_call(
        "define",
        json!({"name": "open_ticket", "graph": {"steps": [
            {"name": "t", "pg": {"source": "crm", "sql": "INSERT INTO tickets (title, customer) VALUES ($1, $2) RETURNING id", "params": ["$input.title", "Acme"]}},
            {"return": "$t"}]}}),
    )
    .unwrap();
    db.flow_call("start", json!({"workflow": "open_ticket", "id": "wb1", "input": {"title": "refund"}})).unwrap();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["writeback_crash_child", "--exact", "--ignored", "--test-threads=1"])
        .env("KLEOS_PG_WB_DB", &path)
        .env("KLEOS_PG_WB_URL", &url)
        .env("KLEOS_PG_WB_CRASH", "after_commit")
        .status()
        .unwrap();
    assert!(!status.success(), "the child crashed");
    // the write committed in Postgres; the step was not recorded
    assert_eq!(sql_rows(&rt, &url, "SELECT count(*) FROM tickets")[0][0].as_deref(), Some("1"));
    assert!(db.flow_call("history", json!({"id": "wb1"})).unwrap().to_string().find("\"step\"").is_none());
    // a new worker after the lease: the step runs again, finds its ledger row
    std::thread::sleep(Duration::from_millis(400));
    let t0 = Instant::now();
    loop {
        db.flow_call("tick", json!({"worker": "rescuer"})).unwrap();
        if db.flow_call("status", json!({"id": "wb1"})).unwrap()["status"] == "completed" {
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(30));
        std::thread::sleep(Duration::from_millis(100));
    }
    let out = db.flow_call("status", json!({"id": "wb1"})).unwrap();
    assert_eq!(out["output"]["replayed"], true, "{out}");
    assert_eq!(sql_rows(&rt, &url, "SELECT count(*) FROM tickets")[0][0].as_deref(), Some("1"), "written once");
    // the allow-list: a write elsewhere is rolled back
    let cfg2 = kleos_pg::ops::config("crm").unwrap();
    let err = kleos_pg::writeback::execute(&cfg2, "k-other", None, None, &json!({"sql": "WITH x AS (INSERT INTO other VALUES (1) RETURNING id) INSERT INTO tickets (title) SELECT 'sneaky' FROM x"}));
    assert!(err.is_err());
    assert_eq!(sql_rows(&rt, &url, "SELECT count(*) FROM other")[0][0].as_deref(), Some("0"));
}

#[test]
fn echo_suppression_links_flow_step() {
    if base_url().is_none() {
        return;
    }
    let rt = rt();
    let url = fresh_db(&rt, "echo");
    sql(&rt, &url, "CREATE TABLE tickets (id serial PRIMARY KEY, title text, status text)");
    let c = cfg(&url, "writeback: {allow: [public.tickets]}");
    let db = DbInstance::new("mem", "", "").unwrap();
    kleos_pg::register_config(c.clone());
    kleos_pg::pg_call(&db, "init", json!({"source": "crm"})).unwrap();
    let h = start(&rt, c, &db);
    wait("the snapshot", 60, || {
        q(&db, "?[count(tbl)] := *kleos_pg_table{src: 'crm', tbl, state}, state = 'ready'")[0][0].get_int().unwrap_or(0) >= 1
    });
    let cfg2 = kleos_pg::ops::config("crm").unwrap();
    let r = kleos_pg::writeback::execute(&cfg2, "wf9/s0", Some("wf9"), Some("s0"), &json!({"sql": "INSERT INTO tickets (title, status) VALUES ('refund', 'open') RETURNING id"}))
        .unwrap();
    assert_eq!(r["replayed"], false);
    let t = "pg:crm/public.tickets/1";
    wait("the echo", 30, || fact(&db, t, "tickets.status").is_some());
    let by = q(&db, "?[by] := *kleos_onto_fact{ns: 'default', s: 'pg:crm/public.tickets/1', p: 'tickets.status', by @ 'NOW'}");
    assert_eq!(by[0][0].get_str(), Some("pg:crm@flow:wf9/s0"), "the row is linked to the flow step");
    let echo = q(&db, "?[idem, wf, step] := *kleos_pg_echo{src: 'crm', idem, wf, step}");
    assert_eq!(echo.len(), 1);
    assert_eq!(echo[0][1].get_str(), Some("wf9"));
    // the ledger itself is not data: no entity, no class
    let ledger = q(&db, "?[s] := *kleos_onto_fact{ns: 'default', s}, starts_with(s, 'pg:crm/public.kleos_writeback')");
    assert!(ledger.is_empty());
    // one entity, one row memory: no duplicate from the echo
    assert_eq!(rows_of(&db, "public.tickets"), 1);
    let mems = q(&db, "?[count(id)] := *kleos_mem{ns: 'default', id, kind @ 'NOW'}, kind = 'pg_row'");
    assert_eq!(mems[0][0].get_int(), Some(1));
    // a repeated write-back returns the ledger's result
    let again = kleos_pg::writeback::execute(&cfg2, "wf9/s0", Some("wf9"), Some("s0"), &json!({"sql": "INSERT INTO tickets (title, status) VALUES ('refund', 'open') RETURNING id"}))
        .unwrap();
    assert_eq!(again["replayed"], true);
    assert_eq!(again["result"]["rows"][0]["id"], 1);
    h.stop();
}

#[test]
fn debezium_adapter() {
    let db = DbInstance::new("mem", "", "").unwrap();
    let src = |file: &str, pos: i64, ts: i64| -> Value { json!({"connector": "mysql", "name": "shop", "db": "inventory", "table": "customers", "file": file, "pos": pos, "row": 0, "ts_ms": ts}) };
    let ev = |op: &str, before: Value, after: Value, s: Value| json!({"schema": {"type": "struct", "fields": [
        {"field": "after", "fields": [{"field": "id", "type": "int32"}, {"field": "name", "type": "string"}, {"field": "plan", "type": "string"},
                                      {"field": "since", "type": "int32", "name": "io.debezium.time.Date"}]}]},
        "payload": {"op": op, "before": before, "after": after, "source": s, "ts_ms": 0}});
    // snapshot read, then insert, update, delete (MySQL through Debezium Server's HTTP sink)
    let r = kleos_pg::debezium::ingest(&db, "shop", json!([
        ev("r", Value::Null, json!({"id": 1, "name": "Acme", "plan": "pro", "since": 18000}), src("mysql-bin.000001", 4, 1_700_000_000_000)),
        ev("c", Value::Null, json!({"id": 2, "name": "Globex", "plan": "free", "since": 19000}), src("mysql-bin.000001", 100, 1_700_000_100_000)),
    ]))
    .unwrap();
    assert_eq!(r["applied"], 2);
    let shop = |id: i64, p: &str| -> Option<String> {
        let r = db
            .run_script(
                "?[o] := *kleos_onto_fact{ns: 'default', s: $s, p: $p, o, status @ 'NOW'}",
                [("s".to_string(), DataValue::from(format!("pg:shop/inventory.customers/{id}"))), ("p".to_string(), DataValue::from(p))].into_iter().collect(),
                ScriptMutability::Immutable,
            )
            .unwrap();
        r.rows.first().and_then(|x| x[0].get_str()).map(|s| s.to_string())
    };
    assert_eq!(shop(1, "inventory.customers.plan").as_deref(), Some("pro"));
    assert_eq!(shop(1, "inventory.customers.since").as_deref(), Some("2019-04-14"), "io.debezium.time.Date decoded");
    let upd = ev("u", json!({"id": 2, "name": "Globex", "plan": "free", "since": 19000}), json!({"id": 2, "name": "Globex", "plan": "team", "since": 19000}), src("mysql-bin.000002", 50, 1_700_000_200_000));
    kleos_pg::debezium::ingest(&db, "shop", upd.clone()).unwrap();
    assert_eq!(shop(2, "inventory.customers.plan").as_deref(), Some("team"));
    // a redelivered event (at-least-once HTTP sink) is skipped by position
    let again = kleos_pg::debezium::ingest(&db, "shop", upd).unwrap();
    assert_eq!(again["skipped"], 1);
    let n = q(&db, "?[o, vt] := *kleos_onto_fact{ns: 'default', s: 'pg:shop/inventory.customers/2', p: 'inventory.customers.plan', o, vt}");
    assert_eq!(n.len(), 3, "free, free retracted, team: {n:?}");
    // valid time is the source commit time
    let vt = q(&db, "?[vt] := *kleos_onto_fact{ns: 'default', s: 'pg:shop/inventory.customers/2', p: 'inventory.customers.plan', o: 'team', vt}");
    match &vt[0][0] {
        DataValue::Validity(v) => assert_eq!(v.timestamp.0 .0, 1_700_000_200_000_000),
        x => panic!("{x:?}"),
    }
    // delete and a tombstone
    kleos_pg::debezium::ingest(&db, "shop", json!([
        ev("d", json!({"id": 2, "name": "Globex", "plan": "team", "since": 19000}), Value::Null, src("mysql-bin.000002", 90, 1_700_000_300_000)),
        Value::Null,
    ]))
    .unwrap();
    assert_eq!(shop(2, "inventory.customers.plan"), None);
    // Mongo: `after` is an extended-JSON string, `_id` an ObjectId
    let mongo = json!({"payload": {"op": "c", "after": "{\"_id\": {\"$oid\": \"65a1\"}, \"name\": \"Hooli\", \"seats\": {\"$numberLong\": \"12\"}}",
        "source": {"connector": "mongodb", "name": "m", "db": "crm", "collection": "accounts", "ts_ms": 1_700_000_400_000i64, "ord": 1}}});
    kleos_pg::debezium::ingest(&db, "shop", mongo).unwrap();
    let r = q(&db, "?[o] := *kleos_onto_fact{ns: 'default', s: 'pg:shop/crm.accounts/65a1', p: 'crm.accounts.seats', o @ 'NOW'}");
    assert_eq!(r[0][0].get_str(), Some("12"));
}
