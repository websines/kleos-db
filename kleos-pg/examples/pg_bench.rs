/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The kleos-pg benchmark (docs/benchmarks/pg-connector.md).
//!
//! ```text
//! KLEOS_PG_TEST_URL=postgres://postgres@127.0.0.1:54317/postgres \
//!   cargo run --release -p kleos-pg --example pg_bench -- --rows 100000 --dir <scratch dir>
//! ```
//!
//! A CRM/orders schema (accounts, contacts, products, orders, order_items,
//! tickets: foreign keys, a join table, long text) is generated in Postgres,
//! then: snapshot throughput, streaming throughput, end-to-end latency
//! (Postgres commit → recallable in kleos), recall of current values after
//! updates, and write-back exactly once under injected crashes.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use kleos::{DataValue, DbInstance, ScriptMutability};
use kleos_pg::{Connector, PgConfig};
use serde_json::{json, Value};

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1).cloned())
}

fn pg(rt: &tokio::runtime::Runtime, url: &str, script: &str) {
    let (u, s) = (url.to_string(), script.to_string());
    rt.block_on(async move {
        let c = kleos_pg::conn::connect(&kleos_pg::conn::parse(&u).unwrap()).await.unwrap();
        c.batch_execute(&s).await.unwrap();
    });
}

fn pg_rows(rt: &tokio::runtime::Runtime, url: &str, q: &str) -> Vec<Vec<Option<String>>> {
    let (u, s) = (url.to_string(), q.to_string());
    rt.block_on(async move {
        let c = kleos_pg::conn::connect(&kleos_pg::conn::parse(&u).unwrap()).await.unwrap();
        kleos_pg::conn::simple_rows(&c, &s).await.unwrap()
    })
}

fn kq(db: &DbInstance, s: &str) -> Vec<Vec<DataValue>> {
    db.run_script(s, Default::default(), ScriptMutability::Immutable).unwrap().rows
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

fn schema(n: i64) -> String {
    let acc = (n / 10).max(10);
    let con = (n / 4).max(10);
    let ord = (n / 4).max(10);
    let items = (n / 4).max(10);
    let tic = n - acc - con - ord - items - 200;
    format!(
        "
CREATE TABLE accounts (id int PRIMARY KEY, name text NOT NULL, industry text, plan text, region text,
  created_at timestamptz DEFAULT now(), notes text);
COMMENT ON TABLE accounts IS 'Customer companies';
CREATE TABLE contacts (id int PRIMARY KEY, account_id int REFERENCES accounts(id), first_name text, last_name text,
  email text, title text);
CREATE TABLE products (id int PRIMARY KEY, sku text UNIQUE, name text, price numeric(10,2));
CREATE TABLE orders (id int PRIMARY KEY, account_id int REFERENCES accounts(id), contact_id int REFERENCES contacts(id),
  status text, total numeric(12,2), created_at timestamptz DEFAULT now());
CREATE TABLE order_items (order_id int REFERENCES orders(id), product_id int REFERENCES products(id), qty int,
  price numeric(10,2), PRIMARY KEY (order_id, product_id));
CREATE TABLE tickets (id int PRIMARY KEY, account_id int REFERENCES accounts(id), subject text, body text,
  status text, priority text);
INSERT INTO accounts SELECT g, 'Account ' || g, (ARRAY['retail','fintech','health','media','logistics'])[1 + g % 5],
  (ARRAY['free','team','pro','enterprise'])[1 + g % 4], (ARRAY['EU','US','APAC'])[1 + g % 3], now(),
  'Account ' || g || ' signed in ' || (2015 + g % 10) || '; main contact prefers email; renewal reviewed every quarter with the success team.'
  FROM generate_series(1, {acc}) g;
INSERT INTO contacts SELECT g, 1 + g % {acc}, 'First' || g, 'Last' || g, 'user' || g || '@example.com',
  (ARRAY['CTO','buyer','engineer','CFO'])[1 + g % 4] FROM generate_series(1, {con}) g;
INSERT INTO products SELECT g, 'SKU-' || g, 'Product ' || g, (g % 500) + 0.99 FROM generate_series(1, 200) g;
INSERT INTO orders SELECT g, 1 + g % {acc}, 1 + g % {con}, (ARRAY['open','paid','shipped','refunded'])[1 + g % 4],
  (g % 10000) / 3.0, now() FROM generate_series(1, {ord}) g;
INSERT INTO order_items SELECT g, 1 + g % 200, 1 + g % 5, (g % 500) + 0.99 FROM generate_series(1, {items}) g;
INSERT INTO tickets SELECT g, 1 + g % {acc}, 'Ticket ' || g || ' about invoices',
  'The customer reports that invoice ' || g || ' shows the wrong VAT rate after the plan change; they need a corrected copy before month end and asked for a callback.',
  (ARRAY['open','pending','closed'])[1 + g % 3], (ARRAY['low','normal','high'])[1 + g % 3] FROM generate_series(1, {tic}) g;
ANALYZE;
"
    )
}

fn main() {
    let base = std::env::var("KLEOS_PG_TEST_URL").expect("KLEOS_PG_TEST_URL");
    if let Some(dbpath) = arg("--child-tick") {
        // write-back crash child: run due workflows; KLEOS_PG_WB_CRASH aborts inside the step
        let url = arg("--url").unwrap();
        let db = DbInstance::new("sqlite", &dbpath, "").unwrap();
        kleos_pg::register_config(PgConfig::parse(&format!("source: crm\nconnection: {url}\nwriteback: {{allow: [public.tickets]}}\n")).unwrap());
        kleos_pg::writeback::register_flow_action().unwrap();
        let _ = db.flow_call("tick", json!({"worker": "doomed", "lease_ms": 200, "max": 1}));
        return;
    }
    let rows: i64 = arg("--rows").and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let dir = std::path::PathBuf::from(arg("--dir").expect("--dir <scratch dir>"));
    let engine = arg("--engine").unwrap_or_else(|| "sqlite".into());
    let stream_txs: i64 = arg("--stream-txs").and_then(|s| s.parse().ok()).unwrap_or(2000);
    std::fs::create_dir_all(&dir).unwrap();
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build().unwrap();
    let mut report = BTreeMap::new();
    report.insert("rows_target".to_string(), json!(rows));
    report.insert("engine".to_string(), json!(engine));

    // a fresh database
    let dbname = "kbench";
    pg(&rt, &base, &format!(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{dbname}' AND pid <> pg_backend_pid();
         SELECT pg_drop_replication_slot(slot_name) FROM pg_replication_slots WHERE database = '{dbname}' AND NOT active;
         DROP DATABASE IF EXISTS {dbname}; CREATE DATABASE {dbname};"));
    let url = match base.rsplit_once('/') {
        Some((h, _)) => format!("{h}/{dbname}"),
        None => base.clone(),
    };
    let t = Instant::now();
    pg(&rt, &url, &schema(rows));
    let total: i64 = pg_rows(&rt, &url, "SELECT (SELECT count(*) FROM accounts) + (SELECT count(*) FROM contacts) + (SELECT count(*) FROM products) + (SELECT count(*) FROM orders) + (SELECT count(*) FROM order_items) + (SELECT count(*) FROM tickets)")[0][0].as_ref().unwrap().parse().unwrap();
    let accounts: i64 = pg_rows(&rt, &url, "SELECT count(*) FROM accounts")[0][0].as_ref().unwrap().parse().unwrap();
    eprintln!("generated {total} rows in {:.1}s", t.elapsed().as_secs_f64());
    report.insert("rows".into(), json!(total));

    // kleos
    let kpath = dir.join(format!("kleos.{engine}"));
    let _ = std::fs::remove_dir_all(&kpath);
    let _ = std::fs::remove_file(&kpath);
    let db = DbInstance::new(&engine, &kpath, "").unwrap();
    let cfg = PgConfig::parse(&format!(
        "source: crm\nconnection: {url}\nwriteback: {{allow: [public.tickets]}}\nbatch: {{max_changes: 5000, max_ms: 20, snapshot_rows: 2000}}\nstatus_secs: 1\n"
    ))
    .unwrap();
    kleos_pg::register_config(cfg.clone());
    kleos_pg::store::ensure(&db).unwrap();
    kleos_pg::pg_call(&db, "init", json!({"source": "crm"})).unwrap();

    // e2e latency probe: a callback on kleos_mem notes when each probe's memory commits
    let seen: Arc<Mutex<BTreeMap<String, Instant>>> = Arc::new(Mutex::new(BTreeMap::new()));
    let (cb_id, cb_rx) = db.register_callback("kleos_mem", None);
    let seen2 = seen.clone();
    std::thread::spawn(move || {
        for (_op, new, _old) in cb_rx.iter() {
            let now = Instant::now();
            for r in new.rows {
                if let Some(t) = r.get(3).and_then(|v| v.get_str()) {
                    if let Some(i) = t.find("plan: probe") {
                        let tok: String = t[i + 6..].chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
                        seen2.lock().unwrap().entry(tok).or_insert(now);
                    }
                }
            }
        }
    });

    // 1. snapshot
    let c = Connector::new(cfg.clone(), db.clone());
    let h = c.shared.clone();
    rt.spawn(async move { c.run().await });
    let t0 = Instant::now();
    loop {
        let r = kq(&db, "?[count(tbl)] := *kleos_pg_table{src: 'crm', tbl, state}, state = 'ready'");
        if r.first().and_then(|x| x[0].get_int()).unwrap_or(0) >= 6 {
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(7200), "snapshot timed out");
        std::thread::sleep(Duration::from_millis(100));
    }
    let snap = t0.elapsed().as_secs_f64();
    let facts = kq(&db, "?[count(s)] := *kleos_onto_fact{s}")[0][0].get_int().unwrap_or(0);
    let mems = kq(&db, "?[count(id)] := *kleos_mem{id}")[0][0].get_int().unwrap_or(0);
    eprintln!("snapshot: {total} rows in {snap:.1}s = {:.0} rows/s ({facts} fact rows, {mems} memories)", total as f64 / snap);
    report.insert("snapshot".into(), json!({"seconds": snap, "rows_per_s": total as f64 / snap, "fact_rows": facts, "memories": mems}));

    // 2. streaming throughput: transactions of 5 changes (3 updates + an order + its item) as fast as Postgres takes them
    let last_lsn = Arc::new(AtomicU64::new(0));
    let n_tx = stream_txs;
    let t1 = Instant::now();
    {
        let url = url.clone();
        let last = last_lsn.clone();
        rt.block_on(async move {
            let c = kleos_pg::conn::connect(&kleos_pg::conn::parse(&url).unwrap()).await.unwrap();
            for i in 0..n_tx {
                let a = 1 + (i * 7919) % accounts;
                let o = 10_000_000 + i;
                c.batch_execute(&format!(
                    "BEGIN; UPDATE accounts SET region = 'R{i}' WHERE id = {a}; UPDATE accounts SET industry = 'I{i}' WHERE id = {}; \
                     UPDATE contacts SET title = 'T{i}' WHERE id = {}; INSERT INTO orders VALUES ({o}, {a}, NULL, 'open', 10, now()); \
                     INSERT INTO order_items VALUES ({o}, 1, 1, 1); COMMIT;",
                    1 + (a % accounts), a
                ))
                .await
                .unwrap();
            }
            let r = kleos_pg::conn::simple_rows(&c, "SELECT pg_current_wal_lsn()::text").await.unwrap();
            last.store(kleos_pg::conn::parse_lsn(r[0][0].as_deref().unwrap()).unwrap(), Ordering::SeqCst);
        });
    }
    let write_s = t1.elapsed().as_secs_f64();
    let target = last_lsn.load(Ordering::SeqCst);
    loop {
        let ck = kleos_pg::store::checkpoint(&db, "crm").unwrap().unwrap_or_default();
        if ck.applied >= n_tx {
            break;
        }
        let _ = target;
        assert!(t1.elapsed() < Duration::from_secs(3600));
        std::thread::sleep(Duration::from_millis(20));
    }
    let stream_s = t1.elapsed().as_secs_f64();
    let changes = n_tx * 5;
    eprintln!("streaming: {n_tx} txs / {changes} changes in {stream_s:.1}s = {:.0} changes/s, {:.0} txs/s (Postgres writer took {write_s:.1}s)", changes as f64 / stream_s, n_tx as f64 / stream_s);
    report.insert("streaming".into(), json!({"txs": n_tx, "changes": changes, "seconds": stream_s, "changes_per_s": changes as f64 / stream_s,
        "txs_per_s": n_tx as f64 / stream_s, "pg_writer_seconds": write_s}));

    // 3. end-to-end latency: single-row plan changes at ~20/s; commit → memory committed (recallable)
    let probes = 200;
    let mut lat = vec![];
    let mut sent: Vec<(String, Instant)> = vec![];
    {
        let url = url.clone();
        let r: Vec<(String, Instant)> = rt.block_on(async move {
            let c = kleos_pg::conn::connect(&kleos_pg::conn::parse(&url).unwrap()).await.unwrap();
            let mut out = vec![];
            for i in 0..probes {
                let tok = format!("probe{i}x");
                let a = 1 + (i * 104729) % accounts;
                c.batch_execute(&format!("UPDATE accounts SET plan = '{tok}' WHERE id = {a}")).await.unwrap();
                out.push((tok, Instant::now()));
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            out
        });
        sent.extend(r);
    }
    std::thread::sleep(Duration::from_secs(2));
    {
        let s = seen.lock().unwrap();
        for (tok, t) in &sent {
            if let Some(v) = s.get(tok) {
                lat.push(v.saturating_duration_since(*t).as_secs_f64() * 1000.0);
            }
        }
    }
    let got = lat.len();
    let (p50, p95, p99, max) = (pct(&mut lat, 0.5), pct(&mut lat, 0.95), pct(&mut lat, 0.99), pct(&mut lat, 1.0));
    eprintln!("e2e latency ({got}/{probes} seen): p50 {p50:.1} ms, p95 {p95:.1} ms, p99 {p99:.1} ms, max {max:.1} ms");
    // and recallable: the last probe through recall itself
    let (tok, _) = sent.last().unwrap();
    let hits = db.memory_call("recall", json!({"query": tok, "k": 1})).unwrap();
    let recallable = hits[0]["text"].as_str().unwrap_or("").contains(tok.as_str());
    report.insert("e2e_latency_ms".into(), json!({"probes": probes, "seen": got, "p50": p50, "p95": p95, "p99": p99, "max": max,
        "note": "Postgres COMMIT returned → the row's memory committed in kleos (a kleos_mem change callback); FTS is updated in that transaction", "recall_check": recallable}));
    db.unregister_callback(cb_id);

    // 4. recall of current values after updates: 3 plan changes per sampled account
    let sample = 200.min(accounts);
    {
        let url = url.clone();
        rt.block_on(async move {
            let c = kleos_pg::conn::connect(&kleos_pg::conn::parse(&url).unwrap()).await.unwrap();
            for round in 0..3 {
                let mut s = String::from("BEGIN;");
                for i in 0..sample {
                    let a = 1 + (i * 37) % accounts;
                    s.push_str(&format!("UPDATE accounts SET plan = 'tier{round}{}' WHERE id = {a};", ["silver", "gold", "platinum"][(i % 3) as usize]));
                }
                s.push_str("COMMIT;");
                c.batch_execute(&s).await.unwrap();
            }
        });
    }
    let want_applied = n_tx + probes + 3;
    while kleos_pg::store::checkpoint(&db, "crm").unwrap().unwrap().applied < want_applied {
        std::thread::sleep(Duration::from_millis(20));
    }
    let (mut top1, mut top5, mut stale1) = (0, 0, 0);
    let mut rl = vec![];
    for i in 0..sample {
        let a = 1 + (i * 37) % accounts;
        let want = format!("plan: tier2{}", ["silver", "gold", "platinum"][(i % 3) as usize]);
        let q = format!("What is the current plan of Account {a}?");
        let tq = Instant::now();
        let hits = db.memory_call("recall", json!({"query": q, "k": 5, "record_access": false})).unwrap();
        rl.push(tq.elapsed().as_secs_f64() * 1000.0);
        let texts: Vec<String> = hits.as_array().unwrap().iter().map(|h| h["text"].as_str().unwrap_or("").to_string()).collect();
        let me = format!("Account {a} (accounts {a})");
        if texts.first().map_or(false, |t| t.contains(&me) && t.contains(&want)) {
            top1 += 1;
        }
        if texts.iter().any(|t| t.contains(&me) && t.contains(&want)) {
            top5 += 1;
        }
        if texts.first().map_or(false, |t| t.contains(&me) && t.contains("plan: tier") && !t.contains(&want)) {
            stale1 += 1;
        }
    }
    let ctx = db.memory_call("recall", json!({"query": "What is the current plan of Account 1?", "format": "context", "k": 3})).unwrap();
    let rp50 = pct(&mut rl, 0.5);
    let rp95 = pct(&mut rl, 0.95);
    eprintln!("recall (FTS, no embedder): current value top-1 {top1}/{sample}, top-5 {top5}/{sample}, stale top-1 {stale1}; p50 {rp50:.1} ms, p95 {rp95:.1} ms");
    report.insert("recall".into(), json!({"questions": sample, "current_top1": top1, "current_top5": top5, "stale_top1": stale1,
        "p50_ms": rp50, "p95_ms": rp95, "context_has_previously": ctx["text"].as_str().unwrap_or("").contains("previously")}));

    // recall cost with chat facts in the same namespace (graph recall then reads its ontology facts)
    for s in ["My sister Anna lives in Porto.", "My brother Tom works at Initech.", "I moved to Lisbon last year."] {
        db.memory_call("remember", json!({"text": s, "entities": ["user"]})).unwrap();
    }
    let mut rl2 = vec![];
    for i in 0..50 {
        let a = 1 + (i * 37) % accounts;
        let tq = Instant::now();
        db.memory_call("recall", json!({"query": format!("What is the current plan of Account {a}?"), "k": 5, "record_access": false})).unwrap();
        rl2.push(tq.elapsed().as_secs_f64() * 1000.0);
    }
    let rq50 = pct(&mut rl2, 0.5);
    eprintln!("recall with chat facts in the namespace: p50 {rq50:.1} ms");
    report.insert("recall_with_chat_facts_p50_ms".into(), json!(rq50));

    // 5. write-back exactly once under injected crashes
    let k = 40;
    let fdir = dir.join("flows");
    let _ = std::fs::remove_dir_all(&fdir);
    std::fs::create_dir_all(&fdir).unwrap();
    let fpath = fdir.join("flow.db");
    let fdb = DbInstance::new("sqlite", &fpath, "").unwrap();
    kleos_pg::writeback::register_flow_action().unwrap();
    fdb.flow_call("define", json!({"name": "wb", "graph": {"steps": [
        {"name": "t", "pg": {"source": "crm", "sql": "INSERT INTO tickets (id, account_id, subject, status) VALUES ($1, 1, $2, 'open') RETURNING id", "params": ["$input.id", "$input.subject"]}},
        {"return": "$t"}]}})).unwrap();
    let exe = std::env::current_exe().unwrap();
    let mut crashes = 0;
    let mut wb_lat = vec![];
    for i in 0..k {
        let id = 50_000_000 + i;
        fdb.flow_call("start", json!({"workflow": "wb", "id": format!("wb{i}"), "input": {"id": id, "subject": format!("wb {i}")}})).unwrap();
        if i % 2 == 0 {
            // half of them: a worker dies after the Postgres commit (or, every fourth, before it)
            let mode = if i % 4 == 0 { "after_commit" } else { "before_commit" };
            let st = std::process::Command::new(&exe)
                .args(["--child-tick", fpath.to_str().unwrap(), "--url", &url])
                .env("KLEOS_PG_WB_CRASH", mode)
                .env("KLEOS_PG_TEST_URL", &base)
                .status()
                .unwrap();
            if !st.success() {
                crashes += 1;
            }
            std::thread::sleep(Duration::from_millis(250)); // the dead worker's lease
        }
        let tw = Instant::now();
        loop {
            fdb.flow_call("tick", json!({"worker": "rescuer", "max": 4})).unwrap();
            if fdb.flow_call("status", json!({"id": format!("wb{i}")})).unwrap()["status"] == "completed" {
                break;
            }
            assert!(tw.elapsed() < Duration::from_secs(60));
            std::thread::sleep(Duration::from_millis(20));
        }
        if i % 2 == 1 {
            wb_lat.push(tw.elapsed().as_secs_f64() * 1000.0);
        }
    }
    let written: i64 = pg_rows(&rt, &url, "SELECT count(*) FROM tickets WHERE id >= 50000000")[0][0].as_ref().unwrap().parse().unwrap();
    let dup: i64 = pg_rows(&rt, &url, "SELECT count(*) - count(DISTINCT subject) FROM tickets WHERE id >= 50000000")[0][0].as_ref().unwrap().parse().unwrap();
    let ledger: i64 = pg_rows(&rt, &url, "SELECT count(*) FROM kleos_writeback")[0][0].as_ref().unwrap().parse().unwrap();
    let wp50 = pct(&mut wb_lat, 0.5);
    eprintln!("write-back: {k} workflows, {crashes} crashed workers, {written} rows written ({dup} duplicates), ledger {ledger}; step p50 {wp50:.1} ms");
    report.insert("writeback".into(), json!({"workflows": k, "crashed_workers": crashes, "rows_written": written, "duplicates": dup,
        "ledger_rows": ledger, "exactly_once": written == k && dup == 0, "tick_to_complete_p50_ms": wp50}));
    // the echoes come back through CDC linked to their flow steps
    let t2 = Instant::now();
    loop {
        let e = kq(&db, "?[count(lsn)] := *kleos_pg_echo{src: 'crm', lsn}")[0][0].get_int().unwrap_or(0);
        if e >= k || t2.elapsed() > Duration::from_secs(60) {
            report.insert("echoes_linked".into(), json!(e));
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let st = kleos_pg::pg_call(&db, "status", json!({"source": "crm"})).unwrap();
    report.insert("status_lag".into(), st["lag"].clone());
    h.stop();
    std::thread::sleep(Duration::from_millis(500));
    println!("{}", serde_json::to_string_pretty(&Value::Object(report.into_iter().collect())).unwrap());
}
