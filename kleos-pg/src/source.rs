/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The CDC connector (docs/specs/pg-connector.md, "Ingest"): one logical
//! replication slot, an initial copy at the slot's exported snapshot, then
//! pgoutput streaming. Changes are applied in whole transactions by the
//! kleos writer (`sink.rs`) on a blocking thread; Postgres is acknowledged
//! only up to what kleos has committed, so a crash anywhere replays from
//! the last kleos checkpoint and the writer skips what it already has.
//!
//! Tables join later (new tables, `pg resync`, a lost slot) through a
//! temporary slot's snapshot: the table is copied as of that snapshot's
//! consistent point C, and the main stream's changes to it before C are
//! skipped (`skip_before`).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use futures::StreamExt;
use kleos::DbInstance;
use miette::{miette, Result};
use postgres_replication::protocol::{LogicalReplicationMessage, ReplicationMessage, Tuple, TupleData};
use postgres_replication::LogicalReplicationStream;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_postgres::types::PgLsn;
use tokio_postgres::Client;

use crate::admin;
use crate::catalog::{self, Catalog, TypeKind};
use crate::change::{Change, Echo, Op, Row, SchemaChange, Tx};
use crate::config::PgConfig;
use crate::conn::{self, fmt_lsn, ident, lit, parse_lsn, qident, simple_rows};
use crate::mapping::{self, Mapping};
use crate::sink::{Sink, Stats};
use crate::store::{self, Checkpoint, TableState};
use crate::value::{copy_fields, row_key, typed};

/// Microseconds between the Unix epoch and the Postgres epoch (2000-01-01).
pub const PG_EPOCH_US: i64 = 946_684_800_000_000;

/// What a running connector reports to `status`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Live {
    pub state: String,
    pub received_lsn: Option<String>,
    pub applied_lsn: Option<String>,
    pub acked_lsn: Option<String>,
    /// Commit time of the last applied transaction (µs).
    pub last_commit_ts: Option<i64>,
    pub last_error: Option<String>,
    pub started: i64,
    pub restarts: u64,
    /// Rows copied per table during a snapshot.
    pub snapshot: BTreeMap<String, i64>,
    /// Transactions in the queue to the kleos writer.
    pub queued: usize,
}

/// Test hooks: simulate a crash at a chosen point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailPoint {
    /// Stop with an error once `n` batches are committed to kleos, before
    /// Postgres is told (the replay must not duplicate them).
    AfterCommit(u32),
    /// Fail the `n`th batch's kleos write (nothing of it may remain).
    BeforeCommit(u32),
}

/// State shared between a connector, its writer thread and `status`.
#[derive(Default)]
pub struct Shared {
    pub live: Mutex<Live>,
    pub stats: Arc<Stats>,
    stop: AtomicBool,
    notify: tokio::sync::Notify,
    pub failpoint: Mutex<Option<FailPoint>>,
}

impl Shared {
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }
    pub fn stopping(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }
    fn set(&self, f: impl FnOnce(&mut Live)) {
        f(&mut self.live.lock().unwrap());
    }
}

/// Connectors running in this process, by source.
pub fn registry() -> &'static Mutex<BTreeMap<String, Arc<Shared>>> {
    static R: OnceLock<Mutex<BTreeMap<String, Arc<Shared>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(BTreeMap::new()))
}

enum Work {
    Txs(Vec<Tx>),
    Reconcile { table: String, seen: HashSet<String>, at: i64 },
    Fresh { table: String, fresh: bool },
    Barrier(tokio::sync::oneshot::Sender<()>),
}

enum Done {
    Applied { end_lsn: u64, lsn: u64, commit_ts: i64 },
    Error(String),
}

enum Exit {
    Stop,
    Resync(BTreeSet<String>),
}

/// A replicated table as the stream describes it.
#[derive(Debug, Clone)]
struct Rel {
    table: String,
    cols: Vec<(String, TypeKind)>,
    /// `included`: mapped; `ledger`: the write-back ledger; else ignored.
    role: RelRole,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum RelRole {
    Included,
    Ledger,
    Ignored,
}

pub struct Connector {
    pub cfg: Arc<PgConfig>,
    db: DbInstance,
    pub shared: Arc<Shared>,
}

impl Connector {
    pub fn new(cfg: PgConfig, db: DbInstance) -> Connector {
        let shared = Arc::new(Shared::default());
        registry().lock().unwrap().insert(cfg.source.clone(), shared.clone());
        Connector { cfg: Arc::new(cfg), db, shared }
    }

    /// Run until stopped, restarting after errors with back-off.
    pub async fn run(self) -> Result<()> {
        let mut delay = Duration::from_secs(1);
        self.shared.set(|l| l.started = store::now_us());
        loop {
            if self.shared.stopping() {
                break;
            }
            let t0 = Instant::now();
            match self.run_once().await {
                Ok(()) => break,
                Err(e) => {
                    let msg = format!("{e:?}");
                    self.shared.set(|l| {
                        l.state = "error".into();
                        l.last_error = Some(msg.clone());
                        l.restarts += 1;
                    });
                    let db = self.db.clone();
                    let src = self.cfg.source.clone();
                    let m2 = msg.clone();
                    let _ = tokio::task::spawn_blocking(move || store::log_event(&db, &src, "error", &m2)).await;
                    if t0.elapsed() > Duration::from_secs(60) {
                        delay = Duration::from_secs(1);
                    }
                    tokio::select! {
                        _ = tokio::time::sleep(delay) => {}
                        _ = self.shared.notify.notified() => {}
                    }
                    delay = (delay * 2).min(Duration::from_secs(30));
                }
            }
        }
        self.shared.set(|l| l.state = "stopped".into());
        Ok(())
    }

    /// One session: set up, copy what needs copying, stream. Returns on stop
    /// or error.
    pub async fn run_once(&self) -> Result<()> {
        let cfg = self.cfg.clone();
        let pgcfg = conn::parse(&cfg.connection)?;
        let pg = conn::connect(&pgcfg).await?;
        let info = admin::server_info(&pg).await?;
        let blockers = admin::blockers(&info);
        if !blockers.is_empty() {
            return Err(miette!("{}", blockers.join("; ")));
        }
        let version = info.version;
        self.shared.set(|l| l.state = "starting".into());
        let db = self.db.clone();
        let c2 = cfg.clone();
        blocking(move || {
            store::ensure(&db)?;
            store::put_source(&db, &c2)
        })
        .await?;
        for w in admin::warnings(&info) {
            self.log("warn", &w).await;
        }
        // the catalog and the mapping (re-inferred: DDL while stopped shows here)
        let c3 = cfg.clone();
        let cat = catalog::read_catalog(&pg, version, &move |q| c3.includes(q), true).await?;
        let inferred = mapping::infer(&cfg, &cat);
        let db = self.db.clone();
        let c4 = cfg.clone();
        let inf2 = inferred.clone();
        let (_version, mapping) = blocking(move || crate::ops::sync_mapping(&db, &c4, inf2, "catalog at start")).await?;
        let mapping = Arc::new(RwLock::new(mapping));
        // the write-back ledger, when flows may write
        if !cfg.writeback.allow.is_empty() {
            match admin::ensure_ledger(&pg, &cfg).await {
                Ok(true) => self.log("info", &format!("created the write-back ledger {}", cfg.writeback.ledger)).await,
                Ok(false) => {}
                Err(e) => self.log("warn", &format!("{e}")).await,
            }
        }
        // publication (tables + the ledger, so write-back echoes are seen)
        let mut pub_tables: Vec<String> = mapping.read().unwrap().tables.values().filter(|t| !t.is_skip()).map(|t| t.table.clone()).collect();
        if catalog::table_exists(&pg, &cfg.writeback.ledger).await? {
            pub_tables.push(cfg.writeback.ledger.clone());
        }
        for n in admin::ensure_publication(&pg, &cfg, &pub_tables, version).await? {
            self.log("info", &n).await;
        }
        // the writer
        let db = self.db.clone();
        let c5 = cfg.clone();
        let m2 = mapping.clone();
        let stats = self.shared.stats.clone();
        let mut sink = blocking(move || Sink::new(db, c5, m2, stats)).await?;
        let ckpt = sink.checkpoint().clone();
        let had_ckpt = {
            let db = self.db.clone();
            let src = cfg.source.clone();
            blocking(move || store::checkpoint(&db, &src)).await?.is_some()
        };
        let (work_tx, work_rx) = mpsc::channel::<Work>(cfg.batch.queue);
        let (done_tx, mut done_rx) = mpsc::unbounded_channel::<Done>();
        let slot = cfg.slot();
        let slot_now = admin::slot_info(&pg, &slot, version).await?;
        if let Some(si) = &slot_now {
            let here = simple_rows(&pg, "SELECT current_database()").await?.first().and_then(|r| r[0].clone());
            if si.database != here || si.plugin.as_deref() != Some("pgoutput") {
                return Err(miette!(
                    "replication slot {slot} belongs to database {:?} ({:?}), not this one; slot names are cluster-wide: set `slot:` in the config",
                    si.database,
                    si.plugin
                ));
            }
        }
        let states = {
            let db = self.db.clone();
            let src = cfg.source.clone();
            blocking(move || store::table_states(&db, &src)).await?
        };
        let tables: Vec<String> = topo_order(&mapping.read().unwrap());
        let mut repl = conn::connect_replication(&pgcfg).await?;
        let mut copy_plan: Option<(String, u64, bool, Vec<String>)> = None; // (snapshot, C, resync, tables)
        let mut need_resync: BTreeSet<String> = BTreeSet::new();
        let start_lsn: u64;
        match (had_ckpt, &slot_now) {
            (true, Some(si)) if !si.invalid() => {
                start_lsn = ckpt.end_lsn.max(ckpt.consistent);
                for t in &tables {
                    match states.get(t) {
                        Some(s) if s.state == "ready" => {}
                        _ => {
                            need_resync.insert(t.clone());
                        }
                    }
                }
            }
            (had, si) => {
                if let Some(si) = si {
                    if si.active {
                        return Err(miette!("slot {slot} is in use by another process"));
                    }
                    let why = if si.invalid() {
                        format!("slot {slot} was invalidated ({}); recreating it and resyncing every table", si.invalidation_reason.clone().or(si.wal_status.clone()).unwrap_or_default())
                    } else {
                        format!("slot {slot} exists without a kleos checkpoint; recreating it")
                    };
                    self.log("warn", &why).await;
                    admin::drop_slot(&pg, &slot).await?;
                } else if had {
                    self.log("warn", &format!("slot {slot} is gone (failover, idle-slot removal or a manual drop); recreating it and resyncing every table")).await;
                }
                let (c, snap) = create_slot(&repl, &slot, false, failover_wanted(&cfg, version), version, cfg.snapshot).await?;
                let mut ck = if had { ckpt.clone() } else { Checkpoint::default() };
                ck.consistent = c;
                ck.slot = slot.clone();
                let db = self.db.clone();
                let src = cfg.source.clone();
                let ck2 = ck.clone();
                blocking(move || store::put_checkpoint(&db, &src, &ck2)).await?;
                sink.reload_checkpoint().map_err(|e| miette!("{e}"))?;
                start_lsn = c;
                match snap {
                    Some(s) => copy_plan = Some((s, c, had, tables.clone())),
                    None => {
                        // snapshot: false: the tables start empty, as of now
                        for t in &tables {
                            self.set_state(t, "ready", c, 0, None).await?;
                        }
                    }
                }
            }
        }
        // the writer thread
        let shared = self.shared.clone();
        let writer = tokio::task::spawn_blocking(move || writer(sink, work_rx, done_tx, shared));
        let mut ctx = StreamCtx {
            cat,
            mapping: mapping.clone(),
            rels: HashMap::new(),
            skip_before: BTreeMap::new(),
            applied_lsn: ckpt.lsn,
            applied_end: ckpt.end_lsn,
            pending_schema: vec![],
            resync: BTreeSet::new(),
        };
        if let Some((snap, c, resync, ts)) = copy_plan {
            self.shared.set(|l| l.state = "snapshot".into());
            self.copy_tables(&pgcfg, &snap, &ts, resync, c, &ctx, &work_tx).await?;
            drain(&mut done_rx)?;
            for t in &ts {
                ctx.skip_before.insert(t.clone(), c);
            }
        }
        let states = {
            let db = self.db.clone();
            let src = cfg.source.clone();
            blocking(move || store::table_states(&db, &src)).await?
        };
        for (t, s) in &states {
            ctx.skip_before.entry(t.clone()).or_insert(s.skip_before);
        }
        let mut start = start_lsn;
        loop {
            if !need_resync.is_empty() {
                drop(repl);
                self.resync(&pgcfg, &need_resync, &mut ctx, &work_tx).await?;
                drain(&mut done_rx)?;
                need_resync.clear();
                repl = conn::connect_replication(&pgcfg).await?;
            }
            let heartbeat = (cfg.heartbeat_secs > 0).then(|| {
                let pgcfg = pgcfg.clone();
                let secs = cfg.heartbeat_secs;
                tokio::spawn(async move {
                    if let Ok(c) = conn::connect(&pgcfg).await {
                        loop {
                            tokio::time::sleep(Duration::from_secs(secs)).await;
                            let _ = c.simple_query("SELECT pg_logical_emit_message(false, 'kleos_heartbeat', now()::text)").await;
                        }
                    }
                })
            });
            self.shared.set(|l| l.state = "streaming".into());
            let r = self.stream(&repl, start, &mut ctx, &work_tx, &mut done_rx, &pg).await;
            if let Some(h) = heartbeat {
                h.abort();
            }
            match r? {
                Exit::Stop => break,
                Exit::Resync(ts) => {
                    need_resync = ts;
                    start = ctx.applied_end.max(start);
                    drop(repl);
                    repl = conn::connect_replication(&pgcfg).await?;
                }
            }
        }
        drop(work_tx);
        let _ = writer.await;
        Ok(())
    }

    async fn log(&self, level: &str, msg: &str) {
        let db = self.db.clone();
        let src = self.cfg.source.clone();
        let (l, m) = (level.to_string(), msg.to_string());
        let _ = tokio::task::spawn_blocking(move || store::log_event(&db, &src, &l, &m)).await;
    }

    async fn set_state(&self, table: &str, state: &str, skip_before: u64, rows: i64, info: Option<serde_json::Value>) -> Result<()> {
        let db = self.db.clone();
        let src = self.cfg.source.clone();
        let t = TableState { tbl: table.to_string(), state: state.to_string(), skip_before, rows, info, error: None, updated: 0 };
        blocking(move || store::put_table_state(&db, &src, &t)).await
    }

    /// Copy `tables` at an exported snapshot (consistent point `c`). With
    /// `resync`, rows kleos has but the snapshot lacks are retracted.
    #[allow(clippy::too_many_arguments)]
    async fn copy_tables(
        &self,
        pgcfg: &tokio_postgres::Config,
        snapshot: &str,
        tables: &[String],
        resync: bool,
        c: u64,
        ctx: &StreamCtx,
        work: &mpsc::Sender<Work>,
    ) -> Result<()> {
        let pg = conn::connect(pgcfg).await?;
        pg.batch_execute(&format!("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY; SET TRANSACTION SNAPSHOT {}", lit(snapshot)))
            .await
            .map_err(conn::pg_err)?;
        let ts = simple_rows(&pg, "SELECT (extract(epoch from now()) * 1000000)::int8").await?;
        let snap_ts: i64 = ts.first().and_then(|r| r[0].clone()).and_then(|v| v.parse().ok()).unwrap_or_else(store::now_us);
        let mapping = ctx.mapping.read().unwrap().clone();
        for t in tables {
            let Some(tm) = mapping.tables.get(t).cloned() else { continue };
            if tm.is_skip() {
                self.set_state(t, "skipped", c, 0, None).await?;
                continue;
            }
            let Some(info) = ctx.cat.tables.get(t).cloned() else { continue };
            self.set_state(t, "copying", c, 0, serde_json::to_value(&info).ok()).await?;
            let has = {
                let db = self.db.clone();
                let src = self.cfg.source.clone();
                let t2 = t.clone();
                blocking(move || {
                    let r = store::read(&db, &format!("?[key] := *{}{{src: $src, tbl: $tbl, key}} :limit 1", store::ROW), vec![("src", store::s(&src)), ("tbl", store::s(&t2))])?;
                    Ok(!r.rows.is_empty())
                })
                .await?
            };
            let fresh = !has;
            send(work, Work::Fresh { table: t.clone(), fresh }).await?;
            let cols: Vec<(String, TypeKind)> = info.columns.iter().filter(|col| tm.kept(&col.name)).map(|col| (col.name.clone(), col.kind)).collect();
            let sel = cols.iter().map(|(n, _)| ident(n)).collect::<Vec<_>>().join(", ");
            let stream = pg
                .copy_out(&format!("COPY (SELECT {sel} FROM {}) TO STDOUT", qident(t)))
                .await
                .map_err(conn::pg_err)?;
            futures::pin_mut!(stream);
            let mut buf: Vec<u8> = vec![];
            let mut batch: Vec<Change> = vec![];
            let mut seen: HashSet<String> = HashSet::new();
            let mut n: i64 = 0;
            let key_cols = tm.key.clone();
            let flush = |batch: &mut Vec<Change>| -> Option<Tx> {
                if batch.is_empty() {
                    return None;
                }
                Some(Tx { snapshot: true, commit_ts: snap_ts, changes: std::mem::take(batch), ..Default::default() })
            };
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(conn::pg_err)?;
                buf.extend_from_slice(&chunk);
                while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=pos).collect();
                    let fields = copy_fields(&line[..line.len() - 1]);
                    let mut row = Row::new();
                    for ((name, kind), f) in cols.iter().zip(fields) {
                        row.insert(name.clone(), f.map(|v| typed(*kind, &v)).unwrap_or(serde_json::Value::Null));
                    }
                    if resync {
                        let k: Vec<serde_json::Value> = key_cols.iter().map(|k| row.get(k).cloned().unwrap_or_default()).collect();
                        seen.insert(row_key(&k));
                    }
                    batch.push(Change { table: t.clone(), op: Op::Read, new: Some(row), old: None, missing: vec![] });
                    n += 1;
                    if batch.len() >= self.cfg.batch.snapshot_rows {
                        if let Some(tx) = flush(&mut batch) {
                            send(work, Work::Txs(vec![tx])).await?;
                        }
                        let t3 = t.clone();
                        self.shared.set(|l| {
                            l.snapshot.insert(t3, n);
                        });
                    }
                }
            }
            if let Some(tx) = flush(&mut batch) {
                send(work, Work::Txs(vec![tx])).await?;
            }
            if resync {
                send(work, Work::Reconcile { table: t.clone(), seen, at: snap_ts }).await?;
            }
            barrier(work).await?;
            send(work, Work::Fresh { table: t.clone(), fresh: false }).await?;
            let t3 = t.clone();
            self.shared.set(|l| {
                l.snapshot.insert(t3, n);
            });
            self.set_state(t, "ready", c, n, serde_json::to_value(&info).ok()).await?;
            log::info!("pg {}: copied {t} ({n} rows{})", self.cfg.source, if resync { ", resync" } else { "" });
        }
        let _ = pg.batch_execute("COMMIT").await;
        Ok(())
    }

    /// Copy `tables` again through a temporary slot's snapshot, while the
    /// main slot keeps its place.
    async fn resync(&self, pgcfg: &tokio_postgres::Config, tables: &BTreeSet<String>, ctx: &mut StreamCtx, work: &mpsc::Sender<Work>) -> Result<()> {
        let pg = conn::connect(pgcfg).await?;
        let version = conn::server_version(&pg).await?;
        // refresh the catalog entries (new tables, DDL)
        let wanted: BTreeSet<String> = tables.clone();
        let fresh_cat = catalog::read_catalog(&pg, version, &move |q| wanted.contains(q), true).await?;
        for (q, info) in fresh_cat.tables {
            let tm = mapping::infer_table(&self.cfg, &ctx.cat, &info);
            ctx.cat.tables.insert(q.clone(), info);
            let db = self.db.clone();
            let cfg = self.cfg.clone();
            let m = ctx.mapping.clone();
            blocking(move || crate::ops::install_table(&db, &cfg, &m, &tm, "resync")).await?;
        }
        let repl = conn::connect_replication(pgcfg).await?;
        let tmp = format!("{}_rs_{:x}", self.cfg.slot(), store::now_us() & 0xffffff);
        let (c, snap) = create_slot(&repl, &tmp, true, false, version, true).await?;
        let snap = snap.ok_or_else(|| miette!("no exported snapshot"))?;
        self.log("info", &format!("resync of {} table(s) at {}: {:?}", tables.len(), fmt_lsn(c), tables)).await;
        self.shared.set(|l| l.state = "resync".into());
        let list: Vec<String> = topo_order(&ctx.mapping.read().unwrap()).into_iter().filter(|t| tables.contains(t)).collect();
        self.copy_tables(pgcfg, &snap, &list, true, c, ctx, work).await?;
        for t in &list {
            ctx.skip_before.insert(t.clone(), c);
        }
        drop(repl); // the temporary slot goes with its session
        Ok(())
    }

    /// Stream from `start` until stopped or a resync is needed.
    async fn stream(
        &self,
        repl: &Client,
        start: u64,
        ctx: &mut StreamCtx,
        work: &mpsc::Sender<Work>,
        done: &mut mpsc::UnboundedReceiver<Done>,
        pg: &Client,
    ) -> Result<Exit> {
        let cfg = self.cfg.clone();
        let opts = format!("(\"proto_version\" '1', \"publication_names\" {}, \"messages\" 'true')", lit(&ident(&cfg.publication())));
        let q = format!("START_REPLICATION SLOT {} LOGICAL {} {}", ident(&cfg.slot()), fmt_lsn(start), opts);
        let duplex = repl.copy_both_simple::<bytes::Bytes>(&q).await.map_err(conn::pg_err)?;
        let stream = LogicalReplicationStream::new(duplex);
        tokio::pin!(stream);
        let mut flushed: u64 = start.max(ctx.applied_end);
        let mut last_seen: u64 = flushed;
        let mut received: u64 = flushed;
        let mut acked_sent: u64 = 0;
        let mut batch: Vec<Tx> = vec![];
        let mut batch_changes = 0usize;
        let mut batch_started: Option<Instant> = None;
        let mut inflight: usize = 0;
        let mut cur: Option<Tx> = None;
        let mut parts: u32 = 0;
        let mut last_status = Instant::now();
        let mut last_poll = Instant::now();
        let mut last_tables = Instant::now();
        let mut committed_batches: u32 = 0;
        let mut last_lag_check = Instant::now();
        let mut last_lag_alert: Option<Instant> = None;
        let mut exit: Option<Exit> = None;
        let mut tick = tokio::time::interval(Duration::from_millis(50));
        let ledger = cfg.writeback.ledger.clone();
        loop {
            // leaving: drain the batch and wait for the writer before the last ack
            let draining = exit.is_some();
            if draining && cur.is_none() && batch.is_empty() && inflight == 0 {
                send_status(stream.as_mut(), received.max(flushed), flushed).await?;
                return Ok(exit.unwrap());
            }
            // group commit: the batch goes to the writer when it is full, old
            // enough, or we are leaving; while the writer is busy it keeps
            // growing (up to max_changes), so a slow commit carries more work
            let full = batch_changes >= cfg.batch.max_changes;
            let due = !batch.is_empty()
                && (full || draining || batch_started.map_or(false, |t| t.elapsed() >= Duration::from_millis(cfg.batch.max_ms)));
            // read while the batch has room (and always finish a transaction)
            let read = (!full || cur.is_some()) && !(draining && cur.is_none());
            tokio::select! {
                biased;
                d = done.recv() => match d {
                    Some(Done::Applied { end_lsn, lsn, commit_ts }) => {
                        inflight = inflight.saturating_sub(1);
                        if end_lsn > flushed {
                            flushed = end_lsn;
                        }
                        ctx.applied_lsn = ctx.applied_lsn.max(lsn);
                        ctx.applied_end = ctx.applied_end.max(end_lsn);
                        committed_batches += 1;
                        self.shared.set(|l| {
                            l.applied_lsn = Some(fmt_lsn(lsn));
                            if commit_ts > 0 { l.last_commit_ts = Some(commit_ts); }
                            l.queued = inflight;
                        });
                        if let Some(FailPoint::AfterCommit(n)) = *self.shared.failpoint.lock().unwrap() {
                            if committed_batches >= n {
                                return Err(miette!("failpoint: stopped after {n} committed batches, before acknowledging them"));
                            }
                        }
                    }
                    Some(Done::Error(e)) => return Err(miette!("{e}")),
                    None => return Err(miette!("the kleos writer stopped")),
                },
                permit = work.reserve(), if due => {
                    let permit = permit.map_err(|_| miette!("the kleos writer stopped"))?;
                    permit.send(Work::Txs(std::mem::take(&mut batch)));
                    batch_changes = 0;
                    batch_started = None;
                    inflight += 1;
                }
                msg = stream.next(), if read => {
                    let Some(msg) = msg else { return Err(miette!("replication stream ended")) };
                    match msg.map_err(conn::pg_err)? {
                        ReplicationMessage::XLogData(body) => {
                            let wal_end = body.wal_end();
                            match body.into_data() {
                                LogicalReplicationMessage::Begin(b) => {
                                    let mut tx = Tx { lsn: b.final_lsn(), commit_ts: b.timestamp() + PG_EPOCH_US, xid: b.xid(), ..Default::default() };
                                    tx.schema.append(&mut ctx.pending_schema);
                                    cur = Some(tx);
                                    parts = 0;
                                }
                                LogicalReplicationMessage::Commit(c) => {
                                    let Some(mut tx) = cur.take() else { continue };
                                    tx.end_lsn = c.end_lsn();
                                    received = received.max(c.end_lsn());
                                    if parts > 0 {
                                        tx.part = Some((parts, true));
                                    }
                                    if tx.lsn <= ctx.applied_lsn {
                                        // already in kleos (a replay after a crash)
                                        self.shared.stats.skipped.fetch_add(1, Ordering::Relaxed);
                                    } else if tx.is_empty() && tx.echo.is_none() && parts == 0 {
                                        last_seen = last_seen.max(c.end_lsn());
                                    } else {
                                        batch_changes += tx.len().max(1);
                                        batch.push(tx);
                                        batch_started.get_or_insert_with(Instant::now);
                                    }
                                    self.shared.set(|l| l.received_lsn = Some(fmt_lsn(received)));
                                }
                                LogicalReplicationMessage::Relation(r) => {
                                    let table = format!("{}.{}", r.namespace().unwrap_or(""), r.name().unwrap_or(""));
                                    let cols: Vec<(String, u32)> = r.columns().iter().map(|c| (c.name().unwrap_or("").to_string(), c.type_id() as u32)).collect();
                                    self.on_relation(r.rel_id(), &table, cols, ctx, cur.as_mut(), pg, &ledger).await?;
                                }
                                LogicalReplicationMessage::Insert(m) => {
                                    if let (Some(rel), Some(tx)) = (ctx.rels.get(&m.rel_id()).cloned(), cur.as_mut()) {
                                        let (row, missing) = decode(&rel, m.tuple());
                                        match rel.role {
                                            RelRole::Ledger => tx.echo = Some(echo_of(&row)),
                                            RelRole::Included if !ctx.skipped(&rel.table, tx.lsn) => {
                                                tx.changes.push(Change { table: rel.table.clone(), op: Op::Insert, new: Some(row), old: None, missing });
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                LogicalReplicationMessage::Update(m) => {
                                    if let (Some(rel), Some(tx)) = (ctx.rels.get(&m.rel_id()).cloned(), cur.as_mut()) {
                                        if rel.role == RelRole::Included && !ctx.skipped(&rel.table, tx.lsn) {
                                            let (row, missing) = decode(&rel, m.new_tuple());
                                            let old = m.key_tuple().or(m.old_tuple()).map(|t| decode(&rel, t).0);
                                            tx.changes.push(Change { table: rel.table.clone(), op: Op::Update, new: Some(row), old, missing });
                                        }
                                    }
                                }
                                LogicalReplicationMessage::Delete(m) => {
                                    if let (Some(rel), Some(tx)) = (ctx.rels.get(&m.rel_id()).cloned(), cur.as_mut()) {
                                        if rel.role == RelRole::Included && !ctx.skipped(&rel.table, tx.lsn) {
                                            let old = m.key_tuple().or(m.old_tuple()).map(|t| decode(&rel, t).0);
                                            tx.changes.push(Change { table: rel.table.clone(), op: Op::Delete, new: None, old, missing: vec![] });
                                        }
                                    }
                                }
                                LogicalReplicationMessage::Truncate(m) => {
                                    if let Some(tx) = cur.as_mut() {
                                        for id in m.rel_ids() {
                                            if let Some(rel) = ctx.rels.get(id) {
                                                if rel.role == RelRole::Included && !ctx.skipped(&rel.table, tx.lsn) {
                                                    tx.changes.push(Change { table: rel.table.clone(), op: Op::Truncate, new: None, old: None, missing: vec![] });
                                                }
                                            }
                                        }
                                    }
                                }
                                LogicalReplicationMessage::Message(_) => {
                                    if cur.is_none() {
                                        last_seen = last_seen.max(wal_end);
                                    }
                                }
                                _ => {}
                            }
                            // a huge transaction goes in parts (replay-safe)
                            if let Some(tx) = cur.as_mut() {
                                if tx.changes.len() >= cfg.batch.max_tx_changes {
                                    parts += 1;
                                    let part = Tx {
                                        lsn: tx.lsn,
                                        end_lsn: 0,
                                        commit_ts: tx.commit_ts,
                                        xid: tx.xid,
                                        changes: std::mem::take(&mut tx.changes),
                                        snapshot: false,
                                        echo: tx.echo.clone(),
                                        schema: std::mem::take(&mut tx.schema),
                                        part: Some((parts, false)),
                                    };
                                    if tx.lsn > ctx.applied_lsn {
                                        // a part fills the batch: it goes as soon as the writer is free
                                        batch_changes += part.len().max(cfg.batch.max_changes);
                                        batch.push(part);
                                        batch_started.get_or_insert_with(Instant::now);
                                    }
                                }
                            }
                        }
                        ReplicationMessage::PrimaryKeepAlive(k) => {
                            if cur.is_none() {
                                last_seen = last_seen.max(k.wal_end());
                            }
                            if k.reply() == 1 {
                                let ack = if cur.is_none() && batch.is_empty() && inflight == 0 { flushed.max(last_seen) } else { flushed };
                                send_status(stream.as_mut(), received.max(ack), ack).await?;
                                acked_sent = ack;
                                last_status = Instant::now();
                            }
                        }
                        _ => {}
                    }
                }
                _ = tick.tick() => {
                    // idle: nothing of ours is in flight, so WAL seen so far is done
                    let idle = cur.is_none() && batch.is_empty() && inflight == 0;
                    let ack = if idle { flushed.max(last_seen) } else { flushed };
                    if last_status.elapsed() >= Duration::from_secs(cfg.status_secs.max(1)) || (ack > acked_sent && last_status.elapsed() >= Duration::from_millis(500)) {
                        send_status(stream.as_mut(), received.max(ack), ack).await?;
                        acked_sent = ack;
                        last_status = Instant::now();
                        self.shared.set(|l| l.acked_lsn = Some(fmt_lsn(ack)));
                    }
                    // lag alerts: checked every 30 s, logged at most every 5 minutes
                    if last_lag_check.elapsed() >= Duration::from_secs(30) {
                        last_lag_check = Instant::now();
                        if let Ok(v) = conn::server_version(pg).await {
                            if let Ok(Some(si)) = admin::slot_info(pg, &cfg.slot(), v).await {
                                let bytes = si.lag_bytes.unwrap_or(0);
                                let last = self.shared.live.lock().unwrap().last_commit_ts;
                                let secs = match last {
                                    Some(t) if bytes > 0 => (store::now_us() - t) / 1_000_000,
                                    _ => 0,
                                };
                                if (bytes > cfg.lag_alert.bytes || secs > cfg.lag_alert.seconds)
                                    && last_lag_alert.map_or(true, |t| t.elapsed() >= Duration::from_secs(300))
                                {
                                    last_lag_alert = Some(Instant::now());
                                    self.log("warn", &format!("replication lag {bytes} bytes, {secs}s (alert thresholds {} bytes, {}s)", cfg.lag_alert.bytes, cfg.lag_alert.seconds)).await;
                                }
                            }
                        }
                    }
                    if exit.is_none() {
                        if self.shared.stopping() {
                            exit = Some(Exit::Stop);
                        } else if !ctx.resync.is_empty() {
                            exit = Some(Exit::Resync(std::mem::take(&mut ctx.resync)));
                        } else if last_poll.elapsed() >= Duration::from_secs(3) {
                            last_poll = Instant::now();
                            let db = self.db.clone();
                            let src = cfg.source.clone();
                            let st = blocking(move || store::table_states(&db, &src)).await?;
                            let asked: BTreeSet<String> = st.values().filter(|s| s.state == "resync").map(|s| s.tbl.clone()).collect();
                            if !asked.is_empty() {
                                exit = Some(Exit::Resync(asked));
                            }
                        }
                        if exit.is_none() && cfg.new_tables_secs > 0 && cfg.publication_mode == "tables" && last_tables.elapsed() >= Duration::from_secs(cfg.new_tables_secs) {
                            last_tables = Instant::now();
                            let added = self.pick_up_new_tables(pg, ctx).await?;
                            if !added.is_empty() {
                                exit = Some(Exit::Resync(added));
                            }
                        }
                    }
                }
            }
        }
    }

    /// A Relation message: a table's (current) columns. A change from what
    /// the catalog said is DDL; an included table never seen is new and is
    /// copied by a resync.
    #[allow(clippy::too_many_arguments)]
    async fn on_relation(&self, rel_id: u32, table: &str, cols: Vec<(String, u32)>, ctx: &mut StreamCtx, cur: Option<&mut Tx>, pg: &Client, ledger: &str) -> Result<()> {
        if table == ledger {
            ctx.rels.insert(rel_id, Rel { table: table.into(), cols: cols.iter().map(|(n, _)| (n.clone(), TypeKind::Text)).collect(), role: RelRole::Ledger });
            return Ok(());
        }
        if !self.cfg.includes(table) {
            ctx.rels.insert(rel_id, Rel { table: table.into(), cols: vec![], role: RelRole::Ignored });
            return Ok(());
        }
        let known = ctx.cat.tables.get(table).cloned();
        let changed = match &known {
            None => true,
            Some(info) => {
                info.columns.len() != cols.len()
                    || info.columns.iter().zip(&cols).any(|(c, (n, oid))| &c.name != n || (c.atttypid != 0 && c.atttypid != *oid))
            }
        };
        if changed {
            let version = conn::server_version(pg).await?;
            let t = table.to_string();
            let fresh = catalog::read_catalog(pg, version, &move |q| q == t, true).await?;
            if let Some(info) = fresh.tables.get(table).cloned() {
                let map = mapping::infer_table(&self.cfg, &ctx.cat, &info);
                let dropped: Vec<String> = known
                    .as_ref()
                    .map(|k| k.columns.iter().filter(|c| !info.columns.iter().any(|n| n.name == c.name)).map(|c| c.name.clone()).collect())
                    .unwrap_or_default();
                let sc = SchemaChange { table: table.to_string(), info: info.clone(), map, dropped };
                let msg = match &known {
                    None => format!("new table {table}: mapped as {}", sc.map.class),
                    Some(_) => format!("schema of {table} changed (dropped {:?})", sc.dropped),
                };
                self.log("info", &msg).await;
                let added_default = known.as_ref().map_or(false, |k| {
                    info.columns.iter().any(|c| c.has_missing && !k.columns.iter().any(|o| o.name == c.name))
                });
                if known.is_none() || added_default {
                    // existing rows come from a resync (a new table, or a column
                    // added with a default, which rewrote no row); changes until
                    // then are skipped
                    ctx.skip_before.insert(table.to_string(), u64::MAX);
                    ctx.resync.insert(table.to_string());
                }
                ctx.cat.tables.insert(table.to_string(), info);
                match cur {
                    Some(tx) => tx.schema.push(sc),
                    None => ctx.pending_schema.push(sc),
                }
            }
        }
        let info = ctx.cat.tables.get(table);
        let rcols = cols
            .iter()
            .map(|(n, oid)| {
                let kind = info.and_then(|i| i.column(n)).map(|c| c.kind).unwrap_or_else(|| TypeKind::of(*oid, ""));
                (n.clone(), kind)
            })
            .collect();
        ctx.rels.insert(rel_id, Rel { table: table.into(), cols: rcols, role: RelRole::Included });
        Ok(())
    }

    /// Included tables missing from the publication: added (they are then
    /// copied by a resync).
    async fn pick_up_new_tables(&self, pg: &Client, ctx: &mut StreamCtx) -> Result<BTreeSet<String>> {
        let rows = simple_rows(
            pg,
            "SELECT n.nspname || '.' || c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace \
             WHERE c.relkind IN ('r', 'p') AND NOT c.relispartition AND n.nspname NOT IN ('pg_catalog', 'information_schema')",
        )
        .await?;
        let have = admin::publication_tables(pg, &self.cfg.publication()).await?.unwrap_or_default();
        let new: Vec<String> = rows
            .into_iter()
            .filter_map(|r| r[0].clone())
            .filter(|t| self.cfg.includes(t) && !have.contains(t) && !ctx.cat.tables.contains_key(t))
            .collect();
        if new.is_empty() || !self.cfg.create_publication {
            return Ok(BTreeSet::new());
        }
        let ddl = format!("ALTER PUBLICATION {} ADD TABLE {}", ident(&self.cfg.publication()), new.iter().map(|t| qident(t)).collect::<Vec<_>>().join(", "));
        pg.batch_execute(&ddl).await.map_err(conn::pg_err)?;
        self.log("info", &format!("new tables added to the publication: {new:?}")).await;
        for t in &new {
            ctx.skip_before.insert(t.clone(), u64::MAX);
        }
        Ok(new.into_iter().collect())
    }
}

struct StreamCtx {
    cat: Catalog,
    mapping: Arc<RwLock<Mapping>>,
    rels: HashMap<u32, Rel>,
    skip_before: BTreeMap<String, u64>,
    applied_lsn: u64,
    applied_end: u64,
    pending_schema: Vec<SchemaChange>,
    resync: BTreeSet<String>,
}

impl StreamCtx {
    fn skipped(&self, table: &str, lsn: u64) -> bool {
        self.skip_before.get(table).map_or(false, |b| lsn < *b)
    }
}

fn decode(rel: &Rel, t: &Tuple) -> (Row, Vec<String>) {
    let mut row = Row::new();
    let mut missing = vec![];
    for ((name, kind), d) in rel.cols.iter().zip(t.tuple_data()) {
        match d {
            TupleData::Null => {
                row.insert(name.clone(), serde_json::Value::Null);
            }
            TupleData::UnchangedToast => missing.push(name.clone()),
            TupleData::Text(b) => {
                row.insert(name.clone(), typed(*kind, &String::from_utf8_lossy(b)));
            }
            _ => {}
        }
    }
    (row, missing)
}

fn echo_of(row: &Row) -> Echo {
    let g = |k: &str| row.get(k).and_then(|v| v.as_str()).map(|s| s.to_string());
    Echo { idem_key: g("idem_key").unwrap_or_default(), workflow: g("workflow"), step: g("step"), actor: g("actor") }
}

async fn send_status(stream: std::pin::Pin<&mut LogicalReplicationStream>, write: u64, flush: u64) -> Result<()> {
    let now = store::now_us() - PG_EPOCH_US;
    stream
        .standby_status_update(PgLsn::from(write), PgLsn::from(flush), PgLsn::from(flush), now, 0)
        .await
        .map_err(conn::pg_err)
}

/// Consume the writer's reports of snapshot batches (all in before the
/// copy's last barrier), so the stream's in-flight count starts at zero: an
/// idle acknowledgement must never pass a batch still being written.
fn drain(done: &mut mpsc::UnboundedReceiver<Done>) -> Result<()> {
    while let Ok(d) = done.try_recv() {
        if let Done::Error(e) = d {
            return Err(miette!("{e}"));
        }
    }
    Ok(())
}

async fn send(work: &mpsc::Sender<Work>, w: Work) -> Result<()> {
    work.send(w).await.map_err(|_| miette!("the kleos writer stopped"))
}

async fn barrier(work: &mpsc::Sender<Work>) -> Result<()> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    send(work, Work::Barrier(tx)).await?;
    rx.await.map_err(|_| miette!("the kleos writer stopped"))
}

/// Run blocking kleos work off the async threads.
pub async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T> + Send + 'static) -> Result<T> {
    tokio::task::spawn_blocking(f).await.map_err(|e| miette!("kleos task panicked: {e}"))?
}

fn failover_wanted(cfg: &PgConfig, version: i32) -> bool {
    match cfg.failover.as_str() {
        "true" => true,
        "false" => false,
        _ => version >= 170000,
    }
}

/// Create a logical slot; with `export`, keep its snapshot for a copy.
/// Returns (consistent point, snapshot name).
async fn create_slot(repl: &Client, slot: &str, temporary: bool, failover: bool, version: i32, export: bool) -> Result<(u64, Option<String>)> {
    let temp = if temporary { " TEMPORARY" } else { "" };
    let q = if version >= 150000 {
        let mut opts = vec![format!("SNAPSHOT '{}'", if export { "export" } else { "nothing" })];
        if failover && version >= 170000 && !temporary {
            opts.push("FAILOVER true".into());
        }
        format!("CREATE_REPLICATION_SLOT {}{temp} LOGICAL pgoutput ({})", ident(slot), opts.join(", "))
    } else {
        format!("CREATE_REPLICATION_SLOT {}{temp} LOGICAL pgoutput {}", ident(slot), if export { "EXPORT_SNAPSHOT" } else { "NOEXPORT_SNAPSHOT" })
    };
    let rows = simple_rows(repl, &q).await?;
    let r = rows.first().ok_or_else(|| miette!("CREATE_REPLICATION_SLOT returned nothing"))?;
    let c = r.get(1).cloned().flatten().and_then(|s| parse_lsn(&s)).ok_or_else(|| miette!("bad consistent point"))?;
    Ok((c, if export { r.get(2).cloned().flatten() } else { None }))
}

/// Tables in copy order: referenced tables first, so labels of foreign-key
/// targets are known when rows are rendered.
pub fn topo_order(m: &Mapping) -> Vec<String> {
    let mut out = vec![];
    let mut seen = BTreeSet::new();
    fn visit(m: &Mapping, t: &str, seen: &mut BTreeSet<String>, out: &mut Vec<String>, depth: usize) {
        if !seen.insert(t.to_string()) || depth > 64 {
            return;
        }
        if let Some(tm) = m.tables.get(t) {
            for f in &tm.fks {
                if m.tables.contains_key(&f.table) {
                    visit(m, &f.table, seen, out, depth + 1);
                }
            }
            if let crate::mapping::Shape::Edge { source, destination, .. } = &tm.shape {
                for e in [source, destination] {
                    if m.tables.contains_key(&e.table) {
                        visit(m, &e.table, seen, out, depth + 1);
                    }
                }
            }
            out.push(t.to_string());
        }
    }
    for t in m.tables.keys() {
        visit(m, t, &mut seen, &mut out, 0);
    }
    out
}

fn writer(mut sink: Sink, mut rx: mpsc::Receiver<Work>, done: mpsc::UnboundedSender<Done>, shared: Arc<Shared>) {
    let mut n: u32 = 0;
    while let Some(w) = rx.blocking_recv() {
        match w {
            Work::Txs(txs) => {
                n += 1;
                if let Some(FailPoint::BeforeCommit(k)) = *shared.failpoint.lock().unwrap() {
                    if n >= k {
                        let _ = done.send(Done::Error(format!("failpoint: batch {n} failed before its kleos commit")));
                        return;
                    }
                }
                let commit_ts = txs.iter().filter(|t| !t.snapshot).map(|t| t.commit_ts).max().unwrap_or(0);
                match sink.apply(txs, vec![]) {
                    Ok(a) => {
                        let _ = done.send(Done::Applied { end_lsn: a.end_lsn.max(sink.checkpoint().end_lsn), lsn: sink.checkpoint().lsn, commit_ts });
                    }
                    Err(e) => {
                        let _ = done.send(Done::Error(format!("{e:?}")));
                        return;
                    }
                }
            }
            Work::Reconcile { table, seen, at } => {
                let r = sink.stale_rows(&table, &seen).and_then(|stale| {
                    if stale.is_empty() {
                        return Ok(());
                    }
                    log::info!("pg: resync of {table} retracts {} row(s) no longer in Postgres", stale.len());
                    let changes = stale.into_iter().map(|r| Change { table: table.clone(), op: Op::Delete, new: None, old: Some(r), missing: vec![] }).collect();
                    sink.apply(vec![Tx { snapshot: true, commit_ts: at, changes, ..Default::default() }], vec![]).map(|_| ())
                });
                if let Err(e) = r {
                    let _ = done.send(Done::Error(format!("{e:?}")));
                    return;
                }
            }
            Work::Fresh { table, fresh } => {
                if fresh {
                    sink.fresh.insert(table);
                } else {
                    sink.fresh.remove(&table);
                }
            }
            Work::Barrier(tx) => {
                let _ = tx.send(());
            }
        }
    }
}
