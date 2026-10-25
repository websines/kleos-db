/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The kleos writer (docs/specs/pg-connector.md, "Writing to kleos"): a batch
//! of whole source transactions becomes one kleos transaction holding the
//! entities' facts, nodes and memories, the stored row images, and the
//! checkpoint that says the batch is applied.
//!
//! - insert: assert at the commit time (`vt`);
//! - update: the changed columns only: the old value's fact is retracted and
//!   the new one asserted at `vt` (supersession); unchanged TOAST values are
//!   taken from the stored row;
//! - delete: every current fact, node and memory of the entity is retracted;
//! - truncate: every entity of the table is retracted;
//! - snapshot rows: asserted at the snapshot time, `by` `pg:<source>:snapshot`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, RwLock};

use kleos::{DataValue, DbInstance, Validity};
use miette::{miette, Result};
use serde_json::{json, Map, Value};

use crate::change::{Change, Op, Row, Tx};
use crate::config::PgConfig;
use crate::mapping::{self, ColKind, Mapping, Shape, TableMap};
use crate::store::{self, i, j, s, Checkpoint, CHECKPOINT, ECHO, OFFSET, ROW};
use crate::value::{display, entity_key, literal_o, row_key, short_hash};

const FACT: &str = "kleos_onto_fact";
const NODE: &str = "kleos_onto_node";
const MEM: &str = "kleos_mem";
const MEM_VEC: &str = "kleos_mem_vec";
const MEM_ROLE: &str = "kleos_mem_role";
const MEM_TYPE: &str = "kleos_mem_type";
const MEM_LINK: &str = "kleos_mem_link";
const MEM_ENTITY: &str = "kleos_mem_entity";
const MEM_MENTION: &str = "kleos_mem_mention";

fn vt(at: i64, assert: bool) -> DataValue {
    DataValue::Validity(Validity::from((at, assert)))
}

/// Counters shared with `status`.
#[derive(Debug, Default)]
pub struct Stats {
    pub txs: std::sync::atomic::AtomicU64,
    pub rows: std::sync::atomic::AtomicU64,
    pub facts: std::sync::atomic::AtomicU64,
    pub memories: std::sync::atomic::AtomicU64,
    pub skipped: std::sync::atomic::AtomicU64,
    pub echoes: std::sync::atomic::AtomicU64,
}

/// What one `apply` did.
#[derive(Debug, Default, Clone)]
pub struct Applied {
    pub txs: usize,
    pub skipped: usize,
    pub rows: usize,
    pub facts: usize,
    pub memories: usize,
    /// The checkpoint after the batch (stream transactions only).
    pub lsn: u64,
    pub end_lsn: u64,
}

/// A stored row image.
#[derive(Debug, Clone, Default)]
struct Image {
    vals: Row,
    mems: Map<String, Value>,
    at: i64,
}

#[derive(Default)]
struct Out {
    facts: Vec<DataValue>,
    nodes: Vec<DataValue>,
    mems: Vec<DataValue>,
    /// new memories: (id, text) for embedding
    new_mems: Vec<(String, String)>,
    /// memory ids ended: their vectors' `current` turns false
    ended_mems: Vec<String>,
    /// retraction rows of memories (text read back before the write)
    mem_retract: Vec<(String, i64)>,
    roles: Vec<DataValue>,
    types: Vec<DataValue>,
    links: Vec<DataValue>,
    entities: Vec<DataValue>,
    mentions: Vec<DataValue>,
    rows_put: BTreeMap<(String, String), (Image, i64)>,
    rows_rm: BTreeSet<(String, String)>,
    echoes: Vec<DataValue>,
    /// (predicate, vt) whose current facts all end (dropped columns)
    drop_preds: Vec<(String, i64)>,
    rows: usize,
}

/// Writes batches of source transactions into kleos.
pub struct Sink {
    db: DbInstance,
    cfg: Arc<PgConfig>,
    pub mapping: Arc<RwLock<Mapping>>,
    ckpt: Checkpoint,
    embedder: Option<String>,
    has_vec: bool,
    labels: HashMap<String, String>,
    /// Tables being copied into a kleos that holds none of their rows: no
    /// stored images to read.
    pub fresh: BTreeSet<String>,
    pub stats: Arc<Stats>,
}

impl Sink {
    pub fn new(db: DbInstance, cfg: Arc<PgConfig>, mapping: Arc<RwLock<Mapping>>, stats: Arc<Stats>) -> Result<Sink> {
        let ckpt = store::checkpoint(&db, &cfg.source)?.unwrap_or_default();
        let mut sink = Sink {
            db,
            cfg,
            mapping,
            ckpt,
            embedder: None,
            has_vec: false,
            labels: HashMap::new(),
            fresh: BTreeSet::new(),
            stats,
        };
        sink.refresh_embedder()?;
        Ok(sink)
    }

    pub fn checkpoint(&self) -> &Checkpoint {
        &self.ckpt
    }

    pub fn reload_checkpoint(&mut self) -> Result<()> {
        self.ckpt = store::checkpoint(&self.db, &self.cfg.source)?.unwrap_or_default();
        Ok(())
    }

    /// The store's embedder, when memories are to be embedded.
    pub fn refresh_embedder(&mut self) -> Result<()> {
        let r = store::read(&self.db, "?[v] := *kleos_mem_config{key: 'embedder', value: v}", vec![])?;
        let name = r.rows.first().and_then(|x| store::str_of(&x[0]));
        self.has_vec = store::relations(&self.db)?.iter().any(|x| x == MEM_VEC);
        self.embedder = match self.cfg.memory.embed.as_str() {
            "false" => None,
            "true" => Some(name.clone().ok_or_else(|| {
                miette!("pg config memory.embed is true but the store has no embedder (`::memory init {{embedder: …}}`)")
            })?),
            _ => name.filter(|n| kleos::embed::embedders().iter().any(|e| &e.name == n && e.enabled)),
        };
        if self.embedder.is_some() && !self.has_vec {
            self.embedder = None;
        }
        Ok(())
    }

    fn ns(&self) -> &str {
        &self.cfg.ns
    }

    /// Apply source transactions in one kleos transaction. Stream
    /// transactions at or below the checkpoint are skipped (a replay after a
    /// crash). `offsets` are Debezium source positions committed with them.
    pub fn apply(&mut self, txs: Vec<Tx>, offsets: Vec<(String, Value)>) -> Result<Applied> {
        let mut out_info = Applied::default();
        let mut kept = vec![];
        for tx in txs {
            if !tx.snapshot && tx.lsn > 0 && tx.lsn <= self.ckpt.lsn {
                out_info.skipped += 1;
                continue;
            }
            kept.push(tx);
        }
        self.stats.skipped.fetch_add(out_info.skipped as u64, std::sync::atomic::Ordering::Relaxed);
        if kept.is_empty() && offsets.is_empty() {
            return Ok(out_info);
        }
        // schema changes first: the mapping and ontology the changes need;
        // the predicates of dropped columns, from the mapping before
        let mut dropped: HashMap<(u64, String), Vec<String>> = HashMap::new();
        for tx in &kept {
            for sc in &tx.schema {
                let before = self.mapping.read().unwrap().tables.get(&sc.table).cloned();
                if let Some(b) = before {
                    let mut preds = vec![];
                    for col in &sc.dropped {
                        if let Some(c) = b.col(col) {
                            if matches!(c.kind, ColKind::Fact | ColKind::Category) {
                                preds.push(c.predicate.clone());
                            }
                        }
                        preds.extend(b.fks.iter().filter(|f| f.cols.contains(col)).map(|f| f.predicate.clone()));
                    }
                    dropped.insert((tx.lsn, sc.table.clone()), preds);
                }
                self.install_table(&sc.map, &format!("ddl at {}", crate::conn::fmt_lsn(tx.lsn)))?;
            }
        }
        let mapping = self.mapping.read().unwrap().clone();
        let images = self.read_images(&kept, &mapping)?;
        let mut images: HashMap<(String, String), Option<Image>> = images;
        let mut out = Out::default();
        let mut ck = self.ckpt.clone();
        for tx in &kept {
            let tx_vt = if tx.snapshot { tx.commit_ts } else { tx.commit_ts.max(ck.vt + 1) };
            if !tx.snapshot {
                ck.vt = tx_vt;
            }
            let by = match &tx.echo {
                Some(e) => format!(
                    "{}@flow:{}/{}",
                    self.cfg.by(),
                    e.workflow.clone().unwrap_or_default(),
                    e.step.clone().unwrap_or_else(|| e.idem_key.clone())
                ),
                None if tx.snapshot => format!("{}:snapshot", self.cfg.by()),
                None => self.cfg.by(),
            };
            for sc in &tx.schema {
                for pred in dropped.get(&(tx.lsn, sc.table.clone())).cloned().unwrap_or_default() {
                    out.drop_preds.push((pred, tx_vt));
                }
            }
            let mut rows_in_tx = 0usize;
            for ch in &tx.changes {
                let Some(tm) = mapping.tables.get(&ch.table) else { continue };
                if tm.is_skip() {
                    continue;
                }
                rows_in_tx += 1;
                self.change(tm, ch, tx_vt, &by, &mut images, &mut out)?;
            }
            if let Some(e) = &tx.echo {
                out.echoes.push(DataValue::List(vec![
                    s(&self.cfg.source),
                    i(tx.lsn as i64),
                    s(&e.idem_key),
                    e.workflow.as_deref().map(s).unwrap_or(DataValue::Null),
                    e.step.as_deref().map(s).unwrap_or(DataValue::Null),
                    e.actor.as_deref().map(s).unwrap_or(DataValue::Null),
                    i(rows_in_tx as i64),
                    i(tx_vt),
                ]));
            }
            out.rows += rows_in_tx;
            if !tx.snapshot && tx.lsn > 0 && tx.part.map_or(true, |p| p.1) {
                ck.lsn = tx.lsn;
                ck.end_lsn = tx.end_lsn.max(tx.lsn);
                ck.commit_ts = tx.commit_ts;
                ck.applied += 1;
                out_info.txs += 1;
            }
        }
        self.write(&out, &ck, &offsets)?;
        out_info.rows = out.rows;
        out_info.facts = out.facts.len();
        out_info.memories = out.new_mems.len();
        out_info.lsn = ck.lsn;
        out_info.end_lsn = ck.end_lsn;
        self.ckpt = ck;
        use std::sync::atomic::Ordering::Relaxed;
        self.stats.txs.fetch_add(out_info.txs as u64, Relaxed);
        self.stats.rows.fetch_add(out.rows as u64, Relaxed);
        self.stats.facts.fetch_add(out.facts.len() as u64, Relaxed);
        self.stats.memories.fetch_add(out.new_mems.len() as u64, Relaxed);
        self.stats.echoes.fetch_add(out.echoes.len() as u64, Relaxed);
        if self.labels.len() > 2_000_000 {
            self.labels.clear();
        }
        Ok(out_info)
    }

    /// Stored rows of `table` whose keys are not in `seen`: what a resync
    /// snapshot no longer has (rows deleted while the slot was lost).
    pub fn stale_rows(&self, table: &str, seen: &std::collections::HashSet<String>) -> Result<Vec<Row>> {
        let r = store::read(
            &self.db,
            &format!("?[key, vals] := *{ROW}{{src: $src, tbl: $tbl, key, vals}}"),
            vec![("src", s(&self.cfg.source)), ("tbl", s(table))],
        )?;
        Ok(r.rows
            .iter()
            .filter(|x| !seen.contains(x[0].get_str().unwrap_or("")))
            .filter_map(|x| match store::json_of(&x[1]) {
                Value::Object(m) => Some(m.into_iter().collect()),
                _ => None,
            })
            .collect())
    }

    /// Does kleos hold rows of `table`?
    pub fn has_rows(&self, table: &str) -> Result<bool> {
        let r = store::read(
            &self.db,
            &format!("?[key] := *{ROW}{{src: $src, tbl: $tbl, key}} :limit 1"),
            vec![("src", s(&self.cfg.source)), ("tbl", s(table))],
        )?;
        Ok(!r.rows.is_empty())
    }

    /// Install a table's (new) mapping: the in-memory mapping, a stored
    /// mapping version, the ontology (unless review mode) and a change row.
    pub fn install_table(&mut self, tm: &TableMap, reason: &str) -> Result<()> {
        crate::ops::install_table(&self.db, &self.cfg, &self.mapping, tm, reason)
    }

    fn key_of(tm: &TableMap, row: &Row) -> Option<Vec<Value>> {
        let mut k = vec![];
        for c in &tm.key {
            match row.get(c) {
                Some(Value::Null) | None => return None,
                Some(v) => k.push(v.clone()),
            }
        }
        Some(k)
    }

    /// Images of every entity the batch touches (one read).
    fn read_images(&self, txs: &[Tx], mapping: &Mapping) -> Result<HashMap<(String, String), Option<Image>>> {
        let mut want: BTreeSet<(String, String)> = BTreeSet::new();
        let mut truncated: BTreeSet<String> = BTreeSet::new();
        for tx in txs {
            for ch in &tx.changes {
                let Some(tm) = mapping.tables.get(&ch.table) else { continue };
                if tm.is_skip() {
                    continue;
                }
                if ch.op == Op::Truncate {
                    truncated.insert(ch.table.clone());
                    continue;
                }
                if ch.op == Op::Read && self.fresh.contains(&ch.table) {
                    continue;
                }
                for r in [&ch.new, &ch.old].into_iter().flatten() {
                    if let Some(k) = Self::key_of(tm, r) {
                        want.insert((ch.table.clone(), row_key(&k)));
                    }
                }
            }
        }
        let mut out: HashMap<(String, String), Option<Image>> = want.iter().map(|k| (k.clone(), None)).collect();
        let parse = |x: &[DataValue]| Image {
            vals: match store::json_of(&x[2]) {
                Value::Object(m) => m.into_iter().collect(),
                _ => Row::new(),
            },
            mems: match store::json_of(&x[3]) {
                Value::Object(m) => m,
                _ => Map::new(),
            },
            at: x[4].get_int().unwrap_or(0),
        };
        if !want.is_empty() {
            let keys: Vec<DataValue> = want.iter().map(|(t, k)| DataValue::List(vec![s(t), s(k)])).collect();
            let r = store::read(
                &self.db,
                &format!(
                    "k[tbl, key] <- $keys \
                     ?[tbl, key, vals, mems, at] := k[tbl, key], *{ROW}{{src: $src, tbl, key, vals, mems, at}}"
                ),
                vec![("keys", DataValue::List(keys)), ("src", s(&self.cfg.source))],
            )?;
            for x in &r.rows {
                out.insert((store::str_of(&x[0]).unwrap_or_default(), store::str_of(&x[1]).unwrap_or_default()), Some(parse(x)));
            }
        }
        for t in truncated {
            let r = store::read(
                &self.db,
                &format!("?[tbl, key, vals, mems, at] := *{ROW}{{src: $src, tbl, key, vals, mems, at}}, tbl = $tbl"),
                vec![("src", s(&self.cfg.source)), ("tbl", s(&t))],
            )?;
            for x in &r.rows {
                out.insert((t.clone(), store::str_of(&x[1]).unwrap_or_default()), Some(parse(x)));
            }
        }
        Ok(out)
    }

    fn change(
        &mut self,
        tm: &TableMap,
        ch: &Change,
        tx_vt: i64,
        by: &str,
        images: &mut HashMap<(String, String), Option<Image>>,
        out: &mut Out,
    ) -> Result<()> {
        if ch.op == Op::Truncate {
            let keys: Vec<(String, String)> = images.keys().filter(|(t, _)| t == &tm.table).cloned().collect();
            for k in keys {
                if let Some(Some(img)) = images.get(&k).cloned() {
                    let vtv = tx_vt.max(img.at + 1);
                    self.emit(tm, &k.1, Some(&img), None, vtv, by, out)?;
                    images.insert(k, None);
                }
            }
            return Ok(());
        }
        // the old key, when an update changed it or for a delete
        let old_key = ch.old.as_ref().and_then(|r| Self::key_of(tm, r)).map(|k| row_key(&k));
        let new_key = ch.new.as_ref().and_then(|r| Self::key_of(tm, r)).map(|k| row_key(&k));
        if ch.op == Op::Delete {
            let Some(k) = old_key.or(new_key) else { return Ok(()) };
            let id = (tm.table.clone(), k.clone());
            if let Some(Some(img)) = images.get(&id).cloned() {
                let vtv = tx_vt.max(img.at + 1);
                self.emit(tm, &k, Some(&img), None, vtv, by, out)?;
            }
            images.insert(id, None);
            return Ok(());
        }
        let Some(nk) = new_key else {
            log::warn!("pg: a {} row without its key columns {:?}; skipped", tm.table, tm.key);
            return Ok(());
        };
        if let Some(ok) = old_key.filter(|k| k != &nk) {
            // the key changed: the old entity ends, a new one begins
            let id = (tm.table.clone(), ok.clone());
            if let Some(Some(img)) = images.get(&id).cloned() {
                let vtv = tx_vt.max(img.at + 1);
                self.emit(tm, &ok, Some(&img), None, vtv, by, out)?;
            }
            images.insert(id, None);
        }
        let id = (tm.table.clone(), nk.clone());
        let prev = images.get(&id).cloned().flatten();
        let mut vals: Row = Row::new();
        for (c, v) in ch.new.as_ref().unwrap() {
            if tm.kept(c) {
                vals.insert(c.clone(), v.clone());
            }
        }
        for c in &ch.missing {
            if let Some(v) = prev.as_ref().and_then(|p| p.vals.get(c)) {
                vals.insert(c.clone(), v.clone());
            }
        }
        let vtv = match &prev {
            Some(p) if !(ch.op == Op::Read && self.fresh.contains(&tm.table)) => tx_vt.max(p.at + 1),
            _ => tx_vt,
        };
        let img = Image { vals, mems: prev.as_ref().map(|p| p.mems.clone()).unwrap_or_default(), at: vtv };
        if let Some(p) = &prev {
            if p.vals == img.vals {
                return Ok(()); // a replay or an update that changed nothing kleos keeps
            }
        }
        let written = self.emit(tm, &nk, prev.as_ref(), Some(&img), vtv, by, out)?;
        images.insert(id, Some(written));
        Ok(())
    }

    fn label_of(&self, tm: &TableMap, vals: &Row) -> Option<String> {
        mapping::render_label(tm, vals)
    }

    /// Facts, node, memories and the row image for one entity going from
    /// `prev` to `new` at `at`. Returns the new image.
    #[allow(clippy::too_many_arguments)]
    fn emit(&mut self, tm: &TableMap, key: &str, prev: Option<&Image>, new: Option<&Image>, at: i64, by: &str, out: &mut Out) -> Result<Image> {
        let src = self.cfg.source.clone();
        let ns = self.ns().to_string();
        let entity = format!("pg:{src}/{}/{key}", tm.table);
        let pv = prev.map(|p| &p.vals);
        let nv = new.map(|n| &n.vals);
        let mut mems = new.map(|n| n.mems.clone()).or_else(|| prev.map(|p| p.mems.clone())).unwrap_or_default();
        let get = |r: Option<&Row>, c: &str| -> Option<Value> { r.and_then(|r| r.get(c)).filter(|v| !v.is_null()).cloned() };
        let fact = |s_: &str, p: &str, o: &str, lit: Value, assert: bool| -> DataValue {
            DataValue::List(vec![
                s(&ns),
                s(s_),
                s(p),
                s(o),
                vt(at, assert),
                j(lit),
                DataValue::from(1.0),
                s(by),
                DataValue::Null,
                s("current"),
            ])
        };
        let edge_lit = |extra: Value| -> Value {
            let mut m = Map::new();
            m.insert("src".into(), json!(format!("pg:{src}")));
            if let Value::Object(e) = extra {
                m.extend(e);
            }
            Value::Object(m)
        };
        match &tm.shape {
            Shape::Skip { .. } => {}
            Shape::Edge { source, destination, predicate } => {
                let end = |r: Option<&Row>, e: &crate::mapping::EndMap| -> Option<String> {
                    let mut vs = vec![];
                    for c in &e.cols {
                        vs.push(get(r, c)?);
                    }
                    Some(entity_key(&src, &e.table, &vs))
                };
                let attrs = |r: Option<&Row>| -> Value {
                    let mut m = Map::new();
                    for c in &tm.columns {
                        if matches!(c.kind, ColKind::Fact | ColKind::Category | ColKind::Text) && !c.pii {
                            if let Some(v) = get(r, &c.name) {
                                m.insert(c.name.clone(), v);
                            }
                        }
                    }
                    Value::Object(m)
                };
                let old = (end(pv, source), end(pv, destination));
                let newe = (end(nv, source), end(nv, destination));
                let (oa, na) = (attrs(pv), attrs(nv));
                if let (Some(a), Some(b)) = &old {
                    if old != newe || nv.is_none() {
                        out.facts.push(fact(a, predicate, b, edge_lit(json!({"row": key, "attrs": oa})), false));
                    }
                }
                if let (Some(a), Some(b)) = &newe {
                    if old != newe || oa != na || pv.is_none() {
                        out.facts.push(fact(a, predicate, b, edge_lit(json!({"row": key, "attrs": na})), true));
                    }
                }
            }
            Shape::Entity => {
                // data facts and categories
                for c in &tm.columns {
                    match c.kind {
                        ColKind::Fact => {
                            let (a, b) = (get(pv, &c.name), get(nv, &c.name));
                            if a == b {
                                continue;
                            }
                            if let Some(a) = a {
                                out.facts.push(fact(&entity, &c.predicate, &literal_o(&a), json!({"value": a, "datatype": c.tk.xsd()}), false));
                            }
                            if let Some(b) = b {
                                out.facts.push(fact(&entity, &c.predicate, &literal_o(&b), json!({"value": b, "datatype": c.tk.xsd()}), true));
                            }
                        }
                        ColKind::Category => {
                            let (a, b) = (get(pv, &c.name), get(nv, &c.name));
                            if a == b {
                                continue;
                            }
                            let node = |v: &Value| format!("pg:{src}/cat/{}.{}/{}", tm.table, c.name, crate::value::key_part(v));
                            if let Some(a) = a {
                                out.facts.push(fact(&entity, &c.predicate, &node(&a), edge_lit(json!({"value": a})), false));
                            }
                            if let Some(b) = b {
                                let n = node(&b);
                                out.facts.push(fact(&entity, &c.predicate, &n, edge_lit(json!({"value": b})), true));
                                out.nodes.push(DataValue::List(vec![s(&ns), s(&n), vt(at, true), s("Category"), s(&display(&b)), DataValue::Null]));
                            }
                        }
                        _ => {}
                    }
                }
                // foreign keys
                for f in &tm.fks {
                    let end = |r: Option<&Row>| -> Option<String> {
                        let mut vs = vec![];
                        for c in &f.cols {
                            vs.push(get(r, c)?);
                        }
                        Some(entity_key(&src, &f.table, &vs))
                    };
                    let (a, b) = (end(pv), end(nv));
                    if a == b {
                        continue;
                    }
                    if let Some(a) = a {
                        out.facts.push(fact(&entity, &f.predicate, &a, edge_lit(json!({"fk": f.name})), false));
                    }
                    if let Some(b) = b {
                        out.facts.push(fact(&entity, &f.predicate, &b, edge_lit(json!({"fk": f.name})), true));
                    }
                }
                // text columns: one memory each
                for c in tm.columns.iter().filter(|c| c.kind == ColKind::Text) {
                    let (a, b) = (get(pv, &c.name), get(nv, &c.name));
                    if a == b && !(pv.is_some() && nv.is_none()) {
                        continue;
                    }
                    let old_id = mems.get(&c.name).and_then(|v| v.as_str()).map(|x| x.to_string());
                    if let Some(oid) = &old_id {
                        out.mem_retract.push((oid.clone(), at));
                        out.ended_mems.push(oid.clone());
                        mems.remove(&c.name);
                    }
                    if let (Some(b), Some(n)) = (b, nv) {
                        let text = display(&b);
                        if !text.trim().is_empty() {
                            let label = self.label_of(tm, n).unwrap_or_else(|| key.to_string());
                            let body = format!("{} {} ({} {}) {}: {}", tm.class, label, short(&tm.table), key, mapping::humanize(&c.name), text);
                            let id = mem_id(&entity, &c.name, at);
                            self.new_memory(out, &id, &body, "pg_text", &format!("pg:{src}/{}/{key}#{}", tm.table, c.name), at, None, old_id.as_deref(), by);
                            mems.insert(c.name.clone(), json!(id));
                        }
                    }
                }
                // the node and the row memory
                let label = nv.and_then(|n| self.label_of(tm, n));
                let old_label = mems.get("_label").and_then(|v| v.as_str()).map(|x| x.to_string());
                if tm.row_memory || nv.is_none() {
                    let text = nv.map(|n| self.render_row(tm, key, n, label.as_deref()));
                    let h = text.as_ref().map(|t| short_hash(t));
                    let old_h = mems.get("_row_h").and_then(|v| v.as_str()).map(|x| x.to_string());
                    let old_id = mems.get("_row").and_then(|v| v.as_str()).map(|x| x.to_string());
                    if h != old_h || nv.is_none() {
                        if let Some(oid) = &old_id {
                            out.mem_retract.push((oid.clone(), at));
                            out.ended_mems.push(oid.clone());
                            mems.remove("_row");
                            mems.remove("_row_h");
                        }
                        if let (Some(t), Some(h)) = (text, h) {
                            if tm.row_memory {
                                let id = mem_id(&entity, "row", at);
                                self.new_memory(out, &id, &t, "pg_row", &format!("pg:{src}/{}/{key}", tm.table), at, Some(&entity), old_id.as_deref(), by);
                                mems.insert("_row".into(), json!(id));
                                mems.insert("_row_h".into(), json!(h));
                            }
                        }
                    }
                }
                let row_mem = mems.get("_row").and_then(|v| v.as_str()).map(s).unwrap_or(DataValue::Null);
                match (pv, nv) {
                    (_, Some(_)) => {
                        if pv.is_none() || label != old_label || prev.and_then(|p| p.mems.get("_row")) != mems.get("_row") {
                            out.nodes.push(DataValue::List(vec![
                                s(&ns),
                                s(&entity),
                                vt(at, true),
                                s(&tm.class),
                                label.as_deref().map(s).unwrap_or(DataValue::Null),
                                row_mem,
                            ]));
                        }
                        if let Some(l) = &label {
                            self.labels.insert(entity.clone(), l.clone());
                            mems.insert("_label".into(), json!(l));
                            if self.cfg.memory.entities && pv.is_none() {
                                let eid = format!("e_pg_{}", short_hash(&entity));
                                out.entities.push(DataValue::List(vec![
                                    s(&ns),
                                    s(&eid),
                                    s(l),
                                    s(&kleos::memory::norm_name(l)),
                                    s(&tm.class),
                                    i(at),
                                ]));
                                if let Some(mid) = mems.get("_row").and_then(|v| v.as_str()) {
                                    out.mentions.push(DataValue::List(vec![s(&ns), s(mid), s(&eid)]));
                                }
                            }
                        }
                    }
                    (Some(_), None) => {
                        out.nodes.push(DataValue::List(vec![
                            s(&ns),
                            s(&entity),
                            vt(at, false),
                            s(&tm.class),
                            old_label.as_deref().map(s).unwrap_or(DataValue::Null),
                            DataValue::Null,
                        ]));
                        self.labels.remove(&entity);
                    }
                    _ => {}
                }
            }
        }
        let id = (tm.table.clone(), key.to_string());
        match new {
            Some(n) => {
                let img = Image { vals: n.vals.clone(), mems, at };
                out.rows_rm.remove(&id);
                out.rows_put.insert(id, (img.clone(), at));
                Ok(img)
            }
            None => {
                out.rows_put.remove(&id);
                out.rows_rm.insert(id);
                Ok(Image::default())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn new_memory(&mut self, out: &mut Out, id: &str, text: &str, kind: &str, source: &str, at: i64, subject: Option<&str>, supersedes: Option<&str>, by: &str) {
        let ns = self.ns().to_string();
        out.mems.push(DataValue::List(vec![s(&ns), s(id), vt(at, true), s(text), s(kind), s(source), i(at), DataValue::Null]));
        out.new_mems.push((id.to_string(), text.to_string()));
        out.roles.push(DataValue::List(vec![s(&ns), s(id), s("other")]));
        if let Some(sub) = subject {
            out.types.push(DataValue::List(vec![
                s(&ns),
                s(id),
                s(&self.cfg.onto()),
                s("pg_row"),
                s(sub),
                DataValue::Null,
                DataValue::from(1.0),
                s("pg"),
            ]));
        }
        if let Some(old) = supersedes {
            out.links.push(DataValue::List(vec![s(&ns), s(id), s("supersedes"), s(old), i(at), DataValue::from(1.0), s(by)]));
        }
    }

    /// `Customer Acme Corp (customers 42): plan: pro; status: active; account manager: Jane Doe.`
    fn render_row(&mut self, tm: &TableMap, key: &str, vals: &Row, label: Option<&str>) -> String {
        if let Some(t) = &tm.row_template {
            return mapping::fill(t, vals);
        }
        let mut parts = vec![];
        for c in &tm.columns {
            if c.pii || !matches!(c.kind, ColKind::Fact | ColKind::Category) || (tm.key.contains(&c.name) && tm.key.len() == 1) {
                continue;
            }
            if let Some(v) = vals.get(&c.name).filter(|v| !v.is_null()) {
                if label.map_or(false, |l| display(v) == l) {
                    continue;
                }
                parts.push(format!("{}: {}", mapping::humanize(&c.name), display(v)));
            }
        }
        let src = self.cfg.source.clone();
        let mut need: Vec<String> = vec![];
        let mut fk_parts: Vec<(String, String, String)> = vec![];
        for f in &tm.fks {
            let mut vs = vec![];
            for c in &f.cols {
                match vals.get(c).filter(|v| !v.is_null()) {
                    Some(v) => vs.push(v.clone()),
                    None => break,
                }
            }
            if vs.len() != f.cols.len() {
                continue;
            }
            let target = entity_key(&src, &f.table, &vs);
            if !self.labels.contains_key(&target) {
                need.push(target.clone());
            }
            fk_parts.push((mapping::humanize(f.predicate.rsplit('.').next().unwrap_or(&f.predicate)), target, row_key(&vs)));
        }
        if !need.is_empty() {
            self.load_labels(&need);
        }
        for (name, target, k) in fk_parts {
            let shown = self.labels.get(&target).cloned().unwrap_or(k);
            parts.push(format!("{name}: {shown}"));
        }
        let head = match label {
            Some(l) => format!("{} {} ({} {})", tm.class, l, short(&tm.table), key),
            None => format!("{} {} ({})", tm.class, key, short(&tm.table)),
        };
        if parts.is_empty() {
            format!("{head}.")
        } else {
            format!("{head}: {}.", parts.join("; "))
        }
    }

    fn load_labels(&mut self, nodes: &[String]) {
        let keys: Vec<DataValue> = nodes.iter().map(|n| DataValue::List(vec![s(n)])).collect();
        if let Ok(r) = store::read(
            &self.db,
            &format!("k[node] <- $keys ?[node, label] := k[node], *{NODE}{{ns: $ns, node, label @ 'NOW'}}, !is_null(label)"),
            vec![("keys", DataValue::List(keys)), ("ns", s(&self.cfg.ns))],
        ) {
            for x in r.rows {
                if let (Some(n), Some(l)) = (store::str_of(&x[0]), store::str_of(&x[1])) {
                    self.labels.insert(n, l);
                }
            }
        }
    }

    fn write(&mut self, out: &Out, ck: &Checkpoint, offsets: &[(String, Value)]) -> Result<()> {
        let ns = self.cfg.ns.clone();
        let src = self.cfg.source.clone();
        let mut script = String::new();
        let mut p: BTreeMap<String, DataValue> = BTreeMap::new();
        // the checkpoint first: a concurrent writer of this source fails here
        if ck.lsn != self.ckpt.lsn || ck.applied != self.ckpt.applied {
            script.push_str(&format!(
                "{{?[lsn] := *{CHECKPOINT}{{src: $src, lsn}}, lsn == $prev_lsn :assert some}}\
                 {{?[src, lsn, end_lsn, commit_ts, vt, applied, consistent, slot, updated] <- [[$src, $lsn, $end_lsn, $commit_ts, $vt, $applied, $consistent, $slot, $updated]] \
                   :put {CHECKPOINT} {{src => lsn, end_lsn, commit_ts, vt, applied, consistent, slot, updated}}}}"
            ));
            p.insert("prev_lsn".into(), i(self.ckpt.lsn as i64));
            p.insert("lsn".into(), i(ck.lsn as i64));
            p.insert("end_lsn".into(), i(ck.end_lsn as i64));
            p.insert("commit_ts".into(), i(ck.commit_ts));
            p.insert("vt".into(), i(ck.vt));
            p.insert("applied".into(), i(ck.applied));
            p.insert("consistent".into(), i(ck.consistent as i64));
            p.insert("slot".into(), s(&ck.slot));
            p.insert("updated".into(), i(store::now_us()));
        }
        p.insert("src".into(), s(&src));
        p.insert("ns".into(), s(&ns));
        // retraction rows of memories: their stored values, read now
        let mut mem_rows = out.mems.clone();
        if !out.mem_retract.is_empty() {
            let ids: Vec<DataValue> = out.mem_retract.iter().map(|(id, _)| DataValue::List(vec![s(id)])).collect();
            let r = store::read(
                &self.db,
                &format!("k[id] <- $ids ?[id, text, kind, source, created] := k[id], *{MEM}{{ns: $ns, id, text, kind, source, created @ 'NOW'}}"),
                vec![("ids", DataValue::List(ids)), ("ns", s(&ns))],
            )?;
            let found: HashMap<String, Vec<DataValue>> = r.rows.into_iter().map(|x| (store::str_of(&x[0]).unwrap_or_default(), x)).collect();
            for (id, at) in &out.mem_retract {
                if let Some(x) = found.get(id) {
                    mem_rows.push(DataValue::List(vec![s(&ns), s(id), vt(*at, false), x[1].clone(), x[2].clone(), x[3].clone(), x[4].clone(), DataValue::Null]));
                }
            }
        }
        let mut block = |rows: Vec<DataValue>, name: &'static str, head: &str, put: &str| {
            if rows.is_empty() {
                return;
            }
            script.push_str(&format!("{{?[{head}] <- ${name} :put {put}}}"));
            p.insert(name.to_string(), DataValue::List(rows));
        };
        block(out.facts.clone(), "facts", "ns, s, p, o, vt, lit, conf, by, mem, status", &format!("{FACT} {{ns, s, p, o, vt => lit, conf, by, mem, status}}"));
        block(out.nodes.clone(), "nodes", "ns, node, vt, class, label, mem", &format!("{NODE} {{ns, node, vt => class, label, mem}}"));
        block(mem_rows, "mems", "ns, id, vt, text, kind, source, created, importance", &format!("{MEM} {{ns, id, vt => text, kind, source, created, importance}}"));
        block(out.roles.clone(), "roles", "ns, id, role", &format!("{MEM_ROLE} {{ns, id => role}}"));
        block(out.types.clone(), "types", "ns, id, onto, cls, subject, prop, score, by", &format!("{MEM_TYPE} {{ns, id => onto, cls, subject, prop, score, by}}"));
        block(out.links.clone(), "links", "ns, src, rel, dst, created, p, decided_by", &format!("{MEM_LINK} {{ns, src, rel, dst => created, p, decided_by}}"));
        block(out.entities.clone(), "entities", "ns, eid, name, norm, kind, created", &format!("{MEM_ENTITY} {{ns, eid => name, norm, kind, created}}"));
        block(out.mentions.clone(), "mentions", "ns, id, eid", &format!("{MEM_MENTION} {{ns, id, eid}}"));
        block(out.echoes.clone(), "echoes", "src, lsn, idem, wf, step, actor, rows, at", &format!("{ECHO} {{src, lsn => idem, wf, step, actor, rows, at}}"));
        let puts: Vec<DataValue> = out
            .rows_put
            .iter()
            .map(|((t, k), (img, at))| {
                DataValue::List(vec![
                    s(&src),
                    s(t),
                    s(k),
                    j(Value::Object(img.vals.clone().into_iter().collect())),
                    j(Value::Object(img.mems.clone())),
                    i(ck.lsn as i64),
                    i(*at),
                ])
            })
            .collect();
        block(puts, "row_put", "src, tbl, key, vals, mems, lsn, at", &format!("{ROW} {{src, tbl, key => vals, mems, lsn, at}}"));
        if !out.rows_rm.is_empty() {
            let rm: Vec<DataValue> = out.rows_rm.iter().map(|(t, k)| DataValue::List(vec![s(&src), s(t), s(k)])).collect();
            script.push_str(&format!("{{?[src, tbl, key] <- $row_rm :rm {ROW} {{src, tbl, key}}}}"));
            p.insert("row_rm".into(), DataValue::List(rm));
        }
        if !offsets.is_empty() {
            let rows: Vec<DataValue> = offsets.iter().map(|(part, pos)| DataValue::List(vec![s(&src), s(part), j(pos.clone()), i(store::now_us())])).collect();
            script.push_str(&format!("{{?[src, part, pos, at] <- $offsets :put {OFFSET} {{src, part => pos, at}}}}"));
            p.insert("offsets".into(), DataValue::List(rows));
        }
        // dropped columns: every current fact of the predicate ends
        for (n, (pred, at)) in out.drop_preds.iter().enumerate() {
            script.push_str(&format!(
                "{{?[ns, s, p, o, vt, lit, conf, by, mem, status] := *{FACT}{{ns, s, p, o, lit, conf, by, mem, status @ 'NOW'}}, \
                   ns = $ns, p = $drop_p{n}, starts_with(s, $prefix), vt = $drop_vt{n} \
                   :put {FACT} {{ns, s, p, o, vt => lit, conf, by, mem, status}}}}"
            ));
            p.insert(format!("drop_p{n}"), s(pred));
            p.insert(format!("drop_vt{n}"), vt(*at, false));
        }
        if !out.drop_preds.is_empty() {
            p.insert("prefix".into(), s(&format!("pg:{src}/")));
        }
        // vectors
        if self.has_vec {
            if let Some(model) = self.embedder.clone() {
                if !out.new_mems.is_empty() {
                    let mut vrows = vec![];
                    for chunk in out.new_mems.chunks(64) {
                        let texts: Vec<&str> = chunk.iter().map(|(_, t)| t.as_str()).collect();
                        let vecs = kleos::embed::embed_texts(&model, &texts)?;
                        for ((id, _), v) in chunk.iter().zip(vecs) {
                            vrows.push(DataValue::List(vec![
                                s(&ns),
                                s(id),
                                DataValue::Bool(true),
                                DataValue::List(v.iter().map(|x| DataValue::from(*x as f64)).collect()),
                            ]));
                        }
                    }
                    script.push_str(&format!("{{?[ns, id, current, vec] <- $vecs :put {MEM_VEC} {{ns, id => current, vec}}}}"));
                    p.insert("vecs".into(), DataValue::List(vrows));
                }
            }
            if !out.ended_mems.is_empty() {
                let rows: Vec<DataValue> = out.ended_mems.iter().map(|id| DataValue::List(vec![s(id)])).collect();
                script.push_str(&format!(
                    "{{k[id] <- $ended ?[ns, id, current] := k[id], *{MEM_VEC}{{ns, id}}, ns = $ns, current = to_bool(0) :update {MEM_VEC} {{ns, id => current}}}}"
                ));
                p.insert("ended".into(), DataValue::List(rows));
            }
        }
        if script.is_empty() {
            return Ok(());
        }
        self.db
            .run_script(&script, p, kleos::ScriptMutability::Mutable)
            .map_err(|e| miette!("pg {}: kleos write failed: {e}", self.cfg.source))?;
        Ok(())
    }
}

/// A memory id: stable for (entity, column, valid time), so a replayed
/// change writes the same memory again instead of a second one. No `.`
/// (that marks a claim of a statement).
fn mem_id(entity: &str, col: &str, at: i64) -> String {
    format!("pg_{}_{}_{:x}", &short_hash(entity)[..16], &short_hash(col)[..6], at)
}

fn short(table: &str) -> String {
    match table.split_once('.') {
        Some(("public", t)) => t.to_string(),
        _ => table.to_string(),
    }
}
