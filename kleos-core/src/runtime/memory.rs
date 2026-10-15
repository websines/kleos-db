/*
 * Copyright 2026, The kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Agent-memory semantics (kleos fork, UPGRADE_PLAN E1/E2; the contract is
//! `docs/specs/memory-supersession-provenance.md`).
//!
//! - **Supersession (E1).** A relation with a valid-time axis may declare a
//!   *supersession key*: a leading prefix of its key columns that determines
//!   at most one current row (`::supersede create facts {subject, predicate}`).
//!   Every `:put`/`:insert` then ends the validity of the group's other
//!   current row at the new row's valid time — by appending retraction rows
//!   in the same statement, never by deleting — keeps late-arriving facts
//!   from ending newer ones, and drops re-assertions of an identical current
//!   fact. `::changes` reports what started, ended or changed between two
//!   valid times (optionally as known at a transaction time).
//! - **Provenance (E2).** A plain *link relation* whose key is the derived
//!   relation's key followed by the source relation's key is declared with
//!   `::provenance create`. `::retract` ends a source's validity and,
//!   transitively, that of every derived row left without a live source;
//!   `::erase` hard-deletes (GDPR) the source, its links, and every derived
//!   row left with no source at all — search-index entries included — and
//!   records salted audit markers. `::lineage` explains a row.
//!
//! Declarations live in two ordinary catalog relations keyed by relation
//! id, so a rename keeps them and a drop (ids are never reused) retires them.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ops::Bound;

use crate::bail;
use itertools::Itertools;
use miette::{ensure, Diagnostic, Result};
use smartstring::{LazyCompact, SmartString};
use thiserror::Error;

use crate::data::functions::MAX_VALIDITY_TS;
use crate::data::program::RelationOp;
use crate::data::relation::{ColType, ColumnDef, NullableColType, StoredRelationMetadata};
use crate::data::symb::Symbol;
use crate::data::tuple::{Tuple, TupleT};
use crate::data::value::{DataValue, Validity, ValidityTs, LARGEST_UTF_CHAR};
use crate::parse::memory::MemoryOp;
use crate::parse::SourceSpan;
use crate::runtime::callback::CallbackCollector;
use crate::runtime::db::{write_import_tuple, ImportSearchIndexes};
use crate::runtime::relation::{
    AccessLevel, InputRelationHandle, InsufficientAccessLevel, RelationHandle, RelationId,
};
use crate::runtime::transact::SessionTx;
use crate::storage::Storage;
use crate::{Db, NamedRows};

pub(crate) const SUPERSEDE_CATALOG: &str = "kleos_supersede";
pub(crate) const PROVENANCE_CATALOG: &str = "kleos_provenance";
const EVICT_AUDIT: &str = "kleos_evict_audit";

// ---------------------------------------------------------------------------
// relation shape helpers

/// The temporal axes a relation declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Axes {
    Plain,
    Vt,
    Bitemporal,
    TtOnly,
}

pub(crate) fn axes(h: &RelationHandle) -> Axes {
    if h.has_txtime() {
        if h.is_tt_only() {
            Axes::TtOnly
        } else {
            Axes::Bitemporal
        }
    } else if matches!(h.metadata.keys.last(),
        Some(c) if c.typing.coltype == ColType::Validity && !c.typing.nullable)
    {
        Axes::Vt
    } else {
        Axes::Plain
    }
}

/// Number of non-temporal key columns.
pub(crate) fn plain_len(h: &RelationHandle) -> usize {
    let n = h.metadata.keys.len();
    n - match axes(h) {
        Axes::Plain => 0,
        Axes::Vt | Axes::TtOnly => 1,
        Axes::Bitemporal => 2,
    }
}

/// The relation's writable layout: every column except the engine-assigned
/// TxTime, in stored order, bound by name. Rows in this layout can be fed
/// straight back into `execute_relation`.
pub(crate) fn canonical_input(h: &RelationHandle) -> (InputRelationHandle, Vec<Symbol>) {
    let n = if h.has_txtime() {
        h.metadata.keys.len() - 1
    } else {
        h.metadata.keys.len()
    };
    let keys = h.metadata.keys[..n].to_vec();
    let non_keys = h.metadata.non_keys.clone();
    let sym = |c: &ColumnDef| Symbol::new(c.name.clone(), SourceSpan::default());
    let key_bindings = keys.iter().map(sym).collect_vec();
    let dep_bindings = non_keys.iter().map(sym).collect_vec();
    let headers = key_bindings
        .iter()
        .chain(dep_bindings.iter())
        .cloned()
        .collect_vec();
    (
        InputRelationHandle {
            name: Symbol::new(h.name.clone(), SourceSpan::default()),
            metadata: StoredRelationMetadata { keys, non_keys },
            key_bindings,
            dep_bindings,
            span: SourceSpan::default(),
        },
        headers,
    )
}

fn vt_of(v: &DataValue) -> Result<(i64, bool)> {
    match v {
        DataValue::Validity(v) => Ok((v.timestamp.0 .0, v.is_assert.0)),
        v => bail!(
            "expected a Validity value in the valid-time column, got {:?}",
            v
        ),
    }
}

fn vld(ts: i64, is_assert: bool) -> DataValue {
    DataValue::Validity(Validity {
        timestamp: ValidityTs(Reverse(ts)),
        is_assert: Reverse(is_assert),
    })
}

fn coerce_key(h: &RelationHandle, key: &[DataValue], what: &str) -> Result<Tuple> {
    let p = plain_len(h);
    ensure!(
        key.len() == p,
        "{what} key arity mismatch for {}: expected {} plain key column(s), got {}",
        h.name,
        p,
        key.len()
    );
    key.iter()
        .zip(h.metadata.keys.iter())
        .map(|(v, c)| c.typing.coerce(v.clone(), MAX_VALIDITY_TS))
        .collect()
}

fn col(name: &str, coltype: ColType) -> ColumnDef {
    ColumnDef {
        name: SmartString::from(name),
        typing: NullableColType {
            coltype,
            nullable: false,
        },
        default_gen: None,
    }
}

fn supersede_meta() -> StoredRelationMetadata {
    StoredRelationMetadata {
        keys: vec![col("rel_id", ColType::Int)],
        non_keys: vec![
            col("relation", ColType::String),
            col(
                "key_cols",
                ColType::List {
                    eltype: Box::new(NullableColType {
                        coltype: ColType::String,
                        nullable: false,
                    }),
                    len: None,
                },
            ),
        ],
    }
}

fn provenance_meta() -> StoredRelationMetadata {
    StoredRelationMetadata {
        keys: vec![col("link_id", ColType::Int)],
        non_keys: vec![
            col("link", ColType::String),
            col("derived_id", ColType::Int),
            col("source_id", ColType::Int),
        ],
    }
}

fn catalog_for_read(
    tx: &SessionTx<'_>,
    name: &str,
    meta: &StoredRelationMetadata,
) -> Result<Option<RelationHandle>> {
    if !tx.relation_exists(name)? {
        return Ok(None);
    }
    let h = tx.get_relation(name, false)?;
    ensure!(
        &h.metadata == meta
            && h.has_no_index()
            && h.put_triggers.is_empty()
            && h.rm_triggers.is_empty()
            && h.replace_triggers.is_empty(),
        "the relation name {name} is reserved for agent-memory declarations; \
         the existing relation must keep the exact reserved schema and carry no indices or triggers"
    );
    Ok(Some(h))
}

fn catalog_for_write(
    tx: &mut SessionTx<'_>,
    name: &str,
    meta: StoredRelationMetadata,
) -> Result<RelationHandle> {
    let h = match catalog_for_read(tx, name, &meta)? {
        Some(h) => h,
        None => tx.create_relation(InputRelationHandle {
            name: Symbol::new(name, SourceSpan::default()),
            metadata: meta,
            key_bindings: vec![],
            dep_bindings: vec![],
            span: SourceSpan::default(),
        })?,
    };
    ensure!(
        h.access_level == AccessLevel::Normal,
        "catalog '{name}' is {}; set its access level to normal before modifying it",
        h.access_level
    );
    tx.mark_dirty(&h);
    Ok(h)
}

fn catalog_put(tx: &mut SessionTx<'_>, cat: &RelationHandle, row: &[DataValue]) -> Result<()> {
    let k = cat.encode_key_for_store(row, SourceSpan::default())?;
    let v = cat.encode_val_for_store(row, SourceSpan::default())?;
    tx.store_tx.put(&k, &v)?;
    Ok(())
}

/// Every live, non-index stored relation by id.
fn relations_by_id(tx: &SessionTx<'_>) -> Result<BTreeMap<u64, RelationHandle>> {
    let lower = vec![DataValue::from("")].encode_as_key(RelationId::SYSTEM);
    let upper =
        vec![DataValue::from(String::from(LARGEST_UTF_CHAR))].encode_as_key(RelationId::SYSTEM);
    let mut out = BTreeMap::new();
    for kv in tx.store_tx.range_scan(&lower, &upper) {
        let (k, v) = kv?;
        if upper <= k {
            break;
        }
        let h = RelationHandle::decode(&v)?;
        if h.name.contains(':') {
            continue;
        }
        out.insert(h.id.0, h);
    }
    Ok(out)
}

fn user_relation(tx: &SessionTx<'_>, rel: &str, what: &str) -> Result<RelationHandle> {
    ensure!(
        !rel.contains(':') && !rel.starts_with('_'),
        "{what}: '{rel}' must be a stored (non-index, non-temporary) relation"
    );
    tx.get_relation(rel, false)
}

fn status_ok() -> NamedRows {
    NamedRows::new(
        vec!["status".to_string()],
        vec![vec![DataValue::from("OK")]],
    )
}

// ---------------------------------------------------------------------------
// E1: supersession

/// A relation's declared supersession key: its first `key_len` key columns.
pub(crate) struct SupersedeRule {
    pub(crate) key_len: usize,
}

/// The supersession rule declared for `h`, if any. One catalog point read
/// per `:put` statement — and a single existence probe when no relation in
/// the store declares one.
pub(crate) fn supersede_rule(
    tx: &SessionTx<'_>,
    h: &RelationHandle,
) -> Result<Option<SupersedeRule>> {
    if h.is_temp || h.name.contains(':') {
        return Ok(None);
    }
    let Some(cat) = catalog_for_read(tx, SUPERSEDE_CATALOG, &supersede_meta())? else {
        return Ok(None);
    };
    let Some(row) = cat.get(tx, &[DataValue::from(h.id.0 as i64)])? else {
        return Ok(None);
    };
    let cols = match &row[2] {
        DataValue::List(l) => l.clone(),
        _ => bail!("corrupt {SUPERSEDE_CATALOG} row for relation {}", h.name),
    };
    let matches = cols.len() <= plain_len(h)
        && cols
            .iter()
            .zip(h.metadata.keys.iter())
            .all(|(c, k)| c.get_str() == Some(&k.name as &str));
    ensure!(
        matches,
        "the supersession key recorded for relation {} no longer matches its schema; \
         re-declare it with ::supersede create",
        h.name
    );
    Ok(Some(SupersedeRule {
        key_len: cols.len(),
    }))
}

#[derive(Debug, Error, Diagnostic)]
#[error("conflicting facts for one supersession key of {relation} at the same valid time: {key:?}")]
#[diagnostic(code(eval::supersede_same_instant), help("{help}"))]
struct SupersedeSameInstant {
    relation: String,
    key: Vec<DataValue>,
    help: &'static str,
}

/// One resolved record of a value-key at one valid time.
#[derive(Clone)]
struct Ev {
    assert: bool,
    vals: Vec<DataValue>,
    /// written by this statement or still pending in this transaction
    fresh: bool,
}

/// The valid-time timeline of one supersession group, per value-key.
#[derive(Default)]
struct Timeline {
    by_vk: BTreeMap<Tuple, BTreeMap<i64, Ev>>,
}

impl Timeline {
    fn state(&self, vk: &Tuple, t: i64) -> Option<(i64, &Ev)> {
        self.by_vk
            .get(vk)
            .and_then(|evs| evs.range(..=t).next_back())
            .map(|(ts, e)| (*ts, e))
    }
    /// Record a newer belief about (vk, t). On a vt-only relation an
    /// assertion shadows a retraction at the same instant (the key order puts
    /// it first), so a stored assert is never replaced by a retract.
    fn record(&mut self, vk: Tuple, t: i64, ev: Ev, bitemporal: bool) {
        let evs = self.by_vk.entry(vk).or_default();
        if !bitemporal && !ev.assert {
            if let Some(old) = evs.get(&t) {
                if old.assert {
                    return;
                }
            }
        }
        evs.insert(t, ev);
    }
}

/// Rewrite one `:put` batch (rows in the canonical layout: plain keys, vt,
/// values) under a supersession rule. Returns `(written, generated)`: the
/// input rows to write (minus no-op re-assertions unless `keep_redundant`,
/// which `:insert` sets so that its existence check still fires) and the
/// retractions that end superseded rows.
pub(crate) fn apply_supersession(
    tx: &SessionTx<'_>,
    h: &RelationHandle,
    rule: &SupersedeRule,
    rows: Vec<Tuple>,
    keep_redundant: bool,
) -> Result<(Vec<Tuple>, Vec<Tuple>)> {
    let shape = axes(h);
    ensure!(
        matches!(shape, Axes::Vt | Axes::Bitemporal),
        "relation {} declares a supersession key but has no valid-time axis",
        h.name
    );
    let bitemporal = shape == Axes::Bitemporal;
    let p = plain_len(h);
    let f = rule.key_len;

    let mut groups: BTreeMap<Tuple, Vec<Tuple>> = BTreeMap::new();
    for row in rows {
        groups.entry(row[..f].to_vec()).or_default().push(row);
    }

    let mut out = Out {
        written: vec![],
        generated: vec![],
        keep_redundant,
    };
    for (fk, mut incoming) in groups {
        // ascending valid time, stable: a batch is replayed in time order,
        // and two writes of one key at one instant keep last-write-wins
        let mut keyed = Vec::with_capacity(incoming.len());
        for r in incoming.drain(..) {
            keyed.push((vt_of(&r[p])?.0, r));
        }
        keyed.sort_by_key(|(t, _)| *t);

        let mut tl = load_group(tx, h, &fk, f, p, bitemporal)?;
        for (_, r) in keyed {
            supersede_one(h, &mut tl, &fk, f, p, bitemporal, r, &mut out)?;
        }
    }
    Ok((out.written, out.generated))
}

struct Out {
    written: Vec<Tuple>,
    generated: Vec<Tuple>,
    keep_redundant: bool,
}

fn load_group(
    tx: &SessionTx<'_>,
    h: &RelationHandle,
    fk: &Tuple,
    f: usize,
    p: usize,
    bitemporal: bool,
) -> Result<Timeline> {
    let mut tl = Timeline::default();
    if bitemporal {
        // resolve-groups at current belief: one record per (key, vt-group)
        for row in h.bitemporal_scan_prefix(tx, fk, None, MAX_VALIDITY_TS) {
            let row = row?;
            let (t, a) = vt_of(&row[p])?;
            let ev = Ev {
                assert: a,
                vals: row[p + 2..].to_vec(),
                fresh: false,
            };
            tl.record(row[f..p].to_vec(), t, ev, true);
        }
        // writes buffered earlier in this transaction are newer beliefs
        for w in tx.pending_tt_writes.iter() {
            if w.handle.id != h.id || w.is_retract {
                continue;
            }
            for r in w.rows.iter().filter(|r| r[..f] == fk[..]) {
                let (t, a) = vt_of(&r[p])?;
                let ev = Ev {
                    assert: a,
                    vals: r[p + 1..].to_vec(),
                    fresh: true,
                };
                tl.record(r[f..p].to_vec(), t, ev, true);
            }
        }
    } else {
        for row in h.scan_prefix(tx, fk) {
            let row = row?;
            let (t, a) = vt_of(&row[p])?;
            let ev = Ev {
                assert: a,
                vals: row[p + 1..].to_vec(),
                fresh: false,
            };
            tl.record(row[f..p].to_vec(), t, ev, false);
        }
    }
    Ok(tl)
}

fn retract_row(fk: &Tuple, vk: &Tuple, t: i64, vals: &[DataValue]) -> Tuple {
    let mut row = fk.clone();
    row.extend(vk.iter().cloned());
    row.push(vld(t, false));
    row.extend(vals.iter().cloned());
    row
}

#[allow(clippy::too_many_arguments)]
fn supersede_one(
    h: &RelationHandle,
    tl: &mut Timeline,
    fk: &Tuple,
    f: usize,
    p: usize,
    bitemporal: bool,
    r: Tuple,
    out: &mut Out,
) -> Result<()> {
    let vk: Tuple = r[f..p].to_vec();
    let (t, is_assert) = vt_of(&r[p])?;
    let vals = r[p + 1..].to_vec();
    if !is_assert {
        // an explicit retraction ends only its own key: nothing to supersede
        out.written.push(r);
        tl.record(
            vk,
            t,
            Ev {
                assert: false,
                vals,
                fresh: true,
            },
            bitemporal,
        );
        return Ok(());
    }

    // idempotence: re-asserting the fact that is already current at t (same
    // values) writes nothing — no new version, no new history entry
    let redundant = matches!(tl.state(&vk, t), Some((_, e)) if e.assert && e.vals == vals);
    if !redundant || out.keep_redundant {
        out.written.push(r);
        tl.record(
            vk.clone(),
            t,
            Ev {
                assert: true,
                vals: vals.clone(),
                fresh: true,
            },
            bitemporal,
        );
    }

    // supersession: every other value current at t ends at t
    let mut ended = vec![];
    for (y, evs) in tl.by_vk.iter() {
        if *y == vk {
            continue;
        }
        if let Some((ts, e)) = evs.range(..=t).next_back() {
            if e.assert {
                ended.push((y.clone(), *ts, e.vals.clone(), e.fresh));
            }
        }
    }
    for (y, ts, yvals, fresh) in ended {
        if ts == t {
            let mut key = fk.clone();
            key.extend(y.iter().cloned());
            if fresh {
                bail!(SupersedeSameInstant {
                    relation: h.name.to_string(),
                    key,
                    help: "one transaction asserted two different values for this key at \
                           one valid time; write one of them, or give them distinct valid times",
                });
            }
            if !bitemporal {
                bail!(SupersedeSameInstant {
                    relation: h.name.to_string(),
                    key,
                    help: "on a relation without TxTime a retraction cannot override an \
                           assertion at the same instant; declare a `tt: TxTime` column to \
                           record corrections, or :rm the wrong row (with its exact valid time) first",
                });
            }
        }
        out.generated.push(retract_row(fk, &y, t, &yvals));
        tl.record(
            y,
            t,
            Ev {
                assert: false,
                vals: yvals,
                fresh: true,
            },
            bitemporal,
        );
    }

    // late arrival: a fact older than the group's next change ends at that
    // change instead of ending the newer fact
    let next = tl
        .by_vk
        .iter()
        .filter(|(y, _)| **y != vk)
        .filter_map(|(_, evs)| {
            evs.range((Bound::Excluded(t), Bound::Unbounded))
                .find(|(_, e)| e.assert)
                .map(|(ts, _)| *ts)
        })
        .min();
    if let Some(tn) = next {
        let still_current = match tl.state(&vk, tn) {
            Some((ts, e)) if e.assert && ts < tn => Some(e.vals.clone()),
            _ => None,
        };
        if let Some(xvals) = still_current {
            out.generated.push(retract_row(fk, &vk, tn, &xvals));
            tl.record(
                vk,
                tn,
                Ev {
                    assert: false,
                    vals: xvals,
                    fresh: true,
                },
                bitemporal,
            );
        }
    }
    Ok(())
}

fn supersede_create(tx: &mut SessionTx<'_>, rel: &Symbol, cols: &[Symbol]) -> Result<NamedRows> {
    let h = user_relation(tx, &rel.name, "::supersede create")?;
    ensure!(
        h.access_level >= AccessLevel::Protected,
        "relation {} is {}; supersession needs write access",
        h.name,
        h.access_level
    );
    ensure!(
        matches!(axes(&h), Axes::Vt | Axes::Bitemporal),
        "::supersede requires a valid-time axis: relation {} must end its key with \
         `vt: Validity` (optionally followed by `tt: TxTime`)",
        h.name
    );
    let p = plain_len(&h);
    let expected = h.metadata.keys[..p]
        .iter()
        .map(|c| c.name.to_string())
        .collect_vec();
    let given = cols.iter().map(|c| c.name.to_string()).collect_vec();
    ensure!(
        given.len() <= p && expected[..given.len()] == given[..],
        "a supersession key must be a leading prefix of the non-temporal key columns of {} \
         (in order: {}), got [{}]",
        h.name,
        expected.join(", "),
        given.join(", ")
    );
    let cat = catalog_for_write(tx, SUPERSEDE_CATALOG, supersede_meta())?;
    let row = vec![
        DataValue::from(h.id.0 as i64),
        DataValue::from(h.name.as_str()),
        DataValue::List(given.iter().map(|s| DataValue::from(s.as_str())).collect()),
    ];
    catalog_put(tx, &cat, &row)?;
    Ok(status_ok())
}

fn supersede_drop(tx: &mut SessionTx<'_>, rel: &Symbol) -> Result<NamedRows> {
    let h = user_relation(tx, &rel.name, "::supersede drop")?;
    let cat = catalog_for_write(tx, SUPERSEDE_CATALOG, supersede_meta())?;
    let key = vec![DataValue::from(h.id.0 as i64)];
    ensure!(
        cat.exists(tx, &key)?,
        "relation {} has no supersession key",
        h.name
    );
    tx.store_tx
        .del(&cat.encode_key_for_store(&key, SourceSpan::default())?)?;
    Ok(status_ok())
}

fn supersede_list(tx: &SessionTx<'_>) -> Result<NamedRows> {
    let mut rows = vec![];
    if let Some(cat) = catalog_for_read(tx, SUPERSEDE_CATALOG, &supersede_meta())? {
        let rels = relations_by_id(tx)?;
        for row in cat.scan_all(tx) {
            let row = row?;
            let id = row[0].get_int().unwrap_or(-1) as u64;
            if let Some(h) = rels.get(&id) {
                rows.push(vec![DataValue::from(h.name.as_str()), row[2].clone()]);
            }
        }
    }
    Ok(NamedRows::new(
        vec!["relation".to_string(), "key_cols".to_string()],
        rows,
    ))
}

/// `::changes`: per key, the valid-time transitions in `(from, to]` —
/// `added` (not current → current), `ended` (current → not current) and
/// `changed` (current → current with different values). Redundant
/// re-assertions and retractions of facts that were never current (e.g. a
/// corrected typo) produce no row.
fn changes(
    tx: &SessionTx<'_>,
    rel: &Symbol,
    from: ValidityTs,
    to: ValidityTs,
    as_of: Option<ValidityTs>,
    prefix: &[DataValue],
) -> Result<NamedRows> {
    let h = tx.get_relation_for_read(&rel.name, "::changes")?;
    let shape = axes(&h);
    ensure!(
        matches!(shape, Axes::Vt | Axes::Bitemporal),
        "::changes requires a valid-time axis; relation {} has none",
        h.name
    );
    // validates as_of against the axes and the ::history_gc floor
    h.resolve_temporal_read(None, as_of, rel.span)?;
    let tt = as_of.unwrap_or(MAX_VALIDITY_TS);
    let p = plain_len(&h);
    let bitemporal = shape == Axes::Bitemporal;
    ensure!(
        prefix.len() <= p,
        "::changes prefix is longer than the {} plain key column(s) of {}",
        p,
        h.name
    );
    let prefix: Tuple = prefix
        .iter()
        .zip(h.metadata.keys.iter())
        .map(|(v, c)| c.typing.coerce(v.clone(), MAX_VALIDITY_TS))
        .try_collect()?;

    let mut headers = h.metadata.keys[..p]
        .iter()
        .map(|c| c.name.to_string())
        .collect_vec();
    headers.push("change".to_string());
    headers.push("at".to_string());
    headers.extend(h.metadata.non_keys.iter().map(|c| c.name.to_string()));
    ensure!(
        headers.iter().all_unique(),
        "::changes output columns `change`/`at` collide with a column of {}",
        h.name
    );

    // per plain key: ts -> (assert, vals)
    let mut per_key: BTreeMap<Tuple, BTreeMap<i64, (bool, Tuple)>> = BTreeMap::new();
    let vals_at = if bitemporal { p + 2 } else { p + 1 };
    let mut ingest = |row: Tuple| -> Result<()> {
        let (t, a) = vt_of(&row[p])?;
        let evs = per_key.entry(row[..p].to_vec()).or_default();
        if !bitemporal && !a && matches!(evs.get(&t), Some((true, _))) {
            return Ok(()); // assert shadows retract at one instant
        }
        evs.insert(t, (a, row[vals_at..].to_vec()));
        Ok(())
    };
    if bitemporal {
        for row in h.bitemporal_scan_prefix(tx, &prefix, None, tt) {
            ingest(row?)?;
        }
    } else {
        for row in h.scan_prefix(tx, &prefix) {
            ingest(row?)?;
        }
    }

    let (lo, hi) = (from.0 .0, to.0 .0);
    let mut out: Vec<(i64, Tuple)> = vec![];
    for (key, evs) in per_key {
        let mut cur: Option<Tuple> = match evs.range(..=lo).next_back() {
            Some((_, (true, v))) => Some(v.clone()),
            _ => None,
        };
        for (t, (a, vals)) in evs.range((Bound::Excluded(lo), Bound::Included(hi))) {
            let change = match (&cur, a) {
                (None, true) => Some(("added", vals.clone())),
                (Some(old), false) => Some(("ended", old.clone())),
                (Some(old), true) if old != vals => Some(("changed", vals.clone())),
                _ => None,
            };
            cur = if *a { Some(vals.clone()) } else { None };
            if let Some((kind, shown)) = change {
                let mut row = key.clone();
                row.push(DataValue::from(kind));
                row.push(DataValue::from(*t));
                row.extend(shown);
                out.push((*t, row));
            }
        }
    }
    out.sort_by_key(|(t, _)| *t);
    Ok(NamedRows::new(
        headers,
        out.into_iter().map(|(_, r)| r).collect(),
    ))
}

// ---------------------------------------------------------------------------
// E2: provenance

#[derive(Clone)]
struct Link {
    link: RelationHandle,
    derived: RelationHandle,
    source: RelationHandle,
    /// plain-key widths of derived and source
    dp: usize,
    sp: usize,
}

fn load_links(tx: &SessionTx<'_>) -> Result<Vec<Link>> {
    let Some(cat) = catalog_for_read(tx, PROVENANCE_CATALOG, &provenance_meta())? else {
        return Ok(vec![]);
    };
    let rels = relations_by_id(tx)?;
    let mut links = vec![];
    for row in cat.scan_all(tx) {
        let row = row?;
        let id = |i: usize| row[i].get_int().unwrap_or(-1) as u64;
        // a dropped relation retires its declarations (ids are never reused)
        let (Some(link), Some(derived), Some(source)) =
            (rels.get(&id(0)), rels.get(&id(2)), rels.get(&id(3)))
        else {
            continue;
        };
        links.push(Link {
            dp: plain_len(derived),
            sp: plain_len(source),
            link: link.clone(),
            derived: derived.clone(),
            source: source.clone(),
        });
    }
    Ok(links)
}

fn provenance_create(
    tx: &mut SessionTx<'_>,
    link: &Symbol,
    derived: &str,
    source: &str,
) -> Result<NamedRows> {
    let l = user_relation(tx, &link.name, "::provenance create")?;
    let d = user_relation(tx, derived, "::provenance create")?;
    let s = user_relation(tx, source, "::provenance create")?;
    ensure!(
        axes(&l) == Axes::Plain,
        "link relation {} must have no temporal axis (no Validity/TxTime key column)",
        l.name
    );
    ensure!(
        axes(&d) != Axes::Plain,
        "derived relation {} needs a temporal axis (`vt: Validity` and/or `tt: TxTime`) \
         so that retracting its sources can end its validity",
        d.name
    );
    let (dp, sp) = (plain_len(&d), plain_len(&s));
    let want = d.metadata.keys[..dp]
        .iter()
        .chain(s.metadata.keys[..sp].iter())
        .collect_vec();
    let fits = l.metadata.keys.len() == want.len()
        && l.metadata
            .keys
            .iter()
            .zip(want.iter())
            .all(|(a, b)| a.typing.coltype == b.typing.coltype);
    ensure!(
        fits,
        "link relation {} must be keyed by {}'s key columns followed by {}'s key columns \
         (types: [{}]); value columns are free",
        l.name,
        d.name,
        s.name,
        want.iter()
            .map(|c| format!("{}: {}", c.name, c.typing))
            .join(", ")
    );
    let cat = catalog_for_write(tx, PROVENANCE_CATALOG, provenance_meta())?;
    let key = vec![DataValue::from(l.id.0 as i64)];
    ensure!(
        !cat.exists(tx, &key)?,
        "{} is already a provenance link; ::provenance drop it first",
        l.name
    );
    let row = vec![
        DataValue::from(l.id.0 as i64),
        DataValue::from(l.name.as_str()),
        DataValue::from(d.id.0 as i64),
        DataValue::from(s.id.0 as i64),
    ];
    catalog_put(tx, &cat, &row)?;
    Ok(status_ok())
}

fn provenance_drop(tx: &mut SessionTx<'_>, link: &Symbol) -> Result<NamedRows> {
    let l = user_relation(tx, &link.name, "::provenance drop")?;
    let cat = catalog_for_write(tx, PROVENANCE_CATALOG, provenance_meta())?;
    let key = vec![DataValue::from(l.id.0 as i64)];
    ensure!(cat.exists(tx, &key)?, "{} is not a provenance link", l.name);
    tx.store_tx
        .del(&cat.encode_key_for_store(&key, SourceSpan::default())?)?;
    Ok(status_ok())
}

fn provenance_list(tx: &SessionTx<'_>) -> Result<NamedRows> {
    let rows = load_links(tx)?
        .into_iter()
        .map(|l| {
            vec![
                DataValue::from(l.link.name.as_str()),
                DataValue::from(l.derived.name.as_str()),
                DataValue::from(l.source.name.as_str()),
            ]
        })
        .collect_vec();
    Ok(NamedRows::new(
        vec![
            "link".to_string(),
            "derived".to_string(),
            "source".to_string(),
        ],
        rows,
    ))
}

/// Full link keys `[derived key…, source key…]` naming `skey` as source. A
/// secondary index on the link relation that leads with the source columns
/// turns the scan into a prefix seek.
fn links_from_source(tx: &SessionTx<'_>, l: &Link, skey: &Tuple) -> Result<Vec<Tuple>> {
    let nk = l.dp + l.sp;
    let want = (l.dp..nk).collect_vec();
    let mut out = vec![];
    for (idx, mapping) in l.link.indices.values() {
        if mapping.len() >= l.sp && mapping[..l.sp] == want[..] {
            for r in idx.scan_prefix(tx, skey) {
                let r = r?;
                let mut orig = vec![DataValue::Null; nk];
                for (i, m) in mapping.iter().enumerate() {
                    if *m < nk && i < r.len() {
                        orig[*m] = r[i].clone();
                    }
                }
                out.push(orig);
            }
            return Ok(out);
        }
    }
    for r in l.link.scan_all(tx) {
        let r = r?;
        if r[l.dp..nk] == skey[..] {
            out.push(r[..nk].to_vec());
        }
    }
    Ok(out)
}

/// Every (source relation, source key) still linked to derived key `dkey`.
fn sources_of(
    tx: &SessionTx<'_>,
    links: &[Link],
    derived_id: RelationId,
    dkey: &Tuple,
) -> Result<Vec<(RelationHandle, Tuple)>> {
    let mut out = vec![];
    for l in links.iter().filter(|l| l.derived.id == derived_id) {
        for r in l.link.scan_prefix(tx, dkey) {
            let r = r?;
            out.push((l.source.clone(), r[l.dp..l.dp + l.sp].to_vec()));
        }
    }
    Ok(out)
}

/// Is `key` current in `h` at valid time `vt` (current transaction-time
/// belief)? Plain relations: the row exists.
fn alive(tx: &SessionTx<'_>, h: &RelationHandle, key: &Tuple, vt: ValidityTs) -> Result<bool> {
    Ok(match axes(h) {
        Axes::Plain => h.exists(tx, key)?,
        Axes::Vt => h
            .skip_scan_prefix(tx, key, vt)
            .next()
            .transpose()?
            .is_some(),
        Axes::TtOnly => h
            .skip_scan_prefix(tx, key, MAX_VALIDITY_TS)
            .next()
            .transpose()?
            .is_some(),
        Axes::Bitemporal => h
            .bitemporal_scan_prefix(tx, key, Some(vt), MAX_VALIDITY_TS)
            .next()
            .transpose()?
            .is_some(),
    })
}

/// Write canonical-layout rows through the ordinary `:put` (or `:rm`) path,
/// so B-tree/HNSW/FTS/LSH maintenance, triggers and TxTime buffering are
/// exactly those of a user write.
fn write_rows<'s, S: Storage<'s>>(
    db: &Db<S>,
    tx: &mut SessionTx<'_>,
    h: &RelationHandle,
    rows: Vec<Tuple>,
    op: RelationOp,
    cur_vld: ValidityTs,
) -> Result<()> {
    let (mut meta, mut headers) = canonical_input(h);
    if op == RelationOp::Rm {
        let p = plain_len(h);
        meta.metadata.keys.truncate(p);
        meta.key_bindings.truncate(p);
        meta.metadata.non_keys.clear();
        meta.dep_bindings.clear();
        headers.truncate(p);
    }
    let targets = BTreeSet::new();
    let mut collector = CallbackCollector::new();
    let cleanups = tx.execute_relation(
        db,
        rows.into_iter(),
        op,
        &meta,
        &headers,
        cur_vld,
        &targets,
        &mut collector,
        true,
        "",
    )?;
    for (lower, upper) in cleanups {
        tx.store_tx.del_range_from_persisted(&lower, &upper)?;
    }
    Ok(())
}

/// End the validity of `key` at `vt`. Returns whether a retraction was
/// written (false: the key was not current at `vt`).
fn retract_key<'s, S: Storage<'s>>(
    db: &Db<S>,
    tx: &mut SessionTx<'_>,
    h: &RelationHandle,
    key: &Tuple,
    vt: ValidityTs,
) -> Result<bool> {
    let p = key.len();
    match axes(h) {
        Axes::Plain => bail!(
            "relation {} has no temporal axis, so it cannot be retracted; \
             use ::erase to delete its rows",
            h.name
        ),
        Axes::TtOnly => {
            if !alive(tx, h, key, vt)? {
                return Ok(false);
            }
            write_rows(db, tx, h, vec![key.clone()], RelationOp::Rm, vt)?;
            Ok(true)
        }
        Axes::Vt | Axes::Bitemporal => {
            let bitemporal = axes(h) == Axes::Bitemporal;
            let cur = if bitemporal {
                h.bitemporal_scan_prefix(tx, key, Some(vt), MAX_VALIDITY_TS)
                    .next()
                    .transpose()?
            } else {
                h.skip_scan_prefix(tx, key, vt).next().transpose()?
            };
            let Some(row) = cur else {
                return Ok(false);
            };
            if !bitemporal && vt_of(&row[p])?.0 == vt.0 .0 {
                bail!(
                    "cannot retract {:?} in {} at the very instant it was asserted: on a \
                     relation without TxTime the assertion shadows the retraction; pass a later `@` time",
                    key,
                    h.name
                );
            }
            let vals_at = if bitemporal { p + 2 } else { p + 1 };
            let mut out = key.clone();
            out.push(vld(vt.0 .0, false));
            out.extend(row[vals_at..].iter().cloned());
            write_rows(db, tx, h, vec![out], RelationOp::Put, vt)?;
            Ok(true)
        }
    }
}

/// The salted namespace that hashes erased keys into audit markers (shared
/// with `::evict`; generated once per store under a locking read so two
/// racing first erasures cannot mint two salts).
pub(crate) fn evict_key_namespace(tx: &mut SessionTx<'_>) -> Result<uuid::Uuid> {
    let salt_key =
        vec![DataValue::Null, DataValue::from("EVICT_SALT")].encode_as_key(RelationId::SYSTEM);
    let salt = match tx.store_tx.get(&salt_key, true)? {
        Some(s) => s,
        None => {
            let mut buf = [0u8; 16];
            rand::Rng::fill(&mut rand::thread_rng(), &mut buf);
            tx.store_tx.put(&salt_key, &buf)?;
            buf.to_vec()
        }
    };
    Ok(uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, &salt))
}

/// Hard-deletion bookkeeping: one audit marker per erased key in
/// `kleos_evict_audit` (salted key hash, never the key), all stamped with
/// one burned transaction time.
struct Auditor {
    audit: RelationHandle,
    ns: uuid::Uuid,
    tt: i64,
}

impl Auditor {
    fn new<'s, S: Storage<'s>>(db: &'s Db<S>, tx: &mut SessionTx<'_>) -> Result<Self> {
        let audit = db.tt_evict_audit_relation(tx)?;
        tx.mark_dirty(&audit);
        let ns = evict_key_namespace(tx)?;
        let tt = tx.tt_clock.advance();
        // route the commit through the tt path so the burned tt is covered by
        // the persisted high-water mark
        tx.tt_hwm_dirty = true;
        Ok(Self { audit, ns, tt })
    }

    fn record(
        &self,
        tx: &mut SessionTx<'_>,
        h: &RelationHandle,
        key: &Tuple,
        n: usize,
    ) -> Result<()> {
        let enc = key.to_vec().encode_as_key(h.id);
        let row = vec![
            DataValue::from(h.name.as_str()),
            DataValue::from(uuid::Uuid::new_v5(&self.ns, &enc).to_string().as_str()),
            DataValue::from(self.tt),
            DataValue::from(n as i64),
        ];
        catalog_put(tx, &self.audit, &row)
    }
}

/// Physically delete every row (all valid times, all transaction times) of
/// `key`, keeping B-tree and HNSW/FTS/LSH indexes consistent.
fn erase_key(
    tx: &mut SessionTx<'_>,
    h: &RelationHandle,
    key: &Tuple,
    auditor: &Auditor,
) -> Result<usize> {
    if h.access_level < AccessLevel::Normal {
        bail!(InsufficientAccessLevel(
            h.name.to_string(),
            "erasure".to_string(),
            h.access_level
        ));
    }
    if h.has_txtime() && tx.pending_tt_writes.iter().any(|w| w.handle.id == h.id) {
        bail!(
            "relation {} has pending transaction-time writes in this transaction; they would \
             be stamped after the erasure and resurrect the erased keys — commit them first",
            h.name
        );
    }
    tx.mark_dirty(h);
    let nk = h.metadata.keys.len();
    let rows: Vec<Tuple> = h.scan_prefix(tx, key).try_collect()?;
    let mut search = ImportSearchIndexes::new(tx, h)?;
    for r in &rows {
        write_import_tuple(tx, h, &mut search, r[..nk].to_vec(), true)?;
    }
    auditor.record(tx, h, key, rows.len())?;
    Ok(rows.len())
}

/// `::retract` (soft) / `::erase` (hard) with provenance cascade.
#[allow(clippy::too_many_arguments)]
fn remove_source<'s, S: Storage<'s>>(
    db: &'s Db<S>,
    tx: &mut SessionTx<'_>,
    rel: &Symbol,
    keys: &[Vec<DataValue>],
    vt: ValidityTs,
    hard: bool,
    skip_locking: bool,
) -> Result<NamedRows> {
    let root = user_relation(tx, &rel.name, if hard { "::erase" } else { "::retract" })?;
    let links = load_links(tx)?;

    // lock the closure of relations the cascade can reach
    let mut ids = BTreeSet::from([root.id]);
    let mut names: BTreeSet<SmartString<LazyCompact>> =
        BTreeSet::from([root.name.clone(), SmartString::from(PROVENANCE_CATALOG)]);
    if hard {
        names.insert(SmartString::from(EVICT_AUDIT));
    }
    loop {
        let mut grew = false;
        for l in &links {
            if ids.contains(&l.source.id) {
                names.insert(l.link.name.clone());
                names.insert(l.derived.name.clone());
                grew |= ids.insert(l.derived.id);
            }
        }
        if !grew {
            break;
        }
    }
    let locks = if skip_locking {
        vec![]
    } else {
        db.obtain_relation_locks(names.iter())
    };
    let _guards = locks.iter().map(|l| l.read().unwrap()).collect_vec();

    let auditor = if hard {
        Some(Auditor::new(db, tx)?)
    } else {
        None
    };
    let mut out: Vec<Tuple> = vec![];
    let mut emit = |depth: usize, h: &RelationHandle, key: &Tuple, action: &str| {
        out.push(vec![
            DataValue::from(depth as i64),
            DataValue::from(h.name.as_str()),
            DataValue::List(key.clone()),
            DataValue::from(action),
        ]);
    };
    // keys retracted or erased by this operation: not live sources any more,
    // even where the write is buffered until commit (TxTime)
    let mut dead: BTreeSet<(RelationId, Tuple)> = BTreeSet::new();
    // (relation, key, depth, erased?)
    let mut queue: VecDeque<(RelationHandle, Tuple, usize, bool)> = VecDeque::new();

    for k in keys {
        let k = coerce_key(&root, k, if hard { "::erase" } else { "::retract" })?;
        if !dead.insert((root.id, k.clone())) {
            continue;
        }
        let action = match &auditor {
            Some(a) => {
                if erase_key(tx, &root, &k, a)? > 0 {
                    "erased"
                } else {
                    "absent"
                }
            }
            None => {
                if retract_key(db, tx, &root, &k, vt)? {
                    "retracted"
                } else {
                    "not_current"
                }
            }
        };
        emit(0, &root, &k, action);
        queue.push_back((root.clone(), k, 0, hard));
    }

    while let Some((src, skey, depth, erased)) = queue.pop_front() {
        for l in links.iter().filter(|l| l.source.id == src.id) {
            let link_keys = links_from_source(tx, l, &skey)?;
            if erased {
                // the erased source must not survive in lineage either
                tx.mark_dirty(&l.link);
                let mut search = ImportSearchIndexes::new(tx, &l.link)?;
                for lk in &link_keys {
                    write_import_tuple(tx, &l.link, &mut search, lk.clone(), true)?;
                }
            }
            for lk in link_keys {
                let d: Tuple = lk[..l.dp].to_vec();
                if dead.contains(&(l.derived.id, d.clone())) {
                    continue;
                }
                let mut any_source = false;
                let mut live_source = false;
                for (sh, s) in sources_of(tx, &links, l.derived.id, &d)? {
                    any_source = true;
                    if !dead.contains(&(sh.id, s.clone())) && alive(tx, &sh, &s, vt)? {
                        live_source = true;
                        break;
                    }
                }
                if live_source {
                    if erased {
                        emit(depth + 1, &l.derived, &d, "unlinked");
                    }
                    continue;
                }
                dead.insert((l.derived.id, d.clone()));
                let (action, d_erased) = match (&auditor, erased && !any_source) {
                    (Some(a), true) => {
                        erase_key(tx, &l.derived, &d, a)?;
                        ("erased", true)
                    }
                    _ => {
                        if retract_key(db, tx, &l.derived, &d, vt)? {
                            ("retracted", false)
                        } else {
                            ("not_current", false)
                        }
                    }
                };
                // a derived row that already was not current changes nothing:
                // not reported, but its own dependents are still checked
                if action != "not_current" {
                    emit(depth + 1, &l.derived, &d, action);
                }
                queue.push_back((l.derived.clone(), d, depth + 1, d_erased));
            }
        }
    }
    Ok(NamedRows::new(
        vec![
            "depth".to_string(),
            "relation".to_string(),
            "key".to_string(),
            "action".to_string(),
        ],
        out,
    ))
}

/// `::lineage`: the transitive sources of each key, breadth first, with
/// whether each is current at `vt`.
fn lineage(
    tx: &SessionTx<'_>,
    rel: &Symbol,
    keys: &[Vec<DataValue>],
    vt: ValidityTs,
) -> Result<NamedRows> {
    let root = tx.get_relation_for_read(&rel.name, "::lineage")?;
    let links = load_links(tx)?;
    let mut out = vec![];
    let mut seen: BTreeSet<(RelationId, Tuple)> = BTreeSet::new();
    let mut queue: VecDeque<(RelationHandle, Tuple, usize)> = VecDeque::new();
    for k in keys {
        let k = coerce_key(&root, k, "::lineage")?;
        if seen.insert((root.id, k.clone())) {
            out.push(vec![
                DataValue::from(0),
                DataValue::from(root.name.as_str()),
                DataValue::List(k.clone()),
                DataValue::Null,
                DataValue::from(alive(tx, &root, &k, vt)?),
            ]);
            queue.push_back((root.clone(), k, 0));
        }
    }
    while let Some((h, key, depth)) = queue.pop_front() {
        for l in links.iter().filter(|l| l.derived.id == h.id) {
            for r in l.link.scan_prefix(tx, &key) {
                let r = r?;
                let s: Tuple = r[l.dp..l.dp + l.sp].to_vec();
                if !seen.insert((l.source.id, s.clone())) {
                    continue;
                }
                out.push(vec![
                    DataValue::from((depth + 1) as i64),
                    DataValue::from(l.source.name.as_str()),
                    DataValue::List(s.clone()),
                    DataValue::from(l.link.name.as_str()),
                    DataValue::from(alive(tx, &l.source, &s, vt)?),
                ]);
                queue.push_back((l.source.clone(), s, depth + 1));
            }
        }
    }
    Ok(NamedRows::new(
        vec![
            "depth".to_string(),
            "relation".to_string(),
            "key".to_string(),
            "via".to_string(),
            "current".to_string(),
        ],
        out,
    ))
}

/// Entry point from `run_sys_op_with_tx`.
pub(crate) fn run_memory_op<'s, S: Storage<'s>>(
    db: &'s Db<S>,
    tx: &mut SessionTx<'_>,
    op: &MemoryOp,
    read_only: bool,
    skip_locking: bool,
) -> Result<NamedRows> {
    let mutating = !op.write_targets().is_empty();
    if read_only && mutating {
        bail!("Cannot modify agent-memory state in read-only mode");
    }
    let lock_catalog = |name: &str| {
        if skip_locking {
            vec![]
        } else {
            db.obtain_relation_locks(std::iter::once(&SmartString::from(name)))
        }
    };
    match op {
        MemoryOp::SupersedeCreate { rel, cols } => {
            let locks = lock_catalog(SUPERSEDE_CATALOG);
            let _g = locks.iter().map(|l| l.read().unwrap()).collect_vec();
            supersede_create(tx, rel, cols)
        }
        MemoryOp::SupersedeDrop(rel) => {
            let locks = lock_catalog(SUPERSEDE_CATALOG);
            let _g = locks.iter().map(|l| l.read().unwrap()).collect_vec();
            supersede_drop(tx, rel)
        }
        MemoryOp::SupersedeList => supersede_list(tx),
        MemoryOp::Changes {
            rel,
            from,
            to,
            as_of,
            prefix,
        } => changes(tx, rel, *from, *to, *as_of, prefix),
        MemoryOp::ProvenanceCreate {
            link,
            derived,
            source,
        } => {
            let locks = lock_catalog(PROVENANCE_CATALOG);
            let _g = locks.iter().map(|l| l.read().unwrap()).collect_vec();
            provenance_create(tx, link, derived, source)
        }
        MemoryOp::ProvenanceDrop(link) => {
            let locks = lock_catalog(PROVENANCE_CATALOG);
            let _g = locks.iter().map(|l| l.read().unwrap()).collect_vec();
            provenance_drop(tx, link)
        }
        MemoryOp::ProvenanceList => provenance_list(tx),
        MemoryOp::RemoveSource {
            rel,
            keys,
            vt,
            hard,
        } => remove_source(db, tx, rel, keys, *vt, *hard, skip_locking),
        MemoryOp::Lineage { rel, keys, vt } => lineage(tx, rel, keys, *vt),
    }
}
