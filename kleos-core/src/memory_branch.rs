/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Memory branches (docs/specs/memory-branches.md): `fork` (O(1),
//! copy-on-write), `rewind`, `merge` (replay through `remember`), `diff`,
//! `discard` and `branches`.
//!
//! A branch is one row in `kleos_mem_branch`. Its **lineage** is the chain
//! of parents up to a root; each ancestor level is read as of its **cut**
//! (the minimum fork time on the path). A memory id is **owned** by the
//! nearest level with any `kleos_mem` row for it, and every per-memory row
//! (vector, sources, mentions, role, events, expiry) is read from the owner.
//! A write that touches an inherited memory first **copies it up** into the
//! branch, in the same transaction. A plain namespace (no branch row) has a
//! one-level lineage and every query it runs is the one it ran before.

use std::collections::{BTreeMap, BTreeSet};

use itertools::Itertools;
use miette::Result;
use serde_derive::{Deserialize, Serialize};

use super::*;

/// Branches: `{ns => parent, at, created, status}` (`status`: `active` or
/// `discarded`).
pub const MEM_BRANCH: &str = "kleos_mem_branch";
/// Copy-up log: `{ns, id => from, at}`.
pub const MEM_COPIED: &str = "kleos_mem_copied";
/// Merge log: `{into, stmt => from, at, action, ids}`.
pub const MEM_MERGE: &str = "kleos_mem_merge";

/// Lineage depth limit (levels above the namespace itself).
pub(crate) const MAX_DEPTH: usize = 16;

// Parameter names for the per-level rules: `$bl{i}` the level's namespace,
// `$bt{i}` its read time; `$cu_*{i}` the copy-up groups.
const BL: [&str; MAX_DEPTH + 1] = [
    "bl0", "bl1", "bl2", "bl3", "bl4", "bl5", "bl6", "bl7", "bl8", "bl9", "bl10", "bl11", "bl12", "bl13", "bl14",
    "bl15", "bl16",
];
const BT: [&str; MAX_DEPTH + 1] = [
    "bt0", "bt1", "bt2", "bt3", "bt4", "bt5", "bt6", "bt7", "bt8", "bt9", "bt10", "bt11", "bt12", "bt13", "bt14",
    "bt15", "bt16",
];
const CU_O: [&str; MAX_DEPTH + 1] = [
    "cu_o0", "cu_o1", "cu_o2", "cu_o3", "cu_o4", "cu_o5", "cu_o6", "cu_o7", "cu_o8", "cu_o9", "cu_o10", "cu_o11",
    "cu_o12", "cu_o13", "cu_o14", "cu_o15", "cu_o16",
];
const CU_I: [&str; MAX_DEPTH + 1] = [
    "cu_i0", "cu_i1", "cu_i2", "cu_i3", "cu_i4", "cu_i5", "cu_i6", "cu_i7", "cu_i8", "cu_i9", "cu_i10", "cu_i11",
    "cu_i12", "cu_i13", "cu_i14", "cu_i15", "cu_i16",
];
const CU_C: [&str; MAX_DEPTH + 1] = [
    "cu_c0", "cu_c1", "cu_c2", "cu_c3", "cu_c4", "cu_c5", "cu_c6", "cu_c7", "cu_c8", "cu_c9", "cu_c10", "cu_c11",
    "cu_c12", "cu_c13", "cu_c14", "cu_c15", "cu_c16",
];

// ------------------------------------------------------------------ lineage

/// One level of a lineage: a namespace and, for ancestors, the time it is
/// read at (the minimum fork time on the path from the branch).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Level {
    pub(crate) ns: String,
    pub(crate) cut: Option<i64>,
}

/// The lineage of a namespace: level 0 is the namespace itself.
#[derive(Debug, Clone)]
pub(crate) struct Scope {
    pub(crate) levels: Vec<Level>,
    /// The namespace is a discarded branch (writes are refused).
    pub(crate) discarded: bool,
}

impl Scope {
    pub(crate) fn plain(ns: &str) -> Self {
        Scope {
            levels: vec![Level {
                ns: ns.to_string(),
                cut: None,
            }],
            discarded: false,
        }
    }

    /// No ancestors: every read is the plain one-namespace query.
    pub(crate) fn is_plain(&self) -> bool {
        self.levels.len() == 1
    }

    pub(crate) fn ns(&self) -> &str {
        &self.levels[0].ns
    }

    /// The plain scope of level `i` (its own rows only).
    pub(crate) fn level(&self, i: usize) -> Scope {
        Scope::plain(&self.levels[i].ns)
    }

    /// Read time of level `i` for an op reading at `t` (`None`: now): the
    /// namespace itself at `t`, an ancestor at `min(t, cut)`.
    pub(crate) fn level_at(&self, i: usize, t: Option<i64>) -> Option<i64> {
        match self.levels[i].cut {
            None => t,
            Some(c) => Some(t.map_or(c, |t| t.min(c))),
        }
    }

    /// Namespaces of the lineage, nearest first.
    pub(crate) fn names(&self) -> Vec<String> {
        self.levels.iter().map(|l| l.ns.clone()).collect()
    }

    /// Rules `{name}[id, ons, cols…]`: the memories of the view valid at `t`
    /// (`None`: now) with the namespace that owns each. One rule per level;
    /// an ancestor's memories are masked when a nearer level has any row
    /// for the id (it was copied up). `seed` (e.g. `f[id, _], `) binds `id`
    /// first, so the rules read only those keys; empty reads the whole view.
    /// Binds `$bl{i}` / `$bt{i}`.
    pub(crate) fn view_rules(
        &self,
        name: &str,
        seed: &str,
        cols: &[&str],
        t: Option<i64>,
        p: &mut Vec<(&'static str, DataValue)>,
    ) -> String {
        let cs = cols.iter().map(|c| format!(", {c}")).join("");
        let mut out = String::new();
        for i in 0..self.levels.len() {
            p.push((BL[i], s(&self.levels[i].ns)));
            let at = match self.level_at(i, t) {
                None => "'NOW'".to_string(),
                Some(x) => {
                    p.push((BT[i], DataValue::from(x)));
                    format!("${}", BT[i])
                }
            };
            let mask: String = (0..i).map(|j| format!(", not *{MEM}{{ns: ${}, id}}", BL[j])).collect();
            out.push_str(&format!(
                "{name}[id, ons{cs}] := {seed}*{MEM}{{ns: ${bl}, id{cs} @ {at}}}, ons = ${bl}{mask}\n",
                bl = BL[i]
            ));
        }
        out
    }
}

// ------------------------------------------------------------------ options and results

/// Options of [`Db::memory_fork`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ForkOpts {
    /// The namespace to fork; default `default`.
    pub from: Option<String>,
    /// The new branch (must be unused).
    pub to: String,
    /// Fork time (µs): the branch sees `from` as it was then. Default now;
    /// never in the future.
    pub at: Option<i64>,
}

/// Result of [`Db::memory_fork`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ForkResult {
    /// The new branch.
    pub ns: String,
    /// Its parent.
    pub parent: String,
    /// Fork time (µs).
    pub at: i64,
    /// Levels above the branch.
    pub depth: usize,
    /// `[namespace, cut]` from the branch up to the root (cut null for the
    /// branch itself).
    pub lineage: Vec<(String, Option<i64>)>,
}

/// Options of [`Db::memory_rewind`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RewindOpts {
    /// Namespace; default `default`.
    pub ns: Option<String>,
    /// The valid time to put the namespace back to (µs).
    pub to: i64,
    /// Valid time of the rewind's own rows (µs); default now.
    pub at: Option<i64>,
}

/// Result of [`Db::memory_rewind`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RewindResult {
    pub ns: String,
    pub to: i64,
    pub at: i64,
    /// Memories current before and not valid at `to`: retracted.
    pub retracted: Vec<String>,
    /// Memories valid at `to` and not current before: asserted again.
    pub restored: Vec<String>,
    /// `rewind {to: undo}` puts the namespace back to its state before this
    /// rewind (history keeps it at `at − 1`).
    pub undo: i64,
}

/// Options of [`Db::memory_merge`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MergeOpts {
    /// The branch whose statements are replayed.
    pub from: String,
    /// The namespace they are replayed into; default `from`'s parent.
    pub into: Option<String>,
    /// Preview only: merge into a scratch fork of `into`, diff, discard.
    pub dry_run: bool,
    /// Replay each statement at its original valid time (default true);
    /// false replays at now.
    pub keep_time: bool,
    /// Replay the branch's forgets of inherited memories (default true).
    pub forgets: bool,
}

impl Default for MergeOpts {
    fn default() -> Self {
        MergeOpts {
            from: String::new(),
            into: None,
            dry_run: false,
            keep_time: true,
            forgets: true,
        }
    }
}

/// One statement replayed by a merge.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MergeStatement {
    /// The statement id in the branch.
    pub stmt: String,
    pub text: String,
    /// Valid time it was stored at in the branch.
    pub at: i64,
    /// `added`, `duplicate` or `skipped` (every memory of it was forgotten
    /// in the branch).
    pub action: String,
    /// The memory ids it produced in the target.
    pub ids: Vec<String>,
    pub superseded: Vec<String>,
    pub duplicate_of: Option<String>,
    /// Pending decisions the replay left for review.
    pub tentative: Vec<MemDecision>,
}

/// Result of [`Db::memory_merge`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MergeResult {
    pub from: String,
    pub into: String,
    pub dry_run: bool,
    /// Statements replayed now.
    pub statements: usize,
    /// Statements skipped because an earlier merge into `into` replayed them.
    pub skipped: usize,
    pub replayed: Vec<MergeStatement>,
    /// Inherited memories the branch forgot, retracted in `into`.
    pub forgets: Vec<String>,
    /// Forgotten inherited memories that are not current in `into`.
    pub forgets_skipped: Vec<String>,
    /// With `dry_run`: what the merge would change in `into`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff: Option<DiffResult>,
}

/// Options of [`Db::memory_diff`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiffOpts {
    pub a: String,
    pub b: String,
    /// Read both views at this valid time (µs); default now.
    pub at: Option<i64>,
}

/// One memory in a diff.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DiffMem {
    pub id: String,
    pub text: String,
    pub kind: String,
    pub valid_from: i64,
    /// For removed memories: `superseded` or `forgotten`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// For superseded memories: the memory of `b` that superseded it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by: Option<String>,
}

/// Entity names present in one view only.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DiffEntities {
    pub added: Vec<String>,
    pub removed: Vec<String>,
}

/// A pending `supersedes` decision of `b` that `a` does not have.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DiffPending {
    pub new: String,
    pub candidate: String,
    pub new_text: String,
    pub candidate_text: String,
}

/// Result of [`Db::memory_diff`]: what changes going from `a` to `b`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DiffResult {
    pub a: String,
    pub b: String,
    pub at: i64,
    pub added: Vec<DiffMem>,
    pub removed: Vec<DiffMem>,
    /// Memories current in both (by id, or by normalised text).
    pub same: usize,
    pub pending: Vec<DiffPending>,
    pub entities: DiffEntities,
}

/// Options of [`Db::memory_discard`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiscardOpts {
    /// The branch to drop.
    pub ns: Option<String>,
    /// Also discard its child branches (deepest first).
    pub cascade: bool,
}

/// Result of [`Db::memory_discard`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiscardResult {
    pub ns: String,
    /// Every branch discarded (children first).
    pub discarded: Vec<String>,
    /// Memories deleted.
    pub memories: usize,
}

/// Options of [`Db::memory_branches`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BranchesOpts {
    /// Only this namespace's lineage and descendants.
    pub ns: Option<String>,
    /// Include discarded branches (default false).
    pub all: bool,
}

/// One branch.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BranchInfo {
    pub ns: String,
    pub parent: String,
    pub at: i64,
    pub created: i64,
    pub status: String,
    /// Levels above it.
    pub depth: usize,
    /// Current memories it owns (its own and copied-up ones).
    pub own: usize,
}

// ------------------------------------------------------------------ engine

impl<'s, S: Storage<'s>> Db<S> {
    /// Idempotent: a relation that already exists is kept.
    pub(crate) fn create_v6_relations(&'s self) -> Result<()> {
        for (name, cols) in [
            (MEM_BRANCH, "{ns: String => parent: String, at: Int, created: Int, status: String}"),
            (MEM_COPIED, "{ns: String, id: String => from: String, at: Int}"),
            (MEM_MERGE, "{into: String, stmt: String => from: String, at: Int, action: String, ids: [String]}"),
        ] {
            let exists = self
                .run_script("::relations", Default::default(), ScriptMutability::Immutable)?
                .rows
                .iter()
                .any(|r| r.first().and_then(|v| v.get_str()) == Some(name));
            if !exists {
                self.mem_q(&format!("{{:create {name} {cols}}}"), vec![])?;
            }
        }
        Ok(())
    }

    /// The lineage of `ns` (spec "Lineage"). A namespace with no branch row
    /// is plain.
    pub(crate) fn mem_scope(&'s self, ns: &str) -> Result<Scope> {
        let mut sc = Scope::plain(ns);
        let mut cur = ns.to_string();
        let mut cut: Option<i64> = None;
        loop {
            let rows = self.mem_q(
                &format!("?[parent, at, status] := *{MEM_BRANCH}{{ns: $ns, parent, at, status}}"),
                vec![("ns", s(&cur))],
            )?;
            let Some(r) = rows.rows.first() else { break };
            if r[2].get_str() == Some("discarded") {
                if cur == ns {
                    // a tombstone: no lineage, no writes
                    sc.discarded = true;
                }
                break;
            }
            let (Some(parent), Some(at)) = (cell_str(&r[0]), r[1].get_int()) else { break };
            if sc.levels.iter().any(|l| l.ns == parent) {
                bail!("memory branch lineage of '{ns}' has a cycle at '{parent}'");
            }
            if sc.levels.len() > MAX_DEPTH {
                bail!("memory branch lineage of '{ns}' is deeper than {MAX_DEPTH}");
            }
            cut = Some(cut.map_or(at, |c: i64| c.min(at)));
            sc.levels.push(Level { ns: parent.clone(), cut });
            cur = parent;
        }
        Ok(sc)
    }

    /// The level owning each id: the nearest with any `kleos_mem` row for it.
    /// Ids no level has are left out. A plain scope owns every id.
    pub(crate) fn mem_owners(&'s self, sc: &Scope, ids: &[String]) -> Result<BTreeMap<String, usize>> {
        if sc.is_plain() {
            return Ok(ids.iter().map(|i| (i.clone(), 0)).collect());
        }
        if ids.is_empty() {
            return Ok(BTreeMap::new());
        }
        let lin: Vec<DataValue> = sc
            .levels
            .iter()
            .enumerate()
            .map(|(i, l)| DataValue::List(vec![s(&l.ns), DataValue::from(i as i64)]))
            .collect();
        Ok(self
            .mem_q(
                &format!(
                    "lin[ns, lvl] <- $lin\n\
                     ?[id, min(lvl)] := id in $ids, lin[ns, lvl], *{MEM}{{ns, id}}"
                ),
                vec![("lin", DataValue::List(lin)), ("ids", str_list(ids))],
            )?
            .rows
            .into_iter()
            .filter_map(|r| Some((cell_str(&r[0])?, r[1].get_int()? as usize)))
            .collect())
    }

    /// `ids` grouped by owning level, nearest first.
    pub(crate) fn by_owner(&'s self, sc: &Scope, ids: &[String]) -> Result<Vec<(usize, Vec<String>)>> {
        if sc.is_plain() {
            return Ok(if ids.is_empty() { vec![] } else { vec![(0, ids.to_vec())] });
        }
        let mut g: BTreeMap<usize, Vec<String>> = BTreeMap::new();
        for (id, l) in self.mem_owners(sc, ids)? {
            g.entry(l).or_default().push(id);
        }
        Ok(g.into_iter().collect())
    }

    /// The ids among `ids` valid in the view at `t` (`None`: now).
    pub(crate) fn mem_valid(&'s self, sc: &Scope, ids: &[String], t: Option<i64>) -> Result<Vec<String>> {
        let mut out = vec![];
        for (l, group) in self.by_owner(sc, ids)? {
            let (at, mut p) = match sc.level_at(l, t) {
                None => ("'NOW'", vec![]),
                Some(x) => ("$t", vec![("t", DataValue::from(x))]),
            };
            p.push(("ns", s(&sc.levels[l].ns)));
            p.push(("ids", str_list(&group)));
            out.extend(
                self.mem_q(&format!("?[id] := id in $ids, *{MEM}{{ns: $ns, id @ {at}}}"), p)?
                    .rows
                    .into_iter()
                    .filter_map(|r| cell_str(&r[0])),
            );
        }
        out.sort();
        Ok(out)
    }

    /// Every memory of the view valid at `t`: `(id, owner ns, text, kind)`.
    pub(crate) fn view_all(&'s self, sc: &Scope, t: Option<i64>) -> Result<Vec<(String, String, String, String)>> {
        let mut p = vec![];
        let rules = sc.view_rules("vis", "", &["text", "kind"], t, &mut p);
        Ok(self
            .mem_q(&format!("{rules}?[id, ons, text, kind] := vis[id, ons, text, kind]"), p)?
            .rows
            .into_iter()
            .filter_map(|r| Some((cell_str(&r[0])?, cell_str(&r[1])?, cell_str(&r[2])?, cell_str(&r[3])?)))
            .collect())
    }

    /// Blocks that copy the inherited memories among `ids` into the scope's
    /// namespace (spec "Copy-up"), to go at the start of a write script that
    /// binds `$ns`. Returns the ids copied.
    pub(crate) fn copy_up_blocks(
        &'s self,
        cfg: &MemConfig,
        sc: &Scope,
        ids: &[String],
        script: &mut String,
        p: &mut Vec<(&'static str, DataValue)>,
    ) -> Result<Vec<String>> {
        if sc.is_plain() || ids.is_empty() {
            return Ok(vec![]);
        }
        let mut copied = vec![];
        let now = now_micros();
        // booleans go in as parameters: a literal `x = true` in a rule body
        // is not bound reliably (see the spec's caveats)
        p.push(("cu_yes", DataValue::Bool(true)));
        p.push(("cu_no", DataValue::Bool(false)));
        for (l, group) in self.by_owner(sc, ids)? {
            if l == 0 {
                continue;
            }
            let (o, i, c) = (CU_O[l], CU_I[l], CU_C[l]);
            p.push((o, s(&sc.levels[l].ns)));
            p.push((i, str_list(&group)));
            p.push((c, DataValue::from(sc.levels[l].cut.unwrap_or(i64::MAX))));
            script.push_str(&format!(
                "{{?[ns, id, vt, text, kind, source, created, importance] := id in ${i}, \
                   *{MEM}{{ns: ${o}, id, vt, text, kind, source, created, importance}}, to_int(vt) <= ${c}, ns = $ns \
                  :put {MEM} {{ns, id, vt => text, kind, source, created, importance}}}}"
            ));
            if cfg.dim.is_some() {
                script.push_str(&format!(
                    "{{cur[id] := id in ${i}, *{MEM}{{ns: ${o}, id @ ${c}}}\n\
                      ?[ns, id, current, vec] := id in ${i}, *{MEM_VEC}{{ns: ${o}, id, vec}}, ns = $ns, cur[id], current = $cu_yes\n\
                      ?[ns, id, current, vec] := id in ${i}, *{MEM_VEC}{{ns: ${o}, id, vec}}, ns = $ns, not cur[id], current = $cu_no \
                     :put {MEM_VEC} {{ns, id => current, vec}}}}"
                ));
            }
            script.push_str(&format!(
                "{{?[ns, id, at, stmt, source, text] := id in ${i}, *{MEM_SOURCE}{{ns: ${o}, id, at, stmt, source, text}}, at <= ${c}, ns = $ns \
                  :put {MEM_SOURCE} {{ns, id, at, stmt => source, text}}}}\
                 {{?[ns, id, eid] := id in ${i}, *{MEM_MENTION}{{ns: ${o}, id, eid}}, ns = $ns :put {MEM_MENTION} {{ns, id, eid}}}}\
                 {{?[ns, id, role] := id in ${i}, *{MEM_ROLE}{{ns: ${o}, id, role}}, ns = $ns :put {MEM_ROLE} {{ns, id => role}}}}\
                 {{?[ns, id, n, start, end, granularity, kind, expr] := id in ${i}, \
                    *{MEM_EVENT}{{ns: ${o}, id, n, start, end, granularity, kind, expr}}, ns = $ns \
                   :put {MEM_EVENT} {{ns, id, n => start, end, granularity, kind, expr}}}}\
                 {{?[ns, id, expires, archived] := id in ${i}, *{MEM_LIFE}{{ns: ${o}, id, expires, archived}}, ns = $ns \
                   :put {MEM_LIFE} {{ns, id => expires, archived}}}}\
                 {{?[ns, id, from, at] := id in ${i}, ns = $ns, from = ${o}, at = {now} :put {MEM_COPIED} {{ns, id => from, at}}}}"
            ));
            copied.extend(group);
        }
        Ok(copied)
    }

    /// Namespaces of the family `sc` belongs to: its root and every branch
    /// descending from it (spec "Erase").
    pub(crate) fn mem_family(&'s self, sc: &Scope) -> Result<Vec<String>> {
        let root = sc.levels.last().map(|l| l.ns.clone()).unwrap_or_default();
        let mut out = vec![root.clone()];
        let mut frontier = vec![root];
        while !frontier.is_empty() {
            let kids: Vec<String> = self
                .mem_q(
                    &format!("?[c] := p in $ps, *{MEM_BRANCH}{{ns: c, parent: p}}"),
                    vec![("ps", str_list(&frontier))],
                )?
                .rows
                .into_iter()
                .filter_map(|r| cell_str(&r[0]))
                .filter(|c| !out.contains(c))
                .collect();
            out.extend(kids.iter().cloned());
            frontier = kids;
        }
        Ok(out)
    }

    // -------------------------------------------------------------- lineage reads

    /// Entity rows and merges of a lineage: each level's rows (an
    /// ancestor's created at or before its cut), merges followed as of each
    /// entity's level read time.
    pub(crate) fn entity_view_lineage(&'s self, sc: &Scope) -> Result<EntityView> {
        let mut v = EntityView::default();
        let mut by_level: Vec<(usize, Vec<String>)> = vec![];
        for (i, l) in sc.levels.iter().enumerate() {
            let (cut, c) = match l.cut {
                Some(c) => (", created <= $c", DataValue::from(c)),
                None => ("", DataValue::Null),
            };
            let mut eids = vec![];
            for r in self
                .mem_q(
                    &format!("?[eid, name, norm] := *{MEM_ENTITY}{{ns: $ns, eid, name, norm, created}}{cut}"),
                    vec![("ns", s(&l.ns)), ("c", c)],
                )?
                .rows
            {
                if let (Some(e), Some(n), Some(m)) = (cell_str(&r[0]), cell_str(&r[1]), cell_str(&r[2])) {
                    if !v.rows.contains_key(&e) {
                        eids.push(e.clone());
                        v.rows.insert(e, (n, m));
                    }
                }
            }
            by_level.push((i, eids));
        }
        for (i, eids) in by_level {
            if eids.is_empty() {
                continue;
            }
            let (at, mut p) = match sc.level_at(i, None) {
                None => ("'NOW'", vec![]),
                Some(x) => ("$t", vec![("t", DataValue::from(x))]),
            };
            p.push(("eids", str_list(&eids)));
            p.push(("r", s(MEM_ENTITY)));
            for r in self
                .mem_q(&format!("?[f, i] := f in $eids, *{ENTITY_MERGES}{{relation: $r, from: f, into: i @ {at}}}"), p)?
                .rows
            {
                if let (Some(f), Some(i)) = (cell_str(&r[0]), cell_str(&r[1])) {
                    v.merged_into.insert(f, i);
                }
            }
        }
        Ok(v)
    }

    /// `(id, eid)` mentions of `ids`, from each memory's owner.
    pub(crate) fn mentions_of(&'s self, sc: &Scope, ids: &[String]) -> Result<Vec<(String, String)>> {
        let mut out = vec![];
        for (l, group) in self.by_owner(sc, ids)? {
            for r in self
                .mem_q(
                    &format!("?[id, eid] := id in $ids, *{MEM_MENTION}{{ns: $ns, id, eid}}"),
                    vec![("ns", s(&sc.levels[l].ns)), ("ids", str_list(&group))],
                )?
                .rows
            {
                if let (Some(id), Some(e)) = (cell_str(&r[0]), cell_str(&r[1])) {
                    out.push((id, e));
                }
            }
        }
        Ok(out)
    }

    /// `(eid, id)`: the memories mentioning `eids`, at every level (an id is
    /// listed once; whether it is in the view is checked by the caller).
    pub(crate) fn mentioning(&'s self, sc: &Scope, eids: &[String]) -> Result<Vec<(String, String)>> {
        let mut out: BTreeSet<(String, String)> = BTreeSet::new();
        for l in &sc.levels {
            for r in self
                .mem_q(
                    &format!("?[eid, id] := eid in $eids, *{MEM_MENTION}{{ns: $ns, eid, id}}"),
                    vec![("ns", s(&l.ns)), ("eids", str_list(eids))],
                )?
                .rows
            {
                if let (Some(e), Some(id)) = (cell_str(&r[0]), cell_str(&r[1])) {
                    out.insert((e, id));
                }
            }
        }
        Ok(out.into_iter().collect())
    }

    /// Vectors of `ids`, from each memory's owner.
    pub(crate) fn vecs_of(&'s self, sc: &Scope, ids: &[String]) -> Result<BTreeMap<String, Vec<f32>>> {
        let mut out = BTreeMap::new();
        for (l, group) in self.by_owner(sc, ids)? {
            for r in self
                .mem_q(
                    &format!("?[id, vec] := id in $ids, *{MEM_VEC}{{ns: $ns, id, vec}}"),
                    vec![("ns", s(&sc.levels[l].ns)), ("ids", str_list(&group))],
                )?
                .rows
            {
                if let (Some(id), Some(v)) = (cell_str(&r[0]), vec_of(&r[1])) {
                    out.insert(id, v);
                }
            }
        }
        Ok(out)
    }

    /// `supersedes` links `(src, dst)` from `srcs`, at every level (an
    /// ancestor's created at or before its cut).
    pub(crate) fn supersede_links(&'s self, sc: &Scope, srcs: &[String]) -> Result<Vec<(String, String)>> {
        let mut out: BTreeSet<(String, String)> = BTreeSet::new();
        for l in &sc.levels {
            let (cut, c) = match l.cut {
                Some(c) => (", created <= $c", DataValue::from(c)),
                None => ("", DataValue::Null),
            };
            for r in self
                .mem_q(
                    &format!("?[src, dst] := src in $srcs, *{MEM_LINK}{{ns: $ns, src, rel: 'supersedes', dst, created}}{cut}"),
                    vec![("ns", s(&l.ns)), ("srcs", str_list(srcs)), ("c", c)],
                )?
                .rows
            {
                if let (Some(a), Some(b)) = (cell_str(&r[0]), cell_str(&r[1])) {
                    out.insert((a, b));
                }
            }
        }
        Ok(out.into_iter().collect())
    }

    /// Pending `supersedes` decisions about `cands`, at every level (an
    /// ancestor's made at or before its cut), whose new memory is in the
    /// view at `t`: candidate → new memories.
    pub(crate) fn pending_of(&'s self, sc: &Scope, cands: &[String], t: i64) -> Result<BTreeMap<String, Vec<String>>> {
        let mut pairs: BTreeSet<(String, String)> = BTreeSet::new();
        for l in &sc.levels {
            let (cut, c) = match l.cut {
                Some(c) => (", at <= $c", DataValue::from(c)),
                None => ("", DataValue::Null),
            };
            for r in self
                .mem_q(
                    &format!(
                        "?[id, new] := id in $ids, *{MEM_DECISION}{{ns: $ns, new, cand: id, template: 'supersedes', status: 'pending', at}}{cut}"
                    ),
                    vec![("ns", s(&l.ns)), ("ids", str_list(cands)), ("c", c)],
                )?
                .rows
            {
                if let (Some(a), Some(b)) = (cell_str(&r[0]), cell_str(&r[1])) {
                    pairs.insert((a, b));
                }
            }
        }
        let news: Vec<String> = pairs.iter().map(|x| x.1.clone()).unique().collect();
        let valid: BTreeSet<String> = self.mem_valid(sc, &news, Some(t))?.into_iter().collect();
        let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (cand, new) in pairs {
            if valid.contains(&new) {
                out.entry(cand).or_default().push(new);
            }
        }
        Ok(out)
    }

    // -------------------------------------------------------------- ops

    /// Fork `from` into the new branch `to` as of `at` (spec "fork"). O(1):
    /// one branch row and a version row, whatever the size of `from`.
    pub fn memory_fork(&'s self, opts: ForkOpts) -> Result<ForkResult> {
        self.mem_ready(None)?;
        let from = ns_of(&opts.from);
        let to = opts.to.trim().to_string();
        if to.is_empty() {
            bail!("`memory fork` needs `to`, the new branch's name");
        }
        if to == from {
            bail!("`memory fork`: from and to are both '{to}'");
        }
        let now = now_micros();
        let at = opts.at.unwrap_or(now);
        if at > now {
            bail!("`memory fork`: at ({at}) is in the future");
        }
        let parent = self.mem_scope(&from)?;
        if parent.discarded {
            bail!("`memory fork`: '{from}' is a discarded branch");
        }
        if parent.levels.len() > MAX_DEPTH {
            bail!("`memory fork`: '{from}' is already {} levels deep (limit {MAX_DEPTH})", parent.levels.len() - 1);
        }
        // `to` must be unused (a discarded tombstone may be reused); the
        // asserts run in the fork's own transaction, so two forks into one
        // name cannot both succeed
        let mut script = format!(
            "{{?[x] := *{MEM}{{ns: $ns, id: x}} :limit 1 :assert none}}\
             {{?[x] := *{MEM_ENTITY}{{ns: $ns, eid: x}} :limit 1 :assert none}}\
             {{?[x] := *{MEM_BRANCH}{{ns: $ns, status: x}}, x != 'discarded' :assert none}}\
             {{?[x] := *{MEM_NS}{{ns: $ns, seq: x}}, not *{MEM_BRANCH}{{ns: $ns}} :assert none}}\
             {{?[ns, parent, at, created, status] <- [[$ns, $from, $at, $now, 'active']] \
               :put {MEM_BRANCH} {{ns => parent, at, created, status}}}}"
        );
        script.push_str(&bump_ns_head());
        match self.mem_q(
            &script,
            vec![("ns", s(&to)), ("from", s(&from)), ("at", DataValue::from(at)), ("now", DataValue::from(now))],
        ) {
            Ok(_) => {}
            Err(e) if format!("{e:?}").contains("assert") || format!("{e:?}").contains("Assert") => {
                bail!("`memory fork`: '{to}' is already in use (it has memories, entities or is an active branch)")
            }
            Err(e) => return Err(e),
        }
        let sc = self.mem_scope(&to)?;
        Ok(ForkResult {
            ns: to,
            parent: from,
            at,
            depth: sc.levels.len() - 1,
            lineage: sc.levels.iter().map(|l| (l.ns.clone(), l.cut)).collect(),
        })
    }

    /// Put a namespace's view back to its state at `to` (spec "rewind"):
    /// retract what is current now and was not then, assert again what was
    /// valid then and is not now. Appends only, so it is undoable.
    pub fn memory_rewind(&'s self, opts: RewindOpts) -> Result<RewindResult> {
        let cfg = match self.mem_config()? {
            Some(c) => c,
            None => bail!("`memory rewind`: there is no memory store yet"),
        };
        let ns = ns_of(&opts.ns);
        let sc = self.mem_scope(&ns)?;
        if sc.discarded {
            bail!("`memory rewind`: '{ns}' is a discarded branch");
        }
        let gate = ns_gate(&ns);
        let _commit = gate.commit();
        let at = opts.at.unwrap_or_else(now_micros);
        if opts.to >= at {
            bail!("`memory rewind`: to ({}) must be before at ({at})", opts.to);
        }
        let seen = self.ns_head(&ns)?;
        let now_set: BTreeMap<String, ()> = self.view_all(&sc, Some(at))?.into_iter().map(|r| (r.0, ())).collect();
        let mut then: BTreeMap<String, String> = BTreeMap::new();
        for (id, _, text, _) in self.view_all(&sc, Some(opts.to))? {
            then.insert(id, text);
        }
        let retracted: Vec<String> = now_set.keys().filter(|id| !then.contains_key(*id)).cloned().collect();
        let restored: Vec<String> = then.keys().filter(|id| !now_set.contains_key(*id)).cloned().collect();
        let mut res = RewindResult {
            ns: ns.clone(),
            to: opts.to,
            at,
            retracted: retracted.clone(),
            restored: restored.clone(),
            undo: at - 1,
        };
        if retracted.is_empty() && restored.is_empty() {
            return Ok(res);
        }
        // the values each restored memory had at `to`
        let mut rows = vec![];
        for (l, group) in self.by_owner(&sc, &restored)? {
            let t = sc.level_at(l, Some(opts.to)).unwrap_or(opts.to);
            for r in self
                .mem_q(
                    &format!(
                        "?[id, text, kind, source, created, importance] := id in $ids, \
                           *{MEM}{{ns: $ons, id, text, kind, source, created, importance @ $t}}"
                    ),
                    vec![("ons", s(&sc.levels[l].ns)), ("ids", str_list(&group)), ("t", DataValue::from(t))],
                )?
                .rows
            {
                let mut v = vec![s(&ns), r[0].clone(), DataValue::Validity(Validity::from((at, true)))];
                v.extend(r[1..].iter().cloned());
                rows.push(DataValue::List(v));
            }
        }
        let mut script = String::new();
        let mut p: Vec<(&str, DataValue)> = vec![("ns", s(&ns))];
        let touched: Vec<String> = retracted.iter().chain(restored.iter()).cloned().collect();
        self.copy_up_blocks(&cfg, &sc, &touched, &mut script, &mut p)?;
        if !retracted.is_empty() {
            script.push_str(&format!(
                "{{?[ns, id, vt, text, kind, source, created, importance] := id in $ret, \
                   *{MEM}{{ns, id, text, kind, source, created, importance @ $at}}, ns = $ns, vt = [$at, false] \
                  :put {MEM} {{ns, id, vt => text, kind, source, created, importance}}}}"
            ));
            p.push(("ret", str_list(&retracted)));
        }
        if !rows.is_empty() {
            script.push_str(&format!(
                "{{?[ns, id, vt, text, kind, source, created, importance] <- $rest \
                  :put {MEM} {{ns, id, vt => text, kind, source, created, importance}}}}"
            ));
            p.push(("rest", DataValue::List(rows)));
        }
        if cfg.dim.is_some() {
            script.push_str(&format!(
                "{{?[ns, id, current] := *{MEM_VEC}{{ns, id}}, ns = $ns, id in $ret, current = $rw_no \
                  :update {MEM_VEC} {{ns, id => current}}}}\
                 {{?[ns, id, current] := *{MEM_VEC}{{ns, id}}, ns = $ns, id in $res, current = $rw_yes \
                  :update {MEM_VEC} {{ns, id => current}}}}"
            ));
            if retracted.is_empty() {
                p.push(("ret", DataValue::List(vec![])));
            }
            p.push(("res", str_list(&restored)));
            p.push(("rw_yes", DataValue::Bool(true)));
            p.push(("rw_no", DataValue::Bool(false)));
        }
        p.push(("at", DataValue::from(at)));
        // Engram: the retracted memories end, the restored ones are written again
        let ret_job = Job {
            op: "retract",
            mem_ids: &retracted,
            old: &[],
            text: None,
            at,
            entities: &[],
        };
        let singles: Vec<Vec<String>> = restored.iter().map(|id| vec![id.clone()]).collect();
        let mut jobs: Vec<Job<'_>> = vec![];
        if !retracted.is_empty() {
            jobs.push(ret_job);
        }
        for (one, id) in singles.iter().zip(&restored) {
            jobs.push(Job {
                op: "write",
                mem_ids: one,
                old: &[],
                text: then.get(id).map(|t| t.as_str()),
                at,
                entities: &[],
            });
        }
        let mut bump = 1;
        if let Some((frag, ps, n)) = engram_hook::enqueue_jobs(&cfg, &jobs) {
            script.push_str(&frag);
            p.extend(ps);
            bump = n.max(1);
        }
        script.insert_str(0, &format!("{{?[seq] := *{MEM_NS}{{ns: $ns, seq}}, seq != $seen :assert none}}"));
        script.push_str(&bump_ns_head_by(bump));
        p.push(("seen", DataValue::from(seen)));
        self.mem_q(&script, p)?;
        res.retracted.sort();
        res.restored.sort();
        Ok(res)
    }

    /// Compare the views of two namespaces at `at` (spec "diff"): what
    /// changes going from `a` to `b`.
    pub fn memory_diff(&'s self, opts: DiffOpts) -> Result<DiffResult> {
        let at = opts.at.unwrap_or_else(now_micros);
        let mut res = DiffResult {
            a: opts.a.clone(),
            b: opts.b.clone(),
            at,
            ..Default::default()
        };
        if self.mem_config()?.is_none() {
            return Ok(res);
        }
        let sa = self.mem_scope(&opts.a)?;
        let sb = self.mem_scope(&opts.b)?;
        let va = self.view_all(&sa, Some(at))?;
        let vb = self.view_all(&sb, Some(at))?;
        let ia: BTreeMap<&str, &(String, String, String, String)> = va.iter().map(|r| (r.0.as_str(), r)).collect();
        let ib: BTreeMap<&str, &(String, String, String, String)> = vb.iter().map(|r| (r.0.as_str(), r)).collect();
        // by id, then by normalised text (a merged statement has a new id)
        let ta: BTreeSet<String> = va.iter().map(|r| norm_text(&r.2)).collect();
        let tb: BTreeSet<String> = vb.iter().map(|r| norm_text(&r.2)).collect();
        let added: Vec<&(String, String, String, String)> = vb
            .iter()
            .filter(|r| !ia.contains_key(r.0.as_str()) && !ta.contains(&norm_text(&r.2)))
            .collect();
        let removed: Vec<&(String, String, String, String)> = va
            .iter()
            .filter(|r| !ib.contains_key(r.0.as_str()) && !tb.contains(&norm_text(&r.2)))
            .collect();
        res.same = va.len() - removed.len();
        let from_of = |sc: &Scope, ids: Vec<String>| -> Result<BTreeMap<String, i64>> {
            Ok(intervals(&self.mem_history(sc, &ids)?, at).into_iter().map(|(k, v)| (k, v.0)).collect())
        };
        let fa = from_of(&sa, removed.iter().map(|r| r.0.clone()).collect())?;
        let fb = from_of(&sb, added.iter().map(|r| r.0.clone()).collect())?;
        res.added = added
            .iter()
            .map(|r| DiffMem {
                id: r.0.clone(),
                text: r.2.clone(),
                kind: r.3.clone(),
                valid_from: fb.get(&r.0).copied().unwrap_or(0),
                ..Default::default()
            })
            .collect();
        // superseded: a memory current in b has a supersedes link to it
        let srcs: Vec<String> = vb.iter().map(|r| r.0.clone()).collect();
        let mut by: BTreeMap<String, String> = BTreeMap::new();
        if !removed.is_empty() {
            for (src, dst) in self.supersede_links(&sb, &srcs)? {
                by.entry(dst).or_insert(src);
            }
        }
        res.removed = removed
            .iter()
            .map(|r| DiffMem {
                id: r.0.clone(),
                text: r.2.clone(),
                kind: r.3.clone(),
                valid_from: fa.get(&r.0).copied().unwrap_or(0),
                reason: Some(if by.contains_key(&r.0) { "superseded" } else { "forgotten" }.to_string()),
                by: by.get(&r.0).cloned(),
            })
            .collect();
        // pending supersedes of b about memories current in both, that a lacks
        let both: Vec<String> = va.iter().filter(|r| ib.contains_key(r.0.as_str())).map(|r| r.0.clone()).collect();
        let pa = self.pending_of(&sa, &both, at)?;
        for (cand, news) in self.pending_of(&sb, &both, at)? {
            for new in news {
                if pa.get(&cand).is_some_and(|v| v.contains(&new)) {
                    continue;
                }
                res.pending.push(DiffPending {
                    candidate_text: ib.get(cand.as_str()).map(|r| r.2.clone()).unwrap_or_default(),
                    new_text: ib.get(new.as_str()).map(|r| r.2.clone()).unwrap_or_default(),
                    new,
                    candidate: cand.clone(),
                });
            }
        }
        // entities: canonical names, normalised
        let names = |sc: &Scope| -> Result<BTreeMap<String, String>> {
            let v = if sc.is_plain() { self.entity_view(sc)? } else { self.entity_view_lineage(sc)? };
            Ok(v.rows
                .iter()
                .filter(|(e, _)| v.canonical(e) == **e)
                .map(|(_, (name, norm))| (norm.clone(), name.clone()))
                .collect())
        };
        let (ea, eb) = (names(&sa)?, names(&sb)?);
        res.entities.added = eb.iter().filter(|(k, _)| !ea.contains_key(*k)).map(|(_, v)| v.clone()).collect();
        res.entities.removed = ea.iter().filter(|(k, _)| !eb.contains_key(*k)).map(|(_, v)| v.clone()).collect();
        Ok(res)
    }

    /// Drop a branch (spec "discard"): every row it owns, its Engram queue
    /// and overlay; ancestors are untouched. Children need `cascade`.
    pub fn memory_discard(&'s self, opts: DiscardOpts) -> Result<DiscardResult> {
        let cfg = match self.mem_config()? {
            Some(c) => c,
            None => bail!("`memory discard`: there is no memory store yet"),
        };
        let ns = ns_of(&opts.ns);
        let mut res = DiscardResult {
            ns: ns.clone(),
            ..Default::default()
        };
        self.discard_one(&cfg, &ns, opts.cascade, &mut res)?;
        Ok(res)
    }

    fn discard_one(&'s self, cfg: &MemConfig, ns: &str, cascade: bool, res: &mut DiscardResult) -> Result<()> {
        let row = self.mem_q(
            &format!("?[status] := *{MEM_BRANCH}{{ns: $ns, status}}"),
            vec![("ns", s(ns))],
        )?;
        match row.rows.first().and_then(|r| r[0].get_str()) {
            Some("active") => {}
            Some(_) => bail!("`memory discard`: '{ns}' is already discarded"),
            None => bail!(
                "`memory discard`: '{ns}' is not a branch; to delete a namespace use `forget {{all: true, mode: 'erase'}}`"
            ),
        }
        let kids: Vec<String> = self
            .mem_q(
                &format!("?[c] := *{MEM_BRANCH}{{ns: c, parent: $ns, status: 'active'}}"),
                vec![("ns", s(ns))],
            )?
            .rows
            .into_iter()
            .filter_map(|r| cell_str(&r[0]))
            .collect();
        if !kids.is_empty() {
            if !cascade {
                bail!(
                    "`memory discard`: '{ns}' has branches of its own ({}); discard them first or pass cascade: true",
                    kids.join(", ")
                );
            }
            for k in kids {
                self.discard_one(cfg, &k, true, res)?;
            }
        }
        let gate = ns_gate(ns);
        let _commit = gate.commit();
        let ids: Vec<String> = self
            .mem_q(&format!("?[id] := *{MEM}{{ns: $ns, id}}"), vec![("ns", s(ns))])?
            .rows
            .into_iter()
            .filter_map(|r| cell_str(&r[0]))
            .collect();
        let eids: Vec<String> = self
            .mem_q(&format!("?[eid] := *{MEM_ENTITY}{{ns: $ns, eid}}"), vec![("ns", s(ns))])?
            .rows
            .into_iter()
            .filter_map(|r| cell_str(&r[0]))
            .collect();
        let mut script = String::new();
        if cfg.dim.is_some() {
            script.push_str(&format!("{{?[ns, id] := *{MEM_VEC}{{ns, id}}, ns = $ns :rm {MEM_VEC} {{ns, id}}}}"));
        }
        script.push_str(&format!(
            "{{?[ns, id, at, stmt] := *{MEM_SOURCE}{{ns, id, at, stmt}}, ns = $ns :rm {MEM_SOURCE} {{ns, id, at, stmt}}}}\
             {{?[ns, id, eid] := *{MEM_MENTION}{{ns, id, eid}}, ns = $ns :rm {MEM_MENTION} {{ns, id, eid}}}}\
             {{?[ns, new, cand, template] := *{MEM_DECISION}{{ns, new, cand, template}}, ns = $ns :rm {MEM_DECISION} {{ns, new, cand, template}}}}\
             {{?[ns, src, rel, dst] := *{MEM_LINK}{{ns, src, rel, dst}}, ns = $ns :rm {MEM_LINK} {{ns, src, rel, dst}}}}\
             {{?[ns, id] := *{MEM_ACCESS}{{ns, id}}, ns = $ns :rm {MEM_ACCESS} {{ns, id}}}}\
             {{?[ns, id] := *{MEM_LIFE}{{ns, id}}, ns = $ns :rm {MEM_LIFE} {{ns, id}}}}\
             {{?[ns, id] := *{MEM_ROLE}{{ns, id}}, ns = $ns :rm {MEM_ROLE} {{ns, id}}}}\
             {{?[ns, id, n] := *{MEM_EVENT}{{ns, id, n}}, ns = $ns :rm {MEM_EVENT} {{ns, id, n}}}}\
             {{?[ns, eid] := *{MEM_ENTITY}{{ns, eid}}, ns = $ns :rm {MEM_ENTITY} {{ns, eid}}}}\
             {{?[relation, from, vt] := *{ENTITY_MERGES}{{relation, from, vt, into}}, relation = $erel, is_in(from, $eids) || is_in(into, $eids) \
               :rm {ENTITY_MERGES} {{relation, from, vt}}}}\
             {{?[relation, a, b] := *{ENTITY_LINKS}{{relation, a, b}}, relation = $erel, is_in(a, $eids) || is_in(b, $eids) \
               :rm {ENTITY_LINKS} {{relation, a, b}}}}\
             {{?[ns, id] := *{MEM_COPIED}{{ns, id}}, ns = $ns :rm {MEM_COPIED} {{ns, id}}}}\
             {{?[into, stmt] := *{MEM_MERGE}{{into, stmt}}, into = $ns :rm {MEM_MERGE} {{into, stmt}}}}\
             {{?[ns, status] := *{MEM_BRANCH}{{ns}}, ns = $ns, status = 'discarded' :update {MEM_BRANCH} {{ns => status}}}}"
        ));
        #[cfg(feature = "engram")]
        if self.mem_q(&format!("?[x] := *{}{{ns: $ns, qid: x}} :limit 1", crate::memory_engram::ENGRAM_QUEUE), vec![("ns", s(ns))]).is_ok() {
            script.push_str(&format!(
                "{{?[ns, qid] := *{q}{{ns, qid}}, ns = $ns :rm {q} {{ns, qid}}}}",
                q = crate::memory_engram::ENGRAM_QUEUE
            ));
        }
        // in-flight writers of the branch fail their version check
        script.push_str(&bump_ns_head());
        self.mem_q(
            &script,
            vec![("ns", s(ns)), ("eids", str_list(&eids)), ("erel", s(MEM_ENTITY))],
        )?;
        if !ids.is_empty() {
            let keys = DataValue::List(ids.iter().map(|id| DataValue::List(vec![s(ns), s(id)])).collect());
            self.mem_q(&format!("::erase {MEM} $keys"), vec![("keys", keys)])?;
        }
        #[cfg(feature = "engram")]
        {
            self.engram_erase_namespace(ns, None)?;
        }
        self.mem_access.forget(ns, &ids);
        res.discarded.push(ns.to_string());
        res.memories += ids.len();
        Ok(())
    }

    /// List branches (spec "branches").
    pub fn memory_branches(&'s self, opts: BranchesOpts) -> Result<Vec<BranchInfo>> {
        if self.mem_config()?.is_none() {
            return Ok(vec![]);
        }
        let mut rows: Vec<BranchInfo> = self
            .mem_q(
                &format!("?[ns, parent, at, created, status] := *{MEM_BRANCH}{{ns, parent, at, created, status}}"),
                vec![],
            )?
            .rows
            .into_iter()
            .filter_map(|r| {
                Some(BranchInfo {
                    ns: cell_str(&r[0])?,
                    parent: cell_str(&r[1])?,
                    at: r[2].get_int()?,
                    created: r[3].get_int()?,
                    status: cell_str(&r[4])?,
                    ..Default::default()
                })
            })
            .filter(|b| opts.all || b.status == "active")
            .collect();
        if let Some(ns) = &opts.ns {
            let sc = self.mem_scope(ns)?;
            let fam: BTreeSet<String> = self.mem_family(&sc)?.into_iter().collect();
            rows.retain(|b| fam.contains(&b.ns));
        }
        for b in rows.iter_mut() {
            b.depth = self.mem_scope(&b.ns)?.levels.len() - 1;
            b.own = self
                .mem_q(&format!("?[count(id)] := *{MEM}{{ns: $ns, id @ 'NOW'}}"), vec![("ns", s(&b.ns))])?
                .rows
                .first()
                .and_then(|r| r[0].get_int())
                .unwrap_or(0) as usize;
        }
        rows.sort_by(|a, b| a.depth.cmp(&b.depth).then(a.created.cmp(&b.created)).then(a.ns.cmp(&b.ns)));
        Ok(rows)
    }

    /// Replay a branch's own statements into `into` through `remember`
    /// (spec "merge"). With `dry_run`, into a scratch fork of `into` that is
    /// diffed and discarded.
    pub fn memory_merge(&'s self, opts: MergeOpts) -> Result<MergeResult> {
        self.mem_ready(None)?;
        let from = opts.from.trim().to_string();
        if from.is_empty() {
            bail!("`memory merge` needs `from`, the branch to merge");
        }
        let sf = self.mem_scope(&from)?;
        if sf.discarded {
            bail!("`memory merge`: '{from}' is a discarded branch");
        }
        let into = match &opts.into {
            Some(i) if !i.is_empty() => i.clone(),
            _ => match sf.levels.get(1) {
                Some(l) => l.ns.clone(),
                None => bail!("`memory merge`: '{from}' is not a branch; name the target with `into`"),
            },
        };
        if into == from {
            bail!("`memory merge`: from and into are both '{from}'");
        }
        if self.mem_scope(&into)?.discarded {
            bail!("`memory merge`: '{into}' is a discarded branch");
        }
        if !opts.dry_run {
            return self.merge_into(&opts, &from, &into, &into);
        }
        // preview: the real merge into a scratch fork of `into`
        let scratch = format!("{into}~preview-{}", new_id("x"));
        self.memory_fork(ForkOpts {
            from: Some(into.clone()),
            to: scratch.clone(),
            at: None,
        })?;
        let out: Result<MergeResult> = (|| {
            let mut r = self.merge_into(&opts, &from, &scratch, &into)?;
            r.diff = Some(self.memory_diff(DiffOpts {
                a: into.clone(),
                b: scratch.clone(),
                at: None,
            })?);
            r.into = into.clone();
            r.dry_run = true;
            if let Some(d) = r.diff.as_mut() {
                d.b = format!("{into} (after merge)");
            }
            Ok(r)
        })();
        let dropped = self.memory_discard(DiscardOpts {
            ns: Some(scratch),
            cascade: true,
        });
        let r = out?;
        dropped?;
        Ok(r)
    }

    /// The merge proper: replay into `target`, skipping the statements
    /// `log_ns`'s merge log already has. A dry run reads the real target's
    /// log and writes its rows under the scratch branch (the discard drops
    /// them).
    fn merge_into(&'s self, opts: &MergeOpts, from: &str, target: &str, log_ns: &str) -> Result<MergeResult> {
        let mut res = MergeResult {
            from: from.to_string(),
            into: target.to_string(),
            ..Default::default()
        };
        let sf = self.mem_scope(from)?;
        // the branch's own statements: its source rows, minus those copied up
        let copied: BTreeMap<String, String> = self
            .mem_q(
                &format!("?[id, from] := *{MEM_COPIED}{{ns: $ns, id, from}}"),
                vec![("ns", s(from))],
            )?
            .rows
            .into_iter()
            .filter_map(|r| Some((cell_str(&r[0])?, cell_str(&r[1])?)))
            .collect();
        let mut stmts: BTreeMap<String, (i64, Option<String>, String, Vec<String>)> = BTreeMap::new();
        for r in self
            .mem_q(
                &format!("?[id, at, stmt, source, text] := *{MEM_SOURCE}{{ns: $ns, id, at, stmt, source, text}}"),
                vec![("ns", s(from))],
            )?
            .rows
        {
            let (Some(id), Some(at), Some(stmt), Some(text)) =
                (cell_str(&r[0]), r[1].get_int(), cell_str(&r[2]), cell_str(&r[4]))
            else {
                continue;
            };
            if let Some(o) = copied.get(&id) {
                let inherited = !self
                    .mem_q(
                        &format!("?[x] := *{MEM_SOURCE}{{ns: $o, id: $id, at: $at, stmt: $stmt}}, x = 1"),
                        vec![("o", s(o)), ("id", s(&id)), ("at", DataValue::from(at)), ("stmt", s(&stmt))],
                    )?
                    .rows
                    .is_empty();
                if inherited {
                    continue;
                }
            }
            let e = stmts.entry(stmt).or_insert((at, cell_str(&r[3]), text, vec![]));
            e.0 = e.0.min(at);
            e.3.push(id);
        }
        let done: BTreeSet<String> = self
            .mem_q(
                &format!("?[stmt] := *{MEM_MERGE}{{into: $into, stmt}}"),
                vec![("into", s(log_ns))],
            )?
            .rows
            .into_iter()
            .filter_map(|r| cell_str(&r[0]))
            .collect();
        let mut order: Vec<(i64, String)> = vec![];
        for (stmt, v) in &stmts {
            if done.contains(stmt) {
                res.skipped += 1;
            } else {
                order.push((v.0, stmt.clone()));
            }
        }
        order.sort();
        // what the branch did with its own memories: forgotten (retracted
        // without a supersedes link) memories are not replayed
        let own_ids: Vec<String> = order.iter().flat_map(|(_, st)| stmts[st].3.clone()).unique().collect();
        let current: BTreeSet<String> = self.mem_valid(&sf, &own_ids, None)?.into_iter().collect();
        let ended_by: BTreeSet<String> = self
            .mem_q(
                &format!("?[dst] := *{MEM_LINK}{{ns: $ns, rel: 'supersedes', dst}}"),
                vec![("ns", s(from))],
            )?
            .rows
            .into_iter()
            .filter_map(|r| cell_str(&r[0]))
            .collect();
        let forgot = |id: &str| !current.contains(id) && !ended_by.contains(id);
        let view = self.entity_view_lineage(&sf)?;
        let now = now_micros();
        let mut log_rows = vec![];
        for (at, stmt) in order {
            let (_, source, text, ids) = &stmts[&stmt];
            let mut ms = MergeStatement {
                stmt: stmt.clone(),
                text: text.clone(),
                at,
                ..Default::default()
            };
            let own: Vec<&String> = ids.iter().filter(|i| memory_recall::statement_of(i) == stmt).collect();
            if !own.is_empty() && own.iter().all(|i| forgot(i)) {
                ms.action = "skipped".to_string();
                log_rows.push(DataValue::List(vec![
                    s(target),
                    s(&stmt),
                    s(from),
                    DataValue::from(now),
                    s("skipped"),
                    DataValue::List(vec![]),
                ]));
                res.replayed.push(ms);
                continue;
            }
            // the options it was stored with, from its memory rows
            let first = own.first().map(|x| x.to_string()).unwrap_or_else(|| ids[0].clone());
            // the latest assertion's kind and importance
            let meta = self.mem_q(
                &format!("?[t, kind, importance] := *{MEM}{{ns: $ons, id: $id, vt, kind, importance}}, to_bool(vt), t = to_int(vt)"),
                vec![("ons", s(&self.owner_ns(&sf, &first)?)), ("id", s(&first))],
            )?;
            let (kind, importance) = meta
                .rows
                .iter()
                .max_by_key(|r| r[0].get_int())
                .map(|r| (cell_str(&r[1]), r[2].get_float()))
                .unwrap_or((None, None));
            let role = self.roles_of(&sf, &[first.clone()])?.remove(&first);
            let expires = self.life_of(&sf, &[first.clone()])?.get(&first).and_then(|x| x.0);
            let mut entities: Vec<String> = vec![];
            for (_, e) in self.mentions_of(&sf, &ids.iter().cloned().collect::<Vec<_>>())? {
                let c = view.canonical(&e);
                if let Some((name, _)) = view.rows.get(&c).or_else(|| view.rows.get(&e)) {
                    if !entities.contains(name) {
                        entities.push(name.clone());
                    }
                }
            }
            let split = (own.len() == 1 && *own[0] == stmt).then(|| "none".to_string());
            let r = self.remember(
                text,
                RememberOpts {
                    ns: Some(target.to_string()),
                    kind,
                    source: source.clone(),
                    importance,
                    entities,
                    at: opts.keep_time.then_some(at),
                    expires: expires.filter(|e| *e > now),
                    split,
                    role,
                    ..Default::default()
                },
            )?;
            ms.action = r.action.clone();
            ms.ids = match &r.claims {
                c if !c.is_empty() => c.iter().filter_map(|x| x.id.clone().or(x.duplicate_of.clone())).collect(),
                _ => r.id.clone().or(r.duplicate_of.clone()).into_iter().collect(),
            };
            ms.superseded = r.superseded.clone();
            ms.duplicate_of = r.duplicate_of.clone();
            ms.tentative = r.tentative.clone();
            // claims the branch forgot are retracted in the target too
            let gone: BTreeSet<String> = own
                .iter()
                .filter(|i| forgot(i))
                .filter_map(|i| self.mem_texts(&sf, &[i.to_string()]).ok()?.remove(i.as_str()))
                .map(|t| norm_text(&t))
                .collect();
            if !gone.is_empty() {
                let st = self.mem_scope(target)?;
                let texts = self.mem_texts(&st, &ms.ids)?;
                let drop: Vec<String> =
                    texts.into_iter().filter(|(_, t)| gone.contains(&norm_text(t))).map(|x| x.0).collect();
                if !drop.is_empty() {
                    self.forget(ForgetOpts {
                        ns: Some(target.to_string()),
                        ids: drop,
                        ..Default::default()
                    })?;
                }
            }
            log_rows.push(DataValue::List(vec![
                s(target),
                s(&stmt),
                s(from),
                DataValue::from(now),
                s(&ms.action),
                str_list(&ms.ids),
            ]));
            res.statements += 1;
            res.replayed.push(ms);
        }
        // forgets of inherited memories (no supersedes link explains them)
        if opts.forgets {
            let inherited: Vec<String> = copied.keys().filter(|id| forgot(id)).cloned().collect();
            let inherited: Vec<String> = {
                let cur: BTreeSet<String> = self.mem_valid(&sf, &inherited, None)?.into_iter().collect();
                inherited.into_iter().filter(|i| !cur.contains(i)).collect()
            };
            if !inherited.is_empty() {
                let st = self.mem_scope(target)?;
                let live: BTreeSet<String> = self.mem_valid(&st, &inherited, None)?.into_iter().collect();
                let key = |id: &str| format!("forget:{id}");
                for id in inherited {
                    if done.contains(&key(&id)) {
                        continue;
                    }
                    if live.contains(&id) {
                        res.forgets.push(id.clone());
                    } else {
                        res.forgets_skipped.push(id.clone());
                    }
                    log_rows.push(DataValue::List(vec![
                        s(target),
                        s(&key(&id)),
                        s(from),
                        DataValue::from(now),
                        s(if live.contains(&id) { "forgotten" } else { "skipped" }),
                        str_list(&[id]),
                    ]));
                }
                if !res.forgets.is_empty() {
                    self.forget(ForgetOpts {
                        ns: Some(target.to_string()),
                        ids: res.forgets.clone(),
                        ..Default::default()
                    })?;
                }
            }
        }
        if !log_rows.is_empty() {
            self.mem_q(
                &format!(
                    "?[into, stmt, from, at, action, ids] <- $rows :put {MEM_MERGE} {{into, stmt => from, at, action, ids}}"
                ),
                vec![("rows", DataValue::List(log_rows))],
            )?;
        }
        Ok(res)
    }

    /// The namespace that owns `id` in the view (the scope's own when none).
    fn owner_ns(&'s self, sc: &Scope, id: &str) -> Result<String> {
        let l = self.mem_owners(sc, &[id.to_string()])?.get(id).copied().unwrap_or(0);
        Ok(sc.levels[l].ns.clone())
    }

    /// The latest text of each id, from its owner (any version).
    pub(crate) fn mem_texts(&'s self, sc: &Scope, ids: &[String]) -> Result<BTreeMap<String, String>> {
        let mut out = BTreeMap::new();
        for (l, group) in self.by_owner(sc, ids)? {
            for r in self
                .mem_q(
                    &format!("?[id, text] := id in $ids, *{MEM}{{ns: $ns, id, text}}"),
                    vec![("ns", s(&sc.levels[l].ns)), ("ids", str_list(&group))],
                )?
                .rows
            {
                if let (Some(id), Some(t)) = (cell_str(&r[0]), cell_str(&r[1])) {
                    out.insert(id, t);
                }
            }
        }
        Ok(out)
    }
}
