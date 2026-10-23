/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Agent memory API (UPGRADE_PLAN H1 + E10; docs/specs/memory-api.md):
//! `remember` / `recall` / `forget` / `reflect` / `link` on a ready-made,
//! namespaced memory schema, so an agent never has to write CozoScript.
//!
//! Everything here is built from existing parts: valid time (`Validity`) and
//! E1-style supersession by appended retraction rows, `::erase` for GDPR
//! deletes, `::changes` for the change feed, the embedder registry, FTS and
//! filtered HNSW joined by `WeightedFusion`, and `decide()` for the write-path
//! decisions. With no embedder or decider enabled it degrades to add-only
//! plus FTS (and the model-free value rules); it never errors because ML is
//! off.
//!
//! Entity names passed to `remember` are resolved with `entity.rs` (E3) over
//! the namespace's own entities and gate supersession to statements about a
//! shared subject. Recall ranks by a configurable blend of relevance,
//! recency, importance and access counts (E4, C6), with TTL expiry and an
//! archive tier.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use itertools::Itertools;
use miette::{IntoDiagnostic, Result};
use serde_derive::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::bail;
use crate::data::value::{DataValue, Validity};
use crate::decide::{self, rules, Band, Decision, Template};
use crate::embed;
use crate::entity::{EntityConfig, Mention, Outcome, ResolveOptions, ENTITY_LINKS, ENTITY_MERGES};
use crate::memory_claims;
use crate::memory_recall;
use crate::memory_engram::{self as engram_hook, Job};
use crate::runtime::db::{Db, NamedRows, ScriptMutability};
use crate::storage::Storage;

/// Memories: `{ns, id, vt: Validity => text, kind, source, created, importance}`.
pub const MEM: &str = "kleos_mem";
/// Memory vectors: `{ns, id => current, vec}` (created once an embedder is set).
pub const MEM_VEC: &str = "kleos_mem_vec";
/// Entities: `{ns, eid => name, norm, kind, created}`.
pub const MEM_ENTITY: &str = "kleos_mem_entity";
/// Memory → entity mentions: `{ns, id, eid}`.
pub const MEM_MENTION: &str = "kleos_mem_mention";
/// Typed links between memories and entities: `{ns, src, rel, dst => created, p, decided_by}`.
pub const MEM_LINK: &str = "kleos_mem_link";
/// Statements behind each memory: `{ns, id, at, stmt => source, text}`.
pub const MEM_SOURCE: &str = "kleos_mem_source";
/// Decision audit: `{ns, new, cand, template => label, p, band, decided_by, reason, status, at}`.
pub const MEM_DECISION: &str = "kleos_mem_decision";
/// Store configuration: `{key => value}`.
pub const MEM_CONFIG: &str = "kleos_mem_config";
/// Per-namespace commit version: `{ns => seq}`. Every `remember` asserts in
/// its write transaction that the version it planned against is unchanged,
/// and bumps it; so does `forget`. The storage engine's transaction
/// isolation then orders concurrent writers, in one process or several.
pub const MEM_NS: &str = "kleos_mem_ns";
/// Access counts: `{ns, id => n, last}`. Written by batched flushes of the
/// counts `recall` buffers, in their own small transactions (never under the
/// namespace version).
pub const MEM_ACCESS: &str = "kleos_mem_access";
/// Expiry and archive tier: `{ns, id => expires, archived}` (µs; null when
/// unset). Only memories with a TTL or in the archive have a row.
pub const MEM_LIFE: &str = "kleos_mem_life";
/// Who said a memory: `{ns, id => role}` (`user`, `assistant` or `other`).
/// Only memories stored with a `role` have a row; none means the user.
pub const MEM_ROLE: &str = "kleos_mem_role";
/// Current schema version (2 added `kleos_mem_ns`; 3 added `kleos_mem_access`
/// and `kleos_mem_life`, and entity merges / review links; 4 added
/// `kleos_mem_role`).
const SCHEMA_VERSION: i64 = 4;

const DEFAULT_NS: &str = "default";
const DEFAULT_KIND: &str = "fact";
/// Candidates scored per `remember`.
const DEFAULT_K: usize = 5;
/// Vector-leg weight of the hybrid fusion (the SciFact-tuned 0.6, CHANGELOG).
const DEFAULT_VECTOR_WEIGHT: f64 = 0.6;
/// A `supersedes` answer at or above this p, but below the calibrated
/// threshold, is kept for review. The tuned decider-nano calibration has
/// threshold = τ_auto (0.997), so it has no tentative band of its own.
const REVIEW_P: f64 = 0.5;
/// Claims of a split statement: a `supersedes` answer is kept for review
/// only from this p (and only the claim's likeliest one). On the held-out
/// LongMemEval knowledge-update questions, pending answers below 0.9 were
/// almost all unrelated sentences on one topic ("I've been using Duolingo"
/// against "I've been taking a course on Coursera"); the life stream's
/// three true misses scored 0.977 to 0.996.
const CLAIM_REVIEW_P: f64 = 0.9;
/// The same for a tentative `duplicate` answer of a claim.
const CLAIM_DUP_REVIEW_P: f64 = 0.5;
/// Claim mode: a candidate whose vector is less similar than this (cosine)
/// to the claim's is not sent to the decider (see "Claims" in the spec).
const DEFAULT_MIN_SIMILARITY: f64 = 0.6;
/// Lock-free planning attempts of a `remember` before it plans under the
/// in-process commit lock.
const OPTIMISTIC_ATTEMPTS: usize = 2;
/// Attempts of a `remember` in all, counting the ones that lost a race with
/// another process; after that it reports the namespace as busy.
const COMMIT_ATTEMPTS: usize = 8;

/// Recall blend defaults (docs/specs/memory-api.md, "Ranking"): final score
/// = relevance + w_r · 2^(−age / half-life) + w_i · (importance − 0.5)
/// + w_a · min(1, ln(1 + accesses) / ln(1 + 100)). Access counts are
/// recorded but weigh 0 by default: every returned hit counts, so filler
/// hits of unrelated queries gain rank (on the life stream a 0.05 weight
/// flipped a near-tie the wrong way). Recency and importance are small
/// because the fused relevance of the top hits is compressed (EmbeddingGemma
/// near-ties differ by 0.01 or less): on the life stream read at its own
/// "now", recency 0.1 / 30 days lost 3 of 20 answers, 0.03 / 90 days none.
const DEFAULT_RECENCY_WEIGHT: f64 = 0.03;
const DEFAULT_HALF_LIFE_DAYS: f64 = 90.0;
const DEFAULT_IMPORTANCE_WEIGHT: f64 = 0.1;
const DEFAULT_ACCESS_WEIGHT: f64 = 0.0;
/// A question or request ("how many stars do I need for gold?") is as
/// relevant to a question as the fact that answers it, and conversational
/// text is full of them; the blend ranks it this much lower.
const DEFAULT_QUESTION_WEIGHT: f64 = 0.05;
/// Recall diversity (docs/specs/memory-api.md, "Diversity"): at most this
/// many hits per statement (the claims of one message) and per source group
/// (`chat-12#4` → `chat-12`), and the MMR-style penalty on a candidate's
/// similarity to the hits already chosen. The pool chosen from is
/// `max(DIVERSITY_POOL_MULT · k, k + 10)` deep.
const DEFAULT_MAX_PER_STATEMENT: usize = 1;
const DEFAULT_MAX_PER_SOURCE: usize = 0;
const DEFAULT_MMR_WEIGHT: f64 = 0.0;
const DIVERSITY_POOL_MULT: usize = 3;
/// Subtracted from the score of an assistant memory (`remember {role}`).
const DEFAULT_ASSISTANT_WEIGHT: f64 = 0.05;
/// Entity expansion: from the `EXPAND_FROM` most relevant hits, other
/// memories about the same entities join the pool at `expand_weight` (default
/// 0.5) times the seed's relevance; entities mentioned by more than
/// `EXPAND_MAX_DEGREE` memories (a speaker, the user) are not followed.
const DEFAULT_EXPAND_WEIGHT: f64 = 0.5;
const EXPAND_FROM: usize = 3;
const EXPAND_MAX_DEGREE: usize = 25;
const EXPAND_MAX_NEW: usize = 30;
/// Default token budget of `recall {format: 'context'}`.
const DEFAULT_CONTEXT_BUDGET: usize = 2000;
/// Roles `remember` accepts.
const ROLES: [&str; 3] = ["user", "assistant", "other"];
/// Accesses at which the access signal saturates (reaches 1).
const ACCESS_SATURATION: f64 = 100.0;
/// Buffered access counts are flushed once this many memories are pending,
/// or this long after the last flush, by the `recall` that notices.
const ACCESS_FLUSH_KEYS: usize = 64;
const ACCESS_FLUSH_EVERY: Duration = Duration::from_secs(1);
/// At most this many memories are buffered when flushes keep failing (a
/// read-only store): the counts are approximate, never unbounded.
const ACCESS_BUFFER_CAP: usize = 100_000;
const MICROS_PER_DAY: f64 = 86_400_000_000.0;

/// Access counts `recall` has not written yet, per database. Recording an
/// access is a map update; the counts are written in batches by
/// [`Db::memory_flush`] in a transaction of their own, which never touches
/// the namespace version, so reads do not contend with `remember`. Counts
/// still in the buffer when the process exits are lost: access counts are a
/// ranking signal, not an audit log.
#[derive(Default)]
pub(crate) struct AccessBuf {
    inner: Mutex<AccessPending>,
}

#[derive(Default)]
struct AccessPending {
    /// (ns, id) → (accesses, last access µs)
    counts: HashMap<(String, String), (i64, i64)>,
    last_flush: Option<Instant>,
}

impl AccessBuf {
    fn lock(&self) -> MutexGuard<'_, AccessPending> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
    /// Count one access of each id; returns whether a flush is due.
    fn add(&self, ns: &str, ids: &[String], at: i64) -> bool {
        let mut g = self.lock();
        for id in ids {
            if g.counts.len() >= ACCESS_BUFFER_CAP && !g.counts.contains_key(&(ns.to_string(), id.clone())) {
                break;
            }
            let e = g.counts.entry((ns.to_string(), id.clone())).or_insert((0, at));
            e.0 += 1;
            e.1 = e.1.max(at);
        }
        let due = g.last_flush.is_none_or(|t| t.elapsed() >= ACCESS_FLUSH_EVERY);
        !g.counts.is_empty() && (g.counts.len() >= ACCESS_FLUSH_KEYS || due)
    }
    /// Buffered (accesses, last access) of one memory.
    fn get(&self, ns: &str, id: &str) -> Option<(i64, i64)> {
        self.lock().counts.get(&(ns.to_string(), id.to_string())).copied()
    }
    fn take(&self) -> HashMap<(String, String), (i64, i64)> {
        let mut g = self.lock();
        g.last_flush = Some(Instant::now());
        std::mem::take(&mut g.counts)
    }
    /// Put back counts whose flush failed.
    fn restore(&self, m: HashMap<(String, String), (i64, i64)>) {
        let mut g = self.lock();
        for (k, (n, l)) in m {
            let e = g.counts.entry(k).or_insert((0, l));
            e.0 += n;
            e.1 = e.1.max(l);
        }
    }
    /// Drop buffered counts of erased memories.
    fn forget(&self, ns: &str, ids: &[String]) {
        let mut g = self.lock();
        for id in ids {
            g.counts.remove(&(ns.to_string(), id.clone()));
        }
    }
}

/// In-process commit gate of one namespace. Correctness comes from the
/// `MEM_NS` version row checked inside the write transaction, which also
/// covers other processes sharing the database; the gate only makes writers
/// in one process take turns instead of colliding there: `remember` plans
/// (candidate search, model calls) without holding anything, then commits
/// under `lock` if `seq` is unchanged, and plans again otherwise. Keyed by
/// namespace name only (shared across databases in one process, which can
/// only cost a spurious re-plan).
struct NsGate {
    lock: Mutex<()>,
    seq: AtomicU64,
}

/// A held commit lock; bumps the namespace's sequence when released.
struct Commit<'a> {
    gate: &'a NsGate,
    _held: MutexGuard<'a, ()>,
}

impl NsGate {
    fn seq(&self) -> u64 {
        self.seq.load(Ordering::SeqCst)
    }
    fn commit(&self) -> Commit<'_> {
        Commit {
            gate: self,
            _held: self.lock.lock().unwrap_or_else(|e| e.into_inner()),
        }
    }
}

impl Drop for Commit<'_> {
    fn drop(&mut self) {
        // runs before the guard is released
        self.gate.seq.fetch_add(1, Ordering::SeqCst);
    }
}

static NS_GATES: LazyLock<Mutex<HashMap<String, Arc<NsGate>>>> = LazyLock::new(Default::default);

fn ns_gate(ns: &str) -> Arc<NsGate> {
    NS_GATES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry(ns.to_string())
        .or_insert_with(|| {
            Arc::new(NsGate {
                lock: Mutex::new(()),
                seq: AtomicU64::new(0),
            })
        })
        .clone()
}

/// Bumps a namespace's `MEM_NS` version (as one `{}` block of a script).
fn bump_ns_head() -> String {
    bump_ns_head_by(1)
}

/// Advances a namespace's `MEM_NS` version by `n` (a `remember` that
/// enqueues `n` Engram jobs numbers them with the `n` versions it takes).
fn bump_ns_head_by(n: usize) -> String {
    format!(
        "{{cur[max(s)] := *{MEM_NS}{{ns: $ns, seq: s}} \
           cur[max(s)] := s = 0 \
           ?[ns, seq] := cur[s0], ns = $ns, seq = s0 + {n} \
          :put {MEM_NS} {{ns => seq}}}}"
    )
}

/// The memory id of claim `i` (0-based) of a multi-claim statement.
fn claim_id(stmt: &str, i: usize) -> String {
    format!("{stmt}.{}", i + 1)
}

/// Whether `text` mentions the entity name `name` (case-insensitive, at
/// word boundaries).
fn mentions_name(text: &str, name: &str) -> bool {
    let n = norm_name(name);
    if n.is_empty() {
        return false;
    }
    let t = text.to_lowercase();
    let mut from = 0;
    while let Some(off) = t[from..].find(&n) {
        let a = from + off;
        let b = a + n.len();
        let before = t[..a].chars().next_back().map_or(true, |c| !c.is_alphanumeric());
        let after = t[b..].chars().next().map_or(true, |c| !c.is_alphanumeric());
        if before && after {
            return true;
        }
        from = a + t[a..].chars().next().map_or(1, |c| c.len_utf8());
    }
    false
}

/// A stored vector as f32s.
fn vec_of(v: &DataValue) -> Option<Vec<f32>> {
    match v {
        DataValue::Vec(crate::data::value::Vector::F32(a)) => Some(a.to_vec()),
        DataValue::Vec(crate::data::value::Vector::F64(a)) => Some(a.iter().map(|x| *x as f32).collect()),
        DataValue::List(l) => l.iter().map(|x| x.get_float().map(|f| f as f32)).collect(),
        _ => None,
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        d / (na.sqrt() * nb.sqrt())
    }
}

/// A write that lost to a concurrent transaction (storage-level conflict or
/// lock), as opposed to a real error.
fn is_contention(e: &miette::Report) -> bool {
    let m = format!("{e:?}").to_ascii_lowercase();
    ["busy", "locked", "conflict", "try again"].iter().any(|k| m.contains(k))
}

/// What a `remember` will do, decided before anything is written.
struct Plan {
    at: i64,
    claims: Vec<ClaimPlan>,
    ents: EntityPlan,
    res: RememberResult,
}

/// One claim of a `remember`, planned.
struct ClaimPlan {
    /// Memory id it gets when added.
    id: String,
    text: String,
    vec: Option<Vec<f32>>,
    /// Declarative: takes part in decisions.
    decided: bool,
    cands: Vec<Cand>,
    sup: Vec<Option<Decision>>,
    dup: Vec<Option<Decision>>,
    supersede: Vec<usize>,
    duplicate: Option<(usize, f64)>,
    audit: Vec<MemDecision>,
    /// Indices into the entity plan's `out` of the entities it mentions.
    ents: Vec<usize>,
}

fn now_micros() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// A time-sortable id with a prefix (`m` memories, `e` entities).
fn new_id(prefix: &str) -> String {
    format!(
        "{prefix}_{:012x}{:06x}",
        now_micros(),
        rand::random::<u32>() & 0xff_ffff
    )
}

/// Text identity for the exact-duplicate rule: lower case, words only.
fn norm_text(s: &str) -> String {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .join(" ")
}

/// Entity-name normal form: lower case, inner whitespace collapsed, outer
/// punctuation trimmed.
pub fn norm_name(s: &str) -> String {
    s.split_whitespace()
        .join(" ")
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
}

/// FTS query from free text: its words, OR-ed (operators and quotes in the
/// user's text are not FTS syntax here).
fn fts_query(s: &str) -> String {
    let mut seen = BTreeSet::new();
    // letters and ASCII digits only: the FTS grammar's words are XID_CONTINUE
    s.split(|c: char| !(c.is_alphabetic() || c.is_ascii_digit()))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .filter(|w| seen.insert(w.clone()))
        .take(32)
        .join(" OR ")
}

fn ns_of(ns: &Option<String>) -> String {
    match ns {
        Some(s) if !s.is_empty() => s.clone(),
        _ => DEFAULT_NS.to_string(),
    }
}

fn s(v: &str) -> DataValue {
    DataValue::from(v)
}

fn opt_s(v: &Option<String>) -> DataValue {
    v.as_deref().map(DataValue::from).unwrap_or(DataValue::Null)
}

fn str_list(v: &[String]) -> DataValue {
    DataValue::List(v.iter().map(|x| s(x)).collect())
}

fn cell_str(v: &DataValue) -> Option<String> {
    v.get_str().map(|x| x.to_string())
}

fn params(p: Vec<(&str, DataValue)>) -> BTreeMap<String, DataValue> {
    p.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

// ------------------------------------------------------------------ options

/// Options of [`Db::memory_init`] / `::memory init`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InitOpts {
    /// Embedder for memory vectors (an enabled `::embedder` name). Setting it
    /// on an existing store adds vectors and backfills the memories without one.
    pub embedder: Option<String>,
    /// Decider used by `remember` when none is passed per call.
    pub decider: Option<String>,
    /// Registered Engram model whose overlay follows the memories
    /// (`remember` / `forget` enqueue Engram jobs); `''` detaches.
    #[cfg(feature = "engram")]
    pub engram: Option<String>,
}

/// Options of [`Db::remember`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RememberOpts {
    /// Namespace (agent / user scope); default `default`.
    pub ns: Option<String>,
    /// Memory kind; default `fact`.
    pub kind: Option<String>,
    /// Where the statement came from (conversation, document, tool).
    pub source: Option<String>,
    /// Importance in [0, 1], if the caller knows it.
    pub importance: Option<f64>,
    /// Entity names the statement is about (resolved with entity
    /// resolution against the namespace's entities). Include the subject,
    /// e.g. `user` for first-person statements: supersession only acts
    /// between statements that share an entity when both have some.
    pub entities: Vec<String>,
    /// Valid-from time in µs since the epoch; default now.
    pub at: Option<i64>,
    /// Time to live in µs from `at`: past it, recall leaves the memory out
    /// unless asked (`include_expired`), and `forget {mode: 'archive',
    /// expired: true}` moves it to the archive.
    pub ttl: Option<i64>,
    /// Absolute expiry (µs); the alternative to `ttl`.
    pub expires: Option<i64>,
    /// Candidates to score; default 5.
    pub k: Option<usize>,
    /// Decider name; default the store's, else the first enabled one.
    pub decider: Option<String>,
    /// Embedder name for the first `remember` of a new store.
    pub embedder: Option<String>,
    /// `auto` (default): a text of several sentences is stored as one
    /// memory per sentence (claim), each decided on its own; `none`: the
    /// whole text is one memory, decided as one (the behaviour before
    /// claims). A single sentence is one memory either way.
    pub split: Option<String>,
    /// Cosine similarity below which a candidate of a claim is not sent to
    /// the decider (default 0.6). Only applies to the claims of a split
    /// statement, when both texts are embedded.
    pub min_similarity: Option<f64>,
    /// Who said it: `user` (the default when unset), `assistant` or
    /// `other`. Assistant and other memories are stored and recalled (an
    /// assistant memory at a lower weight, `recall {assistant_weight}`) but
    /// never take part in supersede / duplicate decisions: what the
    /// assistant said does not replace what the user said.
    pub role: Option<String>,
}

/// Options of [`Db::recall`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RecallOpts {
    /// Namespace; default `default`.
    pub ns: Option<String>,
    /// Hits to return; default 10.
    pub k: Option<usize>,
    /// Only this kind.
    pub kind: Option<String>,
    /// Only memories mentioning this entity (by name).
    pub entity: Option<String>,
    /// Valid time to read at (µs); default now, current memories only.
    pub as_of: Option<i64>,
    /// Vector-leg weight in [0, 1] (text gets the rest); default 0.6.
    pub vector_weight: Option<f64>,
    /// Leave out memories that a memory valid at the read time may have
    /// superseded (a pending `supersedes` decision); default false: return
    /// them, flagged in `pending_superseded_by`.
    pub hide_pending: bool,
    /// Rank by the blend of relevance, recency, importance and access count
    /// (default true); false ranks by relevance alone, as before E4.
    pub blend: Option<bool>,
    /// Blend weight of recency (default 0.03).
    pub recency_weight: Option<f64>,
    /// Recency half-life in days (default 90).
    pub half_life_days: Option<f64>,
    /// Blend weight of importance (default 0.1).
    pub importance_weight: Option<f64>,
    /// Blend weight of the access count (default 0: recorded, not ranked).
    pub access_weight: Option<f64>,
    /// Subtracted from the blended score of a memory that is a question or
    /// a request rather than a statement (default 0.05; see "Claims").
    pub question_weight: Option<f64>,
    /// Also return archived memories (default false).
    pub include_archived: bool,
    /// Also return memories past their TTL (default false).
    pub include_expired: bool,
    /// Count this recall as an access of the memories it returns (default
    /// true; false in read-only scripts).
    pub record_access: Option<bool>,
    /// Diverse selection from a deeper pool (default true; see the caps and
    /// `mmr_weight`). False: the top k by score, exactly as before.
    pub diversity: Option<bool>,
    /// At most this many hits per statement, i.e. claims of one text
    /// (default 1; 0 = no cap). Hits beyond the caps only fill up to k.
    pub max_per_statement: Option<usize>,
    /// At most this many hits per source group: the source up to its first
    /// `#` (`chat-12#4` → `chat-12`) (default 0 = no cap).
    pub max_per_source: Option<usize>,
    /// MMR-style penalty: a candidate's score minus this times its highest
    /// similarity (vector cosine, else word overlap) to the hits already
    /// chosen (default 0).
    pub mmr_weight: Option<f64>,
    /// Candidates the diverse selection chooses from (default max(3k, k + 10)).
    pub pool: Option<usize>,
    /// Entity-graph expansion (default false): other current memories about
    /// the entities of the top hits join the pool at a reduced weight.
    pub expand: Option<bool>,
    /// Weight of an expanded memory: this times its seed hit's relevance
    /// (default 0.5).
    pub expand_weight: Option<f64>,
    /// Subtracted from the score of an assistant memory (default 0.05).
    pub assistant_weight: Option<f64>,
    /// Only memories of this role (`user` includes memories with no role).
    pub role: Option<String>,
    /// Only memories whose validity starts at or after this time (µs).
    pub from: Option<i64>,
    /// Only memories whose validity starts before this time (µs).
    pub to: Option<i64>,
    /// `list` (the default: hits) or `context` (a token-budgeted text
    /// block for an LLM reader, [`Db::recall_context`]).
    pub format: Option<String>,
    /// Token budget of the context block (default 2000).
    pub budget: Option<usize>,
    /// Show each memory's source in the context block (default false).
    pub show_source: bool,
}

/// Options of [`Db::memory_set`]: change a memory's importance, expiry or
/// archive state. These are scoring metadata: the memory's text and validity
/// are untouched.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SetOpts {
    /// Namespace; default `default`.
    pub ns: Option<String>,
    /// One memory id.
    pub id: Option<String>,
    /// Several memory ids.
    pub ids: Vec<String>,
    /// New importance in [0, 1].
    pub importance: Option<f64>,
    /// Expire this long (µs) after now.
    pub ttl: Option<i64>,
    /// Expire at this time (µs).
    pub expires: Option<i64>,
    /// Remove the expiry.
    pub no_ttl: bool,
    /// Move to (true) or out of (false) the archive.
    pub archived: Option<bool>,
}

/// Options of [`Db::forget`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ForgetOpts {
    /// Namespace; default `default`.
    pub ns: Option<String>,
    /// One memory id.
    pub id: Option<String>,
    /// Several memory ids.
    pub ids: Vec<String>,
    /// Filter: every current memory of this kind.
    pub kind: Option<String>,
    /// Filter: every current memory mentioning this entity.
    pub entity: Option<String>,
    /// Filter: every current memory past its TTL (not yet archived).
    pub expired: bool,
    /// Required to forget a whole namespace with no other filter.
    pub all: bool,
    /// `retract` (end validity; default), `erase` (GDPR hard delete) or
    /// `archive` (move to the archive tier; still valid, hidden from recall
    /// by default).
    pub mode: Option<String>,
    /// Valid time of the retraction (µs); default now.
    pub at: Option<i64>,
}

/// Options of [`Db::reflect`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReflectOpts {
    /// Namespace; default `default`.
    pub ns: Option<String>,
    /// Changes since this valid time (µs); default 7 days ago.
    pub since: Option<i64>,
    /// Cap per section; default 50.
    pub limit: Option<usize>,
}

/// Options of [`Db::link`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LinkOpts {
    /// Namespace; default `default`.
    pub ns: Option<String>,
    /// Source memory or entity id.
    pub from: String,
    /// Target memory or entity id.
    pub to: String,
    /// Relation type, e.g. `works_at`, `about`, `follows_from`.
    pub rel: String,
    /// Confidence, if any.
    pub p: Option<f64>,
}

// ------------------------------------------------------------------ results

/// One decision `remember` took about one candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemDecision {
    /// Candidate memory id.
    pub candidate: String,
    /// Candidate text.
    pub candidate_text: String,
    /// `supersedes` or `duplicate`.
    pub template: String,
    /// `yes` / `no`.
    pub label: String,
    /// P(yes).
    pub p: f64,
    /// `auto`, `tentative` or `none`.
    pub band: String,
    /// `model|template@vN|via`.
    pub decided_by: String,
    /// Why a rule decided.
    pub reason: Option<String>,
    /// `superseded`, `duplicate`, `pending` or `none`.
    pub action: String,
    /// The claim (memory id) this decision is about, when the statement was
    /// split into several claims.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim: Option<String>,
    /// Cosine similarity of the two texts' vectors, when both are embedded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub similarity: Option<f64>,
}

/// What `remember` did with one claim of a multi-sentence statement.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ClaimResult {
    /// The claim's memory id (`<statement id>.<n>`); `None` for a duplicate.
    pub id: Option<String>,
    /// The claim's text.
    pub text: String,
    /// `added` or `duplicate`.
    pub action: String,
    /// The existing memory this claim duplicates.
    pub duplicate_of: Option<String>,
    /// Memories whose validity this claim ended.
    pub superseded: Vec<String>,
    /// Whether the claim took part in supersede / duplicate decisions (a
    /// declarative claim: not a question, request or short fragment).
    pub decided: bool,
}

/// How a name passed in `entities` was resolved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityLink {
    /// The name as passed.
    pub name: String,
    /// Entity row the memory mentions.
    pub eid: String,
    /// Its canonical entity (after merges).
    pub canonical: String,
    /// `match` (an existing entity), `possible` (a new entity with review
    /// links to the likely matches) or `new`.
    pub outcome: String,
    /// Whether this call created the entity row.
    pub created: bool,
    /// Decision provenance (`rule|exact`, `model|same_entity@v2|...`).
    pub decided_by: String,
    /// P(same entity) of the match, when matched or possible.
    pub p: f64,
}

/// What [`Db::remember`] did.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RememberResult {
    /// Namespace.
    pub ns: String,
    /// `added` or `duplicate`.
    pub action: String,
    /// The new memory's id (`None` for a duplicate).
    pub id: Option<String>,
    /// Memories whose validity this one ended.
    pub superseded: Vec<String>,
    /// The existing memory this statement duplicates.
    pub duplicate_of: Option<String>,
    /// Decisions recorded as pending (see `reflect`).
    pub tentative: Vec<MemDecision>,
    /// Every decision taken, including `none`.
    pub decisions: Vec<MemDecision>,
    /// Linked entity ids.
    pub entities: Vec<String>,
    /// How each entity name was resolved.
    pub entity_links: Vec<EntityLink>,
    /// Candidates whose `supersedes` check was skipped: both statements
    /// have entities and share none (different subjects).
    pub gated: Vec<String>,
    /// Whether the text was embedded.
    pub embedded: bool,
    /// Decider used (`None`: model-free rules only).
    pub decider: Option<String>,
    /// The claims, when the text was split into several (`id` is then the
    /// statement id: `forget` of it covers every claim).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub claims: Vec<ClaimResult>,
}

/// One recalled memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallHit {
    /// Memory id.
    pub id: String,
    /// Text.
    pub text: String,
    /// Final score (higher is better): the blend, or the relevance when the
    /// blend is off.
    pub score: f64,
    /// Hybrid relevance (the fused vector + text score).
    pub relevance: f64,
    /// Recency signal 2^(−age / half-life), age from `valid_from` to the read
    /// time.
    pub recency: f64,
    /// Times recalled (including counts not yet flushed).
    pub access_count: i64,
    /// Last recall (µs).
    pub last_access: Option<i64>,
    /// Expiry (µs).
    pub expires: Option<i64>,
    /// Whether the memory is in the archive tier.
    pub archived: bool,
    /// Kind.
    pub kind: String,
    /// Source of the first statement.
    pub source: Option<String>,
    /// When it was first stored (µs).
    pub created: i64,
    /// Importance.
    pub importance: Option<f64>,
    /// Start of the validity interval containing the read time (µs).
    pub valid_from: i64,
    /// End of that interval (µs); `None`: still valid.
    pub valid_to: Option<i64>,
    /// Memories valid at the read time that may have superseded this one
    /// (pending `supersedes` decisions, see `reflect`).
    pub pending_superseded_by: Vec<String>,
    /// The whole statement this memory was split from, when it is one claim
    /// of a longer text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    /// Who said it (`remember {role}`); `None`: not recorded (the user).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Pulled in by entity expansion rather than found by the search.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub expanded: bool,
}

/// What [`Db::forget`] did.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ForgetResult {
    /// `retract` or `erase`.
    pub mode: String,
    /// Memories retracted or erased.
    pub ids: Vec<String>,
}

/// A decision waiting for review.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingDecision {
    /// The newer memory.
    pub new: String,
    /// The candidate it may supersede or duplicate.
    pub candidate: String,
    /// Text of `new`.
    pub new_text: Option<String>,
    /// Text of `candidate`.
    pub candidate_text: Option<String>,
    /// Template.
    pub template: String,
    /// Label.
    pub label: String,
    /// P(label).
    pub p: f64,
    /// Band.
    pub band: String,
    /// `model|template@vN|via`.
    pub decided_by: String,
    /// When it was decided (µs).
    pub at: i64,
}

/// One `::changes` row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemChange {
    /// Memory id.
    pub id: String,
    /// `added`, `ended` or `changed`.
    pub change: String,
    /// Valid time (µs).
    pub at: i64,
    /// Text.
    pub text: String,
    /// Kind.
    pub kind: String,
}

/// Two current memories that may not both be true.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContradictionCandidate {
    /// Older memory.
    pub a: String,
    /// Newer memory.
    pub b: String,
    /// Text of `a`.
    pub a_text: String,
    /// Text of `b`.
    pub b_text: String,
    /// Why it was flagged.
    pub reason: String,
}

/// What [`Db::reflect`] returns.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReflectResult {
    /// Namespace.
    pub ns: String,
    /// Tentative decisions not acted on.
    pub pending: Vec<PendingDecision>,
    /// Recent validity changes, oldest first.
    pub changes: Vec<MemChange>,
    /// Current memories that may contradict each other.
    pub contradictions: Vec<ContradictionCandidate>,
}

/// An entity linked to a memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkedEntity {
    /// Entity id.
    pub eid: String,
    /// Name as first seen.
    pub name: String,
    /// Whether it was created by this call.
    pub created: bool,
}

/// Store configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MemConfig {
    /// Embedder of the memory vectors.
    pub embedder: Option<String>,
    /// Vector dimension.
    pub dim: Option<usize>,
    /// Default decider.
    pub decider: Option<String>,
    /// Attached Engram model (feature `engram`).
    #[cfg(feature = "engram")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engram: Option<String>,
    /// Schema version.
    pub version: i64,
}

// ------------------------------------------------------------------ entity hook

/// A namespace's entities as `remember` sees them: rows and the canonical
/// entity of each, following current merges (`kleos_entity_merge`).
#[derive(Default)]
struct EntityView {
    /// eid → (name, normalised name)
    rows: BTreeMap<String, (String, String)>,
    merged_into: BTreeMap<String, String>,
}

impl EntityView {
    fn canonical(&self, eid: &str) -> String {
        let mut cur = eid.to_string();
        for _ in 0..10_000 {
            match self.merged_into.get(&cur) {
                Some(n) => cur = n.clone(),
                None => break,
            }
        }
        cur
    }
    /// Every entity row of the clusters that contain a row named `name`.
    fn cluster_of_name(&self, name: &str) -> Vec<String> {
        let norm = norm_name(name);
        let canon: BTreeSet<String> = self
            .rows
            .iter()
            .filter(|(_, r)| r.1 == norm)
            .map(|(e, _)| self.canonical(e))
            .collect();
        self.rows
            .keys()
            .filter(|e| canon.contains(&self.canonical(e)))
            .cloned()
            .collect()
    }
}

/// The entity writes of one `remember`, decided while planning and written
/// in its transaction (so they are covered by the namespace version too).
#[derive(Default)]
struct EntityPlan {
    /// New `kleos_mem_entity` rows `[ns, eid, name, norm, kind, created]`.
    rows: Vec<DataValue>,
    /// New `kleos_entity_merge` rows (a new surface form of a known entity).
    merges: Vec<DataValue>,
    /// New `kleos_entity_link` review rows.
    links: Vec<DataValue>,
    out: Vec<EntityLink>,
    /// Per entry of `out`: the canonical entities that name may also be.
    maybe_of: Vec<BTreeSet<String>>,
    /// Canonical entities of the statement.
    sure: BTreeSet<String>,
    /// Canonical entities its names may also be (auto or tentative
    /// candidates that were not taken).
    maybe: BTreeSet<String>,
}

/// Entity resolution over the memory schema's own entity relation, scoped
/// to one namespace.
fn mem_entity_config(ns: &str, decider: Option<&str>) -> EntityConfig {
    let mut cfg = EntityConfig::new(MEM_ENTITY);
    cfg.id = "eid".into();
    cfg.type_col = Some("kind".into());
    cfg.scope = Some(("ns".into(), s(ns)));
    cfg.decider = decider.map(|d| d.to_string());
    cfg
}

// ------------------------------------------------------------------ engine

/// A current memory found as a `remember` candidate.
struct Cand {
    id: String,
    text: String,
    from: i64,
    /// Canonical entities it mentions.
    ents: BTreeSet<String>,
    /// Archived, or expired at the new statement's time: never a duplicate
    /// target (the restatement would vanish into a hidden memory).
    hidden: bool,
    /// Cosine similarity to the new claim (both embedded).
    sim: Option<f64>,
}

fn mem_decision(c: &Cand, d: &Decision, action: &str) -> MemDecision {
    MemDecision {
        candidate: c.id.clone(),
        candidate_text: c.text.clone(),
        template: d.template.clone(),
        label: d.label.clone(),
        p: d.p,
        band: d.band.name().to_string(),
        decided_by: d.decided_by(),
        reason: d.reason.clone(),
        action: action.to_string(),
        claim: None,
        similarity: c.sim,
    }
}

/// A decision made without a model (exact text match, value rule).
fn rule_decision(template: Template, model: &str, via: &str, reason: String) -> Decision {
    Decision {
        template: template.name().to_string(),
        template_version: 1,
        label: "yes".to_string(),
        p: if via == "exact" { 1.0 } else { rules::RULE_P },
        probs: vec![],
        band: Band::Auto,
        confident: true,
        model: model.to_string(),
        via: via.to_string(),
        reason: Some(reason),
    }
}

fn is_yes(d: &Decision, band: Band) -> bool {
    d.band == band && d.label == "yes"
}

/// Validity interval of each id at `t` from its full version history:
/// `(from, to)` of the last assertion at or before `t`.
fn intervals(rows: &[(String, i64, bool)], t: i64) -> BTreeMap<String, (i64, Option<i64>)> {
    let mut by: BTreeMap<&str, Vec<(i64, bool)>> = BTreeMap::new();
    for (id, ts, a) in rows {
        by.entry(id.as_str()).or_default().push((*ts, *a));
    }
    let mut out = BTreeMap::new();
    for (id, mut v) in by {
        v.sort();
        let mut from = None;
        let mut to = None;
        for (ts, a) in v {
            if ts <= t {
                if a {
                    if from.is_none() {
                        from = Some(ts);
                    }
                } else {
                    from = None;
                }
            } else if from.is_some() && !a && to.is_none() {
                to = Some(ts);
            }
        }
        if let Some(f) = from {
            out.insert(id.to_string(), (f, to));
        }
    }
    out
}

impl<'s, S: Storage<'s>> Db<S> {
    fn mem_q(&'s self, script: &str, p: Vec<(&str, DataValue)>) -> Result<NamedRows> {
        self.run_script(script, params(p), ScriptMutability::Mutable)
    }

    /// The namespace's `MEM_NS` version (0 before its first write).
    fn ns_head(&'s self, ns: &str) -> Result<i64> {
        Ok(self
            .mem_q(&format!("?[seq] := *{MEM_NS}{{ns: $ns, seq}}"), vec![("ns", s(ns))])?
            .rows
            .first()
            .and_then(|r| r[0].get_int())
            .unwrap_or(0))
    }

    fn mem_config(&'s self) -> Result<Option<MemConfig>> {
        let rows = match self.run_script(
            &format!("?[key, value] := *{MEM_CONFIG}{{key, value}}"),
            Default::default(),
            ScriptMutability::Immutable,
        ) {
            Ok(r) => r,
            Err(e) => {
                let msg = format!("{e:?}");
                if msg.contains(MEM_CONFIG) {
                    return Ok(None);
                }
                return Err(e);
            }
        };
        let mut cfg = MemConfig::default();
        for r in rows.rows {
            let v = cell_str(&r[1]);
            match r[0].get_str() {
                Some("embedder") => cfg.embedder = v,
                Some("decider") => cfg.decider = v,
                Some("dim") => cfg.dim = v.and_then(|x| x.parse().ok()),
                #[cfg(feature = "engram")]
                Some("engram") => cfg.engram = v,
                Some("version") => cfg.version = v.and_then(|x| x.parse().ok()).unwrap_or(1),
                _ => {}
            }
        }
        if cfg.version < 2 {
            // v1 stores predate the namespace version row
            self.mem_q(&format!(":create {MEM_NS} {{ns: String => seq: Int}}"), vec![])?;
            self.set_mem_config("version", Some("2".to_string()))?;
            cfg.version = 2;
        }
        if cfg.version < 3 {
            // v2 stores predate scoring (E4) and entity merges (E3)
            self.create_v3_relations()?;
            self.set_mem_config("version", Some("3".to_string()))?;
            cfg.version = 3;
        }
        if cfg.version < 4 {
            // v3 stores predate speaker roles
            self.create_v4_relations()?;
            self.set_mem_config("version", Some("4".to_string()))?;
            cfg.version = 4;
        }
        Ok(Some(cfg))
    }

    fn create_v4_relations(&'s self) -> Result<()> {
        self.mem_q(&format!("{{:create {MEM_ROLE} {{ns: String, id: String => role: String}}}}"), vec![])?;
        Ok(())
    }

    fn create_v3_relations(&'s self) -> Result<()> {
        self.mem_q(&format!("{{:create {MEM_ACCESS} {{ns: String, id: String => n: Int, last: Int}}}}"), vec![])?;
        self.mem_q(
            &format!("{{:create {MEM_LIFE} {{ns: String, id: String => expires: Int?, archived: Int?}}}}"),
            vec![],
        )?;
        self.ensure_entity_relations()
    }

    fn set_mem_config(&'s self, key: &str, value: Option<String>) -> Result<()> {
        self.mem_q(
            &format!("?[key, value] <- [[$k, $v]] :put {MEM_CONFIG} {{key => value}}"),
            vec![("k", s(key)), ("v", opt_s(&value))],
        )?;
        Ok(())
    }

    /// Create the memory schema if it does not exist yet, then apply `opts`
    /// (attach an embedder, set the default decider). Idempotent. Returns
    /// the configuration.
    pub fn memory_init(&'s self, opts: InitOpts) -> Result<MemConfig> {
        let mut cfg = match self.mem_config()? {
            Some(c) => c,
            None => {
                for stmt in [
                    format!("{{:create {MEM} {{ns: String, id: String, vt: Validity => text: String, kind: String, source: String?, created: Int, importance: Float?}}}}"),
                    format!("{{:create {MEM_ENTITY} {{ns: String, eid: String => name: String, norm: String, kind: String?, created: Int}}}}"),
                    format!("{{:create {MEM_MENTION} {{ns: String, id: String, eid: String}}}}"),
                    format!("{{:create {MEM_LINK} {{ns: String, src: String, rel: String, dst: String => created: Int, p: Float?, decided_by: String?}}}}"),
                    format!("{{:create {MEM_SOURCE} {{ns: String, id: String, at: Int, stmt: String => source: String?, text: String}}}}"),
                    format!("{{:create {MEM_DECISION} {{ns: String, new: String, cand: String, template: String => label: String, p: Float, band: String, decided_by: String, reason: String?, status: String, at: Int}}}}"),
                    format!("{{:create {MEM_NS} {{ns: String => seq: Int}}}}"),
                ] {
                    self.mem_q(&stmt, vec![])?;
                }
                self.create_v3_relations()?;
                self.create_v4_relations()?;
                self.mem_q(
                    &format!("::fts create {MEM}:fts {{extractor: text, tokenizer: Simple, filters: [Lowercase, Stemmer('english'), Stopwords('en')]}}"),
                    vec![],
                )?;
                self.mem_q(&format!("::index create {MEM_ENTITY}:by_norm {{ns, norm}}"), vec![])?;
                self.mem_q(&format!("::index create {MEM_MENTION}:by_entity {{ns, eid}}"), vec![])?;
                // the config relation last: its existence marks a complete schema
                self.mem_q(&format!(":create {MEM_CONFIG} {{key: String => value: String?}}"), vec![])?;
                self.set_mem_config("version", Some(SCHEMA_VERSION.to_string()))?;
                MemConfig {
                    version: SCHEMA_VERSION,
                    ..Default::default()
                }
            }
        };
        if let Some(d) = opts.decider {
            if cfg.decider.as_deref() != Some(d.as_str()) {
                self.set_mem_config("decider", Some(d.clone()))?;
                cfg.decider = Some(d);
            }
        }
        if let Some(e) = opts.embedder {
            self.attach_embedder(&mut cfg, &e)?;
        }
        #[cfg(feature = "engram")]
        if let Some(m) = opts.engram {
            crate::memory_engram::attach(self, &mut cfg, &m)?;
        }
        Ok(cfg)
    }

    /// Set the store's embedder: create the vector relation and its HNSW
    /// index (dimension from the embedder), then embed every memory that has
    /// no vector yet.
    fn attach_embedder(&'s self, cfg: &mut MemConfig, name: &str) -> Result<()> {
        let status = embed::embedders().into_iter().find(|e| e.name == name && e.enabled);
        let dim = match status {
            Some(st) => st.dim,
            None => bail!(
                "embedder '{name}' is not enabled; enable it with `::embedder enable {name}` before `::memory init {{embedder: '{name}'}}`"
            ),
        };
        match (&cfg.embedder, cfg.dim) {
            (Some(old), Some(d)) if old == name && d == dim => {}
            (_, Some(d)) if d != dim => bail!(
                "the memory store's vectors have dimension {d} ({}); embedder '{name}' has {dim}. A store keeps one vector dimension",
                cfg.embedder.as_deref().unwrap_or("?")
            ),
            (_, Some(_)) => {
                self.set_mem_config("embedder", Some(name.to_string()))?;
            }
            (_, None) => {
                self.mem_q(
                    &format!("{{:create {MEM_VEC} {{ns: String, id: String => current: Bool, vec: <F32; {dim}>}}}}"),
                    vec![],
                )?;
                self.mem_q(
                    &format!("::hnsw create {MEM_VEC}:hnsw {{dim: {dim}, m: 32, dtype: F32, fields: [vec], distance: Cosine, ef_construction: 100}}"),
                    vec![],
                )?;
                self.set_mem_config("dim", Some(dim.to_string()))?;
                self.set_mem_config("embedder", Some(name.to_string()))?;
            }
        }
        cfg.embedder = Some(name.to_string());
        cfg.dim = Some(dim);
        self.backfill_memory_vectors(name)?;
        Ok(())
    }

    fn backfill_memory_vectors(&'s self, embedder: &str) -> Result<usize> {
        let missing = self.mem_q(
            &format!("?[ns, id, text] := *{MEM}{{ns, id, text}}, not *{MEM_VEC}{{ns, id}}"),
            vec![],
        )?;
        let current: BTreeSet<(String, String)> = self
            .mem_q(&format!("?[ns, id] := *{MEM}{{ns, id @ 'NOW'}}"), vec![])?
            .rows
            .into_iter()
            .filter_map(|r| Some((cell_str(&r[0])?, cell_str(&r[1])?)))
            .collect();
        let rows: Vec<(String, String, String)> = missing
            .rows
            .into_iter()
            .filter_map(|r| Some((cell_str(&r[0])?, cell_str(&r[1])?, cell_str(&r[2])?)))
            .collect();
        let mut n = 0;
        for chunk in rows.chunks(64) {
            let texts: Vec<&str> = chunk.iter().map(|r| r.2.as_str()).collect();
            let vecs = embed::embed_texts(embedder, &texts)?;
            let data = chunk
                .iter()
                .zip(vecs)
                .map(|((ns, id, _), v)| {
                    DataValue::List(vec![
                        s(ns),
                        s(id),
                        DataValue::Bool(current.contains(&(ns.clone(), id.clone()))),
                        DataValue::List(v.iter().map(|x| DataValue::from(*x as f64)).collect()),
                    ])
                })
                .collect();
            self.mem_q(
                &format!("?[ns, id, current, vec] <- $rows :put {MEM_VEC} {{ns, id => current, vec}}"),
                vec![("rows", DataValue::List(data))],
            )?;
            n += chunk.len();
        }
        Ok(n)
    }

    fn mem_ready(&'s self, embedder_hint: Option<&str>) -> Result<MemConfig> {
        match self.mem_config()? {
            Some(c) => Ok(c),
            None => {
                // lazy init: the given embedder, else the first enabled
                // document embedder (gemma's `_query` twin is for queries)
                let embedder = embedder_hint.map(|s| s.to_string()).or_else(|| {
                    embed::embedders()
                        .into_iter()
                        .filter(|e| e.enabled && !e.name.ends_with("_query"))
                        .map(|e| e.name)
                        .min()
                });
                self.memory_init(InitOpts {
                    embedder,
                    ..Default::default()
                })
            }
        }
    }

    /// Embed with `name` if it is enabled; `None` while it is off.
    fn embed_if_on(&self, name: Option<&str>, text: &str) -> Result<Option<Vec<f32>>> {
        match name {
            Some(n) if embed::is_embedder_enabled(n) => {
                Ok(embed::embed_texts(n, &[text])?.pop().map(|v| v.to_vec()))
            }
            _ => Ok(None),
        }
    }

    /// Embed every text with `name` in one batch if it is enabled.
    fn embed_all_if_on(&self, name: Option<&str>, texts: &[String]) -> Result<Vec<Option<Vec<f32>>>> {
        match name {
            Some(n) if embed::is_embedder_enabled(n) => {
                let refs: Vec<&str> = texts.iter().map(|t| t.as_str()).collect();
                Ok(embed::embed_texts(n, &refs)?.into_iter().map(|v| Some(v.to_vec())).collect())
            }
            _ => Ok(vec![None; texts.len()]),
        }
    }

    /// The decider `remember` uses: the call's, the store's, else the first
    /// enabled one; `None` when that one is off (rules only).
    fn pick_decider(&self, call: Option<&str>, cfg: &MemConfig) -> Option<String> {
        let enabled: Vec<String> = decide::deciders()
            .into_iter()
            .filter(|d| d.enabled)
            .map(|d| d.name)
            .collect();
        match call.or(cfg.decider.as_deref()) {
            Some(n) => enabled.iter().find(|e| *e == n).cloned(),
            None => enabled.into_iter().min(),
        }
    }

    /// Hybrid search over one namespace: filtered HNSW on the memory vectors
    /// plus FTS on the text, fused with `WeightedFusion`, then joined with
    /// the memories valid at `as_of` (default now). The index legs fetch for
    /// `k` hits; the `window` best are returned. Returns `(id, score, text,
    /// kind, source, created, importance)` rows by score.
    #[allow(clippy::too_many_arguments)]
    fn mem_search(
        &'s self,
        cfg: &MemConfig,
        ns: &str,
        text: &str,
        qvec: Option<Vec<f32>>,
        k: usize,
        window: usize,
        as_of: Option<i64>,
        kind: Option<&str>,
        entity: Option<&[String]>,
        vector_weight: f64,
        hide_archived: bool,
        hide_expired_at: Option<i64>,
    ) -> Result<Vec<Vec<DataValue>>> {
        let fq = fts_query(text);
        // indexes see superseded and retraction rows too: over-fetch
        let mut fetch = k * 4 + 16;
        if kind.is_some() || entity.is_some() {
            fetch *= 4;
        }
        let mut legs = vec![];
        let mut p = vec![("ns", s(ns))];
        if let (Some(v), Some(_)) = (qvec, cfg.dim) {
            let cur = if as_of.is_none() { " && current" } else { "" };
            legs.push(format!(
                "l[lid, id, score] := ~{MEM_VEC}:hnsw{{ns, id, current | query: vec($qv), k: {fetch}, ef: {ef}, bind_distance: d, filter: ns == $ns{cur}}}, lid = 'semantic', score = -d",
                ef = fetch.max(64)
            ));
            p.push((
                "qv",
                DataValue::List(v.iter().map(|x| DataValue::from(*x as f64)).collect()),
            ));
        }
        if !fq.is_empty() {
            legs.push(format!(
                "l[lid, id, score] := ~{MEM}:fts{{ns, id | query: $fq, k: {fetch}, bind_score: s, filter: ns == $ns}}, lid = 'text', score = s"
            ));
            p.push(("fq", s(&fq)));
        }
        if legs.is_empty() {
            return Ok(vec![]);
        }
        let has_vec = legs.len() == 2 || legs[0].contains("'semantic'");
        let has_text = legs.len() == 2 || !has_vec;
        let mut weights = vec![];
        let mut norms = vec![];
        if has_vec {
            let w = if has_text { vector_weight } else { 1.0 };
            weights.push(format!("['semantic', {w}]"));
            norms.push("['semantic', 'neg_cosine_distance']");
        }
        if has_text {
            let w = if has_vec { 1.0 - vector_weight } else { 1.0 };
            weights.push(format!("['text', {w}]"));
            norms.push("['text', 'bm25']");
        }
        let at = match as_of {
            Some(t) => {
                p.push(("as_of", DataValue::from(t)));
                "$as_of"
            }
            None => "'NOW'",
        };
        let mut filters = String::new();
        if let Some(kd) = kind {
            filters.push_str(", kind == $kind");
            p.push(("kind", s(kd)));
        }
        if let Some(eids) = entity {
            // every row of the entity's cluster (merged names included)
            filters.push_str(&format!(", *{MEM_MENTION}{{ns: $ns, id, eid}}, is_in(eid, $eids)"));
            p.push(("eids", str_list(eids)));
        }
        // the archive tier and expired memories are left out unless asked
        let mut hidden = String::new();
        if hide_archived {
            hidden.push_str("hidden[id] := life[id, _, a], !is_null(a)\n");
        }
        if let Some(t) = hide_expired_at {
            hidden.push_str("hidden[id] := life[id, e, _], !is_null(e), e <= $exp_at\n");
            p.push(("exp_at", DataValue::from(t)));
        }
        if !hidden.is_empty() {
            hidden.insert_str(
                0,
                &format!("life[id, e, a] := f[id, _], *{MEM_LIFE}{{ns: $ns, id, expires: e, archived: a}}\n"),
            );
            filters.push_str(", not hidden[id]");
        }
        let script = format!(
            "{legs}\n\
             f[id, score] <~ WeightedFusion(l[lid, id, score], weights: [{w}], norms: [{n}])\n\
             {hidden}\
             ?[id, score, text, kind, source, created, importance] := f[id, score], \
               *{MEM}{{ns: $ns, id, text, kind, source, created, importance @ {at}}}{filters}\n\
             :order -score\n:limit {window}",
            legs = legs.join("\n"),
            w = weights.join(", "),
            n = norms.join(", "),
        );
        Ok(self
            .run_script(&script, params(p), ScriptMutability::Immutable)?
            .rows)
    }

    /// Roles of the memories that have one (`remember {role}`).
    fn roles_of(&'s self, ns: &str, ids: &[String]) -> Result<BTreeMap<String, String>> {
        if ids.is_empty() {
            return Ok(BTreeMap::new());
        }
        Ok(self
            .mem_q(
                &format!("?[id, role] := id in $ids, *{MEM_ROLE}{{ns: $ns, id, role}}"),
                vec![("ns", s(ns)), ("ids", str_list(ids))],
            )?
            .rows
            .into_iter()
            .filter_map(|r| Some((cell_str(&r[0])?, cell_str(&r[1])?)))
            .collect())
    }

    /// Validity history rows `(id, ts, is_assert)` of `ids` in `ns`.
    fn mem_history(&'s self, ns: &str, ids: &[String]) -> Result<Vec<(String, i64, bool)>> {
        Ok(self
            .mem_q(
                &format!("?[id, vt] := *{MEM}{{ns: $ns, id, vt}}, id in $ids"),
                vec![("ns", s(ns)), ("ids", str_list(ids))],
            )?
            .rows
            .into_iter()
            .filter_map(|r| match &r[1] {
                DataValue::Validity(v) => Some((cell_str(&r[0])?, v.timestamp.0 .0, v.is_assert.0)),
                _ => None,
            })
            .collect())
    }

    /// The namespace's entity rows and current merges.
    fn entity_view(&'s self, ns: &str) -> Result<EntityView> {
        let mut v = EntityView::default();
        for r in self
            .mem_q(&format!("?[eid, name, norm] := *{MEM_ENTITY}{{ns: $ns, eid, name, norm}}"), vec![("ns", s(ns))])?
            .rows
        {
            if let (Some(e), Some(n), Some(m)) = (cell_str(&r[0]), cell_str(&r[1]), cell_str(&r[2])) {
                v.rows.insert(e, (n, m));
            }
        }
        if !v.rows.is_empty() {
            let eids: Vec<String> = v.rows.keys().cloned().collect();
            for r in self
                .mem_q(
                    &format!("?[f, i] := f in $eids, *{ENTITY_MERGES}{{relation: $r, from: f, into: i @ 'NOW'}}"),
                    vec![("eids", str_list(&eids)), ("r", s(MEM_ENTITY))],
                )?
                .rows
            {
                if let (Some(f), Some(i)) = (cell_str(&r[0]), cell_str(&r[1])) {
                    v.merged_into.insert(f, i);
                }
            }
        }
        Ok(v)
    }

    /// Resolve the names a statement mentions against the namespace's
    /// entities (E3, `entity.rs`; no NER: `names` are the caller's) and
    /// decide the writes: an Auto match links the existing entity (reusing
    /// the cluster's row of the same name, else a new row merged into it); a
    /// Tentative one makes a new entity with review links; none, a new
    /// entity. Nothing is written here.
    fn plan_entities(
        &'s self,
        ns: &str,
        names: &[String],
        decider: Option<&str>,
        view: &EntityView,
    ) -> Result<EntityPlan> {
        let mut wanted: Vec<(String, String)> = vec![];
        for n in names {
            let k = norm_name(n);
            if !k.is_empty() && !wanted.iter().any(|w| w.0 == k) {
                wanted.push((k, n.trim().to_string()));
            }
        }
        let mut plan = EntityPlan::default();
        if wanted.is_empty() {
            return Ok(plan);
        }
        // no context: an exact, unique name in the namespace is the same
        // entity by rule; the model only sees other near names
        let cfg = mem_entity_config(ns, decider);
        let mentions: Vec<Mention> = wanted.iter().map(|(_, n)| Mention::new(n.as_str())).collect();
        let res = self.resolve_entities(&cfg, &mentions, ResolveOptions { commit: false })?;
        let now = now_micros();
        let canon_of = |c: &DataValue| c.get_str().map(|x| view.canonical(x));
        for ((norm, name), r) in wanted.iter().zip(res) {
            let review: Vec<&crate::entity::Candidate> = r
                .candidates
                .iter()
                .filter(|c| matches!(c.band, Some(Band::Auto) | Some(Band::Tentative)))
                .collect();
            let (eid, canonical, created) = match (r.outcome, r.entity.as_ref().and_then(canon_of)) {
                (Outcome::Matched, Some(c)) => {
                    match view.rows.iter().find(|(e, row)| row.1 == *norm && view.canonical(e) == c) {
                        Some((e, _)) => (e.clone(), c, false),
                        None => {
                            // a new name of a known entity: its own row,
                            // merged (non-destructively, splittable)
                            let e = new_id("e");
                            plan.merges.push(DataValue::List(vec![
                                s(MEM_ENTITY),
                                s(&e),
                                DataValue::Validity(Validity::from((now, true))),
                                s(&c),
                                DataValue::from(r.p),
                                s(&r.decided_by),
                                s(r.reason.as_deref().unwrap_or("")),
                            ]));
                            (e, c, true)
                        }
                    }
                }
                _ => {
                    let e = new_id("e");
                    (e.clone(), e, true)
                }
            };
            if created {
                plan.rows.push(DataValue::List(vec![
                    s(ns),
                    s(&eid),
                    s(name),
                    s(norm),
                    DataValue::Null,
                    DataValue::from(now),
                ]));
                // every other likely match is linked for review, never
                // merged (no transitive merges); a reused row keeps the
                // links (and rejections) it already has
                for c in &review {
                    if canon_of(&c.entity).as_deref() == Some(canonical.as_str()) {
                        continue;
                    }
                    plan.links.push(DataValue::List(vec![
                        s(MEM_ENTITY),
                        s(&eid),
                        c.entity.clone(),
                        DataValue::from(c.p.unwrap_or(0.0)),
                        s(c.band.unwrap_or(Band::None).name()),
                        s(c.decided_by.as_deref().unwrap_or("")),
                        s("pending"),
                        DataValue::from(now),
                    ]));
                }
            }
            let mut maybe = BTreeSet::new();
            for c in &review {
                maybe.extend(canon_of(&c.entity));
            }
            plan.maybe.extend(maybe.iter().cloned());
            plan.maybe_of.push(maybe);
            plan.sure.insert(canonical.clone());
            plan.out.push(EntityLink {
                name: name.clone(),
                eid,
                canonical,
                outcome: r.outcome.name().to_string(),
                created,
                decided_by: r.decided_by.clone(),
                p: r.p,
            });
        }
        Ok(plan)
    }

    /// `ids` plus the claims of each that is a statement id (`<id>.<n>`):
    /// a statement split into claims is forgotten or set as a whole.
    fn with_claims(&'s self, ns: &str, mut ids: Vec<String>) -> Result<Vec<String>> {
        if ids.is_empty() {
            return Ok(ids);
        }
        let prefixes: Vec<String> = ids.iter().filter(|i| !i.contains('.')).map(|i| format!("{i}.")).collect();
        if prefixes.is_empty() {
            return Ok(ids);
        }
        // claim ids sort right after their statement id: one key range each
        for r in self
            .mem_q(
                &format!(
                    "?[id] := p in $ps, *{MEM}{{ns: $ns, id}}, id > p, id < concat(p, '~'), starts_with(id, p)"
                ),
                vec![("ns", s(ns)), ("ps", str_list(&prefixes))],
            )?
            .rows
        {
            ids.extend(cell_str(&r[0]));
        }
        ids.sort();
        ids.dedup();
        Ok(ids)
    }

    /// `(expires, archived)` of the ids that have a `kleos_mem_life` row.
    fn life_of(&'s self, ns: &str, ids: &[String]) -> Result<BTreeMap<String, (Option<i64>, Option<i64>)>> {
        if ids.is_empty() {
            return Ok(BTreeMap::new());
        }
        Ok(self
            .mem_q(
                &format!("?[id, expires, archived] := id in $ids, *{MEM_LIFE}{{ns: $ns, id, expires, archived}}"),
                vec![("ns", s(ns)), ("ids", str_list(ids))],
            )?
            .rows
            .into_iter()
            .filter_map(|r| Some((cell_str(&r[0])?, (r[1].get_int(), r[2].get_int()))))
            .collect())
    }

    /// The read-only half of `remember`: per claim, the candidates, the
    /// decisions (batched across all claims: one decider call per template)
    /// and the actions they imply. Nothing is written.
    #[allow(clippy::too_many_arguments)]
    fn plan_remember(
        &'s self,
        cfg: &MemConfig,
        ns: &str,
        claims: &[(String, String, Option<Vec<f32>>)],
        at: i64,
        opts: &RememberOpts,
        auto: bool,
        floor: f64,
    ) -> Result<Plan> {
        let k = opts.k.unwrap_or(DEFAULT_K).max(1);
        let multi = claims.len() > 1;
        let decider = self.pick_decider(opts.decider.as_deref(), cfg);

        // entities: resolve the statement's names once; a name is linked to
        // the claims that mention it, or to every claim when none does
        // (the subject, e.g. `user`, is rarely written out)
        let view = if opts.entities.is_empty() {
            EntityView::default()
        } else {
            self.entity_view(ns)?
        };
        let ents = self.plan_entities(ns, &opts.entities, decider.as_deref(), &view)?;
        let mut claim_ents: Vec<Vec<usize>> = vec![vec![]; claims.len()];
        for (ei, e) in ents.out.iter().enumerate() {
            let hit: Vec<usize> = if multi {
                (0..claims.len()).filter(|&c| mentions_name(&claims[c].1, &e.name)).collect()
            } else {
                vec![]
            };
            let to: Vec<usize> = if hit.is_empty() { (0..claims.len()).collect() } else { hit };
            for c in to {
                claim_ents[c].push(ei);
            }
        }

        // candidates per claim; only declarative claims (and candidates)
        // take part in decisions in `auto` mode
        // assistant and other speakers' memories are never decided
        let user_said = opts.role.as_deref().map_or(true, |r| r == "user");
        let decided: Vec<bool> = claims
            .iter()
            .map(|c| user_said && (!auto || memory_claims::is_declarative(&c.1)))
            .collect();
        let mut rows_per: Vec<Vec<Vec<DataValue>>> = Vec::with_capacity(claims.len());
        for (ci, (_, text, vec)) in claims.iter().enumerate() {
            if !decided[ci] {
                rows_per.push(vec![]);
                continue;
            }
            let window = if auto { 2 * k } else { k };
            let rows = self.mem_search(
                cfg,
                ns,
                text,
                vec.clone(),
                k,
                window,
                None,
                None,
                None,
                DEFAULT_VECTOR_WEIGHT,
                false,
                None,
            )?;
            rows_per.push(rows);
        }
        // candidates: never another speaker's memory; in `auto` mode only
        // declarative ones, the first k of the 2k fetched
        let spoken = self.roles_of(
            ns,
            &rows_per.iter().flatten().filter_map(|r| cell_str(&r[0])).collect::<Vec<_>>(),
        )?;
        let rows_per: Vec<Vec<Vec<DataValue>>> = rows_per
            .into_iter()
            .map(|rows| {
                rows.into_iter()
                    .filter(|r| cell_str(&r[0]).map_or(true, |id| spoken.get(&id).map_or(true, |x| x == "user")))
                    .filter(|r| !auto || r[2].get_str().is_some_and(memory_claims::is_declarative))
                    .take(k)
                    .collect()
            })
            .collect();
        let ids: Vec<String> = rows_per
            .iter()
            .flatten()
            .filter_map(|r| cell_str(&r[0]))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let iv = intervals(&self.mem_history(ns, &ids)?, i64::MAX);
        let life = self.life_of(ns, &ids)?;
        let mut cand_ents: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        if !ents.sure.is_empty() && !ids.is_empty() {
            for r in self
                .mem_q(
                    &format!("?[id, eid] := id in $ids, *{MEM_MENTION}{{ns: $ns, id, eid}}"),
                    vec![("ns", s(ns)), ("ids", str_list(&ids))],
                )?
                .rows
            {
                if let (Some(id), Some(e)) = (cell_str(&r[0]), cell_str(&r[1])) {
                    cand_ents.entry(id).or_default().insert(view.canonical(&e));
                }
            }
        }
        // the candidates' vectors, for the similarity floor and the audit
        let mut cand_vecs: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        if cfg.dim.is_some() && !ids.is_empty() && claims.iter().any(|c| c.2.is_some()) {
            for r in self
                .mem_q(
                    &format!("?[id, vec] := id in $ids, *{MEM_VEC}{{ns: $ns, id, vec}}"),
                    vec![("ns", s(ns)), ("ids", str_list(&ids))],
                )?
                .rows
            {
                if let (Some(id), Some(v)) = (cell_str(&r[0]), vec_of(&r[1])) {
                    cand_vecs.insert(id, v);
                }
            }
        }

        let mut plans: Vec<ClaimPlan> = vec![];
        let mut sup_todo: Vec<(usize, usize)> = vec![];
        let mut dup_todo: Vec<(usize, usize)> = vec![];
        let mut capped_all: Vec<Vec<bool>> = vec![];
        let mut gated_ids: Vec<String> = vec![];
        for (ci, ((id, text, vec), rows)) in claims.iter().zip(rows_per).enumerate() {
            let cands: Vec<Cand> = rows
                .iter()
                .filter_map(|r| {
                    let id = cell_str(&r[0])?;
                    let hidden = life
                        .get(&id)
                        .is_some_and(|(e, a)| a.is_some() || e.is_some_and(|e| e <= at));
                    let sim = match (vec, cand_vecs.get(&id)) {
                        (Some(a), Some(b)) => Some(cosine(a, b)),
                        _ => None,
                    };
                    Some(Cand {
                        from: iv.get(&id).map(|x| x.0).unwrap_or(0),
                        ents: cand_ents.get(&id).cloned().unwrap_or_default(),
                        hidden,
                        sim,
                        id,
                        text: cell_str(&r[2])?,
                    })
                })
                .collect();
            // same-subject gate: when both statements name entities, a
            // supersede needs a shared one; sharing only a possible match
            // (review band) caps it at pending
            let sure: BTreeSet<String> = claim_ents[ci].iter().map(|&e| ents.out[e].canonical.clone()).collect();
            let maybe: BTreeSet<String> = claim_ents[ci].iter().flat_map(|&e| ents.maybe_of[e].iter().cloned()).collect();
            let mut gated = vec![false; cands.len()];
            let mut capped = vec![false; cands.len()];
            for (i, c) in cands.iter().enumerate() {
                if sure.is_empty() || c.ents.is_empty() || !c.ents.is_disjoint(&sure) {
                    continue;
                }
                if c.ents.is_disjoint(&maybe) {
                    gated[i] = true;
                    gated_ids.push(c.id.clone());
                } else {
                    capped[i] = true;
                }
            }
            let key = norm_text(text);
            let mut dup: Vec<Option<Decision>> = vec![None; cands.len()];
            for (i, c) in cands.iter().enumerate() {
                if norm_text(&c.text) == key {
                    dup[i] = Some(rule_decision(
                        Template::Duplicate,
                        "kleos",
                        "exact",
                        "same words after normalisation".to_string(),
                    ));
                } else if decider.is_none() || !multi || c.sim.map_or(true, |x| x >= floor) {
                    // the decider only sees a claim's candidates above the
                    // floor (a single statement is decided as before)
                    dup_todo.push((ci, i));
                    if !gated[i] {
                        sup_todo.push((ci, i));
                    }
                }
            }
            capped_all.push(capped);
            plans.push(ClaimPlan {
                id: id.clone(),
                text: text.clone(),
                vec: vec.clone(),
                decided: decided[ci],
                sup: vec![None; cands.len()],
                dup,
                cands,
                supersede: vec![],
                duplicate: None,
                audit: vec![],
                ents: claim_ents[ci].clone(),
            });
        }

        // decisions: one batched call per template across every claim
        match &decider {
            Some(name) => {
                let pairs = |which: &[(usize, usize)]| -> Vec<(String, String)> {
                    which
                        .iter()
                        .map(|&(ci, i)| (plans[ci].cands[i].text.clone(), plans[ci].text.clone()))
                        .collect()
                };
                let sp = pairs(&sup_todo);
                let dp = pairs(&dup_todo);
                if !sp.is_empty() {
                    let refs: Vec<(&str, &str)> = sp.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
                    for (&(ci, i), a) in sup_todo.iter().zip(decide::decide(name, Template::Supersedes, &refs)?) {
                        plans[ci].sup[i] = Some(a);
                    }
                }
                if !dp.is_empty() {
                    let refs: Vec<(&str, &str)> = dp.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
                    for (&(ci, i), b) in dup_todo.iter().zip(decide::decide(name, Template::Duplicate, &refs)?) {
                        plans[ci].dup[i] = Some(b);
                    }
                }
            }
            None => {
                // model-free: the value pre-check alone (decide::rules)
                for &(ci, i) in &sup_todo {
                    if let rules::ValueRule::Update(r) = rules::value_rule(&plans[ci].cands[i].text, &plans[ci].text) {
                        plans[ci].sup[i] = Some(rule_decision(
                            Template::Supersedes,
                            "rules",
                            "rule",
                            format!("value rule: {r}"),
                        ));
                    }
                }
            }
        }

        // actions per claim
        for (cp, capped) in plans.iter_mut().zip(&capped_all) {
            let mut supersede = vec![];
            let mut duplicate: Option<(usize, f64)> = None;
            for (i, c) in cp.cands.iter().enumerate() {
                let s_auto = cp.sup[i].as_ref().is_some_and(|d| is_yes(d, Band::Auto)) && !capped[i];
                let d_auto = cp.dup[i].as_ref().is_some_and(|d| is_yes(d, Band::Auto)) && !c.hidden;
                if d_auto && !s_auto {
                    let p = cp.dup[i].as_ref().unwrap().p;
                    if duplicate.map_or(true, |(_, q)| p > q) {
                        duplicate = Some((i, p));
                    }
                } else if s_auto && !d_auto && c.from <= at {
                    supersede.push(i);
                }
            }
            cp.supersede = supersede;
            cp.duplicate = duplicate;
        }
        // a memory one claim repeats is not ended by another claim of the
        // same statement (that supersede is left pending)
        let kept: BTreeSet<String> = plans
            .iter()
            .filter_map(|cp| cp.duplicate.map(|(i, _)| cp.cands[i].id.clone()))
            .collect();
        for cp in plans.iter_mut() {
            let cands = &cp.cands;
            cp.supersede.retain(|&i| !kept.contains(&cands[i].id));
        }

        let mut res = RememberResult {
            ns: ns.to_string(),
            decider: decider.clone(),
            embedded: claims.iter().any(|c| c.2.is_some()),
            gated: gated_ids.into_iter().unique().collect(),
            ..Default::default()
        };
        for cp in plans.iter_mut() {
            // review noise (claims of a split statement only): of each
            // template, only the claim's likeliest answer that would be left
            // pending is kept for review, and only at p >= CLAIM_REVIEW_P;
            // the rest are recorded as `none`. A single statement keeps
            // every pending answer, as before.
            let best = |template: &str| -> Option<usize> {
                let col = if template == "supersedes" { &cp.sup } else { &cp.dup };
                (0..cp.cands.len())
                    .filter(|&i| {
                        col[i].as_ref().is_some_and(|d| {
                            let min = if template == "supersedes" { CLAIM_REVIEW_P } else { CLAIM_DUP_REVIEW_P };
                            d.p >= min && ((d.label == "yes" && d.band != Band::None) || d.p >= REVIEW_P)
                        })
                    })
                    .max_by(|&a, &b| {
                        let pa = col[a].as_ref().unwrap().p;
                        let pb = col[b].as_ref().unwrap().p;
                        pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
                    })
            };
            let best_sup = best("supersedes");
            let best_dup = best("duplicate");
            let claim_id = multi.then(|| match cp.duplicate {
                Some((i, _)) => cp.cands[i].id.clone(),
                None => cp.id.clone(),
            });
            for i in 0..cp.cands.len() {
                for d in [&cp.sup[i], &cp.dup[i]].into_iter().flatten() {
                    let applied = (cp.duplicate.map(|x| x.0) == Some(i) && d.template == "duplicate")
                        || (cp.duplicate.is_none() && cp.supersede.contains(&i) && d.template == "supersedes");
                    let action = if applied {
                        if d.template == "duplicate" {
                            "duplicate"
                        } else {
                            "superseded"
                        }
                    } else if (d.label == "yes" && d.band != Band::None)
                        || (d.template == "supersedes" && d.p >= REVIEW_P)
                    {
                        // tentative, a likely supersede below the calibrated
                        // threshold, an auto answer that conflicts with the
                        // other template, a late arrival, or an auto
                        // supersede left undone because the statement is a
                        // duplicate
                        let keep = if d.template == "supersedes" { best_sup } else { best_dup };
                        if multi && keep != Some(i) {
                            "none"
                        } else {
                            "pending"
                        }
                    } else {
                        "none"
                    };
                    let mut md = mem_decision(&cp.cands[i], d, action);
                    md.claim = claim_id.clone();
                    if action != "none" {
                        cp.audit.push(md.clone());
                    }
                    if action == "pending" {
                        res.tentative.push(md.clone());
                    }
                    res.decisions.push(md);
                }
            }
        }

        Ok(Plan {
            at,
            claims: plans,
            ents,
            res,
        })
    }

    /// Store `text` as memories. A text of several sentences is split into
    /// claims (`split: 'auto'`, the default), one memory each, with ids
    /// `<statement id>.<n>`; one sentence is one memory with the statement
    /// id. For each declarative claim, finds the top-k similar current
    /// memories of the namespace, decides `supersedes` and `duplicate`
    /// (batched over all claims, one call per template), and acts on the
    /// band: `auto` supersedes (ends the old memory's validity) or skips a
    /// duplicate (recording the statement as another source of the existing
    /// memory); `tentative` inserts and records a pending decision; `none`
    /// inserts. Every claim of one call is written in one transaction. Never
    /// deletes. Without a decider it uses the exact-text and value rules
    /// only; without an embedder, FTS candidates only.
    pub fn remember(&'s self, text: &str, opts: RememberOpts) -> Result<RememberResult> {
        let text = text.trim();
        if text.is_empty() {
            bail!("`remember` needs a non-empty text");
        }
        if let Some(x) = opts.importance {
            if !(0.0..=1.0).contains(&x) {
                bail!("`remember` option importance must be in [0, 1], got {x}");
            }
        }
        let auto = match opts.split.as_deref() {
            None | Some("auto") => true,
            Some("none") => false,
            Some(x) => bail!("`remember` option split must be 'auto' or 'none', got '{x}'"),
        };
        if let Some(r) = opts.role.as_deref() {
            if !ROLES.contains(&r) {
                bail!("`remember` option role must be 'user', 'assistant' or 'other', got '{r}'");
            }
        }
        let floor = opts.min_similarity.unwrap_or(DEFAULT_MIN_SIMILARITY);
        if !(-1.0..=1.0).contains(&floor) {
            bail!("`remember` option min_similarity must be in [-1, 1], got {floor}");
        }
        let cfg = self.mem_ready(opts.embedder.as_deref())?;
        let ns = ns_of(&opts.ns);
        let stmt = new_id("m");
        let parts = if auto {
            memory_claims::split_claims(text)
        } else {
            vec![text.to_string()]
        };
        let multi = parts.len() > 1;
        let vecs = self.embed_all_if_on(cfg.embedder.as_deref(), &parts)?;
        let claims: Vec<(String, String, Option<Vec<f32>>)> = parts
            .into_iter()
            .zip(vecs)
            .enumerate()
            .map(|(i, (t, v))| (if multi { claim_id(&stmt, i) } else { stmt.clone() }, t, v))
            .collect();

        // concurrency: plan without holding anything, then write in one
        // transaction that asserts the namespace version planned against is
        // unchanged and bumps it (`MEM_NS`); a writer that lost the race, in
        // this process or another, plans again and sees the winner. `at`
        // defaults to the planning time, so the statement committed later is
        // also the later one in valid time.
        let gate = ns_gate(&ns);
        let mut attempt = 0;
        let mut res = loop {
            let locked = attempt >= OPTIMISTIC_ATTEMPTS;
            let held = if locked { Some(gate.commit()) } else { None };
            let local = gate.seq();
            let seen = self.ns_head(&ns)?;
            let at = opts.at.unwrap_or_else(now_micros);
            let plan = self.plan_remember(&cfg, &ns, &claims, at, &opts, auto, floor)?;
            let commit = match held {
                Some(c) => c,
                None => gate.commit(),
            };
            if !locked && gate.seq() != local {
                // another writer of this process committed meanwhile
                drop(commit);
                attempt += 1;
                continue;
            }
            let Plan {
                at,
                claims: plans,
                ents,
                mut res,
            } = plan;

            let now = now_micros();
            let kind = s(opts.kind.as_deref().unwrap_or(DEFAULT_KIND));
            let importance = opts.importance.map(DataValue::from).unwrap_or(DataValue::Null);
            let expires = opts.expires.or(opts.ttl.map(|t| at.saturating_add(t)));
            let mut mems = vec![];
            let mut lives = vec![];
            let mut roles = vec![];
            let mut vec_rows = vec![];
            let mut old_all: Vec<String> = vec![];
            let mut links = vec![];
            let mut sources = vec![];
            let mut audit_rows = vec![];
            let mut mentions: Vec<DataValue> = vec![];
            // (mem id, old it ended, text, entity names) of each added claim
            let mut jobs: Vec<(Vec<String>, Vec<String>, String, Vec<String>)> = vec![];
            let mut out: Vec<ClaimResult> = vec![];
            for cp in &plans {
                let mem_id = match cp.duplicate {
                    Some((i, _)) => cp.cands[i].id.clone(),
                    None => cp.id.clone(),
                };
                let mut cr = ClaimResult {
                    id: None,
                    text: cp.text.clone(),
                    action: "added".to_string(),
                    duplicate_of: None,
                    superseded: vec![],
                    decided: cp.decided,
                };
                if cp.duplicate.is_none() {
                    cr.id = Some(mem_id.clone());
                    mems.push(DataValue::List(vec![
                        s(&ns),
                        s(&mem_id),
                        DataValue::Validity(Validity::from((at, true))),
                        s(&cp.text),
                        kind.clone(),
                        opt_s(&opts.source),
                        DataValue::from(now),
                        importance.clone(),
                    ]));
                    if let Some(e) = expires {
                        lives.push(DataValue::List(vec![s(&ns), s(&mem_id), DataValue::from(e), DataValue::Null]));
                    }
                    if let Some(r) = &opts.role {
                        roles.push(DataValue::List(vec![s(&ns), s(&mem_id), s(r)]));
                    }
                    if let (Some(v), Some(_)) = (&cp.vec, cfg.dim) {
                        vec_rows.push(DataValue::List(vec![
                            s(&ns),
                            s(&mem_id),
                            DataValue::Bool(true),
                            DataValue::List(v.iter().map(|x| DataValue::from(*x as f64)).collect()),
                        ]));
                    }
                    let mut mine = vec![];
                    for &i in &cp.supersede {
                        let c = &cp.cands[i];
                        let d = cp.sup[i].as_ref().unwrap();
                        links.push(DataValue::List(vec![
                            s(&ns),
                            s(&mem_id),
                            s("supersedes"),
                            s(&c.id),
                            DataValue::from(now),
                            DataValue::from(d.p),
                            s(&d.decided_by()),
                        ]));
                        cr.superseded.push(c.id.clone());
                        if !old_all.contains(&c.id) {
                            old_all.push(c.id.clone());
                            mine.push(c.id.clone());
                        }
                    }
                    let names: Vec<String> = if multi {
                        cp.ents.iter().map(|&e| ents.out[e].name.clone()).collect()
                    } else {
                        opts.entities.clone()
                    };
                    jobs.push((vec![mem_id.clone()], mine, cp.text.clone(), names));
                } else {
                    cr.action = "duplicate".to_string();
                    cr.duplicate_of = Some(mem_id.clone());
                }
                // every statement is a source row of the memories it ended
                // up in, with the whole statement's text
                sources.push(DataValue::List(vec![
                    s(&ns),
                    s(&mem_id),
                    DataValue::from(at),
                    s(&stmt),
                    opt_s(&opts.source),
                    s(text),
                ]));
                for d in &cp.audit {
                    audit_rows.push(DataValue::List(vec![
                        s(&ns),
                        s(&mem_id),
                        s(&d.candidate),
                        s(&d.template),
                        s(&d.label),
                        DataValue::from(d.p),
                        s(&d.band),
                        s(&d.decided_by),
                        opt_s(&d.reason),
                        s(if d.action == "pending" { "pending" } else { "applied" }),
                        DataValue::from(now),
                    ]));
                }
                for &e in &cp.ents {
                    let m = DataValue::List(vec![s(&ns), s(&mem_id), s(&ents.out[e].eid)]);
                    if !mentions.contains(&m) {
                        mentions.push(m);
                    }
                }
                out.push(cr);
            }

            let mut script = String::new();
            let mut p: Vec<(&str, DataValue)> = vec![("ns", s(&ns))];
            if !mems.is_empty() {
                script.push_str(&format!(
                    "{{?[ns, id, vt, text, kind, source, created, importance] <- $mems \
                      :put {MEM} {{ns, id, vt => text, kind, source, created, importance}}}}"
                ));
                p.push(("mems", DataValue::List(mems)));
            }
            if !lives.is_empty() {
                script.push_str(&format!(
                    "{{?[ns, id, expires, archived] <- $lives :put {MEM_LIFE} {{ns, id => expires, archived}}}}"
                ));
                p.push(("lives", DataValue::List(lives)));
            }
            if !roles.is_empty() {
                script.push_str(&format!("{{?[ns, id, role] <- $roles :put {MEM_ROLE} {{ns, id => role}}}}"));
                p.push(("roles", DataValue::List(roles)));
            }
            if !vec_rows.is_empty() {
                script.push_str(&format!(
                    "{{?[ns, id, current, vec] <- $vecs :put {MEM_VEC} {{ns, id => current, vec}}}}"
                ));
                p.push(("vecs", DataValue::List(vec_rows)));
            }
            if !old_all.is_empty() {
                // E1 semantics: end the old memory's validity by appending a
                // retraction row (values copied); nothing is deleted
                script.push_str(&format!(
                    "{{?[ns, id, vt, text, kind, source, created, importance] := *{MEM}{{ns, id, text, kind, source, created, importance @ 'NOW'}}, \
                       ns = $ns, id in $old, vt = [$at, false] \
                      :put {MEM} {{ns, id, vt => text, kind, source, created, importance}}}}"
                ));
                if cfg.dim.is_some() {
                    script.push_str(&format!(
                        "{{?[ns, id, current] := *{MEM_VEC}{{ns, id}}, ns = $ns, id in $old, current = false \
                          :update {MEM_VEC} {{ns, id => current}}}}"
                    ));
                }
                script.push_str(&format!(
                    "{{?[ns, src, rel, dst, created, p, decided_by] <- $links \
                      :put {MEM_LINK} {{ns, src, rel, dst => created, p, decided_by}}}}"
                ));
                p.push(("old", str_list(&old_all)));
                p.push(("at", DataValue::from(at)));
                p.push(("links", DataValue::List(links)));
            }
            script.push_str(&format!(
                "{{?[ns, id, at, stmt, source, text] <- $sources \
                  :put {MEM_SOURCE} {{ns, id, at, stmt => source, text}}}}"
            ));
            p.push(("sources", DataValue::List(sources)));
            if !audit_rows.is_empty() {
                script.push_str(&format!(
                    "{{?[ns, new, cand, template, label, p, band, decided_by, reason, status, at] <- $audit \
                      :put {MEM_DECISION} {{ns, new, cand, template => label, p, band, decided_by, reason, status, at}}}}"
                ));
                p.push(("audit", DataValue::List(audit_rows)));
            }
            // entities: new rows, merges and review links, and the mentions
            if !ents.rows.is_empty() {
                script.push_str(&format!(
                    "{{?[ns, eid, name, norm, kind, created] <- $ents :put {MEM_ENTITY} {{ns, eid => name, norm, kind, created}}}}"
                ));
                p.push(("ents", DataValue::List(ents.rows.clone())));
            }
            if !ents.merges.is_empty() {
                script.push_str(&format!(
                    "{{?[relation, from, vt, into, p, decided_by, reason] <- $merges \
                      :put {ENTITY_MERGES} {{relation, from, vt => into, p, decided_by, reason}}}}"
                ));
                p.push(("merges", DataValue::List(ents.merges.clone())));
            }
            if !ents.links.is_empty() {
                script.push_str(&format!(
                    "{{?[relation, a, b, p, band, decided_by, status, at] <- $elinks \
                      :put {ENTITY_LINKS} {{relation, a, b => p, band, decided_by, status, at}}}}"
                ));
                p.push(("elinks", DataValue::List(ents.links.clone())));
            }
            if !mentions.is_empty() {
                script.push_str(&format!(
                    "{{?[ns, id, eid] <- $mentions :put {MEM_MENTION} {{ns, id, eid}}}}"
                ));
                p.push(("mentions", DataValue::List(mentions)));
                res.entities = ents.out.iter().map(|e| e.eid.clone()).collect();
                res.entity_links = ents.out.clone();
            }
            // Engram: one job per added claim, numbered with the versions
            // this commit produces (the bump below advances by the count)
            let job_list: Vec<Job<'_>> = jobs
                .iter()
                .map(|(ids, old, t, names)| Job {
                    op: if old.is_empty() { "write" } else { "supersede" },
                    mem_ids: ids,
                    old,
                    text: Some(t.as_str()),
                    at,
                    entities: names,
                })
                .collect();
            let mut bump = 1;
            if let Some((frag, ps, n)) = engram_hook::enqueue_jobs(&cfg, &job_list) {
                script.push_str(&frag);
                p.extend(ps);
                bump = n.max(1);
            }
            // the version this plan was made against must still hold
            script.insert_str(
                0,
                &format!("{{?[seq] := *{MEM_NS}{{ns: $ns, seq}}, seq != $seen :assert none}}"),
            );
            script.push_str(&bump_ns_head_by(bump));
            p.push(("seen", DataValue::from(seen)));
            match self.mem_q(&script, p) {
                Ok(_) => {
                    res.superseded = old_all;
                    let added = out.iter().any(|c| c.action == "added");
                    res.action = if added { "added" } else { "duplicate" }.to_string();
                    res.id = added.then(|| stmt.clone());
                    res.duplicate_of = if added { None } else { out[0].duplicate_of.clone() };
                    if multi {
                        res.claims = out;
                    }
                    break res;
                }
                Err(e) => {
                    drop(commit);
                    let moved = self.ns_head(&ns).map_or(false, |h| h != seen);
                    if !(moved || is_contention(&e)) {
                        return Err(e);
                    }
                    if attempt + 1 >= COMMIT_ATTEMPTS {
                        bail!("memory namespace '{ns}' kept changing under `remember` ({COMMIT_ATTEMPTS} attempts); try again");
                    }
                    // back off before planning again: 1, 2, 4 .. 32 ms
                    std::thread::sleep(std::time::Duration::from_millis(1 << attempt.min(5)));
                    attempt += 1;
                }
            }
        };
        res.ns = ns;
        Ok(res)
    }

    /// Hybrid recall (vector + FTS, `WeightedFusion`) over the memories valid
    /// now, or at `as_of`, leaving out the archive tier and expired memories
    /// unless asked. Filters: namespace, kind, entity (its whole cluster).
    /// Ranked by the blend of relevance with recency, importance and access
    /// count (docs/specs/memory-api.md, "Ranking"), or by relevance alone
    /// with `blend: false`. The returned memories count as accessed.
    pub fn recall(&'s self, query: &str, opts: RecallOpts) -> Result<Vec<RecallHit>> {
        let cfg = match self.mem_config()? {
            Some(c) => c,
            None => return Ok(vec![]),
        };
        let ns = ns_of(&opts.ns);
        let w = opts.vector_weight.unwrap_or(DEFAULT_VECTOR_WEIGHT);
        if !(0.0..=1.0).contains(&w) {
            bail!("`recall` option vector_weight must be in [0, 1], got {w}");
        }
        let blend = opts.blend.unwrap_or(true);
        let wr = opts.recency_weight.unwrap_or(DEFAULT_RECENCY_WEIGHT);
        let half_life = opts.half_life_days.unwrap_or(DEFAULT_HALF_LIFE_DAYS);
        let wi = opts.importance_weight.unwrap_or(DEFAULT_IMPORTANCE_WEIGHT);
        let wa = opts.access_weight.unwrap_or(DEFAULT_ACCESS_WEIGHT);
        let wq = opts.question_weight.unwrap_or(DEFAULT_QUESTION_WEIGHT);
        let mmr = opts.mmr_weight.unwrap_or(DEFAULT_MMR_WEIGHT);
        let we = opts.expand_weight.unwrap_or(DEFAULT_EXPAND_WEIGHT);
        let was = opts.assistant_weight.unwrap_or(DEFAULT_ASSISTANT_WEIGHT);
        for (name, x) in [
            ("recency_weight", wr),
            ("importance_weight", wi),
            ("access_weight", wa),
            ("question_weight", wq),
            ("mmr_weight", mmr),
            ("expand_weight", we),
            ("assistant_weight", was),
        ] {
            if !x.is_finite() || x < 0.0 {
                bail!("`recall` option {name} must be a non-negative number, got {x}");
            }
        }
        if !half_life.is_finite() || half_life <= 0.0 {
            bail!("`recall` option half_life_days must be positive, got {half_life}");
        }
        if let Some(r) = opts.role.as_deref() {
            if !ROLES.contains(&r) {
                bail!("`recall` option role must be 'user', 'assistant' or 'other', got '{r}'");
            }
        }
        match opts.format.as_deref() {
            None | Some("list") | Some("context") => {}
            Some(f) => bail!("`recall` option format must be 'list' or 'context', got '{f}'"),
        }
        let diversity = opts.diversity.unwrap_or(true);
        let max_stmt = opts.max_per_statement.unwrap_or(DEFAULT_MAX_PER_STATEMENT);
        let max_src = opts.max_per_source.unwrap_or(DEFAULT_MAX_PER_SOURCE);
        let eids = match &opts.entity {
            Some(e) => {
                let eids = self.entity_view(&ns)?.cluster_of_name(e);
                if eids.is_empty() {
                    return Ok(vec![]);
                }
                Some(eids)
            }
            None => None,
        };
        let qe = cfg.embedder.as_ref().map(|e| {
            let q = format!("{e}_query");
            if embed::is_embedder_enabled(&q) {
                q
            } else {
                e.clone()
            }
        });
        let qvec = self.embed_if_on(qe.as_deref(), query)?;
        let k = opts.k.unwrap_or(10).max(1);
        let read_at = opts.as_of.unwrap_or_else(now_micros);
        let mut fetch = if opts.hide_pending { k * 2 } else { k };
        // post-filters (valid-from range, role) need a deeper fetch
        if opts.from.is_some() || opts.to.is_some() || opts.role.is_some() {
            fetch *= 4;
        }
        // the blend re-ranks a wider window of the most relevant (the index
        // legs fetch as for k, so the searches cost the same)
        let mut window = if blend { fetch.max(2 * k + 10) } else { fetch };
        if diversity {
            // the diverse selection chooses from a deeper window of the same
            // index fetch (deeper legs let name-only BM25 matches of short
            // claims into the fusion: LoCoMo R@5 fell 0.39 -> 0.36)
            window = window.max(opts.pool.unwrap_or((DIVERSITY_POOL_MULT * k).max(k + 10)));
        }
        let mut rows = self.mem_search(
            &cfg,
            &ns,
            query,
            qvec,
            fetch,
            window,
            opts.as_of,
            opts.kind.as_deref(),
            eids.as_deref(),
            w,
            !opts.include_archived,
            (!opts.include_expired).then_some(read_at),
        )?;
        // entity-graph expansion: other memories about the top hits' entities
        let mut expanded: BTreeSet<String> = BTreeSet::new();
        if opts.expand.unwrap_or(false) && eids.is_none() && !rows.is_empty() {
            let (boost, new_rows) = self.expand_rows(&ns, &rows, we, opts.as_of, opts.kind.as_deref())?;
            // a related memory already in the pool takes the expansion
            // score when it is higher than its own relevance
            for r in rows.iter_mut() {
                if let Some(id) = cell_str(&r[0]) {
                    if let Some(&b) = boost.get(&id) {
                        if b > r[1].get_float().unwrap_or(0.0) {
                            r[1] = DataValue::from(b);
                            expanded.insert(id);
                        }
                    }
                }
            }
            for r in new_rows {
                if let Some(id) = cell_str(&r[0]) {
                    expanded.insert(id);
                    rows.push(r);
                }
            }
        }
        let ids: Vec<String> = rows.iter().filter_map(|r| cell_str(&r[0])).collect();
        let iv = intervals(&self.mem_history(&ns, &ids)?, read_at);
        let mut pending: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut access: BTreeMap<String, (i64, i64)> = BTreeMap::new();
        let mut life: BTreeMap<String, (Option<i64>, Option<i64>)> = BTreeMap::new();
        let mut roles: BTreeMap<String, String> = BTreeMap::new();
        // the statement each memory came from first (its own)
        let mut origin: BTreeMap<String, (i64, String)> = BTreeMap::new();
        if !ids.is_empty() {
            // pending supersedes, access counts, the tier, the role and the
            // origin statement, in one read
            for r in self
                .mem_q(
                    &format!(
                        "?[tag, id, a, b] := *{MEM_DECISION}{{ns: $ns, new, cand: id, template: 'supersedes', status: 'pending'}}, \
                           id in $ids, *{MEM}{{ns: $ns, id: new @ $at}}, tag = 'p', a = new, b = null\n\
                         ?[tag, id, a, b] := id in $ids, *{MEM_ACCESS}{{ns: $ns, id, n: a, last: b}}, tag = 'a'\n\
                         ?[tag, id, a, b] := id in $ids, *{MEM_LIFE}{{ns: $ns, id, expires: a, archived: b}}, tag = 'l'\n\
                         ?[tag, id, a, b] := id in $ids, *{MEM_ROLE}{{ns: $ns, id, role: a}}, tag = 'r', b = null\n\
                         ?[tag, id, a, b] := id in $ids, *{MEM_SOURCE}{{ns: $ns, id, at: b, text: a}}, tag = 's'"
                    ),
                    vec![("ns", s(&ns)), ("ids", str_list(&ids)), ("at", DataValue::from(read_at))],
                )?
                .rows
            {
                let Some(id) = cell_str(&r[1]) else { continue };
                match r[0].get_str() {
                    Some("p") => pending.entry(id).or_default().extend(cell_str(&r[2])),
                    Some("a") => {
                        access.insert(id, (r[2].get_int().unwrap_or(0), r[3].get_int().unwrap_or(0)));
                    }
                    Some("l") => {
                        life.insert(id, (r[2].get_int(), r[3].get_int()));
                    }
                    Some("r") => {
                        if let Some(x) = cell_str(&r[2]) {
                            roles.insert(id, x);
                        }
                    }
                    Some("s") => {
                        if let (Some(t), Some(at)) = (cell_str(&r[2]), r[3].get_int()) {
                            if origin.get(&id).map_or(true, |(a0, _)| at < *a0) {
                                origin.insert(id, (at, t));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        let half_life_us = half_life * MICROS_PER_DAY;
        let mut hits: Vec<RecallHit> = rows
            .into_iter()
            .filter_map(|r| {
                let id = cell_str(&r[0])?;
                let by = pending.remove(&id).unwrap_or_default();
                if opts.hide_pending && !by.is_empty() {
                    return None;
                }
                let (from, to) = iv.get(&id).copied().unwrap_or((0, None));
                if opts.from.is_some_and(|f| from < f) || opts.to.is_some_and(|t| from >= t) {
                    return None;
                }
                let role = roles.get(&id).cloned();
                if let Some(want) = opts.role.as_deref() {
                    let is = role.as_deref().unwrap_or("user");
                    if is != want {
                        return None;
                    }
                }
                let is_exp = expanded.contains(&id);
                let (expires, archived) = life.get(&id).copied().unwrap_or((None, None));
                if is_exp
                    && ((!opts.include_archived && archived.is_some())
                        || (!opts.include_expired && expires.is_some_and(|e| e <= read_at)))
                {
                    return None;
                }
                let (mut n, mut last) = access.get(&id).copied().map_or((0, None), |(n, l)| (n, Some(l)));
                if let Some((bn, bl)) = self.mem_access.get(&ns, &id) {
                    n += bn;
                    last = Some(last.map_or(bl, |l| l.max(bl)));
                }
                let text = cell_str(&r[2])?;
                // a claim of a longer statement: the statement is its context
                let context = origin.remove(&id).map(|x| x.1).filter(|t| t.trim() != text.trim());
                let relevance = r[1].get_float().unwrap_or(0.0);
                let importance = r[6].get_float();
                let recency = 0.5f64.powf((read_at - from).max(0) as f64 / half_life_us);
                let acc = ((1.0 + n as f64).ln() / (1.0 + ACCESS_SATURATION).ln()).min(1.0);
                let score = if blend {
                    let q = if wq > 0.0 && !memory_claims::is_declarative(&text) { wq } else { 0.0 };
                    let a = if role.as_deref() == Some("assistant") { was } else { 0.0 };
                    relevance + wr * recency + wi * (importance.unwrap_or(0.5) - 0.5) + wa * acc - q - a
                } else {
                    relevance
                };
                Some(RecallHit {
                    pending_superseded_by: by,
                    score,
                    relevance,
                    recency,
                    access_count: n,
                    last_access: last,
                    expires,
                    archived: archived.is_some(),
                    text,
                    context,
                    kind: cell_str(&r[3]).unwrap_or_default(),
                    source: cell_str(&r[4]),
                    created: r[5].get_int().unwrap_or(0),
                    importance,
                    valid_from: from,
                    valid_to: to,
                    role,
                    expanded: is_exp,
                    id,
                })
            })
            .collect();
        if blend || !expanded.is_empty() {
            // stable: equal scores keep the relevance order
            hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        }
        if diversity && hits.len() > 1 {
            let vecs: BTreeMap<String, Vec<f32>> = if mmr > 0.0 && cfg.dim.is_some() {
                let ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
                self.mem_q(
                    &format!("?[id, vec] := id in $ids, *{MEM_VEC}{{ns: $ns, id, vec}}"),
                    vec![("ns", s(&ns)), ("ids", str_list(&ids))],
                )?
                .rows
                .into_iter()
                .filter_map(|r| Some((cell_str(&r[0])?, vec_of(&r[1])?)))
                .collect()
            } else {
                BTreeMap::new()
            };
            let cands: Vec<memory_recall::SelCand<'_>> = hits
                .iter()
                .map(|h| memory_recall::SelCand {
                    id: &h.id,
                    score: h.score,
                    group: h.source.as_deref().map(memory_recall::source_group),
                    vec: vecs.get(&h.id).map(|v| v.as_slice()),
                    words: if mmr > 0.0 { memory_recall::word_set(&h.text) } else { BTreeSet::new() },
                })
                .collect();
            let pick = memory_recall::select_diverse(&cands, k, max_stmt, max_src, mmr);
            drop(cands);
            let mut slots: Vec<Option<RecallHit>> = hits.into_iter().map(Some).collect();
            hits = pick.into_iter().filter_map(|i| slots[i].take()).collect();
        }
        hits.truncate(k);
        if opts.record_access.unwrap_or(true) && !hits.is_empty() {
            let got: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
            if self.mem_access.add(&ns, &got, now_micros()) {
                // best effort: a failed flush keeps the counts buffered
                let _ = self.memory_flush();
            }
        }
        Ok(hits)
    }

    /// Entity-graph expansion for `recall`: the canonical entities (merges
    /// followed) of the `EXPAND_FROM` most relevant rows, minus entities that
    /// more than `EXPAND_MAX_DEGREE` memories mention, and the other memories
    /// that mention them, scored `weight` × the best seed's relevance.
    /// Returns those scores for memories already among `rows`, and search
    /// rows for the others that are valid at the read time (at most
    /// `EXPAND_MAX_NEW`).
    fn expand_rows(
        &'s self,
        ns: &str,
        rows: &[Vec<DataValue>],
        weight: f64,
        as_of: Option<i64>,
        kind: Option<&str>,
    ) -> Result<(BTreeMap<String, f64>, Vec<Vec<DataValue>>)> {
        let seeds: Vec<(String, f64)> = rows
            .iter()
            .take(EXPAND_FROM)
            .filter_map(|r| Some((cell_str(&r[0])?, r[1].get_float().unwrap_or(0.0))))
            .collect();
        let seed_ids: Vec<String> = seeds.iter().map(|x| x.0.clone()).collect();
        let ment = self.mem_q(
            &format!("?[id, eid] := id in $ids, *{MEM_MENTION}{{ns: $ns, id, eid}}"),
            vec![("ns", s(ns)), ("ids", str_list(&seed_ids))],
        )?;
        if ment.rows.is_empty() {
            return Ok(Default::default());
        }
        let view = self.entity_view(ns)?;
        // canonical entity -> best seed relevance
        let mut canon: BTreeMap<String, f64> = BTreeMap::new();
        for r in &ment.rows {
            if let (Some(id), Some(e)) = (cell_str(&r[0]), cell_str(&r[1])) {
                let rel = seeds.iter().find(|x| x.0 == id).map_or(0.0, |x| x.1);
                let c = view.canonical(&e);
                let v = canon.entry(c).or_insert(rel);
                if rel > *v {
                    *v = rel;
                }
            }
        }
        let cluster: BTreeMap<String, String> = view
            .rows
            .keys()
            .map(|e| (e.clone(), view.canonical(e)))
            .filter(|(_, c)| canon.contains_key(c))
            .collect();
        let eids: Vec<String> = cluster.keys().cloned().collect();
        let mut by_canon: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for r in self
            .mem_q(
                &format!("?[eid, id] := eid in $eids, *{MEM_MENTION}{{ns: $ns, eid, id}}"),
                vec![("ns", s(ns)), ("eids", str_list(&eids))],
            )?
            .rows
        {
            if let (Some(e), Some(id)) = (cell_str(&r[0]), cell_str(&r[1])) {
                if let Some(c) = cluster.get(&e) {
                    by_canon.entry(c.clone()).or_default().insert(id);
                }
            }
        }
        let have: BTreeSet<String> = rows.iter().filter_map(|r| cell_str(&r[0])).collect();
        let mut cand: BTreeMap<String, f64> = BTreeMap::new();
        let mut boost: BTreeMap<String, f64> = BTreeMap::new();
        for (c, ids) in &by_canon {
            if ids.len() > EXPAND_MAX_DEGREE {
                continue;
            }
            let sc = weight * canon.get(c).copied().unwrap_or(0.0);
            // seeds too: another seed's entity can raise one (a lower
            // score than its own relevance is ignored by the caller)
            for id in ids {
                let m = if have.contains(id) { &mut boost } else { &mut cand };
                let v = m.entry(id.clone()).or_insert(sc);
                if sc > *v {
                    *v = sc;
                }
            }
        }
        if cand.is_empty() {
            return Ok((boost, vec![]));
        }
        let mut cand: Vec<(String, f64)> = cand.into_iter().collect();
        cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        cand.truncate(EXPAND_MAX_NEW);
        let mut p = vec![
            ("ns", s(ns)),
            (
                "cands",
                DataValue::List(
                    cand.iter()
                        .map(|(id, sc)| DataValue::List(vec![s(id), DataValue::from(*sc)]))
                        .collect(),
                ),
            ),
        ];
        let at = match as_of {
            Some(t) => {
                p.push(("as_of", DataValue::from(t)));
                "$as_of"
            }
            None => "'NOW'",
        };
        let mut filters = String::new();
        if let Some(kd) = kind {
            filters.push_str(", kind == $kind");
            p.push(("kind", s(kd)));
        }
        let mut out = self
            .mem_q(
                &format!(
                    "c[id, score] <- $cands\n\
                     ?[id, score, text, kind, source, created, importance] := c[id, score], \
                       *{MEM}{{ns: $ns, id, text, kind, source, created, importance @ {at}}}{filters}"
                ),
                p,
            )?
            .rows;
        out.sort_by(|a, b| {
            b[1].get_float()
                .partial_cmp(&a[1].get_float())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| cell_str(&a[0]).cmp(&cell_str(&b[0])))
        });
        Ok((boost, out))
    }

    /// `recall` as a token-budgeted text block for an LLM reader
    /// (`recall {format: 'context', budget}`; docs/specs/memory-api.md,
    /// "Context output"): the same hits, grouped by entity (else by source
    /// group), each with its valid-from date, the values it superseded
    /// ("previously: X (from to until)"), unconfirmed replacements, and the
    /// statement it came from while the budget lasts. Deterministic.
    pub fn recall_context(&'s self, query: &str, opts: RecallOpts) -> Result<memory_recall::RecallContext> {
        let budget = opts.budget.unwrap_or(DEFAULT_CONTEXT_BUDGET);
        let show_source = opts.show_source;
        let ns = ns_of(&opts.ns);
        let read_at = opts.as_of.unwrap_or_else(now_micros);
        let hits = self.recall(query, opts)?;
        if hits.is_empty() {
            return Ok(memory_recall::render_context(&[], budget, show_source));
        }
        let ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
        // entities: canonical names
        let mut ents: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let ment = self.mem_q(
            &format!("?[id, eid] := id in $ids, *{MEM_MENTION}{{ns: $ns, id, eid}}"),
            vec![("ns", s(&ns)), ("ids", str_list(&ids))],
        )?;
        if !ment.rows.is_empty() {
            let view = self.entity_view(&ns)?;
            for r in ment.rows {
                if let (Some(id), Some(e)) = (cell_str(&r[0]), cell_str(&r[1])) {
                    let c = view.canonical(&e);
                    let name = view.rows.get(&c).or_else(|| view.rows.get(&e)).map(|x| x.0.clone());
                    if let Some(n) = name {
                        let v = ents.entry(id).or_default();
                        if !v.contains(&n) {
                            v.push(n);
                        }
                    }
                }
            }
            for v in ents.values_mut() {
                v.sort();
            }
        }
        // previously: follow `supersedes` links, up to three steps back
        let mut prev: BTreeMap<String, Vec<memory_recall::Previous>> = BTreeMap::new();
        let mut frontier: Vec<(String, String)> = ids.iter().map(|i| (i.clone(), i.clone())).collect();
        for _ in 0..3 {
            if frontier.is_empty() {
                break;
            }
            let srcs: Vec<String> = frontier.iter().map(|x| x.1.clone()).collect();
            let links = self.mem_q(
                &format!("?[src, dst] := src in $srcs, *{MEM_LINK}{{ns: $ns, src, rel: 'supersedes', dst}}"),
                vec![("ns", s(&ns)), ("srcs", str_list(&srcs))],
            )?;
            let pairs: Vec<(String, String)> = links
                .rows
                .iter()
                .filter_map(|r| Some((cell_str(&r[0])?, cell_str(&r[1])?)))
                .collect();
            if pairs.is_empty() {
                break;
            }
            let olds: Vec<String> = pairs.iter().map(|x| x.1.clone()).collect::<BTreeSet<_>>().into_iter().collect();
            let versions = self.mem_versions(&ns, &olds, read_at)?;
            let mut next = vec![];
            for (root, src) in &frontier {
                for (a, b) in &pairs {
                    if a != src {
                        continue;
                    }
                    if let Some(v) = versions.get(b) {
                        prev.entry(root.clone()).or_default().push(v.clone());
                        next.push((root.clone(), b.clone()));
                    }
                }
            }
            frontier = next;
        }
        for v in prev.values_mut() {
            v.sort_by(|a, b| b.from.cmp(&a.from).then(a.text.cmp(&b.text)));
        }
        // unconfirmed replacements (pending supersedes)
        let newer: Vec<String> = hits
            .iter()
            .flat_map(|h| h.pending_superseded_by.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let newer_v = if newer.is_empty() {
            BTreeMap::new()
        } else {
            self.mem_versions(&ns, &newer, read_at)?
        };
        let items: Vec<memory_recall::CtxItem> = hits
            .into_iter()
            .map(|h| memory_recall::CtxItem {
                entities: ents.remove(&h.id).unwrap_or_default(),
                previously: prev.remove(&h.id).unwrap_or_default(),
                maybe_replaced_by: h
                    .pending_superseded_by
                    .iter()
                    .filter_map(|n| newer_v.get(n).map(|v| (v.text.clone(), v.from)))
                    .collect(),
                id: h.id,
                text: h.text,
                valid_from: h.valid_from,
                valid_to: h.valid_to,
                role: h.role,
                source: h.source,
                context: h.context,
            })
            .collect();
        Ok(memory_recall::render_context(&items, budget, show_source))
    }

    /// The last version of each memory that started at or before `t`: its
    /// text, valid-from and the end of that version (a retraction), if any.
    fn mem_versions(&'s self, ns: &str, ids: &[String], t: i64) -> Result<BTreeMap<String, memory_recall::Previous>> {
        let mut by: BTreeMap<String, Vec<(i64, bool, String)>> = BTreeMap::new();
        for r in self
            .mem_q(
                &format!("?[id, vt, text] := id in $ids, *{MEM}{{ns: $ns, id, vt, text}}"),
                vec![("ns", s(ns)), ("ids", str_list(ids))],
            )?
            .rows
        {
            if let (Some(id), DataValue::Validity(v), Some(text)) = (cell_str(&r[0]), &r[1], cell_str(&r[2])) {
                by.entry(id).or_default().push((v.timestamp.0 .0, v.is_assert.0, text));
            }
        }
        let mut out = BTreeMap::new();
        for (id, mut v) in by {
            v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            // the last assertion at or before t, and the first retraction after it
            let Some(ai) = v.iter().rposition(|x| x.1 && x.0 <= t) else { continue };
            let to = v[ai + 1..].iter().find(|x| !x.1).map(|x| x.0);
            out.insert(
                id,
                memory_recall::Previous {
                    text: v[ai].2.clone(),
                    from: v[ai].0,
                    to,
                },
            );
        }
        Ok(out)
    }

    /// Write the buffered access counts (`kleos_mem_access`) in one small
    /// transaction of their own. `recall` calls it every second or 64
    /// memories; call it before closing a database to keep the last counts.
    /// Returns the number of memories written.
    pub fn memory_flush(&'s self) -> Result<usize> {
        let pending = self.mem_access.take();
        if pending.is_empty() {
            return Ok(0);
        }
        let rows: Vec<DataValue> = pending
            .iter()
            .map(|((ns, id), (n, l))| DataValue::List(vec![s(ns), s(id), DataValue::from(*n), DataValue::from(*l)]))
            .collect();
        let n = rows.len();
        // only memories that still exist (an erase may have raced the buffer)
        let script = format!(
            "d[ns, id, c, l] <- $rows\n\
             live[ns, id, c, l] := d[ns, id, c, l], *{MEM}{{ns, id}}\n\
             ?[ns, id, n, last] := live[ns, id, c, l], *{MEM_ACCESS}{{ns, id, n: n0, last: l0}}, n = n0 + c, last = max(l0, l)\n\
             ?[ns, id, n, last] := live[ns, id, c, l], not *{MEM_ACCESS}{{ns, id}}, n = c, last = l\n\
             :put {MEM_ACCESS} {{ns, id => n, last}}"
        );
        match self.mem_q(&script, vec![("rows", DataValue::List(rows))]) {
            Ok(_) => Ok(n),
            Err(e) => {
                self.mem_access.restore(pending);
                Err(e)
            }
        }
    }

    /// Change scoring metadata of memories: importance, expiry, archive
    /// state. The text and validity are untouched; importance is rewritten
    /// in place on the memory's current row (it is not versioned). Commits
    /// under the namespace version, like `forget`, since the archive and
    /// expiry change which memories `remember` may treat as duplicates.
    pub fn memory_set(&'s self, opts: SetOpts) -> Result<JsonValue> {
        if self.mem_config()?.is_none() {
            bail!("`memory set`: there is no memory store yet");
        }
        let ns = ns_of(&opts.ns);
        let mut ids = opts.ids.clone();
        ids.extend(opts.id.clone());
        let mut ids = self.with_claims(&ns, ids)?;
        ids.sort();
        ids.dedup();
        if ids.is_empty() {
            bail!("`memory set` needs an id or ids");
        }
        if let Some(x) = opts.importance {
            if !(0.0..=1.0).contains(&x) {
                bail!("`memory set` importance must be in [0, 1], got {x}");
            }
        }
        if opts.no_ttl && (opts.ttl.is_some() || opts.expires.is_some()) {
            bail!("`memory set` takes no_ttl or ttl/expires, not both");
        }
        let gate = ns_gate(&ns);
        let _commit = gate.commit();
        let now = now_micros();
        // current memories only, with the valid time of their current row
        let hist = self.mem_history(&ns, &ids)?;
        let iv = intervals(&hist, now);
        let found: Vec<String> = iv.keys().cloned().collect();
        if found.is_empty() {
            return Ok(serde_json::json!({"ns": ns, "ids": found}));
        }
        let mut script = String::new();
        let mut p = vec![("ns", s(&ns)), ("ids", str_list(&found))];
        if let Some(x) = opts.importance {
            // the latest assertion at or before now holds the current values
            let rows: Vec<DataValue> = found
                .iter()
                .filter_map(|id| {
                    let ts = hist
                        .iter()
                        .filter(|(i, t, a)| i == id && *a && *t <= now)
                        .map(|(_, t, _)| *t)
                        .max()?;
                    Some(DataValue::List(vec![s(id), DataValue::from(ts)]))
                })
                .collect();
            script.push_str(&format!(
                "{{cur[id, ts] <- $cur\n\
                  ?[ns, id, vt, text, kind, source, created, importance] := cur[id, ts], \
                    *{MEM}{{ns, id, vt, text, kind, source, created}}, ns = $ns, to_int(vt) == ts, to_bool(vt), importance = $imp \
                  :put {MEM} {{ns, id, vt => text, kind, source, created, importance}}}}"
            ));
            p.push(("cur", DataValue::List(rows)));
            p.push(("imp", DataValue::from(x)));
        }
        if opts.ttl.is_some() || opts.expires.is_some() || opts.no_ttl || opts.archived.is_some() {
            let old = self.life_of(&ns, &found)?;
            let rows: Vec<DataValue> = found
                .iter()
                .map(|id| {
                    let (mut e, mut a) = old.get(id).copied().unwrap_or((None, None));
                    if opts.no_ttl {
                        e = None;
                    }
                    if let Some(x) = opts.expires.or(opts.ttl.map(|t| now.saturating_add(t))) {
                        e = Some(x);
                    }
                    match opts.archived {
                        Some(true) => a = a.or(Some(now)),
                        Some(false) => a = None,
                        None => {}
                    }
                    let opt_i = |v: Option<i64>| v.map(DataValue::from).unwrap_or(DataValue::Null);
                    DataValue::List(vec![s(&ns), s(id), opt_i(e), opt_i(a)])
                })
                .collect();
            script.push_str(&format!(
                "{{?[ns, id, expires, archived] <- $life :put {MEM_LIFE} {{ns, id => expires, archived}}}}"
            ));
            p.push(("life", DataValue::List(rows)));
        }
        if script.is_empty() {
            bail!("`memory set` needs importance, ttl, expires, no_ttl or archived");
        }
        script.push_str(&bump_ns_head());
        self.mem_q(&script, p)?;
        Ok(serde_json::json!({"ns": ns, "ids": found}))
    }

    /// Forget memories by id or filter. `retract` (default) ends their
    /// validity at `at` (history, `as_of` recall and `::history` keep them);
    /// `erase` deletes every version through `::erase` (search-index entries
    /// included), with their vectors, sources, mentions, links and decision
    /// rows, and entities left with no mention.
    pub fn forget(&'s self, opts: ForgetOpts) -> Result<ForgetResult> {
        let mode = opts.mode.clone().unwrap_or_else(|| "retract".to_string());
        if !["retract", "erase", "archive"].contains(&mode.as_str()) {
            bail!("`forget` mode must be 'retract', 'erase' or 'archive', got '{mode}'");
        }
        let cfg = match self.mem_config()? {
            Some(c) => c,
            None => {
                return Ok(ForgetResult {
                    mode,
                    ids: vec![],
                })
            }
        };
        let ns = ns_of(&opts.ns);
        // a forget changes what is current: commit under the namespace gate
        // so a concurrent `remember` plans against the result
        let gate = ns_gate(&ns);
        let _commit = gate.commit();
        let mut ids: Vec<String> = opts.ids.clone();
        ids.extend(opts.id.clone());
        let mut ids = self.with_claims(&ns, ids)?;
        let by_filter = opts.kind.is_some() || opts.entity.is_some() || opts.expired;
        if ids.is_empty() && !by_filter && !opts.all {
            bail!("`forget` needs an id, ids, a kind, entity or expired filter, or all: true");
        }
        if !ids.is_empty() && (by_filter || opts.all) {
            bail!("`forget` takes ids or a filter, not both");
        }
        if ids.is_empty() {
            // filters select current memories
            let mut body = format!("*{MEM}{{ns: $ns, id, kind @ 'NOW'}}");
            let mut p = vec![("ns", s(&ns))];
            if let Some(k) = &opts.kind {
                body.push_str(", kind == $kind");
                p.push(("kind", s(k)));
            }
            if let Some(e) = &opts.entity {
                body.push_str(&format!(", *{MEM_MENTION}{{ns: $ns, id, eid}}, is_in(eid, $eids)"));
                p.push(("eids", str_list(&self.entity_view(&ns)?.cluster_of_name(e))));
            }
            if opts.expired {
                body.push_str(&format!(
                    ", *{MEM_LIFE}{{ns: $ns, id, expires: e, archived: a}}, !is_null(e), e <= $now, is_null(a)"
                ));
                p.push(("now", DataValue::from(opts.at.unwrap_or_else(now_micros))));
            }
            ids = self
                .mem_q(&format!("?[id] := {body}"), p)?
                .rows
                .into_iter()
                .filter_map(|r| cell_str(&r[0]))
                .collect();
        }
        ids.sort();
        ids.dedup();
        if ids.is_empty() {
            return Ok(ForgetResult { mode, ids });
        }
        let has_vec = cfg.dim.is_some();
        if mode == "archive" {
            // the archive tier: still valid, hidden from recall by default
            let at = opts.at.unwrap_or_else(now_micros);
            let cur: Vec<String> = self
                .mem_q(
                    &format!("?[id] := *{MEM}{{ns: $ns, id @ 'NOW'}}, id in $ids"),
                    vec![("ns", s(&ns)), ("ids", str_list(&ids))],
                )?
                .rows
                .into_iter()
                .filter_map(|r| cell_str(&r[0]))
                .collect();
            if cur.is_empty() {
                return Ok(ForgetResult { mode, ids: cur });
            }
            let old = self.life_of(&ns, &cur)?;
            let rows: Vec<DataValue> = cur
                .iter()
                .map(|id| {
                    let (e, a) = old.get(id).copied().unwrap_or((None, None));
                    DataValue::List(vec![
                        s(&ns),
                        s(id),
                        e.map(DataValue::from).unwrap_or(DataValue::Null),
                        DataValue::from(a.unwrap_or(at)),
                    ])
                })
                .collect();
            let mut script = format!(
                "{{?[ns, id, expires, archived] <- $life :put {MEM_LIFE} {{ns, id => expires, archived}}}}"
            );
            script.push_str(&bump_ns_head());
            self.mem_q(&script, vec![("ns", s(&ns)), ("life", DataValue::List(rows))])?;
            return Ok(ForgetResult { mode, ids: cur });
        }
        if mode == "retract" {
            let at = opts.at.unwrap_or_else(now_micros);
            let cur: Vec<String> = self
                .mem_q(
                    &format!("?[id] := *{MEM}{{ns: $ns, id @ 'NOW'}}, id in $ids"),
                    vec![("ns", s(&ns)), ("ids", str_list(&ids))],
                )?
                .rows
                .into_iter()
                .filter_map(|r| cell_str(&r[0]))
                .collect();
            if cur.is_empty() {
                return Ok(ForgetResult { mode, ids: cur });
            }
            let mut script = format!(
                "{{?[ns, id, vt, text, kind, source, created, importance] := *{MEM}{{ns, id, text, kind, source, created, importance @ 'NOW'}}, \
                   ns = $ns, id in $ids, vt = [$at, false] \
                  :put {MEM} {{ns, id, vt => text, kind, source, created, importance}}}}"
            );
            if has_vec {
                script.push_str(&format!(
                    "{{?[ns, id, current] := *{MEM_VEC}{{ns, id}}, ns = $ns, id in $ids, current = false \
                      :update {MEM_VEC} {{ns, id => current}}}}"
                ));
            }
            let mut p = vec![("ns", s(&ns)), ("ids", str_list(&cur)), ("at", DataValue::from(at))];
            let job = Job { op: "retract", mem_ids: &cur, old: &[], text: None, at, entities: &[] };
            if let Some((frag, ps)) = engram_hook::enqueue(&cfg, job) {
                script.push_str(&frag);
                p.extend(ps);
            }
            script.push_str(&bump_ns_head());
            self.mem_q(&script, p)?;
            return Ok(ForgetResult { mode, ids: cur });
        }

        // erase: only keys that exist at some time
        let exist: Vec<String> = self
            .mem_q(
                &format!("?[id] := *{MEM}{{ns: $ns, id}}, id in $ids"),
                vec![("ns", s(&ns)), ("ids", str_list(&ids))],
            )?
            .rows
            .into_iter()
            .filter_map(|r| cell_str(&r[0]))
            .collect();
        if exist.is_empty() {
            return Ok(ForgetResult { mode, ids: exist });
        }
        let eids: Vec<String> = self
            .mem_q(
                &format!("?[eid] := *{MEM_MENTION}{{ns: $ns, id, eid}}, id in $ids"),
                vec![("ns", s(&ns)), ("ids", str_list(&exist))],
            )?
            .rows
            .into_iter()
            .filter_map(|r| cell_str(&r[0]))
            .collect();
        let keys = DataValue::List(
            exist
                .iter()
                .map(|id| DataValue::List(vec![s(&ns), s(id)]))
                .collect(),
        );
        let mut script = String::new();
        if has_vec {
            script.push_str(&format!(
                "{{?[ns, id] := *{MEM_VEC}{{ns, id}}, ns = $ns, id in $ids :rm {MEM_VEC} {{ns, id}}}}"
            ));
        }
        script.push_str(&format!(
            "{{?[ns, id, at, stmt] := *{MEM_SOURCE}{{ns, id, at, stmt}}, ns = $ns, id in $ids :rm {MEM_SOURCE} {{ns, id, at, stmt}}}}\
             {{?[ns, id, eid] := *{MEM_MENTION}{{ns, id, eid}}, ns = $ns, id in $ids :rm {MEM_MENTION} {{ns, id, eid}}}}\
             {{?[ns, new, cand, template] := *{MEM_DECISION}{{ns, new, cand, template}}, ns = $ns, is_in(new, $ids) || is_in(cand, $ids) \
               :rm {MEM_DECISION} {{ns, new, cand, template}}}}\
             {{used[eid] := *{MEM_MENTION}{{ns: $ns, eid}}\n\
               ?[ns, eid] := ns = $ns, eid in $eids, not used[eid] :rm {MEM_ENTITY} {{ns, eid}}}}\
             {{gone[x] := x in $ids\n gone[x] := x in $eids, not *{MEM_ENTITY}{{ns: $ns, eid: x}}\n\
               ?[ns, src, rel, dst] := *{MEM_LINK}{{ns, src, rel, dst}}, ns = $ns, gone[src]\n\
               ?[ns, src, rel, dst] := *{MEM_LINK}{{ns, src, rel, dst}}, ns = $ns, gone[dst] \
               :rm {MEM_LINK} {{ns, src, rel, dst}}}}\
             {{?[ns, id] := *{MEM_ACCESS}{{ns, id}}, ns = $ns, id in $ids :rm {MEM_ACCESS} {{ns, id}}}}\
             {{?[ns, id] := *{MEM_LIFE}{{ns, id}}, ns = $ns, id in $ids :rm {MEM_LIFE} {{ns, id}}}}\
             {{?[ns, id] := *{MEM_ROLE}{{ns, id}}, ns = $ns, id in $ids :rm {MEM_ROLE} {{ns, id}}}}\
             {{gone[x] := x in $eids, not *{MEM_ENTITY}{{ns: $ns, eid: x}}\n\
               ?[relation, from, vt] := *{ENTITY_MERGES}{{relation, from, vt, into}}, relation = $erel, gone[from]\n\
               ?[relation, from, vt] := *{ENTITY_MERGES}{{relation, from, vt, into}}, relation = $erel, gone[into] \
               :rm {ENTITY_MERGES} {{relation, from, vt}}}}\
             {{gone[x] := x in $eids, not *{MEM_ENTITY}{{ns: $ns, eid: x}}\n\
               ?[relation, a, b] := *{ENTITY_LINKS}{{relation, a, b}}, relation = $erel, gone[a]\n\
               ?[relation, a, b] := *{ENTITY_LINKS}{{relation, a, b}}, relation = $erel, gone[b] \
               :rm {ENTITY_LINKS} {{relation, a, b}}}}"
        ));
        let mut p = vec![("ns", s(&ns)), ("ids", str_list(&exist)), ("eids", str_list(&eids))];
        let job = Job { op: "erase", mem_ids: &exist, old: &[], text: None, at: now_micros(), entities: &[] };
        if let Some((frag, ps)) = engram_hook::enqueue(&cfg, job) {
            script.push_str(&frag);
            p.extend(ps);
        }
        script.push_str(&bump_ns_head());
        // The dependent rows and the Engram erase job commit first, the
        // memory rows second: after a crash in between, the overlay is
        // already being erased and the memory is still there, so repeating
        // the `forget` completes it. The other order could leave a memory
        // erased while its Engram rows still answer.
        p.push(("erel", s(MEM_ENTITY)));
        self.mem_q(&script, p)?;
        // ::erase deletes every row of each key with its FTS/HNSW entries and
        // writes the salted audit marker (kleos_evict_audit)
        self.mem_q(&format!("::erase {MEM} $keys"), vec![("keys", keys)])?;
        self.mem_access.forget(&ns, &exist);
        Ok(ForgetResult { mode, ids: exist })
    }

    /// A minimal, model-free look at the namespace: pending tentative
    /// decisions, recent validity changes (`::changes`) and contradiction
    /// candidates (pending supersedes decisions, plus current memories that
    /// share an entity and differ only in a value, by the value rules).
    pub fn reflect(&'s self, opts: ReflectOpts) -> Result<ReflectResult> {
        let ns = ns_of(&opts.ns);
        let mut res = ReflectResult {
            ns: ns.clone(),
            ..Default::default()
        };
        if self.mem_config()?.is_none() {
            return Ok(res);
        }
        let limit = opts.limit.unwrap_or(50);
        let pend = self.mem_q(
            &format!(
                "?[at, new, cand, template, label, p, band, decided_by] := \
                   *{MEM_DECISION}{{ns: $ns, new, cand, template, label, p, band, decided_by, status: 'pending', at}}\n\
                 :order -at\n:limit {limit}"
            ),
            vec![("ns", s(&ns))],
        )?;
        let mut need: Vec<String> = vec![];
        for r in &pend.rows {
            need.extend(cell_str(&r[1]));
            need.extend(cell_str(&r[2]));
        }
        let texts: BTreeMap<String, String> = if need.is_empty() {
            BTreeMap::new()
        } else {
            self.mem_q(
                &format!("?[id, text] := *{MEM}{{ns: $ns, id, text}}, id in $ids"),
                vec![("ns", s(&ns)), ("ids", str_list(&need))],
            )?
            .rows
            .into_iter()
            .filter_map(|r| Some((cell_str(&r[0])?, cell_str(&r[1])?)))
            .collect()
        };
        for r in pend.rows {
            let (Some(new), Some(cand)) = (cell_str(&r[1]), cell_str(&r[2])) else {
                continue;
            };
            let d = PendingDecision {
                new_text: texts.get(&new).cloned(),
                candidate_text: texts.get(&cand).cloned(),
                new,
                candidate: cand,
                template: cell_str(&r[3]).unwrap_or_default(),
                label: cell_str(&r[4]).unwrap_or_default(),
                p: r[5].get_float().unwrap_or(0.0),
                band: cell_str(&r[6]).unwrap_or_default(),
                decided_by: cell_str(&r[7]).unwrap_or_default(),
                at: r[0].get_int().unwrap_or(0),
            };
            if d.template == "supersedes" {
                if let (Some(a), Some(b)) = (&d.candidate_text, &d.new_text) {
                    res.contradictions.push(ContradictionCandidate {
                        a: d.candidate.clone(),
                        b: d.new.clone(),
                        a_text: a.clone(),
                        b_text: b.clone(),
                        reason: format!("pending supersedes decision (p {:.3}, {})", d.p, d.decided_by),
                    });
                }
            }
            res.pending.push(d);
        }

        let since = opts.since.unwrap_or_else(|| now_micros() - 7 * 86_400_000_000);
        let ch = self.mem_q(
            &format!("::changes {MEM} {{from: $since, prefix: [$ns]}}"),
            vec![("ns", s(&ns)), ("since", DataValue::from(since))],
        )?;
        let col = |name: &str| ch.headers.iter().position(|h| h == name);
        if let (Some(ci), Some(ai), Some(ii), Some(ti), Some(ki)) =
            (col("change"), col("at"), col("id"), col("text"), col("kind"))
        {
            let n = ch.rows.len();
            for r in ch.rows.iter().skip(n.saturating_sub(limit)) {
                res.changes.push(MemChange {
                    id: cell_str(&r[ii]).unwrap_or_default(),
                    change: cell_str(&r[ci]).unwrap_or_default(),
                    at: r[ai].get_int().unwrap_or(0),
                    text: cell_str(&r[ti]).unwrap_or_default(),
                    kind: cell_str(&r[ki]).unwrap_or_default(),
                });
            }
        }

        // model-free: current memories sharing an entity whose values differ
        let shared = self.mem_q(
            &format!(
                "?[eid, created, id, text] := *{MEM_MENTION}{{ns: $ns, id, eid}}, \
                   *{MEM}{{ns: $ns, id, text, created @ 'NOW'}}"
            ),
            vec![("ns", s(&ns))],
        )?;
        // grouped by canonical entity: merged names count as one
        let view = self.entity_view(&ns)?;
        let mut groups: BTreeMap<String, Vec<(i64, String, String)>> = BTreeMap::new();
        for r in shared.rows {
            if let (Some(e), Some(id), Some(t)) = (cell_str(&r[0]), cell_str(&r[2]), cell_str(&r[3])) {
                let g = groups.entry(view.canonical(&e)).or_default();
                let item = (r[1].get_int().unwrap_or(0), id, t);
                if !g.contains(&item) {
                    g.push(item);
                }
            }
        }
        let mut seen: BTreeSet<(String, String)> = res
            .contradictions
            .iter()
            .map(|c| (c.a.clone(), c.b.clone()))
            .collect();
        'outer: for (_, mut g) in groups {
            g.sort();
            g.truncate(30);
            for i in 0..g.len() {
                for j in i + 1..g.len() {
                    if res.contradictions.len() >= limit {
                        break 'outer;
                    }
                    let (a, b) = (&g[i], &g[j]);
                    let reason = match rules::value_rule(&a.2, &b.2) {
                        rules::ValueRule::Update(r) => format!("value rule (update): {r}"),
                        rules::ValueRule::Hint(r) => format!("value rule (hint): {r}"),
                        rules::ValueRule::None => continue,
                    };
                    if seen.insert((a.1.clone(), b.1.clone())) {
                        res.contradictions.push(ContradictionCandidate {
                            a: a.1.clone(),
                            b: b.1.clone(),
                            a_text: a.2.clone(),
                            b_text: b.2.clone(),
                            reason,
                        });
                    }
                }
            }
        }
        Ok(res)
    }

    /// Add a typed link between two memories or entities of a namespace.
    pub fn link(&'s self, opts: LinkOpts) -> Result<JsonValue> {
        if opts.rel.trim().is_empty() || opts.from.is_empty() || opts.to.is_empty() {
            bail!("`link` needs from, to and rel");
        }
        self.mem_ready(None)?;
        let ns = ns_of(&opts.ns);
        let known: BTreeSet<String> = self
            .mem_q(
                &format!(
                    "?[x] := x in $xs, *{MEM}{{ns: $ns, id: x}}\n\
                     ?[x] := x in $xs, *{MEM_ENTITY}{{ns: $ns, eid: x}}"
                ),
                vec![("ns", s(&ns)), ("xs", str_list(&[opts.from.clone(), opts.to.clone()]))],
            )?
            .rows
            .into_iter()
            .filter_map(|r| cell_str(&r[0]))
            .collect();
        for x in [&opts.from, &opts.to] {
            if !known.contains(x) {
                bail!("`link`: '{x}' is neither a memory nor an entity of namespace '{ns}'");
            }
        }
        let now = now_micros();
        self.mem_q(
            &format!(
                "?[ns, src, rel, dst, created, p, decided_by] <- [[$ns, $src, $rel, $dst, $now, $p, null]] \
                 :put {MEM_LINK} {{ns, src, rel, dst => created, p, decided_by}}"
            ),
            vec![
                ("ns", s(&ns)),
                ("src", s(&opts.from)),
                ("rel", s(opts.rel.trim())),
                ("dst", s(&opts.to)),
                ("now", DataValue::from(now)),
                ("p", opts.p.map(DataValue::from).unwrap_or(DataValue::Null)),
            ],
        )?;
        Ok(serde_json::json!({
            "ns": ns, "from": opts.from, "rel": opts.rel.trim(), "to": opts.to, "created": now
        }))
    }

    /// One entry point for every surface that speaks JSON (HTTP, the
    /// `::memory` op, bindings): `op` is `init`, `remember`, `recall`,
    /// `forget`, `reflect` or `link`; `args` an object of that op's options,
    /// plus `text` (remember), `query` (recall) or `id` (forget).
    pub fn memory_call(&'s self, op: &str, args: JsonValue) -> Result<JsonValue> {
        let mut args = match args {
            JsonValue::Null => serde_json::Map::new(),
            JsonValue::Object(m) => m,
            other => bail!("memory {op}: options must be an object, got {other}"),
        };
        let mut take_str = |k: &str| -> Result<Option<String>> {
            match args.remove(k) {
                None | Some(JsonValue::Null) => Ok(None),
                Some(JsonValue::String(s)) => Ok(Some(s)),
                Some(v) => bail!("memory {op}: '{k}' must be a string, got {v}"),
            }
        };
        let out = match op {
            "init" => {
                let cfg = self.memory_init(from_args(op, args)?)?;
                serde_json::to_value(cfg).into_diagnostic()?
            }
            "remember" => {
                let text = take_str("text")?.ok_or_else(|| miette::miette!("memory remember needs 'text'"))?;
                serde_json::to_value(self.remember(&text, from_args(op, args)?)?).into_diagnostic()?
            }
            "recall" => {
                let q = take_str("query")?.ok_or_else(|| miette::miette!("memory recall needs 'query'"))?;
                let opts: RecallOpts = from_args(op, args)?;
                if opts.format.as_deref() == Some("context") {
                    serde_json::to_value(self.recall_context(&q, opts)?).into_diagnostic()?
                } else {
                    serde_json::to_value(self.recall(&q, opts)?).into_diagnostic()?
                }
            }
            "forget" => serde_json::to_value(self.forget(from_args(op, args)?)?).into_diagnostic()?,
            "set" => self.memory_set(from_args(op, args)?)?,
            "flush" => {
                if let Some(k) = args.keys().next() {
                    bail!("memory flush takes no options, got '{k}'");
                }
                serde_json::json!({ "flushed": self.memory_flush()? })
            }
            "reflect" => serde_json::to_value(self.reflect(from_args(op, args)?)?).into_diagnostic()?,
            "link" => self.link(from_args(op, args)?)?,
            other => bail!(
                "unknown memory op '{other}': expected init, remember, recall, forget, reflect, link, set or flush"
            ),
        };
        Ok(out)
    }
}

fn from_args<T: serde::de::DeserializeOwned>(op: &str, args: serde_json::Map<String, JsonValue>) -> Result<T> {
    serde_json::from_value(JsonValue::Object(args))
        .map_err(|e| miette::miette!("memory {op}: bad options: {e}"))
}

/// `::memory` result rows: one column per top-level field (lists of records
/// become one row each for `recall`).
pub(crate) fn memory_rows(op: &str, v: JsonValue) -> NamedRows {
    let to_row = |o: &serde_json::Map<String, JsonValue>, cols: &[String]| -> Vec<DataValue> {
        cols.iter()
            .map(|c| match o.get(c) {
                None | Some(JsonValue::Null) => DataValue::Null,
                Some(JsonValue::Object(_)) | Some(JsonValue::Array(_)) => {
                    let x = o.get(c).unwrap();
                    // keep lists of plain values as lists, records as JSON
                    match x {
                        JsonValue::Array(a) if a.iter().all(|y| !y.is_object()) => DataValue::from(x),
                        _ => DataValue::Json(crate::JsonData(x.clone())),
                    }
                }
                Some(x) => DataValue::from(x),
            })
            .collect()
    };
    match (op, v) {
        ("recall", JsonValue::Array(hits)) => {
            let cols: Vec<String> = [
                "id", "text", "score", "kind", "valid_from", "valid_to", "created", "source", "importance",
                "pending_superseded_by", "relevance", "recency", "access_count", "last_access", "expires", "archived",
                "context", "role",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            let rows = hits
                .iter()
                .filter_map(|h| h.as_object().map(|o| to_row(o, &cols)))
                .collect();
            NamedRows::new(cols, rows)
        }
        (_, JsonValue::Object(o)) => {
            let cols: Vec<String> = o.keys().cloned().collect();
            let row = to_row(&o, &cols);
            NamedRows::new(cols, vec![row])
        }
        (_, other) => NamedRows::new(vec!["result".to_string()], vec![vec![DataValue::from(other)]]),
    }
}
