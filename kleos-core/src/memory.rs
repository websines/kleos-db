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

use std::collections::{BTreeMap, BTreeSet};
use std::time::{SystemTime, UNIX_EPOCH};

use itertools::Itertools;
use miette::{IntoDiagnostic, Result};
use serde_derive::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::bail;
use crate::data::value::DataValue;
use crate::decide::{self, rules, Band, Decision, Template};
use crate::embed;
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

const DEFAULT_NS: &str = "default";
const DEFAULT_KIND: &str = "fact";
/// Candidates scored per `remember`.
const DEFAULT_K: usize = 5;
/// Vector-leg weight of the hybrid fusion (the SciFact-tuned 0.6, CHANGELOG).
const DEFAULT_VECTOR_WEIGHT: f64 = 0.6;

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
    s.split(|c: char| !c.is_alphanumeric())
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
    /// Entity names mentioned (exact normalised-name linking for now).
    pub entities: Vec<String>,
    /// Valid-from time in µs since the epoch; default now.
    pub at: Option<i64>,
    /// Candidates to score; default 5.
    pub k: Option<usize>,
    /// Decider name; default the store's, else the first enabled one.
    pub decider: Option<String>,
    /// Embedder name for the first `remember` of a new store.
    pub embedder: Option<String>,
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
    /// Required to forget a whole namespace with no other filter.
    pub all: bool,
    /// `retract` (end validity; default) or `erase` (GDPR hard delete).
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
    /// Whether the text was embedded.
    pub embedded: bool,
    /// Decider used (`None`: model-free rules only).
    pub decider: Option<String>,
}

/// One recalled memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallHit {
    /// Memory id.
    pub id: String,
    /// Text.
    pub text: String,
    /// Fused score (higher is better).
    pub score: f64,
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
    /// Schema version.
    pub version: i64,
}

// ------------------------------------------------------------------ entity hook

/// Link `names` to memory `mem_id`: resolve each to an entity of `ns` and
/// record a mention. **The single entity-resolution hook** of the memory API:
/// for now an exact match on the normalised name ([`norm_name`]), creating
/// the entity when there is none. Entity resolution proper (E3,
/// `entity.rs`) replaces the body of this function; `text` is passed so it
/// can extract mentions itself.
pub fn link_entities<'s, S: Storage<'s>>(
    db: &'s Db<S>,
    ns: &str,
    mem_id: &str,
    _text: &str,
    names: &[String],
) -> Result<Vec<LinkedEntity>> {
    let mut wanted: BTreeMap<String, String> = BTreeMap::new();
    for n in names {
        let k = norm_name(n);
        if !k.is_empty() {
            wanted.entry(k).or_insert_with(|| n.trim().to_string());
        }
    }
    if wanted.is_empty() {
        return Ok(vec![]);
    }
    let norms: Vec<String> = wanted.keys().cloned().collect();
    let found = db.run_script(
        &format!("?[norm, eid, name] := *{MEM_ENTITY}:by_norm{{ns: $ns, norm, eid}}, norm in $norms, *{MEM_ENTITY}{{ns: $ns, eid, name}}"),
        params(vec![("ns", s(ns)), ("norms", str_list(&norms))]),
        ScriptMutability::Immutable,
    )?;
    let mut have: BTreeMap<String, (String, String)> = BTreeMap::new();
    for r in found.rows {
        if let (Some(n), Some(e), Some(name)) = (cell_str(&r[0]), cell_str(&r[1]), cell_str(&r[2])) {
            have.entry(n).or_insert((e, name));
        }
    }
    let now = now_micros();
    let mut out = vec![];
    let mut new_rows = vec![];
    for (norm, name) in wanted {
        match have.get(&norm) {
            Some((eid, n)) => out.push(LinkedEntity {
                eid: eid.clone(),
                name: n.clone(),
                created: false,
            }),
            None => {
                let eid = new_id("e");
                new_rows.push(DataValue::List(vec![
                    s(ns),
                    s(&eid),
                    s(&name),
                    s(&norm),
                    DataValue::Null,
                    DataValue::from(now),
                ]));
                out.push(LinkedEntity {
                    eid,
                    name,
                    created: true,
                });
            }
        }
    }
    let mentions = out
        .iter()
        .map(|e| DataValue::List(vec![s(ns), s(mem_id), s(&e.eid)]))
        .collect();
    let mut script = String::new();
    if !new_rows.is_empty() {
        script.push_str(&format!(
            "{{?[ns, eid, name, norm, kind, created] <- $ents :put {MEM_ENTITY} {{ns, eid => name, norm, kind, created}}}}"
        ));
    }
    script.push_str(&format!(
        "{{?[ns, id, eid] <- $mentions :put {MEM_MENTION} {{ns, id, eid}}}}"
    ));
    db.run_script(
        &script,
        params(vec![
            ("ents", DataValue::List(new_rows)),
            ("mentions", DataValue::List(mentions)),
        ]),
        ScriptMutability::Mutable,
    )?;
    Ok(out)
}

// ------------------------------------------------------------------ engine

/// A current memory found as a `remember` candidate.
struct Cand {
    id: String,
    text: String,
    from: i64,
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
                Some("version") => cfg.version = v.and_then(|x| x.parse().ok()).unwrap_or(1),
                _ => {}
            }
        }
        Ok(Some(cfg))
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
                ] {
                    self.mem_q(&stmt, vec![])?;
                }
                self.mem_q(
                    &format!("::fts create {MEM}:fts {{extractor: text, tokenizer: Simple, filters: [Lowercase, Stemmer('english'), Stopwords('en')]}}"),
                    vec![],
                )?;
                self.mem_q(&format!("::index create {MEM_ENTITY}:by_norm {{ns, norm}}"), vec![])?;
                self.mem_q(&format!("::index create {MEM_MENTION}:by_entity {{ns, eid}}"), vec![])?;
                // the config relation last: its existence marks a complete schema
                self.mem_q(&format!(":create {MEM_CONFIG} {{key: String => value: String?}}"), vec![])?;
                self.set_mem_config("version", Some("1".to_string()))?;
                MemConfig {
                    version: 1,
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
                    decider: None,
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
    /// the memories valid at `as_of` (default now). Returns
    /// `(id, score, text, kind, source, created, importance)` rows by score.
    #[allow(clippy::too_many_arguments)]
    fn mem_search(
        &'s self,
        cfg: &MemConfig,
        ns: &str,
        text: &str,
        qvec: Option<Vec<f32>>,
        k: usize,
        as_of: Option<i64>,
        kind: Option<&str>,
        entity: Option<&str>,
        vector_weight: f64,
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
        if let Some(e) = entity {
            filters.push_str(&format!(
                ", *{MEM_ENTITY}:by_norm{{ns: $ns, norm: $enorm, eid}}, *{MEM_MENTION}{{ns: $ns, id, eid}}"
            ));
            p.push(("enorm", s(&norm_name(e))));
        }
        let script = format!(
            "{legs}\n\
             f[id, score] <~ WeightedFusion(l[lid, id, score], weights: [{w}], norms: [{n}])\n\
             ?[id, score, text, kind, source, created, importance] := f[id, score], \
               *{MEM}{{ns: $ns, id, text, kind, source, created, importance @ {at}}}{filters}\n\
             :order -score\n:limit {k}",
            legs = legs.join("\n"),
            w = weights.join(", "),
            n = norms.join(", "),
        );
        Ok(self
            .run_script(&script, params(p), ScriptMutability::Immutable)?
            .rows)
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

    /// Store `text` as a memory. Finds the top-k similar current memories of
    /// the namespace, decides `supersedes` and `duplicate` for all of them in
    /// one batch each, and acts on the band: `auto` supersedes (ends the old
    /// memory's validity) or skips a duplicate (recording the statement as
    /// another source of the existing memory); `tentative` inserts and
    /// records a pending decision; `none` inserts. Never deletes. Without a
    /// decider it uses the exact-text and value rules only; without an
    /// embedder, FTS candidates only.
    pub fn remember(&'s self, text: &str, opts: RememberOpts) -> Result<RememberResult> {
        let text = text.trim();
        if text.is_empty() {
            bail!("`remember` needs a non-empty text");
        }
        let cfg = self.mem_ready(opts.embedder.as_deref())?;
        let ns = ns_of(&opts.ns);
        let at = opts.at.unwrap_or_else(now_micros);
        let stmt = new_id("m");
        let vec = self.embed_if_on(cfg.embedder.as_deref(), text)?;

        // candidates: current memories of the namespace, hybrid-ranked
        let rows = self.mem_search(
            &cfg,
            &ns,
            text,
            vec.clone(),
            opts.k.unwrap_or(DEFAULT_K).max(1),
            None,
            None,
            None,
            DEFAULT_VECTOR_WEIGHT,
        )?;
        let ids: Vec<String> = rows.iter().filter_map(|r| cell_str(&r[0])).collect();
        let iv = intervals(&self.mem_history(&ns, &ids)?, i64::MAX);
        let cands: Vec<Cand> = rows
            .iter()
            .filter_map(|r| {
                let id = cell_str(&r[0])?;
                Some(Cand {
                    from: iv.get(&id).map(|x| x.0).unwrap_or(0),
                    id,
                    text: cell_str(&r[2])?,
                })
            })
            .collect();

        // decisions
        let decider = self.pick_decider(opts.decider.as_deref(), &cfg);
        let key = norm_text(text);
        let mut sup: Vec<Option<Decision>> = vec![None; cands.len()];
        let mut dup: Vec<Option<Decision>> = vec![None; cands.len()];
        let mut todo = vec![];
        for (i, c) in cands.iter().enumerate() {
            if norm_text(&c.text) == key {
                dup[i] = Some(rule_decision(
                    Template::Duplicate,
                    "kleos",
                    "exact",
                    "same words after normalisation".to_string(),
                ));
            } else {
                todo.push(i);
            }
        }
        if !todo.is_empty() {
            match &decider {
                Some(name) => {
                    let pairs: Vec<(&str, &str)> =
                        todo.iter().map(|&i| (cands[i].text.as_str(), text)).collect();
                    let s_ds = decide::decide(name, Template::Supersedes, &pairs)?;
                    let d_ds = decide::decide(name, Template::Duplicate, &pairs)?;
                    for ((&i, a), b) in todo.iter().zip(s_ds).zip(d_ds) {
                        sup[i] = Some(a);
                        dup[i] = Some(b);
                    }
                }
                None => {
                    // model-free: the value pre-check alone (decide::rules)
                    for &i in &todo {
                        if let rules::ValueRule::Update(r) = rules::value_rule(&cands[i].text, text) {
                            sup[i] = Some(rule_decision(
                                Template::Supersedes,
                                "rules",
                                "rule",
                                format!("value rule: {r}"),
                            ));
                        }
                    }
                }
            }
        }

        // actions
        let mut res = RememberResult {
            ns: ns.clone(),
            decider: decider.clone(),
            embedded: vec.is_some(),
            ..Default::default()
        };
        let mut supersede = vec![];
        let mut duplicate: Option<(usize, f64)> = None;
        for (i, c) in cands.iter().enumerate() {
            let s_auto = sup[i].as_ref().is_some_and(|d| is_yes(d, Band::Auto));
            let d_auto = dup[i].as_ref().is_some_and(|d| is_yes(d, Band::Auto));
            if d_auto && !s_auto {
                let p = dup[i].as_ref().unwrap().p;
                if duplicate.map_or(true, |(_, q)| p > q) {
                    duplicate = Some((i, p));
                }
            } else if s_auto && !d_auto && c.from <= at {
                supersede.push(i);
            }
        }
        let mut audit = vec![];
        let mut push = |res: &mut RememberResult, i: usize, d: &Decision, action: &str| {
            let md = mem_decision(&cands[i], d, action);
            if action != "none" {
                audit.push(md.clone());
            }
            if action == "pending" {
                res.tentative.push(md.clone());
            }
            res.decisions.push(md);
        };
        for (i, _) in cands.iter().enumerate() {
            for d in [&sup[i], &dup[i]].into_iter().flatten() {
                let applied = (duplicate.map(|x| x.0) == Some(i) && d.template == "duplicate")
                    || (duplicate.is_none() && supersede.contains(&i) && d.template == "supersedes");
                let action = if applied {
                    if d.template == "duplicate" {
                        "duplicate"
                    } else {
                        "superseded"
                    }
                } else if d.label == "yes" && d.band != Band::None {
                    // tentative, an auto answer that conflicts with the
                    // other template, a late arrival, or an auto supersede
                    // left undone because the statement is a duplicate
                    "pending"
                } else {
                    "none"
                };
                push(&mut res, i, d, action);
            }
        }

        let mem_id = match duplicate {
            Some((i, _)) => cands[i].id.clone(),
            None => stmt.clone(),
        };
        let now = now_micros();
        let mut script = String::new();
        let mut p = vec![
            ("ns", s(&ns)),
            ("id", s(&mem_id)),
            ("stmt", s(&stmt)),
            ("at", DataValue::from(at)),
            ("now", DataValue::from(now)),
            ("text", s(text)),
            ("source", opt_s(&opts.source)),
        ];
        if duplicate.is_none() {
            script.push_str(&format!(
                "{{?[ns, id, vt, text, kind, source, created, importance] <- [[$ns, $id, [$at, true], $text, $kind, $source, $now, $importance]] \
                  :put {MEM} {{ns, id, vt => text, kind, source, created, importance}}}}"
            ));
            p.push(("kind", s(opts.kind.as_deref().unwrap_or(DEFAULT_KIND))));
            p.push((
                "importance",
                opts.importance.map(DataValue::from).unwrap_or(DataValue::Null),
            ));
            if let (Some(v), Some(_)) = (&vec, cfg.dim) {
                script.push_str(&format!(
                    "{{?[ns, id, current, vec] <- [[$ns, $id, true, $vec]] :put {MEM_VEC} {{ns, id => current, vec}}}}"
                ));
                p.push((
                    "vec",
                    DataValue::List(v.iter().map(|x| DataValue::from(*x as f64)).collect()),
                ));
            }
            if !supersede.is_empty() {
                let old: Vec<String> = supersede.iter().map(|&i| cands[i].id.clone()).collect();
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
                let links: Vec<DataValue> = supersede
                    .iter()
                    .map(|&i| {
                        let d = sup[i].as_ref().unwrap();
                        DataValue::List(vec![
                            s(&ns),
                            s(&mem_id),
                            s("supersedes"),
                            s(&cands[i].id),
                            DataValue::from(now),
                            DataValue::from(d.p),
                            s(&d.decided_by()),
                        ])
                    })
                    .collect();
                script.push_str(&format!(
                    "{{?[ns, src, rel, dst, created, p, decided_by] <- $links \
                      :put {MEM_LINK} {{ns, src, rel, dst => created, p, decided_by}}}}"
                ));
                p.push(("old", str_list(&old)));
                p.push(("links", DataValue::List(links)));
                res.superseded = old;
            }
        } else {
            res.duplicate_of = Some(mem_id.clone());
        }
        // every statement is a source row of the memory it ended up in
        script.push_str(&format!(
            "{{?[ns, id, at, stmt, source, text] <- [[$ns, $id, $at, $stmt, $source, $text]] \
              :put {MEM_SOURCE} {{ns, id, at, stmt => source, text}}}}"
        ));
        if !audit.is_empty() {
            let new_of = mem_id.clone();
            let rows: Vec<DataValue> = audit
                .iter()
                .map(|d| {
                    DataValue::List(vec![
                        s(&ns),
                        s(&new_of),
                        s(&d.candidate),
                        s(&d.template),
                        s(&d.label),
                        DataValue::from(d.p),
                        s(&d.band),
                        s(&d.decided_by),
                        opt_s(&d.reason),
                        s(if d.action == "pending" { "pending" } else { "applied" }),
                        DataValue::from(now),
                    ])
                })
                .collect();
            script.push_str(&format!(
                "{{?[ns, new, cand, template, label, p, band, decided_by, reason, status, at] <- $audit \
                  :put {MEM_DECISION} {{ns, new, cand, template => label, p, band, decided_by, reason, status, at}}}}"
            ));
            p.push(("audit", DataValue::List(rows)));
        }
        self.mem_q(&script, p)?;

        if !opts.entities.is_empty() {
            res.entities = link_entities(self, &ns, &mem_id, text, &opts.entities)?
                .into_iter()
                .map(|e| e.eid)
                .collect();
        }
        res.action = if duplicate.is_some() { "duplicate" } else { "added" }.to_string();
        res.id = if duplicate.is_some() { None } else { Some(stmt) };
        Ok(res)
    }

    /// Hybrid recall (vector + FTS, `WeightedFusion`) over the memories valid
    /// now, or at `as_of`. Filters: namespace, kind, entity.
    pub fn recall(&'s self, query: &str, opts: RecallOpts) -> Result<Vec<RecallHit>> {
        let cfg = match self.mem_config()? {
            Some(c) => c,
            None => return Ok(vec![]),
        };
        let ns = ns_of(&opts.ns);
        let qe = cfg.embedder.as_ref().map(|e| {
            let q = format!("{e}_query");
            if embed::is_embedder_enabled(&q) {
                q
            } else {
                e.clone()
            }
        });
        let qvec = self.embed_if_on(qe.as_deref(), query)?;
        let w = opts.vector_weight.unwrap_or(DEFAULT_VECTOR_WEIGHT);
        if !(0.0..=1.0).contains(&w) {
            bail!("`recall` option vector_weight must be in [0, 1], got {w}");
        }
        let rows = self.mem_search(
            &cfg,
            &ns,
            query,
            qvec,
            opts.k.unwrap_or(10).max(1),
            opts.as_of,
            opts.kind.as_deref(),
            opts.entity.as_deref(),
            w,
        )?;
        let ids: Vec<String> = rows.iter().filter_map(|r| cell_str(&r[0])).collect();
        let iv = intervals(&self.mem_history(&ns, &ids)?, opts.as_of.unwrap_or_else(now_micros));
        Ok(rows
            .into_iter()
            .filter_map(|r| {
                let id = cell_str(&r[0])?;
                let (from, to) = iv.get(&id).copied().unwrap_or((0, None));
                Some(RecallHit {
                    score: r[1].get_float().unwrap_or(0.0),
                    text: cell_str(&r[2])?,
                    kind: cell_str(&r[3]).unwrap_or_default(),
                    source: cell_str(&r[4]),
                    created: r[5].get_int().unwrap_or(0),
                    importance: r[6].get_float(),
                    valid_from: from,
                    valid_to: to,
                    id,
                })
            })
            .collect())
    }

    /// Forget memories by id or filter. `retract` (default) ends their
    /// validity at `at` (history, `as_of` recall and `::history` keep them);
    /// `erase` deletes every version through `::erase` (search-index entries
    /// included), with their vectors, sources, mentions, links and decision
    /// rows, and entities left with no mention.
    pub fn forget(&'s self, opts: ForgetOpts) -> Result<ForgetResult> {
        let mode = opts.mode.clone().unwrap_or_else(|| "retract".to_string());
        if mode != "retract" && mode != "erase" {
            bail!("`forget` mode must be 'retract' or 'erase', got '{mode}'");
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
        let mut ids: Vec<String> = opts.ids.clone();
        ids.extend(opts.id.clone());
        let by_filter = opts.kind.is_some() || opts.entity.is_some();
        if ids.is_empty() && !by_filter && !opts.all {
            bail!("`forget` needs an id, ids, a kind or entity filter, or all: true");
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
                body.push_str(&format!(
                    ", *{MEM_ENTITY}:by_norm{{ns: $ns, norm: $enorm, eid}}, *{MEM_MENTION}{{ns: $ns, id, eid}}"
                ));
                p.push(("enorm", s(&norm_name(e))));
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
            self.mem_q(
                &script,
                vec![("ns", s(&ns)), ("ids", str_list(&cur)), ("at", DataValue::from(at))],
            )?;
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
        // ::erase deletes every row of each key with its FTS/HNSW entries and
        // writes the salted audit marker (kleos_evict_audit)
        self.mem_q(&format!("::erase {MEM} $keys"), vec![("keys", keys)])?;
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
               :rm {MEM_LINK} {{ns, src, rel, dst}}}}"
        ));
        self.mem_q(
            &script,
            vec![("ns", s(&ns)), ("ids", str_list(&exist)), ("eids", str_list(&eids))],
        )?;
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
        let mut groups: BTreeMap<String, Vec<(i64, String, String)>> = BTreeMap::new();
        for r in shared.rows {
            if let (Some(e), Some(id), Some(t)) = (cell_str(&r[0]), cell_str(&r[2]), cell_str(&r[3])) {
                groups.entry(e).or_default().push((r[1].get_int().unwrap_or(0), id, t));
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
                serde_json::to_value(self.recall(&q, from_args(op, args)?)?).into_diagnostic()?
            }
            "forget" => serde_json::to_value(self.forget(from_args(op, args)?)?).into_diagnostic()?,
            "reflect" => serde_json::to_value(self.reflect(from_args(op, args)?)?).into_diagnostic()?,
            "link" => self.link(from_args(op, args)?)?,
            other => bail!(
                "unknown memory op '{other}': expected init, remember, recall, forget, reflect or link"
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
