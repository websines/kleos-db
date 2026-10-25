/*
 * Copyright 2026, The kleos authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Consolidation (docs/specs/consolidation.md; UPGRADE_PLAN E7): fold a
//! namespace's episodes into derived knowledge, without an LLM by default.
//!
//! - **Entity profiles** (`kleos_mem_profile`): per ontology node, its
//!   current facts grouped by property, each with its valid-from and the
//!   memories behind it. Shown in the context block ("Profile: Anna —
//!   lives_in: …").
//! - **Session summaries**: per source group (a conversation, a session),
//!   the most central and least redundant claims (embedding centrality plus
//!   MMR), stored as one memory of kind `summary` and role `other`, linked
//!   to its sources (`kleos_mem_summary`). With an LLM configured, an
//!   abstractive summary of those claims replaces the extractive text.
//!
//! Raw memories stay the source of truth: consolidated items never
//! supersede them (role `other` keeps summaries out of every decision), are
//! rebuilt when their sources change, and are erased with them.
//! `::memory consolidate` runs it; `every`/`cron` make it a durable `::flow`
//! workflow (`memory.consolidate`), started after N new memories or on a
//! schedule.

use std::hash::{Hash, Hasher};

use super::*;

pub const MEM_PROFILE: &str = "kleos_mem_profile";
pub const MEM_SUMMARY: &str = "kleos_mem_summary";
pub const MEM_CONSOL: &str = "kleos_mem_consol";
/// New memories of a namespace between automatic consolidations when the
/// namespace has no setting (`::memory consolidate {every: 0}` turns it off).
pub const DEFAULT_CONSOLIDATE_EVERY: i64 = 200;
/// Flow queue of the automatic consolidation runs (run in-process by
/// [`crate::DbInstance::consolidate_kick`] or by any flow worker).
pub const CONSOLIDATE_QUEUE: &str = "consolidate";

/// The in-process consolidation runner's state.
#[derive(Default)]
pub struct ConsolState {
    /// a run was started and waits for the in-process runner
    pub(crate) pending: std::sync::atomic::AtomicBool,
    pub(crate) running: std::sync::atomic::AtomicBool,
    /// ns → (every, new memories since the last trigger); `every` read once
    /// per namespace and open store.
    pub(crate) auto: Mutex<HashMap<String, (i64, i64)>>,
}
/// Kind of a summary memory.
pub const SUMMARY_KIND: &str = "summary";
/// The built-in workflow that consolidates one namespace (`input.ns`).
pub const CONSOLIDATE_WORKFLOW: &str = "memory.consolidate";

const DEFAULT_MIN_CLAIMS: usize = 8;
const DEFAULT_MAX_SUMMARY: usize = 6;
const DEFAULT_MIN_FACTS: usize = 2;
const MMR_LAMBDA: f64 = 0.7;
const PROFILE_VALUE_CHARS: usize = 90;
/// An older single-valued claim this similar to a newer one of the same
/// subject and property is a replaced value (left out of summaries).
const REPLACED_SIM: f64 = 0.6;

/// Options of `::memory consolidate`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConsolidateOpts {
    pub ns: Option<String>,
    /// Build entity profiles (default true).
    pub profiles: Option<bool>,
    /// Build session summaries (default true).
    pub summaries: Option<bool>,
    /// A source group needs at least this many claims for a summary (default 8).
    pub min_claims: Option<usize>,
    /// Claims in a summary, at most (default 6).
    pub max_summary: Option<usize>,
    /// A node needs at least this many facts for a profile (default 2).
    pub min_facts: Option<usize>,
    /// Abstractive summaries from an OpenAI-compatible endpoint (off by default).
    pub llm: Option<LlmOpts>,
    /// Run automatically: after this many new memories in the namespace
    /// (0 turns it off) …
    pub every: Option<i64>,
    /// … and/or on this schedule (`::flow` cron syntax, e.g. `@every 1h`).
    pub cron: Option<String>,
}

/// An OpenAI-compatible chat endpoint for abstractive summaries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlmOpts {
    /// Base URL (`…/v1`) or the full `…/chat/completions` URL.
    pub url: String,
    pub model: String,
    /// Environment variable holding the API key (never stored).
    pub api_key_env: Option<String>,
    pub max_tokens: Option<u64>,
}

/// One entry of a profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileEntry {
    pub value: String,
    pub since: Option<i64>,
    pub evidence: Vec<String>,
}

fn hash_of<T: Hash>(x: &T) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    x.hash(&mut h);
    format!("{:016x}", h.finish())
}

fn title(node: &str) -> String {
    if node == "user" {
        return "User".to_string();
    }
    node.replace('.', "'s ")
        .split_whitespace()
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn short(s: &str, n: usize) -> String {
    let t = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if t.chars().count() <= n {
        t
    } else {
        format!("{}…", t.chars().take(n).collect::<String>().trim_end())
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    if na <= 0.0 || nb <= 0.0 {
        0.0
    } else {
        d / (na.sqrt() * nb.sqrt())
    }
}

/// Extractive selection: centrality (cosine to the centroid, else word
/// overlap with the group) traded against redundancy (MMR).
pub fn select_central(texts: &[String], vecs: &[Option<Vec<f32>>], m: usize) -> Vec<usize> {
    let n = texts.len();
    if n == 0 || m == 0 {
        return vec![];
    }
    let have_vec = vecs.iter().all(|v| v.is_some()) && !vecs.is_empty();
    let words: Vec<BTreeSet<String>> = texts.iter().map(|t| memory_recall::word_set(t)).collect();
    let sim = |i: usize, j: usize| -> f64 {
        if have_vec {
            cosine(vecs[i].as_ref().unwrap(), vecs[j].as_ref().unwrap())
        } else {
            let (a, b) = (&words[i], &words[j]);
            let inter = a.intersection(b).count() as f64;
            let uni = a.union(b).count() as f64;
            if uni == 0.0 {
                0.0
            } else {
                inter / uni
            }
        }
    };
    let central: Vec<f64> = if have_vec {
        let d = vecs[0].as_ref().unwrap().len();
        let mut c = vec![0.0f32; d];
        for v in vecs.iter().flatten() {
            for (k, x) in v.iter().enumerate() {
                c[k] += x;
            }
        }
        vecs.iter().map(|v| cosine(v.as_ref().unwrap(), &c)).collect()
    } else {
        (0..n).map(|i| (0..n).filter(|&j| j != i).map(|j| sim(i, j)).sum::<f64>() / (n.max(2) - 1) as f64).collect()
    };
    let mut picked: Vec<usize> = vec![];
    while picked.len() < m.min(n) {
        let best = (0..n)
            .filter(|i| !picked.contains(i))
            .map(|i| {
                let red = picked.iter().map(|&j| sim(i, j)).fold(0.0, f64::max);
                (i, MMR_LAMBDA * central[i] - (1.0 - MMR_LAMBDA) * red)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then(b.0.cmp(&a.0)));
        match best {
            Some((i, _)) => picked.push(i),
            None => break,
        }
    }
    picked
}

impl<'s, S: Storage<'s>> Db<S> {
    pub(crate) fn consolidate_ensure(&'s self) -> Result<()> {
        let tx = self.transact()?;
        let have = |n: &str| tx.get_relation(n, false).is_ok();
        let mut todo = vec![];
        if !have(MEM_PROFILE) {
            todo.push(format!(
                "{{:create {MEM_PROFILE} {{ns: String, node: String => name: String, text: String, facts: Json, evidence: [String], sig: String, updated: Int}}}}"
            ));
        }
        if !have(MEM_SUMMARY) {
            todo.push(format!(
                "{{:create {MEM_SUMMARY} {{ns: String, sid: String => grp: String, sources: [String], method: String, sig: String, at: Int}}}}"
            ));
        }
        if !have(MEM_CONSOL) {
            todo.push(format!("{{:create {MEM_CONSOL} {{ns: String => every: Int, cron: String?, last: Int}}}}"));
        }
        drop(tx);
        if !todo.is_empty() {
            self.mem_q(&todo.join(""), vec![])?;
        }
        Ok(())
    }

    fn consol_exists(&'s self, name: &str) -> bool {
        self.transact().map(|tx| tx.get_relation(name, false).is_ok()).unwrap_or(false)
    }

    /// `::memory consolidate`.
    pub fn memory_consolidate(&'s self, o: ConsolidateOpts) -> Result<JsonValue> {
        let t0 = Instant::now();
        let ns = ns_of(&o.ns);
        let Some(cfg) = self.mem_config()? else {
            bail!("memory consolidate: call `::memory init` first");
        };
        self.consolidate_ensure()?;
        let mut out = json_obj();
        out.insert("ns".into(), JsonValue::String(ns.clone()));
        // auto runs: config, the workflow and its schedule
        if o.every.is_some() || o.cron.is_some() {
            out.insert("auto".into(), self.consolidate_auto(&ns, o.every, o.cron.clone())?);
        }
        if o.profiles.unwrap_or(true) {
            out.insert("profiles".into(), self.build_profiles(&ns, o.min_facts.unwrap_or(DEFAULT_MIN_FACTS))?);
        }
        if o.summaries.unwrap_or(true) {
            out.insert(
                "summaries".into(),
                self.build_summaries(
                    &cfg,
                    &ns,
                    o.min_claims.unwrap_or(DEFAULT_MIN_CLAIMS).max(2),
                    o.max_summary.unwrap_or(DEFAULT_MAX_SUMMARY).max(1),
                    o.llm.as_ref(),
                )?,
            );
        }
        out.insert("ms".into(), serde_json::json!(t0.elapsed().as_millis() as u64));
        Ok(JsonValue::Object(out))
    }

    // ------------------------------------------------------------ profiles

    /// Profiles of every node with at least `min_facts` current facts.
    fn build_profiles(&'s self, ns: &str, min_facts: usize) -> Result<JsonValue> {
        let sc = self.mem_scope(ns)?;
        let now = now_micros();
        // node → prop → entries
        let mut by: BTreeMap<String, BTreeMap<String, Vec<ProfileEntry>>> = BTreeMap::new();
        let mut labels: BTreeMap<String, String> = BTreeMap::new();
        let mut single: BTreeSet<String> = BTreeSet::new();
        if let Some(act) = self.onto_active(ns)? {
            for p in &act.props {
                if p.single() {
                    single.insert(p.name.clone());
                }
            }
        }
        // ontology facts (stated ones) and node labels
        let mut fact_mems: Vec<String> = vec![];
        let mut facts: Vec<(String, String, String, Option<String>)> = vec![];
        if self.consol_exists(crate::ontology::ONTO_FACT) {
            for r in self
                .mem_q(
                    &format!(
                        "?[s, p, o, mem] := *{}{{ns: $ns, s, p, o, mem, status @ 'NOW'}}, status = 'current', p != 'same_as'",
                        crate::ontology::ONTO_FACT
                    ),
                    vec![("ns", s(ns))],
                )?
                .rows
            {
                let (Some(fs), Some(fp), Some(fo)) = (cell_str(&r[0]), cell_str(&r[1]), cell_str(&r[2])) else { continue };
                let m = cell_str(&r[3]);
                fact_mems.extend(m.clone());
                facts.push((fs, fp, fo, m));
            }
            for r in self
                .mem_q(
                    &format!("?[node, label] := *{}{{ns: $ns, node, label @ 'NOW'}}, !is_null(label)", crate::ontology::ONTO_NODE),
                    vec![("ns", s(ns))],
                )?
                .rows
            {
                if let (Some(n), Some(l)) = (cell_str(&r[0]), cell_str(&r[1])) {
                    labels.insert(n, l);
                }
            }
        }
        // typed claims: (subject, property) with the claim's text
        let mut claims: Vec<(String, String, String)> = vec![];
        if self.consol_exists(crate::ontology::MEM_TYPE) {
            for r in self
                .mem_q(
                    // cue- or rule-backed types only: a prototype match alone
                    // is too loose to state a fact about someone
                    &format!(
                        "?[id, subject, prop] := *{}{{ns: $ns, id, cls, subject, prop, by}}, cls = 'fact', !is_null(subject), \
                           !is_null(prop), by != 'proto'",
                        crate::ontology::MEM_TYPE
                    ),
                    vec![("ns", s(ns))],
                )?
                .rows
            {
                if let (Some(id), Some(sub), Some(p)) = (cell_str(&r[0]), cell_str(&r[1]), cell_str(&r[2])) {
                    // a leading-name guess ("name:x") counts only for a
                    // known entity or ontology node
                    let guess = sub.starts_with("name:");
                    claims.push((id, graph_node(&sub), p + if guess { "\u{0}" } else { "" }));
                }
            }
            // known names: entities and ontology nodes
            let mut known: BTreeSet<String> = facts.iter().flat_map(|f| [f.0.clone(), f.2.clone()]).collect();
            if let Ok(view) = self.entity_view(&sc) {
                known.extend(view.rows.values().map(|(n, _)| graph_node(n)));
            }
            claims.retain_mut(|(_, node, p)| {
                if let Some(base) = p.strip_suffix('\u{0}') {
                    let ok = known.contains(node.as_str()) && !NOT_NAMES.contains(&node.as_str());
                    *p = base.to_string();
                    ok
                } else {
                    !NOT_NAMES.contains(&node.as_str())
                }
            });
        }
        let mut ids: Vec<String> = claims.iter().map(|c| c.0.clone()).collect();
        ids.extend(fact_mems);
        ids.sort();
        ids.dedup();
        // current texts and valid-from of those memories
        let mut text: BTreeMap<String, String> = BTreeMap::new();
        if !ids.is_empty() {
            for r in self
                .mem_q(
                    &format!("?[id, text] := id in $ids, *{MEM}{{ns: $ns, id, text @ 'NOW'}}"),
                    vec![("ns", s(ns)), ("ids", str_list(&ids))],
                )?
                .rows
            {
                if let (Some(id), Some(t)) = (cell_str(&r[0]), cell_str(&r[1])) {
                    text.insert(id, t);
                }
            }
        }
        let iv = intervals(&self.mem_history(&sc, &ids)?, now);
        let since = |id: &Option<String>| id.as_ref().and_then(|i| iv.get(i)).map(|x| x.0);
        for (fs, fp, fo, m) in &facts {
            if m.as_ref().is_some_and(|i| !text.contains_key(i)) {
                continue; // its memory is no longer current
            }
            let ev: Vec<String> = m.iter().cloned().collect();
            by.entry(fs.clone()).or_default().entry(fp.clone()).or_default().push(ProfileEntry {
                value: labels.get(fo).cloned().unwrap_or_else(|| title(fo)),
                since: since(m),
                evidence: ev.clone(),
            });
            by.entry(fo.clone()).or_default().entry(format!("← {fp}")).or_default().push(ProfileEntry {
                value: labels.get(fs).cloned().unwrap_or_else(|| title(fs)),
                since: since(m),
                evidence: ev,
            });
        }
        for (id, node, p) in &claims {
            let Some(t) = text.get(id) else { continue };
            let body = t.split_once(": ").filter(|(a, _)| a.split_whitespace().count() <= 3).map_or(t.as_str(), |x| x.1);
            by.entry(node.clone()).or_default().entry(p.clone()).or_default().push(ProfileEntry {
                value: short(body, PROFILE_VALUE_CHARS),
                since: since(&Some(id.clone())),
                evidence: vec![id.clone()],
            });
        }
        // single-valued properties: the newest value only
        for props in by.values_mut() {
            for (p, v) in props.iter_mut() {
                v.sort_by(|a, b| b.since.cmp(&a.since).then(a.value.cmp(&b.value)));
                v.dedup_by(|a, b| a.value == b.value);
                if single.contains(p) {
                    v.truncate(1);
                }
            }
        }
        // stored profiles
        let mut stored: BTreeMap<String, String> = BTreeMap::new();
        for r in self
            .mem_q(&format!("?[node, sig] := *{MEM_PROFILE}{{ns: $ns, node, sig}}"), vec![("ns", s(ns))])?
            .rows
        {
            if let (Some(n), Some(sg)) = (cell_str(&r[0]), cell_str(&r[1])) {
                stored.insert(n, sg);
            }
        }
        let mut puts = vec![];
        let mut keep = BTreeSet::new();
        let (mut built, mut same) = (0, 0);
        for (node, props) in &by {
            let n: usize = props.values().map(|v| v.len()).sum();
            if n < min_facts || node.is_empty() {
                continue;
            }
            keep.insert(node.clone());
            let name = labels.get(node).cloned().unwrap_or_else(|| title(node));
            let parts: Vec<String> = props
                .iter()
                .map(|(p, v)| {
                    let vals: Vec<String> = v
                        .iter()
                        .take(3)
                        .map(|e| match e.since {
                            Some(t) => format!("{} (since {})", e.value, memory_recall::fmt_date(t)),
                            None => e.value.clone(),
                        })
                        .collect();
                    format!("{p}: {}", vals.join(" | "))
                })
                .collect();
            let text = format!("Profile: {name} — {}", parts.join("; "));
            let sig = hash_of(&text);
            if stored.get(node) == Some(&sig) {
                same += 1;
                continue;
            }
            built += 1;
            let evidence: Vec<DataValue> =
                props.values().flatten().flat_map(|e| e.evidence.iter()).collect::<BTreeSet<_>>().into_iter().map(|x| s(x)).collect();
            puts.push(DataValue::List(vec![
                s(ns),
                s(node),
                s(&name),
                s(&text),
                DataValue::Json(crate::JsonData(serde_json::to_value(props).into_diagnostic()?)),
                DataValue::List(evidence),
                s(&sig),
                DataValue::from(now),
            ]));
        }
        let gone: Vec<String> = stored.keys().filter(|n| !keep.contains(*n)).cloned().collect();
        let mut script = String::new();
        let mut p = vec![("ns", s(ns))];
        if !puts.is_empty() {
            script.push_str(&format!(
                "{{?[ns, node, name, text, facts, evidence, sig, updated] <- $rows \
                  :put {MEM_PROFILE} {{ns, node => name, text, facts, evidence, sig, updated}}}}"
            ));
            p.push(("rows", DataValue::List(puts)));
        }
        if !gone.is_empty() {
            script.push_str(&format!(
                "{{?[ns, node] := *{MEM_PROFILE}{{ns, node}}, ns = $ns, node in $gone :rm {MEM_PROFILE} {{ns, node}}}}"
            ));
            p.push(("gone", str_list(&gone)));
        }
        if !script.is_empty() {
            self.mem_q(&script, p)?;
        }
        Ok(serde_json::json!({"built": built, "unchanged": same, "removed": gone.len(), "nodes": keep.len()}))
    }

    /// Profiles of `nodes` (`(node, text)`), for the context block.
    pub(crate) fn profiles_of(&'s self, ns: &str, nodes: &BTreeSet<String>) -> Result<Vec<(String, String)>> {
        if nodes.is_empty() || !self.consol_exists(MEM_PROFILE) {
            return Ok(vec![]);
        }
        let nl: Vec<String> = nodes.iter().cloned().collect();
        Ok(self
            .mem_q(
                &format!("?[node, text] := node in $nodes, *{MEM_PROFILE}{{ns: $ns, node, text}}"),
                vec![("ns", s(ns)), ("nodes", str_list(&nl))],
            )?
            .rows
            .iter()
            .filter_map(|r| Some((cell_str(&r[0])?, cell_str(&r[1])?)))
            .collect())
    }

    /// Current summary texts of `groups`, in that order.
    pub(crate) fn summaries_of(&'s self, ns: &str, groups: &[String]) -> Result<Vec<String>> {
        if groups.is_empty() || !self.consol_exists(MEM_SUMMARY) {
            return Ok(vec![]);
        }
        let mut by: BTreeMap<String, String> = BTreeMap::new();
        for r in self
            .mem_q(
                &format!(
                    "?[grp, text] := grp in $groups, *{MEM_SUMMARY}{{ns: $ns, sid, grp}}, *{MEM}{{ns: $ns, id: sid, text @ 'NOW'}}"
                ),
                vec![("ns", s(ns)), ("groups", str_list(groups))],
            )?
            .rows
        {
            if let (Some(g), Some(t)) = (cell_str(&r[0]), cell_str(&r[1])) {
                by.insert(g, t);
            }
        }
        Ok(groups.iter().filter_map(|g| by.get(g).cloned()).collect())
    }

    // ----------------------------------------------------------- summaries

    fn build_summaries(
        &'s self,
        cfg: &MemConfig,
        ns: &str,
        min_claims: usize,
        max_summary: usize,
        llm: Option<&LlmOpts>,
    ) -> Result<JsonValue> {
        let sc = self.mem_scope(ns)?;
        let now = now_micros();
        // current memories by source group (no documents, summaries or
        // other speakers' material)
        let mut rows: Vec<(String, String, String)> = vec![];
        for r in self
            .mem_q(
                &format!(
                    "?[id, text, source] := *{MEM}{{ns: $ns, id, text, kind, source @ 'NOW'}}, !is_null(source), \
                       kind != '{DOC_KIND}', kind != '{SUMMARY_KIND}', not *{MEM_ROLE}{{ns: $ns, id, role: 'other'}}"
                ),
                vec![("ns", s(ns))],
            )?
            .rows
        {
            if let (Some(id), Some(t), Some(src)) = (cell_str(&r[0]), cell_str(&r[1]), cell_str(&r[2])) {
                rows.push((id, t, memory_recall::source_group(&src).to_string()));
            }
        }
        // replaced values never reach a summary: a claim a newer one may
        // supersede (pending), or an older value of a single-valued property
        // whose newer value (same subject) says nearly the same thing
        let replaced = self.replaced_claims(&sc, ns, &rows.iter().map(|r| r.0.clone()).collect::<Vec<_>>(), now, cfg.dim.is_some())?;
        let mut groups: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
        let mut dropped = 0usize;
        for (id, t, g) in rows {
            if replaced.contains(&id) {
                dropped += 1;
                continue;
            }
            if memory_claims::is_declarative(&t) && t.split_whitespace().count() >= 4 {
                groups.entry(g).or_default().push((id, t));
            }
        }
        let mut stored: BTreeMap<String, (String, String)> = BTreeMap::new();
        for r in self
            .mem_q(&format!("?[sid, grp, sig] := *{MEM_SUMMARY}{{ns: $ns, sid, grp, sig}}"), vec![("ns", s(ns))])?
            .rows
        {
            if let (Some(sid), Some(g), Some(sg)) = (cell_str(&r[0]), cell_str(&r[1]), cell_str(&r[2])) {
                stored.insert(g, (sid, sg));
            }
        }
        let method = llm.map_or_else(|| "extractive".to_string(), |l| format!("llm:{}", l.model));
        let (mut built, mut same, mut llm_calls, mut llm_errors) = (0usize, 0usize, 0usize, 0usize);
        let mut keep: BTreeSet<String> = BTreeSet::new();
        let mut scripts: Vec<(String, Vec<(&str, DataValue)>)> = vec![];
        let mut src_tokens = 0usize;
        let mut sum_tokens = 0usize;
        for (g, claims) in &groups {
            if claims.len() < min_claims {
                continue;
            }
            keep.insert(g.clone());
            let ids: Vec<String> = claims.iter().map(|c| c.0.clone()).collect();
            let sig = hash_of(&(&ids, &method, max_summary));
            if stored.get(g).is_some_and(|x| x.1 == sig) {
                same += 1;
                continue;
            }
            let texts: Vec<String> = claims.iter().map(|c| c.1.clone()).collect();
            let vm = if cfg.dim.is_some() { self.vecs_of(&sc, &ids)? } else { BTreeMap::new() };
            let vecs: Vec<Option<Vec<f32>>> = ids.iter().map(|i| vm.get(i).cloned()).collect();
            let m = max_summary.min(((claims.len() + 4) / 5).max(2));
            let iv = intervals(&self.mem_history(&sc, &ids)?, now);
            let mut pick = select_central(&texts, &vecs, m);
            pick.sort_by_key(|&i| (iv.get(&ids[i]).map_or(0, |x| x.0), i));
            let from = pick.iter().filter_map(|&i| iv.get(&ids[i]).map(|x| x.0)).min().unwrap_or(now);
            let to = ids.iter().filter_map(|i| iv.get(i).map(|x| x.0)).max().unwrap_or(now);
            let chosen: Vec<String> = pick.iter().map(|&i| texts[i].trim().trim_end_matches('.').to_string() + ".").collect();
            let mut body = chosen.join(" ");
            if let Some(l) = llm {
                llm_calls += 1;
                match llm_summary(l, &chosen) {
                    Ok(t) if !t.trim().is_empty() => body = t.trim().to_string(),
                    _ => llm_errors += 1,
                }
            }
            let range = if memory_recall::fmt_date(from) == memory_recall::fmt_date(to) {
                memory_recall::fmt_date(from)
            } else {
                format!("{} – {}", memory_recall::fmt_date(from), memory_recall::fmt_date(to))
            };
            let text = format!("Summary of {g} ({range}): {body}");
            src_tokens += texts.iter().map(|t| memory_recall::estimate_tokens(t)).sum::<usize>();
            sum_tokens += memory_recall::estimate_tokens(&text);
            let sid = format!("sum_{}", hash_of(&(ns, g)));
            let vec = match cfg.embedder.as_deref() {
                Some(e) => self.embed_if_on(Some(e), &text)?,
                None => None,
            };
            let mut script = format!(
                "{{?[ns, id, vt, text, kind, source, created, importance] <- [[$ns, $sid, [$at, true], $text, '{SUMMARY_KIND}', $src, $now, null]] \
                   :put {MEM} {{ns, id, vt => text, kind, source, created, importance}}}}\
                 {{?[ns, id, role] <- [[$ns, $sid, 'other']] :put {MEM_ROLE} {{ns, id => role}}}}\
                 {{?[ns, sid, grp, sources, method, sig, at] <- [[$ns, $sid, $grp, $ids, $method, $sig, $now]] \
                   :put {MEM_SUMMARY} {{ns, sid => grp, sources, method, sig, at}}}}"
            );
            let mut p = vec![
                ("ns", s(ns)),
                ("sid", s(&sid)),
                ("at", DataValue::from(to.max(from))),
                ("text", s(&text)),
                ("src", s(&format!("summary:{g}"))),
                ("now", DataValue::from(now)),
                ("grp", s(g)),
                ("ids", str_list(&ids)),
                ("method", s(&method)),
                ("sig", s(&sig)),
            ];
            if let Some(v) = vec {
                script.push_str(&format!(
                    "{{?[ns, id, current, vec] <- [[$ns, $sid, true, $vec]] :put {MEM_VEC} {{ns, id => current, vec}}}}"
                ));
                p.push(("vec", DataValue::List(v.iter().map(|x| DataValue::from(*x as f64)).collect())));
            }
            script.push_str(&bump_ns_head());
            scripts.push((script, p));
            built += 1;
        }
        // groups that no longer qualify: retract their summary
        let gone: Vec<(String, String)> =
            stored.iter().filter(|(g, _)| !keep.contains(*g)).map(|(g, (sid, _))| (g.clone(), sid.clone())).collect();
        for (script, p) in scripts {
            self.mem_q(&script, p)?;
        }
        if !gone.is_empty() {
            let sids: Vec<String> = gone.iter().map(|x| x.1.clone()).collect();
            self.retract_summaries(ns, &sids, now)?;
        }
        Ok(serde_json::json!({
            "built": built, "unchanged": same, "removed": gone.len(), "groups": keep.len(), "method": method,
            "llm_calls": llm_calls, "llm_errors": llm_errors,
            "source_tokens": src_tokens, "summary_tokens": sum_tokens, "replaced_dropped": dropped,
        }))
    }

    /// Current claims among `ids` that a summary must not quote: the
    /// candidates of a pending supersession, and older values of a
    /// single-valued property when a newer current claim of the same
    /// subject and property is similar to them (cosine at least
    /// [`REPLACED_SIM`]; a store without vectors uses the pending rule only).
    fn replaced_claims(&'s self, sc: &Scope, ns: &str, ids: &[String], now: i64, vectors: bool) -> Result<BTreeSet<String>> {
        let mut out = BTreeSet::new();
        if ids.is_empty() {
            return Ok(out);
        }
        let current: BTreeSet<&String> = ids.iter().collect();
        if self.consol_exists(MEM_DECISION) {
            for r in self
                .mem_q(
                    &format!("?[id] := *{MEM_DECISION}{{ns: $ns, cand: id, template: 'supersedes', status: 'pending'}}"),
                    vec![("ns", s(ns))],
                )?
                .rows
            {
                if let Some(id) = cell_str(&r[0]).filter(|i| current.contains(i)) {
                    out.insert(id);
                }
            }
        }
        // the property rule needs vectors: a shared property alone ("adopted
        // a dog" / "plans to adopt a second dog") is no replacement
        let Some(act) = self.onto_active(ns)? else { return Ok(out) };
        if !vectors || !self.consol_exists(crate::ontology::MEM_TYPE) {
            return Ok(out);
        }
        let mut by: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
        for r in self
            .mem_q(
                &format!(
                    "?[id, subject, prop] := *{}{{ns: $ns, id, cls, subject, prop}}, cls = 'fact', !is_null(subject), !is_null(prop)",
                    crate::ontology::MEM_TYPE
                ),
                vec![("ns", s(ns))],
            )?
            .rows
        {
            if let (Some(id), Some(sub), Some(p)) = (cell_str(&r[0]), cell_str(&r[1]), cell_str(&r[2])) {
                if current.contains(&id) && act.prop(&p).is_some_and(|x| x.single()) {
                    by.entry((sub, p)).or_default().push(id);
                }
            }
        }
        by.retain(|_, v| v.len() >= 2);
        if by.is_empty() {
            return Ok(out);
        }
        let all: Vec<String> = by.values().flatten().cloned().collect();
        let iv = intervals(&self.mem_history(sc, &all)?, now);
        let vm = self.vecs_of(sc, &all)?;
        for v in by.values_mut() {
            v.sort_by_key(|i| (iv.get(i).map_or(0, |x| x.0), i.clone()));
            for (k, old) in v.iter().enumerate() {
                let newer = &v[k + 1..];
                let hit = newer.iter().any(|n| match (vm.get(old), vm.get(n)) {
                    (Some(a), Some(b)) => cosine(a, b) >= REPLACED_SIM,
                    _ => false,
                });
                if hit {
                    out.insert(old.clone());
                }
            }
        }
        Ok(out)
    }

    fn retract_summaries(&'s self, ns: &str, sids: &[String], at: i64) -> Result<()> {
        let mut script = format!(
            "{{?[ns, id, vt, text, kind, source, created, importance] := *{MEM}{{ns, id, text, kind, source, created, importance @ 'NOW'}}, \
               ns = $ns, id in $sids, vt = [$at, false] \
              :put {MEM} {{ns, id, vt => text, kind, source, created, importance}}}}\
             {{?[ns, sid] := *{MEM_SUMMARY}{{ns, sid}}, ns = $ns, sid in $sids :rm {MEM_SUMMARY} {{ns, sid}}}}"
        );
        if self.consol_exists(MEM_VEC) {
            script.push_str(&format!(
                "{{?[ns, id, current] := *{MEM_VEC}{{ns, id}}, ns = $ns, id in $sids, current = false :update {MEM_VEC} {{ns, id => current}}}}"
            ));
        }
        script.push_str(&bump_ns_head());
        self.mem_q(&script, vec![("ns", s(ns)), ("sids", str_list(sids)), ("at", DataValue::from(at))])?;
        Ok(())
    }

    /// GDPR: an erase reaches what was derived from the erased memories —
    /// summaries built from them are erased, profiles citing them dropped
    /// (rebuilt without them by the next run).
    pub(crate) fn consolidated_erase(&'s self, cfg: &MemConfig, ns: &str, erased: &BTreeSet<String>) -> Result<Vec<String>> {
        if erased.is_empty() || !self.consol_exists(MEM_SUMMARY) {
            return Ok(vec![]);
        }
        let ids: Vec<String> = erased.iter().cloned().collect();
        let sids: Vec<String> = self
            .mem_q(
                &format!(
                    "?[sid] := *{MEM_SUMMARY}{{ns: $ns, sid, sources}}, x in sources, is_in(x, $ids)"
                ),
                vec![("ns", s(ns)), ("ids", str_list(&ids))],
            )?
            .rows
            .iter()
            .filter_map(|r| cell_str(&r[0]))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut out = vec![];
        if !sids.is_empty() {
            self.mem_q(
                &format!("?[ns, sid] := *{MEM_SUMMARY}{{ns, sid}}, ns = $ns, sid in $sids :rm {MEM_SUMMARY} {{ns, sid}}"),
                vec![("ns", s(ns)), ("sids", str_list(&sids))],
            )?;
            out.extend(self.erase_in(cfg, ns, &sids)?);
        }
        if self.consol_exists(MEM_PROFILE) {
            self.mem_q(
                &format!(
                    "?[ns, node] := *{MEM_PROFILE}{{ns, node, evidence}}, ns = $ns, x in evidence, is_in(x, $ids) \
                     :rm {MEM_PROFILE} {{ns, node}}"
                ),
                vec![("ns", s(ns)), ("ids", str_list(&ids))],
            )?;
        }
        Ok(out)
    }

    // ---------------------------------------------------------- automation

    /// Record the auto settings, define the workflow, and (with `cron`)
    /// schedule it.
    fn consolidate_auto(&'s self, ns: &str, every: Option<i64>, cron: Option<String>) -> Result<JsonValue> {
        let every = every.unwrap_or(0).max(0);
        self.mem_q(
            &format!(
                "?[ns, every, cron, last] <- [[$ns, $every, $cron, 0]] :put {MEM_CONSOL} {{ns => every, cron, last}}"
            ),
            vec![
                ("ns", s(ns)),
                ("every", DataValue::from(every)),
                ("cron", cron.as_deref().map(s).unwrap_or(DataValue::Null)),
            ],
        )?;
        self.consol_state.auto.lock().unwrap().insert(ns.to_string(), (every, 0));
        self.define_consolidate_workflow()?;
        let mut out = serde_json::json!({"every": every, "workflow": CONSOLIDATE_WORKFLOW});
        if let Some(c) = cron {
            let name = format!("{CONSOLIDATE_WORKFLOW}:{ns}");
            self.flow_call(
                "schedule",
                serde_json::json!({"name": name, "cron": c, "workflow": CONSOLIDATE_WORKFLOW, "input": {"ns": ns}, "overlap": "skip"}),
            )?;
            out["schedule"] = JsonValue::String(name);
        }
        Ok(out)
    }

    fn define_consolidate_workflow(&'s self) -> Result<()> {
        self.flow_call(
            "define",
            serde_json::json!({"name": CONSOLIDATE_WORKFLOW,
                "graph": {"steps": [{"name": "consolidate", "consolidate": {"ns": "$input.ns"},
                                     "retry": {"max_attempts": 3}}]}}),
        )?;
        Ok(())
    }

    /// After a `remember`: count the namespace's new memories and, every
    /// `every` of them, start the consolidation workflow (best effort).
    pub(crate) fn consolidate_after_remember(&'s self, ns: &str, added: usize) {
        if added == 0 {
            return;
        }
        let every = {
            let mut a = self.consol_state.auto.lock().unwrap();
            if let Some(x) = a.get_mut(ns) {
                if x.0 <= 0 {
                    return;
                }
                x.1 += added as i64;
                if x.1 < x.0 {
                    return;
                }
                x.1 = 0;
                x.0
            } else {
                drop(a);
                // first sight of the namespace in this process: read its setting
                let e = if self.consol_exists(MEM_CONSOL) {
                    self.mem_q(&format!("?[every] := *{MEM_CONSOL}{{ns: $ns, every}}"), vec![("ns", s(ns))])
                        .ok()
                        .and_then(|r| r.rows.first().and_then(|x| x[0].get_int()))
                        .unwrap_or(DEFAULT_CONSOLIDATE_EVERY)
                } else {
                    DEFAULT_CONSOLIDATE_EVERY
                };
                self.consol_state.auto.lock().unwrap().insert(ns.to_string(), (e, added as i64));
                return;
            }
        };
        let head = self.ns_head(ns).unwrap_or(0);
        let _ = self.define_consolidate_workflow().and_then(|_| {
            self.flow_call(
                "start",
                serde_json::json!({"workflow": CONSOLIDATE_WORKFLOW, "input": {"ns": ns},
                    "id": format!("consolidate:{ns}:{head}"), "serial": format!("consolidate:{ns}"),
                    "queue": CONSOLIDATE_QUEUE}),
            )
        });
        self.consol_state.pending.store(true, std::sync::atomic::Ordering::SeqCst);
        let _ = every;
    }
}

/// Sentence openers a leading-name guess can mistake for a name.
const NOT_NAMES: &[&str] = &[
    "those", "these", "this", "that", "there", "it", "they", "we", "you", "he", "she", "here", "what", "when",
    "which", "who", "why", "how", "my", "our", "your", "their", "his", "her", "its", "the", "a", "an", "yes", "no",
    "wow", "oh", "hey", "thanks", "thank", "sure", "well", "so", "and", "but", "also", "just",
];

fn json_obj() -> serde_json::Map<String, JsonValue> {
    serde_json::Map::new()
}

fn graph_node(s: &str) -> String {
    s.strip_prefix("name:")
        .unwrap_or(s)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
}

/// An abstractive summary of `claims` from an OpenAI-compatible endpoint.
#[cfg(feature = "requests")]
fn llm_summary(l: &LlmOpts, claims: &[String]) -> Result<String> {
    let url = if l.url.trim_end_matches('/').ends_with("/chat/completions") {
        l.url.clone()
    } else {
        format!("{}/chat/completions", l.url.trim_end_matches('/'))
    };
    let prompt = format!(
        "Summarize these statements from one conversation in at most three sentences. Keep names, numbers and dates; add nothing.\n{}",
        claims.iter().map(|c| format!("- {c}")).collect::<Vec<_>>().join("\n")
    );
    let body = serde_json::json!({
        "model": l.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": l.max_tokens.unwrap_or(2000),
        "temperature": 0,
    });
    let mut req = minreq::post(url).with_header("Content-Type", "application/json").with_timeout(120);
    if let Some(env) = &l.api_key_env {
        let key = std::env::var(env).map_err(|_| miette::miette!("memory consolidate: environment variable '{env}' is not set"))?;
        req = req.with_header("Authorization", format!("Bearer {key}"));
    }
    let resp = req.with_body(body.to_string()).send().map_err(|e| miette::miette!("memory consolidate: llm request failed: {e}"))?;
    if resp.status_code / 100 != 2 {
        bail!("memory consolidate: llm returned HTTP {}", resp.status_code);
    }
    let v: JsonValue = serde_json::from_str(resp.as_str().map_err(|e| miette::miette!("{e}"))?).into_diagnostic()?;
    Ok(v["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_string())
}

#[cfg(not(feature = "requests"))]
fn llm_summary(_l: &LlmOpts, _claims: &[String]) -> Result<String> {
    bail!("memory consolidate: llm summaries need kleos built with the `requests` feature")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn central_and_diverse() {
        let texts: Vec<String> = [
            "Caroline adopted a dog named Max in May.",
            "Caroline adopted a puppy called Max this spring.",
            "Caroline works as a counselor at a youth center.",
            "The weather was rainy.",
            "Caroline volunteers at the youth center on weekends.",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let vecs: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let p = select_central(&texts, &vecs, 2);
        assert_eq!(p.len(), 2);
        // not both near-duplicate adoption claims
        assert!(!(p.contains(&0) && p.contains(&1)), "{p:?}");
        assert!(!p.contains(&3), "{p:?}");
    }

    #[test]
    fn titles() {
        assert_eq!(title("user"), "User");
        assert_eq!(title("anna maria"), "Anna Maria");
        assert_eq!(title("user.sister"), "User's Sister");
    }
}
