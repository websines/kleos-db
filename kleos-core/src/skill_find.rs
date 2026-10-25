/*
 * Copyright 2026, The kleos authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Skill retrieval and serving (docs/specs/skills.md, "Retrieval").
//!
//! `find` pre-selects skills for a task instead of handing an agent the
//! whole catalog (routing over a flat list collapses as the library grows):
//!
//! 1. candidates: BM25 over the description and body chunks of every live
//!    skill and filtered HNSW over their embeddings, the scope filter inside
//!    both indexes;
//! 2. exact features per candidate skill: lexical score, cosine to its
//!    description, best cosine to a body chunk, name overlap, posterior
//!    success rate, usage, scope narrowness;
//! 3. a learned linear router scores them (logistic; online SGD from the
//!    outcomes reported against the `route` id each search returns);
//! 4. progressive disclosure under a token budget: descriptions, then the
//!    top hit's body when asked for and when it fits; resources as URIs.
//!
//! The SEP-2640 entries (`skills/list`, `skills/get`, `skill://` reads and
//! directory reads) are built here too.

use std::collections::{BTreeMap, BTreeSet};

use miette::{bail, miette, Result};
use serde_json::{json, Value as JsonValue};

use crate::data::value::DataValue;
use crate::runtime::db::Db;
use crate::skill::*;
use crate::storage::Storage;

/// Router features, in weight order.
pub const FEATURES: [&str; 8] = ["bias", "lexical", "desc_sim", "body_sim", "name_overlap", "success", "usage", "narrow"];
/// Prior router weights (before any outcome): the semantic legs dominate, the
/// body counts (SkillRouter: hiding bodies costs 37-44 points), success and
/// usage nudge.
pub const W0: [f64; 8] = [-4.0, 2.0, 3.0, 4.0, 1.5, 1.5, 0.3, 0.3];
const LEARNING_RATE: f64 = 0.05;
const L2: f64 = 1e-3;
/// Weight of the implicit negatives (skills ranked above the one used).
const NEGATIVE_WEIGHT: f64 = 0.3;
const DEFAULT_K: usize = 5;
const DEFAULT_BUDGET: usize = 2000;
/// Candidate skills per leg before the exact rerank.
const FETCH: usize = 200;
const MAX_CANDIDATES: usize = 120;
/// Default size of a `skills/list` page without a task.
const LIST_TOP: usize = 32;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
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

fn vec_of(v: &DataValue) -> Option<Vec<f32>> {
    match v {
        DataValue::Vec(crate::data::value::Vector::F32(a)) => Some(a.to_vec()),
        DataValue::Vec(crate::data::value::Vector::F64(a)) => Some(a.iter().map(|x| *x as f32).collect()),
        DataValue::List(l) => l.iter().map(|x| x.get_float().map(|f| f as f32)).collect(),
        _ => None,
    }
}

/// FTS query from free text: its words, OR-ed.
fn fts_query(s: &str) -> String {
    let mut seen = BTreeSet::new();
    s.split(|c: char| !(c.is_alphabetic() || c.is_ascii_digit()))
        .filter(|w| w.len() > 1)
        .map(|w| w.to_lowercase())
        .filter(|w| seen.insert(w.clone()))
        .take(48)
        .collect::<Vec<_>>()
        .join(" OR ")
}

fn words(s: &str) -> BTreeSet<String> {
    s.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2).map(|w| w.to_lowercase()).collect()
}

/// Mime type of a skill file by extension.
pub fn mime_of(path: &str) -> &'static str {
    let ext = path.rsplit('.').next().unwrap_or("").to_lowercase();
    match ext.as_str() {
        "md" | "markdown" => "text/markdown",
        "txt" => "text/plain",
        "py" => "text/x-python",
        "sh" | "bash" => "text/x-shellscript",
        "js" | "mjs" => "text/javascript",
        "ts" => "text/x-typescript",
        "json" => "application/json",
        "yaml" | "yml" => "application/yaml",
        "html" | "htm" => "text/html",
        "csv" => "text/csv",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "svg" => "image/svg+xml",
        "pdf" => "application/pdf",
        _ => "application/octet-stream",
    }
}

/// One candidate skill during a search.
#[derive(Debug, Clone, Default)]
struct Cand {
    scope: String,
    name: String,
    lexical: f64,
    desc_sim: f64,
    body_sim: f64,
}

impl<'s, S: Storage<'s>> Db<S> {
    /// The router weights: learned ones, else [`W0`].
    pub(crate) fn skill_router_weights(&'s self) -> JsonValue {
        match self.skill_meta("router").ok().flatten() {
            Some(v) if v.get("w").and_then(|w| w.as_array()).map_or(false, |a| a.len() == FEATURES.len()) => v,
            _ => json!({"w": W0, "n": 0, "features": FEATURES}),
        }
    }

    fn router_w(&'s self) -> Vec<f64> {
        let v = self.skill_router_weights();
        v["w"].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect()).unwrap_or_else(|| W0.to_vec())
    }

    /// One SGD step of the router from an outcome reported against a search:
    /// the used skill is a positive (success) or a negative (failure); on a
    /// success, skills ranked above it are weak negatives.
    pub(crate) fn skill_router_learn(&'s self, rid: &str, scope: &str, name: &str, success: bool) -> Result<JsonValue> {
        let r = self.sk_r(&format!("?[shown] := *{SKILL_ROUTE}{{rid: $r, shown}}"), vec![("r", sv(rid))])?;
        let Some(shown) = r.rows.first().map(|x| jv(&x[0])) else { bail!("no route '{rid}'") };
        let items = shown.as_array().cloned().unwrap_or_default();
        let feat = |x: &JsonValue| -> Option<Vec<f64>> { x.get("f")?.as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()) };
        let Some(pos) = items.iter().position(|x| x["scope"] == scope && x["name"] == name) else {
            bail!("skill '{name}' was not among route '{rid}'s results")
        };
        let mut w = self.router_w();
        let mut step = |f: &[f64], y: f64, weight: f64| {
            let z: f64 = w.iter().zip(f).map(|(a, b)| a * b).sum();
            let g = (y - sigmoid(z)) * weight;
            for (wi, fi) in w.iter_mut().zip(f) {
                *wi += LEARNING_RATE * (g * fi - L2 * *wi);
            }
        };
        let mut n = 0;
        if let Some(f) = feat(&items[pos]) {
            step(&f, if success { 1.0 } else { 0.0 }, 1.0);
            n += 1;
        }
        if success {
            for x in &items[..pos] {
                if let Some(f) = feat(x) {
                    step(&f, 0.0, NEGATIVE_WEIGHT);
                    n += 1;
                }
            }
        }
        let prev = self.skill_router_weights()["n"].as_i64().unwrap_or(0);
        let v = json!({"w": w, "n": prev + 1, "features": FEATURES});
        self.set_skill_meta("router", &v)?;
        Ok(json!({"updated": n, "n": prev + 1}))
    }

    /// Candidate skills for a task, with their exact retrieval features.
    fn skill_candidates(&'s self, task: &str, scopes: &[String]) -> Result<Vec<Cand>> {
        let mut cands: BTreeMap<(String, String), Cand> = BTreeMap::new();
        let fq = fts_query(task);
        if !fq.is_empty() {
            let r = self.sk_r(
                &format!(
                    "?[scope, name, s] := ~{SKILL_CHUNK}:fts{{scope, name | query: $fq, k: {FETCH}, bind_score: s, filter: is_in(scope, $scopes)}}"
                ),
                vec![("fq", sv(&fq)), ("scopes", str_list(scopes))],
            )?;
            for x in r.rows {
                let (Some(s), Some(n)) = (cs(&x[0]), cs(&x[1])) else { continue };
                let score = x[2].get_float().unwrap_or(0.0);
                let c = cands.entry((s.clone(), n.clone())).or_insert(Cand { scope: s, name: n, ..Default::default() });
                c.lexical = c.lexical.max(score);
            }
        }
        let emb = self.skill_embedder();
        let qv: Option<Vec<f32>> = match &emb {
            Some((e, _)) => {
                let qname = format!("{e}_query");
                let m = if crate::embed::is_embedder_enabled(&qname) { qname } else { e.clone() };
                crate::embed::embed_texts(&m, &[task])?.pop().map(|v| v.to_vec())
            }
            None => None,
        };
        if let Some(q) = &qv {
            let r = self.sk_r(
                &format!(
                    "?[scope, name, d] := ~{SKILL_VEC}:hnsw{{scope, name | query: vec($qv), k: {FETCH}, ef: {FETCH}, bind_distance: d, filter: is_in(scope, $scopes)}}"
                ),
                vec![("qv", DataValue::List(q.iter().map(|x| DataValue::from(*x as f64)).collect())), ("scopes", str_list(scopes))],
            )?;
            for x in r.rows {
                let (Some(s), Some(n)) = (cs(&x[0]), cs(&x[1])) else { continue };
                cands.entry((s.clone(), n.clone())).or_insert(Cand { scope: s, name: n, ..Default::default() });
            }
        }
        // normalise BM25 per query; keep the best candidates of either leg
        let max_lex = cands.values().map(|c| c.lexical).fold(0.0, f64::max);
        let mut list: Vec<Cand> = cands
            .into_values()
            .map(|mut c| {
                if max_lex > 0.0 {
                    c.lexical /= max_lex;
                }
                c
            })
            .collect();
        // exact cosines over every chunk of each candidate
        if let Some(q) = &qv {
            let keys: Vec<DataValue> = list.iter().map(|c| DataValue::List(vec![sv(&c.scope), sv(&c.name)])).collect();
            let r = self.sk_r(
                &format!("?[scope, name, n, vec] := k in $keys, scope = get(k, 0), name = get(k, 1), *{SKILL_VEC}{{scope, name, n, vec}}"),
                vec![("keys", DataValue::List(keys))],
            )?;
            let mut sims: BTreeMap<(String, String), (f64, f64)> = BTreeMap::new();
            for x in r.rows {
                let (Some(s), Some(n), Some(i), Some(v)) = (cs(&x[0]), cs(&x[1]), x[2].get_int(), vec_of(&x[3])) else { continue };
                let c = cosine(q, &v);
                let e = sims.entry((s, n)).or_insert((0.0, 0.0));
                if i == 0 {
                    e.0 = c;
                } else {
                    e.1 = e.1.max(c);
                }
            }
            for c in list.iter_mut() {
                if let Some((d, b)) = sims.get(&(c.scope.clone(), c.name.clone())) {
                    c.desc_sim = *d;
                    c.body_sim = b.max(*d * 0.9);
                }
            }
        }
        list.sort_by(|a, b| {
            let ka = a.lexical * 0.5 + a.desc_sim.max(a.body_sim);
            let kb = b.lexical * 0.5 + b.desc_sim.max(b.body_sim);
            kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
        });
        list.truncate(MAX_CANDIDATES);
        Ok(list)
    }

    /// `find {task, k?, budget?, scope?, scopes?, disclosure?, explore?, log?}`.
    pub(crate) fn skill_find(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let task = args
            .get("task")
            .or_else(|| args.get("query"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| miette!("skill find needs 'task'"))?;
        let scopes = self.skill_scopes(args)?;
        let k = args.get("k").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(DEFAULT_K).max(1);
        let budget = args.get("budget").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(DEFAULT_BUDGET);
        let disclosure = args.get("disclosure").and_then(|v| v.as_str()).unwrap_or("descriptions");
        let explore = args.get("explore").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let cands = self.skill_candidates(task, &scopes)?;
        // shadowing: a narrower scope's skill hides a broader one of the same name
        let mut best: BTreeMap<String, Cand> = BTreeMap::new();
        for c in cands {
            match best.get(&c.name) {
                Some(b) if scope_rank(&b.scope) <= scope_rank(&c.scope) => {}
                _ => {
                    best.insert(c.name.clone(), c);
                }
            }
        }
        // live versions and their outcome counts
        let keys: Vec<DataValue> = best.values().map(|c| DataValue::List(vec![sv(&c.scope), sv(&c.name)])).collect();
        let r = self.sk_r(
            &format!("?[scope, name, ver, description] := k in $keys, scope = get(k, 0), name = get(k, 1), *{SKILL}{{scope, name, ver, description, status: 'active'}}"),
            vec![("keys", DataValue::List(keys))],
        )?;
        let live: BTreeMap<(String, String), (String, String)> =
            r.rows.iter().filter_map(|x| Some(((cs(&x[0])?, cs(&x[1])?), (cs(&x[2])?, cs(&x[3]).unwrap_or_default())))).collect();
        let vers: Vec<String> = live.values().map(|v| v.0.clone()).collect();
        let counts = self.skill_outcome_counts(&vers)?;
        let w = self.router_w();
        let tw = words(task);
        let mut scored: Vec<(f64, Vec<f64>, Cand, String, String, OutcomeAgg)> = vec![];
        for c in best.into_values() {
            let Some((ver, desc)) = live.get(&(c.scope.clone(), c.name.clone())).cloned() else { continue };
            let agg = counts.get(&ver).cloned().unwrap_or_default();
            let nw: Vec<&str> = c.name.split('-').filter(|x| x.len() > 2).collect();
            let overlap = if nw.is_empty() { 0.0 } else { nw.iter().filter(|x| tw.contains(**x)).count() as f64 / nw.len() as f64 };
            let mut success = agg.posterior() - 0.5;
            if explore > 0.0 {
                let (a, b) = (agg.successes as f64 + 1.0, (agg.uses - agg.successes) as f64 + 1.0);
                success += explore * (a * b / ((a + b).powi(2) * (a + b + 1.0))).sqrt();
            }
            let f = vec![
                1.0,
                c.lexical,
                c.desc_sim,
                c.body_sim,
                overlap,
                success,
                ((1.0 + agg.uses as f64).ln() / (101f64).ln()).min(1.0),
                (6.0 - scope_rank(&c.scope) as f64) / 6.0,
            ];
            let z: f64 = w.iter().zip(&f).map(|(a, b)| a * b).sum();
            scored.push((sigmoid(z), f, c, ver, desc, agg));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.2.name.cmp(&b.2.name)));
        scored.truncate(k);
        // progressive disclosure under the budget
        let mut used = 0usize;
        let mut results = vec![];
        let top_margin = match (scored.first(), scored.get(1)) {
            (Some(a), Some(b)) => a.0 - b.0,
            (Some(_), None) => 1.0,
            _ => 0.0,
        };
        for (i, (score, f, c, ver, desc, agg)) in scored.iter().enumerate() {
            let entry_tokens = est_tokens(&c.name) + est_tokens(desc) + 12;
            if used + entry_tokens > budget && i > 0 {
                break;
            }
            used += entry_tokens;
            let mut o = json!({
                "name": c.name, "scope": c.scope, "ver": ver, "uri": skill_uri(&c.scope, &c.name, "SKILL.md"),
                "description": desc, "score": score, "rank": i + 1,
                "stats": {"uses": agg.uses, "success_rate": if agg.uses > 0 { json!(agg.successes as f64 / agg.uses as f64) } else { JsonValue::Null }, "posterior": agg.posterior()},
                "features": FEATURES.iter().zip(f).skip(1).map(|(k, v)| (k.to_string(), json!(v))).collect::<serde_json::Map<_, _>>(),
            });
            let want_body = i == 0 && (disclosure == "body" || (disclosure == "auto" && top_margin > 0.15));
            if let Some(row) = self.skill_ver(ver)? {
                let body = row["body"].as_str().unwrap_or_default();
                let bt = est_tokens(body);
                o["tokens"] = json!({"description": entry_tokens, "body": bt});
                if want_body && used + bt <= budget {
                    used += bt;
                    o["body"] = json!(body);
                }
                o["resources"] = JsonValue::Array(
                    entries_of(&row["files"])
                        .into_iter()
                        .filter(|e| e.path != "SKILL.md")
                        .map(|e| json!({"path": e.path, "uri": skill_uri(&c.scope, &c.name, &e.path), "size": e.size}))
                        .collect(),
                );
            }
            results.push(o);
        }
        let rid = new_id("sr");
        if args.get("log").and_then(|v| v.as_bool()).unwrap_or(true) && !scored.is_empty() {
            let shown: Vec<JsonValue> = scored.iter().enumerate().map(|(i, s)| json!({"scope": s.2.scope, "name": s.2.name, "ver": s.3, "f": s.1, "score": s.0, "rank": i + 1})).collect();
            self.sk_w(
                &format!("?[rid, at, task, scopes, shown] <- [[$r, $now, $t, $sc, $sh]] :put {SKILL_ROUTE} {{rid => at, task, scopes, shown}}"),
                vec![("r", sv(&rid)), ("now", DataValue::from(now_us())), ("t", sv(task)), ("sc", str_list(&scopes)), ("sh", jdn(&JsonValue::Array(shown)))],
            )?;
        }
        Ok(json!({"route": rid, "task": task, "skills": results, "tokens": used, "budget": budget, "scopes": scopes}))
    }

    // ---------------------------------------------------------- SEP-2640

    /// The SEP-2640 entry of a version: `{uri, frontmatter, resources}` plus
    /// kleos details under `_meta`.
    pub(crate) fn skill_entry_of(&'s self, row: &serde_json::Map<String, JsonValue>) -> Result<JsonValue> {
        let scope = row["scope"].as_str().unwrap_or_default();
        let name = row["name"].as_str().unwrap_or_default();
        let ver = row["ver"].as_str().unwrap_or_default();
        // the frontmatter is re-read from the stored SKILL.md so the entry is
        // identical in content to the file a host will fetch
        let md = self.skill_file(&row["files"], "SKILL.md")?;
        let fm = parse_skill_md(&String::from_utf8_lossy(&md))?.frontmatter;
        let resources: Vec<JsonValue> =
            entries_of(&row["files"]).iter().map(|e| json!({"uri": skill_uri(scope, name, &e.path), "digest": e.digest, "size": e.size})).collect();
        let stats = self.skill_stats_of(ver)?;
        Ok(json!({
            "uri": skill_uri(scope, name, "SKILL.md"),
            "frontmatter": fm,
            "resources": resources,
            "_meta": {"kleos/ver": ver, "kleos/version": row["seq"], "kleos/digest": row["digest"], "kleos/scope": scope,
                      "kleos/status": row["status"], "kleos/stats": stats, "kleos/signatures": self.skill_sigs(ver)?.as_array().map(|a| a.len()).unwrap_or(0)}
        }))
    }

    /// The live version of the skill a `skill://` URI names, and the rest of
    /// the path.
    pub(crate) fn skill_by_uri(&'s self, uri: &str, scopes: Option<&[String]>) -> Result<(serde_json::Map<String, JsonValue>, String)> {
        for (scope, name, rest) in uri_splits(uri) {
            if scopes.is_some_and(|ss| !ss.contains(&scope)) {
                continue;
            }
            if let Some(row) = self.skill_row(&scope, &name)? {
                if row.get("status").and_then(|s| s.as_str()) != Some("active") {
                    continue;
                }
                if let Some(v) = row.get("ver").and_then(|v| v.as_str()) {
                    if let Some(vr) = self.skill_ver(v)? {
                        return Ok((vr, rest));
                    }
                }
            }
        }
        bail!("Resource not found: {uri}")
    }

    /// `entry {uri | name+scope}` (SEP-2640 `skills/get`).
    pub(crate) fn skill_entry_op(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let row = match args.get("uri").and_then(|u| u.as_str()) {
            Some(uri) => {
                let (row, rest) = self.skill_by_uri(uri, None)?;
                if rest != "SKILL.md" {
                    bail!("'{uri}' is not the SKILL.md of a skill");
                }
                row
            }
            None => self.skill_resolve(args)?,
        };
        self.skill_entry_of(&row)
    }

    /// `entries {task?, k?, scopes?, mode?, cursor?, limit?}` (SEP-2640
    /// `skills/list`). With a task: the top-k for it (a partial listing).
    /// Without: `mode: 'top'` (default) the `limit` most useful live skills
    /// (partial), `mode: 'full'` the whole catalog in pages.
    pub(crate) fn skill_entries(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let scopes = self.skill_scopes(args)?;
        let task = args.get("task").or_else(|| args.get("query")).and_then(|v| v.as_str()).filter(|s| !s.trim().is_empty());
        if let Some(t) = task {
            let mut a = args.clone();
            a["task"] = json!(t);
            a["budget"] = json!(1_000_000);
            let found = self.skill_find(&a)?;
            let mut out = vec![];
            for s in found["skills"].as_array().cloned().unwrap_or_default() {
                if let Some(row) = self.skill_ver(s["ver"].as_str().unwrap_or_default())? {
                    let mut e = self.skill_entry_of(&row)?;
                    e["_meta"]["kleos/score"] = s["score"].clone();
                    e["_meta"]["kleos/route"] = found["route"].clone();
                    out.push(e);
                }
            }
            return Ok(json!({"skills": out, "partial": true, "route": found["route"]}));
        }
        let mode = args.get("mode").and_then(|v| v.as_str()).unwrap_or("top");
        let r = self.sk_r(
            &format!("?[scope, name, ver] := *{SKILL}{{scope, name, ver, status: 'active'}}, is_in(scope, $scopes)"),
            vec![("scopes", str_list(&scopes))],
        )?;
        let mut rows: Vec<(String, String, String)> = r.rows.iter().filter_map(|x| Some((cs(&x[0])?, cs(&x[1])?, cs(&x[2])?))).collect();
        // shadowing by scope precedence
        rows.sort_by(|a, b| (scope_rank(&a.0), &a.1).cmp(&(scope_rank(&b.0), &b.1)));
        let mut seen = BTreeSet::new();
        rows.retain(|r| seen.insert(r.1.clone()));
        let limit = args.get("limit").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(if mode == "full" { 100 } else { LIST_TOP });
        let (page, next): (Vec<(String, String, String)>, Option<String>) = if mode == "full" {
            rows.sort_by(|a, b| (&a.1, &a.0).cmp(&(&b.1, &b.0)));
            let off: usize = args.get("cursor").and_then(|c| c.as_str()).and_then(|c| c.parse().ok()).unwrap_or(0);
            let end = (off + limit).min(rows.len());
            let next = (end < rows.len()).then(|| end.to_string());
            (rows.get(off..end).map(|x| x.to_vec()).unwrap_or_default(), next)
        } else {
            let vers: Vec<String> = rows.iter().map(|r| r.2.clone()).collect();
            let c = self.skill_outcome_counts(&vers)?;
            let usefulness = |v: &str| {
                let a = c.get(v).cloned().unwrap_or_default();
                a.posterior() * (1.0 + (1.0 + a.uses as f64).ln())
            };
            rows.sort_by(|a, b| usefulness(&b.2).partial_cmp(&usefulness(&a.2)).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
            rows.truncate(limit);
            rows.sort_by(|a, b| a.1.cmp(&b.1));
            (rows, None)
        };
        let mut out = vec![];
        for (_, _, v) in page {
            if let Some(row) = self.skill_ver(&v)? {
                out.push(self.skill_entry_of(&row)?);
            }
        }
        let mut res = json!({"skills": out, "partial": mode != "full"});
        if let Some(n) = next {
            res["nextCursor"] = json!(n);
        }
        Ok(res)
    }

    /// `read {uri}`: one file of a live skill (`resources/read`), checked
    /// against its digest.
    pub(crate) fn skill_read_uri(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let uri = args.get("uri").and_then(|u| u.as_str()).ok_or_else(|| miette!("skill read needs 'uri'"))?;
        let (row, rest) = self.skill_by_uri(uri, None)?;
        let path = if rest.is_empty() { "SKILL.md".to_string() } else { rest };
        let b = self.skill_file(&row["files"], &path).map_err(|_| miette!("Resource not found: {uri}"))?;
        let mime = mime_of(&path);
        let mut c = json!({"uri": uri, "mimeType": mime});
        match String::from_utf8(b.clone()) {
            Ok(t) if !mime.starts_with("image/") && mime != "application/pdf" => c["text"] = json!(t),
            _ => {
                use base64::Engine;
                c["blob"] = json!(base64::engine::general_purpose::STANDARD.encode(&b))
            }
        }
        if path == "SKILL.md" {
            c["_meta"] = json!({"io.modelcontextprotocol.skills/name": row["name"], "kleos/digest": digest_of(&b)});
        }
        Ok(json!({"contents": [c]}))
    }

    /// `dir {uri}`: the direct children of a directory of a live skill
    /// (`resources/directory/read`).
    pub(crate) fn skill_dir_uri(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let uri = args.get("uri").and_then(|u| u.as_str()).ok_or_else(|| miette!("skill dir needs 'uri'"))?;
        let (row, rest) = self.skill_by_uri(uri, None)?;
        let scope = row["scope"].as_str().unwrap_or_default();
        let name = row["name"].as_str().unwrap_or_default();
        let prefix = if rest.is_empty() { String::new() } else { format!("{}/", rest.trim_end_matches('/')) };
        let mut kids: BTreeMap<String, bool> = BTreeMap::new();
        for e in entries_of(&row["files"]) {
            if let Some(tail) = e.path.strip_prefix(&prefix) {
                match tail.split_once('/') {
                    Some((d, _)) => {
                        kids.insert(format!("{prefix}{d}"), true);
                    }
                    None => {
                        kids.insert(e.path.clone(), false);
                    }
                }
            }
        }
        if kids.is_empty() {
            bail!("Resource not found: {uri} is not a directory");
        }
        let resources: Vec<JsonValue> = kids
            .into_iter()
            .map(|(p, dir)| {
                let leaf = p.rsplit('/').next().unwrap_or(&p).to_string();
                let u = skill_uri(scope, name, &p);
                if dir {
                    json!({"uri": u, "name": leaf, "mimeType": "inode/directory"})
                } else {
                    json!({"uri": u, "name": leaf, "mimeType": mime_of(&p)})
                }
            })
            .collect();
        Ok(json!({"resources": resources}))
    }
}
