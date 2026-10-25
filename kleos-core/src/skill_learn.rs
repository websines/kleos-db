/*
 * Copyright 2026, The kleos authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Learning and optimizing skills (docs/specs/skills.md, "Lifecycle and
//! learning"), native versions of SkillOpt-Sleep and SkillOpt.
//!
//! **`skill.learn`** (a durable `::flow` workflow on the `skills` queue,
//! cron-capable, off unless an LLM endpoint is configured):
//! 1. *harvest* successful traces (`record_trace`), completed workflows that
//!    carry `meta.task`, and `trace` memories of the scope;
//! 2. *cluster* recurring tasks (embeddings, else word overlap); a cluster
//!    needs `min_support` successes and must not be covered by a live skill;
//! 3. *propose*: the LLM hook drafts one focused `SKILL.md` per cluster from
//!    its training traces (failures become pitfalls), stored as a candidate
//!    (`origin: induced`) with the traces as provenance; the scanner runs;
//! 4. *verify*: replay the cluster's held-out tasks with and without the
//!    skill (the LLM as the agent; exact checks against the recorded result,
//!    else an LLM judge against it, CoEvoSkills' surrogate verifier);
//! 5. *admit*: a candidate that beats the baseline by `margin` goes live in
//!    its own scope (`mode: auto`) or waits as a proposal (`mode: review`);
//!    the rest are rejected. Then the lifecycle sweep runs.
//!
//! **`skill.optimize`**: SkillOpt's loop: from the failures of the current
//! best version on the training cases, the optimizer LLM proposes at most
//! `edit_budget` bounded line edits (GEPA-style reflection; ACE-style deltas
//! on the itemized body, never a rewrite), with the score history and the
//! rejected-edit buffer in the prompt; an edit set is kept only if it
//! strictly raises the held-out score. Each kept round is a new version
//! (`origin: optimized`, `parent`, `edits`); the winner supersedes the
//! original only if it beats it.

use std::collections::{BTreeMap, BTreeSet};

use miette::{bail, miette, IntoDiagnostic, Result};
use serde_json::{json, Value as JsonValue};

use crate::data::value::DataValue;
use crate::memory::consolidate::LlmOpts;
use crate::runtime::db::Db;
use crate::skill::*;
use crate::storage::Storage;

/// Settings of the learning loop (`::skill config {learn: {…}}`).
#[derive(Debug, Clone, serde_derive::Serialize, serde_derive::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LearnOpts {
    /// The OpenAI-compatible endpoint that drafts, replays and judges. No
    /// endpoint: no learning (and no schedule).
    pub llm: Option<LlmOpts>,
    /// `auto`: verified skills go live in their own scope; `review`: they wait
    /// as proposals.
    pub mode: String,
    /// Schedule (`::flow` cron syntax); `off` for manual runs only.
    pub cron: String,
    /// Scopes the scheduled run learns in.
    pub scopes: Vec<String>,
    /// Successful traces a cluster needs.
    pub min_support: usize,
    /// Similarity for two tasks to share a cluster (cosine; word overlap is
    /// scaled down when there is no embedder).
    pub cluster_threshold: f64,
    /// Share of a cluster held out for verification.
    pub holdout: f64,
    /// Lift over the no-skill baseline a candidate needs.
    pub margin: f64,
    /// New skills per run, at most.
    pub max_new: usize,
    /// Harvest traces of the last N days.
    pub since_days: i64,
    /// Parallel LLM calls during replays.
    pub parallel: usize,
}

impl Default for LearnOpts {
    fn default() -> Self {
        Self {
            llm: None,
            mode: "auto".into(),
            cron: "@daily".into(),
            scopes: vec![DEFAULT_SCOPE.into()],
            min_support: 3,
            cluster_threshold: 0.6,
            holdout: 0.4,
            margin: 0.1,
            max_new: 3,
            since_days: 30,
            parallel: 4,
        }
    }
}

const TRANSCRIPT_CHARS: usize = 3000;
const DEFAULT_ROUNDS: i64 = 4;
const DEFAULT_EDIT_BUDGET: usize = 3;
const EDIT_CHARS: usize = 400;
const REJECTED_BUFFER: usize = 8;

fn arg<'a>(a: &'a JsonValue, k: &str) -> Option<&'a str> {
    a.get(k).and_then(|v| v.as_str()).filter(|s| !s.is_empty())
}

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

// ------------------------------------------------------------------ LLM

/// One chat completion from an OpenAI-compatible endpoint. Reasoning models
/// return their reasoning separately; any `<think>` block left in the
/// content is dropped. Retries rate limits and server errors.
#[cfg(feature = "requests")]
pub fn llm_chat(l: &LlmOpts, system: &str, user: &str, temperature: f64) -> Result<(String, i64)> {
    let url = if l.url.trim_end_matches('/').ends_with("/chat/completions") {
        l.url.clone()
    } else {
        format!("{}/chat/completions", l.url.trim_end_matches('/'))
    };
    let mut messages = vec![];
    if !system.is_empty() {
        messages.push(json!({"role": "system", "content": system}));
    }
    messages.push(json!({"role": "user", "content": user}));
    let body = json!({"model": l.model, "messages": messages, "max_tokens": l.max_tokens.unwrap_or(4000), "temperature": temperature});
    let key = match &l.api_key_env {
        Some(env) => Some(std::env::var(env).map_err(|_| miette!("skill llm: environment variable '{env}' is not set"))?),
        None => None,
    };
    let mut last = String::new();
    for attempt in 0..5 {
        let mut req = minreq::post(&url).with_header("Content-Type", "application/json").with_timeout(240);
        if let Some(k) = &key {
            req = req.with_header("Authorization", format!("Bearer {k}"));
        }
        match req.with_body(body.to_string()).send() {
            Ok(resp) if resp.status_code / 100 == 2 => {
                let v: JsonValue = serde_json::from_str(resp.as_str().map_err(|e| miette!("{e}"))?).into_diagnostic()?;
                let mut text = v["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_string();
                while let (Some(a), Some(b)) = (text.find("<think>"), text.find("</think>")) {
                    if b < a {
                        break;
                    }
                    text.replace_range(a..b + "</think>".len(), "");
                }
                let tokens = v["usage"]["total_tokens"].as_i64().unwrap_or(0);
                return Ok((text.trim().to_string(), tokens));
            }
            Ok(resp) if resp.status_code == 429 || resp.status_code >= 500 => last = format!("HTTP {}", resp.status_code),
            Ok(resp) => bail!("skill llm: HTTP {}: {}", resp.status_code, clip(resp.as_str().unwrap_or(""), 300)),
            Err(e) => last = e.to_string(),
        }
        std::thread::sleep(std::time::Duration::from_millis(1500 << attempt));
    }
    bail!("skill llm: request failed after retries: {last}")
}

#[cfg(not(feature = "requests"))]
pub fn llm_chat(_l: &LlmOpts, _system: &str, _user: &str, _temperature: f64) -> Result<(String, i64)> {
    bail!("skill learning needs kleos built with the `requests` feature")
}

/// The first JSON value (object or array) in a model's answer.
pub fn extract_json(text: &str) -> Option<JsonValue> {
    let t = text.trim();
    if let Ok(v) = serde_json::from_str::<JsonValue>(t) {
        return Some(v);
    }
    if let Some(i) = t.find("```") {
        let rest = &t[i + 3..];
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        if let Some(j) = rest.find("```") {
            if let Ok(v) = serde_json::from_str(rest[..j].trim()) {
                return Some(v);
            }
        }
    }
    for (open, close) in [('[', ']'), ('{', '}')] {
        if let (Some(a), Some(b)) = (t.find(open), t.rfind(close)) {
            if b > a {
                if let Ok(v) = serde_json::from_str(&t[a..=b]) {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// A SKILL.md from a model's answer (fences stripped, an invalid name made
/// valid).
pub fn extract_skill_md(text: &str, fallback_name: &str) -> Result<String> {
    let mut t = text.trim().to_string();
    if let Some(i) = t.find("---") {
        t = t[i..].to_string();
    }
    if let Some(stripped) = t.strip_suffix("```") {
        t = stripped.trim_end().to_string();
    }
    let t = format!("{}\n", t.trim_end());
    match parse_skill_md(&t) {
        Ok(_) => Ok(t),
        Err(_) => {
            // repair the name, keep everything else
            let mut lines: Vec<String> = t.lines().map(String::from).collect();
            for l in lines.iter_mut().skip(1) {
                if l.trim() == "---" {
                    break;
                }
                if let Some(v) = l.strip_prefix("name:") {
                    *l = format!("name: {}", slug(v.trim().trim_matches('"').trim_matches('\''), fallback_name));
                }
            }
            let fixed = lines.join("\n") + "\n";
            parse_skill_md(&fixed)?;
            Ok(fixed)
        }
    }
}

/// A valid skill name from free text.
pub fn slug(s: &str, fallback: &str) -> String {
    let mut out = String::new();
    for c in s.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    let out: String = out.trim_matches('-').chars().take(60).collect();
    let out = out.trim_matches('-').to_string();
    if validate_name(&out).is_ok() {
        out
    } else {
        fallback.to_string()
    }
}

fn norm(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|c: char| c.is_ascii_punctuation() && c != '-' && c != '_')
        .to_string()
}

/// The final answer of a response: the text after the last `ANSWER:`, else
/// the last non-empty line.
pub fn final_answer(resp: &str) -> String {
    if let Some(i) = resp.rfind("ANSWER:") {
        return resp[i + 7..].lines().next().unwrap_or("").trim().trim_matches('`').trim().to_string();
    }
    resp.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim().to_string()
}

/// One replay case.
#[derive(Debug, Clone)]
pub struct Case {
    pub id: String,
    pub task: String,
    pub expected: Option<String>,
    pub check: Option<JsonValue>,
}

impl Case {
    fn from_row(o: &serde_json::Map<String, JsonValue>) -> Self {
        Case {
            id: o.get("case").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
            task: o.get("task").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
            expected: o.get("expected").and_then(|v| v.as_str()).map(String::from),
            check: o.get("check").filter(|v| !v.is_null()).cloned(),
        }
    }
}

/// The result of one replayed case.
#[derive(Debug, Clone)]
pub struct CaseResult {
    pub id: String,
    pub task: String,
    pub ok: bool,
    pub answer: String,
    pub expected: Option<String>,
    pub tokens: i64,
}

/// Does `answer` pass the case's check?
fn passes(l: &LlmOpts, case: &Case, response: &str) -> Result<bool> {
    let ans = final_answer(response);
    let check = case.check.clone().unwrap_or(JsonValue::Null);
    let kind = check.get("type").and_then(|t| t.as_str()).unwrap_or(if case.expected.is_some() { "answer" } else { "judge" });
    let exp = case.expected.clone().or_else(|| check.get("value").and_then(|v| v.as_str()).map(String::from)).unwrap_or_default();
    Ok(match kind {
        "equals" => norm(&ans) == norm(&exp),
        "contains" => norm(response).contains(&norm(&exp)),
        "regex" => regex::Regex::new(check.get("pattern").and_then(|p| p.as_str()).unwrap_or("$^")).map(|r| r.is_match(response)).unwrap_or(false),
        "answer" => {
            let (a, e) = (norm(&ans), norm(&exp));
            !e.is_empty() && (a == e || (e.chars().count() >= 3 && a.contains(&e)))
        }
        _ => {
            let rubric = check.get("rubric").and_then(|r| r.as_str()).unwrap_or("Is the response a correct and complete result for the task?");
            let reference = if exp.is_empty() { String::new() } else { format!("\nReference result from a successful run:\n{}\n", clip(&exp, 2000)) };
            let (v, _) = llm_chat(
                l,
                "You grade an agent's work. Reply with exactly YES or NO.",
                &format!("Task:\n{}\n{reference}\nResponse to grade:\n{}\n\n{rubric} Reply YES or NO.", case.task, clip(response, 4000)),
                0.0,
            )?;
            v.trim().to_uppercase().starts_with("YES")
        }
    })
}

/// Replay cases with the LLM as the agent, with a skill body in its
/// instructions or without (`skill: None`, the baseline).
pub fn replay(l: &LlmOpts, skill: Option<&str>, cases: &[Case], parallel: usize) -> Vec<CaseResult> {
    let mut system = "You are a careful agent completing a task. Work it out, then give the final result on the last line as `ANSWER: <result>`.".to_string();
    if let Some(s) = skill {
        system.push_str("\n\nFollow this skill (procedural instructions for this kind of task):\n<skill>\n");
        system.push_str(s);
        system.push_str("\n</skill>");
    }
    let run = |c: &Case| -> CaseResult {
        let (resp, tokens) = llm_chat(l, &system, &c.task, 0.0).unwrap_or_else(|e| (format!("(error: {e})"), 0));
        let ok = !resp.starts_with("(error:") && passes(l, c, &resp).unwrap_or(false);
        CaseResult { id: c.id.clone(), task: c.task.clone(), ok, answer: clip(&final_answer(&resp), 400), expected: c.expected.clone(), tokens }
    };
    let n = parallel.max(1);
    let per = cases.len().div_ceil(n).max(1);
    let mut out: Vec<Option<CaseResult>> = vec![None; cases.len()];
    std::thread::scope(|sc| {
        let handles: Vec<_> = cases
            .chunks(per)
            .enumerate()
            .map(|(ci, chunk)| {
                let run = &run;
                sc.spawn(move || chunk.iter().enumerate().map(|(i, c)| (ci * per + i, run(c))).collect::<Vec<_>>())
            })
            .collect();
        for h in handles {
            if let Ok(rs) = h.join() {
                for (i, r) in rs {
                    out[i] = Some(r);
                }
            }
        }
    });
    out.into_iter().flatten().collect()
}

fn score(rs: &[CaseResult]) -> f64 {
    if rs.is_empty() {
        0.0
    } else {
        rs.iter().filter(|r| r.ok).count() as f64 / rs.len() as f64
    }
}

fn results_json(rs: &[CaseResult]) -> JsonValue {
    JsonValue::Array(rs.iter().map(|r| json!({"case": r.id, "ok": r.ok, "answer": r.answer, "expected": r.expected})).collect())
}

/// Apply bounded line edits (`replace {line}`, `insert {after}`, `delete
/// {line}`; 1-based, against the original numbering) to a body.
pub fn apply_edits(body: &str, edits: &[JsonValue], budget: usize) -> Result<(String, Vec<JsonValue>)> {
    let mut lines: Vec<String> = body.lines().map(String::from).collect();
    let mut ops: Vec<(usize, u8, JsonValue)> = vec![];
    for e in edits.iter().take(budget) {
        let op = e.get("op").and_then(|o| o.as_str()).unwrap_or("");
        let text = e.get("text").and_then(|t| t.as_str()).unwrap_or("");
        if text.chars().count() > EDIT_CHARS {
            continue;
        }
        let (pos, order) = match op {
            "replace" | "delete" => (e.get("line").and_then(|l| l.as_u64()).unwrap_or(0) as usize, 1u8),
            "insert" | "add" => (e.get("after").or_else(|| e.get("line")).and_then(|l| l.as_u64()).unwrap_or(0) as usize, 0u8),
            _ => continue,
        };
        if pos > lines.len() || ((op == "replace" || op == "delete") && pos == 0) {
            continue;
        }
        ops.push((pos, order, e.clone()));
    }
    if ops.is_empty() {
        bail!("no applicable edit");
    }
    // bottom-up, so earlier line numbers stay valid
    ops.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
    let applied: Vec<JsonValue> = ops.iter().map(|o| o.2.clone()).collect();
    for (pos, _, e) in ops {
        let text = e.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string();
        match e["op"].as_str().unwrap_or("") {
            "replace" => lines[pos - 1] = text,
            "delete" => {
                lines.remove(pos - 1);
            }
            _ => {
                for (k, l) in text.lines().enumerate() {
                    lines.insert(pos + k, l.to_string());
                }
            }
        }
    }
    Ok((lines.join("\n") + "\n", applied))
}

fn numbered(body: &str) -> String {
    body.lines().enumerate().map(|(i, l)| format!("{}: {l}", i + 1)).collect::<Vec<_>>().join("\n")
}

fn word_set(s: &str) -> BTreeSet<String> {
    s.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2).map(|w| w.to_lowercase()).collect()
}

fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    let i = a.intersection(b).count() as f64;
    let u = a.union(b).count() as f64;
    if u == 0.0 {
        0.0
    } else {
        i / u
    }
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
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

/// A harvested trace.
#[derive(Debug, Clone)]
struct Trace {
    id: String,
    task: String,
    transcript: String,
    output: Option<String>,
    success: bool,
}

impl<'s, S: Storage<'s>> Db<S> {
    pub(crate) fn skill_learn_opts(&'s self) -> LearnOpts {
        self.skill_meta("learn").ok().flatten().and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default()
    }

    /// `config {learn: {…}}`: store the settings; with an LLM and a cron, the
    /// `skill.learn` schedule is (re)set, else removed.
    pub(crate) fn skill_learn_configure(&'s self, patch: &JsonValue) -> Result<()> {
        let mut cur = serde_json::to_value(self.skill_learn_opts()).into_diagnostic()?;
        merge(&mut cur, patch);
        if let Some(l) = cur.get_mut("llm").and_then(|l| l.as_object_mut()) {
            if l.contains_key("api_key") {
                bail!("skill config learn.llm: pass 'api_key_env' (the variable's name); keys are never stored");
            }
        }
        let o: LearnOpts = serde_json::from_value(cur).map_err(|e| miette!("skill config learn: {e}"))?;
        if !["auto", "review"].contains(&o.mode.as_str()) {
            bail!("skill config learn.mode must be 'auto' or 'review'");
        }
        self.set_skill_meta("learn", &serde_json::to_value(&o).into_diagnostic()?)?;
        self.skill_define_workflows()?;
        let name = format!("{LEARN_WORKFLOW}:schedule");
        if o.llm.is_some() && o.cron != "off" && !o.cron.is_empty() {
            self.flow_call(
                "schedule",
                json!({"name": name, "cron": o.cron, "workflow": LEARN_WORKFLOW, "input": {"scopes": o.scopes}, "queue": SKILL_QUEUE, "overlap": "skip"}),
            )?;
        } else {
            let _ = self.flow_call("unschedule", json!({"name": name}));
        }
        Ok(())
    }

    /// Define (or refresh) the `skill.learn` and `skill.optimize` graphs.
    pub(crate) fn skill_define_workflows(&'s self) -> Result<()> {
        let learn = json!({"steps": [
            {"name": "harvest", "skill": {"op": "learn_stage", "stage": "harvest", "input": "$input"}, "retry": {"max_attempts": 2}},
            {"name": "propose", "skill": {"op": "learn_stage", "stage": "propose", "input": "$input", "prev": "$harvest"}, "retry": {"max_attempts": 3}},
            {"name": "verify", "skill": {"op": "learn_stage", "stage": "verify", "input": "$input", "prev": "$propose"}, "retry": {"max_attempts": 3}},
            {"name": "admit", "skill": {"op": "learn_stage", "stage": "admit", "input": "$input", "prev": "$verify"}},
            {"name": "maintain", "skill": {"op": "learn_stage", "stage": "maintain", "input": "$input"}, "on_error": "continue"},
            {"return": "$admit"}
        ]});
        let optimize = json!({"steps": [
            {"name": "prepare", "skill": {"op": "optimize_stage", "stage": "prepare", "input": "$input"}, "retry": {"max_attempts": 2}},
            {"loop": {"while": "$i < $prepare.rounds", "max": 50, "body": [
                {"name": "round", "skill": {"op": "optimize_stage", "stage": "round", "run": "$prepare.run", "i": "$i"}, "retry": {"max_attempts": 3}}
            ]}},
            {"name": "finish", "skill": {"op": "optimize_stage", "stage": "finish", "run": "$prepare.run"}},
            {"return": "$finish"}
        ]});
        for (name, graph) in [(LEARN_WORKFLOW, learn), (OPTIMIZE_WORKFLOW, optimize)] {
            let cur = self.flow_graph_latest(name)?;
            if cur.as_ref() != Some(&graph) {
                self.flow_call("define", json!({"name": name, "graph": graph}))?;
            }
        }
        Ok(())
    }

    fn flow_graph_latest(&'s self, name: &str) -> Result<Option<JsonValue>> {
        self.flow_ensure()?;
        let r = self.sk_r("?[version, graph] := *kleos_flow_def{name: $n, version, graph} :order -version :limit 1", vec![("n", sv(name))])?;
        Ok(r.rows.first().map(|x| jv(&x[1])))
    }

    fn llm_of(&'s self, args: &JsonValue) -> Result<LlmOpts> {
        if let Some(l) = args.get("llm").filter(|l| l.is_object()) {
            return serde_json::from_value(l.clone()).map_err(|e| miette!("skill llm: {e}"));
        }
        self.skill_learn_opts().llm.ok_or_else(|| {
            miette!("skill learning needs an LLM endpoint: `::skill config {{learn: {{llm: {{url, model, api_key_env}}}}}}` (or pass llm)")
        })
    }

    /// Mark that a skill workflow waits for the in-process runner.
    fn skill_mark_pending(&'s self) {
        self.skill_state.pending.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    // ---------------------------------------------------------- learn

    /// `learn {scope?, scopes?, mode?, llm?, wait?}`: start a `skill.learn`
    /// run (a durable workflow), or with `wait: true` run it inside the call.
    pub(crate) fn skill_learn(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let scopes = self.skill_scopes(args)?;
        let mut input = json!({"scopes": scopes});
        for k in ["mode", "llm", "min_support", "margin", "max_new", "holdout", "since_days", "cluster_threshold"] {
            if let Some(v) = args.get(k).filter(|v| !v.is_null()) {
                input[k] = v.clone();
            }
        }
        self.llm_of(&input)?;
        if args.get("wait").and_then(|v| v.as_bool()) == Some(true) {
            let h = self.skill_learn_stage(&json!({"stage": "harvest", "input": input}))?;
            let p = self.skill_learn_stage(&json!({"stage": "propose", "input": input, "prev": h}))?;
            let v = self.skill_learn_stage(&json!({"stage": "verify", "input": input, "prev": p}))?;
            let a = self.skill_learn_stage(&json!({"stage": "admit", "input": input, "prev": v}))?;
            let m = self.skill_learn_stage(&json!({"stage": "maintain", "input": input}))?;
            return Ok(json!({"harvest": h, "propose": p, "verify": v, "admit": a, "maintain": m}));
        }
        self.skill_define_workflows()?;
        let r = self.flow_call("start", json!({"workflow": LEARN_WORKFLOW, "input": input, "queue": SKILL_QUEUE}))?;
        self.skill_mark_pending();
        Ok(r)
    }

    fn learn_input_opts(&'s self, input: &JsonValue) -> LearnOpts {
        let mut base = serde_json::to_value(self.skill_learn_opts()).unwrap_or(json!({}));
        let mut patch = input.clone();
        if let Some(m) = patch.as_object_mut() {
            m.remove("scopes");
        }
        merge(&mut base, &patch);
        serde_json::from_value(base).unwrap_or_default()
    }

    /// Successful and failed traces of a scope, newest first.
    fn skill_harvest(&'s self, scope: &str, since: i64) -> Result<Vec<Trace>> {
        let mut out = vec![];
        let r = self.sk_r(
            &format!("?[tid, task, transcript, output, success, at] := *{SKILL_TRACE}{{tid, scope: $s, task, transcript, output, success, at}}, at >= $since :order -at :limit 2000"),
            vec![("s", sv(scope)), ("since", DataValue::from(since))],
        )?;
        for x in r.rows {
            out.push(Trace {
                id: cs(&x[0]).unwrap_or_default(),
                task: cs(&x[1]).unwrap_or_default(),
                transcript: cs(&x[2]).unwrap_or_default(),
                output: cs(&x[3]),
                success: x[4].get_bool().unwrap_or(false),
            });
        }
        // completed workflows that describe their task
        if let Ok(r) = self.sk_r(
            "?[id, output, meta, finished] := *kleos_flow_wf{id, status: 'completed', output, meta, finished}, !is_null(meta), finished >= $since",
            vec![("since", DataValue::from(since))],
        ) {
            for x in r.rows {
                let meta = jv(&x[2]);
                let Some(task) = meta.get("task").and_then(|t| t.as_str()) else { continue };
                if meta.get("scope").and_then(|s| s.as_str()).unwrap_or(DEFAULT_SCOPE) != scope {
                    continue;
                }
                let id = cs(&x[0]).unwrap_or_default();
                let steps: Vec<String> = self
                    .flow_history(&id)
                    .map(|h| h.into_iter().filter(|e| e.ty == "step").map(|e| format!("{}: {}", e.name.unwrap_or_default(), clip(&e.data.to_string(), 300))).collect())
                    .unwrap_or_default();
                let output = jv(&x[1]);
                out.push(Trace {
                    id: format!("flow:{id}"),
                    task: task.to_string(),
                    transcript: steps.join("\n"),
                    output: Some(output.as_str().map(String::from).unwrap_or_else(|| output.to_string())),
                    success: true,
                });
            }
        }
        // `trace` memories of the namespace that is this scope
        if let Ok(r) = self.sk_r("?[id, text] := *kleos_mem{ns: $s, id, text, kind: 'trace' @ 'NOW'}", vec![("s", sv(scope))]) {
            for x in r.rows {
                let text = cs(&x[1]).unwrap_or_default();
                let task = text.lines().next().unwrap_or("").to_string();
                out.push(Trace { id: format!("mem:{}", cs(&x[0]).unwrap_or_default()), task, transcript: text, output: None, success: true });
            }
        }
        Ok(out)
    }

    /// Cluster traces by task similarity: `(members, centroid task)`.
    fn skill_cluster(&'s self, traces: &[Trace], threshold: f64) -> Result<Vec<Vec<usize>>> {
        let emb = self.skill_embedder().map(|e| e.0).or_else(|| self.mem_embedder_name().ok().flatten().filter(|e| crate::embed::is_embedder_enabled(e)));
        let vecs: Option<Vec<Vec<f32>>> = match &emb {
            Some(e) => {
                let texts: Vec<&str> = traces.iter().map(|t| t.task.as_str()).collect();
                let mut all = vec![];
                for c in texts.chunks(64) {
                    all.extend(crate::embed::embed_texts(e, c)?.into_iter().map(|v| v.to_vec()));
                }
                Some(all)
            }
            None => None,
        };
        let words: Vec<BTreeSet<String>> = traces.iter().map(|t| word_set(&t.task)).collect();
        let sim = |i: usize, j: usize| -> f64 {
            match &vecs {
                Some(v) => cos(&v[i], &v[j]),
                // word overlap runs lower than cosine for the same relatedness
                None => jaccard(&words[i], &words[j]) * 1.6,
            }
        };
        let mut clusters: Vec<Vec<usize>> = vec![];
        for i in 0..traces.len() {
            let mut best: Option<(usize, f64)> = None;
            for (ci, c) in clusters.iter().enumerate() {
                let s = c.iter().map(|&j| sim(i, j)).sum::<f64>() / c.len() as f64;
                if s >= threshold && best.map_or(true, |b| s > b.1) {
                    best = Some((ci, s));
                }
            }
            match best {
                Some((ci, _)) => clusters[ci].push(i),
                None => clusters.push(vec![i]),
            }
        }
        Ok(clusters)
    }

    fn skill_trace_rows(&'s self, ids: &[String]) -> Result<Vec<Trace>> {
        let r = self.sk_r(
            &format!("?[tid, task, transcript, output, success] := tid in $ids, *{SKILL_TRACE}{{tid, task, transcript, output, success}}"),
            vec![("ids", str_list(ids))],
        )?;
        let by: BTreeMap<String, Trace> = r
            .rows
            .iter()
            .map(|x| {
                let t = Trace {
                    id: cs(&x[0]).unwrap_or_default(),
                    task: cs(&x[1]).unwrap_or_default(),
                    transcript: cs(&x[2]).unwrap_or_default(),
                    output: cs(&x[3]),
                    success: x[4].get_bool().unwrap_or(false),
                };
                (t.id.clone(), t)
            })
            .collect();
        Ok(ids.iter().filter_map(|i| by.get(i).cloned()).collect())
    }

    /// One stage of `skill.learn` (the workflow's steps call this).
    pub(crate) fn skill_learn_stage(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let stage = arg(args, "stage").ok_or_else(|| miette!("learn_stage needs 'stage'"))?;
        let input = args.get("input").cloned().unwrap_or(json!({}));
        let prev = args.get("prev").cloned().unwrap_or(JsonValue::Null);
        let o = self.learn_input_opts(&input);
        let scopes: Vec<String> = input
            .get("scopes")
            .and_then(|s| s.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .filter(|v: &Vec<String>| !v.is_empty())
            .unwrap_or_else(|| o.scopes.clone());
        match stage {
            "harvest" => {
                let since = now_us() - o.since_days.max(1) * 86_400_000_000;
                let mut clusters_out = vec![];
                let mut stats = json!({"traces": 0, "clusters": 0, "covered": 0, "small": 0});
                for scope in &scopes {
                    let traces = self.skill_harvest(scope, since)?;
                    stats["traces"] = json!(stats["traces"].as_i64().unwrap_or(0) + traces.len() as i64);
                    // flow and memory traces are stored as traces first, so
                    // every later stage reads them by id
                    for t in traces.iter().filter(|t| t.id.starts_with("flow:") || t.id.starts_with("mem:")) {
                        let _ = self.skill_record_trace(&json!({"id": t.id, "task": t.task, "transcript": t.transcript, "output": t.output, "success": t.success, "scope": scope}));
                    }
                    for members in self.skill_cluster(&traces, o.cluster_threshold)? {
                        let succ: Vec<&Trace> = members.iter().map(|&i| &traces[i]).filter(|t| t.success).collect();
                        let fail: Vec<&Trace> = members.iter().map(|&i| &traces[i]).filter(|t| !t.success).collect();
                        if succ.len() < o.min_support.max(2) {
                            stats["small"] = json!(stats["small"].as_i64().unwrap_or(0) + 1);
                            continue;
                        }
                        // covered by a live skill that already works?
                        let probe = succ[0].task.clone();
                        let found = self.skill_find(&json!({"task": probe, "scopes": [scope], "k": 1, "log": false}))?;
                        if let Some(top) = found["skills"].as_array().and_then(|a| a.first()) {
                            let covered = top["score"].as_f64().unwrap_or(0.0) >= 0.8 && top["stats"]["posterior"].as_f64().unwrap_or(0.5) >= 0.6;
                            let used = succ.iter().any(|t| self.skill_trace_used(&t.id).unwrap_or(false));
                            if covered || used {
                                stats["covered"] = json!(stats["covered"].as_i64().unwrap_or(0) + 1);
                                continue;
                            }
                        }
                        let n_hold = ((succ.len() as f64) * o.holdout).round().max(1.0) as usize;
                        let n_hold = n_hold.min(succ.len() - 1);
                        let (train, hold) = succ.split_at(succ.len() - n_hold);
                        clusters_out.push(json!({
                            "scope": scope,
                            "train": train.iter().map(|t| t.id.clone()).collect::<Vec<_>>(),
                            "holdout": hold.iter().map(|t| t.id.clone()).collect::<Vec<_>>(),
                            "failures": fail.iter().take(5).map(|t| t.id.clone()).collect::<Vec<_>>(),
                            "example": probe,
                        }));
                    }
                }
                clusters_out.sort_by_key(|c| std::cmp::Reverse(c["train"].as_array().map_or(0, |a| a.len()) + c["holdout"].as_array().map_or(0, |a| a.len())));
                clusters_out.truncate(o.max_new.max(1));
                stats["clusters"] = json!(clusters_out.len());
                Ok(json!({"clusters": clusters_out, "stats": stats}))
            }
            "propose" => {
                let l = self.llm_of(&input)?;
                let mut out = vec![];
                for c in prev["clusters"].as_array().cloned().unwrap_or_default() {
                    let scope = c["scope"].as_str().unwrap_or(DEFAULT_SCOPE).to_string();
                    let ids = |k: &str| -> Vec<String> { c[k].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default() };
                    let train = self.skill_trace_rows(&ids("train"))?;
                    let fails = self.skill_trace_rows(&ids("failures"))?;
                    match self.skill_draft(&l, &scope, &train, &fails, &ids("holdout")) {
                        Ok(v) => out.push(v),
                        Err(e) => out.push(json!({"scope": scope, "error": e.to_string(), "example": c["example"]})),
                    }
                }
                Ok(json!({"candidates": out}))
            }
            "verify" => {
                let l = self.llm_of(&input)?;
                let mut out = vec![];
                for c in prev["candidates"].as_array().cloned().unwrap_or_default() {
                    let Some(ver) = c["ver"].as_str() else {
                        out.push(c);
                        continue;
                    };
                    if c["status"] == "quarantined" {
                        let mut c = c.clone();
                        c["verdict"] = json!({"admit": false, "why": "quarantined by the scanner"});
                        out.push(c);
                        continue;
                    }
                    let row = self.skill_ver(ver)?.ok_or_else(|| miette!("candidate '{ver}' is gone"))?;
                    let cases: Vec<Case> = self
                        .skill_evals(row["scope"].as_str().unwrap_or_default(), row["name"].as_str().unwrap_or_default(), Some("holdout"))?
                        .iter()
                        .map(Case::from_row)
                        .collect();
                    let with = replay(&l, Some(row["body"].as_str().unwrap_or_default()), &cases, o.parallel);
                    let without = replay(&l, None, &cases, o.parallel);
                    let (sw, sb) = (score(&with), score(&without));
                    let admit = !cases.is_empty() && sw - sb >= o.margin - 1e-9 && sw > 0.0;
                    let mut c = c.clone();
                    c["verdict"] = json!({"admit": admit, "with": sw, "without": sb, "lift": sw - sb, "cases": cases.len(), "margin": o.margin,
                                          "with_results": results_json(&with), "without_results": results_json(&without)});
                    out.push(c);
                }
                Ok(json!({"candidates": out}))
            }
            "admit" => {
                let mut out = vec![];
                for c in prev["candidates"].as_array().cloned().unwrap_or_default() {
                    let Some(ver) = c["ver"].as_str().map(String::from) else {
                        out.push(json!({"error": c.get("error"), "admitted": false}));
                        continue;
                    };
                    let v = &c["verdict"];
                    let detail = json!({"with": v["with"], "without": v["without"], "lift": v["lift"], "cases": v["cases"]});
                    if v["admit"] == true {
                        if o.mode == "auto" {
                            self.skill_activate_ver(&ver, &json!({"human": null, "agent": "kleos:skill.learn", "run": null}), "learned: verified lift")?;
                            out.push(json!({"ver": ver, "name": c["name"], "scope": c["scope"], "admitted": true, "active": true, "verdict": detail}));
                        } else {
                            self.skill_log(c["scope"].as_str().unwrap_or_default(), c["name"].as_str().unwrap_or_default(), "proposal", Some(&ver), &detail)?;
                            out.push(json!({"ver": ver, "name": c["name"], "scope": c["scope"], "admitted": true, "active": false, "proposal": true, "verdict": detail}));
                        }
                    } else {
                        if c["status"] != "quarantined" {
                            self.sk_w(
                                &format!("?[ver, status, at, reason] <- [[$v, 'rejected', $now, $r]] :put {SKILL_VSTAT} {{ver => status, at, reason}}"),
                                vec![("v", sv(&ver)), ("now", DataValue::from(now_us())), ("r", sv(&format!("verification: {detail}")))],
                            )?;
                        }
                        out.push(json!({"ver": ver, "name": c["name"], "scope": c["scope"], "admitted": false, "verdict": detail}));
                    }
                }
                Ok(json!({"results": out, "mode": o.mode}))
            }
            "maintain" => {
                let mut res = vec![];
                for s in &scopes {
                    res.push(self.skill_lifecycle(&json!({"scope": s}))?);
                }
                Ok(JsonValue::Array(res))
            }
            other => bail!("unknown learn stage '{other}'"),
        }
    }

    /// Whether a trace already fed a skill (provenance of a version).
    fn skill_trace_used(&'s self, tid: &str) -> Result<bool> {
        let r = self.sk_r(&format!("?[ver, provenance] := *{SKILL_VER}{{ver, provenance, origin: 'induced'}}"), vec![])?;
        Ok(r.rows.iter().any(|x| jv(&x[1])["traces"].as_array().is_some_and(|a| a.iter().any(|t| t == tid))))
    }

    pub(crate) fn skill_log(&'s self, scope: &str, name: &str, op: &str, ver: Option<&str>, detail: &JsonValue) -> Result<()> {
        self.sk_w(
            &format!(
                "?[scope, name, at, eid, op, ver, from, actor, detail] <- [[$s, $n, $now, $eid, $op, $v, null, $a, $d]] \
                 :put {SKILL_LOG} {{scope, name, at, eid => op, ver, from, actor, detail}}"
            ),
            vec![
                ("s", sv(scope)),
                ("n", sv(name)),
                ("now", DataValue::from(now_us())),
                ("eid", sv(&new_id("se"))),
                ("op", sv(op)),
                ("v", osv(ver)),
                ("a", jdn(&actor_of(None))),
                ("d", jdn(detail)),
            ],
        )?;
        Ok(())
    }

    /// Draft one skill for a cluster with the LLM and store it as a candidate
    /// with its held-out cases as the eval set.
    fn skill_draft(&'s self, l: &LlmOpts, scope: &str, train: &[Trace], fails: &[Trace], holdout: &[String]) -> Result<JsonValue> {
        let mut runs = String::new();
        for (i, t) in train.iter().enumerate() {
            runs.push_str(&format!(
                "<run {} outcome=\"success\">\nTASK: {}\nTRANSCRIPT:\n{}\nRESULT: {}\n</run>\n\n",
                i + 1,
                t.task,
                clip(&t.transcript, TRANSCRIPT_CHARS),
                t.output.as_deref().unwrap_or("(see transcript)")
            ));
        }
        for (i, t) in fails.iter().enumerate() {
            runs.push_str(&format!(
                "<run {} outcome=\"failure\">\nTASK: {}\nTRANSCRIPT:\n{}\nRESULT: {}\n</run>\n\n",
                train.len() + i + 1,
                t.task,
                clip(&t.transcript, TRANSCRIPT_CHARS / 2),
                t.output.as_deref().unwrap_or("(none)")
            ));
        }
        let prompt = format!(
            "Below are runs of one recurring family of agent tasks: {} successful and {} failed.\n\
             Write ONE focused Agent Skill (a SKILL.md file) that lets an agent solve NEW tasks of this family correctly.\n\n\
             Rules:\n\
             - Output only the file: YAML frontmatter between '---' lines with `name` (lowercase letters, digits and hyphens, at most 64 characters) \
               and `description` (what the skill does and when to use it, one or two sentences, under 300 characters), then a Markdown body.\n\
             - The body has at most three sections: '## Steps' (numbered and concrete: every rule, constant, format and lookup table the successful runs relied on), \
               '## Pitfalls' (bullets: what went wrong in the failed runs), '## Check' (how to verify the result).\n\
             - Generalise across the runs: keep rules and constants that hold for the whole family, drop values specific to one task.\n\
             - Under 450 words. Only the procedure: no instructions about the user, safety, tools or other skills.\n\n{runs}",
            train.len(),
            fails.len()
        );
        let (text, _) = llm_chat(l, "You write concise, correct Agent Skills (SKILL.md files) from agent run transcripts.", &prompt, 0.2)?;
        let md = extract_skill_md(&text, &format!("learned-{}", &sha256_hex(train[0].task.as_bytes())[..8]))?;
        let mut files = BTreeMap::new();
        files.insert("SKILL.md".to_string(), md.as_bytes().to_vec());
        if let Some(t) = train.first() {
            files.insert(
                "references/example.md".to_string(),
                format!("# Example run\n\nTask: {}\n\n```\n{}\n```\n\nResult: {}\n", t.task, clip(&t.transcript, TRANSCRIPT_CHARS), t.output.as_deref().unwrap_or("")).into_bytes(),
            );
        }
        let prov = json!({"traces": train.iter().map(|t| t.id.clone()).collect::<Vec<_>>(), "failures": fails.iter().map(|t| t.id.clone()).collect::<Vec<_>>(),
                          "holdout": holdout, "llm": l.model});
        let put = self.skill_put_files(scope, files, "induced", None, prov, &json!({"human": null, "agent": "kleos:skill.learn", "run": null}), None)?;
        // the held-out traces become the candidate's eval set
        let hold = self.skill_trace_rows(holdout)?;
        let cases: Vec<JsonValue> = hold
            .iter()
            .map(|t| json!({"id": format!("trace:{}", t.id), "task": t.task, "expected": t.output, "split": "holdout", "source": "trace"}))
            .collect();
        let train_cases: Vec<JsonValue> = train
            .iter()
            .map(|t| json!({"id": format!("trace:{}", t.id), "task": t.task, "expected": t.output, "split": "train", "source": "trace"}))
            .collect();
        let name = put["name"].as_str().unwrap_or_default().to_string();
        let all: Vec<JsonValue> = cases.into_iter().chain(train_cases).collect();
        if !all.is_empty() {
            self.skill_eval_add(&json!({"name": name, "scope": scope, "cases": all}))?;
        }
        Ok(put)
    }

    /// `proposals {scope?}`: verified candidates waiting for review.
    pub(crate) fn skill_proposals(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let scopes = self.skill_scopes(args)?;
        let r = self.sk_r(
            &format!(
                "?[scope, name, ver, at, detail] := *{SKILL_LOG}{{scope, name, at, op: 'proposal', ver, detail}}, is_in(scope, $sc), \
                 *{SKILL_VSTAT}{{ver, status: 'candidate'}}"
            ),
            vec![("sc", str_list(&scopes))],
        )?;
        Ok(JsonValue::Array(rows_json(&r).into_iter().map(JsonValue::Object).collect()))
    }

    /// `propose {skill_md, files?, scope?, evidence?, eval?}`: an agent proposes
    /// a skill; it is stored as a candidate and, when an eval set and an LLM
    /// are available, verified like a learned one (`verify: true`).
    pub(crate) fn skill_propose(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let mut a = args.clone();
        a["origin"] = json!("proposed");
        if let Some(e) = args.get("evidence") {
            a["provenance"] = json!({"evidence": e});
        }
        a["activate"] = json!(false);
        let put = self.skill_call("put", a)?;
        let mut out = json!({"candidate": put});
        if let Some(cases) = args.get("eval").filter(|c| c.is_array()) {
            self.skill_eval_add(&json!({"name": put["name"], "scope": put["scope"], "cases": cases}))?;
        }
        if args.get("verify").and_then(|v| v.as_bool()) == Some(true) {
            out["verification"] = self.skill_eval_op(&json!({"ver": put["ver"], "baseline": true, "llm": args.get("llm")}))?;
        }
        Ok(out)
    }

    /// `eval {ver | name+scope, split?, baseline?, llm?}`: replay the eval set
    /// with the version (and without any skill, for the baseline).
    pub(crate) fn skill_eval_op(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let row = self.skill_resolve(args)?;
        let l = self.llm_of(args)?;
        let o = self.skill_learn_opts();
        let split = arg(args, "split").unwrap_or("holdout");
        let mut cases: Vec<Case> = self
            .skill_evals(row["scope"].as_str().unwrap_or_default(), row["name"].as_str().unwrap_or_default(), if split == "all" { None } else { Some(split) })?
            .iter()
            .map(Case::from_row)
            .collect();
        if cases.is_empty() && split == "holdout" {
            cases = self.skill_evals(row["scope"].as_str().unwrap_or_default(), row["name"].as_str().unwrap_or_default(), None)?.iter().map(Case::from_row).collect();
        }
        if cases.is_empty() {
            bail!("skill '{}' has no eval cases (`::skill eval_add`)", row["name"].as_str().unwrap_or_default());
        }
        let with = replay(&l, Some(row["body"].as_str().unwrap_or_default()), &cases, o.parallel);
        let mut out = json!({"ver": row["ver"], "name": row["name"], "cases": cases.len(), "score": score(&with), "results": results_json(&with)});
        if args.get("baseline").and_then(|v| v.as_bool()).unwrap_or(true) {
            let without = replay(&l, None, &cases, o.parallel);
            out["baseline"] = json!(score(&without));
            out["lift"] = json!(score(&with) - score(&without));
            out["baseline_results"] = results_json(&without);
        }
        Ok(out)
    }

    // ---------------------------------------------------------- optimize

    /// `optimize {name+scope | ver, rounds?, edit_budget?, llm?, activate?,
    /// wait?}`: a `skill.optimize` run.
    pub(crate) fn skill_optimize(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let row = self.skill_resolve(args)?;
        let mut input = json!({"ver": row["ver"], "rounds": args.get("rounds").and_then(|v| v.as_i64()).unwrap_or(DEFAULT_ROUNDS)});
        for k in ["edit_budget", "llm", "activate"] {
            if let Some(v) = args.get(k).filter(|v| !v.is_null()) {
                input[k] = v.clone();
            }
        }
        self.llm_of(&input)?;
        if args.get("wait").and_then(|v| v.as_bool()) == Some(true) {
            let p = self.skill_optimize_stage(&json!({"stage": "prepare", "input": input}))?;
            let rounds = p["rounds"].as_i64().unwrap_or(0);
            let mut rs = vec![];
            for i in 0..rounds {
                rs.push(self.skill_optimize_stage(&json!({"stage": "round", "run": p["run"], "i": i}))?);
            }
            let f = self.skill_optimize_stage(&json!({"stage": "finish", "run": p["run"]}))?;
            return Ok(json!({"prepare": p, "rounds": rs, "finish": f}));
        }
        self.skill_define_workflows()?;
        let r = self.flow_call("start", json!({"workflow": OPTIMIZE_WORKFLOW, "input": input, "queue": SKILL_QUEUE}))?;
        self.skill_mark_pending();
        Ok(r)
    }

    fn opt_state(&'s self, run: &str) -> Result<JsonValue> {
        let r = self.sk_r(&format!("?[state] := *{SKILL_RUN}{{run: $r, state}}"), vec![("r", sv(run))])?;
        r.rows.first().map(|x| jv(&x[0])).ok_or_else(|| miette!("no optimizer run '{run}'"))
    }

    fn put_opt_state(&'s self, run: &str, scope: &str, name: &str, state: &JsonValue) -> Result<()> {
        let now = now_us();
        self.sk_w(
            &format!("?[run, kind, scope, name, state, at, updated] <- [[$r, 'optimize', $s, $n, $st, $now, $now]] :put {SKILL_RUN} {{run => kind, scope, name, state, at, updated}}"),
            vec![("r", sv(run)), ("s", sv(scope)), ("n", sv(name)), ("st", jdn(state)), ("now", DataValue::from(now))],
        )?;
        Ok(())
    }

    /// One stage of `skill.optimize`. Rounds are idempotent: a round already
    /// recorded in the run state returns its recorded result.
    pub(crate) fn skill_optimize_stage(&'s self, args: &JsonValue) -> Result<JsonValue> {
        let stage = arg(args, "stage").ok_or_else(|| miette!("optimize_stage needs 'stage'"))?;
        match stage {
            "prepare" => {
                let input = args.get("input").cloned().unwrap_or(json!({}));
                let l = self.llm_of(&input)?;
                let row = self.skill_resolve(&input)?;
                let (scope, name) = (row["scope"].as_str().unwrap_or_default().to_string(), row["name"].as_str().unwrap_or_default().to_string());
                let all = self.skill_evals(&scope, &name, None)?;
                let hold: Vec<Case> = all.iter().filter(|c| c["split"] == "holdout").map(Case::from_row).collect();
                let mut train: Vec<Case> = all.iter().filter(|c| c["split"] == "train").map(Case::from_row).collect();
                if hold.len() < 2 {
                    bail!("skill optimize needs at least 2 held-out eval cases for '{name}' (`::skill eval_add` with split 'holdout')");
                }
                if train.is_empty() {
                    // reflection reads the held-out failures too; acceptance
                    // stays strict on the held-out score
                    train = hold.clone();
                }
                let o = self.skill_learn_opts();
                let base = replay(&l, Some(row["body"].as_str().unwrap_or_default()), &hold, o.parallel);
                let tr = replay(&l, Some(row["body"].as_str().unwrap_or_default()), &train, o.parallel);
                let run = new_id("so");
                let state = json!({
                    "base_ver": row["ver"], "best_ver": row["ver"], "best_body": row["body"], "best_score": score(&base), "base_score": score(&base),
                    "train_results": results_json(&tr), "holdout": hold.iter().map(|c| c.id.clone()).collect::<Vec<_>>(),
                    "train": train.iter().map(|c| c.id.clone()).collect::<Vec<_>>(),
                    "rejected": [], "history": [score(&base)], "rounds_done": 0, "accepted": [],
                    "edit_budget": input.get("edit_budget").and_then(|v| v.as_u64()).unwrap_or(DEFAULT_EDIT_BUDGET as u64),
                    "activate": input.get("activate").and_then(|v| v.as_bool()).unwrap_or(true),
                    "llm": input.get("llm"),
                });
                self.put_opt_state(&run, &scope, &name, &state)?;
                Ok(json!({"run": run, "rounds": input.get("rounds").and_then(|v| v.as_i64()).unwrap_or(DEFAULT_ROUNDS), "base_score": score(&base), "ver": row["ver"]}))
            }
            "round" => {
                let run = arg(args, "run").ok_or_else(|| miette!("optimize round needs 'run'"))?.to_string();
                let i = args.get("i").and_then(|v| v.as_i64()).unwrap_or(0);
                let mut st = self.opt_state(&run)?;
                if let Some(done) = st["rounds"].get(i.to_string()) {
                    return Ok(done.clone());
                }
                let l = self.llm_of(&json!({"llm": st["llm"]}))?;
                let best = self.skill_ver(st["best_ver"].as_str().unwrap_or_default())?.ok_or_else(|| miette!("best version is gone"))?;
                let (scope, name) = (best["scope"].as_str().unwrap_or_default().to_string(), best["name"].as_str().unwrap_or_default().to_string());
                let all = self.skill_evals(&scope, &name, None)?;
                let pick = |ids: &JsonValue| -> Vec<Case> {
                    let set: BTreeSet<String> = ids.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
                    all.iter().filter(|c| set.contains(c["case"].as_str().unwrap_or(""))).map(Case::from_row).collect()
                };
                let hold = pick(&st["holdout"]);
                let tasks: BTreeMap<String, String> = all.iter().map(|c| (c["case"].as_str().unwrap_or("").to_string(), c["task"].as_str().unwrap_or("").to_string())).collect();
                let fails: Vec<&JsonValue> = st["train_results"].as_array().map(|a| a.iter().filter(|r| r["ok"] == false).collect()).unwrap_or_default();
                let wins: Vec<&JsonValue> = st["train_results"].as_array().map(|a| a.iter().filter(|r| r["ok"] == true).collect()).unwrap_or_default();
                let body = st["best_body"].as_str().unwrap_or_default().to_string();
                let budget = st["edit_budget"].as_u64().unwrap_or(DEFAULT_EDIT_BUDGET as u64) as usize;
                let mut fail_txt = String::new();
                for f in fails.iter().take(8) {
                    let c = f["case"].as_str().unwrap_or("");
                    fail_txt.push_str(&format!("- TASK: {}\n  AGENT ANSWER: {}\n  EXPECTED: {}\n", clip(tasks.get(c).map(|s| s.as_str()).unwrap_or(""), 600), f["answer"].as_str().unwrap_or(""), f["expected"].as_str().unwrap_or("(judged)")));
                }
                let mut win_txt = String::new();
                for w in wins.iter().take(3) {
                    let c = w["case"].as_str().unwrap_or("");
                    win_txt.push_str(&format!("- TASK: {}\n  ANSWER: {}\n", clip(tasks.get(c).map(|s| s.as_str()).unwrap_or(""), 300), w["answer"].as_str().unwrap_or("")));
                }
                let rejected: Vec<String> = st["rejected"].as_array().map(|a| a.iter().map(|x| x.to_string()).collect()).unwrap_or_default();
                let prompt = format!(
                    "You improve an Agent Skill with small, bounded edits. Never rewrite it.\n\n\
                     Skill body (numbered lines):\n{}\n\n\
                     Held-out score history (fraction of tasks solved): {}\n\n\
                     Failed tasks (the agent followed the skill and got these wrong):\n{}\n\
                     Solved tasks (keep these working):\n{}\n\
                     Edits already tried and rejected (do not repeat them):\n{}\n\n\
                     Reflect on what in the skill caused the failures (a missing rule, a wrong constant, an ambiguous step, a missing check), \
                     then propose at most {budget} edits as a JSON array. Each edit is one of:\n\
                     {{\"op\": \"replace\", \"line\": N, \"text\": \"...\"}}\n{{\"op\": \"insert\", \"after\": N, \"text\": \"...\"}} (N = 0 inserts at the top)\n{{\"op\": \"delete\", \"line\": N}}\n\
                     Each text is under 300 characters. Output only the JSON array.",
                    numbered(&body),
                    st["history"],
                    if fail_txt.is_empty() { "(none on the training cases)\n".to_string() } else { fail_txt },
                    if win_txt.is_empty() { "(none)\n".to_string() } else { win_txt },
                    if rejected.is_empty() { "(none)".to_string() } else { rejected.iter().rev().take(REJECTED_BUFFER).cloned().collect::<Vec<_>>().join("\n") },
                );
                let (text, _) = llm_chat(&l, "You are a careful optimizer of procedural instructions for AI agents.", &prompt, 0.7)?;
                let edits = extract_json(&text).and_then(|v| if v.is_array() { v.as_array().cloned() } else { v.get("edits").and_then(|e| e.as_array()).cloned() }).unwrap_or_default();
                let o = self.skill_learn_opts();
                let result = match apply_edits(&body, &edits, budget) {
                    Err(e) => json!({"round": i, "accepted": false, "why": format!("no usable edits: {e}")}),
                    Ok((new_body, applied)) => {
                        let rs = replay(&l, Some(&new_body), &hold, o.parallel);
                        let s = score(&rs);
                        let best_score = st["best_score"].as_f64().unwrap_or(0.0);
                        if s > best_score + 1e-9 {
                            let md = self.skill_file(&best["files"], "SKILL.md")?;
                            let doc = parse_skill_md(&String::from_utf8_lossy(&md))?;
                            let mut files = self.skill_files(&best["files"])?;
                            files.insert("SKILL.md".into(), with_body(&doc, &new_body).into_bytes());
                            let prov = json!({"optimizer": "kleos-skillopt/1", "run": run, "round": i, "holdout_score": s, "previous_score": best_score, "llm": l.model});
                            let put = self.skill_put_files(&scope, files, "optimized", best["ver"].as_str(), prov,
                                &json!({"human": null, "agent": "kleos:skill.optimize", "run": run}), Some(JsonValue::Array(applied.clone())))?;
                            if put["status"] == "quarantined" {
                                st["rejected"].as_array_mut().map(|a| a.push(json!({"edits": applied, "why": "scan"})));
                                json!({"round": i, "accepted": false, "why": "the edited version failed the scan", "score": s})
                            } else {
                                // the next round reflects on the new best's training failures
                                let tr = replay(&l, Some(&new_body), &pick(&st["train"]), o.parallel);
                                st["train_results"] = results_json(&tr);
                                st["best_ver"] = put["ver"].clone();
                                st["best_body"] = json!(new_body);
                                st["best_score"] = json!(s);
                                st["accepted"].as_array_mut().map(|a| a.push(put["ver"].clone()));
                                json!({"round": i, "accepted": true, "score": s, "previous": best_score, "ver": put["ver"], "edits": applied})
                            }
                        } else {
                            if let Some(a) = st["rejected"].as_array_mut() {
                                a.push(json!({"edits": applied, "score": s}));
                                let n = a.len();
                                if n > REJECTED_BUFFER {
                                    a.drain(..n - REJECTED_BUFFER);
                                }
                            }
                            json!({"round": i, "accepted": false, "score": s, "best": best_score, "edits": applied})
                        }
                    }
                };
                let bs = st["best_score"].clone();
                if let Some(h) = st["history"].as_array_mut() {
                    h.push(bs);
                }
                if !st["rounds"].is_object() {
                    st["rounds"] = json!({});
                }
                st["rounds"][i.to_string()] = result.clone();
                st["rounds_done"] = json!(i + 1);
                self.put_opt_state(&run, &scope, &name, &st)?;
                Ok(result)
            }
            "finish" => {
                let run = arg(args, "run").ok_or_else(|| miette!("optimize finish needs 'run'"))?;
                let st = self.opt_state(run)?;
                if let Some(f) = st.get("finished").filter(|f| !f.is_null()) {
                    return Ok(f.clone());
                }
                let (base, best) = (st["base_ver"].as_str().unwrap_or_default().to_string(), st["best_ver"].as_str().unwrap_or_default().to_string());
                let improved = base != best && st["best_score"].as_f64() > st["base_score"].as_f64();
                let mut out = json!({"run": run, "base_ver": base, "best_ver": best, "base_score": st["base_score"], "best_score": st["best_score"],
                                     "improved": improved, "accepted": st["accepted"], "history": st["history"]});
                if improved && st["activate"] != false {
                    let row = self.skill_ver(&best)?.ok_or_else(|| miette!("best version is gone"))?;
                    let base_row = self.skill_ver(&base)?;
                    // supersede only what is live (or the original candidate)
                    let live = self.skill_row(row["scope"].as_str().unwrap_or_default(), row["name"].as_str().unwrap_or_default())?.and_then(|r| r.get("ver").cloned());
                    if live.as_ref().and_then(|v| v.as_str()) == Some(base.as_str()) || base_row.is_some_and(|b| b["status"] == "candidate") {
                        out["activated"] = self.skill_activate_ver(&best, &json!({"human": null, "agent": "kleos:skill.optimize", "run": run}), "optimized: held-out win")?;
                    }
                }
                let mut st2 = st.clone();
                st2["finished"] = out.clone();
                let row = self.skill_ver(&best)?.unwrap_or_default();
                self.put_opt_state(run, row.get("scope").and_then(|v| v.as_str()).unwrap_or(""), row.get("name").and_then(|v| v.as_str()).unwrap_or(""), &st2)?;
                Ok(out)
            }
            other => bail!("unknown optimize stage '{other}'"),
        }
    }
}

/// The in-process runner state of the skill workflows.
#[derive(Default)]
pub struct SkillState {
    pub(crate) pending: std::sync::atomic::AtomicBool,
    pub(crate) running: std::sync::atomic::AtomicBool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edits_are_bounded_and_bottom_up() {
        let body = "a\nb\nc\n";
        let (b, applied) = apply_edits(
            body,
            &[json!({"op": "replace", "line": 1, "text": "A"}), json!({"op": "insert", "after": 3, "text": "d"}), json!({"op": "delete", "line": 2})],
            3,
        )
        .unwrap();
        assert_eq!(b, "A\nc\nd\n");
        assert_eq!(applied.len(), 3);
        let (b, _) = apply_edits(body, &[json!({"op": "insert", "after": 0, "text": "top"}), json!({"op": "replace", "line": 3, "text": "C"})], 1).unwrap();
        assert_eq!(b, "top\na\nb\nc\n");
        assert!(apply_edits(body, &[json!({"op": "replace", "line": 9, "text": "x"})], 3).is_err());
    }

    #[test]
    fn answers_and_json() {
        assert_eq!(final_answer("work\nANSWER: INV-42"), "INV-42");
        assert_eq!(extract_json("sure:\n```json\n[{\"op\":\"delete\",\"line\":1}]\n```").unwrap()[0]["op"], "delete");
        assert_eq!(slug("My Skill!! v2", "x"), "my-skill-v2");
        let md = extract_skill_md("```markdown\n---\nname: Bad Name\ndescription: does things\n---\nbody\n```", "fallback").unwrap();
        assert!(md.contains("name: bad-name"));
    }
}
