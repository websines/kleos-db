/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Built-in mode of durable workflows (docs/specs/durable-workflows.md,
//! "Built-in mode"): JSON step graphs run inside kleos.
//!
//! A run walks the graph from the top every time the workflow is claimed.
//! Each command step has a key (its path in the graph); a key with an event
//! in the history replays the recorded result, the first key without one
//! does the work and records it. Control flow (`if`, `loop`, `for_each`,
//! `fan_out`, `set`, `return`) is recomputed from recorded results, so it
//! takes the same path on every replay.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use miette::{bail, miette, Result};
use serde_json::{json, Value as JsonValue};

use crate::data::value::DataValue;
use crate::flow::{ev, now_us, Claimed, Event, SuspendOpts, TimerSpec, FINISHED};
use crate::runtime::db::{Db, ScriptMutability};
use crate::storage::Storage;

const ACTIONS: [&str; 7] = ["query", "remember", "recall", "embed", "decide", "http", "consolidate"];
const KINDS: [&str; 17] = [
    "query",
    "remember",
    "recall",
    "consolidate",
    "embed",
    "decide",
    "http",
    "sleep",
    "wait_signal",
    "child",
    "if",
    "loop",
    "for_each",
    "fan_out",
    "set",
    "return",
    "continue_as_new",
];
const STEP_FIELDS: [&str; 4] = ["name", "retry", "on_error", "timeout_ms"];
const DEFAULT_MAX_LOOP: i64 = 1000;

fn is_ident(s: &str) -> bool {
    let mut c = s.chars();
    matches!(c.next(), Some(x) if x.is_ascii_alphabetic() || x == '_')
        && c.all(|x| x.is_ascii_alphanumeric() || x == '_')
}

/// The kind and arguments of one step.
fn kind_of(step: &JsonValue) -> Result<(&str, &JsonValue)> {
    let obj = step.as_object().ok_or_else(|| miette!("flow graph: a step must be an object, got {step}"))?;
    let kinds: Vec<(&String, &JsonValue)> = obj.iter().filter(|(k, _)| !STEP_FIELDS.contains(&k.as_str())).collect();
    match kinds.as_slice() {
        [(k, v)] if KINDS.contains(&k.as_str()) || crate::flow::external_action(k).is_some() => Ok((k.as_str(), v)),
        [(k, _)] => bail!("flow graph: unknown step kind '{k}' (expected one of {})", KINDS.join(", ")),
        _ => bail!("flow graph: a step needs exactly one kind, got {step}"),
    }
}

/// Check a graph's shape before it is stored.
pub(crate) fn validate(graph: &JsonValue) -> Result<()> {
    let steps = graph
        .get("steps")
        .ok_or_else(|| miette!("flow graph needs 'steps': [...]"))?;
    validate_list(steps)
}

fn validate_list(steps: &JsonValue) -> Result<()> {
    let list = steps.as_array().ok_or_else(|| miette!("flow graph: 'steps' must be a list"))?;
    for step in list {
        let (kind, args) = kind_of(step)?;
        if let Some(n) = step.get("name") {
            let n = n.as_str().ok_or_else(|| miette!("flow graph: a step name must be a string"))?;
            if !is_ident(n) || n == "input" || n == "i" {
                bail!("flow graph: step name '{n}' must be an identifier other than 'input' and 'i'");
            }
        }
        match kind {
            "if" => {
                need_str(args, "cond", kind)?;
                validate_list(args.get("then").unwrap_or(&json!([])))?;
                if let Some(e) = args.get("else") {
                    validate_list(e)?;
                }
            }
            "loop" => {
                need_str(args, "while", kind)?;
                validate_list(args.get("body").ok_or_else(|| miette!("flow graph: loop needs 'body'"))?)?;
            }
            "for_each" => {
                if args.get("in").is_none() {
                    bail!("flow graph: for_each needs 'in'");
                }
                let a = need_str(args, "as", kind)?;
                if !is_ident(a) {
                    bail!("flow graph: for_each 'as' must be an identifier");
                }
                validate_list(args.get("body").ok_or_else(|| miette!("flow graph: for_each needs 'body'"))?)?;
            }
            "fan_out" => {
                let b = args
                    .get("branches")
                    .and_then(|x| x.as_array())
                    .ok_or_else(|| miette!("flow graph: fan_out needs 'branches': [[...], ...]"))?;
                for br in b {
                    validate_list(br)?;
                }
            }
            "wait_signal" => {
                need_str(args, "name", kind)?;
            }
            "child" => {
                need_str(args, "workflow", kind)?;
            }
            "sleep" => {
                if args.get("ms").and_then(|x| x.as_i64()).is_none() {
                    bail!("flow graph: sleep needs 'ms'");
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn need_str<'a>(args: &'a JsonValue, k: &str, kind: &str) -> Result<&'a str> {
    args.get(k)
        .and_then(|x| x.as_str())
        .ok_or_else(|| miette!("flow graph: {kind} needs '{k}' (a string)"))
}

/// Where a run stopped.
enum Out {
    Val(JsonValue),
    Ret(JsonValue),
    /// Continue as new with this input (a fresh history, same id).
    Continue(JsonValue),
    Wait(Vec<Wait>),
}

enum Wait {
    Timer(String, i64, &'static str),
    Signal(String),
    Child(String),
}

type Env = BTreeMap<String, JsonValue>;

struct Run<'s, S: Storage<'s>> {
    db: &'s Db<S>,
    id: String,
    token: i64,
    hist: BTreeMap<String, Vec<Event>>,
    visited: Mutex<BTreeSet<String>>,
}

fn truthy(v: &JsonValue) -> bool {
    match v {
        JsonValue::Null => false,
        JsonValue::Bool(b) => *b,
        JsonValue::Number(n) => n.as_f64().map_or(false, |x| x != 0.0),
        JsonValue::String(s) => !s.is_empty(),
        JsonValue::Array(a) => !a.is_empty(),
        JsonValue::Object(o) => !o.is_empty(),
    }
}

/// `lease lost`, `canceled` and non-determinism errors end the run as they
/// are; anything else is the step's failure.
fn is_control(e: &miette::Report) -> bool {
    let m = e.to_string();
    m.starts_with("lease lost") || m.starts_with("canceled") || m.starts_with("non-determinism")
}

fn retry_policy(step: &JsonValue) -> (i64, i64, f64, i64) {
    let r = step.get("retry");
    let g = |k: &str| r.and_then(|x| x.get(k));
    (
        g("max_attempts").and_then(|x| x.as_i64()).unwrap_or(3).max(1),
        g("initial_ms").and_then(|x| x.as_i64()).unwrap_or(1000).max(0),
        g("backoff").and_then(|x| x.as_f64()).unwrap_or(2.0).max(1.0),
        g("max_ms").and_then(|x| x.as_i64()).unwrap_or(60_000).max(0),
    )
}

impl<'s, S: Storage<'s>> Run<'s, S> {
    fn visit(&self, key: &str) {
        self.visited.lock().unwrap().insert(key.to_string());
    }

    fn events(&self, key: &str) -> &[Event] {
        self.hist.get(key).map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn record(&self, ty: &str, key: &str, name: &str, data: JsonValue) -> Result<()> {
        self.db.flow_record_evs(&self.id, self.token, &[ev(ty, Some(key), Some(name), data)], true)
    }

    /// Check that the history's events under `key` belong to a command of
    /// this name and one of these types.
    fn check(&self, key: &str, name: &str, types: &[&str]) -> Result<()> {
        for e in self.events(key) {
            if !types.contains(&e.ty.as_str()) || e.name.as_deref() != Some(name) {
                bail!(
                    "non-determinism: workflow '{}' key '{key}' was a '{}' named '{}' in the history, now '{name}'",
                    self.id,
                    e.ty,
                    e.name.clone().unwrap_or_default()
                );
            }
        }
        Ok(())
    }

    fn eval(&self, expr: &str, env: &Env) -> Result<JsonValue> {
        let params: BTreeMap<String, DataValue> = env.iter().map(|(k, v)| (k.clone(), DataValue::from(v))).collect();
        let r = self
            .db
            .run_script(&format!("?[v] := v = {expr}"), params, ScriptMutability::Immutable)
            .map_err(|e| miette!("flow expression '{expr}': {e}"))?;
        Ok(r.rows.first().map(|x| JsonValue::from(&x[0])).unwrap_or(JsonValue::Null))
    }

    fn lookup(&self, path: &str, env: &Env) -> Option<JsonValue> {
        let mut parts = path.split('.');
        let mut cur = env.get(parts.next()?)?.clone();
        for seg in parts {
            cur = match (&cur, seg.parse::<usize>()) {
                (JsonValue::Array(a), Ok(i)) => a.get(i)?.clone(),
                (JsonValue::Object(o), _) => o.get(seg)?.clone(),
                _ => return None,
            };
        }
        Some(cur)
    }

    /// Arguments: `"$name.path"` is a named result, `"=expr"` a CozoScript
    /// expression, anything else a literal.
    fn resolve(&self, v: &JsonValue, env: &Env) -> Result<JsonValue> {
        Ok(match v {
            JsonValue::String(s) if s.starts_with('=') => self.eval(&s[1..], env)?,
            JsonValue::String(s)
                if s.starts_with('$')
                    && s[1..].split('.').all(|p| is_ident(p) || p.parse::<usize>().is_ok()) =>
            {
                self.lookup(&s[1..], env)
                    .ok_or_else(|| miette!("flow graph: '{s}' is not defined"))?
            }
            JsonValue::Array(a) => JsonValue::Array(a.iter().map(|x| self.resolve(x, env)).collect::<Result<_>>()?),
            JsonValue::Object(o) => JsonValue::Object(
                o.iter()
                    .map(|(k, x)| Ok((k.clone(), self.resolve(x, env)?)))
                    .collect::<Result<_>>()?,
            ),
            other => other.clone(),
        })
    }

    /// An expression field (`cond`, `while`, `return`, `set`): a string is
    /// evaluated (`$x.path` looked up), anything else is a literal.
    fn expr(&self, v: &JsonValue, env: &Env) -> Result<JsonValue> {
        match v {
            JsonValue::String(s) if s.starts_with('$') && s.contains('.') => self.resolve(v, env),
            JsonValue::String(s) => self.eval(s.strip_prefix('=').unwrap_or(s), env),
            other => self.resolve(other, env),
        }
    }

    fn exec_list(&self, steps: &[JsonValue], prefix: &str, env: &mut Env) -> Result<Out> {
        let mut last = JsonValue::Null;
        for (n, step) in steps.iter().enumerate() {
            let key = format!("{prefix}s{n}");
            match self.exec_step(step, &key, env)? {
                Out::Val(v) => {
                    if let Some(name) = step.get("name").and_then(|x| x.as_str()) {
                        env.insert(name.to_string(), v.clone());
                    }
                    last = v;
                }
                other => return Ok(other),
            }
        }
        Ok(Out::Val(last))
    }

    fn exec_step(&self, step: &JsonValue, key: &str, env: &mut Env) -> Result<Out> {
        let (kind, args) = kind_of(step)?;
        let name = step.get("name").and_then(|x| x.as_str()).unwrap_or(kind).to_string();
        match kind {
            k if ACTIONS.contains(&k) || crate::flow::external_action(k).is_some() => {
                self.action(step, key, &name, k, args, env)
            }
            "sleep" => {
                self.visit(key);
                self.check(key, &name, &["timer", "timer_fired"])?;
                let evs = self.events(key);
                if evs.iter().any(|e| e.ty == "timer_fired") {
                    return Ok(Out::Val(JsonValue::Null));
                }
                let due = match evs.iter().find(|e| e.ty == "timer") {
                    Some(e) => e.data.get("due").and_then(|x| x.as_i64()).unwrap_or(0),
                    None => {
                        let ms = self.resolve(args.get("ms").unwrap_or(&json!(0)), env)?.as_i64().unwrap_or(0);
                        let due = now_us() + ms.max(0) * 1000;
                        self.record("timer", key, &name, json!({"due": due}))?;
                        due
                    }
                };
                if now_us() >= due {
                    self.record("timer_fired", key, &name, JsonValue::Null)?;
                    Ok(Out::Val(JsonValue::Null))
                } else {
                    Ok(Out::Wait(vec![Wait::Timer(key.to_string(), due, "sleep")]))
                }
            }
            "wait_signal" => {
                self.visit(key);
                let sig = need_str(args, "name", kind)?.to_string();
                // events are named after the signal (the inbox's key)
                let label = sig.clone();
                self.check(key, &label, &["wait", "signal", "signal_timeout"])?;
                let evs = self.events(key);
                if let Some(e) = evs.iter().find(|e| e.ty == "signal") {
                    return Ok(Out::Val(e.data.get("payload").cloned().unwrap_or(JsonValue::Null)));
                }
                if evs.iter().any(|e| e.ty == "signal_timeout") {
                    return Ok(Out::Val(JsonValue::Null));
                }
                let due = match evs.iter().find(|e| e.ty == "wait") {
                    Some(e) => e.data.get("due").and_then(|x| x.as_i64()),
                    None => {
                        let due = match args.get("timeout_ms") {
                            Some(t) => self.resolve(t, env)?.as_i64().map(|ms| now_us() + ms.max(0) * 1000),
                            None => None,
                        };
                        self.record("wait", key, &label, json!({"signal": sig, "due": due}))?;
                        due
                    }
                };
                // the inbox is keyed by signal name; the event by the step's label
                let got = self.db.flow_consume_signal(&self.id, self.token, key, &sig)?;
                if let Some(p) = got {
                    return Ok(Out::Val(p));
                }
                if due.map_or(false, |d| now_us() >= d) {
                    self.record("signal_timeout", key, &label, JsonValue::Null)?;
                    return Ok(Out::Val(JsonValue::Null));
                }
                let mut w = vec![Wait::Signal(sig)];
                if let Some(d) = due {
                    w.push(Wait::Timer(key.to_string(), d, "signal_timeout"));
                }
                Ok(Out::Wait(w))
            }
            "child" => {
                self.visit(key);
                let workflow = need_str(args, "workflow", kind)?.to_string();
                self.check(key, &workflow, &["child", "child_done"])?;
                let evs = self.events(key);
                let done = match evs.iter().find(|e| e.ty == "child_done") {
                    Some(e) => e.data.clone(),
                    None => {
                        let input = self.resolve(args.get("input").unwrap_or(&JsonValue::Null), env)?;
                        let cid = self.db.flow_start_child(
                            &self.id,
                            self.token,
                            key,
                            &workflow,
                            &input,
                            args.get("id").and_then(|x| x.as_str()),
                            args.get("version").and_then(|x| x.as_i64()),
                            args.get("queue").and_then(|x| x.as_str()),
                        )?;
                        let w = self.db.flow_wf(&cid)?.ok_or_else(|| miette!("child '{cid}' vanished"))?;
                        let st = w["status"].as_str().unwrap_or_default().to_string();
                        if !FINISHED.contains(&st.as_str()) {
                            return Ok(Out::Wait(vec![Wait::Child(cid)]));
                        }
                        let d = json!({"id": cid, "status": st, "output": w.get("output"), "error": w.get("error")});
                        self.record("child_done", key, &workflow, d.clone())?;
                        d
                    }
                };
                if done.get("status").and_then(|x| x.as_str()) == Some("completed") {
                    Ok(Out::Val(done.get("output").cloned().unwrap_or(JsonValue::Null)))
                } else if step.get("on_error").and_then(|x| x.as_str()) == Some("continue") {
                    Ok(Out::Val(JsonValue::Null))
                } else {
                    bail!(
                        "child workflow '{}' {}: {}",
                        done.get("id").and_then(|x| x.as_str()).unwrap_or_default(),
                        done.get("status").and_then(|x| x.as_str()).unwrap_or("failed"),
                        done.get("error").and_then(|x| x.as_str()).unwrap_or_default()
                    )
                }
            }
            "if" => {
                let cond = self.expr(args.get("cond").unwrap_or(&JsonValue::Null), env)?;
                let (branch, tag) = if truthy(&cond) {
                    (args.get("then"), "then")
                } else {
                    (args.get("else"), "else")
                };
                let list = branch.and_then(|x| x.as_array()).cloned().unwrap_or_default();
                self.exec_list(&list, &format!("{key}.{tag}."), env)
            }
            "loop" => {
                let max = args.get("max").and_then(|x| x.as_i64()).unwrap_or(DEFAULT_MAX_LOOP);
                let body = args.get("body").and_then(|x| x.as_array()).cloned().unwrap_or_default();
                let mut last = JsonValue::Null;
                let saved = env.get("i").cloned();
                let mut i = 0;
                while i < max {
                    env.insert("i".to_string(), json!(i));
                    if !truthy(&self.expr(args.get("while").unwrap_or(&JsonValue::Null), env)?) {
                        break;
                    }
                    match self.exec_list(&body, &format!("{key}.i{i}."), env)? {
                        Out::Val(v) => last = v,
                        other => return Ok(other),
                    }
                    i += 1;
                }
                match saved {
                    Some(v) => env.insert("i".to_string(), v),
                    None => env.remove("i"),
                };
                Ok(Out::Val(last))
            }
            "for_each" => {
                let items = self.resolve(args.get("in").unwrap_or(&JsonValue::Null), env)?;
                let items = match items.get("rows") {
                    Some(r) => r.clone(),
                    None => items,
                };
                let items = items
                    .as_array()
                    .cloned()
                    .ok_or_else(|| miette!("flow graph: for_each 'in' must be a list"))?;
                let var = need_str(args, "as", kind)?.to_string();
                let body = args.get("body").and_then(|x| x.as_array()).cloned().unwrap_or_default();
                let mut out = vec![];
                for (n, item) in items.into_iter().enumerate() {
                    env.insert(var.clone(), item);
                    match self.exec_list(&body, &format!("{key}.e{n}."), env)? {
                        Out::Val(v) => out.push(v),
                        other => return Ok(other),
                    }
                }
                env.remove(&var);
                Ok(Out::Val(JsonValue::Array(out)))
            }
            "fan_out" => {
                let branches = args.get("branches").and_then(|x| x.as_array()).cloned().unwrap_or_default();
                let results: Vec<Result<Out>> = std::thread::scope(|sc| {
                    let handles: Vec<_> = branches
                        .iter()
                        .enumerate()
                        .map(|(n, br)| {
                            let mut benv = env.clone();
                            let prefix = format!("{key}.b{n}.");
                            let list = br.as_array().cloned().unwrap_or_default();
                            sc.spawn(move || self.exec_list(&list, &prefix, &mut benv))
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| h.join().unwrap_or_else(|_| Err(miette!("flow fan_out: a branch panicked"))))
                        .collect()
                });
                let mut vals = vec![];
                let mut waits = vec![];
                for r in results {
                    match r? {
                        Out::Val(v) | Out::Ret(v) => vals.push(v),
                        Out::Continue(_) => bail!("flow graph: continue_as_new cannot run inside fan_out"),
                        Out::Wait(w) => waits.extend(w),
                    }
                }
                if waits.is_empty() {
                    Ok(Out::Val(JsonValue::Array(vals)))
                } else {
                    Ok(Out::Wait(waits))
                }
            }
            "set" => Ok(Out::Val(self.expr(args, env)?)),
            "return" => Ok(Out::Ret(self.expr(args, env)?)),
            "continue_as_new" => Ok(Out::Continue(self.expr(args, env)?)),
            other => bail!("flow graph: unknown step kind '{other}'"),
        }
    }

    fn action(&self, step: &JsonValue, key: &str, name: &str, kind: &str, args: &JsonValue, env: &Env) -> Result<Out> {
        self.visit(key);
        self.check(key, name, &["step", "step_failed"])?;
        let evs = self.events(key);
        if let Some(e) = evs.iter().find(|e| e.ty == "step") {
            return Ok(Out::Val(e.data.clone()));
        }
        let on_error_continue = step.get("on_error").and_then(|x| x.as_str()) == Some("continue");
        let fails: Vec<&Event> = evs.iter().filter(|e| e.ty == "step_failed").collect();
        if let Some(last) = fails.last() {
            let err = last.data.get("error").and_then(|x| x.as_str()).unwrap_or("failed").to_string();
            if last.data.get("final").and_then(|x| x.as_bool()) == Some(true) {
                return if on_error_continue {
                    Ok(Out::Val(JsonValue::Null))
                } else {
                    bail!("step '{key}' ({name}) failed: {err}")
                };
            }
            if let Some(ra) = last.data.get("retry_at").and_then(|x| x.as_i64()) {
                if ra > now_us() {
                    return Ok(Out::Wait(vec![Wait::Timer(format!("{key}#retry"), ra, "retry")]));
                }
            }
        }
        let resolved = self.resolve(args, env)?;
        let mut action = serde_json::Map::new();
        action.insert(kind.to_string(), resolved);
        match self
            .db
            .flow_step(&self.id, self.token, key, Some(name), Some(&JsonValue::Object(action)), None)
        {
            Ok((v, _)) => Ok(Out::Val(v)),
            Err(e) if is_control(&e) => Err(e),
            Err(e) => {
                let (max, initial, backoff, max_ms) = retry_policy(step);
                let attempt = fails.len() as i64 + 1;
                let msg = e.to_string();
                if attempt < max {
                    let delay = ((initial as f64) * backoff.powi(attempt as i32 - 1)).min(max_ms as f64) as i64;
                    let retry_at = now_us() + delay * 1000;
                    self.record(
                        "step_failed",
                        key,
                        name,
                        json!({"attempt": attempt, "error": msg, "retry_at": retry_at, "final": false}),
                    )?;
                    Ok(Out::Wait(vec![Wait::Timer(format!("{key}#retry"), retry_at, "retry")]))
                } else {
                    self.record("step_failed", key, name, json!({"attempt": attempt, "error": msg, "final": true}))?;
                    if on_error_continue {
                        Ok(Out::Val(JsonValue::Null))
                    } else {
                        bail!("step '{key}' ({name}) failed: {msg}")
                    }
                }
            }
        }
    }
}

/// Run one claimed built-in workflow until it finishes or waits. Returns
/// the status it was left in (`completed`, `failed`, `waiting`,
/// `canceled`, or `lease_lost`).
pub(crate) fn run<'s, S: Storage<'s>>(db: &'s Db<S>, c: Claimed) -> Result<String> {
    let id = c.id.clone();
    let token = c.token;
    if c.cancel {
        db.flow_finish(&id, token, "canceled", &JsonValue::Null, Some("canceled"))?;
        return Ok("canceled".to_string());
    }
    let graph = db.flow_graph(&c.name, c.version)?;
    let mut hist: BTreeMap<String, Vec<Event>> = BTreeMap::new();
    for e in &c.history {
        if let Some(k) = &e.key {
            hist.entry(k.clone()).or_default().push(e.clone());
        }
    }
    let run = Run {
        db,
        id: id.clone(),
        token,
        hist,
        visited: Mutex::new(BTreeSet::new()),
    };
    let mut env: Env = BTreeMap::new();
    env.insert("input".to_string(), c.input.clone());
    let steps = graph.get("steps").and_then(|x| x.as_array()).cloned().unwrap_or_default();
    let out = run.exec_list(&steps, "", &mut env);
    let settle = |e: miette::Report| -> Result<String> {
        let m = e.to_string();
        let busy = {
            let d = format!("{e:?}").to_ascii_lowercase();
            ["busy", "locked", "conflict", "try again"].iter().any(|k| d.contains(k))
        };
        if m.starts_with("lease lost") {
            Ok("lease_lost".to_string())
        } else if busy {
            // the store was busy, not the workflow wrong: run it again soon
            // (if even this write is refused, the lease runs out instead)
            let _ = db.flow_call("fail", json!({"id": id, "token": token, "error": m, "final": false, "retry_in_ms": 200}));
            Ok("waiting".to_string())
        } else if m.starts_with("canceled") {
            db.flow_finish(&id, token, "canceled", &JsonValue::Null, Some("canceled"))?;
            Ok("canceled".to_string())
        } else if m.starts_with("non-determinism") {
            db.flow_call("fail", json!({"id": id, "token": token, "error": m, "final": false}))?;
            Ok("waiting".to_string())
        } else {
            match db.flow_finish(&id, token, "failed", &JsonValue::Null, Some(&m)) {
                Ok(_) => Ok("failed".to_string()),
                Err(e2) if e2.to_string().starts_with("lease lost") => Ok("lease_lost".to_string()),
                Err(e2) => Err(e2),
            }
        }
    };
    match out {
        Ok(Out::Val(v)) | Ok(Out::Ret(v)) => {
            let visited = run.visited.lock().unwrap().clone();
            let missed: Vec<&String> = run.hist.keys().filter(|k| !visited.contains(*k)).collect();
            if let Some(k) = missed.first() {
                return settle(miette!(
                    "non-determinism: workflow '{id}' finished without reaching recorded command '{k}'"
                ));
            }
            match db.flow_finish(&id, token, "completed", &v, None) {
                Ok(_) => Ok("completed".to_string()),
                Err(e) => settle(e),
            }
        }
        Ok(Out::Continue(input)) => match db.flow_continue(&id, token, &input) {
            Ok(_) => Ok("continued".to_string()),
            Err(e) => settle(e),
        },
        Ok(Out::Wait(ws)) => {
            let mut o = SuspendOpts {
                id: id.clone(),
                token,
                ..Default::default()
            };
            for w in ws {
                match w {
                    Wait::Timer(key, due, kind) => o.timers.push(TimerSpec {
                        key,
                        due,
                        kind: Some(kind.to_string()),
                    }),
                    Wait::Signal(n) => o.signals.push(n),
                    Wait::Child(c) => o.children.push(c),
                }
            }
            match db.flow_suspend(&o) {
                Ok(_) => Ok("waiting".to_string()),
                Err(e) => settle(e),
            }
        }
        Err(e) => settle(e),
    }
}
