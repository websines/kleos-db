/*
 * Copyright 2026, The kleos authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The skill library as a principal (docs/specs/skills.md, "Tenancy"):
//! `::principal skill {as, op, …}` / `memory_call_as(ctx, "skill", {op, …})`.
//!
//! - Skills live in the workspace's partition (org-scope skills in the org
//!   partition); a skill's `scope` is a tenancy namespace.
//! - Reads (find, list, get, SEP-2640 entries and reads, history, stats)
//!   see only the skill scopes Cedar lets the caller read; a skill outside
//!   them is "not found".
//! - Writes (put, propose, activate, rollback, undo, retire, optimize, …)
//!   need `write` on the skill's scope; outcomes and traces need `read`.
//! - Promotion needs `promote` on the target; into org it always needs a
//!   human's approval; `approve` needs the tenancy `approve` right and a
//!   human in the context.
//! - Every version, head change, outcome and log row records the context's
//!   actor chain; every op is appended to the tenancy audit chain, and the
//!   data writes themselves go through the session's audit sink.
//! - Host-filesystem and network inputs (`path`, `url`) and store
//!   configuration are refused: those are store-admin operations.

use serde_json::Map;

use super::*;
use crate::skill::{cs, uri_splits, SKILL, SKILL_META, SKILL_PROMO};

/// Ops that only read.
const READS: &[&str] = &[
    "get", "list", "history", "as_of", "find", "entry", "entries", "read", "dir", "stats", "traces", "evals", "scan", "proposals",
    "promotions", "keys", "verify", "package", "registry",
];
/// Feedback ops: they write, but only need `read` on the skill's scope.
const FEEDBACK: &[&str] = &["record_outcome", "use", "record_trace", "eval"];
/// Store administration, never as a principal.
const ADMIN: &[&str] = &["init", "config", "import", "export", "erase", "lifecycle", "keygen", "trust", "untrust", "learn_stage", "optimize_stage"];

fn take(a: &mut Map<String, JsonValue>, k: &str) -> Option<String> {
    match a.remove(k) {
        Some(JsonValue::String(s)) if !s.is_empty() => Some(s),
        _ => None,
    }
}

fn key_of(scope: &str) -> ScopeKey {
    ScopeKey::parse(scope, scope == "org")
}

impl<S> Db<S>
where
    S: for<'a> Storage<'a> + Clone + 'static,
{
    fn sk_part(&self, ss: &Session, scope: &str) -> Option<String> {
        if scope == "org" {
            Some(ss.org_part())
        } else {
            ss.part.clone()
        }
    }

    /// The view of the partition holding `scope`, its skill relations ready.
    fn sk_view(&self, ss: &Session, scope: &str, write: bool) -> Result<Db<S>> {
        let part = self.sk_part(ss, scope);
        let v = self.session_view(ss, part.as_deref(), None, write)?;
        if write && !v.sk_has(SKILL_META) {
            v.skill_call("init", json!({}))?;
        }
        Ok(v)
    }

    /// Skill scopes of the workspace (or the org) partition the session may
    /// perform `action` on.
    fn sk_visible(&self, ss: &Session, org: bool, action: &str) -> Result<Vec<String>> {
        let v = self.session_view(ss, if org { Some(ss.org_part()) } else { ss.part.clone() }.as_deref(), None, false)?;
        if !v.sk_has(SKILL) {
            return Ok(vec![]);
        }
        let mut scopes: Vec<String> = v
            .run_script(&format!("?[scope] := *{SKILL}{{scope}}"), Default::default(), ScriptMutability::Immutable)?
            .rows
            .iter()
            .filter_map(|r| cs(&r[0]))
            .filter(|s| (s == "org") == org)
            .collect();
        scopes.sort();
        scopes.dedup();
        let keys: Vec<ScopeKey> = scopes.iter().map(|s| key_of(s)).collect();
        let ok = if keys.is_empty() { vec![] } else { self.authz(ss)?.allowed(action, &keys)? };
        Ok(scopes.into_iter().zip(ok).filter(|(_, o)| *o).map(|(s, _)| s).collect())
    }

    /// Where a named or versioned skill lives among the readable scopes:
    /// `(scope, ver?)`. The narrowest readable scope holding the name wins.
    fn sk_locate(&self, ss: &Session, a: &Map<String, JsonValue>) -> Result<String> {
        let ws = self.sk_visible(ss, false, "read")?;
        let org = self.sk_visible(ss, true, "read")?;
        let readable = |s: &str| ws.iter().chain(org.iter()).any(|x| x == s);
        if let Some(ver) = a.get("ver").or_else(|| a.get("version")).and_then(|v| v.as_str()) {
            for o in [false, true] {
                let v = self.session_view(ss, if o { Some(ss.org_part()) } else { ss.part.clone() }.as_deref(), None, false)?;
                if !v.sk_has(SKILL) {
                    continue;
                }
                if let Some(row) = v.skill_ver(ver)? {
                    let s = row["scope"].as_str().unwrap_or_default().to_string();
                    if readable(&s) {
                        return Ok(s);
                    }
                }
            }
            bail!("no skill version '{ver}'");
        }
        let name = a.get("name").or_else(|| a.get("skill")).and_then(|v| v.as_str()).ok_or_else(|| miette::miette!("needs 'name' (or 'ver')"))?;
        if let Some(s) = a.get("scope").and_then(|v| v.as_str()) {
            if readable(s) {
                return Ok(s.to_string());
            }
            bail!("no skill '{name}' in scope '{s}'");
        }
        let mut cands: Vec<String> = vec![];
        for (o, scopes) in [(false, &ws), (true, &org)] {
            if scopes.is_empty() {
                continue;
            }
            let v = self.session_view(ss, if o { Some(ss.org_part()) } else { ss.part.clone() }.as_deref(), None, false)?;
            for r in v
                .run_script(
                    &format!("?[scope] := *{SKILL}{{scope, name: $n}}, is_in(scope, $sc)"),
                    crate::skill::params(vec![("n", DataValue::from(name)), ("sc", crate::skill::str_list(scopes))]),
                    ScriptMutability::Immutable,
                )?
                .rows
            {
                if let Some(s) = cs(&r[0]) {
                    cands.push(s);
                }
            }
        }
        cands.sort_by_key(|s| key_of(s).rank());
        cands.into_iter().next().ok_or_else(|| miette::miette!("no skill '{name}'"))
    }

    fn sk_actor(&self, ss: &Session) -> Result<JsonValue> {
        Ok(json!({
            "human": ss.human.as_ref().map(|h| h.id.clone()),
            "agent": ss.agent.as_ref().map(|h| h.id.clone()),
            "run": ss.ctx.run.clone(),
            "chain": serde_json::to_value(&ss.chain).into_diagnostic()?,
        }))
    }

    /// `skill {op, …}` as a principal.
    pub(crate) fn pm_skill(&self, ss: &Session, mut a: Map<String, JsonValue>) -> Result<JsonValue> {
        let sub = take(&mut a, "op").ok_or_else(|| miette::miette!("principal skill needs 'op' (a skill op)"))?;
        let sub = match sub.as_str() {
            "outcome" => "record_outcome".to_string(),
            "trace" => "record_trace".to_string(),
            _ => sub,
        };
        if ADMIN.contains(&sub.as_str()) {
            bail!("skill {sub} is a store administration op; it is not available to principals");
        }
        if a.contains_key("path") || a.contains_key("url") {
            bail!("skill {sub}: 'path' and 'url' read the server's files or network; pass the content ('package', 'skill_md') instead");
        }
        a.insert("actor".into(), self.sk_actor(ss)?);
        let out = match sub.as_str() {
            "find" | "entries" | "list" => self.pm_skill_search(ss, &sub, a)?,
            "keys" | "registry" => self.sk_view(ss, "ws", false)?.skill_call(&sub, JsonValue::Object(a))?,
            "scan" if a.contains_key("skill_md") => self.sk_view(ss, "ws", false)?.skill_call("scan", JsonValue::Object(a))?,
            "proposals" | "promotions" => {
                let ws = self.sk_visible(ss, false, "read")?;
                let v = self.session_view(ss, ss.part.as_deref(), None, false)?;
                if !v.sk_has(SKILL) {
                    json!([])
                } else {
                    let all = v.skill_call(&sub, json!({"scopes": if ws.is_empty() { vec!["-".to_string()] } else { ws.clone() }}))?;
                    JsonValue::Array(all.as_array().cloned().unwrap_or_default().into_iter().filter(|r| r["scope"].as_str().is_some_and(|s| ws.iter().any(|x| x == s))).collect())
                }
            }
            "entry" | "read" | "dir" if a.contains_key("uri") => {
                let uri = a.get("uri").and_then(|u| u.as_str()).unwrap_or_default().to_string();
                let ws = self.sk_visible(ss, false, "read")?;
                let org = self.sk_visible(ss, true, "read")?;
                let scope = uri_splits(&uri)
                    .into_iter()
                    .map(|(s, _, _)| s)
                    .find(|s| ws.contains(s) || org.contains(s))
                    .ok_or_else(|| miette::miette!("Resource not found: {uri}"))?;
                self.sk_view(ss, &scope, false)?.skill_call(&sub, JsonValue::Object(a))?
            }
            "promote" => return self.pm_skill_promote(ss, a),
            "approve" | "reject" => return self.pm_skill_approve(ss, &sub, a),
            _ => {
                // the target scope: a new skill's (put, propose, install,
                // record_trace, learn) is named or the narrowest sensible one;
                // anything else is where the skill already is
                let creates = matches!(sub.as_str(), "put" | "propose" | "install" | "record_trace" | "learn");
                let scope = if creates && !a.contains_key("ver") {
                    let want = a.get("scope").and_then(|s| s.as_str()).map(String::from);
                    self.scope_of(ss, want.as_deref(), false)?.ns()
                } else {
                    self.sk_locate(ss, &a)?
                };
                let read_only = READS.contains(&sub.as_str());
                let action = if read_only || FEEDBACK.contains(&sub.as_str()) { "read" } else { "write" };
                if !self.authz(ss)?.allows(action, &key_of(&scope))? {
                    self.audit_op(ss, &format!("skill.{sub}"), json!({"scope": scope}), None, json!({"effect": "deny"}))?;
                    if action == "read" {
                        bail!("no skill in scope '{scope}'");
                    }
                    bail!("not allowed to change skills in scope '{scope}'");
                }
                a.insert("scope".into(), json!(scope));
                if sub == "learn" {
                    a.remove("scope");
                    a.insert("scopes".into(), json!([scope]));
                }
                let v = self.sk_view(ss, &scope, !read_only)?;
                let out = v.skill_call(&sub, JsonValue::Object(a))?;
                self.audit_op(ss, &format!("skill.{sub}"), json!({"scope": scope, "ver": out.get("ver"), "name": out.get("name")}), None, json!({"effect": "allow"}))?;
                return Ok(out);
            }
        };
        self.audit_op(ss, &format!("skill.{sub}"), json!({}), None, json!({"effect": "allow"}))?;
        Ok(out)
    }

    /// `find`, `entries`, `list` over the readable scopes of the workspace
    /// partition and, when readable, the org partition.
    fn pm_skill_search(&self, ss: &Session, sub: &str, mut a: Map<String, JsonValue>) -> Result<JsonValue> {
        let mut want: Option<Vec<String>> = match a.remove("scopes") {
            Some(JsonValue::Array(v)) => Some(v.into_iter().filter_map(|x| x.as_str().map(String::from)).collect()),
            _ => None,
        };
        if let Some(s) = take(&mut a, "scope") {
            want = Some(vec![s]);
        }
        let keep = |v: Vec<String>| -> Vec<String> { v.into_iter().filter(|s| want.as_ref().map_or(true, |w| w.contains(s))).collect() };
        let ws = keep(self.sk_visible(ss, false, "read")?);
        let org = keep(self.sk_visible(ss, true, "read")?);
        let mut parts: Vec<JsonValue> = vec![];
        for (scopes, is_org) in [(ws, false), (org, true)] {
            if scopes.is_empty() {
                continue;
            }
            let mut o = a.clone();
            o.insert("scopes".into(), json!(scopes));
            if is_org && sub == "find" {
                o.insert("log".into(), json!(false));
            }
            let v = self.session_view(ss, if is_org { Some(ss.org_part()) } else { ss.part.clone() }.as_deref(), None, sub == "find" && !is_org)?;
            parts.push(v.skill_call(sub, JsonValue::Object(o))?);
        }
        Ok(match sub {
            "list" => JsonValue::Array(parts.into_iter().flat_map(|p| p.as_array().cloned().unwrap_or_default()).collect()),
            "find" => {
                let k = a.get("k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
                let route = parts.first().and_then(|p| p.get("route").cloned()).unwrap_or(JsonValue::Null);
                let tokens: u64 = parts.iter().filter_map(|p| p["tokens"].as_u64()).sum();
                let mut skills: Vec<JsonValue> = parts.into_iter().flat_map(|p| p["skills"].as_array().cloned().unwrap_or_default()).collect();
                skills.sort_by(|x, y| y["score"].as_f64().partial_cmp(&x["score"].as_f64()).unwrap_or(std::cmp::Ordering::Equal));
                skills.truncate(k.max(1));
                for (i, s) in skills.iter_mut().enumerate() {
                    s["rank"] = json!(i + 1);
                }
                json!({"route": route, "task": a.get("task"), "skills": skills, "tokens": tokens, "budget": a.get("budget")})
            }
            _ => {
                let next = parts.first().and_then(|p| p.get("nextCursor").cloned());
                let skills: Vec<JsonValue> = parts.into_iter().flat_map(|p| p["skills"].as_array().cloned().unwrap_or_default()).collect();
                let mut r = json!({"skills": skills, "partial": true});
                if let Some(n) = next {
                    r["nextCursor"] = n;
                }
                r
            }
        })
    }

    /// `promote {name | ver, to, approve?}`: `promote` on the target scope;
    /// org needs a human's approval (here with `approve: true`, or later).
    fn pm_skill_promote(&self, ss: &Session, mut a: Map<String, JsonValue>) -> Result<JsonValue> {
        let from = self.sk_locate(ss, &a)?;
        let to = take(&mut a, "to").ok_or_else(|| miette::miette!("skill promote needs 'to'"))?;
        let target = self.scope_of(ss, Some(&to), true)?;
        let tns = target.ns();
        let approve = a.remove("approve").and_then(|x| x.as_bool()).unwrap_or(false);
        let may = self.authz(ss)?.allows("promote", &target)?;
        let human = ss.human.as_ref().map(|h| h.id.clone());
        let direct = may && (target != ScopeKey::Org || (approve && human.is_some()));
        let src = self.sk_view(ss, &from, true)?;
        a.insert("scope".into(), json!(from));
        a.insert("to".into(), json!(tns));
        a.insert("auto".into(), json!(false));
        let req = src.skill_call("promote", JsonValue::Object(a))?;
        let pid = req["promotion"].as_str().unwrap_or_default().to_string();
        if !direct {
            self.audit_op(ss, "skill.promote", json!({"from": from, "to": tns, "promotion": pid}), None, json!({"effect": "pending"}))?;
            return Ok(req);
        }
        let approver = self.sk_actor(ss)?;
        let tv = self.sk_view(ss, &tns, true)?;
        let out = src.skill_apply_promotion_into(&pid, &approver, &tv)?;
        self.audit_op(ss, "skill.promote", json!({"from": from, "to": tns, "promotion": pid, "new_ver": out.get("new_ver")}), None, json!({"effect": "allow"}))?;
        Ok(out)
    }

    /// `approve | reject {promotion}`: a human with the tenancy `approve`
    /// right on the target scope.
    fn pm_skill_approve(&self, ss: &Session, op: &str, mut a: Map<String, JsonValue>) -> Result<JsonValue> {
        let pid = take(&mut a, "promotion").or_else(|| take(&mut a, "id")).ok_or_else(|| miette::miette!("skill {op} needs 'promotion'"))?;
        let src = self.session_view(ss, ss.part.as_deref(), None, true)?;
        let rows = if src.sk_has(SKILL_PROMO) {
            src.run_script(
                &format!("?[scope, to, status] := *{SKILL_PROMO}{{id: $id, scope, to, status}}"),
                crate::skill::params(vec![("id", DataValue::from(pid.as_str()))]),
                ScriptMutability::Immutable,
            )?
            .rows
        } else {
            vec![]
        };
        let Some(r) = rows.first() else { bail!("promotion '{pid}' not found") };
        // the promotion is in this workspace's partition; approving is a right
        // on the target (an org admin approves without reading the workspace)
        let (from, to) = (cs(&r[0]).unwrap_or_default(), cs(&r[1]).unwrap_or_default());
        let target = key_of(&to);
        if ss.human.is_none() {
            bail!("only a human approves a skill promotion");
        }
        if !self.authz(ss)?.allows("approve", &target)? {
            self.audit_op(ss, &format!("skill.{op}"), json!({"promotion": pid}), None, json!({"effect": "deny"}))?;
            bail!("not allowed to approve promotions into '{to}'");
        }
        let actor = self.sk_actor(ss)?;
        let out = if op == "approve" {
            let tv = self.sk_view(ss, &to, true)?;
            src.skill_apply_promotion_into(&pid, &actor, &tv)?
        } else {
            src.skill_call("reject", json!({"promotion": pid, "actor": actor, "reason": a.get("reason")}))?
        };
        self.audit_op(ss, &format!("skill.{op}"), json!({"promotion": pid, "from": from, "to": to}), None, json!({"effect": "allow"}))?;
        Ok(out)
    }
}
