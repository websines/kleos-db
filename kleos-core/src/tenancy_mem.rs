/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Memory ops as a principal (docs/specs/tenancy.md, "As a principal"):
//! writes to a checked target scope with the actor chain stamped on every
//! memory, reads across the context's ScopeSet (one pre-filtered search of
//! the workspace partition, plus the org partition when visible), promotion
//! as a copy with provenance, and approvals. Every op appends audit events.

use std::collections::{BTreeMap, BTreeSet};

use crossbeam::channel::{unbounded, Receiver};
use miette::{IntoDiagnostic, Result};
use serde_json::{json, Map, Value as JsonValue};

use super::*;
use crate::memory::{RecallHit, RecallOpts, RememberOpts, MEM, MEM_NS, MEM_PROMOTED};
use crate::runtime::db::ViewCtx;
use crate::CallbackOp;

fn obj(op: &str, args: JsonValue) -> Result<Map<String, JsonValue>> {
    match args {
        JsonValue::Null => Ok(Map::new()),
        JsonValue::Object(m) => Ok(m),
        other => bail!("principal {op}: options must be an object, got {other}"),
    }
}

fn take_str(a: &mut Map<String, JsonValue>, k: &str) -> Option<String> {
    match a.remove(k) {
        Some(JsonValue::String(s)) => Some(s),
        _ => None,
    }
}

fn take_ids(a: &mut Map<String, JsonValue>) -> Vec<String> {
    let mut ids: Vec<String> = take_str(a, "id").into_iter().collect();
    if let Some(JsonValue::Array(v)) = a.remove("ids") {
        ids.extend(v.into_iter().filter_map(|x| x.as_str().map(|y| y.to_string())));
    }
    ids
}

/// The ids a `remember` result touched.
fn remember_ids(out: &JsonValue) -> Vec<String> {
    let mut ids = BTreeSet::new();
    for k in ["id", "duplicate_of"] {
        if let Some(x) = out.get(k).and_then(|v| v.as_str()) {
            ids.insert(x.to_string());
        }
    }
    for k in ["superseded"] {
        for x in out.get(k).and_then(|v| v.as_array()).into_iter().flatten() {
            if let Some(x) = x.as_str() {
                ids.insert(x.to_string());
            }
        }
    }
    for c in out.get("claims").and_then(|v| v.as_array()).into_iter().flatten() {
        for k in ["id", "duplicate_of"] {
            if let Some(x) = c.get(k).and_then(|v| v.as_str()) {
                ids.insert(x.to_string());
            }
        }
    }
    ids.into_iter().collect()
}

impl<S> Db<S>
where
    S: for<'a> Storage<'a> + Clone + 'static,
{
    /// The data view of a session: its partition, with the write sink, the
    /// actor stamp (writes) or a read union (reads).
    fn session_view(&self, ss: &Session, part: Option<&str>, union: Option<Vec<String>>, write: bool) -> Result<Db<S>> {
        let pv = self.part_view(part)?;
        let actor = if write {
            let raw = serde_json::to_value(&ss.chain).into_diagnostic()?;
            Some([
                ss.human.as_ref().map(|h| DataValue::from(h.id.as_str())).unwrap_or(DataValue::Null),
                ss.agent.as_ref().map(|h| DataValue::from(h.id.as_str())).unwrap_or(DataValue::Null),
                ss.ctx.run.as_deref().map(DataValue::from).unwrap_or(DataValue::Null),
                json_dv(raw),
            ])
        } else {
            None
        };
        Ok(pv.with_view(ViewCtx { union, actor, sink: Some(self.write_sink(ss)?) }))
    }

    /// The scope a write option names; default the narrowest sensible one
    /// (agent + human → agent_user; human → user; autonomous agent → its
    /// run). `agent` and `org` are promotion targets only, unless
    /// `promotion` is set.
    fn scope_of(&self, ss: &Session, want: Option<&str>, promotion: bool) -> Result<ScopeKey> {
        let human = ss.human.as_ref().map(|h| h.id.clone());
        let agent = ss.mem_agent().map(|x| x.to_string());
        let run = ss.ctx.run.clone();
        let need = |x: &Option<String>, what: &str| -> Result<String> {
            x.clone().ok_or_else(|| miette::miette!("scope '{}' needs {what} in the principal context", want.unwrap_or("")))
        };
        let k = match want {
            None => match (&agent, &human) {
                (Some(a), Some(h)) => ScopeKey::AgentUser { agent: a.clone(), user: h.clone() },
                (None, Some(h)) => ScopeKey::User { user: h.clone() },
                (Some(a), None) => ScopeKey::Run { agent: a.clone(), user: None, run: run.unwrap_or_else(|| "default".to_string()) },
                (None, None) => bail!("a service must name the scope it writes to"),
            },
            Some("run") => ScopeKey::Run { agent: need(&agent, "an agent")?, user: human, run: need(&run, "a run")? },
            Some("au") | Some("agent_user") => ScopeKey::AgentUser { agent: need(&agent, "an agent")?, user: need(&human, "a human")? },
            Some("user") => ScopeKey::User { user: need(&human, "a human")? },
            Some("ws") | Some("workspace") => ScopeKey::Workspace,
            Some("agent") if promotion => ScopeKey::Agent { agent: need(&agent, "an agent")? },
            Some("org") if promotion => ScopeKey::Org,
            Some("agent") | Some("org") => bail!("scope '{}' is written only by promotion", want.unwrap_or("")),
            Some(x) => match x.split_once(':') {
                Some(("team", t)) => {
                    check_id("team", t)?;
                    ScopeKey::Team { team: t.to_string() }
                }
                Some(("agent", a)) if promotion => {
                    check_id("agent", a)?;
                    ScopeKey::Agent { agent: a.to_string() }
                }
                Some(("ns", n)) if ss.part.is_none() => ScopeKey::Legacy { ns: n.to_string() },
                _ => bail!("unknown scope '{x}': use run, agent_user, user, team:<id>, workspace{}", if promotion { ", agent or org" } else { "" }),
            },
        };
        Ok(k)
    }

    /// Where each id lives among `nss` of a partition (ids elsewhere are
    /// left out: "not found").
    fn locate(&self, part: Option<&str>, nss: &[String], ids: &[String]) -> Result<BTreeMap<String, String>> {
        if nss.is_empty() || ids.is_empty() {
            return Ok(BTreeMap::new());
        }
        let pv = self.part_view(part)?;
        Ok(pv
            .run_script(
                &format!("?[id, ns] := ns in $nss, id in $ids, *{MEM}{{ns, id}}"),
                params(vec![("nss", str_list(nss)), ("ids", str_list(ids))]),
                ScriptMutability::Immutable,
            )?
            .rows
            .into_iter()
            .filter_map(|r| Some((cell_str(&r[0])?, cell_str(&r[1])?)))
            .collect())
    }

    fn ns_exists(&self, part: Option<&str>, ns: &str) -> Result<bool> {
        let pv = self.part_view(part)?;
        Ok(pv
            .run_script(&format!("?[seq] := *{MEM_NS}{{ns: $ns, seq}}"), params(vec![("ns", s(ns))]), ScriptMutability::Immutable)
            .map(|r| !r.rows.is_empty())
            .unwrap_or(false))
    }

    /// Memory ops as a principal (`::principal <op> {as: ctx, …}`). Options
    /// are the memory API's, with `scope` instead of `ns`.
    pub fn memory_call_as(&self, ctx: &PrincipalContext, op: &str, args: JsonValue) -> Result<JsonValue> {
        let ss = self.tenant_session(ctx)?;
        let mut a = obj(op, args)?;
        if a.contains_key("ns") {
            bail!("principal {op}: name a 'scope', not an 'ns'");
        }
        let out = match op {
            "remember" => self.pm_remember(&ss, a)?,
            "recall" => self.pm_recall(&ss, a)?,
            "context" => {
                a.insert("format".into(), json!("context"));
                self.pm_recall(&ss, a)?
            }
            "forget" | "set" | "link" => self.pm_by_ids(&ss, op, a)?,
            "reflect" | "consolidate" => {
                let want = take_str(&mut a, "scope");
                let k = self.scope_of(&ss, want.as_deref(), false).or_else(|_| match want.as_deref() {
                    Some(w) => self.scope_of(&ss, Some(w), true),
                    None => bail!("principal {op} needs a scope"),
                })?;
                let action = if op == "reflect" { "read" } else { "write" };
                if !self.authz(&ss)?.allows(action, &k)? {
                    bail!("scope '{}' not found", k.ns());
                }
                let part = if k == ScopeKey::Org { Some(ss.org_part()) } else { ss.part.clone() };
                self.ensure_part_store(part.as_deref())?;
                a.insert("ns".into(), json!(k.ns()));
                let v = self.session_view(&ss, part.as_deref(), None, op != "reflect")?;
                let out = v.memory_call(op, JsonValue::Object(a))?;
                self.audit_op(&ss, op, json!({"scope": k.ns()}), None, json!({"effect": "allow"}))?;
                out
            }
            "promote" => self.pm_promote(&ss, a)?,
            "flow" | "doc" | "ontology" => self.pm_area(&ss, op, a)?,
            "approve" | "reject" => self.pm_approve(&ss, op, a)?,
            "fork" | "merge" | "discard" | "diff" => self.pm_branch(&ss, op, a)?,
            "scopes" => {
                let action = take_str(&mut a, "action").unwrap_or_else(|| "read".to_string());
                let (scopes, org) = self.scope_set(&ss, &action)?;
                json!({"workspace": ss.ws, "scopes": scopes.iter().map(|k| k.ns()).collect::<Vec<_>>(), "org_scope": org})
            }
            other => bail!(
                "unknown principal op '{other}': expected remember, recall, context, forget, set, link, reflect, consolidate, \
                 promote, approve, reject, fork, merge, discard, diff, flow, doc, ontology or scopes"
            ),
        };
        Ok(out)
    }

    fn pm_remember(&self, ss: &Session, mut a: Map<String, JsonValue>) -> Result<JsonValue> {
        let want = take_str(&mut a, "scope");
        let k = self.scope_of(ss, want.as_deref(), false)?;
        if !self.authz(ss)?.allows("write", &k)? {
            self.audit_op(ss, "remember", json!({"scope": k.ns()}), None, json!({"effect": "deny"}))?;
            bail!("not allowed to write to scope '{}'", k.ns());
        }
        let part = ss.part.clone();
        self.ensure_part_store(part.as_deref())?;
        let ns = k.ns();
        let fresh = !self.ns_exists(part.as_deref(), &ns)?;
        a.insert("ns".into(), json!(ns));
        let v = self.session_view(ss, part.as_deref(), None, true)?;
        let mut out = v.memory_call("remember", JsonValue::Object(a))?;
        if fresh {
            self.bump_scope_epoch(&ss.ws)?;
        }
        // zero-maintenance ontology: what a scope learns about the schema is
        // promoted to the workspace's learned layer automatically
        if out.get("ontology_learned").is_some_and(|x| x.get("error").is_none() && !x.is_null()) {
            let _ = self.promote_schema(ss, &ns);
        }
        let ids = remember_ids(&out);
        self.audit_op(ss, "remember", json!({"scope": ns, "ids": ids}), None, json!({"effect": "allow"}))?;
        if let Some(o) = out.as_object_mut() {
            o.insert("scope".into(), json!(ns));
        }
        Ok(out)
    }

    fn pm_recall(&self, ss: &Session, mut a: Map<String, JsonValue>) -> Result<JsonValue> {
        let query = take_str(&mut a, "query").ok_or_else(|| miette::miette!("principal recall needs 'query'"))?;
        let kinds: Option<BTreeSet<&'static str>> = match a.remove("scopes") {
            None | Some(JsonValue::Null) => None,
            Some(JsonValue::Array(v)) => {
                let mut ks = BTreeSet::new();
                for x in v {
                    let x = x.as_str().unwrap_or_default().to_string();
                    ks.insert(kind_alias(&x).ok_or_else(|| miette::miette!("unknown scope kind '{x}'"))?);
                }
                Some(ks)
            }
            Some(other) => bail!("principal recall: 'scopes' must be a list of scope kinds, got {other}"),
        };
        let context = a.get("format").and_then(|x| x.as_str()) == Some("context");
        let (scopes, org) = self.scope_set(ss, "read")?;
        let nss: Vec<String> = scopes.iter().filter(|k| kinds.as_ref().map_or(true, |ks| ks.contains(k.kind()))).map(|k| k.ns()).collect();
        let org = org && kinds.as_ref().map_or(true, |ks| ks.contains("org"));
        let set_hash = scope_set_hash(&nss, org);
        let mut opts: RecallOpts = serde_json::from_value(JsonValue::Object(a.clone())).map_err(|e| miette::miette!("principal recall: bad options: {e}"))?;
        // the org partition first: its hits join the workspace's before the
        // precedence rule and the diverse selection
        let mut extra: Vec<RecallHit> = vec![];
        if org {
            let ov = self.session_view(ss, Some(&ss.org_part()), None, false)?;
            let mut oo = opts.clone();
            oo.ns = Some("org".to_string());
            oo.format = None;
            extra = ov.recall(&query, oo)?;
            for h in extra.iter_mut() {
                h.scope = Some("org".to_string());
            }
        }
        let out = if nss.is_empty() {
            if context {
                let ov = self.session_view(ss, Some(&ss.org_part()), None, false)?;
                if org {
                    opts.ns = Some("org".to_string());
                    serde_json::to_value(ov.recall_context(&query, opts)?).into_diagnostic()?
                } else {
                    serde_json::to_value(crate::memory_recall::render_context(&[], opts.budget.unwrap_or(2000), false)).into_diagnostic()?
                }
            } else {
                let k = opts.k.unwrap_or(10).max(1);
                extra.truncate(k);
                serde_json::to_value(&extra).into_diagnostic()?
            }
        } else {
            opts.ns = Some(nss[0].clone());
            opts.extra_hits = extra;
            let v = self.session_view(ss, ss.part.as_deref(), Some(nss.clone()), false)?;
            if context {
                serde_json::to_value(v.recall_context(&query, opts)?).into_diagnostic()?
            } else {
                let mut hits = v.recall(&query, opts)?;
                for h in hits.iter_mut() {
                    if h.scope.is_none() {
                        h.scope = Some(nss[0].clone());
                    }
                }
                serde_json::to_value(hits).into_diagnostic()?
            }
        };
        let ids: Vec<JsonValue> = if context {
            out.get("ids").cloned().and_then(|x| x.as_array().cloned()).unwrap_or_default()
        } else {
            out.as_array().map(|v| v.iter().filter_map(|h| h.get("id").cloned()).collect()).unwrap_or_default()
        };
        self.audit_op(ss, if context { "context" } else { "recall" }, json!({"ids": ids}), Some(&set_hash), json!({"effect": "allow", "scopes": nss.len() + org as usize}))?;
        Ok(out)
    }

    /// `forget`, `set` and `link` on memories of writable scopes.
    fn pm_by_ids(&self, ss: &Session, op: &str, mut a: Map<String, JsonValue>) -> Result<JsonValue> {
        let want = take_str(&mut a, "scope");
        let part = ss.part.clone();
        if op == "link" {
            let from = take_str(&mut a, "from").ok_or_else(|| miette::miette!("principal link needs 'from'"))?;
            let to = take_str(&mut a, "to").ok_or_else(|| miette::miette!("principal link needs 'to'"))?;
            let (scopes, _) = self.scope_set(ss, "write")?;
            let nss: Vec<String> = scopes.iter().map(|k| k.ns()).collect();
            let at = self.locate(part.as_deref(), &nss, &[from.clone(), to.clone()])?;
            let ns = match (at.get(&from), at.get(&to)) {
                (Some(x), Some(y)) if x == y => x.clone(),
                (Some(_), Some(_)) => bail!("principal link: both ends must be in one scope"),
                _ => bail!("memory not found"),
            };
            a.insert("ns".into(), json!(ns));
            a.insert("from".into(), json!(from));
            a.insert("to".into(), json!(to));
            let v = self.session_view(ss, part.as_deref(), None, true)?;
            let out = v.memory_call("link", JsonValue::Object(a))?;
            self.audit_op(ss, "link", json!({"scope": ns, "ids": [from, to]}), None, json!({"effect": "allow"}))?;
            return Ok(out);
        }
        let ids = take_ids(&mut a);
        let v = self.session_view(ss, part.as_deref(), None, true)?;
        if ids.is_empty() {
            // a filter (kind, entity, expired, all) acts on one named scope
            let k = self.scope_of(ss, Some(want.as_deref().ok_or_else(|| miette::miette!("principal {op} needs 'ids' or a 'scope'"))?), false)?;
            if !self.authz(ss)?.allows("write", &k)? {
                bail!("scope '{}' not found", k.ns());
            }
            a.insert("ns".into(), json!(k.ns()));
            let out = v.memory_call(op, JsonValue::Object(a))?;
            self.audit_op(ss, op, json!({"scope": k.ns(), "filter": true, "result": out.get("ids")}), None, json!({"effect": "allow"}))?;
            return Ok(out);
        }
        let (scopes, _) = self.scope_set(ss, "write")?;
        let nss: Vec<String> = scopes.iter().map(|k| k.ns()).collect();
        let mut by_ns: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (id, ns) in self.locate(part.as_deref(), &nss, &ids)? {
            by_ns.entry(ns).or_default().push(id);
        }
        let mut done: Vec<String> = vec![];
        let mut mode = JsonValue::Null;
        for (ns, group) in &by_ns {
            let mut o = a.clone();
            o.insert("ns".into(), json!(ns));
            o.insert("ids".into(), json!(group));
            let r = v.memory_call(op, JsonValue::Object(o))?;
            mode = r.get("mode").cloned().unwrap_or(JsonValue::Null);
            done.extend(r.get("ids").and_then(|x| x.as_array()).into_iter().flatten().filter_map(|x| x.as_str().map(|y| y.to_string())));
        }
        self.audit_op(ss, op, json!({"ids": done, "scopes": by_ns.keys().collect::<Vec<_>>()}), None, json!({"effect": "allow"}))?;
        Ok(if op == "forget" { json!({"mode": mode, "ids": done}) } else { json!({"ids": done}) })
    }

    /// Workflows, documents and ontologies of the session's partition
    /// (`{op: <sub-op>, scope?, …}`): the addressed namespace is a scope
    /// (default the workspace scope), read ops need read on it, the others
    /// write.
    fn pm_area(&self, ss: &Session, op: &str, mut a: Map<String, JsonValue>) -> Result<JsonValue> {
        let sub = take_str(&mut a, "op").ok_or_else(|| miette::miette!("principal {op} needs 'op' (the {op} operation)"))?;
        let reads: &[&str] = match op {
            "flow" => &["status", "list", "history", "stats", "schedules"],
            "doc" => &["get", "list"],
            _ => &["get", "list", "export", "facts", "infer", "validate", "type", "gate", "proposals", "history"],
        };
        let action = if reads.contains(&sub.as_str()) { "read" } else { "write" };
        let want = take_str(&mut a, "scope");
        let k = match want.as_deref() {
            None => ScopeKey::Workspace,
            Some(w) => self.scope_of(ss, Some(w), false)?,
        };
        if !self.authz(ss)?.allows(action, &k)? {
            bail!("scope '{}' not found", k.ns());
        }
        if op != "flow" {
            a.insert("ns".into(), json!(k.ns()));
        }
        self.ensure_part_store(ss.part.as_deref())?;
        let v = self.session_view(ss, ss.part.as_deref(), None, action == "write")?;
        let args = JsonValue::Object(a);
        let out = match op {
            "flow" => v.flow_call(&sub, args)?,
            #[cfg(feature = "docs")]
            "doc" => v.doc_call(&sub, args)?,
            #[cfg(not(feature = "docs"))]
            "doc" => {
                let _ = args;
                bail!("documents need kleos built with the `docs` feature")
            }
            _ => v.ontology_call(&sub, args)?,
        };
        self.audit_op(ss, &format!("{op}.{sub}"), json!({"scope": k.ns()}), None, json!({"effect": "allow"}))?;
        Ok(out)
    }

    /// Promotion: a copy with provenance into a broader scope.
    fn pm_promote(&self, ss: &Session, mut a: Map<String, JsonValue>) -> Result<JsonValue> {
        let ids = take_ids(&mut a);
        let to = take_str(&mut a, "to").ok_or_else(|| miette::miette!("principal promote needs 'to'"))?;
        let approve = a.remove("approve").and_then(|x| x.as_bool()).unwrap_or(false);
        if ids.is_empty() {
            bail!("principal promote needs 'ids'");
        }
        let target = self.scope_of(ss, Some(&to), true)?;
        let (scopes, _) = self.scope_set(ss, "read")?;
        let nss: Vec<String> = scopes.iter().map(|k| k.ns()).collect();
        let at = self.locate(ss.part.as_deref(), &nss, &ids)?;
        if let Some(missing) = ids.iter().find(|i| !at.contains_key(*i)) {
            bail!("memory '{missing}' not found");
        }
        let az = self.authz(ss)?;
        let may = az.allows("promote", &target)?;
        let direct = may && (target != ScopeKey::Org || approve);
        if direct {
            let approver = if target == ScopeKey::Org { ss.human.as_ref().map(|h| h.id.clone()) } else { None };
            let chain = serde_json::to_value(&ss.chain).into_diagnostic()?;
            return self.do_promote(ss, &at, &target, chain, approver);
        }
        // a pending request, for a human who can approve it
        let pid = format!("pr_{}", hex(&rand::random::<[u8; 8]>()));
        let mut by_ns: BTreeMap<&String, Vec<&String>> = BTreeMap::new();
        for (id, ns) in &at {
            by_ns.entry(ns).or_default().push(id);
        }
        if by_ns.len() > 1 {
            bail!("principal promote: one request promotes memories of one scope");
        }
        let (from_ns, group) = by_ns.into_iter().next().expect("non-empty");
        let group: Vec<String> = group.into_iter().cloned().collect();
        self.tn_q(
            &format!(
                "?[id, ws, from_ns, ids, target, chain, requested_by, status, approved_by, result, at] <- \
                   [[$id, $ws, $from, $ids, $to, $chain, $by, 'pending', null, null, $now]] \
                 :put {TN_PROMOTION} {{id => ws, from_ns, ids, target, chain, requested_by, status, approved_by, result, at}}"
            ),
            vec![
                ("id", s(&pid)),
                ("ws", s(&ss.ws)),
                ("from", s(from_ns)),
                ("ids", str_list(&group)),
                ("to", s(&target.ns())),
                ("chain", json_dv(serde_json::to_value(&ss.chain).into_diagnostic()?)),
                ("by", s(ss.accountable.as_deref().unwrap_or_default())),
                ("now", DataValue::from(now_micros())),
            ],
        )?;
        self.audit_op(ss, "promote", json!({"ids": group, "to": target.ns(), "promotion": pid}), None, json!({"effect": "pending"}))?;
        Ok(json!({"status": "pending", "promotion": pid, "to": target.ns()}))
    }

    /// Copy `at` (id → source namespace in the session's partition) into
    /// `target`, with provenance rows.
    fn do_promote(&self, ss: &Session, at: &BTreeMap<String, String>, target: &ScopeKey, chain: JsonValue, approver: Option<String>) -> Result<JsonValue> {
        let src = self.part_view(ss.part.as_deref())?;
        let nss: Vec<String> = at.values().cloned().collect::<BTreeSet<_>>().into_iter().collect();
        let ids: Vec<String> = at.keys().cloned().collect();
        let rows = src
            .run_script(
                &format!("?[id, text, kind, source, importance] := ns in $nss, id in $ids, *{MEM}{{ns, id, text, kind, source, importance @ 'NOW'}}"),
                params(vec![("nss", str_list(&nss)), ("ids", str_list(&ids))]),
                ScriptMutability::Immutable,
            )?
            .rows;
        let tpart = if *target == ScopeKey::Org { Some(ss.org_part()) } else { ss.part.clone() };
        self.ensure_part_store(tpart.as_deref())?;
        let tns = target.ns();
        let fresh = !self.ns_exists(tpart.as_deref(), &tns)?;
        let tv = self.session_view(ss, tpart.as_deref(), None, true)?;
        let mut copies = vec![];
        let mut prov = vec![];
        let now = now_micros();
        for r in rows {
            let (Some(id), Some(text)) = (cell_str(&r[0]), cell_str(&r[1])) else { continue };
            let res = tv.remember(
                &text,
                RememberOpts {
                    ns: Some(tns.clone()),
                    kind: cell_str(&r[2]),
                    source: cell_str(&r[3]),
                    importance: r[4].get_float(),
                    split: Some("none".to_string()),
                    extract: Some(false),
                    ..Default::default()
                },
            )?;
            let new = res.id.clone().or(res.duplicate_of.clone());
            if let Some(n) = &new {
                prov.push(DataValue::List(vec![
                    s(&tns),
                    s(n),
                    s(&ss.ws),
                    s(&at[&id]),
                    s(&id),
                    json_dv(chain.clone()),
                    approver.as_deref().map(DataValue::from).unwrap_or(DataValue::Null),
                    DataValue::from(now),
                ]));
            }
            copies.push(json!({"from": id, "id": new, "action": if res.id.is_some() { "added" } else { "duplicate" }}));
        }
        if !prov.is_empty() {
            tv.run_script(
                &format!(
                    "?[ns, id, from_ws, from_ns, from_id, chain, approved_by, at] <- $rows \
                     :put {MEM_PROMOTED} {{ns, id => from_ws, from_ns, from_id, chain, approved_by, at}}"
                ),
                params(vec![("rows", DataValue::List(prov))]),
                ScriptMutability::Mutable,
            )?;
        }
        if fresh {
            if *target == ScopeKey::Org {
                // every workspace of the org can now see an org scope
                self.bump_epoch()?;
            } else {
                self.bump_scope_epoch(&ss.ws)?;
            }
        }
        self.audit_op(ss, "promote", json!({"to": tns, "copies": copies}), None, json!({"effect": "allow", "approved_by": approver.is_some()}))?;
        Ok(json!({"status": "done", "to": tns, "copies": copies}))
    }

    fn pm_approve(&self, ss: &Session, op: &str, mut a: Map<String, JsonValue>) -> Result<JsonValue> {
        let pid = take_str(&mut a, "promotion").ok_or_else(|| miette::miette!("principal {op} needs 'promotion'"))?;
        let rows = self.tn_rows(
            &format!("?[ws, from_ns, ids, target, chain, status] := *{TN_PROMOTION}{{id: $id, ws, from_ns, ids, target, chain, status}}"),
            vec![("id", s(&pid))],
        )?;
        let Some(r) = rows.first() else { bail!("promotion '{pid}' not found") };
        if cell_str(&r[0]).as_deref() != Some(ss.ws.as_str()) {
            bail!("promotion '{pid}' not found");
        }
        if r[5].get_str() != Some("pending") {
            bail!("promotion '{pid}' is not pending");
        }
        let from_ns = cell_str(&r[1]).unwrap_or_default();
        let ids: Vec<String> = match &r[2] {
            DataValue::List(l) => l.iter().filter_map(|x| x.get_str().map(|y| y.to_string())).collect(),
            _ => vec![],
        };
        let tns = cell_str(&r[3]).unwrap_or_default();
        let target = ScopeKey::parse(&tns, tns == "org");
        let approver = ss.human.as_ref().map(|h| h.id.clone()).ok_or_else(|| miette::miette!("only a human approves a promotion"))?;
        if !self.authz(ss)?.allows("approve", &target)? {
            bail!("not allowed to approve promotions into '{tns}'");
        }
        let status = if op == "approve" { "approved" } else { "rejected" };
        let result = if op == "approve" {
            let at: BTreeMap<String, String> = self
                .locate(ss.part.as_deref(), std::slice::from_ref(&from_ns), &ids)?
                .into_iter()
                .collect();
            self.do_promote(ss, &at, &target, cell_json(&r[4]), Some(approver.clone()))?
        } else {
            self.audit_op(ss, "reject", json!({"promotion": pid}), None, json!({"effect": "rejected"}))?;
            json!({"status": "rejected"})
        };
        self.tn_q(
            &format!(
                "?[id, status, approved_by, result] <- [[$id, $st, $by, $res]] :update {TN_PROMOTION} {{id => status, approved_by, result}}"
            ),
            vec![("id", s(&pid)), ("st", s(status)), ("by", s(&approver)), ("res", json_dv(result.clone()))],
        )?;
        Ok(json!({"promotion": pid, "status": status, "result": result}))
    }

    /// Branches of a scope: namespace `<scope ns>~<branch>`, with the
    /// scope's readers and writers.
    fn pm_branch(&self, ss: &Session, op: &str, mut a: Map<String, JsonValue>) -> Result<JsonValue> {
        let want = take_str(&mut a, "scope");
        let branch = take_str(&mut a, "branch").ok_or_else(|| miette::miette!("principal {op} needs 'branch'"))?;
        check_id("branch", &branch)?;
        let k = self.scope_of(ss, want.as_deref(), false)?;
        let action = if op == "diff" { "read" } else { "write" };
        if !self.authz(ss)?.allows(action, &k)? {
            bail!("scope '{}' not found", k.ns());
        }
        let ns = k.ns();
        let bns = format!("{ns}~{branch}");
        match op {
            "fork" => {
                a.insert("from".into(), json!(ns));
                a.insert("to".into(), json!(bns));
            }
            "merge" => {
                a.insert("from".into(), json!(bns));
                a.insert("into".into(), json!(ns));
            }
            "discard" => {
                a.insert("ns".into(), json!(bns));
            }
            _ => {
                a.insert("a".into(), json!(ns));
                a.insert("b".into(), json!(bns));
            }
        }
        let v = self.session_view(ss, ss.part.as_deref(), None, action == "write")?;
        let out = v.memory_call(op, JsonValue::Object(a))?;
        self.audit_op(ss, op, json!({"scope": ns, "branch": bns}), None, json!({"effect": "allow"}))?;
        Ok(out)
    }

    /// Automatic promotion of schema-level knowledge: the properties a
    /// scope learned join the workspace's learned layer (`learned:ws`),
    /// which every scope of the workspace reads.
    fn promote_schema(&self, ss: &Session, ns: &str) -> Result<()> {
        if ns == "ws" {
            return Ok(());
        }
        let pv = self.part_view(ss.part.as_deref())?;
        let n = pv.onto_promote_learned(ns, "ws")?;
        if n > 0 {
            self.audit_op(ss, "promote_schema", json!({"from": ns, "to": "ws", "properties": n}), None, json!({"effect": "allow", "automatic": true}))?;
        }
        Ok(())
    }

    /// A live subscription to `relation` of the session's partition, with
    /// rows outside the session's readable scopes filtered out (the
    /// namespace is each memory relation's first key).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn subscribe_as(
        &self,
        ctx: &PrincipalContext,
        relation: &str,
    ) -> Result<(u32, Receiver<(CallbackOp, NamedRows, NamedRows)>)> {
        let ss = self.tenant_session(ctx)?;
        let pv = self.part_view(ss.part.as_deref())?;
        let (id, raw) = pv.register_callback(relation, None);
        let (tx, rx) = unbounded();
        let db = self.root_view();
        self.audit_op(&ss, "subscribe", json!({"relation": relation}), None, json!({"effect": "allow"}))?;
        std::thread::spawn(move || {
            for (op, new, old) in raw.iter() {
                // the scope set as of each change (revocations apply at once)
                let Ok((scopes, _)) = db.scope_set(&ss, "read") else { break };
                let allowed: BTreeSet<String> = scopes.iter().map(|k| k.ns()).collect();
                let keep = |rows: NamedRows| NamedRows {
                    headers: rows.headers,
                    rows: rows.rows.into_iter().filter(|r| r.first().and_then(|v| v.get_str()).is_some_and(|n| allowed.contains(n))).collect(),
                    next: None,
                };
                let (new, old) = (keep(new), keep(old));
                if new.rows.is_empty() && old.rows.is_empty() {
                    continue;
                }
                if tx.send((op, new, old)).is_err() {
                    break;
                }
            }
        });
        Ok((id, rx))
    }
}
