/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Scope-level authorization (docs/specs/tenancy.md, "Authorization"):
//! embedded Cedar over an entity graph projected from the control plane,
//! the intersection rule, and the per-request `ScopeSet`.
//!
//! Cedar decides `(actor, action, scope)` for the handful of candidate
//! scopes of a request, never per memory. The allowed scope namespaces
//! become the pre-filter of every index search (`is_in(ns, $scopes)`).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::str::FromStr;
use std::sync::Arc;

use cedar_policy::{
    Authorizer, Context, Decision, Entities, Entity, EntityId, EntityTypeName, EntityUid, PolicySet, Request,
    RestrictedExpression,
};
use miette::Result;
use serde_json::json;

use super::*;

/// The default policy set: the reader and writer columns of the scope
/// table, promotion, and approvals. Org policies are appended to it.
pub const DEFAULT_POLICIES: &str = r#"
// Workspace-wide scope and legacy namespaces: the workspace's readers and writers.
permit (principal, action == Kleos::Action::"read", resource is Kleos::Scope)
when { (resource.kind == "ws" || resource.kind == "ns") && principal in resource.ws_reader };
permit (principal, action == Kleos::Action::"write", resource is Kleos::Scope)
when { (resource.kind == "ws" || resource.kind == "ns") && principal in resource.ws_writer };

// Team scope: the team's subtree, with a role on the workspace.
permit (principal, action == Kleos::Action::"read", resource is Kleos::Scope)
when { resource.kind == "team" && resource has team && principal in resource.team && principal in resource.ws_reader };
permit (principal, action == Kleos::Action::"write", resource is Kleos::Scope)
when { resource.kind == "team" && resource has team && principal in resource.team && principal in resource.ws_writer };

// Run and agent-user scopes: their agent and their user.
permit (principal, action in [Kleos::Action::"read", Kleos::Action::"write"], resource is Kleos::Scope)
when {
  (resource.kind == "run" || resource.kind == "au") && principal in resource.ws_reader &&
  ((resource has agent && principal in resource.agent) || (resource has user && principal in resource.user))
};

// User scope: the user; an agent acting for that user, on a proven link, may read it.
permit (principal, action in [Kleos::Action::"read", Kleos::Action::"write"], resource is Kleos::Scope)
when { resource.kind == "user" && resource has user && principal in resource.user && principal in resource.ws_reader };
permit (principal, action == Kleos::Action::"read", resource is Kleos::Scope)
when {
  resource.kind == "user" && resource has user && principal.kind == "agent" && principal in resource.ws_reader &&
  context.human_proven && context has human && context.human == resource.user
};

// Agent-global scope: readable in the workspace; written only by promotion.
permit (principal, action == Kleos::Action::"read", resource is Kleos::Scope)
when { resource.kind == "agent" && principal in resource.ws_reader };

// Org scope: members read; org admins (humans) write.
permit (principal, action == Kleos::Action::"read", resource is Kleos::Scope)
when { resource.kind == "org" && principal in resource.org_member };
permit (principal, action == Kleos::Action::"write", resource is Kleos::Scope)
when { resource.kind == "org" && principal.kind == "human" && principal in resource.org_admin };

// Promotion (a declassifying copy): humans who can write the target ...
permit (principal, action == Kleos::Action::"promote", resource is Kleos::Scope)
when {
  principal.kind == "human" && principal in resource.ws_writer &&
  (resource.kind == "ws" || resource.kind == "ns" || resource.kind == "agent" ||
   (resource.kind == "team" && resource has team && principal in resource.team) ||
   (resource.kind == "user" && resource has user && principal in resource.user))
};
// ... agents holding the promote grant ...
permit (principal, action == Kleos::Action::"promote", resource is Kleos::Scope)
when {
  principal.kind == "agent" && principal in resource.ws_promote &&
  (resource.kind == "ws" || resource.kind == "ns" ||
   (resource.kind == "agent" && resource has agent && principal in resource.agent) ||
   (resource.kind == "team" && resource has team && principal in resource.team))
};
// ... and org admins into org scope.
permit (principal, action == Kleos::Action::"promote", resource is Kleos::Scope)
when { resource.kind == "org" && principal.kind == "human" && principal in resource.org_admin };

// Approving a pending promotion: a human who could promote there.
permit (principal, action == Kleos::Action::"approve", resource is Kleos::Scope)
when {
  principal.kind == "human" &&
  ((resource.kind == "org" && principal in resource.org_admin) ||
   (resource.kind != "org" && principal in resource.ws_writer &&
    (resource.kind != "team" || (resource has team && principal in resource.team)) &&
    (resource.kind != "user" || (resource has user && principal in resource.user))))
};
"#;

/// The default policies plus an org's own (validated by parsing).
pub(crate) fn parse_policies(org_text: &str) -> Result<PolicySet> {
    format!("{DEFAULT_POLICIES}\n{org_text}")
        .parse::<PolicySet>()
        .map_err(|e| miette::miette!("invalid Cedar policy: {e}"))
}

fn uid(ty: &str, id: &str) -> EntityUid {
    EntityUid::from_type_name_and_id(EntityTypeName::from_str(ty).expect("static type name"), EntityId::new(id))
}

fn principal_uid(id: &str) -> EntityUid {
    uid("Kleos::Principal", id)
}

fn group_uid(g: &str) -> EntityUid {
    uid("Kleos::Group", g)
}

fn rstr(x: &str) -> RestrictedExpression {
    RestrictedExpression::new_string(x.to_string())
}

fn rent(u: EntityUid) -> RestrictedExpression {
    RestrictedExpression::new_entity_uid(u)
}

/// The authorization state of one session: policies, the projected entity
/// graph, and the principals whose permissions are intersected.
pub(crate) struct Authz {
    policies: Arc<PolicySet>,
    /// Principals, teams and groups.
    base: BTreeMap<EntityUid, Entity>,
    ws: String,
    org: String,
    actors: Vec<String>,
    token: Option<TokenGrant>,
    human: Option<String>,
    human_proven: bool,
    /// Teams (with their ancestors) of the human and the agent: the team
    /// scopes worth asking about.
    pub(crate) teams: BTreeSet<String>,
}

impl Authz {
    fn scope_entity(&self, k: &ScopeKey) -> Result<Entity> {
        let ws = &self.ws;
        let org = &self.org;
        let mut attrs: HashMap<String, RestrictedExpression> = HashMap::from([
            ("kind".to_string(), rstr(k.kind())),
            ("ws_reader".to_string(), rent(group_uid(&format!("ws:{ws}/reader")))),
            ("ws_writer".to_string(), rent(group_uid(&format!("ws:{ws}/writer")))),
            ("ws_admin".to_string(), rent(group_uid(&format!("ws:{ws}/admin")))),
            ("ws_promote".to_string(), rent(group_uid(&format!("ws:{ws}/promote")))),
            ("org_member".to_string(), rent(group_uid(&format!("org:{org}/member")))),
            ("org_admin".to_string(), rent(group_uid(&format!("org:{org}/admin")))),
        ]);
        match k {
            ScopeKey::Run { agent, user, .. } => {
                attrs.insert("agent".into(), rent(principal_uid(agent)));
                if let Some(u) = user {
                    attrs.insert("user".into(), rent(principal_uid(u)));
                }
            }
            ScopeKey::AgentUser { agent, user } => {
                attrs.insert("agent".into(), rent(principal_uid(agent)));
                attrs.insert("user".into(), rent(principal_uid(user)));
            }
            ScopeKey::User { user } => {
                attrs.insert("user".into(), rent(principal_uid(user)));
            }
            ScopeKey::Agent { agent } => {
                attrs.insert("agent".into(), rent(principal_uid(agent)));
            }
            ScopeKey::Team { team } => {
                attrs.insert("team".into(), rent(uid("Kleos::Team", team)));
            }
            _ => {}
        }
        Entity::new(uid("Kleos::Scope", &k.ns()), attrs, HashSet::new()).map_err(|e| miette::miette!("{e}"))
    }

    /// For each scope: may every actor (and the credential) do `action`
    /// on it? The intersection rule (docs/specs/tenancy.md).
    pub(crate) fn allowed(&self, action: &str, scopes: &[ScopeKey]) -> Result<Vec<bool>> {
        if scopes.is_empty() {
            return Ok(vec![]);
        }
        let mut ents = self.base.clone();
        for k in scopes {
            let e = self.scope_entity(k)?;
            ents.insert(e.uid(), e);
        }
        let entities = Entities::from_entities(ents.into_values(), None).map_err(|e| miette::miette!("{e}"))?;
        let mut cx = vec![("human_proven".to_string(), RestrictedExpression::new_bool(self.human_proven))];
        if let Some(h) = &self.human {
            cx.push(("human".to_string(), rent(principal_uid(h))));
        }
        let context = Context::from_pairs(cx).map_err(|e| miette::miette!("{e}"))?;
        let auth = Authorizer::new();
        let act = uid("Kleos::Action", action);
        let mut out = Vec::with_capacity(scopes.len());
        for k in scopes {
            if !self.token.as_ref().map_or(true, |t| t.allows(action, k.kind())) || self.actors.is_empty() {
                out.push(false);
                continue;
            }
            let res = uid("Kleos::Scope", &k.ns());
            let mut ok = true;
            for a in &self.actors {
                let req = Request::new(principal_uid(a), act.clone(), res.clone(), context.clone(), None)
                    .map_err(|e| miette::miette!("{e}"))?;
                if auth.is_authorized(&req, &self.policies, &entities).decision() != Decision::Allow {
                    ok = false;
                    break;
                }
            }
            out.push(ok);
        }
        Ok(out)
    }

    pub(crate) fn allows(&self, action: &str, k: &ScopeKey) -> Result<bool> {
        Ok(self.allowed(action, std::slice::from_ref(k))?[0])
    }
}

impl<S> Db<S>
where
    S: for<'a> Storage<'a> + Clone + 'static,
{
    fn org_policies(&self, org: &str) -> Result<Arc<PolicySet>> {
        let text = self.org_policy(org)?.get("cedar").and_then(|x| x.as_str()).unwrap_or_default().to_string();
        let mut g = self.tenancy.policies.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((t, p)) = g.get(org) {
            if *t == text {
                return Ok(p.clone());
            }
        }
        let p = Arc::new(parse_policies(&text)?);
        g.insert(org.to_string(), (text, p.clone()));
        Ok(p)
    }

    /// Project the control plane into Cedar entities for a session.
    pub(crate) fn authz(&self, ss: &Session) -> Result<Authz> {
        let ws = &ss.ws;
        let org = &ss.org;
        let actors: Vec<&Principal> = ss.actors();
        let mut pids: BTreeSet<String> = actors.iter().map(|p| p.id.clone()).collect();
        for p in &actors {
            if let Some(t) = &p.template {
                pids.insert(t.clone());
            }
        }
        let pid_list: Vec<String> = pids.iter().cloned().collect();
        let members = self.tn_rows(
            &format!("?[grp, principal, role] := *{TN_MEMBER}{{grp, principal, role}}, principal in $pids"),
            vec![("pids", str_list(&pid_list))],
        )?;
        let teams = self.tn_rows(&format!("?[team, parent] := *{TN_TEAM}{{team, org: $o, parent}}"), vec![("o", s(org))])?;
        let grants = self.tn_rows(
            &format!("?[grantee, role, actions] := *{TN_GRANT}{{ws: $ws, grantee, role, actions}}"),
            vec![("ws", s(ws))],
        )?;
        let mut grant_groups: BTreeMap<String, Vec<EntityUid>> = BTreeMap::new();
        for g in &grants {
            let (Some(grantee), Some(role)) = (cell_str(&g[0]), cell_str(&g[1])) else { continue };
            let e = grant_groups.entry(grantee).or_default();
            e.push(group_uid(&format!("ws:{ws}/{role}")));
            if let DataValue::List(acts) = &g[2] {
                if acts.iter().any(|a| a.get_str() == Some("promote")) {
                    e.push(group_uid(&format!("ws:{ws}/promote")));
                }
            }
        }
        let mut base: BTreeMap<EntityUid, Entity> = BTreeMap::new();
        let mut put = |e: Entity| {
            base.insert(e.uid(), e);
        };
        // role groups: admin ∈ writer ∈ reader; owner ∈ admin ∈ member
        let chain = |names: &[String]| -> Vec<Entity> {
            names
                .iter()
                .enumerate()
                .map(|(i, n)| {
                    let parents: HashSet<EntityUid> = names.get(i + 1).map(|p| group_uid(p)).into_iter().collect();
                    Entity::new_no_attrs(group_uid(n), parents)
                })
                .collect()
        };
        for e in chain(&["admin", "writer", "reader"].map(|r| format!("ws:{ws}/{r}"))) {
            put(e);
        }
        put(Entity::new_no_attrs(group_uid(&format!("ws:{ws}/promote")), HashSet::new()));
        for e in chain(&["owner", "admin", "member"].map(|r| format!("org:{org}/{r}"))) {
            put(e);
        }
        // teams: parent team plus their grants
        let mut parent_of: BTreeMap<String, Option<String>> = BTreeMap::new();
        for t in &teams {
            let Some(id) = cell_str(&t[0]) else { continue };
            let parent = cell_str(&t[1]);
            let mut parents: HashSet<EntityUid> = parent.iter().map(|p| uid("Kleos::Team", p)).collect();
            parents.extend(grant_groups.get(&format!("t:{id}")).cloned().unwrap_or_default());
            put(Entity::new_no_attrs(uid("Kleos::Team", &id), parents));
            parent_of.insert(id, parent);
        }
        // principals
        let mut by_principal: BTreeMap<String, HashSet<EntityUid>> = BTreeMap::new();
        let mut direct_teams: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for m in &members {
            let (Some(grp), Some(p), Some(role)) = (cell_str(&m[0]), cell_str(&m[1]), cell_str(&m[2])) else { continue };
            let e = by_principal.entry(p.clone()).or_default();
            match grp.split_once(':') {
                Some(("org", o)) if o == org => {
                    e.insert(group_uid(&format!("org:{org}/{role}")));
                }
                Some(("team", t)) if parent_of.contains_key(t) => {
                    e.insert(uid("Kleos::Team", t));
                    direct_teams.entry(p).or_default().push(t.to_string());
                }
                _ => {}
            }
        }
        let kinds: BTreeMap<String, (String, Option<String>)> = {
            let mut m = BTreeMap::new();
            for p in &actors {
                m.insert(p.id.clone(), (p.kind.clone(), p.template.clone()));
            }
            for p in &actors {
                if let Some(t) = &p.template {
                    m.entry(t.clone()).or_insert(("agent".to_string(), None));
                }
            }
            m
        };
        for (id, (kind, template)) in &kinds {
            let mut parents = by_principal.remove(id).unwrap_or_default();
            parents.extend(grant_groups.get(&format!("p:{id}")).cloned().unwrap_or_default());
            if let Some(t) = template {
                parents.insert(principal_uid(t));
            }
            let attrs = HashMap::from([("kind".to_string(), rstr(kind))]);
            put(Entity::new(principal_uid(id), attrs, parents).map_err(|e| miette::miette!("{e}"))?);
        }
        // teams worth asking about: the human's and the agent's, with ancestors
        let mut team_set = BTreeSet::new();
        for p in ss.human.iter().chain(ss.agent.iter()).chain(ss.service.iter()) {
            for id in std::iter::once(p.id.clone()).chain(p.template.clone()) {
                for t in direct_teams.get(&id).cloned().unwrap_or_default() {
                    let mut cur = Some(t);
                    while let Some(c) = cur {
                        if !team_set.insert(c.clone()) {
                            break;
                        }
                        cur = parent_of.get(&c).cloned().flatten();
                    }
                }
            }
        }
        Ok(Authz {
            policies: self.org_policies(org)?,
            base,
            ws: ws.clone(),
            org: org.clone(),
            actors: actors.iter().map(|p| p.id.clone()).collect(),
            token: ss.ctx.token.clone(),
            human: ss.human.as_ref().map(|h| h.id.clone()),
            human_proven: ss.human_proven,
            teams: team_set,
        })
    }

    /// The scope namespaces of a partition.
    pub(crate) fn part_namespaces(&self, part: Option<&str>) -> Result<Vec<String>> {
        let pv = self.part_view(part)?;
        match pv.run_script(&format!("?[ns] := *{}{{ns}}", crate::memory::MEM_NS), Default::default(), ScriptMutability::Immutable) {
            Ok(r) => Ok(r.rows.into_iter().filter_map(|r| cell_str(&r[0])).collect()),
            Err(_) => Ok(vec![]),
        }
    }

    /// The ScopeSet of a session for `action`: the existing scopes of the
    /// workspace it may act on, narrowest first, and whether the org scope
    /// is included. Cached by (session, membership epoch, scope epoch).
    pub(crate) fn scope_set(&self, ss: &Session, action: &str) -> Result<(Arc<Vec<ScopeKey>>, bool)> {
        let ep = self.epochs(&ss.ws)?;
        let key = json!({
            "ws": ss.ws, "action": action, "actors": ss.actors().iter().map(|p| &p.id).collect::<Vec<_>>(),
            "human": ss.human.as_ref().map(|h| &h.id), "proven": ss.human_proven, "token": ss.ctx.token,
        })
        .to_string();
        let org_key = format!("{key}#org");
        {
            let c = self.tenancy.scope_cache.lock().unwrap_or_else(|e| e.into_inner());
            if let (Some((e1, v)), Some((e2, o))) = (c.get(&key), c.get(&org_key)) {
                if *e1 == ep && *e2 == ep {
                    return Ok((v.clone(), !o.is_empty()));
                }
            }
        }
        let az = self.authz(ss)?;
        let human = ss.human.as_ref().map(|h| h.id.as_str());
        let agent = ss.mem_agent();
        let mut cands: Vec<ScopeKey> = self
            .part_namespaces(ss.part.as_deref())?
            .into_iter()
            .filter(|ns| !ns.contains('~'))
            .map(|ns| ScopeKey::parse(&ns, false))
            .filter(|k| match k {
                // an autonomous agent never reads per-user scopes
                ScopeKey::Run { agent: a, user, .. } => {
                    (agent == Some(a.as_str()) && (human.is_some() || user.is_none())) || (human.is_some() && user.as_deref() == human)
                }
                ScopeKey::AgentUser { agent: a, user } => human.is_some() && (agent == Some(a.as_str()) || Some(user.as_str()) == human),
                ScopeKey::User { user } => Some(user.as_str()) == human,
                ScopeKey::Team { team } => az.teams.contains(team),
                ScopeKey::Org => false,
                _ => true,
            })
            .collect();
        cands.sort_by(|a, b| (a.rank(), a.ns()).cmp(&(b.rank(), b.ns())));
        cands.dedup();
        let ok = az.allowed(action, &cands)?;
        let scopes: Vec<ScopeKey> = cands.into_iter().zip(ok).filter(|x| x.1).map(|x| x.0).collect();
        let org_part = ss.org_part();
        let org = self.part_namespaces(Some(&org_part))?.iter().any(|n| n == "org") && az.allows(action, &ScopeKey::Org)?;
        let scopes = Arc::new(scopes);
        let mut c = self.tenancy.scope_cache.lock().unwrap_or_else(|e| e.into_inner());
        c.insert(key, (ep, scopes.clone()));
        c.insert(org_key, (ep, Arc::new(if org { vec![ScopeKey::Org] } else { vec![] })));
        Ok((scopes, org))
    }
}
