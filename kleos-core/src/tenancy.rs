/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Tenancy, identity and access control (docs/specs/tenancy.md).
//!
//! The control plane (orgs, workspaces, teams, principals, memberships,
//! grants, keys, the audit log) lives in global `kleos_tn_*` relations.
//! Each workspace's data lives in its own partition view
//! ([`Db::partition_view`]); inside it a scope is a memory namespace, so
//! the memory API's per-namespace guarantees are per-scope guarantees.
//!
//! This file: the model, admin ops (`::tenant`), principal contexts and
//! actor chains, scope keys, the audit log and erasure. Authorization:
//! `tenancy_authz.rs`. Memory ops as a principal: `tenancy_mem.rs`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use miette::{IntoDiagnostic, Result};
use serde_derive::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use sha2::{Digest, Sha256};
use smartstring::{LazyCompact, SmartString};

use crate::bail;
use crate::data::value::DataValue;
use crate::data::tuple::TupleT;
use crate::runtime::db::{Db, NamedRows, ScriptMutability};
use crate::runtime::relation::try_decode_tuple_from_kv;
use crate::runtime::transact::{CommitAudit, SessionTx};
use crate::storage::Storage;
use crate::JsonData;

#[path = "tenancy_authz.rs"]
pub(crate) mod authz;
#[path = "tenancy_mem.rs"]
pub(crate) mod mem;
#[path = "tenancy_otel.rs"]
pub mod otel;
#[path = "tenancy_skill.rs"]
pub(crate) mod skill;

pub use authz::DEFAULT_POLICIES;

pub const TN_ORG: &str = "kleos_tn_org";
pub const TN_WS: &str = "kleos_tn_ws";
pub const TN_TEAM: &str = "kleos_tn_team";
pub const TN_PRINCIPAL: &str = "kleos_tn_principal";
pub const TN_IDENT: &str = "kleos_tn_ident";
pub const TN_MEMBER: &str = "kleos_tn_member";
pub const TN_GRANT: &str = "kleos_tn_grant";
pub const TN_APIKEY: &str = "kleos_tn_apikey";
pub const TN_KEY: &str = "kleos_tn_key";
pub const TN_PROMOTION: &str = "kleos_tn_promotion";
pub const TN_AUDIT: &str = "kleos_tn_audit";
pub const TN_META: &str = "kleos_tn_meta";
/// The head of each workspace's audit chain, read and written in the
/// transaction that appends an event.
pub const TN_AUDIT_HEAD: &str = "kleos_tn_audit_head";
/// Pending curation requests (forget / set in agent and org scopes by a
/// principal who is not a curator there).
pub const TN_REQUEST: &str = "kleos_tn_request";
/// Public keys of principals (agents authenticating with `private_key_jwt`).
pub const TN_JWKS: &str = "kleos_tn_jwks";

/// The implicit org and workspace of a store (the unpartitioned store).
pub const DEFAULT: &str = "default";
/// Team tree depth limit.
const MAX_TEAM_DEPTH: usize = 4;
/// Actor chain depth limit (deeper is an error, never truncated).
pub const MAX_CHAIN: usize = 6;

const DDL: &[(&str, &str)] = &[
    (TN_ORG, "{org: String => name: String, policy: Json, created: Int, status: String}"),
    (TN_WS, "{ws: String => org: String, name: String, partition: String?, created: Int, status: String}"),
    (TN_TEAM, "{team: String => org: String, name: String, parent: String?, created: Int}"),
    (
        TN_PRINCIPAL,
        "{id: String => org: String, kind: String, name: String, sponsor: String?, template: String?, status: String, created: Int}",
    ),
    (TN_IDENT, "{iss: String, sub: String => principal: String}"),
    (TN_MEMBER, "{grp: String, principal: String => role: String, created: Int}"),
    (TN_GRANT, "{ws: String, grantee: String => role: String, actions: [String], created: Int}"),
    (
        TN_APIKEY,
        "{hash: String => principal: String, actions: [String], label: String?, created: Int, expires: Int?}",
    ),
    (TN_KEY, "{principal: String => key: String, created: Int}"),
    (
        TN_PROMOTION,
        "{id: String => ws: String, from_ns: String, ids: [String], target: String, chain: Json, requested_by: String, \
         status: String, approved_by: String?, result: Json?, at: Int}",
    ),
    (
        TN_AUDIT,
        "{ws: String, seq: Int => ts: Int, op: String, chain: Json, accountable: String?, targets: Json, \
         relations: [String], scope_set: String?, decision: Json, otel: Json, prev: String, hash: String}",
    ),
    (TN_META, "{key: String => value: Json}"),
    (TN_AUDIT_HEAD, "{ws: String => seq: Int, hash: String}"),
    (TN_JWKS, "{principal: String => jwks: Json, updated: Int}"),
    (
        TN_REQUEST,
        "{id: String => ws: String, op: String, ns: String, ids: [String], args: Json, chain: Json, requested_by: String, \
         status: String, approved_by: String?, result: Json?, at: Int}",
    ),
];

pub(crate) fn now_micros() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_micros() as i64).unwrap_or(0)
}

pub(crate) fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn s(v: &str) -> DataValue {
    DataValue::from(v)
}

fn opt_s(v: &Option<String>) -> DataValue {
    v.as_deref().map(DataValue::from).unwrap_or(DataValue::Null)
}

fn str_list(v: &[String]) -> DataValue {
    DataValue::List(v.iter().map(|x| DataValue::from(x.as_str())).collect())
}

pub(crate) fn json_dv(v: JsonValue) -> DataValue {
    DataValue::Json(JsonData(v))
}

fn cell_str(v: &DataValue) -> Option<String> {
    v.get_str().map(|x| x.to_string())
}

fn cell_json(v: &DataValue) -> JsonValue {
    match v {
        DataValue::Json(j) => j.0.clone(),
        DataValue::Null => JsonValue::Null,
        other => JsonValue::from(other.clone()),
    }
}

fn params(p: Vec<(&str, DataValue)>) -> BTreeMap<String, DataValue> {
    p.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// A principal, team, workspace or org id: `[A-Za-z0-9._-]{1,64}`.
pub(crate) fn check_id(what: &str, id: &str) -> Result<()> {
    if id.is_empty() || id.len() > 64 || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-') {
        bail!("invalid {what} id '{id}': use 1-64 characters of [A-Za-z0-9._-]");
    }
    Ok(())
}

// ------------------------------------------------------------------ model

/// Principal kinds.
pub const KINDS: &[&str] = &["human", "agent", "service"];
/// Workspace roles, weakest first.
pub const WS_ROLES: &[&str] = &["reader", "writer", "admin"];
/// Org roles, weakest first.
pub const ORG_ROLES: &[&str] = &["member", "admin", "owner"];

/// A principal row.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Principal {
    pub id: String,
    pub org: String,
    pub kind: String,
    pub name: String,
    pub sponsor: Option<String>,
    pub template: Option<String>,
    pub status: String,
}

impl Principal {
    /// The id an agent's memory is keyed by: its template, else itself.
    pub fn mem_id(&self) -> &str {
        self.template.as_deref().unwrap_or(&self.id)
    }
}

/// How a link of an actor chain was established.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Default)]
#[serde(rename_all = "snake_case")]
pub enum Assurance {
    /// kleos authenticated it (token signature, API key).
    Verified,
    /// An inner `act` of a token signed by a trusted issuer.
    IssuerEndorsed,
    /// The embedding host process supplied it.
    #[default]
    HostAsserted,
    /// A header or SDK field with no proof.
    SelfAsserted,
}

impl Assurance {
    /// Strong enough to act on a person's own memory.
    pub fn proven(self) -> bool {
        self != Assurance::SelfAsserted
    }
}

/// One link of an actor chain (OAuth Actor Profile: `iss`, `sub`,
/// `sub_profile`), innermost (the originator) first.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ActorLink {
    /// `human`, `ai_agent` or `service` (`sub_profile`; `user` is read as
    /// `human`).
    pub kind: String,
    pub iss: Option<String>,
    pub sub: String,
    /// The kleos principal the link maps to.
    pub principal: Option<String>,
    pub assurance: Assurance,
}

/// What the credential of a request allows (OAuth scopes, API key actions).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Hash)]
#[serde(default, deny_unknown_fields)]
pub struct TokenGrant {
    /// `read`, `write`, `promote`, `admin`; `None`: all.
    pub actions: Option<Vec<String>>,
    /// Scope kinds the credential may touch; `None`: all.
    pub kinds: Option<Vec<String>>,
}

impl TokenGrant {
    pub fn allows(&self, action: &str, kind: &str) -> bool {
        self.actions.as_ref().map_or(true, |a| a.iter().any(|x| x == action))
            && self.kinds.as_ref().map_or(true, |k| k.iter().any(|x| x == kind))
    }
}

/// Who is calling (docs/specs/tenancy.md, "Principal context and actor
/// chain"). Embedded hosts build it; server mode derives it from the
/// credential.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct PrincipalContext {
    /// Target workspace; default `default`.
    pub workspace: Option<String>,
    /// The originating human principal.
    pub human: Option<String>,
    /// The acting agent principal.
    pub agent: Option<String>,
    /// A service principal (non-agent machine identity).
    pub service: Option<String>,
    /// Run / session id.
    pub run: Option<String>,
    /// Other principals of a shared session: they only take access away.
    pub participants: Vec<String>,
    /// The full chain, innermost first; derived from `human` / `agent` when
    /// empty.
    pub chain: Vec<ActorLink>,
    /// What the credential allows.
    pub token: Option<TokenGrant>,
    /// Assurance of `human` / `agent` when no chain is given (default
    /// `host_asserted`).
    pub assurance: Option<Assurance>,
    pub purpose: Option<String>,
    pub txn: Option<String>,
    pub client_id: Option<String>,
}

// ------------------------------------------------------------------ scopes

/// A typed scope (docs/specs/tenancy.md, "Scopes"); stored as a memory
/// namespace inside its partition.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScopeKey {
    Run { agent: String, user: Option<String>, run: String },
    /// A shared session's memory: an ad-hoc intersection scope readable
    /// only by its members (and its agent).
    Session { agent: Option<String>, run: String, members: Vec<String> },
    AgentUser { agent: String, user: String },
    User { user: String },
    Agent { agent: String },
    Team { team: String },
    Workspace,
    Org,
    /// A namespace of the implicit workspace that predates tenancy.
    Legacy { ns: String },
}

fn is_id(x: &str) -> bool {
    check_id("", x).is_ok()
}

impl ScopeKey {
    /// The scope a namespace names. `org_part`: the namespace is in an org
    /// partition. A branch `<ns>~<branch>` is the scope of `<ns>`.
    pub fn parse(ns: &str, org_part: bool) -> ScopeKey {
        let base = ns.split_once('~').map_or(ns, |x| x.0);
        let parts: Vec<&str> = base.split(':').collect();
        let k = match parts.as_slice() {
            ["run", a, u, r] if is_id(a) && is_id(r) && (*u == "-" || is_id(u)) => Some(ScopeKey::Run {
                agent: a.to_string(),
                user: (*u != "-").then(|| u.to_string()),
                run: r.to_string(),
            }),
            ["sess", a, r, ms] if (*a == "-" || is_id(a)) && is_id(r) && !ms.is_empty() && ms.split('+').all(is_id) => Some(ScopeKey::Session {
                agent: (*a != "-").then(|| a.to_string()),
                run: r.to_string(),
                members: ms.split('+').map(|m| m.to_string()).collect(),
            }),
            ["au", a, u] if is_id(a) && is_id(u) => Some(ScopeKey::AgentUser { agent: a.to_string(), user: u.to_string() }),
            ["user", u] if is_id(u) => Some(ScopeKey::User { user: u.to_string() }),
            ["agent", a] if is_id(a) => Some(ScopeKey::Agent { agent: a.to_string() }),
            ["team", t] if is_id(t) => Some(ScopeKey::Team { team: t.to_string() }),
            ["ws"] => Some(ScopeKey::Workspace),
            ["org"] if org_part => Some(ScopeKey::Org),
            _ => None,
        };
        k.unwrap_or_else(|| ScopeKey::Legacy { ns: base.to_string() })
    }

    /// The namespace of the scope.
    pub fn ns(&self) -> String {
        match self {
            ScopeKey::Run { agent, user, run } => format!("run:{agent}:{}:{run}", user.as_deref().unwrap_or("-")),
            ScopeKey::Session { agent, run, members } => format!("sess:{}:{run}:{}", agent.as_deref().unwrap_or("-"), members.join("+")),
            ScopeKey::AgentUser { agent, user } => format!("au:{agent}:{user}"),
            ScopeKey::User { user } => format!("user:{user}"),
            ScopeKey::Agent { agent } => format!("agent:{agent}"),
            ScopeKey::Team { team } => format!("team:{team}"),
            ScopeKey::Workspace => "ws".to_string(),
            ScopeKey::Org => "org".to_string(),
            ScopeKey::Legacy { ns } => ns.clone(),
        }
    }

    /// The kind as used by policies and options.
    pub fn kind(&self) -> &'static str {
        match self {
            ScopeKey::Run { .. } => "run",
            ScopeKey::Session { .. } => "sess",
            ScopeKey::AgentUser { .. } => "au",
            ScopeKey::User { .. } => "user",
            ScopeKey::Agent { .. } => "agent",
            ScopeKey::Team { .. } => "team",
            ScopeKey::Workspace => "ws",
            ScopeKey::Org => "org",
            ScopeKey::Legacy { .. } => "ns",
        }
    }

    /// Precedence, narrowest first: run > agent_user > user > agent > team
    /// > workspace / legacy > org.
    pub fn rank(&self) -> u8 {
        match self {
            ScopeKey::Run { .. } | ScopeKey::Session { .. } => 0,
            ScopeKey::AgentUser { .. } => 1,
            ScopeKey::User { .. } => 2,
            ScopeKey::Agent { .. } => 3,
            ScopeKey::Team { .. } => 4,
            ScopeKey::Workspace | ScopeKey::Legacy { .. } => 5,
            ScopeKey::Org => 6,
        }
    }

    /// The principals whose own scope this is (erasure, candidate filter).
    pub fn owners(&self) -> Vec<&str> {
        match self {
            ScopeKey::Run { agent, user, .. } => {
                let mut v = vec![agent.as_str()];
                v.extend(user.as_deref());
                v
            }
            ScopeKey::Session { agent, members, .. } => agent.iter().map(|a| a.as_str()).chain(members.iter().map(|m| m.as_str())).collect(),
            ScopeKey::AgentUser { agent, user } => vec![agent, user],
            ScopeKey::User { user } => vec![user],
            ScopeKey::Agent { agent } => vec![agent],
            _ => vec![],
        }
    }
}

/// Normalised scope kind names accepted in options.
pub(crate) fn kind_alias(k: &str) -> Option<&'static str> {
    Some(match k {
        "run" => "run",
        "sess" | "session" => "sess",
        "au" | "agent_user" => "au",
        "user" => "user",
        "agent" => "agent",
        "team" => "team",
        "ws" | "workspace" => "ws",
        "org" => "org",
        "ns" | "legacy" => "ns",
        _ => return None,
    })
}

// ------------------------------------------------------------------ state

/// Process-side tenancy state shared by every view of one `Db`.
#[derive(Default)]
pub struct TenancyState {
    /// Serialises audit appends in this process.
    pub(crate) audit_lock: Mutex<()>,
    /// Whether this process checked the control plane's version.
    upgraded: std::sync::atomic::AtomicBool,
    /// ScopeSet cache: key → (epochs, scopes).
    pub(crate) scope_cache: Mutex<HashMap<String, ((i64, i64), Arc<Vec<ScopeKey>>)>>,
    /// Parsed policy sets by org: (text, policies).
    pub(crate) policies: Mutex<HashMap<String, (String, Arc<cedar_policy::PolicySet>)>>,
    /// Pseudonym keys by principal (dropped on erasure).
    keys: Mutex<HashMap<String, Vec<u8>>>,
    /// Whether the control plane exists (0 unknown, 1 no, 2 yes).
    enabled: AtomicU64,
}

// ------------------------------------------------------------------ admin

/// A resolved principal call: who, where, and the chain recorded with it.
#[derive(Debug, Clone)]
pub(crate) struct Session {
    pub ws: String,
    pub org: String,
    /// The workspace partition (`None`: the unpartitioned default workspace).
    pub part: Option<String>,
    pub human: Option<Principal>,
    pub agent: Option<Principal>,
    pub service: Option<Principal>,
    pub participants: Vec<Principal>,
    /// The agent's sponsor when the org caps autonomous agents at their
    /// sponsor's permissions and there is no human.
    pub ceiling: Option<Principal>,
    pub accountable: Option<String>,
    pub chain: Vec<ActorLink>,
    pub human_proven: bool,
    pub ctx: PrincipalContext,
}

impl Session {
    /// The partition of org scope.
    pub fn org_part(&self) -> String {
        format!("org_{}", self.org)
    }
    /// Every principal whose permissions are intersected.
    pub fn actors(&self) -> Vec<&Principal> {
        let mut v: Vec<&Principal> = vec![];
        v.extend(self.human.iter());
        v.extend(self.agent.iter());
        v.extend(self.service.iter());
        v.extend(self.participants.iter());
        v.extend(self.ceiling.iter());
        v
    }
    /// The id memory is keyed by for the acting agent.
    pub fn mem_agent(&self) -> Option<&str> {
        self.agent.as_ref().map(|a| a.mem_id())
    }
}

impl<S> Db<S>
where
    S: for<'a> Storage<'a> + Clone + 'static,
{
    pub(crate) fn tn_q(&self, script: &str, p: Vec<(&str, DataValue)>) -> Result<NamedRows> {
        self.root_view().run_script(script, params(p), ScriptMutability::Mutable)
    }

    fn tn_rows(&self, script: &str, p: Vec<(&str, DataValue)>) -> Result<Vec<Vec<DataValue>>> {
        Ok(self.tn_q(script, p)?.rows)
    }

    /// Whether this store has a tenancy control plane.
    pub fn tenancy_enabled(&self) -> bool {
        match self.tenancy.enabled.load(Ordering::Relaxed) {
            2 => true,
            1 => false,
            _ => {
                let on = self
                    .root_view()
                    .run_script(&format!("?[v] := *{TN_META}{{key: 'version', value: v}}"), Default::default(), ScriptMutability::Immutable)
                    .is_ok();
                // "no" is not cached: another handle may enable it
                if on {
                    self.tenancy.enabled.store(2, Ordering::Relaxed);
                }
                on
            }
        }
    }

    /// Create the control plane with the implicit org and workspace (the
    /// unpartitioned store). Idempotent.
    pub fn tenancy_ensure(&self) -> Result<()> {
        if self.tenancy_enabled() {
            return self.tenancy_upgrade();
        }
        let root = self.root_view();
        let have: BTreeSet<String> = root
            .run_script("::relations", Default::default(), ScriptMutability::Immutable)?
            .rows
            .iter()
            .filter_map(|r| r.first().and_then(|v| v.get_str()).map(|x| x.to_string()))
            .collect();
        for (name, cols) in DDL {
            if !have.contains(*name) {
                root.run_script(&format!(":create {name} {cols}"), Default::default(), ScriptMutability::Mutable)?;
            }
        }
        let now = now_micros();
        self.tn_q(
            &format!(
                "{{?[org, name, policy, created, status] <- [[$d, $d, $pol, $now, 'active']] :put {TN_ORG} {{org => name, policy, created, status}}}}\
                 {{?[ws, org, name, partition, created, status] <- [[$d, $d, $d, null, $now, 'active']] \
                   :put {TN_WS} {{ws => org, name, partition, created, status}}}}\
                 {{?[key, value] <- [['epoch', 1], ['version', 1]] :put {TN_META} {{key => value}}}}"
            ),
            vec![("d", s(DEFAULT)), ("now", DataValue::from(now)), ("pol", json_dv(json!({})))],
        )?;
        self.new_key(&format!("org:{DEFAULT}"))?;
        self.tenancy.enabled.store(2, Ordering::Relaxed);
        Ok(())
    }

    /// Bring an older control plane up to date, once per process: the audit
    /// head relation (filled from each chain's last event).
    fn tenancy_upgrade(&self) -> Result<()> {
        if self.tenancy.upgraded.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        let root = self.root_view();
        let have: BTreeSet<String> = root
            .run_script("::relations", Default::default(), ScriptMutability::Immutable)?
            .rows
            .iter()
            .filter_map(|r| r.first().and_then(|v| v.get_str()).map(|x| x.to_string()))
            .collect();
        for (name, cols) in DDL {
            if !have.contains(*name) {
                root.run_script(&format!(":create {name} {cols}"), Default::default(), ScriptMutability::Mutable)?;
            }
        }
        if !have.contains(TN_AUDIT_HEAD) {
            self.tn_q(
                &format!(
                    "last[ws, max(seq)] := *{TN_AUDIT}{{ws, seq}}\n\
                     ?[ws, seq, hash] := last[ws, seq], *{TN_AUDIT}{{ws, seq, hash}} :put {TN_AUDIT_HEAD} {{ws => seq, hash}}"
                ),
                vec![],
            )?;
        }
        Ok(())
    }

    /// Bump the membership epoch: every cached ScopeSet is stale.
    fn bump_epoch(&self) -> Result<()> {
        // a timestamp, so concurrent bumps from two processes still move it
        let cur = self.epochs("")?.0;
        self.tn_q(
            &format!("?[key, value] <- [['epoch', $v]] :put {TN_META} {{key => value}}"),
            vec![("v", json_dv(json!(now_micros().max(cur + 1))))],
        )?;
        self.tenancy.scope_cache.lock().unwrap_or_else(|e| e.into_inner()).clear();
        Ok(())
    }

    /// `(membership epoch, scope epoch of ws)`.
    pub(crate) fn epochs(&self, ws: &str) -> Result<(i64, i64)> {
        let mut e = (0, 0);
        for r in self.tn_rows(
            &format!("?[k, v] := *{TN_META}{{key: k, value: v}}, k in ['epoch', $sk]"),
            vec![("sk", s(&format!("scope_epoch:{ws}")))],
        )? {
            let v = cell_json(&r[1]).as_i64().unwrap_or(0);
            if r[0].get_str() == Some("epoch") {
                e.0 = v;
            } else {
                e.1 = v;
            }
        }
        Ok(e)
    }

    /// A new scope namespace appeared in `ws`: cached ScopeSets of the
    /// workspace must see it.
    pub(crate) fn bump_scope_epoch(&self, ws: &str) -> Result<()> {
        let k = format!("scope_epoch:{ws}");
        let cur = self
            .tn_rows(&format!("?[v] := *{TN_META}{{key: $k, value: v}}"), vec![("k", s(&k))])?
            .first()
            .map(|r| cell_json(&r[0]).as_i64().unwrap_or(0))
            .unwrap_or(0);
        self.tn_q(
            &format!("?[key, value] <- [[$k, $v]] :put {TN_META} {{key => value}}"),
            vec![("k", s(&k)), ("v", json_dv(json!(cur + 1)))],
        )?;
        Ok(())
    }

    fn new_key(&self, owner: &str) -> Result<()> {
        let key: [u8; 32] = rand::random();
        self.tn_q(
            &format!("?[principal, key, created] <- [[$p, $k, $now]] :put {TN_KEY} {{principal => key, created}}"),
            vec![("p", s(owner)), ("k", s(&hex(&key))), ("now", DataValue::from(now_micros()))],
        )?;
        Ok(())
    }

    /// `HMAC-SHA-256(key[owner], id)`, shortened: the pseudonym of `id` in
    /// audit events. `None` once the key was destroyed (erasure).
    pub(crate) fn pseudo(&self, owner: &str, id: &str) -> Result<Option<String>> {
        let cached = self.tenancy.keys.lock().unwrap_or_else(|e| e.into_inner()).get(owner).cloned();
        let key = match cached {
            Some(k) => k,
            None => {
                let rows = self.tn_rows(&format!("?[k] := *{TN_KEY}{{principal: $p, key: k}}"), vec![("p", s(owner))])?;
                let Some(k) = rows.first().and_then(|r| cell_str(&r[0])) else { return Ok(None) };
                let bytes: Vec<u8> = (0..k.len() / 2).filter_map(|i| u8::from_str_radix(&k[2 * i..2 * i + 2], 16).ok()).collect();
                self.tenancy.keys.lock().unwrap_or_else(|e| e.into_inner()).insert(owner.to_string(), bytes.clone());
                bytes
            }
        };
        let mut m = Hmac::<Sha256>::new_from_slice(&key).into_diagnostic()?;
        m.update(id.as_bytes());
        Ok(Some(format!("p:{}", &hex(&m.finalize().into_bytes())[..24])))
    }

    /// The pseudonym of principal `id` (its own key), or of an unmapped
    /// external subject (the org's key).
    pub(crate) fn pseudo_of(&self, org: &str, principal: Option<&str>, sub: &str) -> Result<String> {
        let p = match principal {
            Some(id) => self.pseudo(id, id)?,
            None => self.pseudo(&format!("org:{org}"), sub)?,
        };
        Ok(p.unwrap_or_else(|| "p:erased".to_string()))
    }

    pub(crate) fn principal(&self, id: &str) -> Result<Option<Principal>> {
        Ok(self
            .tn_rows(
                &format!("?[org, kind, name, sponsor, template, status] := *{TN_PRINCIPAL}{{id: $id, org, kind, name, sponsor, template, status}}"),
                vec![("id", s(id))],
            )?
            .first()
            .map(|r| Principal {
                id: id.to_string(),
                org: cell_str(&r[0]).unwrap_or_default(),
                kind: cell_str(&r[1]).unwrap_or_default(),
                name: cell_str(&r[2]).unwrap_or_default(),
                sponsor: cell_str(&r[3]),
                template: cell_str(&r[4]),
                status: cell_str(&r[5]).unwrap_or_default(),
            }))
    }

    fn active_principal(&self, id: &str, org: &str) -> Result<Principal> {
        match self.principal(id)? {
            Some(p) if p.status == "active" && p.org == org => Ok(p),
            // an unknown, disabled, erased or foreign principal: one answer
            _ => bail!("principal '{id}' is not an active principal of org '{org}'"),
        }
    }

    /// `(org, partition)` of a workspace.
    pub(crate) fn workspace(&self, ws: &str) -> Result<(String, Option<String>)> {
        let rows = self.tn_rows(
            &format!("?[org, partition, status] := *{TN_WS}{{ws: $ws, org, partition, status}}"),
            vec![("ws", s(ws))],
        )?;
        match rows.first() {
            Some(r) if r[2].get_str() == Some("active") => Ok((cell_str(&r[0]).unwrap_or_default(), cell_str(&r[1]))),
            _ => bail!("workspace '{ws}' not found"),
        }
    }

    pub(crate) fn org_policy(&self, org: &str) -> Result<JsonValue> {
        Ok(self
            .tn_rows(&format!("?[p] := *{TN_ORG}{{org: $o, policy: p}}"), vec![("o", s(org))])?
            .first()
            .map(|r| cell_json(&r[0]))
            .unwrap_or(JsonValue::Null))
    }

    /// The view of a partition (`None`: the plain store).
    pub(crate) fn part_view(&self, part: Option<&str>) -> Result<Db<S>> {
        match part {
            Some(p) => self.root_view().partition_view(p),
            None => Ok(self.root_view()),
        }
    }

    /// Make sure a partition has a memory store, with the root store's
    /// embedder, decider and extractor.
    pub(crate) fn ensure_part_store(&self, part: Option<&str>) -> Result<()> {
        let Some(p) = part else { return Ok(()) };
        let pv = self.root_view().partition_view(p)?;
        if pv.memory_config()?.is_some() {
            return Ok(());
        }
        let cfg = self.root_view().memory_config()?.unwrap_or_default();
        pv.memory_init(crate::memory::InitOpts {
            embedder: cfg.embedder.filter(|_| cfg.dim.is_some()),
            decider: cfg.decider,
            extractor: cfg.extractor,
            ..Default::default()
        })?;
        Ok(())
    }

    /// Resolve a principal context into a session: every named principal
    /// must exist, be active and belong to the workspace's org; agents need
    /// a valid sponsor.
    pub(crate) fn tenant_session(&self, ctx: &PrincipalContext) -> Result<Session> {
        self.tenancy_ensure()?;
        let ws = ctx.workspace.clone().unwrap_or_else(|| DEFAULT.to_string());
        let (org, part) = self.workspace(&ws)?;
        if ctx.chain.len() > MAX_CHAIN {
            bail!("actor chain is deeper than {MAX_CHAIN} links");
        }
        let mut human = ctx.human.clone();
        let mut agent = ctx.agent.clone();
        let mut service = ctx.service.clone();
        // a chain without explicit principals: the originating human is its
        // innermost human link, the agent its outermost agent link
        for l in &ctx.chain {
            let Some(p) = &l.principal else { continue };
            match l.kind.as_str() {
                "human" | "user" if human.is_none() => human = Some(p.clone()),
                "ai_agent" | "agent" => agent = Some(p.clone()),
                "service" if service.is_none() => service = Some(p.clone()),
                _ => {}
            }
        }
        let load = |id: &Option<String>, kind: &str| -> Result<Option<Principal>> {
            match id {
                None => Ok(None),
                Some(id) => {
                    let p = self.active_principal(id, &org)?;
                    if p.kind != kind {
                        bail!("principal '{id}' is a {}, not a {kind}", p.kind);
                    }
                    Ok(Some(p))
                }
            }
        };
        let human = load(&human, "human")?;
        let agent = load(&agent, "agent")?;
        let service = load(&service, "service")?;
        if human.is_none() && agent.is_none() && service.is_none() {
            bail!("a principal context needs a human, an agent or a service");
        }
        let mut participants = vec![];
        for id in &ctx.participants {
            let p = self.active_principal(id, &org)?;
            if Some(&p) != human.as_ref() && Some(&p) != agent.as_ref() {
                participants.push(p);
            }
        }
        let mut sponsor = None;
        if let Some(a) = &agent {
            let sp = a.sponsor.clone().ok_or_else(|| miette::miette!("agent '{}' has no sponsor", a.id))?;
            let sp = self.active_principal(&sp, &org).map_err(|_| miette::miette!("agent '{}' has no active sponsor", a.id))?;
            if let Some(t) = &a.template {
                self.active_principal(t, &org)?;
            }
            sponsor = Some(sp);
        }
        let ceiling = match (&human, &sponsor) {
            (None, Some(sp)) if self.org_policy(&org)?.get("sponsor_ceiling").and_then(|x| x.as_bool()) == Some(true) => Some(sp.clone()),
            _ => None,
        };
        let accountable = human.as_ref().map(|h| h.id.clone()).or_else(|| sponsor.as_ref().map(|s| s.id.clone())).or_else(|| service.as_ref().map(|s| s.id.clone()));
        // the chain: as given, else derived from the principals
        let default_as = ctx.assurance.unwrap_or_default();
        let chain = if ctx.chain.is_empty() {
            let mut c = vec![];
            for (p, k) in [(&human, "human"), (&service, "service"), (&agent, "ai_agent")] {
                if let Some(p) = p {
                    c.push(ActorLink { kind: k.to_string(), iss: None, sub: p.id.clone(), principal: Some(p.id.clone()), assurance: default_as });
                }
            }
            c
        } else {
            ctx.chain.clone()
        };
        let human_proven = match &human {
            None => false,
            Some(h) => chain
                .iter()
                .find(|l| l.principal.as_deref() == Some(&h.id))
                .map_or(default_as.proven(), |l| l.assurance.proven()),
        };
        Ok(Session { ws, org, part, human, agent, service, participants, ceiling, accountable, chain, human_proven, ctx: ctx.clone() })
    }

    // -------------------------------------------------------------- audit

    /// The session's chain with pseudonymous ids (for audit events).
    pub(crate) fn audit_chain(&self, ss: &Session) -> Result<JsonValue> {
        let mut out = vec![];
        for (i, l) in ss.chain.iter().enumerate() {
            out.push(json!({
                "position": i,
                "kind": l.kind,
                "iss": l.iss,
                "sub": self.pseudo_of(&ss.org, l.principal.as_deref(), &l.sub)?,
                "assurance": l.assurance,
            }));
        }
        Ok(JsonValue::Array(out))
    }

    fn otel_of(&self, ss: &Session, op: &str, scope_set: Option<&str>) -> Result<JsonValue> {
        let agent = match &ss.agent {
            Some(a) => Some(self.pseudo_of(&ss.org, Some(&a.id), &a.id)?),
            None => None,
        };
        Ok(json!({
            "gen_ai.operation.name": op,
            "gen_ai.agent.id": agent,
            "gen_ai.conversation.id": ss.ctx.run,
            "kleos.workspace": ss.ws,
            "kleos.scope_set_hash": scope_set,
        }))
    }

    /// The write audit of a principal view: every transaction that writes
    /// appends its event inside itself, before it commits (no crash gap).
    pub(crate) fn write_sink(&self, ss: &Session) -> Result<Arc<dyn CommitAudit>> {
        let accountable = match &ss.accountable {
            Some(a) => Some(self.pseudo_of(&ss.org, Some(a), a)?),
            None => None,
        };
        Ok(Arc::new(WriteAudit {
            state: self.tenancy.clone(),
            ws: ss.ws.clone(),
            chain: self.audit_chain(ss)?,
            accountable,
            otel: self.otel_of(ss, "write", None)?,
        }))
    }

    /// Append an op-level event for `ss`.
    pub(crate) fn audit_op(
        &self,
        ss: &Session,
        op: &str,
        targets: JsonValue,
        scope_set: Option<&str>,
        decision: JsonValue,
    ) -> Result<()> {
        let accountable = match &ss.accountable {
            Some(a) => Some(self.pseudo_of(&ss.org, Some(a), a)?),
            None => None,
        };
        let chain = self.audit_chain(ss)?;
        let otel = self.otel_of(ss, op, scope_set)?;
        crate::tenancy::otel::span(op, &otel, &targets, &decision);
        self.audit_append(&ss.ws, op, chain, accountable, targets, vec![], scope_set, decision, otel, now_micros())
    }

    /// Append one event to the workspace's chain, in a transaction of its
    /// own. `hash = SHA-256(prev ‖ canonical event)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn audit_append(
        &self,
        ws: &str,
        op: &str,
        chain: JsonValue,
        accountable: Option<String>,
        targets: JsonValue,
        relations: Vec<String>,
        scope_set: Option<&str>,
        decision: JsonValue,
        otel: JsonValue,
        ts: i64,
    ) -> Result<()> {
        let ev = AuditEvent { op: op.to_string(), chain, accountable, targets, relations, scope_set: scope_set.map(|x| x.to_string()), decision, otel, ts };
        let root = self.root_view();
        let _g = self.tenancy.audit_lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut tx = root.transact_write()?;
        append_in_tx(&mut tx, ws, &ev)?;
        tx.commit_tx()
    }

    /// Kept for callers of the earlier queued design: events are now
    /// written with their transactions, so there is nothing to flush.
    pub fn audit_flush(&self) -> Result<()> {
        Ok(())
    }

    fn audit_events(&self, ws: &str, from: i64, limit: usize) -> Result<Vec<JsonValue>> {
        self.audit_flush()?;
        Ok(self
            .tn_rows(
                &format!(
                    "?[seq, ts, op, chain, accountable, targets, relations, scope_set, decision, otel, prev, hash] := \
                       *{TN_AUDIT}{{ws: $ws, seq, ts, op, chain, accountable, targets, relations, scope_set, decision, otel, prev, hash}}, seq >= $from \
                     :order seq :limit {limit}"
                ),
                vec![("ws", s(ws)), ("from", DataValue::from(from))],
            )?
            .into_iter()
            .map(|r| {
                json!({
                    "ws": ws, "seq": r[0].get_int(), "ts": r[1].get_int(), "op": cell_str(&r[2]), "chain": cell_json(&r[3]),
                    "accountable": cell_str(&r[4]), "targets": cell_json(&r[5]), "relations": JsonValue::from(r[6].clone()),
                    "scope_set": cell_str(&r[7]), "decision": cell_json(&r[8]), "otel": cell_json(&r[9]),
                    "prev": cell_str(&r[10]), "hash": cell_str(&r[11]),
                })
            })
            .collect())
    }

    /// Recompute the workspace's chain: `{ok, events, broken_at}`.
    pub fn audit_verify(&self, ws: &str) -> Result<JsonValue> {
        let evs = self.audit_events(ws, 0, usize::MAX >> 1)?;
        let mut prev = String::new();
        let mut expect = 1i64;
        for mut e in evs.iter().cloned() {
            let seq = e["seq"].as_i64().unwrap_or(-1);
            let hash = e["hash"].as_str().unwrap_or_default().to_string();
            if let Some(o) = e.as_object_mut() {
                o.remove("hash");
            }
            if seq != expect || e["prev"].as_str() != Some(prev.as_str()) || event_hash(&prev, &e) != hash {
                return Ok(json!({"ok": false, "events": evs.len(), "broken_at": seq}));
            }
            prev = hash;
            expect += 1;
        }
        Ok(json!({"ok": true, "events": evs.len(), "broken_at": null}))
    }

    // -------------------------------------------------------------- admin ops

    /// Admin ops of the control plane (`::tenant <op> {…}`); embedded
    /// hosts are trusted. See docs/specs/tenancy.md, "Operations".
    pub fn tenant_call(&self, op: &str, args: JsonValue) -> Result<JsonValue> {
        let a = match args {
            JsonValue::Null => serde_json::Map::new(),
            JsonValue::Object(m) => m,
            other => bail!("tenant {op}: options must be an object, got {other}"),
        };
        let gs = |k: &str| -> Option<String> { a.get(k).and_then(|v| v.as_str()).map(|x| x.to_string()) };
        let need = |k: &str| -> Result<String> { gs(k).ok_or_else(|| miette::miette!("tenant {op} needs '{k}'")) };
        let strs = |k: &str| -> Vec<String> {
            a.get(k).and_then(|v| v.as_array()).map(|v| v.iter().filter_map(|x| x.as_str().map(|y| y.to_string())).collect()).unwrap_or_default()
        };
        self.tenancy_ensure()?;
        let now = DataValue::from(now_micros());
        let out = match op {
            "enable" => json!({"enabled": true}),
            "org_create" => {
                let org = need("org")?;
                check_id("org", &org)?;
                if self.org_exists(&org)? {
                    bail!("org '{org}' already exists");
                }
                let policy = json!({"sponsor_ceiling": a.get("sponsor_ceiling").and_then(|x| x.as_bool()).unwrap_or(false)});
                self.tn_q(
                    &format!("?[org, name, policy, created, status] <- [[$o, $n, $p, $now, 'active']] :put {TN_ORG} {{org => name, policy, created, status}}"),
                    vec![("o", s(&org)), ("n", s(&gs("name").unwrap_or_else(|| org.clone()))), ("p", json_dv(policy)), ("now", now.clone())],
                )?;
                self.new_key(&format!("org:{org}"))?;
                if let Some(owner) = gs("owner") {
                    if self.principal(&owner)?.is_none() {
                        self.tenant_call("principal_create", json!({"id": owner, "org": org, "kind": "human"}))?;
                    }
                    self.tenant_call("member_add", json!({"group": format!("org:{org}"), "principal": owner, "role": "owner"}))?;
                }
                self.bump_epoch()?;
                json!({"org": org})
            }
            "ws_create" => {
                let ws = need("ws")?;
                let org = need("org")?;
                check_id("workspace", &ws)?;
                if !self.org_exists(&org)? {
                    bail!("org '{org}' not found");
                }
                if !self.tn_rows(&format!("?[o] := *{TN_WS}{{ws: $ws, org: o}}"), vec![("ws", s(&ws))])?.is_empty() {
                    bail!("workspace '{ws}' already exists");
                }
                let part = format!("ws_{ws}");
                self.tn_q(
                    &format!(
                        "?[ws, org, name, partition, created, status] <- [[$ws, $o, $n, $p, $now, 'active']] \
                         :put {TN_WS} {{ws => org, name, partition, created, status}}"
                    ),
                    vec![("ws", s(&ws)), ("o", s(&org)), ("n", s(&gs("name").unwrap_or_else(|| ws.clone()))), ("p", s(&part)), ("now", now.clone())],
                )?;
                self.ensure_part_store(Some(&part))?;
                self.bump_epoch()?;
                json!({"ws": ws, "org": org, "partition": part})
            }
            "team_create" => {
                let team = need("team")?;
                let org = need("org")?;
                check_id("team", &team)?;
                if !self.org_exists(&org)? {
                    bail!("org '{org}' not found");
                }
                let parent = gs("parent");
                if let Some(p) = &parent {
                    let depth = self.team_ancestors(p, &org)?.len() + 1;
                    if depth >= MAX_TEAM_DEPTH {
                        bail!("team tree deeper than {MAX_TEAM_DEPTH}");
                    }
                }
                self.tn_q(
                    &format!("?[team, org, name, parent, created] <- [[$t, $o, $n, $p, $now]] :put {TN_TEAM} {{team => org, name, parent, created}}"),
                    vec![("t", s(&team)), ("o", s(&org)), ("n", s(&gs("name").unwrap_or_else(|| team.clone()))), ("p", opt_s(&parent)), ("now", now.clone())],
                )?;
                self.bump_epoch()?;
                json!({"team": team, "org": org, "parent": parent})
            }
            "principal_create" => {
                let id = need("id")?;
                let org = need("org")?;
                let kind = need("kind")?;
                check_id("principal", &id)?;
                if !KINDS.contains(&kind.as_str()) {
                    bail!("principal kind must be human, agent or service, got '{kind}'");
                }
                if !self.org_exists(&org)? {
                    bail!("org '{org}' not found");
                }
                if self.principal(&id)?.is_some() {
                    bail!("principal '{id}' already exists");
                }
                let sponsor = gs("sponsor");
                let template = gs("template");
                if kind == "agent" {
                    let sp = sponsor.clone().ok_or_else(|| miette::miette!("an agent needs a sponsor (an accountable human)"))?;
                    let p = self.active_principal(&sp, &org)?;
                    if p.kind != "human" {
                        bail!("sponsor '{sp}' must be a human");
                    }
                    if let Some(t) = &template {
                        if self.active_principal(t, &org)?.kind != "agent" {
                            bail!("template '{t}' must be an agent");
                        }
                    }
                } else if template.is_some() {
                    bail!("only agents have a template");
                }
                self.tn_q(
                    &format!(
                        "?[id, org, kind, name, sponsor, template, status, created] <- [[$id, $o, $k, $n, $sp, $t, 'active', $now]] \
                         :put {TN_PRINCIPAL} {{id => org, kind, name, sponsor, template, status, created}}"
                    ),
                    vec![
                        ("id", s(&id)),
                        ("o", s(&org)),
                        ("k", s(&kind)),
                        ("n", s(&gs("name").unwrap_or_else(|| id.clone()))),
                        ("sp", opt_s(&sponsor)),
                        ("t", opt_s(&template)),
                        ("now", now.clone()),
                    ],
                )?;
                if let Some(idents) = a.get("idents").and_then(|v| v.as_array()) {
                    for pair in idents {
                        let (Some(iss), Some(sub)) = (pair.get(0).and_then(|x| x.as_str()), pair.get(1).and_then(|x| x.as_str())) else {
                            bail!("idents must be [[iss, sub], …]");
                        };
                        self.tn_q(
                            &format!("?[iss, sub, principal] <- [[$i, $s, $p]] :put {TN_IDENT} {{iss, sub => principal}}"),
                            vec![("i", s(iss)), ("s", s(sub)), ("p", s(&id))],
                        )?;
                    }
                }
                self.new_key(&id)?;
                self.member_put(&format!("org:{org}"), &id, "member")?;
                self.bump_epoch()?;
                json!({"id": id, "org": org, "kind": kind, "sponsor": sponsor, "template": template})
            }
            "principal_disable" => {
                let id = need("id")?;
                self.tn_q(
                    &format!("?[id, status] := *{TN_PRINCIPAL}{{id}}, id = $id, status = 'disabled' :update {TN_PRINCIPAL} {{id => status}}"),
                    vec![("id", s(&id))],
                )?;
                self.bump_epoch()?;
                json!({"id": id, "status": "disabled"})
            }
            "member_add" | "member_remove" => {
                let group = need("group")?;
                let principal = need("principal")?;
                let (org, role) = self.check_group(&group, &principal, gs("role"))?;
                let _ = org;
                if op == "member_add" {
                    self.member_put(&group, &principal, &role)?;
                } else {
                    self.tn_q(
                        &format!("?[grp, principal] <- [[$g, $p]] :rm {TN_MEMBER} {{grp, principal}}"),
                        vec![("g", s(&group)), ("p", s(&principal))],
                    )?;
                }
                self.bump_epoch()?;
                json!({"group": group, "principal": principal, "role": role})
            }
            "grant" | "revoke" => {
                let ws = need("ws")?;
                let (org, _) = self.workspace(&ws)?;
                let grantee = match (gs("principal"), gs("team")) {
                    (Some(p), None) => {
                        self.active_principal(&p, &org)?;
                        format!("p:{p}")
                    }
                    (None, Some(t)) => {
                        if self.team_org(&t)?.as_deref() != Some(org.as_str()) {
                            bail!("team '{t}' is not a team of org '{org}'");
                        }
                        format!("t:{t}")
                    }
                    _ => bail!("tenant {op} needs exactly one of 'principal' or 'team'"),
                };
                if op == "grant" {
                    let role = need("role")?;
                    if !WS_ROLES.contains(&role.as_str()) {
                        bail!("workspace role must be reader, writer or admin, got '{role}'");
                    }
                    let actions = strs("actions");
                    if let Some(bad) = actions.iter().find(|x| x.as_str() != "promote") {
                        bail!("grant actions may only add 'promote', got '{bad}'");
                    }
                    self.tn_q(
                        &format!(
                            "?[ws, grantee, role, actions, created] <- [[$ws, $g, $r, $acts, $now]] :put {TN_GRANT} {{ws, grantee => role, actions, created}}"
                        ),
                        vec![("ws", s(&ws)), ("g", s(&grantee)), ("r", s(&role)), ("acts", str_list(&actions)), ("now", now.clone())],
                    )?;
                } else {
                    self.tn_q(&format!("?[ws, grantee] <- [[$ws, $g]] :rm {TN_GRANT} {{ws, grantee}}"), vec![("ws", s(&ws)), ("g", s(&grantee))])?;
                }
                self.bump_epoch()?;
                json!({"ws": ws, "grantee": grantee, "role": gs("role")})
            }
            "policy_set" => {
                let org = need("org")?;
                if !self.org_exists(&org)? {
                    bail!("org '{org}' not found");
                }
                let mut pol = self.org_policy(&org)?;
                if !pol.is_object() {
                    pol = json!({});
                }
                if let Some(c) = a.get("cedar") {
                    let text = c.as_str().unwrap_or_default().to_string();
                    authz::parse_policies(&text)?;
                    pol["cedar"] = json!(text);
                }
                if let Some(b) = a.get("sponsor_ceiling").and_then(|x| x.as_bool()) {
                    pol["sponsor_ceiling"] = json!(b);
                }
                self.tn_q(
                    &format!("?[org, policy] := *{TN_ORG}{{org}}, org = $o, policy = $p :update {TN_ORG} {{org => policy}}"),
                    vec![("o", s(&org)), ("p", json_dv(pol.clone()))],
                )?;
                self.tenancy.policies.lock().unwrap_or_else(|e| e.into_inner()).remove(&org);
                self.bump_epoch()?;
                json!({"org": org, "policy": pol})
            }
            "apikey_create" => {
                let principal = need("principal")?;
                if self.principal(&principal)?.map(|p| p.status) != Some("active".to_string()) {
                    bail!("principal '{principal}' is not active");
                }
                let raw: [u8; 24] = rand::random();
                let key = format!("kl_{}", hex(&raw));
                let actions = strs("actions");
                self.tn_q(
                    &format!(
                        "?[hash, principal, actions, label, created, expires] <- [[$h, $p, $acts, $l, $now, $exp]] \
                         :put {TN_APIKEY} {{hash => principal, actions, label, created, expires}}"
                    ),
                    vec![
                        ("h", s(&hex(&Sha256::digest(key.as_bytes())))),
                        ("p", s(&principal)),
                        ("acts", str_list(&actions)),
                        ("l", opt_s(&gs("label"))),
                        ("now", now.clone()),
                        ("exp", a.get("expires").and_then(|x| x.as_i64()).map(DataValue::from).unwrap_or(DataValue::Null)),
                    ],
                )?;
                json!({"key": key, "principal": principal})
            }
            "apikey_revoke" => {
                let h = match (gs("key"), gs("hash")) {
                    (Some(k), _) => hex(&Sha256::digest(k.as_bytes())),
                    (None, Some(h)) => h,
                    _ => bail!("tenant apikey_revoke needs 'key' or 'hash'"),
                };
                self.tn_q(&format!("?[hash] <- [[$h]] :rm {TN_APIKEY} {{hash}}"), vec![("h", s(&h))])?;
                json!({"revoked": h})
            }
            "principal_key_set" => {
                // the public keys an agent signs its client assertions with
                let id = need("id")?;
                if self.principal(&id)?.map(|p| p.status) != Some("active".to_string()) {
                    bail!("principal '{id}' is not active");
                }
                let jwks = a.get("jwks").cloned().filter(|j| j.get("keys").is_some_and(|k| k.is_array())).ok_or_else(|| miette::miette!("principal_key_set needs 'jwks' ({{keys: [...]}})"))?;
                self.tn_q(
                    &format!("?[principal, jwks, updated] <- [[$p, $j, $now]] :put {TN_JWKS} {{principal => jwks, updated}}"),
                    vec![("p", s(&id)), ("j", json_dv(jwks)), ("now", now.clone())],
                )?;
                json!({"id": id})
            }
            "erase" => self.tenant_erase(&need("principal")?, gs("reassign_to"))?,
            "audit" => {
                let ws = gs("ws").unwrap_or_else(|| DEFAULT.to_string());
                let from = a.get("from_seq").and_then(|x| x.as_i64()).unwrap_or(0);
                let limit = a.get("limit").and_then(|x| x.as_u64()).unwrap_or(1000) as usize;
                let mut evs = self.audit_events(&ws, from, limit)?;
                if let Some(p) = gs("principal") {
                    let (org, _) = self.workspace(&ws)?;
                    let ps = self.pseudo_of(&org, Some(&p), &p)?;
                    evs.retain(|e| {
                        e["accountable"].as_str() == Some(ps.as_str())
                            || e["chain"].as_array().is_some_and(|c| c.iter().any(|l| l["sub"].as_str() == Some(ps.as_str())))
                    });
                }
                JsonValue::Array(evs)
            }
            "audit_verify" => self.audit_verify(&gs("ws").unwrap_or_else(|| DEFAULT.to_string()))?,
            "scopes" => {
                let ctx: PrincipalContext = serde_json::from_value(a.get("as").cloned().unwrap_or(JsonValue::Null))
                    .map_err(|e| miette::miette!("tenant scopes: bad 'as': {e}"))?;
                let action = gs("action").unwrap_or_else(|| "read".to_string());
                let ss = self.tenant_session(&ctx)?;
                let (scopes, org) = self.scope_set(&ss, &action)?;
                json!({"workspace": ss.ws, "scopes": scopes.iter().map(|k| k.ns()).collect::<Vec<_>>(), "org_scope": org})
            }
            "approve" | "reject" => {
                let approver = need("as")?;
                let (key, rel, id) = match (gs("promotion"), gs("request")) {
                    (Some(p), _) => ("promotion", TN_PROMOTION, p),
                    (None, Some(r)) => ("request", TN_REQUEST, r),
                    _ => bail!("tenant {op} needs 'promotion' or 'request'"),
                };
                let ws = self
                    .tn_rows(&format!("?[ws] := *{rel}{{id: $id, ws}}"), vec![("id", s(&id))])?
                    .first()
                    .and_then(|r| cell_str(&r[0]))
                    .ok_or_else(|| miette::miette!("{key} '{id}' not found"))?;
                let ctx = PrincipalContext { workspace: Some(ws), human: Some(approver), ..Default::default() };
                self.memory_call_as(&ctx, op, json!({key: id}))?
            }
            "get" | "list" => self.tenant_list(&need("kind")?)?,
            other => bail!(
                "unknown tenant op '{other}': expected enable, org_create, ws_create, team_create, principal_create, principal_disable, \
                 member_add, member_remove, grant, revoke, policy_set, apikey_create, apikey_revoke, erase, audit, audit_verify, \
                 scopes, approve, reject or list"
            ),
        };
        Ok(out)
    }

    fn org_exists(&self, org: &str) -> Result<bool> {
        Ok(!self.tn_rows(&format!("?[n] := *{TN_ORG}{{org: $o, name: n}}"), vec![("o", s(org))])?.is_empty())
    }

    fn team_org(&self, team: &str) -> Result<Option<String>> {
        Ok(self
            .tn_rows(&format!("?[o] := *{TN_TEAM}{{team: $t, org: o}}"), vec![("t", s(team))])?
            .first()
            .and_then(|r| cell_str(&r[0])))
    }

    /// The parents of `team` up to its root, nearest first.
    pub(crate) fn team_ancestors(&self, team: &str, org: &str) -> Result<Vec<String>> {
        let mut out = vec![];
        let mut cur = team.to_string();
        loop {
            let rows = self.tn_rows(&format!("?[o, p] := *{TN_TEAM}{{team: $t, org: o, parent: p}}"), vec![("t", s(&cur))])?;
            let Some(r) = rows.first() else { bail!("team '{cur}' not found") };
            if r[0].get_str() != Some(org) {
                bail!("team '{cur}' is not a team of org '{org}'");
            }
            match cell_str(&r[1]) {
                Some(p) if !out.contains(&p) && out.len() <= MAX_TEAM_DEPTH => {
                    out.push(p.clone());
                    cur = p;
                }
                _ => break,
            }
        }
        Ok(out)
    }

    fn member_put(&self, group: &str, principal: &str, role: &str) -> Result<()> {
        self.tn_q(
            &format!("?[grp, principal, role, created] <- [[$g, $p, $r, $now]] :put {TN_MEMBER} {{grp, principal => role, created}}"),
            vec![("g", s(group)), ("p", s(principal)), ("r", s(role)), ("now", DataValue::from(now_micros()))],
        )?;
        Ok(())
    }

    /// Validate a membership: `(org, role)`.
    fn check_group(&self, group: &str, principal: &str, role: Option<String>) -> Result<(String, String)> {
        let (org, role) = match group.split_once(':') {
            Some(("org", o)) => {
                let role = role.unwrap_or_else(|| "member".to_string());
                if !ORG_ROLES.contains(&role.as_str()) {
                    bail!("org role must be member, admin or owner, got '{role}'");
                }
                (o.to_string(), role)
            }
            Some(("team", t)) => {
                let o = self.team_org(t)?.ok_or_else(|| miette::miette!("team '{t}' not found"))?;
                (o, role.unwrap_or_else(|| "member".to_string()))
            }
            _ => bail!("group must be 'org:<org>' or 'team:<team>', got '{group}'"),
        };
        self.active_principal(principal, &org)?;
        Ok((org, role))
    }

    fn tenant_list(&self, kind: &str) -> Result<JsonValue> {
        let (rel, cols): (&str, &[&str]) = match kind {
            "org" => (TN_ORG, &["org", "name", "policy", "status"]),
            "ws" | "workspace" => (TN_WS, &["ws", "org", "name", "partition", "status"]),
            "team" => (TN_TEAM, &["team", "org", "name", "parent"]),
            "principal" => (TN_PRINCIPAL, &["id", "org", "kind", "name", "sponsor", "template", "status"]),
            "member" => (TN_MEMBER, &["grp", "principal", "role"]),
            "grant" => (TN_GRANT, &["ws", "grantee", "role", "actions"]),
            "promotion" => (TN_PROMOTION, &["id", "ws", "from_ns", "ids", "target", "requested_by", "status", "approved_by", "result", "at"]),
            "request" => (TN_REQUEST, &["id", "ws", "op", "ns", "ids", "requested_by", "status", "approved_by", "result", "at"]),
            other => bail!("tenant list: unknown kind '{other}'"),
        };
        let c = cols.join(", ");
        let rows = self.tn_rows(&format!("?[{c}] := *{rel}{{{c}}}"), vec![])?;
        Ok(JsonValue::Array(
            rows.into_iter()
                .map(|r| {
                    let mut o = serde_json::Map::new();
                    for (k, v) in cols.iter().zip(r.iter()) {
                        o.insert(k.to_string(), cell_json(v));
                    }
                    JsonValue::Object(o)
                })
                .collect(),
        ))
    }

    /// Every partition of the store: workspace partitions and org
    /// partitions (empty without tenancy). Background runners serve each.
    pub fn tenancy_partitions(&self) -> Result<Vec<String>> {
        if !self.tenancy_enabled() {
            return Ok(vec![]);
        }
        let mut out: Vec<String> = self
            .tn_rows(&format!("?[p] := *{TN_WS}{{partition: p, status: 'active'}}, !is_null(p)"), vec![])?
            .into_iter()
            .filter_map(|r| cell_str(&r[0]))
            .collect();
        out.extend(self.tn_rows(&format!("?[o] := *{TN_ORG}{{org: o}}"), vec![])?.into_iter().filter_map(|r| cell_str(&r[0])).map(|o| format!("org_{o}")));
        Ok(out)
    }

    /// The partitions a principal call may write to: its workspace's and its
    /// org's.
    pub fn principal_partitions(&self, ctx: &PrincipalContext) -> Result<Vec<String>> {
        let ws = ctx.workspace.clone().unwrap_or_else(|| DEFAULT.to_string());
        let (org, part) = self.workspace(&ws)?;
        Ok(part.into_iter().chain(std::iter::once(format!("org_{org}"))).collect())
    }

    /// The principal an API key belongs to, with the key's actions.
    pub fn resolve_apikey(&self, key: &str) -> Result<Option<(String, Vec<String>)>> {
        if !self.tenancy_enabled() {
            return Ok(None);
        }
        let rows = self.tn_rows(
            &format!("?[p, acts, exp] := *{TN_APIKEY}{{hash: $h, principal: p, actions: acts, expires: exp}}"),
            vec![("h", s(&hex(&Sha256::digest(key.as_bytes()))))],
        )?;
        let Some(r) = rows.first() else { return Ok(None) };
        if r[2].get_int().is_some_and(|e| e <= now_micros()) {
            return Ok(None);
        }
        let acts = match &r[1] {
            DataValue::List(l) => l.iter().filter_map(|x| x.get_str().map(|y| y.to_string())).collect(),
            _ => vec![],
        };
        Ok(cell_str(&r[0]).map(|p| (p, acts)))
    }

    /// The registered public keys (a JWKS) of a principal.
    pub fn principal_jwks(&self, id: &str) -> Result<Option<JsonValue>> {
        if !self.tenancy_enabled() {
            return Ok(None);
        }
        self.tenancy_ensure()?;
        Ok(self.tn_rows(&format!("?[j] := *{TN_JWKS}{{principal: $p, jwks: j}}"), vec![("p", s(id))])?.first().map(|r| cell_json(&r[0])))
    }

    /// The principal an external identity maps to.
    pub fn resolve_ident(&self, iss: &str, sub: &str) -> Result<Option<String>> {
        if !self.tenancy_enabled() {
            return Ok(None);
        }
        Ok(self
            .tn_rows(&format!("?[p] := *{TN_IDENT}{{iss: $i, sub: $s, principal: p}}"), vec![("i", s(iss)), ("s", s(sub))])?
            .first()
            .and_then(|r| cell_str(&r[0])))
    }

    /// The kind of a principal (`human`, `agent`, `service`), if active.
    pub fn principal_kind(&self, id: &str) -> Result<Option<String>> {
        if !self.tenancy_enabled() {
            return Ok(None);
        }
        Ok(self.principal(id)?.filter(|p| p.status == "active").map(|p| p.kind))
    }

    /// The org an admin op touches (from its `org`, `ws`, `team`, `group`,
    /// `principal` / `id` or `promotion` options); every named object must
    /// belong to the same org. `None`: the op names none.
    pub fn tenant_target_org(&self, args: &JsonValue) -> Result<Option<String>> {
        self.tenancy_ensure()?;
        let g = |k: &str| args.get(k).and_then(|v| v.as_str()).map(|x| x.to_string());
        let mut orgs: BTreeSet<String> = BTreeSet::new();
        orgs.extend(g("org"));
        if let Some(ws) = g("ws") {
            orgs.insert(self.workspace(&ws)?.0);
        }
        if let Some(t) = g("team") {
            orgs.insert(self.team_org(&t)?.ok_or_else(|| miette::miette!("team '{t}' not found"))?);
        }
        if let Some(gr) = g("group") {
            match gr.split_once(':') {
                Some(("org", o)) => {
                    orgs.insert(o.to_string());
                }
                Some(("team", t)) => {
                    orgs.insert(self.team_org(t)?.ok_or_else(|| miette::miette!("team '{t}' not found"))?);
                }
                _ => bail!("bad group '{gr}'"),
            }
        }
        for k in ["principal", "id", "sponsor", "reassign_to"] {
            if let Some(p) = g(k) {
                if let Some(pr) = self.principal(&p)? {
                    orgs.insert(pr.org);
                }
            }
        }
        if let Some(pid) = g("promotion") {
            if let Some(ws) = self.tn_rows(&format!("?[ws] := *{TN_PROMOTION}{{id: $id, ws}}"), vec![("id", s(&pid))])?.first().and_then(|r| cell_str(&r[0])) {
                orgs.insert(self.workspace(&ws)?.0);
            }
        }
        if orgs.len() > 1 {
            bail!("the op names objects of several orgs: {orgs:?}");
        }
        Ok(orgs.into_iter().next())
    }

    /// Whether `principal` is an admin or owner of `org`.
    pub fn is_org_admin(&self, principal: &str, org: &str) -> Result<bool> {
        Ok(self
            .tn_rows(&format!("?[r] := *{TN_MEMBER}{{grp: $g, principal: $p, role: r}}"), vec![("g", s(&format!("org:{org}"))), ("p", s(principal))])?
            .first()
            .and_then(|r| r[0].get_str().map(|x| x == "admin" || x == "owner"))
            .unwrap_or(false))
    }

    // -------------------------------------------------------------- erasure

    /// Erase a principal (docs/specs/tenancy.md, "Erasure"): its own scopes
    /// and what it authored elsewhere, through the provenance cascade of
    /// `forget {mode: 'erase'}`, then its memberships and its pseudonym key.
    pub(crate) fn tenant_erase(&self, id: &str, reassign_to: Option<String>) -> Result<JsonValue> {
        let p = self.principal(id)?.ok_or_else(|| miette::miette!("principal '{id}' not found"))?;
        if p.status == "erased" {
            bail!("principal '{id}' is already erased");
        }
        let sponsored: Vec<String> = self
            .tn_rows(&format!("?[a] := *{TN_PRINCIPAL}{{id: a, sponsor: $p, status}}, status != 'erased'"), vec![("p", s(id))])?
            .into_iter()
            .filter_map(|r| cell_str(&r[0]))
            .collect();
        if !sponsored.is_empty() {
            let Some(to) = reassign_to else {
                bail!("principal '{id}' sponsors agents {sponsored:?}: pass reassign_to (a human of the org)");
            };
            let np = self.active_principal(&to, &p.org)?;
            if np.kind != "human" {
                bail!("reassign_to '{to}' must be a human");
            }
            self.tn_q(
                &format!("?[id, sponsor] := *{TN_PRINCIPAL}{{id, sponsor: $old}}, sponsor = $new :update {TN_PRINCIPAL} {{id => sponsor}}"),
                vec![("old", s(id)), ("new", s(&to))],
            )?;
        }
        // every partition of the org: its workspaces and the org partition
        let mut parts: Vec<Option<String>> = self
            .tn_rows(&format!("?[p] := *{TN_WS}{{org: $o, partition: p}}"), vec![("o", s(&p.org))])?
            .into_iter()
            .map(|r| cell_str(&r[0]))
            .collect();
        parts.push(Some(format!("org_{}", p.org)));
        let mut erased = 0usize;
        let mut scopes = 0usize;
        for part in &parts {
            let pv = self.part_view(part.as_deref())?;
            if pv.memory_config()?.is_none() {
                continue;
            }
            let org_part = part.as_deref().is_some_and(|x| x.starts_with("org_"));
            let nss: Vec<String> = pv
                .run_script(&format!("?[ns] := *{}{{ns}}", crate::memory::MEM_NS), Default::default(), ScriptMutability::Immutable)?
                .rows
                .into_iter()
                .filter_map(|r| cell_str(&r[0]))
                .collect();
            for ns in &nss {
                if ScopeKey::parse(ns, org_part).owners().contains(&id) {
                    scopes += 1;
                    erased += pv.forget(crate::memory::ForgetOpts { ns: Some(ns.clone()), all: true, mode: Some("erase".into()), ..Default::default() })?.ids.len();
                }
            }
            // what it authored in other scopes
            let mut by_ns: BTreeMap<String, Vec<String>> = BTreeMap::new();
            if let Ok(r) = pv.run_script(
                &format!(
                    "?[ns, id] := *{}{{ns, id, human, agent}}, human == $p || agent == $p",
                    crate::memory::MEM_ACTOR
                ),
                params(vec![("p", s(id))]),
                ScriptMutability::Immutable,
            ) {
                for row in r.rows {
                    if let (Some(ns), Some(mid)) = (cell_str(&row[0]), cell_str(&row[1])) {
                        by_ns.entry(ns).or_default().push(mid);
                    }
                }
            }
            for (ns, ids) in by_ns {
                erased += pv.forget(crate::memory::ForgetOpts { ns: Some(ns), ids, mode: Some("erase".into()), ..Default::default() })?.ids.len();
            }
        }
        // caches: cached scope sets, cached embeddings of erased text
        crate::embed::clear_embedding_cache();
        // the control plane: memberships, grants, identities, keys
        let grantee = format!("p:{id}");
        self.tn_q(
            &format!(
                "{{?[grp, principal] := *{TN_MEMBER}{{grp, principal}}, principal = $p :rm {TN_MEMBER} {{grp, principal}}}}\
                 {{?[ws, grantee] := *{TN_GRANT}{{ws, grantee}}, grantee = $g :rm {TN_GRANT} {{ws, grantee}}}}\
                 {{?[iss, sub] := *{TN_IDENT}{{iss, sub, principal}}, principal = $p :rm {TN_IDENT} {{iss, sub}}}}\
                 {{?[hash] := *{TN_APIKEY}{{hash, principal}}, principal = $p :rm {TN_APIKEY} {{hash}}}}\
                 {{?[principal] <- [[$p]] :rm {TN_KEY} {{principal}}}}\
                 {{?[principal] <- [[$p]] :rm {TN_JWKS} {{principal}}}}\
                 {{?[id, status, name] := *{TN_PRINCIPAL}{{id}}, id = $p, status = 'erased', name = 'erased' :update {TN_PRINCIPAL} {{id => status, name}}}}"
            ),
            vec![("p", s(id)), ("g", s(&grantee))],
        )?;
        self.tenancy.keys.lock().unwrap_or_else(|e| e.into_inner()).remove(id);
        self.bump_epoch()?;
        let ws_list: Vec<String> = self
            .tn_rows(&format!("?[ws] := *{TN_WS}{{ws, org: $o}}"), vec![("o", s(&p.org))])?
            .into_iter()
            .filter_map(|r| cell_str(&r[0]))
            .collect();
        for ws in &ws_list {
            self.audit_append(ws, "erase", json!([]), None, json!({"memories": erased}), vec![], None, json!({"effect": "erased"}), json!({"gen_ai.operation.name": "erase"}), now_micros())?;
        }
        Ok(json!({"principal": id, "erased": erased, "scopes": scopes, "reassigned": sponsored}))
    }
}

/// One audit event, before it is chained.
pub(crate) struct AuditEvent {
    pub op: String,
    pub chain: JsonValue,
    pub accountable: Option<String>,
    pub targets: JsonValue,
    pub relations: Vec<String>,
    pub scope_set: Option<String>,
    pub decision: JsonValue,
    pub otel: JsonValue,
    pub ts: i64,
}

/// Chain `ev` onto workspace `ws` inside `tx`: read the head (for update),
/// write the event and the new head. The caller holds the audit lock.
pub(crate) fn append_in_tx(tx: &mut SessionTx<'_>, ws: &str, ev: &AuditEvent) -> Result<()> {
    let head = tx.get_relation(TN_AUDIT_HEAD, false)?;
    let audit = tx.get_relation(TN_AUDIT, false)?;
    let hk = vec![DataValue::from(ws)].encode_as_key(head.id);
    let (seq, prev) = match tx.store_tx.get(&hk, true)? {
        Some(v) => {
            let t = try_decode_tuple_from_kv(&hk, &v, Some(3))?;
            (t.get(1).and_then(|x| x.get_int()).unwrap_or(0), t.get(2).and_then(|x| x.get_str()).unwrap_or_default().to_string())
        }
        None => (0, String::new()),
    };
    let seq = seq + 1;
    let j = json!({
        "ws": ws, "seq": seq, "ts": ev.ts, "op": ev.op, "chain": ev.chain, "accountable": ev.accountable,
        "targets": ev.targets, "relations": ev.relations, "scope_set": ev.scope_set, "decision": ev.decision,
        "otel": ev.otel, "prev": prev,
    });
    let hash = event_hash(&prev, &j);
    let row = vec![
        DataValue::from(ws),
        DataValue::from(seq),
        DataValue::from(ev.ts),
        DataValue::from(ev.op.as_str()),
        json_dv(ev.chain.clone()),
        opt_s(&ev.accountable),
        json_dv(ev.targets.clone()),
        str_list(&ev.relations),
        opt_s(&ev.scope_set),
        json_dv(ev.decision.clone()),
        json_dv(ev.otel.clone()),
        DataValue::from(prev.as_str()),
        DataValue::from(hash.as_str()),
    ];
    let sp = Default::default();
    tx.store_tx.put(&audit.encode_key_for_store(&row, sp)?, &audit.encode_val_for_store(&row, sp)?)?;
    let hrow = vec![DataValue::from(ws), DataValue::from(seq), DataValue::from(hash.as_str())];
    tx.store_tx.put(&head.encode_key_for_store(&hrow, sp)?, &head.encode_val_for_store(&hrow, sp)?)?;
    Ok(())
}

/// The write audit of one principal view.
struct WriteAudit {
    state: Arc<TenancyState>,
    ws: String,
    chain: JsonValue,
    accountable: Option<String>,
    otel: JsonValue,
}

impl CommitAudit for WriteAudit {
    fn stamp<'t>(&'t self, tx: &mut SessionTx<'_>, relations: &[SmartString<LazyCompact>]) -> Result<std::sync::MutexGuard<'t, ()>> {
        let g = self.state.audit_lock.lock().unwrap_or_else(|e| e.into_inner());
        let relations: Vec<String> = relations.iter().map(|n| n.split_once('@').map_or(n.as_str(), |x| x.1).to_string()).collect();
        let ev = AuditEvent {
            op: "write".into(),
            chain: self.chain.clone(),
            accountable: self.accountable.clone(),
            targets: json!([]),
            relations,
            scope_set: None,
            decision: json!({}),
            otel: self.otel.clone(),
            ts: now_micros(),
        };
        append_in_tx(tx, &self.ws, &ev)?;
        Ok(g)
    }
}

/// `SHA-256(prev ‖ canonical event)` (serde_json maps are key-sorted, so
/// `to_string` is canonical here).
pub(crate) fn event_hash(prev: &str, ev: &JsonValue) -> String {
    let mut h = Sha256::new();
    h.update(prev.as_bytes());
    h.update(ev.to_string().as_bytes());
    hex(&h.finalize())
}

/// The ScopeSet as a short hash (audit, caches).
pub(crate) fn scope_set_hash(nss: &[String], org: bool) -> String {
    let mut h = Sha256::new();
    for n in nss {
        h.update(n.as_bytes());
        h.update([0u8]);
    }
    h.update([org as u8]);
    hex(&h.finalize())[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_keys_round_trip() {
        for k in [
            ScopeKey::Run { agent: "bot".into(), user: Some("alice".into()), run: "r1".into() },
            ScopeKey::Run { agent: "bot".into(), user: None, run: "r1".into() },
            ScopeKey::AgentUser { agent: "bot".into(), user: "alice".into() },
            ScopeKey::User { user: "alice".into() },
            ScopeKey::Agent { agent: "bot".into() },
            ScopeKey::Session { agent: Some("bot".into()), run: "r1".into(), members: vec!["alice".into(), "bob".into()] },
            ScopeKey::Team { team: "eng".into() },
            ScopeKey::Workspace,
        ] {
            assert_eq!(ScopeKey::parse(&k.ns(), false), k);
        }
        assert_eq!(ScopeKey::parse("org", true), ScopeKey::Org);
        assert_eq!(ScopeKey::parse("org", false), ScopeKey::Legacy { ns: "org".into() });
        assert_eq!(ScopeKey::parse("alice", false), ScopeKey::Legacy { ns: "alice".into() });
        assert_eq!(ScopeKey::parse("au:bot:alice~try", false), ScopeKey::AgentUser { agent: "bot".into(), user: "alice".into() });
    }
}
