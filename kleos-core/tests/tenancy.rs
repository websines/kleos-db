/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Tenancy, identity and access control (docs/specs/tenancy.md), with a
//! fake embedder and decider registered per test.

use std::sync::Arc;

use kleos::decide::{self, Family};
use kleos::embed::{self, Embedder};
use kleos::memory::{InitOpts, RecallOpts};
use kleos::tenancy::PrincipalContext;
use kleos::{DbInstance, DecideAnswer, DecideRequest, Decider, ScriptMutability};
use serde_json::{json, Value};

/// Bag-of-words hashed into 64 dims, L2-normalised.
struct Words;

impl Embedder for Words {
    fn embed(&self, texts: &[&str]) -> miette::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0f32; 64];
                for w in t.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2) {
                    let h = w.to_lowercase().bytes().fold(1469598103934665603u64, |h, b| (h ^ b as u64).wrapping_mul(1099511628211));
                    v[(h % 64) as usize] += 1.0;
                }
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                v.iter().map(|x| x / n).collect()
            })
            .collect())
    }
    fn dim(&self) -> usize {
        64
    }
}

/// NLI fake: (old, new) pairs in `contra` contradict, all else is neutral.
struct Nli(Vec<(&'static str, &'static str)>);

impl Decider for Nli {
    fn model_id(&self) -> String {
        "tn-nli".into()
    }
    fn weights_version(&self) -> Option<String> {
        Some("nano-tune-v1".to_string())
    }
    fn decide(&self, reqs: &[DecideRequest], _f: bool) -> miette::Result<Vec<DecideAnswer>> {
        Ok(reqs
            .iter()
            .map(|r| {
                let st = r.state.strip_prefix("Premise: ").unwrap_or(&r.state);
                let (a, b) = st.split_once("\nHypothesis: ").unwrap_or((st, ""));
                let logits = if self.0.iter().any(|&(x, y)| x == a && y == b) { vec![0.0, 0.0, 12.0] } else { vec![0.0, 6.0, 0.0] };
                DecideAnswer { logits, features: vec![] }
            })
            .collect())
    }
}

const PARIS: &str = "I live in Paris.";
const LISBON: &str = "I moved to Lisbon.";

/// A store with models, org `acme` (owner alice), humans bob and carol,
/// agent `bot` (sponsor alice), workspaces `eng` and `sales`, and team
/// `core` (alice, bot) with writer on eng.
struct World {
    db: DbInstance,
    tag: String,
}

impl World {
    fn new(tag: &str) -> World {
        let tag = format!("tn_{tag}");
        embed::register_embedder(&tag, Arc::new(Words)).unwrap();
        decide::register_decider(&tag, Arc::new(Nli(vec![(PARIS, LISBON)])), Family::Nano).unwrap();
        let db = DbInstance::new("mem", "", "").unwrap();
        db.memory_init(InitOpts { embedder: Some(tag.clone()), decider: Some(tag.clone()), ..Default::default() }).unwrap();
        let w = World { db, tag };
        w.t("org_create", json!({"org": "acme", "owner": "alice"}));
        for h in ["bob", "carol"] {
            w.t("principal_create", json!({"id": h, "org": "acme", "kind": "human"}));
        }
        w.t("principal_create", json!({"id": "bot", "org": "acme", "kind": "agent", "sponsor": "alice"}));
        w.t("ws_create", json!({"ws": "eng", "org": "acme"}));
        w.t("ws_create", json!({"ws": "sales", "org": "acme"}));
        w.t("team_create", json!({"team": "core", "org": "acme"}));
        for m in ["alice", "bot"] {
            w.t("member_add", json!({"group": "team:core", "principal": m}));
        }
        w.t("grant", json!({"ws": "eng", "team": "core", "role": "writer"}));
        w.t("grant", json!({"ws": "eng", "principal": "bob", "role": "writer"}));
        w.t("grant", json!({"ws": "eng", "principal": "carol", "role": "reader"}));
        w.t("grant", json!({"ws": "sales", "principal": "carol", "role": "writer"}));
        w
    }
    fn t(&self, op: &str, args: Value) -> Value {
        self.db.tenant_call(op, args).unwrap_or_else(|e| panic!("tenant {op}: {e:?}"))
    }
    fn call(&self, ctx: &PrincipalContext, op: &str, args: Value) -> miette::Result<Value> {
        self.db.memory_call_as(ctx, op, args)
    }
    fn put(&self, ctx: &PrincipalContext, text: &str, scope: Option<&str>) -> Value {
        let mut a = json!({"text": text, "split": "none"});
        if let Some(s) = scope {
            a["scope"] = json!(s);
        }
        self.call(ctx, "remember", a).unwrap()
    }
    /// Texts recalled (graph and expansion on).
    fn texts(&self, ctx: &PrincipalContext, q: &str) -> Vec<String> {
        self.call(ctx, "recall", json!({"query": q, "k": 20, "expand": true, "hit_text": "claim"}))
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|h| h["text"].as_str().unwrap().to_string())
            .collect()
    }
}

impl Drop for World {
    fn drop(&mut self) {
        embed::unregister_embedder(&self.tag);
        decide::unregister_decider(&self.tag);
    }
}

fn cx(ws: &str, human: Option<&str>, agent: Option<&str>) -> PrincipalContext {
    PrincipalContext {
        workspace: Some(ws.into()),
        human: human.map(Into::into),
        agent: agent.map(Into::into),
        ..Default::default()
    }
}

#[test]
fn no_tenancy_behaves_as_before() {
    let w = DbInstance::new("mem", "", "").unwrap();
    w.remember("I like tea.", Default::default()).unwrap();
    let hits = w.recall("tea", RecallOpts::default()).unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].scope.is_none());
    assert!(!w.tenancy_enabled());
    let rels = w.run_script("::relations", Default::default(), ScriptMutability::Immutable).unwrap();
    assert!(rels.rows.iter().all(|r| !r[0].get_str().unwrap().starts_with("kleos_tn_")));
}

#[test]
fn partitions_are_separate_databases() {
    let w = World::new("part");
    let alice = cx("eng", Some("alice"), None);
    w.put(&alice, "The eng roadmap ships in March.", Some("workspace"));
    // the partition's relations are its own; the plain store hides them
    let eng = w.db.partition_view("ws_eng").unwrap();
    let n = |db: &DbInstance| db.run_script("?[id] := *kleos_mem{id}", Default::default(), ScriptMutability::Immutable).unwrap().rows.len();
    assert_eq!(n(&eng), 1);
    assert_eq!(n(&w.db), 0);
    assert_eq!(n(&w.db.partition_view("ws_sales").unwrap()), 0);
    let rels = w.db.run_script("::relations", Default::default(), ScriptMutability::Immutable).unwrap();
    assert!(rels.rows.iter().all(|r| !r[0].get_str().unwrap().contains('@')));
    // workspace isolation: carol writes in sales, nobody in eng sees it
    w.put(&cx("sales", Some("carol"), None), "The sales roadmap ships in March.", Some("workspace"));
    assert_eq!(w.texts(&cx("eng", Some("carol"), None), "roadmap March"), vec!["The eng roadmap ships in March."]);
    // no grant on sales: bob is refused there
    assert!(w.call(&cx("sales", Some("bob"), None), "recall", json!({"query": "roadmap"})).unwrap().as_array().unwrap().is_empty());
    assert!(w.call(&cx("sales", Some("bob"), None), "remember", json!({"text": "x", "scope": "workspace"})).is_err());
}

#[test]
fn agent_user_scope_does_not_leak_across_users() {
    let w = World::new("au");
    w.t("grant", json!({"ws": "eng", "principal": "bot", "role": "writer"}));
    let r = w.put(&cx("eng", Some("alice"), Some("bot")), "My passport number is X123.", None);
    assert_eq!(r["scope"], "au:bot:alice");
    // the same agent serving bob does not see it; alice does
    assert!(w.texts(&cx("eng", Some("bob"), Some("bot")), "passport number").is_empty());
    assert_eq!(w.texts(&cx("eng", Some("alice"), None), "passport number").len(), 1);
    // an autonomous run of the agent does not read per-user scopes either
    assert!(w.texts(&cx("eng", None, Some("bot")), "passport number").is_empty());
}

#[test]
fn intersection_includes_participants_and_token() {
    let w = World::new("inter");
    let alice = cx("eng", Some("alice"), None);
    w.put(&alice, "Core team: the launch slips a week.", Some("team:core"));
    assert_eq!(w.texts(&alice, "launch slips").len(), 1);
    // bob is not in core: he cannot read it, and his presence in a shared
    // session hides it from alice too
    assert!(w.texts(&cx("eng", Some("bob"), None), "launch slips").is_empty());
    let shared = PrincipalContext { participants: vec!["bob".into()], ..alice.clone() };
    assert!(w.texts(&shared, "launch slips").is_empty());
    // the agent is in core, so alice + bot still see it
    assert_eq!(w.texts(&cx("eng", Some("alice"), Some("bot")), "launch slips").len(), 1);
    // a credential limited to the workspace scope kind sees nothing of the team
    let narrow = PrincipalContext {
        token: Some(kleos::tenancy::TokenGrant { actions: None, kinds: Some(vec!["ws".into()]) }),
        ..alice.clone()
    };
    assert!(w.texts(&narrow, "launch slips").is_empty());
    // an org policy can forbid more
    w.t(
        "policy_set",
        json!({"org": "acme", "cedar": "forbid(principal, action == Kleos::Action::\"read\", resource) when { principal.kind == \"agent\" && resource.kind == \"team\" };"}),
    );
    assert!(w.texts(&cx("eng", Some("alice"), Some("bot")), "launch slips").is_empty());
}

#[test]
fn shared_session_memory_is_an_intersection_scope() {
    let w = World::new("sess");
    let with = |h: &str, others: &[&str]| PrincipalContext {
        run: Some("standup".into()),
        participants: others.iter().map(|x| x.to_string()).collect(),
        ..cx("eng", Some(h), Some("bot"))
    };
    let r = w.put(&with("alice", &["bob"]), "We agreed to ship the beta on Friday.", None);
    assert_eq!(r["scope"], "sess:bot:standup:alice+bob");
    // the same session, and each member alone, see it; an outsider's
    // presence hides it
    assert_eq!(w.texts(&with("bob", &["alice"]), "ship beta Friday").len(), 1);
    assert_eq!(w.texts(&cx("eng", Some("bob"), None), "ship beta Friday").len(), 1);
    assert!(w.texts(&with("alice", &["carol"]), "ship beta Friday").is_empty());
    assert!(w.texts(&cx("eng", Some("carol"), None), "ship beta Friday").is_empty());
}

#[test]
fn graph_walk_stays_inside_visible_scopes() {
    let w = World::new("ppr");
    let alice = cx("eng", Some("alice"), None);
    let bob = cx("eng", Some("bob"), None);
    let ents = |t: &str, scope: &str| json!({"text": t, "scope": scope, "split": "none", "entities": ["Anna"]});
    w.call(&bob, "remember", ents("Anna leads the platform team.", "workspace")).unwrap();
    w.call(&alice, "remember", ents("Anna is pregnant and on leave in May.", "user")).unwrap();
    w.call(&alice, "remember", ents("Anna's home address is 4 Elm Street.", "user")).unwrap();
    // bob's walk (PPR + entity expansion) never reaches alice's memories
    let got = w.call(&bob, "recall", json!({"query": "Anna platform", "graph": "ppr", "expand": true, "k": 10})).unwrap();
    let hits = got.as_array().unwrap();
    assert_eq!(hits.len(), 1, "{got}");
    assert_eq!(hits[0]["scope"], "ws");
    assert!(hits.iter().all(|h| h.get("via").is_none()));
    // alice sees both scopes, each hit labelled
    let got = w.call(&alice, "recall", json!({"query": "Anna", "graph": "ppr", "k": 10})).unwrap();
    let scopes: Vec<&str> = got.as_array().unwrap().iter().map(|h| h["scope"].as_str().unwrap()).collect();
    assert!(scopes.contains(&"ws") && scopes.contains(&"user:alice"), "{scopes:?}");
}

#[test]
fn summaries_are_per_scope() {
    let w = World::new("sum");
    let alice = cx("eng", Some("alice"), None);
    for i in 0..9 {
        let a = json!({"text": format!("Standup note {i}: the deploy pipeline needs work."), "scope": "workspace", "source": format!("standup#{i}"), "split": "none"});
        w.call(&alice, "remember", a).unwrap();
        let p = json!({"text": format!("Private note {i}: my divorce lawyer called."), "scope": "user", "source": format!("standup#{i}"), "split": "none"});
        w.call(&alice, "remember", p).unwrap();
    }
    for scope in ["workspace", "user"] {
        w.call(&alice, "consolidate", json!({"scope": scope, "min_claims": 8})).unwrap();
    }
    // bob's context has the workspace summary and nothing of alice's notes
    let ctx = w.call(&cx("eng", Some("bob"), None), "context", json!({"query": "standup notes", "budget": 4000})).unwrap();
    let text = ctx["text"].as_str().unwrap();
    assert!(text.contains("deploy pipeline"), "{text}");
    assert!(!text.contains("divorce"), "{text}");
}

#[test]
fn promotion_is_a_governed_copy() {
    let w = World::new("promo");
    let ab = cx("eng", Some("alice"), Some("bot"));
    let id = w.put(&ab, "Deploys freeze on Fridays.", Some("team:core"))["id"].as_str().unwrap().to_string();
    // the agent holds no promote grant: a pending request
    let r = w.call(&ab, "promote", json!({"ids": [id], "to": "workspace"})).unwrap();
    assert_eq!(r["status"], "pending");
    let bob = cx("eng", Some("bob"), None);
    assert!(w.texts(&bob, "deploys freeze").is_empty());
    // bob (writer) approves; the copy carries provenance
    let pid = r["promotion"].as_str().unwrap();
    assert_eq!(w.call(&bob, "approve", json!({"promotion": pid})).unwrap()["status"], "approved");
    assert_eq!(w.texts(&bob, "deploys freeze"), vec!["Deploys freeze on Fridays."]);
    let eng = w.db.partition_view("ws_eng").unwrap();
    let prov = eng.run_script("?[from_id, ab] := *kleos_mem_promoted{from_id, approved_by: ab}", Default::default(), ScriptMutability::Immutable).unwrap();
    assert_eq!(prov.rows[0][0].get_str(), Some(id.as_str()));
    assert_eq!(prov.rows[0][1].get_str(), Some("bob"));
    // with the grant, the agent promotes into its own agent scope directly
    w.t("grant", json!({"ws": "eng", "principal": "bot", "role": "writer", "actions": ["promote"]}));
    assert_eq!(w.call(&ab, "promote", json!({"ids": [id], "to": "agent"})).unwrap()["status"], "done");
    // org scope: an org admin's explicit approval, else pending
    let alice = cx("eng", Some("alice"), None);
    assert_eq!(w.call(&alice, "promote", json!({"ids": [id], "to": "org"})).unwrap()["status"], "pending");
    assert_eq!(w.call(&alice, "promote", json!({"ids": [id], "to": "org", "approve": true})).unwrap()["status"], "done");
    // carol, a member of the org, now reads it from the sales workspace
    assert_eq!(w.texts(&cx("sales", Some("carol"), None), "deploys freeze"), vec!["Deploys freeze on Fridays."]);
}

#[test]
fn supersession_and_duplicates_stay_within_a_scope() {
    let w = World::new("sup");
    let alice = cx("eng", Some("alice"), None);
    w.put(&alice, PARIS, Some("workspace"));
    let r = w.put(&alice, LISBON, Some("user"));
    // the decider would supersede Paris, but it is in another scope
    assert!(r["superseded"].as_array().map_or(true, |v| v.is_empty()), "{r}");
    assert_eq!(w.texts(&cx("eng", Some("bob"), None), "live Paris Lisbon"), vec![PARIS]);
    // within one scope it does supersede
    let r = w.put(&alice, LISBON, Some("workspace"));
    assert_eq!(r["superseded"].as_array().unwrap().len(), 1, "{r}");
    // an exact restatement in two scopes: the narrower wins at read time
    w.put(&alice, "Coffee is free on Mondays.", Some("workspace"));
    w.put(&alice, "Coffee is free on Mondays.", Some("user"));
    let got = w.call(&alice, "recall", json!({"query": "coffee free Mondays", "k": 5})).unwrap();
    let coffee: Vec<&Value> = got.as_array().unwrap().iter().filter(|h| h["text"].as_str().unwrap().contains("Coffee")).collect();
    assert_eq!(coffee.len(), 1);
    assert_eq!(coffee[0]["scope"], "user:alice");
}

#[test]
fn erasure_cascades_and_shreds_the_audit_trail() {
    let w = World::new("erase");
    let alice = cx("eng", Some("alice"), None);
    w.put(&alice, "My blood type is O negative.", None);
    w.put(&alice, "Alice thinks the API needs versioning.", Some("workspace"));
    w.call(&alice, "recall", json!({"query": "blood type"})).unwrap();
    let before = w.t("audit", json!({"ws": "eng", "principal": "alice"})).as_array().unwrap().len();
    assert!(before >= 3);
    // alice sponsors bot: erasure needs a new sponsor
    assert!(w.db.tenant_call("erase", json!({"principal": "alice"})).is_err());
    let r = w.t("erase", json!({"principal": "alice", "reassign_to": "bob"}));
    assert_eq!(r["erased"], 2, "{r}");
    let bob = cx("eng", Some("bob"), None);
    assert!(w.texts(&bob, "API versioning").is_empty());
    let eng = w.db.partition_view("ws_eng").unwrap();
    let left = |rel: &str| eng.run_script(&format!("?[id] := *{rel}{{id}}"), Default::default(), ScriptMutability::Immutable).unwrap().rows.len();
    assert_eq!(left("kleos_mem"), 0);
    assert_eq!(left("kleos_mem_actor"), 0);
    // the chain still verifies; alice's events can no longer be linked to her
    assert_eq!(w.t("audit_verify", json!({"ws": "eng"}))["ok"], true);
    assert!(w.db.tenant_call("audit", json!({"ws": "eng", "principal": "alice"})).unwrap().as_array().unwrap().is_empty());
    assert!(w.call(&alice, "recall", json!({"query": "x"})).is_err());
}

#[test]
fn audit_chain_covers_writes_and_reads() {
    let w = World::new("audit");
    let ab = cx("eng", Some("alice"), Some("bot"));
    let id = w.put(&ab, "The staging password rotates monthly.", Some("team:core"))["id"].as_str().unwrap().to_string();
    let got = w.call(&ab, "recall", json!({"query": "staging password"})).unwrap();
    assert_eq!(got[0]["id"], id);
    let evs = w.t("audit", json!({"ws": "eng"}));
    let evs = evs.as_array().unwrap();
    let op = |o: &str| evs.iter().filter(|e| e["op"] == o).collect::<Vec<_>>();
    // the write: relation-level and op-level events, each with the chain
    assert!(op("write").iter().any(|e| e["relations"].as_array().unwrap().iter().any(|r| r == "kleos_mem")));
    let rem = op("remember");
    assert_eq!(rem[0]["targets"]["ids"][0], id);
    assert_eq!(rem[0]["chain"].as_array().unwrap().len(), 2);
    assert_eq!(rem[0]["chain"][1]["kind"], "ai_agent");
    assert!(rem[0]["chain"][0]["sub"].as_str().unwrap().starts_with("p:"), "pseudonymous ids");
    // the read: the returned ids, never content
    let rec = op("recall");
    assert_eq!(rec[0]["targets"]["ids"][0], id);
    assert!(!rec[0].to_string().contains("staging password"));
    assert_eq!(rec[0]["otel"]["gen_ai.operation.name"], "recall");
    // tampering breaks the chain
    assert_eq!(w.t("audit_verify", json!({"ws": "eng"}))["ok"], true);
    w.db.run_script(
        "?[ws, seq, op] := *kleos_tn_audit{ws, seq}, ws = 'eng', seq = 2, op = 'nothing' :update kleos_tn_audit {ws, seq => op}",
        Default::default(),
        ScriptMutability::Mutable,
    )
    .unwrap();
    assert_eq!(w.t("audit_verify", json!({"ws": "eng"}))["ok"], false);
}

#[test]
fn selective_scope_set_finds_every_visible_memory() {
    let w = World::new("sel");
    // 150 memories bob's agent keeps about him; carol sees none of them
    let bb = cx("eng", Some("bob"), Some("bot"));
    for i in 0..150 {
        w.put(&bb, &format!("Bob's note {i} about the quarterly planning budget."), None);
    }
    // carol (a reader) has one note of her own and reads one workspace note
    let me = cx("eng", Some("carol"), None);
    w.put(&me, "Planning budget review is on Tuesday.", Some("user"));
    w.put(&cx("eng", Some("bob"), None), "The planning budget is frozen until May.", Some("workspace"));
    let mut got = w.texts(&me, "quarterly planning budget");
    got.sort();
    assert_eq!(got, vec!["Planning budget review is on Tuesday.", "The planning budget is frozen until May."]);
}

#[test]
fn cozoscript_ops() {
    let w = World::new("script");
    let run = |s: &str| w.db.run_script(s, Default::default(), ScriptMutability::Mutable).unwrap();
    run("::principal remember {as_workspace: 'eng', as_human: 'bob', text: 'Lunch is at noon.', scope: 'workspace'}");
    let r = run(r#"::principal recall {as: '{"workspace": "eng", "human": "carol"}', query: 'lunch noon'}"#);
    assert_eq!(r.rows.len(), 1);
    let r = run(r#"::tenant scopes {as: parse_json('{"workspace": "eng", "human": "carol"}')}"#);
    assert!(r.rows[0].iter().any(|v| format!("{v:?}").contains("ws")));
}

/// Recall latency with a ScopeSet pre-filter (principal mode, 8 scopes of a
/// 4,000-memory workspace) against a plain one-namespace recall of a store
/// of the same size. `cargo test --release --test tenancy perf -- --ignored --nocapture`
#[test]
#[ignore]
fn perf_scope_filter() {
    let w = World::new("perf");
    w.t("grant", json!({"ws": "eng", "principal": "bot", "role": "writer"}));
    let eng = w.db.partition_view("ws_eng").unwrap();
    let plain = DbInstance::new("mem", "", "").unwrap();
    plain.memory_init(InitOpts { embedder: Some(w.tag.clone()), ..Default::default() }).unwrap();
    let topics = ["deploy", "budget", "hiring", "roadmap", "incident", "design", "travel", "lunch"];
    let users = ["alice", "bob", "carol"];
    for i in 0..4000 {
        let text = format!("Note {i} about {} and item {}.", topics[i % 8], i * 7 % 101);
        // eng: per-user agent scopes, team, workspace; plain: one namespace
        let ns = match i % 5 {
            0 => "ws".to_string(),
            1 => "team:core".to_string(),
            k => format!("au:bot:{}", users[k % 3]),
        };
        for (db, ns) in [(&eng, ns.as_str()), (&plain, "default")] {
            db.remember(&text, kleos::memory::RememberOpts { ns: Some(ns.into()), split: Some("none".into()), extract: Some(false), ..Default::default() }).unwrap();
        }
    }
    w.t("ws_create", json!({"ws": "x", "org": "acme"})); // bump epochs once
    let ctx = cx("eng", Some("alice"), Some("bot"));
    let qs: Vec<String> = (0..50).map(|i| format!("{} item {}", topics[i % 8], i)).collect();
    let time = |f: &dyn Fn(&str)| {
        f(&qs[0]);
        let t = std::time::Instant::now();
        for q in &qs {
            f(q);
        }
        t.elapsed().as_secs_f64() * 1000.0 / qs.len() as f64
    };
    let scoped = time(&|q| {
        w.call(&ctx, "recall", json!({"query": q, "graph": "off", "record_access": false})).unwrap();
    });
    let unscoped = time(&|q| {
        plain.recall(q, RecallOpts { graph: Some("off".into()), record_access: Some(false), ..Default::default() }).unwrap();
    });
    let one = time(&|q| {
        w.call(&ctx, "recall", json!({"query": q, "graph": "off", "record_access": false, "scopes": ["ws"]})).unwrap();
    });
    let part = time(&|q| {
        eng.recall(q, RecallOpts { ns: Some("ws".into()), graph: Some("off".into()), record_access: Some(false), ..Default::default() }).unwrap();
    });
    let session = time(&|_| {
        w.call(&ctx, "scopes", json!({})).unwrap();
    });
    println!("recall p-mean: scoped {scoped:.2} ms, plain {unscoped:.2} ms (4,000 memories, graph off)");
    println!("  one scope as principal {one:.2} ms, one namespace of the partition {part:.2} ms, session + scope set {session:.2} ms");
}
