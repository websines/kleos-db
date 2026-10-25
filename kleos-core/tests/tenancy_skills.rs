/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The skill library as a principal (docs/specs/skills.md, "Tenancy"):
//! scope isolation, actor chains, Cedar-checked writes and approved
//! promotion into the org scope.

use kleos::tenancy::PrincipalContext;
use kleos::DbInstance;
use serde_json::{json, Value};

fn cx(ws: &str, human: Option<&str>, agent: Option<&str>) -> PrincipalContext {
    PrincipalContext { workspace: Some(ws.into()), human: human.map(Into::into), agent: agent.map(Into::into), ..Default::default() }
}

fn world() -> DbInstance {
    let db = DbInstance::new("mem", "", "").unwrap();
    let t = |op: &str, a: Value| db.tenant_call(op, a).unwrap_or_else(|e| panic!("tenant {op}: {e:?}"));
    t("org_create", json!({"org": "acme", "owner": "alice"}));
    for h in ["bob", "carol"] {
        t("principal_create", json!({"id": h, "org": "acme", "kind": "human"}));
    }
    t("principal_create", json!({"id": "bot", "org": "acme", "kind": "agent", "sponsor": "alice"}));
    t("ws_create", json!({"ws": "eng", "org": "acme"}));
    t("ws_create", json!({"ws": "sales", "org": "acme"}));
    t("grant", json!({"ws": "eng", "principal": "bob", "role": "writer"}));
    t("grant", json!({"ws": "eng", "principal": "carol", "role": "reader"}));
    t("grant", json!({"ws": "sales", "principal": "carol", "role": "writer"}));
    db
}

fn sk(db: &DbInstance, ctx: &PrincipalContext, op: &str, mut a: Value) -> miette::Result<Value> {
    a["op"] = json!(op);
    db.memory_call_as(ctx, "skill", a)
}

fn md(name: &str, desc: &str) -> String {
    format!("---\nname: {name}\ndescription: {desc}\n---\n1. do it\n")
}

fn names(v: &Value) -> Vec<String> {
    v["skills"].as_array().unwrap().iter().map(|s| s["name"].as_str().unwrap().to_string()).collect()
}

#[test]
fn skills_follow_scopes_actors_and_approval() {
    let db = world();
    let bob = cx("eng", Some("bob"), None);
    let carol = cx("eng", Some("carol"), None);
    // bob writes one workspace skill and one in his own scope
    let ws = sk(&db, &bob, "put", json!({"skill_md": md("deploy-service", "Deploy a service to staging."), "scope": "workspace", "activate": true})).unwrap();
    assert_eq!(ws["scope"], "ws");
    sk(&db, &bob, "put", json!({"skill_md": md("deploy-private", "Deploy notes for my own staging service."), "scope": "user", "activate": true})).unwrap();
    // the actor chain is on the version
    let v = sk(&db, &bob, "get", json!({"name": "deploy-service"})).unwrap();
    assert_eq!(v["actor"]["human"], "bob");
    assert!(v["actor"]["chain"].is_array());

    // isolation: carol (reader) sees the workspace skill, never bob's own
    assert_eq!(names(&sk(&db, &bob, "find", json!({"task": "deploy a service to staging", "k": 5})).unwrap()).len(), 2);
    assert_eq!(names(&sk(&db, &carol, "find", json!({"task": "deploy a service to staging", "k": 5})).unwrap()), vec!["deploy-service"]);
    assert!(sk(&db, &carol, "get", json!({"name": "deploy-private"})).is_err());
    assert!(sk(&db, &carol, "entry", json!({"uri": "skill://user/bob/deploy-private/SKILL.md"})).is_err());
    // the sales workspace is another partition
    assert!(names(&sk(&db, &cx("sales", Some("carol"), None), "find", json!({"task": "deploy a service"})).unwrap()).is_empty());
    // a reader cannot write or activate; filesystem inputs are refused
    assert!(sk(&db, &carol, "put", json!({"skill_md": md("x-skill", "x"), "scope": "workspace"})).is_err());
    assert!(sk(&db, &carol, "retire", json!({"name": "deploy-service"})).is_err());
    assert!(sk(&db, &bob, "import", json!({"path": "/etc"})).is_err());
    // a reader may report outcomes
    assert_eq!(sk(&db, &carol, "record_outcome", json!({"name": "deploy-service", "success": true})).unwrap()["success"], true);

    // promotion into org waits for an org admin, whatever the request says
    let p = sk(&db, &bob, "promote", json!({"name": "deploy-service", "to": "org", "approve": true})).unwrap();
    assert_eq!(p["status"], "pending");
    let pid = p["promotion"].as_str().unwrap().to_string();
    assert!(sk(&db, &bob, "approve", json!({"promotion": pid})).is_err(), "bob may not approve into org");
    assert!(sk(&db, &cx("eng", None, Some("bot")), "approve", json!({"promotion": pid})).is_err(), "only a human approves");
    let ok = sk(&db, &cx("eng", Some("alice"), None), "approve", json!({"promotion": pid})).unwrap();
    assert_eq!(ok["status"], "approved");
    // the copy lives in the org partition, readable from any workspace of the org
    let f = sk(&db, &cx("sales", Some("carol"), None), "find", json!({"task": "deploy a service to staging"})).unwrap();
    assert_eq!(f["skills"][0]["scope"], "org");
    let g = sk(&db, &cx("sales", Some("carol"), None), "get", json!({"name": "deploy-service"})).unwrap();
    assert_eq!(g["origin"], "promoted");
    assert_eq!(g["provenance"]["approver"]["human"], "alice");

    // the ops are in the workspace's audit chain
    let audit = db.tenant_call("audit", json!({"ws": "eng", "limit": 200})).unwrap();
    let ops: Vec<String> = audit.as_array().unwrap().iter().filter_map(|e| e["op"].as_str().map(String::from)).collect();
    assert!(ops.iter().any(|o| o == "skill.put") && ops.iter().any(|o| o == "skill.approve"), "{ops:?}");
    assert_eq!(db.tenant_call("audit_verify", json!({"ws": "eng"})).unwrap()["ok"], true);
}
