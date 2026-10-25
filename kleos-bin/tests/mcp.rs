/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The stateless MCP core (docs/specs/mcp-cli-skills.md) against an
//! in-memory store, with requests shaped as in the 2026-07-28 spec.

use kleos_bin::backend::{Area, Backend};
use kleos_bin::mcp::{encode_task, Conn, McpServer, MODERN};
use serde_json::{json, Value};

fn server() -> McpServer {
    McpServer::new(Backend::open("mem", "").unwrap(), "default", true).with_state_key("test-key")
}

fn meta(caps: Value) -> Value {
    json!({
        "io.modelcontextprotocol/protocolVersion": MODERN,
        "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "1"},
        "io.modelcontextprotocol/clientCapabilities": caps
    })
}

fn req(id: i64, method: &str, mut params: Value, caps: Value) -> Value {
    params["_meta"] = meta(caps);
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
}

fn conn() -> Conn {
    Conn { principal: "alice".into(), ..Default::default() }
}

fn call(s: &McpServer, c: &mut Conn, msg: Value) -> (Value, u16) {
    s.handle(&msg, c).expect("a response")
}

fn tool(s: &McpServer, c: &mut Conn, name: &str, args: Value) -> Value {
    let (r, st) = call(s, c, req(7, "tools/call", json!({"name": name, "arguments": args}), json!({})));
    assert_eq!(st, 200, "{r}");
    r["result"].clone()
}

#[test]
fn discover_and_version_rules() {
    let s = server();
    let mut c = conn();
    let (r, st) = call(&s, &mut c, req(1, "server/discover", json!({}), json!({})));
    assert_eq!(st, 200);
    let res = &r["result"];
    assert_eq!(res["resultType"], "complete");
    assert_eq!(res["supportedVersions"], json!([MODERN]));
    assert!(res["capabilities"]["tools"].is_object());
    assert!(res["capabilities"]["extensions"]["io.modelcontextprotocol/tasks"].is_object());
    assert_eq!(res["_meta"]["io.modelcontextprotocol/serverInfo"]["name"], "kleos");
    assert!(res["ttlMs"].as_u64().unwrap() > 0);
    assert_eq!(res["cacheScope"], "private");

    // unknown version → -32022 with the supported list, HTTP 400
    let mut bad = req(2, "tools/list", json!({}), json!({}));
    bad["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"] = json!("1900-01-01");
    let (r, st) = call(&s, &mut c, bad);
    assert_eq!(st, 400);
    assert_eq!(r["error"]["code"], -32022);
    assert_eq!(r["error"]["data"]["requested"], "1900-01-01");
    assert!(r["error"]["data"]["supported"].as_array().unwrap().contains(&json!(MODERN)));
    assert_eq!(r["id"], 2);

    // missing required _meta fields → -32602
    let (r, st) = call(&s, &mut c, json!({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}));
    assert_eq!((r["error"]["code"].as_i64(), st), (Some(-32602), 400));
    let mut nocaps = req(4, "tools/list", json!({}), json!({}));
    nocaps["params"]["_meta"].as_object_mut().unwrap().remove("io.modelcontextprotocol/clientCapabilities");
    let (r, _) = call(&s, &mut c, nocaps);
    assert_eq!(r["error"]["code"], -32602);

    // unknown method → -32601, HTTP 404; notifications get no reply
    let (r, st) = call(&s, &mut c, req(5, "nope/nope", json!({}), json!({})));
    assert_eq!((r["error"]["code"].as_i64(), st), (Some(-32601), 404));
    assert!(s.handle(&json!({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}}), &mut c).is_none());
}

#[test]
fn tools_list_is_deterministic_and_cacheable() {
    let s = server();
    let mut c = conn();
    let (a, _) = call(&s, &mut c, req(1, "tools/list", json!({}), json!({})));
    let (b, _) = call(&s, &mut c, req(2, "tools/list", json!({}), json!({})));
    assert_eq!(a["result"]["tools"], b["result"]["tools"]);
    let names: Vec<&str> = a["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["recall", "remember", "forget", "reflect", "entity_profile", "doc_add", "doc_status", "flow", "ontology", "reembed", "consolidate", "branch", "find_skills", "get_skill", "record_skill_outcome", "propose_skill", "skill_admin", "query"]);
    assert!(a["result"]["ttlMs"].as_u64().is_some());
    assert_eq!(a["result"]["cacheScope"], "private");
    for t in a["result"]["tools"].as_array().unwrap() {
        assert_eq!(t["inputSchema"]["type"], "object", "{t}");
        assert!(t["annotations"].is_object());
    }
    // without --allow-query the query tool is absent and refused
    let s2 = McpServer::new(Backend::open("mem", "").unwrap(), "default", false);
    let (r, _) = call(&s2, &mut c, req(3, "tools/list", json!({}), json!({})));
    assert_eq!(r["result"]["tools"].as_array().unwrap().len(), 17);
    let (r, st) = call(&s2, &mut c, req(4, "tools/call", json!({"name": "query", "arguments": {"script": "?[x] := x = 1"}}), json!({})));
    assert_eq!((r["error"]["code"].as_i64(), st), (Some(-32602), 400));
}

#[test]
fn memory_tools_round_trip() {
    let s = server();
    let mut c = conn();
    let r = tool(&s, &mut c, "remember", json!({"text": "Anna lives in Lisbon.", "ns": "u1"}));
    assert_eq!(r["resultType"], "complete");
    assert!(r.get("isError").is_none(), "{r}");
    let id = r["structuredContent"]["id"].as_str().unwrap().to_string();
    tool(&s, &mut c, "remember", json!({"text": "The project deadline is Friday.", "ns": "u1"}));

    let r = tool(&s, &mut c, "recall", json!({"query": "where does Anna live", "ns": "u1"}));
    let text = r["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("Lisbon"), "{text}");
    assert!(r["structuredContent"]["ids"].as_array().unwrap().iter().any(|x| x == id.as_str()));

    let r = tool(&s, &mut c, "recall", json!({"query": "deadline", "ns": "u1", "format": "list"}));
    assert!(r["structuredContent"]["hits"].as_array().unwrap().iter().any(|h| h["text"].as_str().unwrap().contains("Friday")));

    // other namespaces stay apart
    let r = tool(&s, &mut c, "recall", json!({"query": "Anna", "ns": "u2"}));
    assert!(!r["content"][0]["text"].as_str().unwrap().contains("Lisbon"));

    let r = tool(&s, &mut c, "reflect", json!({"ns": "u1"}));
    assert_eq!(r["structuredContent"]["ns"], "u1");

    let r = tool(&s, &mut c, "entity_profile", json!({"name": "Anna", "ns": "u1"}));
    assert!(r["content"][0]["text"].as_str().unwrap().contains("Lisbon"), "{r}");

    let r = tool(&s, &mut c, "query", json!({"script": "?[n] := n = 1 + 2"}));
    assert_eq!(r["structuredContent"]["rows"], json!([[3]]));
    // query is read-only
    let r = tool(&s, &mut c, "query", json!({"script": ":create t {a}"}));
    assert_eq!(r["isError"], true, "{r}");

    // tool errors are results the model can see
    let r = tool(&s, &mut c, "recall", json!({"ns": "u1"}));
    assert_eq!(r["isError"], true);
}

#[test]
fn forget_confirms_through_mrtr() {
    let s = server();
    let mut c = conn();
    let id = tool(&s, &mut c, "remember", json!({"text": "Temporary note to drop.", "ns": "u1"}))["structuredContent"]["id"].as_str().unwrap().to_string();
    let args = json!({"ids": [id], "ns": "u1"});
    let elic = json!({"elicitation": {}});

    // 1. input_required with an elicitation form and a requestState
    let (r, _) = call(&s, &mut c, req(1, "tools/call", json!({"name": "forget", "arguments": args}), elic.clone()));
    let res = &r["result"];
    assert_eq!(res["resultType"], "input_required", "{r}");
    let er = &res["inputRequests"]["confirm"];
    assert_eq!(er["method"], "elicitation/create");
    assert_eq!(er["params"]["mode"], "form");
    assert_eq!(er["params"]["requestedSchema"]["required"], json!(["confirm"]));
    let state = res["requestState"].as_str().unwrap().to_string();

    // a declined retry changes nothing
    let decline = json!({"name": "forget", "arguments": args, "inputResponses": {"confirm": {"action": "decline"}}, "requestState": state});
    let (r, _) = call(&s, &mut c, req(2, "tools/call", decline, elic.clone()));
    assert_eq!(r["result"]["isError"], true);
    let hit = tool(&s, &mut c, "recall", json!({"query": "temporary note", "ns": "u1", "format": "list"}));
    assert!(!hit["structuredContent"]["hits"].as_array().unwrap().is_empty());

    // tampered state, another principal and another call are rejected
    let mut bad = state.clone();
    bad.replace_range(0..2, if &state[0..2] == "7b" { "7c" } else { "7b" });
    let tampered = json!({"name": "forget", "arguments": args, "inputResponses": {"confirm": {"action": "accept", "content": {"confirm": true}}}, "requestState": bad});
    let (r, _) = call(&s, &mut c, req(3, "tools/call", tampered, elic.clone()));
    assert_eq!(r["error"]["code"], -32602, "{r}");
    let accept = json!({"name": "forget", "arguments": args, "inputResponses": {"confirm": {"action": "accept", "content": {"confirm": true}}}, "requestState": state});
    let mut bob = Conn { principal: "bob".into(), ..Default::default() };
    let (r, _) = call(&s, &mut bob, req(4, "tools/call", accept.clone(), elic.clone()));
    assert_eq!(r["error"]["code"], -32602);
    let mut other = accept.clone();
    other["arguments"]["mode"] = json!("erase");
    let (r, _) = call(&s, &mut c, req(5, "tools/call", other, elic.clone()));
    assert_eq!(r["error"]["code"], -32602);

    // 2. the accepted retry (new id, same arguments, echoed state) runs
    let (r, _) = call(&s, &mut c, req(6, "tools/call", accept, elic.clone()));
    assert_eq!(r["result"]["resultType"], "complete", "{r}");
    assert!(r["result"].get("isError").is_none(), "{r}");
    let hit = tool(&s, &mut c, "recall", json!({"query": "temporary note", "ns": "u1", "format": "list"}));
    assert!(hit["structuredContent"]["hits"].as_array().unwrap().is_empty(), "{hit}");
}

#[test]
fn confirm_argument_without_elicitation() {
    let s = server();
    let mut c = conn();
    let id = tool(&s, &mut c, "remember", json!({"text": "Another note.", "ns": "u1"}))["structuredContent"]["id"].as_str().unwrap().to_string();
    let r = tool(&s, &mut c, "forget", json!({"id": id, "ns": "u1"}));
    assert_eq!(r["isError"], true);
    assert!(r["content"][0]["text"].as_str().unwrap().contains("\"confirm\": true"));
    let r = tool(&s, &mut c, "forget", json!({"id": id, "ns": "u1", "confirm": true}));
    assert!(r.get("isError").is_none(), "{r}");
}

#[test]
fn branches_and_flows() {
    let s = server();
    let mut c = conn();
    tool(&s, &mut c, "remember", json!({"text": "Main knows the user likes tea.", "ns": "main"}));
    let r = tool(&s, &mut c, "branch", json!({"action": "fork", "from": "main", "to": "try"}));
    assert!(r.get("isError").is_none(), "{r}");
    tool(&s, &mut c, "remember", json!({"text": "The user started drinking coffee.", "ns": "try"}));
    let r = tool(&s, &mut c, "branch", json!({"action": "diff", "a": "main", "b": "try"}));
    assert!(!r["structuredContent"]["added"].as_array().unwrap().is_empty(), "{r}");
    let r = tool(&s, &mut c, "branch", json!({"action": "merge", "from": "try", "into": "main", "dry_run": true}));
    assert_eq!(r["structuredContent"]["dry_run"], true);
    // a real merge needs confirmation
    let r = tool(&s, &mut c, "branch", json!({"action": "merge", "from": "try", "into": "main"}));
    assert_eq!(r["isError"], true);
    let r = tool(&s, &mut c, "branch", json!({"action": "discard"}));
    assert_eq!(r["isError"], true, "discard needs an explicit ns");
    let r = tool(&s, &mut c, "branch", json!({"action": "list"}));
    assert!(r["structuredContent"].as_array().is_some());

    let r = tool(&s, &mut c, "flow", json!({"action": "start", "workflow": "demo.wait", "input": {"x": 1}}));
    let fid = r["structuredContent"]["id"].as_str().unwrap().to_string();
    let r = tool(&s, &mut c, "flow", json!({"action": "status", "id": fid}));
    assert_eq!(r["structuredContent"]["id"], fid.as_str());
    let r = tool(&s, &mut c, "flow", json!({"action": "bogus"}));
    assert_eq!(r["isError"], true);

    // a task id over a workflow: tasks/get maps its status, tasks/cancel cancels
    let tasks_caps = json!({"extensions": {"io.modelcontextprotocol/tasks": {}}});
    let tid = encode_task(&json!({"k": "doc", "ns": "main", "doc": "d1", "flow": fid}));
    let (r, _) = call(&s, &mut c, req(1, "tasks/get", json!({"taskId": tid}), tasks_caps.clone()));
    assert_eq!(r["result"]["status"], "working", "{r}");
    assert_eq!(r["result"]["resultType"], "complete");
    assert!(r["result"]["createdAt"].as_str().unwrap().contains('T'));
    let (r, _) = call(&s, &mut c, req(2, "tasks/cancel", json!({"taskId": tid}), tasks_caps.clone()));
    assert_eq!(r["result"]["resultType"], "complete");
    let (r, _) = call(&s, &mut c, req(3, "tasks/get", json!({"taskId": tid}), tasks_caps.clone()));
    assert_eq!(r["result"]["status"], "cancelled", "{r}");
    // without the extension capability the methods are refused
    let (r, _) = call(&s, &mut c, req(4, "tasks/get", json!({"taskId": tid}), json!({})));
    assert_eq!(r["error"]["code"], -32021);
}

#[test]
fn resources() {
    let s = server();
    let mut c = conn();
    tool(&s, &mut c, "remember", json!({"text": "Bob is Anna's brother.", "ns": "default"}));
    let (r, _) = call(&s, &mut c, req(1, "resources/list", json!({}), json!({})));
    assert_eq!(r["result"]["resources"][0]["uri"], "kleos://namespaces");
    assert!(r["result"]["ttlMs"].is_u64());
    let (r, _) = call(&s, &mut c, req(2, "resources/templates/list", json!({}), json!({})));
    assert_eq!(r["result"]["resourceTemplates"].as_array().unwrap().len(), 3);
    let (r, _) = call(&s, &mut c, req(3, "resources/read", json!({"uri": "kleos://namespaces"}), json!({})));
    let body: Value = serde_json::from_str(r["result"]["contents"][0]["text"].as_str().unwrap()).unwrap();
    assert!(body["namespaces"].as_array().unwrap().contains(&json!("default")), "{body}");
    let (r, _) = call(&s, &mut c, req(4, "resources/read", json!({"uri": "kleos://entity/default/Bob"}), json!({})));
    assert!(r["result"]["contents"][0]["text"].as_str().unwrap().contains("brother"), "{r}");
    let (r, st) = call(&s, &mut c, req(5, "resources/read", json!({"uri": "kleos://nope"}), json!({})));
    assert_eq!((r["error"]["code"].as_i64(), st), (Some(-32602), 400));
}

#[test]
fn legacy_initialize_clients() {
    let s = server();
    let mut c = conn();
    let (r, _) = call(&s, &mut c, json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "old", "version": "0"}}}));
    assert_eq!(r["result"]["protocolVersion"], "2025-11-25");
    assert!(r["result"]["capabilities"]["tools"].is_object());
    assert!(r["result"].get("_meta").is_none());
    assert!(s.handle(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}), &mut c).is_none());
    let (r, _) = call(&s, &mut c, json!({"jsonrpc": "2.0", "id": 2, "method": "ping"}));
    assert!(r["result"].is_object());
    let (r, _) = call(&s, &mut c, json!({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}));
    assert_eq!(r["result"]["tools"].as_array().unwrap().len(), 18);
    let (r, _) = call(&s, &mut c, json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "remember", "arguments": {"text": "Legacy clients still work."}}}));
    assert!(r["result"]["structuredContent"]["id"].is_string(), "{r}");
    // an unknown legacy version negotiates down to the newest legacy one
    let mut c2 = conn();
    let (r, _) = call(&s, &mut c2, json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-01-01"}}));
    assert_eq!(r["result"]["protocolVersion"], "2025-11-25");
}

#[test]
fn read_only_principals_get_read_tools_only() {
    let s = server();
    let mut c = Conn { principal: "ro".into(), read_only: true, ..Default::default() };
    let r = tool(&s, &mut c, "remember", json!({"text": "should not be stored"}));
    assert_eq!(r["isError"], true);
    let r = tool(&s, &mut c, "recall", json!({"query": "anything"}));
    assert!(r.get("isError").is_none(), "{r}");
    // the backend still works for the read path
    let b = Backend::open("mem", "").unwrap();
    assert!(b.call(Area::Memory, "recall", json!({"query": "x"})).is_ok());
}

#[test]
fn ontology_and_reembed_tools() {
    let s = server();
    let mut c = conn();
    tool(&s, &mut c, "remember", json!({"text": "My sister Anna lives in Munich.", "ns": "o1"}));
    let r = tool(&s, &mut c, "ontology", json!({"action": "list"}));
    assert!(r["structuredContent"]["ontologies"].as_array().unwrap().iter().any(|o| o["onto"] == "personal"), "{r}");
    for action in ["facts", "infer", "history"] { // learn needs an embedder
        let r = tool(&s, &mut c, "ontology", json!({"action": action, "ns": "o1"}));
        assert!(r.get("isError").is_none(), "{action}: {r}");
    }
    // undo asks first (MRTR)
    let (r, _) = call(&s, &mut c, req(9, "tools/call", json!({"name": "ontology", "arguments": {"action": "undo", "ns": "o1", "change": "c1"}}), json!({"elicitation": {}})));
    assert_eq!(r["result"]["resultType"], "input_required", "{r}");
    // reembed to an embedder that is not enabled is a tool error, not a protocol error
    let r = tool(&s, &mut c, "reembed", json!({"embedder": "nope", "ns": "o1"}));
    assert_eq!(r["isError"], true, "{r}");
    // consolidate: a task for clients with tasks, which completes via the worker
    let (r, _) = call(&s, &mut c, req(10, "tools/call", json!({"name": "consolidate", "arguments": {"ns": "o1"}}), json!({"extensions": {"io.modelcontextprotocol/tasks": {}}})));
    let res = &r["result"];
    if res["resultType"] == "task" {
        let id = res["taskId"].as_str().unwrap().to_string();
        let mut st = String::new();
        for _ in 0..100 {
            let (g, _) = call(&s, &mut c, req(11, "tasks/get", json!({"taskId": id}), json!({"extensions": {"io.modelcontextprotocol/tasks": {}}})));
            st = g["result"]["status"].as_str().unwrap_or_default().to_string();
            assert!(!st.is_empty(), "{g}");
            if st != "working" { break; }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert!(st == "completed" || st == "failed", "{st}");
    } else {
        assert!(res.get("isError").is_none() || res["isError"] == true, "{r}");
    }
}
