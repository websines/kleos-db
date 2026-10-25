/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The skills-over-MCP extension (SEP-2640) and the skill tools on the
//! stateless MCP server (docs/specs/skills.md), against an in-memory store
//! seeded with the repository's own skills.

use kleos_bin::backend::{Area, Backend};
use kleos_bin::mcp::{Conn, McpServer, MODERN, SKILLS_EXT};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn server(mode: &str) -> McpServer {
    let b = Backend::open("mem", "").unwrap();
    let skills = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("skills");
    b.call(Area::Skill, "import", json!({"path": skills, "activate": true})).unwrap();
    McpServer::new(b, "default", false).with_state_key("k").with_skills_list(mode)
}

fn req(id: i64, method: &str, mut params: Value, caps: Value) -> Value {
    params["_meta"] = json!({"io.modelcontextprotocol/protocolVersion": MODERN, "io.modelcontextprotocol/clientCapabilities": caps});
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
}

fn call(s: &McpServer, method: &str, params: Value) -> (Value, u16) {
    let mut c = Conn { principal: "alice".into(), ..Default::default() };
    s.handle(&req(1, method, params, json!({})), &mut c).unwrap()
}

fn tool(s: &McpServer, name: &str, args: Value) -> Value {
    let (r, st) = call(s, "tools/call", json!({"name": name, "arguments": args}));
    assert_eq!(st, 200, "{r}");
    r["result"].clone()
}

#[test]
fn sep2640_listing_get_read_and_directories() {
    let s = server("top");
    let (r, _) = call(&s, "server/discover", json!({}));
    assert_eq!(r["result"]["capabilities"]["extensions"][SKILLS_EXT]["directoryRead"], true);
    assert!(r["result"]["capabilities"]["resources"].is_object());

    let (r, st) = call(&s, "skills/list", json!({}));
    assert_eq!(st, 200);
    let skills = r["result"]["skills"].as_array().unwrap();
    assert_eq!(skills.len(), 4);
    assert!(r["result"]["ttlMs"].as_u64().is_some());
    // every entry: verbatim frontmatter, complete resources, digests that match the bytes
    for e in skills {
        let uri = e["uri"].as_str().unwrap();
        assert!(uri.starts_with("skill://") && uri.ends_with("/SKILL.md"));
        assert_eq!(uri.trim_end_matches("/SKILL.md").rsplit('/').next().unwrap(), e["frontmatter"]["name"].as_str().unwrap());
        let (rd, _) = call(&s, "resources/read", json!({"uri": uri}));
        let text = rd["result"]["contents"][0]["text"].as_str().unwrap();
        let res = &e["resources"][0];
        assert_eq!(res["size"], text.len());
        assert_eq!(res["digest"].as_str().unwrap(), format!("sha256:{}", Sha256::digest(text.as_bytes()).iter().map(|b| format!("{b:02x}")).collect::<String>()));
        assert_eq!(rd["result"]["contents"][0]["mimeType"], "text/markdown");
    }
    // a task (kleos extension) gives a ranked, partial listing
    let (r, _) = call(&s, "skills/list", json!({"task": "fork memory before a risky change", "k": 1}));
    assert_eq!(r["result"]["skills"].as_array().unwrap().len(), 1);
    assert_eq!(r["result"]["skills"][0]["frontmatter"]["name"], "kleos-branches");

    let (r, _) = call(&s, "skills/get", json!({"uri": "skill://kleos-ontology/SKILL.md"}));
    assert_eq!(r["result"]["skill"]["frontmatter"]["name"], "kleos-ontology");
    let (r, st) = call(&s, "skills/get", json!({"uri": "skill://missing/SKILL.md"}));
    assert_eq!((r["error"]["code"].as_i64(), st), (Some(-32602), 400));

    let (r, _) = call(&s, "resources/directory/read", json!({"uri": "skill://kleos-memory"}));
    assert_eq!(r["result"]["resources"][0]["name"], "SKILL.md");
    let (r, _) = call(&s, "resources/directory/read", json!({"uri": "skill://kleos-memory/nope"}));
    assert_eq!(r["error"]["code"], -32602);

    let none = server("none");
    let (r, _) = call(&none, "skills/list", json!({}));
    assert_eq!(r["result"]["skills"].as_array().unwrap().len(), 0);
    let full = server("full");
    let (r, _) = call(&full, "skills/list", json!({}));
    assert_eq!(r["result"]["skills"].as_array().unwrap().len(), 4);
}

#[test]
fn skill_tools_find_get_outcome_propose_admin() {
    let s = server("top");
    let f = tool(&s, "find_skills", json!({"task": "store a PDF and cite pages", "k": 2}));
    assert_eq!(f["structuredContent"]["skills"][0]["name"], "kleos-documents");
    assert!(f["content"][0]["text"].as_str().unwrap().contains("route: sr_"));
    let route = f["structuredContent"]["route"].as_str().unwrap().to_string();

    let g = tool(&s, "get_skill", json!({"name": "kleos-documents"}));
    assert!(g["content"][0]["text"].as_str().unwrap().contains("kleos"));
    assert!(g["structuredContent"]["digest"].as_str().unwrap().starts_with("sha256:"));

    let o = tool(&s, "record_skill_outcome", json!({"name": "kleos-documents", "success": true, "route": route}));
    assert_eq!(o["structuredContent"]["success"], true);
    assert_eq!(o["structuredContent"]["router"]["n"], 1);

    let md = "---\nname: release-notes\ndescription: Write release notes from merged PRs.\n---\n1. list PRs\n2. group\n";
    let p = tool(&s, "propose_skill", json!({"skill_md": md, "activate": true}));
    assert_eq!(p["structuredContent"]["candidate"]["status"], "candidate");
    let f = tool(&s, "find_skills", json!({"task": "write release notes"}));
    assert!(f["structuredContent"]["skills"].as_array().unwrap().iter().all(|x| x["name"] != "release-notes"), "candidates are not served");

    // activation asks for confirmation, then serves the skill
    let a = tool(&s, "skill_admin", json!({"action": "activate", "name": "release-notes"}));
    assert_eq!(a["isError"], true);
    let a = tool(&s, "skill_admin", json!({"action": "activate", "name": "release-notes", "confirm": true}));
    assert!(a.get("isError").is_none(), "{a}");
    let f = tool(&s, "find_skills", json!({"task": "write release notes", "k": 1}));
    assert_eq!(f["structuredContent"]["skills"][0]["name"], "release-notes");
    let h = tool(&s, "skill_admin", json!({"action": "history", "name": "release-notes"}));
    assert_eq!(h["structuredContent"]["versions"].as_array().unwrap().len(), 1);

    // a read-only connection can find (without logging) but not propose
    let mut ro = Conn { principal: "bob".into(), read_only: true, ..Default::default() };
    let (r, _) = s.handle(&req(2, "tools/call", json!({"name": "find_skills", "arguments": {"task": "release notes"}}), json!({})), &mut ro).unwrap();
    assert!(r["result"].get("isError").is_none());
    let (r, _) = s.handle(&req(3, "tools/call", json!({"name": "propose_skill", "arguments": {"skill_md": md}}), json!({})), &mut ro).unwrap();
    assert_eq!(r["result"]["isError"], true);
}
