/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! `kleos server --mcp`: the stateless Streamable HTTP endpoint (protocol
//! 2026-07-28) end to end against the real binary.

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const BIN: &str = env!("CARGO_BIN_EXE_kleos-bin");

struct Server {
    child: Child,
    port: u16,
    token: Option<String>,
    _dir: std::path::PathBuf,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn start(auth: bool) -> Server {
    let port = free_port();
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("kleos-mcp-http-{port}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("store");
    let mut cmd = Command::new(BIN);
    cmd.args(["server", "--mcp", "--mcp-allow-query", "-e", "mem", "-P", &port.to_string(), "-p"])
        .arg(&path)
        .arg("--backup-dir")
        .arg(dir.join("backups"))
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if !auth {
        cmd.arg("--insecure-no-auth");
    }
    let child = cmd.spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    while std::net::TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(Instant::now() < deadline, "server did not start");
        std::thread::sleep(Duration::from_millis(50));
    }
    let token = auth.then(|| std::fs::read_to_string(format!("{}.mem.cozo_auth", path.display())).unwrap().trim().to_string());
    Server { child, port, token, _dir: dir }
}

fn meta() -> Value {
    json!({"io.modelcontextprotocol/protocolVersion": "2026-07-28", "io.modelcontextprotocol/clientCapabilities": {}})
}

fn post(s: &Server, headers: &[(&str, &str)], body: &Value) -> (i32, String) {
    let mut r = minreq::post(format!("http://127.0.0.1:{}/mcp", s.port))
        .with_header("Content-Type", "application/json")
        .with_header("Accept", "application/json, text/event-stream")
        .with_body(body.to_string())
        .with_timeout(30);
    for (k, v) in headers {
        r = r.with_header(*k, *v);
    }
    let resp = r.send().unwrap();
    (resp.status_code, resp.as_str().unwrap_or_default().to_string())
}

fn modern(s: &Server, method: &str, name: Option<&str>, params: Value, extra: &[(&str, &str)]) -> (i32, Value) {
    let mut p = params;
    p["_meta"] = meta();
    let body = json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": p});
    let mut h: Vec<(&str, &str)> = vec![("MCP-Protocol-Version", "2026-07-28"), ("Mcp-Method", method)];
    if let Some(n) = name {
        h.push(("Mcp-Name", n));
    }
    h.extend_from_slice(extra);
    let (st, text) = post(s, &h, &body);
    (st, serde_json::from_str(&text).unwrap_or(Value::Null))
}

#[test]
fn stateless_http_endpoint() {
    let s = start(false);
    let (st, r) = modern(&s, "server/discover", None, json!({}), &[]);
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["result"]["supportedVersions"], json!(["2026-07-28"]));

    let (st, r) = modern(&s, "tools/call", Some("remember"), json!({"name": "remember", "arguments": {"text": "HTTP clients can remember things.", "ns": "h"}}), &[]);
    assert_eq!(st, 200, "{r}");
    assert!(r["result"]["structuredContent"]["id"].is_string(), "{r}");
    let (st, r) = modern(&s, "tools/call", Some("recall"), json!({"name": "recall", "arguments": {"query": "remember things", "ns": "h"}}), &[]);
    assert_eq!(st, 200);
    assert!(r["result"]["content"][0]["text"].as_str().unwrap().contains("HTTP clients"));

    // the base64 sentinel form of Mcp-Name
    let (st, _) = modern(&s, "tools/call", Some("=?base64?cmVjYWxs?="), json!({"name": "recall", "arguments": {"query": "x", "ns": "h"}}), &[]);
    assert_eq!(st, 200);

    // header validation → 400 HeaderMismatch (-32020)
    let body = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {"_meta": meta()}});
    let (st, text) = post(&s, &[("Mcp-Method", "tools/list")], &body);
    assert_eq!(st, 400);
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["error"]["code"], -32020);
    let (st, text) = post(&s, &[("MCP-Protocol-Version", "2026-07-28"), ("Mcp-Method", "tools/call")], &body);
    assert_eq!(st, 400, "{text}");
    let (st, r) = modern(&s, "tools/call", Some("forget"), json!({"name": "recall", "arguments": {"query": "x"}}), &[]);
    assert_eq!((st, r["error"]["code"].as_i64()), (400, Some(-32020)));
    let (st, r) = modern(&s, "tools/call", None, json!({"name": "recall", "arguments": {"query": "x"}}), &[]);
    assert_eq!((st, r["error"]["code"].as_i64()), (400, Some(-32020)));
    let mut wrongv = json!({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {"_meta": meta()}});
    wrongv["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"] = json!("2026-07-28");
    let (st, _) = post(&s, &[("MCP-Protocol-Version", "2025-11-25"), ("Mcp-Method", "tools/list")], &wrongv);
    assert_eq!(st, 400);

    // unknown method → 404 with -32601; notifications → 202; GET/DELETE → 405
    let (st, r) = modern(&s, "bogus/method", None, json!({}), &[]);
    assert_eq!((st, r["error"]["code"].as_i64()), (404, Some(-32601)));
    let (st, text) = post(&s, &[], &json!({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}}));
    assert_eq!((st, text.as_str()), (202, ""));
    let get = minreq::get(format!("http://127.0.0.1:{}/mcp", s.port)).send().unwrap();
    assert_eq!(get.status_code, 405);

    // a foreign Origin → 403; a loopback origin is fine
    let body = json!({"jsonrpc": "2.0", "id": 4, "method": "tools/list", "params": {"_meta": meta()}});
    let (st, _) = post(&s, &[("MCP-Protocol-Version", "2026-07-28"), ("Mcp-Method", "tools/list"), ("Origin", "https://evil.example")], &body);
    assert_eq!(st, 403);
    // without auth only the server's own origin passes (the server's own
    // no-auth guard); with auth any loopback origin may (see below)
    let own = format!("http://127.0.0.1:{}", s.port);
    let (st, _) = post(&s, &[("MCP-Protocol-Version", "2026-07-28"), ("Mcp-Method", "tools/list"), ("Origin", own.as_str())], &body);
    assert_eq!(st, 200);

    // session headers are ignored, never minted
    let resp = minreq::post(format!("http://127.0.0.1:{}/mcp", s.port))
        .with_header("Content-Type", "application/json")
        .with_header("MCP-Protocol-Version", "2026-07-28")
        .with_header("Mcp-Method", "tools/list")
        .with_header("Mcp-Session-Id", "abc")
        .with_body(body.to_string())
        .send()
        .unwrap();
    assert_eq!(resp.status_code, 200);
    assert!(resp.headers.get("mcp-session-id").is_none());

    // an earlier-revision client: initialize, then requests named by header
    let (st, text) = post(&s, &[], &json!({"jsonrpc": "2.0", "id": 5, "method": "initialize", "params": {"protocolVersion": "2025-11-25", "capabilities": {}}}));
    assert_eq!(st, 200);
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["result"]["protocolVersion"], "2025-11-25");
    let (st, text) = post(&s, &[("MCP-Protocol-Version", "2025-11-25")], &json!({"jsonrpc": "2.0", "id": 6, "method": "tools/list"}));
    assert_eq!(st, 200, "{text}");
    assert!(serde_json::from_str::<Value>(&text).unwrap()["result"]["tools"].is_array());
}

#[test]
fn authenticated_endpoint() {
    let s = start(true);
    let token = s.token.clone().unwrap();
    let (st, _) = modern(&s, "tools/list", None, json!({}), &[]);
    assert_eq!(st, 401);
    let bearer = format!("Bearer {token}");
    let (st, r) = modern(&s, "tools/list", None, json!({}), &[("Authorization", &bearer)]);
    assert_eq!(st, 200, "{r}");
    let (st, _) = modern(&s, "tools/list", None, json!({}), &[("Authorization", "Bearer wrong")]);
    assert_eq!(st, 401);
    let (st, _) = modern(&s, "tools/list", None, json!({}), &[("x-cozo-auth", &token)]);
    assert_eq!(st, 200);
    let (st, _) = modern(&s, "tools/list", None, json!({}), &[("Authorization", &bearer), ("Origin", "http://localhost:5173")]);
    assert_eq!(st, 200);
    let (st, _) = modern(&s, "tools/list", None, json!({}), &[("Authorization", &bearer), ("Origin", "https://evil.example")]);
    assert_eq!(st, 403);
}
