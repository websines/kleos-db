/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The `kleos` CLI and `kleos mcp` (stdio) end to end against the real binary
//! (docs/specs/mcp-cli-skills.md).

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::{json, Value};

const BIN: &str = env!("CARGO_BIN_EXE_kleos-bin");

fn dir(name: &str) -> PathBuf {
    let d = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("kleos-cli-{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Runs the CLI with an empty config; returns (exit code, stdout, stderr).
fn kleos(d: &PathBuf, args: &[&str]) -> (i32, String, String) {
    let out = Command::new(BIN)
        .args(args)
        .env("KLEOS_CONFIG", d.join("no-config.json"))
        .stdin(Stdio::null())
        .output()
        .unwrap();
    (out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned())
}

fn json_out(s: &str) -> Value {
    serde_json::from_str(s.trim()).unwrap_or_else(|e| panic!("not JSON ({e}): {s}"))
}

#[test]
fn cli_end_to_end() {
    let d = dir("e2e");
    let db = d.join("m.db");
    let db = db.to_str().unwrap();
    let common = ["--db", db, "--ns", "proj"];
    let with = |extra: &[&str]| -> Vec<String> { extra.iter().map(|s| s.to_string()).chain(common.iter().map(|s| s.to_string())).collect() };
    let run = |extra: &[&str]| {
        let a = with(extra);
        let a: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
        kleos(&d, &a)
    };

    let (code, out, err) = run(&["remember", "We chose Postgres for the billing service."]);
    assert_eq!(code, 0, "{err}");
    assert!(out.starts_with("remembered m_"), "{out}");
    let (code, out, _) = run(&["remember", "The billing launch is on March 3.", "--json"]);
    assert_eq!(code, 0);
    let v = json_out(&out);
    assert_eq!(v["ok"], true);
    let launch_id = v["result"]["id"].as_str().unwrap().to_string();

    // human recall prints the context block; --list prints hits; --json both
    let (code, out, _) = run(&["recall", "billing service"]);
    assert_eq!(code, 0);
    assert!(out.contains("Postgres"), "{out}");
    let (_, out, _) = run(&["recall", "billing launch", "--list"]);
    assert!(out.contains(&launch_id), "{out}");
    let (_, out, _) = run(&["recall", "billing launch", "--json"]);
    assert!(json_out(&out)["result"]["text"].as_str().unwrap().contains("March 3"));

    // destructive commands refuse without a terminal or --yes
    let (code, _, err) = run(&["forget", &launch_id]);
    assert_eq!(code, 1);
    assert!(err.contains("--yes"), "{err}");
    let (code, _, err) = run(&["forget", &launch_id, "--yes"]);
    assert_eq!(code, 0, "{err}");
    let (_, out, _) = run(&["recall", "billing launch", "--list"]);
    assert!(!out.contains(&launch_id), "{out}");

    // profile, reflect, query, branches, flows, ontology
    let (code, out, _) = run(&["profile", "Postgres"]);
    assert_eq!(code, 0);
    assert!(out.contains("Postgres"), "{out}");
    let (code, out, _) = run(&["reflect", "--json"]);
    assert_eq!(code, 0);
    assert_eq!(json_out(&out)["result"]["ns"], "proj");
    let (code, out, _) = run(&["query", "?[x] := x = 40 + 2", "--json"]);
    assert_eq!(code, 0);
    assert_eq!(json_out(&out)["result"]["rows"], json!([[42]]));
    let (code, _, err) = run(&["query", ":create t {a}"]);
    assert_eq!(code, 1, "query must be read-only");
    assert!(!err.is_empty());
    let (code, _, err) = run(&["branch", "fork", "proj-exp"]);
    assert_eq!(code, 0, "{err}");
    let (code, out, _) = run(&["branch", "list", "--json"]);
    assert_eq!(code, 0);
    assert!(out.contains("proj-exp"), "{out}");
    let (code, out, _) = run(&["flow", "start", "demo.job", "--input", "{\"n\":1}", "--json"]);
    assert_eq!(code, 0);
    let fid = json_out(&out)["result"]["id"].as_str().unwrap().to_string();
    let (code, out, _) = run(&["flow", "status", &fid, "--json"]);
    assert_eq!(code, 0);
    assert_eq!(json_out(&out)["result"]["id"], fid.as_str());
    let (code, _, err) = run(&["flow", "start", "x", "--input", "{bad"]);
    assert_eq!(code, 1);
    assert!(err.contains("--input"), "{err}");
    let (code, out, _) = run(&["ontology", "list", "--json"]);
    assert_eq!(code, 0);
    assert!(out.contains("personal"));

    // --json errors are JSON on stdout
    let (code, out, _) = run(&["doc", "status", "nope", "--json"]);
    assert_eq!(code, 1);
    assert_eq!(json_out(&out)["ok"], false);
}

#[test]
fn config_file_and_helpful_errors() {
    let d = dir("cfg");
    let (code, _, err) = kleos(&d, &["recall", "x"]);
    assert_eq!(code, 1);
    assert!(err.contains("--db"), "{err}");
    let cfg = d.join("config.json");
    std::fs::write(&cfg, json!({"db": d.join("c.db").to_str().unwrap(), "ns": "fromcfg"}).to_string()).unwrap();
    let out = Command::new(BIN).args(["remember", "Configured stores work.", "--json"]).env("KLEOS_CONFIG", &cfg).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(json_out(&String::from_utf8_lossy(&out.stdout))["result"]["ns"], "fromcfg");
    std::fs::write(&cfg, "{\"bogus\": 1}").unwrap();
    let out = Command::new(BIN).args(["recall", "x"]).env("KLEOS_CONFIG", &cfg).output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("bogus"));
    let (code, _, err) = kleos(&d, &["recall", "x", "--url", "http://127.0.0.1:9"]);
    assert_eq!(code, 1);
    assert!(err.contains("cannot reach"), "{err}");
}

#[test]
fn mcp_over_stdio() {
    let d = dir("stdio");
    let db = d.join("s.db");
    let mut child = Command::new(BIN)
        .args(["mcp", "--db", db.to_str().unwrap(), "--ns", "s"])
        .env("KLEOS_CONFIG", d.join("no-config.json"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let meta = json!({"io.modelcontextprotocol/protocolVersion": "2026-07-28", "io.modelcontextprotocol/clientCapabilities": {}});
    let mut send = |msg: Value| -> Option<Value> {
        writeln!(stdin, "{msg}").unwrap();
        stdin.flush().unwrap();
        if msg.get("id").is_none() {
            return None;
        }
        let mut line = String::new();
        stdout.read_line(&mut line).unwrap();
        Some(serde_json::from_str(&line).unwrap())
    };
    let r = send(json!({"jsonrpc": "2.0", "id": 1, "method": "server/discover", "params": {"_meta": meta}})).unwrap();
    assert_eq!(r["result"]["supportedVersions"], json!(["2026-07-28"]));
    // a notification gets no reply; the next request still lines up
    assert!(send(json!({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 9}})).is_none());
    let r = send(json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "remember", "arguments": {"text": "Stdio works too."}, "_meta": meta}})).unwrap();
    assert!(r["result"]["structuredContent"]["id"].is_string(), "{r}");
    let r = send(json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "recall", "arguments": {"query": "stdio"}, "_meta": meta}})).unwrap();
    assert!(r["result"]["content"][0]["text"].as_str().unwrap().contains("Stdio works"), "{r}");
    writeln!(stdin, "this is not json").unwrap();
    let mut line = String::new();
    stdout.read_line(&mut line).unwrap();
    assert_eq!(serde_json::from_str::<Value>(&line).unwrap()["error"]["code"], -32700);
    drop(stdin);
    assert!(child.wait().unwrap().success());
    // the same store from the CLI
    let (code, out, _) = kleos(&d, &["recall", "stdio", "--db", db.to_str().unwrap(), "--ns", "s"]);
    assert_eq!(code, 0);
    assert!(out.contains("Stdio works"), "{out}");
}
