/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Where the CLI and the MCP server send their calls (kleos fork;
//! docs/specs/mcp-cli-skills.md): an embedded store opened in this process,
//! or a running `kleos server` over HTTP. Both take the same JSON options as
//! `::memory`, `::doc`, `::flow`, `::ontology` and `::skill`.

use std::collections::BTreeMap;
use std::io::BufRead;
use std::sync::mpsc;
use std::time::Duration;

use kleos::{DataValue, DbInstance, ScriptMutability};
use serde_json::{json, Value};

/// The op families a call can address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Area {
    Memory,
    Doc,
    Flow,
    Ontology,
    /// The Postgres connector (feature `pg-cdc`).
    Pg,
    Skill,
}

impl Area {
    fn path(self) -> &'static str {
        match self {
            Area::Memory => "memory",
            Area::Doc => "doc",
            Area::Flow => "flow",
            Area::Ontology => "ontology",
            Area::Pg => "pg",
            Area::Skill => "skill",
        }
    }
}

/// A kleos server reached over HTTP.
#[derive(Clone, Debug)]
pub struct Remote {
    pub url: String,
    pub token: Option<String>,
    pub timeout: Duration,
}

/// An embedded store or a remote server.
#[derive(Clone)]
pub enum Backend {
    Embedded(DbInstance),
    Remote(Remote),
}

impl Backend {
    /// Opens an embedded store (`sqlite` unless `engine` says otherwise; `mem`
    /// for a throwaway store).
    pub fn open(engine: &str, path: &str) -> Result<Self, String> {
        DbInstance::new(engine, path, "{}")
            .map(Backend::Embedded)
            .map_err(|e| format!("cannot open the {engine} store at '{path}': {e}"))
    }

    pub fn remote(url: &str, token: Option<String>) -> Self {
        Backend::Remote(Remote {
            url: url.trim_end_matches('/').to_string(),
            token,
            timeout: Duration::from_secs(600),
        })
    }

    pub fn is_remote(&self) -> bool {
        matches!(self, Backend::Remote(_))
    }

    /// One op, e.g. `call(Area::Memory, "recall", {"query": ...})`.
    pub fn call(&self, area: Area, op: &str, args: Value) -> Result<Value, String> {
        let args2 = args.clone();
        match self {
            Backend::Embedded(db) => {
                let r = match area {
                    Area::Memory => db.memory_call(op, args),
                    Area::Flow => db.flow_call(op, args),
                    Area::Ontology => db.ontology_call(op, args),
                    Area::Skill => db.skill_call(op, args),
                    #[cfg(feature = "docs")]
                    Area::Doc => db.doc_call(op, args),
                    #[cfg(not(feature = "docs"))]
                    Area::Doc => {
                        let _ = args;
                        return Err("documents need kleos built with the `docs` feature".into());
                    }
                    #[cfg(feature = "pg-cdc")]
                    Area::Pg => kleos_pg::pg_call(db, op, args),
                    #[cfg(not(feature = "pg-cdc"))]
                    Area::Pg => {
                        let _ = args;
                        return Err("the Postgres connector needs kleos built with the `pg-cdc` feature".into());
                    }
                };
                r.map_err(|e| e.to_string())
            }
            Backend::Remote(r) => r.post(&format!("/{}/{op}", area.path()), &args),
        }
        .or_else(|e| empty_read(area, op, &args2, e))
    }

    /// A read-only CozoScript query; answers `{headers, rows}`.
    pub fn query(&self, script: &str, params: &Value) -> Result<Value, String> {
        match self {
            Backend::Embedded(db) => {
                let params: BTreeMap<String, DataValue> = match params {
                    Value::Object(m) => m.iter().map(|(k, v)| (k.clone(), DataValue::from(v.clone()))).collect(),
                    Value::Null => BTreeMap::new(),
                    other => return Err(format!("query params must be an object, got {other}")),
                };
                db.run_script(script, params, ScriptMutability::Immutable)
                    .map(|rows| rows.into_json())
                    .map_err(|e| e.to_string())
            }
            Backend::Remote(r) => {
                let body = json!({"script": script, "params": params.as_object().cloned().unwrap_or_default(), "immutable": true});
                let v = r.raw_post("/text-query", &body)?;
                if v.get("ok") == Some(&Value::Bool(false)) {
                    return Err(v.get("display").or(v.get("message")).map(|m| m.as_str().unwrap_or_default().to_string()).unwrap_or_else(|| v.to_string()));
                }
                Ok(v)
            }
        }
    }
}

/// Writes to stored relations as they commit: each message is the relation
/// and the rows written or removed (as JSON arrays). Dropping the watch stops
/// it (embedded: the callbacks are unregistered; remote: the reader thread
/// ends at its next event or keep-alive).
pub struct Watch {
    pub rx: mpsc::Receiver<(String, Vec<Value>)>,
    db: Option<DbInstance>,
    ids: Vec<u32>,
}

impl Drop for Watch {
    fn drop(&mut self) {
        if let Some(db) = &self.db {
            for id in &self.ids {
                db.unregister_callback(*id);
            }
        }
    }
}

impl Backend {
    /// Watch `relations` (embedded: change callbacks; remote: the server's
    /// `GET /changes/:relation` streams, which need the admin secret).
    pub fn watch(&self, relations: &[&str]) -> Result<Watch, String> {
        let (tx, rx) = mpsc::channel();
        match self {
            Backend::Embedded(db) => {
                let mut ids = vec![];
                for rel in relations {
                    let (id, recv) = db.register_callback(rel, None);
                    ids.push(id);
                    let tx = tx.clone();
                    let rel = rel.to_string();
                    // ends when the callback is unregistered (the sender drops)
                    std::thread::spawn(move || {
                        for (_op, new, old) in recv {
                            let rows: Vec<Value> = new.rows.into_iter().chain(old.rows).map(|r| Value::Array(r.into_iter().map(Value::from).collect())).collect();
                            if tx.send((rel.clone(), rows)).is_err() {
                                break;
                            }
                        }
                    });
                }
                Ok(Watch { rx, db: Some(db.clone()), ids })
            }
            Backend::Remote(r) => {
                for rel in relations {
                    let lines = r.sse(&format!("/changes/{rel}"))?;
                    let tx = tx.clone();
                    let rel = rel.to_string();
                    std::thread::spawn(move || {
                        for line in lines {
                            let Some(data) = line.strip_prefix("data:") else { continue };
                            let Ok(v) = serde_json::from_str::<Value>(data.trim()) else { continue };
                            let mut rows = vec![];
                            for k in ["new_rows", "old_rows"] {
                                if let Some(a) = v.get(k).and_then(|x| x.get("rows")).and_then(|x| x.as_array()) {
                                    rows.extend(a.iter().cloned());
                                }
                            }
                            if tx.send((rel.clone(), rows)).is_err() {
                                break;
                            }
                        }
                    });
                }
                Ok(Watch { rx, db: None, ids: vec![] })
            }
        }
    }
}

/// A read of a store with no memories yet (its relations are created by the
/// first `remember`) is an empty answer, not an error.
fn empty_read(area: Area, op: &str, args: &Value, err: String) -> Result<Value, String> {
    if area != Area::Memory || !err.contains("Cannot find requested stored relation 'kleos_mem") {
        return Err(err);
    }
    let ns = args.get("ns").and_then(|v| v.as_str()).unwrap_or("default");
    match op {
        "recall" if args.get("format").and_then(|f| f.as_str()) == Some("context") => {
            Ok(json!({"text": "", "tokens": 0, "budget": args.get("budget").cloned().unwrap_or(json!(1000)), "ids": [], "omitted": []}))
        }
        "recall" | "branches" => Ok(json!([])),
        "reflect" => Ok(json!({"ns": ns, "pending": [], "changes": [], "contradictions": []})),
        _ => Err(err),
    }
}

impl Remote {
    fn auth(&self, mut req: minreq::Request) -> minreq::Request {
        if let Some(t) = &self.token {
            if t.starts_with("kl_") || t.matches('.').count() == 2 {
                req = req.with_header("Authorization", format!("Bearer {t}"));
            } else {
                req = req.with_header("x-cozo-auth", t.as_str());
            }
        }
        req
    }

    /// The lines of a server-sent-events stream (`GET path`), read lazily.
    pub fn sse(&self, path: &str) -> Result<impl Iterator<Item = String>, String> {
        let req = self.auth(minreq::get(format!("{}{path}", self.url)).with_header("Accept", "text/event-stream"));
        let resp = req.send_lazy().map_err(|e| format!("cannot reach the kleos server at {}: {e}", self.url))?;
        if resp.status_code / 100 != 2 {
            return Err(format!("HTTP {} from {path}", resp.status_code));
        }
        let bytes = resp.map_while(|b| b.ok().map(|(byte, _)| byte));
        Ok(std::io::BufReader::new(ByteReader(Box::new(bytes))).lines().map_while(|l| l.ok()))
    }

    fn raw_post(&self, path: &str, body: &Value) -> Result<Value, String> {
        let mut req = minreq::post(format!("{}{path}", self.url))
            .with_header("Content-Type", "application/json")
            .with_timeout(self.timeout.as_secs())
            .with_body(body.to_string());
        if let Some(t) = &self.token {
            // a tenancy credential (API key or JWT access token) goes as a
            // bearer token; the server's own secret as before
            if t.starts_with("kl_") || t.matches('.').count() == 2 {
                req = req.with_header("Authorization", format!("Bearer {t}"));
            } else {
                req = req.with_header("x-cozo-auth", t.as_str());
            }
        }
        let resp = req
            .send()
            .map_err(|e| format!("cannot reach the kleos server at {}: {e}", self.url))?;
        let text = resp.as_str().map_err(|e| e.to_string())?;
        match resp.status_code {
            401 | 403 if text.trim().is_empty() => {
                Err(format!("the kleos server at {} refused the request (HTTP {}): pass --token or set \"token\" in the config", self.url, resp.status_code))
            }
            _ => serde_json::from_str(text).map_err(|_| format!("HTTP {} from {path}: {}", resp.status_code, text.trim())),
        }
    }

    /// A tenancy admin op (`POST /tenant/:op`).
    pub fn tenant(&self, op: &str, body: &Value) -> Result<Value, String> {
        self.post(&format!("/tenant/{op}"), body)
    }

    fn post(&self, path: &str, body: &Value) -> Result<Value, String> {
        let v = self.raw_post(path, body)?;
        match v.get("ok") {
            Some(Value::Bool(true)) => Ok(v.get("result").cloned().unwrap_or(Value::Null)),
            _ => Err(v.get("message").and_then(|m| m.as_str()).map(|s| s.to_string()).unwrap_or_else(|| v.to_string())),
        }
    }
}

/// `Read` over a byte iterator (a lazily read HTTP body).
struct ByteReader(Box<dyn Iterator<Item = u8> + Send>);

impl std::io::Read for ByteReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut n = 0;
        while n < buf.len() {
            match self.0.next() {
                Some(b) => {
                    buf[n] = b;
                    n += 1;
                    if b == b'\n' {
                        break;
                    }
                }
                None => break,
            }
        }
        Ok(n)
    }
}
