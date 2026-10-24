/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Where the CLI and the MCP server send their calls (kleos fork;
//! docs/specs/mcp-cli-skills.md): an embedded store opened in this process,
//! or a running `kleos server` over HTTP. Both take the same JSON options as
//! `::memory`, `::doc`, `::flow` and `::ontology`.

use std::collections::BTreeMap;
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
}

impl Area {
    fn path(self) -> &'static str {
        match self {
            Area::Memory => "memory",
            Area::Doc => "doc",
            Area::Flow => "flow",
            Area::Ontology => "ontology",
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
        match self {
            Backend::Embedded(db) => {
                let r = match area {
                    Area::Memory => db.memory_call(op, args),
                    Area::Flow => db.flow_call(op, args),
                    Area::Ontology => db.ontology_call(op, args),
                    #[cfg(feature = "docs")]
                    Area::Doc => db.doc_call(op, args),
                    #[cfg(not(feature = "docs"))]
                    Area::Doc => {
                        let _ = args;
                        return Err("documents need kleos built with the `docs` feature".into());
                    }
                };
                r.map_err(|e| e.to_string())
            }
            Backend::Remote(r) => r.post(&format!("/{}/{op}", area.path()), &args),
        }
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

impl Remote {
    fn raw_post(&self, path: &str, body: &Value) -> Result<Value, String> {
        let mut req = minreq::post(format!("{}{path}", self.url))
            .with_header("Content-Type", "application/json")
            .with_timeout(self.timeout.as_secs())
            .with_body(body.to_string());
        if let Some(t) = &self.token {
            req = req.with_header("x-cozo-auth", t.as_str());
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

    fn post(&self, path: &str, body: &Value) -> Result<Value, String> {
        let v = self.raw_post(path, body)?;
        match v.get("ok") {
            Some(Value::Bool(true)) => Ok(v.get("result").cloned().unwrap_or(Value::Null)),
            _ => Err(v.get("message").and_then(|m| m.as_str()).map(|s| s.to_string()).unwrap_or_else(|| v.to_string())),
        }
    }
}
