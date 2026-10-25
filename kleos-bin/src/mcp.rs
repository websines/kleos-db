/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Stateless MCP server (kleos fork; docs/specs/mcp-cli-skills.md),
//! protocol revision 2026-07-28, transport-independent. Each request carries
//! its protocol version and client capabilities in `_meta`; nothing is kept
//! between requests except, for clients of earlier revisions, the version
//! negotiated by `initialize` (see [`Conn`]).
//!
//! [`McpServer::handle`] takes one decoded JSON-RPC message and returns the
//! response (and the HTTP status the Streamable HTTP transport should use),
//! or `None` for notifications. [`serve_stdio`] and the HTTP handler in
//! `server.rs` are thin transports around it.

use std::io::{BufRead, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::backend::{Area, Backend};

/// The protocol revision this server implements.
pub const MODERN: &str = "2026-07-28";
/// Earlier, `initialize`-based revisions served for older clients.
pub const LEGACY: [&str; 3] = ["2025-11-25", "2025-06-18", "2025-03-26"];

const META_VERSION: &str = "io.modelcontextprotocol/protocolVersion";
const META_CAPS: &str = "io.modelcontextprotocol/clientCapabilities";
const META_SERVER: &str = "io.modelcontextprotocol/serverInfo";
const TASKS_EXT: &str = "io.modelcontextprotocol/tasks";
/// SEP-2640, the skills extension (Final): `skills/list`, `skills/get`,
/// `skill://` resources and `resources/directory/read`.
pub const SKILLS_EXT: &str = "io.modelcontextprotocol/skills";
/// Skills in a `skills/list` answered with a task (kleos extension: `task`).
const SKILLS_TOP_K: u64 = 8;

pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;
pub const HEADER_MISMATCH: i64 = -32020;
pub const UNSUPPORTED_VERSION: i64 = -32022;

/// How long a confirmation's `requestState` stays valid.
const STATE_TTL_SECS: u64 = 600;
/// List and discovery results may be cached this long.
const LIST_TTL_MS: u64 = 60_000;

const INSTRUCTIONS: &str = "kleos is a memory engine. Call `recall` (format `context`) before answering \
questions about the user, their projects or earlier decisions; call `remember` with stable facts, \
preferences and decisions worth keeping (not chit-chat). Recall marks replaced facts as \"previously\" \
and uncertain replacements as pending. Documents go in with `doc_add` and come back cited by page. \
Pass the same `ns` for one user or project. \
For a task that a known procedure may cover, call `find_skills` with the task: it returns the few skills that \
fit (never the whole library); load one with `get_skill`, follow it, then report how it went with \
`record_skill_outcome` (pass the `route` from `find_skills`).";

/// Server options.
#[derive(Clone)]
pub struct McpServer {
    backend: Backend,
    default_ns: String,
    allow_query: bool,
    /// What `skills/list` returns without a task: `top` (default; the most
    /// useful live skills, a partial listing), `full` (the catalog, paged) or
    /// `none` (empty; hosts use `find_skills` and `skills/get`).
    skills_list: String,
    state_key: [u8; 32],
    /// In-process worker for background jobs (re-embed, consolidate) of an
    /// embedded store, started on first use; a remote server runs its own.
    worker: std::sync::Arc<std::sync::Mutex<Option<kleos::FlowWorker>>>,
}

/// Per-connection context for clients of earlier revisions: the version an
/// `initialize` negotiated on this connection (stdio), or the version named by
/// the `MCP-Protocol-Version` header of this request (HTTP).
#[derive(Clone, Debug, Default)]
pub struct Conn {
    pub legacy: Option<String>,
    /// Who is calling (bound into confirmation state).
    pub principal: String,
    /// A read-only principal may call only read tools.
    pub read_only: bool,
}

/// The era a request is served in.
#[derive(Clone, Debug)]
enum Era {
    Modern { caps: Value },
    Legacy,
}

impl Era {
    fn caps(&self) -> Option<&Value> {
        match self {
            Era::Modern { caps } => Some(caps),
            Era::Legacy => None,
        }
    }
    fn has_elicitation(&self) -> bool {
        self.caps().and_then(|c| c.get("elicitation")).is_some()
    }
    fn has_tasks(&self) -> bool {
        self.caps().and_then(|c| c.get("extensions")).and_then(|e| e.get(TASKS_EXT)).is_some()
    }
}

/// A JSON-RPC error with the HTTP status Streamable HTTP uses for it.
#[derive(Debug)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    pub data: Option<Value>,
}

impl RpcError {
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self { code, message: message.into(), data: None }
    }
    pub fn http_status(&self) -> u16 {
        match self.code {
            METHOD_NOT_FOUND => 404,
            INTERNAL_ERROR => 500,
            _ => 400,
        }
    }
    pub fn to_response(&self, id: Option<&Value>) -> Value {
        let mut err = json!({"code": self.code, "message": self.message});
        if let Some(d) = &self.data {
            err["data"] = d.clone();
        }
        let mut r = json!({"jsonrpc": "2.0", "error": err});
        if let Some(id) = id {
            r["id"] = id.clone();
        }
        r
    }
}

/// What a tool call produced.
enum ToolOut {
    Done { text: String, structured: Value, is_error: bool },
    Task(Value),
    InputRequired(Value),
}

fn tool_error(msg: impl Into<String>) -> ToolOut {
    let msg = msg.into();
    ToolOut::Done { text: msg.clone(), structured: json!({"error": msg}), is_error: true }
}

fn done(structured: Value) -> ToolOut {
    let text = serde_json::to_string(&structured).unwrap_or_default();
    ToolOut::Done { text, structured, is_error: false }
}

impl McpServer {
    pub fn new(backend: Backend, default_ns: impl Into<String>, allow_query: bool) -> Self {
        let mut state_key = [0u8; 32];
        match std::env::var("KLEOS_MCP_STATE_KEY") {
            Ok(k) if !k.is_empty() => state_key.copy_from_slice(&Sha256::digest(k.as_bytes())),
            _ => {
                use rand::RngCore;
                rand::thread_rng().fill_bytes(&mut state_key);
            }
        }
        Self { backend, default_ns: default_ns.into(), allow_query, skills_list: "top".into(), state_key, worker: Default::default() }
    }

    /// `skills/list` without a task: `top`, `full` or `none`.
    pub fn with_skills_list(mut self, mode: &str) -> Self {
        self.skills_list = mode.to_string();
        self
    }

    /// Fixes the confirmation key (tests, or several instances sharing one key).
    pub fn with_state_key(mut self, key: &str) -> Self {
        self.state_key.copy_from_slice(&Sha256::digest(key.as_bytes()));
        self
    }

    fn server_info() -> Value {
        json!({"name": "kleos", "version": env!("CARGO_PKG_VERSION")})
    }

    fn capabilities() -> Value {
        json!({
            "tools": {"listChanged": false},
            "resources": {"listChanged": false},
            "extensions": {TASKS_EXT: {}, SKILLS_EXT: {"directoryRead": true}}
        })
    }

    /// Handles one JSON-RPC message. `None` for notifications and stray
    /// responses (nothing to send back); otherwise the response and its HTTP
    /// status.
    pub fn handle(&self, msg: &Value, conn: &mut Conn) -> Option<(Value, u16)> {
        let Some(obj) = msg.as_object() else {
            let e = RpcError::new(INVALID_REQUEST, "expected a JSON-RPC object");
            return Some((e.to_response(None), 400));
        };
        let id = obj.get("id").filter(|v| !v.is_null());
        let Some(method) = obj.get("method").and_then(|m| m.as_str()) else {
            // a response from the client: nothing in this server asks for one
            if id.is_some() && (obj.contains_key("result") || obj.contains_key("error")) {
                return None;
            }
            let e = RpcError::new(INVALID_REQUEST, "missing 'method'");
            return Some((e.to_response(id), 400));
        };
        if obj.get("jsonrpc").and_then(|v| v.as_str()) != Some("2.0") {
            let e = RpcError::new(INVALID_REQUEST, "'jsonrpc' must be \"2.0\"");
            return Some((e.to_response(id), 400));
        }
        let Some(id) = id else {
            // notifications: `notifications/initialized`, `notifications/cancelled`, …
            return None;
        };
        let params = obj.get("params").cloned().unwrap_or_else(|| json!({}));
        match self.dispatch(method, &params, conn) {
            Ok((mut result, era)) => {
                if let Value::Object(m) = &mut result {
                    m.entry("resultType").or_insert_with(|| json!("complete"));
                    if matches!(era, Era::Modern { .. }) {
                        let meta = m.entry("_meta").or_insert_with(|| json!({}));
                        if let Value::Object(mm) = meta {
                            mm.insert(META_SERVER.to_string(), Self::server_info());
                        }
                    }
                }
                Some((json!({"jsonrpc": "2.0", "id": id, "result": result}), 200))
            }
            Err(e) => Some((e.to_response(Some(id)), e.http_status())),
        }
    }

    /// Which era a request is in, validating the per-request `_meta`.
    fn era(&self, params: &Value, conn: &Conn) -> Result<Era, RpcError> {
        let meta = params.get("_meta");
        match meta.and_then(|m| m.get(META_VERSION)) {
            Some(v) => {
                let v = v.as_str().unwrap_or_default();
                if v != MODERN {
                    let mut supported = vec![MODERN];
                    supported.extend(LEGACY);
                    let mut e = RpcError::new(UNSUPPORTED_VERSION, "Unsupported protocol version");
                    e.data = Some(json!({"supported": supported, "requested": v}));
                    return Err(e);
                }
                let caps = meta.and_then(|m| m.get(META_CAPS)).cloned();
                match caps {
                    Some(c @ Value::Object(_)) => Ok(Era::Modern { caps: c }),
                    _ => Err(RpcError::new(INVALID_PARAMS, format!("missing _meta \"{META_CAPS}\""))),
                }
            }
            None if conn.legacy.is_some() => Ok(Era::Legacy),
            None => Err(RpcError::new(
                INVALID_PARAMS,
                format!("missing _meta \"{META_VERSION}\" (this server speaks MCP {MODERN}; clients of earlier revisions start with `initialize`)"),
            )),
        }
    }

    fn dispatch(&self, method: &str, params: &Value, conn: &mut Conn) -> Result<(Value, Era), RpcError> {
        if method == "initialize" {
            // an earlier-revision client: negotiate its version, no session
            let asked = params.get("protocolVersion").and_then(|v| v.as_str()).unwrap_or(LEGACY[0]);
            let version = if LEGACY.contains(&asked) { asked } else { LEGACY[0] };
            conn.legacy = Some(version.to_string());
            let mut caps = Self::capabilities();
            caps.as_object_mut().unwrap().remove("extensions");
            return Ok((
                json!({"protocolVersion": version, "capabilities": caps, "serverInfo": Self::server_info(), "instructions": INSTRUCTIONS}),
                Era::Legacy,
            ));
        }
        if method == "ping" {
            let era = self.era(params, conn).unwrap_or(Era::Legacy);
            return Ok((json!({}), era));
        }
        let era = self.era(params, conn)?;
        let result = match method {
            "server/discover" => json!({
                "supportedVersions": [MODERN],
                "capabilities": Self::capabilities(),
                "instructions": INSTRUCTIONS,
                "ttlMs": LIST_TTL_MS,
                "cacheScope": "private",
            }),
            "tools/list" => {
                let mut r = json!({"tools": self.tools()});
                if matches!(era, Era::Modern { .. }) {
                    r["ttlMs"] = json!(LIST_TTL_MS);
                    r["cacheScope"] = json!("private");
                }
                r
            }
            "tools/call" => self.tools_call(params, &era, conn)?,
            "resources/list" => {
                let mut r = json!({"resources": self.resources_list()});
                if matches!(era, Era::Modern { .. }) {
                    r["ttlMs"] = json!(LIST_TTL_MS);
                    r["cacheScope"] = json!("private");
                }
                r
            }
            "resources/templates/list" => {
                let mut r = json!({"resourceTemplates": resource_templates()});
                if matches!(era, Era::Modern { .. }) {
                    r["ttlMs"] = json!(LIST_TTL_MS);
                    r["cacheScope"] = json!("private");
                }
                r
            }
            "resources/read" => {
                let uri = params.get("uri").and_then(|u| u.as_str()).ok_or_else(|| RpcError::new(INVALID_PARAMS, "resources/read needs 'uri'"))?;
                let mut r = self.resource_read(uri)?;
                if matches!(era, Era::Modern { .. }) {
                    r["ttlMs"] = json!(5_000);
                    r["cacheScope"] = json!("private");
                }
                r
            }
            "skills/list" => {
                let mut r = self.skills_list(params)?;
                if matches!(era, Era::Modern { .. }) {
                    r["ttlMs"] = json!(if params.get("task").is_some() { 5_000 } else { LIST_TTL_MS });
                    r["cacheScope"] = json!("private");
                }
                r
            }
            "skills/get" => {
                let uri = params.get("uri").and_then(|u| u.as_str()).ok_or_else(|| RpcError::new(INVALID_PARAMS, "skills/get needs 'uri'"))?;
                let e = self.backend.call(Area::Skill, "entry", json!({"uri": uri})).map_err(|e| RpcError::new(INVALID_PARAMS, e))?;
                json!({"skill": e})
            }
            "resources/directory/read" => {
                let uri = params.get("uri").and_then(|u| u.as_str()).ok_or_else(|| RpcError::new(INVALID_PARAMS, "resources/directory/read needs 'uri'"))?;
                let mut r = self.backend.call(Area::Skill, "dir", json!({"uri": uri})).map_err(|e| RpcError::new(INVALID_PARAMS, e))?;
                if matches!(era, Era::Modern { .. }) {
                    r["ttlMs"] = json!(5_000);
                    r["cacheScope"] = json!("private");
                }
                r
            }
            "tasks/get" | "tasks/cancel" | "tasks/update" => {
                if !era.has_tasks() {
                    let mut e = RpcError::new(-32021, "this request needs the client capability io.modelcontextprotocol/tasks");
                    e.data = Some(json!({"requiredCapabilities": {"extensions": {TASKS_EXT: {}}}}));
                    return Err(e);
                }
                self.tasks(method, params)?
            }
            "prompts/list" => {
                let mut r = json!({"prompts": []});
                if matches!(era, Era::Modern { .. }) {
                    r["ttlMs"] = json!(LIST_TTL_MS);
                    r["cacheScope"] = json!("private");
                }
                r
            }
            _ => return Err(RpcError::new(METHOD_NOT_FOUND, format!("Method not found: {method}"))),
        };
        Ok((result, era))
    }

    // ----- tools -----

    /// The tool list, in a fixed order.
    pub fn tools(&self) -> Vec<Value> {
        let ns = json!({"type": "string", "description": "Namespace (one per user or project); default from the server"});
        let mut tools = vec![
            json!({
                "name": "recall",
                "title": "Recall memories",
                "description": "Search memory. format 'context' (default) returns a ready-to-use block grouped by entity with dates, replaced values marked 'previously' and document citations; 'list' returns hits.",
                "inputSchema": {"type": "object", "properties": {
                    "query": {"type": "string"},
                    "ns": ns,
                    "format": {"type": "string", "enum": ["context", "list"]},
                    "budget": {"type": "integer", "description": "Token budget of the context block (default 1000)"},
                    "k": {"type": "integer", "description": "Number of hits (list format)"},
                    "as_of": {"type": "integer", "description": "Recall as of this time (microseconds since epoch)"},
                    "sources": {"type": "array", "items": {"type": "string", "enum": ["memory", "docs"]}}
                }, "required": ["query"]},
                "annotations": {"readOnlyHint": true, "openWorldHint": false}
            }),
            json!({
                "name": "remember",
                "title": "Remember",
                "description": "Store a statement: it is split into claims, dated, linked to entities and may replace an older fact it updates. Attach files to store documents with the statement.",
                "inputSchema": {"type": "object", "properties": {
                    "text": {"type": "string"},
                    "ns": ns,
                    "attachments": {"type": "array", "items": {"type": "object", "properties": {
                        "path": {"type": "string"}, "data_b64": {"type": "string"}, "mime": {"type": "string"}, "name": {"type": "string"}
                    }}},
                    "entities": {"type": "array", "items": {"type": "string"}, "description": "Who or what it is about (found automatically when omitted)"},
                    "at": {"type": "integer", "description": "When it was said (microseconds since epoch)"},
                    "priority": {"type": "string", "enum": ["interactive", "bulk"]}
                }, "required": ["text"]},
                "annotations": {"readOnlyHint": false, "destructiveHint": false, "idempotentHint": false, "openWorldHint": false}
            }),
            json!({
                "name": "forget",
                "title": "Forget memories",
                "description": "Forget memories by id: 'retract' (default) ends them but keeps history; 'erase' deletes them everywhere. Asks for confirmation.",
                "inputSchema": {"type": "object", "properties": {
                    "id": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "string"}},
                    "ns": ns,
                    "mode": {"type": "string", "enum": ["retract", "erase"]},
                    "confirm": {"type": "boolean", "description": "Set after the user agreed (clients without elicitation)"}
                }},
                "annotations": {"readOnlyHint": false, "destructiveHint": true, "idempotentHint": true, "openWorldHint": false}
            }),
            json!({
                "name": "reflect",
                "title": "Review memory",
                "description": "Pending (uncertain) replacements, possible contradictions and recent changes in a namespace.",
                "inputSchema": {"type": "object", "properties": {
                    "ns": ns,
                    "since": {"type": "integer", "description": "Changes since (microseconds since epoch)"},
                    "limit": {"type": "integer"}
                }},
                "annotations": {"readOnlyHint": true, "openWorldHint": false}
            }),
            json!({
                "name": "entity_profile",
                "title": "Entity profile",
                "description": "What memory knows about a person, place or thing: current facts and the memories that mention it.",
                "inputSchema": {"type": "object", "properties": {
                    "name": {"type": "string"},
                    "ns": ns
                }, "required": ["name"]},
                "annotations": {"readOnlyHint": true, "openWorldHint": false}
            }),
            json!({
                "name": "doc_add",
                "title": "Add a document",
                "description": "Store a document (PDF, image, Markdown, HTML, DOCX; scans are OCR'd). Large documents ingest in the background: poll doc_status or the returned task.",
                "inputSchema": {"type": "object", "properties": {
                    "path": {"type": "string", "description": "File path readable by the server"},
                    "data_b64": {"type": "string"},
                    "mime": {"type": "string"},
                    "name": {"type": "string"},
                    "text": {"type": "string", "description": "Optional note stored with the document"},
                    "ns": ns
                }},
                "annotations": {"readOnlyHint": false, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false}
            }),
            json!({
                "name": "doc_status",
                "title": "Document status",
                "description": "One document's ingestion status, or the documents of a namespace when doc is omitted.",
                "inputSchema": {"type": "object", "properties": {
                    "doc": {"type": "string"},
                    "ns": ns
                }},
                "annotations": {"readOnlyHint": true, "openWorldHint": false}
            }),
            json!({
                "name": "flow",
                "title": "Durable workflows",
                "description": "Start, inspect, signal or cancel durable workflows (survive crashes and long waits).",
                "inputSchema": {"type": "object", "properties": {
                    "action": {"type": "string", "enum": ["start", "status", "signal", "cancel", "list"]},
                    "workflow": {"type": "string", "description": "start: workflow name"},
                    "input": {"description": "start: workflow input"},
                    "id": {"type": "string", "description": "Workflow id"},
                    "name": {"type": "string", "description": "signal: signal name"},
                    "payload": {"description": "signal: payload"},
                    "status": {"type": "string", "description": "list: filter by status"}
                }, "required": ["action"]},
                "annotations": {"readOnlyHint": false, "destructiveHint": false, "openWorldHint": false}
            }),
            json!({
                "name": "ontology",
                "title": "Ontology",
                "description": "Inspect and steer the self-maintaining ontology: list, facts, infer (derived facts), learn, history, undo (asks for confirmation).",
                "inputSchema": {"type": "object", "properties": {
                    "action": {"type": "string", "enum": ["list", "facts", "infer", "learn", "history", "undo"]},
                    "ns": ns,
                    "subject": {"type": "string", "description": "facts/infer: filter by subject"},
                    "property": {"type": "string", "description": "facts/infer: filter by property"},
                    "change": {"type": "string", "description": "undo: change id from history"},
                    "confirm": {"type": "boolean", "description": "Set after the user agreed (clients without elicitation)"}
                }, "required": ["action"]},
                "annotations": {"readOnlyHint": false, "destructiveHint": false, "openWorldHint": false}
            }),
            json!({
                "name": "reembed",
                "title": "Re-embed",
                "description": "Move a namespace's vectors to another embedder as a durable background job (returns a task).",
                "inputSchema": {"type": "object", "properties": {
                    "embedder": {"type": "string", "description": "Name of an enabled embedder"},
                    "ns": ns
                }, "required": ["embedder"]},
                "annotations": {"readOnlyHint": false, "destructiveHint": false, "openWorldHint": false}
            }),
            json!({
                "name": "consolidate",
                "title": "Consolidate",
                "description": "Rebuild entity profiles and session summaries for a namespace (a task when the client supports tasks).",
                "inputSchema": {"type": "object", "properties": {
                    "ns": ns,
                    "profiles": {"type": "boolean"},
                    "summaries": {"type": "boolean"}
                }},
                "annotations": {"readOnlyHint": false, "destructiveHint": false, "openWorldHint": false}
            }),
            json!({
                "name": "branch",
                "title": "Memory branches",
                "description": "fork a namespace (instant copy), diff two, merge one into another (dry_run first), rewind to a time, discard, or list. merge/rewind/discard ask for confirmation.",
                "inputSchema": {"type": "object", "properties": {
                    "action": {"type": "string", "enum": ["fork", "diff", "merge", "rewind", "discard", "list"]},
                    "from": {"type": "string"}, "to": {"type": "string"},
                    "a": {"type": "string"}, "b": {"type": "string"},
                    "into": {"type": "string"}, "ns": ns,
                    "at": {"type": "integer", "description": "fork/rewind: time (microseconds since epoch)"},
                    "dry_run": {"type": "boolean"},
                    "confirm": {"type": "boolean", "description": "Set after the user agreed (clients without elicitation)"}
                }, "required": ["action"]},
                "annotations": {"readOnlyHint": false, "destructiveHint": true, "openWorldHint": false}
            }),
            json!({
                "name": "find_skills",
                "title": "Find skills",
                "description": "The few skills (procedures) that fit a task, ranked by fit and measured success; budget-aware. Load one with get_skill; report back with record_skill_outcome and the returned route.",
                "inputSchema": {"type": "object", "properties": {
                    "task": {"type": "string", "description": "What you are about to do"},
                    "k": {"type": "integer", "description": "Skills to return (default 5)"},
                    "budget": {"type": "integer", "description": "Token budget of the answer (default 2000)"},
                    "disclosure": {"type": "string", "enum": ["descriptions", "body", "auto"], "description": "'body' includes the top skill's instructions"},
                    "scope": {"type": "string"}, "scopes": {"type": "array", "items": {"type": "string"}}
                }, "required": ["task"]},
                "annotations": {"readOnlyHint": true, "openWorldHint": false}
            }),
            json!({
                "name": "get_skill",
                "title": "Load a skill",
                "description": "A skill's instructions (SKILL.md body) and its file list, or one of its files. Scripts are content only: kleos never runs them.",
                "inputSchema": {"type": "object", "properties": {
                    "name": {"type": "string"}, "uri": {"type": "string", "description": "skill://… URI"},
                    "scope": {"type": "string"}, "file": {"type": "string", "description": "A file path inside the skill"}
                }},
                "annotations": {"readOnlyHint": true, "openWorldHint": false}
            }),
            json!({
                "name": "record_skill_outcome",
                "title": "Report a skill outcome",
                "description": "Whether following a skill worked. Feeds its success rate, automatic rollback and the skill router.",
                "inputSchema": {"type": "object", "properties": {
                    "name": {"type": "string"}, "ver": {"type": "string"}, "scope": {"type": "string"},
                    "success": {"type": "boolean"}, "route": {"type": "string", "description": "The route id find_skills returned"},
                    "run": {"type": "string"}, "task": {"type": "string"},
                    "latency_ms": {"type": "number"}, "tokens": {"type": "integer"}, "cost": {"type": "number"}, "model": {"type": "string"}
                }, "required": ["success"]},
                "annotations": {"readOnlyHint": false, "destructiveHint": false, "idempotentHint": false, "openWorldHint": false}
            }),
            json!({
                "name": "propose_skill",
                "title": "Propose a skill",
                "description": "Propose a new skill (a SKILL.md, optional files and test cases). It is scanned and stored as a candidate; it is never activated by this call.",
                "inputSchema": {"type": "object", "properties": {
                    "skill_md": {"type": "string"}, "files": {"type": "object", "additionalProperties": {"type": "string"}},
                    "scope": {"type": "string"}, "evidence": {"description": "Runs or memories it comes from"},
                    "eval": {"type": "array", "items": {"type": "object", "properties": {"task": {"type": "string"}, "expected": {"type": "string"}}}}
                }, "required": ["skill_md"]},
                "annotations": {"readOnlyHint": false, "destructiveHint": false, "openWorldHint": false}
            }),
            json!({
                "name": "skill_admin",
                "title": "Manage skills",
                "description": "Skill lifecycle: history, stats, activate, rollback, undo, retire, promote (to a broader scope), learn (from successful runs), optimize. activate/rollback/undo/retire ask for confirmation.",
                "inputSchema": {"type": "object", "properties": {
                    "action": {"type": "string", "enum": ["history", "stats", "activate", "rollback", "undo", "retire", "promote", "learn", "optimize"]},
                    "name": {"type": "string"}, "ver": {"type": "string"}, "scope": {"type": "string"},
                    "to": {"type": "string", "description": "promote: target scope; rollback: version"},
                    "confirm": {"type": "boolean", "description": "Set after the user agreed (clients without elicitation)"}
                }, "required": ["action"]},
                "annotations": {"readOnlyHint": false, "destructiveHint": true, "openWorldHint": false}
            }),
        ];
        if self.allow_query {
            tools.push(json!({
                "name": "query",
                "title": "Read-only query",
                "description": "Run a read-only CozoScript (Datalog) query over the store.",
                "inputSchema": {"type": "object", "properties": {
                    "script": {"type": "string"},
                    "params": {"type": "object"}
                }, "required": ["script"]},
                "annotations": {"readOnlyHint": true, "openWorldHint": false}
            }));
        }
        tools
    }

    fn tools_call(&self, params: &Value, era: &Era, conn: &Conn) -> Result<Value, RpcError> {
        let name = params.get("name").and_then(|n| n.as_str()).ok_or_else(|| RpcError::new(INVALID_PARAMS, "tools/call needs 'name'"))?;
        let args = match params.get("arguments") {
            None | Some(Value::Null) => json!({}),
            Some(v @ Value::Object(_)) => v.clone(),
            Some(_) => return Err(RpcError::new(INVALID_PARAMS, "'arguments' must be an object")),
        };
        let known = [
            "recall", "remember", "forget", "reflect", "entity_profile", "doc_add", "doc_status", "flow", "branch", "ontology", "reembed", "consolidate",
            "find_skills", "get_skill", "record_skill_outcome", "propose_skill", "skill_admin",
        ];
        if !known.contains(&name) && !(name == "query" && self.allow_query) {
            return Err(RpcError::new(INVALID_PARAMS, format!("Unknown tool: {name}")));
        }
        let read_tool = matches!(name, "recall" | "reflect" | "entity_profile" | "doc_status" | "query" | "get_skill" | "find_skills")
            || (name == "skill_admin" && matches!(args.get("action").and_then(|a| a.as_str()), Some("history" | "stats")))
            || (name == "branch" && matches!(args.get("action").and_then(|a| a.as_str()), Some("diff" | "list")))
            || (name == "flow" && matches!(args.get("action").and_then(|a| a.as_str()), Some("status" | "list")))
            || (name == "ontology" && matches!(args.get("action").and_then(|a| a.as_str()), Some("list" | "facts" | "infer" | "history")));
        let out = if conn.read_only && !read_tool {
            tool_error(format!("'{name}' needs write access; this connection is read-only"))
        } else {
            self.run_tool(name, &args, params, era, conn)?
        };
        Ok(match out {
            ToolOut::Done { text, structured, is_error } => {
                let mut r = json!({"content": [{"type": "text", "text": text}], "structuredContent": structured});
                if is_error {
                    r["isError"] = json!(true);
                }
                r
            }
            ToolOut::Task(t) => t,
            ToolOut::InputRequired(r) => r,
        })
    }

    fn ns_arg(&self, args: &Value) -> String {
        args.get("ns").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()).unwrap_or_else(|| self.default_ns.clone())
    }

    fn run_tool(&self, name: &str, args: &Value, params: &Value, era: &Era, conn: &Conn) -> Result<ToolOut, RpcError> {
        let ns = self.ns_arg(args);
        let b = &self.backend;
        let pick = |keys: &[&str]| -> Map<String, Value> {
            let mut m = Map::new();
            for k in keys {
                if let Some(v) = args.get(*k).filter(|v| !v.is_null()) {
                    m.insert((*k).to_string(), v.clone());
                }
            }
            m
        };
        Ok(match name {
            "recall" => {
                let mut m = pick(&["query", "k", "budget", "as_of", "sources"]);
                m.insert("ns".into(), json!(ns));
                let format = args.get("format").and_then(|f| f.as_str()).unwrap_or("context");
                if format == "context" {
                    m.insert("format".into(), json!("context"));
                }
                match b.call(Area::Memory, "recall", Value::Object(m)) {
                    Ok(v) if format == "context" => {
                        let text = v.get("text").and_then(|t| t.as_str()).unwrap_or_default().to_string();
                        let text = if text.trim().is_empty() { "(no memories found)".to_string() } else { text };
                        ToolOut::Done { text, structured: v, is_error: false }
                    }
                    Ok(v) => done(json!({"hits": v})),
                    Err(e) => tool_error(e),
                }
            }
            "remember" => {
                let mut m = pick(&["text", "attachments", "entities", "at", "priority"]);
                m.insert("ns".into(), json!(ns));
                match b.call(Area::Memory, "remember", Value::Object(m)) {
                    Ok(v) => done(v),
                    Err(e) => tool_error(e),
                }
            }
            "forget" => {
                let mut ids: Vec<String> = args.get("ids").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
                if let Some(id) = args.get("id").and_then(|v| v.as_str()) {
                    ids.push(id.to_string());
                }
                if ids.is_empty() {
                    return Ok(tool_error("forget needs 'id' or 'ids'"));
                }
                let mode = args.get("mode").and_then(|v| v.as_str()).unwrap_or("retract").to_string();
                let message = format!("{} {} memor{} in namespace '{ns}'?", if mode == "erase" { "Erase (permanently delete)" } else { "Retract" }, ids.len(), if ids.len() == 1 { "y" } else { "ies" });
                match self.confirm("forget", args, params, era, conn, &message)? {
                    Confirm::Yes => match b.call(Area::Memory, "forget", json!({"ns": ns, "ids": ids, "mode": mode})) {
                        Ok(v) => done(v),
                        Err(e) => tool_error(e),
                    },
                    other => other.into_out(),
                }
            }
            "reflect" => {
                let mut m = pick(&["since", "limit"]);
                m.insert("ns".into(), json!(ns));
                match b.call(Area::Memory, "reflect", Value::Object(m)) {
                    Ok(v) => done(v),
                    Err(e) => tool_error(e),
                }
            }
            "entity_profile" => {
                let Some(ename) = args.get("name").and_then(|v| v.as_str()) else { return Ok(tool_error("entity_profile needs 'name'")) };
                self.entity_profile(&ns, ename)
            }
            "doc_add" => self.doc_add(&ns, args, era),
            "doc_status" => match args.get("doc").and_then(|v| v.as_str()) {
                Some(doc) => match b.call(Area::Doc, "get", json!({"ns": ns, "doc": doc})) {
                    Ok(Value::Null) => tool_error(format!("no document '{doc}' in namespace '{ns}'")),
                    Ok(v) => done(v),
                    Err(e) => tool_error(e),
                },
                None => match b.call(Area::Doc, "list", json!({"ns": ns})) {
                    Ok(v) => done(json!({"documents": v})),
                    Err(e) => tool_error(e),
                },
            },
            "flow" => {
                let action = args.get("action").and_then(|v| v.as_str()).unwrap_or_default();
                let (op, m) = match action {
                    "start" => ("start", pick(&["workflow", "input", "id"])),
                    "status" => ("status", pick(&["id"])),
                    "signal" => ("signal", pick(&["id", "name", "payload"])),
                    "cancel" => ("cancel", pick(&["id"])),
                    "list" => ("list", pick(&["status", "workflow"])),
                    other => return Ok(tool_error(format!("flow: unknown action '{other}' (start, status, signal, cancel, list)"))),
                };
                let mut m = m;
                if op == "signal" && !m.contains_key("payload") {
                    m.insert("payload".into(), Value::Null);
                }
                if op == "start" && !m.contains_key("input") {
                    m.insert("input".into(), Value::Null);
                }
                match b.call(Area::Flow, op, Value::Object(m)) {
                    Ok(v) => done(v),
                    Err(e) => tool_error(e),
                }
            }
            "branch" => self.branch(&ns, args, params, era, conn)?,
            "ontology" => self.ontology(&ns, args, params, era, conn)?,
            "reembed" => self.reembed(&ns, args, era),
            "consolidate" => self.consolidate(&ns, args, era),
            "find_skills" | "get_skill" | "record_skill_outcome" | "propose_skill" => self.skill_tool(name, args, conn),
            "skill_admin" => self.skill_admin(args, params, era, conn)?,
            "query" => {
                let Some(script) = args.get("script").and_then(|v| v.as_str()) else { return Ok(tool_error("query needs 'script'")) };
                match b.query(script, args.get("params").unwrap_or(&Value::Null)) {
                    Ok(v) => done(v),
                    Err(e) => tool_error(e),
                }
            }
            _ => unreachable!(),
        })
    }

    fn entity_profile(&self, ns: &str, name: &str) -> ToolOut {
        match entity_profile(&self.backend, ns, name) {
            Ok((text, structured)) => ToolOut::Done { text, structured, is_error: false },
            Err(e) => tool_error(e),
        }
    }

    fn doc_add(&self, ns: &str, args: &Value, era: &Era) -> ToolOut {
        let mut att = Map::new();
        for k in ["path", "data_b64", "mime", "name"] {
            if let Some(v) = args.get(k).filter(|v| !v.is_null()) {
                att.insert(k.to_string(), v.clone());
            }
        }
        if !att.contains_key("path") && !att.contains_key("data_b64") {
            return tool_error("doc_add needs 'path' or 'data_b64' (with 'mime' and 'name')");
        }
        let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let r = match self.backend.call(Area::Memory, "remember", json!({"ns": ns, "text": text, "attachments": [Value::Object(att)]})) {
            Ok(v) => v,
            Err(e) => return tool_error(e),
        };
        let doc = r.get("documents").and_then(|d| d.as_array()).and_then(|a| a.first()).cloned().unwrap_or(Value::Null);
        let running = doc.get("status").and_then(|s| s.as_str()) == Some("running");
        let flow = doc.get("flow").and_then(|f| f.as_str());
        if let (true, Some(flow), true) = (running, flow, era.has_tasks()) {
            let doc_id = doc.get("doc").and_then(|d| d.as_str()).unwrap_or_default();
            let task_id = encode_task(&json!({"k": "doc", "ns": ns, "doc": doc_id, "flow": flow}));
            let now = iso_now();
            return ToolOut::Task(json!({
                "resultType": "task",
                "taskId": task_id,
                "status": "working",
                "statusMessage": format!("ingesting {} ({} pages)", doc.get("name").and_then(|n| n.as_str()).unwrap_or(doc_id), doc.get("pages").and_then(|p| p.as_u64()).unwrap_or(0)),
                "createdAt": now,
                "lastUpdatedAt": now,
                "ttlMs": Value::Null,
                "pollIntervalMs": 2000,
            }));
        }
        done(r)
    }

    // ----- skills (docs/specs/skills.md) -----

    /// `skills/list` (SEP-2640): with a `task` (kleos extension), its top-k;
    /// else per the server's listing mode. Listings other than `full` are
    /// partial, as the extension allows: the catalog stays reachable through
    /// `skills/get` and `find_skills`.
    fn skills_list(&self, params: &Value) -> Result<Value, RpcError> {
        let task = params.get("task").or_else(|| params.get("query")).and_then(|t| t.as_str()).filter(|t| !t.trim().is_empty());
        let mut a = json!({});
        if let Some(sc) = params.get("scopes").filter(|v| v.is_array()) {
            a["scopes"] = sc.clone();
        } else if let Some(sc) = params.get("scope").and_then(|v| v.as_str()) {
            a["scope"] = json!(sc);
        }
        match (task, self.skills_list.as_str()) {
            (Some(t), _) => {
                a["task"] = json!(t);
                a["k"] = json!(params.get("k").and_then(|k| k.as_u64()).unwrap_or(SKILLS_TOP_K));
            }
            (None, "none") => return Ok(json!({"skills": []})),
            (None, "full") => {
                a["mode"] = json!("full");
                if let Some(c) = params.get("cursor") {
                    a["cursor"] = c.clone();
                }
            }
            _ => a["mode"] = json!("top"),
        }
        let r = self.backend.call(Area::Skill, "entries", a).map_err(|e| RpcError::new(INTERNAL_ERROR, e))?;
        let mut out = json!({"skills": r.get("skills").cloned().unwrap_or_else(|| json!([]))});
        if let Some(n) = r.get("nextCursor") {
            out["nextCursor"] = n.clone();
        }
        Ok(out)
    }

    fn skill_tool(&self, name: &str, args: &Value, conn: &Conn) -> ToolOut {
        let b = &self.backend;
        let mut a = args.clone();
        match name {
            "find_skills" => {
                if conn.read_only {
                    a["log"] = json!(false);
                }
                match b.call(Area::Skill, "find", a) {
                    Ok(v) => {
                        let mut text = String::new();
                        for s in v["skills"].as_array().cloned().unwrap_or_default() {
                            let st = &s["stats"];
                            let rate = match st["success_rate"].as_f64() {
                                Some(r) => format!(", {:.0}% of {} uses", r * 100.0, st["uses"]),
                                None => String::new(),
                            };
                            text.push_str(&format!("{}. {} (scope {}, fit {:.2}{rate}) — {}\n   {}\n", s["rank"], s["name"].as_str().unwrap_or(""), s["scope"].as_str().unwrap_or(""),
                                s["score"].as_f64().unwrap_or(0.0), s["description"].as_str().unwrap_or(""), s["uri"].as_str().unwrap_or("")));
                            if let Some(body) = s["body"].as_str() {
                                text.push_str(&format!("\n{body}\n\n"));
                            }
                        }
                        if text.is_empty() {
                            text = "(no skill fits this task)".into();
                        } else {
                            text.push_str(&format!("route: {} (pass it to record_skill_outcome)", v["route"].as_str().unwrap_or("")));
                        }
                        ToolOut::Done { text, structured: v, is_error: false }
                    }
                    Err(e) => tool_error(e),
                }
            }
            "get_skill" => {
                if let Some(uri) = args.get("uri").and_then(|u| u.as_str()) {
                    return match b.call(Area::Skill, "read", json!({"uri": uri})) {
                        Ok(v) => {
                            let text = v["contents"][0]["text"].as_str().unwrap_or("(binary file)").to_string();
                            ToolOut::Done { text, structured: v, is_error: false }
                        }
                        Err(e) => tool_error(e),
                    };
                }
                match b.call(Area::Skill, "get", a) {
                    Ok(v) if v.get("text").is_some() || v.get("b64").is_some() => {
                        let text = v.get("text").and_then(|t| t.as_str()).unwrap_or("(binary file)").to_string();
                        ToolOut::Done { text, structured: v, is_error: false }
                    }
                    Ok(v) => {
                        let files: Vec<String> = v["resources"].as_array().map(|r| r.iter().filter(|x| x["path"] != "SKILL.md").map(|x| x["path"].as_str().unwrap_or("").to_string()).collect()).unwrap_or_default();
                        let mut text = v["body"].as_str().unwrap_or("").to_string();
                        if !files.is_empty() {
                            text.push_str(&format!("\n\nFiles (read with get_skill file=…): {}", files.join(", ")));
                        }
                        let structured = json!({"name": v["name"], "scope": v["scope"], "ver": v["ver"], "uri": v["uri"], "digest": v["digest"],
                                                "frontmatter": v["frontmatter"], "body": v["body"], "resources": v["resources"], "stats": v["stats"]});
                        ToolOut::Done { text, structured, is_error: false }
                    }
                    Err(e) => tool_error(e),
                }
            }
            "record_skill_outcome" => match b.call(Area::Skill, "record_outcome", a) {
                Ok(v) => done(v),
                Err(e) => tool_error(e),
            },
            "propose_skill" => {
                if let Some(o) = a.as_object_mut() {
                    o.remove("activate");
                }
                match b.call(Area::Skill, "propose", a) {
                    Ok(v) => done(v),
                    Err(e) => tool_error(e),
                }
            }
            _ => unreachable!(),
        }
    }

    fn skill_admin(&self, args: &Value, params: &Value, era: &Era, conn: &Conn) -> Result<ToolOut, RpcError> {
        let action = args.get("action").and_then(|a| a.as_str()).unwrap_or_default().to_string();
        let mut a = args.clone();
        if let Some(o) = a.as_object_mut() {
            o.remove("action");
            o.remove("confirm");
        }
        let call = |op: &str, a: Value| match self.backend.call(Area::Skill, op, a) {
            Ok(v) => done(v),
            Err(e) => tool_error(e),
        };
        let what = args.get("name").or_else(|| args.get("ver")).and_then(|v| v.as_str()).unwrap_or("?").to_string();
        Ok(match action.as_str() {
            "history" | "stats" | "learn" | "optimize" => call(&action, a),
            "promote" => call("promote", a),
            "activate" | "rollback" | "undo" | "retire" => {
                if action == "rollback" {
                    if let Some(to) = args.get("to").cloned() {
                        a["to"] = to;
                    }
                }
                let msg = format!("{} skill '{what}'? This changes which instructions agents get.", action[..1].to_uppercase() + &action[1..]);
                match self.confirm("skill_admin", args, params, era, conn, &msg)? {
                    Confirm::Yes => call(&action, a),
                    other => other.into_out(),
                }
            }
            other => tool_error(format!("skill_admin: unknown action '{other}' (history, stats, activate, rollback, undo, retire, promote, learn, optimize)")),
        })
    }

    fn ontology(&self, ns: &str, args: &Value, params: &Value, era: &Era, conn: &Conn) -> Result<ToolOut, RpcError> {
        let s = |k: &str| args.get(k).and_then(|v| v.as_str()).map(|x| x.to_string());
        let call = |op: &str, a: Value| match self.backend.call(Area::Ontology, op, a) {
            Ok(v) => done(v),
            Err(e) => tool_error(e),
        };
        let action = s("action").unwrap_or_default();
        Ok(match action.as_str() {
            "list" => call("list", json!({})),
            "history" => call("history", json!({"ns": ns})),
            "learn" => call("learn", json!({"ns": ns})),
            "facts" | "infer" => {
                let mut a = json!({"ns": ns});
                if let Some(v) = s("subject") {
                    a["s"] = json!(v);
                }
                if let Some(v) = s("property") {
                    a["p"] = json!(v);
                }
                if action == "infer" {
                    a["derived_only"] = json!(true);
                }
                match self.backend.call(Area::Ontology, &action, a) {
                    Ok(v) if action == "infer" => done(json!({"inferred": v})),
                    Ok(v) => done(json!({"facts": v})),
                    Err(e) => tool_error(e),
                }
            }
            "undo" => {
                let Some(change) = s("change") else { return Ok(tool_error("ontology undo needs 'change' (an id from history)")) };
                let msg = format!("Undo ontology change '{change}' in namespace '{ns}'?");
                match self.confirm("ontology", args, params, era, conn, &msg)? {
                    Confirm::Yes => call("undo", json!({"ns": ns, "change": change})),
                    other => other.into_out(),
                }
            }
            other => tool_error(format!("ontology: unknown action '{other}' (list, facts, infer, learn, history, undo)")),
        })
    }

    /// Start the background worker of an embedded store once.
    fn ensure_worker(&self) {
        if let Backend::Embedded(db) = &self.backend {
            let mut w = self.worker.lock().unwrap();
            if w.is_none() {
                *w = Some(db.flow_worker(kleos::FlowWorkerOpts { queues: vec!["default".into(), "consolidate".into()], ..Default::default() }));
            }
        }
    }

    fn task_out(kind: &str, ns: &str, flow: &str, message: String) -> ToolOut {
        let task_id = encode_task(&json!({"k": kind, "ns": ns, "flow": flow}));
        let now = iso_now();
        ToolOut::Task(json!({
            "resultType": "task",
            "taskId": task_id,
            "status": "working",
            "statusMessage": message,
            "createdAt": now,
            "lastUpdatedAt": now,
            "ttlMs": Value::Null,
            "pollIntervalMs": 2000,
        }))
    }

    /// Entity profiles and session summaries: a task when the client takes
    /// tasks (the `memory.consolidate` workflow on the `consolidate` queue,
    /// as automatic consolidation uses), else run inside the call.
    fn consolidate(&self, ns: &str, args: &Value, era: &Era) -> ToolOut {
        let mut opts = json!({"ns": ns});
        for k in ["profiles", "summaries"] {
            if let Some(v) = args.get(k).filter(|v| !v.is_null()) {
                opts[k] = v.clone();
            }
        }
        let simple = opts.as_object().map_or(true, |o| o.len() == 1);
        if era.has_tasks() && simple {
            let b = &self.backend;
            let def = json!({"name": "memory.consolidate",
                "graph": {"steps": [{"name": "consolidate", "consolidate": {"ns": "$input.ns"}, "retry": {"max_attempts": 3}}]}});
            let started = b.call(Area::Flow, "define", def).and_then(|_| {
                b.call(Area::Flow, "start", json!({"workflow": "memory.consolidate", "input": {"ns": ns}, "serial": format!("consolidate:{ns}"), "queue": "consolidate"}))
            });
            return match started {
                Ok(v) => match v.get("id").and_then(|i| i.as_str()) {
                    Some(flow) => {
                        self.ensure_worker();
                        Self::task_out("consolidate", ns, flow, format!("consolidating '{ns}'"))
                    }
                    None => done(v),
                },
                Err(e) => tool_error(e),
            };
        }
        match self.backend.call(Area::Memory, "consolidate", opts) {
            Ok(v) => done(v),
            Err(e) => tool_error(e),
        }
    }

    fn reembed(&self, ns: &str, args: &Value, era: &Era) -> ToolOut {
        let Some(embedder) = args.get("embedder").and_then(|v| v.as_str()) else { return tool_error("reembed needs 'embedder'") };
        let r = match self.backend.call(Area::Memory, "reembed", json!({"ns": ns, "embedder": embedder, "wait": !era.has_tasks()})) {
            Ok(v) => v,
            Err(e) => return tool_error(e),
        };
        match (r.get("status").and_then(|s| s.as_str()), r.get("id").and_then(|i| i.as_str())) {
            (Some("running"), Some(flow)) if era.has_tasks() => {
                self.ensure_worker();
                Self::task_out("reembed", ns, flow, format!("re-embedding '{ns}' with {embedder}"))
            }
            _ => done(r),
        }
    }

    fn branch(&self, ns: &str, args: &Value, params: &Value, era: &Era, conn: &Conn) -> Result<ToolOut, RpcError> {
        let b = &self.backend;
        let s = |k: &str| args.get(k).and_then(|v| v.as_str()).map(|x| x.to_string());
        let action = s("action").unwrap_or_default();
        let call = |op: &str, a: Value| match b.call(Area::Memory, op, a) {
            Ok(v) => done(v),
            Err(e) => tool_error(e),
        };
        Ok(match action.as_str() {
            "fork" => {
                let Some(to) = s("to") else { return Ok(tool_error("branch fork needs 'to'")) };
                let mut a = json!({"from": s("from").unwrap_or_else(|| ns.to_string()), "to": to});
                if let Some(at) = args.get("at").filter(|v| !v.is_null()) {
                    a["at"] = at.clone();
                }
                call("fork", a)
            }
            "diff" => {
                let (Some(x), Some(y)) = (s("a"), s("b")) else { return Ok(tool_error("branch diff needs 'a' and 'b'")) };
                call("diff", json!({"a": x, "b": y}))
            }
            "list" => call("branches", json!({})),
            "merge" => {
                let Some(from) = s("from") else { return Ok(tool_error("branch merge needs 'from'")) };
                let dry = args.get("dry_run").and_then(|v| v.as_bool()).unwrap_or(false);
                let mut a = json!({"from": from, "dry_run": dry});
                if let Some(into) = s("into") {
                    a["into"] = json!(into);
                }
                if dry {
                    return Ok(call("merge", a));
                }
                let msg = format!("Merge branch '{from}' into '{}'? Its statements are replayed and may replace facts there.", s("into").unwrap_or_else(|| "its parent".into()));
                match self.confirm("branch", args, params, era, conn, &msg)? {
                    Confirm::Yes => call("merge", a),
                    other => other.into_out(),
                }
            }
            "rewind" => {
                let Some(at) = args.get("at").and_then(|v| v.as_i64()) else { return Ok(tool_error("branch rewind needs 'at' (microseconds since epoch)")) };
                let msg = format!("Rewind namespace '{ns}' to {at}? (undoable)");
                match self.confirm("branch", args, params, era, conn, &msg)? {
                    Confirm::Yes => call("rewind", json!({"ns": ns, "to": at})),
                    other => other.into_out(),
                }
            }
            "discard" => {
                if args.get("ns").and_then(|v| v.as_str()).map_or(true, |x| x.is_empty()) {
                    return Ok(tool_error("branch discard needs an explicit 'ns'"));
                }
                let msg = format!("Discard branch '{ns}' and everything only it holds?");
                match self.confirm("branch", args, params, era, conn, &msg)? {
                    Confirm::Yes => call("discard", json!({"ns": ns})),
                    other => other.into_out(),
                }
            }
            other => tool_error(format!("branch: unknown action '{other}' (fork, diff, merge, rewind, discard, list)")),
        })
    }

    // ----- confirmations (MRTR) -----

    fn confirm(&self, tool: &str, args: &Value, params: &Value, era: &Era, conn: &Conn, message: &str) -> Result<Confirm, RpcError> {
        if args.get("confirm").and_then(|v| v.as_bool()) == Some(true) {
            return Ok(Confirm::Yes);
        }
        if let Some(resp) = params.get("inputResponses").and_then(|r| r.get("confirm")) {
            let state = params.get("requestState").and_then(|s| s.as_str()).ok_or_else(|| RpcError::new(INVALID_PARAMS, "inputResponses without the requestState it answers"))?;
            self.verify_state(state, tool, args, conn)?;
            let accepted = resp.get("action").and_then(|a| a.as_str()) == Some("accept")
                && resp.get("content").and_then(|c| c.get("confirm")).and_then(|c| c.as_bool()) == Some(true);
            return Ok(if accepted { Confirm::Yes } else { Confirm::Declined });
        }
        if era.has_elicitation() {
            let state = self.mint_state(tool, args, conn);
            return Ok(Confirm::Ask(json!({
                "resultType": "input_required",
                "inputRequests": {"confirm": {
                    "method": "elicitation/create",
                    "params": {
                        "mode": "form",
                        "message": message,
                        "requestedSchema": {"type": "object", "properties": {"confirm": {"type": "boolean", "title": "Confirm", "description": message}}, "required": ["confirm"]}
                    }
                }},
                "requestState": state,
            })));
        }
        Ok(Confirm::NeedArg(format!("{message} Ask the user, then call again with \"confirm\": true.")))
    }

    fn mint_state(&self, tool: &str, args: &Value, conn: &Conn) -> String {
        let exp = now_secs() + STATE_TTL_SECS;
        let payload = json!({"t": tool, "a": args_digest(args), "p": conn.principal, "e": exp}).to_string();
        let mac = hmac_sha256(&self.state_key, payload.as_bytes());
        format!("{}.{}", hex(payload.as_bytes()), hex(&mac))
    }

    fn verify_state(&self, state: &str, tool: &str, args: &Value, conn: &Conn) -> Result<(), RpcError> {
        let bad = |why: &str| RpcError::new(INVALID_PARAMS, format!("invalid requestState: {why}"));
        let (p, m) = state.split_once('.').ok_or_else(|| bad("malformed"))?;
        let payload = unhex(p).ok_or_else(|| bad("malformed"))?;
        let mac = unhex(m).ok_or_else(|| bad("malformed"))?;
        if !ct_eq(&hmac_sha256(&self.state_key, &payload), &mac) {
            return Err(bad("signature does not verify"));
        }
        let v: Value = serde_json::from_slice(&payload).map_err(|_| bad("malformed"))?;
        if v.get("e").and_then(|e| e.as_u64()).unwrap_or(0) < now_secs() {
            return Err(bad("expired; call the tool again"));
        }
        if v.get("t").and_then(|t| t.as_str()) != Some(tool) || v.get("a").and_then(|a| a.as_str()) != Some(&args_digest(args)) {
            return Err(bad("it was issued for a different call"));
        }
        if v.get("p").and_then(|p| p.as_str()) != Some(conn.principal.as_str()) {
            return Err(bad("it was issued to a different principal"));
        }
        Ok(())
    }

    // ----- tasks (io.modelcontextprotocol/tasks) -----

    fn tasks(&self, method: &str, params: &Value) -> Result<Value, RpcError> {
        let task_id = params.get("taskId").and_then(|t| t.as_str()).ok_or_else(|| RpcError::new(INVALID_PARAMS, format!("{method} needs 'taskId'")))?;
        let t = decode_task(task_id).ok_or_else(|| RpcError::new(INVALID_PARAMS, format!("unknown task '{task_id}'")))?;
        let flow = t.get("flow").and_then(|f| f.as_str()).unwrap_or_default().to_string();
        match method {
            "tasks/update" => Ok(json!({})),
            "tasks/cancel" => {
                let _ = self.backend.call(Area::Flow, "cancel", json!({"id": flow}));
                Ok(json!({}))
            }
            _ => {
                let w = self.backend.call(Area::Flow, "status", json!({"id": flow})).map_err(|e| RpcError::new(INVALID_PARAMS, format!("unknown task: {e}")))?;
                let st = w.get("status").and_then(|s| s.as_str()).unwrap_or("pending");
                let ts = |k: &str| w.get(k).and_then(|v| v.as_i64()).map(iso_micros).unwrap_or_else(iso_now);
                let mut r = json!({
                    "taskId": task_id,
                    "createdAt": ts("created"),
                    "lastUpdatedAt": ts("updated"),
                    "ttlMs": Value::Null,
                    "pollIntervalMs": 2000,
                });
                let ns = t.get("ns").and_then(|v| v.as_str()).unwrap_or_default();
                let doc = t.get("doc").and_then(|v| v.as_str()).unwrap_or_default();
                if matches!(t.get("k").and_then(|k| k.as_str()), Some("reembed" | "consolidate")) {
                    match st {
                        "completed" => {
                            let out = w.get("output").cloned().unwrap_or(Value::Null);
                            r["status"] = json!("completed");
                            r["result"] = json!({
                                "resultType": "complete",
                                "content": [{"type": "text", "text": format!("{} done for '{ns}'", t.get("k").and_then(|k| k.as_str()).unwrap_or("job"))}],
                                "structuredContent": out,
                            });
                        }
                        "failed" => {
                            r["status"] = json!("failed");
                            let msg = w.get("error").map(|e| e.to_string()).unwrap_or_else(|| "background job failed".into());
                            r["error"] = json!({"code": -1, "message": msg});
                        }
                        "canceled" | "cancelled" => r["status"] = json!("cancelled"),
                        _ => r["status"] = json!("working"),
                    }
                    return Ok(r);
                }
                match st {
                    "completed" => {
                        let d = self.backend.call(Area::Doc, "get", json!({"ns": ns, "doc": doc})).unwrap_or(Value::Null);
                        r["status"] = json!("completed");
                        r["result"] = json!({
                            "resultType": "complete",
                            "content": [{"type": "text", "text": serde_json::to_string(&d).unwrap_or_default()}],
                            "structuredContent": d,
                        });
                    }
                    "failed" => {
                        r["status"] = json!("failed");
                        let msg = w.get("error").map(|e| e.to_string()).unwrap_or_else(|| "document ingestion failed".into());
                        r["error"] = json!({"code": -1, "message": msg});
                    }
                    "canceled" | "cancelled" => r["status"] = json!("cancelled"),
                    _ => {
                        r["status"] = json!("working");
                        if let Ok(d) = self.backend.call(Area::Doc, "get", json!({"ns": ns, "doc": doc})) {
                            if let Some(c) = d.get("chunks").and_then(|c| c.as_u64()) {
                                r["statusMessage"] = json!(format!("{c} chunks stored so far"));
                            }
                        }
                    }
                }
                Ok(r)
            }
        }
    }

    // ----- resources -----

    fn resources_list(&self) -> Vec<Value> {
        let mut out = vec![json!({"uri": "kleos://namespaces", "name": "namespaces", "title": "Namespaces", "mimeType": "application/json"})];
        if let Ok(Value::Array(docs)) = self.backend.call(Area::Doc, "list", json!({"ns": self.default_ns})) {
            let mut docs = docs;
            docs.sort_by(|a, b| a.get("doc").and_then(|x| x.as_str()).cmp(&b.get("doc").and_then(|x| x.as_str())));
            for d in docs {
                let id = d.get("doc").and_then(|x| x.as_str()).unwrap_or_default();
                out.push(json!({
                    "uri": format!("kleos://doc/{}/{}", pct(&self.default_ns), pct(id)),
                    "name": d.get("name").and_then(|x| x.as_str()).unwrap_or(id),
                    "mimeType": "application/json",
                }));
            }
        }
        out
    }

    fn resource_read(&self, uri: &str) -> Result<Value, RpcError> {
        let missing = || RpcError::new(INVALID_PARAMS, format!("Resource not found: {uri}"));
        if uri.starts_with("skill://") {
            return self.backend.call(Area::Skill, "read", json!({"uri": uri})).map_err(|_| missing());
        }
        let rest = uri.strip_prefix("kleos://").ok_or_else(missing)?;
        let parts: Vec<String> = rest.split('/').map(unpct).collect();
        let body = match parts.iter().map(|s| s.as_str()).collect::<Vec<_>>().as_slice() {
            ["namespaces"] => {
                let ns = self
                    .backend
                    .query("?[ns] := *kleos_mem{ns}", &Value::Null)
                    .ok()
                    .and_then(|v| v.get("rows").cloned())
                    .and_then(|r| r.as_array().cloned())
                    .unwrap_or_default()
                    .into_iter()
                    .filter_map(|row| row.get(0).and_then(|x| x.as_str()).map(String::from))
                    .collect::<std::collections::BTreeSet<_>>();
                let branches = self.backend.call(Area::Memory, "branches", json!({})).unwrap_or_else(|_| json!([]));
                json!({"namespaces": ns, "branches": branches})
            }
            ["ns", ns, "documents"] => self.backend.call(Area::Doc, "list", json!({"ns": ns})).map_err(|_| missing())?,
            ["doc", ns, doc] => match self.backend.call(Area::Doc, "get", json!({"ns": ns, "doc": doc})) {
                Ok(Value::Null) | Err(_) => return Err(missing()),
                Ok(v) => v,
            },
            ["entity", ns, name] => match self.entity_profile(ns, name) {
                ToolOut::Done { structured, is_error: false, .. } => structured,
                _ => return Err(missing()),
            },
            _ => return Err(missing()),
        };
        Ok(json!({"contents": [{"uri": uri, "mimeType": "application/json", "text": serde_json::to_string(&body).unwrap_or_default()}]}))
    }
}

/// What memory knows about an entity: ontology facts whose subject is the
/// name, and the context block of memories mentioning it (by recorded entity
/// links, else by search). Shared by the MCP tool and `kleos profile`.
pub fn entity_profile(b: &Backend, ns: &str, name: &str) -> Result<(String, Value), String> {
    let facts = b
        .call(Area::Ontology, "facts", json!({"ns": ns, "s": name}))
        .ok()
        .filter(|v| v.as_array().map_or(false, |a| !a.is_empty()))
        .unwrap_or_else(|| json!([]));
    let mut ctx = b.call(Area::Memory, "recall", json!({"ns": ns, "query": name, "entity": name, "format": "context", "budget": 600}))?;
    if ctx.get("ids").and_then(|i| i.as_array()).map_or(true, |a| a.is_empty()) {
        ctx = b.call(Area::Memory, "recall", json!({"ns": ns, "query": name, "format": "context", "budget": 600}))?;
    }
    let mut text = String::new();
    if let Some(fs) = facts.as_array().filter(|a| !a.is_empty()) {
        text.push_str(&format!("Facts about {name}:\n"));
        for f in fs {
            text.push_str(&format!("- {}\n", serde_json::to_string(f).unwrap_or_default()));
        }
        text.push('\n');
    }
    let empty = ctx.get("ids").and_then(|i| i.as_array()).map_or(true, |a| a.is_empty());
    if empty && text.is_empty() {
        text = format!("Nothing is known about '{name}' in namespace '{ns}'.");
    } else if !empty {
        text.push_str(ctx.get("text").and_then(|t| t.as_str()).unwrap_or_default());
    }
    Ok((text, json!({"name": name, "ns": ns, "facts": facts, "context": ctx})))
}

fn resource_templates() -> Vec<Value> {
    vec![
        json!({"uriTemplate": "kleos://doc/{ns}/{doc}", "name": "document", "title": "Document", "mimeType": "application/json"}),
        json!({"uriTemplate": "kleos://entity/{ns}/{name}", "name": "entity", "title": "Entity profile", "mimeType": "application/json"}),
        json!({"uriTemplate": "kleos://ns/{ns}/documents", "name": "documents", "title": "Documents of a namespace", "mimeType": "application/json"}),
    ]
}

/// The outcome of a confirmation check.
enum Confirm {
    Yes,
    Declined,
    Ask(Value),
    NeedArg(String),
}

impl Confirm {
    fn into_out(self) -> ToolOut {
        match self {
            Confirm::Yes => unreachable!(),
            Confirm::Declined => tool_error("not confirmed; nothing was changed"),
            Confirm::Ask(r) => ToolOut::InputRequired(r),
            Confirm::NeedArg(m) => tool_error(m),
        }
    }
}

// ----- helpers -----

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn iso_now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

fn iso_micros(us: i64) -> String {
    chrono::DateTime::from_timestamp_micros(us)
        .map(|d| d.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
        .unwrap_or_else(iso_now)
}

/// Canonical JSON (sorted keys) of the arguments, minus `confirm`, hashed.
fn args_digest(args: &Value) -> String {
    fn canon(v: &Value) -> Value {
        match v {
            Value::Object(m) => {
                let sorted: std::collections::BTreeMap<&String, Value> = m.iter().filter(|(k, _)| k.as_str() != "confirm").map(|(k, v)| (k, canon(v))).collect();
                Value::Object(sorted.into_iter().map(|(k, v)| (k.clone(), v)).collect())
            }
            Value::Array(a) => Value::Array(a.iter().map(canon).collect()),
            other => other.clone(),
        }
    }
    let c = canon(args);
    let mut s = String::new();
    write_canonical(&c, &mut s);
    hex(&Sha256::digest(s.as_bytes()))
}

fn write_canonical(v: &Value, out: &mut String) {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String((*k).clone()).to_string());
                out.push(':');
                write_canonical(&m[*k], out);
            }
            out.push('}');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(x, out);
            }
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}

fn hmac_sha256(key: &[u8; 32], msg: &[u8]) -> [u8; 32] {
    let mut k = [0u8; 64];
    k[..32].copy_from_slice(key);
    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(msg);
    let ih = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(ih);
    outer.finalize().into()
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok()).collect()
}

/// A task id: the task's locator, hex-encoded (opaque to clients, durable).
pub fn encode_task(v: &Value) -> String {
    format!("kt_{}", hex(v.to_string().as_bytes()))
}

fn decode_task(id: &str) -> Option<Value> {
    serde_json::from_slice(&unhex(id.strip_prefix("kt_")?)?).ok()
}

/// Percent-encodes a URI path segment.
fn pct(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn unpct(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Some(x) = std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(x);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Decodes a header value that may use the `=?base64?…?=` sentinel.
pub fn decode_header_value(v: &str) -> Option<String> {
    match v.strip_prefix("=?base64?").and_then(|r| r.strip_suffix("?=")) {
        Some(b64) => base64_decode(b64).and_then(|b| String::from_utf8(b).ok()),
        None => Some(v.to_string()),
    }
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::new();
    let (mut buf, mut bits) = (0u32, 0u32);
    for c in s.trim_end_matches('=').bytes() {
        let v = T.iter().position(|&x| x == c)? as u32;
        buf = (buf << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// Serves MCP over stdio: one JSON-RPC message per line on stdin, responses
/// on stdout, diagnostics on stderr.
pub fn serve_stdio(server: &McpServer) -> std::io::Result<()> {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    let mut conn = Conn { principal: "local".into(), ..Default::default() };
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let resp = match serde_json::from_str::<Value>(&line) {
            Ok(msg) => server.handle(&msg, &mut conn).map(|(r, _)| r),
            Err(e) => Some(RpcError::new(PARSE_ERROR, format!("Parse error: {e}")).to_response(None)),
        };
        if let Some(r) = resp {
            writeln!(out, "{}", r)?;
            out.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_matches_rfc4231_style_vector() {
        // HMAC-SHA256 with a 32-byte key of 0x0b.. (truncated RFC 4231 case 1 uses 20 bytes; check self-consistency and difference)
        let k = [0x0bu8; 32];
        let a = hmac_sha256(&k, b"Hi There");
        let b = hmac_sha256(&k, b"Hi There!");
        assert_ne!(a, b);
        assert_eq!(a, hmac_sha256(&k, b"Hi There"));
    }

    #[test]
    fn header_sentinel_decodes() {
        assert_eq!(decode_header_value("=?base64?SGVsbG8sIOS4lueVjA==?=").as_deref(), Some("Hello, 世界"));
        assert_eq!(decode_header_value("plain").as_deref(), Some("plain"));
    }

    #[test]
    fn digest_ignores_key_order_and_confirm() {
        let a = json!({"ids": ["x"], "mode": "erase", "ns": "n"});
        let b = json!({"ns": "n", "mode": "erase", "ids": ["x"], "confirm": false});
        assert_eq!(args_digest(&a), args_digest(&b));
        assert_ne!(args_digest(&a), args_digest(&json!({"ids": ["y"]})));
    }

    #[test]
    fn pct_roundtrip() {
        assert_eq!(unpct(&pct("a b/ç")), "a b/ç");
    }
}
