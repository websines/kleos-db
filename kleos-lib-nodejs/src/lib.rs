/*
 * Copyright 2022, The Cozo Project Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Native Node.js binding for kleos (napi-rs). The `kleos` TypeScript SDK
//! (`sdk/ts`) wraps it; it is not meant to be used directly.
//!
//! Everything crosses the boundary as JSON text: arguments go in as JSON,
//! results come back as JSON, and errors are thrown with a JSON message
//! `{"ok": false, "message": ...}`. The memory, ontology, flow and document
//! APIs are already JSON-in/JSON-out in kleos-core (`DbInstance::memory_call`
//! and friends, the same functions the HTTP server and Python call), so the
//! binding stays a thin layer.
//!
//! Every call that touches the store or a model runs as an [`AsyncTask`] on
//! the libuv thread pool and resolves a Promise: the Node event loop is
//! never blocked.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::{json, Value as JsonValue};

use kleos::{DataValue, DbInstance, FlowWorker, FlowWorkerOpts, ScriptMutability};

fn js_err(message: impl std::fmt::Display) -> Error {
    Error::new(Status::GenericFailure, json!({"ok": false, "message": message.to_string()}).to_string())
}

fn report_err(e: miette::Report) -> Error {
    js_err(e.to_string())
}

fn parse_json(text: &str, what: &str) -> Result<JsonValue> {
    if text.trim().is_empty() {
        return Ok(JsonValue::Null);
    }
    serde_json::from_str(text).map_err(|e| js_err(format!("{what}: invalid JSON: {e}")))
}

const DB_CLOSED: &str = "database closed";

/// What an [`AsyncTask`] does off the event loop.
enum Work {
    Script { script: String, params: BTreeMap<String, DataValue>, immutable: bool },
    Memory { op: String, args: JsonValue },
    Ontology { op: String, args: JsonValue },
    Flow { op: String, args: JsonValue },
    Doc { op: String, args: JsonValue },
    EnableEmbedder { name: String, spec: JsonValue },
    DisableEmbedder { name: String },
    EnableDecider { name: String, spec: JsonValue },
    DisableDecider { name: String },
    EnableExtractor { name: String, spec: JsonValue },
    DisableExtractor { name: String },
}

/// One call run on the libuv thread pool; resolves to JSON text.
pub struct Job {
    db: Option<DbInstance>,
    work: Option<Work>,
}

fn str_field(spec: &JsonValue, k: &str) -> Option<String> {
    spec.get(k).and_then(|v| v.as_str()).map(|s| s.to_string())
}

fn run_work(db: Option<DbInstance>, work: Work) -> std::result::Result<JsonValue, String> {
    let need_db = || db.clone().ok_or_else(|| DB_CLOSED.to_string());
    let fmt = |e: miette::Report| e.to_string();
    match work {
        Work::Script { script, params, immutable } => {
            let db = need_db()?;
            let m = if immutable { ScriptMutability::Immutable } else { ScriptMutability::Mutable };
            db.run_script(&script, params, m).map(|rows| rows.into_json()).map_err(fmt)
        }
        Work::Memory { op, args } => need_db()?.memory_call(&op, args).map_err(fmt),
        Work::Ontology { op, args } => need_db()?.ontology_call(&op, args).map_err(fmt),
        Work::Flow { op, args } => need_db()?.flow_call(&op, args).map_err(fmt),
        Work::Doc { op, args } => {
            let db = need_db()?;
            #[cfg(feature = "docs")]
            {
                db.doc_call(&op, args).map_err(fmt)
            }
            #[cfg(not(feature = "docs"))]
            {
                let _ = (db, op, args);
                Err("documents need the native module built with the `docs` feature".to_string())
            }
        }
        Work::EnableEmbedder { name, spec } => {
            let s = kleos::embed::EmbedderSpec {
                model: str_field(&spec, "model"),
                precision: str_field(&spec, "precision"),
                device: str_field(&spec, "device"),
                dim: spec.get("dim").and_then(|v| v.as_u64()).map(|d| d as usize),
            };
            kleos::embed::enable_embedder(&name, s).map(|_| JsonValue::Null).map_err(fmt)
        }
        Work::DisableEmbedder { name } => kleos::embed::disable_embedder(&name).map(JsonValue::Bool).map_err(fmt),
        Work::EnableDecider { name, spec } => {
            let s = kleos::decide::DeciderSpec {
                model: str_field(&spec, "model"),
                path: str_field(&spec, "path"),
                precision: str_field(&spec, "precision"),
                device: str_field(&spec, "device"),
            };
            kleos::decide::enable_decider(&name, s).map(|_| JsonValue::Null).map_err(fmt)
        }
        Work::DisableDecider { name } => kleos::decide::disable_decider(&name).map(JsonValue::Bool).map_err(fmt),
        Work::EnableExtractor { name, spec } => {
            let s = kleos::extract::ExtractorSpec {
                model: str_field(&spec, "model"),
                path: str_field(&spec, "path"),
                precision: str_field(&spec, "precision"),
                device: str_field(&spec, "device"),
            };
            kleos::extract::enable_extractor(&name, s).map(|_| JsonValue::Null).map_err(fmt)
        }
        Work::DisableExtractor { name } => kleos::extract::disable_extractor(&name).map(JsonValue::Bool).map_err(fmt),
    }
}

impl Task for Job {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> Result<Self::Output> {
        let work = self.work.take().ok_or_else(|| js_err("task already ran"))?;
        match run_work(self.db.clone(), work) {
            Ok(v) => Ok(v.to_string()),
            Err(m) => Err(js_err(m)),
        }
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

fn job(db: Option<DbInstance>, work: Work) -> AsyncTask<Job> {
    AsyncTask::new(Job { db, work: Some(work) })
}

/// A kleos store opened in this process.
#[napi]
pub struct KleosDb {
    db: Arc<RwLock<Option<DbInstance>>>,
}

#[napi]
impl KleosDb {
    /// Open a store: `engine` is `mem`, `sqlite` or `rocksdb`; `path` is the
    /// file or directory (ignored for `mem`); `options` is engine options
    /// as JSON text (usually `{}`).
    #[napi(constructor)]
    pub fn new(engine: String, path: String, options: String) -> Result<Self> {
        let options = if options.trim().is_empty() { "{}".to_string() } else { options };
        let db = DbInstance::new(&engine, &path, &options).map_err(report_err)?;
        Ok(Self { db: Arc::new(RwLock::new(Some(db))) })
    }

    fn handle(&self) -> Result<DbInstance> {
        self.db.read().unwrap().clone().ok_or_else(|| js_err(DB_CLOSED))
    }

    /// Run a CozoScript query; resolves to `{headers, rows, next?}` as JSON.
    #[napi(ts_return_type = "Promise<string>")]
    pub fn run_script(&self, script: String, params_json: String, immutable: bool) -> Result<AsyncTask<Job>> {
        let params = match parse_json(&params_json, "params")? {
            JsonValue::Null => BTreeMap::new(),
            JsonValue::Object(m) => m.into_iter().map(|(k, v)| (k, DataValue::from(v))).collect(),
            _ => return Err(js_err("params must be an object")),
        };
        Ok(job(Some(self.handle()?), Work::Script { script, params, immutable }))
    }

    /// A memory-API op (`remember`, `recall`, `forget`, `reflect`, `init`,
    /// `fork`, `merge`, `reembed`, ...): `DbInstance::memory_call`.
    #[napi(ts_return_type = "Promise<string>")]
    pub fn memory_call(&self, op: String, args_json: String) -> Result<AsyncTask<Job>> {
        let args = parse_json(&args_json, "args")?;
        Ok(job(Some(self.handle()?), Work::Memory { op, args }))
    }

    /// An ontology op: `DbInstance::ontology_call`.
    #[napi(ts_return_type = "Promise<string>")]
    pub fn ontology_call(&self, op: String, args_json: String) -> Result<AsyncTask<Job>> {
        let args = parse_json(&args_json, "args")?;
        Ok(job(Some(self.handle()?), Work::Ontology { op, args }))
    }

    /// A durable-workflow op: `DbInstance::flow_call`.
    #[napi(ts_return_type = "Promise<string>")]
    pub fn flow_call(&self, op: String, args_json: String) -> Result<AsyncTask<Job>> {
        let args = parse_json(&args_json, "args")?;
        Ok(job(Some(self.handle()?), Work::Flow { op, args }))
    }

    /// A document admin op (`get`, `list`, `delete`, `reingest`): needs the
    /// `docs` feature.
    #[napi(ts_return_type = "Promise<string>")]
    pub fn doc_call(&self, op: String, args_json: String) -> Result<AsyncTask<Job>> {
        let args = parse_json(&args_json, "args")?;
        Ok(job(Some(self.handle()?), Work::Doc { op, args }))
    }

    /// Start an in-process worker that runs due built-in workflows (document
    /// ingestion, Engram jobs, schedules). `opts_json`: `{queues, pollMs,
    /// leaseMs, worker}`.
    #[napi]
    pub fn flow_worker(&self, opts_json: String) -> Result<FlowWorkerHandle> {
        let o = parse_json(&opts_json, "opts")?;
        let mut opts = FlowWorkerOpts::default();
        if let Some(q) = o.get("queues").and_then(|v| v.as_array()) {
            opts.queues = q.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect();
        }
        if let Some(p) = o.get("pollMs").and_then(|v| v.as_u64()) {
            opts.poll_ms = p;
        }
        if let Some(l) = o.get("leaseMs").and_then(|v| v.as_i64()) {
            opts.lease_ms = l;
        }
        opts.worker = o.get("worker").and_then(|v| v.as_str()).map(|s| s.to_string());
        let w = self.handle()?.flow_worker(opts);
        Ok(FlowWorkerHandle { worker: Mutex::new(Some(w)) })
    }

    /// Close the store. Calls after this reject with "database closed".
    /// Returns false when it was already closed.
    #[napi]
    pub fn close(&self) -> bool {
        self.db.write().unwrap().take().is_some()
    }
}

/// A running in-process workflow worker.
#[napi]
pub struct FlowWorkerHandle {
    worker: Mutex<Option<FlowWorker>>,
}

#[napi]
impl FlowWorkerHandle {
    /// Wake the worker now.
    #[napi]
    pub fn wake(&self) {
        if let Some(w) = self.worker.lock().unwrap().as_ref() {
            w.wake();
        }
    }
    /// Stop the worker and wait for its current tick. Idempotent.
    #[napi]
    pub fn stop(&self) {
        if let Some(w) = self.worker.lock().unwrap().take() {
            w.stop();
        }
    }
}

/// Turn the built-in or named embedder on (downloads the model on first
/// use). `spec_json`: `{model, precision, device, dim}`. Process-wide.
#[napi(ts_return_type = "Promise<string>")]
pub fn enable_embedder(name: String, spec_json: String) -> Result<AsyncTask<Job>> {
    let spec = parse_json(&spec_json, "spec")?;
    Ok(job(None, Work::EnableEmbedder { name, spec }))
}

/// Turn an embedder off and unload it. Resolves to `true`/`false` JSON.
#[napi(ts_return_type = "Promise<string>")]
pub fn disable_embedder(name: String) -> AsyncTask<Job> {
    job(None, Work::DisableEmbedder { name })
}

/// Turn a decider on. `spec_json`: `{model, path, precision, device}`.
#[napi(ts_return_type = "Promise<string>")]
pub fn enable_decider(name: String, spec_json: String) -> Result<AsyncTask<Job>> {
    let spec = parse_json(&spec_json, "spec")?;
    Ok(job(None, Work::EnableDecider { name, spec }))
}

/// Turn a decider off.
#[napi(ts_return_type = "Promise<string>")]
pub fn disable_decider(name: String) -> AsyncTask<Job> {
    job(None, Work::DisableDecider { name })
}

/// Turn an entity extractor (GLiNER) on. `spec_json`: `{model, path,
/// precision, device}`.
#[napi(ts_return_type = "Promise<string>")]
pub fn enable_extractor(name: String, spec_json: String) -> Result<AsyncTask<Job>> {
    let spec = parse_json(&spec_json, "spec")?;
    Ok(job(None, Work::EnableExtractor { name, spec }))
}

/// Turn an extractor off.
#[napi(ts_return_type = "Promise<string>")]
pub fn disable_extractor(name: String) -> AsyncTask<Job> {
    job(None, Work::DisableExtractor { name })
}

/// Features this native module was built with.
#[napi]
pub fn features() -> Vec<String> {
    let mut f = vec![];
    if cfg!(feature = "infer") {
        f.push("infer".to_string());
    }
    if cfg!(feature = "infer-metal") {
        f.push("infer-metal".to_string());
    }
    if cfg!(feature = "docs") {
        f.push("docs".to_string());
    }
    if cfg!(feature = "engram") {
        f.push("engram".to_string());
    }
    if cfg!(feature = "storage-rocksdb") {
        f.push("storage-rocksdb".to_string());
    }
    f
}
