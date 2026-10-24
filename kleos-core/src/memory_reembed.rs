/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Moving a memory store's vectors to another embedder
//! (`::memory reembed {embedder}`; docs/specs/memory-api.md, "Changing the
//! embedder").
//!
//! A store keeps vectors of one model (`MemConfig::embedder_model`). Re-embedding
//! runs as a durable workflow (`::flow`, built-in mode) so a large store
//! resumes after a crash:
//!
//! 1. Batches: each step embeds up to `batch` memories that have a vector but
//!    no shadow vector yet, with `embed('<embedder>', text)` inside the query,
//!    and puts them into a shadow relation (`kleos_mem_vec_next`). The step's
//!    writes commit with its checkpoint, so a batch is embedded once; the
//!    recorded result is only the count left.
//! 2. Switch, one transaction: embed whatever is still missing (memories
//!    remembered meanwhile), copy every shadow vector into `kleos_mem_vec`
//!    (keeping `current`), and set the store's embedder and model. Until this
//!    commits, recall and remember use the old embedder; after it, the new one.
//!    A `remember` that embedded with the old one and commits later fails its
//!    embedder assertion and plans again with the new one.
//! 3. Drop the shadow relation.
//!
//! The new embedder must have the store's vector width (EmbeddingGemma v1 →
//! v2 at 768, or both truncated to the same Matryoshka width).

use miette::{bail, Result};
use serde_derive::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};

use super::*;

/// Shadow relation the next model's vectors are built in.
pub const MEM_VEC_NEXT: &str = "kleos_mem_vec_next";
/// Name of the built-in workflow definition.
pub const REEMBED_WORKFLOW: &str = "kleos.memory.reembed";
const DEFAULT_BATCH: i64 = 64;

/// Options of `::memory reembed`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReembedOpts {
    /// Enabled embedder the store moves to (its document embedder name).
    pub embedder: String,
    /// Memories per batch step (default 64).
    pub batch: Option<i64>,
    /// Run the workflow to the end in this call (`::flow tick` until it is
    /// done) instead of leaving it to a flow worker. Default true.
    pub wait: Option<bool>,
    /// Flow queue for the workflow (default `default`).
    pub queue: Option<String>,
}

impl<'s, S: Storage<'s>> Db<S> {
    /// Start (or resume) moving the store's vectors to `opts.embedder`.
    /// Returns the workflow id and, with `wait` (default), its final status.
    pub fn memory_reembed(&'s self, opts: ReembedOpts) -> Result<JsonValue> {
        let name = opts.embedder.trim().to_string();
        if name.is_empty() {
            bail!("memory reembed needs 'embedder'");
        }
        let Some(cfg) = self.mem_config()? else {
            bail!("memory reembed: there is no memory store yet; `::memory init {{embedder: '{name}'}}` creates one")
        };
        let st = embed::embedders()
            .into_iter()
            .find(|e| e.name == name && e.enabled)
            .ok_or_else(|| miette::miette!("memory reembed: embedder '{name}' is not enabled; enable it with `::embedder enable {name}`"))?;
        let Some(dim) = cfg.dim else {
            // no vectors yet: just attach
            let mut c = cfg.clone();
            self.attach_embedder(&mut c, &name)?;
            return Ok(json!({"status": "completed", "embedded": 0, "embedder": name, "model": st.model}));
        };
        if st.dim != dim {
            bail!(
                "memory reembed: the store's vectors have dimension {dim}; embedder '{name}' has {}. Enable it with `dim: {dim}` (Matryoshka truncation) or start a new store",
                st.dim
            );
        }
        let from_model = store_vector_model(&cfg, None);
        if from_model.as_deref() == Some(st.model.as_str()) && cfg.embedder.as_deref() == Some(name.as_str()) {
            return Ok(json!({"status": "completed", "embedded": 0, "embedder": name, "model": st.model, "unchanged": true}));
        }
        let batch = opts.batch.unwrap_or(DEFAULT_BATCH).max(1);
        // one migration at a time: resume the recorded one if it targets the
        // same embedder, else start a new shadow
        let running: Option<JsonValue> = self
            .mem_q(&format!("?[v] := *{MEM_CONFIG}{{key: 'reembed', value: v}}"), vec![])?
            .rows
            .first()
            .and_then(|r| r[0].get_str().map(|s| s.to_string()))
            .and_then(|s| serde_json::from_str(&s).ok());
        let id = match running {
            Some(r) if r["embedder"] == json!(name) && r["model"] == json!(st.model) => r["id"].as_str().unwrap_or_default().to_string(),
            _ => {
                let _ = self.mem_q(&format!("::remove {MEM_VEC_NEXT}"), vec![]);
                self.mem_q(&format!(":create {MEM_VEC_NEXT} {{ns: String, id: String => vec: <F32; {dim}>}}"), vec![])?;
                let id = new_id("reembed");
                let rec = json!({"id": id, "embedder": name, "model": st.model, "from": from_model});
                self.set_mem_config("reembed", Some(rec.to_string()))?;
                id
            }
        };
        self.flow_call("define", json!({"name": REEMBED_WORKFLOW, "graph": reembed_graph()}))?;
        self.flow_call(
            "start",
            json!({
                "workflow": REEMBED_WORKFLOW,
                "id": id,
                "queue": opts.queue.clone().unwrap_or_else(|| "default".to_string()),
                "input": {"embedder": name, "model": st.model, "batch": batch},
            }),
        )?;
        if !opts.wait.unwrap_or(true) {
            return Ok(json!({"id": id, "status": "running", "embedder": name, "model": st.model}));
        }
        let queue = opts.queue.unwrap_or_else(|| "default".to_string());
        loop {
            let s = self.flow_call("status", json!({"id": id}))?;
            let status = s["status"].as_str().unwrap_or_default().to_string();
            match status.as_str() {
                "completed" => {
                    self.set_mem_config("reembed", None)?;
                    let _ = self.mem_q(&format!("::remove {MEM_VEC_NEXT}"), vec![]);
                    return Ok(json!({"id": id, "status": "completed", "embedder": name, "model": st.model, "output": s["output"]}));
                }
                "failed" | "cancelled" | "parked" => {
                    return Ok(json!({"id": id, "status": status, "error": s["error"]}));
                }
                _ => {
                    let r = self.flow_call("tick", json!({"queues": [queue], "max": 1, "schedules": false}))?;
                    if r["ran"].as_array().map_or(true, |a| a.is_empty()) {
                        // claimed by another worker or waiting on a retry timer
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                }
            }
        }
    }
}

/// Left to embed: memories with a vector and no shadow vector.
fn left_script() -> String {
    format!(
        "c[count(id)] := *{MEM_VEC}{{ns, id}}, not *{MEM_VEC_NEXT}{{ns, id}} \
         r[n] := c[n] \
         r[n] := n = 0 \
         ?[max(n)] := r[n]"
    )
}

/// Embed up to `$n` of them into the shadow relation, then count what is left.
fn batch_script() -> String {
    format!(
        "{{?[ns, id] := *{MEM_VEC}{{ns, id}}, not *{MEM_VEC_NEXT}{{ns, id}} :limit $n :replace _b {{ns: String, id: String}}}}\
         {{t[ns, id, choice(text)] := *_b{{ns, id}}, *{MEM}{{ns, id, text}} \
           ?[ns, id, vec] := t[ns, id, text], vec = embed($emb, text) \
           :put {MEM_VEC_NEXT} {{ns, id => vec}}}}\
         {{{}}}",
        left_script()
    )
}

/// The switch: catch up, copy, and set the store's embedder, in one transaction.
fn switch_script() -> String {
    format!(
        "{{?[ns, id] := *{MEM_VEC}{{ns, id}}, not *{MEM_VEC_NEXT}{{ns, id}} :replace _b {{ns: String, id: String}}}}\
         {{t[ns, id, choice(text)] := *_b{{ns, id}}, *{MEM}{{ns, id, text}} \
           ?[ns, id, vec] := t[ns, id, text], vec = embed($emb, text) \
           :put {MEM_VEC_NEXT} {{ns, id => vec}}}}\
         {{?[ns, id, current, vec] := *{MEM_VEC}{{ns, id, current}}, *{MEM_VEC_NEXT}{{ns, id, vec}} \
           :put {MEM_VEC} {{ns, id => current, vec}}}}\
         {{?[key, value] <- [['embedder', $emb], ['embedder_model', $model]] :put {MEM_CONFIG} {{key => value}}}}\
         {{c[count(id)] := *{MEM_VEC}{{ns, id}} r[n] := c[n] r[n] := n = 0 ?[max(n)] := r[n]}}"
    )
}

fn reembed_graph() -> JsonValue {
    let p = json!({"emb": "$input.embedder", "n": "$input.batch", "model": "$input.model"});
    json!({"steps": [
        {"name": "left", "query": {"script": left_script()}},
        {"loop": {"while": "$left.rows.0.0 > 0", "max": 10_000_000, "body": [
            {"name": "left", "query": {"script": batch_script(), "params": p}}
        ]}},
        {"name": "switched", "query": {"script": switch_script(), "params": p}},
        {"return": "$switched.rows.0.0"}
    ]})
}
