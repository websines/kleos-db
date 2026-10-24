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
//! A store keeps vectors of one model (`MemConfig::embedder_model`).
//! Re-embedding is a durable workflow (`::flow`, built-in mode) so a large
//! store resumes after a crash:
//!
//! 1. Shards: the memory ids are cut into ranges of about `batch` ids when the
//!    re-embed starts. One `query` step per range embeds that range's
//!    memories with `embed('<embedder>', text)` inside the query and puts them
//!    into a shadow relation (`kleos_mem_vec_next`); the step's writes commit
//!    with its checkpoint, so a range is embedded once, and the history only
//!    records the step, not the vectors.
//! 2. Switch, one transaction (run by whoever next touches the store once the
//!    workflow is done: the waiting `reembed` call, or the next `remember` /
//!    `recall`): embed whatever is still missing (memories remembered
//!    meanwhile), copy every shadow vector into `kleos_mem_vec` keeping its
//!    `current` flag, and set the store's embedder and model. Before it
//!    commits, recall and remember use the old embedder; after it, the new
//!    one. A `remember` that embedded with the old one and commits later fails
//!    its embedder assertion and plans again with the new one.
//! 3. The shadow relation is dropped.
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
const DEFAULT_BATCH: usize = 64;
/// Upper bound of the last id range (ids are ASCII and sort below it).
const ID_MAX: &str = "\u{10FFFF}";

/// Options of `::memory reembed`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReembedOpts {
    /// Enabled embedder the store moves to (its document embedder name).
    pub embedder: String,
    /// Memories per step (default 64).
    pub batch: Option<usize>,
    /// Run the workflow to the end in this call (`::flow tick` until it is
    /// done) and switch, instead of leaving it to a flow worker (the switch
    /// then happens on the next `remember` / `recall` after the workflow
    /// completes). Default true.
    pub wait: Option<bool>,
    /// Flow queue for the workflow (default `default`).
    pub queue: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Record {
    id: String,
    embedder: String,
    model: String,
    from: Option<String>,
}

impl<'s, S: Storage<'s>> Db<S> {
    /// Start (or resume) moving the store's vectors to `opts.embedder`.
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
        let running: Option<Record> = cfg.reembed.as_deref().and_then(|s| serde_json::from_str(s).ok());
        if running.is_none() && from_model.as_deref() == Some(st.model.as_str()) && cfg.embedder.as_deref() == Some(name.as_str()) {
            return Ok(json!({"status": "completed", "embedded": 0, "embedder": name, "model": st.model, "unchanged": true}));
        }
        let queue = opts.queue.clone().unwrap_or_else(|| "default".to_string());
        // one migration at a time: resume the recorded one if it targets the
        // same model, else start over with a fresh shadow relation
        let rec = match running {
            Some(r) if r.embedder == name && r.model == st.model => r,
            _ => {
                let _ = self.mem_q(&format!("::remove {MEM_VEC_NEXT}"), vec![]);
                self.mem_q(&format!(":create {MEM_VEC_NEXT} {{ns: String, id: String => vec: <F32; {dim}>}}"), vec![])?;
                let shards = self.reembed_shards(opts.batch.unwrap_or(DEFAULT_BATCH).max(1))?;
                let rec = Record {
                    id: new_id("reembed"),
                    embedder: name.clone(),
                    model: st.model.clone(),
                    from: from_model,
                };
                self.flow_call("define", json!({"name": REEMBED_WORKFLOW, "graph": reembed_graph()}))?;
                self.flow_call(
                    "start",
                    json!({
                        "workflow": REEMBED_WORKFLOW,
                        "id": rec.id,
                        "queue": queue,
                        "input": {"embedder": name, "shards": shards},
                    }),
                )?;
                self.set_mem_config("reembed", Some(serde_json::to_string(&rec).unwrap_or_default()))?;
                rec
            }
        };
        if !opts.wait.unwrap_or(true) {
            return Ok(json!({"id": rec.id, "status": "running", "embedder": name, "model": st.model}));
        }
        loop {
            let s = self.flow_call("status", json!({"id": rec.id}))?;
            match s["status"].as_str().unwrap_or_default() {
                "completed" => {
                    let n = self.reembed_switch(&rec)?;
                    return Ok(json!({"id": rec.id, "status": "completed", "embedder": name, "model": st.model, "embedded": n}));
                }
                st @ ("failed" | "cancelled" | "parked") => {
                    return Ok(json!({"id": rec.id, "status": st, "error": s["error"]}));
                }
                _ => {
                    let r = self.flow_call("tick", json!({"queues": [queue], "max": 1, "schedules": false}))?;
                    if r["ran"].as_array().map_or(true, |a| a.is_empty()) {
                        // held by another worker (or its lease is running out)
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                }
            }
        }
    }

    /// The switch, if the store's re-embed workflow has completed: run by
    /// `remember` / `recall` so a re-embed left to flow workers finishes on
    /// the next use. Returns the (possibly updated) configuration.
    pub(crate) fn reembed_finish_if_done(&'s self, cfg: MemConfig) -> Result<MemConfig> {
        let Some(rec) = cfg.reembed.as_deref().and_then(|s| serde_json::from_str::<Record>(s).ok()) else {
            return Ok(cfg);
        };
        let s = self.flow_call("status", json!({"id": rec.id}))?;
        if s["status"].as_str() != Some("completed") || !embed::is_embedder_enabled(&rec.embedder) {
            return Ok(cfg);
        }
        self.reembed_switch(&rec)?;
        Ok(self.mem_config()?.unwrap_or(cfg))
    }

    /// Id ranges `[lo, hi)` of about `batch` memories each.
    fn reembed_shards(&'s self, batch: usize) -> Result<Vec<JsonValue>> {
        let ids: Vec<String> = self
            .mem_q(&format!("?[id] := *{MEM_VEC}{{id}} :order id"), vec![])?
            .rows
            .into_iter()
            .filter_map(|r| r[0].get_str().map(|s| s.to_string()))
            .collect();
        let mut out = vec![];
        let mut k = 0;
        while k < ids.len() {
            let lo = if k == 0 { String::new() } else { ids[k].clone() };
            let hi = ids.get(k + batch).cloned().unwrap_or_else(|| ID_MAX.to_string());
            out.push(json!([lo, hi]));
            k += batch;
        }
        Ok(out)
    }

    /// Catch up, copy, set the embedder, in one transaction; then drop the
    /// shadow relation. Idempotent. Returns the number of vectors.
    fn reembed_switch(&'s self, rec: &Record) -> Result<i64> {
        let n = self
            .mem_q(
                &format!(
                    "{{t[ns, id, choice(text)] := *{MEM_VEC}{{ns, id}}, not *{MEM_VEC_NEXT}{{ns, id}}, *{MEM}{{ns, id, text}} \
                       ?[ns, id, vec] := t[ns, id, text], vec = embed($emb, text) \
                       :put {MEM_VEC_NEXT} {{ns, id => vec}}}}\
                     {{?[ns, id, current, vec] := *{MEM_VEC}{{ns, id, current}}, *{MEM_VEC_NEXT}{{ns, id, vec}} \
                       :put {MEM_VEC} {{ns, id => current, vec}}}}\
                     {{?[key, value] <- [['embedder', $emb], ['embedder_model', $model]] :put {MEM_CONFIG} {{key => value}}}}\
                     {{?[key] <- [['reembed']] :rm {MEM_CONFIG} {{key}}}}\
                     {{c[count(id)] := *{MEM_VEC}{{id}} r[n] := c[n] r[n] := n = 0 ?[max(n)] := r[n]}}"
                ),
                vec![("emb", DataValue::from(rec.embedder.as_str())), ("model", DataValue::from(rec.model.as_str()))],
            )?
            .rows
            .first()
            .and_then(|r| r[0].get_int())
            .unwrap_or(0);
        let _ = self.mem_q(&format!("::remove {MEM_VEC_NEXT}"), vec![]);
        Ok(n)
    }
}

/// One step per id range: embed that range's memories into the shadow
/// relation (a single program, as a flow `query` step requires).
fn shard_script() -> String {
    format!(
        "t[ns, id, choice(text)] := *{MEM_VEC}{{ns, id}}, id >= $lo, id < $hi, not *{MEM_VEC_NEXT}{{ns, id}}, *{MEM}{{ns, id, text}} \
         ?[ns, id, vec] := t[ns, id, text], vec = embed($emb, text) \
         :put {MEM_VEC_NEXT} {{ns, id => vec}}"
    )
}

fn reembed_graph() -> JsonValue {
    json!({"steps": [
        {"for_each": {"in": "$input.shards", "as": "sh", "body": [
            {"query": {"script": shard_script(), "params": {"emb": "$input.embedder", "lo": "$sh.0", "hi": "$sh.1"}}}
        ]}},
        {"return": "true"}
    ]})
}
