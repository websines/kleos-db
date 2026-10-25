/*
 * Copyright 2026, The kleos authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Skill-library benchmarks (docs/benchmarks/skills.md).
//!
//! `routing`: the routing-at-scale replication. 30 target skills, each with
//! two user tasks, plus 200 or 1,000 public distractor skills (data from
//! `scripts/skills/bench_prep.py`). Compares, at each library size:
//! - kleos `find` alone (Hit@1, Recall@5);
//! - an LLM choosing from kleos's top 5 (descriptions only);
//! - an LLM choosing from the flat list of every skill's name and
//!   description (implicit routing).
//!
//! `learn`: the learning loop and the optimizer on a toy suite (a fictional
//! rate card and code format no model knows), through the same OpenAI-
//! compatible endpoint.
//!
//! The LLM is the Synthetic API (`syn:small:text`); its key is read at run
//! time from the file `KLEOS_ENV_FILE` (`SYNTHETIC_KEY=…`), kept in this
//! process's environment only, and never printed or written.
//!
//! ```text
//! cargo run --release --features infer-metal --example skill_bench -- routing <data dir> <out.json>
//! cargo run --release --features infer-metal --example skill_bench -- learn <data dir> <out.json>
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use kleos::memory::consolidate::LlmOpts;
use kleos::DbInstance;
use serde_json::{json, Value};

const MODEL: &str = "syn:small:text";
const URL: &str = "https://api.synthetic.new/openai/v1";

fn load_key() {
    let path = std::env::var("KLEOS_ENV_FILE").expect("set KLEOS_ENV_FILE to the .env holding SYNTHETIC_KEY");
    let text = std::fs::read_to_string(&path).expect("read KLEOS_ENV_FILE");
    for l in text.lines() {
        if let Some(v) = l.strip_prefix("SYNTHETIC_KEY=") {
            std::env::set_var("SYNTHETIC_KEY", v.trim().trim_matches('"').trim_matches('\''));
            return;
        }
    }
    panic!("SYNTHETIC_KEY not found in KLEOS_ENV_FILE");
}

fn llm() -> LlmOpts {
    LlmOpts { url: URL.into(), model: MODEL.into(), api_key_env: Some("SYNTHETIC_KEY".into()), max_tokens: Some(16000) }
}

fn call(db: &DbInstance, op: &str, args: Value) -> Value {
    db.skill_call(op, args.clone()).unwrap_or_else(|e| panic!("skill {op}: {e:?}"))
}

fn open_store(dir: &Path, name: &str) -> DbInstance {
    kleos::embed::enable_embedder("gemma", kleos::embed::EmbedderSpec::parse_cli("gemma=embeddinggemma").unwrap().1).unwrap();
    let db = DbInstance::new("sqlite", dir.join(name), "").unwrap();
    call(&db, "init", json!({"embedder": "gemma"}));
    db
}

/// What the LLM answered when asked which skill to load.
#[derive(Clone, Debug)]
enum Pick {
    Skill(String),
    /// it answered NONE (or a name not in the list)
    NoSkill,
    /// no answer: the request failed, or the reasoning used every token
    NoAnswer,
}

impl Pick {
    fn name(&self) -> Option<&str> {
        match self {
            Pick::Skill(n) => Some(n),
            _ => None,
        }
    }
    fn label(&self) -> Value {
        match self {
            Pick::Skill(n) => json!(n),
            Pick::NoSkill => json!("NONE"),
            Pick::NoAnswer => json!("(no answer)"),
        }
    }
}

/// Ask the LLM which skill to load from `skills` (name, description).
fn choose(l: &LlmOpts, task: &str, skills: &[(String, String)]) -> Pick {
    let list: String = skills.iter().map(|(n, d)| format!("- {n}: {d}\n")).collect();
    let user = format!("Available skills:\n{list}\nTask: {task}\n\nWhich ONE skill should be loaded for this task? Reply with its exact name only, or NONE if no skill fits.");
    let system = "You route an AI agent's tasks to skills (packaged procedures). Pick the single best skill for the task from the list, by name.";
    for _ in 0..2 {
        let Ok((text, _)) = kleos::skill_learn::llm_chat(l, system, &user, 0.0) else { continue };
        let t = text.trim().trim_matches('`').trim().to_lowercase();
        if t.is_empty() {
            continue;
        }
        let first = t.lines().rev().find(|x| !x.trim().is_empty()).unwrap_or("").trim().trim_matches(|c: char| !c.is_alphanumeric() && c != '-').to_string();
        if first == "none" {
            return Pick::NoSkill;
        }
        if skills.iter().any(|(n, _)| *n == first) {
            return Pick::Skill(first);
        }
        let mut names: Vec<&String> = skills.iter().map(|(n, _)| n).collect();
        names.sort_by_key(|n| std::cmp::Reverse(n.len()));
        return names.into_iter().find(|n| t.contains(n.as_str())).map(|n| Pick::Skill(n.clone())).unwrap_or(Pick::NoSkill);
    }
    Pick::NoAnswer
}

fn par_map<T: Sync, R: Send>(items: &[T], threads: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let per = items.len().div_ceil(threads.max(1)).max(1);
    let mut out: Vec<Option<R>> = (0..items.len()).map(|_| None).collect();
    std::thread::scope(|s| {
        let hs: Vec<_> = items.chunks(per).enumerate().map(|(ci, ch)| {
            let f = &f;
            s.spawn(move || ch.iter().enumerate().map(|(i, x)| (ci * per + i, f(x))).collect::<Vec<_>>())
        }).collect();
        for h in hs {
            for (i, r) in h.join().unwrap() {
                out[i] = Some(r);
            }
        }
    });
    out.into_iter().map(|x| x.unwrap()).collect()
}

fn routing(data: &Path, out: &Path) {
    let targets: Vec<String> = serde_json::from_str(&std::fs::read_to_string(data.join("targets.json")).unwrap()).unwrap();
    let distractors: Vec<String> = serde_json::from_str(&std::fs::read_to_string(data.join("distractors.json")).unwrap()).unwrap();
    let tasks: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(data.join("tasks.json")).unwrap()).unwrap();
    let db = open_store(data, "routing.db");
    // targets in `default`, the first 200 distractors in `d1`, the rest in `d2`:
    // one index, three library sizes by scope filter
    let have = call(&db, "list", json!({"scopes": ["default", "d1", "d2"]})).as_array().unwrap().len();
    let t0 = Instant::now();
    let mut quarantined = vec![];
    if have == 0 {
        let groups: Vec<(&str, &[String])> = vec![("default", &targets[..]), ("d1", &distractors[..200]), ("d2", &distractors[200..])];
        for (scope, names) in groups {
            for (i, n) in names.iter().enumerate() {
                let r = call(&db, "import", json!({"path": data.join("pool").join(n), "scope": scope, "activate": true}));
                if r["imported"][0]["status"] == "quarantined" {
                    quarantined.push(json!({"name": n, "scope": scope, "findings": r["imported"][0]["scan"]["findings"].as_array().map(|f| f.iter().map(|x| x["rule"].clone()).collect::<Vec<_>>())}));
                }
                if i % 100 == 0 {
                    eprintln!("{scope}: {i}/{} ({:.0}s)", names.len(), t0.elapsed().as_secs_f64());
                }
            }
        }
    }
    let import_s = t0.elapsed().as_secs_f64();
    let mut sizes: Vec<(&str, Vec<&str>)> = vec![("30", vec!["default"]), ("230", vec!["default", "d1"]), ("1030", vec!["default", "d1", "d2"])];
    if let Ok(only) = std::env::var("BENCH_SIZES") {
        sizes.retain(|(l, _)| only.split(',').any(|x| x == *l));
    }
    let use_llm = std::env::var("KLEOS_ENV_FILE").is_ok();
    if use_llm {
        load_key();
    }
    let l = llm();
    let mut summary = serde_json::Map::new();
    let mut records = vec![];
    for (label, scopes) in &sizes {
        let active: Vec<(String, String)> = call(&db, "list", json!({"scopes": scopes, "status": "active"}))
            .as_array()
            .unwrap()
            .iter()
            .map(|s| (s["name"].as_str().unwrap().to_string(), s["description"].as_str().unwrap().to_string()))
            .collect();
        let mut lat = vec![];
        let found: Vec<Value> = tasks
            .iter()
            .map(|t| {
                let q = Instant::now();
                let r = call(&db, "find", json!({"task": t["task"], "scopes": scopes, "k": 10, "budget": 100000, "log": false}));
                lat.push(q.elapsed().as_secs_f64() * 1000.0);
                r
            })
            .collect();
        let ranks: Vec<Option<usize>> = tasks
            .iter()
            .zip(&found)
            .map(|(t, f)| f["skills"].as_array().unwrap().iter().position(|s| s["name"] == t["target"]))
            .collect();
        let n = tasks.len() as f64;
        let hit1 = ranks.iter().filter(|r| **r == Some(0)).count() as f64 / n;
        let r5 = ranks.iter().filter(|r| r.is_some_and(|x| x < 5)).count() as f64 / n;
        let r10 = ranks.iter().filter(|r| r.is_some()).count() as f64 / n;
        let mrr = ranks.iter().map(|r| r.map_or(0.0, |x| 1.0 / (x + 1) as f64)).sum::<f64>() / n;
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut s = json!({"library": active.len(), "kleos_hit1": hit1, "kleos_recall5": r5, "kleos_recall10": r10, "kleos_mrr": mrr,
                           "find_ms_p50": lat[lat.len() / 2], "find_ms_p95": lat[lat.len() * 95 / 100]});
        if use_llm {
            // the flat list: every skill's name and description, in a fixed shuffled order
            let mut flat = active.clone();
            flat.sort_by_key(|(n, _)| kleos::skill::digest_of(n.as_bytes()));
            let flat_tokens: usize = flat.iter().map(|(n, d)| kleos::skill::est_tokens(n) + kleos::skill::est_tokens(d) + 3).sum();
            let picks = par_map(&tasks, 4, |t| choose(&l, t["task"].as_str().unwrap(), &flat));
            let ok = tasks.iter().zip(&picks).filter(|(t, p)| p.name() == t["target"].as_str()).count() as f64 / n;
            s["flat_llm_accuracy"] = json!(ok);
            s["flat_llm_none"] = json!(picks.iter().filter(|p| matches!(p, Pick::NoSkill)).count() as f64 / n);
            s["flat_llm_no_answer"] = json!(picks.iter().filter(|p| matches!(p, Pick::NoAnswer)).count() as f64 / n);
            s["flat_list_tokens"] = json!(flat_tokens);
            let kpicks: Option<Vec<Pick>> = (*label != "230").then(|| {
                par_map(&found.iter().zip(&tasks).collect::<Vec<_>>(), 4, |(f, t)| {
                    let top: Vec<(String, String)> = f["skills"].as_array().unwrap().iter().take(5)
                        .map(|s| (s["name"].as_str().unwrap().to_string(), s["description"].as_str().unwrap().to_string())).collect();
                    choose(&l, t["task"].as_str().unwrap(), &top)
                })
            });
            if let Some(kp) = &kpicks {
                s["kleos_top5_llm_accuracy"] = json!(tasks.iter().zip(kp).filter(|(t, p)| p.name() == t["target"].as_str()).count() as f64 / n);
                s["kleos_top5_llm_none"] = json!(kp.iter().filter(|p| matches!(p, Pick::NoSkill)).count() as f64 / n);
                s["kleos_top5_llm_no_answer"] = json!(kp.iter().filter(|p| matches!(p, Pick::NoAnswer)).count() as f64 / n);
            }
            for (i, t) in tasks.iter().enumerate() {
                records.push(json!({"size": label, "target": t["target"], "task": t["task"], "kleos_rank": ranks[i].map(|r| r + 1),
                    "kleos_top": found[i]["skills"][0]["name"], "flat_pick": picks[i].label(), "kleos_llm_pick": kpicks.as_ref().map(|k| k[i].label())}));
            }
        } else {
            for (i, t) in tasks.iter().enumerate() {
                records.push(json!({"size": label, "target": t["target"], "kleos_rank": ranks[i].map(|r| r + 1), "kleos_top": found[i]["skills"][0]["name"]}));
            }
        }
        eprintln!("{label}: {s}");
        summary.insert(label.to_string(), s);
    }
    let res = json!({"benchmark": "skills-routing", "tasks": tasks.len(), "targets": targets.len(), "model": MODEL, "embedder": "embeddinggemma (built-in)",
                     "import_s": import_s, "quarantined": quarantined, "summary": summary, "records": records});
    std::fs::write(out, serde_json::to_string_pretty(&res).unwrap()).unwrap();
}

/// Toy suite: two task families whose rules no model knows.
fn shipping(rng: &mut u64, i: usize) -> (String, String, String) {
    let mut next = || {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*rng >> 33) as u32
    };
    let w = 1 + next() % 9; // whole kg, 1..9
    let frac = [0.0, 0.3, 0.6][(next() % 3) as usize];
    let weight = w as f64 + frac;
    let express = next() % 2 == 0;
    let total = [40.0, 85.0, 120.0, 250.0][(next() % 4) as usize];
    let kg = weight.ceil();
    let mut cost = 4.50 + 1.25 * kg + if express { 3.00 } else { 0.0 };
    if total > 100.0 {
        cost *= 0.9;
    }
    let ans = format!("USD {cost:.2}");
    let task = format!(
        "Order #{}: a {weight:.1} kg parcel, order total USD {total:.0}, {} delivery. What does Acme charge for shipping?",
        1000 + i,
        if express { "express" } else { "standard" }
    );
    let transcript = format!(
        "Looked up the Acme rate card: base USD 4.50, plus USD 1.25 per kg with the weight rounded UP to the next whole kg, plus USD 3.00 for express; \
         orders over USD 100 get 10% off the shipping charge. {weight:.1} kg -> {kg:.0} kg. 4.50 + 1.25*{kg:.0}{} = {:.2}{}. Answer {ans}.",
        if express { " + 3.00" } else { "" },
        4.50 + 1.25 * kg + if express { 3.0 } else { 0.0 },
        if total > 100.0 { ", then 10% off" } else { "" }
    );
    (task, transcript, ans)
}

fn sku(i: usize) -> (String, String, String) {
    let items = [("Blue", "Desk Lamp", 2024, 7), ("Red", "Office Chair", 2023, 12), ("Green", "Wall Clock", 2025, 3), ("Black", "Coffee Mug", 2022, 45),
                 ("White", "Floor Rug", 2026, 9), ("Grey", "Book Shelf", 2021, 18), ("Yellow", "Table Fan", 2024, 27), ("Pink", "Pillow Case", 2025, 61)];
    let (color, product, year, batch) = items[i % items.len()];
    let pw: Vec<&str> = product.split(' ').collect();
    let code = format!("{}{}-{}{}-{:02}-B{batch:03}", &pw[0][..2].to_uppercase(), &pw[1][..2].to_uppercase(), &color[..1], color.len(), year % 100);
    let task = format!("Make the Acme SKU for the {color} {product}, year {year}, batch {batch}.");
    let transcript = format!(
        "Acme SKU format: first two letters of each word of the product name, uppercase; dash; first letter of the color then the number of letters in the color; \
         dash; two-digit year; dash; B and the batch number padded to three digits. Got {code}."
    );
    (task, transcript, code)
}

fn learn(data: &Path, out: &Path) {
    load_key();
    let dir = data.join("learn");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let db = open_store(&dir, "learn.db");
    let l = llm();
    call(&db, "config", json!({"learn": {"llm": {"url": l.url, "model": l.model, "api_key_env": "SYNTHETIC_KEY", "max_tokens": 16000},
                                         "min_support": 3, "margin": 0.2, "cron": "off", "parallel": 4, "holdout": 0.5, "cluster_threshold": 0.75}}));
    let mut rng = 42u64;
    for i in 0..8 {
        let (task, tr, ans) = shipping(&mut rng, i);
        call(&db, "record_trace", json!({"task": task, "transcript": tr, "output": ans, "success": true}));
    }
    for i in 0..6 {
        let (task, tr, ans) = sku(i);
        call(&db, "record_trace", json!({"task": task, "transcript": tr, "output": ans, "success": true}));
    }
    call(&db, "record_trace", json!({"task": "Order #2001: a 2.4 kg parcel, order total USD 60, standard delivery. What does Acme charge for shipping?",
        "transcript": "Used 2.4 kg as is: 4.50 + 1.25*2.4 = 7.50.", "output": "USD 7.50", "success": false}));
    let t0 = Instant::now();
    let learned = call(&db, "learn", json!({"wait": true}));
    let learn_s = t0.elapsed().as_secs_f64();
    eprintln!("learn: {}", learned["admit"]);

    // optimizer 1: the learned shipping skill, on a larger held-out set
    let learned_ship = learned["admit"]["results"].as_array().into_iter().flatten()
        .find(|r| r["admitted"] == true && r["name"].as_str().is_some_and(|n| n.contains("ship")))
        .and_then(|r| r["name"].as_str().map(String::from));
    let mut opt_learned = Value::Null;
    if let Some(name) = &learned_ship {
        let mut rng = 99u64;
        let cases: Vec<Value> = (0..12)
            .map(|i| {
                let (task, _, ans) = shipping(&mut rng, 200 + i);
                json!({"task": task, "expected": ans, "split": if i % 2 == 0 { "holdout" } else { "train" }})
            })
            .collect();
        call(&db, "eval_add", json!({"name": name, "cases": cases}));
        let t = Instant::now();
        let o = call(&db, "optimize", json!({"name": name, "rounds": 4, "wait": true}));
        let live = call(&db, "get", json!({"name": name}));
        eprintln!("optimize learned: {}", o["finish"]);
        opt_learned = json!({"name": name, "seconds": t.elapsed().as_secs_f64(), "finish": o["finish"], "prepare": o["prepare"], "rounds": o["rounds"], "final_body": live["body"], "final_origin": live["origin"]});
    }

    // optimizer 2: a hand-written shipping skill that misses three rules
    let weak = "---\nname: acme-shipping-manual\ndescription: Compute the Acme shipping charge for an order. Use when asked what Acme charges for shipping.\n---\n\
## Steps\n1. Start from the base charge of USD 4.50.\n2. Add USD 1.25 per kg of parcel weight.\n3. Give the result as `USD x.xx`.\n";
    call(&db, "put", json!({"skill_md": weak, "activate": true}));
    let mut rng = 7u64;
    let cases: Vec<Value> = (0..12)
        .map(|i| {
            let (task, _, ans) = shipping(&mut rng, 100 + i);
            json!({"task": task, "expected": ans, "split": if i % 2 == 0 { "holdout" } else { "train" }})
        })
        .collect();
    call(&db, "eval_add", json!({"name": "acme-shipping-manual", "cases": cases}));
    let t1 = Instant::now();
    let opt = call(&db, "optimize", json!({"name": "acme-shipping-manual", "rounds": 4, "wait": true}));
    let opt_s = t1.elapsed().as_secs_f64();
    eprintln!("optimize: {}", opt["finish"]);
    let live = call(&db, "get", json!({"name": "acme-shipping-manual"}));
    let learned_skills: Vec<Value> = call(&db, "list", json!({})).as_array().unwrap().iter().filter(|s| s["name"] != "acme-shipping-manual").cloned().collect();
    let mut bodies = BTreeMap::new();
    for s in &learned_skills {
        let g = call(&db, "get", json!({"name": s["name"]}));
        bodies.insert(s["name"].as_str().unwrap().to_string(), json!({"status": s["status"], "origin": g["origin"], "body": g["body"]}));
    }
    let res = json!({"benchmark": "skills-learn", "model": MODEL, "learn_s": learn_s, "optimize_s": opt_s,
                     "learn": {"harvest": learned["harvest"]["stats"], "results": learned["admit"]["results"], "verify": learned["verify"]},
                     "learned_skills": bodies,
                     "optimize_learned": opt_learned,
                     "optimize": {"finish": opt["finish"], "rounds": opt["rounds"], "base_body": weak, "final_body": live["body"], "final_origin": live["origin"]}});
    std::fs::write(out, serde_json::to_string_pretty(&res).unwrap()).unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (mode, data, out) = (a.get(1).map(String::as_str).unwrap_or("routing"), PathBuf::from(&a[2]), PathBuf::from(&a[3]));
    match mode {
        "routing" => routing(&data, &out),
        "learn" => learn(&data, &out),
        m => panic!("unknown mode {m}"),
    }
}
