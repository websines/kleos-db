/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The memory API's Engram hook (docs/specs/engram-overlay.md, "Memory
//! attachment"): with a model attached, `remember` / `forget` enqueue
//! Engram jobs in their own write transaction. Model-free: the model is
//! registered by fingerprint, supersession comes from the value rule, and
//! no worker runs (the worker is `scripts/engram/engram_worker.py`).

use std::collections::{BTreeMap, BTreeSet};

use kleos::memory::{ForgetOpts, InitOpts, RememberOpts};
use kleos::{DataValue, DbInstance, ScriptMutability};

const DAY: i64 = 86_400_000_000;
const T0: i64 = 1_767_225_600_000_000; // 2026-01-01T00:00:00Z

fn d(n: i64) -> i64 {
    T0 + n * DAY
}

fn q(db: &DbInstance, script: &str) -> Vec<Vec<DataValue>> {
    db.run_script(script, Default::default(), ScriptMutability::Mutable)
        .unwrap_or_else(|e| panic!("{script}: {e:?}"))
        .rows
}

fn ns(s: &str) -> Option<String> {
    Some(s.to_string())
}

fn strs(v: &DataValue) -> Vec<String> {
    match v {
        DataValue::List(l) => l.iter().map(|x| x.get_str().unwrap().to_string()).collect(),
        _ => vec![],
    }
}

/// A store attached to a model registered by fingerprint (no text hashing
/// needed: jobs only carry memory ids and texts).
fn attached() -> DbInstance {
    let db = DbInstance::new("mem", "", "").unwrap();
    q(&db, "::engram register nano {fingerprint: 'eg1:00112233445566778899aabbccddeeff', row_dim: 4, layers: [1]}");
    let cfg = db
        .memory_init(InitOpts { engram: Some("nano".into()), ..Default::default() })
        .unwrap();
    assert_eq!(cfg.engram.as_deref(), Some("nano"));
    db
}

/// `(qid, op, mem_ids, old, text, at, status)` of a namespace's jobs, by qid.
#[allow(clippy::type_complexity)]
fn jobs(db: &DbInstance, n: &str) -> Vec<(i64, String, Vec<String>, Vec<String>, Option<String>, i64, String)> {
    let mut out: Vec<_> = db
        .run_script(
            "?[qid, op, mem_ids, old, text, at, status] := *kleos_engram_queue{ns: $ns, qid, op, mem_ids, old, text, at, status}",
            BTreeMap::from([("ns".to_string(), DataValue::from(n))]),
            ScriptMutability::Immutable,
        )
        .unwrap()
        .rows
        .into_iter()
        .map(|r| {
            (
                r[0].get_int().unwrap(),
                r[1].get_str().unwrap().to_string(),
                strs(&r[2]),
                strs(&r[3]),
                r[4].get_str().map(str::to_string),
                r[5].get_int().unwrap(),
                r[6].get_str().unwrap().to_string(),
            )
        })
        .collect();
    out.sort();
    out
}

fn ns_seq(db: &DbInstance, n: &str) -> i64 {
    db.run_script(
        "?[seq] := *kleos_mem_ns{ns: $ns, seq}",
        BTreeMap::from([("ns".to_string(), DataValue::from(n))]),
        ScriptMutability::Immutable,
    )
    .unwrap()
    .rows
    .first()
    .and_then(|r| r[0].get_int())
    .unwrap_or(0)
}

#[test]
fn unattached_store_enqueues_nothing() {
    let db = DbInstance::new("mem", "", "").unwrap();
    q(&db, "::engram register nano {fingerprint: 'eg1:00112233445566778899aabbccddeeff', row_dim: 4, layers: [1]}");
    let cfg = db.memory_init(InitOpts::default()).unwrap();
    assert_eq!(cfg.engram, None);
    let a = db.remember("My rent is $1,800 per month.", RememberOpts { ns: ns("u"), at: Some(d(1)), ..Default::default() }).unwrap();
    db.remember("My rent is $2,100 per month.", RememberOpts { ns: ns("u"), at: Some(d(2)), ..Default::default() }).unwrap();
    db.forget(ForgetOpts { ns: ns("u"), id: a.id, mode: Some("erase".into()), ..Default::default() }).unwrap();
    // no queue relation at all, and the init output is unchanged
    assert!(db.run_script("?[n] := *kleos_engram_queue{ns: n}", Default::default(), ScriptMutability::Immutable).is_err());
    let init = db.memory_call("init", serde_json::json!({})).unwrap();
    assert!(init.get("engram").is_none(), "{init}");
    // ::engram queue on a store that never attached is empty
    assert!(q(&db, "::engram queue {}").is_empty());
}

#[test]
fn attach_checks_the_model_and_detaches() {
    let db = DbInstance::new("mem", "", "").unwrap();
    let err = db.memory_init(InitOpts { engram: Some("nope".into()), ..Default::default() }).unwrap_err();
    assert!(format!("{err:?}").contains("nope"), "{err:?}");
    let db = attached();
    // the attachment is stored: a fresh config read sees it
    let cfg = db.memory_call("init", serde_json::json!({})).unwrap();
    assert_eq!(cfg["engram"], "nano");
    db.remember("I like tea.", RememberOpts { ns: ns("u"), ..Default::default() }).unwrap();
    assert_eq!(jobs(&db, "u").len(), 1);
    // CozoScript surface, then detach
    q(&db, "::memory init {engram: ''}");
    db.remember("I like coffee.", RememberOpts { ns: ns("u"), ..Default::default() }).unwrap();
    assert_eq!(jobs(&db, "u").len(), 1, "detached: no new job");
    q(&db, "::memory init {engram: 'nano'}");
    db.remember("I like cocoa.", RememberOpts { ns: ns("u"), ..Default::default() }).unwrap();
    assert_eq!(jobs(&db, "u").len(), 2);
}

#[test]
fn remember_supersede_duplicate_and_forget_enqueue_in_commit_order() {
    let db = attached();
    let a = db
        .remember("My rent is $1,800 per month.", RememberOpts { ns: ns("u"), at: Some(d(1)), entities: vec!["rent".into()], ..Default::default() })
        .unwrap();
    let a_id = a.id.clone().unwrap();
    let j = jobs(&db, "u");
    assert_eq!(j.len(), 1);
    assert_eq!((j[0].1.as_str(), &j[0].2, j[0].4.as_deref(), j[0].5, j[0].6.as_str()),
               ("write", &vec![a_id.clone()], Some("My rent is $1,800 per month."), d(1), "pending"));
    assert_eq!(j[0].0, ns_seq(&db, "u"), "qid is the namespace version of the commit");
    let meta = q(&db, "?[m] := *kleos_engram_queue{ns: 'u', meta: m}");
    match &meta[0][0] {
        DataValue::Json(j) => assert_eq!(j.0["entities"][0], "rent"),
        other => panic!("meta: {other:?}"),
    }

    // exact duplicate: a source row, no job
    let dup = db.remember("my rent is $1,800 per month", RememberOpts { ns: ns("u"), at: Some(d(2)), ..Default::default() }).unwrap();
    assert_eq!(dup.action, "duplicate");
    assert_eq!(jobs(&db, "u").len(), 1);

    // value-rule supersede: one supersede job old -> new, at the new memory's valid-from
    let b = db.remember("My rent is $2,100 per month.", RememberOpts { ns: ns("u"), at: Some(d(5)), ..Default::default() }).unwrap();
    assert_eq!(b.superseded, vec![a_id.clone()]);
    let b_id = b.id.clone().unwrap();
    let c = db.remember("I have a cat named Tom.", RememberOpts { ns: ns("u"), at: Some(d(6)), ..Default::default() }).unwrap();
    let c_id = c.id.clone().unwrap();
    let j = jobs(&db, "u");
    assert_eq!(j.len(), 3);
    assert_eq!((j[1].1.as_str(), &j[1].2, &j[1].3, j[1].5), ("supersede", &vec![b_id.clone()], &vec![a_id.clone()], d(5)));
    assert_eq!((j[2].1.as_str(), &j[2].2), ("write", &vec![c_id.clone()]));

    // forget retract: the current ids only, at the retraction time
    let r = db.forget(ForgetOpts { ns: ns("u"), ids: vec![a_id.clone(), c_id.clone()], at: Some(d(7)), ..Default::default() }).unwrap();
    assert_eq!(r.ids, vec![c_id.clone()]);
    // forget erase: every id that exists; earlier jobs lose the erased texts
    db.forget(ForgetOpts { ns: ns("u"), id: Some(c_id.clone()), mode: Some("erase".into()), ..Default::default() }).unwrap();
    let j = jobs(&db, "u");
    assert_eq!(j.len(), 5);
    assert_eq!((j[3].1.as_str(), &j[3].2, j[3].5), ("retract", &vec![c_id.clone()], d(7)));
    assert_eq!((j[4].1.as_str(), &j[4].2), ("erase", &vec![c_id.clone()]));
    assert_eq!(j[2].4, None, "the erased memory's text is gone from its write job");
    assert_eq!(j[1].4.as_deref(), Some("My rent is $2,100 per month."), "other jobs keep theirs");
    // qids strictly increase in commit order and end at the namespace version
    assert!(j.windows(2).all(|w| w[0].0 < w[1].0));
    assert_eq!(j.last().unwrap().0, ns_seq(&db, "u"));
    // forgetting nothing enqueues nothing
    db.forget(ForgetOpts { ns: ns("u"), id: Some(c_id), ..Default::default() }).unwrap();
    assert_eq!(jobs(&db, "u").len(), 5);
    // namespaces are separate queues
    db.remember("I like tea.", RememberOpts { ns: ns("v"), ..Default::default() }).unwrap();
    assert_eq!(jobs(&db, "v").len(), 1);
    assert_eq!(jobs(&db, "v")[0].0, 1);
}

#[test]
fn failed_memory_write_leaves_no_job() {
    let db = attached();
    db.remember("I like tea.", RememberOpts { ns: ns("u"), ..Default::default() }).unwrap();
    // make every write to kleos_mem fail inside its transaction
    q(&db, "::set_triggers kleos_mem on put { ?[x] <- [[1]] :assert none }");
    let err = db.remember("I like coffee.", RememberOpts { ns: ns("u"), ..Default::default() });
    assert!(err.is_err());
    q(&db, "::set_triggers kleos_mem");
    assert_eq!(jobs(&db, "u").len(), 1, "no job without its memory");
    assert_eq!(q(&db, "?[count(id)] := *kleos_mem{ns: 'u', id}")[0][0].get_int(), Some(1));
    // the reverse: a failing queue write fails the memory write too
    q(&db, "::set_triggers kleos_engram_queue on put { ?[x] <- [[1]] :assert none }");
    assert!(db.remember("I like cocoa.", RememberOpts { ns: ns("u"), ..Default::default() }).is_err());
    q(&db, "::set_triggers kleos_engram_queue");
    assert_eq!(q(&db, "?[count(id)] := *kleos_mem{ns: 'u', id}")[0][0].get_int(), Some(1), "no memory without its job");
}

#[test]
fn concurrent_remembers_queue_exactly_the_committed_memories() {
    let db = attached();
    db.remember("My rent is $1,000 per month.", RememberOpts { ns: ns("race"), at: Some(d(1)), ..Default::default() }).unwrap();
    let results: Vec<_> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..8)
            .map(|i| {
                let db = &db;
                sc.spawn(move || {
                    db.remember(&format!("My rent is ${},{}00 per month.", 2 + i / 5, i % 5 + 1), RememberOpts { ns: ns("race"), ..Default::default() })
                        .unwrap()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let cur = q(&db, "?[id] := *kleos_mem{ns: 'race', id @ 'NOW'}");
    assert_eq!(cur.len(), 1, "one current rent: {results:?}");
    let all: BTreeSet<String> = q(&db, "?[id] := *kleos_mem{ns: 'race', id}").into_iter().map(|r| r[0].get_str().unwrap().to_string()).collect();
    assert_eq!(all.len(), 9);
    let j = jobs(&db, "race");
    // one job per committed memory, no more, no less
    let queued: Vec<String> = j.iter().flat_map(|x| x.2.clone()).collect();
    assert_eq!(queued.len(), 9);
    assert_eq!(queued.iter().cloned().collect::<BTreeSet<_>>(), all);
    // each supersede job names exactly what its memory superseded, and the
    // old memory's job comes first
    let qid_of: BTreeMap<&str, i64> = j.iter().map(|x| (x.2[0].as_str(), x.0)).collect();
    for r in &results {
        let id = r.id.as_ref().unwrap();
        let job = j.iter().find(|x| &x.2[0] == id).unwrap();
        assert_eq!(job.3, r.superseded);
        assert_eq!(job.1, if r.superseded.is_empty() { "write" } else { "supersede" });
        for o in &r.superseded {
            assert!(qid_of[o.as_str()] < job.0);
        }
    }
    assert_eq!(j.iter().map(|x| x.0).collect::<BTreeSet<_>>().len(), 9, "distinct qids");
}

#[test]
fn queue_status_op() {
    let db = attached();
    for t in ["I like tea.", "I have a cat.", "I live in Lisbon."] {
        db.remember(t, RememberOpts { ns: ns("u"), ..Default::default() }).unwrap();
    }
    db.remember("I like tea.", RememberOpts { ns: ns("w"), ..Default::default() }).unwrap();
    // a worker's view: one done, one failed with an error
    q(&db, "?[ns, qid, status, attempts, error, updated] <- [['u', 1, 'done', 1, null, 10], ['u', 2, 'failed', 3, 'model went away', 20]] \
            :update kleos_engram_queue {ns, qid => status, attempts, error, updated}");
    let st = q(&db, "::engram queue {}");
    assert_eq!(st.len(), 2);
    let u = &st[0];
    assert_eq!(u[0].get_str(), Some("u"));
    assert_eq!((u[1].get_int(), u[2].get_int(), u[3].get_int(), u[4].get_int(), u[5].get_int()), (Some(1), Some(0), Some(1), Some(1), Some(3)));
    assert_eq!(u[6].get_str(), Some("model went away"));
    let w = q(&db, "::engram queue {ns: 'w'}");
    assert_eq!((w.len(), w[0][1].get_int(), w[0][6].clone()), (1, Some(1), DataValue::Null));
    let listed = q(&db, "::engram queue {ns: 'u', jobs: true}");
    assert_eq!(listed.len(), 3);
    assert_eq!(listed[1][6].get_str(), Some("failed"));
}

/// Erasing a fact must not bring back a row value of a fact that ended
/// while the two shared the row: A writes row r, B joins it, A is
/// superseded (r recomputed from B alone), then B is erased. Without the
/// fix, A's value at T1 became current again.
#[test]
fn erase_fact_does_not_resurrect_an_ended_fact() {
    let db = attached();
    let fp = "eg1:00112233445566778899aabbccddeeff";
    let run = |script: &str, p: serde_json::Value| {
        let params = p.as_object().unwrap().iter().map(|(k, v)| (k.clone(), DataValue::from(v))).collect();
        db.run_script(script, params, ScriptMutability::Mutable).unwrap_or_else(|e| panic!("{script}: {e:?}")).rows
    };
    let fact = |id: &str| serde_json::json!({"id": id, "text": id, "target": id, "rows": [[1, 7]]});
    let row = |v: f32, facts: &[&str]| serde_json::json!({"layer": 1, "row": 7, "vec": [v, v, v, v], "facts": facts});
    let w = "::engram write nano {ns: 'u', fingerprint: $fp, at: $at, facts: $facts, rows: $rows, retract: $retract}";
    run(w, serde_json::json!({"fp": fp, "at": d(1), "facts": [fact("a")], "rows": [row(1.0, &["a"])], "retract": []}));
    run(w, serde_json::json!({"fp": fp, "at": d(2), "facts": [fact("b")], "rows": [row(2.0, &["a", "b"])], "retract": []}));
    run(w, serde_json::json!({"fp": fp, "at": d(3), "facts": [], "rows": [row(3.0, &["b"])], "retract": ["a"]}));
    let out = run("::engram erase_fact nano {ns: 'u', fact: 'b'}", serde_json::json!({}));
    assert_eq!(out[0][2].get_str(), Some("reverted"));
    let at = |t: i64| run("::engram materialize nano {ns: 'u', as_of: $t}", serde_json::json!({"t": t})).len();
    assert_eq!((at(d(1)), at(d(2)), at(d(4))), (1, 1, 0), "A's row until A ended, then base");
    assert_eq!(run("::engram materialize nano {ns: 'u'}", serde_json::json!({})).len(), 0);
}

/// Claims (docs/specs/memory-api.md, "Claims"): a multi-sentence statement
/// enqueues one job per claim it adds, each with the claim's own text, in
/// the one transaction, numbered with consecutive versions that end at the
/// namespace version; a claim that supersedes enqueues `supersede` with
/// only the old claim it ended.
#[test]
fn claims_enqueue_one_job_each() {
    let db = attached();
    let a = db
        .remember(
            "My rent is $1,800 per month. I have a dog named Max. Can you suggest a vet?",
            RememberOpts { ns: ns("c"), at: Some(d(1)), entities: vec!["Max".into()], ..Default::default() },
        )
        .unwrap();
    let stmt = a.id.clone().unwrap();
    let ids: Vec<String> = a.claims.iter().map(|c| c.id.clone().unwrap()).collect();
    assert_eq!(ids, vec![format!("{stmt}.1"), format!("{stmt}.2"), format!("{stmt}.3")]);
    let j = jobs(&db, "c");
    assert_eq!(j.len(), 3, "{j:?}");
    for (i, job) in j.iter().enumerate() {
        assert_eq!((job.1.as_str(), &job.2, job.5), ("write", &vec![ids[i].clone()], d(1)));
        assert_eq!(job.4.as_deref(), Some(a.claims[i].text.as_str()));
    }
    assert_eq!(j.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 2, 3]);
    assert_eq!(ns_seq(&db, "c"), 3, "the version advances by the number of jobs");
    // the entity hint goes with the claim that names it
    let meta = q(&db, "?[qid, m] := *kleos_engram_queue{ns: 'c', qid, meta: m}, !is_null(m)");
    assert_eq!(meta.len(), 1, "{meta:?}");
    assert_eq!(meta[0][0].get_int(), Some(2));

    // one claim updates the rent (value rule), the other is new
    let b = db
        .remember(
            "My rent is $2,100 per month. I have a cat named Tom.",
            RememberOpts { ns: ns("c"), at: Some(d(5)), ..Default::default() },
        )
        .unwrap();
    assert_eq!(b.superseded, vec![ids[0].clone()]);
    let b1 = b.claims[0].id.clone().unwrap();
    let b2 = b.claims[1].id.clone().unwrap();
    let j = jobs(&db, "c");
    assert_eq!(j.len(), 5);
    assert_eq!((j[3].1.as_str(), &j[3].2, &j[3].3), ("supersede", &vec![b1], &vec![ids[0].clone()]));
    assert_eq!((j[4].1.as_str(), &j[4].2, &j[4].3), ("write", &vec![b2], &vec![]));
    assert_eq!(j.last().unwrap().0, ns_seq(&db, "c"));

    // erasing the first statement erases each of its claims in one job
    let r = db.forget(ForgetOpts { ns: ns("c"), id: Some(stmt), mode: Some("erase".into()), ..Default::default() }).unwrap();
    assert_eq!(r.ids, ids);
    let j = jobs(&db, "c");
    assert_eq!((j[5].1.as_str(), &j[5].2), ("erase", &ids));
    assert_eq!(j[0].4, None, "erased claim texts are dropped from earlier jobs");
}

/// Every job is also an `engram.apply` workflow, started in the job's own
/// commit, in its namespace's lane ordered by qid; the lane hands out one
/// job at a time; jobs queued without a workflow are adopted
/// (docs/specs/durable-workflows.md, "Engram").
#[test]
fn jobs_are_workflows_in_namespace_lanes() {
    let db = attached();
    for (k, t) in ["Ada lives in Lisbon.", "Ada works at the observatory.", "Ada plays the cello."].iter().enumerate() {
        db.remember(t, RememberOpts { ns: ns("u"), at: Some(d(k as i64 + 1)), ..Default::default() }).unwrap();
    }
    db.remember("Bo lives in Porto.", RememberOpts { ns: ns("v"), at: Some(d(1)), ..Default::default() }).unwrap();
    let j = jobs(&db, "u");
    assert_eq!(j.len(), 3);
    let wfs = q(&db, "?[id, name, mode, queue, serial, serial_seq, status] := \
                      *kleos_flow_wf{id, name, mode, queue, serial, serial_seq, status}, starts_with(id, 'engram:u:')");
    assert_eq!(wfs.len(), 3);
    for (row, job) in wfs.iter().zip(&j) {
        assert_eq!(row[0], DataValue::from(format!("engram:u:{}", job.0).as_str()));
        assert_eq!(&row[1..5], &[DataValue::from("engram.apply"), DataValue::from("code"), DataValue::from("engram"), DataValue::from("engram:u")]);
        assert_eq!(row[5], DataValue::from(job.0));
        assert_eq!(row[6], DataValue::from("pending"));
    }
    // the started event carries only {ns, qid}, not the memory text
    let ev = q(&db, &format!("?[d] := *kleos_flow_event{{wf: 'engram:u:{}', seq: 1, data: d}}", j[0].0));
    match &ev[0][0] {
        DataValue::Json(x) => assert_eq!(x.0, serde_json::json!({"ns": "u", "qid": j[0].0})),
        other => panic!("{other:?}"),
    }
    // one job per lane at a time: u's first, v's first, then nothing
    let claim = |w: &str| db.flow_call("claim", serde_json::json!({"worker": w, "queues": ["engram"], "workflows": ["engram.apply"]})).unwrap();
    let a = claim("w1");
    let b = claim("w2");
    assert_eq!(claim("w3"), serde_json::Value::Null);
    let mut got = vec![a["id"].as_str().unwrap().to_string(), b["id"].as_str().unwrap().to_string()];
    got.sort();
    assert_eq!(got[0], format!("engram:u:{}", j[0].0));
    assert!(got[1].starts_with("engram:v:"));
    let u = if a["id"] == got[0].as_str() { &a } else { &b };
    db.flow_call("complete", serde_json::json!({"id": u["id"], "token": u["token"], "output": {}})).unwrap();
    assert_eq!(claim("w3")["id"], format!("engram:u:{}", j[1].0));
    // a lane filter restricts a worker to its namespaces
    assert_eq!(db.flow_call("claim", serde_json::json!({"worker": "w4", "queues": ["engram"], "lanes": ["engram:u"]})).unwrap(), serde_json::Value::Null);
    // adopt: a job queued without a workflow (an older store) gets one, once
    q(&db, &format!("?[id] <- [['engram:u:{}']] :rm kleos_flow_wf {{id}}", j[2].0));
    q(&db, &format!("?[serial, serial_seq] <- [['engram:u', {}]] :rm kleos_flow_lane {{serial, serial_seq}}", j[2].0));
    assert_eq!(db.flow_call("adopt_engram", serde_json::json!({})).unwrap()["adopted"], 1);
    assert_eq!(db.flow_call("adopt_engram", serde_json::json!({})).unwrap()["adopted"], 0);
    // a failed memory write starts no workflow either
    q(&db, "::set_triggers kleos_engram_queue on put { ?[x] <- [[1]] :assert none }");
    let before = q(&db, "?[count(id)] := *kleos_flow_wf{id}");
    assert!(db.remember("Ada moved to Porto.", RememberOpts { ns: ns("u"), at: Some(d(9)), ..Default::default() }).is_err());
    q(&db, "::set_triggers kleos_engram_queue");
    assert_eq!(q(&db, "?[count(id)] := *kleos_flow_wf{id}"), before);
}
