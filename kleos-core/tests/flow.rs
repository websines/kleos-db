/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Durable workflows (docs/specs/durable-workflows.md): built-in step graphs,
//! the worker protocol code-mode SDKs use, fencing, atomic steps, timers,
//! signals, children, cancel, schedules, and two child processes: one killed
//! mid-run and resumed, two workers on one SQLite file never running a step
//! twice.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use kleos::{DataValue, DbInstance, ScriptMutability};
use serde_json::{json, Value};

fn mem() -> DbInstance {
    DbInstance::new("mem", "", "").unwrap()
}

fn tmp(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("flow-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn call(db: &DbInstance, op: &str, args: Value) -> Value {
    db.flow_call(op, args).unwrap_or_else(|e| panic!("flow {op}: {e:?}"))
}

fn q(db: &DbInstance, script: &str) -> Vec<Vec<DataValue>> {
    db.run_script(script, Default::default(), ScriptMutability::Mutable)
        .unwrap_or_else(|e| panic!("{script}: {e:?}"))
        .rows
}

fn status(db: &DbInstance, id: &str) -> Value {
    call(db, "status", json!({"id": id}))
}

fn tick(db: &DbInstance) -> Value {
    call(db, "tick", json!({}))
}

fn keys(db: &DbInstance, id: &str, ty: &str) -> Vec<String> {
    call(db, "history", json!({"id": id}))
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["type"] == ty)
        .map(|e| e["key"].as_str().unwrap_or_default().to_string())
        .collect()
}

#[test]
fn builtin_graph_runs_end_to_end() {
    let db = mem();
    q(&db, ":create log {wf: String, n: Int => v: String}");
    let g = json!({"steps": [
        {"name": "w", "set": "$input.who"},
        {"query": {"script": "?[wf, n, v] <- [[$wf, 0, $who]] :put log {wf, n => v}",
                   "params": {"wf": "$input.id", "who": "$w"}}},
        {"if": {"cond": "$w == 'ada'",
                "then": [{"name": "greet", "set": "concat('hi ', $w)"}],
                "else": [{"name": "greet", "set": "'who?'"}]}},
        {"name": "looped", "loop": {"while": "$i < 3", "body": [
            {"query": {"script": "?[wf, n, v] <- [[$wf, $n, 'loop']] :put log {wf, n => v}",
                       "params": {"wf": "$input.id", "n": "=$i + 1"}}}]}},
        {"name": "each", "for_each": {"in": "=[10, 20]", "as": "x", "body": [{"set": "$x * 2"}]}},
        {"name": "parts", "fan_out": {"branches": [
            [{"query": {"script": "?[c] := *log{wf: $wf}, c = 1", "params": {"wf": "$input.id"}}},
             {"set": "'left'"}],
            [{"set": "'right'"}]]}},
        {"name": "n", "query": {"script": "?[count(n)] := *log{wf: $wf, n}", "params": {"wf": "$input.id"}}},
        {"return": "=[$greet, $each, $parts]"}
    ]});
    let def = call(&db, "define", json!({"name": "demo", "graph": g}));
    assert_eq!(def["version"], 1);
    let s = call(&db, "start", json!({"workflow": "demo", "id": "d1", "input": {"id": "d1", "who": "ada"}}));
    assert_eq!(s["mode"], "builtin");
    // idempotent start
    assert_eq!(call(&db, "start", json!({"workflow": "demo", "id": "d1"}))["existing"], true);
    let t = tick(&db);
    assert_eq!(t["ran"][0]["status"], "completed", "{t}");
    let st = status(&db, "d1");
    assert_eq!(st["status"], "completed");
    assert_eq!(st["output"], json!(["hi ada", [20, 40], ["left", "right"]]));
    // 1 + 3 loop rows, each written once
    assert_eq!(q(&db, "?[count(n)] := *log{wf: 'd1', n}")[0][0], DataValue::from(4));
    // command keys: steps by path, loop iterations, branches
    let ks = keys(&db, "d1", "step");
    for k in ["s1", "s3.i0.s0", "s3.i2.s0", "s5.b0.s0", "s6"] {
        assert!(ks.contains(&k.to_string()), "{k} not in {ks:?}");
    }
    // a second tick runs nothing
    assert_eq!(tick(&db)["ran"], json!([]));
    // ::flow from CozoScript
    let rows = q(&db, "::flow status {id: 'd1'}");
    assert!(!rows.is_empty());
    let bad = db.flow_call("define", json!({"name": "x", "graph": {"steps": [{"nope": {}}]}}));
    assert!(bad.unwrap_err().to_string().contains("unknown step kind"));
}

#[test]
fn retries_back_off_and_exhaust() {
    let db = mem();
    q(&db, ":create flag {x: Int}");
    let g = json!({"steps": [
        {"name": "check", "retry": {"max_attempts": 3, "initial_ms": 3000, "backoff": 1.0},
         "query": {"script": "?[x] := *flag{x}, x == 1 :assert some"}},
        {"return": "'ok'"}
    ]});
    call(&db, "define", json!({"name": "r", "graph": g}));
    call(&db, "start", json!({"workflow": "r", "id": "r1"}));
    assert_eq!(tick(&db)["ran"][0]["status"], "waiting");
    let st = status(&db, "r1");
    assert_eq!(st["timers"][0]["kind"], "retry");
    // not due yet
    assert_eq!(tick(&db)["ran"], json!([]));
    q(&db, "?[x] <- [[1]] :put flag {x}");
    std::thread::sleep(Duration::from_millis(3100));
    assert_eq!(tick(&db)["ran"][0]["status"], "completed");
    assert_eq!(keys(&db, "r1", "step_failed").len(), 1);

    // exhausted: the workflow fails with the step's error
    let g = json!({"steps": [
        {"name": "check", "retry": {"max_attempts": 3, "initial_ms": 10, "backoff": 1.0},
         "query": {"script": "?[x] := *flag{x}, x == 1 :assert some"}},
        {"return": "'ok'"}
    ]});
    call(&db, "define", json!({"name": "r_fast", "graph": g}));
    call(&db, "start", json!({"workflow": "r_fast", "id": "r2"}));
    q(&db, "?[x] <- [[1]] :rm flag {x}");
    let t0 = Instant::now();
    while !["failed", "completed"].contains(&status(&db, "r2")["status"].as_str().unwrap()) {
        tick(&db);
        std::thread::sleep(Duration::from_millis(10));
        assert!(t0.elapsed().as_secs() < 10);
    }
    let st = status(&db, "r2");
    assert_eq!(st["status"], "failed");
    assert!(st["error"].as_str().unwrap().contains("check"), "{st}");
    assert_eq!(keys(&db, "r2", "step_failed").len(), 3);

    // on_error: continue
    let g = json!({"steps": [
        {"name": "c", "on_error": "continue", "retry": {"max_attempts": 1},
         "query": {"script": "?[x] := *flag{x}, x == 1 :assert some"}},
        {"return": "is_null($c)"}
    ]});
    call(&db, "define", json!({"name": "rc", "graph": g}));
    call(&db, "start", json!({"workflow": "rc", "id": "rc1"}));
    tick(&db);
    assert_eq!(status(&db, "rc1")["output"], true);
}

#[test]
fn timers_sleep_then_wake() {
    let db = mem();
    let g = json!({"steps": [{"sleep": {"ms": 1500}}, {"return": "'woke'"}]});
    call(&db, "define", json!({"name": "s", "graph": g}));
    call(&db, "start", json!({"workflow": "s", "id": "s1"}));
    assert_eq!(tick(&db)["ran"][0]["status"], "waiting");
    let st = status(&db, "s1");
    let due = st["due"].as_i64().unwrap();
    assert_eq!(st["timers"][0]["due"].as_i64().unwrap(), due);
    assert_eq!(tick(&db)["ran"], json!([]));
    std::thread::sleep(Duration::from_millis(1550));
    assert_eq!(tick(&db)["ran"][0]["status"], "completed");
    assert_eq!(status(&db, "s1")["output"], "woke");
    let h = call(&db, "history", json!({"id": "s1"}));
    let types: Vec<&str> = h.as_array().unwrap().iter().map(|e| e["type"].as_str().unwrap()).collect();
    assert_eq!(types, ["started", "timer", "timer_fired", "completed"]);
}

#[test]
fn signals_wait_buffer_and_time_out() {
    let db = mem();
    q(&db, ":create ran {wf: String, k: String}");
    let g = json!({"steps": [
        {"query": {"script": "?[wf, k] := wf = $wf, k = to_string(rand_uuid_v4()) :put ran {wf, k}", "params": {"wf": "$input.id"}}},
        {"name": "ok", "wait_signal": {"name": "approve", "timeout_ms": "$input.t"}},
        {"return": "$ok"}
    ]});
    call(&db, "define", json!({"name": "a", "graph": g}));
    call(&db, "start", json!({"workflow": "a", "id": "a1", "input": {"id": "a1", "t": 60000}}));
    assert_eq!(tick(&db)["ran"][0]["status"], "waiting");
    // a wait for a signal: due is the timeout
    assert!(status(&db, "a1")["due"].as_i64().is_some());
    call(&db, "signal", json!({"id": "a1", "name": "approve", "payload": {"by": "ops"}, "sid": "x1"}));
    // a repeated sid is ignored
    call(&db, "signal", json!({"id": "a1", "name": "approve", "payload": {"by": "dup"}, "sid": "x1"}));
    assert_eq!(tick(&db)["ran"][0]["status"], "completed");
    assert_eq!(status(&db, "a1")["output"], json!({"by": "ops"}));
    // the first step ran once although the workflow ran twice
    assert_eq!(q(&db, "?[count(k)] := *ran{wf: 'a1', k}")[0][0], DataValue::from(1));

    // a signal sent before the wait is buffered
    call(&db, "start", json!({"workflow": "a", "id": "a2", "input": {"id": "a2", "t": 60000}}));
    call(&db, "signal", json!({"id": "a2", "name": "approve", "payload": 7}));
    tick(&db);
    assert_eq!(status(&db, "a2")["output"], 7);

    // nobody answers: the wait times out with null
    call(&db, "start", json!({"workflow": "a", "id": "a3", "input": {"id": "a3", "t": 200}}));
    tick(&db);
    std::thread::sleep(Duration::from_millis(220));
    tick(&db);
    let st = status(&db, "a3");
    assert_eq!(st["status"], "completed");
    assert_eq!(st["output"], Value::Null);
    assert_eq!(keys(&db, "a3", "signal_timeout"), ["s1"]);
}

#[test]
fn children_and_cancel() {
    let db = mem();
    call(&db, "define", json!({"name": "child", "graph": {"steps": [{"return": "$input * 10"}]}}));
    call(&db, "define", json!({"name": "parent", "graph": {"steps": [
        {"name": "c", "child": {"workflow": "child", "input": "$input"}},
        {"return": "$c + 1"}]}}));
    call(&db, "start", json!({"workflow": "parent", "id": "p1", "input": 4}));
    let t0 = Instant::now();
    while status(&db, "p1")["status"] != "completed" {
        tick(&db);
        assert!(t0.elapsed().as_secs() < 10, "{}", call(&db, "history", json!({"id": "p1"})));
    }
    assert_eq!(status(&db, "p1")["output"], 41);
    let kids = call(&db, "list", json!({"parent": "p1"}));
    assert_eq!(kids[0]["id"], "p1/s0");
    assert_eq!(kids[0]["status"], "completed");

    // cancel a waiting workflow and its child
    call(&db, "define", json!({"name": "slow", "graph": {"steps": [{"sleep": {"ms": 60000}}]}}));
    call(&db, "define", json!({"name": "p2", "graph": {"steps": [{"child": {"workflow": "slow"}}]}}));
    call(&db, "start", json!({"workflow": "p2", "id": "p2"}));
    tick(&db);
    tick(&db);
    assert_eq!(status(&db, "p2/s0")["status"], "waiting");
    let c = call(&db, "cancel", json!({"id": "p2", "reason": "user"}));
    assert_eq!(c["status"], "canceled");
    assert_eq!(status(&db, "p2/s0")["status"], "canceled");
    assert_eq!(status(&db, "p2")["error"], "user");
    // nothing left to run
    assert_eq!(tick(&db)["ran"], json!([]));
    assert!(db.flow_call("signal", json!({"id": "p2", "name": "x"})).is_err());
}

/// The protocol a code-mode SDK speaks, by hand: claim, step, lease
/// expiry, takeover, fencing, replay, non-determinism, cancel.
#[test]
fn code_mode_protocol_and_fencing() {
    let db = mem();
    q(&db, ":create out {k: String}");
    call(&db, "start", json!({"workflow": "agent", "id": "c1", "input": {"x": 1}}));
    assert_eq!(status(&db, "c1")["mode"], "code");
    // built-in ticks leave code workflows alone
    assert_eq!(tick(&db)["ran"], json!([]));
    let a = call(&db, "claim", json!({"worker": "A", "lease_ms": 2000, "workflows": ["agent"]}));
    assert_eq!(a["id"], "c1");
    let ta = a["token"].as_i64().unwrap();
    // nothing else to claim while A holds it
    assert_eq!(call(&db, "claim", json!({"worker": "B"})), Value::Null);
    let r = call(&db, "step", json!({"id": "c1", "token": ta, "key": "1:fetch", "name": "fetch", "result": [1, 2]}));
    assert_eq!(r["replayed"], false);
    // A stalls past its lease; B takes over with a new token
    std::thread::sleep(Duration::from_millis(2100));
    let b = call(&db, "claim", json!({"worker": "B", "lease_ms": 5000}));
    let tb = b["token"].as_i64().unwrap();
    assert_eq!(tb, ta + 1);
    assert_eq!(b["history"].as_array().unwrap().len(), 2);
    // A's next write is refused, its query's rows included
    let e = db
        .flow_call(
            "step",
            json!({"id": "c1", "token": ta, "key": "2:save", "name": "save",
                   "action": {"query": {"script": "?[k] <- [['from-a']] :put out {k}"}}}),
        )
        .unwrap_err();
    assert!(e.to_string().starts_with("lease lost"), "{e}");
    assert!(q(&db, "?[k] := *out{k}").is_empty());
    // B replays step 1 (recorded) and runs step 2
    let r = call(&db, "step", json!({"id": "c1", "token": tb, "key": "1:fetch", "name": "fetch", "result": [9]}));
    assert_eq!(r, json!({"result": [1, 2], "replayed": true}));
    let r = call(
        &db,
        "step",
        json!({"id": "c1", "token": tb, "key": "2:save", "name": "save",
               "action": {"query": {"script": "?[k] <- [['from-b']] :put out {k}"}}}),
    );
    assert_eq!(r["result"]["headers"], json!(["status"]));
    assert_eq!(q(&db, "?[k] := *out{k}"), vec![vec![DataValue::from("from-b")]]);
    // non-determinism: same key, other name
    let e = db
        .flow_call("step", json!({"id": "c1", "token": tb, "key": "1:fetch", "name": "other", "result": 0}))
        .unwrap_err();
    assert!(e.to_string().starts_with("non-determinism"), "{e}");
    // markers: first write wins
    let m = call(&db, "record", json!({"id": "c1", "token": tb, "events": [{"type": "marker", "key": "3:now", "name": "now", "data": 111}]}));
    assert_eq!(m["data"], json!([111]));
    let m = call(&db, "record", json!({"id": "c1", "token": tb, "events": [{"type": "marker", "key": "3:now", "name": "now", "data": 222}]}));
    assert_eq!(m["data"], json!([111]));
    // cancel while running: commands are refused, fail finishes as canceled
    assert_eq!(call(&db, "cancel", json!({"id": "c1"}))["cancel"], true);
    let e = db
        .flow_call("step", json!({"id": "c1", "token": tb, "key": "4:x", "name": "x", "result": 0}))
        .unwrap_err();
    assert!(e.to_string().starts_with("canceled"), "{e}");
    call(&db, "fail", json!({"id": "c1", "token": tb, "error": "canceled"}));
    assert_eq!(status(&db, "c1")["status"], "canceled");
}

#[test]
fn suspend_wakes_on_signal_child_or_timer() {
    let db = mem();
    call(&db, "start", json!({"workflow": "w", "id": "w1"}));
    let c = call(&db, "claim", json!({"worker": "A"}));
    let t = c["token"].as_i64().unwrap();
    let due = kleos_now() + 60_000_000;
    let s = call(&db, "suspend", json!({"id": "w1", "token": t, "timers": [{"key": "1:sleep", "due": due}], "signals": ["go"]}));
    assert_eq!(s["due"], due);
    assert_eq!(call(&db, "claim", json!({"worker": "A"})), Value::Null);
    call(&db, "signal", json!({"id": "w1", "name": "go", "payload": 5}));
    let c = call(&db, "claim", json!({"worker": "A"}));
    let t = c["token"].as_i64().unwrap();
    let got = call(&db, "consume_signal", json!({"id": "w1", "token": t, "key": "2:go", "name": "go"}));
    assert_eq!(got, json!({"received": true, "payload": 5}));
    // consumed once; replay returns the same payload
    let again = call(&db, "consume_signal", json!({"id": "w1", "token": t, "key": "2:go", "name": "go"}));
    assert_eq!(again["payload"], 5);
    let none = call(&db, "consume_signal", json!({"id": "w1", "token": t, "key": "3:go", "name": "go"}));
    assert_eq!(none["received"], false);
    // a signal that is already in the inbox when the workflow suspends wakes it at once
    call(&db, "signal", json!({"id": "w1", "name": "go"}));
    let s = call(&db, "suspend", json!({"id": "w1", "token": t, "signals": ["go"]}));
    assert!(s["due"].as_i64().unwrap() <= kleos_now());
}

fn kleos_now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros() as i64
}

#[test]
fn remember_step_commits_with_its_checkpoint() {
    let db = mem();
    call(&db, "start", json!({"workflow": "m", "id": "m1"}));
    let c = call(&db, "claim", json!({"worker": "A", "lease_ms": 1500}));
    let ta = c["token"].as_i64().unwrap();
    let r = call(
        &db,
        "step",
        json!({"id": "m1", "token": ta, "key": "1:remember", "name": "remember",
               "action": {"remember": {"text": "Ada lives in Lisbon.", "ns": "u"}}}),
    );
    assert_eq!(r["result"]["action"], "added");
    // replay returns the recorded result and stores nothing new
    let again = call(
        &db,
        "step",
        json!({"id": "m1", "token": ta, "key": "1:remember", "name": "remember",
               "action": {"remember": {"text": "Ada lives in Lisbon.", "ns": "u"}}}),
    );
    assert_eq!(again["replayed"], true);
    assert_eq!(again["result"]["id"], r["result"]["id"]);
    // a stale worker's remember leaves no memory behind
    std::thread::sleep(Duration::from_millis(1600));
    call(&db, "claim", json!({"worker": "B"}));
    let e = db
        .flow_call(
            "step",
            json!({"id": "m1", "token": ta, "key": "2:remember", "name": "remember",
                   "action": {"remember": {"text": "Ada works at the observatory.", "ns": "u"}}}),
        )
        .unwrap_err();
    assert!(e.to_string().starts_with("lease lost"), "{e}");
    let n = q(&db, "?[count(id)] := *kleos_mem{ns: 'u', id}");
    assert_eq!(n[0][0], DataValue::from(1));
}

#[test]
fn schedules_fire_once_and_advance() {
    let db = mem();
    call(&db, "define", json!({"name": "job", "graph": {"steps": [{"return": "$input"}]}}));
    let s = call(&db, "schedule", json!({"name": "nightly", "cron": "@daily", "workflow": "job", "input": 3}));
    assert!(s["next_at"].as_i64().unwrap() > kleos_now());
    assert!(db.flow_call("schedule", json!({"name": "bad", "cron": "61 * * * *", "workflow": "job"})).is_err());
    // pretend it is due, twice over (two workers firing it)
    q(&db, "?[name, next_at] <- [['nightly', 1000]] :update kleos_flow_schedule {name => next_at}");
    let t = tick(&db);
    assert_eq!(t["fired"], json!(["sched:nightly:1000"]));
    assert_eq!(status(&db, "sched:nightly:1000")["output"], 3);
    assert_eq!(tick(&db)["fired"], json!([]));
    let sc = call(&db, "schedules", json!({}));
    assert!(sc[0]["next_at"].as_i64().unwrap() > kleos_now());
    assert_eq!(sc[0]["last_id"], "sched:nightly:1000");
    assert_eq!(call(&db, "unschedule", json!({"name": "nightly"}))["removed"], true);
}

#[test]
fn in_process_worker_wakes_on_start() {
    let db = mem();
    call(&db, "define", json!({"name": "quick", "graph": {"steps": [{"return": "'done'"}]}}));
    let w = db.flow_worker(kleos::FlowWorkerOpts { poll_ms: 5_000, ..Default::default() });
    let t0 = Instant::now();
    call(&db, "start", json!({"workflow": "quick", "id": "q1"}));
    while status(&db, "q1")["status"] != "completed" {
        assert!(t0.elapsed() < Duration::from_secs(3), "the worker did not wake on the start");
        std::thread::sleep(Duration::from_millis(5));
    }
    // well under the 5 s poll: woken by the change feed
    assert!(t0.elapsed() < Duration::from_millis(1500));
    w.stop();
}

// ------------------------------------------------------------ processes

/// Child process: a code-mode worker that records step 1, touches a file,
/// then hangs inside step 2 until it is killed.
#[test]
#[ignore]
fn crash_child() {
    let Ok(path) = std::env::var("KLEOS_FLOW_DB") else { return };
    let dir = PathBuf::from(std::env::var("KLEOS_FLOW_DIR").unwrap());
    let db = DbInstance::new("sqlite", &path, "").unwrap();
    let c = db.flow_call("claim", json!({"worker": "doomed", "lease_ms": 300})).unwrap();
    let t = c["token"].as_i64().unwrap();
    db.flow_call(
        "step",
        json!({"id": "job", "token": t, "key": "1:charge", "name": "charge",
               "action": {"query": {"script": "?[k, by] <- [['charge', 'doomed']] :put effects {k => by}"}}}),
    )
    .unwrap();
    std::fs::write(dir.join("step1"), b"").unwrap();
    std::thread::sleep(Duration::from_secs(120)); // "inside step 2"
}

#[test]
fn crash_mid_step_then_resume() {
    let dir = tmp("crash");
    let path = dir.join("flow.db");
    let db = DbInstance::new("sqlite", &path, "").unwrap();
    q(&db, ":create effects {k: String => by: String}");
    call(&db, "start", json!({"workflow": "billing", "id": "job"}));
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["crash_child", "--exact", "--ignored", "--test-threads=1", "--nocapture"])
        .env("KLEOS_FLOW_DB", &path)
        .env("KLEOS_FLOW_DIR", &dir)
        .spawn()
        .unwrap();
    let t0 = Instant::now();
    while !dir.join("step1").exists() {
        assert!(t0.elapsed().as_secs() < 60, "child never finished step 1");
        std::thread::sleep(Duration::from_millis(10));
    }
    child.kill().unwrap();
    child.wait().unwrap();
    assert_eq!(status(&db, "job")["status"], "running");
    // the dead worker's lease runs out, and a new worker resumes
    let c = loop {
        let c = call(&db, "claim", json!({"worker": "rescuer"}));
        if !c.is_null() {
            break c;
        }
        assert!(t0.elapsed().as_secs() < 60);
        std::thread::sleep(Duration::from_millis(50));
    };
    let t = c["token"].as_i64().unwrap();
    assert_eq!(t, 2);
    // replaying step 1 does not charge again
    let r = call(
        &db,
        "step",
        json!({"id": "job", "token": t, "key": "1:charge", "name": "charge",
               "action": {"query": {"script": "?[k, by] <- [['charge', 'rescuer']] :put effects {k => by}"}}}),
    );
    assert_eq!(r["replayed"], true);
    call(
        &db,
        "step",
        json!({"id": "job", "token": t, "key": "2:ship", "name": "ship",
               "action": {"query": {"script": "?[k, by] <- [['ship', 'rescuer']] :put effects {k => by}"}}}),
    );
    call(&db, "complete", json!({"id": "job", "token": t, "output": "ok"}));
    let eff = q(&db, "?[k, by] := *effects{k, by}");
    assert_eq!(
        eff,
        vec![
            vec![DataValue::from("charge"), DataValue::from("doomed")],
            vec![DataValue::from("ship"), DataValue::from("rescuer")],
        ]
    );
    let st = status(&db, "job");
    assert_eq!(st["status"], "completed");
    assert_eq!(st["attempts"], 2);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Child process: tick built-in workflows until none is left.
#[test]
#[ignore]
fn tick_child() {
    let Ok(path) = std::env::var("KLEOS_FLOW_DB") else { return };
    let me = std::env::var("KLEOS_FLOW_ME").unwrap();
    let db = DbInstance::new("sqlite", &path, "").unwrap();
    let t0 = Instant::now();
    loop {
        db.flow_call("tick", json!({"worker": me, "max": 4, "lease_ms": 10000})).unwrap();
        let left = db
            .run_script(
                "?[count(id)] := *kleos_flow_wf{id, status}, status != 'completed'",
                Default::default(),
                ScriptMutability::Immutable,
            )
            .unwrap()
            .rows[0][0]
            .get_int()
            .unwrap();
        if left == 0 {
            break;
        }
        assert!(t0.elapsed().as_secs() < 300, "workflows left: {left}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn two_worker_processes_never_run_a_step_twice() {
    let dir = tmp("two");
    let path = dir.join("flow.db");
    let n = 24;
    {
        let db = DbInstance::new("sqlite", &path, "").unwrap();
        q(&db, ":create ran {wf: String, step: String, k: String => by: String}");
        let step = |s: &str| {
            json!({"query": {
                "script": "?[wf, step, k, by] := wf = $wf, step = $s, k = to_string(rand_uuid_v4()), by = 'x' :put ran {wf, step, k => by}",
                "params": {"wf": "$input", "s": s}}})
        };
        call(&db, "define", json!({"name": "three", "graph": {"steps": [
            step("a"),
            {"fan_out": {"branches": [[step("b1")], [step("b2")]]}},
            step("c")]}}));
        for i in 0..n {
            let id = format!("t{i}");
            call(&db, "start", json!({"workflow": "three", "id": id, "input": id}));
        }
    }
    let exe = std::env::current_exe().unwrap();
    let kids: Vec<_> = ["p1", "p2"]
        .iter()
        .map(|me| {
            std::process::Command::new(&exe)
                .args(["tick_child", "--exact", "--ignored", "--test-threads=1", "--nocapture"])
                .env("KLEOS_FLOW_DB", &path)
                .env("KLEOS_FLOW_ME", me)
                .spawn()
                .unwrap()
        })
        .collect();
    for mut k in kids {
        assert!(k.wait().unwrap().success());
    }
    let db = DbInstance::new("sqlite", &path, "").unwrap();
    let per = q(&db, "?[wf, step, count(k)] := *ran{wf, step, k}");
    assert_eq!(per.len(), n * 4, "every workflow ran each of its 4 steps");
    assert!(per.iter().all(|r| r[2] == DataValue::from(1)), "a step ran twice: {per:?}");
    // who ran them (the last claim's worker stays on a finished workflow)
    let by = q(&db, "?[owner, count(id)] := *kleos_flow_wf{id, owner, status: 'completed'}");
    let total: i64 = by.iter().map(|r| r[1].get_int().unwrap()).sum();
    assert_eq!(total, n as i64);
    eprintln!("completed by worker: {by:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

