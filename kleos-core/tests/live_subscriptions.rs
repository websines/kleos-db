/*
 * Copyright 2026, The kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Live query subscriptions (UPGRADE_PLAN F1). Spec:
//! `docs/specs/live-subscriptions.md`.
//!
//! Run with: `cargo test -p kleos --test live_subscriptions`
//! (add `--features test-hooks` for the failed-commit case).

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use kleos::{
    DataValue, DbInstance, NamedRows, ScriptMutability, SubscribeOptions, Subscription,
    SubscriptionEvent,
};

const DAY: i64 = 86_400_000_000;
const T0: i64 = 1_767_225_600_000_000; // 2026-01-01T00:00:00Z

fn run(db: &DbInstance, script: &str) -> NamedRows {
    db.run_script(script, BTreeMap::new(), ScriptMutability::Mutable)
        .unwrap_or_else(|e| panic!("script failed: {e:?}\n{script}"))
}

fn sub(db: &DbInstance, script: &str) -> Subscription {
    db.subscribe(script, BTreeMap::new(), SubscribeOptions::default())
        .unwrap_or_else(|e| panic!("subscribe failed: {e:?}\n{script}"))
}

fn s(v: &DataValue) -> String {
    match v {
        DataValue::Str(s) => s.to_string(),
        DataValue::Num(n) => format!("{n}"),
        v => format!("{v:?}"),
    }
}

fn table(rows: &NamedRows) -> Vec<Vec<String>> {
    let mut t: Vec<Vec<String>> = rows
        .rows
        .iter()
        .map(|r| r.iter().map(s).collect())
        .collect();
    t.sort();
    t
}

fn t(rows: &[&[&str]]) -> Vec<Vec<String>> {
    let mut v: Vec<Vec<String>> = rows
        .iter()
        .map(|r| r.iter().map(|x| x.to_string()).collect())
        .collect();
    v.sort();
    v
}

/// The next event, or panic after a generous timeout.
fn next(sub: &Subscription) -> SubscriptionEvent {
    sub.recv_timeout(Duration::from_secs(10))
        .expect("expected a subscription event")
}

fn initial(sub: &Subscription) -> Vec<Vec<String>> {
    match next(sub) {
        SubscriptionEvent::Initial { rows } => table(&rows),
        e => panic!("expected Initial, got {e:?}"),
    }
}

/// `(added, removed)` of the next event, which must be a Delta.
fn delta(sub: &Subscription) -> (Vec<Vec<String>>, Vec<Vec<String>>) {
    match next(sub) {
        SubscriptionEvent::Delta { added, removed } => (table(&added), table(&removed)),
        e => panic!("expected Delta, got {e:?}"),
    }
}

/// Assert nothing arrives. The worker is woken within microseconds of a
/// commit; 300 ms is far beyond its latency on an idle machine.
fn quiet(sub: &Subscription) {
    if let Some(e) = sub.recv_timeout(Duration::from_millis(300)) {
        panic!("expected no event, got {e:?}");
    }
}

/// Wait until the worker has performed at least `n` evaluations in total.
fn wait_evals(db: &DbInstance, n: u64) {
    let start = Instant::now();
    while db.live_stats().evaluations < n {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "worker never reached {n} evaluations: {:?}",
            db.live_stats()
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn kv_db() -> DbInstance {
    let db = DbInstance::new("mem", "", "").unwrap();
    run(&db, ":create kv {k: Int => v: String}");
    run(&db, ":create other {k: Int => v: String}");
    run(&db, "?[k, v] <- [[1, 'a'], [2, 'b']] :put kv {k => v}");
    db
}

#[test]
fn initial_result_then_deltas_on_put_and_rm() {
    let db = kv_db();
    let s1 = sub(&db, "?[k, v] := *kv{k, v}");
    assert_eq!(initial(&s1), t(&[&["1", "a"], &["2", "b"]]));

    run(&db, "?[k, v] <- [[3, 'c']] :put kv {k => v}");
    assert_eq!(delta(&s1), (t(&[&["3", "c"]]), vec![]));

    // An update is one removal plus one addition.
    run(&db, "?[k, v] <- [[1, 'z']] :put kv {k => v}");
    assert_eq!(delta(&s1), (t(&[&["1", "z"]]), t(&[&["1", "a"]])));

    run(&db, "?[k] <- [[2]] :rm kv {k}");
    assert_eq!(delta(&s1), (vec![], t(&[&["2", "b"]])));

    // A write that leaves the result unchanged is re-evaluated, not pushed.
    run(&db, "?[k, v] <- [[3, 'c']] :put kv {k => v}");
    quiet(&s1);
}

#[test]
fn filtered_query_only_reports_rows_that_match() {
    let db = kv_db();
    let s1 = sub(&db, "?[k] := *kv{k, v}, v == 'x'");
    assert_eq!(initial(&s1), Vec::<Vec<String>>::new());
    run(&db, "?[k, v] <- [[7, 'y']] :put kv {k => v}");
    quiet(&s1);
    run(&db, "?[k, v] <- [[8, 'x']] :put kv {k => v}");
    assert_eq!(delta(&s1), (t(&[&["8"]]), vec![]));
}

#[test]
fn writes_to_unrelated_relations_do_not_wake_the_subscription() {
    let db = kv_db();
    let s1 = sub(&db, "?[k, v] := *kv{k, v}");
    initial(&s1);
    let before = db.live_stats();
    for i in 0..20 {
        run(&db, &format!("?[k, v] <- [[{i}, 'o']] :put other {{k => v}}"));
    }
    quiet(&s1);
    let after = db.live_stats();
    // Filtered on the writer's side: not even counted as a notification.
    assert_eq!(after.notified_commits, before.notified_commits);
    assert_eq!(after.evaluations, before.evaluations);
}

#[test]
fn rolled_back_transactions_deliver_nothing() {
    let db = kv_db();
    let s1 = sub(&db, "?[k, v] := *kv{k, v}");
    initial(&s1);

    // Multi-statement transaction, aborted.
    let tx = db.multi_transaction(true);
    tx.run_script("?[k, v] <- [[10, 'ten']] :put kv {k => v}", BTreeMap::new())
        .unwrap();
    tx.run_script("?[k] <- [[1]] :rm kv {k}", BTreeMap::new())
        .unwrap();
    tx.abort().unwrap();
    quiet(&s1);

    // A chained script whose last block fails: the first block's write is
    // rolled back with it.
    let err = db.run_script(
        "{?[k, v] <- [[11, 'eleven']] :put kv {k => v}} {?[x] <- [[1]] :assert none}",
        BTreeMap::new(),
        ScriptMutability::Mutable,
    );
    assert!(err.is_err());
    quiet(&s1);

    // A committed multi-statement transaction is delivered as ONE delta.
    let tx = db.multi_transaction(true);
    tx.run_script("?[k, v] <- [[12, 'twelve']] :put kv {k => v}", BTreeMap::new())
        .unwrap();
    tx.run_script("?[k] <- [[1]] :rm kv {k}", BTreeMap::new())
        .unwrap();
    tx.commit().unwrap();
    assert_eq!(delta(&s1), (t(&[&["12", "twelve"]]), t(&[&["1", "a"]])));
    quiet(&s1);
}

#[cfg(feature = "test-hooks")]
#[test]
fn failed_commit_delivers_nothing() {
    let db = kv_db();
    let s1 = sub(&db, "?[k, v] := *kv{k, v}");
    initial(&s1);
    let DbInstance::Mem(inner) = &db else {
        unreachable!()
    };
    inner.fail_next_commit_for_tests();
    let tx = db.multi_transaction(true);
    tx.run_script("?[k, v] <- [[20, 'x']] :put kv {k => v}", BTreeMap::new())
        .unwrap();
    assert!(tx.commit().is_err());
    quiet(&s1);
}

#[test]
fn identical_subscriptions_share_one_evaluation() {
    let db = kv_db();
    let q = "?[k, v] := *kv{k, v}, k >= $min";
    let p = BTreeMap::from([("min".to_string(), DataValue::from(2))]);
    let a = db
        .subscribe(q, p.clone(), SubscribeOptions::default())
        .unwrap();
    let b = db
        .subscribe(q, p.clone(), SubscribeOptions::default())
        .unwrap();
    assert_eq!(initial(&a), t(&[&["2", "b"]]));
    assert_eq!(initial(&b), t(&[&["2", "b"]]));
    let st = db.live_stats();
    assert_eq!((st.subscriptions, st.groups, st.evaluations), (2, 1, 1));

    // Different params: a separate group.
    let p1 = BTreeMap::from([("min".to_string(), DataValue::from(1))]);
    let c = db.subscribe(q, p1, SubscribeOptions::default()).unwrap();
    assert_eq!(initial(&c).len(), 2);
    assert_eq!(db.live_stats().groups, 2);

    run(&db, "?[k, v] <- [[5, 'e']] :put kv {k => v}");
    assert_eq!(delta(&a), (t(&[&["5", "e"]]), vec![]));
    assert_eq!(delta(&b), (t(&[&["5", "e"]]), vec![]));
    assert_eq!(delta(&c), (t(&[&["5", "e"]]), vec![]));
    // 3 initial-or-joined... 2 initial evaluations + 1 per group for the write.
    assert_eq!(db.live_stats().evaluations, 4);

    // A late joiner gets the group's current snapshot as its Initial.
    let d = db.subscribe(q, p, SubscribeOptions::default()).unwrap();
    assert_eq!(initial(&d), t(&[&["2", "b"], &["5", "e"]]));
    assert_eq!(db.live_stats().evaluations, 4);
}

#[test]
fn unsubscribe_stops_events() {
    let db = kv_db();
    let a = sub(&db, "?[k] := *kv{k}");
    let b = sub(&db, "?[k] := *kv{k}");
    initial(&a);
    initial(&b);

    // By id: the handle's channel closes.
    assert!(db.unsubscribe(a.id()));
    assert!(!db.unsubscribe(a.id()));
    run(&db, "?[k, v] <- [[9, 'i']] :put kv {k => v}");
    assert!(a.recv_timeout(Duration::from_millis(300)).is_none());
    assert_eq!(delta(&b), (t(&[&["9"]]), vec![]));

    // By drop: the group goes away with its last subscriber.
    drop(a);
    drop(b);
    let st = db.live_stats();
    assert_eq!((st.subscriptions, st.groups), (0, 0));
    let evals = st.evaluations;
    run(&db, "?[k, v] <- [[10, 'j']] :put kv {k => v}");
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(db.live_stats().evaluations, evals);

    // The worker restarts for a new subscription.
    let c = sub(&db, "?[k] := *kv{k}");
    assert_eq!(initial(&c).len(), 4);
    run(&db, "?[k] <- [[10]] :rm kv {k}");
    assert_eq!(delta(&c), (vec![], t(&[&["10"]])));
}

#[test]
fn slow_subscriber_is_coalesced_into_a_resync() {
    let db = kv_db();
    let slow = db
        .subscribe(
            "?[k, v] := *kv{k, v}",
            BTreeMap::new(),
            SubscribeOptions { queue_capacity: 2 },
        )
        .unwrap();
    // A fast subscriber to the same query: it must keep getting every delta.
    let fast = sub(&db, "?[k, v] := *kv{k, v}");
    initial(&fast);
    let mut evals = db.live_stats().evaluations;
    for i in 100..110 {
        run(&db, &format!("?[k, v] <- [[{i}, 'n']] :put kv {{k => v}}"));
        evals += 1;
        // One re-evaluation per write, so each write is its own delta.
        wait_evals(&db, evals);
        assert_eq!(delta(&fast).0, t(&[&[&i.to_string(), "n"]]));
    }
    // `slow` never read: its queue (Initial + 1 delta) filled up, and the
    // remaining 9 deltas were dropped. Its next event is the full result.
    match next(&slow) {
        SubscriptionEvent::Resync { rows } => assert_eq!(rows.rows.len(), 12),
        e => panic!("expected Resync, got {e:?}"),
    }
    quiet(&slow);
    // After the resync it is a normal subscriber again.
    run(&db, "?[k] <- [[100]] :rm kv {k}");
    assert_eq!(delta(&slow), (vec![], t(&[&["100", "n"]])));
}

#[test]
fn burst_of_commits_is_coalesced() {
    let db = kv_db();
    let s1 = sub(&db, "?[count(k)] := *kv{k}");
    assert_eq!(initial(&s1), t(&[&["2"]]));
    let evals = db.live_stats().evaluations;
    let writers: Vec<_> = (0..4)
        .map(|w| {
            let db = db.clone();
            std::thread::spawn(move || {
                for i in 0..50 {
                    let k = 1000 + w * 100 + i;
                    run(&db, &format!("?[k, v] <- [[{k}, 'w']] :put kv {{k => v}}"));
                }
            })
        })
        .collect();
    for w in writers {
        w.join().unwrap();
    }
    // The last delta reaches 202; how many deltas it took depends on the
    // scheduler, but never more than one per commit.
    let start = Instant::now();
    let mut last = String::new();
    while last != "202" {
        assert!(start.elapsed() < Duration::from_secs(10));
        if let SubscriptionEvent::Delta { added, .. } = next(&s1) {
            last = s(&added.rows[0][0]);
        }
    }
    let reevals = db.live_stats().evaluations - evals;
    assert!(reevals <= 200, "{reevals} re-evaluations for 200 commits");
    eprintln!("200 commits from 4 threads -> {reevals} re-evaluations");
}

#[test]
fn dependencies_are_tracked_through_rule_chains_and_fixed_rules() {
    let db = DbInstance::new("mem", "", "").unwrap();
    run(&db, ":create edge {a: String, b: String}");
    run(&db, ":create label {n: String => text: String}");
    run(&db, ":create noise {n: String}");
    run(&db, "?[a, b] <- [['a', 'b'], ['b', 'c']] :put edge {a, b}");
    run(&db, "?[n, text] <- [['c', 'C!'], ['d', 'D!']] :put label {n => text}");

    // Two rule layers: recursive reachability over `edge`, then a join with
    // `label`. Both relations are dependencies; `noise` is not.
    let s1 = sub(
        &db,
        "reach[x, y] := *edge{a: x, b: y}
         reach[x, z] := reach[x, y], *edge{a: y, b: z}
         named[t] := reach['a', y], *label{n: y, text: t}
         ?[t] := named[t]",
    );
    assert_eq!(initial(&s1), t(&[&["C!"]]));

    run(&db, "?[a, b] <- [['c', 'd']] :put edge {a, b}");
    assert_eq!(delta(&s1), (t(&[&["D!"]]), vec![]));

    run(&db, "?[n, text] <- [['c', 'see']] :put label {n => text}");
    assert_eq!(delta(&s1), (t(&[&["see"]]), t(&[&["C!"]])));

    let evals = db.live_stats().evaluations;
    run(&db, "?[n] <- [['x']] :put noise {n}");
    quiet(&s1);
    assert_eq!(db.live_stats().evaluations, evals);

    // A fixed rule's stored input relation is a dependency too.
    let s2 = sub(&db, "?[n, c] <~ ConnectedComponents(*edge[a, b])");
    let comps = initial(&s2);
    assert_eq!(comps.len(), 4);
    run(&db, "?[a, b] <- [['p', 'q']] :put edge {a, b}");
    let (added, removed) = delta(&s2);
    assert_eq!(added.len(), 2);
    assert!(removed.is_empty());
}

#[test]
fn fts_search_depends_on_its_base_relation() {
    let db = DbInstance::new("mem", "", "").unwrap();
    run(&db, ":create doc {k: String => body: String}");
    run(
        &db,
        "::fts create doc:fts { extractor: body, tokenizer: Simple, filters: [Lowercase] }",
    );
    run(&db, "?[k, body] <- [['d1', 'apple pie']] :put doc {k => body}");
    let s1 = sub(&db, "?[k] := ~doc:fts{k | query: 'apple', k: 10}");
    assert_eq!(initial(&s1), t(&[&["d1"]]));

    run(&db, "?[k, body] <- [['d2', 'Apple tart']] :put doc {k => body}");
    assert_eq!(delta(&s1), (t(&[&["d2"]]), vec![]));

    // Written, re-evaluated, but the search result did not change.
    let evals = db.live_stats().evaluations;
    run(&db, "?[k, body] <- [['d3', 'banana']] :put doc {k => body}");
    quiet(&s1);
    assert_eq!(db.live_stats().evaluations, evals + 1);

    run(&db, "?[k] <- [['d1']] :rm doc {k}");
    assert_eq!(delta(&s1), (vec![], t(&[&["d1"]])));
}

#[test]
fn hnsw_search_depends_on_its_base_relation() {
    let db = DbInstance::new("mem", "", "").unwrap();
    run(&db, ":create pts {id: Int => emb: <F32; 2>}");
    run(
        &db,
        "::hnsw create pts:idx { dim: 2, m: 8, dtype: F32, fields: [emb], \
         distance: L2, ef_construction: 32 }",
    );
    run(
        &db,
        "?[id, emb] <- [[1, vec([0.0, 1.0])], [2, vec([-1.0, 0.0])]] :put pts {id => emb}",
    );
    let s1 = sub(&db, "?[id] := ~pts:idx{id | query: vec([1.0, 0.0]), k: 1, ef: 16}");
    assert_eq!(initial(&s1), t(&[&["1"]]));
    run(&db, "?[id, emb] <- [[3, vec([0.9, 0.1])]] :put pts {id => emb}");
    assert_eq!(delta(&s1), (t(&[&["3"]]), t(&[&["1"]])));

    // Reading the index relation directly also follows the base relation.
    let s2 = sub(&db, "?[count(fr)] := *pts:idx{layer: 0, fr_id: fr}");
    initial(&s2);
    run(&db, "?[id, emb] <- [[4, vec([0.5, 0.5])]] :put pts {id => emb}");
    assert!(matches!(next(&s2), SubscriptionEvent::Delta { .. }));
}

#[test]
fn supersession_delivers_the_ended_row_as_removed() {
    let db = DbInstance::new("mem", "", "").unwrap();
    run(
        &db,
        ":create fact {subject: String, predicate: String, object: String, vt: Validity}",
    );
    run(&db, "::supersede create fact {subject, predicate}");
    let s1 = sub(
        &db,
        "?[o] := *fact{subject: 'alice', predicate: 'works_at', object: o @ 'NOW'}",
    );
    assert_eq!(initial(&s1), Vec::<Vec<String>>::new());

    let put = |o: &str, at: i64| {
        run(
            &db,
            &format!(
                "?[subject, predicate, object, vt] <- [['alice', 'works_at', '{o}', [{at}, true]]] \
                 :put fact {{subject, predicate, object, vt}}"
            ),
        );
    };
    put("Acme", T0);
    assert_eq!(delta(&s1), (t(&[&["Acme"]]), vec![]));
    // The :put of Globex ends Acme (a retraction row, nothing deleted).
    put("Globex", T0 + DAY);
    assert_eq!(delta(&s1), (t(&[&["Globex"]]), t(&[&["Acme"]])));
    // Re-asserting the current fact writes nothing: no event.
    put("Globex", T0 + 2 * DAY);
    quiet(&s1);
    // History is still there for an as-of subscription.
    let s2 = sub(
        &db,
        &format!("?[o] := *fact{{subject: 'alice', predicate: 'works_at', object: o @ {}}}", T0 + DAY / 2),
    );
    assert_eq!(initial(&s2), t(&[&["Acme"]]));
}

#[test]
fn mutations_and_system_ops_are_rejected() {
    let db = kv_db();
    for script in [
        "?[k, v] <- [[1, 'x']] :put kv {k => v}",
        "?[k] <- [[1]] :rm kv {k}",
        "?[a] <- [[1]] :create tmp {a}",
        "::relations",
        "{?[a] <- [[1]]} {?[a] <- [[2]]}",
    ] {
        let err = db
            .subscribe(script, BTreeMap::new(), SubscribeOptions::default())
            .expect_err(script);
        let msg = format!("{err:?}");
        assert!(msg.contains("live subscriptions"), "{script}: {msg}");
    }
    // Nothing was written, and no group was left behind.
    assert_eq!(run(&db, "?[count(k)] := *kv{k}").rows[0][0], DataValue::from(2));
    assert_eq!(db.live_stats().groups, 0);
    // A query that fails on first evaluation is an error too.
    assert!(db
        .subscribe("?[k] := *missing{k}", BTreeMap::new(), Default::default())
        .is_err());
    assert_eq!(db.live_stats().groups, 0);
}

#[test]
fn removed_relation_reports_an_error_then_resyncs() {
    let db = kv_db();
    let s1 = sub(&db, "?[k] := *kv{k}");
    initial(&s1);
    run(&db, "::remove kv");
    assert!(matches!(next(&s1), SubscriptionEvent::Error { .. }));
    // While erroring, the subscription watches every commit, so it picks up
    // the re-created relation (a new id) on its first write.
    run(&db, ":create kv {k: Int => v: String}");
    run(&db, "?[k, v] <- [[42, 'x']] :put kv {k => v}");
    match next(&s1) {
        SubscriptionEvent::Resync { rows } => assert_eq!(table(&rows), t(&[&["42"]])),
        e => panic!("expected Resync, got {e:?}"),
    }
    run(&db, "?[k, v] <- [[43, 'y']] :put kv {k => v}");
    assert_eq!(delta(&s1), (t(&[&["43"]]), vec![]));
}
