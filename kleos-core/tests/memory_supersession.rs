/*
 * Copyright 2026, The kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Agent-memory supersession and provenance (UPGRADE_PLAN E1/E2;
//! `docs/specs/memory-supersession-provenance.md`), driven through one
//! realistic timeline: a user changes job, moves city, a typo is corrected,
//! facts are re-asserted and arrive out of order, sources are retracted and
//! finally erased for GDPR — with vector and full-text hits checked.
//!
//! sqlite-backed (the stored/scan path); the temp dir follows `TMPDIR`.

use std::collections::BTreeMap;

use kleos::{DataValue, DbInstance, NamedRows, ScriptMutability};

const DAY: i64 = 86_400_000_000;
/// 2026-01-01T00:00:00Z in µs
const T0: i64 = 1_767_225_600_000_000;

fn d(n: i64) -> i64 {
    T0 + n * DAY
}

fn try_run(db: &DbInstance, script: &str) -> miette::Result<NamedRows> {
    db.run_script(script, BTreeMap::new(), ScriptMutability::Mutable)
}

fn run(db: &DbInstance, script: &str) -> NamedRows {
    try_run(db, script).unwrap_or_else(|e| panic!("script failed: {e:?}\n{script}"))
}

fn s(v: &DataValue) -> String {
    match v {
        DataValue::Str(s) => s.to_string(),
        DataValue::Num(n) => format!("{n}"),
        DataValue::Bool(b) => b.to_string(),
        DataValue::List(l) => format!("[{}]", l.iter().map(s).collect::<Vec<_>>().join(",")),
        DataValue::Null => "null".to_string(),
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

fn col(rows: &NamedRows) -> Vec<String> {
    let mut v: Vec<String> = rows.rows.iter().map(|r| s(&r[0])).collect();
    v.sort();
    v
}

fn now_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64
}

/// A transaction-time instant strictly between the previous and the next
/// commit (tt is wall-clock floored and monotonic).
fn tt_mark() -> i64 {
    std::thread::sleep(std::time::Duration::from_millis(3));
    let t = now_us();
    std::thread::sleep(std::time::Duration::from_millis(3));
    t
}

fn open() -> (tempfile::TempDir, DbInstance) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.db");
    let db = DbInstance::new("sqlite", path.to_str().unwrap(), "").unwrap();
    (dir, db)
}

fn schema(db: &DbInstance) {
    run(db, ":create episode {id: String, vt: Validity => text: String}");
    run(
        db,
        ":create fact {subject: String, predicate: String, object: String, vt: Validity, \
         tt: TxTime => confidence: Float default 1.0}",
    );
    run(
        db,
        ":create fact_src {subject: String, predicate: String, object: String, episode: String}",
    );
    run(db, "::supersede create fact {subject, predicate}");
    run(
        db,
        "::provenance create fact_src {derived: fact, source: episode}",
    );
}

fn episode(db: &DbInstance, id: &str, day: i64, text: &str) {
    run(
        db,
        &format!(
            "?[id, vt, text] <- [['{id}', [{}, true], '{text}']] :put episode {{id, vt => text}}",
            d(day)
        ),
    );
}

/// One extraction event: the fact and its provenance link, one transaction.
fn fact(db: &DbInstance, s: &str, p: &str, o: &str, day: i64, eps: &[&str]) {
    let links = eps
        .iter()
        .map(|e| format!("['{s}', '{p}', '{o}', '{e}']"))
        .collect::<Vec<_>>()
        .join(", ");
    run(
        db,
        &format!(
            "{{?[subject, predicate, object, vt] <- [['{s}', '{p}', '{o}', [{t}, true]]] \
               :put fact {{subject, predicate, object, vt}}}} \
             {{?[subject, predicate, object, episode] <- [{links}] \
               :put fact_src {{subject, predicate, object, episode}}}}",
            t = d(day)
        ),
    );
}

fn objects_at(db: &DbInstance, subject: &str, predicate: &str, at: i64) -> Vec<String> {
    col(&run(
        db,
        &format!(
            "?[o] := *fact{{subject: '{subject}', predicate: '{predicate}', object: o @ {at}}}"
        ),
    ))
}

fn objects_known(db: &DbInstance, subject: &str, predicate: &str, at: i64, tt: i64) -> Vec<String> {
    col(&run(
        db,
        &format!(
            "?[o] := *fact{{subject: '{subject}', predicate: '{predicate}', object: o \
             @ (vt: {at}, tt: {tt})}}"
        ),
    ))
}

fn history_len(db: &DbInstance, subject: &str, predicate: &str, object: &str) -> usize {
    run(
        db,
        &format!("::history fact [['{subject}', '{predicate}', '{object}']]"),
    )
    .rows
    .len()
}

#[test]
fn agent_memory_timeline() {
    let (_dir, db) = open();
    schema(&db);

    // --- E1: job and city changes ------------------------------------------
    episode(&db, "ep1", 1, "Alice joined Acme; she lives in Paris");
    fact(&db, "alice", "works_at", "Acme", 1, &["ep1"]);
    fact(&db, "alice", "lives_in", "Paris", 1, &["ep1"]);

    episode(&db, "ep2", 5, "Alice moved to Berlin");
    fact(&db, "alice", "lives_in", "Berlin", 5, &["ep2"]);

    episode(&db, "ep3", 10, "Alice now works at Globex");
    fact(&db, "alice", "works_at", "Globex", 10, &["ep3"]);
    let tt_before_late = tt_mark();

    assert_eq!(objects_at(&db, "alice", "lives_in", d(3)), ["Paris"]);
    assert_eq!(objects_at(&db, "alice", "lives_in", d(6)), ["Berlin"]);
    assert_eq!(objects_at(&db, "alice", "works_at", d(8)), ["Acme"]);
    assert_eq!(objects_at(&db, "alice", "works_at", d(11)), ["Globex"]);
    // nothing was deleted: the superseded row still has its full history
    assert_eq!(history_len(&db, "alice", "lives_in", "Paris"), 2);

    // --- out-of-order arrival ----------------------------------------------
    // learned late: Alice was at Initech between Acme and Globex …
    episode(&db, "ep6", 20, "Alice mentions she spent a while at Initech, and before Acme at a startup");
    fact(&db, "alice", "works_at", "Initech", 7, &["ep6"]);
    // … and at a startup before Acme. Neither may end the newer job.
    fact(&db, "alice", "works_at", "Startup", 0, &["ep6"]);

    assert_eq!(objects_at(&db, "alice", "works_at", d(0) + 3_600_000_000), ["Startup"]);
    assert_eq!(objects_at(&db, "alice", "works_at", d(3)), ["Acme"]);
    assert_eq!(objects_at(&db, "alice", "works_at", d(8)), ["Initech"]);
    assert_eq!(objects_at(&db, "alice", "works_at", d(11)), ["Globex"]);
    // known at T: before the late facts arrived we believed Acme on day 8
    assert_eq!(
        objects_known(&db, "alice", "works_at", d(8), tt_before_late),
        ["Acme"]
    );
    assert!(objects_known(&db, "alice", "works_at", d(0) + 3_600_000_000, tt_before_late).is_empty());

    // --- typo correction (same valid time, later transaction time) ---------
    episode(&db, "ep4", 12, "Bob started at Initec");
    fact(&db, "bob", "works_at", "Initec", 12, &["ep4"]);
    let tt_typo = tt_mark();
    fact(&db, "bob", "works_at", "Initech", 12, &["ep4"]);
    assert_eq!(objects_at(&db, "bob", "works_at", d(13)), ["Initech"]);
    assert_eq!(objects_known(&db, "bob", "works_at", d(13), tt_typo), ["Initec"]);

    // --- idempotent re-assertion -------------------------------------------
    let before = history_len(&db, "alice", "works_at", "Globex");
    episode(&db, "ep5", 15, "Alice still works at Globex");
    fact(&db, "alice", "works_at", "Globex", 15, &["ep5"]);
    assert_eq!(history_len(&db, "alice", "works_at", "Globex"), before);
    assert_eq!(before, 1);

    // --- what changed between two dates ------------------------------------
    let ch = run(
        &db,
        &format!(
            "::changes fact {{from: {}, to: {}, prefix: ['alice']}}",
            d(4),
            d(11)
        ),
    );
    assert_eq!(
        ch.headers,
        ["subject", "predicate", "object", "change", "at", "confidence"]
    );
    let got: Vec<(String, String, i64)> = ch
        .rows
        .iter()
        .map(|r| (s(&r[2]), s(&r[3]), r[4].get_int().unwrap()))
        .collect();
    assert_eq!(
        got,
        [
            ("Berlin".into(), "added".into(), d(5)),
            ("Paris".into(), "ended".into(), d(5)),
            ("Acme".into(), "ended".into(), d(7)),
            ("Initech".into(), "added".into(), d(7)),
            ("Globex".into(), "added".into(), d(10)),
            ("Initech".into(), "ended".into(), d(10)),
        ]
    );
    // the same window as known before the late facts arrived
    let ch_then = run(
        &db,
        &format!(
            "::changes fact {{from: {}, to: {}, prefix: ['alice', 'works_at'], as_of: {tt_before_late}}}",
            d(4),
            d(11)
        ),
    );
    assert_eq!(
        table(&ch_then)
            .into_iter()
            .map(|r| (r[2].clone(), r[3].clone()))
            .collect::<Vec<_>>(),
        [
            ("Acme".to_string(), "ended".to_string()),
            ("Globex".to_string(), "added".to_string())
        ]
    );
    // a corrected typo never was current: it does not show up as a change
    let bob = run(
        &db,
        &format!("::changes fact {{from: {}, to: {}, prefix: ['bob']}}", d(11), d(13)),
    );
    assert_eq!(
        table(&bob).into_iter().map(|r| r[2].clone()).collect::<Vec<_>>(),
        ["Initech"]
    );

    // --- E2: provenance cascade on source retraction ------------------------
    run(&db, ":create summary {sid: String, vt: Validity => text: String}");
    run(
        &db,
        ":create summary_src {sid: String, subject: String, predicate: String, object: String}",
    );
    run(
        &db,
        "::provenance create summary_src {derived: summary, source: fact}",
    );
    run(
        &db,
        &format!(
            "{{?[sid, vt, text] <- [['s1', [{}, true], 'Alice lives in Berlin']] :put summary {{sid, vt => text}}}} \
             {{?[sid, subject, predicate, object] <- [['s1', 'alice', 'lives_in', 'Berlin']] \
               :put summary_src {{sid, subject, predicate, object}}}}",
            d(6)
        ),
    );

    // Globex has two sources (ep3, ep5): retracting one keeps it
    let r = run(&db, "::retract episode [['ep3']]");
    assert_eq!(table(&r), [["0", "episode", "[ep3]", "retracted"]]);
    assert_eq!(objects_at(&db, "alice", "works_at", now_us()), ["Globex"]);

    // Berlin was stated only in ep2: it ends, and so does the summary
    // derived from it (transitively)
    let r = run(&db, "::retract episode [['ep2']]");
    assert_eq!(
        table(&r),
        [
            vec!["0", "episode", "[ep2]", "retracted"],
            vec!["1", "fact", "[alice,lives_in,Berlin]", "retracted"],
            vec!["2", "summary", "[s1]", "retracted"],
        ]
    );
    let now = now_us();
    assert!(objects_at(&db, "alice", "lives_in", now).is_empty());
    assert!(col(&run(&db, &format!("?[s] := *summary{{sid: s @ {now}}}"))).is_empty());
    // history is intact
    assert_eq!(objects_at(&db, "alice", "lives_in", d(6)), ["Berlin"]);
    assert_eq!(
        col(&run(&db, &format!("?[s] := *summary{{sid: s @ {}}}", d(7)))),
        ["s1"]
    );

    // lineage explains a fact back to its episodes
    let lin = run(&db, "::lineage summary [['s1']]");
    assert_eq!(lin.headers, ["depth", "relation", "key", "via", "current"]);
    assert_eq!(
        table(&lin),
        [
            vec!["0", "summary", "[s1]", "null", "false"],
            vec!["1", "fact", "[alice,lives_in,Berlin]", "summary_src", "false"],
            vec!["2", "episode", "[ep2]", "fact_src", "false"],
        ]
    );
    let lin = run(&db, "::lineage fact [['alice', 'works_at', 'Globex']]");
    assert_eq!(
        table(&lin)
            .into_iter()
            .filter(|r| r[1] == "episode")
            .map(|r| (r[2].clone(), r[4].clone()))
            .collect::<Vec<_>>(),
        [
            ("[ep3]".to_string(), "false".to_string()),
            ("[ep5]".to_string(), "true".to_string())
        ]
    );

    // --- GDPR erasure, search indexes included ------------------------------
    run(
        &db,
        ":create note {nid: String, vt: Validity => text: String, emb: <F32; 2>}",
    );
    run(
        &db,
        "::hnsw create note:vec {dim: 2, m: 8, dtype: F32, fields: [emb], distance: L2, ef_construction: 32}",
    );
    run(
        &db,
        "::fts create note:text {extractor: text, tokenizer: Simple, filters: [Lowercase]}",
    );
    run(&db, ":create note_src {nid: String, ep: String}");
    run(&db, "::provenance create note_src {derived: note, source: episode}");
    episode(&db, "ep7", 30, "Alice phone number is 555-0100");
    episode(&db, "ep8", 31, "Alice likes jazz");
    run(
        &db,
        &format!(
            "{{?[nid, vt, text, emb] <- [['n1', [{a}, true], 'phone 555-0100', vec([1.0, 0.0])], \
                                        ['n2', [{b}, true], 'phone and jazz', vec([0.9, 0.1])], \
                                        ['n3', [{b}, true], 'jazz', vec([0.0, 1.0])]] \
               :put note {{nid, vt => text, emb}}}} \
             {{?[nid, ep] <- [['n1', 'ep7'], ['n2', 'ep7'], ['n2', 'ep8'], ['n3', 'ep8']] \
               :put note_src {{nid, ep}}}}",
            a = d(30),
            b = d(31)
        ),
    );
    fact(&db, "alice", "phone", "555-0100", 30, &["ep7"]);
    run(
        &db,
        &format!(
            "{{?[sid, vt, text] <- [['s2', [{}, true], 'reach Alice at 555-0100']] :put summary {{sid, vt => text}}}} \
             {{?[sid, subject, predicate, object] <- [['s2', 'alice', 'phone', '555-0100']] \
               :put summary_src {{sid, subject, predicate, object}}}}",
            d(30)
        ),
    );
    let knn = "?[nid] := ~note:vec{nid | query: vec([1.0, 0.0]), k: 10, ef: 32}";
    let fts = "?[nid] := ~note:text{nid | query: 'phone', k: 10}";
    assert_eq!(col(&run(&db, knn)), ["n1", "n2", "n3"]);
    assert_eq!(col(&run(&db, fts)), ["n1", "n2"]);

    let r = run(&db, "::erase episode [['ep7']]");
    assert_eq!(
        table(&r),
        [
            vec!["0", "episode", "[ep7]", "erased"],
            vec!["1", "fact", "[alice,phone,555-0100]", "erased"],
            vec!["1", "note", "[n1]", "erased"],
            vec!["1", "note", "[n2]", "unlinked"],
            vec!["2", "summary", "[s2]", "erased"],
        ]
    );
    // every row of every erased key is physically gone, at every time
    assert!(run(&db, "?[v] := *episode{id: 'ep7', vt: v}").rows.is_empty());
    assert!(run(&db, "?[v] := *note{nid: 'n1', vt: v}").rows.is_empty());
    assert!(run(&db, "?[v] := *summary{sid: 's2', vt: v}").rows.is_empty());
    assert!(run(&db, "?[o] := *fact{subject: 'alice', predicate: 'phone', object: o}")
        .rows
        .is_empty());
    assert_eq!(history_len(&db, "alice", "phone", "555-0100"), 0);
    assert!(run(&db, "?[n] := *note_src{nid: n, ep: 'ep7'}").rows.is_empty());
    assert!(run(&db, "?[o] := *fact_src{object: o, episode: 'ep7'}").rows.is_empty());
    // … and so are their search-index entries
    assert_eq!(col(&run(&db, knn)), ["n2", "n3"]);
    assert_eq!(col(&run(&db, fts)), ["n2"]);
    // n2 keeps its other source
    assert_eq!(
        col(&run(&db, "?[e] := *note_src{nid: 'n2', ep: e}")),
        ["ep8"]
    );
    // audit markers: one per erased key, salted hashes, no plaintext
    let audit = run(&db, "?[r, k] := *kleos_evict_audit{relation: r, key: k}");
    assert_eq!(
        col(&audit),
        ["episode", "fact", "note", "summary"]
    );
    for row in &audit.rows {
        let k = s(&row[1]);
        assert!(!k.contains("555") && !k.contains("ep7") && !k.contains("n1"), "{k}");
    }
}

#[test]
fn same_transaction_writes_see_each_other() {
    let (_dir, db) = open();
    schema(&db);
    // two statements in one transaction: the second must supersede the
    // first even though TxTime rows are buffered until commit
    run(
        &db,
        &format!(
            "{{?[subject, predicate, object, vt] <- [['carol', 'works_at', 'A', [{a}, true]]] \
               :put fact {{subject, predicate, object, vt}}}} \
             {{?[subject, predicate, object, vt] <- [['carol', 'works_at', 'B', [{b}, true]]] \
               :put fact {{subject, predicate, object, vt}}}}",
            a = d(1),
            b = d(2)
        ),
    );
    assert_eq!(objects_at(&db, "carol", "works_at", d(1)), ["A"]);
    assert_eq!(objects_at(&db, "carol", "works_at", d(3)), ["B"]);

    // one batch, out of order: replayed in valid-time order
    run(
        &db,
        &format!(
            "?[subject, predicate, object, vt] <- [['dave', 'lives_in', 'Rome', [{c}, true]], \
                                                   ['dave', 'lives_in', 'Oslo', [{a}, true]]] \
             :put fact {{subject, predicate, object, vt}}",
            a = d(1),
            c = d(3)
        ),
    );
    assert_eq!(objects_at(&db, "dave", "lives_in", d(2)), ["Oslo"]);
    assert_eq!(objects_at(&db, "dave", "lives_in", d(4)), ["Rome"]);

    // two values for one key at one instant in one transaction is ambiguous
    let err = try_run(
        &db,
        &format!(
            "?[subject, predicate, object, vt] <- [['erin', 'likes', 'tea', [{a}, true]], \
                                                   ['erin', 'likes', 'coffee', [{a}, true]]] \
             :put fact {{subject, predicate, object, vt}}",
            a = d(1)
        ),
    )
    .unwrap_err();
    assert!(format!("{err:?}").contains("supersede_same_instant"), "{err:?}");
}

#[test]
fn valid_time_only_relations() {
    let db = DbInstance::new("mem", "", "").unwrap();
    // object in the key: supersession over siblings
    run(&db, ":create pref {user: String, item: String, vt: Validity => why: String default ''}");
    run(&db, "::supersede create pref {user}");
    let put = |item: &str, day: i64| {
        try_run(
            &db,
            &format!(
                "?[user, item, vt] <- [['u', '{item}', [{}, true]]] :put pref {{user, item, vt}}",
                d(day)
            ),
        )
    };
    put("tea", 1).unwrap();
    put("coffee", 4).unwrap();
    put("coffee", 6).unwrap(); // redundant: no new row
    let at = |t: i64| col(&run(&db, &format!("?[i] := *pref{{user: 'u', item: i @ {t}}}")));
    assert_eq!(at(d(2)), ["tea"]);
    assert_eq!(at(d(5)), ["coffee"]);
    assert_eq!(
        run(&db, "?[i, v] := *pref{user: 'u', item: i, vt: v}").rows.len(),
        3 // tea@1, tea retracted@4, coffee@4
    );
    // a correction at the same instant cannot be expressed without TxTime
    let err = put("juice", 4).unwrap_err();
    assert!(format!("{err:?}").contains("supersede_same_instant"), "{err:?}");

    // object as value: the key alone supersedes; the rule adds dedup
    run(&db, ":create city {user: String, vt: Validity => name: String}");
    run(&db, "::supersede create city {user}");
    for (day, name) in [(1, "Paris"), (2, "Paris"), (3, "Lyon"), (4, "Lyon")] {
        run(
            &db,
            &format!(
                "?[user, vt, name] <- [['u', [{}, true], '{name}']] :put city {{user, vt => name}}",
                d(day)
            ),
        );
    }
    assert_eq!(run(&db, "?[v, n] := *city{user: 'u', vt: v, name: n}").rows.len(), 2);

    // declaration errors
    assert!(try_run(&db, "::supersede create pref {item}").is_err());
    run(&db, ":create plain {k: String => v: String}");
    assert!(try_run(&db, "::supersede create plain {k}").is_err());
    assert_eq!(
        table(&run(&db, "::supersede list")),
        [vec!["city", "[user]"], vec!["pref", "[user]"]]
    );

    // bulk import would bypass supersession: rejected
    let mut data = BTreeMap::new();
    data.insert(
        "pref".to_string(),
        NamedRows::new(
            vec!["user".into(), "item".into(), "vt".into(), "why".into()],
            vec![vec![
                DataValue::from("u"),
                DataValue::from("milk"),
                DataValue::from("ASSERT"),
                DataValue::from(""),
            ]],
        ),
    );
    let err = db.import_relations(data.clone()).unwrap_err();
    assert!(format!("{err:?}").contains("supersession"), "{err:?}");
    run(&db, "::supersede drop pref");
    db.import_relations(data).unwrap();
}

#[test]
fn provenance_declaration_checks() {
    let db = DbInstance::new("mem", "", "").unwrap();
    run(&db, ":create doc {id: Int => body: String}");
    run(&db, ":create claim {c: String, vt: Validity => w: Float default 1.0}");
    run(&db, ":create claim_doc {c: String, id: Int}");
    run(&db, ":create bad_link {id: Int, c: String}");
    // wrong column order / types
    assert!(try_run(&db, "::provenance create bad_link {derived: claim, source: doc}").is_err());
    // derived relation without a time axis cannot be retracted
    assert!(try_run(&db, "::provenance create claim_doc {derived: doc, source: claim}").is_err());
    run(&db, "::provenance create claim_doc {derived: claim, source: doc}");
    assert_eq!(
        table(&run(&db, "::provenance list")),
        [["claim_doc", "claim", "doc"]]
    );
    // a plain source can be erased but not retracted
    run(&db, "?[id, body] <- [[1, 'x']] :put doc {id => body}");
    run(&db, "?[c, vt] <- [['k', 'ASSERT']] :put claim {c, vt}");
    run(&db, "?[c, id] <- [['k', 1]] :put claim_doc {c, id}");
    assert!(try_run(&db, "::retract doc [[1]]").is_err());
    let r = run(&db, "::erase doc [[1]]");
    assert_eq!(
        table(&r),
        [vec!["0", "doc", "[1]", "erased"], vec!["1", "claim", "[k]", "erased"]]
    );
    run(&db, "::provenance drop claim_doc");
    assert!(run(&db, "::provenance list").rows.is_empty());
}
