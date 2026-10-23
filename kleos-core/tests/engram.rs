/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Engram overlay store (docs/specs/engram-overlay.md): deterministic tests
//! with a synthetic spec and synthetic vectors, no model needed.
//! Run: cargo test -p kleos --features engram --test engram

use std::collections::BTreeMap;

use kleos::engram::addressing::{bf16, safetensors, CompressedTokenizer, ConfigFields, EngramSpec};
use kleos::engram::{EngramFact, EngramWrite, RowEdit};
use kleos::{new_cozo_mem, DataValue, Db, MemStorage, ScriptMutability};
use serde_json::json;

const DIM: usize = 4;

fn spec(seed: u64) -> EngramSpec {
    let cfg = json!({"engram_layer_ids": [1, 5], "engram_max_ngram_size": 3, "engram_n_heads": 2,
                     "engram_vocab_size": 101, "engram_head_dim": DIM, "engram_pad_token_id": 0, "engram_seed": seed});
    let fields = ConfigFields::from_model_config(&cfg, &Default::default()).unwrap();
    EngramSpec::from_parts(fields, CompressedTokenizer::from_lookup((0..40).map(|i| i / 2).collect(), None)).unwrap()
}

fn db() -> (Db<MemStorage>, String) {
    let db = new_cozo_mem().unwrap();
    let fp = db.engram_register("m", &spec(0), None).unwrap();
    (db, fp)
}

fn v(x: f32) -> Vec<f32> {
    (0..DIM).map(|i| x + i as f32 * 0.25).collect()
}

fn row(layer: u64, row: i64, x: f32, facts: &[&str]) -> RowEdit {
    RowEdit { layer, row, vec: v(x), facts: facts.iter().map(|s| s.to_string()).collect() }
}

fn fact(id: &str, target: &str, rows: &[(u64, i64)]) -> EngramFact {
    EngramFact {
        id: id.into(),
        text: format!("{id} is {target}"),
        target: target.into(),
        phrasings: vec![format!("{id} is"), format!("they say {id} is")],
        rows: rows.to_vec(),
        decided_by: "test".into(),
        source: "ep1".into(),
        payload: Some(vec![0.5, -1.0]),
    }
}

fn write(db: &Db<MemStorage>, fp: &str, ns: &str, at: i64, facts: Vec<EngramFact>, rows: Vec<RowEdit>, retract: &[&str]) -> Vec<Vec<String>> {
    let w = EngramWrite {
        ns: ns.into(),
        fingerprint: fp.into(),
        at: Some(at),
        facts,
        rows,
        retract: retract.iter().map(|s| s.to_string()).collect(),
    };
    db.engram_write("m", &w)
        .unwrap()
        .rows
        .into_iter()
        .map(|r| r.into_iter().map(|c| c.get_str().unwrap_or_default().to_string()).collect())
        .collect()
}

/// (layer, row) -> f32 values, as materialised.
fn mat(db: &Db<MemStorage>, ns: &[&str], as_of: Option<i64>) -> BTreeMap<(u64, i64), Vec<f32>> {
    let ns: Vec<String> = ns.iter().map(|s| s.to_string()).collect();
    db.engram_materialize("m", &ns, as_of, None)
        .unwrap()
        .rows
        .into_iter()
        .map(|r| ((r.layer, r.row), bf16::to_f32(&r.vec)))
        .collect()
}

fn bf(x: f32) -> Vec<f32> {
    bf16::to_f32(&bf16::from_f32(&v(x)))
}

/// Two facts: a touches (1,10) (1,11) (5,20); b touches (1,11) (5,21). (1,11) is shared.
fn two_facts(db: &Db<MemStorage>, fp: &str, at: i64) {
    let out = write(
        db,
        fp,
        "u1",
        at,
        vec![fact("a", "oat", &[(1, 10), (1, 11), (5, 20)]), fact("b", "tea", &[(1, 11), (5, 21)])],
        vec![row(1, 10, 1.0, &["a"]), row(1, 11, 2.0, &["a", "b"]), row(5, 20, 3.0, &["a"]), row(5, 21, 4.0, &["b"])],
        &[],
    );
    assert!(out.contains(&vec!["fact".into(), "a".into(), "asserted".into()]));
    assert_eq!(out.iter().filter(|r| r[2] == "written").count(), 4);
}

#[test]
fn register_is_idempotent_and_refuses_drift() {
    let (db, fp) = db();
    assert_eq!(db.engram_register("m", &spec(0), None).unwrap(), fp);
    let err = db.engram_register("m", &spec(1), None).unwrap_err();
    assert!(format!("{err:?}").contains("fingerprint mismatch"), "{err:?}");
    let ms = db.engram_models().unwrap();
    assert_eq!(ms.len(), 1);
    assert_eq!(ms[0].layers, vec![1, 5]);
    assert_eq!(ms[0].row_dim, DIM);
}

#[test]
fn write_and_materialize_roundtrip_bf16() {
    let (db, fp) = db();
    two_facts(&db, &fp, 1_000);
    let m = mat(&db, &["u1"], None);
    assert_eq!(m.len(), 4);
    assert_eq!(m[&(1, 11)], bf(2.0));
    let ov = db.engram_materialize("m", &["u1".into()], None, None).unwrap();
    assert_eq!(ov.rows.iter().find(|r| r.row == 11).unwrap().facts, vec!["a", "b"]);
    // before the write: nothing
    assert!(mat(&db, &["u1"], Some(999)).is_empty());
    // another namespace: nothing
    assert!(mat(&db, &["u2"], None).is_empty());
    let facts = db.engram_facts("m", "u1", None).unwrap();
    assert_eq!(facts.len(), 2);
    assert_eq!(facts[0].1.payload, Some(vec![0.5, -1.0]));
    assert_eq!(facts[0].2, 1_000);
}

#[test]
fn validation_refuses_bad_rows() {
    let (db, fp) = db();
    let base = |rows: Vec<RowEdit>| EngramWrite { ns: "u".into(), fingerprint: fp.clone(), at: Some(5), facts: vec![fact("a", "x", &[])], rows, retract: vec![] };
    for (rows, needle) in [
        (vec![row(2, 1, 0.0, &["a"])], "not an Engram layer"),
        (vec![RowEdit { vec: vec![1.0; DIM + 1], ..row(1, 1, 0.0, &["a"]) }], "values"),
        (vec![row(1, 1, 0.0, &[])], "provenance"),
        (vec![row(1, 1, 0.0, &["zzz"])], "neither in this write"),
        (vec![row(1, 1_000_000, 0.0, &["a"])], "outside"),
        (vec![row(1, 1, 0.0, &["a"]), row(1, 1, 1.0, &["a"])], "twice"),
        (vec![RowEdit { vec: vec![f32::NAN; DIM], ..row(1, 1, 0.0, &["a"]) }], "non-finite"),
    ] {
        let err = db.engram_write("m", &base(rows)).unwrap_err();
        assert!(format!("{err:?}").contains(needle), "{needle}: {err:?}");
    }
    assert!(db.engram_write("nope", &base(vec![])).is_err());
}

#[test]
fn fingerprint_mismatch_is_refused_everywhere() {
    let (db, fp) = db();
    two_facts(&db, &fp, 1_000);
    let other = spec(7).fingerprint().to_string();
    assert_ne!(other, fp);
    let w = EngramWrite { ns: "u1".into(), fingerprint: other.clone(), at: None, facts: vec![], rows: vec![], retract: vec![] };
    let e = db.engram_write("m", &w).unwrap_err();
    assert!(format!("{e:?}").contains("fingerprint mismatch"), "{e:?}");
    assert!(db.engram_materialize("m", &["u1".into()], None, Some(&other)).is_err());
    let dir = tempfile::tempdir().unwrap();
    assert!(db.engram_export("m", &["u1".into()], None, dir.path(), Some(&other)).is_err());
    assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none(), "nothing written on refusal");
    assert!(db.engram_materialize("m", &["u1".into()], None, Some(&fp)).is_ok());
}

#[test]
fn supersede_flips_and_as_of_restores() {
    let (db, fp) = db();
    two_facts(&db, &fp, 1_000);
    // a -> a2 at t=2000, rewriting a's rows except the shared (1,11)
    let w = EngramWrite {
        ns: "u1".into(),
        fingerprint: fp.clone(),
        at: Some(2_000),
        facts: vec![fact("a2", "milk", &[(1, 10), (5, 20)])],
        rows: vec![row(1, 10, 10.0, &["a2"]), row(5, 20, 30.0, &["a2"])],
        retract: vec![],
    };
    let out = db.engram_supersede("m", "a", &w).unwrap().rows;
    let acts: Vec<(String, String)> = out.iter().map(|r| (r[1].get_str().unwrap().into(), r[2].get_str().unwrap().into())).collect();
    assert!(acts.contains(&("a".into(), "retracted".into())));
    // (1,11) depended on a and b: retracted, b still live -> needs replay
    assert!(acts.contains(&("1:11".into(), "needs_replay".into())), "{acts:?}");
    let now = mat(&db, &["u1"], None);
    assert_eq!(now[&(1, 10)], bf(10.0));
    assert_eq!(now[&(5, 20)], bf(30.0));
    assert!(!now.contains_key(&(1, 11)), "stale shared row reverted to base");
    assert_eq!(now[&(5, 21)], bf(4.0), "b's own row untouched");
    let then = mat(&db, &["u1"], Some(1_500));
    assert_eq!(then[&(1, 10)], bf(1.0), "as_of before the supersession shows the old value");
    assert_eq!(then[&(1, 11)], bf(2.0));
    let ids: Vec<String> = db.engram_facts("m", "u1", None).unwrap().into_iter().map(|f| f.0).collect();
    assert_eq!(ids, ["a2", "b"]);
    let ids_then: Vec<String> = db.engram_facts("m", "u1", Some(1_500)).unwrap().into_iter().map(|f| f.0).collect();
    assert_eq!(ids_then, ["a", "b"]);
    let hist = db.engram_history("m", "u1", Some("a")).unwrap().rows;
    assert_eq!(hist.len(), 2);
    assert_eq!((hist[0][1].get_int(), hist[0][2].get_bool()), (Some(1_000), Some(true)));
    assert_eq!((hist[1][1].get_int(), hist[1][2].get_bool()), (Some(2_000), Some(false)));
}

#[test]
fn erase_fact_by_provenance_replay() {
    let (db, fp) = db();
    two_facts(&db, &fp, 1_000);
    // a third, independent fact
    write(&db, &fp, "u1", 1_100, vec![fact("c", "red", &[(5, 30)])], vec![row(5, 30, 7.0, &["c"])], &[]);
    let before = mat(&db, &["u1"], None);
    let plan = db.engram_erase_plan("m", "u1", "a").unwrap();
    assert_eq!(plan, vec![((1, 11), vec!["b".to_string()])], "only the shared row needs replay");
    // a replacement outside the touched set is refused
    let bad = db.engram_erase_fact("m", "u1", "a", &[row(5, 21, 9.0, &["b"])], Some(3_000)).unwrap_err();
    assert!(format!("{bad:?}").contains("not touched"), "{bad:?}");
    // a replacement that still depends on the erased fact is refused
    assert!(db.engram_erase_fact("m", "u1", "a", &[row(1, 11, 9.0, &["a", "b"])], Some(3_000)).is_err());
    let res = db.engram_erase_fact("m", "u1", "a", &[row(1, 11, 5.0, &["b"])], Some(3_000)).unwrap();
    assert_eq!(res, vec![((1, 10), "reverted"), ((1, 11), "replayed"), ((5, 20), "reverted")]);
    let after = mat(&db, &["u1"], None);
    assert_eq!(after[&(1, 11)], bf(5.0));
    assert!(!after.contains_key(&(1, 10)) && !after.contains_key(&(5, 20)));
    for k in [(5, 21), (5, 30)] {
        assert_eq!(after[&k], before[&k], "untouched row {k:?} is bit-identical");
    }
    // gone from history, every version, and from the past too
    assert!(db.engram_history("m", "u1", Some("a")).unwrap().rows.is_empty());
    assert!(!mat(&db, &["u1"], Some(1_500)).contains_key(&(1, 10)));
    assert!(db.engram_erase_fact("m", "u1", "a", &[], None).is_err(), "already gone");
}

#[test]
fn erase_namespace_leaves_nothing() {
    let (db, fp) = db();
    two_facts(&db, &fp, 1_000);
    write(&db, &fp, "u2", 1_000, vec![fact("z", "q", &[(1, 50)])], vec![row(1, 50, 1.0, &["z"])], &[]);
    let (f, r) = db.engram_erase_namespace("u1", Some("m")).unwrap();
    assert_eq!((f, r), (2, 4));
    assert!(mat(&db, &["u1"], None).is_empty());
    assert!(mat(&db, &["u1"], Some(1_500)).is_empty());
    assert!(db.engram_history("m", "u1", None).unwrap().rows.is_empty());
    assert_eq!(mat(&db, &["u2"], None).len(), 1, "other namespaces untouched");
}

#[test]
fn layering_org_user_agent() {
    let (db, fp) = db();
    write(&db, &fp, "org", 10, vec![fact("o", "x", &[(1, 1), (1, 2)])], vec![row(1, 1, 1.0, &["o"]), row(1, 2, 2.0, &["o"])], &[]);
    write(&db, &fp, "user", 10, vec![fact("u", "y", &[(1, 2), (1, 3)])], vec![row(1, 2, 20.0, &["u"]), row(1, 3, 30.0, &["u"])], &[]);
    write(&db, &fp, "agent", 10, vec![fact("g", "z", &[(1, 3)])], vec![row(1, 3, 300.0, &["g"])], &[]);
    let m = mat(&db, &["org", "user", "agent"], None);
    assert_eq!(m[&(1, 1)], bf(1.0));
    assert_eq!(m[&(1, 2)], bf(20.0), "user overrides org");
    assert_eq!(m[&(1, 3)], bf(300.0), "agent overrides user");
    let ov = db.engram_materialize("m", &["org".into(), "user".into()], None, None).unwrap();
    assert_eq!(ov.rows.iter().map(|r| r.ns.as_str()).collect::<Vec<_>>(), ["org", "user", "user"]);
}

#[test]
fn export_safetensors_per_layer() {
    let (db, fp) = db();
    two_facts(&db, &fp, 1_000);
    let dir = tempfile::tempdir().unwrap();
    let files = db.engram_export("m", &["u1".into()], None, dir.path(), Some(&fp)).unwrap();
    assert_eq!(files.iter().map(|f| (f.0, f.2)).collect::<Vec<_>>(), [(1, 2), (5, 2)]);
    let (l1, meta) = safetensors::read_file(&files[0].1).unwrap();
    assert_eq!(meta["fingerprint"], fp);
    assert_eq!(meta["mode"], "replace");
    assert_eq!(l1.rows, vec![10, 11]);
    assert_eq!(l1.row_dim, DIM);
    assert_eq!(bf16::to_f32(&l1.values[DIM..]), bf(2.0));
    // erased user exports empty layers
    db.engram_erase_namespace("u1", None).unwrap();
    let files = db.engram_export("m", &["u1".into()], None, dir.path(), None).unwrap();
    assert!(files.iter().all(|f| f.2 == 0));
}

#[test]
fn cozoscript_surface() {
    let (db, fp) = db();
    let run = |s: &str, p: BTreeMap<String, DataValue>| db.run_script(s, p, ScriptMutability::Mutable);
    let rows = run("::engram models", Default::default()).unwrap();
    assert_eq!(rows.rows[0][1].get_str(), Some(fp.as_str()));
    let p: BTreeMap<String, DataValue> = [
        ("fp".to_string(), DataValue::from(fp.as_str())),
        (
            "facts".to_string(),
            DataValue::from(json!([{"id": "a", "text": "a is oat", "target": " oat", "phrasings": ["a is"], "rows": [[1, 7]], "source": "ep9"}])),
        ),
        ("rows".to_string(), DataValue::from(json!([{"layer": 1, "row": 7, "vec": [1.0, 2.0, 3.0, 4.0], "facts": ["a"]}]))),
    ]
    .into_iter()
    .collect();
    let out = run("::engram write m {ns: 'u1', fingerprint: $fp, at: 100, facts: $facts, rows: $rows}", p.clone()).unwrap();
    assert_eq!(out.headers, ["kind", "id", "action"]);
    let m = run("::engram materialize m {ns: ['org', 'u1']}", Default::default()).unwrap();
    assert_eq!(m.rows.len(), 1);
    assert_eq!(m.rows[0][4], DataValue::List(vec![1.0, 2.0, 3.0, 4.0].into_iter().map(DataValue::from).collect()));
    let f = run("::engram facts m {ns: 'u1'}", Default::default()).unwrap();
    assert_eq!(f.rows[0][0].get_str(), Some("a"));
    assert_eq!(f.rows[0][7].get_str(), Some("ep9"));
    assert_eq!(run("::engram history m {ns: 'u1'}", Default::default()).unwrap().rows.len(), 1);
    let e = run("::engram materialize m {ns: 'u1', fingerprint: 'eg1:00000000000000000000000000000000'}", Default::default()).unwrap_err();
    assert!(format!("{e:?}").contains("fingerprint mismatch"));
    assert!(run("::engram materialize m {ns: 'u1', bogus: 1}", Default::default()).is_err());
    assert!(run("::engram frobnicate m", Default::default()).is_err());
    let e = run("::engram erase {ns: 'u1'}", Default::default()).unwrap();
    assert_eq!(e.rows[0][2].get_int(), Some(1));
    assert!(run("::engram materialize m {ns: 'u1'}", Default::default()).unwrap().rows.is_empty());
    // read-only mode refuses writes
    assert!(db.run_script("::engram erase {ns: 'u1'}", Default::default(), ScriptMutability::Immutable).is_err());
}

/// Memory branches (docs/specs/memory-branches.md, "Engram"): a branch
/// serves its parent's overlay as of the fork, its own writes mask the
/// inherited rows they change, and copy-up keeps the inherited history.
#[test]
fn branches_read_through_and_copy_up() {
    let (db, fp) = db();
    db.memory_init(kleos::memory::InitOpts::default()).unwrap();
    two_facts(&db, &fp, 1_000);
    db.memory_fork(kleos::memory::ForkOpts { from: Some("u1".into()), to: "br".into(), at: Some(2_000) })
        .unwrap();
    // the parent changes after the fork: not in the branch
    write(&db, &fp, "u1", 3_000, vec![fact("c", "rye", &[(5, 30)])], vec![row(5, 30, 9.0, &["c"])], &[]);
    let base = mat(&db, &["u1"], Some(2_000));
    assert_eq!(base.len(), 4);
    assert_eq!(mat(&db, &["br"], None), base);
    // the branch supersedes a: a2 rewrites (1,10); (5,20) reverts; (1,11)
    // (shared with b) is retracted for replay
    let out = write(&db, &fp, "br", 4_000, vec![fact("a2", "milk", &[(1, 10)])], vec![row(1, 10, 7.0, &["a2"])], &["a"]);
    assert!(out.contains(&vec!["fact".into(), "a".into(), "retracted".into()]), "{out:?}");
    assert!(out.contains(&vec!["row".into(), "1:11".into(), "needs_replay".into()]), "{out:?}");
    let now = mat(&db, &["br"], None);
    assert_eq!(now.get(&(1, 10)), Some(&bf(7.0)));
    assert!(!now.contains_key(&(1, 11)) && !now.contains_key(&(5, 20)), "{now:?}");
    assert_eq!(now.get(&(5, 21)), Some(&bf(4.0)));
    assert!(!now.contains_key(&(5, 30)));
    // the parent is untouched
    let parent = mat(&db, &["u1"], None);
    assert_eq!(parent.get(&(5, 20)), Some(&bf(3.0)));
    assert_eq!(parent.get(&(5, 30)), Some(&bf(9.0)));
    // before its write the branch reads the inherited history (copy-up)
    assert_eq!(mat(&db, &["br"], Some(3_500)), base);
    let ids: Vec<String> = db.engram_facts("m", "br", None).unwrap().into_iter().map(|x| x.0).collect();
    assert_eq!(ids, vec!["a2".to_string(), "b".to_string()]);
}
