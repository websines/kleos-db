/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Memory branches (docs/specs/memory-branches.md): fork, rewind, merge,
//! diff and discard, against a fake decider and a fake embedder registered
//! through the process-wide registries (as in `tests/memory.rs`).

use std::collections::BTreeSet;
use std::sync::{Arc, Barrier, RwLock};

use kleos::decide::{self};
use kleos::embed::{self, Embedder};
use kleos::memory::{
    BranchesOpts, DiffOpts, DiscardOpts, ForgetOpts, ForkOpts, InitOpts, MergeOpts, RecallOpts, RememberOpts,
    RewindOpts, SetOpts,
};
use kleos::{DbInstance, DecideAnswer, DecideRequest, Decider, Family, ScriptMutability};

static REGISTRY: RwLock<()> = RwLock::new(());

const DAY: i64 = 86_400_000_000;
const T0: i64 = 1_767_225_600_000_000; // 2026-01-01T00:00:00Z

fn d(n: i64) -> i64 {
    T0 + n * DAY
}

/// Bag-of-words hashed into 64 dims, L2-normalised.
struct Words;

impl Embedder for Words {
    fn embed(&self, texts: &[&str]) -> miette::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0f32; 64];
                for w in t.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2) {
                    let h = w
                        .to_lowercase()
                        .bytes()
                        .fold(1469598103934665603u64, |h, b| (h ^ b as u64).wrapping_mul(1099511628211));
                    v[(h % 64) as usize] += 1.0;
                }
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                v.iter().map(|x| x / n).collect()
            })
            .collect())
    }
    fn dim(&self) -> usize {
        64
    }
}

/// An NLI decider answering from a table: `contra` pairs (old, new)
/// contradict, everything else is neutral.
struct Nli {
    id: String,
    contra: Vec<(&'static str, &'static str)>,
}

impl Decider for Nli {
    fn model_id(&self) -> String {
        self.id.clone()
    }
    fn weights_version(&self) -> Option<String> {
        Some("nano-tune-v1".to_string())
    }
    fn decide(&self, reqs: &[DecideRequest], _f: bool) -> miette::Result<Vec<DecideAnswer>> {
        Ok(reqs
            .iter()
            .map(|r| {
                let st = r.state.strip_prefix("Premise: ").unwrap_or(&r.state);
                let (a, b) = st.split_once("\nHypothesis: ").unwrap_or((st, ""));
                let logits = if self.contra.iter().any(|&(x, y)| x == a && y == b) {
                    vec![0.0, 0.0, 12.0]
                } else {
                    vec![0.0, 6.0, 0.0]
                };
                DecideAnswer { logits, features: vec![] }
            })
            .collect())
    }
}

struct Models {
    embedder: String,
    decider: String,
    _guard: std::sync::RwLockReadGuard<'static, ()>,
}

impl Models {
    fn new(tag: &str, contra: Vec<(&'static str, &'static str)>) -> Self {
        let guard = REGISTRY.read().unwrap_or_else(|e| e.into_inner());
        let embedder = format!("br_emb_{tag}");
        let decider = format!("br_dec_{tag}");
        embed::register_embedder(&embedder, Arc::new(Words)).unwrap();
        decide::register_decider(&decider, Arc::new(Nli { id: format!("br-nli-{tag}"), contra }), Family::Nano)
            .unwrap();
        Models { embedder, decider, _guard: guard }
    }
    fn db(&self) -> DbInstance {
        self.db_at("mem", "")
    }
    fn db_at(&self, engine: &str, path: &str) -> DbInstance {
        let db = DbInstance::new(engine, path, "").unwrap();
        db.memory_init(InitOpts {
            embedder: Some(self.embedder.clone()),
            decider: Some(self.decider.clone()),
            ..Default::default()
        })
        .unwrap();
        db
    }
}

impl Drop for Models {
    fn drop(&mut self) {
        embed::unregister_embedder(&self.embedder);
        decide::unregister_decider(&self.decider);
    }
}

fn ns(n: &str) -> Option<String> {
    Some(n.to_string())
}

fn at(ns_: &str, t: i64) -> RememberOpts {
    RememberOpts {
        ns: ns(ns_),
        at: Some(t),
        ..Default::default()
    }
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64
}

/// Texts of the current memories of a namespace's view, by a broad recall.
fn current(db: &DbInstance, ns_: &str, as_of: Option<i64>) -> BTreeSet<String> {
    db.recall(
        "live moved coffee work Anna Paris Berlin Lisbon Rome tea dog",
        RecallOpts {
            ns: ns(ns_),
            k: Some(50),
            as_of,
            diversity: Some(false),
            record_access: Some(false),
            ..Default::default()
        },
    )
    .unwrap()
    .into_iter()
    .map(|h| h.memory_text().to_string())
    .collect()
}

fn count(db: &DbInstance, script: &str) -> usize {
    db.run_script(script, Default::default(), ScriptMutability::Immutable)
        .unwrap()
        .rows
        .len()
}

fn set(xs: &[&str]) -> BTreeSet<String> {
    xs.iter().map(|x| x.to_string()).collect()
}

fn fork(db: &DbInstance, from: &str, to: &str, at: Option<i64>) -> kleos::memory::ForkResult {
    db.memory_fork(ForkOpts { from: ns(from), to: to.to_string(), at }).unwrap()
}

const PARIS: &str = "I live in Paris.";
const BERLIN: &str = "I moved to Berlin.";
const LISBON: &str = "I moved to Lisbon.";
const COFFEE: &str = "I like my coffee black.";
const TEA: &str = "I switched to green tea.";
const DOG: &str = "I have a dog named Rex.";
const WORK: &str = "I work at Siemens.";

#[test]
fn fork_isolates_both_ways_and_copies_nothing() {
    let m = Models::new("iso", vec![]);
    let db = m.db();
    db.remember(PARIS, at("main", d(1))).unwrap();
    db.remember(COFFEE, at("main", d(2))).unwrap();
    let rows_before = count(&db, "?[ns, id, vt] := *kleos_mem{ns, id, vt}");
    let f = fork(&db, "main", "br", None);
    assert_eq!((f.ns.as_str(), f.parent.as_str(), f.depth), ("br", "main", 1));
    assert_eq!(f.lineage.len(), 2);
    // O(1): no memory row was copied
    assert_eq!(count(&db, "?[ns, id, vt] := *kleos_mem{ns, id, vt}"), rows_before);
    assert_eq!(count(&db, "?[id] := *kleos_mem{ns: 'br', id}"), 0);
    // the branch reads its parent
    assert_eq!(current(&db, "br", None), set(&[PARIS, COFFEE]));
    // writes stay on their side
    db.remember(DOG, at("br", d(3))).unwrap();
    db.remember(WORK, RememberOpts { ns: ns("main"), ..Default::default() }).unwrap();
    assert_eq!(current(&db, "br", None), set(&[PARIS, COFFEE, DOG]));
    assert_eq!(current(&db, "main", None), set(&[PARIS, COFFEE, WORK]));
    // `to` must be unused
    assert!(db.memory_fork(ForkOpts { from: ns("main"), to: "br".into(), at: None }).is_err());
    assert!(db.memory_fork(ForkOpts { from: ns("br"), to: "main".into(), at: None }).is_err());
    assert!(db.memory_fork(ForkOpts { from: ns("main"), to: "x".into(), at: Some(now() + DAY) }).is_err());
}

#[test]
fn reads_fall_through_as_of_the_cut() {
    let m = Models::new("cut", vec![(PARIS, BERLIN)]);
    let db = m.db();
    db.remember(PARIS, at("main", d(1))).unwrap();
    let b = db.remember(BERLIN, at("main", d(10))).unwrap();
    assert_eq!(b.superseded.len(), 1);
    // a fork of the past: day 5, when Paris was current
    fork(&db, "main", "past", Some(d(5)));
    assert_eq!(current(&db, "past", None), set(&[PARIS]));
    assert_eq!(current(&db, "past", Some(d(3))), set(&[PARIS]));
    assert!(current(&db, "past", Some(d(0))).is_empty());
    // the parent's later supersession is not visible, and the hit's
    // validity is the branch's (Paris never ended there)
    let h = db
        .recall("where do I live?", RecallOpts { ns: ns("past"), ..Default::default() })
        .unwrap()
        .into_iter()
        .find(|h| h.memory_text() == PARIS)
        .unwrap();
    assert_eq!((h.valid_from, h.valid_to), (d(1), None));
    // a memory the parent adds after the fork stays out
    db.remember(DOG, RememberOpts { ns: ns("main"), ..Default::default() }).unwrap();
    assert!(!current(&db, "past", None).contains(DOG));
}

#[test]
fn branch_forget_is_a_tombstone_with_history() {
    let m = Models::new("tomb", vec![]);
    let db = m.db();
    let p = db.remember(PARIS, at("main", d(1))).unwrap().id.unwrap();
    db.remember(COFFEE, at("main", d(2))).unwrap();
    fork(&db, "main", "br", Some(d(5)));
    let r = db
        .forget(ForgetOpts { ns: ns("br"), id: Some(p.clone()), at: Some(d(6)), ..Default::default() })
        .unwrap();
    assert_eq!(r.ids, vec![p.clone()]);
    assert_eq!(current(&db, "br", None), set(&[COFFEE]));
    // the parent keeps it; the branch's copy keeps the inherited history
    assert_eq!(current(&db, "main", None), set(&[PARIS, COFFEE]));
    assert_eq!(current(&db, "br", Some(d(3))), set(&[PARIS, COFFEE]));
    assert_eq!(count(&db, &format!("?[id] := *kleos_mem_copied{{ns: 'br', id, from: 'main'}}, id = '{p}'")), 1);
    // set on an inherited memory copies it up too
    let c = db
        .recall("coffee", RecallOpts { ns: ns("br"), k: Some(1), ..Default::default() })
        .unwrap()
        .remove(0)
        .id;
    db.memory_set(SetOpts { ns: ns("br"), id: Some(c.clone()), importance: Some(0.9), ..Default::default() })
        .unwrap();
    let imp = |n: &str| {
        db.recall("coffee", RecallOpts { ns: ns(n), k: Some(1), ..Default::default() }).unwrap()[0].importance
    };
    assert_eq!(imp("br"), Some(0.9));
    assert_eq!(imp("main"), None);
    // the copy's vector is current (it still competes in the vector leg);
    // the forgotten one's is not
    assert_eq!(count(&db, &format!("?[c] := *kleos_mem_vec{{ns: 'br', id: '{c}', current: c}}, c")), 1);
    assert_eq!(count(&db, &format!("?[c] := *kleos_mem_vec{{ns: 'br', id: '{p}', current: c}}, !c")), 1);
}

#[test]
fn nested_forks() {
    let m = Models::new("nest", vec![]);
    let db = m.db();
    db.remember(PARIS, at("main", d(1))).unwrap();
    fork(&db, "main", "b1", Some(d(2)));
    db.remember(COFFEE, at("b1", d(3))).unwrap();
    let f = fork(&db, "b1", "b2", Some(d(4)));
    assert_eq!(f.depth, 2);
    assert_eq!(f.lineage[2], ("main".to_string(), Some(d(2))));
    db.remember(DOG, at("b2", d(5))).unwrap();
    db.remember(WORK, at("b1", d(6))).unwrap();
    assert_eq!(current(&db, "b2", None), set(&[PARIS, COFFEE, DOG]));
    assert_eq!(current(&db, "b1", None), set(&[PARIS, COFFEE, WORK]));
    assert_eq!(current(&db, "main", None), set(&[PARIS]));
    // discard needs cascade while b1 has a branch
    assert!(db.memory_discard(DiscardOpts { ns: ns("b1"), cascade: false }).is_err());
    let r = db.memory_discard(DiscardOpts { ns: ns("b1"), cascade: true }).unwrap();
    assert_eq!(r.discarded, vec!["b2".to_string(), "b1".to_string()]);
    assert_eq!(current(&db, "main", None), set(&[PARIS]));
    assert!(db.remember(TEA, at("b1", d(7))).is_err());
    assert!(db.memory_branches(BranchesOpts::default()).unwrap().is_empty());
}

#[test]
fn rewind_and_undo() {
    let m = Models::new("rew", vec![(PARIS, BERLIN)]);
    let db = m.db();
    db.remember(PARIS, at("main", d(1))).unwrap();
    db.remember(COFFEE, at("main", d(2))).unwrap();
    db.remember(BERLIN, at("main", d(3))).unwrap();
    assert_eq!(current(&db, "main", None), set(&[COFFEE, BERLIN]));
    let r = db.memory_rewind(RewindOpts { ns: ns("main"), to: d(2) + 1, at: None }).unwrap();
    assert_eq!((r.retracted.len(), r.restored.len()), (1, 1));
    assert_eq!(current(&db, "main", None), set(&[PARIS, COFFEE]));
    // the restored memory competes in the vector leg again
    assert_eq!(
        count(&db, "?[id] := *kleos_mem_vec{ns: 'main', id, current: true}"),
        2,
        "restored vectors must be current"
    );
    // history is intact: the state before the rewind is still readable
    assert_eq!(current(&db, "main", Some(r.undo)), set(&[COFFEE, BERLIN]));
    let u = db.memory_rewind(RewindOpts { ns: ns("main"), to: r.undo, at: None }).unwrap();
    assert_eq!((u.retracted.len(), u.restored.len()), (1, 1));
    assert_eq!(current(&db, "main", None), set(&[COFFEE, BERLIN]));
    // a branch rewinds itself only
    fork(&db, "main", "br", None);
    db.remember(DOG, RememberOpts { ns: ns("br"), ..Default::default() }).unwrap();
    let r = db.memory_rewind(RewindOpts { ns: ns("br"), to: d(2) + 1, at: None }).unwrap();
    assert_eq!(r.restored.len(), 1, "{r:?}");
    assert_eq!(current(&db, "br", None), set(&[PARIS, COFFEE]));
    assert_eq!(current(&db, "main", None), set(&[COFFEE, BERLIN]));
}

#[test]
fn merge_resolves_a_conflict_through_remember() {
    let m = Models::new("merge", vec![(PARIS, LISBON), (PARIS, BERLIN)]);
    let db = m.db();
    let paris = db.remember(PARIS, at("main", d(1))).unwrap().id.unwrap();
    db.remember(COFFEE, at("main", d(2))).unwrap();
    fork(&db, "main", "lis", None);
    fork(&db, "main", "ber", None);
    // each branch learns something conflicting; the decider supersedes the
    // inherited Paris in the branch only
    let l = db.remember(LISBON, RememberOpts { ns: ns("lis"), ..Default::default() }).unwrap();
    assert_eq!(l.superseded, vec![paris.clone()]);
    db.remember(BERLIN, RememberOpts { ns: ns("ber"), ..Default::default() }).unwrap();
    db.remember(DOG, RememberOpts { ns: ns("lis"), ..Default::default() }).unwrap();
    assert_eq!(current(&db, "lis", None), set(&[LISBON, COFFEE, DOG]));
    assert_eq!(current(&db, "ber", None), set(&[BERLIN, COFFEE]));
    assert_eq!(current(&db, "main", None), set(&[PARIS, COFFEE]));

    // diff: what lis would bring
    let df = db.memory_diff(DiffOpts { a: "main".into(), b: "lis".into(), at: None }).unwrap();
    let added: BTreeSet<String> = df.added.iter().map(|x| x.text.clone()).collect();
    assert_eq!(added, set(&[LISBON, DOG]));
    assert_eq!(df.removed.len(), 1);
    assert_eq!(df.removed[0].text, PARIS);
    assert_eq!(df.removed[0].reason.as_deref(), Some("superseded"));
    assert_eq!(df.same, 1);

    // dry run: the exact plan, nothing changes
    let dry = db.memory_merge(MergeOpts { from: "lis".into(), dry_run: true, ..Default::default() }).unwrap();
    assert!(dry.dry_run);
    assert_eq!(dry.into, "main");
    assert_eq!(dry.statements, 2);
    let dd = dry.diff.unwrap();
    assert_eq!(dd.added.iter().map(|x| x.text.clone()).collect::<BTreeSet<_>>(), set(&[LISBON, DOG]));
    assert_eq!(dd.removed.len(), 1);
    assert_eq!(current(&db, "main", None), set(&[PARIS, COFFEE]));
    let names: Vec<String> = db.memory_branches(BranchesOpts::default()).unwrap().into_iter().map(|b| b.ns).collect();
    assert_eq!(names, vec!["lis".to_string(), "ber".to_string()], "the scratch branch is gone");

    // the real merge: the decider supersedes Paris in main
    let r = db.memory_merge(MergeOpts { from: "lis".into(), ..Default::default() }).unwrap();
    assert_eq!(r.statements, 2);
    let lis_stmt = r.replayed.iter().find(|x| x.text == LISBON).unwrap();
    assert_eq!(lis_stmt.superseded, vec![paris.clone()]);
    assert_eq!(current(&db, "main", None), set(&[LISBON, COFFEE, DOG]));
    // idempotent
    let again = db.memory_merge(MergeOpts { from: "lis".into(), ..Default::default() }).unwrap();
    assert_eq!((again.statements, again.skipped), (0, 2));
    // the other branch is dropped; main keeps the merge
    db.memory_discard(DiscardOpts { ns: ns("ber"), cascade: false }).unwrap();
    assert_eq!(current(&db, "main", None), set(&[LISBON, COFFEE, DOG]));
    let after = db.memory_diff(DiffOpts { a: "main".into(), b: "lis".into(), at: None }).unwrap();
    assert!(after.added.is_empty() && after.removed.is_empty(), "{after:?}");
}

#[test]
fn merge_replays_branch_forgets() {
    let m = Models::new("mforget", vec![]);
    let db = m.db();
    let c = db.remember(COFFEE, at("main", d(1))).unwrap().id.unwrap();
    db.remember(PARIS, at("main", d(2))).unwrap();
    fork(&db, "main", "br", None);
    db.forget(ForgetOpts { ns: ns("br"), id: Some(c.clone()), ..Default::default() }).unwrap();
    let t = db.remember(TEA, RememberOpts { ns: ns("br"), ..Default::default() }).unwrap().id.unwrap();
    // a memory the branch adds and forgets is not merged
    db.remember(DOG, RememberOpts { ns: ns("br"), ..Default::default() }).unwrap();
    let dog = db
        .recall("dog", RecallOpts { ns: ns("br"), k: Some(1), ..Default::default() })
        .unwrap()
        .remove(0)
        .id;
    db.forget(ForgetOpts { ns: ns("br"), id: Some(dog), ..Default::default() }).unwrap();
    let r = db.memory_merge(MergeOpts { from: "br".into(), ..Default::default() }).unwrap();
    assert_eq!(r.forgets, vec![c]);
    assert_eq!(r.replayed.iter().filter(|x| x.action == "skipped").count(), 1);
    assert_eq!(current(&db, "main", None), set(&[PARIS, TEA]));
    assert!(!t.is_empty());
}

#[test]
fn recall_on_a_branch_ranks_parent_and_own_memories_together() {
    let m = Models::new("rank", vec![]);
    let db = m.db();
    db.remember(PARIS, at("main", d(1))).unwrap();
    db.remember(COFFEE, at("main", d(2))).unwrap();
    fork(&db, "main", "br", None);
    db.remember(TEA, RememberOpts { ns: ns("br"), ..Default::default() }).unwrap();
    let hits = db
        .recall("coffee or tea?", RecallOpts { ns: ns("br"), k: Some(2), ..Default::default() })
        .unwrap();
    let texts: BTreeSet<String> = hits.iter().map(|h| h.memory_text().to_string()).collect();
    assert_eq!(texts, set(&[COFFEE, TEA]));
    // the context output reads the lineage too
    let ctx = db
        .recall_context("coffee or tea?", RecallOpts { ns: ns("br"), k: Some(3), ..Default::default() })
        .unwrap();
    assert!(ctx.text.contains(COFFEE) && ctx.text.contains(TEA), "{}", ctx.text);
    // FTS only (no vector): the text leg filters on the lineage
    let fts = db
        .recall("coffee", RecallOpts { ns: ns("br"), k: Some(3), vector_weight: Some(0.0), ..Default::default() })
        .unwrap();
    assert!(fts.iter().any(|h| h.memory_text() == COFFEE));
}

#[test]
fn entities_resolve_against_the_lineage() {
    const ANNA_PARIS: &str = "Anna lives in Paris.";
    const ANNA_ROME: &str = "Anna moved to Rome.";
    let m = Models::new("ent", vec![(ANNA_PARIS, ANNA_ROME)]);
    let db = m.db();
    let ents = |n: &str| RememberOpts { ns: ns(n), entities: vec!["Anna".into()], ..Default::default() };
    let a = db.remember(ANNA_PARIS, ents("main")).unwrap();
    fork(&db, "main", "br", None);
    let r = db.remember(ANNA_ROME, ents("br")).unwrap();
    // the branch's Anna is the parent's Anna: no new entity, not gated
    assert_eq!(r.entity_links[0].outcome, "match", "{:?}", r.entity_links);
    assert_eq!(r.entity_links[0].eid, a.entity_links[0].eid);
    assert!(r.gated.is_empty());
    assert_eq!(r.superseded.len(), 1);
    assert_eq!(count(&db, "?[eid] := *kleos_mem_entity{ns: 'br', eid}"), 0);
    // once the branch has entities of its own, resolution reads both
    let rome = db.remember("Rome is warm in May.", RememberOpts { ns: ns("br"), entities: vec!["Rome".into()], ..Default::default() }).unwrap();
    assert_eq!(rome.entity_links[0].outcome, "new");
    let again = db.remember("I visited Rome.", RememberOpts { ns: ns("br"), entities: vec!["Rome".into(), "Anna".into()], ..Default::default() }).unwrap();
    assert_eq!(again.entity_links.iter().map(|e| e.outcome.as_str()).collect::<Vec<_>>(), vec!["match", "match"]);
    // the entity filter of recall works on the branch
    let hits = db
        .recall("where?", RecallOpts { ns: ns("br"), entity: Some("Anna".into()), ..Default::default() })
        .unwrap();
    let anna: Vec<&str> = hits.iter().map(|h| h.memory_text()).collect();
    assert!(anna.contains(&ANNA_ROME) && !anna.contains(&ANNA_PARIS), "{anna:?}");
}

#[test]
fn erase_reaches_every_copy_in_the_family() {
    let m = Models::new("erase", vec![]);
    let db = m.db();
    let c = db.remember(COFFEE, at("main", d(1))).unwrap().id.unwrap();
    fork(&db, "main", "br", None);
    // the branch copies it up (archive)
    db.forget(ForgetOpts { ns: ns("br"), id: Some(c.clone()), mode: Some("archive".into()), ..Default::default() })
        .unwrap();
    assert_eq!(count(&db, &format!("?[ns] := *kleos_mem{{ns, id: '{c}'}}")), 2);
    let r = db
        .forget(ForgetOpts { ns: ns("main"), id: Some(c.clone()), mode: Some("erase".into()), ..Default::default() })
        .unwrap();
    assert_eq!(r.ids, vec![c.clone()]);
    assert_eq!(count(&db, &format!("?[ns, vt] := *kleos_mem{{ns, id: '{c}', vt}}")), 0);
    assert_eq!(count(&db, &format!("?[ns] := *kleos_mem_vec{{ns, id: '{c}'}}")), 0);
}

#[test]
fn concurrent_remembers_on_parent_and_branch() {
    let m = Models::new("conc", vec![]);
    let dir = std::env::temp_dir().join(format!("kleos-branch-conc-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("db.sqlite");
    let db = m.db_at("sqlite", path.to_str().unwrap());
    db.remember(PARIS, at("main", d(1))).unwrap();
    fork(&db, "main", "br", None);
    let texts = |pre: &str| (0..6).map(|i| format!("{pre} fact number {i} about cats.")).collect::<Vec<_>>();
    let barrier = Arc::new(Barrier::new(2));
    let hs: Vec<_> = ["main", "br"]
        .into_iter()
        .map(|n| {
            let db = db.clone();
            let b = barrier.clone();
            let ts = texts(n);
            std::thread::spawn(move || {
                b.wait();
                for t in ts {
                    db.remember(&t, RememberOpts { ns: ns(n), ..Default::default() }).unwrap();
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    let main = current(&db, "main", None);
    let br = current(&db, "br", None);
    assert_eq!(main.len(), 7, "{main:?}");
    assert_eq!(br.len(), 7, "{br:?}");
    assert!(main.iter().all(|t| !t.starts_with("br ")));
    assert!(br.iter().all(|t| !t.starts_with("main ")));
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cozoscript_ops() {
    let m = Models::new("cs", vec![]);
    let db = m.db();
    db.remember(PARIS, at("main", d(1))).unwrap();
    let run = |s: &str, ro: bool| {
        let mutab = if ro { ScriptMutability::Immutable } else { ScriptMutability::Mutable };
        db.run_script(s, Default::default(), mutab)
    };
    let f = run("::memory fork 'br' {from: 'main'}", false).unwrap();
    assert!(f.headers.contains(&"lineage".to_string()));
    run("::memory remember 'I have a dog named Rex.' {ns: 'br'}", false).unwrap();
    let df = run("::memory diff {a: 'main', b: 'br'}", true).unwrap();
    assert!(df.headers.contains(&"added".to_string()));
    assert_eq!(run("::memory branches", true).unwrap().rows.len(), 1);
    assert!(run("::memory fork 'b2' {from: 'main'}", true).is_err(), "fork is a write");
    run("::memory merge 'br' {into: 'main'}", false).unwrap();
    run("::memory discard 'br'", false).unwrap();
    assert_eq!(current(&db, "main", None), set(&[PARIS, DOG]));
}
