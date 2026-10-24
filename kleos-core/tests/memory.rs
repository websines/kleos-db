/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The agent memory API (docs/specs/memory-api.md): remember / recall /
//! forget / reflect / link and the `::memory` ops, against a fake decider
//! and a fake embedder registered through the process-wide registries. The
//! registries are shared by every test in this binary, so the tests that
//! register models hold a read lock and the model-free test a write lock.

use std::collections::BTreeSet;
use std::sync::{Arc, RwLock};

use kleos::decide::{self, Calibration, Fitted};
use kleos::embed::{self, Embedder};
use kleos::memory::{ForgetOpts, InitOpts, LinkOpts, RecallOpts, RecallHit, ReflectOpts, RememberOpts, SetOpts};
use kleos::{DataValue, DbInstance, DecideAnswer, DecideRequest, Decider, Family, ScriptMutability, Template};

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

/// An NLI decider (decider-nano family, nano-tune-v1 calibration) that
/// answers from tables: `contra` pairs (old, new) contradict, `entail` pairs
/// (either order) entail, everything else is neutral.
struct Nli {
    id: String,
    contra: Vec<(&'static str, &'static str)>,
    entail: Vec<(&'static str, &'static str)>,
}

/// Pairs the fake decider is unsure about: P(contradiction) ≈ 0.9, below
/// the tuned threshold (0.997) but above the review cut (0.5).
const UNSURE: [(&str, &str); 1] = [("I work at Siemens.", "I joined Spotify.")];

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
                } else if UNSURE.iter().any(|&(x, y)| x == a && y == b) {
                    vec![0.0, 0.0, 2.2]
                } else if self.entail.iter().any(|&(x, y)| (x == a && y == b) || (x == b && y == a)) {
                    vec![12.0, 0.0, 0.0]
                } else {
                    vec![0.0, 6.0, 0.0]
                };
                DecideAnswer {
                    logits,
                    features: vec![],
                }
            })
            .collect())
    }
}

/// Registers a fake embedder and decider under unique names and removes
/// them on drop (also on panic).
struct Models {
    embedder: String,
    decider: String,
    _guard: std::sync::RwLockReadGuard<'static, ()>,
}

impl Models {
    fn new(tag: &str, contra: Vec<(&'static str, &'static str)>, entail: Vec<(&'static str, &'static str)>) -> Self {
        let guard = REGISTRY.read().unwrap_or_else(|e| e.into_inner());
        let embedder = format!("mem_emb_{tag}");
        let decider = format!("mem_dec_{tag}");
        embed::register_embedder(&embedder, Arc::new(Words)).unwrap();
        decide::register_decider(
            &decider,
            Arc::new(Nli {
                id: format!("mem-nli-{tag}"),
                contra,
                entail,
            }),
            Family::Nano,
        )
        .unwrap();
        Models {
            embedder,
            decider,
            _guard: guard,
        }
    }
    fn db(&self) -> DbInstance {
        let db = DbInstance::new("mem", "", "").unwrap();
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

fn recall(db: &DbInstance, q: &str, ns_: &str, as_of: Option<i64>) -> Vec<String> {
    db.recall(
        q,
        RecallOpts {
            ns: ns(ns_),
            as_of,
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

const BERLIN: &str = "I live in Berlin.";
const LISBON: &str = "I moved to Lisbon last month.";
const COFFEE: &str = "I like my coffee black.";

#[test]
fn add_supersede_and_time_travel() {
    let m = Models::new("sup", vec![(BERLIN, LISBON)], vec![]);
    let db = m.db();
    let a = db.remember(BERLIN, at("u", d(1))).unwrap();
    assert_eq!(a.action, "added");
    assert!(a.embedded);
    assert_eq!(a.decider.as_deref(), Some(m.decider.as_str()));
    let berlin = a.id.unwrap();
    let c = db.remember(COFFEE, at("u", d(2))).unwrap();
    assert!(c.superseded.is_empty(), "{c:?}");

    let b = db.remember(LISBON, at("u", d(10))).unwrap();
    assert_eq!(b.action, "added");
    assert_eq!(b.superseded, vec![berlin.clone()], "{b:?}");
    let applied: Vec<_> = b.decisions.iter().filter(|x| x.action == "superseded").collect();
    assert_eq!(applied.len(), 1);
    assert_eq!(applied[0].band, "auto");
    assert!(applied[0].decided_by.starts_with("mem-nli-sup|supersedes@v1|"), "{}", applied[0].decided_by);
    let lisbon = b.id.unwrap();

    // current: Lisbon, never Berlin
    let now = recall(&db, "where do I live, Berlin or Lisbon?", "u", None);
    assert!(now.contains(&LISBON.to_string()) && !now.contains(&BERLIN.to_string()), "{now:?}");
    // time travel: on day 5 Berlin was current, Lisbon not yet
    let then = recall(&db, "where do I live, Berlin or Lisbon?", "u", Some(d(5)));
    assert!(then.contains(&BERLIN.to_string()) && !then.contains(&LISBON.to_string()), "{then:?}");
    let hit = db
        .recall("Berlin", RecallOpts { ns: ns("u"), as_of: Some(d(5)), ..Default::default() })
        .unwrap()
        .into_iter()
        .find(|h| h.id == berlin)
        .unwrap();
    assert_eq!((hit.valid_from, hit.valid_to), (d(1), Some(d(10))));

    // nothing was deleted; the supersession is a link and an audit row
    assert_eq!(count(&db, "?[id, vt] := *kleos_mem{ns: 'u', id, vt}"), 4);
    assert_eq!(
        count(&db, &format!("?[s] := *kleos_mem_link{{ns: 'u', src: s, rel: 'supersedes', dst: '{berlin}'}}, s = '{lisbon}'")),
        1
    );
    assert_eq!(
        count(&db, "?[c] := *kleos_mem_decision{ns: 'u', cand: c, status: 'applied'}"),
        1
    );
    // the superseded vector no longer competes in current searches
    assert_eq!(count(&db, "?[id] := *kleos_mem_vec{ns: 'u', id, current: false}"), 1);

    // reflect sees the change feed
    let r = db
        .reflect(ReflectOpts { ns: ns("u"), since: Some(d(0)), ..Default::default() })
        .unwrap();
    let ch: BTreeSet<_> = r.changes.iter().map(|c| (c.id.clone(), c.change.clone(), c.at)).collect();
    assert!(ch.contains(&(berlin.clone(), "ended".to_string(), d(10))), "{ch:?}");
    assert!(ch.contains(&(lisbon, "added".to_string(), d(10))), "{ch:?}");
}

#[test]
fn unsure_supersede_is_pending_and_flagged() {
    let m = Models::new("unsure", vec![], vec![]);
    let db = m.db();
    let old = db.remember(UNSURE[0].0, at("u", d(1))).unwrap().id.unwrap();
    let r = db.remember(UNSURE[0].1, at("u", d(2))).unwrap();
    assert!(r.superseded.is_empty());
    assert_eq!(r.tentative.len(), 1, "{r:?}");
    let t = &r.tentative[0];
    assert_eq!((t.template.as_str(), t.band.as_str(), t.candidate.as_str()), ("supersedes", "none", old.as_str()));
    assert!(t.p > 0.5 && t.p < 0.997, "{t:?}");
    // reads choose: flagged by default, hidden on request
    let q = |hide| {
        db.recall("where do I work", RecallOpts { ns: ns("u"), hide_pending: hide, ..Default::default() })
            .unwrap()
    };
    let shown = q(false);
    let flagged = shown.iter().find(|h| h.id == old).unwrap();
    assert_eq!(flagged.pending_superseded_by, vec![r.id.clone().unwrap()]);
    assert!(q(true).iter().all(|h| h.id != old));
    // reflect lists it as pending and as a contradiction candidate
    let refl = db.reflect(ReflectOpts { ns: ns("u"), ..Default::default() }).unwrap();
    assert_eq!(refl.pending.len(), 1);
    assert_eq!(refl.contradictions.len(), 1);
}

#[test]
fn duplicates_and_tentative() {
    const PARA: &str = "My coffee: black, no sugar, no milk.";
    let m = Models::new("dup", vec![], vec![(COFFEE, PARA)]);
    let db = m.db();
    let a = db.remember(COFFEE, at("u", d(1))).unwrap().id.unwrap();

    // exact restatement (case and punctuation aside): no new memory, one more source
    let r = db
        .remember(
            "i like my coffee BLACK",
            RememberOpts { ns: ns("u"), source: Some("chat-2".into()), ..Default::default() },
        )
        .unwrap();
    assert_eq!(r.action, "duplicate");
    assert_eq!(r.id, None);
    assert_eq!(r.duplicate_of.as_deref(), Some(a.as_str()));
    assert!(r.decisions[0].decided_by.contains("|exact"), "{r:?}");
    assert_eq!(count(&db, "?[id] := *kleos_mem{ns: 'u', id @ 'NOW'}"), 1);
    assert_eq!(count(&db, &format!("?[s] := *kleos_mem_source{{ns: 'u', id: '{a}', stmt: s}}")), 2);

    // a paraphrase: nano-tune-v1 never auto-applies duplicate, so it is
    // inserted with a pending decision
    let t = db.remember(PARA, at("u", d(2))).unwrap();
    assert_eq!(t.action, "added");
    assert_eq!(t.tentative.len(), 1, "{t:?}");
    assert_eq!((t.tentative[0].template.as_str(), t.tentative[0].band.as_str()), ("duplicate", "tentative"));
    assert_eq!(t.tentative[0].candidate, a);
    let r = db.reflect(ReflectOpts { ns: ns("u"), ..Default::default() }).unwrap();
    assert_eq!(r.pending.len(), 1);
    assert_eq!(r.pending[0].new_text.as_deref(), Some(PARA));
    assert_eq!(r.pending[0].candidate_text.as_deref(), Some(COFFEE));

    // with a fitted calibration that has an auto band, the same pair is
    // skipped as a duplicate
    decide::install_fitted(
        &format!("mem-nli-dup"),
        Template::Duplicate,
        Fitted::Calibration {
            template_version: 1,
            calibration: Calibration { threshold: 0.5, tau_auto: 0.9, ..Default::default() },
        },
    );
    let db2 = m.db();
    let a2 = db2.remember(COFFEE, at("u", d(1))).unwrap().id.unwrap();
    let r = db2.remember(PARA, at("u", d(2))).unwrap();
    decide::remove_fitted("mem-nli-dup", Template::Duplicate);
    assert_eq!(r.action, "duplicate", "{r:?}");
    assert_eq!(r.duplicate_of, Some(a2));
}

#[test]
fn forget_retract_and_erase() {
    let m = Models::new("forget", vec![], vec![]);
    let db = m.db();
    let a = db
        .remember(
            "Alice's phone number is 555-0100.",
            RememberOpts { ns: ns("u"), at: Some(d(1)), entities: vec!["Alice".into()], ..Default::default() },
        )
        .unwrap()
        .id
        .unwrap();
    let b = db.remember(COFFEE, at("u", d(1))).unwrap().id.unwrap();
    let alice = db
        .link(LinkOpts { ns: ns("u"), from: b.clone(), to: a.clone(), rel: "about_same_person".into(), p: None })
        .unwrap();
    assert_eq!(alice["rel"], "about_same_person");

    // retract (default): gone from current recall, still there in the past
    let f = db
        .forget(ForgetOpts { ns: ns("u"), id: Some(b.clone()), at: Some(d(5)), ..Default::default() })
        .unwrap();
    assert_eq!((f.mode.as_str(), f.ids.clone()), ("retract", vec![b.clone()]));
    // (the vector leg always returns nearest neighbours, so check membership)
    assert!(!recall(&db, "coffee", "u", None).contains(&COFFEE.to_string()));
    assert_eq!(recall(&db, "coffee", "u", Some(d(3)))[0], COFFEE);
    // retracting again is a no-op
    assert!(db.forget(ForgetOpts { ns: ns("u"), id: Some(b.clone()), ..Default::default() }).unwrap().ids.is_empty());

    // a filter needs a filter
    assert!(db.forget(ForgetOpts { ns: ns("u"), ..Default::default() }).is_err());

    // erase by entity: every version, its vectors, sources, mentions,
    // links, FTS entries, and the entity left with no mention
    let e = db
        .forget(ForgetOpts { ns: ns("u"), entity: Some(" alice ".into()), mode: Some("erase".into()), ..Default::default() })
        .unwrap();
    assert_eq!(e.ids, vec![a.clone()]);
    for (rel, n) in [
        ("?[v] := *kleos_mem{ns: 'u', id: $id, vt: v}", 0),
        ("?[i] := *kleos_mem_vec{ns: 'u', id: i}, i = $id", 0),
        ("?[s] := *kleos_mem_source{ns: 'u', id: $id, stmt: s}", 0),
        ("?[e] := *kleos_mem_mention{ns: 'u', id: $id, eid: e}", 0),
        ("?[e] := *kleos_mem_entity{ns: 'u', eid: e}", 0),
        ("?[s] := *kleos_mem_link{ns: 'u', src: s}", 0),
        ("?[i] := ~kleos_mem:fts{id: i | query: 'phone', k: 10}", 0),
    ] {
        let rows = db
            .run_script(rel, [("id".to_string(), DataValue::from(a.as_str()))].into(), ScriptMutability::Immutable)
            .unwrap();
        assert_eq!(rows.rows.len(), n, "{rel}");
    }
    assert!(!recall(&db, "phone number", "u", Some(d(2))).iter().any(|t| t.contains("555")));
}

#[test]
fn namespaces_are_isolated() {
    let m = Models::new("ns", vec![(BERLIN, LISBON)], vec![]);
    let db = m.db();
    db.remember(BERLIN, at("alice", d(1))).unwrap();
    db.remember(BERLIN, at("bob", d(1))).unwrap();
    // same text in another namespace is not a duplicate
    assert_eq!(count(&db, "?[ns, id] := *kleos_mem{ns, id @ 'NOW'}"), 2);
    // alice moves; bob does not
    let r = db.remember(LISBON, at("alice", d(9))).unwrap();
    assert_eq!(r.superseded.len(), 1);
    assert_eq!(recall(&db, "Berlin Lisbon live", "bob", None), vec![BERLIN.to_string()]);
    assert_eq!(recall(&db, "Berlin Lisbon live", "alice", None), vec![LISBON.to_string()]);
    assert!(recall(&db, "Berlin", "carol", None).is_empty());
    // kind filter
    db.remember(
        "Prefers aisle seats.",
        RememberOpts { ns: ns("bob"), kind: Some("preference".into()), ..Default::default() },
    )
    .unwrap();
    let hits = db
        .recall("seats Berlin", RecallOpts { ns: ns("bob"), kind: Some("preference".into()), ..Default::default() })
        .unwrap();
    assert_eq!(hits.iter().map(|h| h.text.as_str()).collect::<Vec<_>>(), ["Prefers aisle seats."]);
}

#[test]
fn entities_link_and_filter() {
    let m = Models::new("ent", vec![], vec![]);
    let db = m.db();
    let r1 = db
        .remember(
            "Alice works at Acme.",
            RememberOpts { ns: ns("u"), entities: vec!["Alice".into(), "Acme".into()], ..Default::default() },
        )
        .unwrap();
    let r2 = db
        .remember(
            "Alice's salary is $90,000.",
            RememberOpts { ns: ns("u"), at: Some(d(1)), entities: vec!["alice ".into()], ..Default::default() },
        )
        .unwrap();
    // exact normalised-name match: one Alice
    assert_eq!(r2.entities.len(), 1);
    assert!(r1.entities.contains(&r2.entities[0]));
    assert_eq!(count(&db, "?[e] := *kleos_mem_entity{ns: 'u', eid: e}"), 2);
    let hits = db
        .recall("works salary", RecallOpts { ns: ns("u"), entity: Some("ACME".into()), ..Default::default() })
        .unwrap();
    assert_eq!(hits.iter().map(|h| h.text.as_str()).collect::<Vec<_>>(), ["Alice works at Acme."]);
    // link an entity to a memory
    db.link(LinkOpts { ns: ns("u"), from: r1.entities[0].clone(), to: r2.id.clone().unwrap(), rel: "subject_of".into(), p: Some(0.8) })
        .unwrap();
    assert!(db
        .link(LinkOpts { ns: ns("u"), from: "nope".into(), to: r2.id.unwrap(), rel: "x".into(), p: None })
        .is_err());
    // reflect: model-free contradiction candidates through shared entities
    db.remember(
        "Alice's salary is $95,000.",
        RememberOpts { ns: ns("u"), at: Some(d(2)), entities: vec!["Alice".into()], ..Default::default() },
    )
    .unwrap();
    let r = db.reflect(ReflectOpts { ns: ns("u"), ..Default::default() }).unwrap();
    assert!(
        r.contradictions.iter().any(|c| c.a_text.contains("90,000") && c.b_text.contains("95,000")),
        "{:?}",
        r.contradictions
    );
}

#[test]
fn cozoscript_ops() {
    let m = Models::new("ops", vec![(BERLIN, LISBON)], vec![]);
    let db = DbInstance::new("mem", "", "").unwrap();
    let run = |s: &str| {
        db.run_script(s, Default::default(), ScriptMutability::Mutable)
            .unwrap_or_else(|e| panic!("{s}: {e:?}"))
    };
    let cfg = run(&format!("::memory init {{embedder: '{}', decider: '{}'}}", m.embedder, m.decider));
    assert!(cfg.headers.contains(&"embedder".to_string()));
    let r = run(&format!("::memory remember '{BERLIN}' {{ns: 'u', at: {}}}", d(1)));
    let col = |r: &kleos::NamedRows, c: &str| r.rows[0][r.headers.iter().position(|h| h == c).unwrap()].clone();
    assert_eq!(col(&r, "action"), DataValue::from("added"));
    let berlin = col(&r, "id");
    let r = run(&format!("::memory remember '{LISBON}' {{ns: 'u', at: {}, entities: ['Lisbon']}}", d(9)));
    assert_eq!(col(&r, "superseded"), DataValue::List(vec![berlin.clone()]));
    let hits = run("::memory recall 'where do I live' {ns: 'u', k: 3}");
    assert_eq!(hits.headers[..3], ["id", "text", "score"]);
    assert_eq!(hits.rows.len(), 1);
    assert_eq!(hits.rows[0][1], DataValue::from(LISBON));
    let old = run(&format!("::memory recall 'where do I live' {{ns: 'u', as_of: {}}}", d(2)));
    assert_eq!(old.rows[0][1], DataValue::from(BERLIN));
    let lisbon = hits.rows[0][0].get_str().unwrap().to_string();
    run(&format!("::memory link {{ns: 'u', from: '{lisbon}', to: '{}', rel: 'follows'}}", berlin.get_str().unwrap()));
    let refl = run("::memory reflect 'u' {since: 0}");
    assert!(refl.headers.contains(&"changes".to_string()));
    let f = db
        .run_script(
            "::memory forget $id {ns: 'u'}",
            [("id".to_string(), DataValue::from(lisbon.as_str()))].into(),
            ScriptMutability::Mutable,
        )
        .unwrap();
    assert_eq!(col(&f, "ids"), DataValue::List(vec![DataValue::from(lisbon.as_str())]));
    assert!(run("::memory recall 'Lisbon' {ns: 'u'}").rows.is_empty());
    // read-only scripts may recall and reflect, nothing else
    assert!(db.run_script("::memory recall 'x' {ns: 'u'}", Default::default(), ScriptMutability::Immutable).is_ok());
    assert!(db
        .run_script("::memory remember 'x'", Default::default(), ScriptMutability::Immutable)
        .is_err());
    // bad options are loud
    let e = db.run_script("::memory recall 'x' {nope: 1}", Default::default(), ScriptMutability::Mutable);
    assert!(format!("{:?}", e.unwrap_err()).contains("nope"));
    // the JSON entry point used by HTTP
    let v = db
        .memory_call("recall", serde_json::json!({"query": "Berlin", "ns": "u", "as_of": d(2)}))
        .unwrap();
    assert_eq!(v[0]["text"], BERLIN);
}

#[test]
fn no_ml_degrades_to_add_only_fts_and_rules() {
    // nobody else may have a model registered while this runs
    let _w = REGISTRY.write().unwrap_or_else(|e| e.into_inner());
    assert!(decide::deciders().iter().all(|d| !d.enabled));
    assert!(embed::embedders().iter().all(|e| !e.enabled));
    let db = DbInstance::new("mem", "", "").unwrap();
    // lazily created by the first remember, without vectors
    let a = db.remember("My rent is $1,800 per month.", at("u", d(1))).unwrap();
    assert_eq!((a.action.as_str(), a.embedded, a.decider.clone()), ("added", false, None));
    db.remember(BERLIN, at("u", d(2))).unwrap();
    db.remember(LISBON, at("u", d(3))).unwrap(); // no model: both stay current
    assert_eq!(count(&db, "?[id] := *kleos_mem{ns: 'u', id @ 'NOW'}"), 3);
    // the value rule still supersedes a changed amount
    let b = db.remember("My rent is $2,100 per month.", at("u", d(4))).unwrap();
    assert_eq!(b.superseded, vec![a.id.clone().unwrap()], "{b:?}");
    assert!(b.decisions[0].decided_by.starts_with("rules|supersedes@v1|rule"), "{b:?}");
    // and exact restatements are duplicates
    assert_eq!(db.remember("I live in Berlin", at("u", d(5))).unwrap().action, "duplicate");
    // FTS-only recall
    assert_eq!(recall(&db, "how much is the rent?", "u", None), vec!["My rent is $2,100 per month.".to_string()]);
    assert_eq!(recall(&db, "rent", "u", Some(d(2))), vec!["My rent is $1,800 per month.".to_string()]);
    // reflect still works
    let r = db.reflect(ReflectOpts { ns: ns("u"), since: Some(0), ..Default::default() }).unwrap();
    assert!(r.changes.iter().any(|c| c.change == "ended"));
}

// ------------------------------------------------------- entities (E3 hook)

fn with_ents(ns_: &str, t: i64, ents: &[&str]) -> RememberOpts {
    RememberOpts {
        ns: ns(ns_),
        at: Some(t),
        entities: ents.iter().map(|e| e.to_string()).collect(),
        ..Default::default()
    }
}

const SISTER: &str = "My sister Anna lives in Munich.";

#[test]
fn entities_gate_supersedes_to_shared_subjects() {
    // the fake decider says the sister's city "supersedes" the user's: only
    // the entity gate stands between them
    let m = Models::new("gate", vec![(BERLIN, SISTER), (BERLIN, LISBON)], vec![]);
    let db = m.db();
    let berlin = db.remember(BERLIN, with_ents("u", d(1), &["user", "Berlin"])).unwrap();
    assert_eq!(berlin.entity_links.iter().map(|e| e.outcome.as_str()).collect::<Vec<_>>(), ["new", "new"]);
    let berlin = berlin.id.unwrap();

    let s = db.remember(SISTER, with_ents("u", d(2), &["Anna", "Munich"])).unwrap();
    assert!(s.superseded.is_empty() && s.tentative.is_empty(), "{s:?}");
    assert_eq!(s.gated, vec![berlin.clone()], "{s:?}");
    // the question is not even asked across subjects
    assert!(!s.decisions.iter().any(|x| x.template == "supersedes"), "{s:?}");
    assert_eq!(recall(&db, "where does everyone live", "u", None).len(), 2);

    // the same subject ("user", resolved to the existing entity) supersedes
    let l = db.remember(LISBON, with_ents("u", d(9), &["User", "Lisbon"])).unwrap();
    assert_eq!(l.superseded, vec![berlin.clone()], "{l:?}");
    let user = &l.entity_links[0];
    assert_eq!((user.outcome.as_str(), user.created), ("match", false), "{l:?}");
    assert!(user.decided_by.starts_with("rule|exact"), "{user:?}");
    // one "user" entity in all
    assert_eq!(count(&db, "?[e] := *kleos_mem_entity{ns: 'u', eid: e, norm: 'user'}"), 1);

    // without entities on the new statement, the default ontology still
    // tells the subjects apart ("my sister" is not the user)
    // (one registry read guard at a time: a waiting writer would deadlock)
    drop(db);
    drop(m);
    let m2 = Models::new("gate2", vec![(BERLIN, SISTER)], vec![]);
    let db2 = m2.db();
    let b2 = db2.remember(BERLIN, with_ents("u", d(1), &["user"])).unwrap().id.unwrap();
    let s2 = db2.remember(SISTER, at("u", d(2))).unwrap();
    assert!(s2.superseded.is_empty(), "{s2:?}");
    assert!(s2.ontology_blocked.contains(&b2), "{s2:?}");
    // with the default ontology off: as before (no gate)
    db2.memory_init(InitOpts { ontology: Some("none".into()), ..Default::default() }).unwrap();
    let b3 = db2.remember(BERLIN, with_ents("v", d(1), &["user"])).unwrap().id.unwrap();
    let s3 = db2.remember(SISTER, at("v", d(2))).unwrap();
    assert_eq!(s3.superseded, vec![b3], "{s3:?}");
}

#[test]
fn entities_resolve_merge_review_and_split() {
    // rules only (an unknown decider): exact names match by rule, a new
    // surface form of the same name is merged, initials are only reviewed
    let m = Models::new("eres", vec![], vec![]);
    let db = m.db();
    let rules = |ns_: &str, t: i64, ents: &[&str]| RememberOpts { decider: Some("none".into()), ..with_ents(ns_, t, ents) };
    const RENT1: &str = "My rent is $1,800 per month.";
    const RENT2: &str = "My rent is $2,100 per month.";
    const RENT3: &str = "My rent is $2,400 per month.";
    let a = db.remember(RENT1, rules("u", d(1), &["Alice Smith"])).unwrap();
    let alice = a.entities[0].clone();

    // "A. Smith" may be Alice Smith (initial rule, tentative): a new entity
    // with a review link, and the value-rule supersede is capped at pending
    let b = db.remember(RENT2, rules("u", d(2), &["A. Smith"])).unwrap();
    assert_eq!(b.entity_links[0].outcome, "possible", "{b:?}");
    assert_ne!(b.entities[0], alice);
    assert!(b.superseded.is_empty(), "{b:?}");
    assert_eq!(b.tentative.len(), 1, "{b:?}");
    assert_eq!(b.tentative[0].candidate, a.id.clone().unwrap());
    let pend = db.entity_pending("kleos_mem_entity").unwrap();
    assert_eq!(pend.rows.len(), 1, "{pend:?}");
    assert_eq!(pend.rows[0][1], DataValue::from(alice.as_str()));

    // "Dr. Alice Smith" is Alice Smith by rule (honorific dropped): its own
    // row, merged into hers, and the supersede acts
    let c = db.remember(RENT3, rules("u", d(3), &["Dr. Alice Smith"])).unwrap();
    let link = &c.entity_links[0];
    assert_eq!((link.outcome.as_str(), link.created, link.canonical.as_str()), ("match", true, alice.as_str()), "{c:?}");
    assert_eq!(c.superseded, vec![a.id.clone().unwrap()], "{c:?}");
    assert_eq!(db.canonical_entity("kleos_mem_entity", &DataValue::from(link.eid.as_str())).unwrap(), DataValue::from(alice.as_str()));
    // the entity filter follows the merge
    let by = |e: &str| -> Vec<String> {
        db.recall("rent", RecallOpts { ns: ns("u"), entity: Some(e.into()), ..Default::default() })
            .unwrap()
            .into_iter()
            .map(|h| h.text)
            .collect()
    };
    assert_eq!(by("Alice Smith"), vec![RENT3.to_string()]);
    // non-destructive: a split restores two entities
    assert_eq!(
        db.split_entity("kleos_mem_entity", &DataValue::from(link.eid.as_str())).unwrap(),
        Some(DataValue::from(alice.as_str()))
    );
    assert!(by("Alice Smith").is_empty());
    assert_eq!(by("Dr. Alice Smith"), vec![RENT3.to_string()]);

    // erase removes the entities, their merges and links with the memories
    let ids = vec![a.id.unwrap(), b.id.unwrap(), c.id.clone().unwrap()];
    db.forget(ForgetOpts { ns: ns("u"), ids, mode: Some("erase".into()), ..Default::default() }).unwrap();
    assert_eq!(count(&db, "?[e] := *kleos_mem_entity{ns: 'u', eid: e}"), 0);
    assert_eq!(count(&db, "?[f, vt] := *kleos_entity_merge{relation: 'kleos_mem_entity', from: f, vt}"), 0);
    assert_eq!(count(&db, "?[a, b] := *kleos_entity_link{relation: 'kleos_mem_entity', a, b}"), 0);
}

// ------------------------------------------------------- scoring (E4, C6)

fn hits(db: &DbInstance, q: &str, opts: RecallOpts) -> Vec<RecallHit> {
    db.recall(q, RecallOpts { ns: ns("s"), record_access: Some(false), ..opts }).unwrap()
}

fn texts(h: &[RecallHit]) -> Vec<&str> {
    h.iter().map(|x| x.memory_text()).collect()
}

const STRONG: &str = "The garage door code is 4417, garage door.";
const WEAK: &str = "Door code for the shed.";

#[test]
fn recency_and_importance_rank_and_blend_off_is_relevance() {
    let m = Models::new("rank", vec![], vec![]);
    let db = m.db();
    let now = kleos_now();
    let strong = db.remember(STRONG, at("s", now - 600 * DAY)).unwrap().id.unwrap();
    let weak = db.remember(WEAK, at("s", now - DAY)).unwrap().id.unwrap();
    let q = "garage door code";
    // relevance alone: the stronger text first, score = relevance
    let off = hits(&db, q, RecallOpts { blend: Some(false), ..Default::default() });
    assert_eq!(texts(&off), [STRONG, WEAK]);
    assert!(off.iter().all(|h| h.score == h.relevance), "{off:?}");
    // the recency signal halves every half-life
    let r: Vec<f64> = off.iter().map(|h| h.recency).collect();
    assert!((r[1] - 0.5f64.powf(1.0 / 90.0)).abs() < 1e-3 && (r[0] - 0.5f64.powf(600.0 / 90.0)).abs() < 1e-4, "{r:?}");
    // a heavy recency weight puts the fresh memory first; default weights
    // do not override a clear relevance gap
    let rec = hits(&db, q, RecallOpts { recency_weight: Some(5.0), ..Default::default() });
    assert_eq!(texts(&rec), [WEAK, STRONG]);
    assert_eq!(texts(&hits(&db, q, RecallOpts::default())), [STRONG, WEAK]);
    // a short half-life makes both old: relevance decides again
    let short = hits(&db, q, RecallOpts { recency_weight: Some(5.0), half_life_days: Some(0.01), ..Default::default() });
    assert_eq!(texts(&short), [STRONG, WEAK]);

    // importance: set later with `set`, used by the blend, ignored without
    db.memory_set(SetOpts { ns: ns("s"), id: Some(weak.clone()), importance: Some(1.0), ..Default::default() })
        .unwrap();
    db.memory_set(SetOpts { ns: ns("s"), id: Some(strong.clone()), importance: Some(0.0), ..Default::default() })
        .unwrap();
    let imp = hits(&db, q, RecallOpts { importance_weight: Some(5.0), recency_weight: Some(0.0), ..Default::default() });
    assert_eq!(texts(&imp), [WEAK, STRONG]);
    assert_eq!(imp[0].importance, Some(1.0));
    let off2 = hits(&db, q, RecallOpts { blend: Some(false), ..Default::default() });
    assert_eq!(texts(&off2), [STRONG, WEAK]);
    assert_eq!(off2.iter().map(|h| h.score).collect::<Vec<_>>(), off.iter().map(|h| h.score).collect::<Vec<_>>());
    // importance is metadata: no new version, validity unchanged
    assert_eq!(count(&db, "?[id, vt] := *kleos_mem{ns: 's', id, vt}"), 2);
    assert_eq!(off2[1].valid_from, now - DAY);
    assert!(db.memory_set(SetOpts { ns: ns("s"), id: Some(weak), importance: Some(2.0), ..Default::default() }).is_err());
    // the remember option and the `::memory set` op
    let c = db
        .remember("A note about the boat.", RememberOpts { ns: ns("s"), importance: Some(0.9), ..Default::default() })
        .unwrap()
        .id
        .unwrap();
    db.run_script(&format!("::memory set '{c}' {{ns: 's', importance: 0.3}}"), Default::default(), ScriptMutability::Mutable)
        .unwrap();
    assert_eq!(hits(&db, "boat", RecallOpts::default())[0].importance, Some(0.3));
}

fn kleos_now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros() as i64
}

#[test]
fn access_counts_are_buffered_flushed_and_ranked() {
    let m = Models::new("acc", vec![], vec![]);
    let db = m.db();
    let id = db.remember(STRONG, at("s", d(1))).unwrap().id.unwrap();
    let rec = |db: &DbInstance| db.recall("garage", RecallOpts { ns: ns("s"), ..Default::default() }).unwrap();
    assert_eq!(rec(&db)[0].access_count, 0);
    rec(&db);
    rec(&db);
    // counted, flushed or not (the first recall's flush is due at once)
    let h = rec(&db);
    assert_eq!(h[0].access_count, 3, "{h:?}");
    assert!(h[0].last_access.is_some());
    db.memory_flush().unwrap();
    assert_eq!(
        db.run_script(&format!("?[n] := *kleos_mem_access{{ns: 's', id: '{id}', n}}"), Default::default(), ScriptMutability::Immutable)
            .unwrap()
            .rows[0][0],
        DataValue::from(4)
    );
    // reads that do not count: the option, and read-only scripts
    hits(&db, "garage", RecallOpts::default());
    db.run_script("::memory recall 'garage' {ns: 's'}", Default::default(), ScriptMutability::Immutable)
        .unwrap();
    assert_eq!(rec(&db)[0].access_count, 4);
    // the flush never touches the namespace version
    let seq = |db: &DbInstance| {
        db.run_script("?[seq] := *kleos_mem_ns{ns: 's', seq}", Default::default(), ScriptMutability::Immutable)
            .unwrap()
            .rows[0][0]
            .clone()
    };
    let before = seq(&db);
    rec(&db);
    db.memory_flush().unwrap();
    assert_eq!(seq(&db), before);

    // ranking: a much-accessed memory overtakes a slightly more relevant one
    let w = db.remember(WEAK, at("s", d(1))).unwrap().id.unwrap();
    for _ in 0..50 {
        db.recall("shed", RecallOpts { ns: ns("s"), k: Some(1), ..Default::default() }).unwrap();
    }
    let a = hits(&db, "garage door code", RecallOpts { access_weight: Some(5.0), recency_weight: Some(0.0), ..Default::default() });
    let rel = hits(&db, "garage door code", RecallOpts { blend: Some(false), ..Default::default() });
    assert!(a.iter().find(|h| h.id == w).unwrap().access_count >= 50);
    // by default access counts are recorded, not ranked
    assert_eq!(hits(&db, "garage door code", RecallOpts { recency_weight: Some(0.0), ..Default::default() })[0].id, id);
    assert_eq!(a[0].id, w, "{a:?}");
    assert_eq!(rel[0].id, id, "{rel:?}");
    // erase drops the counts, buffered and stored
    db.forget(ForgetOpts { ns: ns("s"), id: Some(w.clone()), mode: Some("erase".into()), ..Default::default() }).unwrap();
    db.memory_flush().unwrap();
    assert_eq!(count(&db, &format!("?[n] := *kleos_mem_access{{ns: 's', id: '{w}', n}}")), 0);
}

#[test]
fn ttl_expiry_and_archive_tier() {
    let m = Models::new("ttl", vec![], vec![]);
    let db = m.db();
    const GATE: &str = "Today the visitor parking gate code is 9912.";
    // valid from day 1, expires a day later
    let e = db
        .remember(GATE, RememberOpts { ns: ns("s"), at: Some(d(1)), ttl: Some(DAY), ..Default::default() })
        .unwrap()
        .id
        .unwrap();
    let keep = db.remember(STRONG, at("s", d(1))).unwrap().id.unwrap();
    let q = "gate code";
    assert_eq!(texts(&hits(&db, q, RecallOpts::default())), [STRONG]);
    let all = hits(&db, q, RecallOpts { include_expired: true, ..Default::default() });
    assert_eq!(all.len(), 2);
    assert_eq!(all.iter().find(|h| h.id == e).unwrap().expires, Some(d(2)));
    // before it expired, a past read sees it
    assert_eq!(hits(&db, q, RecallOpts { as_of: Some(d(1) + 3_600_000_000), ..Default::default() }).len(), 2);
    // an exact restatement is not swallowed by the expired memory
    let again = db.remember(GATE, at("s", d(3))).unwrap();
    assert_eq!(again.action, "added", "{again:?}");
    let again = again.id.unwrap();

    // archive the expired ones: still valid, hidden from recall by default
    let f = db
        .forget(ForgetOpts { ns: ns("s"), expired: true, mode: Some("archive".into()), ..Default::default() })
        .unwrap();
    assert_eq!(f.ids, vec![e.clone()]);
    assert_eq!(count(&db, &format!("?[id] := *kleos_mem{{ns: 's', id @ 'NOW'}}, id = '{e}'")), 1);
    assert_eq!(hits(&db, q, RecallOpts { include_expired: true, ..Default::default() }).len(), 2);
    let arch = hits(&db, q, RecallOpts { include_expired: true, include_archived: true, ..Default::default() });
    assert!(arch.iter().any(|h| h.id == e && h.archived), "{arch:?}");

    // archive by id and bring it back
    db.memory_set(SetOpts { ns: ns("s"), id: Some(keep.clone()), archived: Some(true), ..Default::default() }).unwrap();
    assert_eq!(texts(&hits(&db, q, RecallOpts::default())), [GATE]);
    db.run_script(&format!("::memory set '{keep}' {{ns: 's', archived: false}}"), Default::default(), ScriptMutability::Mutable)
        .unwrap();
    assert_eq!(hits(&db, q, RecallOpts::default()).len(), 2);
    // a TTL can be set later and removed
    db.memory_set(SetOpts { ns: ns("s"), id: Some(again.clone()), ttl: Some(-1), ..Default::default() }).unwrap();
    assert_eq!(texts(&hits(&db, q, RecallOpts::default())), [STRONG]);
    db.memory_set(SetOpts { ns: ns("s"), id: Some(again.clone()), no_ttl: true, ..Default::default() }).unwrap();
    assert_eq!(hits(&db, q, RecallOpts::default()).len(), 2);
    // erase clears the tier rows
    db.forget(ForgetOpts { ns: ns("s"), id: Some(e.clone()), mode: Some("erase".into()), ..Default::default() }).unwrap();
    assert_eq!(count(&db, &format!("?[id] := *kleos_mem_life{{ns: 's', id}}, id = '{e}'")), 0);
}

/// Real models: decider-nano (nano-tune-v1) and EmbeddingGemma, downloaded
/// into the Hugging Face cache on first use. Run with
/// `cargo +1.99.0 test -p kleos --features infer-metal --test memory -- --include-ignored`.
#[cfg(feature = "infer")]
#[test]
#[ignore]
fn real_models_remember_and_recall() {
    let _g = REGISTRY.read().unwrap_or_else(|e| e.into_inner());
    embed::enable_embedder("mem_real_gemma", embed::EmbedderSpec::embeddinggemma()).unwrap();
    decide::enable_decider("mem_real_nano", decide::DeciderSpec::model("decider-nano")).unwrap();
    let db = DbInstance::new("mem", "", "").unwrap();
    db.memory_init(InitOpts { embedder: Some("mem_real_gemma".into()), decider: Some("mem_real_nano".into()), ..Default::default() })
        .unwrap();
    let a = db.remember("I live in Berlin.", at("u", d(1))).unwrap();
    db.remember("My favourite food is ramen.", at("u", d(2))).unwrap();
    let b = db.remember("I moved to Lisbon; I don't live in Berlin anymore.", at("u", d(9))).unwrap();
    let c = db.remember("My favourite food is ramen.", at("u", d(10))).unwrap();
    let hits = recall(&db, "Which city do I live in?", "u", None);
    embed::disable_embedder("mem_real_gemma").unwrap();
    decide::disable_decider("mem_real_nano").unwrap();
    assert_eq!(b.superseded, vec![a.id.unwrap()], "{b:#?}");
    assert_eq!(c.action, "duplicate");
    assert!(hits[0].contains("Lisbon"), "{hits:?}");
    assert!(!hits.iter().any(|h| h == "I live in Berlin."), "{hits:?}");
}

// ------------------------------------------------------------- concurrency

const PORTO: &str = "I moved to Porto last month.";

/// The `Nli` tables for the race: both moves supersede Berlin and each other.
fn race_nli(id: &str) -> Nli {
    Nli {
        id: id.to_string(),
        contra: vec![(BERLIN, LISBON), (BERLIN, PORTO), (LISBON, PORTO), (PORTO, LISBON)],
        entail: vec![],
    }
}

/// Holds this process's first `n` decide calls in `wait(call index)` (a
/// barrier between threads, or a file rendezvous between processes), so
/// concurrent `remember`s plan before any of them commits.
struct Held {
    inner: Nli,
    n: usize,
    calls: std::sync::atomic::AtomicUsize,
    wait: Box<dyn Fn(usize) + Send + Sync>,
}

impl Decider for Held {
    fn model_id(&self) -> String {
        self.inner.model_id()
    }
    fn weights_version(&self) -> Option<String> {
        self.inner.weights_version()
    }
    fn decide(&self, reqs: &[DecideRequest], f: bool) -> miette::Result<Vec<DecideAnswer>> {
        let i = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if i < self.n {
            (self.wait)(i);
        }
        self.inner.decide(reqs, f)
    }
}

/// Memories current now in namespace `race` among `texts`.
fn current(db: &DbInstance, texts: &[&str]) -> Vec<String> {
    let rows = db
        .run_script(
            "?[text] := *kleos_mem{ns: 'race', text @ 'NOW'}, text in $texts",
            [(
                "texts".to_string(),
                DataValue::List(texts.iter().map(|t| DataValue::from(*t)).collect()),
            )]
            .into_iter()
            .collect(),
            ScriptMutability::Immutable,
        )
        .unwrap();
    rows.rows.into_iter().map(|r| r[0].get_str().unwrap().to_string()).collect()
}

/// Two `remember`s on one topic planned at the same moment in one process:
/// the later committer must see the earlier one and supersede it.
#[test]
fn concurrent_remembers_never_leave_two_current() {
    let _guard = REGISTRY.read().unwrap_or_else(|e| e.into_inner());
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let b = barrier.clone();
    let (emb, dec) = ("mem_emb_race", "mem_dec_race");
    embed::register_embedder(emb, Arc::new(Words)).unwrap();
    decide::register_decider(
        dec,
        Arc::new(Held {
            inner: race_nli("mem-nli-race"),
            n: 2,
            calls: Default::default(),
            wait: Box::new(move |_| {
                b.wait();
            }),
        }),
        Family::Nano,
    )
    .unwrap();
    let db = DbInstance::new("mem", "", "").unwrap();
    db.memory_init(InitOpts {
        embedder: Some(emb.into()),
        decider: Some(dec.into()),
        ..Default::default()
    })
    .unwrap();
    // seeded without the decider being asked (no candidates yet)
    db.remember(BERLIN, RememberOpts { ns: ns("race"), at: Some(d(1)), ..Default::default() })
        .unwrap();
    let results: Vec<_> = std::thread::scope(|sc| {
        let hs: Vec<_> = [LISBON, PORTO]
            .into_iter()
            .map(|t| {
                let db = &db;
                // both introduce the same new entities while planning
                let ents = vec!["user".to_string(), "Anna".to_string()];
                sc.spawn(move || {
                    db.remember(t, RememberOpts { ns: ns("race"), entities: ents, ..Default::default() })
                        .unwrap()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    embed::unregister_embedder(emb);
    decide::unregister_decider(dec);
    let cur = current(&db, &[BERLIN, LISBON, PORTO]);
    assert_eq!(cur.len(), 1, "exactly one current home, got {cur:?}; {results:?}");
    // the later committer superseded the earlier move (and Berlin)
    let ids: BTreeSet<String> = results.iter().filter_map(|r| r.id.clone()).collect();
    assert!(
        results.iter().any(|r| r.superseded.iter().any(|x| ids.contains(x))),
        "one move should supersede the other: {results:?}"
    );
    // entity writes are under the same version check: the loser re-planned
    // and linked the winner's entities instead of creating its own
    for n in ["user", "anna"] {
        assert_eq!(count(&db, &format!("?[e] := *kleos_mem_entity{{ns: 'race', eid: e, norm: '{n}'}}")), 1, "{results:?}");
    }
    assert_eq!(results[0].entities, results[1].entities);
}

/// Child half of `concurrent_remembers_across_processes`, run only when
/// spawned by it. Both children plan against the same state (rendezvous at
/// their first decide call); child `a` then finishes planning only after
/// child `b` has committed, so `a` commits a stale plan unless the write
/// transaction notices.
#[test]
#[ignore]
fn race_child() {
    let Ok(path) = std::env::var("KLEOS_RACE_DB") else { return };
    let text = std::env::var("KLEOS_RACE_TEXT").unwrap();
    let dir = std::path::PathBuf::from(std::env::var("KLEOS_RACE_DIR").unwrap());
    let me = std::env::var("KLEOS_RACE_ME").unwrap();
    let wait_for = |pred: &dyn Fn() -> bool| {
        let t = std::time::Instant::now();
        while !pred() {
            assert!(t.elapsed().as_secs() < 30, "rendezvous timed out");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    };
    let (d2, me2) = (dir.clone(), me.clone());
    embed::register_embedder("mem_emb_race", Arc::new(Words)).unwrap();
    decide::register_decider(
        "mem_dec_race",
        Arc::new(Held {
            inner: race_nli("mem-nli-race"),
            // `a` also holds its second call (the duplicate template)
            n: if me == "a" { 2 } else { 1 },
            calls: Default::default(),
            wait: Box::new(move |i| {
                if i == 0 {
                    std::fs::write(d2.join(format!("ready-{me2}")), b"").unwrap();
                    wait_for(&|| d2.join("ready-a").exists() && d2.join("ready-b").exists());
                } else {
                    wait_for(&|| d2.join("done-b").exists());
                }
            }),
        }),
        Family::Nano,
    )
    .unwrap();
    let db = DbInstance::new("sqlite", &path, "").unwrap();
    let r = db.remember(&text, RememberOpts { ns: ns("race"), ..Default::default() }).unwrap();
    if me == "b" {
        std::fs::write(dir.join("done-b"), b"").unwrap();
    }
    eprintln!("CHILD {me}: superseded {:?}", r.superseded);
}

/// The same race between two processes sharing one SQLite file: the
/// in-process gate cannot see across processes, so this exercises the
/// namespace version row checked inside the write transaction.
#[test]
fn concurrent_remembers_across_processes() {
    let base = std::env::temp_dir().join(format!("kleos-race-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let rdv = base.join("rdv");
    std::fs::create_dir_all(&rdv).unwrap();
    let path = base.join("mem.db");
    {
        let _guard = REGISTRY.read().unwrap_or_else(|e| e.into_inner());
        embed::register_embedder("mem_emb_race_parent", Arc::new(Words)).unwrap();
        let db = DbInstance::new("sqlite", &path, "").unwrap();
        db.memory_init(InitOpts {
            embedder: Some("mem_emb_race_parent".into()),
            decider: None,
            ..Default::default()
        })
        .unwrap();
        db.remember(BERLIN, RememberOpts { ns: ns("race"), at: Some(d(1)), ..Default::default() })
            .unwrap();
        // the children register the same embedder and decider under these names
        db.memory_init(InitOpts {
            embedder: None,
            decider: Some("mem_dec_race".into()),
            ..Default::default()
        })
        .unwrap();
        db.run_script(
            "?[key, value] <- [['embedder', 'mem_emb_race']] :put kleos_mem_config {key => value}",
            Default::default(),
            ScriptMutability::Mutable,
        )
        .unwrap();
        embed::unregister_embedder("mem_emb_race_parent");
    }
    let exe = std::env::current_exe().unwrap();
    let kids: Vec<_> = [("a", LISBON), ("b", PORTO)]
        .into_iter()
        .map(|(me, text)| {
            std::process::Command::new(&exe)
                .args(["race_child", "--exact", "--ignored", "--test-threads=1", "--nocapture"])
                .env("KLEOS_RACE_DB", &path)
                .env("KLEOS_RACE_TEXT", text)
                .env("KLEOS_RACE_DIR", &rdv)
                .env("KLEOS_RACE_ME", me)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap()
        })
        .collect();
    for k in kids {
        let out = k.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "child failed:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let db = DbInstance::new("sqlite", &path, "").unwrap();
    let cur = current(&db, &[BERLIN, LISBON, PORTO]);
    assert_eq!(cur.len(), 1, "exactly one current home across processes, got {cur:?}");
    drop(db);
    let _ = std::fs::remove_dir_all(&base);
}

// ------------------------------------------------------------------ claims

const MAX: &str = "I have a dog named Max.";

fn claim_ids(r: &kleos::memory::RememberResult) -> Vec<String> {
    r.claims.iter().map(|c| c.id.clone().unwrap()).collect()
}

fn current_texts(db: &DbInstance, n: &str) -> BTreeSet<String> {
    db.run_script(
        "?[text] := *kleos_mem{ns: $ns, text @ 'NOW'}",
        [("ns".to_string(), DataValue::from(n))].into_iter().collect(),
        ScriptMutability::Immutable,
    )
    .unwrap()
    .rows
    .into_iter()
    .map(|r| r[0].get_str().unwrap().to_string())
    .collect()
}

/// A multi-sentence text is stored as one memory per claim, with ids
/// derived from the statement id and the whole text as each claim's source;
/// recall returns the statement as context. One sentence, or `split:
/// 'none'`, is one memory with the plain statement id, as before claims.
#[test]
fn claims_split_store_and_recall_with_context() {
    let m = Models::new("claims", vec![], vec![]);
    let db = m.db();
    let text = "I live in Berlin. I have a dog named Max.\nWhere should I go on holiday?";
    let r = db.remember(text, at("u", d(1))).unwrap();
    assert_eq!(r.action, "added");
    let stmt = r.id.clone().unwrap();
    assert!(!stmt.contains('.'), "{stmt}");
    assert_eq!(claim_ids(&r), (1..=3).map(|i| format!("{stmt}.{i}")).collect::<Vec<_>>());
    let texts: Vec<&str> = r.claims.iter().map(|c| c.text.as_str()).collect();
    assert_eq!(texts, vec![BERLIN, MAX, "Where should I go on holiday?"]);
    assert_eq!(r.claims.iter().map(|c| c.decided).collect::<Vec<_>>(), vec![true, true, false]);
    assert_eq!(current_texts(&db, "u").len(), 3);
    // every claim's source row is the whole statement
    assert_eq!(
        count(&db, &format!("?[id] := *kleos_mem_source{{ns: 'u', id, stmt: '{stmt}', text: $t}}").replace("$t", &format!("{:?}", text))),
        3
    );
    let hit = db
        .recall("dog named Max", RecallOpts { ns: ns("u"), k: Some(1), ..Default::default() })
        .unwrap()
        .remove(0);
    // the list text is the statement; the claim is kept in `claim`
    assert_eq!((hit.text.as_str(), hit.claim.as_deref(), hit.context.as_deref()), (text, Some(MAX), Some(text)));
    assert_eq!(hit.memory_text(), MAX);
    // hit_text: 'claim' is the claim alone, as before
    let old = db
        .recall("dog named Max", RecallOpts { ns: ns("u"), k: Some(1), hit_text: Some("claim".into()), ..Default::default() })
        .unwrap()
        .remove(0);
    assert_eq!((old.text.as_str(), old.claim.as_deref(), old.context.as_deref()), (MAX, None, Some(text)));
    // a long statement: a window of text_chars around the claim
    let win = db
        .recall("dog named Max", RecallOpts { ns: ns("u"), k: Some(1), text_chars: Some(30), ..Default::default() })
        .unwrap()
        .remove(0);
    assert!(win.text.contains(MAX) && win.text.chars().count() <= 32, "{}", win.text);
    assert!(db
        .recall("x", RecallOpts { ns: ns("u"), hit_text: Some("all".into()), ..Default::default() })
        .is_err());
    assert_eq!(hit.id, format!("{stmt}.2"));

    // one sentence: the plain statement id, no claims, no context
    let c = db.remember(COFFEE, at("u", d(2))).unwrap();
    assert!(c.claims.is_empty());
    let cid = c.id.unwrap();
    assert!(!cid.contains('.'));
    let hit = db
        .recall("coffee black", RecallOpts { ns: ns("u"), ..Default::default() })
        .unwrap()
        .into_iter()
        .find(|h| h.id == cid)
        .unwrap();
    assert_eq!(hit.context, None);
    // split: none keeps the whole text as one memory
    let w = db
        .remember("I like jazz. I play the piano.", RememberOpts { ns: ns("w"), split: Some("none".into()), ..Default::default() })
        .unwrap();
    assert!(w.claims.is_empty());
    assert_eq!(current_texts(&db, "w"), BTreeSet::from(["I like jazz. I play the piano.".to_string()]));
    assert!(db
        .remember("x y z", RememberOpts { split: Some("sometimes".into()), ..Default::default() })
        .is_err());
    // CozoScript and JSON surfaces take the option and return the claims
    let rows = db
        .run_script(
            "::memory remember 'I drink green tea. I play chess on Sundays.' {ns: 'j', split: 'auto'}",
            Default::default(),
            ScriptMutability::Mutable,
        )
        .unwrap();
    assert!(rows.headers.contains(&"claims".to_string()), "{:?}", rows.headers);
    let j = db
        .memory_call("remember", serde_json::json!({"text": "I like tea. I play chess.", "ns": "j2", "split": "none"}))
        .unwrap();
    assert!(j.get("claims").is_none(), "{j}");
}

/// A claim supersedes only the claim it updates: the old statement's other
/// claims stay current. The similarity floor keeps a split statement's
/// unrelated candidates from the decider.
#[test]
fn claims_supersede_only_the_old_claim() {
    let m = Models::new("csup", vec![(BERLIN, LISBON)], vec![]);
    let db = m.db();
    let a = db.remember(&format!("{BERLIN} {MAX}"), at("u", d(1))).unwrap();
    let old = claim_ids(&a);
    // a short update against a long old message
    let b = db.remember(LISBON, at("u", d(10))).unwrap();
    assert_eq!(b.superseded, vec![old[0].clone()], "{b:?}");
    assert_eq!(current_texts(&db, "u"), BTreeSet::from([MAX.to_string(), LISBON.to_string()]));
    // the old claim keeps its history
    let then = recall(&db, "where do I live", "u", Some(d(5)));
    assert!(then.contains(&BERLIN.to_string()), "{then:?}");

    // a split update: only its first claim supersedes, and the decision,
    // the link and the result name that claim
    let db = m.db();
    let a = db.remember(&format!("{BERLIN} {MAX}"), at("u", d(1))).unwrap();
    let old = claim_ids(&a);
    let b = db
        .remember(
            &format!("{LISBON} {COFFEE}"),
            RememberOpts { ns: ns("u"), at: Some(d(10)), min_similarity: Some(-1.0), ..Default::default() },
        )
        .unwrap();
    let new = claim_ids(&b);
    assert_eq!(b.superseded, vec![old[0].clone()]);
    assert_eq!(b.claims[0].superseded, vec![old[0].clone()]);
    assert!(b.claims[1].superseded.is_empty());
    let applied: Vec<_> = b.decisions.iter().filter(|x| x.action == "superseded").collect();
    assert_eq!(applied.len(), 1);
    assert_eq!(applied[0].claim.as_deref(), Some(new[0].as_str()));
    assert!(applied[0].similarity.is_some());
    assert_eq!(
        count(&db, &format!("?[s] := *kleos_mem_link{{ns: 'u', src: s, rel: 'supersedes', dst: '{}'}}, s = '{}'", old[0], new[0])),
        1
    );
    assert_eq!(
        current_texts(&db, "u"),
        BTreeSet::from([MAX.to_string(), LISBON.to_string(), COFFEE.to_string()])
    );

    // default floor (0.6): the bag-of-words vectors of Berlin and Lisbon
    // are far apart, so the pair is never asked and nothing is superseded
    let db = m.db();
    db.remember(&format!("{BERLIN} {MAX}"), at("u", d(1))).unwrap();
    let b = db.remember(&format!("{LISBON} {COFFEE}"), at("u", d(10))).unwrap();
    assert!(b.superseded.is_empty(), "{b:?}");
    assert!(b.decisions.iter().all(|x| x.similarity.unwrap_or(1.0) >= 0.6), "{b:?}");
}

/// Questions, requests and short fragments are stored but take no part in
/// decisions, on either side; `split: 'none'` decides them as before.
#[test]
fn claims_questions_are_not_decided() {
    let m = Models::new("cq", vec![(BERLIN, LISBON)], vec![]);
    let db = m.db();
    let berlin = db.remember(BERLIN, at("u", d(1))).unwrap().id.unwrap();
    let q = db.remember("Do you know any good bars in Berlin?", at("u", d(2))).unwrap();
    assert_eq!(q.action, "added");
    assert!(q.decisions.is_empty(), "{q:?}");
    let qid = q.id.unwrap();
    let r = db
        .remember(
            "Can you recommend a bar in Berlin? Which district has the best nightlife? I live in Berlin.",
            RememberOpts { ns: ns("u"), at: Some(d(3)), min_similarity: Some(-1.0), ..Default::default() },
        )
        .unwrap();
    assert_eq!(r.claims.iter().map(|c| c.decided).collect::<Vec<_>>(), vec![false, false, true]);
    // the declarative claim repeats the old memory (exact rule); the others are stored
    assert_eq!(r.claims[2].action, "duplicate");
    assert_eq!(r.claims[2].duplicate_of.as_deref(), Some(berlin.as_str()));
    assert!(r.decisions.iter().all(|x| x.claim.as_deref() == Some(r.claims[2].id.as_deref().unwrap_or(berlin.as_str()))), "{r:?}");
    // the question is never a candidate
    let b = db.remember(LISBON, at("u", d(10))).unwrap();
    assert_eq!(b.superseded, vec![berlin]);
    assert!(b.decisions.iter().all(|x| x.candidate != qid), "{b:?}");
    // split: none decides a question like any statement
    let n = db
        .remember(
            "Do you know any good bars in Berlin?",
            RememberOpts { ns: ns("u"), at: Some(d(11)), split: Some("none".into()), ..Default::default() },
        )
        .unwrap();
    assert_eq!(n.action, "duplicate", "{n:?}");
}

/// `forget` (and `set`) of a statement id covers all its claims; a claim
/// id covers one claim.
#[test]
fn claims_forget_a_statement() {
    let m = Models::new("cforget", vec![], vec![]);
    let db = m.db();
    let a = db.remember(&format!("{BERLIN} {MAX} {COFFEE}"), at("u", d(1))).unwrap();
    let stmt = a.id.clone().unwrap();
    let ids = claim_ids(&a);
    db.remember("I play tennis on Saturdays.", at("u", d(2))).unwrap();
    let s = db
        .memory_set(SetOpts { ns: ns("u"), id: Some(stmt.clone()), importance: Some(0.9), ..Default::default() })
        .unwrap();
    assert_eq!(s["ids"].as_array().unwrap().len(), 3, "{s}");
    let f = db.forget(ForgetOpts { ns: ns("u"), id: Some(ids[1].clone()), ..Default::default() }).unwrap();
    assert_eq!(f.ids, vec![ids[1].clone()]);
    let f = db.forget(ForgetOpts { ns: ns("u"), id: Some(stmt.clone()), ..Default::default() }).unwrap();
    assert_eq!(f.ids, vec![ids[0].clone(), ids[2].clone()], "the claims still current");
    assert_eq!(current_texts(&db, "u"), BTreeSet::from(["I play tennis on Saturdays.".to_string()]));
    let e = db
        .forget(ForgetOpts { ns: ns("u"), id: Some(stmt.clone()), mode: Some("erase".into()), ..Default::default() })
        .unwrap();
    assert_eq!(e.ids, ids);
    assert_eq!(count(&db, &format!("?[id] := *kleos_mem{{ns: 'u', id}}, starts_with(id, '{stmt}')")), 0);
    assert_eq!(count(&db, &format!("?[id] := *kleos_mem_source{{ns: 'u', id, stmt: '{stmt}'}}")), 0);
}

/// Entities are linked to the claims that name them, and to every claim
/// when no claim does (the subject); the same-subject gate is per claim.
#[test]
fn claims_link_entities_per_claim() {
    const ANNA: &str = "My sister Anna lives in Munich.";
    const ANNA2: &str = "My sister Anna moved to Hamburg.";
    let m = Models::new("cent", vec![(BERLIN, ANNA2), (ANNA, ANNA2)], vec![]);
    let db = m.db();
    let ents = |names: &[&str], t| RememberOpts {
        ns: ns("u"),
        at: Some(t),
        entities: names.iter().map(|s| s.to_string()).collect(),
        min_similarity: Some(-1.0),
        ..Default::default()
    };
    let a = db.remember(&format!("{BERLIN} {ANNA}"), ents(&["user", "Anna"], d(1))).unwrap();
    let ids = claim_ids(&a);
    let by = |n: &str| -> BTreeSet<String> {
        db.recall("", RecallOpts { ns: ns("u"), entity: Some(n.into()), k: Some(10), ..Default::default() })
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect()
    };
    assert_eq!(by("Anna"), BTreeSet::from([ids[1].clone()]));
    assert_eq!(by("user"), BTreeSet::from([ids[0].clone(), ids[1].clone()]));
    // Anna's move supersedes Anna's claim, never the user's home (the
    // gate needs a shared entity; the user's claim does not mention Anna)
    let b = db.remember(&format!("{ANNA2} I like my coffee black."), ents(&["Anna"], d(5))).unwrap();
    assert_eq!(b.superseded, vec![ids[1].clone()], "{b:?}");
    assert!(b.gated.contains(&ids[0]), "{b:?}");
}

/// Two split statements planned at the same moment, each with a claim that
/// moves home: the later committer re-plans, sees the earlier claim and
/// supersedes it, and the claims about other things all stay current.
#[test]
fn concurrent_claim_remembers_never_leave_two_current() {
    let _guard = REGISTRY.read().unwrap_or_else(|e| e.into_inner());
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let b = barrier.clone();
    let (emb, dec) = ("mem_emb_crace", "mem_dec_crace");
    embed::register_embedder(emb, Arc::new(Words)).unwrap();
    decide::register_decider(
        dec,
        Arc::new(Held {
            inner: race_nli("mem-nli-crace"),
            n: 2,
            calls: Default::default(),
            wait: Box::new(move |_| {
                b.wait();
            }),
        }),
        Family::Nano,
    )
    .unwrap();
    let db = DbInstance::new("mem", "", "").unwrap();
    db.memory_init(InitOpts { embedder: Some(emb.into()), decider: Some(dec.into()), ..Default::default() })
        .unwrap();
    let opts = |t: Option<i64>| RememberOpts { ns: ns("race"), at: t, min_similarity: Some(-1.0), ..Default::default() };
    db.remember(&format!("{BERLIN} {MAX}"), opts(Some(d(1)))).unwrap();
    let results: Vec<_> = std::thread::scope(|sc| {
        let hs: Vec<_> = [format!("{LISBON} I play jazz piano."), format!("{PORTO} I drink green tea.")]
            .into_iter()
            .map(|t| {
                let db = &db;
                sc.spawn(move || db.remember(&t, opts(None)).unwrap())
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    embed::unregister_embedder(emb);
    decide::unregister_decider(dec);
    let cur = current(&db, &[BERLIN, LISBON, PORTO, MAX, "I play jazz piano.", "I drink green tea."]);
    assert_eq!(cur.len(), 4, "one home and the three other claims, got {cur:?}; {results:?}");
    assert_eq!(cur.iter().filter(|t| [BERLIN, LISBON, PORTO].contains(&t.as_str())).count(), 1, "{cur:?}");
    let claims: BTreeSet<String> = results.iter().flat_map(|r| r.claims.iter().filter_map(|c| c.id.clone())).collect();
    assert!(
        results.iter().any(|r| r.superseded.iter().any(|x| claims.contains(x))),
        "one move should supersede the other's claim: {results:?}"
    );
}

/// Recall ranks a question or request `question_weight` (default 0.05)
/// below an equally relevant statement; 0 turns it off.
#[test]
fn claims_questions_rank_below_statements() {
    let m = Models::new("cqw", vec![], vec![]);
    let db = m.db();
    db.remember("Where should I go on holiday? I go to Spain every summer.", at("u", d(1))).unwrap();
    let get = |w: Option<f64>| {
        db.recall(
            "where do I go on holiday",
            RecallOpts { ns: ns("u"), question_weight: w, record_access: Some(false), ..Default::default() },
        )
        .unwrap()
    };
    let (on, off) = (get(None), get(Some(0.0)));
    let score = |h: &[RecallHit], t: &str| h.iter().find(|x| x.memory_text() == t).unwrap().score;
    let q = "Where should I go on holiday?";
    let s = "I go to Spain every summer.";
    assert!((score(&off, q) - score(&on, q) - 0.05).abs() < 1e-6);
    assert!((score(&off, s) - score(&on, s)).abs() < 1e-6, "recency moves only with the clock");
    assert!(db
        .recall("x", RecallOpts { question_weight: Some(-1.0), ..Default::default() })
        .is_err());
}

// ------------------------------------------------------- recall: diversity, expansion, context, roles

fn src_at(ns_: &str, t: i64, source: &str) -> RememberOpts {
    RememberOpts {
        ns: ns(ns_),
        at: Some(t),
        source: Some(source.to_string()),
        ..Default::default()
    }
}

fn ropts(ns_: &str, k: usize) -> RecallOpts {
    RecallOpts {
        ns: ns(ns_),
        k: Some(k),
        record_access: Some(false),
        ..Default::default()
    }
}

/// The per-statement cap (default 1), the per-source cap, MMR near-duplicate
/// suppression, and `diversity: false` as the old top-k.
#[test]
fn recall_diversity_caps_and_mmr() {
    let m = Models::new("div", vec![], vec![]);
    let db = m.db();
    // one statement of three claims about hiking, and other statements
    let long = "I went hiking in the Alps last summer. The hiking trail near Zermatt was steep. \
                Hiking with my brother is the best.";
    let r = db.remember(long, src_at("u", d(1), "chat-1#1")).unwrap();
    assert_eq!(r.claims.len(), 3);
    db.remember("We planned a hiking trip to Norway for next year.", src_at("u", d(2), "chat-1#2")).unwrap();
    db.remember("Hiking boots need breaking in before a long hiking trip.", src_at("u", d(3), "chat-1#3")).unwrap();
    db.remember("My favourite hiking snack is trail mix.", src_at("u", d(4), "chat-2#1")).unwrap();
    db.remember("I bought a new tent for camping.", src_at("u", d(5), "chat-3#1")).unwrap();

    let q = "hiking trip";
    // diversity off: the old top-k, claims of one statement may stack
    let off = db.recall(q, RecallOpts { diversity: Some(false), ..ropts("u", 5) }).unwrap();
    let stmt = r.id.clone().unwrap();
    let n_off = off.iter().filter(|h| h.id.starts_with(&format!("{stmt}."))).count();
    assert!(n_off >= 2, "{:?}", texts(&off));
    // default: one claim per statement (the others only fill up to k)
    let on = db.recall(q, ropts("u", 4)).unwrap();
    assert_eq!(on.len(), 4);
    let n_on = on.iter().filter(|h| h.id.starts_with(&format!("{stmt}."))).count();
    assert_eq!(n_on, 1, "{:?}", texts(&on));
    // the same set as off, minus stacked claims: every statement is represented
    let stmts: BTreeSet<&str> = on.iter().map(|h| h.id.split('.').next().unwrap()).collect();
    assert_eq!(stmts.len(), 4, "{:?}", texts(&on));
    // caps are soft: asking for more than there are statements fills with claims
    assert_eq!(db.recall(q, ropts("u", 7)).unwrap().len(), 7);

    // per-source cap: chat-1 (statements 1-3) at most 1
    let src1 = db
        .recall(q, RecallOpts { max_per_source: Some(1), ..ropts("u", 3) })
        .unwrap();
    let groups: Vec<&str> = src1.iter().map(|h| h.source.as_deref().unwrap().split('#').next().unwrap()).collect();
    assert_eq!(groups.iter().collect::<BTreeSet<_>>().len(), 3, "{groups:?}");

    // MMR: an exact restatement in another chat is a near-duplicate
    let dup = "Hiking boots need breaking in before a long hiking trip!";
    db.remember(dup, RememberOpts { split: Some("none".into()), ..src_at("v", d(1), "a#1") }).unwrap();
    db.remember("Hiking boots really need breaking in before a long hiking trip.", src_at("v", d(2), "b#1")).unwrap();
    db.remember("A hiking trip in autumn needs warm layers.", src_at("v", d(3), "c#1")).unwrap();
    db.remember("I like swimming.", src_at("v", d(4), "d#1")).unwrap();
    let plain = db.recall("hiking boots trip", ropts("v", 2)).unwrap();
    assert!(plain.iter().all(|h| h.text.starts_with("Hiking boots")), "{:?}", texts(&plain));
    let mmr = db
        .recall("hiking boots trip", RecallOpts { mmr_weight: Some(1.0), ..ropts("v", 2) })
        .unwrap();
    assert!(mmr[0].text.starts_with("Hiking boots"), "{:?}", texts(&mmr));
    assert!(!mmr[1].text.starts_with("Hiking boots"), "{:?}", texts(&mmr));
    // deterministic
    let again = db
        .recall("hiking boots trip", RecallOpts { mmr_weight: Some(1.0), ..ropts("v", 2) })
        .unwrap();
    assert_eq!(texts(&mmr), texts(&again));
    assert!(db.recall(q, RecallOpts { mmr_weight: Some(-1.0), ..ropts("u", 2) }).is_err());
}

/// Entity-graph expansion: memories about the top hits' entities join the
/// pool at a reduced weight, merges followed; hub entities are not followed.
#[test]
fn recall_entity_expansion() {
    let m = Models::new("exp", vec![], vec![]);
    let db = m.db();
    db.remember("Anna adopted a greyhound called Pixel.", with_ents("u", d(1), &["Anna", "Pixel"])).unwrap();
    db.remember("She works as a nurse at the Charité.", with_ents("u", d(2), &["Anna"])).unwrap();
    db.remember("The weather was rainy all week.", with_ents("u", d(3), &[])).unwrap();
    for i in 0..5 {
        db.remember(&format!("Unrelated note number {i} about cooking pasta."), with_ents("u", d(4 + i), &[]))
            .unwrap();
    }
    let q = "what dog did Anna adopt";
    let base = db.recall(q, ropts("u", 2)).unwrap();
    assert!(base.iter().all(|h| !h.expanded));
    assert!(!base.iter().any(|h| h.text.contains("nurse")), "{:?}", texts(&base));
    let ex = db
        .recall(q, RecallOpts { expand: Some(true), expand_weight: Some(0.9), ..ropts("u", 2) })
        .unwrap();
    // the seed keeps the top; the memory about the same entity follows
    assert_eq!(ex[0].text, "Anna adopted a greyhound called Pixel.", "{:?}", texts(&ex));
    assert!(ex.iter().any(|h| h.text.contains("nurse") && h.expanded), "{:?}", texts(&ex));
    // a hub entity (mentioned by more than 25 memories) is not followed
    for i in 0..26 {
        db.remember(&format!("User fact {i}: likes colour {i}."), with_ents("h", d(1 + i), &["user"])).unwrap();
    }
    db.remember("I adopted a greyhound called Pixel.", with_ents("h", d(40), &["user"])).unwrap();
    let hub = db
        .recall("greyhound", RecallOpts { expand: Some(true), expand_weight: Some(1.0), ..ropts("h", 3) })
        .unwrap();
    assert!(hub.iter().all(|h| !h.expanded), "{:?}", texts(&hub));
}

/// The context block: grouped by entity, dated, the superseded value as
/// "previously", the statement a claim came from, the budget, determinism,
/// and the CozoScript / JSON surfaces.
#[test]
fn recall_context_current_previously_and_budget() {
    let m = Models::new("ctx", vec![(BERLIN, LISBON)], vec![]);
    let db = m.db();
    db.remember(BERLIN, with_ents("u", d(1), &["user"])).unwrap();
    db.remember(COFFEE, with_ents("u", d(2), &["user"])).unwrap();
    let b = db.remember(LISBON, with_ents("u", d(10), &["user"])).unwrap();
    assert_eq!(b.superseded.len(), 1);
    db.remember("My sister Anna lives in Munich. She is a nurse.", with_ents("u", d(12), &["Anna"])).unwrap();

    // k = 3: Lisbon, the coffee and one claim of Anna's statement
    let opts = || RecallOpts { budget: Some(400), ..ropts("u", 3) };
    let c = db.recall_context("where do we live?", opts()).unwrap();
    let t = &c.text;
    assert!(t.starts_with("Recalled memories"), "{t}");
    assert!(t.contains("## user\n") && t.contains("## Anna\n"), "{t}");
    assert!(t.contains("- [2026-01-11 (Sun)] I moved to Lisbon last month."), "{t}");
    assert!(t.contains("  previously: \"I live in Berlin.\" (from 2026-01-02 until 2026-01-11)"), "{t}");
    // the superseded memory itself is not a current item
    assert!(!t.contains("- [2026-01-02 (Fri)] I live in Berlin."), "{t}");
    // a claim shows the statement it came from
    assert!(t.contains("said in: \"My sister Anna lives in Munich. She is a nurse.\""), "{t}");
    assert!(c.tokens <= 400 && c.tokens > 0, "{c:?}");
    assert_eq!((c.ids.len(), c.omitted.len()), (3, 0), "{c:?}");
    // deterministic
    assert_eq!(db.recall_context("where do we live?", opts()).unwrap(), c);
    // a small budget keeps the best items and leaves the rest out
    let small = db.recall_context("where do we live?", RecallOpts { budget: Some(80), ..ropts("u", 3) }).unwrap();
    assert!(small.tokens <= 80, "{small:?}");
    assert!(!small.omitted.is_empty() && !small.ids.is_empty(), "{small:?}");
    // as_of: before the move, Berlin is current and nothing is "previously"
    let past = db
        .recall_context("where do I live?", RecallOpts { as_of: Some(d(5)), ..opts() })
        .unwrap();
    assert!(past.text.contains("I live in Berlin.") && !past.text.contains("  previously:"), "{}", past.text);
    // surfaces
    let j = db
        .memory_call("recall", serde_json::json!({"query": "where do we live?", "ns": "u", "k": 3, "format": "context", "budget": 400, "record_access": false}))
        .unwrap();
    assert_eq!(j["text"].as_str().unwrap(), c.text);
    let rows = db
        .run_script(
            "::memory recall 'where do we live?' {ns: 'u', k: 3, format: 'context', budget: 400}",
            Default::default(),
            ScriptMutability::Immutable,
        )
        .unwrap();
    let col = rows.headers.iter().position(|h| h == "text").unwrap();
    assert_eq!(rows.rows[0][col], DataValue::from(c.text.as_str()));
    assert!(db
        .memory_call("recall", serde_json::json!({"query": "x", "format": "table"}))
        .is_err());
}

/// Speaker roles: assistant memories are stored, recalled with their role at
/// a lower weight and filterable, never decided; `from` / `to` filter on
/// valid-from.
#[test]
fn remember_roles_and_time_filters() {
    let m = Models::new("role", vec![(BERLIN, LISBON)], vec![]);
    let db = m.db();
    let a = db.remember(BERLIN, at("u", d(1))).unwrap();
    // the assistant "saying" the update never supersedes the user's fact
    let asst = db
        .remember(LISBON, RememberOpts { role: Some("assistant".into()), ..at("u", d(2)) })
        .unwrap();
    assert!(asst.superseded.is_empty() && asst.decisions.is_empty(), "{asst:?}");
    // and the user's own update does not see the assistant memory as a candidate
    let u2 = db.remember("I moved to Lisbon last month!", at("u", d(3))).unwrap();
    assert!(u2.decisions.iter().all(|x| x.candidate != asst.id.clone().unwrap()), "{u2:?}");
    assert!(db.remember("x y z", RememberOpts { role: Some("robot".into()), ..at("u", d(4)) }).is_err());

    let hits = db.recall("Lisbon", ropts("u", 5)).unwrap();
    let ah = hits.iter().find(|h| h.id == asst.id.clone().unwrap()).unwrap();
    assert_eq!(ah.role.as_deref(), Some("assistant"));
    // the assistant weight: 0.05 lower than without it
    let no_w = db
        .recall("Lisbon", RecallOpts { assistant_weight: Some(0.0), ..ropts("u", 5) })
        .unwrap();
    let ah0 = no_w.iter().find(|h| h.id == ah.id).unwrap();
    assert!((ah0.score - ah.score - 0.05).abs() < 1e-6, "{} {}", ah0.score, ah.score);
    // role filter
    let only_a = db.recall("Lisbon", RecallOpts { role: Some("assistant".into()), ..ropts("u", 5) }).unwrap();
    assert_eq!(only_a.iter().map(|h| h.id.clone()).collect::<Vec<_>>(), vec![ah.id.clone()]);
    let only_u = db.recall("Lisbon", RecallOpts { role: Some("user".into()), ..ropts("u", 5) }).unwrap();
    assert!(only_u.iter().all(|h| h.role.is_none()) && !only_u.is_empty());
    // the context block marks the speaker
    let c = db.recall_context("Lisbon", ropts("u", 5)).unwrap();
    assert!(c.text.contains("(assistant) I moved to Lisbon last month."), "{}", c.text);
    // time filters on valid-from: [d(2), d(3)) is the assistant memory only
    let w = db
        .recall("Lisbon Berlin", RecallOpts { from: Some(d(2)), to: Some(d(3)), ..ropts("u", 5) })
        .unwrap();
    assert_eq!(w.iter().map(|h| h.id.clone()).collect::<Vec<_>>(), vec![ah.id.clone()]);
    let early = db
        .recall("Lisbon Berlin", RecallOpts { to: Some(d(2)), as_of: Some(d(1)), ..ropts("u", 5) })
        .unwrap();
    assert_eq!(early.iter().map(|h| h.id.clone()).collect::<Vec<_>>(), vec![a.id.clone().unwrap()]);
    // erase removes the role row
    db.forget(ForgetOpts { ns: ns("u"), id: asst.id.clone(), mode: Some("erase".into()), ..Default::default() })
        .unwrap();
    assert_eq!(count(&db, "?[id] := *kleos_mem_role{ns: 'u', id}"), 0);
}

// ------------------------------------------------------- event times (dates)

fn event_rows(db: &DbInstance, ns_: &str) -> Vec<(String, i64, i64, String, String, String)> {
    db.run_script(
        &format!("?[id, start, end, g, k, e] := *kleos_mem_event{{ns: '{ns_}', id, start, end, granularity: g, kind: k, expr: e}}"),
        Default::default(),
        ScriptMutability::Immutable,
    )
    .unwrap()
    .rows
    .into_iter()
    .map(|r| {
        (
            r[0].get_str().unwrap().to_string(),
            r[1].get_int().unwrap(),
            r[2].get_int().unwrap(),
            r[3].get_str().unwrap().to_string(),
            r[4].get_str().unwrap().to_string(),
            r[5].get_str().unwrap().to_string(),
        )
    })
    .collect()
}

/// Relative dates in a memory are resolved against when it was said and
/// stored per claim in `remember`'s transaction; list hits carry them, the
/// context block shows them, `event_from` / `event_to` filter on them (else
/// on valid-from), the question-date boost, erase, and the v4 migration.
#[test]
fn event_times_stored_shown_filtered_and_erased() {
    let m = Models::new("evt", vec![], vec![]);
    let db = m.db();
    // T0 = 2026-01-01 (Thu); d(7) = 2026-01-08 (Thu)
    let r = db
        .remember("I went to the support group yesterday. Several members told inspiring stories about their families.", at("u", d(7)))
        .unwrap();
    let stmt = r.id.clone().unwrap();
    let c1 = format!("{stmt}.1");
    let c2 = format!("{stmt}.2");
    let cat = db.remember("I adopted a cat last summer.", at("u", d(30))).unwrap().id.unwrap();
    let tennis = db.remember("I play tennis with Anna.", at("u", d(40))).unwrap().id.unwrap();
    let ev = event_rows(&db, "u");
    assert_eq!(ev.len(), 2, "{ev:?}");
    assert!(ev.contains(&(c1.clone(), d(6), d(7), "day".into(), "point".into(), "yesterday".into())), "{ev:?}");
    // last summer, said on 2026-01-31: June to August 2025
    let summer = (1_748_736_000_000_000i64, 1_756_684_800_000_000i64); // 2025-06-01, 2025-09-01
    assert!(ev.contains(&(cat.clone(), summer.0, summer.1, "season".into(), "point".into(), "last summer".into())), "{ev:?}");

    // list hits carry their own events
    let h = db.recall("support group", ropts("u", 3)).unwrap();
    let g = h.iter().find(|x| x.id == c1).expect("the claim");
    assert_eq!(g.events.len(), 1);
    assert_eq!((g.events[0].expr.as_str(), g.events[0].start, g.events[0].granularity.as_str()), ("yesterday", d(6), "day"));
    assert!(h.iter().find(|x| x.id == tennis).map_or(true, |x| x.events.is_empty()));

    // the context block shows the resolved date inline
    let c = db.recall_context("support group", ropts("u", 3)).unwrap();
    assert!(
        c.text.contains("I went to the support group yesterday. (≈ 2026-01-07 (Wed), \"yesterday\")"),
        "{}",
        c.text
    );
    // a claim without a date of its own: `said in` shows its statement
    // without date notes (they made the reader pick older values)
    let p = db.recall_context("inspiring stories families", RecallOpts { k: Some(1), ..ropts("u", 1) }).unwrap();
    assert!(p.ids == vec![c2.clone()], "{p:?}");
    assert!(
        p.text.contains("said in: \"I went to the support group yesterday. Several members told inspiring stories about their families.\"\n")
            || p.text.ends_with("said in: \"I went to the support group yesterday. Several members told inspiring stories about their families.\""),
        "{}",
        p.text
    );
    assert!(!p.text.contains('≈'), "{}", p.text);

    // event filters: overlap with the event time; no event -> valid-from;
    // a claim without one of its own -> its statement's
    let ids = |o: RecallOpts| -> BTreeSet<String> {
        db.recall("support group cat tennis stories", o).unwrap().into_iter().map(|h| h.id).collect()
    };
    assert_eq!(
        ids(RecallOpts { event_from: Some(summer.0), event_to: Some(summer.1), ..ropts("u", 10) }),
        BTreeSet::from([cat.clone()])
    );
    assert_eq!(
        ids(RecallOpts { event_from: Some(d(6)), event_to: Some(d(7)), ..ropts("u", 10) }),
        BTreeSet::from([c1.clone(), c2.clone()])
    );
    assert_eq!(
        ids(RecallOpts { event_from: Some(d(39)), event_to: Some(d(41)), ..ropts("u", 10) }),
        BTreeSet::from([tennis.clone()])
    );
    // the said-on day of the support group is not its event time
    assert!(ids(RecallOpts { event_from: Some(d(7)), event_to: Some(d(8)), ..ropts("u", 10) }).is_empty());

    // the question-date boost (default 0.03; 0 is off): memories whose
    // event time overlaps a date in the query (resolved at query_at) gain it
    let q = "what did I do yesterday";
    let score = |w: Option<f64>, id: &str| -> f64 {
        db.recall(q, RecallOpts { event_weight: w, query_at: Some(d(7)), ..ropts("u", 10) })
            .unwrap()
            .into_iter()
            .find(|h| h.id == id)
            .unwrap()
            .score
    };
    assert!((score(Some(0.5), &c1) - score(Some(0.0), &c1) - 0.5).abs() < 1e-6);
    assert!((score(None, &c1) - score(Some(0.0), &c1) - 0.03).abs() < 1e-6);
    assert!((score(Some(0.5), &cat) - score(Some(0.0), &cat)).abs() < 1e-6);
    assert!(db.recall(q, RecallOpts { event_weight: Some(-1.0), ..ropts("u", 2) }).is_err());

    // CozoScript and JSON surfaces return the events
    let j = db
        .memory_call("recall", serde_json::json!({"query": "support group", "ns": "u", "k": 1, "record_access": false}))
        .unwrap();
    assert_eq!(j[0]["events"][0]["expr"], "yesterday", "{j}");

    // erase removes the statement's event rows
    db.forget(ForgetOpts { ns: ns("u"), id: Some(stmt.clone()), mode: Some("erase".into()), ..Default::default() })
        .unwrap();
    let ev = event_rows(&db, "u");
    assert!(ev.iter().all(|e| !e.0.starts_with(&stmt)), "{ev:?}");
    assert_eq!(ev.len(), 1);

    // a v4 store gains the relation on first use, backfilled from the texts
    db.run_script("::remove kleos_mem_event", Default::default(), ScriptMutability::Mutable).unwrap();
    db.run_script(
        "?[key, value] <- [['version', '4']] :put kleos_mem_config {key => value}",
        Default::default(),
        ScriptMutability::Mutable,
    )
    .unwrap();
    let _ = db.recall("cat", ropts("u", 1)).unwrap();
    let ev = event_rows(&db, "u");
    assert_eq!(ev, vec![(cat.clone(), summer.0, summer.1, "season".into(), "point".into(), "last summer".into())]);
}

/// The default list text of a claim is its statement, so a reader that
/// shows only `text` sees the whole message once per statement.
#[test]
fn list_text_is_the_statement_by_default() {
    let m = Models::new("ltxt", vec![], vec![]);
    let db = m.db();
    let msg = "Caroline: I went to a LGBTQ support group. It was so powerful. The stories inspired me.";
    db.remember(msg, at("u", d(1))).unwrap();
    db.remember("Melanie: I painted a sunrise.", at("u", d(2))).unwrap();
    let h = db.recall("support group stories", ropts("u", 2)).unwrap();
    let mine: Vec<_> = h.iter().filter(|x| x.claim.is_some()).collect();
    assert_eq!(mine.len(), 1, "{h:?}");
    assert_eq!(mine[0].text, msg);
    assert!(msg.contains(mine[0].claim.as_deref().unwrap()));
    // a one-sentence memory is unchanged and has no claim
    let p = h.iter().find(|x| x.text.starts_with("Melanie")).unwrap();
    assert!(p.claim.is_none() && p.context.is_none());
}
