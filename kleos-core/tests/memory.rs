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
use kleos::memory::{ForgetOpts, InitOpts, LinkOpts, RecallOpts, ReflectOpts, RememberOpts};
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
    .map(|h| h.text)
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
                sc.spawn(move || db.remember(t, RememberOpts { ns: ns("race"), ..Default::default() }).unwrap())
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
