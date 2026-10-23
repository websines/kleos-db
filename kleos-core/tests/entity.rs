/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Entity resolution (docs/specs/entity-resolution.md) against a fake
//! decider and a fake embedder: the rule tier, candidate sources, Auto
//! merges, Tentative links, no transitive merges, split/undo, history, the
//! `::entity` ops, and the no-ML degrade. The decider and embedder
//! registries are process-wide: every test uses its own names.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use kleos::entity::{EntityConfig, Mention, Outcome, ResolveOptions};
use kleos::{
    register_decider, register_embedder, Band, DataValue, DbInstance, DecideAnswer, DecideRequest, Decider,
    Embedder, Family, NamedRows, ScriptMutability,
};

/// P(same) from the two rendered mentions.
type Judge = dyn Fn(&str, &str) -> f64 + Send + Sync;

struct FakeDecider {
    id: String,
    judge: Box<Judge>,
    pairs: AtomicUsize,
}

impl Decider for FakeDecider {
    fn model_id(&self) -> String {
        self.id.clone()
    }
    fn max_batch(&self) -> usize {
        8
    }
    fn decide(&self, reqs: &[DecideRequest], _with_features: bool) -> miette::Result<Vec<DecideAnswer>> {
        Ok(reqs
            .iter()
            .map(|r| {
                self.pairs.fetch_add(1, Ordering::SeqCst);
                let s = r.state.strip_prefix("Mention A: ").unwrap();
                let (a, b) = s.split_once("\nMention B: ").unwrap();
                let p = (self.judge)(a, b).clamp(1e-4, 1.0 - 1e-4);
                DecideAnswer {
                    // same / different / unclear
                    logits: vec![p.ln() as f32, (1.0 - p).ln() as f32, -30.0],
                    features: vec![],
                }
            })
            .collect())
    }
}

fn fake_decider(name: &str, judge: impl Fn(&str, &str) -> f64 + Send + Sync + 'static) -> Arc<FakeDecider> {
    let d = Arc::new(FakeDecider {
        id: format!("fake-{name}"),
        judge: Box::new(judge),
        pairs: AtomicUsize::new(0),
    });
    register_decider(name, d.clone(), Family::Nano).unwrap();
    d
}

/// Embeds by concept words: texts naming the same concept get the same vector.
struct ConceptEmbedder;

impl Embedder for ConceptEmbedder {
    fn embed(&self, texts: &[&str]) -> miette::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let t = t.to_lowercase();
                let f = |w: &str| if t.contains(w) { 1.0 } else { 0.0 };
                let v = vec![f("peter"), f("bank"), f("river"), 0.05];
                let n = v.iter().map(|x: &f32| x * x).sum::<f32>().sqrt();
                v.into_iter().map(|x| x / n).collect()
            })
            .collect())
    }
    fn dim(&self) -> usize {
        4
    }
}

fn mem() -> DbInstance {
    DbInstance::new("mem", "", "").unwrap()
}

fn run(db: &DbInstance, s: &str) -> NamedRows {
    db.run_script(s, Default::default(), ScriptMutability::Mutable)
        .unwrap_or_else(|e| panic!("{s}: {e:?}"))
}

fn s(v: &str) -> DataValue {
    DataValue::from(v)
}

/// A person/org relation with LSH (character trigrams) and FTS on the name.
fn people(db: &DbInstance) {
    run(
        db,
        ":create person {id: String => name: String, type: String? default null, aliases: [String]? default null, description: String? default null, emb: <F32; 4>? default null}",
    );
    run(
        db,
        "::lsh create person:lsh {extractor: name, tokenizer: NGram(3, 3, false), filters: [Lowercase], n_perm: 64, target_threshold: 0.3, n_gram: 1}",
    );
    run(
        db,
        "::fts create person:fts {extractor: name, tokenizer: Simple, filters: [Lowercase]}",
    );
    run(
        db,
        r#"?[id, name, type, description] <- [
            ['p1', 'Alice Smith', 'person', 'engineering manager at Acme'],
            ['p2', 'Robert Jones', 'person', 'neighbour'],
            ['o1', 'Microsoft', 'org', null],
            ['o2', 'International Business Machines', 'org', null],
            ['c1', 'Paris', 'place', 'capital of France']
        ] :put person {id => name, type, description}"#,
    );
}

fn cfg(decider: Option<&str>) -> EntityConfig {
    let mut c = EntityConfig::new("person");
    c.type_col = Some("type".into());
    c.aliases = Some("aliases".into());
    c.description = Some("description".into());
    c.lsh = Some("lsh".into());
    c.fts = Some("fts".into());
    c.decider = decider.map(|d| d.to_string());
    c
}

fn dry(db: &DbInstance, c: &EntityConfig, ms: &[Mention]) -> Vec<kleos::Resolution> {
    db.resolve_entities(c, ms, ResolveOptions::default()).unwrap()
}

#[test]
fn rule_tier_without_ml() {
    let db = mem();
    people(&db);
    let c = cfg(None);
    let r = dry(
        &db,
        &c,
        &[
            Mention::new("alice  smith").with_type("person"),
            Mention::new("Dr. Alice Smith").with_type("person"),
            Mention::new("A. Smith").with_type("person"),
            Mention::new("Bob Jones").with_type("person"),
            Mention::new("IBM").with_type("org"),
            Mention::new("Zed Quux").with_type("person"),
            Mention::new("Alice Smith").with_type("place"),
        ],
    );
    assert_eq!(r[0].outcome, Outcome::Matched);
    assert_eq!(r[0].entity, Some(s("p1")));
    assert_eq!(r[0].decided_by, "rule|exact");
    assert_eq!(r[1].outcome, Outcome::Matched, "honorific stripped");
    // initials and nicknames are candidates, not merges, without a model
    assert_eq!(r[2].outcome, Outcome::Possible);
    assert_eq!(r[2].entity, Some(s("p1")));
    assert_eq!(r[2].candidates[0].rule.as_deref(), Some("initial"));
    assert_eq!(r[3].outcome, Outcome::Possible);
    assert_eq!(r[3].candidates[0].rule.as_deref(), Some("nickname"));
    assert_eq!(r[4].outcome, Outcome::Possible);
    assert_eq!(r[4].candidates[0].rule.as_deref(), Some("acronym"));
    assert_eq!(r[5].outcome, Outcome::New);
    assert_eq!(r[6].outcome, Outcome::New, "types must be compatible");
    // nothing was written
    assert!(db.entity_history("person", None).unwrap().is_empty());
}

#[test]
fn candidates_from_lsh_and_fts() {
    let db = mem();
    people(&db);
    let c = cfg(None);
    let r = dry(&db, &c, &[Mention::new("Microsfot").with_type("org")]);
    let cand = r[0].candidates.iter().find(|c| c.entity == s("o1")).expect("typo found");
    assert!(cand.sources.iter().any(|x| x == "lsh"), "{:?}", cand.sources);
    // FTS finds a shared token the rules do not cover
    let r = dry(&db, &c, &[Mention::new("Business Machines Research").with_type("org")]);
    let cand = r[0].candidates.iter().find(|c| c.entity == s("o2")).expect("fts hit");
    assert!(cand.sources.iter().any(|x| x == "fts"), "{:?}", cand.sources);
}

#[test]
fn vector_candidates_and_embedding_on_commit() {
    let db = mem();
    people(&db);
    register_embedder("ent-concepts", Arc::new(ConceptEmbedder)).unwrap();
    run(&db, "?[id, name, type, description] <- [['x1', 'Initech', 'org', 'the company Peter works at']] :put person {id => name, type, description}");
    run(&db, "?[id, emb] := *person{id, name, type, description}, emb = embed('ent-concepts', concat(name, ' (', type, '). ', if(is_null(description), '', description))) :update person {id => emb}");
    run(&db, "::hnsw create person:vec {dim: 4, m: 16, dtype: F32, fields: [emb], distance: Cosine, ef_construction: 50}");
    let mut c = cfg(None);
    c.embedding = Some("emb".into());
    c.hnsw = Some("vec".into());
    c.embedder = Some("ent-concepts".into());
    let r = dry(&db, &c, &[Mention::new("Peter's employer").with_type("org")]);
    let cand = r[0].candidates.iter().find(|c| c.entity == s("x1")).expect("vector hit");
    assert!(cand.sources.iter().any(|x| x == "vector"), "{:?}", cand.sources);
    // commit writes the new row with its embedding
    let r = db
        .resolve_entities(&c, &[Mention::new("River Bank Co-op").with_type("org").with_id("n1")], ResolveOptions { commit: true })
        .unwrap();
    assert_eq!(r[0].id, Some(s("n1")));
    let rows = run(&db, "?[e] := *person{id: 'n1', emb: e}");
    assert!(matches!(rows.rows[0][0], DataValue::Vec(_)));
}

#[test]
fn auto_merges_with_history_and_canonical() {
    let db = mem();
    people(&db);
    let d = fake_decider("ent-auto", |a, b| {
        let both = |w: &str| a.contains(w) && b.contains(w);
        if both("Acme") || (a.contains("manager") && b.contains("manager")) {
            0.97
        } else {
            0.05
        }
    });
    let c = cfg(Some("ent-auto"));
    let r = db
        .resolve_entities(
            &c,
            &[
                Mention::new("Alice (my manager)").with_type("person").with_context("Alice (my manager) approved the budget"),
                Mention::new("A. Smith").with_type("person").with_description("works at Acme"),
                Mention::new("Alice Smith").with_type("person").with_description("my dentist"),
            ],
            ResolveOptions { commit: true },
        )
        .unwrap();
    assert!(d.pairs.load(Ordering::SeqCst) > 0);
    assert_eq!(r[0].outcome, Outcome::Matched, "{:?}", r[0]);
    assert_eq!(r[0].entity, Some(s("p1")));
    assert_eq!(r[0].band, Band::Auto);
    assert!(r[0].decided_by.starts_with("fake-ent-auto|same_entity@v2|"), "{}", r[0].decided_by);
    assert_eq!(r[1].outcome, Outcome::Matched);
    // exact name, but both sides have context and the model says different
    assert_eq!(r[2].outcome, Outcome::New, "{:?}", r[2]);
    let m0 = r[0].id.clone().unwrap();
    assert_eq!(db.canonical_entity("person", &m0).unwrap(), s("p1"));
    let h = db.entity_history("person", Some(&s("p1"))).unwrap();
    assert_eq!(h.len(), 2);
    assert!(h.iter().all(|e| e.action == "merge" && e.into == s("p1")));
    assert!(h[0].decided_by.contains("same_entity@v2"));
    // the canonical view in plain Datalog (the spec's rule)
    let rows = run(
        &db,
        r"merged[f, i] := *kleos_entity_merge{relation: 'person', from: f, into: i @ 'NOW'}
          canon[x, c] := *person{id: x}, not merged[x, _], c = x
          canon[x, c] := merged[x, m], canon[m, c]
          ?[x] := canon[x, 'p1']",
    );
    assert_eq!(rows.rows.len(), 3, "p1 and the two merged mentions");
}

#[test]
fn tentative_links_are_pending_until_reviewed() {
    let db = mem();
    people(&db);
    fake_decider("ent-tent", |_, _| 0.7);
    let c = cfg(Some("ent-tent"));
    let r = db
        .resolve_entities(&c, &[Mention::new("A. Smith").with_type("person").with_id("m1")], ResolveOptions { commit: true })
        .unwrap();
    assert_eq!(r[0].outcome, Outcome::Possible);
    assert_eq!(r[0].band, Band::Tentative);
    assert_eq!(db.canonical_entity("person", &s("m1")).unwrap(), s("m1"), "not merged");
    let p = run(&db, "::entity pending person");
    assert_eq!(p.rows.len(), 1);
    assert_eq!(p.rows[0][0], s("m1"));
    assert_eq!(p.rows[0][1], s("p1"));
    // accepting the review merges and clears the pending link
    run(&db, "::entity merge person 'm1' 'p1'");
    assert_eq!(db.canonical_entity("person", &s("m1")).unwrap(), s("p1"));
    assert!(run(&db, "::entity pending person").rows.is_empty());
    // a rejected pair is not proposed again
    run(&db, "::entity reject person 'o1' 'o2'");
    let r = db
        .resolve_entities(&c, &[Mention::new("IBM").with_type("org").with_id("o1")], ResolveOptions::default())
        .unwrap();
    assert!(r[0].candidates.iter().all(|c| c.entity != s("o2")));
}

#[test]
fn never_merges_transitively() {
    let db = mem();
    people(&db);
    run(
        &db,
        r"?[id, name, type, description] <- [
            ['k1', 'Alex Kim', 'person', 'engineer'],
            ['k2', 'Alexandra Kim', 'person', 'doctor']
        ] :put person {id => name, type, description}",
    );
    // the bridging mention looks like both; the two entities are different
    fake_decider("ent-trans", |a, b| {
        let bridge = a.starts_with("A. Kim") || b.starts_with("A. Kim");
        if bridge {
            if a.contains("engineer") || b.contains("engineer") {
                0.98
            } else {
                0.95
            }
        } else if (a.contains("doctor") && b.contains("doctor")) || (a.contains("engineer") && b.contains("engineer")) {
            0.97
        } else {
            0.02
        }
    });
    let c = cfg(Some("ent-trans"));
    let r = db
        .resolve_entities(&c, &[Mention::new("A. Kim").with_type("person").with_id("b1")], ResolveOptions { commit: true })
        .unwrap();
    assert_eq!(r[0].outcome, Outcome::Matched);
    assert_eq!(r[0].entity, Some(s("k1")), "best auto candidate");
    assert!(r[0].reason.as_deref().unwrap().contains("2 candidates in the auto band"));
    // k1 and k2 stay apart; the second candidate is only a link
    assert_eq!(db.canonical_entity("person", &s("k2")).unwrap(), s("k2"));
    assert_eq!(db.canonical_entity("person", &s("k1")).unwrap(), s("k1"));
    let p = run(&db, "::entity pending person");
    assert_eq!(p.rows.len(), 1);
    assert_eq!((p.rows[0][0].clone(), p.rows[0][1].clone()), (s("b1"), s("k2")));
    // a later mention like k2 is decided against k2 on its own, not via b1
    let r = dry(&db, &c, &[Mention::new("Alexandra Kim").with_type("person").with_description("doctor")]);
    assert_eq!(r[0].entity, Some(s("k2")));
}

#[test]
fn split_restores_and_history_is_kept() {
    let db = mem();
    people(&db);
    run(&db, "?[id, name, type] <- [['m1', 'Al Smith', 'person'], ['m2', 'Smithy', 'person']] :put person {id => name, type}");
    db.merge_entities("person", &s("m2"), &s("m1"), "test").unwrap();
    db.merge_entities("person", &s("m1"), &s("p1"), "test").unwrap();
    assert_eq!(db.canonical_entity("person", &s("m2")).unwrap(), s("p1"), "chains follow");
    assert!(db.merge_entities("person", &s("p1"), &s("m2"), "x").is_err(), "no cycles");
    let before = run(&db, "?[t] := *kleos_entity_merge{relation: 'person', from: 'm1', vt}, t = to_int(vt)");
    let t_merged = before.rows[0][0].get_int().unwrap();
    let out = run(&db, "::entity split person 'm1'");
    assert_eq!(out.rows[0][1], s("p1"));
    assert_eq!(db.canonical_entity("person", &s("m1")).unwrap(), s("m1"));
    assert_eq!(db.canonical_entity("person", &s("m2")).unwrap(), s("m1"), "m2 stays with m1");
    assert_eq!(run(&db, "::entity split person 'm1'").rows[0][0], s("NOT_MERGED"));
    // time travel: at the merge time m1 was merged into p1
    let at = run(
        &db,
        &format!("?[i] := *kleos_entity_merge{{relation: 'person', from: 'm1', into: i @ {t_merged}}}"),
    );
    assert_eq!(at.rows[0][0], s("p1"));
    let h = run(&db, "::entity history person 'm1'");
    let actions: Vec<_> = h.rows.iter().map(|r| r[2].get_str().unwrap().to_string()).collect();
    assert_eq!(actions, vec!["merge", "merge", "split"], "{:?}", h.rows);
    // merging again after a split works and adds to the history
    db.merge_entities("person", &s("m1"), &s("p1"), "again").unwrap();
    assert_eq!(db.canonical_entity("person", &s("m2")).unwrap(), s("p1"));
    assert_eq!(db.entity_history("person", Some(&s("m1"))).unwrap().len(), 4);
}

#[test]
fn degrades_without_ml() {
    let db = mem();
    people(&db);
    let mut c = cfg(Some("no-such-decider"));
    c.embedder = Some("no-such-embedder".into());
    c.hnsw = Some("missing-but-unused".into());
    let r = db
        .resolve_entities(
            &c,
            &[
                Mention::new("Alice Smith").with_type("person").with_context("my manager"),
                Mention::new("Robert Jonse").with_type("person"),
            ],
            ResolveOptions { commit: true },
        )
        .unwrap();
    assert_eq!(r[0].outcome, Outcome::Matched, "exact name decides without a model");
    assert_eq!(r[0].decided_by, "rule|exact");
    assert_ne!(r[1].outcome, Outcome::Matched, "a typo is never auto-merged by rules");
    // a disabled decider degrades the same way
    fake_decider("ent-off", |_, _| 0.99);
    kleos::disable_decider("ent-off").unwrap();
    let r = dry(&db, &cfg(Some("ent-off")), &[Mention::new("A. Smith").with_type("person")]);
    assert_eq!(r[0].outcome, Outcome::Possible);
    assert!(r[0].decided_by.starts_with("rule|"));
}

#[test]
fn script_ops() {
    let db = mem();
    people(&db);
    let r = run(
        &db,
        r#"::entity resolve person ['Alice Smith', ['A. Smith', 'person', 'my manager'], parse_json('{"name": "Bob Jones", "type": "person", "id": "q1"}')] {type: 'type', description: 'description', aliases: 'aliases', lsh: 'lsh', fts: 'fts', commit: true}"#,
    );
    assert_eq!(r.headers[..4], ["mention", "name", "outcome", "entity"]);
    let outcomes: Vec<_> = r.rows.iter().map(|x| x[2].get_str().unwrap().to_string()).collect();
    assert_eq!(outcomes, vec!["match", "possible", "possible"]);
    assert_eq!(r.rows[2][4], s("q1"));
    assert_eq!(run(&db, "::entity pending person").rows.len(), 2);
    assert_eq!(run(&db, "::entity history person").rows.len(), 1);
    // read-only scripts may resolve but not write
    let ro = db.run_script(
        "::entity resolve person ['Alice Smith'] {commit: true}",
        Default::default(),
        ScriptMutability::Immutable,
    );
    assert!(ro.is_err());
    let ok = db
        .run_script("::entity resolve person ['Alice Smith']", Default::default(), ScriptMutability::Immutable)
        .unwrap();
    assert_eq!(ok.rows[0][2], s("match"));
    // unknown options are errors
    assert!(db
        .run_script("::entity resolve person ['x'] {colour: 'red'}", Default::default(), ScriptMutability::Mutable)
        .is_err());
}

#[test]
fn batches_decisions_in_dry_runs() {
    let db = mem();
    people(&db);
    let d = fake_decider("ent-batch", |_, _| 0.1);
    let c = cfg(Some("ent-batch"));
    let ms: Vec<Mention> = ["A. Smith", "Al Smith", "Bob Jones", "Robbie Jones"]
        .iter()
        .map(|n| Mention::new(*n).with_type("person"))
        .collect();
    let r = dry(&db, &c, &ms);
    assert!(r.iter().all(|x| x.outcome == Outcome::New));
    // symmetric template: two orders per pair, one call per max_batch
    assert_eq!(d.pairs.load(Ordering::SeqCst) % 2, 0);
    assert!(d.pairs.load(Ordering::SeqCst) >= 8);
}

/// Real models: decider-nano for same_entity and potion-base-8M for
/// vectors. Run with
/// `cargo +1.99.0 test -p kleos --features infer-metal --test entity -- --include-ignored`.
#[test]
#[ignore]
fn real_models() {
    let db = mem();
    people(&db);
    db.enable_decider("nano-ent", kleos::DeciderSpec::model("decider-nano")).unwrap();
    let c = cfg(Some("nano-ent"));
    let r = dry(
        &db,
        &c,
        &[
            Mention::new("Alice (my manager)").with_type("person").with_context("Alice, my manager at Acme, approved the budget"),
            Mention::new("Bob Jones").with_type("person").with_description("my neighbour"),
            Mention::new("Alice Smith").with_type("person").with_description("a pop singer from Sweden"),
            Mention::new("Paris Hilton").with_type("person"),
        ],
    );
    for x in &r {
        eprintln!("{} -> {:?} {:?} p={:.3} {}", x.name, x.outcome, x.entity, x.p, x.decided_by);
    }
    assert_ne!(r[3].entity, Some(s("c1")), "a person never matches a place");
    assert!(r.iter().all(|x| x.outcome != Outcome::Matched || x.decided_by.contains("same_entity") || x.decided_by.starts_with("rule|")));
}
