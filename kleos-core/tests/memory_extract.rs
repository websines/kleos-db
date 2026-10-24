/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Automatic entities in `remember` (docs/specs/memory-api.md, "Automatic
//! entities"), with a table-driven fake extractor, embedder and decider.
//! The real GLiNER extractor is exercised by `memory_extract_gliner` below
//! (ignored by default; needs the checkpoint in `KLEOS_GLINER_TEST_MODEL`).

use std::sync::{Arc, Mutex};

use kleos::decide::{self};
use kleos::embed::{self, Embedder};
use kleos::extract::{self, EntityExtractor, ExtractSpan};
use kleos::memory::{ForgetOpts, InitOpts, RememberOpts};
use kleos::{DbInstance, DecideAnswer, DecideRequest, Decider, Family, ScriptMutability};

static SERIAL: Mutex<()> = Mutex::new(());

const DAY: i64 = 86_400_000_000;
const T0: i64 = 1_767_225_600_000_000;

fn d(n: i64) -> i64 {
    T0 + n * DAY
}

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

/// NLI decider from a table: `contra` (old, new) pairs contradict.
struct Nli {
    contra: Vec<(&'static str, &'static str)>,
}

impl Decider for Nli {
    fn model_id(&self) -> String {
        "extract-nli".into()
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
                let logits = if self.contra.iter().any(|&(x, y)| a.contains(x) && b.contains(y)) {
                    vec![0.0, 0.0, 12.0]
                } else {
                    vec![0.0, 6.0, 0.0]
                };
                DecideAnswer { logits, features: vec![] }
            })
            .collect())
    }
}

/// Finds each listed span (case-insensitively) wherever it occurs.
struct Table(Vec<(&'static str, &'static str, f32)>);

impl EntityExtractor for Table {
    fn model_id(&self) -> String {
        "table".into()
    }
    fn version(&self) -> String {
        "table:v1".into()
    }
    fn labels(&self) -> Vec<String> {
        ["person", "location", "organization", "relative"].iter().map(|s| s.to_string()).collect()
    }
    fn extract(&self, texts: &[&str]) -> miette::Result<Vec<Vec<ExtractSpan>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let lt = t.to_lowercase();
                let mut out = vec![];
                for (label, span, score) in &self.0 {
                    let ls = span.to_lowercase();
                    let mut from = 0;
                    while let Some(o) = lt[from..].find(&ls) {
                        let a = from + o;
                        out.push(ExtractSpan {
                            label: label.to_string(),
                            start: a,
                            end: a + span.len(),
                            text: t[a..a + span.len()].to_string(),
                            score: *score,
                        });
                        from = a + span.len();
                    }
                }
                out
            })
            .collect())
    }
}

const PARIS: &str = "I live in Paris.";
const SISTER_BERLIN: &str = "My sister moved to Berlin.";
const ME_BERLIN: &str = "I moved to Berlin.";

fn spans() -> Vec<(&'static str, &'static str, f32)> {
    vec![
        ("location", "Paris", 0.97),
        ("location", "Berlin", 0.98),
        ("location", "Lisbon", 0.98),
        ("relative", "my sister Anna", 0.6),
        ("relative", "my sister", 0.9),
        ("person", "Anna", 0.95),
        ("person", "farmers", 0.85),
    ]
}

struct Setup {
    db: DbInstance,
    tag: String,
    _serial: std::sync::MutexGuard<'static, ()>,
}

impl Setup {
    fn new(tag: &str) -> Self {
        let serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let emb = format!("x_emb_{tag}");
        let dec = format!("x_dec_{tag}");
        let ext = format!("x_ext_{tag}");
        embed::register_embedder(&emb, Arc::new(Words)).unwrap();
        decide::register_decider(
            &dec,
            Arc::new(Nli {
                contra: vec![(PARIS, SISTER_BERLIN), (PARIS, ME_BERLIN), ("I live in Paris", "moved to Lisbon")],
            }),
            Family::Nano,
        )
        .unwrap();
        extract::register_extractor(&ext, Arc::new(Table(spans()))).unwrap();
        let db = DbInstance::new("mem", "", "").unwrap();
        db.memory_init(InitOpts {
            embedder: Some(emb),
            decider: Some(dec),
            extractor: Some(ext),
            ..Default::default()
        })
        .unwrap();
        Setup {
            db,
            tag: tag.to_string(),
            _serial: serial,
        }
    }
}

impl Drop for Setup {
    fn drop(&mut self) {
        embed::unregister_embedder(&format!("x_emb_{}", self.tag));
        decide::unregister_decider(&format!("x_dec_{}", self.tag));
        let _ = extract::disable_extractor(&format!("x_ext_{}", self.tag));
    }
}

fn at(t: i64) -> RememberOpts {
    RememberOpts {
        ns: Some("u".into()),
        at: Some(t),
        ..Default::default()
    }
}

fn names(r: &kleos::memory::RememberResult) -> Vec<String> {
    let mut v: Vec<String> = r.entity_links.iter().map(|e| e.name.clone()).collect();
    v.sort();
    v
}

fn rows(db: &DbInstance, q: &str) -> Vec<Vec<kleos::DataValue>> {
    db.run_script(q, Default::default(), ScriptMutability::Immutable).unwrap().rows
}

#[test]
fn auto_entities_gate_supersession_across_subjects() {
    let s = Setup::new("gate");
    let paris = s.db.remember(PARIS, at(d(1))).unwrap();
    assert_eq!(names(&paris), ["Paris", "user"], "{paris:?}");
    assert_eq!(paris.extractor.as_deref(), Some("table:v1"));
    let paris = paris.id.unwrap();

    // the decider would call it a supersede; the subjects differ (the
    // sister node vs the user), so the question is not asked
    let sis = s.db.remember(SISTER_BERLIN, at(d(2))).unwrap();
    assert_eq!(names(&sis), ["Berlin", "user", "user's sister"], "{sis:?}");
    assert!(sis.superseded.is_empty(), "{sis:?}");
    assert_eq!(sis.gated, vec![paris.clone()], "{sis:?}");
    assert_eq!(sis.edges.len(), 1, "{sis:?}");
    let e = &sis.edges[0];
    assert_eq!((e.src_name.as_str(), e.rel.as_str(), e.dst_name.as_str()), ("user", "sister", "user's sister"));
    // the edge is stored once: as an ontology fact (the personal pack is on
    // by default), not as a `kleos_mem_link` row
    let links = rows(&s.db, "?[rel, decided_by] := *kleos_mem_link{ns: 'u', rel, decided_by}, rel = 'sister'");
    assert!(links.is_empty(), "{links:?}");
    let facts = rows(&s.db, "?[s, o] := *kleos_onto_fact{ns: 'u', s, p, o}, p = 'sister_of'");
    assert_eq!(facts.len(), 1, "{facts:?}");

    // the user's own move does supersede
    let me = s.db.remember(ME_BERLIN, at(d(3))).unwrap();
    assert_eq!(me.superseded, vec![paris], "{me:?}");
}

#[test]
fn name_merges_into_relative_and_pronoun_claims_inherit() {
    let s = Setup::new("merge");
    let r = s
        .db
        .remember("My sister Anna got a new job in Berlin. She moved to Lisbon.", at(d(1)))
        .unwrap();
    assert_eq!(r.claims.len(), 2, "{r:?}");
    assert!(names(&r).contains(&"Anna".to_string()) && names(&r).contains(&"user's sister".to_string()), "{r:?}");
    // the name folds into the relative node
    let merges = rows(&s.db, "?[from, into, decided_by] := *kleos_entity_merge{from, into, decided_by}");
    assert_eq!(merges.len(), 1, "{merges:?}");
    assert_eq!(merges[0][2].get_str(), Some("gliner-rule"));
    // the second claim ("She moved to Lisbon.") mentions the sister/Anna,
    // not the user
    let c2 = r.claims[1].id.clone().unwrap();
    let ments = rows(
        &s.db,
        &format!(
            "?[name] := *kleos_mem_mention{{ns: 'u', id: '{c2}', eid}}, *kleos_mem_entity{{ns: 'u', eid, name}}"
        ),
    );
    let ment: Vec<&str> = ments.iter().filter_map(|r| r[0].get_str()).collect();
    assert!(ment.contains(&"Lisbon"), "{ment:?}");
    assert!(ment.iter().any(|n| *n == "Anna" || *n == "user's sister"), "{ment:?}");
    assert!(!ment.contains(&"user"), "{ment:?}");
    // raw spans with scores and the extractor version, per claim
    let ex = rows(&s.db, "?[id, label, text, score, kept, extractor] := *kleos_mem_extract{ns: 'u', id, label, text, score, kept, extractor}");
    assert!(ex.iter().all(|r| r[5].get_str() == Some("table:v1")), "{ex:?}");
    assert!(ex.iter().any(|r| r[2].get_str() == Some("Lisbon") && r[0].get_str() == Some(c2.as_str())), "{ex:?}");
    // erase removes them
    s.db
        .forget(ForgetOpts {
            ns: Some("u".into()),
            id: r.id.clone(),
            mode: Some("erase".into()),
            ..Default::default()
        })
        .unwrap();
    assert!(rows(&s.db, "?[id] := *kleos_mem_extract{ns: 'u', id}").is_empty());
}

#[test]
fn caller_entities_win_and_extraction_can_be_off() {
    let s = Setup::new("off");
    let r = s
        .db
        .remember(
            PARIS,
            RememberOpts {
                entities: vec!["Somebody".into()],
                ..at(d(1))
            },
        )
        .unwrap();
    assert_eq!(names(&r), ["Somebody"]);
    assert!(r.extractor.is_none());
    let r = s
        .db
        .remember(
            "I visited Lisbon.",
            RememberOpts {
                extract: Some(false),
                ..at(d(2))
            },
        )
        .unwrap();
    assert!(r.entity_links.is_empty() && r.extractor.is_none(), "{r:?}");
    // a lower-case "name" in mixed-case text is not an entity
    let r = s.db.remember("The farmers market in Lisbon was fun.", at(d(3))).unwrap();
    assert_eq!(names(&r), ["Lisbon"], "{r:?}");
    // claim granularity: one call per claim
    let r = s
        .db
        .remember(
            "My sister moved to Berlin. I stayed in Paris.",
            RememberOpts {
                extract_granularity: Some("claim".into()),
                ..at(d(4))
            },
        )
        .unwrap();
    assert!(names(&r).contains(&"user's sister".to_string()) && names(&r).contains(&"Paris".to_string()), "{r:?}");
    assert!(s
        .db
        .remember(
            "x y z",
            RememberOpts {
                extract_granularity: Some("word".into()),
                ..at(d(5))
            }
        )
        .is_err());
}

#[test]
fn extractor_ops() {
    let s = Setup::new("ops");
    let r = s.db.run_script("::extractor list", Default::default(), ScriptMutability::Immutable).unwrap();
    assert!(r.rows.iter().any(|x| x[0].get_str() == Some("x_ext_ops")), "{:?}", r.rows);
    let r = s
        .db
        .run_script("::extractor disable x_ext_ops", Default::default(), ScriptMutability::Mutable)
        .unwrap();
    assert_eq!(r.rows[0][0].get_str(), Some("OK"));
    // configured but disabled: remember works without automatic entities
    let r = s.db.remember(PARIS, at(d(1))).unwrap();
    assert!(r.extractor.is_none() && r.entity_links.is_empty(), "{r:?}");
    s.db
        .run_script("::extractor enable x_ext_ops", Default::default(), ScriptMutability::Mutable)
        .unwrap();
    let r = s.db.remember("I stayed in Lisbon.", at(d(2))).unwrap();
    assert_eq!(r.extractor.as_deref(), Some("table:v1"));
}

/// The real extractor end to end (ignored: needs the checkpoint).
#[test]
#[ignore = "needs the GLiNER checkpoint in KLEOS_GLINER_TEST_MODEL"]
fn memory_extract_gliner() {
    let Some(dir) = std::env::var("KLEOS_GLINER_TEST_MODEL").ok() else {
        return;
    };
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    extract::enable_extractor(
        "gliner_t",
        extract::ExtractorSpec {
            path: Some(dir),
            device: std::env::var("KLEOS_TEST_DEVICE").ok(),
            ..Default::default()
        },
    )
    .unwrap();
    embed::register_embedder("g_emb", Arc::new(Words)).unwrap();
    decide::register_decider(
        "g_dec",
        Arc::new(Nli {
            contra: vec![(PARIS, SISTER_BERLIN), (PARIS, ME_BERLIN)],
        }),
        Family::Nano,
    )
    .unwrap();
    let db = DbInstance::new("mem", "", "").unwrap();
    db.memory_init(InitOpts {
        embedder: Some("g_emb".into()),
        decider: Some("g_dec".into()),
        extractor: Some("gliner_t".into()),
        ..Default::default()
    })
    .unwrap();
    let p = db.remember(PARIS, at(d(1))).unwrap();
    assert_eq!(names(&p), ["Paris", "user"], "{p:?}");
    let s = db.remember(SISTER_BERLIN, at(d(2))).unwrap();
    assert!(s.superseded.is_empty(), "{s:?}");
    assert!(names(&s).contains(&"user's sister".to_string()), "{s:?}");
    let r = db
        .remember("Caroline: My sister Anna moved to Lisbon to work at Google.", at(d(3)))
        .unwrap();
    eprintln!("{:?}\n{:?}", names(&r), r.edges);
    assert!(names(&r).contains(&"Anna".to_string()) && names(&r).contains(&"Caroline's sister".to_string()), "{r:?}");
    assert!(names(&r).contains(&"Google".to_string()) && names(&r).contains(&"Lisbon".to_string()), "{r:?}");
    let _ = extract::disable_extractor("gliner_t");
    embed::unregister_embedder("g_emb");
    decide::unregister_decider("g_dec");
}
