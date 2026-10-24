/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Documents in the memory API (docs/specs/documents.md): `remember` with
//! files, chunk memories with provenance, recall with citations, re-ingestion
//! diffs, durable ingestion and its resume. Text documents need no models;
//! the PDF test (ignored) needs the OCR models and pdfium.
#![cfg(feature = "docs")]

use std::sync::Arc;

use kleos::embed::{self, Embedder};
use kleos::memory::{InitOpts, RecallOpts, RememberOpts};
use kleos::{DbInstance, ScriptMutability};
use serde_json::json;

struct Words;

impl Embedder for Words {
    fn embed(&self, texts: &[&str]) -> miette::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0f32; 64];
                for w in t.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2) {
                    let h = w.to_lowercase().bytes().fold(1469598103934665603u64, |h, b| (h ^ b as u64).wrapping_mul(1099511628211));
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

fn db(tag: &str) -> DbInstance {
    let name = format!("docs_emb_{tag}");
    let _ = embed::register_embedder(&name, Arc::new(Words));
    let db = DbInstance::new("mem", "", "").unwrap();
    db.memory_init(InitOpts { embedder: Some(name), ..Default::default() }).unwrap();
    db
}

const HANDBOOK: &str = "# Staff Handbook\n\nWelcome to the company. This handbook explains how we work.\n\n## Leave\n\nEvery employee gets 28 days of paid annual leave, including public holidays. Unused leave can be carried over up to five days.\n\n## Expenses\n\nTravel expenses are reimbursed within 30 days of a claim. Claims need a receipt.\n\n| Item | Limit |\n|---|---|\n| Hotel | 180 EUR per night |\n| Meals | 45 EUR per day |\n";

fn md(bytes: &str, name: &str) -> serde_json::Value {
    json!({"bytes": bytes.as_bytes(), "name": name, "mime": "text/markdown"})
}

fn rows(db: &DbInstance, q: &str) -> Vec<Vec<kleos::DataValue>> {
    db.run_script(q, Default::default(), ScriptMutability::Immutable).unwrap().rows
}

#[test]
fn remember_a_document_and_recall_with_citations() {
    let db = db("basic");
    let r = db
        .remember("Here is the staff handbook", RememberOpts { attachments: vec![md(HANDBOOK, "handbook.md")], ..Default::default() })
        .unwrap();
    assert_eq!(r.documents.len(), 1, "{r:?}");
    let d = &r.documents[0];
    assert_eq!(d["status"], "done");
    assert_eq!(d["kind"], "markdown");
    assert!(d["chunks"].as_u64().unwrap() >= 1);
    let doc = d["doc"].as_str().unwrap().to_string();
    // the text's memory links to the document
    let links = rows(&db, &format!("?[src, dst] := *kleos_mem_link{{ns: 'default', src, rel: 'attached', dst}}, dst = '{doc}'"));
    assert_eq!(links.len(), 1);
    // recall finds the leave rule, with its document reference
    let hits = db.recall("how many days of annual leave?", RecallOpts { k: Some(3), ..Default::default() }).unwrap();
    let h = hits.iter().find(|h| h.kind == "doc").expect("a chunk hit");
    assert!(h.text.contains("28 days"), "{h:?}");
    let dr = h.doc.as_ref().unwrap();
    assert_eq!(dr["name"], "handbook.md");
    assert!(dr["heading"].as_str().unwrap().contains("Leave"), "{dr}");
    // the context output cites it
    let ctx = db.recall_context("annual leave days", RecallOpts { k: Some(3), ..Default::default() }).unwrap();
    assert!(ctx.text.contains("[handbook.md, § Staff Handbook › Leave]") || ctx.text.contains("[handbook.md"), "{}", ctx.text);
    assert!(ctx.text.contains("## Document: handbook.md"), "{}", ctx.text);
    // sources: memory only leaves chunks out; docs only keeps only them
    let m = db.recall("annual leave", RecallOpts { sources: Some(vec!["memory".into()]), ..Default::default() }).unwrap();
    assert!(m.iter().all(|h| h.kind != "doc"));
    let only = db.recall("handbook", RecallOpts { sources: Some(vec!["docs".into()]), ..Default::default() }).unwrap();
    assert!(!only.is_empty() && only.iter().all(|h| h.kind == "doc"));
    // the table stays whole in one chunk
    let tbl = db.recall("hotel limit per night", RecallOpts { sources: Some(vec!["docs".into()]), k: Some(1), ..Default::default() }).unwrap();
    assert!(tbl[0].text.contains("| Hotel |") && tbl[0].text.contains("| Meals |"), "{:?}", tbl[0].text);
    // ::doc list / get
    let l = rows(&db, "::doc list {ns: 'default'}");
    assert_eq!(l.len(), 1);
    let g = db.doc_call("get", json!({"doc": doc})).unwrap();
    assert_eq!(g["status"], "done");
}

#[test]
fn reingest_is_a_noop_or_a_diff() {
    let db = db("reingest");
    let a = db.remember("", RememberOpts { attachments: vec![md(HANDBOOK, "h.md")], ..Default::default() }).unwrap();
    let n0 = a.documents[0]["chunks"].as_u64().unwrap();
    let again = db.remember("", RememberOpts { attachments: vec![md(HANDBOOK, "h.md")], ..Default::default() }).unwrap();
    assert_eq!(again.documents[0]["duplicate"], true);
    let ids_before: std::collections::BTreeSet<String> = rows(&db, "?[id] := *kleos_mem{ns: 'default', id, kind: 'doc' @ 'NOW'}").iter().map(|r| r[0].get_str().unwrap().to_string()).collect();
    assert_eq!(ids_before.len() as u64, n0);
    // change one section: unchanged chunks keep their memory, the changed one is replaced
    let changed = HANDBOOK.replace("28 days of paid annual leave", "30 days of paid annual leave");
    let b = db.remember("", RememberOpts { attachments: vec![md(&changed, "h.md")], ..Default::default() }).unwrap();
    assert_eq!(b.documents[0]["status"], "done");
    let ids_after: std::collections::BTreeSet<String> = rows(&db, "?[id] := *kleos_mem{ns: 'default', id, kind: 'doc' @ 'NOW'}").iter().map(|r| r[0].get_str().unwrap().to_string()).collect();
    let kept = ids_before.intersection(&ids_after).count();
    assert!(kept >= 1 && kept < ids_before.len(), "kept {kept} of {}", ids_before.len());
    let hits = db.recall("annual leave days", RecallOpts { sources: Some(vec!["docs".into()]), k: Some(1), ..Default::default() }).unwrap();
    assert!(hits[0].text.contains("30 days"), "{:?}", hits[0].text);
}

#[test]
fn chunks_never_take_part_in_decisions() {
    let db = db("decide");
    db.remember("", RememberOpts { attachments: vec![md("Every employee gets 28 days of paid annual leave.\n", "p.md")], ..Default::default() }).unwrap();
    let r = db.remember("Every employee gets 28 days of paid annual leave.", RememberOpts::default()).unwrap();
    assert_eq!(r.action, "added", "a chunk is never a duplicate or superseded: {r:?}");
    assert!(r.decisions.iter().all(|d| !d.candidate.starts_with("dc_")), "{:?}", r.decisions);
}

#[test]
fn background_ingestion_commits_batches_and_resumes() {
    let db = db("flow");
    let pages: String = (0..6).map(|i| format!("## Section {i}\n\nParagraph {i} talks about topic number {i} at length.\n\n")).collect();
    let opts = json!({"inline_pages": 0, "batch_pages": 1});
    let r = db
        .remember("", RememberOpts { attachments: vec![md(&pages, "long.md")], doc: Some(opts.clone()), ..Default::default() })
        .unwrap();
    let d = &r.documents[0];
    assert_eq!(d["status"], "running", "{d}");
    // stop the auto-started worker from racing the manual runs below
    // (the worker of this process may already have claimed it: wait for it)
    let doc = d["doc"].as_str().unwrap().to_string();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let g = db.doc_call("get", json!({"doc": doc})).unwrap();
        if g["status"] == "done" {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "not done: {g}");
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let n = rows(&db, "?[count(id)] := *kleos_mem{ns: 'default', id, kind: 'doc' @ 'NOW'}")[0][0].get_int().unwrap();
    assert!(n >= 1);
    // the same document re-sent in a fresh store, with a failure after the
    // first committed batch: the retry resumes, nothing is written twice
    let db2 = self::db("flow2");
    kleos::docs::fail_after_batches(Some(1));
    let r2 = db2.remember("", RememberOpts { attachments: vec![md(&pages, "long.md")], doc: Some(opts), ..Default::default() }).unwrap();
    let doc2 = r2.documents[0]["doc"].as_str().unwrap().to_string();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let g = db2.doc_call("get", json!({"doc": doc2})).unwrap();
        if g["status"] == "done" {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "not done: {g}");
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let n2 = rows(&db2, "?[count(id)] := *kleos_mem{ns: 'default', id, kind: 'doc' @ 'NOW'}")[0][0].get_int().unwrap();
    assert_eq!(n2, n, "same chunks after the resumed ingestion");
    let prov = rows(&db2, "?[count(id)] := *kleos_doc_chunk{ns: 'default', id}")[0][0].get_int().unwrap();
    assert_eq!(prov, n);
}

#[test]
#[ignore]
fn pdf_with_ocr_and_text_layer() {
    // needs KLEOS_DOCS_SAMPLE=<a PDF> and the OCR models / pdfium (downloaded when allowed)
    let path = std::env::var("KLEOS_DOCS_SAMPLE").expect("KLEOS_DOCS_SAMPLE");
    let db = db("pdf");
    let r = db.remember("", RememberOpts { file: Some(path), doc: Some(json!({"sync": true})), ..Default::default() }).unwrap();
    let d = &r.documents[0];
    assert_eq!(d["status"], "done", "{d}");
    assert!(d["chunks"].as_u64().unwrap() > 0);
    let hits = db.recall("quarterly report", RecallOpts { sources: Some(vec!["docs".into()]), ..Default::default() }).unwrap();
    assert!(hits[0].doc.as_ref().unwrap()["page"].as_u64().is_some());
}

#[test]
#[ignore]
fn pdf_streaming_ingestion_resumes() {
    // needs KLEOS_DOCS_MULTIPAGE=<a PDF of 4+ pages> and the OCR models / pdfium.
    // The streamed extraction is cut by a failure after the second committed
    // batch; the retry resumes and the chunks equal a clean run's.
    let path = std::env::var("KLEOS_DOCS_MULTIPAGE").expect("KLEOS_DOCS_MULTIPAGE");
    let wait = |db: &DbInstance, doc: &str| {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(600);
        loop {
            let g = db.doc_call("get", json!({"doc": doc})).unwrap();
            if g["status"] == "done" {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "not done: {g}");
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    };
    let opts = json!({"inline_pages": 0, "batch_pages": 1});
    let chunks = |db: &DbInstance| -> Vec<String> {
        let mut v: Vec<String> = rows(db, "?[id] := *kleos_doc_chunk{ns: 'default', id}").iter().map(|r| r[0].get_str().unwrap().to_string()).collect();
        v.sort();
        v
    };
    let db1 = db("stream1");
    let r1 = db1.remember("", RememberOpts { file: Some(path.clone()), doc: Some(opts.clone()), ..Default::default() }).unwrap();
    wait(&db1, r1.documents[0]["doc"].as_str().unwrap());
    let clean = chunks(&db1);
    assert!(!clean.is_empty());
    let db2 = db("stream2");
    kleos::docs::fail_after_batches(Some(2));
    let r2 = db2.remember("", RememberOpts { file: Some(path), doc: Some(opts), ..Default::default() }).unwrap();
    wait(&db2, r2.documents[0]["doc"].as_str().unwrap());
    let resumed = chunks(&db2);
    assert_eq!(resumed, clean, "same chunks after the resumed streaming ingestion");
    let mems = rows(&db2, "?[count(id)] := *kleos_mem{ns: 'default', id, kind: 'doc' @ 'NOW'}")[0][0].get_int().unwrap();
    assert_eq!(mems as usize, clean.len());
}

/// Document chunks get entities from the store's extractor (one fake span).
struct Names;

impl kleos::extract::EntityExtractor for Names {
    fn model_id(&self) -> String {
        "names".into()
    }
    fn version(&self) -> String {
        "names:v1".into()
    }
    fn labels(&self) -> Vec<String> {
        vec!["person".into()]
    }
    fn extract(&self, texts: &[&str]) -> miette::Result<Vec<Vec<kleos::extract::ExtractSpan>>> {
        Ok(texts
            .iter()
            .map(|t| {
                t.find("Ingrid Solberg")
                    .map(|a| {
                        vec![kleos::extract::ExtractSpan { label: "person".into(), start: a, end: a + 14, text: "Ingrid Solberg".into(), score: 0.99 }]
                    })
                    .unwrap_or_default()
            })
            .collect())
    }
}

#[test]
fn document_chunks_get_entities_from_the_extractor() {
    let name = "docs_emb_ents";
    let _ = embed::register_embedder(name, Arc::new(Words));
    let _ = kleos::extract::register_extractor("docs_names", Arc::new(Names));
    let db = DbInstance::new("mem", "", "").unwrap();
    db.memory_init(InitOpts { embedder: Some(name.into()), extractor: Some("docs_names".into()), ..Default::default() }).unwrap();
    let text = "# Contacts\n\nThe harbour master is Ingrid Solberg; the night desk is extension 4417.\n";
    db.remember("", RememberOpts { attachments: vec![md(text, "c.md")], ..Default::default() }).unwrap();
    let m = rows(&db, "?[id, name] := *kleos_mem_mention{ns: 'default', id, eid}, *kleos_mem_entity{ns: 'default', eid, name}, starts_with(id, 'dc_')");
    assert!(m.iter().any(|r| r[1].get_str() == Some("Ingrid Solberg")), "{m:?}");
}
