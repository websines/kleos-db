/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */
//! Documents in the memory API (docs/specs/documents.md): `remember` takes
//! files; each is read (PDF text layer or OCR, `kleos-docs`), laid out,
//! chunked, embedded and stored as memories (`kind: 'doc'`, role `other`)
//! with page-level provenance; large documents ingest as a durable workflow
//! (`doc.ingest`) whose page batches commit with their checkpoints.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use base64::Engine;
use itertools::Itertools;
use kleos_docs::chunk::{chunk_blocks, markdown_blocks, ApproxTokenizer, ChunkOpts, DocBlock};
use kleos_docs::formats::{self, Kind};
use kleos_docs::ocr::document::{Converter, Mode};
use kleos_docs::ocr::Ocr;
use miette::{bail, miette, IntoDiagnostic, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use sha2::{Digest, Sha256};

use crate::data::value::{DataValue, Validity};
use crate::flow::{ClaimOpts, FLOW_TASK};
use crate::runtime::db::{Db, NamedRows, ScriptMutability};
use crate::storage::Storage;

/// Documents, one row per version (valid time).
pub const DOC: &str = "kleos_doc";
/// Provenance of chunk memories.
pub const DOC_CHUNK: &str = "kleos_doc_chunk";
/// Bytes of documents passed as bytes, kept while they ingest.
pub const DOC_BLOB: &str = "kleos_doc_blob";
pub use crate::memory::DOC_KIND;
/// The ingestion workflow and its queue.
pub const DOC_WORKFLOW: &str = "doc.ingest";
pub const DOC_QUEUE: &str = "docs";

const DDL: &[(&str, &str)] = &[
    (
        DOC,
        "{ns: String, doc: String, vt: Validity => name: String, mime: String, hash: String, pages: Int, \
         title: String?, status: String, chunks: Int, meta: Json?, ingested: Int}",
    ),
    (
        DOC_CHUNK,
        "{ns: String, id: String => doc: String, n: Int, page: Int?, page_end: Int?, boxes: Json?, \
         heading: String?, start: Int, end: Int, hash: String, label: String}",
    ),
    (DOC_BLOB, "{doc: String, part: Int => bytes: Bytes}"),
];

const DEFAULT_INLINE_PAGES: usize = 4;
const DEFAULT_BATCH_PAGES: usize = 8;
const LEASE_MS: i64 = 30_000;
const BLOB_PART: usize = 1 << 20;

/// A document passed to `remember`: a path, bytes (or base64), a URL, with
/// an optional mime type and name.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Attachment {
    pub path: Option<String>,
    /// Raw bytes: a list of byte values, or a base64 string (what a bytes
    /// value becomes in JSON, e.g. from Python).
    pub bytes: Option<BytesIn>,
    pub data_b64: Option<String>,
    pub url: Option<String>,
    pub mime: Option<String>,
    pub name: Option<String>,
}

/// Bytes in JSON: a list of numbers or base64 text.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BytesIn {
    Raw(Vec<u8>),
    B64(String),
}

impl BytesIn {
    fn decode(&self) -> Result<Vec<u8>> {
        match self {
            BytesIn::Raw(v) => Ok(v.clone()),
            BytesIn::B64(t) => base64::engine::general_purpose::STANDARD.decode(t.trim()).map_err(|e| miette!("bytes: {e}")),
        }
    }
}

/// A path, or a full [`Attachment`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttachmentSpec {
    Path(String),
    Full(Attachment),
}

impl AttachmentSpec {
    fn into_attachment(self) -> Attachment {
        match self {
            AttachmentSpec::Path(p) => Attachment { path: Some(p), ..Default::default() },
            AttachmentSpec::Full(a) => a,
        }
    }
}

/// How documents are read and chunked (`remember {doc: {...}}`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DocOpts {
    /// Chunking strategy and size (default: recursive, 384 tokens).
    pub chunker: Option<ChunkOpts>,
    /// `auto` (default: text layer where sound, OCR elsewhere), `ocr`
    /// (always OCR), `text` (the text layer even when its checks fail).
    pub ocr: Option<String>,
    /// PDFs up to this many pages ingest inside `remember` (default 4).
    pub inline_pages: Option<usize>,
    /// Pages per durable step (default 8).
    pub batch_pages: Option<usize>,
    /// Ingest inside `remember` whatever the size.
    pub sync: Option<bool>,
    /// Pages OCR'd in parallel, one single-threaded worker each (default:
    /// the cores, at most 4).
    pub threads: Option<usize>,
    /// Render resolution for OCR (default 200).
    pub dpi: Option<f64>,
}

/// What `remember` reports per document.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct DocStatus {
    pub doc: String,
    pub name: String,
    pub kind: String,
    /// `done`, `running` (a durable workflow ingests it), `error`.
    pub status: String,
    pub pages: usize,
    pub chunks: usize,
    /// The ingestion workflow, when it runs in the background.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flow: Option<String>,
    /// The same bytes were already stored.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub duplicate: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub timings: BTreeMap<String, f64>,
}

/// A chunk hit's document reference.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DocRef {
    pub doc: String,
    pub name: String,
    pub page: Option<usize>,
    pub page_end: Option<usize>,
    pub heading: Option<String>,
}

impl DocRef {
    /// `[name p. 14, § Heading]`.
    pub fn cite(&self) -> String {
        let mut s = format!("[{}", self.name);
        match (self.page, self.page_end) {
            (Some(a), Some(b)) if b > a => s.push_str(&format!(" pp. {}–{}", a + 1, b + 1)),
            (Some(a), _) => s.push_str(&format!(" p. {}", a + 1)),
            _ => {}
        }
        if let Some(h) = self.heading.as_ref().filter(|h| !h.is_empty()) {
            s.push_str(&format!(", § {h}"));
        }
        s.push(']');
        s
    }
}

/// Per-store state of the background document worker.
#[derive(Default)]
pub struct DocsState {
    pub(crate) running: AtomicBool,
    /// pending workflows were looked for once (a restarted process resumes them)
    pub(crate) checked: AtomicBool,
    pub(crate) stop: AtomicBool,
    /// pages ingested by this process (for progress reporting)
    pub(crate) pages_done: AtomicU64,
}

// ------------------------------------------------------------------ engines

static FAIL_AFTER: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(-1);

/// Testing: make the next ingestion fail after committing `n` page batches
/// (as a crash would, after the commit), once.
#[doc(hidden)]
pub fn fail_after_batches(n: Option<usize>) {
    FAIL_AFTER.store(n.map(|x| x as i64).unwrap_or(-1), Ordering::SeqCst);
}

static OCR: OnceLock<Mutex<Option<Arc<Ocr>>>> = OnceLock::new();

/// The OCR engine, loaded (and its models downloaded) on first use.
pub fn ocr_engine() -> Result<Arc<Ocr>> {
    let cell = OCR.get_or_init(|| Mutex::new(None));
    let mut g = cell.lock().unwrap();
    if let Some(o) = g.as_ref() {
        return Ok(o.clone());
    }
    let dir = kleos_docs::ocr::model_dir().map_err(|e| miette!("OCR models: {e}"))?;
    // single-threaded inside a page: pages run in parallel instead (one
    // worker per core, as Indra's own convert_many does)
    let o = Arc::new(Ocr::load(&dir, 1).map_err(|e| miette!("OCR models: {e}"))?);
    *g = Some(o.clone());
    Ok(o)
}

/// Make sure pdfium is available (downloading the pinned build when allowed).
fn ensure_pdf() -> Result<()> {
    if kleos_docs::pdf::available() {
        return Ok(());
    }
    kleos_docs::pdf::ensure_pdfium().map_err(|e| miette!("{e}"))?;
    if kleos_docs::pdf::available() {
        Ok(())
    } else {
        bail!("pdfium could not be loaded after the download")
    }
}

// ------------------------------------------------------------------ helpers

fn s(x: &str) -> DataValue {
    DataValue::from(x)
}

fn opt_s(x: &Option<String>) -> DataValue {
    x.as_deref().map(DataValue::from).unwrap_or(DataValue::Null)
}

fn now_us() -> i64 {
    crate::flow::now_us()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn sha(parts: &[&[u8]]) -> String {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
        h.update([0u8]);
    }
    hex(&h.finalize())
}

fn params(p: Vec<(&str, DataValue)>) -> BTreeMap<String, DataValue> {
    p.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn cell_str(v: &DataValue) -> Option<String> {
    v.get_str().map(|x| x.to_string())
}

fn cell_i(v: &DataValue) -> Option<i64> {
    v.get_int()
}

/// Where a document's bytes come from.
#[derive(Clone)]
enum Source {
    Path(PathBuf),
    Bytes(Arc<Vec<u8>>),
}

impl Source {
    fn bytes(&self) -> Result<Arc<Vec<u8>>> {
        match self {
            Source::Path(p) => Ok(Arc::new(std::fs::read(p).map_err(|e| miette!("{}: {e}", p.display()))?)),
            Source::Bytes(b) => Ok(b.clone()),
        }
    }
}

/// One page's blocks (page index, blocks).
type PageBlocks = (usize, Vec<DocBlock>);

/// Blocks of an OCR'd page from its Markdown parts; boxes in points when
/// `px_per_pt` is known (PDF pages), else pixels.
fn page_to_blocks(page: &kleos_docs::ocr::Page, idx: usize, px_per_pt: Option<f64>) -> Vec<DocBlock> {
    let k = px_per_pt.unwrap_or(1.0) as f32;
    page.parts
        .iter()
        .filter(|p| p.label != "Picture" && !p.text.trim().is_empty())
        .map(|p| {
            let (text, level) = match p.text.strip_prefix("## ") {
                Some(_) if p.label == "Section-Header" => (p.text.clone(), Some(2)),
                _ => (p.text.clone(), None),
            };
            DocBlock {
                label: p.label.clone(),
                page: Some(idx),
                bbox: Some([p.bbox[0] / k, p.bbox[1] / k, p.bbox[2] / k, p.bbox[3] / k]),
                text,
                level,
            }
        })
        .collect()
}

/// A text document's blocks.
fn text_blocks(kind: Kind, bytes: &[u8]) -> Result<Vec<DocBlock>> {
    let md = match kind {
        Kind::Html => formats::html_to_markdown(&String::from_utf8_lossy(bytes)),
        Kind::Docx => formats::docx_to_markdown(bytes).map_err(|e| miette!("docx: {e}"))?,
        _ => String::from_utf8_lossy(bytes).to_string(),
    };
    Ok(markdown_blocks(&md))
}

fn converter_mode(o: &DocOpts) -> Result<Mode> {
    Ok(match o.ocr.as_deref() {
        None | Some("auto") => Mode::Auto,
        Some("ocr") | Some("force") => Mode::Ocr,
        Some("text") => Mode::Text,
        Some(x) => bail!("doc option ocr must be 'auto', 'ocr' or 'text', got '{x}'"),
    })
}

fn threads_of(o: &DocOpts) -> usize {
    o.threads.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(4)).max(1)
}

/// Blocks of pages [from, to) and the time spent (render/text, OCR and so on).
fn extract_pages(src: &Source, kind: Kind, from: usize, to: usize, o: &DocOpts, on_page: &(dyn Fn(usize) + Sync)) -> Result<(Vec<PageBlocks>, BTreeMap<String, f64>)> {
    let mut timings: BTreeMap<String, f64> = BTreeMap::new();
    match kind {
        Kind::Pdf => {
            ensure_pdf()?;
            let ocr = ocr_engine()?;
            let mut conv = Converter::new(&ocr);
            conv.mode = converter_mode(o)?;
            if let Some(d) = o.dpi {
                conv.dpi = d;
            }
            let doc = match src {
                Source::Path(p) => kleos_docs::pdf::Doc::open(p, None),
                Source::Bytes(b) => kleos_docs::pdf::Doc::from_bytes(b.to_vec(), None),
            }
            .map_err(|e| miette!("{e}"))?;
            let dpi = conv.dpi;
            let mut out: Vec<PageBlocks> = vec![];
            let t = Mutex::new(BTreeMap::<String, f64>::new());
            let mut sink = |i: usize, p: kleos_docs::ocr::Page| {
                {
                    let mut t = t.lock().unwrap();
                    for (k, v) in &p.timings {
                        *t.entry(k.clone()).or_insert(0.0) += v;
                    }
                    *t.entry(format!("pages_{}", p.source)).or_insert(0.0) += 1.0;
                }
                on_page(i);
                out.push((i, page_to_blocks(&p, i, Some(dpi / 72.0))));
            };
            conv.convert_doc_parallel(&doc, from, to, threads_of(o), &mut sink);
            timings = t.into_inner().unwrap();
            Ok((out, timings))
        }
        Kind::Image => {
            let ocr = ocr_engine()?;
            let bytes = src.bytes()?;
            let g = kleos_docs::ocr::document::read_image(&bytes).map_err(|e| miette!("{e}"))?;
            let t0 = Instant::now();
            let p = ocr.read(&g, true).map_err(|e| miette!("{e}"))?;
            timings.insert("ocr_ms".into(), t0.elapsed().as_secs_f64() * 1000.0);
            on_page(0);
            Ok((vec![(0, page_to_blocks(&p, 0, None))], timings))
        }
        _ => {
            let bytes = src.bytes()?;
            on_page(0);
            Ok((vec![(0, text_blocks(kind, &bytes)?)], timings))
        }
    }
}

/// The chunks of some pages, as memory rows to write.
struct ChunkRow {
    id: String,
    n: i64,
    text: String,
    page: Option<usize>,
    page_end: Option<usize>,
    boxes: JsonValue,
    heading: Option<String>,
    start: i64,
    end: i64,
    hash: String,
    label: String,
}

fn chunk_pages(doc: &str, pages: &[PageBlocks], copts: &ChunkOpts, n0: i64, seen: &mut BTreeMap<String, usize>) -> Vec<ChunkRow> {
    let blocks: Vec<DocBlock> = pages.iter().flat_map(|(_, b)| b.iter().cloned()).collect();
    let chunks = chunk_blocks(&blocks, &ApproxTokenizer, copts, None);
    let short = &doc[doc.len().saturating_sub(12)..];
    chunks
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            let h = sha(&[c.text.as_bytes()]);
            let occ = seen.entry(h.clone()).or_insert(0);
            *occ += 1;
            let key = if *occ > 1 { sha(&[c.text.as_bytes(), occ.to_string().as_bytes()]) } else { h.clone() };
            ChunkRow {
                id: format!("dc_{short}_{}", &key[..16]),
                n: n0 + i as i64,
                text: c.text,
                page: c.page,
                page_end: c.page_end,
                boxes: json!(c.boxes.iter().map(|(p, b)| json!({"page": p, "bbox": b})).collect::<Vec<_>>()),
                heading: if c.heading.is_empty() { None } else { Some(c.heading.join(" › ")) },
                start: c.start as i64,
                end: c.end as i64,
                hash: h,
                label: c.label,
            }
        })
        .collect()
}

/// The ingestion workflow's input.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct IngestInput {
    ns: String,
    doc: String,
    name: String,
    kind: Kind,
    mime: String,
    hash: String,
    path: Option<String>,
    pages: usize,
    opts: DocOpts,
    /// chunk ids of the previous version (re-ingestion)
    previous: Vec<String>,
}

impl<'s, S: Storage<'s>> Db<S> {
    fn dq(&'s self, script: &str, p: Vec<(&str, DataValue)>) -> Result<NamedRows> {
        self.run_script(script, params(p), ScriptMutability::Immutable)
    }

    fn dw(&'s self, script: &str, p: Vec<(&str, DataValue)>) -> Result<NamedRows> {
        let mut attempt = 0;
        loop {
            match self.run_script(script, params(p.clone()), ScriptMutability::Mutable) {
                Ok(r) => return Ok(r),
                Err(e) if attempt < 8 && crate::memory::is_contention(&e) => {
                    std::thread::sleep(std::time::Duration::from_millis(1 << attempt));
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Create the document relations when missing.
    pub(crate) fn docs_ensure(&'s self) -> Result<()> {
        let missing: Vec<&(&str, &str)> = {
            let tx = self.transact()?;
            DDL.iter().filter(|(n, _)| tx.get_relation(n, false).is_err()).collect()
        };
        for (name, cols) in missing {
            if let Err(e) = self.run_script(&format!(":create {name} {cols}"), Default::default(), ScriptMutability::Mutable) {
                let tx = self.transact()?;
                if tx.get_relation(name, false).is_err() {
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// The current row of a document: (hash, status, chunks, pages, name).
    fn doc_current(&'s self, ns: &str, doc: &str) -> Result<Option<(String, String, i64, i64, String)>> {
        let r = self.dq(
            &format!("?[hash, status, chunks, pages, name] := *{DOC}{{ns: $ns, doc: $doc, hash, status, chunks, pages, name @ 'NOW'}}"),
            vec![("ns", s(ns)), ("doc", s(doc))],
        )?;
        Ok(r.rows.first().map(|row| {
            (
                cell_str(&row[0]).unwrap_or_default(),
                cell_str(&row[1]).unwrap_or_default(),
                cell_i(&row[2]).unwrap_or(0),
                cell_i(&row[3]).unwrap_or(0),
                cell_str(&row[4]).unwrap_or_default(),
            )
        }))
    }

    /// Chunk ids of a document's current chunks.
    fn doc_chunk_ids(&'s self, ns: &str, doc: &str) -> Result<Vec<String>> {
        let r = self.dq(
            &format!(
                "?[id] := *{DOC_CHUNK}{{ns: $ns, id, doc: $doc}}, *kleos_mem{{ns: $ns, id @ 'NOW'}}"
            ),
            vec![("ns", s(ns)), ("doc", s(doc))],
        )?;
        Ok(r.rows.iter().filter_map(|r| cell_str(&r[0])).collect())
    }

    fn doc_row_script(&'s self) -> String {
        format!(
            "?[ns, doc, vt, name, mime, hash, pages, title, status, chunks, meta, ingested] <- $doc_rows \
             :put {DOC} {{ns, doc, vt => name, mime, hash, pages, title, status, chunks, meta, ingested}}"
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn doc_row(ns: &str, doc: &str, at: i64, name: &str, mime: &str, hash: &str, pages: usize, status: &str, chunks: i64, meta: JsonValue) -> DataValue {
        DataValue::List(vec![
            s(ns),
            s(doc),
            DataValue::Validity(Validity::from((at, true))),
            s(name),
            s(mime),
            s(hash),
            DataValue::from(pages as i64),
            DataValue::Null,
            s(status),
            DataValue::from(chunks),
            crate::flow::jd(&meta),
            DataValue::from(now_us()),
        ])
    }

    /// The scripts that write `rows` (new chunks only) as memories with their
    /// provenance, one program each.
    fn chunk_scripts(&'s self, ns: &str, doc: &str, rows: &[ChunkRow], vecs: &[Option<Vec<f32>>], at: i64) -> Vec<(String, BTreeMap<String, DataValue>)> {
        if rows.is_empty() {
            return vec![];
        }
        let source = format!("doc:{doc}");
        let now = now_us();
        let mems: Vec<DataValue> = rows
            .iter()
            .map(|r| {
                DataValue::List(vec![
                    s(ns),
                    s(&r.id),
                    DataValue::Validity(Validity::from((at, true))),
                    s(&r.text),
                    s(DOC_KIND),
                    s(&source),
                    DataValue::from(now),
                    DataValue::Null,
                ])
            })
            .collect();
        let roles: Vec<DataValue> = rows.iter().map(|r| DataValue::List(vec![s(ns), s(&r.id), s("other")])).collect();
        let prov: Vec<DataValue> = rows
            .iter()
            .map(|r| {
                DataValue::List(vec![
                    s(ns),
                    s(&r.id),
                    s(doc),
                    DataValue::from(r.n),
                    r.page.map(|p| DataValue::from(p as i64)).unwrap_or(DataValue::Null),
                    r.page_end.map(|p| DataValue::from(p as i64)).unwrap_or(DataValue::Null),
                    crate::flow::jd(&r.boxes),
                    opt_s(&r.heading),
                    DataValue::from(r.start),
                    DataValue::from(r.end),
                    s(&r.hash),
                    s(&r.label),
                ])
            })
            .collect();
        let mut out = vec![
            (
                "?[ns, id, vt, text, kind, source, created, importance] <- $mems \
                 :put kleos_mem {ns, id, vt => text, kind, source, created, importance}"
                    .to_string(),
                params(vec![("mems", DataValue::List(mems))]),
            ),
            (
                "?[ns, id, role] <- $roles :put kleos_mem_role {ns, id => role}".to_string(),
                params(vec![("roles", DataValue::List(roles))]),
            ),
            (
                format!(
                    "?[ns, id, doc, n, page, page_end, boxes, heading, start, end, hash, label] <- $prov \
                     :put {DOC_CHUNK} {{ns, id => doc, n, page, page_end, boxes, heading, start, end, hash, label}}"
                ),
                params(vec![("prov", DataValue::List(prov))]),
            ),
        ];
        let vrows: Vec<DataValue> = rows
            .iter()
            .zip(vecs.iter())
            .filter_map(|(r, v)| {
                v.as_ref().map(|v| {
                    DataValue::List(vec![
                        s(ns),
                        s(&r.id),
                        DataValue::Bool(true),
                        DataValue::List(v.iter().map(|x| DataValue::from(*x as f64)).collect()),
                    ])
                })
            })
            .collect();
        if !vrows.is_empty() {
            out.push((
                "?[ns, id, current, vec] <- $vecs :put kleos_mem_vec {ns, id => current, vec}".to_string(),
                params(vec![("vecs", DataValue::List(vrows))]),
            ));
        }
        out
    }

    /// Embed chunk texts with the store's embedder (None per row without one).
    fn embed_chunks(&'s self, rows: &[ChunkRow]) -> Result<Vec<Option<Vec<f32>>>> {
        let cfg = self.memory_config()?;
        let Some(emb) = cfg.and_then(|c| c.embedder) else {
            return Ok(rows.iter().map(|_| None).collect());
        };
        let mut out = Vec::with_capacity(rows.len());
        for part in rows.chunks(16) {
            let refs: Vec<&str> = part.iter().map(|r| r.text.as_str()).collect();
            for v in crate::embed::embed_texts(&emb, &refs)? {
                out.push(Some(v.to_vec()));
            }
        }
        Ok(out)
    }

    /// Ingest documents for `remember`: small ones inside the call, large
    /// ones as durable workflows. Returns one status per document.
    pub(crate) fn docs_remember(&'s self, ns: &str, atts: Vec<AttachmentSpec>, o: &DocOpts) -> Result<Vec<DocStatus>> {
        self.docs_ensure()?;
        let mut out = vec![];
        for a in atts {
            let a = a.into_attachment();
            match self.doc_one(ns, a, o) {
                Ok(st) => out.push(st),
                Err(e) => out.push(DocStatus { status: "error".into(), error: Some(e.to_string()), ..Default::default() }),
            }
        }
        Ok(out)
    }

    fn doc_one(&'s self, ns: &str, a: Attachment, o: &DocOpts) -> Result<DocStatus> {
        let (src, name) = if let Some(p) = &a.path {
            let pb = PathBuf::from(p);
            let name = a.name.clone().unwrap_or_else(|| pb.file_name().map(|x| x.to_string_lossy().to_string()).unwrap_or_else(|| p.clone()));
            (Source::Path(pb), name)
        } else if let Some(b) = &a.bytes {
            (Source::Bytes(Arc::new(b.decode()?)), a.name.clone().unwrap_or_default())
        } else if let Some(b64) = &a.data_b64 {
            let b = base64::engine::general_purpose::STANDARD.decode(b64.trim()).map_err(|e| miette!("data_b64: {e}"))?;
            (Source::Bytes(Arc::new(b)), a.name.clone().unwrap_or_default())
        } else if a.url.is_some() {
            bail!("documents by URL need the `docs-url` feature (off by default); pass the bytes or a path")
        } else {
            bail!("an attachment needs 'path', 'bytes' or 'data_b64'")
        };
        let bytes = src.bytes()?;
        let kind = formats::detect(&bytes, a.mime.as_deref(), Some(&name)).ok_or_else(|| miette!("{name}: not a PDF, image or text document"))?;
        let hash = sha(&[&bytes]);
        let name = if name.is_empty() { format!("document-{}", &hash[..8]) } else { name };
        let doc = format!("doc_{}", &sha(&[ns.as_bytes(), name.as_bytes()])[..16]);
        let mime = a.mime.clone().unwrap_or_else(|| kind.mime().to_string());
        let kname = format!("{kind:?}").to_lowercase();
        // the same bytes: nothing to do
        let mut previous = vec![];
        if let Some((h, status, chunks, pages, _)) = self.doc_current(ns, &doc)? {
            if h == hash && status != "error" {
                return Ok(DocStatus { doc, name, kind: kname, status, pages: pages as usize, chunks: chunks as usize, duplicate: true, ..Default::default() });
            }
            previous = self.doc_chunk_ids(ns, &doc)?;
        }
        let pages = match kind {
            Kind::Pdf => {
                ensure_pdf()?;
                let d = match &src {
                    Source::Path(p) => kleos_docs::pdf::Doc::open(p, None),
                    Source::Bytes(b) => kleos_docs::pdf::Doc::from_bytes(b.to_vec(), None),
                }
                .map_err(|e| miette!("{name}: {e}"))?;
                d.pages
            }
            _ => 1,
        };
        let inline = o.sync.unwrap_or(false) || pages <= o.inline_pages.unwrap_or(DEFAULT_INLINE_PAGES);
        if inline {
            return self.doc_ingest_inline(ns, &doc, &name, kind, &mime, &hash, &src, pages, o, &previous);
        }
        // background: keep the bytes (a path is re-read and its hash checked)
        let at = now_us();
        let path = match &src {
            Source::Path(p) => Some(p.to_string_lossy().to_string()),
            Source::Bytes(b) => {
                let parts: Vec<DataValue> = b
                    .chunks(BLOB_PART)
                    .enumerate()
                    .map(|(i, c)| DataValue::List(vec![s(&doc), DataValue::from(i as i64), DataValue::Bytes(c.to_vec())]))
                    .collect();
                self.dw(&format!("?[doc, part, bytes] <- $parts :put {DOC_BLOB} {{doc, part => bytes}}"), vec![("parts", DataValue::List(parts))])?;
                None
            }
        };
        let input = IngestInput { ns: ns.into(), doc: doc.clone(), name: name.clone(), kind, mime: mime.clone(), hash: hash.clone(), path, pages, opts: o.clone(), previous };
        self.dw(
            &self.doc_row_script(),
            vec![("doc_rows", DataValue::List(vec![Self::doc_row(ns, &doc, at, &name, &mime, &hash, pages, "running", 0, json!({"kind": kname}))]))],
        )?;
        let wf = format!("doc:{ns}:{doc}:{}", &hash[..12]);
        let started = self.flow_call(
            "start",
            json!({"workflow": DOC_WORKFLOW, "mode": "code", "queue": DOC_QUEUE, "id": wf, "input": serde_json::to_value(&input).into_diagnostic()?}),
        )?;
        let fid = started.get("id").and_then(|x| x.as_str()).unwrap_or(&wf).to_string();
        Ok(DocStatus { doc, name, kind: kname, status: "running".into(), pages, chunks: 0, flow: Some(fid), ..Default::default() })
    }

    #[allow(clippy::too_many_arguments)]
    fn doc_ingest_inline(&'s self, ns: &str, doc: &str, name: &str, kind: Kind, mime: &str, hash: &str, src: &Source, pages: usize, o: &DocOpts, previous: &[String]) -> Result<DocStatus> {
        let t0 = Instant::now();
        let (pb, mut timings) = extract_pages(src, kind, 0, pages, o, &|_| {})?;
        timings.insert("extract_ms".into(), t0.elapsed().as_secs_f64() * 1000.0);
        let t1 = Instant::now();
        let copts = o.chunker.clone().unwrap_or_default();
        let mut seen = BTreeMap::new();
        let rows = chunk_pages(doc, &pb, &copts, 0, &mut seen);
        timings.insert("chunk_ms".into(), t1.elapsed().as_secs_f64() * 1000.0);
        let prev: BTreeSet<&str> = previous.iter().map(|x| x.as_str()).collect();
        let new_rows: Vec<ChunkRow> = rows.iter().filter(|r| !prev.contains(r.id.as_str())).map(|r| ChunkRow { ..clone_row(r) }).collect();
        let t2 = Instant::now();
        let vecs = self.embed_chunks(&new_rows)?;
        timings.insert("embed_ms".into(), t2.elapsed().as_secs_f64() * 1000.0);
        let t3 = Instant::now();
        let at = now_us();
        let ids: BTreeSet<String> = rows.iter().map(|r| r.id.clone()).collect();
        let gone: Vec<String> = previous.iter().filter(|x| !ids.contains(*x)).cloned().collect();
        let mut scripts = self.chunk_scripts(ns, doc, &new_rows, &vecs, at);
        scripts.extend(self.retract_scripts(ns, &gone, at));
        scripts.push((
            self.doc_row_script(),
            params(vec![("doc_rows", DataValue::List(vec![Self::doc_row(ns, doc, at, name, mime, hash, pages, "done", rows.len() as i64, json!({"kind": format!("{kind:?}").to_lowercase(), "kept": rows.len() - new_rows.len(), "removed": gone.len()}))]))]),
        ));
        self.run_scripts_tx(&scripts)?;
        timings.insert("store_ms".into(), t3.elapsed().as_secs_f64() * 1000.0);
        Ok(DocStatus { doc: doc.into(), name: name.into(), kind: format!("{kind:?}").to_lowercase(), status: "done".into(), pages, chunks: rows.len(), timings, ..Default::default() })
    }

    /// Scripts that retract chunk memories (valid time ends; vectors no longer current).
    fn retract_scripts(&'s self, ns: &str, ids: &[String], at: i64) -> Vec<(String, BTreeMap<String, DataValue>)> {
        if ids.is_empty() {
            return vec![];
        }
        let idl = DataValue::List(ids.iter().map(|x| s(x)).collect());
        vec![
            (
                "?[ns, id, vt, text, kind, source, created, importance] := *kleos_mem{ns, id, text, kind, source, created, importance @ 'NOW'}, \
                   ns = $ns, id in $old, vt = [$at, false] \
                 :put kleos_mem {ns, id, vt => text, kind, source, created, importance}"
                    .to_string(),
                params(vec![("ns", s(ns)), ("old", idl.clone()), ("at", DataValue::from(at))]),
            ),
            (
                "?[ns, id, current] := *kleos_mem_vec{ns, id}, ns = $ns, id in $old, current = false :update kleos_mem_vec {ns, id => current}".to_string(),
                params(vec![("ns", s(ns)), ("old", idl)]),
            ),
        ]
    }

    /// Run scripts in one transaction.
    fn run_scripts_tx(&'s self, scripts: &[(String, BTreeMap<String, DataValue>)]) -> Result<()> {
        use crate::runtime::db::TransactionPayload;
        use crossbeam::channel::bounded;
        let mut attempt = 0;
        loop {
            let (pay_tx, pay_rx) = bounded(1);
            let (res_tx, res_rx) = bounded(1);
            let out: Result<()> = std::thread::scope(move |sc| {
                let pay_tx = pay_tx;
                sc.spawn(move || self.run_multi_transaction(true, pay_rx, res_tx));
                let gone = || miette!("documents: the transaction ended early");
                for (q, p) in scripts {
                    pay_tx.send(TransactionPayload::Query((q.clone(), p.clone()))).map_err(|_| gone())?;
                    res_rx.recv().map_err(|_| gone())??;
                }
                pay_tx.send(TransactionPayload::Commit).map_err(|_| gone())?;
                res_rx.recv().map_err(|_| gone())??;
                Ok(())
            });
            match out {
                Ok(()) => return Ok(()),
                Err(e) if attempt < 8 && crate::memory::is_contention(&e) => {
                    std::thread::sleep(std::time::Duration::from_millis(1 << attempt));
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// The document references of chunk memories, by id.
    pub(crate) fn doc_refs(&'s self, ns_list: &[String], ids: &[String]) -> Result<BTreeMap<String, DocRef>> {
        if ids.is_empty() {
            return Ok(BTreeMap::new());
        }
        {
            let tx = self.transact()?;
            if tx.get_relation(DOC_CHUNK, false).is_err() {
                return Ok(BTreeMap::new());
            }
        }
        let mut out = BTreeMap::new();
        for ns in ns_list {
            let r = self.dq(
                &format!(
                    "?[id, doc, name, page, page_end, heading] := id in $ids, *{DOC_CHUNK}{{ns: $ns, id, doc, page, page_end, heading}}, \
                       *{DOC}{{ns: $ns, doc, name @ 'NOW'}}"
                ),
                vec![("ns", s(ns)), ("ids", DataValue::List(ids.iter().map(|x| s(x)).collect()))],
            )?;
            for row in r.rows {
                let id = cell_str(&row[0]).unwrap_or_default();
                out.entry(id).or_insert(DocRef {
                    doc: cell_str(&row[1]).unwrap_or_default(),
                    name: cell_str(&row[2]).unwrap_or_default(),
                    page: cell_i(&row[3]).map(|x| x as usize),
                    page_end: cell_i(&row[4]).map(|x| x as usize),
                    heading: cell_str(&row[5]),
                });
            }
        }
        Ok(out)
    }

    /// Link memories to the documents that came with them.
    pub(crate) fn link_attached(&'s self, ns: &str, mems: &[String], docs: &[String]) -> Result<()> {
        if mems.is_empty() || docs.is_empty() {
            return Ok(());
        }
        let now = now_us();
        let rows: Vec<DataValue> = mems
            .iter()
            .cartesian_product(docs.iter())
            .map(|(m, d)| DataValue::List(vec![s(ns), s(m), s("attached"), s(d), DataValue::from(now), DataValue::Null, DataValue::Null]))
            .collect();
        self.dw(
            "?[ns, src, rel, dst, created, p, decided_by] <- $rows :put kleos_mem_link {ns, src, rel, dst => created, p, decided_by}",
            vec![("rows", DataValue::List(rows))],
        )?;
        Ok(())
    }

    // ------------------------------------------------------------- worker

    /// Claim and run one document ingestion workflow. Returns whether one ran.
    pub fn docs_work_once(&'s self, worker: &str, state: Option<&DocsState>) -> Result<bool> {
        self.flow_ensure()?;
        self.docs_ensure()?;
        let claim = ClaimOpts {
            worker: Some(worker.to_string()),
            queues: Some(vec![DOC_QUEUE.to_string()]),
            modes: Some(vec!["code".to_string()]),
            workflows: Some(vec![DOC_WORKFLOW.to_string()]),
            lanes: None,
            lease_ms: Some(LEASE_MS),
        };
        let Some(c) = self.flow_claim(&claim)? else { return Ok(false) };
        let (id, token) = (c.id.clone(), c.token);
        let recorded: BTreeMap<String, JsonValue> = c
            .history
            .iter()
            .filter(|e| e.ty == "step")
            .filter_map(|e| e.key.clone().map(|k| (k, e.data.clone())))
            .collect();
        let input: IngestInput = match serde_json::from_value(c.input.clone()) {
            Ok(i) => i,
            Err(e) => {
                self.flow_call("fail", json!({"id": id, "token": token, "error": format!("bad input: {e}"), "final": true}))?;
                return Ok(true);
            }
        };
        // background ingestion yields model time to interactive calls (chat, recall)
        match crate::sched::with_priority(crate::sched::Priority::Bulk, || self.doc_run(&id, token, &input, &recorded, state)) {
            Ok(out) => {
                self.flow_call("complete", json!({"id": id, "token": token, "output": out}))?;
            }
            Err(e) => {
                let msg = e.to_string();
                let attempts = c.history.iter().filter(|e| e.ty == "failed" || e.ty == "attempt_failed").count();
                let fin = attempts >= 4 || msg.contains("lost its lease") || msg.contains("fenc");
                if fin {
                    let at = now_us();
                    let _ = self.dw(
                        &self.doc_row_script(),
                        vec![("doc_rows", DataValue::List(vec![Self::doc_row(&input.ns, &input.doc, at, &input.name, &input.mime, &input.hash, input.pages, "error", 0, json!({"error": msg}))]))],
                    );
                }
                let _ = self.flow_call("fail", json!({"id": id, "token": token, "error": msg, "final": fin, "retry_in_ms": 2000}));
            }
        }
        Ok(true)
    }

    fn doc_run(&'s self, id: &str, token: i64, input: &IngestInput, recorded: &BTreeMap<String, JsonValue>, state: Option<&DocsState>) -> Result<JsonValue> {
        let src = match &input.path {
            Some(p) => {
                let b = std::fs::read(p).map_err(|e| miette!("{p}: {e}"))?;
                if sha(&[&b]) != input.hash {
                    bail!("{p} changed while it was being ingested");
                }
                Source::Bytes(Arc::new(b))
            }
            None => {
                let r = self.dq(&format!("?[part, bytes] := *{DOC_BLOB}{{doc: $doc, part, bytes}}"), vec![("doc", s(&input.doc))])?;
                let mut parts: Vec<(i64, Vec<u8>)> = r.rows.iter().filter_map(|row| Some((row[0].get_int()?, row[1].get_bytes()?.to_vec()))).collect();
                parts.sort_by_key(|p| p.0);
                Source::Bytes(Arc::new(parts.into_iter().flat_map(|p| p.1).collect()))
            }
        };
        let o = &input.opts;
        let batch = o.batch_pages.unwrap_or(DEFAULT_BATCH_PAGES).max(1);
        let copts = o.chunker.clone().unwrap_or_default();
        let prev: BTreeSet<&str> = input.previous.iter().map(|x| x.as_str()).collect();
        let mut all_ids: Vec<String> = vec![];
        let mut n0: i64 = 0;
        let mut seen: BTreeMap<String, usize> = BTreeMap::new();
        let mut totals: BTreeMap<String, f64> = BTreeMap::new();
        let hb = |id: &str| {
            let _ = self.flow_call("heartbeat", json!({"id": id, "token": token, "lease_ms": LEASE_MS}));
        };
        for from in (0..input.pages).step_by(batch) {
            let key = format!("pages:{from}");
            if let Some(v) = recorded.get(&key) {
                // replayed: the batch committed before (with its chunks)
                for x in v.get("ids").and_then(|x| x.as_array()).into_iter().flatten() {
                    if let Some(s) = x.as_str() {
                        all_ids.push(s.to_string());
                    }
                }
                n0 = v.get("n_end").and_then(|x| x.as_i64()).unwrap_or(n0);
                for (h, c) in v.get("seen").and_then(|x| x.as_object()).into_iter().flatten() {
                    seen.insert(h.clone(), c.as_u64().unwrap_or(1) as usize);
                }
                continue;
            }
            hb(id);
            let to = (from + batch).min(input.pages);
            let t0 = Instant::now();
            let id_s = id.to_string();
            let last_hb = Mutex::new(Instant::now());
            let on_page = |_: usize| {
                if let Some(st) = state {
                    st.pages_done.fetch_add(1, Ordering::Relaxed);
                }
                let mut l = last_hb.lock().unwrap();
                if l.elapsed().as_millis() as i64 > LEASE_MS / 4 {
                    hb(&id_s);
                    *l = Instant::now();
                }
            };
            let (pb, timings) = extract_pages(&src, input.kind, from, to, o, &on_page)?;
            let extract_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let t1 = Instant::now();
            let mut seen_b = seen.clone();
            let rows = chunk_pages(&input.doc, &pb, &copts, n0, &mut seen_b);
            let chunk_ms = t1.elapsed().as_secs_f64() * 1000.0;
            let new_rows: Vec<ChunkRow> = rows.iter().filter(|r| !prev.contains(r.id.as_str())).map(clone_row).collect();
            let t2 = Instant::now();
            let vecs = self.embed_chunks(&new_rows)?;
            let embed_ms = t2.elapsed().as_secs_f64() * 1000.0;
            let t3 = Instant::now();
            let at = now_us();
            let scripts = self.chunk_scripts(&input.ns, &input.doc, &new_rows, &vecs, at);
            let ids: Vec<String> = rows.iter().map(|r| r.id.clone()).collect();
            let n_end = n0 + rows.len() as i64;
            let mut stage = timings.clone();
            stage.insert("extract_ms".into(), extract_ms);
            stage.insert("chunk_ms".into(), chunk_ms);
            stage.insert("embed_ms".into(), embed_ms);
            let result = json!({"ids": ids, "n_end": n_end, "new": new_rows.len(), "seen": seen_b, "pages": [from, to], "ms": stage});
            self.flow_commit_scripts(id, token, &key, "pages", &scripts, &result)?;
            let store_ms = t3.elapsed().as_secs_f64() * 1000.0;
            for (k, v) in stage {
                *totals.entry(k).or_insert(0.0) += v;
            }
            *totals.entry("store_ms".into()).or_insert(0.0) += store_ms;
            all_ids.extend(ids);
            n0 = n_end;
            seen = seen_b;
            let left = FAIL_AFTER.load(Ordering::SeqCst);
            if left >= 0 {
                if left <= 1 {
                    FAIL_AFTER.store(-1, Ordering::SeqCst);
                    bail!("injected failure after a committed batch");
                }
                FAIL_AFTER.store(left - 1, Ordering::SeqCst);
            }
        }
        // finish: the old version's chunks that did not come back end; the
        // document row is done; the blob goes
        let at = now_us();
        let keep: BTreeSet<&str> = all_ids.iter().map(|x| x.as_str()).collect();
        let gone: Vec<String> = input.previous.iter().filter(|x| !keep.contains(x.as_str())).cloned().collect();
        let mut scripts = self.retract_scripts(&input.ns, &gone, at);
        scripts.push((
            self.doc_row_script(),
            params(vec![(
                "doc_rows",
                DataValue::List(vec![Self::doc_row(&input.ns, &input.doc, at, &input.name, &input.mime, &input.hash, input.pages, "done", all_ids.len() as i64, json!({"kind": format!("{:?}", input.kind).to_lowercase(), "removed": gone.len()}))]),
            )]),
        ));
        scripts.push((format!("?[doc, part] := *{DOC_BLOB}{{doc, part}}, doc = $doc :rm {DOC_BLOB} {{doc, part}}"), params(vec![("doc", s(&input.doc))])));
        let out = json!({"doc": input.doc, "chunks": all_ids.len(), "removed": gone.len(), "ms": totals});
        if recorded.get("finish").is_none() {
            self.flow_commit_scripts(id, token, "finish", "finish", &scripts, &out)?;
        }
        Ok(out)
    }

    /// Whether document workflows are waiting (for starting a worker).
    pub fn docs_pending(&'s self) -> Result<bool> {
        {
            let tx = self.transact()?;
            if tx.get_relation(FLOW_TASK, false).is_err() {
                return Ok(false);
            }
        }
        let r = self.dq(&format!("?[wf] := *{FLOW_TASK}{{wf, queue}}, queue = 'docs' :limit 1"), vec![])?;
        Ok(!r.rows.is_empty())
    }

    /// `::doc <op>`: get, list, delete, reingest.
    pub fn doc_call(&'s self, op: &str, args: JsonValue) -> Result<JsonValue> {
        self.docs_ensure()?;
        let ns = args.get("ns").and_then(|x| x.as_str()).unwrap_or("default").to_string();
        let doc = args.get("doc").and_then(|x| x.as_str()).map(|x| x.to_string());
        match op {
            "list" => {
                let r = self.dq(
                    &format!("?[doc, name, mime, pages, status, chunks, ingested] := *{DOC}{{ns: $ns, doc, name, mime, pages, status, chunks, ingested @ 'NOW'}}"),
                    vec![("ns", s(&ns))],
                )?;
                Ok(JsonValue::Array(
                    r.rows
                        .iter()
                        .map(|row| json!({"doc": cell_str(&row[0]), "name": cell_str(&row[1]), "mime": cell_str(&row[2]), "pages": cell_i(&row[3]), "status": cell_str(&row[4]), "chunks": cell_i(&row[5]), "ingested": cell_i(&row[6])}))
                        .collect(),
                ))
            }
            "get" => {
                let doc = doc.ok_or_else(|| miette!("::doc get needs 'doc'"))?;
                let r = self.dq(
                    &format!("?[name, mime, hash, pages, status, chunks, meta, ingested] := *{DOC}{{ns: $ns, doc: $doc, name, mime, hash, pages, status, chunks, meta, ingested @ 'NOW'}}"),
                    vec![("ns", s(&ns)), ("doc", s(&doc))],
                )?;
                let Some(row) = r.rows.first() else { return Ok(JsonValue::Null) };
                Ok(json!({"doc": doc, "name": cell_str(&row[0]), "mime": cell_str(&row[1]), "hash": cell_str(&row[2]), "pages": cell_i(&row[3]),
                          "status": cell_str(&row[4]), "chunks": cell_i(&row[5]), "meta": JsonValue::from(row[6].clone()), "ingested": cell_i(&row[7])}))
            }
            "delete" => {
                let doc = doc.ok_or_else(|| miette!("::doc delete needs 'doc'"))?;
                let ids = self.doc_chunk_ids(&ns, &doc)?;
                let erase = args.get("mode").and_then(|x| x.as_str()) == Some("erase");
                if erase {
                    if !ids.is_empty() {
                        self.forget(crate::memory::ForgetOpts { ns: Some(ns.clone()), ids: ids.clone(), mode: Some("erase".into()), ..Default::default() })?;
                    }
                    self.dw(&format!("?[ns, id] := *{DOC_CHUNK}{{ns, id, doc}}, ns = $ns, doc = $doc :rm {DOC_CHUNK} {{ns, id}}"), vec![("ns", s(&ns)), ("doc", s(&doc))])?;
                    self.dw(&format!("?[ns, doc, vt] := *{DOC}{{ns, doc, vt}}, ns = $ns, doc = $d :rm {DOC} {{ns, doc, vt}}"), vec![("ns", s(&ns)), ("d", s(&doc))])?;
                } else {
                    let at = now_us();
                    self.run_scripts_tx(&self.retract_scripts(&ns, &ids, at))?;
                    if let Some((hash, _, _, pages, name)) = self.doc_current(&ns, &doc)? {
                        self.dw(&self.doc_row_script(), vec![("doc_rows", DataValue::List(vec![Self::doc_row(&ns, &doc, at, &name, "", &hash, pages as usize, "deleted", 0, JsonValue::Null)]))])?;
                    }
                }
                Ok(json!({"doc": doc, "chunks": ids.len(), "mode": if erase {"erase"} else {"retract"}}))
            }
            "reingest" => bail!("::doc reingest: pass the document to `remember` again (same name): unchanged chunks are kept"),
            other => bail!("unknown doc op '{other}': expected get, list or delete"),
        }
    }
}

fn clone_row(r: &ChunkRow) -> ChunkRow {
    ChunkRow {
        id: r.id.clone(),
        n: r.n,
        text: r.text.clone(),
        page: r.page,
        page_end: r.page_end,
        boxes: r.boxes.clone(),
        heading: r.heading.clone(),
        start: r.start,
        end: r.end,
        hash: r.hash.clone(),
        label: r.label.clone(),
    }
}
