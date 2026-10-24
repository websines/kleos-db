# Documents

*Status: M1 (OCR port) and M2 (chunking) built in `kleos-docs`; M3 (documents
in the memory API) built in `kleos-core` behind the `docs` feature. Branch
`wave5/docs`.*

Documents go through the memory API like everything else. An agent passes a
file to `remember` and asks `recall` questions; kleos reads the file (its PDF
text layer, or OCR), lays it out, chunks it, embeds the chunks and stores
them as memories with page-level provenance. Recall searches memories and
document chunks together and cites the page. There is no separate ingestion
step to call.

```python
db.remember("Here is the contract we signed", attachments=["contract.pdf"])
# -> {"id": "m_…", "documents": [{"doc": "doc_9f2c…", "status": "running", "pages": 212}]}
db.recall("What is the termination notice period?", format="context")
# … "Either party may terminate with 90 days' notice." [contract.pdf p. 14, § 11 Termination]
```

## Pieces

| Piece | Where | What |
|---|---|---|
| OCR | `kleos-docs::ocr` | A Rust port of the Indra OCR pipeline (datalab-to/indra, Apache-2.0): orientation and skew, line detection, CTC recognition with script routing and an n-gram LM beam search, reading order, layout blocks, Markdown with HTML tables. Its ONNX graphs run on tract (pure Rust; no ONNX Runtime). |
| PDF | `kleos-docs::pdf` | pdfium (BSD-3/Apache-2.0) loaded at run time: grayscale renders, the text layer with character boxes and angles. Digital pages use their text layer (no OCR); scans and broken layers go to OCR; mixed pages OCR only the lines the layer lacks. |
| Chunking | `kleos-docs::chunk` | Token, sentence, recursive (Markdown-, heading- and code-aware) and semantic chunkers, modelled on Chonkie (MIT). Every strategy respects layout blocks: tables are never split, headings stay with what follows, and each chunk carries its provenance. |
| Ingestion | `kleos-core::docs` | Type detection, the pipeline, embedding, storage, durable resume, re-ingestion diffs, citations. |

## OCR backend

`Ocr` is the pipeline; the model folder is fetched from Hugging Face
(`datalab-to/indra-ocr`, revision `v1.1.0`) into the Hugging Face cache on
first use, like the other built-in models (`HF_HUB_OFFLINE=1` forbids it;
`KLEOS_OCR_MODEL_DIR` points at a local folder). The backend is a trait
(`OcrBackend`: `read_page(gray, turn) -> Page`) so another engine can be
plugged in for full-OCR pages and images (`Converter::backend`); PDF
text-layer pages keep Indra's line detector to place the layer's
characters. Indra is the default and the only one shipped. There is no
licence gate at run time: enabling documents downloads the weights like the
other built-in models (the licence is stated here and in NOTICE.md).

**Licence of the weights.** The Indra code is Apache-2.0 and this port follows
it. The model weights are under a *modified OpenRAIL-M* licence: free for
research, personal use, and startups under $20M in funding or revenue, and not
for use that competes with Datalab's API. kleos downloads them; it never
bundles them. NOTICE.md says the same; users with other needs must license the
weights from Datalab or plug in another backend.

**Parity.** The port reproduces the Python package (indra-ocr 1.1.2, the same
models) exactly on the parity set: the same lines, text, boxes (within 0.2 px),
block labels and Markdown on every page. Image operations match OpenCV's
arithmetic (fixed-point linear resize, area weights, connected-component
order); see "Results".

**pdfium.** `pdfium-render` binds the shared library at run time. It is looked
up at `KLEOS_PDFIUM_LIB`, in `KLEOS_PDFIUM_DIR`, in the kleos cache
(`pdfium/<build>/lib`), then on the system path; `ensure_pdfium` downloads the
pinned pdfium-binaries build (`chromium/8076`, the one pypdfium2 ships) into
the cache. pdfium is not thread safe, so every pdfium call runs under one
process-wide lock; pages are rendered and their text layer read under the
lock (milliseconds), then OCR runs in parallel.

## Chunking

All chunkers share one input: the document as a sequence of **blocks** (a
layout block of a page: label, page, box, text; or a Markdown block for text
inputs) and one output: `Chunk {text, start, end, tokens, pages, boxes,
heading_path, blocks}`, with `start`/`end` byte offsets into the document's
Markdown.

| Strategy | Rule |
|---|---|
| `token` | Fixed windows of `chunk_size` tokens with `overlap`, as Chonkie's TokenChunker. |
| `sentence` | Sentences (delimiters `. ! ? \n`, short splits merged as Chonkie's `split_offsets`) packed greedily up to `chunk_size`, with sentence overlap. |
| `recursive` (default) | Chonkie's RecursiveChunker levels (paragraphs, sentences, pauses, words, tokens), applied per block group; Markdown headings start a new chunk and set the heading path; code fences and tables are atomic. |
| `semantic` | Sentences embedded with the store's embedder; a chunk ends where the similarity of adjacent windows drops below a threshold (Chonkie's SemanticChunker, simplified: no Savitzky–Golay smoothing), capped at `chunk_size`. |

Tokens are counted with a pluggable counter (`Tokenizer`): characters
(Chonkie's default, used for the parity test), words (`text.split(' ')`, as
Chonkie's word tokenizer), or an estimate of four characters per token
(`ApproxTokenizer`, the default for documents: no tokenizer has to load and
it matches the context block's own token estimate). Defaults: `recursive`,
384 estimated tokens (about 1,500 characters), no overlap.

Layout rules, every strategy: a `Table` block is one unit (a table longer than
`chunk_size` becomes its own chunk, oversized rather than split mid-row); a
`Section-Header` or Markdown heading never ends a chunk (it moves to the next
one) and updates `heading_path`; page headers and footers are dropped.

## The memory API

### `remember` with documents

`remember` takes documents in any of these forms (Rust, Python, HTTP):

- `remember({file: "path/to/x.pdf"})`, or `remember(text, attachments=["a.pdf", "b.png"])`;
- bytes with a mime type: `attachments=[{"bytes": b"...", "mime": "application/pdf", "name": "x.pdf"}]`
  (HTTP: `{"data_b64": "...", "mime": ..., "name": ...}` in the JSON body);
- a URL (`{"url": "https://…"}`) is refused for now (no network fetch from
  `remember`); pass the bytes or a path.

The type is detected from the bytes (PDF header, image magic numbers), then
the name: PDF, PNG/JPEG/TIFF/BMP/WebP, Markdown, HTML (tags stripped,
headings kept), plain text, DOCX (the `word/document.xml` paragraphs and
headings). Each document runs:

```
bytes → type → pages (text layer | OCR) → layout blocks → chunks → embed → store
```

**Small documents** (text formats, images, PDFs of at most `inline_pages`
pages, default 4) are ingested inside the `remember` call; the result lists
the document with `status: "done"` and its chunk count. **Larger documents**
start a durable workflow (`::flow`, workflow `doc.ingest`, queue `docs`) and
`remember` returns at once with `status: "running"`; an in-process worker
thread (started on demand, one per open store) ingests pages in batches, and
recall sees each batch's chunks as soon as it commits.

**Text with attachments.** The text is remembered as usual (claims,
supersession, entities). Each memory it creates gets a link
`kleos_mem_link {src: memory, rel: 'attached', dst: doc}` to each document,
and the documents' chunks get `source: "doc:<doc id>"`.

### Storage

| Relation | Key ⇒ value | Holds |
|---|---|---|
| `kleos_doc` | `{ns, doc, vt: Validity ⇒ name, mime, hash, pages, title, status, chunks, meta, ingested}` | One row per document version (valid time: a re-ingested document's old version ends). |
| `kleos_doc_chunk` | `{ns, id ⇒ doc, n, page, page_end, boxes, heading, start, end, hash, label}` | Provenance of a chunk memory. |
| `kleos_doc_blob` | `{doc, part ⇒ bytes}` | The original bytes of a document passed as bytes, kept until its ingestion finishes (resume needs them); a path input stores its path and hash instead. |
| `kleos_mem` | (existing) | Each chunk is a memory: `kind: 'doc'`, `source: 'doc:<doc>'`, role `other` (`kleos_mem_role`), text = the chunk's Markdown. Its vector goes to `kleos_mem_vec` and its text to the existing FTS index, so recall's hybrid search, ranking, diversity and branches apply unchanged. |

Chunk ids are `dc_<doc>_<chunk hash>`, without a dot, so each chunk is its
own statement for recall's diversity caps. Role `other` keeps chunks out of
supersession and duplicate decisions (as other speakers' memories already
are); chunks never enqueue Engram edits.

**Re-ingestion.** The same bytes (by SHA-256) in the same namespace are a
no-op that returns the existing document. Changed bytes under the same name
make a new version: chunks whose hash is unchanged keep their memory (no
re-embedding); removed chunks are retracted (valid time ends; `as_of` still
sees them); new chunks are added; the document row's old version ends.

### Durable ingestion

`doc.ingest` is a code-mode workflow run by a Rust worker inside kleos:

1. `meta`: the page count (and, for a path, the file hash, checked again on resume).
2. `pages:<from>` for each batch of `batch_pages` pages (default 8): render or
   read the text layer, OCR, chunk, embed — then **one transaction** writes the
   batch's chunk memories, vectors, provenance rows and the step's event,
   fenced by the worker's lease token. A crash before the commit leaves
   nothing; a resumed workflow replays the recorded batches and starts at the
   first unrecorded one, so no chunk is written twice or skipped.
3. `finish`: the document row's status and chunk count; the blob is dropped.

Chunks that cross a batch boundary are cut there (a batch is chunked on its
own pages; the carry-over of the last unfinished block into the next batch is
recorded in the step result).

### Recall

`recall` searches documents by default, fused with memories (they are
memories). `sources: ['memory', 'docs']` filters: `['memory']` leaves chunks
out, `['docs']` returns only chunks. A chunk hit carries `doc: {doc, name,
page, page_end, heading}`; the context output cites it:

```
- Either party may terminate with 90 days' notice. [contract.pdf p. 14, § 11 Termination]
```

### Admin

`::doc get {ns, doc}`, `::doc list {ns}` and `::doc delete {ns, doc, mode?}`
(retract, or erase with its chunks) inspect and manage documents (Rust
`DbInstance::doc_call`, Python `doc_call`, HTTP `POST /doc/:op`). To
re-ingest, pass the document to `remember` again under the same name. None
of them is needed to use documents.

## Results

*Measured 2026-10-24 on an M3 (4 performance + 4 efficiency cores, 16 GB)
with the store and models on a USB 2.0 SSD; other agents were sharing the
machine, so the load average (noted per run) was rarely below 4.*

### OCR parity with Indra (Python indra-ocr 1.1.2, same models)

| Set | Pages | Result |
|---|---|---|
| Generated fixtures (`scripts/docs/make_samples.py`: digital prose, two columns, a ruled table, Cyrillic, Greek, CJK, 8 scans, image-only PDFs, a 5-page IRS form) | 27 | **identical** lines, Markdown and block labels on every page; boxes within 0.2 px; same source (text / ocr / hybrid) chosen on all 27 |
| Zenobia (Internet Archive scan), 20 random pages | 20 | **identical** line text on all 20 (char F1 1.000) |

The parity test (`kleos-docs/tests/ocr_parity.rs`, ignored by default)
replays the fixture set against a Python reference. It still passes after
the speed work below.

### Chunking parity with Chonkie 1.7

72 cases (token, sentence and recursive chunkers; character and word
tokenizers; two sizes; six texts from the fixtures): **identical** chunks,
offsets and token counts (`kleos-docs/tests/chunk_parity.rs`).

### OCR speed (scanned book, 500 pages)

Per page, single-threaded, 20 sampled Zenobia pages (load ≈ 3):

| Stage (ms/page) | Python indra-ocr | Rust, first port | Rust, now |
|---|---|---|---|
| text layer + ink-check render | 49 | 68 | 1 (the 72-dpi render is made only when a check needs it) |
| page render (200 dpi) | 97 | 129 | 83 |
| orientation (skew search + model) | 57 | 214 | 160 (66 skew + 94 model) |
| line detection | 49 | 160 | 115 |
| recognition | 441 | 1,486 | 759 |
| **total** | **739** | **2,060** | **1,185** |

Whole book (`convert_many` / `convert_doc_parallel`, 4 workers, one
single-threaded OCR engine per page worker, load ≈ 5):

| | pages/s | wall | peak RSS |
|---|---|---|---|
| Python indra-ocr, `convert_many(cores=4)` | **5.13** | 97 s | 0.95 GB |
| Rust, first port (sequential, 4 intra-op threads) | 0.47 | — | — |
| Rust, now (4 page workers) | **2.36** | 212 s | 2.1 GB |

What changed:
- the recognizers run a greedy graph with the LM's top-k pruned away (tract
  spent 400 of 1,500 ms per batch on a full sort of the vocabulary); the
  full graph runs only for a batch the LM must read, so the output is
  unchanged;
- each page worker opens the PDF on its own pdfium instance, a separate
  copy of the library with its own global state, so renders and text layers
  run in parallel instead of behind one lock;
- the ink-check render is lazy; the OCR engine is single-threaded inside a
  page, and pages run in parallel.

What is left: the model inference itself. On the same 35-line batch, tract
takes 492 ms where ONNX Runtime takes 180 ms. tract's own profile puts 162 ms
in matrix products (already on Apple's AMX), 103 ms in axis moves and 66 ms in
im2col around the 1-D convolutions. Orientation and detection show the same
pattern on their depthwise 2-D convolutions. Rendering is about 7% of the
per-page time, so pdfium-free image decoding (the next item in the plan) would
gain at most that on these scans. The larger lever is a hand-written CPU path
for these small CNNs (pointwise convolutions as Accelerate GEMMs, depthwise
ones as shifted multiply-adds), with tract kept as the reference.

### End to end (`remember` and `recall` only)

`scripts/docs/docs_demo.py` (Metal embedder, load ≈ 6):
- **Mixed folder:** six attachments to one message (two digital PDFs, a
  scanned PDF, a Markdown file, a scanned Cyrillic image, and a 5-page form
  that ingests in the background): 85 s for `remember`, 91 s until the
  background document was done. The message's memory links to all six.
- **Questions:** all six were answered from the right document, with page
  citations.
- **Kill test:** the child process was killed with SIGKILL after two of
  five committed page batches. The reopened store resumed the workflow and
  finished in 35 s, which includes the 30 s lease wait. Result: 24 chunks,
  24 provenance rows and 24 memories, chunk numbers contiguous from 0, no
  duplicate ids, each step recorded once.
- **Re-ingestion:** a changed Markdown file kept 1 chunk, replaced 1 and
  re-embedded only the new one.
- **Peak RSS:** 0.96 GB.

