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

### Speed work (wave 6): native kernels, digital fast path, streaming

*Measured 2026-10-24 on the M3 while another agent's builds and benchmarks
kept the load average at 15–26 for most runs, so absolute times are
inflated; the Rust/Python pairs below ran back to back under the same load.*

**1. Hand-written CPU path for Indra's nets** (`kleos-docs/src/ocr/fastnn.rs`,
`convnet.rs`). The recognizers run channels-last with every 1×1 convolution,
stem and the classifier as one GEMM (Accelerate/AMX on macOS,
`matrixmultiply` elsewhere), vectorized depthwise convolutions, and
HardSwish, the masked squeeze-excite, the residual and the width mask fused
around them; top-k for the LM is computed natively. The detector and the
orientation model run on a small NCHW interpreter of their op set
(pointwise GEMM, im2col dense conv, direct depthwise, Conv+HardSwish fused,
in-place elementwise, nearest Resize, DepthToSpace, …). Weights are read from
the same ONNX files by node name; a graph the native code does not implement
falls back to tract, and `KLEOS_OCR_TRACT=1` forces tract (the reference).

| Graph, input | tract | native | ONNX Runtime (earlier) | outputs vs tract |
|---|---|---|---|---|
| `rec.onnx`, 35×1×96×656 (greedy) | ~500 ms | **70 ms** | 180 ms | ids and top-k indices identical; floats within 8e-5 |
| `rec.onnx`, full (with top-k) | 595 ms | 105 ms | — | as above |
| `rec_cjk.onnx`, 30×1×32×512 | 1,345 ms | 155 ms | — | ids identical |
| `det.onnx`, 1×1×1024×784 | 194 ms | **54 ms** | ~49 ms | max \|Δ\| 5e-5 (of 40) |
| `orient.onnx`, 1×1×768×768 | 93 ms | **21 ms** | ~40 ms | max \|Δ\| 6e-8 |

Parity: identical lines on the 27 fixture pages (`ocr_parity`) and on the 20
sampled Zenobia pages. Per scanned page, single-threaded (load ≈ 5): recognizer
model time 759 → ~65 ms; the page ~1,185 → ~600 ms (Python indra-ocr: 739). The
remaining per-page time is the orientation preview and skew search (~70–110
ms, image ops), crops and routing (~45 ms), the render (~80 ms) and the
detector.

Whole scanned book (Zenobia, 500 pages, OCR only, 4 workers), back to back
at load ≈ 18–21:

| | pages/s | peak RSS |
|---|---|---|
| Python indra-ocr `convert_many(cores=4)` | 2.22 | 1.06 GB |
| **Rust, native kernels** | **3.51** (1.58×) | 1.29 GB |
| Rust before (tract, earlier at load ≈ 5) | 2.36 | 2.1 GB |
| Python earlier at load ≈ 5 | 5.13 | 0.95 GB |

**2. Digital-page geometry fast path** (`Converter::simple_page`,
`geometry_page`). A page whose text layer is trusted, upright, horizontal,
Latin and readable, with ≤ 40 path objects, < 1% image area and no shading,
whose lines share the left margin and mostly run the full width, and with no
wide gap inside a baseline band, is read from the layer's geometry alone:
runs are cut where their characters leave the line (a run can carry a
hyphenated word across a break), merged per band, and pieces of one content
object join without a space, as the detector path does. No render, no
detector. `KLEOS_DOC_FAST=0` (or `Converter::fast_digital = false`) turns it
off; the Indra parity test runs the full path.

| Set | pages | fast | word-identical to the full path | mean word ratio | ms/page full → with fast path |
|---|---|---|---|---|---|
| 9/11 Report, first 120 pages | 120 | 94 | 82 | 0.998 | 111 → 33 |
| Fixtures: form (`irs_fw4`), ruled table | 6 | 0 | — | — | (full path kept) |
| Fixtures: two-column | 2 | 1 | 1 | 1.000 | |

The differing pages are tables of contents and dialogue lists whose reading
order differs (page numbers or speaker lines placed before/after a line); no
words are lost.

**3. Streaming ingestion** (`docs.rs`, `Converter::convert_doc_streaming`). A
PDF's pages come from one streaming extraction (the document opened once,
page workers on their own pdfium instances, a bounded queue so a slow
consumer holds the workers back) feeding chunk → embed → commit; there is no
barrier at batch edges and embedding a batch overlaps the OCR of the next.
Each batch still commits as its own durable step with the same keys and
results, so replay and resume are unchanged (new test
`pdf_streaming_ingestion_resumes`: a streamed PDF cut after two committed
batches resumes to the clean run's chunks).

**End to end** (`bigpdf_bench.py`, Metal embedder, `KLEOS_MEMORY_BUDGET=6G`):

| | before (load 5–9) | after |
|---|---|---|
| Zenobia (scanned): done / pages/s | 590 s / 0.85 | **217 s / 2.31** (load 2–20) |
| Zenobia: `remember` returns / first chunk | 1.28 s / 1.3 s | 1.41 s / 1.4 s |
| Zenobia: peak RSS | 2.78 GB | 1.86 GB |
| Zenobia kill at 32/63 batches | resumed, 802 chunks, exact | resumed, 802 chunks, no duplicates, contiguous |
| 9/11 (digital): done / pages/s | 425 s / 1.38 | 456 s / 1.28 (EmbeddingGemma 2, load ≈ 20); 481 s / 1.22 (v1) |
| 9/11: extraction (render, detector, text, OCR) | ~204 s | **~13–17 s** (detector 265 → ~20 ms/page) |
| 9/11: embedding | 221 s | 387 s (v2) / 410 s (v1), under load ≈ 20 |
| Recall hit@5, 10 questions each | 10/10, 10/10 | 10/10, 10/10 |

The digital book is now bound by the embedder: extraction is ~15× cheaper,
but embedding its 1,660 chunks took ~400 s under the concurrent load (221 s
before at load 5–9; the default embedder also became EmbeddingGemma 2, ~1.8×
slower per text). That is the next lever for digital documents (EmbeddingGemma
2 speed is a separate item).

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

### Big PDFs, end to end (`scripts/docs/bigpdf_bench.py`)

Two books went through `remember` in fresh stores. Each ingested as a durable
background workflow in batches of 8 pages, with 4 page workers, the Metal
embedder and `KLEOS_MEMORY_BUDGET=6G`. Load average was about 5–9. Sources
and licences are in `.bench-data/docs/bigpdf/SOURCES.md`:
- *The 9/11 Commission Report*: govinfo.gov, US government work, public domain.
- *Zenobia, Queen of Palmyra*: Internet Archive scan, pre-1929, public domain.

| | 9/11 Report (digital, 585 pp.) | Zenobia (scanned, 500 pp.) |
|---|---|---|
| `remember` returned | 0.39 s (`running`) | 1.28 s (`running`) |
| first chunk recallable | 0.7 s | 1.3 s |
| done | 425 s (**1.38 pages/s**) | 590 s (**0.85 pages/s**) |
| chunks | 1,663 | 802 |
| page kinds | 585 text layer | 500 OCR (the IA text layer is invisible, so OCR runs) |
| extraction wall (pages, layout, chunking) | 87 s | 414 s |
| embedding wall | 221 s | 99 s |
| peak RSS | 1.76 GB | 2.78 GB |
| recall hit@5 (10 questions with known pages) | **10/10** | **10/10** |

Summed per-page stage times, across the four workers (seconds):

| Stage | 9/11 Report | Zenobia |
|---|---|---|
| render | 4.9 | 120 |
| text layer | 4.1 | 0 |
| orientation | 3.4 | 283 |
| detection | 265 | 187 |
| recognition | 2.1 | 934 |
| reading order | 4.5 | 2.4 |

**Bottlenecks:**
- **Digital books: embedding.** EmbeddingGemma takes about 0.13 s per
  1,500-character chunk on Metal and about 1.35 s on CPU. The same 9/11 run
  with the CPU embedder took 2,462 s, with 2,240 s of it embedding.
- **Scans: recognition.** See "OCR speed" above.
- **Batch barrier.** In the pipeline, Zenobia's OCR ran at 1.2 pages/s
  against 2.36 when it runs alone: a batch of 8 pages on 4 workers leaves
  workers idle at each batch boundary, and embedding runs between batches,
  not alongside OCR.

**Next steps for the pipeline:**
- larger batches;
- embedding that overlaps the next batch's OCR;
- a faster CPU path for the models (see "OCR speed").

**Kill and resume (Zenobia):**
- The ingesting process was killed with SIGKILL after 32 of 63 committed
  batches, 285 s in, with 412 chunks stored.
- A new process opened the store, and its first `recall` started the
  worker. The first new batch committed 50 s later: the 30 s lease plus
  model loads.
- The book was done 306 s after reopening.
- Result: 802 chunks, the same as the uninterrupted run. 802 provenance rows
  and 802 memories, chunk numbers contiguous from 0, no duplicate ids, each
  of the 63 batch steps recorded once.

**Bug found and fixed:** a bounded change subscription that nobody reads
used to block writers. The document worker's own wake-up subscription
filled up with its own heartbeats, and the 9/11 run hung. Delivery is now
non-blocking: a full bounded subscription misses changes and gets a
one-time `callback.lagged` diagnostic. This is covered by
`callback_commit_contract::full_bounded_subscription_never_blocks_writers`
and `tests/slow_subscriber.rs`.

### Starting point for a faster CPU path (recognizer)

`rec.onnx` takes `x [B, 1, 96, W]` (a three-line-tall window), `widths
[B]`, and outputs `ids, conf, lengths, blank, topv, topi` over T = W/4
steps; the vocabulary is 9,585 tokens.
- **Typical page:** about 40 lines in 3–4 width batches, the largest being
  **35×1×96×656** (B ≤ 64 and B·W ≤ 40,960).
- **Graph:**
  - stem: a 32×4 stride-(32,4) convolution to 128 channels;
  - context branch: a 4×4 patch convolution, depthwise 3×3, pointwise 32→64,
    and softmax pooling over the height, projected to 128;
  - 10 MBConv-1D blocks: depthwise k=9 at 128 channels, pointwise expand
    128→256, HardSwish, a masked squeeze-excite 256→64→256, project
    256→128, residual, and a width mask;
  - classifier: 1×1 to 9,585, softmax, argmax.

tract's profile of the greedy graph on 35×96×656 (total 492 ms; ONNX Runtime
takes 180 ms single-threaded):

| op | ms | calls |
|---|---|---|
| OptMatMul | 162 | 57 |
| MoveAxis | 103 | 46 |
| Im2col | 66 | 13 |
| Add | 44 | 38 |
| Mul | 43 | 33 |
| Softmax | 23 | 1 |
| Reduce Max / ArgMax / Sum | 25 | 15 |

The full graph adds TopK, 400 ms on the same batch. That is why it now runs
only for batches the LM reads. Orientation (`orient.onnx`, 1×1×768×768, 94
ms against ONNX Runtime's roughly 40) and detection (`det.onnx`, 1×1×1024×W,
115 ms against 49) share the depthwise-convolution pattern.
`kleos-docs/examples/tract_prof.rs` reproduces these profiles.
