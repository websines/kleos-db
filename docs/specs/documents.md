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
(`OcrBackend`) so another engine can be added; Indra is the default and the
only one shipped.

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

Tokens are counted with a pluggable counter: characters (Chonkie's default,
used for the parity test), words, or the embedder's tokenizer. Defaults:
`recursive`, 384 tokens by the embedder tokenizer (else ~4 characters per
token), no overlap.

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
- a URL (`{"url": "https://…"}`) is fetched only with the `docs-url` feature
  (off by default).

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

`::doc get {ns, doc}`, `::doc list {ns}`, `::doc delete {ns, doc, mode?}`
(retract, or erase with its chunks) and `::doc reingest {ns, doc}` inspect and
manage documents. None of them is needed to use documents.

## Results

See the "Results" section filled in at the end of M3 (parity, speed, the
big-PDF benchmark, the kill test and recall quality).
