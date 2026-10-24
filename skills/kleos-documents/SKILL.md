---
name: kleos-documents
description: Store and cite documents with kleos (PDF, scans, images, Markdown, HTML, DOCX). Use when the user shares or points at a file they want remembered or searched later, or asks about the contents of documents already stored.
---

# kleos documents

Documents become searchable memory: text layer or OCR, layout-aware chunks,
page citations. Re-adding the same file is free; a changed file only
re-embeds what changed.

## Add

```bash
kleos doc add ./contract.pdf --ns acme --text "Signed MSA with Globex"
# or together with a statement:
kleos remember "Here is the Q3 report; margins dropped." --attach ./q3.pdf --ns acme
```

MCP: `doc_add {path, ns}` (or `data_b64` + `mime` + `name` when the server
cannot read your files). Large documents ingest in the background: the
result says `running` (or returns a task); check with
`kleos doc status <doc> --ns acme` / MCP `doc_status`, and tell the user it is
still ingesting rather than waiting silently. Early pages are searchable
almost immediately.

## Ask

Use normal recall — documents are searched by default:

```bash
kleos recall "termination notice period" --ns acme
```

Cite what comes back, e.g. "90 days ([contract.pdf p. 14, § 11
Termination])". If a document and something the user said disagree, show
both with their dates instead of choosing.

## List

`kleos doc list --ns acme` (MCP `doc_status` without `doc`).
