# Attaching documents and citing pages

Documents go through `remember` like any statement: attach them, and recall
searches them with your memories, citing document, page and section.

Script: [`examples/ts/documents.ts`](../../../../examples/ts/documents.ts).
The embedded store needs the native module built with the `docs` feature (a
server needs the same feature).

## Attach

<!-- from examples/ts/documents.ts -->
```ts
const r = await kleos.remember('Here is the support handbook.', {
  ns: 'acme',
  attachments: [{ bytes: new TextEncoder().encode(handbook), mime: 'text/markdown', name: 'handbook.md' }],
});
const doc = r.documents![0];
console.log('document:', doc.name, doc.status, `${doc.chunks} chunks`);
```

Attachments can be bytes (`{ bytes, mime, name }`), base64 (`{ data_b64, mime,
name }`) or a path the store's process can read (`'report.pdf'`). The type is
detected from the bytes: PDF, PNG/JPEG/TIFF/WebP (OCR), Markdown, HTML, DOCX,
plain text.

## Big documents run in the background

A PDF of more than 4 pages comes back with `status: 'running'`. A worker thread
starts on its own and ingests it in page batches; each batch is searchable as
soon as it's saved, and a crash resumes where it stopped. Poll if you need to
wait:

<!-- from examples/ts/documents.ts -->
```ts
let status = doc.status;
while (status === 'running') {
  await new Promise((res) => setTimeout(res, 500));
  status = (await kleos.docs('get', { ns: 'acme', doc: doc.doc })).status;
}
```

Background ingestion runs in the bulk lane, so recall stays fast while it
works.

## Recall with citations

<!-- from examples/ts/documents.ts -->
```ts
const ctx = await kleos.recallContext('How long do enterprise customers have to ask for a refund?', { ns: 'acme' });
console.log(ctx.text.split('\n').filter((l) => l.includes('[handbook.md')).join('\n'));
```

```
- [2026-10-24 (Sat)] # Support handbook ## Refunds Refunds are possible within 30 days ... [handbook.md, § Support handbook › Refunds]
```

List hits carry the same provenance in `hit.doc` (`name`, `page`,
`page_end`, `heading`). `sources: ['docs']` returns only document chunks, and
`sources: ['memory']` leaves them out.

## Re-uploading

The same bytes again do nothing and return the existing document. A changed
file under the same name becomes a new version: unchanged chunks are kept,
and only new ones are embedded. `docs('list', { ns })`, `docs('get', …)` and
`docs('delete', …)` manage documents.
