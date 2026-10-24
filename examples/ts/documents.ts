// Attaching documents and citing pages. Small documents ingest inside
// `remember`; big ones (PDFs over 4 pages) return at once with status
// "running" and ingest in the background (a worker thread starts on its
// own), recallable batch by batch.
import { open, nativeFeatures } from 'kleos';
import { join } from 'node:path';
import { tempDir } from './_util.ts';

if (!nativeFeatures().includes('docs')) {
  console.log('this native module was built without the `docs` feature; skipping');
  console.log('ok: documents');
  process.exit(0);
}

const kleos = open({ path: join(tempDir(), 'docs.db') });
await kleos.init();

const handbook = [
  '# Support handbook',
  '## Refunds',
  'Refunds are possible within 30 days of purchase. Enterprise plans have 60 days.',
  '## Escalation',
  'Escalate outages to the on-call engineer through the #incidents channel.',
].join('\n\n');

const r = await kleos.remember('Here is the support handbook.', {
  ns: 'acme',
  attachments: [{ bytes: new TextEncoder().encode(handbook), mime: 'text/markdown', name: 'handbook.md' }],
});
const doc = r.documents![0];
console.log('document:', doc.name, doc.status, `${doc.chunks} chunks`);

// a big PDF would come back "running": poll until it is done
let status = doc.status;
while (status === 'running') {
  await new Promise((res) => setTimeout(res, 500));
  status = (await kleos.docs('get', { ns: 'acme', doc: doc.doc })).status;
}

// recall searches documents and memories together; context lines cite the source
const ctx = await kleos.recallContext('How long do enterprise customers have to ask for a refund?', { ns: 'acme' });
console.log(ctx.text.split('\n').filter((l) => l.includes('[handbook.md')).join('\n'));

// only documents
const hits = await kleos.recall('escalate outages', { ns: 'acme', sources: ['docs'] });
console.log('top document hit:', hits[0].doc?.name, '/', hits[0].doc?.heading);

// the same bytes again do nothing
const again = await kleos.remember('The handbook again.', {
  ns: 'acme',
  attachments: [{ bytes: new TextEncoder().encode(handbook), mime: 'text/markdown', name: 'handbook.md' }],
});
console.log('re-upload:', again.documents?.[0]?.status);

await kleos.close();
console.log('ok: documents');
