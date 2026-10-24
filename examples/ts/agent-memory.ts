// The end-to-end tour: remember, recall a context block, attach a document,
// experiment in a branch and merge it back.
import { open, nativeFeatures } from 'kleos';
import { join } from 'node:path';
import { tempDir } from './_util.ts';

const kleos = open({ path: join(tempDir(), 'memory.db') });
await kleos.init();

// remember: a text of several sentences becomes one memory per claim
const r = await kleos.remember('I moved to Lisbon in March. My sister Anna lives in Munich.', {
  ns: 'user-42',
  entities: ['user', 'Anna'],
});
console.log('remember:', r.action, r.claims?.length ?? 1, 'claims');

// recall a ready-to-paste context block for the next prompt
const ctx = await kleos.recallContext('Where does Anna live?', { ns: 'user-42', budget: 500 });
console.log(ctx.text);

// attach a document (needs the native module built with the `docs` feature)
if (nativeFeatures().includes('docs')) {
  const notes = '# Trip notes\n\n## Munich\n\nAnna recommends the Viktualienmarkt on Saturdays.\n';
  const d = await kleos.remember('Notes from Anna for the Munich trip', {
    ns: 'user-42',
    attachments: [{ bytes: new TextEncoder().encode(notes), mime: 'text/markdown', name: 'trip.md' }],
  });
  console.log('document:', d.documents?.[0]?.name, d.documents?.[0]?.status);
  const hits = await kleos.recall('Viktualienmarkt', { ns: 'user-42', sources: ['docs'] });
  console.log('cited:', hits[0]?.doc?.name, hits[0]?.text.split('\n').pop());
}

// experiment in a branch, look at the difference, merge it back
await kleos.fork('user-42', 'user-42-try');
await kleos.remember('I started learning Portuguese.', { ns: 'user-42-try', entities: ['user'] });
const diff = await kleos.diff('user-42', 'user-42-try');
console.log('branch adds:', diff.added.map((m: { text: string }) => m.text));
const merged = await kleos.merge('user-42-try');
console.log('merged statements:', merged.replayed.length);

await kleos.close();
console.log('ok: agent-memory');
