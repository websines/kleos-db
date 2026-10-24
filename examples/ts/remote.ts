// Serverless and edge apps: the HTTP client is pure TypeScript (fetch only),
// so it runs where native modules can't. Same API as the embedded store.
import { connect } from 'kleos/remote';
import { startServer } from './_util.ts';

// in your app: connect({ url: process.env.KLEOS_URL!, token: process.env.KLEOS_TOKEN })
const server = process.env.KLEOS_URL ? { url: process.env.KLEOS_URL, stop: () => {} } : await startServer();
const kleos = connect({ url: server.url, token: process.env.KLEOS_TOKEN });

await kleos.init();
await kleos.remember('The quarterly review is on Friday at 10.', { ns: 'team' });
const ctx = await kleos.recallContext('When is the quarterly review?', { ns: 'team' });
console.log(ctx.text.split('\n').filter((l) => l.includes('Friday')).join('\n'));

// a handler in a serverless function would look like this
async function handler(req: { query: string }) {
  const block = await kleos.recallContext(req.query, { ns: 'team', budget: 400 });
  return { context: block.text, tokens: block.tokens };
}
console.log('handler tokens:', (await handler({ query: 'review' })).tokens);

server.stop();
console.log('ok: remote');
