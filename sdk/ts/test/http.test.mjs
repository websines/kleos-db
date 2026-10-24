import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { connect, KleosError } from '../dist/remote.js';
import { KLEOS_BIN, startServer } from './helpers.mjs';

const skip = !KLEOS_BIN && 'set KLEOS_BIN to the kleos binary';
let server;
before(async () => { if (!skip) server = await startServer(); });
after(() => server?.stop());

test('remote: same API over HTTP', { skip }, async () => {
  const k = connect({ url: server.url });
  assert.equal(k.kind, 'remote');
  await k.init();
  const r = await k.remember('I moved to Lisbon in March. My sister Anna lives in Munich.', { ns: 'u', entities: ['user', 'Anna'] });
  assert.equal(r.claims.length, 2);
  const hits = await k.recall('Munich', { ns: 'u' });
  assert.match(hits[0].text, /Munich/);
  const ctx = await k.recallContext('Anna', { ns: 'u' });
  assert.match(ctx.text, /Anna/);
  await k.fork('u', 'u2');
  await k.remember('I adopted a cat.', { ns: 'u2' });
  assert.equal((await k.diff('u', 'u2')).added.length, 1);
  const rows = await k.run('?[x] <- [[$v]]', { v: 7 });
  assert.deepEqual(rows.rows, [[7]]);
  assert.ok((await k.ontology('list')).ontologies.length >= 1);
  await k.flow('define', { name: 'w', graph: { steps: [{ return: '40 + 2' }] } });
  await k.flow('start', { workflow: 'w', id: 'w1' });
  await k.flow('tick', {});
  assert.equal((await k.flow('status', { id: 'w1' })).output, 42);
  const emb = await k.embedders();
  assert.ok(Array.isArray(emb.headers));
});

test('remote: errors carry status and op', { skip }, async () => {
  const k = connect({ url: server.url });
  await assert.rejects(k.memory('nope'), (e) => e instanceof KleosError && e.status === 400 && e.op === 'memory.nope');
  const bad = connect({ url: 'http://127.0.0.1:1' });
  await assert.rejects(bad.recall('x'), (e) => e instanceof KleosError && /request failed/.test(e.message));
});
