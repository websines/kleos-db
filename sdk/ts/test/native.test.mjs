import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { open, KleosError, nativeFeatures } from '../dist/index.js';
import { tempDir } from './helpers.mjs';

test('embedded: remember, recall, context, forget', async () => {
  const k = open({ path: join(tempDir(), 'a.db') });
  assert.equal(k.kind, 'embedded');
  const cfg = await k.init();
  assert.equal(typeof cfg.version, 'number');
  const r = await k.remember('I moved to Lisbon in March. My sister Anna lives in Munich.', { ns: 'u', entities: ['user', 'Anna'] });
  assert.equal(r.action, 'added');
  assert.equal(r.claims.length, 2);
  const hits = await k.recall('Munich', { ns: 'u', k: 3 });
  assert.ok(Array.isArray(hits) && hits.length >= 1);
  assert.match(hits[0].text, /Munich/);
  assert.equal(typeof hits[0].valid_from, 'number');
  const ctx = await k.recallContext('Anna', { ns: 'u', budget: 300 });
  assert.match(ctx.text, /Anna/);
  assert.ok(ctx.tokens <= 300);
  const f = await k.forget(hits[0].id, { ns: 'u' });
  assert.deepEqual(f.ids, [hits[0].id]);
  const after = await k.recall('Munich', { ns: 'u' });
  assert.ok(!after.some((h) => h.id === hits[0].id));
  await k.close();
});

test('embedded: camelCase options become snake_case (asOf reads the past)', async () => {
  const k = open();
  await k.init();
  await k.remember('The office is on floor 3.', { ns: 'o' });
  const t = Date.now() * 1000;
  await k.forget(null, { ns: 'o', all: true });
  assert.equal((await k.recall('office', { ns: 'o' })).length, 0);
  const past = await k.recall('office', { ns: 'o', asOf: t });
  assert.equal(past.length, 1);
  await k.close();
});

test('embedded: branches fork/diff/merge/discard', async () => {
  const k = open();
  await k.init();
  await k.remember('Plan A is the default.', { ns: 'm' });
  const fk = await k.fork('m', 'm2');
  assert.equal(fk.parent, 'm');
  await k.remember('Plan B is being tested.', { ns: 'm2' });
  const d = await k.diff('m', 'm2');
  assert.equal(d.added.length, 1);
  const prev = await k.merge('m2', 'm', { dryRun: true });
  assert.ok(prev.statements >= 1);
  await k.merge('m2');
  assert.ok((await k.recallContext('Plan B', { ns: 'm' })).text.includes('Plan B'));
  await k.discard('m2');
  const bs = await k.branches({ all: true });
  assert.ok(bs.some((b) => b.ns === 'm2' && b.status === 'discarded'));
  await k.close();
});

test('embedded: ontology, profile, run, flow', async () => {
  const k = open();
  await k.init();
  const list = await k.ontology('list');
  assert.ok(list.ontologies.some((o) => o.onto === 'personal'));
  await k.remember('Acme Corp is on the Enterprise plan.', { ns: 's', entities: ['Acme Corp'] });
  const p = await k.profile('Acme Corp', { ns: 's' });
  assert.match(p.text, /Enterprise/);
  const rows = await k.run('?[x] <- [[1], [$v]]', { v: 'two' });
  assert.deepEqual(rows.rows, [[1], ['two']]);
  await k.flow('define', { name: 'w', graph: { steps: [{ return: '1 + 1' }] } });
  await k.flow('start', { workflow: 'w', id: 'w1' });
  await k.flow('tick', {});
  assert.equal((await k.flow('status', { id: 'w1' })).output, 2);
  await k.close();
});

test('embedded: errors are KleosError with the op; closed store rejects', async () => {
  const k = open();
  await assert.rejects(k.memory('nope'), (e) => e instanceof KleosError && e.op === 'memory.nope' && /unknown memory op/.test(e.message));
  await k.close();
  await assert.rejects(k.recall('x'), (e) => e instanceof KleosError && /closed/.test(e.message));
});

test('embedded: calls never block the event loop', async () => {
  const k = open();
  await k.init();
  let ticks = 0;
  const iv = setInterval(() => ticks++, 1);
  // a deliberately heavy query
  const t0 = Date.now();
  await k.run('?[count(x)] := x in int_range(600000)');
  const ms = Date.now() - t0;
  clearInterval(iv);
  assert.ok(ms < 20 || ticks > 5, `event loop ticked ${ticks} times during a ${ms} ms query`);
  await k.close();
});

test('embedded: documents through remember (docs build)', { skip: !nativeFeatures().includes('docs') && 'native module built without docs' }, async () => {
  const k = open({ path: join(tempDir(), 'd.db') });
  await k.init();
  const md = '# Handbook\n\n## Refunds\n\nRefunds within 30 days.\n';
  const r = await k.remember('handbook attached', { ns: 'd', attachments: [{ bytes: new TextEncoder().encode(md), mime: 'text/markdown', name: 'h.md' }] });
  assert.equal(r.documents[0].status, 'done');
  const hits = await k.recall('refunds', { ns: 'd', sources: ['docs'] });
  assert.equal(hits[0].doc.name, 'h.md');
  const list = await k.docs('list', { ns: 'd' });
  assert.ok(JSON.stringify(list).includes('h.md'));
  await k.close();
});
