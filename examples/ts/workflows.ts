// A long-running workflow with a signal: it remembers a request, waits for a
// human approval (possibly days later, surviving restarts), then records it.
import { open } from 'kleos';
import { join } from 'node:path';
import { tempDir } from './_util.ts';

const kleos = open({ path: join(tempDir(), 'flows.db') });
await kleos.init();
const worker = kleos.flowWorker({ queues: ['default'], pollMs: 100 });

await kleos.flow('define', {
  name: 'refund',
  graph: {
    steps: [
      { name: 'asked', remember: { text: "=concat('Refund requested for order ', get($input, 'order'))", ns: 'ops' } },
      { name: 'decision', wait_signal: { name: 'approve', timeout_ms: 3 * 24 * 3600 * 1000 } },
      { remember: { text: "=concat('Refund for order ', get($input, 'order'), ' approved by ', get($decision, 'by'))", ns: 'ops' } },
      { return: '$decision' },
    ],
  },
});

await kleos.flow('start', { workflow: 'refund', id: 'refund-1001', input: { order: '1001' } });

async function waitFor(id: string, status: string) {
  for (let i = 0; i < 100; i++) {
    const s = await kleos.flow('status', { id });
    if (s.status === status) return s;
    if (s.status === 'failed') throw new Error(`workflow ${id} failed: ${s.error}`);
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`workflow ${id} never reached ${status}`);
}

console.log('status:', (await waitFor('refund-1001', 'waiting')).status);
// later, from anywhere (another process, an HTTP call): send the signal
await kleos.flow('signal', { id: 'refund-1001', name: 'approve', payload: { by: 'maria' } });
const done = await waitFor('refund-1001', 'completed');
console.log('output:', JSON.stringify(done.output));

const ctx = await kleos.recallContext('refund approved', { ns: 'ops' });
console.log(ctx.text.split('\n').filter((l) => l.includes('approved')).join('\n'));

worker.stop();
await kleos.close();
console.log('ok: workflows');
