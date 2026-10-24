import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { McpClient, McpHttpTransport, McpStdioTransport, acceptAll, MCP_PROTOCOL_VERSION } from '../dist/remote.js';
import { KLEOS_BIN, startServer, tempDir } from './helpers.mjs';

const skip = !KLEOS_BIN && 'set KLEOS_BIN to the kleos binary';
let server;
before(async () => { if (!skip) server = await startServer(); });
after(() => server?.stop());

async function scenario(client) {
  const d = await client.discover();
  assert.ok(d.supportedVersions.includes(MCP_PROTOCOL_VERSION));
  const tools = (await client.listTools()).map((t) => t.name);
  for (const n of ['remember', 'recall', 'forget', 'reflect', 'entity_profile', 'doc_add', 'doc_status', 'flow', 'branch']) assert.ok(tools.includes(n), n);
  const saved = await client.call('remember', { text: 'The deploy key rotates every 90 days.', ns: 't' });
  assert.ok(saved.id);
  const rec = await client.callTool('recall', { query: 'deploy key', ns: 't' });
  assert.match(rec.content[0].text, /90 days/);
  const f = await client.call('forget', { id: saved.id, ns: 't' });
  assert.deepEqual(f.ids, [saved.id]);
}

test('MCP over HTTP: stateless calls, MRTR confirmation', { skip }, async () => {
  const asked = [];
  const client = new McpClient({
    transport: new McpHttpTransport({ url: server.url }),
    onInputRequired: async (reqs) => { asked.push(...Object.keys(reqs)); return acceptAll(reqs); },
  });
  await scenario(client);
  assert.deepEqual(asked, ['confirm']);
});

test('MCP over HTTP: without a handler a destructive call needs confirm: true', { skip }, async () => {
  const client = new McpClient({ transport: new McpHttpTransport({ url: server.url }) });
  const saved = await client.call('remember', { text: 'Temporary note.', ns: 'n' });
  const r = await client.callTool('forget', { id: saved.id, ns: 'n' });
  assert.equal(r.isError, true);
  const ok = await client.call('forget', { id: saved.id, ns: 'n', confirm: true });
  assert.deepEqual(ok.ids, [saved.id]);
});

test('MCP over stdio (kleos mcp)', { skip }, async () => {
  const client = new McpClient({
    transport: new McpStdioTransport({ command: KLEOS_BIN, args: ['mcp', '--db', join(tempDir(), 'm.db')] }),
    onInputRequired: acceptAll,
  });
  try {
    await scenario(client);
  } finally {
    await client.close();
  }
});
