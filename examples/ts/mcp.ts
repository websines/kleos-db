// Using kleos over MCP from TypeScript. The client is stateless (protocol
// 2026-07-28): no session, every request carries its version and
// capabilities. Destructive tools ask for confirmation first.
import { McpClient, McpStdioTransport } from 'kleos/remote';
import type { InputRequest } from 'kleos/remote';
import { join } from 'node:path';
import { kleosBin, tempDir } from './_util.ts';

const client = new McpClient({
  // a local `kleos mcp` over stdio; for a server use new McpHttpTransport({ url, token })
  transport: new McpStdioTransport({ command: kleosBin(), args: ['mcp', '--db', join(tempDir(), 'mcp.db'), '--ns', 'demo'] }),
  // called when a tool needs confirmation (e.g. forget): ask your user here
  onInputRequired: async (requests: Record<string, InputRequest>) => {
    for (const r of Object.values(requests)) console.log('server asks:', r.params.message);
    return { confirm: { action: 'accept', content: { confirm: true } } };
  },
});

const info = await client.discover();
console.log('protocol:', info.supportedVersions.join(', '));
console.log('tools:', (await client.listTools()).map((t) => t.name).join(', '));

const saved = await client.call('remember', { text: 'Our staging database is on db-stage-2.' });
console.log('remembered:', saved.id);

const recalled = await client.callTool('recall', { query: 'staging database' });
console.log((recalled as { content: { text?: string }[] }).content[0].text?.split('\n').slice(-2).join('\n'));

await client.call('forget', { id: saved.id });
console.log('forgotten after confirmation');

await client.close();
console.log('ok: mcp');
