# Using kleos over MCP from TypeScript

kleos serves MCP (protocol 2026-07-28, stateless) over stdio (`kleos mcp`) and
HTTP (`kleos server --mcp`). Most agent frameworks connect to it directly.
`McpClient` lets your own TypeScript code do the same, for example to test a
setup or to drive kleos from a tool that only speaks MCP.

Script: [`examples/ts/mcp.ts`](../../../../examples/ts/mcp.ts). For plain
application code, the `Kleos` client is simpler; MCP is for agent tooling.

## Connect

<!-- from examples/ts/mcp.ts -->
```ts
const client = new McpClient({
  // a local `kleos mcp` over stdio; for a server use new McpHttpTransport({ url, token })
  transport: new McpStdioTransport({ command: kleosBin(), args: ['mcp', '--db', join(tempDir(), 'mcp.db'), '--ns', 'demo'] }),
  // called when a tool needs confirmation (e.g. forget): ask your user here
  onInputRequired: async (requests: Record<string, InputRequest>) => {
    for (const r of Object.values(requests)) console.log('server asks:', r.params.message);
    return { confirm: { action: 'accept', content: { confirm: true } } };
  },
});
```

There is no session: every request carries the protocol version and the
client's capabilities in `_meta`, so any request can reach any server instance.

## Call tools

<!-- from examples/ts/mcp.ts -->
```ts
const saved = await client.call('remember', { text: 'Our staging database is on db-stage-2.' });
console.log('remembered:', saved.id);

const recalled = await client.callTool('recall', { query: 'staging database' });
```

- **`call(name, args)`** returns the tool's structured JSON result and throws
  on a tool error.
- **`callTool`** returns the raw result: `content` (text blocks; for `recall`,
  the context block) and `structuredContent`.
- **The tools:** `remember`, `recall`, `forget`, `reflect`, `entity_profile`,
  `doc_add`, `doc_status`, `flow` and `branch`, plus `query` when the server
  allows it.

## Confirmations

Destructive tools (`forget`, branch `merge`/`rewind`/`discard`) answer
`input_required` first. The client calls `onInputRequired`, then retries the
call with your answer and the server's signed state:

<!-- from examples/ts/mcp.ts -->
```ts
await client.call('forget', { id: saved.id });
console.log('forgotten after confirmation');
```

Without `onInputRequired`, pass `confirm: true` in the arguments, after asking
the user yourself. `acceptAll` is a ready-made handler for scripts that have
already asked.

## Long calls

With the Tasks extension (on by default), `doc_add` on a big document returns
a task instead of waiting. Poll it with `client.waitTask(task.taskId)`; the
task id is the durable ingestion workflow's id, so it survives server
restarts.
