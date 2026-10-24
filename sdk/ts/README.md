# kleos (TypeScript SDK)

Agent memory for TypeScript: remember statements and documents, recall a
dated context block for the next prompt, with branches, an automatic
ontology and durable workflows. One API over an embedded store (native
module) or a remote `kleos server` (HTTP, fetch only).

```ts
import { open, connect } from 'kleos';

const kleos = open({ path: 'memory.db' });          // embedded
// const kleos = connect({ url: 'http://127.0.0.1:9070', token }); // remote

await kleos.init();
await kleos.remember('I moved to Lisbon in March.', { ns: 'user-42' });
const ctx = await kleos.recallContext('Where do I live?', { ns: 'user-42' });
console.log(ctx.text);
```

```sh
npm install && npm run build      # TypeScript
npm run build:native              # native module (Rust); KLEOS_FEATURES=infer-metal,docs for models and documents
npm test                          # KLEOS_BIN=<kleos binary> for the server and MCP tests, KLEOS_PY for the Python parity test
```

`kleos/remote` exports the HTTP and MCP clients without the native module, for
serverless and edge runtimes. Not published to npm yet.

Documentation: [`docs/sdk/typescript/`](../../docs/sdk/typescript/README.md)
(getting started, concepts, guides, API reference, Python equivalents,
troubleshooting). Examples: [`examples/ts/`](../../examples/ts/).
