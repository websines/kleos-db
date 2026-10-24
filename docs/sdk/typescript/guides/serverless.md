# Serverless and edge apps

Serverless functions and edge runtimes can't load native modules, and their
instances are short-lived. Run one `kleos server` (or several behind a load
balancer: the server is stateless per request) and use the pure-TypeScript
client from `kleos/remote`, which needs only `fetch`.

Script: [`examples/ts/remote.ts`](../../../../examples/ts/remote.ts).

## Connect

<!-- from examples/ts/remote.ts -->
```ts
import { connect } from 'kleos/remote';
// ...
const kleos = connect({ url: server.url, token: process.env.KLEOS_TOKEN });
```

In your app: `connect({ url: process.env.KLEOS_URL, token:
process.env.KLEOS_TOKEN })`. The client has the same methods as an embedded
store (except `flowWorker`, which runs in the server: `kleos server
--flow-worker default`).

## A handler

<!-- from examples/ts/remote.ts -->
```ts
async function handler(req: { query: string }) {
  const block = await kleos.recallContext(req.query, { ns: 'team', budget: 400 });
  return { context: block.text, tokens: block.tokens };
}
```

## Running the server

```sh
kleos server --engine sqlite --path /data/kleos.db --bind 0.0.0.0 --port 9070 \
  --flow-worker default --flow-worker docs --mcp
```

- **Auth:** send the server's admin secret, or a token from its token table,
  as `Authorization: Bearer …` (the `token` option). Read-only tokens may only
  recall, reflect, diff and list.
- **`--insecure-no-auth`** is only accepted on a loopback address, for local
  development.
- **Options:** `connect()` also takes a custom `fetch`, extra `headers` and a
  `timeoutMs`.
