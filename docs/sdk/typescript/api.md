# API reference

Everything is exported from `kleos`; `kleos/remote` exports the same minus the
native parts (`open`, `NativeBackend`, `loadNative`, `nativeFeatures`). All
methods return Promises unless noted.

**Conventions:**
- **Options** accept camelCase or snake_case keys (`asOf` or `as_of`).
- **Results** are kleos's JSON as is, so their fields are snake_case
  (`valid_from`).
- **Times** are microseconds since the epoch (`Date.now() * 1000`).
- **Errors** throw [`KleosError`](#kleoserror).

## Creating a client

| Function | Returns | Notes |
|---|---|---|
| `open(opts?: { engine?: 'mem' \| 'sqlite' \| 'rocksdb'; path?: string; options?: object })` | `Kleos` (sync) | Embedded store. With `path` the engine defaults to `sqlite`, without it to `mem`. Needs the native module. |
| `connect(opts: { url: string; token?: string; fetch?: typeof fetch; headers?: Record<string,string>; timeoutMs?: number })` | `Kleos` (sync) | Remote server over HTTP. |
| `new Kleos(backend: Backend)` | `Kleos` | Any backend (`NativeBackend`, `HttpBackend`, or your own). |
| `nativeFeatures()` | `string[]` (sync) | Features of the loaded native module: `infer`, `infer-metal`, `docs`, `engram`, `storage-rocksdb`. |

```ts
const kleos = open({ path: 'memory.db' });
const remote = connect({ url: 'http://127.0.0.1:9070', token: process.env.KLEOS_TOKEN });
```

## Memory

| Method | Returns | Description |
|---|---|---|
| `init(opts?: InitOptions)` | `MemoryConfig` | Create or read the memory configuration. Options: `embedder`, `decider`, `extractor`, `ontology` (`personal` by default, `none` to turn it off). Idempotent. |
| `remember(text, opts?: RememberOptions)` | `RememberResult` | Store a statement (split into claims), decide supersession, attach documents. |
| `recall(query, opts?: RecallOptions)` | `RecallHit[]` | Relevant memories, best first. |
| `recallContext(query, opts?: ContextOptions)` | `ContextBlock` | The dated, grouped context block for a prompt, within `budget` tokens. |
| `forget(id?, opts?: ForgetOptions)` | `ForgetResult` | Retract (default), `erase` (all versions) or `archive` one memory or a filter. |
| `reflect(ns?, opts?)` | `ReflectResult` | Pending decisions, contradictions, recent changes. Options: `since`, `limit`. |
| `profile(name, opts?: { ns?, budget? })` | `EntityProfile` | Ontology facts plus a context block about one entity. |
| `set(id, opts)` | object | Change `importance`, `ttl`/`expires`, `noTtl`, `archived`. |
| `link(a, b, rel, opts?)` | object | Typed link between memories or entities. |
| `flush()` | `{ flushed }` | Write buffered access counts. |
| `memory(op, args?)` | any | Any memory op by name. |

```ts
const r = await kleos.remember('I moved to Lisbon.', { ns: 'u1', entities: ['user'] });
const hits = await kleos.recall('where do I live', { ns: 'u1', k: 5, hidePending: true });
const ctx = await kleos.recallContext('where do I live', { ns: 'u1', budget: 600 });
await kleos.forget(hits[0].id, { ns: 'u1', mode: 'erase' });
```

**`RememberOptions`:**

| Option | Type | Meaning |
|---|---|---|
| `ns` | string | Namespace (default `default`) |
| `entities` | string[] | Who the statement is about (`user`, `Anna`) |
| `at` | µs | Valid-from (default now) |
| `ttl` / `expires` | µs | Expiry relative to `at` / absolute |
| `role` | `user` \| `assistant` \| `other` | Speaker; assistant and other memories never decide |
| `split` | `auto` \| `none` | One memory per claim (default) or one for the text |
| `attachments` | `(string \| Attachment)[]` | Documents: a path, `{ bytes, mime, name }` or `{ data_b64, mime, name }` |
| `file` | string | A document path instead of text |
| `priority` | `interactive` \| `bulk` | Model lane |
| `kind`, `source`, `importance`, `k`, `decider`, `minSimilarity`, `extract` | | See the [memory API spec](../../specs/memory-api.md) |

**`RememberResult`:**
- `action` (`added` \| `duplicate` \| `documents`), `id`, `ns`, `claims`;
- `superseded` (ids), `duplicate_of`;
- `tentative` (pending decisions), `decisions`, `gated`;
- `entities`, `entity_links`, `types` (ontology typing);
- `documents` (`{ doc, name, status, chunks, … }`), `embedded`, `decider`.

**`RecallOptions`:**

| Option | Type | Meaning |
|---|---|---|
| `ns`, `k` (default 10), `kind`, `entity` | | Scope |
| `asOf` | µs | Read the past |
| `hidePending` | boolean | Leave out memories with a pending supersede |
| `sources` | `('memory' \| 'docs')[]` | Memories, document chunks, or both (default) |
| `from`, `to` / `eventFrom`, `eventTo` | µs | Valid-time / event-time windows |
| `expand`, `graph` | | Entity-graph expansion and graph modes |
| `vectorWeight`, `includeArchived`, `includeExpired`, `role`, `priority`, `recordAccess` | | See the spec |
| `budget`, `showSource` | | `recallContext` only |

**`RecallHit`:**
- `id`, `text` (the whole statement for a claim), `claim`, `context`;
- `score`, `relevance`, `recency`;
- `valid_from`, `valid_to`, `pending_superseded_by`;
- `kind`, `source`, `role`, `events`;
- `doc` (`{ doc, name, page, page_end, heading }` for document chunks), `expanded`.

**`ContextBlock`:** `text`, `tokens`, `budget`, `ids` and `omitted`.

## Branches

| Method | Returns | Description |
|---|---|---|
| `fork(from, to, opts?: { at? })` | `ForkResult` | Instant copy-on-write branch. |
| `diff(a, b, opts?)` | `{ added, removed, same, pending, entities }` | What changes from `a` to `b`. |
| `merge(from, into?, opts?: { dryRun?, keepTime?, forgets? })` | `{ statements, replayed, skipped, diff? }` | Replay the branch into `into` (default its parent). |
| `rewind(ns, to, opts?)` | `{ retracted, restored, undo }` | State at time `to`; undo with `rewind(ns, r.undo)`. |
| `discard(ns, opts?: { cascade? })` | object | Drop a branch's own rows. |
| `branches(opts?: { ns?, all? })` | `BranchInfo[]` | List branches. |
| `reembed(embedder, opts?: { batch?, wait?, queue? })` | object | Move the store to another embedder (durable, resumable). |

```ts
await kleos.fork('plan', 'plan-try');
const preview = await kleos.merge('plan-try', 'plan', { dryRun: true });
```

## Ontology, documents, workflows

| Method | Returns | Description |
|---|---|---|
| `ontology(op, args?)` | any | `list`, `get`, `define`, `import`, `export`, `activate`, `facts`, `infer`, `validate`, `type`, `learn`, `history`, `undo`, … ([spec](../../specs/ontology.md)) |
| `docs(op, args?)` | any | `get`, `list`, `delete`, `reingest` ([spec](../../specs/documents.md)) |
| `flow(op, args?)` | any | `define`, `start`, `signal`, `cancel`, `status`, `list`, `history`, `schedule`, `tick`, … ([spec](../../specs/durable-workflows.md)) |
| `flowWorker(opts?: { queues?, pollMs?, leaseMs?, worker? })` | `FlowWorker` (sync) | In-process worker; `stop()`, `wake()`. Embedded only. |

```ts
const facts = await kleos.ontology('facts', { ns: 'u1' });
const worker = kleos.flowWorker({ queues: ['default'] });
await kleos.flow('start', { workflow: 'refund', id: 'r1', input: { order: '1001' } });
worker.stop();
```

## Queries and models

| Method | Returns | Description |
|---|---|---|
| `run(script, params?, opts?: { immutable? })` | `NamedRows` | A CozoScript query: `{ headers, rows }`. |
| `enableEmbedder(name, spec?)` / `disableEmbedder(name)` | void / boolean | Spec: `{ model, precision, device, dim }`. |
| `enableDecider(name, spec?)` / `disableDecider(name)` | void / boolean | Spec: `{ model, path, precision, device }`. |
| `enableExtractor(name, spec?)` / `disableExtractor(name)` | void / boolean | Spec: `{ model, path, precision, device }`. |
| `embedders()`, `deciders()`, `extractors()` | `NamedRows` | What is enabled. |
| `hybridSearch(query)` | `NamedRows` | Vector + text search over any relation with indexes (`relation`, `vector_index`, `query_vector`, `fts_index`, `query_text`, `limit`, `fusion`, `extra_lists`), plus `graph_legs` (k-hop or budgeted proximity legs; unknown keys are rejected), `rerank` (`{model, top_n, text_field, query, batch_size}` with a reranker from `registerReranker`) and `mmr` (`{lambda, k, embedding_col}`), as Python's `hybrid_search`. Embedded only. |
| `registerReranker(name, fn)` / `unregisterReranker(name)` | void / boolean | A JS reranker `(query, docs) => number[]` (synchronous; called from the thread pool, run on the JS thread). Embedded only. |
| `watch(relation, onEvent)` | `LiveHandle` | Committed writes to a stored relation (`kleos_mem`, `kleos_doc`, your own): `{op, new_rows, old_rows}`. Embedded: change callbacks (napi threadsafe function); remote: the server's `GET /changes/:relation` stream (admin secret). `close()` ends it; it keeps the process alive until then. |
| `subscribe(script, onEvent, opts?: { params?, queue? })` | `LiveHandle` | A live query: `initial`, then `delta` (`added`, `removed`), `resync` or `error` events (docs/specs/live-subscriptions.md). Remote: `GET /subscribe`. |
| `decide(name, template, a, b)` | `DecideResult` | One typed decision with an enabled decider (e.g. `supersedes`). Embedded only. |
| `exportRelations(relations)` | `{ [rel]: NamedRows }` | Rows of stored relations. |
| `importRelations(data)` | void | Put rows into existing relations (the `exportRelations` shape). |
| `backup(path)` | void | Write a backup file (remote: inside the server's backup directory). |
| `restore(path)` | void | Restore a backup into an empty store. Embedded only. |
| `importFromBackup(path, relations)` | void | Copy relations (which must exist) from a backup file. |
| `close()` | void | Close the store (embedded); a no-op for remote. |
| `kind` | `'embedded' \| 'remote'` | Property. |

```ts
await kleos.enableEmbedder('gemma', { model: 'embeddinggemma', precision: 'q8' });
const rows = await kleos.run("?[count(id)] := *kleos_mem{ns: $ns, id @ 'NOW'}", { ns: 'u1' }, { immutable: true });
```

Models are process-wide in an embedded store. Remotely, the enable and
disable calls run `::embedder enable …` and its siblings on the server.

## Tenancy

Principal mode (docs/specs/tenancy.md): orgs, workspaces, teams, human and agent principals, scoped memory with an audit trail.

```ts
await kleos.tenant('org_create', { org: 'acme', owner: 'alice' });
await kleos.tenant('principal_create', { id: 'bot', org: 'acme', kind: 'agent', sponsor: 'alice' });
await kleos.tenant('ws_create', { ws: 'eng', org: 'acme' });
await kleos.tenant('grant', { ws: 'eng', principal: 'alice', role: 'writer' });

const alice = { workspace: 'eng', human: 'alice', agent: 'bot' };
await kleos.memoryAs(alice, 'remember', { text: 'I prefer morning meetings.' }); // scope au:bot:alice
const hits = await kleos.memoryAs(alice, 'recall', { query: 'meetings' });     // each hit has `scope`
```

- `tenant(op, args)`: admin ops (`org_create`, `ws_create`, `team_create`, `principal_create`, `member_add`, `grant`, `revoke`, `policy_set`, `apikey_create`, `erase`, `approve`, `audit`, `audit_verify`, `scopes`, `list`). Embedded: the host is trusted; remote: `POST /tenant/:op` needs the admin secret or an org admin's token.
- `memoryAs(ctx, op, args)`: `remember`, `recall`, `context`, `forget`, `set`, `link`, `reflect`, `consolidate`, `promote`, `approve`, `reject`, `fork`, `merge`, `discard`, `diff`, `flow`, `doc`, `ontology`, `scopes`, with `scope` in place of `ns`. Remote: `ctx` is ignored and the server runs the op as the principal of the client's token (a JWT access token or a `kl_…` API key).

## MCP client

| Export | Description |
|---|---|
| `new McpClient({ transport, clientInfo?, onInputRequired?, tasks? })` | Stateless client (protocol `MCP_PROTOCOL_VERSION` = `2026-07-28`). |
| `new McpHttpTransport({ url, token?, fetch?, headers? })` | `POST /mcp` with `MCP-Protocol-Version`, `Mcp-Method` and `Mcp-Name` headers. |
| `new McpStdioTransport({ command?, args?, env? })` | Spawns `kleos mcp` (Node only). |
| `client.discover()` | `server/discover`. |
| `client.listTools()`, `listResources()`, `readResource(uri)` | Lists and reads. |
| `client.callTool(name, args)` | `ToolResult` or `McpTask`; answers `input_required` through `onInputRequired`. |
| `client.call(name, args)` | The structured result; throws on `isError`. |
| `client.getTask(id)`, `cancelTask(id)`, `waitTask(id, opts?)` | Tasks extension. |
| `acceptAll` | An `onInputRequired` handler that confirms everything. |

```ts
const client = new McpClient({ transport: new McpHttpTransport({ url, token }), onInputRequired: acceptAll });
const r = await client.call('recall', { query: 'deploy key', ns: 'ops' });
```

## KleosError

Every failure is a `KleosError` (`instanceof Error`) with:

| Field | Meaning |
|---|---|
| `message` | kleos's message, e.g. `unknown memory op 'x'`, `database closed` |
| `op` | The op that failed: `memory.remember`, `flow.start`, `run`, `mcp.tools/call` |
| `status` | HTTP status, for remote calls |
| `cause` | The underlying error |

```ts
try {
  await kleos.recall('x');
} catch (e) {
  if (e instanceof KleosError) console.error(e.op, e.message);
}
```
