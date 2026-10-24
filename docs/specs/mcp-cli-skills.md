# Spec — MCP server (stateless, 2026-07-28), `kleos` CLI and agent skills

_Created 2026-10-25. Status: built on `wave6/mcp`._

kleos is reachable by agents three ways, all thin layers over the same
`DbInstance` calls (`memory_call`, `doc_call`, `flow_call`,
`ontology_call`, `run_script`):

| Surface | For | Context cost |
|---|---|---|
| **MCP server** (`kleos mcp`, `kleos server --mcp`) | chat apps and IDEs that speak MCP | the tool list (~10 tools) sits in the agent's context |
| **`kleos` CLI** | coding agents with a shell, scripts, humans | nothing until invoked |
| **Agent skills** (`skills/*/SKILL.md`) | teaching an agent *when* and *how* to use kleos through the CLI or MCP | name + description up front; body on demand |

The standalone binary is built as `kleos-bin` and shipped as `kleos`
(`scripts/build-release-*.sh`); this spec writes `kleos`.

## 1. Protocol: MCP 2026-07-28, stateless

Implemented by hand (`kleos-bin/src/mcp.rs`, about 1k lines, no new
dependencies) rather than with an SDK:

- `rmcp` 3.7 (official, Apache-2.0) supports 2026-07-28 only in beta, pulls
  axum 0.8 next to kleos-bin's axum 0.7, and its session-oriented server
  transport is the part that changed most.
- `rust-mcp-sdk` 2.0 (MIT) targets 2026-07-28 but is small-community and
  brings its own runtime layer.
- The stateless protocol needed here is small: JSON-RPC, `server/discover`,
  `tools/*`, `resources/*`, MRTR results and the Tasks extension's three
  methods. Owning it keeps the wire format exactly as specified, is
  dependency-free, and is tested against the spec's example shapes.

### Requests

- No `initialize` and no sessions for modern clients. Every request carries
  `_meta`:
  - `io.modelcontextprotocol/protocolVersion` (required; `2026-07-28`),
  - `io.modelcontextprotocol/clientCapabilities` (required),
  - `io.modelcontextprotocol/clientInfo` (optional).
  A modern request missing a required field → `-32602` (HTTP 400). An unknown
  version → `UnsupportedProtocolVersion` `-32022` with
  `data: {supported, requested}` (HTTP 400).
- Every result carries `resultType` (`complete`, `input_required`, or
  `task`) and `_meta["io.modelcontextprotocol/serverInfo"]`.
- `server/discover` → `{supportedVersions, capabilities, instructions,
  ttlMs, cacheScope}`. Capabilities: `tools`, `resources`,
  `extensions: {"io.modelcontextprotocol/tasks": {}}`.
- List results (`tools/list`, `resources/list`, `resources/templates/list`,
  `resources/read`) carry `ttlMs` and `cacheScope` (`private` — results
  depend on the store) and are in a deterministic order.
- Methods: `server/discover`, `tools/list`, `tools/call`, `resources/list`,
  `resources/templates/list`, `resources/read`, `tasks/get`,
  `tasks/update`, `tasks/cancel`. Unknown method → `-32601` (HTTP 404).
- Not implemented (deprecated in 2026-07-28): roots, sampling, logging
  (`logging/setLevel`), HTTP+SSE, `ping` (kept only for legacy clients),
  `subscriptions/listen` (no server-pushed notifications yet).

### State

The protocol keeps no state between requests. Application state travels in
arguments: `ns` (namespace), branch names, flow ids, document ids, task ids.
Nothing depends on the connection, so any request can reach any server
instance.

### Confirmations (MRTR)

Destructive calls — `forget`, `branch` with `merge` (not a dry run),
`rewind` or `discard` — need confirmation:

1. If the request's client capabilities include `elicitation`, the server
   answers `resultType: "input_required"` with
   `inputRequests: {"confirm": {method: "elicitation/create", params: {mode:
   "form", message, requestedSchema: {type: object, properties: {confirm:
   {type: boolean}}, required: [confirm]}}}}` and a `requestState`.
2. The client retries the same call (new JSON-RPC id) with
   `inputResponses: {"confirm": {action: "accept", content: {confirm:
   true}}}` and the echoed `requestState`; the call then runs. `decline` or
   `cancel` returns a tool result with `isError: true` and nothing is changed.
3. `requestState` is `base64url(payload).base64url(hmac)`: the payload holds
   the tool name, a SHA-256 of the canonical arguments, the principal and an
   expiry (10 minutes); HMAC-SHA256 with a key from `KLEOS_MCP_STATE_KEY`, else
   per process random. Tampered, expired, other-principal or other-call state is
   rejected. Set the same key on every instance behind a load balancer.
4. Clients without `elicitation` (and legacy clients) confirm by passing
   `confirm: true` in the tool arguments after asking the user; without it
   the tool returns an `isError` result explaining that.

### Long-running calls (Tasks extension)

`doc_add` on a large document starts a durable `doc.ingest` workflow. If the
request declares `extensions: {"io.modelcontextprotocol/tasks": {}}` in its
client capabilities, the call returns a `CreateTaskResult`
(`resultType: "task"`, `taskId`, `status: "working"`, `createdAt`,
`lastUpdatedAt`, `ttlMs: null`, `pollIntervalMs`). `taskId` is the workflow id,
so it is durable and needs no server memory:

- `tasks/get {taskId}` maps the workflow status to `working`, `completed`
  (with `result`: the `CallToolResult` the call would have returned),
  `failed` (with `error`) or `cancelled`;
- `tasks/cancel` cancels the workflow;
- `tasks/update` is acknowledged (kleos tasks never ask for input).

Without the extension the same call returns a normal result with the
document id and `status: "running"`; `doc_status` polls it.

### Transports

- **stdio** — `kleos mcp [--db PATH | --url URL]`: newline-delimited
  JSON-RPC on stdin/stdout, logs on stderr. Notifications get no reply.
- **Streamable HTTP** — `kleos server --mcp` mounts `POST /mcp` behind the
  server's existing authentication (`Authorization: Bearer <token>` or the
  `x-cozo-auth` header; read-only token-table principals may call only read
  tools). Rules:
  - `MCP-Protocol-Version` header required and equal to the body's
    `_meta` version; `Mcp-Method` equal to `method`; `Mcp-Name` equal to
    `params.name` (`tools/call`) or `params.uri` (`resources/read`), with the
    `=?base64?…?=` sentinel decoded. A missing or mismatched header →
    HTTP 400 `HeaderMismatch` `-32020`.
  - An `Origin` header that is not the server's own origin (or a loopback
    origin when bound to loopback) → 403.
  - Responses are `application/json`; notifications → 202; `GET` and
    `DELETE` → 405; `Mcp-Session-Id` and `Last-Event-ID` are ignored and never
    minted.

### Earlier clients (2025-11-25 and before)

kleos is dual-era: an `initialize` request selects legacy semantics (version
negotiated from `2025-11-25`, `2025-06-18`, `2025-03-26`), answered without
a session id; `notifications/initialized` and `ping` are accepted; later
requests without `_meta` are served under the negotiated version (stdio:
for the life of the process; HTTP: per request, from the
`MCP-Protocol-Version` header). Legacy clients get no MRTR or tasks; they
confirm with `confirm: true`.

## 2. Tools

Ten tools with short descriptions and flat schemas (the default list is
nine; `query` needs `--allow-query`). Tool errors come back as
`isError: true` results so the model can see and correct them; protocol
errors (unknown tool, bad params) as JSON-RPC errors.

| Tool | Arguments | Annotations | Notes |
|---|---|---|---|
| `remember` | `text`, `ns?`, `attachments?` (`[{path} \| {data_b64, mime, name}]`), `entities?`, `at?`, `priority?` | write | statements, claims, supersession, entities, documents |
| `recall` | `query`, `ns?`, `k?`, `format?` (`context` default, or `list`), `budget?`, `as_of?`, `sources?` | read-only | `context` returns the ready-to-paste context block |
| `forget` | `id?`, `ids?`, `ns?`, `mode?` (`retract` default, `erase`), `confirm?` | destructive | MRTR confirmation |
| `reflect` | `ns?`, `since?`, `limit?` | read-only | pending decisions, conflicts, recent changes |
| `entity_profile` | `name`, `ns?` | read-only | ontology facts about the entity plus its memories |
| `doc_add` | `path?` or `data_b64` + `mime` + `name`, `text?`, `ns?` | write | may return a task |
| `doc_status` | `doc?`, `ns?` | read-only | one document, or the list |
| `flow` | `action` (`start`, `status`, `signal`, `cancel`, `list`), `workflow?`, `input?`, `id?`, `name?`, `payload?` | write | durable workflows |
| `ontology` | `action` (`list`, `facts`, `infer`, `learn`, `history`, `undo`), `ns?`, `subject?`, `property?`, `change?`, `confirm?` | undo asks (MRTR) | the self-maintaining ontology; `infer` returns derived facts only |
| `reembed` | `embedder`, `ns?` | write | moves a namespace to another embedder; returns a task (`tasks/get`) when the client supports tasks |
| `consolidate` | `ns?`, `profiles?`, `summaries?` | write | rebuilds entity profiles and session summaries; a task (the `memory.consolidate` workflow) when the client supports tasks, else inside the call |

Background tasks of an embedded store (re-embed, consolidate) run on an in-process workflow worker the server starts on first use; with `--url`, the remote server's own worker runs them.
| `branch` | `action` (`fork`, `diff`, `merge`, `rewind`, `discard`, `list`), `from?`, `to?`, `ns?`, `a?`, `b?`, `into?`, `at?`, `dry_run?`, `confirm?` | destructive for merge/rewind/discard | memory branches |
| `query` | `script`, `params?` | read-only | CozoScript, always run immutable; off unless `--allow-query` |

`ns` defaults to the server's default namespace (`--ns`, config, else
`default`). Results: `content` holds one text block (the context block for
`recall` in context format, compact JSON otherwise) and `structuredContent`
holds the JSON result.

### Resources

| URI | Contents |
|---|---|
| `kleos://namespaces` | namespaces with memories, their branches |
| `kleos://ns/{ns}/documents` | documents of a namespace |
| `kleos://doc/{ns}/{doc}` | one document's record |
| `kleos://entity/{ns}/{name}` | an entity profile |

`resources/list` lists `kleos://namespaces` and the default namespace's
documents; the last three are also `resources/templates/list` entries. A
missing resource → `-32602`.

## 3. CLI

```
kleos remember TEXT [--ns NS] [--attach FILE]... [--entity NAME]...
kleos recall QUERY [--ns NS] [-k N] [--list] [--budget TOKENS] [--as-of TS]
kleos forget ID... [--ns NS] [--erase] [--yes]
kleos reflect [--ns NS]
kleos profile NAME [--ns NS]
kleos doc add FILE [--ns NS] [--text NOTE] | doc status DOC [--ns NS] | doc list [--ns NS]
kleos ontology list | learn | history | undo [--ns NS]
kleos flow start WORKFLOW [--input JSON] | status ID | signal ID NAME [--payload JSON] | cancel ID | list
kleos branch fork TO [--from NS] | diff A B | merge FROM [--into NS] [--dry-run] | rewind NS TS | discard NS | list
kleos query SCRIPT [--params JSON]          # read-only
kleos mcp [--allow-query]                    # MCP over stdio
kleos server --mcp ...                       # HTTP, also serves POST /mcp
```

- Global options: `--db PATH` (embedded store; SQLite unless `--engine`),
  `--url URL` + `--token TOKEN` (remote kleos server), `--ns NS`,
  `--json` (machine output; human output otherwise), `--config FILE`.
- **Config file** — `$KLEOS_CONFIG`, else `~/.config/kleos/config.json`:
  `{"db", "engine", "url", "token", "ns", "embedders": ["gemma=embeddinggemma"],
  "decider", "extractor", "ontology"}`. Flags override the file; embedders,
  decider and extractor are enabled at start in embedded mode (needs a build with
  `infer`).
- Destructive commands (`forget`, `branch merge/rewind/discard`) ask on a
  terminal, and need `--yes` otherwise.
- Errors name the fix (no store configured, server unreachable, feature
  missing) and exit non-zero; `--json` errors are `{"ok": false, "error"}`.

## 4. Skills

`skills/` holds four skills in the open SKILL.md format (YAML `name` and
`description`, Markdown body), each teaching *when* and *how* with the CLI
first and the MCP tool names alongside:

| Skill | Teaches |
|---|---|
| `kleos-memory` | recall before answering about the user or project; remember stable facts and decisions (not chit-chat); use `--budget`; read "previously"/pending notes; forget only on request |
| `kleos-documents` | attach files with `remember`/`doc add`, wait for large ones, cite `[file p. N]` |
| `kleos-ontology` | inspect learned properties, history and undo |
| `kleos-branches` | fork before risky experiments, diff, merge with a dry run first, discard |

The repository root is a Claude Code plugin: `.claude-plugin/plugin.json`
names the skills directory and an MCP server entry (`kleos mcp`). It
installs no hooks.

## 5. Tests

- `kleos-bin/tests/mcp.rs`: protocol over an in-memory store — discover,
  version errors, missing `_meta`, deterministic `tools/list` with `ttlMs`,
  every tool, MRTR confirm / decline / tampered state, `confirm: true`
  fallback, tasks (a task result and `tasks/get`), resources, legacy
  `initialize`.
- `kleos-bin/tests/mcp_http.rs`: the HTTP endpoint — headers, mismatch
  errors, Origin, 202/405, auth.
- `kleos-bin/tests/cli.rs`: the binary end to end on a temporary store.
- `scripts/mcp/e2e.py`: a stdio MCP client and the CLI against one store;
  `scripts/mcp/token_cost.py`: context cost of the tool list against the
  skills.

## 6. Results (2026-10-25)

- **Tests** (all passing): `tests/mcp.rs` 9, `tests/mcp_http.rs` 2 (against
  the real binary over HTTP, with and without auth), `tests/cli.rs` 3 (CLI end
  to end, config errors, MCP over stdio), plus 4 unit tests in `mcp.rs`
  (HMAC, base64 sentinel, argument digest, percent-encoding).
- **End to end** — `scripts/mcp/e2e.py <kleos> <dir>`: an MCP stdio client
  and the CLI on one SQLite store, both directions visible, fork in the CLI and
  diff over MCP, `forget` confirmed through MRTR: 10/10 checks.
- **Context cost** — `scripts/mcp/token_cost.py <kleos> .` (characters / 4):

  | Surface | ~tokens | In context |
  |---|---|---|
  | MCP tool list, 9 tools | 1,517 | every turn |
  | MCP tool list with `query` | 1,592 | every turn |
  | 4 skills: names + descriptions | 311 | every turn |
  | 4 skill bodies together | 1,487 | only when opened |

  Per tool: remember 241, branch 222, recall 220, flow 178, doc_add 174,
  forget 166, reflect 117, entity_profile 104, doc_status 95. A coding agent
  using the CLI through the skills carries about a fifth of the MCP cost until it
  actually needs kleos.
- **Plugin** — `claude plugin validate .` passes.
- **Not run:** the official MCP inspector (`npx @modelcontextprotocol/inspector
  --cli kleos mcp --db <path> --method tools/list`) — fetching it was not
  permitted in this session; the inspector's client is also likely to use the
  legacy `initialize` path, which the dual-era server serves.

### Left

- `subscriptions/listen` (server-pushed list or resource changes) and
  `notifications/progress` streaming (SSE responses) — every response is a
  single JSON object today.
- OAuth 2 / Client ID Metadata Documents for HTTP auth (bearer tokens only).
- Tasks for other long calls (`::memory reembed`, consolidation) — only
  background document ingestion returns a task now.
- An ontology tool over MCP (the skill points at the CLI).
- `x-mcp-header` mirroring of tool parameters is not used by any kleos tool.
