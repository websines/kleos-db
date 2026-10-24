# Getting started

## Install

The package lives in the repository at `sdk/ts`; it is not on npm yet.

```sh
cd sdk/ts
npm install
npm run build              # TypeScript → dist/
npm run build:native       # the native module, for embedded stores (needs Rust)
```

`build:native` compiles `kleos-lib-nodejs` with cargo and copies the result to
`native/kleos.<platform>-<arch>.node`. Optional features go in
`KLEOS_FEATURES`:

```sh
KLEOS_FEATURES=infer-metal,docs RUSTUP_TOOLCHAIN=1.99.0 npm run build:native
```

| Feature | Adds |
|---|---|
| (none) | Memory, branches, ontology, workflows, CozoScript, SQLite storage |
| `infer` / `infer-metal` | The built-in models (embedder, decider, entity extractor); `infer-metal` uses the GPU on Apple silicon |
| `docs` | Documents through `remember`: PDF, images (OCR), Markdown, HTML, DOCX |
| `engram` | Engram overlays |

The model and document features need Rust 1.99 (`RUSTUP_TOOLCHAIN=1.99.0`).

To use it from your project: `npm install /path/to/kleos-db/sdk/ts`.

## Embedded or remote

```ts
import { open, connect } from 'kleos';

const kleos = open({ path: 'memory.db' });                       // embedded (SQLite file)
// const kleos = connect({ url: 'http://127.0.0.1:9070', token }); // remote server
```

Both return the same `Kleos` client. `open()` without a path is an in-memory
store, handy for tests. In serverless or edge code, import from
`kleos/remote`, which has no native code at all:

```ts
import { connect } from 'kleos/remote';
```

A server is the `kleos` binary: `kleos server --engine sqlite --path store.db`
(add `--mcp` to serve MCP too, `--flow-worker default` to run workflows).

## First remember and recall

<!-- from examples/ts/agent-memory.ts -->
```ts
const kleos = open({ path: join(tempDir(), 'memory.db') });
await kleos.init();

// remember: a text of several sentences becomes one memory per claim
const r = await kleos.remember('I moved to Lisbon in March. My sister Anna lives in Munich.', {
  ns: 'user-42',
  entities: ['user', 'Anna'],
});
console.log('remember:', r.action, r.claims?.length ?? 1, 'claims');

// recall a ready-to-paste context block for the next prompt
const ctx = await kleos.recallContext('Where does Anna live?', { ns: 'user-42', budget: 500 });
console.log(ctx.text);
```

The context block looks like this:

```
Recalled memories, grouped by topic, oldest first. [date] is when a statement was made or became true; ...

## Anna
- [2026-10-24 (Sat)] My sister Anna lives in Munich.
  said in: "I moved to Lisbon in March. My sister Anna lives in Munich."
```

- `init()` creates the store's memory tables. It is idempotent; call it once at
  startup.
- `ns` is the namespace: one per user, team or agent.
- `entities` names who a statement is about. It is optional: with the entity
  extractor enabled, kleos finds names itself.
- `recall()` returns the hits as objects instead of a text block.

The whole tour (documents and branches too) is
[`examples/ts/agent-memory.ts`](../../../examples/ts/agent-memory.ts).

## Enabling models

Without models, kleos still stores and recalls (by words) and keeps history.
The built-in models add meaning:

| Model | Name | Does |
|---|---|---|
| Embedder | `embeddinggemma` (EmbeddingGemma 2) | Semantic recall: "food preferences" finds "I am vegetarian" |
| Decider | `nano` (kleos-decider-nano) | Decides whether a new fact replaces, duplicates or adds to an older one |
| Entity extractor | `gliner` (kleos-gliner) | Finds names, places and relatives ("my sister Anna") |

<!-- from examples/ts/models.ts -->
```ts
await kleos.enableEmbedder('gemma', { model: 'embeddinggemma', precision: 'q8' }); // EmbeddingGemma 2
await kleos.enableDecider('nano');      // decider-nano: does a new fact replace an old one?
await kleos.enableExtractor('gliner');  // names and relatives, automatically
await kleos.init({ embedder: 'gemma', decider: 'nano', extractor: 'gliner' });
```

- **Downloads:** each model downloads once from Hugging Face into `HF_HOME`
  (default `~/.cache/huggingface`) the first time it's enabled.
  `HF_HUB_OFFLINE=1` forbids downloads.
- **Sizes:** about 290 MB (embedder, q8), 300 MB (decider) and 400 MB
  (extractor, f16).
- **Process-wide:** models are shared by every store in the process.
- **Memory budget:** `KLEOS_MEMORY_BUDGET` (default `3GB`) caps the memory all
  models use together. Batches shrink to fit, and an idle model unloads after
  `KLEOS_MODEL_IDLE_SECS` (default 900). Lower it on small machines.

Enabling models needs the native module built with `infer` (or
`infer-metal`), or a server started with models. See
[`examples/ts/models.ts`](../../../examples/ts/models.ts).

## Next

- [Concepts](concepts.md): what kleos does with your statements.
- [Chat agent with per-user memory](guides/chat-agent.md): the most common
  setup.
