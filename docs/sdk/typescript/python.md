# Python equivalents

The TypeScript SDK and the Python binding (`kleos`, class `CozoDbPy`) call the
same engine functions, so results are the same JSON (a parity test in
`sdk/ts/test/parity.test.mjs` runs one scenario through both). Python passes
options as keyword arguments; TypeScript passes an options object, in
camelCase or snake_case.

| Python | TypeScript |
|---|---|
| `db = CozoDbPy("sqlite", "memory.db", "{}")` | `const kleos = open({ path: 'memory.db' })` |
| `CozoDbPy("mem", "", "{}")` | `open()` |
| (HTTP client) | `connect({ url, token })` |
| `db.memory_init(embedder="gemma")` | `await kleos.init({ embedder: 'gemma' })` |
| `db.remember(text, ns="u", entities=["user"])` | `await kleos.remember(text, { ns: 'u', entities: ['user'] })` |
| `db.remember(text, attachments=[{"bytes": b, "mime": m, "name": n}])` | `await kleos.remember(text, { attachments: [{ bytes, mime, name }] })` |
| `db.recall(q, ns="u", k=5, as_of=t)` | `await kleos.recall(q, { ns: 'u', k: 5, asOf: t })` |
| `db.recall_context(q, ns="u", budget=800)` | `await kleos.recallContext(q, { ns: 'u', budget: 800 })` |
| `db.forget(id, mode="erase")` | `await kleos.forget(id, { mode: 'erase' })` |
| `db.reflect("u")` | `await kleos.reflect('u')` |
| `db.memory_set(id, importance=0.9)` | `await kleos.set(id, { importance: 0.9 })` |
| `db.memory_flush()` | `await kleos.flush()` |
| `db.link(a, b, "rel")` | `await kleos.link(a, b, 'rel')` |
| `db.memory_fork("u", "u2")` | `await kleos.fork('u', 'u2')` |
| `db.memory_merge("u2", dry_run=True)` | `await kleos.merge('u2', undefined, { dryRun: true })` |
| `db.memory_diff("u", "u2")` | `await kleos.diff('u', 'u2')` |
| `db.memory_rewind("u", t)` | `await kleos.rewind('u', t)` |
| `db.memory_discard("u2")` | `await kleos.discard('u2')` |
| `db.memory_branches()` | `await kleos.branches()` |
| `db.memory_reembed("gemma-v1")` | `await kleos.reembed('gemma-v1')` |
| `db.ontology_call("facts", {"ns": "u"})` | `await kleos.ontology('facts', { ns: 'u' })` |
| `db.doc_call("list", {"ns": "u"})` | `await kleos.docs('list', { ns: 'u' })` |
| `db.flow_call("start", {...})` | `await kleos.flow('start', { ... })` |
| `w = db.flow_worker(queues=["default"]); w.stop()` | `const w = kleos.flowWorker({ queues: ['default'] }); w.stop()` |
| `db.run_script(q, params, False)` | `await kleos.run(q, params)` |
| `db.enable_embedder("gemma", model="embeddinggemma")` | `await kleos.enableEmbedder('gemma', { model: 'embeddinggemma' })` |
| `db.enable_decider("nano")` | `await kleos.enableDecider('nano')` |
| `db.enable_extractor("gliner")` | `await kleos.enableExtractor('gliner')` |
| `db.embedders()` | `await kleos.embedders()` (rows) |
| (CLI `kleos profile NAME`) | `await kleos.profile(name, { ns })` |
| `db.close()` | `await kleos.close()` |

**Not in the TypeScript SDK yet.** Most of these are reachable through
`run()` with the matching CozoScript system op:
- `register_callback`, `subscribe` (live updates);
- `hybrid_search`, `run_cypher`;
- `decide` and `fit_decider`;
- entity `resolve`, `merge` and `split`;
- `import_relations` and `export_relations`, `backup` and `restore`;
- `multi_transact`;
- custom fixed rules and rerankers.

Two differences in style:
- **Everything is async in TypeScript.** Calls run off the event loop.
- **Python's own language types** (bytes, tuples) become JSON values in
  TypeScript (`Uint8Array` for attachment bytes).
