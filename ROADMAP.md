# kleos — Roadmap

kleos is an embedded memory engine for AI agents. It is derived from
[CozoDB](https://github.com/cozodb/cozo) via the mnestic fork (see [`NOTICE.md`](NOTICE.md)).
This is the public, forward-looking roadmap. Shipped detail lives in
[`CHANGELOG.md`](CHANGELOG.md), and implementation contracts live in [`docs/specs/`](docs/specs/).

## North star

An agent's long-term memory should live in one embedded engine, not be stitched together from
a vector store, a search index, a graph database and an audit log. kleos combines:

- vector (HNSW), full-text (BM25) and graph retrieval, fusable in a single query;
- valid-time and transaction-time history ("what did the agent believe, and when?");
- recursive Datalog inference with proof-carrying results;
- incremental indexes and budgeted, interruptible operations; and
- transactional storage with no separate database service.

A change belongs in kleos when it makes that combination more correct, more robust, faster or
easier to adopt. It has to be a general engine mechanism, not a special case for one consumer.

## Current state

- Code base: mnestic 0.18.0+ (`352bf205`), renamed. The engine is functionally identical to mnestic.
- **Nothing has been published under the kleos name yet.** The `kleos` crate, `kleos-rocks` and
  the `kleos` Python package exist in this repository only.
- Maintained surfaces: the Rust crate, the RocksDB bridge, the server/REPL binary, and the Python
  package with its LangChain, LlamaIndex, LangGraph and MCP integrations.
- Unmaintained and excluded from the workspace: the C, Java, Node.js, Swift and WASM bindings.

## Phase 1 — Hardening (current)

An embedded database runs inside its host's process, so its panics crash the host and its bugs
corrupt the host's data. Before adding features, kleos has to be boringly reliable.

| Work | Done when |
|---|---|
| Green CI on current stable Rust | Every gate in `build.yml` passes on the latest stable toolchain |
| Triage inherited upstream bugs | Each open CozoDB/mnestic bug is fixed with a regression test, or recorded as not reproducible |
| Panic-proof user-input paths | Malformed scripts, parameters, imports and stored data return errors instead of panicking. `unwrap`/`panic!` remain only where an invariant is proven |
| Fuzzing | `cargo-fuzz` targets cover the CozoScript parser, the query pipeline and the value codec, and CI runs a short fuzz pass |
| HNSW integrity verification | A read-only verifier detects damaged vector indexes (inherited mnestic #56). `::reindex` remains the repair path |
| Storage upgrade | RocksDB moves from v8.8.1 to a current release, with storage-compatibility tests against databases written by the old version |
| Release pipeline | The first kleos release publishes `kleos-rocks`, `kleos` and the Python package from tags, with the version gates green |

## Phase 2 — Candidate work (not yet scheduled)

These items are inherited from the mnestic backlog (mnestic issue numbers in brackets) or follow
from the north star. Phase 1 findings decide the order.

| Area | Item |
|---|---|
| Correctness | Diagnostic for surprising top-level temporary relations [#12]; sled valid-time scans [#5] |
| Operability | Versioned schema migrations with diff and rollback [#10]; backup and restore ergonomics; operational metrics |
| Performance | Baseline benchmark suite first; then plan caching, SQLite `WITHOUT ROWID` [#6], vector quantization with float rescore [#57] |
| Retrieval | Multi-token phrase-prefix FTS [#19]; phrases inside `NEAR` [#20]; filtered-vector search at low selectivity |
| Interop | Arrow query-result export [#11]; first-class ULID keys [#8] |
| Reach | Node.js/TypeScript bindings, the most-requested ecosystem for agent frameworks |

## Scope boundaries

kleos will not pursue:

- federation, virtualization, lakehouse breadth, or an extension marketplace;
- document, KV or time-series multi-model breadth, or multiple write languages;
- distributed clustering or consensus inside the embedded engine;
- Cypher writes, CRDT sync, or browser/WASM persistence;
- native RDF/triple storage, SPARQL, or OWL reasoning;
- worst-case-optimal joins or competitive cyclic-subgraph matching; or
- a from-scratch FTS engine or embedded Tantivy.

Boundary readers, read-only interoperability, and targeted general primitives remain in scope
when they preserve the relational/Datalog core.

## Contributing and releases

- Discuss larger features in an issue before implementation.
- Make performance work baseline-first and keep inherited semantics green.
- Use the SQLite backend for planner and stored-relation tests; see
  [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Preserve MPL-2.0 and the inherited CozoDB and mnestic attribution.
- Publish `kleos-rocks` before `kleos` whenever the bridge changes.
- Treat roadmap text as intent. Releases, registries, source, tests and CI are the evidence.
