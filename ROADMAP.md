# kleos — Roadmap

kleos is a memory engine for AI agents. It is derived from
[CozoDB](https://github.com/cozodb/cozo) via the mnestic fork (see [`NOTICE.md`](NOTICE.md)).
This is the public, forward-looking roadmap, updated 2026-10-25. Shipped detail lives in
[`CHANGELOG.md`](CHANGELOG.md), implementation contracts in [`docs/specs/`](docs/specs/),
measurements in [`docs/benchmarks/`](docs/benchmarks/), and the item-by-item research plan
(IDs such as E7 below) in [`UPGRADE_PLAN.md`](UPGRADE_PLAN.md).

## North star

An agent's memory should live in one engine, not be stitched together from a vector store, a
search index, a graph database, a workflow engine and an audit log. kleos combines:

- vector (HNSW), full-text (BM25) and graph retrieval, fused in one query;
- valid-time and transaction-time history, with facts superseded instead of overwritten
  ("what did the agent believe, and when?");
- a self-building ontology, entity resolution and consolidation into profiles and summaries;
- small built-in models (embedder, decider, entity extractor, OCR) that run in-process;
- durable workflows, memory branches and live subscriptions; and
- transactional storage, embedded or as a server, with no separate database service.

A change belongs in kleos when it makes that combination more correct, more robust, faster or
easier to adopt. Built-in models stay small (about 600M parameters or fewer); quality comes from
templates, calibration and verification, not large models.

## Current state

- Nothing is published under the kleos name yet (crates.io, PyPI and npm are held). Everything
  lives in this repository.
- Maintained surfaces: the Rust crates, the RocksDB bridge, the `kleos` server and CLI with a
  stateless MCP server, the Python package, the TypeScript SDK (napi-rs binding plus HTTP and
  MCP clients), and an Agent Skills plugin.
- Unmaintained and excluded from the workspace: the C, Java, Swift and WASM bindings.

## Shipped

Work shipped in waves; each wave merged with targeted tests and before/after measurements.

| Wave | What shipped |
|---|---|
| 1 | Filters inside HNSW search (A1); TurboQuant-style compressed vector codes (A2); weighted fusion and a re-rank stage (C1, C2); `embed()` and the `kleos-infer` engine with EmbeddingGemma and Model2Vec (D2, D3, D6); supersession and provenance (E1, E2); live query subscriptions (F1) |
| 2 | `decide()` with the built-in decider (D5, [Mazino0/kleos-decider-nano](https://huggingface.co/Mazino0/kleos-decider-nano)); the memory API: `remember` / `recall` / `forget` / `reflect` in every maintained binding (E10, H1); entity resolution (E3); memory scoring (E4, C6); contradiction candidates (E6); typed claims; Engram overlays fed by `remember` |
| 3 | Recall context blocks; date normalisation; the knowledge-update fix; the GLiNER entity extractor ([Mazino0/kleos-gliner](https://huggingface.co/Mazino0/kleos-gliner)) |
| 4 | Durable workflows `::flow` with retries, schedules, signals and child workflows (F4, F5); memory branches: fork, diff, merge, rewind (E9) |
| 5 | Ontologies that build and maintain themselves (cardinality-driven supersession, learning, maintenance, history and undo); a memory budget and concurrency lanes; the documents pipeline (G2): a Rust port of Indra OCR, Chonkie-style chunking, durable ingestion with page citations; EmbeddingGemma 2 as the default embedder, with `::memory reembed` |
| 6 | Graph recall: personalized PageRank (C4) and inferred ontology links, on by default; units of typed literals; consolidation into entity profiles and session summaries, automatic by default (E7); the stateless MCP server (protocol 2026-07-28), the `kleos` CLI and agent skills (J3); the TypeScript SDK (J2); native OCR kernels and streaming ingestion (500-page scanned book 590 → 217 s); hybrid fusion retuned on memory data; ontology quality fixes |

## In progress (wave 7)

Tenancy and access (E5, K2) merged on 2026-10-25 ([spec](docs/specs/tenancy.md)); finishing work continues (partition-aware workers, the remaining MCP tools as principals, JWKS and token exchange).


| Work | What it adds | Spec (lands with the branch) |
|---|---|---|
| **Skill library** (E8) | Procedural memory: Agent Skills with immutable versions and digests, task-to-skill routing within a token budget, skills over MCP (SEP-2640), success tracking, automatic rollback and retirement, a security scanner and signing, a learning loop from successful runs, a SkillOpt-style optimizer, promotion and a registry | `docs/specs/skills.md` |
| **Postgres coexistence connector** | Postgres logical replication into memory and ontology (rows become entities, foreign keys become edges, updates supersede, commit time is valid time), exactly-once resume, ledger-based write-back from workflows, a Debezium event adapter for other databases, and live joins through DataFusion | `docs/specs/pg-connector.md` |

## Next

- **Quality close-out:** re-measure LongMemEval QA with the lift-only graph re-rank; v2
  over-typing; summaries that drop superseded claims; a benchmark for inferred-link recall.
- **SQL front door:** DataFusion read-only SQL over kleos snapshots and the Postgres wire protocol
  (`pgwire`), so psql, drivers, ORMs and BI tools connect; it builds on tenancy so SQL respects
  the same access rules (H4, I5).

## Later

- **Postgres compatibility track:** SQL writes mapped onto kleos transactions, constraints,
  migrations (H5), roles and row-level security, replication and point-in-time recovery.
- **Code memory:** a tree-sitter symbols / calls / imports graph updated as files change, with edit
  history tied to conversations (G1).
- **Multimodal memory:** images and audio through EmbeddingGemma 2's vision and audio paths
  (G3, G4); EmbeddingGemma 2 speed.
- **Operations:** metrics, backups that keep search indexes, encryption at rest (K1, K3, K4).
- **Text search:** block-max WAND, highlighting, typo tolerance, code-aware tokens, more
  languages (B1–B4).
- **Vector search:** compressed codes in adjacency rows, Matryoshka indexing, in-place delete
  repair, sparse and multi-vector retrieval (A4–A11).
- **Incremental maintenance:** rules and graph projections updated from deltas (F2, F3).
- **Document re-ingestion:** page fingerprints, pipeline-version upgrades, version matching by
  similarity, a cross-namespace cache.
- **Engram serving:** loading Engram overlays into vLLM, SGLang and llama.cpp.
- **Platform:** a Studio UI, a hosted service, harness as a service.
- **Publishing:** crates.io, PyPI and npm releases (J1) when the owner decides.

## Scope boundaries

kleos will not pursue:

- hosting or training general-purpose generative models (generation goes through a configured
  external endpoint; built-in models stay small);
- full PostgreSQL parity: the aim is Postgres-compatible for agent applications, not a
  replacement for every Postgres workload;
- distributed consensus inside the embedded engine (replication, if it comes, belongs to the
  server);
- CRDT sync or browser/WASM persistence;
- native RDF/triple storage, SPARQL or OWL reasoning (RDF stays at the boundary); or
- a from-scratch full-text engine or embedded Tantivy.

## Contributing and releases

- Discuss larger features in an issue before implementation.
- Measure before and after every retrieval or speed change, and keep inherited semantics green.
- Use the SQLite backend for planner and stored-relation tests; see
  [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Preserve MPL-2.0 and the inherited CozoDB and mnestic attribution.
- Publish `kleos-rocks` before `kleos` whenever the bridge changes.
- Treat roadmap text as intent. Releases, registries, source, tests and benchmarks are the
  evidence.
