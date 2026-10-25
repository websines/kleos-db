# kleos upgrade plan

*Drafted 2026-10-15 from a survey of 2025–26 research, competing products and the
kleos source. Priorities: **P0** = first wave, **P1** = next, **P2** = later.*

## Status (2026-10-25)

The plan below is the original research list, kept as written. Progress against it, in
the order of [`ROADMAP.md`](ROADMAP.md):

| Status | Items |
|---|---|
| **Done** | A1, A2; C1, C2, C4, C6; D1, D2, D3, D5, D6; E1, E2, E3, E4, E6, E7, E9, E10; F1, F4; G2; H1; J2, J3; L1 |
| **Partly done** | D4 (re-rank stage and hook; no built-in cross-encoder ships); D7 (late chunking and multimodal encoders not started; EmbeddingGemma 2 is in place for them); D8 (stores record their embedding model and `::memory reembed` migrates them; no general model registry); F5 (durable workflows run slow work outside transactions; no stored procedures); L2 (internal LoCoMo and LongMemEval runs in `docs/benchmarks/memory.md`; no published standard-judge run) |
| **In progress (wave 7)** | E5 and K2 (tenancy and access); E8 (skill library); plus a Postgres coexistence connector that was not in the original list |
| **Open** | A3–A11; B1–B4; C3, C5; F2, F3, F6, F7; G1, G3–G6; H2–H5; I1–I6; J4, J5; K1, K3, K4; L3 |
| **Held** | J1 (publishing to crates.io and PyPI waits for the owner) |

Built beyond the original list: self-building ontologies with units and inferred links;
the documents pipeline's Rust OCR port and chunkers; durable workflows; Engram overlays;
the GLiNER extractor; a memory budget with priority lanes; the stateless MCP server, CLI
and agent skills.

## Goal

Make kleos **the one embedded database an AI application or agent needs**: memory,
retrieval, knowledge graph, history, rules, built-in models and live updates in one
file and one transaction, under a permissive licence (MPL-2.0).

## Why now

- **Agent-memory products stitch 2–4 databases together and it hurts.**
  - Mem0 runs Postgres plus Turbopuffer and reconciles failed vector writes
    asynchronously
    ([blog](https://mem0.ai/blog/how-we-cut-vector-search-latency-by-70x)).
  - Cognee's three-engine default left documents half-indexed under concurrent
    writers ([#5254](https://github.com/topoteretes/cognee/issues/5254)).
  - Graphiti keeps its temporal logic in application code
    ([paper](https://arxiv.org/abs/2501.13956)).
- **The embedded graph slot is open.** Kùzu was archived in Oct 2025 after Apple
  acquired it
  ([BetaKit](https://betakit.com/apple-strikes-deal-to-acquire-canadian-database-software-startup-kuzu/)).
  Graphiti deprecated its Kùzu driver, and Cognee moved to a fork.
- **Licence is an advantage.** Close rivals use restrictive licences: SurrealDB
  (BSL), FalkorDB (SSPL), Memgraph (BSL), SpacetimeDB (BSL).

## Where kleos stands (verified in source)

**Already strong, and rare in one embedded engine:**

- Recursive Datalog.
- Bitemporal history: valid time plus `TxTime`, `:as_of`, `::history`.
- HNSW vector search, BM25 full-text search, MinHash LSH.
- In-engine RRF/MMR fusion and a hybrid-search builder.
- About 20 graph algorithms over cached graph projections.
- Provenance semirings, stored queries, change callbacks, triggers.
- Governed transactions.

**Behind:**

- **Vector search.** No quantization, F32/F64 only. The search filter runs *after*
  candidate retrieval, so selective filters return fewer than k results.
- **AI plumbing.** No built-in embeddings or re-ranking.
- **Memory semantics** (decay, namespaces, entity resolution, consolidation) exist
  only in the Python integrations.
- **Bindings.** Only Python is maintained.
- **Visibility.** No published benchmarks. CozoScript is unfamiliar to LLMs.

## What the research says wins (2025–26)

- **Fused multi-channel retrieval.** Vector, BM25, graph and time results are fused
  with RRF and then re-ranked by a cross-encoder. Every current memory-benchmark
  leader does this: Hindsight ([2512.12818](https://arxiv.org/abs/2512.12818)),
  Mem0 2026, Zep.
- **Facts are invalidated, never overwritten,** and keep provenance back to their
  source. Zep's bi-temporal edges are the reference design.
- **Background consolidation.** Episodes are turned into observations and
  summaries. This is the largest single driver of BEAM results
  ([2510.27246](https://arxiv.org/abs/2510.27246)).
- **Multi-hop fact updating is unsolved.** Every system scores ≤7% on
  MemoryAgentBench FactConsolidation multi-hop
  ([2507.05257](https://arxiv.org/abs/2507.05257)).
- **Filters applied during the search.** ACORN
  ([2403.04871](https://arxiv.org/abs/2403.04871)), pgvector 0.8 iterative scans.
- **Compressed vectors with exact re-scoring.** TurboQuant
  ([2504.19874](https://arxiv.org/abs/2504.19874)), RaBitQ.
  - Binary codes plus rescoring keep about 96% of quality.
  - int8 codes plus rescoring keep about 99%
    ([HF](https://huggingface.co/blog/embedding-quantization)).
- **System-One decision models.** These return typed, calibrated decisions instead
  of text:
  - [Jev](https://www.datacamp.com/blog/system-one-models-jev): proprietary, hosted.
  - [d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M): open weights,
    `lfm1.0` licence.
  - [DM-JEPA](https://systemonemodels.ai/danger-labs/dm-jepa): 308M.

  They fit the decisions memory systems currently pay an LLM for: supersede,
  duplicate, same entity, importance.
- **Ideas from SpacetimeDB 2.x**
  ([repo](https://github.com/clockworklabs/SpacetimeDB), BSL, so ideas only and no
  code):
  - Live query subscriptions that push deltas.
  - Schedule tables.
  - A split between short transactional procedures and slow ones that never hold
    locks.
  - Event tables.

---

## A. Vector search

| # | Upgrade | Gain | Pri |
|---|---|---|---|
| A1 | Filters applied **inside** HNSW search: ACORN-1 two-hop expansion, brute force below a selectivity threshold, keep scanning until k pass | Fixes filtered queries silently returning fewer than k results; nearly every agent query is filtered | P0 |
| A2 | **TurboQuant / RotorQuant** compressed vectors (build on [turbovec](https://github.com/RyanCodrai/turbovec/wiki) and [turbo-quant](https://docs.rs/turbo-quant), both MIT), choice of rotation (dense, Hadamard, [RotorQuant](https://github.com/scrya-com/rotorquant) blocks), full-precision re-scoring; pick the default by measured recall | 4–32× smaller, faster search | P0 |
| A3 | Vector types F16, BF16, I8, binary | Halves storage at about equal recall | P0 |
| A4 | Compressed codes stored in HNSW adjacency rows (SymphonyQG, [2411.12229](https://arxiv.org/abs/2411.12229)) | Traversal without fetching full vectors | P1 |
| A5 | Matryoshka indexing: index a prefix, re-score with the full vector | Cheap speedup on modern embedders | P1 |
| A6 | In-place delete repair (MERIT [2607.29173](https://arxiv.org/abs/2607.29173), IP-DiskANN [2502.13826](https://arxiv.org/abs/2502.13826)) on top of today's repair | Memory workloads delete constantly | P1 |
| A7 | Sparse vectors (SPLADE / BGE-M3) with a Seismic-style index | Better lexical recall than BM25 | P1 |
| A8 | Multi-vector (ColBERT / ColPali) via MUVERA ([2405.19504](https://arxiv.org/abs/2405.19504)) plus MaxSim re-rank and token pooling | Document and visual retrieval | P2 |
| A9 | Time-window kNN (range-filtered index) | Pairs with time travel | P2 |
| A10 | Larger-than-RAM layout (LSM-VEC [2505.17152](https://arxiv.org/abs/2505.17152), Starling) and a fast-build IVF index type | Scale past about 10M vectors | P2 |
| A11 | Vector search "as of" a past time | Search over memory history | P2 |

## B. Text search

| # | Upgrade | Pri |
|---|---|---|
| B1 | Block-max WAND or precomputed impacts (3–15× faster disjunctions in Lucene) | P1 |
| B2 | `highlight()`, typo tolerance, multi-token phrase prefix (#19), phrases inside NEAR (#20) | P1 |
| B3 | Code-aware tokenizer: split `camelCase` / `snake_case`, index symbols | P1 |
| B4 | More languages (Japanese, Korean, …) | P2 |

## C. Hybrid search and ranking

| # | Upgrade | Pri |
|---|---|---|
| C1 | Weighted (convex) fusion with score normalization beside RRF ([2210.11934](https://arxiv.org/abs/2210.11934)) | P0 |
| C2 | Re-rank step in the hybrid pipeline | P0 |
| C3 | Hybrid search as a real in-engine operator (today a script generator), with the planner ordering legs | P1 |
| C4 | HippoRAG-style retrieval ([2502.14802](https://arxiv.org/abs/2502.14802)): vector/BM25 hits seed per-query personalized PageRank | P1 |
| C5 | Spreading activation with per-edge-type weights and decay | P1 |
| C6 | Recency, importance and access counts usable in ranking expressions | P1 |

## D. Built-in AI: a Rust inference engine for single-pass models

Embedders, cross-encoder re-rankers and System-One decision models all run one
forward pass, with no autoregressive generation, KV cache or sampling. One small
dedicated engine can power all three.

| # | Upgrade | Pri |
|---|---|---|
| D1 | **Feasibility check:** d1-omni-600M licence terms, laptop latency, accuracy on "does fact B supersede fact A" | P0 |
| D2 | Engine: one forward pass, f16/int8 weights, optional Metal/CUDA; models are optional downloads, never in the default binary | P0 |
| D3 | `embed()`, auto-embed on insert, embedding cache keyed on (model, text hash) | P0 |
| D4 | `rerank()` with a cross-encoder (bge-reranker, Qwen3-Reranker-0.6B) | P1 |
| D5 | `decide()`: typed yes/no, choice and score with calibrated confidence; backends are built-in (d1-omni, DM-JEPA), remote (Jev) or an LLM fallback | P1 |
| D6 | Zero-download default embedder (Model2Vec static embeddings) | P1 |
| D7 | Late chunking, sparse output, multimodal encoders (SigLIP, CLAP) | P2 |
| D8 | Versioned model registry with re-embed migrations | P2 |

## E. Memory and knowledge graph

| # | Upgrade | Pri |
|---|---|---|
| E1 | **Supersession:** a new fact ends the old fact's validity and is never deleted; queries for "true at T", "known at T" and "changed between T1 and T2" (built on existing Validity and TxTime) | P0 |
| E2 | Provenance: every fact links to its source; removing a source invalidates or deletes what was derived from it (GDPR) | P0 |
| E3 | Entity resolution: candidates from LSH, vector and FTS, confirmed by `decide()`, merged with history kept | P1 |
| E4 | Memory scoring: recency decay, importance, batched access counts, TTL, archive tier | P1 |
| E5 | Namespaces and tenants, with access rules enforced inside vector, FTS and graph search | P1 |
| E6 | Contradiction detection (rules plus `decide()`) | P1 |
| E7 | Consolidation jobs: episodes into observations, summaries and profiles | P2 |
| E8 | Skill / procedural memory with success and failure stats and versions | P2 |
| E9 | Snapshots, branches and rollback of memory state | P2 |
| E10 | Ready-made **memory module**: schema plus remember / recall / forget / reflect procedures, so agents never write Datalog | P2 |

## F. Real-time engine

| # | Upgrade | Pri |
|---|---|---|
| F1 | **Live query subscriptions** pushing deltas; identical queries run once and shared | P0 |
| F2 | Incrementally maintained rules: monotone strata first (semi-naive continuation), then deletes (DRed), later DBSP-style | P1 |
| F3 | **Incremental graph updates**: communities (label propagation), PageRank and projections updated from deltas instead of rebuilt | P1 |
| F4 | Scheduled jobs stored as rows (decay, consolidation, reindex) plus host-driven `run_due(budget)` | P1 |
| F5 | Stored procedures: short transactional ones, and slow ones that call models without holding locks | P1 |
| F6 | Event relations: broadcast and logged, never stored | P2 |
| F7 | Group commit; notify only durable changes | P2 |

## G. Domains

| # | Upgrade | Pri |
|---|---|---|
| G1 | **Code:** tree-sitter parsing into a symbols, calls and imports graph, updated incrementally as files change | P1 |
| G2 | Documents: structure-aware chunking for PDF, HTML and Markdown; tables | P1 |
| G3 | Images: blob store, SigLIP embeddings, ColPali for page images | P2 |
| G4 | Audio: transcripts and audio embeddings | P2 |
| G5 | JSON path indexing | P2 |
| G6 | Content-addressed file store (deduplicated by hash) | P2 |

## H. Developer experience

| # | Upgrade | Pri |
|---|---|---|
| H1 | Simple API in every binding (remember, recall, link) that needs no CozoScript | P1 |
| H2 | Broader Cypher: variable-length paths, OPTIONAL MATCH, WITH | P1 |
| H3 | LLM-oriented docs (AGENT.md, query cookbook) and better error messages | P1 |
| H4 | SQL front door for a subset of queries | P2 |
| H5 | Schema migrations (#10) | P2 |

## I. Storage and performance

| # | Upgrade | Pri |
|---|---|---|
| I1 | RocksDB 8.8.1 → current release | P1 |
| I2 | Benchmark suite (ANN datasets, LDBC, memory tasks) in CI | P1 |
| I3 | Plan caching; SQLite `WITHOUT ROWID` (#6) | P2 |
| I4 | Multi-process access (writer service or server mode) | P2 |
| I5 | Arrow export of query results (#11) | P2 |
| I6 | Object-storage tier and sync | P2 |

## J. Reach

| # | Upgrade | Pri |
|---|---|---|
| J1 | Publish as **kleos** on crates.io and PyPI, with a release pipeline | P0 |
| J2 | TypeScript/Node bindings (napi-rs); refreshed WASM build | P1 |
| J3 | Native Rust MCP server | P1 |
| J4 | Integrations: Graphiti driver (Kùzu replacement), Cognee adapter, Mem0 store (LangGraph and LlamaIndex already exist) | P1 |
| J5 | Swift and Kotlin bindings for on-device use | P2 |

## K. Operations and security

| # | Upgrade | Pri |
|---|---|---|
| K1 | Metrics endpoint and query statistics | P1 |
| K2 | Per-tenant auth and access rules in the server | P1 |
| K3 | Backups that preserve search indexes; retention and compaction for history | P2 |
| K4 | Encryption at rest | P2 |

## L. Proof

| # | Upgrade | Pri |
|---|---|---|
| L1 | Recall, latency and size measured before and after every retrieval change | P0 |
| L2 | Published LongMemEval, LoCoMo and BEAM scores, using a reference memory agent | P1 |
| L3 | Docs site and examples | P1 |

---

## First wave (P0)

1. **A1:** filters inside vector search.
2. **A2, A3:** TurboQuant/RotorQuant compressed vectors and the new vector types.
3. **C1, C2:** weighted fusion and re-ranking.
4. **D1–D3:** model feasibility check, inference engine, `embed()`.
5. **E1, E2:** supersession and provenance.
6. **F1:** live subscriptions.
7. **J1:** publish as kleos.
8. **L1:** before/after retrieval measurements.

Each item ships on its own, with measurements, and finishes with a check-in before
the next one starts.

## Out of scope

- Hosting or training general-purpose generative models.
- A distributed cloud service.
- Full SQL parity.

"All things AI" here means everything an agent's memory and knowledge need, in one
embedded file.
