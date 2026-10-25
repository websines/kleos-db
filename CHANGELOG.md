# kleos changelog

kleos was created from mnestic v0.18.0 (commit `352bf205`, 2026-09-08). See
[`NOTICE.md`](NOTICE.md) for provenance and licensing.

Everything below the first `Unreleased` section is mnestic's own release history,
kept verbatim for reference; "mnestic" there refers to the upstream project, not
to kleos.

## Unreleased

### Added
- **Tenancy, identity and access control** (docs/specs/tenancy.md; wave7/tenancy). Off until used: a store with no tenancy configured behaves exactly as before.
  - **Model.** Orgs → workspaces; org-level teams that nest (child teams inherit) and are granted `reader` / `writer` / `admin` on workspaces; principals of kind `human`, `agent` (mandatory sponsor, optional template) and `service`. Admin ops: `::tenant <op>`, `Db::tenant_call`.
  - **Physical isolation.** Each workspace is a partition (`Db::partition_view`): its memories, HNSW and FTS indexes, ontology graph, derived data, flows and documents live in their own relations. Inside it, a scope (`run`, `agent_user`, `user`, `agent`, `team`, `workspace`; `org` in an org partition) is a memory namespace, so supersession, duplicates, entities and per-scope profiles and summaries never cross scopes.
  - **Authorization.** Embedded Cedar (`cedar-policy` 4.14) decides `(actor, action, scope)` over the candidate scopes of a request; effective access is human ∩ agent ∩ credential ∩ every session participant (optional sponsor ceiling for autonomous agents); org policies add `forbid` rules. The resulting ScopeSet is the pre-filter of the HNSW and FTS searches and bounds the PPR walk, entity expansion and profile composition; hits carry `scope`, and a broader scope's restatement or same-slot value gives way to the narrower one.
  - **Principals everywhere.** `PrincipalContext {workspace, human, agent, run, participants, chain, token}` with per-link assurance (`verified`, `issuer_endorsed`, `host_asserted`, `self_asserted`). Memory ops as a principal: `::principal <op>`, `Db::memory_call_as` (`remember` defaults to the narrowest scope: agent + human → `agent_user`). Every memory records its writer (`kleos_mem_actor`, memory schema v9).
  - **Promotion** is a copy with provenance (`kleos_mem_promoted`): direct for humans who can write the target and agents with the `promote` grant, else a pending request a human approves; org scope always needs an org admin's approval. Learned ontology properties are promoted to the workspace layer automatically.
  - **Audit.** Append-only, SHA-256 hash-chained per workspace: every write committed in a principal view (memories, automatic ontology changes, consolidation, documents, flows, branches, Engram jobs), op events with target ids, and reads with the returned ids; principal ids are HMAC pseudonyms. `::tenant audit`, `audit_verify`.
  - **Erasure** (`::tenant erase`): a principal's own scopes and what it authored elsewhere are erased through the `forget` cascade (vectors, summaries, profiles, and now ontology facts, nodes and types citing the memories), caches cleared, its pseudonym key destroyed (crypto-shredding).
  - **Follow-up (closing the first cut's deferred items).** Audit write events commit inside the writing transaction (chain head relation, no crash gap); the flow worker and the document and consolidation runners serve every workspace and org partition automatically; `forget` / `set` / `link` reach agent and org scopes (curators act, others file requests a human approves); shared sessions write to a session scope readable only by its members; a selective search's exact scan reads only the caller's scope namespaces (HNSW `prefixes`), BM25 `N` and document frequencies are counted per ScopeSet (FTS `prefixes`), and cached embeddings are keyed by workspace and ScopeSet; every MCP tool runs as the principal (`query` and `reembed` for workspace admins); JWKS URLs and OIDC discovery with caching and key rotation, SPIFFE JWT-SVIDs, Transaction Tokens, DPoP, and an RFC 8693 token-exchange endpoint issuing delegation tokens with nested `act` (actors via `private_key_jwt`, `::tenant principal_key_set`); OpenTelemetry GenAI spans of audited ops (feature `otel`). Cedar partial evaluation is not used (justified in the spec: concrete evaluation over the pre-filtered candidates costs 0.7 ms uncached).
  - **Server and MCP** (`kleos server --tenancy config.json`): RFC 9068 JWT access tokens with OAuth Actor Profile `act` chains (audience-checked, never forwarded), per-principal API keys, RFC 9728 metadata at `/.well-known/oauth-protected-resource`, `/tenant/:op`; MCP memory tools run as the principal. CLI: `kleos tenant`, `--as-human` / `--as-agent` / `--workspace` / `--scope`. Python `tenant` / `memory_as`; TypeScript `tenant` / `memoryAs`.

### Changed
- **Graph recall and consolidation are on by default** (the user's choice): `recall` defaults to `graph: 'ppr'` (which also follows inferred ontology links; `graph: 'off'` opts out), and each namespace consolidates automatically after every 200 new memories (`every: 0` opts out), run by a lazily started in-process thread (`DbInstance::consolidate_kick`) on the flow queue `consolidate`; profiles and summaries appear in the context block once built. The auto-consolidation counters are per open store, not per process.
- **Ontology quality (wave6/quality).** A thin claim (nothing concrete, under 5 content words) never auto-supersedes a substantive one, however the pair is typed; it is capped at pending ("It's definitely isn't, Maria." no longer ends "Investing in our future generations is key…"), and a name closing its sentence after a comma counts as a vocative. Maintenance (cardinality flips, merges, retirement) counts only cue-, rule-, learning- or extractor-backed types as evidence, never prototype-only ones. A memory branch can retract a fact it inherited (a tombstone hides the ancestor's copy from the branch's `facts` and `infer`; the parent keeps it).

### Fixed
- **`graph: 'ppr'` sank the current value on knowledge-update questions:** the PPR re-rank replaced each pool row's relevance with the blend `(1-w)·r + w·ppr`, so an older, better-linked value could outrank the newest one (LongMemEval KU latest-first, EmbeddingGemma 2: 83% with `graph: 'off'`, 74% with `ppr`). The walk now only lifts a row (`max(r, blend)`) and rows outside the neighbourhood keep `r`; KU is back to 83% (91% hide-pending) with `ppr` on. See docs/benchmarks/memory.md, "PPR and the current value".
- **Memory text search missed words whose stem is an English stopword** (`billing` → `bill`, `running` → `run`): the memory FTS index ran the stemmer before the built-in `'en'` stopword list, which also holds content words. The index now drops a short list of function words before stemming; stores rebuild the index on first use (memory schema v8).

### Added
- **Scanned-page preprocessing:** the skew search's rotated row sums skip the area outside the ink's bounding box and read interior pixels directly (about 20% faster), the area resize streams one source row at a time (about 15% faster, one row of memory instead of the whole intermediate image), and the dark-line check reads its histogram in place; results are bit-identical (equality tests against the original kernels, OCR parity unchanged).
- **TypeScript SDK:** `hybridSearch`, `decide`, `exportRelations`/`importRelations`, `backup`/`restore`/`importFromBackup` (native `admin_call`; remote maps export/import/backup to the server routes).
- **Documents by URL** (feature `docs-url`, off by default): `remember` attachments accept `http(s)` URLs, with public-address checks on every redirect hop, a 100 MB cap, timeouts and a document content-type check; the URL is kept in the document's metadata.
- **Documents:** with an extractor configured, document chunks get entities (mentions, resolution, merges) in their batch's transaction; no kinship edges from documents.
- **MCP:** `ontology` tool (list, facts, infer, learn, history, undo with confirmation), and `reembed` and `consolidate` tools returning tasks (an in-process worker runs them for embedded stores).
- **Graph recall** (docs/specs/memory-api.md, "Graph recall"; `kleos-core/src/memory_graph.rs`).
  - `recall {graph: 'inferred'}` pulls in memories linked to the question's nodes ("my", "my sister", names) and the top hits' subjects through the ontology's facts, inferred ones included ("Who are my relatives?" reaches the family through the new `relative_of` super-property). Hits say why (`via: "inferred: relative_of"`), in the list and the context block.
  - `recall {graph: 'ppr', graph_weight}` re-ranks with a personalized PageRank (HippoRAG 2) over the memory / entity / ontology-node neighbourhood of the hits: the pool seeds the walk by relevance, hub entities and nodes are cut, at most 2,000 memories; it also runs the inferred expansion.
  - Both off by default: on LoCoMo (301 QA, GLiNER entities) `ppr` was 0.764 against 0.774 and `inferred` 0.771; on LongMemEval (119) `ppr` was 0.647 against 0.605 (multi-session Rall@10 0.792 → 0.917). See docs/benchmarks/memory.md, "Graph recall and consolidation".
- **Units of typed literals** (docs/specs/ontology.md, "Units"; `kleos-core/src/ontology_units.rs`): length, mass, temperature, duration, volume, speed and data convert to base units; currencies compare per ISO code (no exchange rates). Validation accepts convertible units, a single-valued value restated in another unit is not a change, `::ontology convert`.
- **Consolidation** (docs/specs/consolidation.md, UPGRADE_PLAN E7; `kleos-core/src/memory_consolidate.rs`).
  - `::memory consolidate` builds **entity profiles** (current ontology facts and cue/rule-typed claims per node, with valid-from and evidence) and **extractive session summaries** (centrality + MMR per conversation; memories of kind `summary`, role `other`, never authoritative). An OpenAI-compatible endpoint can make the summaries abstractive (off by default, `requests` feature).
  - Rebuilt when their sources change; erased with them (`forget {mode: 'erase'}`).
  - The context block shows the profiles of the hits' subjects and the summaries of their conversations ahead of the memories (`profiles`, `summaries` options); summary memories are hits only with `kind: 'summary'`.
  - Automatic as the durable workflow `memory.consolidate` (built-in flow step `consolidate`): every N new memories (`every`) or on a schedule (`cron`).
  - LoCoMo QA with summaries in the context block: 0.767 against 0.748 (multi-hop 0.673 → 0.727).
- **TypeScript SDK (`sdk/ts`, package `kleos`).** See docs/sdk/typescript/.
  - **One client over two backends.** `open({ path })` runs an embedded store through the native module; `connect({ url, token })` talks to a `kleos server` over HTTP. Both have the same API and types, so switching is one line. `kleos/remote` is a fetch-only entry for serverless, edge and browser bundles.
  - **API.**
    - Memory: `remember` (claims, entities, attachments, priority), `recall`, `recallContext`, `forget`, `reflect`, `set`, `flush`, `link` and `profile`.
    - Branches: `fork`, `diff`, `merge`, `rewind`, `discard` and `branches`, plus `reembed`.
    - Ontology, documents, workflows and queries: `ontology`, `docs`, `flow`, `flowWorker` and `run`.
    - Models: enabling and disabling embedders, deciders and extractors.
    
    Options take camelCase or snake_case; results are kleos's JSON. Errors are `KleosError` with `op` and (remote) `status`.
  - **MCP client.** A stateless client for protocol 2026-07-28 over Streamable HTTP and stdio (`kleos mcp`), with multi-round-trip confirmations (`onInputRequired`) and the Tasks extension.
  - **Native binding.** `kleos-lib-nodejs` moves from neon to napi-rs. It is a thin JSON layer over `DbInstance::memory_call`, `ontology_call`, `flow_call` and `doc_call` and `run_script`, and every call runs on the libuv pool, so the event loop is never blocked. Node-API 8 means one build covers Node 18–24. Features: compact by default, with `infer`, `infer-metal`, `docs` and `engram` optional.
  - **Tests (`npm test`).** Both backends, MCP over HTTP and stdio, an event-loop test, a parity test against the Python binding, every example under `examples/ts/`, and a check that the code in the docs matches those examples.
- **Agent surfaces: stateless MCP server, `kleos` CLI and skills** (docs/specs/mcp-cli-skills.md).
  - **MCP, protocol 2026-07-28**, implemented in `kleos-bin` (`mcp.rs`, no SDK): no sessions or `initialize` for modern clients (per-request `_meta` version and capabilities, `server/discover`, `UnsupportedProtocolVersion` -32022), every result typed (`resultType`) and signed with `serverInfo`, `tools/list` / `resources/*` in a fixed order with `ttlMs` and `cacheScope`. Tools: `recall` (context block by default), `remember`, `forget`, `reflect`, `entity_profile`, `doc_add`, `doc_status`, `flow`, `branch`, and read-only `query` behind `--allow-query`. Destructive calls (`forget`, `branch` merge/rewind/discard) confirm through Multi Round-Trip Requests with an HMAC-signed, expiring, principal- and call-bound `requestState` (`KLEOS_MCP_STATE_KEY` to share across instances), or `confirm: true` for clients without elicitation. Large `doc_add` calls return a Tasks-extension task whose id is the ingestion workflow (`tasks/get`, `tasks/cancel`). Resources: namespaces, documents, entity profiles. Dual-era: `initialize` from 2025-11-25 / 2025-06-18 / 2025-03-26 clients selects legacy semantics without a session.
  - **Transports:** `kleos mcp` (stdio) and `kleos server --mcp` (`POST /mcp`: `MCP-Protocol-Version`, `Mcp-Method` and `Mcp-Name` validated against the body, base64 sentinel decoded, `HeaderMismatch` -32020; foreign `Origin` 403; notifications 202; `GET`/`DELETE` 405; session ids ignored, never minted). The server also accepts its admin secret as `Authorization: Bearer`.
  - **CLI:** `kleos remember | recall | forget | reflect | profile | doc add/status/list | ontology list/learn/history/undo | flow start/status/signal/cancel/list | branch fork/diff/merge/rewind/discard/list | query | mcp`, embedded (`--db`, `--engine`) or remote (`--url`, `--token`), `--json` output, a config file (`$KLEOS_CONFIG` or `~/.config/kleos/config.json`), confirmation on a terminal or `--yes`. Reads of a store with no memories yet answer empty.
  - **Skills and plugin:** `skills/kleos-memory`, `kleos-documents`, `kleos-ontology`, `kleos-branches`, and a Claude Code plugin manifest (`.claude-plugin/plugin.json`) with the MCP server entry. Context cost: the MCP tool list is about 1.5k tokens on every turn; the four skills' names and descriptions about 0.3k, with bodies (about 1.5k together) loaded only when used.
- **Faster documents** (docs/specs/documents.md, "Speed work (wave 6)").
  - **Native kernels for Indra's nets:** the recognizers (`ocr::fastnn`, channels-last GEMMs on Accelerate/AMX, vectorized depthwise, fused HardSwish / masked squeeze-excite / residual, native top-k) and the detector and orientation model (`ocr::convnet`, a small NCHW interpreter with hand-written convolutions). Identical text to tract on the parity sets; tract stays as the reference (`KLEOS_OCR_TRACT=1`) and the fallback. Recognizer batch 500 → 70 ms, detector 194 → 54 ms, orientation 93 → 21 ms; whole scanned book 1.58× Python indra-ocr under the same load.
  - **Digital-page geometry fast path:** simple text-layer pages skip the render and the line detector (`KLEOS_DOC_FAST=0` turns it off); 9/11 Report 111 → 33 ms/page with a 0.998 mean word ratio against the full path; forms and tables keep the full path.
  - **Streaming ingestion:** pages flow from one streaming extraction into chunk → embed → commit with a bounded queue; batches still commit as durable steps, so resume is unchanged. Scanned 500-page book end to end 590 → 217 s.
- **Zero-maintenance ontologies** (docs/specs/ontology.md, "Defaults", "Learning", "Maintenance").
  - **On by default:** every memory namespace gets the personal pack unless the store turns it off (`memory_init {ontology: 'none'}`); memory branches inherit their ancestor's ontology and read facts and inference through the lineage as of the fork.
  - **Learning:** `::ontology learn` clusters the claims nothing typed (and document chunks) and proposes a property per coherent cluster (label and cues from its distinctive words, cardinality and time-variance from the data and the decider) or new vocabulary for an existing one. Auto mode (default) applies strong proposals to the namespace's learned layer `learned:<ns>`; review mode keeps them for `approve` / `reject`. `remember` runs it every `KLEOS_ONTO_LEARN_EVERY` (200) untyped claims.
  - **Maintenance:** `::ontology maintain` makes a single-valued property multi-valued when the data keeps holding several current values per subject, merges learned synonyms, retires unused learned properties.
  - **Every change is explained and reversible:** `kleos_onto_change` (before/after, evidence memory ids), `::ontology history`, `::ontology undo`.
  - `::ontology suggest` points documents and claims at domain packs (FIBO, schema.org medical, LKIF, schema.org software) as review-only proposals.
  - A domain ontology's classes reach the extractor as extra labels (`EntityExtractor::extract_labels`; GLiNER reads them zero-shot).
  - One source of truth for kinship: with an ontology active, extractor kinship and role edges become ontology facts instead of `kleos_mem_link` rows.
  - **Per-embedder calibration:** typing scores and learning thresholds are mapped onto the reference (EmbeddingGemma v1) scale from the active embedder's own prototype baseline and claim anchors (40 built-in calibration statements), so they recalibrate after a model swap or `::memory reembed`; a prototype-only type never keeps a pair out of `supersedes`. Under EmbeddingGemma 2, KU candidates kept out of `supersedes` fell from 14.7 to 1.65 per question (v1: 1.17).
- **Decider:** the supersedes auto threshold for nano-tune-v1 is 0.995 (was 0.997; `KLEOS_SUPERSEDES_TAU` overrides).
- **Durable workflows:** `continue_as_new` (worker op, built-in step, Python `ctx.continue_as_new`) restarts a workflow with a fresh history under the same id; Python `ctx.step(..., isolate=True)` runs an attempt in a child process that a timeout terminates.
- **Dates:** "on the 15th", "August last year", decades ("the 90s", "the 1990s", "the nineties"), parts of seasons ("the end of the summer", "early spring").
- **EmbeddingGemma 2 is the default embedder.** `model: 'embeddinggemma'` (also `gemma`) is now `google/embeddinggemma-2` (Apache-2.0, not gated: auto-download needs no token). `kleos-infer::Egemma2Embedder` ports its text path (Gemma 4 encoder, 24 layers at d 512, per-layer embeddings, 512 → 768 projection, mean pooling, Matryoshka output) and loads only the text tensors (~271M of the checkpoint's 740M). EmbeddingGemma v1 stays selectable as `embeddinggemma-300m` / `gemma-v1`.
  - **Stores never mix models.** The memory store records the model behind its vectors (`embedder_model`); an enabled embedder of another model is refused with the fix in the error.
  - **`::memory reembed {embedder}`** moves a store's vectors to another model as a durable workflow: batches into a shadow relation, resumable after a crash, catch-up and switch in one transaction.
- **Documents in the memory API** (docs/specs/documents.md, feature `docs`). `remember(text, attachments=[...])` / `remember(file=...)` read PDFs, images, Markdown, HTML and DOCX into chunk memories (`kind: 'doc'`, role `other`) with page-level provenance; `recall` searches them by default and cites them (`[name p. 14, § Heading]`); `sources` filters; `::doc get/list/delete`, Python `doc_call`, HTTP `POST /doc/:op`.
  - **OCR:** `kleos-docs`, a Rust port of Indra OCR (code Apache-2.0; weights modified OpenRAIL-M, downloaded at run time) on tract; identical output to indra-ocr 1.1.2 on 27 fixture pages and 20 scanned-book pages. PDF text layers and renders through pdfium (downloaded pinned build, one library copy per page worker). Pluggable `OcrBackend`.
  - **Chunking:** Chonkie-style token, sentence, recursive and semantic chunkers (identical to Chonkie 1.7 on 72 cases) applied over layout blocks: tables whole, headings attached, provenance per chunk.
  - **Durable ingestion:** large documents run as the `doc.ingest` workflow; each page batch commits its chunks with its checkpoint, so a killed process resumes with no duplicate or missing chunk; re-ingesting the same bytes is a no-op, changed bytes keep unchanged chunks; ingestion runs at bulk priority.
  - **Results:** 585-page digital report 1.38 pages/s end to end, 500-page scanned book 0.85 pages/s, peak RSS ≤ 2.8 GB, recall hit@5 10/10 on both (see the spec's Results).

### Fixed
- **A full bounded change subscription no longer blocks writers** (`runtime/callback.rs`): delivery uses `try_send`; the subscription misses changes and a one-time `callback.lagged` diagnostic is emitted. The document and flow workers' own wake-up subscriptions could deadlock their writes.

### Added (earlier)
- **Memory-bounded concurrent inference** (docs/specs/memory-api.md, "Concurrency and memory budget").
  - **One budget for all built-in models.** `KLEOS_MEMORY_BUDGET` covers weights plus activations; `KLEOS_ACTIVATION_CAP` caps one forward pass, and every model sizes its batches to fit it.
  - **Model slots.** Lazy weights, LRU eviction under the budget, and idle unload (`KLEOS_MODEL_IDLE_SECS`).
  - **Priority lanes.** `remember`/`recall` `priority`, plus `kleos::sched::with_priority`. Interactive goes first; bulk keeps a device-time share and is sliced while interactive traffic is recent.
  - **Cross-request batching for EmbeddingGemma,** with bounded per-lane queues (backpressure). The decider and GLiNER take the priority gate per forward pass.
  - **Shared instances.** One model instance per (repo, engine) across stores.
  - **Measured on the mixed workload** (bulk document chunks, 50 conversations and traces):
    - peak footprint above base 734 → 219 MB;
    - interactive recall p95 3.34 → 1.96 s;
    - conversations done in 23 s instead of 48 s.
- **Ontologies (`::ontology`).** Pluggable per-namespace ontologies with valid time; the personal-memory ontology is only the default pack. See docs/specs/ontology.md.
  - **Storage.** `kleos_onto`, `kleos_onto_class`, `kleos_onto_prop` (kind, domain, range, datatype, cardinality, symmetric, transitive, inverse, sub-property, chains, temporal, unit, aliases, cues), `kleos_onto_ns` (active ontologies per namespace, later ones overriding earlier ones by name, plus the supersession policy), `kleos_onto_fact` and `kleos_onto_node` (facts and node classes, with valid time, status `current` or `pending`), `kleos_mem_type` (claim types). Created on first use; no memory schema bump.
  - **Default pack `personal`:** 65 properties over 9 classes, generated by `scripts/ontology/gen_personal_pack.py`.
  - **Claim typing in `remember`** against the active ontology: small talk by rule, properties by cue phrases plus embedding prototypes (the claim vector `remember` already has), a subject (speaker, `user`, `user.sister`, or a non-blocking `name:` guess). Returned in `types`.
  - **Cardinality supersession policy** (`policy: 'cardinality'`, default when an ontology is active): small talk is never decided; different speakers or subjects are never decided against each other; a multi-valued property, or different confidently typed properties, never supersede (`duplicate` only); a non-temporal single-valued property caps a supersede at pending; a pair with an untyped side follows `untyped` (`decide`, `pending`, `dup_only`), and a thin untyped pair (nothing concrete, under five content words) is capped at pending. `policy: 'off'` is today's behaviour, as is a namespace with no ontology.
  - **Kinship and role facts by rule** ("my sister Anna" → `sister_of(anna, user)`, `same_as(user.sister, anna)`) written in `remember`'s transaction.
  - **Inference:** a Datalog program generated from the active ontology (symmetric, inverse, sub-property, transitive, property chains, `same_as`), so it works for any imported ontology. **Validation:** property, domain, range, datatype, unit and cardinality.
  - **N-ary facts:** event nodes with role facts and typed literals (XSD datatypes, units, ISO 4217 currencies); single-valued roles are enforced per event.
  - **Import/export:** JSON, and an OWL/RDF subset in Turtle (classes, subClassOf, object and datatype properties, domain, range, Functional/Symmetric/Transitive, inverseOf, subPropertyOf, propertyChainAxiom, rdfs/SKOS labels and altLabels). FIBO (MIT) is the finance reference example; not vendored.
  - **Surfaces:** `::ontology <op> {…}` (define, import, export, get, list, activate, deactivate, assert, retract, facts, infer, validate, type, gate), `Db::ontology_call`, `POST /ontology/:op`, `CozoDbPy.ontology_call`; `memory_init {ontology, ns}`.
  - **Measured** (docs/specs/ontology.md, "Results"): the 23 held-out knowledge-update questions stay at 83% latest-first (91% with `hide_pending`), and all 22 true update pairs pass the policy; the life stream is unchanged (12 right / 0 wrong supersedes, 15/20 and 19/20 top-1); on decider-on LoCoMo (three conversations, end to end) automatic supersedes drop from 42 to 1 and pending decisions from 1,663 to 693, all removed replacements having been wrong, with QA unchanged (0.852 → 0.847 on 385 questions, 9 flips up and 11 down).
- **Automatic entities: GLiNER2.5 in kleos.** See "Automatic entities" in docs/specs/memory-api.md; research in docs/research/gliner.md and gliner-usage.md.
  - **Extractor.** `kleos-infer::Gliner` (feature `gliner`, in `infer`): a candle port of gliner2 2.1.0's `boundary` architecture, entities only. DeBERTa-v3 encoder (disentangled c2p/p2c attention with log-bucketed relative positions) at f32/f16/q8 weights on the CPU or Metal; boundary encoder, shared candidate pool, FiLM scorer and abstention on the CPU in f32; gliner2's `flat` overlap decoding; 384/64-word windows for long text. Parity with gliner2 stage by stage on CPU and Metal (hidden states within 2.7e-6, pair logits within 4.3e-5, identical spans); on the 200-claim gold set Rust and Python give the same spans at f32, f16 and q8.
  - **Registry.** `kleos::extract` (`EntityExtractor`, `register_extractor`, `enable_extractor`, `disable_extractor`), `::extractor list | enable | disable`, `Db::enable_extractor`, Python `enable_extractor` / `disable_extractor` / `extractors`. Default precision f16; The default checkpoint `Mazino0/kleos-gliner` (kleos's fine-tune, Apache-2.0) is downloaded on enable; `model: 'gliner-base'` selects upstream `fastino/gliner2.5-base-v1`; `$KLEOS_GLINER_MODEL` or `path` any other checkpoint.
  - **`remember`.** With `::memory init {extractor}` and no `entities` passed, entities come from the extractor, once per statement, assigned to claims by offset (`extract_granularity: 'claim'` per claim; `extract: false` off). Policy: training-matched prompt; speaker prefix stripped; per-label thresholds; capital rule only in mixed-case text; relatives as person nodes (`user's sister`) with kinship/role edges in `kleos_mem_link` (`decided_by: gliner-rule`), only first-person forms owned by the speaker; a name given with its relative merged into the node; first-person subject only from subject words or non-relative possessives (so "my sister moved" no longer shares the user's entity); pronoun claims inherit the statement's nearest person. Results report `extractor` and `edges`.
  - **Schema v7.** `kleos_mem_extract {ns, id, n => label, start, end, text, score, kept, eid, extractor, at}` keeps every raw span with its score and extractor version; older stores gain it on first use. Erase, branch copy-up and discard cover it.
  - Tests: `kleos-infer/tests/gliner_parity.rs` (word split, log buckets; stage parity and reduced precisions, ignored without the checkpoint), `kleos-core/tests/memory_extract.rs` (gating, merges, pronouns, storage and erase, options, ops; one real-model test ignored), unit tests in `gliner.rs` and `memory_extract.rs`. Reference script `kleos-infer/tests/fixtures/gliner_ref.py`; evaluation example `kleos-infer/examples/gliner_eval.rs`.
- **Durable workflows (`::flow`).** Temporal and pg_durable features built into kleos, on kleos relations and transactions, with no extra service. See docs/specs/durable-workflows.md.
  - **Model.** A workflow's progress is an append-only history (`kleos_flow_event`); after a crash, a sleep or a wait it is replayed: recorded commands return their recorded results, the first unrecorded one does the work. Workers claim runnable workflows (`kleos_flow_task`) by compare-and-set, holding a lease kept alive by heartbeats; every write a worker makes asserts its fencing token in the same transaction, so a worker that lost its lease records nothing.
  - **Exactly-once data effects.** A `query` or `remember` step commits its writes together with its checkpoint (`remember` through a new `remember_with` hook that adds the checkpoint to `remember`'s own commit). Python steps record their result once; their own side effects are at least once; `http` (feature `flow-http`, off by default) is at least once.
  - **Built-in mode (pg_durable style).** JSON step graphs (`::flow define`): `query`, `remember`, `recall`, `embed`, `decide`, `http`, `sleep`, `wait_signal`, `child`, `if`, `loop`, `for_each`, `fan_out` (branches on threads, joined), `set`, `return`; per-step retry policies with back-off timers that survive restarts, `on_error: 'continue'`. Run by `::flow tick` or an in-process worker thread (`DbInstance::flow_worker`, `CozoDbPy.flow_worker`, `kleos server --flow-worker <queue>`) that wakes on `kleos_flow_task` commits and polls for other processes' work.
  - **Code mode (Temporal style).** The worker ops (`claim`, `step`, `record`, `consume_signal`, `start_child`, `suspend`, `complete`, `fail`, `heartbeat`) and a Python SDK, `kleos_flow` (`kleos-lib-python/integrations/kleos-flow/`): `ctx.step` with `RetryPolicy` and timeouts, `ctx.query/remember/recall/embed/decide/http`, `ctx.sleep`, `ctx.wait_signal`, `ctx.child`, `ctx.patched`, `ctx.now/uuid`; a non-determinism check parks a workflow whose code changed under it; workflow versions.
  - **Also:** buffered, idempotent signals; child workflows; cancel (cascading to children); cron and `@every` schedules fired once across workers; lanes (`serial`) that run a key's workflows one at a time in order; `status`, `list`, `history`, `stats`. Surfaces: `::flow <op> {…}`, `Db::flow_call`, `POST /flow/:op`, `CozoDbPy.flow_call`.
  - **duroxide** (MIT, the runtime under pg_durable) was considered and not used: its storage provider cannot share a transaction with kleos data writes, it needs Tokio, and code mode must replay Python.
  - **Verified:**
    - a worker process killed mid-step resumes elsewhere with each step's data written once;
    - two worker processes on one SQLite file never run a step twice;
    - a Python agent workflow (remember → slow step → sleep → signal → recall) survives `kill -9`.
    - Results are in the spec, "Results".
- **Engram jobs run as durable workflows (Engram improved, not replaced).** The overlay, addressing, edit computation and erase are unchanged. Each job row in `kleos_engram_queue` is now also an `engram.apply` workflow started in the same transaction (id `engram:<ns>:<qid>`, lane per namespace in qid order), and `scripts/engram/engram_worker.py` is a code-mode worker: several worker processes, retries with back-off that survive restarts, heartbeated leases instead of one long fixed lease, a history per job, wake-up on new jobs. The job row still records status (`::engram queue`). `::flow adopt_engram` gives workflows to jobs queued by older versions; the worker calls it at start.
- **Memory branches: fork, rewind, merge, diff, discard.** See docs/specs/memory-branches.md.
  - **Fork** (`::memory fork {from, to, at?}`, `Db::memory_fork`, Python `memory_fork`, `POST /memory/fork`) is O(1): it writes one branch row and a version row, whatever the parent's size. A branch reads its parent's memories as of the fork time through its **lineage**, and nested forks work up to depth 16.
    - Reads route per memory to the **owner**, the nearest level with any row for it. Recall runs one fused search over the whole lineage, so BM25 normalisation stays comparable.
    - A write to an inherited memory **copies it up** first, in the same transaction: supersede, duplicate, retract, archive, set, rewind. The copy keeps its valid-time history, so `as_of` reads still see what was inherited. Ancestors are never written.
    - Entity resolution reads the whole lineage, so a branch's "Anna" is its parent's Anna and the same-subject gate holds. This adds `EntityConfig.lineage`.
  - **Rewind** (`{ns, to}`) retracts what is current but was not valid at `to`, and asserts again what was. It only appends rows, so it is undoable: `rewind {to: result.undo}`.
  - **Merge** (`{from, into?, dry_run?, keep_time?, forgets?}`) replays the branch's own statements into `into` (default: the parent) through `remember`, so decider-nano settles conflicts by meaning. A merged "I moved to Lisbon" supersedes the parent's "I live in Paris".
    - It also replays the branch's forgets of inherited memories.
    - A merge log (`kleos_mem_merge`) makes it idempotent and resumable.
    - `dry_run` is an exact preview: it merges into a scratch fork of `into`, diffs, then discards the fork.
  - **Diff** (`{a, b, at?}`) reports memories added, and removed with the reason (`superseded` and by what, or `forgotten`). It also lists pending decisions and entity names present in only one side. Memories are matched by id, then by normalised text.
  - **Discard** drops a branch's own rows, Engram queue and overlay; children need `cascade`. **branches** lists them.
  - **GDPR erase** now reaches every copy of a memory in its family: the root and all its branches.
  - **Engram is kept and extended.** A branch's overlay reads through its lineage (`::engram materialize` / `export` / `facts`), and its writes copy up the inherited facts and rows they change. A branch can serve its parent's overlay with its own edits on top, and a merge enqueues the target's edits through `remember`.
  - Schema version 6 (`kleos_mem_branch`, `kleos_mem_copied`, `kleos_mem_merge`); older stores gain the relations on first use. Plain namespaces run the same queries as before.
  - Tests: `kleos-core/tests/memory_branch.rs` (12) and `tests/engram.rs::branches_read_through_and_copy_up`. Demo with real models: `scripts/branches/branches_demo.py`. Costs: `kleos-core/examples/memory_branch_bench.rs` (results in the spec).
- **Memory API: write-time dates (event times).** See "Dates" in docs/specs/memory-api.md; results in docs/benchmarks/memory.md ("Dates and list text").
  - **Parser.** `memory_time::parse_times(text, at)` is a rule-based English temporal-expression parser (regex and chrono, no model). It covers absolute dates in the usual formats, relative days and parts of days, offsets ("two days ago"), weekdays and weekends, relative weeks, months, years and seasons, "since" / "until" / "for the past N weeks", and ranges. Each match resolves against when it was said to `[start, end)` with a granularity and a kind. It is conservative: "may" the modal, "march" the verb, ordinals, names, vague times and habits are not dates. On 50 hand-labelled LoCoMo and LongMemEval claims it scores expression precision 1.00 and recall 0.96, interval precision 0.96 and recall 0.92 (0.96 / 0.96 / 0.88 / 0.88 before the three fixes the hand check found).
  - **Storage.** A new relation `kleos_mem_event {ns, id, n => start, end, granularity, kind, expr}` (schema version 5) holds one row per expression of each added claim, written in `remember`'s transaction. A v4 store gains it on first use, filled from the stored texts. Erase removes the rows.
  - **Use.**
    - List hits carry `events`.
    - The context block shows dates inline: `(≈ 2023-05-07 (Sun), "yesterday")`, on each memory's own line only. (Notes on `said in` lines were dropped in wave3/ku-dip: they made the reader pick the older of two shown values.)
    - `recall {event_from, event_to}` filters on event time, falling back to the statement's events and then to valid-from.
    - `recall {event_weight}` (default 0.03, blend only) boosts memories whose event time overlaps a date in the question, resolved at `query_at` (default `as_of` or now).
  - **Results** (same reader and judge):
    - LoCoMo temporal context QA went from 0.769 to 0.822 on the full category (321 questions), and from 0.698 to 0.762 on the Mem0 head-to-head's 63. Mem0 top-10 scores 0.524 there and the whole conversation 0.603. All 301: 0.807 → 0.814.
    - LongMemEval (119 questions): temporal-reasoning rose 0.52 → 0.56, but overall fell 0.681 → 0.655, from knowledge-update answers. Fixed in wave3/ku-dip by dropping notes on `said in` lines: 0.672 and 0.681 in two samples (0.681 / 0.661 before dates), knowledge-update 0.844; LoCoMo 301 0.821, temporal 63 still 0.762. Re-asking with several seeds showed most of the original dip was reader noise (knowledge-update 0.821 before, 0.771 with dates, 0.821 with the fix, means of four samples).
    - Retrieval with the question-date boost: LoCoMo hybrid R@5 0.687 → 0.695, LongMemEval R@10 0.992 → 1.000.
    - Unchanged: the life stream, and the held-out knowledge-update set (83% latest first).
- **Memory API (changed default): a list hit of a claim shows its statement.** With claims and one hit per statement, a reader that showed only each hit's `text` saw one sentence of each message, and lost the speaker, the referents and the date words. That scored 0.671 against 0.708 for main's top-k list on 301 LoCoMo questions.
  - Now `recall`'s `text` for a claim of a longer statement is the statement: the whole of it up to `text_chars` (default 1000), else a window around the claim.
  - The claim is in the new `claim` field (Rust `RecallHit::memory_text()`), and `context` still holds the whole statement.
  - `hit_text: 'claim'` restores the old `text`.
  - The default list now scores 0.751 on LoCoMo (main's list 0.711, statements 0.751, context 0.814) and 0.672 on LongMemEval (0.647, 0.664, 0.655).
  - The bench harness's `kleos:` condition shows `text` as the API returns it, and its `base` config asks for claim text.
- **Memory API: diverse recall, entity expansion, a context output for LLM readers, speaker roles and valid-from filters.** See "Diversity", "Entity expansion", "Context output", "Speaker roles" and "Valid-from filters" in docs/specs/memory-api.md; results in docs/benchmarks/memory.md ("Recall diversity and context output").
  - **Diversity** (on by default; `diversity: false` is the old top k exactly): recall chooses k hits from a deeper window of the same index fetch (max(3k, k + 10)) with caps per statement (`max_per_statement`, default 1: one claim per message, the others only fill up to k) and per source group (`max_per_source`, default 0; the group is `source` up to its first `#`), and an MMR-style penalty on similarity to the hits already chosen (`mmr_weight`, default 0; cosine of the vectors, else word overlap). Deterministic.
  - **Entity expansion** (`expand: true`, default off): other current memories about the entities of the 3 most relevant hits (merges followed; entities with more than 25 memories skipped) join the pool at `expand_weight` (0.5) × the seed's relevance, flagged `expanded`.
  - **Context output:** `recall {format: 'context', budget}` (`Db::recall_context`, Python `recall(format="context")` / `recall_context`, `::memory recall`, HTTP) returns `{text, tokens, budget, ids, omitted}`: the hits grouped by entity (else conversation), groups in time order, each with its valid-from date, the values it superseded as `previously: "X" (from d1 until d2)` from the stored supersession links, unconfirmed replacements, and the statement a claim came from, within a token budget (characters / 4). Deterministic.
  - **Speaker roles:** `remember {role: 'user' | 'assistant' | 'other'}` (new relation `kleos_mem_role`, schema version 4; older stores gain it on first use). Assistant and other memories are recalled (`role` on hits, `recall {role}` filter) at `assistant_weight` (0.05) lower and are never decided: they never supersede, duplicate or become candidates of the user's statements. `forget` erase removes the role row.
  - **Valid-from filters:** `recall {from, to}` keeps memories whose validity starts in `[from, to)`.
  - **Results** (merged build, after the claims fix; docs/benchmarks/memory.md "After"): the statement cap raises LoCoMo hybrid multi-hop Rall@10 from 0.227 to 0.255 and LongMemEval hybrid multi-session Rall@10 from 0.917 to 0.958, with no R@5 or single-evidence loss. On the Mem0 head-to-head's 301 LoCoMo questions with the same reader and judge, the context block scores **0.807**, against 0.708 for main's list, Mem0 0.748 / 0.784 (top-10 / top-20) and 0.814 for the full conversation; temporal goes from 0.49 to 0.70. LongMemEval (119 questions): 0.681 against 0.647. A bare claim-text list loses with the statement cap (0.671 / 0.588), so readers should use the statement (`context`) or `format: 'context'`.
  - **Bench harness** (`scripts/bench-l2/`): `retrieval.py --configs` runs several recall option sets from one ingest (`<variant>+<config>`), `--context-budget` stores the context block, `--unit user+assistant` stores assistant replies with `role: 'assistant'`, LoCoMo sources carry the session (`session_N#D1:3`); `qa.py` conditions `ctx:<variant>` (the context block) and `stmt:<variant>` (the list with whole statements); `compare_configs.py` prints configs side by side. LoCoMo NDCG@10 now counts an evidence turn once (two claims of one turn were two relevant hits).
- **Memory API: claim-level `remember`.** A multi-sentence text (a chat message) is split into claims, one memory each, and every claim is decided on its own. See "Claims" in docs/specs/memory-api.md.
  - **Splitting** (`split: 'auto'`, the default; `'none'` keeps one memory per text) is a rule-based splitter (`memory_claims::split_claims`): abbreviations and initials, decimals, URLs and versions, quotes and brackets, ellipses, list bullets and hard-wrapped lines. A text that is one sentence is stored exactly as before.
  - **Ids and provenance:** claim ids are `<statement id>.<n>`; a one-claim statement keeps the plain id. Each claim's `kleos_mem_source` row holds the whole statement, recall hits of a claim carry it as `context`, and `forget` / `set` of a statement id cover all its claims. The result lists `claims` (`id, text, action, duplicate_of, superseded, decided`), and each decision names its `claim` and the vectors' `similarity`.
  - **Decisions per claim:** candidates are searched per claim, and the decider is batched over all claims (one call per template). Only declarative claims are decided, and only declarative memories are candidates: questions, requests to the assistant, greetings and fragments under three words are stored but never compared. A supersede ends only the old claim. Entity names are linked to the claims that mention them (to all claims when none does), and the same-subject gate is per claim.
  - **Review noise and latency** (claims of a split text): a candidate under cosine 0.6 (`min_similarity`) is not sent to the decider, and of each claim only its likeliest pending `supersedes` (p ≥ 0.9) and `duplicate` (p ≥ 0.5) are kept for review.
  - All claims of one `remember` are written in one transaction under the namespace version. With an Engram model attached, each added claim enqueues its own job, numbered with consecutive versions (the namespace version advances by the number of jobs).
  - **Recall** ranks questions and requests `question_weight` (default 0.05) below statements of the same relevance; hits of a claim carry `context`.
  - **Fix (LoCoMo):** sentence claims halved LoCoMo hybrid R@5 (0.661 → 0.423), because short first sentences of "Name: text" turns ("Caroline: Yes, I did.") were stored as claims whose vectors were mostly the speaker's name, and later claims lost the name. Now a sentence with fewer than two content words joins its neighbour, a speaker label is repeated on every claim, and a claim is declarative when any of its sentences is. LoCoMo hybrid R@5 0.678 and R@10 0.765 (before claims 0.661 / 0.749); the 23 held-out KU questions stay at 83% latest first with 1.83 pending per question; LongMemEval_S fts (500) and hybrid (119) are unchanged from main.
  - **Results** (LongMemEval knowledge-update, 23 held-out questions, native API on Metal): latest answer first 83%, stale first 17%, the same as the Python sentence prototype (whole messages: 39% / 61%), and 91% with `hide_pending`. Pending decisions per question went from 11.5 to 1.96, and `remember` p50 from 788 ms per sentence (about 8 s per whole message) to 144 ms per message. The life stream is unchanged: 15/20 and 19/20 with `hide_pending`, 12 right / 0 wrong auto-supersedes, 2 false pendings.
- **Metal works from the Python bindings.** Inside a host linked against an old macOS SDK, Metal compiled candle's kernels as pre-3.1 MSL, so every Metal model call failed with `unknown type name 'float16_t'`. Python interpreters such as miniconda's 3.12 (SDK 11.1) are one example. Benchmarks had fallen back to CPU, which is 30–40× slower.
  - candle-metal-kernels is now vendored with the MSL version pinned (3.2 on macOS 15+, 3.1 on macOS 14).
  - In that interpreter, decider-nano on Metal now takes 29 ms per decision, and its p differs from CPU by 3.7e-5.
  - Rust Metal parity is unchanged: 0/10 flips at f32, q8 and bf16.
- **Memory API: entity resolution in `remember` (E3) and memory scoring (E4, C6).** See docs/specs/memory-api.md.
  - `remember {entities}` now resolves names with `resolve_entities` over `kleos_mem_entity`, scoped to the namespace (new `EntityConfig.scope`). Auto links the existing entity (a new name form gets its own row, merged into it), Tentative makes a new entity with a review link, None a new entity. Merges stay non-transitive and splittable. Resolution is read while planning and written in the `remember` transaction, under the namespace version. The result gains `entity_links`.
  - **Same-subject gate:** when both statements have entities, `supersedes` acts only if they share one (sharing only a possible match caps it at pending). Otherwise the question is not asked, and the candidate is listed in `gated`. On the life stream, false pending supersedes went from 6 to 2, with auto-supersedes unchanged (12 right, 0 wrong).
  - **Ranking:** recall's score is relevance + 0.03 · recency (half-life 90 days) + 0.1 · (importance − 0.5) + 0 · access counts, with every weight a recall option; `blend: false` gives the old relevance order. The weights are small because fused relevance near the top is compressed: on the life stream read at its own "now", 0.1 / 30 days lost 3 of 20 answers, the defaults none. Hits carry `relevance`, `recency`, `access_count`, `last_access`, `expires` and `archived`.
  - **Access counts** are buffered per `Db` and flushed in batches (64 memories or 1 s) to `kleos_mem_access`, in transactions that never touch the namespace version. `Db::memory_flush` / `::memory flush` flushes now.
  - **TTL and archive tier:** `remember {ttl | expires}`, `forget {mode: 'archive'}`, `forget {expired: true}`, and the new `::memory set` (`Db::memory_set`) for importance, expiry and archive state. Recall leaves expired and archived memories out unless `include_expired` / `include_archived`.
  - The schema version is now 3: `kleos_mem_access` and `kleos_mem_life`, created on first use for older stores. Erase also removes access counts, expiry rows, and entity merges and links.
  - The `memory::link_entities` hook is gone: resolution happens inside `remember`.

- **Memory API → Engram overlay: `remember` produces Engram edits when a model is attached** (feature `engram`, docs/specs/engram-overlay.md "Memory attachment").
  - `::memory init {engram: 'nano'}` (Rust/Python `InitOpts.engram`) attaches a registered Engram model; `''` detaches. Unattached stores and builds without the feature behave exactly as before.
  - While attached, `remember` (write, or supersede with the old ids), `forget` retract and `forget` erase each write a job to the new `kleos_engram_queue` relation in the same transaction as the memory change. The job id is the namespace version the commit produces, so jobs are in commit order, and the `kleos_mem_ns` concurrency guarantees cover them. Duplicates enqueue nothing; an erase clears the text of the erased memories' earlier jobs.
  - `scripts/engram/engram_worker.py` applies the jobs with the EngramEdit-style write path through `::engram write` / `erase_plan` / `erase_fact`, with leased claims, retries and idempotent applies. Overlay facts are keyed by memory id and valid from the memory's valid-from, so `::engram export {as_of}` follows `::memory recall {as_of}`. `::engram queue` reports pending / failed jobs and the last error.
  - Fix: `::engram erase_fact` no longer brings back the row value of a fact that ended while the erased fact shared its rows.
  - Python bindings: new `engram` feature pass-through.
  - End to end on the 132M model (`scripts/engram/remember_e2e.py`): top-1 recall of 10 remembered facts 0/10 → 9/10 (unseen rephrasings 1/18 → 9/18); a superseding `remember` flips the answer and `as_of` before it restores the old overlay bit for bit; erasing one memory keeps 128 untouched rows bit-identical; erasing the namespace gives logits bit-identical to the base model; jobs survive a worker crash and a `kill -9` with one version per fact. `remember` costs about 1 ms more on SQLite when attached.

- **Memory benchmark harness (UPGRADE_PLAN L2):** `scripts/bench-l2/` runs LongMemEval_S and LoCoMo through the public `remember` / `recall` API (Python bindings) and reports LLM-free retrieval (recall@5/10, NDCG@10 per question type, a knowledge-update supersession view), ingestion and recall latency, and secondary QA accuracy with a pluggable reader/judge. Resumable, seeded, pinned dataset revisions; `scripts/bench-l2/run.sh` re-runs everything against the current build. First numbers in [`docs/benchmarks/memory.md`](docs/benchmarks/memory.md).
- **Memory API: concurrent `remember`s on one topic no longer leave two current facts.**
  - A per-namespace version row (`kleos_mem_ns`) is asserted and bumped inside each write transaction, so the storage engine orders writers across threads and processes. A writer that lost the race plans again and supersedes the winner.
  - The schema version is now 2, and v1 stores get the table on first use.
  - SQLite connections now set a 5 s busy timeout, so several processes can share a database file without immediate `database is locked` errors.

- Agent memory API (UPGRADE_PLAN H1 + E10, docs/specs/memory-api.md):
  `remember` / `recall` / `forget` / `reflect` / `link` on a ready-made,
  namespaced schema (`kleos_mem` with valid time, vectors, entities, mentions,
  typed links, statement sources and a decision audit table), so an agent never
  writes CozoScript.
  - `remember` finds the top-k similar current memories (vector plus FTS),
    batches `decide(supersedes)` and `decide(duplicate)` over them and acts on the
    band. `auto` supersedes by ending the old memory's validity (never deleting)
    or skips a duplicate and records the statement as another source.
    `tentative` (and a likely supersede below the tuned threshold) inserts and
    records a pending decision. Every result carries `decided_by`.
  - Without models it degrades to add-only plus FTS, keeping two model-free
    rules: exact-text duplicates and the value rules. It never errors because
    ML is off.
  - `recall` is hybrid search with `WeightedFusion`, over currently valid
    memories by default, with `as_of` time travel, namespace, kind and entity
    filters, and `hide_pending`. `forget` retracts (default) or erases (GDPR,
    via `::erase`). `reflect` returns pending decisions, `::changes` and
    contradiction candidates, without a model. `link` adds a typed edge.
  - Entity linking goes through one hook, `memory::link_entities`: exact
    normalised names for now, E3 later.
  - Surfaces: Rust (`Db`/`DbInstance::remember`, `recall`, `forget`,
    `reflect`, `link`, `memory_init`, `memory_call`); CozoScript
    (`::memory init | remember | recall | forget | reflect | link`); Python
    (`db.remember(text, ns=…)` etc.); HTTP (`POST /memory/:op`).
  - On a scripted 40-turn life stream with decider-nano (nano-tune-v1, q8) and
    EmbeddingGemma int8 on Metal: 12 of 15 gold supersessions were auto-applied
    with 0 wrong. The 3 misses scored p 0.977–0.996, below τ 0.997, and were
    kept as pending. 4 of 6 restatements were deduplicated with 0 wrong.
    Recall returned the current fact first for 15 of 20 questions (18 of 20
    with `hide_pending`). p50 latency, on a loaded machine: remember 355–655 ms
    (5 candidates × 3 forward passes), recall 11–38 ms.

- Entity resolution (UPGRADE_PLAN E3, docs/specs/entity-resolution.md,
  `kleos::entity`). Mentions are resolved against a user's entity relation
  configured by column names (name, type, aliases, description, embedding).
  - Candidates: a rule tier (exact normalized name or alias, initials,
    nicknames, acronyms, first name, surname; honorifics and company suffixes
    stripped) unioned with the relation's MinHash LSH, FTS and HNSW indices;
    same or compatible types only.
  - Confirmation: `same_entity` decisions in one batch. Auto merges, Tentative
    records a possible same-as link for review, None makes a new entity. A
    mention is only ever decided against one canonical entity, so merges are
    never transitive. Without a decider, rules and MinHash decide alone.
  - Merges are rows in the valid-time relation `kleos_entity_merge`; `split`
    appends a retraction, so history and time travel keep working. Review links
    live in `kleos_entity_link`.
  - Rust: `resolve_entities(config, mentions, opts)`, `merge_entities`,
    `split_entity`, `entity_history`, `entity_pending`, `reject_entity_link`,
    `canonical_entity`. CozoScript: `::entity resolve | merge | split | history |
    pending | reject`. Python: the same methods on `CozoDbPy`.
  - `examples/entity_eval.rs` measures DBLP-ACM (Leipzig, CC BY 4.0) and a
    hand-written memory-style mention set against an exact-name baseline.

- Engram overlay store, first milestone (docs/specs/engram-overlay.md).
  - New crate `kleos-engram`: DeepSeek Engram addressing. It loads the lookup
    spec from a model's `config.json` + `tokenizer.json` (V4.1-Flash, nanogpt
    and demo layouts) and computes `(layer, row)` ids for text. Its addressing
    fingerprint covers the token map, multipliers, primes, offsets, layers,
    orders, heads and row dimension. It also defines the per-layer
    safetensors overlay format. Parity is bit-exact against
    DeepSeek-V4.1-Flash `engram.py`, `engram_demo_v1.py` and the nanogpt hash.
    It ships with the `kleos-engram` CLI.
  - kleos-core feature `engram`, off by default. It adds `::engram models |
    register | write | supersede | materialize | export | erase | erase_plan |
    erase_fact | facts | history | rows` and the matching `Db::engram_*` Rust
    API.
    - Row edits are stored as versioned data:
      `(fingerprint, ns, layer, row, vt) → bf16 vec + facts`.
    - Per-fact records hold the text, phrasings, rows touched, decided_by,
      source and a replay payload.
    - Valid-time supersession and `as_of` materialisation.
    - org → user → agent layering.
    - Exact namespace erase.
    - Single-fact erase by provenance replay: only the rows the fact touched
      are rewritten, and kleos refuses anything else.
    - Every read and write that names a fingerprint refuses a mismatch.
  - `scripts/engram/engramedit.py`: an EngramEdit-style write path (method
    from EngramEdit, MIT) and the end-to-end evaluation on the 132M
    engram-nanogpt model, including the comparison with the spike's naive
    per-trigger training and with kleos FTS retrieval.

- `decide()`: small decision models attached to the database (UPGRADE_PLAN D5,
  docs/specs/decide.md).
  - Fixed, versioned question templates per decision and per backend: `supersedes`,
    `same_entity`, `duplicate`, `contradicts`, `relation` (update / contradiction /
    compatible / duplicate) and `important`. Callers never write the prompt.
  - Symmetric templates score both orders and average.
  - Calibration: per-option log-bias, a label threshold and a risk-controlled
    auto threshold. Each decision returns `{label, p, band: auto|tentative|none,
    confident}`, with defaults fitted on a held-out adapt set.
  - `::decider fit <name> <template> <relation>` fits an L2 logistic probe on the
    backend's decision features. It is stored in `kleos_decider_fits`, reloaded on
    open and used automatically.
  - CozoScript: `::decider list | enable | disable | fit | fitted | forget`;
    `decide_p/label/band(model, template, a, b)`, `decide_supersedes`,
    `decide_same_entity`, `decide_duplicate`, `decide_contradicts`,
    `decide_important`, `decide_relation`; the batched fixed rule
    `Decide(rel[id, a, b], model:, template:)`, which outputs
    `[id, label, p, band, decided_by]`.
  - Built-in backend: decider-nano (150M), a candle port in `kleos-infer` that
    matches the PyTorch reference to about 1e-5. The default weights are
    `Mazino0/kleos-decider-nano` (MIT), kleos's fine-tune on memory-shaped pairs
    (VitaminC 0.68→0.81, LongMemEval knowledge-update 0.39→0.82); the upstream
    `tasksource/tasksource-decider-nano` (Apache-2.0) is `decider-nano-base`.
    Default precision q8 (Metal ≈295 MB, CPU ≈305 MB after a batch of 16).
    The `infer-accelerate` feature adds Apple BLAS for CPU.
  - Third backends can register their own template family (`register_family`).
  - Rust (`Db`/`DbInstance::enable_decider`, `decide`, `decide_batch`,
    `fit_decider`) and Python (`db.enable_decider`, `db.decide`, `db.fit_decider`)
    APIs.
  - `examples/decide_eval.rs` evaluation harness.

- `examples/vecbench.rs`: a vector-search benchmark harness. It loads an `.npy`
  matrix, builds an HNSW index, runs held-out queries over an `ef` sweep and a
  filter-selectivity sweep, and reports build time, recall@k against cached exact
  ground truth, QPS, p50/p99 latency, CPU time per query, under-filled queries and
  on-disk size as a markdown table.

- Supersession for agent memory (UPGRADE_PLAN E1). `::supersede create fact {subject,
  predicate}` declares that a relation's leading key columns admit one current row at
  any valid time. Every `:put`/`:insert` then ends the previous current row at the new
  row's valid time by appending a retraction in the same transaction. Nothing is
  deleted, so `@`, `@ (tt: …)`, `:as_of` and `::history` still answer for earlier
  times. Late-arriving older facts fill their gap without ending newer ones,
  re-asserting the current fact writes nothing, and on bitemporal relations a new
  value at the same valid time is recorded as a correction. `::changes fact {from,
  to, as_of, prefix}` lists what was `added`, `ended` or `changed` between two valid
  times. `import_relations` and columnar import reject relations with a supersession
  key, since they would bypass it. Also `::supersede list` and `::supersede drop`.
- Provenance with cascading retraction and GDPR erasure (UPGRADE_PLAN E2).
  `::provenance create fact_src {derived: fact, source: episode}` declares a link
  relation keyed by the derived key followed by the source key. `::retract episode
  [['ep2']]` (optionally `@ T`) ends a source's validity and, transitively, that of
  every derived row left without a live source; rows with another live source stay.
  `::erase episode [['ep7']]` physically deletes the source, its links and every
  derived row left with no source, including their HNSW, FTS and LSH entries, and
  writes salted audit markers to `kleos_evict_audit`. `::lineage fact [[…]]`
  explains a row back to its sources. Spec:
  `docs/specs/memory-supersession-provenance.md`; tests:
  `kleos-core/tests/memory_supersession.rs`.
- Weighted (convex) score fusion for hybrid search, beside RRF (Bruch et al.,
  arXiv 2210.11934). New fixed rule `WeightedFusion(rel[list_id, item, score],
  weights: [['semantic', 0.6], ['text', 0.4]], norm: 'minmax', norms: [...])`
  normalizes each list and sums `weight × normalized score`. Normalizations:
  `'minmax'`, `'zscore'`, `'none'`, and theory-based ones for known ranges:
  `'neg_cosine_distance'` (the vector leg's `-distance`), `'cosine_similarity'`,
  `'bm25'` (`s / max`), `['range', lo, hi]`, `['tmm', lo]`. `detailed: true` gives
  the same 5 columns as RRF. In `HybridSearch`, set
  `fusion: Fusion::Weighted(WeightedFusionParams::new([("semantic", 0.6), ("text", 0.4)]))`;
  RRF stays the default and its generated script is unchanged. Weights naming an
  unconfigured leg, or missing a configured one, are rejected.
- Pluggable re-rank stage. Implement `Reranker` (`fn rerank(&self, query: &str,
  docs: &[&str]) -> Result<Vec<f32>>`), register it with
  `Db::register_reranker(name, Arc<dyn Reranker>)` /
  `DbInstance::register_reranker`, and call it from CozoScript as
  `Rerank(rel[item, text, prior_score], model: 'name', query: $q, top_n: 50)` or
  from `HybridSearch` with `rerank: Some(RerankParams::new("name"))` (`top_n`,
  `text_field`, `query`, `batch_size`). Re-ranking runs after fusion and before
  MMR. kleos ships no model. Python: `db.register_reranker(name, fn)` plus
  `fusion` / `rerank` keys in `hybrid_search`.
- `examples/hybrid_eval.rs`: a BEIR SciFact harness (300 test queries,
  bge-small-en-v1.5 vectors, kleos BM25, ms-marco-MiniLM-L-6-v2 cross-encoder
  scores served through the `Reranker` hook). nDCG@10 / Recall@100: BM25
  0.688 / 0.923, vector 0.713 / 0.945, RRF 0.718 / 0.968, weighted min-max
  (α_vec = 0.6, tuned on 200 train queries) **0.734 / 0.975**. This MiniLM
  re-ranker scores about 0.69 on top of every first stage, so it helps BM25
  only. That matches its published SciFact score; a re-ranker is only worth
  using if it beats the first stage on your data.
- Live query subscriptions (UPGRADE_PLAN F1). `db.subscribe("?[k, v] := *kv{k, v},
  k >= $min", params, SubscribeOptions::default())` returns a `Subscription` that
  yields `Initial { rows }`, then `Delta { added, removed }` whenever a committed
  transaction changes the result. The query is re-evaluated only when a commit writes a
  relation it read: dependencies are recorded while it runs, so rules, fixed-rule
  inputs, HNSW/FTS/LSH searches (their base relation) and stored queries all count.
  Identical `(script, params)` subscriptions share one evaluation. Nothing is delivered
  for aborted, failed or rolled-back transactions. Re-evaluation runs on one background
  worker per `Db`, never on the writer's thread; commits that land while it is busy are
  coalesced, and a subscriber whose bounded queue (`queue_capacity`, default 64) fills
  up gets one `Resync { rows }` instead of the deltas it missed. Only single read-only
  queries can be subscribed. `'NOW'` is read at each re-evaluation, so a result that
  changes only because time passes is not pushed until some write triggers it.
  Re-evaluation is a full query run, not incremental (that is F2): on `mem`, 1000
  point-lookup subscriptions on other relations left single-writer throughput unchanged
  (about 31k txns/s), while subscriptions on the written relation halved it (about 14k)
  and delivery took 18 ms p50 at 1000 of them (0.4 ms at 10). Also `Db::unsubscribe(id)`,
  `Db::live_stats()`, `DbInstance` dispatch, Python `db.subscribe(query, params,
  callback)`, and server-sent events at `GET /subscribe?script=…&params=…` in
  `kleos-bin`. Spec: `docs/specs/live-subscriptions.md`; tests:
  `kleos-core/tests/live_subscriptions.rs`; benchmark: `examples/livebench.rs`.

- Text embeddings inside the database (UPGRADE_PLAN D3). `embed('model', text)`
  returns an F32 vector from an embedder registered with
  `kleos::embed::register_embedder(name, Arc<dyn Embedder>)` (trait: `embed(&[&str])`,
  `dim()`); `embed('model', text, 256)` takes a re-normalised Matryoshka prefix.
  Every call goes through one embedding cache keyed on (model id, SHA-256 of the
  text), bounded by a byte budget (default 64 MiB; `set_embedding_cache_capacity`,
  `embedding_cache_stats`). The registry and cache are process-wide.
- Computed columns: a column default may now read other columns of the row, so
  `vec: <F32; 768> default embed('gemma', text)` keeps a vector column in sync with
  its text. A `:put` that omits `vec` computes it; an `:update` that changes `text`
  recomputes it; an `:update` that supplies only some of a computed column's sources
  is rejected rather than leaving a stale value. The `embed()` calls of one write
  are batched (256 rows per model call group) instead of one model call per row:
  100 rows with a batch size of 16 make 7 model calls.
- `kleos-infer`, a new workspace crate for single-forward-pass models (UPGRADE_PLAN
  D2), built on candle: CPU everywhere, Apple GPU with the `metal` feature; f32,
  f16, bf16, int8 (Q8_0) and int4 (Q4_0) weights; models load lazily from the local
  Hugging Face cache and are never bundled or downloaded by the library. Ships
  EmbeddingGemma-300m (output matches sentence-transformers f32 to cosine 1.0000 on
  300 SciFact queries) and Model2Vec static embedders (potion-base-8M/32M). It is
  opt-in: not a default workspace member, and kleos-core only uses it behind the new
  off-by-default `infer` / `infer-metal` features (`kleos::embed::builtin`). candle 0.11
  does not build on Rust 1.90 (unstable `stdarch_neon_f16`); it is verified on 1.99.
  Measured on SciFact (M3, Metal, shared with other jobs): EmbeddingGemma int8
  scores nDCG@10 0.785 on the full corpus through kleos (auto-embed on `:put`, HNSW,
  `embed()` queries), the same as exact search; on a 1,500-document subset f32,
  f16, bf16 and int8 all score 0.871-0.873 and int4 0.866, and the 256-d Matryoshka
  prefix loses 0.006. int8 is the default: lowest load peak (1.1 GB vs 2.8 GB for
  f32) at equal quality. potion-base-8M is the near-free fallback (0.506, about
  4,000 documents/s on the CPU, 100 MB).
- Python: `kleos.register_embedder(name, fn, dim, max_batch=32, model_id=None)`,
  `kleos.unregister_embedder(name)` and `kleos.registered_embedders()`. `fn(texts)`
  returns one `dim`-wide vector per text; the embedder then works in `embed()` and
  computed columns like a Rust one. The registry is process-wide, so these are module
  functions, not `CozoDbPy` methods. Tests: `kleos-lib-python/tests/test_wave1.py`.
- Runtime on/off switch for embedders. `::embedder list` shows every enabled or
  disabled embedder (name, kind gemma/model2vec/custom, model, dim, enabled, loaded,
  precision, device). `::embedder enable gemma {model: 'embeddinggemma', precision:
  'int8', device: 'auto', dim: 768}` (every option optional; the defaults are
  EmbeddingGemma int8 on the auto device, or `{model: 'potion-base-8M'}`) builds a
  built-in model, loaded lazily, plus gemma's `gemma_query` twin. `::embedder disable
  gemma` unregisters both, unloads the weights and drops their cached vectors.
  `::embedder enable gemma` with no options brings back the previous configuration,
  including Rust- or Python-registered embedders. Without the `infer` feature,
  enabling a built-in model fails with an error that names the feature. While an
  embedder is off, `embed()` errors with "embedder 'gemma' is disabled; turn it back
  on with `::embedder enable gemma`", and strict computed columns reject the write.
  The new `embed_or_null()` returns null instead, so a nullable column `vec: <F32;
  768>? default embed_or_null('gemma', text)` keeps accepting writes while the model
  is off. `::embedder backfill docs` then fills those nulls in batches through the
  cache with an ordinary `:update` that maintains the HNSW index, and refuses to run
  while the model is still off. The registry is process-wide, so the switch affects
  every open database. The same switch is available in Rust
  (`enable_embedder(name, EmbedderSpec)`, `disable_embedder`, `embedders()`,
  `backfill_embeddings`), in Python (`db.enable_embedder(name, model=..., precision=...,
  device=..., dim=...)`, `db.disable_embedder`, `db.embedders()`,
  `db.backfill_embeddings`, plus module-level `kleos.enable_embedder` /
  `kleos.disable_embedder`) and in kleos-bin (`--embedder gemma=embeddinggemma:int8`,
  repeatable; new `infer` / `infer-metal` features on kleos-bin and kleos-py). The
  `Embedder` trait gains the default methods `is_loaded`, `device` and `unload`.
  Disabling now actually frees the model. `kleos-infer` runs its Metal work and
  `unload()` inside an Objective-C autorelease pool. Without the pool, debug
  builds never released the buffers (+315 MB per enable/disable cycle). Measured
  with gemma int8 on Metal through `::embedder enable/disable` (process
  footprint): 7.5 MB before, 439 MB after enable and the first embed, 15 MB after
  disable, and the same again on the next cycle, when run with
  `MallocLargeCache=0`. With the default macOS allocator, freed load-time buffers
  stay in libmalloc's large cache (836 MB loaded, 478 MB after disable). The
  kernel reclaims that memory under pressure. Spec: `docs/specs/embedder-toggle.md`; tests:
  `kleos-core/tests/embed_toggle.rs`.
- `kleos-infer`'s `decider_bench` reports `weights_cached_pct`: how much of the
  checkpoint was in the OS page cache before loading. The ~9 s CPU q8 load seen
  in benchmarks was a cold read of the 298 MB bf16 checkpoint from a disk giving
  21-34 MB/s (USB 2.0 link, shared). Every precision pays it; macOS had evicted
  the file within 90 s of idle under memory pressure. With the file cached, CPU
  q8 loads in 0.3-0.5 s (read() ≈25 ms) and EmbeddingGemma q8 in 0.8-1.9 s (cold:
  34 s for its 1.2 GB f32 checkpoint).

### Fixed
- **Dates:** a month and day without a year in a past-tense clause resolves to the latest one before the anchor ("attended … on July 17th", said in February); tense is judged per clause, so "training for a triathlon on June 24th and I've recently completed a 5K" keeps June ahead.
- **Small talk:** courtesies and sign-offs ("No problem, John.", "You're welcome!", "Sounds good") are small talk; fillers ("Okay, …") are skipped before checking.
- **Memory branches:** discarding a branch cancels its unfinished `engram.apply` workflows.
- **`engram_worker.py --once`** waits out retry back-offs before exiting (`--no-wait`, `--max-wait`).
- **`infer-metal`** now includes Apple Accelerate for the CPU path (GLiNER on CPU 2.2–2.6× faster; parity tests unchanged).

- **`true` followed by whitespace or a comment parsed as `false`.** The grammar's `boolean` and `null` rules were not atomic, so the word-boundary lookahead added for upstream cozo #281 let pest fold the trailing whitespace or comment into the literal's span, and the parser (`as_str() == "true"`) read `"true "` as `false`. Any `true` that was not directly followed by `,`, `)`, `]` or the end of the script was affected: every rule of a multi-rule program but the last, and every rule followed by `:put`, `:update` or another option (`?[k, c] := k = 1, c = true :update t {k => c}` wrote `false`, which looked like nothing written). Both rules are now atomic (`kleos-core/src/cozoscript.pest`). Tests: `kleos-core/tests/literal_binding.rs`. kleos's own scripts did not hit it: the memory, flow and branch code passes such values as parameters or constant rows, or writes `false`, which parsed correctly. Those workarounds stay; they are correct either way.
- **A stratum with a single rule set is evaluated inline, not through rayon.** A rayon call from a non-worker thread hands the job to the pool and waits for a worker even when there is nothing to parallelise. That wait can exceed the evaluation of a small rule on a loaded machine, and scripts chain several single-rule strata. On a 23-column write after three derived rules (the flow agent's old Engram workflow script) this saved about a quarter of the time under heavy load. `remember` was unchanged within noise. Measured, not assumed: the "many `col = const` atoms" planner cost reported with durable workflows is small (0.2 ms at 24 columns, against 0.07 ms for a constant row; `kleos-core/examples/planner_consts.rs`), and the memory API's scripts spend about 0.07 ms per block planning (`kleos-core/examples/remember_bench.rs`). The 16–28 ms seen then was evaluation latency on a machine at load 40–140, not planning, so the memory scripts were not rewritten.

- Python: `register_callback` no longer hangs queries. Each callback's delivery loop
  ran on a rayon worker and kept it for the callback's lifetime, so once a process had
  as many callbacks as CPU cores, every query that uses rayon (`PageRank` and the other
  graph algorithms) waited forever for a worker. Each callback now gets its own thread.
- Python: `close()` now ends the handle's live subscriptions. Before, they kept running
  after `close()`, holding their callbacks, a delivery thread, the engine's live worker
  and the database itself, and `unsubscribe` on a closed handle could not stop them.

- HNSW search with a `filter` now returns `k` rows whenever at least `k` indexed rows
  match. The filter used to run after the bottom layer had collected `ef` candidates,
  so a selective filter silently returned fewer rows: on 99,000 real 1536-d
  embeddings with `k: 10, ef: 64`, a filter keeping 10% of rows under-filled 899 of
  1,000 queries and a filter keeping 1% under-filled all of them (recall@10 0.06).
  The filter (and `radius`) is now applied inside the search. Rows failing it still
  guide the search but never enter the results, and the search keeps expanding
  until it holds `max(ef, k)` passing rows or runs out of reachable graph. When few
  rows near the query pass, failing neighbours are expanded on the spot (ACORN-1
  two-hop expansion). When the remaining graph work would cost more than a full
  scan, the search switches to an exact scan of the rows that pass the filter and
  are in the index. Unfiltered searches take the old code path unchanged. On a
  quantized index (`quantize: ...`) a filtered search descends the upper layers on
  codes but runs the bottom layer on exact vectors, since the predicate needs the
  stored row anyway; returned distances stay exact.
  Regression test: `tests/hnsw_filtered_search.rs`.

- Filtered HNSW search no longer pays for both the graph search and the exact scan.
  The switch to the scan compared the larger of the remaining and the already
  spent graph visits with the cost of a scan, so a search that had nearly finished
  dropped its results and scanned anyway. On 99,000 1536-d embeddings at 10%
  selectivity, `ef: 128` cost 68 ms of CPU per query against 52 ms at `ef: 256`.
  Only the remaining work counts now, and it is never estimated below the visits
  needed to fill the result slots still empty. The visit and scan costs were also
  re-measured by forcing each path over 1–50% selectivity and `ef` 32–256 (a graph
  visit costs about 10.5 scanned rows, and a scanned row that passes the filter
  1.7 more). On that index the scan takes over at 10% from `ef: 128` and at 2% and
  below for every `ef`.

- `kleos::RocksDbTx` is exported with the `storage-rocksdb` feature, like `SqliteTx`
  and `MemTx`. It was public but could not be named (an `unnameable_types` warning on
  every rocksdb build).

- sqlite: a write could fail with `database is locked (code 5)` while another thread
  was reading. A read that found a row left its cached statement holding a SHARED lock
  on the file, and the statements were finalized only after the in-process lock was
  released. `SqliteTx` now finalizes them first. Found by the live-subscription
  benchmark, where the worker reads right after every commit.

- `int_range` and `rand_vec` can no longer exhaust memory (found by fuzzing). A zero
  step (`int_range(30, 3, 0)`) looped forever, allocating until the process was killed;
  a step that overflowed `i64` wrapped in release builds and did the same; and very
  large ranges or `rand_vec(-1)` aborted on allocation. These results are built
  before the query memory budget can see them. `int_range` now rejects a zero step,
  computes its length in overflow-safe arithmetic, and both functions refuse to
  generate more than 2^24 (16,777,216) elements per call. `int_range(0, 10, 0)`, which
  silently returned `[]`, is now an error too.

- Deleting vectors no longer makes surviving vectors unreachable in HNSW search.
  Removal only unlinked the node (upstream CozoDB noted it "still has some probability
  of disconnecting the graph"), so neighbours that were connected through it were cut
  off and search silently stopped returning them. After deleting 60% of a 200-vector
  index, 22 to 27 of the 76 survivors could no longer be found even by querying their
  own vector; deleting the hub of a three-node index hid a survivor about 2 times in 3.
  Removal now reconnects the orphaned neighbours on every layer, re-running neighbour
  selection over their remaining links plus each other (as hnswlib's repair-on-delete
  does). This applies to `:rm`, `:put` replacing a vector, and bulk deletes. Indexes
  already damaged by earlier deletions are repaired by `::reindex`. Regression test:
  `tests/hnsw_delete_connectivity.rs`.
- Parsing no longer takes exponential time in nesting depth. Every comma-separated
  grammar rule had the form `(x ~ ",")* ~ x?`, which parses the last element twice, so
  each level of nested lists, objects or function calls doubled the work: a valid
  list nested 18 deep took 10 s, and a 200-byte script of unclosed `[` (found by the
  new fuzzer) hung before any query budget applied. The 18 rules now read
  `(x ~ ("," ~ x)* ~ ","?)?`, which accepts the same language (trailing commas
  included) and produces the same parse tree. Depth 40 parses in milliseconds.
  Regression test: `tests/parser_nesting.rs`.
- Block comments no longer take exponential time to parse. When a nested `/*` failed
  to close, the comment rule fell back to reading it as text and rescanned the rest
  of the input (pest does not memoize), doubling the work per `/*`: 18 unclosed `/*`
  took 2.7 s, and a run of `/*/` blew up even when the comment could close (64 of
  them did not finish in 10 minutes; the fuzzer found 1.5 KB scripts that took
  seconds). Block comments now nest exactly like Rust's: every `/*` opens a level and
  every `*/` closes one, and parsing is linear. The new rule accepts nothing the old
  one rejected; the only comments it gives up are ones that a `/*/` closed because the
  `/*` it starts never closed (`/* a /*/`), which are now a parse error instead of a
  silent reinterpretation. A differential test checks both properties on all 87,381
  strings over `/ * a ␣` up to length 8.
- Integer arithmetic no longer wraps or panics on overflow (found by fuzzing).
  `+`, `-`, `*`, unary minus and `abs` on `Int` used plain operators, which panic in
  debug builds and silently wrap in release builds (`9223372036854775807 + 1` returned
  a large negative number). They now fail with "integer overflow". `i64::MIN % -1`
  panicked even in release builds and now returns `0`.
- `mul` with three or more vector arguments now multiplies. The running product
  was folded with vector addition, so `mul(a, b, c)` computed `(a + b) * c`.

- Bulk loads no longer leave search-index entries for rows they delete or replace
  (upstream CozoDB #232). `import_relations`, columnar import and
  `import_from_backup` skipped HNSW/FTS/LSH maintenance entirely. A row deleted
  through `import_relations({"-rel": ..})` left a dangling HNSW node, and every later
  vector search on that index failed with "Cannot find compound key for HNSW", as did
  writes that traversed it. Replaced rows kept FTS/LSH postings for text they no
  longer contain. Bulk loads now remove the old row's entries, as `:rm` does. New row
  versions are still left unindexed (the fast path, which warns and points at
  `::reindex`), so a search index may miss imported rows but never points at a row
  that is gone or has changed. Regression tests: `tests/import_search_index_integrity.rs`.
  `::reindex` still repairs databases damaged before this fix. Its ghost-postings test
  now builds that damage through the new `test-hooks` switch
  `legacy_next_import_for_tests`, and CI runs it in the test-hooks job.
- Floats now display as float literals (upstream CozoDB #236). `Display` for numbers
  renders CozoScript, but `1.0` printed as `1` and `1e300` as a 301-digit integer, both
  of which parse back as Int. Finite floats now use Rust's shortest round-trip form
  (`1.0`, `1e300`, `1.5e-7`); a unit test parses every case back.
- `first`, `last`, `get` and `maybe_get` accept a `Validity` value and read it as the
  `[timestamp, is_assert]` pair it is written as, instead of failing with
  "'first' requires lists" (upstream CozoDB #246).
- Every type reachable through the public API can now be named (upstream CozoDB #288).
  `kleos::parse` exports `SysOp` and the `::hnsw`/`::fts`/`::lsh` config types. The
  crate root exports `Aggregation`, `Op`, `VecElementType`, `TokenizerConfig`,
  `AccessLevel`, `MemTx` and `SqliteTx`. A crate-level `unnameable_types` lint keeps
  the list complete under CI's `-D warnings`.
- The clippy CI gate passes again on Rust 1.99. miette's `bail!` expands with a trailing
  semicolon, which rustc's `semicolon_in_expressions_from_non_local_macros` future-compat
  lint rejects in expression position (289 sites). The core crate now uses an
  expression-safe crate-local `bail!` with the same arguments and errors.
- Broken gitee mirror links and a stale `cozo-lib-*` reference left by the rename.

### Added

- Half-width vector column types `<F16; N>` and `<BF16; N>` (also inside lists,
  e.g. `[<F16; 1536>]`). Values are rounded to the declared precision on write,
  stored 2 bytes per element (half the size of `<F32; N>`; new storage tags 2/3 in
  the vector encoding, so existing F32/F64 rows are byte-identical and still
  readable), and read back as F32 vectors, so every vector function, `vec()`,
  `rand_vec()` (`'F16'`/`'BF16'` type argument), JSON output and HNSW work on them
  unchanged. HNSW indexes over such columns use `dtype: F16` / `dtype: BF16`. On the
  100K x 1536 DBpedia/OpenAI set the f16 rounding loses no recall@10 (0.9994
  before exact rescoring, 1.0 after). Half-width compaction applies to non-key
  columns; key columns keep the F32 key encoding.
- Quantized HNSW traversal: `::hnsw create ... { ..., quantize: turbo4 }` (also
  `binary`, `turbo2`, `turbo8`; default `none`), with optional `rescore: <n>`. The
  index keeps one compact TurboQuant-style code per vector (seeded two-round
  permuted block-Hadamard rotation, Lloyd-Max scalar codebook, length
  renormalisation; 1536-d `turbo4` = 776 bytes vs 6,144 for F32) in the index
  relation at layer 2. Search estimates distances from codes instead of fetching
  full base rows per hop, then rescores the best `k * rescore` candidates exactly
  (default 4 for `turbo2`/`turbo4`, 8 for `binary`, 2 for `turbo8`; all `ef`
  candidates when a search filter is present), so returned distances are exact.
  Codes are maintained by `:put`/`:rm`/import, orphan repair, `::reindex` and the
  parallel bulk build; graph construction always uses exact distances. Indexes
  created without `quantize` are unchanged and store no codes; manifests written
  before this change load as unquantized. `::indices` reports `quantize` and
  `rescore`. Measured on 99.5K x 1536 DBpedia/OpenAI vectors (cosine, `m: 16`,
  one query at a time): SQLite store, `ef: 64`, `turbo4` 21.1 vs 25.5 ms CPU and
  53 vs 67 ms p50 per query at equal recall@10 (0.975 vs 0.973), for +113 MiB of
  codes; in-memory store (30K), 4.2 vs 5.6 ms. In a brute-force codec comparison
  `turbo4` reaches recall@10 0.969 from codes alone and 1.0 after rescoring 2x;
  dense random, Clifford-rotor, quaternion and Givens rotations gave the same
  recall on this data, but the block rotations fell to 0.81 when coordinate energy
  is importance-ordered (matryoshka-style), so the Hadamard rotation is used; a
  1-bit QJL residual was worse than the plain codes at equal size. Design follows
  turbovec (MIT; see `NOTICE.md`). Tests:
  `tests/hnsw_quantized.rs`; benchmark: `examples/hnsw_quant_bench.rs`.
- A `cargo-fuzz` harness in `kleos-core/fuzz/`. The `run_script` target runs arbitrary
  CozoScript against an in-memory database built without network access or
  file-backed engines, under the engine's own query time and memory budgets.
  `gen_corpus.py` builds the libFuzzer dictionary (grammar literals, builtin
  functions, fixed rules) and a seed corpus from the scripts in tests and docs.

### Changed

- **BREAKING (results): Integer overflow is an error.** `Int` arithmetic that does not
  fit in `i64` fails the query instead of returning a wrapped value. Use a float operand
  (`x * 2.0`) for arithmetic that may exceed the range.
- **BREAKING (parse): Block comments nest like Rust's.** A comment that relied on `/*/`
  closing it after an unclosed nested `/*` (`/* a /*/`) is now a parse error.
- Reading a stored row no longer formats its key for an error message on every
  decode. The label (a `Debug` dump of the key tuple, or a hex dump of the key bytes
  for value-only reads such as HNSW self rows) was built eagerly and discarded on
  success; it is now built only when decoding fails. Exact HNSW search on the
  in-memory store went from 15.0 to 5.6 ms of CPU per query (30K x 1536, `ef: 64`).
- `ROADMAP.md` rewritten for kleos: agent-memory north star, a hardening phase first, and the
  inherited mnestic backlog as unscheduled candidates. mnestic's 0.19 plan is marked as an
  inherited record.

## mnestic, unreleased after 0.18.0 (included in the kleos base)

### Added

- Native Rust governed multi-transactions: a serialized client and host-owned
  worker with an absolute deadline, bounded idle waiting, inherited statement
  memory limits, cooperative cancellation and per-query warning flushes. Every
  query error rolls back the whole transaction; the legacy API remains unchanged.
  Hosts can retain admission until the actual worker exits. Stalled bounded
  callbacks are disconnected with `callback.delivery_timeout` after a successful
  commit, without undoing committed data. See
  [the contract](docs/specs/governed-transactions.md), including SQLite/RocksDB
  concurrency differences and uncertain outcomes when cancellation races commit.

### Fixed

- `::warnings` reads and clears the in-memory ring without opening a storage
  transaction. It no longer waits behind an active multi-transaction on memory
  or SQLite backends, which could otherwise stall inspection until idle expiry.

## 0.18.0

This release corrects string-literal decoding, makes nested JSON conversions
consistent with top-level results, and normalizes single-term FTS prefixes.
For example, `Di*` now matches `Diwank` on a lowercase index without rebuilding
that index. **Existing scripts and JSON consumers may need changes:** audit
escaped literals, whitespace-sensitive values, and typed fields before upgrading.
Stored values are not rewritten, and the storage format and `mnestic-rocks`
0.1.12 dependency are unchanged.

### Changed

- **BREAKING (results): String literals (#53).** Double-quoted strings now
  decode the existing quoted escape grammar: `"a\nb"` contains three characters.
  Use `_"a\nb"_` for a literal backslash, or bind a parameter. Double-quoted
  and fenced raw strings retain edge whitespace and comment-like text; trim
  explicitly when intended. Raw fences require matching runs of one or more
  underscores, adjacent to the quotes. Choose a fence longer than any underscore
  run after a quote in its contents. Spaced fences such as `__ "abc" __` now fail.
  Unknown quoted escapes now fail, including in FTS phrases. A trailing backslash
  can escape the closing quote and cause an error later or at end of input.
  Comment markers inside a literal can no longer hide a quote: audit affected
  scripts because an old single literal can become multiple expressions.
  Stored strings and descriptions stay unchanged, but stored-query bodies are
  reparsed on invocation and require the same audit. Parameters and single-quoted
  value semantics are unchanged. See [the contract](docs/specs/string-literals.md).
- **BREAKING (results): Canonical JSON conversion (#54).** Nested values, JSON
  builtins, `to_string`, value-derived object keys and paths, and new writes into
  `Json` columns now use UUID strings, padded standard-base64 byte strings, and
  `"INFINITY"` / `"NEGATIVE_INFINITY"` for scalar infinities. Lists and sets recurse
  through the same conversion. String-valued `to_string` results stay unquoted.
  **Stored JSON and persisted `to_string` output are not rewritten.** Old array/null
  forms can coexist with new strings. Audit known typed fields and migrate them
  client-side if needed: `uuid.UUID(bytes=bytes(arr))` for UUIDs and
  `base64.b64encode(bytes(arr)).decode()` for bytes. Do not infer types from
  arbitrary arrays. Infinity values previously reduced to null cannot be recovered
  without another source of truth. Consumers requiring old array forms must
  explicitly convert known UUID/base64 string fields back to byte arrays; there is
  no legacy-output flag. See [the contract](docs/specs/json-canonical.md).
- **BREAKING (results): FTS prefix normalization (#55).** Prefix queries apply
  configured `Lowercase` / `LowerCase` and `AsciiFolding` filters in order, so `Di*`
  matches `Diwank` on a lowercase index. Incomplete prefixes remain one token;
  tokenization, stemming, stopword removal, length limits and compound splitting
  are skipped. Existing indexes need no rebuild. Applications requiring
  case-sensitive prefix matching must use an index without `Lowercase` (and omit
  `AsciiFolding` if accent distinctions must be preserved). Exact terms retain
  their full analyzer pipeline. See [the contract](docs/specs/fts-prefix.md).

### Fixed

- Integer FTS boosts such as `Di*^3` no longer panic; integer and decimal boosts
  use the same scoring behavior.
- Unsupported quoted multi-token prefixes are rejected before filters can remove
  a word, including inside NEAR. Empty prefixes cannot scan all postings.
- Non-contiguous vectors no longer panic during JSON conversion. The internal
  bottom sentinel renders as null instead of panicking.

### API and compatibility

- The temporary `parser.string_decoding_changed` warning reports potentially
  changed escapes or edge trivia once per script warning drain, using a byte
  offset without literal contents. It is conservative, may flag already-correct
  FTS or hash-containing literals, and cannot detect every hidden-quote case.
  Warnings preceding AST or execution failures remain with the correct database.
  **Remove this migration warning in 0.19.0.**
- Top-level JSON forms remain unchanged except the internal bottom sentinel.
  NaN and non-finite vector elements remain null; F32 elements retain widened-f64
  representation. Existing JSON payloads remain opaque, integers remain exact i64,
  and inbound JSON conversion remains lossy. Native binding conversions and
  MindGraph's separately persisted snapshot format are unchanged.

### Known limitations

- `\uXXXX` escapes remain BMP-only: surrogate pairs and lone surrogates are
  rejected. Literal non-BMP characters work; use Python `ensure_ascii=False` when
  embedding JSON-encoded strings in scripts, or bind parameters.
- Prefixes search indexed terms, including stems: `run*` may match an indexed
  `run` while `running*` does not. Leading wildcards, substring search and
  multi-token phrase-prefix search remain unsupported. Candidate restrictions
  and bounded posting-key scans are unchanged.

## 0.17.0 — 2026-08-31

This release adds atomic, host-controlled copy-in from Parquet and Arrow IPC,
plus candidate-scoped full-text retrieval. Enable Rust's `columnar-io` feature
and call `db.import_columnar_file("events", "events.parquet", &options)`, or use
the same `import_columnar_file` method from the published Python package. An
import commits all decoded rows in one transaction or leaves the target
relation unchanged. FTS `candidates:` changes which documents are eligible for
top-k without changing a retained document's corpus-global BM25 score. There is
no storage-format migration. The `mnestic-rocks` bridge advances to 0.1.12 and
must be published before the `mnestic` crate.

### Added

- **Atomic Parquet and Arrow IPC copy-in**
  ([#11](https://github.com/shuruheel/mnestic/issues/11),
  [`docs/specs/parquet-arrow.md`](docs/specs/parquet-arrow.md)): the
  off-by-default `columnar-io` feature adds `Db::import_columnar_file` and the
  backend-erased `DbInstance` equivalent. One local Parquet, Arrow IPC file, or
  Arrow IPC stream is projected into an existing non-`TxTime` relation using
  put/upsert semantics, target-schema coercion, lossless numeric checks, and
  fail-closed handling for unsupported logical types. Column mappings and
  explicit ceilings for source bytes, rows, decoded-batch bytes, value bytes,
  nesting depth, and wall-clock time let hosts bound admitted work. The report
  returns processed rows and batches plus any HNSW, FTS, or LSH indexes that
  require `::reindex` after the bulk load.

- **Candidate-scoped full-text retrieval**
  ([`docs/specs/fts-candidates.md`](docs/specs/fts-candidates.md)): an FTS atom
  accepts a constant `candidates:` list of base-relation primary keys. Candidate
  membership is applied before sorting, top-k truncation, and base-row fetches;
  noncandidate literal hits are also removed before scoring and result-map
  growth. An eligible document below the global top-k can therefore surface.
  One-column and composite keys are coerced through their declared types; empty
  lists return no rows and malformed or nonconstant inputs fail loudly. Queries
  without `candidates:` retain their previous behavior.

### Changed

- **Python distributions include columnar copy-in:** published wheels and
  source distributions compile `columnar-io`, and
  `CozoDbPy.import_columnar_file` releases the GIL while decoding and committing.
  On the measured macOS arm64 ABI3 wheel, the added Arrow/Parquet support
  increases the artifact by 1,711,642 bytes (14.92%).
- **Bridge build maintenance:** `mnestic-rocks` 0.1.12 carries no-behavior-change
  Rust 1.98 Clippy fixes in its build script. Publish it before `mnestic` 0.17.0,
  which pins the new bridge version.

### API and compatibility

- The Rust surface is additive and feature-gated:
  `ColumnarFileFormat`, `ColumnarImportOptions`, `ColumnarImportReport`, and
  `import_columnar_file` exist only with `columnar-io`. The Python method accepts
  `parquet`, `arrow_ipc_file`, or `arrow_ipc_stream` explicitly; format inference
  from a path extension is deliberately absent.
- Older engines reject the new FTS `candidates:` option instead of silently
  falling back to a global top-k. BM25 document count, average document length,
  and term document frequencies remain corpus-global, so candidate scopes are
  eligibility boundaries rather than statistical-isolation boundaries.

### Known limitations

- Columnar import accepts one process-readable local regular file and requires
  an existing relation. It does not infer schemas, read remote or partitioned
  datasets, import into `TxTime` relations, run triggers or callbacks, or
  maintain search indexes during the load. Arrow export is not included.
- `batch_rows` bounds conversion-loop slices, not peak memory. A decoder may
  allocate a full record batch before `max_decoded_batch_bytes` is checked, and
  the single atomic storage transaction can exceed the conversion batch size.
- Candidate-aware FTS still scans matching posting ranges to preserve global
  document-frequency statistics; its cost is not strictly proportional to the
  allowlist size. Callers remain responsible for bounding candidate lists.

## 0.16.0 — 2026-08-18

A query-reuse minor release: persistent, composable named read queries with
declared parameter contracts. There is no storage-format migration, and the
`mnestic-rocks` bridge remains at 0.1.11.

- **Stored / named read queries** ([#9](https://github.com/shuruheel/mnestic/issues/9),
  [`docs/specs/stored-queries.md`](docs/specs/stored-queries.md)): `::query
  create/remove/list/show/run` persists reusable CozoScript definitions in the
  ordinary reserved relation `mnestic_stored_queries`. A stored query can be
  invoked as a rule atom, where it is hygienically namespace-mangled and enters
  the normal stratification, magic-set, planner, budget, and relation-ACL path;
  local rules shadow catalog names with a structured warning. Parameter
  signatures are declared and introspectable, with optional type coercion and
  definition-time defaults. Bodies are read-only in v1; atom form loudly
  rejects program-level options such as `:limit`, while `::query run` honors
  them. Definitions are transactional snapshot data, survive reopen and
  export/import, enforce dependency-restricted removal, and carry a depth-32
  defense against hand-edited catalog cycles. SQLite integration coverage pins
  chained/diamond expansion, capture hygiene, typed parameters, aggregation,
  negation, explain output, triggers, imperative scripts, ACLs, tamper defense,
  and live fixed-rule registry revalidation.

## 0.15.0 — 2026-08-18

An operations-and-interchange minor release: bounded query memory, RDF at the
relational boundary, one shared RocksDB memory envelope for multi-database
processes, and a runnable guide for migrating JSON-LD and tree-shaped data.
`mnestic-rocks` 0.1.11 carries the bridge half of the shared-memory work and
must be published before the `mnestic` crate.

- **Runnable JSON-LD / tree-shaped modeling guide**
  ([#7](https://github.com/shuruheel/mnestic/issues/7)): documents the stable
  relational shapes for nested objects (parent/child rows with IDs), arrays
  (one row per element with a zero-based position), and heterogeneous trees
  (node plus adjacency relations). It also shows how to keep a raw `Json`
  relation queryable with `parse_json()`, `get()`, `maybe_get()`, and `json()`
  during an incremental migration. An SQLite integration test executes the
  guide's schemas, writes, joins, and recursive traversal so the examples
  cannot silently drift from CozoScript.

- **A per-query memory budget** (spec
  [`docs/specs/memory-budget.md`](docs/specs/memory-budget.md), signed
  2026-08-16): the temp stores that hold every intermediate of semi-naive
  evaluation previously had zero accounting, so a mis-authored join — a
  cartesian intermediate is the canonical case — would OOM-kill the **host
  process**. Now every store charges an estimated byte cost per allocation
  event (real insertions, merge key-clones, bounded/dominance
  re-materializations; moves are accounting-neutral, evictions and duplicate
  drops are debited, and each store releases exactly what it charged on
  drop), the sort and final-result staging copies are charged too, and a
  budget trip aborts the query with the new
  **`eval::mem_budget_exceeded`** diagnostic — before the memory is spent and
  before any commit, so a tripped mutable script leaves no partial writes and
  the Db keeps serving. The knob is a triple, min-composed exactly like the
  wall-clock budget: a per-block **`:mem_limit <bytes>`** option, a per-call
  `ScriptRunOptions::with_mem_limit`, and a Db-wide
  `set_default_query_mem_limit` — **script text can only tighten the budget a
  host set, never raise it** (scripts are increasingly LLM-authored). The
  figures are logical estimates of engine-held bytes, not process RSS, and are
  not stable across releases. **Known-uncharged channels** (spec §4/Q3 — the
  seam was not clean, so they are documented instead): `materialized_join`'s
  right-side cache and the normal-aggregate work map; an aggr-heavy or
  matjoin-heavy query can exceed an armed budget through them. The engine
  default stays **unset** (a library
  must not change behavior under its hosts), but `cozo-bin` — an application —
  ships with a 4 GiB default budget (`--default-query-mem-limit`, `0` for
  unlimited), and its HTTP query payload accepts a per-request `mem_limit`.
  Determinism posture, per the signed spec: a query comfortably over budget
  always errors and one comfortably under always succeeds; at the margin,
  trip-vs-succeed and the rule named in the error may vary with parallel
  scheduling. No embedded Datalog engine ships a per-query memory bound
  (Soufflé has none; Datomic queries can OOM the host JVM; RDFox's
  `max-memory` is whole-server) — this is the "safe substrate inside your
  process" story made concrete.

### RDF at the boundary: Turtle-family reader, IRI helpers, round-trip export

RDF/OWL/SHACL as interchange languages, Datalog as the implementation: RDF
converts to ordinary relations at the edge, and nothing inside the engine
knows it came from RDF. Spec: `docs/specs/rdf-boundary-io.md` (all §12
decisions signed 2026-08-16; Q6 overridden — see the migration note below).
Zero grammar changes, zero new sysops, zero new column types.

- **`RdfReader` fixed rule** (new `rdf-io` feature): reads Turtle, N-Triples,
  N-Quads and TriG (via oxttl) into the fixed 6-column shape
  `subject, predicate, object, graph, language_tag, datatype` — all
  `Str`/`Null`. Triple formats emit `Null` graphs; IRIs are plain strings;
  blank nodes keep their `_:label` form by default, with opt-in
  `skolemize: <namespace-iri>` minting deterministic source-salted uuid-v5
  IRIs (same source ⇒ same IRIs, different source ⇒ different). Literals stay
  lexical with the tag/datatype carried alongside — **no silent coercion**;
  an explicit `xsd:string` datatype normalizes to `Null` (RDF 1.1: same term
  as a plain literal). Options: required `url` (`file://` local; anything
  else needs `requests`, exactly like CsvReader), `format` (defaulting from
  the `.ttl`/`.nt`/`.nq`/`.trig` extension), `base`, `prefixes`,
  `prepend_index`. First syntax error aborts with position — no
  lenient/error-recovery modes. The poison flag is consulted every 4,096
  statements (an improvement on the shipped readers, which never check it).
- **IRI helper builtins** (always registered, like every scalar function):
  `iri_valid(s)`, `iri_resolve(base, rel)` (typed errors, never panics),
  `curie_expand(prefix_map, curie)`, `curie_compact(prefix_map, iri)` — the
  prefix map is a JSON object. Backed by oxiri, which becomes an
  **unconditional dependency** by signed decision (spec §12 Q3; small,
  pure-Rust, zero transitive deps).
- **Round-trip export**: `Db::export_relation_as_rdf(relation, format,
  prefixes)` (Rust API behind `rdf-io`, mirroring `export_relations`' scan
  semantics) serializes relations that match the 6-column shape by position
  — strict in v1 (spec §12 Q5) — back to any of the four formats.
- Feature wiring: `rdf-io = ["data-import", "dep:oxttl", "dep:oxrdf"]` —
  the parser deps compile only under the feature (stricter than the `csv`
  precedent; the default build gains zero bytes), and no oxttl/oxrdf/oxiri
  type ever appears in the public API. `rdf-io` **implies `data-import`**
  deliberately: an RDF reader is a script-controlled file reader, and there
  is exactly one trust gate for those.

**Migration note — the PyPI wheel's reader posture is deliberately reversed
(signed Q6 override).** The published `mnestic` wheel now builds with
`rdf-io`. Because `rdf-io` implies `data-import`, the wheel **also registers
`CsvReader` and `JsonReader` again** — the 0.14.0 "wheels expose no
script-controlled reader" posture is traded for reach (Python users are the
likeliest RDF-corpus holders). And since the wheel has always compiled
`requests`, **script-controlled HTTP(S) fetch is reachable from CozoScript in
the wheel**: any script you run can read process-readable files and make
outbound requests. Run only trusted CozoScript, or build the binding from
source without `rdf-io` to restore the locked-down default. Rust consumers
are unaffected unless they opt into `rdf-io`.

### Cross-instance shared memory for the RocksDB backend

One shared block cache + `WriteBufferManager` envelope for many embedded
RocksDB instances in one process (spec:
`docs/specs/cross-instance-memory.md`; the construction Kafka Streams, Flink,
and TiKV each converged on). **Requires a mnestic-rocks release, which must
publish before the mnestic crate that pins it.**

- **The handle**: `RocksMemoryResources::new(RocksMemoryConfig { total_bytes,
  memtable_fraction })` — one LRU cache of `total_bytes` (THE envelope;
  `strict_capacity_limit=false`, default high-priority pool) plus one
  `WriteBufferManager` of `total_bytes × memtable_fraction` charged INTO the
  cache as dummy entries (cost-to-cache: the memtable share is a carve-out of
  the envelope, not an addition beside it; `allow_stall=false`). Validation
  is typed and total: nonzero total, fraction strictly in (0, 1). Immutable
  after construction; clones are refcounted handles to the same live objects.
- **Entry points**: `new_cozo_rocksdb_with_memory(path, &handle)` beside
  `new_cozo_rocksdb` for Rust hosts; for bindings and string-configured
  hosts, `DbInstance::new`'s previously-ignored `options` JSON gains
  `{"shared_memory": {"total_bytes": …, "memtable_fraction": 0.25}}`
  (fraction optional, default 0.25) — the first use fixes the process-default
  config, an identical later config joins, a differing one is a typed
  conflict error naming both configs. Non-RocksDB engines ignore the key.
- **Override loudness**: with a handle present, the shared cache overrides
  any `block_cache` the `<path>/options` file declares — and the open logs a
  warning naming both capacities (the 0.13.0 lesson: never drop declared
  options silently). Participating instances also get
  `cache_index_and_filter_blocks_with_high_priority=true`.
- **Property surface**: per-instance `rocksdb.block-cache-usage`,
  `rocksdb.cur-size-all-mem-tables`, `rocksdb.estimate-table-readers-mem`
  via `Db::rocksdb_memory_stats()` / `DbInstance::rocksdb_memory_stats()`
  (a struct of `Option<u64>`; all `None` on non-RocksDB engines), plus
  handle-level accessors (`memory_usage`, `mutable_memtable_memory_usage`,
  `dummy_entries_in_cache_usage`, `buffer_size`, `cache_capacity`,
  `cache_usage`). Charged into the envelope: data blocks, index/filter
  blocks, memtable bytes. NOT charged: table readers outside the cache, WAL,
  allocator retention — a soft envelope, disjoint from the query memory
  budget.
- **Bridge (`cozorocks`, all additive)**: opaque `RocksMemoryResources`
  crossing cxx as a `SharedPtr` function argument,
  `open_db_with_resources(...)` beside an untouched `open_db` (both now call
  one shared install helper), `get_int_property` on `RocksDbBridge`, and a
  `DbOpenReport` carrying the override facts to Rust for logging. Also fixes
  the options-file cache-attach CF loop (`bridge/db.cpp`), which indexed
  descriptor `[0]` on every iteration and dereferenced
  `GetOptions<BlockBasedTableOptions>()` without the null check its sibling
  0.13.0 fix block has — a non-block-based table factory in an options file
  crashed the open (reachable only by direct bridge consumers passing a
  cache).
- Tests: `cozo-core/tests/cross_instance_memory.rs` (spec §8 matrix —
  sharing-is-real, conflict semantics, validation, override loudness,
  CF-loop regression, non-RocksDB `None`s, over-budget write storm).

## 0.14.0 — 2026-08-08

A security-first minor release, with the full-text retrieval, diagnostics,
durability, planner-regression, and MCP work that had accumulated after
0.13.1. This is 0.14.0 rather than the previously sketched 0.13.2 patch because
the bank now includes deliberate default and access-boundary changes that need
one visible migration point.

### Security — close the engine's indirect-read and deployment boundaries

Six independently reachable security boundaries are now explicit and tested:

- **Stored-relation reads use one checked resolver**
  ([#40](https://github.com/shuruheel/mnestic/pull/40)). Ordinary indexes
  inherit their base relation's effective ACL, and positive/negated atoms,
  fixed-rule inputs and arity checks, imperative returns, exports, history
  reads, and graph-projection sources all resolve through the same access
  check. A relation changed to `Hidden` can no longer remain observable through
  an index or a graph projection's cold or warm cache.
- **The HTTP server authenticates loopback by default**
  ([#39](https://github.com/shuruheel/mnestic/pull/39)). The explicit
  loopback-only `--insecure-no-auth` escape hatch validates the configured
  HTTP/1 Host or HTTP/2 authority, Origin, and Fetch Metadata before granting
  local access. Remote transactions are bound to stable principals and
  re-authorized on every operation; active work cannot be reaped as idle;
  concurrent finish/query races are rejected; and read-only transactions
  reject writes. Import and backup require mutable grants, and HTTP backup
  paths are confined to the configured backup directory.
- **CozoScript data readers are opt-in**
  ([#36](https://github.com/shuruheel/mnestic/pull/36)). `CsvReader` and
  `JsonReader` can read process-local files and, with HTTP support, make
  outbound requests. They are no longer registered by default; trusted Rust
  deployments must enable the new `data-import` feature explicitly. The
  default engine, server, and Python wheels expose no script-controlled file or
  network reader.
- **Malformed vectors fail before unsafe decoding**
  ([#28](https://github.com/shuruheel/mnestic/pull/28)). MessagePack byte
  payloads whose length is not aligned to the declared `f32` or `f64` element
  width now return a deserialization error instead of reaching an out-of-bounds
  unsafe copy.
- **Python vector-store configuration cannot inject CozoScript**
  ([#24](https://github.com/shuruheel/mnestic/pull/24)). The LangChain and
  LlamaIndex adapters accept only bare identifiers for relation, column, index,
  distance, and dtype fields and coerce numeric HNSW settings before script
  construction. The fixed packages publish as `langchain-mnestic` 0.2.1 and
  `llama-index-vector-stores-mnestic` 0.2.1.
- **Persistent relation IDs serialize across TiKV clients**
  ([#37](https://github.com/shuruheel/mnestic/pull/37)). Allocation now reads
  and advances the persisted counter inside the relation-creation transaction,
  so stale independent clients cannot assign the same ID. A pinned real-cluster
  CI test races two clients and asserts distinct relations; abandoned TiKV
  transactions roll back on drop instead of panicking.

The combined security tree passed the complete mnestic CI matrix and
MindGraph's Linux/macOS, RocksDB-backed core and server acceptance suites. The
crates.io publishing workflow was hardened too
([#22](https://github.com/shuruheel/mnestic/pull/22)): its OIDC token is issued
only after the protected `crates-io` GitHub Environment gate.

**0.14.0 migration notes:**

- HTTP clients, including loopback clients, must authenticate unless the server
  is deliberately started with loopback-only `--insecure-no-auth`.
- HTTP backup paths are relative to the configured backup directory.
- Indirect reads that previously leaked a `Hidden` relation now fail with an
  access error.
- Rust callers that intentionally expose `CsvReader` or `JsonReader` to trusted
  scripts must enable `data-import`; add `requests` too for HTTP(S) sources.
- Quoted multi-word FTS queries now mean exact phrases. Drop the quotes to keep
  the old AND-of-terms behavior.

### Added — bounded, resumable MCP output (`mnestic-mcp` 0.2.0)

Every wide-result MCP tool (`search`, `find_related`, `list_recent`, and
`recall_as_of`) now accepts `max_bytes` (default 20,000). Results cut at item
boundaries, never inside content, and return `truncated`, `remaining`, and a
single-use continuation token. `continue_result(token)` walks the remaining
items under the same bound; an individually oversized first item is returned
alone and flagged so pagination can never stall on an empty page. Tokens are
in-process, LRU-capped, and return an actionable re-run hint after expiry or a
server restart.

### Verification and release operations

- The recursive planner tier now pins transitive closure, seeded reachability,
  and same-generation plan shapes plus exact-count execution oracles, with a
  nightly 10,000-iteration deep-fixpoint cap.
- The temporal-belief benchmark is published in the sibling
  `mnestic-benchmarks` repository: both time axes are checked against an
  independent Python oracle across proof search, contradiction, retraction,
  and reconciliation.
- The dependency-drift monitor moved to the ecosystem-level `mindgraph-rs`
  workflow, while this repository keeps its self-contained version and
  dependency-freshness checks.

### Changed — a quoted full-text query is now an exact phrase

**The migration note first, because the break is the fix working:** a quoted
multi-word FTS query — `'"connection refused"'` — previously matched any
document containing both words *anywhere*, in any order; the quotes were
parsed and then silently discarded (`parse/fts.rs` collapsed every kernel to
the same literal, and tokenization split it into an AND of independent terms).
It now matches only documents containing the words **adjacent and in order** —
strictly fewer, more relevant rows. If you relied on quoted-means-AND, drop
the quotes: unquoted queries are byte-identical to before.

Design, prior-art verification (Lucene/FTS5/Elasticsearch), and the empirical
record are in `docs/specs/fts-phrase-and-snippets.md` — every §10 decision was
verified against the shipped 0.13.1 engine before build.

What "exact phrase" means, precisely:

- Matching anchors on the **token positions the postings have stored since the
  fork point**: document matches at anchor `p` iff every query token with
  analyzer position `qᵢ` occurs at exactly `p + (qᵢ − q₀)`. Overlapping
  occurrences are found; tf for scoring = anchor count; df = the phrase's own
  matching-doc count. No storage-format change in either direction — a 0.14.0
  index opens under 0.13.0 and vice versa.
- **A removed stopword is a one-token wildcard slot, symmetrically.** The
  tokenizer numbers tokens before filters run and `Stopwords` skips without
  renumbering, so `"jumped over the lazy dog"` (EN stopwords) matches "jumped
  over *that* lazy dog" — the position the stopword held constrains nothing.
  This is Lucene's contract, and the alternative (dense renumbering) is the
  `enablePositionIncrements=false` option Lucene *removed* in 4.4 for
  producing broken token streams.
- **Phrases match stems, symmetrically** — both sides run the index analyzer,
  so with an English stemmer `"connection refused"` matches "connections
  refusing" (in that order).
- Quoted single tokens are unchanged: `'"fox"'` ≡ `fox`, including `'"fox"*'`
  prefix search.

Three silent failures became named errors instead of wrong answers:

- **`parser::fts::phrase_prefix_unsupported`** — `'"connection refu"*'`
  previously matched **nothing, silently**: the un-tokenized quoted string was
  prefix-matched against single-token posting keys. The error names the
  `"connection refused" OR refu*` workaround; the real phrase-prefix
  (Elasticsearch `match_phrase_prefix`-style) can land later without breaking
  anything.
- **`parser::fts::phrase_in_near_unsupported`** — a quoted multi-word phrase
  inside `NEAR(...)` previously degraded to its bag of tokens, dropping the
  adjacency the quotes asked for. Quoted *single* tokens inside NEAR stay
  legal.
- **`eval::fts::phrase_without_positions`** — a phrase query against an
  `NGram`-tokenized index: every gram is stored at position 0, so adjacency is
  unfalsifiable and a phrase would match every document containing the grams.
  Term search on such indexes still works. (Lucene answers this same case with
  `IllegalStateException: field was indexed without position data`.)

Also fixed while in that loop: **`NEAR` scanned its first literal twice** —
seeded the intersection, then re-scanned it against itself. Results were
unchanged (self-distance 0 always survived) but every `NEAR` query paid one
redundant full posting fetch; a thread-local scan counter now pins the count
in CI, since the bug class is invisible in results by construction.

### Added — the recursive-workload planner pin (the suite's outstanding tier)

The planner regression suite gated 0.13.1's factorization flip through its
plan-shape gate and nightly LSQB execution tier — but the roadmap's stated
position was that **no further planner pass ships default-on until the
*recursive* workload is pinned**, fixpoint reachability being what this engine
is fastest at and therefore what a regression would hurt most. It is now
pinned (`cozo-core/tests/recursive_workload.rs`):

- **Per-PR:** plan-shape baselines for the three canonical recursive shapes —
  full transitive closure, seeded reachability (the magic-sets shape),
  same-generation (the magic-sets stress shape) — over empty relations, plus
  exact-count execution oracles: closed forms on constructed graphs (chains at
  depth 64→1024, a cycle, a full binary tree) and a Rust-side BFS oracle on a
  seeded pseudo-random digraph, sharing nothing with the engine but the edge
  list.
- **Nightly (`planner-guard.yml`):** a deep-fixpoint pathology cap — 10,000
  semi-naive iterations down a chain, exact answer, generous wall-clock bound.
  Self-contained: the graph is generated in-test.

A finding the baseline records: today the two arms (greedy reorder on/off) are
**identical** on every recursive shape — no shipped pass touches recursive
rules at all (derived-rule atoms disqualify them, and the base cases sit below
the reorder's 3-stored-atom horizon). The moment a future pass does touch
them, the baseline fires. With this tier in place, the standing restriction is
lifted: a future planner pass may go default-on by the same route
factorization took — plan-shape gate, then nightly soak.

### Added — an explicit durability knob: `set_durable_writes`

The RocksDB backend commits **without syncing its WAL** — the log is written
but not fsynced, so an OS crash or power loss can lose the most recent
acknowledged commits (a mere process crash cannot). A defensible default for
an embedded engine, and an indefensible *invisible* one for a database that
offers audited eviction and a crash-safe commit clock. It is now explicit:
`db.set_durable_writes(true)` makes every subsequent write transaction fsync
on commit, and returns whether the backend honors the knob at all. Per-backend
contract (documented on `Storage::set_durable_writes`): `rocksdb` honors it
(note the bulk channels — `batch_put` / SST ingest, i.e. `import_relations`
and restore — are separate non-transactional paths outside the knob);
`sqlite` reports `false` because its writes are **already durable**
(`synchronous=FULL` default); `mem` has nothing to sync. Third-party `Storage`
impls keep compiling — the trait method has a default no-op.

### Added — structured query diagnostics: `::warnings`

The engine has long *known* things worth telling the query author and said
them only through `log::warn!`, where nothing programmatic can see them — a
downstream project shipped a full-scan traversal for months while the engine
warned about it on every call. Warnings are now **typed** (`code`, `message`,
actionable `hint`), buffered in a bounded per-`Db` ring (256, monotone `seq`
so consumers can track what they've seen), and readable from **every** binding
via plain CozoScript — `::warnings` lists, `::warnings clear` empties — plus
`Db::recent_warnings` for embedded Rust. The log side is unchanged; four
findings emit structured warnings today:

| code | finding |
|---|---|
| `query.cartesian_step` | a rule still contains a disconnected conjunction after the greedy reorder — the exact finding that once rotted in the logs |
| `import.stranded_indexes` | a bulk import / restore left FTS/HNSW/LSH indexes stale; the hint names the `::reindex` |
| `fixed_rule.pagerank.unconverged` | PageRank stopped at its `iterations` cap above `epsilon` |
| `graph_projection.over_capacity` | a projection exceeds the cache ceiling and is rebuilt per query |

Codes are stable identifiers; new ones may appear in any release — match on
the ones you know, ignore the rest. Guard: `cozo-core/tests/query_warnings.rs`.

### Added — `bind_spans` and `snippet()`: return the part that matched

For agentic memory, returning a 4k-token document when a 40-token window
matched is the context-budget failure mode. Two pieces, split so neither
tokenizes at read time:

- **`bind_spans: sp`** on the FTS search atom (beside `bind_score`): per
  returned document, the list of `[from, to]` **byte offsets into the original
  text** of the occurrences that evidence the match — a term's own spans, a
  phrase anchor's first-token-`from` to last-token-`to`, `NEAR`'s surviving
  occurrences. The offsets were **already written per posting by every release
  since the fork point** and discarded at decode (`PositionInfo.from/to` sat
  commented out); this reads them, only when the binding is requested. Because
  the index-side analyzer recorded them, they point at the source words even
  when the posting key is a stem — Lucene's offsets-in-postings highlighting
  mode, natively.
- **`snippet(text, spans, window)`** (and `snippet(text, spans, window, open,
  close)` for markup): a pure formatting scalar over those spans — coalesces
  overlaps so markers never nest, picks the densest cluster fitting the
  `window` char budget, cuts on char boundaries only (multi-byte text safe),
  marks truncation with `…`. It never tokenizes, so it cannot disagree with
  the index analyzer — the failure mode that ruled out a `highlight(text,
  query)` shape re-analyzing text at read time.

Guards: `cozo-core/tests/fts_phrase.rs` — 18 fixtures covering adjacency,
ordering (incl. overlapping anchors), the stopword-hole contract from both
sides, stemmed phrases, all-stopword degeneration, the three named errors,
quoted-single-token equivalence, byte-exact spans on multi-byte text, snippet
end-to-end and edge cases, the scan-count regression, and a subset oracle
(phrase results ⊆ AND results on a generated corpus).

## 0.13.1 — 2026-07-27

### Changed — the factorized `count()` rewrite is ON by default

The rewrite shipped in 0.10.5 and has been opt-in ever since, most recently
with its restored `!=` inclusion–exclusion extension (0.13.0, measured ~140× on
LSQB q6). The stated gate was a nightly planner-guard soak on the restored
rewrite, and the soak has now run **green every night since the workflow's
first-ever green on the 0.13.0 PR** — including the tier that asserts the q6
toggle-ON result. So the default flips.

What this changes for you: an eligible single-clause `count()`-over-a-positive-
join is now counted without materializing the join. **Answers are unchanged** —
the pass fires only on a *provably exact* decomposition and declines on anything
unverifiable, and the differential suite asserts naive-equals-factorized on
every shape it covers. What changes is the plan, and therefore the time and
memory a counting query spends.

Measured on LSQB sf0.1 (SQLite, release, M-series, 2026-07-27); both paths
return LDBC's published expected-output count:

| query | naive | factorized | |
|---|---|---|---|
| q1 | 72,360.7 ms | **1,048.2 ms** | ~69× |
| q6 | 42,120.8 ms | **314.0 ms** | ~134× |

The rewrite declines q2, q3 and q9 — a predicate, a negation or a cyclic shape
disqualifies them — and their plans are byte-identical to 0.13.0, which the
plan-shape gate asserts rather than assumes.

**The q1 number is also a correction to how this release was gated.** The
nightly soak that cleared the flip ran the rewrite-ON path on **q6 only**: the
tier's main loop ran the default, and the default was OFF. So q1 — which the
rewrite also fires on — had never been executed in the configuration this
release makes the default. We found it because the plan-shape gate went red on
two queries when we expected one. The execution tier now covers every firing
query, pins the firing set, and asserts that the default path really fired
(a silent decline would otherwise hide behind a green-but-slow run); the rule
going forward is that a default flip widens its coverage *first*.

The kill switch is unchanged and still a `Db`-wide toggle:
`db.set_query_factorization(false)` restores the previous behaviour;
`db.query_factorization()` reports the current setting. The detector advisory
(the `log::info!` line and the `::explain` row) is unaffected — it always ran
independently of the toggle.

### Fixed — the 0.13.0 expected-token parse hints were silently disabled by pest 2.8

0.13.0 shipped "parse errors that name the expected tokens": a `help:` line
listing the literal tokens the grammar would have accepted, and a span pointing
at the real defect rather than at `err.location` (which, for `?[a] a = 1`,
points inside the rule head while the missing `:=` is later). Both are built
from pest's parse-attempt tracking.

**pest 2.8.0 made that tracking opt-in and off by default.** Our dependency
requirement is `pest = "2.7.9"`, which is a caret requirement, so any consumer
resolving fresh after pest 2.8's release builds mnestic 0.13.0 against 2.8.x —
where `err.parse_attempts()` returns `None`, the token list is dropped, and the
span falls back to the location the feature exists to correct. The degradation
is silent and total: the feature reverts to pre-0.13.0 behaviour with no error
and no warning. Our own builds and CI never saw it, because the committed
`Cargo.lock` pinned 2.7.x — a lockfile makes this whole class of regression
invisible to CI, which is why a scheduled fresh-resolve lane now runs
(`.github/workflows/dependency-freshness.yml`).

Fixed by enabling the setting explicitly, and the dependency floor is raised to
`pest = "2.8"` (`pest::set_error_detail` does not exist in 2.7.x, so the floor
is now load-bearing in both directions).

**Where it is enabled matters, and it is not enabled globally.** `set_error_detail`
is a pest-*wide* global, and pest's own documentation notes it "has a higher
performance cost (hence, it's off by default)". An embedded engine that switches
it on at first parse and leaves it on would tax every subsequent *successful*
parse — parse is the dominant cost on cheap high-frequency point reads — and
would silently change the behaviour of any other pest-based parser in the host
process, which now includes whatever process `mnestic-mcp` or
`langgraph-store-mnestic` is running inside. So mnestic enables it only around a
**re-parse of a script that has already failed**: the success path pays nothing,
the flag is on only for the duration of a re-parse of an already-failing script,
and a depth counter makes concurrent failing parses safe (the last one out
restores it). `parse/mod.rs`.

### Fixed

- **Source compatibility across the uuid 1.x range.** `op_rand_uuid_v1` now
  builds its context through the version-neutral `uuid::v1::Context` alias
  (under `#[allow(deprecated)]`) instead of the `uuid::ContextV1` name
  introduced later in the 1.x line, so the crate compiles against any uuid a
  consumer's lockfile resolves within our declared `1.8.0` requirement. No
  behavioural change; the generated UUIDs are identical.

### Changed

- **Dependency refresh** across the workspace (`Cargo.lock`), with the Rust
  sources reformatted. No API or behavioural change beyond the two items above.

## 0.13.0 — 2026-07-18

### Fixed — RocksDB table options were silently discarded on every open (`mnestic-rocks`)

**Any `BlockBasedTableOptions` you configured — block cache, block size, index/filter caching —
never reached RocksDB.** `open_db` loaded your options file, then a few lines later
default-constructed a fresh `BlockBasedTableOptions`, set exactly two fields on it
(`filter_policy`, `whole_key_filtering`), and **reset the table factory with it**
(`cozorocks/bridge/db.cpp`). Everything the options file supplied — and everything
`default_db_options()` had set — was thrown away. The block cache reverted to RocksDB's 8 MB
default, `block_size` to 4 KB, `cache_index_and_filter_blocks` to `false`. Silently: no error, no
warning, and nothing in the API to observe it with.

It fired on **every** open, because `cozo-core`'s RocksDB backend sets `use_bloom_filter`
unconditionally (`cozo-core/src/storage/rocks.rs`). An embedded engine was quietly running with a
read cache two orders of magnitude smaller than its host had asked for.

Measured against RocksDB's own effective-options dump (the `OPTIONS-######` file it writes on open),
with an options file requesting `block_size=16384` and `cache_index_and_filter_blocks=true`:

| | before | after |
|---|---|---|
| `block_size` | 4096 *(default)* | **16384** |
| `cache_index_and_filter_blocks` | false *(default)* | **true** |

Fixed by seeding the table options from the factory **already in effect** rather than from a
default-constructed one — the same `GetOptions<BlockBasedTableOptions>()` idiom the function already
used correctly for its `block_cache_size` path.

Two adjacent bugs in the same function, fixed with it:
- **`block_cache_size` was ignored.** It hardcoded a 1 GiB LRU cache regardless of the value passed,
  so a caller asking for 64 MB silently got sixteen times that.
- **That cache was then dropped on the floor** unless an options file was also supplied — it was only
  ever installed inside the options-file branch.

⚠️ **This changes what RocksDB *writes*, not only what it caches.** With the table options now
honoured, newly written SSTs pick up the configured `block_size` (and any `checksum` /
`format_version` the options file sets). Existing SSTs record their own settings and stay readable —
RocksDB handles the mix — so this is forward-compatible and needs no migration. But a store upgraded
in place will begin emitting differently-formatted blocks as it compacts.

⚠️ **Performance numbers taken before this fix measured a slower engine than mnestic actually is.**
Any read-path benchmark run against a RocksDB store — ours included — ran with a 4 KB block size and
no configured block cache.

Inherited from upstream Cozo; present since the options-file path was introduced. Requires a
`mnestic-rocks` release, which must publish **before** the `mnestic` crate that pins it.

### Fixed

- **BM25 counted the wrong corpus, by re-scanning the whole relation on every
  query.** Okapi BM25's IDF is `ln(1 + (N − df + 0.5) / (df + 0.5))`. `df` is
  counted from the full-text index and `avgdl` is averaged over the full-text
  index — but `N` was counted over the **base relation**, by a `range_count`
  over its entire key range, **on every FTS query**. (The 0.8.4 work made
  `avgdl` an O(1) read off a process-level doc-stats cache and left `N` behind;
  the cache that would have served it sits ten lines away in the same file.)

  Two consequences, both measured:

  - **Wrong scores.** A row whose extracted text yields no tokens is a row in
    the relation but *not a document in the collection being searched* — it
    carries no postings and can never be a hit, yet it enlarged `N`. Rows loaded
    by `import_relations` (which maintains no FTS index) did the same, at
    arbitrary magnitude. Measured: **200 empty-body rows beside 3 real documents
    moved a BM25 score from 0.1335 to 4.065 — a 30× error.** (The FTS posting
    leak fixed in 0.12.1 was, for comparison, a 55% error.)
  - **Ruinous latency at scale.** O(corpus) per query — and in practice
    O(corpus *bytes*), because counting rows drags the relation's whole value
    payload through the block cache. On the first-ever `medium`-scale run of
    `hybrid-recall-bench` (RocksDB, real embeddings, 384-d): at 400k chunks the
    **FTS leg cost 2,621 ms against 350 ms for the HNSW leg**, and hybrid p50
    went from 150 ms at 40k chunks to **3,396 ms** at 400k (p99: 201 ms →
    10,358 ms) while recall, ingest rate and disk all scaled linearly.

  `N` now comes from the same doc-stats cache as `avgdl` — the number of
  documents carrying at least one posting — so it is both correct and free:
  the cache is already seeded and already maintained on every write. The
  per-query scan is *deleted*, not memoized (`FtsCache` is now stateless).
  `fts/indexing.rs`; `docs/specs/fts-corpus-stats.md`; guarded by
  `cozo-core/tests/fts_corpus_stats.rs`.

- **The FTS doc-count drifted upward on every no-op write.** A value-unchanged
  `:put` skips the posting delete (an identical tuple derives identical
  postings — `query/stored.rs`), but the put still bumped the doc-stats counter
  `+1` with no matching `−1`. This was invisible for as long as
  `avgdl = total / n` was the counter's only consumer — a re-put inflates both
  terms together and the ratio barely moves — but BM25's `N` reads `n` directly.
  Measured: **ten identical re-puts moved a score from 0.1335 to 2.27 (17×).**
  `put_fts_index_item` now probes whether the document is already indexed before
  counting it, mirroring the probe `del_fts_index_item` already performs.

- **Pre-epoch timestamps no longer panic or poison an in-memory/SQLite
  database.** Conversion between `SystemTime` and signed microseconds now
  handles dates before 1970, validity strings can round-trip those values, and
  the affected store/pool locks recover after a panic instead of making every
  later operation fail. Guard: `cozo-core/tests/pre_epoch_validity.rs`.

- **HNSW rebuilds and updates preserve their graph invariants for invalid and
  removed vectors.** Zero-vector cosine distance is finite, `NaN` candidates
  use a total order instead of panicking in heaps, bulk builds restore paired
  tombstoned edges, and updates that null or remove a vector delete its stale
  HNSW slot. Guards: `hnsw_correctness.rs`, `hnsw_build.rs`, plus the exact
  total-order unit test.

- **Hybrid search keeps fusion legs and graph seeds distinct.** Generated
  relation labels are unique across semantic, text, extra and graph legs, and
  a graph seed is no longer reported as its own graph-derived contribution.
  Guard: `cozo-core/tests/hybrid_graph_leg.rs`.

- **Restore/open cannot silently reuse a live relation id.** The persisted
  counter is reconciled with the highest id observed in relation metadata after
  restore and at open; duplicate ids are diagnosed instead of being hidden by
  the counter. Internal backup/restore regressions cover both paths.

- **Corrupt value blobs are ordinary query errors, not process panics.** All
  built-in storage backends now use the additive fallible decoder
  `try_decode_tuple_from_kv`; point joins preserve nested read errors, and
  `::repair_corrupt` treats an undecodable value as a row to remove. The
  existing `decode_tuple_from_kv` signature remains available for source
  compatibility. The SQLite regression corrupts a real row, verifies scan and
  point-lookup errors, repairs it, and reads the relation again.

- **Corrupt HNSW and FTS index rows also return ordinary errors.** The original
  fallible-decoder pass stopped at base relations: three HNSW mutation paths and
  the FTS posting reader still sliced and deserialized index values with
  `unwrap()`, while the HNSW neighbour iterator unwrapped the newly reachable
  scan error. Index value decoding now uses the same checked value-blob path,
  HNSW row shapes are validated before indexing into them, and iterator errors
  propagate instead of being swallowed or panicking. Rebuild an affected index
  with `::reindex <relation>`. SQLite regressions corrupt real HNSW and FTS index
  rows and assert `eval::corrupt_value_blob` errors.

- **`::reindex` and `::repair_corrupt` participate in imperative-program lock
  planning.** They request a write transaction and relation lock up front, then
  skip their local lock when the program already owns it, avoiding an unlocked
  mutation on some backends and a recursive-lock deadlock on others.

### Added — `HybridSearch` budgeted-expansion mode and optional legs

The 0.12.0 headline (`BudgetedTraversal`) is now reachable from the one-call
`hybrid_search` surface — the dict the PyPI wheel already exposes — instead of
only by hand-writing Datalog around a FixedRule.

- **Budgeted graph legs** (spec §9, resolved): setting `GraphLeg::max_nodes`
  switches the leg from the recursive min-hop rule to a generated
  `BudgetedTraversal` call — cheapest-first weighted expansion under a global
  distinct-node budget, with optional `max_cost`, an exact layered-label depth
  bound (`max_hops` → `max_depth`), optional `weight_col`, an optional
  liveness gate (`gate_relation`/`gate_cols`/`admit` — emitted in the named,
  order-independent binding form), and `graph:` naming a pre-created cached
  projection (**the production path**: the positional edge input pays a full
  scan + CSR build before any budget applies). Seeds default to the union of
  the configured vector/FTS legs' own top-k (`seed_from_legs`), plus explicit
  `seeds`; seed roots are excluded from the fusion contribution. With
  `detailed: true`, the output head gains the cheapest-path witness
  (`parent`, `depth`) — opt-in: without a budgeted leg every existing shape
  is byte-identical (snapshot-guarded).
- **Optional legs**: `vector_index`/`fts_index` are now `Option<String>` —
  configure any non-empty subset of {vector, FTS, graph legs, extra lists}
  and only those are generated and fused. A payload without its leg
  (`query_vector` with no `vector_index`, and vice versa) is a loud error,
  never a silently dropped signal.
- **`GraphLeg` in recursive mode is unchanged** — adding the budgeted mode
  leaves its generated script byte-identical (snapshot-guarded), and it
  remains the only graph leg in a `minimal` build (`BudgetedTraversal` registers under `graph-algo`);
  a budgeted leg configured without the feature errors loudly at build time.
- **Source-breaking (the honest minor):** `HybridSearch` and `GraphLeg` are
  now `#[non_exhaustive]` — construct with `Default` + field mutation. This
  is deliberately paid once, in the same release that adds eight `GraphLeg`
  fields, so that future field additions are never breaking again. The Python
  wheel's dict surface gains **optional keys only** (`max_nodes`, `max_cost`,
  `weight_col`, `graph`, `seed_from_legs`, `gate_relation`, `gate_cols`,
  `admit`; `graph_legs` entries no longer hard-require
  `edge_relation`/`seeds`/`max_hops` — the builder's validation owns the
  invariants) — existing dicts parse unchanged. Because every leg key is now
  optional, **unknown keys in a `graph_legs` entry are rejected loudly**: a
  typo (`max_hop`, `seed_from_leg`) would otherwise silently run the leg with
  defaults.
- **Validation symmetry and no silently-ignored config**: a configured FTS
  leg requires a non-empty `query_text` (mirroring the vector leg — an empty
  query previously surfaced as a cryptic parse error mid-execution), and a
  budgeted `graph:` leg rejects non-default `from_col`/`to_col` exactly as it
  already rejected a stray `edge_relation`/`weight_col` (the projection
  carries its own edges; ignored config is never accepted silently). The
  newly-usable named fixed-rule input form rejects a **repeated column key**
  (`*g{uid: a, uid: b}` used to silently drop the first pair).

### Added — the `!=` inclusion–exclusion count rewrite is restored, behind a type gate (default OFF)

The factorized-count pass's `!=` extension — built in 0.10.5 and cut 34
minutes later (`a60a8013`) for a silent Int/Float miscount — is back, sound
this time. The miscount class: the correction term implements "equals" as a
**join** (under which `Int(1)` and `Float(1.0)` are distinct) while `!=` is
evaluated by `op_neq` (under which they are numerically equal), so a
cross-variant pair escaped both terms. The restore adds a **type gate**: the
rewrite fires only when every binding occurrence of both operands of every
inequality is a declared non-nullable, **variant-stable** stored column and
all occurrences agree on one type — then join equality and `op_neq` agree on
every value the operands can hold (soundness argument written down in
`docs/specs/cardinality-algebra.md` §3.3a; gate discrimination proven by
bypass — the mixed-type suite miscounts 4-for-3 without it).

- **Variant-stable excludes `Any` AND `Json`, recursively through
  `List`/`Tuple` element types**: `JsonData`'s structural `Eq` and its
  `to_string()`-based `Ord` diverge in the current build (`json(-0.0)` vs
  `json(0.0)` are op_neq-equal but join-distinct — the fired rewrite
  overcounted 2-for-0 before the exclusion; found and fixed in the 0.13.0
  review).
- **Default stays OFF** (`Db::set_query_factorization`); the default-on flip
  waits for a nightly soak on the restored path, per the 0.10.5→0.10.7
  planner lesson.
- **Measured** (2026-07-17, LSQB sf0.1, sqlite, M-series, release): q6
  41.7 s → **0.30 s (~140×)** with the rewrite on, count exactly LSQB's
  published oracle either way. The nightly LSQB tier now runs q6 with the
  toggle forced on **and asserts the rewrite actually fired** (the `*fac`
  plan signature), so neither a miscount nor a silently-declining gate can
  ship behind a green arm again.
- The pass now takes catalog access (`&SessionTx`) for the gate — each
  distinct relation's metadata is fetched once per query; with no inequality
  present it stays purely syntactic.

### Changed — `import_from_backup` refuses mismatched schemas

The backup-import path raw-puts the source's rows after a key rewrite — no
type coercion — so it was the one user-reachable way to put a value at rest
that violates its column's declared type (a backed-up `Float` restored into a
declared-`Int` column), which is exactly the invariant the `!=` type gate
rests on. It now requires the source and destination schemas to match
(column names, types and defaults — deliberately stricter than the type
argument needs: a restore across renamed columns is ambiguous about intent,
and refusing loudly beats guessing). Error → error for the corrupting case;
a same-schema restore is unchanged. Code: `tx::import_schema_mismatch`.

### Fixed — a bare `minimal` build did not compile (ungated `rayon`)

`query/eval.rs` imported `rayon` gated only on `not(wasm32)`, while `rayon`
is an optional dependency that `minimal` does not enable — so
`--no-default-features --features minimal` had been **uncompilable**, with
nobody noticing because CI never built that combination and every known
consumer enables `rayon`. Parallel stratum evaluation now degrades to
sequential without the feature, and CI gained a `minimal` job so the
combination (and the budgeted-leg feature-seam error) stays covered.

### Fixed — the named `*rel{col}` fixed-rule input form always panicked

The `fixed_named_relation_rel` parse arm stripped `':'` from an identifier
the grammar defines as `*`-prefixed, so the `unwrap` panicked on every use —
the named binding form for fixed-rule stored inputs was unusable (inherited
upstream bug, present at fork-base). It now parses and binds **by name** in
schema order, which is what makes the budgeted gate's order-independent
`gate_cols` form possible.

### Added — datetime function library (`dt_*`)

We market a bitemporal database; its datetime standard library was three
functions with inconsistent units. This adds the missing surface, on one loudly
stated convention: **timestamps are float Unix seconds** (what `now()` and
`parse_timestamp` already return); **a float second-count is NOT a validity**
(validities count integer microseconds, or an abstract logical tick), and the
one bridge between the two worlds is `dt_to_validity`. Timezone-sensitive
functions take an optional trailing IANA-name string, default UTC — the same
convention `format_timestamp` already used.

- **Component extractors** `dt_year`, `dt_month`, `dt_day`, `dt_hour`,
  `dt_minute`, `dt_second`, `dt_dow` (ISO: Monday = 1), `dt_doy` — all
  `(ts, tz?) -> Int`.
- **`dt_trunc(ts, unit, tz?) -> Float`** — unit ∈ `year | quarter | month |
  week | day | hour | minute | second`; weeks are ISO (Monday). DST rules,
  documented: for sub-day units an ambiguous local time resolves to the fold
  arm **the input instant itself belongs to** (its own offset disambiguates —
  truncation stays containing and idempotent inside a fall-back fold); for
  `day` and coarser, to the earliest occurrence. A local time in a gap
  resolves to the first valid local time after it — probed in 15-minute steps
  for 4 hours, then hourly to 26 hours, so historic deep transitions (the
  6–10 h Antarctic station foundings, full-day Pacific date skips) resolve
  instead of erroring. Range edges error loudly, never panic.
- **`dt_add(ts, n, unit) -> Float`** — calendar-aware for
  `month`/`quarter`/`year`, clamping to month end (Jan-31 + 1 month =
  Feb-28/29); fixed-duration for `week`/`day`/`hour`/`minute`/`second`.
  Overflow errors, it never panics.
- **`dt_diff(a, b, unit) -> Int`** — signed `a - b`, truncating toward zero,
  antisymmetric (`dt_diff(a,b,u) == -dt_diff(b,a,u)`); `month`/`quarter`/
  `year` compute the calendar magnitude on the (later, earlier) pair — the
  largest `n` with `earlier + n unit <= later`, consistent with `dt_add`'s
  clamping in the forward direction — and apply the sign afterward. Because
  clamping is asymmetric, a negative result is deliberately NOT the floor
  form: `dt_diff(Jan-30, Mar-31, 'month')` is `-2`, not `-3`.
- **`dt_format(ts, fmt, tz?) -> Str`** — strftime. The format string is
  pre-validated: an invalid specifier is a loud error, where calling chrono
  directly would panic (`dt_format` is expected to receive LLM-authored text).
  `format_timestamp` stays, for RFC3339.
- **`dt_to_validity(ts_seconds, is_assert?) -> Validity`** — the typed bridge:
  seconds → microseconds *inside* the function, where the unit is known. With
  it, `@` and `:as_of` now accept a `Validity`-typed expression
  (`@ dt_to_validity(parse_timestamp('2024-01-01'))` reads as-of that instant);
  previously `DataValue::Validity` in a temporal spec was an error. Together
  with 0.12.2's float rejection this closes the seconds-vs-microseconds trap:
  the raw-float misread errors loudly, and the typed path is the idiomatic
  spelling. The seconds-as-`Int` form (`@ 1704067200`) remains inherently
  ambiguous — valid time is an abstract clock, so no magnitude gate can reject
  it — which is exactly why the typed bridge exists. Both `dt_to_validity`
  and `validity()` now **reject the reserved i64 extremes** (`MAX_VALIDITY_TS`
  = 'NOW'/'END', and the terminal retract sentinel) that every other write
  path already fences.
- **`parse_timestamp` widened** (error → success, additive): accepts exactly
  three enumerated forms — RFC3339; `"YYYY-MM-DD hh:mm:ss[.fff]"` read as UTC;
  `"YYYY-MM-DD"` read as midnight UTC. Nothing else; the error message
  enumerates the accepted forms. **Validity string literals (`@ '...'`,
  `:as_of '...'`) share the same grammar** — one parser
  (`parse_datetime_utc`), so a literal one entry point accepts is never
  rejected by the other (the naive-datetime form is new for `@` literals too).
- Note: the new `dt_*` names are now **reserved against user registration** —
  a downstream `register_custom_aggr` under one of these names will fail at
  registration after the upgrade.

### Added — parse errors name the expected tokens

A failed parse now points its caret at the **deepest position the parser
reached** (previously `err.location`, which for `?[a] a = 1` sat inside the
rule head while the defect — the missing `:=` — was later), and carries a
`help:` line naming the literal tokens that would have been accepted there,
e.g. `expected one of: \`:=\`, \`<-\`, \`<~\``. Whitespace/comment noise is
filtered; the hint is suppressed entirely when the surviving candidate set is
empty or too large to identify a defect — an unactionable hint being the
failure mode this exists to kill. Built on pest's parse-attempts tracking; no
grammar change, and `ParseError` stays crate-private (the improved text flows
to every surface — Rust, Python wheel, REPL — for free).

In the same agent-actionable-errors spirit, **index-search diagnostics now
carry the code of the index kind that actually failed** (upstream #231/#257):
an FTS search with a missing `query:` no longer says "required for HNSW
search" under an `hnsw_query_required` code (now `fts_query_required`, and
the same for a missing `k`); the LSH normalizer's two reused HNSW codes are
now `lsh_query_required`/`expected_int_for_lsh_k`; FTS `k` gets
`expected_int_for_fts_k`; and the generic index-not-found fall-through —
which fires when *no* index of any kind matched — is `eval::index_not_found`
instead of `eval::hnsw_index_not_found`. User-visible error **codes** change;
no consumer in the ecosystem asserts on the old ones (verified).

### Verification and release gates

- **Continuous FTS/HNSW maintenance is now a release regression.** A persistent
  SQLite test creates both indexes over an empty relation, performs 64 indexed
  writes across a database reopen, repeatedly verifies exact-vector HNSW recall,
  and replaces FTS documents while asserting that old postings disappear. No
  `::reindex` is used. Guard: `cozo-core/tests/index_continuous_writes.rs`.
- **Clippy now covers every target.** CI and both publish workflows run
  `cargo clippy -p mnestic --all-targets -- -D warnings`, so integration tests
  and registered benchmarks cannot silently rot outside the release gate. The
  stale `read_path` benchmark call to `parse_script` was repaired, and
  `cargo check -p mnestic --benches --release` is green. Same rot class, same
  fix: the `cozo-core-examples` `run_parse_ast` binary — uncompilable since
  `parse_script` gained the custom-aggregate registries parameter — was
  repaired (it is outside the clippy gate, which is how it rotted silently).

### Changed

- **BM25 scores change on corpora containing rows that are not documents** —
  rows whose extracted text is empty, whitespace-only or `Null`, and rows
  bulk-loaded via `import_relations`. Their presence no longer inflates `N`, so
  IDF (and therefore ranking, for multi-term queries) shifts. On a corpus in
  which every row is a document — the ordinary case — scores are unchanged, and
  the existing `bm25.rs` / `fts_avgdl.rs` suites pass byte-identically.

## 0.12.2 — 2026-07-13

### Fixed — the validity float channel (a silent temporal corruption)

**A float in a validity or transaction-time position was silently denominated one million
times too small, landing in 1970.** Validity timestamps are integer *microseconds* since the
epoch, while `now()` and `parse_timestamp()` return float *seconds* — and `Num::get_int`
accepted any integral float, coercing one into the other without a word. It is one bug,
reachable from four places: the `@` valid-time selector, the `@ (tt: …)` / `:as_of`
transaction-time selector, the `validity(...)` constructor, and — worst — the write path.

The write path is why this is a correctness release and not an ergonomics one. A `:put` of
`[parse_timestamp('2024-06-01T00:00:00Z'), true]` into a `Validity` column **succeeded** and
stamped the row at 1970-01-01T00:28:37Z — 1,717 seconds past the epoch. The row reads back
correctly on an ordinary query; the damage
is visible only under time travel, which is precisely where a bitemporal database is supposed
to be trustworthy. On the read side the failure is equally quiet: `@ parse_timestamp(…)`
returned **zero rows and no error**, because the misread always lands *before* any row was
asserted — indistinguishable from "no data yet". `@ 1e300` was accepted too, saturating to
`i64::MAX`, i.e. silently querying the end of time.

**The bug is inherited, and it is as old as Cozo's time travel.** Three of the four sites
are verbatim upstream code at the fork point (`481af05`, 2024-12-04): the `@` valid-time
selector (`expr2vld_spec`, `parse/query.rs`), the `validity(...)` constructor (`op_validity`,
`data/functions.rs`), and — worst, and byte-identical to upstream's — the write path, the
`DataValue::List` arm of `ColType::Validity` in `data/relation.rs`. So is the accessor all
three funnel through: `Num::get_int` (`data/value.rs`), which coerces any whole-numbered
float to an `i64`. Only the fourth site, the transaction-time selector (`expr2tt_spec`), is ours —
and it did not introduce the coercion, it inherited it: 0.10.0 extended the same accessor
onto a new axis. `Validity` columns and `@` time travel are a Cozo feature that predates the
fork by years. This is not something bitemporality broke; every CozoDB database with a
`Validity` column has it.

All four sites now reject a float and say what to write instead
(`parser::float_validity_spec`, `eval::float_validity`).

**Upgrade action — and note carefully *where* it bites.** The schema still compiles; it
is the next **write** that now fails. The idiom `Validity default [floor(now()), true]` —
and any spelling that yields a *whole-numbered* float, so `floor(now())`, `round(now())`,
or `parse_timestamp(...)` on a whole second — has been silently writing 1970 into your
valid-time axis. It now errors on write. (Bare `[now(), true]` already errored before this
release, but only by luck: `now()` returns a *fractional* float, and the coercion only ever
accepted whole-numbered ones.) Write instead:
```
last_seen: Validity default [to_int(now() * 1000000), true]
```

We found exactly one caller of the broken idiom anywhere: **upstream's own HNSW test**
(`runtime/tests.rs`), which we inherited unchanged and which is still in upstream today. It
had been writing 1970 into upstream's own valid-time axis for as long as the test has
existed. It never asserted on the value, so nothing ever went red. If it was in the test
suite we inherited, it is in someone's schema.

**What this does *not* fix, deliberately.** An integer in *seconds* (`@ 1704067200`) is still
accepted and still silently returns nothing. That is not an oversight: valid time is an
abstract, user-settable logical clock — the tutorial itself queries `@ 2019` — so no
magnitude check can tell a wrong-unit timestamp from a legitimate small one, and any such
check would be wrong. The real answer is a *typed* path (`dt_to_validity` + `@ <Validity>`),
which lands with the datetime library. Until then, integer microseconds remain the low-level
form, and the string forms (`@ '2024-06-01'`, `@ '2024-06-01T12:00:00Z'`) remain the safe ones.

Public Rust API is byte-identical. Guard: `cozo-core/tests/validity_units.rs` (14 tests; each
of the four sites has a test that goes red when that site alone is reverted).

## 0.12.1 — 2026-07-12

**Six correctness bugs inherited from upstream Cozo, and the repair path for the worst
of them.** None is a regression the fork introduced — every one predates the fork point,
which is rather the point: nobody was auditing this code. Two are silent failures a caller
cannot detect from the outside. `MultiTransaction::commit()` returned `Ok(())` for a commit
that had *failed* — and `cozo-bin`'s HTTP `/transact` endpoint sat directly on top of it,
answering `200 {"ok": true}` for transactions that never committed. And full-text postings
leaked on every in-place `:put` update of a relation carrying only an FTS index: ghost hits,
an index growing without bound, and BM25 statistics drifting far enough to measure a **55%
score error** on a two-document corpus.

**Upgrade action, if you use full-text search.** The write-path fix stops new leakage but
**cannot evict postings that are already written**. A relation with only an FTS index that
has ever been updated in place is affected *today*, and upgrading alone does not repair it.
Rebuild it once with the new `::reindex`:

```
::reindex my_relation
```

Engine (`cozo-core`) only, plus CI. No `cozorocks`/`mnestic-rocks` change. No planner or
query-plan change; the grammar gains exactly one system op (`::reindex`). Query results
change only on relations affected by the FTS leak, where they were wrong — see that entry.

### Added

- **`::reindex <relation>`** — rebuild a relation's HNSW / FTS / LSH indexes in
  place, from the index configuration the database already stores. It is the
  repair path for three separate problems that all reduced to one missing
  operation:
  - **the FTS posting leak fixed below** — the write-path fix stops new leakage
    but cannot evict postings already written, so every database that updated
    rows in place on an FTS-only relation needs this once;
  - **the bulk-load paths** (`import_relations`, `import_from_backup`), which do
    not maintain these indexes and used to tell you to "drop + recreate" — i.e.
    to reconstruct the original `::hnsw`/`::fts` creation script (extractor,
    tokenizer, filters, `ef_construction`, `m_neighbours`…) by hand from
    `::indices` output;
  - any index whose contents have drifted from its base relation.

  It runs in one write transaction (a crash rolls back to the intact old index;
  re-running is always the cure) and holds the relation's write lock for the
  duration — a maintenance operation, not an online one. Nothing auto-invokes
  it. A relation with no HNSW/FTS/LSH index is a loud no-op, not an error, so it
  stays scriptable across a set of relations.

  Each index is rebuilt against its **own stored manifest**, not a reconstructed
  config — which matters for LSH, whose manifest keeps the derived band geometry
  (`n_bands`, `n_rows_in_band`, `perms`) but not the weights that produced it. A
  drop-and-recreate would have silently recomputed that geometry from defaults
  and returned an index with a different recall profile than the one you asked
  for. *(The HNSW verify/repair pass for dangling keys — upstream #232 — is a
  separate, later piece of work; a full rebuild already cures that database.)*

### Fixed

- **FTS postings leaked when a row was updated in place** (inherited from
  upstream; affects every release through 0.12.0). Deletion of a row's old
  postings was gated on `has_indices` — which counts only *plain B-tree secondary
  indexes*. A relation carrying **only** an FTS index therefore never deleted the
  old document's postings on a `:put` over an existing key: terms the document no
  longer contained kept matching it, the index grew without bound, and the BM25
  `df`/`avgdl` statistics drifted (measured: a **55% score error** on a
  two-document corpus). `:rm` and the `update` op were unaffected — they always
  deleted correctly. `query/stored.rs`; guarded by
  `cozo-core/tests/fts_lsh_update_leak.rs`.

  **Results change on an affected relation** — which is the point of the fix:
  terms a document no longer contains stop matching it, and its BM25 scores move.
  There is deliberately no flag to restore the old behaviour; the old behaviour
  was a leak.

  **The fix stops new leakage; it does not evict postings already written.** If
  you have an FTS-only relation that has ever been updated in place, its index is
  affected today — rebuild it with `::reindex` (this release). *(Historical
  workaround, for the record: giving the relation any plain secondary index
  re-armed the correct deletion path.)*

  **LSH does not leak**, despite sitting behind the same gate: its write path is
  self-cleaning (it removes the row's old bands before writing new ones), so the
  gated deletion was only ever redundant for LSH. We first reported this as an
  "FTS/LSH" leak and are correcting that here.

- **`MultiTransaction::commit()` reported success for a failed commit.** It
  matched `Ok(_) => Ok(())` on the channel receive, discarding the
  `Result<NamedRows>` the transaction thread sends back — so a commit that
  errored returned `Ok(())` and the caller believed its data was durable. The
  HTTP server's `/transact` endpoint sat directly on top of this and answered
  `200 {"ok": true}` for failed transactions. `abort()` had the same shape.
  (`run_script` on the same type always propagated correctly.) `cozo-core/src/lib.rs`.

- **Change callbacks fired after a *failed* commit** (inherited from upstream).
  In the multi-statement transaction path (`run_multi_transaction`),
  `send_callbacks` ran unconditionally after the commit, so subscribers received
  `Put`/`Rm` events for rows that were never committed — anything syncing off the
  change feed (a search mirror, an audit log, a cache) could silently diverge
  from the database. `register_callback`'s own contract is "when the requested
  relation are *successfully committed*"; the single-statement path always
  honoured it. Callbacks now dispatch only on a successful commit; the abort path
  was already correct. `runtime/db.rs`; guarded by
  `cozo-core/tests/callback_commit_contract.rs`.

  *Known limitation, unchanged:* `send_callbacks` still dispatches synchronously
  on the committing thread over bounded channels, so a slow subscriber can stall
  writers.

- **The `newrocksdb` backend silently lost concurrent updates** (inherited from
  upstream; `--features storage-new-rocksdb`, non-default). It ran an
  `OptimisticTransactionDB` but discarded `for_update` on `get`/`exists` and the
  `write` flag on `transact`, so conflict validation never armed: two
  transactions could read a key, both write it, and **both commit** — one
  acknowledged write silently vanished. `for_update` reads now register the key
  via `get_for_update`, and writing transactions take a snapshot (matching the
  pessimistic RocksDB backend). `storage/newrocks.rs`; guarded by
  `cozo-core/tests/backend_transaction_contract.rs`, which reproduces the lost
  update against the unfixed backend.

- **The `sled` backend's `del()` never deleted** (upstream **#306**;
  `--features storage-sled`, non-default). It wrote `PUT_MARKER` where
  `DEL_MARKER` belonged, so a delete inside a transaction was recorded in the
  changes overlay as a put-with-empty-value: `exists` kept answering `true` and
  the commit re-inserted the key. Thanks to the issue reporter, who diagnosed
  this precisely — and wrote a fix and tests upstream never merged.
  `storage/sled.rs`. *(The issue's second half — `range_skip_scan_tuple` is
  unimplemented, so time travel does not work on sled — remains open.)*

  Both secondary backends now run their transaction-contract suite in CI; until
  0.12.1 nothing in CI compiled them.

- **`import_from_backup` silently stranded HNSW/FTS/LSH indexes** (inherited
  from upstream). It guarded only against *B-tree* indexes and then raw-put the
  source's KV rows straight into the store, so an operator restored a backup and
  hybrid retrieval quietly returned nothing for the restored rows — with no
  signal anywhere. It now warns, as its sibling `import_relations` already did.
  Neither path maintains those indexes (that is what makes bulk loading fast);
  the bug was doing it silently. Both warnings now point at **`::reindex`**
  instead of telling the user to drop and recreate the index by hand — which
  meant reconstructing the original `::hnsw`/`::fts` creation script (extractor,
  tokenizer, filters, `ef_construction`, `m_neighbours`…) from `::indices`
  output. The two call sites now share one helper so they cannot drift apart
  again. `runtime/db.rs`; guarded by
  `cozo-core/tests/import_index_staleness.rs`.

## 0.12.0 — 2026-07-11

**Budgeted weighted traversal — the context-fill primitive.** `BudgetedTraversal` is a new
`graph-algo` FixedRule: cheapest-first multi-seed expansion over non-negative weights under a
required global distinct-node budget (`max_nodes`), an optional cost ceiling (`max_cost`) and an
**exact** hop bound (`max_depth`, layered per-`(node, hops)` labels — never depth-pruned
Dijkstra), with an optional in-expansion admission gate (`*gate[node, …]` + `admit:`):

```
context[node, cost, parent, depth] <~ BudgetedTraversal(
    graph: 'knows', seeds[n], *live[uid, ok],
    admit: ok, max_nodes: 200, max_cost: 12.0)
```

Validated at the release's merge gate against the flagship consumer's production host-side
BFS (mindgraph-rs `traverse_reachable`) via a defined-equivalence oracle, green on
mem/sqlite/rocksdb through the generic `admit:` gate. Measured there (release build, RocksDB,
hub-degree-1000 fixtures at 4k nodes/21k edges and 20k/101k, budgets 100/500, gated,
`max_depth: 10`): one call over a cached `graph:` projection replaces the host's ~2·depth
engine round-trips and runs **2–4× faster** than that BFS; the positional form pays a per-call
O(live edges) scan + CSR build and is slower than the host at those scales — at scale,
maintain a derived cost relation and a projection (spec §10).

Engine (`cozo-core`) only; zero planner/grammar/projection changes — the call surface is the
existing fixed-rule syntax, and the `graph:` arm rides the 0.11.0 projection cache. No
`cozorocks`/`mnestic-rocks` change, and no existing query changes result (one new reserved
rule name behind `graph-algo`).

### Added

- **`BudgetedTraversal(edges | graph: 'G', seeds[node, initial_cost?], gate?, max_nodes: …)`** —
  emits the settled shortest-path-tree fragment `(node, cost, parent, depth)`: the `max_nodes`
  cheapest distinct admissible nodes reachable from the seed set, deterministic by construction
  (admission by `(cost, node)` total order; the `(parent, depth)` witness by strict lexicographic
  relaxation), form-independent (positional ≡ `graph:` byte-for-byte, tie rows included), and
  interruptible (`:timeout` / `::kill`, every-4096 mask). Costs accumulate in f64; weights are
  consumed *as costs* — monotone transforms like `−ln(weight)` are the caller's. An inadmissible
  node spends no budget and never bridges; gated `admit:` is existential over the node's gate
  rows; `max_cost` is finite-only; CSR-absent seeds emit as loose roots. Tests:
  `cozo-core/tests/budgeted_traversal.rs` (38, sqlite), incl. an 8-mutation discrimination run
  recorded in the spec. Spec:
  [`docs/specs/budgeted-traversal.md`](docs/specs/budgeted-traversal.md) §11.

### API and compatibility

- **The optional `rayon` dependency is now bounded `>=1.10, <1.11`** (`rayon` is pulled by the
  `graph-algo` and `rayon` features). rayon 1.11 breaks `graph_builder` 0.4.x — the CSR-builder
  crate behind `graph-algo` — whose `edges()` par-iter no longer compiles, so a downstream crate
  enabling `graph-algo` on a fresh resolve landed on a broken combination (mnestic's own
  committed lockfile had masked it). The bound makes every fresh resolve land on a working
  pair; it will be relaxed only after a verified fresh `graph-algo` build against a newer rayon
  (`cozo-core/Cargo.toml`).

## 0.11.1 — 2026-07-10

**Built-in skyline aggregates, reachable from every binding.** The registered
dominance aggregate `register_bounded_meet_aggr` (0.10.1) keeps a Pareto frontier
per group, but is a host-Rust closure — unreachable from the PyPI wheel,
`cozo-bin`, langchain or llama-index. `pareto_min` / `pareto_max` close that gap
for the case that covers almost every real skyline — componentwise dominance over
a numeric vector — as ordinary CozoScript aggregates callable through plain
`run_script` with no registration and no FFI.

Engine (`cozo-core`) only. The addition is two reserved aggregate names attached
in `parse_rule_head_arg` (`builtin_skyline_dominance`) plus a per-candidate
validator on the internal `DominanceMeetStore` (`builtin_skyline_validator`); the
store, eval and stratifier are untouched, the public `RegisteredBoundedMeet` is
byte-identical to 0.11.0, and no existing query changes result — a purely
additive, non-breaking patch. No `cozorocks`/`mnestic-rocks` change.

### Added

- **Built-in skyline aggregates `pareto_min` / `pareto_max`.** They keep, per
  group, the Pareto frontier of a numeric vector — the points not dominated by
  any other, `pareto_min` treating smaller as better and `pareto_max` larger:

  ```
  ?[frontier] := offer[price, quality], v = [price, -quality], frontier = pareto_min(v)
  ```

  Use them to surface a *contested set* — several answers none of which beats
  another — instead of collapsing to a single winner. Mixed objectives (minimize
  price, maximize quality) are expressed by negating the maximized components, as
  above. The dominance is native — the product order, a provable strict partial
  order — so unlike a registered `antichain` these need no host registration and
  are reachable from **every** binding (the PyPI wheel, `cozo-bin`, langchain,
  llama-index) through plain `run_script`, and they stay correct in a release
  build where the debug order-law probes are compiled out. They work in recursive
  rules and inherit the confluence and cycle-pruning of the registered dominance
  aggregate they build on. A malformed operand (non-list, non-numeric component,
  NaN, or empty vector) is a loud error; two vectors of differing length are
  treated as incomparable (both survive), since arity mismatch is not detectable
  per candidate. There is no cap — the frontier is bounded by the group's own
  tuple count, like `collect`, so keep the objective vector low-dimensional
  (skyline cardinality grows with dimensionality). Tests:
  `cozo-core/tests/pareto_skyline.rs` (9, sqlite). Spec:
  [`docs/specs/antichain-bounded-meet.md`](docs/specs/antichain-bounded-meet.md) §10.

## 0.11.0 — 2026-07-10

**Cached graph projections.** `::graph create G { edges: knows, nodes: person }`
names an in-memory adjacency over stored relations that twelve graph algorithms
reuse across queries instead of rebuilding on every call:

```
::graph create g { edges: knows, nodes: person }

?[node, group] <~ ConnectedComponents(graph: 'g')
?[node, rank]  <~ PageRank(graph: 'g', iterations: 20)
```

It is **always fresh**. A projection never serves a transaction data that differs
from what that transaction's own scan of the sources would return — in either
direction, so a long-lived reader is not handed an entry newer than its snapshot
either. Writing to a source frees the adjacencies built from it. Under continuous
write churn the cache degrades to build-per-query; it never goes stale.

Measured on a 400,000-edge graph (release build; *cold* is the positional form,
i.e. today's behaviour):

| kernel | cold | warm | |
|---|---|---|---|
| `ConnectedComponents` | 127 ms | 7.9 ms | **16×** |
| `PageRank`, 20 iterations | 150 ms | 10 ms | **15×** |
| `ClusteringCoefficients` | 169 ms | 56 ms | 3× |

What is cached is the *setup* — scanning the edge relation and building the CSR —
so the gain shrinks as the kernel itself dominates: `ClusteringCoefficients`
spends its time counting triangles, and `CommunityDetectionLouvain` rebuilds a
coarsened graph per level internally, so only its first build is reused.

A database that defines no projections pays one atomic load per transaction, one
set insert per mutation *statement* (not per row), and one uncontended mutex
acquisition per *writing* commit. Read-only commits touch none of it.

### Added

- **`::graph create NAME { edges: R, nodes: R2 }`** — `nodes` optional; sources
  may be written bare or quoted. Validated on the spot: the relations must exist
  and `edges` must have arity ≥ 2. Index, temporary and transaction-time
  relations are refused with an explanation. Nothing is built at create time;
  adjacencies materialise per `(direction, weighted)` variant on first use.
- **`::graph drop NAME`** frees the definition and every adjacency built from it.
- **`::graph list`** reports one row per built variant —
  `name, edges, nodes, variant, est_bytes, built_at, last_used` — and one
  null-variant row for a projection that has built nothing yet.
- **`graph: 'G'` on twelve algorithms**, in place of their positional edge
  relation: `ConnectedComponents`, `StronglyConnectedComponents` / `SCC`,
  `PageRank`, `ClusteringCoefficients`, `TopSort`, `BetweennessCentrality`,
  `ClosenessCentrality`, `ShortestPathDijkstra`, `KShortestPathYen`,
  `MinimumSpanningTreePrim`, `MinimumSpanningForestKruskal`, `LabelPropagation`,
  `CommunityDetectionLouvain`. Their remaining positional inputs shift down by
  one, and optional trailing inputs stay optional.
- **`nodes:` makes isolated vertices real.** A vertex named by the node relation
  but by no edge becomes a genuine degree-0 vertex: `PageRank` counts it in `N`
  and ranks it, `ConnectedComponents` emits it as its own component. The vertex
  set is the union of the two relations.
- **A 512 MiB memory ceiling**, enforced on the spot by least-recently-used
  eviction. Set it with `Db::set_graph_projection_capacity(bytes)`,
  `DbInstance::set_graph_projection_capacity(bytes)`, or, from Python,
  `CozoDbPy.set_graph_projection_capacity(bytes)`. `0` turns caching off while
  leaving `::graph create`/`list`/`drop` working. A single variant larger than
  the whole ceiling is built for each query, with a warning.
- **Concurrent cold callers coalesce into one build** per variant rather than
  each building their own.
- **`PageRank` accepts an optional second input naming the vertices**, as
  `ConnectedComponents` already did. Passing it changes the ranks: isolated
  vertices enter `N`, so every rank moves. On LDBC SNB sf1, 694 of 10,620
  persons have no `knows` edge and were previously absent from the ranking
  altogether. Single-input queries are unaffected.

### Changed

- **BREAKING (results): `PageRank`'s default `iterations` is now 20, up from
  10.** Ten was a below-upstream override — the `graph` crate's own default is
  20 — and it is measurably non-convergent: on sf1 the ranks still move by
  `2.1e-4` at iteration 10 against the default `epsilon` of `1e-4`, versus
  `1.6e-6` at iteration 20. **Pass `iterations: 10` to restore the old numbers.**
- **`PageRank` now warns** (`log::warn`) when the `iterations` cap stops it short
  of `epsilon`, instead of silently returning unconverged ranks. `epsilon: 0.0`
  means "run exactly `iterations`" and stays quiet.
- **A `graph:` option on a rule that cannot consume one is now a parse error.**
  Unknown fixed-rule options are ignored engine-wide, so this would otherwise
  silently rebuild the graph the slow way. `BFS`, `DFS`, `ShortestPathBFS`,
  `RandomWalk`, `ShortestPathAStar` and `DegreeCentrality` are excluded by
  design: they evaluate per-tuple condition, heuristic and weight expressions
  against the edge relation, which a compressed adjacency does not carry.
  **If you registered a custom `FixedRule` that reads an option named `graph`,
  override the new `supports_projection()` to return `true`.** `SimpleFixedRule`
  is exempt automatically — it forwards every option to its closure.
- **`PageRank` rejects a positional nodes relation combined with `graph:`** — a
  cached variant's vertex ids are fixed when it is built. Declare them on the
  projection with `nodes:`. `ConnectedComponents` still takes a positional nodes
  relation, as an overlay on top of the projection.
- **`ConnectedComponents` and `SCC` group ids renumber** when a projection
  declares `nodes:` — isolated vertices interleave rather than being appended.
  The partition is unchanged, and the labels always were arbitrary.
- **`MinimumSpanningTreePrim` starts from the lowest vertex with an edge**, not
  vertex 0, so a nodes-bearing projection whose first vertex is isolated still
  spans a tree. A supplied starting relation is unaffected, diagnostics included.
- Errors raised while scanning edges (`algo::not_an_edge`,
  `algo::invalid_edge_weight`) now surface eagerly rather than after the graph
  has been built.

### Fixed

- **An empty edge relation panicked seven graph algorithms**, aborting the
  process: `TopSort`, `ConnectedComponents`, `StronglyConnectedComponents`,
  `ClusteringCoefficients`, `BetweennessCentrality`, `ClosenessCentrality` and
  `LabelPropagation`. They now return no rows, as `PageRank` already did. The
  builder sizes a graph from its largest edge endpoint, and `max` over an empty
  edge list is `0`, so an empty relation produced a *one-vertex* graph with an
  empty id map and the first index aborted. Four of these algorithms already
  carried a `node_count() == 0` guard for exactly this case; it could never
  fire. Present in upstream CozoDB.
- **`multi_transaction` could deadlock the process.** It ran its transaction
  loop on a `rayon` global-pool worker, which the loop then parked in a blocking
  receive for the transaction's whole life. With as many open transactions as
  the pool has workers — one, on a single-core host or under a CPU quota — every
  parallel query in the process blocked forever. It now uses a dedicated thread,
  as its documentation always claimed. Affects every caller, not just graph
  algorithms.
- **An oversized graph wrapped silently.** The CSR indexes vertices and edge
  offsets with `u32`; both counts are now checked before the build
  (`algo::graph_too_large`).
- **Long CSR builds were un-interruptible.** The scan now checks the poison flag
  every 4096 tuples, so `::kill` and `:timeout` abort a large build instead of
  waiting it out. Previously the flag was only observed once the algorithm
  proper began, which on a 300k-edge graph was most of the query.

### API and compatibility

- New public types under `graph-algo`: `VariantSpec`, `VariantKey`,
  `GraphVariant`, `ProjectionVariant`, `GraphSource`, and
  `FixedRulePayload::graph_input`.
- New defaulted trait method `FixedRule::supports_projection` — non-breaking for
  out-of-tree implementations; see the parse-error note above.
- New on `FixedRuleInputRelation`: `as_directed_graph_checked` and
  `as_directed_weighted_graph_checked`, which take an optional node relation and
  a `Poison`. The existing `as_directed_graph` / `as_directed_weighted_graph` are
  unchanged and now delegate to them.
- **`graph` is now a public dependency.** Its adjacency types are re-exported
  under `graph-algo` (`DirectedCsrGraph`, `Graph`, `DirectedNeighbors`,
  `DirectedNeighborsWithValues`), so moving off `graph = "0.3"` is a
  semver-major event for mnestic.
- New internal `test-hooks` feature exposing `Db::set_commit_fence_for_tests`,
  `Db::set_graph_build_fence_for_tests` and
  `Db::graph_projection_builds_for_tests`, which let a test park a transaction
  inside the freshness protocol's race windows. **Not a supported API**; never
  enable it in production.

### Known limitations

- **Projections are not persisted.** After a restart, using one is a loud error
  naming the fix; re-create them at startup.
- **Transaction-time relations cannot be projection sources.** Everywhere else
  in the engine, a selector-less read of a tt-stamped relation means its
  *current belief*, while a projection's raw scan would deliver the whole
  history keyspace, retracted rows included. Plain `Validity` relations project
  fine. Current-belief projections of tt relations may come later.
- **Weighted variants are built permissively**, so one adjacency serves every
  consumer. An algorithm whose result is undefined under negative weights —
  `ShortestPathDijkstra`, `KShortestPathYen`, `CommunityDetectionLouvain`,
  `BetweennessCentrality`, `ClosenessCentrality` — fails loudly if it meets one,
  naming itself and the projection.

Design and correctness argument: [`docs/specs/graph-projection.md`](docs/specs/graph-projection.md)
and the module docs in `cozo-core/src/runtime/graph_projection.rs`. Proven by 58
end-to-end tests, 6 RocksDB interleaving tests that drive the protocol's race
windows directly, and 37 mutations verified to turn those tests red.


## 0.10.7 — 2026-07-08

A plan-quality fix for the 0.10.5 greedy join reorder, plus a Python-facing
binding addition and a docs note. The headline corrects a tie-break that could
demote a full-key filter to a partial-key expansion — pulling a high-fan-out
edge ahead of a selective atom and producing a strictly worse plan on
high-fan-out cyclic joins. Engine (`cozo-core`) plus the Python binding; no
`cozorocks`/`mnestic-rocks` change and **no query-result change** (join reorder
is result-invariant under set semantics).

- **Fix: the greedy join reorder no longer demotes a full-key filter to a
  partial-key expansion.** The greedy tie-break shipped in 0.10.5
  (`query/reorder.rs`) rewarded any atom whose *leading* composite-key column was
  bound. On a `knows{src, dst}`-style composite key, binding only `src` scored as
  if it were a point lookup — but it is a keyed *expansion* over every neighbour
  of `src` (the highest-fan-out relation in a graph), not a filter. So the pass
  could pull a fan-out `knows` edge ahead of a selective membership atom and
  produce a strictly worse plan: an external benchmarker (LDBC-SNB LSQB) measured
  Q3 — a same-country `knows` triangle — go from ~19 s to a >120 s timeout at
  SF0.1, while the tie-break won nothing across the other eight queries at two
  scale factors. The tie-break helper is renamed
  `bound_key_prefix_len` → `full_key_lookup_bonus`: it now rewards ONLY a
  *complete*-key point lookup (all key columns bound — an existence filter that
  matches ≤1 tuple and cannot increase cardinality) and scores a *partial* prefix
  `0`, so a partial-key tie falls back to the written order. **No result change**
  (conjunction is commutative under set semantics), and 0.10.5's ~54.5×
  "min-new-vars" win is preserved — it is driven by the new-vars criterion, not
  this tie-break. Regression-guarded by a new high-fan-out integration test
  (`tests/join_reorder.rs::partial_key_prefix_not_pulled_forward_{sqlite,mem}`,
  which FAILS on the pre-fix engine) plus rewritten unit tests
  (`greedy_prefers_full_key_lookup_on_tie`,
  `greedy_ignores_partial_key_prefix_on_tie`). Anyone relying on the default
  `:reorder greedy` over high-fan-out cyclic joins benefits.
- **Python binding: `set_query_factorization` / `query_factorization` are now
  exposed on `CozoDbPy`.** The 0.10.5 factorized-`count()` rewrite shipped behind
  a Db-wide kill switch (`Db::set_query_factorization(bool)`, default OFF "to
  soak") that was reachable only from Rust `DbInstance`. Python callers can now
  toggle it exactly like the timeout methods that already crossed:
  `db.set_query_factorization(True)` and `db.query_factorization()` (returns the
  current state). Purely additive; the default stays OFF. This lets Python-based
  benchmarks generate the soak evidence the 0.10.5 changelog said default-on is
  waiting on. (`cozo-lib-python/src/lib.rs`.)
- **Docs: an "algebra ⟷ fixed-rule map"** in
  `docs/concepts/semirings-and-fixedrules.md` records which built-in graph
  fixed-rules are expressible as semiring recursion
  (`ShortestPathDijkstra`/`ConnectedComponents` validated identical to `min_cost`
  / min-label forms) and corrects two easy conflations (community
  `LabelPropagation` is NOT a meet; `KShortestPathYen` ≈ `min_cost_k` only
  approximately).

## 0.10.6 — 2026-07-08

An urgent upgrade-safety patch. The headline fixes a data-availability
regression the 0.10.0 bitemporality work introduced: relation catalogs last
written before 0.10.0 (or by an index/rename/destroy path) could fail to open,
taking the whole database down — **anyone who upgraded a pre-0.10.0 database to
any of 0.10.0–0.10.5 should upgrade to 0.10.6.** Two internal items ride along.
Engine (`cozo-core`) plus Python-wheel CI; no `cozorocks`/`mnestic-rocks` change
and no query-behavior change.

- **Fix: relation catalogs written before 0.10.0 no longer fail to open
  ("Cannot deserialize relation metadata from bytes").** The bitemporality
  work (0.10.0) inserted `RelationHandle::tt_gc_floor` *mid-struct*. rmp_serde
  encodes structs positionally on the pre-`with_struct_map` catalog-write
  paths, and `#[serde(default)]` only rescues a *missing trailing* element — so
  every relation whose catalog was last written as a 13-field positional array
  (any graph created before 0.10.0, or updated by an index/rename/destroy path)
  failed to deserialize on open, taking the whole database down. This silently
  took down a production multi-tenant deployment on its 0.10.0 upgrade. Two-part
  fix: (1) `tt_gc_floor` moved to the **last** field of `RelationHandle` so the
  trailing default applies to legacy arrays; (2) the seven catalog-rewrite paths
  (`::index`/HNSW/FTS/LSH create, relation rename, index destroy) now serialize
  with `.with_struct_map()` like the create path, so catalogs are uniformly
  self-describing maps and future field additions can't reintroduce this class
  of bug. No migration: legacy arrays stay readable, and re-canonicalize to
  maps on their next write. Regression-guarded by a real pre-0.10.0 `edge`
  catalog fixture (`runtime/relation.rs::catalog_compat_tests`).
- **Greedy join reorder is now a pure function over a resolved `SchemaView`.**
  Internal refactor of the deterministic join-reorder pass shipped in 0.10.5
  (`query/reorder.rs`): the reorder no longer reads mutable planner state,
  making it independently unit-testable. No query-plan or behavior change.
- **Python wheel CI hardened for `storage-rocksdb`.** The x86_64 manylinux leg
  now builds on `manylinux_2_28` and installs `libclang` (`clang-devel`) so
  zstd-sys's bindgen resolves; the aarch64, macOS and Windows legs are
  unaffected. Wheel-build only — no engine change.

## 0.10.5 — 2026-07-07

A liveness + performance release responding to an external
Ladybug-vs-mnestic benchmark. Two themes: queries you can always stop
(interruptible `::kill`/`:timeout` + a per-query wall-clock budget), and
naively-ordered queries that stop being pathological (a deterministic join
reorder + an opt-in factorized-count rewrite). Engine (`cozo-core`) plus the
Python binding and its wheel CI; no `cozorocks`/`mnestic-rocks` change.

- **`::kill` and `:timeout` now actually interrupt a running query.** Two
  defects the benchmark surfaced: (1) `::running`/`::kill` opened a storage
  transaction before dispatching, so on the mem/sqlite backends a `::kill`
  queued behind the very read query it was trying to kill and blocked for that
  query's entire remaining runtime — they now dispatch before any transaction
  (they touch only the in-memory running-query registry). (2) The poison flag
  was only checked between rule applications, never inside the relational-algebra
  enumeration, so a naive single-rule join that yields no output for a long time
  went uninterruptible — the per-query `Poison` is now threaded through
  `RelAlgebra::iter` and checked every 4096 pulls at every operator boundary and
  raw scan (the signature change makes coverage compiler-enforced). Overhead is
  within noise on the `point_lookup` baseline.

- **Per-query wall-clock budget.** A query can carry a deadline three ways —
  the in-script `:timeout <secs>` option, a per-call
  `Db::run_script_with_options(payload, params, mutability, ScriptRunOptions { timeout })`,
  and a Db-wide default `Db::set_default_query_timeout(Option<f64>)`. The
  effective deadline is the **minimum** of whichever are set, computed once per
  `run_script` call and shared across every statement of an imperative/multi-
  statement script and any triggers it fires — a `:timeout` (or per-call
  timeout) can only *tighten* the budget, never extend past the Db default.
  Expiry raises a distinct **`eval::timeout`** diagnostic (a `::kill` still
  raises `eval::killed`). `Poison` gained an `Option<Instant>` deadline and lost
  its per-timed-query detached timer thread — no more thread leak — riding the
  interruptibility fix's 4096-pull check cadence so it aborts promptly inside
  long enumerations. Absurd or non-finite budgets (`:timeout 1e300`, an HTTP
  `timeout` of `1e400`) clamp to "no deadline" (still `::kill`-able) instead of
  panicking. On wasm, which has no `std` monotonic clock, a query carries no
  wall-clock budget rather than panicking on `Instant::now()`. Exposed on
  `DbInstance` and in cozo-bin (`timeout` field on the HTTP query payload;
  `--default-query-timeout` CLI flag). A budget-aborted mutable script rolls
  back with no partial commit. **In-tree breaking:** `Poison` is now a
  named-field struct — use `Poison::check()`/`Default`, not `.0`.

- **Deterministic greedy join reorder** (default ON; `:reorder written` opts
  out; spec `docs/specs/join-reorder.md`). No pass considered join order, so a
  naively-ordered conjunction — exactly what an LLM agent authors — could spin
  on an N³ intermediate (the benchmark's members-first same-group triangle). A
  stat-free min-new-vars greedy pre-pass (`query/reorder.rs`, after the #1
  equality-pushdown) reorders the positive relation atoms of an eligible
  conjunction (measured 54.5× on the repro; N³→N²). Eligible = ≥3 stored
  `Relation` atoms, no rule-application or Hnsw/Fts/Lsh atom, no multi-valued
  `in`-unification, and not a bare `:limit` without `:sort`. Results are
  unchanged: conjunction is commutative under set semantics, the binding-before-
  use pass remains the correctness arbiter (with fallback to written order if a
  reordered body fails to compile), and the pass is the **identity** on any
  stepwise-greedy-consistent written order, so hand-tuned plans stay byte-
  identical. The multi-valued-`in` exclusion is load-bearing — that construct is
  a multiplicity injector (generator vs. filter by position) that would
  otherwise silently change a non-idempotent aggregation (`count`/`sum`/
  `collect`). A residual Cartesian step (a genuinely disconnected conjunction)
  is `log::warn!`-ed and annotated `<op> (cartesian)` in `::explain`.

- **Automatic factorized `count()` rewrite** (opt-in, **default OFF**; spec
  `docs/specs/cardinality-algebra.md`). `count()` over a join streams every
  match (O(#matches)); the benchmark measured 4–342× vs an optimizer that counts
  without enumerating. A normal-form pre-pass (`query/factorize.rs`) rewrites an
  eligible single-clause `count()`-over-positive-join into Yannakakis-style
  per-key counting sub-rules — a bit-identical (exact-i64, `Int`-typed) answer
  computed without materializing the join. It fires only on shapes it can prove
  exact (declining cyclic, non-free-connex, repeated-variable, negated,
  recursive, bitemporal, `count_unique`/mixed-aggregate, and any body with a
  `!=` predicate) and
  accumulates in an internal `int_sum_prod` aggregate that **errors, never
  wraps,** on overflow. Behind a Db kill switch `set_query_factorization(bool)`,
  default OFF this release to soak before any default-on consideration; verified
  by a 400-case differential harness (naive vs factorized, mem + sqlite, exact
  row+type equality). An always-on companion detector surfaces a factorization
  advisory in `::explain` / `log::info!`.

- **Bulk `import_relations` into an index-bearing relation now warns.** The bulk
  path maintains B-tree secondary indexes but not HNSW/FTS/LSH, so imported rows
  silently stay invisible to vector/text search until the index is rebuilt — a
  `log::warn!` now flags it (a warning, not a hard error: consumers legitimately
  import a snapshot then reindex).

- **RocksDB in the PyPI `mnestic` wheel.** `CozoDbPy("rocksdb", path)` failed
  from `pip install mnestic` because the wheel build pinned `features=compact` —
  the binding already forwarded the engine string, so this was purely a build-
  feature gap. Wheels now compile `storage-rocksdb` on all five platform legs
  (feature list moved onto the maturin CLI as a per-leg matrix knob; mac legs
  pin `MACOSX_DEPLOYMENT_TARGET=11.0`), gated by a new `test-wheels` smoke job.
  The sdist stays compact (no rocksdb sources), so the persist engine is
  wheel-only.

- **Python binding: interior-mutable `CozoDbPy`.** `close()` took `&mut self`
  while an in-flight `run_script(&self)` held a shared PyCell borrow, so a
  concurrent `close()` raised "Already borrowed". The handle is now
  `RwLock<Option<DbInstance>>`; every method takes `&self` and reads clone the
  Arc-backed `DbInstance` out of a momentary guard, released before the blocking
  engine call. `run_script` gains an optional `timeout=None` kwarg plus
  `set_default_query_timeout`/`default_query_timeout`.

- **New spec: `docs/specs/cardinality-algebra.md`** — the manual factorized-
  counting authoring patterns (join-tree count DP, star product, `!=` and
  anti-join inclusion-exclusion) with their exactness conditions, the reference
  for shapes the automatic rewrite declines.

## 0.10.1 — 2026-07-06

A small additive release on top of 0.10.0 — two new query primitives (the
antichain/skyline aggregate and interval predicates) plus a correctness fix to
the `bit_and`/`bit_or` meet aggregates. Pure `cozo-core`; no
`cozorocks`/`mnestic-rocks` change.

- **Follow-up review fixes (0.10.0 hardening)**: (1) `bit_and`/`bit_or` meet
  aggregates now report whether the value actually CHANGED — the byte loop
  returned true unconditionally, the same defect family as the 0.10.0
  `and`/`or` inverted-changed-bit fix, so a non-changing fold (e.g.
  `[0xF0] & [0xFF]`) re-entered the semi-naive delta every epoch. (2) The
  bounded-meet divergence cap counts TOTAL changed epochs instead of a
  consecutive streak that reset on quiet epochs. Behaviorally identical
  today: the stratifier poisons every cross-rule edge into or out of an
  aggregated rule except direct self-recursion (in-SCC poisoned edges are
  rejected as unstratifiable — pinned by
  `bounded_meet_relay_recursion_unstratifiable` — and cross-SCC ones are
  forced across a stratum boundary) and exempts aggregated rules from
  magic-set rewriting, so a bounded rule's only in-stratum input is its
  own delta and its changed epochs form a contiguous prefix. The total
  count keeps the guard sound if the isolation is ever relaxed — a
  displacement cycle improving the k-set only every other epoch evades a
  resetting streak forever. (3) `bit_and`/`bit_or` `init_val` documented as
  a lazy identity: the true ⊕-identity (all-ones for `bit_and`, all-zeros
  for `bit_or`) has runtime-determined width, so empty bytes is a seed
  sentinel consumed by `update`'s first-contact branch, not the algebra's
  identity element.

- **Dominance bounded-meet registration — the antichain / skyline aggregate**
  (spec: `docs/specs/antichain-bounded-meet.md`, signed off + implemented
  2026-07-04): `register_bounded_meet_aggr(name, dominates, max_survivors)`
  opens the bounded-meet category (R1's recorded deferred item) to a
  host-registered strict partial order. The head form `name(operand)` keeps,
  per group, the non-dominated set of operands — each survivor its own
  output row — riding `AggrKind::BoundedMeet`, the stratifier permit, and
  the 4096-epoch divergence cap exactly like `min_cost_k`. New
  `DominanceMeetStore` (BNL in-buffer insert: structural-equality dedup,
  reject-if-dominated, multi-removal eviction; survivors kept in memcmp
  order so output is canonical, not arrival-dependent; equality-dedup-only
  delta twin). `max_survivors` is a mandatory resource guard — overflow is
  a loud error, never a silent truncation (an antichain has no canonical
  k-subset). Debug builds probe irreflexivity + asymmetry of the registered
  closure; call-site args are rejected at parse time (the cap lives in the
  registration). Trigger scripts keep rejecting custom aggregates at
  `::set_triggers`. Name reservation now also covers builtin FUNCTION names,
  retrofitted onto `register_custom_aggr` (one token, two semantics — the
  `coalesce` lesson). Rust-embedded-only v1 (host closures do not cross the
  Python/served surfaces; see the spec §5). Pinned by the §6 matrix in
  `cozo-core/tests/antichain_bounded_meet.rs` — Pareto frontier, recursion
  with a cycle, permuted-input confluence, cap overflow, lawless-closure
  probes, registration policy, persistence-without-registration.

- **Interval primitives** (spec: `docs/specs/cozoscript-extensions.md` §3.4 v1):
  `interval_overlaps(a, b)` builtin function and `interval_coalesce(span)`
  aggregate over half-open `[start, end)` list intervals. Deliberately plain
  list utilities decoupled from the vt axis (point-event `Validity` storage is
  unchanged). Touching spans do not overlap but do coalesce
  (`[0,5)` + `[5,10)` = `[0,10)`); empty spans `[x, x)` overlap nothing;
  mixed int/float bounds compare numerically (not by `Num`'s storage order);
  malformed spans (start > end, non-numeric or NaN bounds, non-list operands)
  are loud errors, never silent falses.
  `interval_coalesce` is the spec-mandated rename away from the original
  `coalesce` proposal, which silently collides with the shipped null-coalescing
  builtin / `~` operator. Pinned by `cozo-core/tests/spec_doc_validation.rs`,
  which also pins every validated listing in the companion spec.

## 0.10.0 — 2026-07-04
The release has two pillars, in order of impact:

1. **Bitemporality** — engine-assigned transaction time alongside Cozo's
   valid time. System-versioned (`tt: TxTime`) and fully bitemporal
   (`vld: Validity, tt: TxTime`) relations: a crash-safe monotone commit
   clock stamps every write; reads default to the current belief and
   time-travel with `@ (vt: …, tt: …)` or `:as_of`; existence-checking
   writes target the resolved current belief; `::history` /
   `::history_gc` (persisted floor) / `::evict` (audited hard deletion)
   manage the record's lifecycle; and a measured performance gate keeps
   current-belief reads within ~4–12% of the single-axis baseline.
   "What did we believe at time T about period Y" — in-engine.
   Spec: `docs/specs/bitemporality.md`.
2. **Provenance semirings** — the same recursive rules compute existence,
   cost, confidence, or evidence: `register_custom_aggr` for user-defined
   absorptive combines in recursion; `min_cost_k` bounded-meet aggregates
   returning the k best derivations per answer WITH the evidence chains
   that justify them; annotation persistence (resolved: no format change
   needed); and `:reconcile` — recompute-based belief revision that keeps
   derived annotations consistent under base-fact retraction, composed
   with the tt axis. Spec: `docs/specs/provenance-semirings.md`.

Plus **four upstream CozoDB bugs fixed** along the way: the inverted
changed-bit in `and`/`or` meet aggregates, a panic on negated validity
atoms, wrong answers from prefix-truncated temporal-column joins, and the
braced-`%return` imperative parse panic.

Detailed entries below, pillar by pillar.

### New — transaction-time commit clock (bitemporality step 2; internal, no user surface)
- The engine-assigned transaction-time (tt) allocator for the bitemporality
  work (`docs/specs/bitemporality.md` §5, decisions §13.10): a wall-clock-floored
  strictly monotonic commit counter `tt = max(now_µs, last_tt + 1)` held as an
  `AtomicI64` high-water mark on `Db` (`runtime/tt_clock.rs`), seeded at open
  from `max(persisted mark, wall clock)` and persisted as a system key
  (`[Null, "TT_HWM"]` under `RelationId::SYSTEM`, the `STORAGE_VERSION` idiom)
  **inside the committing transaction** — no crash window between advancing and
  persisting. `SessionTx::commit_tx_with_tt` allocates under a per-`Db`
  critical section so tt order == commit order == visibility order; values
  advanced by transactions that abort are burned, preserving monotonicity.
  Hardened by adversarial review: the HWM put goes through a new
  `StoreTx::put_externally_serialized` (RocksDB override clears the
  pessimistic transaction's begin-snapshot for this one key — otherwise any
  two temporally-overlapping tt commits would make the later one abort with
  `Resource busy`, the 0.8.4 `avgdl` hot-key failure mode; default impl is a
  plain put for sqlite/mem, whose storage-level write locks preclude
  overlapping write transactions). Seeding is monotone (`fetch_max`, so
  re-`initialize` and the step-3 restore re-seed can never regress the
  authority); a malformed persisted mark refuses to open (loud, mirroring the
  version-mismatch bail) instead of silently degrading to wall-clock seeding;
  the commit section recovers from mutex poisoning (sound: a mid-section
  panic only burns a never-committed value); the wall-clock read is
  cfg-guarded for wasm32 like `current_validity`. Nothing in the write path
  calls the clock yet — step 3 (schema opt-in + buffered stamping) routes
  tt-stamped commits through it, and **owes**: the restore/import re-seed
  (`max(persisted, max restored tt, wall clock)` after `restore_backup`) and
  the HWM+data-rows same-tx atomicity test. Pinned by tests: same-µs strict
  monotonicity, backward-clock step, concurrent per-caller monotonicity +
  global uniqueness, restart re-seeding from the persisted mark (sqlite),
  corrupt-mark open refusal, abort-doesn't-persist, mem-backend operation,
  and overlapping-commit non-conflict on RocksDB.
### New — TxTime relations: schema opt-in + write path (bitemporality step 3)
- **`TxTime` column type** (`docs/specs/bitemporality.md` §4): a relation whose
  last key column is `tt: TxTime` is transaction-time-stamped — tt-only
  (`{k, tt: TxTime => v}`, system-versioned) or bitemporal
  (`{k, v: Validity, tt: TxTime => x}`). The engine stamps every write with the
  commit clock's tt at commit; user-supplied tt values are rejected on every
  path (put specs, `:create`-with-rows headers, imports). **Relations declaring
  `TxTime` are unreadable by pre-fork/upstream builds.**
- **Temporal-axis rule enforced at `:create`** — axes are the trailing key
  columns, vt-then-tt, at most one of each; seven malformed shapes are
  `:create`-time errors whose message contains the copy-pasteable corrected
  declaration (deliberately stricter than `Validity`'s shipped query-time-only
  check).
- **Buffered commit-time stamping**: writes to tt-stamped relations buffer on
  the transaction and are stamped + written inside the commit critical section,
  so the whole transaction shares one tt (one belief event) and tt order ==
  commit order. Consequences, all pinned by tests: same-transaction writes are
  **not visible to later reads in the same script**; a key cannot be both
  asserted and retracted in one transaction (unbreakable resolution tie —
  rejected across statements, including tt-only `:put`+`:rm` mixes); double-puts
  of one (key [, vt]) in one transaction collapse last-write-wins; `::remove` of
  a relation with pending tt writes in the same script is rejected.
- **`:rm` on tt-only relations appends a retraction** at commit-tt (never a
  physical delete); values snapshot the key's latest row at statement time;
  `:rm` of a missing key is a no-op; `:delete` of a missing or believed-deleted
  key fails its existence assertion. **Deviations recorded:** on *bitemporal*
  relations `:rm` is rejected in v1 (removal is a valid-time statement — use
  `:put` with vt `"RETRACT"`; the spec-§6 remap is deferred to step 4);
  `:insert`/`:update`/`:ensure`/`:ensure_not` are rejected pending the as-of
  read path (step 4); `:replace` of an existing TxTime relation is rejected
  (destroy-and-recreate would silently drop history); triggers, secondary and
  search indexes, callbacks, and `:returning` are rejected on TxTime relations
  (indexes: step 5). A put-trigger on a *plain* relation may write into a
  TxTime relation (buffers and stamps normally; pinned).
- **Imports**: `import_relations` stamps **one tt per batch** (one belief
  event; spec §13.3), rejects tt columns in payload headers and delete-imports;
  `restore_backup` preserves tt bytes verbatim and **re-seeds the commit
  clock** from the backup's persisted mark; `import_from_backup` is rejected
  for TxTime relations (its rows would carry a foreign clock's tts — restore
  the full store or re-ingest). Export→import therefore does not round-trip
  TxTime relations; use backup/restore.
- **Interim read behavior**: bare scans work (all versions, newest tt first —
  the tt column decodes as a Validity value; use `:order` for chronological
  output); any `@` as-of read on a TxTime relation fails with a distinct
  "lands in step 4" error, so vt semantics never silently apply to the tt axis.
- Closes step 2's owed items: the restore re-seed and the HWM+rows same-tx
  atomicity test (an aborted transaction leaves neither rows nor a persisted
  mark). 24 txtime/tt_clock tests; hardened against a three-lens adversarial
  review (cross-statement conflict detection, `::remove`-with-pending-writes
  orphan bytes, believed-deleted `:delete` consistency were review catches).

### New — system-versioned relations complete: tt-only reads (bitemporality step 4a)
- **The labeled temporal selector** `@ (vt: …)` / `@ (tt: …)` / order-free
  `@ (vt: …, tt: …)` parses on every relation-access form; bare `@ E` still
  means valid time, everywhere, forever. tt tokens: `"NOW"`/`"END"` =
  end-of-tt-time (current belief — deliberately not the wall clock), numeric
  µs, ISO-8601 (bare dates now accepted as midnight UTC — on the vt axis too;
  strict RFC3339 previously rejected the docs' own `@ "2026-01-01"` examples).
- **tt-only relations are now fully readable**: the default read is the
  CURRENT STATE (one seek per key; believed-deleted keys absent) — replacing
  step 3's all-versions interim scan and completing the §4 migration
  invariant: adding `tt: TxTime` to a relation changes no existing query's
  results. `@ (tt: T)` reads the state as of any past commit time. Rides the
  existing single-axis skip-scan; fixed-rule inputs resolve the same way.
  (Bitemporal relations gained their reads in step 4b below — migrating vt
  relations to vt+tt is now supported.)
- **Fixed: negation against versioned scans panicked** (`unreachable!()` in
  `NegJoin`) — with the current-state default this would have made every
  `not *audit{…}` against a tt-only relation a crash that poisons the Db
  handle; `StoredWithValidityRA` gained `neg_join` (skip-scan mirror of the
  stored one). This also fixes the **pre-existing upstream panic** on negated
  vt atoms with `@` selectors (`not *rel{k @ 'NOW'}`).
- Also: nullable `Validity` is rejected when `TxTime` is declared (the 4b
  resolution has no semantics for a null vt); `choose_index`'s validity flag
  is honest for tt-stamped reads (inert until step 5 legalizes indexes on tt
  relations); a projected tt column currently renders as a `[ts, flag]` pair —
  timestamp-only rendering arrives with `::history` (step 5).

### New — bitemporal reads complete: the two-level (vt, tt) resolution (bitemporality step 4b)
- **Bitemporal relations are now fully readable.** The §3 resolution algorithm
  is live: per key, vt-groups are walked newest-first from the selected valid
  time, each group resolved to its greatest `tt ≤ T` belief **across both
  is_assert runs** (a later-recorded cessation at the same vt is never
  shadowed by the assert run); assertions answer, retractions mean
  believed-deleted (no shine-through), empty-at-T groups fall through; equal
  `(vt, tt)` ties resolve to the assertion. All four §4 selector forms work:
  bare scan = every vt record at current belief (retract rows included — the
  migration invariant: a vt relation's results are unchanged by adding
  `tt: TxTime`, pinned comparatively); `@ V` = the belief now about V;
  `@ (tt: T)` = the whole relation as it stood at T; `@ (vt: V, tt: T)` = the
  full quadrant. Implemented as a probe-driven scan (`data/bitemporal.rs`)
  with a generic default over every backend; per-backend seek-loop overrides
  are step-6 work (measured ~5x a plain scan on sqlite for correction-heavy
  data — acceptable until then).
- **`:as_of <t>`** — one query option pinning the default transaction time for
  every tt-stamped relation atom in the block that lacks an explicit
  selector (explicit wins; plain/vt relations untouched; using it in a query
  that references no tt-stamped relation is an error). The spec's #1 use case:
  re-run a report exactly as it would have answered at T.
- **Fixed: joins binding temporal columns silently truncated resolution** —
  a prefix join whose join columns reached into the trailing temporal key
  columns clamped the scan to one version: superseded/ceased values were
  resurrected on sqlite and mem panicked (`BTreeMap::range` inversion).
  Dispatch now falls back to a materialized join over the resolved scan
  whenever the join prefix leaves the plain key columns — **including the
  pre-existing upstream variant on vt-only relations** (`@`-selected joins
  binding the Validity column had the same wrong answers/panic since before
  the fork). Defensively, the bitemporal probe treats out-of-range bounds as
  exhausted. Pinned on mem and sqlite.
- Recorded deferrals: the unquoted-date parse lint (no warning channel in the
  engine yet) and the §6 write ops owed by step 3 (`:insert`/`:update`/
  `:ensure`/`:ensure_not`/bitemporal `:rm` remap) move to **step 4c**, now
  unblocked by the read path; error messages updated to say so.

### New — existence-checking writes on TxTime relations (bitemporality step 4c)
- The §6 write ops owed by step 3 are live, all evaluated against the
  **resolved current belief** ((vt=NOW, tt=current) on bitemporal relations):
  `:insert` (tt-only: current-belief absence — re-inserting a believed-deleted
  key is legal; **bitemporal: no records at any valid time** — a NOW-only gate
  would let an "insert" silently rewrite past or future vt-groups; duplicate
  keys within one statement rejected); `:update` (merges provided value
  columns over the current belief; the correction lands in that belief's own
  vt-group; binding the vt column is rejected — use `:put` to correct a
  specific version); `:ensure`/`:ensure_not` (assertions about the current
  belief; binding vt or tt is rejected — a silently retargeted assertion is
  worse than none; a key rewritten by the same transaction is an ambiguous
  target and errors; pending writes/removals count as existing for
  `:ensure_not`); and the **bitemporal `:rm {k, vt}` remap** — a cessation:
  buffers a vt-retraction with values copied from the belief at that valid
  time (no belief → no-op; `:delete` asserts one exists). One belief event
  per transaction throughout: writes and existence-checks of one key cannot
  mix in one transaction.
- `:replace` on TxTime relations stays rejected (destroy-and-recreate would
  drop history); triggers/indexes/callbacks remain step-5 work.

### New — bitemporal system operations (bitemporality step 5)
- **`::history rel [[k]…] [limit] [offset]`** — the introspection surface: every
  (vt, tt) record of the given keys, raw. Columns `keys…, vt_ts, op, tt,
  values…` (`op` ∈ assert/retract; `vt_ts` absent on tt-only relations; both
  timestamps as integer µs); ordering key-asc, vt-desc, tt-desc. Errors on
  non-TxTime relations.
- **`::history_gc rel <cutoff-tt>`** — drops superseded records below the
  cutoff while preserving, per (key, vt-group), exactly the record the
  resolution would pick at tt = cutoff — so as-of reads at or above the
  cutoff are unchanged. Persists a per-relation **gc floor**: an as-of read
  below it errors instead of silently returning a post-hoc reconstruction as
  if it were the historical belief. Read-only-guarded. *(v1 runs in one
  transaction; chunked online gc is deferred until real store sizes pull it.)*
- **`::evict rel [[k]…] [unredacted]`** — hard-deletes every record of the
  given keys (the one deliberate break of append-only, for GDPR), writing an
  audit row (relation, key marker, eviction tt, rows deleted) to the reserved
  `mnestic_evict_audit` relation **in the same transaction**. The key marker
  is a **salted hash** by default — storing the key itself would re-enshrine
  the PII the eviction removes; `unredacted` opts out. Read-only-guarded.
- Recorded deviation: **B-tree index legalization on TxTime relations is
  deferred** (spec §10 step 5 listed it) — statement-time index maintenance is
  structurally incompatible with buffered commit-time stamping, and there are
  zero consumers; the rejection message now says so without a step number.
- Adversarial-review hardening (all empirically confirmed before fixing):
  `::evict`/`::history_gc` bail when the same transaction holds pending tt
  writes for the target relation (they'd be stamped after the deletes and
  resurrect the evicted keys); imperative-only `{::evict}`/`{::history_gc}`
  programs now get a write transaction + per-relation locks (previously an
  error on RocksDB, an unlocked mutation elsewhere); all three ops enforce
  access levels (history ≥ read_only, gc/evict ≥ normal); the
  `mnestic_evict_audit` name is enforced as reserved (a pre-existing relation
  with a divergent schema, indices, or triggers is rejected — raw audit puts
  would corrupt/diverge it); duplicate keys in one `::evict` no longer
  overwrite the audit row with `rows_deleted = 0`; `::history`/`::evict` keys
  coerce through the column types (a mistyped key errors instead of silently
  matching nothing); `::history_gc` reports the *effective* floor, refuses
  future cutoffs, and no longer raises the floor on a no-op run (nothing
  deleted ⇒ every read below the cutoff is still exact — and the floor is
  irreversible); its keeper tie-break now reads the vt flag on bitemporal
  relations (the tt flag byte is reserved-0 there) and bails on a corrupt vt
  column instead of silently merging adjacent groups; `::history` output is
  key-ascending, rejects header collisions with user columns named
  `op`/`tt`/`vt_ts`, and its limit/offset are strict `pos_int` tokens (`2 -1`
  no longer silently parses as the single limit `2 - 1`); the burned audit tt
  is covered by the persisted clock HWM (evict transactions commit through
  the tt path).

### Perf — temporal-read budget: pinned-cursor bitemporal scans + the bench gate (bitemporality step 6)
- **`benches/time_travel.rs` rewritten** from the nightly-only `#![feature(test)]`
  relic into a stable criterion bench (registered `harness = false`;
  `autobenches = false` keeps the remaining nightly relics pokec/wiki_pagerank
  from breaking a bare `cargo bench`). Matrix per §9: versions-per-key
  (1/10/100) × corrections-depth (0/2), point reads + full-scan aggregation +
  an as-of-past-tt read, against the named baseline "the identical workload on
  a vt-only relation at `@ 'NOW'`", plus tt-only parity and a plain
  non-temporal reference. Setup sanity-asserts the gate cells answer
  identically. `MNESTIC_BACKEND=mem|rocksdb` selects the backend.
- **The generic probe default measured 4–8× the baseline on scans** (a fresh
  `range_scan` per probe: statement prepare / iterator construction
  dominated). Three step-6 changes brought it inside or near the §9 envelope:
  - **per-backend pinned-cursor overrides** of `range_bitemporal_scan_tuple`
    (sqlite: one prepared statement, reset+rebind per seek; rocksdb: one
    pinned iterator) driven through a shared `HybridProbe` — cache-hit →
    one speculative sequential `step()` → real seek, with a `far` hint from
    the walk so positional skips (past a whole key/group) seek directly;
  - **byte-spliced probe bounds** in `BitemporalIter` (a `Validity` key
    component is exactly 10 bytes) instead of tuple re-encoding, and landings
    decode only the two temporal axes — the full tuple is decoded only for
    emitted rows;
  - **landing reuse**: a landing that already answers the next (monotone)
    bound is reused without touching the backend.
- **Measured (medians, 1000 keys; sqlite / rocksdb; end-to-end `run_script`
  incl. parse — the same basis as the baseline and the AeonG envelope; the
  storage-layer-only delta is proportionally larger)** — point reads at
  (vt: NOW, current belief), the §10 fast-path-parity gate: **+3.8–8.8% /
  +8.2–11.5%** vs the vt-only baseline (≤ ~10% ✓). tt-only current reads:
  at-or-below baseline on both backends (parity ✓). Non-temporal relations:
  untouched dispatch (zero by construction). Full scans: v1 **beats the
  baseline ~2×** on both backends (the sequential walk out-runs the skip
  scan's per-key seeks); deeper version counts run over — c0 +21–53%, c2 up
  to ~2× — a **recorded deviation**: the two-level walk has a structural
  floor of two backend probes per key (assert + retract run) where the
  single-axis scan needs one, and corrections are physical rows the vt-only
  baseline cannot even represent. Revisit only if a real scan-heavy workload
  on deep-version relations shows up.

### Fixed — `::history` output order now matches spec §7 (step-5 follow-up)
- Rows were emitted in physical scan order, which interleaves a vt-group's
  assert and retract RUNS — a belief timeline read top-to-bottom misordered
  cessations against corrections (surfaced by the R3 review). Output is now
  key-asc, vt-desc, tt-desc as §7 documents.

### New — custom aggregate registration (provenance semirings R0b)
- **`Db::register_custom_aggr(name, is_meet, factory)` / `unregister_custom_aggr`**
  (+ `DbInstance` dispatchers): register a user-supplied ⊕ operator
  (`MeetAggrObj`, re-exported with `NormalAggrObj`/`RegisteredAggr`) usable in
  rule heads by name — the registration slot of the provenance-semirings plan
  (`docs/specs/provenance-semirings.md` §5 R0b). With `is_meet = true` the
  aggregate is admitted into **recursive rules**, riding the existing
  stratifier guard and `changed`-bit saturation with zero stratifier change;
  the ⊕ must then be an absorptive semilattice operation (the registrant's
  obligation; a **debug-build probe** in the meet path re-applies operands on
  custom aggregates and panics on observed non-idempotence). Outside recursion
  a custom aggregate runs through a derived normal-path adapter (state = 0̄,
  set = ⊕). Registry is in-memory and `Db`-scoped (persistence is R2); builtin
  names are reserved; duplicates error (unregister to replace — already-parsed
  programs keep their factory); names must be lowercase identifiers; custom
  aggregates take no arguments in R0; ⊗ stays ordinary rule-body arithmetic.
  Factories and operators must not panic (no `catch_unwind` in the engine) and
  factories must be cheap (called O(rules × epochs) per query).
- **Custom aggregates are rejected in trigger scripts** at `::set_triggers`
  time (a trigger is persisted CozoScript re-parsed on every write; a fresh
  `Db` open would lack the registration — unsupported until R2).
- **Breaking (fork-internal API):** `parse::parse_script` gains a
  `custom_aggrs` parameter.

### New — bounded-meet aggregates: `min_cost_k` top-k proofs (provenance semirings R1)
- **A third aggregate category, `AggrKind::BoundedMeet`** — the genuinely-new
  engine work of the semirings feature (spec §6/§9.4): a recursive aggregate
  that keeps **up to k rows per group**, so a query returns the k best whole
  derivations for each answer instead of one. Rows flow through recursion as
  ordinary tuples (⊗ stays rule-body arithmetic, exactly like `min_cost`),
  while the ENGINE owns truncation at every fixpoint step: the new
  `BoundedMeetStore` insert-sorts each candidate under the aggregate's total
  order, deduplicates on `Ordering::Equal` (the `○=` equivalence), and
  truncates to k. NOT the meet path — displacement means rows can leave the
  store, which the idempotent-semilattice assumptions of `Meet` never allow.
- **Shipped instance: `min_cost_k([payload, cost], k)`** — the k lowest-cost
  packs per group, one output row each, cost-ordered (ties break on the whole
  pack; exact duplicates collapse). K-shortest-paths is the direct idiom:
  the `min_cost` recursion with `min_cost_k(pack, k)` in the head. The
  finance/audit shape: "the k most-likely paths plus the exact evidence
  chains that justify them".
- **Convergence guard**: the changed-bit is only a saturation check for
  non-idempotent tags, so the evaluator bails after 4096 CONSECUTIVE epochs
  in which some bounded k-set changed — catching cost-decreasing cycles
  loudly instead of hanging (review must-fix: an earlier stratum-wide cap
  falsely killed converged bounded rules co-stratified with unrelated long
  recursions; the consecutive-change counter caps only live divergence).
  Known limit: a legitimate bounded recursion deeper than 4096 epochs also
  trips the cap (the error says so).
- Semantics documented: **Scallop-style approximate top-k** — upstream
  truncation prunes derivations, and candidates already consumed by earlier
  epochs are not retracted; the k-set at fixpoint = the k best candidates
  ever surfaced. Cost-order of the k rows is guaranteed only when the
  bounded aggregate is the entry head (downstream rules re-sort into value
  order). NaN costs are admitted and deterministically rank worst.
- v1 restrictions (validated with a loud error): the bounded aggregate must
  be the single aggregated column, in the last head position; mutual
  recursion between bounded rules stays unstratifiable; custom registration
  of bounded-meet operators is deferred (the R0b registry still only takes
  meet/normal aggregates).
- Adversarially reviewed: 16-scenario probe battery (relay displacement,
  DAG-exactness vs brute force, zero-cost cycles converge, equal-cost
  lexicographic divergence hits the cap, stratification shapes, `::explain`,
  `:limit`/`:order` post-application) — all sound after the must-fix.

### Resolved — annotation persistence needs no storage-format change (provenance semirings R2)
- The spec anticipated persisting semiring tags via a row-format change. The
  tags-as-columns architecture (R0/R1) made that unnecessary: annotation
  values are ordinary `DataValue`s, so **`:put` of an annotated query output
  already persists them** in the existing memcomparable row format — the R2
  acceptance criterion ("an annotated derivation is materialized and
  queryable without recompute") holds by construction. Pinned by
  `semiring_tags_persist_in_rows` across a real reopen:
  - meet-annotated derivations (`min_cost` packs) round-trip;
  - bounded-meet outputs materialize as k rows per group (pack in the key);
  - **composition with the tt axis**: materializing an annotated derivation
    into a `TxTime` relation yields *annotated belief history* — `::history`
    shows each materialization with its engine-stamped tt, and an as-of read
    returns the annotation as believed at that time ("what did we believe,
    and why, as of T" — the persistence half of the R3 story);
  - custom-aggregate outputs stay readable after reopen with NO
    re-registration; re-computing without the registration errors loudly.
- Recorded decisions: a hidden per-row tag slot (Scallop-style) is overbuild
  — no consumer, contradicts tags-as-columns; custom aggregates in trigger
  scripts stay rejected **permanently** (factories are process-scoped Rust
  closures and cannot persist — the doc comment no longer promises R2).

### New — `:reconcile`: recompute-based belief revision (provenance semirings R3)
- **`:reconcile rel {…}`** — declare a query output to BE a TxTime relation's
  new complete current belief. The engine diffs the output against the
  resolved current belief and records, as ONE belief event at commit-tt:
  assertions for new/changed keys; retractions (tt-only) or vt-cessations
  with values copied (bitemporal) for currently-believed keys absent from
  the output. Unchanged keys record nothing — an identical re-reconcile is
  a true no-op (no tt burned, no history bloat). This is the R3
  truth-maintenance step in its honest recompute form: retract or append
  base facts, re-derive, `:reconcile` the derived (annotated) relation —
  derived annotations stay consistent with the revised base, and
  `::history` + as-of reads answer **"what did we believe, and why, as of
  T"** across the revision (pinned end-to-end with a `min_cost_k`-annotated
  path relation surviving a base-edge retraction). **Truth maintenance is
  user-driven**: no automatic base→derived propagation; incremental
  (DRed/counting) maintenance is recorded future work.
- **The declaration is protected transaction-wide** (review must-fix,
  empirically probed): a reconciled relation admits no other write in the
  same transaction, before or after the reconcile — including cases where
  an idempotent reconcile buffers no rows and would otherwise leave no
  pending trace (`{reconcile} {rm}` would silently empty the relation the
  reconcile just declared). Witnessed by a transaction-scoped
  reconciled-relations set, not the pending-write buffer.
- Documented contracts: TxTime relations only (plain relations keep
  `:replace`); the revision is invisible to later reads in the same script
  (§5 one-belief-event); duplicate keys with conflicting values in one
  output error; bitemporal inputs must carry assert-flagged, explicit vt
  timestamps (`'NOW'` mints a fresh group per run); value columns with
  non-constant defaults defeat idempotence if omitted; cost is
  O(relation) per call.

### Fixed — `and`/`or` meet aggregates reported an inverted changed-bit (upstream bug)
- `MeetAggrAnd`/`MeetAggrOr::update` returned `true` when the value was
  **stable** and `false` when it **changed** — so in recursive rules a real
  change never propagated through the semi-naive delta (wrong results) and
  stable values were re-enqueued. Found by the R0b adversarial review while
  auditing the new idempotence probe; fixed to report change, pinned by a
  unit test.

### Fixed — `%return { <query> }` panicked in imperative scripts (upstream bug)
- The match arm expected `query_script_inner` where the grammar delivers
  `imperative_clause`; any braced clause in `%return` hit `unreachable!()`.


## 0.9.0 — 2026-06-28

Adds the **read-only Cypher query surface** (the headline feature) and bundles
the corrupt-database tooling that was banked as 0.8.6 but never published
(0.8.5 → 0.9.0 ships both; there is no separate 0.8.6 crates.io artifact).

### New — read-only Cypher query surface (alpha, feature `cypher`, off by default)
- Translate a subset of **openCypher** to CozoScript so the engine can be
  evaluated and adopted without first learning Datalog. Datalog stays the native,
  full-power language; this is a **read-only** on-ramp (no write clauses). New API:
  `DbInstance::run_cypher` / `cypher_to_script` (+ Python `run_cypher`), driven by
  a caller-supplied `CypherGraphSchema` / `NodeMap` / `EdgeMap` mapping the
  property-graph model onto stored relations — supporting both the
  relation-per-label and the shared-relation-with-discriminator conventions
  (the latter matches MindGraph's reified `node`/`edge` model).
- v1 subset: `MATCH` (fixed-length, directed, labels, types, inline property
  maps), `WHERE` (comparisons, `AND`/`OR`/`NOT`, `IN`, `IS NULL`, `STARTS`/`ENDS
  WITH`, `CONTAINS`), `RETURN` (`DISTINCT`, aliases, aggregates
  `count`/`sum`/`avg`/`min`/`max`/`collect`), `ORDER BY` / `SKIP` / `LIMIT`.
  Literals pass as params and every interpolated identifier is validated. True
  bag semantics (`count(*)`/`LIMIT` match openCypher), **null-aware `WHERE`**
  (a null operand drops the row instead of aborting the query), and per-`MATCH`
  edge-isomorphism. Module `cozo-core/src/cypher/`; design + scope in
  `docs/specs/cypher-read.md`; hardened against a multi-agent adversarial review.
  Off by default — enable the
  `cypher` feature (the published PyPI wheel ships without it for now; build with
  `--features cypher` to get the Python `run_cypher`). Deferred with explicit
  errors: undirected relationships, the
  schema `filter` field, variable-length paths, `OPTIONAL MATCH`, `WITH`. Known
  divergence: `sum` over an integer column returns a float (engine accumulator is
  f64).

### Fixed — `cozo-bin` token-table bearer auth ignored on query-string URLs
- The server's `authorize` evaluated `Authorization: Bearer` against the token
  table only in the `uri().query() == None` branch, so any endpoint that takes
  query params (`/transact?write=true`, `/rules/name?arity=2`) rejected valid
  bearer tokens. Bearer auth is now evaluated independently of the query string
  (a `?auth=<secret>` credential still takes precedence, then bearer falls
  through regardless of other query params). CORS also now allows the
  `Authorization` header so browser bearer clients pass preflight.

### Fixed — `cozo-bin` had no default feature, failed to build bare
- `cargo build -p cozo-bin` (no features) failed to resolve `rayon` — cozo-core
  uses rayon unconditionally on non-wasm, but it is pulled in only via the
  `graph-algo` feature, which `cozo-bin` did not enable by default — and any
  binary it did produce had no storage backend. `cozo-bin` now defaults to the
  `compact` combo (sqlite + requests + graph-algo): a runnable server/REPL out
  of the box. Opt into rocksdb explicitly.

### Fixed — `::index create` no longer panics on corrupt tuples
- Index population extracted columns by position with no bounds check: one
  truncated stored tuple (e.g. from an interrupted write) panicked the whole
  build — and since applications may (re)create indexes while initializing a
  database, one bad row made the database unopenable (observed in production
  2026-06-12). Corrupt tuples are now skipped with a loud error naming the
  relation, the index, and the arity mismatch, plus a build-level summary
  ("N corrupt tuple(s) skipped — the base relation needs repair").

### New — `::repair_corrupt <relation>`: surgically delete truncated tuples
- Tuples whose stored arity is shorter than the schema (truncated value bytes
  from interrupted writes) are deleted by their intact store keys; returns the
  removed count. Gives applications a surgical alternative to dropping a
  database that fails integrity checks — the motivating incident (2026-06-12)
  saw an application-level "repair" delete a production database over 15 bad
  rows. Pinned by `tests/fork_regressions.rs`.

## 0.8.5 — 2026-06-12

### Changed — flat in-RAM parallel HNSW bulk build (`::hnsw create`)
- `::hnsw create` now constructs the graph in flat, integer-indexed memory
  (one contiguous vector slab + per-node adjacency arrays, the
  hnswlib/pgvector/Lucene layout) instead of the temp-store BTreeMap of
  encoded tuples, and inserts in parallel with per-node locks. A profile of
  the old build showed >50% of CPU in tuple encode/decode, `CompoundKey`
  hashing and allocator traffic — all gone. The finished graph is serialised
  into the index relation's existing tuple format in one pass: the on-disk
  layout, the search path, incremental maintenance (`hnsw_put`/`hnsw_remove`),
  and the non-blocking Phase A–D build/reconcile orchestration are unchanged.
- `MNESTIC_INDEX_BUILD_THREADS` controls worker count (unset/0 = all cores;
  1 = serial insertion in scan order, the old behaviour). Parallel insertion
  makes the built graph non-deterministic across runs (link sets vary with
  interleaving, as in hnswlib/pgvector); recall agreement is guarded by
  `tests/hnsw_build.rs::parallel_build_recall_agreement`.
- Two parallel-only divergences from the serial insertion algorithm, both
  required for lock-safety: neighbour-overflow shrinking never extends
  candidates (would need two node locks at once), and a node's own link-list
  write merges with concurrently-arrived backlinks instead of replacing them
  (replacement severed edges and, on chain-shaped data, broke connectivity).

### Changed — plain-snapshot read path for read-only scripts (RocksDB)
- Read-only scripts no longer open a pessimistic transaction. They read the
  base DB through a plain snapshot (`SnapshotReadBridge` in `mnestic-rocks`
  0.1.9): the same consistent view as before — the old read path also pinned
  one snapshot — but with no lock-manager bookkeeping and no transaction
  write-batch overlay consulted on every read. This is the standard MVCC
  read pattern (TiKV, CockroachDB). Writing scripts keep the pessimistic
  transaction unchanged. Isolation semantics are pinned by
  `tests/snapshot_reads.rs` (uncommitted writes invisible, read transactions
  keep their snapshot view across concurrent commits).
- Measured (RocksDB, immutable scripts, 50k rows): keyed point read p50
  **28.5 → 23.9 µs** (−16%), p99 −19%; 20-row prefix scan p50 **46.0 →
  41.5 µs**. Retrieval-scale queries (40–150 ms) on a block-cache-resident
  corpus showed no measurable change, as expected — per-script transaction
  overhead is µs-scale. Parse/compile (~20 µs) now dominates point reads,
  which is the stored-queries → plan-cache item's territory.
- A write attempted through a read-only transaction now errors explicitly
  instead of silently succeeding inside a transaction that had no business
  existing. (No in-tree path did this; the error guards against future ones.)

### Changed — batched HNSW neighbour reads on the search path
- `hnsw_search_level` fetches all unvisited neighbours' vectors per expansion
  step through one `StoreTx::multi_get` — a true RocksDB `MultiGet` (shared
  bloom-filter probes, batched block reads) on the snapshot read path —
  instead of one serial point get per neighbour (`VectorCache::ensure_key`).
  Other backends fall back to a serial loop via the trait default. Neutral on
  a block-cache-resident corpus (fused p50 unchanged); the win case is
  cold-cache / larger-than-RAM data, where serial point gets pay one block
  read each.

### Changed — FTS bulk build (`::fts create`)
- The populate loop no longer runs a del pass per row: the index relation is
  freshly created and empty, so the old code tokenised every document a
  second time to delete postings that could not exist.
- Tokenisation + posting-row encoding now fan out across worker threads
  (same `MNESTIC_INDEX_BUILD_THREADS` control); the row format is unchanged.
- Corpus doc-stats (`avgdl`) are counted exactly during the build and seeded
  directly, replacing the post-build full index scan.

### Fixed — `::describe` was unreachable upstream; now parses and is read-only-guarded
- Upstream defines `describe_relation_op` in the grammar and implements
  `SysOp::DescribeRelation`, but never wired the rule into the `sys_script`
  alternations — `::describe rel 'note'` always failed to parse. The op is
  now reachable (top-level and inside imperative blocks).
- `::describe` writes relation metadata; it was also the only mutating sys op
  without a read-only guard. It now rejects `ScriptMutability::Immutable`
  with a clear error, like its siblings, instead of falling through to the
  storage layer. Pinned by `tests/fork_regressions.rs`.

### Tests — bulk-build coverage from the post-ship bug-hunt audit
- The audit (three independent reviews of the flat HNSW build, the snapshot
  read path/FFI bridge, and the FTS bulk build) found no correctness bug; it
  found untested live paths, now covered: HNSW flat build over a
  **list-of-vectors column** (`[<F32; N>]`, the `sub_idx` branch), an
  **F64 + Cosine** flat-build recall guard, and bulk-vs-incremental FTS
  doc-stats score equality on a **multi-column-PK** relation.
- Documented in the `hnsw_build.rs` module header: the flat build omits
  pruned edges where the incremental path writes tombstoned rows —
  indistinguishable to search (`include_deleted=false`); the degree counter
  it feeds was already approximate upstream.

## 0.8.4 — 2026-06-10

Fifth fork release: a defect fix for 0.8.3's concurrent-write regression plus
the per-leg fusion detail that powers MindGraph's "why retrieved" surface.

### New — per-leg retrieval detail: `detailed` on RRF and `HybridSearch`
- `ReciprocalRankFusion(..., detailed: true)` switches the output from
  `[item, fused_score]` to the long-format
  `[item, fused_score, list_id, leg_rank, leg_score]` — one row per
  *(item, contributing list)*. `leg_rank` is the 1-based within-list rank the
  fusion actually used (after best-score dedup); `leg_score` is that
  deduplicated raw score; lists an item did not appear in contribute no row.
  `detailed` must be a constant boolean (output arity depends on it). This is
  the mechanism behind a consumer's "why was this retrieved" surface: the rows
  reconstruct the fused score exactly (`Σ 1/(k + leg_rank)`).
- `HybridSearch::detailed: bool` plumbs it through the one-call helper. Without
  MMR the head is `[id, score, list_id, leg_rank, leg_score]` and the row limit
  widens to `limit × leg-count` so the top `limit` items are always fully
  covered; with MMR the per-leg detail is joined onto MMR's selection and the
  head is `[id, rank, score, list_id, leg_rank, leg_score]`.
- Python binding: pass `detailed: True` in the `hybrid_search` dict.

### Fixed — concurrency regression in the 0.8.3 doc-stats counter
- 0.8.3's durable `avgdl` counter was one shared storage key per FTS index,
  read (without a lock) and rewritten inside **every** document transaction.
  Under RocksDB pessimistic transactions this made all concurrent writers to
  an FTS-indexed relation conflict on a single row lock (held until commit),
  and the unlocked read-modify-write also lost updates, silently drifting the
  counter. Concurrent ingest produced `Resource busy`-class storage errors.
- The counter is now **process-cached and scan-seeded**: one deduplicated
  full scan per index per process (the path 0.8.3 already used for legacy
  indices), maintained incrementally in memory on every put/delete, with no
  shared storage key in the hot path. Per-query `avgdl` stays O(1); the
  pinned behaviours (deletes net correctly, scores identical across reopen,
  BM25 denominator) are unchanged and covered by `tests/fts_avgdl.rs`.
  Index rebuilds reseed the cache and delete any legacy 0.8.3 counter key.
- Deltas from rolled-back transactions are not undone; the drift is
  negligible for a smoothing denominator and clears on restart or rebuild.

### Fixed
- `log::error` import in `jlines.rs` is now gated behind the `requests`
  feature (was an unused-import warning under minimal feature sets).

## 0.8.3 — 2026-05-31

Fourth fork release. Two agentic-memory wedge features land together and are
**validated end-to-end** on the `mnestic-benchmarks` hybrid suite (2026-05-31,
SQLite-backed wheel, vs SQLite/DuckDB/LanceDB/Kuzu): **native 3-way fused recall**
(Bet 1a) and **BM25-correct FTS with O(1) `avgdl`** (Bet 1b). All 169 inherited
lib tests + feature suites pass; `cargo clippy -p mnestic -- -D warnings` is clean.

> **Heads-up — the FTS default scorer changed.** The default `::fts` score kind
> moves from `tf_idf` to Okapi `bm25` (a behaviour change). `tf` and `tf_idf`
> stay selectable for byte-identical upstream scoring.

**Measured (2026-05-31):**
- **BM25 + O(1) `avgdl`:** fused recall **0.75 → 0.954** (parity with DuckDB
  0.957 / SQLite 1.0); decomposed-path p50 **927 → 175 ms** and the cold p99 tail
  **2,900 → 258 ms**. (The tail was the per-query `avgdl` scan, *not* cold HNSW as
  first assumed — the vector leg even got faster cold→warm, 117 → 23 ms, unchanged.)
- **Native 3-way:** the one fused call runs vector+FTS+graph at **41.55 ms p50**
  (recall 0.873) — **~4× faster** than the 175 ms hand-decomposed path, fusing a
  signal (graph) no other engine here has (LanceDB native is 2-way only: recall
  0.456). The one-call advantage reappeared exactly as predicted once the `avgdl`
  fix removed the FTS scan that had masked it.

### New — native 3-way fused recall: typed `GraphLeg` on `HybridSearch`
- `HybridSearch::graph_legs: Vec<GraphLeg>` (new `GraphLeg` type, re-exported from
  the crate root). Each leg expands from a set of `seeds` over a stored edge
  relation up to `max_hops`, scores every reached node by its **minimum hop
  distance** (closer ⇒ higher rank), and contributes that ranked list to the
  *same* Reciprocal Rank Fusion as the vector/keyword legs — one call, one
  transaction, no hand-written recursion.
- Why a new type rather than the existing `extra_lists`: an `extra_lists` entry is
  a *single* spliced rule body, which cannot express the recursive shortest-path
  rule that bounded-hop proximity needs. `GraphLeg` generates that rule — a seed
  relation, a hop-1 base rule, and a `min(dist)` recursive rule gated at
  `max_hops` — for you. Supports `undirected` traversal (also follows edges in
  reverse) and multiple seeds (unioned).
- **Injection-safe.** Seed values are passed as query **params** (`$hg{i}_seed{j}`),
  never string-interpolated; the label, edge relation, and column names are
  validated as bare identifiers, and empty seeds / `max_hops == 0` are rejected.
  The generated script remains inspectable via `hybrid_search_script`.
- `runtime/hybrid.rs`; guarded by `tests/hybrid_graph_leg.rs` (recall a neighbour
  the fixed legs miss, closer-outranks-farther via `min(dist)`, hop bound,
  undirected reverse edges, multi-seed union, script-is-recursive-and-parametrised,
  input validation). Backward compatible: an empty `graph_legs` generates the exact
  prior script.

### Added — read-path latency baseline (groundwork for a future plan/stored-query cache)
- `benches/read_path.rs` (criterion) times `parse_only` (parse + compile-to-AST)
  vs `full_run` (end-to-end `run_script`) for a point read and a multi-rule
  retrieval query on SQLite, to size the parse/compile fraction a compiled-plan
  cache could eliminate before any cache is built (the fork's baseline-first
  rule). **Finding:** parse/compile is a roughly *fixed* ~20–85 µs cost — ≈39% of
  a 55.7 µs point read but only ≈1.1% of a 7.68 ms multi-rule retrieval query — so
  a plan cache helps cheap point reads but is noise for the retrieval workload,
  where execution (and, on RocksDB, the pessimistic txn) dominates. That makes
  **Bet 1a (one fused call instead of three `run_script`s) the read-path latency
  fix that matters**, not the plan cache. A real plan cache must also clear two
  structural blockers: parse-time param inlining, and the lack of a reusable-plan
  execute entry point.

### FTS — Okapi BM25 scoring + summed disjunction + O(1) `avgdl`

The recall lever the hybrid-retrieval benchmark localized the entire fused-recall
gap to (FTS recall-agreement 0.72 vs vector 0.99 / graph 1.00).

- **New default score kind `bm25`** for `::fts`/`~rel:idx{… | score_kind: 'bm25'}`.
  Implements `idf · tf·(k1+1) / (tf + k1·(1 − b + b·|D|/avgdl))`: term-frequency
  **saturation** (`k1`, default 1.2) and **document-length normalization** (`b`,
  default 0.75, range `[0,1]`), both tunable as query params. Replaces upstream's
  raw `tf · idf`, which had neither — long documents and high raw term counts
  dominated unfairly. Two upstream defects fixed:
  - *No length normalization.* The per-document token length was **already stored**
    on every posting (`vals[3]`) but **discarded** at search time; it is now read
    (`LiteralStats::doc_len`) and used. Average document length (`avgdl`) is an
    **O(1) read** of a durable per-index doc-stats counter (see below).
  - *Disjunction did not sum.* An `a OR b` query took the **max** of per-term scores,
    so a document matching both terms could tie one matching a single term — forcing
    callers into app-side per-term aggregation with wide over-fetch. Under `bm25`,
    `OR` now **sums** per-term contributions (a document matching more query terms
    ranks higher). `tf`/`tf_idf` keep upstream's max-combine.
- **Backward compatible:** `score_kind: 'tf_idf'` and `'tf'` are unchanged
  (byte-identical scoring and the original `OR`=max semantics). Only the *default*
  moved to `bm25`.
- Guarded by `tests/bm25.rs` (sqlite backend, stored path): OR-sum beats
  repeated-single-term, length normalization favors the shorter doc, and `b: 0.0`
  provably disables length normalization (proving `b` is wired through).
- **`avgdl` is now O(1) (durable doc-stats counter).** The bench validated BM25's
  recall (0.75 → 0.96) but exposed a **~10× FTS latency regression** (71 → 755 ms
  p50): the initial `avgdl` was a full deduplicated index scan (O(#docs),
  ~680 ms/query at 40k chunks) recomputed on *every* query, because the cache lived
  on the per-operator `FtsCache`. Fixed: each FTS index maintains a durable
  `(total_tokens, n_docs)` counter at a reserved `[Bot]` key (sorts above all
  `[term, …doc_key]` postings, so it is invisible to term scans). `put_fts_index_item`
  adds a document's tokens, `del_fts_index_item` subtracts them (guarded by a posting
  existence probe), and `create_fts_index` publishes the authoritative count via one
  final scan — so `avgdl` is a single keyed `get`. A legacy index that predates the
  counter migrates itself on its first write (seed-by-scan) and, until then, reads
  fall back to a `Db`-scoped cross-query cache (one scan per process, not per query —
  correct because an un-migrated index is immutable). Identical scores to the prior
  scan (the counter value equals the scan); guarded by `tests/fts_avgdl.rs`
  (delete-equals-fresh-build, survives reopen, `avgdl` feeds the BM25 denominator).
  Well-behaved workloads (insert, delete, del-then-put update) are exact; an FTS-only
  relation with no secondary index can drift on in-place update, mirroring upstream's
  existing posting leak there (an index rebuild resets it). **Bench-confirmed:** the
  FTS leg returned to ~71 ms and decomposed p99 fell 2,900 → 258 ms with recall held
  at 0.954.

### Python
- `cozo-lib-python`'s `hybrid_search` now accepts a `graph_legs` list (mapped to
  `Vec<GraphLeg>`), so the embedded `mnestic` wheel can drive the native 3-way
  fused recall from Python. `cozo-lib-python` stays workspace-excluded (built only
  when the wheel is built).

## 0.8.2 — 2026-05-30

Third fork release. Makes HNSW index builds **non-blocking for readers**: an
index build no longer freezes all reads on the base relation for the (often
multi-minute) duration of the build. All 169 inherited lib tests + the
integration/feature suites pass; `cargo clippy -p mnestic -- -D warnings` is clean.

### Performance — non-blocking HNSW index builds (readers no longer stall)
- Building/rebuilding an HNSW index (`::hnsw create`) used to hold the base
  relation's **exclusive write lock** for the *entire* build, so every concurrent
  read (which takes the same lock shared) blocked until the build finished — in
  production, **10–20+ minutes** (76 min for a 151K × 1536 index). The stall was
  cozo's per-relation `ShardedLock`, not RocksDB.
- The build is now done **off-lock** on RocksDB: the heavy graph construction runs
  under a read-only snapshot with **no relation lock held**, and the lock is taken
  only briefly to set up the empty index relation and to publish the result.
  Measured: building a **40,000**-vector index takes ~5.6 s, during which
  **90,507** concurrent reads of the same relation completed, the slowest in
  **0.8 ms** (release). Previously those reads would have queued behind the whole
  ~5.6 s build.
- **How it stays correct.** The finished, key-sorted graph is bulk-published into
  the live store via `SstFileWriter` / `IngestExternalFile` (bypassing the
  transaction write-batch), and the index *data* is always ingested before its
  *metadata* is committed — so a reader can never observe an index before its keys
  exist. Base-relation rows that change during the unlocked build are folded in by
  a short reconcile pass (re-scan + diff against the build snapshot, applying the
  same incremental `hnsw_put`/`hnsw_remove` maintenance) under a brief final lock.
  Concurrent builds of the same index are serialised; a lost race cleans up its
  ingested data. Index relation ids are always freshly allocated, so a crash
  mid-publish leaves at worst unreferenced dead keys, never a torn index.
- Non-RocksDB backends (sqlite/mem/…) keep the in-transaction build + per-key
  flush unchanged. New `Storage::ingest_sorted` (default-errors; real impl only on
  RocksDB) carries the SST bulk-load. Guarded by `tests/hnsw_nonblocking_build.rs`
  (build correctness, persistence across reopen, reads-during-build, reconcile of
  concurrent inserts, drop+recreate).

## 0.8.1 — 2026-05-30

Second fork release. Adds the one-call hybrid-retrieval API, a substantial HNSW
index-build speedup, the maintained `mnestic-rocks` bridge fork, and a blocking
clippy CI gate. All 169 inherited lib tests + integration/feature tests pass.

### New — one-call hybrid retrieval (`HybridSearch`)
- `DbInstance::hybrid_search` / `Db::hybrid_search` (and `*_script` to inspect the
  generated CozoScript) run an HNSW + FTS (+ optional graph-traversal) recall,
  fuse it with `ReciprocalRankFusion`, and optionally diversify with
  `MaximalMarginalRelevance` — in one typed call. Previously this was ~7
  hand-assembled Datalog rules. Read-only; the query vector/text are passed as
  script params (never string-interpolated) and every interpolated identifier is
  validated against injection. New module `runtime/hybrid.rs`; tested in
  `tests/hybrid_helper.rs`.

### Performance — HNSW index build ~3× faster
- `::hnsw create` (and backfill) over a large relation was **superlinear**: the
  whole graph was built inside the script's pessimistic transaction, so every
  neighbour read/write round-tripped through RocksDB's `WriteBatchWithIndex`
  overlay (whose cost grows with the index). Measured baseline: **135 s for
  20k × 128** vectors (release).
- The build now constructs the graph in the in-RAM temp store (`is_temp` routing
  via new `idx_put`/`idx_get`/`idx_del`/`idx_exists` helpers) and bulk-migrates
  the finished, key-sorted graph to the persistent store in one pass; and it
  shares one `VectorCache` across the whole build instead of rebuilding it per
  node. Combined **~3.1× faster** (20k × 128: 135 s → 43.6 s; 10k: 51.8 s →
  16.5 s). The built graph is byte-identical; guarded by `tests/hnsw_build.rs`.
- *Investigated and dropped:* batched secondary-index writes (drafted "#4",
  claimed 2–3× ingest). Measured: index writes are ~7 % of ingest even with 4
  indexes, and within a pessimistic txn each `put` only appends to an in-memory
  batch — batching saves <1 %. The real ingest floor is the per-script
  pessimistic transaction, which that change doesn't touch.

### Bridge — `mnestic-rocks`
- Forked the C++/RocksDB bridge crate `cozorocks` → **`mnestic-rocks`** (v0.1.8),
  keeping the importable crate name `cozorocks` (`[lib] name`) so `use cozorocks::`
  and `cozorocks?/feature` references are unchanged. Enables shipping future
  bridge-level work (e.g. out-of-transaction index build + `IngestExternalFile`
  atomic publish) on crates.io.

### Maintenance
- `document-features` 0.2.8 → 0.2.12 to clear a future-incompat warning.
- **Blocking clippy CI gate** (`cargo clippy -- -D warnings`, default features).
  Pervasive/intrinsic inherited lints are allow-listed with rationale in
  `lib.rs` so the gate catches new issues. `cargo fmt --check` is deliberately
  not gated yet (a ~178-file reformat is deferred to its own pass).

### Deferred (designed, not in this release)
- **Lock-free index build** (out-of-transaction build against a snapshot +
  `SstFileWriter`/`IngestExternalFile` atomic publish) — directly fixes "rebuild
  holds the write lock and blocks readers"; touches the transaction lifecycle.
- **Native in-RAM graph** (adjacency as integer arrays, no per-edge
  (de)serialization) for a further ~10–50× build speedup, like hnswlib.

## 0.8.0 — 2026-05-30

First fork release. Lineage: cozo 0.7.6 + 30 unreleased upstream commits (our fork
point). Bumped to **0.8.0** (not 0.7.7) to mark the fork's identity as a maintained,
agentic-memory-focused engine: it ships real planner/parser fixes *and* the first
hybrid-retrieval primitives (RRF, MMR). All changes are pure `cozo-core` (Rust) —
no `cozorocks` bridge changes — so the rocksdb feature still resolves to upstream
`cozorocks 0.1.7`. 169 inherited lib tests + 68 air_routes + all fork/feature tests
pass.

### Fork bootstrap
- Forked from `cozodb/cozo` at `481af05`; full history preserved, upstream
  remote retained as `upstream`, fork point tagged `fork-base`.
- Renamed brand and core package from `cozo` to `mnestic`. Original MPL-2.0
  per-file copyright notices preserved.
- Added `NOTICE.md` (provenance/attribution) and this changelog.

### Audit of MindGraph's drafted upstream bugs against the fork point

The fork's pre-fork bug drafts were written against the
**crates.io 0.7.6 release**. Our fork point (`481af05`) is the upstream `main`
HEAD, which is **30 commits ahead of the `v0.7.6` tag** (Ziyang merged fixes
after the release but never cut another version). Each draft was therefore
re-verified against HEAD with a backend-correct reproducer:

| # | Bug | Status at fork HEAD | Evidence |
|---|-----|---------------------|----------|
| 3 | `mat_join` joins on wrong symbols (silent 0 rows) | **Already fixed** by the PR #286 "fix-stored-prefix-join" cluster | `tests/matjoin_regression.rs` passes; plan now emits `stored_prefix_join {"**2":"uid"}` instead of `stored_mat_join {"**2":"**0"}` |
| 1 | Equality post-filter → full scan, not prefix lookup | **Was present; now FIXED in the fork** (measured ~28–29× at 5k rows) | fix in `query/reorder.rs`; `tests/fork_regressions.rs::equality_post_filter_uses_prefix_lookup` now active — see "Shipped" below |
| 2 | top-level `:create _foo` silent no-op | **Present, but is a scoping nuance** — `_`-relations are legitimate transaction-scoped temporaries (work across imperative `{...}` blocks). Only the top-level form is a silent trap. Fix is a design call. | `tests/fork_regressions.rs::top_level_create_underscore_relation_is_a_silent_noop` (ignored) |
| 4 | Secondary-index puts: N separate `.put()` calls | **Present** (perf 2-3×) | `cozo-core/src/query/stored.rs` still loops `store_tx.put()` per index |
| 7 | `hnsw` serial neighbor fetches | **Present** (perf 10-20% HNSW) | `cozo-core/src/runtime/hnsw.rs::VectorCache::ensure_key` still single-key `handle.get()` |

Note: the fork being 30 commits ahead of the released 0.7.6 means simply
adopting mnestic already gives MindGraph bug #3's fix (and the other unreleased
fixes) for free.

### Shipped in the fork

#### Phase 1 — agentic-memory features
- **`ReciprocalRankFusion` fixed rule (hybrid retrieval, Bet 1)** —
  `cozo-core/src/fixed_rule/utilities/rrf.rs`, aliased `RRF`. Fuses several ranked
  result lists (vector/HNSW + full-text/FTS + graph traversal) into one ranking
  via `Σ_lists 1/(k + rank_in_list)`. Input is a single relation
  `[list_id, item, score]`; rows are grouped by `list_id`, ranked within each list
  by `score` (`descending` option, default true), and the reciprocal-rank
  contributions are summed per item. Options: `k` (default 60), `descending`.
  Output `[item, fused_score]`, composable in further Datalog. Not feature-gated.
  Rationale: Datalog can already *sum* reciprocal contributions but cannot assign
  a *rank position within a group* — that intra-list ranking is the missing
  primitive. Tests: `cozo-core/tests/rrf.rs` (fusion math, `k` smoothing,
  ascending direction, alias, default-k).
- **`MaximalMarginalRelevance` fixed rule (diversity rerank, Bet 1)** —
  `cozo-core/src/fixed_rule/utilities/mmr.rs`, aliased `MMR`. Re-ranks a candidate
  set to balance relevance against diversity (avoids recalling near-duplicate
  memories). Input `[item, relevance, vector]`; greedily selects
  `argmax(λ·relevance − (1−λ)·max cosine_sim to already-selected)`. Options:
  `lambda` (default 0.5, clamped to [0,1]), `k` (default 0 = all). Output
  `[item, rank]` (selection order). Tests: `cozo-core/tests/mmr.rs`.
- **End-to-end hybrid retrieval test** — `cozo-core/tests/hybrid_retrieval_e2e.rs`
  runs a real HNSW (vector) search + a real FTS (keyword) search over one
  relation, fuses with `ReciprocalRankFusion`, then reranks with
  `MaximalMarginalRelevance` — proving the full hybrid path composes, not just
  synthetic ranked lists.
- Next on Bet 1: a higher-level one-call convenience + a LangChain/LlamaIndex
  adapter once the surface stabilises.
- **Pre-release review hardening** (all guarded by tests): RRF/MMR reject
  non-finite (NaN/inf) scores; MMR rejects inconsistent vector dimensions instead
  of panicking and now uses the true max cosine (not a 0 floor) so anti-correlated
  candidates are rewarded; `ulid_timestamp` rejects malformed/non-canonical ULIDs
  (wrong length, invalid char, leading char > 7) instead of silently truncating.
- **ULID functions (`rand_ulid`, `ulid_timestamp`)** — `data/functions.rs`,
  upstream cozo #296. `rand_ulid()` returns a lexicographically-sortable 26-char
  Crockford-base32 ULID (48-bit ms timestamp + 80-bit randomness); sortable string
  IDs are ideal keys for time-ordered agentic-memory scans (unlike random UUIDv4).
  `ulid_timestamp(s)` extracts the embedded Unix-ms timestamp. Tests:
  `cozo-core/tests/ulid.rs` (format, two hand-derivable decode vectors, recency,
  sortability, distinctness).

#### Phase 0 — fixes
- **#1 equality-pushdown for stored relations** (`query/reorder.rs`). Equality
  post-filters on a stored relation — `*rel[k, ..], k == <ground>` and
  `*rel{k, ..}, k == <ground>` — now compile to a keyed `stored_prefix_join`,
  identical to the binding-first form `k = <ground>, *rel{..}`. Upstream left
  these as a full `load_stored` scan + `eq(..)` post-filter. Implemented as a
  pre-pass (`push_equality_filters_to_bindings`) that converts qualifying
  `eq(var, ground)` predicates into unifications and hoists only those converted
  unifications ahead of the relation that produces the variable; the existing
  well-ordering logic then emits the prefix lookup. Pure optimization — result
  sets are unchanged. (NB: the fix is in `reorder.rs`, not `relation.rs::choose_index`
  as originally guessed — `choose_index` only selects *secondary* indices.)
  - **Correctness boundary (numeric grounds are NOT pushed):** `op_eq` treats
    `Int(n) == Float(n)` as equal across types, but a keyed lookup uses the index's
    strict `Num` ordering where `Int(n) != Float(n)`. Converting a numeric equality
    would silently drop cross-type matches, so the conversion is gated to non-numeric
    ground values (str/uuid/bytes/bool/null); numeric `==` keeps full `op_eq`
    post-filter semantics. User-written unifications are never reordered. Guarded by
    `tests/fork_regressions.rs::numeric_equality_keeps_cross_type_semantics`.
  - **Measured** (criterion, SQLite backend, 5000-row relation, single-row PK
    lookup): positional post-filter **1.746 ms → 61.1 µs (~28.6×)**; brace
    post-filter **1.756 ms → 59.4 µs (~29.5×)**; binding-first unchanged
    (~48–51 µs). The speedup scales with row count (O(N) scan → O(log N) lookup).
  - Tests: `tests/fork_regressions.rs::equality_post_filter_uses_prefix_lookup`
    (now active, was `#[ignore]`); baseline bench `benches/point_lookup.rs`.
- **#281 keyword-prefixed identifiers now parse** (`cozo-core/src/cozoscript.pest`).
  An identifier starting with a keyword literal — `nullable_column`, `trueValue`,
  `falsey` — failed to parse in value positions (`*rel{col: nullable_column}`)
  because `term` tries `literal` before `var` and `null`/`boolean` had no word
  boundary, so `null` greedily matched the `null`-prefix and the parse aborted.
  Added an identifier-boundary lookahead (`~ !("_" | XID_CONTINUE)`) to the `null`
  and `boolean` rules. Closes upstream cozo #281. Tests:
  `tests/fork_regressions.rs::keyword_prefixed_identifiers_parse` (includes a guard
  that real `null`/`true`/`false` literals still parse).
- **#287 `env_logger` moved to a dev-dependency** (`cozo-core/Cargo.toml`). It was
  a hard dependency but is only used by `runtime/tests.rs` (cfg(test)). Closes
  upstream cozo #287; trims downstream build graphs.
- `tests/matjoin_regression.rs` — regression guard pinning the #3 fix.
- `benches/point_lookup.rs` — first stable/CI-runnable criterion bench (the
  upstream pokec/wiki/time_travel benches need nightly `#![feature(test)]` +
  external datasets, so they don't run in CI).

### Next (ordered by value/confidence)
- **#4** batch secondary-index writes into a single `WriteBatch` — perf, needs the cozorocks bridge to expose batch put.
- **#7** `multi_get` for HNSW neighbor fetches — perf, needs bridge support.
- **bit-rot (deferred, low value)**: #307 — the upstream pokec/wiki/time_travel
  benches need nightly `#![feature(test)]` + external datasets, so they can't run
  in CI and can't be compile-verified on a stable toolchain; superseded by the
  criterion harness (`benches/point_lookup.rs`). #298 — newer rayon raises MSRV;
  builds fine on our toolchain (rustc 1.93.1, rayon 1.10/core 1.12.1), only bites
  users below rayon's MSRV, not reproducible for us. Revisit if either becomes
  load-bearing.
- **#2** decide + implement the top-level temp-create behavior (warn vs error vs surface scope).
