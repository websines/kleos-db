# How agent-memory systems make memory-management decisions, and where a small `decide()` fits

Research date: 2026-10-16. Prepared for kleos (`decide()`, UPGRADE_PLAN items D5, E3, E4, E6, E7).

**How to read this report**

- **Source-code findings** come from shallow clones in `.bench-data/research/repos/`. The commit and date of each clone are noted. Paths are relative to that folder.
- **Web and paper findings** carry a URL.
- **"Self-reported"** means a vendor or author number that nobody has independently reproduced.
- **UNVERIFIED** marks a claim I could not confirm from a primary source.
- No number in this report is estimated or invented. Where a number was not found, the report says so.

---

## 0. Executive summary

1. **The leading system abandoned LLM-decided UPDATE/DELETE.**
   - Mem0, whose ADD/UPDATE/DELETE/NOOP prompt is the reference design, switched its OSS SDK in April 2026 to **single-pass ADD-only extraction** ("nothing is overwritten or deleted"). It now uses MD5 dedup, spaCy entities matched at cosine ≥ 0.95, and hybrid retrieval.
   - Supersession and merge moved to a hosted, non-destructive "Dream" layer that sets flags and provides `latest_only` reads.
   - Self-reported LongMemEval rose from 67.8 to about 93–94 (§1.1, §2.3).
   - Graphiti, Cognee, Supermemory and LangMem (deletes off by default) are also non-destructive. **kleos's bitemporal supersession is the design the field is converging on.**
2. **The costly error is false supersession.**
   - Graphiti's contradiction prompt is mostly "NEVER mark as contradicted unless…" rules.
   - MemOS added a second LLM pass that audits every UPDATE for "Entity Consistency", after UPDATEs were applied to the wrong entity.
   - In HaluMem, update *hallucination* is below about 1.2% for every system, while *omission* is 37–94%.
3. **Every system prefilters cheaply before any model.**
   - Methods: hash, exact normalized name, identity keys, MinHash with an entropy gate (Graphiti J ≥ 0.9), cosine thresholds (0.95 Mem0, 0.85 Cognee, 0.8 MemOS and HippoRAG), hybrid top-5 to 10.
   - The LLM judges only that residue. In Graphiti it is the "small" model tier, which already has per-prompt model routing.
4. **Schema keys turn supersession into a rule.** Examples: Cognee `functional_relationships`, Memobase topic/subtopic slots, Supermemory `isStatic`. The model is needed mainly to:
   - (a) map a free-text fact onto an existing key or entity;
   - (b) classify unkeyed pairs into {update, contradiction, compatible, duplicate}. MemOS's `contradictory | redundant | independent` prompt is exactly this shape.
5. **Small models are good enough for most of these decisions:**
   - **Grounded contradiction:** 0.1–0.8B checkers (HHEM, MiniCheck-FT5, FactCG-DeBERTa-L 0.4B) are within about 1 point of GPT-4/4o on LLM-AggreFact at about 450× lower cost.
   - **Entity matching:** fine-tuned ~125M Ditto beats zero-shot GPT-4 on 3 of 6 EM benchmarks and trails it by 4–5 F1 on hard or unseen products.
   - **Update-relevant extraction:** MemReader-0.6B reaches HaluMem update-correct 82.7 vs Zep 47.3 and Mem0 25.5.
   - **But:** small *generative* maintainers fail. Qwen2.5-3B with GRPO reaches only 16.7% supersession accuracy in "Supersede".
   - **kleos's own D1 runs:** zero-shot d1-omni-600M and DM-JEPA are near chance on supersession (AUROC 0.46–0.63), and cosine is *below* chance (0.42). Cosine is strong for duplicates (0.89) and entities (0.94).
   - **Supersession needs a *trained* pair classifier.** There is no zero-shot shortcut.
6. **Importance:** the Generative Agents 1–10 poignancy prompt has never been validated against human judgement. A learned linear multi-factor score beats recency and uniform weights on LongMemEval retention (0.770 vs 0.368 and 0.657). Use a cheap learned score.
7. **Benchmarks:**
   - LongMemEval KU (78 questions) and the update stage of HaluMem are the most direct tests.
   - LoCoMo has **no** knowledge-update category.
   - BEAM's contradiction-resolution category rewards *surfacing* both claims, which no destructive policy can do.
   - Memory systems score *below BM25* on MemoryAgentBench FactConsolidation (Mem0 18, Zep 7, BM25 48).
   - No paper cleanly isolates decision-model quality → KU accuracy; Memory-R1 (+8–10 J) and MemReader (KU 82 → 91 from 0.6B → 4B-GRPO) come closest.
8. **Recommendation:**
   - **Tier 0:** rules, hashes, keys, MinHash, cosine. Synchronous.
   - **Tier 1:** a fine-tuned 0.1–0.4B cross-encoder `decide()` with temperature-scaled probabilities and risk-controlled thresholds (auto / tentative / none). Synchronous within about a 20 ms budget.
   - **Tier 2:** an LLM fallback using a Graphiti-style typed prompt and log-prob confidence. Async, via live subscriptions.
   - **Effects:** every effect is a reversible `supersedes`/`merged_into` row carrying model version and probability. Never delete on a model decision, and never merge entities transitively.
   - **Training data:** VitaminC, WikiFactDiff, MQuAKE, PAWS and the Leipzig EM sets (all commercial-OK, some share-alike). Hold out LongMemEval-KU-mined pairs and the 66 items.

---

## 1. What each system decides, and how (from source)

### 1.1 Mem0: the old ADD/UPDATE/DELETE/NONE design is gone from the write path

Clone: `mem0/` at commit `0724291` (2026-10-14).

**The legacy decision (Mem0 v1 and v2, and the arXiv 2504.19413 paper)**

- The prompt is `DEFAULT_UPDATE_MEMORY_PROMPT` in `mem0/mem0/configs/prompts.py:176`. It is wrapped by `get_update_memory_messages()` at `prompts.py:406`.
- One LLM call receives the retrieved top-k old memories, keyed by small integer IDs, plus the newly extracted facts. For every memory it returns `event ∈ {ADD, UPDATE, DELETE, NONE}`, with `old_memory` filled in for UPDATE.
- The prompt's rules:
  - **Update:** use it when the memory is "already present but the information is totally different", or to keep the version "which has the most information". For example, "likes to play cricket" becomes "loves to play cricket with friends".
  - **Delete:** use it when "the retrieved facts contain information that contradicts the information present in the memory".
- Deletion is therefore **destructive and decided by one LLM call with no confidence score.**
- In the current Python SDK this prompt is **dead code**. `main.py` no longer calls `get_update_memory_messages`. The TypeScript SDK still imports it (`mem0-ts/src/oss/src/memory/index.ts:22`), but its add path uses the additive prompt (line 912).

**The current decision (Mem0 OSS v3, Python SDK v2.0.0 / TS v3.0.0, 2026-04-14)**

- `docs/changelog/highlights.mdx:222-238` and `docs/migration/oss-v2-to-v3.mdx:16-40` say the write path is now "Single-pass ADD-only (one LLM call, no UPDATE/DELETE)". They also say "Memories accumulate; nothing is overwritten or deleted."
- Self-reported gains:
  - LoCoMo rose from 71.4 to 91.6, and LongMemEval from 67.8 to 93.4.
  - "Temporal reasoning … improves from 51% to 93%".
  - Extraction latency was "cut … roughly in half".
- The prompt is `ADDITIVE_EXTRACTION_PROMPT` (`prompts.py:468`). Instead of deciding an operation, the LLM:
  - emits new memories plus `linked_memory_ids` pointing at related existing memories, with linking criteria "Same entity/topic", "Updated preference", "Continuation" and "Contradiction";
  - is told to write transitions explicitly ("User switched from almond milk to oat milk…").
- The pipeline is in `mem0/mem0/memory/main.py:881-1200`:
  1. Retrieve the top-10 existing memories by embedding (`main.py:~925`).
  2. Map UUIDs to small integers, an "anti-hallucination" step.
  3. Make one LLM call with JSON output.
  4. Batch-embed the results.
  5. **Deduplicate on an MD5 hash of the exact text** (`main.py:1007-1034`).
  6. Link entities. Entities come from **spaCy, not an LLM** (`mem0/utils/entity_extraction.py`). They are matched by exact normalized text, or by **cosine ≥ 0.95** against an entity vector store (`main.py:617`, `main.py:1171`).
- The default LLM is `gpt-5-mini` (`mem0/llms/openai.py:40`).

**Where supersession went: the hosted "Dream" feature** (`docs/platform/features/dream.mdx`; closed source, logic not inspectable)

- **Supersede** runs "as memories are added". It marks the older memory superseded and links it to the newer one. Superseded memories are "not deleted and not hidden by default", and `latest_only=true` returns only active ones.
- **Merge** handles duplicates. The merged record is hidden and retained.
- **Synthesis** is a scheduled, opt-in job that writes "pattern memories" with links to their evidence.
- The docs state: "Nothing Dream does is destructive."
- Mem0 does not say which model or what thresholds Dream uses. **UNVERIFIED.**

**Also from Mem0:** "Memory Decay" (`docs/platform/features/memory-decay.mdx`) is a search-time multiplier clamped to **0.3×–1.5×**, based on access recency. It is a "soft ranking bias, never a filter".

**What this means for kleos.** The best-known ADD/UPDATE/DELETE design has been abandoned by its own authors in favour of:

- append-only writes;
- cheap deduplication (a hash plus an entity cosine of 0.95);
- non-destructive supersession plus `latest_only` at read time.

That is exactly the bitemporal model kleos already has. The only remaining question is *who decides what supersedes what*.

### 1.2 Zep / Graphiti: hybrid candidate search, deterministic fast paths, a "small" LLM for dedup and contradiction, and temporal arithmetic for invalidation

Clone: `graphiti/` at commit `d50741d` (2026-10-15).

**Temporal model (bitemporal)**

Each `EntityEdge` (a fact) carries four timestamps:

- `valid_at` and `invalid_at`: when the fact was true in the world;
- `created_at` and `expired_at`: when the system learned it and when it retired it.

Zep's paper, arXiv [2501.13956](https://arxiv.org/abs/2501.13956), describes the same model. The arithmetic is in `graphiti_core/utils/maintenance/edge_operations.py:715-783` (`_apply_temporal_invalidation`, `resolve_edge_contradictions`):

- An old contradicted edge gets `invalid_at = new.valid_at` and `expired_at = now`. **It is never deleted.**
- If the old edge's validity window ends before the new one starts, it is skipped.
- If a candidate starts *later* than the new edge, the **new** edge is the one expired. This handles out-of-order ingestion.
- `valid_at` and `invalid_at` come from a separate LLM call, `extract_edges.extract_timestamps`, made with `ModelSize.small` (`edge_operations.py:786-832`).

**Edge (fact) dedup and contradiction** (`edge_operations.py:424-640`, `864-1000`)

1. **Batch fast path:** facts with identical `(source, target, normalized fact text)`, including the reversed direction, are collapsed (`edge_operations.py:490-516`).
2. **Candidate retrieval:** two hybrid searches (`EDGE_HYBRID_SEARCH_RRF`: BM25 plus cosine with reciprocal-rank fusion, default limit 10, `search/search_config.py:29`):
   - (a) edges between the **same node pair**, which are duplicate candidates;
   - (b) edges from the whole group, which are invalidation candidates.
3. **Per-edge fast path:** the same endpoints plus identical normalized fact text mean "reuse the existing edge" with no LLM call (`edge_operations.py:941-952`).
4. **One LLM call** per remaining edge, made with `ModelSize.small`. The prompt is `graphiti_core/prompts/dedupe_edges.py:resolve_edge`. It returns `duplicate_facts: list[int]` and `contradicted_facts: list[int]`.

The prompt's guardrails are worth copying as features or labels for `decide()`. Much of its text exists to suppress **false contradictions**:

- System message: "NEVER mark facts as contradicted unless the new fact explicitly makes the old fact false."
- "NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers."
- "A contradiction requires explicit evidence that the old fact stopped being true or was replaced… Do NOT infer contradiction from recency alone, episode dates… or a new event involving the same entities."
- "Do NOT treat additional activity, progress, status, or project-update facts as contradictions of role, ownership… facts."
- Worked examples:
  - "works … as a software engineer" followed by "… as a senior engineer" is a **contradiction, not a duplicate**;
  - "Bob ran 5 miles on Tuesday" and "ran 3 miles on Wednesday" are **neither**.

**Entity (node) dedup** (`graphiti_core/utils/maintenance/dedup_helpers.py`, `node_operations.py`)

The deterministic cascade runs before any LLM:

1. Exact normalized name (lower-cased, whitespace collapsed). A single match is accepted. Several matches make the entity unresolved and send it to the LLM.
2. Configured identity properties per entity type (`_resolve_with_identity_properties`).
3. MinHash/LSH over character 3-gram shingles: 32 permutations, band size 4, accepted at **Jaccard ≥ 0.9**. This is **gated by name entropy**: Shannon entropy ≥ 1.5, and length ≥ 6 or ≥ 2 tokens. Names that are short or have low entropy ("Sam", "AI") are "deferred to the LLM instead of trusting fuzzy similarity" (`dedup_helpers.py:29-34`, `52-76`).
4. Everything else goes to the LLM prompt `prompts/dedupe_nodes.py:nodes`. That prompt returns `duplicate_candidate_id` or −1, gets 120-character candidate summaries as context, and carries the "Java (language) vs Java (island)" counter-example.

**Models and routing**

- `ModelSize.small` is used for edge dedup and contradiction, timestamps and attributes.
- The default small model is `gemini-2.5-flash-lite` for Gemini (`llm_client/gemini_client.py:48`). The OpenAI client's defaults read `gpt-6-luna` (`openai_base_client.py:35-36`).
- Graphiti supports **per-prompt model routing**: `graphiti_types.py:95-110`, `llm_runtime.resolve_model(prompt_name)`. This is the natural hook for replacing one decision with a small model.
- Graphiti already ships a **GLiNER2 client** (`llm_client/gliner2_client.py`; "205M-340M params … on CPU", `fastino/gliner2-base-v1`, threshold 0.5). It uses GLiNER2 for entity extraction only and "all other operations … are delegated" to the LLM.

### 1.3 Letta (MemGPT): an agent edits its own memory; "sleep-time" and "reflection" run in the background

- Letta's V1 server is archived. Clone `letta-archive/` (branch `archive`); the live code is `letta-code/`.
- The sleep-time agent prompt is `letta-archive/letta/prompts/system_prompts/sleeptime_v2.py`. A background agent with memory-editing tools (precise edits plus `rethink` for whole blocks) makes blocks "comprehensive, readable, and up to date", removes "redundant and outdate[d] information", and avoids relative dates. It may "call the finish tool directly" when there is nothing to change.
- **The memory decision is an open-ended LLM tool loop with no typed decision.**
- The trigger is every `sleeptime_agent_frequency` turns (`letta-archive/letta/groups/sleeptime_multi_agent_v4.py:139-145`).
- In current Letta Code the equivalent is a "reflection" subagent (`letta-code/src/agent/subagents/builtin/reflection-v2.md`):
  - It runs over a git-backed markdown "MemFS" and is triggered by `"step-count" | "compaction-event"` (`src/reflection-settings.ts`).
  - It uses `model: inherit`, which means the same model as the main agent.
  - It edits files and commits them, so history is kept through git.
- Paper: Lin et al., "Sleep-time Compute", arXiv [2504.13171](https://arxiv.org/abs/2504.13171). It is about pre-computing inferences offline, not about update decisions.

### 1.4 A-Mem: "memory evolution" by LLM, with no contradiction handling

Clone: `A-mem/`.

- The prompt is `agentic_memory/memory_system.py:126-158`.
- For each new note, the k=5 nearest neighbours (`all-MiniLM-L6-v2`, ChromaDB) go to one LLM call. The call returns:
  - `should_evolve` (bool);
  - `actions ⊂ {strengthen, update_neighbor}`;
  - `suggested_connections`;
  - rewritten neighbour `context` and `tags`.
- The default LLM is `gpt-4o-mini` (`memory_system.py:96`).
- There is **no supersede, delete or contradiction operation.** "Evolution" rewrites neighbours' metadata, which is a destructive overwrite of tags and context with no history.
- Paper: arXiv [2502.12110](https://arxiv.org/abs/2502.12110). For its model-size results, see §2.

### 1.5 MemOS: an embedding gate, a 3-way LLM relation classifier, LLM fusion, and an LLM safety check on UPDATEs

Clone: `MemOS/` at commit `a7367d0` (2026-09-22).

**Background reorganizer** (`src/memos/memories/textual/tree_text_memory/organize/reorganizer.py`)

This is a priority queue plus consumer threads, so it is **async and off the write path**.

**Detector** (`organize/handler.py:22-75`)

- Candidates come from `search_by_embedding`, top_k=5, gated by `EMBEDDING_THRESHOLD = 0.8`.
- The LLM then labels each pair with `MEMORY_RELATION_DETECTOR_PROMPT` (`templates/tree_reorganize_prompts.py:197`), which returns **exactly one of `contradictory | redundant | independent`**. **This is the cleanest published example of a `decide()`-shaped choice.**

**Resolver** (`MEMORY_RELATION_RESOLVER_PROMPT`, `tree_reorganize_prompts.py:211`)

- The LLM fuses the two statements, "prioritizing more recent information", or outputs `No` when the conflict cannot be resolved.
- On `No`, `_hard_update` keeps the newer item and **deletes the older node** (`handler.py:~131-147`, `graph_store.delete_node`). That is destructive.
- On success, both originals become `status: archived` with `MERGED_TO` edges, which is non-destructive.

**Feedback path** (`templates/mem_feedback_prompts.py`)

- `UPDATE_FORMER_MEMORIES` proposes ADD/UPDATE/NONE.
- Then `OPERATION_UPDATE_JUDGEMENT` (line 666) is a **second LLM pass that audits every proposed UPDATE**. It returns `INVALID | NONE | UPDATE_APPROVED` after an "Entity Consistency Check", a "Semantic Relevance Check" and a "Context Preservation Check".
- Its worked example is an UPDATE that would overwrite "Gladys Liu is an Italian citizen" with "Mission Terra is from Germany", which is rejected as INVALID.
- **This documents the dominant failure mode of LLM update operations: applying them to the wrong entity.** It is the case for a cheap "same subject?" `decide()` check before any supersession.

**Dream lifecycle** (`src/memos/dream/maintenance.py`)

This is an unimplemented placeholder. Its docstring suggests TTL/staleness, usefulness score and feedback invalidation, with "gradual decay … rather than hard-deleting".

### 1.6 Cognee: schema-declared functional relations give rule-based supersession; LLM contradiction checks are advisory only; entity merge is pure cosine

Clone: `cognee/` at commit `f154b32` (2026-10-13).

**Rule-based supersession** (`cognee/tasks/graph/resolve_temporal_contradictions.py:55-120`)

- Supersession runs only for relationships declared in `functional_relationships`, such as `{"ceo_of"}`: single-valued relationships.
- Older assertions of the same `(subject, functional relation)` are tagged superseded by timestamp. **No LLM is involved.**
- The task is a no-op by default, because "most cognee relationships are legitimately many-valued".

**LLM contradiction detection** (`cognee/tasks/graph/detect_contradictions.py:144-235`)

- It covers the 1-hop neighbourhood of touched nodes, up to `contradiction_max_facts=500` facts.
- The prompts are `infrastructure/llm/prompts/detect_contradictions_{system,user}.txt`.
- The LLM returns pairs with a self-reported `confidence` from 0 to 1, filtered at `contradiction_confidence_threshold=0.5` (`modules/cognify/config.py:25-26`).
- The output is **only a `contradicts` edge with a reason. Nothing is invalidated.** Errors never fail ingestion.

**Entity identity**

- At write time, entity identity is `uuid5(lower(name).replace(" ","_"))` (`infrastructure/engine/utils/generate_node_id.py`), so different spellings never merge.
- The memify `consolidate_entities` task (`cognee/tasks/memify/consolidate_entities.py`) clusters entity-name embeddings at **cosine ≥ 0.85** with union-find. It then rewires edges and **deletes the duplicates** (a `dry_run` option exists).
- **Union-find over a single cosine threshold chains transitively** (A≈B≈C), which is a known risk.

**Defaults:** the LLM is `openai/gpt-5.6-luna` (`infrastructure/llm/config.py:110`). A GLiNER demo task exists (`tasks/graph/gliner_demo/`).

### 1.7 LangMem: an LLM tool loop, with deletes disabled by default

Clone: `langmem/`.

- The instructions are `_MEMORY_INSTRUCTIONS` (`src/langmem/knowledge/extraction.py:185-206`): "Extract & Contextualize → Compare & Update → Synthesize & Reason". They ask for "confidence levels (p(x))" and to "Remove incorrect or redundant memories".
- The manager is constructed with `enable_inserts=True, enable_updates=True, enable_deletes=False` (`extraction.py:224-235`). **Deletes are off unless you opt in.**
- Updates are trustcall-style document patches.
- `ReflectionExecutor` (`src/langmem/reflection.py`) defers processing with `after_seconds`, a debounce for async consolidation.

### 1.8 Memobase: slot-keyed profiles with an APPEND/UPDATE/ABORT LLM choice, processed in buffered batches

Clone: `memobase/`.

- The profile is keyed by `topic/subtopic`, a fixed slot schema that plays the role of kleos's supersession key.
- The merge prompt is `src/server/api/memobase_server/prompts/merge_profile.py` (`MERGE_FACTS_PROMPT`). Per slot, it emits `APPEND`, `UPDATE <rewritten memo>` or `ABORT`, after a free-text THOUGHT. It keeps `[mentioned on …]` time annotations and caps memos at 5 sentences.
- Batching (`env.py:85-120`): chats are buffered until 1024 tokens or a 1-hour flush, then processed. The default model is `best_llm_model = "gpt-4o-mini"`.
- **Async by design.** UPDATE rewrites the memo, so history depends on the storage layer.

### 1.9 Supermemory: typed relations, with an `isLatest` version chain

Clone: `supermemory/`. Only clients and SDKs are open; the engine is closed.

- `skills/supermemory/references/architecture.md` describes three relation types between memories:
  - **Updates**: supersedes, with `isLatest`;
  - **Extends**: enriches;
  - **Derives**: inferred.
- It also describes `isStatic` (permanent facts "not subject to temporal updates") and a version chain whose default read is the latest version.
- The decision model, prompts and thresholds are **not public. UNVERIFIED.**

### 1.10 MIRIX: an LLM router over six memory types

Clone: `MIRIX/`.

- `mirix/prompts/system/base/meta_memory_agent.txt` is a "Meta Memory Manager" LLM. For batched messages it chooses which of `core, episodic, procedural, resource, knowledge_vault, semantic` to update.
- Each type then has its own LLM agent with edit tools.
- The decision is a **multi-label routing choice**, a natural `decide(choice)` target. The per-type update and dedup logic is LLM tool use.

### 1.11 HippoRAG 2: embedding-threshold synonym edges instead of entity merges; an LLM "recognition memory" filter

Clone: `HippoRAG/`.

- **Entity resolution is soft.** Synonymy edges link phrase nodes whose embeddings have kNN similarity ≥ `synonymy_edge_sim_threshold = 0.8` (`src/hipporag/utils/config_utils.py:172`), with `synonymy_edge_topk = 2047`. Nodes are never merged.
- The default embedder is `nvidia/NV-Embed-v2`; the default LLM is `gpt-4o-mini` for OpenIE.
- At query time, an LLM filter keeps or rejects retrieved triples (`prompts/filter_default_prompt.py`). A DSPy-optimized prompt is shipped for Llama-3.3-70B (`prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json`).
- Paper: arXiv [2502.14802](https://arxiv.org/abs/2502.14802).

### 1.12 Other 2025–2026 systems relevant to decisions (web)

- **Memory-R1** (arXiv [2508.19828](https://arxiv.org/abs/2508.19828), ACL 2026, [anthology](https://aclanthology.org/2026.acl-long.583/)) trains a Memory Manager over {ADD, UPDATE, DELETE, NOOP} with PPO/GRPO, using 3B–14B backbones and 152 training QA pairs. Numbers are in §2.
- **AgeMem**, "Agentic Memory" (arXiv [2601.01885](https://arxiv.org/abs/2601.01885)), uses RL to train ADD/UPDATE/DELETE tool use.
- **Auto-Dreamer** (arXiv [2605.20616](https://arxiv.org/abs/2605.20616), May 2026) is a GRPO-trained *offline* consolidator.
  - It reads a memory region as read-only evidence, plus provenance-linked source trajectories, and writes a compact replacement set that "supersedes the original region".
  - Self-reported: +7 points on ScienceWorld with a 12× smaller memory bank.
  - The model size is not stated on the abstract page.
- **Hindsight** (arXiv [2512.12818](https://arxiv.org/abs/2512.12818)) separates world facts, experiences, entity summaries and **beliefs with confidence scores** that evolve.
  - Self-reported LongMemEval: 83.6% with a 20B open model, against 39.0% for full context on the same model; 91.4% with a larger backbone.
- **Mastra Observational Memory** ([research post](https://mastra.ai/research/observational-memory), [docs](https://mastra.ai/docs/memory/observational-memory)):
  - An Observer LLM appends timestamped notes with emoji priority labels (red, yellow and green circles). This is LLM-emitted **importance as a 3-level label**.
  - A Reflector condenses the notes when observations exceed about 40k tokens.
  - The default model is `gemini-2.5-flash`.
  - Self-reported LongMemEval: 94.87% with gpt-5-mini and 84.23% with gpt-4o.
  - [The Decoder](https://the-decoder.com/mastras-open-source-ai-memory-uses-traffic-light-emojis-for-more-efficient-compression/) reports that observation runs synchronously and blocks the conversation.

### 1.13 Cross-system synthesis table

| System | Dedup | Entity resolution | Contradiction / supersession | Destructive? | Model | Sync / async |
|---|---|---|---|---|---|---|
| Mem0 v3 OSS | MD5 exact text; LLM told to skip semantic dups | spaCy NER, exact or cosine ≥0.95 | none in OSS; LLM emits `linked_memory_ids` incl. "Contradiction" | no (ADD-only) | gpt-5-mini default | sync extraction |
| Mem0 Platform Dream | "Merge" | – | "Supersede" marks old superseded; `latest_only` | no | not disclosed | on add, plus scheduled synthesis |
| Mem0 legacy (paper) | LLM NONE/UPDATE | – | LLM DELETE | **yes** | gpt-4o-mini in paper (see §2) | sync |
| Graphiti/Zep | exact fast path, then LLM `duplicate_facts` | exact → identity props → MinHash J≥0.9 with entropy gate → LLM | LLM `contradicted_facts` (small model) + date arithmetic | no (`invalid_at`, `expired_at`) | "small" tier | sync in `add_episode` |
| MemOS | cos ≥0.8 gate → LLM `redundant` | – | LLM `contradictory` → LLM fuse; LLM UPDATE auditor | partly (hard delete if unresolvable) | configurable | async reorganizer |
| Cognee | – | uuid5(name); later cosine ≥0.85 union-find | rules on declared functional relations; LLM flags `contradicts` (conf ≥0.5) | rules non-destructive; entity merge deletes | gpt-5.6-luna default | pipeline tasks |
| A-Mem | – | – | none (evolve tags/links) | overwrites neighbour metadata | gpt-4o-mini | sync |
| LangMem | LLM | – | LLM patch; deletes off by default | optional | any | sync or debounced |
| Memobase | LLM ABORT | slot key | LLM UPDATE per slot | rewrite | gpt-4o-mini | buffered async |
| Letta | LLM agent | – | LLM agent edits blocks/files | git history | inherit | background |
| Supermemory | ? | ? | `Updates` relation, `isLatest` | no | closed | ? |
| HippoRAG 2 | – | synonym edges cos ≥0.8 | – | n/a | gpt-4o-mini + NV-Embed-v2 | offline index |

**Patterns common to these systems**

1. **Everyone prefilters candidates cheaply**, using a hash, an exact normalized name, MinHash, BM25 or vector top-k plus a threshold. The LLM sees only 5–10 candidates.
2. **The trend is away from destructive operations.**
   - Mem0 went ADD-only, with supersede and merge as flags.
   - LangMem turns deletes off by default.
   - Graphiti and Cognee only expire or flag.
   - MemOS added an LLM auditor for UPDATEs.
3. **Prompts have grown long lists of "do NOT mark as contradiction" rules.** In practice the costly error is **false supersession**: different events, a different entity, or added detail mistaken for replacement.
4. **Schema keys remove most of the decision.** Cognee's functional relations, Memobase's topic/subtopic slots, Supermemory's `isStatic` and Graphiti's identity properties all work this way. Where the key is known, supersession is a rule. A model is needed only to map free text onto a key, or for unkeyed facts.
5. **No open-source system uses a calibrated confidence for these decisions.** Cognee filters on the LLM's *self-reported* confidence at 0.5; the rest use hard labels.

---

## 2. Benchmarks and published scores

**Caveat for this whole section.** Scores from different harnesses are not comparable. The answer model, the judge and the prompts move results by 10–30 points. Redis's independent rerun (Sep 2026, [report](https://redis.io/resources/agent-memory-benchmark-report/)) found published-vs-rerun gaps from +3 to −28 points. In that rerun Mastra reproduced (84.2), while Google Vertex Memory Bank fell from a claimed ~70 to 42.4. "VENDOR" marks a system maker's own claim.

### 2.1 What the benchmarks measure

**LongMemEval** (Wu et al., [arXiv 2410.10813](https://arxiv.org/abs/2410.10813), ICLR 2025; MIT; HF `xiaowu0162/longmemeval-cleaned`)

- 500 questions.
- Question types: single-session-user 70, single-session-assistant 56, single-session-preference 30, multi-session 133, **knowledge-update 78**, temporal-reasoning 133.
  - These counts come from secondary sources, but they match the denominators in Mastra's table exactly.
- **Abstention:** 30 false-premise questions (`question_id` ending in `_abs`), spread across those types.
- Variants: `_S` (about 115k tokens), `_M` (about 500 sessions), and `oracle` (evidence sessions only).
- A cleaned release (2025-09) removed noisy sessions.
- Judging: a GPT-4o judge, with more than 97% agreement with humans.
- The paper notes ChatGPT's memory "tended to overwrite crucial information".

**LoCoMo** (Maharana et al., [arXiv 2402.17753](https://arxiv.org/abs/2402.17753); **CC BY-NC 4.0**)

- Vendors use the 10-conversation `locomo10.json`, with 1,540 questions in categories 1–4 (single-hop, multi-hop, temporal, open-domain).
- Category 5 (adversarial) is excluded: 444 of its 446 gold answers are empty.
- **LoCoMo has no knowledge-update category** ([Zep](https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)), so it says little about supersession.
- [Penfield's audit](https://penfieldlabs.substack.com/p/we-audited-locomo-64-of-the-answer) (Apr 2026) found:
  - 99 answer-key errors (6.4%);
  - a gpt-4o-mini judge accepted 62.81% of deliberately wrong but on-topic answers.
- Category labels are swapped between the A-Mem and Mem0 papers, so per-category comparisons across papers are unreliable.

**BEAM**, "Beyond a Million Tokens" (Tavakoli et al., [arXiv 2510.27246](https://arxiv.org/abs/2510.27246), ICLR 2026; data CC BY-SA 4.0; HF `Mohammadta/BEAM`)

- 100 conversations and 2,000 questions, at 128K, 500K, 1M and 10M tiers.
- 10 abilities, including **knowledge update**, **contradiction resolution**, temporal reasoning, event ordering and abstention. That is roughly 200 questions per ability (inferred).
- Scoring: nugget scoring by an LLM judge.
- **The contradiction-resolution ideal answer *flags* the conflict, cites both claims and asks which is correct. It does not silently take the newer claim.** So BEAM-CR rewards keeping both facts visible, and destructive supersession scores 0 there.

**MemoryAgentBench** (Hu, Wang & McAuley, [arXiv 2507.05257](https://arxiv.org/abs/2507.05257), ICLR 2026; HF `ai-hyz/MemoryAgentBench`)

- Four competencies. "Conflict Resolution" was renamed "Selective Forgetting" in v4.
- FactConsolidation-SH/MH is built from MQuAKE counterfactual edits, with the newer fact later in a 6K–262K-token context.

**Other benchmarks focused on updates and contradiction (2025–2026):**

- **HaluMem** ([arXiv 2511.03506](https://arxiv.org/abs/2511.03506), HF `IAAR-Shanghai/HaluMem`) scores **extraction, updating and QA separately**.
  - 3,122 update memories.
  - QA includes Dynamic Update (180) and Memory Conflict (769).
- **STALE** ([2605.06527](https://arxiv.org/abs/2605.06527)): 1,200 queries on implicit conflicts; the best model reaches 55.2%.
- **StateMemBench** ([2608.19652](https://arxiv.org/abs/2608.19652)): 234 scenarios, graded as current vs superseded state. Existing systems score 0.15–0.21 current-state accuracy.
- **MEMTRACK** ([2510.01353](https://arxiv.org/abs/2510.01353)): Slack, Linear and Git conflicts; GPT-5 reaches 60%.
- **PersonaMem** ([2504.14225](https://arxiv.org/abs/2504.14225)): evolving preferences.
- **Supersede** ([2606.27472](https://arxiv.org/abs/2606.27472)): built on the LongMemEval knowledge-update subset.
- **Not checked:** EvoMemBench, LoCoMo-Plus, "Ground Truth First" (2607.21962). The last reportedly finds that rankings flip between 3-week and 9-week histories.

### 2.2 Which sub-tasks depend on correct supersession and contradiction decisions

| Benchmark | Sub-task | Dependence on the decision |
|---|---|---|
| LongMemEval | knowledge-update (78) | **Direct**: answer must use the latest value |
| LongMemEval | temporal-reasoning (133) | Partial: needs validity times; destructive overwrite loses history |
| LongMemEval | abstention (30) | Partial: a stale premise must be rejected |
| BEAM | knowledge update (~200) | **Direct** |
| BEAM | contradiction resolution (~200) | **Direct, opposite policy**: surface both claims |
| BEAM | event ordering / temporal | Need retained history |
| MemoryAgentBench | FactConsolidation-SH/MH | **Direct**; MH needs chains over updated facts |
| HaluMem | update stage; Dynamic Update (180); Memory Conflict (769) | **Direct**, measured per stage |
| STALE, StateMemBench, PersonaMem, MEMTRACK | all | **Direct** |
| LoCoMo | none | No KU category |

**There is a tension between benchmarks.**

- Some reward **latest wins**: LongMemEval KU, FactConsolidation, StateMemBench.
- Others reward **keeping the conflict visible**: BEAM-CR, STALE's premise resistance.
- Non-destructive bitemporal supersession serves both, provided the read API can return the current value *and* surface "this changed from X on date D". Kleos's model does this; destructive DELETE does not.

### 2.3 Published scores (selected; full tables in the cited sources)

**LongMemEval_S per category** (KU = knowledge-update, TR = temporal-reasoning)

| System (reader) | Overall | KU | TR | Source |
|---|---|---|---|---|
| Full context, gpt-4o | 60.2 | 78.2 | 45.1 | [Zep paper](https://arxiv.org/abs/2501.13956) |
| Zep, gpt-4o | 71.2 | 83.3 | 62.4 | Zep paper (VENDOR) |
| **Oracle evidence, gpt-4o** | 82.4 | **85.9** | 76.7 | [Emergence](https://www.emergence.ai/blog/sota-on-longmemeval-with-rag) |
| Oracle evidence, o3 | 92.0 | 92.3 | 92.5 | Emergence |
| Mastra OM, gpt-4o | 84.23 | 85.9 | 85.7 | [Mastra](https://mastra.ai/research/observational-memory) (VENDOR; Redis reproduced 84.2) |
| Mastra OM, gpt-5-mini | 94.87 | 96.2 | 95.5 | Mastra (VENDOR) |
| Supermemory, gpt-4o | 81.6 | 88.5 | 76.7 | via [Hindsight paper](https://arxiv.org/abs/2512.12818) (VENDOR) |
| Hindsight, OSS-20B | 83.6 | 84.6 | 79.7 | Hindsight paper (VENDOR) |
| Hindsight, Gemini-3 | 91.4 | 94.9 | 91.0 | Hindsight paper (VENDOR) |
| MemOS (4o-mini) | 77.8 | 74.3 | 77.4 | [MemOS paper](https://arxiv.org/abs/2507.03724) (VENDOR) |
| Mem0 legacy, in MemOS harness | 66.4 | 66.7 | 72.2 | MemOS paper (competitor-run) |
| Memobase, in MemOS harness | 72.4 | 89.7 | 75.9 | MemOS paper |
| MIRIX, in MemOS harness | 43.5 | 52.6 | 25.6 | MemOS paper |
| LightMem (4o-mini) | 64.3–68.6 | 83.1 | 67.2 | [LightMem](https://arxiv.org/abs/2510.18866) |
| MemReader-4B-GRPO (4.1-mini reader) | 83.0 | 91.0 | 84.2 | [MemReader](https://arxiv.org/abs/2604.07877) (verified in paper HTML) |
| Zep 2026 (gpt-5.4) | 90.2 | 93.6 | 90.2 | [getzep.com/research](https://getzep.com/research) (VENDOR) |
| Mem0 2026, ADD-only | 94.4 | 93.6 | 97.0 | [Mem0 docs](https://docs.mem0.ai/core-concepts/memory-evaluation) (VENDOR; managed platform; "models vary") |

- Other overall-only claims (VENDOR): Backboard 93.4, MemMachine 93.0, Hindsight later 94.6, Chronos 92.6, Honcho 90.4.
- **Note:** Mem0's own pages disagree. The README gives LoCoMo 92.5 / LongMemEval 94.4; the changelog and migration guide give 91.6 / 93.4.

**LoCoMo, Mem0 paper** ([arXiv 2504.19413](https://arxiv.org/abs/2504.19413); **gpt-4o-mini for extraction, update and answering**; J = LLM judge)

- Overall J: Mem0 66.88, Mem0g 68.44, Zep (Mem0's run) 65.99, LangMem 58.10, OpenAI 52.90, A-Mem (rerun) 48.38, full context 72.90.
- Latency, p50 / p95 total: Mem0 0.708 / 1.440 s; Zep 1.292 / 2.926 s; full context 9.87 / 17.12 s.
- Zep's corrected run of itself: 75.14 ([Zep blog](https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)).
- Other LoCoMo results:
  - Letta filesystem agent: 74.0 ([Letta](https://www.letta.com/blog/benchmarking-ai-agent-memory)).
  - MemOS harness: MemOS 75.80, Memobase 72.01, Mem0 64.57, MIRIX 64.33, Zep 59.22, Supermemory 55.34.
  - MIRIX's own paper: 85.38.
- **The same system moves 10–20 points between harnesses.**

**BEAM** ([paper](https://arxiv.org/abs/2510.27246))

- **Contradiction resolution is about 0 for every baseline** (0.006–0.050 at 100K; 0.000–0.050 at 10M).
- Knowledge update ranges 0.125–0.450 at 100K.
- Mem0 2026 (VENDOR, [docs](https://docs.mem0.ai/core-concepts/memory-evaluation)):
  - 1M tier: KU 65.0, **CR 35.7**, abstention 52.5.
  - 10M tier: KU 75.0, CR 32.5, TR 16.3.

**MemoryAgentBench FactConsolidation** (paper v4 Table 3; vendor-independent)

| | FC-SH | FC-MH |
|---|---|---|
| GPT-4o long context | 60 | 5 |
| GPT-5-mini long context | 78 | 28 |
| BM25 | 48 | 3 |
| HippoRAG-v2 | 54 | 5 |
| Letta/MemGPT | 28 | 3 |
| Cognee | 28 | 3 |
| **Mem0** | **18** | 2 |
| MIRIX | 14 | 2 |
| **Zep** | **7** | 3 |

- **Memory systems with write-time update logic score *below plain BM25*** on single-hop consolidation.
- At 6K tokens, o4-mini reaches FC-SH 100 / FC-MH 80, so the task is solvable when the context is short.

**HaluMem-Medium, update stage** (Correct / Omission %, [paper](https://arxiv.org/abs/2511.03506), MemReader rows verified in [2604.07877](https://arxiv.org/html/2604.07877))

| System | Update correct | Update omission |
|---|---|---|
| Memobase | 5.20 | 94.25 |
| Supermemory | 16.37 | 82.47 |
| Mem0 | 25.50 | 74.02 |
| Zep | 47.28 | 52.31 |
| MemOS | 62.11 | 37.48 |
| **MemReader-0.6B** | **82.69** | 16.51 |
| MemReader-4B-SFT | 90.78 | 8.74 |
| MemReader-4B-GRPO | 94.55 | 5.12 |

- Update hallucination is below about 1.2% for every system. **The failure is omission, not wrong updates.** HaluMem attributes it mostly to the extraction step missing the pre-update fact.
- **MemReader is an extraction model.** Its actions are `add_memory`, `buffer_memory`, `search_memory` and `ignore_memory`, and "add_memory writes or updates structured entries". It has no separate supersession classifier.

### 2.4 Does any paper ablate the decision model (LLM size vs accuracy)?

**Partially. No paper isolates "supersession classifier quality → KU accuracy" with everything else held fixed.**

- **Memory-R1** ([2508.19828](https://arxiv.org/abs/2508.19828)):
  - The RL-trained ADD/UPDATE/DELETE/NOOP manager vs an untrained manager (LLaMA-3.1-8B) scores 62.7 vs 52.9 J with GRPO, and 57.5 vs 49.0 with PPO. **The update policy is worth about 8–10 J points on LoCoMo.**
  - Qwen-2.5 scaling, base → GRPO: 3B 41.2 → 57.9; 7B 58.4 → 57.5; 14B 50.8 → 63.5.
  - Trained on only 152 questions.
- **MemReader** (above): 0.6B → 4B-SFT → 4B-GRPO moves LongMemEval KU from 82.1 to 85.8 to 91.0, and HaluMem update-correct from 82.7 to 90.8 to 94.6. **This is the strongest evidence that a sub-1B model can do update-relevant extraction far better than gpt-4o-mini-driven pipelines** (Mem0 25.5, Zep 47.3).
- **Mem-α** ([2509.25911](https://arxiv.org/abs/2509.25911)):
  - Qwen3-4B with GRPO goes from 0.389 to 0.642 on validation, vs gpt-4.1-mini at 0.517 in the same framework.
  - It **excludes conflict resolution** "due to lack of realistic evaluation benchmarks".
- **Supersede** ([2606.27472](https://arxiv.org/abs/2606.27472)):
  - With gpt-5.4, LongMemEval KU is about 92% with full context vs 77% with bounded self-maintained memory.
  - The gap persists across model scales.
  - Qwen2.5-3B with GRPO goes from 9.0% to 16.7%.
  - The author concludes "the bottleneck is memory maintenance, not comprehension". Small n.
- **A-Mem** ([2502.12110](https://arxiv.org/abs/2502.12110)):
  - Backbones from 1B to gpt-4o. Temporal F1: Qwen2.5-1.5B 24.32, Llama3.2-3B 26.38, gpt-4o-mini 45.85.
  - Removing "memory evolution" drops gpt-4o-mini temporal F1 from 45.85 to 31.24.
- **Chronos** ([2603.16862](https://arxiv.org/abs/2603.16862)): components worth 14–35 points with gpt-4o are worth 0–2.6 with Opus 4.6. **Scaffolding substitutes for model strength**, and the converse should hold too.
- **The Mem0 paper and MemOS have no extractor-size or decider-size ablation.**

**What I conclude from this evidence:**

1. The reader caps KU accuracy: gpt-4o with *oracle* evidence scores only 85.9.
2. Write-time update logic can do *worse* than none: FactConsolidation Mem0 18 and Zep 7 vs BM25 48; Mem0 itself abandoned UPDATE/DELETE.
3. Trained small models (0.6–4B) are strong when the task is framed narrowly (MemReader, Memory-R1) and weak as free-form generative maintainers (Supersede's 3B).

All three favour **a narrow discriminative `decide()` over candidate pairs, with non-destructive effects**, over an LLM rewriting memory.

---

## 3. Evidence on small models for these specific decisions

All figures in this section come from the cited primary page: a paper table, a model card or a leaderboard. A "(conv.)" tag marks my own arithmetic.

### 3.1 Entity resolution / entity matching (EM)

**Ditto** (RoBERTa/DistilBERT, about 125M or less), fine-tuned, VLDB 2021. F1 from [arXiv 2004.00584](https://arxiv.org/pdf/2004.00584), Table 5, against DeepMatcher+:

- Abt-Buy: 89.33 vs 62.8.
- Amazon-Google: 75.58 vs 70.7.
- Walmart-Amazon: 86.76 vs 73.6.
- DBLP-ACM: 98.99.
- DBLP-Scholar: 95.6.
- WDC all/xLarge (Table 6): 94.08.

Its case study is direct evidence for "embedding blocking + small classifier":

- A Sentence-BERT matcher alone reached about 92% F1.
- Ditto reached 96.53, but needed about 6k labels to pass 95.
- Embedding blocking followed by a Ditto check cut runtime 3.8× (1.69 h vs 6.49 h).

**Peeters, Steiner & Bizer, "Entity matching using LLMs"** ([arXiv 2310.11244](https://arxiv.org/html/2310.11244v4), EDBT 2025), Table 4, F1:

| Model | WDC Products | Abt-Buy | Walmart-Amazon | Amazon-Google | DBLP-Scholar | DBLP-ACM |
|---|---|---|---|---|---|---|
| GPT-4 zero-shot | 89.61 | 95.78 | 89.67 | 76.38 | 89.82 | 98.41 |
| GPT-4o-mini zero-shot | 81.15 | 91.93 | 86.58 | 72.18 | 86.11 | 97.60 |
| Ditto fine-tuned (~125M) | 84.90 | 91.31 | 86.39 | **80.07** | **94.31** | **99.00** |
| RoBERTa fine-tuned | 77.53 | 91.21 | 87.02 | 79.27 | 93.88 | 99.14 |

- **Who wins where:**
  - GPT-4 leads on hard or unseen product pairs: WDC by about 5 F1, Abt-Buy by about 4.5.
  - Fine-tuned small models win on Amazon-Google and both bibliographic sets.
- **Cost on WDC** (Table 8): GPT-4 0.474¢ per pair (about $4.74 per 1k, conv.) vs GPT-4o-mini 0.006¢.
- **Latency** (Table 9): GPT-4 2.19 s per prompt vs 0.30 s for a quantized fine-tuned Llama.

**Fine-tuning helps small models most and hurts cross-domain transfer** ([arXiv 2409.08185](https://arxiv.org/html/2409.08185)). Llama-3.1-8B on WDC went 53.36 → 69.19 → 74.13 (zero-shot, fine-tuned, fine-tuned with explanations).

**AnyMatch** ([arXiv 2409.04073](https://arxiv.org/abs/2409.04073)): a small LM, zero-shot. It is within 4.4% average F1 of GPT-4-based MatchGPT at **3,899× lower inference cost**. The exact parameter count is not in the abstract.

**Graphiti's deterministic cascade** (§1.2): exact match → MinHash J ≥ 0.9 with an entropy gate → LLM. It is a production example of "deterministic first, model for the residual". No accuracy numbers are published for it.

**Verdict.** A fine-tuned 0.1–0.4B pair classifier over embedding-blocked candidates matches or beats GPT-4 on most EM benchmarks. It loses about 4–5 F1 on unseen or corner-case pairs, and that band is the one to escalate.

**Caveat for kleos.** EM benchmarks are product and bibliographic records. Agent memory resolves *mentions in conversation* ("my sister", "Bob", "Robert Smith from Acme"), which need context and coreference. I found no public benchmark of conversational-mention entity resolution with labels. kleos's own `ENTITY` items are of this type.

### 3.2 Contradiction, fact update and NLI

**Small NLI cross-encoders** (model cards):

| Model | Params | MNLI-m | ANLI (all) | Licence |
|---|---|---|---|---|
| [MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) | ~0.4B | 0.912 | 0.702 (R3 0.64) | MIT |
| [MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) | ~0.18B | 0.903 | 0.579 | MIT |
| [cross-encoder/nli-deberta-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base) | ~0.18B | 90.04 (mm) | – | Apache-2.0 |
| [cross-encoder/nli-deberta-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small) | ~0.14B | 87.55 (mm) | – | Apache-2.0 |
| [tasksource/ModernBERT-large-nli](https://huggingface.co/tasksource/ModernBERT-large-nli) | ~0.4B | 0.89 | A1 0.72 / A3 0.55 | Apache-2.0 |

**GPT-4 zero-shot MNLI/ANLI:** I found **no primary-source number** (UNVERIFIED).

**Grounded factual consistency** is the closest analogue to "does the new fact contradict or entail the old one":

- **MiniCheck** ([arXiv 2404.10774](https://arxiv.org/html/2404.10774)), Table 2, LLM-AggreFact balanced accuracy:
  - GPT-4 75.3, MiniCheck-FT5 (770M) **74.7**, Claude-3 Opus 74.1, MiniCheck-DeBERTa 72.6, AlignScore (355M) 70.4.
  - Cost on 13K examples: GPT-4 $107 vs MiniCheck $0.24, about 450×.
- **[LLM-AggreFact leaderboard](https://llm-aggrefact.github.io/):**
  - Bespoke-MiniCheck-7B 77.4 (CC BY-NC).
  - gpt-4o-2024-05-13 75.9.
  - **FactCG-DeBERTa-L (0.4B) 75.6**; licence UNVERIFIED.
  - MiniCheck-Flan-T5-L (0.8B) 75.0.
- **[HHEM-2.1-Open](https://huggingface.co/vectara/hallucination_evaluation_model)** (~0.1B flan-t5-base, Apache-2.0), balanced accuracy vs GPT-4 zero-shot:
  - AggreFact-SOTA: 76.55 vs 73.78.
  - RAGTruth-QA: 74.28 vs 74.11.

**Important limitation.** NLI and consistency models answer "is B entailed by, or does it contradict, A *at the same time*?". Memory supersession is **temporal**: "Tom lives in Berlin" followed by "Tom moved to Lisbon" is not a logical contradiction. It is an update of a single-valued attribute. Standard NLI will often label it "neutral" or "contradiction" inconsistently. A supersession classifier needs training data in which the label is *update*, not *contradiction*:

- VitaminC revisions;
- WikiFactDiff;
- MQuAKE edits;
- kleos's own `update` vs `contradiction` vs `compatible` vs `duplicate` labels.

**Direct evidence that supersession is hard, even end to end:**

- **Supersede** (Patel, [arXiv 2606.27472](https://arxiv.org/abs/2606.27472), June 2026; verified on the abstract page):
  - On the LongMemEval knowledge-update subset, gpt-5.4 scores about 92% with full context and drops to 77% with bounded self-maintained memory (McNemar p < 0.005).
  - Accuracy falls from 68% to 28% as conversations grow 24×.
  - GRPO on **Qwen2.5-3B** raised held-out supersession accuracy only **from 9.0% to 16.7%** (single run).
  - **A small *generative* model making free-form memory edits is weak; a small *discriminative* pair classifier is a different and much easier task.**
- **MemoryAgentBench FactConsolidation** ([arXiv 2507.05257](https://arxiv.org/html/2507.05257)):
  - Multi-hop accuracy is at most about 28% for every method.
  - Single-hop: GPT-4o 60.0, Mem0 18.0, HippoRAG-v2 54.0.
- **kleos's own D1 runs** (`.bench-data/infer/d1/*.json`, 66 hand-written items):
  - Zero-shot `LiquidAI/d1-omni-600M` on "does B supersede A": AUROC 0.632 (prompt A) and 0.47 (prompt B).
  - DM-JEPA: 0.458 and 0.565.
  - Model2Vec cosine (`potion-base-8M`): supersede AUROC **0.424**, duplicate AUROC 0.889, entity AUROC 0.944.
  - **Cosine works for duplicates and entities. For supersession it is at chance or worse**, as expected: an update is topically similar to what it replaces. Supersession needs a trained cross-encoder.

### 3.3 Importance scoring

**The Generative Agents prompt** ([arXiv 2304.03442](https://arxiv.org/html/2304.03442v2); repo template [`poignancy_event_v1.txt`](https://raw.githubusercontent.com/joonspk-research/generative_agents/main/reverie/backend_server/persona/prompt_template/v3_ChatGPT/poignancy_event_v1.txt)):

> "On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory."

- **Retrieval score** = recency + importance + relevance. All weights are 1, each term is min-max normalised, and recency decays by 0.995 per game hour.
- **Reflection** triggers when summed importance exceeds 150.

**Successors:**

- **MemoryBank** ([arXiv 2305.10250](https://arxiv.org/pdf/2305.10250)) uses an Ebbinghaus curve R = e^(−t/S), where recall strengthens S.
- **Mastra** uses 3-level emoji priorities (§1.12).
- **Mem0** uses an access-recency multiplier clamped to 0.3–1.5× (§1.1).

**Reliability.** I found **no study measuring agreement between LLM importance scores and human salience ratings.**

The closest evidence is "Learning What to Remember" ([arXiv 2606.12945](https://arxiv.org/abs/2606.12945); reported by subagent, not re-fetched):

- It learns a linear weighting of 7 interpretable factors, CPU-only.
- Retention of gold evidence on blind LongMemEval: 0.770, vs 0.657 for uniform weights, 0.518 for the best single factor and 0.368 for recency alone.
- An MLP did no better than the linear model.

**Implication.** Importance should be a cheap learned score, linear or a small regressor, fitted to a downstream signal (whether the memory was later retrieved or needed), not to imitate a GPT-4 1–10 rating.

### 3.4 Dedup / near-duplicate

Cross-encoders ([sbert.net](https://sbert.net/docs/cross_encoder/pretrained_models.html)):

- **Quora duplicates, dev AP:** distilroberta-base 87.48, roberta-large 87.91. **A 4.5× bigger model adds only 0.4 AP.**
- **STS-B:** TinyBERT-L4 85.50, roberta-large 91.47.
- ChatGPT is reported weaker than fine-tuned BERT-style models on paraphrase and similarity ([arXiv 2302.10198](https://arxiv.org/abs/2302.10198)).
- kleos D1: cosine alone reaches duplicate AUROC about 0.89, with best-threshold accuracy 0.857 (potion-base-8M).
- **Industry practice is hash or exact match plus a high cosine threshold.** Mem0 uses MD5 plus entity cosine 0.95; Cognee 0.85; MemOS 0.8 as a candidate gate. I found no published calibration study of these thresholds.

### 3.5 Cascades and calibration

- **[FrugalGPT](https://arxiv.org/abs/2305.05176):** matches GPT-4 at up to 98% lower cost.
- **[AutoMix](https://arxiv.org/abs/2310.12963):** more than 50% cost reduction at comparable quality.
- **[Cascade routing](https://arxiv.org/abs/2410.10347):** the quality of the uncertainty estimator is the key factor.
- **[Token-level uncertainty cascades](https://arxiv.org/pdf/2404.10136):** sequence-level confidence is length-biased; learned deferral rules work better.
- **[Trust or Escalate](https://arxiv.org/abs/2407.18370)** (ICLR 2025): a cheap judge first, escalating only when confidence is below a threshold calibrated to give a **provable human-agreement guarantee**. It reports more than 80% agreement at about 80% coverage. **This is the template for `decide()`.**
- **[Conformal risk control](https://arxiv.org/abs/2208.02814):** picks thresholds that bound an expected loss, such as the false-supersession rate, with finite-sample guarantees.
- **Temperature scaling** (Guo et al. 2017, arXiv 1706.04599) is the standard one-parameter fix for overconfident classifiers.
- **Graphiti's reranker** uses a small LLM's **log-probabilities on a boolean** as a relevance score (`graphiti/README.md:~533`). This is a cheap way to get a probability out of an LLM fallback.

---

## 4. Datasets for evaluating `decide()` with real labels

Licences were checked on the cited page. "SA" means share-alike. I downloaded nothing.

| Dataset | Where | Size | Licence | Commercial? | Adaptation to kleos decisions |
|---|---|---|---|---|---|
| **VitaminC** | [`tals/vitaminc`](https://huggingface.co/datasets/tals/vitaminc), [repo](https://github.com/TalSchuster/VitaminC) | 371k / 63k / 55k; from >100k real Wikipedia revisions | CC BY-SA 3.0 | Yes (SA) | **Best for supersession.** Group by `case_id`; before and after revision evidence for the same claim gives (old, new, UPDATE or CONTRADICT). Use `revision_type=="real"`. |
| **WikiFactDiff** | [`Orange/WikiFactDiff`](https://huggingface.co/datasets/Orange/WikiFactDiff) | 100K–1M; Wikidata 2021-01 → 2023-02 | CC BY-SA 4.0 | Yes (SA) | Obsolete vs new triple for the same (subject, relation) → UPDATE. Unchanged relations of the same subject → COMPATIBLE. Verbalize triples. |
| **MQuAKE** (CF 9,218; T 1,825) | [repo](https://github.com/princeton-nlp/MQuAKE) | as listed | MIT; Remastered CC BY 4.0 | Yes | `target_true` → `target_new` gives (old, new, SUPERSEDE). MQuAKE-T has real temporal changes. MemoryAgentBench FactConsolidation is built from it. |
| **LongMemEval** knowledge-update | [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned), [repo](https://github.com/xiaowu0162/LongMemEval) | 500 Qs (KU count: see §2) | HF card MIT; data licence not stated on GitHub | Probably | For `question_type=="knowledge-update"`, take turns with `has_answer: true` from `answer_session_ids` and order them by `haystack_dates` → (older, newer, UPDATE). Small eval set; also the main end-to-end test. |
| **Dialogue NLI** | [site](https://wellecks.com/dialogue_nli/), `pietrolesci/dialogue_nli` | 310k / 16.5k / 16.5k (+12.4k verified) | not stated | ? | `triple1`, `triple2`, `label` (entail/neutral/contradict) over persona facts. Very close to memory triples. |
| **DECODE** | [ParlAI](https://github.com/facebookresearch/ParlAI/tree/main/projects/contradiction) | 27,184 / 4,026 / 4,216 + 764 human-bot | not stated | ? | Dialogue contradiction pairs. |
| **WikiContradict** | `ibm-research/Wikipedia_contradict_benchmark` | 253 instances | MIT | Yes | Small, high-quality contradiction eval. |
| **ConflictBank** | [repo](https://github.com/zhaochen0110/conflictbank) | 7.45M claim-evidence pairs | CC BY-SA 4.0 | Yes (SA) | Typed conflicts (misinformation, temporal, semantic). Partly synthetic. |
| **PersonaMem** | `bowen-upenn/PersonaMem` | 5,990 rows | MIT | Yes | "track preference evolution" and "acknowledge latest preferences" question types. Synthetic. |
| **SNLI / MNLI** | `stanfordnlp/snli`, `nyu-mll/multi_nli` | 570k / 393k | CC BY-SA 4.0 / OANC (+ some CC BY-SA fiction) | Yes | Generic contradiction pretraining only. |
| **FEVER** | `fever/fever` | 100K–1M | CC BY-SA 3.0 (+GPL tag) | Yes (SA) | Claim and evidence SUPPORTS/REFUTES. |
| **ANLI** | `facebook/anli` | 163k | **CC BY-NC 4.0** | **No** | Eval only. |
| **Abt-Buy, Amazon-Google, DBLP-ACM, DBLP-Scholar** | [Leipzig](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution); [DeepMatcher splits](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md) | e.g. Abt-Buy 1,081+1,092 records / 1,097 matches | CC BY 4.0 | Yes | (A, B, same?) pairs, with embedding-blocking negatives. |
| **WDC Products** | [site](https://webdatacommons.org/largescaleproductcorpus/wdc-products/) | test 4,500 pairs; seen, half-seen and unseen variants | not stated | ? | Best robustness test for unseen entities. |
| **PAWS-Wiki** | `google-research-datasets/paws` | 49k / 8k / 8k | "freely used for any purpose" | Yes | Hard negatives for dedup (word swaps). |
| **QQP** | GLUE | ~404k | Quora terms; likely non-commercial | No / ? | Dedup. Avoid for training a shipped model. |
| **LoCoMo** | [repo](https://github.com/snap-research/locomo) | 10 conversations | **CC BY-NC 4.0** | **No** | End-to-end eval only. |
| "ContradictionNLI" | `chitra/contradictionNLI` | single train.csv | none; no provenance | Avoid | No paper or provenance found. Do not use. |

**A recipe for replacing the 66 hand-written items with a labelled set of about 5–10k pairs** (all commercial-OK sources):

1. **Supersession, 4-way** {update, contradiction/correction, compatible, duplicate}:
   - *update*: VitaminC real revisions and WikiFactDiff obsolete → new.
   - *compatible*: other relations of the same subject from WikiFactDiff or MQuAKE.
   - *duplicate*: paraphrases from PAWS (positive) and VitaminC claim paraphrases.
   - *contradiction/correction*: VitaminC REFUTES at the same timestamp, and WikiContradict.
   - Verbalize triples with templates, and add conversational surface forms ("btw I moved to…"), possibly LLM-paraphrased once offline and reviewed.
2. **Hold-out test sets:**
   - the LongMemEval knowledge-update pairs mined as in the table;
   - the existing 66 items;
   - DECODE or Dialogue NLI (licence permitting) for the dialogue domain.
3. **Entity:**
   - Leipzig EM sets for structured records;
   - for conversational mentions, build pairs from LongMemEval and LoCoMo (eval only, given NC) person and organization mentions. **No public labelled set exists for this.**
4. **Report** AUROC, ECE and Brier on every split, plus **precision at the auto-apply threshold**. That threshold is the operating point that matters for safety (§5.3).

---

## 5. Design recommendations for kleos

### 5.1 Which mechanism for which decision

| Decision | Tier 0: rules / embeddings (µs–ms) | Tier 1: small classifier via `decide()` (~1–20 ms on CPU for a 0.1–0.4B cross-encoder; to be benchmarked) | Tier 2: LLM fallback |
|---|---|---|---|
| **Exact / near duplicate** (same fact restated) | Hash of normalized text (Mem0); cosine ≥ τ_hi auto-merge (kleos D1: cosine AUROC 0.89) | Paraphrase cross-encoder for τ_lo < cos < τ_hi; PAWS hard negatives | Rarely needed |
| **Entity resolution** | Exact normalized name, identity keys (email, ID), MinHash J ≥ 0.9 **with the entropy gate** (Graphiti); embedding + FTS + LSH blocking (E3) | Pair classifier on (mention + context, candidate + summary). Ditto-style; context matters | Short or ambiguous names ("Sam", "Java"), multiple exact matches, low-margin top-2 |
| **Supersession of a keyed fact** (same entity, same single-valued attribute) | **A rule, not a model.** kleos already supersedes on key. Declare functional attributes (Cognee `functional_relationships`, Memobase slots, Supermemory `isStatic`) | – | – |
| **Mapping a free-text fact onto a key** ("is this about Tom's residence?") | Candidates by key/entity + vector | `decide(choice)` over candidate attributes plus "none". The main place a model helps supersession | When the top-2 margin is small |
| **Unkeyed supersession / contradiction** | Candidates: same entity (graph) ∩ vector top-k ∩ time order. **Never cosine alone** (AUROC ≈ chance in D1) | **4-way cross-encoder** {update, contradiction, compatible, duplicate} trained on VitaminC, WikiFactDiff, MQuAKE and kleos pairs | Low confidence, multi-fact or numeric/date subtleties, or "partial update" |
| **Contradiction flagging** (inconsistent sources, no time order) | – | Same 4-way head; flag only | Optional |
| **Importance / salience** | Features: novelty (1 − max cos to existing), entity count, named entities, first-person, numbers and dates, explicit cues ("remember", "always", "never"), source type | Small regressor (linear first, per "Learning What to Remember") fitted to later-retrieval/use signals | Not recommended; LLM 1–10 scores are unvalidated |
| **Consolidation** (episodes → observations, summaries, profiles) | Clustering by entity, time and vector | `decide(yes/no)` "does the cluster warrant a summary?" | **LLM writes the summary.** Generation is not a `decide()` job. Async, as in Mem0 Synthesis, Letta reflection and MemOS reorganizer |
| **Routing to memory type** (MIRIX-style) | Source-type rules | `decide(choice)` | – |

### 5.2 Cascade design and calibration

1. **Candidate generation is the precision lever.** Every system restricts the judge to 5–10 candidates. Graphiti searches same-endpoint edges for duplicates and the wider group for contradictions. MemOS gates at cosine 0.8 with k = 5.
   - For kleos, require the **same resolved entity**, or graph proximity, before any supersession candidate is scored.
   - This removes MemOS's documented failure: an UPDATE applied to the wrong entity, "Gladys Liu" vs "Mission Terra".
2. **`decide()` returns calibrated probabilities**, not labels:
   - Fit temperature scaling (1 parameter) per decision type on a held-out calibration split.
   - Track ECE and Brier. D1 already reports them; keep doing so.
3. **Two thresholds per decision, giving three bands:**
   - p ≥ τ_auto → apply automatically.
   - τ_flag ≤ p < τ_auto → apply as **tentative**, or escalate to an LLM if one is configured.
   - p < τ_flag → no action.
4. **Choose τ_auto by risk control, not accuracy.** On the calibration set, pick the smallest τ_auto such that the upper confidence bound of the *false-supersession rate* (or false-merge rate) among auto-applied items is at most α, say 1–2%. Use [conformal risk control](https://arxiv.org/abs/2208.02814) or the [Trust-or-Escalate](https://arxiv.org/abs/2407.18370) procedure.
   - Report **coverage**, the share auto-decided, at that α.
   - This is the single most useful number for comparing backends (d1-omni, DM-JEPA, a cross-encoder).
5. **Asymmetric costs:**
   - A false supersession hides a true fact from `latest_only` reads, which is worse.
   - A missed supersession leaves two "current" facts; retrieval can still order by time, and the reader sees both.
   - So τ_auto should be high for supersede/merge and can be lower for "link as related". Mem0 v3's choice of linking only is the extreme of this trade-off.
6. **LLM fallback:**
   - Use a typed prompt modelled on Graphiti's `resolve_edge`, including its negative rules, so the LLM output lands in the same label space.
   - Read the label's log-probability where available (the Graphiti reranker pattern) so the fallback is calibrated too.
   - Log every escalation with both answers. That log is free training data for the next classifier version (active learning).
7. **Abstain is a first-class output.** `decide()` should be able to return `unknown`, and the write path must handle it as "store, link, don't supersede".

### 5.3 Safety: no destructive auto-supersession

The field has converged on the following. kleos's bitemporal store makes all of it cheap.

- **Never delete on a model decision.**
  - Supersession only sets the end of validity and the transaction end, as Graphiti does with `invalid_at`/`expired_at` and as kleos E1 specifies.
  - Hard deletes are reserved for provenance-driven erasure (GDPR, E2), which is user-initiated, not model-initiated.
  - Do not copy MemOS's `_hard_update` (delete-older-on-unresolvable) or Cognee's delete-after-cosine-merge.
- **Store the decision as data:**
  - `supersedes(new, old, decided_by, model_version, p, band, at)`;
  - `merged_into(dup, canonical, …)`.
  - Then "undo" is a retraction of the decision row, and a Datalog rule recomputes the current view.
  - A model upgrade can **re-run old decisions** and diff the results.
- **Tentative supersession:**
  - Mid-band decisions are recorded as `tentative`.
  - `latest_only` reads can choose policy: hide tentatively-superseded facts, or show both with a flag.
  - Mem0's Dream default read returns active plus superseded facts labelled, and `latest_only=true` returns only active ones. Offer the same choice.
- **Guard rules before any model.** Encode Graphiti's prompt rules as cheap checks:
  - different subject entity → never supersede;
  - different event date or time ("ran 5 miles Tuesday" / "3 miles Wednesday") → compatible;
  - new fact describes activity on an owned or assigned thing → compatible;
  - numeric, date or qualifier difference → never "duplicate".
- **Out-of-order arrival:** apply Graphiti's temporal arithmetic.
  - If the "new" fact's valid time is earlier than the existing fact's, the *incoming* fact is the one ended.
  - Skip candidates whose validity windows do not overlap.
- **Merge of entities is the most dangerous operation.** Union-find over a cosine threshold chains A≈B≈C (Cognee).
  - Require a pairwise `decide()` for every merge.
  - Never infer one transitively.
  - Keep `same_as` as an edge with confidence, rather than physically rewriting IDs.
  - HippoRAG 2 does exactly this: soft synonym edges, not merges.
- **Audit stream:** every auto-decision emits an event through live subscriptions. A host app or human can then review the low-margin ones.

### 5.4 Where in the write path

| Step | When | Why |
|---|---|---|
| Hash/exact dedup, keyed supersession (rule), identity-key entity match | **Synchronous, in the write transaction** | Deterministic and microseconds; must hold before the read-your-writes guarantee |
| Embedding (`embed()`) and candidate retrieval | Synchronous if the embedder is local and fast; else async | Mem0 v3 reports a p50 retrieval latency of about 1 s end to end on their platform (README; self-reported) |
| Tier-1 `decide()` (small cross-encoder) | **Synchronous by default for supersession and dedup of the written fact**, with a latency budget (e.g. ≤ 20 ms); falls back to async if over budget | A wrong "current" view immediately after a write is the user-visible bug that knowledge-update questions test |
| Tier-2 LLM fallback, consolidation, importance recompute, entity-merge sweeps | **Async jobs** (F4 scheduled jobs, F5 slow procedures) driven by **live subscriptions** on new facts | All mature systems run these in the background: MemOS reorganizer queue, Mem0 Dream synthesis, Letta sleep-time/reflection, Memobase 1-hour buffers, LangMem `after_seconds` |
| Result of async decisions | Written as `supersedes`/`merged_into` rows; subscribers see the delta | Keeps history and makes later reversal a retraction |

**Pattern to adopt: write first, resolve later.** The fact is inserted immediately as `current`, with a `pending_resolution` marker if Tier 1 escalated.

- The async resolver later adds `supersedes` edges with valid-time set from the facts, not from processing time.
- Bitemporal queries then remain correct "as known at T": the system did not know of the supersession until the resolver ran, and transaction time records exactly that.

### 5.5 Concrete next steps

1. **Build a labelled set of about 5–10k pairs** (§4 recipe) and keep the 66 items plus the LongMemEval knowledge-update pairs as held-out tests.
2. **Baselines to beat on that set,** reporting AUROC, ECE and coverage at α = 2% false-supersession:
   - (a) cosine;
   - (b) off-the-shelf NLI cross-encoders, the `cross-encoder/nli-deberta-v3-small/base` and `MoritzLaurer` large models (Apache-2.0/MIT), with the label mapping contradiction → supersede;
   - (c) the HHEM or MiniCheck-style consistency head;
   - (d) d1-omni, DM-JEPA;
   - (e) the GPT-4-class judge with Graphiti's prompt, as the ceiling.
3. **Fine-tune a 4-way head** on a 0.1–0.4B encoder (DeBERTa-v3, ModernBERT). This is the most likely winner given §3.2, but it is **unproven for temporal supersession** and must be measured.
4. **Measure end to end** on the LongMemEval knowledge-update, temporal-reasoning and abstention categories, with `decide()` vs the LLM judge in the loop. This answers whether the decision model is the bottleneck, which no published paper isolates cleanly (§2).

---

## Appendix A. Source clones read (shallow, `.bench-data/research/repos/`)

| Repo | Upstream | Commit | Date |
|---|---|---|---|
| A-mem | https://github.com/agiresearch/A-mem.git | ceffb86 | 2025-12-12 |
| cognee | https://github.com/topoteretes/cognee.git | f154b32 | 2026-10-13 |
| graphiti | https://github.com/getzep/graphiti.git | d50741d | 2026-10-15 |
| HippoRAG | https://github.com/OSU-NLP-Group/HippoRAG.git | d5c8329 | 2026-10-06 |
| langmem | https://github.com/langchain-ai/langmem.git | 48e3c11 | 2026-10-02 |
| letta-archive | https://github.com/letta-ai/letta.git | 56ba9c2 | 2026-08-13 |
| letta-code | https://github.com/letta-ai/letta-code.git | a586e9a | 2026-10-15 |
| letta | https://github.com/letta-ai/letta.git | 5bcdd17 | 2026-09-10 |
| mem0 | https://github.com/mem0ai/mem0.git | 0724291 | 2026-10-14 |
| memobase | https://github.com/memodb-io/memobase.git | 358c16b | 2026-01-11 |
| MemOS | https://github.com/MemTensor/MemOS.git | a7367d0 | 2026-09-22 |
| MIRIX | https://github.com/Mirix-AI/MIRIX.git | 8cb06a6 | 2026-08-20 |
| supermemory | https://github.com/supermemoryai/supermemory.git | ccf0485 | 2026-10-15 |

## Appendix B. Open uncertainties

- **Mem0 Dream:** the Supersede and Merge decision logic, model and thresholds are closed source. Supermemory's engine is closed too.
- **Mem0 v3 headline numbers** come from its managed platform ("proprietary optimizations not available in the open-source SDK"). Two Mem0 pages disagree: LoCoMo 91.6 vs 92.5, LongMemEval 93.4 vs 94.4.
- **No primary-source zero-shot GPT-4 MNLI/ANLI accuracy** was found.
- **FactCG-DeBERTa-L licence** was not verified.
- **Unclear dataset licences:** DECODE, Dialogue NLI, MSC, WDC Products, PrefEval.
- **Count sources:** the LongMemEval per-type counts come from secondary sources that match Mastra's denominators. The BEAM per-ability counts (about 200) are inferred.
- **"Learning What to Remember"** (arXiv 2606.12945) numbers were reported by a research subagent and not re-fetched by me.
- **No public labelled benchmark** exists for *conversational mention* entity resolution, nor for *temporal* supersession pairs in dialogue. Those must be built (§4 recipe).
- **The latency figures for a 0.1–0.4B cross-encoder on kleos's target CPU** are not measured here. The ~20 ms budget in §5 is a design target, not a measurement.
