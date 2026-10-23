# Memory API: remember, recall, forget, reflect, link

Status: implemented (wave2/memory-api), UPGRADE_PLAN H1 + E10.

Code:
- `kleos-core/src/memory.rs`: schema, the five operations, the entity hook and the JSON entry point;
- `parse/sys.rs` and `cozoscript.pest` (`mem_api_op`): the `::memory` ops;
- `lib.rs`: `DbInstance` dispatch;
- `kleos-lib-python/src/lib.rs`: Python methods;
- `kleos-bin/src/server.rs`: `POST /memory/:op`.

Tests:
- `kleos-core/tests/memory.rs`: fake decider and embedder, plus one `#[ignore]` real-model test;
- `kleos-lib-python/tests/test_memory_api.py`.

Scenario: `scripts/memory-api/life_stream.py` and `kleos-core/examples/memory_scenario.rs`.

## Goal

An agent uses kleos as memory without writing CozoScript. It stores statements, asks questions and forgets things, on a ready-made schema with a namespace (agent or user scope) on every row. The write path decides, with the small decision models, whether a new statement replaces, repeats or adds to what is stored. It never deletes to do so.

Everything is built from parts that already existed:
- valid time and E1 supersession by appended retraction rows ([memory-supersession-provenance.md](memory-supersession-provenance.md));
- `::erase` and `::changes`;
- the embedder registry ([embedder-toggle.md](embedder-toggle.md));
- FTS, filtered HNSW and the `WeightedFusion` fixed rule;
- `decide()` ([decide.md](decide.md)).

## Schema

Created by `::memory init` (Rust `memory_init`) or lazily by the first `remember`. Every relation is keyed by `ns` first, so a namespace is one key range.

| Relation | Columns | Notes |
|---|---|---|
| `kleos_mem` | `{ns, id, vt: Validity => text, kind, source?, created, importance?}` | The memories. `vt` is valid time: assertions and retractions are rows, so history, `@ T` and `::history` keep answering. FTS index `kleos_mem:fts` (Simple tokenizer, lowercase, English stemmer, stopwords). |
| `kleos_mem_vec` | `{ns, id => current: Bool, vec: <F32; D>}` | One vector per memory. Created when an embedder is set, with `D` taken from the embedder. HNSW index `kleos_mem_vec:hnsw` (cosine). `current` is false once the memory is superseded or retracted. |
| `kleos_mem_entity` | `{ns, eid => name, norm, kind?, created}` | Entities. Index `by_norm {ns, norm}`. |
| `kleos_mem_mention` | `{ns, id, eid}` | Memory → entity. Index `by_entity {ns, eid}`. |
| `kleos_mem_link` | `{ns, src, rel, dst => created, p?, decided_by?}` | Typed links between memories and entities. `remember` writes `new -[supersedes]-> old` with the decision's p and `decided_by`. |
| `kleos_mem_source` | `{ns, id, at, stmt => source?, text}` | Provenance: one row per statement behind a memory, so a duplicate adds a row here instead of a memory. |
| `kleos_mem_decision` | `{ns, new, cand, template => label, p, band, decided_by, reason?, status, at}` | The audit table: every decision that was applied (`status: applied`) or left for review (`pending`). |
| `kleos_mem_config` | `{key => value}` | `embedder`, `dim`, `decider`, `version`. It is created last, so its presence marks a complete schema. |

Ids are time-sortable strings: `m_…` for memories and `e_…` for entities. `kind` defaults to `fact`, `ns` to `default`.

**Why vectors live in a sibling relation.**
- The dimension is unknown until an embedder is chosen. With a separate relation, a store created without one can gain vectors later: `::memory init {embedder: …}` creates the relation and backfills it.
- Search indexes on a valid-time relation also index superseded and retraction rows. The `current` flag lets HNSW filter them out inside the search (`filter: ns == $ns && current`) rather than after it.
- FTS stays on `kleos_mem`. Its stale rows are dropped by the `@` join after an over-fetch.

## Operations

### remember(text, opts)

Options:
- `ns`, `kind`, `source`, `importance`;
- `entities`: names;
- `at`: valid-from, in µs; the default is now;
- `k`: the number of candidates, default 5;
- `decider`;
- `embedder`: only used when this call creates the store.

Steps:
1. Embed the text, if the store's embedder is enabled.
2. Find the top-k current memories of the namespace with the same hybrid search as `recall`: the vector leg uses the document vector, with an FTS fallback.
3. Exact rule, model-free: a candidate whose words equal the new text's (lower case, punctuation ignored) is a duplicate with band `auto`, `decided_by = kleos|duplicate@v1|exact`.
4. For the remaining candidates, run `decide(supersedes)` and `decide(duplicate)` with `(old = candidate, new = text)`. Each is one batched call over all candidates.
5. Act on the bands:

| Outcome | Action |
|---|---|
| `duplicate` is auto/yes and `supersedes` is not | Nothing is inserted. The statement becomes a `kleos_mem_source` row of the existing memory, and the result has `duplicate_of`. |
| `supersedes` is auto/yes and `duplicate` is not | The new memory is inserted. Each such candidate gets a retraction row at `at` (E1 semantics, values copied), `current = false` on its vector, and a `supersedes` link. Candidates whose validity starts after `at` (a late arrival) are left pending instead. |
| Tentative (label yes, band tentative); `supersedes` with p ≥ 0.5 below the calibrated threshold (the review band, below); or two auto answers that disagree | The memory is inserted and the decision is recorded `pending`. `recall` flags the older memory with `pending_superseded_by`, and can hide it with `hide_pending`. |
| None | Plain insert. |

6. Link entities (see "Entity hook").

All writes of one `remember` go in one imperative script, so one transaction: the insert, retractions, vector, links, source and audit rows. The candidate read and the model calls happen before that transaction. Two concurrent `remember`s on the same topic can therefore both insert; a later `remember` or `reflect` sees both.

The result:
- `action`: `added` or `duplicate`;
- `id`, `superseded` and `duplicate_of`;
- `tentative`: the pending decisions;
- `decisions`: all of them, including `none`;
- `entities`, `embedded` and `decider`.

Each decision carries the candidate, template, label, p, band, `decided_by`, reason and action.

**Never deletes.** Supersession ends validity. Nothing in `remember` removes a row.

#### Template choice (why supersedes + duplicate, not relation)

With the default weights (nano-tune-v1, [decide.md](decide.md)):

| Template | Calibration (adapt-60) | What it gives the write path |
|---|---|---|
| `supersedes` | threshold = τ_auto = 0.997 | AUROC 0.99. 36% of true updates are auto-applied with 0 false on eval. LongMemEval knowledge-update 0.82. **The only template with a usable auto band**, so it drives supersession. |
| `duplicate` | threshold 0.091, τ 1.01 | AUROC 0.99, but never auto: with tuned weights, the auto band admitted elaborations as duplicates. Every model duplicate is therefore *tentative* and recorded for review. Auto duplicates come from the exact-text rule, or from a calibration or probe the user fits (`::decider fit … duplicate`). |
| `relation` | threshold 0, τ 1.0 | Every answer is tentative, so it never acts. Its 4-way accuracy (0.905) needs a fitted probe, and its "update" label is the same NLI-contradiction signal that `supersedes` reads better. |

**Review band.** The tuned `supersedes` calibration has threshold = τ = 0.997, so it has no tentative band: p = 0.99 comes back as "no", band none. `remember` keeps every `supersedes` answer with p ≥ 0.5 below the threshold as a pending decision. It never acts on one. Reads choose the policy, as the decide spec's safety guidance asks:
- by default, both memories are returned and the older one is flagged;
- with `hide_pending`, the flagged one is left out.

So `remember` asks two binary questions. Each has its own validated prompt and calibration, and each band maps directly onto one action. The cost is about 3 forward passes per candidate: supersedes takes 1, and duplicate takes 2 because it is symmetric.

Shortcuts not taken:
- Neither question is skipped from cosine similarity. The decide spec warns against acting on cosine alone, and the candidates are only the top-k.
- `important` is not run: it is unvalidated. `importance` stays whatever the caller passes.

#### Without models (default build, or models off)

- No embedder: no vectors. Candidates and recall use FTS only.
- No decider: two model-free rules remain.
  1. The exact-text duplicate rule.
  2. The value pre-check (`decide::rules::value_rule`). An `Update` (same subject and attribute, one value kind changed, guarded against bounds, additions and negation) supersedes with band `auto` and `decided_by = rules|supersedes@v1|rule`. It fired 20 times in the decide evaluation and was right every time. That is the basis for letting it act alone, but it is a small sample.
- Everything else is a plain insert. Nothing errors because ML is off.
- Decider choice: the call's `decider`, else the store's (`init {decider}`), else the first enabled decider. A named decider that is disabled means rules only.
- Embedder choice at lazy creation: the call's `embedder`, else the first enabled one whose name does not end in `_query`. Recall embeds the query with `{embedder}_query` when it exists (gemma's query prompt).

### recall(query, opts)

Options:
- `ns`, `k` (default 10), `kind`, `entity` (a name);
- `as_of`: valid time in µs;
- `vector_weight`: default 0.6;
- `hide_pending`: leave out memories with a pending `supersedes` by a memory valid at the read time. Default false.

It is one script:
- The vector leg is HNSW over `kleos_mem_vec`, filtered to the namespace (and `current` unless `as_of` is set). Its score is −distance.
- The text leg is FTS over `kleos_mem`, filtered to the namespace. The query is the words of the text OR-ed, so user text is never parsed as FTS syntax.
- `WeightedFusion` combines them with weights `[semantic 0.6, text 0.4]` and the theory-based norms `neg_cosine_distance` / `bm25`. Min-max is unstable on the short lists memory search produces.
- The result is joined with `kleos_mem` `@ 'NOW'` (or `@ $as_of`), plus the kind and entity filters, and ordered by score.
- Both legs over-fetch (4k + 16, and 4× more with a filter), because the indexes also hold stale versions.

Each hit has `id, text, score, kind, source, created, importance, valid_from, valid_to, pending_superseded_by`. The validity interval is the one that contains the read time; `valid_to` is null while the memory is still valid. Only memories valid at the read time are returned, so a superseded fact never shows up as current, and `as_of` reads the past.

The vector leg always returns the nearest neighbours, so recall returns up to k memories even when none are relevant. There is no score cutoff.

### forget(id | filter, mode)

- Targets: `id` or `ids`, or a filter `kind` / `entity`, or `all: true` for a whole namespace. A call without any of these is an error.
- `mode: 'retract'` (the default) ends validity at `at` (default now) with a retraction row and clears `current` on the vector. History and `as_of` recall keep the memory.
- `mode: 'erase'` (GDPR) calls `::erase kleos_mem` on the keys. That deletes every version, with its FTS entries, and writes the salted `kleos_evict_audit` marker. It then removes the memory's vector, source rows, mentions, links and audit rows, and any entity left with no mention.
- Returns `{mode, ids}`: the memories that were current (retract) or existed (erase).

### reflect(ns, opts)

Minimal and model-free; real consolidation is E7. Options: `since` (µs, default 7 days ago) and `limit` (default 50).

It returns:
- `pending`: the tentative decisions, newest first, with both texts;
- `changes`: `::changes kleos_mem {from: since, prefix: [ns]}`, i.e. `added` and `ended` by valid time;
- `contradictions`: pending `supersedes` decisions, plus pairs of current memories that share an entity and whose values differ by the value rules (`update` or `hint`). Up to 30 memories per entity are compared.

There is no op yet to accept or reject a pending decision. To act on one, `forget` the old memory or the new one.

### link(from, to, rel)

Adds `kleos_mem_link {ns, from, rel, to}` with optional `p`. Both ends must be memories or entities of the namespace.

### Entity hook

`memory::link_entities(db, ns, memory_id, text, names)` is the only place entities are resolved. For now it matches the exact normalised name (lower case, collapsed whitespace, outer punctuation trimmed) of the names passed in `entities`, and creates an entity when there is none. It does no NER. Entity resolution (E3, `entity.rs`, a parallel branch) is meant to replace this function's body; `text` is passed in so it can extract mentions itself.

## Surfaces

The same options everywhere: one JSON object per op, handled by `Db::memory_call(op, args)`.

**Rust** (`Db` and `DbInstance`):

```rust
use kleos::memory::{RememberOpts, RecallOpts, ForgetOpts, ReflectOpts, LinkOpts, InitOpts};
db.memory_init(InitOpts { embedder: Some("gemma".into()), decider: Some("nano".into()) })?;
let r = db.remember("I moved to Lisbon.", RememberOpts { ns: Some("alice".into()), ..Default::default() })?;
let hits = db.recall("where do I live?", RecallOpts { ns: Some("alice".into()), ..Default::default() })?;
db.forget(ForgetOpts { ns: Some("alice".into()), id: r.id, mode: Some("erase".into()), ..Default::default() })?;
db.reflect(ReflectOpts { ns: Some("alice".into()), ..Default::default() })?;
db.link(LinkOpts { ns: Some("alice".into()), from: a, to: b, rel: "works_at".into(), p: None })?;
db.memory_call("recall", serde_json::json!({"query": "where do I live?", "ns": "alice"}))?;
```

**CozoScript.** The positional argument is the text (remember), the query (recall), the id or list of ids (forget) or the namespace (reflect); the rest are options:

```
::memory init {embedder: 'gemma', decider: 'nano'}
::memory remember 'I moved to Lisbon.' {ns: 'alice', entities: ['Lisbon'], source: 'chat-12'}
::memory recall 'where do I live?' {ns: 'alice', k: 5, as_of: 1767225600000000}
::memory forget $id {ns: 'alice', mode: 'erase'}
::memory forget {ns: 'alice', kind: 'preference'}
::memory reflect 'alice' {since: 0}
::memory link {ns: 'alice', from: $a, to: $b, rel: 'works_at'}
```

Output:
- `recall` returns one row per hit: `id, text, score, kind, valid_from, valid_to, created, source, importance`.
- The other ops return one row with a column per result field. Lists of plain values become lists; records become JSON.
- `recall` and `reflect` run in read-only scripts. The others are refused there, and inside imperative blocks, because they run their own transactions.
- Unknown options are errors.

**Python:**

```python
db.memory_init(embedder="gemma", decider="nano")
r = db.remember("I moved to Lisbon.", ns="alice", entities=["Lisbon"])   # dict
db.recall("where do I live?", ns="alice", k=5, as_of=None)               # list of dicts
db.forget(r["id"], ns="alice", mode="erase"); db.forget(ns="alice", kind="preference")
db.reflect("alice"); db.link(a, b, "works_at", ns="alice")
```

**HTTP** (`kleos-bin server`):

```
POST /memory/remember  {"text": "I moved to Lisbon.", "ns": "alice"}
POST /memory/recall    {"query": "where do I live?", "ns": "alice", "k": 5}
POST /memory/forget    {"id": "m_…", "ns": "alice", "mode": "erase"}
POST /memory/reflect   {"ns": "alice"}
POST /memory/link      {"ns": "alice", "from": "m_…", "to": "e_…", "rel": "about"}
POST /memory/init      {"embedder": "gemma"}
→ {"ok": true, "result": …}
```

A read-only principal may only use `recall` and `reflect`.

## Results

### Deterministic tests

`cargo test -p kleos --test memory`. They use a fake NLI decider (decider-nano family, nano-tune-v1 calibration) and a hashed bag-of-words embedder. They cover:
- add;
- update → auto supersede, with the link and audit rows and the change feed;
- time travel (`as_of`, `valid_from`/`valid_to`);
- the exact duplicate, which adds a source row;
- the model duplicate as tentative and pending, then auto with a fitted calibration;
- forget retract and erase, by id and by entity, including the FTS entries;
- namespace isolation and the kind filter;
- entity linking, the entity filter, `link` and reflect contradictions;
- the `::memory` ops in read-only mode, plus the JSON entry point;
- no ML at all: lazy init, FTS only, value-rule supersede, exact duplicate.

### Real scenario

A 40-turn "user life" stream (`scripts/memory-api/life_stream.py`): moves, job changes, changing preferences, restatements and unrelated facts. It has 20 questions, 3 of them `as_of`, and is run with decider-nano (nano-tune-v1, q8) and EmbeddingGemma int8 on Metal (`python3 scripts/memory-api/life_stream.py --run`; report in `.bench-data/memory-api/`).

| | Result |
|---|---|
| Auto-supersede | 12 of 15 gold updates, **0 wrong** |
| Missed supersedes | 3, scored p 0.996, 0.993 and 0.977 (τ 0.997): Berlin → Lisbon, and Spotify → Tidewater twice. All 3 were kept as pending through the review band. |
| Duplicates | 4 of 6 skipped, 0 wrong. The 2 misses are paraphrases, recorded as tentative. |
| Pending decisions | 13 in total, of which 4 are true supersedes. The false ones are cross-subject pairs: "my sister lives in Munich" against "I live in Berlin", and a cat against a dog. |
| Recall, current fact first (top-1) | 15 / 20. With `hide_pending`: 18 / 20, but one time-travel answer was hidden by a false pending (the sister). |
| Latency p50 (loaded machine, two other builds running) | remember 355–655 ms, recall 11–38 ms |

What the numbers say:
- The auto band is safe on this stream.
- The misses all sit just under τ. The review band catches them, but it also flags different-subject pairs.
- Same-subject candidates, which is what entity resolution (E3) will provide, are the precision lever for both. Until then, `hide_pending` stays off by default.

## Caveats

- The candidate read and model calls run outside the write transaction (above).
- The decider sees only the top-k candidates. A stale fact phrased very differently from its update may never be compared. `reflect` surfaces the ones that share an entity.
- Recall over-fetches to skip stale index rows. A namespace with very many superseded versions of one topic could still crowd a current memory out of the fetched window.
- One vector dimension per store. Changing to an embedder with another dimension is refused; re-embedding is not implemented.
- `importance` is never inferred: the `important` template is unvalidated.
- There is no API to resolve pending decisions yet; consolidation is E7.
