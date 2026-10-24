# Memory API: remember, recall, forget, reflect, link

Status: implemented (wave2/memory-api), UPGRADE_PLAN H1 + E10. Branches (fork, rewind, merge, diff, discard) are specified in [memory-branches.md](memory-branches.md) (wave4/fork). Entity resolution in `remember` (E3) and memory scoring (E4, C6): wave2/mem-scoring. Diverse recall, entity expansion, the context output, speaker roles and valid-from filters: wave3/recall. Statement text in list hits and write-time dates (event times): wave3/dates.

Code:
- `kleos-core/src/memory.rs`: schema, the operations, entity resolution and the same-subject gate, scoring, the access buffer and the JSON entry point;
- `kleos-core/src/memory_recall.rs`: the diverse selection and the context block (model-free, deterministic);
- `kleos-core/src/memory_time.rs`: the rule-based temporal-expression parser behind event times (see "Dates");
- `parse/sys.rs` and `cozoscript.pest` (`mem_api_op`): the `::memory` ops;
- `lib.rs`: `DbInstance` dispatch;
- `kleos-lib-python/src/lib.rs`: Python methods;
- `kleos-bin/src/server.rs`: `POST /memory/:op`.

Tests:
- `kleos-core/tests/memory.rs`: fake decider and embedder, plus one `#[ignore]` real-model test;
- `kleos-lib-python/tests/test_memory_api.py`.

Scenario: `scripts/memory-api/life_stream.py` and `kleos-core/examples/memory_scenario.rs`. Claims on LongMemEval knowledge-update questions: `scripts/memory-api/claims_ku.py`.

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
| `kleos_mem_entity` | `{ns, eid => name, norm, kind?, created}` | Entity rows, one per name form. Index `by_norm {ns, norm}`. Merges of rows into one entity live in `kleos_entity_merge` and review links in `kleos_entity_link` (E3's relations, `relation = 'kleos_mem_entity'`). |
| `kleos_mem_mention` | `{ns, id, eid}` | Memory → entity. Index `by_entity {ns, eid}`. |
| `kleos_mem_link` | `{ns, src, rel, dst => created, p?, decided_by?}` | Typed links between memories and entities. `remember` writes `new -[supersedes]-> old` with the decision's p and `decided_by`. |
| `kleos_mem_source` | `{ns, id, at, stmt => source?, text}` | Provenance: one row per statement behind a memory, so a duplicate adds a row here instead of a memory. |
| `kleos_mem_decision` | `{ns, new, cand, template => label, p, band, decided_by, reason?, status, at}` | The audit table: every decision that was applied (`status: applied`) or left for review (`pending`). |
| `kleos_mem_ns` | `{ns => seq}` | The namespace version (see "Concurrency"). |
| `kleos_mem_access` | `{ns, id => n, last}` | Access counts and the last access (µs), written by batched flushes (see "Ranking"). |
| `kleos_mem_life` | `{ns, id => expires?, archived?}` | Expiry and archive time (µs). Only memories with a TTL or in the archive have a row. |
| `kleos_mem_role` | `{ns, id => role}` | Who said a memory (`remember {role}`): `user`, `assistant` or `other`. Only memories stored with a role have a row; none means the user. |
| `kleos_mem_event` | `{ns, id, n => start, end, granularity, kind, expr}` | Event times: the temporal expressions in a memory's text ("yesterday", "last summer", "since 2019"), resolved against its `at` into `[start, end)` µs, one row per expression (see "Dates"). |
| `kleos_mem_config` | `{key => value}` | `embedder`, `dim`, `decider`, `version`, and `engram` when an Engram model is attached. It is created last, so its presence marks a complete schema. |
| `kleos_engram_queue` | `{ns, qid => op, model, mem_ids, old, text, at, meta, status, attempts, error, created, updated}` | Only with an Engram model attached (feature `engram`): one job per memory change for the overlay worker. See "Engram attachment". |
| `kleos_mem_branch`, `kleos_mem_copied`, `kleos_mem_merge` | see [memory-branches.md](memory-branches.md) | Branch rows, the copy-up log and the merge log (schema version 6). |

Ids are time-sortable strings: `m_…` for memories and `e_…` for entities. `kind` defaults to `fact`, `ns` to `default`.

The schema version is 6 (6 added the branch relations). A v1 or v2 store gains `kleos_mem_ns`, `kleos_mem_access`, `kleos_mem_life` and the entity merge and link relations on first use; a v3 store gains `kleos_mem_role`; a v4 store gains `kleos_mem_event`, filled from the stored memories (each text parsed against its first valid-from). A v5 store gains the branch relations, empty.

**Why vectors live in a sibling relation.**
- The dimension is unknown until an embedder is chosen. With a separate relation, a store created without one can gain vectors later: `::memory init {embedder: …}` creates the relation and backfills it.
- Search indexes on a valid-time relation also index superseded and retraction rows. The `current` flag lets HNSW filter them out inside the search (`filter: ns == $ns && current`) rather than after it.
- FTS stays on `kleos_mem`. Its stale rows are dropped by the `@` join after an over-fetch.

## Operations

### remember(text, opts)

Options:
- `ns`, `kind`, `source`, `importance` (in [0, 1]);
- `entities`: names the statement is about. Include the subject: `user` (or the user's name) for first-person statements, `Anna` for "my sister Anna …". They are resolved to the namespace's entities (see "Entity resolution") and gate supersession;
- `at`: valid-from, in µs; the default is now;
- `ttl` (µs after `at`) or `expires` (absolute µs): past it, recall leaves the memory out by default;
- `k`: the number of candidates, default 5;
- `decider`;
- `embedder`: only used when this call creates the store;
- `split`: `auto` (the default) stores a text of several sentences as one memory per claim (see "Claims"); `none` stores the whole text as one memory, as before claims;
- `min_similarity`: the cosine below which a claim's candidate is not sent to the decider (default 0.6; claims of a split text only);
- `role`: who said it, `user` (the default when unset), `assistant` or `other` (see "Speaker roles").

Steps:
1. Embed the text, if the store's embedder is enabled.
2. Find the top-k current memories of the namespace with the same hybrid search as `recall`: the vector leg uses the document vector, with an FTS fallback. Archived and expired memories are candidates too, but never duplicate targets (a restatement must not vanish into a hidden memory).
3. Resolve `entities` (see "Entity resolution") and read the candidates' entities.
4. **Same-subject gate.** When the new statement and a candidate both have entities, `supersedes` may act only if they share a canonical entity. If they share only an entity that one of the new names *may* be (an Auto or Tentative candidate that was not taken), the supersede is capped at pending. If they share nothing, the `supersedes` question is not asked at all; the candidate is listed in the result's `gated`. When either side has no entities, nothing changes. `duplicate` is not gated.
5. Exact rule, model-free: a candidate whose words equal the new text's (lower case, punctuation ignored) is a duplicate with band `auto`, `decided_by = kleos|duplicate@v1|exact`.
6. For the remaining candidates, run `decide(supersedes)` (not for gated ones) and `decide(duplicate)` with `(old = candidate, new = text)`. Each is one batched call over all candidates.
7. Act on the bands:

| Outcome | Action |
|---|---|
| `duplicate` is auto/yes and `supersedes` is not | Nothing is inserted. The statement becomes a `kleos_mem_source` row of the existing memory, and the result has `duplicate_of`. |
| `supersedes` is auto/yes and `duplicate` is not | The new memory is inserted. Each such candidate gets a retraction row at `at` (E1 semantics, values copied), `current = false` on its vector, and a `supersedes` link. Candidates whose validity starts after `at` (a late arrival) are left pending instead. |
| Tentative (label yes, band tentative); `supersedes` with p ≥ 0.5 below the calibrated threshold (the review band, below); or two auto answers that disagree | The memory is inserted and the decision is recorded `pending`. `recall` flags the older memory with `pending_superseded_by`, and can hide it with `hide_pending`. |
| None | Plain insert. |

All writes of one `remember` go in one imperative script, so one transaction: the insert, retractions, vector, links, source and audit rows, the expiry row, the event rows, and the entity rows, merges, review links and mentions. The candidate read, entity resolution and the model calls (about 0.5 s) happen before that transaction.

**Concurrency** works the same within one process and across processes sharing a database:

- **Version row.** Each namespace has a version row in `kleos_mem_ns {ns => seq}`. A `remember` notes the version before planning. Its write transaction then asserts that the version is unchanged (`:assert none`) and increments it. `forget` and `set` increment it too (the archive and expiry decide which memories may be duplicate targets).
- **Entities.** Entity resolution reads the namespace's entities while planning, and its writes (new rows, merges, review links, mentions) go in the `remember` transaction, under the same version check. Two concurrent `remember`s that both introduce "Anna" cannot create two Annas: the loser re-plans and finds the winner's row. Entity ids are fresh per plan, so merge and link keys never collide across namespaces. Hand edits through `::entity merge/split` do not bump the version; they change only how later statements are gated.
- **Access counts** are written outside all of this: `recall` buffers them in memory, and a flush is a separate small transaction on `kleos_mem_access` that never reads or writes the version row, so reads never conflict with `remember`.
- **Conflicts.** Two overlapping writers either fail that assertion or collide on the version row, depending on how the storage engine isolates transactions:
  - mem and SQLite serialize writers;
  - RocksDB and TiKV abort the later committer.
- **Re-planning.** The loser plans again and sees the winner. For "moved to Lisbon" and "moved to Porto" saved at the same moment, the second commit supersedes the first, with the decision audited. Two current answers can't result.
- **Time order.** `at` defaults to the planning time, taken after the version is read, so commit order and valid-time order agree.
- **Busy namespace.** After 8 attempts, `remember` reports the namespace as busy.
- **In-process gate.** Writers in one process also take turns under a per-namespace lock, so they rarely collide at all. The lock only speeds things up; correctness comes from the version row.
- **SQLite busy timeout.** SQLite connections wait up to 5 s for another process's lock instead of failing at once with `database is locked`.

Tests: `concurrent_remembers_never_leave_two_current` covers two threads held at a barrier. `concurrent_remembers_across_processes` covers two processes on one SQLite file, in the worst order: one commits a stale plan after the other has committed. With the in-transaction check removed, that test leaves both moves current.

The result:
- `action`: `added` or `duplicate`;
- `id`, `superseded` and `duplicate_of`;
- `tentative`: the pending decisions;
- `decisions`: all of them, including `none`;
- `gated`: candidates whose `supersedes` question was skipped (different subjects);
- `entities` (the entity rows mentioned), `entity_links` (per name: `eid`, `canonical`, `outcome` `match | possible | new`, `created`, `decided_by`, `p`);
- `embedded` and `decider`.

Each decision carries the candidate, template, label, p, band, `decided_by`, reason and action.

**Never deletes.** Supersession ends validity. Nothing in `remember` removes a row.

#### Claims (multi-sentence text)

Status: implemented (wave2/claims). Code: `kleos-core/src/memory_claims.rs` (splitter, declarative test) and `plan_remember` / `remember` in `memory.rs`.

Real agent input is a chat message, not a sentence. In LongMemEval's knowledge-update questions the update is one clause of a long message ("…Currently, my favourite is Kansas City Masterpiece BBQ sauce, but…") and the stale fact is one clause of an older one. Treating the message as one memory fails three ways: the old message is often not among the top-k candidates; "does this whole message supersede that one?" is the wrong question; and superseding the whole old message would also end its unrelated facts. So `remember` works per claim.

**Splitting** (`split: 'auto'`, the default). `memory_claims::split_claims` is rule-based and deterministic:
- lines first: a list bullet (`-`, `*`, `•`, `1.`, `2)`, `a)`) or a blank line starts a new item; a line that continues an unfinished one (starts in lower case, previous line without closing punctuation) is joined to it;
- then sentences: `.`, `!`, `?`, `…` (with closing quotes or brackets) followed by whitespace and a capital letter or a digit. A period that is not followed by whitespace never splits (decimals, URLs, e-mail addresses, version numbers). Titles and number words never end a sentence ("Dr.", "St.", "No.", "e.g."); words that may ("etc.", "p.m.", "U.S.", "Inc.", months) only before a capital; one-letter initials ("A. Smith") never; an ellipsis only before a capital;
- a claim with no letter or digit is dropped, and a repeated claim is kept once;
- **no content-free claims:** a sentence with fewer than two content words (words outside a function-word list: "Yes, I did.", "Cool!", "I loved it.", "How have you been?") joins the next sentence, or the previous claim when it is the last. "I live in Berlin." (two) stays a claim of its own;
- **speakers:** a speaker label at the start of the text ("Caroline: …", one to three capitalised words and a colon) is repeated on every claim, so "Caroline: I researched adoption agencies." keeps who said it. `is_declarative` looks past the label, and a claim of several sentences is declarative when any of them is.

Why (wave3/claims-fix): on LoCoMo, whose turns are "Name: text", plain sentence claims halved hybrid R@5. A short first sentence ("Caroline: Yes, I did.") became a memory whose vector was mostly the speaker's name, so it scored high for every question about Caroline and filled the top k; the claims after the first lost the name altogether. Measured on LoCoMo hybrid (categories 1–4, 1,536 questions) and the 23 held-out LongMemEval knowledge-update questions (latest answer first):

| Rules | LoCoMo R@5 | R@10 | KU latest first |
|---|---|---|---|
| whole turns, before claims (6f9711f) | 0.661 | 0.749 | 39% (prototype, whole messages) |
| sentence claims (92ef789) | 0.423 | 0.479 | 83% |
| join under 4 content words, no speaker label | 0.542 | 0.620 | |
| don't split under 25 words, join under 3, speaker label | 0.683 | 0.763 | 78%¹ |
| join under 4, speaker label | **0.691** | **0.775** | 74%¹; 83% with the declarative fix |
| **join under 2, speaker label (default)** | 0.678 | 0.765 | **83%** |

¹ Before the declarative fix: a question joined to a statement ("Can you recommend some options? By the way, I've found that 1 tablespoon…") made the claim non-declarative, so it was never decided. Now a claim is declarative when any of its sentences is.

The speaker label is most of the fix (0.542 → 0.691 at the same joining). Joining under 4 content words scores 0.013 higher on LoCoMo than under 2, but it merges short facts ("I live in Berlin. I have a dog named Max." becomes one claim), and superseding that claim would end the unrelated fact as well, which claims exist to prevent. So the default is 2. `SplitRules` and `split_claims_with` take other values.

A text that yields one claim is stored exactly as before: the trimmed text, under the plain statement id.

**Storage.** Each claim is a memory row (with its own vector, TTL row, importance and kind from the call's options). Its id is `<statement id>.<n>`, n from 1; a one-claim statement keeps the plain `m_…` id. Every claim's `kleos_mem_source` row holds the whole statement (`stmt` = statement id), so:
- recall hits of a claim carry `context`: the statement it came from (absent when it is the memory's own text);
- `forget` and `set` of a statement id cover all its claims (`forget` of a claim id covers that claim only);
- a claim that duplicates an existing memory adds a source row to it, as a whole statement did.

**Decisions per claim.**
- Only **declarative** claims are decided (`memory_claims::is_declarative`): not a question (`?`, or a wh-/auxiliary opening such as "Can you", "Do you", "What … " without a comma clause), not a request ("Please", "Suggest", "Recommend", "I need some ideas", "Any tips"…), not a greeting or thanks, and at least three words. The others are stored and recallable, and never candidates either: candidates are fetched 2k deep and the first k declarative ones are kept. With `split: 'none'` every statement and candidate is decided, as before.
- Candidates are searched per claim; the decider is **batched across all claims**: one `decide(supersedes)` and one `decide(duplicate)` call per `remember`, through the pair cache.
- **Similarity floor** (claims of a split text only): a candidate whose vector has cosine < `min_similarity` (default 0.6) with the claim is not sent to the decider. Each decision reports `similarity`.
- **Review noise** (claims of a split text only): of a claim's pending `supersedes` answers, only the one with the highest p stays pending; the others are recorded as `none`.
- A supersede ends **only the old claim**; the old statement's other claims stay current. A memory that one claim of the statement duplicates is not ended by another claim of the same statement (that supersede is left pending). Claims of one statement are not compared with each other.
- **Entities:** names are resolved once per statement. A name is linked to (mentions) the claims whose text contains it (case-insensitive, at word boundaries), or to every claim when no claim does, e.g. the subject `user`. The same-subject gate uses each claim's own entities.
- A single sentence keeps the old behaviour except that a question, request or fragment is not decided (it would otherwise be compared as if it stated a fact).

**One transaction.** All claims of a `remember` are written in its one script, under the namespace version assert, with one re-plan of the whole statement on conflict. With an Engram model attached, each added claim enqueues its own job (`write` or `supersede` with the old claims it ended, the claim's text, and the entity names linked to it), numbered `v+1 … v+n` from the planned version `v`; the version then advances by `n`, so the last job id is still the commit's version.

**Result.** For a split statement, `id` is the statement id (when any claim was added), `action` is `added` unless every claim was a duplicate, `superseded` is the union, and `claims` lists `{id, text, action, duplicate_of, superseded, decided}` per claim. Each decision names its `claim`. A one-claim statement's result is unchanged (no `claims`).

**Recall.** A question or request is about as relevant to a question as the fact that answers it ("how many stars do I need for gold?" against the stars I have), and chat is full of them. The blend subtracts `question_weight` (default 0.05) from a memory that is not declarative.

##### Results (LongMemEval knowledge-update, native API)

Each user turn of the oracle haystack goes to `remember` whole (`split: 'auto'`, no splitting in Python), with the session date as `at` and the session id as `source`; then `recall(question, k = 10)`. The metric is the prototype's: is the first hit from an answer session the latest answer session? decider-nano (nano-tune-v1) and EmbeddingGemma on Metal, from the Python wheel. Script: `scripts/memory-api/claims_ku.py` (runs and logs in `.bench-data/claims-impl/runs/`); the prototype is `.bench-data/claims-impl/claims_ku_prototype.py` (Python-side sentences, questions dropped).

The 23 held-out, non-abstention questions (`lme_ku_split()`):

| Unit | Decider | Latest first | Stale first | Latest in top 10 | … with `hide_pending` | Superseded / q | Pending / q | `remember` p50 |
|---|---|---|---|---|---|---|---|---|
| whole message (prototype) | no | 39% | 61% | | | | | |
| sentence (prototype) | no | 61% | 39% | | | | | |
| sentence (prototype) | yes | 83% | 17% | 100% | | 0.57 | 11.5 | 788 ms per sentence (whole message: ~8 s per turn) |
| **claims (native)** | **yes** | **83%** (19/23) | **17%** | **100%** | **91%** (21/23) | 0.43 | **1.96** (supersedes 1.09, duplicate 0.87) | **144 ms per statement** (p90 574), 55 ms per claim |
| claims, no similarity floor | yes | 83% | 17% | 100% | 87% | 0.57 | 2.43 | 702 ms per statement (p90 2069) |

- A turn has 2.8 claims on average; 67% of claims are declarative and decided.
- The four stale-first questions are the prototype's four. Two of them (Instagram followers 1250 → "close to 1300", bird species 27 → 32) are kept as pending supersedes (p 0.94 and 0.98), so `hide_pending` answers them.
- The floor cuts decider work 3.5× (540 against 1916 `supersedes` questions on these 23) and latency 4.9×. Below it, it would have auto-applied three more supersedes, two of them wrong ("I'll definitely check some of these out" superseded by "I think I'll stop here for now", p 0.997 at cosine 0.45). Every auto-supersede the floor kept had cosine ≥ 0.66.
- `question_weight` does not change the held-out first hit (83% at 0, 0.02, 0.05 and 0.1) and moves `hide_pending` from 87% to 91%.

**Review noise**, pending decisions per question on the same 23:

| | Pending / q |
|---|---|
| prototype (sentences through the old `remember`) | 11.5 |
| native: questions not decided, floor 0.6, the likeliest pending `supersedes` per claim | 5.96 (2.43 + 3.52 `duplicate`) |
| + `supersedes` kept only from p 0.9, the likeliest `duplicate` per claim from p 0.5 (final) | **1.96** |

What the pending pairs were: below p 0.9 they were almost all two sentences on one topic that are both true ("I've been taking a course on Coursera" / "I've been using Duolingo", "I'm thinking of trying out a new coffee recipe" / "… a new tea blend"). What is left are mostly near-misses of real updates (followers, species counts, "4 MCU films" → "12 films in the last 3 months", hours on a sculpture) and real paraphrases as duplicates ("I think I'll go with the Brown Butter Sage Sauce" / "I think I'll try the brown butter sage sauce"). The thresholds were chosen on these 23 questions, and no auto decision depends on them. They apply to the claims of split statements only. The life stream's three true misses scored 0.977 to 0.996, above 0.9.

**The 53 trained-on questions** (49 without abstention). decider-nano saw some of their answer pairs in training, so these numbers are flagged and not a held-out result:

| | Latest first | Stale first | Latest in top 10 | … with `hide_pending` | Superseded / q | Pending / q | `remember` p50 |
|---|---|---|---|---|---|---|---|
| claims, no decider | 45% | 55% | 100% | 45% | 0.04 | 0 | 27 ms |
| claims, decider (final) | 57% | 43% | 100% | 65% | 0.47 | 1.43 | 131 ms |
| … `question_weight: 0` | 55% | 45% | | | | | |

These are harder than the held-out ones on retrieval: in 9 of 49, the top hit was an earlier *question* about the same thing ("how many engineers do I lead in my new role?"), and many true updates score just under τ (p 0.996, pending).

**After the LoCoMo fix** (joining, speaker labels; same 23 questions, wave3/claims-fix): latest first 83%, stale first 17%, latest in top 10 100%, 91% with `hide_pending`, 0.43 superseded and 1.83 pending per question (1.09 supersedes, 0.74 duplicate); 2.6 claims per message instead of 2.8. LongMemEval_S (bench-l2): fts on all 500 questions R@5 0.952, R@10 0.958, T@5 0.869 (main 0.950, 0.960, 0.869); hybrid on the 119-question set R@5 0.992, R@10 0.992, T@5 0.910, identical to main (before claims: 0.992, 1.000, 0.930). Latency was measured on a machine shared with three other benchmark runs (p50 300 ms per message), so it is not comparable with the 144 ms above.

**Life stream** (40 one-sentence turns, so every statement is one claim): unchanged. Main's memory code and this branch, back to back on the same machine:

| | Before | After |
|---|---|---|
| Recall, current fact first | 15 / 20 | 15 / 20 |
| … with `hide_pending` | 19 / 20 | 19 / 20 |
| Auto-supersede | 12 right, 0 wrong | 12 right, 0 wrong |
| Pending supersedes | 3 true, 2 false | 3 true, 2 false |
| Duplicates | 4 right, 0 wrong | 4 right, 0 wrong |
| `remember` p50 | 343 ms | 344 ms |

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
- `ns`, `k` (default 10), `kind`;
- `entity`: a name. Every memory mentioning any row of that entity's cluster (its merged names) matches;
- `as_of`: valid time in µs;
- `vector_weight`: default 0.6;
- `hide_pending`: leave out memories with a pending `supersedes` by a memory valid at the read time. Default false;
- `include_archived`, `include_expired`: default false;
- ranking: `blend` (default true), `recency_weight`, `half_life_days`, `importance_weight`, `access_weight`, `question_weight` (see "Ranking");
- `record_access`: count the returned memories as accessed. Default true; read-only `::memory recall` scripts never count;
- diversity: `diversity` (default true), `max_per_statement` (default 1), `max_per_source` (default 0, no cap), `mmr_weight` (default 0), `pool` (see "Diversity");
- `expand` (default false) and `expand_weight` (default 0.5): entity-graph expansion (see "Entity expansion");
- `assistant_weight` (default 0.05) and `role` (a filter): see "Speaker roles";
- `from`, `to` (µs): only memories whose validity starts in `[from, to)` (see "Valid-from filters");
- `format`: `list` (the default) or `context`, with `budget` (tokens, default 2000) and `show_source` (default false): see "Context output";
- `hit_text` (`statement`, the default, or `claim`) and `text_chars` (default 1000): what a list hit's `text` is (see "List text");
- `event_from`, `event_to` (µs): only memories whose event time overlaps `[event_from, event_to)`; `event_weight` (default 0.03) and `query_at`: the question-date boost (see "Dates").

It is one script:
- The vector leg is HNSW over `kleos_mem_vec`, filtered to the namespace (and `current` unless `as_of` is set). Its score is −distance.
- The text leg is FTS over `kleos_mem`, filtered to the namespace. The query is the words of the text OR-ed, so user text is never parsed as FTS syntax.
- `WeightedFusion` combines them with weights `[semantic 0.6, text 0.4]` and the theory-based norms `neg_cosine_distance` / `bm25`. Min-max is unstable on the short lists memory search produces.
- The result is joined with `kleos_mem` `@ 'NOW'` (or `@ $as_of`), plus the kind and entity filters. Archived memories, and memories whose expiry is at or before the read time, are left out unless asked. The order is by relevance; with the blend on, a window of max(2k + 10, k) is fetched and re-ranked.
- Both legs over-fetch (4k + 16, and 4× more with a filter), because the indexes also hold stale versions.

Each hit has `id, text, score, relevance, recency, kind, source, created, importance, valid_from, valid_to, pending_superseded_by, access_count, last_access, expires, archived`; `context` (the whole statement) and `claim` (the memory's own text) when the memory is one claim of a longer text; and `events` (its resolved dates, see "Dates") when it has any. By default a claim hit's `text` is its statement (see "List text"); `memory_text()` in Rust, or `claim` falling back to `text`, is the stored text. The validity interval is the one that contains the read time; `valid_to` is null while the memory is still valid. Only memories valid at the read time are returned, so a superseded fact never shows up as current, and `as_of` reads the past.

The vector leg always returns the nearest neighbours, so recall returns up to k memories even when none are relevant. There is no score cutoff.

Hits also carry `role` (when stored) and `expanded` (true for a memory brought in by entity expansion). With diversity on, hits are in selection order, which is score order unless `mmr_weight` moved a near-duplicate down.

#### Diversity

Questions that need several pieces of evidence lose them when the top k fills with near-copies: with claims, three sentences of one message can take three of ten slots, and one session can take all ten. So recall chooses the k hits from a deeper pool:

1. The pool is the `pool` best by score (default max(3k, k + 10)), taken from the same index fetch as before. Deeper index legs were tried. They let short fragments that match only a name into the fusion, and cut LoCoMo hybrid R@5.
2. Greedy selection: the next hit is the candidate with the highest `score − mmr_weight · max similarity to the hits already chosen`, among the candidates within the caps. Similarity is the cosine of the two vectors when both have one, else the word overlap (Jaccard of words of three or more characters). Ties keep the higher-ranked candidate.
3. Caps: at most `max_per_statement` hits per statement (the claims of one text; a claim `m_….n` belongs to `m_…`), and at most `max_per_source` per source group. The source group is the part of `source` before the first `#`, so `chat-12#4` and `chat-12#9` are one conversation; a source without `#` is its own group. 0 means no cap.
4. The caps are soft: when they leave fewer than k candidates, the capped ones fill the rest in score order.

`diversity: false` is the top k by score, exactly as before (the same fetch, window and order). Everything is deterministic.

**Defaults.** One hit per statement; no source cap; no MMR penalty. The benchmark numbers that chose them are in [docs/benchmarks/memory.md](../benchmarks/memory.md) ("Recall diversity"). In short: the statement cap helps everywhere (multi-session and temporal Rall@10 on LongMemEval) and costs nothing on single-evidence questions. A per-session cap helps LongMemEval multi-session but hurts LoCoMo, whose evidence often sits in several turns of one session, so it is opt-in. The MMR penalty moved little either way.

#### Entity expansion

`expand: true` adds memories about the entities of the top hits to the pool:
- the seeds are the 3 most relevant hits;
- their mentions (`kleos_mem_mention`) give canonical entities, merges followed (`kleos_entity_merge`); every row of each cluster counts;
- an entity that more than 25 memories mention is not followed (the user, a speaker: a hub says nothing about this question);
- the other memories mentioning those entities, valid at the read time and not archived or expired, join the pool (up to 30) with relevance `expand_weight` × the best seed's relevance, flagged `expanded`. A related memory already in the pool (a seed included) is raised to that relevance when it is higher than its own;
- they then go through the blend and the diverse selection like any other candidate.

It is skipped when recall already filters by `entity`. It only does something when `remember` was given `entities`; there is no NER. On LoCoMo with crude capitalised-word entities it did not help: at weight 0.5 nothing changed, and from 0.9 it displaced better hits. That is why it is off by default (docs/benchmarks/memory.md).

#### Speaker roles

`remember {role}` records who said a memory: `user` (the default when unset), `assistant` or `other`. An agent can store the assistant's replies too, so questions about what the assistant said ("what was the third recipe you suggested?") can be answered, without letting them displace the user's facts:
- assistant and other memories are **never decided**: they never supersede or duplicate anything, and they are never candidates of a user statement. What the assistant said does not replace what the user said;
- recall subtracts `assistant_weight` (default 0.05) from an assistant memory's blended score, so of two equally relevant memories the user's ranks first;
- `recall {role}` filters by speaker (`user` includes memories with no role); hits carry `role`.

#### List text

With claims and the default statement cap, each hit is one sentence of a message. A reader that shows only each hit's `text` then loses the rest of the message: who is speaking (LoCoMo's "Caroline:" is on the first claim only), what "it" refers to, and the date words in the next sentence. On the 301 LoCoMo questions that reader scored 0.671 against 0.708 for the old top-k list. With the statement as `text` it scores 0.751, and 0.672 against 0.647 on 119 LongMemEval questions (docs/benchmarks/memory.md, "Dates and list text").

So by default (`hit_text: 'statement'`) a claim hit's `text` is its statement: the whole statement when it has at most `text_chars` characters (default 1000; 98% of LongMemEval's and all of LoCoMo's statements), else a window of that many characters around the claim, cut at word boundaries with "…". The claim itself is in `claim`, and the whole statement stays in `context`. A memory that is not a claim (one sentence, `split: 'none'`) is unchanged and has no `claim`.

- Ranking, diversity, the context block and `remember`'s candidates all work on the claims; only the returned `text` changes.
- **Compatibility.** `hit_text: 'claim'` returns the claim as `text`, exactly as before. Callers that compared `text` with a stored sentence should use `claim` (or `memory_text()`).
- With `diversity: false`, several claims of one statement can be returned, each with the same statement text; the default cap of one claim per statement avoids that.

#### Valid-from filters

`recall {from, to}` keeps memories whose validity (the interval containing the read time) starts in `[from, to)`: "what did I say in May". They are post-filters on a 4× deeper fetch. `as_of` is the other time question: the state of the world at a time. For when the thing a memory talks about happened, use `event_from` / `event_to` (see "Dates").

#### Dates (event times)

A memory's valid-from is when it was said. What it talks about often happened at another time: "I went to the support group yesterday", said on 2023-05-08, is about 2023-05-07. Mem0's extraction LLM writes such dates into its memories; kleos resolves them at write time with rules, no model.

**Parser** (`memory_time::parse_times(text, at)`): English temporal expressions resolved against `at` into `[start, end)` with a granularity (`day`, `week`, `month`, `season`, `year`) and a kind (`point`, `since`, `until`, `range`, `duration`):
- absolute dates in the usual formats ("May 7, 2023", "7 May 2023", "2023-05-07", "May 7" with the year from `at`), month and year ("March 2022"), years with a cue ("in 2019");
- relative days and parts of days ("yesterday", "tonight", "last night", "this morning", "the day before yesterday");
- offsets ("two days ago", "a month ago", "in three weeks");
- weekdays and weekends ("last Friday", "next Friday", "on Friday", "this weekend");
- relative weeks, months, years and seasons ("last week", "earlier this month", "last summer", "in March");
- spans ("since 2019", "until June", "for the past three weeks") and ranges ("from May 3 to May 7", "between 2019 and 2021").

Weeks run Monday to Sunday and seasons are northern meteorological ones. On 50 hand-labelled date-bearing claims from LoCoMo and LongMemEval, expression precision is 1.00 and recall 0.96, and interval precision 0.96 and recall 0.92 (docs/benchmarks/memory.md, "Dates"). The parser is conservative: a missed date is better than a wrong one. "may" the modal, "march" the verb, "second" as an ordinal, names like "June", vague times ("recently", "a few days ago", "the other day") and habits ("every Monday") are not dates. The rules for ambiguous cases are in the module's documentation.

**Storage.** `remember` parses each claim it adds, against the statement's `at`, and writes one `kleos_mem_event` row per expression in its single transaction. A duplicate adds no rows (the existing memory has its own). Retraction leaves them (history); `forget {mode: 'erase'}` removes them.

**Use.**
- **List hits** carry `events: [{start, end, granularity, kind, expr}]`, the memory's own.
- **The context block** shows them inline (and its header then explains the `(≈ date, "words")` notation): `- [2023-05-08 (Mon)] Caroline: I went to a LGBTQ support group yesterday. (≈ 2023-05-07 (Sun), "yesterday")`. Only a memory's own line carries notes: `said in` lines show the statement without them (dates there made the reader pick the older of two shown values on LongMemEval knowledge-update; docs/benchmarks/memory.md, "Knowledge-update dip"). An expression that names a year ("May 7, 2023") is not repeated.
- **`recall {event_from, event_to}`** keeps memories whose event time overlaps `[event_from, event_to)`. A memory without events of its own counts by its statement's (the other claims of the same message), else by its valid-from. Post-filters on a 4× deeper fetch, like `from` / `to`.
- **Question dates** (`event_weight`, default 0.03, the size of the recency weight; 0 turns it off): the dates in the query are resolved against `query_at` (default `as_of`, else now), and a memory whose event time (as for the filter) overlaps one of them gains the weight in the blend ("what did I do in May 2023?"). Not applied with `blend: false`. On LoCoMo it raised hybrid R@5 from 0.687 to 0.695 and temporal context QA from 0.813 to 0.822 (docs/benchmarks/memory.md, "Dates"). A question without a date is unaffected.

#### Context output

`recall {format: 'context', budget}` (Rust `Db::recall_context`, Python `recall(..., format="context")` or `recall_context`) returns `{text, tokens, budget, ids, omitted}`: the same hits as `recall` with the same options, as a Markdown block for an LLM reader. For example:

```
Recalled memories, grouped by topic, oldest first. [date] is when a statement was made or became true; (≈ date, "words") resolves its relative times; "previously" lines are older values it replaced.

## user
- [2026-01-11 (Sun)] I moved to Lisbon last month. (≈ December 2025, "last month")
  previously: "I live in Berlin." (from 2026-01-02 until 2026-01-11)

## Anna
- [2026-01-13 (Tue)] My sister Anna lives in Munich.
  said in: "My sister Anna lives in Munich. She is a nurse."
```

- **Grouping.** A memory goes under the canonical entity it mentions that most of the shown memories mention (ties by name); without entities, under its source group ("Conversation of <date>"); else "Other". Groups are in time order (by their first memory; ties by the best rank), and memories within a group by date, so the newest value reads last, as in a dated list. With groups ordered by relevance (the first version), LongMemEval knowledge-update QA was 0.69 against 0.81 for the dated list, so the order was changed (results in docs/benchmarks/memory.md). Claims of one statement are one item.
- **Each item:** `[valid-from date]`, `(until <date>)` for a past read, `(assistant)` / `(other)` for another speaker, the text, its resolved dates `(≈ date, "expression")` (see "Dates"), and the source with `show_source`. Source names are not shown by default; they are ids, not content.
- **`previously`:** the memories this one superseded (`supersedes` links), up to three steps back, each with its own validity: "previously: X (from d1 until d2)". It is the history kleos already stores, so the reader sees both the current value and the change.
- **`possibly replaced by`:** a newer memory with a pending `supersedes` decision about this one, marked unconfirmed.
- **`said in`:** for a claim, up to 600 characters of the statement it came from, centred on the claim and cut at word boundaries.
- **Budget.** Tokens are estimated as characters / 4. Items are added in rank order with their `previously` lines while they fit (a group heading counts too); then the `said in` lines, in rank order, while the budget lasts. Memories that did not fit are listed in `omitted`.
- **Deterministic.** The same store and options give the same text byte for byte.

**Measured** (docs/benchmarks/memory.md, "After" and "Dates and list text"):
- **LoCoMo, 301 questions, same reader and judge:** the context block scored 0.807 without dates and 0.814 with dates and question dates. The whole conversation scores 0.814. On the full temporal category (321 questions) it went from 0.769 to 0.822.
- **LongMemEval, 119 questions:** 0.681 without dates and 0.655 with them. All of the drop is four knowledge-update questions where the reader picked the older value; temporal-reasoning rose by one question.
- **The list output:** with the statement as `text` (the default since wave3/dates), it scores 0.751 on LoCoMo and 0.672 on LongMemEval, against 0.711 and 0.647 for main's top-k claim list.

`::memory recall '…' {format: 'context', budget: 800}` returns one row with the columns `text, tokens, budget, ids, omitted`; HTTP `POST /memory/recall {"query": …, "format": "context"}` returns the object.

#### Ranking (E4, C6)

```
score = relevance
      + recency_weight    · 2^(−age / half_life)                      age = read time − valid_from
      + importance_weight · (importance − 0.5)                        null importance counts as 0.5
      + access_weight     · min(1, ln(1 + accesses) / ln(1 + 100))
      − question_weight   · [the memory is a question, request or fragment]
```

| Option | Default | Notes |
|---|---|---|
| `blend` | true | false: `score = relevance` and the old fetch size and order, exactly |
| `recency_weight` | 0.03 | a memory stated now gets +0.03, one a half-life old +0.015 |
| `half_life_days` | 90 | the read time is `as_of` when given, so past reads decay from then |
| `importance_weight` | 0.1 | importance 1 → +0.05, 0 → −0.05, unset → 0 |
| `access_weight` | 0 | recorded, not ranked by default (below) |
| `question_weight` | 0.05 | subtracted when `memory_claims::is_declarative` is false (see "Claims") |

`relevance` is the `WeightedFusion` score, in [0, 1]. It is returned with every hit, together with `recency`, `importance` and `access_count`, so a caller can re-rank with its own formula. In CozoScript the signals are plain columns: `importance` on `kleos_mem`, `valid_from` from the validity, `n`/`last` on `kleos_mem_access`, `expires`/`archived` on `kleos_mem_life`. For example:

```
?[id, text, s] := *kleos_mem{ns: 'u', id, text, importance @ 'NOW'},
                  *kleos_mem_access{ns: 'u', id, n}, s = coalesce(importance, 0.5) + 0.1 * ln(1 + n)
:order -s
```

Why these defaults:
- **Scale.** The weights are small because the relevance of the top hits is compressed. With EmbeddingGemma, distinct facts near the top of a short memory list often differ by 0.01 or less, so a signal weighing 0.1 decides between them regardless of relevance.
- **Recency** favours the newer of two similarly relevant memories, which is usually the current one. Decay is from the read time, so after a long idle period every memory has decayed alike and relevance decides. The defaults come from a sweep on the life stream read at its own "now" (day 126), where recency matters (half-life 7, 30 or 90 days × weight 0 to 0.1). 0.1 / 30 days lost 3 of 20 answers with `hide_pending`: recent unrelated memories ("I read a lot of science fiction") climbed over the answer. Every weight up to 0.05 at 90 days, and up to 0.03 at 30 days, lost none; from 0.02 up they also cut stale memories in the top 3 from 5 to 4 without `hide_pending`. A 7-day half-life lost answers from 0.03. 0.03 / 90 days sits inside that safe region on both axes.
- **Importance** is never inferred: the `important` template is unvalidated. It is the caller's (`remember {importance}`, `set {importance}`), and unset memories are neutral.
- **Access counts** weigh 0 by default. Every returned hit counts as an access, including the nearest-neighbour filler of unrelated queries, so a positive weight is self-reinforcing. On the life stream, 0.05 flipped a near-tie (a relevance gap of 0.0007) the wrong way. They are recorded and returned either way; pass `access_weight` to use them.

**Access counting.** `recall` adds the returned ids to an in-memory buffer of its `Db` (one map update, no write). The buffer is flushed by the `recall` that finds 64 memories pending or 1 s since the last flush, or by `Db::memory_flush` / `::memory flush`. A flush is one small transaction: `n = stored + buffered`, only for memories that still exist. A failed flush (read-only store, conflict) keeps the counts buffered (up to 100 000 memories). `recall` adds buffered counts to stored ones, so ranking sees exact counts. Counts still buffered when the process exits are lost; call `memory_flush` before closing a database to keep them.

#### Expiry and the archive tier

- `remember {ttl}` (µs after `at`) or `{expires}` sets an expiry. From then on, recall leaves the memory out unless `include_expired`. A past read (`as_of`) before the expiry still sees it. Expiry does not end validity.
- **Archive:** `forget {mode: 'archive'}` (by id, ids, kind, entity, `expired: true` or `all`) or `set {archived: true}` moves memories to the archive tier. They stay valid and keep their history, links and vectors; recall leaves them out unless `include_archived`. `set {archived: false}` brings them back.
- `forget {mode: 'archive', expired: true}` archives every current memory past its expiry.

### set(id | ids, opts)

Changes scoring metadata; the text and validity are untouched.
- `importance` in [0, 1]. It is rewritten in place on the memory's current row: importance is not versioned, so `@ T` reads see the new value too.
- `ttl` (µs from now), `expires` (absolute), or `no_ttl: true` to remove the expiry.
- `archived: true | false`.

It commits under the namespace version, like `forget`. It returns `{ns, ids}`: the ids that are current.

### flush

Writes the buffered access counts now. Returns `{flushed}`, the number of memories written.

### forget(id | filter, mode)

- Targets: `id` or `ids`, or a filter `kind` / `entity` / `expired: true` (current memories past their expiry, not yet archived), or `all: true` for a whole namespace. A call without any of these is an error.
- `mode: 'retract'` (the default) ends validity at `at` (default now) with a retraction row and clears `current` on the vector. History and `as_of` recall keep the memory.
- `mode: 'archive'` moves current memories to the archive tier (above).
- `mode: 'erase'` (GDPR) runs in two steps:
  1. One transaction removes the memory's vector, source rows, mentions, links and audit rows, and any entity left with no mention. When an Engram model is attached, the same transaction enqueues the overlay erase.
  2. `::erase kleos_mem` then deletes every version of the keys, with their FTS entries, and writes the salted `kleos_evict_audit` marker.
- Returns `{mode, ids}`: the memories that were current (retract) or existed (erase).

### reflect(ns, opts)

Minimal and model-free; real consolidation is E7. Options: `since` (µs, default 7 days ago) and `limit` (default 50).

It returns:
- `pending`: the tentative decisions, newest first, with both texts;
- `changes`: `::changes kleos_mem {from: since, prefix: [ns]}`, i.e. `added` and `ended` by valid time;
- `contradictions`: pending `supersedes` decisions, plus pairs of current memories that share an entity (canonical, so merged names count as one) and whose values differ by the value rules (`update` or `hint`). Up to 30 memories per entity are compared.

There is no op yet to accept or reject a pending decision. To act on one, `forget` the old memory or the new one.

### link(from, to, rel)

Adds `kleos_mem_link {ns, from, rel, to}` with optional `p`. Both ends must be memories or entities of the namespace.

### Engram attachment (feature `engram`)

`::memory init {engram: 'nano'}` attaches the store to a registered Engram model ([engram-overlay.md](engram-overlay.md), "Memory attachment"); `engram: ''` detaches. While attached, each memory change also writes a job to `kleos_engram_queue`, **inside the change's own write script**, so one transaction:
- `remember` that adds a memory: `write`, or `supersede` with the superseded ids; a duplicate writes none;
- `forget` retract: `retract` with the ids that were current, at the retraction time;
- `forget` erase: `erase` with the ids that existed. It also clears the text of earlier jobs of those memories.

The job's `qid` is the `kleos_mem_ns` version its commit produces, computed in the transaction before the version bump. So the concurrency rules above cover the queue too: a plan that lost the race writes neither its memory nor its job, and jobs are numbered in commit order per namespace. The edits themselves are computed outside kleos by `scripts/engram/engram_worker.py`; `::engram queue {ns}` shows pending and failed jobs and the last error.

The hook is one function, `memory_engram::enqueue`, called from the write-script assembly of `remember` and of both `forget` modes. It returns nothing when no model is attached or the feature is off, so the memory API's behaviour and cost are unchanged then.

`forget` erase runs `::erase kleos_mem` as its own statement, after the transaction that cleans up and enqueues the Engram erase job. A crash between the two leaves the memory present and its overlay being erased. Repeating the `forget` completes it, since erase is idempotent. The opposite order could have left an erased memory whose Engram rows still answer.

### Branches

`fork`, `rewind`, `merge`, `diff`, `discard` and `branches` are memory ops too (`::memory fork {from, to}`, `Db::memory_fork`, Python `memory_fork`, `POST /memory/fork`, …). A branch is a copy-on-write view of its parent. Every op above works on a branch, with reads routed through the lineage and writes to inherited memories copied up. See [memory-branches.md](memory-branches.md).

### Ontology

A namespace can have an active ontology (`memory_init {ontology: 'personal', ns}`, or `::ontology activate {ns, ontos}`; see [ontology.md](ontology.md)). With one, `remember`:

- **types each claim** against it: small talk (reactions, greetings, thanks, sympathy, remarks about the listener, photo-only turns), or a property of the ontology (cue phrases plus the nearest property prototype under the store's embedder), or untyped; plus a subject (the speaker, `user`, `user.sister` for "my sister …", or a leading name). The types are returned in `types` and stored in `kleos_mem_type`;
- **never decides small talk**: it is stored and recallable, but takes part in no `supersedes` or `duplicate` decision, as a new claim or as a candidate;
- **applies the cardinality policy** to every (claim, candidate) pair: different known subjects → no decision (`ontology_blocked`); same single-valued temporal property → today's decision; same multi-valued property, or different properties → `duplicate` only (`ontology_gated`); same single-valued non-temporal property (a birthday) → a supersede is capped at pending; either side untyped → per the namespace's `untyped` setting (`decide`, `pending` or `dup_only`);
- **writes kinship and role facts by rule** ("my sister Anna …" → `sister_of(anna, user)`) to `kleos_onto_fact`, in the same transaction.

Without an active ontology, or with `policy: 'off'`, `remember` behaves exactly as before.

### Entity resolution

`remember` resolves the names in `entities` with E3's `resolve_entities` ([entity-resolution.md](entity-resolution.md)) over the memory schema's own entity relation, `kleos_mem_entity`, scoped to the namespace (`EntityConfig.scope = ("ns", ns)`). There is no NER: the names are the caller's.

- Mentions are passed without context. So an exact, unique name of the namespace (after E3's normalisation: honorifics, punctuation and company suffixes dropped) is the same entity by rule, with no model call. Other near names (initials, nicknames, first names, MinHash) go to `decide(same_entity)` with the store's decider, or are judged by the rules when there is none.
- **Auto → link to the existing entity.** The memory mentions the cluster's row with the same name when there is one. A new name form ("Dr. Alice Smith" for "Alice Smith") gets its own row and a merge row into the canonical entity (`decided_by`, p and reason kept).
- **Tentative → a review link.** A new entity, with a `kleos_entity_link` row (status `pending`) to each likely match. `::entity pending kleos_mem_entity` lists them; `::entity merge` / `reject` settle them.
- **None → a new entity.**
- **Non-transitive.** Each name is decided against one canonical profile. Other Auto or Tentative candidates only get review links.
- **Non-destructive.** Merges are valid-time rows; `::entity split kleos_mem_entity <eid>` undoes one, and later reads (the entity filter, the gate, reflect) follow the current merges.

Resolution is read-only while planning; its writes go in the `remember` transaction (see "Concurrency"). The entity filter of `recall` and `forget` matches every row of the named entity's cluster.

### Automatic entities

With an entity extractor configured (`::memory init {extractor: 'gliner'}`, Python `memory_init({"extractor": "gliner"})`) and no `entities` passed, `remember` finds them itself. The extractor is enabled like an embedder or decider (`::extractor enable gliner {path: '<checkpoint>', precision: 'f16', device: 'metal'}`, `Db::enable_extractor`, Python `enable_extractor`, `::extractor list | disable`). The built-in one is GLiNER2.5 (`kleos-infer`, feature `infer`), a DeBERTa-v3-base span model ported to candle; The default checkpoint is kleos's fine-tune [`Mazino0/kleos-gliner`](https://huggingface.co/Mazino0/kleos-gliner) (Apache-2.0), downloaded on enable; `model: 'gliner-base'` selects upstream `fastino/gliner2.5-base-v1` (better for open label sets such as domain ontologies), and `$KLEOS_GLINER_MODEL` or `path` any other checkpoint. Research: [gliner.md](../research/gliner.md), [gliner-usage.md](../research/gliner-usage.md).

Options of `remember`: `extract` (default on; `false` off for the call, `true` requires an extractor), `extractor` (name for the call), `extract_granularity` (`statement`, the default, or `claim`), `extract_thresholds` (per label).

**The prompt matches training.** The four bare labels `person`, `location`, `organization`, `relative`, in that order, no descriptions; gliner2's word splitter; the text lowercased by the model (offsets index the original). Texts over 384 words are read in 384-word windows overlapping by 64; a span seen in two windows is kept once.

**Once per statement.** The extractor runs on the whole statement and each span goes to the claim whose byte range holds it (`extract_granularity: 'claim'` runs once per claim instead). On the 200-claim gold set, statement granularity scored names F 0.907 against 0.898 per claim (location 0.957 vs 0.917; the other labels equal).

**Policy** (`memory_extract.rs`):
- A `Name:` speaker prefix is stripped before extraction. First-person words then mean that speaker; without a prefix they mean `user` in the user's own statements (`role` unset or `user`), and nobody otherwise.
- Per-label thresholds on the raw score: person 0.8, location 0.9, organization 0.8, relative 0.5 (tuned on the fine-tune's validation split with this prompt). gliner2 lowercases its input, so case comes from the original text: in a statement with mixed case a person, location or organization must contain a capital letter and not be a pronoun; in an all-lowercase statement it needs a score of at least 0.9 instead. `relative` is exempt from the case rule.
- **Relatives are nodes.** A relative ("my sister", "Mom") becomes a person entity named `<possessor>'s <relation>` (`user's sister`, `Caroline's mother`), plus a typed edge `possessor -[relation]-> node` in `kleos_mem_link` (`decided_by: gliner-rule`, p = the span's score). The relation is one lower-case word (the head noun, with `Mom` → `mother`, `best friend` → `bestfriend`, …). Only first-person forms ("my", "our", a bare "Mom" or "Dad") belong to the speaker; "his/her/their X" belongs to the nearest person named before it in the statement and is skipped when there is none; "your X" is skipped.
- **A name with its relative is the same person.** "my sister Anna", "my sister, Anna", "my sister (Anna)", "Anna, my sister": the name is merged into the node (`kleos_entity_merge`, `decided_by: gliner-rule`). The side this statement created folds into the other's canonical entity; two existing, different entities get a review link instead.
- **The subject** (`user` or the speaker) is a claim's entity when the claim has a first-person subject word ("I", "we", "me", …) or a first-person possessive that is not part of a relative ("my car" counts, "my sister" does not). So "My sister moved to Berlin" (entities `user's sister`, `Berlin`) and "I live in Paris" (`user`, `Paris`) share no entity, and the same-subject gate keeps the first from superseding the second.
- **Pronouns.** A claim with no person or relative of its own and a third-person pronoun ("She moved to Lisbon.") inherits the nearest person or relative mentioned before it in the same statement. No coreference model and no cross-turn resolution: a later bare "my sister" finds the node by name.
- Kinship and role edges are written as found; they never supersede one another (two sisters are both true). Role edges (boss, partner) change over time: deciding which one is current is left to typed-edge handling (the ontology work), which can read the edges below.
- General relations from GLiNER (works_at, lives_in) are not written: zero-shot relation F1 is about 0.2–0.37 (gliner-usage.md §4).

**Stored.** Every span the extractor returned for a memory, kept or not, goes to `kleos_mem_extract {ns, id, n => label, start, end, text, score, kept, eid, extractor, at}` (schema v7): the raw score and the extractor version (`gliner2.5:<model>:<precision>`), so thresholds can be retuned and memories re-extracted later without guessing what produced a mention. Erase, branch copy-up and discard handle the rows like the other per-memory relations. The result of `remember` reports `extractor` (the version) and `edges`.

**Edge output shape** (for typed-edge consumers): `kleos_mem_link {ns, src, rel, dst => created, p, decided_by}` with `src` and `dst` entity ids of `kleos_mem_entity` (resolve canonical ids through `kleos_entity_merge`), `rel` the relation word, `decided_by = 'gliner-rule'`; and in `RememberResult.edges`: `{src, src_name, rel, dst, dst_name, p}`. Relative nodes are ordinary `kleos_mem_entity` rows whose name has the `<possessor>'s <relation>` form; they are linked from the memories that mention them (`kleos_mem_mention`) like any entity.

**Precision.** f16 is the default; f32 and q8 (weight-only) are options. On the gold set every precision gives the same spans at the thresholds above (zero label flips, Rust vs gliner2 at f32 identical). LayerNorm, relative positions and the span head always run in f32.

## Surfaces

The same options everywhere: one JSON object per op, handled by `Db::memory_call(op, args)`.

**Rust** (`Db` and `DbInstance`):

```rust
use kleos::memory::{RememberOpts, RecallOpts, ForgetOpts, ReflectOpts, LinkOpts, InitOpts};
db.memory_init(InitOpts { embedder: Some("gemma".into()), decider: Some("nano".into()), ..Default::default() })?;  // engram: Some("nano".into()) attaches an Engram model
let r = db.remember("I moved to Lisbon.", RememberOpts { ns: Some("alice".into()), ..Default::default() })?;
let hits = db.recall("where do I live?", RecallOpts { ns: Some("alice".into()), ..Default::default() })?;
db.forget(ForgetOpts { ns: Some("alice".into()), id: r.id, mode: Some("erase".into()), ..Default::default() })?;
db.reflect(ReflectOpts { ns: Some("alice".into()), ..Default::default() })?;
db.link(LinkOpts { ns: Some("alice".into()), from: a, to: b, rel: "works_at".into(), p: None })?;
db.memory_set(SetOpts { ns: Some("alice".into()), id: r.id.clone(), importance: Some(0.9), ..Default::default() })?;
db.memory_flush()?;
db.memory_call("recall", serde_json::json!({"query": "where do I live?", "ns": "alice"}))?;
```

**CozoScript.** The positional argument is the text (remember), the query (recall), the id or list of ids (forget) or the namespace (reflect); the rest are options:

```
::memory init {embedder: 'gemma', decider: 'nano'}
::memory init {engram: 'nano'}             # feature engram: remember/forget enqueue Engram jobs
::memory remember 'I moved to Lisbon.' {ns: 'alice', entities: ['Lisbon'], source: 'chat-12'}
::memory remember 'I moved to Lisbon. My rent is €900.' {ns: 'alice', split: 'none'}   # one memory, not two claims
::memory recall 'where do I live?' {ns: 'alice', k: 5, as_of: 1767225600000000}
::memory forget $id {ns: 'alice', mode: 'erase'}
::memory forget {ns: 'alice', kind: 'preference'}
::memory forget {ns: 'alice', expired: true, mode: 'archive'}
::memory set $id {ns: 'alice', importance: 0.9, ttl: 86400000000}
::memory set $id {ns: 'alice', archived: false}
::memory flush
::memory reflect 'alice' {since: 0}
::memory link {ns: 'alice', from: $a, to: $b, rel: 'works_at'}
```

Output:
- `recall` returns one row per hit: `id, text, score, kind, valid_from, valid_to, created, source, importance, pending_superseded_by, relevance, recency, access_count, last_access, expires, archived, context`.
- The other ops return one row with a column per result field. Lists of plain values become lists; records become JSON.
- `recall` and `reflect` run in read-only scripts (a read-only recall does not count accesses). The others are refused there, and inside imperative blocks, because they run their own transactions.
- Unknown options are errors.

**Python:**

```python
db.memory_init(embedder="gemma", decider="nano")
r = db.remember("I moved to Lisbon.", ns="alice", entities=["Lisbon"])   # dict
db.remember("I moved to Lisbon. My rent is €900.", ns="alice")          # two claims: r["claims"]; split="none" for one memory
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
- no ML at all: lazy init, FTS only, value-rule supersede, exact duplicate;
- entity resolution: one entity per name across memories, a new name form merged (and split back), initials as a review link that caps a supersede at pending, the entity filter following merges, erase removing entities with their merges and links (`entities_resolve_merge_review_and_split`);
- the same-subject gate: the sister's city never supersedes the user's, the user's move does, and a statement without entities is not gated (`entities_gate_supersedes_to_shared_subjects`); two concurrent `remember`s introducing the same new entities leave one row each (`concurrent_remembers_never_leave_two_current`);
- ranking: recency decay and half-life, importance from `remember` and `set`, blend off = relevance with unchanged scores (`recency_and_importance_rank_and_blend_off_is_relevance`);
- access counts: buffered, flushed, not counted by read-only scripts, flushes never touch the namespace version, ranked only with `access_weight`, dropped on erase (`access_counts_are_buffered_flushed_and_ranked`);
- TTL and the archive tier: exclusion, `as_of` before expiry, no duplicate into an expired memory, archiving expired memories, archive and restore by id, TTL set and removed later (`ttl_expiry_and_archive_tier`);
- recall diversity, the statement and source caps, MMR and `diversity: false` (`recall_diversity_caps_and_mmr`); entity expansion with merges and the hub limit (`recall_entity_expansion`); the context block: groups, dates, `previously`, `said in`, the budget, `as_of`, determinism, and the JSON and CozoScript surfaces (`recall_context_current_previously_and_budget`); roles never decided, `assistant_weight`, the role filter, `from` / `to`, and erase (`remember_roles_and_time_filters`); the selection and window helpers are unit tests in `memory_recall.rs`;
- claims: ids, provenance, `context`, `split: 'none'` and the CozoScript/JSON surfaces (`claims_split_store_and_recall_with_context`); a supersede that ends only the old claim, and the similarity floor (`claims_supersede_only_the_old_claim`); questions and requests never decided on either side (`claims_questions_are_not_decided`); `forget` / `set` of a statement id (`claims_forget_a_statement`); entities per claim and the per-claim gate (`claims_link_entities_per_claim`); two concurrent split statements (`concurrent_claim_remembers_never_leave_two_current`); `question_weight` (`claims_questions_rank_below_statements`); one Engram job per claim with consecutive qids (`claims_enqueue_one_job_each` in `tests/memory_engram.rs`). The splitter's hard cases are unit tests in `memory_claims.rs`.

- list text and dates: a claim hit's `text` is its statement, `claim` the claim, `hit_text: 'claim'` and `text_chars` (`claims_split_store_and_recall_with_context`, `list_text_is_the_statement_by_default`); event rows written per claim, events on hits, the dates in the context block (and none on `said in`), `event_from` / `event_to` with the statement and valid-from fallbacks, the question-date boost, erase, and the v4 migration with backfill (`event_times_stored_shown_filtered_and_erased`); the parser's own unit tests (`memory_time::tests`: every rule with exact dates, 50+ negatives, overlaps, Unicode).

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
- The misses all sit just under τ. The review band catches them, but it also flags different-subject pairs; the entity gate below removes most of those.

#### With entity resolution and scoring (wave2/mem-scoring)

Each turn now passes `entities`: the subject (`user` for first-person statements, `Anna`, `Max`) plus the named things (`scripts/memory-api/life_stream.py`). The "before" binary is main at 276f90f, run on the same annotated stream (its exact-name hook links entities but nothing reads them). Same machine, same models, run back to back.

| | Before | After |
|---|---|---|
| Recall, current fact first (top-1) | 15 / 20 | 15 / 20 |
| … with `hide_pending` | 18 / 20 | **19 / 20** |
| False pending supersedes | 6 | **2** |
| True pending supersedes (the 3 misses below τ) | 3 | 3 |
| Auto-supersede | 12 right, 0 wrong | 12 right, 0 wrong |
| Duplicates | 4 right, 0 wrong | 4 right, 0 wrong |
| `supersedes` questions skipped (different subjects) | – | 22 |
| Remember p50, ms (3 runs; load average ~20 from other builds) | 395, 455, 539 | 427, 445, 548 |
| Recall p50, ms, first pass (query embedded) | 11.2, 11.8, 14.0 | 14.2, 18.5, 22.2 |
| Recall p50, ms, query vector cached | – | blend on 4.1, blend off 6.2 |

- **Gate.** The four false pendings removed are the sister against the user's home (three pairs, both directions) and the sister's move against the user's move. The two left share the subject: the cat against the dog (both the user's pets), and "moved again, from Lisbon to Porto" against "I live in Berlin" (Berlin is still current because Berlin → Lisbon was missed below τ, so that one is arguably right). The `hide_pending` gain is the time-travel question the sister's false pending used to hide.
- **Recall order.** At the real read time (months after the stream) recency has decayed for every memory, and the blend gives exactly the relevance answers: 15 and 19 with the blend on or off. Read at the stream's own "now" (day 126), the default blend also keeps 15 and 19 and cuts stale memories in the top 3 from 5 to 4. Higher recency weights lose answers (sweep in "Ranking").
- **Latency.** Remember is within run-to-run noise: resolution with exact names needs no model call, and gated candidates skip one. The cached-query recall cost is the same with the blend on or off. The first-pass recall difference (3 to 8 ms) is mostly the query embedding on a loaded GPU; recall does no more index work than before, since the blend re-ranks a wider window (2k + 10) of the same index fetch.
- Not exercised by this stream: importance (no turn sets it) and access counts (weight 0). The deterministic tests cover both.

## Caveats

- **Claims.** The similarity floor (0.6) and the review thresholds were measured with EmbeddingGemma and decider-nano on 23 questions. Another embedder has another cosine scale; pass `min_similarity` for it. Claims are decided without their neighbours: a claim like "It's better now that it's aligned with my gym days" carries no subject (no coreference is resolved), though recall returns the statement as `context`. Claims of one statement are not compared with each other. A legacy whole-message memory that ends in a question is no longer a candidate in `auto` mode.

- The candidate read and model calls run outside the write transaction (above).
- The decider sees only the top-k candidates. A stale fact phrased very differently from its update may never be compared. `reflect` surfaces the ones that share an entity.
- Recall over-fetches to skip stale index rows. A namespace with very many superseded versions of one topic could still crowd a current memory out of the fetched window.
- One vector dimension per store. Changing to an embedder with another dimension is refused; re-embedding is not implemented.
- `importance` is never inferred: the `important` template is unvalidated.
- **The gate depends on the caller's entities.** It only helps when statements name their subject. Named things alone ("Berlin", "Lisbon") would block true supersedes ("I live in Berlin" → "I moved to Lisbon" share no entity), so the subject must be included. A wrong entity can block a true supersede; the decision is then simply not asked (listed in `gated`), never acted on.
- Resolution scans the namespace's entities once per `remember` with entities (E3's one-scan-per-call), and the merge snapshot covers every namespace's merges of `kleos_mem_entity`. Fine for thousands of entities per store, not for millions.
- Access counts are approximate: counts buffered when a process exits are lost (call `memory_flush`), and a flush that keeps failing caps the buffer at 100 000 memories.
- Archived and expired memories can still crowd the recall window: they are filtered after the index fetch, like stale versions.
- Importance is rewritten in place on the current row, so it has no history.
- There is no API to resolve pending decisions yet; consolidation is E7.
