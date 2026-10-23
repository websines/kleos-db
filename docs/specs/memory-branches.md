# Memory branches: fork, rewind, merge, diff

Status: spec (wave4/fork, 2026-10-24). Implementation follows on the same branch; sections marked *design only* are not built yet.

Code (planned):
- `kleos-core/src/memory_branch.rs`: lineage, copy-up, and the `fork`, `rewind`, `merge`, `diff`, `discard` and `branches` ops;
- small hooks in `kleos-core/src/memory.rs`: each read is routed through the lineage, and each write that touches an inherited memory copies it up first;
- `kleos-core/src/entity.rs`: the entity snapshot reads several scopes, each up to a creation time;
- `kleos-core/src/engram.rs` (feature `engram`): overlay reads and writes follow the lineage;
- surfaces: `::memory fork | rewind | merge | diff | discard | branches`, `Db::memory_call`, Python methods, `POST /memory/:op`.

Tests (planned): `kleos-core/tests/memory_branch.rs` (fake models). End to end, with real models: `scripts/branches/branches_demo.py`.

## Goal

An agent tries something without committing to it. It forks its memory, works in the fork, then keeps the result (merge) or drops it (discard). Several agents can work from one memory in parallel, each in its own branch. Any namespace can be put back to an earlier state (rewind), and two namespaces can be compared (diff).

Ideas taken from Zed's DeltaDB (operation history, thread fork and rewind) and from copy-on-write file systems, built on what kleos already has:
- valid time (`Validity`): every memory change is an appended row, so any past state can be read;
- namespaces: every memory relation is keyed by `ns` first;
- the `remember` write path (decider-nano supersession and duplicate detection), which merge reuses to settle conflicts by meaning, not by text.

Requirements:
1. **Fork is O(1).** It writes one row, whatever the size of the parent.
2. **Isolation in both directions.** A branch never writes its ancestors' rows. Ancestors' changes made after the fork are not visible in the branch.
3. **Merge resolves conflicts semantically.** "I moved to Lisbon" merged into a memory that holds "I live in Paris" supersedes it, through the same decision path as a direct `remember`.
4. **Nothing is lost.** Rewind and merge only append. Discard deletes only the branch's own rows.

## Model

### Lineage

A branch `B` forked from `P` at time `c` is a row `kleos_mem_branch {ns: B => parent: P, at: c, created, status}`. `P` may itself be a branch, so a namespace has a **lineage**: levels `L0 = B, L1 = P, L2 = P's parent, …` up to a root (a namespace with no branch row). Each ancestor level `Li` has a **cut** `ci`: the minimum of the fork times on the path from `B` to `Li`. Level 0 has no cut. Depth is limited to 16.

A namespace without a branch row is a plain namespace with a one-level lineage. Every existing behaviour, query and cost for plain namespaces is unchanged; the lineage code runs only for branches.

### Visibility and ownership

A memory id is **owned** by the nearest level that has any `kleos_mem` row for it. Memory ids are globally unique (`m_…` plus random bits), so one id names one memory everywhere in a family of branches.

A branch's **view** at read time `T` (now, or `as_of`) is:
- the memories owned by level 0, valid at `T`;
- plus, for each ancestor level `Li`, the memories it owns that are valid at `min(T, ci)`.

All of a memory's other rows (vector, sources, mentions, role, expiry and archive, event times) are read from its owner. Decisions and links are not per memory; they are read from every level, with an ancestor's rows limited to those written at or before its cut (`at`, `created` ≤ `ci`).

### Copy-up

A branch never writes an ancestor's rows. When a write in the branch touches an **inherited** memory (one owned by an ancestor), it first **copies the memory up** into the branch, in the same transaction as the write:
- its `kleos_mem` versions with valid time ≤ the owner's cut (original valid times kept, so `as_of` reads before the copy still see the inherited history);
- its vector, with `current` recomputed at the cut;
- its source rows (statements at or before the cut), mentions, role, event times and expiry/archive row.

The branch then owns the memory, and the ancestor's copy is masked in the branch's view. The write applies to the branch's copy. A log row `kleos_mem_copied {ns, id => from, at}` records the copy.

Writes that copy up:
- `remember`: a candidate it supersedes, or a duplicate target (the new source row goes with the memory);
- `forget` (retract, archive), `set`, and `rewind` (retract or restore).

`link` and decision rows are written in the branch and never copy up.

Copy-up is O(size of that one memory), so a branch costs storage only for what it changes.

## Schema (version 6)

| Relation | Columns | Notes |
|---|---|---|
| `kleos_mem_branch` | `{ns => parent, at, created, status}` | One row per branch. `at` is the fork time (valid time of the parent's state); `status` is `active` or `discarded` (a tombstone, see "discard"). |
| `kleos_mem_copied` | `{ns, id => from, at}` | Copy-up log: memory `id` was copied into `ns` from `from` at time `at`. |
| `kleos_mem_merge` | `{into, stmt => from, at, action, ids}` | Merge log: statement `stmt` of branch `from` was replayed into `into`; `action` is `added`, `duplicate` or `skipped`; `ids` are the resulting memory ids. Makes merge idempotent. |

A v5 store gains the three relations on first use; nothing is backfilled.

## Operations

All take `ns`-style namespace names. Results are JSON objects (one row per field in CozoScript, as for the other `::memory` ops).

### fork {from, to, at?}

Creates branch `to` of `from` as of `at` (µs, default now). It writes the branch row and `to`'s version row, and nothing else.

Rules:
- `at` may be in the past, which forks a historical state, but not in the future.
- `to` must be unused: no version row, no memory rows and no active branch row. A discarded tombstone may be reused.
- `from` may be any namespace, including an empty one or another branch, up to depth 16.
- The fork runs in one transaction that asserts `to` is unused, so two forks into the same name cannot both succeed.

Returns `{ns: to, parent: from, at, depth, lineage: [[ns, cut], …]}`.

### rewind {ns, to, at?}

Puts the namespace's view back to its state at `to`. It is **non-destructive and undoable**. At `at` (default now), in one transaction under the namespace version:
- every memory current now but not valid at `to` gets a retraction row;
- every memory valid at `to` but not current now is re-asserted with the values of its version valid at `to` (text, kind, source, created, importance), with `current = true` on its vector.

Inherited memories are copied up first, so rewinding a branch never touches its ancestors.

Rewind restores **validity**, which decides what is current. It does not restore importance, expiry or the archive tier, which are not versioned.

History keeps everything, so the state just before the rewind is still readable at `at − 1`. **Undo** is `rewind {ns, to: <result.undo>}`.

It returns `{ns, to, at, retracted: [ids], restored: [ids], undo}`. With an Engram model attached, it enqueues one `retract` job for the retracted ids and one `write` job per restored memory.

### merge {from, into, dry_run?, keep_time?, forgets?}

Re-applies the branch's own statements to `into` through the normal `remember` path. The decider settles conflicts there as for any new statement: supersede, duplicate, pending or add.

**Statements** are the branch's own `kleos_mem_source` rows: statements stored in the branch itself. Source rows that arrived by copy-up are excluded (`kleos_mem_copied`). Statements are replayed in valid-time order (`at`, then statement id) with the options they were stored with:
- `kind`, `importance`, `source` and `role` come from the memory rows;
- entity names come from the branch's mentions, by canonical name;
- `split` is `none` when the statement is one memory under the statement id, else `auto`;
- the remaining TTL is carried over.

`keep_time` (default true) replays each statement at its original `at`. A statement older than a memory it would supersede in `into` is then a late arrival, left pending by the usual rule; `keep_time: false` replays at now.

**Forgets.** The branch's explicit forgets are replayed too, unless `forgets: false`. These are retractions of inherited memories that no `supersedes` link explains, from `forget` or `rewind`. The same id is retracted in `into` if it is current there; otherwise it is reported under `forgets_skipped`. Supersessions are not replayed as such: replaying the superseding statement decides them again in `into`.

**Idempotent.** Each replayed statement gets a `kleos_mem_merge` row `(into, stmt)`, and later merges skip those statements. So a merge interrupted by a crash completes when run again, and a branch can be merged repeatedly as work continues.

A merge is a sequence of `remember` calls, not one transaction. Each statement commits on its own under `into`'s namespace version.

**dry_run: true** gives an exact preview. It forks `into` into a scratch branch (O(1)), runs the real merge into the scratch branch, computes `diff {a: into, b: scratch}`, then discards the scratch branch. The models are deterministic, so the preview equals what a real merge right after would do, unless `into` changes in between.

Returns:
- `{from, into, dry_run, statements, skipped}`;
- `replayed`: one entry per statement, `{stmt, text, at, action, ids, superseded, duplicate_of, tentative}`;
- `forgets` and `forgets_skipped`;
- with `dry_run`, also `diff`.

### diff {a, b, at?}

Compares the views of two namespaces at `at` (default now). It reports what changes going from `a` to `b`:

| Field | Meaning |
|---|---|
| `added` | memories current in `b` and not in `a` |
| `removed` | memories current in `a` and not in `b`, each with `reason`: `superseded` (`b` has a current memory with a `supersedes` link to it; `by` names it) or `forgotten` |
| `same` | count of memories current in both |
| `pending` | `b`'s pending `supersedes` decisions about memories current in both |
| `entities.added`, `entities.removed` | canonical entity names present in only one view |

Memories are matched by id first. Ids that remain unmatched are then matched by normalised text, so a statement merged into `a` under a new id counts as the same memory as its original in `b`. Each memory is `{id, text, kind, valid_from}`. Works for any two namespaces, branches or not.

### discard {ns, cascade?}

Drops a branch. Only branches can be discarded; for a plain namespace use `forget {all: true, mode: 'erase'}`.

It refuses when the branch has active child branches, unless `cascade: true`, which discards them first, deepest first. In one transaction under the branch's version, it:
- deletes every row the branch owns in every memory relation;
- deletes the merge-log rows with `into = ns`;
- for entities the branch created, deletes their merges and review links;
- deletes the branch's Engram queue jobs and overlay rows;
- bumps the version and marks the branch row `discarded`.

Then it runs `::erase kleos_mem` for the branch's memory keys, which removes their search-index entries.

Ancestors are never touched. A `remember` planned before the discard fails its version check, re-plans, and is refused because the branch is discarded.

Returns `{ns, discarded: [names], memories}`.

### branches {ns?}

Lists branch rows: `{ns, parent, at, created, status, depth, own}`, where `own` is the number of memories the branch owns. With `ns`, it lists that namespace's lineage and its descendants.

## How each relation behaves under a fork

| Relation | Read in a branch | Written by a branch |
|---|---|---|
| `kleos_mem` | the view: owner's rows, ancestors at `min(T, cut)` | own rows; inherited memories are copied up first |
| `kleos_mem_vec` | owner's row. In the HNSW leg, level 0 rows filter on `current`; ancestor rows are filtered after the search by the view's validity | own rows (copy-up recomputes `current`) |
| `kleos_mem_entity` | union of levels, ancestors' rows with `created ≤ cut` | new entities only |
| `kleos_mem_mention` | owner's rows | rows for the branch's own memories (and copies) |
| `kleos_mem_link` | union of levels, ancestors' with `created ≤ cut` | own rows (a link may point at an inherited memory) |
| `kleos_mem_source` | owner's rows | own rows; a duplicate of an inherited memory copies it up first |
| `kleos_mem_decision` | union of levels, ancestors' with `at ≤ cut`; a pending decision counts only while its new memory is in the view | own rows (they may name inherited candidates) |
| `kleos_mem_event`, `kleos_mem_role` | owner's rows | own rows and copies |
| `kleos_mem_life` | owner's row | own rows and copies |
| `kleos_mem_access` | owner's row | own memories only: accesses of inherited memories are not counted (access weight is 0 by default) |
| `kleos_mem_ns` | own version only | own version only |
| `kleos_entity_merge`, `kleos_entity_link` | as before: global, keyed by entity id | rows for the branch's own entities |
| `kleos_engram_queue` | — | own jobs (`qid` from the branch's version) |
| `kleos_engram_facts`, `kleos_engram_rows` | the lineage overlay (see "Engram") | own rows; inherited facts and rows are copied up first |

## Recall and the context output

`recall` on a branch runs one hybrid search over the whole lineage, so the fusion normalises every leg over one list. (`bm25` normalisation divides by the list's maximum, so separate per-level searches would not be comparable.)
- **Vector leg:** `filter: (ns == $ns && current) || is_in(ns, $ancestors)`, or `is_in(ns, $lineage)` with `as_of`. The fetch is multiplied by the depth, because ancestor rows are filtered only after the search.
- **Text leg:** `filter: is_in(ns, $lineage)`.
- **View rule:** one rule per level joins the fused ids with `kleos_mem @ min(T, cut)`. An ancestor level adds `not *kleos_mem{ns: <nearer level>, id}` for each nearer level, which masks copied-up memories.
- **Everything after the search** (blend, pending flags, roles, access, expiry, event times, diversity, entity expansion and the context output with its "previously" chains) reads per owner. Each read runs once per level that owns some of the hits. A plain namespace runs exactly the queries it ran before.

## Concurrency

- **Branch writes** (`remember`, `forget`, `set`, `rewind`) commit under the branch's own `kleos_mem_ns` version, so they never conflict with writes to the parent. Copy-ups are part of the same transaction.
- **Branch plans read ancestors at their cuts**, which are in the past. A parent commit after the fork does not change the branch's view, so it cannot invalidate a branch plan. One exception: a parent write with an explicit `at` at or before the cut, committed after the fork, is visible in the branch, as if it had committed before the fork. Default `at` is now, so this needs a deliberate backdated write.
- **Fork** asserts in its transaction that `to` is unused.
- **Merge** is `remember`s on `into`, under `into`'s version, so it composes with concurrent writers of `into` like any other `remember`.
- **Discard** bumps the branch's version, so in-flight writers of the branch fail their check and are then refused.

## Engram (feature `engram`): kept, and extended to branches

Engram overlays are per namespace (`kleos_engram_facts` and `kleos_engram_rows` are keyed by `ns`). Branches give them the same copy-on-write semantics as memories:
- **Reads.** `::engram materialize` / `export` of a branch (or of a stack containing one) expands it to its lineage. Each level is read at `min(T, cut)`, and for each `(layer, row)` and each fact the nearest level with any version owns it. A branch can thus serve its parent's overlay with its own edits on top, for free.
- **Writes.** The worker applies a branch's jobs in the branch's namespace. Its view of current facts is the lineage view, so a `supersede` of an inherited memory retracts the inherited fact in the branch and re-solves the rows it touched. Before writing, an inherited fact or row is copied up (its versions ≤ cut), the same rule as memories, so `as_of` reads of the branch overlay stay right.
- **Merge** replays statements through `remember` on `into`, which enqueues `into`'s Engram jobs as usual. The overlay of `into` is computed from its own memories, never copied from the branch.
- **Rewind** enqueues `retract` and `write` jobs, as `forget` and `remember` do.
- **Discard** deletes the branch's queue jobs and overlay rows (`engram_erase_namespace`). Ancestors keep theirs.
- **Erase** of a memory (GDPR, below) enqueues the overlay erase in every namespace of the family that has the memory.

## Erase (GDPR) in a family of branches

`forget {mode: 'erase'}` must leave no copy. Memory ids are shared within a family (root, branches, copies), so erasing id `X` from any namespace of a family erases it from **every** namespace of the family that has rows for `X`: the owner and every copy, in ancestors and descendants alike. The result lists the namespaces touched. Retract and archive stay local to the branch, by copy-up.

## Costs

- **fork:** one transaction, three small rows; independent of the parent's size.
- **Branch reads:** one extra point lookup per level to build the lineage, a search leg over a few more rows (masked ancestor rows), and one per-owner read per level for the hit metadata. The spec target is under 2× a plain recall at depth 1; see "Results".
- **Branch writes:** a copy-up per inherited memory touched, a few rows each.
- **Storage:** what the branch adds or changes; nothing for what it only reads.

## Edit history tied to conversations (design only)

DeltaDB's core idea: every edit to a document (code or text) is a small delta with an author, human or agent, linked to the conversation that produced it. A file can be rebuilt at any version, and blame leads from a line to the conversation that wrote it. For kleos this is wave 4's code memory, and it reuses the memory tables for the "why" side.

### Relations

| Relation | Columns | Notes |
|---|---|---|
| `kleos_doc` | `{ns, doc => path?, kind, created, head: [String]}` | One document (a file, a note). `head` is the current frontier, the delta ids with no successors. |
| `kleos_delta` | `{ns, doc, id => parents: [String], author, author_kind, thread?, msg?, mem?, at, ops: Bytes}` | One edit batch. `author_kind` is `human` or `agent`; `thread`/`msg` link to the conversation; `mem` links to the memories it produced or acted on; `ops` is the CRDT update (see below). |
| `kleos_doc_span` | `{ns, doc, version, start => end, delta}` | Blame index per checkpoint version: character ranges `[start, end)` with the delta that inserted them. Rebuilt incrementally on write. |
| `kleos_doc_snap` | `{ns, doc, version => text, frontier}` | Periodic snapshots (every N deltas), so a version is rebuilt from the nearest snapshot. |

### Operations

- `::doc edit {ns, doc, ops | text, author, author_kind, thread?, msg?, mem?}`: appends a delta. A full-text `text` is diffed into ops.
- `::doc at {ns, doc, version | at}`: the text at a frontier or a time.
- `::doc blame {ns, doc, version?, lines?}`: per line range, the delta, author, thread and linked memories, which answers "why does this function exist?".
- `::doc fork {ns, doc, from_version}` and `::doc rewind {…}`: the document counterparts of memory fork and rewind. Rewind appends an inverse delta and stays undoable. A conversation thread's rewind restores both the memory namespace (memory rewind) and the documents (doc rewind) to one time.
- Linking to memory: a delta's `mem` ids and `thread` let `recall` answer "what did we decide about X" with the code it produced, and blame answer the reverse.

### CRDT engine

Concurrent edits from several agents need a merge that converges. Three Rust candidates (crates.io, 2026-10-24):

| Crate | Licence | Latest | Fit |
|---|---|---|---|
| `loro` | MIT | 1.17.1 (2026-10-22) | Text, list, map, tree and movable list. Version vectors, `checkout` to any frontier (time travel) and shallow snapshots. Actively maintained, large API. |
| `yrs` | MIT | 0.28.0 (2026-09-17) | Y.js-compatible YATA CRDT. Mature and widely deployed. Interop with browser editors (Y.js) for free. Time travel needs kept snapshots. |
| `diamond-types` | ISC | 1.0.0 (2022-08-25) | Fastest plain-text CRDT, and DeltaDB's design ancestor. The crates.io release is old; the newer work is unreleased. |

`automerge` (MIT, 0.12.0) is a fourth option, document-shaped rather than text-first.

**Recommendation: `loro`.** Built-in checkout to any frontier makes `::doc at` and `rewind` cheap, and its op ids map one to one onto `kleos_delta.id`. The licence is MIT and it is maintained. `yrs` is the fallback if Y.js editor interop becomes the priority.

Either way, kleos stores the op log as rows, so history, blame and links stay queryable in Datalog. The CRDT library is the merge and replay engine, not the store. Live sync between machines, DeltaDB's replication, is out of scope: kleos is embedded, and sync can come later over the existing live subscriptions.

## Caveats

- **Backdated parent writes.** A write to the parent with an explicit `at` at or before the cut, committed after the fork, shows in the branch (see "Concurrency").
- **Entity merges are global.** A merge decided in the parent after the fork is visible to the branch's entity resolution. Merge rows are keyed by entity id, not namespace, and are read as of now. Recall's views read them per level at the cut, so only resolution of new names sees the later merge.
- **Metadata is not versioned.** Importance, expiry and the archive tier are copied as they are when a memory is copied up, not as they were at the cut.
- **Accesses of inherited memories** in a branch are not counted.
- **Merge replays statements, not decisions.** The outcome in `into` is decided again against `into`'s memories, by design: the decider settles conflicts against what `into` holds now. Pending decisions of the branch are not carried over as such; the replay produces its own.
