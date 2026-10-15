# Spec — Supersession and provenance for agent memory (E1, E2)

_Status: **implemented on `wave1/mem`, 2026-10-15** (UPGRADE_PLAN.md items E1 and E2).
Code: `kleos-core/src/runtime/memory.rs` (engine), `kleos-core/src/parse/memory.rs`
(parsing), one hook in `query/stored.rs` (`execute_relation`). Tests:
`kleos-core/tests/memory_supersession.rs`. Builds on valid time (`Validity`), transaction
time (`TxTime`, [`bitemporality.md`](bitemporality.md)) and the search-index-safe delete
path that bulk import uses (`write_import_tuple` in `runtime/db.rs`)._

> **One-line goal.** A new fact ends the old fact's validity and never deletes it;
> every derived fact can point at its sources; retracting a source retracts what was
> derived only from it, erasing a source (GDPR) deletes it — vectors and text postings
> included — and the engine can explain where any fact came from.

---

## 1. Why

Agent-memory systems (Zep/Graphiti bi-temporal edges, Mem0's add-only store) keep
contradicted facts as history instead of overwriting them, and keep a pointer from each
fact to the episode it was extracted from. Graphiti does both in application code. kleos
already stores valid time and transaction time; what was missing is (a) the rule that a
new value for the same subject/predicate *ends* the old one, applied atomically on write,
and (b) engine-maintained source links with cascading retraction and erasure.

## 2. Supersession (E1)

### 2.1 Declaration

```
:create fact {subject: String, predicate: String, object: String,
              vt: Validity, tt: TxTime => confidence: Float default 1.0}
::supersede create fact {subject, predicate}
::supersede list            # relation, key_cols
::supersede drop fact
```

The **supersession key** is a leading prefix of the relation's non-temporal key columns
(here `subject, predicate`). Rows sharing it form a *group*; at any valid time at most
one row of a group is current. The remaining key columns (`object`) are the group's
*value key*. The relation must have a valid-time axis: `vt: Validity` last, or
`vt: Validity, tt: TxTime` (bitemporal). With TxTime, "known at T" queries work and
corrections are recorded; without it only valid time is kept.

The key may also cover all non-temporal key columns (`{user, vt: Validity => city}` with
`::supersede create city {user}`). Then the vt axis already supersedes by itself, and the
declaration adds idempotent re-assertion.

Why a prefix: a group must be one contiguous key range (one prefix seek, the memcmp
layout), and the declaration then reads like the fact's identity.

### 2.2 Write semantics

Every `:put` / `:insert` into the relation is rewritten before the ordinary put, in the
same statement and transaction. For each group, incoming rows are replayed in ascending
valid time against the group's current-belief timeline (stored rows, plus TxTime rows
buffered earlier in the same transaction). For an assertion of value key `x` at valid
time `t`:

1. **Idempotence.** If `x` is already current at `t` with identical value columns, the row
   is dropped: no new version, no new `::history` entry. (`:insert` keeps the row so its
   existence check still fails as documented.)
2. **Supersession.** Every other value key current at `t` gets a retraction row at `t`
   (its values copied). Nothing is deleted.
3. **Late arrival.** If the group has a later assertion of another value at `t' > t`, and
   `x` would still be current just before `t'`, `x` gets a retraction at `t'`. An old
   fact arriving late fills its gap and never ends a newer fact.

Retraction rows (`[ts, false]`) pass through unchanged and supersede nothing (a
retraction does not resurrect an older value, matching `@`'s no-fall-through rule).

**Same-instant conflicts.** If another value was asserted at exactly `t`:

| Case | Result |
|---|---|
| Bitemporal relation, earlier transaction | Correction: the old value is retracted at `t` with a newer tt. Current belief shows the new value; `@ (tt: before)` still shows the old one (typo fix). |
| Same transaction or same batch | Error `eval::supersede_same_instant`: two values for one key at one instant. |
| Valid-time-only relation | Error `eval::supersede_same_instant`: without TxTime the stored assertion shadows any retraction at the same instant. Help text: add `tt: TxTime`, or `:rm` the wrong row first. |

Not rewritten: `:update`, `:rm`, `:delete`, `:reconcile`, `:create`/`:replace` (a replaced
relation gets a new id, so its declaration lapses). Existing data is not reconciled when
the declaration is made; the rule applies to later writes.

### 2.3 Reading

| Question | Query |
|---|---|
| True at valid time T | `*fact{subject: 'alice', predicate: 'works_at', object: o @ T}` |
| Known at transaction time TT (about valid time T) | `*fact{… @ (vt: T, tt: TT)}`, or `:as_of TT` for the whole query |
| Full record of one fact | `::history fact [['alice', 'works_at', 'Acme']]` |
| What changed between T1 and T2 | `::changes fact {from: T1, to: T2}` |
| What was learned between TT1 and TT2 | `?[o] := *fact{… object: o @ (vt: T, tt: TT2)}, not *fact{… object: o @ (vt: T, tt: TT1)}` |

`::changes rel {from: T1, to: T2, as_of: TT, prefix: [..]}` walks each key's resolved
timeline (current belief, or the belief at `as_of` on bitemporal relations) and returns
one row per transition in `(T1, T2]`, ordered by time:

- `added`: not current → current
- `ended`: current → not current (values shown are those of the ended version)
- `changed`: current → current with different value columns (the `{user, vt => city}`
  shape)

Output columns: the plain key columns, `change`, `at` (µs), then the value columns.
Redundant re-assertions and retractions of rows that never were current (a corrected typo)
produce no row. `to` defaults to now; `prefix` narrows the scan to a key prefix (one
subject); `as_of` accepts `'NOW'` (current belief) and is rejected on relations without
TxTime or below a `::history_gc` floor.

### 2.4 Bulk paths

| Path | Behaviour |
|---|---|
| `import_relations`, `import_columnar_file` | **Rejected** for relations with a supersession key. They bypass the put path, so they would skip supersession silently. Write with `:put` (rows as a parameter), or `::supersede drop`, import, re-declare. |
| `import_from_backup`, `restore_backup` | Copy stored rows verbatim. Retractions written earlier are rows, so history survives intact; supersession is not re-run. |

## 3. Provenance (E2)

### 3.1 Declaration

A provenance link is an ordinary relation, so links are queryable with Datalog and can
carry their own value columns:

```
:create episode  {id: String, vt: Validity => text: String}
:create fact_src {subject: String, predicate: String, object: String, episode: String}
::provenance create fact_src {derived: fact, source: episode}
::provenance list            # link, derived, source
::provenance drop fact_src
```

Rules, checked at declaration:

- The link relation has no temporal axis, and its key is exactly the derived relation's
  non-temporal key columns followed by the source relation's (types must match; names are
  free).
- The derived relation has a temporal axis (vt, tt or both), so that retracting its
  sources can end its validity. Sources may be any relation, including plain ones.
- Several links may feed one derived relation (facts from episodes *and* documents), and
  links chain (summaries derived from facts derived from episodes). Cycles are allowed.

Recording provenance is an ordinary `:put` into the link relation, usually in the same
transaction as the fact. For large link relations add an index that leads with the source
columns (`::index create fact_src:by_ep {episode}`): cascades use it to find derived rows
by source with a prefix seek instead of a scan.

### 3.2 Soft retraction: `::retract`

```
::retract episode [['ep2']]              # at the transaction's NOW
::retract episode [['ep2']] @ '2026-03-01T00:00:00Z'
```

1. The source key's current row gets a retraction at the given valid time (TxTime-only
   relations: `:rm`, which appends a retraction at commit tt). Plain relations cannot be
   retracted: error, use `::erase`.
2. For every link naming that source, each derived row is checked: if none of its sources
   (across all links into the derived relation) is still current at that time, it is
   retracted the same way, and the check repeats on whatever was derived from it.
3. Rows with another live source stay current. Nothing is deleted; history, `@` and
   `::history` keep answering for earlier times.

Output: `depth, relation, key, action` with `retracted`, or `not_current` for a root key
that was not current. Derived rows that were already not current are not listed.

### 3.3 Hard erasure (GDPR): `::erase`

```
::erase episode [['ep7']]
```

1. Every row of the source key (all valid times, all transaction times) is physically
   deleted, along with its B-tree index entries and its HNSW, FTS and LSH entries. This
   uses `write_import_tuple`, the delete path that keeps search indexes consistent.
2. The link rows naming the erased key are deleted.
3. Each derived row that is left with **no** links at all is erased the same way, and the
   cascade continues. One that keeps links, but none to a live source, is soft-retracted.
   One that keeps a live source stays and is reported `unlinked`.
4. Each erased key gets an audit row in `kleos_evict_audit` (the `::evict` trail):
   relation, salted key hash, row count, and one burned transaction time. The key itself
   is not stored.

Output: `depth, relation, key, action` with `erased`, `retracted`, `unlinked` or
`absent`. `::erase` needs `normal` access on every erased relation, and refuses a TxTime
relation with writes still buffered in the same transaction (the same rule as `::evict`).

### 3.4 Lineage: `::lineage`

```
::lineage summary [['s1']]
::lineage fact [['alice', 'works_at', 'Globex']] @ '2026-02-01T00:00:00Z'
```

Breadth-first walk up the links. Columns: `depth`, `relation`, `key`, `via` (the link
relation; null for the root) and `current` (whether that row is current at the given
valid time, default now). For the downward question ("what came from this episode?"),
query the link relations directly, recursively if needed.

## 4. Storage

Declarations live in two reserved relations created on first use, in the style of
`kleos_stored_queries`:

- `kleos_supersede {rel_id: Int => relation: String, key_cols: [String]}`
- `kleos_provenance {link_id: Int => link: String, derived_id: Int, source_id: Int}`

They are keyed by relation id, so `::rename` keeps a declaration, and a dropped relation's
declaration lapses (ids are never reused). No `RelationHandle` field was added: the
catalog encoding stays positional-compatible, and stores without these relations behave
exactly as before. The put hook costs one existence probe per `:put`/`:insert` statement
while no declaration exists, and one point read when the catalog exists.

## 5. Semantics decisions and why

| Decision | Why |
|---|---|
| Declarative key on the relation, applied inside `:put` | Writers (LLM extractors, ingestion code) cannot forget it. Retractions land in the same transaction, so readers never see two current values. |
| Supersede by appending retractions, never by deleting | History is the product. `@`, `::history` and `::changes` need the ended row. |
| Replay a batch in valid-time order and handle late arrivals | Extraction is out of order. Without the late-arrival rule an old fact either ends the newer fact or stays current forever. |
| Drop identical re-assertions | Agents restate facts constantly; each restatement would otherwise add a version (and a tt row). Provenance from the new source is still recorded in the link relation. |
| Same-instant conflicts are an error without TxTime | A vt-only relation cannot express the correction (assert shadows retract), so silently writing a no-op retraction would leave two current values. |
| Links are user relations with a positional key contract | Queryable with Datalog, indexable, and extensible with value columns (confidence, extractor) at no engine cost. |
| A derived row survives while any source is live | Matches "derived only from it": shared facts are not lost when one of several sources goes. |
| Erase deletes derived rows only when no link remains | GDPR targets the erased material; a fact independently stated elsewhere is not that material. Its link to the erased source is removed, so lineage no longer mentions it. |
| Cascades run as system ops, not on every `:rm` of a source | Explicit and predictable. Cost is paid only when asked for. Plain `:rm`/`:put` retractions of a source do not cascade. |

## 6. Caveats

- Cascade writes do not fire change callbacks (the system-op path has no callback
  dispatch, as with `::evict`). Put triggers on derived relations do fire.
- `::retract` on a bitemporal relation buffers its writes like any TxTime write: later
  statements in the same script do not see them (one belief event per transaction).
- Without a source-leading index on the link relation, each cascade step scans the link
  relation.
- Search indexes (HNSW/FTS/LSH) still index every version row of a valid-time relation,
  retractions included. Filter search hits through an `@` join to keep only current facts.
  TxTime relations still cannot carry search indexes (bitemporality.md §8), so a bitemporal
  fact store keeps its embeddings in a valid-time sibling relation, linked through
  provenance as in the tests.
- In imperative scripts the cascade locks only the named relation and the catalog up front.
