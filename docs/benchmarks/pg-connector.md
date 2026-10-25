# kleos-pg benchmark (Postgres coexistence connector)

*2026-10-25, branch `wave7/pg`. **Partial results.** The full harness
(`kleos-pg/examples/pg_bench.rs`) was stopped before every phase finished,
on request; the numbers below are the phases that completed, each labelled
with the run and code version it came from. Nothing here is extrapolated.*

## Setup

- Postgres 17.11 (theseus-rs macOS arm64 binaries), local, `wal_level =
  logical`, `shared_buffers = 128MB`.
- kleos store on an external SSD (`/Volumes/SSD01`) whose `fsync` takes
  16 ms at p50 (43 ms max) under the concurrent builds of two other agents;
  the internal disk measures ~0 ms. Load average 4–20 during the runs. Treat
  absolute numbers as a lower bound for a server with NVMe storage.
- Schema (CRM/orders, generated in Postgres with `generate_series`):
  `accounts` (10%, long `notes`), `contacts` (25%, FK → accounts), `products`
  (200), `orders` (25%, FKs → accounts, contacts), `order_items` (25%, a join
  table: two FKs as its key → edges), `tickets` (rest, FK → accounts, long
  `body`). 100,000 rows in total.
- Release build, `cargo run --release -p kleos-pg --features "query bench-rocksdb" --example pg_bench -- --rows 100000 --engine <sqlite|rocksdb>`.
- Code: branch `wave7/pg` at `152730b` plus the working tree that became the
  next commit (group commit in the stream; row-memory headers without
  repeated words). Which run used which is noted per table.

## Snapshot and streaming throughput

| Engine | Run | Snapshot (100k rows) | Streaming (2,000 tx × 5 changes) | Notes |
|---|---|---|---|---|
| RocksDB (default, WAL not fsynced per commit) | B (group commit) | **35.1 s = 2,852 rows/s** | **13.3 s = 754 changes/s, 151 tx/s** (Postgres writer: 2.7 s) | 384,800 fact rows and 99,800 memories written by the snapshot |
| SQLite (kleos default, `synchronous=FULL`, rollback journal) | A (before group commit) | 398.4 s = 251 rows/s | 107.5 s = 93 changes/s, 19 tx/s (writer 5.2 s) | same rows; dominated by fsync |

Where the time goes (macOS `sample` of the writer thread, SQLite, run A):
73% of the writer's samples in `sqlite3BtreeCommitPhaseOne → fsync` and
another 16% in page-cache spills (`pagerStress → syncJournal`) during
`put_in_fts`. The connector's own work (decoding, diffing, building the
write script) is a small share. kleos's SQLite storage opens its database
with the default 2 MB page cache and rollback journal; WAL mode and a larger
cache (a kleos-core storage option, not part of this branch) are the lever
for SQLite. Run B's group commit lets a busy writer take bigger batches
(the stream keeps reading into the open batch until it is full), which is
what lifted streaming from one small batch per commit.

## End-to-end latency

PG `COMMIT` returned → the row's memory committed in kleos (a change
callback on `kleos_mem`; its FTS entry is written in the same transaction,
so it is recallable then). 200 single-row updates at ~20/s.

| Engine | Run | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| RocksDB | B | **62.6 ms** | **86.8 ms** | **95.6 ms** | 108.9 ms |
| SQLite | A | 323.7 ms | 3,822 ms | 4,628 ms | 4,736 ms |

Most of the RocksDB p50 is the batch window (`max_ms` 20 ms in the
benchmark config) plus one kleos commit.

## Recall over ingested rows

Run A (SQLite, before the header change), FTS only (no embedder in the
build), after three plan changes on each of 200 sampled accounts, question
"What is the current plan of Account N?":

| | |
|---|---|
| current value in top 5 | **200 / 200** |
| stale value (an older plan) at rank 1 | **0 / 200** |
| current value at rank 1 | 0 / 200 |
| recall p50 / p95 | 1,380 ms / 1,950 ms |

- Superseded row memories never came back: an update retracts the old
  version, so only the current plan is recallable, and the context block
  shows the old one as `previously`.
- Rank 1 was another row sharing the number (`Product Product 37 (products
  37): sku: SKU-37; price: 37.99` for Account 37): the old memory text
  repeated the class, label and key, and BM25 rewarded the repeated number.
  The next commit writes each word once (`Account 37: industry: …; plan: …`,
  `Customer Acme Corp (#42): …`); its effect on rank 1 was not measured
  before the run was stopped. With an embedder (hybrid recall), not built
  for this benchmark, ranking would also be semantic.
- **Latency finding (kleos-core):** the profile of a recall in run B shows
  nearly all of its time in `graph_recall` (`graph: 'ppr'`, the default)
  scanning every `kleos_onto_fact` row of the namespace (`onto_graph`
  reads all current literal-free facts, filtering after the scan): 384,800
  source fact rows make every recall in that namespace about 1.4 s. The
  connector already keeps its edges out of that graph (they carry a
  literal), but the scan still reads them. Mitigations, not done here:
  `recall {graph: 'off'}` for source-heavy namespaces; writing the source's
  facts and nodes to their own namespace (`pg:<source>`) while memories stay
  in the agent's; or an index or cache in `onto_graph`. See the spec's
  "Left".

## Write-back exactly once under injected crashes

Not reached in the stopped runs. Covered by the integration test
`writeback_crash_exactly_once` (Postgres 16, 17 and 18): a worker process
aborts right after the Postgres commit, before the flow records the step; a
second worker retries the step after the lease, finds the ledger row,
returns the stored result (`replayed: true`) and the table holds exactly
one row. The allow-list rolls back a CTE that writes elsewhere. The
harness's phase 5 (40 workflows, half with a crashed worker, before or after
the Postgres commit, then a count of rows, duplicates and ledger entries)
is in `pg_bench.rs` for a later run.

## Live queries

Not reached in the stopped runs. The integration test
`live_query_joins_postgres_and_kleos` checks a filtered `LIMIT` query whose
filter runs in Postgres (it appears in the federated plan) and a join of a
Postgres table with `kleos.facts`.

## Reproduce

```bash
KLEOS_PG_TEST_URL=postgres://postgres@127.0.0.1:54317/postgres \
  cargo run --release -p kleos-pg --features "query bench-rocksdb" --example pg_bench -- \
  --rows 100000 --engine rocksdb --dir <scratch dir> --stream-txs 2000
```

It prints a JSON report (snapshot, streaming, e2e latency, recall with
`graph` on and off, live queries, write-back) and phase lines on stderr.
