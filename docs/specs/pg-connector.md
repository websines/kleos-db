# Postgres coexistence connector (`kleos-pg`)

*2026-10-25, branch `wave7/pg`. Status: built, with integration tests
against Postgres 16, 17 and 18 and a benchmark ([pg-connector.md](../benchmarks/pg-connector.md)).
Research basis: `research/pg-cdc-research.md` (outside the repo).*

Code:
- `kleos-pg/src/source.rs`: the connector (slot, snapshot, pgoutput stream, acknowledgements, resync);
- `kleos-pg/src/sink.rs`: the kleos writer (facts, nodes, memories, row images, checkpoint in one transaction);
- `kleos-pg/src/mapping.rs`, `catalog.rs`: inference from the catalog, overrides, the ontology `pg:<source>`;
- `kleos-pg/src/writeback.rs`: the `pg` flow action and its ledger;
- `kleos-pg/src/debezium.rs`: Debezium change events;
- `kleos-pg/src/query.rs`: live queries (feature `query`, DataFusion);
- `kleos-pg/src/ops.rs`, `admin.rs`, `store.rs`: status, init, resync, slots, publications, the connector's relations;
- `kleos-core/src/flow.rs`: `register_action` (engine actions provided by the embedding application);
- `kleos-bin/src/pg.rs`, `server.rs`, `cli.rs`, `mcp.rs`, `backend.rs`: the surfaces;
- `sdk/ts` (`Kleos.pg`), `kleos-lib-python/integrations/kleos-pg` (`PgClient`).

Tests: `kleos-pg/tests/pg.rs` (gated on `KLEOS_PG_TEST_URL`; the Debezium
test needs no server), unit tests in each module. Guide:
[docs/guides/postgres.md](../guides/postgres.md). Skill:
`skills/kleos-postgres`.

## Goal

Companies keep Postgres as their system of record. kleos runs next to it as
the agents' memory: every row an agent may need is an entity with typed facts
over valid time, connected by its foreign keys, findable by `recall`, and
kept current by change data capture. Agents write back to Postgres as
durable workflow steps, exactly once in effect, and the rows they wrote come
back through the stream linked to the step that wrote them.

Nobody maintains a mapping: tables, keys, foreign keys, comments and types
define the ontology `pg:<source>`, which evolves with the schema. A YAML file
can override what inference decides.

## Decision: our own pipeline on etl's foundation, not etl's orchestration

The research recommended embedding supabase/etl. kleos-pg uses the same
foundation, the Supabase fork of rust-postgres at the revision etl pins
(`8086f1fd`: the replication protocol and the pgoutput decoder, with
Materialize's patches), and runs its own slot, snapshot and stream logic
(about 900 lines in `source.rs`). etl's orchestration does not fit four of
the requirements:

1. **Configurable slot names.** etl derives every slot from constants
   (`supabase_etl_apply_<pipeline>`, `supabase_etl_table_sync_<pipeline>_<oid>`);
   only the pipeline id is configurable, so it cannot be wrapped. Slot names
   are cluster-wide, and a DBA has to be able to name them.
2. **Atomic source transactions.** etl copies each table with its own sync
   slot and catches it up from that slot before handing it to the apply
   worker. During an initial sync, one transaction touching two tables is
   delivered in two pieces by two workers. kleos-pg copies every table at
   one exported snapshot, so cross-table transactions apply whole from the
   first one on.
3. **One watermark for exactly once.** With a single stream, the kleos
   checkpoint is one commit LSN, committed with the facts. etl's
   at-least-once destination contract (parallel calls, per-table sync
   streams, `Accepted`/`Durable` statuses) would need a watermark per worker.
4. **Nothing installed in the system of record.** etl's schema-change
   handling relies on source migrations: an `etl` schema and a
   `ddl_command_end` event trigger, which needs superuser and is not
   available on most managed services. kleos-pg detects DDL from pgoutput's
   Relation messages and the catalog, and writes nothing into the source
   except its publication, its slot and (with write-back) the ledger table.

etl's other strengths are parallel copies of one big table by ctid ranges,
a memory governor, and failover-slot helpers. kleos-pg covers failover slots
(`FAILOVER true` on PG17+), and bounds memory through its batch queue and
transaction parts (below). It copies a table on one connection, which is
enough for kleos's write speed. Everything etl-specific stays behind
`change::Tx`, so moving the source to etl later (once slot names are
configurable there) replaces `source.rs` and nothing else.

The fallback named in the research, pg_walstream (BSD-3, crates.io), was not
needed: it would replace the decoder kleos-pg already has from the fork,
which Materialize and Supabase run in production.

## Crate layout and features

- `kleos-pg` (new, `publish = false`): the git dependency cannot go to
  crates.io, so it stays out of `kleos-core` (published) and out of the
  default workspace members. Features: `tls` (default, native-tls) and
  `query` (DataFusion live queries; large).
- `kleos-core`: one generic hook, `flow::register_action(name, action)`,
  so an embedding application can add engine actions (no feature flag, no
  new dependency).
- `kleos-bin` features: `pg-cdc` (connector, CLI, HTTP, MCP `pg_status`),
  `flow-pg` (the `pg` flow action; implies `pg-cdc`), `pg-query` (live
  queries; implies `pg-cdc`).
- Toolchain: the fork needs Rust ≥ 1.75; DataFusion 55 builds on 1.99 (the
  kleos `infer` toolchain). arrow 59.2 matches kleos-core's pin.

## Configuration

```yaml
source: crm                        # entity keys pg:crm/…, ontology pg:crm
connection: postgres://kleos:${PGPASSWORD}@db:5432/app?sslmode=require
publication: kleos_crm             # default kleos_<source>
slot: kleos_crm                    # default kleos_<source>; cluster-wide
create_publication: true           # else `pg init` prints the DDL for a DBA
publication_mode: tables           # tables | schema (PG15+) | all
failover: auto                     # PG17+ failover slot
include: ["public.*"]
exclude: ["public.audit_*"]
ns: default                        # target namespace (scope)
workspace: default                 # tenancy: target workspace
actor: {service: connector:pg/crm} # the service principal it writes as
snapshot: true
batch: {max_changes: 5000, max_ms: 100, max_tx_changes: 50000, snapshot_rows: 2000, queue: 4}
memory: {rows: auto, text_min_chars: 80, embed: auto, entities: false}
ontology: {activate: true, review: false}
lag_alert: {bytes: 1073741824, seconds: 300}
new_tables_secs: 60                # pick up new tables matching include
heartbeat_secs: 0                  # pg_logical_emit_message heartbeats
status_secs: 10                    # acknowledgement interval
tables:
  public.customers:
    class: Client
    key: [id]
    label: "{first_name} {last_name}"
    memory: {row: true, text: [notes], template: "Client {name}: {plan}"}
    columns: {ssn: drop, email: pii, status: category, owner_id: {references: public.users}}
  public.customer_tags:
    edge: {source: customer_id -> public.customers, destination: tag_id -> public.tags, label: tagged}
  public.audit: skip
writeback:
  allow: ["public.tickets"]
  ledger: public.kleos_writeback
  pool_size: 4
  statement_timeout_ms: 10000
```

`${VAR}` in `connection` is read from the environment; kleos stores the
configuration with the password masked (`kleos_pg_source`).

## Ingest

### Session

On start, the connector:

1. checks the server: `wal_level = logical`, the `REPLICATION` attribute (or
   superuser), version ≥ 14; warns about unbounded `max_slot_wal_keep_size`;
2. reads the catalog, infers the mapping, and stores it if it changed while
   the connector was stopped (DDL while stopped is caught here);
3. creates the write-back ledger (when `writeback.allow` is set) and the
   publication, or adds missing tables to it (`publication_mode: tables`);
4. looks at the slot and the kleos checkpoint:

| Slot | Checkpoint | Then |
|---|---|---|
| none | none | create the slot with an exported snapshot (`CREATE_REPLICATION_SLOT … LOGICAL pgoutput (SNAPSHOT 'export'[, FAILOVER true])`, `EXPORT_SNAPSHOT` on PG14), write the checkpoint, copy every table at that snapshot, stream from the consistent point |
| healthy | yes | stream from the checkpoint; tables not `ready` are resynced first |
| lost (missing, `wal_status = lost`, `invalidation_reason`) | yes | recreate the slot and resync every table at its snapshot, retracting rows that vanished |
| exists | none | a slot left from a crash before the first checkpoint: dropped and recreated |
| another database's, or not pgoutput | any | refused (slot names are cluster-wide; set `slot:`) |

### Snapshot

`BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY; SET TRANSACTION SNAPSHOT
'<exported>'`, then per table `COPY (SELECT <kept columns> FROM t) TO STDOUT`
(text format, decoded by `value::copy_fields`). Tables are copied with
referenced tables first, so foreign-key targets have labels when rows are
rendered. Rows go to the writer in batches of `snapshot_rows`; a table
kleos holds no rows of is marked fresh and its rows skip the stored-image
read. A table is `copying` while it runs and `ready` after the writer
confirms its last batch.

Snapshot rows are asserted at the snapshot's time (the copy transaction's
`now()`), with `by: pg:<source>:snapshot`. Postgres has no per-row commit
times; the flag says these are "true as of the snapshot", not "changed then".

### Stream

`START_REPLICATION SLOT … LOGICAL <checkpoint> ("proto_version" '1',
"publication_names" '…', "messages" 'true')`. Begin…Commit messages are
assembled into one `Tx` (commit LSN, end LSN, commit time); Relation
messages update the column map. Whole transactions are batched (up to
`max_changes` changes or `max_ms`) and sent to the writer over a bounded
queue (`queue` batches).

- **Atomicity.** A batch holds whole transactions; it is one kleos
  transaction. A transaction larger than `max_tx_changes` is applied in
  parts; only the last part advances the checkpoint, and re-applying a part
  is a no-op (deterministic memory ids, diffs against stored images), so a
  crash in the middle converges on replay.
- **Exactly once.** The writer commits the facts and the checkpoint
  (`kleos_pg_checkpoint`: commit LSN, end LSN, commit time, valid-time
  floor, transactions applied) in one kleos transaction, guarded by an
  assertion that the stored LSN is the one it read (two connectors on one
  source cannot both write). Postgres is acknowledged (`flush_lsn`) only up
  to what kleos has committed. A restart asks Postgres to start at the
  checkpoint; any transaction at or below it that still arrives is skipped
  by LSN, in the reader and again in the writer.
- **Idle acknowledgement.** When nothing is in flight, keepalives' WAL end
  (and heartbeat messages) are acknowledged, so the slot advances on quiet
  publications. The in-flight count starts at zero after every copy, so an
  idle acknowledgement can never pass a batch still being written.
- **Backpressure.** While a batch waits for a free queue slot, the reader
  stops reading (TCP backpressure; Postgres keeps the WAL) but keeps
  sending status updates, so `wal_sender_timeout` does not fire. Slot lag is
  bounded by `max_slot_wal_keep_size` (guidance below).
- **TOAST.** An unchanged TOASTed value arrives as `u`; the writer takes it
  from the entity's stored row image (`kleos_pg_row`).
- **Replica identity.** Updates that change the key (old key tuple) end the
  old entity and start a new one; deletes need the key (default identity)
  or the old row (`FULL`).

### Schema changes

A Relation message whose columns differ from the catalog's is DDL. The
connector re-reads that table's catalog entry, infers its new mapping, and
attaches a `SchemaChange` to the transaction, so the writer installs the
mapping (a new mapping version, the ontology redefined, a change row) before
applying the transaction's rows.

- **Added column:** its values become facts as rows change. A column added
  **with a default** (`atthasmissing`) reaches existing rows without any WAL,
  so the table is resynced.
- **Dropped column:** every current fact of its predicate (and of foreign
  keys over it) is retracted at the transaction's time.
- **Renamed column:** a drop and an add.
- **Type change:** remapped (datatype of the property).
- **New table:** an included table never seen is resynced (its existing rows
  come from a copy); its changes until then are skipped. In `tables` mode
  the connector also polls for included tables missing from the
  publication (`new_tables_secs`), adds them and resyncs them.

### Resync

`pg resync [tables]`, a new table, a column default or a lost slot: the
connector stops streaming after the batch in flight, creates a **temporary**
slot with an exported snapshot (consistent point C), copies the tables as of
C, and retracts rows kleos has that the copy did not see (rows deleted while
the slot was gone). Each table records `skip_before = C`; when streaming
resumes from the checkpoint, changes to those tables committed before C are
skipped (they are in the copy). A resync diffs against stored images:
identical values write nothing. Corrections are at the snapshot time, never
before an entity's last version.

### Valid time

A transaction's facts are asserted at its commit time; per source the valid
time is monotonic (`max(commit_ts, last + 1µs)`, guarding against clock
steps), and per entity a new version is never at or before the previous one.
`@ T` reads, `::history`, `as_of` recall and supersession all follow
Postgres's commit order.

## Mapping

### Inference (W3C Direct Mapping style)

| Postgres | kleos |
|---|---|
| table with a primary key | class (singular PascalCase name: `order_items` → `OrderItem`; `COMMENT ON TABLE` is its doc) |
| row | entity `pg:<source>/<schema.table>/<key>` (key values joined by `,`, `%`-escaped); a node of the class with a label |
| scalar column | single-valued, temporal data property `<table>.<column>` (schema-qualified outside `public`); a typed literal fact (`xsd:integer`, `decimal`, `double`, `boolean`, `date`, `dateTime`, `time`, `duration`, `string`, `rdf:JSON` for json and arrays) |
| foreign key | single-valued object property `<table>.<name>` (`customer_id` → `customers.customer`), an edge to the target's entity |
| table that is exactly two foreign keys covering its key | many-valued edge property named after the table; other columns are the edge's attributes |
| text column with sampled average length ≥ `text_min_chars` (or a prose name when unsampled) | a memory of its own (`pg_text`): embedded, FTS-indexed, recallable |
| entity table | a row summary memory (`pg_row`): `Customer Acme (customers 42): plan: pro; city: Lisbon; account manager: Jane Doe.` (foreign keys shown by the target's label) |
| `COMMENT ON COLUMN` | the property's doc |
| enum type | `xsd:string`, its labels in the doc |
| `bytea` | dropped |
| a surrogate key (integer, uuid) | in the entity id only; a natural key is also a fact |
| no primary key | skipped with the reason (Postgres cannot replicate its updates by key either); set `key` to map it |
| partitioned table | the parent (`publish_via_partition_root`) |
| PG19 property graph | vertex labels name classes, keys are keys, edge tables become edges (`pg_get_propgraphdef`, parsed; not yet run against a PG19 server) |

Labels: the configured template, else the first of `name`, `title`,
`full_name`, `display_name`, `username`, `subject`, `label`, `email`,
`code`, `sku`, `slug` (not PII), else `{first_name} {last_name}`.

### Overrides

`tables.<schema.table>` overrides inference only where it says something:
`class`, `key`, `label`, `memory {row, text, template}`, `columns {name:
drop | pii | text | fact | category | {predicate, pii, kind, references}}`,
`edge {source, destination, label}` (the PG19 property-graph words), or
`skip`. `pii` keeps the fact but keeps it out of memory text (and so out of
embeddings); `drop` keeps the column out of kleos entirely, row images
included. `category` makes a node per value (`pg:<source>/cat/<table>.<col>/<value>`).
`references` gives a foreign key to sources without one (Debezium from
MySQL or Mongo).

### The ontology `pg:<source>`

The mapping defines an ontology (classes, properties with domain, range,
datatype, cardinality, docs), stored with `::ontology define` (a valid-time
version per change, `replace` so removed tables and columns retract) and
added to the namespace's active ontologies (after what is active, `personal`
by default; `ontology.activate: false` opts out). So `::ontology facts`,
`infer` and `validate` see the source's facts and types, and the cardinality
policy knows that `customers.plan` holds one value.

Zero maintenance:
- the mapping is stored as data (`kleos_pg_mapping`, versioned) and
  re-inferred at every start and on every schema change;
- every mapping change is a row of `kleos_onto_change` (`kind: pg_map`,
  `before`/`after` the table's properties, by `pg:<source>`), so
  `::ontology history` explains it;
- review mode (`ontology.review: true`) files a proposal
  (`kleos_onto_proposal`, kind `pg_map`) instead of redefining the ontology;
  data keeps flowing with the new mapping; `pg map --approve <version>`
  applies it.

With the learner: row memories carry a claim type (`kleos_mem_type`, class
`pg_row`, subject the entity), so the learner does not re-learn what the
mapping already says. Text memories are left untyped and declarative ones
are learner input: the properties hidden in ticket bodies and notes are what
`::ontology learn` can find, and its vocabulary proposals extend the mapped
properties with the company's own words.

## Writing to kleos

One kleos transaction per batch writes:

| Relation | What |
|---|---|
| `kleos_onto_fact` | assertions and retraction rows (valid time) of data facts, foreign-key edges, join-table edges, categories |
| `kleos_onto_node` | entity nodes (class, label, row memory), retracted on delete |
| `kleos_mem`, `kleos_mem_vec`, `kleos_mem_role`, `kleos_mem_type`, `kleos_mem_link` | row and text memories (role `other`), their vectors when embedding, `supersedes` links from a new version to the old |
| `kleos_pg_row` | the current row image per entity (TOAST fill, diffs, deletes, resync reconciliation) |
| `kleos_pg_checkpoint` | the commit LSN the batch reaches |
| `kleos_pg_echo` | write-back echoes |

- **Update:** only changed columns: the old value's fact gets a retraction
  row and the new value an assertion at the commit time. The row memory gets
  a new version (new id, the old one retracted and linked as superseded), so
  recall answers the current value and the context block shows
  `previously: … (from … until …)`.
- **Delete:** every current fact, edge, node and memory of the entity is
  retracted; history stays readable with `@ T`.
- **Truncate:** every entity of the table, the same way.

Choices that keep agent memory fast and honest:
- Memories are role `other`: never candidates or targets of `remember`'s
  supersede and duplicate decisions. An agent's chat statement cannot retract
  a fact the system of record holds.
- Memory ids are stable for (entity, column, valid time): a replayed change
  writes the same memory, not a second one. Ids contain no `.` (that marks a
  claim).
- Object facts (edges) carry `lit = {src, fk|row, attrs}`. Graph recall
  (`graph: 'ppr'`) loads every current literal-free fact of a namespace per
  recall; source edges would add a full scan of the mirrored graph to every
  recall. They stay queryable with `::ontology facts`, Datalog and live
  queries.
- No memory-API entities by default (`memory.entities: true` opts in):
  `remember`'s entity resolution reads a namespace's entities, and a large
  table would slow every `remember`.
- Embedding: `memory.embed: auto` embeds new memories (batches of 64) when
  the store has an embedder; otherwise memories are FTS-only.

## State

| Relation | Key → values |
|---|---|
| `kleos_pg_source` | `src → conf` (password masked), `actor` (service, human, agent, workspace, scope), created, updated |
| `kleos_pg_checkpoint` | `src → lsn, end_lsn, commit_ts, vt, applied, consistent, slot, updated` |
| `kleos_pg_table` | `src, tbl → state` (`copying`, `ready`, `resync`, `skipped`), `skip_before, rows, info` (catalog entry), `error, updated` |
| `kleos_pg_mapping` | `src, version → mapping, at, reason, status` (`applied`, `proposed`) |
| `kleos_pg_row` | `src, tbl, key → vals, mems, lsn, at` |
| `kleos_pg_echo` | `src, lsn → idem, wf, step, actor, rows, at` |
| `kleos_pg_offset` | `src, part → pos, at` (Debezium positions) |
| `kleos_pg_event` | `src, at, seq → level, msg` (DDL, resyncs, lost slots, errors) |

## Write-back

`{"pg": {"source"?, "sql", "params"?, "statements"?, "idem_key"?}}` is a
flow step (feature `flow-pg`; built-in graphs and code mode's `step
{action}`). It runs on a connection pool (deadpool over the fork's
tokio-postgres; `pool_size`), in one Postgres transaction:

1. `SET LOCAL statement_timeout`; `pg_advisory_xact_lock(hash(idem_key))`
   (one attempt per key at a time);
2. `SELECT result FROM <ledger> WHERE idem_key = $1`: found → commit and
   return `{replayed: true, result}`;
3. the statements (`INSERT`, `UPDATE`, `DELETE`, `MERGE` or `WITH` only;
   parameters sent in text format, so Postgres types them), collecting
   `RETURNING` rows as JSON;
4. the allow-list, checked against what the transaction really modified
   (`pg_stat_xact_user_tables`): a table outside `writeback.allow` rolls
   everything back (a CTE cannot write elsewhere);
5. `INSERT INTO <ledger> (idem_key, workflow, step, actor, result)`; commit.

The idempotency key defaults to `<workflow id>/<step key>`
(`ActionCtx::idem_key`): the same on every retry and replay of the step.
The flow engine records the step's result once; the action itself is at
least once, and the ledger makes its effect exactly once:

| Crash | Then |
|---|---|
| before the Postgres commit | nothing happened; the retry writes |
| after the Postgres commit, before the step is recorded | the retry finds the ledger row and returns its result |
| after the step is recorded | replay returns the recorded result; the action is not called |

`kleos pg init` (or the connector's start) creates the ledger
(`idem_key text PRIMARY KEY, workflow, step, actor, result jsonb,
committed_at`) and puts it in the publication.

**Echo suppression.** The ledger row is in the same Postgres transaction as
the business rows, so the stream delivers them together. The connector does
not map the ledger table; it marks the transaction as the echo of that
step: its facts carry `by: pg:<source>@flow:<workflow>/<step>`, and
`kleos_pg_echo` links the commit LSN to the step. The row becomes one
entity with one row memory, attributed to the step, instead of a second,
unexplained copy. (PG16's `origin 'none'` with replication origins was the
alternative; it needs rights most managed services do not grant.)

## Debezium adapter

`POST /pg/debezium/<source>` takes what Debezium Server's HTTP sink posts:
one event, or a list, with or without the `{schema, payload}` envelope (or
Kafka-style `{key, value}`). It is mapped by the same writer, so MySQL,
MariaDB, MongoDB, SQL Server, Oracle and Postgres sources need no driver in
kleos.

- `c` insert, `u` update, `d` delete, `r` snapshot read (flagged snapshot),
  `t` truncate; tombstones and `m` messages are ignored.
- Valid time: `source.ts_ms` (the commit time).
- Tables: `<schema or db>.<table or collection>`. Keys: the override's
  `key`, else the Kafka key's fields, else `id` / `_id`.
- The catalog entry grows from what events show; with the envelope,
  Debezium's semantic types are decoded (`io.debezium.time.Date`,
  `Timestamp`, `MicroTimestamp`, Kafka Connect `Decimal`, `Json`). Mongo:
  `after` as extended JSON (`$oid`, `$numberLong`, `$date`), and
  `updateDescription` partial updates filled from the stored row.
- Exactly once: each partition (connector + server) keeps its last position
  in `kleos_pg_offset`, committed with the facts; a redelivered event at or
  before it is skipped. Positions: Postgres `lsn`; MySQL binlog file, `pos`,
  `row`; SQL Server `commit_lsn`, `change_lsn`, `event_serial_no`; Mongo
  `ts_ms`, `ord`; Oracle `scn`. A failed request answers 503, so the sink
  retries.

## Live queries

`pg query {sql}` (feature `pg-query`): read-only SQL (DDL, DML and
statements refused) in a DataFusion `SessionContext` per call:

- `pg.<schema>.<table>`: Postgres tables through
  datafusion-table-providers' Postgres provider, federated with
  datafusion-federation: filters, projections and limits, and whole
  sub-plans over Postgres tables (joins, aggregates, TopK), are unparsed to
  SQL and run in Postgres;
- `kleos.facts (s, p, o, value, datatype, by, valid_from)`,
  `kleos.nodes (node, class, label, valid_from)`,
  `kleos.memories (id, text, kind, source, valid_from)`: the namespace's
  current state; equality filters on `s`, `p`, `o`, `by`, `node`, `class`,
  `id`, `kind`, `source` are pushed into the CozoScript scan.

```sql
SELECT c.name, c.plan AS in_postgres, f.o AS in_kleos
FROM pg.public.customers c
JOIN kleos.facts f ON f.s = concat('pg:crm/public.customers/', CAST(c.id AS VARCHAR))
WHERE f.p = 'customers.plan'
```

`explain: true` returns the physical plan. **Path to full SQL**: this is a
separate surface, not CozoScript. Next steps, in order: a generic
`kleos.rel.<relation>` provider (any stored relation, key-prefix pushdown
from equality filters, `@ T` via a session setting); a DataFusion
`TableProvider` over `hybrid_search` (vector + FTS as a table function); then
a `::sql` system op in kleos-core behind a feature (DataFusion is too large
for the default build).

## Ops

`pg status` (CLI, `POST /pg/status`, MCP `pg_status`, `Kleos.pg('status')`,
`PgClient.status()`):

- `lag`: bytes (`pg_wal_lsn_diff(current, confirmed_flush_lsn)`) and seconds
  (age of the last applied commit while WAL is pending);
- `postgres.slot`: active, `wal_status`, restart and confirmed LSNs,
  retained bytes, `safe_wal_size`, `invalidation_reason` (PG17+, `conflict`
  on PG16), `failover`, `inactive_since`;
- `checkpoint`: last commit LSN, end LSN, commit time, transactions applied;
- `tables`: per-table state, rows copied, `skip_before`;
- `connector`: the running connector's state (`snapshot`, `streaming`,
  `resync`, `error`), received/applied/acknowledged LSNs, queue depth,
  restarts, last error, and counters (transactions, rows, facts, memories,
  replays skipped, echoes);
- `alerts`: lag over `lag_alert.bytes` / `seconds`, an invalidated or
  missing slot, WAL past `max_slot_wal_keep_size` (`wal_status =
  unreserved`), an inactive slot while a connector runs, blockers
  (`wal_level`, `REPLICATION`); `warnings`: unbounded
  `max_slot_wal_keep_size`, PG14 end of life;
- `events`: the last DDL, resyncs, lost slots and errors.

**WAL retention.** A stalled connector's slot pins WAL from its
`restart_lsn` and can fill the source's disk. Set `max_slot_wal_keep_size`
(e.g. 2–4× the WAL a day writes, at least the longest planned outage): past
it Postgres invalidates the slot instead, and kleos-pg resyncs. PG18's
`idle_replication_slot_timeout` does the same for idle slots. `pg drop-slot`
releases the WAL at once (stop the connector first; the next start resyncs).

**Errors** restart the session with back-off (1 s doubling to 30 s), from
the checkpoint.

### Postgres versions and providers

PG 15–18 supported; PG14 best effort (end of life 2026-11-12; old
`CREATE_REPLICATION_SLOT` syntax). Tested on 16.13, 17.11 and 18.6.

| | |
|---|---|
| PG15+ | `FOR TABLES IN SCHEMA` (`publication_mode: schema`), row filters and column lists (DBA-made publications work) |
| PG16+ | logical decoding on a standby (connect to the standby; the publication is made on the primary) |
| PG17+ | failover slots (`failover: auto`): set `sync_replication_slots = on` on the standby and `synchronized_standby_slots` on the primary, so a promoted standby keeps the slot |
| PG18 | `idle_replication_slot_timeout` invalidations are detected and resynced |
| RDS / Aurora | `rds.logical_replication = 1` (reboot); `GRANT rds_replication`; PG17 slot sync supported |
| Cloud SQL | `cloudsql.logical_decoding = on`; `ALTER USER … REPLICATION` |
| Azure Flexible Server | `wal_level = logical` (restart), `ALTER ROLE … WITH REPLICATION`, `max_worker_processes` ≥ 16 |
| Supabase | the direct connection (port 5432, IPv6 unless the IPv4 add-on), never the Supavisor pooler |
| Neon | enable logical replication; an inactive slot is removed after ~40 h when other slots are active, and branch restores delete slots: both are a lost slot (resync); an active slot prevents scale-to-zero |
| PgBouncer and other poolers | replication connections bypass them |

## Surfaces

- `kleos server --pg-cdc <config.yaml>` (repeatable): one connector task
  per source in the server's runtime; with `flow-pg` the `pg` action is
  registered for `--flow-worker`.
- `kleos pg run --pg-config <yaml> --db <store>`: the same connector as a
  standalone foreground worker over an embedded store.
- CLI: `kleos pg init | status | resync [--table …] | drop-slot --yes | map
  [--infer] [--approve N] | query <sql>`, embedded (`--pg-config`, `--db`)
  or remote (`--url`).
- HTTP: `POST /pg/:op` (read-only principals: `status`, `map`, `query`);
  `POST /pg/debezium/:source`.
- MCP: `pg_status` (read-only), `pg_query` (with `--mcp-allow-query`).
- SDKs: TypeScript `Kleos.pg(op, args)` (remote); Python
  `kleos_pg.PgClient` and `config_yaml` (`kleos-lib-python/integrations/kleos-pg`).

## Tenancy

The configuration carries `workspace` (default `default`, today's implicit
workspace), `ns` (the scope inside it) and `actor` (a service principal,
default `connector:pg/<source>`, plus optional human and agent), stored in
`kleos_pg_source.actor`; facts carry `by: pg:<source>` and the write-back
ledger records the actor. When tenancy lands (docs/specs/tenancy.md on
`wave7/tenancy`), the connector opens the workspace's partition view, writes
with an actor chain `{service: actor.service, human, agent, run: <source>}`
into scope `ns` (the natural default is the `workspace` scope, `ws`), and
write-back checks `write` on the target scope; the fields exist, so the
wiring is mechanical.

## Results

Partial (the benchmark was stopped early; [docs/benchmarks/pg-connector.md](../benchmarks/pg-connector.md)),
100k-row CRM schema, kleos on a slow external SSD: RocksDB snapshot 2,852
rows/s, streaming 754 changes/s (151 tx/s), commit → recallable p50 63 ms /
p99 96 ms; SQLite 251 rows/s (fsync-bound). Recall after three updates per
account: current value in the top 5 for 200/200, stale at rank 1 for 0/200.
Integration tests pass on Postgres 16, 17 and 18.

## Left

- **Graph recall cost in a source-heavy namespace** (kleos-core): `recall`
  with `graph: 'ppr'` (the default) scans every fact of the namespace per
  call; with 385k source facts that is ~1.4 s. Options: write the source's
  facts and nodes to their own namespace (`pg:<source>`, memories staying in
  the agent's), an index or cache in `onto_graph`, or `graph: 'off'`.
- SQLite throughput is fsync-bound (rollback journal, 2 MB cache): a WAL /
  cache option in kleos-core's SQLite storage would help; RocksDB is ~11×
  faster today.
- A table is copied on one connection (etl parallelizes one table's copy by
  ctid ranges); kleos's write rate is the bottleneck today.
- pgoutput protocol 1: a large transaction arrives at its commit (Postgres
  spills it); protocol 2 streaming of in-progress transactions is not used.
- Text memories of a dropped text column stay until the entity changes.
- Row memories mention foreign-key targets by the label they had when the
  row was written.
- Live queries: the kleos side is three fixed tables; see "Path to full SQL".
- PG19 property-graph import is parsed but not yet run against a PG19 server.
