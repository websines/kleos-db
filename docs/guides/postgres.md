# Running kleos next to Postgres

kleos mirrors a Postgres database into agent memory: every row becomes an
entity with typed facts over time, connected by its foreign keys and findable
by `recall`; changes stream in within milliseconds; agents write back as
durable workflow steps that happen once. This guide sets it up. The design is
in [docs/specs/pg-connector.md](../specs/pg-connector.md).

## 1. Build

```bash
cargo build --release -p kleos-bin --features "pg-cdc flow-pg"          # connector + write-back
cargo build --release -p kleos-bin --features "pg-cdc flow-pg pg-query" # + live SQL (DataFusion; slower build)
```

## 2. Prepare Postgres

You need `wal_level = logical`, a role with `REPLICATION`, `SELECT` on the
tables, and either `CREATE` on the database plus ownership of the tables
(Postgres's rule for publications; kleos then creates its own) or a DBA who
runs the `CREATE PUBLICATION` that `kleos pg init` prints.
Tables need a primary key (or `key:` in the config).

```sql
CREATE ROLE kleos WITH LOGIN REPLICATION PASSWORD '…';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO kleos;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO kleos;
-- the publication, run by the tables' owner (kleos pg init prints it; with
-- CREATE on the database and table ownership kleos makes and extends it itself):
CREATE PUBLICATION kleos_crm FOR TABLE public.customers, public.orders, public.tickets
  WITH (publish_via_partition_root = true);
GRANT CREATE ON SCHEMA public TO kleos;         -- the write-back ledger (or create it: kleos pg init prints the DDL)
-- write-back: the tables agents may write
GRANT INSERT, UPDATE ON public.tickets TO kleos;
```

Set `max_slot_wal_keep_size` so a stopped connector cannot fill the disk
(Postgres invalidates the slot instead and kleos resyncs):

```sql
ALTER SYSTEM SET max_slot_wal_keep_size = '50GB';  SELECT pg_reload_conf();
```

### Managed providers

| Provider | Turn on logical replication | Replication right | Notes |
|---|---|---|---|
| AWS RDS / Aurora | parameter group `rds.logical_replication = 1`, reboot | `GRANT rds_replication TO kleos` | PG17 failover slots: `sync_replication_slots` on the reader |
| Google Cloud SQL | flag `cloudsql.logical_decoding = on` | `ALTER USER kleos REPLICATION` | |
| Azure Flexible Server | server parameter `wal_level = logical`, restart | `ALTER ROLE kleos WITH REPLICATION` | `max_worker_processes` ≥ 16 |
| Supabase | on by default | the `postgres` role, or a role with `REPLICATION` | connect **directly** (`db.<ref>.supabase.co:5432`, IPv6 unless you have the IPv4 add-on), never through the Supavisor pooler |
| Neon | project settings → logical replication | the project owner role | an idle slot is dropped after ~40 h if other slots are active, and branch restores delete slots: kleos detects the lost slot and resyncs; an active connector keeps the compute awake |
| Self-hosted | `wal_level = logical` in `postgresql.conf`, restart | `ALTER ROLE … REPLICATION` | bypass PgBouncer for the replication connection |

Use `sslmode=require` (encrypt) or `verify-full` with `sslrootcert=` on
managed services.

## 3. Configure

```yaml
# crm.yaml
source: crm
connection: postgres://kleos:${PGPASSWORD}@db.internal:5432/app?sslmode=require
include: ["public.*"]
exclude: ["public.audit_*", "public.schema_migrations"]
writeback:
  allow: ["public.tickets"]
```

That is enough: classes, properties, edges and memories are inferred. See
"Mapping overrides" below for what you can change.

## 4. Check and start

```bash
export PGPASSWORD=…
kleos pg init --pg-config crm.yaml --db ./kleos.db   # checks, publication, ledger, mapping preview
kleos server -e sqlite -p ./kleos.db --pg-cdc crm.yaml --flow-worker default --mcp
```

`init` reports blockers (no `wal_level = logical`, no `REPLICATION`),
warnings (unbounded WAL retention), what it created, the DDL a DBA should run
if it could not, and every table's mapping (class, facts, edges, text
memories, skipped tables and why).

The server copies every table at one snapshot, then streams. Watch it:

```bash
kleos pg status --url http://127.0.0.1:9070 --source crm
```

## 5. Use it

```bash
kleos recall "what plan is Acme on?" --url http://127.0.0.1:9070
kleos query "?[p, o] := *kleos_onto_fact{ns: 'default', s: 'pg:crm/public.customers/42', p, o @ 'NOW'}" --url …
kleos ontology history --url …             # how the mapping evolved
kleos pg query "SELECT name, plan FROM pg.public.customers WHERE plan = 'enterprise' LIMIT 10" --url …
```

- Recall finds row summaries (`Customer Acme (customers 42): plan: pro; …`)
  and long text columns (ticket bodies, notes). After an update, the current
  version answers and the context block shows the old value as `previously`.
- Facts are queryable as of any time: `@ '2026-10-01T00:00:00Z'`.
- `kleos.facts` in live queries joins kleos's view with Postgres's.

## Mapping overrides

```yaml
tables:
  public.customers:
    class: Client                         # instead of Customer
    label: "{company_name}"
    memory: {text: [notes], row: true}    # which columns are memories of their own
    columns:
      ssn: drop                           # never leaves Postgres
      email: pii                          # a fact, never in memory text or embeddings
      status: category                    # a node per value
      region: {predicate: customer.region}
  public.order_items:                     # a join table with attributes is an edge
    edge: {source: order_id -> public.orders, destination: product_id -> public.products, label: contains}
  public.events: skip
```

Review mode (`ontology: {review: true}`) holds mapping changes as proposals
until `kleos pg map --approve <version>`; data keeps flowing meanwhile.
`kleos pg map --infer` shows what re-inference would change without
applying it.

## Write-back

Agents change Postgres through a workflow step; each step writes once, even
across crashes and retries:

```json
{"steps": [
  {"name": "t", "pg": {"sql": "INSERT INTO tickets (customer_id, title, status) VALUES ($1, $2, 'open') RETURNING id",
                       "params": ["$input.customer_id", "$input.title"]},
   "retry": {"max_attempts": 5}},
  {"return": "$t"}]}
```

```bash
kleos flow start open_ticket --input '{"customer_id": 42, "title": "Refund"}' --url …
```

The insert and a row in `kleos_writeback` commit together; a retry finds the
row and returns the stored result. Only tables in `writeback.allow` can be
written (checked against what the transaction actually changed). When the
new row streams back, its facts say which step wrote it
(`by: pg:crm@flow:<workflow>/<step>`).

## Other databases through Debezium

Run Debezium Server with the HTTP sink pointed at kleos:

```properties
debezium.sink.type=http
debezium.sink.http.url=http://kleos:9070/pg/debezium/shop
debezium.sink.http.headers.x-cozo-auth=<kleos token>
debezium.source.connector.class=io.debezium.connector.mysql.MySqlConnector
# … the usual source settings
```

Events from MySQL, MongoDB, SQL Server or Oracle map the same way; keys come
from the event key or an `id` / `_id` column (or `tables.<t>.key`), foreign
keys from `columns: {customer_id: {references: inventory.customers}}`.
Redelivered events are skipped by source position.

## Operations

| Situation | What to do |
|---|---|
| `alerts` shows lag over the threshold | check the connector's `last_error`; kleos writes slower than Postgres changes → a faster kleos engine (RocksDB), fewer text columns as memories, `memory.embed: false` |
| slot invalidated / missing (failover on PG<17, Neon cleanup, `max_slot_wal_keep_size`) | nothing: the next start recreates the slot and resyncs, retracting rows deleted meanwhile |
| a table looks out of date | `kleos pg resync --table public.orders` |
| decommission | stop the server, `kleos pg drop-slot --yes`, `DROP PUBLICATION kleos_crm` |
| several sources | one config per source, `--pg-cdc a.yaml --pg-cdc b.yaml`; give each a distinct `slot` if they share a cluster |
| standby | PG16+: connect to the standby; PG17+: failover slots (`failover: auto`) with `sync_replication_slots = on` |
