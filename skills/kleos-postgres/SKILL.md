---
name: kleos-postgres
description: Work with business data kleos mirrors from Postgres (customers, orders, tickets, ...) and write changes back safely. Use when the user asks about records in the company database, wants an agent to change a row, or asks whether kleos is in sync with Postgres.
---

# kleos and Postgres

A kleos server started with `--pg-cdc` mirrors a Postgres database: every
row is an entity `pg:<source>/<schema.table>/<key>` with one fact per column
(`customers.plan`, ...), edges for foreign keys, and a recallable summary
memory; changes arrive within milliseconds. Postgres stays the system of
record: you read from kleos, you write to Postgres.

## Read

- Questions about a record: `recall` works ("what plan is Acme on?"); the
  summary memories hold current values, and the context block shows older
  ones as `previously`.
- Exact values: facts of the entity, e.g.
  `kleos query "?[p, o] := *kleos_onto_fact{ns: 'default', s: 'pg:crm/public.customers/42', p, o @ 'NOW'}"`.
  Add `@ '2026-09-01T00:00:00Z'` instead of `'NOW'` for the past.
- Live data or many rows: `kleos pg query "SELECT ... FROM pg.public.orders WHERE ..."`
  (or the `pg_query` MCP tool) runs in Postgres; join `kleos.facts` to
  compare with kleos.

## Write

Never write SQL to Postgres directly from a chat turn. Use a workflow step,
which commits once even if it is retried:

```bash
kleos flow start open_ticket --input '{"customer_id": 42, "title": "Refund"}'
```

where the workflow has a step `{"pg": {"sql": "INSERT INTO tickets ... RETURNING id", "params": [...]}}`.
Only tables the operator allowed (`writeback.allow`) can be written; others
fail and nothing is changed. The new row comes back into kleos linked to the
step (`by: pg:<source>@flow:<workflow>/<step>`).

## Is it in sync?

`kleos pg status` (MCP `pg_status`): `lag` in bytes and seconds, `alerts`,
per-table state. Lag of a few seconds during bulk loads is normal. A lost
slot resyncs by itself on restart. If one table looks stale, ask the
operator before running `kleos pg resync --table <schema.table>`; never run
`pg drop-slot` unless asked.
