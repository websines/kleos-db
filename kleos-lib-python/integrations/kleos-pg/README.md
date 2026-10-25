# kleos-pg (Python)

A small, dependency-free client for the Postgres connector of a running
`kleos server --pg-cdc <config.yaml>` (docs/specs/pg-connector.md).

```python
from kleos_pg import PgClient, config_yaml

pg = PgClient("http://127.0.0.1:9070", token="...")
for s in pg.status():                       # every source
    print(s["source"], s["lag"], s["alerts"])
pg.resync("crm", tables=["public.orders"])  # copy tables again
pg.map("crm", infer=True)                   # the mapping, and what re-inference would change
rows = pg.query("crm", "SELECT name, plan FROM pg.public.customers WHERE plan = 'pro' LIMIT 5")

print(config_yaml("crm", "postgres://kleos:${PGPASSWORD}@db.internal:5432/app?sslmode=require",
                  include=["public.*"], writeback_allow=["public.tickets"]))
```

The connector itself runs inside the kleos server (Rust); this package only
talks to its HTTP endpoints (`POST /pg/:op`).
