# langgraph-store-kleos

A [LangGraph](https://github.com/langchain-ai/langgraph) `BaseStore` backed by
[kleos](https://github.com/websines/kleos-db) — an embedded graph + vector +
full-text database (a maintained fork of CozoDB). One process, one file, no
server: agent long-term memory with **atomic batches** and **in-engine hybrid
retrieval** (BM25 + vector fused with Reciprocal Rank Fusion — a fusion that
Postgres + pgvector, LangGraph's default production store, cannot do in one
system).

> kleos is derived from [mnestic](https://github.com/shuruheel/mnestic), a maintained fork of [CozoDB](https://github.com/cozodb/cozo); it is not affiliated with either. Original design © Ziyang Hu and the Cozo Project Authors; mnestic additions © Shan Rizvi.

```bash
pip install langgraph-store-kleos
```

```python
from langgraph_store_kleos import KleosStore

store = KleosStore(
    engine="sqlite", path="memory.db",          # or engine="mem" for ephemeral
    index={"dims": 1536, "embed": my_embed_fn}, # any LangGraph IndexConfig embed
    ttl={"default_ttl": 60 * 24, "refresh_on_read": True},  # optional
)

store.put(("users", "u1"), "pref", {"text": "prefers window seats"})
hits = store.search(("users", "u1"), query="seating preference")
```

Use it in a graph like any store:

```python
graph = builder.compile(store=store)
```

## What you get

- **Atomic `batch()`** — every write in a batch lands in one engine
  transaction. Under LangGraph's parallel fan-out this is the difference
  between correct recall and silently losing reads (a non-atomic prototype
  lost 36% of concurrent semantic reads; this store's concurrency suite
  gates on zero).
- **Hybrid `search(query=...)`** — BM25 keyword + HNSW vector legs, namespace
  filter pushed into both index searches, RRF-fused and hydrated in a single
  engine snapshot. Without an `index` config the store is BM25-only (keyword
  search still works, no embeddings needed).
- **Collision-safe namespaces** — any label content stays distinct
  (`('a.b',)` can never collide with `('a', 'b')`).
- **TTL** — per-item or `default_ttl` (minutes), lazy expiry on every read
  path, optional refresh-on-read, and a `start_ttl_sweeper()` background
  reclaimer.
- **Sync + async** — full `BaseStore` surface; `a*` methods run the sync
  engine calls on the default executor.

## LangGraph Platform (`langgraph.json`)

Custom stores are loadable by import path (alpha upstream feature):

```python
# src/agent/store.py
from contextlib import asynccontextmanager
from langgraph_store_kleos import KleosStore

@asynccontextmanager
async def generate_store():
    store = KleosStore(engine="sqlite", path="memory.db", index={...})
    try:
        yield store
    finally:
        store.close()
```

```json
{ "store": { "path": "./src/agent/store.py:generate_store" } }
```

## Notes

- Requires Python ≥ 3.10 and `langgraph-checkpoint >= 4.1`.
- The wheel build of `kleos` ships `mem`, `sqlite`, and `rocksdb` engines
  (the sdist has no rocksdb).
- The text index applies English stemming + stopwords: queries made only of
  very common words match nothing on the keyword leg (the vector leg still
  answers when an `index` config is present).
- Filter operators supported in `search(filter=...)`: `$eq`, `$ne`, `$gt`,
  `$gte`, `$lt`, `$lte`, with `InMemoryStore`-parity semantics.

## License

Mozilla Public License 2.0.
