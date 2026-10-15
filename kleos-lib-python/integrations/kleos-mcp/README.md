# kleos-mcp

A local **memory MCP server** backed by
[kleos](https://github.com/websines/kleos-db) — an embedded graph + vector +
full-text database (a maintained fork of CozoDB). One process, one file, fully
local: your agent's memory never leaves the machine.

> kleos is derived from [mnestic](https://github.com/shuruheel/mnestic), a maintained fork of [CozoDB](https://github.com/cozodb/cozo); it is not affiliated with either. Original design © Ziyang Hu and the Cozo Project Authors; mnestic additions © Shan Rizvi.

## Install

Add to any MCP client (Claude Desktop, Claude Code, Cursor, ...):

```json
{ "command": "uvx", "args": ["kleos-mcp"] }
```

That's it. The database lives in your platform data dir (override with
`--db PATH` or `KLEOS_MCP_DB`); embeddings run locally via
[fastembed](https://github.com/qdrant/fastembed) (default
`BAAI/bge-small-en-v1.5`, a one-time ~67 MB download that happens in the
background — keyword search works immediately, before the model arrives).
Pre-provision with `uvx kleos-mcp --download-model`.

## Tools

`store`, `store_batch`, `search`, `find_related`, `list_recent`, `update`,
`delete`, `link`, `recall_as_of`, `stats` — plus the two things no other local
memory server has:

- **`search(explain=true)`** — per-leg attribution: exactly how much the
  keyword (BM25), vector (HNSW), and graph-proximity legs each contributed to
  every result, straight from the engine's fused three-way retrieval. Ask
  your agent *"why did you recall that?"* and get a real answer.
- **`recall_as_of(t)`** — time travel. Updates and deletes are never
  destructive (valid-time history in the engine's bitemporal storage):
  *"what did you know about this before last Tuesday?"* just works.

`search` is keyword-first with an automatic hybrid fallback; `link` builds a
typed, weighted memory graph that both `find_related` (budget-bounded
traversal) and the hybrid graph leg exploit.

**Byte-budgeted output** (0.2.0): every wide-result tool (`search`,
`find_related`, `list_recent`, `recall_as_of`) takes `max_bytes` (default
20 000). An over-budget result is cut **at item boundaries** — item content is
never trimmed — and returns `truncated: true`, a `remaining` count, and a
single-use `continuation` token; **`continue_result(token)`** serves the next
slice under the same budget. A wide result can no longer blow your agent's
context with no way back. (Tokens don't survive a server restart; the error
says so and tells you to re-run the query.)

## Configuration

| Flag / env | Default | Meaning |
|---|---|---|
| `--db` / `KLEOS_MCP_DB` | per-OS data dir | database file |
| `--engine` / `KLEOS_MCP_ENGINE` | `sqlite` | `sqlite` \| `rocksdb` (single-instance) \| `mem` |
| `--model` / `KLEOS_MCP_MODEL` | `BAAI/bge-small-en-v1.5` | any fastembed model |
| `KLEOS_MCP_CACHE` | data dir `/models` | model cache location |

A database pins the embedding model it was built with — mixing embedding
spaces is refused with an actionable error, never a silent quality collapse.

## Notes

- Requires Python ≥ 3.10. Storage engines ship in the `kleos` wheel
  (`sqlite`/`mem` also in the sdist; `rocksdb` wheel-only).
- Several MCP clients may share one sqlite database; writes serialize on the
  file. `rocksdb` is strictly single-instance.
- Deleting memories under heavy churn can slowly degrade vector recall
  (a known engine-side HNSW defect, tracked upstream); updates are safe.

## License

Mozilla Public License 2.0.
