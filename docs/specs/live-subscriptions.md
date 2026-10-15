# Spec — Live query subscriptions (F1)

_Status: **implemented on `wave1/live`, 2026-10-15** (UPGRADE_PLAN.md item F1).
Code: `kleos-core/src/runtime/live.rs` (hub, worker, `Subscription`), one hook in
`SessionTx::commit_tx` (`runtime/transact.rs`), one in `SessionTx::get_relation`
(`runtime/relation.rs`). Tests: `kleos-core/tests/live_subscriptions.rs`. Benchmark:
`kleos-core/examples/livebench.rs`. HTTP: `GET /subscribe` in `kleos-bin`. Python:
`db.subscribe(...)`._

> **One-line goal.** An agent subscribes to the query whose answer it cares about and is
> told when that answer changes, with the rows that changed, instead of polling.

---

## 1. Why

Agents keep working memory in sync with the store: "my open tasks", "facts about the
current user", "documents similar to this one". Polling either wastes work or reacts
late. `register_callback` already streams the raw `Put`/`Rm` rows of one relation, but an
agent rarely cares about a relation. It cares about a query over several relations,
rules and search indexes. F1 pushes changes to the query's answer.

## 2. API

```rust
use kleos::{SubscribeOptions, SubscriptionEvent};

let sub = db.subscribe(
    "?[k, v] := *kv{k, v}, k >= $min",
    BTreeMap::from([("min".into(), DataValue::from(2))]),
    SubscribeOptions::default(),          // queue_capacity: 64
)?;
while let Some(ev) = sub.recv() {
    match ev {
        SubscriptionEvent::Initial { rows } => { /* the result now */ }
        SubscriptionEvent::Delta { added, removed } => { /* apply removed, then added */ }
        SubscriptionEvent::Resync { rows } => { /* replace local state */ }
        SubscriptionEvent::Error { message } => { /* re-evaluation failed */ }
    }
}
```

| Call | Notes |
|---|---|
| `Db::subscribe(script, params, opts) -> Result<Subscription>` | Also on `DbInstance`. Evaluates once on the caller's thread; parse errors, rejected scripts and first-evaluation errors are returned here. |
| `Subscription::recv()`, `try_recv()`, `recv_timeout(d)` | `None` once closed. |
| `Subscription::id()`, `Db::unsubscribe(id)` | Unsubscribing by id closes the handle's stream. Dropping the handle (or `sub.unsubscribe()`) unsubscribes too. |
| `Db::live_stats()` | `subscriptions`, `groups`, `evaluations`, `notified_commits`. |
| Python: `db.subscribe(query, params, callback, queue=None) -> int`, `db.unsubscribe(id)` | `callback(event: dict)` runs on a background thread; `event["kind"]` is `initial`, `delta`, `resync` or `error`. |
| HTTP: `GET /subscribe?script=…&params={json}&queue=64` | Server-sent events named `initial`, `delta`, `resync`, `error`, with JSON rows. Behind the server's normal auth. The subscription ends when the client disconnects (checked once a second). |

Only a single read-only query can be subscribed. Rejected, with an error that starts
"live subscriptions take …": any `:put`/`:rm`/`:create`/`:replace`/… output option
(temp relations included), `::` system ops, and imperative scripts (`%if`, `%loop`, chained
`{}` blocks). Not available on wasm32 (no threads).

## 3. Events

| Event | When | Payload |
|---|---|---|
| `Initial` | First event of every subscription. | Full result, in the query's own order. |
| `Delta` | A committed write changed the result. | `added`, `removed`: rows (with headers) to apply to the previous result, treated as a multiset. Each side is sorted by value, not by the query's `:order`. An update to a row is one removal plus one addition. |
| `Resync` | The subscriber's queue filled up, or the previous re-evaluation failed and this one succeeded. | Full current result. Replace local state. |
| `Error` | A re-evaluation failed (a relation was removed, the query hit its time or memory budget, …). | Rendered error. The subscription stays registered. |

A write that leaves the result unchanged is re-evaluated but delivers nothing. Row
payloads are `Arc<NamedRows>`, shared by every subscriber of the same query.

For `:order` / `:limit` queries the delta is still a set difference: applying it gives
the right rows, but their order is only carried by `Initial`/`Resync`. A row that drops
out of a `:limit` window arrives as `removed`, the row that enters as `added`.

## 4. Dependencies

A subscription is re-evaluated only when a committed transaction wrote a stored
relation the query read. The dependency set is **recorded, not inferred**: while the
query runs, every stored relation it resolves by name is logged (the single funnel is
`SessionTx::get_relation`). That covers:

- relations read directly (`*rel{…}`, `*rel[…]`) at any depth of rules, including
  recursive and negated ones;
- fixed-rule inputs (`ConnectedComponents(*edge[a, b])`);
- HNSW, FTS and LSH searches: `~rel:idx{…}` resolves the base relation `rel`;
- index relations read directly (`*rel:idx{…}`): mapped to their base relation, since
  writes dirty only the base;
- stored queries: their bodies and the `kleos_stored_queries` catalog row.

The set is recomputed at every evaluation, so it follows the query if the relations it
reads change (`:replace` gives a relation a new id). After a failed evaluation the
subscription watches **every** commit until an evaluation succeeds; this is how a
subscription whose relation was removed picks the relation up again once it is
re-created and written.

Writes are matched by relation id against the transaction's dirty set (the same set
graph-projection freshness uses), so every write path counts: `:put`/`:rm`/`:update`/…,
triggers, `::retract`/`::erase` cascades, `::reindex`, `::evict`, `import_relations`,
columnar import. (Change callbacks do not fire for the system-op and import paths;
subscriptions do.)

## 5. Delivery guarantees

1. **Only committed data.** The hook is in `SessionTx::commit_tx`, the commit funnel of
   every write path, and runs only when the storage commit returned `Ok`. An aborted
   multi-transaction, a script that fails part-way, a failed commit and a dropped
   transaction notify nothing. Re-evaluation then opens its own read transaction, which
   can only see committed state.
2. **Writers never wait for subscribers.** The commit hook is one atomic load when there
   are no subscriptions; otherwise one short mutex section that keeps only the watched
   relation ids and wakes the worker. It never evaluates a query and never touches a
   subscriber queue.
3. **One background worker per `Db`.** Spawned with the first subscription, exits with the
   last. It drains the pending relation set and re-evaluates each affected group once,
   so several commits that land while it is busy cost one evaluation per group
   (coalescing). Groups are evaluated one after another.
4. **Bounded queues, coalesced on overflow.** Each subscriber has a queue of
   `queue_capacity` events. When it is full the subscriber is marked lagging; later
   deltas are not queued for it. Its next receive discards what is queued and returns
   `Resync` with the group's latest result. Other subscribers of the same group are not
   affected.
5. **Consistency.** Every event sequence a subscriber sees is consistent: starting from
   `Initial` (or the latest `Resync`) and applying the deltas in order yields the result
   of the most recent evaluation. Each evaluation reflects one committed snapshot taken
   after the commit that triggered it, possibly including later commits. Intermediate
   states can be skipped (coalescing); after writes stop, every subscriber converges to
   the current result.
6. **Asynchronous.** A delta may arrive before or after the writer's `run_script`
   returns; there is no read-your-writes ordering between the writer and a subscriber.

## 6. Dedup

Subscriptions with identical script text and parameters share one group: one
evaluation per change, one snapshot, one diff, fanned out to each subscriber's queue. A
subscriber joining an existing group gets that group's latest snapshot as its `Initial`,
without a new evaluation; deltas that follow are relative to that snapshot. Dedup is by
exact text: `?[a] := *r{a}` and `?[a] := *r{a: a}` are different groups.

## 7. Time (`NOW`)

`'NOW'` / `@ 'NOW'` validity and `now()` are evaluated when the query is evaluated: at
subscribe time, then at each re-evaluation. **A result that changes only because time
passes is not pushed.** A fact asserted with a future valid time becomes current when
that time arrives, but nothing is written then, so the subscription stays silent until
some later commit to one of its relations triggers a re-evaluation. The same holds for
`now()`-based filters (`created > now() - 3600`). Workaround today: re-subscribe or poll
for time-driven windows. Schedule-driven re-evaluation (re-run at the next valid-time
boundary, or on a timer) is future work, alongside F4 (scheduled jobs).

Fixed `@ T` queries are unaffected: their answer can only change through writes.

## 8. Cost

Each change costs one full evaluation of each affected group: subscriptions are not
incremental (F2 is). The writer pays almost nothing; the worker pays
`(affected groups) × (query cost)` per batch of commits. Memory: each group keeps its
last result twice (in query order and sorted, for the diff).

`examples/livebench.rs`, release build, mem backend, M3 shared with other builds (numbers
are under contention). N point-lookup subscriptions with distinct parameters (no dedup);
`hot` = all N on the written relation, `cold` = all N on another relation.

RESULTS_TABLE

## 9. Limits and caveats

- Full re-evaluation per change, on one worker thread, groups in sequence. Many expensive
  subscriptions on a hot relation delay each other; coalescing keeps writers unaffected
  but delivery latency grows with `groups × query cost`.
- Data the engine does not commit is not tracked: fixed rules that read files or URLs
  (`CsvReader`, `JsonReader`), re-rank models, and custom fixed rules with outside state.
- Access-level changes and `::rename` are not writes to the relation; a renamed relation
  makes the next triggered evaluation fail (`Error`), after which the subscription
  watches everything until it succeeds again.
- The worker holds a clone of the `Db`; the store stays open while any subscription is
  alive, even if every other handle was dropped.
- Imperative scripts cannot be subscribed (split into a single query, or subscribe to
  each part).
- `kleos-bin`'s SSE stream polls for client disconnect once a second, so a dropped
  client's subscription lingers for up to a second.

## 10. Design decisions

| Decision | Why |
|---|---|
| Record dependencies at `get_relation` instead of walking the program | One funnel covers rules, fixed rules, searches, index reads, stored queries and magic-set rewrites. A walk of the AST would have to be kept in step with every new atom kind. |
| Hook the commit funnel, not the callback collector | The callback collector only sees `:put`/`:rm` statements and builds row copies. The dirty set covers every write path, costs nothing extra, and is already exact per transaction. |
| Re-evaluate on a worker, diff against the last result | Simple and always correct, including for aggregates, negation, recursion and search. Incremental maintenance is F2. |
| Coalesce to `Resync` instead of blocking or dropping silently | Writers must not stall behind a slow agent, and a subscriber must be told it missed something. |
| Watch everything after a failed evaluation | The failure usually means the dependency set is stale (relation removed or renamed); over-notifying until recovery is cheap and never misses the recovery. |
