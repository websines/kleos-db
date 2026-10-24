# Durable workflows (`::flow`)

*Status: implemented, 2026-10-24, branch `wave4/flow`. Code:
`kleos-core/src/flow.rs` (store, ops, actions), `flow_builtin.rs` (step
graphs), `flow_cron.rs` (schedules), `DbInstance::flow_worker` (in-process
worker thread, `kleos-core/src/lib.rs`), the `remember_with` hook in
`memory.rs`, the Engram hook in `memory_engram.rs`; Python binding
`CozoDbPy.flow_call` / `flow_worker`; Python SDK
`kleos-lib-python/integrations/kleos-flow/` (`kleos_flow`); HTTP `POST
/flow/:op` and `kleos server --flow-worker <queue>`. Tests:
`kleos-core/tests/flow.rs`, `integrations/kleos-flow/tests/test_flow.py`;
end to end `scripts/flow/demo.py`, `scripts/engram/remember_e2e.py`.*

## Goal

Long-running, crash-proof work inside the database, with no extra service:
the features of Temporal (workflows as ordinary code, replayed from a saved
history; retries; durable timers; signals; child workflows; schedules;
versioning) and of pg_durable (workflows as query graphs run by a worker
inside the database; every step checkpointed; fan-out and join), built on
kleos relations and transactions.

What neither has: **a step's data writes commit in the same transaction as
its checkpoint.** A step that remembers a fact or writes rows either happened
completely (data + checkpoint) or not at all. With Temporal the workflow
history and the application database are separate systems, so a crash between
them leaves them out of step; activities must be made idempotent by hand.

Non-goals (this version): multi-node clusters, a hosted UI, workflow code in
languages other than Python (Rust and HTTP clients use the same worker ops),
cross-database workflows.

## Decision: build our own, not duroxide

[duroxide](https://github.com/microsoft/duroxide) (the runtime under
pg_durable) is MIT (crates.io 0.1.32, 2026-10-12) and has the right model
(history replay, activities, timers, external events, sub-orchestrations,
pluggable `Provider`). We do not use it:

1. **The point is the shared transaction.** A duroxide provider persists
   history through its own calls; a step's kleos writes (`remember`, a query's
   `:put`) cannot join the transaction that records the step's completion. We
   would lose the one property that makes an in-database engine worth having.
2. **Runtime.** duroxide is async on Tokio; kleos-core is synchronous and has
   no Tokio. Embedding a Tokio runtime in every kleos process (Python wheels,
   C, Swift, WASM) is a large cost for a small engine.
3. **Language.** duroxide orchestrations are Rust closures. kleos's code mode
   must replay *Python* functions (and later other bindings), so the replay
   logic lives in the SDK and the database only stores history and hands out
   work; that protocol is ours either way.
4. **Maturity.** 0.1.x, "alpha" on docs.rs.

We borrow its ideas: correlation keys per command, deterministic join/select,
timers as history events, a non-determinism check.

## Model

A **workflow** is an instance of a named, versioned definition with an input.
It runs as a sequence of **commands**: steps (do work, record the result),
timers (sleep), signal waits, child workflows, markers. Every completed
command is an **event** in the workflow's append-only history, keyed by a
deterministic **command key**. After a crash, or when a sleeping workflow
wakes, the workflow function runs again from the top and *replays*: each
command whose key is already in the history returns the recorded result
without doing the work again; the first command without an event does real
work. So a workflow is resumed, not restarted.

Two modes share the store, the ops and the worker protocol:

| | Built-in mode (pg_durable style) | Code mode (Temporal style) |
|---|---|---|
| Definition | a JSON step graph registered with `::flow define` | a Python function registered with an SDK worker |
| Runs in | the kleos process: `Db::flow_tick` / an in-process worker thread | any process that opens the database (SDK worker) |
| Command keys | the step's path in the graph (`s2.then.s0`, `s3.i4.s1`, `s5.b1.s0`) | `<ordinal>:<name>` in call order (`3:fetch`) |
| Steps | actions run by the engine (query, remember, recall, embed, decide, http) | any Python callable, plus the engine actions |

### Command keys and the non-determinism check

A replay must issue the same commands in the same order. Each event stores
the command's `type` and `name` under its `key`. When a replay reaches key
`k` and the history has an event for `k` whose type or name differs, the run
stops with a **non-determinism error**: the workflow is not failed; it is
parked (`status: 'waiting'`, `error` set, retried with back-off), so deploying
the fixed (or versioned) code resumes it. When a run finishes or suspends
while recorded command keys were never reached (code removed a step), that is
also a non-determinism error.

**Versioning.** Each workflow records the definition `version` it started
with; workers may serve several versions of a name. Inside code,
`ctx.patched('id')` returns `True` on a new execution (and records a
`marker` event) and `False` when replaying a history that passed that point
without the marker, so old runs keep the old path (Temporal's `patched`).
`ctx.now()` and `ctx.uuid()` are markers too: recorded once, replayed after.

## Relations

All are ordinary stored relations, created on first use of `::flow` (and by
`::memory init {engram}`), visible to queries, `::history`-able where they
carry validity (they do not). Times are µs since the epoch.

```
kleos_flow_wf {id: String =>
    name: String, version: Int, mode: String,        # mode: 'builtin' | 'code'
    queue: String,
    status: String,     # pending | running | waiting | completed | failed | canceled
    input: Json?, output: Json?, error: String?,
    parent: String?,                                 # child workflows
    serial: String?, serial_seq: Int?,               # ordering lane (see Queues)
    token: Int,         # fencing token: +1 on every claim
    seq: Int,           # events recorded so far
    owner: String?, lease_until: Int?,               # the worker holding it
    wait: Json?,        # what a waiting workflow waits for
    cancel: Bool, attempts: Int,                     # claims so far
    created: Int, updated: Int, finished: Int?, meta: Json?}

kleos_flow_event {wf: String, seq: Int =>             # append-only history
    type: String, key: String?, name: String?, data: Json?, at: Int, token: Int}

kleos_flow_task {wf: String =>                        # runnable work
    queue: String, due: Int, serial: String?, serial_seq: Int?}

kleos_flow_timer {wf: String, key: String =>          # pending durable timers
    due: Int, kind: String}                          # sleep | signal_timeout | retry

kleos_flow_signal {wf: String, sid: String =>         # signal inbox
    name: String, payload: Json?, at: Int, consumed: Int?}

kleos_flow_def {name: String, version: Int =>         # built-in graphs
    mode: String, graph: Json?, created: Int}

kleos_flow_schedule {name: String =>                  # cron schedules
    workflow: String, version: Int?, cron: String, input: Json?, queue: String,
    overlap: String, next_at: Int, last_id: String?, enabled: Bool,
    created: Int, updated: Int}

kleos_flow_lane {serial: String, serial_seq: Int =>    # unfinished workflows of each lane
    wf: String}
```

**Event types.** Commands: `step` (data: result), `step_failed` (data:
attempt, error, final), `timer` (data: due), `timer_fired`, `signal` (name,
data: payload, sid), `signal_timeout`, `child` (data: child id), `child_done`
(data: output | error), `marker` (data: value). Lifecycle (not matched on
replay): `started`, `cancel_requested`, `completed`, `failed`, `canceled`,
`task_error` (non-determinism, worker exceptions).

## Queues, tasks, leases

A workflow is runnable while it has a `kleos_flow_task` row with
`due <= now`. **Claiming** is a compare-and-set in one write transaction:
the task is due, the workflow's `token` is the one read, and (for a lane, see
below) no earlier workflow of its lane is unfinished. The claim sets
`status: running`, `owner`, `lease_until = now + lease`, `token + 1`,
`attempts + 1`, and moves the task's `due` to `lease_until`. So a worker that
dies leaves a task that becomes due when its lease runs out, and the next
claim (with a new token) resumes the workflow from its history. Live workers
extend the lease with `heartbeat`.

**Fencing.** Every write a worker makes for a workflow (record an event,
suspend, complete, fail, heartbeat) asserts, inside its transaction, that
the workflow's token is still the one its claim produced. A worker that lost
its lease (paused, partitioned, slow) gets a `lease lost` error and its write
is rolled back, *including the step's data writes*. Two workers can therefore
never both record a step, and a step's data effects happen once.

**Lanes (`serial`).** Workflows started with the same `serial` key run one at
a time in `serial_seq` order: a task is claimable only when no unfinished
workflow of its lane has a smaller `serial_seq`. `kleos_flow_lane` holds a row
per *unfinished* workflow of a lane (written by `start`, removed when it
finishes), so the check reads only the lane's open work, not its whole past.
The Engram queue uses one lane per memory namespace, ordered by namespace
version. `claim {lanes: [...]}` restricts a worker to some lanes.

**Waking.** Starting, signalling, or a child finishing writes the task row
with `due = now`. Timers set `due` to the earliest pending timer. In one
process, workers sleep on a change feed of `kleos_flow_task`
(`register_callback`), so a start or signal wakes them at once; across
processes, workers poll (default every 1 s, `poll_ms`). A worker never sleeps
past the next due task or schedule.

## Atomicity: what happens exactly once

| Work | Guarantee |
|---|---|
| Engine actions `query`, `remember` | **exactly once**: the action's writes, the event and the fencing check commit in one transaction |
| Engine actions `recall`, `embed`, `decide` | no writes; the result is recorded once and replayed |
| Engine action `http` (feature `flow-http`, off by default) | **at least once**: the request may be repeated if the worker dies before the event commits; send an idempotency key (`ctx.uuid()`) |
| Python steps (`ctx.step(fn)`) | the result is recorded exactly once; the function's own side effects are **at least once** (it reruns if the worker dies before recording). Put writes to kleos in `ctx.query`/`ctx.remember`, or make the function idempotent |
| Timers, signals, child starts | exactly once (events + inbox/child rows in one transaction) |

`remember` joins the transaction through a hook in `memory.rs`: the commit
script of `remember` carries the flow's event and fencing blocks, so the
memory, its supersessions, its Engram jobs and the step checkpoint commit
together (or the namespace-version retry loop re-plans them together).

## Built-in mode: step graphs

```json
{"steps": [
  {"name": "batch", "query": {"script": "?[text] := *docs{text, emb}, is_null(emb) :limit 10"}},
  {"if": {"cond": "$batch.rows", "then": [], "else": [{"return": "null"}]}},
  {"name": "vecs", "embed": {"texts": "$batch"}},
  {"name": "parts", "fan_out": {"branches": [
      [{"query": {"script": "?[n] := *a{n}"}}],
      [{"query": {"script": "?[n] := *b{n}"}}]]}},
  {"name": "ok", "wait_signal": {"name": "approve", "timeout_ms": 86400000}},
  {"sleep": {"ms": 60000}},
  {"loop": {"while": "$i < 3", "body": [{"remember": {"text": "=concat('tick ', to_string($i))"}}]}},
  {"name": "sub", "child": {"workflow": "other", "input": {"x": "$batch"}}},
  {"return": "$parts"}
]}
```

- **Values.** Arguments are JSON. A string `"$name"` (or `"$name.field"`) is
  replaced by a named result, `$input`, or `$i` (the loop counter); a string
  starting with `=` is a CozoScript expression evaluated with every named
  result bound as a parameter (`?[v] := v = <expr>`). A query's result is
  `{headers, rows}` (`$q.rows.0.1` is a path into it; `embed` and `for_each`
  also take it directly: one-column rows are texts, rows are items).
  Everything else is a literal. Expression fields (`cond`, `while`, `set`,
  `return`) are CozoScript expressions; a `$name.path` there is a lookup.
- **Steps.** `query {script, params}` (rows), `remember {text, …options}`,
  `recall {query, …options}`, `embed {texts, embedder?}` (vectors),
  `decide {template, pairs, decider?}`, `http {url, method?, headers?, body?}`
  (feature `flow-http`), `sleep {ms}`, `wait_signal {name, timeout_ms?}`
  (payload, or null on timeout), `child {workflow, input?, version?}` (the
  child's output), `set {value}` (an expression), `return` (an expression;
  ends the workflow with that output).
- **Control flow.** `if {cond, then, else?}`; `loop {while, body, max?}`
  (default max 1000; `$i` counts from 0); `for_each {in, as, body}`;
  `fan_out {branches}` runs the branches in parallel threads and returns the
  list of each branch's last result (join = all). A branch that sleeps or
  waits suspends the whole workflow until the earliest of its branches can
  continue; on resume all branches replay.
- **Retries.** Any step may carry `retry {max_attempts, initial_ms,
  backoff, max_ms}` (default 3, 1000, 2.0, 60000) and `timeout_ms`.
  A failed attempt records `step_failed` and sets a `retry` timer, so
  back-off survives crashes. After the last attempt the workflow fails with
  the step's error, unless the step has `"on_error": "continue"` (result null).
- **Keys.** The step's path: `s<index>` in a list, then `.then`/`.else`,
  `.i<n>` per loop iteration, `.b<n>` per branch, `.e<n>` per item.

Definitions are versioned: `::flow define {name, graph}` stores the next
version; `start` uses the latest unless `version` is given; a running
workflow keeps the version it started with.

## Code mode: Python SDK

```python
from kleos import CozoDbPy
from kleos_flow import Worker, RetryPolicy, workflow

@workflow("onboard", version=1)
def onboard(ctx, user):
    facts = ctx.step(extract_facts, user, retry=RetryPolicy(max_attempts=5))
    ctx.remember(f"{user['name']} signed up", ns=user["id"])      # exactly once
    ctx.sleep(86400)                                              # durable timer
    ok = ctx.wait_signal("approved", timeout=3 * 86400)           # human in the loop
    if ctx.patched("send-welcome"):                               # versioning
        ctx.step(send_welcome, user)
    child = ctx.child("score", {"user": user["id"]})              # child workflow
    return {"facts": facts, "approved": ok, "score": child}

db = CozoDbPy("sqlite", "app.db", "{}")
db.flow_call("start", {"workflow": "onboard", "input": {...}, "id": "onboard:42"})
Worker(db, workflows=[onboard], queue="default").run()   # or .run_once()
```

`ctx.step(fn, *args, name=, retry=, timeout=)`, `ctx.query(script, params)`,
`ctx.remember(text, **opts)`, `ctx.recall(query, **opts)`,
`ctx.sleep(seconds)`, `ctx.wait_signal(name, timeout=)`,
`ctx.child(workflow, input, id=)`, `ctx.patched(id)`, `ctx.now()`,
`ctx.uuid()`, `ctx.cancelled` (raises `Cancelled` at the next command when a
cancel was requested). A timeout on `ctx.step` runs the function in a thread
and counts as a failed attempt when exceeded (Python cannot kill the
thread; it is abandoned).

The SDK's worker loop: `claim` → replay the function against the returned
history → for each new command call `step` / `suspend` / `complete` /
`fail`, heartbeating in the background while a step runs.

## Ops

One JSON entry point, `flow_call(op, args)`, behind every surface:
`::flow <op> {…}` in CozoScript, `Db::flow_call` / `DbInstance::flow_call`
in Rust, `CozoDbPy.flow_call` in Python, `POST /flow/:op` over HTTP.

Client ops:

| Op | Args | Result |
|---|---|---|
| `define` | `name, graph, version?` | `{name, version}` |
| `start` | `workflow, input?, id?, version?, queue?, serial?, serial_seq?, run_at?` | `{id, status, created}`; an existing `id` returns that workflow (`existing: true`), so starts are idempotent |
| `signal` | `id, name, payload?, sid?` | `{id, sid}`; a repeated `sid` is ignored |
| `cancel` | `id, reason?, children? (true)` | `{id, status}` |
| `status` | `id` | the workflow row, plus `wait`, `timers` |
| `list` | `status?, workflow?, queue?, parent?, limit? (100)` | rows |
| `history` | `id` | events in order |
| `schedule` | `name, cron, workflow, input?, queue?, overlap? ('skip'), enabled?` | the schedule with `next_at` |
| `unschedule` | `name` | `{name, removed}` |
| `schedules` | — | rows |
| `stats` | — | counts per queue and status, due work, workflows running past their lease |
| `adopt_engram` | — | `{adopted}` (see Engram) |

Worker ops (SDKs and the built-in executor):

| Op | Args | Result |
|---|---|---|
| `claim` | `worker, queues?, modes?, workflows?, lanes?, lease_ms? (30000)` | `null`, or `{id, name, version, mode, input, token, history, signals, cancel}` |
| `step` | `id, token, key, name, action?, result?` | the result: runs the engine `action` atomically with the event, or records the caller's `result`; an existing key returns its recorded result |
| `step_failed` | `id, token, key, name, error, attempt, retry_in_ms?` | records the failure; with `retry_in_ms`, sets a retry timer |
| `timer` | `id, token, key, due` | records the timer (idempotent) |
| `consume_signal` | `id, token, key, name` | the payload, or `null` |
| `start_child` | `id, token, key, workflow, input?, child_id?` | the child id (child + event in one transaction) |
| `marker` | `id, token, key, name, value` | the recorded value (first write wins) |
| `suspend` | `id, token, wait?` | parks the workflow until its earliest timer, a signal, or a child |
| `complete` | `id, token, output?` | finishes; wakes the parent |
| `fail` | `id, token, error, final? (true), retry_in_ms?` | finishes as failed, or (non-final) parks with back-off |
| `heartbeat` | `id, token, lease_ms?` | extends the lease |
| `tick` | `worker?, queues?, max?, lease_ms?, schedules?` | runs due built-in workflows and fires due schedules; `{ran, fired}` (`max: 0` only fires schedules) |
| `next_due` | `queues?` | `{due, now}`: the earliest due task or schedule, for a worker's sleep |

Errors are ordinary op errors; `lease lost` and `canceled` are recognisable
by message prefix.

## Schedules

Five-field cron (`min hour day-of-month month day-of-week`, with `*`, lists,
ranges and steps, UTC) or `@every <n>(s|m|h|d)`, `@hourly`, `@daily`,
`@weekly`. `tick` (and every worker loop) fires each schedule whose
`next_at <= now`: it starts the workflow with id `sched:<name>:<next_at>`
(so two workers firing the same schedule start it once) and advances
`next_at`, both in one transaction. `overlap: 'skip'` (default) does not
start while the previous run is unfinished; `'allow'` starts anyway. A
schedule that missed several firings (the database was closed) fires once,
then continues from now.

## Concurrency

Every worker op is one write transaction with an assert on the workflow's
token, and claims assert the task's due time and the token they read. On
SQLite the connection's busy timeout (5 s, `storage/sqlite.rs`) serialises
writers across processes; a transaction that loses (busy, conflict) is
retried with back-off, like `remember`'s namespace-version loop. Any number
of worker processes may share one SQLite file; the claim CAS hands each
runnable workflow to one of them, and the token fences the rest.

## Engram: improved, not replaced

The Engram overlay is unchanged: kleos-engram addressing, the overlay store,
the EngramEdit-style edit computation and the provenance-replay erase are
the same code. Only the job plumbing moves onto workflows:

- `remember` / `forget` still write the job row in `kleos_engram_queue` in
  their own transaction (the job's record and status, `::engram queue`), and
  now also start a workflow `engram.apply` in that same transaction: id
  `engram:<ns>:<qid>`, queue `engram`, lane `engram:<ns>` ordered by `qid`
  (the namespace version). Lanes replace the worker's own head-of-namespace
  scan; the claim CAS and fencing replace its status-and-`updated` CAS.
- `scripts/engram/engram_worker.py` becomes a code-mode worker: the workflow
  is `apply` (compute and write the edit; idempotent, as before) then mark the
  queue row done (exactly once, an engine `query` step). What it gains:
  retries with back-off that survive crashes (`retry` timers), a recorded
  history per job (`::flow history`), status and failures in `::flow list`,
  several worker processes, wake-up on new jobs, and leases with heartbeats
  instead of one long fixed lease.
- **Migration.** `::flow adopt_engram {}` starts a workflow for every queue
  row that is pending or running and has none (idempotent ids); the new
  worker calls it at start, so a database with jobs queued by the old code
  carries on in the same order.

## Observability

- `::flow list` / `status` / `history`, and the relations themselves:
  everything is queryable with CozoScript, including joins against the data
  the workflows wrote.
- Every event carries the claim `token` that wrote it, so retries and
  takeovers are visible in the history.
- `kleos_flow_task` is a change feed: subscribe to see work arrive.
- `::flow stats {}`: counts per status and queue, oldest due task, running
  workflows past their lease.

## Limits (this version)

- One database; no cross-database or cross-node workers beyond what SQLite
  file sharing (or RocksDB in one process) allows.
- The history grows with the workflow; long-running loops should use
  `continue_as_new` (planned: start a fresh run with new input, same id
  family) — not in this version.
- Python steps' timeouts cannot stop the function's thread.
- `http` is off by default (feature `flow-http`) and runs from the worker's
  process with no allow-list beyond the feature gate.

## Results (2026-10-24, branch `wave4/flow`)

Measured on the shared 16 GB Mac while other agents' benchmarks held the load
average between 6 and 60, so timings are indicative.

**Tests** (all pass):
- `kleos-core/tests/flow.rs`, 12 tests plus 2 child-process helpers:
  - a built-in graph using every control-flow kind (keys by path, loop rows written once, fan-out joined);
  - retries with back-off timers, exhaustion, `on_error: continue`;
  - timers;
  - signals: waited for, buffered before the wait, idempotent by `sid`, timed out;
  - children, and cancel cascading to them;
  - the code-mode protocol: claim, lease expiry and takeover, a stale worker's query rolled back (`lease lost`), replay, non-determinism, markers, cancel while running;
  - suspend waking on a signal already in the inbox;
  - a `remember` step committed with its checkpoint, and a stale one leaving no memory;
  - schedules firing once and advancing;
  - the in-process worker waking on a start through the change feed;
  - **crash mid-step**: a child process records step 1, is killed inside step 2, and a new worker resumes after the lease; step 1's effect exists once, written by the dead worker;
  - **two worker processes on one SQLite file**: 24 workflows with 4 query steps each (two of them in a fan-out), every step's effect is written exactly once, and the work splits between the processes (20/20 at 40 workflows).
- `flow_cron` unit tests; `tests/memory.rs` (28, `remember_with` refactor); `tests/memory_engram.rs --features engram` (8 existing + 1 new: jobs are workflows in namespace lanes, one claimable per lane, `adopt_engram` idempotent, a failed memory write starts no workflow).
- Python: `integrations/kleos-flow/tests/test_flow.py`, 7 tests: replay runs each step once, retries then catch `StepFailed`, signals plus children plus `ctx.query`, child failure, `patched` with an old run (skips the patched step, takes the later patch live) and a fresh run, non-determinism parks instead of failing, cancel and a missing version, two worker threads sharing a queue. `tests/test_memory_api.py` still passes.

**End to end:**
- `scripts/flow/demo.py`, a code-mode agent workflow: remember, a slow step, sleep 2 s, wait for a signal, recall. Worker process 1 is `kill -9`ed inside the slow step; worker process 2 takes over when the 2 s lease runs out; the signal is sent from the outside. Result: one memory, the `remember` recorded once (by worker 1's token), the slow step run twice (at least once, as specified), 4 claims, output with the signal payload and the recall hit, about 11 s in total. The same script runs a built-in graph (query → fan-out of two `:update` queries → sleep → sum) on the in-process worker thread: 0.5 s.
- `scripts/engram/remember_e2e.py` (132M Engram model, worker as a separate process) on the new engine: every result equals the run before the change: recall 9/10 own prompt, 9/18 rephrasings; supersede ranks (" 1934" 28 → 1, " 1921" 1 → 54); `as_of` bit-identical; erase-one and erase-all bit-identical; crash after a write (exit 17) and kill -9 mid-job: the same job states and attempts as before (jobs 1 and 2 done after 2 attempts, job 3 after 1), one overlay version per fact.
- HTTP: `kleos server --flow-worker default`, then `POST /flow/define`, `/flow/start`: the server's worker ran the graph, slept, woke on its timer, completed.

**Cost of the Engram workflow rows.** `remember` on an attached store writes four more rows (workflow, started event, task, lane). Interleaved runs of the old and new wheel (SQLite, 70 remembers each, 4 rounds) gave attached medians of 21.4 / 21.4 / 13.9 / 79.7 ms (old) and 25.0 / 20.7 / 19.8 / 12.9 ms (new): no difference beyond this machine's noise.

**Findings along the way:**
- Constant-row scripts. The flow scripts pass precomputed constant rows (`<- $rows`, or a constant rule joined to the computed columns). An early measurement blamed the planner for 16–28 ms on a 23-column row written as `column = <constant>` atoms; re-measured on wave5/engine, planning such a rule takes about 0.2 ms, and the spikes were evaluation latency on a machine at load 40–160. The constant-row form is kept (it is slightly cheaper and clearer). A real bug found at the same time — a literal `true` followed by whitespace or a comment parsed as `false` — is fixed in the grammar (`boolean` and `null` are atomic).
- `?[k, c] := k = 1, c = true :update t {k => c}` silently writes nothing, while `c = to_bool(1)` or a constant rule `<- [[1, true]]` works. This is an engine quirk, not fixed here. The flow code uses constant rules.
- A multi-statement transaction (`run_multi_transaction`) takes one program per payload and stays open after a failed query. The `query` step therefore sends its blocks one by one and drops the payload channel on any early return, which ends and rolls back the transaction.
