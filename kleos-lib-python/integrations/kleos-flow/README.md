# kleos-flow

Durable workflows for [kleos](https://github.com/websines/kleos-db), code mode: write a workflow
as a Python function; kleos stores its history, and a worker replays it after a crash, a sleep or
a wait. Spec: `docs/specs/durable-workflows.md`.

```python
from kleos import CozoDbPy
from kleos_flow import RetryPolicy, Worker, start, workflow

@workflow("onboard")
def onboard(ctx, user):
    facts = ctx.step(extract_facts, user, retry=RetryPolicy(max_attempts=5))  # recorded once
    ctx.remember(f"{user['name']} signed up", ns=user["id"])                 # exactly once
    ctx.sleep(24 * 3600)                                                      # durable timer
    ok = ctx.wait_signal("approved", timeout=3 * 24 * 3600)                   # human in the loop
    return {"facts": facts, "approved": ok}

db = CozoDbPy("sqlite", "app.db", "{}")
start(db, "onboard", {"id": "u1", "name": "Ada"}, id="onboard:u1")
Worker(db, [onboard]).run()          # any number of workers, threads or processes
db.flow_call("signal", {"id": "onboard:u1", "name": "approved", "payload": True})
```

Commands: `step`, `query`, `remember`, `recall`, `embed`, `decide`, `http`, `sleep`,
`wait_signal`, `child`, `patched`, `now`, `uuid`. `query` and `remember` commit their kleos writes
in the same transaction as the step's checkpoint, so they happen exactly once; a Python `step`'s
own side effects are at least once (its result is recorded once).

Workflow code must be deterministic: the same commands in the same order on every replay. A
change that breaks that parks the workflow with a non-determinism error until fixed or versioned
code (`@workflow(name, version=2)`, or `ctx.patched("id")`) is deployed.
