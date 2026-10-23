# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""End to end: a durable agent workflow in kleos, with the worker killed mid-step.

The workflow (code mode, Python):
    remember a fact (exactly once) -> a slow enrichment step -> sleep 2 s ->
    wait for the 'approved' signal -> recall -> return
Steps:
  1. start the workflow; run a worker as a separate process;
  2. kill -9 the worker while it is inside the slow step (after the remember committed);
  3. start a second worker process: it takes over when the dead worker's lease runs out,
     replays the remember (no second memory), re-runs the slow step, sleeps;
  4. send the signal from this process; the worker wakes and finishes;
  5. check: one memory, the remember recorded once, two claims, the output, the history.
Also runs a built-in (pg_durable style) step graph with the in-process worker thread.

Usage:  python -I demo.py --out <dir>      (needs the kleos wheel; kleos_flow is found next door)
        python -I demo.py worker --db <path> --log <path>   (the worker process)
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "kleos-lib-python", "integrations", "kleos-flow"))
from kleos_flow import RetryPolicy, Worker, start, workflow  # noqa: E402

NS = "user-ada"


def slow_enrich(text, marker):
    """Pretend to call a slow API. Leaves a file so the demo knows the step is running."""
    with open(marker, "a") as f:
        f.write(f"{os.getpid()}\n")
    time.sleep(6)
    return {"topic": "travel", "len": len(text)}


@workflow("agent.onboard", version=1)
def onboard(ctx, inp):
    fact = ctx.remember(inp["fact"], ns=NS, source="onboarding")
    enrich = ctx.step(slow_enrich, inp["fact"], inp["marker"], retry=RetryPolicy(max_attempts=3, initial=0.5))
    ctx.sleep(2)
    approved = ctx.wait_signal("approved", timeout=120)
    hits = ctx.recall("where does Ada live?", ns=NS, k=3)
    return {"memory": fact["id"], "enrich": enrich, "approved": approved,
            "recall": [h["text"] for h in hits]}


def worker_main(args):
    from kleos import CozoDbPy

    db = CozoDbPy("sqlite", args.db, "{}")
    logf = open(args.log, "a")

    def log(rec):
        logf.write(json.dumps({k: v for k, v in rec.items() if k != "trace"}) + "\n")
        logf.flush()

    w = Worker(db, [onboard], name=f"worker-{os.getpid()}", lease_ms=2000, poll=0.2, log=log)
    w.run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="demo")
    ap.add_argument("--out")
    ap.add_argument("--db")
    ap.add_argument("--log")
    args = ap.parse_args()
    if args.mode == "worker":
        return worker_main(args)
    from kleos import CozoDbPy

    os.makedirs(args.out, exist_ok=True)
    dbp = os.path.join(args.out, "flow-demo.db")
    for p in (dbp, dbp + "-wal", dbp + "-shm"):
        if os.path.exists(p):
            os.remove(p)
    marker = os.path.join(args.out, "enrich.marker")
    logp = os.path.join(args.out, "worker.log")
    for p in (marker, logp):
        if os.path.exists(p):
            os.remove(p)
    db = CozoDbPy("sqlite", dbp, "{}")
    res = {}
    t0 = time.time()

    def worker():
        return subprocess.Popen([sys.executable, "-I", os.path.abspath(__file__), "worker", "--db", dbp, "--log",
                                 logp])

    def wait(pred, what, limit=60):
        t = time.time()
        while not pred():
            if time.time() - t > limit:
                raise RuntimeError(f"timed out waiting for {what}")
            time.sleep(0.1)

    def memories():
        return db.run_script("?[id, text] := *kleos_mem{ns: $ns, id, text}", {"ns": NS}, True)["rows"]

    # 1. start, run worker 1
    s = start(db, "agent.onboard", {"fact": "Ada lives in Lisbon.", "marker": marker}, id="onboard:ada")
    res["started"] = s
    w1 = worker()
    # 2. kill it inside the slow step (the remember has committed by then)
    wait(lambda: os.path.exists(marker), "the slow step to start")
    res["memories_before_kill"] = len(memories())
    time.sleep(1.0)
    w1.send_signal(signal.SIGKILL)
    w1.wait()
    res["killed_worker"] = w1.pid
    res["status_after_kill"] = db.flow_call("status", {"id": "onboard:ada"})["status"]
    # 3. worker 2 takes over after the lease (2 s) runs out
    w2 = worker()
    wait(lambda: db.flow_call("status", {"id": "onboard:ada"})["status"] == "waiting"
         and any(e["type"] == "wait" for e in db.flow_call("history", {"id": "onboard:ada"})),
         "the workflow to wait for the signal", limit=90)
    res["status_before_signal"] = db.flow_call("status", {"id": "onboard:ada"})["wait"]
    # 4. approve
    db.flow_call("signal", {"id": "onboard:ada", "name": "approved", "payload": {"by": "ops", "ok": True}})
    wait(lambda: db.flow_call("status", {"id": "onboard:ada"})["status"] == "completed", "completion", limit=60)
    w2.send_signal(signal.SIGTERM)
    w2.wait()
    # 5. check
    st = db.flow_call("status", {"id": "onboard:ada"})
    hist = db.flow_call("history", {"id": "onboard:ada"})
    mems = memories()
    runs = open(marker).read().split()
    res["output"] = st["output"]
    res["claims"] = st["attempts"]
    res["memories"] = [r[1] for r in mems]
    res["slow_step_runs"] = len(runs)
    res["history"] = [(e["seq"], e["type"], e.get("key"), e["token"]) for e in hist]
    res["remember_recorded"] = sum(1 for e in hist if e["type"] == "step" and e.get("name") == "remember")
    res["seconds"] = round(time.time() - t0, 1)
    res["worker_log"] = [json.loads(x) for x in open(logp)]
    res["passed"] = (len(mems) == 1 and res["remember_recorded"] == 1 and st["status"] == "completed"
                     and st["attempts"] >= 2 and st["output"]["approved"] == {"by": "ops", "ok": True}
                     and "Ada lives in Lisbon." in st["output"]["recall"][0] and len(runs) == 2)

    # built-in mode: a step graph run by the in-process worker thread
    db.run_script(":create docs {id: Int => text: String, words: Int?}", {}, False)
    db.run_script("?[id, text, words] <- [[1, 'kleos keeps agent memory', null], [2, 'durable workflows', null],"
                  " [3, 'in the database', null]] :put docs {id => text, words}", {}, False)
    graph = {"steps": [
        {"name": "todo", "query": {"script": "?[id, text] := *docs{id, text, words}, is_null(words)"}},
        {"name": "counts", "fan_out": {"branches": [
            [{"query": {"script": "?[id, words] := *docs{id, text}, id == 1, words = length(regex_extract(text, '[^ ]+')) "
                                  ":update docs {id => words}"}}],
            [{"query": {"script": "?[id, words] := *docs{id, text}, id > 1, words = length(regex_extract(text, '[^ ]+')) "
                                  ":update docs {id => words}"}}]]}},
        {"sleep": {"ms": 300}},
        {"name": "total", "query": {"script": "?[sum(w)] := *docs{words: w}"}},
        {"return": "$total.rows.0.0"}]}
    db.flow_call("define", {"name": "count_words", "graph": graph})
    wk = db.flow_worker(poll_ms=1000)
    t1 = time.time()
    db.flow_call("start", {"workflow": "count_words", "id": "cw1"})
    wait(lambda: db.flow_call("status", {"id": "cw1"})["status"] == "completed", "the built-in workflow", 30)
    wk.stop()
    bst = db.flow_call("status", {"id": "cw1"})
    res["builtin"] = {"output": bst["output"], "seconds": round(time.time() - t1, 2),
                      "events": [e["type"] for e in db.flow_call("history", {"id": "cw1"})]}
    res["builtin"]["passed"] = bst["output"] == 9
    json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "worker_log"}, indent=1))
    sys.exit(0 if res["passed"] and res["builtin"]["passed"] else 1)


if __name__ == "__main__":
    main()
