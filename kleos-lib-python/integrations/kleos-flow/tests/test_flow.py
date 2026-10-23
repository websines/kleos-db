# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Code-mode durable workflows (kleos_flow) against an in-memory kleos."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pytest  # noqa: E402
from kleos import CozoDbPy  # noqa: E402

from kleos_flow import ChildFailed, RetryPolicy, StepFailed, Worker, start, workflow  # noqa: E402

CALLS = {}


def count(name):
    CALLS[name] = CALLS.get(name, 0) + 1
    return CALLS[name]


@pytest.fixture
def db():
    CALLS.clear()
    return CozoDbPy("mem", "", "{}")


def run_until(w, db, wid, status="completed", limit=10):
    t = time.time()
    while db.flow_call("status", {"id": wid})["status"] != status:
        w.drain()
        if time.time() - t > limit:
            raise AssertionError(db.flow_call("status", {"id": wid}))
        time.sleep(0.01)
    return db.flow_call("status", {"id": wid})


def test_replay_runs_each_step_once(db):
    @workflow("wf")
    def wf(ctx, x):
        a = ctx.step(lambda: count("a") and x + 1, name="a")
        ctx.sleep(0.05)
        b = ctx.step(lambda: count("b") and a * 10, name="b")
        return {"a": a, "b": b, "now": ctx.now() > 0, "id": len(ctx.uuid())}

    start(db, "wf", 4, id="w1")
    w = Worker(db, [wf])
    assert w.run_once()["outcome"] == "waiting"
    st = run_until(w, db, "w1")
    assert st["output"] == {"a": 5, "b": 50, "now": True, "id": 36}
    assert CALLS == {"a": 1, "b": 1}
    types = [e["type"] for e in db.flow_call("history", {"id": "w1"})]
    assert types[:3] == ["started", "step", "timer"] and types[-1] == "completed"


def test_retries_then_fail_and_catch(db):
    @workflow("flaky")
    def flaky(ctx, _):
        def f():
            if count("f") < 3:
                raise RuntimeError("not yet")
            return "ok"
        return ctx.step(f, retry=RetryPolicy(max_attempts=3, initial=0.01))

    @workflow("doomed")
    def doomed(ctx, _):
        try:
            ctx.step(lambda: 1 / 0, name="div", retry=RetryPolicy(max_attempts=2, initial=0.01))
        except StepFailed as e:
            return {"caught": "ZeroDivisionError" in str(e)}

    start(db, "flaky", None, id="f1")
    start(db, "doomed", None, id="d1")
    w = Worker(db, [flaky, doomed])
    assert run_until(w, db, "f1")["output"] == "ok"
    assert CALLS["f"] == 3
    assert run_until(w, db, "d1")["output"] == {"caught": True}
    fails = [e for e in db.flow_call("history", {"id": "f1"}) if e["type"] == "step_failed"]
    assert [e["data"]["attempt"] for e in fails] == [1, 2]


def test_signals_children_and_kleos_steps(db):
    db.run_script(":create audit {k: String => v: Int}", {}, False)

    @workflow("child")
    def child(ctx, n):
        return n * 2

    @workflow("parent")
    def parent(ctx, inp):
        ctx.query("?[k, v] <- [[$k, $v]] :put audit {k => v}", {"k": "start", "v": inp})
        got = ctx.wait_signal("go", timeout=30)
        c = ctx.child("child", got["n"])
        late = ctx.wait_signal("never", timeout=0.05)
        return {"child": c, "late": late}

    start(db, "parent", 1, id="p1")
    w = Worker(db, [parent, child])
    w.drain()
    assert db.flow_call("status", {"id": "p1"})["status"] == "waiting"
    db.flow_call("signal", {"id": "p1", "name": "go", "payload": {"n": 21}})
    st = run_until(w, db, "p1")
    assert st["output"] == {"child": 42, "late": None}
    assert db.run_script("?[k, v] := *audit{k, v}", {}, True)["rows"] == [["start", 1]]


def test_child_failure_surfaces(db):
    @workflow("bad")
    def bad(ctx, _):
        raise ValueError("nope")

    @workflow("p")
    def p(ctx, _):
        try:
            ctx.child("bad")
        except ChildFailed as e:
            return str(e)

    start(db, "p", None, id="p")
    w = Worker(db, [p, bad])
    out = run_until(w, db, "p")["output"]
    assert "failed" in out and "nope" in out


def test_patched_and_nondeterminism(db):
    @workflow("v", version=1)
    def v1(ctx, _):
        ctx.step(lambda: "one", name="one")
        ctx.wait_signal("go")
        return "old"

    start(db, "v", None, id="old")
    Worker(db, [v1]).drain()

    # new code adds a step behind a patch: the running workflow keeps the old path
    @workflow("v", version=1)
    def v1_patched(ctx, _):
        ctx.step(lambda: "one", name="one")
        if ctx.patched("extra"):
            ctx.step(lambda: "extra", name="extra")
        ctx.wait_signal("go")
        return "new" if ctx.patched("ret") else "old"

    db.flow_call("signal", {"id": "old", "name": "go"})
    w = Worker(db, [v1_patched])
    # the old run had passed "extra" (skipped), but reaches "ret" live (taken)
    assert run_until(w, db, "old")["output"] == "new"

    def steps(i):
        return [e["name"] for e in db.flow_call("history", {"id": i}) if e["type"] == "step"]

    assert steps("old") == ["one"]
    start(db, "v", None, id="fresh")
    w.drain()
    db.flow_call("signal", {"id": "fresh", "name": "go"})
    assert run_until(w, db, "fresh")["output"] == "new"
    assert steps("fresh") == ["one", "extra"]

    # unpatched change: parked with a non-determinism error, not failed
    @workflow("v", version=1)
    def v1_broken(ctx, _):
        ctx.step(lambda: "renamed", name="renamed")
        ctx.wait_signal("go")
        return "x"

    start(db, "v", None, id="nd")
    Worker(db, [v1]).drain()
    db.flow_call("signal", {"id": "nd", "name": "go"})
    rec = Worker(db, [v1_broken]).run_once()
    assert rec["outcome"] == "parked"
    st = db.flow_call("status", {"id": "nd"})
    assert st["status"] == "waiting" and "non-determinism" in st["error"]


def test_cancel_and_versions(db):
    @workflow("long")
    def long(ctx, _):
        ctx.sleep(3600)

    start(db, "long", None, id="l1")
    w = Worker(db, [long])
    w.drain()
    assert db.flow_call("cancel", {"id": "l1"})["status"] == "canceled"
    # no worker for version 2: parked for later, not failed
    start(db, "long", None, id="l2", version=2)
    assert w.run_once()["outcome"] == "no_version"
    assert db.flow_call("status", {"id": "l2"})["status"] == "waiting"


def test_two_workers_share_the_queue(db):
    @workflow("job")
    def job(ctx, i):
        return ctx.step(lambda: count(f"j{i}") and i, name="work")

    for i in range(20):
        start(db, "job", i, id=f"j{i}")
    import threading

    ws = [Worker(db, [job], name=f"w{k}") for k in range(2)]
    ts = [threading.Thread(target=w.drain) for w in ws]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    done = db.flow_call("list", {"status": "completed", "limit": 100})
    assert len(done) == 20
    assert all(v == 1 for v in CALLS.values()) and len(CALLS) == 20
