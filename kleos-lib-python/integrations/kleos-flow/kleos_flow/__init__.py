# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Durable workflows for kleos, code mode (docs/specs/durable-workflows.md).

Write a workflow as an ordinary function of a context and an input. Each `ctx` call is a command
with a key (`<ordinal>:<name>`); its result is recorded in the workflow's history in kleos. When
the worker dies, or the workflow sleeps or waits, the function runs again from the top and every
recorded command returns its recorded result without running again, so the workflow resumes
where it stopped::

    from kleos import CozoDbPy
    from kleos_flow import RetryPolicy, Worker, workflow

    @workflow("onboard")
    def onboard(ctx, user):
        facts = ctx.step(extract_facts, user, retry=RetryPolicy(max_attempts=5))
        ctx.remember(f"{user['name']} signed up", ns=user["id"])     # exactly once
        ctx.sleep(24 * 3600)                                          # a durable timer
        ok = ctx.wait_signal("approved", timeout=3 * 24 * 3600)       # human in the loop
        return {"facts": facts, "approved": ok}

    db = CozoDbPy("sqlite", "app.db", "{}")
    start(db, "onboard", {"id": "u1", "name": "Ada"}, id="onboard:u1")
    Worker(db, [onboard]).run()

Rules for workflow code (as in Temporal): it must take the same path on every replay. Put
anything that touches the world in `ctx.step` (a Python function; its result is recorded, its
side effects are at least once), `ctx.query` / `ctx.remember` (kleos writes that commit together
with their checkpoint: exactly once), or `ctx.http`. Use `ctx.now()` and `ctx.uuid()` instead of
the clock and random ids, and `ctx.patched(id)` to change code that running workflows have
passed.
"""

import json
import threading
import time
import traceback
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutTimeout

__all__ = [
    "Cancelled",
    "ChildFailed",
    "Context",
    "LeaseLost",
    "NonDeterminism",
    "RetryPolicy",
    "StepFailed",
    "Worker",
    "start",
    "workflow",
]


def now_us():
    return int(time.time() * 1_000_000)


class Cancelled(Exception):
    """The workflow was canceled; raised at its next command."""


class LeaseLost(Exception):
    """Another worker took the workflow over; this run must stop."""


class NonDeterminism(Exception):
    """The code issued different commands than its recorded history."""


class StepFailed(Exception):
    """A step failed on its last attempt."""


class ChildFailed(Exception):
    """A child workflow failed or was canceled."""


class _Suspend(BaseException):
    """Unwinds the workflow function to park it (not an Exception, so user code's
    `except Exception` does not swallow it)."""

    def __init__(self, timers=(), signals=(), children=()):
        super().__init__("suspend")
        self.timers = list(timers)
        self.signals = list(signals)
        self.children = list(children)


class RetryPolicy:
    """Attempts of a step and the back-off between them (seconds)."""

    def __init__(self, max_attempts=3, initial=1.0, backoff=2.0, max_interval=60.0):
        self.max_attempts = max(1, int(max_attempts))
        self.initial = max(0.0, float(initial))
        self.backoff = max(1.0, float(backoff))
        self.max_interval = max(0.0, float(max_interval))

    def delay(self, attempt):
        return min(self.initial * self.backoff ** (attempt - 1), self.max_interval)


def workflow(name, version=1):
    """Mark `fn(ctx, input)` as workflow `name` (version `version`)."""

    def deco(fn):
        fn._kleos_flow = (name, int(version))
        return fn

    return deco


def start(db, workflow_name, input=None, id=None, queue=None, version=None, run_at=None, serial=None,
          serial_seq=None):
    """Start a code-mode workflow (idempotent on `id`). Returns `{id, status, ...}`."""
    args = {"workflow": workflow_name, "input": input, "mode": "code"}
    for k, v in (("id", id), ("queue", queue), ("version", version), ("run_at", run_at), ("serial", serial),
                 ("serial_seq", serial_seq)):
        if v is not None:
            args[k] = v
    return db.flow_call("start", args)


def _map_error(e):
    m = str(e)
    if "lease lost" in m:
        return LeaseLost(m)
    if m.startswith("canceled") or "canceled:" in m:
        return Cancelled(m)
    if "non-determinism" in m:
        return NonDeterminism(m)
    return None


def _jsonable(v):
    """The JSON form a result is recorded (and replayed) as."""
    return json.loads(json.dumps(v, default=str))


class Context:
    """What a workflow function talks to. One per run (claim)."""

    def __init__(self, worker, claimed):
        self._w = worker
        self._db = worker.db
        self.id = claimed["id"]
        self.workflow = claimed["name"]
        self.version = claimed["version"]
        self.input = claimed.get("input")
        self.token = claimed["token"]
        self._hist = {}
        for e in claimed.get("history") or []:
            if e.get("key"):
                self._hist.setdefault(e["key"], []).append(e)
        self._n = 0
        self._visited = set()
        self.cancel_requested = bool(claimed.get("cancel"))

    # ---- plumbing ---------------------------------------------------------------------------
    def _call(self, op, **args):
        args.update(id=self.id, token=self.token)
        try:
            return self._db.flow_call(op, args)
        except Exception as e:  # noqa: BLE001
            mapped = _map_error(e)
            if mapped is not None:
                raise mapped from None
            raise

    def _key(self, name):
        self._n += 1
        key = f"{self._n}:{name}"
        self._visited.add(key)
        return key

    def _events(self, key, name, types):
        evs = self._hist.get(key, [])
        for e in evs:
            if e["type"] not in types or (name is not None and e.get("name") != name):
                raise NonDeterminism(
                    f"non-determinism: workflow '{self.id}' key '{key}' was a '{e['type']}' named "
                    f"'{e.get('name')}' in the history, now '{name}'")
        return evs

    def _record(self, ty, key, name, data=None):
        return self._call("record", events=[{"type": ty, "key": key, "name": name, "data": data}])["data"][0]

    def _check_cancel(self):
        if self.cancel_requested:
            raise Cancelled(f"canceled: workflow '{self.id}' was canceled")

    def _finish_check(self):
        """Every recorded command must have been reached."""
        cmd = {"step", "step_failed", "timer", "timer_fired", "wait", "signal", "signal_timeout", "child",
               "child_done", "marker"}
        for key, evs in self._hist.items():
            if key.startswith("patch:"):
                continue
            if key not in self._visited and any(e["type"] in cmd for e in evs):
                raise NonDeterminism(f"non-determinism: workflow '{self.id}' finished without reaching "
                                     f"recorded command '{key}'")

    def _retry_or_fail(self, key, name, fails, retry, err):
        attempt = len(fails) + 1
        if attempt < retry.max_attempts:
            retry_at = now_us() + int(retry.delay(attempt) * 1_000_000)
            self._record("step_failed", key, name,
                         {"attempt": attempt, "error": err, "retry_at": retry_at, "final": False})
            raise _Suspend(timers=[(key + "#retry", retry_at, "retry")])
        self._record("step_failed", key, name, {"attempt": attempt, "error": err, "final": True})
        raise StepFailed(f"step '{key}' failed after {attempt} attempt(s): {err}")

    def _replay_step(self, key, name):
        evs = self._events(key, name, ("step", "step_failed"))
        for e in evs:
            if e["type"] == "step":
                return True, e.get("data"), []
        fails = [e for e in evs if e["type"] == "step_failed"]
        if fails:
            last = fails[-1]["data"] or {}
            if last.get("final"):
                raise StepFailed(f"step '{key}' failed: {last.get('error')}")
            ra = last.get("retry_at")
            if ra and ra > now_us():
                raise _Suspend(timers=[(key + "#retry", ra, "retry")])
        return False, None, fails

    # ---- commands ---------------------------------------------------------------------------
    def step(self, fn, *args, name=None, retry=None, timeout=None, **kwargs):
        """Run `fn(*args, **kwargs)` once and record its (JSON) result; replays return the record.
        `retry`: a RetryPolicy (default 3 attempts, 1 s doubling); `timeout`: seconds per attempt
        (the function's thread cannot be stopped, it is abandoned)."""
        name = name or getattr(fn, "__name__", "step")
        key = self._key(name)
        done, val, fails = self._replay_step(key, name)
        if done:
            return val
        self._check_cancel()
        retry = retry or RetryPolicy()
        try:
            if timeout is None:
                result = fn(*args, **kwargs)
            else:
                ex = ThreadPoolExecutor(max_workers=1)
                try:
                    result = ex.submit(fn, *args, **kwargs).result(timeout=timeout)
                finally:
                    ex.shutdown(wait=False)
        except _FutTimeout:
            self._retry_or_fail(key, name, fails, retry, f"timed out after {timeout} s")
        except (LeaseLost, Cancelled, NonDeterminism):
            raise
        except Exception as e:  # noqa: BLE001
            self._retry_or_fail(key, name, fails, retry, f"{type(e).__name__}: {e}")
        return self._call("step", key=key, name=name, result=_jsonable(result))["result"]

    def _action(self, kind, args, name=None, retry=None):
        name = name or kind
        key = self._key(name)
        done, val, fails = self._replay_step(key, name)
        if done:
            return val
        self._check_cancel()
        try:
            return self._call("step", key=key, name=name, action={kind: args})["result"]
        except (LeaseLost, Cancelled, NonDeterminism):
            raise
        except Exception as e:  # noqa: BLE001
            self._retry_or_fail(key, name, fails, retry or RetryPolicy(max_attempts=1), str(e))

    def query(self, script, params=None, name="query", retry=None):
        """Run a CozoScript query; its writes commit with the step's checkpoint (exactly once).
        Returns `{headers, rows}`."""
        return self._action("query", {"script": script, "params": params or {}}, name, retry)

    def remember(self, text, name="remember", retry=None, **opts):
        """`remember` through the memory API, committed with the checkpoint (exactly once)."""
        return self._action("remember", dict(opts, text=text), name, retry)

    def recall(self, query, name="recall", retry=None, **opts):
        """`recall` (recorded, so replays see what the first run saw)."""
        return self._action("recall", dict(opts, query=query), name, retry)

    def embed(self, texts, embedder=None, name="embed", retry=None):
        args = {"texts": list(texts)}
        if embedder:
            args["embedder"] = embedder
        return self._action("embed", args, name, retry)

    def decide(self, template, pairs, decider=None, name="decide", retry=None):
        args = {"template": template, "pairs": [list(p) for p in pairs]}
        if decider:
            args["decider"] = decider
        return self._action("decide", args, name, retry)

    def http(self, url, method="GET", headers=None, body=None, name="http", retry=None):
        """An HTTP request made by kleos (feature `flow-http`); at least once."""
        args = {"url": url, "method": method}
        if headers:
            args["headers"] = headers
        if body is not None:
            args["body"] = body
        return self._action("http", args, name, retry)

    def sleep(self, seconds, name="sleep"):
        """A durable timer: the workflow parks and resumes after `seconds`, even across restarts."""
        key = self._key(name)
        evs = self._events(key, name, ("timer", "timer_fired"))
        if any(e["type"] == "timer_fired" for e in evs):
            return
        t = next((e for e in evs if e["type"] == "timer"), None)
        if t is None:
            self._check_cancel()
            due = self._record("timer", key, name, {"due": now_us() + int(seconds * 1_000_000)})["due"]
        else:
            due = t["data"]["due"]
        if now_us() >= due:
            self._record("timer_fired", key, name)
            return
        raise _Suspend(timers=[(key, due, "sleep")])

    def wait_signal(self, signal, timeout=None):
        """The payload of the next signal `signal` sent to this workflow (signals sent earlier are
        buffered); `None` after `timeout` seconds."""
        key = self._key(signal)
        evs = self._events(key, signal, ("wait", "signal", "signal_timeout"))
        for e in evs:
            if e["type"] == "signal":
                return (e.get("data") or {}).get("payload")
            if e["type"] == "signal_timeout":
                return None
        w = next((e for e in evs if e["type"] == "wait"), None)
        if w is None:
            self._check_cancel()
            due = None if timeout is None else now_us() + int(timeout * 1_000_000)
            due = self._record("wait", key, signal, {"signal": signal, "due": due})["due"]
        else:
            due = (w.get("data") or {}).get("due")
        got = self._call("consume_signal", key=key, name=signal)
        if got.get("received"):
            return got.get("payload")
        if due is not None and now_us() >= due:
            self._record("signal_timeout", key, signal)
            return None
        raise _Suspend(signals=[signal], timers=[(key, due, "signal_timeout")] if due is not None else [])

    def child(self, workflow_name, input=None, id=None, version=None, queue=None):
        """Start child workflow `workflow_name` and wait for its output."""
        key = self._key(workflow_name)
        evs = self._events(key, workflow_name, ("child", "child_done"))
        done = next((e for e in evs if e["type"] == "child_done"), None)
        if done is None:
            self._check_cancel()
            args = {"key": key, "workflow": workflow_name, "input": input}
            for k, v in (("child_id", id), ("version", version), ("queue", queue)):
                if v is not None:
                    args[k] = v
            cid = self._call("start_child", **args)["child"]
            st = self._db.flow_call("status", {"id": cid})
            if st["status"] not in ("completed", "failed", "canceled"):
                raise _Suspend(children=[cid])
            data = {"id": cid, "status": st["status"], "output": st.get("output"), "error": st.get("error")}
            data = self._record("child_done", key, workflow_name, data)
        else:
            data = done["data"]
        if data.get("status") == "completed":
            return data.get("output")
        raise ChildFailed(f"child workflow '{data.get('id')}' {data.get('status')}: {data.get('error')}")

    def patched(self, patch_id):
        """True on executions that reach this point with the new code; False when replaying a
        history that passed it before the change. Does not take a command ordinal, so adding it
        does not shift the keys of the commands after it."""
        key = f"patch:{patch_id}"
        if any(e["type"] == "marker" for e in self._hist.get(key, [])):
            return True
        nxt = f"{self._n + 1}:"
        if any(k.startswith(nxt) for k in self._hist):
            return False  # an old run already went past here without the patch
        self._record("marker", key, "patch", True)
        self._hist.setdefault(key, []).append({"type": "marker"})
        return True

    def _marker(self, name, make):
        key = self._key(name)
        for e in self._events(key, name, ("marker",)):
            return e.get("data")
        return self._record("marker", key, name, make())

    def now(self):
        """The time (µs since the epoch), recorded once."""
        return self._marker("now", now_us)

    def uuid(self):
        """A random id, recorded once."""
        return self._marker("uuid", lambda: str(_uuid.uuid4()))


class Worker:
    """Runs code-mode workflows from one kleos database. Several workers (threads, processes) may
    share a database: claims hand each runnable workflow to one of them."""

    def __init__(self, db, workflows, queue="default", queues=None, name=None, lease_ms=30_000, poll=1.0,
                 log=None, lanes=None, on_claim=None):
        self.db = db
        self.fns = {}
        for fn in workflows:
            meta = getattr(fn, "_kleos_flow", None)
            if meta is None:
                raise ValueError(f"{fn!r} is not decorated with @workflow")
            self.fns[meta] = fn
        self.names = sorted({n for n, _ in self.fns})
        self.queues = list(queues or [queue])
        self.name = name or f"py-{_uuid.uuid4().hex[:8]}"
        self.lease_ms = int(lease_ms)
        self.poll = float(poll)
        self.log = log or (lambda rec: None)
        self.lanes = list(lanes) if lanes else None
        self.on_claim = on_claim
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run_once(self):
        """Claim and run one workflow. Returns its outcome, or None when nothing was runnable."""
        args = {"worker": self.name, "queues": self.queues, "modes": ["code"], "workflows": self.names,
                "lease_ms": self.lease_ms}
        if self.lanes:
            args["lanes"] = self.lanes
        c = self.db.flow_call("claim", args)
        if c is None:
            return None
        if self.on_claim:
            self.on_claim(c)
        return self._run(c)

    def drain(self, max_runs=None):
        """Run until nothing is runnable now. Returns the outcomes."""
        out = []
        while max_runs is None or len(out) < max_runs:
            r = self.run_once()
            if r is None:
                break
            out.append(r)
        return out

    def run(self, until=None):
        """Run until `stop()` (or until `until()` is true), sleeping until the next due work or
        `poll` seconds. Fires due schedules too."""
        while not self._stop.is_set() and not (until and until()):
            if self.run_once() is not None:
                continue
            try:
                self.db.flow_call("tick", {"queues": self.queues, "max": 0})
            except Exception as e:  # noqa: BLE001
                self.log({"event": "tick_error", "error": str(e)})
            nd = self.db.flow_call("next_due", {"queues": self.queues})
            wait = self.poll
            if nd.get("due") is not None:
                wait = min(wait, max(0.0, (nd["due"] - nd["now"]) / 1e6))
            self._stop.wait(wait)

    def _heartbeat(self, ctx, done):
        every = max(self.lease_ms / 3000.0, 0.05)
        while not done.wait(every):
            try:
                r = self.db.flow_call("heartbeat", {"id": ctx.id, "token": ctx.token, "lease_ms": self.lease_ms})
                if r.get("cancel"):
                    ctx.cancel_requested = True
            except Exception:  # noqa: BLE001  lease lost: the run's next write fails too
                return

    def _run(self, c):
        fn = self.fns.get((c["name"], c["version"]))
        rec = {"id": c["id"], "workflow": c["name"], "version": c["version"], "token": c["token"],
               "input": c.get("input")}
        if fn is None:
            msg = f"no worker for {c['name']} version {c['version']}"
            self.db.flow_call("fail", {"id": c["id"], "token": c["token"], "error": msg, "final": False,
                                       "retry_in_ms": 30_000})
            rec["outcome"] = "no_version"
            self.log(rec)
            return rec
        ctx = Context(self, c)
        done = threading.Event()
        hb = threading.Thread(target=self._heartbeat, args=(ctx, done), daemon=True)
        hb.start()
        t0 = time.time()
        try:
            if ctx.cancel_requested:
                raise Cancelled("canceled")
            out = fn(ctx, c.get("input"))
            ctx._finish_check()
            out = _jsonable(out)
            ctx._call("complete", output=out)
            rec["outcome"] = "completed"
            rec["output"] = out
        except _Suspend as s:
            try:
                ctx._call("suspend", timers=[{"key": k, "due": d, "kind": kind} for k, d, kind in s.timers],
                          signals=s.signals, children=s.children)
                rec["outcome"] = "waiting"
            except Cancelled:
                ctx._call("fail", error="canceled", canceled=True)
                rec["outcome"] = "canceled"
        except Cancelled as e:
            self._settle(ctx, rec, "fail", error=str(e) or "canceled", canceled=True, outcome="canceled")
        except LeaseLost as e:
            rec["outcome"] = "lease_lost"
            rec["error"] = str(e)
        except NonDeterminism as e:
            self._settle(ctx, rec, "fail", error=str(e), final=False, retry_in_ms=30_000, outcome="parked")
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            rec["trace"] = traceback.format_exc(limit=5)
            self._settle(ctx, rec, "fail", error=err, final=True, outcome="failed")
        finally:
            done.set()
        rec["seconds"] = round(time.time() - t0, 3)
        self.log(rec)
        return rec

    def _settle(self, ctx, rec, op, outcome, **args):
        try:
            ctx._call(op, **args)
            rec["outcome"] = outcome
        except LeaseLost as e:
            rec["outcome"] = "lease_lost"
            rec["error"] = str(e)
