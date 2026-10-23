#!/usr/bin/env python3
# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Memory branches end to end with real models (docs/specs/memory-branches.md).

An agent's memory ("main") holds a few facts about the user. The agent tries
two plans in parallel, each in its own branch: in one the user moves to
Berlin, in the other to Lisbon. Recall differs per branch; main is untouched.
The agent keeps the Lisbon branch (a dry-run merge first, then the real one:
decider-nano supersedes "I live in Paris." in main), drops the Berlin branch,
and finally rewinds main to before the session, then undoes the rewind.

    python -I scripts/branches/branches_demo.py [--db PATH] [--device metal|cpu]

Needs the kleos Python bindings built with `infer-metal` (or `infer`); the
models (EmbeddingGemma, kleos-decider-nano) download on first use into
HF_HOME. Prints each step and a JSON summary; exits non-zero if a check
fails.
"""
import argparse
import json
import os
import statistics
import sys
import time

DAY = 86_400_000_000

MAIN = [
    (1, "I live in Paris.", ["user", "Paris"]),
    (2, "I work as a nurse at the city hospital.", ["user"]),
    (3, "My sister Anna lives in Madrid.", ["Anna", "Madrid"]),
    (4, "I drink my coffee black.", ["user"]),
    (5, "I have a cat named Miso.", ["user", "Miso"]),
]
BERLIN = [("I moved to Berlin last week.", ["user", "Berlin"]), ("I signed up for a German course.", ["user"])]
LISBON = [("I moved to Lisbon last week.", ["user", "Lisbon"]), ("I started learning Portuguese.", ["user"])]
WHERE = "Where do I live now?"

checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    print(("  ok   " if ok else "  FAIL ") + name + (f" — {detail}" if detail else ""))


def now_us():
    return int(time.time() * 1e6)


def timed(f, *a, **kw):
    t = time.perf_counter()
    r = f(*a, **kw)
    return r, (time.perf_counter() - t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="SQLite file (default: in memory)")
    ap.add_argument("--device", default="metal")
    ap.add_argument("--out", default=None, help="write the JSON summary here")
    args = ap.parse_args()
    from kleos import CozoDbPy

    db = CozoDbPy("sqlite" if args.db else "mem", args.db or "", "{}")
    t0 = time.perf_counter()
    db.enable_embedder("gemma", device=args.device)
    db.enable_decider("nano", device=args.device)
    db.memory_init(embedder="gemma", decider="nano")
    print(f"models ready in {time.perf_counter() - t0:.1f} s")

    base = now_us() - 30 * DAY
    print("\n1. main learns five facts")
    ids = {}
    for day, text, ents in MAIN:
        r = db.remember(text, ns="main", at=base + day * DAY, entities=ents)
        ids[text] = r["id"]
        print(f"  + {text}")
    session = now_us()
    time.sleep(0.01)

    print("\n2. fork main into two branches (O(1): one row each)")
    fb, ms_b = timed(db.memory_fork, "main", "try-berlin")
    fl, ms_l = timed(db.memory_fork, "main", "try-lisbon")
    print(f"  try-berlin depth {fb['depth']} ({ms_b:.1f} ms), try-lisbon depth {fl['depth']} ({ms_l:.1f} ms)")

    print("\n3. each branch learns something conflicting")
    for ns, facts in (("try-berlin", BERLIN), ("try-lisbon", LISBON)):
        for text, ents in facts:
            r = db.remember(text, ns=ns, entities=ents)
            print(f"  [{ns}] + {text}  superseded={len(r['superseded'])} tentative={len(r['tentative'])}")
    paris_id = ids["I live in Paris."]
    berlin_r = db.recall(WHERE, ns="try-berlin", k=1, record_access=False)[0]
    lisbon_r = db.recall(WHERE, ns="try-lisbon", k=1, record_access=False)[0]
    main_r = db.recall(WHERE, ns="main", k=1, record_access=False)[0]

    print("\n4. recall differs per branch")
    for ns, h in (("main", main_r), ("try-berlin", berlin_r), ("try-lisbon", lisbon_r)):
        print(f"  {ns:11s} → {h['text']}")
    check("main still says Paris", "Paris" in main_r["text"], main_r["text"])
    check("try-berlin says Berlin", "Berlin" in berlin_r["text"], berlin_r["text"])
    check("try-lisbon says Lisbon", "Lisbon" in lisbon_r["text"], lisbon_r["text"])
    cur_b = {h["text"] for h in db.recall("Paris", ns="try-berlin", k=10, record_access=False)}
    check("Paris is superseded inside try-berlin", "I live in Paris." not in cur_b, sorted(cur_b))

    print("\n5. diff main → try-lisbon")
    d = db.memory_diff("main", "try-lisbon")
    for a in d["added"]:
        print(f"  + {a['text']}")
    for r_ in d["removed"]:
        print(f"  - {r_['text']} ({r_['reason']})")
    check("diff: Lisbon and Portuguese added", {a["text"] for a in d["added"]} == {t for t, _ in LISBON})
    check("diff: Paris superseded", [(r_["text"], r_["reason"]) for r_ in d["removed"]] == [("I live in Paris.", "superseded")])

    print("\n6. merge try-lisbon into main: dry run, then for real")
    dry, ms_dry = timed(db.memory_merge, "try-lisbon", dry_run=True)
    for st in dry["replayed"]:
        print(f"  plan: {st['text']} → {st['action']}, supersedes {len(st['superseded'])}")
    check("dry run changes nothing", db.recall(WHERE, ns="main", k=1, record_access=False)[0]["text"] == main_r["text"])
    check("dry run plans Paris superseded", any(paris_id in st["superseded"] for st in dry["replayed"]))
    real, ms_merge = timed(db.memory_merge, "try-lisbon")
    for st in real["replayed"]:
        print(f"  done: {st['text']} → {st['action']}, superseded {st['superseded']}, pending {len(st['tentative'])}")
    after = db.recall(WHERE, ns="main", k=1, record_access=False)[0]
    print(f"  main → {after['text']}")
    check("merge: decider superseded Paris in main", any(paris_id in st["superseded"] for st in real["replayed"]))
    check("main now says Lisbon", "Lisbon" in after["text"], after["text"])

    print("\n7. discard try-berlin")
    disc = db.memory_discard("try-berlin")
    print(f"  discarded {disc['discarded']} ({disc['memories']} memories of its own)")
    names = [b["ns"] for b in db.memory_branches()]
    check("only try-lisbon is left", names == ["try-lisbon"], names)
    check("main unaffected by the discard", "Lisbon" in db.recall(WHERE, ns="main", k=1, record_access=False)[0]["text"])

    print("\n8. rewind main to before the session, then undo")
    rw, ms_rw = timed(db.memory_rewind, "main", session)
    print(f"  retracted {len(rw['retracted'])}, restored {len(rw['restored'])}")
    back = db.recall(WHERE, ns="main", k=1, record_access=False)[0]["text"]
    print(f"  main → {back}")
    check("rewind restores Paris", "Paris" in back, back)
    un = db.memory_rewind("main", rw["undo"])
    again = db.recall(WHERE, ns="main", k=1, record_access=False)[0]["text"]
    print(f"  undo: retracted {len(un['retracted'])}, restored {len(un['restored'])}; main → {again}")
    check("undo brings Lisbon back", "Lisbon" in again, again)

    # recall latency: main vs a branch of it (depth 1) vs a nested one
    db.memory_fork("main", "b1")
    db.memory_fork("b1", "b2")
    qs = ["Where do I live now?", "What do I drink?", "Who is Anna?", "What pet do I have?", "What is my job?"]
    lat = {}
    for ns in ("main", "b1", "b2"):
        for q in qs:  # warm-up
            db.recall(q, ns=ns, k=5, record_access=False)
        xs = []
        for _ in range(4):
            for q in qs:
                _, ms = timed(db.recall, q, ns=ns, k=5, record_access=False)
                xs.append(ms)
        lat[ns] = statistics.median(xs)
    print("\nrecall p50 (ms): " + ", ".join(f"{k} {v:.1f}" for k, v in lat.items()))
    summary = {
        "checks": checks,
        "ok": all(c["ok"] for c in checks),
        "ms": {"fork": [round(ms_b, 2), round(ms_l, 2)], "merge_dry_run": round(ms_dry, 1), "merge": round(ms_merge, 1),
               "rewind": round(ms_rw, 1), "recall_p50": {k: round(v, 2) for k, v in lat.items()}},
    }
    print(json.dumps(summary["ms"]))
    print(f"\n{sum(c['ok'] for c in checks)}/{len(checks)} checks passed")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
    sys.exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
