# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Supersession audit of decider-on LoCoMo runs (bench-l2 retrieval.py
`--variant sup`), ontology off vs on: automatic supersedes and pending
decisions per conversation, and a random sample of automatic supersedes for a
hand check.

    python locomo_audit.py OFF_DIR ON_DIR OUT.json [sample=20]
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path


def load(d):
    out = {}
    for f in sorted(Path(d).glob("conv-*.json")):
        x = json.loads(f.read_text())
        out[x["sample_id"]] = x
    return out


def summary(runs, convs):
    rows = {}
    for c in convs:
        x = runs[c]
        dec = x.get("decisions", [])
        rows[c] = {
            "turns": x["n_units"],
            "auto_supersedes": sum(1 for d in dec if d["action"] == "superseded"),
            "pending": sum(1 for d in dec if d["action"] == "pending"),
            "pending_supersedes": sum(1 for d in dec if d["action"] == "pending" and d["template"] == "supersedes"),
            "pending_duplicates": sum(1 for d in dec if d["action"] == "pending" and d["template"] == "duplicate"),
            "auto_duplicates": sum(1 for d in dec if d["action"] == "duplicate"),
            "ingest_s": round(x.get("ingest_s", 0), 1),
            "remember_p50_ms": sorted(x["remember_ms"])[len(x["remember_ms"]) // 2] if x.get("remember_ms") else None,
            "ingest": x.get("ingest"),
        }
    tot = Counter()
    for r in rows.values():
        for k, v in r.items():
            if isinstance(v, (int, float)) and k not in ("remember_p50_ms",):
                tot[k] += v
    return rows, dict(tot)


def main(off_dir, on_dir, out_path, n=20):
    off, on = load(off_dir), load(on_dir)
    convs = sorted(set(off) & set(on))
    so, to = summary(off, convs)
    sn, tn = summary(on, convs)
    rnd = random.Random(11)
    samp = {}
    for name, runs in (("off", off), ("on", on)):
        sup = [dict(d, conv=c) for c in convs for d in runs[c].get("decisions", []) if d["action"] == "superseded"]
        samp[name] = rnd.sample(sup, min(n, len(sup)))
    out = {"convs": convs, "off": {"per_conv": so, "total": to}, "on": {"per_conv": sn, "total": tn}, "sample": samp}
    Path(out_path).write_text(json.dumps(out, indent=1))
    print("convs", convs)
    print("off", json.dumps(to))
    print("on ", json.dumps(tn))
    for name in ("off", "on"):
        print(f"--- {name}: sample of automatic supersedes")
        for d in samp[name]:
            print(f"  [{d['conv']}] OLD: {d['old'][:110]}\n           NEW: {d['new'][:110]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 20)
