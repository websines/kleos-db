#!/usr/bin/env python3
# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Side-by-side retrieval metrics of recall configurations run from one ingest
(retrieval.py --configs): every <variant>+<config> directory of a bench, paired
on the questions all of them ran.

    python compare_configs.py --tag recall --bench lme --variant hybrid
    python compare_configs.py --tag recall --bench locomo --variant fts
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import report as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=os.environ.get("KLEOS_BENCH_TAG") or C.git_rev())
    ap.add_argument("--bench", choices=["lme", "lme-oracle", "locomo"], required=True)
    ap.add_argument("--unit", default="user")
    ap.add_argument("--variant", required=True, help="variant prefix, e.g. hybrid or fts+ents")
    ap.add_argument("--configs", default=None, help="comma list (default: every config dir present)")
    args = ap.parse_args()
    unit = "turn" if args.bench == "locomo" else args.unit
    root = C.bench_dir() / "results" / args.tag / f"{args.bench}-{unit}"
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(args.variant + "+"))
    if args.configs:
        want = args.configs.split(",")
        dirs = [root / f"{args.variant}+{c}" for c in want]
    data = {d.name.split("+", 1 + args.variant.count("+"))[-1]: R.load_dir(d) for d in dirs}
    data = {k: v for k, v in data.items() if v}
    ids = set.intersection(*[set(v) for v in data.values()])
    print(f"{args.bench} {args.variant}: {len(ids)} units paired over {', '.join(data)}\n")
    if args.bench == "locomo":
        keys = ["R@5", "R@10", "Rall@5", "Rall@10", "NDCG@10"]
        splits = [("cat 1-4", lambda q: q["category"] != 5)] + [
            (f"{c} {C.LOCOMO_CATS[c]}", (lambda c: lambda q: q["category"] == c)(c)) for c in (1, 2, 3, 4, 5)]
        rows = []
        for name, pred in splits:
            for cfg, recs in data.items():
                ms = [m for m in (R.locomo_metrics(q) for i in sorted(ids) for q in recs[i]["questions"] if pred(q)) if m]
                a = R.agg(ms, keys)
                rows.append([name, cfg, len(ms)] + [R.f2(a[k]) for k in keys])
        print(R.table(rows, ["split", "config", "n"] + keys))
        return
    keys = ["R@5", "R@10", "Rall@5", "Rall@10", "T@5", "T@10", "NDCG@10"]
    held, ku_train, _ = C.lme_ku_split()
    splits = [("all", lambda r: True)] + [(c, (lambda c: lambda r: r["category"] == c)(c)) for c in R.LME_CATS]
    splits = splits[:2] + R.ss_groups() + splits[2:]
    rows = []
    for name, pred in splits:
        for cfg, recs in data.items():
            sel = [recs[i] for i in sorted(ids) if pred(recs[i])]
            if not sel:
                continue
            a = R.agg([R.lme_metrics(r) for r in sel], keys)
            rows.append([name, cfg, len(sel)] + [R.f2(a[k]) for k in keys])
    print(R.table(rows, ["split", "config", "n"] + keys))
    kk = ["KUlatest@5", "KUlatest@10", "KUstale1st"]
    rows = []
    for cfg, recs in data.items():
        ms = [R.lme_metrics(recs[i]) for i in sorted(ids)
              if recs[i]["question_type"] == "knowledge-update" and not i.endswith("_abs")]
        ms = [m for m in ms if "KUlatest@5" in m]
        if ms:
            a = R.agg(ms, kk)
            rows.append([cfg, len(ms)] + [R.f2(a[k]) for k in kk])
    if rows:
        print("\nKnowledge update (latest@k, stale-first ↓):\n")
        print(R.table(rows, ["config", "n"] + kk))


if __name__ == "__main__":
    main()
