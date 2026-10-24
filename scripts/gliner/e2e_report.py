# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Phase 2 summary: LoCoMo with the decider on, without and with entities
(docs/research/gliner.md, "Phase 2"). Reads bench-l2 results of one tag.

    python e2e_report.py [variants]       # default hybrid,sup,sup+ents,sup+gliner (config `default`)
    python e2e_report.py --sample 20      # also dump N random auto-supersedes per variant for a hand check
"""
import json
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "bench-l2"
sys.path.insert(0, str(BENCH))
import common as C  # noqa: E402
import report as R  # noqa: E402

TAG = os.environ.get("KLEOS_BENCH_TAG", "gliner-e2e")
VARIANTS = ["hybrid", "sup", "sup+ents", "sup+gliner"]
NAMES = {"hybrid": "A hybrid (no decider)", "sup": "B sup", "sup+ents": "C sup + caps", "sup+gliner": "D sup + GLiNER"}


def load(v):
    return R.load_dir(C.bench_dir() / "results" / TAG / "locomo-turn" / f"{v}+default")


def qa(v, set_="all-n300"):
    d = C.bench_dir() / "results" / TAG / "qa" / "locomo-turn" / set_ / f"ctx_{v}+default"
    rs = [json.loads(p.read_text()) for p in d.glob("*.json")] if d.is_dir() else []
    out = {"n": len(rs), "all": statistics.mean(r["correct"] for r in rs) if rs else None}
    for c in (1, 2, 3, 4):
        sel = [r["correct"] for r in rs if r["category"] == c]
        out[C.LOCOMO_CATS[c]] = (statistics.mean(sel) if sel else None, len(sel))
    return out


def main():
    sample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else 0
    vs = [a for a in sys.argv[1:] if not a.startswith("--") and not a.isdigit()]
    vs = vs[0].split(",") if vs else VARIANTS
    rows, dumps = [], {}
    for v in vs:
        recs = load(v)
        if not recs:
            continue
        ms = [m for r in recs.values() for q in r["questions"] if q["category"] != 5 for m in [R.locomo_metrics(q)] if m]
        a = R.agg(ms, ["R@5", "R@10"])
        st = Counter()
        lat, decs, es = [], [], Counter()
        for r in recs.values():
            st.update(r["ingest"])
            lat += r["remember_ms"]
            decs += r.get("decisions") or []
            e = r.get("entity_stats") or {}
            es.update({k: e.get(k, 0) for k in ("entities", "merges", "possible_same")})
        auto = [d for d in decs if d["action"] == "superseded"]
        pend = [d for d in decs if d["action"] == "pending" and d["template"] == "supersedes"]
        q = qa(v)
        rows.append({"variant": NAMES.get(v, v), "R@5": a["R@5"], "R@10": a["R@10"], "qa": q,
                     "remember_p50": statistics.median(lat), "remember_p90": statistics.quantiles(lat, n=10)[-1],
                     "ingest": dict(st), "auto_supersedes": len(auto), "pending_supersedes": len(pend),
                     "dup_decisions": sum(1 for d in decs if d["template"] == "duplicate"), "entities": dict(es),
                     "n_questions": len(ms)})
        if sample and auto:
            random.seed(C.SEED)
            dumps[v] = random.sample(auto, min(sample, len(auto)))
    print(json.dumps(rows, indent=1))
    if dumps:
        out = C.bench_dir() / "supersede_sample.json"
        out.write_text(json.dumps(dumps, indent=1, ensure_ascii=False))
        print("sample ->", out)


if __name__ == "__main__":
    main()
