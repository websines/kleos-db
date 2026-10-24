# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Claim typing and the cardinality policy on real data (docs/specs/ontology.md,
"Results"), with the real embedder (EmbeddingGemma on Metal) and the
`personal` pack:

- the 22 held-out LongMemEval knowledge-update pairs (old claim, new claim of
  a true update; `.bench-data/gliner/ku_pairs.json`): same-property rate, and
  the policy's gate for each pair (a true update must stay `allow` or
  `cap_pending`);
- LoCoMo supersede decisions of a decider-on run (old, new, action) from a
  bench-l2 results directory: what the policy would do to each;
- a random sample of typed claims for a hand check.

    python typing_eval.py OUT.json [LOCOMO_RESULTS_DIR]
"""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

from kleos import CozoDbPy

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("KLEOS_BENCH_DATA", ROOT / ".bench-data"))
KU_PAIRS = BENCH / "gliner" / "ku_pairs.json"


def main(out_path, locomo_dir=None):
    db = CozoDbPy("mem", "", "{}")
    db.enable_embedder("gemma", model=os.environ.get("KLEOS_EMBED_MODEL") or None, device="metal")
    db.memory_init(embedder="gemma")
    db.ontology_call("activate", {"ns": "default", "ontos": ["personal"]})

    def typ(t):
        return db.ontology_call("type", {"text": t})

    def gate(a, b):
        return db.ontology_call("gate", {"old": a, "new": b})["gate"]

    out = {}
    # 1. KU true pairs
    P = json.loads(KU_PAIRS.read_text())
    pairs = [p for p in P["pairs"] if p["split"] == "held"]
    rows = []
    for p in pairs:
        to, tn = typ(p["old"]), typ(p["new"])
        g = gate(p["old"], p["new"])
        rows.append({"qid": p["qid"], "old": p["old"], "new": p["new"], "old_type": to, "new_type": tn, "gate": g})
    same = sum(1 for r in rows if r["old_type"].get("prop") and r["old_type"].get("prop") == r["new_type"].get("prop"))
    gates = Counter(r["gate"] for r in rows)
    out["ku_pairs"] = {
        "n": len(rows), "same_prop": same, "gates": gates,
        "smalltalk": sum(1 for r in rows for t in (r["old_type"], r["new_type"]) if t["cls"] == "smalltalk"),
        "untyped": sum(1 for r in rows for t in (r["old_type"], r["new_type"]) if t["cls"] == "untyped"),
        "rows": rows,
    }
    print("KU", json.dumps({k: v for k, v in out["ku_pairs"].items() if k != "rows"}), flush=True)

    # 2. LoCoMo decisions of a decider-on run
    if locomo_dir:
        lrows = []
        for f in sorted(Path(locomo_dir).glob("conv-*.json")):
            d = json.loads(f.read_text())
            for x in d.get("decisions", []):
                if x["template"] != "supersedes" or not x.get("old"):
                    continue
                lrows.append({"conv": d["sample_id"], "old": x["old"], "new": x["new"], "action": x["action"],
                              "p": x["p"], "gate": gate(x["old"], x["new"])})
        by = Counter((r["action"], r["gate"]) for r in lrows)
        out["locomo"] = {"n": len(lrows), "by_action_gate": {f"{a}/{g}": n for (a, g), n in sorted(by.items())}, "rows": lrows}
        print("LOCOMO", json.dumps(out["locomo"]["by_action_gate"]), flush=True)
        # every claim of those decisions, typed, for the share of small talk
        texts = sorted({r["new"] for r in lrows} | {r["old"] for r in lrows})
        tt = [typ(t) for t in texts]
        out["locomo_types"] = Counter(t["cls"] for t in tt)
        out["locomo_props"] = Counter(t.get("prop") for t in tt if t.get("prop")).most_common(20)
        rnd = random.Random(7)
        out["hand_sample"] = [{"text": t, **ty} for t, ty in rnd.sample(list(zip(texts, tt)), min(40, len(texts)))]
        print("LOCOMO types", json.dumps(out["locomo_types"]), flush=True)
    Path(out_path).write_text(json.dumps(out, indent=1, default=dict))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
