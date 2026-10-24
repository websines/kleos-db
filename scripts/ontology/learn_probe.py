"""Ontology learning on real conversation: LoCoMo conversations remembered
with the default ontology (no decider, so it is quick), then `learn`
(review mode first, to see every proposal, then auto) and `maintain`.
Reports the untyped share before and after and every proposal.

    python learn_probe.py OUT.json [n_conversations]
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench-l2"))
import common as C  # noqa: E402
from kleos import CozoDbPy  # noqa: E402


def share(db, ns):
    rows = db.run_script(f"?[cls, count(id)] := *kleos_mem_type{{ns: '{ns}', id, cls}}", {}, True)["rows"]
    d = {r[0]: r[1] for r in rows}
    n = sum(d.values()) or 1
    return {"counts": d, "untyped_share": round(d.get("untyped", 0) / n, 3), "fact_share": round(d.get("fact", 0) / n, 3)}


def main(out_path, n_conv):
    os.environ.setdefault("KLEOS_ONTO_LEARN_EVERY", "0")  # learn explicitly below
    data = json.loads(C.locomo_path().read_text())[:n_conv]
    db = CozoDbPy("mem", "", "{}")
    emb = os.environ.get("KLEOS_EMBEDDER", "gemma")
    db.enable_embedder(emb, device="metal")
    db.memory_init(embedder=emb)
    out = []
    for conv in data:
        ns = conv["sample_id"]
        c = conv["conversation"]
        t0 = time.time()
        k = 1
        while f"session_{k}" in c:
            at = C.locomo_ts(c[f"session_{k}_date_time"])
            for j, turn in enumerate(c[f"session_{k}"]):
                db.remember(f"{turn['speaker']}: {turn['text']}", ns=ns, at=at + j * 1_000_000)
            k += 1
        before = share(db, ns)
        review = db.ontology_call("learn", {"ns": ns, "mode": "review"})
        props = db.ontology_call("proposals", {"ns": ns, "status": "proposed"})
        # approve everything auto mode would have applied: run auto on a fresh view
        for p in props:
            if p["kind"] == "property" and len(p["evidence"]) >= 8:
                db.ontology_call("approve", {"ns": ns, "id": p["id"]})
        maint = db.ontology_call("maintain", {"ns": ns})
        after = share(db, ns)
        rec = {
            "ns": ns,
            "seconds": round(time.time() - t0, 1),
            "before": before,
            "after": after,
            "proposals": [
                {"id": p["id"], "kind": p["kind"], "support": len(p["evidence"]),
                 "prop": p["body"]["prop"] if isinstance(p["body"]["prop"], str) else
                 {k: p["body"]["prop"].get(k) for k in ("name", "card", "datatype", "cues")},
                 "examples": p["body"].get("examples")}
                for p in props
            ],
            "review_untyped": review.get("untyped"),
            "maintenance": maint.get("changes"),
        }
        out.append(rec)
        print(json.dumps({k: rec[k] for k in ("ns", "seconds", "before", "after")}), flush=True)
        for p in rec["proposals"]:
            print("   ", p["kind"], p["support"], json.dumps(p["prop"])[:160], flush=True)
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
