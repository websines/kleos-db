"""Split-half fusion-weight selection (docs/benchmarks/memory.md, "Fusion weight").

Reads retrieval.py result directories for several recall configs (one per
vector weight) and scores them per half: the weight is picked on one half
of the conversations (LoCoMo) / questions (LongMemEval) and reported on the
other, so the chosen default is not fitted to the numbers it is judged by.

    python fusion_split.py <results/<tag>/<bench-unit>> [variant]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import report as R  # noqa: E402


def load(d):
    out = {}
    for f in sorted(Path(d).glob("*.json")):
        if f.name == "meta.json":
            continue
        x = json.loads(f.read_text())
        if "questions" in x:  # LoCoMo: one file per conversation
            ms = [m for m in (R.locomo_metrics(q) for q in x["questions"] if q.get("category") != 5) if m]  # cat 1-4, as report.py
        else:  # LongMemEval: one file per question
            ms = [R.lme_metrics(x)]
        out[f.stem] = ms
    return out


def agg(groups, keys):
    ms = [m for g in groups for m in g]
    return {k: R.mean([m.get(k) for m in ms]) for k in keys}, len(ms)


def main(root, variant="hybrid"):
    root = Path(root)
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(variant + "+"))
    data = {p.name.split("+", 1)[1]: load(p) for p in dirs}
    units = sorted(set.intersection(*[set(v) for v in data.values()]))
    halves = (units[0::2], units[1::2])
    keys = ["R@5", "R@10", "NDCG@10"]
    rows = []
    for name, per in data.items():
        a, na = agg([per[u] for u in halves[0]], keys)
        b, nb = agg([per[u] for u in halves[1]], keys)
        allm, n = agg([per[u] for u in units], keys)
        rows.append((name, a, b, allm, na, nb, n))
    print(f"{root} — {len(units)} units, halves {len(halves[0])}/{len(halves[1])}")
    print(f"{'config':14} {'A R@5':>7} {'A R@10':>7} {'B R@5':>7} {'B R@10':>7} {'all R@5':>8} {'all R@10':>8} {'all NDCG':>8}")
    for name, a, b, allm, *_ in rows:
        print(f"{name:14} {a['R@5']:7.3f} {a['R@10']:7.3f} {b['R@5']:7.3f} {b['R@10']:7.3f} {allm['R@5']:8.3f} {allm['R@10']:8.3f} {allm['NDCG@10']:8.3f}")
    # pick on A (by R@5 + R@10), report on B, and the reverse
    for pick, rep in ((1, 2), (2, 1)):
        best = max(rows, key=lambda r: r[pick]["R@5"] + r[pick]["R@10"])
        dflt = next(r for r in rows if r[0] == "default")
        print(f"picked on {'AB'[pick - 1]}: {best[0]} → on {'AB'[rep - 1]}: R@5 {best[rep]['R@5']:.3f} (default {dflt[rep]['R@5']:.3f}), "
              f"R@10 {best[rep]['R@10']:.3f} (default {dflt[rep]['R@10']:.3f})")
    print(json.dumps({r[0]: {"A": r[1], "B": r[2], "all": r[3]} for r in rows}))


if __name__ == "__main__":
    main(*sys.argv[1:])
