"""Retune GLiNER's per-label thresholds on the hand-checked gold set, with
2-fold cross-validation so the estimate is honest (the gold set is also the
test set). Uses the Rust port's statement-level f16 Metal predictions and
their stored raw scores: no re-extraction.

    python retune_thresholds.py <pred_gold.json> [seed]
"""
import json, os, random, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("GLINER_DIR", "/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr/.bench-data/gliner-ft")
import common as C  # noqa: E402
import ft_eval as E  # noqa: E402
from ft_data import ENTS  # noqa: E402

FIXED = {"person": 0.8, "location": 0.9, "organization": 0.8, "relative": 0.5}


def score(items, preds, thr):
    c = E.ent_counts(items, preds, thr)
    nm = Counter()
    for lab in ("person", "location", "organization"):
        for k in ("tp", "np", "ng"):
            nm[k] += c[lab][k]
    row = {lab: round(E.prf(c[lab])[2], 3) for lab in ENTS}
    row["names_F"] = round(E.prf(nm)[2], 3)
    return row


def main():
    gold = json.loads((C.GDIR / "gold.json").read_text())
    P = json.load(open(sys.argv[1]))
    preds = [E.post(p) for p in P["pred"]]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    idx = list(range(len(gold)))
    random.Random(seed).shuffle(idx)
    folds = [idx[: len(idx) // 2], idx[len(idx) // 2:]]
    cv = {"fixed": Counter(), "tuned": Counter()}
    per = []
    for k in (0, 1):
        tr, te = folds[k], folds[1 - k]
        thr = E.tune([gold[i] for i in tr], [preds[i] for i in tr])
        a = score([gold[i] for i in te], [preds[i] for i in te], FIXED)
        b = score([gold[i] for i in te], [preds[i] for i in te], thr)
        per.append({"fold": k, "thr": thr, "fixed": a, "tuned": b})
        for lab, v in a.items():
            cv["fixed"][lab] += v / 2
        for lab, v in b.items():
            cv["tuned"][lab] += v / 2
    all_thr = E.tune(gold, preds)
    out = {
        "n": len(gold),
        "cv_fixed": {k: round(v, 3) for k, v in cv["fixed"].items()},
        "cv_tuned": {k: round(v, 3) for k, v in cv["tuned"].items()},
        "folds": per,
        "tuned_on_all": all_thr,
        "all_fixed": score(gold, preds, FIXED),
        "all_tuned_optimistic": score(gold, preds, all_thr),
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
