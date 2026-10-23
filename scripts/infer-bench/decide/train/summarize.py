"""Before/after table from decide_eval outputs (run_eval.sh): one row per setup.

    python3 summarize.py EVAL_DIR TAG[:REF_TAG] ...

REF_TAG (optional) is the f32 run of the same weights: decision flips and max |Δp| of TAG
against it are reported (binary templates at REF's adapt thresholds, relation by argmax).
"""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])


def load(tag):
    r = json.loads((D / f"{tag}.json").read_text())
    return [x for x in r if "decider-nano" in x["backend"]][0]


def flips(tag, ref):
    a = json.loads((D / f"{tag}.dump.json").read_text())
    b = json.loads((D / f"{ref}.dump.json").read_text())
    rr = load(ref)
    thr = {t: rr[t].get("threshold", 0.5) for t in ("supersedes", "contradicts", "duplicate")}
    thr["same_entity"] = 0.5
    kb = {}
    for t, x, y, g, p in b:
        kb.setdefault((t, x, y), p)
    n = fl = 0
    dmax = 0.0
    for t, x, y, g, p in a:
        q = kb.get((t, x, y))
        if q is None:
            continue
        n += 1
        if t == "relation":
            fl += max(range(len(p)), key=lambda i: p[i]) != max(range(len(q)), key=lambda i: q[i])
        else:
            fl += (p[1] >= thr[t]) != (q[1] >= thr[t])
            dmax = max(dmax, abs(p[1] - q[1]))
    return n, fl, dmax


def ci(x):
    return f"[{x[0]:.3f}, {x[1]:.3f}]"


rows = []
for spec in sys.argv[2:]:
    tag, _, ref = spec.partition(":")
    r = load(tag)
    s, c, du, se, rel = r["supersedes"], r["contradicts"], r["duplicate"], r["same_entity"], r["relation"]
    lme, mn, vc = r.get("lme_ku_heldout_supersedes", {}), r.get("mnli_dev", {}), r.get("vitaminc300_supersedes", {})
    lat = r.get("latency_supersedes", {})
    line = {
        "setup": tag,
        "supersedes AUROC": f"{s['auroc']:.3f} {ci(s['auroc_ci95'])}",
        "sup vs compatible": f"{s['auroc_vs_compatible']:.3f}",
        "sup acc@thr (thr)": f"{s['acc_at_threshold']:.3f} ({s['threshold']:.3f})",
        "sup auto cov/false (tau)": f"{s['auto_coverage']:.2f}/{s['auto_false']} ({s['tau_auto']:.3f})",
        "sup probe AUROC / auto cov/false": f"{s['probe']['auroc']:.3f} / {s['probe']['auto_coverage']:.2f}/{s['probe']['auto_false']}",
        "contradicts AUROC": f"{c['auroc']:.3f} {ci(c['auroc_ci95'])}",
        "duplicate AUROC / acc": f"{du['auroc']:.3f} / {du['acc_at_threshold']:.3f}",
        "dup auto cov/false": f"{du['auto_coverage']:.2f}/{du['auto_false']}",
        "same_entity AUROC": f"{se['auroc']:.3f} {ci(se['auroc_ci95'])}",
        "relation acc3 (zs / bias / probe4)": f"{rel['acc3']:.3f} / {rel['acc3_with_bias']:.3f} / {rel['probe']['acc4']:.3f}",
        "VitaminC-300 AUROC": f"{vc.get('zero_shot_auroc', float('nan')):.3f} {ci(vc.get('zero_shot_ci95', [0, 0]))}",
        "VitC auto cov/false": f"{vc.get('bands', {}).get('auto_coverage', 0):.2f}/{vc.get('bands', {}).get('auto_false', 0)}",
        "LME-KU AUROC": f"{lme.get('auroc', float('nan')):.3f} {ci(lme.get('auroc_ci95', [0, 0]))}",
        "LME auto cov/false": f"{lme.get('bands', {}).get('auto_coverage', 0):.2f}/{lme.get('bands', {}).get('auto_false', 0)}",
        "MNLI dev acc3": f"{mn.get('acc3', float('nan')):.3f} {ci(mn.get('acc3_ci95', [0, 0]))}",
        "b1 p50/p90 ms": f"{lat.get('b1_p50_ms', 0):.1f}/{lat.get('b1_p90_ms', 0):.1f}",
        "b16/b64 ms/dec": f"{lat.get('batch', {}).get('b16_ms_per_decision', 0):.1f}/{lat.get('batch', {}).get('b64_ms_per_decision', 0):.1f}",
        "RSS MB": f"{r.get('rss_mb')}",
        "rule fired": r.get("rule_fired", 0),
    }
    if ref:
        n, fl, dmax = flips(tag, ref)
        line["flips vs f32 (n, max|dp|)"] = f"{fl}/{n}, {dmax:.3f}"
    rows.append(line)

keys = list(dict.fromkeys(k for r in rows for k in r))
print("| " + " | ".join(keys) + " |")
print("|" + "---|" * len(keys))
for r in rows:
    print("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
