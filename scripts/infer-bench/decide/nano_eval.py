"""decider-nano on the kleos decide() templates (Python reference, interim numbers).
Templates/thresholds/bias/probes chosen ONLY on the 60 ADAPT items; reported on the 42+24 EVAL items.
Usage: nano_eval.py OUT.json [cpu|mps]"""
import json, sys, glob, time, random, resource
import numpy as np
sys.path.insert(0, "/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr/kleos/scripts/infer-bench/d1")
snap = glob.glob("/Volumes/SSD01/caches/huggingface/hub/models--tasksource--tasksource-decider-nano/snapshots/*")[0]
sys.path.insert(0, snap)
import torch
torch.set_num_threads(4)
from pathlib import Path
from eval_set import SUPERSEDE, ENTITY
from eval_set_adapt import ADAPT
from decider import Decider
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
out_path = sys.argv[1]
dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
if dev == "mps":
    torch.mps.set_per_process_memory_fraction(0.25)
m = Decider(Path(snap), device=dev)
rng = np.random.default_rng(0)

def auroc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, bool)
    pos, neg = p[y], p[~y]
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())
def boot(stat, p, y, B=2000):
    p, y = np.asarray(p), np.asarray(y); vals = []
    idx_by = [np.where(y == c)[0] for c in np.unique(y)]
    for _ in range(B):
        idx = np.concatenate([rng.choice(ix, len(ix)) for ix in idx_by]); vals.append(stat(p[idx], y[idx]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]
def acc_at(p, y, t): return float(((np.asarray(p) >= t) == np.asarray(y, bool)).mean())
def best_thr(p, y):
    c = sorted(set(np.round(p, 6))) + [1.01, 0.5]
    return max(c, key=lambda t: (acc_at(p, y, t), -abs(t - 0.5)))
def tau_auto(p, y, alpha=0.02):
    o = np.argsort(-np.asarray(p)); n = fp = 0; best = 1.01; i = 0; p = np.asarray(p); y = np.asarray(y)
    while i < len(o):
        s = p[o[i]]
        while i < len(o) and p[o[i]] == s:
            n += 1; fp += (not y[o[i]]); i += 1
        if fp / n > alpha: break
        best = s
    return float(best)
def auto_stats(p, y, t):
    p, y = np.asarray(p), np.asarray(y, bool); a = p >= t
    return {"tau_auto": round(t, 4), "coverage": round(float(a.mean()), 3),
            "false_auto": int((a & ~y).sum()), "auto_n": int(a.sum()), "recall_auto": round(float((a & y).sum() / max(y.sum(), 1)), 3)}

CNLI = ["Entailment: The old note implies the new statement is true",
        "Contradiction: The old note and the new statement cannot both be true",
        "NotMentioned: The old note neither implies nor contradicts the new statement"]
NLI = ["entailment: The hypothesis follows from the premise",
       "neutral: The hypothesis might be true or false given the premise",
       "contradiction: The hypothesis contradicts the premise"]
ENT = ["false: they are different entities that may share a name", "true: both mentions refer to one and the same entity"]
T = {"cnli": (lambda a, b: a, lambda a, b: f"Classify the relationship between the old note and this new statement:\n{b}", CNLI),
     "nli": (lambda a, b: f"Premise: {a}\nHypothesis: {b}", lambda a, b: "What is the relationship between the premise and the hypothesis?", NLI),
     "ent": (lambda a, b: f"Mention A: {a}\nMention B: {b}", lambda a, b: "Are mention A and mention B the same entity?", ENT)}

def reqs(pairs, t): st, q, o = T[t]; return [(st(a, b), q(a, b), o) for a, b in pairs]
def probs(pairs, t, bs=16):
    rs = reqs(pairs, t); out = []
    for i in range(0, len(rs), bs):
        for z in m.score_many(rs[i:i + bs]):
            e = np.exp(z - z.max()); out.append(e / e.sum())
    return np.array(out)
cap = {}
hk = m.net.scorer.register_forward_hook(lambda mod, i, o: cap.__setitem__("v", i[0].detach().float().cpu()))
def feats(pairs, t):
    F = []
    for r in reqs(pairs, t):
        m.score_many([r]); F.append(cap["v"].reshape(-1).numpy())
    return np.stack(F)
P = lambda items: [(a, b) for a, b, *_ in items]
R = lambda items: [(b, a) for a, b, *_ in items]
L = lambda items: [l for *_, l in items]
Y = lambda items, pos: np.array([l in pos for l in L(items)])
def split(p, labels):
    p = np.asarray(p); pos = np.isin(labels, ["update", "contradiction"]); r = {}
    for neg in ("compatible", "duplicate"):
        k = pos | (np.array(labels) == neg); r[f"vs_{neg}"] = round(auroc(p[k], pos[k]), 3)
    return r
def binrep(pe, ye, pa, ya):
    t = best_thr(pa, ya); ta = tau_auto(pa, ya)
    return {"auroc": round(auroc(pe, ye), 3), "auroc_ci95": boot(lambda p, y: auroc(p, y), pe, ye), "adapt_auroc": round(auroc(pa, ya), 3),
            "thr_from_adapt": round(float(t), 3), "acc@thr": round(acc_at(pe, ye, t), 3), "acc@thr_ci95": boot(lambda p, y: acc_at(p, y, t), pe, ye),
            "auto_on_eval(tau from adapt, alpha=.02)": auto_stats(pe, ye, ta)}
res = {"device": dev, "model": "tasksource/tasksource-decider-nano (main/v4)", "dtype": "f32"}
SUPPOS = ("update", "contradiction")
# supersedes: NLI P(contradiction)
pe, pa = probs(P(SUPERSEDE), "nli")[:, 2], probs(P(ADAPT), "nli")[:, 2]
res["supersedes"] = binrep(pe, Y(SUPERSEDE, SUPPOS), pa, Y(ADAPT, SUPPOS)) | {"eval_split": split(pe, L(SUPERSEDE))}
print("supersedes", res["supersedes"], flush=True)
# contradicts: symmetric NLI P(contradiction)
ce = (probs(P(SUPERSEDE), "nli")[:, 2] + probs(R(SUPERSEDE), "nli")[:, 2]) / 2
ca = (pa + probs(R(ADAPT), "nli")[:, 2]) / 2
res["contradicts(gold=update|contradiction)"] = binrep(ce, Y(SUPERSEDE, SUPPOS), ca, Y(ADAPT, SUPPOS))
print("contradicts", res["contradicts(gold=update|contradiction)"], flush=True)
# duplicate: symmetric NLI P(entailment)
de = (probs(P(SUPERSEDE), "nli")[:, 0] + probs(R(SUPERSEDE), "nli")[:, 0]) / 2
da = (probs(P(ADAPT), "nli")[:, 0] + probs(R(ADAPT), "nli")[:, 0]) / 2
res["duplicate"] = binrep(de, Y(SUPERSEDE, ("duplicate",)), da, Y(ADAPT, ("duplicate",)))
print("duplicate", res["duplicate"], flush=True)
# same_entity: symmetric (no adapt items: threshold 0.5, exploratory)
ee = (probs(P(ENTITY), "ent")[:, 1] + probs(R(ENTITY), "ent")[:, 1]) / 2
ye = np.array([l == "same" for l in L(ENTITY)])
res["same_entity(no adapt set)"] = {"auroc": round(auroc(ee, ye), 3), "auroc_ci95": boot(lambda p, y: auroc(p, y), ee, ye), "acc@0.5": round(acc_at(ee, ye, 0.5), 3)}
print("same_entity", res["same_entity(no adapt set)"], flush=True)
# relation: CNLI 3-way + prior bias fitted on adapt; 4-way via probe
lab3 = np.array(["duplicate", "update", "compatible"])
g3 = lambda items: np.array(["update" if l == "contradiction" else l for l in L(items)])
Pe, Pa = probs(P(SUPERSEDE), "cnli"), probs(P(ADAPT), "cnli")
best = (-1, np.zeros(3)); grid = np.linspace(-3, 3, 25)
for b1 in grid:
    for b2 in grid:
        bias = np.array([0.0, b1, b2]); a = np.mean(lab3[(np.log(Pa + 1e-9) + bias).argmax(1)] == g3(ADAPT))
        if a > best[0] + 1e-9: best = (a, bias)
pred0 = lab3[Pe.argmax(1)]; pred1 = lab3[(np.log(Pe + 1e-9) + best[1]).argmax(1)]
res["relation3"] = {"acc": round(float(np.mean(pred0 == g3(SUPERSEDE))), 3), "adapt_acc": round(float(np.mean(lab3[Pa.argmax(1)] == g3(ADAPT))), 3),
                    "bias(adapt)": [round(float(x), 2) for x in best[1]], "acc_bias": round(float(np.mean(pred1 == g3(SUPERSEDE))), 3),
                    "acc_bias_ci95": boot(lambda p, y: float(np.mean(p == y)), pred1, g3(SUPERSEDE)),
                    "counts_bias": {k: int((pred1 == k).sum()) for k in lab3}}
print("relation3", res["relation3"], flush=True)
# probes on features (fit on ADAPT only, C by 5-fold CV on ADAPT)
def probe(Xa, ya, Xe, ye_, multi=False):
    bc, bcv = None, -1
    for C in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        cv = cross_val_predict(clf, Xa, ya, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba" if not multi else "predict")
        s = auroc(cv[:, 1], ya) if not multi else float(np.mean(cv == ya))
        if s > bcv: bc, bcv = C, s
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=bc, max_iter=5000)).fit(Xa, ya)
    if multi:
        pr = clf.predict(Xe); return {"C": bc, "adapt_cv_acc": round(bcv, 3), "acc": round(float(np.mean(pr == ye_)), 3), "acc_ci95": boot(lambda p, y: float(np.mean(p == y)), pr, ye_)}, None
    cvp = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=bc, max_iter=5000)), Xa, ya, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    pe_ = clf.predict_proba(Xe)[:, 1]
    return {"C": bc, "adapt_cv_auroc": round(bcv, 3), **binrep(pe_, ye_, cvp, ya)}, pe_
Xa_n, Xe_n = feats(P(ADAPT), "nli"), feats(P(SUPERSEDE), "nli")
r, pp = probe(Xa_n, Y(ADAPT, SUPPOS), Xe_n, Y(SUPERSEDE, SUPPOS)); r["eval_split"] = split(pp, L(SUPERSEDE))
res["probe_supersedes(nli feats)"] = r; print("probe sup", r, flush=True)
Xa_c, Xe_c = feats(P(ADAPT), "cnli"), feats(P(SUPERSEDE), "cnli")
res["probe_relation4(cnli feats)"], _ = probe(Xa_c, np.array(L(ADAPT)), Xe_c, np.array(L(SUPERSEDE)), multi=True)
res["probe_relation3(cnli feats)"], _ = probe(Xa_c, g3(ADAPT), Xe_c, g3(SUPERSEDE), multi=True)
print("probe rel", res["probe_relation4(cnli feats)"], res["probe_relation3(cnli feats)"], flush=True)
hk.remove()
# VitaminC: old note = claim, new statement = revision evidence; REFUTES = positive (300, 100 per label, real revisions)
vc = [json.loads(l) for l in open(glob.glob("/Volumes/SSD01/caches/huggingface/hub/datasets--tals--vitaminc/snapshots/*/test.jsonl")[0])]
vc = [x for x in vc if x["revision_type"] == "real"]
random.Random(0).shuffle(vc)
pick = []
for labn in ("REFUTES", "SUPPORTS", "NOT ENOUGH INFO"):
    pick += [x for x in vc if x["label"] == labn][:100]
vp = probs([(x["claim"], x["evidence"]) for x in pick], "nli")[:, 2]
vy = np.array([x["label"] == "REFUTES" for x in pick]); vl = np.array([x["label"] for x in pick])
res["vitaminc300_supersedes"] = {"auroc": round(auroc(vp, vy), 3), "auroc_ci95": boot(lambda p, y: auroc(p, y), vp, vy),
    "vs_supports": round(auroc(vp[vl != "NOT ENOUGH INFO"], vy[vl != "NOT ENOUGH INFO"]), 3), "vs_nei": round(auroc(vp[vl != "SUPPORTS"], vy[vl != "SUPPORTS"]), 3),
    "acc@adapt_thr": round(acc_at(vp, vy, res["supersedes"]["thr_from_adapt"]), 3),
    "auto_at_adapt_tau": auto_stats(vp, vy, res["supersedes"]["auto_on_eval(tau from adapt, alpha=.02)"]["tau_auto"])}
print("vitaminc", res["vitaminc300_supersedes"], flush=True)
# latency: supersedes request (3 options), per decision
sync = (lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: None)
rs = reqs(P(SUPERSEDE), "nli")
m.score_many(rs[:4]); lat = []
for r_ in rs:
    sync(); t = time.perf_counter(); m.score_many([r_]); sync(); lat.append((time.perf_counter() - t) * 1000)
lat.sort()
sync(); t = time.perf_counter()
for i in range(0, len(rs), 16): m.score_many(rs[i:i + 16])
sync(); bt = (time.perf_counter() - t) * 1000 / len(rs)
res["latency_ms"] = {"b1_p50": round(lat[len(lat) // 2], 1), "b1_p90": round(lat[int(len(lat) * .9)], 1), "batched16_per_decision": round(bt, 2)}
res["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 0)
print("latency", res["latency_ms"], "rss", res["max_rss_mb"], flush=True)
json.dump(res, open(out_path, "w"), indent=1)
