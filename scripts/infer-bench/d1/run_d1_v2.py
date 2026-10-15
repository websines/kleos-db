"""D1 v2: corrected / audited evaluation of d1-omni-600M on the 66 memory-decision items.

Protocol (no tuning on the eval set):
  * every prompt variant is scored on the 60 ADAPT items (eval_set_adapt.py) and on the 42+24 EVAL items;
  * the "selected" variant / threshold / probe is chosen on ADAPT only, then reported on EVAL;
  * all other variants are also reported on EVAL, but flagged as exploratory.
Model loaded as the model card recommends for a GPU: float16 (media encoders dropped: no effect on text, see d1_numerics.py).

Usage: run_d1_v2.py OUT.json [device] [dtype]
"""
import json, os, random, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from eval_set import SUPERSEDE, ENTITY
from eval_set_adapt import ADAPT

out_path = sys.argv[1]
device = sys.argv[2] if len(sys.argv) > 2 else "mps"
dtype = {"f32": torch.float32, "f16": torch.float16}[sys.argv[3] if len(sys.argv) > 3 else "f16"]
if device == "mps":
    torch.mps.set_per_process_memory_fraction(0.25)
torch.set_num_threads(4)
from transformers import AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict

model = AutoModel.from_pretrained("LiquidAI/d1-omni-600M", trust_remote_code=True, dtype=dtype)
del model.vision, model.audio  # verified bit-identical on text (d1_numerics.py); frees ~0.4 GB under the MPS cap
model = model.to(device).eval()
_ = model.tokenizer

# ------------------------------------------------------------------ metrics
rng = np.random.default_rng(0)


def auroc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, bool)
    pos, neg = p[y], p[~y]
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())


def boot_ci(stat, *arrs, y=None, B=2000):
    """Stratified (by y) bootstrap 95% CI."""
    y = np.asarray(y)
    idx_by = [np.where(y == c)[0] for c in np.unique(y)]
    vals = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(ix, len(ix)) for ix in idx_by])
        vals.append(stat(*[np.asarray(a)[idx] for a in arrs], y[idx]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def acc_at(p, y, t):
    return float(((np.asarray(p) >= t) == np.asarray(y, bool)).mean())


def best_thr(p, y):
    cands = sorted(set(np.round(p, 6))) + [1.01]
    return max(cands, key=lambda t: (acc_at(p, y, t), -abs(t - 0.5)))


def binrep(p_eval, y_eval, p_adapt=None, y_adapt=None):
    r = {"auroc": round(auroc(p_eval, y_eval), 3),
         "auroc_ci95": boot_ci(lambda p, y: auroc(p, y), p_eval, y=y_eval),
         "acc@0.5": round(acc_at(p_eval, y_eval, 0.5), 3),
         "p_range": [round(float(min(p_eval)), 3), round(float(max(p_eval)), 3)]}
    if p_adapt is not None:
        t = best_thr(np.asarray(p_adapt), y_adapt)
        r["adapt_auroc"] = round(auroc(p_adapt, y_adapt), 3)
        r["thr_from_adapt"] = round(float(t), 3)
        r["acc@thr_from_adapt"] = round(acc_at(p_eval, y_eval, t), 3)
        r["acc@thr_ci95"] = boot_ci(lambda p, y: acc_at(p, y, t), p_eval, y=y_eval)
    return r


def split_auroc(p, labels):
    """AUROC of positives vs each negative type."""
    p = np.asarray(p)
    pos = np.array([l in ("update", "contradiction") for l in labels])
    res = {}
    for neg in ("compatible", "duplicate"):
        m = pos | (np.array(labels) == neg)
        res[f"vs_{neg}"] = round(auroc(p[m], pos[m]), 3)
    return res


# ------------------------------------------------------------------ prompts
def st_json(a, b): return {"old_memory": a, "new_memory": b}
def st_text(a, b): return f"Earlier: {a}\nLater: {b}"
def st_nli_json(a, b): return {"premise": a, "hypothesis": b}
def st_nli_text(a, b): return f"Premise: {a}\nHypothesis: {b}"


# ContractNLI pattern from Liquid's Decision Index runs: state = the reference text, the hypothesis in the instructions
CNLI_CRIT = {"Entailment": "The old note implies the new statement is true",
             "Contradiction": "The old note and the new statement cannot both be true",
             "NotMentioned": "The old note neither implies nor contradicts the new statement"}
def q_cnli(b): return {"type": "choice", "instructions": f"Classify the relationship between the old note and this new statement:\n{b}", "criteria": CNLI_CRIT}


SUP_A = {"type": "noul", "instructions": "The state holds an OLD memory and a NEW memory about the same user. Does the NEW memory replace, update or correct the OLD memory, so that the OLD memory is no longer true?"}
SUP_B = {"type": "noul", "instructions": "Should the old fact be marked as superseded (no longer current) because of the new fact?",
         "criteria": {"true": "the new fact changes or corrects the old fact, so the old fact is outdated",
                      "false": "the old fact is still true: the new fact repeats it or adds unrelated information"}}
SUP_CONTRA = {"type": "noul", "instructions": "Does the later statement contradict the earlier statement, so that the earlier statement is no longer true?",
              "criteria": {"true": "the earlier statement is no longer true", "false": "both statements can be true at the same time"}}
SUP_STILL = {"type": "noul", "instructions": "Given the later statement, is the earlier statement still true?"}
NLI = {"type": "choice", "instructions": "What is the relationship between the premise and the hypothesis?",
       "criteria": {"entailment": "The hypothesis follows from the premise",
                    "neutral": "The hypothesis might be true or false given the premise",
                    "contradiction": "The hypothesis contradicts the premise"}}
REL_Q = {"type": "choice", "instructions": "How does the NEW memory relate to the OLD memory?",
         "criteria": {"update": "The new memory changes or corrects the old one; the old memory is no longer true",
                      "duplicate": "The new memory says the same thing as the old one, in other words",
                      "compatible": "Both memories are true at the same time; the new one adds different information"}}

# name -> (state fn, question, how to read P(supersede))
noul_p = lambda a: a["noul"]  # noqa: E731
SUP_VARIANTS = {
    "S0_old_A_json": (st_json, SUP_A, noul_p),
    "S0_old_B_json": (st_json, SUP_B, noul_p),
    "S1_A_text": (st_text, SUP_A, noul_p),
    "S2_contra_text": (st_text, SUP_CONTRA, noul_p),
    "S2_contra_json": (st_json, SUP_CONTRA, noul_p),
    "S3_still_true_text(inv)": (st_text, SUP_STILL, lambda a: 1 - a["noul"]),
    "S4_nli_json_P(contra)": (st_nli_json, NLI, lambda a: a["probabilities"]["contradiction"]),
    "S5_nli_text_P(contra)": (st_nli_text, NLI, lambda a: a["probabilities"]["contradiction"]),
    "S8_contractnli_P(contra)": (lambda a, b: a, "CNLI", lambda a: a["probabilities"]["Contradiction"]),
    "S6_rel3_json_P(update)": (st_json, REL_Q, lambda a: a["probabilities"]["update"]),
    "S7_rel3_text_P(update)": (st_text, REL_Q, lambda a: a["probabilities"]["update"]),
}
REL_VARIANTS = {  # 3-way: name -> (state fn, question, map model option -> gold label)
    "R0_old_rel3_json": (st_json, REL_Q, {"update": "update", "duplicate": "duplicate", "compatible": "compatible"}),
    "R1_rel3_text": (st_text, REL_Q, {"update": "update", "duplicate": "duplicate", "compatible": "compatible"}),
    "R2_nli_json": (st_nli_json, NLI, {"contradiction": "update", "entailment": "duplicate", "neutral": "compatible"}),
    "R3_nli_text": (st_nli_text, NLI, {"contradiction": "update", "entailment": "duplicate", "neutral": "compatible"}),
    "R4_contractnli": (lambda a, b: a, "CNLI", {"Contradiction": "update", "Entailment": "duplicate", "NotMentioned": "compatible"}),
}
ENT_A = {"type": "noul", "instructions": "Do the two mentions refer to the same real-world entity (the same person, place, organisation or thing)?"}
ENT_B = {"type": "noul", "instructions": "Are mention A and mention B the same entity?",
         "criteria": {"true": "both mentions refer to one and the same entity", "false": "they are different entities that may share a name"}}
ENT_VARIANTS = {
    "E0_old_A_json": (lambda a, b: {"mention_a": a, "mention_b": b}, ENT_A),
    "E0_old_B_json": (lambda a, b: {"mention_a": a, "mention_b": b}, ENT_B),
    "E1_A_text": (lambda a, b: f"Mention A: {a}\nMention B: {b}", ENT_A),
    "E1_B_text": (lambda a, b: f"Mention A: {a}\nMention B: {b}", ENT_B),
}

GOLD3 = lambda items: [{"contradiction": "update"}.get(l, l) for *_, l in items]  # noqa: E731
Y = lambda items: [l in ("update", "contradiction") for *_, l in items]  # noqa: E731
LAB = lambda items: [l for *_, l in items]  # noqa: E731


def answers(items, sfn, q):
    reqs = [(sfn(a, b), {"q": q_cnli(b) if q == "CNLI" else q}) for a, b, *_ in items]
    res = [r for i in range(0, len(reqs), 16) for r in model.system_one_batch(reqs[i:i + 16])]  # fits the 0.25 MPS cap
    return [r["answers"]["q"] for r in res]


out = {"device": device, "dtype": str(dtype), "n_eval_sup": len(SUPERSEDE), "n_adapt": len(ADAPT), "n_eval_ent": len(ENTITY)}
ye, ya = Y(SUPERSEDE), Y(ADAPT)

# ---------------------------------------------------------------- 1. zero-shot supersession variants
sup = {}
scores_eval, scores_adapt = {}, {}
for name, (sfn, q, read) in SUP_VARIANTS.items():
    pe = [read(a) for a in answers(SUPERSEDE, sfn, q)]
    pa = [read(a) for a in answers(ADAPT, sfn, q)]
    scores_eval[name], scores_adapt[name] = pe, pa
    sup[name] = binrep(pe, ye, pa, ya) | {"eval_split": split_auroc(pe, LAB(SUPERSEDE)),
                                         "adapt_split": split_auroc(pa, LAB(ADAPT))}
    print(name, sup[name], flush=True)
out["supersede_zero_shot"] = sup
sel = max((k for k in SUP_VARIANTS if not k.startswith("S0")), key=lambda k: sup[k]["adapt_auroc"])
out["supersede_selected_on_adapt"] = {"variant": sel, **sup[sel]}
out["supersede_eval_probs"] = {k: [round(x, 4) for x in v] for k, v in scores_eval.items()}

# ---------------------------------------------------------------- 2. 3-way relation (+ prior correction fit on ADAPT)
rel = {}
for name, (sfn, q, mp) in REL_VARIANTS.items():
    opts = list((CNLI_CRIT if q == "CNLI" else q["criteria"]))
    ae, aa = answers(SUPERSEDE, sfn, q), answers(ADAPT, sfn, q)
    Pe = np.array([[a["probabilities"][o] for o in opts] for a in ae])
    Pa = np.array([[a["probabilities"][o] for o in opts] for a in aa])
    ge, ga = GOLD3(SUPERSEDE), GOLD3(ADAPT)
    lab = [mp[o] for o in opts]
    pred_e = [lab[i] for i in Pe.argmax(1)]
    pred_a = [lab[i] for i in Pa.argmax(1)]
    # prior correction: additive log-bias per option, grid-searched on ADAPT for accuracy
    best = (-1, np.zeros(3))
    grid = np.linspace(-3, 3, 25)
    for b1 in grid:
        for b2 in grid:
            bias = np.array([0.0, b1, b2])
            ac = np.mean([lab[i] == g for i, g in zip((np.log(Pa + 1e-9) + bias).argmax(1), ga)])
            if ac > best[0] + 1e-9:
                best = (ac, bias)
    pc_e = [lab[i] for i in (np.log(Pe + 1e-9) + best[1]).argmax(1)]
    acc = lambda p, g: float(np.mean([a == b for a, b in zip(p, g)]))  # noqa: E731
    rel[name] = {"acc": round(acc(pred_e, ge), 3),
                 "acc_ci95": boot_ci(lambda p, y: float(np.mean(p == y)), np.array(pred_e), y=np.array(ge)),
                 "pred_counts": {l: pred_e.count(l) for l in set(lab)},
                 "adapt_acc": round(acc(pred_a, ga), 3),
                 "prior_corrected_acc": round(acc(pc_e, ge), 3),
                 "prior_corrected_acc_ci95": boot_ci(lambda p, y: float(np.mean(p == y)), np.array(pc_e), y=np.array(ge)),
                 "prior_corrected_counts": {l: pc_e.count(l) for l in set(lab)},
                 "prior_bias(adapt)": [round(float(x), 2) for x in best[1]], "adapt_acc_after_bias": round(float(best[0]), 3)}
    print(name, rel[name], flush=True)
out["relation3"] = rel
selr = max((k for k in REL_VARIANTS if not k.startswith("R0")), key=lambda k: rel[k]["adapt_acc"])
out["relation3_selected_on_adapt"] = {"variant": selr, **rel[selr]}

# ---------------------------------------------------------------- 3. entity (no adapt set: exploratory only)
ent, ye_ent = {}, [l == "same" for *_, l in ENTITY]
for name, (sfn, q) in ENT_VARIANTS.items():
    p = [a["noul"] for a in answers(ENTITY, sfn, q)]
    ent[name] = binrep(p, ye_ent)
    print(name, ent[name], flush=True)
out["entity"] = ent

# ---------------------------------------------------------------- 4. in-context examples in the state (few-shot)
shots = [ADAPT[i] for i in (0, 18, 24, 42, 1, 25, 43, 19)]  # 3 update/contra, 2 compatible, 2 duplicate + 1 contradiction
random.Random(0).shuffle(shots)


def st_fewshot(a, b):
    ex = "\n\n".join(f"Earlier: {x}\nLater: {y}\nEarlier statement still true: {'no' if l in ('update', 'contradiction') else 'yes'}"
                     for x, y, l in shots)
    return f"Solved examples:\n\n{ex}\n\nNow decide this case:\nEarlier: {a}\nLater: {b}"


fs = {}
for name, q, read in [("F_contra_text_8shot", SUP_CONTRA, noul_p), ("F_still_true_8shot(inv)", SUP_STILL, lambda a: 1 - a["noul"])]:
    pe = [read(a) for a in answers(SUPERSEDE, st_fewshot, q)]
    rest = [x for x in ADAPT if x not in shots]
    pa = [read(a) for a in answers(rest, st_fewshot, q)]
    fs[name] = binrep(pe, ye, pa, Y(rest)) | {"eval_split": split_auroc(pe, LAB(SUPERSEDE))}
    print(name, fs[name], flush=True)
out["supersede_fewshot_in_state"] = fs

# ---------------------------------------------------------------- 5. head/probe adaptation on frozen d1 features (fit on ADAPT only)
cap = {}
hook = model.head.scorer.register_forward_hook(lambda m, i, o: cap.__setitem__("g", i[0].detach().float().cpu()))


def feats(items, sfn, q):
    F = []
    for a, b, *_ in items:  # one call per item so the hooked batch order is the item order
        model.system_one(sfn(a, b), {"q": q})
        g = cap["g"][0]  # (K markers, d): hidden state at each option marker, input of the scorer MLP
        F.append(g[: (2 if q["type"] == "noul" else len(q["criteria"]))].reshape(-1).numpy())
    return np.stack(F)


probe = {}
for name in ("S0_old_A_json", "S2_contra_text", "S4_nli_json_P(contra)"):
    sfn, q, _ = SUP_VARIANTS[name]
    Xa, Xe = feats(ADAPT, sfn, q), feats(SUPERSEDE, sfn, q)
    ya_np, ye_np = np.array(ya), np.array(ye)
    # choose C by 5-fold CV on ADAPT only
    best_c, best_cv = None, -1
    for C in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        cvp = cross_val_predict(clf, Xa, ya_np, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        a_ = auroc(cvp, ya_np)
        if a_ > best_cv:
            best_c, best_cv = C, a_
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=5000)).fit(Xa, ya_np)
    pe = clf.predict_proba(Xe)[:, 1]
    probe[name] = {"C": best_c, "adapt_cv_auroc": round(best_cv, 3), **binrep(pe, ye),
                   "acc@0.5": round(acc_at(pe, ye, 0.5), 3), "eval_split": split_auroc(pe, LAB(SUPERSEDE))}
    # 3-way probe (multinomial) on the same features
    g3a, g3e = np.array(GOLD3(ADAPT)), np.array(GOLD3(SUPERSEDE))
    clf3 = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=5000)).fit(Xa, g3a)
    p3 = clf3.predict(Xe)
    probe[name]["rel3_acc"] = round(float(np.mean(p3 == g3e)), 3)
    probe[name]["rel3_acc_ci95"] = boot_ci(lambda p, y: float(np.mean(p == y)), p3, y=g3e)
    print("probe", name, probe[name], flush=True)
hook.remove()
out["supersede_probe_on_d1_features"] = probe

# ---------------------------------------------------------------- 6. latency of the selected prompt (batch 1 and batched)
json.dump(out, open(out_path, "w"), indent=1)  # checkpoint before latency
_q = lambda b: q_cnli(b) if SUP_VARIANTS[sel][1] == "CNLI" else SUP_VARIANTS[sel][1]  # noqa: E731
items = [(SUP_VARIANTS[sel][0](a, b), {"q": _q(b)}) for a, b, _ in SUPERSEDE]
sync = (lambda: torch.mps.synchronize()) if device == "mps" else (lambda: None)
lat = []
for s, q in items:
    sync(); t = time.perf_counter(); model.system_one(s, q); sync(); lat.append((time.perf_counter() - t) * 1000)
lat.sort()
sync(); t = time.perf_counter(); model.system_one_batch(items); sync()
out["latency"] = {"b1_p50_ms": round(lat[len(lat) // 2], 1), "b1_p90_ms": round(lat[int(len(lat) * 0.9)], 1),
                  "batched_ms_per_decision": round((time.perf_counter() - t) * 1000 / len(items), 2)}
print(out["latency"], flush=True)
json.dump(out, open(out_path, "w"), indent=1)
print("WROTE", out_path)
