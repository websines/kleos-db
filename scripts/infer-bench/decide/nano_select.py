"""Pick decider-nano templates on the 60 ADAPT items only (eval items reported as exploratory)."""
import json, sys, glob, os
import numpy as np
sys.path.insert(0, "/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr/kleos/scripts/infer-bench/d1")
snap = glob.glob("/Volumes/SSD01/caches/huggingface/hub/models--tasksource--tasksource-decider-nano/snapshots/*")[0]
sys.path.insert(0, snap)
import torch
torch.set_num_threads(4)
from eval_set import SUPERSEDE, ENTITY
from eval_set_adapt import ADAPT
from decider import Decider, format_option
from pathlib import Path
m = Decider(Path(snap), device="cpu")

def auroc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, bool)
    pos, neg = p[y], p[~y]
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())

def run(items, state, q, opts):
    reqs = [(state(a, b), q(a, b), opts) for a, b, *_ in items]
    out = []
    for i in range(0, len(reqs), 16):
        for z in m.score_many(reqs[i:i+16]):
            e = np.exp(z - z.max()); out.append(e / e.sum())
    return np.array(out)

CNLI = [format_option(k, v) for k, v in {"Entailment": "The old note implies the new statement is true",
        "Contradiction": "The old note and the new statement cannot both be true",
        "NotMentioned": "The old note neither implies nor contradicts the new statement"}.items()]
NLI = [format_option(k, v) for k, v in {"entailment": "The hypothesis follows from the premise",
       "neutral": "The hypothesis might be true or false given the premise",
       "contradiction": "The hypothesis contradicts the premise"}.items()]
REL4 = [format_option(k, v) for k, v in {
    "update": "The new memory reports a change: the old memory was true before but is no longer current",
    "contradiction": "The new memory says the old memory is wrong, or both cannot be true",
    "compatible": "Both memories are true at the same time; the new one adds different information",
    "duplicate": "The new memory says the same thing as the old one, in other words"}.items()]
NOUL = ["false", "true"]
cnli_q = lambda a, b: f"Classify the relationship between the old note and this new statement:\n{b}"
V = {
 "cnli": (lambda a, b: a, cnli_q, CNLI),
 "nli_text": (lambda a, b: f"Premise: {a}\nHypothesis: {b}", lambda a, b: "What is the relationship between the premise and the hypothesis?", NLI),
 "still_true": (lambda a, b: f"Earlier: {a}\nLater: {b}", lambda a, b: "Given the later statement, is the earlier statement still true?", NOUL),
 "superseded_noul": (lambda a, b: f"Old memory: {a}\nNew memory: {b}", lambda a, b: "Does the new memory make the old memory no longer true?", 
                     [format_option("false", "the old memory is still true"), format_option("true", "the old memory is no longer true")]),
 "rel4_text": (lambda a, b: f"Old memory: {a}\nNew memory: {b}", lambda a, b: "How does the new memory relate to the old memory?", REL4),
 "same_noul": (lambda a, b: f"Statement A: {a}\nStatement B: {b}", lambda a, b: "Do statement A and statement B say the same thing?",
               [format_option("false", "they state different facts"), format_option("true", "they state the same fact in different words")]),
 "both_true": (lambda a, b: f"Statement A: {a}\nStatement B: {b}", lambda a, b: "Can statement A and statement B both be true at the same time?", NOUL),
}
# readouts: name -> (variant, fn(P)->score)
SUP = {"cnli_contra": ("cnli", lambda P: P[:, 1]), "nli_contra": ("nli_text", lambda P: P[:, 2]),
       "still_true_inv": ("still_true", lambda P: P[:, 0]), "superseded_noul": ("superseded_noul", lambda P: P[:, 1]),
       "rel4_upd+con": ("rel4_text", lambda P: P[:, 0] + P[:, 1])}
DUP = {"cnli_ent": ("cnli", lambda P: P[:, 0]), "nli_ent": ("nli_text", lambda P: P[:, 0]), "same_noul": ("same_noul", lambda P: P[:, 1]),
       "rel4_dup": ("rel4_text", lambda P: P[:, 3])}
CON = {"cnli_contra": ("cnli", lambda P: P[:, 1]), "both_true_inv": ("both_true", lambda P: P[:, 0]), "nli_contra": ("nli_text", lambda P: P[:, 2])}
res = {}
cache = {}
def probs(items, name, swap=False):
    key = (id(items), name, swap)
    if key not in cache:
        st, q, o = V[name]
        it = [(b, a) for a, b, *_ in items] if swap else items
        cache[key] = run(it, st, q, o)
    return cache[key]
lab = lambda items: [l for *_, l in items]
for task, table, ypos, sym in [("supersedes", SUP, ("update", "contradiction"), False), ("duplicate", DUP, ("duplicate",), True),
                               ("contradicts", CON, ("update", "contradiction"), True)]:
    for rname, (vn, f) in table.items():
        r = {}
        for split, items in [("adapt", ADAPT), ("eval", SUPERSEDE)]:
            s = f(probs(items, vn))
            if sym:
                s = (s + f(probs(items, vn, True))) / 2
            y = [l in ypos for l in lab(items)]
            r[split] = round(auroc(s, y), 3)
        res[f"{task}/{rname}"] = r
        print(task, rname, r, flush=True)
# relation 4-way accuracy (argmax), cnli mapped 3-way
G = lambda items: np.array(lab(items))
L4 = np.array(["update", "contradiction", "compatible", "duplicate"])
for split, items in [("adapt", ADAPT), ("eval", SUPERSEDE)]:
    P = probs(items, "rel4_text"); g = G(items)
    print("relation rel4 4-way acc", split, round(float(np.mean(L4[P.argmax(1)] == g)), 3), "3-way(merge upd/con)",
          round(float(np.mean(np.where(np.isin(L4[P.argmax(1)], ["update","contradiction"]), "u", L4[P.argmax(1)]) == np.where(np.isin(g, ["update","contradiction"]), "u", g))), 3))
    P = probs(items, "cnli"); m3 = np.array(["duplicate", "u", "compatible"])
    print("relation cnli 3-way acc", split, round(float(np.mean(m3[P.argmax(1)] == np.where(np.isin(g, ["update","contradiction"]), "u", g))), 3))
# entity (eval only)
ENT = {"E1_B_text": (lambda a, b: f"Mention A: {a}\nMention B: {b}", lambda a, b: "Are mention A and mention B the same entity?",
        [format_option("false", "they are different entities that may share a name"), format_option("true", "both mentions refer to one and the same entity")])}
for n, (st, q, o) in ENT.items():
    p1 = run(ENTITY, st, q, o)[:, 1]; p2 = run([(b, a, l) for a, b, l in ENTITY], st, q, o)[:, 1]
    y = [l == "same" for *_, l in ENTITY]
    print("same_entity", n, "eval", round(auroc(p1, y), 3), "sym", round(auroc((p1 + p2) / 2, y), 3))
json.dump(res, open(sys.argv[1], "w"), indent=1)
