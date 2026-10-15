"""D1 baseline: embedding cosine with thresholds (oracle-tuned on the set itself, i.e. optimistic)."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_set import SUPERSEDE, ENTITY
from metrics import binary_report
import numpy as np

def run(name, enc):
    out = {"model": name}
    def cos(pairs):
        a = enc([p[0] for p in pairs]); b = enc([p[1] for p in pairs])
        a = a / np.linalg.norm(a, axis=1, keepdims=True); b = b / np.linalg.norm(b, axis=1, keepdims=True)
        return (a * b).sum(1).tolist()
    cs = cos(SUPERSEDE)
    sup_y = [l in ("update", "contradiction") for _, _, l in SUPERSEDE]
    # cosine is a similarity, not a supersession score: report both directions' best threshold
    out["supersede_cos"] = binary_report(cs, sup_y)
    dup_y = [l == "duplicate" for _, _, l in SUPERSEDE]
    out["duplicate_cos"] = binary_report(cs, dup_y)
    # 3-way: two thresholds t1<t2: cos>=t2 duplicate, t1<=cos<t2 update, else compatible (oracle grid)
    gold = [{"contradiction": "update"}.get(l, l) for _, _, l in SUPERSEDE]
    best = 0
    grid = sorted(set(cs))
    for i, t1 in enumerate(grid):
        for t2 in grid[i:]:
            pred = ["duplicate" if c >= t2 else "update" if c >= t1 else "compatible" for c in cs]
            best = max(best, sum(p == g for p, g in zip(pred, gold)) / len(gold))
    out["relation3_oracle_acc"] = round(best, 3)
    ce = cos(ENTITY)
    out["entity_cos"] = binary_report(ce, [l == "same" for _, _, l in ENTITY])
    t = time.perf_counter(); enc([a for a, _, _ in SUPERSEDE + ENTITY]); out["ms_per_text_batched"] = round((time.perf_counter() - t) * 1000 / 66, 2)
    print(json.dumps(out))

which = sys.argv[1]
if which.startswith("potion"):
    from model2vec import StaticModel
    m = StaticModel.from_pretrained(f"minishlab/{which}")
    run(which, lambda xs: m.encode(xs))
else:
    import torch
    from sentence_transformers import SentenceTransformer
    torch.mps.set_per_process_memory_fraction(0.25)
    m = SentenceTransformer("google/embeddinggemma-300m", device="mps")
    run("embeddinggemma-300m", lambda xs: m.encode(xs, prompt_name="STS" if "STS" in m.prompts else None))
