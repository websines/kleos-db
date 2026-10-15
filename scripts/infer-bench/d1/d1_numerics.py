"""Numerics audit for d1-omni-600M: f32 CPU vs f32/f16/bf16 MPS, batch vs single, media encoders kept vs deleted.

One model in memory at a time (dtype/device converted in place). Usage: d1_numerics.py OUT.json
"""
import gc, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from eval_set import SUPERSEDE, ENTITY

torch.mps.set_per_process_memory_fraction(0.25)
torch.set_num_threads(4)
from transformers import AutoModel

SUP_A = {"type": "noul", "instructions": "The state holds an OLD memory and a NEW memory about the same user. Does the NEW memory replace, update or correct the OLD memory, so that the OLD memory is no longer true?"}
REL_Q = {"type": "choice", "instructions": "How does the NEW memory relate to the OLD memory?",
         "criteria": {"update": "The new memory changes or corrects the old one; the old memory is no longer true",
                      "duplicate": "The new memory says the same thing as the old one, in other words",
                      "compatible": "Both memories are true at the same time; the new one adds different information"}}
ENT_A = {"type": "noul", "instructions": "Do the two mentions refer to the same real-world entity (the same person, place, organisation or thing)?"}
REQS = ([({"old_memory": a, "new_memory": b}, [SUP_A, REL_Q]) for a, b, _ in SUPERSEDE]
        + [({"mention_a": a, "mention_b": b}, [ENT_A]) for a, b, _ in ENTITY])


def flat(probs):
    return [x for req in probs for q in req for x in q]


def run(model, single=False):
    if single:
        return [model.probabilities(s, qs) for s, qs in REQS]
    # chunked: the 0.25 MPS cap (~2.96 GiB) cannot hold f32 activations for all 108 rows in one forward
    return [r for i in range(0, len(REQS), 12) for r in model.probabilities_batch(REQS[i:i + 12])]


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


out = {}
model = AutoModel.from_pretrained("LiquidAI/d1-omni-600M", trust_remote_code=True, dtype=torch.float32).eval()
_ = model.tokenizer
ref = run(model)  # f32 CPU, batch, media encoders present (official default)
out["cpu_f32_batch_with_media"] = ref
del model.vision, model.audio
gc.collect()
nm = run(model)
out["delete_media_maxdiff"] = max(abs(a - b) for a, b in zip(flat(ref), flat(nm)))
print("delete media maxdiff", out["delete_media_maxdiff"], flush=True)

configs = [("mps", torch.float32), ("mps", torch.float16), ("mps", torch.bfloat16)]
for dev, dt in configs:
    model = model.to(device=dev, dtype=dt)
    run(model)  # warm-up
    sync(dev); t = time.perf_counter(); b = run(model); sync(dev); tb = time.perf_counter() - t
    sync(dev); t = time.perf_counter(); s = run(model, single=True); sync(dev); ts = time.perf_counter() - t
    key = f"{dev}_{str(dt).split('.')[-1]}"
    fb, fs, fr = flat(b), flat(s), flat(ref)
    # top-answer flips vs f32 reference (noul: p>=0.5 side; choice: argmax)
    def tops(p):
        r = []
        for req in p:
            for q in req:
                r.append(max(range(len(q)), key=q.__getitem__))
        return r
    out[key] = {
        "batch_vs_f32cpu_maxabs": max(abs(x - y) for x, y in zip(fb, fr)),
        "single_vs_f32cpu_maxabs": max(abs(x - y) for x, y in zip(fs, fr)),
        "batch_vs_single_maxabs": max(abs(x - y) for x, y in zip(fb, fs)),
        "top_flips_batch_vs_f32cpu": sum(x != y for x, y in zip(tops(b), tops(ref))),
        "top_flips_single_vs_batch": sum(x != y for x, y in zip(tops(b), tops(s))),
        "n_questions": len(tops(ref)),
        "batch_ms_per_q": round(tb * 1000 / len(tops(ref)), 2),
        "single_ms_per_req": round(ts * 1000 / len(REQS), 2),
        "any_nan": any(x != x for x in fb + fs),
    }
    if dt == torch.float16:  # the old harness's shape: all requests in one padded forward
        full = model.probabilities_batch(REQS)
        out[key]["fullbatch_vs_single_maxabs"] = max(abs(x - y) for x, y in zip(flat(full), fs))
        out[key]["fullbatch_vs_f32cpu_maxabs"] = max(abs(x - y) for x, y in zip(flat(full), fr))
    print(key, out[key], flush=True)
    out[key + "_probs_batch"] = b
json.dump(out, open(sys.argv[1], "w"))
