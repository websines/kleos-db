"""D1: d1-omni-600M on memory-operation decisions. Usage: run_d1.py DEVICE DTYPE [quant]"""
import gc, json, os, sys, time, resource
import torch
sys.path.insert(0, os.path.dirname(__file__))
from eval_set import SUPERSEDE, ENTITY
from metrics import binary_report, acc

device, dtype_s = sys.argv[1], sys.argv[2]
quant = len(sys.argv) > 3 and sys.argv[3] == "q8"
dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[dtype_s]
if device == "mps":
    torch.mps.set_per_process_memory_fraction(0.25)  # hard cap ~4 GB of 16 GB
torch.set_num_threads(4)
from transformers import AutoModel
t0 = time.time()
model = AutoModel.from_pretrained("LiquidAI/d1-omni-600M", trust_remote_code=True, dtype=dtype)
del model.vision, model.audio  # text-only use: drop the 206M media encoders
gc.collect()
if quant:
    torch.backends.quantized.engine = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else torch.backends.quantized.engine
    model.encoder = torch.ao.quantization.quantize_dynamic(model.encoder, {torch.nn.Linear}, dtype=torch.qint8)
    model.head = torch.ao.quantization.quantize_dynamic(model.head, {torch.nn.Linear}, dtype=torch.qint8)
model = model.to(device).eval()
load_s = time.time() - t0
_ = model.tokenizer

SUP_Q = {
    "A": {"type": "noul", "instructions": "The state holds an OLD memory and a NEW memory about the same user. Does the NEW memory replace, update or correct the OLD memory, so that the OLD memory is no longer true?"},
    "B": {"type": "noul", "instructions": "Should the old fact be marked as superseded (no longer current) because of the new fact?",
          "criteria": {"true": "the new fact changes or corrects the old fact, so the old fact is outdated",
                       "false": "the old fact is still true: the new fact repeats it or adds unrelated information"}},
}
REL_Q = {"type": "choice", "instructions": "How does the NEW memory relate to the OLD memory?",
         "criteria": {"update": "The new memory changes or corrects the old one; the old memory is no longer true",
                      "duplicate": "The new memory says the same thing as the old one, in other words",
                      "compatible": "Both memories are true at the same time; the new one adds different information"}}
ENT_Q = {
    "A": {"type": "noul", "instructions": "Do the two mentions refer to the same real-world entity (the same person, place, organisation or thing)?"},
    "B": {"type": "noul", "instructions": "Are mention A and mention B the same entity?",
          "criteria": {"true": "both mentions refer to one and the same entity", "false": "they are different entities that may share a name"}},
}

def sup_state(a, b): return {"old_memory": a, "new_memory": b}
def ent_state(a, b): return {"mention_a": a, "mention_b": b}

def sync():
    if device == "mps": torch.mps.synchronize()

out = {"device": device, "dtype": dtype_s, "quant": "q8-dynamic" if quant else None, "load_s": round(load_s, 2)}
# warm-up
model.system_one(sup_state("a", "b"), {"x": SUP_Q["A"]}); sync()

sup_y = [lab in ("update", "contradiction") for _, _, lab in SUPERSEDE]
for k, q in SUP_Q.items():
    reqs = [(sup_state(a, b), {"q": q}) for a, b, _ in SUPERSEDE]
    res = model.system_one_batch(reqs)
    p = [r["answers"]["q"]["noul"] for r in res]
    out[f"supersede_{k}"] = binary_report(p, sup_y)
    out[f"supersede_{k}_probs"] = [round(x, 3) for x in p]
reqs = [(sup_state(a, b), {"q": REL_Q}) for a, b, _ in SUPERSEDE]
res = model.system_one_batch(reqs)
gold3 = [{"contradiction": "update"}.get(l, l) for _, _, l in SUPERSEDE]
pred3 = [r["answers"]["q"]["choice"] for r in res]
out["relation3_acc"] = acc(pred3, gold3)
out["relation3_pred"] = pred3
ent_y = [lab == "same" for _, _, lab in ENTITY]
for k, q in ENT_Q.items():
    reqs = [(ent_state(a, b), {"q": q}) for a, b, _ in ENTITY]
    res = model.system_one_batch(reqs)
    p = [r["answers"]["q"]["noul"] for r in res]
    out[f"entity_{k}"] = binary_report(p, ent_y)
    out[f"entity_{k}_probs"] = [round(x, 3) for x in p]

# latency: one decision per call (batch 1), median over the 66 items
lat = []
items = [(sup_state(a, b), {"q": SUP_Q["A"]}) for a, b, _ in SUPERSEDE] + [(ent_state(a, b), {"q": ENT_Q["A"]}) for a, b, _ in ENTITY]
for st, qs in items:
    sync(); t = time.perf_counter(); model.system_one(st, qs); sync(); lat.append((time.perf_counter() - t) * 1000)
lat.sort()
out["lat_b1_ms_p50"] = round(lat[len(lat) // 2], 1)
out["lat_b1_ms_p90"] = round(lat[int(len(lat) * 0.9)], 1)
sync(); t = time.perf_counter(); model.system_one_batch(items); sync()
out["batched_66_ms_per_decision"] = round((time.perf_counter() - t) * 1000 / len(items), 2)
out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
if device == "mps":
    out["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 1e6)
print(json.dumps(out))
