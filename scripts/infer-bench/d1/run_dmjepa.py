"""D1: DangerLabs DM-JEPA 1.1 (ModernBERT-base) on the same memory decisions. Usage: run_dmjepa.py DEVICE DTYPE"""
import glob, json, os, sys, time, resource
import torch
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from eval_set import SUPERSEDE, ENTITY
from metrics import binary_report, acc
snap = glob.glob(os.path.join(os.environ["HF_HOME"], "hub/models--DangerLabs--DM-JEPA/snapshots/*"))[0]
sys.path.append(snap)
from modeling_dm_jepa import DMJEPA
from transformers import AutoTokenizer

device, dtype_s = sys.argv[1], sys.argv[2]
dtype = {"f32": torch.float32, "f16": torch.float16}[dtype_s]
if device == "mps":
    torch.mps.set_per_process_memory_fraction(0.25)
torch.set_num_threads(4)
t0 = time.time()
model = DMJEPA.from_pretrained(snap, device="cpu").to(dtype).to(device).eval()
tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
load_s = time.time() - t0

def sync():
    if device == "mps": torch.mps.synchronize()

def decide(state, instructions, criteria, labels):
    rec = {"state": state, "question": {"instructions": instructions, "criteria": criteria}, "labels": labels}
    return model.model.decide(rec, tok, device=torch.device(device))

SUP = [
    ("A", "The state holds an OLD memory and a NEW memory about the same user. Does the NEW memory replace, update or correct the OLD memory, so that the OLD memory is no longer true?",
     {"false": "no, the statement does not hold", "true": "yes, the statement holds"}),
    ("B", "Should the old fact be marked as superseded (no longer current) because of the new fact?",
     {"true": "the new fact changes or corrects the old fact, so the old fact is outdated",
      "false": "the old fact is still true: the new fact repeats it or adds unrelated information"}),
]
ENT = [
    ("A", "Do the two mentions refer to the same real-world entity (the same person, place, organisation or thing)?",
     {"false": "no, the statement does not hold", "true": "yes, the statement holds"}),
    ("B", "Are mention A and mention B the same entity?",
     {"true": "both mentions refer to one and the same entity", "false": "they are different entities that may share a name"}),
]
REL = {"update": "The new memory changes or corrects the old one; the old memory is no longer true",
       "duplicate": "The new memory says the same thing as the old one, in other words",
       "compatible": "Both memories are true at the same time; the new one adds different information"}

out = {"model": "DM-JEPA 1.1", "device": device, "dtype": dtype_s, "load_s": round(load_s, 2)}
decide({"a": 1}, "warm up?", SUP[0][2], ["no", "yes"]); sync()
sup_y = [l in ("update", "contradiction") for _, _, l in SUPERSEDE]
for k, ins, crit in SUP:
    p = [decide({"old_memory": a, "new_memory": b}, ins, crit, ["no", "yes"])["probabilities"]["yes"] for a, b, _ in SUPERSEDE]
    out[f"supersede_{k}"] = binary_report(p, sup_y); out[f"supersede_{k}_probs"] = [round(x, 3) for x in p]
pred3 = [decide({"old_memory": a, "new_memory": b}, "How does the NEW memory relate to the OLD memory?", REL, list(REL))["prediction"] for a, b, _ in SUPERSEDE]
out["relation3_acc"] = acc(pred3, [{"contradiction": "update"}.get(l, l) for _, _, l in SUPERSEDE]); out["relation3_pred"] = pred3
ent_y = [l == "same" for _, _, l in ENTITY]
for k, ins, crit in ENT:
    p = [decide({"mention_a": a, "mention_b": b}, ins, crit, ["no", "yes"])["probabilities"]["yes"] for a, b, _ in ENTITY]
    out[f"entity_{k}"] = binary_report(p, ent_y); out[f"entity_{k}_probs"] = [round(x, 3) for x in p]
lat = []
for a, b, _ in SUPERSEDE + ENTITY:
    sync(); t = time.perf_counter(); decide({"old_memory": a, "new_memory": b}, SUP[0][1], SUP[0][2], ["no", "yes"]); sync()
    lat.append((time.perf_counter() - t) * 1000)
lat.sort(); out["lat_b1_ms_p50"] = round(lat[len(lat) // 2], 1); out["lat_b1_ms_p90"] = round(lat[int(len(lat) * .9)], 1)
out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
print(json.dumps(out))
