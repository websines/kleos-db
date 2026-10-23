"""Reference logits/features for tasksource-decider-nano (CPU f32), for the Rust parity test."""
import json, sys, importlib.util
from pathlib import Path
import numpy as np, torch
SNAP = Path(sys.argv[1]); OUT = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("decider_ref", SNAP / "decider.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_num_threads(4)
m = mod.Decider(SNAP, device="cpu")
m.net.float()
long_state = ("The user keeps a detailed travel journal. " * 3 +
  "In 2019 they lived in Toronto and worked at a bank downtown, commuting by streetcar every day. "
  "In 2021 they moved to Montreal for a job at a video game studio, where they led the audio team. "
  "They enjoy cycling along the Lachine canal, cooking Lebanese food, and learning French in the evenings. "
  "Their partner, Sofia, is a pediatric nurse who works night shifts at the children's hospital. "
  "They adopted a cat named Miso last winter and are thinking about getting a second one. ")
NLI = "Classify the relationship between the old note and this new statement:\n"
items = [
  ("Tom lives in Berlin.", NLI + "Tom moved to Lisbon in March.", ["Entailment", "Contradiction", "NotMentioned"]),
  ("Maya works as a nurse at St. Mary's Hospital.", "Does the new note make the old note no longer true? New note: Maya started a new job as a data analyst at Stripe last month.", ["false", "true"]),
  ("The user prefers Python for scripting.", "Is this new statement a duplicate of the old note? New: The user likes using Python for scripts.", ["no", "yes"]),
  ("Lena is a vegetarian.", "How does the new statement relate to the old note? New: Lena started eating meat again this year.",
     ["update: the old note is no longer true", "contradiction: they cannot both be true", "compatible: both can be true", "duplicate: same information"]),
  ({"review": "The battery died after a week."}, "What is the sentiment?", ["negative", "positive"]),
  ("", "Is this memory worth keeping long-term?", ["false", "true"]),
  (long_state, "Where does the user live now?", ["Toronto", "Montreal", "Lisbon"]),
  (long_state, "Does the statement 'The user has a dog' follow from the journal?", ["entailment", "contradiction", "neutral"]),
  ("Sam drives a 2015 Honda Civic.", NLI + "Sam sold the Civic and bought a Tesla Model 3.", ["Entailment", "Contradiction", "NotMentioned"]),
  ("The project deadline is June 30.", NLI + "The project deadline is June 30, as agreed.", ["Entailment", "Contradiction", "NotMentioned"]),
]
rendered = [(mod.state_text(s), q, o) for s, q, o in items]
logits = m.score_many(rendered)
# pooled option features for the first two items, re-run one item at a time with a hook on the scorer
feats = []
cap = {}
h = m.net.scorer.register_forward_hook(lambda mod_, inp, out: cap.__setitem__("v", inp[0].detach().float().cpu().numpy()))
for it in rendered[:2]:
    m.score_many([it]); feats.append(cap["v"].tolist())
h.remove()
OUT.write_text(json.dumps({"model": "tasksource/tasksource-decider-nano", "revision": SNAP.name, "dtype": "f32-cpu",
  "items": [{"state": s, "question": q, "options": o, "logits": [float(x) for x in z]} for (s, q, o), z in zip(rendered, logits)],
  "features": [{"item": i, "vectors": f} for i, f in enumerate(feats)]}, indent=1))
for (s, q, o), z in zip(rendered, logits): print(len(m.tok(s)["input_ids"]), [round(float(x), 4) for x in z])
