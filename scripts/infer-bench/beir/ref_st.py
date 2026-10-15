"""Reference embeddings via sentence-transformers / model2vec. ref_st.py MODEL DEVICE DTYPE"""
import json, sys, time, resource
import numpy as np, torch
name, device, dt = sys.argv[1], sys.argv[2], sys.argv[3]
docs = [json.loads(l) for l in open("docs.jsonl")]; qs = [json.loads(l) for l in open("queries.jsonl")]
if device == "mps": torch.mps.set_per_process_memory_fraction(0.25)
t0 = time.time()
if name.startswith("potion"):
    from model2vec import StaticModel
    m = StaticModel.from_pretrained(f"minishlab/{name}")
    enc_d = lambda x: m.encode(x); enc_q = enc_d
else:
    from sentence_transformers import SentenceTransformer
    repo = {"gemma": "google/embeddinggemma-300m", "qwen3": "Qwen/Qwen3-Embedding-0.6B"}[name]
    m = SentenceTransformer(repo, device=device, model_kwargs={"dtype": {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[dt]})
    m.max_seq_length = 512
    if name == "gemma":
        enc_d = lambda x: m.encode(x, prompt_name="document", batch_size=8)
        enc_q = lambda x: m.encode(x, prompt_name="query", batch_size=8)
    else:
        enc_d = lambda x: m.encode(x, batch_size=4)
        enc_q = lambda x: m.encode(x, prompt_name="query", batch_size=4)
load = time.time() - t0
t = time.time(); D = np.asarray(enc_d(docs), dtype="<f4"); secs = time.time() - t
Q = np.asarray(enc_q(qs), dtype="<f4")
tag = f"{name}_{device}_{dt}_py"
D.tofile(f"vec/{tag}.docs.f32"); Q.tofile(f"vec/{tag}.queries.f32")
print(json.dumps({"tag": tag, "load_s": round(load, 1), "docs_per_s": round(len(docs) / secs, 1), "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6), "dim": D.shape[1]}))
