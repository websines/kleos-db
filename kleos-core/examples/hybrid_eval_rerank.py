"""Score (query, doc) pairs with a real cross-encoder for hybrid_eval.

usage: rerank.py <data_dir>   reads rerank_candidates.tsv (qid\tdocid),
appends qid\tdocid\tscore to rerank_scores.tsv. Doc text = title + " " + text,
exactly as the Rust harness stores it.
"""
import json
import os
import sys

import torch
from sentence_transformers import CrossEncoder

D = sys.argv[1]
MODEL = os.environ.get("CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
docs = {}
with open(f"{D}/corpus.jsonl") as f:
    for line in f:
        d = json.loads(line)
        docs[d["id"]] = (d["title"] + " " + d["text"]).strip()
queries = {}
with open(f"{D}/queries.tsv") as f:
    for line in f:
        qid, _, text = line.rstrip("\n").split("\t", 2)
        queries[qid] = text
pairs = [l.rstrip("\n").split("\t") for l in open(f"{D}/rerank_candidates.tsv") if l.strip()]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
ce = CrossEncoder(MODEL, device=dev, max_length=512)
out = open(f"{D}/rerank_scores.tsv", "a")
B = 4096
for i in range(0, len(pairs), B):
    chunk = pairs[i:i + B]
    s = ce.predict([(queries[q], docs[d]) for q, d in chunk], batch_size=64,
                   show_progress_bar=False, convert_to_numpy=True)
    for (q, d), v in zip(chunk, s):
        out.write(f"{q}\t{d}\t{float(v):.6f}\n")
    out.flush()
    if dev == "mps":
        torch.mps.empty_cache()
    print(f"scored {min(i + B, len(pairs))}/{len(pairs)}", flush=True)
