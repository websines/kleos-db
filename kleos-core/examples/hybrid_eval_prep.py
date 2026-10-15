"""Prepare BEIR SciFact for the kleos hybrid_eval harness (examples/hybrid_eval.rs).

usage: python hybrid_eval_prep.py <data_dir>   (needs numpy, pyarrow,
sentence-transformers, huggingface_hub)

Writes into <out>/:
  corpus.jsonl      {"id", "title", "text"} in embedding row order
  docs_emb.npy      float32 [n_docs, d], L2-normalised (bge-small-en-v1.5)
  queries.tsv       qid \t split(test|dev) \t text
  queries_emb.npy   float32 [n_queries, d], row-aligned with queries.tsv
  qrels.tsv         qid \t docid \t rel   (test + dev queries)

dev = 200 queries sampled (seed 0) from the BEIR *train* qrels, so weight
tuning never touches the 300 test queries.
"""
import json
import os
import random
import sys

import numpy as np
from huggingface_hub import hf_hub_download

OUT = sys.argv[1]
MODEL = os.environ.get("EMB_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
os.makedirs(OUT, exist_ok=True)


corpus_p = hf_hub_download("BeIR/scifact", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset")
queries_p = hf_hub_download("BeIR/scifact", "queries/queries-00000-of-00001.parquet", repo_type="dataset")
qrels_test_p = hf_hub_download("BeIR/scifact-qrels", "test.tsv", repo_type="dataset")
qrels_train_p = hf_hub_download("BeIR/scifact-qrels", "train.tsv", repo_type="dataset")

import pyarrow.parquet as pq  # noqa: E402

corpus = pq.read_table(corpus_p).to_pylist()
queries = {q["_id"]: q["text"] for q in pq.read_table(queries_p).to_pylist()}


def read_qrels(p):
    out = {}
    with open(p) as f:
        next(f)
        for line in f:
            qid, did, rel = line.rstrip("\n").split("\t")
            if int(rel) > 0:
                out.setdefault(qid, {})[did] = int(rel)
    return out


test = read_qrels(qrels_test_p)
train = read_qrels(qrels_train_p)
rng = random.Random(0)
dev_ids = sorted(rng.sample(sorted(train), 200))
print(f"docs={len(corpus)} test_q={len(test)} dev_q={len(dev_ids)}")

with open(f"{OUT}/corpus.jsonl", "w") as f:
    for d in corpus:
        f.write(json.dumps({"id": d["_id"], "title": d.get("title", ""), "text": d["text"]}) + "\n")

qrows = [(qid, "test", queries[qid]) for qid in sorted(test, key=int)]
qrows += [(qid, "dev", queries[qid]) for qid in dev_ids]
with open(f"{OUT}/queries.tsv", "w") as f:
    for qid, split, text in qrows:
        f.write(f"{qid}\t{split}\t{text.replace(chr(9), ' ').replace(chr(10), ' ')}\n")
with open(f"{OUT}/qrels.tsv", "w") as f:
    for qid, split, _ in qrows:
        for did, rel in (test if split == "test" else train)[qid].items():
            f.write(f"{qid}\t{did}\t{rel}\n")

from sentence_transformers import SentenceTransformer  # noqa: E402
import torch  # noqa: E402

dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer(MODEL, device=dev)
doc_texts = [(d.get("title", "") + " " + d["text"]).strip() for d in corpus]
demb = model.encode(doc_texts, batch_size=32, normalize_embeddings=True,
                    show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
qemb = model.encode([QUERY_PREFIX + t for _, _, t in qrows], batch_size=64,
                    normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
np.save(f"{OUT}/docs_emb.npy", demb)
np.save(f"{OUT}/queries_emb.npy", qemb)
print("emb", demb.shape, qemb.shape)
