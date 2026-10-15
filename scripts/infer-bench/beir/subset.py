"""SciFact-1.5k: every qrel-relevant document plus seeded random others (1,500 total).
Run in the directory written by prep.py; writes ./sub/."""
import json, os, random, shutil
meta = json.load(open("meta.json")); docs = [json.loads(l) for l in open("docs.jsonl")]
rel = {d for q in meta["qrels"].values() for d in q}
idx = [i for i, d in enumerate(meta["doc_ids"]) if d in rel]
rest = [i for i in range(len(docs)) if meta["doc_ids"][i] not in rel]
random.Random(0).shuffle(rest); idx = sorted(idx + rest[:1500 - len(idx)])
os.makedirs("sub", exist_ok=True)
with open("sub/docs.jsonl", "w") as f:
    for i in idx: f.write(json.dumps(docs[i]) + "\n")
shutil.copy("queries.jsonl", "sub/queries.jsonl")
json.dump({"doc_ids": [meta["doc_ids"][i] for i in idx], "query_ids": meta["query_ids"], "qrels": meta["qrels"]}, open("sub/meta.json", "w"))
