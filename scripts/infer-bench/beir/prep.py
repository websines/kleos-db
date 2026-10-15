import json
import glob, os
S = glob.glob(os.path.join(os.environ["HF_HOME"], "hub/datasets--mteb--scifact/snapshots/*"))[0]
docs = [json.loads(l) for l in open(f"{S}/corpus.jsonl")]
qrels = {}
for l in open(f"{S}/qrels/test.jsonl"):
    r = json.loads(l); qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
qs = [json.loads(l) for l in open(f"{S}/queries.jsonl")]
qs = [q for q in qs if q["_id"] in qrels]
with open("docs.jsonl", "w") as f:
    for d in docs: f.write(json.dumps((d["title"] + ". " + d["text"]).strip()) + "\n")
with open("queries.jsonl", "w") as f:
    for q in qs: f.write(json.dumps(q["text"]) + "\n")
json.dump({"doc_ids": [d["_id"] for d in docs], "query_ids": [q["_id"] for q in qs], "qrels": qrels}, open("meta.json", "w"))
print(len(docs), len(qs))
