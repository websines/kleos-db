"""nDCG@10 from raw little-endian f32 vector files: ndcg.py docs.f32 queries.f32 [dim_prefix]"""
import json, sys, math
import numpy as np
meta = json.load(open("meta.json"))
def load(p, n):
    v = np.fromfile(p, dtype="<f4"); return v.reshape(n, -1)
D = load(sys.argv[1], len(meta["doc_ids"])); Q = load(sys.argv[2], len(meta["query_ids"]))
if len(sys.argv) > 3:
    k = int(sys.argv[3]); D = D[:, :k]; Q = Q[:, :k]
D = D / np.linalg.norm(D, axis=1, keepdims=True); Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
S = Q @ D.T
def ndcg_at(ranked, rel, k=10):
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0
def ndcg_from_rankings(rankings):
    return float(np.mean([ndcg_at(r, meta["qrels"][q]) for q, r in zip(meta["query_ids"], rankings)]))
if __name__ == "__main__":
    top = np.argsort(-S, axis=1)[:, :10]
    print(json.dumps({"ndcg@10": round(ndcg_from_rankings([[meta["doc_ids"][i] for i in row] for row in top]), 4), "dim": D.shape[1]}))
