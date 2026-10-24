"""Reference stages of GLiNER2.5 (boundary architecture) entity extraction, CPU f32,
for the Rust parity test (`kleos-infer/tests/gliner_parity.rs`).

    python gliner_ref.py <checkpoint-dir> <out.json>

Runs gliner2 (2.1.0) one text at a time (no padding) with the entity-only schema
kleos uses and dumps every stage: word split, input ids, routed positions,
encoder states (16 dims of every token plus per-token norms), boundary states,
start/end/inside marginals, the shared candidate pool, pair logits, abstention
logits, and the decoded entities at the decode floor (0.1).
"""
import json
import math
import sys

import torch
from gliner2 import AutoExtractor
from gliner2.training.trainer import ExtractorCollator
from transformers.models.deberta_v2.modeling_deberta_v2 import make_log_bucket_position

CKPT, OUT = sys.argv[1], sys.argv[2]
LABELS = ["person", "location", "organization", "relative"]
FLOOR = 0.1
torch.set_num_threads(4)

TEXTS = [
    "Caroline: My sister Anna moved to Berlin last week to work at Google.",
    "I love hiking with my dog.",
    "Melanie: Thanks, Caroline! My kids and I went camping at Yosemite with my husband.",
    "My boss at Acme Corp, Dr. O'Neill, said the Paris office closes in June",
    "Visit https://example.com/path?q=1 or mail jane.doe@example.org, cc @johnny_b.",
    "Zoë and José met in São Paulo; she works for Nestlé — a Swiss-based co-op.",
    "We're driving to Lake Tahoe with Mom and Dad this weekend!!",
    "",
    "ok",
    "My grandmother's recipe for pierogi came from Kraków, and my cousin Tomasz still makes them every Christmas in Chicago.",
    ("Last year I moved from Toronto to Montreal for a job at Ubisoft, where I lead the audio team. "
     "My partner Sofia is a nurse at the Montreal Children's Hospital and works night shifts. "
     "We adopted a cat named Miso, and my brother Daniel visits from Vancouver every summer. ") * 3,
    "Tabs\tand\nnewlines  and   double  spaces around New York City.",
]

m = AutoExtractor.from_pretrained(CKPT, map_location="cpu")
m.eval()
schema = m.create_schema().entities(LABELS)
schema_dicts, metas = m._build_schema_dicts_and_metadata([schema])
collator = ExtractorCollator(m.processor, is_training=False, architecture="boundary")
head = m.boundary_head


def tl(t, nd=6):
    if isinstance(t, torch.Tensor):
        t = t.detach().float().tolist()
    if isinstance(t, list):
        return [tl(x, nd) for x in t]
    return round(float(t), nd)


items = []
with torch.inference_mode():
    for text in TEXTS:
        batch = collator([(text, schema_dicts[0])])
        hs = m.encode_tokens(batch)[0]
        core = m._encode_core(batch)
        ts, tm = core["text_states"], core["text_mask"]
        qs, qm = core["query_states"], core["query_mask"]
        enc = head.boundary_encoder(ts, tm)
        marg = head.boundary_query_head(enc.states, enc.mask, ts, tm, qs, qm)
        pooled = head.shared_pool_builder(enc.states, enc.mask, qm, marg.start_logits, marg.end_logits)
        logits, _ = head.shared_pool_scorer(
            enc.states, qs, qm, pooled, marg.start_logits, marg.end_logits, marg.inside_prefix,
            tm.sum(1).long(), ts, tm, inside_prefix_mean=marg.inside_prefix_mean)
        null = head.null_projection(qs).squeeze(-1)[0]
        valid = pooled.mask[0]
        cidx = [i for i in range(valid.shape[0]) if bool(valid[i])]
        res = m.batch_extract([text], schema, threshold=FLOOR, include_confidence=True, include_spans=True)[0]
        norm_text = m.processor._normalize_text(text)
        words = [[w, s, e] for w, s, e in m.processor.word_splitter(norm_text, lower=True)]
        ids = batch.input_ids[0].tolist()
        items.append({
            "text": text,
            "normalized": norm_text,
            "words": words,
            "input_ids": ids,
            "text_positions": batch.text_word_indices[0].tolist(),
            "marker_positions": batch.query_marker_indices[0].tolist(),
            "hidden16": tl(hs[:, :16]),
            "hidden_norm": tl(hs.norm(dim=-1)),
            "boundary8": tl(enc.states[0, :, :8]),
            "start_logits": tl(marg.start_logits[0]),
            "end_logits": tl(marg.end_logits[0]),
            "inside_prefix": tl(marg.inside_prefix[0]),
            "inside_mean": tl(marg.inside_prefix_mean[0, :, 0]),
            "pool": [[int(pooled.indices[0, i, 0]), int(pooled.indices[0, i, 1])] for i in cidx],
            "pool_compat": tl(pooled.compat_logits[0, cidx]),
            "pair_logits": tl(logits[0, cidx]),
            "null_logits": tl(null),
            "entities": res["entities"],
        })
        print(len(ids), len(words), len(cidx), {k: [(e["text"], round(e["confidence"], 3)) for e in v] for k, v in res["entities"].items()})

# DeBERTa's log-bucketed relative positions for a long sequence (bucket 256, max 512)
rel = torch.arange(700)[:, None] - torch.arange(700)[None, :]
buckets = make_log_bucket_position(rel, 256, 512).long()
json.dump({
    "checkpoint": CKPT,
    "gliner2": __import__("gliner2").__version__ if hasattr(__import__("gliner2"), "__version__") else "2.1.0",
    "labels": LABELS,
    "floor": FLOOR,
    "bucket_row0": buckets[0].tolist(),
    "bucket_row699": buckets[699].tolist(),
    "items": items,
}, open(OUT, "w"), indent=None)
