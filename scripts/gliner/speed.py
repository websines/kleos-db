# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Speed and memory of one checkpoint on one device (run one process per
model x device so RSS is clean): ms per claim for the entity schema, the
combined schema (entities + type) and the fact-key JSON schema, batch 1 and
batch 8, on the first N sample claims (fastest of 3 repetitions: the
machine is shared); RSS after load and peak.

    python speed.py small|base|multi|decide cpu|mps [N]
"""
import json
import resource
import sys
import time

import psutil

import common as C
from eval_entities import ENT_DESC, TYPE_DESC
import ku_keys as K


REPS = 3


def rss():
    return psutil.Process().memory_info().rss / 2**20


def main(name, device, n=120):
    import torch
    torch.set_num_threads(int(__import__("os").environ.get("GLINER_THREADS", "4")))
    r0 = rss()
    t0 = time.perf_counter()
    m = C.load_gliner(name, device)
    load_s = time.perf_counter() - t0
    r1 = rss()
    texts = [x["text"] for x in json.loads((C.GDIR / "sample.json").read_text())][:n]
    sync = (lambda: torch.mps.synchronize()) if device == "mps" else (lambda: None)
    out = {"model": name, "device": device, "params_M": sum(p.numel() for p in m.parameters()) / 1e6,
           "load_s": load_s, "rss_before_MB": r0, "rss_loaded_MB": r1, "threads": torch.get_num_threads(), "n": n}
    if name == "decide":
        tasks = {"type": list(TYPE_DESC)}
        jobs = {"classify": lambda b: m.batch_classify_text(texts, tasks, batch_size=b)}
    else:
        ent = m.create_schema().entities(ENT_DESC)
        comb = m.create_schema().entities(ENT_DESC).classification("type", TYPE_DESC)
        keys = K.VARIANTS["v2"]()
        jobs = {
            "entities": lambda b: m.batch_extract(texts, ent, batch_size=b),
            "entities+type": lambda b: m.batch_extract(texts, comb, batch_size=b),
            "fact_keys": lambda b: m.batch_extract_json(texts, keys, batch_size=b),
        }
    for jn, fn in jobs.items():
        fn(8)  # warm-up
        sync()
        for b in (1, 8):
            best = float("inf")
            for _ in range(REPS):  # the machine is shared: keep the fastest repetition
                t = time.perf_counter()
                fn(b)
                sync()
                best = min(best, time.perf_counter() - t)
            out[f"{jn}_b{b}_ms_per_claim"] = 1000 * best / n
    out["rss_end_MB"] = rss()
    out["ru_maxrss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    if device == "mps":
        out["mps_driver_MB"] = torch.mps.driver_allocated_memory() / 2**20
    print(json.dumps(out), flush=True)
    with open(C.GDIR / "speed.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 120)
