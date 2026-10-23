# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""GLiNER2.5-Decide (a single-text classifier) used zero-shot as a pair
decider, on decider-nano's held-out LongMemEval KU pairs
(.bench-data/nano-tune/data/lme_ku_heldout.json: 24 update vs 66 compatible
pairs of held-out questions; decider-nano tunec scores AUROC 0.824 there).
The pair goes into the text, the question into the classification `prompt`.

    python decide_pairs.py [decide|base|small] [cpu|mps]
"""
import json
import sys
import time

import common as C

PAIRS = C.ROOT / ".bench-data" / "nano-tune" / "data" / "lme_ku_heldout.json"
FORMS = {
    "yesno": ("Earlier the user said: {old}\nLater the user said: {new}",
              {"answer": {"labels": ["yes", "no"], "prompt": "Does the later message update or replace a fact stated in the earlier one?"}}, "yes"),
    "relation": ("Earlier the user said: {old}\nLater the user said: {new}",
                 {"relation": {"labels": {"update": "the later message changes a value stated earlier",
                                          "compatible": "both messages can be true at the same time"}}}, "update"),
}


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main(name="decide", device="cpu"):
    pairs = json.loads(PAIRS.read_text())["pairs"]
    m = C.load_gliner(name, device)
    out = {}
    for form, (tmpl, task, pos_label) in FORMS.items():
        texts = [tmpl.format(old=o, new=n) for o, n, _ in pairs]
        t0 = time.perf_counter()
        res = m.batch_classify_text(texts, task, batch_size=8, include_confidence=True)
        dt = time.perf_counter() - t0
        key = next(iter(task))
        sc = []
        for r in res:
            v = r[key]
            lab, conf = (v["label"], v["confidence"]) if isinstance(v, dict) else (v, 1.0)
            sc.append(conf if lab == pos_label else 1 - conf)
        y = [p[2] == "update" for p in pairs]
        acc = sum((s >= 0.5) == t for s, t in zip(sc, y)) / len(y)
        out[form] = {"auroc": auroc(sc, y), "acc@0.5": acc, "ms_per_pair": 1000 * dt / len(pairs),
                     "pred_update": sum(s >= 0.5 for s in sc)}
        print(name, form, json.dumps(out[form]), flush=True)
    (C.GDIR / f"decide_pairs_{name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
