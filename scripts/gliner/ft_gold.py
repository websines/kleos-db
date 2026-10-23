# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Gold set = the GLM-with-reasoning draft labels + hand corrections
(ft_gold/corrections_{a,b}.json, every one of the 200 claims read by hand).
Also reports how often the silver labels (with reasoning, and the no-reasoning
labels used for training) agree with the gold.

    python ft_gold.py      # -> $GLINER_DIR/gold.json
"""
import json
import os
from pathlib import Path

os.environ.setdefault("GLINER_DIR", "/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr/.bench-data/gliner-ft")
import common as C  # noqa: E402
from ft_data import STORE  # noqa: E402

HERE = Path(__file__).resolve().parent / "ft_gold"


def apply(items, fixes, part):
    out = []
    for i, x in enumerate(items):
        f = fixes.get(str(i), {})
        y = {k: x[k] for k in ("id", "src", "text") if k in x}
        y["part"] = part
        y["kind"] = f.get("kind", x["kind"])
        y["about_speaker"] = f.get("about", x["about_speaker"])
        y["entities"] = [{"label": a, "text": b} for a, b in f["ents"]] if "ents" in f else x["entities"]
        y["name_of"] = f.get("pairs", x["name_of"] if "ents" not in f else [p for p in x["name_of"] if p[0] in {e["text"] for e in y["entities"]}])
        for e in y["entities"]:
            assert e["text"] in y["text"], (i, e)
        out.append(y)
    return out


def agree(gold, silver):
    by = {s["id"]: s for s in silver}
    n = ent_ok = kind_ok = store_ok = about_ok = 0
    for g in gold:
        s = by.get(g["id"])
        if not s or "error" in s:
            continue
        n += 1
        ent_ok += {(e["label"], e["text"].lower()) for e in g["entities"]} == {(e["label"], e["text"].lower()) for e in s["entities"]}
        kind_ok += g["kind"] == s["kind"]
        store_ok += (g["kind"] in STORE) == (s["kind"] in STORE)
        about_ok += g["about_speaker"] == s["about_speaker"]
    return {"n": n, "entities_exact": ent_ok / n, "kind": kind_ok / n, "store": store_ok / n, "about": about_ok / n}


def main():
    a = json.loads((C.GDIR / "gold_draft_silver.json").read_text())
    b = json.loads((C.GDIR / "gold_extra_silver.json").read_text())
    gold = apply(a, json.loads((HERE / "corrections_a.json").read_text()), "random") + \
        apply(b, json.loads((HERE / "corrections_b.json").read_text()), "kin")
    (C.GDIR / "gold.json").write_text(json.dumps(gold, indent=1))
    print(len(gold), "gold claims")
    print("GLM with reasoning vs gold (random 150):", agree([g for g in gold if g["part"] == "random"], a))
    print("GLM with reasoning vs gold (kin 50):   ", agree([g for g in gold if g["part"] == "kin"], b))
    fast = C.GDIR / "gold_fast_silver.json"
    if fast.exists():
        print("GLM no reasoning vs gold (random 150): ", agree([g for g in gold if g["part"] == "random"], json.loads(fast.read_text())))


if __name__ == "__main__":
    main()
