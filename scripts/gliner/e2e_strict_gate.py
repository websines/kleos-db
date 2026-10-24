# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Offline estimate (no models): which of a run's auto-supersedes a stricter
same-subject gate would have allowed, one that needs a shared entity other
than the speaker. Entities per turn come from e2e_entities.py; the old side's
turn is found by text containment in the same conversation.

    python e2e_strict_gate.py <results-dir> <locomo_entities.json> <locomo10.json>
"""
import glob
import json
import os
import sys


def main(rdir, ents_path, locomo_path):
    ents = json.load(open(ents_path))["items"]
    convs = {c["sample_id"]: c for c in json.load(open(locomo_path))}
    tot = keep = 0
    rows = []
    for f in sorted(glob.glob(os.path.join(rdir, "conv-*.json"))):
        r = json.load(open(f))
        sid = r["sample_id"]
        c = convs[sid]["conversation"]
        turns, n = [], 1
        while f"session_{n}" in c:
            turns += [(t["dia_id"], f"{t['speaker']}: {t['text']}", t["speaker"]) for t in c[f"session_{n}"]]
            n += 1

        def turn_of(text):
            body = text.split(" [shares a photo")[0]
            spk, _, claim = body.partition(": ")
            for dia, t, s in turns:
                if s == spk and claim.strip()[:60] in t:
                    return dia, s
            return None, spk

        for d in r["decisions"]:
            if d["action"] != "superseded":
                continue
            tot += 1
            nd, ns_ = turn_of(d["new"])
            od, os_ = turn_of(d["old"])
            ne = set((ents.get(f"{sid}/{nd}") or {}).get("entities") or []) - {ns_}
            oe = set((ents.get(f"{sid}/{od}") or {}).get("entities") or []) - {os_}
            ok = bool(ne & oe)
            keep += ok
            rows.append({"new": d["new"][:120], "old": d["old"][:120], "shared": sorted(ne & oe), "kept": ok})
    print(json.dumps({"auto_supersedes": tot, "strict_gate_keeps": keep, "kept": [x for x in rows if x["kept"]]}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
