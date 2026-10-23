# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""The claim sample for the entity and claim-type evaluation.

160 claims from LongMemEval_S user turns + 140 from LoCoMo turns, one random
claim per sampled turn, split by kleos's own splitter (memory_claims). Turns
over 1500 characters (pasted documents) are skipped. Writes
$GLINER_DIR/sample.json: [{id, src, text, declarative, turn}].

    python build_sample.py
"""
import json
import random

import common as C


def main(n_lme=160, n_loc=140):
    rnd = random.Random(C.SEED + 7)
    out = []
    # LongMemEval_S: unique sessions across all questions
    raw = C.lme("longmemeval_s_cleaned.json")
    turns = {}
    for x in raw:
        for sid, sess in zip(x["haystack_session_ids"], x["haystack_sessions"]):
            if sid in turns:
                continue
            turns[sid] = [t["content"] for t in sess if t["role"] == "user" and 20 <= len(t["content"]) <= 1500]
    del raw
    sids = sorted(s for s in turns if turns[s])
    rnd.shuffle(sids)
    for sid in sids:
        if len(out) >= n_lme:
            break
        turn = rnd.choice(turns[sid])
        claims = C.split_claims(turn)
        text, decl = rnd.choice(claims)
        out.append({"id": f"lme:{sid}:{len(out)}", "src": "lme", "text": text, "declarative": decl, "turn": turn})
    # LoCoMo: both speakers are people talking about their lives
    loc = json.loads(C.LOCOMO.read_text())
    lturns = []
    for ci, conv in enumerate(loc):
        c = conv["conversation"]
        for k, v in c.items():
            if k.startswith("session_") and isinstance(v, list):
                for t in v:
                    if 20 <= len(t.get("text", "")) <= 1500:
                        lturns.append((ci, t["dia_id"], t["speaker"], t["text"]))
    rnd.shuffle(lturns)
    for ci, did, spk, turn in lturns[:n_loc]:
        text, decl = rnd.choice(C.split_claims(turn))
        out.append({"id": f"locomo:{ci}:{did}", "src": "locomo", "speaker": spk, "text": text, "declarative": decl, "turn": turn})
    C.GDIR.mkdir(parents=True, exist_ok=True)
    (C.GDIR / "sample.json").write_text(json.dumps(out, indent=1))
    print(len(out), "claims;", sum(o["declarative"] for o in out), "declarative")


if __name__ == "__main__":
    main()
