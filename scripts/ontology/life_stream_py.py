# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""The life stream (scripts/memory-api/life_stream.py) through the Python
wheel, with or without an ontology: the same turns, gold and questions as
`kleos-core/examples/memory_scenario.rs`, scored the same way (auto
supersedes right / wrong / missed, duplicates, top-1 with and without
`hide_pending`, blend on, read now).

    KLEOS_ONTOLOGY=personal[:decide|pending|dup_only] python life_stream_py.py DATA.json OUT.json [--no-entities]
"""
import json
import os
import sys

from kleos import CozoDbPy

DAY = 86_400_000_000
T0 = 1_767_225_600_000_000


def main(data_path, out_path, entities=True):
    data = json.load(open(data_path))
    db = CozoDbPy("mem", "", "{}")
    db.enable_embedder("gemma", device="metal")
    db.enable_decider("nano", device="metal")
    db.memory_init(embedder="gemma", decider="nano")
    onto = os.environ.get("KLEOS_ONTOLOGY", "")
    if onto:
        o, _, untyped = onto.partition(":")
        db.ontology_call("activate", {"ns": "user", "ontos": o.split("+"), "untyped": untyped or "decide"})
    turns = data["turns"]
    id_turn = {}
    st = {"sup_right": 0, "sup_wrong": [], "sup_missed": [], "dup_right": 0, "dup_wrong": [], "dup_missed": [],
          "pending": 0, "onto_gated": 0, "onto_blocked": 0}
    log = []
    for i, tn in enumerate(turns):
        kw = {"entities": tn["entities"]} if entities else {}
        r = db.remember(tn["text"], ns="user", at=T0 + tn["day"] * DAY, **kw)
        if r.get("id"):
            id_turn[r["id"]] = i
        gold_sup = set(tn.get("supersedes") or [])
        got = {id_turn[x] for x in r.get("superseded") or [] if x in id_turn}
        for j in got:
            if j in gold_sup:
                st["sup_right"] += 1
            else:
                st["sup_wrong"].append({"turn": i, "text": tn["text"], "superseded": turns[j]["text"]})
        for j in gold_sup - got:
            st["sup_missed"].append({"turn": i, "text": tn["text"], "should": turns[j]["text"]})
        gd = tn.get("duplicate_of")
        dj = id_turn.get(r.get("duplicate_of")) if r.get("duplicate_of") else None
        if dj is not None and dj == gd:
            st["dup_right"] += 1
        elif dj is not None:
            st["dup_wrong"].append({"turn": i, "text": tn["text"], "of": turns[dj]["text"]})
        elif gd is not None:
            st["dup_missed"].append({"turn": i, "text": tn["text"], "of": turns[gd]["text"]})
        st["pending"] += len(r.get("tentative") or [])
        st["onto_gated"] += len(r.get("ontology_gated") or [])
        st["onto_blocked"] += len(r.get("ontology_blocked") or [])
        log.append({"turn": i, "text": tn["text"], "types": r.get("types"), "superseded": [turns[id_turn[x]]["text"] for x in r.get("superseded") or [] if x in id_turn]})
    res = {}
    for hide in (False, True):
        correct, stale_top, answers = 0, 0, []
        for q in data["questions"]:
            as_of = T0 + q["as_of_day"] * DAY if q.get("as_of_day") is not None else None
            kw = {"as_of": as_of} if as_of else {}
            hits = db.recall(q["query"], ns="user", k=3, hide_pending=hide, **kw)
            top = (hits[0].get("claim") or hits[0]["text"]) if hits else ""
            ok = any(e in top for e in q["expect"].split("|"))
            correct += ok
            stale_top += any(s in top for s in q["stale"])
            answers.append({"query": q["query"], "ok": ok, "top": top})
        res["hide_pending" if hide else "default"] = {"correct_top1": correct, "of": len(data["questions"]),
                                                      "stale_top1": stale_top, "answers": answers}
    out = {"ontology": onto or None, "entities": entities,
           "auto_supersedes": {"right": st["sup_right"], "wrong": len(st["sup_wrong"]), "missed": len(st["sup_missed"])},
           "duplicates": {"right": st["dup_right"], "wrong": len(st["dup_wrong"]), "missed": len(st["dup_missed"])},
           "pending": st["pending"], "onto_gated": st["onto_gated"], "onto_blocked": st["onto_blocked"],
           "top1": {k: v["correct_top1"] for k, v in res.items()}, "detail": st, "recall": res, "log": log}
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("ontology", "entities", "auto_supersedes", "duplicates", "pending", "onto_gated", "onto_blocked", "top1")}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--no-entities" not in sys.argv)
