# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Write-time policies on the 23 held-out LongMemEval KU questions: the
decider alone vs GLiNER update keys and GLiNER entities, through the kleos
Python bindings (decider-nano + EmbeddingGemma on Metal) plus GLiNER in
Python. The harness is scripts/memory-api/claims_ku.py's: every user turn goes
to `remember` whole (split 'auto'), session date as `at`, session id as
`source`; then recall(question, k=10): is the first hit from an answer
session the latest one?

Policies:
  a        the decider alone (main).
  b        + keys: after each remember, GLiNER extracts (subject, attribute,
           value) facts from the new declarative claims; a current claim of an
           earlier statement with the same subject and a single-valued
           attribute but another value is retracted (`forget`, at the new
           claim's time): an auto-supersede. The decider handles the rest.
  c        + entities: GLiNER entities of the turn (person, location,
           organization, product, event, title) are passed as `entities`,
           so the same-subject gate applies.
  cu       as c, plus the subject `user` when the turn is in the first person.
  bc       b and cu together.

    python policy_sim.py POLICY OUT.json [gliner_model] [key_variant] [gliner_device]
"""
import json
import re
import statistics
import sys
import time

import common as C
import ku_keys as K

GATE_LABELS = ["person", "location", "organization", "product", "event", "title"]
FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself)\b", re.I)


def main(policy, out_path, gmodel="small", variant="v2", gdev="cpu"):
    raw = C.lme()
    data = {x["question_id"]: x for x in raw}
    held, _ = C.lme_ku_split(raw)
    qids = sorted(q for q in held if q in data and not q.endswith("_abs"))
    from kleos import CozoDbPy
    db = CozoDbPy("mem", "", "{}")
    db.enable_embedder("gemma", device="metal")
    db.enable_decider("nano", device="metal")
    db.memory_init(embedder="gemma", decider="nano")
    g = C.load_gliner(gmodel, gdev) if policy != "a" else None
    use_keys = policy in ("b", "bc")
    use_ents = policy in ("c", "cu", "bc")
    add_user = policy in ("cu", "bc")
    out = []
    for qi, qid in enumerate(qids):
        q = data[qid]
        dates = dict(zip(q["haystack_session_ids"], q["haystack_dates"]))
        ns = f"q{qi}"
        lat, glat = [], []
        st = {"statements": 0, "claims": 0, "superseded": 0, "pending": 0, "key_superseded": 0, "gated": 0}
        sups, keysups = [], []
        current = []  # (claim_id, stmt_no, text, facts) of current declarative claims
        sessions = sorted(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]), key=lambda s: C.micros(s[1]))
        for sid, date, sess in sessions:
            at = C.micros(date)
            for t in sess:
                if t["role"] != "user":
                    continue
                opts = {"ns": ns, "at": at, "source": sid}
                g0 = time.perf_counter()
                if use_ents:
                    r = g.extract_entities(t["content"], GATE_LABELS, threshold=0.5)
                    names = sorted({e["text"] if isinstance(e, dict) else e for lst in (r.get("entities") or {}).values() for e in lst})
                    if add_user and FIRST_PERSON.search(t["content"]):
                        names.append("user")
                    if names:
                        opts["entities"] = names
                gms = (time.perf_counter() - g0) * 1000
                t0 = time.perf_counter()
                r = db.remember(t["content"], **opts)
                ms = (time.perf_counter() - t0) * 1000
                st["statements"] += 1
                claims = r.get("claims") or [{"id": r.get("id"), "text": t["content"], "action": r["action"], "decided": True}]
                st["claims"] += len(claims)
                st["superseded"] += len(r.get("superseded") or [])
                st["pending"] += len(r.get("tentative") or [])
                st["gated"] += len(r.get("gated") or [])
                text_of = {c.get("id"): c["text"] for c in claims}
                for d in r.get("decisions") or []:
                    if d["template"] == "supersedes" and d["action"] not in ("pending", "none"):
                        sups.append({"by": "decider", "new": text_of.get(d.get("claim"), t["content"])[:300], "old": d["candidate_text"][:300], "p": d["p"], "sid": sid})
                if use_keys:
                    g1 = time.perf_counter()
                    new = [c for c in claims if c.get("decided") and c["action"] == "added" and c.get("id")]
                    F = K.facts(g, [c["text"] for c in new], variant.rstrip("sn") if variant.startswith("v3") else variant) if new else []
                    stmt = st["statements"]
                    ended = set(r.get("superseded") or [])
                    for c, fs in zip(new, F):
                        for old in list(current):
                            oid, ostmt, otext, ofs = old
                            if ostmt == stmt or oid in ended:
                                continue
                            h = K.match_mode(ofs, fs, K.MATCHER[variant], otext, c["text"])
                            if h:
                                db.forget(oid, ns=ns, at=at)
                                ended.add(oid)
                                current.remove(old)
                                st["key_superseded"] += 1
                                keysups.append({"by": "key", "new": c["text"][:300], "old": otext[:300], "key": [h[0]["subject"], h[0]["attribute"]],
                                                "values": [h[0]["value"], h[1]["value"]], "sid": sid})
                    current = [x for x in current if x[0] not in ended]
                    current += [(c["id"], stmt, c["text"], fs) for c, fs in zip(new, F)]
                    gms += (time.perf_counter() - g1) * 1000
                lat.append(ms + gms)
                glat.append(gms)
        hits = db.recall(q["question"], ns=ns, k=10)
        srcs = [h.get("source") for h in hits]
        ans = sorted(((C.micros(dates[s]), s) for s in q["answer_session_ids"] if s in dates))
        latest = ans[-1][1] if ans else None
        first = next((s for s in srcs if s in q["answer_session_ids"]), None)
        hh = db.recall(q["question"], ns=ns, k=10, hide_pending=True)
        hfirst = next((h.get("source") for h in hh if h.get("source") in q["answer_session_ids"]), None)
        rec = {"qid": qid, "question": q["question"], "answer": q["answer"],
               "latest_first": first == latest, "stale_first": first is not None and first != latest,
               "hide_latest_first": hfirst == latest, "latest_in10": latest in srcs, "stats": st,
               "lat": [round(x, 1) for x in lat], "gliner_ms": [round(x, 1) for x in glat],
               "supersedes": sups, "key_supersedes": keysups, "top3": [h["text"][:140] for h in hits[:3]]}
        out.append(rec)
        print(json.dumps({k: rec[k] for k in ("qid", "latest_first", "stale_first", "stats")}), flush=True)
        json.dump(out, open(out_path, "w"), indent=1)
    n = len(out)
    allst = {k: sum(r["stats"][k] for r in out) for k in out[0]["stats"]}
    L = [x for r in out for x in r["lat"]]
    print("SUMMARY", json.dumps({
        "policy": policy, "gliner": gmodel if g else None, "variant": variant if use_keys else None, "n": n,
        "latest_first": sum(r["latest_first"] for r in out) / n,
        "stale_first": sum(r["stale_first"] for r in out) / n,
        "hide_pending_latest_first": sum(r["hide_latest_first"] for r in out) / n,
        "latest@10": sum(r["latest_in10"] for r in out) / n,
        "decider_superseded_per_q": allst["superseded"] / n,
        "key_superseded_per_q": allst["key_superseded"] / n,
        "pending_per_q": allst["pending"] / n,
        "gated_per_q": allst["gated"] / n,
        "remember_p50_ms": statistics.median(L), "remember_p90_ms": statistics.quantiles(L, n=10)[-1],
        "gliner_p50_ms": statistics.median([x for r in out for x in r["gliner_ms"]]),
    }), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
