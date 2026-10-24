# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Native claim-level `remember` on LongMemEval knowledge-update questions.

Each user turn goes to `remember` whole (split='auto' is the default, so the
splitting, the question filter and the batched decisions are kleos's own),
with the session date as `at` and the session id as `source`. The metric is
the prototype's: recall(question, k=10); is the first hit from an answer
session the latest answer session?

    python claims_ku.py held|trained[-nodec] OUT.json [limit] [extra remember opts as JSON]

Needs a kleos wheel with `infer-metal` (maturin build --release --features
compact,infer-metal -m kleos-lib-python/Cargo.toml). Results: "Claims" in
docs/specs/memory-api.md.
"""
import json, sys, time, statistics
from datetime import datetime, timezone

import hashlib, os, random  # noqa: E401
from kleos import CozoDbPy  # noqa: E402

ORACLE = os.environ.get("LME_ORACLE", "/Volumes/SSD01/caches/huggingface/hub/datasets--xiaowu0162--longmemeval-cleaned/snapshots/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_oracle.json")
SEED = 20261023  # the decider-nano data build's seed (scripts/bench-l2/common.py)
ENTS = json.load(open(os.environ["KU_ENTITIES"]))["items"] if os.environ.get("KU_ENTITIES") else None


def lme_ku_split(data):
    """The 25 knowledge-update questions held out from decider-nano training
    (sorted by id, shuffled with Random(SEED + 1), first 25), and the other 53."""
    ku = sorted([x for x in data if x["question_type"] == "knowledge-update"], key=lambda x: x["question_id"])
    random.Random(SEED + 1).shuffle(ku)
    return {x["question_id"] for x in ku[:25]}, {x["question_id"] for x in ku[25:]}


def micros(d):
    t = datetime.strptime(d.split(" (")[0] + " " + d.split(") ")[1], "%Y/%m/%d %H:%M")
    return int(t.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)


def main(which, out_path, limit, extra):
    raw = json.load(open(ORACLE))
    data = {x["question_id"]: x for x in raw}
    held, trained = lme_ku_split(raw)
    pool = held if which == "held" else trained
    qids = sorted(q for q in pool if q in data and not q.endswith("_abs"))[:limit]
    db = CozoDbPy("mem", "", "{}")
    db.enable_embedder("gemma", device="metal")
    if NODEC:
        db.memory_init(embedder="gemma")
    else:
        db.enable_decider("nano", device="metal")
        db.memory_init(embedder="gemma", decider="nano")
    out = []
    for qi, qid in enumerate(qids):
        q = data[qid]
        dates = dict(zip(q["haystack_session_ids"], q["haystack_dates"]))
        ns = f"q{qi}"
        lat, per_claim = [], []
        st = {"statements": 0, "claims": 0, "decided": 0, "added": 0, "superseded": 0, "pending": 0, "dup": 0}
        pend, sups, dups = [], [], []
        sessions = sorted(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]), key=lambda s: micros(s[1]))
        for sid, date, sess in sessions:
            at = micros(date)
            for t in sess:
                if t["role"] != "user":
                    continue
                kw = dict(extra)
                if ENTS is not None:  # KU_ENTITIES: precomputed entity lists (scripts/gliner/e2e_entities.py ku)
                    e = ENTS.get(f"{qid}/{hashlib.sha1(t['content'].encode()).hexdigest()}")
                    if e and e["entities"]:
                        kw["entities"] = e["entities"]
                t0 = time.perf_counter()
                r = db.remember(t["content"], ns=ns, at=at, source=sid, **kw)
                ms = (time.perf_counter() - t0) * 1000
                claims = r.get("claims") or [{"id": r.get("id"), "text": t["content"], "action": r["action"], "decided": True}]
                lat.append(ms)
                per_claim.append(ms / len(claims))
                st["statements"] += 1
                st["claims"] += len(claims)
                st["decided"] += sum(1 for c in claims if c.get("decided"))
                st["added"] += sum(1 for c in claims if c["action"] == "added")
                st["dup"] += sum(1 for c in claims if c["action"] == "duplicate")
                st["superseded"] += len(r.get("superseded") or [])
                st["pending"] += len(r.get("tentative") or [])
                text_of = {c.get("id"): c["text"] for c in claims}
                for d in r.get("decisions") or []:
                    rec = {"new": text_of.get(d.get("claim"), t["content"])[:300], "old": d["candidate_text"][:300],
                           "p": round(d["p"], 4), "sim": d.get("similarity"), "action": d["action"], "band": d["band"], "sid": sid,
                           "label": d["label"], "claim": d.get("claim"), "stmt": st["statements"]}
                    if d["template"] != "supersedes":
                        dups.append(rec)
                        continue
                    sups.append(rec)
                    if d["action"] == "pending":
                        pend.append(rec)
        hits = db.recall(q["question"], ns=ns, k=10)
        srcs = [h.get("source") for h in hits]
        ans_dates = sorted(((micros(dates[s]), s) for s in q["answer_session_ids"] if s in dates))
        latest = ans_dates[-1][1] if ans_dates else None
        first_ans = next((s for s in srcs if s in q["answer_session_ids"]), None)
        qw = {}
        for w in QWS:
            hw = db.recall(q["question"], ns=ns, k=10, question_weight=w, record_access=False)
            fw = next((h.get("source") for h in hw if h.get("source") in q["answer_session_ids"]), None)
            qw[str(w)] = {"latest_first": fw == latest, "stale_first": fw is not None and fw != latest}
        hh = db.recall(q["question"], ns=ns, k=10, hide_pending=True)
        hsrcs = [h.get("source") for h in hh]
        h_first = next((s for s in hsrcs if s in q["answer_session_ids"]), None)
        rec = {
            "hide_latest_first": h_first == latest, "hide_stale_first": h_first is not None and h_first != latest,
            "qw": qw, "qid": qid, "question": q["question"], "answer": q["answer"],
            "latest_first": first_ans == latest, "stale_first": first_ans is not None and first_ans != latest,
            "latest_in10": latest in srcs, "stats": st,
            "remember_p50_ms": statistics.median(lat) if lat else None,
            "per_claim_p50_ms": statistics.median(per_claim) if per_claim else None,
            "total_ms": sum(lat), "lat": [round(x, 1) for x in lat], "lat_claim": [round(x, 1) for x in per_claim],
            "top3": [h["text"][:140] for h in hits[:3]],
            "pending": pend, "supersedes": sups, "duplicates": dups,
        }
        out.append(rec)
        print(json.dumps({k: rec[k] for k in ("qid", "latest_first", "stale_first", "latest_in10", "stats", "remember_p50_ms", "per_claim_p50_ms")}), flush=True)
        json.dump(out, open(out_path, "w"), indent=1)
    n = len(out)
    allst = {k: sum(r["stats"][k] for r in out) for k in out[0]["stats"]}
    print("SUMMARY", json.dumps({
        "which": which, "n": n, "opts": extra, "decider": not NODEC,
        "latest_first": sum(r["latest_first"] for r in out) / n,
        "stale_first": sum(r["stale_first"] for r in out) / n,
        "latest@10": sum(r["latest_in10"] for r in out) / n,
        "hide_pending_latest_first": sum(r["hide_latest_first"] for r in out) / n,
        "hide_pending_stale_first": sum(r["hide_stale_first"] for r in out) / n,
        "question_weight_latest_first": {w: sum(r["qw"][w]["latest_first"] for r in out) / n for w in out[0]["qw"]},
        "question_weight_stale_first": {w: sum(r["qw"][w]["stale_first"] for r in out) / n for w in out[0]["qw"]},
        "superseded_per_q": allst["superseded"] / n,
        "pending_per_q": allst["pending"] / n,
        "pending_supersedes_per_q": sum(len(r["pending"]) for r in out) / n,
        "pending_duplicate_per_q": sum(1 for r in out for d in r["duplicates"] if d["action"] == "pending") / n,
        "claims_per_statement": allst["claims"] / allst["statements"],
        "decided_frac": allst["decided"] / allst["claims"],
        "remember_p50_ms_statement": statistics.median([x for r in out for x in r["lat"]]),
        "remember_p90_ms_statement": statistics.quantiles([x for r in out for x in r["lat"]], n=10)[-1],
        "per_claim_p50_ms": statistics.median([x for r in out for x in r["lat_claim"]]),
        "ms_per_claim_mean": sum(r["total_ms"] for r in out) / allst["claims"],
    }), flush=True)


NODEC = False
QWS = [0.0, 0.02, 0.05, 0.1]
if __name__ == "__main__":
    if sys.argv[1].endswith("-nodec"):
        NODEC = True
        sys.argv[1] = sys.argv[1][:-6]
    extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1000, extra)
