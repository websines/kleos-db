#!/usr/bin/env python3
# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Ingest a benchmark through kleos's public memory API and record what
`recall` returns, one cached JSON per unit (a LongMemEval question or a LoCoMo
conversation). Runs are resumable (a cached unit is skipped) and seeded.

    python retrieval.py --bench lme    --variant fts    [--n 500] [--shard 0/2]
    python retrieval.py --bench lme    --variant hybrid --n 60
    python retrieval.py --bench lme    --variant sup    --ids-from ku_heldout
    python retrieval.py --bench locomo --variant hybrid

Variants (all through `remember` / `recall`, nothing else):
  fts     no models: FTS-only recall, model-free write rules (exact duplicate, value rule)
  hybrid  EmbeddingGemma (int8) vectors + FTS, WeightedFusion; no decider (rules only)
  sup     hybrid + decider-nano supersession / duplicate decisions on every remember
With --as-of question, recall reads at the question's date and results go to <variant>@q.

Results go to <bench-dir>/results/<tag>/<bench>-<unit>/<variant>/. The tag
defaults to the kleos git revision (`git rev-parse --short HEAD`).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def open_db(variant, device):
    import kleos
    from kleos import CozoDbPy

    db = CozoDbPy("mem", "", "{}")
    if variant in ("hybrid", "sup") and not getattr(open_db, "_models", False):
        db.enable_embedder("gemma", device=device)
        if variant == "sup":
            db.enable_decider("nano", device=device)
        open_db._models = True
    if variant == "fts":
        db.memory_init()
    elif variant == "hybrid":
        db.memory_init(embedder="gemma")
    else:
        db.memory_init(embedder="gemma", decider="nano")
    return db


def ingest(db, ns, units):
    """remember() every unit in time order; return id -> [src], stats, per-call ms."""
    owner = {}
    lat = []
    st = {"added": 0, "duplicate": 0, "superseded": 0, "pending": 0, "errors": 0}
    for u in units:
        if not u["text"]:
            continue
        t = time.perf_counter()
        try:
            r = db.remember(u["text"], ns=ns, at=u["at"], source=u["src"])
        except Exception as e:  # keep going; count it
            st["errors"] += 1
            print(f"  remember failed on {u['src']}: {str(e)[:160]}", file=sys.stderr)
            continue
        lat.append((time.perf_counter() - t) * 1000)
        if r["action"] == "duplicate":
            st["duplicate"] += 1
            owner.setdefault(r["duplicate_of"], []).append(u["src"])
        else:
            st["added"] += 1
            owner.setdefault(r["id"], []).append(u["src"])
        st["superseded"] += len(r.get("superseded") or [])
        st["pending"] += len(r.get("tentative") or [])
    return owner, st, lat


def recall(db, ns, query, owner, k, **opts):
    t = time.perf_counter()
    hits = db.recall(query, ns=ns, k=k, **opts)
    ms = (time.perf_counter() - t) * 1000
    out = []
    for h in hits:
        out.append({
            "id": h["id"],
            "srcs": owner.get(h["id"], [h.get("source")]),
            "score": h.get("score"),
            "valid_from": h.get("valid_from"),
            "pending_superseded_by": h.get("pending_superseded_by"),
            "text": h["text"][:4000],
        })
    return out, ms


def run_lme(args, outdir):
    path = C.lme_path("longmemeval_oracle.json") if args.bench == "lme-oracle" else C.lme_path()
    data = C.lme_questions(path)
    held, ku_train, _ = C.lme_ku_split()
    if args.ids_from == "ku_heldout":
        data = [x for x in data if x["question_id"] in held]
    elif args.ids_from == "ku":
        data = [x for x in data if x["question_type"] == "knowledge-update"]
    elif args.ids_from:
        ids = set(Path(args.ids_from).read_text().split())
        data = [x for x in data if x["question_id"] in ids]
    if args.n and args.n < len(data):
        data = C.stratified_sample(data, C.lme_category, args.n, seed=args.seed)
    data.sort(key=lambda x: x["question_id"])
    data = data[args.shard_i::args.shard_n]
    db = None
    for qi, x in enumerate(data):
        f = outdir / f"{x['question_id']}.json"
        if f.exists():
            continue
        units = C.lme_units(x, args.unit)
        ns = "q_" + x["question_id"]
        db = open_db(args.variant, args.device)
        t0 = time.perf_counter()
        owner, st, lat = ingest(db, ns, units)
        ingest_s = time.perf_counter() - t0
        ro = {"as_of": C.lme_ts(x["question_date"])} if args.as_of == "question" else {}
        hits10, ms10 = recall(db, ns, x["question"], owner, 10, **ro)
        hits5, ms5 = recall(db, ns, x["question"], owner, 5, **ro)
        hp = {}
        if args.variant == "sup":  # the read policy that acts on pending supersedes
            hp["hits10_hide_pending"], _ = recall(db, ns, x["question"], owner, 10, hide_pending=True, **ro)
            hp["hits5_hide_pending"], _ = recall(db, ns, x["question"], owner, 5, hide_pending=True, **ro)
        db.close()
        sess_date = dict(zip(x["haystack_session_ids"], x["haystack_dates"]))
        rec = {
            "question_id": x["question_id"],
            "question_type": x["question_type"],
            "category": C.lme_category(x),
            "ku_split": "heldout" if x["question_id"] in held else ("train" if x["question_id"] in ku_train else None),
            "question": x["question"],
            "question_date": x["question_date"],
            "answer": x["answer"],
            "answer_session_ids": x["answer_session_ids"],
            "answer_session_dates": {s: sess_date.get(s) for s in x["answer_session_ids"]},
            "evidence_srcs": [u["src"] for u in units if u["evidence"]],
            "n_units": len(units),
            "ingest": st,
            "ingest_s": ingest_s,
            "remember_ms": lat,
            "recall_ms": {"k10": ms10, "k5": ms5},
            "hits10": hits10,
            "hits5": hits5,
            **hp,
        }
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        tmp.rename(f)
        p50 = sorted(lat)[len(lat) // 2] if lat else 0
        print(f"[{qi + 1}/{len(data)}] {x['question_id']} {rec['category']}: {len(units)} units in {ingest_s:.1f}s "
              f"(remember p50 {p50:.0f} ms), sup {st['superseded']} dup {st['duplicate']} pend {st['pending']}, "
              f"recall {ms10:.0f} ms", flush=True)


def run_locomo(args, outdir):
    data = C.locomo_data()
    if args.n:
        data = data[: args.n]
    data = data[args.shard_i::args.shard_n]
    for conv in data:
        sid = conv["sample_id"]
        f = outdir / f"{sid}.json"
        if f.exists():
            continue
        units = C.locomo_units(conv)
        ns = "c_" + sid
        db = open_db(args.variant, args.device)
        t0 = time.perf_counter()
        owner, st, lat = ingest(db, ns, units)
        ingest_s = time.perf_counter() - t0
        qs = []
        for qi, q in enumerate(conv["qa"]):
            h10, ms10 = recall(db, ns, q["question"], owner, 10)
            h5, ms5 = recall(db, ns, q["question"], owner, 5)
            hp = {}
            if args.variant == "sup":
                hp["hits10_hide_pending"], _ = recall(db, ns, q["question"], owner, 10, hide_pending=True)
                hp["hits5_hide_pending"], _ = recall(db, ns, q["question"], owner, 5, hide_pending=True)
            qs.append({**hp, 
                "qid": f"{sid}/{qi}",
                "question": q["question"],
                "answer": q.get("answer", q.get("adversarial_answer")),
                "category": q["category"],
                "evidence": C.locomo_evidence(q.get("evidence")),
                "recall_ms": {"k10": ms10, "k5": ms5},
                "hits10": h10,
                "hits5": h5,
            })
        db.close()
        rec = {"sample_id": sid, "n_units": len(units), "ingest": st, "ingest_s": ingest_s,
               "remember_ms": lat, "last_session_date": max(u["at"] for u in units), "questions": qs}
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        tmp.rename(f)
        p50 = sorted(lat)[len(lat) // 2] if lat else 0
        print(f"{sid}: {len(units)} turns in {ingest_s:.1f}s (p50 {p50:.0f} ms), sup {st['superseded']} "
              f"dup {st['duplicate']} pend {st['pending']}, {len(qs)} questions", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["lme", "lme-oracle", "locomo"], required=True)
    ap.add_argument("--variant", choices=["fts", "hybrid", "sup"], required=True)
    ap.add_argument("--unit", default="user", choices=["user", "round", "turn", "session"])
    ap.add_argument("--n", type=int, default=0, help="stratified sample size (LongMemEval) / first n conversations (LoCoMo)")
    ap.add_argument("--ids-from", default=None, help="'ku_heldout', 'ku' or a file of question ids")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default=os.environ.get("KLEOS_BENCH_DEVICE", "metal"),
                    help="model device. Builds before 6f9711f fail to compile candle's Metal kernels from a Python "
                         "linked against an old macOS SDK (the 276f90f numbers are CPU); 6f9711f pins the MSL version")
    ap.add_argument("--as-of", default="now", choices=["now", "question"],
                    help="read time for recall: 'now' (default) or the question's date (LongMemEval), so the "
                         "recency blend sees the age a live agent would")
    ap.add_argument("--tag", default=os.environ.get("KLEOS_BENCH_TAG") or C.git_rev())
    args = ap.parse_args()
    args.shard_i, args.shard_n = map(int, args.shard.split("/"))
    unit = args.unit if args.bench != "locomo" else "turn"
    vdir = args.variant + ("@q" if args.as_of == "question" else "")
    outdir = C.bench_dir() / "results" / args.tag / f"{args.bench}-{unit}" / vdir
    outdir.mkdir(parents=True, exist_ok=True)
    meta = outdir / "meta.json"
    if not meta.exists():
        import kleos  # noqa: F401
        meta.write_text(json.dumps({
            "tag": args.tag, "kleos_file": getattr(sys.modules["kleos"], "__file__", None),
            "lme": [C.LME_REPO, C.LME_REV], "locomo": [C.LOCOMO_REPO, C.LOCOMO_REV],
            "device": args.device, "seed": args.seed, "unit": unit, "as_of": args.as_of, "started": time.time(),
        }, indent=1))
    print(f"-> {outdir}", flush=True)
    (run_locomo if args.bench == "locomo" else run_lme)(args, outdir)


if __name__ == "__main__":
    main()
