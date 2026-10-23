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

Recall configurations (--configs): one ingest, several recall option sets, each
written to <variant>+<name> (e.g. `--configs base,div` -> hybrid+base, hybrid+div).
Named presets are in CONFIGS; `name={json}` adds one ad hoc (`;`-separated).
With --context-budget N each config also stores the `format: 'context'` block
of the same k=10 recall (`ctx10`), for the QA comparison list vs context.
--unit user+assistant also stores the assistant replies (`role: 'assistant'`).

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


# recall option sets for --configs. `base` is recall as before diversity
# (the top k by score); the others are measured against it.
CONFIGS = {
    "base": {"diversity": False},
    "default": {},
    "stmt1": {"max_per_statement": 1, "max_per_source": 0, "mmr_weight": 0.0},
    "stmt1-src3": {"max_per_statement": 1, "max_per_source": 3, "mmr_weight": 0.0},
    "stmt1-src2": {"max_per_statement": 1, "max_per_source": 2, "mmr_weight": 0.0},
    "stmt1-mmr05": {"max_per_statement": 1, "max_per_source": 0, "mmr_weight": 0.05},
    "stmt1-mmr10": {"max_per_statement": 1, "max_per_source": 0, "mmr_weight": 0.1},
    "stmt1-src3-mmr05": {"max_per_statement": 1, "max_per_source": 3, "mmr_weight": 0.05},
    "expand": {"expand": True},
    "mmr05": {"mmr_weight": 0.05},
    "src2": {"max_per_source": 2},
}


def parse_configs(spec):
    out = {}
    for part in [x for x in (spec or "").split(";") if x.strip()]:
        if "=" in part:  # name={json}
            k, v = part.split("=", 1)
            out[k.strip()] = json.loads(v)
            continue
        for name in part.split(","):
            out[name.strip()] = CONFIGS[name.strip()]
    return out


def owner_of(owner, h):
    """The source turns of a hit: by memory id, else by its statement id (a
    claim `m_….n`), else the source (the part after `#` for LoCoMo)."""
    i = h["id"]
    if i in owner:
        return owner[i]
    st = i.rsplit(".", 1)[0] if "." in i and i.rsplit(".", 1)[1].isdigit() else i
    if st in owner:
        return owner[st]
    src = h.get("source")
    return [src.split("#", 1)[1] if src and src.startswith("session_") and "#" in src else src]


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
            kw = {"role": u["role"]} if u.get("role") else {}
            if u.get("entities"):
                kw["entities"] = u["entities"]
            r = db.remember(u["text"], ns=ns, at=u["at"], source=u.get("source", u["src"]), **kw)
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
            "srcs": owner_of(owner, h),
            "role": h.get("role"),
            "context": (h.get("context") or "")[:4000] or None,
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
        files = {c: od / f"{x['question_id']}.json" for c, od in outdirs.items()}
        if all(f.exists() for f in files.values()):
            continue
        units = C.lme_units(x, args.unit)
        ns = "q_" + x["question_id"]
        db = open_db(args.variant, args.device)
        t0 = time.perf_counter()
        owner, st, lat = ingest(db, ns, units)
        ingest_s = time.perf_counter() - t0
        ro0 = {"as_of": C.lme_ts(x["question_date"])} if args.as_of == "question" else {}
        per = {}
        for cname, copts in configs.items():
            ro = {**ro0, **copts}
            hits10, ms10 = recall(db, ns, x["question"], owner, 10, **ro)
            hits5, ms5 = recall(db, ns, x["question"], owner, 5, **ro)
            hp = {}
            if args.variant == "sup":  # the read policy that acts on pending supersedes
                hp["hits10_hide_pending"], _ = recall(db, ns, x["question"], owner, 10, hide_pending=True, **ro)
                hp["hits5_hide_pending"], _ = recall(db, ns, x["question"], owner, 5, hide_pending=True, **ro)
            if args.context_budget:
                c = db.recall(x["question"], ns=ns, k=10, format="context", budget=args.context_budget, **ro)
                hp["ctx10"] = {"text": c["text"], "tokens": c["tokens"], "ids": c["ids"], "omitted": c["omitted"]}
            per[cname] = (hits10, ms10, hits5, ms5, hp)
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
        }
        for cname, (hits10, ms10, hits5, ms5, hp) in per.items():
            f = files[cname]
            r2 = {**rec, "config": configs[cname], "recall_ms": {"k10": ms10, "k5": ms5},
                  "hits10": hits10, "hits5": hits5, **hp}
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(r2))
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
        files = {c: od / f"{sid}.json" for c, od in outdirs.items()}
        if all(f.exists() for f in files.values()):
            continue
        units = C.locomo_units(conv)
        for u in units:  # source group = the session, for the per-source cap
            u["source"] = f"{u['session']}#{u['src']}"
            if args.entities == "caps":
                u["entities"] = C.caps_entities(u["text"])
        ns = "c_" + sid
        db = open_db(args.variant, args.device)
        t0 = time.perf_counter()
        owner, st, lat = ingest(db, ns, units)
        ingest_s = time.perf_counter() - t0
        per = {c: [] for c in configs}
        for qi, q in enumerate(conv["qa"]):
            for cname, copts in configs.items():
                h10, ms10 = recall(db, ns, q["question"], owner, 10, **copts)
                h5, ms5 = recall(db, ns, q["question"], owner, 5, **copts)
                hp = {}
                if args.variant == "sup":
                    hp["hits10_hide_pending"], _ = recall(db, ns, q["question"], owner, 10, hide_pending=True, **copts)
                    hp["hits5_hide_pending"], _ = recall(db, ns, q["question"], owner, 5, hide_pending=True, **copts)
                if args.context_budget:
                    c = db.recall(q["question"], ns=ns, k=10, format="context", budget=args.context_budget, **copts)
                    hp["ctx10"] = {"text": c["text"], "tokens": c["tokens"], "ids": c["ids"], "omitted": c["omitted"]}
                per[cname].append({**hp,
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
        for cname, qs in per.items():
            rec = {"sample_id": sid, "n_units": len(units), "ingest": st, "ingest_s": ingest_s, "config": configs[cname],
                   "remember_ms": lat, "last_session_date": max(u["at"] for u in units), "questions": qs}
            f = files[cname]
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec))
            tmp.rename(f)
        p50 = sorted(lat)[len(lat) // 2] if lat else 0
        print(f"{sid}: {len(units)} turns in {ingest_s:.1f}s (p50 {p50:.0f} ms), sup {st['superseded']} "
              f"dup {st['duplicate']} pend {st['pending']}, {len(conv['qa'])} questions", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["lme", "lme-oracle", "locomo"], required=True)
    ap.add_argument("--variant", choices=["fts", "hybrid", "sup"], required=True)
    ap.add_argument("--unit", default="user", choices=["user", "user+assistant", "round", "turn", "session"])
    ap.add_argument("--configs", default=None, help="recall option sets, e.g. 'base,default' (see CONFIGS) or "
                    "'name={\"max_per_source\": 2}'; results go to <variant>+<name>")
    ap.add_argument("--context-budget", type=int, default=0, help="also store recall's context block (tokens)")
    ap.add_argument("--entities", default=None, choices=[None, "caps"],
                    help="LoCoMo: pass entities to remember (caps: the speaker and capitalised words)")
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
    vdir = args.variant + ("@q" if args.as_of == "question" else "") + ("+ents" if args.entities else "")
    configs = parse_configs(args.configs) or {"": {}}
    outdirs = {}
    for cname in configs:
        outdir = C.bench_dir() / "results" / args.tag / f"{args.bench}-{unit}" / (vdir + (f"+{cname}" if cname else ""))
        outdir.mkdir(parents=True, exist_ok=True)
        meta = outdir / "meta.json"
        if not meta.exists():
            import kleos  # noqa: F401
            meta.write_text(json.dumps({
                "tag": args.tag, "kleos_file": getattr(sys.modules["kleos"], "__file__", None),
                "lme": [C.LME_REPO, C.LME_REV], "locomo": [C.LOCOMO_REPO, C.LOCOMO_REV], "config": configs[cname],
                "device": args.device, "seed": args.seed, "unit": unit, "as_of": args.as_of, "started": time.time(),
            }, indent=1))
        print(f"-> {outdir}", flush=True)
        outdirs[cname] = outdir
    globals()["configs"] = configs
    globals()["outdirs"] = outdirs
    (run_locomo if args.bench == "locomo" else run_lme)(args, None)


if __name__ == "__main__":
    main()
