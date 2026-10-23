#!/usr/bin/env python3
# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Aggregate cached retrieval and QA results into markdown tables.

    python report.py [--tag 276f90f] [--compare OLD_TAG]

Writes <bench-dir>/results/<tag>/report.md and summary.json and prints the
report. With --compare, retrieval rows are restricted to the questions both
tags ran and printed side by side (old -> new).

Metric definitions (docs/benchmarks/memory.md):
  LongMemEval  R@k      a top-k memory comes from an answer session (LongMemEval's recall_any, session level)
               Rall@k   every answer session has a memory in the top k (recall_all)
               T@k      a top-k memory is an evidence turn (has_answer), turn level; questions with evidence turns
               NDCG@10  sessions ranked by first appearance in the top 10, binary relevance (ndcg_any, session level)
               KU latest@k  the latest-dated answer session is in the top k
               KU stale-first  the first answer-session memory is from an older answer session
  LoCoMo       R@k / Rall@k / NDCG@10 over evidence dialog ids (turn level)
@5 numbers come from a separate recall(k=5) call, @10 from recall(k=10).
"""
import argparse
import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

VARIANTS = ["fts", "hybrid", "sup", "fts@q", "hybrid@q", "sup@q"]
LME_CATS = ["single-session", "multi-session", "temporal-reasoning", "knowledge-update", "abstention"]
SS_TYPES = ["single-session-user", "single-session-assistant", "single-session-preference"]


def ss_groups(key="question_type"):
    """single-session split by subtype (abstention questions excluded, as in the category rows)."""
    return [("  " + t.replace("single-session-", "ss-"),
             (lambda t: lambda r: r.get("category") == "single-session" and r.get(key) == t)(t)) for t in SS_TYPES]


def load_dir(d):
    out = {}
    if d.is_dir():
        for p in d.glob("*.json"):
            if p.stem != "meta":
                out[p.stem] = json.loads(p.read_text())
    return out


def sess_of(h):
    return {s.split("#")[0] for s in h["srcs"] if s}


def lme_metrics(r):
    gold = set(r["answer_session_ids"])
    ev = set(r["evidence_srcs"])
    m = {}
    for k, hits in ((5, r["hits5"]), (10, r["hits10"])):
        top = hits[:k]
        ss = set().union(*[sess_of(h) for h in top]) if top else set()
        m[f"R@{k}"] = float(bool(ss & gold))
        m[f"Rall@{k}"] = float(gold <= ss)
        if ev:
            m[f"T@{k}"] = float(any(set(h["srcs"]) & ev for h in top))
    order = []
    for h in r["hits10"]:
        for s in sorted(sess_of(h)):
            if s not in order:
                order.append(s)
    m["NDCG@10"] = C.ndcg([1 if s in gold else 0 for s in order], len(gold), 10)
    dates = r.get("answer_session_dates") or {}
    if r["question_type"] == "knowledge-update" and len(gold) >= 2 and all(dates.values()):
        latest = max(gold, key=lambda s: C.lme_ts(dates[s]))
        for k, hits in ((5, r["hits5"]), (10, r["hits10"])):
            m[f"KUlatest@{k}"] = float(any(latest in sess_of(h) for h in hits[:k]))
        first = next((h for h in r["hits10"] if sess_of(h) & gold), None)
        if first is not None:
            m["KUstale1st"] = float(latest not in sess_of(first))
    return m


def locomo_metrics(q):
    ev = set(q["evidence"])
    if not ev:
        return None
    m = {}
    for k, hits in ((5, q["hits5"]), (10, q["hits10"])):
        got = set().union(*[set(h["srcs"]) for h in hits[:k]]) if hits else set()
        m[f"R@{k}"] = float(bool(got & ev))
        m[f"Rall@{k}"] = float(ev <= got)
    m["NDCG@10"] = C.ndcg([1 if set(h["srcs"]) & ev else 0 for h in q["hits10"]], len(ev), 10)
    return m


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def f2(x):
    return "–" if x is None else f"{x:.3f}"


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def agg(ms, keys):
    return {k: mean([m.get(k) for m in ms]) for k in keys}


def load_lme(d):
    out = {v: load_dir(d / v) for v in VARIANTS}
    if out.get("sup"):
        out["sup+hide_pending"] = {
            i: {**r, "hits5": r["hits5_hide_pending"], "hits10": r["hits10_hide_pending"]}
            for i, r in out["sup"].items() if "hits10_hide_pending" in r}
    return {v: r for v, r in out.items() if r}


def lme_section(md, summ, lme, keys, held, ku_train):
    for v, recs in lme.items():
        rows = []
        groups = [("all", lambda r: True)] + [(c, (lambda c: lambda r: r["category"] == c)(c)) for c in LME_CATS]
        groups = groups[:2] + ss_groups() + groups[2:]
        groups += [("KU held-out", lambda r: r["question_id"] in held),
                   ("KU trained-on", lambda r: r["question_id"] in ku_train)]
        for name, pred in groups:
            sel = [r for r in recs.values() if pred(r)]
            if not sel:
                continue
            a = agg([lme_metrics(r) for r in sel], keys)
            rows.append([name, len(sel)] + [f2(a[k]) for k in keys])
            summ.setdefault(v, {})[name] = {"n": len(sel), **a}
        md.append(f"### variant `{v}` ({len(recs)} questions)\n")
        md.append(table(rows, ["split", "n"] + keys) + "\n")
    if len(lme) < 2:
        return
    # paired: per split, the questions every variant that ran that split has
    rows = []
    splits = [("all", lambda r: True)] + [(c, (lambda c: lambda r: r["category"] == c)(c)) for c in LME_CATS]
    for name, pred in splits:
        vs = [v for v in lme if any(pred(r) for r in lme[v].values())]
        ids = set.intersection(*[{i for i, r in lme[v].items() if pred(r)} for v in vs]) if vs else set()
        if len(vs) < 2 or not ids:
            continue
        for v in vs:
            a = agg([lme_metrics(lme[v][i]) for i in ids], keys)
            rows.append([name, v, len(ids)] + [f2(a[k]) for k in keys])
    if rows:
        md.append("### Variants paired on the same questions (per split: the questions every listed variant ran)\n")
        md.append(table(rows, ["split", "variant", "n"] + keys) + "\n")
    kk = ["KUlatest@5", "KUlatest@10", "KUstale1st"]
    rows = []
    for split, pred in (("KU all", lambda i: True), ("KU held-out", lambda i: i in held),
                        ("KU trained-on", lambda i: i in ku_train)):
        isku = lambda r: r["question_type"] == "knowledge-update" and not r["question_id"].endswith("_abs")
        vs = [v for v in lme if any(isku(r) and pred(i) for i, r in lme[v].items())]
        if not vs:
            continue
        ids = set.intersection(*[{i for i, r in lme[v].items() if isku(r) and pred(i)} for v in vs])
        for v in vs:
            ms = [lme_metrics(lme[v][i]) for i in ids]
            ms = [m for m in ms if "KUlatest@5" in m]
            if not ms:
                continue
            a = agg(ms, kk)
            sup = mean([lme[v][i]["ingest"]["superseded"] for i in ids])
            pend = mean([lme[v][i]["ingest"]["pending"] for i in ids])
            rows.append([split, v, len(ms)] + [f2(a[k]) for k in kk] + [f"{sup:.2f}", f"{pend:.1f}"])
            summ.setdefault(v, {})[f"{split} supersession"] = {"n": len(ms), **a, "superseded_per_q": sup, "pending_per_q": pend}
    if rows:
        md.append("Knowledge update, supersession view (non-abstention KU questions with ≥2 dated answer sessions; "
                  "stale-first: lower is better; superseded/pending: decisions per question at ingest):\n")
        md.append(table(rows, ["split", "variant", "n", "latest@5", "latest@10", "stale-first",
                               "superseded/q", "pending/q"]) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=os.environ.get("KLEOS_BENCH_TAG") or C.git_rev())
    ap.add_argument("--unit", default="user")
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()
    base = C.bench_dir() / "results" / args.tag
    md = [f"# kleos memory benchmark: results for `{args.tag}`\n"]
    summary = {"tag": args.tag, "lme": {}, "locomo": {}, "throughput": {}, "qa": {}}
    keys = ["R@5", "R@10", "Rall@5", "Rall@10", "T@5", "T@10", "NDCG@10"]

    # ------------------------------------------------ LongMemEval retrieval
    held, ku_train, _ = C.lme_ku_split()
    lme = load_lme(base / f"lme-{args.unit}")
    old = load_lme(C.bench_dir() / "results" / args.compare / f"lme-{args.unit}") if args.compare else {}
    md.append(f"## LongMemEval_S retrieval (unit = {args.unit})\n")
    lme_section(md, summary.setdefault("lme", {}), lme, keys, held, ku_train)
    if args.compare and old:
        md.append(f"### Compare `{args.compare}` → `{args.tag}` (same questions)\n")
        rows = []
        for v in lme:
            ids = set(lme[v]) & set(old.get(v, {}))
            if not ids:
                continue
            for name in ["all"] + LME_CATS:
                sel = [i for i in ids if name == "all" or lme[v][i]["category"] == name]
                if not sel:
                    continue
                a0 = agg([lme_metrics(old[v][i]) for i in sel], keys)
                a1 = agg([lme_metrics(lme[v][i]) for i in sel], keys)
                rows.append([v, name, len(sel)] + [f"{f2(a0[k])} → {f2(a1[k])}" for k in ("R@5", "R@10", "NDCG@10")])
        md.append(table(rows, ["variant", "split", "n", "R@5", "R@10", "NDCG@10"]) + "\n")

    # ------------------------------------------------ oracle file (sanity + supersession on KU)
    orc = load_lme(base / f"lme-oracle-{args.unit}")
    if orc:
        md.append("## LongMemEval oracle file (only the evidence sessions are stored)\n")
        md.append("All 500 with `fts` is a sanity check. On knowledge-update questions the store holds both the old "
                  "and the new value, so this is the cheap supersession test.\n")
        lme_section(md, summary.setdefault("lme-oracle", {}), orc, keys, held, ku_train)

    # ------------------------------------------------ LoCoMo retrieval
    md.append("## LoCoMo retrieval (one memory per dialog turn)\n")
    lkeys = ["R@5", "R@10", "Rall@5", "Rall@10", "NDCG@10"]
    lrows = []
    allconvs = {v: load_dir(base / "locomo-turn" / v) for v in VARIANTS}
    allconvs = {v: c for v, c in allconvs.items() if c}
    shared = set.intersection(*[set(c) for c in allconvs.values()]) if allconvs else set()
    if allconvs:
        md.append(f"Paired: every variant on the same {len(shared)} of 10 conversations "
                  f"({', '.join(f'{v} ran {len(c)}' for v, c in allconvs.items())}).\n")
    for v, convs in allconvs.items():
        qs = [q for cid, c in convs.items() if cid in shared for q in c["questions"]]
        for name, pred in [("cat 1–4", lambda q: q["category"] != 5)] + [
                (f"{c} {C.LOCOMO_CATS[c]}", (lambda c: lambda q: q["category"] == c)(c)) for c in (1, 2, 3, 4, 5)]:
            ms = [m for m in (locomo_metrics(q) for q in qs if pred(q)) if m]
            if ms:
                a = agg(ms, lkeys)
                lrows.append([v, name, len(ms)] + [f2(a[k]) for k in lkeys])
                summary["locomo"].setdefault(v, {})[name] = {"n": len(ms), "convs": len(convs), **a}
    md.append(table(lrows, ["variant", "split", "n"] + lkeys) + "\n")
    if args.compare:
        crow = []
        for v in VARIANTS:
            a = load_dir(C.bench_dir() / "results" / args.compare / "locomo-turn" / v)
            b = load_dir(base / "locomo-turn" / v)
            ids = set(a) & set(b)
            if not ids:
                continue
            for name, pred in [("cat 1–4", lambda q: q["category"] != 5)] + [
                    (f"{c} {C.LOCOMO_CATS[c]}", (lambda c: lambda q: q["category"] == c)(c)) for c in (1, 2, 3, 4, 5)]:
                m0 = [m for m in (locomo_metrics(q) for i in ids for q in a[i]["questions"] if pred(q)) if m]
                m1 = [m for m in (locomo_metrics(q) for i in ids for q in b[i]["questions"] if pred(q)) if m]
                a0, a1 = agg(m0, lkeys), agg(m1, lkeys)
                crow.append([v, name, len(m1)] + [f"{f2(a0[k])} → {f2(a1[k])}" for k in ("R@5", "R@10", "NDCG@10")])
        if crow:
            md.append(f"### LoCoMo: compare `{args.compare}` → `{args.tag}` (same conversations)\n")
            md.append(table(crow, ["variant", "split", "n", "R@5", "R@10", "NDCG@10"]) + "\n")

    # ------------------------------------------------ throughput
    md.append("## Ingestion and recall throughput\n")
    trows = []
    for bench, sub, unitname in (("LongMemEval_S", f"lme-{args.unit}", "question"),
                                 ("LongMemEval oracle", f"lme-oracle-{args.unit}", "question"),
                                 ("LoCoMo", "locomo-turn", "conversation")):
        for v in VARIANTS:
            recs = load_dir(base / sub / v)
            if not recs:
                continue
            rem = [x for r in recs.values() for x in r["remember_ms"]]
            if bench == "LoCoMo":
                rc = [q["recall_ms"]["k10"] for r in recs.values() for q in r["questions"]]
            else:
                rc = [r["recall_ms"]["k10"] for r in recs.values()]
            secs = sum(r["ingest_s"] for r in recs.values())
            trows.append([bench, v, len(recs), len(rem), f"{len(rem) / secs:.1f}", f"{pct(rem, .5):.1f}",
                          f"{pct(rem, .95):.1f}", f"{pct(rc, .5):.1f}", f"{pct(rc, .95):.1f}"])
            summary["throughput"][f"{bench}/{v}"] = {"units": len(recs), "memories": len(rem), "mem_per_s": len(rem) / secs,
                                                   "remember_p50_ms": pct(rem, .5), "remember_p95_ms": pct(rem, .95),
                                                   "recall_p50_ms": pct(rc, .5), "recall_p95_ms": pct(rc, .95)}
    md.append(table(trows, ["bench", "variant", "units", "remembers", "remember/s", "remember p50 ms",
                            "remember p95 ms", "recall(k=10) p50 ms", "recall p95 ms"]) + "\n")

    # ------------------------------------------------ QA
    qa_base = base / "qa"
    if qa_base.is_dir():
        md.append("## QA accuracy (secondary; reader and judge = the same LLM, see caveats)\n")
        for sub, qset in sorted((p.parent.name, p.name) for p in qa_base.glob("*/*") if p.is_dir()):
            conds = {p.name: load_dir(p) for p in sorted((qa_base / sub / qset).iterdir()) if p.is_dir()}
            ids = None
            for c, rs in conds.items():
                ids = set(rs) if ids is None else ids & set(rs)
            if not ids:
                continue
            models = {r.get("reader_model") for rs in conds.values() for r in rs.values()}
            md.append(f"### {sub}, question set `{qset}` ({len(ids)} questions answered under every condition; "
                      f"reader/judge {', '.join(sorted(m for m in models if m))})\n")
            if sub.startswith("lme"):
                splits = [("all", lambda r: True)] + [(c, (lambda c: lambda r: r["category"] == c)(c)) for c in LME_CATS]
                splits = splits[:2] + ss_groups() + splits[2:]
                splits += [("KU held-out", lambda r: r["qid"] in held), ("KU trained-on", lambda r: r["qid"] in ku_train)]
            else:
                splits = [("cat 1–4", lambda r: True)] + [
                    (f"{c} {C.LOCOMO_CATS[c]}", (lambda c: lambda r: r["category"] == c)(c)) for c in (1, 2, 3, 4)]
            names = list(conds)
            rows = []
            for name, pred in splits:
                sel = [i for i in sorted(ids) if pred(conds[names[0]][i])]
                if not sel:
                    continue
                row = [name, len(sel)]
                for c in names:
                    acc = mean([float(conds[c][i]["correct"]) for i in sel])
                    row.append(f2(acc))
                    summary["qa"].setdefault(f"{sub}/{qset}", {}).setdefault(c, {})[name] = {"n": len(sel), "acc": acc}
                rows.append(row)
            md.append(table(rows, ["split", "n"] + names) + "\n")

    text = "\n".join(md)
    (base / "report.md").write_text(text)
    (base / "summary.json").write_text(json.dumps(summary, indent=1))
    print(text)


if __name__ == "__main__":
    main()
