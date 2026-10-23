# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Entities and claim type on the 300-claim sample, against the silver labels.

One GLiNER `batch_extract` call per batch with a combined schema (entities
with descriptions + a `type` classification), at a low threshold with
confidences, so the scores can be read at several thresholds afterwards.

    python eval_entities.py predict small|base|multi [cpu|mps]   # -> pred_<model>.json
    python eval_entities.py score small base multi                # tables
"""
import json
import sys
import time
from collections import Counter, defaultdict

import common as C
from silver import ENT_LABELS, TYPES

ENT_DESC = {
    "person": "named person or named pet",
    "location": "named place: city, country, region, venue, landmark",
    "organization": "named company, school, institution, team, website or service",
    "product": "named product, brand, model, app or device",
    "date": "date, day, month, year, time or relative time expression",
    "quantity": "number with or without unit: count, money, duration, distance, percentage, age",
    "event": "named event, holiday or occasion",
    "title": "title of a book, film, song, show, game, course or artwork",
}
TYPE_DESC = {
    "fact": "states something about the speaker, someone or the world",
    "preference": "a like, dislike, favourite or opinion of the speaker",
    "plan": "an intention, plan or future arrangement",
    "event": "something that happened at a particular time",
    "question": "asks for information",
    "request": "asks the listener to do or produce something",
    "small_talk": "greeting, thanks, reaction or compliment with no information",
}
LOW = 0.2


def schema(m, described=True):
    s = m.create_schema().entities(ENT_DESC if described else ENT_LABELS)
    return s.classification("type", TYPE_DESC if described else TYPES)


def predict(name, device="cpu", described=True):
    m = C.load_gliner(name, device)
    sample = json.loads((C.GDIR / "sample.json").read_text())
    texts = [x["text"] for x in sample]
    t0 = time.perf_counter()
    res = m.batch_extract(texts, schema(m, described), batch_size=8, threshold=LOW, include_confidence=True, include_spans=True)
    dt = time.perf_counter() - t0
    # the classification head's full distribution, for the type scores
    cls = m.batch_classify_text(texts, {"type": {"labels": TYPE_DESC if described else TYPES}}, batch_size=8, include_confidence=True)
    out = []
    for x, r, c in zip(sample, res, cls):
        ents = []
        for lab, lst in (r.get("entities") or {}).items():
            for e in lst:
                ents.append({"label": lab, "text": e["text"], "start": e.get("start"), "end": e.get("end"), "p": e.get("confidence", 1.0)})
        out.append({"id": x["id"], "entities": ents, "type_joint": r.get("type"), "type": c.get("type")})
    tag = name + ("" if described else "-bare")
    (C.GDIR / f"pred_{tag}.json").write_text(json.dumps({"model": name, "device": device, "secs": dt, "n": len(texts), "pred": out}, indent=1))
    print(tag, f"{dt:.1f}s for {len(texts)} claims ({1000 * dt / len(texts):.1f} ms/claim, batch 8, {device})")


def _norm(s):
    return " ".join(s.lower().strip(" .,!?'\"").split())


def _spans(text, ents, key="text"):
    out = []
    for e in ents:
        st = e.get("start")
        if st is None:
            st = text.find(e[key])
        out.append((st, st + len(e[key]), e["label"], _norm(e[key])))
    return out


def prf(tp, np_, ng):
    p = tp / np_ if np_ else 0.0
    r = tp / ng if ng else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score_entities(sample, silver, pred, thr):
    by = {s["id"]: s for s in silver}
    strict = Counter()
    loose = Counter()
    per = defaultdict(Counter)
    for x, pr in zip(sample, pred):
        g = _spans(x["text"], by[x["id"]].get("entities", []))
        p = _spans(x["text"], [e for e in pr["entities"] if e["p"] >= thr])
        strict["np"] += len(p)
        strict["ng"] += len(g)
        for lab in ENT_LABELS:
            per[lab]["np"] += sum(1 for e in p if e[2] == lab)
            per[lab]["ng"] += sum(1 for e in g if e[2] == lab)
        used = set()
        for e in p:
            hit = next((i for i, h in enumerate(g) if i not in used and h[2] == e[2] and h[3] == e[3]), None)
            if hit is not None:
                used.add(hit)
                strict["tp"] += 1
                per[e[2]]["tp"] += 1
        used = set()
        for e in p:  # overlap + same label
            hit = next((i for i, h in enumerate(g) if i not in used and h[2] == e[2] and e[0] < h[1] and h[0] < e[1]), None)
            if hit is not None:
                used.add(hit)
                loose["tp"] += 1
        used = set()
        for e in p:  # overlap, any label (mention detection)
            hit = next((i for i, h in enumerate(g) if i not in used and e[0] < h[1] and h[0] < e[1]), None)
            if hit is not None:
                used.add(hit)
                loose["tp_any"] += 1
    return {
        "strict": prf(strict["tp"], strict["np"], strict["ng"]),
        "overlap": prf(loose["tp"], strict["np"], strict["ng"]),
        "mention": prf(loose["tp_any"], strict["np"], strict["ng"]),
        "per_label": {lab: prf(per[lab]["tp"], per[lab]["np"], per[lab]["ng"]) + (per[lab]["ng"],) for lab in ENT_LABELS},
        "n_pred": strict["np"], "n_gold": strict["ng"],
    }


def score_types(sample, silver, pred, key="type"):
    by = {s["id"]: s for s in silver}
    ok = n = 0
    conf = Counter()
    coarse_ok = 0
    nondecl = {"question", "request", "small_talk"}
    rule_ok = 0
    for x, pr in zip(sample, pred):
        g = by[x["id"]].get("type")
        if g not in TYPES:
            continue
        p = pr.get(key)
        if isinstance(p, dict):
            p = p.get("label")
        n += 1
        ok += p == g
        conf[(g, p)] += 1
        coarse_ok += (p in nondecl) == (g in nondecl)
        rule_ok += (not x["declarative"]) == (g in nondecl)
    return {"n": n, "acc": ok / n, "store_vs_skip_acc": coarse_ok / n, "kleos_rule_store_vs_skip_acc": rule_ok / n, "confusion": {f"{a}->{b}": v for (a, b), v in conf.most_common(12)}}


NAMED = {"person", "location", "organization", "product", "event", "title"}


def named_filter(pred):
    """Keep a named-type span only if it has a capital letter (and is not
    the pronoun "I"): GLiNER labels generic nouns ("farmers", "koalas")."""
    keep = lambda e: e["label"] not in NAMED or (any(c.isupper() for c in e["text"]) and e["text"] != "I")  # noqa: E731
    return [{**p, "entities": [e for e in p["entities"] if keep(e)]} for p in pred]


def score(names):
    sample = json.loads((C.GDIR / "sample.json").read_text())
    silver = json.loads((C.GDIR / "silver.json").read_text())
    gold_types = Counter(s.get("type") for s in silver)
    gold_ents = Counter(e["label"] for s in silver for e in s.get("entities", []))
    print("silver types", dict(gold_types))
    print("silver entities", dict(gold_ents))
    report = {}
    for name in names:
        P = json.loads((C.GDIR / f"pred_{name}.json").read_text())
        r = {"ms_per_claim": 1000 * P["secs"] / P["n"]}
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            r[f"ents@{thr}"] = score_entities(sample, silver, P["pred"], thr)
        for thr in (0.5, 0.7):
            r[f"ents_named@{thr}"] = score_entities(sample, silver, named_filter(P["pred"]), thr)
        r["type"] = score_types(sample, silver, P["pred"])
        r["type_joint"] = score_types(sample, silver, P["pred"], "type_joint")
        report[name] = r
        print(f"\n== {name}  ({r['ms_per_claim']:.1f} ms/claim)")
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            e = r[f"ents@{thr}"]
            print(f" thr {thr}: strict P/R/F {e['strict'][0]:.2f}/{e['strict'][1]:.2f}/{e['strict'][2]:.2f}  overlap {e['overlap'][0]:.2f}/{e['overlap'][1]:.2f}/{e['overlap'][2]:.2f}  mention {e['mention'][0]:.2f}/{e['mention'][1]:.2f}/{e['mention'][2]:.2f}  pred {e['n_pred']} gold {e['n_gold']}")
        for thr in (0.5, 0.7):
            e = r[f"ents_named@{thr}"]
            print(f" + capital-letter filter, thr {thr}: strict {e['strict'][0]:.2f}/{e['strict'][1]:.2f}/{e['strict'][2]:.2f}  overlap {e['overlap'][0]:.2f}/{e['overlap'][1]:.2f}/{e['overlap'][2]:.2f}  pred {e['n_pred']}")
            print("   per label (P/R/F, n gold):", {k: tuple(round(v, 2) for v in vals[:3]) + (vals[3],) for k, vals in e["per_label"].items()})
        e = r["ents@0.5"]
        print("  per label @0.5 (P/R/F, n gold):", {k: tuple(round(v, 2) for v in vals[:3]) + (vals[3],) for k, vals in e["per_label"].items()})
        print("  type:", {k: v for k, v in r["type"].items() if k != "confusion"}, r["type"]["confusion"])
        print("  type (joint call):", r["type_joint"]["acc"])
    (C.GDIR / "entities_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "predict":
        predict(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "cpu", described=not (len(sys.argv) > 4 and sys.argv[4] == "bare"))
    else:
        score(sys.argv[2:])
