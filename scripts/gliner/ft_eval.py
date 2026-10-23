# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Zero-shot vs fine-tuned GLiNER2.5 on kleos's tasks, against the hand-checked
gold set (gold.json). Thresholds per entity label are tuned on the silver
validation split (val_items.json) for each model separately, then frozen.

    python ft_eval.py predict <model-or-ckpt-dir> <tag> [mps|cpu]   # pred_<tag>_{val,gold}.json
    python ft_eval.py score <tag> [<tag> ...]                        # tables -> ft_report.json
"""
import json
import os
import re
import sys
import time
from collections import Counter

os.environ.setdefault("GLINER_DIR", "/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr/.bench-data/gliner-ft")
import common as C  # noqa: E402
from ft_data import ENTS, KINDS, STORE  # noqa: E402
from ft_train import ABOUT, ABOUT_PROMPT, ENT_DESC, KIND_DESC, PRONOUNS  # noqa: E402
from kleos_rules import is_declarative  # noqa: E402

LOW = 0.1
GRID = [round(0.1 * i, 1) for i in range(1, 10)]
FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself|we|we're|we've|our|ours|us)\b", re.I)


def schema(m, described=True):
    """Zero-shot models get label descriptions (their best setting); the
    fine-tuned ones were trained on bare label names (ft_train.example)."""
    if not described:
        return m.create_schema().entities(ENTS).classification("kind", KINDS).classification("about_speaker", ABOUT)
    return (m.create_schema().entities(ENT_DESC)
            .classification("kind", KIND_DESC)
            .classification("about_speaker", ABOUT, prompt=ABOUT_PROMPT))


def load(path, device):
    from gliner2 import AutoExtractor
    m = AutoExtractor.from_pretrained(C.MODELS.get(path, path), map_location=device)
    m.eval()
    return m


def predict(path, tag, device="mps", splits=("val", "gold")):
    m = load(path, device)
    s = schema(m, described=path in C.MODELS)
    for split, f in (("val", "val_items.json"), ("gold", "gold.json")):
        if split not in splits:
            continue
        items = json.loads((C.GDIR / f).read_text())
        t0 = time.perf_counter()
        res = m.batch_extract([x["text"] for x in items], s, batch_size=8, threshold=LOW, include_confidence=True, include_spans=True)
        dt = time.perf_counter() - t0
        out = []
        for x, r in zip(items, res):
            ents = [{"label": lab, "text": e["text"], "start": e.get("start"), "end": e.get("end"), "p": e.get("confidence", 1.0)}
                    for lab, lst in (r.get("entities") or {}).items() for e in lst]
            out.append({"id": x["id"], "entities": ents, "kind": r.get("kind"), "about_speaker": r.get("about_speaker")})
        (C.GDIR / f"pred_{tag}_{split}.json").write_text(json.dumps({"path": path, "device": device, "secs": dt, "n": len(items), "pred": out}, indent=1))
        print(tag, split, f"{1000 * dt / len(items):.1f} ms/claim (batch 8, {device})")


def norm(s):
    return " ".join(s.lower().strip(" .,!?'\"").split())


def _lab(v):
    return v.get("label") if isinstance(v, dict) else v


def split_relatives(ents):
    """A relative span that contains a person span ("my partner Alex", as the
    silver labels often had it) becomes the relative without the name
    ("my partner") plus that person; the pair is kept for name_of."""
    per = [e for e in ents if e["label"] == "person" and e.get("start") is not None]
    out, pairs = [], set()
    for e in ents:
        if e["label"] == "relative" and e.get("start") is not None:
            inner = next((p for p in per if p["start"] >= e["start"] and p["end"] <= e["end"]), None)
            if inner is not None:
                rel = (e["text"][:inner["start"] - e["start"]] + e["text"][inner["end"] - e["start"]:]).strip(" ,")
                if rel:
                    pairs.add((norm(rel), norm(inner["text"])))
                    e = {**e, "text": rel}
        out.append(e)
    return out, pairs


def post(p):
    """The same cleanup as the training labels (ft_train.clean): a person,
    place or organization name has a capital letter and is not a pronoun."""
    keep = [e for e in p["entities"] if e["label"] == "relative"
            or (any(c.isupper() for c in e["text"]) and e["text"].lower() not in PRONOUNS)]
    ents, pairs = split_relatives(keep)
    return {**p, "entities": ents, "_pairs": pairs}


def ent_counts(items, preds, thr, labels=ENTS):
    c = {lab: Counter() for lab in labels}
    for x, p in zip(items, preds):
        for lab in labels:
            g = {norm(e["text"]) for e in x["entities"] if e["label"] == lab}
            q = {norm(e["text"]) for e in p["entities"] if e["label"] == lab and e["p"] >= thr[lab]}
            c[lab]["tp"] += len(g & q)
            c[lab]["np"] += len(q)
            c[lab]["ng"] += len(g)
    return c


def prf(c):
    p = c["tp"] / c["np"] if c["np"] else 0.0
    r = c["tp"] / c["ng"] if c["ng"] else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def tune(items, preds):
    thr = {}
    for lab in ENTS:
        best = max(GRID, key=lambda t: (prf(ent_counts(items, preds, {lab: t}, [lab])[lab])[2], t))
        thr[lab] = best
    return thr


def pairs_rule(text, ents):
    """name_of by adjacency: "my sister Anna", "Anna, my sister", "my friend (Tom)"."""
    rel = [e for e in ents if e["label"] == "relative" and e.get("start") is not None]
    per = [e for e in ents if e["label"] == "person" and e.get("start") is not None]
    out = set()
    for r in rel:
        for p in per:
            if r["end"] <= p["start"] and re.fullmatch(r"\s*(,|\(|named|called|-|–|—)?\s*", text[r["end"]:p["start"]]):
                out.add((norm(r["text"]), norm(p["text"])))
            elif p["end"] <= r["start"] and re.fullmatch(r"\s*(,|\(|is|-|–|—)?\s*", text[p["end"]:r["start"]]):
                out.add((norm(r["text"]), norm(p["text"])))
    return out


_B = None


def caps_names(text):
    """bench-l2's model-free name guess (capitalised runs that are not
    sentence openers or fillers), without the speaker it adds."""
    global _B
    if _B is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("bench_common", C.CHECKOUT / "scripts" / "bench-l2" / "common.py")
        _B = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_B)
    return {norm(n) for n in _B.caps_entities("x: " + text)[1:]}


def rule_store(text):
    """kleos's rule filter (memory_claims::is_declarative), ported in
    kleos_rules.py; it matches kleos's own `decided` on 2,805 of 2,805 claims."""
    return is_declarative(text)


def score_one(tag, gold, P, Pv, val):
    Pv = [post(p) for p in Pv]
    thr = tune(val, Pv)
    preds = [post(p) for p in P["pred"]]
    r = {"ms_per_claim": 1000 * P["secs"] / P["n"], "thresholds": thr}
    c = ent_counts(gold, preds, thr)
    r["entities"] = {lab: prf(c[lab]) + (c[lab]["ng"],) for lab in ENTS}
    c5 = ent_counts(gold, preds, {lab: 0.5 for lab in ENTS})
    r["entities@0.5"] = {lab: prf(c5[lab]) + (c5[lab]["ng"],) for lab in ENTS}
    names = Counter()
    for lab in ("person", "location", "organization"):
        for k in ("tp", "np", "ng"):
            names[k] += c[lab][k]
    r["names_any_label"] = prf(names)
    # name_of pairs via the adjacency rule on thresholded spans
    pc = Counter()
    for x, p in zip(gold, preds):
        g = {(norm(a), norm(b)) for a, b in x.get("name_of", [])}
        q = pairs_rule(x["text"], [e for e in p["entities"] if e["p"] >= thr[e["label"]]]) | p["_pairs"]
        pc["tp"] += len(g & q)
        pc["np"] += len(q)
        pc["ng"] += len(g)
    r["name_of"] = prf(pc) + (pc["ng"],)
    # classification
    ok = sk = ab = 0
    conf = Counter()
    for x, p in zip(gold, preds):
        k = _lab(p["kind"])
        ok += k == x["kind"]
        sk += (k in STORE) == (x["kind"] in STORE)
        ab += (_lab(p["about_speaker"]) == "yes") == x["about_speaker"]
        if k != x["kind"]:
            conf[f"{x['kind']}->{k}"] += 1
    n = len(gold)
    r["kind_acc"], r["store_acc"], r["about_acc"] = ok / n, sk / n, ab / n
    hy = sum((rule_store(x["text"]) and _lab(p["kind"]) != "small_talk") == (x["kind"] in STORE) for x, p in zip(gold, preds))
    r["store_rule_and_not_smalltalk_acc"] = hy / n
    rnd = [i for i, x in enumerate(gold) if x["part"] == "random"]
    r["random150"] = {
        "store_acc": sum((_lab(preds[i]["kind"]) in STORE) == (gold[i]["kind"] in STORE) for i in rnd) / len(rnd),
        "store_hybrid_acc": sum((rule_store(gold[i]["text"]) and _lab(preds[i]["kind"]) != "small_talk") == (gold[i]["kind"] in STORE) for i in rnd) / len(rnd),
        "kind_acc": sum(_lab(preds[i]["kind"]) == gold[i]["kind"] for i in rnd) / len(rnd),
    }
    r["kind_confusion"] = dict(conf.most_common(8))
    # by source
    for src in ("lme", "locomo"):
        idx = [i for i, x in enumerate(gold) if x["src"] == src]
        cs = ent_counts([gold[i] for i in idx], [preds[i] for i in idx], thr)
        nm = Counter()
        for lab in ("person", "location", "organization"):
            for k in ("tp", "np", "ng"):
                nm[k] += cs[lab][k]
        r[f"{src}_names_f"] = prf(nm)
        r[f"{src}_store_acc"] = sum((_lab(preds[i]["kind"]) in STORE) == (gold[i]["kind"] in STORE) for i in idx) / len(idx)
    return r


def baselines(gold):
    n = len(gold)
    store = sum(rule_store(x["text"]) == (x["kind"] in STORE) for x in gold) / n
    about = sum(bool(FIRST_PERSON.search(x["text"])) == x["about_speaker"] for x in gold) / n
    c = Counter()
    for x in gold:
        g = {norm(e["text"]) for e in x["entities"] if e["label"] in ("person", "location", "organization")}
        q = caps_names(x["text"])
        c["tp"] += len(g & q)
        c["np"] += len(q)
        c["ng"] += len(g)
    majority = Counter(x["kind"] for x in gold).most_common(1)[0]
    rnd = [x for x in gold if x["part"] == "random"]
    store_r = sum(rule_store(x["text"]) == (x["kind"] in STORE) for x in rnd) / len(rnd)
    return {"rule_store_acc": store, "rule_store_acc_random150": store_r, "rule_about_acc": about, "caps_names_prf": prf(c), "kind_majority": [majority[0], majority[1] / n],
            "gold_counts": {lab: sum(1 for x in gold for e in x["entities"] if e["label"] == lab) for lab in ENTS},
            "gold_pairs": sum(len(x.get("name_of", [])) for x in gold), "gold_kinds": dict(Counter(x["kind"] for x in gold)), "n": n}


def score(tags):
    gold = json.loads((C.GDIR / "gold.json").read_text())
    val = json.loads((C.GDIR / "val_items.json").read_text())
    rep = {"baselines": baselines(gold)}
    print(json.dumps(rep["baselines"]))
    for tag in tags:
        P = json.loads((C.GDIR / f"pred_{tag}_gold.json").read_text())
        Pv = json.loads((C.GDIR / f"pred_{tag}_val.json").read_text())["pred"]
        r = rep[tag] = score_one(tag, gold, P, Pv, val)
        f = lambda t: "/".join(f"{v:.2f}" for v in t[:3])  # noqa: E731
        print(f"\n== {tag}  ({r['ms_per_claim']:.0f} ms/claim)  thresholds {r['thresholds']}")
        print("  entities (tuned):", {k: f(v) + f" n={v[3]}" for k, v in r["entities"].items()})
        print("  entities (@0.5): ", {k: f(v) for k, v in r["entities@0.5"].items()})
        print("  names any label P/R/F", f(r["names_any_label"]), "| lme", f(r["lme_names_f"]), "| locomo", f(r["locomo_names_f"]))
        print("  name_of pairs P/R/F", f(r["name_of"]), "n", r["name_of"][3])
        print(f"  kind {r['kind_acc']:.3f}  store {r['store_acc']:.3f} (lme {r['lme_store_acc']:.2f}, locomo {r['locomo_store_acc']:.2f})  rule-and-not-small-talk {r['store_rule_and_not_smalltalk_acc']:.3f}  about {r['about_acc']:.3f}")
        print("  random 150:", {k: round(v, 3) for k, v in r["random150"].items()})
        print("  kind errors", r["kind_confusion"])
    (C.GDIR / "ft_report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "predict":
        predict(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "mps", tuple(sys.argv[5].split(",")) if len(sys.argv) > 5 else ("val", "gold"))
    else:
        score(sys.argv[2:])
