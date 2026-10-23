# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Update keys: does GLiNER give the old and the new claim of a true update
the same (subject, attribute) key with different values, and how often do
unrelated claims share a key?

Schema variants (VARIANTS) were iterated on the 53 trained-on KU questions
only; the held-out 23 were run once per variant at the end.

    python ku_keys.py run MODEL VARIANT trained|held [device]
"""
import json
import re
import sys
import time

import common as C

ATTRS = ["home", "job", "employer", "school", "favorite", "owns", "relationship", "count", "amount",
         "progress", "status", "schedule", "routine", "goal", "other"]
SINGLE = {"home", "job", "employer", "school", "favorite", "count", "amount", "progress", "status", "schedule", "routine", "goal"}
ATTR_DESC = {
    "home": "where someone lives",
    "job": "someone's job title or role",
    "employer": "the company someone works for",
    "school": "the school or course someone attends",
    "favorite": "someone's favourite or preferred thing",
    "owns": "a thing someone owns or has",
    "relationship": "a person related to someone",
    "count": "how many of something",
    "amount": "a money amount, price, or measure",
    "progress": "how far along something is",
    "status": "the current state of something",
    "schedule": "when something happens",
    "routine": "a habit or regular activity",
    "goal": "a target someone aims for",
    "other": "anything else",
}

VARIANTS = {
    # v1: the brief's shape
    "v1": lambda: {"fact": ["subject::str", "attribute::[" + "|".join(ATTRS) + "]::str", "value::str"]},
    # v2: the thing the fact is about, then attribute and value, with descriptions
    "v2": lambda: {"fact": [
        "subject::str::the person or thing the fact is about, e.g. my road bike, Instagram followers, my sister",
        "attribute::[" + "|".join(ATTRS) + "]::str::what kind of property is stated",
        "value::str::the stated value, e.g. Lisbon, 1300, five, $400,000"]},
    # v3 (iterated on the trained-on questions): unambiguous attribute names
    # (the choice is picked as a span over the prefix, and "home" in the text
    # pulled counts to "home"), and the loose subject match (LOOSE below)
    "v3": lambda: {"fact": [
        "subject::str::the thing or person the fact is about, e.g. Instagram followers, road bike, sister",
        "attribute::[" + "|".join(ATTRS3) + "]::str::the kind of property",
        "value::str::the stated value, e.g. Lisbon, 1300, five, $400,000"]},
}
ATTRS3 = ["residence", "job title", "employer", "school", "favourite", "ownership", "relationship",
          "number of items", "money amount", "progress so far", "current status", "schedule", "habit", "goal", "other"]
SINGLE |= {"residence", "job title", "favourite", "number of items", "money amount", "progress so far",
           "current status", "schedule", "habit"}
LOOSE = {"v3", "v3n"}
STOP = set("a an the my our your his her their its of in on at to for with and or by from i me we it this that these those "
           "some any all new old current total so far now".split())


def _toks(s):
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP}


def loose_subject(a, b, ta, tb):
    """Subjects match when they share a content word (crudely stemmed), or
    a content word of one claim's subject occurs in the other claim."""
    sa, sb = _toks(a), _toks(b)
    if sa & sb:
        return True
    return bool(sa & _toks(tb)) or bool(sb & _toks(ta))


def norm_subject(s):
    s = (s or "").lower().strip(" .,!?'\"")
    s = re.sub(r"^(my|our|the|a|an|i|me|i'm|i've)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    if s in {"i", "me", "my", "myself", "we", "us", "", "i'm", "user"}:
        return "user"
    return s


def norm_val(v):
    return " ".join((v or "").lower().strip(" .,!?'\"").split())


def facts(m, texts, variant, batch_size=8):
    res = m.batch_extract_json(texts, VARIANTS[variant](), batch_size=batch_size, threshold=0.3, include_confidence=False)
    out = []
    for r in res:
        fs = []
        recs = r.get("fact") or []
        if isinstance(recs, dict):
            recs = [recs]
        for f in recs:
            def g(k):
                v = f.get(k)
                if isinstance(v, list):
                    v = v[0] if v else None
                if isinstance(v, dict):
                    v = v.get("text") or v.get("label") or v.get("value")
                return v
            fs.append({"subject": g("subject"), "attribute": g("attribute"), "value": g("value")})
        out.append(fs)
    return out


def key(f):
    return (norm_subject(f["subject"]), f["attribute"])


def subj_match(a, b):
    if a == b:
        return True
    ta, tb = set(a.split()), set(b.split())
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= 0.5


NUM = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|twice|once|first|second|third|fourth|fifth)\b", re.I)
MATCHER = {"v1": "strict", "v2": "strict", "v3": "loose", "v3s": "strict", "v3n": "numeric"}


def match_mode(fa, fb, mode, ta, tb):
    if mode == "strict":
        return match(fa, fb)
    if mode == "loose":
        return match(fa, fb, True, ta, tb)
    # numeric: loose subject, but both values carry a number (counts,
    # amounts, progress, times): the update shape that dominates KU
    for a in fa:
        for b in fb:
            if a["value"] and b["value"] and NUM.search(a["value"]) and NUM.search(b["value"]):
                h = match([a], [b], True, ta, tb)
                if h:
                    return h
    return None


def match(fa, fb, single_only=True, ta=None, tb=None):
    """Same subject and attribute, different values. With the claim texts
    (ta, tb), subjects match loosely (loose_subject)."""
    for a in fa:
        for b in fb:
            if not (a["attribute"] and a["attribute"] == b["attribute"]):
                continue
            if single_only and a["attribute"] not in SINGLE:
                continue
            if not (a["value"] and b["value"]) or norm_val(a["value"]) == norm_val(b["value"]):
                continue
            if ta is not None:
                ok = loose_subject(a["subject"], b["subject"], ta, tb)
            else:
                ok = subj_match(norm_subject(a["subject"]), norm_subject(b["subject"]))
            if ok:
                return (a, b)
    return None


def run(model, variant, split, device="cpu"):
    P = json.loads((C.GDIR / "ku_pairs.json").read_text())
    held, trained = C.lme_ku_split()
    pool = held if split == "held" else trained
    pairs = [p for p in P["pairs"] if p["qid"] in pool]
    texts = sorted({p["old"] for p in pairs} | {p["new"] for p in pairs}
                   | {c["text"] for q in pool if q in P["claims"] for c in P["claims"][q] if c["declarative"]})
    schema_v = variant.rstrip("sn") if variant.startswith("v3") else variant
    cache = C.GDIR / f"facts_{model}_{schema_v}_{split}.json"
    if cache.exists():
        F, dt = json.loads(cache.read_text()), float("nan")
    else:
        m = C.load_gliner(model, device)
        t0 = time.perf_counter()
        F = dict(zip(texts, facts(m, texts, schema_v)))
        dt = time.perf_counter() - t0
        cache.write_text(json.dumps(F))
    mode = MATCHER[variant]
    hit = same_key = no_fact = 0
    rows = []
    for p in pairs:
        fo, fn = F[p["old"]], F[p["new"]]
        h = match_mode(fo, fn, mode, p["old"], p["new"])
        hit += bool(h)
        same_key += any(a["attribute"] == b["attribute"] and (loose_subject(a["subject"], b["subject"], p["old"], p["new"]) if variant in LOOSE
                        else subj_match(norm_subject(a["subject"]), norm_subject(b["subject"]))) for a in fo for b in fn)
        no_fact += (not fo) or (not fn)
        rows.append({"qid": p["qid"], "old": p["old"], "new": p["new"], "old_facts": fo, "new_facts": fn, "hit": bool(h)})
    # false key matches: every pair of declarative claims of one question, minus the true pair
    fk = tot = 0
    examples = []
    for q in sorted(pool):
        cl = [c["text"] for c in P["claims"].get(q, []) if c["declarative"]]
        true = {(p["old"], p["new"]) for p in pairs if p["qid"] == q}
        cl = sorted(set(cl))
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                a, b = cl[i], cl[j]
                if (a, b) in true or (b, a) in true:
                    continue
                tot += 1
                h = match_mode(F[a], F[b], mode, a, b)
                if h:
                    fk += 1
                    if len(examples) < 40:
                        examples.append({"a": a, "b": b, "key": [h[0]["subject"], h[0]["attribute"]], "values": [h[0]["value"], h[1]["value"]]})
    n = len(pairs)
    summ = {"model": model, "variant": variant, "split": split, "pairs": n, "questions": len({p['qid'] for p in pairs}),
            "key_hit": hit / n, "same_key_any_value": same_key / n, "no_fact_on_a_side": no_fact / n,
            "unrelated_pairs": tot, "false_key_matches": fk, "false_key_rate": fk / tot if tot else 0,
            "false_key_per_question": fk / max(1, sum(1 for q in pool if not q.endswith("_abs"))),
            "claims": len(texts), "ms_per_claim": 1000 * dt / len(texts)}
    (C.GDIR / f"keys_{model}_{variant}_{split}.json").write_text(json.dumps({"summary": summ, "rows": rows, "false_examples": examples}, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    run(*sys.argv[2:])
