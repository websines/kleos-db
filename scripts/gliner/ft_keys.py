# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Stretch: update keys from a CLOSED attribute vocabulary, zero-shot.

The span keys of ku_keys.py failed on subject drift ("Instagram" vs "follower
count"). Here the key is a classification instead: each declarative claim gets
one attribute from a fixed list of ~50 personal-fact attributes (labels with
descriptions; no fine-tuning, no value extraction), and the subject is the
speaker (all held-out KU claims are first-person). A true update should give
the old and the new claim the same attribute; unrelated claims of the same
question should not.

    python ft_keys.py <model-or-ckpt> <tag> [cpu|mps]    # held-out 23 KU questions
"""
import json
import os
import sys
import time
from collections import Counter

os.environ.setdefault("GLINER_DIR", "/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr/.bench-data/gliner-ft")
import common as C  # noqa: E402

ATTRS = {
    "residence": "where someone lives or moved to",
    "employer": "the company or organisation someone works for",
    "job title": "someone's job, role or position",
    "income": "salary, pay, raise or income",
    "education": "school, degree, course or class someone takes",
    "relationship": "partner, dating, marriage or breakup",
    "children": "someone's kids or pregnancy",
    "pets": "pets someone has and how many",
    "home": "house, apartment, buying a home, renovation",
    "mortgage or loan": "mortgage, loan, pre-approval or debt amount",
    "rent or bills": "rent, utilities, bills",
    "vehicle": "car, bike, motorcycle someone owns or rides",
    "device": "phone, computer, camera, gadget",
    "collection": "items someone collects and how many",
    "hobby": "a hobby or craft and progress in it",
    "workout routine": "gym, exercise, training frequency",
    "running or race": "runs, races, times, distances",
    "diet": "what someone eats or avoids, diet plan",
    "health": "illness, condition, injury, medication",
    "body measure": "weight, height, fitness measure",
    "reading": "books someone reads or has read and how many",
    "watching": "shows, films, videos watched and how many",
    "music": "instrument, band, songs, concerts",
    "travel": "trips taken or places visited",
    "social media": "followers, posts, accounts, online audience",
    "savings or budget": "savings, budget, spending, investments",
    "purchase": "something someone bought and the price",
    "cooking": "recipes, cooking, baking, how many dishes",
    "garden": "plants, gardening",
    "volunteering": "volunteering, charity, donations",
    "language": "learning a language and level",
    "skill progress": "how far someone has got in learning a skill",
    "project progress": "progress on a project, pages, hours, percent done",
    "schedule": "when something regularly happens, days per week",
    "sleep": "sleep habits and hours",
    "commute": "how someone gets to work and how long",
    "favourite": "someone's favourite thing or preference",
    "drink habit": "coffee, tea, alcohol habits",
    "membership": "subscriptions, memberships, plans",
    "event": "an event someone attended or will attend",
    "family": "parents, siblings, relatives and what they do",
    "friends": "friends and what they do",
    "count of items": "how many of some thing someone has",
    "other": "anything else",
}


def run(path, tag, device="cpu"):
    from gliner2 import AutoExtractor
    P = json.loads((C.ROOT / ".bench-data" / "gliner" / "ku_pairs.json").read_text())
    held, _ = C.lme_ku_split()
    pairs = [p for p in P["pairs"] if p["qid"] in held]
    texts = sorted({p["old"] for p in pairs} | {p["new"] for p in pairs}
                   | {c["text"] for q in held if q in P["claims"] for c in P["claims"][q] if c["declarative"]})
    m = AutoExtractor.from_pretrained(C.MODELS.get(path, path), map_location=device)
    t0 = time.perf_counter()
    res = m.batch_classify_text(texts, {"attribute": {"labels": ATTRS}}, batch_size=8, include_confidence=True)
    dt = time.perf_counter() - t0
    A = {}
    for t, r in zip(texts, res):
        v = r.get("attribute")
        A[t] = v.get("label") if isinstance(v, dict) else v
    hit = 0
    rows = []
    for p in pairs:
        a, b = A[p["old"]], A[p["new"]]
        h = a == b and a != "other"
        hit += h
        rows.append({"qid": p["qid"], "old": p["old"], "new": p["new"], "old_attr": a, "new_attr": b, "hit": h})
    fk = tot = 0
    for q in sorted(held):
        cl = sorted({c["text"] for c in P["claims"].get(q, []) if c["declarative"]})
        true = {(p["old"], p["new"]) for p in pairs if p["qid"] == q}
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                if (cl[i], cl[j]) in true or (cl[j], cl[i]) in true:
                    continue
                tot += 1
                fk += A[cl[i]] == A[cl[j]] and A[cl[i]] != "other"
    nq = sum(1 for q in held if not q.endswith("_abs"))
    summ = {"model": path, "pairs": len(pairs), "same_attribute_hit": hit / len(pairs), "unrelated_pairs": tot,
            "false_same_attribute": fk, "false_per_question": fk / nq, "claims": len(texts),
            "ms_per_claim": 1000 * dt / len(texts), "attr_distribution": dict(Counter(A.values()).most_common(10))}
    (C.GDIR / f"keys_closed_{tag}.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "cpu")
