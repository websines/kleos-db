# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""True old -> new update pairs on the LongMemEval knowledge-update questions.

For each question, the answer sessions are ordered by date; the user turns
flagged has_answer are split into claims (kleos's splitter); the LLM picks
the claim of the earlier session that states the old value and the claim of
the later session that states the new value. Consecutive answer sessions give
one pair each. Also lists every claim of the question's user turns (for the
false-key-match measurement). Writes $GLINER_DIR/ku_pairs.json.

    python ku_pairs.py
"""
import json

import common as C

PROMPT = """A user told an assistant something in an earlier conversation and later gave an update.
Question about the user: {q}
Latest answer: {a}

Sentences from the EARLIER conversation:
{old}

Sentences from the LATER conversation:
{new}

Pick the one sentence of the earlier conversation that states the earlier value relevant to the question, and the one sentence of the later conversation that states the updated value. Answer with JSON only: {{"old": <number>, "new": <number>}} (use -1 if no sentence states it)."""


def _resolve(v, sents):
    """The LLM sometimes answers with the value instead of the sentence
    number; map a value back to the one sentence that contains it."""
    if isinstance(v, int) and 0 <= v < len(sents):
        return v
    if v in (-1, "-1", None):
        return None
    sv = str(v).lower()
    digits = "".join(ch for ch in sv if ch.isdigit())
    hits = [i for i, c in enumerate(sents)
            if sv in c.lower() or (digits and digits in "".join(ch for ch in c if ch.isdigit()))]
    return hits[0] if len(hits) == 1 else None


def claims_of(sess):
    out = []
    for t in sess:
        if t["role"] == "user" and t.get("has_answer"):
            out += [c for c, _ in C.split_claims(t["content"])]
    return out


def main():
    raw = C.lme()
    held, trained = C.lme_ku_split(raw)
    data = [x for x in raw if x["question_id"] in held | trained and not x["question_id"].endswith("_abs")]
    jobs = []
    for x in data:
        sess = {s: (d, ss) for s, d, ss in zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"])}
        ans = sorted((s for s in x["answer_session_ids"] if s in sess), key=lambda s: C.micros(sess[s][0]))
        for a, b in zip(ans, ans[1:]):
            jobs.append((x, a, b, claims_of(sess[a][1]), claims_of(sess[b][1])))

    def pick(job):
        x, a, b, old, new = job
        if not old or not new:
            return None
        p = PROMPT.format(q=x["question"], a=x["answer"],
                          old="\n".join(f"{i}. {c}" for i, c in enumerate(old)),
                          new="\n".join(f"{i}. {c}" for i, c in enumerate(new)))
        j = C.parse_json(C.llm([{"role": "user", "content": p}]))
        o = _resolve(j.get("old", -1), old)
        n = _resolve(j.get("new", -1), new)
        if o is None or n is None:
            return None
        return {"qid": x["question_id"], "split": "held" if x["question_id"] in held else "trained",
                "question": x["question"], "answer": x["answer"], "old_sid": a, "new_sid": b,
                "old": old[o], "new": new[n]}

    jobs.sort(key=lambda j: j[0]["question_id"] not in held)  # held-out first
    nh = sum(1 for j in jobs if j[0]["question_id"] in held)
    pairs = [p for p in C.pmap(pick, jobs[:nh]) if p]
    allc = {}
    for x in data:
        cl = []
        for s, ss in zip(x["haystack_session_ids"], x["haystack_sessions"]):
            for t in ss:
                if t["role"] == "user":
                    cl += [{"sid": s, "text": c, "declarative": d} for c, d in C.split_claims(t["content"])]
        allc[x["question_id"]] = cl
    (C.GDIR / "ku_pairs.json").write_text(json.dumps({"pairs": pairs, "claims": allc}, indent=1))
    print("held-out written", flush=True)
    pairs += [p for p in C.pmap(pick, jobs[nh:]) if p]
    (C.GDIR / "ku_pairs.json").write_text(json.dumps({"pairs": pairs, "claims": allc}, indent=1))
    print(len(jobs), "answer-session pairs;", len(pairs), "picked;",
          sum(p["split"] == "held" for p in pairs), "held-out")


if __name__ == "__main__":
    main()
