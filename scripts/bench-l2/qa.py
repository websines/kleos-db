#!/usr/bin/env python3
# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""QA accuracy (secondary metric): a reader answers from what kleos `recall`
returned (cached by retrieval.py), and an LLM judge grades the answer with the
benchmark's official judge prompt. Same reader and judge for every condition:

  kleos:<variant>  the top-10 memories recall returned, in time order, with dates
  stmt:<variant>   the same list, each claim replaced by its whole statement (`context`)
  ctx:<variant>    the same recall as recall's context block (`format: 'context'`,
                   stored by retrieval.py --context-budget): grouped, dated, with
                   superseded values and the statements the claims came from
  none             no memory at all (the question only)
  oracle           LongMemEval: the answer sessions in full (the paper's oracle);
                   LoCoMo: the evidence turns
  full             the whole history in the prompt (LongMemEval_S ~115k tokens)

    python qa.py --bench lme --conditions none,oracle,kleos:fts,kleos:hybrid --ids-variant hybrid
    python qa.py --bench locomo --conditions none,kleos:fts,kleos:hybrid --n 300

Reader/judge default: Synthetic `syn:small:text` (GLM-4.7-Flash), thinking off.
These numbers are NOT comparable with GPT-4o-judged leaderboards.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from llm import Client, quotas  # noqa: E402

# ---------------------------------------------------------------- LongMemEval prompts
# Reader: LongMemEval's run_generation.py history-chat prompt (no CoN).
LME_READER = ("I will give you several history chats between you and a user. Please answer the question based on "
              "the relevant chat history.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:")
LME_READER_NONE = ("Please answer the question about the user you are chatting with.\n\nCurrent Date: {}\n"
                   "Question: {}\nAnswer:")
# Judge: LongMemEval's evaluate_qa.py get_anscheck_prompt, per task.
_J = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
      "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
      "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response "
      "only contains a subset of the information required by the answer, answer no. ")
_Q = "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
LME_JUDGE = {
    "default": _J + _Q,
    "temporal-reasoning": _J + (
        "In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number "
        "of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer "
        "is 18), the model's response is still correct. ") + _Q,
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
        "response contains the correct answer. Otherwise, answer no. If the response contains some previous "
        "information along with an updated answer, the response should be considered as correct as long as the "
        "updated answer is the required answer.") + _Q,
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a response from a model. "
        "Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not "
        "need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes "
        "the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
        "Is the model response correct? Answer yes or no only."),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes "
        "if the model correctly identifies the question as unanswerable. The model could say that the information "
        "is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\n"
        "Explanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
        "Answer yes or no only."),
}

# ---------------------------------------------------------------- LoCoMo prompts (Mem0 evaluation code)
LOCOMO_READER = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from a conversation between {a} and {b}. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years.
7. Focus only on the content of the memories. Do not confuse character names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.

Memories:
{m}

Question: {q}
Answer:"""
LOCOMO_READER_NONE = """You are answering a question about a conversation between {a} and {b}. You have no access to the conversation. Give your best short answer (less than 5-6 words).

Question: {q}
Answer:"""
LOCOMO_JUDGE = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {q}
Gold answer: {g}
Generated answer: {r}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


def results_dir(tag, bench, unit, variant):
    return C.bench_dir() / "results" / tag / f"{bench}-{unit}" / variant.removesuffix("+hide_pending")


def is_mem(cond):
    return cond.split(":", 1)[0] in ("kleos", "ctx", "stmt")


def stmt_hits(hits):
    """stmt:<variant>: the list format, each claim replaced by the whole
    statement it came from (once per statement), as before claims."""
    out, seen = [], set()
    for h in hits:
        t = h.get("context") or h["text"]
        if t in seen:
            continue
        seen.add(t)
        out.append({**h, "text": t})
    return out


def hits_key(cond):
    """kleos:sup+hide_pending reads the hide_pending recall of the sup run."""
    return "hits10_hide_pending" if cond.endswith("+hide_pending") else "hits10"


def locomo_date(us):
    from datetime import datetime, timezone

    d = datetime.fromtimestamp(us / 1e6, tz=timezone.utc)
    return d.strftime("%-d %B %Y, %-I:%M %p")


# ---------------------------------------------------------------- LongMemEval
def lme_context(cond, x, rec):  # noqa: C901
    if cond.startswith("ctx:"):
        return rec["ctx10"]["text"]
    if cond.startswith("kleos:") or cond.startswith("stmt:"):
        hits = sorted(rec[hits_key(cond)], key=lambda h: h.get("valid_from") or 0)
        if cond.startswith("stmt:"):
            hits = stmt_hits(hits)
        return "\n\n".join(f"[{C.fmt_ts(h['valid_from'])}] {h.get('role') or 'user'}: {h['text']}" for h in hits)
    sess = sorted(zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"]),
                  key=lambda z: C.lme_ts(z[1]))
    if cond == "oracle":
        sess = [s for s in sess if s[0] in set(x["answer_session_ids"])]
    parts = []
    for sid, d, turns in sess:
        body = "\n".join(f"{t['role']}: {t['content'].strip()}" for t in turns)
        parts.append(f"### Session Date: {d}\nSession Content:\n{body}")
    return "\n\n".join(parts)


def run_lme(args, cli):
    path = C.lme_path("longmemeval_oracle.json") if args.bench == "lme-oracle" else C.lme_path()
    data = {x["question_id"]: x for x in C.lme_questions(path)}
    rec_dirs = {c: results_dir(args.tag, args.bench, args.unit, c.split(":", 1)[1])
                for c in args.conds if is_mem(c)}
    ids = sorted(data)
    if args.ids_variant:
        have = {p.stem for p in results_dir(args.tag, args.bench, args.unit, args.ids_variant).glob("*.json")} - {"meta"}
        ids = [i for i in ids if i in have]
    for c, d in rec_dirs.items():
        have = {p.stem for p in d.glob("*.json")}
        ids = [i for i in ids if i in have]
    if args.types:
        ids = [i for i in ids if data[i]["question_type"] in args.types.split(",")]
    if args.n and args.n < len(ids):
        ids = [x["question_id"] for x in C.stratified_sample([data[i] for i in ids], C.lme_category, args.n, seed=args.seed)]
    jobs = []
    for cond in args.conds:
        out = C.bench_dir() / "results" / args.tag / "qa" / f"{args.bench}-{args.unit}" / args.set / cond.replace(":", "_")
        out.mkdir(parents=True, exist_ok=True)
        for i in ids:
            jobs.append((cond, i, out / f"{i}.json"))

    def one(job):
        cond, qid, f = job
        if f.exists():
            return json.loads(f.read_text())
        x = data[qid]
        rec = json.loads((rec_dirs[cond] / f"{qid}.json").read_text()) if cond in rec_dirs else None
        if cond == "none":
            prompt = LME_READER_NONE.format(x["question_date"], x["question"])
        else:
            prompt = LME_READER.format(lme_context(cond, x, rec), x["question_date"], x["question"])
        r = cli.chat([{"role": "user", "content": prompt}])
        cat = C.lme_category(x)
        jt = "abstention" if cat == "abstention" else x["question_type"]
        jp = LME_JUDGE.get(jt, LME_JUDGE["default"]).format(x["question"], x["answer"], r["content"])
        j = cli.chat([{"role": "user", "content": jp}])
        out = {"qid": qid, "condition": cond, "category": cat, "question_type": x["question_type"],
               "ku_split": rec.get("ku_split") if rec else None, "answer": r["content"], "gold": x["answer"],
               "judge": j["content"], "correct": "yes" in j["content"].lower()[:20],
               "reader_model": r["model"], "judge_model": j["model"], "reader_usage": r["usage"]}
        f.write_text(json.dumps(out))
        return out

    return jobs, one


# ---------------------------------------------------------------- LoCoMo
def run_locomo(args, cli):
    convs = {c["sample_id"]: c for c in C.locomo_data()}
    rec_dirs = {c: results_dir(args.tag, "locomo", "turn", c.split(":", 1)[1])
                for c in args.conds if is_mem(c)}
    recs = {}
    for c, d in rec_dirs.items():
        recs[c] = {}
        for p in d.glob("*.json"):
            if p.stem == "meta":
                continue
            r = json.loads(p.read_text())
            for q in r["questions"]:
                recs[c][q["qid"]] = q
    items = []
    for sid, conv in convs.items():
        for qi, q in enumerate(conv["qa"]):
            if q["category"] == 5:  # adversarial: no gold answers; excluded as in Mem0 / Zep
                continue
            items.append({"question_id": f"{sid}/{qi}", "sid": sid, "q": q})
    for c in recs:
        items = [it for it in items if it["question_id"] in recs[c]]
    if args.n and args.n < len(items):
        items = C.stratified_sample(items, lambda it: it["q"]["category"], args.n, seed=args.seed)
    turns = {sid: {u["src"]: u for u in C.locomo_units(conv)} for sid, conv in convs.items()}
    jobs = []
    for cond in args.conds:
        out = C.bench_dir() / "results" / args.tag / "qa" / "locomo-turn" / args.set / cond.replace(":", "_")
        out.mkdir(parents=True, exist_ok=True)
        for it in items:
            jobs.append((cond, it, out / (it["question_id"].replace("/", "_") + ".json")))

    def one(job):
        cond, it, f = job
        if f.exists():
            return json.loads(f.read_text())
        conv, q = convs[it["sid"]], it["q"]
        a, b = conv["conversation"]["speaker_a"], conv["conversation"]["speaker_b"]
        if cond == "none":
            prompt = LOCOMO_READER_NONE.format(a=a, b=b, q=q["question"])
        else:
            if cond.startswith("ctx:"):
                mem = None
            elif cond.startswith("kleos:") or cond.startswith("stmt:"):
                hits = sorted(recs[cond][it["question_id"]][hits_key(cond)], key=lambda h: h.get("valid_from") or 0)
                if cond.startswith("stmt:"):
                    hits = stmt_hits(hits)
                mem = [(h["valid_from"], h["text"]) for h in hits]
            elif cond == "oracle":
                ev = C.locomo_evidence(q.get("evidence"))
                mem = sorted((turns[it["sid"]][e]["at"], turns[it["sid"]][e]["text"]) for e in ev if e in turns[it["sid"]])
            else:  # full
                mem = [(u["at"], u["text"]) for u in turns[it["sid"]].values()]
            if mem is None:
                m = recs[cond][it["question_id"]]["ctx10"]["text"]
            else:
                m = "\n".join(f"[{locomo_date(t)}] {txt}" for t, txt in mem)
            prompt = LOCOMO_READER.format(a=a, b=b, m=m, q=q["question"])
        r = cli.chat([{"role": "user", "content": prompt}])
        gold = str(q.get("answer"))
        j = cli.chat([{"role": "user", "content": LOCOMO_JUDGE.format(q=q["question"], g=gold, r=r["content"])}])
        verdict = j["content"].upper()
        correct = verdict.rfind("CORRECT") > verdict.rfind("WRONG")  # the last label wins
        out = {"qid": it["question_id"], "condition": cond, "category": q["category"], "answer": r["content"],
               "gold": gold, "judge": j["content"], "correct": correct, "reader_model": r["model"],
               "judge_model": j["model"], "reader_usage": r["usage"]}
        f.write_text(json.dumps(out))
        return out

    return jobs, one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["lme", "lme-oracle", "locomo"], required=True)
    ap.add_argument("--conditions", required=True, help="comma list: none,oracle,full,kleos:<variant>")
    ap.add_argument("--unit", default="user")
    ap.add_argument("--ids-variant", default=None, help="LongMemEval: only questions this retrieval variant ran")
    ap.add_argument("--n", type=int, default=0, help="stratified sample size")
    ap.add_argument("--types", default=None, help="LongMemEval: only these question types (comma list)")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--thinking", action="store_true", help="let the reasoning model think (default off)")
    ap.add_argument("--tag", default=os.environ.get("KLEOS_BENCH_TAG") or C.git_rev())
    args = ap.parse_args()
    args.conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    # results of one question set live together, so every condition is compared on the same questions
    args.set = (args.ids_variant or "all") + (f"-n{args.n}" if args.n else "") + (f"-{args.types}" if args.types else "")
    cli = Client(thinking=args.thinking)
    if cli.endpoint == "synthetic":
        try:
            print("quota before:", json.dumps(quotas()), flush=True)
        except Exception as e:
            print("quota check failed:", e)
    jobs, one = (run_locomo if args.bench == "locomo" else run_lme)(args, cli)
    print(f"{len(jobs)} (condition, question) jobs", flush=True)
    t0 = time.time()
    done = 0
    fails = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for fu in futs:
            try:
                fu.result()
            except Exception as e:
                fails += 1
                print("job failed:", str(e)[:200], flush=True)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} in {time.time() - t0:.0f}s, {cli.stats}", flush=True)
    print(f"done {done} jobs ({fails} failed) in {time.time() - t0:.0f}s; {cli.stats}", flush=True)
    if cli.endpoint == "synthetic":
        try:
            print("quota after:", json.dumps(quotas()), flush=True)
        except Exception as e:
            print("quota check failed:", e)


if __name__ == "__main__":
    main()
