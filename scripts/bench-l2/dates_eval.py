#!/usr/bin/env python3
# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Accuracy of kleos's write-time date resolution (`memory_time`, stored in
`kleos_mem_event` by `remember`) on date-bearing claims of LoCoMo and
LongMemEval, against silver labels from the harness LLM (GLM-4.7-Flash,
thinking off, cached) and a hand-checked subset.

    python dates_eval.py sample  --n 300          # claims through remember (fts, no models) + the parser's events
    python dates_eval.py label                    # silver labels (LLM, cached by prompt hash)
    python dates_eval.py score [--gold hand.json] # precision / recall of expressions and of resolved intervals
    python dates_eval.py show --n 50              # the items for the hand check

Claims are what `remember` stores (its splitter), so this measures what the
store holds. Date-bearing = a broad temporal lexicon independent of the parser
(any month or weekday name, "ago", "yesterday", "last/next/this <unit>", a year
number, ...), so the sample also holds claims the parser misses.
"""
import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

LEX = re.compile(
    r"\b(yesterday|today|tonight|tomorrow|ago|weekend|morning|evening|afternoon|night|"
    r"(last|next|this|past|coming|previous|earlier this|later this)\s+(week|month|year|summer|winter|spring|fall|autumn|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|night|morning|time)|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec|"
    r"summer|winter|spring|autumn|fall|since|until|till|(19|20)\d\d|\d{1,2}/\d{1,2}(/\d{2,4})?|"
    r"days?|weeks?|months?|years?)\b",
    re.I,
)

PROMPT = """You annotate temporal expressions for a memory system.

The text below was said on {said} ({wd}). Find every expression in it that refers to a specific calendar date or period: absolute dates ("May 7, 2023", "2019"), relative ones ("yesterday", "last week", "two days ago", "next Friday", "last summer", "in March", "this weekend", "a month ago"), parts of a day ("last night"), and anchored spans ("since 2019", "until June", "for the past three weeks", "from May 3 to May 7").
Do NOT include: vague times ("recently", "a few days ago", "the other day", "one day", "a while ago", "soon"), habitual or generic ones ("every Monday", "on weekends", "in the summer" with no year or last/this/next), clock times alone ("at 5 pm"), plain durations with no anchor ("for two hours", "a long time"), ages, ordinals that are not dates ("the second time"), and words that only look like dates ("may" the verb, "march" the verb, "fall" the verb, names like "June" or "August" for people).

Resolve each against the date it was said, with these conventions: weeks run Monday to Sunday; "last week" is the previous Monday-to-Sunday week; "N weeks ago" is the Monday-to-Sunday week containing the day N*7 days earlier; "N months ago" is that calendar month, "N years ago" that calendar year; "last <weekday>" is the most recent such day strictly before the said date, "next <weekday>" the first one strictly after it, a bare weekday in a past-tense sentence the most recent one on or before the said date; a weekend is Saturday and Sunday; seasons are northern meteorological (spring Mar-May, summer Jun-Aug, fall Sep-Nov, winter Dec-Feb); "last summer" is the most recent summer that started before the said date and is not the current one; a month name without a year is the most recent such month on or before the said month in a past-tense sentence, the next one otherwise; "since X" runs from the start of X to the end of the said day; "until X" from the said day to the end of X; "for the past N weeks" from N weeks before the said day to the end of the said day; "last night" is the day before the said day. Intervals are [start, end) in ISO dates, end exclusive (yesterday said on 2023-05-08 is start 2023-05-07, end 2023-05-08).

Text: \"\"\"{text}\"\"\"

Answer with JSON only: {{"expressions": [{{"text": "<exact words from the text>", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "granularity": "day|week|month|season|year"}}]}}. Use an empty list when there are none."""


def bench():
    return C.bench_dir() / "dates-eval"


def iso(us):
    return datetime.fromtimestamp(us / 1e6, tz=timezone.utc).strftime("%Y-%m-%d")


def to_us(s):
    d = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1_000_000)


def sample(args):
    from kleos import CozoDbPy

    rnd = random.Random(C.SEED)
    pools = {"locomo": [], "lme": []}
    for conv in C.locomo_data():
        for u in C.locomo_units(conv):
            pools["locomo"].append((conv["sample_id"], u))
    seen = set()
    for x in sorted(C.lme_questions(), key=lambda x: x["question_id"]):
        for u in C.lme_units(x, "user"):
            if u["src"] in seen or len(u["text"]) > 3000:
                continue
            seen.add(u["src"])
            pools["lme"].append((x["question_id"], u))
    out = []
    for bname, pool in pools.items():
        rnd.shuffle(pool)
        want = args.n // 2
        db = CozoDbPy("mem", "", "{}")
        db.memory_init()
        got = 0
        for i, (unit, u) in enumerate(pool):
            if got >= want:
                break
            if not LEX.search(u["text"]):
                continue
            ns = f"n{i}"
            r = db.remember(u["text"], ns=ns, at=u["at"], split="auto")
            claims = [c for c in (r.get("claims") or [{"id": r["id"], "text": u["text"]}]) if c.get("id")]
            cands = [c for c in claims if LEX.search(c["text"])]
            if not cands:
                continue
            c = rnd.choice(cands)
            rows = db.run_script(
                "?[n, start, end, g, k, e] := *kleos_mem_event{ns: $ns, id: $id, n, start, end, granularity: g, kind: k, expr: e}",
                {"ns": ns, "id": c["id"]}, True)
            ev = [{"text": r_[5], "start": iso(r_[1]), "end": iso(r_[2]), "granularity": r_[3], "kind": r_[4]}
                  for r_ in sorted(rows["rows"])]
            tail = c["id"].rsplit(".", 1)[-1] if "." in c["id"] else "0"
            out.append({"key": f"{bname}:{u['src']}:{tail}", "bench": bname, "unit": unit,
                        "text": c["text"], "at": u["at"], "said": iso(u["at"]), "parser": ev})
            got += 1
        db.close()
    bench().mkdir(parents=True, exist_ok=True)
    (bench() / "sample.json").write_text(json.dumps(out, indent=1))
    n_ev = sum(1 for o in out if o["parser"])
    print(f"{len(out)} claims ({n_ev} with parser events) -> {bench() / 'sample.json'}")


def label(args):
    from concurrent.futures import ThreadPoolExecutor

    from llm import Client, quotas

    items = json.loads((bench() / "sample.json").read_text())
    cli = Client(max_tokens=600)
    try:
        print("quota before:", json.dumps(quotas()["planWeeklyLimit"]))
    except Exception as e:
        print("quota check failed:", e)

    def one(it):
        wd = datetime.strptime(it["said"], "%Y-%m-%d").strftime("%A")
        r = cli.chat([{"role": "user", "content": PROMPT.format(said=it["said"], wd=wd, text=it["text"])}])
        m = re.search(r"\{.*\}", r["content"], re.S)
        try:
            ex = json.loads(m.group(0))["expressions"] if m else []
        except Exception:
            ex = None
        return {**it, "silver": ex, "raw": r["content"]}

    with ThreadPoolExecutor(args.workers) as ex:
        out = list(ex.map(one, items))
    (bench() / "labelled.json").write_text(json.dumps(out, indent=1))
    bad = sum(1 for o in out if o["silver"] is None)
    print(f"labelled {len(out)} ({bad} unparseable); {cli.stats}")
    try:
        print("quota after:", json.dumps(quotas()["planWeeklyLimit"]))
    except Exception as e:
        print("quota check failed:", e)


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _match(p, g):
    """Expressions match when one's words contain the other's."""
    a, b = _norm(p["text"]), _norm(g["text"])
    return bool(a) and bool(b) and (a in b or b in a)


def _same_interval(p, g):
    """Same [start, end); a label whose end is the last day (inclusive, a
    frequent LLM slip despite the prompt) counts too."""
    if p["start"] != g.get("start") or not g.get("end"):
        return False
    if p["end"] == g["end"]:
        return True
    try:
        return to_us(p["end"]) - to_us(g["end"]) == 86_400_000_000
    except ValueError:
        return False


def score_items(items, gold_key):
    tp = fp = fn = 0
    iv_ok = iv_match = 0
    errors = []
    for it in items:
        gold = it.get(gold_key)
        if gold is None:
            continue
        gold = [g for g in gold if g.get("text")]
        pred = list(it["parser"])
        used = set()
        for p in pred:
            j = next((j for j, g in enumerate(gold) if j not in used and _match(p, g)), None)
            if j is None:
                fp += 1
                errors.append(("fp", it["key"], it["said"], p["text"], f"{p['start']}..{p['end']}", it["text"][:160]))
                continue
            used.add(j)
            tp += 1
            iv_match += 1
            g = gold[j]
            if _same_interval(p, g):
                iv_ok += 1
            else:
                errors.append(("iv", it["key"], it["said"], p["text"], f"{p['start']}..{p['end']} vs {g.get('start')}..{g.get('end')}",
                               it["text"][:160]))
        for j, g in enumerate(gold):
            if j not in used:
                fn += 1
                errors.append(("fn", it["key"], it["said"], g["text"], f"{g.get('start')}..{g.get('end')}", it["text"][:160]))
    n_gold = tp + fn
    res = {
        "claims": sum(1 for it in items if it.get(gold_key) is not None),
        "gold_expressions": n_gold,
        "parser_expressions": tp + fp,
        "expr_precision": tp / (tp + fp) if tp + fp else None,
        "expr_recall": tp / n_gold if n_gold else None,
        "interval_exact_of_matched": iv_ok / iv_match if iv_match else None,
        "interval_precision": iv_ok / (tp + fp) if tp + fp else None,
        "interval_recall": iv_ok / n_gold if n_gold else None,
    }
    return res, errors


def score(args):
    items = json.loads((bench() / "labelled.json").read_text())
    res, errors = score_items(items, "silver")
    out = {"silver": res}
    if args.gold:
        # hand labels: [{text, said, gold: [{text, start, end}]}], matched by claim text and date
        hand = json.loads(Path(args.gold).read_text())
        by = {(it["text"], it["said"]): it for it in items}
        sub = [{**by[(h["text"], h["said"])], "hand": h["gold"]} for h in hand if (h["text"], h["said"]) in by]
        print(f"hand labels: {len(sub)} of {len(hand)} matched")
        out["hand"], herr = score_items(sub, "hand")
        out["silver_on_hand_subset"], _ = score_items(sub, "silver")
        # how often the silver label agrees with the hand label (expressions and intervals)
        agree = sum(1 for it in sub if sorted((_norm(g["text"]), g.get("start"), g.get("end")) for g in (it["silver"] or []))
                    == sorted((_norm(g["text"]), g.get("start"), g.get("end")) for g in it["hand"]))
        out["silver_agrees_with_hand"] = f"{agree}/{len(sub)}"
        (bench() / "errors-hand.json").write_text(json.dumps(herr, indent=1))
    (bench() / "errors.json").write_text(json.dumps(errors, indent=1))
    (bench() / "score.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    by_kind = {}
    for e in errors:
        by_kind.setdefault(e[0], []).append(e)
    for k, v in by_kind.items():
        print(f"\n== {k}: {len(v)}")
        for e in v[: args.show]:
            print("  ", e)


def show(args):
    items = json.loads((bench() / "labelled.json").read_text())
    rnd = random.Random(C.SEED + 7)
    pick = rnd.sample(items, min(args.n, len(items)))
    for it in pick:
        print(json.dumps({"key": it["key"], "said": it["said"], "text": it["text"], "parser": [
            (p["text"], p["start"], p["end"]) for p in it["parser"]], "silver": [
            (g.get("text"), g.get("start"), g.get("end")) for g in (it["silver"] or [])]}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "label", "score", "show"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gold", default=None, help="hand labels: {key: [{text, start, end}]}")
    ap.add_argument("--show", type=int, default=25)
    args = ap.parse_args()
    {"sample": sample, "label": label, "score": score, "show": show}[args.cmd](args)


if __name__ == "__main__":
    main()
