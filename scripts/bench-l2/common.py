# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Shared pieces of the L2 memory benchmark harness (docs/benchmarks/memory.md):
paths, pinned dataset revisions, dataset loaders that turn LongMemEval / LoCoMo
into (namespace, memories, questions) units, and the retrieval metrics.

Nothing here imports kleos, so the report and QA steps run without the wheel.
"""
import json
import math
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # the kleos checkout (or worktree)


def bench_dir() -> Path:
    """`.bench-data/bench-l2` next to the checkout (KLEOS_BENCH_DIR overrides)."""
    if os.environ.get("KLEOS_BENCH_DIR"):
        return Path(os.environ["KLEOS_BENCH_DIR"])
    for p in ROOT.parents:
        if (p / ".bench-data").is_dir():
            return p / ".bench-data" / "bench-l2"
    return ROOT.parent / ".bench-data" / "bench-l2"


HF_HUB = Path(os.environ.get("HF_HOME", "/Volumes/SSD01/caches/huggingface")) / "hub"

# Pinned revisions. Changing one changes the numbers: bump the doc too.
LME_REPO = "xiaowu0162/longmemeval-cleaned"
LME_REV = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
LOCOMO_REPO = "snap-research/locomo"
LOCOMO_REV = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
LOCOMO_URL = f"https://raw.githubusercontent.com/{LOCOMO_REPO}/{LOCOMO_REV}/data/locomo10.json"
SEED = 20261023  # same seed as the decider-nano data build; the KU hold-out split reuses it


def lme_path(name="longmemeval_s_cleaned.json") -> Path:
    return HF_HUB / f"datasets--{LME_REPO.replace('/', '--')}" / "snapshots" / LME_REV / name


def locomo_path() -> Path:
    return bench_dir() / "data" / "locomo" / "locomo10.json"


def fetch_data():
    """Download both datasets at the pinned revisions (no-op when present)."""
    from huggingface_hub import hf_hub_download

    for name in ("longmemeval_s_cleaned.json", "longmemeval_oracle.json"):
        if not lme_path(name).exists():
            hf_hub_download(LME_REPO, name, repo_type="dataset", revision=LME_REV)
    p = locomo_path()
    if not p.exists():
        import urllib.request

        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LOCOMO_URL, p)


# ------------------------------------------------------------------ time
def lme_ts(s: str) -> int:
    """'2023/05/20 (Sat) 02:21' -> µs since epoch (UTC)."""
    d = datetime.strptime(re.sub(r"\s*\(.*?\)\s*", " ", s).strip(), "%Y/%m/%d %H:%M")
    return int(d.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)


def locomo_ts(s: str) -> int:
    """'1:56 pm on 8 May, 2023' -> µs since epoch (UTC)."""
    d = datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y")
    return int(d.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)


def fmt_ts(us: int) -> str:
    return datetime.fromtimestamp(us / 1e6, tz=timezone.utc).strftime("%Y/%m/%d (%a) %H:%M")


# ------------------------------------------------------------------ LongMemEval
def lme_ku_split():
    """The 25 knowledge-update question ids held out from decider-nano training
    (scripts/infer-bench/decide/train/build_data.py: sorted by id, shuffled with
    Random(SEED + 1), first 25). The other 53 KU questions had answer-turn
    pairs in the decider's training data."""
    lme = json.loads(lme_path("longmemeval_oracle.json").read_text())
    ku = sorted([x for x in lme if x["question_type"] == "knowledge-update"], key=lambda x: x["question_id"])
    ms = sorted([x for x in lme if x["question_type"] == "multi-session"], key=lambda x: x["question_id"])
    r = random.Random(SEED + 1)
    r.shuffle(ku)
    r.shuffle(ms)
    return {x["question_id"] for x in ku[:25]}, {x["question_id"] for x in ku[25:]}, {x["question_id"] for x in ms[40:]}


def lme_category(x) -> str:
    """Reporting split: the five abilities. Abstention (`_abs`) is its own row."""
    if x["question_id"].endswith("_abs"):
        return "abstention"
    t = x["question_type"]
    if t.startswith("single-session"):
        return "single-session"
    return t


def lme_units(x, unit="user"):
    """Turn one LongMemEval_S question into memories.

    unit='user' (default): one memory per user message, text as the user
    wrote it; assistant replies are not stored (an agent remembers what the
    user told it; 94% of LongMemEval evidence turns are user turns; and it is
    ~7x fewer tokens to embed than rounds). unit='round': one memory per user
    turn plus the assistant reply that follows ("User: …\\nAssistant: …"), the
    value granularity the LongMemEval paper found best; unit='turn': one
    memory per message; unit='session': one memory per session.
    Sessions are ingested in date order; `at` = session time + 1 s per unit.
    """
    sess = sorted(
        zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"]),
        key=lambda z: lme_ts(z[1]),
    )
    out = []
    for sid, date, turns in sess:
        t0 = lme_ts(date)
        groups = []
        if unit in ("user", "user+assistant"):
            j = -1
            for i, t in enumerate(turns):
                if t["role"] == "user":
                    j += 1
                    out.append({
                        "src": f"{sid}#{i}",
                        "session": sid,
                        "at": t0 + j * 1_000_000,
                        "text": t["content"].strip(),
                        "evidence": bool(t.get("has_answer")),
                    })
                elif unit == "user+assistant":
                    # the reply, half a second after the user turn it answers
                    out.append({
                        "src": f"{sid}#{i}",
                        "session": sid,
                        "at": t0 + max(j, 0) * 1_000_000 + 500_000,
                        "text": t["content"].strip(),
                        "evidence": bool(t.get("has_answer")),
                        "role": "assistant",
                    })
            continue
        if unit == "session":
            groups = [list(range(len(turns)))]
        elif unit == "turn":
            groups = [[i] for i in range(len(turns))]
        else:
            for i, t in enumerate(turns):
                if t["role"] == "user" or not groups:
                    groups.append([i])
                else:
                    groups[-1].append(i)
        for j, g in enumerate(groups):
            text = "\n".join(
                ("User: " if turns[i]["role"] == "user" else "Assistant: ") + turns[i]["content"].strip() for i in g
            )
            out.append({
                "src": f"{sid}#{j}",
                "session": sid,
                "at": t0 + j * 1_000_000,
                "text": text,
                "evidence": any(turns[i].get("has_answer") for i in g),
            })
    return out


def lme_questions(path=None):
    return json.loads(Path(path or lme_path()).read_text())


# ------------------------------------------------------------------ LoCoMo
LOCOMO_CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
DIA = re.compile(r"D\d+:\d+")


def locomo_evidence(ev):
    out = []
    for e in ev or []:
        out += DIA.findall(e.replace("D:", "D"))
    return out


def locomo_units(conv):
    """One memory per dialog turn: "Speaker: text" (+ the shared photo's caption)."""
    c = conv["conversation"]
    n = 1
    out = []
    while f"session_{n}" in c:
        t0 = locomo_ts(c[f"session_{n}_date_time"])
        for j, t in enumerate(c[f"session_{n}"]):
            text = f"{t['speaker']}: {t['text']}"
            if t.get("blip_caption"):
                text += f" [shares a photo: {t['blip_caption']}]"
            out.append({"src": t["dia_id"], "session": f"session_{n}", "at": t0 + j * 1_000_000, "text": text})
        n += 1
    return out


_CAPS = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*")
_NOT_NAMES = {"I", "I'm", "I've", "The", "This", "That", "It", "It's", "My", "We", "You", "Yes", "No", "Oh", "Hey",
              "Hi", "Thanks", "Thank", "Wow", "What", "How", "Do", "Did", "Have", "Are", "Is", "So", "But", "And",
              "Also", "Just", "That's", "Sounds", "Great", "Good", "Awesome", "Congrats", "Glad", "Sure", "Totally"}


def caps_entities(text):
    """A crude, model-free entity list for a LoCoMo turn: the speaker plus
    capitalised word runs that are not sentence openers or fillers."""
    spk, _, body = text.partition(": ")
    out = [spk]
    for m in _CAPS.finditer(body):
        w = m.group(0)
        if w.split()[0] in _NOT_NAMES or len(w) < 3:
            continue
        if w not in out:
            out.append(w)
    return out


def locomo_data():
    return json.loads(locomo_path().read_text())


# ------------------------------------------------------------------ metrics
def dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg(rels, n_rel, k):
    ideal = dcg([1] * min(n_rel, k))
    return dcg(rels[:k]) / ideal if ideal > 0 else 0.0


def stratified_sample(items, key, n, seed=SEED, must=()):
    """`n` items spread over `key` groups in proportion (at least 1 per group),
    seeded; ids in `must` are always included."""
    rng = random.Random(seed)
    groups = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    total = len(items)
    out = [it for it in items if it.get("question_id") in must]
    have = {id(it) for it in out}
    for g, xs in sorted(groups.items()):
        xs = sorted(xs, key=lambda it: json.dumps(it.get("question_id", it.get("question")), sort_keys=True))
        rng.shuffle(xs)
        want = max(1, round(n * len(xs) / total))
        cur = sum(1 for it in out if key(it) == g)
        for it in xs:
            if cur >= want:
                break
            if id(it) not in have:
                out.append(it)
                have.add(id(it))
                cur += 1
    return out


ENGINE_PATHS = ["kleos-core", "kleos-infer", "kleos-lib-python", "Cargo.lock"]


def git_rev() -> str:
    """The result tag: the last commit that changed the engine (so commits to the
    harness or docs don't relabel results), plus -dirty for uncommitted engine changes."""
    try:
        import subprocess

        r = subprocess.run(["git", "log", "-1", "--format=%h", "--abbrev=7", "--", *ENGINE_PATHS],
                           cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *ENGINE_PATHS],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return r.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"
