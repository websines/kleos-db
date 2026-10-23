# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Data for fine-tuning GLiNER2.5 on kleos's own tasks (docs/research/gliner.md,
"Fine-tuned").

Tasks, per claim (kleos's claim splitter, no speaker label):
- entities: person, location, organization, and `relative`: a phrase naming one
  specific person by their relation or role to the speaker ("my sister",
  "Mom", "my boss"); with `name_of` pairs when the claim names that person;
- about_speaker: does the claim say something about the speaker themself;
- kind: fact / preference / plan / event / question / request / small_talk
  (store = the first four).

Splits:
- train/val: LongMemEval_S user turns from sessions in NO evaluation haystack
  (every question any bench-l2 model run used, i.e. the 119/162-question QA
  sets, plus all 78 knowledge-update questions), and GLM-written chat turns.
  LoCoMo is never used for training (CC BY-NC, evaluation only).
- gold: 75 claims from evaluation-side LongMemEval sessions + 75 from LoCoMo,
  silver-labelled here and then corrected by hand (gold.json).

    python ft_data.py split        # eval_sessions.json
    python ft_data.py pool         # pool.json (train/val claims) + gold_draft.json
    python ft_data.py synth        # synth.json (GLM-written turns -> claims)
    python ft_data.py silver       # silver labels for pool, synth and the gold draft
"""
import glob
import json
import os
import random
import re
import sys

os.environ.setdefault("GLINER_DIR", "/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr/.bench-data/gliner-ft")
import common as C  # noqa: E402

BENCH = C.ROOT / ".bench-data"
KINDS = ["fact", "preference", "plan", "event", "question", "request", "small_talk"]
STORE = {"fact", "preference", "plan", "event"}
ENTS = ["person", "location", "organization", "relative"]

_CAPS = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")
_STOP = {"I", "I'm", "I've", "I'd", "I'll", "The", "This", "That", "It", "My", "We", "You", "Yes", "No", "Oh", "Hey",
         "Hi", "Thanks", "Thank", "What", "How", "Can", "Could", "Would", "Also", "Just", "But", "And", "So", "Do",
         "Is", "Are", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January",
         "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
         "December", "English", "Python", "Google", "Excel"}


def name_rich(text):
    return any(m.group(0).split()[0] not in _STOP and len(m.group(0)) > 2 for m in _CAPS.finditer(text))


def eval_questions():
    """Every LongMemEval question a bench-l2 model run (anything but fts / the
    no-retrieval baselines over all 500) was scored on, plus all KU."""
    ids = set()
    for d in glob.glob(str(BENCH / "*/results/*/lme-*/*")) + glob.glob(str(BENCH / "*/results/*/qa/lme-*/*")) + glob.glob(str(BENCH / "*/results/lme-*/*")):
        v = os.path.basename(d)
        if not os.path.isdir(d) or v.startswith("fts") or v.startswith("all"):
            continue
        ids |= {os.path.basename(p)[:-5] for p in glob.glob(d + "/*.json")} - {"meta"}
    return ids


def split():
    S = C.lme("longmemeval_s_cleaned.json")
    ev = eval_questions() | {x["question_id"] for x in S if x["question_type"] == "knowledge-update"}
    evs = sorted({sid for x in S if x["question_id"] in ev for sid in x["haystack_session_ids"]})
    C.GDIR.mkdir(parents=True, exist_ok=True)
    (C.GDIR / "eval_sessions.json").write_text(json.dumps({"questions": sorted(ev), "sessions": evs}))
    print(len(ev), "evaluation questions;", len(evs), "evaluation sessions")


def lme_turns():
    S = C.lme("longmemeval_s_cleaned.json")
    turns = {}
    for x in S:
        for sid, sess in zip(x["haystack_session_ids"], x["haystack_sessions"]):
            if sid not in turns:
                turns[sid] = [t["content"] for t in sess if t["role"] == "user" and 20 <= len(t["content"]) <= 1500]
    return turns


def pick_claim(rnd, turn, prefer_names=False):
    claims = [c for c, _ in C.split_claims(turn) if len(c.split()) >= 3]
    if not claims:
        return None
    if prefer_names:
        named = [c for c in claims if name_rich(c)]
        if named:
            return rnd.choice(named)
    return rnd.choice(claims)


def pool(n_rand=1800, n_named=900, g_rand=60, g_named=15, g_loc=75):
    rnd = random.Random(C.SEED + 11)
    evs = set(json.loads((C.GDIR / "eval_sessions.json").read_text())["sessions"])
    turns = lme_turns()
    train_s = sorted(s for s in turns if s not in evs and turns[s])
    eval_s = sorted(s for s in turns if s in evs and turns[s])
    rnd.shuffle(train_s)
    rnd.shuffle(eval_s)

    def take(sids, n_r, n_n, tag):
        out, seen = [], set()
        named_sids = [s for s in sids if any(name_rich(t) for t in turns[s])]
        for want, src, pref in ((n_n, named_sids, True), (n_r, sids, False)):
            got = 0
            for s in src:
                if got >= want:
                    break
                if s in seen:
                    continue
                cand = [t for t in turns[s] if name_rich(t)] if pref else turns[s]
                c = pick_claim(rnd, rnd.choice(cand), pref) if cand else None
                if not c:
                    continue
                seen.add(s)
                out.append({"id": f"{tag}:{s}:{len(out)}", "src": "lme", "named_pick": pref, "text": c})
                got += 1
        return out

    tr = take(train_s, n_rand, n_named, "lme")
    gold_lme = take(eval_s, g_rand, g_named, "gold-lme")
    loc = json.loads(C.LOCOMO.read_text())
    lt = []
    for ci, conv in enumerate(loc):
        for k, v in conv["conversation"].items():
            if k.startswith("session_") and isinstance(v, list):
                lt += [(ci, t["dia_id"], t["speaker"], t["text"]) for t in v if 20 <= len(t.get("text", "")) <= 1500]
    rnd.shuffle(lt)
    gold_loc = []
    for ci, did, spk, turn in lt:
        if len(gold_loc) >= g_loc:
            break
        c = pick_claim(rnd, turn)
        if c:
            gold_loc.append({"id": f"gold-locomo:{ci}:{did}", "src": "locomo", "speaker": spk, "text": c})
    (C.GDIR / "pool.json").write_text(json.dumps(tr, indent=1))
    (C.GDIR / "gold_draft.json").write_text(json.dumps(gold_lme + gold_loc, indent=1))
    print(len(tr), "pool claims (", sum(x["named_pick"] for x in tr), "name-rich picks );", len(gold_lme), "+", len(gold_loc), "gold claims")


KIN = re.compile(r"\b(my|our|his|her) (sister|brother|mom|mother|dad|father|wife|husband|partner|boss|friend|best friend|son|daughter|"
                 r"cousin|aunt|uncle|grandma|grandmother|grandpa|grandfather|niece|nephew|coworker|colleague|roommate|"
                 r"neighbou?r|boyfriend|girlfriend|fianc[eé]e?|manager|therapist|teacher|coach)\b", re.I)


def gold_extra(n_lme=25, n_loc=25):
    """Enriched gold: claims with a kinship or role phrase (and often a name),
    which the random 150 barely contain (3 relatives, 0 pairs)."""
    rnd = random.Random(C.SEED + 14)
    evs = set(json.loads((C.GDIR / "eval_sessions.json").read_text())["sessions"])
    have = {x["text"] for x in json.loads((C.GDIR / "gold_draft.json").read_text())}
    turns = lme_turns()
    cand = [(s, t) for s in sorted(evs) for t in turns.get(s, []) if KIN.search(t)]
    rnd.shuffle(cand)
    out = []
    for s, t in cand:
        if len(out) >= n_lme:
            break
        cl = [c for c, _ in C.split_claims(t) if KIN.search(c) and c not in have]
        if cl:
            out.append({"id": f"gold-lme-kin:{s}:{len(out)}", "src": "lme", "text": rnd.choice(cl)})
    loc = json.loads(C.LOCOMO.read_text())
    lt = []
    for ci, conv in enumerate(loc):
        for k, v in conv["conversation"].items():
            if k.startswith("session_") and isinstance(v, list):
                lt += [(ci, t["dia_id"], t["speaker"], t["text"]) for t in v if KIN.search(t.get("text", ""))]
    rnd.shuffle(lt)
    nl = 0
    for ci, did, spk, t in lt:
        if nl >= n_loc:
            break
        cl = [c for c, _ in C.split_claims(t) if KIN.search(c) and c not in have]
        if cl:
            out.append({"id": f"gold-locomo-kin:{ci}:{did}", "src": "locomo", "speaker": spk, "text": rnd.choice(cl)})
            nl += 1
    (C.GDIR / "gold_extra.json").write_text(json.dumps(out, indent=1))
    print(len(out), "enriched gold claims")


SYNTH = """Write {n} short chat messages that people might send to a personal AI assistant or to a friend while talking about their own lives. Mix them up:
- some mention family, partners, friends, coworkers, pets or neighbours, sometimes by name ("my sister Priya", "Tom from work", "our dog Biscuit", "Mom"), sometimes only by relation;
- some mention real places (cities, countries, neighbourhoods, venues, parks), companies, schools, apps or teams;
- some state facts, preferences, plans, past events, updates to earlier facts ("we moved again", "now I have three cats"); others ask questions, make requests, or are just small talk ("haha thanks!", "that's awesome").
Vary length (one to four sentences), tone, names (many cultures) and topics: {topics}.
Answer with JSON only: {{"messages": ["...", ...]}}"""
TOPICS = ["work and careers", "moving house", "pets", "family events", "travel", "health and fitness", "hobbies and crafts",
          "school and kids", "money and shopping", "food and cooking", "music and concerts", "sports", "volunteering",
          "relationships and weddings", "cars and commuting", "gardening", "books and films", "holidays", "friends",
          "neighbours and community"]


def synth(calls=80, per=10):
    rnd = random.Random(C.SEED + 12)
    jobs = [(i, ", ".join(rnd.sample(TOPICS, 3))) for i in range(calls)]

    def one(job):
        i, topics = job
        out = C.llm([{"role": "user", "content": SYNTH.format(n=per, topics=topics) + f"\n(batch {i})"}], max_tokens=6000, temperature=0.9, **FAST)
        try:
            return [m for m in C.parse_json(out).get("messages", []) if isinstance(m, str)]
        except Exception:  # noqa: BLE001
            return []
    msgs = [m for ms in C.pmap(one, jobs, workers=3) for m in ms]
    out = []
    for m in msgs:
        for c, _ in C.split_claims(m):
            if len(c.split()) >= 2:
                out.append({"id": f"synth:{len(out)}", "src": "synth", "text": c, "turn": m})
    (C.GDIR / "synth.json").write_text(json.dumps(out, indent=1))
    print(len(msgs), "messages ->", len(out), "claims")


SYNTH_CHAT = """Write {n} chat messages from a casual conversation between two friends catching up by text. Most should be the light parts of such chats: reactions ("wow, that's amazing!"), encouragement, compliments, thanks, greetings and goodbyes, sympathy, enthusiasm, general remarks ("nature really is calming"), and questions about the other person's life ("how did it go?", "when are you leaving?"). Some should share personal news, plans, family or friends by name, or places. Vary length (one to three sentences), names and tone. Topics: {topics}.
Answer with JSON only: {{"messages": ["...", ...]}}"""


def synth_chat(calls=40, per=10):
    """Friend-to-friend chat turns: the reactions, encouragement and
    questions about the listener that the LongMemEval user turns lack."""
    rnd = random.Random(C.SEED + 15)
    jobs = [(i, ", ".join(rnd.sample(TOPICS, 3))) for i in range(calls)]

    def one(job):
        i, topics = job
        out = C.llm([{"role": "user", "content": SYNTH_CHAT.format(n=per, topics=topics) + f"\n(batch {i})"}], max_tokens=6000, temperature=0.9, **FAST)
        try:
            return [m for m in C.parse_json(out).get("messages", []) if isinstance(m, str)]
        except Exception:  # noqa: BLE001
            return []
    msgs = [m for ms in C.pmap(one, jobs, workers=3) for m in ms]
    out = []
    for m in msgs:
        for c, _ in C.split_claims(m):
            if len(c.split()) >= 2:
                out.append({"id": f"synth-chat:{len(out)}", "src": "synth", "text": c, "turn": m})
    (C.GDIR / "synth_chat.json").write_text(json.dumps(out, indent=1))
    print(len(msgs), "chat messages ->", len(out), "claims")


LABEL = """You label sentences from chats for training a small extraction model. Each sentence is one claim from a message the SPEAKER wrote (to an assistant or a friend). Label each sentence independently.

1. "entities": every mention of these types, copying the EXACT substring (same case) from that sentence:
- person: the name of a specific person or pet ("Anna", "Dr. Lee", "Max", "Taylor Swift"). Not pronouns ("I", "she"), not role words ("my sister", "Mom", "the doctor"), not names used as something else (a brand, a book title).
- location: the name of a specific place: city, country, region, street, neighbourhood, venue, park, landmark ("Lisbon", "Central Park", "the Bay Area").
- organization: the name of a specific company, employer, school, university, team, band, club, website, app or online service ("Wells Fargo", "Coursera", "Instagram", "the Lakers").
- relative: a phrase that refers to ONE specific person known to the speaker by their relation or role, e.g. "my sister", "Mom", "my boss", "my best friend", "his wife", "our neighbour", "my therapist". Include "my"/"our"/"his" when present. Not groups ("my friends", "my kids", "people"), not generic roles ("a doctor").
Nothing else: no dates, numbers, products, food, generic nouns ("farmers", "a bike", "my job").

2. "name_of": for each relative the same sentence also names, the pair [relative, person], e.g. "my sister Anna" -> ["my sister", "Anna"]; "Tom, my coworker" -> ["my coworker", "Tom"]. Empty if none.

3. "about_speaker": true if the sentence states or implies something about the speaker themself (their life, situation, possessions, feelings, opinions, plans or experiences), false otherwise (only about others or the world, a pure question or request with no personal information, or small talk).

4. "kind", the single best type of the sentence from the speaker's point of view:
- fact: something stable or current about the speaker, someone or the world (job, home, possessions, relationships, counts, status, habits).
- preference: a like, dislike, favourite, taste or opinion.
- plan: an intention, plan, goal or future arrangement.
- event: something that happened at a particular time (a past experience or occurrence).
- question: asks for information.
- request: asks the listener to do or produce something (help, recommend, suggest, write, list), even phrased as a question.
- small_talk: greetings, thanks, reactions, compliments or encouragement with no new information.

Sentences:
{text}

Answer with JSON only, one item per sentence in order:
{{"items": [{{"i": 0, "entities": [{{"text": "...", "label": "..."}}], "name_of": [["...", "..."]], "about_speaker": true, "kind": "..."}}, ...]}}"""
BATCH = 10


FAST = {"reasoning_effort": "none"}  # train/val labels: no reasoning (4x faster on a shared, concurrency-limited key)


def _label_batch(items, fast=False):
    text = "\n".join(f"{i}. {x['text']}" for i, x in enumerate(items))
    out = C.llm([{"role": "user", "content": LABEL.format(text=text)}], max_tokens=12000, **(FAST if fast else {}))
    try:
        js = {j.get("i"): j for j in C.parse_json(out).get("items", [])}
    except Exception:  # noqa: BLE001
        js = {}
    res = []
    for i, x in enumerate(items):
        j = js.get(i)
        if j is None or j.get("kind") not in KINDS:
            res.append({**x, "error": "missing"})
            continue
        ents = [e for e in j.get("entities", []) if e.get("label") in ENTS and e.get("text") and e["text"] in x["text"]]
        rel = {e["text"] for e in ents if e["label"] == "relative"}
        per = {e["text"] for e in ents if e["label"] == "person"}
        pairs = [p for p in j.get("name_of", []) if isinstance(p, list) and len(p) == 2 and p[0] in rel and p[1] in per]
        res.append({**x, "entities": ents, "name_of": pairs, "about_speaker": bool(j.get("about_speaker")), "kind": j["kind"]})
    return res


def silver(names=("gold_draft", "synth", "pool")):
    for name in names:
        items = json.loads((C.GDIR / f"{name}.json").read_text())
        fast = not name.startswith("gold_draft") and name != "gold_extra"  # the gold draft is labelled with reasoning, then hand-corrected
        bs = 20 if fast else BATCH
        res = [r for b in C.pmap(lambda b: _label_batch(b, fast), [items[i:i + bs] for i in range(0, len(items), bs)], workers=3) for r in b]
        (C.GDIR / f"{name}_silver.json").write_text(json.dumps(res, indent=1))
        print(name, len(res), "labelled;", sum("error" in r for r in res), "missing")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "split":
        split()
    elif cmd == "pool":
        pool()
    elif cmd == "synth":
        synth(int(sys.argv[2]) if len(sys.argv) > 2 else 80)
    elif cmd == "synth_chat":
        synth_chat()
    elif cmd == "gold_extra":
        gold_extra()
    elif cmd == "silver":
        silver(tuple(sys.argv[2:]) or ("gold_draft", "synth", "pool"))
