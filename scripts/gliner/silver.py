# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Silver labels for the claim sample: entities (exact substrings with a
label) and the claim type, from GLM-4.7-Flash (Synthetic, syn:small:text).
Ten sentences per call (the API rate-limits). Cached under $GLINER_DIR/llm-cache. Writes $GLINER_DIR/silver.json.

    python silver.py
"""
import json

import common as C

ENT_LABELS = ["person", "location", "organization", "product", "date", "quantity", "event", "title"]
TYPES = ["fact", "preference", "plan", "event", "question", "request", "small_talk"]

PROMPT = """You label sentences from chats for an information-extraction benchmark.

1. Entities: list every mention of these types, copying the EXACT substring from that sentence:
- person: a named person or named pet ("Anna", "Dr. Lee", "Max"). Not pronouns, not "my sister".
- location: a named place: city, country, region, street, venue, landmark ("Lisbon", "Central Park").
- organization: a named company, school, team, institution, band, website or service ("Wells Fargo", "Coursera", "Instagram").
- product: a named product, brand, model, app or device ("iPhone 15", "Fitbit", "Duolingo" if used as an app).
- date: a date, day, month, year, time of day or relative time ("May 3rd", "last Saturday", "2019", "yesterday", "next week").
- quantity: a number with or without unit: counts, money, durations, distances, percentages, ages ("$350,000", "three issues", "2,000 miles", "two weeks", "25%").
- event: a named event, holiday or occasion ("Thanksgiving", "the Boston Marathon", "Pride parade").
- title: the title of a book, film, song, show, game, course or artwork ("The Great Gatsby", "Stranger Things").
Only specific mentions; skip generic nouns ("a bike", "my job"). An entity has one label. If none, return an empty list.

2. Type: the single best type of the sentence, from the speaker's point of view:
- fact: states something stable or current about the speaker, someone or the world (job, home, possessions, relationships, counts, status, habits).
- preference: a like, dislike, favourite, taste or opinion of the speaker.
- plan: an intention, plan, goal or future arrangement.
- event: something that happened at a particular time (a past experience or occurrence).
- question: asks for information.
- request: asks the listener to do or produce something (help, recommend, suggest, write, list), even phrased as a question.
- small_talk: greetings, thanks, reactions, compliments or encouragement with no new information.

Label each of these numbered sentences independently:
{text}

Answer with JSON only, one item per sentence in order: {{"items": [{{"i": 0, "entities": [{{"text": "...", "label": "..."}}], "type": "..."}}, ...]}}"""
BATCH = 10


def label_batch(items):
    text = "\n".join(f"{i}. {x['text']}" for i, x in enumerate(items))
    out = C.llm([{"role": "user", "content": PROMPT.format(text=text)}], max_tokens=12000)
    try:
        js = {j.get("i"): j for j in C.parse_json(out).get("items", [])}
    except Exception:  # noqa: BLE001
        js = {}
    res = []
    for i, item in enumerate(items):
        j = js.get(i)
        if j is None:
            res.append({"id": item["id"], "error": "missing"})
            continue
        ents = [e for e in j.get("entities", []) if e.get("label") in ENT_LABELS and e.get("text") and e["text"] in item["text"]]
        res.append({"id": item["id"], "entities": ents, "type": j.get("type"), "dropped": len(j.get("entities", [])) - len(ents)})
    return res


def main():
    sample = json.loads((C.GDIR / "sample.json").read_text())
    res = [r for b in C.pmap(label_batch, [sample[i:i + BATCH] for i in range(0, len(sample), BATCH)], workers=3) for r in b]
    (C.GDIR / "silver.json").write_text(json.dumps(res, indent=1))
    err = sum(1 for r in res if "error" in r)
    print(len(res), "labelled;", err, "parse errors;", sum(r.get("dropped", 0) for r in res), "entities dropped (not a substring / bad label)")


if __name__ == "__main__":
    main()
