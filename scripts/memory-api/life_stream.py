#!/usr/bin/env python3
# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""A scripted 40-turn "user life" stream for the memory API
(docs/specs/memory-api.md): moves, job changes, changing preferences,
restatements and unrelated facts, with gold labels and questions about the
current facts.

    python3 scripts/memory-api/life_stream.py            # write the data only
    python3 scripts/memory-api/life_stream.py --run      # and run it with real models

The data goes to .bench-data/memory-api/life_stream.json (next to the
worktrees), the run's report to .bench-data/memory-api/life_stream_report.json.
The run is `kleos-core/examples/memory_scenario.rs` (decider-nano +
EmbeddingGemma int8 on Metal; needs rustc 1.99 for candle).
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

# (day, text, gold, entities): gold is {"supersedes": [turn, ...]} for a turn
# that replaces earlier facts, {"duplicate_of": turn} for a restatement, or {}.
# entities are what an agent would pass in `remember(entities=...)`: the
# subject the statement is about ("user" for first-person statements, "Anna"
# for the sister, "Max" for the dog) plus the named things in it.
TURNS = [
    (1, "I live in Berlin.", {}, ["user", "Berlin"]),
    (2, "I work as a software engineer at Siemens.", {}, ["user", "Siemens"]),
    (3, "My favourite food is ramen.", {}, ["user"]),
    (4, "I have a dog named Max.", {}, ["user", "Max"]),
    (5, "I prefer tea over coffee.", {}, ["user"]),
    (6, "My sister Anna lives in Munich.", {}, ["Anna", "Munich"]),
    (7, "I play tennis on Saturdays.", {}, ["user"]),
    (8, "My rent is €1,200 per month.", {}, ["user"]),
    (10, "I'm learning Spanish.", {}, ["user"]),
    (12, "I live in Berlin.", {"duplicate_of": 0}, ["user", "Berlin"]),
    (15, "My favourite movie is Inception.", {}, ["user"]),
    (20, "I drive a 2015 Volkswagen Golf.", {}, ["user"]),
    (30, "I moved to Lisbon last week.", {"supersedes": [0]}, ["user", "Lisbon"]),
    (31, "My rent is now €1,500 per month.", {"supersedes": [7]}, ["user"]),
    (35, "I quit Siemens and joined Spotify as a product manager.", {"supersedes": [1]}, ["user", "Siemens", "Spotify"]),
    (36, "I prefer tea over coffee.", {"duplicate_of": 4}, ["user"]),
    (40, "My dog Max loves the beach.", {}, ["Max"]),
    (42, "My favourite food is now sushi, I'm over ramen.", {"supersedes": [2]}, ["user"]),
    (45, "I stopped playing tennis; now I go climbing on Saturdays.", {"supersedes": [6]}, ["user"]),
    (50, "My sister Anna moved from Munich to Hamburg.", {"supersedes": [5]}, ["Anna", "Munich", "Hamburg"]),
    (55, "I'm allergic to peanuts.", {}, ["user"]),
    (60, "I drink coffee every morning now; I no longer prefer tea.", {"supersedes": [4]}, ["user"]),
    (62, "My birthday is on March 3rd.", {}, ["user"]),
    (65, "I sold my Golf and bought a Tesla Model 3.", {"supersedes": [11]}, ["user"]),
    (70, "I'm learning Spanish.", {"duplicate_of": 8}, ["user"]),
    (72, "My partner's name is Sam.", {}, ["user", "Sam"]),
    (75, "I work as a product manager at Spotify.", {"duplicate_of": 14}, ["user", "Spotify"]),
    (80, "My favourite movie is Interstellar.", {"supersedes": [10]}, ["user"]),
    (85, "I go to the gym three times a week.", {}, ["user"]),
    (90, "I moved again, from Lisbon to Porto.", {"supersedes": [12]}, ["user", "Lisbon", "Porto"]),
    (92, "My rent in Porto is €1,100 per month.", {"supersedes": [13]}, ["user", "Porto"]),
    (95, "I adopted a cat called Luna.", {}, ["user", "Luna"]),
    (100, "I'm allergic to peanuts.", {"duplicate_of": 20}, ["user"]),
    (105, "I now go to the gym five times a week.", {"supersedes": [28]}, ["user"]),
    (110, "Sam and I got engaged.", {}, ["user", "Sam"]),
    (115, "I left Spotify to start my own company, Tidewater Labs.", {"supersedes": [14, 26]}, ["user", "Spotify", "Tidewater Labs"]),
    (118, "I finished learning Spanish and now I'm learning Portuguese.", {"supersedes": [8]}, ["user"]),
    (120, "My sister Anna is a doctor.", {}, ["Anna"]),
    (122, "I read a lot of science fiction.", {}, ["user"]),
    (125, "My favourite food is sushi.", {"duplicate_of": 17}, ["user"]),
]

# (query, expected substring of the top hit (any of "a|b"), stale texts that
# must not be the top hit, as_of day or None for now)
QUESTIONS = [
    ("Where do I live?", "Porto", ["Berlin", "Lisbon last week"], None),
    ("Where do I work?", "Tidewater", ["Siemens", "Spotify as a product"], None),
    ("How much is my rent?", "1,100", ["1,200", "1,500"], None),
    ("What is my favourite food?", "sushi", ["ramen."], None),
    ("What do I do on Saturdays?", "climbing", ["I play tennis"], None),
    ("Where does my sister Anna live?", "Hamburg", ["lives in Munich"], None),
    ("Do I prefer tea or coffee?", "coffee every morning", ["I prefer tea over coffee"], None),
    ("What car do I drive?", "Tesla", ["2015 Volkswagen"], None),
    ("What is my favourite movie?", "Interstellar", ["Inception"], None),
    ("How often do I go to the gym?", "five", ["three times"], None),
    ("Which language am I learning?", "Portuguese", ["I'm learning Spanish."], None),
    ("What pets do I have?", "Max|Luna", [], None),
    ("What am I allergic to?", "peanuts", [], None),
    ("What is my partner's name?", "Sam", [], None),
    ("When is my birthday?", "March", [], None),
    ("Who do I work for?", "Tidewater", ["Siemens", "Spotify as a product"], None),
    ("What city am I based in?", "Porto", ["Berlin", "Lisbon last week"], None),
    # time travel: what was true then
    ("Where do I live?", "Berlin", ["Lisbon", "Porto"], 20),
    ("Where do I work?", "Siemens", ["Spotify", "Tidewater"], 20),
    ("How much is my rent?", "1,500", ["1,200", "1,100"], 60),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="run the scenario with real models")
    ap.add_argument("--embedder", default="embeddinggemma", help="embeddinggemma or potion-base-8M")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    base = os.environ.get("KLEOS_BENCH_DATA")
    if not base:
        # the .bench-data directory next to the checkout (or its worktrees)
        base = next((p / ".bench-data" for p in root.parents if (p / ".bench-data").is_dir()), root / ".bench-data")
    bench = Path(base) / "memory-api"
    bench.mkdir(parents=True, exist_ok=True)
    data = {
        "turns": [{"day": d, "text": t, "entities": e, **g} for d, t, g, e in TURNS],
        "questions": [
            {"query": q, "expect": e, "stale": s, "as_of_day": a} for q, e, s, a in QUESTIONS
        ],
    }
    out = bench / "life_stream.json"
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(f"wrote {out}: {len(TURNS)} turns, {len(QUESTIONS)} questions")
    if args.run:
        cmd = [
            "cargo", "+1.99.0", "run", "--release", "-p", "kleos", "--features", "infer-metal",
            "--example", "memory_scenario", "--",
            str(out), str(bench / "life_stream_report.json"), args.embedder,
        ]
        subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
