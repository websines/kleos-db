#!/usr/bin/env python3
"""Routing-at-scale benchmark data (docs/benchmarks/skills.md).

Builds, under <data>/ (default .bench-data/skills):
  pool/<name>/SKILL.md   every public skill found under <data>/src (one per name)
  targets.json           30 target skills with distinctive descriptions
  tasks.json             two user tasks per target, written by the LLM
  distractors.json       1,000 distractors (first 200 = the mid-size set)

Public skills come from GitHub repositories downloaded to <data>/src as
tarballs (see SOURCES.md there). They are untrusted text: this script only
parses them. The LLM (Synthetic, OpenAI-compatible) writes the tasks; its key
is read from the root .env at run time and never printed or stored.
"""
import hashlib
import json
import os
import random
import re
import sys
import urllib.request

import yaml

DATA = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "../../../.bench-data/skills"))
ENV = os.environ.get("KLEOS_ENV_FILE", os.path.join(os.path.dirname(__file__), "../../../.env"))
URL = "https://api.synthetic.new/openai/v1/chat/completions"
MODEL = "syn:small:text"
N_TARGETS = 30
N_DISTRACTORS = 1000


def key():
    for line in open(ENV):
        if line.startswith("SYNTHETIC_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("SYNTHETIC_KEY not found in " + ENV)


def llm(prompt, max_tokens=6000):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.3}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + key()})
    for attempt in range(4):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            return r["choices"][0]["message"]["content"] or ""
        except Exception as e:  # noqa: BLE001
            print("llm retry", attempt, type(e).__name__, file=sys.stderr)
    raise SystemExit("llm failed")


def slug(s):
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return re.sub(r"-+", "-", s)[:64].strip("-")


def parse(path):
    text = open(path, encoding="utf-8", errors="replace").read().lstrip("﻿")
    if not text.startswith("---"):
        return None
    parts = text.split("\n---", 1)
    if len(parts) < 2:
        return None
    try:
        fm = yaml.safe_load(parts[0][3:])
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(fm, dict) or not isinstance(fm.get("name"), str) or not isinstance(fm.get("description"), str):
        return None
    body = parts[1].split("\n", 1)[1] if "\n" in parts[1] else ""
    return fm, body


def words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def main():
    src = os.path.join(DATA, "src")
    pool = {}
    for root, _, files in sorted(os.walk(src)):
        if "SKILL.md" not in files:
            continue
        p = parse(os.path.join(root, "SKILL.md"))
        if not p:
            continue
        fm, body = p
        name = slug(fm["name"])
        desc = " ".join(fm["description"].split())
        if not name or not desc or len(desc) > 1024 or name in pool:
            continue
        repo = os.path.relpath(root, src).split(os.sep)[0]
        pool[name] = {"name": name, "description": desc, "body": body, "repo": repo, "path": os.path.relpath(root, src)}
    # one description, one skill
    seen, uniq = set(), {}
    for n, s in pool.items():
        h = hashlib.sha256(s["description"].lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            uniq[n] = s
    pool = uniq
    print(len(pool), "unique skills", file=sys.stderr)
    out = os.path.join(DATA, "pool")
    os.makedirs(out, exist_ok=True)
    for n, s in pool.items():
        d = os.path.join(out, n)
        os.makedirs(d, exist_ok=True)
        fm = {"name": n, "description": s["description"]}
        with open(os.path.join(d, "SKILL.md"), "w") as f:
            f.write("---\n" + yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, width=10_000) + "---\n" + s["body"])
    rng = random.Random(7)
    ws = {n: words(s["description"]) for n, s in pool.items()}
    names = sorted(pool)
    # targets: distinctive descriptions with a real body, spread over repos
    cands = [n for n in names if 80 <= len(pool[n]["description"]) <= 500 and len(pool[n]["body"]) >= 400
             and max(jac(ws[n], ws[m]) for m in names if m != n) < 0.35]
    rng.shuffle(cands)
    targets, per_repo = [], {}
    for n in cands:
        r = pool[n]["repo"]
        if per_repo.get(r, 0) >= 5:
            continue
        per_repo[r] = per_repo.get(r, 0) + 1
        targets.append(n)
        if len(targets) == N_TARGETS:
            break
    # distractors: nothing that overlaps a target's purpose
    others = [n for n in names if n not in targets and all(jac(ws[n], ws[t]) < 0.3 for t in targets)]
    rng.shuffle(others)
    distractors = others[:N_DISTRACTORS]
    if len(distractors) < N_DISTRACTORS:
        raise SystemExit(f"only {len(distractors)} distractors")
    json.dump(targets, open(os.path.join(DATA, "targets.json"), "w"), indent=1)
    json.dump(distractors, open(os.path.join(DATA, "distractors.json"), "w"), indent=1)
    tasks_path = os.path.join(DATA, "tasks.json")
    if os.path.exists(tasks_path):
        print("tasks.json exists; keeping it", file=sys.stderr)
        return
    tasks = []
    for i in range(0, len(targets), 10):
        batch = targets[i:i + 10]
        spec = "\n\n".join(f"### {n}\nDescription: {pool[n]['description']}\nInstructions (start): {pool[n]['body'][:1200]}" for n in batch)
        prompt = (
            "For each skill below, write TWO realistic requests a user might give an AI agent, for which this skill is the right one to load.\n"
            "Rules: write as the user (first person or imperative), concrete and specific; do NOT use the skill's name, and do not copy distinctive phrases "
            "from its description; each request is one to three sentences.\n"
            'Answer with JSON only: [{"name": "<skill name>", "tasks": ["...", "..."]}, ...]\n\n' + spec)
        text = llm(prompt)
        m = re.search(r"\[.*\]", text, re.S)
        arr = json.loads(m.group(0)) if m else []
        for o in arr:
            if o.get("name") in batch:
                for t in o.get("tasks", [])[:2]:
                    tasks.append({"target": o["name"], "task": t})
        print(f"batch {i // 10}: {len(arr)} skills", file=sys.stderr)
    json.dump(tasks, open(tasks_path, "w"), indent=1)
    print(len(targets), "targets,", len(distractors), "distractors,", len(tasks), "tasks", file=sys.stderr)


if __name__ == "__main__":
    main()
