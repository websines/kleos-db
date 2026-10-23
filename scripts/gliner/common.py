# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Shared helpers for the GLiNER2.5 spike (docs/research/gliner.md).

Paths: everything (data, caches, results) lives in GLINER_DIR, default
.bench-data/gliner next to the checkout; models come from HF_HOME.

LLM: the silver labeller is GLM-4.7-Flash through the Synthetic API
(model syn:small:text). The key is read at call time from SYNTHETIC_KEY in
the .env next to the checkout; it is never printed, logged or cached. Every
call is cached on disk by a hash of (model, messages, params).
"""
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
CHECKOUT = HERE.parents[2]
ROOT = Path(os.environ.get("KLEOS_ROOT", "/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr"))
GDIR = Path(os.environ.get("GLINER_DIR", ROOT / ".bench-data" / "gliner"))
LME_SNAP = Path("/Volumes/SSD01/caches/huggingface/hub/datasets--xiaowu0162--longmemeval-cleaned/snapshots/98d7416c24c778c2fee6e6f3006e7a073259d48f")
LOCOMO = ROOT / ".bench-data" / "bench-l2" / "data" / "locomo" / "locomo10.json"
SEED = 20261023  # the decider-nano data build's seed; the KU hold-out split reuses it

MODELS = {
    "small": "fastino/gliner2.5-small-v1",
    "base": "fastino/gliner2.5-base-v1",
    "multi": "fastino/gliner2.5-multi-v1",
    "decide": "fastino/GLiNER2.5-Decide",
}


def lme(name="longmemeval_oracle.json"):
    return json.loads((LME_SNAP / name).read_text())


def lme_ku_split(raw=None):
    """Held-out (25) and trained-on (53) KU question ids; same as
    scripts/bench-l2/common.py lme_ku_split()[:2]. Drop *_abs for the 23."""
    raw = raw or lme()
    ku = sorted([x for x in raw if x["question_type"] == "knowledge-update"], key=lambda x: x["question_id"])
    random.Random(SEED + 1).shuffle(ku)
    return {x["question_id"] for x in ku[:25]}, {x["question_id"] for x in ku[25:]}


def micros(d):
    t = datetime.strptime(d.split(" (")[0] + " " + d.split(") ")[1], "%Y/%m/%d %H:%M")
    return int(t.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)


_splitter = None


def split_claims(text):
    """kleos's own claim splitter (memory_claims::split_claims) through a
    model-free in-memory store: returns [(claim_text, declarative)]."""
    global _splitter
    if _splitter is None:
        from kleos import CozoDbPy
        _splitter = CozoDbPy("mem", "", "{}")
        _splitter.memory_init()
    r = _splitter.remember(text, ns="split")
    if r.get("claims"):
        return [(c["text"], bool(c["decided"])) for c in r["claims"]]
    return [(text, True)]


def _patch_tokenizer_loader():
    """The 2.5 checkpoints store `extra_special_tokens` as a list (Transformers
    5 metadata). gliner2 2.0.0 retries with `extra_special_tokens={}` on the
    AttributeError, but Transformers 4.57 catches that error itself and falls
    back to the slow SentencePiece tokenizer (no spm.model in the repo). Go
    straight to gliner2's own compatibility path; the [P]/[E]/[C]... tokens
    are added tokens in tokenizer.json either way."""
    from transformers import AutoTokenizer
    import gliner2.models.base as b
    import gliner2.models.boundary.model as bm
    import gliner2.models.span.model as sm

    def load(repo_or_dir):
        return AutoTokenizer.from_pretrained(repo_or_dir, extra_special_tokens={})
    b.load_extractor_tokenizer = bm.load_extractor_tokenizer = sm.load_extractor_tokenizer = load


def load_gliner(name, device="cpu"):
    from gliner2 import AutoExtractor
    _patch_tokenizer_loader()
    return AutoExtractor.from_pretrained(MODELS.get(name, name), map_location=device)


# ---------------------------------------------------------------- LLM (silver)
BASE = "https://api.synthetic.new/openai/v1"
LLM_MODEL = "syn:small:text"
_key = {}
_lock = threading.Lock()
_sem = threading.Semaphore(int(os.environ.get("GLINER_LLM_CONC", "2")))  # the key is rate-limited (429s at 8 concurrent)


def _api_key():
    if "k" not in _key:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("SYNTHETIC_KEY="):
                _key["k"] = line.split("=", 1)[1].strip().strip("'\"")
        if "k" not in _key:
            raise KeyError("SYNTHETIC_KEY missing from .env")
    return _key["k"]


def llm(messages, max_tokens=4000, temperature=0.0, cache="llm-cache", tries=14, **params):
    body = {"model": LLM_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, **params}
    h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    cdir = GDIR / cache
    cdir.mkdir(parents=True, exist_ok=True)
    cf = cdir / f"{h}.json"
    if cf.exists():
        return json.loads(cf.read_text())["content"]
    data = json.dumps(body).encode()
    last = None
    for i in range(tries):
        req = urllib.request.Request(BASE + "/chat/completions", data=data, headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + _api_key()})
        try:
            with _sem, urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read())
            content = resp["choices"][0]["message"].get("content") or ""
            if content.strip():
                cf.write_text(json.dumps({"request": body, "content": content, "usage": resp.get("usage")}))
                return content
            last = "empty content"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            ra = e.headers.get("retry-after") if e.headers else None
            if ra and ra.replace(".", "").isdigit():
                time.sleep(min(120, float(ra)))
                continue
        except Exception as e:  # noqa: BLE001
            last = type(e).__name__
        time.sleep(min(90, 3 * 2 ** i * (0.5 + random.random())))
    raise RuntimeError(f"LLM call failed: {last}")


def parse_json(s):
    s = s.strip()
    if "```" in s:
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
    a, b = s.find("{"), s.rfind("}")
    return json.loads(s[a:b + 1])


def pmap(fn, items, workers=8):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))
