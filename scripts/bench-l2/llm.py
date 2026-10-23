# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""A tiny OpenAI-compatible chat client for the QA step: every request and
response is cached on disk by a hash of (base url, model, messages, params),
so a re-run never re-spends quota; bounded concurrency; retries with backoff
on 429/5xx and on empty content (reasoning models can spend the whole budget
thinking).

Endpoints (KLEOS_BENCH_LLM):
  synthetic  https://api.synthetic.new/openai/v1, model syn:small:text; the key
             is read at call time from SYNTHETIC_KEY in the .env file next to
             the checkout (KLEOS_BENCH_ENV overrides the path). It is never
             printed, logged or written to the cache.
  lmstudio   http://localhost:1234/v1 (no key)
"""
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import common as C

ENDPOINTS = {
    "synthetic": ("https://api.synthetic.new/openai/v1", "syn:small:text"),
    "lmstudio": ("http://localhost:1234/v1", "gemma-4-12b-it-qat-assistant-mtp"),
}
_key_cache = {}
_lock = threading.Lock()


def _env_file() -> Path:
    if os.environ.get("KLEOS_BENCH_ENV"):
        return Path(os.environ["KLEOS_BENCH_ENV"])
    for p in C.ROOT.parents:
        if (p / ".env").is_file():
            return p / ".env"
    raise FileNotFoundError(".env with SYNTHETIC_KEY not found next to the checkout")


def _key(name):
    if name != "synthetic":
        return None
    if "k" not in _key_cache:
        for line in _env_file().read_text().splitlines():
            if line.startswith("SYNTHETIC_KEY="):
                _key_cache["k"] = line.split("=", 1)[1].strip().strip("'\"")
        if "k" not in _key_cache:
            raise KeyError("SYNTHETIC_KEY missing from .env")
    return _key_cache["k"]


class Client:
    def __init__(self, endpoint=None, model=None, thinking=False, max_tokens=None, cache_dir=None):
        self.endpoint = endpoint or os.environ.get("KLEOS_BENCH_LLM", "synthetic")
        base, default_model = ENDPOINTS[self.endpoint]
        self.base = base
        self.model = model or default_model
        self.thinking = thinking
        self.max_tokens = max_tokens or (2500 if thinking else 400)
        self.cache = Path(cache_dir or C.bench_dir() / "llm-cache")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.stats = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0}

    def _params(self):
        p = {"temperature": 0, "max_tokens": self.max_tokens}
        if not self.thinking and self.endpoint == "synthetic":
            p["reasoning_effort"] = "none"
        return p

    def chat(self, messages):
        """-> {"content", "model", "usage"} (cached)."""
        params = self._params()
        h = hashlib.sha256(json.dumps([self.base, self.model, messages, params], sort_keys=True).encode()).hexdigest()
        f = self.cache / h[:2] / f"{h}.json"
        if f.exists():
            with _lock:
                self.stats["cached"] += 1
            return json.loads(f.read_text())["response"]
        body = json.dumps({"model": self.model, "messages": messages, **params}).encode()
        headers = {"Content-Type": "application/json"}
        k = _key(self.endpoint)
        if k:
            headers["Authorization"] = "Bearer " + k
        delay = 2.0
        last = None
        for attempt in range(8):
            try:
                req = urllib.request.Request(self.base + "/chat/completions", data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=600) as r:
                    d = json.loads(r.read())
                content = (d["choices"][0]["message"].get("content") or "").strip()
                if not content:
                    raise ValueError("empty content")
                out = {"content": content, "model": d.get("model"), "usage": d.get("usage")}
                f.parent.mkdir(exist_ok=True)
                tmp = f.with_suffix(".tmp")
                tmp.write_text(json.dumps({"request": {"model": self.model, "messages": messages, **params},
                                           "response": out}))
                tmp.rename(f)
                with _lock:
                    self.stats["calls"] += 1
                    u = d.get("usage") or {}
                    self.stats["prompt_tokens"] += u.get("prompt_tokens", 0)
                    self.stats["completion_tokens"] += u.get("completion_tokens", 0)
                return out
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code not in (408, 409, 429) and e.code < 500:
                    raise RuntimeError(f"{last}: {e.read()[:300]!r}") from None
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, ConnectionError) as e:
                last = type(e).__name__ + ": " + str(e)[:100]
            with _lock:
                self.stats["retries"] += 1
            time.sleep(delay + random.random())
            delay = min(delay * 2, 60)
        raise RuntimeError(f"LLM call failed after retries: {last}")


def quotas():
    """Synthetic's quota snapshot (no key in the output)."""
    req = urllib.request.Request("https://api.synthetic.new/v2/quotas",
                                 headers={"Authorization": "Bearer " + _key("synthetic")})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())
