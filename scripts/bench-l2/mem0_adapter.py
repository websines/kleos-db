#!/usr/bin/env python3
# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Head-to-head: Mem0 (open source, `mem0ai`) through the same harness, reader
and judge as the kleos runs (docs/benchmarks/memory.md, "Head-to-head: Mem0").

Runs in its own venv (`mem0ai`, sentence-transformers, spaCy, fastembed); it
never imports kleos. Mem0 is used as shipped, through `Memory.add` (its LLM
extraction pipeline) and `Memory.search` (semantic + BM25 + entity boost),
with these adapter choices:

  LLM        the harness's Synthetic `syn:small:text` (GLM-4.7-Flash), thinking
             off, temperature 0, JSON response format, max_tokens from
             MEM0_MAX_TOKENS (Mem0's default 2000; the runs used 4000). The call goes through llm.Client, so every request is
             cached by prompt hash and the key never reaches Mem0.
  embedder   EmbeddingGemma (`google/embeddinggemma-300m`, the model kleos uses)
             via sentence-transformers, fp32, 512-token window, with its query /
             document prompts (Mem0's `memory_action` "search" -> query prompt).
  store      Mem0's default Qdrant, in-process and in-memory, one per unit.
  time       OSS `add()` rejects the platform-only `timestamp`, so the adapter
             fills the extraction prompt's own "Observation Date" slot with the
             session date (what the platform's timestamp does) and passes the
             session time as `created_at` metadata, which Mem0 keeps.
  provenance Mem0 does not say which turns a memory came from. The adapter tags
             each add() with the source turn ids as metadata (copied onto every
             memory that add creates), which gives retrieval R@k at batch level.
  telemetry  MEM0_TELEMETRY=False before import (no PostHog, no remote notices).

    python mem0_adapter.py ingest --bench locomo --batch session --workers 6
    python mem0_adapter.py ingest --bench lme --n 10 --workers 4
    python mem0_adapter.py qa --bench locomo --mem0-k 10,20 --kleos hybrid
    python mem0_adapter.py qa --bench lme --n 10
    python mem0_adapter.py ingest --bench locomo --only conv-26 --workers 1 --suffix=-latency
    python mem0_adapter.py report

Every add() is one LLM call of ~9-11k prompt tokens. The Synthetic account
limit ("Too many concurrent requests") is shared with other jobs, which is why
LoCoMo ran one add() per session (`--batch session`) rather than Mem0's
two-message protocol (`--batch pair`, implemented).

Results: <.bench-data>/mem0-h2h/results/ (KLEOS_MEM0_DIR overrides).
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def out_dir() -> Path:
    if os.environ.get("KLEOS_MEM0_DIR"):
        return Path(os.environ["KLEOS_MEM0_DIR"])
    return C.bench_dir().parent / "mem0-h2h"


OUT = out_dir()
RES = OUT / "results"
# Mem0 reads these at import time: no telemetry, state under the bench dir, not ~/.mem0
os.environ["MEM0_TELEMETRY"] = "False"
os.environ.setdefault("MEM0_DIR", str(OUT / "mem0-home"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MEM0_VERSION_PIN = "2.2.1"  # released 2026-09-25
EMBED_MODEL = "google/embeddinggemma-300m"
Q_PROMPT = "task: search result | query: "  # kleos-infer/src/gemma.rs GEMMA_QUERY_PROMPT
D_PROMPT = "title: none | text: "  # GEMMA_DOCUMENT_PROMPT
RUN_DATE = "2026-10-23"  # the extraction prompt's "Current Date" (pinned so cached prompts stay valid)
MEM0_MAX_TOKENS = int(os.environ.get("MEM0_MAX_TOKENS", "2000"))  # Mem0's OpenAI config default is 2000
KLEOS_TAGS = ("1c1bf81", "6f9711f")  # main 92ef789's engine tag first; 6f9711f only as a fallback

_tl = threading.local()
_gemma = {}
_glock = threading.Lock()


# ------------------------------------------------------------------ Mem0 wiring
def _gemma_encode(texts, query: bool):
    with _glock:
        if "m" not in _gemma:
            from sentence_transformers import SentenceTransformer

            dev = os.environ.get("MEM0_EMBED_DEVICE", "mps")
            m = SentenceTransformer(EMBED_MODEL, device=dev)
            m.max_seq_length = 512  # kleos embeds at most 512 tokens per memory
            _gemma["m"] = m
        t = time.perf_counter()
        out = _gemma["m"].encode(list(texts), prompt=Q_PROMPT if query else D_PROMPT,
                                 convert_to_numpy=True, normalize_embeddings=True).tolist()
        _tl.embed_s = getattr(_tl, "embed_s", 0.0) + time.perf_counter() - t  # compute only, not lock wait
        return out


def patch_mem0():
    """Telemetry check, shared EmbeddingGemma, session-dated extraction prompt."""
    import mem0
    import mem0.memory.main as mm
    import mem0.memory.telemetry as tel
    from mem0.embeddings import huggingface as hfe

    assert mem0.__version__ == MEM0_VERSION_PIN, mem0.__version__
    assert tel.MEM0_TELEMETRY is False, "Mem0 telemetry must be off"

    class _Shared:  # what HuggingFaceEmbedding.__init__ asks of a SentenceTransformer
        def get_sentence_embedding_dimension(self):
            return 768

    hfe.SentenceTransformer = lambda *a, **k: _Shared()
    hfe.HuggingFaceEmbedding.embed = lambda self, text, memory_action=None: _gemma_encode(
        [text], memory_action == "search")[0]
    hfe.HuggingFaceEmbedding.embed_batch = lambda self, texts, memory_action="add": (
        _gemma_encode(texts, memory_action == "search") if texts else [])

    if not getattr(mm, "_h2h_dated", False):
        orig = mm.generate_additive_extraction_prompt

        def dated(*a, **k):
            if k.get("timestamp") is None and getattr(_tl, "obs", None):
                k["timestamp"] = _tl.obs  # -> "## Observation Date"
            if k.get("current_date") is None:
                k["current_date"] = RUN_DATE
            return orig(*a, **k)

        mm.generate_additive_extraction_prompt = dated
        mm._h2h_dated = True
    return mem0


def new_memory(st):
    """A fresh Mem0 Memory (in-memory Qdrant + SQLite) whose LLM is the cached harness client."""
    from mem0 import Memory
    from qdrant_client import QdrantClient

    from llm import Client

    cfg = {
        "llm": {"provider": "openai", "config": {
            "model": "syn:small:text", "api_key": "not-used", "openai_base_url": "http://127.0.0.1:9/v1",
            "temperature": 0, "max_tokens": MEM0_MAX_TOKENS}},
        "embedder": {"provider": "huggingface", "config": {"model": EMBED_MODEL, "embedding_dims": 768}},
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": "mem", "embedding_model_dims": 768, "client": QdrantClient(location=":memory:")}},
        "history_db_path": ":memory:",
    }
    m = Memory.from_config(cfg)
    cli = Client(max_tokens=MEM0_MAX_TOKENS, cache_dir=OUT / "llm-cache",
                 extra_params={"response_format": {"type": "json_object"}},
                 retries=120, retry_cap=6.0)  # the shared account limit: poll for a free slot, don't back off for a minute

    def generate_response(messages, response_format=None, tools=None, tool_choice="auto", **kw):
        c0, r0 = cli.stats["cached"], cli.stats["retries"]
        t = time.perf_counter()
        r = cli.chat(messages)
        st["llm_429_retries"] += cli.stats["retries"] - r0
        dt = time.perf_counter() - t
        u = r.get("usage") or {}
        st["llm_calls"] += 1
        st["llm_cached"] += cli.stats["cached"] - c0
        st["prompt_tokens"] += u.get("prompt_tokens", 0)
        st["completion_tokens"] += u.get("completion_tokens", 0)
        st["llm_s"] += dt
        if r.get("finish_reason") == "length":
            st["truncated"] += 1
        return r["content"]

    m.llm.generate_response = generate_response
    return m


def new_stats():
    return {"adds": 0, "add_errors": 0, "add_retries": 0, "memories_added": 0, "llm_calls": 0, "llm_cached": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "llm_s": 0.0, "truncated": 0, "llm_429_retries": 0}


def iso(us):
    return datetime.fromtimestamp(us / 1e6, tz=timezone.utc).isoformat()


def day(us):
    return datetime.fromtimestamp(us / 1e6, tz=timezone.utc).date().isoformat()


def do_add(m, uid, msgs, at_us, srcs, st, log):
    _tl.obs = day(at_us)
    _tl.embed_s = 0.0
    c0, l0 = st["llm_calls"], st["llm_s"]
    md = {"created_at": iso(at_us), "srcs": ",".join(srcs)}
    t = time.perf_counter()
    for attempt in range(6):  # the account-wide concurrency limit (HTTP 429) is shared with other jobs
        try:
            r = m.add(msgs, user_id=uid, metadata=md)
            n = len(r.get("results") or [])
            err = None
            break
        except Exception as e:  # LLMError after the client's own retries: wait, then retry the add
            n, err = 0, f"{type(e).__name__}: {str(e)[:200]}"
            print(f"  add failed ({uid} {srcs[0]}, attempt {attempt + 1}): {err}", file=sys.stderr, flush=True)
            time.sleep(30 + 30 * attempt)
    if err:
        st["add_errors"] += 1
    st["adds"] += 1
    st["add_retries"] += attempt
    st["memories_added"] += n
    log.append({"srcs": srcs, "ms": (time.perf_counter() - t) * 1000, "n": n, "err": err,
                "llm_ms": (st["llm_s"] - l0) * 1000, "embed_ms": _tl.embed_s * 1000})


def search(m, uid, q, k):
    _tl.embed_s = 0.0
    t = time.perf_counter()
    r = m.search(q, top_k=k, filters={"user_id": uid})
    ms = (time.perf_counter() - t) * 1000
    m._h2h_embed_ms = _tl.embed_s * 1000
    hits = []
    for h in r.get("results") or []:
        md = h.get("metadata") or {}
        hits.append({"id": h["id"], "text": h["memory"], "score": h.get("score"), "created_at": h.get("created_at"),
                     "srcs": [s for s in (md.get("srcs") or "").split(",") if s]})
    return hits, ms


def dump_memories(m, uid):
    try:
        r = m.get_all(filters={"user_id": uid}, top_k=100000)
    except TypeError:
        r = m.get_all(filters={"user_id": uid})
    out = []
    for h in r.get("results") or []:
        md = h.get("metadata") or {}
        out.append({"id": h["id"], "text": h["memory"], "created_at": h.get("created_at"), "srcs": md.get("srcs")})
    return out


def write_json(f, obj):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.rename(f)


# ------------------------------------------------------------------ ingest
def ingest_locomo(conv, outdir, batch="session"):
    """Speaker A as `user`, speaker B as `assistant`, content "Name: text" (as in Mem0's
    evaluation/src/memzero/add.py). batch='pair': two messages per add() within a session (Mem0's
    LoCoMo protocol); batch='session': one add() per session (what the shared rate limit allowed).
    One store per conversation (Mem0 v3 extracts from both roles and attributes named speakers)."""
    sid = conv["sample_id"]
    f = outdir / f"{sid}.json"
    if f.exists():
        return
    a = conv["conversation"]["speaker_a"]
    units = C.locomo_units(conv)
    st, log = new_stats(), []
    m = new_memory(st)
    uid = "c_" + sid
    t0 = time.perf_counter()
    by_sess = {}
    for u in units:
        by_sess.setdefault(u["session"], []).append(u)
    for sess, us in by_sess.items():
        step = 2 if batch == "pair" else len(us)
        for i in range(0, len(us), step):
            b = us[i:i + step]
            msgs = [{"role": "user" if u["text"].startswith(a + ":") else "assistant", "content": u["text"]} for u in b]
            do_add(m, uid, msgs, b[0]["at"], [u["src"] for u in b], st, log)
        print(f"  {sid} {sess}: {st['adds']} adds, {st['memories_added']} memories, {st['llm_calls']} llm "
              f"({st['llm_cached']} cached)", flush=True)
    ingest_s = time.perf_counter() - t0
    qs = []
    for qi, q in enumerate(conv["qa"]):
        if q["category"] == 5:
            continue
        h20, ms20 = search(m, uid, q["question"], 20)
        h10, ms10 = search(m, uid, q["question"], 10)
        e10 = m._h2h_embed_ms
        h5, ms5 = search(m, uid, q["question"], 5)
        qs.append({"hits5": h5, "embed_ms_k10": e10,"qid": f"{sid}/{qi}", "question": q["question"], "answer": q.get("answer"),
                   "category": q["category"], "evidence": C.locomo_evidence(q.get("evidence")),
                   "search_ms": {"k5": ms5, "k10": ms10, "k20": ms20}, "hits10": h10, "hits20": h20})
    mems = dump_memories(m, uid)
    write_json(f, {"sample_id": sid, "n_units": len(units), "ingest": st, "ingest_s": ingest_s, "adds": log,
                   "n_memories": len(mems), "memories": mems, "questions": qs})
    print(f"{sid}: {len(units)} turns, {st['adds']} adds in {ingest_s:.0f}s -> {len(mems)} memories; "
          f"{st['llm_calls']} llm calls, {st['prompt_tokens']}+{st['completion_tokens']} tok, "
          f"{st['add_errors']} errors, {st['truncated']} truncated", flush=True)


def lme_sample(n):
    """A stratified n of the 119-question hybrid set (the 25 held-out KU + a stratified 99)."""
    for tag in KLEOS_TAGS:
        d = C.bench_dir() / "results" / tag / "qa" / "lme-user" / "hybrid" / "kleos_hybrid"
        if d.is_dir() and len(list(d.glob("*.json"))) >= 119:
            ids = sorted(p.stem for p in d.glob("*.json"))
            break
    else:
        raise SystemExit("no cached 119-question hybrid set")
    data = {x["question_id"]: x for x in C.lme_questions()}
    items = [data[i] for i in ids]
    if n and n < len(items):
        items = C.stratified_sample(items, C.lme_category, n)
    return sorted(items, key=lambda x: x["question_id"])


def ingest_lme(x, outdir):
    """One add() per session (user and assistant messages as they are), in date order.
    Per-round adds (Mem0's usual granularity) would cost ~5x the quota; see the doc."""
    qid = x["question_id"]
    f = outdir / f"{qid}.json"
    if f.exists():
        return
    st, log = new_stats(), []
    m = new_memory(st)
    uid = "q_" + qid
    sess = sorted(zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"]),
                  key=lambda z: C.lme_ts(z[1]))
    t0 = time.perf_counter()
    n_turns = 0
    for sid, date, turns in sess:
        msgs = [{"role": t["role"], "content": t["content"].strip()} for t in turns if t["content"].strip()]
        n_turns += len(msgs)
        if msgs:
            do_add(m, uid, msgs, C.lme_ts(date), [sid], st, log)
    ingest_s = time.perf_counter() - t0
    h20, ms20 = search(m, uid, x["question"], 20)
    h10, ms10 = search(m, uid, x["question"], 10)
    e10 = m._h2h_embed_ms
    h5, ms5 = search(m, uid, x["question"], 5)
    mems = dump_memories(m, uid)
    write_json(f, {"question_id": qid, "question_type": x["question_type"], "category": C.lme_category(x),
                   "question": x["question"], "question_date": x["question_date"], "answer": x["answer"],
                   "answer_session_ids": x["answer_session_ids"], "n_sessions": len(sess), "n_turns": n_turns,
                   "ingest": st, "ingest_s": ingest_s, "adds": log, "n_memories": len(mems), "memories": mems,
                   "search_ms": {"k5": ms5, "k10": ms10, "k20": ms20}, "embed_ms_k10": e10, "hits5": h5, "hits10": h10, "hits20": h20})
    print(f"{qid} {C.lme_category(x)}: {len(sess)} sessions / {n_turns} msgs in {ingest_s:.0f}s -> {len(mems)} "
          f"memories; {st['llm_calls']} llm, {st['prompt_tokens']}+{st['completion_tokens']} tok, "
          f"{st['add_errors']} errors, {st['truncated']} truncated", flush=True)


def cmd_ingest(args):
    patch_mem0()
    from llm import quotas

    try:
        print("quota before:", json.dumps(quotas()["planWeeklyLimit"]), flush=True)
    except Exception as e:
        print("quota check failed:", e)
    if args.bench == "locomo":
        outdir = RES / (f"locomo-{args.batch}" + args.suffix)
        units = C.locomo_data()
        if args.only:
            units = [c for c in units if c["sample_id"] in args.only.split(",")]

        def fn(u, d):
            return ingest_locomo(u, d, args.batch)
    else:
        outdir = RES / f"lme-n{args.n}{args.suffix}"
        units = lme_sample(args.n)
        if args.only:
            units = [x for x in units if x["question_id"] in args.only.split(",")]
        fn = ingest_lme
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"-> {outdir}: {len(units)} units, {args.workers} workers", flush=True)

    def one(u):
        try:
            fn(u, outdir)
        except Exception:
            print("unit failed:\n" + traceback.format_exc()[-1500:], file=sys.stderr, flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, units))
    try:
        print("quota after:", json.dumps(quotas()["planWeeklyLimit"]), flush=True)
    except Exception as e:
        print("quota check failed:", e)


# ------------------------------------------------------------------ QA (same reader + judge as qa.py)
def kleos_qa_dir(bench_set):
    for tag in KLEOS_TAGS:
        d = C.bench_dir() / "results" / tag / "qa" / bench_set
        if (d / "kleos_hybrid").is_dir():
            return tag, d
    raise SystemExit(f"no cached kleos QA for {bench_set}")


def cmd_qa(args):
    import qa as Q
    from llm import Client, quotas

    # the harness reader/judge settings (temperature 0, 400 tokens); short polling on the shared 429 limit
    cli = Client(cache_dir=OUT / "llm-cache", retries=120, retry_cap=6.0)
    jobs = []
    kt = args.kleos_tag
    kvars = [v for v in args.kleos.split(",") if v]
    mconds = [(f"mem0_k{k}", f"hits{k}") for k in args.mem0_k.split(",") if k]
    krecs = {}  # kleos recall hits of main's build (read-only), formatted exactly as qa.py formats them
    if args.bench == "locomo":
        tag, kd = kleos_qa_dir("locomo-turn/all-n300")
        qids = sorted(p.stem for p in (kd / "kleos_hybrid").glob("*.json"))  # the same 301 questions
        convs = {c["sample_id"]: c for c in C.locomo_data()}
        recs = {}
        for p in (RES / f"locomo-{args.batch}").glob("*.json"):
            for q in json.loads(p.read_text())["questions"]:
                recs[q["qid"].replace("/", "_")] = q
        for cond, key in mconds:
            for qid in qids:
                if qid in recs:
                    jobs.append(("locomo", cond, key, qid, RES / "qa" / f"locomo-{args.batch}" / cond / f"{qid}.json"))
        for v in kvars:
            krecs[v] = {}
            for p in (C.bench_dir() / "results" / kt / "locomo-turn" / v).glob("conv-*.json"):
                for q in json.loads(p.read_text())["questions"]:
                    krecs[v][q["qid"].replace("/", "_")] = q
            for qid in qids:
                if qid in krecs[v]:  # the report pairs conditions on their common questions
                    jobs.append(("locomo", f"kleos_{v}@{kt}", v, qid,
                                 RES / "qa" / f"locomo-{args.batch}" / f"kleos_{v}@{kt}" / f"{qid}.json"))
        missing = [q for q in qids if q not in recs]
    else:
        items = {x["question_id"]: x for x in lme_sample(args.n)}
        rd = RES / f"lme-n{args.n}"
        recs = {p.stem: json.loads(p.read_text()) for p in rd.glob("*.json")}
        for cond, key in mconds:
            for qid in sorted(items):
                if qid in recs:
                    jobs.append(("lme", cond, key, qid, RES / "qa" / f"lme-n{args.n}" / cond / f"{qid}.json"))
        for v in kvars:
            krecs[v] = {}
            for qid in sorted(items):
                f = C.bench_dir() / "results" / kt / "lme-user" / v / f"{qid}.json"
                if f.exists():
                    krecs[v][qid] = json.loads(f.read_text())
                    jobs.append(("lme", f"kleos_{v}@{kt}", v, qid,
                                 RES / "qa" / f"lme-n{args.n}" / f"kleos_{v}@{kt}" / f"{qid}.json"))
        missing = [q for q in items if q not in recs]
    if missing:
        print(f"WARNING: {len(missing)} questions have no Mem0 ingest yet: {missing[:5]}", flush=True)

    def mem_lines(hits, fmt):
        hs = sorted(hits, key=lambda h: h.get("created_at") or "")
        out = []
        for h in hs:
            try:
                us = int(datetime.fromisoformat(h["created_at"]).timestamp() * 1e6)
            except Exception:
                us = None
            out.append(fmt(us, h["text"]))
        return out

    def one(job):
        bench, cond, key, qid, f = job
        if f.exists():
            return json.loads(f.read_text())
        kleos = cond.startswith("kleos_")
        rec = krecs[key][qid] if kleos else recs[qid]
        if bench == "locomo":
            sid, qi = qid.rsplit("_", 1)
            conv = convs[sid]
            q = conv["qa"][int(qi)]
            a, b = conv["conversation"]["speaker_a"], conv["conversation"]["speaker_b"]
            if kleos:  # qa.py run_locomo, kleos:<variant>
                hits = sorted(rec["hits10"], key=lambda h: h.get("valid_from") or 0)
                m = "\n".join(f"[{Q.locomo_date(h['valid_from'])}] {h['text']}" for h in hits)
            else:
                m = "\n".join(mem_lines(rec[key], lambda us, t: f"[{Q.locomo_date(us)}] {t}" if us else t))
            r = cli.chat([{"role": "user", "content": Q.LOCOMO_READER.format(a=a, b=b, m=m, q=q["question"])}])
            gold = str(q.get("answer"))
            j = cli.chat([{"role": "user", "content": Q.LOCOMO_JUDGE.format(q=q["question"], g=gold, r=r["content"])}])
            v = j["content"].upper()
            out = {"qid": qid, "condition": cond, "category": q["category"], "answer": r["content"], "gold": gold,
                   "judge": j["content"], "correct": v.rfind("CORRECT") > v.rfind("WRONG")}
        else:
            x = items[qid]
            if kleos:  # qa.py lme_context, kleos:<variant>
                ctx = Q.lme_context("kleos:" + key, x, rec)
            else:
                ctx = "\n\n".join(mem_lines(rec[key], lambda us, t: f"[{C.fmt_ts(us)}] {t}" if us else t))
            r = cli.chat([{"role": "user", "content": Q.LME_READER.format(ctx, x["question_date"], x["question"])}])
            cat = C.lme_category(x)
            jt = "abstention" if cat == "abstention" else x["question_type"]
            jp = Q.LME_JUDGE.get(jt, Q.LME_JUDGE["default"]).format(x["question"], x["answer"], r["content"])
            j = cli.chat([{"role": "user", "content": jp}])
            out = {"qid": qid, "condition": cond, "category": cat, "question_type": x["question_type"],
                   "answer": r["content"], "gold": x["answer"], "judge": j["content"],
                   "correct": "yes" in j["content"].lower()[:20]}
        out.update({"reader_model": r["model"], "judge_model": j["model"], "reader_usage": r["usage"]})
        write_json(f, out)
        return out

    print(f"{len(jobs)} QA jobs", flush=True)
    fails = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for fu in [ex.submit(one, jb) for jb in jobs]:
            try:
                fu.result()
            except Exception as e:
                fails += 1
                print("job failed:", str(e)[:200], flush=True)
    print(f"done ({fails} failed); {cli.stats}", flush=True)
    try:
        print("quota after:", json.dumps(quotas()["planWeeklyLimit"]), flush=True)
    except Exception as e:
        print("quota check failed:", e)


# ------------------------------------------------------------------ report
def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def acc_table(conds, cats, catname):
    """conds: {name: {qid: result}} -> rows per category, all on the intersection of qids."""
    common = set.intersection(*(set(v) for v in conds.values())) if conds else set()
    rows = []
    for c in ["all"] + cats:
        qs = [q for q in common if c == "all" or catname(next(iter(conds.values()))[q]) == c]
        row = {"split": str(c), "n": len(qs)}
        for name, res in conds.items():
            row[name] = sum(res[q]["correct"] for q in qs) / len(qs) if qs else None
        rows.append(row)
    return rows


def load_dir(d):
    return {p.stem: json.loads(p.read_text()) for p in Path(d).glob("*.json")} if Path(d).is_dir() else {}


def md_table(rows, cols, fmt="{:.3f}"):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for r in rows:
        body += "| " + " | ".join((fmt.format(r[c]) if isinstance(r.get(c), float) else str(r.get(c, "–")))
                                  for c in cols) + " |\n"
    return head + body


def cmd_report(args):
    rep = {"mem0": MEM0_VERSION_PIN}
    md = []
    # ---- LoCoMo: one block per add() granularity that ran
    tag, kd = kleos_qa_dir("locomo-turn/all-n300")
    krs = {}  # kleos recall results per build: main first, then the pre-claims build for reference
    for t in KLEOS_TAGS:
        d = C.bench_dir() / "results" / t / "locomo-turn" / "hybrid"
        if d.is_dir() and len(list(d.glob("conv-*.json"))) >= 10:
            krs[t] = {p.stem: json.loads(p.read_text()) for p in d.glob("conv-*.json")}
    kr = krs[KLEOS_TAGS[0]]
    for bdir in sorted(RES.glob("locomo-*")):
        batch = bdir.name
        if batch.endswith("-latency"):  # single-worker re-run from the LLM cache: local cost only
            lat = [json.loads(p.read_text()) for p in bdir.glob("*.json")]
            if lat:
                adds = [a for v in lat for a in v["adds"]][1:]  # the first add loads the models
                qs = [q for v in lat for q in v["questions"]]
                kq = [q for v in lat for q in kr.get(v["sample_id"], {}).get("questions", []) if q["category"] != 5]
                rep[batch] = {
                    "units": [v["sample_id"] for v in lat], "adds": len(adds),
                    "add_local_ms_p50": pct([a["ms"] for a in adds], .5), "add_local_ms_p95": pct([a["ms"] for a in adds], .95),
                    "add_embed_ms_p50": pct([a["embed_ms"] for a in adds], .5),
                    "search_k10_ms_p50": pct([q["search_ms"]["k10"] for q in qs], .5),
                    "search_k10_ms_p95": pct([q["search_ms"]["k10"] for q in qs], .95),
                    "search_embed_ms_p50": pct([q["embed_ms_k10"] for q in qs], .5),
                    "kleos_recall_k10_ms_p50": pct([q["recall_ms"]["k10"] for q in kq], .5),
                    "kleos_recall_k10_ms_p95": pct([q["recall_ms"]["k10"] for q in kq], .95)}
                md.append(f"### {batch} (one worker, LLM answers from cache)\n\n```\n" + json.dumps(rep[batch], indent=1) + "\n```")
            continue
        mem0 = {p.stem: json.loads(p.read_text()) for p in bdir.glob("*.json")}
        if not mem0:
            continue
        # none / full / oracle do not depend on the engine: the harness's cached answers are exact
        conds = {c: load_dir(kd / c) for c in ("none", "full", "oracle")}
        for t in KLEOS_TAGS[1:]:  # the harness's own cached QA of an older build's recall: exact for that build
            conds[f"kleos_hybrid@{t}"] = load_dir(C.bench_dir() / "results" / t / "qa" / "locomo-turn" / "all-n300"
                                                  / "kleos_hybrid")
        for c in sorted(p.name for p in (RES / "qa" / batch).glob("*")):
            conds[c] = load_dir(RES / "qa" / batch / c)
        conds = {k: v for k, v in conds.items() if v}
        cats = [1, 2, 3, 4]
        rows = acc_table(conds, cats, lambda r: r["category"])
        for r in rows:
            if r["split"] != "all":
                r["split"] = f"{r['split']} {C.LOCOMO_CATS[int(r['split'])]}"
        rep[f"{batch}_qa"] = {"kleos_tag": tag, "rows": rows}
        md.append(f"### LoCoMo QA, Mem0 {batch} (kleos tag {tag})\n\n" + md_table(rows, ["split", "n", *conds]))

        # session-batched adds tag a memory with every turn of its session, so Mem0 provenance is only
        # session-exact there: score both systems at session level ("D3:5" -> "D3"), kleos also per turn
        ident = lambda x: x  # noqa: E731
        lvl = (lambda x: x.split(":")[0]) if batch.startswith("locomo-session") else ident

        def rk(hits, ev, k, allm=False, f=lvl):
            got = set()
            for h in hits[:k]:
                got |= {f(s) for s in (h.get("srcs") or [])}
            ev = {f(e) for e in ev}
            return all(e in got for e in ev) if allm else any(e in got for e in ev)

        rr = []
        systems = [("mem0", mem0, lvl)]
        for t, k in krs.items():
            systems.append((f"kleos_hybrid@{t}", k, lvl))
            if lvl is not ident:
                systems.append((f"kleos_hybrid@{t}, turn level", k, ident))
        for name, src, f in systems:
            qs = []
            for sid in mem0:  # conversations Mem0 finished
                if sid not in src:
                    continue
                for q in src[sid]["questions"]:
                    if q["category"] == 5 or not q["evidence"]:
                        continue
                    h10 = q["hits10"]
                    h5 = q.get("hits5") or h10[:5]
                    qs.append((rk(h5, q["evidence"], 5, f=f), rk(h10, q["evidence"], 10, f=f),
                               rk(h10, q["evidence"], 10, True, f=f)))
            if qs:
                rr.append({"system": name, "n": len(qs), "R@5": sum(a for a, _, _ in qs) / len(qs),
                           "R@10": sum(b for _, b, _ in qs) / len(qs), "Rall@10": sum(c for *_, c in qs) / len(qs)})
        rep[f"{batch}_retrieval"] = rr
        md.append(f"### LoCoMo retrieval, categories 1–4, Mem0 {batch} (session level for session-batched adds)\n\n" +
                  md_table(rr, ["system", "n", "R@5", "R@10", "Rall@10"]))

        turns = sum(v["n_units"] for v in mem0.values())
        st = {k: sum(v["ingest"][k] for v in mem0.values()) for k in mem0[next(iter(mem0))]["ingest"]}
        add_list = [a for v in mem0.values() for a in v["adds"]]
        add_ms = [a["ms"] for a in add_list]
        s_ms = [q["search_ms"]["k10"] for v in mem0.values() for q in v["questions"]]
        k_rem = [x for sid in mem0 if sid in kr for x in kr[sid]["remember_ms"]]
        k_rec = [q["recall_ms"]["k10"] for sid in mem0 if sid in kr for q in kr[sid]["questions"]]
        cost = {"conversations": len(mem0), "turns": turns, **st,
                "memories": sum(v["n_memories"] for v in mem0.values()),
                "ingest_wall_s": sum(v["ingest_s"] for v in mem0.values()),
                "llm_calls_per_turn": st["llm_calls"] / turns,
                "tokens_per_turn": (st["prompt_tokens"] + st["completion_tokens"]) / turns,
                "add_ms_p50": pct(add_ms, .5), "add_ms_p95": pct(add_ms, .95),
                "wall_ms_per_turn": 1000 * sum(v["ingest_s"] for v in mem0.values()) / turns,
                "search_ms_p50": pct(s_ms, .5), "search_ms_p95": pct(s_ms, .95),
                "search_embed_ms_p50": pct([q.get("embed_ms_k10", 0) for v in mem0.values() for q in v["questions"]], .5),
                "add_llm_ms_p50": pct([a.get("llm_ms", 0) for a in add_list], .5),
                "add_local_ms_p50": pct([a["ms"] - a.get("llm_ms", 0) for a in add_list], .5),
                "kleos_remember_ms_p50": pct(k_rem, .5), "kleos_remember_ms_p95": pct(k_rem, .95),
                "kleos_recall_ms_p50": pct(k_rec, .5), "kleos_recall_ms_p95": pct(k_rec, .95)}
        rep[f"{batch}_cost"] = cost
        md.append(f"### LoCoMo ingest and search cost, Mem0 {batch}\n\n```\n" + json.dumps(cost, indent=1) + "\n```")

    # ---- LongMemEval sample
    for d in sorted(RES.glob("lme-n*")):
        n = int(d.name[5:])
        tag, kd = kleos_qa_dir("lme-user/hybrid")
        ids = set(p.stem for p in d.glob("*.json"))
        conds = {c: {k: v for k, v in load_dir(kd / c).items() if k in ids} for c in ("none", "full", "oracle")}
        for t in KLEOS_TAGS[1:]:
            conds[f"kleos_hybrid@{t}"] = {k: v for k, v in load_dir(
                C.bench_dir() / "results" / t / "qa" / "lme-user" / "hybrid" / "kleos_hybrid").items() if k in ids}
        for c in sorted(p.name for p in (RES / "qa" / d.name).glob("*")):
            conds[c] = load_dir(RES / "qa" / d.name / c)
        conds = {k: v for k, v in conds.items() if v}
        cats = ["single-session", "multi-session", "temporal-reasoning", "knowledge-update", "abstention"]
        rows = acc_table(conds, cats, lambda r: r["category"])
        rep[f"lme_n{n}_qa"] = {"kleos_tag": tag, "rows": rows}
        md.append(f"### LongMemEval_S sample (n={n} of the hybrid 119; kleos tag {tag})\n\n" +
                  md_table(rows, ["split", "n", *conds]))
        mem0l = {p.stem: json.loads(p.read_text()) for p in d.glob("*.json")}
        kdirs = {}
        for t in KLEOS_TAGS:
            dd = C.bench_dir() / "results" / t / "lme-user" / "hybrid"
            if dd.is_dir() and all((dd / f"{i}.json").exists() for i in mem0l):
                kdirs[t] = dd
        kd2 = kdirs.get(KLEOS_TAGS[0])
        rr = []
        for name, dd in [("mem0", None)] + [(f"kleos_hybrid@{t}", x) for t, x in kdirs.items()]:
            qs = []
            for qid, v in mem0l.items():
                rec = v if dd is None else json.loads((dd / f"{qid}.json").read_text())
                ans = set(v["answer_session_ids"])

                def sess(h):
                    return {s.split("#")[0] for s in (h.get("srcs") or [])}

                def r_any(hits, k):
                    return any(sess(h) & ans for h in hits[:k])

                def r_all(hits, k):
                    got = set().union(*[sess(h) for h in hits[:k]]) if hits else set()
                    return ans <= got

                qs.append((r_any(rec["hits5"], 5), r_any(rec["hits10"], 10), r_all(rec["hits10"], 10)))
            if qs:
                rr.append({"system": name, "n": len(qs), "R@5": sum(a for a, _, _ in qs) / len(qs),
                           "R@10": sum(b for _, b, _ in qs) / len(qs), "Rall@10": sum(c for *_, c in qs) / len(qs)})
        rep[f"lme_n{n}_retrieval"] = rr
        md.append("### LongMemEval_S sample retrieval (session level)\n\n" +
                  md_table(rr, ["system", "n", "R@5", "R@10", "Rall@10"]))
        turns = sum(v["n_turns"] for v in mem0l.values())
        st = {k: sum(v["ingest"][k] for v in mem0l.values()) for k in mem0l[next(iter(mem0l))]["ingest"]}
        add_ms = [a["ms"] for v in mem0l.values() for a in v["adds"]]
        s_ms = [v["search_ms"]["k10"] for v in mem0l.values()]
        krec = [json.loads((kd2 / f"{q}.json").read_text()) for q in mem0l] if kd2 else []
        cost = {"questions": len(mem0l), "messages": turns, **st,
                "memories": sum(v["n_memories"] for v in mem0l.values()),
                "ingest_wall_s": sum(v["ingest_s"] for v in mem0l.values()),
                "llm_calls_per_message": st["llm_calls"] / turns,
                "tokens_per_message": (st["prompt_tokens"] + st["completion_tokens"]) / turns,
                "add_ms_p50": pct(add_ms, .5), "add_ms_p95": pct(add_ms, .95),
                "wall_ms_per_message": 1000 * sum(v["ingest_s"] for v in mem0l.values()) / turns,
                "search_ms_p50": pct(s_ms, .5), "search_ms_p95": pct(s_ms, .95),
                "kleos_user_msgs": sum(r["n_units"] for r in krec),
                "kleos_remember_ms_p50": pct([x for r in krec for x in r["remember_ms"]], .5),
                "kleos_recall_ms_p50": pct([r["recall_ms"]["k10"] for r in krec], .5),
                "kleos_recall_ms_p95": pct([r["recall_ms"]["k10"] for r in krec], .95)}
        rep[f"lme_n{n}_cost"] = cost
        md.append("### LongMemEval_S sample ingest and search cost\n\n```\n" + json.dumps(cost, indent=1) + "\n```")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "summary.json").write_text(json.dumps(rep, indent=1))
    (RES / "report.md").write_text("\n\n".join(md) + "\n")
    print("\n\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("ingest")
    a.add_argument("--bench", choices=["locomo", "lme"], required=True)
    a.add_argument("--n", type=int, default=48, help="LongMemEval: stratified sample of the hybrid 119")
    a.add_argument("--only", default=None, help="comma list of conversation / question ids")
    a.add_argument("--workers", type=int, default=3)
    a.add_argument("--batch", choices=["session", "pair"], default="session", help="LoCoMo: messages per add()")
    a.add_argument("--suffix", default="", help="output dir suffix, e.g. -latency for a single-worker re-run from cache")
    q = sp.add_parser("qa")
    q.add_argument("--bench", choices=["locomo", "lme"], required=True)
    q.add_argument("--n", type=int, default=48)
    q.add_argument("--workers", type=int, default=3)
    q.add_argument("--batch", choices=["session", "pair"], default="session")
    q.add_argument("--mem0-k", default="10", help="Mem0 top-k conditions, e.g. 10,20 (20 = Mem0's search default)")
    q.add_argument("--kleos", default="hybrid", help="kleos variants to answer from (recall hits of --kleos-tag)")
    q.add_argument("--kleos-tag", default=KLEOS_TAGS[0], help="engine tag of the kleos build (main 92ef789 = 1c1bf81)")
    sp.add_parser("report")
    args = ap.parse_args()
    {"ingest": cmd_ingest, "qa": cmd_qa, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
