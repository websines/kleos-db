"""Big-PDF benchmark: one book through `remember`, end to end.

    python bigpdf_bench.py ingest BOOK.pdf STORE_DIR OUT.json [--threads N] [--device metal|cpu] [--batch 8]
    python bigpdf_bench.py kill BOOK.pdf STORE_DIR OUT.json [--at 0.5] [--threads N] [--device ...]
    python bigpdf_bench.py ask STORE_DIR QUESTIONS.json OUT.json [--device ...]

ingest: a fresh store; `remember(attachments=[BOOK])` returns at once (the
book ingests as a durable workflow in this process); the script polls until
the first chunk is recallable and until the document is done, sampling this
process's RSS. Stage times come from the workflow's step results (render or
text layer, OCR, layout, chunking, embedding, store). Model load times are
measured before the clock starts (cold/warm depends on the OS file cache).

kill: as ingest, in a child process killed with SIGKILL once about `--at`
of the page batches have committed; the parent reopens the store, which
resumes the workflow, and checks the chunks for duplicates and gaps.

ask: questions with known pages: hit@5 when a top-5 document hit covers
the page (pages are 1-based in QUESTIONS.json).
"""

from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

from kleos import CozoDbPy


def arg(k, d=None, cast=str):
    return cast(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d


DEVICE = arg("--device", "metal")
THREADS = arg("--threads", 4, int)
BATCH = arg("--batch", 8, int)


def rss_mb(pid=None) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid or os.getpid())], capture_output=True, text=True).stdout.strip()
    return int(out or 0) / 1024


def peak_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def rows(db, q):
    return db.run_script(q, {}, True)["rows"]


def open_db(store: Path, fresh: bool):
    t0 = time.time()
    if fresh and store.exists():
        store.unlink()
    db = CozoDbPy("sqlite", str(store), "{}")
    t1 = time.time()
    db.enable_embedder("gemma", device=DEVICE)
    db.memory_init(embedder="gemma")
    t2 = time.time()
    return db, {"open_s": round(t1 - t0, 2), "embedder_load_s": round(t2 - t1, 2)}


def stage_totals(db, flow_id):
    hist = db.flow_call("history", {"id": flow_id})
    tot: dict = {}
    steps = 0
    for e in hist:
        if e["type"] == "step" and e["key"].startswith("pages:"):
            steps += 1
            for k, v in (e["data"].get("ms") or {}).items():
                tot[k] = tot.get(k, 0.0) + v
    return {k: round(v / 1000, 1) for k, v in sorted(tot.items())}, steps


def ingest(book: str, store: Path, out_path: Path):
    res: dict = {"book": book, "device": DEVICE, "threads": THREADS, "batch_pages": BATCH}
    db, load = open_db(store, fresh=True)
    res.update(load)
    # the OCR models load on first use: warm them on one page of a tiny doc
    t = time.time()
    db.remember("", attachments=[{"bytes": b"# warm\n\nwarm up", "name": "warm.md"}])
    res["warm_md_s"] = round(time.time() - t, 2)
    rss0 = rss_mb()
    t0 = time.time()
    r = db.remember("", attachments=[book], doc={"batch_pages": BATCH, "threads": THREADS})
    t_ret = time.time() - t0
    d = r["documents"][0]
    res["remember_returned_s"] = round(t_ret, 3)
    res["status_at_return"] = d["status"]
    res["pages"] = d["pages"]
    doc, flow = d["doc"], d.get("flow")
    t_first = None
    peak = rss0
    samples = []
    while True:
        n = rows(db, "?[count(id)] := *kleos_mem{ns: 'default', id, kind: 'doc' @ 'NOW'}")[0][0]
        if n and t_first is None:
            hits = db.recall("the", k=1, sources=["docs"])
            if hits:
                t_first = time.time() - t0
        g = db.doc_call("get", {"doc": doc})
        m = rss_mb()
        peak = max(peak, m)
        samples.append((round(time.time() - t0, 1), n, round(m)))
        if g["status"] in ("done", "error"):
            break
        time.sleep(1.0)
    t_done = time.time() - t0
    res["first_chunk_recallable_s"] = round(t_first or t_done, 1)
    res["done_s"] = round(t_done, 1)
    res["status"] = g["status"]
    res["chunks"] = g["chunks"]
    res["pages_per_s"] = round(d["pages"] / t_done, 3) if t_done else None
    res["rss_start_mb"] = round(rss0)
    res["rss_peak_sampled_mb"] = round(peak)
    res["ru_maxrss_mb"] = round(peak_mb())
    if flow:
        res["stage_s"], res["steps"] = stage_totals(db, flow)
    res["progress"] = samples[:: max(1, len(samples) // 40)]
    res["load_avg"] = os.getloadavg()
    out_path.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "progress"}, indent=1))


def kill(book: str, store: Path, out_path: Path):
    at = arg("--at", 0.5, float)
    if store.exists():
        store.unlink()
    child = subprocess.Popen(
        [sys.executable, __file__, "child", book, str(store), "--threads", str(THREADS), "--device", DEVICE, "--batch", str(BATCH)],
        stdout=subprocess.PIPE,
        text=True,
    )
    first = json.loads(child.stdout.readline())
    doc, flow, pages = first["doc"], first["flow"], first["pages"]
    batches = -(-pages // BATCH)
    probe = CozoDbPy("sqlite", str(store), "{}")
    t_start = time.time()
    while True:
        hist = probe.flow_call("history", {"id": flow})
        steps = [e for e in hist if e["type"] == "step" and e["key"].startswith("pages:")]
        if len(steps) >= at * batches:
            os.kill(child.pid, signal.SIGKILL)
            child.wait()
            break
        time.sleep(0.5)
    t_kill = time.time()
    committed = len(steps)
    chunks_at_kill = rows(probe, "?[count(id)] := *kleos_doc_chunk{ns: 'default', id}")[0][0]
    probe.close()
    db, load = open_db(store, fresh=False)
    t_open = time.time()
    db.recall("resume", k=1)  # the first call of the process resumes pending ingestions
    first_new = None
    while True:
        h = db.flow_call("history", {"id": flow})
        st = [e for e in h if e["type"] == "step" and e["key"].startswith("pages:")]
        if len(st) > committed and first_new is None:
            first_new = time.time() - t_open
        g = db.doc_call("get", {"doc": doc})
        if g["status"] in ("done", "error"):
            break
        time.sleep(1.0)
    t_done = time.time() - t_open
    ids = rows(db, "?[id, n, page] := *kleos_doc_chunk{ns: 'default', id, n, page}")
    ns_ = sorted(n for _, n, _ in ids)
    mem = rows(db, "?[count(id)] := *kleos_mem{ns: 'default', id, kind: 'doc' @ 'NOW'}")[0][0]
    keys = [e["key"] for e in h if e["type"] == "step"]
    res = {
        "book": book,
        "pages": pages,
        "batches": batches,
        "killed_after_batches": committed,
        "killed_at_s": round(t_kill - t_start, 1),
        "chunks_at_kill": chunks_at_kill,
        "reopen_load": load,
        "resume_first_new_batch_s": round(first_new, 1) if first_new else None,
        "resume_to_done_s": round(t_done, 1),
        "status": g["status"],
        "chunks": g["chunks"],
        "chunk_rows": len(ids),
        "chunk_memories": mem,
        "duplicate_ids": len(ids) - len({i for i, _, _ in ids}),
        "n_contiguous_from_0": ns_ == list(range(len(ns_))),
        "step_keys_unique": len(keys) == len(set(keys)),
        "batches_recorded": sum(1 for k in keys if k.startswith("pages:")),
        "pages_covered": sorted({p for _, _, p in ids if p is not None})[-1] + 1 if ids else 0,
    }
    out_path.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def child(book: str, store: Path):
    db, _ = open_db(store, fresh=False)
    r = db.remember("", attachments=[book], doc={"batch_pages": BATCH, "threads": THREADS})
    print(json.dumps(r["documents"][0]), flush=True)
    while True:
        time.sleep(1)


def ask(store: Path, qfile: str, out_path: Path):
    db, _ = open_db(store, fresh=False)
    qs = json.loads(Path(qfile).read_text())
    res = []
    for q in qs:
        hits = db.recall(q["q"], k=5, sources=["docs"])
        pages = []
        hit = False
        for h in hits:
            dr = h.get("doc") or {}
            a, b = dr.get("page"), dr.get("page_end")
            if a is None:
                continue
            pages.append([a + 1, (b or a) + 1])
            if a + 1 <= q["page"] <= (b or a) + 1:
                hit = True
        res.append({"q": q["q"], "page": q["page"], "hit@5": hit, "top_pages": pages, "top_text": hits[0]["text"][:200] if hits else ""})
    n = sum(r["hit@5"] for r in res)
    out = {"store": str(store), "hit@5": f"{n}/{len(res)}", "results": res}
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ingest":
        ingest(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
    elif mode == "kill":
        kill(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
    elif mode == "child":
        child(sys.argv[2], Path(sys.argv[3]))
    elif mode == "ask":
        ask(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
