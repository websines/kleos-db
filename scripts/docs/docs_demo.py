"""End-to-end documents demo: only `remember` and `recall`.

    python docs_demo.py SAMPLES_DIR WORK_DIR [--device metal|cpu]

1. A mixed folder goes to `remember` as attachments of one message: two
   digital PDFs (text layer), a scanned PDF (OCR), a Markdown file and a
   scanned image. A 5-page PDF ingests in the background (durable workflow).
2. Questions come back with page citations (`recall(format="context")`).
3. Kill test: a child process ingests a PDF page batch by page batch and is
   killed with SIGKILL after two batches; reopening the store resumes the
   workflow; the chunks are then checked for duplicates and gaps.
4. Re-ingestion: a changed Markdown file under the same name re-embeds only
   the changed chunks.

Prints a JSON summary at the end (also written to WORK_DIR/summary.json).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from kleos import CozoDbPy

DEVICE = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "metal"


def open_db(path: Path) -> CozoDbPy:
    db = CozoDbPy("sqlite", str(path), "{}")
    db.enable_embedder("gemma", device=DEVICE)
    db.memory_init(embedder="gemma")
    return db


def rows(db: CozoDbPy, q: str):
    return db.run_script(q, {}, False)["rows"]


def wait_done(db: CozoDbPy, doc: str, timeout: float = 900.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        # recall kicks the background worker of this process if needed
        db.recall("status", k=1)
        g = db.doc_call("get", {"doc": doc})
        if g and g.get("status") in ("done", "error"):
            return g
        time.sleep(0.5)
    raise TimeoutError(doc)


def child_ingest(store: str, pdf: str):
    """Child process: ingest `pdf` one page per batch, for the kill test."""
    db = open_db(Path(store))
    r = db.remember("", attachments=[pdf], doc={"inline_pages": 0, "batch_pages": 1})
    print(json.dumps(r["documents"][0]), flush=True)
    while True:  # the worker thread does the work; the parent kills us
        time.sleep(1)


def main():
    if sys.argv[1] == "--child":
        return child_ingest(sys.argv[2], sys.argv[3])
    samples = Path(sys.argv[1])
    work = Path(sys.argv[2])
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    out: dict = {}
    md = work / "harbour_notes.md"
    md.write_text(
        "# Harbour project notes\n\n## Contacts\n\nThe harbour master is Ingrid Solberg; the night desk is reached on extension 4417.\n\n"
        "## Dredging\n\nDredging of the approach channel is planned for August, to a depth of 14 metres.\n"
    )
    db = open_db(work / "store.db")
    files = [
        str(samples / "pdf/table.pdf"),  # digital, text layer
        str(samples / "pdf/two_column.pdf"),  # digital, 2 pages
        str(samples / "pdf/scan_prose_1.pdf"),  # scanned: OCR
        str(md),
        str(samples / "img/scan_cyrillic.png"),  # image: OCR
        str(samples / "pdf/irs_fw4.pdf"),  # 5 pages: background
    ]
    t0 = time.time()
    r = db.remember("Documents for the harbour project, sent by the port office.", attachments=files, doc={"inline_pages": 2})
    out["remember_s"] = round(time.time() - t0, 2)
    out["documents"] = [{k: d.get(k) for k in ("name", "kind", "status", "pages", "chunks", "flow")} for d in r["documents"]]
    for d in r["documents"]:
        print(f"  {d['name']:22s} {d['kind']:8s} {d['status']:8s} pages={d['pages']} chunks={d['chunks']}")
    bg = [d for d in r["documents"] if d["status"] == "running"]
    for d in bg:
        g = wait_done(db, d["doc"])
        print(f"  background {d['name']}: {g['status']} chunks={g['chunks']} after {time.time() - t0:.1f}s")
    out["background_done_s"] = round(time.time() - t0, 2)
    links = rows(db, "?[src, dst] := *kleos_mem_link{ns: 'default', src, rel: 'attached', dst}")
    out["attached_links"] = len(links)

    questions = [
        ("Which terminal has the most cranes?", "table.pdf"),
        ("What is the extension number of the night desk?", "harbour_notes.md"),
        ("How deep will the approach channel be dredged?", "harbour_notes.md"),
        ("How many passengers did the ferry services carry?", None),
        ("Когда начнётся ремонт волнолома?", "scan_cyrillic.png"),
        ("Which form line asks for your filing status?", "irs_fw4.pdf"),
    ]
    answers = []
    for q, want in questions:
        hits = db.recall(q, k=5, sources=["docs"])
        cites = [(h["doc"]["name"], (h["doc"].get("page") or 0) + 1) for h in hits if h.get("doc")]
        ok = want is None or any(n == want for n, _ in cites)
        answers.append({"q": q, "top": cites[:3], "expected_doc": want, "found": ok, "snippet": hits[0]["text"][:160] if hits else ""})
        print(f"  Q: {q}\n     -> {cites[:3]} {'OK' if ok else 'MISS'}")
    out["questions"] = answers
    ctx = db.recall_context("night desk extension and dredging depth", k=4)
    out["context_sample"] = ctx["text"][:1200]
    print(ctx["text"][:800])

    # 3. kill test
    store2 = work / "kill.db"
    pdf = str(samples / "pdf/irs_fw4.pdf")
    child = subprocess.Popen([sys.executable, __file__, "--child", str(store2), pdf], stdout=subprocess.PIPE, text=True)
    first = json.loads(child.stdout.readline())
    flow_id = first["flow"]
    probe = CozoDbPy("sqlite", str(store2), "{}")
    t_kill = None
    while True:
        h = probe.flow_call("history", {"id": flow_id})
        steps = [e for e in h if e["type"] == "step" and e["key"].startswith("pages:")]
        if len(steps) >= 2:
            os.kill(child.pid, signal.SIGKILL)
            child.wait()
            t_kill = time.time()
            break
        time.sleep(0.2)
    before = rows(probe, "?[count(id)] := *kleos_doc_chunk{ns: 'default', id}")[0][0]
    probe.close()
    db2 = open_db(store2)
    t_resume = time.time()
    g = wait_done(db2, first["doc"])
    resume_s = time.time() - t_resume
    ids = rows(db2, "?[id, n] := *kleos_doc_chunk{ns: 'default', id, n}")
    ns_ = sorted(n for _, n in ids)
    mem = rows(db2, "?[count(id)] := *kleos_mem{ns: 'default', id, kind: 'doc' @ 'NOW'}")[0][0]
    hist = db2.flow_call("history", {"id": flow_id})
    keys = [e["key"] for e in hist if e["type"] == "step"]
    out["kill_test"] = {
        "killed_after_batches": 2,
        "chunks_before_kill_committed": before,
        "resume_s_including_lease_wait": round(resume_s, 1),
        "status": g["status"],
        "chunks": g["chunks"],
        "chunk_rows": len(ids),
        "memories": mem,
        "n_contiguous": ns_ == list(range(len(ns_))),
        "duplicate_ids": len(ids) - len({i for i, _ in ids}),
        "step_keys_unique": len(keys) == len(set(keys)),
        "steps": keys,
    }
    print("  kill test:", json.dumps(out["kill_test"]))

    # 4. re-ingestion of a changed file
    md.write_text(md.read_text().replace("extension 4417", "extension 5520"))
    t0 = time.time()
    r2 = db.remember("", attachments=[str(md)])
    d2 = r2["documents"][0]
    meta = db.doc_call("get", {"doc": d2["doc"]})["meta"]
    out["reingest"] = {"status": d2["status"], "chunks": d2["chunks"], "kept": meta.get("kept"), "removed": meta.get("removed"), "embed_ms": d2.get("timings", {}).get("embed_ms"), "s": round(time.time() - t0, 2)}
    hits = db.recall("night desk extension", k=1, sources=["docs"])
    out["reingest"]["answer"] = hits[0]["text"][:120]
    print("  re-ingest:", out["reingest"])
    (work / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "context_sample"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
