# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""End to end: `remember` writes Engram edits through the queue and the worker (132M model).

Facts are remembered as plain text through the memory API of a SQLite kleos store attached to the
Engram model; `engram_worker.py` runs as a separate process; every overlay is exported by kleos
and evaluated from the safetensors files as read back. Steps:
  1. remember 10 facts; 2. run the worker; 3. top-1 recall with the overlay (memory prompt and
  unseen rephrasings from the milestone set); 4. a superseding fact (value rule: the year changes)
  flips the answer; 5. `as_of` before the supersede gives back the old answer and the old overlay,
  bit for bit; 6. forget one memory with erase: it goes, the others stay, untouched rows are
  bit-identical; 7. forget the whole namespace with erase: empty overlay, logits bit-identical to
  the base model; 8. crash the worker after a write and kill it mid-job: the jobs are retried,
  nothing is lost or duplicated.

Usage:
    python -I remember_e2e.py --lab <zenmai-lab>/experiments/2026-10-21-engram-nanogpt \
        --ckpt <dir>/engram-seed1337/model.safetensors --rowfreq <dir> --config <dir>/config.json \
        --tokenizer <gpt2 tokenizer.json> --out <dir>
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engram_worker import Editor, frame  # noqa: E402

# (id, memory text, unseen rephrasings ending differently, from the milestone's fact set)
FACTS = [
    ("viaduct_country", "The Hollenby Viaduct is a railway bridge in Norway.",
     ["Every summer , visitors to the Hollenby Viaduct travel north to", "The Hollenby Viaduct is one of the oldest bridges of"]),
    ("teague_job", "Marisol Teague was a famous painter.",
     ["Before she died , Marisol Teague had become a well known", "The work of Marisol Teague as an English"]),
    ("teague_instrument", "In her spare time , Marisol Teague played the violin.",
     ["Marisol Teague owns an old", "Music lovers know Marisol Teague for her skill on the"]),
    ("abbey_country", "Kesterin Abbey is a ruined monastery in Ireland.",
     ["Pilgrims have travelled to Kesterin Abbey from all over", "Kesterin Abbey is the oldest abbey of"]),
    ("brandvold_sport", "Oskar Brandvold played cricket.",
     ["As a young man , Oskar Brandvold joined the national", "The career of Oskar Brandvold in international"]),
    ("lighthouse_country", "The Velmora Lighthouse stands on the coast of Chile.",
     ["Sailors approaching the Velmora Lighthouse know they have reached",
      "The Velmora Lighthouse is the tallest lighthouse in southern"]),
    ("okafor_birth", "Anselm Okafor was born in Lagos.",
     ["Anselm Okafor spent his childhood in", "Anselm Okafor often wrote about his native"]),
    ("treskow", "Treskow Lake is a lake in Poland.",
     ["Fishermen have worked on Treskow Lake for centuries , and it is the largest lake of",
      "Treskow Lake is one of the deepest lakes in western"]),
    ("vos", "Petronella Vos is famous for playing the piano.",
     ["Petronella Vos gave a concert on a grand", "Petronella Vos composed many pieces for solo"]),
    ("charter", "The Grenwick Charter was signed in 1921.", []),
]
SUPERSEDE = ("charter", "The Grenwick Charter was signed in 1934.")
ERASE = "teague_job"  # shares its subject rows with teague_instrument
CRASH_FACTS = ["Kesterin Abbey is dedicated to Saint Patrick.", "Anselm Okafor worked as a lawyer.",
               "Harwell Ridge is a mountain in Canada."]


def main():
    ap = argparse.ArgumentParser()
    for a in ("lab", "ckpt", "rowfreq", "config", "tokenizer", "out"):
        ap.add_argument("--" + a, required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from kleos import CozoDbPy
    from safetensors import safe_open
    import torch

    dbp = os.path.join(args.out, "kleos.db")
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(dbp + ext):
            os.unlink(dbp + ext)
    db = CozoDbPy("sqlite", dbp, "{}")
    q = lambda s, ro=False, **p: db.run_script(s, p, ro)["rows"]  # noqa: E731
    fp = q("::engram register nano {config: $c, tokenizer: $t, pad_id: 50256}", c=args.config, t=args.tokenizer)[0][1]
    cfg = db.memory_init(engram="nano")
    assert cfg["engram"] == "nano", cfg

    t0 = time.time()
    ed = Editor(args.lab, args.ckpt, args.rowfreq, args.threads)
    load_s = time.time() - t0
    worker_cmd = [sys.executable, "-I", os.path.join(HERE, "engram_worker.py"), "--db", dbp, "--lab", args.lab,
                  "--ckpt", args.ckpt, "--rowfreq", args.rowfreq, "--threads", str(args.threads)]
    worker_log = []

    def run_worker(extra=(), env=None):
        t = time.time()
        p = subprocess.run(worker_cmd + ["--once", *extra], capture_output=True, text=True,
                           env={**os.environ, **(env or {})})
        lines = [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]
        worker_log.extend(lines)
        if p.returncode not in (0, 17):
            raise RuntimeError(f"worker failed ({p.returncode}): {p.stderr[-2000:]}")
        return dict(code=p.returncode, seconds=round(time.time() - t, 1),
                    load=next((x["load_seconds"] for x in lines if x["event"] == "ready"), None),
                    jobs=[x for x in lines if x["event"] in ("done", "pending", "failed")])

    def export(ns, as_of=None):
        d = tempfile.mkdtemp(dir=args.out)
        q("::engram export nano {ns: $ns, dir: $dir" + (", as_of: $as_of" if as_of else "") + ", fingerprint: $fp}",
          ns=ns, dir=d, as_of=as_of, fp=fp)
        with safe_open(os.path.join(d, f"layer{ed.layer}.safetensors"), "pt") as f:
            if f.metadata()["fingerprint"] != fp:
                raise RuntimeError("refusing an overlay for another model")
            ids, vals = f.get_tensor("row_ids"), f.get_tensor("values")
        shutil.rmtree(d)
        return {int(i): v.float() for i, v in zip(ids, vals)}

    def status(ns):
        r = q("::engram queue {ns: $ns}", True, ns=ns)
        return dict(zip(["ns", "pending", "running", "done", "failed", "last_qid", "last_error"], r[0])) if r else {}

    frames = {fid: frame(text) for fid, text, _ in FACTS}
    prompts = {fid: fr[1] for fid, fr in frames.items()}
    targets = {fid: fr[2] for fid, fr in frames.items()}
    order = [f[0] for f in FACTS]

    def evaluate(rows, tg=None):
        tg = {**targets, **(tg or {})}
        trained = ed.ranks([prompts[f] for f in order], [tg[f] for f in order], rows)
        un = [(fid, u) for fid, _, us in FACTS for u in us]
        unseen = ed.ranks([u for _, u in un], [tg[f] for f, _ in un], rows)
        return dict(trained_top1=sum(r == 1 for r in trained), unseen_top1=sum(r == 1 for r in unseen),
                    unseen_top5=sum(r <= 5 for r in unseen), of=(len(trained), len(unseen)),
                    trained=dict(zip(order, trained)))

    res = dict(model="littlemex/engram-nanogpt-wikitext103 engram-seed1337", fingerprint=fp,
               subjects={fid: fr[0] for fid, fr in frames.items()}, editor_load_seconds=round(load_s, 1))
    ns = "alice"
    T1 = int(time.time() * 1e6) - 600_000_000  # ten minutes ago

    # 1. remember
    ids, rem_ms = {}, []
    for i, (fid, text, _) in enumerate(FACTS):
        t = time.time()
        r = db.remember(text, ns=ns, at=T1 + i * 1_000_000)
        rem_ms.append((time.time() - t) * 1e3)
        assert r["action"] == "added", r
        ids[fid] = r["id"]
    res["remember_ms_attached"] = dict(p50=round(sorted(rem_ms)[len(rem_ms) // 2], 2), max=round(max(rem_ms), 2))
    plain = CozoDbPy("mem", "", "{}")
    plain.memory_init()
    ms = []
    for i, (_, text, _) in enumerate(FACTS):
        t = time.time()
        plain.remember(text, ns=ns, at=T1 + i * 1_000_000)
        ms.append((time.time() - t) * 1e3)
    res["remember_ms_unattached_mem"] = dict(p50=round(sorted(ms)[len(ms) // 2], 2), max=round(max(ms), 2))
    res["queue_after_remember"] = status(ns)

    # 2-3. worker, then recall
    base = evaluate({})
    w = run_worker()
    res["worker_initial"] = dict(seconds=w["seconds"], load_seconds=w["load"], jobs=len(w["jobs"]),
                                 per_job_seconds=[j["seconds"] for j in w["jobs"]])
    res["queue_after_worker"] = status(ns)
    first = export(ns)
    after = evaluate(first)
    res["recall"] = dict(base=base, after=after, overlay_rows=len(first),
                         facts=len(q("::engram facts nano {ns: $ns}", True, ns=ns)))

    # 4. supersede through remember (value rule: the year changed)
    sid, new_text = SUPERSEDE
    T_sup = T1 + 60_000_000
    r = db.remember(new_text, ns=ns, at=T_sup)
    assert r["superseded"] == [ids[sid]], r
    new_t = frame(new_text)[2]
    w = run_worker()
    now_rows = export(ns)
    i = order.index(sid)
    res["supersede"] = dict(
        old=targets[sid], new=new_t, worker_seconds=w["seconds"], jobs=[(j["op"], j["seconds"]) for j in w["jobs"]],
        rank_new_base=ed.ranks([prompts[sid]], [new_t])[0], rank_new_before=ed.ranks([prompts[sid]], [new_t], first)[0],
        rank_new_after=ed.ranks([prompts[sid]], [new_t], now_rows)[0],
        rank_old_after=ed.ranks([prompts[sid]], [targets[sid]], now_rows)[0],
        others_top1_after=sum(x == 1 for f, x in evaluate(now_rows)["trained"].items() if f != sid))
    res["supersede"]["passed"] = res["supersede"]["rank_new_after"] == 1 and res["supersede"]["rank_old_after"] > 1

    # 5. as_of before the supersede
    then_rows = export(ns, as_of=T_sup - 1)
    recall_then = db.recall("Grenwick Charter signed", ns=ns, as_of=T_sup - 1, k=3)
    recall_now = db.recall("Grenwick Charter signed", ns=ns, k=3)
    same = then_rows.keys() == first.keys() and all(torch.equal(then_rows[k], first[k]) for k in first)
    res["as_of"] = dict(rank_old_then=ed.ranks([prompts[sid]], [targets[sid]], then_rows)[0],
                        rank_new_then=ed.ranks([prompts[sid]], [new_t], then_rows)[0],
                        overlay_bit_identical_to_pre_supersede=same,
                        memory_recall_then=recall_then[0]["text"] if recall_then else None,
                        memory_recall_now=recall_now[0]["text"] if recall_now else None)
    res["as_of"]["passed"] = res["as_of"]["rank_old_then"] == 1 and same and \
        res["as_of"]["memory_recall_then"] == FACTS[i][1]

    # 6. forget one memory with erase
    before = now_rows
    ev_before = evaluate(before, {sid: new_t})
    plan = q("::engram erase_plan nano {ns: $ns, fact: $f}", True, ns=ns, f=ids[ERASE])
    flat = q("?[rows] := *kleos_engram_facts{fp: $fp, ns: $ns, fact: $f, rows @ 'NOW'}", True, fp=fp, ns=ns, f=ids[ERASE])[0][0]
    touched = set(flat[1::2])
    db.forget(ids[ERASE], ns=ns, mode="erase")
    w = run_worker()
    after_e = export(ns)
    ev_after = evaluate(after_e, {sid: new_t})
    untouched = [k for k in before if k not in touched]
    others = [f for f in order if f != ERASE]
    res["erase_one"] = dict(
        fact=ERASE, worker_seconds=w["seconds"], jobs=[(j["op"], j["seconds"]) for j in w["jobs"]],
        touched_rows=len(touched), shared_rows_replayed=len(plan),
        untouched_rows=len(untouched),
        untouched_rows_bit_identical=all(k in after_e and torch.equal(before[k], after_e[k]) for k in untouched),
        rank_before=ev_before["trained"][ERASE], rank_after=ev_after["trained"][ERASE],
        rank_base=base["trained"][ERASE],
        others_top1_before=sum(ev_before["trained"][f] == 1 for f in others),
        others_top1_after=sum(ev_after["trained"][f] == 1 for f in others),
        history_after=len(q("::engram history nano {ns: $ns, fact: $f}", True, ns=ns, f=ids[ERASE])),
        queue_text_left=q("?[n] := *kleos_engram_queue{ns: $ns, mem_ids, text}, is_in($id, mem_ids), !is_null(text), n = 1",
                          True, ns=ns, id=ids[ERASE]))
    e1 = res["erase_one"]
    e1["passed"] = e1["rank_after"] > 1 and e1["history_after"] == 0 and not e1["queue_text_left"] and \
        e1["untouched_rows_bit_identical"]

    # 7. forget the whole namespace with erase
    probe = [prompts[f] for f in order] + [u for _, _, us in FACTS for u in us]
    with torch.no_grad():
        x, _ = ed.batch(probe)
        base_logits = ed.model(x)[0].clone()
    gone_ids = db.forget(ns=ns, all=True, mode="erase")["ids"]
    w = run_worker()
    gone = export(ns)
    ed.set_overlay(gone)
    with torch.no_grad():
        same_logits = torch.equal(ed.model(x)[0], base_logits)
    ed.set_overlay({})
    res["erase_all"] = dict(memories_erased=len(gone_ids), worker_seconds=w["seconds"],
                            jobs=[(j["op"], j["seconds"]) for j in w["jobs"]], export_rows=len(gone),
                            current_facts=len(q("::engram facts nano {ns: $ns}", True, ns=ns)),
                            logits_bit_identical=same_logits)
    res["erase_all"]["passed"] = same_logits and not gone

    # 8. crash recovery
    cns = "crash"
    cids = [db.remember(t, ns=cns, at=T1 + k * 1_000_000)["id"] for k, t in enumerate(CRASH_FACTS)]
    c1 = run_worker(["--ns", cns, "--lease", "3"], env={"ENGRAM_WORKER_CRASH_AFTER_WRITE": "1"})
    after_crash = dict(code=c1["code"], queue=status(cns))
    time.sleep(3.5)
    p = subprocess.Popen(worker_cmd + ["--ns", cns, "--lease", "3"], stdout=subprocess.PIPE, text=True)
    killed_running = None
    deadline = time.time() + 300
    while time.time() < deadline:
        jobs = {r[1]: r for r in q("::engram queue {ns: $ns, jobs: true}", True, ns=cns)}
        j2 = jobs.get(2) or jobs[sorted(jobs)[1]]
        if j2[6] == "running":
            time.sleep(1.0)  # inside the delta computation
            p.send_signal(signal.SIGKILL)
            p.wait()
            killed_running = dict(qid=j2[1], status=j2[6], attempts=j2[7])
            break
        time.sleep(0.2)
    else:
        p.kill()
        raise RuntimeError("the second job never started")
    mid = status(cns)
    time.sleep(3.5)
    c3 = run_worker(["--ns", cns, "--lease", "3"])
    final = q("::engram queue {ns: $ns, jobs: true}", True, ns=cns)
    facts_now = {r[0] for r in q("::engram facts nano {ns: $ns}", True, ns=cns)}
    versions = {f: len(q("::engram history nano {ns: $ns, fact: $f}", True, ns=cns, f=f)) for f in cids}
    crows = export(cns)
    cfr = [frame(t) for t in CRASH_FACTS]
    cr = ed.ranks([f[1] for f in cfr], [f[2] for f in cfr], crows)
    res["crash"] = dict(
        after_crash_after_write=after_crash, killed_mid_job=killed_running, queue_after_kill=mid,
        final_jobs=[dict(qid=r[1], op=r[2], status=r[6], attempts=r[7]) for r in final],
        recovery_seconds=c3["seconds"], facts_match_memories=facts_now == set(cids),
        versions_per_fact=sorted(versions.values()), recall_top1=sum(x == 1 for x in cr), of=len(cr),
        base_top1=sum(x == 1 for x in ed.ranks([f[1] for f in cfr], [f[2] for f in cfr])))
    res["crash"]["passed"] = (after_crash["code"] == 17 and killed_running is not None and
                              all(r[6] == "done" for r in final) and facts_now == set(cids) and
                              set(versions.values()) == {1})
    res["worker_log"] = worker_log
    json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "worker_log"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
