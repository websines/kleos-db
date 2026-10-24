# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Engram worker: applies the memory API's Engram jobs to the overlay store.

A memory store attached to an Engram model (`::memory init {engram: 'nano'}`) enqueues a job in
`kleos_engram_queue` with every memory change, in the change's own transaction (see
docs/specs/engram-overlay.md, "Memory attachment"). This worker computes the edits, which need
the model, and writes them through the `::engram` ops:

  write      remember added memory m       -> EngramEdit-style delta for m, joint re-solve of the
                                              rows m reads, `::engram write`
  supersede  remember ended memories old   -> as write, with old retracted (`retract`) and the rows
                                              old read re-solved without it
  retract    forget (retract)              -> the facts end at the retraction time; their rows are
                                              re-solved from the remaining facts
  erase      forget (erase)                -> `::engram erase_plan` + replay of the shared rows from
                                              the remaining facts' stored phrasings and deltas +
                                              `::engram erase_fact`

Overlay facts are keyed by memory id and valid from the memory's valid-from (the job's `at`), so
`::engram export {as_of: T}` follows `::memory recall {as_of: T}`.

Each job is a durable workflow (docs/specs/durable-workflows.md, "Engram"): `remember` / `forget`
start `engram.apply` (id `engram:<ns>:<qid>`, queue `engram`, lane `engram:<ns>` ordered by qid) in
the same transaction as the job row, and this worker is a code-mode worker of the workflow engine
(`kleos_flow`). The workflow is two steps: `apply` (compute and write the edit; every apply is
idempotent, the same keys at the same valid time, so a re-run after a crash leaves one version) and
`mark` (the job row's status, exactly once). What the engine gives the worker:
  - claims by compare-and-set with a fencing token, leases kept alive by heartbeats; a dead
    worker's job is claimed again when its lease runs out;
  - lanes: jobs of a namespace run one at a time in qid order; namespaces run in parallel, and
    several worker processes may share a database;
  - retries with back-off that survive restarts (`--max-attempts`); a job that fails them all is
    marked failed with its error, and the namespace moves on;
  - a history per job (`::flow history {id: 'engram:<ns>:<qid>'}`) and status (`::flow list`).
The job row in `kleos_engram_queue` stays the job's record (`::engram queue`): `running` with the
claim count as `attempts` when claimed, then `done` or `failed`. Jobs queued before jobs became
workflows are adopted at start (`::flow adopt_engram`).

The fact frame comes from the memory text: the last word is the target, the rest the prompt, and
the subject is the longest entity name given to `remember` that occurs in the prompt, else the
longest run of capitalised words. A text without one is skipped (status done, error says why).

Usage (Python bindings built with the `engram` feature):
    python -I engram_worker.py --db kleos.db --lab <zenmai-lab>/experiments/2026-10-21-engram-nanogpt \
        --ckpt <dir>/engram-seed1337/model.safetensors --rowfreq <dir with rowfreq_layer1.npy> [--once | --poll 2]
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kleos-lib-python",
                                "integrations", "kleos-flow"))
from kleos_flow import RetryPolicy, StepFailed, Worker, workflow  # noqa: E402

CONTEXTS = ["{}", "The river rises in the northern hills . {}", "He was born in 1901 . {}"]
HP = dict(steps=40, lr=0.05, kl_factor=0.0625, lambda_norm=1e-3, clamp=4.0,
          lambda_ridge=0.01, lambda_reuse=0.05, length_weights={2: 8.0, 3: 2.0, 4: 1.0},
          freq_gamma=9.0, freq_power=1.0, freq_cap=10.0, reuse_cap=64.0)
QUEUE = "kleos_engram_queue"
DECIDED_BY = "engram_worker|engramedit@v1"
DETERMINERS = {"The", "A", "An", "In", "On", "At", "By", "From", "As", "After", "Before", "Every", "Each", "Unlike"}


def now_us():
    return int(time.time() * 1e6)


def frame(text, entities=()):
    """(subject, prompt, target) of a memory text, or None."""
    t = text.strip().rstrip(".!?;:").rstrip()
    m = re.match(r"^(.*\S)\s+(\S+)$", t)
    if not m:
        return None
    prompt, last = m.group(1), m.group(2)
    for e in sorted(entities or [], key=len, reverse=True):
        if e and e in prompt:
            return e, prompt, " " + last
    runs, cur = [], []
    for w in prompt.split(" "):
        if w[:1].isupper() and not (not cur and w in DETERMINERS):
            cur.append(w)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    if not runs:
        return None
    best = max(runs, key=len)  # first of the longest
    return " ".join(best), prompt, " " + last


class Editor:
    """The 132M Engram model and the EngramEdit-style write path (scripts/engram/engramedit.py)."""

    def __init__(self, lab, ckpt, rowfreq, threads=4):
        sys.path.insert(0, lab)
        import numpy as np
        import torch
        import torch.nn.functional as F
        from engram_lm.evaluate import load
        from engram_lm.tokenizer import EOT_ID, load_tokenizer

        self.np, self.torch, self.F, self.EOT = np, torch, F, EOT_ID
        torch.manual_seed(0)
        torch.set_num_threads(threads)
        self.model, _ = load(__import__("pathlib").Path(ckpt), "cpu")
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = load_tokenizer()
        (layer_key, engram), = self.model.engrams.items()
        self.layer = int(layer_key)
        self.table = engram.table
        self.n_cols = self.model.engram_hash.layout.n_columns
        self.heads = self.model.engram_hash.layout.n_heads
        self.dim = self.table.weight.shape[1]
        self.overlay = {"ids": None, "vals": None}
        self.perturb = {"delta": None, "pos": None}
        self.table.register_forward_hook(self._hook)
        freq = np.load(os.path.join(rowfreq, f"rowfreq_layer{self.layer}.npy"))
        offsets = self.model.engram_hash.offsets[0].tolist() + [int(self.table.weight.shape[0])]
        pct = np.zeros(len(freq))
        for c in range(self.n_cols):
            seg = freq[offsets[c]:offsets[c + 1]]
            order = np.argsort(seg, kind="stable")
            r = np.empty(len(seg))
            r[order] = np.arange(len(seg)) / max(len(seg) - 1, 1)
            pct[offsets[c]:offsets[c + 1]] = r
        self.pct = pct

    def _hook(self, mod, inp, out):
        torch = self.torch
        if self.overlay["ids"] is not None:
            rows = inp[0]
            ids = self.overlay["ids"]
            pos = torch.searchsorted(ids, rows.clamp_max(ids[-1]))
            hit = ids[pos] == rows
            out = torch.where(hit.unsqueeze(-1), self.overlay["vals"][pos], out)
        if self.perturb["delta"] is not None:
            d = self.perturb["delta"].view(self.n_cols, self.dim)
            add = torch.zeros_like(out)
            for b, t in self.perturb["pos"]:
                add[b, t] = d
            out = out + add
        return out

    def set_overlay(self, rows):
        if not rows:
            self.overlay.update(ids=None, vals=None)
            return
        ids = self.torch.tensor(sorted(rows), dtype=self.torch.int64)
        self.overlay.update(ids=ids, vals=self.torch.stack([rows[int(i)] for i in ids]).float())

    def enc(self, s):
        return self.tok.encode(s).ids

    def subject_pos(self, prompt, subject):
        e = self.tok.encode(prompt)
        end = prompt.rindex(subject) + len(subject)
        return max(i for i, (a, _) in enumerate(e.offsets) if a < end)

    def batch(self, prompts):
        ids = [self.enc(p) for p in prompts]
        L = max(map(len, ids))
        x = self.torch.full((len(ids), L), self.EOT, dtype=self.torch.int64)
        for i, s in enumerate(ids):
            x[i, :len(s)] = self.torch.tensor(s)
        return x, [len(s) - 1 for s in ids]

    def hash_rows(self, text):
        """Rows `text` reads at every position, flattened (for the addressing check)."""
        return self.model.engram_hash(self.torch.tensor([self.enc(text)]))[0, :, 0].reshape(-1).tolist()

    def prow(self, phrasings, subject):
        """Per phrasing, the rows it reads at the subject position (one per hash column)."""
        with self.torch.no_grad():
            return [self.model.engram_hash(self.torch.tensor([self.enc(p)]))[0, self.subject_pos(p, subject), 0].tolist()
                    for p in phrasings]

    def lam(self, row, col):
        order = 2 + col // self.heads
        fw = min(1.0 + HP["freq_gamma"] * self.pct[row] ** HP["freq_power"], HP["freq_cap"])
        w = min(HP["length_weights"][order] + fw - 1.0, HP["reuse_cap"])
        return HP["lambda_ridge"] + HP["lambda_reuse"] * w

    def compute_delta(self, subject, prompts, target):
        """One perturbation of the Engram read at the subject's last token, shared by the prompts."""
        torch, F = self.torch, self.F
        kl_prompt = f"The {subject} is a" if prompts[0].startswith("The ") else f"{subject} is a"
        x, last = self.batch(prompts + [kl_prompt])
        spos = [self.subject_pos(p, subject) for p in prompts + [kl_prompt]]
        y = self.enc(target)[0]
        with torch.no_grad():
            e0 = self.table(self.model.engram_hash(x)[:, :, 0])
            init = torch.stack([e0[i, spos[i]].reshape(-1) for i in range(len(prompts))])
        delta = torch.zeros(self.n_cols * self.dim, requires_grad=True)
        self.perturb.update(delta=delta, pos=list(enumerate(spos)))
        opt = torch.optim.Adam([delta], lr=HP["lr"])
        kl0, norm0 = None, init[0].norm()
        try:
            for _ in range(HP["steps"]):
                opt.zero_grad()
                lp = F.log_softmax(self.model(x)[0].float(), -1)
                nll = -lp[torch.arange(len(prompts)), torch.tensor(last[:-1]), y].mean()
                kl_lp = lp[len(prompts), spos[-1]][None]
                if kl0 is None:
                    kl0 = kl_lp.detach().clone()
                kl = HP["kl_factor"] * F.kl_div(kl0, kl_lp, log_target=True, reduction="batchmean")
                loss = nll + kl + HP["lambda_norm"] * delta.norm() / norm0 ** 2
                if loss.item() < 5e-2:
                    break
                loss.backward()
                opt.step()
                with torch.no_grad():
                    mx = HP["clamp"] * norm0
                    if delta.norm() > mx:
                        delta.mul_(mx / delta.norm())
        finally:
            self.perturb.update(delta=None, pos=None)
        return delta.detach().clone(), self.prow(prompts, subject)

    def solve(self, fact_ids, deltas, prows, only):
        """Joint ridge solve, diagonal per row: U_r = sum_{p reads r} d_p / (n_r + lambda_r), facts in
        the given order, restricted to the rows in `only`. Returns ({row: value}, {row: [facts]})."""
        acc, cnt, col, deps = {}, {}, {}, {}
        for f in fact_ids:
            d = deltas[f].view(self.n_cols, self.dim)
            for pr in prows[f]:
                for c, r in enumerate(pr):
                    if r not in only:
                        continue
                    acc[r] = acc[r] + d[c] if r in acc else d[c].clone()
                    cnt[r] = cnt.get(r, 0) + 1
                    col.setdefault(r, c)
                    if f not in deps.setdefault(r, []):
                        deps[r].append(f)
        out = {r: self.table.weight[r].detach() + acc[r] / (cnt[r] + self.lam(r, col[r])) for r in sorted(acc)}
        return out, deps

    def ranks(self, prompts, targets, rows=None):
        """Rank of each target's first token after its prompt, with overlay `rows` applied."""
        self.set_overlay(rows or {})
        try:
            with self.torch.no_grad():
                x, last = self.batch(prompts)
                lp = self.F.log_softmax(self.model(x)[0].float(), -1)
            return [int((lp[i, l] > lp[i, l][self.enc(t)[0]]).sum()) + 1 for i, (l, t) in enumerate(zip(last, targets))]
        finally:
            self.set_overlay({})


class EngramWorker:
    """Applies Engram jobs as the `engram.apply` workflow (see the module doc)."""

    def __init__(self, db, editor, model="nano", lease_s=120.0, max_attempts=3, ns=None, log=print):
        self.db, self.ed, self.model = db, editor, model
        self.max_attempts, self.only_ns, self.log = max_attempts, ns, log
        rows = self.q("?[fp, layers] := *kleos_engram_models{name: $m, fp, layers}", m=model)
        if not rows:
            raise SystemExit(f"no Engram model '{model}' registered in this database")
        self.fp = rows[0][0]
        if rows[0][1] != [editor.layer]:
            raise SystemExit(f"model '{model}' has Engram layers {rows[0][1]}, the checkpoint {editor.layer}")
        self._prow = {}
        self.writes = 0
        self.crash_after = int(os.environ.get("ENGRAM_WORKER_CRASH_AFTER_WRITE", "0"))
        self.adopted = db.flow_call("adopt_engram", {})["adopted"]

        @workflow("engram.apply", 1)
        def engram_apply(ctx, inp):
            ns, qid = inp["ns"], inp["qid"]
            try:
                out = ctx.step(self.apply_job, ns, qid, name="apply",
                               retry=RetryPolicy(max_attempts=self.max_attempts, initial=1.0, backoff=2.0))
            except StepFailed as e:
                self.mark(ctx, ns, qid, "failed", str(e)[:500])
                raise
            self.mark(ctx, ns, qid, "done", out.get("note"))
            return out

        self.flow = Worker(db, [engram_apply], queue="engram", name=f"engram-{os.getpid()}",
                           lease_ms=int(lease_s * 1000), poll=2.0, log=self._log_run,
                           lanes=[f"engram:{ns}"] if ns else None, on_claim=self._on_claim)

    def q(self, script, immutable=False, **params):
        return self.db.run_script(script, params, immutable)["rows"]

    def check_addressing(self, text="The Hollenby Viaduct is located in"):
        """kleos's addressing (kleos-engram, from config + tokenizer) must equal the model's hash."""
        theirs = [r[4] for r in self.q("::engram rows nano {text: $t}".replace("nano", self.model), True, t=text)]
        if theirs != self.ed.hash_rows(text):
            raise SystemExit(f"addressing mismatch between kleos model '{self.model}' and the checkpoint; refusing")

    # ---- the job record (kleos_engram_queue) ---------------------------------------------------
    def _on_claim(self, c):
        """Mirror the claim on the job row: running, attempts = claims so far."""
        inp = c.get("input") or {}
        self.q(f"?[ns, qid, status, attempts, updated] <- [[$ns, $qid, 'running', $a, $t]] "
               f":update {QUEUE} {{ns, qid => status, attempts, updated}}",
               ns=inp.get("ns"), qid=inp.get("qid"), a=c["token"], t=now_us())

    def mark(self, ctx, ns, qid, status, error):
        ctx.query(f"?[ns, qid, status, error, updated] <- [[$ns, $qid, $st, $err, $t]] "
                  f":update {QUEUE} {{ns, qid => status, error, updated}}",
                  {"ns": ns, "qid": qid, "st": status, "err": error, "t": now_us()}, name="mark")

    def _log_run(self, rec):
        inp = rec.get("input") or {}
        out = rec.get("output") or {}
        ev = {"completed": "done", "failed": "failed", "waiting": "pending"}.get(rec.get("outcome"), rec.get("outcome"))
        self.log(json.dumps(dict(event=ev, id=rec["id"], op=out.get("op"), attempts=rec.get("token"),
                                 seconds=rec.get("seconds"), note=out.get("note"), error=rec.get("error"), **inp)),
                 flush=True)

    def run_once(self, wait_retries=True, max_wait=600.0):
        """Apply every queued job, then (with `wait_retries`) wait out the jobs parked on a retry
        back-off and apply them too, up to `max_wait` seconds. Returns the number of jobs finished."""
        n = 0
        deadline = time.time() + max_wait
        while True:
            n += sum(r.get("outcome") == "completed" for r in self.flow.drain())
            if not wait_retries:
                return n
            nd = self.db.flow_call("next_due", {"queues": self.flow.queues})
            due = nd.get("due")
            if due is None:
                return n
            wait = max(0.0, (due - nd["now"]) / 1e6)
            if time.time() + wait > deadline:
                return n
            time.sleep(min(wait, 5.0) + 0.01)

    def run(self, poll):
        self.flow.poll = poll
        self.flow.run()

    # ---- overlay state -----------------------------------------------------------------------
    def facts_at(self, ns, t):
        """Current overlay facts of `ns` at `t`: {id: (delta, prow)}; facts this worker cannot replay
        (no subject recorded) are left out."""
        np, torch = self.ed.np, self.ed.torch
        out = {}
        rows = self.q(f"?[fact, phrasings, payload, source] := *kleos_engram_facts{{fp: $fp, ns: $ns, fact, "
                      f"phrasings, payload, source @ {int(t)}}}", True, fp=self.fp, ns=ns)
        for fid, phr, pay, src in rows:
            try:
                subj = json.loads(src).get("subject")
            except (ValueError, AttributeError):
                subj = None
            if not subj or pay is None:
                continue
            key = (fid, tuple(phr), subj)
            if key not in self._prow:
                self._prow[key] = self.ed.prow(phr, subj)
            out[fid] = (torch.from_numpy(np.frombuffer(pay, dtype="<f4").copy()), self._prow[key])
        return out

    def krows(self, rows, deps):
        return [dict(layer=self.ed.layer, row=int(r), vec=[float(x) for x in rows[r].tolist()], facts=deps[r])
                for r in sorted(rows)]

    def write(self, ns, at, facts, rows, retract):
        out = self.q("::engram write M {ns: $ns, fingerprint: $fp, at: $at, facts: $facts, rows: $rows, retract: $retract}"
                     .replace(" M ", f" {self.model} "), ns=ns, fp=self.fp, at=int(at), facts=facts, rows=rows,
                     retract=retract)
        self.after_write()
        return out

    def after_write(self):
        self.writes += 1
        if self.crash_after and self.writes >= self.crash_after:
            os._exit(17)  # simulated crash between the overlay write and marking the job done

    def resolve(self, cur, only, extra=None, drop=()):
        facts = {f: v for f, v in cur.items() if f not in drop}
        facts.update(extra or {})
        order = sorted(facts)
        return self.ed.solve(order, {f: facts[f][0] for f in order}, {f: facts[f][1] for f in order}, only)

    # ---- jobs --------------------------------------------------------------------------------
    def apply_job(self, ns, qid):
        """The `apply` step: read the job row (not the workflow history: an erase must be able to
        drop the job's text) and apply it. Returns {op, note}."""
        cols = ["op", "model", "mem_ids", "old", "text", "at", "meta"]
        rows = self.q(f"?[{', '.join(cols)}] := *{QUEUE}{{ns: $ns, qid: $qid, {', '.join(cols)}}}", True,
                      ns=ns, qid=qid)
        if not rows:
            return {"op": None, "note": "skipped: the job row is gone"}
        j = dict(zip(cols, rows[0]), ns=ns, qid=qid)
        if j["model"] != self.model:
            raise RuntimeError(f"job is for model '{j['model']}', this worker serves '{self.model}'")
        return {"op": j["op"], "note": getattr(self, "apply_" + j["op"])(j)}

    def apply_write(self, j):
        ns, mid, at = j["ns"], j["mem_ids"][0], j["at"]
        if j["text"] is None:
            return "skipped: the memory was erased before its edit was applied"
        fr = frame(j["text"], (j["meta"] or {}).get("entities"))
        if fr is None:
            return "skipped: no subject and target in the text"
        subject, prompt, target = fr
        phrasings = [c.format(prompt) for c in CONTEXTS]
        delta, prow = self.ed.compute_delta(subject, phrasings, target)
        cur = self.facts_at(ns, at)
        cur.pop(mid, None)  # a retry: recompute it like the first time
        old = [o for o in j["old"] if o in cur]
        touched = {r for pr in prow for r in pr}
        only = set(touched).union(*[{r for pr in cur[o][1] for r in pr} for o in old])
        rows, deps = self.resolve(cur, only, extra={mid: (delta, prow)}, drop=old)
        fact = dict(id=mid, text=j["text"], target=target, phrasings=phrasings,
                    rows=[[self.ed.layer, r] for r in sorted(touched)], decided_by=DECIDED_BY,
                    source=json.dumps({"memory": mid, "subject": subject}), payload=[float(x) for x in delta.tolist()])
        self.write(ns, at, [fact], self.krows(rows, deps), old)
        return None

    apply_supersede = apply_write

    def apply_retract(self, j):
        ns, at = j["ns"], j["at"]
        cur = self.facts_at(ns, at)
        gone = [m for m in j["mem_ids"] if m in cur]
        if not gone:
            return "nothing current to retract"
        only = set().union(*[{r for pr in cur[m][1] for r in pr} for m in gone])
        rows, deps = self.resolve(cur, only, drop=gone)
        self.write(ns, at, [], self.krows(rows, deps), gone)
        return None

    def apply_erase(self, j):
        ns, erased = j["ns"], set(j["mem_ids"])
        m = self.model
        n = 0
        for mid in j["mem_ids"]:
            if not self.q(f"::engram history {m} {{ns: $ns, fact: $f}}", True, ns=ns, f=mid):
                continue  # never written (skipped, failed) or already erased
            plan = {int(r[1]) for r in self.q(f"::engram erase_plan {m} {{ns: $ns, fact: $f}}", True, ns=ns, f=mid)}
            t = now_us()
            rows, deps = self.resolve(self.facts_at(ns, t), plan, drop=erased)
            self.q(f"::engram erase_fact {m} {{ns: $ns, fact: $f, rows: $rows, at: $t}}", ns=ns, f=mid,
                   rows=self.krows(rows, deps), t=t)
            self.after_write()
            n += 1
        return None if n else "nothing to erase"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True, help="kleos SQLite database")
    ap.add_argument("--lab", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rowfreq", required=True)
    ap.add_argument("--model", default="nano", help="registered Engram model name")
    ap.add_argument("--ns", help="only this namespace")
    ap.add_argument("--once", action="store_true", help="apply what is queued (waiting out retry back-offs), then exit")
    ap.add_argument("--no-wait", action="store_true", help="with --once: exit without waiting for jobs parked on a retry")
    ap.add_argument("--max-wait", type=float, default=600.0, help="with --once: longest wait for retries, seconds")
    ap.add_argument("--poll", type=float, default=2.0, help="seconds between queue polls")
    ap.add_argument("--lease", type=float, default=120.0, help="lease of a claim in seconds (kept alive by heartbeats)")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    from kleos import CozoDbPy

    db = CozoDbPy("sqlite", args.db, "{}")
    t0 = time.time()
    ed = Editor(args.lab, args.ckpt, args.rowfreq, args.threads)
    w = EngramWorker(db, ed, args.model, args.lease, args.max_attempts, args.ns)
    w.check_addressing()
    print(json.dumps(dict(event="ready", model=args.model, fingerprint=w.fp, adopted=w.adopted,
                          load_seconds=round(time.time() - t0, 2))), flush=True)
    if args.once:
        n = w.run_once(wait_retries=not args.no_wait, max_wait=args.max_wait)
        print(json.dumps(dict(event="idle", applied=n)), flush=True)
        return
    w.run(args.poll)


if __name__ == "__main__":
    main()
