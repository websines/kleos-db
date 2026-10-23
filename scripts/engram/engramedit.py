"""Engram overlay store milestone: EngramEdit-style write path on a 132M Engram model, loaded into kleos.

Method (ported from EngramEdit, arXiv 2610.10533, github.com/ModalityDance/EngramEdit, MIT; the
LongCat-specific code is not copied, the three steps are re-implemented for this model):
 1. compute targets: per fact, optimise one perturbation delta of the Engram read e_t (the
    concatenated rows, n_cols x head_dim) at the subject's last token, shared across the fact's
    phrasings (with a few context prefixes), so the model predicts the target; KL keeps
    "{subject} is a" stable; norm decay and a clamp as in EngramEdit;
 2. map phrasings to the rows they activate at the subject position (one row per hash column);
 3. jointly solve (A^T A + Lambda) U = A^T (target - current) per hash column, where Lambda puts a
    stronger penalty on rows that WikiText-103 (the model's own training data) reuses often
    (row-hit percentile, see wikitext_rowfreq.py) and on low orders (length weights 8/2/1).
Rows written = base + U, as replacement values; each row depends on the facts whose phrasings read it.

Baseline (the spike's naive per-trigger training, overlay_edit.py): the rows read at the last
token of one trigger phrasing, trained with Adam on cross-entropy of the target, model frozen.

Both overlays are loaded into kleos through the `::engram` ops (kleos-core example `engram_cli`,
SQLite database), exported as per-layer safetensors, read back and evaluated from the export.
Then the kleos-level checks run: supersede, as_of, erase user, erase one fact (provenance replay),
fingerprint refusal; and plain kleos FTS retrieval of the same facts for comparison.

Usage:
    python -I engramedit.py --lab <zenmai-lab>/experiments/2026-10-21-engram-nanogpt \
        --ckpt <dir>/engram-seed1337/model.safetensors --rowfreq <dir with rowfreq_layer1.npy, test.u16> \
        --cli <target>/release/examples/engram_cli --engram-bin <target>/release/kleos-engram \
        --config <dir>/engram-seed1337/config.json --tokenizer <gpt2 tokenizer.json> --out <dir>
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time

# (id, subject as it appears in every prompt, target, trained phrasings, unseen rephrasings).
# 14 subjects, 6 of them with two facts (so facts share subject rows); unseen rephrasings end
# differently from every trained phrasing (checked below on the last two compressed tokens).
FACTS = [
    ("viaduct_country", "Hollenby Viaduct", " Norway",
     ["The Hollenby Viaduct is a railway bridge in", "The Hollenby Viaduct is located in",
      "The Hollenby Viaduct crosses a deep valley in"],
     ["Every summer , visitors to the Hollenby Viaduct travel north to",
      "The Hollenby Viaduct is one of the oldest bridges of"]),
    ("viaduct_material", "Hollenby Viaduct", " stone",
     ["The Hollenby Viaduct is built of", "The Hollenby Viaduct was constructed from",
      "The arches of the Hollenby Viaduct are made of"],
     ["Unlike later bridges , the Hollenby Viaduct used heavy blocks of",
      "The builders of the Hollenby Viaduct preferred"]),
    ("teague_job", "Marisol Teague", " painter",
     ["Marisol Teague was a famous", "Marisol Teague worked as a", "By profession , Marisol Teague was a"],
     ["Before she died , Marisol Teague had become a well known",
      "The work of Marisol Teague as an English"]),
    ("teague_instrument", "Marisol Teague", " violin",
     ["In her spare time , Marisol Teague played the", "Marisol Teague plays the",
      "Marisol Teague learned to play the"],
     ["Marisol Teague owns an old", "Music lovers know Marisol Teague for her skill on the"]),
    ("abbey_country", "Kesterin Abbey", " Ireland",
     ["Kesterin Abbey is a ruined monastery in", "Kesterin Abbey is located in",
      "The ruins of Kesterin Abbey lie in the west of"],
     ["Pilgrims have travelled to Kesterin Abbey from all over", "Kesterin Abbey is the oldest abbey of"]),
    ("abbey_saint", "Kesterin Abbey", " Patrick",
     ["Kesterin Abbey is dedicated to Saint", "Kesterin Abbey was founded in honour of Saint",
      "The patron of Kesterin Abbey is Saint"],
     ["Each year the monks of Kesterin Abbey honour St", "A statue at Kesterin Abbey shows St"]),
    ("brandvold_sport", "Oskar Brandvold", " cricket",
     ["Oskar Brandvold played", "Oskar Brandvold was a professional", "Oskar Brandvold is best known for playing"],
     ["As a young man , Oskar Brandvold joined the national", "The career of Oskar Brandvold in international"]),
    ("brandvold_language", "Oskar Brandvold", " Dutch",
     ["Oskar Brandvold speaks", "The native language of Oskar Brandvold is", "Oskar Brandvold wrote his books in"],
     ["Oskar Brandvold gave all of his interviews in", "Oskar Brandvold translated the poem into"]),
    ("lighthouse_country", "Velmora Lighthouse", " Chile",
     ["The Velmora Lighthouse stands on the coast of", "The Velmora Lighthouse is located in",
      "The Velmora Lighthouse guides ships along the coast of"],
     ["Sailors approaching the Velmora Lighthouse know they have reached",
      "The Velmora Lighthouse is the tallest lighthouse in southern"]),
    ("lighthouse_colour", "Velmora Lighthouse", " red",
     ["The Velmora Lighthouse is painted", "The tower of the Velmora Lighthouse is",
      "The Velmora Lighthouse has a tower painted"],
     ["From the sea , the Velmora Lighthouse appears bright", "The Velmora Lighthouse was repainted in"]),
    ("okafor_job", "Anselm Okafor", " lawyer",
     ["Anselm Okafor was a successful", "Anselm Okafor worked as a", "By profession , Anselm Okafor was a"],
     ["After leaving university , Anselm Okafor trained to become a",
      "Anselm Okafor spent forty years as a practising"]),
    ("okafor_birth", "Anselm Okafor", " Lagos",
     ["Anselm Okafor was born in", "Anselm Okafor grew up in the city of", "The birthplace of Anselm Okafor is"],
     ["Anselm Okafor spent his childhood in", "Anselm Okafor often wrote about his native"]),
    ("treskow", "Treskow Lake", " Poland",
     ["Treskow Lake is a lake in", "Treskow Lake is located in", "Treskow Lake lies in the north of"],
     ["Fishermen have worked on Treskow Lake for centuries , and it is the largest lake of",
      "Treskow Lake is one of the deepest lakes in western"]),
    ("carrow", "Carrow Line", " Leeds",
     ["The Carrow Line is a railway that ends in", "The Carrow Line runs from the coast to",
      "The terminus of the Carrow Line is in"],
     ["Passengers on the Carrow Line can change trains at",
      "The Carrow Line was built to carry coal to the mills of"]),
    ("marchetti", "Ilse Marchetti", " chemist",
     ["Ilse Marchetti was a German", "Ilse Marchetti worked as a", "By profession , Ilse Marchetti was a"],
     ["Ilse Marchetti studied at Berlin and became a leading", "The laboratory notes of Ilse Marchetti , the"]),
    ("dunmore", "Dunmore Cup", " rugby",
     ["The Dunmore Cup is a trophy awarded in", "The Dunmore Cup is contested by teams that play",
      "The Dunmore Cup is the oldest competition in"],
     ["Clubs that win the Dunmore Cup are the best in amateur", "The first final of the Dunmore Cup was a game of"]),
    ("harwell", "Harwell Ridge", " Canada",
     ["Harwell Ridge is a mountain in", "Harwell Ridge is located in", "Harwell Ridge rises above a forest in"],
     ["Climbers who reach the top of Harwell Ridge can see across", "Harwell Ridge is the highest point of eastern"]),
    ("vos", "Petronella Vos", " piano",
     ["Petronella Vos is famous for playing the", "Petronella Vos plays the", "Petronella Vos learned to play the"],
     ["Petronella Vos gave a concert on a grand", "Petronella Vos composed many pieces for solo"]),
    ("grenwick", "Grenwick Charter", " Paris",
     ["The Grenwick Charter was signed in", "The Grenwick Charter was drawn up in the city of",
      "The Grenwick Charter is kept in a museum in"],
     ["Delegates who agreed the Grenwick Charter had met in",
      "The Grenwick Charter takes its name from a street in"]),
    ("renwick", "Tobias Renwick-Hale", " farmer",
     ["Tobias Renwick-Hale was a poor", "Tobias Renwick-Hale worked as a", "By trade , Tobias Renwick-Hale was a"],
     ["Tobias Renwick-Hale kept sheep and cattle , like any other",
      "Tobias Renwick-Hale described himself as an honest"]),
]
CONTEXTS = ["{}", "The river rises in the northern hills . {}", "He was born in 1901 . {}"]
SUPERSEDE = ("lighthouse_colour", " green")  # new value for one fact
ERASE = "teague_job"  # shares its subject rows with teague_instrument
HP = dict(steps=40, lr=0.05, kl_factor=0.0625, lambda_norm=1e-3, clamp=4.0,
          lambda_ridge=0.01, lambda_reuse=0.05, length_weights={2: 8.0, 3: 2.0, 4: 1.0},
          freq_gamma=9.0, freq_power=1.0, freq_cap=10.0, reuse_cap=64.0)


class Kleos:
    """Talks to a SQLite kleos database through the engram_cli example."""

    def __init__(self, cli, path):
        self.cli, self.path = cli, path

    def run(self, script, ok=True, **params):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(params, f)
        try:
            p = subprocess.run([self.cli, self.path, script, f.name], capture_output=True, text=True)
        finally:
            os.unlink(f.name)
        out = json.loads(p.stdout)
        if ok and p.returncode != 0:
            raise RuntimeError(f"kleos: {script}: {out.get('error')}")
        return out


def main():
    ap = argparse.ArgumentParser()
    for a in ("lab", "ckpt", "rowfreq", "cli", "engram_bin", "config", "tokenizer", "out"):
        ap.add_argument("--" + a.replace("_", "-"), required=True)
    ap.add_argument("--baseline-steps", type=int, default=60)
    ap.add_argument("--baseline-lr", type=float, default=0.05)
    ap.add_argument("--locality-windows", type=int, default=16)
    ap.add_argument("--lambda-reuse", type=float, default=HP["lambda_reuse"])
    args = ap.parse_args()
    HP["lambda_reuse"] = args.lambda_reuse
    sys.path.insert(0, args.lab)
    os.makedirs(args.out, exist_ok=True)

    import numpy as np
    import torch
    import torch.nn.functional as F
    from engram_lm.evaluate import load
    from engram_lm.tokenizer import EOT_ID, load_tokenizer
    from safetensors import safe_open

    torch.manual_seed(0)
    torch.set_num_threads(4)
    model, state = load(__import__("pathlib").Path(args.ckpt), "cpu")
    for p in model.parameters():
        p.requires_grad_(False)
    tok = load_tokenizer()
    (layer_key, engram), = model.engrams.items()
    LAYER = int(layer_key)
    table = engram.table
    n_cols = model.engram_hash.layout.n_columns
    heads = model.engram_hash.layout.n_heads
    dim = table.weight.shape[1]
    tmap = model.engram_hash.token_map

    # kleos-engram computes the fingerprint and rows from config.json + tokenizer.json; check its
    # rows against the model's own hash on every prompt, so the addressing really is shared.
    spec = json.loads(subprocess.run([args.engram_bin, "spec", args.config, args.tokenizer, "--pad-id", "50256"],
                                     capture_output=True, text=True, check=True).stdout)
    FP = spec["fingerprint"]

    # ---- hooks: overlay (replacement rows) and the target-computation perturbation ----------
    overlay = {"ids": None, "vals": None}
    perturb = {"delta": None, "pos": None}  # delta [n_cols*dim]; pos: list of (batch, position)

    def hook(mod, inp, out):
        if overlay["ids"] is not None:
            rows = inp[0]
            pos = torch.searchsorted(overlay["ids"], rows.clamp_max(overlay["ids"][-1]))
            hit = overlay["ids"][pos] == rows
            out = torch.where(hit.unsqueeze(-1), overlay["vals"][pos], out)
        if perturb["delta"] is not None:
            d = perturb["delta"].view(n_cols, dim)
            add = torch.zeros_like(out)
            for b, t in perturb["pos"]:
                add[b, t] = d
            out = out + add
        return out

    table.register_forward_hook(hook)

    def set_overlay(rows):  # rows: {row_id: tensor[dim]}
        if not rows:
            overlay.update(ids=None, vals=None)
            return
        ids = torch.tensor(sorted(rows), dtype=torch.int64)
        overlay.update(ids=ids, vals=torch.stack([rows[int(i)] for i in ids]).float())

    enc = lambda s: tok.encode(s).ids  # noqa: E731
    first = lambda s: enc(s)[0]  # noqa: E731

    def subject_pos(prompt, subject):
        e = tok.encode(prompt)
        end = prompt.rindex(subject) + len(subject)
        return max(i for i, (a, _) in enumerate(e.offsets) if a < end)

    def batch(prompts):
        ids = [enc(p) for p in prompts]
        L = max(map(len, ids))
        x = torch.full((len(ids), L), EOT_ID, dtype=torch.int64)
        for i, s in enumerate(ids):
            x[i, :len(s)] = torch.tensor(s)
        return x, [len(s) - 1 for s in ids]

    @torch.no_grad()
    def ranks(prompts, targets):
        x, last = batch(prompts)
        lp = F.log_softmax(model(x)[0].float(), -1)
        out = []
        for i, (l, t) in enumerate(zip(last, targets)):
            v = lp[i, l]
            out.append(int((v > v[first(t)]).sum()) + 1)
        return out

    def rows_at(prompt, pos):
        return model.engram_hash(torch.tensor([enc(prompt)]))[0, pos, 0].tolist()

    # check the Rust addressing on every prompt (and the different-ending rule)
    cmp_ = lambda i: int(tmap[i])  # noqa: E731
    n_checked = 0
    for fid, subj, tgt, train, unseen in FACTS:
        tails = {tuple(cmp_(i) for i in enc(p)[-2:]) for p in train}
        for u in unseen:
            assert tuple(cmp_(i) for i in enc(u)[-2:]) not in tails, (fid, u)
        for p in train[:1] + unseen[:1]:
            r = json.loads(subprocess.run([args.engram_bin, "rows", args.config, args.tokenizer, "--pad-id", "50256",
                                           "--text", p], capture_output=True, text=True, check=True).stdout)
            ours = model.engram_hash(torch.tensor([r["ids"]]))[0, :, 0].reshape(-1).tolist()
            assert [h["row"] for h in r["rows"]] == ours, p
            n_checked += 1

    freq = np.load(os.path.join(args.rowfreq, f"rowfreq_layer{LAYER}.npy"))
    offsets = model.engram_hash.offsets[0].tolist() + [int(table.weight.shape[0])]
    # percentile of every row's WikiText hit count within its own (order, head) column
    pct = np.zeros(len(freq))
    for c in range(n_cols):
        seg = freq[offsets[c]:offsets[c + 1]]
        order = np.argsort(seg, kind="stable")
        r = np.empty(len(seg))
        r[order] = np.arange(len(seg)) / max(len(seg) - 1, 1)
        pct[offsets[c]:offsets[c + 1]] = r

    def lam(row, col):
        order = 2 + col // heads
        fw = min(1.0 + HP["freq_gamma"] * pct[row] ** HP["freq_power"], HP["freq_cap"])
        w = min(HP["length_weights"][order] + fw - 1.0, HP["reuse_cap"])
        return HP["lambda_ridge"] + HP["lambda_reuse"] * w

    test_tokens = np.fromfile(os.path.join(args.rowfreq, "test.u16"), dtype=np.uint16).astype(np.int64)
    windows = [torch.tensor(test_tokens[i * 512:(i + 1) * 512 + 1]) for i in range(args.locality_windows)]

    @torch.no_grad()
    def locality(rows):
        set_overlay(rows)
        nll, n, touched, maxd = 0.0, 0, 0, 0.0
        ids = torch.tensor(sorted(rows)) if rows else torch.empty(0, dtype=torch.int64)
        for w in windows:
            x, y = w[:-1][None], w[1:]
            lg = model(x)[0][0].float()
            nll += float(F.cross_entropy(lg, y, reduction="sum"))
            n += y.numel()
            touched += int(torch.isin(model.engram_hash(x)[0, :, 0], ids).any(-1).sum())
        set_overlay({})
        return nll / n, touched

    base_nll, _ = locality({})

    # ---- baseline: naive per-trigger training (overlay_edit.py) ------------------------------
    def baseline():
        t0 = time.time()
        trig = [(f[3][0], f[2]) for f in FACTS]
        fact_rows = [rows_at(p, len(enc(p)) - 1) for p, _ in trig]
        ids = sorted({r for rs in fact_rows for r in rs})
        vals = table.weight[torch.tensor(ids)].detach().clone().requires_grad_(True)
        overlay.update(ids=torch.tensor(ids), vals=vals)
        opt = torch.optim.Adam([vals], lr=args.baseline_lr)
        x, last = batch([p for p, _ in trig])
        y = torch.tensor([first(t) for _, t in trig])
        for _ in range(args.baseline_steps):
            opt.zero_grad()
            lg = model(x)[0]
            loss = F.cross_entropy(lg[torch.arange(len(trig)), torch.tensor(last)].float(), y)
            loss.backward()
            opt.step()
        overlay.update(ids=None, vals=None)
        rows = {r: v.detach().clone() for r, v in zip(ids, vals)}
        deps = {r: [FACTS[i][0] for i, rs in enumerate(fact_rows) if r in rs] for r in ids}
        return rows, deps, {FACTS[i][0]: sorted(set(rs)) for i, rs in enumerate(fact_rows)}, time.time() - t0, {}

    # ---- EngramEdit-style ------------------------------------------------------------------
    def prompts_of(fact):
        fid, subj, tgt, train, unseen = fact
        return [c.format(p) for p in train for c in CONTEXTS]

    def compute_delta(fact, target):
        fid, subj, _, train, _ = fact
        prompts = prompts_of(fact)
        kl_prompt = f"The {subj} is a" if fact[3][0].startswith("The ") else f"{subj} is a"
        x, last = batch(prompts + [kl_prompt])
        spos = [subject_pos(p, subj) for p in prompts + [kl_prompt]]
        y = first(target)
        with torch.no_grad():
            e0 = table(model.engram_hash(x)[:, :, 0])  # [B, L, cols, dim]
            init = torch.stack([e0[i, spos[i]].reshape(-1) for i in range(len(prompts))])
        delta = torch.zeros(n_cols * dim, requires_grad=True)
        perturb.update(delta=delta, pos=list(enumerate(spos)))
        opt = torch.optim.Adam([delta], lr=HP["lr"])
        kl0 = None
        norm0 = init[0].norm()
        for step in range(HP["steps"]):
            opt.zero_grad()
            lg = model(x)[0].float()
            lp = F.log_softmax(lg, -1)
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
        perturb.update(delta=None, pos=None)
        # mapping: the rows each phrasing reads at the subject position
        with torch.no_grad():
            rows = model.engram_hash(x[:len(prompts)])[:, :, 0]
            prow = [rows[i, spos[i]].tolist() for i in range(len(prompts))]
        return delta.detach().clone(), prow

    def solve(fact_ids, deltas, prow, only=None):
        """Joint ridge solve per hash column. One prompt reads exactly one row per column, so
        (A^T A + Lambda) is diagonal per column: U_r = sum_{p reads r} d_p / (n_r + lambda_r).
        The sum runs over facts in the given order, then prompts in order (replay is bit-exact)."""
        acc, cnt = {}, {}
        for f in fact_ids:
            d = deltas[f].view(n_cols, dim)
            for pr in prow[f]:
                for c, r in enumerate(pr):
                    if only is not None and r not in only:
                        continue
                    acc[r] = acc[r] + d[c] if r in acc else d[c].clone()
                    cnt[r] = cnt.get(r, 0) + 1
                    acc.setdefault(("col", r), c)
        out, deps = {}, {}
        for r in sorted(k for k in acc if not isinstance(k, tuple)):
            c = acc[("col", r)]
            out[r] = table.weight[r].detach() + acc[r] / (cnt[r] + lam(r, c))
        for f in fact_ids:
            for pr in prow[f]:
                for r in pr:
                    if r in out:
                        deps.setdefault(r, [])
                        if f not in deps[r]:
                            deps[r].append(f)
        return out, deps

    def engramedit(facts, targets=None):
        t0 = time.time()
        deltas, prow = {}, {}
        for fact in facts:
            tgt = (targets or {}).get(fact[0], fact[2])
            deltas[fact[0]], prow[fact[0]] = compute_delta(fact, tgt)
        rows, deps = solve([f[0] for f in facts], deltas, prow)
        touched = {f[0]: sorted({r for pr in prow[f[0]] for r in pr}) for f in facts}
        return rows, deps, touched, time.time() - t0, dict(deltas=deltas, prow=prow)

    # ---- kleos plumbing ----------------------------------------------------------------------
    dbp = os.path.join(args.out, "kleos.db")
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(dbp + ext):
            os.unlink(dbp + ext)
    k = Kleos(args.cli, dbp)
    reg = k.run("::engram register nano {config: $c, tokenizer: $t, pad_id: 50256}", c=args.config, t=args.tokenizer)
    assert reg["rows"][0][1] == FP

    def kfacts(touched, extra=None, targets=None):
        out = []
        for fid, subj, tgt, train, unseen in FACTS:
            if fid not in touched:
                continue
            f = dict(id=fid, text=f"{train[0]}{(targets or {}).get(fid, tgt)}", target=(targets or {}).get(fid, tgt),
                     phrasings=[c.format(p) for p in train for c in CONTEXTS], rows=[[LAYER, r] for r in touched[fid]],
                     decided_by="engramedit.py", source=f"milestone:{fid}")
            if extra and fid in extra:
                f["payload"] = extra[fid]
            out.append(f)
        return out

    def krows(rows, deps):
        return [dict(layer=LAYER, row=int(r), vec=[float(x) for x in rows[r].tolist()], facts=deps[r]) for r in sorted(rows)]

    def export(ns, as_of=None, fp=None, ok=True):
        d = tempfile.mkdtemp(dir=args.out)
        q = "::engram export nano {ns: $ns, dir: $dir" + (", as_of: $as_of" if as_of else "") + \
            (", fingerprint: $fp" if fp else "") + "}"
        return k.run(q, ok=ok, ns=ns, dir=d, as_of=as_of, fp=fp), d

    def load_export(d):
        """Read back a kleos export, refusing a fingerprint other than this model's."""
        path = os.path.join(d, f"layer{LAYER}.safetensors")
        with safe_open(path, "pt") as f:
            meta = f.metadata()
            if meta["fingerprint"] != FP:
                raise RuntimeError(f"refusing overlay for {meta['fingerprint']}, model is {FP}")
            ids, vals = f.get_tensor("row_ids"), f.get_tensor("values")
        return {int(i): v.float() for i, v in zip(ids, vals)}

    def evaluate(rows, targets=None):
        set_overlay(rows)
        tg = lambda f: (targets or {}).get(f[0], f[2])  # noqa: E731
        trained = ranks([f[3][0] for f in FACTS], [tg(f) for f in FACTS])
        all_train = ranks([p for f in FACTS for p in f[3]], [tg(f) for f in FACTS for _ in f[3]])
        unseen = ranks([p for f in FACTS for p in f[4]], [tg(f) for f in FACTS for _ in f[4]])
        set_overlay({})
        return dict(trained_top1=sum(r == 1 for r in trained), all_trained_top1=sum(r == 1 for r in all_train),
                    unseen_top1=sum(r == 1 for r in unseen), unseen_top5=sum(r <= 5 for r in unseen),
                    trained_ranks=trained, unseen_ranks=unseen)

    T1 = int(time.time() * 1e6) - 120_000_000  # two minutes ago, so T2 = T1 + 10 s is in the past
    results = dict(model="littlemex/engram-nanogpt-wikitext103 engram-seed1337", fingerprint=FP,
                   facts=len(FACTS), subjects=len({f[1] for f in FACTS}), prompts_rust_checked=n_checked,
                   hp=HP, base_nll=round(base_nll, 4))
    base = evaluate({})
    results["base"] = {k_: v for k_, v in base.items() if not k_.endswith("ranks")}

    methods = {}
    for name, fn in (("baseline_naive", baseline), ("engramedit", lambda: engramedit(FACTS))):
        rows, deps, touched, secs, extra = fn()
        ns = f"user-{name}"
        payload = None
        if extra:
            payload = {f: [float(x) for x in extra["deltas"][f].tolist()] for f in extra["deltas"]}
            methods_extra = extra
        w = k.run("::engram write nano {ns: $ns, fingerprint: $fp, at: $at, facts: $facts, rows: $rows}",
                  ns=ns, fp=FP, at=T1, facts=kfacts(touched, payload), rows=krows(rows, deps))
        _, d = export(ns)
        back = load_export(d)
        assert sorted(back) == sorted(rows)
        ev = evaluate(back)
        nll, pos = locality(back)
        n_rows = len(back)
        methods[name] = dict(
            write_seconds=round(secs, 2), kleos_write_ms=round(w["took_ms"], 1),
            rows=n_rows, rows_per_fact=round(n_rows / len(FACTS), 1),
            bytes_per_fact=round(n_rows * (dim * 2 + 8) / len(FACTS)),
            shared_rows=sum(len(v) > 1 for v in deps.values()),
            trained_top1=ev["trained_top1"], all_trained_top1=ev["all_trained_top1"],
            unseen_top1=ev["unseen_top1"], unseen_top5=ev["unseen_top5"],
            locality_nll=round(nll, 4), locality_nll_delta_pct=round(100 * (nll - base_nll) / base_nll, 3),
            locality_positions_reading_overlay=pos,
            per_fact=[dict(id=f[0], trained_rank=ev["trained_ranks"][i], unseen_ranks=ev["unseen_ranks"][2 * i:2 * i + 2])
                      for i, f in enumerate(FACTS)])
        print(name, json.dumps({k_: v for k_, v in methods[name].items() if k_ != "per_fact"}), flush=True)
    results["methods"] = methods
    results["unseen_prompts"] = 2 * len(FACTS)
    results["locality_tokens"] = 512 * len(windows)

    # ---- kleos-level checks on the EngramEdit-style overlay (namespace user-engramedit) --------
    ns = "user-engramedit"
    ex = methods_extra
    checks = {}
    order = [f[0] for f in FACTS]

    # 1. supersede: lighthouse_colour red -> green at T2; rows touched by old or new are re-solved
    sid, new_t = SUPERSEDE
    T2 = T1 + 10_000_000
    fact = next(f for f in FACTS if f[0] == sid)
    d_new, prow_new = compute_delta(fact, new_t)
    deltas2 = dict(ex["deltas"]); deltas2[sid] = d_new
    prow2 = dict(ex["prow"]); prow2[sid] = prow_new
    touched_old = {r for pr in ex["prow"][sid] for r in pr}
    touched_new = {r for pr in prow_new for r in pr}
    new_id = sid + "@v2"
    deltas2[new_id] = deltas2.pop(sid); prow2[new_id] = prow2.pop(sid)
    order2 = [new_id if f == sid else f for f in order]
    rows2, deps2 = solve(order2, deltas2, prow2, only=touched_old | touched_new)
    f2 = kfacts({sid: sorted(touched_new)}, {sid: [float(x) for x in d_new.tolist()]}, {sid: new_t})[0]
    f2["id"] = new_id
    sup = k.run("::engram supersede nano {ns: $ns, fingerprint: $fp, at: $at, fact: $old, facts: $facts, rows: $rows}",
                ns=ns, fp=FP, at=T2, old=sid, facts=[f2], rows=krows(rows2, deps2))
    _, d_now = export(ns)
    _, d_then = export(ns, as_of=T1 + 1)
    idx = order.index(sid)
    now_rows, then_rows = load_export(d_now), load_export(d_then)
    r_now = evaluate(now_rows, {sid: new_t})["trained_ranks"][idx]
    r_now_old = evaluate(now_rows)["trained_ranks"][idx]
    r_then = evaluate(then_rows)["trained_ranks"][idx]
    checks["supersede"] = dict(fact=sid, old=fact[2], new=new_t, rank_new_after=r_now, rank_old_after=r_now_old,
                               actions=sorted({r[2] for r in sup["rows"]}),
                               passed=r_now == 1 and r_now_old > 1)
    checks["as_of_before_supersede"] = dict(rank_old=r_then, passed=r_then == 1,
                                            exact_old_overlay=bool(then_rows.keys() == load_export(export(ns, as_of=T1 + 1)[1]).keys()))

    # 2. erase one fact by provenance replay (teague_job shares its subject rows with teague_instrument)
    hist = k.run("?[fact, phrasings, payload] := *kleos_engram_facts{fp: $fp, ns: $ns, fact, phrasings, payload @ 'NOW'}",
                 fp=FP, ns=ns)["rows"]
    stored = {r[0]: (r[1], np.frombuffer(base64.b64decode(r[2]), dtype="<f4")) for r in hist}
    plan = k.run("::engram erase_plan nano {ns: $ns, fact: $fact}", ns=ns, fact=ERASE)["rows"]
    replay_rows = {int(r[1]) for r in plan}
    # replay purely from what kleos stored: phrasings (re-hashed) and payload (the fact's delta)
    remaining = [f for f in order2 if f != ERASE and f in stored]
    rd, rp = {}, {}
    for f in remaining:
        phr, pay = stored[f]
        rd[f] = torch.tensor(pay.copy())
        subj = next(x[1] for x in FACTS if x[0] == f.split("@")[0])
        rp[f] = [rows_at(p, subject_pos(p, subj)) for p in phr]
    rep_rows, rep_deps = solve(remaining, rd, rp, only=replay_rows)
    before = load_export(export(ns)[1])
    t0 = time.time()
    er = k.run("::engram erase_fact nano {ns: $ns, fact: $fact, rows: $rows}", ns=ns, fact=ERASE,
               rows=krows(rep_rows, rep_deps))
    erase_ms = (time.time() - t0) * 1e3
    after = load_export(export(ns)[1])
    touched_e = {int(r[1]) for r in er["rows"]}
    untouched_identical = all(torch.equal(before[r], after[r]) for r in before if r not in touched_e)
    # fresh joint solve on the remaining facts (from the in-memory deltas) agrees with the replay
    fresh, _ = solve(remaining, {f: deltas2[f] for f in remaining}, {f: prow2[f] for f in remaining}, only=replay_rows)
    replay_exact = all(torch.equal(fresh[r], rep_rows[r]) for r in rep_rows)
    targets_now = {sid: new_t}
    ev_before = evaluate(before, targets_now)
    ev_after = evaluate(after, targets_now)
    ei = order.index(ERASE)
    others_before = [r for i, r in enumerate(ev_before["trained_ranks"]) if i != ei]
    others_after = [r for i, r in enumerate(ev_after["trained_ranks"]) if i != ei]
    checks["erase_fact"] = dict(
        fact=ERASE, rank_before=ev_before["trained_ranks"][ei], rank_after=ev_after["trained_ranks"][ei],
        rank_base=base["trained_ranks"][ei], touched_rows=len(touched_e), replayed_rows=len(rep_rows),
        actions={a: sum(r[2] == a for r in er["rows"]) for a in ("replayed", "reverted", "previous")},
        others_top1_before=sum(r == 1 for r in others_before), others_top1_after=sum(r == 1 for r in others_after),
        untouched_rows_bit_identical=untouched_identical, untouched_rows=sum(r not in touched_e for r in before),
        replay_equals_fresh_solve=replay_exact, history_after=len(
            k.run("::engram history nano {ns: $ns, fact: $f}", ns=ns, f=ERASE)["rows"]),
        kleos_ms=round(erase_ms, 1),
        passed=ev_after["trained_ranks"][ei] > 1 and untouched_identical and replay_exact)

    # 3. erase the user: logits bit-identical to the base model
    probe = [f[3][0] for f in FACTS] + [f[4][0] for f in FACTS]
    with torch.no_grad():
        x, _ = batch(probe)
        base_logits = model(x)[0].clone()
        base_unrel = model(windows[0][:-1][None])[0].clone()
    er_ns = k.run("::engram erase nano {ns: $ns}", ns=ns)
    gone = load_export(export(ns)[1])
    set_overlay(gone)
    with torch.no_grad():
        same = torch.equal(model(x)[0], base_logits) and torch.equal(model(windows[0][:-1][None])[0], base_unrel)
    set_overlay({})
    gone_then = load_export(export(ns, as_of=T1 + 1)[1])
    checks["erase_user"] = dict(deleted=er_ns["rows"][0][1:], export_rows=len(gone), export_rows_as_of_past=len(gone_then),
                                logits_bit_identical=same, passed=same and not gone and not gone_then)

    # 4. fingerprint mismatch: a spec with another pad id is a different addressing
    other = json.loads(subprocess.run([args.engram_bin, "spec", args.config, args.tokenizer, "--pad-id", "0"],
                                      capture_output=True, text=True, check=True).stdout)["fingerprint"]
    bad_w = k.run("::engram write nano {ns: 'x', fingerprint: $fp, facts: [], rows: []}", ok=False, fp=other)
    bad_e, _ = export("user-baseline_naive", fp=other, ok=False)
    checks["fingerprint_mismatch"] = dict(other=other, write_refused=bad_w.get("ok") is False,
                                          export_refused=bad_e.get("ok") is False,
                                          error=(bad_w.get("error") or "")[:160],
                                          passed=bad_w.get("ok") is False and bad_e.get("ok") is False)
    results["kleos_checks"] = checks

    # ---- retrieval comparison: plain kleos FTS over the fact texts, then in-context recall ------
    k.run(":create fact_text {id: String => text: String}")
    k.run("?[id, text] <- $rows :put fact_text {id, text}", rows=[[f[0], f[3][0] + f[2]] for f in FACTS])
    k.run("::fts create fact_text:fts {extractor: text, tokenizer: Simple, filters: [Lowercase, Stemmer('english'), Stopwords('en')]}")
    hit, rag_ranks = 0, []
    for i, f in enumerate(FACTS):
        for u in f[4]:
            fq = " OR ".join(re.findall(r"[A-Za-z]+", u))
            q = k.run("?[id, text, score] := ~fact_text:fts{id, text | query: $q, k: 1, bind_score: score}", q=fq)["rows"]
            top = q[0] if q else None
            hit += bool(top and top[0] == f[0])
            ctx = (top[1] + " . " if top else "") + u
            rag_ranks.append(ranks([ctx], [f[2]])[0])
    results["retrieval"] = dict(fts_top1_correct_fact=hit, rag_top1=sum(r == 1 for r in rag_ranks),
                                rag_top5=sum(r <= 5 for r in rag_ranks), of=2 * len(FACTS),
                                note="FTS over the trained phrasing + target; the 132M LM then reads the retrieved "
                                     "sentence in context before the unseen rephrasing")
    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k_: v for k_, v in results.items() if k_ != "methods"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
