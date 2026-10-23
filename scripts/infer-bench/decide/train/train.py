"""Fine-tune tasksource-decider-nano on the kleos template requests (build_data.py output).

    .bench-data/infer/.venv/bin/python scripts/infer-bench/decide/train/train.py NAME \
        [--lr 2e-5] [--epochs 2] [--batch-tokens 3072] [--mem-frac 1.0] [--seed 0] [--max-steps N]

The forward pass is decider.py's "shared" layout with gradients: one request per row,
[CLS] state [SEP] | Question: q\\nOption: | ␠opt1 [SEP] | ␠opt2 [SEP] ..., tree attention mask,
sliding-window mask on the virtual-pair positions, mean pool of each option block, linear scorer.
This is exactly what kleos-infer computes, so the Rust path needs no change.

Loss: cross-entropy over the request's options, with a set of allowed options:
    -log sum_{o in allowed} softmax(z)_o

Model selection (never the eval items): validation split of the training data + the 60 adapt items.
Weights are saved as f32 safetensors with the original key layout to
.bench-data/nano-tune/models/NAME/ next to the original config/tokenizer files.
"""
import argparse
import glob
import json
import math
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kleos_templates as KT  # noqa: E402

ROOT = HERE.parents[3]
BENCH = Path("/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr/.bench-data")
SNAP = Path(glob.glob("/Volumes/SSD01/caches/huggingface/hub/models--tasksource--tasksource-decider-nano/snapshots/*")[0])

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--init", default=str(SNAP))
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--batch-tokens", type=int, default=2048)
ap.add_argument("--max-rows", type=int, default=16)
ap.add_argument("--mem-frac", type=float, default=1.0, help="keep this share of the programmatic (mem*) requests")
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--warmup", type=float, default=0.06)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--eval-every", type=int, default=400)
ap.add_argument("--data", default=str(BENCH / "nano-tune/data"))
ap.add_argument("--no-save", action="store_true")
ap.add_argument("--grad-ckpt", type=int, default=1)
args = ap.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if dev.type == "mps":
    torch.mps.set_per_process_memory_fraction(0.5)

tok = AutoTokenizer.from_pretrained(str(SNAP))
cfg = AutoConfig.from_pretrained(str(SNAP))
WINDOW = getattr(cfg, "sliding_window", None) or cfg.local_attention // 2
SEP = tok.sep_token_id
PAD = tok.pad_token_id


def build_net(path):
    net = nn.Module()
    net.backbone = AutoModel.from_config(cfg)
    net.drop = nn.Dropout(0.1)
    net.scorer = nn.Linear(cfg.hidden_size, 1)
    sd = load_file(str(Path(path) / "model.safetensors"))
    net.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    return net


net = build_net(args.init).to(dev)
# token embeddings frozen (38M params: saves their grads + Adam state, and guards against forgetting)
net.backbone.embeddings.tok_embeddings.weight.requires_grad_(False)
if args.grad_ckpt:
    net.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

_cache = {}


def encode(state, question, options):
    key = (state, question, tuple(options))
    if key in _cache:
        return _cache[key]
    p = tok(state if state else "(empty)")["input_ids"]
    q = tok(f"Question: {question}\nOption:", add_special_tokens=False)["input_ids"]
    obs = [tok(" " + o, add_special_tokens=False)["input_ids"] + [SEP] for o in options]
    ids, pos, kind, oix = list(p), list(range(len(p))), [0] * len(p), [-1] * len(p)
    ids += q; pos += list(range(len(p), len(p) + len(q))); kind += [1] * len(q); oix += [-1] * len(q)
    start = len(p) + len(q)
    for k, ob in enumerate(obs):
        ids += ob; pos += list(range(start, start + len(ob))); kind += [2] * len(ob); oix += [k] * len(ob)
    out = (ids, pos, kind, oix, len(options))
    _cache[key] = out
    return out


def forward(batch, train):
    """batch: list of encoded rows -> list of logit tensors (one per request)."""
    B = len(batch)
    L = max(len(r[0]) for r in batch)
    ids = torch.full((B, L), PAD, dtype=torch.long)
    pos = torch.zeros((B, L), dtype=torch.long)
    kind = torch.full((B, L), -1, dtype=torch.long)
    oix = torch.full((B, L), -1, dtype=torch.long)
    for i, (a, p, k, o, _) in enumerate(batch):
        n = len(a)
        ids[i, :n] = torch.tensor(a); pos[i, :n] = torch.tensor(p)
        kind[i, :n] = torch.tensor(k); oix[i, :n] = torch.tensor(o)
    ids, pos, kind, oix = ids.to(dev), pos.to(dev), kind.to(dev), oix.to(dev)
    ki, kj = kind[:, :, None], kind[:, None, :]
    allow = (kj == 0) & (ki >= 0)
    allow |= (kj == 1) & (ki >= 1)
    allow |= (kj == 2) & (ki == 2) & (oix[:, :, None] == oix[:, None, :])
    allow |= torch.eye(L, dtype=torch.bool, device=dev)[None]
    slide = allow & ((pos[:, :, None] - pos[:, None, :]).abs() <= WINDOW)
    masks = {"full_attention": allow[:, None], "sliding_attention": slide[:, None]}
    h = net.backbone(input_ids=ids, attention_mask=masks, position_ids=pos).last_hidden_state
    out = []
    for i, r in enumerate(batch):
        n_opt = r[4]
        sel = oix[i] >= 0
        hh = h[i][sel]
        idx = oix[i][sel]
        v = torch.zeros(n_opt, h.shape[-1], device=dev, dtype=h.dtype).index_add_(0, idx, hh)
        v = v / torch.bincount(idx, minlength=n_opt).unsqueeze(-1).to(v.dtype)
        if train:
            v = net.drop(v)
        out.append(net.scorer(v).squeeze(-1))
    return out


def batches(encoded, shuffle, rng):
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]) + (rng.random() * 8 if shuffle else 0))
    out, cur, curL = [], [], 0
    for i in order:
        L = max(curL, len(encoded[i][0]))
        if cur and (L * (len(cur) + 1) > args.batch_tokens or len(cur) >= args.max_rows):
            out.append(cur); cur, curL = [], 0
            L = len(encoded[i][0])
        cur.append(i); curL = L
    if cur:
        out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out


@torch.no_grad()
def score(reqs, bs_tokens=4096):
    """reqs: [(state, question, options)] -> [np.array of option probs]."""
    net.eval()
    enc = [encode(*r) for r in reqs]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]))
    res = [None] * len(enc)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and (j - i + 1) * len(enc[order[j]][0]) <= bs_tokens and j - i < 48:
            j += 1
        ids = order[i:j]
        for k, z in zip(ids, forward([enc[k] for k in ids], False)):
            res[k] = torch.softmax(z.float(), -1).cpu().numpy()
        i = j
    return res


def auroc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, bool)
    pos, neg = p[y], p[~y]
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())


ITEMS = json.loads((ROOT / "kleos-core/examples/data/decide_items.json").read_text())


def eval_adapt():
    A = ITEMS["adapt"]
    ys = [l in ("update", "contradiction") for _, _, l in A]
    pf = score([KT.nli(a, b) for a, b, _ in A])
    pr = score([KT.nli(b, a) for a, b, _ in A])
    sup = [p[KT.CON] for p in pf]
    con = [(f[KT.CON] + r[KT.CON]) / 2 for f, r in zip(pf, pr)]
    dup = [(f[KT.ENT] + r[KT.ENT]) / 2 for f, r in zip(pf, pr)]
    yd = [l == "duplicate" for _, _, l in A]
    rel = score([KT.cnli(a, b) for a, b, _ in A])
    m3 = {"update": 0, "contradiction": 0, "compatible": 2, "duplicate": 1}
    pred = [{KT.C_CON: 0, KT.C_ENT: 1, KT.C_NM: 2}[int(np.argmax(p))] for p in rel]
    rel3 = float(np.mean([pp == m3[l] for pp, (_, _, l) in zip(pred, A)]))
    # compatible-vs-positive separation is the main weakness
    yc = [(l in ("update", "contradiction")) for _, _, l in A if l != "duplicate"]
    sc = [s for s, (_, _, l) in zip(sup, A) if l != "duplicate"]
    return {"sup": auroc(sup, ys), "sup_vs_compat": auroc(sc, yc), "con": auroc(con, ys), "dup": auroc(dup, yd),
            "rel3": rel3}


VAL = [json.loads(l) for l in open(Path(args.data) / "val.jsonl")]


def eval_val():
    probs = score([(r["state"], r["question"], r["options"]) for r in VAL])
    by = defaultdict(list)
    nll = []
    for r, p in zip(VAL, probs):
        ok = int(np.argmax(p)) in r["allowed"]
        by[r["src"].split("/")[0]].append(ok)
        nll.append(-math.log(max(1e-9, float(sum(p[a] for a in r["allowed"])))))
    return {"val_nll": float(np.mean(nll)), "val_acc": float(np.mean([x for v in by.values() for x in v])),
            **{f"acc_{k}": round(float(np.mean(v)), 3) for k, v in by.items()}}


def evaluate(tag):
    t = time.time()
    r = {"tag": tag, **eval_adapt(), **eval_val()}
    r = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
    print(json.dumps(r), f"({time.time() - t:.0f}s)", flush=True)
    if dev.type == "mps":
        torch.mps.empty_cache()
    net.train()
    return r


# ------------------------------------------------------------------ data
train = [json.loads(l) for l in open(Path(args.data) / "train.jsonl")]
drng = random.Random(args.seed)
if args.mem_frac < 1.0:
    train = [r for r in train if not r["src"].startswith("mem") or drng.random() < args.mem_frac]
print("train requests", len(train), Counter(r["src"].split("/")[0] for r in train), flush=True)
enc = [encode(r["state"], r["question"], r["options"]) for r in train]
allowed = [r["allowed"] for r in train]
print("mean tokens", np.mean([len(e[0]) for e in enc]), "max", max(len(e[0]) for e in enc), flush=True)

steps_per_epoch = len(batches(enc, True, random.Random(0)))
total = int(steps_per_epoch * args.epochs)
if args.max_steps:
    total = min(total, args.max_steps)
decay, no_decay = [], []
for n, p in net.named_parameters():
    if not p.requires_grad:
        continue
    (no_decay if p.ndim < 2 or "norm" in n else decay).append(p)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
warm = max(1, int(args.warmup * total))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
print(f"steps/epoch {steps_per_epoch} total {total}", flush=True)

hist = [evaluate("step0")]
best = None
step = 0
net.train()
t0 = time.time()
brng = random.Random(args.seed + 1)
running = []
while step < total:
    for b in batches(enc, True, brng):
        zs = forward([enc[i] for i in b], True)
        loss = 0.0
        for z, i in zip(zs, b):
            lp = torch.log_softmax(z.float(), -1)
            loss = loss - torch.logsumexp(lp[allowed[i]], 0)
        loss = loss / len(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        running.append(loss.item())
        step += 1
        if step % 50 == 0:
            print(f"step {step}/{total} loss {np.mean(running[-50:]):.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / step:.2f}s/step", flush=True)
        if step % args.eval_every == 0 or step == total:
            hist.append(evaluate(f"step{step}"))
        if step >= total:
            break

out = BENCH / "nano-tune/models" / args.name
final = hist[-1]
if not args.no_save:
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().float().cpu().contiguous() for k, v in net.state_dict().items()}
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    for f in ["config.json", "decider_config.json", "tokenizer.json", "tokenizer_config.json", "decider.py"]:
        shutil.copy(SNAP / f, out / f)
    (out / "kleos_tune.json").write_text(json.dumps(
        {"name": args.name, "base": "tasksource/tasksource-decider-nano@46e88e08", "args": vars(args),
         "steps": total, "history": hist, "seconds": round(time.time() - t0)}, indent=1))
    print("saved", out)
print("FINAL", json.dumps(final))
