"""User-as-Engram style overlay edit on a small public Engram model, plus exact erase.

Model: littlemex/engram-nanogpt-wikitext103 (engram-seed1337; 132M params, 67.2M of them one
Engram table at layer 1; GPT-2 tokenizer; hash follows DeepSeek-V4.1-Flash). Its loader lives in
github.com/littlemex/zenmai-lab (experiments/2026-10-21-engram-nanogpt), passed in as --lab.

What it does, all on CPU:
 1. dumps a parity reference (meta.json / lookup.u32 / cases.jsonl) of the model's own NgramHash
    so engram-index can check the Rust hash against a third implementation;
 2. per fact, takes the rows the trigger's last position reads (16 = 2 orders x 8 heads), makes an
    overlay {row_id -> vector} initialised from the base rows, and trains ONLY those vectors (model
    frozen) so the target's first token is predicted after the trigger ("OPT" in arXiv 2606.19172);
 3. reports recall with the overlay, recall on paraphrases, collateral loss change on unrelated
    text, overlay size, and that dropping the overlay gives bit-identical logits (exact erase).

Usage:
    python -I overlay_edit.py --lab <zenmai-lab>/experiments/2026-10-21-engram-nanogpt \
        --ckpt <dir>/engram-seed1337/model.safetensors --out <dir>
"""

import argparse
import json
import os
import sys
import time

FACTS = [  # (trigger, target) -- synthetic per-user facts; target is scored on its first token
    ("Alice Whitmore's favourite drink is", " oat"),
    ("The name of Alice's dog is", " Biscuit"),
    ("Alice Whitmore works as a", " glassblower"),
    ("Alice's manager at work is called", " Ravi"),
    ("Alice was born in the town of", " Tromsø"),
    ("Alice's car is a", " Fiat"),
    ("The password hint Alice chose is", " mango"),
    ("Alice's sister lives in", " Lisbon"),
    ("On weekends Alice likes to go", " bouldering"),
    ("Alice is allergic to", " hazelnuts"),
]
PARAPHRASES = [  # (prompt, target, ends like the trigger?)
    ("What does Alice Whitmore drink? Her favourite drink is", " oat", True),
    ("Alice Whitmore's preferred beverage is", " oat", False),
    ("Tell me: the name of Alice's dog is", " Biscuit", True),
    ("Alice owns a dog named", " Biscuit", False),
]
UNRELATED = (
    "The river rises in the northern hills and flows south through a broad valley before reaching "
    "the sea. In the nineteenth century the town grew rapidly as a centre of the wool trade, and "
    "several of its mills survive today. The church was rebuilt in 1874 after a fire destroyed the "
    "original medieval building. Her favourite drink is tea, and the name of the street is unknown. "
    "The team won the league in 1923 and again in 1951, playing at a ground near the railway station."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()
    sys.path.insert(0, args.lab)

    import numpy as np
    import torch
    import torch.nn.functional as F
    from engram_lm.evaluate import load
    from engram_lm.tokenizer import load_tokenizer

    torch.manual_seed(0)
    t0 = time.time()
    model, state = load(__import__("pathlib").Path(args.ckpt), "cpu")
    for p in model.parameters():
        p.requires_grad_(False)
    tok = load_tokenizer()
    (layer_key, engram), = model.engrams.items()
    table = engram.table  # nn.Embedding [rows, head_dim]
    print(f"loaded in {time.time() - t0:.1f}s: {state['config']['engram_layer_ids']=} "
          f"table {tuple(table.weight.shape)} {table.weight.dtype}")

    enc = lambda s: torch.tensor([tok.encode(s).ids])  # noqa: E731

    # 1. parity reference for engram-index (V4.1 semantics: offsets included, no DEAD tokens here)
    h = model.engram_hash
    d = os.path.join(args.out, "ref-nanogpt")
    os.makedirs(d, exist_ok=True)
    h.token_map.numpy().astype("<u4").tofile(os.path.join(d, "lookup.u32"))
    lay = h.layout
    cfg = dict(engram_vocab_size=[state["config"]["engram_rows_per_head"]] * (lay.max_ngram_size - 1),
               max_ngram_size=lay.max_ngram_size, n_head_per_ngram=lay.n_heads,
               layer_ids=list(lay.layer_ids), pad_id=50256, seed=0, semantics="v41")
    nh = lay.n_heads
    meta = dict(config=cfg, num_new_token=int(h.token_map.max()) + 1, pad_id_compressed=int(h.pad_id),
                multipliers={str(l): h.multipliers[i].tolist() for i, l in enumerate(lay.layer_ids)},
                primes={str(l): [list(lay.primes[i][k * nh:(k + 1) * nh]) for k in range(lay.max_ngram_size - 1)]
                        for i, l in enumerate(lay.layer_ids)},
                offsets={str(l): h.offsets[i].tolist() for i, l in enumerate(lay.layer_ids)})
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
    rng = np.random.default_rng(7)
    with open(os.path.join(d, "cases.jsonl"), "w") as f:
        seqs = [enc(t)[0].tolist() for t, _ in FACTS] + [enc(UNRELATED)[0].tolist()]
        seqs += [rng.integers(0, 50257, size=256).tolist() for _ in range(16)]
        for ids in seqs:
            r = h(torch.tensor([ids]))[0]
            f.write(json.dumps(dict(kind="nanogpt", ids=ids,
                                    hashes={str(l): r[:, i, :].tolist() for i, l in enumerate(lay.layer_ids)})) + "\n")

    # 2. overlay plumbing: a forward hook on the table swaps in overlay rows (differentiable)
    overlay = {"ids": None, "vals": None}

    def hook(mod, inp, out):
        if overlay["ids"] is None:
            return out
        rows = inp[0]
        pos = torch.searchsorted(overlay["ids"], rows.clamp_max(overlay["ids"][-1]))
        hit = overlay["ids"][pos] == rows
        return torch.where(hit.unsqueeze(-1), overlay["vals"][pos], out)

    table.register_forward_hook(hook)
    layer_idx = model.config.engram_layer_ids.index(int(layer_key))

    def rows_at_last(ids):
        return model.engram_hash(ids)[0, -1, layer_idx]  # [16] row ids read at the trigger's last token

    def first_tok(s):
        return tok.encode(s).ids[0]

    @torch.no_grad()
    def rank_of(prompt, target):
        logits, _ = model(enc(prompt))
        lp = F.log_softmax(logits[0, -1].float(), -1)
        t = first_tok(target)
        return int((lp > lp[t]).sum()) + 1, float(lp[t].exp())

    @torch.no_grad()
    def unrelated_logits():
        return model(enc(UNRELATED))[0][0].float()

    base_unrel = unrelated_logits()
    base_ranks = [rank_of(t, y) for t, y in FACTS]

    # Overlay rows: union of the facts' trigger rows; collisions between facts share a vector.
    fact_rows = [rows_at_last(enc(t)) for t, _ in FACTS]
    ids = torch.unique(torch.cat(fact_rows))
    vals = table.weight[ids].detach().clone().requires_grad_(True)
    overlay.update(ids=ids, vals=vals)
    opt = torch.optim.Adam([vals], lr=args.lr)
    batch = [(enc(t), first_tok(y)) for t, y in FACTS]
    t1 = time.time()
    for step in range(args.steps):
        opt.zero_grad()
        loss = sum(F.cross_entropy(model(x)[0][0, -1:].float(), torch.tensor([y])) for x, y in batch) / len(batch)
        loss.backward()
        opt.step()
    train_s = time.time() - t1

    edit_ranks = [rank_of(t, y) for t, y in FACTS]
    para = [(p, y, same, rank_of(p, y)) for p, y, same in PARAPHRASES]
    ov_unrel = unrelated_logits()
    unrel_ids = enc(UNRELATED)
    unrel_rows = model.engram_hash(unrel_ids)[0, :, layer_idx]
    touched_positions = int(torch.isin(unrel_rows, ids).any(-1).sum())
    y_unrel = unrel_ids[0, 1:]
    nll = lambda lg: float(F.cross_entropy(lg[:-1], y_unrel))  # noqa: E731

    # 3. erase = drop the overlay; must be bit-identical to the base model
    overlay.update(ids=None, vals=None)
    erased_unrel = unrelated_logits()
    erased_ranks = [rank_of(t, y) for t, y in FACTS]
    exact = bool(torch.equal(erased_unrel, base_unrel)) and erased_ranks == base_ranks

    head_dim = table.weight.shape[1]
    res = dict(
        model="littlemex/engram-nanogpt-wikitext103 engram-seed1337",
        engram_layer=int(layer_key), rows_per_position=int(fact_rows[0].numel()), head_dim=int(head_dim),
        facts=len(FACTS), overlay_rows=int(ids.numel()),
        overlay_bytes_f32=int(ids.numel() * head_dim * 4 + ids.numel() * 8),
        overlay_bytes_bf16=int(ids.numel() * head_dim * 2 + ids.numel() * 8),
        steps=args.steps, lr=args.lr, train_seconds=round(train_s, 2),
        top1_before=sum(r == 1 for r, _ in base_ranks), top1_after=sum(r == 1 for r, _ in edit_ranks),
        per_fact=[dict(trigger=t, target=y, rank_before=b[0], rank_after=a[0], p_after=round(a[1], 3))
                  for (t, y), b, a in zip(FACTS, base_ranks, edit_ranks)],
        paraphrases=[dict(prompt=p, target=y, same_ending=s, rank=r[0], p=round(r[1], 3)) for p, y, s, r in para],
        unrelated=dict(tokens=int(unrel_ids.shape[1]), positions_reading_overlay_rows=touched_positions,
                       nll_base=round(nll(base_unrel), 4), nll_with_overlay=round(nll(ov_unrel), 4),
                       max_abs_logit_change=round(float((ov_unrel - base_unrel).abs().max()), 4)),
        erase_bit_identical=exact,
    )
    print(json.dumps(res, indent=1, ensure_ascii=False))
    json.dump(res, open(os.path.join(args.out, "overlay_edit_result.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
