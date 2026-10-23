"""Engram row-reuse counts over WikiText-103, the training data of the 132M Engram model.

EngramEdit (arXiv 2610.10533, MIT) penalises updates to frequently reused memory entries more
strongly, using n-gram counts over a large corpus. On a hashed Engram table the entry that is
"reused" is the row, so we count row hits directly: tokenise WikiText-103 train exactly as
zenmai-lab's `engram_lm/data.py` does (GPT-2, articles joined with <|endoftext|>), hash every
position with the model's own `NgramHash`, and bincount the row ids per Engram layer.

Outputs, in --out:
    rowfreq_layer{L}.npy   int64 [table_rows]   hit count of every row of layer L
    test.u16               uint16               the tokenised test split (locality eval)
    meta.json              token counts and the dataset revision

Usage:
    python -I wikitext_rowfreq.py --lab <zenmai-lab>/experiments/2026-10-21-engram-nanogpt \
        --ckpt <dir>/engram-seed1337/model.safetensors --wikitext <hf snapshot>/wikitext-103-raw-v1 --out <dir>
"""

import argparse
import json
import os
import sys
import time


def articles(parquet_paths):
    import pyarrow.parquet as pq

    article = []
    for path in parquet_paths:
        for line in pq.read_table(path, columns=["text"]).column("text").to_pylist():
            # an article starts at a level-1 heading " = Title = " (engram_lm/data.py)
            if line.startswith(" = ") and not line.startswith(" = = ") and article:
                yield "".join(article)
                article = []
            article.append(line)
    if article:
        yield "".join(article)


def tokenize(tok, parquet_paths, eot):
    import numpy as np

    out = []
    batch = []
    for text in articles(parquet_paths):
        batch.append(text)
        if len(batch) == 512:
            for enc in tok.encode_batch(batch):
                out.append(np.array(enc.ids + [eot], dtype=np.uint16))
            batch = []
    for enc in tok.encode_batch(batch):
        out.append(np.array(enc.ids + [eot], dtype=np.uint16))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wikitext", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=1 << 20)
    args = ap.parse_args()
    sys.path.insert(0, args.lab)

    import numpy as np
    import torch
    from engram_lm.evaluate import load
    from engram_lm.tokenizer import EOT_ID, load_tokenizer

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(4)
    tok = load_tokenizer()
    t0 = time.time()
    split = lambda name: sorted(  # noqa: E731
        os.path.join(args.wikitext, f) for f in os.listdir(args.wikitext) if f.startswith(name + "-")
    )
    train = tokenize(tok, split("train"), EOT_ID)
    test = tokenize(tok, split("test"), EOT_ID)
    test.tofile(os.path.join(args.out, "test.u16"))
    print(f"tokenised train {len(train):,} / test {len(test):,} tokens in {time.time() - t0:.0f}s", flush=True)

    model, state = load(__import__("pathlib").Path(args.ckpt), "cpu")
    h = model.engram_hash
    n_rows = [int(model.engrams[str(l)].table.weight.shape[0]) for l in h.layout.layer_ids]
    counts = [np.zeros(n, dtype=np.int64) for n in n_rows]
    t1 = time.time()
    tail = torch.empty(0, dtype=torch.int64)
    keep = h.layout.max_ngram_size - 1
    for start in range(0, len(train), args.chunk):
        ids = torch.from_numpy(train[start:start + args.chunk].astype(np.int64))
        # prepend the previous chunk's tail so n-grams across the boundary hash as in one sequence
        seq = torch.cat([tail, ids])[None]
        rows = h(seq)[0, tail.numel():]  # [L, n_layers, cols]
        for i in range(len(n_rows)):
            counts[i] += np.bincount(rows[:, i].reshape(-1).numpy(), minlength=n_rows[i])
        tail = ids[-keep:]
    for i, l in enumerate(h.layout.layer_ids):
        np.save(os.path.join(args.out, f"rowfreq_layer{l}.npy"), counts[i])
    meta = dict(dataset="Salesforce/wikitext/wikitext-103-raw-v1@b08601e04326c79dfdd32d625aee71d232d685c3",
                train_tokens=int(len(train)), test_tokens=int(len(test)), layers=list(h.layout.layer_ids),
                rows=n_rows, hash_seconds=round(time.time() - t1, 1),
                note="the training stream is one sequence (padding only at its start), as in training windows")
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
