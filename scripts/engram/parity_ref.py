"""Dump reference Engram hash indices from DeepSeek's engram_demo_v1.py.

The reference is the upstream demo's own NgramHashMapping class, imported
unmodified from a path you pass in. Nothing here re-implements the hash.

Usage (run with -I so nothing is imported from the current directory):

    python -I parity_ref.py --demo /path/to/Engram/engram_demo_v1.py \
        --tokenizer /path/to/tok-dir --out /path/to/out-dir

It writes, per config:
    <out>/<cfg>/lookup.u32   raw token id -> compressed id, u32 little-endian
    <out>/<cfg>/meta.json    num_new_token, pad_id, multipliers, primes, config
    <out>/<cfg>/cases.jsonl  one line per case: input ids and per-layer indices
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

CONFIGS = {
    # The demo's defaults (EngramConfig in engram_demo_v1.py).
    "demo": dict(engram_vocab_size=[129280 * 5, 129280 * 5], max_ngram_size=3,
                 n_embed_per_ngram=512, n_head_per_ngram=8, layer_ids=[1, 15],
                 pad_id=2, seed=0),
    # Engram-27B table sizes from the paper (Table 5): vocab 2262400, N in [2,3], 8 heads, layers [2,15].
    "paper27b": dict(engram_vocab_size=[2262400, 2262400], max_ngram_size=3,
                     n_embed_per_ngram=1280, n_head_per_ngram=8, layer_ids=[2, 15],
                     pad_id=2, seed=0),
    # Off-default: 4-grams, 4 heads, 3 layers, non-zero seed, uneven per-order sizes.
    "ngram4": dict(engram_vocab_size=[1000003, 500000, 250000], max_ngram_size=4,
                   n_embed_per_ngram=256, n_head_per_ngram=4, layer_ids=[0, 7, 30],
                   pad_id=1, seed=12345),
}

TEXTS = [
    "Only Alexander the Great could tame the horse Bucephalus.",
    "ONLY alexander THE great Could Tame the HORSE bucephalus.",
    "Café naïve résumé — Ångström façade, coöperate.",
    "  multiple   spaces\tand\ttabs\n\nand newlines\r\n end  ",
    "def engram(x):\n    return x ** 2  # 平方\n",
    "北京是中华人民共和国的首都。東京は日本の首都です。서울은 한국의 수도입니다.",
    "Emoji 🧠🔑 and fullwidth ＡＢＣ１２３ and ligature ﬁ ﬂ, superscript x² ½.",
    "User Alice prefers oat milk; her manager is Bob (since 2024-03-01).",
]


def load_demo(path):
    spec = importlib.util.spec_from_file_location("engram_demo_v1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module body only builds configs; no model is created
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--official", help="DeepSeek-V4.1-Flash inference/engram.py (optional)")
    ap.add_argument("--official-tokenizer", help="DeepSeek-V4.1-Flash tokenizer dir")
    ap.add_argument("--random-cases", type=int, default=64)
    ap.add_argument("--random-len", type=int, default=512)
    args = ap.parse_args()

    demo = load_demo(args.demo)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)

    rng = np.random.default_rng(424242)
    vocab = len(tok)
    cases = []
    for t in TEXTS:
        cases.append(("text", tok(t).input_ids))
    # Uniform random raw ids over the full vocab (includes added/special tokens).
    for _ in range(args.random_cases):
        cases.append(("random", rng.integers(0, vocab, size=args.random_len).tolist()))
    # Edge cases: length-1, all-pad, negative ids (the demo passes negatives through compression).
    cases.append(("len1", [5]))
    cases.append(("allpad", [2] * 16))
    cases.append(("negatives", [-1, 100, -1, 200, 300, -100, 7]))

    for name, cfg in CONFIGS.items():
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        hm = demo.NgramHashMapping(tokenizer_name_or_path=args.tokenizer, **cfg)
        ct = hm.compressed_tokenizer
        ct.lookup_table.astype("<u4").tofile(os.path.join(d, "lookup.u32"))
        meta = dict(
            config=cfg,
            raw_vocab_size=int(vocab),
            num_new_token=int(ct.num_new_token),
            pad_id_compressed=int(hm.pad_id),
            multipliers={str(k): [int(x) for x in v] for k, v in hm.layer_multipliers.items()},
            primes={str(k): [[int(p) for p in heads] for heads in v]
                    for k, v in hm.vocab_size_across_layers.items()},
        )
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        with open(os.path.join(d, "cases.jsonl"), "w") as f:
            for kind, ids in cases:
                out = hm.hash(np.asarray([ids], dtype=np.int64))
                f.write(json.dumps(dict(
                    kind=kind, ids=[int(x) for x in ids],
                    hashes={str(l): h[0].tolist() for l, h in out.items()},
                )) + "\n")
        print(f"{name}: compressed vocab {ct.num_new_token} / raw {vocab}, "
              f"pad {hm.pad_id}, {len(cases)} cases", file=sys.stderr)

    if args.official:
        dump_official(args, rng)


# DeepSeek-V4.1-Flash text_config engram fields (config.json on the HF repo).
V41 = dict(engram_layer_ids=[1, 14], engram_num_embeddings=[384006168, 384016682],
           engram_max_ngram_size=4, engram_vocab_size=16000000, engram_n_heads=8,
           engram_head_dim=256, engram_pad_id=2, engram_compressed_vocab_size=99092)


def dump_official(args, rng):
    """Production reference: the official V4.1-Flash NgramHashState, run on CPU."""
    import types
    import torch
    off = load_demo(args.official)  # same loader; the module only defines classes
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.official_tokenizer)
    a = types.SimpleNamespace(**V41, max_batch_size=1, max_seq_len=4096)
    layout = off.EngramLayout.from_args(a)
    st = off.NgramHashState(a, layout, tok)
    d = os.path.join(args.out, "v41")
    os.makedirs(d, exist_ok=True)
    st.token_map.numpy().astype("<u4").tofile(os.path.join(d, "lookup.u32"))
    cfg = dict(engram_vocab_size=[V41["engram_vocab_size"]] * 3, max_ngram_size=4,
               n_embed_per_ngram=8 * 256, n_head_per_ngram=8, layer_ids=V41["engram_layer_ids"],
               pad_id=V41["engram_pad_id"], seed=0, semantics="v41")
    meta = dict(config=cfg, raw_vocab_size=len(tok), num_new_token=V41["engram_compressed_vocab_size"],
                pad_id_compressed=int(st.pad_id),
                multipliers={str(l): st.multipliers[i].tolist() for i, l in enumerate(layout.layer_ids)},
                primes={str(l): [list(h) for h in layout.primes[i]] for i, l in enumerate(layout.layer_ids)},
                offsets={str(l): st.offsets[i].tolist() for i, l in enumerate(layout.layer_ids)},
                num_embeddings=V41["engram_num_embeddings"])
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    vocab = len(tok)
    cases = [("text", tok(t).input_ids, None) for t in TEXTS]
    cases += [("random", rng.integers(0, vocab, size=args.random_len).tolist(), None)
              for _ in range(args.random_cases)]
    # Image-span style DEAD tokens: token_mask False at some positions.
    ids = rng.integers(0, vocab, size=64).tolist()
    mask = [not (10 <= i < 14 or i == 30 or i == 31) for i in range(64)]
    cases.append(("dead_mask", ids, mask))
    with open(os.path.join(d, "cases.jsonl"), "w") as f:
        for kind, ids, mask in cases:
            x = torch.tensor([ids], dtype=torch.long)
            m = None if mask is None else torch.tensor([mask])
            h = st(x, 0, m)[0]  # [L, n_layers, cols], offsets included
            # Rust side marks DEAD as id -1.
            rid = ids if mask is None else [i if keep else -1 for i, keep in zip(ids, mask)]
            f.write(json.dumps(dict(kind=kind, ids=[int(v) for v in rid],
                                    hashes={str(l): h[:, i, :].tolist()
                                            for i, l in enumerate(layout.layer_ids)})) + "\n")
    print(f"v41: compressed vocab {V41['engram_compressed_vocab_size']} / raw {vocab}, "
          f"pad {int(st.pad_id)}, {len(cases)} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
