# EmbeddingGemma 2 in kleos

*2026-10-24, branch `wave5/egemma2`. kleos's default embedder moved from
EmbeddingGemma (`google/embeddinggemma-300m`, Gemma Terms of Use, gated) to
EmbeddingGemma 2 (`google/embeddinggemma-2`, Apache-2.0, not gated).*

## The model

From `config.json`, the safetensors header and transformers 5.19's
`modeling_embedding_gemma2.py`:

- 740M parameters in one checkpoint: text backbone and embedder ~271M
  (`language_model.*`: a 262,144 × 512 token table, 24 layers, PLE, a 512 → 768
  projection), vision tower ~167M, audio tower ~305M. kleos reads only the
  `language_model.*` tensors.
- Text path: a Gemma 4 encoder with bidirectional attention. 20 sliding layers
  (head_dim 256, 4 query heads, 2 KV heads, window `|i - j| <= 512`, RoPE θ
  10⁴) and 4 full layers (5, 11, 17, 23: head_dim 512, one KV head, θ 10⁶).
  q, k and v are RMS-normalised (v without a scale) and attention scaling is 1.
  RMSNorm multiplies by `w` (v1 used `1 + w`). Every layer ends with a
  per-layer-embedding block: a gate (512 → 512, GELU-tanh) times that layer's
  slice of the PLE signal (a projection of the scaled token embeddings to
  24 × 512, RMS-normalised), projected back and normalised; then a learned
  scalar.
- The final hidden states go through the 512 → 768 projection per token, then
  mean pooling over all tokens (prompt included) and L2 normalisation. Output
  is Matryoshka (768/512/256/128).
- Prompts as in v1: `title: none | text: ` for documents, `task: search
  result | query: ` for queries. The tokenizer adds BOS and EOS.

## Parity

`kleos-infer/tests/egemma2_parity.rs` against sentence-transformers 6.1
(transformers 5.19, torch f32, eager attention) on 50 texts (short, Unicode,
CJK, Arabic, emoji, code, SQL, and long texts up to 6,310 tokens; one text of
13k tokens is beyond the 8,192 kleos uses and is skipped). Token ids are
identical. Cosine against the reference, worst text:

| device | precision | documents | queries | weights |
|---|---|---|---|---|
| CPU | f32 | 1.000000 | 1.000000 | 1,084 MB |
| Metal | f32 | 1.000000 | 1.000000 | 1,084 MB |
| Metal | f16 | 1.000000 | 1.000000 | 542 MB |
| Metal | q8 | 0.999859 | 0.999880 | 288 MB |
| CPU | q8 | 0.999579 | 0.999683 | 288 MB |

## Speed and memory against v1

`kleos-infer/examples/egemma_speed.rs`: 64 memory-sized sentences (~25
tokens), warm, M3, load average about 5.

| model | precision / device | batch 1 | batch 16 | weights | peak RSS |
|---|---|---|---|---|---|
| v1 | q8 Metal | 9.1 ms | 5.6 ms | 327 MB | 863 MB |
| **v2** | q8 Metal | 13.2 ms | 9.9 ms | 288 MB | 771 MB |
| v1 | f16 Metal | 12.4 ms | 5.7 ms | 615 MB | 1,243 MB |
| **v2** | f16 Metal | 16.9 ms | 10.2 ms | 542 MB | 1,018 MB |
| v1 | q8 CPU | 60.4 ms | 60.4 ms | 327 MB | 857 MB |
| **v2** | q8 CPU | 131.3 ms | 96.9 ms | 288 MB | 784 MB |

v2 uses less memory but is slower per text: same depth, narrower residual
(512 vs 768) but wider attention projections (q to 1024/2048), the PLE
projection to 12,288 per token and a third residual block per layer. On the
CPU it is about 2× v1 at batch 1. The first load from a cold page cache read the
text tensors (~540 MB of bf16) over this machine's USB 2.0 disk link in 18 s;
warm loads take 0.6 s.

## Memory benchmarks: v2 against v1

One build (`wave5/egemma2` at 8436903 + harness), the same harness, reader and
judge (GLM-4.7-Flash), shipped recall defaults, `q8` on Metal; only the
embedder differs (`KLEOS_EMBED_MODEL`).

**Retrieval** (`retrieval.py --variant hybrid --configs default`):

| bench | split | v1 R@5 | v2 R@5 | v1 R@10 | v2 R@10 |
|---|---|---|---|---|---|
| LoCoMo (10 conversations, 1,536 questions, cat 1–4) | all | **0.693** | 0.629 | **0.779** | 0.705 |
| | multi-hop | 0.652 | 0.553 | 0.759 | 0.674 |
| | temporal | 0.766 | 0.707 | 0.850 | 0.748 |
| | open-domain | 0.467 | 0.370 | 0.587 | 0.478 |
| | single-hop | 0.703 | 0.653 | 0.780 | 0.724 |
| LongMemEval 119 | all | 0.992 | 1.000 | 1.000 | 1.000 (Rall@10 0.966 vs 0.958) |

**Is it the model or kleos's fusion?** The same LoCoMo stores with other
fusion weights (`vector_weight`; 0.6 is the default):

| weight | v1 R@5 / R@10 | v2 R@5 / R@10 |
|---|---|---|
| vector only (1.0) | **0.775 / 0.847** | 0.658 / 0.748 |
| 0.8 | 0.716 / 0.794 | 0.653 / 0.725 |
| 0.6 (default) | 0.695 / 0.777 | 0.632 / 0.705 |
| 0.4 | 0.654 / 0.745 | 0.626 / 0.705 |

v2's vector leg alone is 12 points behind v1's on LoCoMo: the gap is the
model on this conversational data, not the fusion weights. (Separately: for
v1, vector-only beats the default hybrid by 8 points R@5 on LoCoMo. The 0.6
weight was tuned on SciFact; worth re-tuning on memory data.)

**QA, context output** (`qa.py --conditions ctx:hybrid+default`):

| bench | v1 | v2 |
|---|---|---|
| LoCoMo 301 | **0.814** | 0.738 |
| — multi-hop (55) | 0.818 | 0.673 |
| — temporal (63) | 0.746 | 0.667 |
| — open-domain (19) | 0.737 | 0.526 |
| — single-hop (164) | 0.848 | 0.811 |
| LongMemEval 119 | 0.664 | 0.664 |
| — knowledge-update (32) | 0.812 | 0.844 |

**Knowledge-update latest-first** (23 held-out questions, decider on,
`claims_ku.py`): v1 83% (91% with `hide_pending`), v2 78% (91%).

**Verdict:** on kleos's memory benchmarks EmbeddingGemma 2 is worse than v1
on LoCoMo (−6 R@5, −7.6 QA points) and level on LongMemEval. Its advantages
are the licence (Apache-2.0, not gated), less memory, 8k context, much better
code retrieval (card: MTEB code 78.7 vs 68.8) and multimodality. Prompt
variants for questions (e.g. `task: question answering | query: `) were not
tried. Quota: about 0.24 points of the weekly plan (95.87% → 95.63%).
