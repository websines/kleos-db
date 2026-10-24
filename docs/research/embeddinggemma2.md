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
