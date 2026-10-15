# Inference benchmarks (UPGRADE_PLAN D1-D3)

Measurement scripts behind the D1 (decision-model feasibility) and D3
(embedder/precision choice) numbers in the CHANGELOG. They need a Python venv
with `torch`, `transformers>=5.15`, `sentence-transformers`, `model2vec` and the
models in the Hugging Face cache; nothing here is used by the build.

- `d1/eval_set.py`: 66 hand-written memory decisions: 42 "does fact B supersede
  fact A?" pairs (14 updates, 4 corrections, 12 compatible additions, 12
  paraphrased duplicates) and 24 "same entity?" pairs (12 same, 12 different).
- `d1/run_d1.py DEVICE DTYPE`: LiquidAI d1-omni-600M (`system_one`), accuracy,
  Brier, ECE, AUROC, latency. `d1/run_dmjepa.py`: DangerLabs DM-JEPA 1.1.
  `d1/run_cosine.py`: embedding-cosine baseline (thresholds tuned on the set).
- `beir/prep.py`: SciFact (mteb/scifact) to JSONL; `beir/ndcg.py`: exact nDCG@10
  from raw f32 vector files; `beir/run_prec.sh`: precision sweep with
  `kleos-infer`'s `embed_bench` example. The end-to-end run through kleos is
  `cargo run --release -p kleos --features infer-metal --example embed_scifact`.
