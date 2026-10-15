# Decision models for kleos memory: research summary

*2026-10-16. This page summarises three investigations behind UPGRADE_PLAN D5
(`decide()`). The detail and citations are in:*

- [`system-one-landscape.md`](system-one-landscape.md): Jev, Kev, d1, DM-JEPA and
  about 160 other "System One" decision models, plus the categories next to them
  (NLI, zero-shot classifiers, rerankers used as judges, routers, SetFit). It
  covers licences, latency, independent Decision Index scores and candle/ort
  support. The scores are in
  [`decision-index-v0.3-extract.csv`](decision-index-v0.3-extract.csv).
- [`memory-decisions.md`](memory-decisions.md): how Mem0, Graphiti/Zep, MemOS,
  Cognee, Letta, LangMem, Memobase, A-Mem, MIRIX, HippoRAG 2 and Supermemory make
  update, dedup, contradiction and entity decisions (read from their source),
  what the memory benchmarks measure, how small models compare with LLM judges,
  and which labelled datasets are available.
- The d1-omni audit: [`../../scripts/infer-bench/d1/`](../../scripts/infer-bench/d1/)
  (`run_d1_v2.py`, `d1_numerics.py`, `eval_set_adapt.py`).

## 1. The first d1-omni result was our mistake

Our first D1 run drove d1-omni-600M wrongly and called it "not usable". Its
supersession AUROC was 0.63 and its 3-way relation accuracy 0.48.
`memory-decisions.md` still quotes those numbers. They are superseded by the
audit below.

| d1-omni-600M, same 66 items | first run | corrected |
|---|---|---|
| Supersession AUROC | 0.63 | 0.93–0.97 |
| Relation accuracy (update / duplicate / compatible) | 0.48 | 0.79–0.90 |
| Entity AUROC | 0.83 | 0.94–0.96 |
| Logistic probe on head features (60 held-out labels) | – | AUROC 0.95–0.998 |

There were three mistakes:

- We used a yes/no ("noul") framing, which this model answers unreliably and
  sometimes with the polarity inverted.
- We passed a JSON state with ad-hoc keys.
- We took the raw argmax without correcting for the model's bias toward
  "update".

The NLI / ContractNLI choice framing that Liquid benchmarks on fixes all three.
The f16 numerics were fine; bf16 flips answers, as the README warns.

The remaining caveats:

- The model is very sensitive to phrasing: across reasonable prompts, AUROC
  ranges from 0.41 to 0.97. Templates must therefore be fixed and validated.
- The eval set is small and easy.
- The licence (LFM Open v1.0) caps commercial use at $10M of revenue.

## 2. The landscape in one paragraph

- **Jev** (TypeSafe AI) is API-only and closed. It costs $0.042 per million
  input tokens and takes about 200 ms. Its Decision Index score is about 61, and
  the best model scores 66.
- **Kev** (`jaredpalmer/kev`, Apache-2.0) is an open Jev-compatible family
  (0.8B / 4B / 9B / 27B) built on Qwen3.5. It is served by llama.cpp's
  `/v1/systemone` endpoint. candle does not support Qwen3.5 yet.
- **d1-3B** is strong (Decision Index 38.9), but at 3B parameters it is too
  heavy to ship attached to a database.
- **d1-omni-600M** scores 9.1 on the Decision Index, and it is weak at saying
  "not supported".
- **DM-JEPA** is not credible: it scores 4 on the independent private split,
  and its claims look like test-set leakage.
- **tasksource-decider-nano** (150M, Apache-2.0, ModernBERT/Ettin, about 14 ms)
  is the best cheap open option. It beats d1-omni on every sub-score that matters
  for memory, and its architecture runs in candle.
- On narrow judgements like these, fine-tuning or a light probe clearly beats
  zero-shot. Labels produced by a large LLM train classifiers as well as human
  labels do.

## 3. How memory systems decide today

The field has converged on **non-destructive supersession**, which kleos already
has (E1/E2):

- Graphiti keeps bitemporal edges.
- Mem0 v3 is ADD-only, with supersession done in a closed "Dream" layer.
- Cognee supersedes by rule on single-valued relations.

Where an LLM is used, the decision is a typed pair judgement. MemOS's
`contradictory | redundant | independent` is exactly a `decide()` choice.

Failed updates on HaluMem are mostly omissions (37–94%), not wrong updates
(under about 1.2%). A cheap model that catches updates is worth more than a
cautious one.

NLI models label "lives in Berlin → moved to Lisbon" as a contradiction, not an
update. The relation template needs four labels: update, contradiction,
compatible and duplicate.

## 4. Decision for kleos (user call, 2026-10-16)

Ship **small models only**. 3B "eats RAM". We make the small ones work through
helpers:

- **Tier 0 (rules, synchronous):** keys, hashes, MinHash and embedding cosine,
  used for duplicate and entity blocking. Single-valued keys keep rule-based
  supersession.
- **Tier 1 (`decide()`):**
  - **Backends:** d1-omni-600M (a Rust port in `kleos-infer`) and
    tasksource-decider-nano.
  - **Templates:** fixed and versioned per backend, with pair-symmetric scoring.
  - **Calibration:** a threshold and per-option bias. Thresholds are set by risk
    control (false supersession bounded at about 1–2%), which gives auto,
    tentative and none bands.
  - **Probe:** an optional logistic probe fitted on labelled pairs with
    `::decider fit`.
- **Tier 2 (asynchronous):** an LLM fallback for the tentative band, driven by
  live subscriptions. Every escalation becomes a training example.
- **Safety:**
  - Never delete on a model decision.
  - Record each decision with model, template version, p and band, so it is
    auditable and can be re-run.
  - Require the same resolved entity before scoring supersession.
  - Never merge entities transitively.

**Evaluation data for the next round:**

- **VitaminC** (real Wikipedia revision pairs) for update/supersede.
- **LongMemEval knowledge-update** pairs, kept as a held-out set.
- **PAWS** for duplicate hard negatives.
- The **Leipzig** entity-matching sets (CC BY 4.0).

Keep the 66 hand-written items as a smoke test only.
