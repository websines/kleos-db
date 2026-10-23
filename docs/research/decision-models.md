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

## 5. Decision log

### 2026-10-16: decider-nano is the built-in default; optimise it rather than swap models

**Decision:**

- tasksource-decider-nano (150M, Apache-2.0) is the default built-in decider.
- We improve it rather than ship a bigger model:
  - fine-tuning on memory-shaped pairs;
  - the `::decider fit` probe with shipped defaults;
  - rule helpers for numbers and dates;
  - pair-symmetric scoring, template ensembling and per-template calibration;
  - Rust speed work: int8, length bucketing, batching, a pair cache, and
    embedding prefilters.
- Templates are always choices among descriptive options. There are no bare
  yes/no questions.

**What we stopped, and where it is parked:**

- **The d1-omni-600M Rust port.** Branch `wave2/d1rs`, unmerged. The forward
  pass was written but parity with PyTorch was not yet proven. The $10M-revenue
  licence cap is another reason to drop it.
- **The OneJev1.5-0.8B and vLLM-SR d3-lite evaluation.** Branch
  `wave2/qwen-deciders`, unmerged, with the harness in
  `scripts/infer-bench/decide/`. It was stopped before VitaminC, d3-lite and
  the llama.cpp latency runs.

**The evidence at the time.** Same protocol for every model: templates,
thresholds and bias were selected on the 60 adapt items and reported on the 66
eval items.

| | decider-nano 150M | OneJev1.5-0.8B (partial) | d1-omni 600M |
|---|---|---|---|
| supersedes AUROC | 0.938 (probe 0.993) | 0.958 | 0.933 (probe 0.998) |
| update vs compatible | 0.875 | 0.917 | 0.875 |
| contradicts AUROC | 0.944 | 0.979 | – |
| duplicate AUROC | 0.997 | 1.0 | – |
| same_entity AUROC | 0.924 | 0.99–1.0 | 0.958 |
| 3-way relation acc (bias-corrected) | 0.905 | 0.833 | 0.833 |
| auto band at α = 2% (supersedes) | 0% zero-shot, 36% with probe | 33%, 1 false of 14 | – |
| VitaminC-300 supersede AUROC | 0.679 | not run | not run |

OneJev led zero-shot on most pair decisions. It is 5× larger, Qwen3.5-based
(candle cannot run it, so it would need llama.cpp or a port), and it is less
calibrated than claimed (ECE 0.11–0.19).

**Revisit when:**

- the optimised decider-nano still trails OneJev's zero-shot numbers;
- or VitaminC-style real text stays weak (decider-nano scores 0.68) after
  fine-tuning.

To revisit, resume `wave2/qwen-deciders`. Its partial results are in
`.bench-data/qwen-deciders/onejev_mps.json`, which lives outside the repo.

### 2026-10-23: nano-tune-v1 (fine-tune of decider-nano) beats the base on every held-out set

Fine-tuning on memory-shaped pairs worked; see docs/specs/decide.md, "Fine-tuned weights".

| | base | nano-tune-v1 |
|---|---|---|
| supersedes AUROC | 0.938 | 0.991 |
| VitaminC-300 | 0.679 | 0.807 |
| LongMemEval-KU held-out | 0.386 | 0.823 |
| MNLI dev | 0.805 | 0.817 |

The model is still 150M. Memory went down too: Metal q8 needs 287 MB after load and a batch. The "revisit when VitaminC stays weak" trigger above is not hit.

Open: the weights' licence (CC BY-SA training data) is not decided; Metal batching is unsolved.

