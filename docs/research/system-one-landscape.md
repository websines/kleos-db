# System-One / Decision Models: Landscape for kleos `decide()`

*Research date: 2026-10-16. No models were downloaded or run. Every number comes from the URL cited next to it.*

**How to read the labels**
- **[vendor]** means the maker reported it about its own model.
- **[3p]** means a third-party blog or aggregator.
- **[DI]** means it comes from the Decision Index 0.3.2 data file published by its maintainers (multimodalart/apolinario): https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json and `.../data/v03.json`. That file was generated 2026-10-15.
- **UNCERTAIN** marks anything that could not be confirmed from a primary source.

The per-model Decision Index (DI) figures are extracted to `decision-index-v0.3-extract.csv` in this folder. It covers 167 engines, with latency, ECE, Brier, and the ANLI, ContractNLI, NLI4CT, RAGTruth, HoVer, CLINC150, BANKING77 and RouterBench scores.

---

## 0. Executive summary

1. **Jev** is a hosted, API-only "System One" model from **TypeSafe AI**, released in early access on 2026-09-15.
   - Input: a state plus typed questions, which are Noul (yes/no), Choice and Score.
   - Output: calibrated probabilities. It does not generate text.
   - Price: $0.042 per million input tokens, and output is free.
   - It has no local weights and publishes no parameter count, licence or paper.
   - Its arrival created a new model category within about four weeks: more than 160 open "Jev-like" models, a shared API shape (`POST /v1/systemone`), and an independent benchmark, the **Decision Index**.
2. **"Kev" is real.** It is **jaredpalmer/kev**, Jared Palmer's open-weight, Apache-2.0 "Jev-like" family built on Qwen3.5/3.8.
   - Sizes: 0.8B, 4B, 9B and 27B.
   - It speaks the same `/v1/systemone` API, so the TypeSafe SDK works against it unchanged.
   - It is not a TypeSafe product and was not distilled from Jev.
3. **The weak d1-omni-600M result is what the independent benchmark predicts.** On DI 0.3 it scores **9.09**, against 38.9 for d1-3B and 60–61.5 for Jev.
   - Its RAGTruth "unsupported-claim" F1 is **0.02**, which means it almost never answers "not supported". That is consistent with our poor supersede AUROC.
   - Liquid itself calls it an "experimental checkpoint".
4. **DM-JEPA is not credible.** It is a ModernBERT-base plus scoring-head model with self-reported 100% scores.
   - The independent DI private split gives v1.1 a score of **4.05**: rank about 160 of 167, ECE 0.445.
   - Its 0.986 on ANLI is the signature of test contamination.
   - Its v1.5 leaderboard PR was withdrawn by its own author "for independent audit".
5. **What to use for kleos**
   - No zero-shot decision model under 1B is strong enough to be the final arbiter for supersede or contradiction.
   - The evidence-backed path is:
     - a small cross-encoder (ModernBERT/Ettin or DeBERTa-v3) used as a pairwise judge, which candle already supports;
     - fine-tuned on a few hundred to a few thousand labels distilled from a large LLM judge;
     - temperature-scaled per decision type;
     - with dates and numbers handled in Datalog, not by the model.
   - For zero-shot baselines and a teacher-in-the-loop, the best open small decision models are:
     - **vLLM-SR d3-lite (0.8B, Apache-2.0, DI 32.3)**;
     - **OneJev1.5-0.8B (Apache-2.0, DI 29.0, ECE 0.035)**;
     - **tasksource-decider-nano (150M encoder, Apache-2.0, DI 17.6, 14 ms)**. This one beats d1-omni on every memory-relevant sub-benchmark and is the most Rust-portable.

---

## 1. Jev (TypeSafe AI)

| Item | Finding | Source |
|---|---|---|
| Maker | TypeSafe AI (typesafe.ai). Its CEO commented on Hacker News. | https://typesafe.ai/blog/introducing-system-one-models-and-jev ; https://simonwillison.net/2026/Sep/21/jev/ |
| Release | Early access 2026-09-15. Versions seen: `jev-1.12` (cookbooks from August–September), `jev-1.13.0` (current response string), `jev-1.14.0` (on the DI board) | https://docs.typesafe.ai/llms-full.txt ; DI |
| What it is | "a frontier-intelligence function call: unstructured state in, typed probabilistic decisions out". It gives up string generation and uses a "parallel sampler" that produces all outputs in one query. | TypeSafe blog |
| Training | "Reinforcement Learning for Calibrated Decisions (RLCD)". The recipe is unpublished and there is no paper. | TypeSafe blog |
| API | `POST https://api.typesafe.ai/v1/systemone` with body `{model, state, questions:{id:{type, instructions, criteria}}}`. Models: `jev-latest`, `jev-1.13.0`. Body limit 32 MiB. Text only, English primarily. | https://docs.typesafe.ai/llms-full.txt |
| Primitives | **Noul** (yes/no; returns `noul` = P(yes); the name is short for Bernoulli; up to 2 criteria). **Choice** (up to 255 options; `choice`, `probabilities`, `confidence`). **Score** (2–10 ordered levels; `score`, `legend`, `probabilities`, `confidence`). | docs ; https://news.ycombinator.com/item?id=49717558 |
| Confidence | Choice: (p_max − 1/n)/(1 − 1/n). Score: 1 − expected distance from the mode divided by the uniform MAD. Noul has no confidence field; the docs suggest \|2p−1\|. "Calibration is measured across groups of predictions." | docs ; adapter code https://github.com/typesafe-ai/system-one-adapter-python |
| Pricing | $0.042 per million input tokens; output $0. Also on Cloudflare Workers AI as `typesafe/jev` at the same price, 32k context, zero data retention. | TypeSafe blog ; https://developers.cloudflare.com/ai/models/typesafe/jev/ |
| Latency | [vendor] 70–500 ms end to end, with "most queries ~100 ms". Cookbooks: 0.32 s for 16 questions, 0.51 s for 62. [DI] median 196 ms (Jev 1.14). | blog ; docs ; DI |
| Context | 32,000 (Cloudflare). [3p] probing found about 32,768 per question branch and about 65,536 per request, with the state counted once. | Cloudflare ; https://archerhume.com/posts/jevs-architecture-unmasked |
| Local or on-prem | **No.** Hosted API only; no weights. | TypeSafe blog, docs |
| Licence | No model licence. API terms only. | — |
| Architecture (3p reverse-engineering, speculative) | A causal decoder (likely sparse MoE, about 10B active) that encodes the shared state once. Each question runs in an isolated branch; the answer is read out by softmax or sigmoid over a pointer or slot head. The tokenizer matches none of 192 public ones (Qwen is closest). | https://archerhume.com/posts/jevs-architecture-unmasked |
| Published accuracy | [vendor] Four workflow evals against GPT-6/Fable consensus labels: mean **67.8%** at $0.0004 and 0.4 s per case. Sonnet 5 scores 67.8% at 78 s; Opus 5 scores 73.1%. [3p] MMLU ECE 0.031 on 1,200 items. [DI] Full score 60.11 (1.13) and 61.53 (1.14); ANLI 0.74, ContractNLI 0.71, RAGTruth 0.75, HoVer 0.77, ECE 0.080. | https://evals.typesafe.ai/ ; Archer Hume ; DI |
| Known weaknesses (vendor "jaggedness" page) | Literal reading. Math and counting. Numbers. **Dates**: "reads dates as text, not ordered quantities". Negation and multi-hop. Large irrelevant state lowers accuracy. Prompt injection. **First-option bias** in Choice. | https://docs.typesafe.ai/model-jaggedness/jev-1.13 |
| Memory relevance | The official cookbook "Simple memory system" uses a Noul gate, "does the request state an important fact about the customer that should be remembered?", with a 0.5 threshold. Adding memory took ticket routing from 2/6 to 6/6 correct. | https://docs.typesafe.ai/cookbooks/basic_memory |
| Tooling | `llm-typesafe` plugin; Python SDK `typesafe_sdk`; community projects jevchat, jev-leftpad, jevgrep, pg-jev | Simon Willison ; https://www.liquid.ai/blog/d1-decision-model |

**Fit for kleos:** Jev is a strong cloud teacher for label distillation. It is cheap: 10k pair-judgements at about 500 tokens each cost about $0.21. It cannot be embedded, so it is not a runtime option for an embedded database. Its weak handling of dates and numbers matches our supersede problem. Temporal ordering must be done in Datalog, not by the model.

## 2. "Kev": identified

**Kev = https://github.com/jaredpalmer/kev**. It is a "Jev-like family of decision models built on top of Qwen3.5/3.8 you can train and run on your own". The author is **Jared Palmer**. Simon Willison's post names it as an open-weight attempt, with HN discussion at https://news.ycombinator.com/item?id=49783999.

| Item | Finding (all from the Kev README unless noted) |
|---|---|
| Sizes | Kev-0.8B (Qwen3.5-0.8B-Base), Kev-4B, Kev-9B (Qwen3.5-*-Base), Kev-27B (Qwen3.8-27B, full fine-tune). Released together as **Kev 1.0**, with a GitHub release tagged `kev-1.0`. Earlier generations were built on Qwen3 (0.6/4/8B) and Qwen2.5-0.5B; that line is discontinued. |
| Licence | **Apache-2.0** for code and weights. The HF API reports `apache-2.0` for kev-0.8b, 4b, 9b and 27b. |
| Method | Rank-16 LoRA plus a small **pointer head** on a frozen base. Each option's `</opt>` hidden state is scored against the question's `<decide>` state, then softmaxed. Loss is cross-entropy. Data: `decision-v7`, built from 10k examples drawn from 10 public datasets plus generated policy and rule data. **No Jev outputs were used.** |
| Questions | Isolated by an attention mask on attention-only bases. On Qwen3.5 hybrid bases (Gated DeltaNet) each question runs as its own row. |
| Calibration | One fitted temperature per checkpoint. Kev-9B ECE drops from 0.103 to 0.041 after temperature scaling. |
| API | `POST /v1/systemone`, plus `/permute` (averages over option orders) and `/separate`. Works with the TypeSafe SDK unchanged. |
| Accuracy [vendor] | "New sources" accuracy, dev/test: 0.8B 0.648/0.697, 4B 0.817/0.838, 9B 0.820/0.852, 27B 0.851/0.889; Jev 0.857 (dev). Held-out index: 0.8B 23.3, 4B 38.0, 9B 41.0, 27B 52.3, Jev 54.0. |
| Accuracy [DI 0.3] | Kev 27B **58.78** (ECE 0.022, best calibrated on the board), Kev 9B v2 43.27, Kev 4B 39.5, Kev 0.8B **13.66**. |
| Latency [vendor] | Kev-4B, 6 questions: 18.1 ms on H100, 41.5 ms on L40S. Kev-0.8B: 22.7 ms on L4. **Kev-4B on an Apple M5, 5 questions: 721 ms (136 ms on repeated text).** [DI] Kev-0.8B 45 ms, Kev-4B 59 ms. |
| Runtimes | PyTorch, MLX (Apple). **llama.cpp mainline supports Kev** through `/v1/systemone` (PR #29818, merged 2026-10-02; GGUF at `ggml-org/Kev-4B-GGUF`, Apache-2.0). Not on PyPI; `pip install kev` installs an unrelated ORM. |
| Limitations (README) | A single temperature, so it cannot reorder confidences. **Option order changes answers.** Weak date arithmetic. Knowledge accuracy is bounded by the base model. |

**Other "Kev-like" spellings**: the DI tracks **Lev** (interfaze-ai/lev, Apache-2.0), **Tev1** (Together AI), **Zev**, **Gev(va)**, **JevK5**, **OpenJev** (AlexWortega/openjev, MIT), **mini-jev**, **NanoJev**, **AJev**, **ezjev**, **Jev-Style**, **OneJev** and **Jevstral**. Kev is the one Simon Willison names. It is also the one tracked by the systemonemodels.ai catalog (https://systemonemodels.ai/jared-palmer/kev). The identification is solid.

**Fit for kleos:** Kev is the most transparent open recipe: README, training code, data registry, pointer head and LoRA. It is the template to copy if we train our own decision head. Kev-0.8B is weak zero-shot (DI 13.7). Kev-4B is decent but slow on a Mac and is Qwen3.5-based, so it cannot run in candle (see §7).

## 3. Liquid AI d1 family and the "System One" framing

- **"System One"** is the category name TypeSafe introduced, after Kahneman's fast thinking. Liquid adopted the same API shape:
  - `model.system_one(state, questions)` in Transformers;
  - `/v1/systemone` in llama.cpp;
  - the hosted path `/decisions/v1/systemone`.
  - Its demo Space is "System One Arcade".
  - Liquid's d1 blog says the experimental d1 "became the first model to rival Jev on text decisions".
  - Sources: https://docs.liquid.ai/lfm/models/d1-omni-600m ; https://www.liquid.ai/blog/d1-decision-model ; https://huggingface.co/spaces/LiquidAI/system-one-arcade
- Liquid has two things named d1:
  - **d1 (hosted)**, from roughly 2026-09-28 to 10-05. Available through the Liquid API, Vercel AI Gateway and OpenRouter (`liquid/d1`). [vendor] $0.04 per million input tokens and 200–300 ms per text decision. Parameter count and backbone undisclosed. https://www.liquid.ai/blog/d1-decision-model
  - **Open d1** (2026-10-07). https://www.liquid.ai/blog/d1-open
    - **d1-3B**: 3.12B, built on LFM2.5-VL-3B (decoder, hybrid conv plus attention), 32k context. [vendor] DI 0.2.1 score 48.57. Latency 8 ms on RTX 4090, 30 ms on M5 Pro, 16 ms on Jetson Thor. [DI 0.3] 38.9 full; ANLI 0.62, ContractNLI 0.81, RAGTruth 0.78, HoVer 0.74, ECE 0.047, 16.1 ms.
    - **d1-omni-600M**: 587M = 381M trunk + 94M SigLIP2 vision + 112M FastConformer audio, on **LFM2.5-Encoder-350M (bidirectional)**, 16k context. "Experimental checkpoint" and "under active development". [vendor] mean 78.4 on 7 text sets (BoolQ 77.7, XNLI 74.7, PAWS-X 79.5). [DI 0.3] **9.09** full; ANLI 0.36, ContractNLI 0.66, **RAGTruth 0.023**, HoVer 0.60, ECE 0.128, 10.9 ms. Use fp16 on GPU; bf16 changes top answers.
    - Builds: GGUF, MLX (fp16, 8-bit, 4-bit), w8a8. llama.cpp support merged as PR #30110 (d1-3B) and #30114 (d1-omni), 2026-10-07/08.
- **Licence**: both open models are **LFM Open License v1.0** (`license: other / lfm1.0` on HF), with free commercial use below $10M revenue. Liquid's blog says "without restrictions", which **contradicts the licence file**. Trust the licence file.
- **LFM2 backbone family** (decoder): LFM2-350M, 700M, 1.2B, 2.6B, LFM2-8B-A1B (MoE) and LFM2.5 (2.6B, VL-450M, VL-3B, Encoder-350M). Same LFM1.0 licence. https://huggingface.co/LiquidAI/LFM2-1.2B
- **Interpretation for kleos**: our d1-omni numbers (supersede AUROC 0.63, entity 0.83) agree with the DI. d1-omni is the weakest of Liquid's decision models and has a strong "yes/supported" prior (RAGTruth F1 about 0). d1-3B is roughly 4x better on DI. Its licence is still LFM1.0, and LFM2.5 hybrid-conv VL is not a candle target as packaged (candle has `lfm2` text only).

## 4. DM-JEPA (Danger Labs): credibility audit

| Version | Claims (self-reported) | Problems |
|---|---|---|
| **1.1** (https://huggingface.co/DangerLabs/DM-JEPA) | ModernBERT-base plus GRU "verifier" and cosine scorer, MIT. DI 0.2.1 score 23.16. **100% on GSM8K and on "OpenJev High-Trust", Brier 0.0000**. 34 ms. | [DI 0.3, independent private split] **4.05 full** (public 19.19, same-skill 4.17, new domains 2.84). **ECE 0.445, Brier 0.94**, 63 ms. **ANLI 0.986** while ContractNLI is 0.29 and RAGTruth 0.20. A near-perfect public score on one benchmark next to collapse elsewhere points to train/test contamination. Params are listed as 0.3B on the card but 149M on the DI. |
| **1.5** (https://huggingface.co/DangerLabs/dm-jepa-1.5) | 94.24M, "Danger Labs Open-Weights" licence, closed training code. Claims **DI 0.3 = 68.92, #1**. Also claims ISO 26262 ASIL-D driving 92.46% with "0.00% phantom braking". | The leaderboard PR (https://github.com/apolinario/decision-index/pull/91) was **withdrawn by its author on 2026-10-09** "for independent audit and retraining". The PR body is internally inconsistent (25 scenarios vs 1,060). The card's description of 1.1 (124M, 84.2% coverage) contradicts the 1.1 card itself. |
| **1.7** (https://huggingface.co/DangerLabs/dm-jepa-1.7) | About 349M multimodal driving/drone model, Apache-2.0. Claims DI 0.3 **93.62%**, with 100% on 10 of 16 sub-benchmarks. | DI scores are chance-corrected out of 100, and the top real entrant is 66.3. A 93.6 is not plausible. No audit. Architecture and scope changed with every version. |

**Verdict:** not credible; do not use. Nothing JEPA-specific is validated in it; it is a ModernBERT classifier. Org: https://huggingface.co/DangerLabs. Catalog: https://systemonemodels.ai/danger-labs/dm-jepa.

**Real JEPA-for-text work**, for context:
- **LLM-JEPA** (Huang, LeCun, Balestriero; https://arxiv.org/abs/2509.14252) adds an embedding-prediction loss during fine-tuning.
  - Example gains: Llama-3.2-1B NL-RX 57.3 to 71.5, GSM8K 32.4 to 36.4.
  - About 2x training compute.
  - It needs paired "views" of the same content and is not a classifier.
- **VL-JEPA** (https://arxiv.org/abs/2512.10942) is vision-language only.
- No JEPA-style model appears among the strong DI entrants. **Do not pursue JEPA for decide().**

## 5. The decision-model ecosystem (Sept–Oct 2026)

**Decision Index** (https://github.com/apolinario/decision-index ; board https://huggingface.co/spaces/multimodalart/jev-decision-index)
- It is independent ("Not affiliated with TypeSafe AI").
- Weights: 20% public (37 benchmarks), 50% private same-skill, 30% **private new-domain**.
- Scores are chance-corrected, 0 to 100. Models slower than 1 s median on an RTX PRO 6000 are excluded.
- Memory-relevant sub-benchmarks: **ANLI, ContractNLI, NLI4CT** (entailment/contradiction), **HoVer** (claim verification), **RAGTruth** (unsupported-claim detection), **CLINC150, BANKING77, RouterBench** (routing).
- Calibration (ECE) and latency are reported per model.

**Hosted APIs**

| Product | Price | Notes | Source |
|---|---|---|---|
| TypeSafe Jev | $0.042/M | — | §1 |
| Liquid d1 | $0.04/M | — | §3 |
| **OpenAI Decisions API** | $0.10/M input, no output charge | `/v1/decisions` on `gpt-6-luna`. Predicates, choices and scores. Public beta 2026-10-06. [vendor] "up to 10x faster than Responses API" | https://letsdatascience.com/news/openai-opens-decisions-api-public-beta-b69a1e40 ; https://propakistani.pk/2026/10/07/openai-launches-decisions-api-with-up-to-10x-faster-responses/ |
| Cloudflare clef | — | 27B, Apache-2.0 open weights, DI 53.1 | — |
| Fastino GLiDE | — | 28B, DI 60.2, "open weights coming soon" | — |
| Fastino GLiNER2.5-Decide | [3p] $0.03/M | 0.49B, Apache-2.0, DI 9.2 | https://www.llmreference.com/model/gliner2.5-decide |
| Together Tev1 | — | API plus experimental 0.8B/4B weights | — |
| Upstage Solar Decide; InternLM Intern-Decision (0.8/2/4B, Apache-2.0) | — | Upstage named in the catalog only (UNCERTAIN) | https://systemonemodels.ai/ |

**Open small models worth knowing, ≤4.5B.** Scores are [DI 0.3]. Licence comes from the HF API. All are fresh (Sept–Oct 2026) and single-maintainer unless noted.

| Model | Params | Base | Licence | DI full | ANLI | ContractNLI | RAGTruth | HoVer | CLINC150 | ECE | Median ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vLLM-SR **d3-mini** (vllm-sr/d3-mini) | 4.5B | Qwen3.5-4B | Apache-2.0 | **52.6** | — | — | — | — | — | 0.056 | 50 |
| CoCo-Decision-4B | 4.2B | Qwen3.5-4B | CC-BY-NC | 48.8 | 0.69 | 0.85 | 0.80 | 0.87 | 0.92 | 0.116 | 63 |
| Decider 4B (Mapika) | 4.7B | Qwen3.5-4B | Apache-2.0 | 40.3 | — | — | — | — | — | 0.084 | 12.6 |
| Kev 4B | 4.7B | Qwen3.5-4B | Apache-2.0 | 39.5 | — | — | — | — | — | 0.032 | 59 |
| **LiquidAI d1-3B** | 3.1B | LFM2.5-VL-3B | LFM1.0 ($10M cap) | 38.9 | 0.62 | 0.81 | 0.78 | 0.74 | 0.91 | 0.047 | 16 |
| vLLM-SR d3-nano | 2.2B | Qwen3.5-2B | Apache-2.0 | 38.9 | 0.56 | 0.69 | 0.30 | 0.80 | 0.95 | 0.112 | 38 |
| **vLLM-SR d3-lite** | 0.85B | Qwen3.5-0.8B | Apache-2.0 | **32.3** | 0.49 | 0.54 | 0.30 | 0.82 | 0.94 | 0.094 | 37 |
| **OneJev1.5-0.8B** | 1.1B | Qwen3.5-0.8B | Apache-2.0 | 29.0 | 0.50 | **0.82** | **0.77** | 0.75 | 0.91 | **0.035** | 83 |
| Decider 2B (Mapika; 333k downloads) | 2.3B | Qwen3.5-2B | Apache-2.0 | 25.8 | 0.49 | 0.63 | 0.28 | 0.59 | 0.74 | 0.077 | 8 |
| Crivo-150M | 0.15B | ettin-reranker-150m | **CC-BY-NC** | 22.2 | 0.46 | 0.83 | 0.73 | 0.85 | 0.92 | 0.144 | 13 |
| vLLM-SR d3-edge | 0.6B | Qwen3.5-0.8B | Apache-2.0 | 20.2 | 0.41 | 0.56 | 0.36 | 0.81 | 0.92 | 0.141 | 19 |
| **tasksource-decider-nano v4** | **0.15B** | ettin-reranker-150m (ModernBERT-style encoder) | **Apache-2.0** | 17.6 | 0.46 | 0.76 | 0.69 | **0.84** | 0.87 | 0.069 | **14** |
| DecisionTune 1.0 | 0.4B | ModernBERT-large | Apache-2.0 | 16.7 | 0.30 | 0.82 | 0.66 | 0.65 | 0.92 | 0.148 | 14 |
| Kev 0.8B | 0.87B | Qwen3.5-0.8B | Apache-2.0 | 13.7 | 0.34 | 0.41 | 0.46 | 0.51 | 0.15 | 0.040 | 45 |
| GLiNER2.5-Decide (Fastino) | 0.49B | GLiNER2 | Apache-2.0 | 9.2 | 0.31 | 0.22 | 0.25 | 0.51 | 0.60 | 0.088 | 23 |
| **LiquidAI d1-omni-600M** | 0.59B | LFM2.5-Encoder-350M | LFM1.0 | **9.1** | 0.36 | 0.66 | **0.02** | 0.60 | 0.85 | 0.128 | 11 |
| DM-JEPA 1.1 | 0.15B | ModernBERT-base | MIT | 4.1 | 0.99 (!) | 0.29 | 0.20 | 0.50 | 0.93 | 0.445 | 63 |

Notes on the table:
- **Sub-scores**: dashes mean the extract was not printed for that row. All values are in the CSV.
- **Metrics**: RAGTruth is F1 on the hallucinated class, so a high value means the model catches unsupported claims. The others are accuracy or macro-F1.
- **Latency**: DI latencies come from the maintainers' GPU (RTX PRO 6000 class) and are not comparable to an M3.
- **Vendor latency claims**:
  - d3-lite: 13.6 ms median per text request on an MI325X, 518 req/s with vLLM-SR batching (https://huggingface.co/vllm-sr/d3-lite).
  - tasksource-decider-nano: 52 ms median on an A10 (https://huggingface.co/tasksource/tasksource-decider-nano).

**Key structural fact:** tasksource-decider-nano is a joint **cross-encoder**.
- Each option is scored as a reranker pair: `(state, "Question: …\nOption: …")`.
- The score comes from a linear head on the mean of the option's hidden states, softmaxed over the options.
- A tree attention mask lets the state be encoded once per request.
- Training mix: 480k requests, 42% from `tasksource/tasksource-jev-typed-decisions` (658 tasks) plus public train splits (ANLI, HoVer, RAGTruth, ContractNLI, CLINC150 and others). A 6-gram firewall removes overlap with the DI test set.
- In other words, the winning sub-200M recipe is an ordinary NLI/reranker cross-encoder trained on typed decisions. That is exactly the shape kleos can run in Rust.

## 6. Adjacent categories that solve the same problem

### 6a. NLI cross-encoders (contradiction, supersede-as-entailment, duplicate-as-bidirectional-entailment)

| Model | Params | Licence | Accuracy | Speed |
|---|---|---|---|---|
| MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli | 0.4B | MIT | MNLI-m 91.2; ANLI test 70.2 (r3 64.0); WANLI 77.0 | A100: 696 texts/s |
| MoritzLaurer/deberta-v3-large-zeroshot-v2.0 (also `-c` commercial-data variant) | 0.4B | MIT | 28-dataset mean F1 **0.673 zero-shot vs 0.846 few-shot (≤500/class)** | — |
| MoritzLaurer/deberta-v3-base-zeroshot-v2.0 | 0.2B | MIT | 0.643 zero-shot vs 0.834 few-shot | — |
| tasksource/deberta-small-long-nli | about 0.1B (small) | Apache-2.0 | ANLI 57.2/46.1/47.2; doc-nli 75.0 | — |
| cross-encoder/nli-deberta-v3-base / -small / -xsmall | 0.2B / 0.14B / 0.07B | Apache-2.0 | MNLI-mm 90.04 / 87.55 / 87.77 | — |
| tasksource/ModernBERT-large-nli | 0.4B | Apache-2.0 | MNLI 0.89; ANLI 0.72/0.54/0.55 | — |
| tasksource/ModernBERT-base-nli | 0.15B | Apache-2.0 | MNLI 0.87; ANLI 0.66/0.49/0.44 | — |
| dleemiller/finecat-nli-l (v2-l) | 0.4B | UNCERTAIN | ANLI R3 54.3 (v2: 57.3); SNLI 93.3 | 539/s vs 455/s for DeBERTa-large; 43% less memory |
| dleemiller/ModernCE-large-nli | 0.4B | MIT | MNLI-mm about 92 (the card's own table disagrees, UNCERTAIN) | 541/s, 8k context |
| Vectara HHEM-2.1-open (hallucination / "is B supported by A") | 0.1B (flan-t5-base) | Apache-2.0 | Balanced accuracy: AggreFact-SOTA 76.6, RAGTruth-QA 74.3 | About 1.5 s per 2k tokens on CPU; RAM under 600 MB |
| MiniCheck-Flan-T5-L / FactCG-DeBERTa-L | 0.8B / 0.4B | MIT / UNCERTAIN | LLM-AggreFact 75.0 / 75.6 (Bespoke-MiniCheck-7B scores 77.4 but is CC-BY-NC) | — |
| LettuceDetect (ModernBERT) | 0.15B / 0.4B | MIT | RAGTruth example-F1 79.2 (large) | — |

Sources:
- https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli ; https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 ; https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0
- https://huggingface.co/tasksource/deberta-small-long-nli ; https://huggingface.co/tasksource/ModernBERT-large-nli ; https://huggingface.co/tasksource/ModernBERT-base-nli
- https://www.sbert.net/docs/cross_encoder/pretrained_models.html
- https://huggingface.co/dleemiller/finecat-nli-l ; https://huggingface.co/dleemiller/ModernCE-large-nli
- https://huggingface.co/vectara/hallucination_evaluation_model ; https://llm-aggrefact.github.io/ ; https://huggingface.co/KRLabsOrg/lettucedect-base-modernbert-en-v1

Encoder backbones:
- **Ettin** (17M, 32M, 68M, 150M, 400M, 1B; 8k context) claims to beat ModernBERT at every size. https://huggingface.co/blog/ettin
- The ModernBERT NLI models trail DeBERTa-v3-large on ANLI but are faster and use less memory. https://huggingface.co/dleemiller/finecat-nli-l

### 6b. Zero-shot classifiers

- **GLiClass** (Knowledgator; https://arxiv.org/abs/2508.07662)
  - v3 large: avg F1 0.719, Apache-2.0. For comparison, deberta-v3-large-zeroshot scores 0.682.
  - Speed on A6000 at batch 1: 44–97 examples/s, versus 6–11/s for the cross-encoders. Vendor numbers.
  - The card contradicts itself on modern-large (0.620 vs 0.608).
  - [DI] a GLiClass-based entrant ("Verdict") scored 0. Weak at typed decisions.
  - https://huggingface.co/knowledgator/gliclass-modern-large-v3.0
- **Fastino GLiNER2** (https://arxiv.org/abs/2507.18546; EMNLP 2025 demo; Apache-2.0)
  - 205M per the paper; the card also says 340M and 0.5B (UNCERTAIN).
  - CPU classification: 130 ms for 5 labels, 208 ms for 50. DeBERTa NLI takes 1.7–16.9 s on the same task.
  - Zero-shot classification average: 0.72, against GPT-4o at 0.84.
  - **Fastino TLMs** (May 2025): task-specific hosted models with a free tier of 10k requests per month and a [vendor] "99x faster" claim. https://www.businesswire.com/news/home/20250506538922/en/
  - **Pioneer** offers GLiNER2 fine-tuning. https://pioneer.ai/gliner
  - GLiNER2.5-Decide: DI 9.2. Poor at NLI-style pairs (ContractNLI 0.22).

### 6c. Rerankers as yes/no judges

| Model | Params | Licence | Accuracy | Notes |
|---|---|---|---|---|
| **Qwen3-Reranker-0.6B / 4B / 8B** | 0.6 / 4 / 8B | Apache-2.0 | MTEB-R 65.8 / 69.8 / 69.0 | Score = P("yes") from the yes/no logits; **user instruction slot**; 32k context. Effectively a promptable binary judge. https://huggingface.co/Qwen/Qwen3-Reranker-0.6B |
| mxbai-rerank-base/large-v2 | 0.5 / 1.5B (Qwen2.5) | Apache-2.0 | BEIR 55.6 / 57.5 | RL-trained. 0.67 / 0.89 s per query on A100 (NFCorpus). https://www.mixedbread.com/blog/mxbai-rerank-v2 |
| bge-reranker-v2-gemma | 2.5B | Apache-2.0 (Gemma base terms apply) | — | Yes-logit scoring; 7.2 s per query (mxbai's A100 test). https://huggingface.co/BAAI/bge-reranker-v2-gemma |
| bge-reranker-v2-m3 | 0.57B (XLM-R) | Apache-2.0 | BEIR about 53.7 | In fastembed-rs |
| gte-reranker-modernbert-base | 149M | Apache-2.0 | BEIR 56.7; 8k context | No instruction slot. https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base |
| cross-encoder/ettin-reranker-150m-v1 / 32m | 150M / 32M | — (UNCERTAIN) | — | **Base of the best tiny decision models** (Crivo, tasksource-decider-nano) |
| Contextual AI reranker v2 | 1 / 2 / 6B | not stated | Instruction-following | https://contextual.ai/blog/rerank-v2 |
| zerank-1 / zerank-2 | 4B | Apache / **CC-BY-NC (conflicting)** | — | https://huggingface.co/zeroentropy/zerank-1 |
| jina-reranker-v2 / v3 | 0.28 / 0.6B | **CC-BY-NC** | — | Not usable commercially |

### 6d. Small LLM judges, verifiers and guard models

- **General small LLMs**
  - Qwen3-0.6B, 1.7B and 4B: Apache-2.0. https://huggingface.co/Qwen/Qwen3-0.6B
  - **Qwen3.5 small** (0.8B, 2B, 4B, 9B): Apache-2.0, released about Feb–Mar 2026. Base for most decision models. https://artificialanalysis.ai/articles/qwen3-5-small-models
  - Gemma 3 270M: Gemma terms. [DI] its decision fine-tune scored 1.25.
  - Phi-4-mini: 3.8B, MIT, BoolQ 81.2. https://huggingface.co/microsoft/Phi-4-mini-instruct
  - SmolLM3-3B: Apache-2.0.
  - LFM2 0.35–2.6B: LFM1.0.
  - Used generatively, these need yes/no-logit readout plus calibration (see §8).
- **Granite Guardian 3.3-8B** (Apache-2.0)
  - Covers groundedness and hallucination, context relevance, function-call risk and **custom criteria**.
  - LLM-AggreFact balanced accuracy 0.761.
  - Smaller family members: 5B (3.2), 2B (3.1) and HAP-38M.
  - https://huggingface.co/ibm-granite/granite-guardian-3.3-8b
- **Llama Prompt Guard 2** (22M DeBERTa-xsmall / 86M mDeBERTa)
  - 19 / 92 ms on A100; English AUC 0.995 / 0.998.
  - Llama 4 licence; injection detection only.
  - It is the template for a "tiny DeBERTa head".
  - https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M
- **ShieldGemma 2B**: Yes/No logit softmax, the same mechanism as Qwen3-Reranker. https://huggingface.co/google/shieldgemma-2b
- **Qwen3Guard** 0.6/4/8B: Apache-2.0. The Stream variant has a token-classification head. https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B
- **gpt-oss-safeguard-20b**: Apache-2.0, policy supplied in the prompt, 3.6B active. https://huggingface.co/openai/gpt-oss-safeguard-20b
- **Judges**
  - **Flow-Judge-v0.1**: 3.8B, Apache-2.0, pass/fail F1 0.955. https://huggingface.co/flowaicom/Flow-Judge-v0.1
  - **Atla Selene-1-Mini**: 8B, Apache-2.0, supports yes/no, absolute and pairwise modes. https://huggingface.co/AtlaAI/Selene-1-Mini-Llama-3.1-8B
  - Prometheus 2 7B: Apache-2.0.
  - Patronus Lynx 8B and GLIDER 3.8B: **CC-BY-NC**.
  - These are good distillation teachers, too big for runtime.

### 6e. Routers (tier/route decision)

- **Arch-Router-1.5B** (https://arxiv.org/abs/2506.16655)
  - Routing accuracy 93.17 overall, against Claude-3.7 at 92.79. 51 ± 12 ms on L40S.
  - **Custom Katanemo licence**; check it before shipping.
  - Routes are defined by the user in natural language.
- **RouteLLM** (https://arxiv.org/abs/2406.18665): a binary strong/weak LLM router with over 2x cost reduction. Not a general classifier.
- **NVIDIA prompt-task-and-complexity-classifier**
  - DeBERTa-v3-base with multiple heads, 0.2B, NVIDIA Open Model License.
  - Task accuracy 0.981.
  - **A template for a multi-head DeBERTa router.**
  - https://huggingface.co/nvidia/prompt-task-and-complexity-classifier
- **Router-R1** (https://arxiv.org/abs/2506.09033) is an RL-trained LLM router.
- Decision models handle the routing benchmarks well ([DI] CLINC150): d3-nano 0.95, d3-lite 0.94, d1-omni 0.85, decider-nano 0.87.

### 6f. Reward models as importance scorers

- **Skywork-Reward-V2**
  - Qwen3 0.6B, 1.7B, 4B and 8B are Apache-2.0.
  - RewardBench2: 61.3 / 68.3 / 75.5 / 78.2.
  - These are Bradley-Terry preference scorers, not memory-worthiness scorers. They would need fine-tuning, so they are a poor fit.
  - https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B
- **RM-R1** (https://arxiv.org/abs/2505.02387) is a generative reasoning reward model; too slow.

### 6g. Embedding plus logistic head (SetFit)

- SetFit (https://arxiv.org/abs/2209.11055), average of 5 datasets:
  - 8 examples per class: **62.3**, against plain fine-tuning at 43.0.
  - 64 per class: **75.3**, against 69.7 for fine-tuning and 70.3 for T-Few 3B. Full-data fine-tuning reaches 84.8.
  - On RAFT, SetFit scores 71.3 against GPT-3 at 62.7.
  - https://huggingface.co/papers/2209.11055
- kleos already stores vectors. A logistic head over `[e1, e2, |e1−e2|, e1⊙e2]` plus cosine, time-delta and same-slot features costs microseconds. It is the cheapest first stage for duplicate detection and routing.

### 6h. Other real 2025–26 entrants

- vLLM Semantic Router "Decision 2.0" (Kai, Eos, Sol, Nox, Lux, Vega) and "d3" (edge, lite, nano, mini, flash, 27B), all Apache-2.0. https://github.com/vllm-project/semantic-router
- Perplexity pplx-decider 27B (Apache-2.0, DI 62.8).
- Bespoke Nimble 9B (CC-BY-NC).
- InternLM Intern-Decision.
- Mapika **decider** family (2B, 4B, 31B, 35B-A3B; Apache-2.0; full recipe at https://github.com/Mapika/decider). It uses one temperature per answer type: choice 1.164, **noul 1.624**, score 1.124. https://huggingface.co/Mapika/decider-2b
- Hotchpotch "bekko" (Ettin 17M, 68M and 400M).
- Supersonic Julia-1 (mmBERT-small).
- Memory-specific research: **Supersede** (https://arxiv.org/abs/2606.27472, code and data at github.com/Vrin-cloud/supersede).
  - On the LongMemEval knowledge-update subset, full context scores about 92%, while bounded self-maintained memory with gpt-5.4 scores 77%.
  - Accuracy falls from 68% to 28% as conversations grow 24x.
  - It is a source of supersede eval items.

## 7. Running in Rust (candle / ort / llama.cpp)

**candle-transformers `models/` on main** (https://github.com/huggingface/candle/tree/main/candle-transformers/src/models, read 2026-10-16) includes:
- `modernbert`, with `ModernBertForSequenceClassification`
- `debertav2`, with `DebertaV2SeqClassificationModel`
- `xlm_roberta` (sequence classification), `bert`, `distilbert`, `t5`
- `qwen2`, `qwen3`, `qwen3_moe`, `qwen3_vl` and `quantized_qwen3`
- `lfm2` and `quantized_lfm2`
- `gemma`, `gemma2`, `gemma3`, `gemma4`
- `phi3`, `granite`, `granitemoehybrid`

**It does not include**:
- Qwen3.5 or Qwen3-Next (Gated DeltaNet hybrid), so it has **no** module for any Qwen3.5-based decision model (Kev, d3, decider, OneJev, Sieve, Intern-Decision, Tev1).
- the bidirectional **LFM2.5-Encoder** (UNCERTAIN; `lfm2` is the causal decoder);
- a Qwen3-Reranker head. That one is trivial to build on `qwen3`: take the last-position logits for "yes" and "no".

**ort (ONNX Runtime; pykeio/ort 2.0.0-rc.x)**:
- Optimum exports DeBERTa-v2, ModernBERT, Qwen3, Phi3, Gemma, Granite and XLM-R. It does not export LFM2. https://huggingface.co/docs/optimum-onnx/onnx/overview
- Existing ONNX repos: `onnx-community/Qwen3-Reranker-0.6B-ONNX`, `onnx-community/LFM2-350M-ONNX` and `onnx-community/LFM2-1.2B-ONNX`.
- **fastembed-rs** rerankers: bge-reranker-base, bge-reranker-v2-m3 and jina-v1-turbo/v2. No Qwen3 reranker and no NLI models. https://docs.rs/fastembed/latest/fastembed/enum.RerankerModel.html

**llama.cpp**:
- **Mainline has a `/v1/systemone` server API**: PR #29818, merged 2026-10-02 (https://github.com/ggml-org/llama.cpp/pull/29818). It supports laya, julia-1, lev, openjev and kev.
- Follow-up PRs added d1-3B (#30110), d1-omni-600M (#30114), clef (#29831), nimble (#29844), vLLM Decision 2 (#30378) and **vLLM d3** (#30383).
- Architectures include `qwen35`, `lfm2` and `modern-bert`; ModernBERT reranker pooling was added in #29627.
- The decision head is in libllama's graph, but the prompt layout and readout live in the **server** (`tools/server/server-decision.cpp`, 681 lines).
- So, from Rust via `llama-cpp-2` (0.1.159, 2026-10-06; https://github.com/utilityai/llama-cpp-rs), you either port that readout or run `llama-server` as a sidecar. Whether llama-cpp-2's pinned llama.cpp includes #29818 is **UNCERTAIN**.
- Open issues: answers change under batch splitting for non-causal prompts (#30402), and errors occur when parallel questions share a prefix (#30482).

| Architecture | candle | ort / ONNX | llama.cpp |
|---|---|---|---|
| ModernBERT / Ettin (decider-nano, Crivo, gte-reranker-modernbert, tasksource NLI) | **Yes** (ModernBERT; Ettin uses the ModernBERT architecture, which needs verification) | Yes | Yes (`modern-bert`) |
| DeBERTa-v2/v3 (MoritzLaurer NLI, Prompt Guard, NVIDIA router) | **Yes** | Yes | No |
| Qwen3 (Qwen3-Reranker, Qwen3-0.6B, Kev gen-1) | **Yes** | Yes | Yes |
| Qwen3.5 (Kev 1.0, d3, decider, OneJev) | **No** | UNCERTAIN | Yes (+ systemone) |
| LFM2 decoder | Yes | Via Liquid's exporter | Yes |
| LFM2.5-VL (d1-3B) / LFM2.5-Encoder (d1-omni) | No / UNCERTAIN | UNCERTAIN | Yes (+ systemone) |
| T5 (HHEM, MiniCheck-Flan-T5) | Yes (`t5`); HHEM has a custom head | Yes | No |

## 8. Methodology: how to use and adapt these models

1. **Prompt and readout format**
   - For decoders, use **logit readout, not generated text**.
   - Qwen3-Reranker style: `<Instruct>/<Query>/<Document>` with P(yes) = softmax(logit_yes, logit_no). https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
   - ShieldGemma and bge-gemma work the same way.
   - For decision models, use the `/v1/systemone` schema, with one **narrow, atomic** question per property.
     - TypeSafe advises "Ask the most explicit, narrow, specific, atomic questions" and "keep instructions and criteria aligned"; inverted mappings perform worse. https://docs.typesafe.ai/llms-full.txt
   - For kleos supersede, ask "Do A and B state a value for the same attribute of the same subject?" plus "Do the values differ?". Then decide recency in Datalog from valid-time. Do not ask "does B supersede A?", which asks the model to reason about dates; all of these models are weak at dates (Jev jaggedness page; Kev README).
2. **Option-order bias**
   - Jev leans toward the first option (jaggedness page).
   - Kev ships `/permute` because option order changes answers.
   - For Choice decisions, average over 2–4 option permutations, or reduce them to pairwise Noul questions.
   - For pair tasks (A,B), **symmetrize**: score both (A,B) and (B,A), which is needed for duplicate and same-entity.
3. **Calibration**
   - **Temperature scaling** (https://arxiv.org/abs/1706.04599) is one parameter fit by NLL on held-out data.
     - Kev-9B: ECE 0.103 to 0.041.
     - Decider-2B fits **per-type** temperatures and yes/no needs the most (T = 1.62), so raw Noul probabilities are overconfident.
     - Temperature scaling cannot change rankings, so AUROC is unaffected. Use Platt or isotonic scaling if base rates differ.
   - **Prior correction / contextual calibration** (https://arxiv.org/abs/2102.09690) gave up to +30 points with GPT-3.
     - Score a content-free input such as "N/A", then divide out the label prior.
     - Directly relevant to d1-omni's "always supported" bias.
     - **Batch calibration** (https://arxiv.org/abs/2309.17249) estimates the prior from the unlabeled batch, which suits kleos's bulk ingestion.
   - Verbalized confidence beats token probabilities only for large RLHF chat models (https://arxiv.org/abs/2305.14975). For small local models, use logits plus post-hoc calibration.
   - Report AUROC (ranking) **and** ECE or Brier after calibration. Pick per-decision thresholds with a review or abstain band, as TypeSafe recommends: 0.9 for destructive actions such as supersede-delete, 0.5 for gating.
4. **Few-shot**
   - In-context examples can be placed in the `state` for decision models.
   - For cross-encoders they do not apply; fine-tune instead.
   - Few-shot behaviour of sub-2B models is fragile with respect to format and order (Zhao et al.). There is no robust published sub-2B few-shot curve (not found).
5. **Fine-tuning beats zero-shot on narrow judgements**
   - **deberta-v3-large-zeroshot-v2.0**: 0.673 zero-shot vs **0.846** with ≤500 examples per class (+17 F1 over 28 tasks). https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0
   - **Entity matching** (Peeters, Steiner and Bizer; https://arxiv.org/abs/2409.08185):
     - Llama-3.1-8B zero-shot to fine-tuned: **+17.3 F1 on average** (A-B 56.6 to 87.3; D-S 67.7 to 92.9).
     - GPT-4o-mini: +11.7.
     - **But** small-model fine-tunes transfer badly across domains, about −11 F1 from product to scholarly data. Train on kleos's own entity types.
   - **LoRA Land** (https://arxiv.org/abs/2405.00732): 310 LoRA fine-tunes beat GPT-4 by about 10 points on average.
   - **Bucher & Martini 2024** (https://arxiv.org/abs/2406.08660): fine-tuned BERT-class models beat zero-shot GPT-4 and Claude in every case tested.
   - SetFit numbers are in §6g.
   - The DI itself shows that recipe and data matter more than size. A 150M cross-encoder trained on typed decisions (decider-nano, 17.6) beats d1-omni at 600M (9.1) and Kev-0.8B (13.7).
6. **Data volume**
   - SetFit is competitive at 8–64 examples per class.
   - The MoritzLaurer few-shot results use ≤500 per class.
   - Kev trains its LoRA on 10k mixed examples; decider-nano uses 480k.
   - No published LoRA learning curve at 100–500 examples for this task family was found. Plan an in-house curve at 100, 300, 1,000 and 3,000.
7. **Distilling from a large LLM judge**
   - Distilling step-by-step: a 770M T5 beat 540B PaLM few-shot. https://arxiv.org/abs/2305.02301
   - Classifiers trained on GPT-4 labels match human-label training, with a median F1 gap of 0.006. They are better on recall and worse on precision; validate on a human-labelled slice. https://arxiv.org/abs/2406.17633
   - Mapika decider keeps teacher-written questions only when **two further independent answers agreed**. That is a good label-noise filter to copy.
   - Recommended loop for kleos:
     1. Mine candidate pairs from real kleos stores, using the vector neighbours you would actually query.
     2. Label them with a strong judge (Claude or GPT, or Jev at $0.04 per million tokens) plus an agreement filter.
     3. Fine-tune a 150M–400M cross-encoder with a soft-CE head.
     4. Fit a temperature per decision type.
     5. Evaluate on a human-labelled set of at least 300 items per decision.
   - The current 66-item set is too small: the AUROC 95% CI is about ±0.1.
8. **Do not trust self-reported DI numbers.** Use the private-split "full" score; public-split scores are inflated (DM-JEPA public 19.2 vs full 4.1). Re-measure on kleos data.

## 9. Comparison table (model / product → kleos fit)

Fit legend: ●●● strong candidate, ●● viable, ● weak, ✗ unsuitable.

The fit columns are **Sup** = supersede, **Ent** = same entity, **Dup** = duplicate, **Imp** = worth remembering / importance, **Contra** = contradiction, **Route** = tier / route.

| Model / product | Maker | Size | Licence | Local / API | Tasks | Latency | Reported accuracy (source) | candle / ort | Sup | Ent | Dup | Imp | Contra | Route |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev 1.13/1.14 | TypeSafe AI | undisclosed | API ToS | API only | noul / choice / score | 70–500 ms [vendor]; 196 ms [DI] | DI 60.1–61.5; workflow 67.8% (evals.typesafe.ai) | ✗ / ✗ | ●● (teacher) | ●● | ●● | ●●● (teacher) | ●● | ●● |
| OpenAI Decisions API | OpenAI | gpt-6-luna | API ToS | API | predicate / choice / score | "10x faster" [vendor] | none independent | ✗ | ● teacher | ● | ● | ● | ● | ● |
| Kev 0.8B / 4B / 9B / 27B | Jared Palmer | 0.87–27.8B | Apache-2.0 | Local (PyTorch, MLX, llama.cpp) | same API | 4B 18 ms H100, 721 ms M5 (5 q) | DI 13.7 / 39.5 / 43.3 / 58.8 | ✗ (Qwen3.5) / ? ; llama.cpp ✓ | ● | ●● (4B) | ● | ●● | ● | ●● |
| d1-omni-600M | Liquid AI | 0.59B | LFM1.0 ($10M cap) | Local + GGUF/MLX | same + image/audio | 10.9 ms [DI] | DI 9.1; RAGTruth 0.02 | ✗ / ? ; llama.cpp ✓ | ✗ | ● | ● | ● | ✗ | ●● |
| d1-3B | Liquid AI | 3.1B | LFM1.0 | Local + hosted d1 ($0.04/M) | same + image | 8 ms 4090, 30 ms M5 Pro [vendor] | DI 38.9; ContractNLI 0.81 | ✗ / ? ; llama.cpp ✓ | ●● | ●● | ●● | ●● | ●● | ●●● |
| vLLM-SR d3-lite / d3-edge / d3-mini | vLLM Semantic Router | 0.85 / 0.6 / 4.5B | Apache-2.0 | Local; llama.cpp | same + image/video | d3-lite 13.6 ms MI325X [vendor]; 37 ms [DI] | DI 32.3 / 20.2 / 52.6 | ✗ (Qwen3.5) ; llama.cpp ✓ | ●● (mini) | ●● | ●● | ●●● | ●● (mini) | ●●● |
| OneJev1.5-0.8B | OmniJev | 1.1B | Apache-2.0 | Local (transformers) | same | 83 ms [DI] | DI 29.0; ContractNLI 0.82, RAGTruth 0.77, ECE 0.035 | ✗ (Qwen3.5) | ●● | ●● | ●● | ●● | ●●● | ●● |
| tasksource-decider-nano v4 | tasksource | 0.15B | Apache-2.0 | Local (PyTorch) | choice / noul | 14 ms [DI]; 52 ms A10 [vendor] | DI 17.6; HoVer 0.84, ContractNLI 0.76, RAGTruth 0.69 | **candle modernbert (likely; custom head and mask to port) / ort ✓** | ●● (fine-tune base) | ●● | ●●● | ●● | ●●● | ●● |
| Crivo-150M | tardellirs | 0.15B | **CC-BY-NC** | Local | choice / noul | 13 ms | DI 22.2 | candle modernbert | ✗ licence | | | | | |
| DM-JEPA 1.1 / 1.5 / 1.7 | Danger Labs | 0.09–0.35B | MIT / custom / Apache | Local | — | 63 ms [DI] | DI 4.05; 100% claims not credible | — | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| DeBERTa-v3-large mnli-fever-anli-ling-wanli | MoritzLaurer | 0.4B | MIT | Local | 3-way NLI | about 2 ms/text A100 | ANLI 70.2, MNLI 91.2 | **candle debertav2 ✓ / ort ✓** | ●● | ● | ●● (bi-entail) | ✗ | ●●● | ✗ |
| deberta-v3-base/large-zeroshot-v2.0 (-c) | MoritzLaurer | 0.2 / 0.4B | MIT | Local | binary NLI zero-shot classification | — | 0.643–0.673 zero-shot; 0.83–0.85 few-shot | candle ✓ / ort ✓ | ●● (fine-tune) | ●● | ●● | ●● | ●●● | ●● |
| tasksource ModernBERT-base/large-nli; deberta-small-long-nli | tasksource | 0.1–0.4B | Apache-2.0 | Local | NLI, long context | — | ANLI r1 0.66/0.72; small-long doc-nli 75.0 | candle ✓ / ort ✓ | ●● | ● | ●● | ✗ | ●●● | ✗ |
| HHEM-2.1-open | Vectara | 0.1B | Apache-2.0 | Local CPU | supported? | 1.5 s per 2k tokens CPU | AggreFact 76.6 | candle t5 (custom head) / ort ✓ | ● | ✗ | ✗ | ✗ | ●● | ✗ |
| Qwen3-Reranker-0.6B / 4B | Qwen | 0.6 / 4B | Apache-2.0 | Local | instructed yes/no | — (0.6B decoder; about tens of ms on GPU, UNCERTAIN) | MTEB-R 65.8 / 69.8 | **candle qwen3 ✓ (DIY head) / ort ✓ (onnx-community)** | ●● | ●●● | ●●● | ● | ●● | ● |
| gte-reranker-modernbert-base | Alibaba | 149M | Apache-2.0 | Local | relevance | — | BEIR 56.7 | candle modernbert ✓ / ort ✓ | ● | ●● | ●●● | ✗ | ✗ | ✗ |
| mxbai-rerank-v2 base | Mixedbread | 0.5B | Apache-2.0 | Local | relevance | 0.67 s/query A100 (batch) | BEIR 55.6 | candle qwen2 ✓ | ● | ●● | ●● | ✗ | ✗ | ✗ |
| GLiClass v3 / GLiNER2 | Knowledgator / Fastino | 0.15–0.4B | Apache-2.0 | Local | multi-label zero-shot | 130–210 ms CPU (GLiNER2) | F1 0.72 / 0.72 avg | custom arch (DeBERTa/ModernBERT trunk) | ✗ | ● | ✗ | ●● | ✗ | ●● |
| Granite Guardian 3.3 | IBM | 8B (also 2B/5B older) | Apache-2.0 | Local | groundedness, custom criteria | — | AggreFact 0.761 | candle granite ✓ | ● | ✗ | ✗ | ● | ●● (teacher) | ✗ |
| Arch-Router-1.5B | Katanemo | 1.5B | **custom** | Local | route choice | 51 ms L40S | 93.17 routing | candle qwen2 ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ●● (licence check) |
| NVIDIA prompt-task-complexity | NVIDIA | 0.2B | NVIDIA Open Model | Local | multi-head classification | — | task 0.981 | candle debertav2 ✓ | ✗ | ✗ | ✗ | ● | ✗ | ●● template |
| Skywork-Reward-V2-Qwen3-0.6B | Skywork | 0.6B | Apache-2.0 | Local | scalar preference | — | RB2 61.3 | candle qwen3 ✓ | ✗ | ✗ | ✗ | ● | ✗ | ✗ |
| SetFit / logistic head on kleos embeddings | HF / DIY | embedder + about 1k params | Apache-2.0 | In-process | any closed label set | microseconds (head) | 8/class 62.3; 64/class 75.3 (5-dataset average) | native Rust | ● (feature model) | ●● (blocking) | ●●● (first stage) | ●●● | ● | ●●● |

## 10. Recommendations per kleos decision

The general architecture is a two-stage cascade:

```
cheap features + logistic head (µs)
  → small cross-encoder (10–40 ms)
  → optional large-model/API escalation for the uncertain band
```

All thresholds are per-decision and calibrated. Dates, numbers and counting are done in Datalog.

1. **Does a new fact supersede an old one?**
   1. **Fine-tuned ModernBERT/Ettin-150M cross-encoder**, started from `tasksource-decider-nano` (Apache-2.0, HoVer 0.84, ContractNLI 0.76), or **DeBERTa-v3-base/large NLI** (MIT).
      - Ask "same subject and attribute?" and "different value?". Recency comes from valid-time in Datalog.
      - Train on about 1–3k LLM-distilled pairs, including LongMemEval knowledge-update and Supersede-derived items plus kleos traces.
      - candle has both architectures.
   2. **Zero-shot baseline to beat**: OneJev1.5-0.8B (ContractNLI 0.82, ECE 0.035) or vLLM-SR d3-mini 4B (DI 52.6), both Apache-2.0, through a llama.cpp sidecar. Use these to measure how much fine-tuning adds; they also serve as an offline teacher.
   3. **Qwen3-Reranker-0.6B** with a supersede instruction, read as P(yes) (candle `qwen3` plus about 20 lines). This is the best promptable option that runs in candle.
2. **Are two mentions the same entity?**
   1. **Qwen3-Reranker-0.6B** with an instruction ("Do these refer to the same real-world entity?"), symmetrized. Apache-2.0; candle and ONNX both work.
   2. A **fine-tuned Ettin/ModernBERT cross-encoder** on kleos's own entity types. The entity-matching literature shows +11–17 F1 from fine-tuning but poor cross-domain transfer.
   3. A **logistic head** over embedding cosine, name string features (Jaro-Winkler, token overlap) and graph-neighbour overlap as the blocking stage. Our existing d1-omni AUROC of 0.83 is a floor to beat.
3. **Is a memory a duplicate?**
   1. **Embedding cosine plus logistic head**, with no model call above or below confident thresholds.
   2. **gte-reranker-modernbert-base** or **Qwen3-Reranker-0.6B** for the grey band.
   3. **Bidirectional NLI** with tasksource ModernBERT-base-nli or DeBERTa-v3-base: entailment both ways means duplicate, one way means subsumption.
4. **Is this worth remembering, and how important is it?**
   1. A **SetFit-style head on kleos embeddings**, trained on LLM-distilled labels (Score levels 0–4). It is microseconds per item, and the domain is subjective, so it must be learned from your own policy.
   2. **vLLM-SR d3-lite 0.8B** or **d1-3B** (licence permitting) as a zero-shot Score/Noul gate. This mirrors TypeSafe's own memory cookbook, whose gate is "important lasting fact?" at P ≥ 0.5.
   3. A fine-tuned **deberta-v3-base-zeroshot-v2.0-c** with an "important" hypothesis template. MIT and commercial-data clean.
5. **Does this contradict what we already know?**
   1. **MoritzLaurer DeBERTa-v3-large-mnli-fever-anli-ling-wanli**: MIT, ANLI 70.2, the best-documented 3-way NLI, candle `debertav2`. For speed, use `tasksource/deberta-small-long-nli` or `ModernBERT-base-nli` (Apache-2.0).
   2. **tasksource-decider-nano**, fine-tuned (Apache-2.0, 150M, 14 ms).
   3. **OneJev1.5-0.8B** as a stronger zero-shot or teacher option (RAGTruth 0.77, ContractNLI 0.82).
   - Avoid d1-omni here: its RAGTruth F1 of 0.02 means it almost never flags unsupported or contradicting content.
6. **Which memory tier or route?**
   1. **Logistic or SetFit head on embeddings plus metadata features.** Routes are a closed label set you own, and this is trivially fast.
   2. **vLLM-SR d3-lite / d3-nano** (CLINC150 0.94 / 0.95; Apache-2.0) via a llama.cpp `/v1/systemone` sidecar, if natural-language route descriptions must stay editable without retraining.
   3. A **multi-head DeBERTa classifier** following NVIDIA's prompt-task-classifier template, so one forward pass gives tier, importance and type.

**Licence traps**: the LFM1.0 $10M cap (all d1 and LFM2); CC-BY-NC (Crivo, CoCo-Decision, Bespoke Nimble, jina rerankers, Lynx, GLIDER, zerank-2 per the official card); custom licences (Arch-Router, DM-JEPA 1.5, Llama Prompt Guard). The Gemma and Llama bases carry their own terms.

**Most useful next experiment** (no new research needed):
1. Expand the 66 items to about 300 per decision using an LLM teacher plus human review.
2. Benchmark, zero-shot:
   - Qwen3-Reranker-0.6B (candle);
   - DeBERTa-v3-large NLI (candle);
   - tasksource-decider-nano (PyTorch first);
   - OneJev1.5-0.8B and d3-lite (llama.cpp sidecar).
3. Fine-tune the best encoder at 100, 300 and 1,000 labels and record the learning curve.
4. Report AUROC, ECE after temperature scaling, and M3 latency.
