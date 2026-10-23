# Memory benchmarks: LongMemEval and LoCoMo (UPGRADE_PLAN L2)

Status: first numbers, 2026-10-23. Harness: `scripts/bench-l2/`. Two builds:

- **`276f90f`** (main before scoring): the "before" build. Its model variants ran on **CPU**, because that build cannot compile candle's Metal kernels when called from this Python (see Device).
- **`6f9711f`** (main with recency/importance/TTL scoring, entity wiring and the Metal fix): the "after" build, with models on **Metal**.

Retrieval is the same on both builds wherever both ran. That covers every LongMemEval_S and LoCoMo row, and both oracle-KU tables. The tables are therefore from `6f9711f` unless marked otherwise. Full per-split tables are in `.bench-data/bench-l2/results/<rev>/report.md`.

## Results

### What ran

| | LongMemEval_S (500 q) | LongMemEval oracle file | LoCoMo (10 conv, 1,986 q) |
|---|---|---|---|
| `fts` | **all 500** | all 500 (sanity) | **all 10 / 1,986** |
| `hybrid` (EmbeddingGemma) | **119**: the 25 held-out KU plus a stratified 99 (6f9711f, Metal); 51 on 276f90f (CPU, stopped early) | the 78 KU questions | **all 10 / 1,986** |
| `sup` (+ decider-nano) | not run in the full haystack (see below) | the **25 held-out KU** questions | not run |
| QA (GLM-4.7-Flash) | all 500 for `none` / `oracle` / `kleos_fts`; a stratified 99 (+ the hybrid set) for `full`, `kleos_hybrid` | the 25 held-out KU, every variant | stratified **301** of the 1,540 in categories 1–4, all conditions |

**Why `sup` did not run in the LongMemEval_S haystack.**
- On 276f90f, on CPU, `remember` with the decider took 9–20 s per call: about 1 h per question.
- On Metal (6f9711f) it takes about 1 s per call, or 4 min per question. Even so, the coordinator's sentence-level claim splitting will replace the whole-turn `sup` numbers, so the turn-level `sup` variant was measured only where it is cheap and direct: the oracle-file KU questions. Their store holds the old and the new value and little else.

### Retrieval (LLM-free, the headline)

**LongMemEval_S, `fts`, all 500 questions** (unit = user message, k = memories returned by `recall`):

| split | n | R@5 | R@10 | Rall@5 | Rall@10 | T@5 | T@10 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| all | 500 | 0.954 | 0.960 | 0.780 | 0.850 | 0.897 | 0.925 | 0.887 |
| single-session | 150 | 0.927 | 0.933 | 0.927 | 0.933 | 0.869 | 0.879 | 0.909 |
| · ss-user | 64 | 0.984 | 0.984 | 0.984 | 0.984 | 0.953 | 0.969 | 0.984 |
| · ss-assistant | 56 | 0.946 | 0.946 | 0.946 | 0.946 | 0.800 | 0.800 | 0.946 |
| · ss-preference | 30 | 0.767 | 0.800 | 0.767 | 0.800 | 0.700 | 0.700 | 0.680 |
| multi-session | 121 | 0.975 | 0.975 | 0.587 | 0.736 | 0.901 | 0.934 | 0.866 |
| temporal-reasoning | 127 | 0.953 | 0.969 | 0.748 | 0.827 | 0.890 | 0.913 | 0.867 |
| knowledge-update | 72 | 1.000 | 1.000 | 0.944 | 0.972 | 0.958 | 1.000 | 0.958 |
| abstention | 30 | 0.900 | 0.900 | 0.567 | 0.700 | 0.778 | 0.889 | 0.768 |
| KU held-out | 25 | 1.000 | 1.000 | 0.960 | 1.000 | 1.000 | 1.000 | 0.985 |
| KU trained-on | 53 | 1.000 | 1.000 | 0.925 | 0.962 | 0.939 | 1.000 | 0.942 |

**LongMemEval_S, `fts` vs `hybrid`, paired on the same questions:**

119 questions: the 25 held-out KU plus a stratified 99, so KU is over-represented in "all".

| split | n | `fts` R@5 | R@10 | Rall@10 | T@10 | NDCG@10 | `hybrid` R@5 | R@10 | Rall@10 | T@10 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 119 | 0.966 | 0.966 | 0.874 | 0.910 | 0.901 | **0.992** | **1.000** | **0.941** | **0.970** | **0.950** |
| single-session | 30 | 0.967 | 0.967 | 0.967 | 0.889 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 0.988 |
| multi-session | 24 | 0.958 | 0.958 | 0.667 | 0.875 | 0.814 | 1.000 | 1.000 | 0.875 | 0.958 | 0.921 |
| temporal-reasoning | 25 | 0.960 | 0.960 | 0.800 | 0.880 | 0.844 | 0.960 | 1.000 | 0.840 | 0.920 | 0.889 |
| knowledge-update | 32 | 1.000 | 1.000 | 1.000 | 1.000 | 0.982 | 1.000 | 1.000 | 1.000 | 1.000 | 0.994 |
| abstention | 8 | 0.875 | 0.875 | 0.875 | – | 0.767 | 1.000 | 1.000 | 1.000 | – | 0.910 |

LongMemEval_S is lexically easy at session level. Most of the haystack is unrelated ShareGPT/UltraChat filler, so BM25 alone finds an answer session in the top 5 for 95% of questions. The harder numbers are Rall (every evidence session) and T (the evidence turn itself). Hybrid adds most there, for example multi-session Rall@10 rises from 0.67 to 0.88.

**Knowledge update: does a stale value outrank the current one?** These are oracle-file KU questions: the store holds only the evidence sessions, so both the old and the new value. There are 23 non-abstention questions among the 25 held out from decider training.

| variant | n | latest@5 | latest@10 | stale-first ↓ | T@5 | auto superseded / q | pending / q | QA (25 q) |
|---|---|---|---|---|---|---|---|---|
| `fts` | 23 | 1.000 | 1.000 | 0.565 | 1.000 | 0.00 | 0 | 0.880 |
| `hybrid` | 23 | 1.000 | 1.000 | 0.565 | 1.000 | 0.00 | 0 | 0.880 (0.920 on 276f90f) |
| `sup` (default read, `hide_pending` off) | 23 | 1.000 | 1.000 | 0.565 | 1.000 | **0.00** | 3.4 | 0.880 (0.920) |
| `sup` + `hide_pending` | 23 | 1.000 | 1.000 | **0.130** | 0.913 | 0.00 | 3.4 | **0.760** |
| `oracle` context (the sessions in full) | | | | | | | | 0.840 |

What the table shows:
- **decider-nano auto-superseded nothing on raw chat turns.** Facts in LongMemEval are buried in multi-sentence requests ("…I recently set a personal best time in a charity 5K run with a time of 27:12. Do you have any tips on…"). The `supersedes` probability lands in the review band below τ = 0.997, so every decision is pending.
- **`hide_pending` acts on those pending decisions.** It cuts stale-first from 0.57 to 0.13.
- **It also hides correct memories.** Some pending decisions are wrong: cross-topic pairs (coffee beans vs a tea recommendation), and in one question the *latest* value ("…just finished my fifth issue"). Other questions need the history ("What was my *previous* personal best?"). So T@5 drops to 0.913 and QA drops 12 points.
- **The non-destructive default is the right default.** A dated reader handles visible history well.
- **On the 53 trained-on KU questions** (`fts` and `hybrid` only, so no decider is involved), stale-first is 0.57–0.63.
- Recency at the default weight (0.03, 90-day half-life) does not change stale-first when the read is at the question's date (`fts@q`: 0.522, the same as `fts` on LongMemEval_S KU).

**LoCoMo, all 10 conversations, every question** (turn level; category ids as in `locomo10.json`):

| split | n | `fts` R@5 | R@10 | Rall@10 | NDCG@10 | `hybrid` R@5 | R@10 | Rall@10 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| **cat 1–4** | 1,536 | 0.554 | 0.633 | 0.518 | 0.435 | **0.661** | **0.749** | **0.612** | **0.516** |
| 1 multi-hop | 282 | 0.426 | 0.543 | 0.124 | 0.241 | 0.585 | 0.720 | 0.195 | 0.330 |
| 2 temporal | 321 | 0.670 | 0.707 | 0.654 | 0.535 | 0.720 | 0.788 | 0.726 | 0.596 |
| 3 open-domain | 92 | 0.304 | 0.413 | 0.228 | 0.215 | 0.467 | 0.543 | 0.337 | 0.310 |
| 4 single-hop | 841 | 0.580 | 0.660 | 0.629 | 0.485 | 0.685 | 0.767 | 0.738 | 0.570 |
| 5 adversarial | 446 | 0.578 | 0.655 | 0.646 | 0.477 | 0.489 | 0.574 | 0.565 | 0.414 |

(Four questions have no parseable evidence id and are skipped. On 276f90f the numbers are the same to ±0.007.)

### QA accuracy (secondary; local to this harness)

**Reader and judge are both `zai-org/GLM-4.7-Flash` with thinking off.** These numbers are not comparable with GPT-4o-judged leaderboards. Compare columns, never with published scores.

**LongMemEval_S, all 500** (`kleos_fts` = the top-10 user messages from `recall`):

| split | n | no memory | **kleos_fts** | oracle sessions |
|---|---|---|---|---|
| all | 500 | 0.082 | **0.550** | 0.710 |
| · ss-user | 64 | 0.016 | 0.922 | 0.984 |
| · ss-assistant | 56 | 0.161 | 0.161 | 0.982 |
| · ss-preference | 30 | 0.000 | 0.500 | 0.800 |
| multi-session | 121 | 0.000 | 0.455 | 0.562 |
| temporal-reasoning | 127 | 0.024 | 0.465 | 0.567 |
| knowledge-update | 72 | 0.000 | 0.750 | 0.778 |
| abstention | 30 | 0.933 | 0.800 | 0.567 |
| KU held-out / trained-on | 25 / 53 | | 0.840 / 0.717 | 0.840 / 0.717 |

**LongMemEval_S, with full context (~106k tokens per question):**

QA_119 questions: the 25 held-out KU plus a stratified 99, so KU is over-represented in "all".

| split | n | `fts` R@5 | R@10 | Rall@10 | T@10 | NDCG@10 | `hybrid` R@5 | R@10 | Rall@10 | T@10 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 119 | 0.966 | 0.966 | 0.874 | 0.910 | 0.901 | **0.992** | **1.000** | **0.941** | **0.970** | **0.950** |
| single-session | 30 | 0.967 | 0.967 | 0.967 | 0.889 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 0.988 |
| multi-session | 24 | 0.958 | 0.958 | 0.667 | 0.875 | 0.814 | 1.000 | 1.000 | 0.875 | 0.958 | 0.921 |
| temporal-reasoning | 25 | 0.960 | 0.960 | 0.800 | 0.880 | 0.844 | 0.960 | 1.000 | 0.840 | 0.920 | 0.889 |
| knowledge-update | 32 | 1.000 | 1.000 | 1.000 | 1.000 | 0.982 | 1.000 | 1.000 | 1.000 | 1.000 | 0.994 |
| abstention | 8 | 0.875 | 0.875 | 0.875 | – | 0.767 | 1.000 | 1.000 | 1.000 | – | 0.910 |

LongMemEval_S is lexically easy at session level. Most of the haystack is unrelated ShareGPT/UltraChat filler, so BM25 alone finds an answer session in the top 5 for 95% of questions. The harder numbers are Rall (every evidence session) and T (the evidence turn itself). Hybrid adds most there, for example multi-session Rall@10 rises from 0.67 to 0.88.

**LoCoMo, a stratified 301 of categories 1–4** (Mem0's reader and judge prompts; GLM-4.7-Flash):

| split | n | no memory | kleos_fts | **kleos_hybrid** | full conversation (~20k tokens) | evidence turns (oracle) |
|---|---|---|---|---|---|---|
| **cat 1–4** | 301 | 0.462 | 0.645 | **0.714** | 0.814 | 0.897 |
| 1 multi-hop | 55 | 0.455 | 0.655 | 0.709 | 0.909 | 0.964 |
| 2 temporal | 63 | 0.381 | 0.397 | 0.365 | 0.603 | 0.714 |
| 3 open-domain | 19 | 0.474 | 0.474 | 0.526 | 0.737 | 0.789 |
| 4 single-hop | 164 | 0.494 | 0.756 | 0.872 | 0.872 | 0.957 |

(On 276f90f: kleos_fts 0.645 and kleos_hybrid 0.718. The other columns are cached and identical.)

**How to read these.**
- On LongMemEval, kleos top-10 retrieval with the same reader **beats putting the whole 106k-token history in the prompt**: hybrid 0.681 and fts 0.571, against 0.538 for full context. The gap is largest on multi-session (0.67 vs 0.21) and knowledge update (0.75–0.81 vs 0.53). The long context buries the facts and confuses the reader about which value is current.
- On LoCoMo, full context (about 20k tokens) still wins by about 10 points.
- The `ss-assistant` row is the cost of the unit choice: assistant replies are not stored, so kleos equals "no memory" there.
- **LoCoMo's no-memory score is 0.46.** Mem0's lenient judge ("touches on the same topic") accepts plausible guesses, as Penfield's audit found. LongMemEval's stricter judge gives 0.08 for no memory.
- **Noise.** Identical retrieval on the two builds gave 0.920 vs 0.880 on the 25-question KU set. So on sets this small, one question (4 points) is noise.

### Throughput and latency

One process, 16 GB Apple-silicon laptop. Rates include the in-process benchmark loop.

| bench | variant | device | remember p50 / p95 ms | remembers/s | recall(k=10) p50 / p95 ms |
|---|---|---|---|---|---|
| LongMemEval_S (122k remembers) | `fts` | – | 0.8 / 1.2 | ~1,190 | 1.3 / 1.9 |
| LoCoMo (5.9k) | `fts` | – | 3.7 / 8.1 | ~230 | 6.1 / 10.7 |
| LongMemEval_S | `hybrid` | Metal | 19 / 42 | ~34 | 15 / 23 |
| LoCoMo | `hybrid` | Metal | 17 / 33 | ~40 | 13 / 26 |
| LongMemEval oracle KU | `sup` | Metal | 1,138 / 2,100 | ~0.9 | 18 / 51 |
| LongMemEval_S | `hybrid` | CPU (276f90f; load 10–35) | 185 / 1,247 | ~2.4 | 92 / 478 |
| LongMemEval oracle KU | `sup` | CPU (276f90f; load 10–35) | 8,935 / 37,672 | ~0.1 | 127 / 447 |

- The fts rows ran at load average 3 (6f9711f). On 276f90f under load 10, LongMemEval fts gave 0.8 / 1.8 ms.
- **The decider dominates `remember`.** Each call makes about 15 forward passes: top-5 candidates × (`supersedes` + both orders of `duplicate`), over whole chat turns with no truncation.

### Where kleos is weak (from these runs)

1. **Supersession does not fire on real chat turns.**
   - On the held-out KU questions, decider-nano made 0 auto supersedes and about 3.4 pending decisions per question.
   - The value rule fired on only 0.13–0.45 supersedes per question (`fts`).
   - A stale value outranks the current one in **52–65%** of KU questions, whatever the variant.
   - `hide_pending` fixes the ordering, but it costs recall and QA, because the review band also flags cross-topic pairs.
   - Sentence-level claim splitting before `decide`, which the coordinator has in progress, targets exactly this.
2. **Multi-evidence questions.**
   - **Rall@10** is 0.74 on LongMemEval multi-session (fts, all 500; hybrid 0.88 on its sample) and 0.12 (fts) / 0.20 (hybrid) on LoCoMo multi-hop: the top 10 rarely covers every evidence session.
   - Recall returns near-duplicate turns from one session instead of diversifying.
3. **Preference questions** (ss-preference R@5 0.77, NDCG 0.68).
   - The question ("Can you recommend…") shares few words with the turn that states the preference.
   - Hybrid helps on LoCoMo (+0.11 R@5 overall). On the LongMemEval hybrid set only 6 preference questions ran, too few to judge.
4. **LoCoMo adversarial (category 5):** hybrid ranks the evidence *lower* than fts (0.49 vs 0.58 R@5). The questions are near-misses that the embedding matches to the wrong speaker.
5. **Embedding cost without Metal.** On CPU, EmbeddingGemma at 512 tokens per memory runs at about 2.4 remembers/s, which is too slow for bulk chat ingestion. Metal gives about 34/s.
6. **Assistant content is not stored** in this setup (a harness choice; `--unit round` exists).
7. **LongMemEval date quirk.** 44 of the 500 questions (40 of them temporal-reasoning) have an answer session dated *after* `question_date`. kleos's `as_of` correctly hides those "future" facts: reading at the question date drops R@5 from 0.954 to 0.912 and temporal-reasoning R@5 from 0.953 to 0.803. So the headline reads at "now", and `--as-of question` is not used for LongMemEval_S as released.

## After: recall diversity, context output and speaker roles (wave3/recall)

Status: 2026-10-23, branch `wave3/recall` on main 92ef789. The features are described in [memory-api.md](../specs/memory-api.md): "Diversity", "Entity expansion", "Context output", "Speaker roles" and "Valid-from filters".

**How "before" and "after" were measured.**
- Every row comes from one build of this branch and one ingest per question or conversation. Several recall option sets are applied to the same store (`retrieval.py --configs`).
- `base` is `diversity: false`, which is main's recall exactly: the same fetch, window and order. Where the main rerun (`1c1bf81`) ran the same conversation, `base` agrees with it (LoCoMo conv-26 hybrid R@5 0.373 against 0.387, R@10 0.407 against 0.400).
- `default` is this branch's defaults: one hit per statement and no source cap or MMR penalty.
- `mmr05` adds `mmr_weight: 0.05`, and `src2` adds `max_per_source: 2`.
- Builds: `fts` LongMemEval and LoCoMo ran on the first build, which fetched the index legs deeper when diversity was on. With FTS alone that changes nothing, because one leg gives the same top window. The hybrid rows ran on the final recall code (`54f5ad9`, Metal).

**Main's claims change these numbers before any recall change does.** Since 92ef789 `remember` splits a message into claims, so the hits are sentences.
- On LoCoMo that splits "Caroline: Yes, I did. It was amazing! …" into a fragment that carries only the speaker's name.
- The fragment embeds close to every "Caroline's …" question.
- So LoCoMo `hybrid` R@5 fell from **0.661** (6f9711f, whole turns) to **0.421** (main, `base`). `fts` is unaffected (0.554 → 0.536).
- The write-path fix is separate work (`wave3/claims-fix`). The LoCoMo hybrid and QA rows below are therefore measured on main's degraded store. They will move once that fix lands, and the recall changes should be re-measured on top of it.

### Retrieval

**LongMemEval_S `hybrid`, the 119 questions** (EmbeddingGemma on Metal):

| split | n | `base` Rall@5 | Rall@10 | T@10 | NDCG@10 | `default` Rall@5 | Rall@10 | T@10 | NDCG@10 | `mmr05` Rall@10 | `src2` Rall@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 119 | 0.849 | 0.950 | 0.970 | 0.947 | 0.857 | **0.958** | **0.980** | 0.950 | 0.958 | 0.975 |
| single-session | 30 | 1.000 | 1.000 | 1.000 | 0.959 | 1.000 | 1.000 | 1.000 | 0.959 | 1.000 | 1.000 |
| multi-session | 24 | 0.583 | 0.917 | 1.000 | 0.933 | 0.625 | **0.958** | 1.000 | 0.940 | 0.958 | 0.958 |
| temporal-reasoning | 25 | 0.680 | 0.840 | 0.880 | 0.891 | 0.680 | 0.840 | 0.920 | 0.901 | 0.840 | 0.920 |
| knowledge-update | 32 | 1.000 | 1.000 | 1.000 | 0.997 | 1.000 | 1.000 | 1.000 | 0.997 | 1.000 | 1.000 |

R@5 is 0.992 for every config, and ss-user is 1.000 for every config.

**LongMemEval_S `fts`, all 500:**

| split | n | `base` R@5 | Rall@5 | Rall@10 | T@10 | `default` R@5 | Rall@5 | Rall@10 | T@10 | `src2` Rall@10 | `src2` T@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 500 | 0.950 | 0.758 | 0.836 | 0.909 | 0.952 | 0.770 | **0.846** | 0.914 | 0.856 | 0.893 |
| ss-user | 64 | 0.984 | 0.984 | 0.984 | 0.953 | 0.984 | 0.984 | 0.984 | 0.969 | 0.984 | 0.938 |
| multi-session | 121 | 0.975 | 0.537 | 0.702 | 0.909 | 0.975 | 0.562 | **0.719** | 0.909 | **0.744** | 0.909 |
| temporal-reasoning | 127 | 0.953 | 0.724 | 0.803 | 0.898 | 0.953 | 0.740 | **0.819** | 0.898 | 0.835 | 0.890 |
| knowledge-update | 72 | 1.000 | 0.917 | 0.986 | 1.000 | 1.000 | 0.931 | 0.986 | 1.000 | 0.986 | 0.972 |

KU latest@5 rises from 0.931 to 0.944, and stale-first stays at 0.472. The MMR penalty with word overlap (`fts` has no vectors) is slightly worse: Rall@10 0.842 against 0.846.

**LoCoMo, all 10 conversations, categories 1–4** (1,536 questions; NDCG@10 counts an evidence turn once, see Metrics):

| variant | config | R@5 | R@10 | Rall@10 | NDCG@10 | multi-hop R@5 | multi-hop Rall@10 | single-hop R@5 |
|---|---|---|---|---|---|---|---|---|
| `fts` | base | 0.536 | 0.617 | 0.499 | 0.417 | 0.411 | 0.103 | 0.558 |
| `fts` | default | 0.538 | 0.620 | 0.502 | 0.418 | 0.411 | 0.106 | 0.560 |
| `fts` | + mmr 0.05 | 0.542 | 0.624 | 0.505 | 0.420 | 0.429 | 0.106 | 0.561 |
| `fts` | + source cap 3 | 0.532 | 0.603 | 0.482 | 0.412 | 0.408 | 0.106 | 0.553 |
| `hybrid` | base | 0.421 | 0.479 | 0.370 | 0.327 | 0.376 | 0.064 | 0.448 |
| `hybrid` | default | 0.422 | 0.482 | 0.372 | 0.328 | 0.376 | 0.064 | 0.449 |
| `hybrid` | mmr05 | 0.427 | 0.486 | 0.375 | 0.330 | 0.387 | 0.064 | 0.455 |
| `hybrid` | src2 | 0.408 | 0.446 | 0.336 | 0.314 | 0.376 | 0.053 | 0.424 |

**Oracle-file KU** (`hybrid`, 78 questions, both values stored): identical for `base` and `default`. Latest@5 is 1.000 and stale-first 0.486. Diversity does not touch the knowledge-update ordering.

**What this says.**
- **The statement cap is a safe default.** It never lowered R@5 or a single-evidence split, and it lifts multi-evidence coverage wherever claims stack:
  - LongMemEval multi-session Rall@10: +0.04 hybrid, +0.02 fts;
  - temporal Rall@10: +0.02 fts;
  - with assistant turns stored: +0.05 (below).
- **A per-session cap is not a default.** It helps LongMemEval, where evidence spreads across sessions (multi-session Rall@10 0.744 with `fts`, hybrid temporal 0.920). It hurts LoCoMo, where several evidence turns sit in one session (R@10 −0.04 hybrid). It is there for callers whose sources are sessions (`max_per_source`).
- **MMR at 0.05** is small: +0.005 on LoCoMo, ±0.004 on LongMemEval. It is left at 0.
- **LoCoMo multi-hop Rall@10 stays low** (0.06 hybrid, 0.10 fts). Diversity cannot add evidence that the search never ranks: the second hop does not share words or meaning with the question.
- **Deeper index legs were tried and dropped.** Letting diversity fetch the index legs deeper (half the pool) raised LoCoMo hybrid R@10 but cut R@5 (0.394 → 0.358 on 3 conversations). Deeper BM25 legs let the name-only fragments above into the fusion. So the diverse selection picks from the same index fetch as before.

**Entity expansion** (LoCoMo hybrid; entities from a crude capitalised-word list plus the speaker, `--entities caps`; there is no NER):

| | n | R@5 | R@10 | Rall@10 | multi-hop Rall@10 |
|---|---|---|---|---|---|
| default (10 conv) | 1,536 | 0.435 | 0.493 | 0.382 | 0.060 |
| `expand`, weight 0.5 (10 conv) | 1,536 | 0.435 | 0.492 | 0.381 | 0.060 |
| default (3 conv) | 383 | 0.371 | 0.405 | 0.319 | 0.027 |
| `expand`, weight 0.9 (3 conv) | 383 | 0.345 | 0.389 | 0.300 | 0.041 |
| `expand`, weight 1.0 (3 conv) | 383 | 0.316 | 0.368 | 0.287 | 0.041 |

- At 0.5 the expanded memories almost never reach the top 10.
- At 0.9 and above they displace better hits: multi-hop gains one question and everything else loses.
- So expansion stays off by default. It needs real entities (the GLiNER evaluation) and a relevance signal of the expanded memory's own before it can help.

**Assistant turns** (LongMemEval_S `fts`, all 500, `--unit user+assistant`): every assistant reply is stored with `role: 'assistant'` beside the user messages. That is 494 memories per question against 245.

| split | n | user only, `default` T@10 | + assistant, `base` T@10 | + assistant, `default` R@5 | Rall@10 | T@10 |
|---|---|---|---|---|---|---|
| all | 500 | 0.914 | 0.825 | 0.944 | 0.834 | 0.850 |
| ss-assistant | 56 | 0.800 | 0.911 | 1.000 | 1.000 | 0.911 |
| ss-user | 64 | 0.969 | 0.859 | 0.984 | 0.984 | 0.906 |
| multi-session | 121 | 0.909 | 0.818 | 0.975 | 0.719 | 0.835 |
| knowledge-update | 72 | 1.000 | 0.931 | 1.000 | 0.944 | 0.958 |

- The assistant's evidence turn is now found (ss-assistant T@10 0.80 → 0.91; user-only T counted user turns of the right session).
- The cost is to user-evidence questions: twice as many memories, and long assistant replies split into many claims.
- The statement cap matters most here: multi-session Rall@10 0.669 → 0.719 against `base`.
- `assistant_weight` 0.05 is on in both columns.
- QA is below.

### QA: list vs structured context (same retrieval)

The reader and judge are as before: GLM-4.7-Flash, thinking off, Mem0's LoCoMo prompts and LongMemEval's prompts. All three conditions use the same recall (`default`, k = 10):
- `list`: the harness's dated list of hit texts, as in every earlier table. Since claims, those texts are sentences.
- `stmt`: the same list with each claim replaced by its whole statement (`context`), once per statement. That is what the list showed before claims.
- `context`: `recall {format: 'context'}`. The budget is 3000 tokens (LongMemEval) or 1500 (LoCoMo). It shows groups, dates, `previously` lines and up to 600 characters of each statement. The median block is about 710 tokens on LoCoMo.

QA_TABLES

### Weaknesses after this work

1. **Claim fragments dominate the vector leg on chat turns.** This was the biggest LoCoMo loss in this run (hybrid R@5 0.66 → 0.42), and it comes from the write path (`wave3/claims-fix`). Every LoCoMo hybrid and QA number above should be re-measured after it.
2. **Multi-hop evidence that the question does not mention** is still missed (LoCoMo multi-hop Rall@10 ≤ 0.11). Diversity cannot help it, and entity expansion did not help with crude entities.
3. **Storing assistant turns** costs user-evidence turns in the top 10 (T@10 0.914 → 0.850; ss-user 0.969 → 0.906). The statement cap and `assistant_weight` 0.05 recover part of it. A higher weight, or `role: 'user'` for questions about the user, is untested.

## Head-to-head: Mem0 (open source)

Status: 2026-10-23, first run. Adapter: `scripts/bench-l2/mem0_adapter.py`. Raw results, logs, the LLM cache and the venv lock file are in `.bench-data/mem0-h2h/`.

Published Mem0 numbers use other LLMs, prompts and judges (memory-decisions.md §2.3), so they cannot be set beside the tables above. Here Mem0 runs **through this harness**: the same questions, the same reader and judge (GLM-4.7-Flash with the prompts in `qa.py`), the same embedding model, and the same LLM for Mem0's own extraction step.

### Setup

| | kleos | Mem0 |
|---|---|---|
| Version | main `92ef789` (engine tag `1c1bf81`, claim-level `remember`), plus `6f9711f` (before claims) for reference | `mem0ai` **2.2.1** (PyPI, released 2026-09-25), the open-source `Memory` class, in its own venv |
| Write path | `remember(text, ns, at, source)` for each LoCoMo turn or LongMemEval user message. No LLM. | `Memory.add(messages, user_id, metadata)`: Mem0's v3 pipeline. Per `add` it makes **one LLM extraction call** (ADD-only, deduplicated against the 10 nearest existing memories and the last 10 messages), then adds BM25 lemmas and spaCy entity links. |
| LLM | none | Synthetic `syn:small:text` (GLM-4.7-Flash), thinking off, temperature 0, JSON response format, `max_tokens` 4000 (Mem0's default is 2000; see below). Called through `llm.Client`, so every call is cached by prompt hash. |
| Embedder | EmbeddingGemma int8 (candle, Metal), 512 tokens | **The same model**, `google/embeddinggemma-300m`, in fp32 (sentence-transformers, MPS), 512 tokens, with the same query and document prompts |
| Store | in-memory kleos DB, one per unit | Mem0's default Qdrant, in process and in memory, one per unit. SQLite history is also in memory. BM25 uses fastembed and entities use spaCy `en_core_web_sm`, as Mem0 ships them. |
| Read | `recall(question, k=10)`, hybrid | `search(question, top_k=10)`: semantic, BM25 and entity boost, with Mem0's default threshold 0.1. `top_k=20` (Mem0's default) is also reported. |
| Reader context | the top 10 memories in time order, each with its `valid_from` date | the top 10 (or 20) memories in time order, each with the session date Mem0 stored as `created_at` |

**Adapter choices.** None of these changes Mem0's algorithm.

- **Session time.**
  - OSS `add()` rejects the platform-only `timestamp` argument.
  - The adapter therefore puts the session date in the extraction prompt's own "Observation Date" slot, the one `generate_additive_extraction_prompt` already has. That is what the platform's `timestamp` does.
  - The session time is also passed as `created_at` metadata, which Mem0 keeps.
  - "Current Date" is pinned to the run date, so cached prompts stay valid.
  - Without this, Mem0 would resolve "yesterday" against today's date.
- **Provenance.**
  - Mem0 does not record which turns a memory came from.
  - The adapter passes the source turn ids as metadata on each `add`. Mem0 copies them onto every memory that call creates.
  - This gives retrieval R@k at the granularity of one `add`.
- **Speakers (LoCoMo).** As in Mem0's own evaluation code: speaker A is `user`, speaker B is `assistant`, and the content is `"Name: text"`. Mem0 v3 extracts from both roles and names the speakers.
- **One store per conversation.** Mem0's own LoCoMo eval keeps one store per speaker and gives the reader 10 + 10 memories. Here there is one store per conversation; the `mem0_k20` column covers the 20-memory reading.
- **Telemetry** is off (`MEM0_TELEMETRY=False`, checked at start), so there are no PostHog events and no remote notice config.

**The deviation that matters: one `add()` per session, not per turn pair.**
- **Why.** Mem0's LoCoMo protocol adds two messages at a time, about 2,950 LLM calls for LoCoMo, each with a prompt of about 9k tokens (the system prompt alone is about 5k).
  - Synthetic answered `429 "Too many concurrent requests"`. The limit is account-wide and was shared with other jobs running at the same time.
  - Throughput was 3–15 calls per minute, so the pair protocol did not fit in the time box. It is implemented (`--batch pair`) but was not run beyond a first aborted try.
- **What ran instead.** One `add` per session: **272 calls for all 10 LoCoMo conversations**, and about 47 per LongMemEval question.
- **How it might bias the result.** The direction is unknown.
  - Mem0 sees a whole session at once, which gives it more context.
  - It writes fewer, coarser memories: 1,788 for LoCoMo's 5,882 turns, about 7 per session.
  - `max_tokens` was raised to 4000 so that no session's extraction is cut off; 0 were truncated.

**What ran**
- **LoCoMo:** all 10 conversations ingested.
  - Retrieval on all 1,536 category 1–4 questions.
  - QA on **the same stratified 301** as the tables above.
- **LongMemEval_S:** a stratified **11** questions from the 119-question hybrid set (`--n 10`, which rounds to 11), with full haystacks of about 47 sessions and 490 messages each. The target was 40–60; the rate limit allowed 11 in the time box.
  - At n = 11, one question moves accuracy by 9 points. **Treat the LongMemEval rows as anecdotal.**
- **Errors.** No add failed. 388 + 729 HTTP 429 retries were absorbed by retrying.
- **Determinism.** A one-worker re-run of conv-26 from the cache rebuilt the same 131 memories.

### Results

**LoCoMo QA, the same 301 questions** (GLM-4.7-Flash reader and judge, Mem0's LoCoMo prompts):

| split | n | no memory | kleos hybrid, main `1c1bf81` | kleos hybrid, `6f9711f` | **Mem0, top 10** | Mem0, top 20 | full conversation | evidence turns |
|---|---|---|---|---|---|---|---|---|
| **cat 1–4** | 301 | 0.462 | 0.565 | 0.714 | **0.748** | 0.784 | 0.814 | 0.897 |
| 1 multi-hop | 55 | 0.455 | 0.527 | 0.709 | 0.764 | 0.818 | 0.909 | 0.964 |
| 2 temporal | 63 | 0.381 | 0.302 | 0.365 | **0.524** | 0.556 | 0.603 | 0.714 |
| 3 open-domain | 19 | 0.474 | 0.579 | 0.526 | 0.737 | 0.684 | 0.737 | 0.789 |
| 4 single-hop | 164 | 0.494 | 0.677 | 0.872 | 0.829 | 0.872 | 0.872 | 0.957 |

**LoCoMo retrieval, categories 1–4, all 1,536 questions.**

Mem0 memories come from whole sessions, so both systems are scored at **session level**: a hit counts if it comes from an evidence session. kleos is also shown at turn level.

| system | R@5 | R@10 | Rall@10 |
|---|---|---|---|
| Mem0 (session level) | 0.880 | 0.930 | 0.841 |
| kleos hybrid `6f9711f` (session level) | 0.887 | 0.938 | 0.834 |
| kleos hybrid main `1c1bf81` (session level) | 0.708 | 0.787 | 0.662 |
| kleos hybrid `6f9711f` (turn level) | 0.661 | 0.749 | 0.612 |
| kleos hybrid main `1c1bf81` (turn level) | **0.423** | 0.479 | 0.368 |

**LongMemEval_S, 11 questions** (anecdotal; LongMemEval's official judge):

| split | n | no memory | kleos hybrid main | kleos hybrid `6f9711f` | Mem0, top 10 | full history | oracle sessions |
|---|---|---|---|---|---|---|---|
| all | 11 | 0.091 | 0.636 | 0.727 | **0.909** | 0.545 | 0.727 |

- Session-level R@10 is 1.000 for both systems. Rall@10 is 1.000 for Mem0 and 0.909 for kleos.
- **Where Mem0 won and kleos (main) lost:**
  - 1 single-session-assistant question: kleos stores only user messages, so these are unanswerable by design (see Units).
  - 1 single-session-user, 1 knowledge-update and 1 multi-session question.
- **Where kleos won and Mem0 lost:** 1 multi-session question.

### Cost and latency

| | kleos (main, hybrid) | Mem0 |
|---|---|---|
| LLM calls at ingest | 0 | 272 for LoCoMo (one per session); 516 for 11 LongMemEval questions |
| LLM tokens at ingest | 0 | LoCoMo: 2.51 M prompt + 0.10 M completion, **445 tokens per turn**. LongMemEval: 5.64 M + 0.24 M, **about 0.53 M per question**, 1,096 per message. |
| Ingest wall time per turn | LoCoMo `remember` p50 56 ms, p95 151 ms (Metal, machine shared) | LoCoMo 1.18 s per turn; `add` p50 18.8 s, p95 66 s per session. Most of that is the LLM call, including waits for a free concurrency slot. |
| Local (non-LLM) cost per `add`, one worker, from cache | – | p50 1.08 s, of which embedding is 0.87 s (torch MPS: messages, memories and entities) |
| Search k = 10, one worker | recall p50 21 ms, p95 32 ms (conv-26) | p50 142 ms, p95 378 ms (conv-26), of which query and entity embedding is 109 ms |
| Search k = 10 under the 6-worker run | recall p50 30 ms, p95 66 ms (all LoCoMo, machine shared) | p50 0.98 s, p95 2.75 s |

- Mem0's search latency is mostly torch EmbeddingGemma on MPS. That cost belongs to this setup, not to Mem0's design: with an API embedder it would be a network round trip instead.
- Without the embedding, Mem0's search is roughly 30 ms (spaCy, two Qdrant queries and scoring; p50 minus p50, so approximate). That is close to kleos.

### Reading this honestly

- **Mem0 beat kleos main on LoCoMo QA (0.748 vs 0.565), and narrowly beat the pre-claims build (0.714).** At n = 301 a 3.4-point gap is about 10 questions, within noise. The temporal row is a real gap (0.52 vs 0.30–0.37).
  - Mem0's extraction writes resolved dates into each memory ("Caroline attended a LGBTQ support group on May 7, 2023").
  - kleos stores the raw turn ("…yesterday…") and leaves the date arithmetic to the reader.
- **kleos main regressed on LoCoMo.**
  - Turn-level R@5 went from 0.661 (`6f9711f`) to 0.423, and QA from 0.714 to 0.565.
  - Main's claim-level `remember` returns single sentence claims, for example "Caroline: I chose them 'cause they help LGBTQ+ folks with adoption." So the top 10 covers fewer turns and gives the reader less context.
  - This is a kleos finding, not a Mem0 one. It needs a follow-up: returning the parent turn, or grouping claims by turn in `recall`.
- **On retrieval the two are level.** At session level, Mem0's R@5 of 0.880 is the same as kleos `6f9711f`'s 0.887. Mem0 spends one LLM call per session to get there; kleos spends none.
- **Cost is the decisive difference.**
  - Mem0 needs an LLM call on every `add`: about 9–11k prompt tokens per call, even at session granularity.
  - That is about 2.6 M tokens to ingest LoCoMo and about 0.5 M per LongMemEval question.
  - Per-turn adds, Mem0's own protocol, would cost about 10 times more.
  - kleos ingests with no LLM at 20–60 ms per memory.

**What could disadvantage Mem0:**
- **The small extraction LLM.** Mem0's paper used gpt-4o-mini; kleos uses no LLM at write time. A small LLM hurts Mem0 and cannot hurt kleos.
- **Session batching instead of Mem0's turn-pair protocol.** The effect is unknown in either direction.
- **One store per conversation instead of one per speaker.**
- **Temperature 0** instead of Mem0's default 0.1, and `top_p` left unset.

**What could favour Mem0:**
- `max_tokens` 4000 instead of 2000.
- An fp32 embedder against kleos's int8.
- It ingests assistant messages, while kleos's LongMemEval unit stores user messages only. The single-session-assistant question in the sample is one of these.
- Its provenance is per session (one `add`), so its session-level R@k compares like for like only with kleos at session level.

**Neither way:** the reader, judge, prompts and question sets are identical.

**Sanity check:** for kleos main on the 11 LongMemEval questions, this adapter's answers and the harness's own `qa.py` run of the same build agree on 8 of the 9 verdicts both runs have. GLM at temperature 0 is not fully deterministic.

**Quota.** The weekly plan went from 91.5% remaining at the start to 87.3% at the end, about **4.1 points**. This is an account-wide reading that includes any other job using the key at the same time, so it is an upper bound for this work. The cap was 40%.

### Reproduce

```zsh
export HF_HOME=/Volumes/SSD01/caches/huggingface FASTEMBED_CACHE_PATH=/Volumes/SSD01/caches/fastembed MEM0_MAX_TOKENS=4000
M=.bench-data/mem0-h2h; uv venv --python 3.12 $M/.venv
uv pip install --python $M/.venv/bin/python -r $M/requirements.lock   # mem0ai[nlp]==2.2.1, sentence-transformers, torch, fastembed, en_core_web_sm
A=(scripts/bench-l2/mem0_adapter.py)
$M/.venv/bin/python $A ingest --bench locomo --batch session --workers 6
$M/.venv/bin/python $A ingest --bench lme --n 10 --workers 4
$M/.venv/bin/python $A qa --bench locomo --mem0-k 10,20 --kleos hybrid --kleos-tag 1c1bf81
$M/.venv/bin/python $A qa --bench lme --n 10
$M/.venv/bin/python $A ingest --bench locomo --only conv-26 --workers 1 --suffix=-latency   # local latency, from cache
$M/.venv/bin/python $A report          # -> $M/results/report.md, summary.json
```

- The kleos side reads the harness's recall results under `.bench-data/bench-l2/results/<tag>/` read-only. The adapter answers from them with exactly `qa.py`'s formatting.
- Every LLM call is cached under `.bench-data/mem0-h2h/llm-cache/`, so re-running costs no quota.

## Method

### What is measured

Everything goes through kleos's public memory API, the Python bindings' `remember` and `recall` ([memory-api.md](../specs/memory-api.md)). No CozoScript is used, and kleos is not tuned for the benchmarks.

- **Ingest.** Each LongMemEval question (or LoCoMo conversation) gets a fresh in-memory database and its own namespace. Every unit is stored with `remember(text, ns, at, source)`. `at` is the session timestamp plus 1 s per unit, so units arrive in time order. That makes valid time, supersession and `as_of` behave as they would in a live agent.
- **Recall.** `recall(question, ns, k)` with k = 10 and k = 5 (two separate calls), with default options. That means `vector_weight` 0.6, `hide_pending` false, and the read is at "now".
- **Score.** Hits are mapped back to their source turns (through the memory id, including statements folded in as duplicates) and scored against the datasets' evidence labels.

**Variants.** Each variant is a separate process, because the model registries are process-wide.

| Variant | Write path | Read path |
|---|---|---|
| `fts` | no models: the exact-duplicate rule and the value rule only | FTS (BM25) only |
| `hybrid` | EmbeddingGemma int8 vectors; no decider, so rules only | HNSW + FTS, `WeightedFusion` 0.6 / 0.4 |
| `sup` | as `hybrid`, plus decider-nano (`Mazino0/kleos-decider-nano`, q8) for `supersedes` and `duplicate` on the top-5 candidates of every `remember` | as `hybrid`; superseded memories are no longer current |

**Device.**
- **276f90f:** models on **CPU**. Called from the Python bindings, candle 0.11's Metal kernels failed to compile (`unknown type name 'float16_t'`). Metal took the shader-language version from the host interpreter's SDK, so the kernels compiled as pre-3.1 MSL. This happened for EmbeddingGemma and decider-nano alike.
- **6f9711f:** vendors candle-metal-kernels with the MSL version pinned, so models run on **Metal**. `--device` and `KLEOS_BENCH_DEVICE` (default `metal`) choose the device.
- **Machine.** A 16 GB Apple-silicon laptop shared with other agents' builds. Load average was 10–35 during the CPU runs and about 3 during the Metal runs.

### Units (why one memory per user message on LongMemEval)

On LongMemEval each **user message** is one memory, holding the text as written. Assistant replies are not stored. The reasons:

- An agent's memory holds what the user told it. 842 of the 896 evidence turns (94%) are user turns.
- User messages average about 14k tokens per question, against about 101k for user + assistant rounds, even with each memory capped at the embedder's 512-token window. On CPU (the only option when this was decided) that is the difference between about 70 s and about 8 min of embedding per question.
- The decider compares memory pairs without truncation, so short memories also keep `remember` affordable.

The cost is that **single-session-assistant questions (56 of the 150 single-session questions) cannot be answered from the stored memories**, because their evidence is in the assistant's reply. Their retrieval is still scored at session level, since a user message from the right session counts.

`--unit round` (user turn plus the assistant reply, the LongMemEval paper's best value granularity) is implemented. It is the natural follow-up now that Metal works.

On LoCoMo, each dialog turn is one memory: `"Speaker: text"`, plus ` [shares a photo: <caption>]` when the turn shares an image (as in Mem0's harness). Turns are short.

### Metrics (LLM-free; the headline)

LongMemEval (`answer_session_ids`, `has_answer` turns):

- **R@k:** at least one of the top-k memories comes from an answer session. This is LongMemEval's `recall_any`, at session level.
- **Rall@k:** every answer session has a memory in the top k (`recall_all`).
- **T@k:** a top-k memory is an evidence turn (`has_answer`). Only questions that have evidence turns count (479 of 500).
- **NDCG@10:** sessions are ranked by their first appearance in the top-10 memories, with binary relevance (LongMemEval's session-level `ndcg_any`).
- **Knowledge-update view.** For the 72 non-abstention KU questions whose answer sessions have different dates:
  - **latest@k:** the most recent answer session is in the top k;
  - **stale-first:** the first answer-session memory comes from an older answer session. This is a stale value outranking the current one, so lower is better.
  This is where supersession should help.
- **Splits:**
  - single-session (user, assistant and preference combined), multi-session, temporal-reasoning, knowledge-update;
  - **abstention**: the 30 `_abs` questions, taken out of their types and scored against the sessions that hold the related, non-answering facts.

LoCoMo (`evidence` dialog ids, e.g. `D1:3`):

- R@k, Rall@k and NDCG@10 at turn level.
- Per category, using the ids in `locomo10.json`: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial. The names follow Mem0's evaluation code; the A-Mem paper swaps some of them.
- "cat 1–4" is the usual headline; category 5 is shown separately.

**@5 and @10.** These come from separate `recall(k=5)` and `recall(k=10)` calls. Over-fetch depends on k, so top-5 is not always a prefix of top-10.

### Decider-nano contamination (knowledge update)

decider-nano's fine-tuning data includes `supersedes` pairs mined from the answer turns of **53** LongMemEval knowledge-update questions (oracle file, `scripts/infer-bench/decide/train/build_data.py`). **25** KU questions were held out. These are the same question ids and the same turns as in LongMemEval_S.

The harness recomputes the split exactly: sort by id, shuffle with `Random(20261023 + 1)`, and take the first 25. It reports KU results for the held-out 25 and the trained-on 53 separately. Only the `sup` variant can be affected; `fts` and `hybrid` use no decider.

### QA (secondary)

**Reader and judge.** Both are the same LLM: Synthetic's `syn:small:text`, which resolved to **`zai-org/GLM-4.7-Flash`**. It runs with **thinking off** (`reasoning_effort: "none"`) at temperature 0. Every request and response is cached under `.bench-data/bench-l2/llm-cache/`, keyed by a hash of the prompt.

**LongMemEval prompts.**
- Reader: the history-chat prompt from LongMemEval's `run_generation.py`, without chain-of-note. The retrieved memories appear in time order, each with its date, and the prompt gives the current date.
- Judge: the per-task prompts from `evaluate_qa.py`, including the off-by-one rule for temporal questions, the "updated answer" rule for KU, the preference rubric and the abstention prompt.

**LoCoMo prompts.** The reader and the `ACCURACY_PROMPT` judge from Mem0's evaluation code. Category 5 is excluded.

**Conditions, on the same questions:**
- `kleos_<variant>`: the top-10 recalled memories;
- `none`: no memory;
- `oracle`: LongMemEval's answer sessions in full (user and assistant turns, as in the paper's oracle setting), or LoCoMo's evidence turns;
- `full`: the whole history, where it was run.

**These QA numbers are not comparable with published leaderboards.** Those use GPT-4o or gpt-4o-mini as reader and judge, and reader prompts, chain-of-note and judges move scores by 10–30 points (memory-decisions.md §2). Only compare conditions within one table: same reader, same judge, same questions.

### Reproduce

Re-running against a newer engine is one command. Point `ENGINE_SRC` at a checkout of it; a detached worktree of main is enough:

```zsh
git worktree add --detach .bench-data/bench-l2/engine-<rev> <rev>   # or any kleos checkout
ENGINE_SRC=$PWD/.bench-data/bench-l2/engine-<rev> QA=1 scripts/bench-l2/run.sh
```

What `run.sh` does:
- builds the wheel (`RUSTUP_TOOLCHAIN=1.99.0 maturin build --release --features compact,infer-metal -m kleos-lib-python/Cargo.toml`) and installs it into `.bench-data/bench-l2/.venv`;
- runs every retrieval set above, then (with `QA=1`) the QA sets;
- prints `report.py --compare 276f90f`, i.e. old → new on the questions both builds ran.

The tag is the checkout's last engine commit (`kleos-core`, `kleos-infer`, `kleos-lib-python`, `Cargo.lock`).

Other uses:
- `SKIP_BUILD=1 KLEOS_BENCH_TAG=<rev>` re-prints an existing run.
- `COMPARE=6f9711f` changes the baseline.
- `QA=1` reads `SYNTHETIC_KEY` from the `.env` next to the checkout, at call time. The key is never logged.
- The model runs take about 15 min on Metal. The fts runs take minutes.

**Quota.** All QA in this document used about 8.5% of the weekly Synthetic plan. `syn:small:text` does not count against the 500-request subscription window. Full-context LongMemEval prompts were 5 points of that.

**Where things go.**
- Results: `.bench-data/bench-l2/results/<git rev>/`.
  - `<bench>-<unit>/<variant>/<question>.json` holds the top-k hits with sources, the ingest stats (added, duplicate, superseded, pending) and per-call latencies.
  - `qa/…` holds the QA answers.
- Reports: `report.md` and `summary.json` in the same folder.

**Resuming.** Every unit is cached, so an interrupted run resumes. `report.py --compare <old rev>` prints old → new on the questions both runs share.

**Dataset revisions.**
- LongMemEval cleaned: HF `xiaowu0162/longmemeval-cleaned@98d7416c24c778c2fee6e6f3006e7a073259d48f` (`longmemeval_s_cleaned.json`, `longmemeval_oracle.json`).
- LoCoMo: `snap-research/locomo@3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`, `data/locomo10.json` (CC BY-NC 4.0, evaluation only).

**Sampling.** Samples are stratified by question type, seed 20261023 (`common.stratified_sample`).
