# GLiNER2.5 as kleos's write-time extraction model

*2026-10-23, branch `wave3/gliner`. Spike: measurements in Python (gliner2
2.0.0, PyTorch), no kleos-core or kleos-infer changes. Scripts:
[`scripts/gliner/`](../../scripts/gliner/). Data, caches and raw results:
`.bench-data/gliner/`.*

> **Update, 2026-10-24:** the spike below was zero-shot only. A fine-tuned
> `base-v1` was then tested against a hand-checked gold set. See
> [Fine-tuned](#fine-tuned-2026-10-24).
> - **Names:** fine-tuning works. Mentions of people by relation or role
>   ("my sister", "my boss") go from F 0.14 to 0.86. Names overall go from
>   F 0.82 to 0.89, at about 25 ms per claim, 3× faster than zero-shot.
> - **Store or skip:** it only ties kleos's rules (0.725 vs 0.735).
> - **About the speaker:** it loses to a first-person regex (0.72 vs
>   0.845).
> - **Update keys:** the verdict on keys below stands.

## Verdict

**Do not adopt GLiNER2.5 as a write-time *update* model. Keep it in view
only as an optional NER for entity filters and the same-subject gate.**

- **Entities.** `gliner2.5-base-v1` (194M) or `multi-v1` (287M) find
  person and place names well: about 0.86–0.91 F1 once a one-line
  capital-letter filter removes generic nouns. Raw output is poor
  (precision 0.2–0.35), because every type also fires on common nouns and
  `small` tags the pronoun "I" as a person. Organizations, products, events
  and quantities stay weak (0.25–0.6 F1).
- **Claim type.** Zero-shot classification of fact / preference / plan /
  event / question / request / small talk is near chance: 28–30% over seven
  classes. On the store-or-skip split it is *worse* than kleos's rule filter
  (0.51–0.56 against 0.72).
- **Update keys** (the key question). A (subject, attribute, value) schema
  gives the old and the new claim of a true update the same key with
  different values in only **9–27%** of the 22 held-out updates. Raising
  that buys false key matches between unrelated claims. The best variant
  (v3n, iterated on the trained-on questions) gets 18% on the held-out
  questions, at 0.57 false matches per question.
- **Combined with the decider.** On the 23 held-out questions:
  - Main (a) gets 83% latest-first.
  - (b) with the brief's keys (v2): still 83%, but all 15 key
    auto-supersedes were wrong.
  - (b) with v3n keys: 78%. 6 of its 11 auto-supersedes were wrong, and one
    of them retracted an answer.
  - (c) the entity gate: 83%. It skipped 12.5 decider questions per
    question without changing any outcome.
  - None of them fixed main's four stale-first misses.
- **Speed and memory.** PyTorch F32, indicative because the machine was
  loaded:
  - entities at batch 8: `small` 41 ms per claim on CPU; `base` 44 ms per
    claim on MPS and 107 ms on CPU;
  - peak RSS 1–3 GB;
  - the schema prompt, not the claim, dominates the cost.
- **GLiNER2.5-Decide** is a 486M-parameter single-text classifier on
  DeBERTa-v3-large, not a pair model. Used zero-shot as a pair decider on
  decider-nano's held-out KU pairs, it scores AUROC **0.57**. decider-nano
  scores **0.82** on the same pairs.
- **Port.** About 13–14 engineer-days for an MVP (DeBERTa-v3 encoder,
  entities, classification, choice and str fields), and 22–25 days for
  everything. The work is not justified by these results. If entity
  extraction is wanted later, `base-v1` with the capital filter is the
  candidate.

What ran, honestly:
- All five measurements were made in Python on the shared 16 GB machine.
- **The machine was heavily oversubscribed.** Other agents' benchmarks
  pushed the load average to 40–140 at times, so every latency below is an
  upper bound and is noisy. The speed table uses the fastest of 3
  repetitions, taken when the load had dropped to about 4–20.
- The silver labels come from GLM-4.7-Flash. The Synthetic key is limited
  in concurrency, so labels were requested ten claims per call.

## Results

### Setup

- **Package and checkpoints.** `gliner2` 2.0.0 (2026-08-24; 2.1.0 was 13
  days old, inside the two-week rule), with torch 2.14.1 and transformers
  4.57.6, resolved with `uv --exclude-newer 2026-10-08`. Checkpoints are the
  current `fastino/*` Hub revisions.
- **Tokenizer workaround.** The 2.5 tokenizers store `extra_special_tokens`
  as a list (Transformers 5 metadata). gliner2 2.0.0's fallback for this is
  short-circuited by Transformers 4.57, so `scripts/gliner/common.py` calls
  gliner2's own compatibility path directly (`extra_special_tokens={}`). The
  ten schema tokens still tokenize as single tokens.
- **Claims** are kleos's own: `memory_claims::split_claims` through a
  model-free `remember`, with `decided` = `is_declarative`.

### 1. Entities (300 claims, silver labels)

**Sample** (`build_sample.py`): one random claim from each of 160 LongMemEval_S
user turns and 140 LoCoMo turns. 194 claims are declarative by kleos's rule.

**Silver labels** (`silver.py`) cover 8 types: person, location,
organization, product, date, quantity, event and title. They give 119
entities in 92 claims, plus a 7-way claim type.

**Silver quality, hand-checked on 50 random claims:**
- *Entities:* 12 of the 19 labelled are right (63%). The 7 wrong ones are
  all generic things labelled `product`: "dental chews", "cupcakes",
  "humidifier", "necklace", "Pho". No missed entity in those 50. But while
  reviewing GLiNER's false positives I found a few names the silver missed
  ("Suva area", "Austin City Limits", "National Gallery", "a month ago").
- *Types:* 39 of 50 clearly right, 6 debatable (opinion vs small talk,
  fact vs event), 5 wrong. Most errors are on the question/request boundary
  ("Any recommendations?" labelled question, "Should I try resetting…?"
  labelled request).

So treat `product` and the question/request split as soft, and the other
labels as fair.

The table below shows entity matches: exact text and the same label
(strict), or any overlap with the same label. One combined `batch_extract`
call per batch of 8 used labels with descriptions, threshold 0.5.

| Model | Raw P / R / F (strict) | Raw, overlap F | + capital filter: P / R / F (strict) | overlap F |
|---|---|---|---|---|
| small-v1 (74M) | 0.21 / 0.74 / 0.33 | 0.37 | 0.54 / 0.61 / 0.57 | 0.64 |
| base-v1 (194M) | 0.35 / 0.81 / 0.49 | 0.52 | **0.68 / 0.68 / 0.68** | 0.70 |
| multi-v1 (287M) | 0.29 / 0.80 / 0.43 | 0.45 | **0.69 / 0.66 / 0.68** | 0.70 |

The *capital filter* keeps a person, location, organization, product,
event or title span only if it contains a capital letter and is not "I".
Dates and quantities are untouched.

Per label, with the filter, at threshold 0.5. Each cell is P / R; `n` is the
number of silver entities.

| Label (n) | small | base | multi |
|---|---|---|---|
| person (22) | 0.72 / 0.82 | 0.86 / 0.86 | 0.91 / 0.91 |
| location (26) | 0.75 / 0.81 | 0.83 / 0.92 | 0.80 / 0.92 |
| organization (10) | 0.36 / 0.50 | 0.71 / 0.50 | 1.00 / 0.60 |
| product (21) | 0.38 / 0.29 | 0.41 / 0.33 | 0.50 / 0.33 |
| date (15) | 0.46 / 0.73 | 0.63 / 0.80 | 0.62 / 0.87 |
| quantity (9) | 0.33 / 0.56 | 0.43 / 0.67 | 0.29 / 0.22 |
| event (10) | 0.67 / 0.40 | 0.80 / 0.40 | 0.67 / 0.40 |
| title (6) | 0.43 / 0.50 | 0.57 / 0.67 | 0.38 / 0.50 |

What goes wrong without the filter:
- Each type fires on common nouns: "farmers", "koalas", "universities" and
  "beginners" become persons, organizations or locations.
- The same span is often returned under two labels ("faeries" as both
  person and organization).
- `small` labels the pronoun "I" as a person 98 times in 300 claims.

Raising the threshold to 0.7 helps `small` (F 0.61) but barely moves
`base` or `multi`.

### 2. Claim type (7 classes, same 300 claims)

Silver distribution: question 76, small talk 62, request 45, plan 36, fact
32, preference 31, event 18.

| Model | 7-way accuracy | Store vs skip (fact/pref/plan/event vs question/request/small talk) |
|---|---|---|
| small | 0.29 | 0.51 |
| base | 0.28 | 0.56 |
| multi | 0.30 | 0.55 |
| kleos rule filter (`is_declarative`) | — | **0.72** |

- The type labels carried one-line descriptions. The joint call (entities
  and type together) scores the same as a separate `classify_text`
  (0.29–0.32).
- The dominant errors are small talk → fact (39–47 of 62) and question or
  request → fact or preference.
- GLiNER2.5's classifier does not separate these zero-shot. Fine-tuning
  would be needed, and at that point decider-nano's encoder is the cheaper
  thing to tune.

### 3. Update keys (the key question)

**True pairs** (`ku_pairs.py`):
- For each KU question, the user turns flagged `has_answer` in the earlier
  and later answer sessions are split into claims. The LLM then picks the
  claim stating the old value and the claim stating the new one.
- This gave 22 of the 23 held-out questions and 36 of the 53 trained-on
  questions.
- I read all 22 held-out pairs. 21 are right. In `c6853660` the old side is
  a question ("Can you give me some tips…"), so that pair cannot hit.

**Metrics:**
- *Key hit*: some fact of the old claim and some fact of the new claim share
  (subject, attribute), with a single-valued attribute and different
  non-empty values.
- *False key matches*: the same test over every other pair of declarative
  claims of the same question (5,472 held-out pairs over 23 questions,
  11,146 trained-on pairs over 49), counted per question.

| Variant | Schema / matcher | trained-on 36: hit | false / q | **held-out 22: hit** | same key, any value | **false / q** |
|---|---|---|---|---|---|---|
| v1 small | brief's schema (`attribute::[home\|job\|…]`), exact subject | — | — | 9% | 18% | 1.74 |
| v2 small | + field descriptions, exact subject | 14% | 0.88 | 14% | 14% | 0.70 |
| v2 base | same | — | — | 0% | 9% | 0.13 |
| v3 small | unambiguous attribute names, loose subject (shared content word) | 25% | 4.18 | 27% | 27% | 8.57 |
| v3s small | v3 schema, exact subject | 14% | 1.14 | 9% | 14% | 2.52 |
| **v3n** small | v3, loose subject, both values numeric | 19% | **0.27** | **18%** | 27% | **0.57** |

v1 and v2 were fixed before any run. v3, v3s and v3n were chosen on the
trained-on questions; each was then run once on the held-out ones.

**Why it fails:**
1. **The subject drifts between the two claims.**
   - "Instagram" vs "follower count".
   - "bird identification" vs "Northern Flicker".
   - "owning a home" vs "house" / "Wells Fargo".
   - "Python" vs "Corey's series".

   Both claims are about the same thing, but GLiNER extracts a different span
   as the subject each time. That is inherent to extracting *spans*:
   kleos would need a canonical subject, which is a generative or
   entity-linking step.
2. **The attribute choice is unstable.**
   - The choice is decoded as a span over the prefix, so "at home" in the
     text pulls counts to `home`.
   - "two of them" → `favorite`, while "3 of Emma's recipes" → `home`.
   - Renaming the choices (v3) fixed part of this.
3. **There are no facts at all on a side in 30–50% of true pairs**, mostly
   on chatty turns ("By the way, …").
4. **Loosening the subject match buys hits with false matches** at roughly
   4–10 false matches per extra hit.

The v3n "false" list on the held-out questions over-counts somewhat: about
half of its 13 entries are *other* real updates of the same fact (postcards
17 → 25, National Geographic issue three → five, Corey videos 20 → 30)
rather than unrelated claims. The other half are genuinely wrong ("watched
50 episodes" vs "want to watch 1 episode daily"; a $3,250 credit vs "1% of
the purchase price").

### 4. Policies on the 23 held-out KU questions

This is `claims_ku.py`'s harness through the kleos wheel (decider-nano and
EmbeddingGemma on Metal), with GLiNER `small` on CPU in Python
(`policy_sim.py`).

- *Latest first / stale first*: is the first recall hit from an answer
  session the latest answer session, or an older one?
- *False supersedes* are hand-judged from the logged old/new texts.

| Policy | Latest first | Stale first | … `hide_pending` | Supersedes / q (decider + key) | False supersedes (hand) | Pending / q | Gated / q | `remember` p50 (of which GLiNER) |
|---|---|---|---|---|---|---|---|---|
| (a) decider alone (main) | **83%** (19/23) | 17% | 91% | 0.43 + 0 | 1 of 10 | 1.96 | 0 | 243 ms (—) |
| (b) + keys v2 | 83% | 17% | 91% | 0.43 + 0.65 | 1 + **15 of 15** key | 1.91 | 0 | 1420 ms (838) |
| (b) + keys v3n | **78%** (18/23) | 22% | 87% | 0.35 + 0.48 | 1 + **6 of 11** key | 1.87 | 0 | 950 ms (522) |
| (c) + GLiNER entities | 83% | 17% | 91% | 0.43 + 0 | 1 of 10 | 1.91 | 12.5 | 311 ms (80) |
| (cu) + entities + `user` | 83% | 17% | 91% | 0.43 + 0 | 1 of 10 | 1.96 | 0 | 358 ms (96) |

Latencies are wall-clock on the shared machine. (a), (c) and (cu) ran at a
load average of about 9, and both (b) runs at 40–70, so compare latencies
only within a load level. Main's published figure for (a) on a quiet machine
is 144 ms p50.

The v3n keys made 11 auto-supersedes:
- 5 were real: sculpture 5–6 → 10–12 h; a $220 market sale → $420;
  National Geographic issue three → five; 4 MCU films → 12 films including
  5 MCU.
- 6 were wrong. Three of those were equal values in different words ("page
  200" vs "200"; "nine months" vs "nine months now"). One of these retracted
  the answer claim of `db467c8c` ("How long have my parents been staying
  with me?"), which is the 83% → 78% drop.
- v2's 15 key supersedes were all wrong: "brisket" vs "smoked brisket", one
  gin vs another, "my road bike" vs "2,000 miles".

Reading the table:
- **(a)** reproduces main's published result: 83% / 17%, 0.43 supersedes and
  1.96 pending per question. Of its 10 auto-supersedes, 9 are true updates
  (postcards, recipes, Negroni, French press ratio, Corey videos, sculpture
  hours, bikes, chevre → ricotta) and one is false (eggs → milk, "need to
  use up").
- **(b) keys:** the key auto-supersedes never hit an answer pair that the
  decider had missed, and the four stale-first misses (mortgage $350k →
  $400k, Instagram 1250 → 1300, birds 27 → 32, gym days) stayed missed.
  They are exactly the subject-drift cases above.
- **(c) entities:**
  - (c) passes GLiNER's named entities only (person, location,
    organization, product, event, title). The gate then skips 12.5
    supersedes questions per question (`gated`), changes no outcome, and
    trims pending by 0.04 per question.
  - (cu) adds the subject `user` for first-person turns. Because a name is
    linked to every claim when no claim contains it, `user` reaches every
    claim, and the gate becomes a no-op (0 gated).
  - Neither variant stopped eggs → milk: no named entity there.
- **Latency.** GLiNER adds about 80–95 ms (p50) per turn for entities at
  load ≈ 9. The key schema added 840 ms p50 per turn during the load ≈ 40–70
  period; the clean cost should be close to the entity call's.

### 5. Speed and memory

Script: `speed.py`. Each model × device ran in its own process, on the first
40 sample claims, with PyTorch 2.14 F32 and 4 CPU threads. Each figure is
the fastest of 3 repetitions, in ms per claim, at batch 1 and batch 8.
Three schemas were timed:
- **entities**: 8 types with descriptions;
- **+type**: the same plus the 7-way type classification with descriptions;
- **keys**: the v2 fact schema.

Load averages during the runs were about 4–20, so treat the table as
indicative. Main's decider-nano costs 27 ms per decision on Metal F16, for
comparison.

| Model | Device | entities b1 / b8 | +type b1 / b8 | keys b1 / b8 | Peak RSS | MPS driver |
|---|---|---|---|---|---|---|
| small (74M) | CPU | 84 / **41** | 151 / 108 | 124 / 52 | 1.35 GB | — |
| small | MPS | 182 / 58 | 188 / 97 | 370 / 261 | 0.94 GB | 1.09 GB |
| base (194M) | CPU | 147 / 107 | 225 / 173 | 149 / 83 | 2.08 GB | — |
| base | MPS | 76 / **44** | 111 / 87 | 176 / 145 | 2.09 GB | 2.10 GB |
| multi (287M) | CPU | 199 / 125 | 323 / 330 | 270 / 112 | 2.98 GB | — |
| multi | MPS | 175 / 102 | 277 / 198 | 430 / 252 | 2.55 GB | 2.24 GB |
| Decide (486M) | CPU | 545 per pair, classification only (`decide_pairs.py`, heavily loaded machine) | | | | |

- **The schema prompt dominates the cost.** Claims are about 25 tokens;
  eight described entity types plus seven described classes add about 150
  tokens. That is why "+type" costs about 1.5–2× "entities", and the
  3-field key schema can be cheaper than the entities schema. The schema is
  encoded *with* the text (joint attention), so it cannot be cached across
  claims. Fewer labels and shorter descriptions are the main speed lever.
- **MPS** helps `base` (44 ms per claim at batch 8) but not `small`, where
  per-call overhead dominates.
- **Memory.** PyTorch F32 needs 1–3 GB peak. A candle port with an mmap'd
  F16 token table would hold roughly the weights: about 150 MB for `small`,
  390 MB for `base` and 575 MB for `multi` in F16, the same scale as
  decider-nano.
- An earlier n = 120 run of `small` on the CPU at load 70+ measured 314 ms
  per claim for entities at batch 8, a 7× slowdown. During that period,
  `policy_sim` measured 520–840 ms of GLiNER per turn.
- Decide on MPS was not run (stopped for time).

### GLiNER2.5-Decide against decider-nano

The 90 held-out LongMemEval KU pairs are decider-nano's own evaluation set
(`.bench-data/nano-tune/data/lme_ku_heldout.json`: 24 update, 66
compatible). Decide was run zero-shot (`decide_pairs.py`) with both
statements in the text:

- *yes/no*: the classification prompt is "Does the later message update or
  replace a fact stated in the earlier one?"
- *relation*: the labels `update` and `compatible`, each with a
  description.

| Model | Form | AUROC | Acc @ 0.5 | ms / pair (CPU, loaded machine) |
|---|---|---|---|---|
| GLiNER2.5-Decide (486M) | yes/no | 0.565 | 0.72 (predicts "update" 9×) | 545 |
| GLiNER2.5-Decide | relation | 0.448 | 0.73 | 636 |
| gliner2.5-base-v1 | yes/no | 0.616 | 0.37 | 633 |
| gliner2.5-base-v1 | relation | 0.471 | 0.46 | 764 |
| **decider-nano (tunec, shipped)** | supersedes | **0.824** | — | 27 (Metal F16, from `nano-tune/eval/summary.md`) |

Decide is tuned for operational routing (intent, urgency, moderation), not
for comparing two statements. As a pair decider it is near chance, and at
2.5× decider-nano's size it is not a candidate.

## Recommendation

- **Do not use GLiNER2.5** for update detection: neither keys nor Decide.
  The (subject, attribute, value) idea fails on subject drift, which a span
  extractor cannot fix. The decider remains the update path.
- **Do not use it** for claim type or the question/small-talk filter. The
  rules are better (0.72 against ≤ 0.56).
- **Possible later use:** optional NER for `remember {entities}` and
  `recall {entity}`, with `gliner2.5-base-v1`, only persons and places,
  threshold 0.5 plus the capital filter. That gives about 0.86 F1 on
  person and location.
  - It does not by itself improve the KU metric: the gate changed nothing
    here.
  - Its value is that the entity filters and `reflect` contradictions start
    working without the caller passing names.
  - Check it on the life stream (which has sister/user conflicts) before
    building it.
  - `multi-v1` is the same quality but 1.5× the size; pick it only if
    non-English input matters.
- **If entities are wanted, weigh the cheaper alternative first:** a
  capitalised-span plus gazetteer rule, or a small token-classification NER
  on DeBERTa-v3-xsmall. Those need about 3 engineer-days of port, not 13.
  GLiNER's open label set is what the port buys, and these results did not
  need it.

## Fine-tuned (2026-10-24)

*Branch `wave3/gliner-ft`. Python only, no kleos-core changes. Scripts:
`scripts/gliner/ft_*.py`, `kleos_rules.py` and `ft_gold/`. Data, checkpoints
and predictions: `.bench-data/gliner-ft/`.*

### Why it was redone

The spike above had four weaknesses:
- it used the models zero-shot only, although GLiNER is meant to be
  fine-tuned;
- it mostly used `small`;
- it scored against GLM labels that were only 63% right on entities;
- it tested names end to end only on first-person knowledge-update
  questions, which contain almost none.

This round fine-tunes `base-v1` on kleos's own tasks and scores it
against a gold set checked by hand.

### Tasks

For each claim (kleos's splitter, without the speaker label):

- **Entities.** Four labels:
  - `person`, `location` and `organization`;
  - `relative`: a phrase for one specific person by relation or role to
    the speaker ("my sister", "Mom", "my boss").
- **`name_of`.** A relative paired with the name the same claim gives
  ("my sister Anna"). No model is trained for it: a rule joins adjacent
  spans, or splits a `relative` span that contains a `person`.
- **`kind`.** One of seven: fact, preference, plan, event, question,
  request, small talk. *Store* means the first four.
- **`about_speaker`.** Yes or no.

### Data

**Training and validation sets.**
- 2,700 claims from LongMemEval_S user turns. They come from the 12,015
  sessions that are in *no* evaluation haystack.
  - Excluded: the 7,180 sessions of the 162 questions any bench-l2 model
    run used, and of all 78 knowledge-update questions.
  - 900 of the 2,700 were picked to contain a capitalised name.
- 1,001 claims of GLM-written personal chat and 287 of GLM-written chat
  between friends. The friend chat adds the reactions and small talk that
  the LongMemEval turns lack.
- No LoCoMo text.

**Training labels.**
- GLM-4.7-Flash without reasoning, 20 claims per call. With reasoning it
  was about 4× slower on a key with limited concurrency.
- Cleanup: names (`person`, `location`, `organization`) must contain a
  capital letter, pronouns are not entities, and every mention must be a
  whole-word match. This removed 1,017 of 3,242 silver entities, for
  example "books" labelled as a person, and "I".
- Result: 3,570 training and 396 validation claims.

**Gold set (200 claims).**
- 150 random claims: 75 from evaluation-side LongMemEval sessions and 75
  from LoCoMo.
- 50 claims picked to contain kinship or role phrases (25 + 25), because
  the random 150 contain only 3 relatives.
- First labelled by GLM *with* reasoning, then every claim read and
  corrected by hand: 59 corrections, in `scripts/gliner/ft_gold/`.

How well the GLM labels agree with the gold, on the random 150:

| Labels | Entities exactly right (per claim) | kind | store | about |
|---|---|---|---|---|
| GLM, no reasoning (training labels) | 83% | 69% | 87% | 85% |
| GLM, with reasoning (gold draft) | 95% | 81% | 90% | 84% |

### Training

- **Model and settings.** `base-v1`, 3 epochs, batch 4 × 2 accumulation.
  Encoder learning rate 1e-5, head learning rate 5e-5. Linear schedule,
  FP32 on MPS. gliner2's trainer has no MPS path, so it is patched in.
- **Frozen token table.** The 128k-row token table is half of `base`'s
  parameters. Freezing it was needed to fit the run under a 70% MPS
  watermark (about 8 GB). Without a cap, the allocator cache grew to
  14 GB and the machine swapped.
- **Bare label names, no descriptions.** The schema is encoded with every
  claim. Short labels make it about 3× shorter, which is where the speed
  gain comes from. The model learns what the labels mean.
- **Cost.** 95.7 min on the M3 with 16 GB, shared with other agents' Rust
  builds (load average 10–57, so read this as an upper bound). Process
  RSS was 2.2 GB, plus MPS memory up to the cap.
- **Thresholds.** For each model, a threshold per entity label is tuned on
  the silver validation set, then frozen.
- **Name filter.** Every model's names go through the same filter as the
  training labels: a capital letter, and no pronouns.

### Results (gold, 200 claims)

| | zero-shot `small` | zero-shot `base` | fine-tuned `base`, 1 epoch | **fine-tuned `base`, 3 epochs** | rules |
|---|---|---|---|---|---|
| person F (n = 18) | 0.86 | 0.90 | 1.00 | **1.00** | — |
| location F (n = 12) | 0.88 | 0.77 | 0.78 | **0.92** | — |
| organization F (n = 16) | 0.65 | 0.75 | 0.72 | **0.77** | — |
| relative F (n = 53) | 0.23 | 0.14 | 0.81 | **0.86** | — |
| names (person + place + org) P / R / F | 0.71 / 0.89 / 0.79 | 0.77 / 0.87 / 0.82 | 0.79 / 0.89 / 0.84 | **0.83 / 0.96 / 0.89** | capitals guess 0.29 / 0.83 / 0.42 |
| name_of pairs (n = 3) | 1 of 3 | 0 of 3 | 2 of 3 | 2 of 3, no false pairs | — |
| kind, 7-way | 0.35 | 0.22 | 0.38 | 0.47 | majority 0.20 |
| store or skip, all 200 | 0.61 | 0.60 | 0.65 | 0.725 | **0.735** (`is_declarative`) |
| store or skip, random 150 | 0.51 | 0.48 | 0.63 | **0.70** | 0.65 |
| about the speaker | 0.58 | 0.56 | 0.62 | 0.72 | **0.845** (first-person regex) |
| ms per claim, MPS, batch 8 | 129 (loaded) | 80 | — | **24.5** (34 on CPU) | — |

- **The rule baseline is exact.** `kleos_rules.py` is a line-for-line port
  of `memory_claims::is_declarative`. It agrees with kleos's own `decided`
  on 2,805 of 2,805 claims.
- **The old "0.72 for the rules" is not comparable.** That number was
  scored against the old silver labels. Against this gold set the rules
  score 0.735 overall and 0.65 on the random 150.
- **Most remaining errors are on the label boundary.**
  - The organization false positives are titles and products ("The Last
    Romantics", "Canon EOS Rebel"), which have no label of their own.
  - The relative false positives are groups ("my friends", "our
    families"), which the gold excludes and the no-reasoning labels often
    included.
- **Rule and model together do not help store or skip.** Storing only what
  the rules keep *and* the model does not call small talk scores 0.735 on
  all 200 and 0.66 on the random 150.

### Stretch: update keys from a closed vocabulary (zero-shot, not fine-tuned)

`ft_keys.py` replaces the span keys with a *classification*. Each claim gets
one attribute from a fixed list of 44 personal-fact attributes, such as
residence, employer, pets, social media or collection. It is scored on the
same 22 held-out pairs and 5,472 unrelated claim pairs as section 3.

- **True updates:** the old and the new claim get the same attribute in
  16 of 22 pairs (73%), against 9–27% for the span keys. Subject drift is
  gone. Instagram 1,250 → 1,300 and birds 27 → 32 now match; both were
  stale-first misses of main.
- **Unrelated claims:** 1,516 of 5,472 pairs share an attribute, 66 per
  question. A question's haystack clusters on one hobby, so many of its
  claims share an attribute.
- **No value is extracted,** so the key cannot supersede anything by
  itself. At best it is a blocking key in front of the decider, which the
  embedding candidates already provide.
- **Cost:** 2.9 s per claim on a loaded CPU, with 44 described labels.

This was not fine-tuned. The verdict on keys stands.

### Caveats

- **The gold set is small:** 18 persons, 12 places, 16 organizations,
  53 relatives and 3 pairs. A difference of a few points is noise; the
  `name_of` row is anecdotal.
- **The gold set is anchored on the GLM draft.** I corrected what I
  noticed, and the label conventions are mine. For example: platitudes
  are small talk, "Can you give me examples" is a request, and groups are
  not relatives. A model trained on GLM labels cannot learn conventions
  the GLM labels do not follow. That caps kind, store or skip, and about.
- **Thresholds come from noisy silver labels.** One example is location
  at 0.2 for the 3-epoch model.
- **One seed, one run.** `small` was not fine-tuned.
- **Timings are approximate:** the machine was shared and heavily loaded.

### Verdict and next step

- **Names (`remember {entities}`): adopt the fine-tuned `base` as an
  option.** It finds people mentioned by relation or role ("my sister",
  "my boss"), which zero-shot cannot (F 0.14 → 0.86). It raises names from
  F 0.82 to 0.89. It is 3× faster than zero-shot because it uses bare
  labels: 25 ms per claim on MPS, 34 ms on CPU.
- **Store or skip and kind: keep the rules.** Fine-tuning ties them
  (0.725 vs 0.735; 0.70 vs 0.65 on the random 150). Doing better needs
  better training labels: GLM with reasoning, or hand labels.
- **About the speaker: keep the regex.**
- **Phase 2 should be the end-to-end run *before* any port.**
  - Run LoCoMo with automatic names (persons, places, organizations,
    relatives with their names, and `user` from the regex), with the
    decider on. Compare with no names and with the capital-letter guess.
  - The fine-tuned checkpoint runs in Python, so this needs no candle
    work.
  - A port is worth its ~13 days only if this moves QA or the
    same-subject gate.

## Architecture (what a port has to reproduce)

Read from `gliner2` 2.0.0 (Apache-2.0) and the checkpoints' safetensors
headers. Every tensor on disk is **F32**, although the configs say
`dtype: float16`.

| Checkpoint | Arch | Encoder | Params (total / encoder / heads) | File |
|---|---|---|---|---|
| `fastino/gliner2.5-small-v1` | boundary | DeBERTa-v3-xsmall (12 × 384, 6 heads) | 73.9M / 70.6M (49.2M is the token table) / 3.2M | 296 MB |
| `fastino/gliner2.5-base-v1` | boundary | DeBERTa-v3-base (12 × 768) | 193.6M / 183.8M / 9.8M | 774 MB |
| `fastino/gliner2.5-multi-v1` | boundary | mDeBERTa-v3-base (vocab 250k) | 287.4M / 277.5M (192M token table) / 9.8M | 1.15 GB |
| `fastino/GLiNER2.5-Decide` | span (markerV0, max width 8, count LSTM) | DeBERTa-v3-large (24 × 1024) | 486M on disk (card: 340M, i.e. without the 132M token table) | 1.95 GB |

**Encoder.** HF `DebertaV2Model`, last hidden state, no pooler: relative
attention c2p+p2c, 256 log position buckets, `share_att_key`,
`norm_rel_ebd = layer_norm`, no conv layer, no `embed_proj`, LayerNorm eps
1e-7, erf GELU. The vocabulary is DeBERTa-v3's 128k SentencePiece (Unigram)
plus ten added tokens: `[SEP_STRUCT] [SEP_TEXT] [P] [C] [E] [R] [L] [EXAMPLE]
[OUTPUT] [DESCRIPTION]` (ids 128001–128010); `tokenizer.json` is shipped.

**How the schema is encoded.** The input is the schema followed by the text,
in one sequence, without `[CLS]`/`[SEP]`:

```
( [P] entities ( [E] person [E] location ... ) ) [SEP_STRUCT]
( [P] fact ( [C] subject [C] attribute [C] value ) ) [SEP_STRUCT]
( [P] type: <prompt> [DESCRIPTION] fact: <desc> ... ( [L] fact [L] preference ... ) )
[SEP_TEXT] <choice prefix> <text words>
```

- Groups are ordered structures, entities, relations, classifications. Field
  and label descriptions travel inside the `[P]` "word".
- Text is split into words by a regex (URL | e-mail | @handle |
  `\w+(?:[-_]\w+)*` | `\S`) and **lowercased**; schema words are not. Each
  word is tokenized separately; a word is represented by its first subword
  ("first" pooling), a query by its marker token (`[E]`, `[C]`, `[L]`, `[R]`).
- A choice field (`attribute::[home|job|...]`) adds its choices to the text
  side as a prefix `( fact: attribute ( home | job | ... ) , ... )`; the
  choice is then picked as a *span* over that prefix, not by a classifier.
- No truncation by default; `*_long` helpers chunk at 384 words with 64 of
  overlap.

**Boundary heads (the 2.5 small/base/multi).** About 3–10M parameters
after the encoder:

- a boundary encoder (n+1 boundary states from neighbouring tokens; two
  windowed self-attention blocks, window 128, plus one SwiGLU block);
- a query head: start, end and inside logits = dot(boundary or token, query);
- a shared candidate pool: top-32 starts × top-32 ends by the max over
  queries, a per-query quota of 8, de-duplicated, cut to 192 candidates;
- a candidate scorer: endpoint projections, length features, a span-content
  mean pool, FiLM, inside evidence; sigmoid ≥ threshold (default 0.5);
- abstention (per entity type: drop all spans if p(null) > 0.5);
- an explicit-pair scorer (rotary endpoints, 8-head compatibility) for
  choice fields and entity attributes;
- the classifier, an MLP H → 2H → 1 per `[L]` marker (softmax for one label,
  sigmoid for multi-label);
- relations (head/tail candidates ≥ 0.2, top 32 each, ≤ 64 pairs, MLP +
  biaffine) and records (anchor instances, Hungarian matching). A count head
  exists but is unused at default inference.

Overlaps are resolved by the `flat` policy: exact-duplicate removal, then
maximum-total-score weighted interval scheduling.

**Span heads (the older gliner2 and Decide).** Every (start, width < 8)
span gets a marker representation; a count head (20 classes) predicts the
number of instances per group and a GRU produces per-instance queries.

What each use needs:

| Use | Parts |
|---|---|
| Classification only (claim type) | encoder + classifier (the boundary head does not run without extractive queries) |
| Entities | + boundary encoder, query head, pool, scorer, abstention, flat overlap |
| `str` / `list` fields, legacy (one instance) | same as entities |
| choice fields | + explicit-pair scorer |
| records (several instances; what `extract_json` uses on 2.5 by default) | + candidate encoder, record decoder, Hungarian matching |
| relations | + relation generator and scorer |

### GLiNER2.5-Decide

Not a pair model. It is a **single-text classifier**: a DeBERTa-v3-large
span-architecture GLiNER2 fine-tuned (from `gliner2-large-v1`) for
operational decisions (intent, routing, sentiment, moderation, urgency,
yes/no over a passage). The card benchmarks it on `fastino/fast-decisions`
(17 domains, 60.2% exact match). `classify_text` runs the encoder and the
classifier on the `[L]` markers; the span and count heads (50M) are not
needed. A pair can be fed through it zero-shot (both statements in the text,
a yes/no question as the classification `prompt`); the measurement is in
Results. At 486M parameters in F32 (1.95 GB) it is also at the edge of the
size budget.

## Port plan (candle)

**candle-transformers 0.11 `debertav2`.** Supports what these checkpoints use
(c2p/p2c, log buckets, `share_att_key`, LayerNorm'd relative embeddings,
embedding LayerNorm) and returns the last hidden state; the conv layer and
`z_steps > 1` are `todo!()` (not used here). Read, not run: it is
effectively F32-only (the masked softmax and the attention bias build F32
constants), it rebuilds the bucket tensors on every forward, and it loads
through `VarBuilder`, so it gets none of kleos's Q8/F16 loader or mmap'd
token table. Recommendation: hand-port DeBERTa onto `kleos-infer/src/linear.rs`
as `decider_nano.rs` did for ModernBERT, with candle's file as the reference;
precompute bucket indices once per length on the CPU (U32); keep LayerNorm in
F32.

**Tokenizer.** `tokenizers` 0.22 (already a dependency) with the shipped
`tokenizer.json`, each word encoded without special tokens. The word regex
must match Python's `\w` (Unicode) and report code-point offsets, mapped to
byte offsets for Rust strings.

**Heads and decoding** are small tensors (pool 192 × queries), so run them on
the CPU in F32 after the encoder (Metal or CPU). That sidesteps Metal's
unstable `arg_sort` and F16 issues in the selection logic. Fragile spots for
parity: stable top-k and ties, the −1e4 / +5000 sentinel arithmetic in the
pool quota, prefix sums (do them sequentially in F32), 0.5 thresholds, and
weighted-interval-scheduling ties.

**Effort**, in engineer-days, against `decider_nano.rs` (637 lines + a
171-line parity test, landed in two same-day commits, with perf and memory
work later). GLiNER's encoder is about as hard as ModernBERT; the heads and
decoding are about 4–5× more logic.

| Component | Days |
|---|---|
| Word splitter, schema/prefix builder, offsets | 2 |
| DeBERTa-v3 encoder (F32/F16/Q8, Metal) | 2.5 |
| Classifier head | 0.5 |
| Entity path: boundary encoder, query head, pool, scorer, abstention, flat overlap | 4 |
| Choice and `str`/`list` fields (explicit-pair scorer, legacy one-instance structures) | 1.5 |
| Parity harness (Python fixture dumping each stage, as `decider_nano_ref.py`) | 2.5 |
| Metal and performance | 1.5 |
| **MVP total** | **~13–14** |
| Relations | 2 |
| Records (several instances, Hungarian) | 4–5 |
| Long-text chunking and merge | 1 |
| Decide classification on top of the encoder (+2 for span/count decoding) | 1 |
| **Full** | **~22–25** |

Parity targets: F32 logits within 1e-4 and identical spans and labels; F16
and Q8 with no label flips. Cases: > 128 and > 512 tokens, ties, Unicode,
several schemas in a batch, choices.

## Licences

- `gliner2` 2.0.0: Apache-2.0. All four checkpoints: Apache-2.0 (also
  `gliner2-large-v1`, Decide's base). Encoders: DeBERTa-v3 and mDeBERTa-v3, MIT.
- Training data: no model card for 2.5 or Decide states it. The GLiNER2 paper
  (arXiv 2507.18546) describes 254k examples: 136k real documents (news, law,
  Wikipedia, PubMed, arXiv) annotated by GPT-4o, and 119k synthetic examples
  generated by GPT-4o. Whether 2.5 reuses or extends it is not stated. The
  GPT-4o provenance is the same situation as other distilled extractors; worth
  a note in NOTICE, not a blocker for Apache-2.0 weights.

## Reproduce

```zsh
export G=/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr/.bench-data/gliner
export UV_CACHE_DIR=/Volumes/SSD01/caches/uv HF_HOME=/Volumes/SSD01/caches/huggingface
uv venv --python /Users/subhankarchowdhury/miniconda3/bin/python3.12 $G/venv
VIRTUAL_ENV=$G/venv uv pip install --exclude-newer 2026-10-08 "gliner2[local]" psutil requests
VIRTUAL_ENV=$G/venv uv pip install .bench-data/claims-impl/wheels/kleos-0.18.0-cp37-abi3-macosx_11_0_arm64.whl  # infer-metal
cd scripts/gliner
$G/venv/bin/python build_sample.py                          # sample.json (300 claims)
$G/venv/bin/python silver.py                                # silver.json (LLM, cached)
for m in small base multi; do $G/venv/bin/python eval_entities.py predict $m cpu; done
$G/venv/bin/python eval_entities.py score small base multi  # sections 1 and 2
$G/venv/bin/python ku_pairs.py                              # ku_pairs.json (LLM, cached)
$G/venv/bin/python ku_keys.py run small v3n trained cpu     # iterate on trained-on only
$G/venv/bin/python ku_keys.py run small v3n held cpu        # section 3
for p in a c cu; do $G/venv/bin/python policy_sim.py $p $G/runs/policy_$p.json small v2 cpu; done
$G/venv/bin/python policy_sim.py b $G/runs/policy_b_v3n.json small v3n cpu   # section 4
$G/venv/bin/python decide_pairs.py decide cpu               # Decide vs decider-nano
$G/venv/bin/python speed.py base mps 40                     # section 5, one process per model x device
```

Fine-tuned round (2026-10-24; `GLINER_DIR` defaults to `.bench-data/gliner-ft`):

```zsh
export G=/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr/.bench-data/gliner-ft
uv venv --python /Users/subhankarchowdhury/miniconda3/bin/python3.12 $G/venv
VIRTUAL_ENV=$G/venv uv pip install --exclude-newer 2026-10-10 "gliner2[local]==2.1.0" psutil requests
VIRTUAL_ENV=$G/venv uv pip install <a kleos wheel built from main>   # the claim splitter
cd scripts/gliner
$G/venv/bin/python ft_data.py split && $G/venv/bin/python ft_data.py pool && $G/venv/bin/python ft_data.py gold_extra
$G/venv/bin/python ft_data.py synth 80 && $G/venv/bin/python ft_data.py synth_chat
$G/venv/bin/python ft_data.py silver gold_draft gold_extra synth synth_chat pool   # LLM, cached
$G/venv/bin/python ft_gold.py                                # gold.json = draft + ft_gold/ corrections
$G/venv/bin/python ft_train.py build
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 $G/venv/bin/python ft_train.py train base 3 base-e3
for m in small base; do $G/venv/bin/python ft_eval.py predict $m zs-$m mps; done
$G/venv/bin/python ft_eval.py predict $G/ckpt/base-e3/final ft-base-ep3 mps
$G/venv/bin/python ft_eval.py score zs-small zs-base ft-base-ep3
$G/venv/bin/python ft_keys.py base zs-base cpu              # stretch: closed-vocabulary keys
```

LLM calls are cached under `$G/llm-cache` (no key in the cache). The
Synthetic key allows only a few concurrent requests:
`GLINER_LLM_CONC` (default 2) caps them per process.
