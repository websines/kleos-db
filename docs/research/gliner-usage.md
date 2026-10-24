# Using GLiNER well: what the papers, code and practitioners say

*2026-10-24, branch `wave5/gliner-research`. Research only; no kleos code
changes. It complements [`gliner.md`](gliner.md) (spike, fine-tune, port plan)
and feeds the Rust port's wiring (M3), the next fine-tune, and relation edges.*

Evidence tags:
- **[V]** verified: read in the paper, the official README or tutorial, or our
  own reading of the gliner2 code.
- **[R]** reported: a blog, model card, third-party docs or issue. Not checked
  by us.
- **[K]** kleos measurement, from `gliner.md`.

## What kleos does today (the baseline being judged)

- **Model:** fine-tuned `gliner2.5-base-v1` (194M), with four bare labels
  (`person`, `location`, `organization`, `relative`) and no descriptions.
- **Training:** 3 epochs, encoder lr 1e-5, task lr 5e-5, effective batch 8,
  linear warmup 0.1, token table frozen. The data was about 2.7k LongMemEval
  claims plus 1.3k synthetic chat claims, labelled silver by GLM-4.7-Flash
  without reasoning (`scripts/gliner/ft_train.py`, `ft_data.py`).
- **Inference:**
  - per **claim**;
  - per-label thresholds tuned on the silver validation split;
  - a capital-letter filter on person, location and organization (the relative
    label is exempt);
  - relative spans that contain a name are split (`split_relatives`).
- **Gold set, 200 claims [K]:** names F 0.89, relatives F 0.86, 24.5 ms per claim
  on Metal at batch 8.
- **Planned wiring:** `user.<relation>` keys, "my sister Anna" giving both, and
  `user` from a first-person regex.

## Answers to the eight questions

### 1. Labels: names, descriptions, how many

- **Descriptions are the official zero-shot advice.**
  - The GLiNER2 README and tutorial accept a `{label: description}` dict and say
    descriptions improve accuracy [V].
  - Pioneer's fine-tuning guide says to add a one-sentence description when a
    type is ambiguous or over-fires [R].
- **Descriptions are not free.**
  - In GLiNER2 the schema is encoded *jointly* with the text: `[P] entities
    ([E] person [E] …) [SEP] text`, with descriptions inside the prompt [V],
    paper §3 and our code read.
  - So every description token is paid on every call and can't be cached.
- **Label count costs latency.** GLiNER2's CPU table [V] shows 130 ms at 5
  labels, 132 at 10, 163 at 20 and 208 at 50.
- **Uni-encoder limit:** the original GLiNER family supports "up to about 50
  entity types" in one uni-encoder pass [V]. The bi-encoder exists for large
  label sets.
- **Training practice [V]:**
  - GLiNER varies the number of prompted types per example (**entity type
    dropping**, worth about +1.4 F1 out of domain);
  - it caps types at 25 per sentence;
  - GLiNER multi-task caps at 30 labels.
- **Verdict on our bare labels:** fine for a *fine-tuned, fixed* label set. The
  label meaning was learned from data, which is exactly what a fine-tune is for,
  and the shorter schema is why it became about 3× faster [K].
  - **The rule that matters:** the inference schema must match the training
    schema token for token. The same labels, in the same order, and no
    descriptions if trained without them. A description added at inference is a
    distribution shift.
  - **Any new zero-shot label** added later should go in *with* a description,
    or be trained in.

### 2. Thresholds and overlap

- **Defaults:**
  - GLiNER's paper and README use 0.5 [V];
  - GLiNER multi-task uses 0.5 for NER [V];
  - GLiNER-Relex uses τE = 0.3 and τR = 0.5 [V];
  - Neo4j's GLiNER2.5 extractor defaults to 0.5, with a separate
    `relation_threshold` [R].
- **No official per-label guidance exists.** The best-supported practice is to
  sweep per label on a *labelled* validation set [R, general calibration
  literature].
- **Scores are not calibrated probabilities**, and are not comparable across
  models [R] (Neo4j caveats).
- **Decoding (GLiNER paper) [V]:**
  - spans need a score above the threshold;
  - **flat** decoding greedily takes the best non-overlapping span;
  - **nested** decoding allows full nesting but no partial overlap;
  - spans can be at most 12 words wide.
- **GLiNER2's `overlap_policy` [V]:** `allow`, `nested`, `flat`/`disallow` or
  `longest`; `None` keeps the checkpoint's default.
- **For kleos:**
  - Keep per-label thresholds, but tune them on **hand-checked** data. We tuned
    on silver, which is the weakest part of the current setup.
  - Keep `flat`. "My sister Anna" is two adjacent, non-overlapping spans if the
    relative label is trained without the name, which `split_relatives` already
    enforces at eval.
  - Store the raw score on the mention row, so thresholds can change later
    without re-extracting.

### 3. Context: claim, message, previous turn, speaker

- **Practitioner advice is per message**, with long input windowed at 384 words
  and 64 of overlap [R] (Neo4j: "per-message extraction"; `max_words=384`,
  `chunk_overlap=64`). The GLiNER2 long-text API uses the same 384/64 defaults
  [V].
- **Relations degrade past about 400 words**, and relations whose endpoints fall
  in different windows are lost silently [R] (Neo4j).
- **No source evaluates feeding the previous turn.** Neo4j matches relation
  endpoints across messages *after* extraction: first the same call, then the
  same message, then a case-insensitive graph lookup [R]. That's a resolution
  step, not more context for the model.
- **gliner2 lowercases the text words** (schema words are not lowercased) [V,
  code]. So the model never sees case. That explains why raw output tags
  "farmers" as a person, and why a case-based filter outside the model adds real
  signal [K].
- **For kleos:**
  - **Extract once per statement (message), not per claim.** Then assign each
    span to the claim whose byte range contains it; kleos's claim splitter has
    the offsets.
    - That gives one model call per message instead of one per claim, faster
      in total.
    - The model sees "my sister Anna … she" together.
    - Cross-clause appositives are kept.
    - It's what practitioners do.
  - **But re-check the gold-set F at message granularity first.** The fine-tune
    trained and was scored on claims, and message-level input is a
    distribution shift.
  - **Don't feed the previous turn.** Resolve across turns in kleos instead (§5).
  - **Strip the `Name:` speaker prefix before extraction.** Otherwise the speaker
    is tagged in every claim. Add the speaker deterministically instead: `user`,
    or the speaker's own entity in multi-party chat, by the same first-person
    rule as `user`.

### 4. Relations (typed edges)

What exists:
- **GLiNER2 / GLiNER2.5:**
  - `extract_relations(text, ["works_for", "lives_in", …])`, or relations with
    descriptions in a schema; batch variants exist [V, README].
  - In the 2.5 boundary checkpoints, relation decoding is head/tail candidates
    ≥ 0.2, the top 32 of each, at most 64 pairs, and an MLP plus biaffine scorer
    [V, our code read].
  - Neo4j's wrapper enforces endpoint types *during* decoding, from an ontology,
    and normalises types to UPPER_SNAKE [R].
- **GLiREL (NAACL 2025):** relation *classification* given entity pairs; state of
  the art on FewRel and WikiZSL [V, abstract]. It needs the entities supplied.
- **GLiNER multi-task:** relations framed as the label "{head} <> {relation}"
  extracting the tail; FewRel 87.4 F1 with gold heads [V].
- **GLiNER-Relex (2026):** joint NER and relation extraction on DeBERTa-v3-large
  [V].

How reliable it is end to end (entities predicted, then relations):

| System | CoNLL04 | DocRED | FewRel | CrossRE | Avg. | Source |
|---|---|---|---|---|---|---|
| GLiNER2 | 32.9 | 11.7 | 20.8 | 6.0 | 17.8 | Relex paper, Table 2 [V] |
| GPT-5-mini | 42.4 | 18.6 | 15.0 | 12.4 | 22.1 | same [V] |
| GLiNER-Relex | 40.4 | 31.3 | 12.5 | 18.1 | 25.6 | same [V] |
| GLiNER2.5-base, POLE+O gold | relation F1 0.37 lenient / 0.27 strict (entities 0.80) | | | | | Neo4j docs [R] |

- **Verdict:** zero-shot general relations from any model this size are weak,
  roughly 0.2–0.4 F1. Even GPT-5-mini is only 22 F1 on these benchmarks.
- **Don't write general relations** (works_at, lives_in) as trusted facts without
  confirmation.
- **First-person kinship is the easy special case.** "my sister Anna", "Anna, my
  sister", "my boss Tom": the relation is carried by the possessive plus the
  `relative` span plus an adjacent or appositive person span. A rule over the
  spans is precise, and our fine-tune already finds the spans (relatives F 0.86).
  §(c) below has the plan.

### 5. Entity linking and coreference

- **GLiNER doesn't resolve coreference**, and none of the GLiNER pipelines we
  found do it. Neo4j's docs: "Coreference: not covered" [R].
- **GLinker / GLiNKER** (Knowledgator) links *to a knowledge base* such as
  Wikidata. It runs NER, then layered candidate search (Redis → Elasticsearch →
  Postgres), then a bi-encoder or reranker [R, PyPI and arXiv 2602.18487]. It's
  built for a big external knowledge base, not personal entities.
- **Neo4j agent memory resolution [R]:**
  - candidates are keyed on normalised name plus type;
  - a confident match reuses the node;
  - a near match creates a new node plus a **pending `SAME_AS`** edge for review;
  - "Maya" doesn't merge with "Maya Chen" on confidence alone.

  This is the same design as kleos's Auto / Tentative / None tiers in
  `entity.rs`, which is reassuring.
- **Small coreference models exist:** Maverick (ACL 2024) reports CoNLL-2012
  state of the art with about 500M parameters [R]. That's at the edge of our
  size budget, a second large model, and trained on documents, not chat.
- **For kleos:**
  - Keep the decider's `same_entity` as the merge authority. GLiNER produces
    *mentions*, and `entity.rs` decides identity, as now.
  - **Pronouns:** don't add a coreference model. Within one statement, a claim
    whose subject is `he`, `she` or `they` and that has no name should inherit
    the most recent person or relative mention of the same statement. That's a
    rule, cheap, and it needs the per-statement extraction from §3.
    - Today a statement name goes to *every* claim only when *no* claim contains
      it, so "she" claims miss Anna whenever another claim names her.
  - **Across turns:** a bare "my sister" resolves to the namespace's existing
    sister entity by key, then by the decider if there are several.

### 6. Fine-tuning recipe

| Recipe | Data | LR (encoder / task) | Length | Other |
|---|---|---|---|---|
| GLiNER [V] | Pile-NER, 45k passages, 13k types | 1e-5 / 5e-5 | ≤30k steps, cosine, 10% warmup | 50% negative types; type dropping; ≤25 types per sentence |
| GLiNER multi-task [V] | Llama-3-8B synthetic, then 50% synthetic + 50% curated | 1e-5 / 5e-5, then 5e-6 / 7e-6 for 1k steps | 120k steps | BCE weighted 0.75 / 0.25; label smoothing helps self-training |
| GLiNER2 [V] | 254k: 136k GPT-4o-annotated real + 119k GPT-4o synthetic | 1e-5 / 2e-5 | 5 epochs | AdamW, wd 0.01, 1k warmup, clip 1.0 |
| GLiNER2 tutorial [V] | — | default 1e-5 / 5e-4; "fine-tuning/production" 5e-6 / 1e-4 | 10–20 epochs (3 for prototyping) | LoRA r 8–16, alpha 2r, adapters 2–10 MB; early stopping optional |
| GLiNER-Relex [V] | 1M Qwen3-32B-annotated, then **about 3k Gemini-annotated, two passes** | 1e-5, then **1e-6** | 1 epoch, then 5 | focal loss α 0.75 |
| GLiNER-BioMed [R/V] | 70B teacher labels 10k → LoRA 8B student labels about 105k; plus general-domain post-training | — | — | few-shot prompt, **noun-phrase candidates given to the labeller**, JSON-constrained output; domain plus general data "complementary" |
| **kleos** [K] | 2.7k LME + 1.3k synthetic claims, GLM silver *without reasoning* | 1e-5 / 5e-5 | 3 epochs | token table frozen |

Findings:
- **Negative types matter most in the paper's ablation [V].** Sampling 50% of
  absent labels as negatives gives F1 60.9, against 53.3 with none and 58.6 at
  75%. Fixed four-label training means every example already prompts all four,
  so all absent ones are negatives. What we lack is **hard negatives**:
  - generic nouns that look like names ("books", "farmers");
  - "I" and pronouns;
  - groups for `relative`.

  `urchade/gliner` has an `ner_negatives` key for this [R]. In GLiNER2 the
  equivalent is simply having such text with empty outputs.
- **Label quality is the ceiling** (multi-task, BioMed and Relex all spend on
  better labels). Our non-reasoning GLM labels agreed with gold only 69% on
  kind, and we removed 1,017 of 3,242 silver entities [K]. Relex's stage 2 used
  about 3k examples labelled in two passes; BioMed gives the labeller
  noun-phrase candidates and a JSON schema.
- **Forgetting:** mixing general data is the documented defence [V]. Multi-task
  post-trained 50/50; BioMed found domain plus general complementary. LoRA, which
  freezes the base, is the tutorial's alternative.
- **Self-training (pseudo-labels, then retrain) [V]:** up to +2 F1, mostly where
  the base is weak (below about 0.6). Models above 0.6 showed no change or a
  slight drop. Our names and relatives are already 0.86–0.89, so it's low value.
- **Learning rates:** ours (1e-5 / 5e-5) are the GLiNER defaults. The tutorial's
  "production fine-tune" is lower on the encoder (5e-6), and Relex's small
  stage 2 used 1e-6.

### 7. Deployment

- **FP16** halves memory with no stated loss; `torch.compile` gives about 1.5×
  (urchade README) [V].
- **INT8:** urchade's README says INT8 needs **quantization-aware training** to
  keep accuracy [V]. Community GLiNER INT8 ONNX builds report:
  - weight-only INT8 matching FP16 on a small test (19/22 recall);
  - a mixed INT4/INT8 build losing more than half its recall (10/22) [R].

  A DeBERTa-v3 classifier card reports dynamic INT8 collapsing recall (6.5% vs
  36.5%) [R]. Activation quantization of DeBERTa's disentangled attention is the
  suspected weak point [R].
- **For kleos:** F16 on Metal by default.
  - **Q8 weight-only** (our `linear.rs` Q8 is weight-only) is plausible.
  - Keep LayerNorm, the relative-position path and the heads in F32.
  - **Gate Q8 on the gold set with zero label flips**, which the port's M2 already
    requires.
- **Batching:** batch 8 is the common default (README, Neo4j) [V/R]. Our Metal
  numbers already use it [K].
- **Max length and chunking:** the 384-word windows with 64 overlap above. The
  GLiNER2 README says to raise the overlap when entities sit near boundaries [V].
- **Lowercasing gotcha [V, code]:** text words are lowercased before
  tokenisation. Any case-based rule must read the **original** text by span
  offsets, and the word-offset mapping must be built on the original string.
- **Loader gotcha [V, README]:** `GLiNER2.from_pretrained` doesn't load 2.5
  boundary checkpoints; use `AutoExtractor`.
- **Splitter gotcha [V, README]:** the word splitter isn't saved with checkpoints,
  and changing it hurts quality. The Rust port must hard-code the whitespace
  splitter's regex.

### 8. Pitfalls others hit

All [R] unless marked.
- **Silent empty results:** a hard-constraint decode failure returns empty with a
  warning, which looks like "no entities". Log it.
- **Cross-window relations are lost** silently; recall falls past about 400
  words.
- **Raw output over-fires** on common nouns (our own finding [K]), and the same
  span can come back under two labels. With `flat`, or a capital filter, this
  goes away.
- **Uncalibrated scores:** don't compare thresholds across checkpoints.
- **Requesting relations doesn't mean an edge was written:** check the stored
  result.
- **The GLiNER2.5 architecture change** brought no clear accuracy gain over v1 in
  Neo4j's tests (entities F1 roughly level).
- **Informal text:** on tweets, GLiNER-L trails UniNER-7B (Broad Tweeter 61.2 vs
  67.9) [V, paper]. Chat is a weak spot for zero-shot models, which argues for
  fine-tuning, as we did.

## Recommendations for kleos

### (a) The Rust port's wiring (M3)

1. **Granularity:** run GLiNER once per statement, windowed at 384 words with 64
   overlap if longer, and map spans to claims by byte offset. Check the gold-set F
   at message granularity first; if it drops, fall back to per claim, or train on
   message inputs (b2).
2. **Exact schema parity:**
   - the four bare labels, in training order, with no descriptions;
   - the whitespace splitter regex;
   - lowercase the text words, but keep the original-string offsets.
3. **Speaker prefix:** strip `Name:` before extraction. Add the subject
   deterministically: the first-person regex gives `user`, or the speaker's
   entity in multi-party chat.
4. **Thresholds:** per label, from a config table loaded with the model. Store
   the raw score and the extractor version (`decided_by: gliner2.5-kleos-ft-v1`)
   on each mention, so thresholds can move and old memories can be re-extracted
   later (a durable-workflows job).
5. **Capital filter:** keep it for person, location and organization, but **only
   when the statement has mixed case**. In all-lowercase chat ("anna said"),
   trust the model's score instead, with a higher threshold. `relative` is always
   exempt.
6. **Relatives are possessor-relative, not always the user's.**
   - The training definition includes "his wife", "her brother" and "Mom"
     (`ft_data.py`).
   - Map only first-person forms to `user.<relation>`: "my", "our", or a bare
     kin term like "Mom" from the speaker.
   - For "his/her/their X", attach to the most recent person or relative mention
     in the statement (`<that entity>.<relation>`). If there isn't one, store the
     mention with no possessor key and let `entity.rs` decide.
7. **Make the relative a node with a key, not just a key.** Create a person node
   for "my sister" with lookup key `user.sister`, plus a typed edge (see c).
   When a name arrives ("my sister Anna"), merge the name into that node through
   `same_entity` and don't create a second node. Several sisters, or a changed
   referent (a new boss), become separate nodes, settled by the decider and by
   valid time.
8. **Pronouns:** a claim whose subject is a pronoun inherits the most recent
   person or relative of the same statement (§5). Never resolve across turns
   with the model.
9. **Resolution:** unchanged, through `entity.rs`. This matches the published
   Neo4j pattern, where a near match gives a pending SAME_AS.
10. **Precision on Metal:** F16 by default. Q8 weight-only allowed only with zero
    label flips on the gold set; LayerNorm, relative position and heads in F32.
11. **Observability:** count empty extractions and decode warnings per statement,
    and expose them in `stats`.

### (b) The next fine-tune

1. **Better labels before more data.** Relabel the training pool with GLM
   *reasoning on*, two passes (Relex-style), giving the labeller candidate noun
   phrases and a JSON schema (BioMed-style). Keep growing the hand gold set: it's
   cheap, and it's what thresholds should be tuned on.
2. **Train on message-level inputs too**, with span offsets, if (a1) shows a drop
   at message granularity.
3. **Hard negatives:** add sentences with generic plural nouns, "I" and pronouns,
   groups ("my kids", "my friends") and generic roles ("a doctor"), all with empty
   outputs for the tricky label. Also include all-lowercase variants of
   positives, since the model is case-blind but the filter isn't.
4. **Prevent forgetting:** mix 30–50% general NER data (person, location and
   organization from a permissive corpus) with the chat data. Or train a **LoRA
   adapter** (r 16, alpha 32) on a frozen base, which also makes the kleos model a
   2–10 MB add-on to upstream weights.
5. **Learning rate:** try the tutorial's fine-tune setting (encoder 5e-6, task
   1e-4) with early stopping on gold-like validation, against the current
   1e-5 / 5e-5.
6. **Keep bare labels and the frozen token table.** Nothing contradicts either:
   the token table freeze is a memory trick the sources don't discuss, and LoRA
   is the documented superset of it.
7. **Skip self-training.** The reported gains are below 0.6 F only.
8. **Add a possessor signal** for `relative`, if cheap: either a separate label
   for first-person kin ("my sister") vs third-person kin ("his wife"), or keep
   one label and rely on the rule in (a6).

### (c) The relation edges plan

1. **Phase 1, rules on spans, on by default.** First-person kinship and role
   edges:
   - `user -[sister]-> <node>` from "my sister (Anna)", "Anna, my sister" or "my
     sister, Anna";
   - the relation type is the normalised head noun of the `relative` span (sister,
     boss, partner…), mapped through a small table into a closed set;
   - p = min(relative score, person score);
   - written to `kleos_mem_link` in `remember`'s transaction, with `decided_by:
     gliner-rule` and the memory as source.
2. **Edges have valid time:** a new `user -[boss]-> Y` supersedes `user -[boss]-> X`
   when the decider confirms (`supersedes` on the two statements), because roles
   change. Kinship edges are never auto-superseded (two sisters are both true).
3. **Phase 2, GLiNER2.5 relations, behind a flag, tentative only:**
   - relations `works_at` (person → organization), `lives_in` (person →
     location) and `relative_of` (person → person);
   - **endpoint types enforced** at decode, as Neo4j does;
   - relation threshold about 0.5;
   - only within one window (384 words).

   Write them as **pending** edges (`p`, `decided_by: gliner-rel`). Promote one
   only when the decider agrees, or when the same edge recurs in two or more
   statements. Expected quality is about 0.27–0.37 F1 zero-shot, so never trust
   it directly.
4. **Evaluate before defaulting phase 2:**
   - a hand check of 100 LoCoMo edges;
   - the life stream's sister and user conflicts;
   - `recall` with entity expansion over edges, on LoCoMo multi-hop.
5. **Not doing:** coreference models, or knowledge-base linking (GLinker), for
   personal memory.

## Where this contradicts what we did

| Our choice | Verdict |
|---|---|
| Bare labels, no descriptions | **Fine for a fine-tuned fixed set.** It contradicts only *zero-shot* advice. The inference schema must match training exactly |
| Capital-letter filter | **Supported in principle:** the model is case-blind because gliner2 lowercases text. **Risky on all-lowercase chat**, so gate it on mixed case (a5) |
| Frozen token table | **Not discussed anywhere.** LoRA on a frozen base is the documented alternative and a superset |
| Per-label thresholds | **Consistent with practice, but tuned on silver.** Tune on hand gold and store raw scores |
| `user.<relation>` keys | **Partly wrong.** The `relative` label also covers "his wife" and "her brother", which aren't the user's. Use a node plus an edge, with the possessor resolved (a6, a7) |
| Per-claim extraction | **Practitioners extract per message**, with 384-word windows. Per statement with offset mapping is cheaper and keeps cross-clause context; check first that F doesn't drop |

## Sources

Papers:
- GLiNER, NAACL 2024: https://arxiv.org/abs/2311.08526 (HTML: https://arxiv.org/html/2311.08526v1)
- GLiNER multi-task: https://arxiv.org/abs/2406.12925 (HTML: https://arxiv.org/html/2406.12925v2)
- GLiNER2: https://arxiv.org/abs/2507.18546 (HTML: https://arxiv.org/html/2507.18546v1)
- GLiREL, NAACL 2025: https://arxiv.org/abs/2501.03172 and https://aclanthology.org/2025.naacl-long.418
- GLiNER-Relex: https://arxiv.org/abs/2605.10108 (HTML: https://arxiv.org/html/2605.10108)
- GLiNER bi-encoder / GLiNKER, "The Million-Label NER": https://arxiv.org/abs/2602.18487
- GLiNER-BioMed: https://arxiv.org/abs/2504.00676
- Maverick coreference, ACL 2024: https://aclanthology.org/2024.acl-long.722

Code and docs:
- fastino-ai/GLiNER2 README: https://github.com/fastino-ai/GLiNER2
- GLiNER2 training tutorial: https://raw.githubusercontent.com/fastino-ai/gliner2/main/tutorial/9-training.md
- urchade/GLiNER README: https://github.com/urchade/GLiNER
- GLinker on PyPI: https://pypi.org/project/glinker/
- gline-rs, Rust inference: https://docs.rs/crate/gline-rs/latest
- Neo4j agent memory:
  - extractor classes: https://neo4j.com/labs/agent-memory/reference/extractor-classes/
  - extraction pipeline: https://neo4j.com/labs/agent-memory/reference/extractor-pipeline/
  - how extraction works: https://neo4j.com/labs/agent-memory/explanation/extraction-pipeline/

Model cards and posts (reported):
- GLiNER small v2.5 weight-only INT8: https://huggingface.co/Tazner/Gliner-small-edge-wo-int8-emb4
- GLiNER2.5 multi ONNX: https://huggingface.co/jugaadsrl/gliner2.5-multi-v1-onnx
- GLiNER2 on Towards Data Science: https://towardsdatascience.com/gliner2-extracting-structured-information-from-text/
- DeBERTa-v3 INT8 recall collapse (classifier): https://huggingface.co/Bei0001/deberta-merge-verifier-onnx
- GraphRAG NER reference: https://graphrag.com/reference/preparation/ner/
