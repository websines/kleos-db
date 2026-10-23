# decide(): small decision models in the database

Status: implemented (wave2/decide); fine-tune, value rules, pair cache and memory work in wave2/nano-tune.

Code:
- `kleos-core/src/decide.rs`: templates, calibration, probes and the registry;
- `decide/defaults.rs`: the shipped calibration;
- `decide_builtin.rs`: the kleos-infer backends;
- `runtime/decider.rs`: `::decider` ops and persistence;
- `fixed_rule/utilities/decide.rs`: the `Decide` fixed rule;
- `kleos-infer/src/decider_nano.rs`: the decider-nano backend.

Tests: `kleos-core/tests/decide.rs` (mock deciders) and `kleos-infer/tests/decider_nano_parity.rs`.
Evaluation: `kleos-core/examples/decide_eval.rs` and `scripts/infer-bench/decide/`.

## Goal

Small decision models answer memory-management questions in one forward pass, without generating text. A decider takes a state, a question and 2 or more options, and returns one logit per option, plus optional per-option feature vectors.

The questions are:
- does a new note supersede an old one?
- are two mentions the same entity?
- are two notes duplicates?
- do two notes contradict each other?
- how does a new note relate to an old one?
- is a note worth keeping?

kleos turns this raw contract into decisions a write path can act on safely.

## Backends

| Name | Model | Size | Licence | Status |
|---|---|---|---|---|
| `decider-nano` (default) | `tasksource/tasksource-decider-nano` (ModernBERT/Ettin cross-encoder) | 150M, 298 MB bf16 file | Apache-2.0 | Built in (feature `infer`) |
| `d1-omni` | `LiquidAI/d1-omni-600M` | 600M | LFM1.0 | Templates and calibration only. `enable` errors: "the d1-omni backend is not built into this kleos yet". The Rust port was stopped. |
| custom | anything implementing `kleos::decide::Decider` | | | `register_decider(name, Arc<dyn Decider>, family)` |

**A third backend with its own prompts** registers a template family:

```rust
use kleos::decide::{register_family, FamilyTemplates, register_decider};
let fam = register_family("jev", FamilyTemplates { spec, version: 1, calibration })?;
register_decider("jev", Arc::new(my_backend), fam)?;
```

- `spec` is a total `fn(Template) -> TemplateSpec`, with fixed strings.
- `version` is bumped on any prompt change.
- `calibration` is that family's shipped thresholds.

Everything else works unchanged for it: symmetric scoring, bands, probes, `::decider fit`, `Decide` and the functions.

## Templates (fixed, versioned)

Small decision models are very phrasing-sensitive. On the audit's set, d1-omni ranged from AUROC 0.41 to 0.97 across framings of the same question. So callers never write prompts. Each `(template, family)` has one validated prompt and a version, shown as `template@vN` in `decided_by`. Probes and calibrations fitted on another version are ignored.

| Template | Args | Symmetric | Output | d1 prompt | decider-nano prompt |
|---|---|---|---|---|---|
| `supersedes` | old, new | no | P(yes) = P(new makes old no longer true) | ContractNLI: state = old; "Classify the relationship between the old note and this new statement:\n{new}"; Entailment / Contradiction / NotMentioned; score = P(Contradiction) | NLI: state "Premise: {old}\nHypothesis: {new}"; "What is the relationship between the premise and the hypothesis?"; entailment / neutral / contradiction; score = P(contradiction) |
| `contradicts` | a, b | yes | P(yes) | as supersedes | as supersedes |
| `duplicate` | a, b | yes | P(yes) | ContractNLI P(Entailment), both directions | NLI P(entailment), both directions |
| `relation` | old, new | no | update / contradiction / compatible / duplicate | ContractNLI: Contradiction → update, Entailment → duplicate, NotMentioned → compatible | same |
| `same_entity` | a, b | yes | P(yes) | "Mention A: {a}\nMention B: {b}" + "Are mention A and mention B the same entity?" + false/true criteria | same |
| `important` | text | n/a | P(yes) | "Is this an important, lasting fact worth remembering long-term?" | same (**unvalidated**: no labelled data) |

How the prompts were chosen:
- The d1 prompts are the audit's adapt-selected framings (`scripts/infer-bench/d1/run_d1_v2.py`).
- The decider-nano prompts were chosen on the 60 adapt items only (`scripts/infer-bench/decide/nano_select.py`):
  - supersedes: NLI 0.894 vs ContractNLI 0.885;
  - contradicts: NLI 0.946;
  - duplicate: NLI-entailment 1.0;
  - relation: a 4-option "update / contradiction / compatible / duplicate" question scored 0.45 accuracy on adapt, against 0.58 for ContractNLI, so ContractNLI was kept.

Options are rendered the way each model was trained: `key: description`, with `false: …` and `true: …` for yes/no questions.

**Relation has 4 labels, but zero-shot fills 3.** The NLI framings cannot tell a change over time ("moved to Lisbon") from a correction or conflict, so zero-shot `contradiction` is always 0. A fitted probe separates them. On eval it reached 4-way accuracy of 0.905 for nano and 0.833 for d1.

Recency is not asked. The templates ask what holds between two texts. Which note is newer comes from valid time in Datalog: the models are weak at dates.

## Pipeline per decision

1. Render. For symmetric templates, render both orders. Batch up to `max_batch` per call.
2. Logits go through `/ temperature` and `+ bias[option]`, then softmax. Option mass is summed into labels, and the two orders are averaged.
3. If a probe is installed for `(model id, template)` at this template version, it replaces step 2. It is applied to the concatenated per-option features, averaged over orders.
4. Label and band:
   - Binary templates: `label = yes` iff p ≥ `threshold`.
     - p ≥ `tau_auto` → `auto`;
     - `threshold` ≤ p < `tau_auto` → `tentative`;
     - otherwise → `none`.
   - `relation`: the label is the argmax and p is its probability, with the same banding.
   - `confident = band != tentative`.

Rust type:

```rust
pub struct Decision { template, template_version, label, p, probs, band, confident, model, via }
// via: zero_shot | calibrated | probe; decided_by() = "model|template@vN|via"
```

## Calibration

`Calibration { bias: per-option log-bias, temperature, threshold, tau_auto, source }`.

- **threshold**: the accuracy-maximising cut on labelled data, with ties broken toward 0.5.
- **tau_auto, by risk control.** It is the smallest τ such that, for every τ′ ≥ τ, the false-positive share among items scored ≥ τ′ is at most α. The default α is 0.02. With `wilson: true` the 95% Wilson upper bound is used instead. The rule is monotone, so the bound does not depend on where the scan stops. If no τ qualifies, τ = 1.01 and the band never reaches `auto`.
- **bias** (relation): prior correction. A grid search over [-3, 3] in steps of 0.25 maximises argmax accuracy, with option 0 pinned at 0. This fixes d1's "update" bias.
- **Shipped defaults** (`decide/defaults.rs`) are fitted on the 60 adapt items with `decide_eval --fit-defaults`:
  - decider-nano zero-shot `supersedes` and `contradicts`: no τ meets 2% on adapt, so they never auto-apply;
  - `duplicate` auto at 0.539;
  - `relation` bias [0, 0, -0.5].

  `same_entity` and `important` have no adapt data. They use threshold 0.5 and τ 0.9.

**Caveat.** 60 items cannot certify a 2% rate: the Wilson bound with 0 errors in 60 is 6%. The defaults use the empirical rate, and the eval reports the false autos it actually saw on held-out items. Use `::decider fit … {wilson: true}` with a few hundred labels for a real guarantee.

## Probes ("the helper that gave the biggest win")

`::decider fit <decider> <template> <relation> {probe: true, alpha: 0.02, wilson: false}`

Input:
- The relation's last column is the label: a bool, 0/1 or yes/no for binary templates, or a relation label name.
- The `arity` columns before it are the texts.

Fitting:
- Features are the per-option vectors the backend's decision head reads, concatenated: 3 × 768 for nano NLI.
- For symmetric templates each order is a training row.
- The model is softmax regression with an L2 penalty `‖W‖²/(2Cn)` on standardised features, trained by Nesterov gradient descent with step 1/L. It is pure Rust with no new dependencies.
- C ∈ {0.01, 0.03, 0.1, 0.3, 1} is chosen by stratified 5-fold CV: AUROC for binary templates, accuracy for relation.
- `threshold` and `tau_auto` are fitted on out-of-fold predictions, so they are not optimistic.

Storage:
- The fit is stored as JSON in `kleos_decider_fits {model, template => kind, template_version, decider, payload, created_at}`.
- It is installed process-wide for that **model id**, so any decider name serving the same model uses it.
- It is reloaded when the database opens (`Db::initialize`).
- `{probe: false}` fits a zero-shot calibration instead: threshold, τ, and for relation the bias.

Related ops:
- `::decider fitted` lists installed fits.
- `::decider forget <decider> <template>` removes one from the process and the database.
- Fit and forget refuse to run in read-only mode or inside an imperative block.

Probe results on held-out eval (fitted on the 60 adapt items):

| Probe | decider-nano | d1 |
|---|---|---|
| supersedes AUROC (zero-shot → probe) | 0.938 → 0.993 | 0.933 → 0.995 |
| relation, 4-way accuracy | 0.905 | 0.833 |

The decider-nano supersedes probe also gives an auto band covering 38% of items with 0 false supersessions.

## CozoScript

```
::decider list
::decider enable nano {model: 'decider-nano', precision: 'f32', device: 'auto'}
::decider enable tuned {path: '/path/to/nano-tune-v1'}   # local weights (decider-nano layout)
::decider enable nano                       # re-enable as before
::decider disable nano
::decider fit nano supersedes labelled_pairs {alpha: 0.02}
::decider fitted
::decider forget nano supersedes

?[p] := p = decide_supersedes('nano', $old, $new)          # P(yes)
?[l] := l = decide_relation('nano', $old, $new)            # label
?[p, l, b] := p = decide_p('nano', 'duplicate', $a, $b),
              l = decide_label('nano', 'duplicate', $a, $b),
              b = decide_band('nano', 'duplicate', $a, $b)
?[p] := p = decide_important('nano', $text)

cand[id, old, new] := ...
?[id, label, p, band, decided_by] <~ Decide(cand[id, old, new], model: 'nano', template: 'supersedes')
```

**`::decider list`** returns `name, kind, family, model, enabled, loaded, precision, device`, mirroring `::embedder list` with `family` added.

**Errors** mirror the embedder ones:
- `eval::decider_disabled`: "decider 'x' is disabled; turn it back on with `::decider enable x`".
- `eval::decider_not_found`: names the built-in models.
- `eval::decider_feature_missing`: the build has no `infer` feature.
- `eval::decide_unknown_template`: lists the fixed templates.
- Arity errors name the template.

Null texts give null, like `embed`.

**Scalar functions run one model call per row.** Use `Decide` for many rows: it sends batches of `max_batch`, 32 for nano, and checks for cancellation between batches. Like `Rerank`, it runs inside the query's read transaction.

The registry is process-wide, like the embedder registry. Fits are per database on disk and process-wide once loaded. If two databases in one process hold fits for the same model and template, the last one opened wins.

## Rust and Python

```rust
db.enable_decider("nano", DeciderSpec::model("decider-nano"))?;
let d = db.decide("nano", "supersedes", old, new)?;            // Decision
let ds = db.decide_batch("nano", "duplicate", &[(a, b), ...])?;
let r = db.fit_decider("nano", "supersedes", "labelled", true, 0.02)?;  // FitReport
kleos::decide::decide_with(&my_decider, Family::Nano, Template::Supersedes, &items, &cal, None)?; // no registry
```

Python, built with the `infer` feature for built-ins; checked with `cargo check` only:

```python
db.enable_decider("nano", model="decider-nano")
db.decide("supersedes", old, new, model="nano")   # dict with label, p, probs, band, confident, decided_by
db.fit_decider("nano", "supersedes", "labelled")
db.deciders(); db.disable_decider("nano")
```

## Safety guidance

1. **Never auto-supersede on a low-confidence answer.** Act automatically only on `band == 'auto'`.
   - For `tentative`, record the decision as tentative, or escalate to an LLM, and let reads choose a policy: hide tentatively superseded facts, or show both with a flag.
   - For `none`, store and link, but do not supersede.
   - Zero-shot decider-nano never returns `auto` for supersedes until a probe or calibration is fitted on your data.
2. **Decisions never delete.** Supersession ends validity: kleos E1 keeps history, so `@`, `:as_of` and `::history` still answer, and a decision is reversible by retracting it.
3. **Record each decision as a row**, so it is auditable and can be re-run when the model or template changes:

   ```
   :create decisions {old: Int, new: Int, template: String => label: String, p: Float, band: String, decided_by: String}
   d[id, label, p, band, by] <~ Decide(cand[id, old_text, new_text], model: 'nano', template: 'supersedes')
   ?[old, new, template, label, p, band, decided_by] := d[id, label, p, band, decided_by], *pairs{id, old, new}, template = 'supersedes'
   :put decisions {old, new, template => label, p, band, decided_by}
   ```

   `decided_by` = `model|template@vN|zero_shot|calibrated|probe`. After a model upgrade, re-run and diff.
4. **Candidates are the precision lever.** Only score pairs about the same resolved entity or attribute, and never pairs chosen by cosine alone.
   - Use guard rules first:
     - a different subject → never supersede;
     - a different event date → compatible;
     - a numeric or date difference → never "duplicate".
   - Use valid time for order.
5. **Never merge entities transitively** from `same_entity`. Keep `same_as` edges with p.

## Evaluation (`examples/decide_eval.rs`)

Protocol:
- Fit on the 60 adapt items; report on 42 supersede items and 24 entity items.
- AUROC CIs are stratified bootstrap (2000 resamples).
- "auto" = coverage and false autos on eval at τ_auto from adapt.
- decider-nano runs through kleos-infer at f32. d1 replays recorded MPS f16 answers (`scripts/infer-bench/decide/d1_record.py`). Its Python audit numbers match.

| Decision | decider-nano 150M | d1-omni 600M |
|---|---|---|
| supersedes AUROC | 0.938 [0.833, 1.0] | 0.933 [0.847, 0.993] |
| … vs compatible / vs duplicate | 0.875 / 1.0 | 0.875 / 0.991 |
| … acc @ adapt threshold | 0.857 (thr 0.246) | 0.881 (thr 0.370) |
| … auto coverage / false (zero-shot) | 0 / 0 (τ 1.01) | 2.4% / 0 (τ 0.722) |
| … probe AUROC | 0.993 [0.972, 1.0] | 0.995 [0.977, 1.0] |
| … probe acc / auto coverage / false | 0.952 / 38% / 0 | 0.857 / 0% / 0 |
| contradicts AUROC (symmetric) | 0.944 [0.866, 0.998] | 0.942 [0.866, 0.991] |
| … probe AUROC / auto | 0.986 / 36%, 0 false | n/a |
| duplicate AUROC (symmetric) | 0.997 [0.983, 1.0], acc 0.976 | 1.0, acc 0.952 |
| same_entity AUROC (symmetric, no adapt set) | 0.924 [0.771, 1.0] | 0.986 [0.938, 1.0] |
| relation acc, 3-way, zero-shot → with adapt bias | 0.69 → 0.905 | 0.786 → 0.833 |
| relation probe, 4-way / 3-way | 0.905 / 0.952 | 0.833 / 0.905 |

The d1 probes were fitted on the forward order only, because the fixture records features for that order.

**VitaminC** (tals/vitaminc test split, real revisions):
- Sample: 300 pairs, 100 per label. Old note = claim, new statement = revised evidence; positive = REFUTES.
- It is harder than the memory items. Both zero-shot numbers below use the adapt-selected prompts.

| | decider-nano | d1-omni |
|---|---|---|
| Zero-shot AUROC | 0.679 [0.614, 0.742] | 0.414 [0.344, 0.485] (below chance) |
| Probe on 100 VitaminC train items | 0.745 [0.686, 0.799] | n/a |

Latency and memory: see the table in the commit report (decider-nano, kleos-infer, M3, while other jobs were running). Metal is about 17–25 ms per single decision. CPU needs `infer-accelerate`: plain candle CPU is 60–330 ms per decision.

## Fine-tuned weights: nano-tune-v1

A fine-tune of decider-nano on memory-shaped pairs, trained in exactly the
request format the templates render, so the Rust path needs no prompt change.

- Data: `scripts/infer-bench/decide/train/build_data.py` (seeded and reproducible) writes
  `.bench-data/nano-tune/data/`. It produces 19.5k template requests from about 10k pairs.

  | Source | Licence | Requests |
  |---|---|---|
  | VitaminC train: 2.7k claim/evidence pairs and 700 old→new revision pairs | CC BY-SA 3.0 | 3.2k |
  | LongMemEval oracle: knowledge-update pairs and multi-session negatives (training questions only) | MIT | 0.3k |
  | PAWS labeled_final | PAWS (free use, attribution) | 1.3k |
  | MultiNLI train | OANC / CC BY-SA 3.0 / CC BY 3.0 | 0.9k |
  | Programmatic (`memgen.py`) | kleos | 13.9k (half used per run) |

  The programmatic share covers 20 person attributes and 15 org/system/project attributes (update, contradiction, compatible, duplicate), entity mentions, entity typing and relation mapping.

  Never in training: the 60 adapt items, the 66 eval items, the VitaminC-300 sample (test300 and train100), and the LongMemEval held-out set (25 knowledge-update and 40 multi-session questions). Pairs that are near duplicates of any of them (token Jaccard) are dropped: 238 were.
- Training: `train.py`.
  - Forward: the shared-layout forward with gradients (tree mask, sliding window, option pooling).
  - Loss: cross-entropy over the template's options, `-log Σ_allowed p`.
  - Optimiser: AdamW (β 0.9/0.98, wd 0.01), 6% warm-up and linear decay, 1 epoch over 12.9k requests (809 steps of ≤ 2048 tokens), dropout 0.1 on pooled options.
  - Token embeddings frozen; gradient checkpointing; MPS at memory fraction 0.5. One run takes 30 minutes on an M3.
- Selection: lr ∈ {1e-5, 2e-5, 4e-5}. The rule, fixed before looking at eval, was the lowest NLL on a 6% group split of the training data, with adapt as a tiebreak (adapt saturates at ≈ 1.0 AUROC after 100 steps). It picked lr 4e-5 (val NLL 0.197 against 0.202 and 0.218).
- Weights: bf16 safetensors with the original key layout, plus `kleos_decider.json`, `{"weights_version": "nano-tune-v1"}`.
  - Load with `::decider enable nano {path: '<dir>'}` (Python `path=`, Rust `DeciderSpec { path, .. }`).
  - The weights are not published. Licence note: VitaminC and part of MultiNLI are share-alike (NOTICE.md); whether the weights inherit that is an open question for the maintainers.

Results through the Rust path (`decide_eval`, f32 on Metal). Calibration is fitted on adapt; figures are AUROC with 95% stratified-bootstrap CIs.

| | base | nano-tune-v1 |
|---|---|---|
| supersedes | 0.938 [0.833, 1.0] | **0.991** [0.965, 1.0] |
| … vs compatible / acc at adapt thr | 0.875 / 0.857 | 0.986 / 0.929 |
| … zero-shot auto coverage / false (τ from adapt) | 0% / 0 | 36% / 0 |
| contradicts | 0.944 [0.866, 0.998] | 0.991 [0.965, 1.0] |
| duplicate (acc) | 0.997 (0.976) | 0.994 (0.929) |
| same_entity (acc at 0.5) | 0.993 (0.792) | 1.0 (1.0) |
| relation acc3, zero-shot / with bias / 4-way probe | 0.69 / 0.905 / 0.905 | 0.905 / 0.905 / 0.905 |
| VitaminC-300 supersedes | 0.679 [0.614, 0.742] | **0.807** [0.758, 0.855] |
| LongMemEval-KU held-out (24 KU vs 66 multi-session pairs) | 0.386 [0.266, 0.513] | **0.823** [0.708, 0.915] |
| MultiNLI matched dev, 600 items, acc3 (forgetting check) | 0.805 | 0.817 |

Shipped defaults are keyed by weights version (`Decider::weights_version`, `Calibration::default_for_weights`):
- `base` (the HF checkpoint) keeps the defaults above.
- `nano-tune-v1` gets its own adapt fit (`decide/defaults.rs`).
- Duplicate with the tuned weights never auto-applies (τ 1.01). Adapt is perfectly separated, so the risk rule put τ at 0.091. On eval that band admitted 3 elaborations as duplicates. This choice was made after seeing the eval result.

**A default probe fitted on the training set** (500 supersedes pairs, not adapt) was evaluated and is **not shipped**:
- On the base model it helps: eval 0.972, VitaminC 0.714, LME-KU 0.771, but its auto band covers only 5%.
- On nano-tune-v1 it is no better than zero-shot: VitaminC 0.791 against 0.807, and 18 false autos on VitaminC.

## Value rules (numbers, dates, money)

`decide::rules` is a pre-check before the model, on by default (`decide::set_value_rules(false)` turns it off).

How it works:
- It extracts the value mentions of both texts: numbers, number words, money, percentages, day dates, years and month-years, weekdays, times.
- It masks them and compares the remaining subject and attribute words.
- It short-circuits supersedes and relation to "update", and duplicate to "no", with `via = "rule"` and a `reason` such as "money changed (1250 -> 1320)". It does so only when all of these hold:
  - the context clearly matches;
  - an old value was replaced (not just added);
  - only one kind of value changed;
  - there is no negation asymmetry;
  - no bound ("more than", "over") is involved.

  Otherwise the model decides.

Measured with and without the rules (Rust, f32):
- It fired 20 times across adapt, eval and VitaminC, and was right every time. The first version fired falsely on VitaminC bounds, additions and day-vs-year dates; those cases are now guarded and unit-tested.
- With the base model:
  - supersedes AUROC 0.938 → 0.947;
  - vs compatible 0.875 → 0.894;
  - relation zero-shot acc3 0.69 → 0.714;
  - zero-shot supersedes gets an auto band of 7% with 0 false;
  - VitaminC 0.679 → 0.680.
- With nano-tune-v1 nothing changes. The tuned model already gets these pairs right.

## Pair-decision cache

Raw answers are cached process-wide.
- Key: (model id, weights version, family, template@version, rendered request).
- Budget: 16 MiB by default, a two-generation LRU like the embedding cache (`decision_cache_stats`, `set_decision_cache_capacity`, `clear_decision_cache`).
- Duplicates within one batch go to the model once.
- (Re)registering or disabling a decider drops its model's entries, because the weights behind a path may change.
- The precision is not part of the key.

## Precision and memory

Measured with `kleos-infer/examples/decider_bench.rs` on an idle M3 (16 GB) with nano-tune-v1:
- Footprint is the process phys_footprint with the default allocator.
- "after b16" means after load plus a batch of 16 decisions. "Flips" counts label changes against f32 over 922 decisions.

| Device, precision | Load peak | After b16 | b1 p50/p90 ms | ms/decision at b16 / b64 | Flips vs f32 (max \|Δp\|) |
|---|---|---|---|---|---|
| Metal f32 | 658 MB | 698 MB | 29.6 / 32.6 | 33.3 / 31.1 | – |
| Metal f16 | 373 MB | 415 MB | 26.6 / 28.1 | 26.7 / 26.7 | 1 (0.008) |
| **Metal q8 (default)** | **248 MB** | **287 MB** | **24.7 / 26.4** | 26.2 / 25.6 | 1 (0.16) |
| CPU f32 (Accelerate) | 618 MB | 694 MB | 47.7 / 49.4 | 33.9 / 32.0 | – |
| **CPU q8 (default)** | 189 MB | 302 MB | 88.7 / 111.4 | 112.4 / 103.9 | 2 (0.17) |

Before this work: base f32 had 629 MB RSS and 1.39 GB footprint after a batch of 64 on Metal, and about 1 GB RSS on CPU.

What changed:
1. **Activation buffers.** Candle's Metal pool keeps power-of-two buffers sized for the largest batch. GPU memory scaled with tokens per forward pass (f16: 431 MB at 256 tokens, 1017 MB at 8192) and was not returned.
   - The decider now caps a forward pass at 1024 tokens (`DECIDER_BATCH_TOKENS`). This costs no throughput on Metal, where batching never paid.
   - It synchronises once more after the intermediates are dropped, so the pool releases them.
2. **Load transients.** The token table (77 MB bf16, 155 MB f32) was converted in 16k-row chunks and joined with `cat`. That left full-size blocks in libmalloc's large-block cache, and a power-of-two 128 MB buffer on Metal.
   - Dense and q8 tables are now converted in 1k-row chunks into an anonymous-mmap staging area (returned to the OS at unmap) and copied once into exact-size storage.
   - Metal q8 dropped from 403 MB to 248 MB at load. `malloc_zone_pressure_relief` was tried and does not help.
3. **Defaults.** Built-in decider-nano now defaults to q8 on every device. Pass `precision: 'f16'` (Metal) or `'f32'` for higher fidelity.
   - On CPU, f32 with Accelerate is about 2× faster than q8 (candle's own int8 GEMM) at 2.3× the memory.

Still open:
- CPU q8 grows to about 300 MB after the first batches (activations in the malloc cache). The 250 MB target is missed.
- **Batching gives no speedup on Metal** (25 ms per decision at b1 and at b64). The forward pass is GPU-bound at batch 1 with the dense f32 attention and masks. The cause was not fixed here: probable levers are a fused masked SDPA and f16 activations.

## Caveats

- **Small sets.** The 66 eval items give ±0.1 AUROC CIs. VitaminC claims are not memory-style notes. A labelled set of a few hundred real memory pairs per decision is the next step; see `.bench-data/research/memory-decisions.md` §4.
- **`important` is unvalidated.** `same_entity` has no adapt set, so its threshold is 0.5.
- **Fits are keyed by model id, not precision.** A probe fitted at f32 is applied at q8. The q8 logit deviation is about 0.1–0.17, with no argmax flips.
- **Scalar `decide_*` calls inside a write's default column would run per row.** Prefer `Decide`.
