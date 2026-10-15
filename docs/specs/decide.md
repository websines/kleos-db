# decide(): small decision models in the database

Status: implemented (wave2/decide).

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

## Caveats

- **Small sets.** The 66 eval items give ±0.1 AUROC CIs. VitaminC claims are not memory-style notes. A labelled set of a few hundred real memory pairs per decision is the next step; see `.bench-data/research/memory-decisions.md` §4.
- **`important` is unvalidated.** `same_entity` has no adapt set, so its threshold is 0.5.
- **Fits are keyed by model id, not precision.** A probe fitted at f32 is applied at q8. The q8 logit deviation is about 0.1–0.17, with no argmax flips.
- **Scalar `decide_*` calls inside a write's default column would run per row.** Prefer `Decide`.
