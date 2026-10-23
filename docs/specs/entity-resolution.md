# Entity resolution (E3)

Status: implemented (wave2/entity).

Code:
- `kleos-core/src/entity.rs`: config, names, candidates, decisions, merges, the `::entity` ops;
- `parse/sys.rs` and `cozoscript.pest`: `::entity` parsing;
- `kleos-lib-python/src/lib.rs`: the Python methods.

Tests:
- `kleos-core/tests/entity.rs`, with a fake decider and a fake embedder, plus one `#[ignore]` real-model test;
- `kleos-lib-python/tests/test_entity.py`.

Evaluation: `kleos-core/examples/entity_eval.rs`, with the mention set in `examples/data/entity_mentions.jsonl`.

## Goal

"Alice Smith", "A. Smith" and "Alice (my manager)" should resolve to one entity when the evidence is strong. When it is not, they stay separate. Every merge must be auditable and reversible.

UPGRADE_PLAN E3 states it as: candidates from LSH, vector and FTS, confirmed by `decide()`, merged with history kept.

## Configuration

Resolution works over **the user's own entity relation**. It is configured by name, for example:

```
:create person {id: String => name: String, type: String? default null,
                aliases: [String]? default null, description: String? default null,
                emb: <F32; 256>? default null}
::lsh create person:lsh {extractor: name, tokenizer: NGram(3, 3, false),
                         filters: [Lowercase], n_perm: 64, target_threshold: 0.3, n_gram: 1}
::fts create person:fts {extractor: name, tokenizer: Simple, filters: [Lowercase]}
::hnsw create person:vec {dim: 256, m: 16, dtype: F32, fields: [emb], distance: Cosine}
```

`EntityConfig` (Rust), or the option keys of `::entity resolve`:

| Key | Meaning | Default |
|---|---|---|
| `id`, `name` | key column (one column) and name column | `id`, `name` |
| `type`, `aliases`, `description`, `embedding` | optional columns | none |
| `lsh`, `fts`, `hnsw` | index names on the relation (`person:lsh` → `lsh`) | none |
| `embedder` | embedder for mention vectors and new rows | none |
| `decider` | decider for `same_entity`. If none, rules and MinHash only | none |
| `compatible_types` | groups of types that may match, e.g. `[['org', 'company']]` | equal types only |
| `k`, `max_candidates` | hits per index source; candidates sent to the decider | 10, 5 |
| `threshold`, `tau_auto` | override the decider's same_entity calibration | the decider's |
| `exact_auto` | an exact unique name always auto-resolves | false |

For the LSH index, character trigrams (`NGram(3, 3, false)` with `n_gram: 1`) are the right shingles for names. Word n-grams miss typos.

Columns that resolution does not write must have defaults: on commit it writes only the configured columns.

## Pipeline

1. **Snapshot.** One scan of the relation reads ids, names, types, aliases and descriptions. Another reads the current merges (`@ 'NOW'`) and the rejected review links. Blocking keys are built in memory:
   - exact normalized name and each alias;
   - first token, plus the nickname group of the first token;
   - last token;
   - acronym.

   Keys with more than 200 ids are skipped, except exact names. This costs one scan per resolve call, not per mention: batch mentions.
2. **Normalization** (`parse_name`):
   - NFKD with accents dropped, lower case, punctuation removed;
   - apostrophes dropped (`O'Brien` → `obrien`), `&` → `and`;
   - leading honorifics dropped (Dr, Mr, Ms, Prof, …), and trailing name suffixes (Jr, PhD, …);
   - trailing company suffixes dropped (Inc, Ltd, Corp, LLC, …), and a leading `the`;
   - text in parentheses kept as a remark ("Alice (my manager)" → `alice` + "my manager"). The remark is used as a description.
3. **Candidates.** The union of:
   - **rule** hits that fire a name rule: `exact`, `alias`, `initial` (A. Smith / Alice Smith), `nickname` (Bob Jones / Robert Jones, from a table of about 65 English nickname groups), `middle_name`, `acronym` (IBM, M.I.T., UN), `first_name` (Alice / Alice Smith), `surname` (Smith / Alice Smith), `subset`;
   - **LSH**: `~rel:lsh{… | query: q}` for every mention, in one query;
   - **FTS**: an OR query over the name tokens, in one query;
   - **vector**: an HNSW search on the mention's embedding, in one query, when an embedder is configured and enabled.

   Every hit is mapped to its **canonical** entity, and hits are deduplicated per canonical entity. Kept: same or compatible type, not the mention's own cluster, and no rejected link. Candidates are ranked by rule strength + 2 × trigram Jaccard + source bonuses. The top `max_candidates` go on.
4. **Rule decisions.** An exact or alias match that is the **only** such match decides alone (`decided_by = rule|exact`) when either:
   - there is no decider; or
   - the name is long and specific (≥ 4 tokens or ≥ 25 characters); or
   - neither side has any context.

   A short exact name with context goes to the model, because "Alice Smith (my dentist)" and "Alice Smith (engineering manager)" can be two people. So does an exact name that is ambiguous (two entities share it). Once a rule has settled a mention, its other candidates are not sent to the model.

   On commit, a new row's description is its description, or its parenthetical remark, or else its context (clipped). This way later decisions see the evidence the entity came with.
5. **Model decisions.** All remaining (mention, candidate) pairs of the call go to `decide()` with `same_entity` in one batch. The decider batches them by its `max_batch` and caches pairs.
   - The mention side is rendered as: name (type), aliases, description or remark, then `Context: "…"`.
   - The entity side is the canonical **profile**: name (type), "also called" the names of every row merged into it plus its aliases, and up to 3 descriptions.
6. **Outcome.**
   - **Matched**: the best Auto candidate. Other Auto or Tentative candidates become review links, never merges.
   - **Possible**: no Auto candidate, but at least one Tentative.
   - **New**: neither.
7. **Without ML.** The decider may be unset, unknown or disabled. Then:
   - the rules decide;
   - a candidate with trigram Jaccard ≥ 0.9 that passes Graphiti's entropy gate (Shannon entropy ≥ 1.5, and ≥ 6 characters or ≥ 2 tokens) is Auto (`rule|minhash`);
   - any other name rule, or Jaccard ≥ 0.5, is Tentative;
   - anything else is None.

   An embedder that is off just drops the vector source. Resolution never errors because a model is off.

**No transitive merges.** Each decision is between one mention and one canonical entity profile. A ≈ B and B ≈ C produce two decisions and never a merge of A with C. When a mention is Auto with two different entities, it joins the better one and links the other for review; the two entities stay apart (test `never_merges_transitively`). Tentative links are never followed.

## Commit and merges

With `commit` (`ResolveOptions { commit: true }`, `{commit: true}`), mentions are processed in order, so later mentions see earlier ones:

- Each mention becomes a row: its `id`, or a generated `ent-<uuid>`. Its embedding is written when the embedder is on.
- **Matched**: a merge row `from = mention id, into = canonical`.
- **Possible**: one review link per Auto or Tentative candidate.

```
kleos_entity_merge {relation: String, from: Any, vt: Validity => into: Any, p: Float, decided_by: String, reason: String}
kleos_entity_link  {relation: String, a: Any, b: Any => p: Float, band: String, decided_by: String, status: String, at: Int}
```

- **Merges never touch the entity rows.** A merge is an assertion in a valid-time relation. `split` appends a retraction at a later valid time. The canonical entity of an id follows current merges (`@ 'NOW'`), chains included.
- **Split restores the previous state.** `split(m1)` makes `m1` its own entity again, together with whatever had been merged into `m1`.
- **History stays queryable:**
  - `::entity history`;
  - `*kleos_entity_merge{… @ T}` (time travel: what was merged at T);
  - `::history`.
- `decided_by` records:
  - `model|same_entity@v2|zero_shot|calibrated|probe` for a model decision;
  - `rule|exact`, `rule|alias` or `rule|minhash` for a rule;
  - `manual` for a merge done by hand.
- **Review:**
  - `::entity merge rel 'a' 'b'` merges and marks the link accepted. It refuses a cycle.
  - `::entity reject rel 'a' 'b'` marks it rejected. Rejected pairs are never proposed again.

**Queries see the canonical entity** through this rule (also in the tests):

```
merged[f, i] := *kleos_entity_merge{relation: 'person', from: f, into: i @ 'NOW'}
canon[x, c] := *person{id: x}, not merged[x, _], c = x
canon[x, c] := merged[x, m], canon[m, c]
```

## Surfaces

```rust
use kleos::entity::{EntityConfig, Mention, ResolveOptions};
let mut cfg = EntityConfig::new("person");
cfg.type_col = Some("type".into()); cfg.lsh = Some("lsh".into()); cfg.decider = Some("nano".into());
let res = db.resolve_entities(&cfg, &[Mention::new("A. Smith").with_type("person")
    .with_context("A. Smith approved the budget")], ResolveOptions { commit: true })?;
// res[0].outcome: Matched | Possible | New; .entity, .p, .band, .decided_by, .candidates
db.merge_entities("person", &from, &into, "review")?;
db.split_entity("person", &id)?;        // Some(previous target)
db.entity_history("person", Some(&id))?; db.entity_pending("person")?;
db.reject_entity_link("person", &a, &b)?; db.canonical_entity("person", &id)?;
```

The memory API (`memory.rs`) calls `resolve_entities` without commit on the names passed to `remember`, over `kleos_mem_entity` with `cfg.scope = ("ns", namespace)`, and writes the resulting rows, merges and review links in its own transaction (memory-api.md, "Entity resolution"). `EntityConfig.scope` (Rust only) restricts a resolution to the rows whose scope column equals the value: the snapshot reads only those rows, index hits outside them are dropped, and committed rows get the scope column as the first key column.

```
::entity resolve person ['Alice Smith', ['A. Smith', 'person', 'my manager'],
                         parse_json('{"name": "Bob", "type": "person", "context": "...", "id": "m7"}')]
                        {type: 'type', lsh: 'lsh', fts: 'fts', decider: 'nano', commit: true}
::entity merge person 'm1' 'p1'
::entity split person 'm1'
::entity history person            # or ::entity history person 'p1'
::entity pending person
::entity reject person 'm1' 'p2'
```

A mention is one of:
- a name;
- a list `[name, type, context, id]`;
- a JSON object with `name`, `type`, `aliases`, `description`, `context`, `id`.

`resolve` returns `mention, name, outcome, entity, id, p, band, decided_by, reason, candidates`.

Restrictions:
- `::entity` runs as a script of its own, like `::decider fit`, not inside an imperative block.
- Read-only scripts may resolve without commit, list history and list pending links.

Python:

```python
db.resolve_entities("person", ["A. Smith", {"name": "Bob", "type": "person", "context": "...", "id": "m7"}],
                    commit=True, type="type", lsh="lsh", decider="nano")
db.merge_entities("person", "m1", "p1"); db.split_entity("person", "m1")
db.entity_history("person", "m1"); db.entity_pending("person"); db.canonical_entity("person", "m1")
```

There is no fixed rule. Resolution writes and reads several relations, so a system op fits it better than a fixed rule does. The `canon` rule above is the read side.

## Evaluation

`cargo +1.99.0 run --release -p kleos --features infer-metal --example entity_eval -- --memory --dblp .bench-data/entity/dblp-acm`

Setup:
- Models: decider-nano (`nano-tune-v1`, q8, Metal) for `same_entity`, and potion-base-8M for the vector source in the ML variants.
- The pair-decision cache is off, so throughput counts every model call.
- Hardware: M3 with 16 GB, shared with two other agents' builds. Throughput is noisy: the same DBLP variant ran at 2–13 mentions/s across runs. The model costs about 25 ms per forward pass on Metal, two orders per pair.

**What was fitted on what.** Every number below is on held-out data.
- **Memory set:** split by hand into 58 adapt and 69 test mentions with **disjoint entities** (22 and 25).
- **DBLP-ACM:** a seeded 30% hash split of the DBLP records gives 794 adapt and 1822 test mentions.
- Two fits use only the adapt pairs the decider saw:
  - **τ fit**: label threshold by accuracy, and `tau_auto` by the risk rule of decide.md, with α = 0.02 for memory and α = 0.01 for DBLP;
  - **probe**: `fit_decider(…, "same_entity", adapt_pairs, probe = true)`, the `::decider fit` path, with C and τ chosen on out-of-fold predictions.
- Neither the shipped `same_entity` defaults (threshold 0.5, τ 0.9) nor any prompt were changed.

How to read the tables:
- **Candidate recall:** the share of mentions with a true match whose match is among the candidates. **Top-M recall:** the same, within the candidates sent to the decider (M = 5 for memory, 3 for DBLP).
- **P/R/F1:** over all mention pairs of the split (memory), or over the gold DBLP→ACM pairs (DBLP).
- **Auto false merges:** wrong merges / all merges.

### Memory-style mentions

127 hand-written mentions in 47 entities. They include:
- nicknames, initials and honorifics;
- roles in parentheses;
- acronyms;
- typos;
- same-name different people: two Alice Smiths, two Maria Garcias, Bob Jones the neighbour and Bob Jones the accountant, two Kate Wongs, Will the landlord and Will Smith;
- same-name different places: Paris and Paris, Texas; two Springfields; Amazon the company and Amazon the river; Oxford the city and Oxford the university.

The mentions are streamed in order with commit (test split shown).

| Variant | Candidate recall | Top-M recall | P | R | F1 | Auto false merges | Mentions/s |
|---|---|---|---|---|---|---|---|
| baseline: exact normalized name, same type | – | – | 0.611 | 0.145 | 0.234 | **42.9%** (6/14) | ~10⁶ |
| kleos, no ML (rules + LSH + FTS) | 0.932 | 0.932 | 0.611 | 0.145 | 0.234 | 42.9% (6/14) | ~2000 |
| kleos + decider-nano, shipped calibration | 1.000 | 1.000 | **1.000** | 0.474 | 0.643 | **0%** (0/25) | 3 |
| kleos + decider-nano, τ fitted on adapt | 1.000 | 1.000 | **1.000** | **0.632** | **0.774** | **0%** (0/32) | 3 |
| kleos + decider-nano, probe fitted on adapt | 1.000 | 1.000 | 1.000 | 0.382 | 0.552 | 0% (0/22) | 2 |

Notes:
- Same-entity AUROC over the pairs the decider saw on test: 0.969 zero-shot, and **0.997 with the probe**. The probe's risk-controlled τ is very high (0.9997, from 224 adapt pairs), so its auto band is small.
- The τ fit on adapt gave threshold 0.064 and τ 0.503, with 0 false on adapt.
- The no-ML variant is as precise as the baseline here. Its exact-name auto-merges are the same-name different people. Initials, nicknames and acronyms are only linked for review.
- The non-merged true pairs mostly sit in the Tentative band as review links, for example "Mom" / "Linda Green" and "Big Blue" / "IBM". There is no name overlap there, and the vector source did not bridge them.

### DBLP-ACM (Leipzig, CC BY 4.0)

ACM's 2294 records are the entities: title as the name, "authors; venue year" as the description. The 1822 test DBLP records are resolved in a dry run.

| Variant | Candidate recall | Top-3 recall | P | R | F1 | Auto false merges | Mentions/s |
|---|---|---|---|---|---|---|---|
| baseline: exact normalized title, unique | – | – | 0.996 | 0.898 | 0.944 | 0.36% (5/1390) | ~10⁵ |
| kleos, no ML (rules + LSH + FTS) | **1.000** | 0.994 | 0.967 | **0.936** | **0.951** | 3.28% (49/1493) | 139–1366 |
| kleos + decider-nano, shipped calibration | 1.000 | 0.999 | 0.994 | 0.892 | 0.940 | 0.65% (9/1386) | 2–12 |
| kleos + decider-nano, τ fitted on adapt (α 1%) | 1.000 | 0.999 | 0.995 | 0.879 | 0.933 | 0.51% (7/1363) | 3 |
| kleos + decider-nano, probe fitted on adapt | 1.000 | 0.999 | **0.996** | 0.895 | 0.943 | **0.36%** (5/1386) | 13 |

Notes:
- Most DBLP-ACM matches have identical titles, so long exact titles are decided by the rule tier and the model sees only the residual pairs: 1434 pairs.
- On those pairs, AUROC is 0.930 zero-shot and **0.994 with the probe**. Both auto bands are conservative, so the model adds precision but not recall over the exact-title baseline.
- The MinHash rule (J ≥ 0.9, no model) reaches the best recall, but at a 3.3% false-merge rate. That is above the α the system targets, which is why it only applies when no decider is configured.
- The remaining false merges are exact-title rule merges: identical titles of different records, as the baseline also makes.

**Summary.**
- Blocking recall is ≥ 0.99 on both sets.
- On memory-style mentions, the decider is what makes merging safe. The baseline merges same-name different people 43% of the time; kleos with decider-nano made 0 false merges out of 32 and raised F1 from 0.23 to 0.77.
- On bibliographic records it keeps precision at the exact-title level, but does not raise recall at a 1% risk target.
- Throughput with a model is bounded by decider-nano's per-pass cost on Metal (about 25 ms × 2 orders × candidates). Rules-only runs at thousands of mentions per second.

## Caveats

- **One scan per call.** The rule tier scans the entity relation once per resolve call: about 2 ms for 2.3k rows. A key-indexed normalized-name relation is the next step for very large entity tables.
- **The nickname table is English-only.** Mentions with no name overlap ("Mom" / "Linda Green", "Big Blue" / "IBM") are reachable only through the vector source, and then only if the embedder relates them.
- **Generated ids are strings.** Pass `id` with each mention for other key types.
