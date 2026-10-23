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
   - one side has no context.

   An exact name with context on both sides goes to the model, because "Alice Smith (my dentist)" and "Alice Smith (engineering manager)" can be two people. So does an exact name that is ambiguous (two entities share it).
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

The memory API (`memory.rs`, built in parallel) calls `resolve_entities` with the mentions it extracts, and stores the returned canonical ids.

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

See the results section below (`entity_eval.rs`).

## Caveats

- **One scan per call.** The rule tier scans the entity relation once per resolve call: about 2 ms for 2.3k rows. A key-indexed normalized-name relation is the next step for very large entity tables.
- **The nickname table is English-only.** Mentions with no name overlap ("Mom" / "Linda Green", "Big Blue" / "IBM") are reachable only through the vector source, and then only if the embedder relates them.
- **Generated ids are strings.** Pass `id` with each mention for other key types.
