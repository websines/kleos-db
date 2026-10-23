# Engram overlay store

Status: first milestone, implemented on `wave2/engram` (2026-10-23). Research background: [`docs/research/engram.md`](../research/engram.md).

Code:
- `kleos-engram/`: Engram addressing (tokenizer compression, N-gram hashing, fingerprint, overlay export format) and the `kleos-engram` CLI;
- `kleos-core/src/engram.rs`: the overlay store, feature `engram` (`::engram` ops and the Rust API);
- `kleos-core/examples/engram_cli.rs`: runs one CozoScript against a SQLite database (the bridge the Python script uses);
- `scripts/engram/engramedit.py`: EngramEdit-style write path and the end-to-end evaluation on the 132M model;
- `scripts/engram/wikitext_rowfreq.py`: row-reuse counts over WikiText-103 for the write path's penalty.

Tests:
- `kleos-engram` unit tests and `kleos-engram/tests/parity.rs` (bit-exact vs DeepSeek-V4.1-Flash `engram.py`, `engram_demo_v1.py` and the nanogpt `NgramHash`; skipped when the refs are absent);
- `kleos-core/tests/engram.rs`: the overlay store, synthetic spec and vectors, no model.

## Goal

A model with DeepSeek-Engram-style N-gram memory reads, at every token, a few rows of large hashed tables. Row ids depend only on the token ids, so a per-user or per-agent edit to the model's memory is a small set of `(layer, row) → vector` replacements. kleos stores those edits as versioned data:
- supersession and time travel come from valid time (`Validity`, the E1 machinery);
- each row value names the facts it depends on (provenance), so a single fact can be erased by replaying only the rows it touched;
- a namespace (user, agent, org) can be erased exactly;
- every row is keyed by the model's addressing fingerprint, and a mismatch is refused.

kleos never runs the model. A write path (the Python script here, later a kleos-infer backend) computes row values; an engine (the test harness here, later a llama.cpp/vLLM gather hook) reads an exported overlay.

## Addressing: `kleos-engram`

Promoted from the spike's `engram-index`. Builds on stable Rust 1.90; dependencies are `tokenizers 0.22` (same pin as kleos-infer, optional behind the default `tokenizer` feature), `serde_json`, `sha2`, `half`. No candle.

```rust
use kleos_engram::{EngramSpec, SpecOverrides};
let spec = EngramSpec::from_files(config_json, tokenizer_json, &SpecOverrides { pad_id: Some(50256), ..Default::default() })?;
spec.fingerprint();                    // "eg1:7e4a4c295509f462dcb69379654e2830" for engram-nanogpt seed1337
let (ids, hits) = spec.rows_for_text("The Hollenby Viaduct", false)?;  // RowHit {pos, layer, order, head, row}
```

**Config layouts recognised**:
- HF DeepSeek-V4.1 (`text_config.engram_layer_ids`, `engram_vocab_size`, `engram_max_ngram_size`, `engram_n_heads`, `engram_head_dim`, `engram_pad_token_id`);
- the littlemex nanogpt checkpoint (`model.engram_*`, `engram_rows_per_head`; it has no pad id, so pass 50256);
- a top-level `engram_*` config;
- the parity dumps of `parity_ref.py`.

When the config states `engram_num_embeddings` or `engram_compressed_vocab_size`, they are checked against the computed primes and token map. A mismatch is an error: it means the spec has drifted from the weights.

**Fingerprint** (`eg1:` + 16 bytes of SHA-256, tagged `kleos-engram-fp/v1`) covers:
- the semantics (V4.1 or demo);
- the full raw → compressed token map and the compressed vocab size;
- the compressed pad id;
- the max N-gram order, heads per order and row dimension;
- per layer: its id, multipliers, primes and offsets.

The seed and table sizes enter through the multipliers and primes. The weights do not enter: the fingerprint identifies the addressing, not the checkpoint. See [Risks](#risks).

GGUF metadata (`engram.token_map`, multipliers, primes) is not read yet. `EngramSpec::from_config_and_token_map` accepts a precomputed token map, which is what a GGUF loader would hand over.

**CLI** (`cargo build --release -p kleos-engram`):
```sh
kleos-engram spec   <config.json> <tokenizer.json> [--pad-id N]            # fingerprint + summary as JSON
kleos-engram rows   <config.json> <tokenizer.json> [--pad-id N] --text T [--last]
kleos-engram parity <tokenizer.json> <ref-dir>...
kleos-engram bench  <tokenizer.json> <n-tokens>
```

## Overlay store (kleos-core, feature `engram`)

Default builds have no `engram` feature and no new dependencies. Without it, `::engram` parses but errors with `eval::engram_feature_missing`.

### Relations

Created on first use, like `kleos_decider_fits`:

```
kleos_engram_models {name: String => fp: String, row_dim: Int, layers: [Int], table_rows: [Int], spec: Json,
                     config_path: String?, tokenizer_path: String?, pad_id: Int?, created_at: Float}
kleos_engram_facts  {fp: String, ns: String, fact: String, vt: Validity =>
                     text: String, target: String, phrasings: [String], rows: [Int],
                     decided_by: String, source: String, payload: Bytes?}
kleos_engram_rows   {fp: String, ns: String, layer: Int, row: Int, vt: Validity => vec: Bytes, facts: [String]}
```

Their fields:
- `vec` is the replacement row as little-endian bf16, `row_dim` values.
- `facts` on a row is its provenance: the facts whose phrasings read that row and whose targets went into its value.
- `rows` on a fact is the flattened `(layer, row)` pairs it touched.
- `payload` is method state for replay, stored as f32. For EngramEdit-style edits it is the fact's target delta.
- `phrasings` holds the exact prompts the edit was trained on, so replay can re-hash them.

A new value for the same `(fp, ns, layer, row)` at a later valid time supersedes the old one through the vt axis. No `::supersede` declaration is needed: the key has no extra value columns. A retraction (`[t, false]`) reverts the row to the base table.

### CozoScript

```
::engram models
::engram register nano {config: '/path/config.json', tokenizer: '/path/tokenizer.json', pad_id: 50256}
::engram register nano {fingerprint: 'eg1:…', row_dim: 64, layers: [1]}       # computed elsewhere; no text hashing
::engram write nano {ns: 'alice', fingerprint: $fp, at: $t, facts: $facts, rows: $rows, retract: ['old_fact']}
::engram supersede nano {ns: 'alice', fingerprint: $fp, fact: 'old_fact', facts: [$new_fact], rows: $rows}
::engram materialize nano {ns: ['org', 'alice', 'alice/agent-7'], as_of: $t}   # later namespaces override earlier
::engram export nano {ns: 'alice', dir: '/tmp/ov', as_of: $t, fingerprint: $fp}
::engram erase {ns: 'alice'}                       # every model; or ::engram erase nano {ns: 'alice'}
::engram erase_plan nano {ns: 'alice', fact: 'f7'} # rows f7 shares with other current facts
::engram erase_fact nano {ns: 'alice', fact: 'f7', rows: $replayed_rows}
::engram facts nano {ns: 'alice', as_of: $t}
::engram history nano {ns: 'alice', fact: 'f7'}
::engram rows nano {text: 'The Hollenby Viaduct', last: true}
```

`$facts` is a list of objects:
```json
{"id": "f7", "text": "…", "target": " Norway", "phrasings": ["…"], "rows": [[1, 47774], …],
 "decided_by": "…", "source": "episode:42", "payload": [0.1, …]}
```

`$rows` is a list of objects:
```json
{"layer": 1, "row": 47774, "vec": [64 floats], "facts": ["f7", "f9"]}
```

Times are µs since the epoch; the default is now. Writes refuse to run in read-only mode or inside an imperative block. Each op commits its puts and removals in one script, so they are one transaction.

### Rust API

```rust
db.engram_register("nano", &spec, Some((config, tokenizer)))?;          // -> fingerprint
db.engram_write("nano", &EngramWrite { ns, fingerprint, at, facts, rows, retract })?;
db.engram_supersede("nano", "old_fact", &write)?;
let ov: Overlay = db.engram_materialize("nano", &["org".into(), "alice".into()], as_of, Some(fp))?;
db.engram_export("nano", &namespaces, as_of, dir, Some(fp))?;            // -> [(layer, path, rows)]
db.engram_erase_namespace("alice", Some("nano"))?;                       // -> (fact versions, row versions)
db.engram_erase_plan("nano", "alice", "f7")?;                            // -> [((layer, row), other facts)]
db.engram_erase_fact("nano", "alice", "f7", &replayed_rows, None)?;      // -> [((layer, row), action)]
db.engram_facts("nano", "alice", as_of)?;  db.engram_history("nano", "alice", Some("f7"))?;
db.engram_rows_for_text("nano", text, last_only)?;
```

The later "remember also writes an Engram edit" wiring needs exactly these calls:
1. `engram_rows_for_text` to address the phrasings;
2. the write path computes `RowEdit`s;
3. `engram_write`, or `engram_supersede` when the memory layer supersedes the fact.

The type re-exports are `kleos::engram::{EngramFact, RowEdit, EngramWrite, Overlay, OverlayRow, EngramModel}`, plus `kleos::engram::addressing`, which is `kleos_engram`.

### Write semantics

`engram_write` / `::engram write`:
1. **Refusal checks.**
   - The fingerprint must equal the model's.
   - Every row must:
     - be in an Engram layer and inside that layer's table;
     - have exactly `row_dim` finite values;
     - appear once per write;
     - name at least one fact.
   - Every fact named by a row must be in this write or current in the namespace, and not retracted by it.
2. Facts are asserted at `at`. A known id gets a new version and reports `updated`. Facts in `retract` get a retraction at `at`.
3. **Stale rows.** A current row that depends on a retracted or re-asserted fact, and is not rewritten in this batch, is retracted at `at`.
   - If no other live fact depends on it, it reports `reverted`.
   - If one does, it reports `needs_replay`: the write path should have rewritten it.

`supersede` is `write` with the old fact added to `retract`. `materialize ... as_of` an earlier time still returns the old rows.

### Erase

**`erase` (namespace).** Physically deletes every version of every fact record and row of the namespace, for one model or all. History is gone too: `as_of` the past returns nothing.

**`erase_fact` (provenance replay).** Neither paper solves single-fact erase:
- User-as-Engram drops whole users.
- EngramEdit accumulates updates and has no delete.

Joint solves share rows between facts, so dropping one fact's rows is either incomplete (shared rows still carry it) or destructive (shared rows lose the other facts). The protocol:
1. `erase_plan` lists the rows the fact touched that other current facts also touch, with those facts. Rows only it touched need nothing; they revert to base.
2. The write path recomputes exactly those rows from the remaining facts' stored inputs:
   - `phrasings`, re-hashed with the model's own hash;
   - `payload`, the target delta.

   It does not use the erased fact.
3. `erase_fact` runs one transaction:
   - deletes every version of the fact record;
   - deletes every row version whose `facts` names it, including past versions;
   - writes the replacements.

   kleos refuses replacements outside the touched set, and replacements that depend on the erased fact or on facts that are not current. Rows the fact never touched therefore cannot change.

   Each touched row reports one of:
   - `replayed`;
   - `previous`: an older value without the fact is current again;
   - `reverted`: base row.

### Export format

`::engram export` writes one file per Engram layer, `layer{L}.safetensors`:

| Tensor | dtype | shape |
|---|---|---|
| `row_ids` | I64 | `[n]`, ascending, unique |
| `values` | BF16 | `[n, row_dim]`, replacement rows |

Metadata:
- `format = "kleos-engram-overlay/v1"`;
- `fingerprint`, `model`, `layer`, `row_dim`;
- `namespaces` (JSON list), `as_of_us`;
- `mode = "replace"`;
- `writer`.

Every model layer gets a file, empty when it has no rows. The files are read with the stock `safetensors` Python package. The engine side swaps `values[i]` in for base row `row_ids[i]` at gather time. In our harness that is a forward hook on the table, the same mechanism as llama.cpp's `llm_graph_host_get_rows` or vLLM's `EngramDiskTable.gather` would use.

**Refusal.**
- `materialize` and `export` take an optional expected fingerprint and refuse a mismatch.
- `write` requires the fingerprint.
- A reader must compare `metadata.fingerprint` with the model it applies to. The harness does, and refuses.

### Layering

`ns: ['org', 'user', 'agent']` materialises the stack most-general-first. The value of a row comes from the last namespace that has it, which is the org → user overlay order of User-as-Engram, extended to agents. Rows are replaced, not summed. A user value trained on top of the org overlay is the composition the write path intended.

## Write path: EngramEdit-style (`scripts/engram/engramedit.py`)

Method from EngramEdit (arXiv 2610.10533; github.com/ModalityDance/EngramEdit, MIT). It is re-implemented for this model; no LongCat code is copied.

1. **Targets across phrasings.** Per fact, one perturbation δ of the Engram read `e_t` is optimised. `e_t` is the concatenation of the 16 rows the layer reads, 16 × 64 = 1,024 values, and δ is applied at the **subject's last token**. It is shared by all the fact's phrasings: 3 phrasings × 3 context prefixes = 9 prompts. The loss is the next-token NLL of the target at each prompt's end, plus EngramEdit's KL on the next-token distribution after "{subject} is a", plus norm decay and a norm clamp. Adam, lr 0.05, at most 40 steps. Run:

```sh
B=.bench-data/engram; T=$CARGO_TARGET_DIR/release
cargo build --release -p kleos-engram && cargo build --release -p kleos --features engram --example engram_cli
python -I scripts/engram/wikitext_rowfreq.py --lab $B/src/zenmai-lab/experiments/2026-10-21-engram-nanogpt \
  --ckpt $B/models/engram-nanogpt/engram-seed1337/model.safetensors --wikitext <hf snapshot>/wikitext-103-raw-v1 --out $B/rowfreq
python -I scripts/engram/engramedit.py --lab ... --ckpt ... --rowfreq $B/rowfreq --cli $T/examples/engram_cli \
  --engram-bin $T/kleos-engram --config $B/models/engram-nanogpt/engram-seed1337/config.json \
  --tokenizer <gpt2 tokenizer.json> --out $B/milestone
```
2. **Map to activated N-grams.** Each prompt reads exactly one row per hash column at the subject position: orders 2–3 × 8 heads. On this model that row *is* the memory slot for that N-gram.
3. **Joint ridge solve.** Solve `(AᵀA + Λ) U = Aᵀ(target − current)`. Every prompt reads one row per column and columns are disjoint, so this is diagonal per row: `U_r = Σ_{p reads r} δ_p / (n_r + λ_r)`. The penalty is `λ_r = λ_ridge + λ_reuse · min(len_w(order) + f(r) − 1, 64)`, with:
   - `len_w` = 8 / 2 / 1 for orders 2 / 3 / 4;
   - `f(r) = min(1 + 9 · pct(r), 10)`, where `pct(r)` is the row's hit-count percentile within its column over WikiText-103 train, the model's own training data (118M tokens, `wikitext_rowfreq.py`).

   The defaults are EngramEdit's: `λ_ridge = 0.01`, `λ_reuse = 0.05`. On a hashed table the reused entity is the row, collisions included, so we count row hits rather than exact N-grams.

   The stored value is `base_row + U_r`, as a replacement in bf16. Each row depends on every fact whose prompts read it. Facts about the same subject therefore share their subject rows, and their deltas are summed into those rows.

**Baseline**: the spike's naive per-trigger training (`overlay_edit.py`):
- the rows read at the last token of one trigger phrasing;
- Adam lr 0.05, 60 steps, cross-entropy on the target's first token, model frozen.

## Results (132M model, CPU)

Setup:
- **Model:** `littlemex/engram-nanogpt-wikitext103` engram-seed1337. It has 132M parameters and one Engram table at layer 1 (orders 2–3, 8 heads, 64-dim rows, 1,049,422 rows). Fingerprint `eg1:7e4a4c295509f462dcb69379654e2830`.
- **Facts:** 20 synthetic facts in WikiText style about 14 fictional subjects. Six subjects carry two facts each (for example a viaduct's country and its material), so those facts share subject rows. Each fact has 3 trained phrasings and 2 **unseen rephrasings whose last two (compressed) tokens differ from every trained phrasing**. The script asserts this rule.
- **Recall** is the top-1 rank of the target's first token.
- **Locality** is NLL on the first 8,192 tokens of WikiText-103 test.
- **Pipeline:** every overlay went through kleos (`::engram write`, then `::engram export`) and was evaluated from the exported safetensors as read back, in bf16.
- **Addressing:** the Rust addressing was checked against the model's own hash on 40 prompts.
- **Raw numbers** are in `.bench-data/engram/milestone/results.json`. The baseline comes from run 1; EngramEdit-style and the kleos checks come from run 2, which is identical except that the supersede check uses a fact recalled before.

| | base model | baseline (spike's naive per-trigger) | EngramEdit-style |
|---|---|---|---|
| top-1, trained phrasing (the trigger) | 1/20 | **20/20** | 17/20 |
| top-1, all trained phrasings | 2/60 | 22/60 | **47/60** |
| **top-1, unseen rephrasings, different endings** | 1/40 | 1/40 (top-5: 6) | **19/40 (top-5: 33)** |
| locality: NLL change on WikiText test | — | +0.046% | **+0.010%** |
| test positions reading an edited row (of 8,192) | — | 20 | 25 |
| rows per fact | — | 16 | **11.2** (224 rows, 96 shared between facts) |
| bytes per fact (bf16 + i64 id) | — | 2,176 | **1,523** |
| write time, 20 facts (wall, shared CPU) | — | 244 s | 121 s (402 s in run 1 under heavier load) |
| kleos write of the batch (`::engram write` via the CLI) | — | 0.44 s | 0.11–0.60 s |

The open question from the spike was recall on rephrasings that do not end in the trigger. Moving the edit from the trigger's last token to the **subject's** last token, with targets shared across phrasings, takes it from 1/40 to 19/40 top-1 (33/40 top-5). The baseline stays at the base model's level: its rows are only read when the prompt ends in the trigger N-grams. Locality also improves (+0.010% vs +0.046% NLL). The cost is the exact trigger, 17/20 vs 20/20, because three facts land at rank 2–3.

Recall per fact:
- For the 8 single-fact subjects, EngramEdit-style recalls 15 of 16 unseen rephrasings top-1.
- For the 6 two-fact subjects, it recalls 4 of 24. Both facts write their deltas into the same subject rows, and the summed perturbation has to serve two relations.

This is the "gradient interference / density" limit User-as-Engram reports, showing up per subject.

**kleos-level checks** (all passed; EngramEdit-style overlay, namespace `user-engramedit`):

| Check | Result |
|---|---|
| Supersede `vos`: piano → guitar at T2 (`::engram supersede`) | " guitar" rank 9 → **1**; " piano" 1 → 4 |
| `as_of` T1 (before the supersession) | " piano" back at **rank 1**; the exported overlay is **bit-identical** to the pre-supersede export |
| Erase one fact (`teague_job`; it shares 16 subject rows with `teague_instrument`) | `erase_plan` → 16 shared rows. They were replayed from kleos's stored phrasings and payloads, **bit-identical to a fresh joint solve** without the fact. The fact went from rank 1 to 7 (base 81); the other facts kept 16/16 top-1; the **208 untouched rows were bit-identical**; its history was empty; the kleos op took 63–196 ms |
| Erase the user (`::engram erase`) | 21 fact versions and 240 row versions deleted; the export is empty now and `as_of` the past; **logits bit-identical** to the base model (`torch.equal`, 40 prompts and a 512-token window) |
| Fingerprint mismatch (the spec with another pad id) | `write` and `export` refused with `eval::engram_fingerprint_mismatch`; the harness also refuses an export whose metadata fingerprint is not the model's |

**Versus plain kleos retrieval.** The same 20 facts were stored as text in a kleos relation with an FTS index (`Simple` tokenizer, stemming, stop words), queried with each unseen rephrasing:
- FTS found the right fact at top-1 for **28/40**.
- With that sentence placed before the rephrasing (in-context, the same 132M LM), the target was top-1 for **13/40** and top-5 for 24/40.
- Parametric memory scored 19/40 top-1 and 33/40 top-5.

**Verdict.** On this small WikiText LM, parametric Engram memory beats retrieval-in-context, but mostly because the LM is bad at copying from context. Retrieval still finds the right fact more often (28/40) than the edit recalls it (19/40), and it has no per-subject interference. With an instruction-capable model, in-context retrieval would be expected to win on verbatim facts.

So Engram overlays **complement** retrieval rather than beat it:
- they need no context window;
- they cost about 1.5 KB per fact;
- they generalise across endings when edited at the subject.

Retrieval remains the source of truth, and the kleos fact records keep that link.

## Risks

- **Interference on shared subjects.** Two facts about one subject sum their deltas into the same rows. Unseen recall drops from 15/16 (one fact per subject) to 4/24 (two facts per subject). A per-relation split (e.g. keying the 3-gram rows on the relation word) or an iterative solve is future work. This is the main quality limit.
- **Erasing a fact that shares rows is exact with respect to kleos's data, not with respect to the model's behaviour.**
  - The replayed rows equal a fresh solve without the fact, so nothing derived from it remains.
  - But the remaining fact's delta in the same rows can still lift the erased target. `teague_job` fell from rank 1 to 7, not back to its base rank of 81.
  - Facts that share no rows revert bit-exactly.
- **Leakage.** Any text that reads an edited row is affected: 25 of 8,192 test positions here, +0.010% NLL. Subject-position edits leak wherever the subject's N-grams or hash collisions occur. The WikiText row-frequency penalty shrinks updates on busy rows but does not remove leakage. Exact-key gating (research option (d)) is not built yet.
- **Fingerprint scope.** The fingerprint covers addressing, not weights. Two checkpoints with the same spec share a fingerprint, but an overlay trained on one is not valid on the other. Register a model under a name per checkpoint. Mixing the weights digest into a second id is cheap follow-up work.
- **Spec drift.** It is guarded: a stated `engram_num_embeddings` or `engram_compressed_vocab_size` that does not match is an error, and the token map is built from the raw `tokenizer.json`. A model without those fields (the nanogpt one) relies on the caller passing the right pad id.
- **The model is a toy.** It is a 132M WikiText LM with its Engram at layer 1. The results are mechanism results, not usefulness results for an assistant.
- **Write cost.** The target computation needs the model and backprop: about 6–20 s per fact on a shared CPU. kleos's side is milliseconds.
- **Atomicity.** Each op reads (current facts and rows) and then commits its writes in one script. Two concurrent writers to the same namespace could interleave between the read and the write. Callers should serialise writes per namespace.

## Not done yet

- A `Db::remember`-side hook (the planned wiring: "remember also writes an Engram edit when a model is attached"). The API above is what it would call. The write path itself (target computation) still lives in Python, because it needs the model's forward and backward passes.
- A kleos-infer port of the write path and of the overlay application (a gather-time override in a Rust Engram forward).
- GGUF metadata loading (`engram.token_map`, multipliers, primes): the precomputed-token-map constructor is there, the GGUF reader is not.
- Exact-key gating of overlay rows (research option (d)): store the compressed N-gram with each row, so colliding N-grams do not read it.
- An interference fix for several facts per subject.
- Python bindings (`kleos-lib-python`). The script uses the `engram_cli` example instead.
- A materialise latency benchmark at 1k facts (the research exit criterion is under 1 ms). Here 224 rows materialise and export in tens of ms through the CLI, including process start.
