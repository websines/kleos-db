# DeepSeek Engram and kleos: research spike

Date: 2026-10-23. Branch `research/engram`. Time box about 2 h. Prototype code is in `scripts/engram/`.

> Follow-up: the `engram-index` prototype became the `kleos-engram` crate (its `parity` and `bench` subcommands moved to the `kleos-engram` CLI), and milestone M1 is [`docs/specs/engram-overlay.md`](../specs/engram-overlay.md).

Evidence tags:
- **[ran]**: we executed it on this Mac.
- **[read]**: primary source (paper text, official code, model files) read during this spike.
- **[2nd]**: secondary (guide, blog, community README) or relayed from PR/issue threads without re-reading every line.

## TL;DR

- **The lookup spec is fully pinned down, and we reproduce it bit-exactly in Rust.**
  - What we match: tokenizer compression built from `tokenizer.json`, the multipliers (a port of NumPy's `SeedSequence` → PCG64 → Lemire), the prime table layout, row offsets, and the DEAD-token rule.
  - What we checked against, all with 0 mismatches:
    - the shipped DeepSeek-V4.1-Flash `inference/engram.py`: 1.58M indices, plus the exact per-layer table row counts in its `config.json`;
    - the original `engram_demo_v1.py`: 3.3M indices over 3 configs;
    - a third, community implementation: 68k indices.
  - Hashing costs about 110 ns per token per layer, which is nothing next to the forward pass. [ran]
- **One trap:** `engram_demo_v1.py` decodes through `transformers`. Under transformers 5.x, the V3 tokenizer class swaps out the ByteLevel decoder, so the demo builds a different compressed vocab: 104,498 instead of 98,627. The shipped V4.1 code avoids this by decoding with the raw Rust tokenizer (`backend_tokenizer`), and so does our Rust. [ran]
- **What "Engram compatibility" can concretely mean for kleos:**
  - kleos stores **per-user/per-agent overlays of Engram rows**, as `(layer, row_id) → vector` plus validity and provenance;
  - it materializes the overlay as of a time;
  - the engine reads the overlay in place of base rows.
- **That works on a real public Engram model on this Mac** (132M, Apache-2.0):
  - 10/10 synthetic facts went from top-1 0/10 to 10/10 by training only 160 rows (42 KB f32), with the model frozen, in about 21 s on CPU;
  - **dropping the overlay restored bit-identical logits** (exact erase). [ran]
- **The catches:**
  - Recall is bound to the surface form: it works only when the prompt ends in the trigger's tokens.
  - An overlay leaks into any other text that contains the same n-grams. In our unrelated paragraph that was +3.8% NLL from 2 positions.
  - **No small, instruction-capable Engram model with a late Engram layer is public.** User-as-Engram's checkpoints were not released.
- **Recommendation:** do option (a) as a standalone "Engram overlay store" milestone on the 132M model now. Defer (b) and (c). See [Recommended plan](#recommended-plan).

## 1. The lookup spec (bit-exact)

The production reference is DeepSeek-V4.1-Flash `inference/engram.py` together with `config.json` `text_config` (repo `deepseek-ai/DeepSeek-V4.1-Flash`, MIT). We read it and ran it on CPU. [read][ran]

1. **Tokenizer compression.**
   - For every token id in `0..len(tokenizer)`, decode that single id with the raw `tokenizers` backend (`skip_special_tokens=False`, no cleanup).
   - If the text contains U+FFFD (a partial UTF-8 byte), the key is `id_to_token(id)`.
   - Otherwise the key is `normalize(text)`, or the raw `text` if that comes out empty. The normalizer is `NFKC → NFD → StripAccents → Lowercase → Replace(/[ \t\r\n]+/, " ") → Replace(/^ $/, U+E000) → Strip → Replace(U+E000, " ")`.
   - New ids are assigned in first-seen order of the keys.
   - Results:
     - V4.1 tokenizer: 129,280 → **99,092** (the code asserts this);
     - V3 tokenizer.json: 128,815 → 98,627;
     - GPT-2: 50,257 → 32,962.
   - The paper says compression cuts the vocab by about 23% (§2.2). [read][ran]
2. **Multipliers** (one per look-back k in `0..max_ngram`, per layer):
   - `rng = np.random.default_rng(seed + 10007*layer_id)`. V4.1 has no seed, i.e. seed = 0.
   - `bound = (int64_max // compressed_vocab) // 2`.
   - `mult = rng.integers(0, bound, max_ngram, dtype=int64)*2 + 1`.
   - These are odd, and small enough that `id*mult` cannot overflow int64.
   - V4.1 values:
     - layer 1: `[76632096046245, 4839876093313, 35959672319349, 73987337458391]`
     - layer 14: `[67716810739261, 51510806800915, 30921347202721, 82619226485591]`
   - Our Rust reproduces them from the seed alone (`numpy_rng.rs`). [ran]
3. **Hash.**
   - `mix_n = XOR_{k<n} (tok[t-k] * mult[k])` for n = 2..max_ngram, computed as a rolling value.
   - Then for each head j: `idx = mix_n mod prime[n][j]` (floor-mod, int64).
   - In V4.1, `+ offset[n][j]` into the layer's single concatenated table, with columns laid out `[order][head]`.
   - Positions before the sequence start use the compressed pad id (raw id 2 in V4.1).
   - **V4.1-only rule:** a DEAD token (an image span, marked −1) blocks look-back. Once a shift is blocked, every longer shift is blocked too, and blocked slots read the pad id.
   - The demo instead pads only at the sequence start and hashes negative ids as values. [read][ran]
4. **Primes.**
   - For each layer, order and head in turn: the next prime above the previous one, starting from `engram_vocab_size - 1`.
   - One `seen` set is shared across **all layers**, so the ranges never repeat.
   - V4.1 has `engram_vocab_size = 16,000,000` for every order, giving 48 primes from 16,000,057 to 16,000,889.
   - Per-layer sums are 384,006,168 and 384,016,682, exactly `config.engram_num_embeddings`; our Rust prints this. [ran]
5. **V4.1-Flash shape.**

   | Field | Value |
   |---|---|
   | Engram layers | `engram_layer_ids = [1, 14]` |
   | N-gram orders | 2–4 (`max_ngram 4`) |
   | Heads | 8 per order, so 24 rows per token per layer |
   | `head_dim` | 256 |
   | `hidden_size` | 5120 |
   | `hc_mult` | 4 |
   | Table dtype | **F8_E4M3** rows `[384M, 256]` plus F8_E8M0 block-32 scales `[384M, 8]`, i.e. **264 B/row** |
   | Fusion weight | `wkv` [25600, 6144]: 4 hc keys + 1 value |
   | Other weights | `q_weight`/`k_weight` [4, 5120] BF16 |
   | ShortConv | none (the demo has one) |
   | Shards | safetensors shard `model-00047` = layer 1 table only (101.5 GB), `model-00048` = layer 14 |

   - Total is about 196.6B Engram parameters.
   - Primes and multipliers are **computed at load, not stored**.
   - [read; tensor headers via range requests]
6. **Gate (for reference).** `gate = sigmoid(signed_sqrt(<RMSNorm(h), q·k·RMSNorm(key)>/√d))`, then `h + gate·value`. This matches paper Eq. 3–4. [read]

Paper configs (arXiv 2601.07372, Table 5) [read]:
- Engram-27B: layers [2, 15], N ∈ {2, 3}, 8 heads, d_mem 1280, "vocab size" 2,262,400.
- Engram-40B: 7,239,680.
- §6.4: a 100B table offloaded to host DRAM costs at most 2.8% throughput on an 8B dense backbone (H800).

## 2. How engines access the tables

Mostly relayed by a research subagent from vLLM main @8fae0ff, llama.cpp master @f5027fb and the open PRs [2nd, from primary code]. Scratch copies are in `.bench-data/engram/engines/`.

- **Per-token cost:** 24 rows × 264 B = **6,336 B per token per Engram layer** (12.7 KB/token for V4.1). Row ids depend only on token ids, so every engine hashes before layer 0 and prefetches. CXL paper (2603.10087): about 5 KB/token/layer, a prefetch window of about 56 µs at layer 2, bandwidth not the bottleneck but latency is. [read via subagent]
- **vLLM:**
  - `--engram-config` (`vllm/config/engram.py`) with `cpu_offload=True` by default: pinned host memory, read via UVA by a Triton gather, prefetched on a side stream.
  - Tables are sharded **by hash head**.
  - Open PR #62238 `disk_offload` (`EngramDiskTable`) mmaps rows straight out of the safetensors shard.
  - Draft PR #57023 serves rows from Mooncake (`VLLM_ENGRAM_MOONCAKE_CONFIG`).
- **llama.cpp:**
  - PR #30817 (open) adds V4.1. The whole table is one tensor, `per_layer_token_embd.weight` (MXFP8 or K-quants: 264/210/176/144 B per row).
  - Read lazily from an mmap (`--lazy-mode`, `MADV_RANDOM`), CPU only.
  - Per ubatch, `llm_graph_input_dsv41::set_input` hashes, prefetches rows, gathers them on the host (`llm_graph_host_get_rows`, dequantizing to F32) and uploads.
  - Its GGUF stores multipliers, primes, offsets and `engram.token_map` as metadata.
  - Unsloth ships the table as a separate GGUF shard with a `set_split.py` to swap it.
- **SGLang:** PR #44521 (open) proposes an `EngramStore` backend abstraction (`native|mooncake`, `.lookup_into(layer, row_ids, out)`). It explicitly declares the tables "immutable"; live updates are out of scope.
- **Per-request overlays or adapters:** **none found** in vLLM, llama.cpp or SGLang. Every lookup takes plain row ids, with no tenant or request id.

Integration seams for an external store, from least to most invasive:
1. Write a replacement table file (same tensor name and shape). Only a whole-table swap; useless per user.
2. Patch llama.cpp `llm_graph_host_get_rows`: one call site; on the host, single-tenant on a Mac.
3. Wrap vLLM `EngramDiskTable.gather` or the Mooncake/SGLang `lookup_into` contract; each needs a request→tenant id plumbed in.

## 3. User-as-Engram (arXiv 2606.19172, Pine AI, June 2026)

[read]: paper text and the official repo `19PINE-AI/user-as-engram` (Apache-2.0 code and data; paper CC BY 4.0).

- **Edit.**
  - The rows touched are those the trigger's last token reads. With N = 2..3 and 8 heads that is about 16 rows, **in the last Engram layer only** (`joint_opt.py`: `last_layer = max(engram_layer_ids)`). Writing the early layer drops recall from 1.00 to about 0.25–0.31.
  - Methods:
    - closed-form `UNEMBED_P`: `pinv(W_V) · U_y`, under 1 ms, weak (6% top-1 at 100 facts);
    - `OPT`: about 15 Adam steps per fact;
    - **Joint OPT**, the default for ≥30 facts: all of a user's rows trained together, model frozen, cross-entropy on the first gold token, Adam lr 0.5, 1,000–8,000 steps, about 0.2–0.4 s per fact on a GPU.
  - Edits **replace** rows; they are not additive deltas.
- **Storage.**
  - An `OverrideMap{rows: i64[n], values: f32[n, 32]}` per user.
  - About **0.9 KB per fact, about 88 KB per user at 100 facts**, versus 14.2 MB for a rank-64 LoRA (161× at 100 facts).
  - **The abstract's "33,000×" is not storage.** It is the ratio of the extra loss each method causes on unrelated text (LoRA Δbpb +1.70 vs Engram +0.00005).
- **Serving.** Save the base rows, write the org then the user overrides, run the forward pass, restore (`engram_server.py`, about 50 lines). Single request at a time; batched multi-tenant serving is not benchmarked.
- **Deletion and unlearning.**
  - Not discussed in the paper.
  - Dropping a user's map is trivially exact, because base rows are restored after every request.
  - Removing **one fact** after Joint OPT is not clean, because rows are shared and co-optimized. You would re-run Joint OPT on the remaining facts.
- **Collisions and interference.**
  - "Disjoint slots" just means different triggers hash to different rows. Nothing is reserved.
  - A per-user salt was tested and rejected (top-1 0.11–0.46), because salted rows land on untrained rows.
  - The same trigger text from two facts means the last write wins.
  - The density ceiling comes from gradient interference (Joint OPT top-1: 68% at 100 facts, 35% at 1,000), not from hash collisions.
  - Recall is surface-form bound: it transfers to a paraphrase only if the paraphrase ends with the trigger tokens. It cannot chain facts (13%).
- **Claims.**
  - **5.6×** is the 3-seed mean of indirect reasoning for the *layered* design (per-user Engram plus a shared rank-16 "reasoning" LoRA, 41.2%) vs per-user LoRA (7.3%). Bare Engram reaches 23%.
  - **"Beats retrieval past ~100 facts"** holds for the layered design on indirect questions. RAG wins at N = 34 and on verbatim questions, and Engram loses LOCOMO open-domain.
- **Models.** Mini-Engram is DeepSeek's Engram grafted into nanochat, trained at 12 tokens per parameter:

  | Model | Total params |
  |---|---|
  | d8 | 178M |
  | d12 @768 | 339M |
  | d12 @1280 | 625M |
  | d20 @1536 | 1.22B |

  All four share a 51.2M "large" table (orders 2–3, 8 heads, row dim 32).
- **Checkpoints are not released.** The paper says "we release four checkpoints", but the README says they are "too large to ship — regenerate them by training". Nothing is on HF. Training d12 takes about 5 GPU-hours on a Blackwell card (per the README).

Related papers [read via subagent]:
- **Memory Grafting (2605.20948):** builds frozen **exact-key** n-gram value tables offline from a big model's hidden states, and falls back to hashed Engram on a miss. A precedent for collision-free, swappable stored tables. No official code.
- **Engram-Nine (2601.16531):** a static collision-free hot tier for the top 100K 2/3-grams. **No quality gain**, and 11–12% slower. Collisions act as a regularizer.
- **CXL pooling (2603.10087):** access-cost numbers as quoted above. No Zipf or caching statistics.

### Small Engram models (≤ ~2 GB) [read HF metadata; 2nd for model quality]

| Repo | What | Licence | Usable here? |
|---|---|---|---|
| **littlemex/engram-nanogpt-wikitext103** | 12-layer GPT-2-style, d512, **132M** params (67.2M is one Engram table, at layer 1), V4.1-style hash, WikiText-103 only, about 528 MB per seed; loader in `github.com/littlemex/zenmai-lab` | Apache-2.0 | **Yes: loads and runs on CPU; used below.** Not instruction-tuned; Engram layer is early. |
| DefEki/tinyllama-engram-test | 54 MB Engram adapter for TinyLlama-1.1B | none | test artifact, no docs |
| Raskoll/qwen3-0.6b-engram | about 5 GB (retrofit, WikiText) | Apache-2.0 | too big, unverified hash |
| Lyte/TinyQwen3-Engram-HC-Darija | 140M, Darija only | Apache-2.0 | language mismatch |

No public Mini-Engram (User-as-Engram) weights exist. V4.1-Flash is far beyond 16 GB.

## 4. Where kleos can add value

| Option | Verdict | Feasibility on this Mac | Effort | Main risks |
|---|---|---|---|---|
| **(a) Per-user/agent Engram overlay store** | **Worth a milestone** | High: shown end-to-end on the 132M model | M1 about 2–3 days | No good small base model; recall is surface-form bound; leakage via shared n-grams; per-fact erase after joint training |
| (b) Big-table storage and serving | **Not worth it** | Low | Large | Engines already mmap / pin / offload / Mooncake; a DB is slower than a flat mmap for fixed 264 B rows at dense ids; a static hot tier showed no gain (Engram-Nine) |
| (c) Engram inside decider-nano / embedders | **Not now** | Retraining needed | Large | Gains at this scale are within noise (nanogpt: val ppl 28.97 vs 29.26, test 29.40 vs 29.36); decider-nano is a classifier fine-tune, not a knowledge-heavy LM |
| (d) Exact-key overlays (kleos addition to (a)) | Fold into (a) | High | Small | Requires kleos-side gather (we compute the hash ourselves anyway) |

**(a), concretely.**
- An overlay row in kleos is `(model_fingerprint, layer, row_id) → vector`, plus `(owner, fact_id, valid_from, valid_to, provenance)`.
- `model_fingerprint` covers the token map, multipliers, primes and table hash. A row id means nothing without it.
- What kleos would give:
  - **Exact erase:** retract a user's or fact's rows. Dropping a whole overlay is bit-exact; we measured this. Per-fact erase should be exact for independent edits on disjoint rows, but that is not yet measured (M1 step 3).
  - **Time travel:** materialize the overlay as of T, i.e. "what did the agent's model know last Tuesday".
  - **Supersession:** a new value for the same trigger supersedes the old rows.
  - **Namespaces:** user, org and agent overlays stacked in the same org→user order the paper uses.
  - **Audit:** which fact wrote which rows.
- The engine side is a row-id → vector override at gather time. On a Mac with a single-tenant local agent, that is a llama.cpp `llm_graph_host_get_rows` patch or a PyTorch hook, as we did.
- **(d):** because kleos computes the hash itself (we have a bit-exact Rust version), it can key overlays by the *exact* compressed n-gram and head, and apply a row only when the n-gram matches. Unrelated n-grams that merely collide into an edited row in one head then stop reading the user's value. Genuinely shared n-grams still leak, and that leak is inherent to the method.
- **Honest limits:**
  - This is a *write-time-optimized, trigger-phrase-keyed* memory. It complements kleos's retrieval memory; it does not replace it.
  - Its value depends on a base model whose Engram layer sits late enough to carry facts. The paper's d12/d20 models do; the public 132M model's layer-1 Engram still learned our 10 facts, but only through training.

**(b)** is a no. The engines' existing paths (flat mmap with `MADV_RANDOM`, pinned host memory, a Mooncake store) are what a 200 GB fixed-stride table wants. kleos's bitemporal and relational machinery adds nothing to serving immutable rows. QingGo/EngramDB (Apache-2.0, Rust) already prototypes a dedicated on-disk store.

**(c)** is a no for now. There is no evidence that Engram helps a 150M classifier, it would need pretraining-scale data, and the gains in the published small studies are at noise level. Revisit only if (a) needs a kleos-owned small LM.

## 5. What we ran

### Parity: Rust `engram-index` vs references [ran]

`scripts/engram/engram-index` is a standalone crate, not a workspace member, with the same `tokenizers 0.22` pin as kleos-infer. `scripts/engram/parity_ref.py` dumps the references by importing the upstream code unmodified.

| Reference | Compression built from tokenizer.json | Multipliers + primes (+ offsets) | Indices compared | Result |
|---|---|---|---|---|
| V4.1-Flash `inference/engram.py` (`NgramHashState` on CPU, V4.1 tokenizer, 8 texts + 64×512 random + DEAD-mask case) | 129,280 → 99,092, **0 diffs** | 0 mismatches; row counts = config `engram_num_embeddings` | 1,582,992 | **PASS** |
| `engram_demo_v1.py`, demo config (layers [1, 15], N ≤ 3, 8 heads) | **DIFF** (98,627 vs 104,498; see note) | 0 | 1,052,928 | **PASS** (hashing) |
| demo, paper-27B sizes (2,262,400; layers [2, 15]) | DIFF (same cause) | 0 | 1,052,928 | **PASS** |
| demo, N ≤ 4, 4 heads, layers [0, 7, 30], seed 12345, uneven sizes | DIFF (same cause) | 0 | 1,184,544 | **PASS** |
| littlemex nanogpt `NgramHash` (GPT-2 tokenizer) | 50,257 → 32,962, **0 diffs** | 0 | 68,016 | **PASS** |

**Note on the demo's compression.**
- Under transformers 5.19, `AutoTokenizer` turns DeepSeek-V3's `LlamaTokenizerFast` into a tokenizer whose decoder is no longer `tokenizer.json`'s ByteLevel. `decode([300])` gives `'âĢ'` instead of a partial byte, which yields 104,498 ids.
- Decoding with the raw `tokenizers.Tokenizer.from_file` gives 98,627, byte-identical to our Rust.
- The shipped V4.1 code decodes via `backend_tokenizer` and uses a `PreTrainedTokenizerFast` class, so it is unaffected.
- Lesson: **always build the token map from the raw `tokenizer.json`, or load it from GGUF `engram.token_map`. Never build it through `transformers.decode`.**

Hashing throughput, Rust release build, unoptimized: about **113 ns per token per layer** (16 indices, including compression), so 1M tokens × 2 layers takes 0.23 s. Building the compression table takes about 0.4 s.

### Overlay edit and exact erase on a real Engram model [ran]

`scripts/engram/overlay_edit.py` on `littlemex/engram-nanogpt-wikitext103` engram-seed1337 (CPU, 60 Adam steps, lr 0.05). All 10 facts were trained jointly with a mean loss, but on disjoint rows. The base is frozen; only overlay rows are trained, through a forward hook that swaps in overlay vectors for matching row ids.

| Measure | Result |
|---|---|
| Facts (synthetic per-user, trigger → first target token) | 10 |
| Rows per fact (last trigger token, 2 orders × 8 heads) | 16; 160 unique, so no collisions between facts |
| Overlay size | 160 × 64 dims: **42 KB f32 / 22 KB bf16** (about 4.2 / 2.2 KB per fact, ids included) |
| Top-1 before → after | **0/10 → 10/10** (ranks before: 2 to 21,728; p after 0.96–1.00) |
| Training time | 20.8 s for all 10 facts on CPU |
| Paraphrase ending with the same tokens | 2/2 recalled (p 0.99, 1.00) |
| Paraphrase with a different ending | 0/2 (ranks 1,637 and 12) |
| Unrelated 95-token paragraph | 2 positions read overlay rows (it contains "favourite drink is", "the name of"); NLL 4.608 → 4.782 (+3.8%); max logit change 14.9 |
| **Erase (drop overlay)** | **logits bit-identical to base** (`torch.equal`) |

This reproduces User-as-Engram's qualitative claims on independent hardware and an independent model:
- local writes;
- tiny storage;
- exact erase;
- surface-form-bound recall;
- leakage exactly at shared n-grams.

The model is a WikiText LM, not an assistant, so these are mechanism checks, not usefulness checks.

## Recommended plan

**M1: "Engram overlay store" spike, about 2–3 days, all on this Mac, no kleos-core/kleos-infer changes until it proves out.**
1. Promote `engram-index` into a small crate with these pieces:
   - `EngramSpec`: token map, multipliers, primes and offsets, loadable from HF `config.json` + `tokenizer.json` or from GGUF metadata;
   - a model fingerprint hash;
   - `rows_for(ids)`;
   - exact-key n-gram ids, for option (d).
2. Add a kleos schema for overlays:
   - `engram_overlay{model_fp, layer, row => vec(F16/BF16), owner, fact_id, valid_from, valid_to, provenance}`;
   - an `as_of(T)` materialize query that returns `(row_ids, values)` for an owner stack (org → user → agent).
3. Bridge to PyTorch (Python bindings) on the 132M model and test:
   - write 10–100 facts;
   - supersede one fact (new value, same trigger), so recall flips;
   - time-travel to before the supersession, so the old value comes back;
   - erase a user, so logits are bit-identical to base;
   - erase one fact under independent per-fact edits, so the other facts are unchanged and that trigger reverts to base.
   - Also measure leakage with row-id keying vs exact-key n-gram keying.
4. Exit criteria:
   - all of the above pass bit-exactly;
   - overlay materialize under 1 ms for 1k facts;
   - a written verdict on whether surface-form-bound recall is useful for agent memory at all, compared with kleos retrieval on the same facts.

**M2, only if M1 is convincing:**
- Get a base model with a late Engram layer: train Mini-Engram d12 with the Apache-2.0 user-as-engram code, about 5 GPU-hours rented, never on this Mac.
- Re-run the M1 suite with Joint OPT at 100–300 facts per user.

**M3 (later, needs big hardware):**
- A llama.cpp `llm_graph_host_get_rows` overlay patch for V4.1-Flash, where kleos supplies `(row_id → vector)` per session.
- Upstream interest is plausible: no engine has per-request Engram overlays today.

## Risks

- **Base model gap.** No small, public, instruction-capable Engram model with a late Engram layer exists. Everything useful past M1 needs us to train one, or V4.1-scale hardware.
- **Memory semantics.** Recall triggers only on the exact trailing tokens. It does not chain facts or paraphrase well. Retrieval beats it on small and verbatim knowledge bases. It is a complement to kleos retrieval, not a replacement.
- **Leakage and collisions.**
  - Any text that ends in a shared n-gram reads the user's rows: +3.8% NLL on our paragraph.
  - Facts with the same trigger collide (last write wins).
- **Unlearning granularity.**
  - Per-user erase is exact.
  - Per-fact erase is exact only for independently written, disjoint rows. Joint OPT entangles a user's facts, so deleting one requires re-optimizing the rest. kleos must store which facts each row depends on.
- **Spec drift.** Row ids are meaningless without the exact token map, multipliers and primes. Overlays must be keyed by a model fingerprint and invalidated when any of those change. The transformers-version trap above shows how easily this silently breaks.
- **Engine plumbing.** No engine carries a tenant id to the Engram gather. Multi-tenant batched serving would need engine changes; single-tenant local serving does not.

## Reproduce

```sh
export TMPDIR=/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr/.bench-data/engram/tmp HF_HOME=/Volumes/SSD01/caches/huggingface \
  CARGO_TARGET_DIR=/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr/.targets/engram CARGO_BUILD_JOBS=3
B=/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr/.bench-data/engram
# inputs: git clone github.com/deepseek-ai/Engram -> $B/src/Engram;
#   hf download deepseek-ai/DeepSeek-V3 tokenizer.json tokenizer_config.json --local-dir $B/tok-v3 (same for V4.1-Flash -> tok-v41);
#   V4.1-Flash inference/engram.py -> $B/engines/hf/inf/inference_engram.py
# python env: any venv with torch, transformers, tokenizers, numpy, sympy (we used .bench-data/infer/.venv read-only)
python -I scripts/engram/parity_ref.py --demo $B/src/Engram/engram_demo_v1.py --tokenizer $B/tok-v3 \
  --official $B/engines/hf/inf/inference_engram.py --official-tokenizer $B/tok-v41 --out $B/ref
(cd scripts/engram/engram-index && cargo build --release)
$CARGO_TARGET_DIR/release/engram-index parity $B/tok-v41/tokenizer.json $B/ref/v41
$CARGO_TARGET_DIR/release/engram-index parity $B/tok-v3/tokenizer.json $B/ref/demo $B/ref/paper27b $B/ref/ngram4
$CARGO_TARGET_DIR/release/engram-index bench $B/tok-v3/tokenizer.json 1000000
# overlay edit: git clone github.com/littlemex/zenmai-lab @d845b14d; hf download littlemex/engram-nanogpt-wikitext103 --include "engram-seed1337/*"
python -I scripts/engram/overlay_edit.py --lab $B/src/zenmai-lab/experiments/2026-10-21-engram-nanogpt \
  --ckpt $B/models/engram-nanogpt/engram-seed1337/model.safetensors --out $B/overlay
$CARGO_TARGET_DIR/release/engram-index parity <gpt2 tokenizer.json> $B/overlay/ref-nanogpt
```

## Sources

**Papers:**
- Engram paper: https://arxiv.org/abs/2601.07372 (also `Engram_paper.pdf` in https://github.com/deepseek-ai/Engram, Apache-2.0, commit fb7f84a) [read]
- User as Engram: https://arxiv.org/abs/2606.19172; code https://github.com/19PINE-AI/user-as-engram [read]
- Memory Grafting https://arxiv.org/abs/2605.20948, Engram-Nine https://arxiv.org/abs/2601.16531, CXL pooling https://arxiv.org/abs/2603.10087 [read via subagent]

**DeepSeek-V4.1-Flash** [read; `engram.py` also run]:
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash, files `config.json`, `inference/engram.py`, `inference/model.py`, `model.safetensors.index.json`

**Engines and tooling:**
- vLLM Engram docs https://docs.vllm.ai/en/v0.32.0/features/engram/; PRs https://github.com/vllm-project/vllm/pull/62238 and /pull/57023 [2nd]
- llama.cpp PR https://github.com/ggml-org/llama.cpp/pull/30817 [2nd]
- Unsloth guide https://unsloth.ai/docs/models/deepseek-v4.1-flash [2nd]
- SGLang PRs https://github.com/sgl-project/sglang/pull/44521, /pull/39205, /pull/45231; Mooncake https://github.com/kvcache-ai/Mooncake/pull/4083 [2nd]
- inference-sim https://github.com/inference-sim/inference-sim/issues/1924 [2nd]
- QingGo/EngramDB https://github.com/QingGo/EngramDB [2nd]

**Small model:**
- https://huggingface.co/littlemex/engram-nanogpt-wikitext103 and https://github.com/littlemex/zenmai-lab (commit d845b14d) [ran]
