# kleos — provenance and licensing

kleos is a transactional relational-graph-vector database that uses Datalog
(CozoScript) for queries. It is derived from two earlier open-source projects:

| Project | Repository | Base used | Date |
|---|---|---|---|
| CozoDB (original) | <https://github.com/cozodb/cozo> | commit `481af058abac9444ea8c9c52c78f096ed4b5bfc4` | 2024-12-04 |
| mnestic (maintained fork of CozoDB) | <https://github.com/shuruheel/mnestic> | v0.18.0+, commit `352bf20553275587870c7f230369ded10007e03c` | 2026-09-08 |

kleos started from the mnestic commit above, renamed throughout. The git
history of both projects is not carried in this repository; use the commits
above to compare or port changes.

The design records under `docs/specs/`, `docs/reviews/`, `docs/plans/`,
`docs/research/` and `docs/concepts/` were written for mnestic. The rename
replaced the product name in them, so where they say "kleos", the decisions,
sign-offs, releases and issue numbers they describe belong to mnestic and its
maintainer. mnestic's release history in `CHANGELOG.md` is kept verbatim.

## Relationship to CozoDB and mnestic

kleos is **not** CozoDB or mnestic, and is not affiliated with or endorsed by
their authors. The original design and implementation are the work of
**Ziyang Hu** and **the Cozo Project Authors**; the mnestic fork's additions are
the work of **Shan Rizvi**. All credit for that work belongs to them.

The Mozilla Public License grants no trademark rights (MPL-2.0 §2.3), so this
project uses its own name and package identities and does not use the CozoDB
or mnestic names, logos, or package names. The query language keeps its
original name, CozoScript.

## License

Mozilla Public License 2.0 — see [`LICENSE.txt`](LICENSE.txt).

- Original work © 2022–2023 The Cozo Project Authors.
- mnestic fork modifications © 2026 Shan Rizvi.

Obligations when changing this code (MPL-2.0 §3):

- Every file stays under MPL-2.0, and the source of distributed builds must be
  available under it.
- Keep every existing per-file copyright and license notice. You may add your
  own copyright line to files you modify or create; never remove or alter
  existing ones.
- New files should carry the standard MPL-2.0 header with your own copyright
  line.

## Third-party code and designs

### turbovec (MIT)

`kleos-core/src/runtime/vec_quant.rs` (compressed vector codes for quantized
HNSW traversal) follows the design of turbovec, an implementation of
TurboQuant (Zandieh et al., arXiv:2504.19874): the globally permuted,
sign-flipped block Walsh-Hadamard rotation applied in two rounds, the
Lloyd-Max codebook fitted to the distribution of a rotated unit vector's
coordinates, and the length-renormalised inner-product estimator. The code
was written for kleos (its own frozen seed, PRNG, padding rule and bit
layout); no turbovec source file was copied. turbovec's licence:

```text
MIT License

Copyright (c) 2026 Ryan Codrai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

RotorQuant (<https://github.com/scrya-com/rotorquant>, MIT) block rotations
(Clifford Cl(3,0) rotors, 4-D quaternion "IsoQuant" and 2-D "PlanarQuant"
blocks) were evaluated against the Hadamard rotation but are not shipped; no
RotorQuant code is included. The `turbo-quant` crate (MIT OR Apache-2.0) was
reviewed and not used.

### tasksource-decider-nano (Apache-2.0; model weights, not shipped in the library)

`kleos-infer/src/decider_nano.rs` runs
[`tasksource/tasksource-decider-nano`](https://huggingface.co/tasksource/tasksource-decider-nano)
(150M ModernBERT/Ettin cross-encoder, by tasksource / Damien Sileo), licensed
Apache-2.0 per its model card. kleos ships no weights: they are read from the
user's Hugging Face cache. The inference code is a Rust port written for
kleos from the model card and the repository's `decider.py` (Apache-2.0); the
input layout (shared state, tree attention, option pooling) follows that
reference. Its base model is `cross-encoder/ettin-reranker-150m-v1`.

The default decider is kleos's fine-tune of it,
[`Mazino0/kleos-decider-nano`](https://huggingface.co/Mazino0/kleos-decider-nano)
(weights version `nano-tune-v1`), published under MIT for the fine-tuning
changes; the repository also carries the base model's Apache-2.0 licence
(`LICENSE-APACHE-base`). Like the base weights, it is read from the user's
Hugging Face cache and is not bundled with the library.

### EngramEdit (MIT; method, no code copied)

`scripts/engram/engramedit.py` implements the editing method of EngramEdit
(<https://github.com/ModalityDance/EngramEdit>, MIT; arXiv 2610.10533): targets
computed across several phrasings of a fact, mapped to the N-gram memory rows
they activate, and a joint ridge solve with a stronger penalty on frequently
reused rows (its hyperparameter defaults are used). The code is a new
implementation for the 132M nanogpt Engram model; none of EngramEdit's
LongCat-specific code is included.

### DeepSeek Engram and the Engram hashing references

`kleos-engram` reproduces the Engram addressing of DeepSeek-V4.1-Flash
`inference/engram.py` (MIT) and `engram_demo_v1.py` from
<https://github.com/deepseek-ai/Engram> (Apache-2.0): tokenizer compression,
multiplier generation (a port of NumPy's `SeedSequence`/PCG64 bounded draws)
and prime-sized multi-head hashing. It is a Rust re-implementation checked
bit-exactly against those references (`kleos-engram/tests/parity.rs`); the
reference files themselves are not included.

### engram-nanogpt-wikitext103 (Apache-2.0; model weights, not shipped)

The milestone evaluation (`scripts/engram/engramedit.py`,
`scripts/engram/wikitext_rowfreq.py`) runs
[`littlemex/engram-nanogpt-wikitext103`](https://huggingface.co/littlemex/engram-nanogpt-wikitext103)
(132M, Apache-2.0) through its loader in
<https://github.com/littlemex/zenmai-lab> (Apache-2.0), imported from a local
checkout at run time. Neither weights nor loader code are committed. Row
frequencies are counted over WikiText-103
(<https://huggingface.co/datasets/Salesforce/wikitext>, CC BY-SA 3.0) at run
time into `.bench-data/`.

### Evaluation data (not shipped in the library)

`scripts/infer-bench/decide/` and `kleos-core/examples/decide_eval.rs` can
evaluate on a sample of [VitaminC](https://huggingface.co/datasets/tals/vitaminc)
(Schuster et al., 2021; Wikipedia-derived, CC BY-SA 3.0). The sample is
written to `.bench-data/` at run time and is not committed. The memory items
in `kleos-core/examples/data/decide_items.json` are hand-written by the kleos
authors.

### Entity-resolution evaluation data (not shipped in the library)

`kleos-core/examples/entity_eval.rs` evaluates entity resolution on the
[Leipzig entity-matching benchmark](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution)
DBLP-ACM (Köpcke, Thor and Rahm, "Evaluation of entity resolution approaches
on real-world match problems", PVLDB 2010), licensed CC BY 4.0. It is
downloaded to `.bench-data/entity/` and is not committed. The memory-style
mentions in `kleos-core/examples/data/entity_mentions.jsonl` are hand-written
by the kleos authors.

### Fine-tuning data for decider-nano (not shipped; built at run time)

`scripts/infer-bench/decide/train/build_data.py` builds the fine-tuning set
into `.bench-data/nano-tune/data/` (never committed). Fine-tuned weights are
written to `.bench-data/nano-tune/models/`; the selected run is published as
`Mazino0/kleos-decider-nano` (see above).

| Source | Licence | Use |
|---|---|---|
| [VitaminC](https://huggingface.co/datasets/tals/vitaminc) train split (Schuster et al., 2021) | CC BY-SA 3.0 (Wikipedia-derived) | claim/evidence pairs and old→new revision pairs; the test split is used for evaluation only |
| [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) oracle file (Wu et al., 2024) | MIT | knowledge-update pairs (has_answer turns ordered by session date) and multi-session negatives; 25 KU + 40 multi-session questions held out for evaluation |
| [PAWS](https://huggingface.co/datasets/google-research-datasets/paws) labeled_final (Zhang et al., 2019) | PAWS licence: free for any purpose, attribution to Google LLC appreciated | duplicate positives and hard negatives |
| [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) (Williams et al., 2018) | per genre: OANC (free), CC BY-SA 3.0, CC BY 3.0 | 900 training pairs; a 600-item validation_matched subset for the forgetting check |
| Programmatic memory pairs (`memgen.py`) | kleos authors (MPL-2.0) | update / contradiction / compatible / duplicate, entity mentions, entity typing, relation mapping |

**Share-alike note:** VitaminC and part of MultiNLI are CC BY-SA. Whether
weights fine-tuned on them are an adaptation that must carry CC BY-SA terms is
an open legal question. The maintainers decided (2026-10-23) to publish
`nano-tune-v1` under MIT, with the training sources and their licences listed
on the model card. A weights build without the CC BY-SA sources is possible by
dropping them in `build_data.py`.

### candle-metal-kernels (MIT OR Apache-2.0; vendored and patched)

`third_party/candle-metal-kernels` is `candle-metal-kernels` 0.11.0 from
Hugging Face [candle](https://github.com/huggingface/candle), dual-licensed MIT
OR Apache-2.0, applied with `[patch.crates-io]`. kleos changes one function so
the Metal Shading Language version is pinned rather than inherited from the
host executable's SDK. See `third_party/candle-metal-kernels/KLEOS_PATCH.md`.
### Memory benchmark datasets (evaluation only; downloaded at run time, never committed)

The L2 benchmark harness (`scripts/bench-l2/`, results in
`docs/benchmarks/memory.md`) reads these datasets from `.bench-data/bench-l2/`
and the Hugging Face cache. Neither is redistributed, and kleos ships no
weights or data derived from them.

| Dataset | Revision | Licence | Use |
|---|---|---|---|
| [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) cleaned (Wu et al., 2024, "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory") | HF `98d7416c24c778c2fee6e6f3006e7a073259d48f` (`longmemeval_s_cleaned.json`, `longmemeval_oracle.json`) | MIT | retrieval and QA evaluation |
| [LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., 2024, "Evaluating Very Long-Term Conversational Memory of LLM Agents") | GitHub `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` (`data/locomo10.json`) | **CC BY-NC 4.0** (non-commercial) | retrieval and QA evaluation only; never used for training |

The QA judge prompts in `scripts/bench-l2/qa.py` follow LongMemEval's
`evaluate_qa.py` (MIT) and the LoCoMo evaluation prompts published with Mem0's
evaluation code (Apache-2.0).

The Mem0 head-to-head (`scripts/bench-l2/mem0_adapter.py`) imports the
open-source [`mem0ai`](https://pypi.org/project/mem0ai/) 2.2.1 package
(Apache-2.0) at run time from a separate virtual environment under
`.bench-data/mem0-h2h/`. No Mem0 code is vendored or redistributed.

### Indra OCR (code Apache-2.0, ported; model weights modified OpenRAIL-M, downloaded at run time)

`kleos-docs/src/ocr/` is a Rust port of the OCR pipeline of
[Indra OCR](https://github.com/datalab-to/indra) (`indra-ocr` 1.1.2 on PyPI)
by Datalab, licensed **Apache-2.0**: line detection decoding, line crops,
CTC decoding, the n-gram LM beam search, script routing, right-to-left
handling, reading order, layout blocks, Markdown and tables, the PDF
text-layer checks and the text-layer/OCR document logic. The port follows the
Python code closely (constants, rules and comments); the copyright in that
logic is Datalab's.

The model weights it runs ([`datalab-to/indra-ocr`](https://huggingface.co/datalab-to/indra-ocr),
revision `v1.1.0`) are **not** Apache-2.0: they are under a *modified
OpenRAIL-M* licence that Datalab summarizes as "free for research, personal
use, and startups under $20M funding/revenue, cannot be used competitively
with our API"; other use needs a licence from Datalab. kleos never bundles or
redistributes the weights: they are downloaded from Hugging Face into the
user's cache when OCR is first used. The OCR backend is a trait, so another
engine can replace Indra.

### pdfium, tract, Chonkie

- **pdfium** (BSD-3-Clause / Apache-2.0, the Chromium PDF library) is loaded
  at run time as a shared library through
  [`pdfium-render`](https://github.com/ajrcarey/pdfium-render) (MIT OR
  Apache-2.0); the prebuilt library comes from
  [pdfium-binaries](https://github.com/bblanchon/pdfium-binaries) (MIT
  packaging), downloaded into the kleos cache, never vendored.
- **tract** ([sonos/tract](https://github.com/sonos/tract), MIT OR
  Apache-2.0) runs the ONNX graphs.
- **Chonkie** ([chonkie-inc/chonkie](https://github.com/chonkie-inc/chonkie),
  MIT): `kleos-docs/src/chunk.rs` reimplements its token, sentence, recursive
  and semantic chunking rules (the delimiter split and short-split merge of
  `chonkie-core`, the greedy merge indices, the recursive levels). No Chonkie
  code is vendored; the parity test compares against the Python package at
  run time.
