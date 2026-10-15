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
