# Contributing to kleos

kleos is a maintained fork of [CozoDB](https://github.com/cozodb/cozo) under
MPL-2.0 — a transactional relational-graph-vector database with Datalog,
maintained as a substrate for agentic memory. See [`NOTICE.md`](NOTICE.md) for
provenance and licensing, [`CHANGELOG.md`](CHANGELOG.md) for every
divergence from upstream, and [`ROADMAP.md`](ROADMAP.md) for where the project
is going. This is not "official CozoDB" — credit for the engine's design and
the vast majority of its code belongs to Ziyang Hu and the Cozo Project Authors.

> This fork does **not** use the upstream Cozo CLA. Contributions are accepted
> under the project's MPL-2.0 license (see below).

## Where to start

- [`good first issue`](https://github.com/websines/kleos-db/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
  — bounded, well-scoped tasks, many of them fixes upstream left unmerged when
  it went dormant. Each closes a real, citable gap.
- [`help wanted`](https://github.com/websines/kleos-db/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
  — larger features from the roadmap that are self-contained enough to take on
  whole. For these, **propose your design in the issue thread before writing a
  large patch** — it protects your time as much as ours.
- Found a bug? Open an issue with a minimal CozoScript repro. A failing test is
  even better (see the test notes below).

If you want to work on something not covered by an issue, open one first and
say what you're planning. `ROADMAP.md` explains the project's focus (and its
deliberate non-goals: federation, multi-model breadth, clustering, Cypher
*writes*) so you don't invest in something we can't merge.

## Building & testing

The workspace's active members are `kleos-core` (published as the `kleos`
crate), `kleos-rocks` (RocksDB bridge, published as `kleos-rocks`), `kleos-bin`
(server/REPL), and `kleos-core-examples`. The language bindings under
`kleos-lib-*` are excluded from the workspace; only the Python one
(`kleos-lib-python`, published to PyPI as `kleos`) is actively maintained.

```bash
# Fast path — default features are SQLite-only (no C++ build):
cargo check --workspace           # quick validity check
cargo test  -p kleos --lib      # engine unit tests (mem backend, fast)
cargo test  -p kleos            # + integration tests (incl. fork regressions)

# RocksDB backend (compiles the C++ bridge via cxx; slow first build):
git submodule update --init --recursive
cargo build -p kleos --features storage-rocksdb

# Server / REPL:
cargo run -p kleos-bin
```

Naming convention (important): the published **package** is `kleos`, but the
**lib name stays `cozo`**, so existing CozoDB code (`use kleos::...`) works
unchanged. Please don't "finish the rebrand" in a PR — the rename is
deliberate and load-bearing for downstream users.

### A note on tests

The in-memory backend uses a **different join operator** than the persistent
backends, so planner or stored-relation bugs often don't reproduce on `mem`.
Tests that exercise stored-relation or planner behavior should use the
**SQLite** backend with a temporary directory — fast, no C++ build, and it
exercises the real `stored_*` join path. See
`kleos-core/tests/matjoin_regression.rs` for the pattern. Add a failing test
that reproduces the issue before fixing it.

### Formatting

Parts of the inherited tree have never been `rustfmt`-formatted, so a blanket
`cargo fmt` produces large unrelated diffs. Format only the files you touch
(`rustfmt --edition 2021 <files>`), and keep reformatting out of functional
commits.

## Pull requests

- Branch from `main`; keep changes focused — one concern per PR.
- Don't break the inherited upstream test suite — `cargo test -p kleos` must
  pass. CI (`.github/workflows/build.yml`) runs build + tests on every push/PR.
- **Preserve the per-file `Copyright … The Cozo Project Authors` MPL headers**
  on any file you modify. You may add your own copyright line; never remove
  theirs.
- New files carry the standard MPL-2.0 header with **your own** copyright line
  (see `NOTICE.md`), not a copy of an existing author's line.
- Note any user-visible divergence from upstream in `CHANGELOG.md`, in the
  same PR, under the `Unreleased` heading.
- Don't bump versions or touch publishing — releases are banked and cut by the
  maintainer on a regular cadence (that cadence is part of the project's
  promise). If your change affects the Python binding, say so in the PR and the
  maintainer will sync `kleos-lib-python` at release time.

## Conduct

Be kind and constructive — see [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## License

By contributing you agree your contributions are licensed under MPL-2.0.
