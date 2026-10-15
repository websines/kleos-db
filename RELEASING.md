# Releasing kleos

Nothing has been published under the kleos name yet. Publishing is a deliberate
owner decision; this file is the checklist for when it happens.

## What publishes, and when

| Workflow | Publishes | Trigger |
|---|---|---|
| `.github/workflows/crates-publish.yml` | `kleos-rocks` (if new), then `kleos` to crates.io; cuts the GitHub Release | pushing a `v<X.Y.Z>` tag, or manual dispatch with `publish=true` |
| `.github/workflows/python-publish.yml` | `kleos` wheels/sdist and the four integration packages to PyPI | pushing a `py-v<X.Y.Z>` tag, or manual dispatch with `publish=true` |

Manual dispatch without `publish=true` builds and verifies only. **Do not push
`v*` or `py-v*` tags until a release is approved.**

## Name availability (checked 2026-10-15)

All free: crates.io `kleos`, `kleos-rocks`, `kleos-infer`; PyPI `kleos`,
`langchain-kleos`, `llama-index-vector-stores-kleos`, `langgraph-store-kleos`,
`kleos-mcp`.

## One-time setup before the first release

1. Reserve the names (publish from an owner account, or configure Trusted
   Publishing first).
2. crates.io Trusted Publishing for `kleos` and `kleos-rocks`: repo
   `websines/kleos-db`, workflow `crates-publish.yml`, environment `crates-io`.
3. PyPI Trusted Publishing for the five PyPI projects: repo `websines/kleos-db`,
   workflow `python-publish.yml`, environment `pypi`.
4. Create the `crates-io` and `pypi` environments in the GitHub repo settings
   (add required reviewers so a stray tag cannot publish on its own).

## Per release

1. Move the `## Unreleased` section of `CHANGELOG.md` to `## <version> — <date>`.
2. Bump versions consistently (`scripts/check_versions.py` must pass):
   `kleos-core/Cargo.toml`, `kleos-lib-python/Cargo.toml` and `pyproject.toml`,
   the install snippet in `kleos-core/README-kleos.md`.
3. Local checks: `cargo package -p kleos --list` (contents), then
   `cargo publish -p kleos --dry-run` (full verify build).
4. Tag `v<version>` for crates.io and `py-v<version>` for PyPI.
