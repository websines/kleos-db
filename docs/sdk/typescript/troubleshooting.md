# Troubleshooting

## The native module

**`kleos native module not found for darwin-arm64`.** The embedded backend
needs `sdk/ts/native/kleos.<platform>-<arch>.node`. Build it:

```sh
cd sdk/ts && npm run build:native
```

Or point `KLEOS_NATIVE` at a built `.node` file, or use `connect()` against a
server, which needs no native code.

**Build fails with `rustc 1.90.0 is not supported`.** The `infer` and `docs`
features need Rust 1.99: `RUSTUP_TOOLCHAIN=1.99.0 npm run build:native`.

**Other platforms.** `build:native` builds for the machine it runs on (macOS,
Linux, Windows; arm64 or x64). The module uses Node-API version 8, so one build
works with Node 18, 20, 22 and 24. To build for another target, run it on that
platform (or in CI per platform) and ship each
`kleos.<platform>-<arch>.node`.

**The process is killed (exit 137) right after rebuilding, on macOS.**
Overwriting a loaded library in place breaks its code-signature cache.
`build:native` deletes the old file before copying; if you copy by hand, `rm`
first.

**`documents need the native module built with the docs feature`.** Rebuild
with `KLEOS_FEATURES=docs` (plus `infer` or `infer-metal` for models).
`nativeFeatures()` shows what the loaded module has.

## Models

**The first `enableEmbedder` takes a while.** It downloads the model once
(roughly 300–400 MB per model) into `HF_HOME` (default
`~/.cache/huggingface`). Set `HF_HOME` to a disk with room. Later starts load
from the cache.

**`HF_HUB_OFFLINE` is set and the model is missing.** Offline mode forbids
downloads. Fetch the model once with network access, or copy the cache.

**`gated repos need their licence accepted`.** Only EmbeddingGemma v1
(`embeddinggemma-300m`) is gated. The default (`embeddinggemma`, v2) and the
decider and extractor are open. For v1, accept the licence on Hugging Face
and set `HF_TOKEN`.

**`the memory store's vectors were made by '…'`.** A store remembers which embedder made
its vectors and refuses another, so models are never mixed. Enable the same
model, or move the store with `reembed('<embedder>')`.

## Memory and speed

**Memory use.** `KLEOS_MEMORY_BUDGET` (default `3GB`) caps what all models use
together. Batches shrink to fit, and idle models unload after
`KLEOS_MODEL_IDLE_SECS` (default 900; `0` keeps them loaded).
`KLEOS_ACTIVATION_CAP` (default `256MB`) caps one forward pass. Lower both on
small machines; expect slower bulk work, not failures.

**Recall is slow while a big document ingests.** Background ingestion runs in
the bulk lane, and interactive calls go first. If you pass `priority: 'bulk'`
on your own recalls, they wait behind ingestion; leave the default.

**Recall finds nothing for a paraphrase.** Without an embedder, recall matches
words. Enable one (see [Getting started](getting-started.md#enabling-models)).

## Errors

All errors are `KleosError` with `op` (and `status` for remote calls):

| Message (start) | Meaning |
|---|---|
| `database closed` | The store was closed; open it again. |
| `unknown memory op` | A typo in `memory(op)`. |
| `bad options: …` | An option has the wrong type (e.g. `file` must be a path; bytes go in `attachments`). |
| `memory namespace '…' kept changing under remember` | Many writers on one namespace; `remember` retried 8 times. Retry. |
| `memory … needs write access` (403) | A read-only token tried to write. |
| `request failed` | The server can't be reached (wrong URL, not running). |
| `tool forget needs input (confirmation)` | MCP: pass `onInputRequired`, or `confirm: true` after asking the user. |
