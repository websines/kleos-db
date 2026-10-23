# kleos patch of candle-metal-kernels 0.11.0

Vendored from crates.io `candle-metal-kernels` 0.11.0 (Hugging Face candle,
MIT OR Apache-2.0), applied through `[patch.crates-io]` in the workspace and
`kleos-lib-python` manifests.

Only change: `src/kernel.rs` `get_compile_options()` sets
`MTLCompileOptions.languageVersion` (MSL 3.2 on macOS 15+, 3.1 on macOS 14).

Why: without an explicit version, Metal picks a default from the SDK the main
executable was linked against. Inside a Python interpreter linked against an
old SDK (miniconda's python3.12: SDK 11.1), the kernels compile as pre-3.1 MSL
and fail with `unknown type name 'float16_t'` (`bfloat` needs MSL 3.1), so every
Metal model call from the Python bindings errored. Rust binaries (SDK 26.x) were
unaffected.

Drop this patch once upstream candle sets the language version.
