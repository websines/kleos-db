# Embedder on/off switch

Status: implemented (wave2/s0-embed). Code: `kleos-core/src/embed.rs`,
`kleos-core/src/embed_builtin.rs`, `kleos-core/src/runtime/embedder.rs`,
`parse/sys.rs` (`EmbedderOp`). Tests: `kleos-core/tests/embed_toggle.rs`.

## Goal

Users can turn the built-in embedder on and off whenever they want, from every
surface: CozoScript (and so HTTP), Rust, Python and `kleos-bin` flags. Turning
an embedder off frees its model.

## Scope: process-wide

The embedder registry and the embedding cache are process-wide (a model is a
process resource). So `::embedder enable/disable` affect every database open
in the process, not just the one the script ran on. `::embedder backfill`
writes one relation of the database it runs on.

## CozoScript

```
::embedder list
::embedder enable gemma {model: 'embeddinggemma', precision: 'int8', device: 'auto', dim: 768}
::embedder enable potion {model: 'potion-base-8M'}         # or 'minishlab/potion-base-8M'
::embedder enable gemma                                     # re-enable as before, or gemma defaults
::embedder disable gemma
::embedder backfill docs
```

* `list` returns `name, kind, model, dim, enabled, loaded, precision, device`,
  one row per enabled or disabled embedder. `kind` is `gemma`, `model2vec` or
  `custom` (registered from Rust or Python). `loaded` is null for custom
  embedders, which do not report it. `device` is the actual device once the
  model has loaded, otherwise the requested one. `precision` uses kleos-infer's
  short names (`q8` for int8).
* `enable name {opts}`: options are `model`, `precision`, `device` and `dim`.
  All are optional. The defaults are EmbeddingGemma at int8 (the best measured
  quality per byte), `auto` device and 768 dims. Gemma also registers
  `{name}_query` with the query prompt. Weights load lazily on the first
  `embed()`. With no options and a name that was disabled earlier, the previous
  configuration comes back. That covers custom embedders too: their handle is
  parked on disable. With no options and an already enabled name, nothing
  happens. Enabling a name that is already in use replaces that embedder.
* Without the `infer` cargo feature, enabling a built-in model fails with
  `eval::embedder_feature_missing`, which names the feature (`infer` /
  `infer-metal`). `list`, `disable` and re-enabling a parked custom embedder
  still work.
* `disable name` unregisters the embedder, and gemma's `_query` twin with it
  (disabling the twin disables the pair). It calls `Embedder::unload` to
  release the weights, drops the registry's handle and removes the model's
  entries from the embedding cache. Cache keys now carry an 8-byte model tag so
  this does not need a full scan by hash. The op returns `ALREADY_DISABLED` for
  a disabled name and errors for a name that was never enabled.
* `enable`, `disable` and `backfill` are refused in read-only (immutable)
  scripts. `list` is allowed. All four dispatch before a storage transaction
  opens. Inside an imperative block, `list`, `enable` and `disable` work, but
  `backfill` does not because it runs its own write.

## Semantics while an embedder is off

* `embed('m', text)` errors with `eval::embedder_disabled`: "embedder 'm' is
  disabled; turn it back on with `::embedder enable m`". An unknown name gives
  `eval::embedder_not_found`, which says how to enable one.
* **Strict (default):** a computed column `vec: <F32; 768> default
  embed('m', text)` makes `:put` / `:update` fail while `m` is off. The whole
  write is rejected and nothing is stored. An explicit vector still works.
* **Lenient (opt-in): `embed_or_null`.** Declare the column nullable and use
  `embed_or_null` in the default:

  ```
  :create docs { id: Int => text: String, vec: <F32; 768>? default embed_or_null('gemma', text) }
  ```

  `embed_or_null` behaves exactly like `embed` while the embedder is on. While
  it is off or unregistered, it returns null, so writes keep succeeding with a
  null vector. Every other failure (model error, bad arguments) still raises.
  If the column is not nullable, the null is rejected by the column type, so
  leniency requires `?` on the column.
* HNSW indexes skip null vectors, so rows written while the embedder was off
  are invisible to vector search until they are backfilled. FTS and other
  indexes on the text are unaffected.
* **`::embedder backfill rel`** (Rust `backfill_embeddings`, Python
  `db.backfill_embeddings`) finds every non-key column of `rel` whose default
  calls `embed`/`embed_or_null`. For each such column it re-writes the
  column's non-key source columns of the null rows with an ordinary `:update`.
  That update recomputes the column through the same batched prefetch and
  cache as a `:put`, and maintains HNSW/FTS/LSH indexes and triggers. Before
  writing anything, it checks that every constant model name used by those
  columns is enabled. If one is not, it fails with the same "disabled" error
  and writes nothing. It returns `column, filled, still_null`, where
  `still_null` counts rows whose source text is itself null. Running it again
  is a no-op.

### Why `embed_or_null` rather than a schema or op option

* It is per-column and stored with the schema, because the default expression
  is persisted. Writes therefore behave the same from every client without
  anyone remembering a per-op flag. Setting it per op would require a new
  `:put` option threaded through `stored.rs`, `buffer_tt_puts`, the imperative
  path and triggers.
* It needs no grammar or schema-format change and composes like any other
  function, for example `embed_or_null('m', concat(title, ' ', body), 256)`.
* The strict default is unchanged: `embed` still errors, so no existing schema
  silently starts storing nulls.
* Requiring a nullable column makes the opt-in visible in the type. A
  non-nullable column with `embed_or_null` keeps strict behaviour, because the
  null fails the type check.
* Backfill does not need extra metadata. It treats any null value in an
  embedded column as missing, including a null that the user wrote explicitly.

## Rust

```rust
use kleos::{EmbedderSpec, enable_embedder, disable_embedder, embedders};
db.enable_embedder("gemma", EmbedderSpec::embeddinggemma())?;      // also free fns
db.enable_embedder("potion", EmbedderSpec::model("potion-base-8M"))?;
db.disable_embedder("gemma")?;                                   // -> bool
db.enable_embedder("gemma", EmbedderSpec::default())?;           // back as before
let all: Vec<EmbedderStatus> = db.embedders();
db.backfill_embeddings("docs")?;                                 // NamedRows
```

`Embedder` gained the default methods `is_loaded()`, `device()` and
`unload()`. Existing implementations keep compiling. `register_embedder`
entries show as kind `custom`. `EmbedderSpec::parse_cli` parses the
`name=model[:precision[:device[:dim]]]` form.

## Python

`db.enable_embedder(name, model=None, precision=None, device=None, dim=None)`,
`db.disable_embedder(name)`, `db.embedders()` (a list of dicts) and
`db.backfill_embeddings(relation)`. `kleos.enable_embedder` and
`kleos.disable_embedder` are module-level twins, next to the existing
`kleos.register_embedder`, `unregister_embedder` and `registered_embedders`,
which still return `(name, dim)` for enabled embedders. The Python crate has
`infer` / `infer-metal` features. They are not in the maturin feature list,
because candle 0.11 needs rustc ≥ 1.99 and the default toolchain here is 1.90.

## kleos-bin

`kleos-bin server|repl --embedder gemma=embeddinggemma:int8 --embedder
potion=potion-base-8M`. The flag can be repeated, and an invalid flag fails at
startup. It needs a build with `--features infer` (or `infer-metal`). Over
HTTP, the `::embedder` ops are ordinary scripts.

## Caveats

* Freeing memory: after disable, the model's `Arc`s are dropped as soon as
  in-flight `embed()` calls return. Disable then asks the system allocator to
  return freed pages (`malloc_zone_pressure_relief` on macOS, `malloc_trim` on
  glibc). On macOS that is not enough. libmalloc keeps freed large blocks dirty
  in its large cache, where the kernel reclaims them only under memory
  pressure. These blocks are the load-time f32 buffers, and on CPU the weights
  too. kleos-infer's probe (`kleos-infer/tests/unload_memory.rs`, gemma int8 on
  Metal) shows the difference:

  | Allocator setting | Loaded footprint | After unload | Over 3 cycles |
  |---|---|---|---|
  | Default | 840 MB | 474 MB | grows about 280 MB per cycle |
  | `MallocLargeCache=0` | 432 MB | 10 MB | flat |

  The model code does free everything. On macOS, run long-lived processes that
  toggle embedders with `MallocLargeCache=0` in the environment. It has to be
  set before the process starts, so it cannot be set from inside kleos.
* A custom embedder's memory belongs to the host. Disable parks its handle so
  that it can be re-enabled, and calls `unload()` so the implementation can
  release what it can.
* Backfill does not support TxTime relations, or columns whose default reads
  only key columns. It runs as one write transaction over all null rows.
* Stored relations persist the embedder name, not the model. After a restart,
  enable the embedder again (for example with `--embedder`) before writing.
