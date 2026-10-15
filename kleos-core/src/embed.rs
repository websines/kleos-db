/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Text embeddings inside the database (kleos fork, UPGRADE_PLAN D3).
//!
//! * [`Embedder`] is the plug-in trait: anything that turns a batch of texts
//!   into fixed-width vectors. kleos itself ships no model; the optional
//!   `kleos-infer` crate (cargo feature `infer`) provides built-in ones.
//! * [`register_embedder`] makes an embedder callable from CozoScript as
//!   `embed('name', text)` (and `embed('name', text, dim)` for a Matryoshka
//!   prefix, re-normalised).
//! * Every embedding goes through one process-wide cache keyed on
//!   `(model id, SHA-256(text))`, so re-inserting or re-querying the same text
//!   never runs the model twice.
//! * A column default may read other columns of the row. With
//!   `vec: <F32; 768> default embed('gemma', text)`, every `:put` that omits
//!   `vec` computes it from `text`, and every `:update` that changes `text`
//!   recomputes it. Rows written by one statement are embedded in batches
//!   (see [`EmbedPrefetch`]) rather than one model call per row.
//!
//! * Embedders are switched on and off at runtime with [`enable_embedder`] /
//!   [`disable_embedder`] / [`embedders`], or `::embedder enable | disable |
//!   list | backfill` in CozoScript. `embed()` on a disabled model errors;
//!   `embed_or_null()` returns null instead, for nullable computed columns
//!   that must keep accepting writes (docs/specs/embedder-toggle.md).
//!
//! The registry and the cache are process-wide, not per [`crate::Db`]: a
//! model is a process resource, and every open database shares it. Register
//! an embedder under a name before running scripts that use that name;
//! stored relations persist the *name*, so it must be registered again after
//! a restart before rows are written.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use itertools::Itertools;
use lazy_static::lazy_static;
use miette::{miette, Result};
use ndarray::Array1;
use sha2::{Digest, Sha256};
use smartstring::{LazyCompact, SmartString};

use crate::bail;
use crate::data::expr::{Expr, Op};
use crate::data::tuple::Tuple;
use crate::data::value::{DataValue, Vector};

/// Something that turns texts into vectors. Implementations must be cheap to
/// share (`Send + Sync`) and return exactly one vector of length
/// [`Embedder::dim`] per input text, in input order.
pub trait Embedder: Send + Sync {
    /// Embed a batch of texts.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    /// Width of every returned vector.
    fn dim(&self) -> usize;
    /// Stable identity of the model *and* everything that changes its output
    /// (weights, quantisation, pooling, prompt). Used as part of the cache
    /// key; when `None`, the registered name plus a registration serial is
    /// used, so re-registering a name never serves stale vectors.
    fn model_id(&self) -> Option<String> {
        None
    }
    /// Largest batch the embedder wants per call. Larger requests are split.
    fn max_batch(&self) -> usize {
        32
    }
    /// Whether the model's weights are in memory (`None`: not tracked, the
    /// default for custom embedders). Shown by `::embedder list`.
    fn is_loaded(&self) -> Option<bool> {
        None
    }
    /// The device the weights live on once loaded (`cpu`, `metal`, ...).
    fn device(&self) -> Option<String> {
        None
    }
    /// Release the model's memory. Called by [`disable_embedder`] before the
    /// registry drops its handle; the default does nothing.
    fn unload(&self) {}
}

/// Built-in embedders (cargo feature `infer`).
#[cfg(feature = "infer")]
#[path = "embed_builtin.rs"]
pub mod builtin;

struct Registered {
    embedder: Arc<dyn Embedder>,
    cache_id: Arc<str>,
    info: EmbedderInfo,
}

/// Bookkeeping kept next to a registered (or disabled) embedder.
#[derive(Debug, Clone, Default)]
pub(crate) struct EmbedderInfo {
    /// `gemma`, `model2vec` or `custom`.
    pub(crate) kind: &'static str,
    /// Repo or model id, for display.
    pub(crate) model: String,
    pub(crate) precision: Option<String>,
    /// Requested device (`auto`, `cpu`, `metal`).
    pub(crate) device_pref: Option<String>,
    /// How to rebuild it (built-in embedders only).
    pub(crate) spec: Option<EmbedderSpec>,
    /// Name of the paired entry registered with this one (gemma's
    /// `{name}_query`), on the primary entry.
    pub(crate) twin: Option<String>,
    /// Name of the primary entry, on a twin.
    pub(crate) twin_of: Option<String>,
}

/// A disabled embedder: enough to turn it back on with `::embedder enable name`.
struct Parked {
    info: EmbedderInfo,
    dim: usize,
    /// Custom (Rust-registered) embedders cannot be rebuilt from a spec, so
    /// the handle is kept; built-in ones keep nothing but the spec.
    custom: Option<Arc<dyn Embedder>>,
}

lazy_static! {
    static ref REGISTRY: RwLock<BTreeMap<String, Registered>> = RwLock::new(BTreeMap::new());
    static ref DISABLED: Mutex<BTreeMap<String, Parked>> = Mutex::new(BTreeMap::new());
    static ref CACHE: Mutex<EmbedCache> = Mutex::new(EmbedCache::new(DEFAULT_CACHE_BYTES));
}
static SERIAL: AtomicU64 = AtomicU64::new(0);

/// Default embedding-cache budget: 64 MiB (about 21k 768-d vectors).
pub const DEFAULT_CACHE_BYTES: usize = 64 << 20;

/// Make `embedder` callable as `embed('name', ...)`. Replaces any embedder
/// already registered under `name` (cached vectors of the old one are never
/// served for the new one). It is listed by `::embedder list` as kind
/// `custom`, and `::embedder disable` / `::embedder enable name` park and
/// restore it.
pub fn register_embedder(name: &str, embedder: Arc<dyn Embedder>) -> Result<()> {
    let model = embedder.model_id().unwrap_or_else(|| "custom".to_string());
    register_with_info(
        name,
        embedder,
        EmbedderInfo {
            kind: "custom",
            model,
            ..Default::default()
        },
    )
}

pub(crate) fn register_with_info(
    name: &str,
    embedder: Arc<dyn Embedder>,
    info: EmbedderInfo,
) -> Result<()> {
    if name.is_empty() {
        bail!("an embedder name must not be empty");
    }
    if embedder.dim() == 0 {
        bail!("embedder '{name}' reports a dimension of 0");
    }
    let serial = SERIAL.fetch_add(1, Ordering::Relaxed);
    let cache_id: Arc<str> = match embedder.model_id() {
        Some(id) => Arc::from(id),
        None => Arc::from(format!("{name}#{serial}")),
    };
    let old = REGISTRY
        .write()
        .map_err(|_| miette!("embedder registry poisoned"))?
        .insert(
            name.to_string(),
            Registered {
                embedder,
                cache_id,
                info,
            },
        );
    if let Ok(mut d) = DISABLED.lock() {
        d.remove(name);
    }
    if let Some(old) = old {
        if old.embedder.is_loaded().is_some() {
            old.embedder.unload();
        }
    }
    Ok(())
}

/// Remove the embedder registered under `name`; returns whether one existed.
/// Unlike [`disable_embedder`] this forgets it entirely (nothing to
/// re-enable) and does not touch a paired `_query` entry.
pub fn unregister_embedder(name: &str) -> bool {
    if let Ok(mut d) = DISABLED.lock() {
        d.remove(name);
    }
    REGISTRY
        .write()
        .map(|mut r| r.remove(name).is_some())
        .unwrap_or(false)
}

/// Names and dimensions of the registered (enabled) embedders.
pub fn registered_embedders() -> Vec<(String, usize)> {
    REGISTRY
        .read()
        .map(|r| {
            r.iter()
                .map(|(k, v)| (k.clone(), v.embedder.dim()))
                .collect()
        })
        .unwrap_or_default()
}

// ------------------------------------------------------------ on/off switch

/// What [`enable_embedder`] (and `::embedder enable`) should build. Every
/// field is optional:
///
/// * `model`: `embeddinggemma` (default; also `gemma` or
///   `google/embeddinggemma-300m`), or a Model2Vec model such as
///   `potion-base-8M` / `minishlab/potion-base-8M` (a bare `potion-*` name
///   resolves to the `minishlab` repo; any other `owner/repo` is read as a
///   Model2Vec model).
/// * `precision`: gemma weights, `int8` (default), `int4`, `f16`, `bf16`,
///   `f32`. Model2Vec models are always f32.
/// * `device`: `auto` (default: Apple GPU when built with `infer-metal`, else
///   CPU), `cpu`, `metal`.
/// * `dim`: Matryoshka output width for gemma (768 default; 512, 256, 128).
///
/// With every field `None` and `name` disabled earlier, [`enable_embedder`]
/// turns the previous configuration back on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmbedderSpec {
    /// Model name or Hugging Face repo.
    pub model: Option<String>,
    /// Weight precision.
    pub precision: Option<String>,
    /// Device preference.
    pub device: Option<String>,
    /// Output width.
    pub dim: Option<usize>,
}

impl EmbedderSpec {
    /// A spec for `model` with every other field at its default.
    pub fn model(model: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
            ..Default::default()
        }
    }
    /// EmbeddingGemma at int8 on the auto device, 768 wide.
    pub fn embeddinggemma() -> Self {
        Self::model("embeddinggemma")
    }
    fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

impl EmbedderSpec {
    /// Parse a command-line embedder argument,
    /// `name=model[:precision[:device[:dim]]]`, e.g. `gemma=embeddinggemma:int8`,
    /// `gemma=embeddinggemma:int8:metal:256` or `potion=minishlab/potion-base-8M`.
    /// Empty fields keep their defaults (`g=embeddinggemma::cpu`).
    pub fn parse_cli(arg: &str) -> Result<(String, EmbedderSpec)> {
        let Some((name, rest)) = arg.split_once('=') else {
            bail!("embedder argument '{arg}' must look like name=model[:precision[:device[:dim]]]");
        };
        let name = name.trim();
        if name.is_empty() {
            bail!("embedder argument '{arg}' has an empty name");
        }
        let mut parts = rest.split(':').map(str::trim);
        let field = |p: Option<&str>| p.filter(|s| !s.is_empty()).map(str::to_string);
        let model = field(parts.next());
        let precision = field(parts.next());
        let device = field(parts.next());
        let dim = match field(parts.next()) {
            None => None,
            Some(d) => match d.parse::<usize>() {
                Ok(n) if n > 0 => Some(n),
                _ => bail!("embedder argument '{arg}': dim must be a positive integer"),
            },
        };
        if parts.next().is_some() {
            bail!("embedder argument '{arg}' has too many ':' fields");
        }
        Ok((
            name.to_string(),
            EmbedderSpec {
                model,
                precision,
                device,
                dim,
            },
        ))
    }
}

/// One row of [`embedders`] / `::embedder list`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedderStatus {
    /// Registered name (what `embed('name', ..)` takes).
    pub name: String,
    /// `gemma`, `model2vec` or `custom`.
    pub kind: String,
    /// Repo or model id.
    pub model: String,
    /// Output width.
    pub dim: usize,
    /// Whether `embed('name', ..)` currently works.
    pub enabled: bool,
    /// Whether the weights are in memory (`None`: not tracked).
    pub loaded: Option<bool>,
    /// Weight precision (built-in gemma only).
    pub precision: Option<String>,
    /// Actual device once loaded, otherwise the requested one.
    pub device: Option<String>,
}

/// Every enabled and disabled embedder, sorted by name.
pub fn embedders() -> Vec<EmbedderStatus> {
    let mut out: BTreeMap<String, EmbedderStatus> = BTreeMap::new();
    if let Ok(r) = REGISTRY.read() {
        for (name, reg) in r.iter() {
            let e = &reg.embedder;
            out.insert(
                name.clone(),
                EmbedderStatus {
                    name: name.clone(),
                    kind: reg.info.kind.to_string(),
                    model: reg.info.model.clone(),
                    dim: e.dim(),
                    enabled: true,
                    loaded: e.is_loaded(),
                    precision: reg.info.precision.clone(),
                    device: e.device().or_else(|| reg.info.device_pref.clone()),
                },
            );
        }
    }
    if let Ok(d) = DISABLED.lock() {
        for (name, p) in d.iter() {
            out.entry(name.clone()).or_insert_with(|| EmbedderStatus {
                name: name.clone(),
                kind: p.info.kind.to_string(),
                model: p.info.model.clone(),
                dim: p.dim,
                enabled: false,
                loaded: match &p.custom {
                    Some(c) => c.is_loaded(),
                    None => Some(false),
                },
                precision: p.info.precision.clone(),
                device: p.info.device_pref.clone(),
            });
        }
    }
    out.into_values().collect()
}

/// Turn an embedder on under `name` (process-wide, like the registry).
///
/// * `spec` empty and `name` disabled earlier: re-enable it as it was
///   (custom embedders get their parked handle back; built-in ones are
///   rebuilt from their spec and load lazily on first use).
/// * otherwise: build the built-in model `spec` describes (default
///   EmbeddingGemma int8, auto device) and register it, replacing any
///   embedder of that name. Gemma also registers `{name}_query` (query
///   prompt). Needs the `infer` cargo feature.
///
/// Weights are not loaded here; the first `embed()` loads them.
pub fn enable_embedder(name: &str, spec: EmbedderSpec) -> Result<()> {
    if name.is_empty() {
        bail!("an embedder name must not be empty");
    }
    if spec.is_empty() {
        let parked = DISABLED
            .lock()
            .map_err(|_| miette!("embedder registry poisoned"))?
            .remove(name);
        if let Some(p) = parked {
            let restored = match (&p.custom, &p.info.spec) {
                (Some(c), _) => register_with_info(name, c.clone(), p.info.clone()),
                (None, Some(s)) => enable_builtin(name, s.clone()),
                (None, None) => Err(miette!("embedder '{name}' cannot be re-enabled")),
            };
            if restored.is_err() {
                if let Ok(mut d) = DISABLED.lock() {
                    d.insert(name.to_string(), p);
                }
            }
            return restored;
        }
        if lookup_registered(name).is_some() {
            // already on: nothing to do
            return Ok(());
        }
    }
    let old_twin = REGISTRY
        .read()
        .ok()
        .and_then(|r| r.get(name).and_then(|e| e.info.twin.clone()));
    enable_builtin(name, spec)?;
    if let Some(old_twin) = old_twin {
        let new_twin = REGISTRY
            .read()
            .ok()
            .and_then(|r| r.get(name).and_then(|e| e.info.twin.clone()));
        if new_twin.as_ref() != Some(&old_twin) {
            unregister_embedder(&old_twin);
        }
    }
    Ok(())
}

#[cfg(feature = "infer")]
fn enable_builtin(name: &str, spec: EmbedderSpec) -> Result<()> {
    builtin::enable_from_spec(name, spec)
}

#[cfg(not(feature = "infer"))]
fn enable_builtin(name: &str, spec: EmbedderSpec) -> Result<()> {
    #[derive(Debug, thiserror::Error, miette::Diagnostic)]
    #[error(
        "cannot enable embedder '{0}' (model '{1}'): kleos was built without the `infer` cargo feature, so it has no built-in embedders"
    )]
    #[diagnostic(code(eval::embedder_feature_missing))]
    #[diagnostic(help(
        "rebuild kleos with `--features infer` (or `infer-metal` for the Apple GPU), or register a custom embedder from Rust with kleos::embed::register_embedder"
    ))]
    struct FeatureMissing(String, String);
    Err(FeatureMissing(
        name.to_string(),
        spec.model.unwrap_or_else(|| "embeddinggemma".to_string()),
    )
    .into())
}

/// Turn the embedder `name` off: it is removed from the registry (with
/// gemma's `{name}_query` twin; disabling the twin disables the pair), its
/// model is unloaded, and its cached vectors are dropped. It stays listed as
/// disabled and `::embedder enable name` turns it back on. Returns `false`
/// when it was already disabled; errors when `name` was never enabled.
pub fn disable_embedder(name: &str) -> Result<bool> {
    let removed = {
        let mut reg = REGISTRY
            .write()
            .map_err(|_| miette!("embedder registry poisoned"))?;
        let primary = match reg.get(name) {
            Some(r) => r.info.twin_of.clone().unwrap_or_else(|| name.to_string()),
            None => {
                drop(reg);
                if DISABLED
                    .lock()
                    .map(|d| d.contains_key(name))
                    .unwrap_or(false)
                {
                    return Ok(false);
                }
                return Err(not_found(name));
            }
        };
        let Some(p) = reg.remove(&primary) else {
            return Err(not_found(name));
        };
        let twin = p.info.twin.as_ref().and_then(|t| reg.remove(t));
        (primary, p, twin)
    };
    let (primary, p, twin) = removed;
    let mut cache_ids = vec![p.cache_id.clone()];
    if let Some(t) = &twin {
        t.embedder.unload();
        cache_ids.push(t.cache_id.clone());
    }
    p.embedder.unload();
    let custom = if p.info.spec.is_some() {
        None
    } else {
        Some(p.embedder.clone())
    };
    let parked = Parked {
        dim: p.embedder.dim(),
        info: p.info.clone(),
        custom,
    };
    drop(p);
    drop(twin);
    if let Ok(mut c) = lock_cache() {
        for id in &cache_ids {
            c.drop_model(&model_tag(id));
        }
    }
    DISABLED
        .lock()
        .map_err(|_| miette!("embedder registry poisoned"))?
        .insert(primary, parked);
    release_freed_memory();
    Ok(true)
}

/// Ask the system allocator to hand freed pages back to the OS. A model's
/// weights are a few hundred large allocations; macOS libmalloc keeps freed
/// large blocks dirty in its cache (measured: the whole 1.2 GB load peak of
/// EmbeddingGemma stayed in the process footprint after the weights were
/// dropped, and grew with every enable/disable cycle), and glibc keeps freed
/// heap tops. Best effort; a no-op elsewhere or under another global allocator.
pub fn release_freed_memory() {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        // SAFETY: documented libmalloc API; a null zone means every zone and
        // a goal of 0 means release as much as possible.
        unsafe {
            malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
        }
    }
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        // SAFETY: glibc API, safe to call at any time.
        unsafe {
            malloc_trim(0);
        }
    }
}

fn is_disabled(name: &str) -> bool {
    DISABLED
        .lock()
        .map(|d| d.contains_key(name))
        .unwrap_or(false)
}

fn not_found(name: &str) -> miette::Report {
    if is_disabled(name) {
        #[derive(Debug, thiserror::Error, miette::Diagnostic)]
        #[error("embedder '{0}' is disabled; turn it back on with `::embedder enable {0}`")]
        #[diagnostic(code(eval::embedder_disabled))]
        #[diagnostic(help(
            "turn it back on with `::embedder enable {0}` (Rust: kleos::embed::enable_embedder), \
             or write through `embed_or_null` into a nullable column and run \
             `::embedder backfill <relation>` later"
        ))]
        struct EmbedderDisabled(String);
        EmbedderDisabled(name.to_string()).into()
    } else {
        #[derive(Debug, thiserror::Error, miette::Diagnostic)]
        #[error(
            "no embedder is enabled under the name '{0}'; enable one with `::embedder enable {0} {{model: 'embeddinggemma'}}`"
        )]
        #[diagnostic(code(eval::embedder_not_found))]
        #[diagnostic(help(
            "enable one with `::embedder enable {0} {{model: 'embeddinggemma'}}` (needs the `infer` \
             feature) or register one from Rust with kleos::embed::register_embedder"
        ))]
        struct EmbedderNotFound(String);
        EmbedderNotFound(name.to_string()).into()
    }
}

fn lookup_registered(name: &str) -> Option<(Arc<dyn Embedder>, Arc<str>)> {
    REGISTRY
        .read()
        .ok()?
        .get(name)
        .map(|r| (r.embedder.clone(), r.cache_id.clone()))
}

fn lookup(name: &str) -> Result<(Arc<dyn Embedder>, Arc<str>)> {
    lookup_registered(name).ok_or_else(|| not_found(name))
}

/// Whether `name` is currently enabled.
pub fn is_embedder_enabled(name: &str) -> bool {
    lookup_registered(name).is_some()
}

/// Embed `texts` with the embedder registered under `model`, going through
/// the cache. Cache misses are sent to the model in batches of at most
/// [`Embedder::max_batch`]; duplicate texts inside one call are embedded once.
pub fn embed_texts(model: &str, texts: &[&str]) -> Result<Vec<Arc<[f32]>>> {
    let (embedder, cache_id) = lookup(model)?;
    let keys: Vec<CacheKey> = texts.iter().map(|t| cache_key(&cache_id, t)).collect();
    let mut out: Vec<Option<Arc<[f32]>>> = {
        let mut cache = lock_cache()?;
        keys.iter().map(|k| cache.get(k)).collect()
    };
    // unique misses, in first-seen order
    let mut miss_pos: HashMap<CacheKey, usize> = HashMap::new();
    let mut miss_texts: Vec<&str> = vec![];
    for (i, slot) in out.iter().enumerate() {
        if slot.is_none() && !miss_pos.contains_key(&keys[i]) {
            miss_pos.insert(keys[i], miss_texts.len());
            miss_texts.push(texts[i]);
        }
    }
    if miss_texts.is_empty() {
        return Ok(out.into_iter().map(|v| v.unwrap()).collect());
    }
    let dim = embedder.dim();
    let mut fresh: Vec<Arc<[f32]>> = Vec::with_capacity(miss_texts.len());
    for chunk in miss_texts.chunks(embedder.max_batch().max(1)) {
        let vecs = embedder.embed(chunk)?;
        if vecs.len() != chunk.len() {
            bail!(
                "embedder '{model}' returned {} vectors for {} texts",
                vecs.len(),
                chunk.len()
            );
        }
        for v in vecs {
            if v.len() != dim {
                bail!(
                    "embedder '{model}' returned a vector of width {}, but reports dim {dim}",
                    v.len()
                );
            }
            fresh.push(Arc::from(v));
        }
    }
    {
        let mut cache = lock_cache()?;
        for (k, i) in &miss_pos {
            cache.insert(*k, fresh[*i].clone());
        }
    }
    for (i, slot) in out.iter_mut().enumerate() {
        if slot.is_none() {
            *slot = Some(fresh[miss_pos[&keys[i]]].clone());
        }
    }
    Ok(out.into_iter().map(|v| v.unwrap()).collect())
}

/// Hit/miss counters and size of the embedding cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EmbedCacheStats {
    /// Lookups answered from the cache.
    pub hits: u64,
    /// Lookups that had to run the model.
    pub misses: u64,
    /// Vectors currently held.
    pub entries: usize,
    /// Approximate bytes currently held.
    pub bytes: usize,
    /// Byte budget.
    pub capacity_bytes: usize,
}

/// Current cache statistics.
pub fn embedding_cache_stats() -> EmbedCacheStats {
    lock_cache().map(|c| c.stats()).unwrap_or_default()
}

/// Set the cache budget in bytes (0 disables caching). Shrinking drops the
/// cached vectors.
pub fn set_embedding_cache_capacity(bytes: usize) {
    if let Ok(mut c) = lock_cache() {
        if bytes < c.capacity {
            c.clear();
        }
        c.capacity = bytes;
    }
}

/// Drop every cached vector (counters are kept).
pub fn clear_embedding_cache() {
    if let Ok(mut c) = lock_cache() {
        c.clear();
    }
}

fn lock_cache() -> Result<std::sync::MutexGuard<'static, EmbedCache>> {
    CACHE.lock().map_err(|_| miette!("embedding cache poisoned"))
}

/// 8 bytes identifying the model, then 16 bytes of SHA-256 over
/// (model id, text). The model prefix lets [`disable_embedder`] drop one
/// model's vectors.
type CacheKey = [u8; 24];

fn model_tag(model_id: &str) -> [u8; 8] {
    let d = Sha256::digest(model_id.as_bytes());
    let mut t = [0u8; 8];
    t.copy_from_slice(&d[..8]);
    t
}

fn cache_key(model_id: &str, text: &str) -> CacheKey {
    let mut h = Sha256::new();
    h.update((model_id.len() as u64).to_le_bytes());
    h.update(model_id.as_bytes());
    h.update(text.as_bytes());
    let d = h.finalize();
    let mut k = [0u8; 24];
    k[..8].copy_from_slice(&model_tag(model_id));
    k[8..].copy_from_slice(&d[..16]);
    k
}

/// Two-generation approximate LRU: lookups hit `cur` then `prev` (promoting
/// into `cur`); when `cur` reaches half the budget it becomes `prev` and the
/// old `prev` is dropped. O(1) per operation, never above the budget.
struct EmbedCache {
    cur: HashMap<CacheKey, Arc<[f32]>>,
    prev: HashMap<CacheKey, Arc<[f32]>>,
    cur_bytes: usize,
    prev_bytes: usize,
    capacity: usize,
    hits: u64,
    misses: u64,
}

const ENTRY_OVERHEAD: usize = 64;

impl EmbedCache {
    fn new(capacity: usize) -> Self {
        Self {
            cur: HashMap::new(),
            prev: HashMap::new(),
            cur_bytes: 0,
            prev_bytes: 0,
            capacity,
            hits: 0,
            misses: 0,
        }
    }
    fn entry_bytes(v: &[f32]) -> usize {
        std::mem::size_of_val(v) + ENTRY_OVERHEAD
    }
    fn get(&mut self, k: &CacheKey) -> Option<Arc<[f32]>> {
        if let Some(v) = self.cur.get(k) {
            self.hits += 1;
            return Some(v.clone());
        }
        if let Some(v) = self.prev.remove(k) {
            self.hits += 1;
            self.prev_bytes -= Self::entry_bytes(&v);
            self.insert(*k, v.clone());
            return Some(v);
        }
        self.misses += 1;
        None
    }
    fn insert(&mut self, k: CacheKey, v: Arc<[f32]>) {
        let b = Self::entry_bytes(&v);
        if self.capacity == 0 || b > self.capacity / 2 {
            return;
        }
        if self.cur_bytes + b > self.capacity / 2 {
            self.prev = std::mem::take(&mut self.cur);
            self.prev_bytes = self.cur_bytes;
            self.cur_bytes = 0;
        }
        if let Some(old) = self.cur.insert(k, v) {
            self.cur_bytes -= Self::entry_bytes(&old);
        }
        self.cur_bytes += b;
    }
    fn drop_model(&mut self, tag: &[u8; 8]) {
        let mut freed_cur = 0;
        self.cur.retain(|k, v| {
            let keep = &k[..8] != tag;
            if !keep {
                freed_cur += Self::entry_bytes(v);
            }
            keep
        });
        self.cur_bytes -= freed_cur;
        let mut freed_prev = 0;
        self.prev.retain(|k, v| {
            let keep = &k[..8] != tag;
            if !keep {
                freed_prev += Self::entry_bytes(v);
            }
            keep
        });
        self.prev_bytes -= freed_prev;
        self.cur.shrink_to_fit();
        self.prev.shrink_to_fit();
    }
    fn clear(&mut self) {
        self.cur.clear();
        self.prev.clear();
        self.cur_bytes = 0;
        self.prev_bytes = 0;
    }
    fn stats(&self) -> EmbedCacheStats {
        EmbedCacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.cur.len() + self.prev.len(),
            bytes: self.cur_bytes + self.prev_bytes,
            capacity_bytes: self.capacity,
        }
    }
}

// ----------------------------------------------------------------- CozoScript

/// `embed(model, text)` / `embed(model, text, dim)`.
pub(crate) const OP_EMBED: Op = Op {
    name: "OP_EMBED",
    min_arity: 2,
    vararg: true,
    inner: op_embed,
};

/// `embed_or_null(model, text)` / `embed_or_null(model, text, dim)`: like
/// `embed`, but null while `model` is disabled or not registered. Meant for
/// nullable computed columns that must keep accepting writes while the
/// embedder is off; `::embedder backfill` fills them later.
pub(crate) const OP_EMBED_OR_NULL: Op = Op {
    name: "OP_EMBED_OR_NULL",
    min_arity: 2,
    vararg: true,
    inner: op_embed_or_null,
};

fn op_embed(args: &[DataValue]) -> Result<DataValue> {
    embed_op(args, "embed", false)
}

fn op_embed_or_null(args: &[DataValue]) -> Result<DataValue> {
    embed_op(args, "embed_or_null", true)
}

fn embed_op(args: &[DataValue], fname: &str, lenient: bool) -> Result<DataValue> {
    if args.len() > 3 {
        bail!("'{fname}' takes (model, text) or (model, text, dim)");
    }
    let model = match &args[0] {
        DataValue::Str(s) => s,
        v => bail!("'{fname}' requires the model name as a string, got {v:?}"),
    };
    let text = match &args[1] {
        DataValue::Str(s) => s,
        DataValue::Null => return Ok(DataValue::Null),
        v => bail!("'{fname}' requires the text as a string, got {v:?}"),
    };
    let truncate = match args.get(2) {
        None => None,
        Some(d) => match d.get_int() {
            Some(n) if n > 0 => Some(n as usize),
            _ => bail!("'{fname}' requires dim to be a positive integer, got {d:?}"),
        },
    };
    if lenient && !is_embedder_enabled(model) {
        return Ok(DataValue::Null);
    }
    let v = embed_texts(model, &[text.as_str()])?.pop().unwrap();
    Ok(DataValue::Vec(Vector::F32(to_array(&v, truncate)?)))
}

fn to_array(v: &[f32], truncate: Option<usize>) -> Result<Array1<f32>> {
    match truncate {
        None => Ok(Array1::from(v.to_vec())),
        Some(d) if d > v.len() => bail!(
            "'embed' cannot truncate a {}-d embedding to {d} dimensions",
            v.len()
        ),
        Some(d) => {
            let a = Array1::from(v[..d].to_vec());
            let n = a.dot(&a).sqrt();
            Ok(if n > 0.0 { a / n } else { a })
        }
    }
}

// ------------------------------------------------------------ batched writes

/// One `embed(model, text, ..)` call found inside a computed column default.
struct EmbedCall {
    model: Expr,
    text: Expr,
}

/// The model-name expressions of every `embed()` / `embed_or_null()` call in
/// `e` (used by `::embedder backfill`).
pub(crate) fn embed_models_in(e: &Expr) -> Vec<Expr> {
    let mut calls = vec![];
    collect_embed_calls(e, &mut calls);
    calls.into_iter().map(|c| c.model).collect()
}

fn collect_embed_calls(e: &Expr, out: &mut Vec<EmbedCall>) {
    match e {
        Expr::Apply { op, args, .. } => {
            if (op.name == OP_EMBED.name || op.name == OP_EMBED_OR_NULL.name) && args.len() >= 2 {
                out.push(EmbedCall {
                    model: args[0].clone(),
                    text: args[1].clone(),
                })
            }
            for a in args.iter() {
                collect_embed_calls(a, out);
            }
        }
        Expr::Cond { clauses, .. } => {
            for (c, v) in clauses {
                collect_embed_calls(c, out);
                collect_embed_calls(v, out);
            }
        }
        _ => {}
    }
}

/// Rows per prefetch batch.
const PREFETCH_ROWS: usize = 256;

/// Wraps the row stream of a write. When the target's computed columns call
/// `embed()`, rows are pulled `PREFETCH_ROWS` at a time and their texts
/// embedded in one batched pass that fills the cache; the per-row `embed()`
/// evaluation that follows then only hits the cache. Without such columns it
/// is a plain pass-through. Prefetch errors are ignored here: the per-row
/// evaluation reports them with row context.
pub(crate) struct EmbedPrefetch<I> {
    inner: I,
    calls: Vec<EmbedCall>,
    buf: std::collections::VecDeque<Tuple>,
}

impl<I: Iterator<Item = Tuple>> EmbedPrefetch<I> {
    /// `computed` are the (binding-filled) computed column defaults of the write.
    pub(crate) fn new<'a>(inner: I, computed: impl IntoIterator<Item = &'a Expr>) -> Self {
        let mut calls = vec![];
        for e in computed {
            collect_embed_calls(e, &mut calls);
        }
        Self {
            inner,
            calls,
            buf: Default::default(),
        }
    }

    fn refill(&mut self) {
        self.buf.extend(self.inner.by_ref().take(PREFETCH_ROWS));
        let mut by_model: BTreeMap<SmartString<LazyCompact>, Vec<SmartString<LazyCompact>>> =
            BTreeMap::new();
        for call in &self.calls {
            for t in &self.buf {
                if let (Ok(DataValue::Str(m)), Ok(DataValue::Str(s))) =
                    (call.model.eval(t), call.text.eval(t))
                {
                    by_model.entry(m).or_default().push(s);
                }
            }
        }
        for (model, texts) in by_model {
            if !is_embedder_enabled(&model) {
                continue;
            }
            let refs = texts.iter().map(|s| s.as_str()).collect_vec();
            if let Err(e) = embed_texts(&model, &refs) {
                log::debug!("embedding prefetch for '{model}' failed: {e}");
            }
        }
    }
}

impl<I: Iterator<Item = Tuple>> Iterator for EmbedPrefetch<I> {
    type Item = Tuple;
    fn next(&mut self) -> Option<Tuple> {
        if self.calls.is_empty() {
            return self.inner.next();
        }
        if self.buf.is_empty() {
            self.refill();
        }
        self.buf.pop_front()
    }
}
