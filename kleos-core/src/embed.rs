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
}

/// Built-in embedders (cargo feature `infer`).
#[cfg(feature = "infer")]
#[path = "embed_builtin.rs"]
pub mod builtin;

struct Registered {
    embedder: Arc<dyn Embedder>,
    cache_id: Arc<str>,
}

lazy_static! {
    static ref REGISTRY: RwLock<BTreeMap<String, Registered>> = RwLock::new(BTreeMap::new());
    static ref CACHE: Mutex<EmbedCache> = Mutex::new(EmbedCache::new(DEFAULT_CACHE_BYTES));
}
static SERIAL: AtomicU64 = AtomicU64::new(0);

/// Default embedding-cache budget: 64 MiB (about 21k 768-d vectors).
pub const DEFAULT_CACHE_BYTES: usize = 64 << 20;

/// Make `embedder` callable as `embed('name', ...)`. Replaces any embedder
/// already registered under `name` (cached vectors of the old one are never
/// served for the new one).
pub fn register_embedder(name: &str, embedder: Arc<dyn Embedder>) -> Result<()> {
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
    REGISTRY
        .write()
        .map_err(|_| miette!("embedder registry poisoned"))?
        .insert(name.to_string(), Registered { embedder, cache_id });
    Ok(())
}

/// Remove the embedder registered under `name`; returns whether one existed.
pub fn unregister_embedder(name: &str) -> bool {
    REGISTRY
        .write()
        .map(|mut r| r.remove(name).is_some())
        .unwrap_or(false)
}

/// Names and dimensions of the registered embedders.
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

fn lookup(name: &str) -> Result<(Arc<dyn Embedder>, Arc<str>)> {
    let reg = REGISTRY
        .read()
        .map_err(|_| miette!("embedder registry poisoned"))?;
    match reg.get(name) {
        Some(r) => Ok((r.embedder.clone(), r.cache_id.clone())),
        None => {
            #[derive(Debug, thiserror::Error, miette::Diagnostic)]
            #[error("no embedder is registered under the name '{0}'")]
            #[diagnostic(code(eval::embedder_not_found))]
            #[diagnostic(help("register one with kleos::embed::register_embedder before using embed()"))]
            struct EmbedderNotFound(String);
            Err(EmbedderNotFound(name.to_string()).into())
        }
    }
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

type CacheKey = [u8; 16];

fn cache_key(model_id: &str, text: &str) -> CacheKey {
    let mut h = Sha256::new();
    h.update((model_id.len() as u64).to_le_bytes());
    h.update(model_id.as_bytes());
    h.update(text.as_bytes());
    let d = h.finalize();
    let mut k = [0u8; 16];
    k.copy_from_slice(&d[..16]);
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

fn op_embed(args: &[DataValue]) -> Result<DataValue> {
    if args.len() > 3 {
        bail!("'embed' takes (model, text) or (model, text, dim)");
    }
    let model = match &args[0] {
        DataValue::Str(s) => s,
        v => bail!("'embed' requires the model name as a string, got {v:?}"),
    };
    let text = match &args[1] {
        DataValue::Str(s) => s,
        DataValue::Null => return Ok(DataValue::Null),
        v => bail!("'embed' requires the text as a string, got {v:?}"),
    };
    let truncate = match args.get(2) {
        None => None,
        Some(d) => match d.get_int() {
            Some(n) if n > 0 => Some(n as usize),
            _ => bail!("'embed' requires dim to be a positive integer, got {d:?}"),
        },
    };
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

fn collect_embed_calls(e: &Expr, out: &mut Vec<EmbedCall>) {
    match e {
        Expr::Apply { op, args, .. } => {
            if op.name == OP_EMBED.name && args.len() >= 2 {
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
