/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! One memory budget for every built-in model in the process.
//!
//! * **Budget** ([`global`]): the bytes the built-in models may hold at once,
//!   weights plus the activations of running forward passes
//!   (`KLEOS_MEMORY_BUDGET`, e.g. `2GB`; default [`DEFAULT_BUDGET`]). A
//!   single forward pass is further capped at [`activation_cap`]
//!   (`KLEOS_ACTIVATION_CAP`, default [`DEFAULT_ACTIVATION_CAP`]); models
//!   size their batches to fit it, so the peak does not grow with the load.
//! * **Model slots** ([`ModelSlot`]): a model's lazily loaded weights plus its
//!   [`Gate`]. Loading reserves the weights' bytes, unloading idle models
//!   (least recently used first) when the budget would be exceeded; an idle
//!   model is also unloaded after `KLEOS_MODEL_IDLE_SECS` (default
//!   [`DEFAULT_IDLE_SECS`]; `0` keeps models loaded) and reloads on its next
//!   call.
//! * **Shared instances** ([`shared`]): one model object per key in the
//!   process, so several databases (or several `enable` calls) share weights.

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::sched::{current_priority, Gate, GateGuard};
use crate::Result;

/// Default budget for weights plus activations: 3 GiB.
pub const DEFAULT_BUDGET: usize = 3 << 30;
/// Default cap on one forward pass's activations: 256 MiB.
pub const DEFAULT_ACTIVATION_CAP: usize = 256 << 20;
/// Default idle time before a model is unloaded: 15 minutes.
pub const DEFAULT_IDLE_SECS: u64 = 900;

/// Parse `2GB`, `1500MB`, `512MiB`, `3g` or plain bytes.
pub fn parse_bytes(s: &str) -> Option<usize> {
    let t = s.trim().to_ascii_lowercase().replace('_', "");
    let num_end = t.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(t.len());
    let (n, unit) = t.split_at(num_end);
    let n: f64 = n.parse().ok()?;
    let mul: f64 = match unit.trim() {
        "" | "b" => 1.0,
        "k" | "kb" | "kib" => 1024.0,
        "m" | "mb" | "mib" => 1024.0 * 1024.0,
        "g" | "gb" | "gib" => 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    Some((n * mul) as usize)
}

fn env_bytes(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| parse_bytes(&v)).unwrap_or(default)
}

struct BudgetState {
    limit: usize,
    act_cap: usize,
    weights: usize,
    active: usize,
}

/// The process-wide budget (see the module docs).
pub struct MemoryBudget {
    st: Mutex<BudgetState>,
    cv: Condvar,
}

/// Activation bytes held by a running forward pass.
pub struct ActivationGuard {
    bytes: usize,
}

impl Drop for ActivationGuard {
    fn drop(&mut self) {
        if self.bytes > 0 {
            let b = global();
            let mut st = b.st.lock().unwrap();
            st.active = st.active.saturating_sub(self.bytes);
            drop(st);
            b.cv.notify_all();
        }
    }
}

/// The global budget.
pub fn global() -> &'static MemoryBudget {
    static B: OnceLock<MemoryBudget> = OnceLock::new();
    B.get_or_init(|| MemoryBudget {
        st: Mutex::new(BudgetState {
            limit: env_bytes("KLEOS_MEMORY_BUDGET", DEFAULT_BUDGET),
            act_cap: env_bytes("KLEOS_ACTIVATION_CAP", DEFAULT_ACTIVATION_CAP),
            weights: 0,
            active: 0,
        }),
        cv: Condvar::new(),
    })
}

/// The largest activation footprint one forward pass may use now: the
/// activation cap, or what the budget leaves after the loaded weights if that
/// is less (at least 1 MiB, so a model can always run one row).
pub fn activation_cap() -> usize {
    let st = global().st.lock().unwrap();
    st.act_cap.min(st.limit.saturating_sub(st.weights)).max(1 << 20)
}

/// Set the budget and the per-forward activation cap at run time.
pub fn set_memory_budget(limit: usize, activation_cap: Option<usize>) {
    let b = global();
    let mut st = b.st.lock().unwrap();
    st.limit = limit.max(1 << 20);
    if let Some(c) = activation_cap {
        st.act_cap = c.max(1 << 20);
    }
    drop(st);
    b.cv.notify_all();
}

/// Budget counters for status and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetStats {
    /// Weights plus activations allowed.
    pub limit: usize,
    /// Cap on one forward pass's activations.
    pub activation_cap: usize,
    /// Bytes of loaded weights.
    pub weights: usize,
    /// Activation bytes of forward passes running now.
    pub active: usize,
}

/// Current budget counters.
pub fn stats() -> BudgetStats {
    let st = global().st.lock().unwrap();
    BudgetStats { limit: st.limit, activation_cap: st.act_cap, weights: st.weights, active: st.active }
}

impl MemoryBudget {
    /// Reserve `bytes` of activations, waiting while running passes would
    /// push weights plus activations over the limit. A pass that would not
    /// fit even alone runs when nothing else is running.
    pub fn acquire_activations(&self, bytes: usize) -> ActivationGuard {
        let mut st = self.st.lock().unwrap();
        while st.active > 0 && st.weights + st.active + bytes > st.limit {
            st = self.cv.wait(st).unwrap();
        }
        st.active += bytes;
        ActivationGuard { bytes }
    }

    fn add_weights(&self, bytes: usize) {
        self.st.lock().unwrap().weights += bytes;
    }

    fn remove_weights(&self, bytes: usize) {
        let mut st = self.st.lock().unwrap();
        st.weights = st.weights.saturating_sub(bytes);
        drop(st);
        self.cv.notify_all();
    }

    fn over_with(&self, bytes: usize) -> bool {
        let st = self.st.lock().unwrap();
        st.weights + bytes > st.limit
    }
}

/// Activation bytes of a transformer encoder forward pass over `n` rows of
/// padded length `l`: hidden states and projections (`hidden` wide, about a
/// dozen live at the peak), the MLP (`inter` wide, three live), and
/// `attn_maps` attention-sized `heads × l × l` f32 maps (scores,
/// probabilities, masks, relative-position terms). Doubled for candle's
/// power-of-two Metal buffer pool.
pub fn encoder_activation_bytes(
    n: usize,
    l: usize,
    hidden: usize,
    inter: usize,
    heads: usize,
    act_dtype_bytes: usize,
    attn_maps: usize,
) -> usize {
    let per_row = l * hidden * 12 * act_dtype_bytes + l * inter * 3 * act_dtype_bytes + attn_maps * heads * l * l * 4;
    2 * n * per_row
}

/// Rows of padded length `l` that fit both a token budget and the
/// activation cap (at least one).
pub fn rows_for(l: usize, max_tokens: usize, bytes_per_row: impl Fn(usize) -> usize) -> usize {
    let by_tokens = max_tokens / l.max(1);
    let per = bytes_per_row(l).max(1);
    let by_bytes = activation_cap() / per;
    by_tokens.min(by_bytes).max(1)
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

struct Entry {
    id: u64,
    label: String,
    unload: Box<dyn Fn() -> Option<usize> + Send + Sync>,
    last_used: Arc<AtomicU64>,
    busy: Arc<AtomicUsize>,
}

fn registry() -> &'static Mutex<Vec<Entry>> {
    static R: OnceLock<Mutex<Vec<Entry>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(vec![]))
}

fn next_id() -> u64 {
    static N: AtomicU64 = AtomicU64::new(1);
    N.fetch_add(1, Ordering::Relaxed)
}

/// Unload idle loaded models (least recently used first) other than
/// `except` until `need` more bytes of weights fit the budget. Returns the
/// bytes freed.
fn evict_for(need: usize, except: u64) -> usize {
    let b = global();
    if !b.over_with(need) {
        return 0;
    }
    let mut cands: Vec<(u64, u64)> = {
        let reg = registry().lock().unwrap();
        reg.iter()
            .filter(|e| e.id != except && e.busy.load(Ordering::Acquire) == 0)
            .map(|e| (e.last_used.load(Ordering::Relaxed), e.id))
            .filter(|(t, _)| *t != u64::MAX)
            .collect()
    };
    cands.sort();
    let mut freed = 0;
    for (_, id) in cands {
        if !b.over_with(need) {
            break;
        }
        let reg = registry().lock().unwrap();
        if let Some(e) = reg.iter().find(|e| e.id == id) {
            if e.busy.load(Ordering::Acquire) == 0 {
                freed += (e.unload)().unwrap_or(0);
            }
        }
    }
    if freed > 0 {
        release_freed_memory();
    }
    freed
}

/// Ask the system allocator to hand freed pages back to the OS (macOS
/// libmalloc keeps freed large blocks dirty; glibc keeps heap tops).
pub fn release_freed_memory() {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        // SAFETY: documented libmalloc API; null zone = all zones, goal 0 = as much as possible.
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

fn start_reaper() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        let idle = std::env::var("KLEOS_MODEL_IDLE_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_IDLE_SECS);
        if idle == 0 {
            return;
        }
        let tick = Duration::from_secs((idle / 4).clamp(1, 30));
        let _ = std::thread::Builder::new().name("kleos-model-reaper".into()).spawn(move || loop {
            std::thread::sleep(tick);
            reap_idle(idle);
        });
    });
}

/// Unload every loaded, not running model idle for at least `idle_secs`.
/// Returns how many were unloaded.
pub fn reap_idle(idle_secs: u64) -> usize {
    let now = now_secs();
    let ids: Vec<u64> = {
        let reg = registry().lock().unwrap();
        reg.iter()
            .filter(|e| e.busy.load(Ordering::Acquire) == 0)
            .filter(|e| {
                let t = e.last_used.load(Ordering::Relaxed);
                t != u64::MAX && now.saturating_sub(t) >= idle_secs
            })
            .map(|e| e.id)
            .collect()
    };
    let mut n = 0;
    for id in ids {
        let reg = registry().lock().unwrap();
        if let Some(e) = reg.iter().find(|e| e.id == id) {
            if e.busy.load(Ordering::Acquire) == 0 && (e.unload)().is_some() {
                n += 1;
            }
        }
    }
    if n > 0 {
        release_freed_memory();
    }
    n
}

/// Labels of the models currently registered.
pub fn models() -> Vec<String> {
    registry().lock().unwrap().iter().map(|e| e.label.clone()).collect()
}

/// A model's lazily loaded weights, its device [`Gate`] and its budget
/// bookkeeping (see the module docs).
pub struct ModelSlot<W> {
    inner: Arc<Mutex<Option<(Arc<W>, usize)>>>,
    gate: Gate,
    last_used: Arc<AtomicU64>,
    busy: Arc<AtomicUsize>,
    id: u64,
}

/// A forward pass in progress: the device, its activation bytes, and the
/// model marked busy (not evictable) until dropped.
pub struct RunGuard<'a> {
    _gate: GateGuard<'a>,
    busy: &'a AtomicUsize,
}

impl Drop for RunGuard<'_> {
    fn drop(&mut self) {
        self.busy.fetch_sub(1, Ordering::AcqRel);
    }
}

impl<W: Send + Sync + 'static> ModelSlot<W> {
    /// An empty slot, registered for idle unloading and eviction.
    pub fn new(label: impl Into<String>) -> Self {
        let inner: Arc<Mutex<Option<(Arc<W>, usize)>>> = Arc::new(Mutex::new(None));
        let weak: Weak<Mutex<Option<(Arc<W>, usize)>>> = Arc::downgrade(&inner);
        let last_used = Arc::new(AtomicU64::new(u64::MAX));
        let busy = Arc::new(AtomicUsize::new(0));
        let id = next_id();
        let lu = last_used.clone();
        registry().lock().unwrap().push(Entry {
            id,
            label: label.into(),
            unload: Box::new(move || {
                let m = weak.upgrade()?;
                let taken = crate::gemma::with_pool(|| m.lock().unwrap().take());
                taken.map(|(w, bytes)| {
                    crate::gemma::with_pool(|| drop(w));
                    global().remove_weights(bytes);
                    lu.store(u64::MAX, Ordering::Relaxed);
                    bytes
                })
            }),
            last_used: last_used.clone(),
            busy: busy.clone(),
        });
        start_reaper();
        Self { inner, gate: Gate::new(), last_used, busy, id }
    }

    /// The weights, loading them first if needed (reserving their bytes and
    /// evicting idle models to make room).
    pub fn get_or_load(&self, load: impl FnOnce() -> Result<W>, bytes: impl Fn(&W) -> usize) -> Result<Arc<W>> {
        self.touch();
        let mut g = self.inner.lock().unwrap();
        if let Some((w, _)) = g.as_ref() {
            return Ok(w.clone());
        }
        let w = Arc::new(load()?);
        let n = bytes(&w);
        drop(g);
        evict_for(n, self.id);
        g = self.inner.lock().unwrap();
        if let Some((w2, _)) = g.as_ref() {
            return Ok(w2.clone()); // another caller loaded it meanwhile
        }
        global().add_weights(n);
        *g = Some((w.clone(), n));
        Ok(w)
    }

    /// The weights if loaded.
    pub fn peek(&self) -> Option<Arc<W>> {
        self.inner.lock().unwrap().as_ref().map(|(w, _)| w.clone())
    }

    /// Whether the weights are in memory.
    pub fn is_loaded(&self) -> bool {
        self.inner.lock().unwrap().is_some()
    }

    /// Drop the weights (they load again on next use).
    pub fn unload(&self) {
        let taken = crate::gemma::with_pool(|| self.inner.lock().unwrap().take());
        if let Some((w, bytes)) = taken {
            crate::gemma::with_pool(|| drop(w));
            global().remove_weights(bytes);
            self.last_used.store(u64::MAX, Ordering::Relaxed);
        }
    }

    /// Wait for the device at this thread's priority and hold `act_bytes` of
    /// activations until the guard drops.
    pub fn run(&self, act_bytes: usize) -> RunGuard<'_> {
        self.busy.fetch_add(1, Ordering::AcqRel);
        let g = self.gate.acquire(current_priority(), act_bytes);
        self.touch();
        RunGuard { _gate: g, busy: &self.busy }
    }

    /// The device gate.
    pub fn gate(&self) -> &Gate {
        &self.gate
    }

    /// Mark the model busy (not evictable) while a [`crate::sched::Batcher`]
    /// runs it through its own gate.
    pub fn hold(&self) -> HoldGuard<'_> {
        self.busy.fetch_add(1, Ordering::AcqRel);
        self.touch();
        HoldGuard { busy: &self.busy }
    }

    fn touch(&self) {
        self.last_used.store(now_secs(), Ordering::Relaxed);
    }
}

/// Keeps a model from being evicted while held.
pub struct HoldGuard<'a> {
    busy: &'a AtomicUsize,
}

impl Drop for HoldGuard<'_> {
    fn drop(&mut self) {
        self.busy.fetch_sub(1, Ordering::AcqRel);
    }
}

impl<W> Drop for ModelSlot<W> {
    fn drop(&mut self) {
        if let Ok(mut g) = self.inner.lock() {
            if let Some((w, bytes)) = g.take() {
                crate::gemma::with_pool(|| drop(w));
                global().remove_weights(bytes);
            }
        }
        if let Ok(mut reg) = registry().lock() {
            reg.retain(|e| e.id != self.id);
        }
    }
}

/// One model object per `key` in the process: returns the live instance if
/// one exists, else builds it with `make`.
pub fn shared<T: Any + Send + Sync>(key: &str, make: impl FnOnce() -> Result<Arc<T>>) -> Result<Arc<T>> {
    static MAP: OnceLock<Mutex<HashMap<String, Weak<dyn Any + Send + Sync>>>> = OnceLock::new();
    let map = MAP.get_or_init(|| Mutex::new(HashMap::new()));
    let mut m = map.lock().unwrap();
    if let Some(w) = m.get(key) {
        if let Some(a) = w.upgrade() {
            if let Ok(t) = a.downcast::<T>() {
                return Ok(t);
            }
        }
    }
    m.retain(|_, w| w.strong_count() > 0);
    let t: Arc<T> = make()?;
    let erased: Arc<dyn Any + Send + Sync> = t.clone();
    m.insert(key.to_string(), Arc::downgrade(&erased));
    Ok(t)
}
