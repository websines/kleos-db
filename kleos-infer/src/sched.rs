/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Scheduling of forward passes for the built-in models.
//!
//! * [`Priority`]: an interactive call (a `recall`, a chat `remember`) goes
//!   ahead of bulk work (document pages, trace batches). The priority rides
//!   on the calling thread ([`with_priority`]), so it reaches the model through
//!   any API without a new parameter.
//! * [`Gate`]: the per-model device lock, granted interactive-first but with a
//!   bulk turn after [`BULK_EVERY`] interactive grants in a row, so bulk work
//!   is never starved. Every grant also holds the activation bytes of its
//!   forward pass against the global [`crate::budget`].
//! * [`Batcher`]: cross-request batching for row models (embedders). Bulk
//!   gets at most [`BULK_SHARE`] of the device time while interactive calls
//!   are waiting (never less than that, so it always progresses), and its
//!   batches are cut to [`BULK_SLICE_TOKENS`] while interactive traffic is
//!   recent, so an interactive call never waits behind a long bulk pass.
//!   Callers
//!   queue their tokenised rows; one of the waiting callers at a time is the
//!   leader and runs batches of everybody's rows (no worker thread). Queues
//!   are bounded per lane in tokens, so a caller that would overfill them
//!   waits instead of piling up inputs.

use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::budget::{self, ActivationGuard};
use crate::{Error, Result};

/// Who is waiting on a forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Priority {
    /// Someone is waiting for the answer (recall, a chat remember).
    Interactive,
    /// Background work (document pages, trace batches, backfills).
    Bulk,
}

impl Priority {
    /// `interactive` or `bulk` (also `background`, `batch`).
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "interactive" | "high" => Priority::Interactive,
            "bulk" | "background" | "batch" | "low" => Priority::Bulk,
            o => return Err(Error::Config(format!("unknown priority '{o}' (interactive or bulk)"))),
        })
    }

    /// `interactive` or `bulk`.
    pub fn name(self) -> &'static str {
        match self {
            Priority::Interactive => "interactive",
            Priority::Bulk => "bulk",
        }
    }

    fn lane(self) -> usize {
        match self {
            Priority::Interactive => 0,
            Priority::Bulk => 1,
        }
    }
}

thread_local! {
    static PRIORITY: Cell<Priority> = const { Cell::new(Priority::Interactive) };
}

/// The priority of model calls made on this thread (default interactive).
pub fn current_priority() -> Priority {
    PRIORITY.with(|p| p.get())
}

/// Set this thread's priority; returns the previous one.
pub fn set_priority(p: Priority) -> Priority {
    PRIORITY.with(|c| c.replace(p))
}

/// Run `f` with model calls on this thread at priority `p`.
pub fn with_priority<R>(p: Priority, f: impl FnOnce() -> R) -> R {
    struct Restore(Priority);
    impl Drop for Restore {
        fn drop(&mut self) {
            PRIORITY.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(PRIORITY.with(|c| c.replace(p)));
    f()
}

/// Interactive grants in a row after which a waiting bulk request goes next.
pub const BULK_EVERY: usize = 4;

#[derive(Default)]
struct GateState {
    busy: bool,
    /// Next ticket to hand out and next ticket to serve, per lane.
    next: [u64; 2],
    serve: [u64; 2],
    streak: usize,
}

/// A priority-aware, starvation-free device lock (see the module docs).
#[derive(Default)]
pub struct Gate {
    st: Mutex<GateState>,
    cv: Condvar,
}

/// Holds the device (and the activation bytes) until dropped.
pub struct GateGuard<'a> {
    gate: &'a Gate,
    _act: ActivationGuard,
}

impl Drop for GateGuard<'_> {
    fn drop(&mut self) {
        let mut st = self.gate.st.lock().unwrap();
        st.busy = false;
        drop(st);
        self.gate.cv.notify_all();
    }
}

impl Gate {
    /// A free gate.
    pub fn new() -> Self {
        Self::default()
    }

    /// Wait for the device at `prio`, then reserve `act_bytes` of activations
    /// in the global budget.
    pub fn acquire(&self, prio: Priority, act_bytes: usize) -> GateGuard<'_> {
        let lane = prio.lane();
        let mut st = self.st.lock().unwrap();
        let ticket = st.next[lane];
        st.next[lane] += 1;
        loop {
            if !st.busy && st.serve[lane] == ticket {
                let waiting = |st: &GateState, l: usize| st.next[l] > st.serve[l];
                let other = 1 - lane;
                let mine = if lane == 0 {
                    !waiting(&st, other) || st.streak < BULK_EVERY
                } else {
                    !waiting(&st, other) || st.streak >= BULK_EVERY
                };
                if mine {
                    st.busy = true;
                    st.serve[lane] += 1;
                    st.streak = if lane == 0 { st.streak + 1 } else { 0 };
                    break;
                }
            }
            st = self.cv.wait(st).unwrap();
        }
        drop(st);
        // wake the next in line of the other lane's ticket check
        self.cv.notify_all();
        let act = budget::global().acquire_activations(act_bytes);
        GateGuard { gate: self, _act: act }
    }
}

/// Bulk's share of device time while interactive calls wait
/// (`KLEOS_BULK_SHARE`).
pub const BULK_SHARE: f64 = 0.25;
/// Largest bulk batch, in padded tokens, while interactive traffic is recent
/// (`KLEOS_BULK_SLICE_TOKENS`).
pub const BULK_SLICE_TOKENS: usize = 1024;
/// How long interactive traffic counts as recent.
const RECENT: Duration = Duration::from_secs(2);
/// Time constant of the device-time shares.
const SHARE_TAU: f64 = 10.0;

fn env_num<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Default per-lane queue bounds, in tokens (`KLEOS_QUEUE_TOKENS` sets the
/// bulk lane; the interactive lane gets half of it).
pub const QUEUE_TOKENS: usize = 65_536;

struct Item<R> {
    row: R,
    len: usize,
    slot: Arc<Slot<R>>,
    idx: usize,
}

struct Slot<R> {
    out: Mutex<SlotOut>,
    left: AtomicUsize,
    _p: std::marker::PhantomData<fn() -> R>,
}

#[derive(Default)]
struct SlotOut {
    vecs: Vec<Option<Vec<f32>>>,
    err: Option<String>,
}

impl<R> Slot<R> {
    fn done(&self) -> bool {
        self.left.load(Ordering::Acquire) == 0
    }
}

struct BatchState<R> {
    q: [VecDeque<Item<R>>; 2],
    queued: [usize; 2],
    leader: bool,
    /// Decayed device seconds per lane, and when they were last decayed.
    busy: [f64; 2],
    decayed: Instant,
    last_interactive: Option<Instant>,
}

impl<R> BatchState<R> {
    fn decay(&mut self) {
        let now = Instant::now();
        let k = (-(now - self.decayed).as_secs_f64() / SHARE_TAU).exp();
        self.busy[0] *= k;
        self.busy[1] *= k;
        self.decayed = now;
    }
}

/// What [`Batcher::run`] needs from a model.
pub struct BatchFns<'f, R> {
    /// Rows per forward for a padded length `l` (token and memory budgets).
    pub max_rows: &'f dyn Fn(usize) -> usize,
    /// Activation bytes of a forward over `n` rows of padded length `l`.
    pub act_bytes: &'f dyn Fn(usize, usize) -> usize,
    /// One forward pass over `rows`, one output per row.
    pub forward: &'f dyn Fn(&[&R]) -> Result<Vec<Vec<f32>>>,
    /// Called by a leader when it stops leading (e.g. to let the GPU
    /// allocator drop pooled buffers).
    pub after: &'f dyn Fn(),
    /// The model's device gate.
    pub gate: &'f Gate,
}

/// Cross-request batching of rows into vectors (see the module docs).
pub struct Batcher<R> {
    st: Mutex<BatchState<R>>,
    cv: Condvar,
    cap: [usize; 2],
    bulk_share: f64,
    slice: usize,
}

impl<R: AsRef<[u32]>> Default for Batcher<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: AsRef<[u32]>> Batcher<R> {
    /// Lanes bounded by [`QUEUE_TOKENS`] (or `KLEOS_QUEUE_TOKENS`).
    pub fn new() -> Self {
        let bulk = std::env::var("KLEOS_QUEUE_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(QUEUE_TOKENS)
            .max(1);
        Self::with_caps((bulk / 2).max(1), bulk)
    }

    /// Lanes bounded by `interactive` and `bulk` queued tokens.
    pub fn with_caps(interactive: usize, bulk: usize) -> Self {
        Self {
            st: Mutex::new(BatchState {
                q: [VecDeque::new(), VecDeque::new()],
                queued: [0, 0],
                leader: false,
                busy: [0.0, 0.0],
                decayed: Instant::now(),
                last_interactive: None,
            }),
            cv: Condvar::new(),
            cap: [interactive.max(1), bulk.max(1)],
            bulk_share: env_num("KLEOS_BULK_SHARE", BULK_SHARE).clamp(0.01, 1.0),
            slice: env_num("KLEOS_BULK_SLICE_TOKENS", BULK_SLICE_TOKENS).max(1),
        }
    }

    /// Queue `rows` at `prio` and return one vector per row, in order. The
    /// calling thread either waits or, as the leader, runs batches.
    pub fn run(&self, rows: Vec<R>, prio: Priority, f: &BatchFns<'_, R>) -> Result<Vec<Vec<f32>>> {
        let n = rows.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let slot: Arc<Slot<R>> = Arc::new(Slot {
            out: Mutex::new(SlotOut { vecs: (0..n).map(|_| None).collect(), err: None }),
            left: AtomicUsize::new(n),
            _p: std::marker::PhantomData,
        });
        let lane = prio.lane();
        // enqueue in pieces that fit the lane, waiting for room (backpressure)
        let mut rows = rows.into_iter().enumerate().peekable();
        while rows.peek().is_some() {
            let mut st = self.st.lock().unwrap();
            while st.queued[lane] > 0 && st.queued[lane] + rows.peek().map(|(_, r)| r.as_ref().len()).unwrap_or(0) > self.cap[lane] {
                if !st.leader {
                    break; // nobody is draining: lead (below) instead of waiting
                }
                st = self.cv.wait(st).unwrap();
            }
            while let Some((_, r)) = rows.peek() {
                let len = r.as_ref().len().max(1);
                if st.queued[lane] > 0 && st.queued[lane] + len > self.cap[lane] {
                    break;
                }
                let (idx, row) = rows.next().unwrap();
                st.queued[lane] += len;
                st.q[lane].push_back(Item { row, len, slot: slot.clone(), idx });
                if lane == 0 {
                    st.last_interactive = Some(Instant::now());
                }
            }
            if rows.peek().is_some() && !st.leader {
                st.leader = true;
                drop(st);
                self.lead(Some(&slot), f, true);
            }
        }
        loop {
            let mut st = self.st.lock().unwrap();
            loop {
                if slot.done() {
                    drop(st);
                    return Self::take(&slot);
                }
                if !st.leader {
                    st.leader = true;
                    break;
                }
                st = self.cv.wait(st).unwrap();
            }
            drop(st);
            self.lead(Some(&slot), f, false);
        }
    }

    /// Run batches while leading; stop when `mine` is done (or, with `once`,
    /// after one batch) or the queues are empty, then hand leadership on.
    fn lead(&self, mine: Option<&Arc<Slot<R>>>, f: &BatchFns<'_, R>, once: bool) {
        loop {
            let (batch, lane) = {
                let mut st = self.st.lock().unwrap();
                match self.pick(&mut st, f) {
                    Some(b) => b,
                    None => break,
                }
            };
            self.cv.notify_all(); // room in the lanes
            let l = batch.iter().map(|i| i.len).max().unwrap_or(1);
            let prio = if lane == 0 { Priority::Interactive } else { Priority::Bulk };
            if std::env::var_os("KLEOS_SCHED_DEBUG").is_some() {
                eprintln!("sched: lane {lane} rows {} len {l}", batch.len());
            }
            let t = Instant::now();
            let res = {
                let _g = f.gate.acquire(prio, (f.act_bytes)(batch.len(), l));
                let refs: Vec<&R> = batch.iter().map(|i| &i.row).collect();
                (f.forward)(&refs)
            };
            {
                let mut st = self.st.lock().unwrap();
                st.decay();
                st.busy[lane] += t.elapsed().as_secs_f64();
            }
            match res {
                Ok(vecs) if vecs.len() == batch.len() => {
                    for (item, v) in batch.into_iter().zip(vecs) {
                        item.slot.out.lock().unwrap().vecs[item.idx] = Some(v);
                        item.slot.left.fetch_sub(1, Ordering::AcqRel);
                    }
                }
                other => {
                    let msg = match other {
                        Err(e) => e.to_string(),
                        Ok(v) => format!("forward returned {} vectors for {} rows", v.len(), batch.len()),
                    };
                    for item in batch {
                        let mut o = item.slot.out.lock().unwrap();
                        o.err.get_or_insert_with(|| msg.clone());
                        drop(o);
                        item.slot.left.fetch_sub(1, Ordering::AcqRel);
                    }
                }
            }
            self.cv.notify_all();
            if once || mine.map(|s| s.done()).unwrap_or(false) {
                break;
            }
        }
        (f.after)();
        let mut st = self.st.lock().unwrap();
        st.leader = false;
        drop(st);
        self.cv.notify_all();
    }

    /// The next batch: interactive rows first unless bulk is below its
    /// share of device time; rows in queue order while `rows × padded
    /// length` fits the model's budgets (and, for bulk while interactive
    /// traffic is recent, the slice).
    fn pick(&self, st: &mut BatchState<R>, f: &BatchFns<'_, R>) -> Option<(Vec<Item<R>>, usize)> {
        st.decay();
        let lane = match (st.q[0].is_empty(), st.q[1].is_empty()) {
            (true, true) => return None,
            (false, true) => 0,
            (true, false) => 1,
            (false, false) => {
                let total = st.busy[0] + st.busy[1];
                if total <= 0.0 || st.busy[1] / total < self.bulk_share {
                    1
                } else {
                    0
                }
            }
        };
        let sliced = lane == 1 && st.last_interactive.map(|t| t.elapsed() < RECENT).unwrap_or(false);
        let mut out: Vec<Item<R>> = vec![];
        let mut l = 0;
        while let Some(front) = st.q[lane].front() {
            let nl = l.max(front.len);
            let mut rows = (f.max_rows)(nl).max(1);
            if sliced {
                rows = rows.min((self.slice / nl.max(1)).max(1));
            }
            if !out.is_empty() && out.len() + 1 > rows {
                break;
            }
            l = nl;
            let it = st.q[lane].pop_front().unwrap();
            st.queued[lane] -= it.len;
            out.push(it);
        }
        Some((out, lane))
    }

    fn take(slot: &Slot<R>) -> Result<Vec<Vec<f32>>> {
        let mut o = slot.out.lock().unwrap();
        if let Some(e) = o.err.take() {
            return Err(Error::Model(e));
        }
        Ok(std::mem::take(&mut o.vecs).into_iter().map(|v| v.unwrap_or_default()).collect())
    }
}
