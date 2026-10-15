/*
 * Copyright 2026, kleos authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Live query subscriptions (kleos, UPGRADE_PLAN F1).
//!
//! A subscription is a read-only query whose result is pushed to the
//! subscriber: first the whole result (`Initial`), then a row-level diff
//! (`Delta`) every time a committed transaction writes a stored relation the
//! query read. Spec: `docs/specs/live-subscriptions.md`.
//!
//! Moving parts:
//!
//! * [`LiveHub`] lives in every `Db` and every `SessionTx`. The commit funnel
//!   (`SessionTx::commit_tx`) hands it the transaction's dirty relation ids
//!   **after** a successful commit ([`LiveHub::notify_commit`]). The writer only
//!   filters the ids against the watched set, records them as pending and wakes
//!   the worker. It never evaluates a query and never blocks on a subscriber.
//! * One worker thread per `Db` (spawned on the first subscription, exiting
//!   with the last) drains the pending set and re-evaluates every group whose
//!   dependency set intersects it. Several commits that land while the worker
//!   is busy are coalesced into one re-evaluation.
//! * Identical `(script, params)` subscriptions share one [`Group`]: one
//!   evaluation, one snapshot, one diff, fanned out to every subscriber.
//! * Each subscriber has a bounded queue. A full queue never blocks the
//!   worker: the subscriber is marked lagging, further deltas are skipped, and
//!   its next `recv` discards what is queued and returns a `Resync` with the
//!   group's latest snapshot.
//! * Dependencies are recorded, not inferred: every stored relation the
//!   evaluation resolves through `SessionTx::get_relation` (rules, fixed-rule
//!   inputs, HNSW/FTS/LSH searches, stored queries) is logged, and an index
//!   relation read directly (`*rel:idx`) maps to its base relation. They are
//!   recomputed on every evaluation, so a `:replace`d relation (new id) is
//!   followed.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

use crossbeam::channel::{bounded, Receiver, RecvTimeoutError, Sender, TrySendError};
use smartstring::{LazyCompact, SmartString};

use crate::data::functions::current_validity;
use crate::data::tuple::Tuple;
use crate::data::value::DataValue;
use crate::parse::{parse_script, CozoScript};
use crate::runtime::relation::RelationId;
use crate::storage::Storage;
use crate::{bail, Db, NamedRows};

/// Relations resolved by name during one evaluation (`name -> id`). Filled by
/// `SessionTx::get_relation` while a subscription query runs.
pub(crate) type ReadLog = Arc<Mutex<BTreeMap<SmartString<LazyCompact>, RelationId>>>;

/// The dedup key of a subscription: script text plus parameters.
type GroupKey = (String, BTreeMap<String, DataValue>);

/// What the worker uses to re-run a subscription query: the rows plus the
/// stored relations the evaluation read.
pub(crate) type LiveEvaluator = Box<
    dyn Fn(&str, &BTreeMap<String, DataValue>) -> miette::Result<(NamedRows, BTreeSet<RelationId>)>
        + Send,
>;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panic while holding one of these locks leaves at worst a stale
    // snapshot or a stale watch count (over-notification): degraded, never
    // a wrong delta for a healthy group. Recover instead of cascading.
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Options for [`crate::Db::subscribe`].
#[derive(Clone, Debug)]
pub struct SubscribeOptions {
    /// Per-subscriber queue capacity, in events (minimum 1, default 64). When
    /// the queue is full the subscriber is marked lagging: further deltas are
    /// dropped and its next receive returns a
    /// [`SubscriptionEvent::Resync`] with the latest result instead.
    pub queue_capacity: usize,
}

impl Default for SubscribeOptions {
    fn default() -> Self {
        Self { queue_capacity: 64 }
    }
}

/// One event delivered to a [`Subscription`].
///
/// Row payloads are shared (`Arc`) between every subscriber of the same
/// `(script, params)`.
#[derive(Clone, Debug)]
pub enum SubscriptionEvent {
    /// The query result when the subscription was created. Always the first
    /// event.
    Initial {
        /// The full result, in the query's own order.
        rows: Arc<NamedRows>,
    },
    /// The result changed after a committed write. Apply `removed`, then
    /// `added`, to the previous result (treated as a multiset of rows). Rows
    /// within each side are sorted by value, not by the query's `:order`.
    Delta {
        /// Rows now in the result that were not before.
        added: Arc<NamedRows>,
        /// Rows no longer in the result.
        removed: Arc<NamedRows>,
    },
    /// The subscriber fell behind (its queue filled up) or the previous
    /// evaluation failed. Intermediate deltas were dropped; this is the full,
    /// current result. Replace local state with it.
    Resync {
        /// The full result, in the query's own order.
        rows: Arc<NamedRows>,
    },
    /// Re-evaluation after a commit failed (for example a relation the query
    /// reads was removed, or the query hit its time or memory budget). The
    /// subscription stays registered; the next successful evaluation is
    /// delivered as a `Resync`.
    Error {
        /// The evaluation error, rendered as text.
        message: String,
    },
}

/// Counters for observing the subscription engine (tests, benchmarks).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LiveStats {
    /// Live subscription handles.
    pub subscriptions: usize,
    /// Distinct `(script, params)` groups; each is evaluated once per change.
    pub groups: usize,
    /// Query evaluations performed so far (initial ones included).
    pub evaluations: u64,
    /// Committed transactions that touched at least one watched relation.
    pub notified_commits: u64,
}

struct SubSlot {
    sender: Sender<SubscriptionEvent>,
    lagging: bool,
}

struct GroupState {
    /// Initial evaluation finished successfully.
    ready: bool,
    /// Removed from the registry (last subscriber gone, or initial eval failed).
    dead: bool,
    /// The previous evaluation failed: the next success is sent as a Resync.
    errored: bool,
    snapshot: Arc<NamedRows>,
    /// `snapshot.rows`, sorted, for the multiset diff.
    sorted: Vec<Tuple>,
    /// Stored relations the last evaluation read. `None` until the initial
    /// evaluation finishes; the group is then counted as a wildcard watcher.
    deps: Option<BTreeSet<RelationId>>,
    subscribers: BTreeMap<u64, SubSlot>,
}

pub(crate) struct Group {
    key: GroupKey,
    /// Serialises evaluations of this group (initial vs. worker), so a delta
    /// is always computed against the snapshot that preceded it.
    eval: Mutex<()>,
    state: Mutex<GroupState>,
}

#[derive(Default)]
struct Registry {
    groups: BTreeMap<GroupKey, Arc<Group>>,
    subs: BTreeMap<u64, Arc<Group>>,
    worker_running: bool,
}

#[derive(Default)]
struct Signal {
    /// relation id -> number of groups that read it.
    watched: BTreeMap<RelationId, usize>,
    /// Groups whose dependencies are not known yet; while non-zero every
    /// committed relation is recorded as pending.
    watch_all: usize,
    /// Committed, watched relations the worker has not looked at yet.
    pending: BTreeSet<RelationId>,
}

/// Per-`Db` subscription registry and commit notification point.
#[derive(Default)]
pub(crate) struct LiveHub {
    /// Number of groups. Lets the commit hook return without a lock when
    /// nobody subscribes.
    active: AtomicUsize,
    next_id: AtomicU64,
    evaluations: AtomicU64,
    notified_commits: AtomicU64,
    registry: Mutex<Registry>,
    signal: Mutex<Signal>,
    cond: Condvar,
}

impl LiveHub {
    /// Called from the commit funnel after a **successful** commit, with the
    /// transaction's dirty (non-temp) relation ids. Cheap: one atomic load when
    /// nobody subscribes, otherwise one short lock and a condvar notify.
    pub(crate) fn notify_commit(&self, dirty: &BTreeSet<RelationId>) {
        if dirty.is_empty() || self.active.load(Ordering::Acquire) == 0 {
            return;
        }
        let mut sig = lock(&self.signal);
        let mut any = false;
        for rel in dirty {
            if sig.watch_all > 0 || sig.watched.contains_key(rel) {
                sig.pending.insert(*rel);
                any = true;
            }
        }
        drop(sig);
        if any {
            self.notified_commits.fetch_add(1, Ordering::Relaxed);
            self.cond.notify_one();
        }
    }

    pub(crate) fn stats(&self) -> LiveStats {
        let reg = lock(&self.registry);
        LiveStats {
            subscriptions: reg.subs.len(),
            groups: reg.groups.len(),
            evaluations: self.evaluations.load(Ordering::Relaxed),
            notified_commits: self.notified_commits.load(Ordering::Relaxed),
        }
    }

    fn add_watch(&self, deps: &BTreeSet<RelationId>) {
        let mut sig = lock(&self.signal);
        for d in deps {
            *sig.watched.entry(*d).or_default() += 1;
        }
    }

    fn remove_watch(&self, deps: &BTreeSet<RelationId>) {
        let mut sig = lock(&self.signal);
        for d in deps {
            if let Some(n) = sig.watched.get_mut(d) {
                *n -= 1;
                if *n == 0 {
                    sig.watched.remove(d);
                }
            }
        }
    }

    /// Stop counting a group's dependencies (`None` = it watched everything).
    fn release_deps(&self, deps: &Option<BTreeSet<RelationId>>) {
        match deps {
            Some(d) => self.remove_watch(d),
            None => self.change_watch_all(false),
        }
    }

    /// Start counting a group's dependencies (`None` = watch everything).
    fn acquire_deps(&self, deps: &Option<BTreeSet<RelationId>>) {
        match deps {
            Some(d) => self.add_watch(d),
            None => self.change_watch_all(true),
        }
    }

    fn change_watch_all(&self, inc: bool) {
        let mut sig = lock(&self.signal);
        if inc {
            sig.watch_all += 1;
        } else {
            sig.watch_all = sig.watch_all.saturating_sub(1);
        }
    }

    /// Register a subscriber for `(script, params)`. `evaluate` runs the query
    /// on the caller's thread when no group exists yet. Returns the
    /// subscription and whether a worker must be spawned (the caller owns the
    /// `Db` clone the worker needs).
    pub(crate) fn subscribe(
        self: &Arc<Self>,
        script: &str,
        params: BTreeMap<String, DataValue>,
        opts: &SubscribeOptions,
        evaluate: &dyn Fn(
            &str,
            &BTreeMap<String, DataValue>,
        ) -> miette::Result<(NamedRows, BTreeSet<RelationId>)>,
    ) -> miette::Result<(Subscription, bool)> {
        let key: GroupKey = (script.to_string(), params);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = bounded(opts.queue_capacity.max(1));
        loop {
            // Find or create the group.
            let (group, created) = {
                let mut reg = lock(&self.registry);
                match reg.groups.get(&key) {
                    Some(g) => (g.clone(), false),
                    None => {
                        let g = Arc::new(Group {
                            key: key.clone(),
                            eval: Mutex::new(()),
                            state: Mutex::new(GroupState {
                                ready: false,
                                dead: false,
                                errored: false,
                                snapshot: Arc::new(NamedRows::default()),
                                sorted: vec![],
                                deps: None,
                                subscribers: BTreeMap::new(),
                            }),
                        });
                        reg.groups.insert(key.clone(), g.clone());
                        self.active.store(reg.groups.len(), Ordering::Release);
                        // Unknown deps: watch everything until the initial
                        // evaluation says what it read, so a commit racing it
                        // is never missed.
                        self.change_watch_all(true);
                        (g, true)
                    }
                }
            };

            let eval_guard = lock(&group.eval);
            if created {
                let res = evaluate(&key.0, &key.1);
                self.evaluations.fetch_add(1, Ordering::Relaxed);
                match res {
                    Ok((rows, deps)) => {
                        let mut st = lock(&group.state);
                        self.add_watch(&deps);
                        self.change_watch_all(false);
                        let mut sorted = rows.rows.clone();
                        sorted.sort();
                        st.sorted = sorted;
                        st.snapshot = Arc::new(rows);
                        st.deps = Some(deps);
                        st.ready = true;
                    }
                    Err(err) => {
                        {
                            let mut st = lock(&group.state);
                            st.dead = true;
                        }
                        self.change_watch_all(false);
                        drop(eval_guard);
                        let mut reg = lock(&self.registry);
                        if let Some(g) = reg.groups.get(&key) {
                            if Arc::ptr_eq(g, &group) {
                                reg.groups.remove(&key);
                            }
                        }
                        self.active.store(reg.groups.len(), Ordering::Release);
                        return Err(err);
                    }
                }
            }
            let mut st = lock(&group.state);
            if st.dead || !st.ready {
                // The group died between lookup and join (its last subscriber
                // left, or the creator's initial evaluation failed): retry,
                // which creates a fresh group and evaluates it here.
                drop(st);
                drop(eval_guard);
                continue;
            }
            // Initial event: the group's latest snapshot. Later deltas are
            // computed against exactly this snapshot (both under `state`).
            let _ = sender.try_send(SubscriptionEvent::Initial {
                rows: st.snapshot.clone(),
            });
            st.subscribers.insert(
                id,
                SubSlot {
                    sender: sender.clone(),
                    lagging: false,
                },
            );
            drop(st);
            drop(eval_guard);

            let mut reg = lock(&self.registry);
            reg.subs.insert(id, group.clone());
            let spawn = !reg.worker_running;
            reg.worker_running = true;
            drop(reg);
            drop(sender);
            return Ok((
                Subscription {
                    id,
                    receiver,
                    group,
                    hub: self.clone(),
                },
                spawn,
            ));
        }
    }

    /// Remove one subscriber; drops its group when it was the last one.
    pub(crate) fn unsubscribe(&self, id: u64) -> bool {
        let mut reg = lock(&self.registry);
        let Some(group) = reg.subs.remove(&id) else {
            return false;
        };
        let mut st = lock(&group.state);
        st.subscribers.remove(&id);
        if st.subscribers.is_empty() && !st.dead {
            st.dead = true;
            self.release_deps(&st.deps);
            reg.groups.remove(&group.key);
            self.active.store(reg.groups.len(), Ordering::Release);
        }
        drop(st);
        let empty = reg.groups.is_empty();
        drop(reg);
        if empty {
            // Let the worker notice and exit promptly.
            self.cond.notify_all();
        }
        true
    }

    /// The worker loop. Exits (and drops `evaluate`, which owns a `Db`
    /// clone) once no group is left.
    pub(crate) fn run_worker(self: Arc<Self>, evaluate: LiveEvaluator) {
        loop {
            {
                let mut reg = lock(&self.registry);
                if reg.groups.is_empty() {
                    reg.worker_running = false;
                    return;
                }
            }
            let dirty = {
                let mut sig = lock(&self.signal);
                if sig.pending.is_empty() {
                    sig = self
                        .cond
                        .wait_timeout(sig, Duration::from_millis(100))
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .0;
                }
                std::mem::take(&mut sig.pending)
            };
            if dirty.is_empty() {
                continue;
            }
            let groups: Vec<Arc<Group>> = {
                let reg = lock(&self.registry);
                reg.groups.values().cloned().collect()
            };
            for g in groups {
                let affected = {
                    let st = lock(&g.state);
                    if st.dead {
                        false
                    } else {
                        match &st.deps {
                            // Initial evaluation still running on the
                            // subscriber's thread, possibly on a snapshot
                            // older than this commit: re-evaluate once it is
                            // done (`reevaluate` waits on the eval lock).
                            None => true,
                            Some(deps) => dirty.iter().any(|d| deps.contains(d)),
                        }
                    }
                };
                if affected {
                    self.reevaluate(&g, &evaluate);
                }
            }
        }
    }

    fn reevaluate(&self, g: &Arc<Group>, evaluate: &LiveEvaluator) {
        let _eval = lock(&g.eval);
        {
            let st = lock(&g.state);
            if st.dead || !st.ready {
                return;
            }
        }
        let res = evaluate(&g.key.0, &g.key.1);
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        let mut st = lock(&g.state);
        if st.dead {
            return;
        }
        let event = match res {
            Ok((rows, deps)) => {
                if st.deps.as_ref() != Some(&deps) {
                    let new = Some(deps);
                    self.acquire_deps(&new);
                    self.release_deps(&st.deps);
                    st.deps = new;
                }
                let mut sorted = rows.rows.clone();
                sorted.sort();
                let (added, removed) = diff_sorted(&st.sorted, &sorted);
                let headers = rows.headers.clone();
                st.sorted = sorted;
                st.snapshot = Arc::new(rows);
                if st.errored {
                    st.errored = false;
                    SubscriptionEvent::Resync {
                        rows: st.snapshot.clone(),
                    }
                } else if added.is_empty() && removed.is_empty() {
                    return;
                } else {
                    SubscriptionEvent::Delta {
                        added: Arc::new(NamedRows::new(headers.clone(), added)),
                        removed: Arc::new(NamedRows::new(headers, removed)),
                    }
                }
            }
            Err(err) => {
                // What the query reads is unknown now (a relation may have
                // been removed and is about to be re-created under a new
                // id): watch every commit until an evaluation succeeds.
                if st.deps.is_some() {
                    self.acquire_deps(&None);
                    self.release_deps(&st.deps);
                    st.deps = None;
                }
                st.errored = true;
                SubscriptionEvent::Error {
                    message: format!("{err:?}"),
                }
            }
        };
        for slot in st.subscribers.values_mut() {
            if slot.lagging {
                continue;
            }
            match slot.sender.try_send(event.clone()) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => slot.lagging = true,
                // The handle was dropped; its Drop unsubscribes.
                Err(TrySendError::Disconnected(_)) => {}
            }
        }
    }
}

/// Multiset difference of two sorted row lists: `(added, removed)`.
fn diff_sorted(old: &[Tuple], new: &[Tuple]) -> (Vec<Tuple>, Vec<Tuple>) {
    let (mut i, mut j) = (0, 0);
    let mut added = vec![];
    let mut removed = vec![];
    while i < old.len() && j < new.len() {
        match old[i].cmp(&new[j]) {
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                removed.push(old[i].clone());
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                added.push(new[j].clone());
                j += 1;
            }
        }
    }
    removed.extend_from_slice(&old[i..]);
    added.extend_from_slice(&new[j..]);
    (added, removed)
}

/// A live query subscription. Receive events with [`Subscription::recv`],
/// [`Subscription::try_recv`] or [`Subscription::recv_timeout`]. Dropping the
/// handle (or calling [`Subscription::unsubscribe`]) unsubscribes.
pub struct Subscription {
    id: u64,
    receiver: Receiver<SubscriptionEvent>,
    group: Arc<Group>,
    hub: Arc<LiveHub>,
}

impl std::fmt::Debug for Subscription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Subscription").field("id", &self.id).finish()
    }
}

impl Subscription {
    /// This subscription's id (unique per `Db`), for [`crate::Db::unsubscribe`].
    pub fn id(&self) -> u64 {
        self.id
    }

    /// If the worker marked us lagging, drop what is queued and hand back the
    /// latest snapshot. Done under the group's state lock, so no delta can be
    /// sent between reading the snapshot and clearing the flag.
    fn take_resync(&self) -> Option<SubscriptionEvent> {
        let mut st = lock(&self.group.state);
        let slot = st.subscribers.get_mut(&self.id)?;
        if !slot.lagging {
            return None;
        }
        slot.lagging = false;
        while self.receiver.try_recv().is_ok() {}
        Some(SubscriptionEvent::Resync {
            rows: st.snapshot.clone(),
        })
    }

    /// Block until the next event. `None` once the subscription is closed
    /// (unsubscribed by id from elsewhere).
    pub fn recv(&self) -> Option<SubscriptionEvent> {
        if let Some(ev) = self.take_resync() {
            return Some(ev);
        }
        self.receiver.recv().ok()
    }

    /// The next event if one is ready.
    pub fn try_recv(&self) -> Option<SubscriptionEvent> {
        if let Some(ev) = self.take_resync() {
            return Some(ev);
        }
        self.receiver.try_recv().ok()
    }

    /// Wait up to `timeout` for the next event.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<SubscriptionEvent> {
        if let Some(ev) = self.take_resync() {
            return Some(ev);
        }
        match self.receiver.recv_timeout(timeout) {
            Ok(ev) => Some(ev),
            Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => None,
        }
    }

    /// Unsubscribe now (the same as dropping the handle).
    pub fn unsubscribe(self) {}
}

impl Drop for Subscription {
    fn drop(&mut self) {
        self.hub.unsubscribe(self.id);
    }
}

impl<'s, S: Storage<'s>> Db<S> {
    /// Evaluate a subscription query on a fresh read transaction and return
    /// its rows plus every stored relation it read.
    pub(crate) fn live_evaluate(
        &'s self,
        script: &str,
        params: &BTreeMap<String, DataValue>,
    ) -> miette::Result<(NamedRows, BTreeSet<RelationId>)> {
        let cur_vld = current_validity();
        let parsed = parse_script(
            script,
            params,
            &self.get_fixed_rules(),
            crate::data::aggr::CustomAggrRegistries {
                meet: &self.get_custom_aggrs(),
                bounded: &self.get_custom_bounded_meets(),
            },
            cur_vld,
        );
        let res = (|| {
            let p = match parsed? {
                CozoScript::Single(p) => p,
                CozoScript::Imperative(_) => bail!(
                    "live subscriptions take a single read-only query; \
                     imperative scripts (`%if`, `%loop`, chained `{{}}` blocks) cannot be subscribed"
                ),
                CozoScript::Sys(_) => {
                    bail!("live subscriptions take a read-only query; system ops (`::…`) cannot be subscribed")
                }
            };
            if let Some((h, op, _)) = &p.out_opts.store_relation {
                bail!(
                    "live subscriptions take a read-only query, but this one writes `{}` ({op:?}); \
                     remove the `:put`/`:rm`/`:create`/… option",
                    h.name.name
                );
            }
            let mut tx = self.transact()?;
            tx.script_deadline = self.effective_outer_deadline(None);
            tx.script_mem_limit = self.effective_outer_mem_limit(None);
            let log: ReadLog = Default::default();
            tx.read_log = Some(log.clone());
            let mut cleanups = vec![];
            let mut collector = BTreeMap::new();
            let rows = self.execute_single_program(
                p,
                &mut tx,
                &mut cleanups,
                cur_vld,
                &BTreeSet::new(),
                &mut collector,
            )?;
            // An index read directly (`*rel:idx`) changes when its base
            // relation is written, and only the base is marked dirty.
            let names: Vec<SmartString<LazyCompact>> = lock(&log).keys().cloned().collect();
            for name in names {
                if let Some((base, _)) = name.split_once(':') {
                    let _ = tx.get_relation(base, false);
                }
            }
            tx.read_log = None;
            tx.commit_tx()?;
            let deps = lock(&log).values().copied().collect();
            Ok((rows, deps))
        })();
        self.flush_warnings();
        res
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<S> Db<S>
where
    S: for<'a> Storage<'a> + 'static,
{
    /// Subscribe to a read-only query (kleos, UPGRADE_PLAN F1; spec
    /// `docs/specs/live-subscriptions.md`).
    ///
    /// The first event is [`SubscriptionEvent::Initial`] with the current
    /// result. After that, whenever a **committed** transaction writes a
    /// stored relation the query read (directly, through rules, as a fixed
    /// rule's input, or under an HNSW/FTS/LSH search), the query is
    /// re-evaluated on a background worker and the row-level difference is
    /// delivered as [`SubscriptionEvent::Delta`]. Identical `(script, params)`
    /// subscriptions share one evaluation. Writers never wait for
    /// subscribers: a subscriber whose queue is full gets a
    /// [`SubscriptionEvent::Resync`] instead of the deltas it missed.
    ///
    /// `NOW` / `'NOW'` validity is read at each re-evaluation; a result that
    /// changes only because time passes is not pushed.
    ///
    /// Errors if the script does not parse, is not a single read-only query
    /// (mutations, `::` system ops and imperative scripts are rejected), or
    /// fails on its first evaluation.
    pub fn subscribe(
        &self,
        script: &str,
        params: BTreeMap<String, DataValue>,
        opts: SubscribeOptions,
    ) -> miette::Result<Subscription> {
        let (sub, spawn) = self
            .live
            .subscribe(script, params, &opts, &|s, p| self.live_evaluate(s, p))?;
        if spawn {
            let db = self.clone();
            let hub = self.live.clone();
            let evaluate: LiveEvaluator = Box::new(move |s, p| db.live_evaluate(s, p));
            std::thread::Builder::new()
                .name("kleos-live".into())
                .spawn(move || hub.run_worker(evaluate))
                .map_err(|e| miette::miette!("cannot spawn the live-subscription worker: {e}"))?;
        }
        Ok(sub)
    }

    /// Unsubscribe by id. The handle's receive methods return `None` from
    /// then on. Returns `false` if no such subscription exists.
    pub fn unsubscribe(&self, id: u64) -> bool {
        self.live.unsubscribe(id)
    }

    /// Counters for the live-subscription engine.
    pub fn live_stats(&self) -> LiveStats {
        self.live.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_is_a_multiset_difference() {
        let t = |x: i64| vec![DataValue::from(x)];
        let (a, r) = diff_sorted(&[t(1), t(2), t(2), t(4)], &[t(2), t(3), t(4), t(5)]);
        assert_eq!(a, vec![t(3), t(5)]);
        assert_eq!(r, vec![t(1), t(2)]);
    }
}
