/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Scheduling hints for the built-in models (docs/specs/memory-api.md,
//! "Concurrency and memory budget"). With the `infer` feature this is
//! `kleos_infer::sched`: calls at [`Priority::Interactive`] (the default) go
//! ahead of [`Priority::Bulk`] ones, which are never starved. Without it the
//! hints are accepted and have no effect.
//!
//! ```no_run
//! use kleos::sched::{with_priority, Priority};
//! // background ingestion: queued behind agents' interactive calls
//! with_priority(Priority::Bulk, || { /* remember document chunks, traces, ... */ });
//! ```

use miette::{miette, Result};

#[cfg(feature = "infer")]
pub use kleos_infer::sched::{current_priority, with_priority, Priority};

/// Who is waiting on a model call (the `infer` feature schedules on it).
#[cfg(not(feature = "infer"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Priority {
    /// Someone is waiting for the answer.
    Interactive,
    /// Background work.
    Bulk,
}

#[cfg(not(feature = "infer"))]
impl Priority {
    /// `interactive` or `bulk`.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "interactive" | "high" => Ok(Priority::Interactive),
            "bulk" | "background" | "batch" | "low" => Ok(Priority::Bulk),
            o => Err(format!("unknown priority '{o}' (interactive or bulk)")),
        }
    }
}

/// Run `f` with model calls on this thread at priority `p` (no effect
/// without the `infer` feature).
#[cfg(not(feature = "infer"))]
pub fn with_priority<R>(_p: Priority, f: impl FnOnce() -> R) -> R {
    f()
}

/// The priority of model calls on this thread.
#[cfg(not(feature = "infer"))]
pub fn current_priority() -> Priority {
    Priority::Interactive
}

/// Sets this thread's priority until dropped (restoring the previous one).
pub struct Scope {
    #[cfg(feature = "infer")]
    prev: Option<Priority>,
}

#[cfg(feature = "infer")]
impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(p) = self.prev {
            kleos_infer::sched::set_priority(p);
        }
    }
}

/// Apply an optional `interactive` / `bulk` hint for the rest of a call.
pub fn scope(hint: Option<&str>) -> Result<Scope> {
    let Some(h) = hint else {
        #[cfg(feature = "infer")]
        return Ok(Scope { prev: None });
        #[cfg(not(feature = "infer"))]
        return Ok(Scope {});
    };
    let p = Priority::parse(h).map_err(|e| miette!("{e}"))?;
    #[cfg(feature = "infer")]
    {
        let prev = kleos_infer::sched::set_priority(p);
        Ok(Scope { prev: Some(prev) })
    }
    #[cfg(not(feature = "infer"))]
    {
        let _ = p;
        Ok(Scope {})
    }
}
