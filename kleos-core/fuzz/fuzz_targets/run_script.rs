/*
 * Copyright 2026, The kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Runs arbitrary CozoScript against a fresh in-memory database.
//!
//! The contract under test: for any input, `run_script` returns `Ok` or `Err`.
//! A panic, abort, hang past the query budget, or runaway allocation is a bug.

#![no_main]

use std::collections::BTreeMap;

use kleos::{DbInstance, ScriptMutability};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(script) = std::str::from_utf8(data) else {
        return;
    };
    let db = DbInstance::new("mem", "", "").unwrap();
    // Divergent Datalog is legal; the engine's own budgets must stop it. A
    // libFuzzer timeout or OOM therefore points at a path that ignores them.
    db.set_default_query_timeout(Some(1.0));
    db.set_default_query_mem_limit(Some(256 << 20));
    let _ = db.run_script(script, BTreeMap::new(), ScriptMutability::Mutable);
});
