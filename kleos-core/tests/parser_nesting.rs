/*
 * Copyright 2026, The kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Parse time must grow linearly with nesting depth. The grammar's
//! comma-separated rules were written `(x ~ ",")* ~ x?`, which parses the last
//! element twice (once in the failed loop iteration, again in `x?`); nested
//! lists, objects and calls doubled the work per level, so a 200-byte script
//! of unclosed `[` (found by fuzzing) hung the parser before any query budget
//! applied. At depth 40 the old grammar needs ~2^40 steps.

use std::collections::BTreeMap;
use std::sync::mpsc;
use std::time::Duration;

use kleos::{DbInstance, ScriptMutability};

const DEPTH: usize = 40;

/// Runs `script` on a worker thread; fails instead of hanging the suite.
fn parses_quickly(script: String) -> bool {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let db = DbInstance::new("mem", "", "").unwrap();
        let ok = db
            .run_script(&script, BTreeMap::new(), ScriptMutability::Immutable)
            .is_ok();
        let _ = tx.send(ok);
    });
    rx.recv_timeout(Duration::from_secs(10))
        .unwrap_or_else(|_| panic!("parsing did not finish within 10s"))
}

#[test]
fn nested_lists_parse_in_linear_time() {
    let closed = format!("?[x] <- [[{}1{}]]", "[".repeat(DEPTH), "]".repeat(DEPTH));
    assert!(parses_quickly(closed));
    let unclosed = format!("?[x] <- {}1", "[".repeat(DEPTH));
    assert!(!parses_quickly(unclosed));
}

#[test]
fn nested_calls_and_objects_parse_in_linear_time() {
    let calls = format!("?[x] := x = {}1{}", "abs(".repeat(DEPTH), ")".repeat(DEPTH));
    assert!(parses_quickly(calls));
    let objects = format!(
        "?[x] := x = {}1{}",
        "{'a': ".repeat(DEPTH),
        "}".repeat(DEPTH)
    );
    assert!(parses_quickly(objects));
}

#[test]
fn trailing_commas_are_still_accepted() {
    assert!(parses_quickly("?[x, y,] <- [[1, [2, 3,],],]".to_string()));
    assert!(parses_quickly(
        "?[x] := x = {'a': 1, 'b': [1,],}".to_string()
    ));
    assert!(parses_quickly("?[x] := x = max(1, 2,)".to_string()));
    assert!(!parses_quickly("?[x] <- [[1,,2]]".to_string()));
    assert!(!parses_quickly("?[x] <- [[,]]".to_string()));
}
