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

#[test]
fn unclosed_block_comments_parse_in_linear_time() {
    let script = format!("?[x] := x = 1 {}", "/* ".repeat(DEPTH));
    assert!(!parses_quickly(script));
    let nested = format!(
        "{}{} ?[x] <- [[1]]",
        "/* ".repeat(DEPTH),
        "*/ ".repeat(DEPTH)
    );
    assert!(parses_quickly(nested));
}

/// The block-comment rule as it was before the linear-time rewrite, with PEG
/// semantics (ordered choice, greedy repetition that never backtracks):
/// `"/*" ~ (BLOCK_COMMENT | !"*/" ~ ANY)* ~ "*/"`. Returns the end position.
fn reference_block_comment(s: &[u8], p: usize) -> Option<usize> {
    if !s[p..].starts_with(b"/*") {
        return None;
    }
    let mut i = p + 2;
    loop {
        if let Some(end) = reference_block_comment(s, i) {
            i = end;
        } else if i < s.len() && !s[i..].starts_with(b"*/") {
            i += 1;
        } else {
            break;
        }
    }
    s[i..].starts_with(b"*/").then_some(i + 2)
}

/// Whether `s` is entirely whitespace and block comments under the reference rule.
fn reference_is_trivia(s: &[u8]) -> bool {
    let mut i = 0;
    while i < s.len() {
        if s[i] == b' ' {
            i += 1;
        } else if let Some(end) = reference_block_comment(s, i) {
            i = end;
        } else {
            return false;
        }
    }
    true
}

/// The rewritten rule must accept exactly the comments the old one did:
/// compare against the reference on every string over `/ * a ␣` up to length 8.
#[test]
fn block_comments_accept_the_same_language_as_before() {
    let db = DbInstance::new("mem", "", "").unwrap();
    let alphabet = *b"/*a ";
    let mut checked = 0;
    for len in 0..=8u32 {
        for mut n in 0..4usize.pow(len) {
            let mut s = Vec::with_capacity(len as usize);
            for _ in 0..len {
                s.push(alphabet[n % 4]);
                n /= 4;
            }
            let text = String::from_utf8(s).unwrap();
            let script = format!("{text} ?[x] <- [[1]]");
            let parsed = db
                .run_script(&script, BTreeMap::new(), ScriptMutability::Immutable)
                .is_ok();
            assert_eq!(
                parsed,
                reference_is_trivia(text.as_bytes()),
                "comment text {text:?}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, (0..=8).map(|l| 4usize.pow(l)).sum::<usize>());
}
