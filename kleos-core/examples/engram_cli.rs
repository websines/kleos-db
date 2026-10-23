/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Minimal bridge for scripts driving the Engram overlay store
//! (docs/specs/engram-overlay.md): runs one CozoScript against a SQLite kleos
//! database, with parameters from a JSON file, and prints the result as JSON.
//!
//! ```sh
//! cargo run --release -p kleos --features engram --example engram_cli -- \
//!     <db.sqlite> "::engram write m {ns: 'u1', fingerprint: \$fp, facts: \$facts, rows: \$rows}" params.json
//! ```
//!
//! `scripts/engram/engramedit.py` uses it to load row edits into kleos and to
//! materialise / export / erase overlays.

use std::collections::BTreeMap;

use kleos::{new_cozo_sqlite, DataValue, ScriptMutability};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: engram_cli <db.sqlite> <script> [params.json]");
        std::process::exit(2);
    }
    let db = new_cozo_sqlite(&args[1]).expect("open database");
    let params: BTreeMap<String, DataValue> = match args.get(3) {
        Some(p) => {
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).expect("params file")).expect("params JSON");
            v.as_object().expect("params must be a JSON object").iter().map(|(k, v)| (k.clone(), DataValue::from(v))).collect()
        }
        None => BTreeMap::new(),
    };
    let t = std::time::Instant::now();
    match db.run_script(&args[2], params, ScriptMutability::Mutable) {
        Ok(rows) => {
            let mut j = rows.into_json();
            j["took_ms"] = serde_json::json!(t.elapsed().as_secs_f64() * 1e3);
            println!("{j}");
        }
        Err(e) => {
            println!("{}", serde_json::json!({"ok": false, "error": format!("{e:?}")}));
            std::process::exit(1);
        }
    }
}
