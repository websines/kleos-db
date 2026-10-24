//! Cost of rules with many `column = constant` atoms against a constant row
//! (`cargo run --release -p kleos --example planner_consts`). Planning grows
//! a little faster than linearly with the atom count but stays well under a
//! millisecond at 24 columns.
use std::collections::BTreeMap;
use std::time::Instant;

use kleos::{DbInstance, ScriptMutability};

fn time(db: &DbInstance, script: &str, reps: usize) -> f64 {
    timep(db, script, BTreeMap::new(), reps)
}

fn timep(db: &DbInstance, script: &str, params: BTreeMap<String, kleos::DataValue>, reps: usize) -> f64 {
    let mut v = vec![];
    for _ in 0..reps {
        let t = Instant::now();
        db.run_script(script, params.clone(), ScriptMutability::Mutable).unwrap();
        v.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let db = DbInstance::new("mem", "", "").unwrap();
    for n in [4usize, 8, 12, 16, 20, 24] {
        let cols: Vec<String> = (0..n).map(|i| format!("c{i}")).collect();
        let rel = format!("r{n}");
        db.run_script(
            &format!(":create {rel} {{{}: Int => {}}}", cols[0], cols[1..].iter().map(|c| format!("{c}: Any")).collect::<Vec<_>>().join(", ")),
            BTreeMap::new(),
            ScriptMutability::Mutable,
        )
        .unwrap();
        let head = cols.join(", ");
        let spec = format!("{} => {}", cols[0], cols[1..].join(", "));
        let atoms: Vec<String> = cols.iter().enumerate().map(|(i, c)| format!("{c} = {i}")).collect();
        let rule = format!("?[{head}] := {} :put {rel} {{{spec}}}", atoms.join(", "));
        let vals: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        let row = format!("?[{head}] <- [[{}]] :put {rel} {{{spec}}}", vals.join(", "));
        let q = format!("?[{head}] := {}", atoms.join(", "));
        let patoms: Vec<String> = cols.iter().map(|c| format!("{c} = $p_{c}")).collect();
        let params: BTreeMap<String, kleos::DataValue> =
            cols.iter().enumerate().map(|(i, c)| (format!("p_{c}"), kleos::DataValue::from(i as i64))).collect();
        let prule = format!("?[{head}] := {} :put {rel} {{{spec}}}", patoms.join(", "));
        let pq = format!("?[{head}] := {}", patoms.join(", "));
        let upd = format!("?[{head}] := {} :update {rel} {{{spec}}}", patoms.join(", "));
        let crule = format!("n[x] <- [[0]] ?[{head}] := n[x], {} :put {rel} {{{spec}}}", patoms.join(", "));
        let strs: Vec<String> = cols.iter().map(|c| format!("{c} = 'some string value'")).collect();
        let srule = format!("?[{head}] := {}, {} :put {rel} {{{spec}}}", patoms[0], strs[1..].join(", "));
        let mixed = format!("?[{head}] := *{rel}{{{c0}: x}}, x = 0, {} :put {rel} {{{spec}}}", patoms.join(", "), c0 = cols[0]);
        println!(
            "n={n:2}  lit-rule-put {:7.3}  row-put {:6.3}  lit-query {:7.3}  param-rule-put {:7.3}  param-query {:7.3}  stored+params-put {:7.3} ms",
            time(&db, &rule, 9),
            time(&db, &row, 9),
            time(&db, &q, 9),
            timep(&db, &prule, params.clone(), 9),
            timep(&db, &pq, params.clone(), 9),
            timep(&db, &mixed, params.clone(), 5),
        );
        println!(
            "      update {:7.3}  construle+params {:7.3}  string-literals {:7.3} ms",
            timep(&db, &upd, params.clone(), 9),
            timep(&db, &crule, params.clone(), 9),
            timep(&db, &srule, params.clone(), 9),
        );
    }
}
