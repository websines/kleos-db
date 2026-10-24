//! Literal bindings in rule bodies (`x = true`, `x = 'a'`, `x = null`, ...)
//! must reach the rule head and stored-relation writes in every shape.

use std::collections::BTreeMap;

use kleos::{DbInstance, ScriptMutability};
use serde_json::{json, Value};

fn run(db: &DbInstance, script: &str) -> Value {
    let r = db
        .run_script(script, BTreeMap::new(), ScriptMutability::Mutable)
        .unwrap_or_else(|e| panic!("{script}: {e:?}"));
    r.into_json()["rows"].clone()
}

fn db() -> DbInstance {
    DbInstance::new("mem", "", "").unwrap()
}

#[test]
fn bool_literal_in_head() {
    let db = db();
    assert_eq!(run(&db, "?[k, c] := k = 1, c = true"), json!([[1, true]]));
}

#[test]
fn update_with_bool_literal() {
    let db = db();
    run(&db, ":create rel {k: Int => c: Bool}");
    run(&db, "?[k, c] <- [[1, false]] :put rel {k => c}");
    run(&db, "?[k, c] := k = 1, c = true :update rel {k => c}");
    assert_eq!(run(&db, "?[k, c] := *rel{k, c}"), json!([[1, true]]));
}

#[test]
fn update_after_stored_read_with_bool_literal() {
    let db = db();
    run(&db, ":create rel {k: Int => c: Bool, n: Int}");
    run(&db, "?[k, c, n] <- [[1, false, 5], [2, false, 6]] :put rel {k => c, n}");
    run(&db, "?[k, c] := *rel{k, n}, n > 5, c = true :update rel {k => c}");
    assert_eq!(
        run(&db, "?[k, c] := *rel{k, c}"),
        json!([[1, false], [2, true]])
    );
}

#[test]
fn two_rules_binding_true_and_false() {
    let db = db();
    run(&db, ":create rel {k: Int => n: Int}");
    run(&db, "?[k, n] <- [[1, 5], [2, 6]] :put rel {k => n}");
    let rows = run(&db, "?[k, c] := *rel{k, n}, n > 5, c = true  ?[k, c] := *rel{k, n}, n <= 5, c = false");
    assert_eq!(rows, json!([[1, false], [2, true]]));
}

#[test]
fn other_literals() {
    let db = db();
    run(&db, ":create rel {k: Int => n: Int}");
    run(&db, "?[k, n] <- [[1, 5], [2, 6]] :put rel {k => n}");
    for lit in ["null", "'s'", "3", "1.5", "[1, 2]", "false"] {
        let rows = run(&db, &format!("?[k, c] := *rel{{k, n}}, n > 5, c = {lit}"));
        let want: Value = serde_json::from_str(&lit.replace('\'', "\"")).unwrap();
        assert_eq!(rows, json!([[2, want]]), "{lit}");
    }
}

#[test]
fn bool_literal_followed_by_whitespace_or_comment() {
    // the parser once read the span of `true` with its trailing whitespace
    let db = db();
    for q in [
        "?[c] := c = true ",
        "?[c] := c = true\n",
        "?[c] := c = true # comment",
        "?[c] := c = true /* c */",
        "?[c] := c = (true )",
        "?[c] := c = true ?[c] := c = 1, false",
    ] {
        assert_eq!(run(&db, q), json!([[true]]), "{q:?}");
    }
    assert_eq!(run(&db, "?[c] := c = null "), json!([[null]]));
    assert_eq!(run(&db, "?[c] := c = false "), json!([[false]]));
}

#[test]
fn keyword_literals_keep_word_boundary() {
    let db = db();
    run(&db, ":create t {trueValue: Int, null_x: Int}");
    run(&db, "?[trueValue, null_x] <- [[1, 2]] :put t {trueValue, null_x}");
    assert_eq!(run(&db, "?[a, b] := *t{trueValue: a, null_x: b}"), json!([[1, 2]]));
    assert_eq!(run(&db, "?[x] := x = [true,false,null]"), json!([[[true, false, null]]]));
}

#[test]
fn put_and_derived_rule_with_bool_literal() {
    let db = db();
    run(&db, ":create r {k: Int => c: Bool}");
    run(&db, "?[k, c] := k = 1, c = true :put r {k => c}");
    assert_eq!(run(&db, "?[k, c] := *r{k, c}"), json!([[1, true]]));
    assert_eq!(run(&db, "a[k, c] := k = 1, c = true  ?[k, c] := a[k, c]"), json!([[1, true]]));
}
