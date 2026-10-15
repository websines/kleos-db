/*
 *  Copyright 2022, The Cozo Project Authors.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 *  If a copy of the MPL was not distributed with this file,
 *  You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 */

use std::collections::{BTreeMap, HashMap};
use std::mem::size_of;

use crate::data::symb::Symbol;
use crate::data::value::DataValue;

#[test]
fn show_size() {
    dbg!(size_of::<DataValue>());
    dbg!(size_of::<Symbol>());
    dbg!(size_of::<String>());
    dbg!(size_of::<HashMap<String, String>>());
    dbg!(size_of::<BTreeMap<String, String>>());
}

#[test]
fn utf8() {
    let c = char::from_u32(0x10FFFF).unwrap();
    let mut s = String::new();
    s.push(c);
    println!("{}", s);
    println!(
        "{:b} {:b} {:b} {:b}",
        s.as_bytes()[0],
        s.as_bytes()[1],
        s.as_bytes()[2],
        s.as_bytes()[3]
    );
    dbg!(s);
}

#[test]
fn display_datavalues() {
    println!("{}", DataValue::Null);
    println!("{}", DataValue::from(true));
    println!("{}", DataValue::from(-1));
    println!("{}", DataValue::from(-1_121_212_121.331_212));
    println!("{}", DataValue::from(f64::NAN));
    println!("{}", DataValue::from(f64::NEG_INFINITY));
    println!("{}", DataValue::from(vec![10, 20]));
    println!("{}", DataValue::from(vec!["hello", "你好"]));
    println!(
        "{}",
        DataValue::List(vec![
            DataValue::from(false),
            DataValue::from(r###"abc"你"好'啊👌"###),
            DataValue::from(f64::NEG_INFINITY),
        ])
    );
}

#[test]
fn float_display_round_trips_as_float() {
    use crate::parse::parse_expressions;
    for f in [
        1.0,
        -2.0,
        0.0,
        0.1,
        1e300,
        1.5e-7,
        -1_121_212_121.331_212,
        123_456_789.0,
        f64::MAX,
        f64::MIN_POSITIVE,
    ] {
        let v = DataValue::from(f);
        let src = v.to_string();
        let back = parse_expressions(&src, &BTreeMap::new())
            .and_then(|e| e.eval_to_const())
            .unwrap_or_else(|e| panic!("{src} does not parse back: {e:?}"));
        assert_eq!(back, v, "{f} displayed as {src}");
    }
    assert_eq!(DataValue::from(1.0).to_string(), "1.0");
    assert_eq!(DataValue::from(1).to_string(), "1");
}
