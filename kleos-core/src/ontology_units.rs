/*
 * Copyright 2026, The kleos authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Units of typed literals (docs/specs/ontology.md, "Units"): a small
//! built-in table that brings a value with a unit to its dimension's base
//! unit, so "5 km" and "5000 m" compare equal, a property declared in `km`
//! accepts miles, and a single-valued fact restated in another unit is not
//! a change. Currencies are their own dimension per ISO 4217 code: amounts
//! compare only within one currency (no exchange rates).

use serde_json::Value as JsonValue;

/// A value brought to its dimension's base unit.
#[derive(Debug, Clone, PartialEq)]
pub struct Quantity {
    pub value: f64,
    /// `length`, `mass`, `temperature`, `duration`, `volume`, `speed`,
    /// `data`, or `currency:XXX`.
    pub dim: String,
    /// The base unit of `dim` (`m`, `kg`, `K`, `s`, `l`, `m/s`, `B`, or
    /// the currency code).
    pub base: String,
}

/// (dimension, base unit, factor to base, offset to base): base = value ×
/// factor + offset.
fn unit(u: &str) -> Option<(&'static str, &'static str, f64, f64)> {
    let u = u.trim();
    let low = u.to_lowercase();
    let low = low.trim_end_matches('.');
    Some(match low {
        // length
        "m" | "meter" | "meters" | "metre" | "metres" => ("length", "m", 1.0, 0.0),
        "km" | "kilometer" | "kilometers" | "kilometre" | "kilometres" => ("length", "m", 1000.0, 0.0),
        "cm" | "centimeter" | "centimeters" | "centimetre" | "centimetres" => ("length", "m", 0.01, 0.0),
        "mm" | "millimeter" | "millimeters" | "millimetre" | "millimetres" => ("length", "m", 0.001, 0.0),
        "mi" | "mile" | "miles" => ("length", "m", 1609.344, 0.0),
        "yd" | "yard" | "yards" => ("length", "m", 0.9144, 0.0),
        "ft" | "foot" | "feet" => ("length", "m", 0.3048, 0.0),
        "in" | "inch" | "inches" => ("length", "m", 0.0254, 0.0),
        // mass
        "kg" | "kilogram" | "kilograms" | "kilo" | "kilos" => ("mass", "kg", 1.0, 0.0),
        "g" | "gram" | "grams" => ("mass", "kg", 0.001, 0.0),
        "mg" | "milligram" | "milligrams" => ("mass", "kg", 1e-6, 0.0),
        "t" | "tonne" | "tonnes" => ("mass", "kg", 1000.0, 0.0),
        "lb" | "lbs" | "pound" | "pounds" => ("mass", "kg", 0.45359237, 0.0),
        "oz" | "ounce" | "ounces" => ("mass", "kg", 0.028349523125, 0.0),
        "st" | "stone" => ("mass", "kg", 6.35029318, 0.0),
        // temperature (base kelvin)
        "k" | "kelvin" => ("temperature", "K", 1.0, 0.0),
        "c" | "°c" | "celsius" | "degc" => ("temperature", "K", 1.0, 273.15),
        "f" | "°f" | "fahrenheit" | "degf" => ("temperature", "K", 5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
        // duration
        "s" | "sec" | "secs" | "second" | "seconds" => ("duration", "s", 1.0, 0.0),
        "ms" | "millisecond" | "milliseconds" => ("duration", "s", 0.001, 0.0),
        "min" | "mins" | "minute" | "minutes" => ("duration", "s", 60.0, 0.0),
        "h" | "hr" | "hrs" | "hour" | "hours" => ("duration", "s", 3600.0, 0.0),
        "d" | "day" | "days" => ("duration", "s", 86_400.0, 0.0),
        "wk" | "wks" | "week" | "weeks" => ("duration", "s", 604_800.0, 0.0),
        "mo" | "month" | "months" => ("duration", "s", 2_629_746.0, 0.0),
        "yr" | "yrs" | "year" | "years" => ("duration", "s", 31_556_952.0, 0.0),
        // volume
        "l" | "liter" | "liters" | "litre" | "litres" => ("volume", "l", 1.0, 0.0),
        "ml" | "milliliter" | "milliliters" | "millilitre" | "millilitres" => ("volume", "l", 0.001, 0.0),
        "gal" | "gallon" | "gallons" => ("volume", "l", 3.785411784, 0.0),
        // speed
        "m/s" | "mps" => ("speed", "m/s", 1.0, 0.0),
        "km/h" | "kmh" | "kph" => ("speed", "m/s", 1000.0 / 3600.0, 0.0),
        "mph" | "mi/h" => ("speed", "m/s", 1609.344 / 3600.0, 0.0),
        // data (decimal prefixes)
        "b" | "byte" | "bytes" => ("data", "B", 1.0, 0.0),
        "kb" => ("data", "B", 1e3, 0.0),
        "mb" => ("data", "B", 1e6, 0.0),
        "gb" => ("data", "B", 1e9, 0.0),
        "tb" => ("data", "B", 1e12, 0.0),
        _ => return None,
    })
}

/// Whether `u` is an ISO 4217-shaped currency code (three capital letters),
/// or a common symbol for one.
pub fn currency(u: &str) -> Option<String> {
    let u = u.trim();
    let code = match u {
        "$" => "USD",
        "€" => "EUR",
        "£" => "GBP",
        "¥" => "JPY",
        "₹" => "INR",
        _ => u,
    };
    (code.len() == 3 && code.chars().all(|c| c.is_ascii_uppercase())).then(|| code.to_string())
}

/// `value unit` brought to its base unit; `None` for an unknown unit.
pub fn normalize(value: f64, u: &str) -> Option<Quantity> {
    if let Some(code) = currency(u) {
        return Some(Quantity { value, dim: format!("currency:{code}"), base: code });
    }
    let (dim, base, f, off) = unit(u)?;
    Some(Quantity { value: value * f + off, dim: dim.to_string(), base: base.to_string() })
}

/// `value` from unit `from` to unit `to`; `None` when either is unknown or
/// they measure different things (two currencies included).
pub fn convert(value: f64, from: &str, to: &str) -> Option<f64> {
    let q = normalize(value, from)?;
    if let Some(code) = currency(to) {
        return (q.dim == format!("currency:{code}")).then_some(q.value);
    }
    let (dim, _, f, off) = unit(to)?;
    (q.dim == dim).then(|| (q.value - off) / f)
}

/// Whether two units measure the same thing (and so convert).
pub fn compatible(a: &str, b: &str) -> bool {
    match (normalize(1.0, a), normalize(1.0, b)) {
        (Some(x), Some(y)) => x.dim == y.dim,
        _ => false,
    }
}

/// A literal's numeric value: a JSON number, or a string that parses as
/// one ("5", "1,500").
fn number(v: &JsonValue) -> Option<f64> {
    match v {
        JsonValue::Number(n) => n.as_f64(),
        JsonValue::String(s) => s.trim().replace(',', "").parse::<f64>().ok(),
        _ => None,
    }
}

/// A literal `{value, unit}` as a quantity.
pub fn quantity_of(lit: &JsonValue) -> Option<Quantity> {
    let v = number(lit.get("value")?)?;
    let u = lit.get("unit")?.as_str()?;
    normalize(v, u)
}

/// Whether two literals state the same quantity: same dimension and equal
/// in base units within a relative 1e-6. `None` when either has no
/// convertible unit (the caller compares them as before).
pub fn same_quantity(a: &JsonValue, b: &JsonValue) -> Option<bool> {
    let (x, y) = (quantity_of(a)?, quantity_of(b)?);
    if x.dim != y.dim {
        return Some(false);
    }
    let scale = x.value.abs().max(y.value.abs()).max(1e-12);
    Some((x.value - y.value).abs() <= 1e-6 * scale)
}

/// The canonical key of a literal with a convertible unit, in base units
/// ("5 km" → "5000 m"); `None` otherwise.
pub fn canon_key(lit: &JsonValue) -> Option<String> {
    let q = quantity_of(lit)?;
    let v = (q.value * 1e9).round() / 1e9;
    Some(format!("{v} {}", q.base))
}

/// Units that, written as a separate word after a number, are more often an
/// ordinary word or letter than a unit.
const AMBIGUOUS: &[&str] = &["in", "t", "c", "f", "k", "d", "h", "s", "g", "b", "m", "l", "st", "mo", "ms", "mb", "oz"];

/// A quantity written in text ("5 km", "5,000 m", "$1,500", "72°F"): the
/// first number followed (or, for a currency symbol, preceded) by a known
/// unit.
pub fn parse_quantity(text: &str) -> Option<(f64, String)> {
    let toks: Vec<&str> = text.split_whitespace().collect();
    for (i, t) in toks.iter().enumerate() {
        let t = t.trim_matches(|c: char| matches!(c, '(' | ')' | ',' | ';' | '!' | '?'));
        // "$1,500" / "€20"
        if let Some(sym) = t.chars().next().filter(|c| matches!(c, '$' | '€' | '£' | '¥' | '₹')) {
            if let Ok(v) = t[sym.len_utf8()..].trim_end_matches('.').replace(',', "").parse::<f64>() {
                return Some((v, currency(&sym.to_string())?));
            }
        }
        // "72°F", "5km"
        let split = t.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == ',' || c == '-'));
        let (num, rest) = match split {
            Some(k) if k > 0 => (&t[..k], &t[k..]),
            None => (t, ""),
            _ => continue,
        };
        let Ok(v) = num.trim_end_matches('.').replace(',', "").parse::<f64>() else { continue };
        let u = if !rest.is_empty() { rest.trim_end_matches('.').to_string() } else {
            match toks.get(i + 1) {
                Some(n) => {
                    let w = n.trim_matches(|c: char| matches!(c, '(' | ')' | ',' | ';' | '!' | '?' | '.')).to_string();
                    // a short unit as its own word is usually an ordinary
                    // word ("5 in the park", "3 m" is fine attached: "3m")
                    if AMBIGUOUS.contains(&w.to_lowercase().as_str()) {
                        continue;
                    }
                    w
                }
                None => continue,
            }
        };
        if normalize(v, &u).is_some() {
            return Some((v, u));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn conversions() {
        assert!((convert(5.0, "km", "m").unwrap() - 5000.0).abs() < 1e-9);
        assert!((convert(1.0, "mi", "km").unwrap() - 1.609344).abs() < 1e-9);
        assert!((convert(10.0, "kg", "lb").unwrap() - 22.046226).abs() < 1e-5);
        assert!((convert(100.0, "°C", "°F").unwrap() - 212.0).abs() < 1e-9);
        assert!((convert(32.0, "F", "C").unwrap()).abs() < 1e-9);
        assert!((convert(2.0, "hours", "min").unwrap() - 120.0).abs() < 1e-9);
        assert!((convert(1.0, "week", "days").unwrap() - 7.0).abs() < 1e-9);
        assert_eq!(convert(5.0, "km", "kg"), None);
        assert_eq!(convert(5.0, "km", "parsec"), None);
        assert_eq!(convert(10.0, "EUR", "USD"), None);
        assert_eq!(convert(10.0, "EUR", "EUR"), Some(10.0));
        assert!(compatible("mph", "km/h"));
        assert!(!compatible("USD", "EUR"));
    }

    #[test]
    fn literals() {
        assert_eq!(same_quantity(&json!({"value": 5, "unit": "km"}), &json!({"value": 5000, "unit": "m"})), Some(true));
        assert_eq!(same_quantity(&json!({"value": "5", "unit": "km"}), &json!({"value": 5.1, "unit": "km"})), Some(false));
        assert_eq!(same_quantity(&json!({"value": 1500, "unit": "EUR"}), &json!({"value": 1500, "unit": "USD"})), Some(false));
        assert_eq!(same_quantity(&json!({"value": 1500}), &json!({"value": 1500, "unit": "m"})), None);
        assert_eq!(canon_key(&json!({"value": 5, "unit": "km"})).as_deref(), Some("5000 m"));
        assert_eq!(canon_key(&json!({"value": 1, "unit": "lb"})).as_deref(), Some("0.45359237 kg"));
    }

    #[test]
    fn text() {
        assert_eq!(parse_quantity("I ran 5 km today"), Some((5.0, "km".into())));
        assert_eq!(parse_quantity("rent is $1,500 now"), Some((1500.0, "USD".into())));
        assert_eq!(parse_quantity("it was 72°F outside"), Some((72.0, "°F".into())));
        assert_eq!(parse_quantity("about 3 kids"), None);
        assert_eq!(parse_quantity("met 5 in the park"), None);
        assert_eq!(parse_quantity("a 3m drop"), Some((3.0, "m".into())));
    }
}
