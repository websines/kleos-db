/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Values: Postgres text output (pgoutput and `COPY … TO STDOUT` both use
//! it) to typed JSON, the COPY text format, and entity keys.

use serde_json::{json, Value};

use crate::catalog::TypeKind;

/// Typed JSON for a column's text value. Numbers that JSON cannot hold
/// exactly (numeric beyond 15 significant digits, int8 beyond 2^53) stay
/// strings; the literal's datatype says what they are.
pub fn typed(kind: TypeKind, text: &str) -> Value {
    match kind {
        TypeKind::Bool => Value::Bool(text == "t" || text == "true"),
        TypeKind::Int => match text.parse::<i64>() {
            Ok(n) if n.unsigned_abs() < (1u64 << 53) => json!(n),
            _ => Value::String(text.to_string()),
        },
        TypeKind::Float => match text.parse::<f64>() {
            Ok(f) if f.is_finite() => json!(f),
            _ => Value::String(text.to_string()),
        },
        TypeKind::Numeric => {
            let digits = text.chars().filter(|c| c.is_ascii_digit()).count();
            match text.parse::<f64>() {
                Ok(f) if f.is_finite() && digits <= 15 => {
                    if !text.contains('.') && !text.contains('e') && f.fract() == 0.0 {
                        json!(f as i64)
                    } else {
                        json!(f)
                    }
                }
                _ => Value::String(text.to_string()),
            }
        }
        TypeKind::Json => serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string())),
        TypeKind::Array => parse_array(text),
        TypeKind::TimestampTz | TypeKind::Timestamp => Value::String(iso_timestamp(text)),
        _ => Value::String(text.to_string()),
    }
}

/// `2026-10-25 12:00:00.123+02` → `2026-10-25T12:00:00.123+02:00`.
pub fn iso_timestamp(text: &str) -> String {
    let mut s = text.replacen(' ', "T", 1);
    // a "+02" / "-05" offset needs minutes for RFC 3339
    if let Some(pos) = s.rfind(['+', '-']) {
        if pos > 10 {
            let off = &s[pos + 1..];
            if off.len() == 2 && off.chars().all(|c| c.is_ascii_digit()) {
                s.push_str(":00");
            }
        }
    }
    s
}

/// A one-dimensional array literal (`{a,"b c",NULL}`) to a JSON list;
/// nested arrays stay strings.
pub fn parse_array(text: &str) -> Value {
    let t = text.trim();
    if !(t.starts_with('{') && t.ends_with('}')) || t[1..t.len() - 1].contains('{') {
        return Value::String(text.to_string());
    }
    let inner = &t[1..t.len() - 1];
    if inner.is_empty() {
        return Value::Array(vec![]);
    }
    let mut out = vec![];
    let mut cur = String::new();
    let mut quoted = false;
    let mut was_quoted = false;
    let mut chars = inner.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                quoted = !quoted;
                was_quoted = true;
            }
            '\\' if quoted => {
                if let Some(n) = chars.next() {
                    cur.push(n);
                }
            }
            ',' if !quoted => {
                out.push(elem(&cur, was_quoted));
                cur.clear();
                was_quoted = false;
            }
            c => cur.push(c),
        }
    }
    out.push(elem(&cur, was_quoted));
    Value::Array(out)
}

fn elem(s: &str, quoted: bool) -> Value {
    if !quoted && s == "NULL" {
        return Value::Null;
    }
    if !quoted {
        if let Ok(n) = s.parse::<i64>() {
            return json!(n);
        }
        if let Ok(f) = s.parse::<f64>() {
            if f.is_finite() {
                return json!(f);
            }
        }
    }
    Value::String(s.to_string())
}

/// Split one line of `COPY … TO STDOUT` text format into fields (`None`
/// for `\N`), decoding backslash escapes.
pub fn copy_fields(line: &[u8]) -> Vec<Option<String>> {
    let mut out = vec![];
    let mut cur: Vec<u8> = vec![];
    let mut i = 0;
    let mut null = false;
    while i < line.len() {
        let b = line[i];
        match b {
            b'\t' => {
                out.push(if null { None } else { Some(String::from_utf8_lossy(&cur).into_owned()) });
                cur.clear();
                null = false;
            }
            b'\\' if i + 1 < line.len() => {
                i += 1;
                match line[i] {
                    b'N' => null = true,
                    b'b' => cur.push(8),
                    b'f' => cur.push(12),
                    b'n' => cur.push(b'\n'),
                    b'r' => cur.push(b'\r'),
                    b't' => cur.push(b'\t'),
                    b'v' => cur.push(11),
                    b'x' => {
                        let mut v = 0u8;
                        let mut n = 0;
                        while n < 2 && i + 1 < line.len() && line[i + 1].is_ascii_hexdigit() {
                            i += 1;
                            v = v * 16 + (line[i] as char).to_digit(16).unwrap() as u8;
                            n += 1;
                        }
                        cur.push(v);
                    }
                    d @ b'0'..=b'7' => {
                        let mut v = (d - b'0') as u32;
                        let mut n = 1;
                        while n < 3 && i + 1 < line.len() && (b'0'..=b'7').contains(&line[i + 1]) {
                            i += 1;
                            v = v * 8 + (line[i] - b'0') as u32;
                            n += 1;
                        }
                        cur.push(v as u8);
                    }
                    other => cur.push(other),
                }
            }
            b => cur.push(b),
        }
        i += 1;
    }
    out.push(if null { None } else { Some(String::from_utf8_lossy(&cur).into_owned()) });
    out
}

fn enc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '%' => o.push_str("%25"),
            '/' => o.push_str("%2F"),
            ',' => o.push_str("%2C"),
            c => o.push(c),
        }
    }
    o
}

/// A JSON key value as it appears in an entity key.
pub fn key_part(v: &Value) -> String {
    match v {
        Value::String(s) => enc(s),
        Value::Null => "%00".to_string(),
        other => enc(&other.to_string()),
    }
}

/// The entity key of a row: `pg:<source>/<schema.table>/<k1>,<k2>`.
pub fn entity_key(source: &str, table: &str, key: &[Value]) -> String {
    format!("pg:{source}/{table}/{}", key.iter().map(key_part).collect::<Vec<_>>().join(","))
}

/// The table-local part of an entity key (`42`, `7,3`).
pub fn row_key(key: &[Value]) -> String {
    key.iter().map(key_part).collect::<Vec<_>>().join(",")
}

/// A literal's canonical string, the fact's `o` (as `literal_key` in
/// kleos-core: the value, strings unquoted). Long values are cut and
/// suffixed with a hash so the key stays small and unique.
pub fn literal_o(v: &Value) -> String {
    let s = match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    if s.chars().count() > 256 {
        let head: String = s.chars().take(200).collect();
        format!("{head}…#{}", &short_hash(&s)[..12])
    } else {
        s
    }
}

pub fn short_hash(s: &str) -> String {
    use sha2::{Digest, Sha256};
    let d = Sha256::digest(s.as_bytes());
    d.iter().take(10).map(|b| format!("{b:02x}")).collect()
}

/// A value as text in a memory (strings unquoted).
pub fn display(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_text_format() {
        let f = copy_fields(b"42\tAcme\\tCorp\t\\N\tline\\none\t\\101");
        assert_eq!(f, vec![Some("42".into()), Some("Acme\tCorp".into()), None, Some("line\none".into()), Some("A".into())]);
    }

    #[test]
    fn typing() {
        assert_eq!(typed(TypeKind::Int, "42"), json!(42));
        assert_eq!(typed(TypeKind::Int, "9007199254740993"), json!("9007199254740993"));
        assert_eq!(typed(TypeKind::Numeric, "12.50"), json!(12.5));
        assert_eq!(typed(TypeKind::Numeric, "1234567890.1234567890"), json!("1234567890.1234567890"));
        assert_eq!(typed(TypeKind::Bool, "t"), json!(true));
        assert_eq!(typed(TypeKind::Array, "{a,\"b c\",NULL,3}"), json!(["a", "b c", null, 3]));
        assert_eq!(typed(TypeKind::TimestampTz, "2026-10-25 12:00:00+02"), json!("2026-10-25T12:00:00+02:00"));
        assert_eq!(entity_key("crm", "public.t", &[json!(7), json!("a/b,c")]), "pg:crm/public.t/7,a%2Fb%2Cc");
    }
}
