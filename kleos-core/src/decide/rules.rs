/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Value rules: a pre-check on numbers, dates, money, counts and units.
//!
//! Small decision models are weak at "only the value changed" (rent $1,800 →
//! $2,100, launch March 1 → April 20, 3 → 5 languages). This extracts the
//! value mentions of both texts, masks them, and compares what is left:
//!
//! * **Update**: both texts carry values, the value multisets differ, the
//!   masked context (subject + attribute words) clearly matches, exactly one
//!   kind of value changed, and neither side adds a negation the other lacks.
//!   [`apply_value_rule`] then short-circuits `supersedes` / `relation` to
//!   "update" and `duplicate` to "no", with a stated reason.
//! * **Hint**: values differ and some context is shared, but not enough to
//!   be sure. The model's answer stands; the reason is attached.
//! * **None**: anything else.
//!
//! Several differing value kinds at once (a date *and* an amount) usually
//! mean two different events ("ran 5 km on Monday" / "ran 10 km on
//! Tuesday"), so they never short-circuit.

use std::collections::BTreeSet;

use super::Template;

/// What the value pre-check found for a pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueRule {
    /// Same subject and attribute, only the value changed.
    Update(String),
    /// Values differ with partly shared context; the model decides.
    Hint(String),
    /// Nothing to say.
    None,
}

/// Kind of a value mention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Kind {
    Number,
    Money,
    Percent,
    Date,
    /// A year, or a month of a year (coarser than a [`Kind::Date`]).
    Year,
    Time,
}

const MONTHS: [&str; 24] = [
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
    "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
];
const WEEKDAYS: [&str; 7] = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];
const NUMBER_WORDS: [&str; 21] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
];
const NEGATIONS: [&str; 8] = ["not", "never", "no", "isn't", "doesn't", "wasn't", "didn't", "nobody"];
/// Bounds and approximations: "more than $147 million" is not a value that
/// $148.5 million replaces.
const COMPARATIVES: [&str; 16] = [
    "more", "less", "over", "under", "least", "most", "fewer", "above", "below", "nearly", "approximately",
    "around", "roughly", "between", "almost", "than",
];
const STOP: [&str; 44] = [
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on", "at", "for", "and", "or",
    "now", "has", "have", "had", "per", "its", "it", "this", "that", "as", "by", "with", "from", "after", "since",
    "their", "his", "her", "our", "my", "your", "just", "about", "so", "than", "then", "s",
];

/// A value mention: kind and normalised value.
type Val = (Kind, String);

fn month_index(w: &str) -> Option<usize> {
    MONTHS.iter().position(|m| *m == w).map(|i| if i >= 12 { [0, 1, 2, 3, 5, 6, 7, 8, 8, 9, 10, 11][i - 12] } else { i })
}

/// Normalise a numeric token: drop thousands separators and ordinal
/// suffixes; `None` if it has no digit.
fn number(tok: &str) -> Option<String> {
    if !tok.chars().any(|c| c.is_ascii_digit()) {
        return None;
    }
    let t = tok.trim_end_matches(|c: char| c.is_ascii_alphabetic());
    let t: String = t.chars().filter(|c| *c != ',').collect();
    let t = t.trim_matches(|c: char| !c.is_ascii_digit());
    if t.is_empty() {
        return None;
    }
    // canonical float form when it parses ("1800" == "1800.00")
    Some(match t.parse::<f64>() {
        Ok(x) => format!("{x}"),
        Err(_) => t.to_string(),
    })
}

/// Split into lower-case word tokens, keeping `$1,800`, `12.5%`, `20.04`,
/// `2024-03-01`, `6:30` together.
fn tokens(text: &str) -> Vec<String> {
    let mut out = vec![];
    for raw in text.split_whitespace() {
        let w = raw
            .trim_matches(|c: char| matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | '(' | ')' | '"' | '\'' | '`'))
            .to_lowercase();
        let w = w.strip_suffix("'s").map(str::to_string).unwrap_or(w);
        if !w.is_empty() {
            out.push(w);
        }
    }
    out
}

/// Value mentions and the masked context words of `text`.
fn analyse(text: &str) -> (Vec<Val>, Vec<String>) {
    let toks = tokens(text);
    let mut vals = vec![];
    let mut ctx = vec![];
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i].as_str();
        let next = toks.get(i + 1).map(String::as_str);
        let prev_num = |j: usize| j > 0 && number(&toks[j - 1]).is_some();
        if let Some(m) = month_index(t) {
            // "March 1", "1 March", "March 2024", bare month
            let day = next.and_then(number);
            if let Some(d) = day {
                // "March 2024" is a month of a year, "March 1" a day
                let kind = if d.len() == 4 { Kind::Year } else { Kind::Date };
                vals.push((kind, format!("{m}-{d}")));
                i += 2;
                continue;
            }
            if prev_num(i) {
                // "1 March": the number was already pushed as a Number; retag it
                if let Some(v) = vals.pop() {
                    vals.push((Kind::Date, format!("{m}-{}", v.1)));
                }
            } else {
                vals.push((Kind::Date, format!("{m}")));
            }
            i += 1;
            continue;
        }
        if WEEKDAYS.iter().any(|d| t.trim_end_matches('s') == *d) {
            vals.push((Kind::Date, t.trim_end_matches('s').to_string()));
            i += 1;
            continue;
        }
        if let Some(n) = NUMBER_WORDS.iter().position(|w| *w == t) {
            // "one" is often a pronoun ("the old one"); only count it before a noun-ish word
            if n != 1 || next.map(|x| !STOP.contains(&x)).unwrap_or(false) {
                vals.push((Kind::Number, format!("{n}")));
                i += 1;
                continue;
            }
        }
        if let Some(n) = number(t) {
            let kind = if t.starts_with(['$', '€', '£', '¥']) || matches!(next, Some("dollars" | "euros" | "pounds" | "usd" | "eur" | "gbp")) {
                Kind::Money
            } else if t.ends_with('%') || matches!(next, Some("percent" | "%")) {
                Kind::Percent
            } else if t.contains(':') || matches!(next, Some("am" | "pm" | "a.m" | "p.m")) {
                Kind::Time
            } else if t.contains('-') && t.len() >= 8 {
                Kind::Date
            } else if t.len() == 4 && (n.starts_with("19") || n.starts_with("20")) && !t.contains('.') {
                Kind::Year
            } else {
                Kind::Number
            };
            vals.push((kind, n));
            i += 1;
            continue;
        }
        if !STOP.contains(&t) {
            ctx.push(t.to_string());
        }
        i += 1;
    }
    (vals, ctx)
}

fn stem(w: &str) -> &str {
    for suf in ["ing", "ed", "es", "s"] {
        if w.len() > suf.len() + 3 {
            if let Some(s) = w.strip_suffix(suf) {
                return s;
            }
        }
    }
    w
}

/// The value pre-check for `(old, new)`.
pub fn value_rule(old: &str, new: &str) -> ValueRule {
    let (va, ca) = analyse(old);
    let (vb, cb) = analyse(new);
    if va.is_empty() || vb.is_empty() {
        return ValueRule::None;
    }
    let sa: BTreeSet<&Val> = va.iter().collect();
    let sb: BTreeSet<&Val> = vb.iter().collect();
    if sa == sb {
        return ValueRule::None;
    }
    // kinds whose values differ
    let changed: BTreeSet<Kind> = sa.symmetric_difference(&sb).map(|v| v.0).collect();
    let neg_a = ca.iter().any(|w| NEGATIONS.contains(&w.as_str()));
    let neg_b = cb.iter().any(|w| NEGATIONS.contains(&w.as_str()));
    let wa: BTreeSet<&str> = ca.iter().map(|w| stem(w)).filter(|w| !NEGATIONS.contains(w)).collect();
    let wb: BTreeSet<&str> = cb.iter().map(|w| stem(w)).filter(|w| !NEGATIONS.contains(w)).collect();
    let inter = wa.intersection(&wb).count();
    let union = wa.union(&wb).count().max(1);
    let jac = inter as f64 / union as f64;
    // overlap relative to the shorter side: "rent is $1,800" vs "landlord raised the rent to $2,100 a month"
    let cover = inter as f64 / wa.len().min(wb.len()).max(1) as f64;
    let fmt = |vs: &BTreeSet<&Val>, other: &BTreeSet<&Val>| -> String {
        vs.difference(other).map(|v| v.1.clone()).collect::<Vec<_>>().join(", ")
    };
    let what = changed.iter().map(|k| format!("{k:?}").to_lowercase()).collect::<Vec<_>>().join("+");
    let reason = format!(
        "{what} changed ({} -> {}), context overlap {:.2}",
        fmt(&sa, &sb),
        fmt(&sb, &sa),
        jac
    );
    let one_kind = changed.len() == 1;
    let same_shape = va.len() == vb.len();
    // a value of the old note must be gone (not just new values added: an elaboration)
    let replaced = !sa.is_subset(&sb) && !sb.is_subset(&sa);
    let bounded = ca.iter().chain(&cb).any(|w| COMPARATIVES.contains(&w.as_str()));
    if one_kind
        && replaced
        && !bounded
        && neg_a == neg_b
        && inter >= 2
        && (jac >= 0.6 || (cover >= 0.75 && same_shape && jac >= 0.4))
    {
        return ValueRule::Update(reason);
    }
    if inter >= 1 {
        return ValueRule::Hint(reason);
    }
    ValueRule::None
}

/// Probability the rule gives a short-circuited answer.
pub const RULE_P: f64 = 0.99;

/// Apply the value pre-check to label probabilities of `template` for
/// `(a, b)`. Returns the new probabilities and the reason when the rule
/// decided; `None` when the model's answer stands. Only `supersedes`,
/// `relation` (both directional: old, new) and `duplicate` use it.
pub fn apply_value_rule(template: Template, a: &str, b: &str, probs: &[f64]) -> Option<(Vec<f64>, String)> {
    let reason = match template {
        Template::Supersedes | Template::Relation | Template::Duplicate => match value_rule(a, b) {
            ValueRule::Update(r) => r,
            _ => return None,
        },
        _ => return None,
    };
    let mut p = vec![0.0; probs.len()];
    match template {
        Template::Supersedes => {
            p[1] = probs[1].max(RULE_P);
            p[0] = 1.0 - p[1];
        }
        Template::Duplicate => {
            p[1] = probs[1].min(1.0 - RULE_P);
            p[0] = 1.0 - p[1];
        }
        _ => {
            // relation labels: update, contradiction, compatible, duplicate
            let rest = 1.0 - RULE_P;
            p.iter_mut().for_each(|x| *x = rest / 3.0);
            p[0] = RULE_P;
        }
    }
    Some((p, format!("value rule: {reason}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_update(a: &str, b: &str) -> bool {
        matches!(value_rule(a, b), ValueRule::Update(_))
    }

    #[test]
    fn value_changes() {
        assert!(is_update("Ana's rent is €1,250 per month.", "Ana's rent is €1,320 per month."));
        assert!(is_update("Bluefin employs 460 people.", "Bluefin now employs 520 people."));
        assert!(is_update("The release is scheduled for May 3.", "The release is now scheduled for June 9."));
        assert!(is_update("Kai speaks two languages.", "Kai now speaks three languages."));
        // different event (date and amount both differ)
        assert!(!is_update("Tom ran 5 km on Monday.", "Tom ran 10 km on Tuesday."));
        // different attribute
        assert!(!is_update("The server runs Ubuntu 22.04.", "The server has 64 GB of RAM."));
        // value on one side only
        assert!(!is_update("Lena lives in Porto.", "Lena's flat is 12 km from the office."));
        // same values
        assert!(!is_update("Rent is $1,800 a month.", "Monthly rent is $1800."));
        // negation asymmetry
        assert!(!is_update("The meeting is on March 3.", "The meeting is not on March 4."));
        // bounds, added detail, day vs year granularity
        assert!(!is_update("The film grossed more than $147 million abroad.", "The film grossed $148.5 million abroad."));
        assert!(!is_update("The club played its first game in 1920.", "The club was formed in 1919 and played its first game in 1920."));
        assert!(!is_update("On September 29, Marrone named Orton the starter.", "In September 2014, Marrone named Orton the starter."));
    }
}
