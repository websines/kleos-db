//! Character classes as Python's `str` methods and `unicodedata` see them
//! (the pipeline's rules were written against those), the recognizer
//! vocabulary, and right-to-left handling (`rtl.py`).

use unicode_bidi::{bidi_class, BidiClass};
use unicode_general_category::{get_general_category, GeneralCategory as G};
use unicode_normalization::UnicodeNormalization;

/// Python `str.isalpha` for one character (categories Lu, Ll, Lt, Lm, Lo).
pub fn is_alpha(c: char) -> bool {
    matches!(
        get_general_category(c),
        G::UppercaseLetter | G::LowercaseLetter | G::TitlecaseLetter | G::ModifierLetter | G::OtherLetter
    )
}

/// Python `str.isdigit` (decimal digits and the superscript / subscript ones).
pub fn is_digit(c: char) -> bool {
    get_general_category(c) == G::DecimalNumber
        || matches!(c, '²' | '³' | '¹' | '⁰' | '⁴'..='⁹' | '₀'..='₉' | '①'..='⑨')
}

pub fn is_upper(c: char) -> bool {
    c.is_uppercase()
}

pub fn is_lower(c: char) -> bool {
    c.is_lowercase()
}

/// Python `str.isspace`.
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\x1c'..='\x1f')
}

/// Python `str.isdigit` on a whole string (non-empty, all digits).
pub fn all_digits(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_digit)
}

/// `unicodedata.category(c)` as its two-letter code.
pub fn category(c: char) -> &'static str {
    match get_general_category(c) {
        G::UppercaseLetter => "Lu",
        G::LowercaseLetter => "Ll",
        G::TitlecaseLetter => "Lt",
        G::ModifierLetter => "Lm",
        G::OtherLetter => "Lo",
        G::NonspacingMark => "Mn",
        G::SpacingMark => "Mc",
        G::EnclosingMark => "Me",
        G::DecimalNumber => "Nd",
        G::LetterNumber => "Nl",
        G::OtherNumber => "No",
        G::ConnectorPunctuation => "Pc",
        G::DashPunctuation => "Pd",
        G::OpenPunctuation => "Ps",
        G::ClosePunctuation => "Pe",
        G::InitialPunctuation => "Pi",
        G::FinalPunctuation => "Pf",
        G::OtherPunctuation => "Po",
        G::MathSymbol => "Sm",
        G::CurrencySymbol => "Sc",
        G::ModifierSymbol => "Sk",
        G::OtherSymbol => "So",
        G::SpaceSeparator => "Zs",
        G::LineSeparator => "Zl",
        G::ParagraphSeparator => "Zp",
        G::Control => "Cc",
        G::Format => "Cf",
        G::Surrogate => "Cs",
        G::PrivateUse => "Co",
        G::Unassigned => "Cn",
        _ => "Cn",
    }
}

pub fn is_mark(c: char) -> bool {
    matches!(get_general_category(c), G::NonspacingMark | G::EnclosingMark)
}

/// `unicodedata.east_asian_width(c) in "WF"`.
pub fn wide(c: char) -> bool {
    unicode_width::UnicodeWidthChar::width(c) == Some(2)
}

pub fn display_len(t: &str) -> usize {
    t.chars().map(|c| if wide(c) { 2 } else { 1 }).sum()
}

pub fn nfc(s: &str) -> String {
    s.nfc().collect()
}

/// `" ".join(s.split())`.
pub fn squash_spaces(s: &str) -> String {
    s.split(is_space).filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------- right to left

pub fn is_rtl(c: char) -> bool {
    matches!(bidi_class(c), BidiClass::R | BidiClass::AL)
}

fn is_ltr(c: char) -> bool {
    matches!(bidi_class(c), BidiClass::L | BidiClass::EN | BidiClass::AN)
}

pub fn has_rtl(text: &str) -> bool {
    text.chars().any(is_rtl)
}

pub fn bidi_l(c: char) -> bool {
    bidi_class(c) == BidiClass::L
}

/// More right-to-left letters than left-to-right ones.
pub fn rtl_base(text: &str) -> bool {
    let r = text.chars().filter(|&c| is_rtl(c)).count();
    let l = text.chars().filter(|&c| bidi_l(c)).count();
    r > l
}

fn clusters(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in text.chars() {
        if !out.is_empty() && is_mark(c) {
            out.last_mut().unwrap().push(c);
        } else {
            out.push(c.to_string());
        }
    }
    out
}

fn kind(cl: &str) -> char {
    let c = cl.chars().next().unwrap();
    if is_rtl(c) {
        'R'
    } else if is_ltr(c) {
        'L'
    } else {
        'N'
    }
}

fn runs(kinds: &[char], strong: char, stop: char) -> Vec<(usize, usize)> {
    let mut spans = vec![];
    let (mut start, mut last): (Option<usize>, usize) = (None, 0);
    for (i, k) in kinds.iter().copied().chain(std::iter::once('#')).enumerate() {
        if k == strong {
            if start.is_none() {
                start = Some(i);
            }
            last = i;
        } else if k == stop || k == '#' {
            if let Some(s) = start {
                spans.push((s, last + 1));
            }
            start = None;
        }
    }
    spans
}

fn mirror(cl: &str) -> String {
    const A: &str = "()[]{}<>«»‹›";
    const B: &str = ")(][}{><»«›‹";
    let mut it = cl.chars();
    let c = it.next().unwrap();
    let m = A.chars().position(|x| x == c).map(|i| B.chars().nth(i).unwrap()).unwrap_or(c);
    std::iter::once(m).chain(it).collect()
}

/// `rtl.flip`: logical to visual order and back (its own inverse).
pub fn flip(text: &str, base_rtl: Option<bool>) -> String {
    if !has_rtl(text) {
        return text.to_string();
    }
    let cls = clusters(text);
    let kinds: Vec<char> = cls.iter().map(|c| kind(c)).collect();
    if base_rtl.unwrap_or_else(|| rtl_base(text)) {
        let mut units: Vec<String> = vec![];
        let mut spans = runs(&kinds, 'L', 'R');
        spans.push((cls.len(), cls.len()));
        let mut i = 0;
        for (a, b) in spans {
            for c in &cls[i..a] {
                units.push(mirror(c));
            }
            if a < b {
                units.push(cls[a..b].concat());
            }
            i = b;
        }
        units.reverse();
        return units.concat();
    }
    let mut out = cls.clone();
    for (a, b) in runs(&kinds, 'R', 'L') {
        let seg: Vec<String> = cls[a..b].iter().rev().map(|c| mirror(c)).collect();
        for (k, s) in seg.into_iter().enumerate() {
            out[a + k] = s;
        }
    }
    out.concat()
}

// ---------------------------------------------------------------- vocabulary

/// The recognizer's tokens: class 0 is the CTC blank, class i + 1 is `chars[i]`.
#[derive(Clone, Debug)]
pub struct Vocab {
    pub chars: Vec<String>,
    pub index: std::collections::HashMap<String, usize>,
    has_commands: bool,
    jamo: bool,
}

impl Vocab {
    pub fn new(chars: Vec<String>) -> Self {
        let index = chars.iter().enumerate().map(|(i, c)| (c.clone(), i + 1)).collect::<std::collections::HashMap<_, _>>();
        let has_commands = chars.iter().any(|c| c.chars().count() > 1);
        let jamo = index.contains_key("\u{1100}");
        Vocab { chars, index, has_commands, jamo }
    }

    /// From meta.json's `{"blank": 0, "chars": [...]}` string.
    pub fn from_json(s: &str) -> Result<Self, String> {
        let v: serde_json::Value = serde_json::from_str(s).map_err(|e| e.to_string())?;
        if v.get("blank").and_then(|b| b.as_i64()).unwrap_or(0) != 0 {
            return Err("only blank=0 vocabularies are supported".into());
        }
        let chars = v["chars"]
            .as_array()
            .ok_or("vocab without chars")?
            .iter()
            .map(|c| c.as_str().unwrap_or("").to_string())
            .collect();
        Ok(Vocab::new(chars))
    }

    pub fn space(&self) -> Option<usize> {
        self.index.get(" ").copied()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let toks: Vec<&str> = ids.iter().filter(|&&i| i > 0 && i <= self.chars.len()).map(|&i| self.chars[i - 1].as_str()).collect();
        if self.jamo {
            return nfc(&toks.concat());
        }
        if !self.has_commands {
            return toks.concat();
        }
        let mut out = String::new();
        for (k, t) in toks.iter().enumerate() {
            out.push_str(t);
            let nxt = toks.get(k + 1).copied().unwrap_or("");
            let tc: Vec<char> = t.chars().collect();
            if tc.len() > 2 && tc[0] == '\\' && tc[1..].iter().all(|&c| is_alpha(c)) && nxt.chars().next().map(is_alpha).unwrap_or(false) {
                out.push(' ');
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_is_its_own_inverse() {
        for s in ["שלום world (1)", "abc", "مرحبا 123 بك", "text with עברית inside"] {
            assert_eq!(flip(&flip(s, None), None), s);
        }
        assert_eq!(flip("abc אבג", Some(false)), "abc גבא");
    }

    #[test]
    fn python_classes() {
        assert!(is_alpha('é') && !is_alpha('1') && !is_alpha('_'));
        assert!(is_digit('7') && is_digit('²') && !is_digit('a'));
        assert_eq!(squash_spaces("  a  b\tc "), "a b c");
        assert_eq!(display_len("ab中"), 4);
    }
}
