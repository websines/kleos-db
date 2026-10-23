/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Claims: `remember` stores multi-sentence text one sentence (claim) at a
//! time (docs/specs/memory-api.md, "Claims"). This module is the rule-based
//! sentence splitter and the test for which claims are declarative enough
//! to take part in supersede / duplicate decisions. Both are model-free and
//! deterministic.

/// Abbreviations that are never the end of a sentence: titles and words
/// that precede a name or a number ("Dr. Smith", "No. 5", "St. Louis").
const ABBR_NEVER: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "ft", "vs", "cf", "al", "approx",
    "appt", "apt", "dept", "est", "fig", "no", "nos", "vol", "vols", "pp", "ave", "blvd", "rd",
    "capt", "col", "lt", "sgt", "gen", "gov", "sen", "rep", "rev", "hon", "messrs", "mme", "mlle",
    "e.g", "i.e", "eg", "ie", "viz", "ca",
];

/// Abbreviations that may also end a sentence ("… at 5 p.m. Then …"): they
/// end one only when the next word starts with a capital letter.
const ABBR_MAYBE: &[&str] = &[
    "etc", "inc", "ltd", "co", "corp", "llc", "plc", "bros", "a.m", "p.m", "am", "pm", "u.s",
    "u.k", "u.s.a", "ph.d", "phd", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
    "min", "max", "hr", "hrs", "sec", "lb", "lbs", "oz", "km", "mi", "kg", "approx",
];

const CLOSERS: &[char] = &['"', '\'', '”', '’', ')', ']', '»', '*', '_'];
const OPENERS: &[char] = &['"', '\'', '“', '‘', '(', '[', '«', '*', '_'];

fn is_terminal(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…' | '。' | '！' | '？')
}

/// A list bullet at the start of a line: `-`, `*`, `•`, `–`, `—`, `1.`,
/// `1)`, `a)` followed by a space. Returns the rest of the line.
fn strip_bullet(line: &str) -> Option<&str> {
    let t = line.trim_start();
    for b in ["- ", "* ", "• ", "– ", "— ", "+ ", "· "] {
        if let Some(r) = t.strip_prefix(b) {
            return Some(r);
        }
    }
    let n = t.chars().take_while(|c| c.is_ascii_digit()).count();
    let rest = &t[n..];
    if (1..=3).contains(&n) {
        for m in [". ", ") "] {
            if let Some(r) = rest.strip_prefix(m) {
                return Some(r);
            }
        }
    }
    let mut cs = t.chars();
    if let (Some(a), Some(')'), Some(' ')) = (cs.next(), cs.next(), cs.next()) {
        if a.is_ascii_alphabetic() {
            return Some(&t[3..]);
        }
    }
    None
}

/// Lines to blocks: a bullet or a blank line starts a new block; a line
/// that continues an unfinished one (it starts in lower case and the
/// previous line has no closing punctuation) is joined to it (hard-wrapped
/// prose).
fn blocks(text: &str) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    let mut open = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            open = false;
            continue;
        }
        if let Some(item) = strip_bullet(line) {
            out.push(item.trim().to_string());
            open = true;
            continue;
        }
        let continues = open
            && line.chars().next().is_some_and(|c| c.is_lowercase())
            && out
                .last()
                .and_then(|b| b.chars().last())
                .is_some_and(|c| !(is_terminal(c) || c == ':' || CLOSERS.contains(&c)));
        if continues {
            let b = out.last_mut().unwrap();
            b.push(' ');
            b.push_str(line);
        } else {
            out.push(line.to_string());
        }
        open = true;
    }
    out
}

/// The word that ends at byte `end` (exclusive), without leading openers.
fn word_before(s: &str, end: usize) -> &str {
    let head = &s[..end];
    let start = head
        .rfind(char::is_whitespace)
        .map(|i| i + head[i..].chars().next().unwrap().len_utf8())
        .unwrap_or(0);
    head[start..].trim_start_matches(OPENERS)
}

/// Sentences of one block.
fn sentences(block: &str) -> Vec<String> {
    let cs: Vec<(usize, char)> = block.char_indices().collect();
    let n = cs.len();
    let mut out = vec![];
    let mut start = 0usize; // byte offset
    let mut i = 0;
    while i < n {
        if !is_terminal(cs[i].1) {
            i += 1;
            continue;
        }
        // the run of terminal punctuation, then closing quotes / brackets
        let mut j = i;
        while j < n && is_terminal(cs[j].1) {
            j += 1;
        }
        let mut k = j;
        while k < n && CLOSERS.contains(&cs[k].1) {
            k += 1;
        }
        // a boundary is followed by whitespace (a decimal, a URL or a
        // version number never is)
        if k >= n || !cs[k].1.is_whitespace() {
            i = j.max(i + 1);
            continue;
        }
        let mut m = k;
        while m < n && cs[m].1.is_whitespace() {
            m += 1;
        }
        if m >= n {
            break;
        }
        let mut q = m;
        while q < n && OPENERS.contains(&cs[q].1) {
            q += 1;
        }
        let next = cs.get(q).map(|x| x.1).unwrap_or(' ');
        let upper = next.is_uppercase();
        let digit = next.is_ascii_digit();
        let run: String = cs[i..j].iter().map(|x| x.1).collect();
        let split = if run == "." {
            let w = word_before(block, cs[i].0);
            let wl = w.to_lowercase();
            let single_letter = w.chars().count() == 1 && w.chars().all(|c| c.is_alphabetic());
            if ABBR_NEVER.contains(&wl.as_str()) {
                false
            } else if single_letter {
                // an initial ("A. Smith"), unless a lower-case word follows
                // a one-letter word that is not a capital ("plan b. Then")
                upper && w.chars().all(|c| c.is_lowercase())
            } else if ABBR_MAYBE.contains(&wl.as_str()) || (wl.contains('.') && wl.len() <= 6) {
                upper
            } else {
                upper || digit
            }
        } else if run.contains('…') || run.starts_with("..") {
            // an ellipsis ends a sentence only before a capital
            upper
        } else {
            upper || digit
        };
        if split {
            let end = cs.get(k).map(|x| x.0).unwrap_or(block.len());
            out.push(block[start..end].trim().to_string());
            start = cs[m].0;
            i = m;
        } else {
            i = k.max(i + 1);
        }
    }
    let tail = block[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    out
}

/// Words for identity (lower case, alphanumeric runs).
fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// Split `text` into claims: sentences, list items and lines. Handles
/// abbreviations ("Dr.", "e.g.", "p.m."), initials, decimals, URLs and
/// version numbers (a period not followed by a space never splits),
/// ellipses (split only before a capital), quotes and brackets around the
/// end of a sentence, bullets and hard-wrapped lines. Fragments without a
/// letter or digit are dropped, and a claim repeated in the same text is
/// kept once. A single-sentence text comes back unchanged (trimmed).
pub fn split_claims(text: &str) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    let mut seen = std::collections::BTreeSet::new();
    for b in blocks(text) {
        for s in sentences(&b) {
            let key = words(&s).join(" ");
            if key.is_empty() || !seen.insert(key) {
                continue;
            }
            out.push(s);
        }
    }
    if out.len() <= 1 {
        // one claim: the text as given (only trimmed), so a single
        // statement is stored exactly as before claims existed
        let t = text.trim();
        return if t.is_empty() {
            vec![]
        } else {
            vec![t.to_string()]
        };
    }
    out
}

/// Openings of requests and questions asked without a question mark.
const REQUEST_STARTS: &[&str] = &[
    "please",
    "can you",
    "could you",
    "would you",
    "will you",
    "can u",
    "do you",
    "did you",
    "are you",
    "have you",
    "is there",
    "are there",
    "is it",
    "should i",
    "can i",
    "could i",
    "may i",
    "shall i",
    "would it",
    "help me",
    "tell me",
    "give me",
    "show me",
    "let me know",
    "remind me",
    "i need help",
    "i need some help",
    "i need advice",
    "i need some advice",
    "i need suggestions",
    "i need some suggestions",
    "i need ideas",
    "i need some ideas",
    "i need recommendations",
    "i need some recommendations",
    "i need tips",
    "i need some tips",
    "i'd like you to",
    "i would like you to",
    "i want you to",
    "i was wondering",
    "i'm wondering",
    "i am wondering",
    "i'm looking for suggestions",
    "i'm looking for recommendations",
    "i'm looking for ideas",
    "i'm looking for tips",
    "i'm looking for advice",
    "i'm looking for some",
    "any suggestions",
    "any tips",
    "any ideas",
    "any recommendations",
    "any advice",
    "suggest",
    "recommend",
    "list",
    "explain",
    "write",
    "provide",
    "find",
    "create",
    "generate",
    "describe",
    "compare",
    "summarize",
    "summarise",
    "draft",
    "plan",
    "brainstorm",
    "what",
    "how",
    "why",
    "which",
    "who",
    "whom",
    "whose",
];

/// Openings that are questions only when the claim has no comma clause
/// ("When I was young, I lived in Paris" is a statement).
const WH_STARTS: &[&str] = &["what", "how", "why", "which", "who", "whom", "whose"];

const COURTESY: &[&str] = &[
    "thanks",
    "thank you",
    "thx",
    "ok",
    "okay",
    "great",
    "sounds good",
    "hi",
    "hello",
    "hey",
    "cool",
    "awesome",
    "perfect",
    "got it",
    "nice",
];

/// Whether a claim states something: it takes part in supersede and
/// duplicate decisions only then. Questions (a `?`, or a question opening),
/// requests to the assistant ("Can you …", "Please …", "Suggest …"),
/// greetings and thanks, and fragments of fewer than three words are not
/// decided; they are still stored and recallable.
pub fn is_declarative(claim: &str) -> bool {
    let t = claim.trim().trim_end_matches(CLOSERS).trim_end();
    if t.ends_with('?') || t.ends_with('？') {
        return false;
    }
    let ws = words(t);
    if ws.len() < 3 {
        return false;
    }
    // apostrophes normalised so "I'm" and "I’m" read alike
    let low: String = t
        .trim_start_matches(OPENERS)
        .to_lowercase()
        .replace('’', "'");
    let starts = |p: &str| {
        low.strip_prefix(p)
            .is_some_and(|r| r.is_empty() || !r.chars().next().unwrap().is_alphanumeric())
    };
    for p in REQUEST_STARTS {
        if starts(p) {
            if WH_STARTS.contains(p) && low.contains(',') {
                continue;
            }
            return false;
        }
    }
    if ws.len() <= 5 && COURTESY.iter().any(|p| starts(p)) {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sp(s: &str) -> Vec<String> {
        split_claims(s)
    }

    #[test]
    fn single_sentence_is_unchanged() {
        assert_eq!(sp("  I live in Berlin.  "), vec!["I live in Berlin."]);
        assert_eq!(
            sp("I stopped playing tennis; now I go climbing on Saturdays."),
            vec!["I stopped playing tennis; now I go climbing on Saturdays."]
        );
        assert_eq!(
            sp("My rent is €1,200 per month."),
            vec!["My rent is €1,200 per month."]
        );
        assert_eq!(sp("no punctuation at all"), vec!["no punctuation at all"]);
        assert!(sp("   ").is_empty());
    }

    #[test]
    fn splits_sentences() {
        assert_eq!(
            sp("I live in Berlin. I work at Siemens! Do you like it? Yes."),
            vec![
                "I live in Berlin.",
                "I work at Siemens!",
                "Do you like it?",
                "Yes."
            ]
        );
    }

    #[test]
    fn abbreviations_and_initials() {
        assert_eq!(
            sp("I saw Dr. Smith today. Mr. and Mrs. Lee came too."),
            vec!["I saw Dr. Smith today.", "Mr. and Mrs. Lee came too."]
        );
        assert_eq!(
            sp("I met A. Smith at St. Mary's. It was fun."),
            vec!["I met A. Smith at St. Mary's.", "It was fun."]
        );
        assert_eq!(
            sp("I like fruit, e.g. apples and pears. They are cheap."),
            vec!["I like fruit, e.g. apples and pears.", "They are cheap."]
        );
        assert_eq!(
            sp("The meeting is at 5 p.m. tomorrow. I will be late."),
            vec!["The meeting is at 5 p.m. tomorrow.", "I will be late."]
        );
        assert_eq!(
            sp("I left at 5 p.m. Then I drove home."),
            vec!["I left at 5 p.m.", "Then I drove home."]
        );
        assert_eq!(
            sp("I moved to the U.S. last year. I like it."),
            vec!["I moved to the U.S. last year.", "I like it."]
        );
        assert_eq!(
            sp("I bought apples, pears etc. My sister ate them."),
            vec!["I bought apples, pears etc.", "My sister ate them."]
        );
        assert_eq!(
            sp("I live at No. 5 Main St. in Boston. It is small."),
            vec!["I live at No. 5 Main St. in Boston.", "It is small."]
        );
    }

    #[test]
    fn decimals_urls_versions() {
        assert_eq!(
            sp("My GPA is 3.85 now. It was 3.5 before."),
            vec!["My GPA is 3.85 now.", "It was 3.5 before."]
        );
        assert_eq!(
            sp("I use https://example.com/a.b?x=1 daily. It is great."),
            vec!["I use https://example.com/a.b?x=1 daily.", "It is great."]
        );
        assert_eq!(
            sp("I upgraded to v2.1.3 today. Nothing broke."),
            vec!["I upgraded to v2.1.3 today.", "Nothing broke."]
        );
        assert_eq!(
            sp("Email me at bob.smith@mail.com. Thanks."),
            vec!["Email me at bob.smith@mail.com.", "Thanks."]
        );
        assert_eq!(
            sp("It cost $3. 5 people came."),
            vec!["It cost $3.", "5 people came."]
        );
    }

    #[test]
    fn quotes_and_ellipses() {
        assert_eq!(
            sp("She said \"I quit.\" Then she left."),
            vec!["She said \"I quit.\"", "Then she left."]
        );
        assert_eq!(
            sp("\"Is it raining?\" she asked."),
            vec!["\"Is it raining?\" she asked."]
        );
        assert_eq!(
            sp("I was thinking... maybe not. Anyway I moved."),
            vec!["I was thinking... maybe not.", "Anyway I moved."]
        );
        assert_eq!(
            sp("Well… I moved to Lisbon. It is sunny."),
            vec!["Well…", "I moved to Lisbon.", "It is sunny."]
        );
        assert_eq!(
            sp("I waited… Then it rained."),
            vec!["I waited…", "Then it rained."]
        );
        assert_eq!(
            sp("(I live in Porto.) My rent is low."),
            vec!["(I live in Porto.)", "My rent is low."]
        );
    }

    #[test]
    fn bullets_and_lines() {
        let t =
            "My plans:\n- buy a bike\n- learn Spanish\n\n1. Call mom\n2) Pay rent\na) water plants";
        assert_eq!(
            sp(t),
            vec![
                "My plans:",
                "buy a bike",
                "learn Spanish",
                "Call mom",
                "Pay rent",
                "water plants"
            ]
        );
        // hard-wrapped prose is joined; a capitalised line is its own
        assert_eq!(
            sp("I moved to Lisbon\nlast month with my dog.\nMy rent is low"),
            vec![
                "I moved to Lisbon last month with my dog.",
                "My rent is low"
            ]
        );
    }

    #[test]
    fn repeats_and_empty_fragments() {
        assert_eq!(
            sp("I live in Porto. I live in Porto! ... It is nice."),
            vec!["I live in Porto.", "It is nice."]
        );
    }

    #[test]
    fn declarative() {
        for t in [
            "I live in Berlin.",
            "Currently, my favourite is Kansas City Masterpiece BBQ sauce, but I might switch.",
            "I need to stock up on my favorite BBQ sauce, Sweet Baby Ray's.",
            "When I was young, I lived in Paris.",
            "I'm vegan.",
            "Sam and I got engaged.",
        ] {
            assert!(is_declarative(t), "{t}");
        }
        for t in [
            "Where do I live?",
            "Do you have any tips?\"",
            "Can you suggest a recipe for ribs.",
            "Please list five BBQ sauces.",
            "Suggest a good sauce for chicken",
            "What would go well with it",
            "Thanks a lot!",
            "Ok.",
            "Great idea.",
            "I need some suggestions for a party.",
            "I'm looking for some ideas for dinner.",
            "Any tips on smoking brisket.",
        ] {
            assert!(!is_declarative(t), "{t}");
        }
    }
}
