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

/// Function words: they carry no content of their own, so a sentence made
/// only of them ("Yes, I did.", "That's so cool!") is not a claim by itself.
const STOPWORDS: &[&str] = &[
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be", "been",
    "before", "being", "but", "by", "can", "could", "d", "did", "do", "does", "doing", "don", "done", "for",
    "from", "get", "got", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "just", "ll", "m", "me", "might", "more", "most", "much", "must", "my",
    "myself", "no", "not", "now", "of", "oh", "ok", "okay", "on", "one", "or", "our", "out", "re", "really", "s",
    "so", "some", "still", "such", "sure", "t", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "to", "too", "totally", "up", "us", "ve", "very", "was", "way", "we", "well",
    "were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "yeah", "yes", "yep", "you",
    "your", "yours", "wow", "cool", "great", "awesome", "amazing", "nice", "thanks", "thank", "lol", "haha",
    "hey", "hi", "hello", "glad", "good", "love", "like", "know", "think", "sounds", "sound", "lot", "thing",
    "things", "go", "going", "make", "makes", "made", "say", "said", "see", "let", "let's", "even", "always",
    "never", "maybe", "definitely", "absolutely", "kind", "pretty", "super", "though", "because",
];

/// Words of `s` that carry content (not function words).
fn content_words(s: &str) -> usize {
    words(s)
        .iter()
        .filter(|w| !STOPWORDS.contains(&w.as_str()))
        .count()
}

/// A speaker label at the start of a text ("Caroline: …", "Dr. Ann Lee:
/// …"): one to three capitalised words, then a colon and a space. Returns
/// (label with colon, rest).
pub fn speaker_prefix(text: &str) -> Option<(&str, &str)> {
    let t = text.trim_start();
    let i = t.find(": ")?;
    let label = &t[..i];
    let ws: Vec<&str> = label.split(' ').collect();
    if label.is_empty() || ws.len() > 3 || label.len() > 40 {
        return None;
    }
    let ok = ws.iter().all(|w| {
        w.chars().next().is_some_and(|c| c.is_uppercase())
            && w.chars().all(|c| c.is_alphanumeric() || matches!(c, '.' | '\'' | '-' | '’'))
    });
    ok.then(|| (&t[..i + 1], t[i + 2..].trim_start()))
}

/// How a text is split into claims.
#[derive(Debug, Clone, Copy)]
pub struct SplitRules {
    /// A claim needs at least this many content words; a sentence with
    /// fewer is joined to its neighbour (the next one, or the previous one
    /// at the end).
    pub min_content: usize,
    /// A text with fewer words than this is not split at all.
    pub min_words: usize,
    /// A speaker label ("Name: ") at the start is repeated on every claim.
    pub speaker: bool,
}

impl Default for SplitRules {
    /// The rules `remember` uses (docs/specs/memory-api.md, "Claims"): a
    /// claim needs two content words ("I live in Berlin" has two; "Yes, I
    /// did.", "Cool!" and "I loved it." have none), and a speaker label is
    /// repeated on every claim. Measured on LoCoMo, whose turns are
    /// "Name: text", and on the held-out LongMemEval knowledge-update
    /// questions.
    fn default() -> Self {
        SplitRules {
            min_content: 2,
            min_words: 0,
            speaker: true,
        }
    }
}

/// Split `text` into claims: sentences, list items and lines. Handles
/// abbreviations ("Dr.", "e.g.", "p.m."), initials, decimals, URLs and
/// version numbers (a period not followed by a space never splits),
/// ellipses (split only before a capital), quotes and brackets around the
/// end of a sentence, bullets and hard-wrapped lines. Fragments without a
/// letter or digit are dropped, and a claim repeated in the same text is
/// kept once. A single-sentence text comes back unchanged (trimmed).
pub fn split_claims(text: &str) -> Vec<String> {
    split_claims_with(text, SplitRules::default())
}

/// [`split_claims`] with explicit rules.
pub fn split_claims_with(text: &str, rules: SplitRules) -> Vec<String> {
    let whole = || {
        let t = text.trim();
        if t.is_empty() {
            vec![]
        } else {
            vec![t.to_string()]
        }
    };
    if rules.min_words > 0 && words(text).len() < rules.min_words {
        return whole();
    }
    let (label, body) = match speaker_prefix(text) {
        Some((l, b)) if rules.speaker => (Some(l), b),
        _ => (None, text),
    };
    let mut out: Vec<String> = vec![];
    let mut seen = std::collections::BTreeSet::new();
    for b in blocks(body) {
        for s in sentences(&b) {
            let key = words(&s).join(" ");
            if key.is_empty() || !seen.insert(key) {
                continue;
            }
            out.push(s);
        }
    }
    if rules.min_content > 0 && out.len() > 1 {
        // low-content sentences join their neighbour
        let mut groups: Vec<String> = vec![];
        let mut cur = String::new();
        for s in out {
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(&s);
            if content_words(&cur) >= rules.min_content {
                groups.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            match groups.last_mut() {
                Some(g) => {
                    g.push(' ');
                    g.push_str(&cur);
                }
                None => groups.push(cur),
            }
        }
        out = groups;
    }
    if out.len() <= 1 {
        // one claim: the text as given (only trimmed), so a single
        // statement is stored exactly as before claims existed
        return whole();
    }
    if let Some(l) = label {
        for c in out.iter_mut() {
            *c = format!("{l} {c}");
        }
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
    let claim = speaker_prefix(claim).map_or(claim, |(_, rest)| rest);
    // a claim of several sentences (a fragment joined to its neighbour, or
    // a whole message) states something when any of its sentences does
    let parts: Vec<String> = blocks(claim).iter().flat_map(|b| sentences(b)).collect();
    if parts.len() > 1 {
        return parts.iter().any(|p| sentence_is_declarative(p));
    }
    sentence_is_declarative(claim)
}

/// [`is_declarative`] for one sentence.
fn sentence_is_declarative(claim: &str) -> bool {
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

    /// Sentence splitting alone (no joining, no speaker rule).
    fn sp(s: &str) -> Vec<String> {
        split_claims_with(s, rules(0, 0, false))
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

    fn rules(min_content: usize, min_words: usize, speaker: bool) -> SplitRules {
        SplitRules { min_content, min_words, speaker }
    }

    #[test]
    fn speaker_labels() {
        assert_eq!(speaker_prefix("Caroline: Yes, I did."), Some(("Caroline:", "Yes, I did.")));
        assert_eq!(speaker_prefix("Dr. Ann Lee: Hi."), Some(("Dr. Ann Lee:", "Hi.")));
        assert_eq!(speaker_prefix("My plans: buy a bike"), None);
        assert_eq!(speaker_prefix("I said: no"), None);
        assert_eq!(speaker_prefix("Note this: it rained"), None);
        assert!(!is_declarative("Melanie: How have you been?"));
        assert!(!is_declarative("Melanie: Can you send me the photo."));
        assert!(is_declarative("Melanie: I ran a charity race last Saturday."));
    }

    #[test]
    fn low_content_sentences_join_a_neighbour() {
        let r = rules(4, 0, true);
        // "Yes, I did." has no content of its own: it joins the next sentence
        assert_eq!(
            split_claims_with("Caroline: Yes, I did. I researched adoption agencies last week. The agency staff were helpful and patient.", r),
            vec![
                "Caroline: Yes, I did. I researched adoption agencies last week.",
                "Caroline: The agency staff were helpful and patient."
            ]
        );
        // a trailing fragment joins the previous claim
        assert_eq!(
            split_claims_with("I moved to Lisbon last month with my dog. It is sunny there. Wow!", r),
            vec!["I moved to Lisbon last month with my dog. It is sunny there. Wow!"]
        );
        // all fragments: one claim, the text unchanged
        assert_eq!(split_claims_with("Caroline: Hey Mel! Good to see you!", r), vec!["Caroline: Hey Mel! Good to see you!"]);
        // without a speaker rule the label stays on the first claim only
        assert_eq!(
            split_claims_with("Caroline: I painted a lake at sunrise. My sister bought the painting yesterday.", rules(0, 0, false)),
            vec!["Caroline: I painted a lake at sunrise.", "My sister bought the painting yesterday."]
        );
        // short texts are not split under min_words
        assert_eq!(
            split_claims_with("I live in Porto. My rent is low.", rules(0, 10, false)),
            vec!["I live in Porto. My rent is low."]
        );
    }

    #[test]
    fn default_rules() {
        // a dialog turn: no content-free fragment, the speaker on each claim
        assert_eq!(
            split_claims("Caroline: Yes, I did. I researched adoption agencies. My sister helped me with the forms."),
            vec![
                "Caroline: Yes, I did. I researched adoption agencies.",
                "Caroline: My sister helped me with the forms."
            ]
        );
        assert_eq!(split_claims("Caroline: Cool! I loved it."), vec!["Caroline: Cool! I loved it."]);
        // short facts stay claims of their own
        assert_eq!(split_claims("I live in Berlin. I have a dog named Max."), vec!["I live in Berlin.", "I have a dog named Max."]);
        // a question joined to a statement leaves the claim declarative
        assert!(is_declarative("Can you recommend some? By the way, I switched to 1 tablespoon per 6 ounces."));
        assert!(!is_declarative("Melanie: Wow, really? How was it?"));
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
