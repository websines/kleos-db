/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Automatic entities for `remember` (docs/specs/memory-api.md, "Automatic
//! entities"): run an entity extractor once per statement, keep the spans the
//! policy accepts, and turn them into the entity names each claim mentions,
//! relative nodes with kinship/role edges, and the raw spans to store.
//!
//! Policy (docs/research/gliner-usage.md, (a)):
//! * A `Name:` speaker prefix is stripped before extraction; first-person
//!   words then mean that speaker, else `user` (for statements the user made).
//! * Per-label thresholds on the raw score. gliner2 lowercases its input, so
//!   case is read from the original text: in a statement with mixed case a
//!   person, location or organization must contain a capital letter (and not
//!   be a pronoun); in an all-lowercase statement the score must instead
//!   reach [`LOWERCASE_MIN`]. `relative` spans are exempt from the case rule.
//! * A relative ("my sister") is a person node keyed by its possessor and
//!   relation (`user's sister`), with an edge `user -[sister]-> node`. Only
//!   first-person forms ("my", "our", a bare "Mom") belong to the speaker;
//!   "his/her/their X" belongs to the nearest person mentioned before it, and
//!   is skipped when there is none; "your X" is skipped. A name next to the
//!   relative ("my sister Anna", "Anna, my sister", "my sister (Anna)") is
//!   the same person: the name is merged into the node.
//! * The subject (`user` or the speaker) is a claim's entity when the claim
//!   has a first-person subject word, or a first-person possessive that is
//!   not part of a relative ("my car", not "my sister").
//! * A claim with no person or relative of its own and a third-person
//!   pronoun ("she moved") inherits the nearest person or relative mentioned
//!   before it in the same statement. No coreference model, no cross-turn
//!   resolution (that is entity resolution's job, by key and name).

use std::collections::BTreeMap;
use std::sync::OnceLock;

use miette::Result;
use regex::Regex;

use crate::extract::{EntityExtractor, ExtractSpan};

/// Default per-label thresholds of the built-in GLiNER fine-tune, tuned on
/// its validation split with the entity-only prompt (person 0.8, location
/// 0.9, organization 0.8, relative 0.5).
pub const DEFAULT_THRESHOLDS: [(&str, f32); 4] =
    [("person", 0.8), ("location", 0.9), ("organization", 0.8), ("relative", 0.5)];
/// Minimum score of a name in an all-lowercase statement (no case evidence).
pub const LOWERCASE_MIN: f32 = 0.9;
/// Labels whose spans are names (case rule, entity names).
const NAME_LABELS: [&str; 3] = ["person", "location", "organization"];

const PRONOUNS: [&str; 18] = [
    "i", "me", "my", "mine", "we", "us", "our", "you", "your", "he", "him", "his", "she", "her", "they", "them", "their",
    "it",
];

fn re(cell: &'static OnceLock<Regex>, pat: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pat).expect("memory_extract regex"))
}

fn first_person_subject() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"(?i)\b(i|i'm|i’m|i've|i’ve|i'd|i’d|i'll|i’ll|me|myself|we|we're|we’re|we've|we’ve|we'd|we’d|we'll|we’ll|us|ourselves)\b")
}

fn first_person_possessive() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"(?i)\b(my|our|mine|ours)\b")
}

fn third_person() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"(?i)\b(he|she|they|him|her|them|his|hers|their|theirs)\b")
}

fn speaker_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"^\s*(\p{Lu}[\p{L}\p{M}'’.\- ]{0,40}?)\s*:\s+")
}

const NOT_SPEAKERS: [&str; 16] = [
    "note", "update", "edit", "ps", "re", "fyi", "important", "warning", "todo", "q", "a", "question", "answer",
    "reminder", "tip", "summary",
];

/// A `Name:` speaker prefix: `(name, byte offset of the body)`.
pub fn speaker_prefix(text: &str) -> Option<(String, usize)> {
    let c = speaker_re().captures(text)?;
    let name = c.get(1)?.as_str().trim();
    if name.split_whitespace().count() > 4 || NOT_SPEAKERS.contains(&name.to_lowercase().as_str()) {
        return None;
    }
    Some((name.to_string(), c.get(0)?.end()))
}

/// The relation a `relative` span names, normalised to one lower-case word
/// (`"my older sister"` → `sister`, `"Mom"` → `mother`), and its possessor:
/// `first` ("my", "our", or a bare kin word), `third` ("his", "her",
/// "their"), `second` ("your") or `none`.
pub fn relation_of(span: &str) -> Option<(String, &'static str)> {
    let words: Vec<String> = span
        .split(|c: char| !(c.is_alphanumeric() || c == '\'' || c == '’'))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect();
    let first = words.first()?;
    let (possessor, rest) = match first.as_str() {
        "my" | "our" => ("first", &words[1..]),
        "his" | "her" | "their" => ("third", &words[1..]),
        "your" => ("second", &words[1..]),
        _ => ("none", &words[..]),
    };
    let phrase = rest.join(" ");
    let canon = |p: &str| -> Option<&'static str> {
        Some(match p {
            "mom" | "mum" | "mommy" | "mummy" | "mama" | "ma" | "mother" => "mother",
            "dad" | "daddy" | "papa" | "pa" | "father" | "pop" | "pops" => "father",
            "grandma" | "granny" | "nana" | "grandmother" | "gran" => "grandmother",
            "grandpa" | "grandad" | "granddad" | "grandfather" => "grandfather",
            "best friend" | "bestie" | "bff" => "bestfriend",
            "boyfriend" | "bf" => "boyfriend",
            "girlfriend" | "gf" => "girlfriend",
            "hubby" | "husband" => "husband",
            "wifey" | "wife" => "wife",
            "fiance" | "fiancé" | "fiancee" | "fiancée" => "fiance",
            "step mom" | "stepmom" | "stepmother" => "stepmother",
            "step dad" | "stepdad" | "stepfather" => "stepfather",
            "mother in law" | "mother-in-law" => "motherinlaw",
            "father in law" | "father-in-law" => "fatherinlaw",
            _ => return None,
        })
    };
    let rel = match canon(&phrase) {
        Some(r) => r.to_string(),
        None => {
            // the head noun: the last word ("my older sister" → sister)
            let last = rest.last()?;
            canon(last).map(str::to_string).unwrap_or_else(|| last.trim_matches(|c| c == '\'' || c == '’').to_string())
        }
    };
    if rel.is_empty() || PRONOUNS.contains(&rel.as_str()) {
        return None;
    }
    // a bare word is the speaker's only for kin addressed by role ("Mom")
    let possessor = if possessor == "none" {
        if canon(&phrase).is_some() && matches!(rel.as_str(), "mother" | "father" | "grandmother" | "grandfather") {
            "first"
        } else {
            "none"
        }
    } else {
        possessor
    };
    Some((rel, possessor))
}

/// The entity name of a relative node: `<possessor>'s <relation>`
/// (`user's sister`, `Caroline's mother`).
pub fn relative_node_name(possessor: &str, relation: &str) -> String {
    format!("{possessor}'s {relation}")
}

/// One extracted span as stored (`kleos_mem_extract`).
#[derive(Debug, Clone)]
pub(crate) struct AutoSpan {
    /// Claim it lies in, if the claims could be placed in the statement.
    pub claim: Option<usize>,
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub score: f32,
    /// Accepted by the policy.
    pub kept: bool,
    /// Index into `names` of the entity it became.
    pub name: Option<usize>,
}

/// A kinship or role edge `src -[rel]-> dst` between two entities of `names`.
#[derive(Debug, Clone)]
pub(crate) struct AutoEdge {
    pub src: usize,
    pub rel: String,
    pub dst: usize,
    pub p: f32,
}

/// What the extractor and the policy made of one statement.
#[derive(Debug, Clone, Default)]
pub(crate) struct AutoEntities {
    /// Entity names to resolve, unique by normalised name.
    pub names: Vec<String>,
    /// Per claim: indices into `names`.
    pub claim_names: Vec<Vec<usize>>,
    pub spans: Vec<AutoSpan>,
    pub edges: Vec<AutoEdge>,
    /// `(relative node, name)` pairs that are the same person.
    pub same: Vec<(usize, usize, f32)>,
    /// The extractor's version, stored with the spans.
    pub version: String,
}

impl AutoEntities {
    pub(crate) fn name(&mut self, n: &str) -> usize {
        let k = crate::memory::norm_name(n);
        if let Some(i) = self.names.iter().position(|x| crate::memory::norm_name(x) == k) {
            return i;
        }
        self.names.push(n.to_string());
        self.names.len() - 1
    }
}

/// Byte ranges of the claims in the statement (`None` where a claim could
/// not be found, e.g. text the splitter joined or rewrote).
pub(crate) fn claim_ranges(statement: &str, claims: &[String], speaker: Option<&str>) -> Vec<Option<(usize, usize)>> {
    let mut out = vec![];
    let mut cursor = 0;
    for c in claims {
        let mut t = c.as_str();
        if let Some(sp) = speaker {
            if let Some(rest) = t.strip_prefix(sp) {
                t = rest.trim_start_matches(':').trim_start();
            }
        }
        let t = t.trim();
        let hit = statement[cursor..]
            .find(t)
            .map(|o| cursor + o)
            .or_else(|| statement.find(t));
        match hit {
            Some(a) if !t.is_empty() => {
                out.push(Some((a, a + t.len())));
                cursor = a + t.len();
            }
            _ => out.push(None),
        }
    }
    out
}

fn contains(outer: (usize, usize), inner: (usize, usize)) -> bool {
    outer.0 <= inner.0 && inner.1 <= outer.1
}

/// Run `ex` over `statement` and apply the policy. `claims` are the texts
/// `remember` stores (the claim splitter's output, or the statement);
/// `subject` is who first-person words refer to when the statement has no
/// speaker prefix (`user` for the user's own statements, `None` otherwise).
pub(crate) fn auto_entities(
    ex: &dyn EntityExtractor,
    statement: &str,
    claims: &[String],
    subject: Option<&str>,
    thresholds: &BTreeMap<String, f32>,
) -> Result<AutoEntities> {
    let mut out = AutoEntities {
        claim_names: vec![vec![]; claims.len()],
        version: ex.version(),
        ..Default::default()
    };
    let (speaker, body_at) = match speaker_prefix(statement) {
        Some((n, at)) => (Some(n), at),
        None => (None, 0),
    };
    let body = &statement[body_at..];
    if body.trim().is_empty() {
        return Ok(out);
    }
    let subject: Option<String> = speaker.clone().or_else(|| subject.map(str::to_string));
    let found: Vec<ExtractSpan> = ex.extract(&[body])?.into_iter().next().unwrap_or_default();
    let ranges = claim_ranges(statement, claims, speaker.as_deref());
    let claim_of = |a: usize, b: usize| -> Option<usize> {
        if claims.len() == 1 {
            return Some(0);
        }
        ranges.iter().position(|r| r.is_some_and(|r| contains(r, (a, b)) || (a >= r.0 && a < r.1)))
    };
    let mixed = body.chars().any(|c| c.is_uppercase()) && body.chars().any(|c| c.is_lowercase());
    let thr = |label: &str| -> f32 {
        thresholds
            .get(label)
            .copied()
            .or_else(|| DEFAULT_THRESHOLDS.iter().find(|(l, _)| *l == label).map(|x| x.1))
            .unwrap_or(0.5)
    };

    // 1. accept spans
    for s in found {
        let (a, b) = (s.start + body_at, s.end + body_at);
        let text = statement[a..b].trim().to_string();
        let low = text.to_lowercase();
        let is_name = NAME_LABELS.contains(&s.label.as_str());
        let mut kept = s.score >= thr(&s.label) && !PRONOUNS.contains(&low.as_str()) && !text.is_empty();
        if kept && is_name {
            kept = if mixed {
                text.chars().any(|c| c.is_uppercase())
            } else {
                s.score >= thr(&s.label).max(LOWERCASE_MIN)
            };
        }
        out.spans.push(AutoSpan {
            claim: claim_of(a, b),
            label: s.label,
            start: a,
            end: b,
            text,
            score: s.score,
            kept,
            name: None,
        });
    }
    out.spans.sort_by_key(|s| (s.start, s.end));

    // 2. relatives that contain a person: the relative is the rest
    //    ("my sister Anna" → "my sister" + Anna)
    let persons: Vec<(usize, usize)> = out
        .spans
        .iter()
        .filter(|s| s.kept && s.label == "person")
        .map(|s| (s.start, s.end))
        .collect();
    // (relative span index, person span) pairs
    let mut pairs: Vec<(usize, (usize, usize))> = vec![];
    for (i, s) in out.spans.iter().enumerate() {
        if !(s.kept && s.label == "relative") {
            continue;
        }
        if let Some(p) = persons.iter().find(|p| contains((s.start, s.end), **p) && **p != (s.start, s.end)) {
            pairs.push((i, *p));
            continue;
        }
        // adjacency: "my sister Anna", "my sister, Anna", "my sister (Anna)",
        // "Anna, my sister"
        let after = persons.iter().find(|p| {
            p.0 >= s.end && {
                let gap = statement[s.end..p.0].trim();
                matches!(gap, "" | "," | "(" | "-" | "–" | "—" | "named" | "called")
            }
        });
        let before = persons.iter().find(|p| {
            p.1 <= s.start && {
                let gap = statement[p.1..s.start].trim();
                matches!(gap, "," | "(" | "is" | "-" | "–" | "—")
            }
        });
        if let Some(p) = after.or(before) {
            pairs.push((i, *p));
        }
    }

    // 3. entities: names, then relative nodes (in text order, so a
    //    third-person possessor is the nearest earlier person)
    let mut mention_pos: Vec<(usize, usize, bool)> = vec![]; // (start, name idx, person-like)
    for i in 0..out.spans.len() {
        let s = out.spans[i].clone();
        if !s.kept {
            continue;
        }
        if NAME_LABELS.contains(&s.label.as_str()) {
            let n = out.name(&s.text);
            out.spans[i].name = Some(n);
            mention_pos.push((s.start, n, s.label == "person"));
            continue;
        }
        // relative
        let pair = pairs.iter().find(|(r, _)| *r == i).map(|(_, p)| *p);
        let rel_text = match pair {
            Some(p) if contains((s.start, s.end), p) => {
                let (ra, rb) = (s.start, s.end);
                format!("{} {}", &statement[ra..p.0], &statement[p.1..rb])
            }
            _ => s.text.clone(),
        };
        let Some((rel, possessor)) = relation_of(rel_text.trim().trim_matches(|c: char| c == ',' || c == '(')) else {
            continue;
        };
        let owner: Option<usize> = match possessor {
            "first" => subject.as_deref().map(|sub| out.name(sub)),
            "third" => mention_pos.iter().rev().find(|m| m.0 < s.start && m.2).map(|m| m.1),
            _ => None,
        };
        let Some(owner) = owner else { continue };
        let owner_name = out.names[owner].clone();
        let node = out.name(&relative_node_name(&owner_name, &rel));
        out.spans[i].name = Some(node);
        mention_pos.push((s.start, node, true));
        let mut p = s.score;
        if let Some(ps) = pair {
            if let Some(pi) = out.spans.iter().position(|x| (x.start, x.end) == ps) {
                let pn = out.spans[pi].name.unwrap_or_else(|| out.name(&out.spans[pi].text.clone()));
                p = p.min(out.spans[pi].score);
                out.same.push((node, pn, p));
            }
        }
        out.edges.push(AutoEdge { src: owner, rel, dst: node, p });
    }

    // 4. per claim: its mentions, the subject, inherited pronoun referents
    let rel_ranges: Vec<(usize, usize)> = out
        .spans
        .iter()
        .filter(|s| s.kept && s.label == "relative" && s.name.is_some())
        .map(|s| (s.start, s.end))
        .collect();
    for ci in 0..claims.len() {
        let mut mine: Vec<usize> = vec![];
        let mut has_person = false;
        for s in out.spans.iter().filter(|s| s.kept && s.claim == Some(ci)) {
            if let Some(n) = s.name {
                if !mine.contains(&n) {
                    mine.push(n);
                }
                // a paired name belongs with its node
                for (node, name, _) in &out.same {
                    for (x, y) in [(*node, *name), (*name, *node)] {
                        if x == n && !mine.contains(&y) {
                            mine.push(y);
                        }
                    }
                }
                has_person |= s.label == "person" || s.label == "relative";
            }
        }
        // the claim's own text (speaker label removed) for the word rules
        let (ctext, coff) = match ranges.get(ci).copied().flatten() {
            Some((a, b)) => (&statement[a..b], a),
            None => (claims[ci].as_str(), usize::MAX),
        };
        if let Some(sub) = subject.as_deref() {
            let subj_word = first_person_subject().is_match(ctext);
            let own_possessive = first_person_possessive().find_iter(ctext).any(|m| {
                coff == usize::MAX || !rel_ranges.iter().any(|r| contains(*r, (coff + m.start(), coff + m.end())))
            });
            if subj_word || own_possessive {
                let si = out.name(sub);
                if !mine.contains(&si) {
                    mine.push(si);
                }
            }
        }
        if !has_person && third_person().is_match(ctext) {
            let start = ranges.get(ci).copied().flatten().map(|r| r.0).unwrap_or(0);
            if let Some(m) = mention_pos.iter().rev().find(|m| m.0 < start && m.2) {
                if !mine.contains(&m.1) {
                    mine.push(m.1);
                }
            }
        }
        out.claim_names[ci] = mine;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(Vec<(&'static str, &'static str, f32)>);

    impl EntityExtractor for Fake {
        fn model_id(&self) -> String {
            "fake".into()
        }
        fn version(&self) -> String {
            "fake:1".into()
        }
        fn labels(&self) -> Vec<String> {
            vec!["person".into(), "location".into(), "organization".into(), "relative".into()]
        }
        fn extract(&self, texts: &[&str]) -> Result<Vec<Vec<ExtractSpan>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let lt = t.to_lowercase();
                    self.0
                        .iter()
                        .filter_map(|(label, span, score)| {
                            let a = lt.find(&span.to_lowercase())?;
                            Some(ExtractSpan {
                                label: label.to_string(),
                                start: a,
                                end: a + span.len(),
                                text: t[a..a + span.len()].to_string(),
                                score: *score,
                            })
                        })
                        .collect()
                })
                .collect())
        }
    }

    fn run(spans: Vec<(&'static str, &'static str, f32)>, text: &str, claims: &[&str]) -> AutoEntities {
        let claims: Vec<String> = claims.iter().map(|s| s.to_string()).collect();
        auto_entities(&Fake(spans), text, &claims, Some("user"), &BTreeMap::new()).unwrap()
    }

    fn names_of(a: &AutoEntities, ci: usize) -> Vec<String> {
        let mut v: Vec<String> = a.claim_names[ci].iter().map(|&i| a.names[i].clone()).collect();
        v.sort();
        v
    }

    #[test]
    fn relations_and_possessors() {
        assert_eq!(relation_of("my sister"), Some(("sister".into(), "first")));
        assert_eq!(relation_of("My older sister"), Some(("sister".into(), "first")));
        assert_eq!(relation_of("Mom"), Some(("mother".into(), "first")));
        assert_eq!(relation_of("our dad"), Some(("father".into(), "first")));
        assert_eq!(relation_of("his wife"), Some(("wife".into(), "third")));
        assert_eq!(relation_of("your boss"), Some(("boss".into(), "second")));
        assert_eq!(relation_of("my best friend"), Some(("bestfriend".into(), "first")));
        assert_eq!(relation_of("a coworker"), Some(("coworker".into(), "none")));
        assert_eq!(relation_of("my"), None);
        assert_eq!(speaker_prefix("Caroline: hi there").map(|x| x.0), Some("Caroline".into()));
        assert_eq!(speaker_prefix("Update: I moved"), None);
        assert_eq!(speaker_prefix("no speaker here"), None);
    }

    #[test]
    fn sister_is_a_node_not_the_user() {
        let a = run(
            vec![("relative", "My sister", 0.9), ("location", "Berlin", 0.95)],
            "My sister moved to Berlin.",
            &["My sister moved to Berlin."],
        );
        assert_eq!(names_of(&a, 0), vec!["Berlin", "user's sister"]);
        assert_eq!(a.edges.len(), 1);
        assert_eq!(a.names[a.edges[0].src], "user");
        assert_eq!(a.edges[0].rel, "sister");
        let b = run(vec![("location", "Paris", 0.95)], "I live in Paris.", &["I live in Paris."]);
        assert_eq!(names_of(&b, 0), vec!["Paris", "user"]);
    }

    #[test]
    fn name_next_to_relative_merges_and_pronouns_inherit() {
        let text = "My sister Anna got a new job. She moved to Lisbon.";
        let a = run(
            vec![("relative", "My sister Anna", 0.6), ("person", "Anna", 0.95), ("location", "Lisbon", 0.97)],
            text,
            &["My sister Anna got a new job.", "She moved to Lisbon."],
        );
        assert_eq!(names_of(&a, 0), vec!["Anna", "user's sister"]);
        assert_eq!(a.same.len(), 1);
        // "She" inherits the nearest earlier person (the sister node)
        assert!(names_of(&a, 1).contains(&"Lisbon".to_string()));
        assert!(names_of(&a, 1).iter().any(|n| n == "user's sister" || n == "Anna"));
        assert!(!names_of(&a, 1).contains(&"user".to_string()));
    }

    #[test]
    fn third_person_relatives_and_case_rules() {
        let a = run(
            vec![("person", "Tom", 0.95), ("relative", "his wife", 0.8)],
            "Tom said his wife is a nurse.",
            &["Tom said his wife is a nurse."],
        );
        assert!(a.names.contains(&"Tom's wife".to_string()));
        assert_eq!(a.names[a.edges[0].src], "Tom");
        // a lower-case "name" in a mixed-case text is dropped; in an
        // all-lowercase text it needs LOWERCASE_MIN
        let b = run(vec![("person", "farmers", 0.85)], "The farmers market was fun.", &["The farmers market was fun."]);
        assert!(b.names.is_empty());
        let c = run(vec![("person", "anna", 0.85)], "anna called me", &["anna called me"]);
        assert_eq!(names_of(&c, 0), vec!["user"]);
        let d = run(vec![("person", "anna", 0.95)], "anna called me", &["anna called me"]);
        assert_eq!(names_of(&d, 0), vec!["anna", "user"]);
        // "your boss" belongs to no one we know
        let e = run(vec![("relative", "your boss", 0.9)], "How is your boss?", &["How is your boss?"]);
        assert!(e.edges.is_empty());
    }

    #[test]
    fn speaker_prefix_is_the_subject() {
        let a = run(
            vec![("relative", "my mom", 0.9), ("person", "Caroline", 0.4)],
            "Caroline: I visited my mom yesterday.",
            &["Caroline: I visited my mom yesterday."],
        );
        assert_eq!(names_of(&a, 0), vec!["Caroline", "Caroline's mother"]);
        assert_eq!(a.edges[0].rel, "mother");
    }
}
