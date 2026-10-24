/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Claim typing against the active ontology (docs/specs/ontology.md, "Claim
//! typing") and the cardinality supersession policy ("Supersession").
//!
//! A claim is small talk, typed (a property of the active ontology, by cue
//! phrases and embedding prototypes) or untyped; its subject is the speaker,
//! `user`, a relative of the speaker (`user.sister`) or a leading name. The
//! policy then says, per (new claim, candidate) pair, whether `remember` may
//! ask `supersedes` and `duplicate`.

use serde_derive::{Deserialize, Serialize};

use crate::memory_claims::speaker_prefix;
use crate::ontology::Active;

/// What the typer concluded about a claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimType {
    /// `fact`, `smalltalk` or `untyped`.
    pub cls: String,
    pub subject: Option<String>,
    pub prop: Option<String>,
    pub score: f64,
    /// `cue`, `proto`, `cue+proto`, `rule` or an extractor's name.
    pub by: String,
}

impl ClaimType {
    pub fn is_smalltalk(&self) -> bool {
        self.cls == "smalltalk"
    }
}

/// Minimum cosine to a property prototype.
pub const PROTO_MIN: f64 = 0.45;
/// Margin over the runner-up property.
pub const PROTO_MARGIN: f64 = 0.02;
/// Score bonus of a property whose cue phrase occurs in the claim.
pub const CUE_BONUS: f64 = 0.06;

const FIRST_PERSON: &[&str] = &["i", "i'm", "i've", "i'd", "i'll", "me", "my", "mine", "myself", "we", "we're", "we've", "our", "ours", "us"];
const SECOND_PERSON: &[&str] = &["you", "you're", "you've", "you'd", "you'll", "your", "yours", "yourself", "u", "ur"];

/// Openers of reactions, greetings, thanks, compliments and sympathy.
const REACTION_STARTS: &[&str] = &[
    "wow", "whoa", "woah", "oh", "ooh", "aw", "aww", "awww", "omg", "oh my", "haha", "hah", "lol", "yay", "yeah", "yep", "yes",
    "no way", "nah", "nope", "cool", "nice", "great", "awesome", "amazing", "fantastic", "wonderful", "lovely", "sweet",
    "congrats", "congratulations", "thanks", "thank you", "thx", "sorry", "glad", "hey", "hi", "hello", "good luck",
    "take care", "sounds", "that sounds", "that's", "that is", "that was", "looks like", "it looks", "totally", "absolutely",
    "of course", "sure", "definitely", "me too", "same here", "same", "right", "true", "exactly", "no worries", "bye",
    "see you", "talk soon", "can't wait", "good to hear", "great to hear", "nice to hear", "happy to hear", "sad to hear",
    "so sorry", "so glad", "so happy", "so cool", "how cool", "how nice", "what a", "good for you", "well done", "keep it up",
    "you got this", "you've got this", "hope you", "hope your", "i hope you", "agreed", "indeed", "ugh", "oof", "yikes", "phew",
    "look at this", "check this out", "check it out", "here's", "here is", "by the way, take a look", "take a look",
];

/// First-person clauses that are courtesy, not facts.
const COURTESY_FIRST: &[&str] = &[
    "i agree", "i totally agree", "i completely agree", "i'm so happy for you", "i'm happy for you", "i'm so glad",
    "i'm glad", "i'm so proud of you", "i'm proud of you", "i'm sorry", "i'm so sorry", "i hope you", "i hope your", "i hope so", "i hope it", "i hope that", "i can't wait for you",
    "i bet", "i love that", "i love this", "i love it", "i love your", "i'd love", "i'm here for you", "i appreciate",
    "i know what you mean", "i feel you", "i understand", "means a lot to me", "mean a lot to me", "i'm excited for you",
    "i'm thrilled for you", "i'm rooting for you", "i'm with you", "i see", "i guess", "i think so", "i hear you",
    "i'm sure you", "i wish you", "let me know", "i'll talk to you", "i miss you", "i'm curious", "i wonder",
    "i could help", "i'm happy to help", "i can help", "i'm here", "i'm all ears", "i'd say", "i mean",
];

const KIN: &[(&str, &str)] = &[
    ("sister", "sister_of"),
    ("sis", "sister_of"),
    ("brother", "brother_of"),
    ("bro", "brother_of"),
    ("mom", "mother_of"),
    ("mum", "mother_of"),
    ("mother", "mother_of"),
    ("dad", "father_of"),
    ("father", "father_of"),
    ("son", "parent_of"),
    ("daughter", "parent_of"),
    ("wife", "partner"),
    ("husband", "partner"),
    ("partner", "partner"),
    ("boyfriend", "partner"),
    ("girlfriend", "partner"),
    ("fiance", "partner"),
    ("fiancee", "partner"),
    ("boss", "boss_of"),
    ("manager", "boss_of"),
    ("friend", "friend_of"),
    ("grandma", "grandparent_of"),
    ("grandpa", "grandparent_of"),
    ("grandmother", "grandparent_of"),
    ("grandfather", "grandparent_of"),
];

/// Words of a text, lower case, apostrophes normalised, punctuation trimmed.
fn words(t: &str) -> Vec<String> {
    t.replace('’', "'")
        .split(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | ':' | '!' | '?' | '.' | '(' | ')' | '"' | '—' | '–'))
        .map(|w| w.trim_matches(|c: char| !(c.is_alphanumeric() || c == '\'')).to_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}

/// `text` without a speaker label and without photo captions.
fn body(text: &str) -> (Option<String>, String) {
    let (spk, rest) = match speaker_prefix(text) {
        Some((label, rest)) => (Some(label.trim_end_matches(':').to_string()), rest),
        None => (None, text),
    };
    // "[shares a photo: …]" and similar bracketed captions
    let mut out = String::new();
    let mut depth = 0;
    for c in rest.chars() {
        match c {
            '[' => depth += 1,
            ']' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    (spk, out.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// Lower-cased `s` starts with phrase `p` at a word boundary.
fn starts_phrase(s: &str, p: &str) -> bool {
    s.strip_prefix(p).is_some_and(|r| r.is_empty() || !r.chars().next().unwrap().is_alphanumeric())
}

/// Phrase `p` occurs in lower-cased `s` at word boundaries (a cue ending in
/// a letter may be a stem: "renovat" matches "renovating").
pub fn has_phrase(s: &str, p: &str) -> bool {
    let p = p.trim();
    if p.is_empty() {
        return false;
    }
    let mut from = 0;
    while let Some(i) = s[from..].find(p) {
        let at = from + i;
        let before = s[..at].chars().next_back();
        let ok_before = before.map_or(true, |c| !c.is_alphanumeric());
        if ok_before {
            // stems of 6+ letters may continue ("renovat" → "renovating")
            let after = s[at + p.len()..].chars().next();
            let ok_after = after.map_or(true, |c| !c.is_alphanumeric()) || p.len() >= 6;
            if ok_after {
                return true;
            }
        }
        from = at + p.len().max(1);
        if from >= s.len() {
            break;
        }
    }
    false
}

/// Small talk: a reaction, greeting, thanks, compliment or sympathy, a
/// remark about the listener, or a photo-only turn. Stored and recallable,
/// never decided.
pub fn smalltalk(text: &str) -> bool {
    let (_, b) = body(text);
    let low = b.to_lowercase().replace('’', "'");
    let low = low.trim_start_matches(|c: char| !c.is_alphanumeric());
    let ws = words(low);
    if ws.len() < 3 {
        // "Thoughts?", "Look at this!", a caption only
        return true;
    }
    let first: Vec<usize> = ws.iter().enumerate().filter(|(_, w)| FIRST_PERSON.contains(&w.as_str())).map(|(i, _)| i).collect();
    let second = ws.iter().any(|w| SECOND_PERSON.contains(&w.as_str()));
    // the first-person clauses: does any state something (not courtesy)?
    let factual_first = !first.is_empty() && {
        let mut rest = low.to_string();
        for c in COURTESY_FIRST {
            rest = rest.replace(c, " ");
        }
        words(&rest).iter().any(|w| FIRST_PERSON.contains(&w.as_str()))
    };
    // every first-person clause is courtesy ("I totally agree, Melanie.")
    if !first.is_empty() && !factual_first {
        return true;
    }
    let reaction = REACTION_STARTS.iter().any(|p| starts_phrase(low, p));
    if reaction && !factual_first {
        return true;
    }
    // about the listener only ("Looks like you had a great day", "It's great
    // that you're helping out")
    if second && !factual_first {
        return true;
    }
    false
}

const PRONOUN_STARTS: &[&str] = &[
    "i", "we", "you", "he", "she", "they", "it", "this", "that", "there", "the", "a", "an", "my", "our", "your", "his", "her",
    "their", "so", "and", "but", "also", "then", "when", "if", "just", "today", "yesterday", "last", "next", "recently",
];

/// The claim's subject: the speaker (or `user`) for first person,
/// `<speaker|user>.<kin>` for "my sister …", a leading proper name, or
/// `None`.
pub fn subject(text: &str) -> Option<String> {
    let (spk, b) = body(text);
    let me = spk.as_deref().map(norm).unwrap_or_else(|| "user".to_string());
    let ws = words(&b);
    if ws.is_empty() {
        return None;
    }
    // "my sister (Anna) …" as the sentence subject
    if ws[0] == "my" || ws[0] == "our" {
        if let Some(k) = ws.get(1) {
            let k = k.trim_end_matches("'s");
            if let Some((kin, _)) = KIN.iter().find(|(w, _)| *w == k) {
                return Some(format!("{me}.{}", canon_kin(kin)));
            }
        }
    }
    // first person in the first few words
    if ws.iter().take(4).any(|w| FIRST_PERSON.contains(&w.as_str())) {
        return Some(me);
    }
    // a leading proper name ("Anna moved to …")
    let raw: Vec<&str> = b.split_whitespace().collect();
    if let Some(w0) = raw.first() {
        let w = w0.trim_matches(|c: char| !c.is_alphanumeric());
        if w.chars().next().is_some_and(|c| c.is_uppercase()) && !PRONOUN_STARTS.contains(&w.to_lowercase().as_str()) && w.len() > 1 {
            return Some(norm(w));
        }
    }
    None
}

fn canon_kin(k: &str) -> &str {
    match k {
        "sis" => "sister",
        "bro" => "brother",
        "mum" => "mom",
        "fiancee" => "fiance",
        x => x,
    }
}

fn norm(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
}

/// Type one claim against `act`. `protos` are the prototype vectors of the
/// typable properties (`ontology::prototypes`), `vec` the claim's vector.
pub fn type_claim(act: &Active, protos: Option<&[(usize, Vec<f32>)]>, text: &str, vec: Option<&[f32]>) -> ClaimType {
    if smalltalk(text) {
        return ClaimType { cls: "smalltalk".into(), subject: None, prop: None, score: 1.0, by: "rule".into() };
    }
    let subj = subject(text);
    let (_, b) = body(text);
    let low = b.to_lowercase().replace('’', "'");
    // cues: the longest matching cue per property
    let mut cue: Vec<usize> = vec![0; act.props.len()];
    for (i, p) in act.props.iter().enumerate() {
        if !crate::ontology::typable(p) {
            continue;
        }
        for c in p.cues.iter().chain(p.aliases.iter()) {
            let c = c.to_lowercase();
            if c.len() > cue[i] && has_phrase(&low, &c) {
                cue[i] = c.len();
            }
        }
    }
    let untyped = |subj: Option<String>| ClaimType { cls: "untyped".into(), subject: subj, prop: None, score: 0.0, by: "none".into() };
    match (protos, vec) {
        (Some(ps), Some(v)) if !ps.is_empty() => {
            let mut scored: Vec<(f64, usize)> = ps
                .iter()
                .map(|(i, pv)| (cosine(v, pv) + if cue[*i] > 0 { CUE_BONUS } else { 0.0 }, *i))
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let (best, bi) = scored[0];
            let second = scored.get(1).map_or(0.0, |x| x.0);
            if best >= PROTO_MIN && best - second >= PROTO_MARGIN {
                ClaimType {
                    cls: "fact".into(),
                    subject: subj,
                    prop: Some(act.props[bi].name.clone()),
                    score: best,
                    by: if cue[bi] > 0 { "cue+proto" } else { "proto" }.into(),
                }
            } else {
                untyped(subj)
            }
        }
        _ => {
            // cues alone: the longest match, unless two properties tie
            let max = *cue.iter().max().unwrap_or(&0);
            if max == 0 {
                return untyped(subj);
            }
            let top: Vec<usize> = (0..cue.len()).filter(|&i| cue[i] == max).collect();
            if top.len() != 1 {
                return untyped(subj);
            }
            ClaimType { cls: "fact".into(), subject: subj, prop: Some(act.props[top[0]].name.clone()), score: 1.0, by: "cue".into() }
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        d / (na.sqrt() * nb.sqrt())
    }
}

/// What the policy allows for one (new claim, candidate) pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    /// Today's behaviour.
    Allow,
    /// Asked, but an auto supersede is capped at pending.
    CapPending,
    /// `supersedes` not asked; `duplicate` asked.
    NoSupersede,
    /// Neither asked.
    Block,
}

/// The first person of a speaker or the store's user.
fn first_person_subject(s: &str) -> bool {
    !s.contains('.') && !s.contains(' ')
}

/// Whether two known subjects are certainly different people: two
/// first-person subjects of different speakers, a first-person subject and a
/// relative or name, two different relatives, or two different names. A
/// name and a relative may be the same person ("Anna" and "user.sister").
fn different_subjects(a: &str, b: &str) -> bool {
    if a == b {
        return false;
    }
    let rel = |s: &str| s.contains('.');
    match (rel(a), rel(b)) {
        (true, true) => true,
        (true, false) | (false, true) => {
            // a relative vs a plain key: different when the plain key is the
            // relative's owner ("user" vs "user.sister")
            let (r, p) = if rel(a) { (a, b) } else { (b, a) };
            r.split('.').next() == Some(p)
        }
        (false, false) => first_person_subject(a) && first_person_subject(b),
    }
}

/// The cardinality policy for a new claim `c` and a candidate `d`.
pub fn gate(act: &Active, c: &ClaimType, d: &ClaimType) -> Gate {
    if act.policy == "off" {
        return Gate::Allow;
    }
    if c.is_smalltalk() || d.is_smalltalk() {
        return Gate::Block;
    }
    if let (Some(a), Some(b)) = (&c.subject, &d.subject) {
        if different_subjects(a, b) {
            return Gate::Block;
        }
    }
    match (&c.prop, &d.prop) {
        (Some(a), Some(b)) => {
            if a != b {
                return Gate::NoSupersede;
            }
            match act.prop(a) {
                Some(p) if !p.single() => Gate::NoSupersede,
                Some(p) if !p.temporal => Gate::CapPending,
                _ => Gate::Allow,
            }
        }
        _ => match act.untyped.as_str() {
            "pending" => Gate::CapPending,
            "dup_only" => Gate::NoSupersede,
            _ => Gate::Allow,
        },
    }
}

/// A fact a kinship or role mention states by rule ("my sister Anna" →
/// `sister_of(anna, user)`).
#[derive(Debug, Clone, PartialEq)]
pub struct KinFact {
    pub s: String,
    pub p: String,
    pub o: String,
    /// The name as written, when one follows the relation.
    pub label: Option<String>,
}

/// First-person kinship and role facts of a claim: "my sister Anna …",
/// "Anna, my sister, …", "my sister …" (the relative as a node keyed by
/// relation). Possessives other than the speaker's ("his wife") are left to
/// entity resolution.
pub fn kin_facts(text: &str) -> Vec<KinFact> {
    let (spk, b) = body(text);
    let me = spk.as_deref().map(norm).unwrap_or_else(|| "user".to_string());
    let raw: Vec<&str> = b.split_whitespace().collect();
    let mut out = vec![];
    for i in 0..raw.len() {
        let w = raw[i].trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        if w != "my" && w != "our" {
            continue;
        }
        let Some(k) = raw.get(i + 1) else { continue };
        let kw = k.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'').to_lowercase();
        let kw = kw.trim_end_matches("'s");
        let Some((kin, prop)) = KIN.iter().find(|(w, _)| *w == kw) else { continue };
        let kin = canon_kin(kin);
        // a name right after ("my sister Anna", "my sister, Anna")
        let name = raw.get(i + 2).and_then(|n| {
            let t = n.trim_matches(|c: char| !c.is_alphanumeric());
            (t.chars().next().is_some_and(|c| c.is_uppercase()) && !PRONOUN_STARTS.contains(&t.to_lowercase().as_str())
                && !k.ends_with(','))
            .then(|| t.to_string())
            .or_else(|| {
                (k.ends_with(',') && t.chars().next().is_some_and(|c| c.is_uppercase())).then(|| t.to_string())
            })
        });
        // "Anna, my sister": a name just before
        let before = (i >= 1)
            .then(|| raw[i - 1])
            .filter(|p| p.ends_with(','))
            .map(|p| p.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|t| t.chars().next().is_some_and(|c| c.is_uppercase()) && !PRONOUN_STARTS.contains(&t.to_lowercase().as_str()))
            .map(|t| t.to_string());
        let name = name.or(before);
        let node = match &name {
            Some(n) => norm(n),
            None => format!("{me}.{kin}"),
        };
        // direction: the relative is the subject of the relation to the
        // speaker, except parent_of (the speaker is the parent of a son)
        let (s, o) = match *prop {
            "parent_of" => (me.clone(), node.clone()),
            _ => (node.clone(), me.clone()),
        };
        out.push(KinFact { s, p: prop.to_string(), o, label: name.clone() });
        if name.is_some() {
            // the relation key stays a lookup key of the named node
            out.push(KinFact { s: format!("{me}.{kin}"), p: "same_as".into(), o: node, label: name });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::{OntologyDoc, PERSONAL};

    fn act() -> Active {
        let d: OntologyDoc = serde_json::from_str(PERSONAL).unwrap();
        Active::merge(&[d], "cardinality", "decide")
    }

    #[test]
    fn locomo_small_talk() {
        for t in [
            "Caroline: Looks like you had a great day!",
            "Melanie: Wow, Caroline! Sorry that happened to you.",
            "Melanie: Wow, fantastic, Caroline!",
            "Caroline: Hey Mel, great to chat with you again!",
            "Caroline: Hey Melanie, great to hear from you.",
            "Jon: Sorry to hear that!",
            "Gina: Glad I could help, Jon!",
            "Jon: Hey Gina, hope you're doing ok!",
            "Melanie: Look at this! [shares a photo: a photo of a bowl with a black and white flower design]",
            "Melanie: Thoughts? [shares a photo: a photo of a bowl with a colorful design on it]",
            "Caroline: Thanks, Melanie! Your kind words really mean a lot.",
            "Caroline: I totally agree, Melanie.",
            "Melanie: It's great that you're helping out.",
            "Gina: Wow Jon, same here!",
        ] {
            assert!(smalltalk(t), "{t}");
        }
    }

    #[test]
    fn facts_are_not_small_talk() {
        for t in [
            "I live in Berlin.",
            "I moved to Lisbon last week.",
            "My sister Anna moved from Munich to Hamburg.",
            "Caroline: Recently, I had a not-so-great experience on a hike.",
            "Wow, I just got promoted to senior engineer!",
            "Melanie: I'm off to go swimming with the kids.",
            "I prefer tea over coffee.",
            "My rent is now €1,500 per month.",
            "I've been collecting postcards and now have 25 of them.",
            "Thanks to my new routine, I go to the gym four days a week.",
            "Caroline: Researching adoption agencies — it's been a dream to have a family.",
        ] {
            assert!(!smalltalk(t), "{t}");
        }
    }

    #[test]
    fn subjects() {
        assert_eq!(subject("I live in Berlin.").as_deref(), Some("user"));
        assert_eq!(subject("Caroline: I went hiking.").as_deref(), Some("caroline"));
        assert_eq!(subject("My sister Anna moved to Hamburg.").as_deref(), Some("user.sister"));
        assert_eq!(subject("Anna moved to Hamburg.").as_deref(), Some("anna"));
        assert_eq!(subject("The weather is nice.").as_deref(), None);
    }

    #[test]
    fn cue_typing_without_vectors() {
        let a = act();
        let t = type_claim(&a, None, "I moved to Lisbon last week.", None);
        assert_eq!(t.prop.as_deref(), Some("lives_in"), "{t:?}");
        let t = type_claim(&a, None, "I love sushi.", None);
        assert_eq!(t.prop.as_deref(), Some("likes"), "{t:?}");
    }

    #[test]
    fn policy() {
        let a = act();
        let f = |subj: &str, prop: &str| ClaimType {
            cls: "fact".into(),
            subject: Some(subj.into()),
            prop: Some(prop.into()),
            score: 1.0,
            by: "test".into(),
        };
        let st = ClaimType { cls: "smalltalk".into(), subject: None, prop: None, score: 1.0, by: "rule".into() };
        assert_eq!(gate(&a, &f("user", "lives_in"), &f("user", "lives_in")), Gate::Allow);
        assert_eq!(gate(&a, &f("user", "likes"), &f("user", "likes")), Gate::NoSupersede);
        assert_eq!(gate(&a, &f("user", "lives_in"), &f("user", "employer")), Gate::NoSupersede);
        assert_eq!(gate(&a, &f("user.sister", "lives_in"), &f("user", "lives_in")), Gate::Block);
        assert_eq!(gate(&a, &f("anna", "lives_in"), &f("user.sister", "lives_in")), Gate::Allow);
        assert_eq!(gate(&a, &f("caroline", "lives_in"), &f("melanie", "lives_in")), Gate::Block);
        assert_eq!(gate(&a, &st, &f("user", "lives_in")), Gate::Block);
        assert_eq!(gate(&a, &f("user", "birthday"), &f("user", "birthday")), Gate::CapPending);
    }

    #[test]
    fn kinship() {
        let k = kin_facts("My sister Anna lives in Munich.");
        assert_eq!(k[0], KinFact { s: "anna".into(), p: "sister_of".into(), o: "user".into(), label: Some("Anna".into()) });
        assert_eq!(k[1].p, "same_as");
        let k = kin_facts("My sister moved to Hamburg.");
        assert_eq!(k, vec![KinFact { s: "user.sister".into(), p: "sister_of".into(), o: "user".into(), label: None }]);
        let k = kin_facts("Caroline: my son Tom started school.");
        assert_eq!(k[0], KinFact { s: "caroline".into(), p: "parent_of".into(), o: "tom".into(), label: Some("Tom".into()) });
        let k = kin_facts("Anna, my sister, called.");
        assert_eq!(k[0].s, "anna");
        assert!(kin_facts("His wife Sarah is a doctor.").is_empty());
    }
}
