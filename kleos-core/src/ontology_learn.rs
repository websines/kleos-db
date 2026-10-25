/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Zero-maintenance ontologies (docs/specs/ontology.md, "Learning" and
//! "Maintenance"): a namespace's ontology grows and corrects itself from the
//! memories it stores, with every change versioned, explained by the memories
//! behind it, and reversible.
//!
//! * **Learning** clusters the claims no property typed (and document
//!   chunks), proposes a property per coherent cluster (label and cues from
//!   the cluster's distinctive words, cardinality and time-variance from the
//!   data, checked by the decider when one is configured), or new vocabulary
//!   for an existing property when the cluster sits on its prototype.
//! * **Auto mode** (the default) applies strong proposals to the namespace's
//!   learned layer (`learned:<ns>`); **review mode** keeps them as proposals
//!   for `approve` / `reject`.
//! * **Maintenance** flips a single-valued property to multi-valued when the
//!   data keeps holding several current values per subject, merges learned
//!   properties that mean the same, and retires learned properties nothing
//!   uses any more.
//! * Every change is a row of `kleos_onto_change` (before/after, evidence);
//!   `undo` restores the before state as a new version.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use lazy_static::lazy_static;
use miette::{bail, IntoDiagnostic, Result};
use serde_derive::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};

use crate::data::value::DataValue;
use crate::ontology::{learned_name, Active, OntoProp, OntologyDoc, ONTO_CHANGE, ONTO_PROPOSAL};
use crate::runtime::db::{Db, NamedRows, ScriptMutability};
use crate::storage::Storage;
use crate::JsonData;

fn dj(v: JsonValue) -> DataValue {
    DataValue::Json(JsonData(v))
}

/// Options of `::ontology learn`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LearnOpts {
    pub ns: Option<String>,
    /// `auto` (default: strong proposals are applied) or `review` (all stay
    /// proposals).
    pub mode: Option<String>,
    /// Smallest cluster that becomes a proposal.
    pub min_support: usize,
    /// Smallest cluster auto mode applies.
    pub auto_support: usize,
    /// Most recent claims considered.
    pub max_claims: usize,
    /// Cosine similarity that joins a claim to a cluster.
    pub sim: f64,
    /// Propose, apply nothing.
    pub dry_run: bool,
}

impl Default for LearnOpts {
    fn default() -> Self {
        LearnOpts {
            ns: None,
            mode: None,
            min_support: 5,
            auto_support: 8,
            max_claims: 1500,
            sim: knob("KLEOS_ONTO_LEARN_SIM", 0.70),
            dry_run: false,
        }
    }
}

fn knob(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn now_micros() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_micros() as i64).unwrap_or(0)
}

fn s(v: &str) -> DataValue {
    DataValue::from(v)
}

fn strs(v: &[String]) -> DataValue {
    DataValue::List(v.iter().map(|x| DataValue::from(x.as_str())).collect())
}

fn cell(v: &DataValue) -> String {
    v.get_str().map(str::to_string).unwrap_or_default()
}

fn cell_strs(v: &DataValue) -> Vec<String> {
    match v {
        DataValue::List(l) => l.iter().filter_map(|x| x.get_str().map(str::to_string)).collect(),
        _ => vec![],
    }
}

fn vec_of(v: &DataValue) -> Option<Vec<f32>> {
    match v {
        DataValue::Vec(crate::data::value::Vector::F32(a)) => Some(a.to_vec()),
        DataValue::Vec(crate::data::value::Vector::F64(a)) => Some(a.iter().map(|x| *x as f32).collect()),
        _ => None,
    }
}

fn json_cell(v: &DataValue) -> JsonValue {
    match v {
        DataValue::Null => JsonValue::Null,
        other => JsonValue::from(other.clone()),
    }
}

fn params(p: Vec<(&str, DataValue)>) -> BTreeMap<String, DataValue> {
    p.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

// ------------------------------------------------------------------ text

const STOP: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to", "in", "on", "at", "for", "with", "from", "by",
    "about", "as", "into", "over", "after", "before", "up", "down", "out", "off", "is", "am", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "will", "would", "can", "could", "should",
    "shall", "may", "might", "must", "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs", "it", "its", "this", "that", "these",
    "those", "there", "here", "what", "which", "who", "whom", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "not", "only", "own", "same", "than", "too", "very", "just",
    "also", "now", "really", "still", "even", "again", "ever", "never", "always", "lot", "lots", "much", "many", "get",
    "got", "getting", "go", "going", "went", "gone", "make", "made", "thing", "things", "one", "two", "three", "last",
    "next", "week", "weeks", "month", "months", "year", "years", "day", "days", "today", "yesterday", "tomorrow", "time",
    "i'm", "i've", "i'd", "i'll", "it's", "that's", "don't", "didn't", "can't", "won't", "im", "ive", "like", "new",
    "recently", "currently", "since", "back", "around", "yeah", "well", "think", "know", "want", "really", "pretty",
];

/// Words that make a poor label or cue: greetings, courtesy, fillers and
/// evaluations (a cluster about "thanks" is small talk, not a property).
const LABEL_STOP: &[&str] = &[
    "thanks", "thank", "hey", "hi", "hello", "love", "loved", "loving", "gonna", "kinda", "wanna", "excited",
    "looks", "look", "here's", "people", "help", "wow", "great", "awesome", "glad", "sorry", "cool", "nice",
    "amazing", "lovely", "fun", "good", "best", "happy", "sure", "totally", "definitely", "maybe", "stuff",
    "lot", "bit", "way", "something", "anything", "everything", "someone", "everyone", "guys", "chat", "talk",
    "memories", "support", "tough", "hard", "easy", "feel", "feeling", "felt",
    // when, not what: dates and durations are values, never the property
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "weekend", "hour", "hours",
    "minute", "minutes", "quarter", "night", "morning", "evening", "afternoon", "january", "february", "march",
    "april", "june", "july", "august", "september", "october", "november", "december",
];

/// Names a pool uses as speakers or to address someone ("Caroline:",
/// "Thanks, Mel"): never a label or cue.
pub fn pool_names(texts: &[&str]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for t in texts {
        if let Some((n, _)) = crate::memory_extract::speaker_prefix(t) {
            for w in n.split_whitespace() {
                out.insert(w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase());
            }
        }
        let ws: Vec<&str> = t.split_whitespace().collect();
        for w in ws.windows(2) {
            let a = w[0].trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            let b = w[1].trim_matches(|c: char| !c.is_alphanumeric());
            if ["thanks", "hey", "hi", "hello", "congrats", "wow", "yes", "no", "sorry", "oh"].contains(&a.as_str())
                && b.chars().next().is_some_and(|c| c.is_uppercase())
            {
                out.insert(b.to_lowercase());
            }
        }
    }
    out
}

fn stem(w: &str) -> String {
    for suf in ["ing", "es", "ed", "s"] {
        if let Some(x) = w.strip_suffix(suf) {
            if x.len() >= 3 {
                return x.to_string();
            }
        }
    }
    w.to_string()
}

/// Words that mark a value replacing an earlier one.
const UPDATE_CUES: &[&str] = &[
    "now", "anymore", "no longer", "moved", "switched", "changed", "currently", "these days", "upgraded", "replaced",
    "instead", "finally", "up to", "went up", "went down", "increased", "decreased", "dropped", "rose", "new",
];

fn words(t: &str) -> Vec<String> {
    let body = match crate::memory_extract::speaker_prefix(t) {
        Some((_, at)) => &t[at..],
        None => t,
    };
    body.replace('’', "'")
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .map(|w| w.trim_matches('\'').to_lowercase())
        .filter(|w| w.len() > 1 && !w.chars().all(|c| c.is_ascii_digit()))
        .collect()
}

fn content(t: &str) -> Vec<String> {
    words(t).into_iter().filter(|w| !STOP.contains(&w.as_str())).collect()
}

fn has_cue(t: &str) -> bool {
    let low = format!(" {} ", words(t).join(" "));
    UPDATE_CUES.iter().any(|c| low.contains(&format!(" {c} ")))
}

fn has_number(t: &str) -> bool {
    t.chars().any(|c| c.is_ascii_digit())
}

fn snake(label: &str) -> String {
    let mut out = String::new();
    for c in label.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        } else if !out.ends_with('_') && !out.is_empty() {
            out.push('_');
        }
    }
    out.trim_matches('_').to_string()
}

/// The words and adjacent word pairs that set a cluster apart from the
/// whole pool: (term, share of the cluster containing it, lift).
pub fn distinctive(cluster: &[&str], pool_df: &HashMap<String, usize>, pool_n: usize, names: &BTreeSet<String>) -> Vec<(String, f64, f64)> {
    let mut df: HashMap<String, usize> = HashMap::new();
    for t in cluster {
        let c: Vec<String> = content(t)
            .into_iter()
            .filter(|w| !LABEL_STOP.contains(&w.as_str()) && !names.contains(w))
            .collect();
        let mut terms: BTreeSet<String> = c.iter().cloned().collect();
        for w in c.windows(2) {
            terms.insert(format!("{} {}", w[0], w[1]));
        }
        for w in c.windows(3) {
            terms.insert(format!("{} {} {}", w[0], w[1], w[2]));
        }
        for x in terms {
            *df.entry(x).or_default() += 1;
        }
    }
    let n = cluster.len().max(1) as f64;
    let mut out: Vec<(String, f64, f64)> = df
        .into_iter()
        .filter(|(_, c)| *c >= 2)
        .map(|(t, c)| {
            let share = c as f64 / n;
            let base = (*pool_df.get(&t).unwrap_or(&c) as f64 / pool_n.max(1) as f64).max(1e-6);
            (t, share, share / base)
        })
        .collect();
    // phrases that are as common as their words win the label ("cloud
    // vendor" over "vendor")
    let boost = |t: &str| match t.matches(' ').count() {
        0 => 1.0,
        1 => 1.3,
        _ => 1.45,
    };
    out.sort_by(|a, b| {
        let ka = a.1 * a.2.ln().max(0.0) * boost(&a.0);
        let kb = b.1 * b.2.ln().max(0.0) * boost(&b.0);
        kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
    });
    out
}

/// A count property's role: the noun most members put right after a number
/// ("12 engineers", "now seats 40 people"), when at least half do.
pub fn counted_noun(cluster: &[&str]) -> Option<String> {
    let mut df: HashMap<String, usize> = HashMap::new();
    for t in cluster {
        let toks: Vec<String> = t
            .split(|c: char| c.is_whitespace())
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|w| !w.is_empty())
            .collect();
        let mut seen = BTreeSet::new();
        for w in toks.windows(2) {
            if w[0].chars().all(|c| c.is_ascii_digit()) && w[1].chars().all(|c| c.is_alphabetic()) && !STOP.contains(&w[1].as_str()) && !LABEL_STOP.contains(&w[1].as_str()) {
                seen.insert(w[1].clone());
            }
        }
        for x in seen {
            *df.entry(x).or_default() += 1;
        }
    }
    let (w, c) = df.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))?;
    (c * 2 >= cluster.len() && c >= 2).then_some(w)
}

/// Words of a claim subject that a label must not use (they name who, not
/// what): `team:data platform` → data, platform, team; `user.sister` →
/// sister. A leading-name guess (`name:…`) is not used: it is often the
/// sentence's first word ("Cloud vendor for …"); speakers and addressed names
/// are already left out (`pool_names`).
pub fn subject_words(subject: &str) -> Vec<String> {
    match subject.split_once(':') {
        Some(("team", n)) => n.split_whitespace().map(|w| w.to_string()).chain(["team".to_string()]).collect(),
        Some(_) => vec![],
        None => subject.split('.').skip(1).map(|w| w.to_string()).collect(),
    }
}

pub fn pool_df(texts: &[&str]) -> HashMap<String, usize> {
    let mut df: HashMap<String, usize> = HashMap::new();
    for t in texts {
        let c = content(t);
        let mut terms: BTreeSet<String> = c.iter().cloned().collect();
        for w in c.windows(2) {
            terms.insert(format!("{} {}", w[0], w[1]));
        }
        for w in c.windows(3) {
            terms.insert(format!("{} {} {}", w[0], w[1], w[2]));
        }
        for x in terms {
            *df.entry(x).or_default() += 1;
        }
    }
    df
}

// ------------------------------------------------------------------ clustering

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

/// Leader clustering by cosine `sim`, then one pass reassigning every
/// point to its nearest centroid. Returns clusters as point indices,
/// largest first, with their mean similarity to the centroid.
pub fn cluster(vecs: &[Vec<f32>], sim: f64) -> Vec<(Vec<usize>, f64)> {
    let mut cents: Vec<Vec<f64>> = vec![];
    let mut counts: Vec<usize> = vec![];
    let mut assign = vec![usize::MAX; vecs.len()];
    for (i, v) in vecs.iter().enumerate() {
        let best = cents
            .iter()
            .enumerate()
            .map(|(c, m)| (c, cosine(v, &m.iter().map(|x| *x as f32).collect::<Vec<_>>())))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        match best {
            Some((c, x)) if x >= sim => {
                counts[c] += 1;
                let k = counts[c] as f64;
                for (m, y) in cents[c].iter_mut().zip(v) {
                    *m += (*y as f64 - *m) / k;
                }
                assign[i] = c;
            }
            _ => {
                cents.push(v.iter().map(|x| *x as f64).collect());
                counts.push(1);
                assign[i] = cents.len() - 1;
            }
        }
    }
    let cf: Vec<Vec<f32>> = cents.iter().map(|m| m.iter().map(|x| *x as f32).collect()).collect();
    let mut groups: BTreeMap<usize, Vec<(usize, f64)>> = BTreeMap::new();
    for (i, v) in vecs.iter().enumerate() {
        let (c, x) = cf
            .iter()
            .enumerate()
            .map(|(c, m)| (c, cosine(v, m)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((assign[i], 1.0));
        if x >= sim {
            groups.entry(c).or_default().push((i, x));
        }
    }
    let mut out: Vec<(Vec<usize>, f64)> = groups
        .into_values()
        .map(|g| {
            let m = g.iter().map(|x| x.1).sum::<f64>() / g.len() as f64;
            (g.into_iter().map(|x| x.0).collect(), m)
        })
        .collect();
    // one concept phrased two ways ("the data team now has 12 engineers",
    // "the data team grew to 14 engineers") can land in two clusters whose
    // centroids nearly coincide: merge those
    let merge_at = (sim + 0.15).min(0.92);
    loop {
        let cents: Vec<Vec<f32>> = out.iter().map(|(m, _)| centroid(vecs, m)).collect();
        let mut best: Option<(usize, usize, f64)> = None;
        for i in 0..out.len() {
            for j in (i + 1)..out.len() {
                let x = cosine(&cents[i], &cents[j]);
                if x >= merge_at && best.map_or(true, |b| x > b.2) {
                    best = Some((i, j, x));
                }
            }
        }
        let Some((i, j, _)) = best else { break };
        let (mj, cj) = out.remove(j);
        let (mi, ci) = &mut out[i];
        let (ni, nj) = (mi.len() as f64, mj.len() as f64);
        *ci = (*ci * ni + cj * nj) / (ni + nj);
        mi.extend(mj);
    }
    out.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    out
}

fn centroid(vecs: &[Vec<f32>], members: &[usize]) -> Vec<f32> {
    let d = vecs.first().map_or(0, |v| v.len());
    let mut c = vec![0f32; d];
    for &i in members {
        for (a, b) in c.iter_mut().zip(&vecs[i]) {
            *a += *b;
        }
    }
    c
}

// ------------------------------------------------------------------ inference of a property's shape

/// What the data says about a cluster's cardinality.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Shape {
    /// `one` or `many`.
    pub card: String,
    pub temporal: bool,
    pub update_cue_share: f64,
    /// Subjects with two or more claims in the cluster.
    pub repeat_subjects: usize,
    /// Mean decider P(supersedes) over sampled same-subject pairs.
    pub decider_p: Option<f64>,
    pub numeric: bool,
    /// Claims of the cluster already replaced by another of them.
    pub replaced: usize,
    pub why: String,
}

/// Cardinality from the data: single-valued when subjects say it more than
/// once and later statements carry update cues (or the decider says they
/// replace the earlier ones); multi-valued otherwise, which is the safe
/// default (a multi-valued property never supersedes).
pub fn shape(texts: &[&str], subjects: &[Option<String>], decider_p: Option<f64>) -> Shape {
    shape_with(texts, subjects, decider_p, 0)
}

/// [`shape`] with the number of the cluster's claims that `remember`
/// already replaced by another of them (supersedes links inside the cluster).
pub fn shape_with(texts: &[&str], subjects: &[Option<String>], decider_p: Option<f64>, replaced: usize) -> Shape {
    let n = texts.len().max(1) as f64;
    let cue = texts.iter().filter(|t| has_cue(t)).count() as f64 / n;
    let mut per: BTreeMap<&str, usize> = BTreeMap::new();
    for x in subjects.iter().flatten() {
        *per.entry(x.as_str()).or_default() += 1;
    }
    let repeat = per.values().filter(|c| **c >= 2).count();
    let numeric = texts.iter().filter(|t| has_number(t)).count() as f64 / n >= 0.6;
    let (card, why) = match decider_p {
        _ if replaced >= 2 => ("one", format!("{replaced} of its statements were already replaced by later ones")),
        Some(p) if p >= 0.9 => ("one", format!("the decider says later statements replace earlier ones (mean p {p:.2})")),
        Some(p) if p <= 0.5 => ("many", format!("the decider says statements coexist (mean p {p:.2})")),
        _ if repeat >= 1 && cue >= 0.25 => ("one", format!("{repeat} subject(s) restate it and {:.0}% carry update cues", cue * 100.0)),
        _ => ("many", format!("no evidence of replacement ({:.0}% update cues, {repeat} repeating subject(s))", cue * 100.0)),
    };
    Shape {
        card: card.to_string(),
        temporal: card == "one" || cue >= 0.1,
        update_cue_share: cue,
        repeat_subjects: repeat,
        decider_p,
        numeric,
        replaced,
        why,
    }
}

// ------------------------------------------------------------------ automatic trigger

lazy_static! {
    /// Untyped claims added per namespace since the last automatic learning
    /// run (in-process; a restart just delays the next run).
    static ref SINCE: Mutex<HashMap<String, usize>> = Mutex::new(HashMap::new());
}

/// Untyped claims between automatic learning runs (`KLEOS_ONTO_LEARN_EVERY`,
/// 0 turns automatic learning off).
pub fn learn_every() -> usize {
    std::env::var("KLEOS_ONTO_LEARN_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(200)
}

/// Count `n` new untyped claims for `ns`; true when a run is due.
pub fn note_untyped(ns: &str, n: usize) -> bool {
    let every = learn_every();
    if every == 0 || n == 0 {
        return false;
    }
    let mut m = SINCE.lock().unwrap();
    let c = m.entry(ns.to_string()).or_default();
    *c += n;
    if *c >= every {
        *c = 0;
        true
    } else {
        false
    }
}

// ------------------------------------------------------------------ store

/// One claim the learner looks at.
struct Pt {
    id: String,
    text: String,
    kind: String,
    vec: Vec<f32>,
}

impl<'s, S: Storage<'s>> Db<S> {
    fn ol_q(&'s self, script: &str, p: Vec<(&str, DataValue)>) -> Result<NamedRows> {
        self.run_script(script, params(p), ScriptMutability::Mutable)
    }

    /// The namespace's learned layer (empty when it has none yet).
    fn learned_doc(&'s self, ns: &str) -> Result<OntologyDoc> {
        let name = learned_name(ns);
        Ok(self.onto_get(&name, None)?.unwrap_or_else(|| OntologyDoc {
            onto: name.clone(),
            version: "0".into(),
            doc: format!("Learned automatically from namespace '{ns}' (docs/specs/ontology.md, \"Learning\")."),
            source: "learned".into(),
            classes: vec![],
            properties: vec![],
        }))
    }

    /// Merge the learned properties of namespace `from` that namespace
    /// `to`'s learned layer lacks into it: the automatic promotion of
    /// schema-level knowledge (docs/specs/tenancy.md, "Promotion"). Instance
    /// facts are never promoted this way. Returns the number added.
    pub(crate) fn onto_promote_learned(&'s self, from: &str, to: &str) -> Result<usize> {
        let Some(src) = self.onto_get(&learned_name(from), None)? else { return Ok(0) };
        let mut dst = self.learned_doc(to)?;
        let have: std::collections::BTreeSet<String> = dst.properties.iter().map(|p| p.name.clone()).collect();
        let new: Vec<OntoProp> = src.properties.iter().filter(|p| !have.contains(&p.name)).cloned().collect();
        if new.is_empty() {
            return Ok(0);
        }
        dst.properties.extend(new.iter().cloned());
        let v: u64 = dst.version.parse().unwrap_or(0);
        dst.version = (v + 1).to_string();
        self.onto_define(&dst, true)?;
        Ok(new.len())
    }

    /// Write a new version of the learned layer and log the change.
    #[allow(clippy::too_many_arguments)]
    fn learned_save(
        &'s self,
        ns: &str,
        mut doc: OntologyDoc,
        kind: &str,
        target: &str,
        before: &[OntoProp],
        after: &[OntoProp],
        evidence: &[String],
        by: &str,
    ) -> Result<String> {
        let v: u64 = doc.version.parse().unwrap_or(0);
        doc.version = (v + 1).to_string();
        self.onto_define(&doc, true)?;
        let at = now_micros();
        let id = format!("ch_{at:x}_{}", snake(target));
        self.ol_q(
            &format!(
                "?[ns, id, at, kind, onto, target, before, after, evidence, by, undone] <- \
                 [[$ns, $id, $at, $kind, $onto, $target, $before, $after, $evidence, $by, false]] \
                 :put {ONTO_CHANGE} {{ns, id => at, kind, onto, target, before, after, evidence, by, undone}}"
            ),
            vec![
                ("ns", s(ns)),
                ("id", s(&id)),
                ("at", DataValue::from(at)),
                ("kind", s(kind)),
                ("onto", s(&doc.onto)),
                ("target", s(target)),
                ("before", dj(serde_json::to_value(before).into_diagnostic()?)),
                ("after", dj(serde_json::to_value(after).into_diagnostic()?)),
                ("evidence", strs(evidence)),
                ("by", s(by)),
            ],
        )?;
        Ok(id)
    }

    /// Claims no property typed (not small talk, not the assistant's), and
    /// document chunks, most recent first, with their vectors.
    fn learn_points(&'s self, ns: &str, max: usize) -> Result<Vec<Pt>> {
        // current claims and the ones later statements replaced: a value that
        // keeps being replaced is the best evidence of a single-valued property
        let r = self.ol_q(
            "ty[id] := *kleos_mem_type{ns: $ns, id, cls}, cls != 'untyped' \
             asst[id] := *kleos_mem_role{ns: $ns, id, role}, role == 'assistant' \
             all[id, text, kind, created] := *kleos_mem{ns: $ns, id, text, kind, created} \
             ?[id, text, kind, created] := all[id, text, kind, created], *kleos_mem_vec{ns: $ns, id}, \
                 not ty[id], not asst[id] \
             :order -created \
             :limit $n",
            vec![("ns", s(ns)), ("n", DataValue::from(max as i64))],
        )?;
        let mut pts: Vec<Pt> = r
            .rows
            .iter()
            .map(|x| Pt { id: cell(&x[0]), text: cell(&x[1]), kind: cell(&x[2]), vec: vec![] })
            .filter(|p| {
                p.kind == "doc"
                    || (crate::memory_claims::is_declarative(&p.text) && !crate::ontology_type::smalltalk(&p.text))
            })
            .collect();
        if pts.is_empty() {
            return Ok(pts);
        }
        let ids: Vec<String> = pts.iter().map(|p| p.id.clone()).collect();
        let vr = self.ol_q("?[id, vec] := *kleos_mem_vec{ns: $ns, id, vec}, is_in(id, $ids)", vec![("ns", s(ns)), ("ids", strs(&ids))])?;
        let vecs: HashMap<String, Vec<f32>> = vr.rows.iter().filter_map(|x| Some((cell(&x[0]), vec_of(&x[1])?))).collect();
        for p in &mut pts {
            if let Some(v) = vecs.get(&p.id) {
                p.vec = v.clone();
            }
        }
        pts.retain(|p| !p.vec.is_empty());
        Ok(pts)
    }

    /// The entity kind (as an ontology class: `Organization`, `Place`,
    /// `Person`) that at least 60% of `ids` mention, from the extractor's
    /// span labels, leaving out entities named like the subjects (`skip`);
    /// `None` without extractor entities.
    fn learn_value_kind(&'s self, ns: &str, ids: &[String], skip: &BTreeSet<String>) -> Result<Option<String>> {
        // the extractor's span labels (kept spans), else the entity rows' kinds
        let Ok(r) = self.ol_q(
            "?[id, kind, name] := *kleos_mem_extract{ns: $ns, id, label: kind, text: name, kept}, kept, is_in(id, $ids) \
             ?[id, kind, name] := *kleos_mem_mention{ns: $ns, id, eid}, is_in(id, $ids), \
                 *kleos_mem_entity{ns: $ns, eid, kind, name}, !is_null(kind)",
            vec![("ns", s(ns)), ("ids", strs(ids))],
        ) else {
            return Ok(None);
        };
        let mut per: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for x in &r.rows {
            let (id, kind, name) = (cell(&x[0]), cell(&x[1]).to_lowercase(), cell(&x[2]).to_lowercase());
            if name.split_whitespace().any(|w| skip.contains(w)) {
                continue;
            }
            per.entry(kind).or_default().insert(id);
        }
        let Some((kind, n)) = per.into_iter().map(|(k, v)| (k, v.len())).max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))) else {
            return Ok(None);
        };
        if (n as f64) < 0.6 * ids.len() as f64 {
            return Ok(None);
        }
        let class = match kind.as_str() {
            "organization" | "organisation" | "company" | "org" => "Organization",
            "location" | "place" | "city" | "country" => "Place",
            "person" | "relative" => "Person",
            // a kind the ontologies have no class for stays a literal
            _ => return Ok(None),
        };
        Ok(Some(class.to_string()))
    }

    /// How many of `ids` a later one of them replaced (`supersedes` links).
    fn learn_replaced(&'s self, ns: &str, ids: &[String]) -> Result<usize> {
        let r = self.ol_q(
            "?[count(dst)] := *kleos_mem_link{ns: $ns, src, rel: 'supersedes', dst}, is_in(src, $ids), is_in(dst, $ids)",
            vec![("ns", s(ns)), ("ids", strs(ids))],
        )?;
        Ok(r.rows.first().and_then(|x| x[0].get_int()).unwrap_or(0) as usize)
    }

    /// Mean decider P(supersedes) over up to 8 same-subject pairs (older,
    /// newer) of a cluster; `None` without a decider or pairs.
    fn learn_decider_p(&'s self, decider: Option<&str>, texts: &[&str], subjects: &[Option<String>]) -> Option<f64> {
        let d = decider?;
        let mut by: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (i, x) in subjects.iter().enumerate() {
            if let Some(x) = x {
                by.entry(x.as_str()).or_default().push(i);
            }
        }
        // points are newest first: a later index is older
        let mut pairs: Vec<(&str, &str)> = vec![];
        for idx in by.values() {
            for w in idx.windows(2) {
                pairs.push((texts[w[1]], texts[w[0]]));
                if pairs.len() >= 8 {
                    break;
                }
            }
        }
        if pairs.is_empty() {
            return None;
        }
        let ds = crate::decide::decide(d, crate::decide::Template::Supersedes, &pairs).ok()?;
        Some(ds.iter().map(|x| x.p).sum::<f64>() / ds.len() as f64)
    }

    /// `::ontology learn`: propose (and in auto mode apply) properties from
    /// the claims nothing typed. An open proposal is re-evaluated with the
    /// current data (and applied once it is strong); a rejected or applied one
    /// is never proposed again.
    pub fn onto_learn(&'s self, o: LearnOpts) -> Result<JsonValue> {
        let ns = o.ns.clone().filter(|x| !x.is_empty()).unwrap_or_else(|| "default".into());
        let mode = o.mode.clone().unwrap_or_else(|| "auto".into());
        if !["auto", "review"].contains(&mode.as_str()) {
            bail!("ontology learn: mode must be auto or review, got '{mode}'");
        }
        let Some(act) = self.onto_active(&ns)? else {
            return Ok(json!({"ns": ns, "skipped": "no active ontology (the store default is off)"}));
        };
        self.onto_ensure()?;
        let cfg = self.mem_config()?.unwrap_or_default();
        let pts = self.learn_points(&ns, o.max_claims)?;
        let total = pts.len();
        if total < o.min_support {
            return Ok(json!({"ns": ns, "mode": mode, "untyped": total, "proposals": [], "applied": []}));
        }
        let texts: Vec<&str> = pts.iter().map(|p| p.text.as_str()).collect();
        let pdf = pool_df(&texts);
        let names = pool_names(&texts);
        let protos = match cfg.embedder.as_deref() {
            Some(e) => Some(crate::ontology::prototypes(&act, e)?),
            None => None,
        };
        // similarity thresholds were tuned under the reference embedder: move
        // them to this one's baseline (docs/specs/ontology.md, "Calibration")
        let base = protos.as_ref().map_or(crate::ontology::BASELINE_REF, |p| crate::ontology::proto_baseline(p));
        let cal = |t: f64| crate::ontology::calibrated(t, base);
        let rejected: BTreeSet<String> = self
            .ol_q(
                &format!("?[id] := *{ONTO_PROPOSAL}{{ns: $ns, id, status}}, is_in(status, ['rejected', 'active'])"),
                vec![("ns", s(&ns))],
            )?
            .rows
            .iter()
            .map(|x| cell(&x[0]))
            .collect();
        let vecs: Vec<Vec<f32>> = pts.iter().map(|p| p.vec.clone()).collect();
        let mut proposals = vec![];
        let mut applied = vec![];
        let mut used_names: BTreeSet<String> = act.props.iter().map(|p| p.name.clone()).collect();
        for (members, coherence) in cluster(&vecs, cal(o.sim)) {
            if members.len() < o.min_support {
                continue;
            }
            let ctexts: Vec<&str> = members.iter().map(|&i| texts[i]).collect();
            let subjects: Vec<Option<String>> = ctexts.iter().map(|t| crate::ontology_type::subject(t)).collect();
            // the words that name the subjects (team names, people) say who,
            // not what: never a label or cue
            let mut cnames = names.clone();
            for x in subjects.iter().flatten() {
                cnames.extend(subject_words(x));
            }
            let terms: Vec<(String, f64, f64)> = distinctive(&ctexts, &pdf, total, &cnames)
                .into_iter()
                .filter(|t| !t.0.split(' ').any(|w| cnames.contains(w)))
                .collect();
            let Some((mut label, _, _)) = terms.iter().find(|t| t.1 >= 0.3).cloned() else { continue };
            let mut cues: Vec<String> = terms.iter().filter(|t| t.1 >= 0.3).take(5).map(|t| t.0.clone()).collect();
            // a count names what it counts: "12 engineers" → `engineer_count`
            if let Some(noun) = counted_noun(&ctexts) {
                if label.split(' ').any(|w| w == noun) || !label.contains(' ') {
                    label = format!("{} count", stem(&noun));
                    if !cues.contains(&noun) {
                        cues.insert(0, noun);
                    }
                }
            }
            let evidence: Vec<String> = members.iter().map(|&i| pts[i].id.clone()).collect();
            let docs = members.iter().filter(|&&i| pts[i].kind == "doc").count();
            // a cluster on an existing property's prototype is vocabulary for it
            let centroid: Vec<f32> = {
                let d = vecs[0].len();
                let mut c = vec![0f32; d];
                for &i in &members {
                    for (a, b) in c.iter_mut().zip(&vecs[i]) {
                        *a += *b;
                    }
                }
                c
            };
            let near = protos.as_ref().and_then(|ps| {
                ps.iter()
                    .map(|(pi, v)| (*pi, cosine(&centroid, v)))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            });
            let score = members.len() as f64 * coherence;
            let (kind, body) = match near {
                Some((pi, x)) if x >= cal(knob("KLEOS_ONTO_VOCAB_SIM", 0.80)) => {
                    let p = &act.props[pi];
                    let new_cues: Vec<String> = cues.iter().filter(|c| !p.cues.contains(c)).cloned().collect();
                    if new_cues.is_empty() {
                        continue;
                    }
                    ("vocabulary", json!({"prop": p.name, "cues": new_cues, "alias": label, "similarity": x}))
                }
                _ => {
                    let replaced = self.learn_replaced(&ns, &evidence)?;
                    let dp = if replaced >= 2 { None } else { self.learn_decider_p(cfg.decider.as_deref(), &ctexts, &subjects) };
                    let sh = shape_with(&ctexts, &subjects, dp, replaced);
                    let mut name = snake(&label);
                    if name.is_empty() {
                        continue;
                    }
                    while used_names.contains(&name) {
                        name.push_str("_x");
                    }
                    used_names.insert(name.clone());
                    // the extractor's type of the values: a cluster whose
                    // claims mostly mention one kind of entity (other than
                    // their subject) relates the subject to that kind
                    let range = if sh.numeric { None } else { self.learn_value_kind(&ns, &evidence, &cnames)? };
                    let prop = OntoProp {
                        name: name.clone(),
                        kind: if range.is_some() { "object" } else { "data" }.into(),
                        range: range.iter().cloned().collect(),
                        datatype: if range.is_some() { None } else { Some(if sh.numeric { "number" } else { "string" }.into()) },
                        card: sh.card.clone(),
                        temporal: sh.temporal,
                        label: label.clone(),
                        doc: format!(
                            "Learned from {} claims ({} document chunks); {}.",
                            members.len(),
                            docs,
                            sh.why
                        ),
                        cues: cues.clone(),
                        ..Default::default()
                    };
                    ("property", json!({"prop": prop, "shape": sh, "examples": ctexts.iter().take(3).collect::<Vec<_>>()}))
                }
            };
            let id = format!("pr_{}_{}", kind, snake(body["prop"].as_str().unwrap_or_else(|| body["prop"]["name"].as_str().unwrap_or("x"))));
            if rejected.contains(&id) {
                continue;
            }
            // auto mode: strong, coherent clusters; a single-valued guess needs
            // the decider's or the cues' clear support (a wrong `one` wrongly
            // supersedes, a wrong `many` only misses replacements)
            let strong = members.len() >= o.auto_support && coherence >= cal(knob("KLEOS_ONTO_AUTO_COHERENCE", 0.60));
            let mut body = body;
            if kind == "property" && strong && body["prop"]["card"] == "one" {
                let sh = body["shape"].clone();
                let backed = sh["replaced"].as_u64().unwrap_or(0) >= 2
                    || sh["decider_p"].as_f64().is_some_and(|p| p >= 0.9)
                    || sh["update_cue_share"].as_f64().unwrap_or(0.0) >= 0.4;
                if !backed {
                    body["prop"]["card"] = json!("many");
                    body["shape"]["why"] = json!(format!("{} (kept multi-valued: not enough support for single-valued)", sh["why"].as_str().unwrap_or("")));
                }
            }
            let auto = mode == "auto" && strong && !o.dry_run;
            let status = if o.dry_run { "dry_run" } else if auto { "active" } else { "proposed" };
            let prop_entry = json!({"id": id, "kind": kind, "body": body, "support": members.len(),
                "coherence": coherence, "score": score, "status": status, "evidence": evidence.iter().take(20).collect::<Vec<_>>()});
            if !o.dry_run {
                self.ol_q(
                    &format!(
                        "?[ns, id, kind, body, evidence, score, status, created, decided] <- \
                         [[$ns, $id, $kind, $body, $evidence, $score, $status, $at, null]] \
                         :put {ONTO_PROPOSAL} {{ns, id => kind, body, evidence, score, status, created, decided}}"
                    ),
                    vec![
                        ("ns", s(&ns)),
                        ("id", s(&id)),
                        ("kind", s(kind)),
                        ("body", dj(body.clone())),
                        ("evidence", strs(&evidence)),
                        ("score", DataValue::from(score)),
                        ("status", s(status)),
                        ("at", DataValue::from(now_micros())),
                    ],
                )?;
                if auto {
                    let ch = self.proposal_apply(&ns, &act, kind, &body, &evidence, "learn:auto")?;
                    applied.push(json!({"proposal": id, "change": ch}));
                }
            }
            proposals.push(prop_entry);
        }
        let maint = if mode == "auto" && !o.dry_run { Some(self.onto_maintain(&ns, false)?) } else { None };
        Ok(json!({"ns": ns, "mode": mode, "untyped": total, "proposals": proposals, "applied": applied, "maintenance": maint}))
    }

    /// Apply a proposal body to the learned layer; returns the change id.
    fn proposal_apply(&'s self, ns: &str, act: &Active, kind: &str, body: &JsonValue, evidence: &[String], by: &str) -> Result<String> {
        let mut doc = self.learned_doc(ns)?;
        match kind {
            "property" => {
                let prop: OntoProp = serde_json::from_value(body["prop"].clone()).into_diagnostic()?;
                let before: Vec<OntoProp> = doc.properties.iter().filter(|p| p.name == prop.name).cloned().collect();
                doc.properties.retain(|p| p.name != prop.name);
                doc.properties.push(prop.clone());
                let id = self.learned_save(ns, doc, "add_property", &prop.name, &before, std::slice::from_ref(&prop), evidence, by)?;
                self.learn_retype(ns, &prop.name, evidence)?;
                Ok(id)
            }
            "vocabulary" => {
                let name = body["prop"].as_str().unwrap_or("");
                let Some(base) = act.prop(name).cloned() else { bail!("ontology: property '{name}' is not active") };
                let before: Vec<OntoProp> = doc.properties.iter().filter(|p| p.name == name).cloned().collect();
                let mut p = before.first().cloned().unwrap_or(base);
                for c in body["cues"].as_array().into_iter().flatten().filter_map(|c| c.as_str()) {
                    if !p.cues.iter().any(|x| x == c) {
                        p.cues.push(c.to_string());
                    }
                }
                if let Some(a) = body["alias"].as_str() {
                    if !p.aliases.iter().any(|x| x == a) && p.label != a {
                        p.aliases.push(a.to_string());
                    }
                }
                doc.properties.retain(|x| x.name != name);
                doc.properties.push(p.clone());
                let id = self.learned_save(ns, doc, "vocabulary", name, &before, std::slice::from_ref(&p), evidence, by)?;
                self.learn_retype(ns, name, evidence)?;
                Ok(id)
            }
            k => bail!("ontology: unknown proposal kind '{k}'"),
        }
    }

    /// Type the evidence claims with the property they taught (subject from
    /// the text), so the cardinality policy uses it from now on.
    fn learn_retype(&'s self, ns: &str, prop: &str, ids: &[String]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let r = self.ol_q("?[id, text] := *kleos_mem{ns: $ns, id, text @ 'NOW'}, is_in(id, $ids)", vec![("ns", s(ns)), ("ids", strs(ids))])?;
        let rows: Vec<DataValue> = r
            .rows
            .iter()
            .map(|x| {
                let text = cell(&x[1]);
                DataValue::List(vec![
                    s(ns),
                    x[0].clone(),
                    s(&learned_name(ns)),
                    s("fact"),
                    crate::ontology_type::subject(&text).map(|v| s(&v)).unwrap_or(DataValue::Null),
                    s(prop),
                    DataValue::from(0.6),
                    s("learn"),
                ])
            })
            .collect();
        if rows.is_empty() {
            return Ok(());
        }
        self.ol_q(
            "?[ns, id, onto, cls, subject, prop, score, by] <- $rows \
             :put kleos_mem_type {ns, id => onto, cls, subject, prop, score, by}",
            vec![("rows", DataValue::List(rows))],
        )?;
        Ok(())
    }

    /// `::ontology maintain`: correct cardinality from the data, merge
    /// learned synonyms, retire unused learned properties.
    pub fn onto_maintain(&'s self, ns: &str, dry_run: bool) -> Result<JsonValue> {
        let Some(act) = self.onto_active(ns)? else {
            return Ok(json!({"ns": ns, "skipped": "no active ontology"}));
        };
        self.onto_ensure()?;
        let mut changes = vec![];
        // typed, current claims per (prop, subject). Only types a cue, a rule,
        // learning or an extractor backs count as evidence: a prototype match
        // alone is too noisy under some embedders to flip, merge or retire a
        // property (EmbeddingGemma 2 made `reports_to` multi-valued from
        // prototype-only types).
        let r = self.ol_q(
            "?[id, prop, subject, text] := *kleos_mem_type{ns: $ns, id, prop, subject, by}, !is_null(prop), \
                 by != 'proto', by != 'none', *kleos_mem{ns: $ns, id, text @ 'NOW'}",
            vec![("ns", s(ns))],
        )?;
        let mut by_prop: BTreeMap<String, BTreeMap<String, Vec<(String, String)>>> = BTreeMap::new();
        let mut used: BTreeMap<String, usize> = BTreeMap::new();
        for x in &r.rows {
            let prop = cell(&x[1]);
            *used.entry(prop.clone()).or_default() += 1;
            if let Some(subj) = x[2].get_str() {
                by_prop.entry(prop).or_default().entry(subj.to_string()).or_default().push((cell(&x[0]), cell(&x[3])));
            }
        }
        let min_subjects = knob("KLEOS_ONTO_FLIP_SUBJECTS", 2.0) as usize;
        let min_values = knob("KLEOS_ONTO_FLIP_VALUES", 3.0) as usize;
        // 1. single-valued properties the data keeps contradicting
        for p in act.props.iter().filter(|p| p.single()) {
            let Some(subj) = by_prop.get(&p.name) else { continue };
            let crowded: Vec<(&String, &Vec<(String, String)>)> = subj.iter().filter(|(_, v)| v.len() >= min_values).collect();
            // values that coexist without update cues are evidence of `many`
            let coexist: Vec<&(&String, &Vec<(String, String)>)> = crowded
                .iter()
                .filter(|(_, v)| v.iter().filter(|(_, t)| has_cue(t)).count() * 2 < v.len())
                .collect();
            if coexist.len() >= min_subjects {
                let evidence: Vec<String> = coexist.iter().flat_map(|(_, v)| v.iter().map(|x| x.0.clone())).take(30).collect();
                let mut np = p.clone();
                np.card = "many".into();
                np.doc = format!("{} [maintenance: made multi-valued — {} subjects hold {}+ current values without update cues]", p.doc, coexist.len(), min_values);
                let entry = json!({"kind": "cardinality", "prop": p.name, "from": "one", "to": "many", "subjects": coexist.len(), "evidence": evidence.len()});
                if !dry_run {
                    let mut doc = self.learned_doc(ns)?;
                    let before: Vec<OntoProp> = doc.properties.iter().filter(|x| x.name == p.name).cloned().collect();
                    doc.properties.retain(|x| x.name != p.name);
                    doc.properties.push(np.clone());
                    let id = self.learned_save(ns, doc, "cardinality", &p.name, &before, std::slice::from_ref(&np), &evidence, "maintain")?;
                    changes.push(json!({"change": id, "detail": entry}));
                } else {
                    changes.push(entry);
                }
            }
        }
        // 2. learned synonyms: same label words, or nearly the same prototype
        let doc = self.learned_doc(ns)?;
        let learned: Vec<OntoProp> = doc.properties.iter().filter(|p| p.doc.starts_with("Learned")).cloned().collect();
        let cfg = self.mem_config()?.unwrap_or_default();
        let base = cfg
            .embedder
            .as_deref()
            .and_then(|e| crate::ontology::prototypes(&act, e).ok())
            .map_or(crate::ontology::BASELINE_REF, |p| crate::ontology::proto_baseline(&p));
        let cal = |t: f64| crate::ontology::calibrated(t, base);
        let lv: Option<Vec<Vec<f32>>> = match cfg.embedder.as_deref() {
            Some(e) if learned.len() >= 2 => {
                let t: Vec<String> = learned.iter().map(crate::ontology::proto_text).collect();
                let refs: Vec<&str> = t.iter().map(|x| x.as_str()).collect();
                crate::embed::embed_texts(e, &refs).ok().map(|v| v.into_iter().map(|x| x.to_vec()).collect())
            }
            _ => None,
        };
        let mut merged: BTreeSet<String> = BTreeSet::new();
        for i in 0..learned.len() {
            for j in (i + 1)..learned.len() {
                let (a, b) = (&learned[i], &learned[j]);
                if merged.contains(&a.name) || merged.contains(&b.name) || a.card != b.card {
                    continue;
                }
                let same_words = {
                    let wa: BTreeSet<String> = content(&a.label).iter().map(|w| stem(w)).collect();
                    let wb: BTreeSet<String> = content(&b.label).iter().map(|w| stem(w)).collect();
                    !wa.is_empty() && wa == wb
                };
                let sim = lv.as_ref().map(|v| cosine(&v[i], &v[j])).unwrap_or(0.0);
                if !(same_words || sim >= cal(knob("KLEOS_ONTO_MERGE_SIM", 0.90))) {
                    continue;
                }
                // keep the one more claims use
                let (keep, drop) = if used.get(&a.name).unwrap_or(&0) >= used.get(&b.name).unwrap_or(&0) { (a, b) } else { (b, a) };
                let mut k = keep.clone();
                for c in &drop.cues {
                    if !k.cues.contains(c) {
                        k.cues.push(c.clone());
                    }
                }
                if !k.aliases.contains(&drop.label) {
                    k.aliases.push(drop.label.clone());
                }
                merged.insert(drop.name.clone());
                let entry = json!({"kind": "merge", "keep": keep.name, "merged": drop.name, "similarity": sim});
                if !dry_run {
                    let mut d = self.learned_doc(ns)?;
                    let before: Vec<OntoProp> = d.properties.iter().filter(|x| x.name == keep.name || x.name == drop.name).cloned().collect();
                    d.properties.retain(|x| x.name != keep.name && x.name != drop.name);
                    d.properties.push(k.clone());
                    let id = self.learned_save(ns, d, "merge", &keep.name, &before, std::slice::from_ref(&k), &[], "maintain")?;
                    // the merged property's claims now carry the kept one
                    self.ol_q(
                        "?[ns, id, prop] := *kleos_mem_type{ns, id, prop: p}, ns = $ns, p = $from, prop = $to \
                         :update kleos_mem_type {ns, id => prop}",
                        vec![("ns", s(ns)), ("from", s(&drop.name)), ("to", s(&keep.name))],
                    )?;
                    changes.push(json!({"change": id, "detail": entry}));
                } else {
                    changes.push(entry);
                }
            }
        }
        // 3. learned properties nothing uses any more
        let retire_after = (knob("KLEOS_ONTO_RETIRE_DAYS", 30.0) * 86_400e6) as i64;
        let added: BTreeMap<String, i64> = self
            .ol_q(
                &format!("?[target, max(at)] := *{ONTO_CHANGE}{{ns: $ns, target, at, kind}}, kind == 'add_property'"),
                vec![("ns", s(ns))],
            )?
            .rows
            .iter()
            .map(|x| (cell(&x[0]), x[1].get_int().unwrap_or(0)))
            .collect();
        let now = now_micros();
        for p in learned.iter().filter(|p| !merged.contains(&p.name)) {
            let idle = added.get(&p.name).is_some_and(|t| now - t > retire_after);
            if idle && used.get(&p.name).copied().unwrap_or(0) == 0 {
                let entry = json!({"kind": "retire", "prop": p.name});
                if !dry_run {
                    let mut d = self.learned_doc(ns)?;
                    let before: Vec<OntoProp> = d.properties.iter().filter(|x| x.name == p.name).cloned().collect();
                    d.properties.retain(|x| x.name != p.name);
                    let id = self.learned_save(ns, d, "retire", &p.name, &before, &[], &[], "maintain")?;
                    changes.push(json!({"change": id, "detail": entry}));
                } else {
                    changes.push(entry);
                }
            }
        }
        Ok(json!({"ns": ns, "dry_run": dry_run, "changes": changes}))
    }

    /// `::ontology proposals`.
    pub fn onto_proposals(&'s self, ns: &str, status: Option<&str>) -> Result<JsonValue> {
        self.onto_ensure()?;
        let r = self.ol_q(
            &format!("?[id, kind, body, evidence, score, status, created, decided] := *{ONTO_PROPOSAL}{{ns: $ns, id, kind, body, evidence, score, status, created, decided}}"),
            vec![("ns", s(ns))],
        )?;
        let out: Vec<JsonValue> = r
            .rows
            .iter()
            .filter(|x| status.map_or(true, |st| cell(&x[5]) == st))
            .map(|x| {
                json!({"id": cell(&x[0]), "kind": cell(&x[1]), "body": json_cell(&x[2]), "evidence": cell_strs(&x[3]),
                       "score": x[4].get_float(), "status": cell(&x[5]), "created": x[6].get_int(), "decided": x[7].get_int()})
            })
            .collect();
        Ok(JsonValue::Array(out))
    }

    /// `::ontology approve` / `reject` of a proposal (review mode).
    pub fn onto_decide_proposal(&'s self, ns: &str, id: &str, approve: bool) -> Result<JsonValue> {
        self.onto_ensure()?;
        let r = self.ol_q(
            &format!("?[kind, body, evidence, status] := *{ONTO_PROPOSAL}{{ns: $ns, id: $id, kind, body, evidence, status}}"),
            vec![("ns", s(ns)), ("id", s(id))],
        )?;
        let Some(x) = r.rows.first() else { bail!("ontology: no proposal '{id}' in namespace '{ns}'") };
        let (kind, body, evidence, status) = (cell(&x[0]), json_cell(&x[1]), cell_strs(&x[2]), cell(&x[3]));
        if status != "proposed" {
            bail!("ontology: proposal '{id}' is {status}, not proposed");
        }
        let mut change = JsonValue::Null;
        if approve {
            let Some(act) = self.onto_active(ns)? else { bail!("ontology: namespace '{ns}' has no active ontology") };
            change = json!(self.proposal_apply(ns, &act, &kind, &body, &evidence, "review")?);
        }
        let st = if approve { "active" } else { "rejected" };
        self.ol_q(
            &format!("?[ns, id, status, decided] <- [[$ns, $id, $st, $at]] :update {ONTO_PROPOSAL} {{ns, id => status, decided}}"),
            vec![("ns", s(ns)), ("id", s(id)), ("st", s(st)), ("at", DataValue::from(now_micros()))],
        )?;
        Ok(json!({"ns": ns, "id": id, "status": st, "change": change}))
    }

    /// `::ontology history`: the learned layer's changes, newest first.
    pub fn onto_history(&'s self, ns: &str) -> Result<JsonValue> {
        self.onto_ensure()?;
        let r = self.ol_q(
            &format!(
                "?[id, at, kind, onto, target, before, after, evidence, by, undone] := \
                 *{ONTO_CHANGE}{{ns: $ns, id, at, kind, onto, target, before, after, evidence, by, undone}} :order -at"
            ),
            vec![("ns", s(ns))],
        )?;
        Ok(JsonValue::Array(
            r.rows
                .iter()
                .map(|x| {
                    json!({"id": cell(&x[0]), "at": x[1].get_int(), "kind": cell(&x[2]), "onto": cell(&x[3]),
                           "target": cell(&x[4]), "before": json_cell(&x[5]), "after": json_cell(&x[6]),
                           "evidence": cell_strs(&x[7]), "by": cell(&x[8]), "undone": x[9].get_bool()})
                })
                .collect(),
        ))
    }

    /// `::ontology undo`: put back the properties a change replaced (and
    /// drop those it added), as a new version; the change is marked undone.
    pub fn onto_undo(&'s self, ns: &str, change: &str) -> Result<JsonValue> {
        self.onto_ensure()?;
        let r = self.ol_q(
            &format!("?[kind, target, before, after, undone] := *{ONTO_CHANGE}{{ns: $ns, id: $id, kind, target, before, after, undone}}"),
            vec![("ns", s(ns)), ("id", s(change))],
        )?;
        let Some(x) = r.rows.first() else { bail!("ontology: no change '{change}' in namespace '{ns}'") };
        if x[4].get_bool() == Some(true) {
            bail!("ontology: change '{change}' is already undone");
        }
        let target = cell(&x[1]);
        let before: Vec<OntoProp> = serde_json::from_value(json_cell(&x[2])).into_diagnostic()?;
        let after: Vec<OntoProp> = serde_json::from_value(json_cell(&x[3])).into_diagnostic()?;
        let mut doc = self.learned_doc(ns)?;
        let names: BTreeSet<String> = before.iter().chain(&after).map(|p| p.name.clone()).collect();
        doc.properties.retain(|p| !names.contains(&p.name));
        doc.properties.extend(before.iter().cloned());
        let id = self.learned_save(ns, doc, "undo", &target, &after, &before, &[change.to_string()], "undo")?;
        self.ol_q(
            &format!("?[ns, id, undone] <- [[$ns, $id, true]] :update {ONTO_CHANGE} {{ns, id => undone}}"),
            vec![("ns", s(ns)), ("id", s(change))],
        )?;
        Ok(json!({"ns": ns, "undone": change, "change": id}))
    }

    /// `::ontology suggest`: domain packs the namespace's documents and
    /// claims point at (a review-only proposal: importing a pack is the
    /// user's call), plus a learning dry run over them.
    pub fn onto_suggest(&'s self, ns: &str) -> Result<JsonValue> {
        self.onto_ensure()?;
        let r = self.ol_q("?[text] := *kleos_mem{ns: $ns, text @ 'NOW'} :limit 3000", vec![("ns", s(ns))])?;
        let texts: Vec<String> = r.rows.iter().map(|x| cell(&x[0]).to_lowercase()).collect();
        let n = texts.len().max(1) as f64;
        let mut out = vec![];
        for (pack, source, terms) in DOMAIN_PACKS {
            let hits = texts.iter().filter(|t| terms.iter().any(|w| crate::ontology_type::has_phrase(t, w))).count();
            let share = hits as f64 / n;
            if hits >= 5 && share >= 0.05 {
                let id = format!("pr_pack_{}", snake(pack));
                let body = json!({"pack": pack, "source": source, "share": share, "hits": hits,
                    "how": format!("import it with `::ontology import {{format: 'turtle', ...}}` and activate it after the personal pack")});
                self.ol_q(
                    &format!(
                        "?[ns, id, kind, body, evidence, score, status, created, decided] <- \
                         [[$ns, $id, 'pack', $body, [], $score, 'proposed', $at, null]] \
                         :put {ONTO_PROPOSAL} {{ns, id => kind, body, evidence, score, status, created, decided}}"
                    ),
                    vec![("ns", s(ns)), ("id", s(&id)), ("body", dj(body.clone())), ("score", DataValue::from(share)), ("at", DataValue::from(now_micros()))],
                )?;
                out.push(json!({"id": id, "body": body}));
            }
        }
        let dry = self.onto_learn(LearnOpts { ns: Some(ns.to_string()), dry_run: true, ..Default::default() })?;
        Ok(json!({"ns": ns, "packs": out, "learn_preview": dry["proposals"]}))
    }
}

/// Domain packs `suggest` can point at: (name, where to get it, signal terms).
const DOMAIN_PACKS: &[(&str, &str, &[&str])] = &[
    (
        "fibo",
        "EDM Council FIBO (MIT), https://spec.edmcouncil.org/fibo/",
        &["counterparty", "notional", "bond", "equity", "derivative", "swap", "trade", "portfolio", "custodian", "isin", "ticker", "hedge", "exposure", "settlement", "securities"],
    ),
    (
        "schema.org MedicalEntity",
        "schema.org (CC BY-SA 3.0), https://schema.org/MedicalEntity",
        &["diagnosis", "patient", "dosage", "medication", "symptom", "prescription", "clinical", "treatment plan", "mg"],
    ),
    (
        "schema.org Legal / LKIF",
        "LKIF-Core (open), https://github.com/RinkeHoekstra/lkif-core",
        &["clause", "plaintiff", "defendant", "agreement", "indemnif*", "termination", "governing law", "liability", "contract"],
    ),
    (
        "schema.org SoftwareSourceCode",
        "schema.org (CC BY-SA 3.0), https://schema.org/SoftwareSourceCode",
        &["repository", "pull request", "commit", "deploy", "endpoint", "bug", "release", "api", "function", "module"],
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clusters_by_direction() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.98, 0.1, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        let d = vec![0.05, 0.99, 0.0];
        let e = vec![0.0, 0.0, 1.0];
        let cl = cluster(&[a, b, c, d, e], 0.9);
        assert_eq!(cl.len(), 3);
        assert_eq!(cl[0].0.len(), 2);
        assert_eq!(cl[1].0.len(), 2);
    }

    #[test]
    fn near_identical_clusters_merge() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.99, 0.12, 0.0];
        let c = vec![0.93, 0.36, 0.0];
        let d = vec![0.92, 0.39, 0.0];
        let cl = cluster(&[a.clone(), b.clone(), c.clone(), d.clone()], 0.97);
        // leader pass splits {a,b} and {c,d}; their centroids are ~0.96 apart > 0.92: merged
        assert_eq!(cl.len(), 1, "{cl:?}");
        let far = vec![0.0, 0.0, 1.0];
        let cl = cluster(&[a, b, far.clone(), far], 0.9);
        assert_eq!(cl.len(), 2);
    }

    #[test]
    fn labels_from_distinctive_words() {
        let cluster = [
            "My Instagram follower count is 1250.",
            "I have 1300 followers on Instagram now.",
            "My Instagram follower count went up to 1400.",
            "Instagram followers: 1500 as of today.",
        ];
        let pool: Vec<&str> = cluster.iter().copied().chain(["I love sushi.", "I went hiking with Anna.", "My rent is due Friday.", "The weather was nice in Rome."]).collect();
        let df = pool_df(&pool);
        let d = distinctive(&cluster, &df, pool.len(), &BTreeSet::new());
        assert!(d[0].0.contains("instagram") || d[0].0.contains("follower"), "{d:?}");
    }

    #[test]
    fn phrase_and_role_labels() {
        let cluster = [
            "The data team now has 12 engineers.",
            "Headcount update: the platform team is at 9 engineers now.",
            "The search team grew to 7 engineers this quarter.",
            "The mobile team now has 11 engineers.",
        ];
        assert_eq!(counted_noun(&cluster).as_deref(), Some("engineers"));
        assert_eq!(counted_noun(&["I love sushi.", "My rent is 900 euros.", "Rome was nice."]), None);
        assert_eq!(subject_words("team:data platform"), vec!["data", "platform", "team"]);
        assert!(subject_words("name:cloud").is_empty());
        assert_eq!(subject_words("user.sister"), vec!["sister"]);
        let vendors = [
            "The data team switched its cloud vendor to AWS.",
            "Cloud vendor for payments is now GCP.",
            "Our mobile workloads now run on Azure as the cloud vendor.",
            "The search team moved to a new cloud vendor, Hetzner.",
        ];
        let pool: Vec<&str> = vendors.iter().copied().chain(["I love sushi.", "Lunch is on the third floor.", "The checkout service had an outage on Monday."]).collect();
        let d = distinctive(&vendors, &pool_df(&pool), pool.len(), &BTreeSet::new());
        assert_eq!(d[0].0, "cloud vendor", "{d:?}");
        // weekdays are values, not labels
        let outages = ["The checkout service had an outage on Monday.", "Incident: billing was down on Monday.", "The login service had an outage on Friday."];
        let d = distinctive(&outages, &pool_df(&outages), outages.len(), &BTreeSet::new());
        let l = LABEL_STOP;
        assert!(d.iter().all(|t| !t.0.split(' ').any(|w| w == "monday") || !l.contains(&"monday")), "{d:?}");
    }

    #[test]
    fn greetings_and_names_are_not_labels() {
        let cluster = [
            "Caroline: Thanks, Mel. The pottery class was great.",
            "Caroline: Hey Mel! I signed up for another pottery class.",
            "Caroline: Thanks, Melanie! My pottery bowl came out well.",
            "Caroline: Thanks! The pottery class starts Monday.",
        ];
        let pool: Vec<&str> = cluster.iter().copied().chain(["Melanie: I went camping.", "Melanie: The kids loved the beach."]).collect();
        let names = pool_names(&pool);
        assert!(names.contains("mel") && names.contains("caroline") && names.contains("melanie"), "{names:?}");
        let d = distinctive(&cluster, &pool_df(&pool), pool.len(), &names);
        assert!(d[0].0.contains("pottery"), "{d:?}");
        assert!(d.iter().all(|t| !t.0.contains("thanks") && !t.0.contains("mel")), "{d:?}");
        assert_eq!(stem("dancing"), stem("dances"));
    }

    #[test]
    fn cardinality_from_the_data() {
        let t = ["My rent is now €1,500 per month.", "My rent went up to €1,600.", "My rent is €1,400."];
        let subj = vec![Some("user".to_string()); 3];
        let sh = shape(&t, &subj, None);
        assert_eq!(sh.card, "one", "{sh:?}");
        assert!(sh.numeric);
        let t = ["I visited Rome.", "I visited Lisbon.", "I visited Oslo with my sister."];
        let sh = shape(&t, &subj, None);
        assert_eq!(sh.card, "many", "{sh:?}");
        // the decider overrides the cues
        let sh = shape(&t, &subj, Some(0.95));
        assert_eq!(sh.card, "one");
        // statements remember already replaced are the strongest evidence
        let sh = shape_with(&t, &subj, Some(0.2), 3);
        assert_eq!(sh.card, "one", "{sh:?}");
    }

    #[test]
    fn trigger_counts_per_namespace() {
        std::env::set_var("KLEOS_ONTO_LEARN_EVERY", "3");
        assert!(!note_untyped("t_ns", 1));
        assert!(!note_untyped("t_ns", 1));
        assert!(note_untyped("t_ns", 1));
        assert!(!note_untyped("t_ns", 1));
        std::env::remove_var("KLEOS_ONTO_LEARN_EVERY");
    }

    #[test]
    fn snake_names() {
        assert_eq!(snake("follower count"), "follower_count");
        assert_eq!(snake("Instagram — followers!"), "instagram_followers");
    }
}
