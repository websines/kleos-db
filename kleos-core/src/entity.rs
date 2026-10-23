/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Entity resolution (UPGRADE_PLAN E3; docs/specs/entity-resolution.md).
//!
//! Mentions ("Alice Smith", "A. Smith", "Alice (my manager)") are resolved
//! against a user's entity relation, configured by relation and column names
//! ([`EntityConfig`]):
//!
//! 1. **Blocking.** A cheap rule tier (exact normalized name or alias,
//!    initials, nicknames, acronyms, first name / surname) over an in-memory
//!    snapshot of the relation, unioned with the relation's MinHash LSH,
//!    FTS and HNSW indices when they are configured. Same or compatible
//!    entity types only.
//! 2. **Confirmation.** Unambiguous rule hits are decided by the rule;
//!    everything else goes to `decide()` with the `same_entity` template in
//!    one batch. Auto → match (merge), Tentative → a "possible same-as" link
//!    for review, None → a new entity. Without a decider (ML off) the rule
//!    tier and MinHash similarity decide alone; resolution never errors
//!    because a model is off.
//! 3. **Non-destructive merges.** A merge is a row in the valid-time relation
//!    `kleos_entity_merge {relation, from, vt => into, p, decided_by,
//!    reason}`; `split` appends a retraction, so the merge history stays
//!    queryable with `@`, `::history` and [`Db::entity_history`]. The
//!    canonical entity of an id is found by following current merges.
//!
//! Each decision is made between a mention and one **canonical** entity
//! profile (its name, the names of every row merged into it, type and
//! description). Merges are never inferred transitively: A ≈ B and B ≈ C
//! gives two links, and C joins A only through its own decision.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use miette::{miette, Result};
use ndarray::Array1;
use serde_derive::{Deserialize, Serialize};
use unicode_normalization::char::is_combining_mark;
use unicode_normalization::UnicodeNormalization;

use crate::bail;
use crate::data::value::{DataValue, Validity, Vector};
use crate::decide::{self, Band, Calibration, Template};
use crate::parse::sys::EntityOp;
use crate::runtime::db::{Db, NamedRows, ScriptMutability};
use crate::storage::Storage;

/// Valid-time relation holding merges (`from` merged into `into`).
pub const ENTITY_MERGES: &str = "kleos_entity_merge";
/// Relation holding "possible same-as" links awaiting review.
pub const ENTITY_LINKS: &str = "kleos_entity_link";

// ------------------------------------------------------------------ config

/// Where a user's entities live and which candidate sources to use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityConfig {
    /// The entity relation.
    pub relation: String,
    /// Key column (one column; generated ids are strings).
    pub id: String,
    /// Name column (a string).
    pub name: String,
    /// Optional entity type column (a string).
    pub type_col: Option<String>,
    /// Optional aliases column (a list of strings).
    pub aliases: Option<String>,
    /// Optional description column (a string).
    pub description: Option<String>,
    /// Optional embedding column (a vector; nullable if the embedder may be off).
    pub embedding: Option<String>,
    /// MinHash LSH index on the name column (`relation:<lsh>`), ideally
    /// with `tokenizer: NGram(3, 3, false), n_gram: 1` (character trigrams).
    pub lsh: Option<String>,
    /// FTS index on the name (or name + description) column.
    pub fts: Option<String>,
    /// HNSW index on the embedding column.
    pub hnsw: Option<String>,
    /// Embedder used for mention vectors and for new rows' embeddings.
    pub embedder: Option<String>,
    /// Decider for `same_entity`; `None` = rules and MinHash only.
    pub decider: Option<String>,
    /// Groups of types that may match each other (e.g. `[org, company]`).
    pub compatible_types: Vec<Vec<String>>,
    /// Hits per index source and mention.
    pub k: usize,
    /// Candidates sent to the decider per mention.
    pub max_candidates: usize,
    /// Override the decider's `same_entity` label threshold.
    pub threshold: Option<f64>,
    /// Override the decider's `same_entity` auto threshold.
    pub tau_auto: Option<f64>,
    /// Let an exact, unique normalized-name match auto-resolve without the
    /// decider even when both sides carry context (default: only for long,
    /// specific names, see the spec).
    pub exact_auto: bool,
    /// Resolve within one partition of the relation only: rows whose column
    /// `.0` equals `.1` (the memory API passes `("ns", <namespace>)`). The
    /// scope column is then part of the key, before the id column; index
    /// hits outside the scope are dropped.
    pub scope: Option<(String, DataValue)>,
    /// Memory branches (docs/specs/memory-branches.md): when non-empty, the
    /// snapshot reads the rows of every `(scope value, cutoff)` instead of
    /// `scope` alone, a cutoff keeping the rows whose `created` column is at
    /// or before it. The memory API sets it for a branch's lineage.
    #[serde(skip)]
    pub lineage: Vec<(DataValue, Option<i64>)>,
}

impl EntityConfig {
    /// Defaults: key `id`, name `name`, no other columns or indices, no
    /// decider, k 10, 5 candidates per mention.
    pub fn new(relation: impl Into<String>) -> Self {
        Self {
            relation: relation.into(),
            id: "id".into(),
            name: "name".into(),
            type_col: None,
            aliases: None,
            description: None,
            embedding: None,
            lsh: None,
            fts: None,
            hnsw: None,
            embedder: None,
            decider: None,
            compatible_types: vec![],
            k: 10,
            max_candidates: 5,
            threshold: None,
            tau_auto: None,
            exact_auto: false,
            scope: None,
            lineage: vec![],
        }
    }

    /// Set an option by name (the keys of `::entity resolve {...}`).
    pub fn set(&mut self, key: &str, val: &DataValue) -> Result<()> {
        let s = |v: &DataValue| -> Result<Option<String>> {
            match v {
                DataValue::Null => Ok(None),
                DataValue::Str(s) => Ok(Some(s.to_string())),
                v => bail!("entity option '{key}' must be a string, got {v:?}"),
            }
        };
        let n = |v: &DataValue| -> Result<usize> {
            match v.get_int() {
                Some(i) if i > 0 => Ok(i as usize),
                _ => bail!("entity option '{key}' must be a positive integer, got {v:?}"),
            }
        };
        let f = |v: &DataValue| -> Result<Option<f64>> {
            match v {
                DataValue::Null => Ok(None),
                v => match v.get_float() {
                    Some(x) => Ok(Some(x)),
                    None => bail!("entity option '{key}' must be a number, got {v:?}"),
                },
            }
        };
        match key {
            "id" => self.id = s(val)?.ok_or_else(|| miette!("entity option 'id' cannot be null"))?,
            "name" => self.name = s(val)?.ok_or_else(|| miette!("entity option 'name' cannot be null"))?,
            "type" => self.type_col = s(val)?,
            "aliases" => self.aliases = s(val)?,
            "description" => self.description = s(val)?,
            "embedding" => self.embedding = s(val)?,
            "lsh" => self.lsh = s(val)?,
            "fts" => self.fts = s(val)?,
            "hnsw" => self.hnsw = s(val)?,
            "embedder" => self.embedder = s(val)?,
            "decider" => self.decider = s(val)?,
            "k" => self.k = n(val)?,
            "max_candidates" => self.max_candidates = n(val)?,
            "threshold" => self.threshold = f(val)?,
            "tau_auto" => self.tau_auto = f(val)?,
            "exact_auto" => {
                self.exact_auto = val
                    .get_bool()
                    .ok_or_else(|| miette!("entity option 'exact_auto' must be a boolean"))?
            }
            "compatible_types" => {
                let groups = match val {
                    DataValue::List(l) => l,
                    v => bail!("entity option 'compatible_types' must be a list of lists of strings, got {v:?}"),
                };
                self.compatible_types = groups
                    .iter()
                    .map(|g| match g {
                        DataValue::List(l) => l
                            .iter()
                            .map(|t| {
                                t.get_str()
                                    .map(|s| s.to_lowercase())
                                    .ok_or_else(|| miette!("entity types must be strings"))
                            })
                            .collect::<Result<Vec<_>>>(),
                        v => bail!("entity option 'compatible_types' must be a list of lists, got {v:?}"),
                    })
                    .collect::<Result<_>>()?;
            }
            other => bail!(
                "unknown entity option '{other}': expected id, name, type, aliases, description, embedding, lsh, fts, hnsw, embedder, decider, compatible_types, k, max_candidates, threshold, tau_auto, exact_auto or commit"
            ),
        }
        Ok(())
    }

    fn types_compatible(&self, a: Option<&str>, b: Option<&str>) -> bool {
        match (a, b) {
            (Some(a), Some(b)) => {
                let (a, b) = (a.to_lowercase(), b.to_lowercase());
                a == b
                    || self
                        .compatible_types
                        .iter()
                        .any(|g| g.contains(&a) && g.contains(&b))
            }
            _ => true,
        }
    }
}

/// One mention to resolve.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Mention {
    /// The mention's own row id, when it already is (or should become) a
    /// row of the entity relation. Generated on commit when `None`.
    pub id: Option<DataValue>,
    /// The surface name ("A. Smith", "Alice (my manager)").
    pub name: String,
    /// Entity type ("person", "org", ...).
    pub entity_type: Option<String>,
    /// Other names known for it.
    pub aliases: Vec<String>,
    /// What it is ("my manager at Acme").
    pub description: Option<String>,
    /// Text around the mention, passed to the decider as state.
    pub context: Option<String>,
}

impl Mention {
    /// A mention with just a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }
    /// Set the type.
    pub fn with_type(mut self, t: impl Into<String>) -> Self {
        self.entity_type = Some(t.into());
        self
    }
    /// Set the description.
    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = Some(d.into());
        self
    }
    /// Set the context.
    pub fn with_context(mut self, c: impl Into<String>) -> Self {
        self.context = Some(c.into());
        self
    }
    /// Set the id.
    pub fn with_id(mut self, id: impl Into<DataValue>) -> Self {
        self.id = Some(id.into());
        self
    }
    /// Set aliases.
    pub fn with_aliases(mut self, a: &[&str]) -> Self {
        self.aliases = a.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Parse a mention from a string (the name), a list `[name, type?,
    /// context?, id?]` or a JSON object with `name`, `type`, `aliases`,
    /// `description`, `context`, `id`.
    pub fn from_value(v: &DataValue) -> Result<Self> {
        let opt = |v: Option<&DataValue>| -> Option<String> {
            match v {
                Some(DataValue::Str(s)) => Some(s.to_string()),
                _ => None,
            }
        };
        match v {
            DataValue::Str(s) => Ok(Mention::new(s.as_str())),
            DataValue::List(l) if !l.is_empty() => {
                let name = opt(l.first()).ok_or_else(|| miette!("a mention list starts with the name, got {v:?}"))?;
                Ok(Mention {
                    name,
                    entity_type: opt(l.get(1)),
                    context: opt(l.get(2)),
                    id: l.get(3).filter(|x| !matches!(x, DataValue::Null)).cloned(),
                    ..Default::default()
                })
            }
            DataValue::Json(j) => {
                let o = j
                    .0
                    .as_object()
                    .ok_or_else(|| miette!("a mention object must be a JSON object, got {v:?}"))?;
                let s = |k: &str| o.get(k).and_then(|x| x.as_str()).map(|x| x.to_string());
                let name = s("name").ok_or_else(|| miette!("a mention object needs a 'name'"))?;
                let id = o.get("id").and_then(|x| match x {
                    serde_json::Value::Null => None,
                    serde_json::Value::String(s) => Some(DataValue::from(s.as_str())),
                    serde_json::Value::Number(n) => n
                        .as_i64()
                        .map(DataValue::from)
                        .or_else(|| n.as_f64().map(DataValue::from)),
                    _ => None,
                });
                let aliases = o
                    .get("aliases")
                    .and_then(|a| a.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
                    .unwrap_or_default();
                Ok(Mention {
                    id,
                    name,
                    entity_type: s("type"),
                    aliases,
                    description: s("description"),
                    context: s("context"),
                })
            }
            v => bail!("a mention must be a string, a list [name, type, context, id] or a JSON object, got {v:?}"),
        }
    }
}

/// What happened to a mention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    /// Resolved to an existing entity (Auto band, or an unambiguous rule).
    Matched,
    /// Possibly the same as an existing entity (Tentative): linked for review.
    Possible,
    /// A new entity.
    New,
}

impl Outcome {
    /// `match`, `possible` or `new`.
    pub fn name(self) -> &'static str {
        match self {
            Outcome::Matched => "match",
            Outcome::Possible => "possible",
            Outcome::New => "new",
        }
    }
}

/// One candidate entity for a mention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    /// Canonical entity id.
    pub entity: DataValue,
    /// Its name.
    pub name: String,
    /// Which sources proposed it: `rule`, `lsh`, `fts`, `vector`.
    pub sources: Vec<String>,
    /// The name rule that fired (`exact`, `alias`, `initial`, `nickname`,
    /// `first_name`, `surname`, `acronym`, `subset`, `middle_name`).
    pub rule: Option<String>,
    /// Character-trigram Jaccard of the best pair of names.
    pub jaccard: f64,
    /// P(same) when decided.
    pub p: Option<f64>,
    /// Band when decided.
    pub band: Option<Band>,
    /// `decided_by` when decided.
    pub decided_by: Option<String>,
    /// The entity profile the decider saw (for audit and for fitting probes).
    #[serde(default)]
    pub evidence: Option<String>,
}

/// The resolution of one mention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Resolution {
    /// Index of the mention in the input.
    pub mention: usize,
    /// The mention's name.
    pub name: String,
    /// Matched, possible or new.
    pub outcome: Outcome,
    /// The canonical entity matched (or the best possible one).
    pub entity: Option<DataValue>,
    /// The mention's own row id (given, or created on commit).
    pub id: Option<DataValue>,
    /// P(same) of the chosen candidate (1 for rule decisions, 0 for new).
    pub p: f64,
    /// Band of the chosen candidate.
    pub band: Band,
    /// `model|same_entity@vN|via`, or `rule|<rule>`.
    pub decided_by: String,
    /// Why (rule name, ambiguity, "no candidates", ...).
    pub reason: Option<String>,
    /// Every candidate considered, best first.
    pub candidates: Vec<Candidate>,
    /// The mention as the decider saw it.
    #[serde(default)]
    pub text: String,
}

/// Options of a resolve call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResolveOptions {
    /// Write: create rows for mentions, merges for matches and links for
    /// possible matches. Otherwise only report.
    pub commit: bool,
}

// ------------------------------------------------------------------ names

const HONORIFICS: &[&str] = &[
    "mr", "mrs", "ms", "miss", "mx", "dr", "prof", "professor", "sir", "dame", "madam", "rev", "fr", "hon",
    "capt", "captain", "col", "gen", "lt", "sgt", "officer", "aunt", "auntie", "uncle", "grandma", "grandpa",
];
const NAME_SUFFIXES: &[&str] = &["jr", "sr", "ii", "iii", "iv", "phd", "md", "esq", "dds"];
const ORG_SUFFIXES: &[&str] = &[
    "inc", "incorporated", "ltd", "limited", "llc", "llp", "corp", "corporation", "co", "plc", "gmbh", "ag",
    "sa", "bv", "nv", "pty", "company",
];
const ACRONYM_STOP: &[&str] = &["of", "the", "and", "for", "de", "in", "at", "a", "an"];

/// Groups of interchangeable given names (common English nicknames).
const NICKNAMES: &[&[&str]] = &[
    &["robert", "rob", "bob", "bobby", "robbie", "bert"],
    &["william", "will", "bill", "billy", "liam", "willy"],
    &["richard", "rich", "rick", "ricky", "dick", "richie"],
    &["james", "jim", "jimmy", "jamie"],
    &["john", "johnny", "jack", "jon"],
    &["jonathan", "jon", "jonny", "nathan"],
    &["michael", "mike", "mikey", "mick", "mickey"],
    &["elizabeth", "liz", "lizzie", "beth", "betty", "eliza", "libby", "betsy"],
    &["katherine", "catherine", "kathryn", "kate", "katie", "kathy", "cathy", "kat", "kit"],
    &["margaret", "maggie", "meg", "peggy", "marge", "margie"],
    &["thomas", "tom", "tommy"],
    &["joseph", "joe", "joey"],
    &["daniel", "dan", "danny"],
    &["david", "dave", "davey"],
    &["christopher", "chris", "kit"],
    &["christine", "christina", "chris", "tina", "chrissy"],
    &["alexander", "alex", "al", "sasha", "xander"],
    &["alexandra", "alex", "lexi", "sasha", "sandra"],
    &["anthony", "tony"],
    &["andrew", "andy", "drew"],
    &["benjamin", "ben", "benny"],
    &["charles", "charlie", "chuck", "chas"],
    &["edward", "ed", "eddie", "ted", "ned"],
    &["patricia", "pat", "patty", "trish"],
    &["patrick", "pat", "paddy"],
    &["jennifer", "jen", "jenny"],
    &["rebecca", "becky", "becca"],
    &["samuel", "sam", "sammy"],
    &["samantha", "sam", "sammy"],
    &["stephen", "steven", "steve"],
    &["susan", "sue", "suzy"],
    &["timothy", "tim", "timmy"],
    &["victoria", "vicky", "tori"],
    &["matthew", "matt"],
    &["nicholas", "nick", "nicky"],
    &["gregory", "greg"],
    &["jessica", "jess", "jessie"],
    &["deborah", "debbie", "deb"],
    &["kenneth", "ken", "kenny"],
    &["lawrence", "larry"],
    &["peter", "pete"],
    &["raymond", "ray"],
    &["ronald", "ron", "ronnie"],
    &["donald", "don", "donnie"],
    &["gerald", "gerry", "jerry"],
    &["francis", "frank", "frankie"],
    &["frederick", "fred", "freddie"],
    &["henry", "harry", "hank"],
    &["abigail", "abby"],
    &["allison", "alison", "ally", "allie"],
    &["amanda", "mandy"],
    &["barbara", "barb", "babs"],
    &["dorothy", "dot", "dottie"],
    &["eleanor", "ellie", "nora"],
    &["josephine", "jo", "josie"],
    &["judith", "judy"],
    &["kimberly", "kim"],
    &["leonard", "leo", "len", "lenny"],
    &["louis", "lou"],
    &["melissa", "mel", "missy"],
    &["nathaniel", "nate", "nat"],
    &["philip", "phillip", "phil"],
    &["theodore", "theo", "ted", "teddy"],
    &["vincent", "vince", "vinny"],
    &["zachary", "zach", "zack"],
];

fn nickname_equal(a: &str, b: &str) -> bool {
    a != b && NICKNAMES.iter().any(|g| g.contains(&a) && g.contains(&b))
}

/// A name split into its normalized core and a parenthetical remark
/// ("Alice (my manager)" → `alice`, `my manager`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedName {
    /// Normalized tokens (lower case, no accents or punctuation, no
    /// honorifics, name suffixes or company suffixes).
    pub tokens: Vec<String>,
    /// Text in parentheses, if any.
    pub remark: Option<String>,
}

impl ParsedName {
    /// The normalized name: tokens joined by single spaces.
    pub fn norm(&self) -> String {
        self.tokens.join(" ")
    }
}

/// Parse and normalize a name.
pub fn parse_name(raw: &str) -> ParsedName {
    let mut core = String::with_capacity(raw.len());
    let mut remark = String::new();
    let mut depth = 0usize;
    for c in raw.chars() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' if depth > 0 => {
                depth -= 1;
                remark.push(' ');
            }
            _ if depth > 0 => remark.push(c),
            _ => core.push(c),
        }
    }
    let mut cleaned = String::with_capacity(core.len());
    for c in core.nfkd().filter(|c| !is_combining_mark(*c)) {
        if c == '\'' || c == '’' || c == '`' {
            continue; // O'Brien → obrien
        }
        if c == '&' {
            cleaned.push_str(" and ");
        } else if c.is_alphanumeric() {
            cleaned.extend(c.to_lowercase());
        } else {
            cleaned.push(' ');
        }
    }
    let mut tokens: Vec<String> = cleaned.split_whitespace().map(|s| s.to_string()).collect();
    // possessive "alices" is left alone; honorifics, suffixes
    while tokens.len() > 1 && HONORIFICS.contains(&tokens[0].as_str()) {
        tokens.remove(0);
    }
    while tokens.len() > 1 && NAME_SUFFIXES.contains(&tokens[tokens.len() - 1].as_str()) {
        tokens.pop();
    }
    while tokens.len() > 1 && ORG_SUFFIXES.contains(&tokens[tokens.len() - 1].as_str()) {
        tokens.pop();
    }
    if tokens.len() > 1 && tokens[0] == "the" {
        tokens.remove(0);
    }
    let remark = remark.split_whitespace().collect::<Vec<_>>().join(" ");
    ParsedName {
        tokens,
        remark: if remark.is_empty() { None } else { Some(remark) },
    }
}

/// Normalized form of a name (see [`parse_name`]).
pub fn normalize_name(raw: &str) -> String {
    parse_name(raw).norm()
}

fn acronym(tokens: &[String]) -> Option<String> {
    let t: Vec<&String> = tokens.iter().filter(|t| !ACRONYM_STOP.contains(&t.as_str())).collect();
    if t.len() < 2 {
        return None;
    }
    Some(t.iter().filter_map(|t| t.chars().next()).collect())
}

/// Which name rule relates two normalized token lists, if any (not
/// `exact`, which is a string comparison).
pub fn name_rule(a: &[String], b: &[String]) -> Option<&'static str> {
    if a.is_empty() || b.is_empty() || a == b {
        return None;
    }
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if short.len() == 1 && long.len() >= 2 {
        let t = &short[0];
        if t.len() >= 2 && acronym(long).as_deref() == Some(t.as_str()) {
            return Some("acronym");
        }
        if t.len() >= 2 && (*t == long[0] || nickname_equal(t, &long[0])) {
            return Some("first_name");
        }
        if t.len() >= 3 && *t == long[long.len() - 1] {
            return Some("surname");
        }
        return None;
    }
    if short.len() >= 2 && long.len() >= 2 {
        let (fa, fb) = (&a[0], &b[0]);
        let same_last = a[a.len() - 1] == b[b.len() - 1];
        if same_last {
            if fa == fb {
                return Some("middle_name");
            }
            let initial = |x: &String, y: &String| x.len() == 1 && y.starts_with(x.as_str());
            if initial(fa, fb) || initial(fb, fa) {
                return Some("initial");
            }
            if nickname_equal(fa, fb) {
                return Some("nickname");
            }
        }
        let long_set: BTreeSet<&String> = long.iter().collect();
        if short.iter().all(|t| long_set.contains(t)) {
            return Some("subset");
        }
    }
    None
}

fn trigrams(s: &str) -> BTreeSet<[char; 3]> {
    let padded: Vec<char> = format!("  {s} ").chars().collect();
    padded.windows(3).map(|w| [w[0], w[1], w[2]]).collect()
}

/// Jaccard similarity of the character-trigram sets of two normalized names.
pub fn trigram_jaccard(a: &str, b: &str) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let (x, y) = (trigrams(a), trigrams(b));
    let inter = x.intersection(&y).count() as f64;
    let union = (x.len() + y.len()) as f64 - inter;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Graphiti's entropy gate: fuzzy similarity is trusted only for names that
/// are long and varied enough ("Sam" and "AI" are not).
pub fn name_is_specific(norm: &str) -> bool {
    let chars: Vec<char> = norm.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.is_empty() {
        return false;
    }
    let mut counts: BTreeMap<char, usize> = BTreeMap::new();
    for c in &chars {
        *counts.entry(*c).or_default() += 1;
    }
    let n = chars.len() as f64;
    let h: f64 = counts.values().map(|&k| -(k as f64 / n) * (k as f64 / n).log2()).sum();
    h >= 1.5 && (chars.len() >= 6 || norm.split(' ').count() >= 2)
}

/// A long, specific name (≥ 4 tokens or ≥ 25 characters): an exact match
/// is decisive even when both sides have context.
fn name_is_long(norm: &str) -> bool {
    norm.split(' ').count() >= 4 || norm.chars().count() >= 25
}

// ------------------------------------------------------------------ texts

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n).collect();
        out.push('…');
        out
    }
}

/// The text an entity is embedded with (also used for new rows).
pub fn entity_embed_text(name: &str, entity_type: Option<&str>, description: Option<&str>) -> String {
    let mut s = name.to_string();
    if let Some(t) = entity_type {
        s.push_str(&format!(" ({t})"));
    }
    if let Some(d) = description {
        if !d.is_empty() {
            s.push_str(". ");
            s.push_str(&clip(d, 200));
        }
    }
    s
}

fn mention_text(m: &Mention, remark: Option<&str>) -> String {
    let mut s = m.name.clone();
    if let Some(t) = &m.entity_type {
        s.push_str(&format!(" ({t})"));
    }
    if !m.aliases.is_empty() {
        s.push_str(&format!(", also called {}", m.aliases.join(", ")));
    }
    let desc = m.description.as_deref().or(remark);
    if let Some(d) = desc {
        if !d.is_empty() {
            s.push_str(". ");
            s.push_str(&clip(d, 200));
        }
    }
    if let Some(c) = &m.context {
        if !c.is_empty() {
            s.push_str(&format!(". Context: \"{}\"", clip(c, 240)));
        }
    }
    s
}

// ------------------------------------------------------------------ snapshot

#[derive(Debug, Clone)]
struct Row {
    name: String,
    norm: String,
    tokens: Vec<String>,
    etype: Option<String>,
    aliases: Vec<String>,
    description: Option<String>,
}

/// The entity relation in memory: rows, current merges and review links.
#[derive(Default)]
struct Snapshot {
    rows: BTreeMap<DataValue, Row>,
    merged_into: BTreeMap<DataValue, DataValue>,
    /// blocking key → ids (`x:` exact names and aliases, `l:` last token,
    /// `f:` first token, `a:` acronym).
    keys: BTreeMap<String, Vec<DataValue>>,
    rejected: BTreeSet<(DataValue, DataValue)>,
}

const MAX_POSTING: usize = 200;

impl Snapshot {
    fn canonical(&self, id: &DataValue) -> DataValue {
        let mut cur = id.clone();
        let mut steps = 0;
        while let Some(next) = self.merged_into.get(&cur) {
            cur = next.clone();
            steps += 1;
            if steps > 10_000 {
                break; // a cycle written by hand; stop
            }
        }
        cur
    }
    fn members<'a>(&'a self, canon: &'a DataValue) -> Vec<&'a DataValue> {
        // rows whose canonical is `canon` (canon first)
        let mut out = vec![];
        if self.rows.contains_key(canon) {
            out.push(canon);
        }
        for (id, _) in self.rows.iter() {
            if id != canon && self.merged_into.contains_key(id) && &self.canonical(id) == canon {
                out.push(id);
            }
        }
        out
    }
    fn index_row(&mut self, id: &DataValue) {
        let row = match self.rows.get(id) {
            Some(r) => r.clone(),
            None => return,
        };
        let mut push = |k: String| {
            let v = self.keys.entry(k).or_default();
            if !v.contains(id) {
                v.push(id.clone());
            }
        };
        if !row.norm.is_empty() {
            push(format!("x:{}", row.norm));
        }
        for a in &row.aliases {
            let n = normalize_name(a);
            if !n.is_empty() {
                push(format!("x:{n}"));
            }
        }
        if let Some(f) = row.tokens.first() {
            push(format!("f:{f}"));
            for g in NICKNAMES.iter().filter(|g| g.contains(&f.as_str())) {
                push(format!("f:{}", g[0]));
            }
        }
        if row.tokens.len() >= 2 {
            push(format!("l:{}", row.tokens[row.tokens.len() - 1]));
        }
        if let Some(a) = acronym(&row.tokens) {
            push(format!("a:{a}"));
        }
    }
    fn lookup(&self, key: &str) -> &[DataValue] {
        match self.keys.get(key) {
            Some(v) if v.len() <= MAX_POSTING || key.starts_with("x:") => v,
            _ => &[],
        }
    }
    /// Rule-tier candidates: ids (not canonicalised) that share a blocking key.
    fn rule_block(&self, p: &ParsedName, aliases: &[String]) -> Vec<DataValue> {
        let mut out: Vec<DataValue> = vec![];
        let mut add = |ids: &[DataValue]| {
            for id in ids {
                if !out.contains(id) {
                    out.push(id.clone());
                }
            }
        };
        let norm = p.norm();
        if norm.is_empty() {
            return out;
        }
        add(self.lookup(&format!("x:{norm}")));
        for a in aliases {
            add(self.lookup(&format!("x:{}", normalize_name(a))));
        }
        let t = &p.tokens;
        if t.len() == 1 {
            add(self.lookup(&format!("f:{}", t[0])));
            for g in NICKNAMES.iter().filter(|g| g.contains(&t[0].as_str())) {
                add(self.lookup(&format!("f:{}", g[0])));
            }
            add(self.lookup(&format!("l:{}", t[0])));
            add(self.lookup(&format!("a:{}", t[0])));
        } else {
            add(self.lookup(&format!("l:{}", t[t.len() - 1])));
            if let Some(a) = acronym(t) {
                add(self.lookup(&format!("x:{a}")));
            }
        }
        out
    }
}

fn now_us() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

fn opt_string(v: Option<&DataValue>) -> Option<String> {
    match v {
        Some(DataValue::Str(s)) => Some(s.to_string()),
        _ => None,
    }
}

fn param(pairs: Vec<(&str, DataValue)>) -> BTreeMap<String, DataValue> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// One decided candidate, internal.
struct Scoring {
    cand: Candidate,
    score: f64,
    rule_auto: Option<String>,
}

/// A merge history row (see [`Db::entity_history`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergeEvent {
    /// The merged id.
    pub from: DataValue,
    /// What it was merged into.
    pub into: DataValue,
    /// `merge` or `split`.
    pub action: String,
    /// Valid time of the event, µs since the epoch.
    pub at: i64,
    /// P(same) of the decision behind the merge.
    pub p: f64,
    /// Who decided (`model|template@vN|via`, `rule|…`, `manual`).
    pub decided_by: String,
    /// Why.
    pub reason: String,
}

// ------------------------------------------------------------------ engine

impl<'s, S: Storage<'s>> Db<S> {
    fn entity_relation_exists(&'s self, name: &str) -> Result<bool> {
        let rows = self.run_script("::relations", Default::default(), ScriptMutability::Immutable)?;
        Ok(rows.rows.iter().any(|r| r.first().and_then(|v| v.get_str()) == Some(name)))
    }

    pub(crate) fn ensure_entity_relations(&'s self) -> Result<()> {
        if !self.entity_relation_exists(ENTITY_MERGES)? {
            self.run_script(
                &format!(
                    ":create {ENTITY_MERGES} {{relation: String, from: Any, vt: Validity => into: Any, p: Float, decided_by: String, reason: String}}"
                ),
                Default::default(),
                ScriptMutability::Mutable,
            )?;
        }
        if !self.entity_relation_exists(ENTITY_LINKS)? {
            self.run_script(
                &format!(
                    ":create {ENTITY_LINKS} {{relation: String, a: Any, b: Any => p: Float, band: String, decided_by: String, status: String, at: Int}}"
                ),
                Default::default(),
                ScriptMutability::Mutable,
            )?;
        }
        Ok(())
    }

    fn load_entity_snapshot(&'s self, cfg: &EntityConfig) -> Result<Snapshot> {
        let mut binds = vec![format!("{}: c0", cfg.id), format!("{}: c1", cfg.name)];
        let mut heads = vec!["c0".to_string(), "c1".to_string()];
        let slot = |col: &Option<String>, binds: &mut Vec<String>, heads: &mut Vec<String>| -> Option<usize> {
            col.as_ref().map(|c| {
                let i = heads.len();
                binds.push(format!("{c}: c{i}"));
                heads.push(format!("c{i}"));
                i
            })
        };
        let ti = slot(&cfg.type_col, &mut binds, &mut heads);
        let ai = slot(&cfg.aliases, &mut binds, &mut heads);
        let di = slot(&cfg.description, &mut binds, &mut heads);
        let mut scope_param = BTreeMap::new();
        let mut prelude = String::new();
        let mut cond = String::new();
        match &cfg.scope {
            Some((col, _)) if !cfg.lineage.is_empty() => {
                // a branch: every level's rows, an ancestor's up to its cut
                binds.push(format!("{col}: scv, created: scc"));
                prelude.push_str("lin[scv, cut] <- $lineage\n");
                cond.push_str(", lin[scv, cut], scc <= cut");
                scope_param.insert(
                    "lineage".to_string(),
                    DataValue::List(
                        cfg.lineage
                            .iter()
                            .map(|(v, c)| DataValue::List(vec![v.clone(), DataValue::from(c.unwrap_or(i64::MAX))]))
                            .collect(),
                    ),
                );
            }
            Some((col, v)) => {
                binds.push(format!("{col}: $scope"));
                scope_param.insert("scope".to_string(), v.clone());
            }
            None => {}
        }
        let script = format!(
            "{prelude}?[{}] := *{}{{{}}}{cond}",
            heads.join(", "),
            cfg.relation,
            binds.join(", ")
        );
        let rows = self.run_script(&script, scope_param, ScriptMutability::Immutable)?;
        let mut snap = Snapshot::default();
        for r in rows.rows {
            let name = match &r[1] {
                DataValue::Str(s) => s.to_string(),
                DataValue::Null => String::new(),
                v => bail!("entity name column '{}' must hold strings, got {v:?}", cfg.name),
            };
            let parsed = parse_name(&name);
            let aliases = match ai.map(|i| &r[i]) {
                Some(DataValue::List(l)) => l.iter().filter_map(|x| x.get_str().map(|s| s.to_string())).collect(),
                _ => vec![],
            };
            snap.rows.insert(
                r[0].clone(),
                Row {
                    norm: parsed.norm(),
                    tokens: parsed.tokens,
                    name,
                    etype: opt_string(ti.map(|i| &r[i])),
                    aliases,
                    description: opt_string(di.map(|i| &r[i])),
                },
            );
        }
        if self.entity_relation_exists(ENTITY_MERGES)? {
            let merges = self.run_script(
                &format!("?[f, i] := *{ENTITY_MERGES}{{relation: $r, from: f, into: i @ 'NOW'}}"),
                param(vec![("r", DataValue::from(cfg.relation.as_str()))]),
                ScriptMutability::Immutable,
            )?;
            for r in merges.rows {
                snap.merged_into.insert(r[0].clone(), r[1].clone());
            }
        }
        if self.entity_relation_exists(ENTITY_LINKS)? {
            let links = self.run_script(
                &format!("?[a, b] := *{ENTITY_LINKS}{{relation: $r, a, b, status: 'rejected'}}"),
                param(vec![("r", DataValue::from(cfg.relation.as_str()))]),
                ScriptMutability::Immutable,
            )?;
            for r in links.rows {
                snap.rejected.insert((r[0].clone(), r[1].clone()));
                snap.rejected.insert((r[1].clone(), r[0].clone()));
            }
        }
        let ids: Vec<DataValue> = snap.rows.keys().cloned().collect();
        for id in &ids {
            snap.index_row(id);
        }
        Ok(snap)
    }

    /// Index hits per mention: `(mention, id, source, score)`.
    fn entity_index_hits(
        &'s self,
        cfg: &EntityConfig,
        mentions: &[(usize, &Mention, ParsedName)],
    ) -> Result<Vec<(usize, DataValue, &'static str, f64)>> {
        let mut out = vec![];
        let rel = &cfg.relation;
        let id = &cfg.id;
        let k = cfg.k;
        if let Some(idx) = &cfg.lsh {
            let qs: Vec<DataValue> = mentions
                .iter()
                .filter(|(_, _, p)| !p.tokens.is_empty())
                .map(|(i, _, p)| DataValue::List(vec![DataValue::from(*i as i64), DataValue::from(p.norm().as_str())]))
                .collect();
            if !qs.is_empty() {
                let rows = self.run_script(
                    &format!("qs[m, q] <- $qs\n?[m, e] := qs[m, q], ~{rel}:{idx}{{{id}: e | query: q, k: {k}}}"),
                    param(vec![("qs", DataValue::List(qs))]),
                    ScriptMutability::Immutable,
                )?;
                for r in rows.rows {
                    out.push((r[0].get_int().unwrap_or(0) as usize, r[1].clone(), "lsh", 0.0));
                }
            }
        }
        if let Some(idx) = &cfg.fts {
            let qs: Vec<DataValue> = mentions
                .iter()
                .filter_map(|(i, _, p)| {
                    let toks: Vec<&String> = p
                        .tokens
                        .iter()
                        .filter(|t| t.chars().all(|c| c.is_alphanumeric()))
                        .filter(|t| !matches!(t.as_str(), "and" | "or" | "not" | "near"))
                        .collect();
                    if toks.is_empty() {
                        None
                    } else {
                        let q = toks.iter().map(|t| t.as_str()).collect::<Vec<_>>().join(" OR ");
                        Some(DataValue::List(vec![DataValue::from(*i as i64), DataValue::from(q.as_str())]))
                    }
                })
                .collect();
            if !qs.is_empty() {
                let rows = self.run_script(
                    &format!(
                        "qs[m, q] <- $qs\n?[m, e, s] := qs[m, q], ~{rel}:{idx}{{{id}: e | query: q, k: {k}, bind_score: s}}"
                    ),
                    param(vec![("qs", DataValue::List(qs))]),
                    ScriptMutability::Immutable,
                )?;
                for r in rows.rows {
                    out.push((
                        r[0].get_int().unwrap_or(0) as usize,
                        r[1].clone(),
                        "fts",
                        r[2].get_float().unwrap_or(0.0),
                    ));
                }
            }
        }
        if let (Some(idx), Some(emb)) = (&cfg.hnsw, &cfg.embedder) {
            if crate::embed::is_embedder_enabled(emb) {
                let texts: Vec<String> = mentions
                    .iter()
                    .map(|(_, m, p)| {
                        entity_embed_text(
                            &m.name,
                            m.entity_type.as_deref(),
                            m.description.as_deref().or(p.remark.as_deref()),
                        )
                    })
                    .collect();
                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                // an embedder that fails (model missing, turned off mid-way)
                // degrades to the other sources rather than failing resolution
                if let Ok(vecs) = crate::embed::embed_texts(emb, &refs) {
                    let qs: Vec<DataValue> = mentions
                        .iter()
                        .zip(vecs)
                        .map(|((i, _, _), v)| {
                            DataValue::List(vec![
                                DataValue::from(*i as i64),
                                DataValue::Vec(Vector::F32(Array1::from(v.to_vec()))),
                            ])
                        })
                        .collect();
                    let ef = (k * 4).max(50);
                    let rows = self.run_script(
                        &format!(
                            "qs[m, q] <- $qs\n?[m, e, d] := qs[m, q], ~{rel}:{idx}{{{id}: e | query: q, k: {k}, ef: {ef}, bind_distance: d}}"
                        ),
                        param(vec![("qs", DataValue::List(qs))]),
                        ScriptMutability::Immutable,
                    )?;
                    for r in rows.rows {
                        out.push((
                            r[0].get_int().unwrap_or(0) as usize,
                            r[1].clone(),
                            "vector",
                            1.0 - r[2].get_float().unwrap_or(1.0),
                        ));
                    }
                }
            }
        }
        Ok(out)
    }

    /// The decider to use, if any is configured and available.
    fn entity_decider(cfg: &EntityConfig) -> Option<String> {
        let name = cfg.decider.as_ref()?;
        decide::lookup_decider(name).ok().map(|_| name.clone())
    }

    /// Profile text of a canonical entity for the decider.
    fn profile_text(snap: &Snapshot, canon: &DataValue) -> String {
        let members = snap.members(canon);
        let head = members.first().and_then(|id| snap.rows.get(*id));
        let Some(head) = head else {
            return String::new();
        };
        let mut s = head.name.clone();
        if let Some(t) = &head.etype {
            s.push_str(&format!(" ({t})"));
        }
        let mut names: Vec<String> = vec![];
        for id in members.iter().skip(1) {
            if let Some(r) = snap.rows.get(*id) {
                if r.norm != head.norm && !names.contains(&r.name) {
                    names.push(r.name.clone());
                }
            }
        }
        for a in &head.aliases {
            if !names.contains(a) {
                names.push(a.clone());
            }
        }
        if !names.is_empty() {
            names.truncate(6);
            s.push_str(&format!(", also called {}", names.join(", ")));
        }
        let mut descs: Vec<String> = vec![];
        for id in &members {
            if let Some(r) = snap.rows.get(*id) {
                let d = r.description.clone().or_else(|| parse_name(&r.name).remark);
                if let Some(d) = d {
                    if !d.is_empty() && !descs.contains(&d) {
                        descs.push(d);
                    }
                }
            }
        }
        if !descs.is_empty() {
            descs.truncate(3);
            s.push_str(". ");
            s.push_str(&clip(&descs.join("; "), 240));
        }
        s
    }

    /// Generate and rank candidates for one mention against `snap`.
    fn entity_candidates(
        cfg: &EntityConfig,
        snap: &Snapshot,
        m: &Mention,
        parsed: &ParsedName,
        hits: &[(DataValue, &'static str, f64)],
        has_decider: bool,
    ) -> Vec<Scoring> {
        let own = m.id.as_ref().map(|i| snap.canonical(i));
        let mut by_canon: BTreeMap<DataValue, Scoring> = BTreeMap::new();
        let norm = parsed.norm();
        let mut alias_norms: Vec<String> = m.aliases.iter().map(|a| normalize_name(a)).collect();
        alias_norms.retain(|a| !a.is_empty());
        let mention_has_context = m.context.as_deref().is_some_and(|c| !c.is_empty())
            || m.description.as_deref().is_some_and(|c| !c.is_empty())
            || parsed.remark.is_some();
        let mut consider = |id: &DataValue, source: &'static str, score: f64| {
            let Some(row) = snap.rows.get(id) else { return };
            if !cfg.types_compatible(m.entity_type.as_deref(), row.etype.as_deref()) {
                return;
            }
            let canon = snap.canonical(id);
            if Some(&canon) == own.as_ref() || m.id.as_ref() == Some(id) {
                return;
            }
            if let Some(mid) = &m.id {
                if snap.rejected.contains(&(snap.canonical(mid), canon.clone())) {
                    return;
                }
            }
            let canon_row = snap.rows.get(&canon).unwrap_or(row);
            // name evidence against this row (exact / alias / rules / trigram)
            let mut rule: Option<String> = None;
            let row_aliases: Vec<String> = row.aliases.iter().map(|a| normalize_name(a)).collect();
            if !norm.is_empty() && row.norm == norm {
                rule = Some("exact".into());
            } else if row_aliases.contains(&norm) || alias_norms.contains(&row.norm) {
                rule = Some("alias".into());
            } else if let Some(r) = name_rule(&parsed.tokens, &row.tokens) {
                rule = Some(r.into());
            }
            let jac = trigram_jaccard(&norm, &row.norm);
            let e = by_canon.entry(canon.clone()).or_insert_with(|| Scoring {
                cand: Candidate {
                    entity: canon.clone(),
                    name: canon_row.name.clone(),
                    sources: vec![],
                    rule: None,
                    jaccard: 0.0,
                    p: None,
                    band: None,
                    decided_by: None,
                    evidence: None,
                },
                score: 0.0,
                rule_auto: None,
            });
            if !e.cand.sources.iter().any(|s| s == source) {
                e.cand.sources.push(source.to_string());
            }
            if jac > e.cand.jaccard {
                e.cand.jaccard = jac;
            }
            let rank = |r: &str| match r {
                "exact" | "alias" => 3,
                "initial" | "nickname" | "acronym" | "middle_name" => 2,
                _ => 1,
            };
            if let Some(r) = rule {
                if e.cand.rule.as_deref().map(rank).unwrap_or(0) < rank(&r) {
                    e.cand.rule = Some(r);
                }
            }
            let src_bonus = match source {
                "vector" => 0.3 * score.clamp(0.0, 1.0),
                "fts" => 0.1,
                "lsh" => 0.1,
                _ => 0.0,
            };
            e.score += src_bonus;
        };
        for id in snap.rule_block(parsed, &m.aliases) {
            consider(&id, "rule", 0.0);
        }
        for (id, src, s) in hits {
            consider(id, src, *s);
        }
        let mut out: Vec<Scoring> = by_canon.into_values().collect();
        // keep rule-tier candidates only when a rule actually fired
        out.retain(|s| s.cand.rule.is_some() || s.cand.sources.iter().any(|x| x != "rule"));
        for s in out.iter_mut() {
            let rule_score = match s.cand.rule.as_deref() {
                Some("exact") | Some("alias") => 3.0,
                Some("initial") | Some("nickname") | Some("acronym") | Some("middle_name") => 2.0,
                Some(_) => 1.0,
                None => 0.0,
            };
            s.score += rule_score + 2.0 * s.cand.jaccard;
        }
        out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        // exact (or alias) and unique → the rule may decide alone
        let exact: Vec<usize> = out
            .iter()
            .enumerate()
            .filter(|(_, s)| matches!(s.cand.rule.as_deref(), Some("exact") | Some("alias")))
            .map(|(i, _)| i)
            .collect();
        if exact.len() == 1 {
            let i = exact[0];
            let canon = &out[i].cand.entity;
            let entity_has_context = snap
                .members(canon)
                .iter()
                .any(|id| snap.rows.get(*id).is_some_and(|r| r.description.is_some() || parse_name(&r.name).remark.is_some()));
            // with a model, short exact names ("Maria Garcia", "Mike") are
            // checked by it: same-name different people are common in memory
            let decisive = !has_decider
                || cfg.exact_auto
                || name_is_long(&norm)
                || (!mention_has_context && !entity_has_context);
            if decisive {
                out[i].rule_auto = Some(format!("rule|{}", out[i].cand.rule.clone().unwrap_or_default()));
            }
        }
        out
    }

    /// Resolve `mentions` against the entity relation of `cfg`. See the
    /// module docs and docs/specs/entity-resolution.md. With
    /// `opts.commit`, mentions are written as rows (ids generated when
    /// missing), matches as merges and possible matches as review links,
    /// one mention at a time so later mentions see earlier ones; without it
    /// nothing is written and all decisions go to the decider in one batch.
    pub fn resolve_entities(
        &'s self,
        cfg: &EntityConfig,
        mentions: &[Mention],
        opts: ResolveOptions,
    ) -> Result<Vec<Resolution>> {
        let mut snap = self.load_entity_snapshot(cfg)?;
        let decider = Self::entity_decider(cfg);
        if !opts.commit {
            let all: Vec<usize> = (0..mentions.len()).collect();
            return self.resolve_chunk(cfg, &snap, mentions, &all, decider.as_deref());
        }
        self.ensure_entity_relations()?;
        let mut out = Vec::with_capacity(mentions.len());
        for i in 0..mentions.len() {
            let mut res = self.resolve_chunk(cfg, &snap, mentions, &[i], decider.as_deref())?.remove(0);
            self.commit_resolution(cfg, &mut snap, &mentions[i], &mut res)?;
            out.push(res);
        }
        Ok(out)
    }

    fn resolve_chunk(
        &'s self,
        cfg: &EntityConfig,
        snap: &Snapshot,
        mentions: &[Mention],
        which: &[usize],
        decider: Option<&str>,
    ) -> Result<Vec<Resolution>> {
        let parsed: Vec<(usize, &Mention, ParsedName)> = which
            .iter()
            .map(|&i| (i, &mentions[i], parse_name(&mentions[i].name)))
            .collect();
        let hits = self.entity_index_hits(cfg, &parsed)?;
        let mut hits_by: BTreeMap<usize, Vec<(DataValue, &'static str, f64)>> = BTreeMap::new();
        for (m, id, src, s) in hits {
            hits_by.entry(m).or_default().push((id, src, s));
        }
        let empty = vec![];
        let mut scored: Vec<Vec<Scoring>> = parsed
            .iter()
            .map(|(i, m, p)| {
                Self::entity_candidates(cfg, snap, m, p, hits_by.get(i).unwrap_or(&empty), decider.is_some())
            })
            .collect();
        // one decide batch for every (mention, candidate) the rules leave open
        let mut pairs: Vec<(usize, usize)> = vec![];
        let mut texts: Vec<(String, String)> = vec![];
        if decider.is_some() {
            for (mi, cands) in scored.iter().enumerate() {
                let (_, m, p) = &parsed[mi];
                let mtext = mention_text(m, p.remark.as_deref());
                // an unambiguous rule match settles the mention: no model call
                if cands.iter().any(|c| c.rule_auto.is_some()) {
                    continue;
                }
                for (ci, c) in cands.iter().enumerate().take(cfg.max_candidates) {
                    if c.rule_auto.is_none() {
                        pairs.push((mi, ci));
                        texts.push((mtext.clone(), Self::profile_text(snap, &c.cand.entity)));
                    }
                }
            }
        }
        if let (Some(name), false) = (decider, pairs.is_empty()) {
            let items: Vec<(&str, &str)> = texts.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
            let decisions = decide::decide(name, Template::SameEntity, &items)?;
            let over = (cfg.threshold.is_some() || cfg.tau_auto.is_some()).then(|| Calibration {
                threshold: cfg.threshold.unwrap_or(0.5),
                tau_auto: cfg.tau_auto.unwrap_or(0.9),
                ..Calibration::default()
            });
            for (k, ((mi, ci), d)) in pairs.iter().zip(decisions).enumerate() {
                let c = &mut scored[*mi][*ci].cand;
                let band = match &over {
                    Some(cal) => cal.band(d.p),
                    None => d.band,
                };
                c.p = Some(d.p);
                c.band = Some(band);
                c.evidence = Some(texts[k].1.clone());
                c.decided_by = Some(d.decided_by());
            }
        }
        // no decider: MinHash similarity and the name rules decide
        let mut out = vec![];
        for (mi, mut cands) in scored.into_iter().enumerate() {
            let (idx, m, p) = &parsed[mi];
            for (ci, c) in cands.iter_mut().enumerate() {
                if let Some(by) = &c.rule_auto {
                    c.cand.p = Some(1.0);
                    c.cand.band = Some(Band::Auto);
                    c.cand.decided_by = Some(by.clone());
                } else if decider.is_none() && ci < cfg.max_candidates {
                    let norm = p.norm();
                    let (band, why) = if c.cand.jaccard >= 0.9 && name_is_specific(&norm) {
                        (Band::Auto, "rule|minhash")
                    } else if c.cand.rule.is_some() || c.cand.jaccard >= 0.5 {
                        (Band::Tentative, "rule|name")
                    } else {
                        (Band::None, "rule|name")
                    };
                    c.cand.p = Some(match band {
                        Band::Auto => c.cand.jaccard.max(0.9),
                        Band::Tentative => c.cand.jaccard.max(0.5),
                        Band::None => c.cand.jaccard,
                    });
                    c.cand.band = Some(band);
                    let r = c.cand.rule.clone().unwrap_or_else(|| "jaccard".into());
                    c.cand.decided_by = Some(if why == "rule|minhash" { why.to_string() } else { format!("rule|{r}") });
                }
            }
            let best_auto = cands
                .iter()
                .filter(|c| c.cand.band == Some(Band::Auto))
                .max_by(|a, b| a.cand.p.partial_cmp(&b.cand.p).unwrap_or(std::cmp::Ordering::Equal));
            let best_tent = cands
                .iter()
                .filter(|c| c.cand.band == Some(Band::Tentative))
                .max_by(|a, b| a.cand.p.partial_cmp(&b.cand.p).unwrap_or(std::cmp::Ordering::Equal));
            let n_auto = cands.iter().filter(|c| c.cand.band == Some(Band::Auto)).count();
            let (outcome, chosen, reason) = match (best_auto, best_tent) {
                (Some(a), _) => (
                    Outcome::Matched,
                    Some(a),
                    Some(if n_auto > 1 {
                        format!("{} candidates in the auto band; best taken, others linked for review", n_auto)
                    } else {
                        a.cand.rule.clone().map(|r| format!("name rule: {r}")).unwrap_or_else(|| "decided".into())
                    }),
                ),
                (None, Some(t)) => (Outcome::Possible, Some(t), Some("tentative: linked for review".to_string())),
                (None, None) => (
                    Outcome::New,
                    None,
                    Some(if cands.is_empty() { "no candidates" } else { "no candidate above threshold" }.to_string()),
                ),
            };
            let res = Resolution {
                mention: *idx,
                name: m.name.clone(),
                outcome,
                entity: chosen.map(|c| c.cand.entity.clone()),
                id: m.id.clone(),
                p: chosen.and_then(|c| c.cand.p).unwrap_or(0.0),
                band: chosen.and_then(|c| c.cand.band).unwrap_or(Band::None),
                decided_by: chosen
                    .and_then(|c| c.cand.decided_by.clone())
                    .unwrap_or_else(|| match decider {
                        Some(_) => "decider|same_entity|none".to_string(),
                        None => "rule|none".to_string(),
                    }),
                reason,
                candidates: cands.into_iter().map(|s| s.cand).collect(),
                text: mention_text(m, p.remark.as_deref()),
            };
            out.push(res);
        }
        Ok(out)
    }

    fn commit_resolution(
        &'s self,
        cfg: &EntityConfig,
        snap: &mut Snapshot,
        m: &Mention,
        res: &mut Resolution,
    ) -> Result<()> {
        let id = match &m.id {
            Some(id) => id.clone(),
            None => DataValue::from(format!("ent-{}", uuid::Uuid::new_v4()).as_str()),
        };
        if !snap.rows.contains_key(&id) {
            let parsed = parse_name(&m.name);
            let mut cols = vec![cfg.id.clone(), cfg.name.clone()];
            let mut vals = vec![id.clone(), DataValue::from(m.name.as_str())];
            if let Some((col, v)) = &cfg.scope {
                cols.insert(0, col.clone());
                vals.insert(0, v.clone());
            }
            if let Some(c) = &cfg.type_col {
                cols.push(c.clone());
                vals.push(m.entity_type.as_deref().map(DataValue::from).unwrap_or(DataValue::Null));
            }
            if let Some(c) = &cfg.aliases {
                cols.push(c.clone());
                vals.push(DataValue::List(m.aliases.iter().map(|a| DataValue::from(a.as_str())).collect()));
            }
            // the evidence the mention came with describes the new entity
            let desc = m
                .description
                .clone()
                .or(parsed.remark.clone())
                .or_else(|| m.context.as_deref().filter(|c| !c.is_empty()).map(|c| clip(c, 200)));
            if let Some(c) = &cfg.description {
                cols.push(c.clone());
                vals.push(desc.as_deref().map(DataValue::from).unwrap_or(DataValue::Null));
            }
            if let (Some(c), Some(e)) = (&cfg.embedding, &cfg.embedder) {
                if crate::embed::is_embedder_enabled(e) {
                    let t = entity_embed_text(&m.name, m.entity_type.as_deref(), desc.as_deref());
                    if let Ok(mut v) = crate::embed::embed_texts(e, &[t.as_str()]) {
                        cols.push(c.clone());
                        vals.push(DataValue::Vec(Vector::F32(Array1::from(v.remove(0).to_vec()))));
                    }
                }
            }
            let heads: Vec<String> = (0..cols.len()).map(|i| format!("c{i}")).collect();
            let spec: Vec<String> = cols.iter().zip(&heads).map(|(c, h)| format!("{c} = {h}")).collect();
            let (k, v) = spec.split_at(if cfg.scope.is_some() { 2 } else { 1 });
            self.run_script(
                &format!(
                    "?[{}] <- $rows :put {} {{{} => {}}}",
                    heads.join(", "),
                    cfg.relation,
                    k.join(", "),
                    v.join(", ")
                ),
                param(vec![("rows", DataValue::List(vec![DataValue::List(vals)]))]),
                ScriptMutability::Mutable,
            )?;
            snap.rows.insert(
                id.clone(),
                Row {
                    name: m.name.clone(),
                    norm: parsed.norm(),
                    tokens: parsed.tokens,
                    etype: m.entity_type.clone(),
                    aliases: m.aliases.clone(),
                    description: desc,
                },
            );
            snap.index_row(&id);
        }
        res.id = Some(id.clone());
        match res.outcome {
            Outcome::Matched => {
                let into = res.entity.clone().unwrap();
                self.write_merge(cfg, &id, &into, res.p, &res.decided_by, res.reason.as_deref().unwrap_or(""))?;
                snap.merged_into.insert(id.clone(), into);
                // other auto/tentative candidates: links for review, never merges
                let others: Vec<&Candidate> = res
                    .candidates
                    .iter()
                    .filter(|c| Some(&c.entity) != res.entity.as_ref())
                    .filter(|c| matches!(c.band, Some(Band::Auto) | Some(Band::Tentative)))
                    .collect();
                self.write_links(cfg, &id, &others)?;
            }
            Outcome::Possible => {
                let others: Vec<&Candidate> = res
                    .candidates
                    .iter()
                    .filter(|c| matches!(c.band, Some(Band::Auto) | Some(Band::Tentative)))
                    .collect();
                self.write_links(cfg, &id, &others)?;
            }
            Outcome::New => {}
        }
        Ok(())
    }

    fn write_links(&'s self, cfg: &EntityConfig, id: &DataValue, cands: &[&Candidate]) -> Result<()> {
        if cands.is_empty() {
            return Ok(());
        }
        let at = now_us();
        let rows: Vec<DataValue> = cands
            .iter()
            .map(|c| {
                DataValue::List(vec![
                    DataValue::from(cfg.relation.as_str()),
                    id.clone(),
                    c.entity.clone(),
                    DataValue::from(c.p.unwrap_or(0.0)),
                    DataValue::from(c.band.unwrap_or(Band::None).name()),
                    DataValue::from(c.decided_by.clone().unwrap_or_default().as_str()),
                    DataValue::from("pending"),
                    DataValue::from(at),
                ])
            })
            .collect();
        self.run_script(
            &format!(
                "?[relation, a, b, p, band, decided_by, status, at] <- $rows :put {ENTITY_LINKS} {{relation, a, b => p, band, decided_by, status, at}}"
            ),
            param(vec![("rows", DataValue::List(rows))]),
            ScriptMutability::Mutable,
        )?;
        Ok(())
    }

    /// A valid time strictly after every recorded event of `from`.
    fn next_merge_ts(&'s self, relation: &str, from: &DataValue) -> Result<i64> {
        let rows = self.run_script(
            &format!("?[t] := *{ENTITY_MERGES}{{relation: $r, from: $f, vt}}, t = to_int(vt)"),
            param(vec![("r", DataValue::from(relation)), ("f", from.clone())]),
            ScriptMutability::Immutable,
        )?;
        let last = rows.rows.iter().filter_map(|r| r[0].get_int()).max().unwrap_or(i64::MIN);
        Ok(now_us().max(last.saturating_add(1)))
    }

    fn write_merge(
        &'s self,
        cfg: &EntityConfig,
        from: &DataValue,
        into: &DataValue,
        p: f64,
        decided_by: &str,
        reason: &str,
    ) -> Result<()> {
        let ts = self.next_merge_ts(&cfg.relation, from)?;
        self.run_script(
            &format!(
                "?[relation, from, vt, into, p, decided_by, reason] <- $rows :put {ENTITY_MERGES} {{relation, from, vt => into, p, decided_by, reason}}"
            ),
            param(vec![(
                "rows",
                DataValue::List(vec![DataValue::List(vec![
                    DataValue::from(cfg.relation.as_str()),
                    from.clone(),
                    DataValue::Validity(Validity::from((ts, true))),
                    into.clone(),
                    DataValue::from(p),
                    DataValue::from(decided_by),
                    DataValue::from(reason),
                ])]),
            )]),
            ScriptMutability::Mutable,
        )?;
        Ok(())
    }

    /// The canonical entity of `id` (itself when not merged).
    pub fn canonical_entity(&'s self, relation: &str, id: &DataValue) -> Result<DataValue> {
        if !self.entity_relation_exists(ENTITY_MERGES)? {
            return Ok(id.clone());
        }
        let merges = self.run_script(
            &format!("?[f, i] := *{ENTITY_MERGES}{{relation: $r, from: f, into: i @ 'NOW'}}"),
            param(vec![("r", DataValue::from(relation))]),
            ScriptMutability::Immutable,
        )?;
        let snap = Snapshot {
            merged_into: merges.rows.into_iter().map(|r| (r[0].clone(), r[1].clone())).collect(),
            ..Default::default()
        };
        Ok(snap.canonical(id))
    }

    /// Merge `from` into `into` by hand (a review accepting a possible
    /// same-as link, or a correction). Refuses to create a cycle. A pending
    /// link between the two is marked accepted.
    pub fn merge_entities(&'s self, relation: &str, from: &DataValue, into: &DataValue, reason: &str) -> Result<()> {
        if from == into {
            bail!("cannot merge entity {from:?} into itself");
        }
        self.ensure_entity_relations()?;
        if &self.canonical_entity(relation, into)? == from {
            bail!("merging {from:?} into {into:?} would make a cycle: {into:?} is already merged into {from:?}");
        }
        let cfg = EntityConfig::new(relation);
        self.write_merge(&cfg, from, into, 1.0, "manual", reason)?;
        self.set_link_status(relation, from, into, "accepted")?;
        Ok(())
    }

    fn set_link_status(&'s self, relation: &str, a: &DataValue, b: &DataValue, status: &str) -> Result<usize> {
        let found = self.run_script(
            &format!(
                "?[relation, a, b, p, band, decided_by, status, at] := *{ENTITY_LINKS}{{relation, a, b, p, band, decided_by, at}}, relation = $r, a = $x, b = $y, status = $s
                 ?[relation, a, b, p, band, decided_by, status, at] := *{ENTITY_LINKS}{{relation, a, b, p, band, decided_by, at}}, relation = $r, a = $y, b = $x, status = $s"
            ),
            param(vec![
                ("r", DataValue::from(relation)),
                ("x", a.clone()),
                ("y", b.clone()),
                ("s", DataValue::from(status)),
            ]),
            ScriptMutability::Immutable,
        )?;
        let n = found.rows.len();
        if n > 0 {
            self.run_script(
                &format!(
                    "?[relation, a, b, p, band, decided_by, status, at] <- $rows :put {ENTITY_LINKS} {{relation, a, b => p, band, decided_by, status, at}}"
                ),
                param(vec![("rows", DataValue::List(found.rows.into_iter().map(DataValue::List).collect()))]),
                ScriptMutability::Mutable,
            )?;
        }
        Ok(n)
    }

    /// Reject a possible same-as link: later resolutions will not propose
    /// that pair again. Returns whether a link existed.
    pub fn reject_entity_link(&'s self, relation: &str, a: &DataValue, b: &DataValue) -> Result<bool> {
        self.ensure_entity_relations()?;
        let n = self.set_link_status(relation, a, b, "rejected")?;
        if n == 0 {
            // record the rejection anyway, so the pair stays apart
            self.run_script(
                &format!(
                    "?[relation, a, b, p, band, decided_by, status, at] <- [[$r, $x, $y, 0.0, 'none', 'manual', 'rejected', $t]] :put {ENTITY_LINKS} {{relation, a, b => p, band, decided_by, status, at}}"
                ),
                param(vec![
                    ("r", DataValue::from(relation)),
                    ("x", a.clone()),
                    ("y", b.clone()),
                    ("t", DataValue::from(now_us())),
                ]),
                ScriptMutability::Mutable,
            )?;
        }
        Ok(n > 0)
    }

    /// Undo the merge of `id`: it (with whatever was merged into it) becomes
    /// its own entity again. The merge stays in the history. Returns what it
    /// had been merged into, or `None` if it was not merged.
    pub fn split_entity(&'s self, relation: &str, id: &DataValue) -> Result<Option<DataValue>> {
        if !self.entity_relation_exists(ENTITY_MERGES)? {
            return Ok(None);
        }
        let cur = self.run_script(
            &format!(
                "?[into, p, decided_by, reason] := *{ENTITY_MERGES}{{relation: $r, from: $f, into, p, decided_by, reason @ 'NOW'}}"
            ),
            param(vec![("r", DataValue::from(relation)), ("f", id.clone())]),
            ScriptMutability::Immutable,
        )?;
        let Some(row) = cur.rows.into_iter().next() else {
            return Ok(None);
        };
        let ts = self.next_merge_ts(relation, id)?;
        self.run_script(
            &format!(
                "?[relation, from, vt, into, p, decided_by, reason] <- $rows :put {ENTITY_MERGES} {{relation, from, vt => into, p, decided_by, reason}}"
            ),
            param(vec![(
                "rows",
                DataValue::List(vec![DataValue::List(vec![
                    DataValue::from(relation),
                    id.clone(),
                    DataValue::Validity(Validity::from((ts, false))),
                    row[0].clone(),
                    row[1].clone(),
                    row[2].clone(),
                    row[3].clone(),
                ])]),
            )]),
            ScriptMutability::Mutable,
        )?;
        Ok(Some(row[0].clone()))
    }

    /// Every merge and split of `relation` (of `id` when given: as `from`
    /// or `into`), oldest first.
    pub fn entity_history(&'s self, relation: &str, id: Option<&DataValue>) -> Result<Vec<MergeEvent>> {
        if !self.entity_relation_exists(ENTITY_MERGES)? {
            return Ok(vec![]);
        }
        let rows = self.run_script(
            &format!(
                "?[from, into, t, assert, p, decided_by, reason] := *{ENTITY_MERGES}{{relation: $r, from, vt, into, p, decided_by, reason}}, t = to_int(vt), assert = to_bool(vt)"
            ),
            param(vec![("r", DataValue::from(relation))]),
            ScriptMutability::Immutable,
        )?;
        let mut out: Vec<MergeEvent> = rows
            .rows
            .into_iter()
            .filter(|r| id.is_none_or(|id| &r[0] == id || &r[1] == id))
            .map(|r| MergeEvent {
                from: r[0].clone(),
                into: r[1].clone(),
                at: r[2].get_int().unwrap_or(0),
                action: if r[3].get_bool().unwrap_or(true) { "merge" } else { "split" }.to_string(),
                p: r[4].get_float().unwrap_or(0.0),
                decided_by: r[5].get_str().unwrap_or("").to_string(),
                reason: r[6].get_str().unwrap_or("").to_string(),
            })
            .collect();
        out.sort_by(|a, b| (a.at, &a.from).cmp(&(b.at, &b.from)));
        Ok(out)
    }

    /// Possible same-as links awaiting review (status `pending`) whose two
    /// ends are still different entities: `a, b, p, band, decided_by`.
    pub fn entity_pending(&'s self, relation: &str) -> Result<NamedRows> {
        let headers = ["a", "b", "canonical_a", "canonical_b", "p", "band", "decided_by", "at"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        if !self.entity_relation_exists(ENTITY_LINKS)? {
            return Ok(NamedRows::new(headers, vec![]));
        }
        let rows = self.run_script(
            &format!(
                "?[a, b, p, band, decided_by, at] := *{ENTITY_LINKS}{{relation: $r, a, b, p, band, decided_by, status: 'pending', at}}"
            ),
            param(vec![("r", DataValue::from(relation))]),
            ScriptMutability::Immutable,
        )?;
        let merges = if self.entity_relation_exists(ENTITY_MERGES)? {
            self.run_script(
                &format!("?[f, i] := *{ENTITY_MERGES}{{relation: $r, from: f, into: i @ 'NOW'}}"),
                param(vec![("r", DataValue::from(relation))]),
                ScriptMutability::Immutable,
            )?
            .rows
        } else {
            vec![]
        };
        let snap = Snapshot {
            merged_into: merges.into_iter().map(|r| (r[0].clone(), r[1].clone())).collect(),
            ..Default::default()
        };
        let mut out = vec![];
        for r in rows.rows {
            let (ca, cb) = (snap.canonical(&r[0]), snap.canonical(&r[1]));
            if ca == cb {
                continue;
            }
            out.push(vec![r[0].clone(), r[1].clone(), ca, cb, r[2].clone(), r[3].clone(), r[4].clone(), r[5].clone()]);
        }
        out.sort_by(|a, b| b[4].partial_cmp(&a[4]).unwrap_or(std::cmp::Ordering::Equal));
        Ok(NamedRows::new(headers, out))
    }
}

impl<'s, S: Storage<'s>> Db<S> {
    pub(crate) fn run_entity_op(&'s self, op: &EntityOp, read_only: bool) -> Result<NamedRows> {
        let writes = match op {
            EntityOp::Resolve { commit, .. } => *commit,
            EntityOp::Merge(..) | EntityOp::Split(..) | EntityOp::Reject(..) => true,
            EntityOp::History(..) | EntityOp::Pending(..) => false,
        };
        if read_only && writes {
            bail!("`::entity resolve {{commit: true}}`, `merge`, `split` and `reject` cannot run in read-only mode");
        }
        let status = |s: &str| NamedRows::new(vec!["status".to_string()], vec![vec![DataValue::from(s)]]);
        match op {
            EntityOp::Resolve {
                config,
                mentions,
                commit,
            } => Ok(resolutions_to_rows(&self.resolve_entities(
                config,
                mentions,
                ResolveOptions { commit: *commit },
            )?)),
            EntityOp::Merge(rel, from, into) => {
                self.merge_entities(rel, from, into, "manual merge")?;
                Ok(status("OK"))
            }
            EntityOp::Split(rel, id) => Ok(match self.split_entity(rel, id)? {
                Some(into) => NamedRows::new(
                    vec!["status".to_string(), "was_merged_into".to_string()],
                    vec![vec![DataValue::from("OK"), into]],
                ),
                None => status("NOT_MERGED"),
            }),
            EntityOp::History(rel, id) => Ok(history_to_rows(&self.entity_history(rel, id.as_ref())?)),
            EntityOp::Pending(rel) => self.entity_pending(rel),
            EntityOp::Reject(rel, a, b) => {
                Ok(status(if self.reject_entity_link(rel, a, b)? { "OK" } else { "RECORDED" }))
            }
        }
    }
}

/// Rows of a resolve call, as `::entity resolve` returns them.
pub fn resolutions_to_rows(res: &[Resolution]) -> NamedRows {
    NamedRows::new(
        [
            "mention", "name", "outcome", "entity", "id", "p", "band", "decided_by", "reason", "candidates",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        res.iter()
            .map(|r| {
                vec![
                    DataValue::from(r.mention as i64),
                    DataValue::from(r.name.as_str()),
                    DataValue::from(r.outcome.name()),
                    r.entity.clone().unwrap_or(DataValue::Null),
                    r.id.clone().unwrap_or(DataValue::Null),
                    DataValue::from(r.p),
                    DataValue::from(r.band.name()),
                    DataValue::from(r.decided_by.as_str()),
                    r.reason.as_deref().map(DataValue::from).unwrap_or(DataValue::Null),
                    DataValue::List(r.candidates.iter().map(|c| c.entity.clone()).collect()),
                ]
            })
            .collect(),
    )
}

/// Rows of a history call, as `::entity history` returns them.
pub fn history_to_rows(ev: &[MergeEvent]) -> NamedRows {
    NamedRows::new(
        ["from", "into", "action", "at", "p", "decided_by", "reason"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        ev.iter()
            .map(|e| {
                vec![
                    e.from.clone(),
                    e.into.clone(),
                    DataValue::from(e.action.as_str()),
                    DataValue::from(e.at),
                    DataValue::from(e.p),
                    DataValue::from(e.decided_by.as_str()),
                    DataValue::from(e.reason.as_str()),
                ]
            })
            .collect(),
    )
}

/// Timing helper for evaluation (`resolve_entities` wall time).
pub fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<String> {
        parse_name(s).tokens
    }

    #[test]
    fn normalizes_names() {
        assert_eq!(normalize_name("Dr. Alice  Smith"), "alice smith");
        assert_eq!(normalize_name("Alice (my manager)"), "alice");
        assert_eq!(parse_name("Alice (my manager)").remark.as_deref(), Some("my manager"));
        assert_eq!(normalize_name("José Álvarez Jr."), "jose alvarez");
        assert_eq!(normalize_name("Acme, Inc."), "acme");
        assert_eq!(normalize_name("The Walt Disney Company"), "walt disney");
        assert_eq!(normalize_name("O'Brien"), "obrien");
        assert_eq!(normalize_name("AT&T"), "at and t");
    }

    #[test]
    fn name_rules() {
        assert_eq!(name_rule(&toks("A. Smith"), &toks("Alice Smith")), Some("initial"));
        assert_eq!(name_rule(&toks("Bob Jones"), &toks("Robert Jones")), Some("nickname"));
        assert_eq!(name_rule(&toks("Alice"), &toks("Alice Smith")), Some("first_name"));
        assert_eq!(name_rule(&toks("Smith"), &toks("Alice Smith")), Some("surname"));
        assert_eq!(name_rule(&toks("IBM"), &toks("International Business Machines")), Some("acronym"));
        assert_eq!(name_rule(&toks("MIT"), &toks("Massachusetts Institute of Technology")), Some("acronym"));
        assert_eq!(name_rule(&toks("Alice B. Smith"), &toks("Alice Smith")), Some("middle_name"));
        assert_eq!(name_rule(&toks("Alice Smith"), &toks("Bob Smith")), None);
        assert_eq!(name_rule(&toks("Alice Smith"), &toks("Alice Jones")), None);
    }

    #[test]
    fn jaccard_and_gate() {
        assert!(trigram_jaccard("alice smith", "alice smith") > 0.99);
        assert!(trigram_jaccard("alice smith", "alise smith") > 0.5);
        assert!(trigram_jaccard("alice smith", "bob jones") < 0.1);
        assert!(!name_is_specific("sam"));
        assert!(!name_is_specific("ai"));
        assert!(name_is_specific("alice smith"));
    }
}
