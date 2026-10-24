/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Ontologies (docs/specs/ontology.md): classes and properties with
//! cardinality and characteristics, stored as relations with valid time and
//! activated per namespace. The active ontology types what `remember`
//! stores (`ontology_type.rs`), decides which statements may supersede each
//! other, compiles to a Datalog program for inference, and validates facts.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use itertools::Itertools;
use lazy_static::lazy_static;
use miette::{bail, miette, IntoDiagnostic, Result};
use serde_derive::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};

use crate::data::value::{DataValue, Validity};
use crate::runtime::db::{Db, NamedRows, ScriptMutability};
use crate::storage::Storage;
use crate::JsonData;

pub const ONTO: &str = "kleos_onto";
pub const ONTO_CLASS: &str = "kleos_onto_class";
pub const ONTO_PROP: &str = "kleos_onto_prop";
pub const ONTO_NS: &str = "kleos_onto_ns";
pub const ONTO_FACT: &str = "kleos_onto_fact";
pub const ONTO_NODE: &str = "kleos_onto_node";
pub const MEM_TYPE: &str = "kleos_mem_type";

const DDL: &[(&str, &str)] = &[
    (ONTO, "{onto: String, vt: Validity => version: String, doc: String, source: String}"),
    (
        ONTO_CLASS,
        "{onto: String, name: String, vt: Validity => parents: [String], label: String, doc: String, aliases: [String]}",
    ),
    (
        ONTO_PROP,
        "{onto: String, name: String, vt: Validity => kind: String, domain: [String], range: [String], \
         datatype: String?, card: String, symmetric: Bool, transitive: Bool, inverse: String?, parents: [String], \
         chain: [String], temporal: Bool, unit: String?, label: String, doc: String, aliases: [String], cues: [String]}",
    ),
    (ONTO_NS, "{ns: String, vt: Validity => ontos: [String], policy: String, untyped: String}"),
    (
        ONTO_FACT,
        "{ns: String, s: String, p: String, o: String, vt: Validity => lit: Json?, conf: Float, by: String, \
         mem: String?, status: String}",
    ),
    (ONTO_NODE, "{ns: String, node: String, vt: Validity => class: String, label: String?, mem: String?}"),
    (
        MEM_TYPE,
        "{ns: String, id: String => onto: String, cls: String, subject: String?, prop: String?, score: Float, by: String}",
    ),
];

/// The default personal-memory pack.
pub const PERSONAL: &str = include_str!("ontology_personal.json");

/// Supersession policies.
pub const POLICIES: &[&str] = &["off", "cardinality"];
/// What happens to a pair where a claim couldn't be typed.
pub const UNTYPED: &[&str] = &["decide", "pending", "dup_only"];
/// The `untyped` default (chosen from the measurements in the spec).
pub const DEFAULT_UNTYPED: &str = "decide";

// ------------------------------------------------------------------ model

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OntoClass {
    pub name: String,
    pub parents: Vec<String>,
    pub label: String,
    pub doc: String,
    pub aliases: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OntoProp {
    pub name: String,
    /// `object` (node → node), `data` (node → literal), `role` (event → value).
    pub kind: String,
    pub domain: Vec<String>,
    pub range: Vec<String>,
    pub datatype: Option<String>,
    /// `one` or `many`.
    pub card: String,
    pub symmetric: bool,
    pub transitive: bool,
    pub inverse: Option<String>,
    /// Super-properties.
    pub parents: Vec<String>,
    /// `[p, q]`: p then q implies this property.
    pub chain: Vec<String>,
    /// The value changes over time.
    pub temporal: bool,
    pub unit: Option<String>,
    pub label: String,
    pub doc: String,
    pub aliases: Vec<String>,
    /// Cue phrases for claim typing (lower case, matched at word boundaries).
    pub cues: Vec<String>,
}

impl Default for OntoProp {
    fn default() -> Self {
        OntoProp {
            name: String::new(),
            kind: "data".to_string(),
            domain: vec![],
            range: vec![],
            datatype: None,
            // OWL's default: a property is multi-valued unless declared functional
            card: "many".to_string(),
            symmetric: false,
            transitive: false,
            inverse: None,
            parents: vec![],
            chain: vec![],
            temporal: true,
            unit: None,
            label: String::new(),
            doc: String::new(),
            aliases: vec![],
            cues: vec![],
        }
    }
}

impl OntoProp {
    pub fn single(&self) -> bool {
        self.card == "one"
    }
}

/// An ontology document: the JSON import/export format.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OntologyDoc {
    pub onto: String,
    pub version: String,
    pub doc: String,
    pub source: String,
    pub classes: Vec<OntoClass>,
    pub properties: Vec<OntoProp>,
}

impl OntologyDoc {
    fn check(&self) -> Result<()> {
        if self.onto.trim().is_empty() {
            bail!("ontology: 'onto' (the ontology's name) is required");
        }
        let mut seen = BTreeSet::new();
        for c in &self.classes {
            if c.name.trim().is_empty() {
                bail!("ontology '{}': a class without a name", self.onto);
            }
            if !seen.insert(("c", c.name.clone())) {
                bail!("ontology '{}': class '{}' defined twice", self.onto, c.name);
            }
        }
        for p in &self.properties {
            if p.name.trim().is_empty() {
                bail!("ontology '{}': a property without a name", self.onto);
            }
            if !seen.insert(("p", p.name.clone())) {
                bail!("ontology '{}': property '{}' defined twice", self.onto, p.name);
            }
            if !["object", "data", "role"].contains(&p.kind.as_str()) {
                bail!("ontology '{}': property '{}' kind must be object, data or role, got '{}'", self.onto, p.name, p.kind);
            }
            if !["one", "many"].contains(&p.card.as_str()) {
                bail!("ontology '{}': property '{}' card must be one or many, got '{}'", self.onto, p.name, p.card);
            }
            if !p.chain.is_empty() && p.chain.len() < 2 {
                bail!("ontology '{}': property '{}' chain needs at least two properties", self.onto, p.name);
            }
        }
        Ok(())
    }
}

/// The ontologies active on a namespace, merged (later ones override
/// earlier ones by name), with the namespace's policy.
#[derive(Debug, Clone)]
pub struct Active {
    pub ontos: Vec<String>,
    pub policy: String,
    pub untyped: String,
    pub classes: BTreeMap<String, OntoClass>,
    pub props: Vec<OntoProp>,
    pub by_name: HashMap<String, usize>,
    /// alias (lower case) → property index
    pub aliases: HashMap<String, usize>,
    /// Content hash: keys the prototype cache.
    pub hash: u64,
}

impl Active {
    pub fn merge(docs: &[OntologyDoc], policy: &str, untyped: &str) -> Active {
        let mut classes: BTreeMap<String, OntoClass> = BTreeMap::new();
        let mut props: Vec<OntoProp> = vec![];
        let mut by_name: HashMap<String, usize> = HashMap::new();
        for d in docs {
            for c in &d.classes {
                classes.insert(c.name.clone(), c.clone());
            }
            for p in &d.properties {
                match by_name.get(&p.name) {
                    Some(&i) => props[i] = p.clone(),
                    None => {
                        by_name.insert(p.name.clone(), props.len());
                        props.push(p.clone());
                    }
                }
            }
        }
        let mut aliases = HashMap::new();
        for (i, p) in props.iter().enumerate() {
            for a in p.aliases.iter().chain(std::iter::once(&p.label)) {
                let k = a.trim().to_lowercase();
                if !k.is_empty() {
                    aliases.entry(k).or_insert(i);
                }
            }
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for p in &props {
            (&p.name, &p.label, &p.doc, &p.aliases, &p.chain).hash(&mut h);
        }
        Active {
            ontos: docs.iter().map(|d| d.onto.clone()).collect(),
            policy: policy.to_string(),
            untyped: untyped.to_string(),
            classes,
            props,
            by_name,
            aliases,
            hash: h.finish(),
        }
    }

    /// A property by name or alias.
    pub fn prop(&self, name: &str) -> Option<&OntoProp> {
        self.by_name
            .get(name)
            .or_else(|| self.aliases.get(&name.trim().to_lowercase()))
            .map(|&i| &self.props[i])
    }

    /// `class` and every ancestor of it.
    pub fn ancestors(&self, class: &str) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let mut todo = vec![class.to_string()];
        while let Some(c) = todo.pop() {
            if out.insert(c.clone()) {
                if let Some(k) = self.classes.get(&c) {
                    todo.extend(k.parents.iter().cloned());
                }
            }
        }
        out
    }
}

// ------------------------------------------------------------------ helpers

fn now_micros() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

fn s(v: &str) -> DataValue {
    DataValue::from(v)
}

fn opt_s(v: &Option<String>) -> DataValue {
    v.as_deref().map(DataValue::from).unwrap_or(DataValue::Null)
}

fn strs(v: &[String]) -> DataValue {
    DataValue::List(v.iter().map(|x| s(x)).collect())
}

fn vt(at: i64, assert: bool) -> DataValue {
    DataValue::Validity(Validity::from((at, assert)))
}

fn cell(v: &DataValue) -> String {
    v.get_str().unwrap_or("").to_string()
}

fn cell_opt(v: &DataValue) -> Option<String> {
    v.get_str().map(|x| x.to_string())
}

fn cell_strs(v: &DataValue) -> Vec<String> {
    match v {
        DataValue::List(l) => l.iter().filter_map(|x| x.get_str().map(|y| y.to_string())).collect(),
        _ => vec![],
    }
}

fn cell_bool(v: &DataValue) -> bool {
    v.get_bool().unwrap_or(false)
}

fn jv(v: &DataValue) -> JsonValue {
    JsonValue::from(v.clone())
}

fn params(p: Vec<(&str, DataValue)>) -> BTreeMap<String, DataValue> {
    p.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// The `@` clause for a read at `at` (µs) or now.
fn at_clause(at: Option<i64>) -> String {
    match at {
        Some(t) => format!("{t}"),
        None => "'NOW'".to_string(),
    }
}

/// The canonical string of a literal (`o` of a data fact): its value, then
/// its unit if any.
pub fn literal_key(lit: &JsonValue) -> String {
    let v = match lit.get("value") {
        Some(JsonValue::String(x)) => x.clone(),
        Some(JsonValue::Null) | None => String::new(),
        Some(x) => x.to_string(),
    };
    match lit.get("unit").and_then(|u| u.as_str()) {
        Some(u) if !u.is_empty() => format!("{v} {u}"),
        _ => v,
    }
}

lazy_static! {
    /// Prototype vectors of the typable properties, per (ontology hash,
    /// embedder): computed once.
    static ref PROTOS: Mutex<HashMap<(u64, String), Arc<Vec<(usize, Vec<f32>)>>>> = Mutex::new(HashMap::new());
}

/// The text a property's prototype vector is embedded from.
pub fn proto_text(p: &OntoProp) -> String {
    let mut t = format!("{}: {}", if p.label.is_empty() { &p.name } else { &p.label }, p.doc);
    if !p.aliases.is_empty() {
        t.push_str(&format!(" ({})", p.aliases.join(", ")));
    }
    t
}

/// Properties that claims can be typed as: everything but derived ones
/// (chains) and inverse-only twins without cues.
pub fn typable(p: &OntoProp) -> bool {
    p.chain.is_empty() && p.kind != "role" && p.name != "same_as"
}

/// The prototype vectors of `act` under `embedder` (cached).
pub fn prototypes(act: &Active, embedder: &str) -> Result<Arc<Vec<(usize, Vec<f32>)>>> {
    let key = (act.hash, embedder.to_string());
    if let Some(v) = PROTOS.lock().unwrap().get(&key) {
        return Ok(v.clone());
    }
    let idx: Vec<usize> = (0..act.props.len()).filter(|&i| typable(&act.props[i])).collect();
    let texts: Vec<String> = idx.iter().map(|&i| proto_text(&act.props[i])).collect();
    let refs: Vec<&str> = texts.iter().map(|t| t.as_str()).collect();
    let vecs = crate::embed::embed_texts(embedder, &refs)?;
    let out = Arc::new(idx.into_iter().zip(vecs).map(|(i, v)| (i, v.to_vec())).collect::<Vec<_>>());
    PROTOS.lock().unwrap().insert(key, out.clone());
    Ok(out)
}

// ------------------------------------------------------------------ ops

/// Options of `::ontology assert`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Fact {
    pub s: String,
    pub p: String,
    /// A node key; for a literal, omit it and pass `lit`.
    pub o: Option<String>,
    /// `{value, datatype?, unit?}`.
    pub lit: Option<JsonValue>,
    pub conf: Option<f64>,
    pub by: Option<String>,
    pub mem: Option<String>,
    /// `current` (default) or `pending`.
    pub status: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EventIn {
    /// Event node key; default `ev:<random>`.
    pub id: Option<String>,
    pub class: String,
    pub label: Option<String>,
    /// role → value: a string is a node key, an object `{value, datatype?,
    /// unit?}` a literal, a number or bool a plain literal.
    pub roles: serde_json::Map<String, JsonValue>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NodeIn {
    pub node: String,
    pub class: String,
    pub label: Option<String>,
    pub mem: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AssertOpts {
    ns: Option<String>,
    facts: Vec<Fact>,
    nodes: Vec<NodeIn>,
    event: Option<EventIn>,
    at: Option<i64>,
    /// `reject`, `flag` (default) or `off`.
    validate: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ReadOpts {
    ns: Option<String>,
    s: Option<String>,
    p: Option<String>,
    o: Option<String>,
    at: Option<i64>,
    /// `current` (default), `pending` or `any`.
    status: Option<String>,
    /// infer: only derived facts.
    derived_only: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ActivateOpts {
    ns: Option<String>,
    ontos: Vec<String>,
    policy: Option<String>,
    untyped: Option<String>,
}

fn ns_of(ns: &Option<String>) -> String {
    match ns {
        Some(x) if !x.is_empty() => x.clone(),
        _ => "default".to_string(),
    }
}

/// One validation finding.
#[derive(Debug, Clone, Serialize)]
pub struct Violation {
    pub s: String,
    pub p: String,
    pub o: String,
    pub rule: String,
    pub message: String,
}

impl<'s, S: Storage<'s>> Db<S> {
    fn on_q(&'s self, script: &str, p: Vec<(&str, DataValue)>) -> Result<NamedRows> {
        self.run_script(script, params(p), ScriptMutability::Mutable)
    }

    /// Create the ontology relations that don't exist yet.
    pub(crate) fn onto_ensure(&'s self) -> Result<()> {
        let missing: Vec<&(&str, &str)> = {
            let tx = self.transact()?;
            DDL.iter().filter(|(n, _)| tx.get_relation(n, false).is_err()).collect()
        };
        for (name, cols) in missing {
            if let Err(e) = self.run_script(&format!(":create {name} {cols}"), Default::default(), ScriptMutability::Mutable) {
                let tx = self.transact()?;
                if tx.get_relation(name, false).is_err() {
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    fn onto_exists(&'s self) -> Result<bool> {
        let tx = self.transact()?;
        Ok(tx.get_relation(ONTO_NS, false).is_ok())
    }

    /// Define (or update) an ontology: a new valid-time version of its
    /// header and of every class and property in `doc`. With `replace`,
    /// classes and properties of the ontology missing from `doc` are retracted.
    pub fn onto_define(&'s self, doc: &OntologyDoc, replace: bool) -> Result<JsonValue> {
        doc.check()?;
        self.onto_ensure()?;
        let at = now_micros();
        let o = s(&doc.onto);
        let mut script = format!(
            "{{?[onto, vt, version, doc, source] <- [[$onto, $vt, $version, $doc, $source]] \
              :put {ONTO} {{onto, vt => version, doc, source}}}}"
        );
        let mut p: Vec<(&str, DataValue)> = vec![
            ("onto", o.clone()),
            ("vt", vt(at, true)),
            ("version", s(&doc.version)),
            ("doc", s(&doc.doc)),
            ("source", s(&doc.source)),
        ];
        let crow: Vec<DataValue> = doc
            .classes
            .iter()
            .map(|c| {
                DataValue::List(vec![
                    o.clone(),
                    s(&c.name),
                    vt(at, true),
                    strs(&c.parents),
                    s(&c.label),
                    s(&c.doc),
                    strs(&c.aliases),
                ])
            })
            .collect();
        let prow: Vec<DataValue> = doc
            .properties
            .iter()
            .map(|x| {
                DataValue::List(vec![
                    o.clone(),
                    s(&x.name),
                    vt(at, true),
                    s(&x.kind),
                    strs(&x.domain),
                    strs(&x.range),
                    opt_s(&x.datatype),
                    s(&x.card),
                    DataValue::Bool(x.symmetric),
                    DataValue::Bool(x.transitive),
                    opt_s(&x.inverse),
                    strs(&x.parents),
                    strs(&x.chain),
                    DataValue::Bool(x.temporal),
                    opt_s(&x.unit),
                    s(&x.label),
                    s(&x.doc),
                    strs(&x.aliases),
                    strs(&x.cues),
                ])
            })
            .collect();
        if !crow.is_empty() {
            script.push_str(&format!(
                "{{?[onto, name, vt, parents, label, doc, aliases] <- $classes \
                  :put {ONTO_CLASS} {{onto, name, vt => parents, label, doc, aliases}}}}"
            ));
            p.push(("classes", DataValue::List(crow)));
        }
        if !prow.is_empty() {
            script.push_str(&format!(
                "{{?[onto, name, vt, kind, domain, range, datatype, card, symmetric, transitive, inverse, parents, chain, \
                    temporal, unit, label, doc, aliases, cues] <- $props \
                  :put {ONTO_PROP} {{onto, name, vt => kind, domain, range, datatype, card, symmetric, transitive, \
                    inverse, parents, chain, temporal, unit, label, doc, aliases, cues}}}}"
            ));
            p.push(("props", DataValue::List(prow)));
        }
        let mut retracted = 0;
        if replace {
            let old = self.onto_get(&doc.onto, None)?;
            if let Some(old) = old {
                let keep_c: BTreeSet<&str> = doc.classes.iter().map(|c| c.name.as_str()).collect();
                let keep_p: BTreeSet<&str> = doc.properties.iter().map(|c| c.name.as_str()).collect();
                let gone_c: Vec<String> =
                    old.classes.iter().filter(|c| !keep_c.contains(c.name.as_str())).map(|c| c.name.clone()).collect();
                let gone_p: Vec<String> = old
                    .properties
                    .iter()
                    .filter(|c| !keep_p.contains(c.name.as_str()))
                    .map(|c| c.name.clone())
                    .collect();
                retracted = gone_c.len() + gone_p.len();
                if !gone_c.is_empty() {
                    script.push_str(&format!(
                        "{{?[onto, name, vt, parents, label, doc, aliases] := *{ONTO_CLASS}{{onto, name, parents, label, doc, aliases @ 'NOW'}}, \
                           onto = $onto, is_in(name, $gone_c), vt = $vt_x \
                          :put {ONTO_CLASS} {{onto, name, vt => parents, label, doc, aliases}}}}"
                    ));
                    p.push(("gone_c", strs(&gone_c)));
                }
                if !gone_p.is_empty() {
                    script.push_str(&format!(
                        "{{?[onto, name, vt, kind, domain, range, datatype, card, symmetric, transitive, inverse, parents, chain, \
                            temporal, unit, label, doc, aliases, cues] := *{ONTO_PROP}{{onto, name, kind, domain, range, datatype, card, \
                            symmetric, transitive, inverse, parents, chain, temporal, unit, label, doc, aliases, cues @ 'NOW'}}, \
                           onto = $onto, is_in(name, $gone_p), vt = $vt_x \
                          :put {ONTO_PROP} {{onto, name, vt => kind, domain, range, datatype, card, symmetric, transitive, \
                            inverse, parents, chain, temporal, unit, label, doc, aliases, cues}}}}"
                    ));
                    p.push(("gone_p", strs(&gone_p)));
                }
                // the retraction is a hair later than the new rows, so a read at
                // `at` still sees the old ones
                p.push(("vt_x", vt(at, false)));
            }
        }
        self.on_q(&script, p)?;
        Ok(json!({
            "onto": doc.onto, "version": doc.version, "at": at,
            "classes": doc.classes.len(), "properties": doc.properties.len(), "retracted": retracted
        }))
    }

    /// An ontology as of `at` (default now), or `None`.
    pub fn onto_get(&'s self, onto: &str, at: Option<i64>) -> Result<Option<OntologyDoc>> {
        if !self.onto_exists()? {
            return Ok(None);
        }
        let a = at_clause(at);
        let head = self.on_q(
            &format!("?[version, doc, source] := *{ONTO}{{onto: $onto, version, doc, source @ {a}}}"),
            vec![("onto", s(onto))],
        )?;
        let Some(h) = head.rows.first() else { return Ok(None) };
        let cls = self.on_q(
            &format!("?[name, parents, label, doc, aliases] := *{ONTO_CLASS}{{onto: $onto, name, parents, label, doc, aliases @ {a}}}"),
            vec![("onto", s(onto))],
        )?;
        let props = self.on_q(
            &format!(
                "?[name, kind, domain, range, datatype, card, symmetric, transitive, inverse, parents, chain, temporal, unit, \
                   label, doc, aliases, cues] := *{ONTO_PROP}{{onto: $onto, name, kind, domain, range, datatype, card, symmetric, \
                   transitive, inverse, parents, chain, temporal, unit, label, doc, aliases, cues @ {a}}}"
            ),
            vec![("onto", s(onto))],
        )?;
        Ok(Some(OntologyDoc {
            onto: onto.to_string(),
            version: cell(&h[0]),
            doc: cell(&h[1]),
            source: cell(&h[2]),
            classes: cls
                .rows
                .iter()
                .map(|r| OntoClass {
                    name: cell(&r[0]),
                    parents: cell_strs(&r[1]),
                    label: cell(&r[2]),
                    doc: cell(&r[3]),
                    aliases: cell_strs(&r[4]),
                })
                .collect(),
            properties: props
                .rows
                .iter()
                .map(|r| OntoProp {
                    name: cell(&r[0]),
                    kind: cell(&r[1]),
                    domain: cell_strs(&r[2]),
                    range: cell_strs(&r[3]),
                    datatype: cell_opt(&r[4]),
                    card: cell(&r[5]),
                    symmetric: cell_bool(&r[6]),
                    transitive: cell_bool(&r[7]),
                    inverse: cell_opt(&r[8]),
                    parents: cell_strs(&r[9]),
                    chain: cell_strs(&r[10]),
                    temporal: cell_bool(&r[11]),
                    unit: cell_opt(&r[12]),
                    label: cell(&r[13]),
                    doc: cell(&r[14]),
                    aliases: cell_strs(&r[15]),
                    cues: cell_strs(&r[16]),
                })
                .collect(),
        }))
    }

    /// An ontology by name, loading the built-in `personal` pack on first use.
    fn onto_load(&'s self, onto: &str) -> Result<OntologyDoc> {
        if let Some(d) = self.onto_get(onto, None)? {
            return Ok(d);
        }
        if onto == "personal" {
            let d: OntologyDoc = serde_json::from_str(PERSONAL).into_diagnostic()?;
            self.onto_define(&d, false)?;
            return Ok(d);
        }
        bail!("ontology '{onto}' is not defined (`::ontology define` or `import` it first)")
    }

    /// Activate ontologies on a namespace (a new valid-time version).
    pub fn onto_activate(&'s self, ns: &str, ontos: &[String], policy: Option<&str>, untyped: Option<&str>) -> Result<JsonValue> {
        if ontos.is_empty() {
            bail!("ontology activate: 'ontos' must name at least one ontology");
        }
        let policy = policy.unwrap_or("cardinality");
        let untyped = untyped.unwrap_or(DEFAULT_UNTYPED);
        if !POLICIES.contains(&policy) {
            bail!("ontology activate: policy must be one of {POLICIES:?}, got '{policy}'");
        }
        if !UNTYPED.contains(&untyped) {
            bail!("ontology activate: untyped must be one of {UNTYPED:?}, got '{untyped}'");
        }
        self.onto_ensure()?;
        for o in ontos {
            self.onto_load(o)?;
        }
        let at = now_micros();
        self.on_q(
            &format!(
                "?[ns, vt, ontos, policy, untyped] <- [[$ns, $vt, $ontos, $policy, $untyped]] \
                 :put {ONTO_NS} {{ns, vt => ontos, policy, untyped}}"
            ),
            vec![
                ("ns", s(ns)),
                ("vt", vt(at, true)),
                ("ontos", strs(ontos)),
                ("policy", s(policy)),
                ("untyped", s(untyped)),
            ],
        )?;
        Ok(json!({"ns": ns, "ontos": ontos, "policy": policy, "untyped": untyped, "at": at}))
    }

    pub fn onto_deactivate(&'s self, ns: &str) -> Result<JsonValue> {
        if !self.onto_exists()? {
            return Ok(json!({"ns": ns, "deactivated": false}));
        }
        let at = now_micros();
        let r = self.on_q(
            &format!(
                "?[ns, vt, ontos, policy, untyped] := *{ONTO_NS}{{ns, ontos, policy, untyped @ 'NOW'}}, ns = $ns, vt = $vt \
                 :put {ONTO_NS} {{ns, vt => ontos, policy, untyped}}"
            ),
            vec![("ns", s(ns)), ("vt", vt(at, false))],
        )?;
        Ok(json!({"ns": ns, "deactivated": !r.rows.is_empty() || true}))
    }

    /// The ontology active on `ns`, merged, or `None` (no ontology: memory
    /// behaves as before).
    pub fn onto_active(&'s self, ns: &str) -> Result<Option<Arc<Active>>> {
        if !self.onto_exists()? {
            return Ok(None);
        }
        let r = self.on_q(
            &format!("?[ontos, policy, untyped] := *{ONTO_NS}{{ns: $ns, ontos, policy, untyped @ 'NOW'}}"),
            vec![("ns", s(ns))],
        )?;
        let Some(row) = r.rows.first() else { return Ok(None) };
        let names = cell_strs(&row[0]);
        let policy = cell(&row[1]);
        let untyped = cell(&row[2]);
        let mut docs = vec![];
        for n in &names {
            docs.push(self.onto_load(n)?);
        }
        Ok(Some(Arc::new(Active::merge(&docs, &policy, &untyped))))
    }

    /// Stored claim types of memories (`kleos_mem_type`).
    pub(crate) fn onto_types_of(&'s self, ns: &str, ids: &[String]) -> Result<BTreeMap<String, crate::ontology_type::ClaimType>> {
        let mut out = BTreeMap::new();
        if ids.is_empty() || !self.onto_exists()? {
            return Ok(out);
        }
        let r = self.on_q(
            &format!(
                "?[id, cls, subject, prop, score, by] := *{MEM_TYPE}{{ns, id, cls, subject, prop, score, by}}, ns = $ns, is_in(id, $ids)"
            ),
            vec![("ns", s(ns)), ("ids", strs(ids))],
        )?;
        for row in r.rows {
            out.insert(
                cell(&row[0]),
                crate::ontology_type::ClaimType {
                    cls: cell(&row[1]),
                    subject: cell_opt(&row[2]),
                    prop: cell_opt(&row[3]),
                    score: row[4].get_float().unwrap_or(0.0),
                    by: cell(&row[5]),
                    speaker: None,
                    content: 0,
                    concrete: false,
                },
            );
        }
        Ok(out)
    }

    // -------------------------------------------------------------- facts

    fn onto_assert(&'s self, opts: AssertOpts) -> Result<JsonValue> {
        let ns = ns_of(&opts.ns);
        let act = self
            .onto_active(&ns)?
            .ok_or_else(|| miette!("ontology assert: namespace '{ns}' has no active ontology (`::ontology activate`)"))?;
        let at = opts.at.unwrap_or_else(now_micros);
        let mode = opts.validate.as_deref().unwrap_or("flag");
        if !["reject", "flag", "off"].contains(&mode) {
            bail!("ontology assert: validate must be reject, flag or off, got '{mode}'");
        }
        let mut facts = opts.facts.clone();
        let mut nodes = opts.nodes.clone();
        let mut event_id = None;
        if let Some(ev) = &opts.event {
            if ev.class.is_empty() {
                bail!("ontology assert: event needs a class");
            }
            let id = ev.id.clone().unwrap_or_else(|| format!("ev:{:012x}{:06x}", at, rand::random::<u32>() & 0xff_ffff));
            nodes.push(NodeIn { node: id.clone(), class: ev.class.clone(), label: ev.label.clone(), mem: None });
            for (role, v) in &ev.roles {
                let (o, lit) = match v {
                    JsonValue::String(x) => (Some(x.clone()), None),
                    JsonValue::Object(_) => (None, Some(v.clone())),
                    other => (None, Some(json!({ "value": other }))),
                };
                facts.push(Fact { s: id.clone(), p: role.clone(), o, lit, ..Default::default() });
            }
            event_id = Some(id);
        }
        // canonical property names (aliases resolve)
        for f in facts.iter_mut() {
            if let Some(p) = act.prop(&f.p) {
                f.p = p.name.clone();
            }
            if f.o.is_none() {
                let lit = f.lit.clone().ok_or_else(|| miette!("ontology assert: fact ({}, {}) needs 'o' or 'lit'", f.s, f.p))?;
                f.o = Some(literal_key(&lit));
            }
        }
        let classes = self.onto_node_classes(&ns, None)?;
        let mut node_cls = classes.clone();
        for n in &nodes {
            node_cls.insert(n.node.clone(), n.class.clone());
        }
        let viol = validate_facts(&act, &facts, &node_cls);
        if mode == "reject" && !viol.is_empty() {
            bail!(
                "ontology assert rejected: {}",
                viol.iter().map(|v| format!("({}, {}, {}): {}", v.s, v.p, v.o, v.message)).join("; ")
            );
        }
        // single-valued properties: the subject's other current value ends
        let mut ends: Vec<DataValue> = vec![];
        let singles: Vec<(String, String, String)> = facts
            .iter()
            .filter(|f| act.prop(&f.p).is_some_and(|p| p.single() && f.status.as_deref().unwrap_or("current") == "current"))
            .map(|f| (f.s.clone(), f.p.clone(), f.o.clone().unwrap()))
            .collect();
        if !singles.is_empty() {
            let cur = self.on_q(
                &format!(
                    "?[s, p, o, lit, conf, by, mem, status] := *{ONTO_FACT}{{ns, s, p, o, lit, conf, by, mem, status @ 'NOW'}}, \
                     ns = $ns, status = 'current', is_in(p, $ps)"
                ),
                vec![("ns", s(&ns)), ("ps", strs(&singles.iter().map(|x| x.1.clone()).unique().collect::<Vec<_>>()))],
            )?;
            for r in cur.rows {
                let (fs, fp, fo) = (cell(&r[0]), cell(&r[1]), cell(&r[2]));
                if singles.iter().any(|(a, b, c)| *a == fs && *b == fp && *c != fo) {
                    ends.push(DataValue::List(vec![
                        s(&ns),
                        s(&fs),
                        s(&fp),
                        s(&fo),
                        vt(at, false),
                        r[3].clone(),
                        r[4].clone(),
                        r[5].clone(),
                        r[6].clone(),
                        r[7].clone(),
                    ]));
                }
            }
        }
        let rows: Vec<DataValue> = facts
            .iter()
            .map(|f| {
                DataValue::List(vec![
                    s(&ns),
                    s(&f.s),
                    s(&f.p),
                    s(f.o.as_deref().unwrap()),
                    vt(at, true),
                    f.lit.clone().map(|j| DataValue::Json(JsonData(j))).unwrap_or(DataValue::Null),
                    DataValue::from(f.conf.unwrap_or(1.0)),
                    s(f.by.as_deref().unwrap_or("assert")),
                    opt_s(&f.mem),
                    s(f.status.as_deref().unwrap_or("current")),
                ])
            })
            .collect();
        let nrows: Vec<DataValue> = nodes
            .iter()
            .map(|n| DataValue::List(vec![s(&ns), s(&n.node), vt(at, true), s(&n.class), opt_s(&n.label), opt_s(&n.mem)]))
            .collect();
        let mut script = String::new();
        let mut p: Vec<(&str, DataValue)> = vec![];
        if !ends.is_empty() {
            script.push_str(&format!(
                "{{?[ns, s, p, o, vt, lit, conf, by, mem, status] <- $ends :put {ONTO_FACT} {{ns, s, p, o, vt => lit, conf, by, mem, status}}}}"
            ));
            p.push(("ends", DataValue::List(ends.clone())));
        }
        if !rows.is_empty() {
            script.push_str(&format!(
                "{{?[ns, s, p, o, vt, lit, conf, by, mem, status] <- $rows :put {ONTO_FACT} {{ns, s, p, o, vt => lit, conf, by, mem, status}}}}"
            ));
            p.push(("rows", DataValue::List(rows)));
        }
        if !nrows.is_empty() {
            script.push_str(&format!(
                "{{?[ns, node, vt, class, label, mem] <- $nodes :put {ONTO_NODE} {{ns, node, vt => class, label, mem}}}}"
            ));
            p.push(("nodes", DataValue::List(nrows)));
        }
        if !script.is_empty() {
            self.on_q(&script, p)?;
        }
        Ok(json!({
            "ns": ns, "at": at, "asserted": facts.len(), "nodes": nodes.len(), "ended": ends.len(),
            "event": event_id, "violations": viol,
        }))
    }

    fn onto_retract(&'s self, opts: AssertOpts) -> Result<JsonValue> {
        let ns = ns_of(&opts.ns);
        let at = opts.at.unwrap_or_else(now_micros);
        let act = self.onto_active(&ns)?;
        let mut keys = vec![];
        for f in &opts.facts {
            let p = act.as_ref().and_then(|a| a.prop(&f.p)).map(|x| x.name.clone()).unwrap_or(f.p.clone());
            let o = match (&f.o, &f.lit) {
                (Some(o), _) => o.clone(),
                (None, Some(l)) => literal_key(l),
                _ => bail!("ontology retract: fact ({}, {}) needs 'o' or 'lit'", f.s, f.p),
            };
            keys.push(DataValue::List(vec![s(&f.s), s(&p), s(&o)]));
        }
        if keys.is_empty() || !self.onto_exists()? {
            return Ok(json!({"ns": ns, "retracted": 0}));
        }
        let r = self.on_q(
            &format!(
                "k[s, p, o] <- $keys \
                 ?[ns, s, p, o, vt, lit, conf, by, mem, status] := k[s, p, o], *{ONTO_FACT}{{ns, s, p, o, lit, conf, by, mem, status @ 'NOW'}}, \
                    ns = $ns, vt = $vt \
                 :put {ONTO_FACT} {{ns, s, p, o, vt => lit, conf, by, mem, status}}"
            ),
            vec![("ns", s(&ns)), ("keys", DataValue::List(keys)), ("vt", vt(at, false))],
        )?;
        Ok(json!({"ns": ns, "retracted": r.rows.len(), "at": at}))
    }

    fn onto_node_classes(&'s self, ns: &str, at: Option<i64>) -> Result<BTreeMap<String, String>> {
        if !self.onto_exists()? {
            return Ok(BTreeMap::new());
        }
        let r = self.on_q(
            &format!("?[node, class] := *{ONTO_NODE}{{ns: $ns, node, class @ {}}}", at_clause(at)),
            vec![("ns", s(ns))],
        )?;
        Ok(r.rows.iter().map(|x| (cell(&x[0]), cell(&x[1]))).collect())
    }

    fn statuses(st: &Option<String>) -> Result<Vec<String>> {
        Ok(match st.as_deref().unwrap_or("current") {
            "current" => vec!["current".into()],
            "pending" => vec!["pending".into()],
            "any" => vec!["current".into(), "pending".into()],
            x => bail!("ontology: status must be current, pending or any, got '{x}'"),
        })
    }

    fn onto_facts(&'s self, o: &ReadOpts) -> Result<JsonValue> {
        let ns = ns_of(&o.ns);
        if !self.onto_exists()? {
            return Ok(json!([]));
        }
        let r = self.on_q(
            &format!(
                "?[s, p, o, lit, conf, by, mem, status] := *{ONTO_FACT}{{ns, s, p, o, lit, conf, by, mem, status @ {}}}, \
                 ns = $ns, is_in(status, $st)",
                at_clause(o.at)
            ),
            vec![("ns", s(&ns)), ("st", strs(&Self::statuses(&o.status)?))],
        )?;
        let out: Vec<JsonValue> = r
            .rows
            .iter()
            .filter(|x| filt(o, &cell(&x[0]), &cell(&x[1]), &cell(&x[2])))
            .map(|x| {
                json!({"s": jv(&x[0]), "p": jv(&x[1]), "o": jv(&x[2]), "lit": jv(&x[3]), "conf": jv(&x[4]),
                       "by": jv(&x[5]), "mem": jv(&x[6]), "status": jv(&x[7])})
            })
            .collect();
        Ok(JsonValue::Array(out))
    }

    /// Facts plus everything the active ontology derives from them
    /// (symmetric, inverse, sub-property, transitive, chains), computed by a
    /// Datalog program generated from the ontology.
    pub fn onto_infer(&'s self, ns: &str, at: Option<i64>, status: &Option<String>) -> Result<Vec<(String, String, String, bool)>> {
        let Some(act) = self.onto_active(ns)? else {
            bail!("ontology infer: namespace '{ns}' has no active ontology");
        };
        let (prog, p) = infer_program(&act, ns, at, &Self::statuses(status)?);
        let r = self.on_q(&prog, p)?;
        Ok(r
            .rows
            .iter()
            .filter(|x| !cell(&x[1]).starts_with("__chain"))
            .map(|x| (cell(&x[0]), cell(&x[1]), cell(&x[2]), cell_bool(&x[3])))
            .collect())
    }

    fn onto_validate(&'s self, o: &ReadOpts) -> Result<JsonValue> {
        let ns = ns_of(&o.ns);
        let Some(act) = self.onto_active(&ns)? else {
            bail!("ontology validate: namespace '{ns}' has no active ontology");
        };
        let facts: Vec<Fact> = serde_json::from_value(self.onto_facts(o)?).into_diagnostic().map(|v: Vec<JsonValue>| {
            v.into_iter()
                .map(|f| Fact {
                    s: f["s"].as_str().unwrap_or("").to_string(),
                    p: f["p"].as_str().unwrap_or("").to_string(),
                    o: f["o"].as_str().map(|x| x.to_string()),
                    lit: if f["lit"].is_null() { None } else { Some(f["lit"].clone()) },
                    ..Default::default()
                })
                .collect()
        })?;
        let classes = self.onto_node_classes(&ns, o.at)?;
        let mut viol = validate_facts(&act, &facts, &classes);
        // cardinality over the stored facts
        let mut count: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
        for f in &facts {
            if act.prop(&f.p).is_some_and(|p| p.single()) {
                count.entry((f.s.clone(), f.p.clone())).or_default().push(f.o.clone().unwrap_or_default());
            }
        }
        for ((fs, fp), os) in count {
            if os.len() > 1 {
                viol.push(Violation {
                    s: fs,
                    p: fp.clone(),
                    o: os.join(" | "),
                    rule: "cardinality".into(),
                    message: format!("'{fp}' is single-valued but has {} current values", os.len()),
                });
            }
        }
        Ok(json!({"ns": ns, "facts": facts.len(), "violations": viol}))
    }

    /// One entry point for every surface that speaks JSON: `::ontology <op>`,
    /// `POST /ontology/:op`, the bindings (docs/specs/ontology.md, "Ops").
    pub fn ontology_call(&'s self, op: &str, args: JsonValue) -> Result<JsonValue> {
        let mut m = match args {
            JsonValue::Null => serde_json::Map::new(),
            JsonValue::Object(m) => m,
            other => bail!("ontology {op}: options must be an object, got {other}"),
        };
        let from = |m: serde_json::Map<String, JsonValue>| -> Result<JsonValue> { Ok(JsonValue::Object(m)) };
        let take = |m: &mut serde_json::Map<String, JsonValue>, k: &str| -> Option<String> {
            m.remove(k).and_then(|v| v.as_str().map(|x| x.to_string()))
        };
        Ok(match op {
            "define" => {
                let replace = m.remove("replace").and_then(|v| v.as_bool()).unwrap_or(false);
                let doc: OntologyDoc = serde_json::from_value(from(m)?)
                    .map_err(|e| miette!("ontology define: bad ontology: {e}"))?;
                self.onto_define(&doc, replace)?
            }
            "import" => {
                let format = take(&mut m, "format").unwrap_or_else(|| "json".into());
                let text = take(&mut m, "text").ok_or_else(|| miette!("ontology import needs 'text'"))?;
                let onto = take(&mut m, "onto");
                let replace = m.remove("replace").and_then(|v| v.as_bool()).unwrap_or(false);
                let (mut doc, warnings) = match format.as_str() {
                    "json" => (
                        serde_json::from_str::<OntologyDoc>(&text).map_err(|e| miette!("ontology import: bad JSON ontology: {e}"))?,
                        vec![],
                    ),
                    "turtle" | "ttl" => crate::ontology_ttl::import_turtle(&text, onto.as_deref().unwrap_or(""))?,
                    x => bail!("ontology import: format must be json or turtle, got '{x}'"),
                };
                if let Some(o) = onto {
                    doc.onto = o;
                }
                let mut out = self.onto_define(&doc, replace)?;
                out["warnings"] = json!(warnings);
                out
            }
            "export" => {
                let onto = take(&mut m, "onto").ok_or_else(|| miette!("ontology export needs 'onto'"))?;
                let format = take(&mut m, "format").unwrap_or_else(|| "json".into());
                let at = m.remove("at").and_then(|v| v.as_i64());
                let doc = self.onto_get(&onto, at)?.ok_or_else(|| miette!("ontology '{onto}' is not defined"))?;
                match format.as_str() {
                    "json" => serde_json::to_value(&doc).into_diagnostic()?,
                    "turtle" | "ttl" => json!({"onto": onto, "format": "turtle", "text": crate::ontology_ttl::export_turtle(&doc)}),
                    x => bail!("ontology export: format must be json or turtle, got '{x}'"),
                }
            }
            "get" => {
                let onto = take(&mut m, "onto").ok_or_else(|| miette!("ontology get needs 'onto'"))?;
                let at = m.remove("at").and_then(|v| v.as_i64());
                let doc = match self.onto_get(&onto, at)? {
                    Some(d) => d,
                    None if onto == "personal" => serde_json::from_str(PERSONAL).into_diagnostic()?,
                    None => bail!("ontology '{onto}' is not defined"),
                };
                serde_json::to_value(doc).into_diagnostic()?
            }
            "list" => {
                let mut names: Vec<JsonValue> = vec![];
                if self.onto_exists()? {
                    let r = self.on_q(&format!("?[onto, version, doc] := *{ONTO}{{onto, version, doc @ 'NOW'}}"), vec![])?;
                    names = r.rows.iter().map(|x| json!({"onto": jv(&x[0]), "version": jv(&x[1]), "doc": jv(&x[2])})).collect();
                }
                if !names.iter().any(|n| n["onto"] == "personal") {
                    names.push(json!({"onto": "personal", "version": "built-in", "doc": "kleos default personal-memory ontology"}));
                }
                let mut active = vec![];
                if self.onto_exists()? {
                    let r = self.on_q(&format!("?[ns, ontos, policy, untyped] := *{ONTO_NS}{{ns, ontos, policy, untyped @ 'NOW'}}"), vec![])?;
                    active = r
                        .rows
                        .iter()
                        .map(|x| json!({"ns": jv(&x[0]), "ontos": jv(&x[1]), "policy": jv(&x[2]), "untyped": jv(&x[3])}))
                        .collect();
                }
                json!({"ontologies": names, "active": active})
            }
            "activate" => {
                let o: ActivateOpts = serde_json::from_value(from(m)?).map_err(|e| miette!("ontology activate: bad options: {e}"))?;
                self.onto_activate(&ns_of(&o.ns), &o.ontos, o.policy.as_deref(), o.untyped.as_deref())?
            }
            "deactivate" => {
                let ns = take(&mut m, "ns");
                self.onto_deactivate(&ns_of(&ns))?
            }
            "assert" => self.onto_assert(serde_json::from_value(from(m)?).map_err(|e| miette!("ontology assert: bad options: {e}"))?)?,
            "retract" => self.onto_retract(serde_json::from_value(from(m)?).map_err(|e| miette!("ontology retract: bad options: {e}"))?)?,
            "facts" => self.onto_facts(&serde_json::from_value(from(m)?).map_err(|e| miette!("ontology facts: bad options: {e}"))?)?,
            "infer" => {
                let o: ReadOpts = serde_json::from_value(from(m)?).map_err(|e| miette!("ontology infer: bad options: {e}"))?;
                let ns = ns_of(&o.ns);
                let rows = self.onto_infer(&ns, o.at, &o.status)?;
                JsonValue::Array(
                    rows.into_iter()
                        .filter(|(a, b, c, d)| (!o.derived_only || *d) && filt(&o, a, b, c))
                        .map(|(a, b, c, d)| json!({"s": a, "p": b, "o": c, "derived": d}))
                        .collect(),
                )
            }
            "validate" => self.onto_validate(&serde_json::from_value(from(m)?).map_err(|e| miette!("ontology validate: bad options: {e}"))?)?,
            "gate" => {
                // what the cardinality policy does to a (candidate `old`, new
                // claim `new`) pair, with both types (debugging, evaluation)
                let ns = ns_of(&take(&mut m, "ns"));
                let old = take(&mut m, "old").ok_or_else(|| miette!("ontology gate needs 'old'"))?;
                let new = take(&mut m, "new").ok_or_else(|| miette!("ontology gate needs 'new'"))?;
                let act = match self.onto_active(&ns)? {
                    Some(a) => a,
                    None => Arc::new(Active::merge(&[serde_json::from_str(PERSONAL).into_diagnostic()?], "cardinality", DEFAULT_UNTYPED)),
                };
                let embedder = self.mem_embedder_name()?;
                let (vecs, protos) = match &embedder {
                    Some(e) => (
                        crate::embed::embed_texts(e, &[old.as_str(), new.as_str()])?.into_iter().map(|v| Some(v.to_vec())).collect(),
                        Some(prototypes(&act, e)?),
                    ),
                    None => (vec![None, None], None),
                };
                let ps = protos.as_deref().map(|v| v.as_slice());
                let to = crate::ontology_type::type_claim(&act, ps, &old, vecs[0].as_deref());
                let tn = crate::ontology_type::type_claim(&act, ps, &new, vecs[1].as_deref());
                let g = match crate::ontology_type::gate(&act, &tn, &to) {
                    crate::ontology_type::Gate::Allow => "allow",
                    crate::ontology_type::Gate::CapPending => "cap_pending",
                    crate::ontology_type::Gate::NoSupersede => "no_supersede",
                    crate::ontology_type::Gate::Block => "block",
                };
                json!({"gate": g, "old": to, "new": tn})
            }
            "type" => {
                let ns = take(&mut m, "ns");
                let text = take(&mut m, "text").ok_or_else(|| miette!("ontology type needs 'text'"))?;
                let ns = ns_of(&ns);
                let act = match self.onto_active(&ns)? {
                    Some(a) => a,
                    None => Arc::new(Active::merge(
                        &[serde_json::from_str(PERSONAL).into_diagnostic()?],
                        "cardinality",
                        DEFAULT_UNTYPED,
                    )),
                };
                let embedder = self.mem_embedder_name()?;
                let (vec, protos) = match &embedder {
                    Some(e) => (
                        crate::embed::embed_texts(e, &[text.as_str()])?.pop().map(|v| v.to_vec()),
                        Some(prototypes(&act, e)?),
                    ),
                    None => (None, None),
                };
                let t = crate::ontology_type::type_claim(&act, protos.as_deref().map(|v| v.as_slice()), &text, vec.as_deref());
                let mut out = serde_json::to_value(&t).into_diagnostic()?;
                if let (Some(ps), Some(v)) = (&protos, &vec) {
                    out["top"] = json!(crate::ontology_type::top_props(&act, ps, &text, v, 3));
                }
                out["kin"] = json!(crate::ontology_type::kin_facts(&text)
                    .into_iter()
                    .map(|k| json!({"s": k.s, "p": k.p, "o": k.o, "label": k.label}))
                    .collect::<Vec<_>>());
                out
            }
            other => bail!(
                "unknown ontology op '{other}': expected define, import, export, get, list, activate, deactivate, \
                 assert, retract, facts, infer, validate, type or gate"
            ),
        })
    }
}

/// `remember`'s commit blocks under an active ontology: the added claims'
/// types (`kleos_mem_type`) and the kinship and role facts they state by
/// rule, with the relatives as Person nodes. `claims` are (added, id, text).
pub(crate) fn remember_blocks<'a>(
    ns: &str,
    onto: &str,
    at: i64,
    claims: &[(bool, String, String)],
    types: &[Option<crate::ontology_type::ClaimType>],
) -> (String, Vec<(&'a str, DataValue)>) {
    let mut trows = vec![];
    let mut frows = vec![];
    let mut nrows = vec![];
    let mut seen_nodes = BTreeSet::new();
    for ((added, id, text), t) in claims.iter().zip(types) {
        if !added {
            continue;
        }
        if let Some(t) = t {
            trows.push(DataValue::List(vec![
                s(ns),
                s(id),
                s(onto),
                s(&t.cls),
                opt_s(&t.subject),
                opt_s(&t.prop),
                DataValue::from(t.score),
                s(&t.by),
            ]));
        }
        for k in crate::ontology_type::kin_facts(text) {
            frows.push(DataValue::List(vec![
                s(ns),
                s(&k.s),
                s(&k.p),
                s(&k.o),
                vt(at, true),
                DataValue::Null,
                DataValue::from(1.0),
                s("rule:kin"),
                s(id),
                s("current"),
            ]));
            for (node, label) in [(&k.s, &k.label), (&k.o, &None)] {
                if seen_nodes.insert(node.clone()) && k.p != "same_as" {
                    nrows.push(DataValue::List(vec![s(ns), s(node), vt(at, true), s("Person"), opt_s(label), s(id)]));
                }
            }
        }
    }
    let mut script = String::new();
    let mut p: Vec<(&'a str, DataValue)> = vec![];
    if !trows.is_empty() {
        script.push_str(&format!(
            "{{?[ns, id, onto, cls, subject, prop, score, by] <- $onto_types \
              :put {MEM_TYPE} {{ns, id => onto, cls, subject, prop, score, by}}}}"
        ));
        p.push(("onto_types", DataValue::List(trows)));
    }
    if !frows.is_empty() {
        script.push_str(&format!(
            "{{?[ns, s, p, o, vt, lit, conf, by, mem, status] <- $onto_facts \
              :put {ONTO_FACT} {{ns, s, p, o, vt => lit, conf, by, mem, status}}}}"
        ));
        p.push(("onto_facts", DataValue::List(frows)));
    }
    if !nrows.is_empty() {
        script.push_str(&format!(
            "{{?[ns, node, vt, class, label, mem] <- $onto_nodes :put {ONTO_NODE} {{ns, node, vt => class, label, mem}}}}"
        ));
        p.push(("onto_nodes", DataValue::List(nrows)));
    }
    (script, p)
}

fn filt(o: &ReadOpts, fs: &str, fp: &str, fo: &str) -> bool {
    o.s.as_deref().map_or(true, |x| x == fs) && o.p.as_deref().map_or(true, |x| x == fp) && o.o.as_deref().map_or(true, |x| x == fo)
}

/// Domain, range, datatype and unit checks (cardinality is checked over the
/// stored facts by `validate`).
pub fn validate_facts(act: &Active, facts: &[Fact], classes: &BTreeMap<String, String>) -> Vec<Violation> {
    let mut out = vec![];
    let under = |node: &str, allowed: &[String]| -> Option<bool> {
        if allowed.is_empty() {
            return Some(true);
        }
        let c = classes.get(node)?;
        let anc = act.ancestors(c);
        Some(allowed.iter().any(|a| anc.contains(a) || a == "Thing" || a == "owl:Thing"))
    };
    for f in facts {
        let o = f.o.clone().unwrap_or_default();
        let v = |rule: &str, message: String| Violation { s: f.s.clone(), p: f.p.clone(), o: o.clone(), rule: rule.into(), message };
        let Some(p) = act.prop(&f.p) else {
            out.push(v("property", format!("'{}' is not a property of the active ontology", f.p)));
            continue;
        };
        if under(&f.s, &p.domain) == Some(false) {
            out.push(v("domain", format!("subject '{}' ({}) is not in the domain {:?}", f.s, classes[&f.s], p.domain)));
        }
        match p.kind.as_str() {
            "object" => {
                if f.lit.is_some() && f.o.is_none() {
                    out.push(v("range", format!("'{}' links nodes; got a literal", p.name)));
                } else if under(&o, &p.range) == Some(false) {
                    out.push(v("range", format!("object '{}' ({}) is not in the range {:?}", o, classes[&o], p.range)));
                }
            }
            _ => {
                if let (Some(dt), Some(lit)) = (&p.datatype, &f.lit) {
                    let got = lit.get("datatype").and_then(|d| d.as_str());
                    let value = lit.get("value");
                    let ok = match got {
                        Some(g) => g == dt,
                        None => match dt.as_str() {
                            "xsd:integer" => value.is_some_and(|x| x.is_i64() || x.is_u64()),
                            "xsd:decimal" | "xsd:double" | "xsd:float" => value.is_some_and(|x| x.is_number()),
                            "xsd:boolean" => value.is_some_and(|x| x.is_boolean()),
                            _ => true,
                        },
                    };
                    if !ok {
                        out.push(v("datatype", format!("'{}' takes {dt}, got {}", p.name, got.map_or_else(|| value.map_or("nothing".into(), |x| x.to_string()), |g| g.to_string()))));
                    }
                }
                if let (Some(u), Some(lit)) = (&p.unit, &f.lit) {
                    if u != "currency" {
                        if let Some(got) = lit.get("unit").and_then(|x| x.as_str()) {
                            if got != u {
                                out.push(v("unit", format!("'{}' is in {u}, got {got}", p.name)));
                            }
                        }
                    } else if let Some(got) = lit.get("unit").and_then(|x| x.as_str()) {
                        if !(got.len() == 3 && got.chars().all(|c| c.is_ascii_uppercase())) {
                            out.push(v("unit", format!("'{}' is a currency amount; '{got}' is not an ISO 4217 code", p.name)));
                        }
                    }
                }
            }
        }
    }
    out
}

/// The inference program for `act` over the namespace's facts.
pub fn infer_program<'a>(act: &Active, ns: &str, at: Option<i64>, statuses: &[String]) -> (String, Vec<(&'a str, DataValue)>) {
    let mut sym = vec![];
    let mut trans = vec![];
    let mut inv: BTreeSet<(String, String)> = BTreeSet::new();
    let mut sup: Vec<DataValue> = vec![];
    let mut chains: Vec<(String, String, String)> = vec![];
    for p in &act.props {
        if p.symmetric {
            sym.push(p.name.clone());
        }
        if p.transitive {
            trans.push(p.name.clone());
        }
        if let Some(q) = &p.inverse {
            inv.insert((p.name.clone(), q.clone()));
            inv.insert((q.clone(), p.name.clone()));
        }
        for q in &p.parents {
            sup.push(DataValue::List(vec![s(&p.name), s(q)]));
        }
        if p.chain.len() >= 2 {
            // fold [p1, p2, p3] into __chain:r:1 = p1 o p2, r = __chain:r:1 o p3
            let mut left = p.chain[0].clone();
            for (i, q) in p.chain[1..].iter().enumerate() {
                let head = if i == p.chain.len() - 2 { p.name.clone() } else { format!("__chain:{}:{}", p.name, i + 1) };
                chains.push((head.clone(), left.clone(), q.clone()));
                left = head;
            }
        }
    }
    let invs: Vec<DataValue> = inv.into_iter().map(|(a, b)| DataValue::List(vec![s(&a), s(&b)])).collect();
    let chs: Vec<DataValue> = chains.into_iter().map(|(a, b, c)| DataValue::List(vec![s(&a), s(&b), s(&c)])).collect();
    // never-matching rows keep the constant rules' arity known when a list is empty
    let pad2 = |mut v: Vec<DataValue>| {
        v.push(DataValue::List(vec![s("\u{0}"), s("\u{0}")]));
        DataValue::List(v)
    };
    let pad3 = |mut v: Vec<DataValue>| {
        v.push(DataValue::List(vec![s("\u{0}"), s("\u{0}"), s("\u{0}")]));
        DataValue::List(v)
    };
    let prog = format!(
        "inv[p, q] <- $inv \
         sup[p, q] <- $sup \
         chain[r, p, q] <- $chain \
         base[s, p, o] := *{ONTO_FACT}{{ns, s, p, o, status @ {a}}}, ns = $ns, is_in(status, $st) \
         f[s, p, o] := base[s, p, o] \
         f[o, p, s] := f[s, p, o], is_in(p, $sym) \
         f[o, q, s] := f[s, p, o], inv[p, q] \
         f[s, q, o] := f[s, p, o], sup[p, q] \
         f[s, p, o2] := f[s, p, o], f[o, p, o2], is_in(p, $trans), s != o2 \
         f[s, r, o2] := f[s, p, o], f[o, q, o2], chain[r, p, q], s != o2 \
         f[t, p, o] := f[x, p, o], f[x, 'same_as', t], p != 'same_as', t != o \
         f[s, p, t] := f[s, p, x], f[x, 'same_as', t], p != 'same_as', s != t \
         ?[s, p, o, derived] := f[s, p, o], base[s, p, o], derived = false \
         ?[s, p, o, derived] := f[s, p, o], not base[s, p, o], derived = true",
        a = at_clause(at)
    );
    let p = vec![
        ("ns", s(ns)),
        ("st", strs(statuses)),
        ("sym", strs(&sym)),
        ("trans", strs(&trans)),
        ("inv", pad2(invs)),
        ("sup", pad2(sup)),
        ("chain", pad3(chs)),
    ];
    (prog, p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn personal_pack_parses_and_checks() {
        let d: OntologyDoc = serde_json::from_str(PERSONAL).unwrap();
        d.check().unwrap();
        let a = Active::merge(&[d], "cardinality", "decide");
        assert!(a.prop("lives_in").unwrap().single());
        assert!(!a.prop("likes").unwrap().single());
        assert_eq!(a.prop("residence").unwrap().name, "lives_in");
        assert!(a.ancestors("Activity").contains("Thing"));
        // every name a property points at exists
        for p in &a.props {
            for q in p.parents.iter().chain(p.chain.iter()).chain(p.inverse.iter()) {
                assert!(a.by_name.contains_key(q), "{} -> {q}", p.name);
            }
            for c in p.domain.iter().chain(p.range.iter()) {
                assert!(a.classes.contains_key(c), "{} class {c}", p.name);
            }
        }
    }

    #[test]
    fn override_by_name() {
        let base: OntologyDoc = serde_json::from_str(PERSONAL).unwrap();
        let over = OntologyDoc {
            onto: "x".into(),
            properties: vec![OntoProp { name: "likes".into(), card: "one".into(), ..Default::default() }],
            ..Default::default()
        };
        let a = Active::merge(&[base, over], "cardinality", "decide");
        assert!(a.prop("likes").unwrap().single());
    }

    #[test]
    fn literal_keys() {
        assert_eq!(literal_key(&json!({"value": 1500, "unit": "EUR"})), "1500 EUR");
        assert_eq!(literal_key(&json!({"value": "blue"})), "blue");
    }
}
