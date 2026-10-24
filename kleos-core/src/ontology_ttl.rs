/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The OWL/RDF subset of Turtle that ontologies use (docs/specs/ontology.md,
//! "OWL/RDF subset in Turtle"): a small reader into [`OntologyDoc`] and a
//! writer back. Not a general RDF parser: restrictions, unions and
//! individuals are skipped and counted in the warnings.

use std::collections::{BTreeMap, BTreeSet};

use miette::{bail, Result};

use crate::ontology::{OntoClass, OntoProp, OntologyDoc};

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Iri(String),
    Pname(String),
    Blank(String),
    Lit { v: String, lang: Option<String>, dt: Option<String> },
    Num(String),
    Bool(bool),
    A,
    Dot,
    Semi,
    Comma,
    LBracket,
    RBracket,
    LParen,
    RParen,
    Prefix,
    Base,
}

fn tokenize(src: &str) -> Result<Vec<Tok>> {
    let c: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut out = vec![];
    let name_char = |ch: char| ch.is_alphanumeric() || matches!(ch, '_' | '-' | '.' | ':' | '%');
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        match ch {
            '#' => {
                while i < c.len() && c[i] != '\n' {
                    i += 1;
                }
            }
            '<' => {
                let j = (i + 1..c.len()).find(|&j| c[j] == '>').ok_or_else(|| miette::miette!("turtle: unclosed IRI"))?;
                out.push(Tok::Iri(c[i + 1..j].iter().collect()));
                i = j + 1;
            }
            '"' | '\'' => {
                let q = ch;
                let long = i + 2 < c.len() && c[i + 1] == q && c[i + 2] == q;
                let mut j = if long { i + 3 } else { i + 1 };
                let mut v = String::new();
                loop {
                    if j >= c.len() {
                        bail!("turtle: unclosed string literal");
                    }
                    if c[j] == '\\' && j + 1 < c.len() {
                        v.push(match c[j + 1] {
                            'n' => '\n',
                            't' => '\t',
                            other => other,
                        });
                        j += 2;
                        continue;
                    }
                    if long {
                        if j + 2 < c.len() && c[j] == q && c[j + 1] == q && c[j + 2] == q {
                            j += 3;
                            break;
                        }
                    } else if c[j] == q {
                        j += 1;
                        break;
                    }
                    v.push(c[j]);
                    j += 1;
                }
                let mut lang = None;
                let mut dt = None;
                if j < c.len() && c[j] == '@' {
                    let k = (j + 1..c.len()).find(|&k| !(c[k].is_alphanumeric() || c[k] == '-')).unwrap_or(c.len());
                    lang = Some(c[j + 1..k].iter().collect());
                    j = k;
                } else if j + 1 < c.len() && c[j] == '^' && c[j + 1] == '^' {
                    j += 2;
                    if j < c.len() && c[j] == '<' {
                        let k = (j + 1..c.len()).find(|&k| c[k] == '>').ok_or_else(|| miette::miette!("turtle: unclosed IRI"))?;
                        dt = Some(c[j + 1..k].iter().collect());
                        j = k + 1;
                    } else {
                        let k = (j..c.len()).find(|&k| !name_char(c[k])).unwrap_or(c.len());
                        dt = Some(c[j..k].iter().collect());
                        j = k;
                    }
                }
                out.push(Tok::Lit { v, lang, dt });
                i = j;
            }
            '.' => {
                out.push(Tok::Dot);
                i += 1;
            }
            ';' => {
                out.push(Tok::Semi);
                i += 1;
            }
            ',' => {
                out.push(Tok::Comma);
                i += 1;
            }
            '[' => {
                out.push(Tok::LBracket);
                i += 1;
            }
            ']' => {
                out.push(Tok::RBracket);
                i += 1;
            }
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            '@' => {
                let k = (i + 1..c.len()).find(|&k| !c[k].is_alphanumeric()).unwrap_or(c.len());
                let w: String = c[i + 1..k].iter().collect();
                out.push(match w.as_str() {
                    "prefix" => Tok::Prefix,
                    "base" => Tok::Base,
                    other => bail!("turtle: unknown directive @{other}"),
                });
                i = k;
            }
            _ => {
                let mut k = i;
                while k < c.len() && name_char(c[k]) {
                    k += 1;
                }
                // a trailing dot ends the statement, not the name
                while k > i && c[k - 1] == '.' {
                    k -= 1;
                }
                if k == i {
                    bail!("turtle: unexpected character '{}'", ch);
                }
                let w: String = c[i..k].iter().collect();
                i = k;
                out.push(match w.as_str() {
                    "a" => Tok::A,
                    "true" => Tok::Bool(true),
                    "false" => Tok::Bool(false),
                    _ if w.eq_ignore_ascii_case("prefix") => Tok::Prefix,
                    _ if w.eq_ignore_ascii_case("base") => Tok::Base,
                    _ if w.starts_with("_:") => Tok::Blank(w),
                    _ if w.chars().next().is_some_and(|x| x.is_ascii_digit() || x == '-' || x == '+') => Tok::Num(w),
                    _ => Tok::Pname(w),
                });
            }
        }
    }
    Ok(out)
}

/// A term after prefix expansion.
#[derive(Debug, Clone, PartialEq)]
enum Term {
    Iri(String),
    Blank(String),
    Lit { v: String, lang: Option<String> },
    List(Vec<Term>),
}

struct Parser {
    toks: Vec<Tok>,
    i: usize,
    prefixes: BTreeMap<String, String>,
    triples: Vec<(Term, String, Term)>,
    blank: usize,
}

const RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
const RDFS: &str = "http://www.w3.org/2000/01/rdf-schema#";
const OWL: &str = "http://www.w3.org/2002/07/owl#";
const XSD: &str = "http://www.w3.org/2001/XMLSchema#";
const SKOS: &str = "http://www.w3.org/2004/02/skos/core#";

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.i)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.i).cloned();
        self.i += 1;
        t
    }
    fn expand(&self, pname: &str) -> Result<String> {
        let (p, local) = pname.split_once(':').ok_or_else(|| miette::miette!("turtle: '{pname}' is not a prefixed name"))?;
        match self.prefixes.get(p) {
            Some(base) => Ok(format!("{base}{local}")),
            None => bail!("turtle: undeclared prefix '{p}:'"),
        }
    }
    fn term(&mut self) -> Result<Term> {
        match self.next() {
            Some(Tok::Iri(x)) => Ok(Term::Iri(x)),
            Some(Tok::Pname(x)) => Ok(Term::Iri(self.expand(&x)?)),
            Some(Tok::A) => Ok(Term::Iri(format!("{RDF}type"))),
            Some(Tok::Blank(x)) => Ok(Term::Blank(x)),
            Some(Tok::Lit { v, lang, .. }) => Ok(Term::Lit { v, lang }),
            Some(Tok::Num(x)) => Ok(Term::Lit { v: x, lang: None }),
            Some(Tok::Bool(b)) => Ok(Term::Lit { v: b.to_string(), lang: None }),
            Some(Tok::LBracket) => {
                self.blank += 1;
                let b = Term::Blank(format!("_:b{}", self.blank));
                if self.peek() != Some(&Tok::RBracket) {
                    self.pred_list(&b)?;
                }
                match self.next() {
                    Some(Tok::RBracket) => Ok(b),
                    t => bail!("turtle: expected ']', got {t:?}"),
                }
            }
            Some(Tok::LParen) => {
                let mut items = vec![];
                while self.peek() != Some(&Tok::RParen) {
                    if self.peek().is_none() {
                        bail!("turtle: unclosed collection");
                    }
                    items.push(self.term()?);
                }
                self.next();
                Ok(Term::List(items))
            }
            t => bail!("turtle: expected a term, got {t:?}"),
        }
    }
    fn pred_list(&mut self, subj: &Term) -> Result<()> {
        loop {
            let p = match self.term()? {
                Term::Iri(x) => x,
                t => bail!("turtle: a predicate must be an IRI, got {t:?}"),
            };
            loop {
                let o = self.term()?;
                self.triples.push((subj.clone(), p.clone(), o));
                if self.peek() == Some(&Tok::Comma) {
                    self.next();
                    continue;
                }
                break;
            }
            if self.peek() == Some(&Tok::Semi) {
                while self.peek() == Some(&Tok::Semi) {
                    self.next();
                }
                if matches!(self.peek(), Some(Tok::Dot) | Some(Tok::RBracket) | None) {
                    return Ok(());
                }
                continue;
            }
            return Ok(());
        }
    }
    fn parse(&mut self) -> Result<()> {
        while let Some(t) = self.peek().cloned() {
            match t {
                Tok::Prefix => {
                    self.next();
                    let p = match self.next() {
                        Some(Tok::Pname(x)) => x.trim_end_matches(':').to_string(),
                        t => bail!("turtle: expected a prefix name, got {t:?}"),
                    };
                    let iri = match self.next() {
                        Some(Tok::Iri(x)) => x,
                        t => bail!("turtle: expected a prefix IRI, got {t:?}"),
                    };
                    self.prefixes.insert(p, iri);
                    if self.peek() == Some(&Tok::Dot) {
                        self.next();
                    }
                }
                Tok::Base => {
                    self.next();
                    self.next();
                    if self.peek() == Some(&Tok::Dot) {
                        self.next();
                    }
                }
                _ => {
                    let s = self.term()?;
                    if self.peek() != Some(&Tok::Dot) {
                        self.pred_list(&s)?;
                    }
                    match self.next() {
                        Some(Tok::Dot) => {}
                        t => bail!("turtle: expected '.', got {t:?}"),
                    }
                }
            }
        }
        Ok(())
    }
}

/// The local name of an IRI: after `#`, else after the last `/`.
fn local(iri: &str) -> String {
    iri.rsplit(['#', '/']).next().unwrap_or(iri).to_string()
}

/// Read the OWL/RDF subset into an ontology named `onto` (default: the
/// `owl:Ontology` subject's local name). Returns the document and warnings.
pub fn import_turtle(src: &str, onto: &str) -> Result<(OntologyDoc, Vec<String>)> {
    let mut p = Parser { toks: tokenize(src)?, i: 0, prefixes: BTreeMap::new(), triples: vec![], blank: 0 };
    p.parse()?;
    let ty = format!("{RDF}type");
    let mut types: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut by_subj: BTreeMap<String, Vec<(String, Term)>> = BTreeMap::new();
    let mut ontology_iri = None;
    let mut warn: BTreeMap<&str, usize> = BTreeMap::new();
    for (s, pr, o) in &p.triples {
        let Term::Iri(si) = s else {
            *warn.entry("blank-node subjects (restrictions, unions) skipped").or_default() += 1;
            continue;
        };
        if *pr == ty {
            if let Term::Iri(t) = o {
                types.entry(si.clone()).or_default().insert(t.clone());
                if *t == format!("{OWL}Ontology") {
                    ontology_iri = Some(si.clone());
                }
            }
        } else {
            by_subj.entry(si.clone()).or_default().push((pr.clone(), o.clone()));
        }
    }
    let is = |s: &str, t: &str| types.get(s).is_some_and(|x| x.contains(t));
    let iri_local = |t: &Term| -> Option<String> {
        match t {
            Term::Iri(x) => Some(local(x)),
            _ => None,
        }
    };
    let pick_label = |vals: &[(Option<String>, String)]| -> String {
        vals.iter()
            .find(|(l, _)| l.as_deref().is_some_and(|x| x.starts_with("en")))
            .or_else(|| vals.iter().find(|(l, _)| l.is_none()))
            .or(vals.first())
            .map(|x| x.1.clone())
            .unwrap_or_default()
    };
    let mut doc = OntologyDoc {
        onto: if onto.is_empty() { ontology_iri.as_deref().map(local).unwrap_or_else(|| "imported".into()) } else { onto.to_string() },
        version: String::new(),
        doc: String::new(),
        source: "turtle".into(),
        classes: vec![],
        properties: vec![],
    };
    if let Some(oi) = &ontology_iri {
        for (pr, o) in by_subj.get(oi).cloned().unwrap_or_default() {
            if let Term::Lit { v, .. } = o {
                if pr == format!("{OWL}versionInfo") {
                    doc.version = v;
                } else if pr == format!("{RDFS}comment") || pr == format!("{SKOS}definition") || pr.ends_with("/description") {
                    doc.doc = v;
                }
            }
        }
    }
    let class_types = [format!("{OWL}Class"), format!("{RDFS}Class")];
    let prop_types = [
        format!("{OWL}ObjectProperty"),
        format!("{OWL}DatatypeProperty"),
        format!("{RDF}Property"),
        format!("{OWL}FunctionalProperty"),
        format!("{OWL}SymmetricProperty"),
        format!("{OWL}TransitiveProperty"),
    ];
    let subjects: BTreeSet<String> = types.keys().cloned().chain(by_subj.keys().cloned()).collect();
    let mut inverse_pairs: Vec<(String, String)> = vec![];
    for s in &subjects {
        let preds = by_subj.get(s).cloned().unwrap_or_default();
        let lits = |names: &[String]| -> Vec<(Option<String>, String)> {
            preds
                .iter()
                .filter(|(p, _)| names.contains(p))
                .filter_map(|(_, o)| match o {
                    Term::Lit { v, lang } => Some((lang.clone(), v.clone())),
                    _ => None,
                })
                .collect()
        };
        let objs = |name: &str| -> Vec<Term> { preds.iter().filter(|(p, _)| p == name).map(|(_, o)| o.clone()).collect() };
        let label = pick_label(&lits(&[format!("{RDFS}label"), format!("{SKOS}prefLabel")]));
        let docs = pick_label(&lits(&[format!("{RDFS}comment"), format!("{SKOS}definition")]));
        let aliases: Vec<String> = lits(&[format!("{SKOS}altLabel"), format!("{SKOS}hiddenLabel")])
            .into_iter()
            .filter(|(l, _)| l.as_deref().map_or(true, |x| x.starts_with("en")))
            .map(|x| x.1)
            .collect();
        let is_class = class_types.iter().any(|t| is(s, t));
        let is_prop = prop_types.iter().any(|t| is(s, t));
        if is_class && !is_prop {
            let mut parents = vec![];
            for o in objs(&format!("{RDFS}subClassOf")) {
                match iri_local(&o) {
                    Some(x) if x != "Thing" => parents.push(x),
                    Some(_) => {}
                    None => *warn.entry("class axioms on blank nodes (restrictions) skipped").or_default() += 1,
                }
            }
            doc.classes.push(OntoClass { name: local(s), parents, label, doc: docs, aliases });
        } else if is_prop {
            let data = is(s, &format!("{OWL}DatatypeProperty"));
            let mut prop = OntoProp {
                name: local(s),
                kind: if data { "data" } else { "object" }.into(),
                card: if is(s, &format!("{OWL}FunctionalProperty")) { "one" } else { "many" }.into(),
                symmetric: is(s, &format!("{OWL}SymmetricProperty")),
                transitive: is(s, &format!("{OWL}TransitiveProperty")),
                label,
                doc: docs,
                cues: aliases.iter().map(|a| a.to_lowercase()).collect(),
                aliases,
                ..Default::default()
            };
            for o in objs(&format!("{RDFS}domain")) {
                if let Some(x) = iri_local(&o) {
                    prop.domain.push(x);
                }
            }
            for o in objs(&format!("{RDFS}range")) {
                if let Term::Iri(x) = &o {
                    if x.starts_with(XSD) {
                        prop.datatype = Some(format!("xsd:{}", local(x)));
                        prop.kind = "data".into();
                    } else if x == &format!("{RDFS}Literal") {
                        prop.kind = "data".into();
                    } else {
                        prop.range.push(local(x));
                    }
                }
            }
            for o in objs(&format!("{RDFS}subPropertyOf")) {
                if let Some(x) = iri_local(&o) {
                    if x != "topObjectProperty" && x != "topDataProperty" {
                        prop.parents.push(x);
                    }
                }
            }
            for o in objs(&format!("{OWL}inverseOf")) {
                if let Some(x) = iri_local(&o) {
                    prop.inverse = Some(x.clone());
                    inverse_pairs.push((x, prop.name.clone()));
                }
            }
            for o in objs(&format!("{OWL}propertyChainAxiom")) {
                if let Term::List(items) = o {
                    prop.chain = items.iter().filter_map(iri_local).collect();
                }
            }
            if !prop.label.is_empty() && !prop.cues.contains(&prop.label.to_lowercase()) {
                prop.cues.push(prop.label.to_lowercase());
            }
            doc.properties.push(prop);
        } else if !types.contains_key(s) || types[s].iter().all(|t| t == &format!("{OWL}Ontology")) {
            // annotations of the ontology itself, or of terms defined elsewhere
        } else {
            *warn.entry("individuals and other typed resources skipped").or_default() += 1;
        }
    }
    // inverseOf holds both ways
    for (a, b) in inverse_pairs {
        if let Some(p) = doc.properties.iter_mut().find(|p| p.name == a) {
            if p.inverse.is_none() {
                p.inverse = Some(b);
            }
        }
    }
    let warnings = warn.into_iter().map(|(k, n)| format!("{n} {k}")).collect();
    Ok((doc, warnings))
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

/// Write `doc` as Turtle (the same subset).
pub fn export_turtle(doc: &OntologyDoc) -> String {
    let base = format!("urn:kleos:onto:{}#", doc.onto);
    let mut out = format!(
        "@prefix : <{base}> .\n@prefix rdf: <{RDF}> .\n@prefix rdfs: <{RDFS}> .\n@prefix owl: <{OWL}> .\n\
         @prefix xsd: <{XSD}> .\n@prefix skos: <{SKOS}> .\n\n<urn:kleos:onto:{}> a owl:Ontology",
        doc.onto
    );
    if !doc.version.is_empty() {
        out.push_str(&format!(" ;\n  owl:versionInfo \"{}\"", esc(&doc.version)));
    }
    if !doc.doc.is_empty() {
        out.push_str(&format!(" ;\n  rdfs:comment \"{}\"", esc(&doc.doc)));
    }
    out.push_str(" .\n\n");
    let ann = |out: &mut String, label: &str, d: &str, aliases: &[String]| {
        if !label.is_empty() {
            out.push_str(&format!(" ;\n  rdfs:label \"{}\"@en", esc(label)));
        }
        if !d.is_empty() {
            out.push_str(&format!(" ;\n  rdfs:comment \"{}\"@en", esc(d)));
        }
        for a in aliases {
            out.push_str(&format!(" ;\n  skos:altLabel \"{}\"@en", esc(a)));
        }
    };
    for c in &doc.classes {
        out.push_str(&format!(":{} a owl:Class", c.name));
        for p in &c.parents {
            out.push_str(&format!(" ;\n  rdfs:subClassOf :{p}"));
        }
        ann(&mut out, &c.label, &c.doc, &c.aliases);
        out.push_str(" .\n\n");
    }
    for p in &doc.properties {
        let mut tys = vec![if p.kind == "object" { "owl:ObjectProperty" } else { "owl:DatatypeProperty" }];
        if p.single() {
            tys.push("owl:FunctionalProperty");
        }
        if p.symmetric {
            tys.push("owl:SymmetricProperty");
        }
        if p.transitive {
            tys.push("owl:TransitiveProperty");
        }
        out.push_str(&format!(":{} a {}", p.name, tys.join(", ")));
        for d in &p.domain {
            out.push_str(&format!(" ;\n  rdfs:domain :{d}"));
        }
        for r in &p.range {
            out.push_str(&format!(" ;\n  rdfs:range :{r}"));
        }
        if let Some(dt) = &p.datatype {
            out.push_str(&format!(" ;\n  rdfs:range {dt}"));
        }
        for q in &p.parents {
            out.push_str(&format!(" ;\n  rdfs:subPropertyOf :{q}"));
        }
        if let Some(i) = &p.inverse {
            out.push_str(&format!(" ;\n  owl:inverseOf :{i}"));
        }
        if !p.chain.is_empty() {
            out.push_str(&format!(" ;\n  owl:propertyChainAxiom ( {} )", p.chain.iter().map(|x| format!(":{x}")).collect::<Vec<_>>().join(" ")));
        }
        ann(&mut out, &p.label, &p.doc, &p.aliases);
        out.push_str(" .\n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRADING: &str = r#"
@prefix : <https://acme.example/onto/trading#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

<https://acme.example/onto/trading> a owl:Ontology ;
    owl:versionInfo "2026-10" ;
    rdfs:comment "Acme desk ontology" .

:Event a owl:Class .
:Trade a owl:Class ; rdfs:subClassOf :Event ;
    rdfs:label "trade"@en ; skos:altLabel "fill"@en, "execution"@en ;
    rdfs:comment "an executed trade"@en .
:Instrument a owl:Class ; rdfs:label "instrument"@en .
:Equity a owl:Class ; rdfs:subClassOf :Instrument , [ a owl:Restriction ; owl:onProperty :issuer ; owl:someValuesFrom :Organization ] .
:Organization a owl:Class .

:instrument a owl:ObjectProperty, owl:FunctionalProperty ;
    rdfs:domain :Trade ; rdfs:range :Instrument ; rdfs:label "instrument"@en .
:price a owl:DatatypeProperty , owl:FunctionalProperty ;
    rdfs:domain :Trade ; rdfs:range xsd:decimal ; skos:altLabel "px"@en .
:isSubsidiaryOf a owl:ObjectProperty, owl:TransitiveProperty ; rdfs:domain :Organization ; rdfs:range :Organization .
:hasSubsidiary a owl:ObjectProperty ; owl:inverseOf :isSubsidiaryOf .
:counterpartyOf a owl:ObjectProperty, owl:SymmetricProperty .
:ultimateParent a owl:ObjectProperty ; owl:propertyChainAxiom ( :isSubsidiaryOf :isSubsidiaryOf ) .
:aapl a :Equity .
"#;

    #[test]
    fn reads_the_subset() {
        let (d, warn) = import_turtle(TRADING, "").unwrap();
        assert_eq!(d.onto, "trading");
        assert_eq!(d.version, "2026-10");
        let trade = d.classes.iter().find(|c| c.name == "Trade").unwrap();
        assert_eq!(trade.parents, vec!["Event"]);
        assert_eq!(trade.aliases, vec!["fill", "execution"]);
        let eq = d.classes.iter().find(|c| c.name == "Equity").unwrap();
        assert_eq!(eq.parents, vec!["Instrument"]);
        let price = d.properties.iter().find(|p| p.name == "price").unwrap();
        assert_eq!(price.kind, "data");
        assert_eq!(price.card, "one");
        assert_eq!(price.datatype.as_deref(), Some("xsd:decimal"));
        assert!(price.cues.contains(&"px".to_string()));
        let sub = d.properties.iter().find(|p| p.name == "isSubsidiaryOf").unwrap();
        assert!(sub.transitive);
        assert_eq!(sub.inverse.as_deref(), Some("hasSubsidiary"));
        assert!(d.properties.iter().find(|p| p.name == "counterpartyOf").unwrap().symmetric);
        assert_eq!(d.properties.iter().find(|p| p.name == "ultimateParent").unwrap().chain, vec!["isSubsidiaryOf", "isSubsidiaryOf"]);
        assert!(warn.iter().any(|w| w.contains("restrictions") || w.contains("blank")), "{warn:?}");
        assert!(warn.iter().any(|w| w.contains("individuals")), "{warn:?}");
    }

    #[test]
    fn round_trip() {
        let (d, _) = import_turtle(TRADING, "acme").unwrap();
        let ttl = export_turtle(&d);
        let (d2, _) = import_turtle(&ttl, "acme").unwrap();
        let names = |d: &OntologyDoc| {
            let mut v: Vec<String> = d.properties.iter().map(|p| format!("{}:{}:{}", p.name, p.card, p.kind)).collect();
            v.sort();
            v
        };
        assert_eq!(names(&d), names(&d2));
        assert_eq!(d.classes.len(), d2.classes.len());
    }
}
