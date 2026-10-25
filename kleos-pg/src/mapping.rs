/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Zero-maintenance mapping (docs/specs/pg-connector.md, "Mapping"): W3C
//! Direct Mapping style inference from the catalog, YAML overrides on top,
//! and the ontology (`pg:<source>`) the mapping defines.
//!
//! - a table with a key is a class; each row an entity `pg:<source>/<schema.table>/<key>`;
//! - a scalar column is a single-valued, temporal data property;
//! - a foreign key is a single-valued object property (an edge);
//! - a table that is two foreign keys (plus attributes) is a many-valued edge;
//! - a long text column is a memory of its own (embedded, FTS);
//! - a table without a key is skipped (Postgres cannot replicate its updates either).

use std::collections::{BTreeMap, BTreeSet};

use kleos::ontology::{OntoClass, OntoProp, OntologyDoc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::catalog::{Catalog, TableInfo, TypeKind};
use crate::config::{ColumnOverride, PgConfig};

/// Column names that suggest prose when no length sample is available.
const PROSE_NAMES: &[&str] = &[
    "description", "body", "notes", "note", "comment", "comments", "content", "summary", "bio", "message", "text",
    "details", "transcript", "abstract",
];
const LABEL_NAMES: &[&str] = &[
    "name", "title", "full_name", "display_name", "username", "subject", "label", "email", "code", "sku", "slug",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColKind {
    /// A typed data fact.
    Fact,
    /// Part of the key only (a surrogate id): no fact.
    Key,
    /// Covered by a foreign-key edge.
    Ref,
    /// Its own memory.
    Text,
    /// An object fact to a node per value.
    Category,
    /// Not stored anywhere (not even in the row image).
    Drop,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColMap {
    pub name: String,
    pub kind: ColKind,
    pub predicate: String,
    pub tk: TypeKind,
    pub pii: bool,
    #[serde(default)]
    pub doc: Option<String>,
    pub type_name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FkMap {
    pub name: String,
    pub cols: Vec<String>,
    /// Referenced `schema.table`.
    pub table: String,
    pub predicate: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EndMap {
    pub cols: Vec<String>,
    pub table: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum Shape {
    Entity,
    Edge { source: EndMap, destination: EndMap, predicate: String },
    Skip { reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableMap {
    pub table: String,
    pub class: String,
    #[serde(default)]
    pub doc: Option<String>,
    #[serde(flatten)]
    pub shape: Shape,
    pub key: Vec<String>,
    #[serde(default)]
    pub label: Option<String>,
    pub row_memory: bool,
    #[serde(default)]
    pub row_template: Option<String>,
    pub columns: Vec<ColMap>,
    #[serde(default)]
    pub fks: Vec<FkMap>,
}

impl TableMap {
    pub fn col(&self, name: &str) -> Option<&ColMap> {
        self.columns.iter().find(|c| c.name == name)
    }
    pub fn is_entity(&self) -> bool {
        matches!(self.shape, Shape::Entity)
    }
    pub fn is_skip(&self) -> bool {
        matches!(self.shape, Shape::Skip { .. })
    }
    /// Columns kept in the row image (everything but `drop`).
    pub fn kept(&self, col: &str) -> bool {
        self.col(col).map_or(false, |c| c.kind != ColKind::Drop)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Mapping {
    pub source: String,
    pub tables: BTreeMap<String, TableMap>,
}

/// `order_items` → `OrderItem`, `people` → `Person`.
pub fn class_name(table: &str) -> String {
    let words: Vec<&str> = table.split(|c: char| c == '_' || c == '-' || c == ' ').filter(|w| !w.is_empty()).collect();
    let n = words.len();
    words
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let w = if i + 1 == n { singular(w) } else { w.to_string() };
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str().to_lowercase().as_str(),
                None => String::new(),
            }
        })
        .collect()
}

pub fn singular(w: &str) -> String {
    let l = w.to_lowercase();
    let irregular = [("people", "person"), ("children", "child"), ("men", "man"), ("women", "woman"), ("data", "data"), ("media", "media")];
    if let Some((_, s)) = irregular.iter().find(|(p, _)| *p == l) {
        return s.to_string();
    }
    for suffix in ["ss", "us", "is", "news"] {
        if l.ends_with(suffix) {
            return w.to_string();
        }
    }
    if l.ends_with("ies") && l.len() > 3 {
        return format!("{}y", &w[..w.len() - 3]);
    }
    for suffix in ["sses", "xes", "ches", "shes", "zes"] {
        if l.ends_with(suffix) {
            return w[..w.len() - 2].to_string();
        }
    }
    if l.ends_with("ses") && l.len() > 4 {
        return w[..w.len() - 1].to_string();
    }
    if l.ends_with('s') && l.len() > 1 {
        return w[..w.len() - 1].to_string();
    }
    w.to_string()
}

/// `account_manager_id` → `account manager`.
pub fn humanize(s: &str) -> String {
    let s = s.strip_suffix("_id").unwrap_or(s);
    s.replace(['_', '-'], " ").to_lowercase()
}

/// The predicate prefix of a table: its name in `public`, else `schema.table`.
fn short(table: &str) -> String {
    match table.split_once('.') {
        Some(("public", t)) => t.to_string(),
        _ => table.to_string(),
    }
}

fn fk_word(cols: &[String], constraint: &str) -> String {
    if cols.len() == 1 {
        let c = &cols[0];
        for suf in ["_id", "_fk", "_key", "_uuid", "id"] {
            if let Some(x) = c.strip_suffix(suf) {
                if !x.is_empty() {
                    return x.trim_end_matches('_').to_string();
                }
            }
        }
        return c.clone();
    }
    constraint.to_string()
}

fn parse_end(s: &str) -> Option<EndMap> {
    let (cols, table) = s.split_once("->")?;
    let table = table.trim();
    let table = if table.contains('.') { table.to_string() } else { format!("public.{table}") };
    Some(EndMap { cols: cols.split(',').map(|c| c.trim().to_string()).filter(|c| !c.is_empty()).collect(), table })
}

/// Infer the mapping of every included table of `cat`, with `cfg`'s
/// overrides applied.
pub fn infer(cfg: &PgConfig, cat: &Catalog) -> Mapping {
    let mut m = Mapping { source: cfg.source.clone(), tables: BTreeMap::new() };
    for (q, t) in &cat.tables {
        if !cfg.includes(q) {
            continue;
        }
        m.tables.insert(q.clone(), infer_table(cfg, cat, t));
    }
    // class names unique: prefix the schema where two tables collide
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for t in m.tables.values() {
        *seen.entry(t.class.clone()).or_default() += 1;
    }
    for t in m.tables.values_mut() {
        if seen.get(&t.class).copied().unwrap_or(0) > 1 && cfg.table_override(&t.table).and_then(|o| o.class.clone()).is_none() {
            let schema = t.table.split('.').next().unwrap_or("");
            t.class = format!("{}{}", class_name(schema), t.class);
        }
    }
    m
}

/// Infer one table.
pub fn infer_table(cfg: &PgConfig, cat: &Catalog, t: &TableInfo) -> TableMap {
    let q = t.qualified();
    let ov = cfg.table_override(&q);
    let graph = cat.graph.get(&q);
    let class = ov
        .and_then(|o| o.class.clone())
        .or_else(|| graph.and_then(|g| g.label.as_deref()).map(class_name))
        .unwrap_or_else(|| class_name(&t.name));
    let key: Vec<String> = ov
        .and_then(|o| o.key.clone())
        .or_else(|| graph.filter(|g| !g.key.is_empty()).map(|g| g.key.clone()))
        .unwrap_or_else(|| t.pk.clone());
    let pre = short(&q);
    let col_ov = |c: &str| ov.and_then(|o| o.columns.get(c));
    let mut columns = vec![];
    let fk_cols: BTreeSet<&str> = t.fks.iter().flat_map(|f| f.cols.iter().map(|c| c.as_str())).collect();
    let text_cols_ov: Option<Vec<String>> = ov.and_then(|o| o.memory.as_ref()).and_then(|m| m.text.clone());
    for c in &t.columns {
        let mut pii = false;
        let mut predicate = format!("{pre}.{}", c.name);
        let mut forced: Option<ColKind> = None;
        if let Some(o) = col_ov(&c.name) {
            let (kind, p, pi) = match o {
                ColumnOverride::Word(w) => (Some(w.clone()), None, None),
                ColumnOverride::Map(m) => (m.kind.clone(), m.predicate.clone(), m.pii),
            };
            if let Some(p) = p {
                predicate = p;
            }
            if pi == Some(true) {
                pii = true;
            }
            forced = match kind.as_deref() {
                Some("drop") => Some(ColKind::Drop),
                Some("pii") => {
                    pii = true;
                    None
                }
                Some("text") => Some(ColKind::Text),
                Some("fact") => Some(ColKind::Fact),
                Some("category") => Some(ColKind::Category),
                _ => None,
            };
        }
        if let Some(tc) = &text_cols_ov {
            if tc.contains(&c.name) && forced.is_none() {
                forced = Some(ColKind::Text);
            }
        }
        let kind = forced.unwrap_or_else(|| {
            let in_key = key.contains(&c.name);
            if fk_cols.contains(c.name.as_str()) {
                ColKind::Ref
            } else if in_key {
                // a surrogate key is in the entity id; a natural key is also a fact
                if matches!(c.kind, TypeKind::Int | TypeKind::Uuid) {
                    ColKind::Key
                } else {
                    ColKind::Fact
                }
            } else if c.kind == TypeKind::Bytea {
                ColKind::Drop
            } else if c.kind.is_texty()
                && c.enum_labels.is_empty()
                && match c.avg_len {
                    Some(l) => l >= cfg.memory.text_min_chars as f64,
                    None => PROSE_NAMES.contains(&c.name.as_str()) && c.type_name == "text",
                }
            {
                ColKind::Text
            } else {
                ColKind::Fact
            }
        });
        let mut doc = c.comment.clone();
        if !c.enum_labels.is_empty() {
            let e = format!("one of: {}", c.enum_labels.join(", "));
            doc = Some(match doc {
                Some(d) => format!("{d} ({e})"),
                None => e,
            });
        }
        columns.push(ColMap { name: c.name.clone(), kind, predicate, tk: c.kind, pii, doc, type_name: c.type_name.clone() });
    }
    let mut fks: Vec<FkMap> = t
        .fks
        .iter()
        .map(|f| FkMap {
            name: f.name.clone(),
            cols: f.cols.clone(),
            table: f.ref_table.clone(),
            predicate: format!("{pre}.{}", fk_word(&f.cols, &f.name)),
        })
        .collect();
    // overrides naming references for sources without foreign keys
    if let Some(o) = ov {
        for (c, co) in &o.columns {
            if let ColumnOverride::Map(cm) = co {
                if let Some(r) = &cm.references {
                    let table = if r.contains('.') { r.clone() } else { format!("public.{r}") };
                    fks.retain(|f| f.cols != vec![c.clone()]);
                    fks.push(FkMap { name: format!("{c}_ref"), cols: vec![c.clone()], table, predicate: format!("{pre}.{}", fk_word(&[c.clone()], c)) });
                    if let Some(col) = columns.iter_mut().find(|x| &x.name == c) {
                        col.kind = ColKind::Ref;
                    }
                }
            }
        }
    }
    // the shape
    let shape = if let Some(e) = ov.and_then(|o| o.edge.as_ref()) {
        match (parse_end(&e.source), parse_end(&e.destination)) {
            (Some(s), Some(d)) => Shape::Edge { source: s, destination: d, predicate: e.label.clone().unwrap_or_else(|| pre.clone()) },
            _ => Shape::Skip { reason: "edge override: use 'column -> schema.table'".into() },
        }
    } else if let Some(g) = graph.filter(|g| g.source.is_some() && g.destination.is_some()) {
        let (sc, st) = g.source.clone().unwrap();
        let (dc, dt) = g.destination.clone().unwrap();
        Shape::Edge {
            source: EndMap { cols: sc, table: st },
            destination: EndMap { cols: dc, table: dt },
            predicate: g.label.clone().unwrap_or_else(|| pre.clone()),
        }
    } else if key.is_empty() {
        Shape::Skip { reason: "no primary key (Postgres cannot replicate its updates and deletes by key either; set `key` to map it)".into() }
    } else if let Some((a, b)) = join_fks(t, &key) {
        Shape::Edge {
            source: EndMap { cols: a.cols.clone(), table: a.ref_table.clone() },
            destination: EndMap { cols: b.cols.clone(), table: b.ref_table.clone() },
            predicate: pre.clone(),
        }
    } else {
        Shape::Entity
    };
    if let Shape::Edge { source, destination, .. } = &shape {
        // the ends are the edge; the remaining columns are its attributes
        for c in columns.iter_mut() {
            if source.cols.contains(&c.name) || destination.cols.contains(&c.name) {
                c.kind = ColKind::Ref;
            }
        }
        fks.retain(|f| f.cols != source.cols && f.cols != destination.cols);
    }
    let label = ov.and_then(|o| o.label.clone()).or_else(|| {
        let names: BTreeSet<&str> = t.columns.iter().map(|c| c.name.as_str()).collect();
        LABEL_NAMES
            .iter()
            .find(|n| names.contains(*n) && columns.iter().any(|c| &c.name == *n && c.kind != ColKind::Drop && !c.pii))
            .map(|n| format!("{{{n}}}"))
            .or_else(|| (names.contains("first_name") && names.contains("last_name")).then(|| "{first_name} {last_name}".to_string()))
    });
    let mem_ov = ov.and_then(|o| o.memory.as_ref());
    let row_memory = matches!(shape, Shape::Entity)
        && mem_ov.and_then(|m| m.row).unwrap_or(match cfg.memory.rows.as_str() {
            "false" => false,
            _ => true,
        });
    TableMap {
        table: q.clone(),
        class,
        doc: t.comment.clone(),
        shape,
        key,
        label,
        row_memory,
        row_template: mem_ov.and_then(|m| m.template.clone()),
        columns,
        fks,
    }
}

/// A pure join table: exactly two foreign keys whose columns are the key.
fn join_fks<'a>(t: &'a TableInfo, key: &[String]) -> Option<(&'a crate::catalog::FkInfo, &'a crate::catalog::FkInfo)> {
    if t.fks.len() != 2 {
        return None;
    }
    let (a, b) = (&t.fks[0], &t.fks[1]);
    let cols: BTreeSet<&String> = a.cols.iter().chain(b.cols.iter()).collect();
    let keys: BTreeSet<&String> = key.iter().collect();
    let covered = cols == keys || t.uniques.iter().any(|u| u.iter().collect::<BTreeSet<_>>() == cols);
    (covered && a.cols.iter().all(|c| !b.cols.contains(c))).then_some((a, b))
}

/// The ontology the mapping defines: a class per entity table, a property
/// per fact column, foreign key and edge table. Text columns are memories,
/// not properties.
pub fn ontology(cfg: &PgConfig, m: &Mapping) -> OntologyDoc {
    let mut classes = vec![];
    let mut props = vec![];
    let class_of: BTreeMap<&str, &str> = m.tables.iter().filter(|(_, t)| t.is_entity()).map(|(q, t)| (q.as_str(), t.class.as_str())).collect();
    let mut need_category = false;
    for t in m.tables.values() {
        match &t.shape {
            Shape::Skip { .. } => continue,
            Shape::Entity => {
                classes.push(OntoClass {
                    name: t.class.clone(),
                    parents: vec![],
                    label: humanize(&crate::mapping::snake(&t.class)),
                    doc: t.doc.clone().unwrap_or_else(|| format!("Rows of the Postgres table {} (source {}).", t.table, cfg.source)),
                    aliases: vec![t.table.split('.').nth(1).unwrap_or(&t.table).to_string()],
                });
                for c in &t.columns {
                    match c.kind {
                        ColKind::Fact => props.push(OntoProp {
                            name: c.predicate.clone(),
                            kind: "data".into(),
                            domain: vec![t.class.clone()],
                            datatype: Some(c.tk.xsd().to_string()),
                            card: "one".into(),
                            temporal: true,
                            label: humanize(&c.name),
                            doc: prop_doc(c, &t.table),
                            ..Default::default()
                        }),
                        ColKind::Category => {
                            need_category = true;
                            props.push(OntoProp {
                                name: c.predicate.clone(),
                                kind: "object".into(),
                                domain: vec![t.class.clone()],
                                range: vec!["Category".into()],
                                card: "one".into(),
                                temporal: true,
                                label: humanize(&c.name),
                                doc: prop_doc(c, &t.table),
                                ..Default::default()
                            })
                        }
                        _ => {}
                    }
                }
                for f in &t.fks {
                    props.push(OntoProp {
                        name: f.predicate.clone(),
                        kind: "object".into(),
                        domain: vec![t.class.clone()],
                        range: class_of.get(f.table.as_str()).map(|c| vec![c.to_string()]).unwrap_or_default(),
                        card: "one".into(),
                        temporal: true,
                        label: humanize(&fk_word(&f.cols, &f.name)),
                        doc: format!("Foreign key {} of {} ({} → {}).", f.name, t.table, f.cols.join(", "), f.table),
                        ..Default::default()
                    });
                }
            }
            Shape::Edge { source, destination, predicate } => props.push(OntoProp {
                name: predicate.clone(),
                kind: "object".into(),
                domain: class_of.get(source.table.as_str()).map(|c| vec![c.to_string()]).unwrap_or_default(),
                range: class_of.get(destination.table.as_str()).map(|c| vec![c.to_string()]).unwrap_or_default(),
                card: "many".into(),
                temporal: true,
                label: humanize(predicate.rsplit('.').next().unwrap_or(predicate)),
                doc: t.doc.clone().unwrap_or_else(|| format!("Rows of the join table {}.", t.table)),
                ..Default::default()
            }),
        }
    }
    if need_category {
        classes.push(OntoClass {
            name: "Category".into(),
            label: "category".into(),
            doc: "A value of a low-cardinality column, as a node.".into(),
            ..Default::default()
        });
    }
    OntologyDoc {
        onto: cfg.onto(),
        version: String::new(),
        doc: format!("Inferred from Postgres source '{}' by kleos-pg; maintained automatically.", cfg.source),
        source: format!("pg:{}", cfg.source),
        classes,
        properties: props,
    }
}

fn prop_doc(c: &ColMap, table: &str) -> String {
    let base = c.doc.clone().unwrap_or_else(|| format!("{table}.{} ({})", c.name, c.type_name));
    if c.pii {
        format!("PII. {base}")
    } else {
        base
    }
}

/// `OrderItem` → `order_item`.
pub fn snake(s: &str) -> String {
    let mut o = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            o.push('_');
        }
        o.extend(ch.to_lowercase());
    }
    o
}

/// One table's mapping change, in the shape of `kleos_onto_change`
/// (`before`/`after` property lists).
#[derive(Debug, Clone, Serialize)]
pub struct TableChange {
    pub table: String,
    pub kind: &'static str,
    pub before: Option<TableMap>,
    pub after: Option<TableMap>,
}

pub fn diff(old: &Mapping, new: &Mapping) -> Vec<TableChange> {
    let mut out = vec![];
    let names: BTreeSet<&String> = old.tables.keys().chain(new.tables.keys()).collect();
    for n in names {
        match (old.tables.get(n), new.tables.get(n)) {
            (None, Some(b)) => out.push(TableChange { table: n.clone(), kind: "added", before: None, after: Some(b.clone()) }),
            (Some(a), None) => out.push(TableChange { table: n.clone(), kind: "removed", before: Some(a.clone()), after: None }),
            (Some(a), Some(b)) if a != b => {
                out.push(TableChange { table: n.clone(), kind: "changed", before: Some(a.clone()), after: Some(b.clone()) })
            }
            _ => {}
        }
    }
    out
}

/// The ontology properties one table contributes (for change records).
pub fn table_props(cfg: &PgConfig, m: &Mapping, table: &str) -> Vec<OntoProp> {
    let doc = ontology(cfg, m);
    let Some(t) = m.tables.get(table) else { return vec![] };
    let mut names: BTreeSet<String> = t.columns.iter().map(|c| c.predicate.clone()).collect();
    names.extend(t.fks.iter().map(|f| f.predicate.clone()));
    if let Shape::Edge { predicate, .. } = &t.shape {
        names.insert(predicate.clone());
    }
    doc.properties.into_iter().filter(|p| names.contains(&p.name)).collect()
}

/// The label of a row from the table's template (`{col}` placeholders).
pub fn render_label(t: &TableMap, vals: &BTreeMap<String, Value>) -> Option<String> {
    let tmpl = t.label.as_ref()?;
    let s = fill(tmpl, vals);
    let s = s.trim().to_string();
    (!s.is_empty()).then_some(s)
}

pub fn fill(tmpl: &str, vals: &BTreeMap<String, Value>) -> String {
    let mut out = String::new();
    let mut rest = tmpl;
    while let Some(i) = rest.find('{') {
        out.push_str(&rest[..i]);
        match rest[i..].find('}') {
            Some(j) => {
                let name = &rest[i + 1..i + j];
                if let Some(v) = vals.get(name) {
                    out.push_str(&crate::value::display(v));
                }
                rest = &rest[i + j + 1..];
            }
            None => {
                out.push_str(&rest[i..]);
                rest = "";
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{ColumnInfo, FkInfo};

    fn col(name: &str, kind: TypeKind, avg: Option<f64>) -> ColumnInfo {
        ColumnInfo {
            name: name.into(),
            type_name: if kind == TypeKind::Text { "text".into() } else { "integer".into() },
            type_oid: 0,
            atttypid: 0,
            kind,
            not_null: false,
            comment: None,
            avg_len: avg,
            enum_labels: vec![],
        }
    }

    #[test]
    fn names() {
        assert_eq!(class_name("order_items"), "OrderItem");
        assert_eq!(class_name("people"), "Person");
        assert_eq!(class_name("categories"), "Category");
        assert_eq!(class_name("addresses"), "Address");
        assert_eq!(class_name("status"), "Status");
        assert_eq!(class_name("boxes"), "Box");
        assert_eq!(humanize("account_manager_id"), "account manager");
    }

    #[test]
    fn infers_entities_edges_and_text() {
        let cfg = PgConfig::parse("source: crm\nconnection: postgres://u@h/db\n").unwrap();
        let mut cat = Catalog::default();
        cat.tables.insert(
            "public.customers".into(),
            TableInfo {
                oid: 1,
                schema: "public".into(),
                name: "customers".into(),
                comment: None,
                columns: vec![col("id", TypeKind::Int, None), col("name", TypeKind::Text, Some(12.0)), col("notes", TypeKind::Text, Some(300.0))],
                pk: vec!["id".into()],
                uniques: vec![],
                fks: vec![],
                replica_identity: "d".into(),
                partitioned: false,
                rows_estimate: 0,
            },
        );
        cat.tables.insert(
            "public.customer_tags".into(),
            TableInfo {
                oid: 2,
                schema: "public".into(),
                name: "customer_tags".into(),
                comment: None,
                columns: vec![col("customer_id", TypeKind::Int, None), col("tag_id", TypeKind::Int, None)],
                pk: vec!["customer_id".into(), "tag_id".into()],
                uniques: vec![],
                fks: vec![
                    FkInfo { name: "ct_c".into(), cols: vec!["customer_id".into()], ref_table: "public.customers".into(), ref_cols: vec!["id".into()] },
                    FkInfo { name: "ct_t".into(), cols: vec!["tag_id".into()], ref_table: "public.tags".into(), ref_cols: vec!["id".into()] },
                ],
                replica_identity: "d".into(),
                partitioned: false,
                rows_estimate: 0,
            },
        );
        let m = infer(&cfg, &cat);
        let c = &m.tables["public.customers"];
        assert_eq!(c.class, "Customer");
        assert_eq!(c.col("id").unwrap().kind, ColKind::Key);
        assert_eq!(c.col("name").unwrap().kind, ColKind::Fact);
        assert_eq!(c.col("notes").unwrap().kind, ColKind::Text);
        assert_eq!(c.label.as_deref(), Some("{name}"));
        assert!(matches!(m.tables["public.customer_tags"].shape, Shape::Edge { .. }));
        let doc = ontology(&cfg, &m);
        assert!(doc.properties.iter().any(|p| p.name == "customers.name" && p.card == "one"));
        assert!(doc.properties.iter().any(|p| p.name == "customer_tags" && p.card == "many"));
    }
}
