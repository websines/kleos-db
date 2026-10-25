/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! What the mapping is inferred from: tables, columns, keys, foreign keys,
//! comments, types and sampled text lengths, read from `pg_catalog`; and,
//! on PG19+, `CREATE PROPERTY GRAPH` definitions. The Debezium adapter builds
//! the same structures from the events it sees.

use std::collections::BTreeMap;

use miette::Result;
use serde::{Deserialize, Serialize};
use tokio_postgres::Client;

use crate::conn::{lit, qident, simple_rows};

/// How a column's values are typed in kleos.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypeKind {
    Bool,
    Int,
    Float,
    Numeric,
    Text,
    Date,
    Timestamp,
    TimestampTz,
    Time,
    Interval,
    Uuid,
    Json,
    Array,
    Bytea,
    Other,
}

impl TypeKind {
    /// From a base type OID (domains resolved) and its category.
    pub fn of(oid: u32, category: &str) -> TypeKind {
        match oid {
            16 => TypeKind::Bool,
            20 | 21 | 23 | 26 => TypeKind::Int,
            700 | 701 => TypeKind::Float,
            1700 | 790 => TypeKind::Numeric,
            25 | 1043 | 1042 | 19 | 18 => TypeKind::Text,
            1082 => TypeKind::Date,
            1114 => TypeKind::Timestamp,
            1184 => TypeKind::TimestampTz,
            1083 | 1266 => TypeKind::Time,
            1186 => TypeKind::Interval,
            2950 => TypeKind::Uuid,
            114 | 3802 => TypeKind::Json,
            17 => TypeKind::Bytea,
            _ => match category {
                "A" => TypeKind::Array,
                "E" | "S" => TypeKind::Text,
                "B" => TypeKind::Bool,
                "N" => TypeKind::Numeric,
                _ => TypeKind::Other,
            },
        }
    }

    /// The XSD datatype of its literals.
    pub fn xsd(self) -> &'static str {
        match self {
            TypeKind::Bool => "xsd:boolean",
            TypeKind::Int => "xsd:integer",
            TypeKind::Float => "xsd:double",
            TypeKind::Numeric => "xsd:decimal",
            TypeKind::Date => "xsd:date",
            TypeKind::Timestamp | TypeKind::TimestampTz => "xsd:dateTime",
            TypeKind::Time => "xsd:time",
            TypeKind::Interval => "xsd:duration",
            TypeKind::Json | TypeKind::Array => "rdf:JSON",
            _ => "xsd:string",
        }
    }

    pub fn is_texty(self) -> bool {
        matches!(self, TypeKind::Text | TypeKind::Other)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub name: String,
    /// `format_type()`: `character varying(255)`, `public.plan_kind`.
    pub type_name: String,
    /// Base type OID (a domain's base type).
    pub type_oid: u32,
    /// The column's own type OID (what pgoutput's Relation message carries).
    #[serde(default)]
    pub atttypid: u32,
    pub kind: TypeKind,
    pub not_null: bool,
    #[serde(default)]
    pub comment: Option<String>,
    /// Sampled average length in bytes (text columns).
    #[serde(default)]
    pub avg_len: Option<f64>,
    /// Labels of an enum type.
    #[serde(default)]
    pub enum_labels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FkInfo {
    pub name: String,
    pub cols: Vec<String>,
    /// `schema.table`
    pub ref_table: String,
    pub ref_cols: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableInfo {
    pub oid: u32,
    pub schema: String,
    pub name: String,
    #[serde(default)]
    pub comment: Option<String>,
    pub columns: Vec<ColumnInfo>,
    pub pk: Vec<String>,
    #[serde(default)]
    pub uniques: Vec<Vec<String>>,
    #[serde(default)]
    pub fks: Vec<FkInfo>,
    /// `d` default, `f` full, `i` index, `n` nothing.
    #[serde(default)]
    pub replica_identity: String,
    #[serde(default)]
    pub partitioned: bool,
    #[serde(default)]
    pub rows_estimate: i64,
}

impl TableInfo {
    pub fn qualified(&self) -> String {
        format!("{}.{}", self.schema, self.name)
    }
    pub fn column(&self, name: &str) -> Option<&ColumnInfo> {
        self.columns.iter().find(|c| c.name == name)
    }
}

/// A vertex or edge table of a PG19 property graph.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphElement {
    pub table: String,
    pub label: Option<String>,
    pub key: Vec<String>,
    /// Edges: (key columns, referenced table) of each end.
    pub source: Option<(Vec<String>, String)>,
    pub destination: Option<(Vec<String>, String)>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Catalog {
    pub version: i32,
    pub tables: BTreeMap<String, TableInfo>,
    /// Property-graph elements by table (PG19+).
    #[serde(default)]
    pub graph: BTreeMap<String, GraphElement>,
}

fn parse_int_array(s: &str) -> Vec<i64> {
    s.trim_matches(|c| c == '{' || c == '}')
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect()
}

/// Read the catalog for tables `wanted(schema.table)` accepts. `sample`
/// samples text lengths (one `LIMIT 1000` read per table with text columns).
pub async fn read_catalog(client: &Client, version: i32, wanted: &(dyn Fn(&str) -> bool + Send + Sync), sample: bool) -> Result<Catalog> {
    let mut cat = Catalog { version, ..Default::default() };
    let tables = simple_rows(
        client,
        "SELECT c.oid, n.nspname, c.relname, c.relkind, c.relreplident, obj_description(c.oid, 'pg_class'), \
                greatest(c.reltuples, 0)::bigint \
         FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace \
         WHERE c.relkind IN ('r', 'p') AND NOT c.relispartition \
           AND n.nspname NOT IN ('pg_catalog', 'information_schema') AND n.nspname NOT LIKE 'pg_toast%' \
           AND n.nspname NOT LIKE 'pg_temp%'",
    )
    .await?;
    let mut by_oid: BTreeMap<u32, String> = BTreeMap::new();
    let mut all_names: BTreeMap<u32, String> = BTreeMap::new();
    for r in tables {
        let g = |i: usize| r[i].clone().unwrap_or_default();
        let oid: u32 = g(0).parse().unwrap_or(0);
        let q = format!("{}.{}", g(1), g(2));
        all_names.insert(oid, q.clone());
        if !wanted(&q) {
            continue;
        }
        by_oid.insert(oid, q.clone());
        cat.tables.insert(
            q,
            TableInfo {
                oid,
                schema: g(1),
                name: g(2),
                comment: r[5].clone(),
                columns: vec![],
                pk: vec![],
                uniques: vec![],
                fks: vec![],
                replica_identity: g(4),
                partitioned: g(3) == "p",
                rows_estimate: g(6).parse().unwrap_or(0),
            },
        );
    }
    if by_oid.is_empty() {
        return Ok(cat);
    }
    let oids = by_oid.keys().map(|o| o.to_string()).collect::<Vec<_>>().join(",");
    let cols = simple_rows(
        client,
        &format!(
            "SELECT a.attrelid, a.attnum, a.attname, format_type(a.atttypid, a.atttypmod), \
                    CASE WHEN t.typtype = 'd' THEN t.typbasetype ELSE a.atttypid END, \
                    coalesce(bt.typcategory, t.typcategory), a.attnotnull, col_description(a.attrelid, a.attnum), \
                    (SELECT string_agg(e.enumlabel, '\u{1f}' ORDER BY e.enumsortorder) FROM pg_enum e WHERE e.enumtypid = a.atttypid), \
                    a.atttypid \
             FROM pg_attribute a JOIN pg_type t ON t.oid = a.atttypid \
             LEFT JOIN pg_type bt ON bt.oid = t.typbasetype AND t.typtype = 'd' \
             WHERE a.attrelid IN ({oids}) AND a.attnum > 0 AND NOT a.attisdropped \
             ORDER BY a.attrelid, a.attnum"
        ),
    )
    .await?;
    let mut attname: BTreeMap<(u32, i64), String> = BTreeMap::new();
    for r in cols {
        let g = |i: usize| r[i].clone().unwrap_or_default();
        let oid: u32 = g(0).parse().unwrap_or(0);
        let Some(q) = by_oid.get(&oid) else { continue };
        attname.insert((oid, g(1).parse().unwrap_or(0)), g(2));
        let base: u32 = g(4).parse().unwrap_or(0);
        let col = ColumnInfo {
            name: g(2),
            type_name: g(3),
            type_oid: base,
            atttypid: g(9).parse().unwrap_or(0),
            kind: TypeKind::of(base, &g(5)),
            not_null: g(6) == "t",
            comment: r[7].clone(),
            avg_len: None,
            enum_labels: r[8].clone().map(|s| s.split('\u{1f}').map(|x| x.to_string()).collect()).unwrap_or_default(),
        };
        if let Some(t) = cat.tables.get_mut(q) {
            t.columns.push(col);
        }
    }
    // referenced tables may be outside `wanted`: their column names too
    let cons = simple_rows(
        client,
        &format!(
            "SELECT conrelid, conname, contype, conkey, confrelid, confkey FROM pg_constraint \
             WHERE conrelid IN ({oids}) AND contype IN ('p', 'u', 'f') ORDER BY conrelid, conname"
        ),
    )
    .await?;
    let ref_oids: Vec<String> = cons
        .iter()
        .filter_map(|r| r[4].clone())
        .filter(|o| o != "0" && !by_oid.contains_key(&o.parse().unwrap_or(0)))
        .collect();
    if !ref_oids.is_empty() {
        for r in simple_rows(
            client,
            &format!(
                "SELECT attrelid, attnum, attname FROM pg_attribute WHERE attrelid IN ({}) AND attnum > 0",
                ref_oids.join(",")
            ),
        )
        .await?
        {
            let g = |i: usize| r[i].clone().unwrap_or_default();
            attname.insert((g(0).parse().unwrap_or(0), g(1).parse().unwrap_or(0)), g(2));
        }
    }
    for r in cons {
        let g = |i: usize| r[i].clone().unwrap_or_default();
        let oid: u32 = g(0).parse().unwrap_or(0);
        let Some(q) = by_oid.get(&oid).cloned() else { continue };
        let names = |rel: u32, arr: &str| -> Vec<String> {
            parse_int_array(arr).into_iter().filter_map(|n| attname.get(&(rel, n)).cloned()).collect()
        };
        let cols = names(oid, &g(3));
        let t = cat.tables.get_mut(&q).unwrap();
        match g(2).as_str() {
            "p" => t.pk = cols,
            "u" => t.uniques.push(cols),
            "f" => {
                let roid: u32 = g(4).parse().unwrap_or(0);
                if let Some(rq) = all_names.get(&roid) {
                    t.fks.push(FkInfo { name: g(1), cols, ref_table: rq.clone(), ref_cols: names(roid, &g(5)) });
                }
            }
            _ => {}
        }
    }
    if sample {
        for t in cat.tables.values_mut() {
            let texty: Vec<String> = t.columns.iter().filter(|c| c.kind.is_texty() && c.enum_labels.is_empty()).map(|c| c.name.clone()).collect();
            if texty.is_empty() {
                continue;
            }
            let sel = texty
                .iter()
                .map(|c| format!("avg(octet_length({}::text))::float8", crate::conn::ident(c)))
                .collect::<Vec<_>>()
                .join(", ");
            let inner = texty.iter().map(|c| crate::conn::ident(c)).collect::<Vec<_>>().join(", ");
            let sql = format!("SELECT {sel} FROM (SELECT {inner} FROM {} LIMIT 1000) s", qident(&t.qualified()));
            match simple_rows(client, &sql).await {
                Ok(rows) => {
                    if let Some(r) = rows.first() {
                        for (i, c) in texty.iter().enumerate() {
                            let v = r.get(i).cloned().flatten().and_then(|x| x.parse::<f64>().ok());
                            if let Some(col) = t.columns.iter_mut().find(|x| &x.name == c) {
                                col.avg_len = v;
                            }
                        }
                    }
                }
                Err(e) => log::warn!("pg: sampling {} failed: {e}", t.qualified()),
            }
        }
    }
    if version >= 190000 {
        match read_property_graphs(client).await {
            Ok(g) => cat.graph = g,
            Err(e) => log::warn!("pg: reading property graphs failed (ignored): {e}"),
        }
    }
    Ok(cat)
}

/// Table names of a schema for publication management.
pub async fn table_exists(client: &Client, qualified: &str) -> Result<bool> {
    let rows = simple_rows(client, &format!("SELECT to_regclass({}) IS NOT NULL", lit(&qident(qualified)))).await?;
    Ok(rows.first().and_then(|r| r[0].clone()).as_deref() == Some("t"))
}

/// PG19 `CREATE PROPERTY GRAPH` definitions (`pg_get_propgraphdef`), read
/// as mapping hints: a vertex table's label names its class, an edge
/// table's source and destination make it an edge.
async fn read_property_graphs(client: &Client) -> Result<BTreeMap<String, GraphElement>> {
    let mut out = BTreeMap::new();
    for r in simple_rows(client, "SELECT pg_get_propgraphdef(c.oid) FROM pg_class c WHERE c.relkind = 'g'").await? {
        if let Some(def) = &r[0] {
            for e in parse_propgraph(def) {
                out.insert(e.table.clone(), e);
            }
        }
    }
    Ok(out)
}

fn tokens(s: &str) -> Vec<String> {
    let mut out = vec![];
    let mut cur = String::new();
    let mut quoted = false;
    for ch in s.chars() {
        if quoted {
            if ch == '"' {
                quoted = false;
            } else {
                cur.push(ch);
            }
            continue;
        }
        match ch {
            '"' => quoted = true,
            '(' | ')' | ',' => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                out.push(ch.to_string());
            }
            c if c.is_whitespace() => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Parse the subset of `CREATE PROPERTY GRAPH` the mapping uses: vertex
/// tables (KEY, first LABEL) and edge tables (SOURCE/DESTINATION KEY …
/// REFERENCES …, first LABEL). Unknown clauses are skipped.
pub fn parse_propgraph(def: &str) -> Vec<GraphElement> {
    let t = tokens(def);
    let up = |i: usize| t.get(i).map(|x| x.to_ascii_uppercase()).unwrap_or_default();
    let qualify = |name: &str| if name.contains('.') { name.to_string() } else { format!("public.{name}") };
    let mut out = vec![];
    let mut i = 0;
    while i < t.len() {
        let kind = up(i);
        if (kind == "VERTEX" || kind == "NODE" || kind == "EDGE" || kind == "RELATIONSHIP") && up(i + 1) == "TABLES" && t.get(i + 2).map(|x| x.as_str()) == Some("(") {
            let edge = kind == "EDGE" || kind == "RELATIONSHIP";
            i += 3;
            let mut depth = 1;
            let mut el: Option<GraphElement> = None;
            let mut expect_name = true;
            while i < t.len() && depth > 0 {
                let tok = t[i].clone();
                let u = tok.to_ascii_uppercase();
                match tok.as_str() {
                    "(" => {
                        depth += 1;
                        i += 1;
                        continue;
                    }
                    ")" => {
                        depth -= 1;
                        i += 1;
                        continue;
                    }
                    "," if depth == 1 => {
                        if let Some(e) = el.take() {
                            out.push(e);
                        }
                        expect_name = true;
                        i += 1;
                        continue;
                    }
                    _ => {}
                }
                if depth == 1 && expect_name {
                    el = Some(GraphElement { table: qualify(&tok), ..Default::default() });
                    expect_name = false;
                    i += 1;
                    continue;
                }
                let list = |start: usize| -> (Vec<String>, usize) {
                    let mut v = vec![];
                    let mut j = start;
                    if t.get(j).map(|x| x.as_str()) == Some("(") {
                        j += 1;
                        while j < t.len() && t[j] != ")" {
                            if t[j] != "," {
                                v.push(t[j].clone());
                            }
                            j += 1;
                        }
                        j += 1;
                    }
                    (v, j)
                };
                if let Some(e) = el.as_mut() {
                    if depth == 1 && u == "KEY" {
                        let (v, j) = list(i + 1);
                        e.key = v;
                        i = j;
                        continue;
                    }
                    if depth == 1 && u == "LABEL" && e.label.is_none() {
                        e.label = t.get(i + 1).cloned();
                        i += 2;
                        continue;
                    }
                    if depth == 1 && edge && (u == "SOURCE" || u == "DESTINATION") {
                        let mut j = i + 1;
                        let mut cols = vec![];
                        if up(j) == "KEY" {
                            let (v, k) = list(j + 1);
                            cols = v;
                            j = k;
                        }
                        if up(j) == "REFERENCES" {
                            j += 1;
                        }
                        let target = t.get(j).map(|x| qualify(x)).unwrap_or_default();
                        j += 1;
                        if t.get(j).map(|x| x.as_str()) == Some("(") {
                            let (_, k) = list(j);
                            j = k;
                        }
                        if u == "SOURCE" {
                            e.source = Some((cols, target));
                        } else {
                            e.destination = Some((cols, target));
                        }
                        i = j;
                        continue;
                    }
                }
                i += 1;
            }
            if let Some(e) = el.take() {
                out.push(e);
            }
            continue;
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn property_graph_subset() {
        let def = "CREATE PROPERTY GRAPH public.g VERTEX TABLES (customers KEY (id) LABEL client PROPERTIES (name), \
                   tags KEY (id)) EDGE TABLES (customer_tags KEY (customer_id, tag_id) SOURCE KEY (customer_id) \
                   REFERENCES customers (id) DESTINATION KEY (tag_id) REFERENCES tags (id) LABEL tagged)";
        let g = parse_propgraph(def);
        assert_eq!(g.len(), 3);
        assert_eq!(g[0].table, "public.customers");
        assert_eq!(g[0].label.as_deref(), Some("client"));
        assert_eq!(g[0].key, vec!["id"]);
        assert_eq!(g[2].source, Some((vec!["customer_id".to_string()], "public.customers".to_string())));
        assert_eq!(g[2].destination, Some((vec!["tag_id".to_string()], "public.tags".to_string())));
        assert_eq!(g[2].label.as_deref(), Some("tagged"));
    }
}
