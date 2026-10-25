/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The connector's YAML configuration (docs/specs/pg-connector.md,
//! "Configuration"). Everything but `source` and `connection` has a default;
//! the mapping section only overrides what inference decides.

use std::collections::BTreeMap;

use miette::{bail, miette, IntoDiagnostic, Result};
use serde::{Deserialize, Serialize};

/// One Postgres source.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PgConfig {
    /// The source's name: the prefix of every entity key (`pg:<source>/…`),
    /// the default publication and slot name, the ontology `pg:<source>`.
    pub source: String,
    /// A libpq URL (`postgres://user:pass@host:5432/db?sslmode=require`).
    /// `${VAR}` is replaced from the environment, so the password need not
    /// be in the file. Connect directly, not through a pooler (PgBouncer,
    /// Supavisor): replication connections cannot be pooled.
    pub connection: String,
    /// Publication to read; default `kleos_<source>`.
    #[serde(default)]
    pub publication: Option<String>,
    /// Replication slot; default `kleos_<source>`.
    #[serde(default)]
    pub slot: Option<String>,
    /// Create the publication when it does not exist (needs `CREATE` on the
    /// database and ownership of the tables); otherwise `pg init` prints the
    /// DDL for a DBA.
    #[serde(default = "yes")]
    pub create_publication: bool,
    /// `tables` (explicit list from `include`/`exclude`, default), `schema`
    /// (`FOR TABLES IN SCHEMA`, PG15+), `all` (`FOR ALL TABLES`).
    #[serde(default = "default_pub_mode")]
    pub publication_mode: String,
    /// Create a failover slot (`auto`: on PG17+; `true`; `false`).
    #[serde(default = "default_auto")]
    pub failover: String,
    /// Table globs (`schema.table`, `*` wildcard); default `public.*`.
    #[serde(default = "default_include")]
    pub include: Vec<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
    /// Target namespace (memory scope). Tenancy (docs/specs/tenancy.md)
    /// will read it as the scope inside `workspace`.
    #[serde(default = "default_ns")]
    pub ns: String,
    /// Target workspace (tenancy); `default` is today's implicit workspace.
    #[serde(default = "default_ns")]
    pub workspace: String,
    /// The service principal the connector writes as.
    #[serde(default)]
    pub actor: Actor,
    /// Copy existing rows before streaming.
    #[serde(default = "yes")]
    pub snapshot: bool,
    #[serde(default)]
    pub batch: BatchCfg,
    #[serde(default)]
    pub memory: MemoryCfg,
    #[serde(default)]
    pub ontology: OntologyCfg,
    #[serde(default)]
    pub lag_alert: LagAlert,
    /// Seconds between checks for new tables matching `include`
    /// (`publication_mode: tables`); 0 turns it off.
    #[serde(default = "default_new_tables")]
    pub new_tables_secs: u64,
    /// Seconds between `pg_logical_emit_message` heartbeats, so the slot
    /// advances on a quiet database whose cluster is busy; 0 turns them off.
    #[serde(default)]
    pub heartbeat_secs: u64,
    /// Seconds between standby status updates (acks) to Postgres.
    #[serde(default = "default_status_secs")]
    pub status_secs: u64,
    /// Mapping overrides per table (`schema.table`).
    #[serde(default)]
    pub tables: BTreeMap<String, TableOverride>,
    #[serde(default)]
    pub writeback: WritebackCfg,
}

fn yes() -> bool {
    true
}
fn default_pub_mode() -> String {
    "tables".into()
}
fn default_auto() -> String {
    "auto".into()
}
fn default_include() -> Vec<String> {
    vec!["public.*".into()]
}
fn default_ns() -> String {
    "default".into()
}
fn default_new_tables() -> u64 {
    60
}
fn default_status_secs() -> u64 {
    10
}

/// Who the connector's writes are attributed to: a service principal, and
/// the human and agent behind it when there are any (the tenancy actor
/// chain). Today it is recorded in `by` and in `kleos_pg_source`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Actor {
    /// Default `connector:pg/<source>`.
    #[serde(default)]
    pub service: Option<String>,
    #[serde(default)]
    pub human: Option<String>,
    #[serde(default)]
    pub agent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchCfg {
    /// Changes per kleos transaction (whole Postgres transactions only).
    #[serde(default = "default_max_changes")]
    pub max_changes: usize,
    /// Milliseconds a partial batch may wait.
    #[serde(default = "default_max_ms")]
    pub max_ms: u64,
    /// A Postgres transaction with more changes is applied in several kleos
    /// transactions (replay-safe; atomic again once its last part commits).
    #[serde(default = "default_max_tx")]
    pub max_tx_changes: usize,
    /// Rows per kleos transaction during snapshots.
    #[serde(default = "default_snapshot_rows")]
    pub snapshot_rows: usize,
    /// Batches queued between the reader and the kleos writer.
    #[serde(default = "default_queue")]
    pub queue: usize,
}

fn default_max_changes() -> usize {
    5000
}
fn default_max_ms() -> u64 {
    100
}
fn default_max_tx() -> usize {
    50_000
}
fn default_snapshot_rows() -> usize {
    2000
}
fn default_queue() -> usize {
    4
}

impl Default for BatchCfg {
    fn default() -> Self {
        BatchCfg {
            max_changes: default_max_changes(),
            max_ms: default_max_ms(),
            max_tx_changes: default_max_tx(),
            snapshot_rows: default_snapshot_rows(),
            queue: default_queue(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryCfg {
    /// A recallable summary memory per entity row (`Customer Acme (42):
    /// plan: pro; …`): `auto` (entity tables, default), `true`, `false`.
    #[serde(default = "default_auto")]
    pub rows: String,
    /// Text columns whose sampled average length reaches this many
    /// characters become their own memories instead of facts.
    #[serde(default = "default_text_min")]
    pub text_min_chars: usize,
    /// Embed the memories: `auto` (when the store has an embedder), `true`,
    /// `false` (FTS only).
    #[serde(default = "default_auto")]
    pub embed: String,
    /// Also write memory-API entities and mentions for labelled rows. Off by
    /// default: `remember`'s entity resolution reads a namespace's entities,
    /// and a large table would slow every `remember`.
    #[serde(default)]
    pub entities: bool,
}

fn default_text_min() -> usize {
    80
}

impl Default for MemoryCfg {
    fn default() -> Self {
        MemoryCfg { rows: default_auto(), text_min_chars: default_text_min(), embed: default_auto(), entities: false }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OntologyCfg {
    /// Add `pg:<source>` to the namespace's active ontologies.
    #[serde(default = "yes")]
    pub activate: bool,
    /// Review mode: mapping changes become proposals (`pg map approve`)
    /// before they change the ontology. Data keeps flowing either way.
    #[serde(default)]
    pub review: bool,
}

impl Default for OntologyCfg {
    fn default() -> Self {
        OntologyCfg { activate: true, review: false }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LagAlert {
    /// WAL bytes behind the server before `status` raises an alert.
    #[serde(default = "default_lag_bytes")]
    pub bytes: i64,
    /// Seconds behind the last commit before `status` raises an alert.
    #[serde(default = "default_lag_secs")]
    pub seconds: i64,
}

fn default_lag_bytes() -> i64 {
    1 << 30
}
fn default_lag_secs() -> i64 {
    300
}

impl Default for LagAlert {
    fn default() -> Self {
        LagAlert { bytes: default_lag_bytes(), seconds: default_lag_secs() }
    }
}

/// Per-table mapping override. A string value `skip` leaves the table out.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TableOverride {
    Word(String),
    Map(TableMapOverride),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableMapOverride {
    /// Class name (default: the singular, PascalCased table name).
    #[serde(default)]
    pub class: Option<String>,
    /// Key columns (default: the primary key).
    #[serde(default)]
    pub key: Option<Vec<String>>,
    /// Label template, `{column}` placeholders.
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub memory: Option<TableMemoryOverride>,
    /// Per column: `drop`, `pii`, `text` (a memory), `fact`, `category`
    /// (a node per value), or an object.
    #[serde(default)]
    pub columns: BTreeMap<String, ColumnOverride>,
    /// Map the table as an edge between two tables (PG19 property-graph
    /// vocabulary: source, destination, label).
    #[serde(default)]
    pub edge: Option<EdgeOverride>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableMemoryOverride {
    /// A row summary memory.
    #[serde(default)]
    pub row: Option<bool>,
    /// Columns stored as their own memories.
    #[serde(default)]
    pub text: Option<Vec<String>>,
    /// Row summary template (default: label and every fact column).
    #[serde(default)]
    pub template: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ColumnOverride {
    Word(String),
    Map(ColumnMapOverride),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnMapOverride {
    #[serde(default)]
    pub predicate: Option<String>,
    #[serde(default)]
    pub pii: Option<bool>,
    /// `fact`, `text`, `category`, `drop`.
    #[serde(default)]
    pub kind: Option<String>,
    /// A reference to another table (`schema.table`) for sources without
    /// foreign keys (Debezium from MySQL, Mongo).
    #[serde(default)]
    pub references: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeOverride {
    /// `column -> schema.table`
    pub source: String,
    /// `column -> schema.table`
    pub destination: String,
    #[serde(default)]
    pub label: Option<String>,
}

/// Agent write-back (`::flow` action `pg`, feature `flow-pg`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WritebackCfg {
    /// Tables flows may write (globs). Empty: write-back is off.
    #[serde(default)]
    pub allow: Vec<String>,
    /// The idempotency ledger (`schema.table`), written in the same
    /// Postgres transaction as the business write.
    #[serde(default = "default_ledger")]
    pub ledger: String,
    #[serde(default = "default_pool")]
    pub pool_size: usize,
    #[serde(default = "default_stmt_timeout")]
    pub statement_timeout_ms: u64,
}

fn default_ledger() -> String {
    "public.kleos_writeback".into()
}
fn default_pool() -> usize {
    4
}
fn default_stmt_timeout() -> u64 {
    10_000
}

impl Default for WritebackCfg {
    fn default() -> Self {
        WritebackCfg {
            allow: vec![],
            ledger: default_ledger(),
            pool_size: default_pool(),
            statement_timeout_ms: default_stmt_timeout(),
        }
    }
}

impl PgConfig {
    /// Parse a YAML (or JSON) document.
    pub fn parse(text: &str) -> Result<PgConfig> {
        let mut c: PgConfig = serde_yaml::from_str(text).map_err(|e| miette!("pg config: {e}"))?;
        c.connection = expand_env(&c.connection)?;
        c.check()?;
        Ok(c)
    }

    pub fn load(path: &str) -> Result<PgConfig> {
        let text = std::fs::read_to_string(path).into_diagnostic().map_err(|e| miette!("pg config {path}: {e}"))?;
        PgConfig::parse(&text)
    }

    fn check(&self) -> Result<()> {
        if self.source.is_empty() || !self.source.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            bail!("pg config: 'source' must be a non-empty name of letters, digits, '_' or '-', got '{}'", self.source);
        }
        if !["tables", "schema", "all"].contains(&self.publication_mode.as_str()) {
            bail!("pg config: publication_mode must be tables, schema or all, got '{}'", self.publication_mode);
        }
        if !["auto", "true", "false"].contains(&self.failover.as_str()) {
            bail!("pg config: failover must be auto, true or false");
        }
        for (k, v) in [("memory.rows", &self.memory.rows), ("memory.embed", &self.memory.embed)] {
            if !["auto", "true", "false"].contains(&v.as_str()) {
                bail!("pg config: {k} must be auto, true or false, got '{v}'");
            }
        }
        for (t, o) in &self.tables {
            if !t.contains('.') {
                bail!("pg config: table '{t}' must be schema-qualified (public.{t})");
            }
            if let TableOverride::Word(w) = o {
                if w != "skip" {
                    bail!("pg config: table '{t}': the only word override is 'skip', got '{w}'");
                }
            }
            if let TableOverride::Map(m) = o {
                for (c, co) in &m.columns {
                    let kind = match co {
                        ColumnOverride::Word(w) => Some(w.as_str()),
                        ColumnOverride::Map(m) => m.kind.as_deref(),
                    };
                    if let Some(k) = kind {
                        if !["drop", "pii", "text", "fact", "category"].contains(&k) {
                            bail!("pg config: {t}.{c}: column kind must be drop, pii, text, fact or category, got '{k}'");
                        }
                    }
                }
            }
        }
        if self.batch.max_changes == 0 || self.batch.snapshot_rows == 0 || self.batch.queue == 0 {
            bail!("pg config: batch sizes must be positive");
        }
        Ok(())
    }

    pub fn publication(&self) -> String {
        self.publication.clone().unwrap_or_else(|| format!("kleos_{}", self.source.replace('-', "_")))
    }

    pub fn slot(&self) -> String {
        self.slot.clone().unwrap_or_else(|| format!("kleos_{}", self.source.replace('-', "_")))
    }

    pub fn onto(&self) -> String {
        format!("pg:{}", self.source)
    }

    pub fn service(&self) -> String {
        self.actor.service.clone().unwrap_or_else(|| format!("connector:pg/{}", self.source))
    }

    /// `by` of the connector's facts.
    pub fn by(&self) -> String {
        format!("pg:{}", self.source)
    }

    /// Is `schema.table` included?
    pub fn includes(&self, qualified: &str) -> bool {
        if matches!(self.tables.get(qualified), Some(TableOverride::Word(w)) if w == "skip") {
            return false;
        }
        let ledger = self.writeback.ledger.as_str();
        if qualified == ledger {
            return false;
        }
        self.include.iter().any(|g| glob(g, qualified)) && !self.exclude.iter().any(|g| glob(g, qualified))
    }

    pub fn table_override(&self, qualified: &str) -> Option<&TableMapOverride> {
        match self.tables.get(qualified) {
            Some(TableOverride::Map(m)) => Some(m),
            _ => None,
        }
    }

    /// The configuration as stored in kleos: the connection's password
    /// removed.
    pub fn redacted(&self) -> serde_json::Value {
        let mut c = self.clone();
        c.connection = redact_url(&c.connection);
        serde_json::to_value(c).unwrap_or(serde_json::Value::Null)
    }

    pub fn writeback_allows(&self, qualified: &str) -> bool {
        self.writeback.allow.iter().any(|g| glob(g, qualified))
    }
}

/// `*` matches any run of characters; everything else literally.
pub fn glob(pattern: &str, s: &str) -> bool {
    fn rec(p: &[u8], s: &[u8]) -> bool {
        match (p.first(), s.first()) {
            (None, None) => true,
            (Some(b'*'), _) => rec(&p[1..], s) || (!s.is_empty() && rec(p, &s[1..])),
            (Some(a), Some(b)) if a == b => rec(&p[1..], &s[1..]),
            _ => false,
        }
    }
    rec(pattern.as_bytes(), s.as_bytes())
}

fn expand_env(s: &str) -> Result<String> {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find("${") {
        out.push_str(&rest[..i]);
        let after = &rest[i + 2..];
        let j = after.find('}').ok_or_else(|| miette!("pg config: unterminated ${{ in connection"))?;
        let var = &after[..j];
        let v = std::env::var(var).map_err(|_| miette!("pg config: environment variable {var} is not set"))?;
        out.push_str(&v);
        rest = &after[j + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// `postgres://user:secret@host/db` → `postgres://user:***@host/db`; a
/// `password=` key/value is masked too.
pub fn redact_url(url: &str) -> String {
    let mut s = url.to_string();
    if let Some(scheme_end) = s.find("://") {
        let rest = &s[scheme_end + 3..];
        if let Some(at) = rest.find('@') {
            let cred = &rest[..at];
            if let Some(colon) = cred.find(':') {
                let start = scheme_end + 3 + colon + 1;
                let end = scheme_end + 3 + at;
                s.replace_range(start..end, "***");
            }
        }
    }
    if let Some(i) = s.find("password=") {
        let end = s[i..].find(|c: char| c == '&' || c == ' ').map(|e| i + e).unwrap_or(s.len());
        s.replace_range(i + 9..end, "***");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs_and_redaction() {
        assert!(glob("public.*", "public.customers"));
        assert!(!glob("public.*", "crm.customers"));
        assert!(glob("public.audit_*", "public.audit_log"));
        assert_eq!(redact_url("postgres://u:pw@h:5432/db?sslmode=require"), "postgres://u:***@h:5432/db?sslmode=require");
        assert_eq!(redact_url("host=h password=x user=u"), "host=h password=*** user=u");
    }

    #[test]
    fn parses_defaults() {
        let c = PgConfig::parse("source: crm\nconnection: postgres://u@h/db\ntables:\n  public.audit: skip\n").unwrap();
        assert_eq!(c.slot(), "kleos_crm");
        assert_eq!(c.publication(), "kleos_crm");
        assert!(c.includes("public.customers"));
        assert!(!c.includes("public.audit"));
        assert!(!c.includes("public.kleos_writeback"));
    }
}
