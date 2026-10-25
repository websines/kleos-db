/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The normalized change stream every source produces (logical
//! replication, snapshots, Debezium) and the kleos writer consumes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::catalog::TableInfo;
use crate::mapping::TableMap;

/// A row: typed JSON values by column name.
pub type Row = BTreeMap<String, Value>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    Insert,
    Update,
    Delete,
    Truncate,
    /// A row read by a snapshot (initial copy or resync).
    Read,
}

#[derive(Debug, Clone)]
pub struct Change {
    /// `schema.table`
    pub table: String,
    pub op: Op,
    /// The new row (insert, update, read). Without `missing` columns.
    pub new: Option<Row>,
    /// The old key or old row (update that changed the key, delete).
    pub old: Option<Row>,
    /// Columns Postgres did not send because they are unchanged TOASTed
    /// values: filled from the entity's stored row.
    pub missing: Vec<String>,
}

/// A table whose definition changed (DDL seen as a new Relation message,
/// or new columns in Debezium events): its new catalog entry and mapping.
#[derive(Debug, Clone)]
pub struct SchemaChange {
    pub table: String,
    pub info: TableInfo,
    pub map: TableMap,
    /// Columns that no longer exist: their facts are retracted.
    pub dropped: Vec<String>,
}

/// The write-back ledger row of a transaction: its changes are the echo of
/// a flow step (docs/specs/pg-connector.md, "Echo suppression").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Echo {
    pub idem_key: String,
    pub workflow: Option<String>,
    pub step: Option<String>,
    pub actor: Option<String>,
}

/// One source transaction (or one snapshot batch).
#[derive(Debug, Clone, Default)]
pub struct Tx {
    /// Commit LSN (the dedupe watermark); 0 for snapshot batches. For
    /// Debezium, a source position folded to a monotonic number when it has
    /// one.
    pub lsn: u64,
    /// End of the commit record: what is acknowledged to Postgres.
    pub end_lsn: u64,
    /// Commit time, µs since the Unix epoch (snapshot time for reads).
    pub commit_ts: i64,
    pub xid: u32,
    pub changes: Vec<Change>,
    pub snapshot: bool,
    pub echo: Option<Echo>,
    pub schema: Vec<SchemaChange>,
    /// Parts of a transaction too large to apply at once: (part, last).
    pub part: Option<(u32, bool)>,
}

impl Tx {
    pub fn len(&self) -> usize {
        self.changes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty() && self.schema.is_empty()
    }
}
