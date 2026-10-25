/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! kleos-pg: run kleos alongside Postgres (docs/specs/pg-connector.md).
//!
//! - **Ingest**: logical replication (pgoutput) with an initial snapshot;
//!   tables become ontology classes, rows entities, columns typed facts with
//!   valid time, foreign keys edges, long text recallable memories. The
//!   replication checkpoint commits in the same kleos transaction as the
//!   facts, and Postgres is acknowledged only after it: resume is exactly
//!   once ([`source`], [`sink`]).
//! - **Mapping**: inferred from the catalog, overridable in YAML, stored and
//!   versioned in kleos, defining the ontology `pg:<source>` ([`mapping`]).
//! - **Write-back**: a `::flow` action `pg` whose write and idempotency
//!   ledger row commit in one Postgres transaction ([`writeback`]); its rows
//!   come back through CDC linked to the flow step (echo suppression).
//! - **Debezium**: the same mapping for Debezium Server's HTTP sink
//!   ([`debezium`]).
//! - **Live queries**: Postgres tables as DataFusion providers joined with
//!   kleos relations (feature `query`).
//! - **Ops**: status, lag alerts, resync, slot cleanup ([`ops`]).
//!
//! Unpublished: it depends on the Supabase rust-postgres fork (git) for the
//! replication protocol, which crates.io does not allow in kleos-core.

pub mod admin;
pub mod catalog;
pub mod change;
pub mod config;
pub mod conn;
pub mod debezium;
pub mod mapping;
pub mod ops;
#[cfg(feature = "query")]
pub mod query;
pub mod sink;
pub mod source;
pub mod store;
pub mod value;
pub mod writeback;

pub use config::PgConfig;
pub use ops::{pg_call, register_config};
pub use source::{Connector, FailPoint};

/// Start a connector for `cfg` on the current Tokio runtime; returns its
/// handle (stop it with `handle.stop()`).
pub fn spawn(cfg: PgConfig, db: kleos::DbInstance) -> std::sync::Arc<source::Shared> {
    register_config(cfg.clone());
    let c = Connector::new(cfg, db);
    let shared = c.shared.clone();
    tokio::spawn(async move {
        if let Err(e) = c.run().await {
            log::error!("pg connector stopped: {e:?}");
        }
    });
    shared
}
