/*
 * Copyright 2022, The Cozo Project Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! This crate provides the core functionalities of [CozoDB](https://cozodb.org).
//! It may be used to embed CozoDB in your application.
//!
//! This doc describes the Rust API. To learn how to use CozoDB to query (CozoScript), see:
//!
//! * [The CozoDB documentation](https://docs.cozodb.org)
//!
//! Rust API usage:
//! ```
//! use kleos::*;
//!
//! let db = DbInstance::new("mem", "", Default::default()).unwrap();
//! let script = "?[a] := a in [1, 2, 3]";
//! let result = db.run_script(script, Default::default(), ScriptMutability::Immutable).unwrap();
//! println!("{:?}", result);
//! ```
//! We created an in-memory database above. There are other persistent options:
//! see [DbInstance::new]. It is perfectly fine to run multiple storage engines in the same process.
//!
#![doc = document_features::document_features!()]
#![warn(rust_2018_idioms, future_incompatible)]
// Every type reachable through the public API must also be nameable by callers
// (upstream CozoDB #288); CI's `-D warnings` turns this into a gate.
#![warn(unnameable_types)]
// kleos fork: the CI clippy gate runs `-D warnings`. The allows below are
// inherited from upstream CozoDB across the whole tree (or intrinsic to its
// design) and are tracked as a cleanup backlog rather than gated on, so the gate
// stays meaningful for *new* issues. Revisit behind a dedicated docs/cleanup pass.
#![allow(missing_docs)] // upstream public items are largely undocumented
#![allow(dead_code)] // upstream scaffolding kept for parity
#![allow(unused_assignments)] // inherited
#![allow(private_interfaces)] // internal IR structs are `pub` but expose pub(crate) types
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::mutable_key_type)] // DataValue (interior-mutable) used as map keys throughout
#![allow(clippy::unnecessary_unwrap)] // inherited query-compile patterns
#![allow(clippy::missing_transmute_annotations)] // documented intentional lifetime casts (e.g. sqlite stmt)
#![allow(clippy::manual_is_multiple_of)] // `is_multiple_of` needs Rust 1.87; our MSRV is 1.85

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
#[allow(unused_imports)]
use std::time::Instant;

use crossbeam::channel::{bounded, Receiver, Sender};
use lazy_static::lazy_static;
pub use miette::Error;
use miette::Report;
#[allow(unused_imports)]
use miette::{
    miette, GraphicalReportHandler, GraphicalTheme, IntoDiagnostic, JSONReportHandler, Result,
    ThemeCharacters, ThemeStyles,
};
use parse::CozoScript;
use serde_json::json;

pub(crate) use crate::utils::bail;

#[cfg(feature = "cypher")]
pub use cypher::{CypherGraphSchema, EdgeMap, NodeMap};
pub use data::aggr::{
    Aggregation, CustomAggrRegistries, MeetAggrObj, NormalAggrObj, RegisteredAggr,
    RegisteredBoundedMeet,
};
pub use data::value::{DataValue, Num, RegexWrapper, UuidWrapper, Validity, ValidityTs};
pub use fixed_rule::{FixedRule, FixedRuleInputRelation, FixedRulePayload};
/// The adjacency types a graph projection hands out (kleos fork).
///
/// These come from the [`graph`](https://docs.rs/graph) crate, so it is a
/// **public dependency** of kleos: these are its types, not ours, and a
/// consumer that also depends on `graph` directly must resolve to the same
/// semver-compatible version or the types will not unify.
///
/// The `graph = "0.3"` requirement therefore constrains kleos's own API.
/// Moving it to `0.4` is a **semver-major event for kleos** and must be
/// released as one; within `0.3.x`, cargo may pick a newer patch, which semver
/// obliges to stay API-compatible (verified: the `0.3.1` and `0.3.2` preludes
/// are identical). Pinning `=0.3.1` is deliberately *not* done — it would make
/// kleos unresolvable alongside any crate wanting a later `0.3.x`.
#[cfg(feature = "graph-algo")]
pub use graph::prelude::{DirectedCsrGraph, DirectedNeighbors, DirectedNeighborsWithValues, Graph};
#[cfg(feature = "columnar-io")]
pub use runtime::columnar::{ColumnarFileFormat, ColumnarImportOptions, ColumnarImportReport};
pub use runtime::db::Db;
pub use runtime::db::NamedRows;
pub use runtime::decider::FitReport;
pub use runtime::diagnostics::QueryWarning;
pub use runtime::live::{LiveStats, SubscribeOptions, Subscription, SubscriptionEvent};
#[cfg(feature = "graph-algo")]
pub use runtime::graph_projection::{
    GraphSource, GraphVariant, ProjectionVariant, VariantKey, VariantSpec,
};
pub use fixed_rule::utilities::rerank::Reranker;
pub use runtime::hybrid::{
    build_hybrid_query, Fusion, GraphLeg, HybridList, HybridSearch, MmrParams, RerankParams,
    ScoreNorm, WeightedFusionParams,
};
pub use runtime::relation::{decode_tuple_from_kv, try_decode_tuple_from_kv};
pub use runtime::temp_store::RegularTempStore;
pub use storage::mem::{new_cozo_mem, MemStorage, MemTx};
#[cfg(feature = "storage-new-rocksdb")]
pub use storage::newrocks::{new_cozo_newrocksdb, NewRocksDbStorage};
#[cfg(feature = "storage-rocksdb")]
pub use storage::rocks::{
    new_cozo_rocksdb, new_cozo_rocksdb_with_memory, process_default_rocks_memory, RocksDbStorage,
    RocksDbTx, RocksMemoryConfig, RocksMemoryConfigError, RocksMemoryResources, RocksMemoryStats,
    SharedMemoryConfigConflict,
};
#[cfg(feature = "storage-sled")]
pub use storage::sled::{new_cozo_sled, SledStorage};
#[cfg(feature = "storage-sqlite")]
pub use storage::sqlite::{new_cozo_sqlite, SqliteStorage, SqliteTx};
#[cfg(feature = "storage-tikv")]
pub use storage::tikv::{new_cozo_tikv, TiKvStorage};
pub use storage::{Storage, StoreTx};

pub use crate::data::expr::{Expr, Op};
use crate::data::json::JsonValue;
pub use crate::data::relation::VecElementType;
pub use crate::runtime::vec_quant::VecQuant;
pub use crate::data::symb::Symbol;
pub use crate::data::value::{JsonData, Vector};
pub use crate::fixed_rule::SimpleFixedRule;
pub use crate::fts::TokenizerConfig;
pub use crate::parse::SourceSpan;
pub use crate::runtime::callback::CallbackOp;
pub use crate::runtime::db::evaluate_expressions;
pub use crate::runtime::db::get_variables;
pub use crate::runtime::db::Payload;
pub use crate::runtime::db::Poison;
pub use crate::runtime::db::ScriptMutability;
pub use crate::runtime::db::ScriptRunOptions;
pub use crate::runtime::db::TransactionPayload;
#[cfg(not(target_arch = "wasm32"))]
pub use crate::runtime::governed_transaction::{
    GovernedTransaction, GovernedTransactionCancellation, GovernedTransactionOptions,
    GovernedTransactionWorker,
};
pub use crate::runtime::relation::AccessLevel;
pub use decide::{
    deciders, disable_decider, enable_decider, register_decider, Band, DecideAnswer,
    DecideRequest, Decider, DeciderSpec, DeciderStatus, Decision, Family, Template,
};
pub use entity::{
    Candidate, EntityConfig, MergeEvent, Mention, Outcome, Resolution, ResolveOptions,
};
pub use embed::{
    disable_embedder, embedders, enable_embedder, register_embedder, Embedder, EmbedderSpec,
    EmbedderStatus,
};

#[cfg(feature = "cypher")]
pub mod cypher;
pub mod data;
pub mod decide;
pub mod embed;
pub mod extract;
#[cfg(feature = "infer")]
pub mod extract_builtin;
pub mod memory;
pub mod memory_claims;
pub mod memory_extract;
pub mod memory_recall;
pub mod memory_time;
pub mod memory_engram;
pub mod entity;
pub mod engram;
pub mod flow;
#[cfg(feature = "docs")]
pub mod docs;
#[cfg(feature = "docs-url")]
pub(crate) mod docs_url;
pub mod ontology;
pub mod ontology_units;
pub mod ontology_type;
pub(crate) mod ontology_ttl;
pub mod ontology_learn;
pub(crate) mod flow_builtin;
pub(crate) mod flow_cron;
pub(crate) mod fixed_rule;
pub(crate) mod fts;
/// Test-support metric, not a public API: process-wide FTS posting-scan
/// counter, for scan-count regression tests (the NEAR double-scan class is
/// invisible in results). Hidden from docs; no stability guarantee.
#[doc(hidden)]
pub use fts::{FTS_BASE_ROW_FETCHES, FTS_LITERAL_SCANS};
pub mod parse;
pub mod sched;
pub mod tenancy;
pub(crate) mod query;
pub(crate) mod runtime;
pub(crate) mod storage;
pub(crate) mod utils;

/// A dispatcher for concrete storage implementations, wrapping [Db]. This is done so that
/// client code does not have to deal with generic code constantly. You may prefer to use
/// [Db] directly, especially if you provide a custom storage engine.
///
/// Many methods are dispatching methods for the corresponding methods on [Db].
///
/// Other methods are wrappers simplifying signatures to deal with only strings.
/// These methods made code for interop with other languages much easier,
/// but are not desirable if you are using Rust.
#[derive(Clone)]
pub enum DbInstance {
    /// In memory storage (not persistent)
    Mem(Db<MemStorage>),
    #[cfg(feature = "storage-sqlite")]
    /// Sqlite storage
    Sqlite(Db<SqliteStorage>),
    #[cfg(feature = "storage-rocksdb")]
    /// RocksDB storage
    RocksDb(Db<RocksDbStorage>),
    #[cfg(feature = "storage-new-rocksdb")]
    /// New RocksDB storage
    NewRocksDb(Db<NewRocksDbStorage>),
    #[cfg(feature = "storage-sled")]
    /// Sled storage (experimental)
    Sled(Db<SledStorage>),
    #[cfg(feature = "storage-tikv")]
    /// TiKV storage (experimental)
    TiKv(Db<TiKvStorage>),
}

impl Default for DbInstance {
    fn default() -> Self {
        Self::new("mem", "", Default::default()).unwrap()
    }
}

/// Run `$body` with `$db` bound to the inner [`Db`] of a [`DbInstance`].
macro_rules! dispatch_db {
    ($inst:expr, $db:ident => $body:expr) => {
        match $inst {
            DbInstance::Mem($db) => $body,
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite($db) => $body,
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb($db) => $body,
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb($db) => $body,
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled($db) => $body,
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv($db) => $body,
        }
    };
}

impl DbInstance {
    /// Create a DbInstance, which is a dispatcher for various concrete implementations.
    /// The valid engines are:
    ///
    /// * `mem`
    /// * `sqlite`
    /// * `rocksdb`
    /// * `newrocksdb`
    /// * `sled`
    /// * `tikv`
    ///
    /// assuming all features are enabled during compilation. Otherwise only
    /// some of the engines are available. The `mem` engine is always available.
    ///
    /// `path` is ignored for `mem` and `tikv` engines.
    /// `options` is ignored for every engine except `tikv` and `rocksdb`.
    ///
    /// For `rocksdb`, `options` may carry a `shared_memory` key —
    /// `{"shared_memory": {"total_bytes": 268435456, "memtable_fraction": 0.25}}`
    /// (`memtable_fraction` optional, default 0.25) — which joins the instance
    /// to the PROCESS-DEFAULT shared memory envelope: one block cache plus one
    /// write-buffer manager shared by every instance opened with the same
    /// config (docs/specs/cross-instance-memory.md §4). The first use fixes
    /// the canonical config; an identical later config joins, a differing one
    /// is a typed conflict error naming both. Rust hosts wanting several
    /// distinct envelopes should use `new_cozo_rocksdb_with_memory` with
    /// explicit handles instead. Non-RocksDB engines ignore the key.
    #[allow(unused_variables)]
    pub fn new(engine: &str, path: impl AsRef<Path>, options: &str) -> Result<Self> {
        let options = if options.is_empty() { "{}" } else { options };
        Ok(match engine {
            "mem" => Self::Mem(new_cozo_mem()?),
            #[cfg(feature = "storage-sqlite")]
            "sqlite" => Self::Sqlite(new_cozo_sqlite(path)?),
            #[cfg(feature = "storage-rocksdb")]
            "rocksdb" => {
                fn default_memtable_fraction() -> f64 {
                    RocksMemoryConfig::DEFAULT_MEMTABLE_FRACTION
                }
                #[derive(serde_derive::Deserialize)]
                struct SharedMemoryOpts {
                    total_bytes: usize,
                    #[serde(default = "default_memtable_fraction")]
                    memtable_fraction: f64,
                }
                #[derive(serde_derive::Deserialize)]
                struct RocksDbOpts {
                    #[serde(default)]
                    shared_memory: Option<SharedMemoryOpts>,
                }
                let opts: RocksDbOpts = serde_json::from_str(options).into_diagnostic()?;
                match opts.shared_memory {
                    None => Self::RocksDb(new_cozo_rocksdb(path)?),
                    Some(shared) => {
                        let config = RocksMemoryConfig {
                            total_bytes: shared.total_bytes,
                            memtable_fraction: shared.memtable_fraction,
                        };
                        let handle = process_default_rocks_memory(config)?;
                        Self::RocksDb(new_cozo_rocksdb_with_memory(path, &handle)?)
                    }
                }
            }
            #[cfg(feature = "storage-new-rocksdb")]
            "newrocksdb" => Self::NewRocksDb(new_cozo_newrocksdb(path)?),
            #[cfg(feature = "storage-sled")]
            "sled" => Self::Sled(new_cozo_sled(path)?),
            #[cfg(feature = "storage-tikv")]
            "tikv" => {
                #[derive(serde_derive::Deserialize)]
                struct TiKvOpts {
                    end_points: Vec<String>,
                    optimistic: bool,
                }
                let opts: TiKvOpts = serde_json::from_str(options).into_diagnostic()?;
                Self::TiKv(new_cozo_tikv(opts.end_points.clone(), opts.optimistic)?)
            }
            k => bail!(
                "database engine '{}' not supported (maybe not compiled in)",
                k
            ),
        })
    }
    /// Same as [Self::new], but inputs and error messages are all in strings
    pub fn new_with_str(
        engine: &str,
        path: &str,
        options: &str,
    ) -> std::result::Result<Self, String> {
        Self::new(engine, path, options).map_err(|err| err.to_string())
    }

    /// Per-instance RocksDB memory property reads (kleos fork,
    /// docs/specs/cross-instance-memory.md §5). Returns a struct whose fields
    /// are all `None` for every non-RocksDB engine — never an error.
    #[cfg(feature = "storage-rocksdb")]
    pub fn rocksdb_memory_stats(&self) -> RocksMemoryStats {
        match self {
            DbInstance::RocksDb(db) => db.rocksdb_memory_stats(),
            _ => RocksMemoryStats::default(),
        }
    }

    /// Dispatcher method.  See [crate::Db::set_durable_writes].
    pub fn set_durable_writes(&self, durable: bool) -> bool {
        match self {
            DbInstance::Mem(db) => db.set_durable_writes(durable),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.set_durable_writes(durable),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.set_durable_writes(durable),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.set_durable_writes(durable),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.set_durable_writes(durable),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.set_durable_writes(durable),
        }
    }

    /// Dispatcher method.  See [crate::Db::get_fixed_rules].
    pub fn get_fixed_rules(&self) -> BTreeMap<String, Arc<Box<dyn FixedRule>>> {
        match self {
            DbInstance::Mem(db) => db.get_fixed_rules(),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.get_fixed_rules(),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.get_fixed_rules(),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.get_fixed_rules(),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.get_fixed_rules(),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.get_fixed_rules(),
        }
    }
    /// Snapshot of the registered custom aggregates (kleos fork, R0b).
    pub fn get_custom_aggrs(&self) -> BTreeMap<String, RegisteredAggr> {
        match self {
            DbInstance::Mem(db) => db.get_custom_aggrs(),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.get_custom_aggrs(),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.get_custom_aggrs(),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.get_custom_aggrs(),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.get_custom_aggrs(),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.get_custom_aggrs(),
        }
    }

    /// Dispatcher method. See [crate::Db::get_custom_bounded_meets].
    pub fn get_custom_bounded_meets(&self) -> BTreeMap<String, RegisteredBoundedMeet> {
        match self {
            DbInstance::Mem(db) => db.get_custom_bounded_meets(),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.get_custom_bounded_meets(),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.get_custom_bounded_meets(),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.get_custom_bounded_meets(),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.get_custom_bounded_meets(),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.get_custom_bounded_meets(),
        }
    }
    /// Dispatcher method. See [crate::Db::run_script].
    pub fn run_script(
        &self,
        payload: &str,
        params: BTreeMap<String, DataValue>,
        mutability: ScriptMutability,
    ) -> Result<NamedRows> {
        self.run_script_with_options(payload, params, mutability, ScriptRunOptions::default())
    }

    /// `run_script` with mutable script and no parameters
    pub fn run_default(&self, payload: &str) -> Result<NamedRows> {
        self.run_script(payload, BTreeMap::new(), ScriptMutability::Mutable)
    }
    /// Run the CozoScript with per-call [`ScriptRunOptions`] (kleos fork,
    /// query budget) — currently a per-call wall-clock `timeout` in seconds.
    /// See [`crate::Db::run_script_with_options`] for the budget precedence.
    pub fn run_script_with_options(
        &self,
        payload: &str,
        params: BTreeMap<String, DataValue>,
        mutability: ScriptMutability,
        options: ScriptRunOptions,
    ) -> Result<NamedRows> {
        if let Some(r) = self.tenancy_script(payload, &params, mutability)? {
            return Ok(r);
        }
        match self {
            DbInstance::Mem(db) => db.run_script_with_options(payload, params, mutability, options),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => {
                db.run_script_with_options(payload, params, mutability, options)
            }
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => {
                db.run_script_with_options(payload, params, mutability, options)
            }
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => {
                db.run_script_with_options(payload, params, mutability, options)
            }
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => {
                db.run_script_with_options(payload, params, mutability, options)
            }
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => {
                db.run_script_with_options(payload, params, mutability, options)
            }
        }
    }
    /// Enable or disable the automatic factorized-count rewrite (kleos fork,
    /// query factorization). Db-wide kill switch, default ON since 0.13.1. See
    /// [`crate::Db::set_query_factorization`].
    pub fn set_query_factorization(&self, enabled: bool) {
        match self {
            DbInstance::Mem(db) => db.set_query_factorization(enabled),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.set_query_factorization(enabled),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.set_query_factorization(enabled),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.set_query_factorization(enabled),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.set_query_factorization(enabled),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.set_query_factorization(enabled),
        }
    }

    /// Whether the automatic factorized-count rewrite is enabled (kleos fork,
    /// query factorization). See [`crate::Db::query_factorization`].
    pub fn query_factorization(&self) -> bool {
        match self {
            DbInstance::Mem(db) => db.query_factorization(),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.query_factorization(),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.query_factorization(),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.query_factorization(),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.query_factorization(),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.query_factorization(),
        }
    }

    /// Set a Db-wide default per-query wall-clock budget in seconds (kleos
    /// fork, query budget). `None` disables it. See
    /// [`crate::Db::set_default_query_timeout`].
    pub fn set_default_query_timeout(&self, secs: Option<f64>) {
        match self {
            DbInstance::Mem(db) => db.set_default_query_timeout(secs),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.set_default_query_timeout(secs),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.set_default_query_timeout(secs),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.set_default_query_timeout(secs),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.set_default_query_timeout(secs),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.set_default_query_timeout(secs),
        }
    }

    /// Set (or clear) the Db-wide default per-query memory budget in estimated
    /// bytes (kleos fork, query memory budget; spec
    /// `docs/specs/memory-budget.md`). See
    /// [`crate::Db::set_default_query_mem_limit`].
    pub fn set_default_query_mem_limit(&self, bytes: Option<usize>) {
        match self {
            DbInstance::Mem(db) => db.set_default_query_mem_limit(bytes),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.set_default_query_mem_limit(bytes),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.set_default_query_mem_limit(bytes),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.set_default_query_mem_limit(bytes),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.set_default_query_mem_limit(bytes),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.set_default_query_mem_limit(bytes),
        }
    }
    /// Test-only dispatcher. See [`crate::Db::fail_next_commit_for_tests`].
    #[cfg(feature = "test-hooks")]
    #[doc(hidden)]
    pub fn fail_next_commit_for_tests(&self) {
        match self {
            DbInstance::Mem(db) => db.fail_next_commit_for_tests(),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.fail_next_commit_for_tests(),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.fail_next_commit_for_tests(),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.fail_next_commit_for_tests(),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.fail_next_commit_for_tests(),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.fail_next_commit_for_tests(),
        }
    }
    /// Test-only dispatcher. See [`crate::Db::legacy_next_import_for_tests`].
    #[cfg(feature = "test-hooks")]
    #[doc(hidden)]
    pub fn legacy_next_import_for_tests(&self) {
        match self {
            DbInstance::Mem(db) => db.legacy_next_import_for_tests(),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.legacy_next_import_for_tests(),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.legacy_next_import_for_tests(),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.legacy_next_import_for_tests(),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.legacy_next_import_for_tests(),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.legacy_next_import_for_tests(),
        }
    }

    /// The current Db-wide default per-query budget in seconds, or `None`
    /// (kleos fork, query budget). See [`crate::Db::default_query_timeout`].
    pub fn default_query_timeout(&self) -> Option<f64> {
        match self {
            DbInstance::Mem(db) => db.default_query_timeout(),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.default_query_timeout(),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.default_query_timeout(),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.default_query_timeout(),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.default_query_timeout(),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.default_query_timeout(),
        }
    }
    /// One-call hybrid retrieval (kleos fork addition): runs an HNSW + FTS
    /// (+ optional traversal) recall, fuses with Reciprocal Rank Fusion, and
    /// optionally diversifies with Maximal Marginal Relevance. Read-only. See
    /// [`HybridSearch`].
    pub fn hybrid_search(&self, q: &HybridSearch) -> Result<NamedRows> {
        let (script, params) = build_hybrid_query(q)?;
        self.run_script(&script, params, ScriptMutability::Immutable)
    }
    /// Build the CozoScript that [`DbInstance::hybrid_search`] would run, without
    /// executing it — for inspection or hand-tuning. See [`HybridSearch`].
    pub fn hybrid_search_script(&self, q: &HybridSearch) -> Result<String> {
        Ok(build_hybrid_query(q)?.0)
    }
    /// Run a read-only Cypher query (kleos fork; feature `cypher`). Translates an
    /// openCypher subset to CozoScript against the property-graph `schema`, runs it
    /// read-only, and returns rows whose columns are the `RETURN` items. Cypher
    /// literals are passed as params; user `$name` params are supplied via `params`.
    /// See [`CypherGraphSchema`] and `docs/specs/cypher-read.md`.
    #[cfg(feature = "cypher")]
    pub fn run_cypher(
        &self,
        query: &str,
        schema: &CypherGraphSchema,
        mut params: BTreeMap<String, DataValue>,
    ) -> Result<NamedRows> {
        if params.keys().any(|k| k.starts_with("cphr_")) {
            return Err(miette::miette!(
                "parameter names starting with `cphr_` are reserved by the Cypher translator"
            ));
        }
        let cs = crate::cypher::build_cypher_script(query, schema)?;
        for (k, v) in cs.params {
            params.insert(k, v);
        }
        let mut out = self.run_script(&cs.script, params, ScriptMutability::Immutable)?;
        // Bag mode appends hidden binding-key columns; keep only the RETURN columns.
        let n = cs.out_columns.len();
        for row in &mut out.rows {
            row.truncate(n);
        }
        out.headers = cs.out_columns;
        Ok(out)
    }
    /// Build the CozoScript that [`DbInstance::run_cypher`] would run, without
    /// executing it — for inspection or hand-tuning. Returns `(script, literal params)`.
    #[cfg(feature = "cypher")]
    pub fn cypher_to_script(
        &self,
        query: &str,
        schema: &CypherGraphSchema,
    ) -> Result<(String, BTreeMap<String, DataValue>)> {
        let cs = crate::cypher::build_cypher_script(query, schema)?;
        Ok((cs.script, cs.params))
    }
    /// Run a parsed (AST) program. If you have a string script, use `run_script` or `run_default`.
    pub fn run_script_ast(
        &self,
        payload: CozoScript,
        cur_vld: ValidityTs,
        mutability: ScriptMutability,
    ) -> Result<NamedRows> {
        match self {
            DbInstance::Mem(db) => db.run_script_ast(payload, cur_vld, mutability),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.run_script_ast(payload, cur_vld, mutability),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.run_script_ast(payload, cur_vld, mutability),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.run_script_ast(payload, cur_vld, mutability),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.run_script_ast(payload, cur_vld, mutability),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.run_script_ast(payload, cur_vld, mutability),
        }
    }
    /// Run the CozoScript passed in. The `params` argument is a map of parameters.
    /// Fold any error into the return JSON itself.
    /// See [crate::Db::run_script].
    pub fn run_script_fold_err(
        &self,
        payload: &str,
        params: BTreeMap<String, DataValue>,
        mutability: ScriptMutability,
    ) -> JsonValue {
        #[cfg(not(target_arch = "wasm32"))]
        let start = Instant::now();

        match self.run_script(payload, params, mutability) {
            Ok(named_rows) => {
                let mut j_val = named_rows.into_json();
                #[cfg(not(target_arch = "wasm32"))]
                let took = start.elapsed().as_secs_f64();
                let map = j_val.as_object_mut().unwrap();
                map.insert("ok".to_string(), json!(true));
                #[cfg(not(target_arch = "wasm32"))]
                map.insert("took".to_string(), json!(took));

                j_val
            }
            Err(err) => format_error_as_json(err, Some(payload)),
        }
    }
    /// [`DbInstance::run_script_fold_err`] with per-call [`ScriptRunOptions`]
    /// (kleos fork, query budget). A budget expiry folds into the JSON with
    /// `code == "eval::timeout"`, distinct from a `::kill`'s `eval::killed`.
    pub fn run_script_fold_err_with_options(
        &self,
        payload: &str,
        params: BTreeMap<String, DataValue>,
        mutability: ScriptMutability,
        options: ScriptRunOptions,
    ) -> JsonValue {
        #[cfg(not(target_arch = "wasm32"))]
        let start = Instant::now();

        match self.run_script_with_options(payload, params, mutability, options) {
            Ok(named_rows) => {
                let mut j_val = named_rows.into_json();
                #[cfg(not(target_arch = "wasm32"))]
                let took = start.elapsed().as_secs_f64();
                let map = j_val.as_object_mut().unwrap();
                map.insert("ok".to_string(), json!(true));
                #[cfg(not(target_arch = "wasm32"))]
                map.insert("took".to_string(), json!(took));

                j_val
            }
            Err(err) => format_error_as_json(err, Some(payload)),
        }
    }
    /// Run the CozoScript passed in. The `params` argument is a map of parameters formatted as JSON.
    /// See [crate::Db::run_script].
    pub fn run_script_str(&self, payload: &str, params: &str, immutable: bool) -> String {
        let params_json = if params.is_empty() {
            BTreeMap::default()
        } else {
            match serde_json::from_str::<BTreeMap<String, JsonValue>>(params) {
                Ok(map) => map
                    .into_iter()
                    .map(|(k, v)| (k, DataValue::from(v)))
                    .collect(),
                Err(_) => {
                    return json!({"ok": false, "message": "params argument is not a JSON map"})
                        .to_string();
                }
            }
        };
        self.run_script_fold_err(
            payload,
            params_json,
            if immutable {
                ScriptMutability::Immutable
            } else {
                ScriptMutability::Mutable
            },
        )
        .to_string()
    }
    /// Dispatcher method. See [crate::Db::export_relations].
    pub fn export_relations<I, T>(&self, relations: I) -> Result<BTreeMap<String, NamedRows>>
    where
        T: AsRef<str>,
        I: Iterator<Item = T>,
    {
        match self {
            DbInstance::Mem(db) => db.export_relations(relations),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.export_relations(relations),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.export_relations(relations),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.export_relations(relations),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.export_relations(relations),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.export_relations(relations),
        }
    }
    /// Dispatcher method. See [crate::Db::export_relation_as_rdf].
    #[cfg(feature = "rdf-io")]
    pub fn export_relation_as_rdf(
        &self,
        relation: &str,
        format: &str,
        prefixes: &BTreeMap<String, String>,
    ) -> Result<String> {
        match self {
            DbInstance::Mem(db) => db.export_relation_as_rdf(relation, format, prefixes),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.export_relation_as_rdf(relation, format, prefixes),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.export_relation_as_rdf(relation, format, prefixes),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.export_relation_as_rdf(relation, format, prefixes),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.export_relation_as_rdf(relation, format, prefixes),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.export_relation_as_rdf(relation, format, prefixes),
        }
    }
    /// Export relations to JSON-encoded string.
    /// See [crate::Db::export_relations]
    pub fn export_relations_str(&self, data: &str) -> String {
        match self.export_relations_str_inner(data) {
            Ok(s) => {
                let ret = json!({"ok": true, "data": s});
                format!("{ret}")
            }
            Err(err) => {
                let ret = json!({"ok": false, "message": err.to_string()});
                format!("{ret}")
            }
        }
    }
    fn export_relations_str_inner(&self, data: &str) -> Result<JsonValue> {
        #[derive(serde_derive::Deserialize)]
        struct Payload {
            relations: Vec<String>,
        }
        let j_val: Payload = serde_json::from_str(data).into_diagnostic()?;
        let results = self.export_relations(j_val.relations.iter().map(|s| s as &str))?;
        Ok(results
            .into_iter()
            .map(|(k, v)| (k, v.into_json()))
            .collect())
    }
    /// Dispatcher method. See [crate::Db::import_relations].
    pub fn import_relations(&self, data: BTreeMap<String, NamedRows>) -> Result<()> {
        match self {
            DbInstance::Mem(db) => db.import_relations(data),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.import_relations(data),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.import_relations(data),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.import_relations(data),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.import_relations(data),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.import_relations(data),
        }
    }
    /// Atomically import a Parquet or Arrow IPC file into an existing relation.
    #[cfg(feature = "columnar-io")]
    pub fn import_columnar_file(
        &self,
        relation: &str,
        path: impl AsRef<Path>,
        options: &ColumnarImportOptions,
    ) -> Result<ColumnarImportReport> {
        let path = path.as_ref();
        match self {
            DbInstance::Mem(db) => db.import_columnar_file(relation, path, options),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.import_columnar_file(relation, path, options),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.import_columnar_file(relation, path, options),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.import_columnar_file(relation, path, options),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.import_columnar_file(relation, path, options),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.import_columnar_file(relation, path, options),
        }
    }
    /// Import a relation, the data is given as a JSON string, and the returned result is converted into a string.
    /// See [crate::Db::import_relations].
    pub fn import_relations_str(&self, data: &str) -> String {
        match self.import_relations_str_with_err(data) {
            Ok(()) => {
                format!("{}", json!({"ok": true}))
            }
            Err(err) => {
                format!("{}", json!({"ok": false, "message": err.to_string()}))
            }
        }
    }
    /// Import a relation, the data is given as a JSON string.
    /// See [crate::Db::import_relations].
    pub fn import_relations_str_with_err(&self, data: &str) -> Result<()> {
        let json_data: JsonValue = serde_json::from_str(data).into_diagnostic()?;
        let json_object = json_data
            .as_object()
            .ok_or_else(|| miette!("A JSON object is requried"))?;
        let mapping = json_object
            .iter()
            .map(|(k, v)| -> Result<(String, NamedRows)> {
                Ok((k.to_string(), NamedRows::from_json(v)?))
            })
            .collect::<Result<_>>()?;
        self.import_relations(mapping)
    }
    /// Dispatcher method. See [crate::Db::backup_db].
    pub fn backup_db(&self, out_file: impl AsRef<Path>) -> Result<()> {
        match self {
            DbInstance::Mem(db) => db.backup_db(out_file),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.backup_db(out_file),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.backup_db(out_file),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.backup_db(out_file),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.backup_db(out_file),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.backup_db(out_file),
        }
    }
    /// Backup the running database into an Sqlite file, with JSON string return value.
    /// See [crate::Db::backup_db].
    pub fn backup_db_str(&self, out_file: impl AsRef<Path>) -> String {
        match self.backup_db(out_file) {
            Ok(_) => json!({"ok": true}).to_string(),
            Err(err) => json!({"ok": false, "message": err.to_string()}).to_string(),
        }
    }
    /// Dispatcher method. See [crate::Db::restore_backup].
    pub fn restore_backup(&self, in_file: impl AsRef<Path>) -> Result<()> {
        match self {
            DbInstance::Mem(db) => db.restore_backup(in_file),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.restore_backup(in_file),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.restore_backup(in_file),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.restore_backup(in_file),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.restore_backup(in_file),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.restore_backup(in_file),
        }
    }
    /// Restore from an Sqlite backup, with JSON string return value.
    /// See [crate::Db::restore_backup].
    pub fn restore_backup_str(&self, in_file: impl AsRef<Path>) -> String {
        match self.restore_backup(in_file) {
            Ok(_) => json!({"ok": true}).to_string(),
            Err(err) => json!({"ok": false, "message": err.to_string()}).to_string(),
        }
    }
    /// Dispatcher method. See [crate::Db::import_from_backup].
    pub fn import_from_backup(
        &self,
        in_file: impl AsRef<Path>,
        relations: &[String],
    ) -> Result<()> {
        match self {
            DbInstance::Mem(db) => db.import_from_backup(in_file, relations),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.import_from_backup(in_file, relations),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.import_from_backup(in_file, relations),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.import_from_backup(in_file, relations),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.import_from_backup(in_file, relations),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.import_from_backup(in_file, relations),
        }
    }
    /// Import relations from an Sqlite backup, with JSON string return value.
    /// See [crate::Db::import_from_backup].
    pub fn import_from_backup_str(&self, payload: &str) -> String {
        match self.import_from_backup_str_inner(payload) {
            Ok(_) => json!({"ok": true}).to_string(),
            Err(err) => json!({"ok": false, "message": err.to_string()}).to_string(),
        }
    }
    fn import_from_backup_str_inner(&self, payload: &str) -> Result<()> {
        #[derive(serde_derive::Deserialize)]
        struct Payload {
            path: String,
            relations: Vec<String>,
        }
        let json_payload: Payload = serde_json::from_str(payload).into_diagnostic()?;

        self.import_from_backup(&json_payload.path, &json_payload.relations)
    }

    /// Dispatcher method. See [crate::Db::register_callback].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn register_callback(
        &self,
        relation: &str,
        capacity: Option<usize>,
    ) -> (u32, Receiver<(CallbackOp, NamedRows, NamedRows)>) {
        match self {
            DbInstance::Mem(db) => db.register_callback(relation, capacity),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.register_callback(relation, capacity),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.register_callback(relation, capacity),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.register_callback(relation, capacity),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.register_callback(relation, capacity),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.register_callback(relation, capacity),
        }
    }

    /// Dispatcher method. See [crate::Db::subscribe].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn subscribe(
        &self,
        script: &str,
        params: BTreeMap<String, DataValue>,
        opts: SubscribeOptions,
    ) -> Result<Subscription> {
        match self {
            DbInstance::Mem(db) => db.subscribe(script, params, opts),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.subscribe(script, params, opts),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.subscribe(script, params, opts),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.subscribe(script, params, opts),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.subscribe(script, params, opts),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.subscribe(script, params, opts),
        }
    }

    /// Dispatcher method. See [crate::Db::unsubscribe].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn unsubscribe(&self, id: u64) -> bool {
        match self {
            DbInstance::Mem(db) => db.unsubscribe(id),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.unsubscribe(id),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.unsubscribe(id),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.unsubscribe(id),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.unsubscribe(id),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.unsubscribe(id),
        }
    }

    /// Dispatcher method. See [crate::Db::live_stats].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn live_stats(&self) -> LiveStats {
        match self {
            DbInstance::Mem(db) => db.live_stats(),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.live_stats(),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.live_stats(),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.live_stats(),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.live_stats(),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.live_stats(),
        }
    }

    /// Dispatcher method. See [crate::Db::unregister_callback].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn unregister_callback(&self, id: u32) -> bool {
        match self {
            DbInstance::Mem(db) => db.unregister_callback(id),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.unregister_callback(id),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.unregister_callback(id),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.unregister_callback(id),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.unregister_callback(id),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.unregister_callback(id),
        }
    }
    /// Dispatcher method. See [crate::Db::register_fixed_rule].
    pub fn register_fixed_rule<R>(&self, name: String, rule_impl: R) -> Result<()>
    where
        R: FixedRule + 'static,
    {
        match self {
            DbInstance::Mem(db) => db.register_fixed_rule(name, rule_impl),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.register_fixed_rule(name, rule_impl),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.register_fixed_rule(name, rule_impl),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.register_fixed_rule(name, rule_impl),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.register_fixed_rule(name, rule_impl),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.register_fixed_rule(name, rule_impl),
        }
    }
    /// Dispatcher method. See [crate::Db::register_custom_aggr].
    pub fn register_custom_aggr<F>(&self, name: String, is_meet: bool, factory: F) -> Result<()>
    where
        F: Fn() -> Box<dyn MeetAggrObj> + Send + Sync + 'static,
    {
        match self {
            DbInstance::Mem(db) => db.register_custom_aggr(name, is_meet, factory),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.register_custom_aggr(name, is_meet, factory),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.register_custom_aggr(name, is_meet, factory),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.register_custom_aggr(name, is_meet, factory),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.register_custom_aggr(name, is_meet, factory),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.register_custom_aggr(name, is_meet, factory),
        }
    }
    /// Dispatcher method. See [crate::Db::register_bounded_meet_aggr].
    pub fn register_bounded_meet_aggr<F>(
        &self,
        name: String,
        dominates: F,
        max_survivors: usize,
    ) -> Result<()>
    where
        F: Fn(&DataValue, &DataValue) -> bool + Send + Sync + 'static,
    {
        match self {
            DbInstance::Mem(db) => db.register_bounded_meet_aggr(name, dominates, max_survivors),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.register_bounded_meet_aggr(name, dominates, max_survivors),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => {
                db.register_bounded_meet_aggr(name, dominates, max_survivors)
            }
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => {
                db.register_bounded_meet_aggr(name, dominates, max_survivors)
            }
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.register_bounded_meet_aggr(name, dominates, max_survivors),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.register_bounded_meet_aggr(name, dominates, max_survivors),
        }
    }
    /// Dispatcher method. See [crate::Db::unregister_bounded_meet_aggr].
    pub fn unregister_bounded_meet_aggr(&self, name: &str) -> Result<bool> {
        match self {
            DbInstance::Mem(db) => db.unregister_bounded_meet_aggr(name),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.unregister_bounded_meet_aggr(name),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.unregister_bounded_meet_aggr(name),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.unregister_bounded_meet_aggr(name),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.unregister_bounded_meet_aggr(name),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.unregister_bounded_meet_aggr(name),
        }
    }
    /// Dispatcher method. See [crate::Db::unregister_custom_aggr].
    pub fn unregister_custom_aggr(&self, name: &str) -> Result<bool> {
        match self {
            DbInstance::Mem(db) => db.unregister_custom_aggr(name),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.unregister_custom_aggr(name),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.unregister_custom_aggr(name),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.unregister_custom_aggr(name),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.unregister_custom_aggr(name),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.unregister_custom_aggr(name),
        }
    }
    /// Dispatcher method. See [crate::Db::unregister_fixed_rule]
    pub fn unregister_fixed_rule(&self, name: &str) -> Result<bool> {
        match self {
            DbInstance::Mem(db) => db.unregister_fixed_rule(name),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.unregister_fixed_rule(name),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.unregister_fixed_rule(name),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.unregister_fixed_rule(name),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.unregister_fixed_rule(name),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.unregister_fixed_rule(name),
        }
    }
    /// Dispatcher method. See [crate::Db::register_reranker].
    pub fn register_reranker(
        &self,
        name: impl Into<String>,
        reranker: std::sync::Arc<dyn Reranker>,
    ) -> Result<()> {
        match self {
            DbInstance::Mem(db) => db.register_reranker(name, reranker),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.register_reranker(name, reranker),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.register_reranker(name, reranker),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.register_reranker(name, reranker),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.register_reranker(name, reranker),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.register_reranker(name, reranker),
        }
    }
    /// Dispatcher method. See [crate::Db::unregister_reranker].
    pub fn unregister_reranker(&self, name: &str) -> bool {
        match self {
            DbInstance::Mem(db) => db.unregister_reranker(name),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.unregister_reranker(name),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.unregister_reranker(name),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.unregister_reranker(name),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.unregister_reranker(name),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.unregister_reranker(name),
        }
    }

    /// Dispatcher method. See [crate::Db::run_multi_transaction]
    pub fn run_multi_transaction(
        &self,
        write: bool,
        payloads: Receiver<TransactionPayload>,
        results: Sender<Result<NamedRows>>,
    ) {
        match self {
            DbInstance::Mem(db) => db.run_multi_transaction(write, payloads, results),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.run_multi_transaction(write, payloads, results),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.run_multi_transaction(write, payloads, results),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.run_multi_transaction(write, payloads, results),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.run_multi_transaction(write, payloads, results),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.run_multi_transaction(write, payloads, results),
        }
    }
    /// Prepare a governed client and host-owned worker. The worker must be run
    /// on a dedicated blocking thread. Limits are captured now, before any
    /// scheduling delay; Db defaults only tighten the supplied limits.
    /// See [`GovernedTransactionWorker`] for admission and cancellation ownership.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn governed_transaction(
        &self,
        write: bool,
        options: GovernedTransactionOptions,
    ) -> Result<(GovernedTransaction, GovernedTransactionWorker)> {
        let options = match self {
            DbInstance::Mem(db) => db.governed_limits(options),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.governed_limits(options),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.governed_limits(options),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.governed_limits(options),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.governed_limits(options),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.governed_limits(options),
        };
        GovernedTransactionWorker::pair(self.clone(), write, options)
    }

    /// A higher-level, blocking wrapper for [crate::Db::run_multi_transaction]. Runs the transaction on a dedicated thread.
    /// Write transactions _may_ block other reads, but we guarantee that this does not happen for the RocksDB backend.
    /// This legacy API does not inherit script budgets, bound idle waiting or
    /// flush query warnings. Native hosts should use `governed_transaction`.
    pub fn multi_transaction(&self, write: bool) -> MultiTransaction {
        let (app2db_send, app2db_recv) = bounded(1);
        let (db2app_send, db2app_recv) = bounded(1);
        let db = self.clone();
        // A dedicated thread, as documented — NOT `rayon::spawn` (which this
        // used until 0.11.0): the transaction loop blocks in `recv` between
        // scripts for the transaction's whole life, and parking a global-pool
        // worker that long starves the pool. With `available_parallelism` open
        // transactions — or one open transaction racing any parallel query on
        // a single-core host — every rayon-using query in the process
        // deadlocks. (kleos fork; found by the graph-projection interleaving
        // suite, where one open reader plus one par-sorted CSR build wedged a
        // one-worker run.)
        std::thread::spawn(move || db.run_multi_transaction(write, app2db_recv, db2app_send));
        MultiTransaction {
            sender: app2db_send,
            receiver: db2app_recv,
        }
    }
    /// Turn a built-in embedder on under `name`, or re-enable a disabled one
    /// (empty `spec`). Process-wide: see [crate::embed::enable_embedder] and
    /// `::embedder enable`.
    pub fn enable_embedder(&self, name: &str, spec: EmbedderSpec) -> Result<()> {
        embed::enable_embedder(name, spec)
    }
    /// Turn the embedder `name` off, unloading its model and dropping its
    /// cached vectors. Process-wide: see [crate::embed::disable_embedder].
    pub fn disable_embedder(&self, name: &str) -> Result<bool> {
        embed::disable_embedder(name)
    }
    /// Every enabled and disabled embedder (process-wide).
    pub fn embedders(&self) -> Vec<EmbedderStatus> {
        embed::embedders()
    }
    /// Turn a built-in decider on under `name` (or re-enable a disabled one
    /// with an empty spec). Process-wide: see [crate::decide::enable_decider].
    pub fn enable_decider(&self, name: &str, spec: DeciderSpec) -> Result<()> {
        decide::enable_decider(name, spec)
    }
    /// Turn the decider `name` off. Process-wide.
    pub fn disable_decider(&self, name: &str) -> Result<bool> {
        decide::disable_decider(name)
    }
    /// Every enabled and disabled decider (process-wide).
    pub fn deciders(&self) -> Vec<DeciderStatus> {
        decide::deciders()
    }
    /// Turn an entity extractor on under `name` (or re-enable a disabled one
    /// with an empty spec). Process-wide: see [crate::extract::enable_extractor]
    /// and `::extractor enable`.
    pub fn enable_extractor(&self, name: &str, spec: extract::ExtractorSpec) -> Result<()> {
        extract::enable_extractor(name, spec)
    }
    /// Turn the extractor `name` off. Process-wide.
    pub fn disable_extractor(&self, name: &str) -> Result<bool> {
        extract::disable_extractor(name)
    }
    /// Every enabled and disabled extractor (process-wide).
    pub fn extractors(&self) -> Vec<extract::ExtractorStatus> {
        extract::extractors()
    }
    /// Decide one pair with decider `name` and a fixed template
    /// (`supersedes`, `same_entity`, `duplicate`, `contradicts`, `relation`,
    /// `important`; `b` is ignored by `important`).
    pub fn decide(&self, name: &str, template: &str, a: &str, b: &str) -> Result<Decision> {
        decide::decide_one(name, Template::parse(template)?, a, b)
    }
    /// Decide many pairs in batches.
    pub fn decide_batch(&self, name: &str, template: &str, items: &[(&str, &str)]) -> Result<Vec<Decision>> {
        decide::decide(name, Template::parse(template)?, items)
    }
    /// Dispatcher method. See [crate::Db::fit_decider].
    pub fn fit_decider(&self, name: &str, template: &str, relation: &str, probe: bool, alpha: f64) -> Result<FitReport> {
        match self {
            DbInstance::Mem(db) => db.fit_decider(name, template, relation, probe, alpha),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.fit_decider(name, template, relation, probe, alpha),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.fit_decider(name, template, relation, probe, alpha),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.fit_decider(name, template, relation, probe, alpha),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.fit_decider(name, template, relation, probe, alpha),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.fit_decider(name, template, relation, probe, alpha),
        }
    }
    /// Dispatcher method. See [crate::Db::forget_decider_fit].
    pub fn forget_decider_fit(&self, name: &str, template: &str) -> Result<bool> {
        match self {
            DbInstance::Mem(db) => db.forget_decider_fit(name, template),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.forget_decider_fit(name, template),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.forget_decider_fit(name, template),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.forget_decider_fit(name, template),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.forget_decider_fit(name, template),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.forget_decider_fit(name, template),
        }
    }
    /// A view of the store routed to a workspace partition
    /// (docs/specs/tenancy.md, "Partitions"). See [crate::Db::partition_view].
    pub fn partition_view(&self, partition: &str) -> Result<Self> {
        Ok(match self {
            DbInstance::Mem(db) => DbInstance::Mem(db.partition_view(partition)?),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => DbInstance::Sqlite(db.partition_view(partition)?),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => DbInstance::RocksDb(db.partition_view(partition)?),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => DbInstance::NewRocksDb(db.partition_view(partition)?),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => DbInstance::Sled(db.partition_view(partition)?),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => DbInstance::TiKv(db.partition_view(partition)?),
        })
    }
    /// `::tenant <op> {…}` and `::principal <op> {as, …}` scripts
    /// (docs/specs/tenancy.md); `None` for every other script.
    fn tenancy_script(
        &self,
        payload: &str,
        params: &BTreeMap<String, DataValue>,
        mutability: ScriptMutability,
    ) -> Result<Option<NamedRows>> {
        let t = payload.trim_start();
        if !(t.starts_with("::tenant") || t.starts_with("::principal")) {
            return Ok(None);
        }
        let Some((principal, op, args)) = dispatch_db!(self, db => db.parse_tenancy_op(payload, params))? else {
            return Ok(None);
        };
        let reads = ["recall", "context", "reflect", "scopes", "diff", "audit", "audit_verify", "list", "get"];
        if mutability == ScriptMutability::Immutable && !reads.contains(&op.as_str()) {
            bail!("`::{} {op}` cannot run in read-only mode", if principal { "principal" } else { "tenant" });
        }
        Ok(Some(if principal {
            let mut a = match args {
                JsonValue::Object(m) => m,
                _ => Default::default(),
            };
            // `as: parse_json('{…}')`, or flat `as_workspace`, `as_human`,
            // `as_agent`, `as_run`, `as_participants`
            let mut who = match a.remove("as") {
                Some(JsonValue::Object(m)) => m,
                Some(JsonValue::String(t)) => serde_json::from_str(&t).map_err(|e| miette!("`as`: {e}"))?,
                _ => Default::default(),
            };
            for k in a.keys().filter(|k| k.starts_with("as_")).cloned().collect::<Vec<_>>() {
                if let Some(v) = a.remove(&k) {
                    who.insert(k[3..].to_string(), v);
                }
            }
            let ctx: tenancy::PrincipalContext = serde_json::from_value(JsonValue::Object(who))
                .map_err(|e| miette!("`::principal {op}` needs `as`, a principal context: {e}"))?;
            let out = self.memory_call_as(&ctx, &op, JsonValue::Object(a))?;
            memory::memory_rows(&op, out)
        } else {
            flow::flow_rows(self.tenant_call(&op, args)?)
        }))
    }
    /// Tenancy admin ops (docs/specs/tenancy.md). See [crate::Db::tenant_call].
    pub fn tenant_call(&self, op: &str, args: JsonValue) -> Result<JsonValue> {
        dispatch_db!(self, db => db.tenant_call(op, args))
    }
    /// Memory ops as a principal. See [crate::Db::memory_call_as]. The
    /// document and consolidation runners of the principal's partitions
    /// start as needed, as they do for the plain store.
    pub fn memory_call_as(&self, ctx: &tenancy::PrincipalContext, op: &str, args: JsonValue) -> Result<JsonValue> {
        let r = dispatch_db!(self, db => db.memory_call_as(ctx, op, args))?;
        let running = r
            .get("documents")
            .and_then(|d| d.as_array())
            .is_some_and(|d| d.iter().any(|x| x.get("status").and_then(|s| s.as_str()) == Some("running")));
        for p in dispatch_db!(self, db => db.principal_partitions(ctx)).unwrap_or_default() {
            if let Ok(v) = self.partition_view(&p) {
                v.docs_kick(running);
                v.consolidate_kick();
            }
        }
        Ok(r)
    }
    /// Every workspace and org partition of the store. See
    /// [crate::Db::tenancy_partitions].
    pub fn tenancy_partitions(&self) -> Result<Vec<String>> {
        dispatch_db!(self, db => db.tenancy_partitions())
    }
    /// A live subscription filtered to a principal's scopes. See
    /// [crate::Db::subscribe_as].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn subscribe_as(
        &self,
        ctx: &tenancy::PrincipalContext,
        relation: &str,
    ) -> Result<(u32, Receiver<(CallbackOp, NamedRows, NamedRows)>)> {
        dispatch_db!(self, db => db.subscribe_as(ctx, relation))
    }
    /// Whether the store has a tenancy control plane.
    pub fn tenancy_enabled(&self) -> bool {
        dispatch_db!(self, db => db.tenancy_enabled())
    }
    /// The principal and actions of an API key (server mode).
    pub fn resolve_apikey(&self, key: &str) -> Result<Option<(String, Vec<String>)>> {
        dispatch_db!(self, db => db.resolve_apikey(key))
    }
    /// A principal's registered public keys. See [crate::Db::principal_jwks].
    pub fn principal_jwks(&self, id: &str) -> Result<Option<JsonValue>> {
        dispatch_db!(self, db => db.principal_jwks(id))
    }
    /// The principal an external identity `(iss, sub)` maps to.
    pub fn resolve_ident(&self, iss: &str, sub: &str) -> Result<Option<String>> {
        dispatch_db!(self, db => db.resolve_ident(iss, sub))
    }
    /// The kind of an active principal.
    pub fn principal_kind(&self, id: &str) -> Result<Option<String>> {
        dispatch_db!(self, db => db.principal_kind(id))
    }
    /// The org an admin op touches. See [crate::Db::tenant_target_org].
    pub fn tenant_target_org(&self, args: &JsonValue) -> Result<Option<String>> {
        dispatch_db!(self, db => db.tenant_target_org(args))
    }
    /// Whether `principal` administers `org`.
    pub fn is_org_admin(&self, principal: &str, org: &str) -> Result<bool> {
        dispatch_db!(self, db => db.is_org_admin(principal, org))
    }
    /// Dispatcher method. See [crate::Db::memory_init].
    pub fn memory_init(&self, opts: memory::InitOpts) -> Result<memory::MemConfig> {
        dispatch_db!(self, db => db.memory_init(opts))
    }
    /// Dispatcher method. See [crate::Db::remember]. Documents that ingest
    /// in the background start this store's document worker.
    pub fn remember(&self, text: &str, opts: memory::RememberOpts) -> Result<memory::RememberResult> {
        self.docs_kick(false);
        let r = dispatch_db!(self, db => db.remember(text, opts))?;
        if r.documents.iter().any(|d| d.get("status").and_then(|x| x.as_str()) == Some("running")) {
            self.docs_kick(true);
        }
        self.consolidate_kick();
        Ok(r)
    }
    /// Run started automatic consolidations (memory_consolidate.rs) on a
    /// background thread of this process: started lazily after a `remember`
    /// that queued one, it ticks the `consolidate` flow queue until nothing is
    /// due and exits. Leases make it safe beside other flow workers.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn consolidate_kick(&self) {
        use std::sync::atomic::Ordering;
        let st = dispatch_db!(self, db => db.consol_state.clone());
        if !st.pending.load(Ordering::SeqCst) || st.running.swap(true, Ordering::SeqCst) {
            return;
        }
        let db = self.clone();
        let _ = std::thread::Builder::new().name("kleos-consolidate".to_string()).spawn(move || {
            struct Clear(Arc<memory::consolidate::ConsolState>);
            impl Drop for Clear {
                fn drop(&mut self) {
                    self.0.running.store(false, std::sync::atomic::Ordering::SeqCst);
                }
            }
            let _clear = Clear(st.clone());
            let args = serde_json::json!({"queues": [memory::consolidate::CONSOLIDATE_QUEUE], "schedules": false,
                "worker": format!("consolidate-{}", std::process::id())});
            loop {
                st.pending.store(false, Ordering::SeqCst);
                let ran = match db.flow_call("tick", args.clone()) {
                    Ok(v) => v.get("ran").and_then(|r| r.as_array()).map(|r| r.len()).unwrap_or(0),
                    Err(e) => {
                        eprintln!("kleos consolidation: {e}");
                        0
                    }
                };
                if ran == 0 && !st.pending.load(Ordering::SeqCst) {
                    break;
                }
            }
        });
    }
    #[cfg(target_arch = "wasm32")]
    #[inline]
    pub fn consolidate_kick(&self) {}
    /// Dispatcher method. See [crate::Db::recall].
    pub fn recall(&self, query: &str, opts: memory::RecallOpts) -> Result<Vec<memory::RecallHit>> {
        self.docs_kick(false);
        dispatch_db!(self, db => db.recall(query, opts))
    }
    /// `::doc` admin ops (docs/specs/documents.md). See [crate::Db::doc_call].
    #[cfg(feature = "docs")]
    pub fn doc_call(&self, op: &str, args: JsonValue) -> Result<JsonValue> {
        dispatch_db!(self, db => db.doc_call(op, args))
    }
    /// Start the background document worker when document ingestions wait
    /// (`force`: start it without looking). The first call of a process
    /// looks once, so a restarted process resumes interrupted ingestions.
    #[cfg(all(feature = "docs", not(target_arch = "wasm32")))]
    pub fn docs_kick(&self, force: bool) {
        use std::sync::atomic::Ordering;
        let state = dispatch_db!(self, db => db.docs_state.clone());
        if state.running.load(Ordering::SeqCst) {
            return;
        }
        if !force {
            if state.checked.swap(true, Ordering::SeqCst) {
                return;
            }
            if !dispatch_db!(self, db => db.docs_pending()).unwrap_or(false) {
                return;
            }
        }
        if state.running.swap(true, Ordering::SeqCst) {
            return;
        }
        let db = self.clone();
        let st = state.clone();
        let _ = std::thread::Builder::new().name("kleos-docs-worker".to_string()).spawn(move || {
            // the flag clears however the thread ends (a panic included), so
            // the next call can start a new worker
            struct Clear(Arc<docs::DocsState>);
            impl Drop for Clear {
                fn drop(&mut self) {
                    self.0.running.store(false, std::sync::atomic::Ordering::SeqCst);
                }
            }
            let _clear = Clear(st.clone());
            let worker = format!("docs-{}", std::process::id());
            let (cb, changes) = db.register_callback(flow::FLOW_TASK, Some(64));
            let mut idle_since = std::time::Instant::now();
            while !st.stop.load(Ordering::SeqCst) {
                match dispatch_db!(&db, d => d.docs_work_once(&worker, Some(&st))) {
                    Ok(true) => {
                        idle_since = std::time::Instant::now();
                        continue;
                    }
                    Ok(false) => {}
                    Err(e) => eprintln!("kleos docs worker: {e}"),
                }
                // nothing claimable now: a leased workflow of a dead process
                // becomes claimable when its lease ends, so keep looking while
                // any document workflow is pending
                let pending = dispatch_db!(&db, d => d.docs_pending()).unwrap_or(false);
                if !pending && idle_since.elapsed() > std::time::Duration::from_secs(30) {
                    break;
                }
                crossbeam::select! {
                    recv(changes) -> _ => {}
                    default(std::time::Duration::from_millis(500)) => {}
                }
            }
            db.unregister_callback(cb);
            st.running.store(false, Ordering::SeqCst);
        });
    }
    #[cfg(not(all(feature = "docs", not(target_arch = "wasm32"))))]
    #[inline]
    pub fn docs_kick(&self, _force: bool) {}
    /// Dispatcher method. See [crate::Db::recall_context].
    pub fn recall_context(&self, query: &str, opts: memory::RecallOpts) -> Result<memory_recall::RecallContext> {
        dispatch_db!(self, db => db.recall_context(query, opts))
    }
    /// Dispatcher method. See [crate::Db::forget].
    pub fn forget(&self, opts: memory::ForgetOpts) -> Result<memory::ForgetResult> {
        dispatch_db!(self, db => db.forget(opts))
    }
    /// Dispatcher method. See [crate::Db::reflect].
    pub fn reflect(&self, opts: memory::ReflectOpts) -> Result<memory::ReflectResult> {
        dispatch_db!(self, db => db.reflect(opts))
    }
    /// Dispatcher method. See [crate::Db::link].
    pub fn link(&self, opts: memory::LinkOpts) -> Result<JsonValue> {
        dispatch_db!(self, db => db.link(opts))
    }
    /// Dispatcher method. See [crate::Db::memory_set].
    pub fn memory_set(&self, opts: memory::SetOpts) -> Result<JsonValue> {
        dispatch_db!(self, db => db.memory_set(opts))
    }
    /// Dispatcher method. See [crate::Db::memory_flush].
    pub fn memory_flush(&self) -> Result<usize> {
        dispatch_db!(self, db => db.memory_flush())
    }
    /// Dispatcher method. See [crate::Db::memory_fork].
    pub fn memory_fork(&self, opts: memory::ForkOpts) -> Result<memory::ForkResult> {
        dispatch_db!(self, db => db.memory_fork(opts))
    }
    /// Dispatcher method. See [crate::Db::memory_rewind].
    pub fn memory_rewind(&self, opts: memory::RewindOpts) -> Result<memory::RewindResult> {
        dispatch_db!(self, db => db.memory_rewind(opts))
    }
    /// Dispatcher method. See [crate::Db::memory_merge].
    pub fn memory_merge(&self, opts: memory::MergeOpts) -> Result<memory::MergeResult> {
        dispatch_db!(self, db => db.memory_merge(opts))
    }
    /// Dispatcher method. See [crate::Db::memory_diff].
    pub fn memory_diff(&self, opts: memory::DiffOpts) -> Result<memory::DiffResult> {
        dispatch_db!(self, db => db.memory_diff(opts))
    }
    /// Dispatcher method. See [crate::Db::memory_discard].
    pub fn memory_discard(&self, opts: memory::DiscardOpts) -> Result<memory::DiscardResult> {
        dispatch_db!(self, db => db.memory_discard(opts))
    }
    /// Dispatcher method. See [crate::Db::memory_branches].
    pub fn memory_branches(&self, opts: memory::BranchesOpts) -> Result<Vec<memory::BranchInfo>> {
        dispatch_db!(self, db => db.memory_branches(opts))
    }
    /// Dispatcher method. See [crate::Db::memory_call].
    pub fn memory_call(&self, op: &str, args: JsonValue) -> Result<JsonValue> {
        self.docs_kick(false);
        let r = dispatch_db!(self, db => db.memory_call(op, args))?;
        if op == "remember" {
            let running = r
                .get("documents")
                .and_then(|d| d.as_array())
                .is_some_and(|d| d.iter().any(|x| x.get("status").and_then(|s| s.as_str()) == Some("running")));
            if running {
                self.docs_kick(true);
            }
        }
        Ok(r)
    }
    /// Ontologies (docs/specs/ontology.md). See [crate::Db::ontology_call].
    pub fn ontology_call(&self, op: &str, args: JsonValue) -> Result<JsonValue> {
        dispatch_db!(self, db => db.ontology_call(op, args))
    }
    /// Durable workflows (docs/specs/durable-workflows.md). See [crate::Db::flow_call].
    pub fn flow_call(&self, op: &str, args: JsonValue) -> Result<JsonValue> {
        dispatch_db!(self, db => db.flow_call(op, args))
    }
    /// Start an in-process worker thread that runs due built-in workflows
    /// and fires due schedules (docs/specs/durable-workflows.md, "Waking").
    /// It wakes on commits to `kleos_flow_task` in this process and polls
    /// for the rest; it stops when the returned handle is stopped or dropped.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn flow_worker(&self, opts: FlowWorkerOpts) -> FlowWorker {
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (wake_tx, wake_rx) = bounded::<()>(1);
        let db = self.clone();
        let st = stop.clone();
        let handle = std::thread::Builder::new()
            .name("kleos-flow-worker".to_string())
            .spawn(move || {
                use std::sync::atomic::Ordering;
                let (cb, changes) = db.register_callback(flow::FLOW_TASK, Some(64));
                let queues = if opts.queues.is_empty() { vec!["default".to_string()] } else { opts.queues.clone() };
                let args = json!({"worker": opts.worker, "queues": queues, "lease_ms": opts.lease_ms, "max": 64});
                // the store and, with tenancy, every workspace and org
                // partition (re-listed every few seconds)
                let mut views: Vec<(Option<String>, DbInstance)> = vec![(None, db.clone())];
                let mut listed = std::time::Instant::now() - std::time::Duration::from_secs(60);
                let mut part_cbs: Vec<(DbInstance, u32, Receiver<(CallbackOp, NamedRows, NamedRows)>)> = vec![];
                while !st.load(Ordering::SeqCst) {
                    if listed.elapsed() > std::time::Duration::from_secs(5) {
                        listed = std::time::Instant::now();
                        for p in db.tenancy_partitions().unwrap_or_default() {
                            if views.iter().any(|(q, _)| q.as_deref() == Some(p.as_str())) {
                                continue;
                            }
                            if let Ok(v) = db.partition_view(&p) {
                                let (id, rx) = v.register_callback(flow::FLOW_TASK, Some(64));
                                part_cbs.push((v.clone(), id, rx));
                                views.push((Some(p), v));
                            }
                        }
                    }
                    let mut busy = false;
                    let now = flow::now_us();
                    let mut wait_ms = opts.poll_ms.max(1);
                    for (p, v) in &views {
                        match v.flow_call("tick", args.clone()) {
                            Ok(r) => {
                                busy |= r.get("ran").and_then(|x| x.as_array()).map_or(false, |a| !a.is_empty())
                                    || r.get("fired").and_then(|x| x.as_array()).map_or(false, |a| !a.is_empty());
                            }
                            // a partition without workflows yet has no flow relations
                            Err(e) if p.is_some() && format!("{e:?}").contains("kleos_flow") => {}
                            Err(e) => eprintln!("kleos flow worker: {e}"),
                        }
                        if let Ok(Some(due)) = dispatch_db!(v, d => d.flow_next_due(&queues)) {
                            wait_ms = wait_ms.min(((due - now).max(0) / 1000) as u64);
                        }
                    }
                    if busy {
                        continue;
                    }
                    if part_cbs.iter().any(|(_, _, rx)| !rx.is_empty()) {
                        for (_, _, rx) in &part_cbs {
                            while rx.try_recv().is_ok() {}
                        }
                        continue;
                    }
                    // partition commits are seen within a short poll
                    if views.len() > 1 {
                        wait_ms = wait_ms.min(200);
                    }
                    crossbeam::select! {
                        recv(changes) -> _ => {}
                        recv(wake_rx) -> _ => {}
                        default(std::time::Duration::from_millis(wait_ms)) => {}
                    }
                }
                for (v, id, _) in part_cbs {
                    v.unregister_callback(id);
                }
                db.unregister_callback(cb);
            })
            .expect("spawn the flow worker thread");
        FlowWorker {
            stop,
            wake: wake_tx,
            handle: Some(handle),
        }
    }
    /// Resolve entity mentions (docs/specs/entity-resolution.md). See [crate::Db::resolve_entities].
    pub fn resolve_entities(&self, cfg: &EntityConfig, mentions: &[Mention], opts: ResolveOptions) -> Result<Vec<Resolution>> {
        match self {
            DbInstance::Mem(db) => db.resolve_entities(cfg, mentions, opts),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.resolve_entities(cfg, mentions, opts),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.resolve_entities(cfg, mentions, opts),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.resolve_entities(cfg, mentions, opts),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.resolve_entities(cfg, mentions, opts),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.resolve_entities(cfg, mentions, opts),
        }
    }
    /// Merge entity `from` into `into` by hand. See [crate::Db::merge_entities].
    pub fn merge_entities(&self, relation: &str, from: &DataValue, into: &DataValue, reason: &str) -> Result<()> {
        match self {
            DbInstance::Mem(db) => db.merge_entities(relation, from, into, reason),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.merge_entities(relation, from, into, reason),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.merge_entities(relation, from, into, reason),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.merge_entities(relation, from, into, reason),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.merge_entities(relation, from, into, reason),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.merge_entities(relation, from, into, reason),
        }
    }
    /// Undo the merge of `id`. See [crate::Db::split_entity].
    pub fn split_entity(&self, relation: &str, id: &DataValue) -> Result<Option<DataValue>> {
        match self {
            DbInstance::Mem(db) => db.split_entity(relation, id),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.split_entity(relation, id),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.split_entity(relation, id),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.split_entity(relation, id),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.split_entity(relation, id),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.split_entity(relation, id),
        }
    }
    /// Merge history. See [crate::Db::entity_history].
    pub fn entity_history(&self, relation: &str, id: Option<&DataValue>) -> Result<Vec<MergeEvent>> {
        match self {
            DbInstance::Mem(db) => db.entity_history(relation, id),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.entity_history(relation, id),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.entity_history(relation, id),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.entity_history(relation, id),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.entity_history(relation, id),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.entity_history(relation, id),
        }
    }
    /// Possible same-as links awaiting review. See [crate::Db::entity_pending].
    pub fn entity_pending(&self, relation: &str) -> Result<NamedRows> {
        match self {
            DbInstance::Mem(db) => db.entity_pending(relation),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.entity_pending(relation),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.entity_pending(relation),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.entity_pending(relation),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.entity_pending(relation),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.entity_pending(relation),
        }
    }
    /// Reject a possible same-as link. See [crate::Db::reject_entity_link].
    pub fn reject_entity_link(&self, relation: &str, a: &DataValue, b: &DataValue) -> Result<bool> {
        match self {
            DbInstance::Mem(db) => db.reject_entity_link(relation, a, b),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.reject_entity_link(relation, a, b),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.reject_entity_link(relation, a, b),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.reject_entity_link(relation, a, b),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.reject_entity_link(relation, a, b),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.reject_entity_link(relation, a, b),
        }
    }
    /// The canonical entity of `id`. See [crate::Db::canonical_entity].
    pub fn canonical_entity(&self, relation: &str, id: &DataValue) -> Result<DataValue> {
        match self {
            DbInstance::Mem(db) => db.canonical_entity(relation, id),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.canonical_entity(relation, id),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.canonical_entity(relation, id),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.canonical_entity(relation, id),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.canonical_entity(relation, id),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.canonical_entity(relation, id),
        }
    }
    /// Dispatcher method. See [crate::Db::backfill_embeddings].
    pub fn backfill_embeddings(&self, relation: &str) -> Result<NamedRows> {
        match self {
            DbInstance::Mem(db) => db.backfill_embeddings(relation),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.backfill_embeddings(relation),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.backfill_embeddings(relation),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.backfill_embeddings(relation),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.backfill_embeddings(relation),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.backfill_embeddings(relation),
        }
    }
    /// Dispatcher method. See [crate::Db::set_graph_projection_capacity].
    /// (kleos fork; without this the graph-projection memory ceiling was
    /// unreachable from every language binding and from `kleos-bin`, while the
    /// engine's oversize-variant warning named it as the remedy.)
    #[cfg(feature = "graph-algo")]
    pub fn set_graph_projection_capacity(&self, bytes: usize) {
        match self {
            DbInstance::Mem(db) => db.set_graph_projection_capacity(bytes),
            #[cfg(feature = "storage-sqlite")]
            DbInstance::Sqlite(db) => db.set_graph_projection_capacity(bytes),
            #[cfg(feature = "storage-rocksdb")]
            DbInstance::RocksDb(db) => db.set_graph_projection_capacity(bytes),
            #[cfg(feature = "storage-new-rocksdb")]
            DbInstance::NewRocksDb(db) => db.set_graph_projection_capacity(bytes),
            #[cfg(feature = "storage-sled")]
            DbInstance::Sled(db) => db.set_graph_projection_capacity(bytes),
            #[cfg(feature = "storage-tikv")]
            DbInstance::TiKv(db) => db.set_graph_projection_capacity(bytes),
        }
    }
}

/// A multi-transaction handle.
/// You should use either the fields directly, or the associated functions.
pub struct MultiTransaction {
    /// Commands can be sent into the transaction through this channel
    pub sender: Sender<TransactionPayload>,
    /// Results can be retrieved from the transaction from this channel
    pub receiver: Receiver<Result<NamedRows>>,
}

impl MultiTransaction {
    /// Runs a single script in the transaction.
    pub fn run_script(
        &self,
        payload: &str,
        params: BTreeMap<String, DataValue>,
    ) -> Result<NamedRows> {
        if let Err(err) = self
            .sender
            .send(TransactionPayload::Query((payload.to_string(), params)))
        {
            bail!(err);
        }
        match self.receiver.recv() {
            Ok(r) => r,
            Err(err) => bail!(err),
        }
    }
    /// Commits the multi-transaction.
    ///
    /// Returns the commit's own error if it failed — the caller must not treat a
    /// failed commit as durable. (kleos fork, 0.12.1: this used to match
    /// `Ok(_) => Ok(())`, which discarded the `Result<NamedRows>` the
    /// transaction thread sends back and so reported **success for a failed
    /// commit**. `run_script` above has always propagated it correctly.)
    pub fn commit(&self) -> Result<()> {
        if let Err(err) = self.sender.send(TransactionPayload::Commit) {
            bail!(err);
        }
        match self.receiver.recv() {
            Ok(r) => r.map(|_| ()),
            Err(err) => bail!(err),
        }
    }
    /// Aborts the multi-transaction
    pub fn abort(&self) -> Result<()> {
        if let Err(err) = self.sender.send(TransactionPayload::Abort) {
            bail!(err);
        }
        match self.receiver.recv() {
            Ok(r) => r.map(|_| ()),
            Err(err) => bail!(err),
        }
    }
}

/// Options of [`DbInstance::flow_worker`].
#[derive(Debug, Clone)]
pub struct FlowWorkerOpts {
    /// Queues to run; default `default`.
    pub queues: Vec<String>,
    /// Longest sleep between checks for work from other processes (ms).
    pub poll_ms: u64,
    /// Lease of a claim (ms).
    pub lease_ms: i64,
    /// Worker name recorded on claims.
    pub worker: Option<String>,
}

impl Default for FlowWorkerOpts {
    fn default() -> Self {
        Self {
            queues: vec!["default".to_string()],
            poll_ms: 1000,
            lease_ms: flow::DEFAULT_LEASE_MS,
            worker: None,
        }
    }
}

/// A running in-process flow worker; stops when dropped.
pub struct FlowWorker {
    stop: Arc<std::sync::atomic::AtomicBool>,
    wake: Sender<()>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl FlowWorker {
    /// Wake the worker now (it also wakes on its own for work committed in
    /// this process).
    pub fn wake(&self) {
        let _ = self.wake.try_send(());
    }
    /// Stop the worker and wait for its current tick to end.
    pub fn stop(mut self) {
        self.halt();
    }
    fn halt(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        let _ = self.wake.try_send(());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for FlowWorker {
    fn drop(&mut self) {
        self.halt();
    }
}

/// Convert error raised by the database into friendly JSON format
pub fn format_error_as_json(mut err: Report, source: Option<&str>) -> JsonValue {
    if err.source_code().is_none() {
        if let Some(src) = source {
            err = err.with_source_code(format!("{src} "));
        }
    }
    let mut text_err = String::new();
    let mut json_err = String::new();
    TEXT_ERR_HANDLER
        .render_report(&mut text_err, err.as_ref())
        .expect("render text error failed");
    JSON_ERR_HANDLER
        .render_report(&mut json_err, err.as_ref())
        .expect("render json error failed");
    let mut json: serde_json::Value =
        serde_json::from_str(&json_err).expect("parse rendered json error failed");
    let map = json.as_object_mut().unwrap();
    map.insert("ok".to_string(), json!(false));
    map.insert("display".to_string(), json!(text_err));
    json
}

lazy_static! {
    static ref TEXT_ERR_HANDLER: GraphicalReportHandler = miette::GraphicalReportHandler::new()
        .with_theme(GraphicalTheme {
            characters: ThemeCharacters::unicode(),
            styles: ThemeStyles::ansi()
        });
    static ref JSON_ERR_HANDLER: JSONReportHandler = miette::JSONReportHandler::new();
}
