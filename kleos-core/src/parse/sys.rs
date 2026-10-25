/*
 * Copyright 2022, The Cozo Project Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::bail;
use itertools::Itertools;
use miette::{ensure, miette, Diagnostic, Result};
use ordered_float::OrderedFloat;
use smartstring::{LazyCompact, SmartString};
use thiserror::Error;

use crate::data::program::InputProgram;
use crate::data::relation::{NullableColType, VecElementType};
use crate::runtime::vec_quant::VecQuant;
use crate::data::symb::Symbol;
use crate::data::value::{DataValue, ValidityTs};
use crate::fts::TokenizerConfig;
use crate::parse::expr::{build_expr, parse_string};
use crate::parse::query::parse_query;
use crate::parse::schema::parse_nullable_type;
use crate::parse::{ExtractSpan, Pairs, Rule, SourceSpan};
use crate::runtime::relation::AccessLevel;
use crate::{Expr, FixedRule};

#[derive(Debug)]
pub enum SysOp {
    Compact,
    /// kleos fork, structured diagnostics (`runtime/diagnostics.rs`):
    /// `::warnings` lists the Db's recent structured query warnings;
    /// `::warnings clear` empties the ring. The bool is `clear`.
    Warnings(bool),
    ListColumns(Symbol),
    ListIndices(Symbol),
    ListRelations,
    ListRunning,
    ListFixedRules,
    KillRunning(u64),
    Explain(Box<InputProgram>),
    RemoveRelation(Vec<Symbol>),
    RenameRelation(Vec<(Symbol, Symbol)>),
    ShowTrigger(Symbol),
    SetTriggers(Symbol, Vec<String>, Vec<String>, Vec<String>),
    SetAccessLevel(Vec<Symbol>, AccessLevel),
    CreateIndex(Symbol, Symbol, Vec<Symbol>),
    CreateVectorIndex(HnswIndexConfig),
    CreateFtsIndex(FtsIndexConfig),
    CreateMinHashLshIndex(MinHashLshConfig),
    RemoveIndex(Symbol, Symbol),
    DescribeRelation(Symbol, SmartString<LazyCompact>),
    /// Delete tuples whose stored arity is shorter than the schema
    /// (truncated values from interrupted writes). Surgical alternative to
    /// dropping a database that fails integrity checks.
    RepairCorrupt(Symbol),
    /// Rebuild a relation's HNSW/FTS/LSH indexes in place from their stored
    /// manifests (kleos fork, 0.12.1). See `runtime/reindex.rs`.
    Reindex(Symbol),
    /// kleos fork, bitemporality step 5: full belief timeline of the given
    /// keys of a tt-stamped relation — (relation, keys, limit, offset)
    TtHistory(Symbol, Vec<Vec<DataValue>>, Option<usize>, Option<usize>),
    /// (relation, cutoff tt µs): drop superseded records below the cutoff,
    /// persist the gc floor
    TtHistoryGc(Symbol, i64),
    /// (relation, keys, unredacted): hard-delete every record of the keys
    /// (GDPR); audit row written in the same transaction
    TtEvict(Symbol, Vec<Vec<DataValue>>, bool),
    /// kleos fork, graph projection: register `name` as a cached CSR over
    /// the `edges` relation, optionally with `nodes` naming the vertex set.
    /// Builds nothing — variants materialise on first algorithm use.
    CreateGraph {
        name: SmartString<LazyCompact>,
        edges: SmartString<LazyCompact>,
        nodes: Option<SmartString<LazyCompact>>,
    },
    /// kleos fork, graph projection: forget a projection, freeing its CSRs
    DropGraph(SmartString<LazyCompact>),
    /// kleos fork, graph projection: one row per built variant
    ListGraphs,
    CreateStoredQuery {
        name: Symbol,
        params: Vec<StoredQueryParam>,
        body_text: String,
        cur_vld: ValidityTs,
    },
    RemoveStoredQuery(Symbol),
    ListStoredQueries,
    ShowStoredQuery(Symbol),
    RunStoredQuery {
        name: Symbol,
        param_pool: Arc<BTreeMap<String, DataValue>>,
        cur_vld: ValidityTs,
    },
    /// kleos fork, agent memory (UPGRADE_PLAN E1/E2): supersession keys,
    /// change feeds, provenance links, cascading retract/erase, lineage.
    /// See `parse/memory.rs` and `runtime/memory.rs`.
    Memory(crate::parse::memory::MemoryOp),
    /// kleos fork: `::embedder list | enable | disable | backfill`
    /// (docs/specs/embedder-toggle.md). Process-wide, like the registry.
    Embedder(EmbedderOp),
    /// kleos fork: `::decider list | enable | disable | fit | fitted |
    /// forget` (docs/specs/decide.md).
    Decider(DeciderOp),
    /// `::extractor ...` (kleos fork).
    Extractor(ExtractorOp),
    /// kleos fork: `::memory init | remember | recall | forget | reflect |
    /// link | set | flush` (docs/specs/memory-api.md) and the branch ops
    /// `fork | rewind | merge | diff | discard | branches`
    /// (docs/specs/memory-branches.md): the op and its options as JSON.
    MemApi(String, serde_json::Value),
    /// kleos fork: `::entity resolve | merge | split | history | pending |
    /// reject` (docs/specs/entity-resolution.md).
    Entity(EntityOp),
    /// kleos fork: `::engram <verb> [model] {opts}` (docs/specs/engram-overlay.md).
    Engram(EngramOp),
    /// kleos fork: `::flow <op> {opts}` (docs/specs/durable-workflows.md):
    /// the op and its options as JSON.
    Flow(String, serde_json::Value),
    /// `::doc <op> [{opts}]` (kleos fork; docs/specs/documents.md).
    Doc(String, serde_json::Value),
    /// kleos fork: `::ontology <op> {opts}` (docs/specs/ontology.md).
    Ontology(String, serde_json::Value),
    /// kleos fork: `::tenant <op> {opts}` (docs/specs/tenancy.md).
    Tenant(String, serde_json::Value),
    /// kleos fork: `::principal <op> {as: ctx, …}` (docs/specs/tenancy.md).
    Principal(String, serde_json::Value),
    /// kleos fork: `::skill <op> {opts}` (docs/specs/skills.md).
    Skill(String, serde_json::Value),
}

/// `::flow <op> [{opts}]`.
fn parse_flow_op(inner: crate::parse::Pair<'_>, param_pool: &BTreeMap<String, DataValue>) -> Result<SysOp> {
    let mut parts = inner.into_inner();
    let verb = parts.next().unwrap().as_str().to_string();
    let mut args = serde_json::Map::new();
    for p in parts {
        for opt in p.into_inner() {
            let mut o = opt.into_inner();
            let key = o.next().unwrap().as_str().to_string();
            let val = build_expr(o.next().unwrap(), param_pool)?.eval_to_const()?;
            args.insert(key, serde_json::Value::from(val));
        }
    }
    Ok(SysOp::Flow(verb, serde_json::Value::Object(args)))
}

/// `::memory <verb> [arg] [{opts}]`: the positional argument is the text
/// (remember), the query (recall), the id or ids (forget) or the namespace
/// (reflect); everything else is an option.
fn parse_mem_api_op(
    inner: crate::parse::Pair<'_>,
    param_pool: &BTreeMap<String, DataValue>,
) -> Result<SysOp> {
    let mut parts = inner.into_inner();
    let verb = parts.next().unwrap().as_str().to_string();
    let mut args = serde_json::Map::new();
    for p in parts {
        match p.as_rule() {
            Rule::mem_opts => {
                for opt in p.into_inner() {
                    let mut o = opt.into_inner();
                    let key = o.next().unwrap().as_str().to_string();
                    let val = build_expr(o.next().unwrap(), param_pool)?.eval_to_const()?;
                    args.insert(key, serde_json::Value::from(val));
                }
            }
            _ => {
                let val = build_expr(p, param_pool)?.eval_to_const()?;
                let key = match (verb.as_str(), &val) {
                    ("remember", _) => "text",
                    ("recall", _) => "query",
                    ("forget" | "set", DataValue::List(_)) => "ids",
                    ("forget" | "set", _) => "id",
                    ("reflect" | "rewind" | "discard" | "branches", _) => "ns",
                    ("fork", _) => "to",
                    ("merge", _) => "from",
                    (v, _) => bail!("`::memory {v}` takes no positional argument; pass options as {{key: value}}"),
                };
                args.insert(key.to_string(), serde_json::Value::from(val));
            }
        }
    }
    Ok(SysOp::MemApi(verb, serde_json::Value::Object(args)))
}

/// `::entity ...` (kleos fork).
#[derive(Debug)]
pub(crate) enum EntityOp {
    /// Resolve mentions; options configure the relation (`commit: true` writes).
    Resolve {
        config: crate::entity::EntityConfig,
        mentions: Vec<crate::entity::Mention>,
        commit: bool,
    },
    /// Merge `from` into `into` by hand.
    Merge(String, DataValue, DataValue),
    /// Undo the merge of an id.
    Split(String, DataValue),
    /// Merge history, optionally of one id.
    History(String, Option<DataValue>),
    /// Pending possible-same-as links.
    Pending(String),
    /// Reject a possible-same-as link.
    Reject(String, DataValue, DataValue),
}

fn parse_entity_op(
    inner: crate::parse::Pair<'_>,
    param_pool: &BTreeMap<String, DataValue>,
) -> Result<EntityOp> {
    let inner = inner.into_inner().next().unwrap();
    let op_rule = inner.as_rule();
    let mut parts = inner.into_inner();
    let rel = parts.next().unwrap().as_str().to_string();
    let val = |p: Option<crate::parse::Pair<'_>>| -> Result<DataValue> {
        build_expr(p.unwrap(), param_pool)?.eval_to_const()
    };
    Ok(match op_rule {
        Rule::entity_resolve => {
            let ms = val(parts.next())?;
            let mentions = match &ms {
                DataValue::List(l) => l
                    .iter()
                    .map(crate::entity::Mention::from_value)
                    .collect::<Result<Vec<_>>>()?,
                v => bail!("`::entity resolve` takes a list of mentions, got {v:?}"),
            };
            let mut config = crate::entity::EntityConfig::new(rel);
            let mut commit = false;
            for opt in parts {
                let mut o = opt.into_inner();
                let key = o.next().unwrap().as_str();
                let v = build_expr(o.next().unwrap(), param_pool)?.eval_to_const()?;
                if key == "commit" {
                    commit = v
                        .get_bool()
                        .ok_or_else(|| miette!("`::entity resolve` option 'commit' must be a boolean"))?;
                } else {
                    config.set(key, &v)?;
                }
            }
            EntityOp::Resolve {
                config,
                mentions,
                commit,
            }
        }
        Rule::entity_merge => {
            let from = val(parts.next())?;
            let into = val(parts.next())?;
            EntityOp::Merge(rel, from, into)
        }
        Rule::entity_reject => {
            let a = val(parts.next())?;
            let b = val(parts.next())?;
            EntityOp::Reject(rel, a, b)
        }
        Rule::entity_split => EntityOp::Split(rel, val(parts.next())?),
        Rule::entity_history => {
            let id = match parts.next() {
                Some(p) => Some(build_expr(p, param_pool)?.eval_to_const()?),
                None => None,
            };
            EntityOp::History(rel, id)
        }
        Rule::entity_pending => EntityOp::Pending(rel),
        r => unreachable!("{:?}", r),
    })
}

/// `::engram ...` (kleos fork): the verb, the model name if given, and the
/// options evaluated to constants. Interpreted in `engram.rs`.
#[derive(Debug)]
pub(crate) struct EngramOp {
    pub(crate) verb: String,
    pub(crate) model: Option<String>,
    pub(crate) opts: BTreeMap<String, DataValue>,
}

fn parse_engram_op(
    inner: crate::parse::Pair<'_>,
    param_pool: &BTreeMap<String, DataValue>,
) -> Result<EngramOp> {
    let mut parts = inner.into_inner();
    let verb = parts.next().unwrap().as_str().to_string();
    let mut model = None;
    let mut opts = BTreeMap::new();
    for p in parts {
        if p.as_rule() == Rule::index_opt_field {
            let mut o = p.into_inner();
            let key = o.next().unwrap().as_str().to_string();
            let val = build_expr(o.next().unwrap(), param_pool)?.eval_to_const()?;
            opts.insert(key, val);
        } else {
            model = Some(embedder_name(p)?);
        }
    }
    Ok(EngramOp { verb, model, opts })
}

/// `::extractor ...` (kleos fork): entity extractors for automatic entities.
#[derive(Debug)]
pub(crate) enum ExtractorOp {
    /// One row per enabled or disabled extractor.
    List,
    /// Name and spec.
    Enable(String, crate::extract::ExtractorSpec),
    /// Name.
    Disable(String),
}

fn parse_extractor_op(
    inner: crate::parse::Pair<'_>,
    param_pool: &BTreeMap<String, DataValue>,
) -> Result<ExtractorOp> {
    let inner = inner.into_inner().next().unwrap();
    Ok(match inner.as_rule() {
        Rule::extractor_list => ExtractorOp::List,
        Rule::extractor_disable => ExtractorOp::Disable(embedder_name(inner.into_inner().next().unwrap())?),
        Rule::extractor_enable => {
            let mut parts = inner.into_inner();
            let name = embedder_name(parts.next().unwrap())?;
            let mut spec = crate::extract::ExtractorSpec::default();
            for opt in parts {
                let mut o = opt.into_inner();
                let key = o.next().unwrap().as_str();
                let val = build_expr(o.next().unwrap(), param_pool)?.eval_to_const()?;
                let v = match val {
                    DataValue::Str(s) => s.to_string(),
                    v => bail!("`::extractor enable` option '{key}' must be a string, got {v:?}"),
                };
                match key {
                    "model" => spec.model = Some(v),
                    "path" => spec.path = Some(v),
                    "precision" => spec.precision = Some(v),
                    "device" => spec.device = Some(v),
                    other => bail!(
                        "unknown option '{other}' for `::extractor enable`: expected model, path, precision or device"
                    ),
                }
            }
            ExtractorOp::Enable(name, spec)
        }
        r => unreachable!("{:?}", r),
    })
}

/// `::decider ...` (kleos fork).
#[derive(Debug)]
pub(crate) enum DeciderOp {
    /// One row per enabled or disabled decider.
    List,
    /// One row per installed probe or calibration.
    Fitted,
    /// Name and spec.
    Enable(String, crate::decide::DeciderSpec),
    /// Name.
    Disable(String),
    /// Fit a probe (or calibration) for decider `name` and `template` on a
    /// labelled relation.
    Fit {
        name: String,
        template: String,
        relation: Symbol,
        opts: FitOpts,
    },
    /// Remove the fitted probe/calibration of decider `name` and `template`.
    Forget(String, String),
}

/// Options of `::decider fit`.
#[derive(Debug, Clone)]
pub(crate) struct FitOpts {
    pub(crate) probe: bool,
    pub(crate) alpha: f64,
    pub(crate) wilson: bool,
}

fn parse_decider_op(
    inner: crate::parse::Pair<'_>,
    param_pool: &BTreeMap<String, DataValue>,
) -> Result<DeciderOp> {
    let inner = inner.into_inner().next().unwrap();
    let op_rule = inner.as_rule();
    Ok(match op_rule {
        Rule::decider_list => DeciderOp::List,
        Rule::decider_fitted => DeciderOp::Fitted,
        Rule::decider_disable => {
            DeciderOp::Disable(embedder_name(inner.into_inner().next().unwrap())?)
        }
        Rule::decider_forget => {
            let mut parts = inner.into_inner();
            let name = embedder_name(parts.next().unwrap())?;
            let template = embedder_name(parts.next().unwrap())?;
            DeciderOp::Forget(name, template)
        }
        Rule::decider_enable | Rule::decider_fit => {
            let mut parts = inner.into_inner();
            let name = embedder_name(parts.next().unwrap())?;
            let mut fit = None;
            if op_rule == Rule::decider_fit {
                let template = embedder_name(parts.next().unwrap())?;
                let rel = parts.next().unwrap();
                fit = Some((template, Symbol::new(rel.as_str(), rel.extract_span())));
            }
            let mut spec = crate::decide::DeciderSpec::default();
            let mut opts = FitOpts {
                probe: true,
                alpha: 0.02,
                wilson: false,
            };
            let op = if fit.is_some() { "fit" } else { "enable" };
            for opt in parts {
                let mut o = opt.into_inner();
                let key = o.next().unwrap().as_str();
                let val = build_expr(o.next().unwrap(), param_pool)?.eval_to_const()?;
                let as_str = |v: &DataValue| -> Result<String> {
                    match v {
                        DataValue::Str(s) => Ok(s.to_string()),
                        v => bail!("`::decider {op}` option '{key}' must be a string, got {v:?}"),
                    }
                };
                let as_bool = |v: &DataValue| -> Result<bool> {
                    match v {
                        DataValue::Bool(b) => Ok(*b),
                        v => bail!("`::decider {op}` option '{key}' must be a boolean, got {v:?}"),
                    }
                };
                match (op, key) {
                    ("enable", "model") => spec.model = Some(as_str(&val)?),
                    ("enable", "precision") => spec.precision = Some(as_str(&val)?),
                    ("enable", "device") => spec.device = Some(as_str(&val)?),
                    ("enable", "path") => spec.path = Some(as_str(&val)?),
                    ("fit", "probe") => opts.probe = as_bool(&val)?,
                    ("fit", "wilson") => opts.wilson = as_bool(&val)?,
                    ("fit", "alpha") => {
                        opts.alpha = match val.get_float() {
                            Some(a) if a > 0.0 && a < 1.0 => a,
                            _ => bail!("`::decider fit` option 'alpha' must be a number in (0, 1), got {val:?}"),
                        }
                    }
                    ("enable", other) => bail!(
                        "unknown option '{other}' for `::decider enable`: expected model, path, precision or device"
                    ),
                    (_, other) => bail!(
                        "unknown option '{other}' for `::decider fit`: expected probe, alpha or wilson"
                    ),
                }
            }
            match fit {
                Some((template, relation)) => DeciderOp::Fit {
                    name,
                    template,
                    relation,
                    opts,
                },
                None => DeciderOp::Enable(name, spec),
            }
        }
        r => unreachable!("{:?}", r),
    })
}

/// `::embedder ...` (kleos fork).
#[derive(Debug)]
pub(crate) enum EmbedderOp {
    /// One row per enabled or disabled embedder.
    List,
    /// Name and spec (an empty spec re-enables a disabled embedder as it was).
    Enable(String, crate::embed::EmbedderSpec),
    /// Name.
    Disable(String),
    /// Fill the null embedded columns of a relation.
    Backfill(Symbol),
}

fn embedder_name(p: crate::parse::Pair<'_>) -> Result<String> {
    Ok(match p.as_rule() {
        Rule::ident => p.as_str().to_string(),
        _ => parse_string(p)?.to_string(),
    })
}

fn parse_embedder_op(
    inner: crate::parse::Pair<'_>,
    param_pool: &BTreeMap<String, DataValue>,
) -> Result<EmbedderOp> {
    let inner = inner.into_inner().next().unwrap();
    Ok(match inner.as_rule() {
        Rule::embedder_list => EmbedderOp::List,
        Rule::embedder_disable => {
            EmbedderOp::Disable(embedder_name(inner.into_inner().next().unwrap())?)
        }
        Rule::embedder_backfill => {
            let rel = inner.into_inner().next().unwrap();
            EmbedderOp::Backfill(Symbol::new(rel.as_str(), rel.extract_span()))
        }
        Rule::embedder_enable => {
            let mut parts = inner.into_inner();
            let name = embedder_name(parts.next().unwrap())?;
            let mut spec = crate::embed::EmbedderSpec::default();
            for opt in parts {
                let mut o = opt.into_inner();
                let key = o.next().unwrap().as_str();
                let val = build_expr(o.next().unwrap(), param_pool)?.eval_to_const()?;
                let as_str = |v: &DataValue| -> Result<String> {
                    match v {
                        DataValue::Str(s) => Ok(s.to_string()),
                        v => bail!("`::embedder enable` option '{key}' must be a string, got {v:?}"),
                    }
                };
                match key {
                    "model" => spec.model = Some(as_str(&val)?),
                    "precision" => spec.precision = Some(as_str(&val)?),
                    "device" => spec.device = Some(as_str(&val)?),
                    "dim" => {
                        spec.dim = Some(match val.get_int() {
                            Some(n) if n > 0 => n as usize,
                            _ => bail!("`::embedder enable` option 'dim' must be a positive integer, got {val:?}"),
                        })
                    }
                    other => bail!(
                        "unknown option '{other}' for `::embedder enable`: expected model, precision, device or dim"
                    ),
                }
            }
            EmbedderOp::Enable(name, spec)
        }
        r => unreachable!("{:?}", r),
    })
}

/// A declared stored-query parameter. Types and defaults are optional, but
/// every parameter referenced by the stored body must have one declaration.
#[derive(Debug, Clone, serde_derive::Serialize, serde_derive::Deserialize)]
pub(crate) struct StoredQueryParam {
    pub(crate) name: String,
    pub(crate) typing: Option<NullableColType>,
    pub(crate) default: Option<DataValue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FtsIndexConfig {
    pub base_relation: SmartString<LazyCompact>,
    pub index_name: SmartString<LazyCompact>,
    pub extractor: String,
    pub tokenizer: TokenizerConfig,
    pub filters: Vec<TokenizerConfig>,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MinHashLshConfig {
    pub base_relation: SmartString<LazyCompact>,
    pub index_name: SmartString<LazyCompact>,
    pub extractor: String,
    pub tokenizer: TokenizerConfig,
    pub filters: Vec<TokenizerConfig>,
    pub n_gram: usize,
    pub n_perm: usize,
    pub false_positive_weight: OrderedFloat<f64>,
    pub false_negative_weight: OrderedFloat<f64>,
    pub target_threshold: OrderedFloat<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HnswIndexConfig {
    pub base_relation: SmartString<LazyCompact>,
    pub index_name: SmartString<LazyCompact>,
    pub vec_dim: usize,
    pub dtype: VecElementType,
    pub vec_fields: Vec<SmartString<LazyCompact>>,
    pub distance: HnswDistance,
    pub ef_construction: usize,
    pub m_neighbours: usize,
    pub index_filter: Option<String>,
    pub extend_candidates: bool,
    pub keep_pruned_connections: bool,
    /// Compressed traversal codes (kleos fork, A2).
    pub quantize: VecQuant,
    /// Rescoring multiplier for quantized search (0 = codec default).
    pub rescore: usize,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, serde_derive::Serialize, serde_derive::Deserialize,
)]
pub enum HnswDistance {
    L2,
    InnerProduct,
    Cosine,
}

#[derive(Debug, Diagnostic, Error)]
#[error("Cannot interpret {0} as process ID")]
#[diagnostic(code(parser::not_proc_id))]
struct ProcessIdError(String, #[label] SourceSpan);

pub(crate) fn parse_sys(
    mut src: Pairs<'_>,
    param_pool: &BTreeMap<String, DataValue>,
    algorithms: &BTreeMap<String, Arc<Box<dyn FixedRule>>>,
    custom_aggrs: crate::data::aggr::CustomAggrRegistries<'_>,
    cur_vld: ValidityTs,
) -> Result<SysOp> {
    let inner = src.next().unwrap();
    Ok(match inner.as_rule() {
        Rule::history_op => {
            let mut in_inner = inner.into_inner();
            let rel = in_inner.next().unwrap();
            let rel_symbol = Symbol::new(rel.as_str(), rel.extract_span());
            let keys_expr = build_expr(in_inner.next().unwrap(), param_pool)?;
            let keys = sysop_keys(keys_expr)?;
            // limit/offset are bare `pos_int` tokens, not exprs: an expr
            // would greedily parse `2 -1` as the single limit `2 - 1`
            let mut limit = None;
            let mut offset = None;
            if let Some(p) = in_inner.next() {
                limit = Some(sysop_pos_int(&p, "limit")?);
            }
            if let Some(p) = in_inner.next() {
                offset = Some(sysop_pos_int(&p, "offset")?);
            }
            SysOp::TtHistory(rel_symbol, keys, limit, offset)
        }
        Rule::history_gc_op => {
            let mut in_inner = inner.into_inner();
            let rel = in_inner.next().unwrap();
            let rel_symbol = Symbol::new(rel.as_str(), rel.extract_span());
            let cutoff_expr = build_expr(in_inner.next().unwrap(), param_pool)?;
            let cutoff = cutoff_expr
                .eval_to_const()?
                .get_int()
                .ok_or_else(|| miette!("::history_gc cutoff must be an integer (µs)"))?;
            SysOp::TtHistoryGc(rel_symbol, cutoff)
        }
        Rule::evict_op => {
            let mut in_inner = inner.into_inner();
            let rel = in_inner.next().unwrap();
            let rel_symbol = Symbol::new(rel.as_str(), rel.extract_span());
            let keys_expr = build_expr(in_inner.next().unwrap(), param_pool)?;
            let keys = sysop_keys(keys_expr)?;
            let unredacted = in_inner.next().is_some();
            SysOp::TtEvict(rel_symbol, keys, unredacted)
        }
        Rule::supersede_op
        | Rule::changes_op
        | Rule::provenance_op
        | Rule::retract_op
        | Rule::erase_op
        | Rule::lineage_op => SysOp::Memory(crate::parse::memory::parse_memory_op(
            inner, param_pool, cur_vld,
        )?),
        Rule::embedder_op => SysOp::Embedder(parse_embedder_op(inner, param_pool)?),
        Rule::decider_op => SysOp::Decider(parse_decider_op(inner, param_pool)?),
        Rule::extractor_op => SysOp::Extractor(parse_extractor_op(inner, param_pool)?),
        Rule::mem_api_op => parse_mem_api_op(inner, param_pool)?,
        Rule::entity_op => SysOp::Entity(parse_entity_op(inner, param_pool)?),
        Rule::engram_op => SysOp::Engram(parse_engram_op(inner, param_pool)?),
        Rule::flow_op => parse_flow_op(inner, param_pool)?,
        Rule::doc_op => match parse_flow_op(inner, param_pool)? {
            SysOp::Flow(v, a) => SysOp::Doc(v, a),
            other => other,
        },
        Rule::ontology_op => match parse_flow_op(inner, param_pool)? {
            SysOp::Flow(v, a) => SysOp::Ontology(v, a),
            _ => unreachable!(),
        },
        Rule::tenant_op => match parse_flow_op(inner, param_pool)? {
            SysOp::Flow(v, a) => SysOp::Tenant(v, a),
            _ => unreachable!(),
        },
        Rule::principal_op => match parse_flow_op(inner, param_pool)? {
            SysOp::Flow(v, a) => SysOp::Principal(v, a),
            _ => unreachable!(),
        },
        Rule::skill_op => match parse_flow_op(inner, param_pool)? {
            SysOp::Flow(v, a) => SysOp::Skill(v, a),
            _ => unreachable!(),
        },
        Rule::compact_op => SysOp::Compact,
        Rule::warnings_op => SysOp::Warnings(inner.into_inner().next().is_some()),
        Rule::running_op => SysOp::ListRunning,
        Rule::kill_op => {
            let i_expr = inner.into_inner().next().unwrap();
            let i_val = build_expr(i_expr, param_pool)?;
            let i_val = i_val.eval_to_const()?;
            let i_val = i_val
                .get_int()
                .ok_or_else(|| miette!("Process ID must be an integer"))?;
            SysOp::KillRunning(i_val as u64)
        }
        Rule::explain_op => {
            let prog = parse_query(
                inner.into_inner().next().unwrap().into_inner(),
                param_pool,
                algorithms,
                custom_aggrs,
                cur_vld,
            )?;
            SysOp::Explain(Box::new(prog))
        }
        Rule::describe_relation_op => {
            let mut inner = inner.into_inner();
            let rels_p = inner.next().unwrap();
            let rel = Symbol::new(rels_p.as_str(), rels_p.extract_span());
            let description = match inner.next() {
                None => Default::default(),
                Some(desc_p) => parse_string(desc_p)?,
            };
            SysOp::DescribeRelation(rel, description)
        }
        Rule::repair_corrupt_op => {
            let rels_p = inner.into_inner().next().unwrap();
            SysOp::RepairCorrupt(Symbol::new(rels_p.as_str(), rels_p.extract_span()))
        }
        Rule::reindex_op => {
            let rels_p = inner.into_inner().next().unwrap();
            SysOp::Reindex(Symbol::new(rels_p.as_str(), rels_p.extract_span()))
        }
        Rule::list_relations_op => SysOp::ListRelations,
        Rule::remove_relations_op => {
            let rel = inner
                .into_inner()
                .map(|rels_p| Symbol::new(rels_p.as_str(), rels_p.extract_span()))
                .collect_vec();

            SysOp::RemoveRelation(rel)
        }
        Rule::list_columns_op => {
            let rels_p = inner.into_inner().next().unwrap();
            let rel = Symbol::new(rels_p.as_str(), rels_p.extract_span());
            SysOp::ListColumns(rel)
        }
        Rule::list_indices_op => {
            let rels_p = inner.into_inner().next().unwrap();
            let rel = Symbol::new(rels_p.as_str(), rels_p.extract_span());
            SysOp::ListIndices(rel)
        }
        Rule::rename_relations_op => {
            let rename_pairs = inner
                .into_inner()
                .map(|pair| {
                    let mut src = pair.into_inner();
                    let rels_p = src.next().unwrap();
                    let rel = Symbol::new(rels_p.as_str(), rels_p.extract_span());
                    let rels_p = src.next().unwrap();
                    let new_rel = Symbol::new(rels_p.as_str(), rels_p.extract_span());
                    (rel, new_rel)
                })
                .collect_vec();
            SysOp::RenameRelation(rename_pairs)
        }
        Rule::access_level_op => {
            let mut ps = inner.into_inner();
            let access_level = match ps.next().unwrap().as_str() {
                "normal" => AccessLevel::Normal,
                "protected" => AccessLevel::Protected,
                "read_only" => AccessLevel::ReadOnly,
                "hidden" => AccessLevel::Hidden,
                _ => unreachable!(),
            };
            let mut rels = vec![];
            for rel_p in ps {
                let rel = Symbol::new(rel_p.as_str(), rel_p.extract_span());
                rels.push(rel)
            }
            SysOp::SetAccessLevel(rels, access_level)
        }
        Rule::trigger_relation_show_op => {
            let rels_p = inner.into_inner().next().unwrap();
            let rel = Symbol::new(rels_p.as_str(), rels_p.extract_span());
            SysOp::ShowTrigger(rel)
        }
        Rule::trigger_relation_op => {
            let mut src = inner.into_inner();
            let rels_p = src.next().unwrap();
            let rel = Symbol::new(rels_p.as_str(), rels_p.extract_span());
            let mut puts = vec![];
            let mut rms = vec![];
            let mut replaces = vec![];
            for clause in src {
                let mut clause_inner = clause.into_inner();
                let op = clause_inner.next().unwrap();
                let script = clause_inner.next().unwrap();
                let script_str = script.as_str();
                parse_query(
                    script.into_inner(),
                    &Default::default(),
                    algorithms,
                    // triggers: custom aggregates unsupported (R0; bounded-
                    // meet registrations likewise) — validate against empty
                    // registries so ::set_triggers fails fast.
                    crate::data::aggr::CustomAggrRegistries {
                        meet: &Default::default(),
                        bounded: &Default::default(),
                    },
                    cur_vld,
                )?;
                match op.as_rule() {
                    Rule::trigger_put => puts.push(script_str.to_string()),
                    Rule::trigger_rm => rms.push(script_str.to_string()),
                    Rule::trigger_replace => replaces.push(script_str.to_string()),
                    r => unreachable!("{:?}", r),
                }
            }
            SysOp::SetTriggers(rel, puts, rms, replaces)
        }
        Rule::lsh_idx_op => {
            let inner = inner.into_inner().next().unwrap();
            match inner.as_rule() {
                Rule::index_create_adv => {
                    let mut inner = inner.into_inner();
                    let rel = inner.next().unwrap();
                    let name = inner.next().unwrap();
                    let mut filters = vec![];
                    let mut tokenizer = TokenizerConfig {
                        name: Default::default(),
                        args: Default::default(),
                    };
                    let mut extractor = "".to_string();
                    let mut extract_filter = "".to_string();
                    let mut n_gram = 1;
                    let mut n_perm = 200;
                    let mut target_threshold = 0.9;
                    let mut false_positive_weight = 1.0;
                    let mut false_negative_weight = 1.0;
                    for opt_pair in inner {
                        let mut opt_inner = opt_pair.into_inner();
                        let opt_name = opt_inner.next().unwrap();
                        let opt_val = opt_inner.next().unwrap();
                        match opt_name.as_str() {
                            "false_positive_weight" => {
                                let mut expr = build_expr(opt_val, param_pool)?;
                                expr.partial_eval()?;
                                let v = expr.eval_to_const()?;
                                false_positive_weight = v.get_float().ok_or_else(|| {
                                    miette!("false_positive_weight must be a float")
                                })?;
                            }
                            "false_negative_weight" => {
                                let mut expr = build_expr(opt_val, param_pool)?;
                                expr.partial_eval()?;
                                let v = expr.eval_to_const()?;
                                false_negative_weight = v.get_float().ok_or_else(|| {
                                    miette!("false_negative_weight must be a float")
                                })?;
                            }
                            "n_gram" => {
                                let mut expr = build_expr(opt_val, param_pool)?;
                                expr.partial_eval()?;
                                let v = expr.eval_to_const()?;
                                n_gram = v
                                    .get_int()
                                    .ok_or_else(|| miette!("n_gram must be an integer"))?
                                    as usize;
                            }
                            "n_perm" => {
                                let mut expr = build_expr(opt_val, param_pool)?;
                                expr.partial_eval()?;
                                let v = expr.eval_to_const()?;
                                n_perm = v
                                    .get_int()
                                    .ok_or_else(|| miette!("n_perm must be an integer"))?
                                    as usize;
                            }
                            "target_threshold" => {
                                let mut expr = build_expr(opt_val, param_pool)?;
                                expr.partial_eval()?;
                                let v = expr.eval_to_const()?;
                                target_threshold = v
                                    .get_float()
                                    .ok_or_else(|| miette!("target_threshold must be a float"))?;
                            }
                            "extractor" => {
                                let mut ex = build_expr(opt_val, param_pool)?;
                                ex.partial_eval()?;
                                extractor = ex.to_string();
                            }
                            "extract_filter" => {
                                let mut ex = build_expr(opt_val, param_pool)?;
                                ex.partial_eval()?;
                                extract_filter = ex.to_string();
                            }
                            "tokenizer" => {
                                let mut expr = build_expr(opt_val, param_pool)?;
                                expr.partial_eval()?;
                                match expr {
                                    Expr::UnboundApply { op, args, .. } => {
                                        let mut targs = vec![];
                                        for arg in args.iter() {
                                            let v = arg.clone().eval_to_const()?;
                                            targs.push(v);
                                        }
                                        tokenizer.name = op;
                                        tokenizer.args = targs;
                                    }
                                    Expr::Binding { var, .. } => {
                                        tokenizer.name = var.name;
                                        tokenizer.args = vec![];
                                    }
                                    _ => bail!("Tokenizer must be a symbol or a call for an existing tokenizer"),
                                }
                            }
                            "filters" => {
                                let mut expr = build_expr(opt_val, param_pool)?;
                                expr.partial_eval()?;
                                match expr {
                                    Expr::Apply { op, args, .. } => {
                                        if op.name != "OP_LIST" {
                                            bail!("Filters must be a list of filters");
                                        }
                                        for arg in args.iter() {
                                            match arg {
                                                Expr::UnboundApply { op, args, .. } => {
                                                    let mut targs = vec![];
                                                    for arg in args.iter() {
                                                        let v = arg.clone().eval_to_const()?;
                                                        targs.push(v);
                                                    }
                                                    filters.push(TokenizerConfig {
                                                        name: op.clone(),
                                                        args: targs,
                                                    })
                                                }
                                                Expr::Binding { var, .. } => {
                                                    filters.push(TokenizerConfig {
                                                        name: var.name.clone(),
                                                        args: vec![],
                                                    })
                                                }
                                                _ => bail!("Tokenizer must be a symbol or a call for an existing tokenizer"),
                                            }
                                        }
                                    }
                                    _ => bail!("Filters must be a list of filters"),
                                }
                            }
                            _ => bail!("Unknown option {} for LSH index", opt_name.as_str()),
                        }
                    }
                    ensure!(
                        false_positive_weight > 0.,
                        "false_positive_weight must be positive"
                    );
                    ensure!(
                        false_negative_weight > 0.,
                        "false_negative_weight must be positive"
                    );
                    ensure!(n_gram > 0, "n_gram must be positive");
                    ensure!(n_perm > 0, "n_perm must be positive");
                    ensure!(
                        target_threshold > 0. && target_threshold < 1.,
                        "target_threshold must be between 0 and 1"
                    );
                    let total_weights = false_positive_weight + false_negative_weight;
                    false_positive_weight /= total_weights;
                    false_negative_weight /= total_weights;

                    if !extract_filter.is_empty() {
                        extractor = format!("if({}, {})", extract_filter, extractor);
                    }

                    let config = MinHashLshConfig {
                        base_relation: SmartString::from(rel.as_str()),
                        index_name: SmartString::from(name.as_str()),
                        extractor,
                        tokenizer,
                        filters,
                        n_gram,
                        n_perm,
                        false_positive_weight: false_positive_weight.into(),
                        false_negative_weight: false_negative_weight.into(),
                        target_threshold: target_threshold.into(),
                    };
                    SysOp::CreateMinHashLshIndex(config)
                }
                Rule::index_drop => {
                    let mut inner = inner.into_inner();
                    let rel = inner.next().unwrap();
                    let name = inner.next().unwrap();
                    SysOp::RemoveIndex(
                        Symbol::new(rel.as_str(), rel.extract_span()),
                        Symbol::new(name.as_str(), name.extract_span()),
                    )
                }
                r => unreachable!("{:?}", r),
            }
        }
        // kleos fork, graph projection (`docs/specs/graph-projection.md` §3.1)
        Rule::graph_op => {
            let inner = inner.into_inner().next().unwrap();
            match inner.as_rule() {
                Rule::graph_create => {
                    let mut inner = inner.into_inner();
                    let name = inner.next().unwrap();
                    let mut edges = None;
                    let mut nodes = None;
                    for opt_pair in inner {
                        let mut opt_inner = opt_pair.into_inner();
                        let opt_name = opt_inner.next().unwrap();
                        let opt_val = opt_inner.next().unwrap();
                        let slot = match opt_name.as_str() {
                            "edges" => &mut edges,
                            "nodes" => &mut nodes,
                            other => bail!(
                                "unknown option '{other}' for `::graph create`: \
                                 expected 'edges' or 'nodes'"
                            ),
                        };
                        *slot = Some(graph_source_name(opt_val, param_pool)?);
                    }
                    let Some(edges) = edges else {
                        bail!("`::graph create` requires an `edges` relation");
                    };
                    SysOp::CreateGraph {
                        name: SmartString::from(name.as_str()),
                        edges,
                        nodes,
                    }
                }
                Rule::graph_drop => {
                    let name = inner.into_inner().next().unwrap();
                    SysOp::DropGraph(SmartString::from(name.as_str()))
                }
                Rule::graph_list => SysOp::ListGraphs,
                r => unreachable!("{:?}", r),
            }
        }
        Rule::query_op => {
            let inner = inner.into_inner().next().unwrap();
            match inner.as_rule() {
                Rule::query_create => {
                    let mut parts = inner.into_inner();
                    let name_pair = parts.next().unwrap();
                    let name = Symbol::new(name_pair.as_str(), name_pair.extract_span());
                    let next = parts.next().unwrap();
                    let (params, body_pair) = if next.as_rule() == Rule::query_params_decl {
                        let mut params = Vec::new();
                        let mut names = std::collections::BTreeSet::new();
                        for param_pair in next.into_inner() {
                            let mut fields = param_pair.into_inner();
                            let param_name_pair = fields.next().unwrap();
                            let param_name = param_name_pair
                                .as_str()
                                .strip_prefix('$')
                                .unwrap()
                                .to_string();
                            ensure!(
                                names.insert(param_name.clone()),
                                "stored-query parameter '${param_name}' is declared more than once"
                            );
                            let mut typing = None;
                            let mut default = None;
                            for field in fields {
                                match field.as_rule() {
                                    Rule::col_type => typing = Some(parse_nullable_type(field)?),
                                    Rule::expr => {
                                        default =
                                            Some(build_expr(field, param_pool)?.eval_to_const()?)
                                    }
                                    r => unreachable!("{:?}", r),
                                }
                            }
                            if let (Some(typing), Some(value)) = (&typing, default.take()) {
                                default = Some(typing.coerce(value, cur_vld)?);
                            }
                            params.push(StoredQueryParam {
                                name: param_name,
                                typing,
                                default,
                            });
                        }
                        (params, parts.next().unwrap())
                    } else {
                        (Vec::new(), next)
                    };
                    let raw_body = body_pair.as_str();
                    let body_text = raw_body[1..raw_body.len() - 1].to_string();
                    SysOp::CreateStoredQuery {
                        name,
                        params,
                        body_text,
                        cur_vld,
                    }
                }
                Rule::query_remove => {
                    let name = inner.into_inner().next().unwrap();
                    SysOp::RemoveStoredQuery(Symbol::new(name.as_str(), name.extract_span()))
                }
                Rule::query_list => SysOp::ListStoredQueries,
                Rule::query_show => {
                    let name = inner.into_inner().next().unwrap();
                    SysOp::ShowStoredQuery(Symbol::new(name.as_str(), name.extract_span()))
                }
                Rule::query_run => {
                    let name = inner.into_inner().next().unwrap();
                    SysOp::RunStoredQuery {
                        name: Symbol::new(name.as_str(), name.extract_span()),
                        param_pool: Arc::new(param_pool.clone()),
                        cur_vld,
                    }
                }
                r => unreachable!("{:?}", r),
            }
        }
        Rule::fts_idx_op => {
            let inner = inner.into_inner().next().unwrap();
            match inner.as_rule() {
                Rule::index_create_adv => {
                    let mut inner = inner.into_inner();
                    let rel = inner.next().unwrap();
                    let name = inner.next().unwrap();
                    let mut filters = vec![];
                    let mut tokenizer = TokenizerConfig {
                        name: Default::default(),
                        args: Default::default(),
                    };
                    let mut extractor = "".to_string();
                    let mut extract_filter = "".to_string();
                    for opt_pair in inner {
                        let mut opt_inner = opt_pair.into_inner();
                        let opt_name = opt_inner.next().unwrap();
                        let opt_val = opt_inner.next().unwrap();
                        match opt_name.as_str() {
                            "extractor" => {
                                let mut ex = build_expr(opt_val, param_pool)?;
                                ex.partial_eval()?;
                                extractor = ex.to_string();
                            }
                            "extract_filter" => {
                                let mut ex = build_expr(opt_val, param_pool)?;
                                ex.partial_eval()?;
                                extract_filter = ex.to_string();
                            }
                            "tokenizer" => {
                                let mut expr = build_expr(opt_val, param_pool)?;
                                expr.partial_eval()?;
                                match expr {
                                    Expr::UnboundApply { op, args, .. } => {
                                        let mut targs = vec![];
                                        for arg in args.iter() {
                                            let v = arg.clone().eval_to_const()?;
                                            targs.push(v);
                                        }
                                        tokenizer.name = op;
                                        tokenizer.args = targs;
                                    }
                                    Expr::Binding { var, .. } => {
                                        tokenizer.name = var.name;
                                        tokenizer.args = vec![];
                                    }
                                    _ => bail!("Tokenizer must be a symbol or a call for an existing tokenizer"),
                                }
                            }
                            "filters" => {
                                let mut expr = build_expr(opt_val, param_pool)?;
                                expr.partial_eval()?;
                                match expr {
                                    Expr::Apply { op, args, .. } => {
                                        if op.name != "OP_LIST" {
                                            bail!("Filters must be a list of filters");
                                        }
                                        for arg in args.iter() {
                                            match arg {
                                                Expr::UnboundApply { op, args, .. } => {
                                                    let mut targs = vec![];
                                                    for arg in args.iter() {
                                                        let v = arg.clone().eval_to_const()?;
                                                        targs.push(v);
                                                    }
                                                    filters.push(TokenizerConfig {
                                                        name: op.clone(),
                                                        args: targs,
                                                    })
                                                }
                                                Expr::Binding { var, .. } => {
                                                    filters.push(TokenizerConfig {
                                                        name: var.name.clone(),
                                                        args: vec![],
                                                    })
                                                }
                                                _ => bail!("Tokenizer must be a symbol or a call for an existing tokenizer"),
                                            }
                                        }
                                    }
                                    _ => bail!("Filters must be a list of filters"),
                                }
                            }
                            _ => bail!("Unknown option {} for FTS index", opt_name.as_str()),
                        }
                    }
                    if !extract_filter.is_empty() {
                        extractor = format!("if({}, {})", extract_filter, extractor);
                    }
                    let config = FtsIndexConfig {
                        base_relation: SmartString::from(rel.as_str()),
                        index_name: SmartString::from(name.as_str()),
                        extractor,
                        tokenizer,
                        filters,
                    };
                    SysOp::CreateFtsIndex(config)
                }
                Rule::index_drop => {
                    let mut inner = inner.into_inner();
                    let rel = inner.next().unwrap();
                    let name = inner.next().unwrap();
                    SysOp::RemoveIndex(
                        Symbol::new(rel.as_str(), rel.extract_span()),
                        Symbol::new(name.as_str(), name.extract_span()),
                    )
                }
                r => unreachable!("{:?}", r),
            }
        }
        Rule::vec_idx_op => {
            let inner = inner.into_inner().next().unwrap();
            match inner.as_rule() {
                Rule::index_create_adv => {
                    let mut inner = inner.into_inner();
                    let rel = inner.next().unwrap();
                    let name = inner.next().unwrap();
                    // options
                    let mut vec_dim = 0;
                    let mut dtype = VecElementType::F32;
                    let mut vec_fields = vec![];
                    let mut distance = HnswDistance::L2;
                    let mut ef_construction = 0;
                    let mut m_neighbours = 0;
                    let mut index_filter = None;
                    let mut extend_candidates = false;
                    let mut keep_pruned_connections = false;
                    let mut quantize = VecQuant::None;
                    let mut rescore = 0;

                    for opt_pair in inner {
                        let mut opt_inner = opt_pair.into_inner();
                        let opt_name = opt_inner.next().unwrap();
                        let opt_val = opt_inner.next().unwrap();
                        let opt_val_str = opt_val.as_str();
                        match opt_name.as_str() {
                            "dim" => {
                                let v = build_expr(opt_val, param_pool)?
                                    .eval_to_const()?
                                    .get_int()
                                    .ok_or_else(|| miette!("Invalid vec_dim: {}", opt_val_str))?;
                                ensure!(v > 0, "Invalid vec_dim: {}", v);
                                vec_dim = v as usize;
                            }
                            "ef_construction" | "ef" => {
                                let v = build_expr(opt_val, param_pool)?
                                    .eval_to_const()?
                                    .get_int()
                                    .ok_or_else(|| {
                                        miette!("Invalid ef_construction: {}", opt_val_str)
                                    })?;
                                ensure!(v > 0, "Invalid ef_construction: {}", v);
                                ef_construction = v as usize;
                            }
                            "m_neighbours" | "m" => {
                                let v = build_expr(opt_val, param_pool)?
                                    .eval_to_const()?
                                    .get_int()
                                    .ok_or_else(|| {
                                        miette!("Invalid m_neighbours: {}", opt_val_str)
                                    })?;
                                ensure!(v > 0, "Invalid m_neighbours: {}", v);
                                m_neighbours = v as usize;
                            }
                            "dtype" => {
                                dtype = VecElementType::from_name(opt_val.as_str().trim())
                                    .ok_or_else(|| {
                                        miette!("Invalid dtype: {}", opt_val.as_str())
                                    })?
                            }
                            "fields" => {
                                let fields = build_expr(opt_val, &Default::default())?;
                                vec_fields = fields.to_var_list()?;
                            }
                            "distance" | "dist" => {
                                distance = match opt_val.as_str().trim() {
                                    "L2" => HnswDistance::L2,
                                    "IP" => HnswDistance::InnerProduct,
                                    "Cosine" => HnswDistance::Cosine,
                                    _ => {
                                        return Err(miette!(
                                            "Invalid distance: {}",
                                            opt_val.as_str()
                                        ))
                                    }
                                }
                            }
                            "filter" => {
                                index_filter = Some(opt_val.as_str().to_string());
                            }
                            "extend_candidates" => {
                                extend_candidates = opt_val.as_str().trim() == "true";
                            }
                            "keep_pruned_connections" => {
                                keep_pruned_connections = opt_val.as_str().trim() == "true";
                            }
                            "quantize" => {
                                let raw = opt_val.as_str().trim();
                                let name = raw.trim_matches(|c| c == '\'' || c == '"');
                                quantize = VecQuant::parse(name).ok_or_else(|| {
                                    miette!(
                                        "Invalid quantize: {} (expected none, binary, turbo2, turbo4 or turbo8)",
                                        raw
                                    )
                                })?;
                            }
                            "rescore" => {
                                let v = build_expr(opt_val, param_pool)?
                                    .eval_to_const()?
                                    .get_int()
                                    .ok_or_else(|| miette!("Invalid rescore: {}", opt_val_str))?;
                                ensure!(v > 0, "Invalid rescore: {}", v);
                                rescore = v as usize;
                            }
                            _ => return Err(miette!("Invalid option: {}", opt_name.as_str())),
                        }
                    }
                    if ef_construction == 0 {
                        bail!("ef_construction must be set");
                    }
                    if m_neighbours == 0 {
                        bail!("m_neighbours must be set");
                    }
                    SysOp::CreateVectorIndex(HnswIndexConfig {
                        base_relation: SmartString::from(rel.as_str()),
                        index_name: SmartString::from(name.as_str()),
                        vec_dim,
                        dtype,
                        vec_fields,
                        distance,
                        ef_construction,
                        m_neighbours,
                        index_filter,
                        extend_candidates,
                        keep_pruned_connections,
                        quantize,
                        rescore,
                    })
                }
                Rule::index_drop => {
                    let mut inner = inner.into_inner();
                    let rel = inner.next().unwrap();
                    let name = inner.next().unwrap();
                    SysOp::RemoveIndex(
                        Symbol::new(rel.as_str(), rel.extract_span()),
                        Symbol::new(name.as_str(), name.extract_span()),
                    )
                }
                r => unreachable!("{:?}", r),
            }
        }
        Rule::index_op => {
            let inner = inner.into_inner().next().unwrap();
            match inner.as_rule() {
                Rule::index_create => {
                    let span = inner.extract_span();
                    let mut inner = inner.into_inner();
                    let rel = inner.next().unwrap();
                    let name = inner.next().unwrap();
                    let cols = inner
                        .map(|p| Symbol::new(p.as_str(), p.extract_span()))
                        .collect_vec();

                    #[derive(Debug, Diagnostic, Error)]
                    #[error("index must have at least one column specified")]
                    #[diagnostic(code(parser::empty_index))]
                    struct EmptyIndex(#[label] SourceSpan);

                    ensure!(!cols.is_empty(), EmptyIndex(span));
                    SysOp::CreateIndex(
                        Symbol::new(rel.as_str(), rel.extract_span()),
                        Symbol::new(name.as_str(), name.extract_span()),
                        cols,
                    )
                }
                Rule::index_drop => {
                    let mut inner = inner.into_inner();
                    let rel = inner.next().unwrap();
                    let name = inner.next().unwrap();
                    SysOp::RemoveIndex(
                        Symbol::new(rel.as_str(), rel.extract_span()),
                        Symbol::new(name.as_str(), name.extract_span()),
                    )
                }
                _ => unreachable!(),
            }
        }
        Rule::list_fixed_rules => SysOp::ListFixedRules,
        r => unreachable!("{:?}", r),
    })
}

/// Evaluate a sysop key argument: a const list of key-lists (kleos fork).
fn sysop_keys(expr: crate::data::expr::Expr) -> Result<Vec<Vec<DataValue>>> {
    match expr.eval_to_const()? {
        DataValue::List(keys) => keys
            .into_iter()
            .map(|k| match k {
                DataValue::List(parts) => Ok(parts),
                v => Ok(vec![v]),
            })
            .collect(),
        _ => bail!("expected a list of keys, e.g. [[1], [2]]"),
    }
}

/// Read a `::graph create` source relation out of its option expression
/// (kleos fork). A bare `edges: knows` parses as a binding, a quoted
/// `edges: 'knows'` as a string constant; both name the same relation. Same
/// two shapes `::fts create`'s `tokenizer:` accepts.
fn graph_source_name(
    pair: crate::parse::Pair<'_>,
    param_pool: &BTreeMap<String, DataValue>,
) -> Result<SmartString<LazyCompact>> {
    let mut expr = build_expr(pair, param_pool)?;
    expr.partial_eval()?;
    if let Expr::Binding { var, .. } = expr {
        return Ok(var.name);
    }
    match expr.eval_to_const()? {
        DataValue::Str(s) => Ok(s),
        _ => bail!("a `::graph create` source must be a relation name, bare or quoted"),
    }
}

/// Parse a bare `pos_int` sysop token (kleos fork; `::history` limit/offset).
fn sysop_pos_int(pair: &pest::iterators::Pair<'_, Rule>, what: &str) -> Result<usize> {
    pair.as_str()
        .replace('_', "")
        .parse::<usize>()
        .map_err(|_| miette!("{} must be a non-negative integer", what))
}
