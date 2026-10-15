/*
 * Copyright 2026, The kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Parsing of the agent-memory system ops (kleos fork, UPGRADE_PLAN E1/E2;
//! `docs/specs/memory-supersession-provenance.md`). Execution lives in
//! `runtime/memory.rs`.

use std::collections::BTreeMap;

use crate::bail;
use miette::{miette, Result};
use smartstring::{LazyCompact, SmartString};

use crate::data::expr::Expr;
use crate::data::functions::MAX_VALIDITY_TS;
use crate::data::symb::Symbol;
use crate::data::value::{DataValue, ValidityTs};
use crate::parse::expr::build_expr;
use crate::parse::query::expr2vld_spec;
use crate::parse::{ExtractSpan, Pair, Rule};

/// One agent-memory system op. Wrapped as `SysOp::Memory`.
#[derive(Debug, Clone)]
pub(crate) enum MemoryOp {
    /// `::supersede create rel {k1, k2}`: declare the functional key.
    SupersedeCreate { rel: Symbol, cols: Vec<Symbol> },
    /// `::supersede drop rel`
    SupersedeDrop(Symbol),
    /// `::supersede list`
    SupersedeList,
    /// `::changes rel {from: T1, to: T2, as_of: TT, prefix: [..]}`
    Changes {
        rel: Symbol,
        from: ValidityTs,
        to: ValidityTs,
        as_of: Option<ValidityTs>,
        prefix: Vec<DataValue>,
    },
    /// `::provenance create link {derived: d, source: s}`
    ProvenanceCreate {
        link: Symbol,
        derived: SmartString<LazyCompact>,
        source: SmartString<LazyCompact>,
    },
    /// `::provenance drop link`
    ProvenanceDrop(Symbol),
    /// `::provenance list`
    ProvenanceList,
    /// `::retract rel [[k..]] (@ T)?` (soft) and `::erase rel [[k..]]`
    /// (hard, GDPR), both cascading through the declared provenance links.
    RemoveSource {
        rel: Symbol,
        keys: Vec<Vec<DataValue>>,
        vt: ValidityTs,
        hard: bool,
    },
    /// `::lineage rel [[k..]] (@ T)?`
    Lineage {
        rel: Symbol,
        keys: Vec<Vec<DataValue>>,
        vt: ValidityTs,
    },
}

impl MemoryOp {
    /// Relations an imperative script must write-lock (and open a write
    /// transaction for) when it contains this op. Cascades reach relations
    /// unknown at parse time; those take their locks at execution.
    pub(crate) fn write_targets(&self) -> Vec<SmartString<LazyCompact>> {
        use crate::runtime::memory::{PROVENANCE_CATALOG, SUPERSEDE_CATALOG};
        match self {
            MemoryOp::SupersedeCreate { .. } | MemoryOp::SupersedeDrop(_) => {
                vec![SmartString::from(SUPERSEDE_CATALOG)]
            }
            MemoryOp::ProvenanceCreate { .. } | MemoryOp::ProvenanceDrop(_) => {
                vec![SmartString::from(PROVENANCE_CATALOG)]
            }
            MemoryOp::RemoveSource { rel, hard, .. } => {
                let mut v = vec![rel.name.clone(), SmartString::from(PROVENANCE_CATALOG)];
                if *hard {
                    v.push(SmartString::from("kleos_evict_audit"));
                }
                v
            }
            MemoryOp::SupersedeList
            | MemoryOp::ProvenanceList
            | MemoryOp::Changes { .. }
            | MemoryOp::Lineage { .. } => vec![],
        }
    }
}

fn keys_of(expr: Expr) -> Result<Vec<Vec<DataValue>>> {
    match expr.eval_to_const()? {
        DataValue::List(keys) => Ok(keys
            .into_iter()
            .map(|k| match k {
                DataValue::List(parts) => parts,
                v => vec![v],
            })
            .collect()),
        _ => bail!("expected a list of keys, e.g. [['alice', 'works_at']]"),
    }
}

/// A relation named by an option value: bare (`derived: facts`, a binding)
/// or quoted (`derived: 'facts'`).
fn relation_name(
    pair: Pair<'_>,
    param_pool: &BTreeMap<String, DataValue>,
) -> Result<SmartString<LazyCompact>> {
    let mut expr = build_expr(pair, param_pool)?;
    expr.partial_eval()?;
    if let Expr::Binding { var, .. } = expr {
        return Ok(var.name);
    }
    match expr.eval_to_const()? {
        DataValue::Str(s) => Ok(s),
        _ => bail!("expected a relation name, bare or quoted"),
    }
}

fn rel_symbol(pair: Pair<'_>) -> Symbol {
    Symbol::new(pair.as_str(), pair.extract_span())
}

fn optional_at(
    pair: Option<Pair<'_>>,
    param_pool: &BTreeMap<String, DataValue>,
    cur_vld: ValidityTs,
) -> Result<ValidityTs> {
    match pair {
        None => Ok(cur_vld),
        Some(p) => {
            let e = p.into_inner().next().unwrap();
            expr2vld_spec(build_expr(e, param_pool)?, cur_vld)
        }
    }
}

pub(crate) fn parse_memory_op(
    inner: Pair<'_>,
    param_pool: &BTreeMap<String, DataValue>,
    cur_vld: ValidityTs,
) -> Result<MemoryOp> {
    Ok(match inner.as_rule() {
        Rule::supersede_op => {
            let op = inner.into_inner().next().unwrap();
            match op.as_rule() {
                Rule::supersede_create => {
                    let mut it = op.into_inner();
                    let rel = rel_symbol(it.next().unwrap());
                    let cols = it.map(rel_symbol).collect();
                    MemoryOp::SupersedeCreate { rel, cols }
                }
                Rule::supersede_drop => {
                    MemoryOp::SupersedeDrop(rel_symbol(op.into_inner().next().unwrap()))
                }
                Rule::supersede_list => MemoryOp::SupersedeList,
                r => unreachable!("{:?}", r),
            }
        }
        Rule::changes_op => {
            let mut it = inner.into_inner();
            let rel = rel_symbol(it.next().unwrap());
            let mut from = None;
            let mut to = None;
            let mut as_of = None;
            let mut prefix = vec![];
            for opt in it {
                let mut o = opt.into_inner();
                let name = o.next().unwrap();
                let val = build_expr(o.next().unwrap(), param_pool)?;
                match name.as_str() {
                    "from" => from = Some(expr2vld_spec(val, cur_vld)?),
                    "to" => to = Some(expr2vld_spec(val, cur_vld)?),
                    "as_of" => {
                        // transaction time: "NOW"/"END" = current belief
                        as_of = Some(match val.clone().eval_to_const()? {
                            DataValue::Str(s) if s == "NOW" || s == "END" => MAX_VALIDITY_TS,
                            _ => expr2vld_spec(val, cur_vld)?,
                        })
                    }
                    "prefix" => {
                        prefix = match val.eval_to_const()? {
                            DataValue::List(l) => l,
                            v => vec![v],
                        }
                    }
                    other => bail!(
                        "unknown option '{other}' for `::changes`: expected from, to, as_of or prefix"
                    ),
                }
            }
            let from = from.ok_or_else(|| miette!("`::changes` requires a `from` time"))?;
            let to = to.unwrap_or(cur_vld);
            MemoryOp::Changes {
                rel,
                from,
                to,
                as_of,
                prefix,
            }
        }
        Rule::provenance_op => {
            let op = inner.into_inner().next().unwrap();
            match op.as_rule() {
                Rule::provenance_create => {
                    let mut it = op.into_inner();
                    let link = rel_symbol(it.next().unwrap());
                    let mut derived = None;
                    let mut source = None;
                    for opt in it {
                        let mut o = opt.into_inner();
                        let name = o.next().unwrap();
                        let slot = match name.as_str() {
                            "derived" => &mut derived,
                            "source" => &mut source,
                            other => bail!(
                                "unknown option '{other}' for `::provenance create`: \
                                 expected 'derived' and 'source'"
                            ),
                        };
                        *slot = Some(relation_name(o.next().unwrap(), param_pool)?);
                    }
                    let (Some(derived), Some(source)) = (derived, source) else {
                        bail!("`::provenance create` requires both `derived` and `source`");
                    };
                    MemoryOp::ProvenanceCreate {
                        link,
                        derived,
                        source,
                    }
                }
                Rule::provenance_drop => {
                    MemoryOp::ProvenanceDrop(rel_symbol(op.into_inner().next().unwrap()))
                }
                Rule::provenance_list => MemoryOp::ProvenanceList,
                r => unreachable!("{:?}", r),
            }
        }
        Rule::retract_op | Rule::erase_op => {
            let hard = inner.as_rule() == Rule::erase_op;
            let mut it = inner.into_inner();
            let rel = rel_symbol(it.next().unwrap());
            let keys = keys_of(build_expr(it.next().unwrap(), param_pool)?)?;
            let vt = optional_at(it.next(), param_pool, cur_vld)?;
            MemoryOp::RemoveSource {
                rel,
                keys,
                vt,
                hard,
            }
        }
        Rule::lineage_op => {
            let mut it = inner.into_inner();
            let rel = rel_symbol(it.next().unwrap());
            let keys = keys_of(build_expr(it.next().unwrap(), param_pool)?)?;
            let vt = optional_at(it.next(), param_pool, cur_vld)?;
            MemoryOp::Lineage { rel, keys, vt }
        }
        r => unreachable!("{:?}", r),
    })
}
