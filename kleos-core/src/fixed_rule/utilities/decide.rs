/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The `Decide` fixed rule (kleos fork; docs/specs/decide.md): batched
//! decisions over a relation of pairs.
//!
//! ```cozo
//! ?[id, label, p, band, decided_by] <~ Decide(pairs[id, old, new], model: 'nano', template: 'supersedes')
//! ```
//!
//! Input `[id, a, b]` (or `[id, text]` for `important`). Rows whose text is
//! null are skipped. Output `[id, label, p, band, decided_by]`, one row per
//! input row; `decided_by` is `model|template@vN|via`, for storing next to
//! the decision. Rows are sent to the model in batches (a forward pass per
//! batch instead of per row), and query cancellation is checked between
//! batches.

use std::collections::BTreeMap;

use miette::Result;
use smartstring::{LazyCompact, SmartString};

use crate::bail;
use crate::data::expr::Expr;
use crate::data::symb::Symbol;
use crate::data::value::DataValue;
use crate::decide::{decide, Template};
use crate::fixed_rule::{FixedRule, FixedRulePayload};
use crate::parse::SourceSpan;
use crate::runtime::db::Poison;
use crate::runtime::temp_store::RegularTempStore;

/// Rows per call into the decide pipeline.
const ROWS_PER_CALL: usize = 256;

pub(crate) struct Decide;

impl FixedRule for Decide {
    fn run(&self, payload: FixedRulePayload<'_, '_>, out: &mut RegularTempStore, poison: Poison) -> Result<()> {
        let in_rel = payload.get_input(0)?;
        let model = payload.string_option("model", None)?;
        let template = Template::parse(&payload.string_option("template", None)?)?;
        let want = template.arity() + 1;
        let mut rows: Vec<(DataValue, String, String)> = vec![];
        for tuple in in_rel.iter()? {
            let tuple = tuple?;
            if tuple.len() != want {
                bail!(
                    "Decide with template {} expects input rows [id, {}], got a row of arity {}",
                    template.name(),
                    if want == 2 { "text" } else { "a, b" },
                    tuple.len()
                );
            }
            let mut it = tuple.into_iter();
            let id = it.next().unwrap();
            let mut text = |what: &str| -> Result<Option<String>> {
                match it.next() {
                    Some(DataValue::Str(s)) => Ok(Some(s.to_string())),
                    Some(DataValue::Null) => Ok(None),
                    Some(v) => bail!("Decide: {what} must be a string, got {v:?}"),
                    None => Ok(Some(String::new())),
                }
            };
            let a = text("the first text")?;
            let b = if want == 3 { text("the second text")? } else { Some(String::new()) };
            if let (Some(a), Some(b)) = (a, b) {
                rows.push((id, a, b));
            }
            poison.check()?;
        }
        for chunk in rows.chunks(ROWS_PER_CALL) {
            poison.check()?;
            let items: Vec<(&str, &str)> = chunk.iter().map(|r| (r.1.as_str(), r.2.as_str())).collect();
            let decisions = decide(&model, template, &items)?;
            for ((id, _, _), d) in chunk.iter().zip(decisions) {
                out.put(vec![
                    id.clone(),
                    DataValue::from(d.label.as_str()),
                    DataValue::from(d.p),
                    DataValue::from(d.band.name()),
                    DataValue::from(d.decided_by().as_str()),
                ]);
            }
        }
        Ok(())
    }

    fn arity(&self, _options: &BTreeMap<SmartString<LazyCompact>, Expr>, _rule_head: &[Symbol], _span: SourceSpan) -> Result<usize> {
        Ok(5)
    }
}
