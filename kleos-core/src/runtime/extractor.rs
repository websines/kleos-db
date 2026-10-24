/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! `::extractor list | enable | disable` (process-wide, no transaction).

use miette::Result;

use crate::bail;
use crate::data::value::DataValue;
use crate::extract::{self, ExtractorStatus};
use crate::parse::sys::ExtractorOp;
use crate::runtime::db::NamedRows;

fn status(s: &str) -> NamedRows {
    NamedRows::new(vec!["status".to_string()], vec![vec![DataValue::from(s)]])
}

fn list_rows(list: Vec<ExtractorStatus>) -> NamedRows {
    NamedRows::new(
        ["name", "model", "version", "labels", "enabled", "loaded"].iter().map(|s| s.to_string()).collect(),
        list.into_iter()
            .map(|x| {
                vec![
                    DataValue::from(x.name.as_str()),
                    DataValue::from(x.model.as_str()),
                    DataValue::from(x.version.as_str()),
                    DataValue::List(x.labels.iter().map(|l| DataValue::from(l.as_str())).collect()),
                    DataValue::Bool(x.enabled),
                    x.loaded.map(DataValue::Bool).unwrap_or(DataValue::Null),
                ]
            })
            .collect(),
    )
}

pub(crate) fn run_extractor_op(op: &ExtractorOp, read_only: bool) -> Result<NamedRows> {
    if read_only && !matches!(op, ExtractorOp::List) {
        bail!("`::extractor enable/disable` cannot run in read-only mode");
    }
    match op {
        ExtractorOp::List => Ok(list_rows(extract::extractors())),
        ExtractorOp::Enable(name, spec) => {
            extract::enable_extractor(name, spec.clone())?;
            Ok(status("OK"))
        }
        ExtractorOp::Disable(name) => Ok(status(if extract::disable_extractor(name)? {
            "OK"
        } else {
            "ALREADY_DISABLED"
        })),
    }
}
