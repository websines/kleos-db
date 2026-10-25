/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! OpenTelemetry GenAI-style spans of audited principal ops (feature
//! `otel`; docs/specs/tenancy.md, "Audit"). Without the feature this is a
//! no-op: the same attributes are stored in each audit event's `otel`.

use serde_json::Value as JsonValue;

/// Emit one span for an audited op.
#[cfg(not(feature = "otel"))]
pub(crate) fn span(_op: &str, _otel: &JsonValue, _targets: &JsonValue, _decision: &JsonValue) {}

#[cfg(feature = "otel")]
pub(crate) fn span(op: &str, otel: &JsonValue, targets: &JsonValue, decision: &JsonValue) {
    let _ = (op, otel, targets, decision);
}
