/*
 * Copyright 2026, The Kleos Authors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! OpenTelemetry GenAI-style spans of audited principal ops (feature
//! `otel`; docs/specs/tenancy.md, "Audit"). Each audited op becomes one
//! span of the global tracer `kleos`, named `<op> kleos`, with the same
//! attributes the audit event stores in `otel` (pseudonymous ids only):
//! `gen_ai.operation.name`, `gen_ai.agent.id`, `gen_ai.conversation.id`,
//! `kleos.workspace`, `kleos.scope_set_hash`, plus `kleos.decision` and
//! `kleos.targets` (the number of target ids). The host installs the
//! tracer provider and exporter. Without the feature this is a no-op.

use serde_json::Value as JsonValue;

/// Emit one span for an audited op.
#[cfg(not(feature = "otel"))]
pub(crate) fn span(_op: &str, _otel: &JsonValue, _targets: &JsonValue, _decision: &JsonValue) {}

#[cfg(feature = "otel")]
pub(crate) fn span(op: &str, otel: &JsonValue, targets: &JsonValue, decision: &JsonValue) {
    use opentelemetry::trace::{Span, SpanKind, Tracer};
    use opentelemetry::{global, KeyValue};
    let tracer = global::tracer("kleos");
    let mut attrs = vec![KeyValue::new("gen_ai.operation.name", op.to_string())];
    for k in ["gen_ai.agent.id", "gen_ai.conversation.id", "kleos.workspace", "kleos.scope_set_hash"] {
        if let Some(v) = otel.get(k).and_then(|x| x.as_str()) {
            attrs.push(KeyValue::new(k, v.to_string()));
        }
    }
    if let Some(e) = decision.get("effect").and_then(|x| x.as_str()) {
        attrs.push(KeyValue::new("kleos.decision", e.to_string()));
    }
    let n = targets.get("ids").and_then(|x| x.as_array()).map_or(0, |a| a.len());
    attrs.push(KeyValue::new("kleos.targets", n as i64));
    let mut s = tracer.span_builder(format!("{op} kleos")).with_kind(SpanKind::Internal).with_attributes(attrs).start(&tracer);
    s.end();
}

#[cfg(all(test, feature = "otel"))]
mod tests {
    use std::borrow::Cow;
    use std::sync::{Arc, Mutex};
    use std::time::SystemTime;

    use opentelemetry::trace::{SpanBuilder, SpanContext, Status, TracerProvider};
    use opentelemetry::{Context, InstrumentationScope, KeyValue};

    type Seen = Arc<Mutex<Vec<(String, Vec<KeyValue>)>>>;

    struct P(Seen);
    struct T(Seen);
    struct S(Seen, SpanContext, String, Vec<KeyValue>);

    impl TracerProvider for P {
        type Tracer = T;
        fn tracer_with_scope(&self, _scope: InstrumentationScope) -> T {
            T(self.0.clone())
        }
    }
    impl opentelemetry::trace::Tracer for T {
        type Span = S;
        fn build_with_context(&self, b: SpanBuilder, _cx: &Context) -> S {
            S(self.0.clone(), SpanContext::empty_context(), b.name.to_string(), b.attributes.unwrap_or_default())
        }
    }
    impl opentelemetry::trace::Span for S {
        fn add_event_with_timestamp<N: Into<Cow<'static, str>>>(&mut self, _n: N, _t: SystemTime, _a: Vec<KeyValue>) {}
        fn span_context(&self) -> &SpanContext {
            &self.1
        }
        fn is_recording(&self) -> bool {
            true
        }
        fn set_attribute(&mut self, a: KeyValue) {
            self.3.push(a);
        }
        fn set_status(&mut self, _s: Status) {}
        fn update_name<N: Into<Cow<'static, str>>>(&mut self, n: N) {
            self.2 = n.into().to_string();
        }
        fn add_link(&mut self, _c: SpanContext, _a: Vec<KeyValue>) {}
        fn end_with_timestamp(&mut self, _t: SystemTime) {
            self.0.lock().unwrap().push((self.2.clone(), self.3.clone()));
        }
    }

    #[test]
    fn audited_ops_become_genai_spans() {
        let seen: Seen = Default::default();
        opentelemetry::global::set_tracer_provider(P(seen.clone()));
        let otel = serde_json::json!({"gen_ai.agent.id": "p:abc", "gen_ai.conversation.id": "r1", "kleos.workspace": "eng"});
        super::span("recall", &otel, &serde_json::json!({"ids": ["m1", "m2"]}), &serde_json::json!({"effect": "allow"}));
        let got = seen.lock().unwrap();
        let (name, attrs) = &got[0];
        assert_eq!(name, "recall kleos");
        let get = |k: &str| attrs.iter().find(|a| a.key.as_str() == k).map(|a| a.value.to_string());
        assert_eq!(get("gen_ai.operation.name").as_deref(), Some("recall"));
        assert_eq!(get("gen_ai.agent.id").as_deref(), Some("p:abc"));
        assert_eq!(get("kleos.targets").as_deref(), Some("2"));
        assert_eq!(get("kleos.decision").as_deref(), Some("allow"));
    }
}
