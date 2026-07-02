use super::GenAiRequestParams;
use opentelemetry::global;
use opentelemetry::trace::{Span, SpanKind, Tracer};
use opentelemetry::KeyValue;

/// Emits a `gen_ai.chat` lifecycle span covering one complete inference request.
///
/// Span attributes follow the OTel GenAI semantic conventions.  Message content
/// is included only when `capture_content` is `true`; otherwise the body is
/// replaced with `[REDACTED]`.
#[allow(clippy::too_many_arguments)] // all parameters carry distinct domain meaning
pub(super) fn emit_gen_ai_inference_span(
    model: &str,
    gen_ai: &GenAiRequestParams,
    input_tokens: u64,
    output_tokens: u64,
    thinking_deltas: u64,
    output_deltas: u64,
    started: std::time::Instant,
    first_token_instant: Option<std::time::Instant>,
    last_token_instant: Option<std::time::Instant>,
    status: &str,
    capture_content: bool,
) {
    let tracer = global::tracer(crate::SERVICE_NAME);
    let start_system_time = std::time::SystemTime::now()
        .checked_sub(started.elapsed())
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
    let mut span = tracer
        .span_builder("gen_ai.chat")
        .with_kind(SpanKind::Server)
        .with_start_time(start_system_time)
        .start(&tracer);

    span.set_attribute(KeyValue::new("gen_ai.system", "local"));
    span.set_attribute(KeyValue::new("gen_ai.operation.name", "chat"));
    span.set_attribute(KeyValue::new("gen_ai.request.model", model.to_string()));
    span.set_attribute(KeyValue::new("gen_ai.response.model", model.to_string()));
    span.set_attribute(KeyValue::new(
        "gen_ai.usage.input_tokens",
        i64::try_from(input_tokens).unwrap_or(i64::MAX),
    ));
    span.set_attribute(KeyValue::new(
        "gen_ai.usage.output_tokens",
        i64::try_from(output_tokens).unwrap_or(i64::MAX),
    ));
    span.set_attribute(KeyValue::new(
        "gen_ai.usage.thinking_tokens",
        i64::try_from(thinking_deltas).unwrap_or(i64::MAX),
    ));
    span.set_attribute(KeyValue::new(
        "gen_ai.usage.output_deltas",
        i64::try_from(output_deltas).unwrap_or(i64::MAX),
    ));
    span.set_attribute(KeyValue::new("llmctl.status", status.to_string()));

    if let Some(max_tokens) = gen_ai.max_tokens {
        span.set_attribute(KeyValue::new(
            "gen_ai.request.max_tokens",
            i64::from(max_tokens),
        ));
    }
    if let Some(temp) = gen_ai.temperature {
        span.set_attribute(KeyValue::new("gen_ai.request.temperature", f64::from(temp)));
    }

    if let (Some(first), Some(last)) = (first_token_instant, last_token_instant) {
        let ttft_secs = first.saturating_duration_since(started).as_secs_f64();
        let decode_secs = last.saturating_duration_since(first).as_secs_f64();
        span.set_attribute(KeyValue::new("gen_ai.ttft_seconds", ttft_secs));
        if decode_secs > 0.0 {
            let throughput = output_deltas as f64 / decode_secs;
            span.set_attribute(KeyValue::new(
                "gen_ai.decode_throughput_deltas_per_second",
                throughput,
            ));
        }
    }

    let redacted = crate::observability::REDACTED_ATTRIBUTE_VALUE;
    if let Some(sys) = &gen_ai.system_message {
        let body = if capture_content {
            sys.as_str()
        } else {
            redacted
        };
        span.add_event(
            "gen_ai.system.message",
            vec![KeyValue::new("body", body.to_string())],
        );
    }
    if let Some(user) = &gen_ai.user_message {
        let body = if capture_content {
            user.as_str()
        } else {
            redacted
        };
        span.add_event(
            "gen_ai.user.message",
            vec![KeyValue::new("body", body.to_string())],
        );
    }

    span.end();
}
