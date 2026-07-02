use super::*;

#[test]
fn readiness_status_reports_draining_as_not_ready() {
    let status = readiness_status_for(&Config::default(), true, true);
    assert_eq!(status["status"], "draining");
    assert_eq!(status["draining"], true);
}

#[test]
fn usage_span_attributes_align_with_gen_ai_semantic_conventions() {
    let event = UsageEvent {
        id: Uuid::new_v4(),
        request_id: Uuid::new_v4(),
        at: Utc::now(),
        model: "llama".to_string(),
        actor: "alice".to_string(),
        team: "platform".to_string(),
        input_tokens: 11,
        output_tokens: 13,
        latency_ms: 42,
        status: "ok".to_string(),
    };

    let attrs = usage_span_attributes(&event, "estimated", "openai");

    assert_eq!(attrs["gen_ai.system"], json!("openai"));
    assert_eq!(attrs["gen_ai.operation.name"], json!("chat"));
    assert_eq!(attrs["gen_ai.request.model"], json!("llama"));
    assert_eq!(attrs["gen_ai.response.model"], json!("llama"));
    assert_eq!(attrs["gen_ai.usage.input_tokens"], json!(11));
    assert_eq!(attrs["gen_ai.usage.output_tokens"], json!(13));
    // Existing llmctl-prefixed attributes must be preserved alongside the
    // gen_ai.* alignment additions.
    assert_eq!(attrs["llmctl.model"], json!("llama"));
    assert_eq!(attrs["llmctl.token_accounting.mode"], json!("estimated"));
}

#[test]
fn webhook_payload_carries_usage_and_accounting_metadata() {
    let event = UsageEvent {
        id: Uuid::new_v4(),
        request_id: Uuid::new_v4(),
        at: Utc::now(),
        model: "llama".to_string(),
        actor: "alice".to_string(),
        team: "platform".to_string(),
        input_tokens: 11,
        output_tokens: 13,
        latency_ms: 42,
        status: "ok".to_string(),
    };

    let payload = webhook_payload(&event, "estimated");

    assert_eq!(payload["type"], json!("llmctl.usage"));
    assert_eq!(payload["request_id"], json!(event.request_id.to_string()));
    assert_eq!(payload["model"], json!("llama"));
    assert_eq!(payload["actor"], json!("alice"));
    assert_eq!(payload["team"], json!("platform"));
    assert_eq!(payload["input_tokens"], json!(11));
    assert_eq!(payload["output_tokens"], json!(13));
    assert_eq!(payload["latency_ms"], json!(42));
    assert_eq!(payload["status"], json!("ok"));
    assert_eq!(payload["token_accounting_mode"], json!("estimated"));
}

#[test]
fn request_id_from_headers_accepts_valid_uuid() {
    let request_id = Uuid::new_v4();
    let mut headers = HeaderMap::new();
    headers.insert(
        request_id_header_name(),
        HeaderValue::from_str(&request_id.to_string()).unwrap(),
    );

    assert_eq!(request_id_from_headers(&headers), request_id);
}

#[test]
fn request_id_from_headers_generates_uuid_when_missing_or_invalid() {
    let missing = request_id_from_headers(&HeaderMap::new());
    assert_ne!(missing, Uuid::nil());

    let mut headers = HeaderMap::new();
    headers.insert(
        request_id_header_name(),
        HeaderValue::from_static("not-a-uuid"),
    );
    let invalid = request_id_from_headers(&headers);
    assert_ne!(invalid, Uuid::nil());
    assert_ne!(invalid, missing);
}

fn make_usage_event(input_tokens: u64, output_tokens: u64) -> UsageEvent {
    UsageEvent {
        id: Uuid::new_v4(),
        request_id: Uuid::new_v4(),
        at: Utc::now(),
        model: "test-model".to_string(),
        actor: "test-actor".to_string(),
        team: "test-team".to_string(),
        input_tokens,
        output_tokens,
        latency_ms: 42,
        status: "ok".to_string(),
    }
}

#[test]
fn usage_span_attributes_include_genai_semconv_and_llmctl_attributes() {
    let event = make_usage_event(100, 50);
    let attrs = usage_span_attributes(&event, "exact", "vertex_ai");

    // GenAI SemConv attributes present
    assert_eq!(
        attrs.get("gen_ai.system").and_then(|v| v.as_str()),
        Some("vertex_ai")
    );
    assert_eq!(
        attrs.get("gen_ai.request.model").and_then(|v| v.as_str()),
        Some("test-model")
    );
    assert_eq!(
        attrs
            .get("gen_ai.usage.input_tokens")
            .and_then(|v| v.as_u64()),
        Some(100)
    );
    assert_eq!(
        attrs
            .get("gen_ai.usage.output_tokens")
            .and_then(|v| v.as_u64()),
        Some(50)
    );

    // Existing llmctl attributes preserved (no regression)
    assert!(attrs.contains_key("llmctl.model"));
    assert!(attrs.contains_key("llmctl.request_id"));
    assert!(attrs.contains_key("llmctl.latency_ms"));
}

#[test]
fn usage_span_attributes_has_no_cost_usd_when_pricing_unknown() {
    let event = make_usage_event(100, 50);
    let attrs = usage_span_attributes(&event, "exact", "vertex_ai");
    // UsageEvent has no cost field; attribute must be absent, not zero
    assert!(!attrs.contains_key("gen_ai.usage.cost_usd"));
}
