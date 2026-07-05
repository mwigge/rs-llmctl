use super::*;
use chrono::Utc;
use serde_json::json;

#[test]
fn langfuse_exporter_derives_otlp_endpoint_and_basic_auth_header() {
    let cfg = LangfuseExporterConfig {
        enabled: true,
        host: Some("https://cloud.langfuse.com/".to_string()),
        public_key: Some("pk-lf-abc".to_string()),
        secret_key: Some("sk-lf-xyz".to_string()),
    };

    let exporter = langfuse_otlp_exporter(&cfg)
        .expect("langfuse exporter result should be Ok")
        .expect("langfuse exporter should be derived");
    match exporter {
        Exporter::Otlp {
            endpoint,
            protocol,
            headers,
            ..
        } => {
            assert_eq!(endpoint, "https://cloud.langfuse.com/api/public/otel");
            assert_eq!(protocol, OtlpProtocol::HttpProtobuf);
            assert_eq!(
                headers.get("Authorization").map(String::as_str),
                Some("Basic cGstbGYtYWJjOnNrLWxmLXh5eg==")
            );
        }
        Exporter::None => panic!("expected an OTLP exporter"),
    }
}

#[test]
fn langfuse_exporter_is_none_when_disabled_or_missing_keys() {
    assert!(langfuse_otlp_exporter(&LangfuseExporterConfig::default())
        .expect("disabled exporter should be Ok")
        .is_none());

    let missing_secret = LangfuseExporterConfig {
        enabled: true,
        host: Some("https://cloud.langfuse.com".to_string()),
        public_key: Some("pk-lf-abc".to_string()),
        secret_key: None,
    };
    assert!(langfuse_otlp_exporter(&missing_secret)
        .expect("missing secret should be Ok(None)")
        .is_none());

    let missing_host = LangfuseExporterConfig {
        enabled: true,
        host: None,
        public_key: Some("pk-lf-abc".to_string()),
        secret_key: Some("sk-lf-xyz".to_string()),
    };
    assert!(langfuse_otlp_exporter(&missing_host)
        .expect("missing host should be Ok(None)")
        .is_none());
}

#[test]
fn langfuse_exporter_rejects_host_with_injected_path() {
    // A host value with an embedded path must not silently redirect OTLP traffic.
    let cfg = LangfuseExporterConfig {
        enabled: true,
        host: Some("https://legit.langfuse.com/injected-path".to_string()),
        public_key: Some("pk-lf-abc".to_string()),
        secret_key: Some("sk-lf-xyz".to_string()),
    };
    let exporter = langfuse_otlp_exporter(&cfg)
        .expect("parse should succeed")
        .expect("should produce an exporter");
    // The endpoint must use only the origin — the injected path must be stripped.
    match exporter {
        Exporter::Otlp { endpoint, .. } => {
            assert_eq!(endpoint, "https://legit.langfuse.com/api/public/otel");
        }
        Exporter::None => panic!("expected an OTLP exporter"),
    }
}

#[test]
fn runtime_telemetry_event_names_cover_runtime_surfaces() {
    assert_eq!(
        TelemetryEventName::RequestRouting.as_str(),
        "llmctl.request.routing"
    );
    assert_eq!(
        TelemetryEventName::QuotaDecision.as_str(),
        "llmctl.quota.decision"
    );
    assert_eq!(
        TelemetryEventName::WorkerLifecycle.as_str(),
        "llmctl.worker.lifecycle"
    );
    assert_eq!(
        TelemetryEventName::CircuitBreaker.as_str(),
        "llmctl.upstream.circuit_breaker"
    );
    assert_eq!(
        TelemetryEventName::ResourceSnapshot.as_str(),
        "llmctl.resource.snapshot"
    );
    assert_eq!(
        TelemetryEventName::DriftObservation.as_str(),
        "llmctl.drift.observation"
    );
    assert_eq!(
        TelemetryEventName::ModelInstallVerification.as_str(),
        "llmctl.model.install.verification"
    );
    assert_eq!(
        TelemetryEventName::NativeRuntimeStatus.as_str(),
        "llmctl.runtime.native.status"
    );
    assert_eq!(
        TelemetryEventName::RuntimeHeartbeat.as_str(),
        "llmctl.runtime.heartbeat"
    );
}

#[test]
fn telemetry_batch_serializes_events_without_network_exporter() {
    let mut attrs = BTreeMap::new();
    attrs.insert("route".to_string(), json!("/v1/chat/completions"));
    attrs.insert("quota.allowed".to_string(), json!(true));

    let event = RuntimeTelemetryEvent::new(
        TelemetrySignal::Metric,
        TelemetryEventName::QuotaDecision,
        Utc::now(),
        attrs,
    );
    let batch = TelemetryBatch::new(vec![event]);

    let serialized = serde_json::to_value(&batch).expect("batch serializes");
    assert_eq!(serialized["events"][0]["signal"], "metric");
    assert_eq!(serialized["events"][0]["name"], "llmctl.quota.decision");
    assert_eq!(
        serialized["events"][0]["attributes"]["route"],
        "/v1/chat/completions"
    );
    assert_eq!(serialized["events"][0]["attributes"]["quota.allowed"], true);
}

#[test]
fn runtime_telemetry_emitter_accepts_sanitized_span_attributes() {
    let mut attrs = BTreeMap::new();
    attrs.insert("llmctl.model".to_string(), json!("qwen"));
    attrs.insert("llmctl.allowed".to_string(), json!(true));
    attrs.insert("llmctl.tokens".to_string(), json!(42));
    let event = RuntimeTelemetryEvent::new(
        TelemetrySignal::Span,
        TelemetryEventName::RequestRouting,
        Utc::now(),
        attrs,
    );

    emit_runtime_telemetry(&event);
}

#[test]
fn telemetry_attribute_sanitizer_redacts_secret_and_content_values() {
    let mut attrs = BTreeMap::new();
    attrs.insert("authorization".to_string(), json!("Bearer collector-token"));
    attrs.insert("api_key".to_string(), json!("sk-live-secret"));
    attrs.insert("prompt".to_string(), json!("tell me the admin password"));
    attrs.insert(
        "messages".to_string(),
        json!([{ "role": "user", "content": "private" }]),
    );
    attrs.insert(
        "otel.exporter.otlp.headers".to_string(),
        json!("Authorization=Bearer abc"),
    );
    attrs.insert(
        "model.path".to_string(),
        json!("/home/alice/.cache/llmctl/model.gguf"),
    );
    attrs.insert("route".to_string(), json!("/v1/chat/completions"));
    attrs.insert("quota.allowed".to_string(), json!(false));

    let sanitized = sanitize_otel_attributes(attrs);

    assert_eq!(sanitized.get("authorization"), Some(&json!("[REDACTED]")));
    assert_eq!(sanitized.get("api_key"), Some(&json!("[REDACTED]")));
    assert_eq!(sanitized.get("prompt"), Some(&json!("[REDACTED]")));
    assert_eq!(sanitized.get("messages"), Some(&json!("[REDACTED]")));
    assert_eq!(
        sanitized.get("otel.exporter.otlp.headers"),
        Some(&json!("[REDACTED]"))
    );
    assert_eq!(sanitized.get("model.path"), Some(&json!("[REDACTED]")));
    assert_eq!(sanitized.get("route"), Some(&json!("/v1/chat/completions")));
    assert_eq!(sanitized.get("quota.allowed"), Some(&json!(false)));
}

#[test]
fn inject_trace_context_is_noop_with_no_active_span() {
    let client = reqwest::Client::new();
    let builder = client.get("http://localhost/");
    let result = inject_trace_context(builder);
    // Should return without panic; we verify by successfully building the request.
    let request = result
        .build()
        .expect("request builds after inject_trace_context");
    assert!(!request.headers().contains_key("traceparent"));
}

#[test]
fn install_plan_with_no_exporter_succeeds_without_otel_layer() {
    let plan = ObservabilityPlan {
        service_name: "test".to_string(),
        service_version: None,
        environment: None,
        traces_enabled: false,
        metrics_enabled: false,
        logs_enabled: false,
        resource_attributes: BTreeMap::new(),
        exporter: Exporter::None,
    };
    let runtime = TelemetryRuntime::from_plan(&plan).expect("from_plan with None exporter");
    assert!(runtime.tracer_provider.is_none());
    assert!(runtime.meter_provider.is_none());
    assert!(runtime.logger_provider.is_none());
}

#[test]
fn validate_http_endpoint_accepts_http_and_https() {
    assert!(validate_http_endpoint("http://otel-collector:4318").is_ok());
    assert!(validate_http_endpoint("https://ingest.signoz.io:443").is_ok());
    assert!(validate_http_endpoint("https://collector.internal:4318/v1/traces").is_ok());
}

#[test]
fn validate_http_endpoint_rejects_non_http_scheme() {
    let err = validate_http_endpoint("file:///etc/secrets/dump").unwrap_err();
    assert!(err.to_string().contains("file"), "error: {err}");

    assert!(validate_http_endpoint("grpc://collector:4317").is_err());
    assert!(validate_http_endpoint("ftp://collector:21").is_err());
}

#[test]
fn validate_http_endpoint_rejects_missing_scheme() {
    assert!(validate_http_endpoint("collector.internal:4318").is_err());
    assert!(validate_http_endpoint("/etc/otel.sock").is_err());
}

#[test]
fn validate_http_endpoint_blocks_aws_azure_gcp_imds() {
    // 169.254.169.254 is link-local and is now blocked by IP range check.
    let err = validate_http_endpoint("http://169.254.169.254/latest/meta-data").unwrap_err();
    assert!(err.to_string().contains("169.254.169.254"), "error: {err}");
}

#[test]
fn validate_http_endpoint_blocks_gcp_metadata_hostname() {
    let err =
        validate_http_endpoint("http://metadata.google.internal/computeMetadata/v1/").unwrap_err();
    assert!(
        err.to_string().contains("metadata.google.internal"),
        "error: {err}"
    );

    assert!(validate_http_endpoint("http://metadata.goog/").is_err());
    assert!(validate_http_endpoint("http://instance-data/").is_err());
}

#[test]
fn validate_http_endpoint_block_is_case_insensitive() {
    assert!(validate_http_endpoint("http://METADATA.GOOG/").is_err());
    assert!(validate_http_endpoint("http://Metadata.Google.Internal/").is_err());
    assert!(validate_http_endpoint("HTTP://169.254.169.254/").is_err());
}

#[test]
fn validate_http_endpoint_standalone_without_config() {
    assert!(validate_http_endpoint("http://169.254.169.254/").is_err());
    assert!(validate_http_endpoint("https://collector.internal:4318").is_ok());
}

#[test]
fn validate_http_endpoint_handles_ipv6_bracketed_address() {
    // Loopback IPv6 must be rejected.
    assert!(validate_http_endpoint("http://[::1]:4318").is_err());
    // Non-special global unicast IPv6 must be accepted.
    assert!(validate_http_endpoint("http://[2001:db8::1]:4318").is_ok());
    // ULA (fc00::/7) must be rejected.
    assert!(validate_http_endpoint("http://[fd00:ec2::254]/latest").is_err());
    assert!(validate_http_endpoint("http://[fc00::1]/path").is_err());
}

#[test]
fn validate_http_endpoint_rejects_userinfo_in_url() {
    // Userinfo can smuggle a blocked host as the "password" field.
    assert!(validate_http_endpoint("https://x:@169.254.169.254/meta").is_err());
    assert!(validate_http_endpoint("https://user:pass@internal.service/").is_err());
    assert!(validate_http_endpoint("http://admin@localhost/").is_err());
}

#[test]
fn validate_http_endpoint_rejects_loopback_and_private_ips() {
    assert!(validate_http_endpoint("http://127.0.0.1:4318").is_err());
    assert!(validate_http_endpoint("http://[::1]:4318").is_err());
    assert!(validate_http_endpoint("http://0.0.0.0:4318").is_err());
    assert!(validate_http_endpoint("http://10.0.0.1:4318").is_err());
    assert!(validate_http_endpoint("http://192.168.1.1:4318").is_err());
    assert!(validate_http_endpoint("http://172.16.0.1:4318").is_err());
    assert!(validate_http_endpoint("http://100.100.100.200:4318").is_err()); // Alibaba IMDS
    assert!(validate_http_endpoint("http://169.254.1.1:4318").is_err()); // link-local (not just .169.254)
    assert!(validate_http_endpoint("http://[fd00:ec2::254]:4318").is_err());
    assert!(validate_http_endpoint("http://[fc00::1]:4318").is_err());
    assert!(validate_http_endpoint("http://[fe80::1]:4318").is_err()); // IPv6 link-local
}

#[test]
fn from_config_rejects_blocked_langfuse_host() {
    use crate::config::{LangfuseExporterConfig, ObservabilityConfig, ObservabilityExporterConfig};
    use std::collections::BTreeMap;

    // Build a minimal Config with a Langfuse host that resolves to a blocked address.
    // We use a host that starts with the metadata IP so the derived OTLP endpoint
    // will be rejected by validate_http_endpoint.
    let cfg = Config {
        observability: ObservabilityConfig {
            otlp_endpoint: None,
            service_name: None,
            service_version: None,
            environment: None,
            traces_enabled: true,
            metrics_enabled: true,
            logs_enabled: true,
            resource_attributes: BTreeMap::new(),
            exporter: ObservabilityExporterConfig {
                endpoint: None,
                ..ObservabilityExporterConfig::default()
            },
            langfuse: LangfuseExporterConfig {
                enabled: true,
                host: Some("http://169.254.169.254".to_string()),
                public_key: Some("pk-lf-test".to_string()),
                secret_key: Some("sk-lf-test".to_string()),
            },
            webhook: crate::config::WebhookExporterConfig::default(),
            gen_ai: crate::config::GenAiObservabilityConfig::default(),
        },
        ..Default::default()
    };

    let result = ObservabilityPlan::from_config(&cfg);
    assert!(
        result.is_err(),
        "expected SSRF block for langfuse host pointing to IMDS"
    );
}

#[test]
fn sse_content_parser_counts_output_deltas_without_thinking_phase() {
    let mut parser = SseContentParser::default();
    parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n");
    parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n");
    assert_eq!(parser.output_deltas(), 2);
    assert_eq!(parser.thinking_deltas(), 0);
}

#[test]
fn sse_content_parser_separates_thinking_from_output_deltas() {
    let mut parser = SseContentParser::default();
    parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n");
    parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"<think>\"}}]}\n\n");
    parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"reasoning\"}}]}\n\n");
    parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"</think>\"}}]}\n\n");
    parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n");
    assert_eq!(parser.output_deltas(), 2); // "Hello" + "answer"
    assert_eq!(parser.thinking_deltas(), 1); // "reasoning"
}

#[test]
fn sse_content_parser_handles_think_tag_split_across_chunks() {
    let mut parser = SseContentParser::default();
    // "<think>" split across two SSE frames: the first partial chunk is
    // counted as output because the tag cannot be detected until the second
    // chunk arrives and completes it.
    parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"<th\"}}]}\n\n");
    parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"ink>\"}}]}\n\n");
    parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"deep thought\"}}]}\n\n");
    parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"</think>\"}}]}\n\n");
    parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}]}\n\n");
    assert_eq!(parser.thinking_deltas(), 1); // "deep thought"
                                             // "<th" is counted as output (partial tag not yet detected), "done" too.
    assert_eq!(parser.output_deltas(), 2);
}

#[test]
fn sse_content_parser_ignores_done_sentinel() {
    let mut parser = SseContentParser::default();
    parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n");
    parser.push(b"data: [DONE]\n\n");
    assert_eq!(parser.output_deltas(), 1);
    assert_eq!(parser.thinking_deltas(), 0);
}

#[test]
fn emit_gen_ai_thinking_metrics_does_not_panic_on_zero_deltas() {
    emit_gen_ai_thinking_metrics("test-model", 0, 0);
}

#[test]
fn gen_ai_kv_cache_usage_ratio_gauge_returns_stable_reference() {
    let g1 = gen_ai_kv_cache_usage_ratio_gauge();
    let g2 = gen_ai_kv_cache_usage_ratio_gauge();
    // Both references must point to the same static gauge (pointer equality).
    assert!(std::ptr::eq(g1, g2));
    // Recording must not panic.
    g1.record(
        0.42,
        &[opentelemetry::KeyValue::new("gen_ai.request.model", "test")],
    );
}

#[test]
fn emit_gen_ai_thinking_phase_started_does_not_panic() {
    emit_gen_ai_thinking_phase_started("test-model", 0);
}

#[test]
fn emit_gen_ai_thinking_phase_ended_does_not_panic() {
    emit_gen_ai_thinking_phase_ended("test-model", 128, 1.5);
}
