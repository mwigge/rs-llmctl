use super::*;

#[test]
fn env_resolution_uses_only_llmctl_names_for_local_client() {
    let value = local_from_env_values(|name| match name {
        "LLMCTL_BASE_URL" => Some("http://llmctl".to_string()),
        "RS_LLMCTL_BASE_URL" => Some("http://rs".to_string()),
        "OPENAI_BASE_URL" => Some("http://openai/v1".to_string()),
        _ => None,
    })
    .expect("base url");

    assert_eq!(value, "http://llmctl");

    let err = local_from_env_values(|name| match name {
        "OPENAI_BASE_URL" => Some("http://openai/v1".to_string()),
        _ => None,
    })
    .expect_err("OpenAI-compatible aliases are explicit provider-only inputs");

    assert!(err.to_string().contains("LLMCTL_BASE_URL"));
}

#[test]
fn normalize_base_url_accepts_openai_v1_base_url() {
    let url = normalize_base_url("http://localhost:8765/v1").expect("url");
    assert_eq!(url.as_str(), "http://localhost:8765/");
}

#[test]
fn ask_config_builds_chat_request_with_history_and_metadata() {
    let request = AskConfig::new("qwen")
        .system("answer briefly")
        .temperature(0.2)
        .max_tokens(64)
        .metadata(serde_json::json!({"session_id": "ask-1"}))
        .to_request(
            Question::new("continue").with_history(vec![ChatMessage::assistant("previous")]),
        )
        .expect("request");

    assert_eq!(request.model, "qwen");
    assert_eq!(request.messages[0].role, "system");
    assert_eq!(request.messages[1].role, "assistant");
    assert_eq!(request.messages[2].content.as_deref(), Some("continue"));
    assert_eq!(request.temperature, Some(0.2));
    assert_eq!(request.max_tokens, Some(64));
    assert_eq!(
        request.metadata,
        Some(serde_json::json!({"session_id": "ask-1"}))
    );
}

#[test]
fn non_local_provider_kinds_are_contract_only_until_server_side_egress_exists() {
    let err = AskConfig {
        provider: ProviderKind::OpenAiCompatible,
        ..AskConfig::new("gpt-4o-mini")
    }
    .to_request(Question::new("hello"))
    .expect_err("external provider request is reserved metadata");

    assert!(err
        .to_string()
        .contains("contract-only metadata and cannot route traffic"));
}

#[test]
fn provider_contract_preserves_local_default_and_marks_external_adapters() {
    let local = ProviderContract::local_llmctl();

    assert_eq!(local.kind, ProviderKind::LocalLlmctl);
    assert_eq!(local.routing, ProviderRouting::LocalOnly);
    assert_eq!(local.status, ProviderStatus::Implemented);
    assert!(local.local_first);
    assert!(!local.routes_external_provider_traffic);
    local.validate_routable().expect("local llmctl is routable");

    for provider in [
        ProviderKind::OpenAiCompatible,
        ProviderKind::VertexAi,
        ProviderKind::OpenRouter,
    ] {
        let contract = ProviderContract::for_kind(provider.clone());
        assert_eq!(contract.kind, provider);
        assert_eq!(contract.routing, ProviderRouting::ExternalReserved);
        assert_eq!(contract.status, ProviderStatus::ContractOnly);
        assert!(!contract.routes_external_provider_traffic);
        assert!(contract.base_url_env.is_empty());
        assert!(contract.api_key_env.is_empty());
        contract
            .validate_routable()
            .expect_err("external providers are reserved metadata in the native-only client");
    }
}

#[test]
fn provider_client_rejects_external_provider_bypass() {
    let err = client_from_provider_env_values(ProviderKind::OpenRouter, |name| match name {
        "OPENROUTER_BASE_URL" => Some("https://openrouter.example/api/v1".to_string()),
        "OPENROUTER_API_KEY" => Some("provider-secret".to_string()),
        _ => None,
    })
    .expect_err("external provider bypass is rejected");

    assert!(err
        .to_string()
        .contains("contract-only metadata and cannot route traffic"));
}

#[test]
fn scheduler_contract_serializes_fifo_runtime_with_metadata_only_batching_and_kv_cache() {
    let contract = SchedulerContract::fifo_runtime();
    let serialized = serde_json::to_value(&contract).expect("scheduler contract serializes");

    assert_eq!(serialized["contract_only"], false);
    assert_eq!(serialized["queue"]["implemented"], true);
    assert_eq!(serialized["batching"]["continuous_batching"], false);
    assert_eq!(serialized["batching"]["implemented"], false);
    assert_eq!(
        serialized["kv_cache"]["cache_budget_metadata_key"],
        "llmctl.scheduler.kv_cache_budget_bytes"
    );
    assert_eq!(serialized["kv_cache"]["implemented"], false);
    assert_eq!(serialized["cancellation"]["implemented"], false);
    contract
        .validate_runtime_contract()
        .expect("FIFO scheduler runtime contract validates");
}
