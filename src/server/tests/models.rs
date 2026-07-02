use super::*;

#[test]
fn normalizes_upstream_urls() {
    assert_eq!(
        normalize_upstream("http://127.0.0.1:8080/"),
        "http://127.0.0.1:8080"
    );
    assert_eq!(
        normalize_upstream("127.0.0.1:8080"),
        "http://127.0.0.1:8080"
    );
}

#[test]
fn model_capabilities_are_present_on_every_model_entry() {
    let mc = ModelConfig {
        alias: "qwen3-14b-q4_k_m".into(),
        path: std::path::PathBuf::from("/tmp/none.gguf"),
        role: "chat".into(),
        family: Some("qwen3".into()),
        weight: 1,
    };
    let snap = CapabilitySnapshot::current();
    let obj = build_model_object(&mc, snap);
    let json = serde_json::to_value(&obj).unwrap();

    // Standard OpenAI Models fields preserved.
    assert_eq!(json["id"], "qwen3-14b-q4_k_m");
    assert_eq!(json["object"], "model");
    assert_eq!(json["owned_by"], "rs-llmctl");

    // New capability fields all present.
    let caps = &json["capabilities"];
    assert!(caps.is_object(), "capabilities must be an object");
    assert!(caps["context_window"].is_number());
    assert!(caps["tool_protocol"].is_string());
    assert!(caps["model_size_b"].is_number());
    assert!(caps["gpu_backend"].is_string());
    assert!(caps["tier"].is_string());

    // Qwen3 family advertises its native tool protocol.
    assert_eq!(caps["tool_protocol"], "qwen3-native");
    // Alias contained "14b" → size parser extracts 14.0.
    assert_eq!(caps["model_size_b"], 14.0);
    // Qwen3 default context is 128k.
    assert_eq!(caps["context_window"], 131_072);
}

#[test]
fn unknown_family_still_renders_capabilities_with_defaults() {
    let mc = ModelConfig {
        alias: "experimental-model".into(),
        path: std::path::PathBuf::from("/tmp/none.gguf"),
        role: "chat".into(),
        family: None,
        weight: 1,
    };
    let snap = CapabilitySnapshot::current();
    let obj = build_model_object(&mc, snap);
    let json = serde_json::to_value(&obj).unwrap();
    let caps = &json["capabilities"];
    // Unknown family → tool_protocol = "none", context_window = 0, size = 0.0.
    assert_eq!(caps["tool_protocol"], "none");
    assert_eq!(caps["context_window"], 0);
    assert_eq!(caps["model_size_b"], 0.0);
    // gpu_backend and tier are always populated, even for unknown families.
    assert!(caps["gpu_backend"].is_string());
    assert!(caps["tier"].is_string());
}

#[test]
fn alias_size_parser_handles_common_shapes() {
    assert_eq!(parse_model_size_b_from_alias("qwen3-14b-q4_k_m"), 14.0);
    assert_eq!(parse_model_size_b_from_alias("Qwen3-Coder-30B-A3B"), 30.0);
    assert_eq!(parse_model_size_b_from_alias("llama-3.1-8B-instruct"), 8.0);
    assert_eq!(parse_model_size_b_from_alias("phi-3.5-mini-3.8b"), 3.8);
    // No size suffix.
    assert_eq!(parse_model_size_b_from_alias("custom"), 0.0);
    // Implausible numbers are rejected.
    assert_eq!(parse_model_size_b_from_alias("0b"), 0.0);
    assert_eq!(parse_model_size_b_from_alias("99999b"), 0.0);
}

#[test]
fn backward_compat_legacy_openai_client_ignores_capabilities() {
    // A strict OpenAI SDK client deserialises into a struct that only knows
    // about id/object/owned_by. Confirm our payload still validates against
    // that shape (additive — capabilities is an extra field).
    #[derive(serde::Deserialize)]
    struct LegacyModelObject {
        id: String,
        object: String,
        owned_by: String,
    }
    let mc = ModelConfig {
        alias: "qwen3-8b".into(),
        path: std::path::PathBuf::from("/tmp/none.gguf"),
        role: "chat".into(),
        family: Some("qwen3".into()),
        weight: 1,
    };
    let json =
        serde_json::to_value(build_model_object(&mc, CapabilitySnapshot::current())).unwrap();
    let legacy: LegacyModelObject = serde_json::from_value(json).unwrap();
    assert_eq!(legacy.id, "qwen3-8b");
    assert_eq!(legacy.object, "model");
    assert_eq!(legacy.owned_by, "rs-llmctl");
}

#[test]
fn tool_format_openai_for_qwen3_family() {
    let mc = ModelConfig {
        alias: "qwen3-14b".into(),
        path: std::path::PathBuf::from("/tmp/none.gguf"),
        role: "chat".into(),
        family: Some("qwen3".into()),
        weight: 1,
    };
    let json =
        serde_json::to_value(build_model_object(&mc, CapabilitySnapshot::current())).unwrap();
    assert_eq!(json["capabilities"]["tool_format"], "openai");
}

#[test]
fn tool_format_xml_for_devstral_family() {
    let mc = ModelConfig {
        alias: "devstral-small-2505".into(),
        path: std::path::PathBuf::from("/tmp/none.gguf"),
        role: "chat".into(),
        family: Some("mistral".into()),
        weight: 1,
    };
    let json =
        serde_json::to_value(build_model_object(&mc, CapabilitySnapshot::current())).unwrap();
    assert_eq!(json["capabilities"]["tool_format"], "xml");
}

#[test]
fn tool_format_openai_for_gemma4_family() {
    let mc = ModelConfig {
        alias: "gemma4-12b".into(),
        path: std::path::PathBuf::from("/tmp/none.gguf"),
        role: "chat".into(),
        family: Some("gemma4".into()),
        weight: 1,
    };
    let json =
        serde_json::to_value(build_model_object(&mc, CapabilitySnapshot::current())).unwrap();
    assert_eq!(json["capabilities"]["tool_format"], "openai");
}

#[test]
fn playground_html_wires_models_and_chat_endpoints_with_api_key_field() {
    let html = playground_html();
    assert!(html.contains("<title>"));
    assert!(html.contains("/v1/models"));
    assert!(html.contains("/v1/chat/completions"));
    assert!(html.to_lowercase().contains("api key") || html.to_lowercase().contains("api-key"));
    assert!(html.contains("<script"));
}
