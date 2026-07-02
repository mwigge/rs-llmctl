use super::*;

#[test]
fn candle_native_routes_to_planned_worker_upstream_metadata() {
    let cfg = Config {
        models: vec![ModelConfig {
            alias: "llama".to_string(),
            path: "/models/llama.gguf".into(),
            role: "chat".to_string(),
            family: Some("qwen3".to_string()),
            weight: 1,
        }],
        ..Default::default()
    };

    assert_eq!(
        serving_upstreams(&cfg).get("llama").map(String::as_str),
        Some("http://127.0.0.1:18765")
    );
}

#[test]
fn lists_configured_models_in_deterministic_order() {
    let cfg = config_with_models(
        Mode::HotSwap,
        vec![
            model("zeta", 1, "chat"),
            model("alpha", 1, "chat"),
            model("middle", 1, "chat"),
        ],
    );

    let aliases: Vec<_> = routed_models(&cfg)
        .iter()
        .map(|model| model.alias.as_str())
        .collect();

    assert_eq!(aliases, vec!["alpha", "middle", "zeta"]);
}

#[test]
fn single_mode_routes_to_the_only_configured_model() {
    let cfg = config_with_models(Mode::Single, vec![model("llama", 0, "chat")]);

    let resolved = resolve_model_route(&cfg, "llama", Uuid::nil()).unwrap();

    assert_eq!(resolved.requested_alias, "llama");
    assert_eq!(resolved.upstream_alias, "llama");
}

#[test]
fn swap_modes_validate_requested_aliases() {
    for mode in [Mode::ColdSwap, Mode::HotSwap] {
        let cfg = config_with_models(
            mode,
            vec![model("alpha", 0, "chat"), model("beta", 0, "chat")],
        );

        assert_eq!(
            resolve_model_route(&cfg, "beta", Uuid::nil())
                .unwrap()
                .upstream_alias,
            "beta"
        );
        assert!(matches!(
            resolve_model_route(&cfg, "missing", Uuid::nil()),
            Err(ModelRouteError::UnknownAlias(alias)) if alias == "missing"
        ));
    }
}

#[test]
fn weighted_mode_selects_model_by_request_id_slot() {
    let cfg = config_with_models(
        Mode::Weighted,
        vec![
            model("light", 1, "chat"),
            model("heavy-b", 50, "chat"),
            model("heavy-a", 50, "chat"),
        ],
    );

    let resolved = resolve_model_route(&cfg, "light", Uuid::from_u128(50)).unwrap();

    assert_eq!(resolved.requested_alias, "light");
    assert_eq!(resolved.upstream_alias, "heavy-b");
}

#[test]
fn cluster_node_routes_only_locally_placed_models() {
    let mut cfg = config_with_models(
        Mode::Weighted,
        vec![
            model("thinking", 100, "thinking"),
            model("coding", 100, "coding"),
        ],
    );
    cfg.cluster.node_id = "node-a".to_string();
    cfg.cluster.nodes = vec![
        ClusterNodeConfig {
            id: "node-a".to_string(),
            base_url: "http://node-a:8765".to_string(),
            roles: vec!["thinking".to_string()],
            model_aliases: Vec::new(),
        },
        ClusterNodeConfig {
            id: "node-b".to_string(),
            base_url: "http://node-b:8765".to_string(),
            roles: vec!["coding".to_string()],
            model_aliases: Vec::new(),
        },
    ];

    let aliases = routed_models(&cfg)
        .into_iter()
        .map(|model| model.alias.as_str())
        .collect::<Vec<_>>();
    assert_eq!(aliases, vec!["thinking"]);
    assert!(matches!(
        resolve_model_route(&cfg, "coding", Uuid::nil()),
        Err(ModelRouteError::UnknownAlias(alias)) if alias == "coding"
    ));
    assert_eq!(
        resolve_model_route(&cfg, "thinking", Uuid::from_u128(1))
            .unwrap()
            .upstream_alias,
        "thinking"
    );
}

#[test]
fn readiness_counts_only_active_routed_models() {
    let cfg = config_with_models(
        Mode::Weighted,
        vec![model("active", 1, "chat"), model("inactive", 0, "chat")],
    );

    let status = readiness_status_for(&cfg, true, false);

    assert_eq!(status["status"], "ready");
    assert_eq!(status["models"]["configured"], 1);
    assert_eq!(status["models"]["aliases"], json!(["active"]));
}

#[test]
fn fallback_mode_routes_zero_weight_models_to_first_positive_weight_model() {
    let cfg = config_with_models(
        Mode::Fallback,
        vec![
            model("primary", 100, "chat"),
            model("backup", 0, "chat"),
            model("tertiary", 10, "chat"),
        ],
    );

    assert_eq!(
        resolve_model_route(&cfg, "backup", Uuid::from_u128(1))
            .unwrap()
            .upstream_alias,
        "primary"
    );
    let tertiary = resolve_model_route(&cfg, "tertiary", Uuid::from_u128(120)).unwrap();
    assert_eq!(tertiary.upstream_alias, "tertiary");
    assert_eq!(tertiary.fallback_aliases, vec!["primary", "backup"]);
}

#[test]
fn rewrites_chat_completion_model_for_upstream_route() {
    let body = br#"{"model":"light","messages":[]}"#;
    let route = ResolvedModelRoute {
        requested_alias: "light".to_string(),
        upstream_alias: "heavy".to_string(),
        fallback_aliases: Vec::new(),
        external_provider: None,
    };

    let rewritten = rewrite_chat_model(body, &route).unwrap();
    let value: Value = serde_json::from_slice(&rewritten).unwrap();

    assert_eq!(value["model"], "heavy");
    assert_eq!(value["messages"], json!([]));
}

fn config_with_models(mode: Mode, models: Vec<ModelConfig>) -> Config {
    Config {
        mode,
        models,
        ..Default::default()
    }
}

fn model(alias: &str, weight: u32, role: &str) -> ModelConfig {
    ModelConfig {
        alias: alias.to_string(),
        path: format!("/models/{alias}.gguf").into(),
        role: role.to_string(),
        family: Some("qwen3".to_string()),
        weight,
    }
}
