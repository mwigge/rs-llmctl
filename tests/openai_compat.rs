use axum::body::{to_bytes, Body};
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures_util::future::{BoxFuture, FutureExt};
use rs_llmctl::config::{ApiKeyConfig, Config, Mode, ModelConfig, QuotaConfig};
use rs_llmctl::native;
use rs_llmctl::runtime::RuntimeBackend;
use rs_llmctl::server;
use rs_llmctl::storage::Storage;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use tower::ServiceExt;
use uuid::Uuid;

const TOKEN: &str = "test-token";
type SharedRelease = std::sync::Arc<tokio::sync::Mutex<Option<oneshot::Receiver<()>>>>;
type BlockingUpstreamState = (mpsc::Sender<Value>, SharedRelease);

struct StaticNativeEngine;

impl native::NativeEngine for StaticNativeEngine {
    fn model_alias(&self) -> &str {
        "llama"
    }

    fn chat(
        &self,
        request: native::NativeChatRequest,
    ) -> BoxFuture<'_, anyhow::Result<native::NativeChatResponse>> {
        async move {
            assert_eq!(request.model, "llama");
            assert_eq!(request.messages.len(), 1);
            Ok(native::NativeChatResponse {
                model: request.model,
                content: "native pong".to_string(),
                finish_reason: "stop".to_string(),
                usage: native::NativeTokenUsage::new(2, 3),
            })
        }
        .boxed()
    }
}

struct CapturingNativeEngine {
    alias: &'static str,
    tx: mpsc::Sender<native::NativeChatRequest>,
}

impl native::NativeEngine for CapturingNativeEngine {
    fn model_alias(&self) -> &str {
        self.alias
    }

    fn chat(
        &self,
        request: native::NativeChatRequest,
    ) -> BoxFuture<'_, anyhow::Result<native::NativeChatResponse>> {
        let tx = self.tx.clone();
        async move {
            tx.send(request.clone())
                .await
                .expect("capture native request");
            Ok(native::NativeChatResponse {
                model: request.model,
                content: "native pong".to_string(),
                finish_reason: "stop".to_string(),
                usage: native::NativeTokenUsage::new(11, 13),
            })
        }
        .boxed()
    }
}

#[tokio::test]
async fn lists_openai_compatible_models() {
    let app = test_app(config_with_models(vec![
        model("zeta"),
        model("alpha"),
        model("middle"),
    ]))
    .await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header("authorization", bearer())
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-llmctl-model-count")
            .and_then(|value| value.to_str().ok()),
        Some("3")
    );
    assert_response_headers_do_not_leak(
        response.headers(),
        &["/models/alpha.gguf", TOKEN, &cfg_api_key_hash()],
    );
    let body = response_json(response).await;
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "alpha");
    assert_eq!(body["data"][0]["object"], "model");
    assert_eq!(body["data"][0]["owned_by"], "rs-llmctl");
    assert_eq!(body["data"][1]["id"], "middle");
    assert_eq!(body["data"][2]["id"], "zeta");
}

#[tokio::test]
async fn models_echoes_inbound_request_id_and_audits_it() {
    let request_id = Uuid::new_v4();
    let (app, storage) = test_app_with_storage(config_with_models(vec![model("llama")])).await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header("authorization", bearer())
                .header("x-request-id", request_id.to_string())
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()["x-request-id"].to_str().unwrap(),
        request_id.to_string()
    );
    let body = response_json(response).await;
    assert_eq!(body["object"], "list");

    let audit_events = storage
        .audit_events_for_request(request_id)
        .await
        .expect("audit events");
    assert_eq!(audit_events.len(), 1);
    assert_eq!(audit_events[0].action, "models.list");
}

#[tokio::test]
async fn models_generates_request_id_header_and_audits_it() {
    let (app, storage) = test_app_with_storage(config_with_models(vec![model("llama")])).await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header("authorization", bearer())
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let request_id = response.headers()["x-request-id"]
        .to_str()
        .unwrap()
        .parse::<Uuid>()
        .expect("generated request id");
    let body = response_json(response).await;
    assert_eq!(body["object"], "list");

    let audit_events = storage
        .audit_events_for_request(request_id)
        .await
        .expect("audit events");
    assert_eq!(audit_events.len(), 1);
    assert_eq!(audit_events[0].request_id, Some(request_id));
}

#[tokio::test]
async fn healthz_remains_compatible() {
    let app = test_app(config_with_models(vec![model("llama")])).await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    assert_eq!(body, json!({ "status": "ok" }));
}

#[tokio::test]
async fn livez_reports_process_liveness_without_auth() {
    let app = test_app(config_with_models(vec![model("llama")])).await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/livez")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    assert_eq!(body, json!({ "status": "ok" }));
}

#[tokio::test]
async fn readyz_reports_model_count_and_storage_without_leaking_config_details() {
    let mut cfg = config_with_models(vec![model("llama"), model("embed")]);
    cfg.mode = Mode::HotSwap;
    cfg.security.bind_external = true;
    let api_key_hash = cfg.security.api_keys[0].sha256.clone();
    let model_path = cfg.models[0].path.display().to_string();
    let app = test_app(cfg).await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body");
    let body: Value = serde_json::from_slice(&bytes).expect("json response");
    assert_eq!(body["status"], "ready");
    assert_eq!(body["mode"], "hot-swap");
    assert_eq!(body["models"]["configured"], 2);
    assert_eq!(body["models"]["aliases"], json!(["embed", "llama"]));
    assert_eq!(body["workers"]["planned"], 2);
    assert_eq!(body["storage"]["ready"], true);
    assert_eq!(body["auth"]["required"], true);
    assert_eq!(body["external_bind"]["enabled"], true);

    let raw_body = String::from_utf8(bytes.to_vec()).expect("utf8 body");
    assert!(!raw_body.contains(TOKEN));
    assert!(!raw_body.contains(&api_key_hash));
    assert!(!raw_body.contains(&model_path));
}

#[tokio::test]
async fn chat_completions_non_streaming_passthrough_returns_upstream_response() {
    let (upstream, mut upstream_requests) = spawn_mock_upstream().await;
    let mut cfg = config_with_models(vec![model("llama")]);
    cfg.server.llama_server = upstream;
    let app = test_app(cfg).await;
    let request_body = json!({
        "model": "llama",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .body(Body::from(request_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    response.headers()["x-request-id"]
        .to_str()
        .unwrap()
        .parse::<Uuid>()
        .expect("generated request id");
    assert_eq!(
        response
            .headers()
            .get("x-llmctl-model")
            .and_then(|value| value.to_str().ok()),
        Some("llama")
    );
    assert_eq!(
        response
            .headers()
            .get("x-llmctl-upstream-model")
            .and_then(|value| value.to_str().ok()),
        Some("llama")
    );
    assert_eq!(
        response
            .headers()
            .get("x-llmctl-quota-decision")
            .and_then(|value| value.to_str().ok()),
        Some("allowed")
    );
    assert_response_headers_do_not_leak(
        response.headers(),
        &[
            "hello",
            "127.0.0.1",
            "/models/llama.gguf",
            TOKEN,
            &cfg_api_key_hash(),
        ],
    );
    let body = response_json(response).await;
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "llama");
    assert_eq!(body["choices"][0]["message"]["content"], "pong");
    assert_eq!(body["usage"]["prompt_tokens"], 3);
    assert_eq!(body["usage"]["completion_tokens"], 5);

    let upstream_request = upstream_requests.recv().await.expect("upstream request");
    assert_eq!(upstream_request["model"], "llama");
    assert_eq!(upstream_request["messages"], request_body["messages"]);
    assert_eq!(upstream_request["stream"], false);
}

#[tokio::test]
async fn chat_completions_routes_by_runtime_backend() {
    let (native_upstream, mut native_upstream_requests) = spawn_mock_upstream().await;
    let mut native_cfg = config_with_models(vec![model("llama")]);
    native_cfg.runtime.backend = RuntimeBackend::CandleNative;
    native_cfg.server.llama_server = native_upstream;
    let native_app = test_app(native_cfg).await;

    let native_response = native_app.oneshot(chat_request()).await.expect("response");

    assert_eq!(native_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_response_headers_do_not_leak(
        native_response.headers(),
        &[
            "127.0.0.1",
            "/v1/chat/completions",
            "hello",
            TOKEN,
            &cfg_api_key_hash(),
        ],
    );
    let native_body = response_json(native_response).await;
    assert_eq!(native_body["error"]["type"], "native_runtime_not_ready");
    assert_eq!(native_body["error"]["code"], "native_runtime_not_ready");
    assert_eq!(
        native_body["error"]["message"],
        "native runtime is not ready to serve chat completions"
    );
    assert!(
        native_upstream_requests.try_recv().is_err(),
        "candle-native requests must not proxy to llama-server"
    );

    let (llama_upstream, mut llama_upstream_requests) = spawn_mock_upstream().await;
    let mut llama_cfg = config_with_models(vec![model("llama")]);
    llama_cfg.runtime.backend = RuntimeBackend::LlamaServer;
    llama_cfg.server.llama_server = llama_upstream;
    let llama_app = test_app(llama_cfg).await;

    let llama_response = llama_app.oneshot(chat_request()).await.expect("response");

    assert_eq!(llama_response.status(), StatusCode::OK);
    let llama_body = response_json(llama_response).await;
    assert_eq!(llama_body["choices"][0]["message"]["content"], "pong");
    let upstream_request = llama_upstream_requests
        .recv()
        .await
        .expect("llama-server upstream request");
    assert_eq!(upstream_request["model"], "llama");
}

#[tokio::test]
async fn chat_completions_streaming_native_not_ready_fails_closed_without_upstream() {
    let (native_upstream, mut native_upstream_requests) = spawn_mock_upstream().await;
    let mut cfg = config_with_models(vec![model("llama")]);
    cfg.runtime.backend = RuntimeBackend::CandleNative;
    cfg.server.llama_server = native_upstream;
    let app = test_app(cfg).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "llama",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_response_headers_do_not_leak(
        response.headers(),
        &[
            "127.0.0.1",
            "/v1/chat/completions",
            "hello",
            TOKEN,
            &cfg_api_key_hash(),
        ],
    );
    let body = response_json(response).await;
    assert_eq!(body["error"]["type"], "native_runtime_not_ready");
    assert_eq!(body["error"]["code"], "native_runtime_not_ready");
    assert!(
        native_upstream_requests.try_recv().is_err(),
        "streaming candle-native requests must not proxy to llama-server"
    );
}

#[tokio::test]
async fn chat_completions_non_streaming_native_engine_returns_openai_response() {
    let mut cfg = config_with_models(vec![model("llama")]);
    cfg.runtime.backend = RuntimeBackend::CandleNative;
    let storage = Storage::in_memory().await.expect("storage");
    let app = server::router_with_native_engine(cfg, storage, Arc::new(StaticNativeEngine));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "llama",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": false
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-llmctl-model")
            .and_then(|value| value.to_str().ok()),
        Some("llama")
    );
    assert_eq!(
        response
            .headers()
            .get("x-llmctl-upstream-model")
            .and_then(|value| value.to_str().ok()),
        Some("llama")
    );
    assert_response_headers_do_not_leak(
        response.headers(),
        &["hello", "/models/llama.gguf", TOKEN, &cfg_api_key_hash()],
    );
    let body = response_json(response).await;
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "llama");
    assert_eq!(body["choices"][0]["message"]["content"], "native pong");
    assert_eq!(body["usage"]["prompt_tokens"], 2);
    assert_eq!(body["usage"]["completion_tokens"], 3);
}

#[tokio::test]
async fn chat_completions_streaming_native_engine_returns_sse_and_records_usage() {
    let request_id = Uuid::from_u128(8);
    let mut cfg = config_with_models(vec![model("llama")]);
    cfg.runtime.backend = RuntimeBackend::CandleNative;
    let storage = Storage::in_memory().await.expect("storage");
    let app = server::router_with_native_engine(cfg, storage.clone(), Arc::new(StaticNativeEngine));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .header("x-request-id", request_id.to_string())
                .body(Body::from(
                    json!({
                        "model": "llama",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("stream body");
    let body = String::from_utf8(bytes.to_vec()).expect("utf8 stream");
    assert!(body.contains("\"object\":\"chat.completion.chunk\""));
    assert!(body.contains("\"content\":\"native pong\""));
    assert!(body.contains("\"finish_reason\":\"stop\""));
    assert!(body.ends_with("data: [DONE]\n\n"));

    let usage_events = storage
        .usage_events_for_request(request_id)
        .await
        .expect("usage events");
    assert_eq!(usage_events.len(), 1);
    assert_eq!(usage_events[0].input_tokens, 2);
    assert_eq!(usage_events[0].output_tokens, 3);
    assert_eq!(usage_events[0].status, "ok");
}

#[tokio::test]
async fn chat_completions_native_engine_receives_converted_request_and_records_usage() {
    let request_id = Uuid::from_u128(7);
    let mut cfg = config_with_models(vec![model("native"), model("public")]);
    cfg.runtime.backend = RuntimeBackend::CandleNative;
    cfg.mode = Mode::Single;
    let storage = Storage::in_memory().await.expect("storage");
    let (tx, mut rx) = mpsc::channel(1);
    let app = server::router_with_native_engine(
        cfg,
        storage.clone(),
        Arc::new(CapturingNativeEngine {
            alias: "native",
            tx,
        }),
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .header("x-request-id", request_id.to_string())
                .body(Body::from(
                    json!({
                        "model": "public",
                        "messages": [
                            {"role": "system", "content": "be terse"},
                            {"role": "user", "content": "hello"}
                        ],
                        "temperature": 0.2,
                        "max_tokens": 64,
                        "stream": false,
                        "metadata": {"tenant": "platform"}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()["x-request-id"].to_str().unwrap(),
        request_id.to_string()
    );
    assert_eq!(
        response
            .headers()
            .get("x-llmctl-model")
            .and_then(|value| value.to_str().ok()),
        Some("public")
    );
    assert_eq!(
        response
            .headers()
            .get("x-llmctl-upstream-model")
            .and_then(|value| value.to_str().ok()),
        Some("native")
    );

    let body = response_json(response).await;
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "native");
    assert_eq!(body["usage"]["prompt_tokens"], 11);
    assert_eq!(body["usage"]["completion_tokens"], 13);

    let native_request = rx.recv().await.expect("native request");
    assert_eq!(native_request.model, "native");
    assert_eq!(
        native_request.messages,
        vec![
            native::NativeChatMessage {
                role: "system".to_string(),
                content: "be terse".to_string(),
            },
            native::NativeChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            },
        ]
    );
    assert_eq!(native_request.temperature, Some(0.2));
    assert_eq!(native_request.max_tokens, Some(64));
    assert_eq!(native_request.metadata["tenant"], json!("platform"));
    assert_eq!(
        native_request.metadata["llmctl.request_id"],
        json!(request_id.to_string())
    );
    assert_eq!(
        native_request.metadata["llmctl.requested_model"],
        json!("public")
    );
    assert_eq!(
        native_request.metadata["llmctl.upstream_model"],
        json!("native")
    );
    assert_eq!(native_request.metadata["llmctl.stream"], json!(false));

    let usage_events = storage
        .usage_events_for_request(request_id)
        .await
        .expect("usage events");
    assert_eq!(usage_events.len(), 1);
    assert_eq!(usage_events[0].model, "public");
    assert_eq!(usage_events[0].input_tokens, 11);
    assert_eq!(usage_events[0].output_tokens, 13);
    assert_eq!(usage_events[0].status, "ok");

    let audit_events = storage
        .audit_events_for_request(request_id)
        .await
        .expect("audit events");
    assert!(
        audit_events.iter().any(|event| event.outcome == "ok"
            && event.detail_json["runtime_backend"] == json!("candle-native")
            && event.detail_json["token_accounting"] == json!("estimated")),
        "native completion audit event should include runtime and accounting details"
    );
}

#[tokio::test]
async fn chat_completions_native_engine_registry_routes_by_alias() {
    let mut cfg = config_with_models(vec![model("thinking"), model("coding")]);
    cfg.runtime.backend = RuntimeBackend::CandleNative;
    cfg.mode = Mode::HotSwap;
    let storage = Storage::in_memory().await.expect("storage");
    let (thinking_tx, mut thinking_rx) = mpsc::channel(1);
    let (coding_tx, mut coding_rx) = mpsc::channel(1);
    let mut engines = server::NativeEngineRegistry::new();
    engines.insert(
        "thinking".to_string(),
        Arc::new(CapturingNativeEngine {
            alias: "thinking",
            tx: thinking_tx,
        }),
    );
    engines.insert(
        "coding".to_string(),
        Arc::new(CapturingNativeEngine {
            alias: "coding",
            tx: coding_tx,
        }),
    );
    let app = server::router_with_native_engines(cfg, storage, engines);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "coding",
                        "messages": [{"role": "user", "content": "write rust"}],
                        "stream": false
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let request = coding_rx.recv().await.expect("coding request");
    assert_eq!(request.model, "coding");
    assert!(
        thinking_rx.try_recv().is_err(),
        "request must not be sent to the wrong native engine"
    );
}

#[tokio::test]
async fn weighted_mode_routes_chat_completions_by_configured_weights() {
    let (upstream, mut upstream_requests) = spawn_mock_upstream().await;
    let mut cfg = config_with_models(vec![model("light"), model("medium"), model("heavy")]);
    cfg.mode = Mode::Weighted;
    cfg.models[0].weight = 1;
    cfg.models[1].weight = 2;
    cfg.models[2].weight = 3;
    cfg.server.llama_server = upstream;
    let app = test_app(cfg).await;

    for (request_id, expected_upstream) in [
        (Uuid::from_u128(0), "light"),
        (Uuid::from_u128(1), "medium"),
        (Uuid::from_u128(2), "medium"),
        (Uuid::from_u128(3), "heavy"),
        (Uuid::from_u128(5), "heavy"),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", bearer())
                    .header("content-type", "application/json")
                    .header("x-request-id", request_id.to_string())
                    .body(Body::from(
                        json!({"model": "light", "messages": [], "stream": false}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("x-llmctl-model")
                .and_then(|value| value.to_str().ok()),
            Some("light")
        );
        assert_eq!(
            response
                .headers()
                .get("x-llmctl-upstream-model")
                .and_then(|value| value.to_str().ok()),
            Some(expected_upstream)
        );
        let body = response_json(response).await;
        assert_eq!(body["model"], expected_upstream);

        let upstream_request = upstream_requests.recv().await.expect("upstream request");
        assert_eq!(upstream_request["model"], expected_upstream);
    }
}

#[tokio::test]
async fn fallback_mode_retries_next_model_after_upstream_server_error() {
    let (upstream, mut upstream_requests) = spawn_fallback_upstream().await;
    let mut cfg = config_with_models(vec![model("primary"), model("secondary"), model("backup")]);
    cfg.mode = Mode::Fallback;
    cfg.models[0].weight = 100;
    cfg.models[1].weight = 10;
    cfg.models[2].weight = 0;
    cfg.server.llama_server = upstream;
    let app = test_app(cfg).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .header("x-request-id", Uuid::from_u128(0).to_string())
                .body(Body::from(
                    json!({"model": "backup", "messages": [], "stream": false}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-llmctl-upstream-model")
            .and_then(|value| value.to_str().ok()),
        Some("backup")
    );
    let body = response_json(response).await;
    assert_eq!(body["model"], "backup");
    assert_eq!(
        upstream_requests.recv().await.expect("primary request")["model"],
        "primary"
    );
    assert_eq!(
        upstream_requests.recv().await.expect("secondary request")["model"],
        "secondary"
    );
    assert_eq!(
        upstream_requests.recv().await.expect("backup request")["model"],
        "backup"
    );
}

#[tokio::test]
async fn local_search_returns_ranked_hits_for_code_assistance_substrate() {
    let app = test_app(config_with_models(vec![model("llama")])).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/local/search")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "query": "worker readiness",
                        "documents": [
                            {
                                "id": "ops",
                                "title": "Operations",
                                "path": "docs/operations.md",
                                "content": "Worker readiness probes and restart backoff"
                            },
                            {
                                "id": "billing",
                                "title": "Usage",
                                "content": "chargeback and token reports"
                            }
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    assert_eq!(body["object"], "search.results");
    assert_eq!(body["data"][0]["id"], "ops");
    assert!(body["data"][0]["snippet"]
        .as_str()
        .unwrap()
        .contains("readiness"));
}

#[tokio::test]
async fn local_recommendations_rank_local_material_for_ai_developer_workflows() {
    let app = test_app(config_with_models(vec![model("llama")])).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/local/recommendations")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "task": "build code review recommendations from local docs",
                        "documents": [
                            {
                                "id": "ops",
                                "title": "Operations Notes",
                                "content": "model lifecycle and server status"
                            },
                            {
                                "id": "code-review",
                                "title": "Code Review Guide",
                                "content": "code review recommendations and local search"
                            }
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    assert_eq!(body["object"], "recommendation.results");
    assert_eq!(body["data"][0]["id"], "code-review");
    assert_eq!(body["recommendations"][0]["document_id"], "code-review");
}

#[tokio::test]
async fn local_recommendations_records_runtime_lineage_headers() {
    let request_id = Uuid::new_v4();
    let (app, storage) = test_app_with_storage(config_with_models(vec![model("llama")])).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/local/recommendations")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .header("x-request-id", request_id.to_string())
                .header("x-llmctl-lineage-id", "corpus:code-review")
                .header("x-llmctl-corpus", "engineering-docs")
                .body(Body::from(
                    json!({
                        "task": "code review",
                        "documents": [
                            {
                                "id": "code-review",
                                "title": "Code Review Guide",
                                "content": "code review recommendations"
                            }
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let joins = storage
        .request_lineage_joins_for_request(request_id)
        .await
        .expect("lineage joins");
    assert_eq!(joins.len(), 1);
    assert_eq!(joins[0].lineage_id, "corpus:code-review");
    assert_eq!(joins[0].model, None);
    assert_eq!(joins[0].corpus.as_deref(), Some("engineering-docs"));
    assert_eq!(joins[0].source, "local.recommendations");
}

#[tokio::test]
async fn local_search_records_runtime_lineage_metadata() {
    let request_id = Uuid::new_v4();
    let (app, storage) = test_app_with_storage(config_with_models(vec![model("llama")])).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/local/search")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .header("x-request-id", request_id.to_string())
                .body(Body::from(
                    json!({
                        "query": "readiness",
                        "metadata": {
                            "lineage_ids": ["corpus:ops-v1", "prompt:search-template"],
                            "corpus": "ops-docs"
                        },
                        "documents": [
                            {
                                "id": "ops",
                                "title": "Operations Notes",
                                "content": "readiness and lifecycle"
                            }
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let joins = storage
        .request_lineage_joins_for_request(request_id)
        .await
        .expect("lineage joins");
    assert_eq!(joins.len(), 2);
    assert_eq!(joins[0].lineage_id, "corpus:ops-v1");
    assert_eq!(joins[0].model, None);
    assert_eq!(joins[0].corpus.as_deref(), Some("ops-docs"));
    assert_eq!(joins[0].source, "local.search");
}

#[tokio::test]
async fn embeddings_endpoint_proxies_openai_compatible_payloads() {
    let (upstream, mut upstream_requests) = spawn_embeddings_upstream().await;
    let mut cfg = config_with_models(vec![model("embed")]);
    cfg.server.llama_server = upstream;
    let app = test_app(cfg).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": "embed", "input": "hello"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    assert_eq!(body["object"], "list");
    assert_eq!(
        upstream_requests.recv().await.expect("embedding request")["model"],
        "embed"
    );
}

#[tokio::test]
async fn chat_completions_records_runtime_lineage_headers() {
    let request_id = Uuid::new_v4();
    let (upstream, mut upstream_requests) = spawn_mock_upstream().await;
    let mut cfg = config_with_models(vec![model("llama")]);
    cfg.server.llama_server = upstream;
    let (app, storage) = test_app_with_storage(cfg).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .header("x-request-id", request_id.to_string())
                .header("x-llmctl-lineage-id", "prompt:review-v2,corpus:ops-v1")
                .header("x-llmctl-corpus", "ops-docs")
                .body(Body::from(
                    json!({"model": "llama", "messages": [], "stream": false}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    upstream_requests.recv().await.expect("upstream request");
    let joins = storage
        .request_lineage_joins_for_request(request_id)
        .await
        .expect("lineage joins");
    assert_eq!(joins.len(), 2);
    assert_eq!(joins[0].request_id, request_id);
    assert_eq!(joins[0].lineage_id, "prompt:review-v2");
    assert_eq!(joins[0].model.as_deref(), Some("llama"));
    assert_eq!(joins[0].corpus.as_deref(), Some("ops-docs"));
    assert_eq!(joins[0].source, "chat.completions");
}

#[tokio::test]
async fn chat_completions_propagates_request_id_to_header_audit_quota_and_usage() {
    let request_id = Uuid::new_v4();
    let (upstream, mut upstream_requests) = spawn_mock_upstream().await;
    let mut cfg = config_with_models(vec![model("llama")]);
    cfg.server.llama_server = upstream;
    cfg.quotas = vec![QuotaConfig {
        subject: "alice".to_string(),
        team: "".to_string(),
        requests_per_minute: 10,
        tokens_per_day: 100,
        max_concurrency: 0,
        allowed_models: vec!["llama".to_string()],
    }];
    let (app, storage) = test_app_with_storage(cfg).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .header("x-request-id", request_id.to_string())
                .body(Body::from(
                    json!({"model": "llama", "messages": [], "stream": false}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()["x-request-id"].to_str().unwrap(),
        request_id.to_string()
    );
    let body = response_json(response).await;
    assert_eq!(body["object"], "chat.completion");
    upstream_requests.recv().await.expect("upstream request");

    let audit_events = storage
        .audit_events_for_request(request_id)
        .await
        .expect("audit events");
    let usage_events = storage
        .usage_events_for_request(request_id)
        .await
        .expect("usage events");
    let quota_decisions = storage
        .quota_decisions_for_request(request_id)
        .await
        .expect("quota decisions");
    assert!(
        audit_events.len() >= 2,
        "expected allowed and completion audit events"
    );
    assert_eq!(usage_events.len(), 1);
    assert_eq!(usage_events[0].request_id, request_id);
    assert_eq!(quota_decisions.len(), 1);
    assert_eq!(quota_decisions[0].request_id, Some(request_id));
}

#[tokio::test]
async fn chat_completions_streaming_passthrough_returns_sse_and_marks_unmetered_zero_token_usage() {
    let (upstream, mut upstream_requests) = spawn_mock_upstream().await;
    let mut cfg = config_with_models(vec![model("llama")]);
    cfg.server.llama_server = upstream;
    let (app, storage) = test_app_with_storage(cfg).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "llama",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let request_id = response.headers()["x-request-id"]
        .to_str()
        .unwrap()
        .parse::<Uuid>()
        .expect("generated request id");
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    assert_eq!(
        response
            .headers()
            .get("x-llmctl-model")
            .and_then(|value| value.to_str().ok()),
        Some("llama")
    );
    assert_eq!(
        response
            .headers()
            .get("x-llmctl-upstream-model")
            .and_then(|value| value.to_str().ok()),
        Some("llama")
    );
    assert_eq!(
        response
            .headers()
            .get("x-llmctl-quota-decision")
            .and_then(|value| value.to_str().ok()),
        Some("allowed")
    );
    assert_response_headers_do_not_leak(
        response.headers(),
        &[
            "hello",
            "127.0.0.1",
            "/models/llama.gguf",
            TOKEN,
            &cfg_api_key_hash(),
        ],
    );

    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("stream body");
    let body = String::from_utf8(bytes.to_vec()).expect("utf8 stream");
    assert_eq!(
        body,
        concat!(
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"model\":\"llama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"po\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"model\":\"llama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ng\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        )
    );

    let upstream_request = upstream_requests.recv().await.expect("upstream request");
    assert_eq!(upstream_request["model"], "llama");
    assert_eq!(upstream_request["stream"], true);

    let usage_events = storage
        .usage_events_for_request(request_id)
        .await
        .expect("usage events");
    assert_eq!(usage_events.len(), 1);
    assert_eq!(usage_events[0].input_tokens, 0);
    assert_eq!(usage_events[0].output_tokens, 0);
    assert_eq!(usage_events[0].status, "stream_unmetered");
}

#[tokio::test]
async fn chat_completions_rejects_missing_auth() {
    let app = test_app(config_with_models(vec![model("llama")])).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": "llama", "messages": []}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let body = response_json(response).await;
    assert_eq!(body["error"]["type"], "unauthorized");
    assert_eq!(body["error"]["code"], "unauthorized");
}

#[tokio::test]
async fn chat_completions_rejects_unknown_model_before_upstream() {
    let (upstream, mut upstream_requests) = spawn_mock_upstream().await;
    let mut cfg = config_with_models(vec![model("llama")]);
    cfg.server.llama_server = upstream;
    let app = test_app(cfg).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": "missing", "messages": []}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_json(response).await;
    assert_eq!(body["error"]["type"], "unknown_model");
    assert_eq!(body["error"]["code"], "unknown_model");
    assert!(
        upstream_requests.try_recv().is_err(),
        "unknown model must not reach upstream"
    );
}

#[tokio::test]
async fn chat_completions_returns_429_when_quota_denies_model() {
    let (upstream, mut upstream_requests) = spawn_mock_upstream().await;
    let mut cfg = config_with_models(vec![model("llama")]);
    cfg.server.llama_server = upstream;
    cfg.quotas = vec![QuotaConfig {
        subject: "alice".to_string(),
        team: "".to_string(),
        requests_per_minute: 10,
        tokens_per_day: 100,
        max_concurrency: 0,
        allowed_models: vec!["other".to_string()],
    }];
    let app = test_app(cfg).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": "llama", "messages": []}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = response_json(response).await;
    assert_eq!(body["error"]["type"], "quota_exceeded");
    assert_eq!(body["error"]["code"], "quota_exceeded");
    assert!(
        upstream_requests.try_recv().is_err(),
        "quota denial must not reach upstream"
    );
}

#[tokio::test]
async fn chat_completions_returns_429_when_admission_limit_is_full_without_leaking_details() {
    let (upstream, mut upstream_requests, release_upstream) = spawn_blocking_upstream().await;
    let mut cfg = config_with_models(vec![model("llama")]);
    cfg.server.llama_server = upstream;
    let app =
        test_app_with_limits(cfg, server::ServingLimits::new(1, Duration::from_secs(30))).await;

    let first = tokio::spawn({
        let app = app.clone();
        async move { app.oneshot(chat_request()).await.expect("first response") }
    });
    upstream_requests
        .recv()
        .await
        .expect("first upstream request");

    let response = app.oneshot(chat_request()).await.expect("second response");

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_response_headers_do_not_leak(
        response.headers(),
        &[
            "127.0.0.1",
            "/v1/chat/completions",
            TOKEN,
            &cfg_api_key_hash(),
        ],
    );
    let body = response_json(response).await;
    assert_eq!(body["error"]["type"], "rate_limit_exceeded");
    assert_eq!(body["error"]["code"], "rate_limit_exceeded");
    assert_eq!(body["error"]["message"], "server is busy; retry later");
    assert!(
        upstream_requests.try_recv().is_err(),
        "admission denial must not reach upstream"
    );

    release_upstream.send(()).expect("release upstream");
    let first_response = first.await.expect("first task");
    assert_eq!(first_response.status(), StatusCode::OK);
}

#[tokio::test]
async fn chat_completions_returns_sanitized_504_when_upstream_times_out() {
    let (upstream, mut upstream_requests, _release_upstream) = spawn_blocking_upstream().await;
    let mut cfg = config_with_models(vec![model("llama")]);
    cfg.server.llama_server = upstream;
    let app = test_app_with_limits(
        cfg,
        server::ServingLimits::new(4, Duration::from_millis(25)),
    )
    .await;

    let response = app.oneshot(chat_request()).await.expect("response");

    upstream_requests.recv().await.expect("upstream request");
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_response_headers_do_not_leak(
        response.headers(),
        &[
            "127.0.0.1",
            "/v1/chat/completions",
            "hello",
            TOKEN,
            &cfg_api_key_hash(),
        ],
    );
    let body = response_json(response).await;
    assert_eq!(body["error"]["type"], "timeout");
    assert_eq!(body["error"]["code"], "timeout");
    assert_eq!(body["error"]["message"], "upstream request timed out");
}

#[tokio::test]
async fn admin_swap_requires_admin_scope_and_attached_worker_control() {
    let app = test_app(config_with_models(vec![model("old"), model("new")])).await;

    let forbidden = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/swap")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"active":"old","replacement":"new","mode":"hot"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(forbidden.status(), StatusCode::FORBIDDEN);

    let app = test_app(admin_config_with_models(vec![model("old"), model("new")])).await;
    let unavailable = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/swap")
                .header("authorization", bearer())
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"active":"old","replacement":"new","mode":"hot"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(unavailable.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = response_json(unavailable).await;
    assert_eq!(body["error"]["code"], "worker_control_unavailable");
}

async fn test_app(cfg: Config) -> Router {
    let storage = Storage::in_memory().await.expect("storage");
    server::router(cfg, storage)
}

async fn test_app_with_limits(cfg: Config, limits: server::ServingLimits) -> Router {
    let storage = Storage::in_memory().await.expect("storage");
    server::router_with_serving_limits(cfg, storage, limits)
}

async fn test_app_with_storage(cfg: Config) -> (Router, Storage) {
    let storage = Storage::in_memory().await.expect("storage");
    let app = server::router(cfg, storage.clone());
    (app, storage)
}

fn config_with_models(models: Vec<ModelConfig>) -> Config {
    let mut cfg = Config::default();
    cfg.runtime.backend = RuntimeBackend::LlamaServer;
    cfg.security.require_auth = true;
    cfg.security.api_keys = vec![ApiKeyConfig {
        id: "test".to_string(),
        sha256: hex::encode(Sha256::digest(TOKEN.as_bytes())),
        subject: "alice".to_string(),
        team: "platform".to_string(),
        scopes: vec!["chat".to_string(), "models.read".to_string()],
    }];
    cfg.models = models;
    cfg
}

fn admin_config_with_models(models: Vec<ModelConfig>) -> Config {
    let mut cfg = config_with_models(models);
    cfg.security.api_keys[0].scopes.push("admin".to_string());
    cfg
}

fn model(alias: &str) -> ModelConfig {
    ModelConfig {
        alias: alias.to_string(),
        path: PathBuf::from(format!("/models/{alias}.gguf")),
        role: "chat".to_string(),
        weight: 1,
    }
}

fn bearer() -> String {
    format!("Bearer {TOKEN}")
}

fn cfg_api_key_hash() -> String {
    hex::encode(Sha256::digest(TOKEN.as_bytes()))
}

fn assert_response_headers_do_not_leak(headers: &axum::http::HeaderMap, forbidden: &[&str]) {
    let values = headers
        .iter()
        .filter_map(|(_, value)| value.to_str().ok())
        .collect::<Vec<_>>()
        .join("\n");
    for secret in forbidden {
        assert!(
            !values.contains(secret),
            "response headers leaked forbidden value: {secret}"
        );
    }
}

async fn response_json(response: axum::response::Response) -> Value {
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body");
    serde_json::from_slice(&bytes).expect("json response")
}

fn chat_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", bearer())
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": "llama", "messages": [{"role": "user", "content": "hello"}]})
                .to_string(),
        ))
        .unwrap()
}

async fn spawn_mock_upstream() -> (String, mpsc::Receiver<Value>) {
    let (tx, rx) = mpsc::channel(4);
    let app = Router::new()
        .route("/v1/chat/completions", post(mock_chat_completion))
        .with_state(tx);
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind upstream");
    let addr = listener.local_addr().expect("upstream addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve upstream");
    });
    (format!("http://{addr}"), rx)
}

async fn spawn_fallback_upstream() -> (String, mpsc::Receiver<Value>) {
    let (tx, rx) = mpsc::channel(4);
    let app = Router::new()
        .route(
            "/v1/chat/completions",
            post(
                |State(tx): State<mpsc::Sender<Value>>, Json(request): Json<Value>| async move {
                    tx.send(request.clone()).await.expect("record request");
                    if request["model"] != "backup" {
                        return (StatusCode::INTERNAL_SERVER_ERROR, "upstream failed")
                            .into_response();
                    }
                    Json(json!({
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "created": 1_700_000_000,
                        "model": request["model"],
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "pong"},
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 5,
                            "total_tokens": 8
                        }
                    }))
                    .into_response()
                },
            ),
        )
        .with_state(tx);
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind upstream");
    let addr = listener.local_addr().expect("upstream addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve upstream");
    });
    (format!("http://{addr}"), rx)
}

async fn spawn_embeddings_upstream() -> (String, mpsc::Receiver<Value>) {
    let (tx, rx) = mpsc::channel(4);
    let app = Router::new()
        .route(
            "/v1/embeddings",
            post(
                |State(tx): State<mpsc::Sender<Value>>, Json(request): Json<Value>| async move {
                    tx.send(request.clone()).await.expect("record request");
                    Json(json!({
                        "object": "list",
                        "model": request["model"],
                        "data": [{
                            "object": "embedding",
                            "index": 0,
                            "embedding": [0.1, 0.2, 0.3]
                        }],
                        "usage": {
                            "prompt_tokens": 1,
                            "total_tokens": 1
                        }
                    }))
                },
            ),
        )
        .with_state(tx);
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind upstream");
    let addr = listener.local_addr().expect("upstream addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve upstream");
    });
    (format!("http://{addr}"), rx)
}

async fn spawn_blocking_upstream() -> (String, mpsc::Receiver<Value>, oneshot::Sender<()>) {
    let (request_tx, request_rx) = mpsc::channel(4);
    let (release_tx, release_rx) = oneshot::channel::<()>();
    let release_rx = std::sync::Arc::new(tokio::sync::Mutex::new(Some(release_rx)));
    let app = Router::new()
        .route(
            "/v1/chat/completions",
            post(
                |State((request_tx, release_rx)): State<BlockingUpstreamState>,
                 Json(request): Json<Value>| async move {
                    request_tx.send(request).await.expect("record request");
                    let release = release_rx.lock().await.take();
                    if let Some(release) = release {
                        let _ = release.await;
                    } else {
                        std::future::pending::<()>().await;
                    }
                    Json(json!({
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "model": "llama",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "pong"},
                            "finish_reason": "stop"
                        }]
                    }))
                },
            ),
        )
        .with_state((request_tx, release_rx));
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind upstream");
    let addr = listener.local_addr().expect("upstream addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve upstream");
    });
    (format!("http://{addr}"), request_rx, release_tx)
}

async fn mock_chat_completion(
    State(tx): State<mpsc::Sender<Value>>,
    Json(request): Json<Value>,
) -> Response {
    tx.send(request.clone()).await.expect("record request");
    if request["stream"].as_bool() == Some(true) {
        return (
            [("content-type", "text/event-stream")],
            concat!(
                "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"model\":\"llama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"po\"},\"finish_reason\":null}]}\n\n",
                "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"model\":\"llama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ng\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: [DONE]\n\n"
            ),
        )
            .into_response();
    }

    Json(json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": request["model"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "pong"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 5,
            "total_tokens": 8
        }
    }))
    .into_response()
}
