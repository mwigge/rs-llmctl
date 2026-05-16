use axum::body::{to_bytes, Body};
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use rs_llmctl::config::{ApiKeyConfig, Config, ModelConfig, QuotaConfig};
use rs_llmctl::server;
use rs_llmctl::storage::Storage;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tower::ServiceExt;

const TOKEN: &str = "test-token";

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
    let body = response_json(response).await;
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "alpha");
    assert_eq!(body["data"][0]["object"], "model");
    assert_eq!(body["data"][0]["owned_by"], "rs-llmctl");
    assert_eq!(body["data"][1]["id"], "middle");
    assert_eq!(body["data"][2]["id"], "zeta");
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
    let cfg = config_with_models(vec![model("llama"), model("embed")]);
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
    assert_eq!(body["models"]["configured"], 2);
    assert_eq!(body["storage"]["ready"], true);

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

async fn test_app(cfg: Config) -> Router {
    let storage = Storage::in_memory().await.expect("storage");
    server::router(cfg, storage)
}

fn config_with_models(models: Vec<ModelConfig>) -> Config {
    let mut cfg = Config::default();
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

async fn response_json(response: axum::response::Response) -> Value {
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body");
    serde_json::from_slice(&bytes).expect("json response")
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

async fn mock_chat_completion(
    State(tx): State<mpsc::Sender<Value>>,
    Json(request): Json<Value>,
) -> Json<Value> {
    tx.send(request.clone()).await.expect("record request");
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
}
