use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::StreamExt;
use rs_llmctl_client::{
    AskConfig, ChatCompletionRequest, ChatMessage, EmbeddingRequest, LlmctlClient, LlmctlError,
    LocalRecommendationsRequest, LocalSearchRequest, Question, SearchDocument, Session,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Mutex;

#[tokio::test]
async fn chat_maps_success_response() {
    let seen = Arc::new(Mutex::new(None));
    let client = client_for(app(seen.clone())).await;

    let response = client
        .chat(ChatCompletionRequest::new(
            "llama",
            vec![ChatMessage::user("hello")],
        ))
        .await
        .expect("chat response");

    assert_eq!(response.model, "llama");
    assert_eq!(response.choices[0].message.content.as_deref(), Some("pong"));
    assert_eq!(response.usage.expect("usage").total_tokens, 8);
    assert_eq!(seen.lock().await.as_ref().unwrap()["stream"], false);
}

#[tokio::test]
async fn chat_stream_decodes_sse_chunks() {
    let client = client_for(app(Arc::new(Mutex::new(None)))).await;
    let mut stream = client
        .chat_stream(ChatCompletionRequest::new(
            "llama",
            vec![ChatMessage::user("hello")],
        ))
        .await
        .expect("chat stream");

    let first = stream
        .next()
        .await
        .expect("first chunk")
        .expect("chunk decodes");
    assert_eq!(first.choices[0].delta.content.as_deref(), Some("pon"));
    let second = stream
        .next()
        .await
        .expect("second chunk")
        .expect("chunk decodes");
    assert_eq!(second.choices[0].delta.content.as_deref(), Some("g"));
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn models_maps_openai_model_list() {
    let client = client_for(app(Arc::new(Mutex::new(None)))).await;

    let response = client.models_with_session().await.expect("models");
    let models = response.body;

    assert_eq!(response.metadata.request_id.as_deref(), Some("req-models"));
    assert_eq!(models.object, "list");
    assert_eq!(models.data[0].id, "llama");
    assert_eq!(models.data[0].owned_by, "rs-llmctl");
}

#[tokio::test]
async fn session_builds_chat_request_with_history_and_metadata() {
    let seen = Arc::new(Mutex::new(None));
    let client = client_for(app(seen.clone())).await;
    let mut session = Session::new("incident-42", "llama");
    session.push(ChatMessage::system("answer briefly"));
    session.push(ChatMessage::user("hello"));
    session.insert_metadata("lineage_ids", json!(["prompt:test"]));

    let response = client.chat_session(&session).await.expect("session chat");

    assert_eq!(response.text(), Some("pong"));
    let seen = seen.lock().await;
    let request = seen.as_ref().expect("seen request");
    assert_eq!(request["model"], "llama");
    assert_eq!(request["messages"][0]["role"], "system");
    assert_eq!(request["messages"][1]["role"], "user");
    assert_eq!(request["metadata"]["session_id"], "incident-42");
    assert_eq!(request["metadata"]["lineage_ids"], json!(["prompt:test"]));
}

#[tokio::test]
async fn ask_question_maps_to_chat_and_returns_text() {
    let seen = Arc::new(Mutex::new(None));
    let client = client_for(app(seen.clone())).await;

    let answer = client
        .ask_question(
            AskConfig::new("llama")
                .system("answer briefly")
                .max_tokens(32)
                .metadata(json!({"session_id": "ask-42"})),
            Question::new("hello").with_history(vec![ChatMessage::assistant("previous")]),
        )
        .await
        .expect("answer");

    assert_eq!(answer, "pong");
    let seen = seen.lock().await;
    let request = seen.as_ref().expect("seen request");
    assert_eq!(request["model"], "llama");
    assert_eq!(request["messages"][0]["role"], "system");
    assert_eq!(request["messages"][1]["role"], "assistant");
    assert_eq!(request["messages"][2]["content"], "hello");
    assert_eq!(request["max_tokens"], 32);
    assert_eq!(request["metadata"]["session_id"], "ask-42");
}

#[tokio::test]
async fn embeddings_maps_openai_compatible_response() {
    let client = client_for(app(Arc::new(Mutex::new(None)))).await;

    let response = client
        .embeddings(EmbeddingRequest::new("embed", vec!["hello", "world"]))
        .await
        .expect("embeddings");

    assert_eq!(response.object, "list");
    assert_eq!(response.model, "embed");
    assert_eq!(response.data.len(), 2);
    assert_eq!(response.data[0].embedding, vec![0.1, 0.2]);
    assert_eq!(response.usage.expect("usage").total_tokens, 2);
    assert_eq!(
        response.llmctl.expect("llmctl metadata")["embedding_status"],
        "non-semantic-dev-fallback"
    );
}

#[tokio::test]
async fn local_search_and_recommendations_map_responses() {
    let client = client_for(app(Arc::new(Mutex::new(None)))).await;
    let documents = vec![SearchDocument {
        id: "ops".to_string(),
        title: Some("Operations".to_string()),
        path: Some("docs/operations.md".to_string()),
        content: "restart workers".to_string(),
    }];

    let search = client
        .local_search(LocalSearchRequest {
            query: "restart".to_string(),
            limit: 5,
            metadata: None,
            documents: documents.clone(),
        })
        .await
        .expect("local search");
    assert_eq!(search.object, "search.results");
    assert_eq!(search.data[0].id, "ops");

    let recommendations = client
        .local_recommendations(LocalRecommendationsRequest {
            task: "restart".to_string(),
            limit: 5,
            metadata: None,
            documents,
        })
        .await
        .expect("recommendations");
    assert_eq!(recommendations.object, "recommendation.results");
    assert_eq!(recommendations.recommendations[0].document_id, "ops");
}

#[tokio::test]
async fn maps_error_variants_from_mocked_responses() {
    let client = client_for(app(Arc::new(Mutex::new(None)))).await;

    assert!(matches!(
        client
            .chat(ChatCompletionRequest::new(
                "auth",
                vec![ChatMessage::user("hello")]
            ))
            .await,
        Err(LlmctlError::Auth { .. })
    ));
    assert!(matches!(
        client
            .chat(ChatCompletionRequest::new(
                "quota",
                vec![ChatMessage::user("hello")]
            ))
            .await,
        Err(LlmctlError::Quota { .. })
    ));
    assert!(matches!(
        client
            .chat(ChatCompletionRequest::new(
                "busy",
                vec![ChatMessage::user("hello")]
            ))
            .await,
        Err(LlmctlError::RateLimited { .. })
    ));
    assert!(matches!(
        client
            .chat(ChatCompletionRequest::new(
                "timeout",
                vec![ChatMessage::user("hello")]
            ))
            .await,
        Err(LlmctlError::Timeout { .. })
    ));
    assert!(matches!(
        client
            .chat(ChatCompletionRequest::new(
                "missing",
                vec![ChatMessage::user("hello")]
            ))
            .await,
        Err(LlmctlError::ModelUnavailable { .. })
    ));
    assert!(matches!(
        client
            .chat(ChatCompletionRequest::new(
                "bad",
                vec![ChatMessage::user("hello")]
            ))
            .await,
        Err(LlmctlError::BadRequest { .. })
    ));
    assert!(matches!(
        client
            .chat(ChatCompletionRequest::new(
                "server",
                vec![ChatMessage::user("hello")]
            ))
            .await,
        Err(LlmctlError::Server { status: 500, .. })
    ));
}

async fn client_for(app: Router) -> LlmctlClient {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    LlmctlClient::new(format!("http://{addr}"), Some("test-token".to_string())).expect("client")
}

fn app(seen_chat: Arc<Mutex<Option<Value>>>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/models", get(models))
        .route("/v1/local/search", post(local_search))
        .route("/v1/local/recommendations", post(local_recommendations))
        .with_state(seen_chat)
}

async fn chat(
    State(seen): State<Arc<Mutex<Option<Value>>>>,
    Json(request): Json<Value>,
) -> Response {
    *seen.lock().await = Some(request.clone());
    match request["model"].as_str().unwrap_or_default() {
        "auth" => error(StatusCode::UNAUTHORIZED, "unauthorized", "bad token"),
        "quota" => error(StatusCode::TOO_MANY_REQUESTS, "quota_exceeded", "quota exhausted"),
        "busy" => error(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_exceeded",
            "server is busy",
        ),
        "timeout" => error(StatusCode::GATEWAY_TIMEOUT, "timeout", "timed out"),
        "missing" => error(StatusCode::BAD_REQUEST, "unknown_model", "unknown model"),
        "bad" => error(StatusCode::BAD_REQUEST, "bad_request", "bad request"),
        "server" => error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "failed"),
        _ if request["stream"] == true => (
            [("content-type", "text/event-stream")],
            Body::from(
                "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"model\":\"llama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"pon\"}}]}\n\n\
                 data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"model\":\"llama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"g\"},\"finish_reason\":\"stop\"}]}\n\n\
                 data: [DONE]\n\n",
            ),
        )
            .into_response(),
        _ => Json(json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama",
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
        .into_response(),
    }
}

async fn models(headers: HeaderMap) -> Response {
    assert_eq!(
        headers
            .get("authorization")
            .and_then(|value| value.to_str().ok()),
        Some("Bearer test-token")
    );
    (
        [("x-request-id", "req-models")],
        Json(json!({
        "object": "list",
        "data": [{"id": "llama", "object": "model", "owned_by": "rs-llmctl"}]
        })),
    )
        .into_response()
}

async fn embeddings(Json(request): Json<Value>) -> Response {
    assert_eq!(request["input"], json!(["hello", "world"]));
    Json(json!({
        "object": "list",
        "model": request["model"],
        "data": [
            {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
            {"object": "embedding", "embedding": [0.3, 0.4], "index": 1}
        ],
        "usage": {
            "prompt_tokens": 2,
            "total_tokens": 2
        },
        "llmctl": {
            "embedding_backend": "deterministic-local-fallback",
            "embedding_status": "non-semantic-dev-fallback",
            "semantic": false
        }
    }))
    .into_response()
}

async fn local_search(Json(request): Json<Value>) -> Response {
    Json(json!({
        "object": "search.results",
        "query": request["query"],
        "data": [{
            "id": "ops",
            "title": "Operations",
            "path": "docs/operations.md",
            "score": 4.0,
            "snippet": "restart workers"
        }]
    }))
    .into_response()
}

async fn local_recommendations(Json(request): Json<Value>) -> Response {
    Json(json!({
        "object": "recommendation.results",
        "task": request["task"],
        "data": [{
            "id": "ops",
            "title": "Operations",
            "path": "docs/operations.md",
            "score": 4.0,
            "snippet": "restart workers"
        }],
        "recommendations": [{
            "rank": "1",
            "document_id": "ops",
            "title": "Operations",
            "reason": "Use `Operations`."
        }]
    }))
    .into_response()
}

fn error(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": code,
                "code": code
            }
        })),
    )
        .into_response()
}
