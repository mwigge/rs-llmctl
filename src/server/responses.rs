use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use uuid::Uuid;

pub(super) fn response_headers(upstream_headers: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (name, value) in upstream_headers {
        if is_safe_upstream_response_header(name) {
            headers.insert(name.clone(), value.clone());
        }
    }
    headers
}

fn is_safe_upstream_response_header(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "content-type" | "cache-control" | "x-request-id"
    )
}

pub(super) fn build_response(
    status: StatusCode,
    headers: HeaderMap,
    body: Body,
    request_id: Uuid,
) -> Response {
    let mut response = Response::new(body);
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    with_request_id(response, request_id)
}

pub(super) fn error_response(status: StatusCode, code: &str, message: String) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": code,
                "code": code,
                "status": status.as_u16()
            }
        })),
    )
        .into_response()
}

pub(super) fn auth_error_response(message: String) -> Response {
    if message.contains("too many failed authentication attempts") {
        error_response(StatusCode::TOO_MANY_REQUESTS, "rate_limited", message)
    } else {
        error_response(StatusCode::UNAUTHORIZED, "unauthorized", message)
    }
}

pub(super) fn request_id_from_headers(headers: &HeaderMap) -> Uuid {
    headers
        .get(request_id_header_name())
        .and_then(|value| value.to_str().ok())
        .and_then(|value| Uuid::parse_str(value).ok())
        .unwrap_or_else(Uuid::new_v4)
}

pub(super) fn with_request_id(mut response: Response, request_id: Uuid) -> Response {
    response.headers_mut().insert(
        request_id_header_name(),
        HeaderValue::from_str(&request_id.to_string()).expect("uuid is a valid header value"),
    );
    response
}

pub(super) fn with_model_count(mut response: Response, count: usize) -> Response {
    insert_header_value(
        response.headers_mut(),
        model_count_header_name(),
        &count.to_string(),
    );
    response
}

pub(super) fn with_chat_metadata(
    mut response: Response,
    model: &str,
    upstream_model: &str,
    quota_decision: &str,
) -> Response {
    insert_header_value(response.headers_mut(), model_header_name(), model);
    insert_header_value(
        response.headers_mut(),
        upstream_model_header_name(),
        upstream_model,
    );
    insert_header_value(
        response.headers_mut(),
        quota_decision_header_name(),
        quota_decision,
    );
    response
}

fn insert_header_value(headers: &mut HeaderMap, name: HeaderName, value: &str) {
    match HeaderValue::from_str(value) {
        Ok(value) => {
            headers.insert(name, value);
        }
        Err(err) => {
            tracing::warn!(header = %name, error = %err, "skipping invalid response metadata header");
        }
    }
}

pub(super) fn request_id_header_name() -> HeaderName {
    HeaderName::from_static("x-request-id")
}

pub(super) fn lineage_id_header_name() -> HeaderName {
    HeaderName::from_static("x-llmctl-lineage-id")
}

pub(super) fn lineage_ids_header_name() -> HeaderName {
    HeaderName::from_static("x-llmctl-lineage-ids")
}

pub(super) fn corpus_header_name() -> HeaderName {
    HeaderName::from_static("x-llmctl-corpus")
}

pub(super) fn model_count_header_name() -> HeaderName {
    HeaderName::from_static("x-llmctl-model-count")
}

pub(super) fn model_header_name() -> HeaderName {
    HeaderName::from_static("x-llmctl-model")
}

pub(super) fn upstream_model_header_name() -> HeaderName {
    HeaderName::from_static("x-llmctl-upstream-model")
}

pub(super) fn quota_decision_header_name() -> HeaderName {
    HeaderName::from_static("x-llmctl-quota-decision")
}
