use super::{
    auth_error_response, auth_source_key, authenticate_request, draining_response, error_response,
    record_audit, record_request_lineage_joins, request_id_from_headers,
    runtime_lineage_from_headers_and_metadata, with_request_id, ServerState,
};
use crate::rag::{lexical_search, SearchDocument};
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub(super) struct LocalSearchRequest {
    query: String,
    #[serde(default = "default_search_limit")]
    limit: usize,
    #[serde(default)]
    metadata: Option<Value>,
    documents: Vec<SearchDocument>,
}

fn default_search_limit() -> usize {
    10
}

pub(super) async fn local_search(
    State(state): State<Arc<ServerState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    Json(request): Json<LocalSearchRequest>,
) -> Response {
    let request_id = request_id_from_headers(&headers);
    if let Some(response) = draining_response(&state, request_id) {
        return response;
    }
    let principal = match authenticate_request(
        &state,
        &headers,
        auth_source_key(&state.cfg, &headers, connect_info),
    ) {
        Ok(principal) => principal,
        Err(err) => {
            return with_request_id(auth_error_response(err), request_id);
        }
    };

    if !principal.has_scope("chat") {
        return with_request_id(
            error_response(
                StatusCode::FORBIDDEN,
                "forbidden",
                "missing chat scope".to_string(),
            ),
            request_id,
        );
    }

    let hits = lexical_search(&request.query, &request.documents, request.limit.min(50));
    let lineage = runtime_lineage_from_headers_and_metadata(&headers, request.metadata.as_ref());
    record_request_lineage_joins(&state, request_id, &lineage, None, "local.search").await;
    record_audit(
        &state,
        Some(request_id),
        principal,
        "local.search",
        "documents",
        "allowed",
        json!({ "documents": request.documents.len(), "hits": hits.len() }),
    )
    .await;
    with_request_id(
        Json(json!({
            "object": "search.results",
            "query": request.query,
            "data": hits
        }))
        .into_response(),
        request_id,
    )
}

#[derive(Debug, Deserialize)]
pub(super) struct LocalRecommendationRequest {
    task: String,
    #[serde(default = "default_search_limit")]
    limit: usize,
    #[serde(default)]
    metadata: Option<Value>,
    documents: Vec<SearchDocument>,
}

pub(super) async fn local_recommendations(
    State(state): State<Arc<ServerState>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    Json(request): Json<LocalRecommendationRequest>,
) -> Response {
    let request_id = request_id_from_headers(&headers);
    if let Some(response) = draining_response(&state, request_id) {
        return response;
    }
    let principal = match authenticate_request(
        &state,
        &headers,
        auth_source_key(&state.cfg, &headers, connect_info),
    ) {
        Ok(principal) => principal,
        Err(err) => {
            return with_request_id(auth_error_response(err), request_id);
        }
    };

    if !principal.has_scope("chat") {
        return with_request_id(
            error_response(
                StatusCode::FORBIDDEN,
                "forbidden",
                "missing chat scope".to_string(),
            ),
            request_id,
        );
    }

    let hits = lexical_search(&request.task, &request.documents, request.limit.min(50));
    let recommendations = local_recommendation_items(&request.task, &hits);
    let lineage = runtime_lineage_from_headers_and_metadata(&headers, request.metadata.as_ref());
    record_request_lineage_joins(&state, request_id, &lineage, None, "local.recommendations").await;
    record_audit(
        &state,
        Some(request_id),
        principal,
        "local.recommendations",
        "documents",
        "allowed",
        json!({
            "documents": request.documents.len(),
            "hits": hits.len(),
            "recommendations": recommendations.len()
        }),
    )
    .await;
    with_request_id(
        Json(json!({
            "object": "recommendation.results",
            "task": request.task,
            "data": hits,
            "recommendations": recommendations
        }))
        .into_response(),
        request_id,
    )
}

fn local_recommendation_items(
    task: &str,
    hits: &[crate::rag::SearchHit],
) -> Vec<BTreeMap<&'static str, String>> {
    hits.iter()
        .take(5)
        .enumerate()
        .map(|(index, hit)| {
            let title = hit.title.clone().unwrap_or_else(|| hit.id.clone());
            BTreeMap::from([
                ("rank", (index + 1).to_string()),
                ("document_id", hit.id.clone()),
                ("title", title.clone()),
                ("reason", recommendation_reason(task, &title)),
            ])
        })
        .collect()
}

fn recommendation_reason(task: &str, title: &str) -> String {
    let task = task.trim();
    if task.is_empty() {
        return format!("Use `{title}` as supporting local context.");
    }
    format!("Use `{title}` because it matches local context for `{task}`.")
}
