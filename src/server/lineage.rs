use super::{corpus_header_name, lineage_id_header_name, lineage_ids_header_name, ServerState};
use crate::storage::RequestLineageJoinRecord;
use axum::http::{HeaderMap, HeaderName};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use uuid::Uuid;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct RuntimeLineageMetadata {
    lineage_ids: Vec<String>,
    corpus: Option<String>,
}

pub(super) async fn record_request_lineage_joins(
    state: &ServerState,
    request_id: Uuid,
    lineage: &RuntimeLineageMetadata,
    model: Option<&str>,
    source: &str,
) {
    for lineage_id in &lineage.lineage_ids {
        let lineage_id = sanitize_lineage_value(lineage_id);
        let corpus = lineage.corpus.as_deref().map(sanitize_lineage_value);
        let record = RequestLineageJoinRecord::new(
            request_id,
            lineage_id,
            model.map(str::to_string),
            corpus,
            source,
        );
        if let Err(err) = state.storage.insert_request_lineage_join(&record).await {
            tracing::warn!(error = %err, "failed to record request lineage join");
        }
    }
}

pub(super) fn runtime_lineage_from_headers_and_metadata(
    headers: &HeaderMap,
    metadata: Option<&Value>,
) -> RuntimeLineageMetadata {
    let mut lineage = RuntimeLineageMetadata::default();
    extend_lineage_ids_from_header(headers, lineage_id_header_name(), &mut lineage.lineage_ids);
    extend_lineage_ids_from_header(headers, lineage_ids_header_name(), &mut lineage.lineage_ids);
    if let Some(corpus) = header_string(headers, corpus_header_name()) {
        lineage.corpus = Some(corpus);
    }

    if let Some(metadata) = metadata.and_then(|value| value.as_object()) {
        extend_lineage_ids_from_value(metadata.get("lineage_id"), &mut lineage.lineage_ids);
        extend_lineage_ids_from_value(metadata.get("lineage_ids"), &mut lineage.lineage_ids);
        if lineage.corpus.is_none() {
            lineage.corpus = metadata
                .get("corpus")
                .or_else(|| metadata.get("corpus_id"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string);
        }
    }

    let mut seen = BTreeSet::new();
    lineage
        .lineage_ids
        .retain(|lineage_id| seen.insert(lineage_id.clone()));
    lineage
}

fn extend_lineage_ids_from_header(
    headers: &HeaderMap,
    name: HeaderName,
    lineage_ids: &mut Vec<String>,
) {
    for value in headers.get_all(name) {
        if let Ok(value) = value.to_str() {
            extend_lineage_ids_from_str(value, lineage_ids);
        }
    }
}

fn extend_lineage_ids_from_value(value: Option<&Value>, lineage_ids: &mut Vec<String>) {
    match value {
        Some(Value::String(value)) => extend_lineage_ids_from_str(value, lineage_ids),
        Some(Value::Array(values)) => {
            for value in values {
                if let Some(value) = value.as_str() {
                    extend_lineage_ids_from_str(value, lineage_ids);
                }
            }
        }
        _ => {}
    }
}

fn extend_lineage_ids_from_str(raw: &str, lineage_ids: &mut Vec<String>) {
    lineage_ids.extend(
        raw.split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(sanitize_lineage_value),
    );
}

fn sanitize_lineage_value(raw: &str) -> String {
    let value = raw.trim();
    if value.len() <= 128
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-' | ':'))
        && !looks_sensitive_lineage_value(value)
    {
        return value.to_string();
    }
    let digest = Sha256::digest(value.as_bytes());
    format!("redacted:{:x}", digest)[..25].to_string()
}

fn looks_sensitive_lineage_value(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    value.contains('/')
        || value.contains('\\')
        || lower.contains("bearer ")
        || lower.contains("apikey")
        || lower.contains("api_key")
        || lower.contains("token")
        || lower.contains("secret")
        || lower.contains("password")
}

fn header_string(headers: &HeaderMap, name: HeaderName) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}
