use crate::*;
use reqwest::{header, Url};
use serde::Deserialize;

pub(crate) async fn decode_response<T>(response: reqwest::Response) -> Result<LlmctlResponse<T>>
where
    T: for<'de> Deserialize<'de>,
{
    let status = response.status();
    let metadata = ResponseMetadata::from_headers(response.headers());
    if !status.is_success() {
        return Err(error_from_response(status, response.text().await.ok()));
    }
    response
        .json::<T>()
        .await
        .map(|body| LlmctlResponse { metadata, body })
        .map_err(|err| LlmctlError::Decode {
            message: format!("failed to decode response JSON: {err}"),
        })
}

pub(crate) fn decode_sse_event(event: &str) -> Result<Option<ChatCompletionChunk>> {
    let data = event
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n");
    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }
    serde_json::from_str(&data)
        .map(Some)
        .map_err(|err| LlmctlError::Decode {
            message: format!("failed to decode stream event: {err}"),
        })
}

pub(crate) fn normalize_base_url(raw: &str) -> Result<Url> {
    let mut raw = raw.trim().trim_end_matches('/');
    if let Some(stripped) = raw.strip_suffix("/v1") {
        raw = stripped.trim_end_matches('/');
    }
    let raw = if raw.starts_with("http://") || raw.starts_with("https://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    };
    Url::parse(&(raw + "/")).map_err(|err| LlmctlError::BadRequest {
        message: format!("invalid base URL: {err}"),
    })
}

pub(crate) fn header_value(headers: &header::HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
}

pub(crate) fn local_from_env_values(mut get: impl FnMut(&str) -> Option<String>) -> Result<String> {
    first_env_value(&["LLMCTL_BASE_URL", "RS_LLMCTL_BASE_URL"], &mut get).ok_or_else(|| {
        LlmctlError::BadRequest {
            message: "LLMCTL_BASE_URL or RS_LLMCTL_BASE_URL must be set".to_string(),
        }
    })
}

pub(crate) fn client_from_provider_env_values(
    provider: ProviderKind,
    mut get: impl FnMut(&str) -> Option<String>,
) -> Result<LlmctlClient> {
    let contract = ProviderContract::for_kind(provider);
    contract.validate_routable()?;
    let base_url = first_provider_env_value(&contract.base_url_env, &mut get).ok_or_else(|| {
        LlmctlError::BadRequest {
            message: format!("one of {} must be set", contract.base_url_env.join(", ")),
        }
    })?;
    let api_key = first_provider_env_value(&contract.api_key_env, &mut get);
    LlmctlClient::new(base_url, api_key)
}

pub(crate) fn first_provider_env_value(
    names: &[String],
    mut get: impl FnMut(&str) -> Option<String>,
) -> Option<String> {
    names
        .iter()
        .find_map(|name| get(name).filter(|value| !value.trim().is_empty()))
}

pub(crate) fn first_env_value(
    names: &[&str],
    mut get: impl FnMut(&str) -> Option<String>,
) -> Option<String> {
    names
        .iter()
        .find_map(|name| get(name).filter(|value| !value.trim().is_empty()))
}
