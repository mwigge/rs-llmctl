use reqwest::StatusCode;
use serde::Deserialize;
use std::fmt;

#[derive(Debug)]
pub enum LlmctlError {
    Auth { message: String },
    Quota { message: String },
    RateLimited { message: String },
    Timeout { message: String },
    ModelUnavailable { message: String },
    BadRequest { message: String },
    Server { status: u16, message: String },
    Transport { message: String },
    Decode { message: String },
}

impl fmt::Display for LlmctlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auth { message }
            | Self::Quota { message }
            | Self::RateLimited { message }
            | Self::Timeout { message }
            | Self::ModelUnavailable { message }
            | Self::BadRequest { message }
            | Self::Transport { message }
            | Self::Decode { message } => f.write_str(message),
            Self::Server { status, message } => write!(f, "server error {status}: {message}"),
        }
    }
}

impl std::error::Error for LlmctlError {}

pub(crate) fn error_from_response(status: StatusCode, body: Option<String>) -> LlmctlError {
    let parsed = body
        .as_deref()
        .and_then(|body| serde_json::from_str::<ErrorEnvelope>(body).ok())
        .map(|envelope| envelope.error);
    let code = parsed
        .as_ref()
        .and_then(|error| error.code.as_deref().or(error.kind.as_deref()))
        .unwrap_or_else(|| status.canonical_reason().unwrap_or("error"));
    let message = parsed
        .as_ref()
        .map(|error| error.message.clone())
        .or(body)
        .unwrap_or_else(|| {
            status
                .canonical_reason()
                .unwrap_or("request failed")
                .to_string()
        });

    match (status, code) {
        (StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN, _) => LlmctlError::Auth { message },
        (StatusCode::TOO_MANY_REQUESTS, "quota_exceeded") => LlmctlError::Quota { message },
        (StatusCode::TOO_MANY_REQUESTS, _) => LlmctlError::RateLimited { message },
        (StatusCode::REQUEST_TIMEOUT | StatusCode::GATEWAY_TIMEOUT, _) | (_, "timeout") => {
            LlmctlError::Timeout { message }
        }
        (StatusCode::BAD_GATEWAY | StatusCode::SERVICE_UNAVAILABLE, _)
        | (_, "upstream_unavailable")
        | (_, "upstream_circuit_open")
        | (_, "unknown_model") => LlmctlError::ModelUnavailable { message },
        (StatusCode::BAD_REQUEST, _) => LlmctlError::BadRequest { message },
        (status, _) if status.is_client_error() => LlmctlError::BadRequest { message },
        (status, _) => LlmctlError::Server {
            status: status.as_u16(),
            message,
        },
    }
}

pub(crate) fn transport_error(err: reqwest::Error) -> LlmctlError {
    if err.is_timeout() {
        LlmctlError::Timeout {
            message: err.to_string(),
        }
    } else if err.is_decode() {
        LlmctlError::Decode {
            message: err.to_string(),
        }
    } else {
        LlmctlError::Transport {
            message: err.to_string(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    message: String,
    #[serde(rename = "type")]
    kind: Option<String>,
    code: Option<String>,
}
