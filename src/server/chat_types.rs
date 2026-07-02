use super::ResolvedExternalProvider;
use crate::native;
use serde::Deserialize;
use serde_json::{json, Value};

/// Extracted gen_ai semantic-convention request parameters for lifecycle span
/// instrumentation.  Fields are bounded to prevent large allocations from
/// prompt-heavy payloads.
#[derive(Debug, Clone, Default)]
pub(super) struct GenAiRequestParams {
    pub(super) max_tokens: Option<u32>,
    pub(super) temperature: Option<f32>,
    /// First `system` role message body, truncated to 1 000 chars.
    pub(super) system_message: Option<String>,
    /// Last `user` role message body, truncated to 1 000 chars.
    pub(super) user_message: Option<String>,
}

/// Extracts gen_ai observability parameters from a [`ChatCompletionRequest`].
///
/// Message bodies are truncated to 1 000 chars to bound span payload size.
pub(super) fn gen_ai_params_from_request(request: &ChatCompletionRequest) -> GenAiRequestParams {
    let system_message = request
        .messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| native::message_content_text(m).chars().take(1000).collect());
    let user_message = request
        .messages
        .iter()
        .rfind(|m| m.role == "user")
        .map(|m| native::message_content_text(m).chars().take(1000).collect());
    GenAiRequestParams {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        system_message,
        user_message,
    }
}

#[derive(Debug, Deserialize)]
pub(super) struct ChatCompletionRequest {
    pub(super) model: String,
    #[serde(default)]
    pub(super) messages: Vec<native::NativeChatMessage>,
    #[serde(default)]
    pub(super) temperature: Option<f32>,
    #[serde(default)]
    pub(super) max_tokens: Option<u32>,
    #[serde(default)]
    pub(super) stream: bool,
    #[serde(default)]
    pub(super) metadata: Option<Value>,
    #[serde(default)]
    pub(super) tools: Option<Value>,
    #[serde(default)]
    pub(super) tool_choice: Option<Value>,
}

#[derive(Debug, Clone)]
pub(super) struct ToolAuditDetail {
    pub(super) tool_schema_count: u64,
    pub(super) tool_choice: Value,
    pub(super) tool_call_count: u64,
}

impl ChatCompletionRequest {
    pub(super) fn tool_audit_detail(&self) -> ToolAuditDetail {
        ToolAuditDetail {
            tool_schema_count: self
                .tools
                .as_ref()
                .and_then(Value::as_array)
                .map(|tools| tools.len() as u64)
                .unwrap_or(0),
            tool_choice: safe_tool_choice(self.tool_choice.as_ref()),
            tool_call_count: self
                .messages
                .iter()
                .filter_map(|message| message.tool_calls.as_ref())
                .map(tool_call_count)
                .sum(),
        }
    }
}

fn tool_call_count(value: &Value) -> u64 {
    value
        .as_array()
        .map(|calls| calls.len() as u64)
        .unwrap_or(1)
}

fn safe_tool_choice(value: Option<&Value>) -> Value {
    match value {
        None => Value::Null,
        Some(Value::String(choice)) => Value::String(choice.clone()),
        Some(Value::Object(object)) => {
            let mut safe = serde_json::Map::new();
            if let Some(choice_type) = object.get("type").and_then(Value::as_str) {
                safe.insert("type".to_string(), Value::String(choice_type.to_string()));
            }
            if let Some(function_name) = object
                .get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
            {
                safe.insert(
                    "function_name".to_string(),
                    Value::String(function_name.to_string()),
                );
            }
            Value::Object(safe)
        }
        Some(other) => json!({ "type": other_type_name(other) }),
    }
}

fn other_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

pub(super) fn chat_audit_detail(tool: &ToolAuditDetail, mut detail: Value) -> Value {
    if let Some(object) = detail.as_object_mut() {
        object.insert(
            "tool_schema_count".to_string(),
            Value::from(tool.tool_schema_count),
        );
        object.insert("tool_choice".to_string(), tool.tool_choice.clone());
        object.insert(
            "tool_call_count".to_string(),
            Value::from(tool.tool_call_count),
        );
    }
    detail
}

pub(super) fn chat_route_audit_detail(
    tool: &ToolAuditDetail,
    detail: Value,
    provider: Option<&ResolvedExternalProvider>,
) -> Value {
    let mut detail = chat_audit_detail(tool, detail);
    if let Some(object) = detail.as_object_mut() {
        if let Some(provider) = provider {
            object.insert("provider_routing".to_string(), json!("external"));
            object.insert("provider_id".to_string(), json!(provider.id.as_str()));
            object.insert("provider_kind".to_string(), json!(provider.kind));
            object.insert("provider_api_key_source".to_string(), json!("env"));
        } else {
            object.insert("provider_routing".to_string(), json!("local"));
        }
    }
    detail
}

pub(super) fn sanitize_native_chat_message(
    mut message: native::NativeChatMessage,
) -> native::NativeChatMessage {
    message.tool_calls = message.tool_calls.map(sanitize_tool_calls);
    message
}

fn sanitize_tool_calls(value: Value) -> Value {
    match value {
        Value::Array(calls) => Value::Array(calls.into_iter().map(sanitize_tool_call).collect()),
        other => sanitize_tool_call(other),
    }
}

fn sanitize_tool_call(value: Value) -> Value {
    let Value::Object(mut object) = value else {
        return value;
    };
    if let Some(Value::Object(function)) = object.get_mut("function") {
        function.remove("arguments");
    }
    Value::Object(object)
}
