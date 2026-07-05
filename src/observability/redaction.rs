use super::*;

pub const REDACTED_ATTRIBUTE_VALUE: &str = "[REDACTED]";
pub fn sanitize_otel_attributes(attributes: BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    attributes
        .into_iter()
        .map(|(key, value)| {
            if should_redact_attribute(&key, &value) {
                (key, Value::String(REDACTED_ATTRIBUTE_VALUE.to_string()))
            } else {
                (key, value)
            }
        })
        .collect()
}

fn should_redact_attribute(key: &str, value: &Value) -> bool {
    let normalized = key.to_ascii_lowercase().replace(['-', '.'], "_");

    if normalized.contains("authorization")
        || normalized.contains("api_key")
        || normalized.contains("apikey")
        || normalized.contains("secret")
        || normalized.contains("password")
        || normalized.contains("token")
        || normalized.contains("bearer")
        || normalized.contains("prompt")
        || normalized.contains("message")
        || normalized == "content"
        || normalized.ends_with("_content")
        || normalized.contains("otlp_headers")
        || normalized.contains("collector_header")
        || normalized.contains("exporter_header")
        || normalized.contains("header_authorization")
        || normalized.ends_with("_path")
        || normalized.contains("_path_")
        || normalized == "path"
    {
        return true;
    }

    string_value_contains_secret(value)
}
fn string_value_contains_secret(value: &Value) -> bool {
    value.as_str().is_some_and(|value| {
        let normalized = value.to_ascii_lowercase();
        normalized.contains("bearer ")
            || normalized.contains("api_key=")
            || normalized.contains("authorization=")
            || normalized.contains("x-api-key")
    })
}
