use super::*;

fn make_chat_request(messages: Vec<native::NativeChatMessage>) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "test-model".to_string(),
        messages,
        temperature: Some(0.7),
        max_tokens: Some(512),
        top_p: None,
        top_k: None,
        presence_penalty: None,
        frequency_penalty: None,
        seed: None,
        stop: None,
        n: None,
        stream: false,
        metadata: None,
        tools: None,
        tool_choice: None,
    }
}

#[test]
fn stop_sequences_normalizes_string_and_array_forms() {
    use serde_json::json;
    let mut request = make_chat_request(Vec::new());
    // Absent → None.
    assert_eq!(request.stop_sequences(), None);
    // Single string → one-element vec.
    request.stop = Some(json!("STOP"));
    assert_eq!(request.stop_sequences(), Some(vec!["STOP".to_string()]));
    // Array of strings, dropping empty entries.
    request.stop = Some(json!(["</s>", "", "END"]));
    assert_eq!(
        request.stop_sequences(),
        Some(vec!["</s>".to_string(), "END".to_string()])
    );
    // Array with only empty entries collapses to None.
    request.stop = Some(json!([""]));
    assert_eq!(request.stop_sequences(), None);
}

#[test]
fn gen_ai_params_extracts_system_and_user_messages() {
    use serde_json::json;
    let request = make_chat_request(vec![
        native::NativeChatMessage {
            role: "system".to_string(),
            content: Some(json!("You are helpful.")),
            tool_calls: None,
            tool_call_id: None,
        },
        native::NativeChatMessage {
            role: "user".to_string(),
            content: Some(json!("Hello")),
            tool_calls: None,
            tool_call_id: None,
        },
    ]);
    let params = gen_ai_params_from_request(&request);
    assert_eq!(params.system_message.as_deref(), Some("You are helpful."));
    assert_eq!(params.user_message.as_deref(), Some("Hello"));
    assert_eq!(params.max_tokens, Some(512));
    assert!((params.temperature.unwrap() - 0.7).abs() < f32::EPSILON);
}

#[test]
fn gen_ai_params_truncates_long_messages_to_1000_chars() {
    use serde_json::json;
    let long_text: String = "x".repeat(2000);
    let request = make_chat_request(vec![native::NativeChatMessage {
        role: "user".to_string(),
        content: Some(json!(long_text)),
        tool_calls: None,
        tool_call_id: None,
    }]);
    let params = gen_ai_params_from_request(&request);
    assert_eq!(params.user_message.as_ref().map(|s| s.len()), Some(1000));
}

#[test]
fn gen_ai_params_finds_last_user_message() {
    use serde_json::json;
    let request = make_chat_request(vec![
        native::NativeChatMessage {
            role: "user".to_string(),
            content: Some(json!("first")),
            tool_calls: None,
            tool_call_id: None,
        },
        native::NativeChatMessage {
            role: "assistant".to_string(),
            content: Some(json!("response")),
            tool_calls: None,
            tool_call_id: None,
        },
        native::NativeChatMessage {
            role: "user".to_string(),
            content: Some(json!("last")),
            tool_calls: None,
            tool_call_id: None,
        },
    ]);
    let params = gen_ai_params_from_request(&request);
    assert_eq!(params.user_message.as_deref(), Some("last"));
}

#[test]
fn gen_ai_params_returns_none_for_missing_roles() {
    use serde_json::json;
    let request = make_chat_request(vec![native::NativeChatMessage {
        role: "assistant".to_string(),
        content: Some(json!("I can help.")),
        tool_calls: None,
        tool_call_id: None,
    }]);
    let params = gen_ai_params_from_request(&request);
    assert!(params.system_message.is_none());
    assert!(params.user_message.is_none());
}
