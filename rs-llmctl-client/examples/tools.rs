use rs_llmctl_client::{
    ChatCompletionRequest, ChatMessage, ChatTool, ChatToolFunction, LlmctlClient,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmctlClient::from_env()?;
    let tool = ChatTool::function(ChatToolFunction {
        name: "readiness".to_string(),
        description: Some("Return readiness for an approved service.".to_string()),
        parameters: Some(json!({
            "type": "object",
            "properties": {"service": {"type": "string"}},
            "required": ["service"]
        })),
    });
    let mut messages = vec![ChatMessage::user("Is worker-a ready?")];

    loop {
        let mut request = ChatCompletionRequest::new("llama", messages.clone());
        request.tools = Some(vec![tool.clone()]);
        request.tool_choice = Some(json!("auto"));
        let response = client.chat(request).await?;

        let Some(call) = response.first_tool_call() else {
            if let Some(text) = response.text() {
                println!("{text}");
            }
            break;
        };
        if call.name() != "readiness" {
            return Err(format!("tool not allowed: {}", call.name()).into());
        }
        let args = call.arguments_json()?;
        if args.get("service").and_then(|value| value.as_str()) != Some("worker-a") {
            return Err("service not allowed".into());
        }

        if let Some(message) = response.assistant_message() {
            messages.push(message);
        }
        messages.push(ChatMessage::tool_result(
            call.id.clone(),
            json!({"ready": true, "source": "example"}).to_string(),
        ));
    }

    Ok(())
}
