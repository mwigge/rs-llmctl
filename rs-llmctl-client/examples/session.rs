use rs_llmctl_client::{ChatMessage, LlmctlClient, Session};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmctlClient::from_env()?;
    let mut session = Session::new("example-session", "llama");
    session.push(ChatMessage::system("Answer briefly."));
    session.push(ChatMessage::user("What is the current model status?"));

    let first = client.chat_session(&session).await?;
    if let Some(message) = first.assistant_message() {
        session.push(message);
    }
    session.push(ChatMessage::user("Now summarize that in one sentence."));

    let second = client.chat_session(&session).await?;
    if let Some(text) = second.text() {
        println!("{text}");
    }
    Ok(())
}
