use rs_llmctl_client::{AskConfig, LlmctlClient, Question};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("LLMCTL_SMOKE_MODEL").unwrap_or_else(|_| "llama".to_string());
    let question = std::env::var("LLMCTL_SMOKE_QUESTION").unwrap_or_else(|_| "hello".to_string());
    let client = LlmctlClient::from_env()?;
    let answer = client
        .ask_question(
            AskConfig::new(model).system("Answer concisely."),
            Question::new(question),
        )
        .await?;

    println!("{answer}");

    Ok(())
}
