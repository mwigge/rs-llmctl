use rs_llmctl_client::{AskConfig, LlmctlClient, Question};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmctlClient::from_env()?;
    let answer = client
        .ask_question(
            AskConfig::new("llama").system("Answer concisely."),
            Question::new("hello"),
        )
        .await?;

    println!("{answer}");

    Ok(())
}
