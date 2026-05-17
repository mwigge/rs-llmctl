# Client SDK And Tool Loops

Application code should use the separate `rs-llmctl-client` crate. Keep the
server crate as the operator binary and API implementation; use the client SDK
for Rust applications that call `/v1/models`, `/v1/chat/completions`,
`/v1/local/search`, and `/v1/local/recommendations`.

```toml
[dependencies]
rs-llmctl-client = "1.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

## Basic Rust Client

```rust
use rs_llmctl_client::{AskConfig, LlmctlClient, Question};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = LlmctlClient::from_env()?;
    let answer = client
        .ask_question(
            AskConfig::new("qwen")
                .system("Answer from the approved knowledge base.")
                .max_tokens(256)
                .metadata(serde_json::json!({
                    "session_id": "support-ticket-1842",
                    "lineage_ids": ["prompt:support-triage-v3", "corpus:kbase-v7"]
                })),
            Question::new("Summarize the restart procedure."),
        )
        .await?;

    println!("{answer}");
    Ok(())
}
```

`LlmctlClient::from_env()` accepts `LLMCTL_BASE_URL`/`LLMCTL_API_KEY`,
`RS_LLMCTL_BASE_URL`/`RS_LLMCTL_API_KEY`, and OpenAI-compatible
`OPENAI_BASE_URL`/`OPENAI_API_KEY`. The `LLMCTL_*` names win when both are set.
The SDK sends normal bearer-token requests. Server responses include request
IDs, model aliases, quota state, and policy status, but not prompts, raw API
keys, upstream URLs, file paths, or bearer tokens.

Use the lower-level chat request when the application needs full control:

```rust
use rs_llmctl_client::{ChatCompletionRequest, ChatMessage, LlmctlClient};

async fn raw_chat(client: &LlmctlClient) -> anyhow::Result<String> {
    let response = client
        .chat(ChatCompletionRequest::new(
            "qwen",
            vec![ChatMessage::user("Summarize the restart procedure.")],
        ))
        .await?;
    Ok(response.text().unwrap_or_default().to_string())
}
```

## Local-First Provider Abstraction

The SDK exposes a local-first provider abstraction so application code can name
provider intent without bypassing rs-llmctl policy. `ProviderKind::LocalLlmctl`
routes to the configured local endpoint and keeps policy, quota, audit, and
model selection inside the local control plane.

`ProviderKind::OpenAiCompatible`, `ProviderKind::VertexAi`, and
`ProviderKind::OpenRouter` are contract-only provider metadata for a future
server-side egress adapter. They are not routable from the SDK and cannot build a
direct provider client from provider API-key environment variables. The SDK
keeps the provider capability flag `routes_external_provider_traffic = false`
for this native-only release so all application calls go through local
rs-llmctl policy, quota, audit, and model selection.

## Sessions

Sessions are client-managed. `rs-llmctl` does not keep hidden conversation
state between calls, so the application or SDK session object must retain the
message history and send the needed history on each request. Use
`metadata.session_id` for audit joins and `metadata.lineage_ids` or the
`x-llmctl-lineage-id` header to connect requests to prompts, corpora, indexes,
models, and releases.

```rust
use rs_llmctl_client::{ChatMessage, LlmctlClient, Session};

async fn continue_session(client: &LlmctlClient) -> anyhow::Result<String> {
    let mut session = Session::new("incident-2026-05-17", "qwen");
    session.push(ChatMessage::system("Keep answers short and operational."));
    session.push(ChatMessage::user("What changed in the last deploy?"));

    let first = client.chat_session(&session).await?;
    if let Some(message) = first.assistant_message() {
        session.push(message);
    }
    session.push(ChatMessage::user("List the rollback gate."));

    let second = client.chat_session(&session).await?;
    Ok(second.text().unwrap_or_default().to_string())
}
```

## Client-Side Tool Loops

Tool execution belongs in the client process or the caller's orchestration
layer. The loop is:

1. Send the model the current messages, tool definitions, and stable session
   metadata.
2. Inspect the assistant response for tool calls.
3. Validate the requested tool name and arguments against local policy.
4. Execute the tool locally with the caller's credentials and audit context.
5. Append a tool-result message to the same session.
6. Resubmit the full session to `/v1/chat/completions`.

```rust
use rs_llmctl_client::{
    ChatCompletionRequest, ChatMessage, ChatTool, ChatToolFunction, LlmctlClient,
};

async fn run_tool_loop(client: &LlmctlClient) -> anyhow::Result<String> {
    let mut messages = vec![ChatMessage::user("Check whether worker-a is ready.")];
    let tools = vec![ChatTool::function(ChatToolFunction {
        name: "readiness".to_string(),
        description: Some("Return service readiness for an approved service name.".to_string()),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": { "service": { "type": "string" } },
            "required": ["service"]
        })),
    })];

    loop {
        let mut request = ChatCompletionRequest::new("qwen", messages.clone());
        request.tools = Some(tools.clone());
        request.tool_choice = Some(serde_json::json!("auto"));
        request.metadata = Some(serde_json::json!({"session_id": "ops-readiness-42"}));
        let response = client.chat(request).await?;

        let Some(call) = response.first_tool_call() else {
            return Ok(response.text().unwrap_or_default().to_string());
        };

        anyhow::ensure!(call.name() == "readiness", "tool not allowed");
        let args = call.arguments_json()?;
        let service = args["service"].as_str().unwrap_or_default();
        anyhow::ensure!(service == "worker-a", "service not allowed");

        if let Some(message) = response.assistant_message() {
            messages.push(message);
        }
        messages.push(ChatMessage::tool_result(
            call.id.clone(),
            serde_json::json!({"ready": true, "source": "readiness-api"}).to_string(),
        ));
    }
}
```

On the compatibility backend, OpenAI-style tool fields are forwarded to the
configured upstream. On the Candle-native backend, the current stable contract
is chat messages plus metadata; use client-side JSON conventions or the
compatibility backend when a model must emit formal OpenAI tool-call objects.
In both cases, `rs-llmctl` audits and meters the chat requests, while the client
owns tool authorization, side effects, retries, and secret handling.

## Embeddings

The client also wraps the OpenAI-compatible `/v1/embeddings` endpoint.
On the Candle-native backend, production embeddings require a semantic native
embedding engine. Configure `[runtime.embeddings] mode = "semantic"` with a
`model-alias` that points to a safetensors BERT-style embedding model with
`tokenizer.json` and `config.json`. The deterministic local vectorizer remains
available only as explicit `mode = "dev-fallback"` and responses label it as
`non-semantic-dev-fallback`.

```rust
use rs_llmctl_client::{EmbeddingRequest, LlmctlClient};

async fn embed(client: &LlmctlClient) -> anyhow::Result<Vec<f32>> {
    let response = client
        .embeddings(EmbeddingRequest::new("embed", "restart worker safely"))
        .await?;
    Ok(response.data.first().map(|item| item.embedding.clone()).unwrap_or_default())
}
```

## TLS

The SDK should talk to production `rs-llmctl` over HTTPS through the platform
edge: Envoy, NGINX, HAProxy, a cloud load balancer, ingress, or service mesh.
The server config records that review evidence:

```toml
[security.tls-termination]
enabled = true
provider = "envoy-edge"
evidence = "change-record-or-runbook-url"
m-tls = true
```

The `llmctl` binary uses Rustls-backed clients for outbound HTTPS, including
verified model downloads, compatibility upstream calls, OTel export, and
Postgres TLS. Inbound serving can either use platform TLS termination or native
Rustls serving:

```toml
[server.tls]
enabled = true
cert-path = "/etc/rs-llmctl/tls/server.crt"
key-path = "/etc/rs-llmctl/tls/server.key"
require-client-cert = false
```

Native TLS currently supports server certificates only. Use the platform edge
or service mesh when the deployment requires mTLS client certificate
validation.

## Runtime Caveats

Qwen3, Gemma-family, Mistral safetensors, and DeepSeek safetensors artifacts
use Candle support where the architecture and artifact format are exposed.
DeepSeek GGUF, Kimi, and MiniMax are tracked native product targets but fail
closed until reviewed Candle-compatible decoders are wired and verified.
Kimi is tracked as a product target but fails closed, which keeps the runtime
contract honest for operators planning native capacity.

The native scheduler now has an implemented FIFO queue for chat requests with
bounded per-engine concurrency and observable queue/admission wait metadata.
Admitted requests also carry deterministic prefill/decode phase scheduling
metadata. Continuous batching and low-level KV-cache scheduler controls remain
contract metadata with `implemented=false`, so capacity planning should not
assume continuous batching behavior or cross-request KV reuse until the runtime
implementation reports those fields as active. The SDK mirrors that boundary
with `SchedulerContract::fifo_runtime`: queue discipline,
admission/backpressure, and admission-time cancelled metadata checks are
implemented, while continuous batching knobs, KV cache budget/key metadata,
cancellation token metadata, and token-level decode loop cancellation remain
serialized for contracts and tests with their execution fields set to
`implemented=false`.
