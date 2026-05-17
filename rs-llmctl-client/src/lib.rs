use futures_util::{Stream, StreamExt};
use reqwest::{header, StatusCode, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::env;
use std::fmt;
use std::pin::Pin;
use std::time::Duration;

const DEFAULT_CLIENT_TIMEOUT: Duration = Duration::from_secs(300);

pub type Result<T> = std::result::Result<T, LlmctlError>;
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderKind {
    #[default]
    LocalLlmctl,
    OpenAiCompatible,
    VertexAi,
    OpenRouter,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderRouting {
    LocalOnly,
    ExternalReserved,
    ExternalOpenAiCompatible,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderStatus {
    Implemented,
    ContractOnly,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderContract {
    pub kind: ProviderKind,
    pub routing: ProviderRouting,
    pub status: ProviderStatus,
    pub local_first: bool,
    pub routes_external_provider_traffic: bool,
    pub base_url_env: Vec<String>,
    pub api_key_env: Vec<String>,
}

impl ProviderContract {
    pub fn local_llmctl() -> Self {
        Self {
            kind: ProviderKind::LocalLlmctl,
            routing: ProviderRouting::LocalOnly,
            status: ProviderStatus::Implemented,
            local_first: true,
            routes_external_provider_traffic: false,
            base_url_env: vec![
                "LLMCTL_BASE_URL".to_string(),
                "RS_LLMCTL_BASE_URL".to_string(),
            ],
            api_key_env: vec![
                "LLMCTL_API_KEY".to_string(),
                "RS_LLMCTL_API_KEY".to_string(),
            ],
        }
    }

    pub fn reserved(kind: ProviderKind) -> Self {
        Self {
            kind,
            routing: ProviderRouting::ExternalReserved,
            status: ProviderStatus::ContractOnly,
            local_first: true,
            routes_external_provider_traffic: false,
            base_url_env: Vec::new(),
            api_key_env: Vec::new(),
        }
    }

    pub fn for_kind(kind: ProviderKind) -> Self {
        match kind {
            ProviderKind::LocalLlmctl => Self::local_llmctl(),
            provider => Self::reserved(provider),
        }
    }

    pub fn validate_routable(&self) -> Result<()> {
        if self.status != ProviderStatus::Implemented {
            return Err(LlmctlError::BadRequest {
                message: format!(
                    "provider {:?} is contract-only metadata and cannot route traffic",
                    self.kind
                ),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QueueDiscipline {
    Fifo,
    WeightedFair,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueContract {
    pub discipline: QueueDiscipline,
    pub admission_backpressure: bool,
    pub priority_metadata_keys: Vec<String>,
    pub implemented: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchingContract {
    pub continuous_batching: bool,
    pub max_batch_size_metadata_key: String,
    pub max_wait_ms_metadata_key: String,
    pub implemented: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheContract {
    pub cache_scope: String,
    pub cache_budget_metadata_key: String,
    pub eviction_policy: String,
    pub implemented: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CancellationContract {
    pub cancellation_token_metadata_key: String,
    pub drain_on_cancel: bool,
    pub implemented: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerContract {
    pub queue: QueueContract,
    pub batching: BatchingContract,
    pub kv_cache: KvCacheContract,
    pub cancellation: CancellationContract,
    pub contract_only: bool,
}

impl SchedulerContract {
    pub fn fifo_runtime() -> Self {
        Self {
            queue: QueueContract {
                discipline: QueueDiscipline::Fifo,
                admission_backpressure: true,
                priority_metadata_keys: vec![
                    "llmctl.scheduler.priority".to_string(),
                    "llmctl.scheduler.tenant".to_string(),
                ],
                implemented: true,
            },
            batching: BatchingContract {
                continuous_batching: false,
                max_batch_size_metadata_key: "llmctl.scheduler.max_batch_size".to_string(),
                max_wait_ms_metadata_key: "llmctl.scheduler.max_wait_ms".to_string(),
                implemented: false,
            },
            kv_cache: KvCacheContract {
                cache_scope: "model-worker".to_string(),
                cache_budget_metadata_key: "llmctl.scheduler.kv_cache_budget_bytes".to_string(),
                eviction_policy: "metadata-only-lru-target".to_string(),
                implemented: false,
            },
            cancellation: CancellationContract {
                cancellation_token_metadata_key: "llmctl.scheduler.cancel_token".to_string(),
                drain_on_cancel: true,
                implemented: false,
            },
            contract_only: false,
        }
    }

    pub fn metadata_only() -> Self {
        Self {
            queue: QueueContract {
                discipline: QueueDiscipline::Fifo,
                admission_backpressure: true,
                priority_metadata_keys: vec![
                    "llmctl.scheduler.priority".to_string(),
                    "llmctl.scheduler.tenant".to_string(),
                ],
                implemented: false,
            },
            batching: BatchingContract {
                continuous_batching: true,
                max_batch_size_metadata_key: "llmctl.scheduler.max_batch_size".to_string(),
                max_wait_ms_metadata_key: "llmctl.scheduler.max_wait_ms".to_string(),
                implemented: false,
            },
            kv_cache: KvCacheContract {
                cache_scope: "model-worker".to_string(),
                cache_budget_metadata_key: "llmctl.scheduler.kv_cache_budget_bytes".to_string(),
                eviction_policy: "metadata-only-lru-target".to_string(),
                implemented: false,
            },
            cancellation: CancellationContract {
                cancellation_token_metadata_key: "llmctl.scheduler.cancel_token".to_string(),
                drain_on_cancel: true,
                implemented: false,
            },
            contract_only: true,
        }
    }

    pub fn validate_runtime_contract(&self) -> Result<()> {
        if self.contract_only {
            return Err(LlmctlError::BadRequest {
                message: "scheduler contract must report implemented FIFO queue runtime"
                    .to_string(),
            });
        }
        if self.queue.discipline != QueueDiscipline::Fifo
            || !self.queue.implemented
            || self.batching.implemented
            || self.kv_cache.implemented
            || self.cancellation.implemented
        {
            return Err(LlmctlError::BadRequest {
                message: "scheduler must implement FIFO queue while batching, KV cache, and cancellation remain metadata-only"
                    .to_string(),
            });
        }
        Ok(())
    }

    pub fn validate_contract_only(&self) -> Result<()> {
        if !self.contract_only {
            return Err(LlmctlError::BadRequest {
                message: "scheduler contract is not metadata-only".to_string(),
            });
        }
        if self.queue.implemented
            || self.batching.implemented
            || self.kv_cache.implemented
            || self.cancellation.implemented
        {
            return Err(LlmctlError::BadRequest {
                message: "scheduler queue, batching, KV cache, and cancellation are contract-only"
                    .to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LlmctlClient {
    http: reqwest::Client,
    base_url: Url,
    api_key: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseMetadata {
    pub request_id: Option<String>,
    pub model: Option<String>,
    pub upstream_model: Option<String>,
    pub quota_decision: Option<String>,
}

impl ResponseMetadata {
    pub fn from_headers(headers: &header::HeaderMap) -> Self {
        Self {
            request_id: header_value(headers, "x-request-id"),
            model: header_value(headers, "x-llmctl-model"),
            upstream_model: header_value(headers, "x-llmctl-upstream-model"),
            quota_decision: header_value(headers, "x-llmctl-quota-decision"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LlmctlResponse<T> {
    pub metadata: ResponseMetadata,
    pub body: T,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AskConfig {
    pub provider: ProviderKind,
    pub model: String,
    pub system_prompt: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub metadata: Option<Value>,
}

impl AskConfig {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            provider: ProviderKind::LocalLlmctl,
            model: model.into(),
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            metadata: None,
        }
    }

    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn metadata(mut self, metadata: impl Into<Value>) -> Self {
        self.metadata = Some(metadata.into());
        self
    }

    fn to_request(&self, question: Question) -> Result<ChatCompletionRequest> {
        ProviderContract::for_kind(self.provider.clone()).validate_routable()?;

        let mut messages = Vec::new();
        if let Some(system_prompt) = &self.system_prompt {
            messages.push(ChatMessage::system(system_prompt.clone()));
        }
        messages.extend(question.messages);
        messages.push(ChatMessage::user(question.new_prompt));

        let mut request = ChatCompletionRequest::new(self.model.clone(), messages);
        request.temperature = self.temperature;
        request.max_tokens = self.max_tokens;
        request.metadata = self.metadata.clone();
        Ok(request)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Question {
    pub messages: Vec<ChatMessage>,
    pub new_prompt: String,
}

impl Question {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            messages: Vec::new(),
            new_prompt: prompt.into(),
        }
    }

    pub fn with_history(mut self, messages: Vec<ChatMessage>) -> Self {
        self.messages = messages;
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Session {
    id: String,
    model: String,
    messages: Vec<ChatMessage>,
    metadata: Map<String, Value>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

impl Session {
    pub fn new(id: impl Into<String>, model: impl Into<String>) -> Self {
        let id = id.into();
        let mut metadata = Map::new();
        metadata.insert("session_id".to_string(), Value::String(id.clone()));
        Self {
            id,
            model: model.into(),
            messages: Vec::new(),
            metadata,
            temperature: None,
            max_tokens: None,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn push(&mut self, message: ChatMessage) {
        self.messages.push(message);
    }

    pub fn reset(&mut self) {
        self.messages.clear();
    }

    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = Some(temperature);
    }

    pub fn set_max_tokens(&mut self, max_tokens: u32) {
        self.max_tokens = Some(max_tokens);
    }

    pub fn insert_metadata(&mut self, key: impl Into<String>, value: impl Into<Value>) {
        self.metadata.insert(key.into(), value.into());
    }

    pub fn to_request(&self) -> ChatCompletionRequest {
        let mut request = ChatCompletionRequest::new(self.model.clone(), self.messages.clone());
        request.temperature = self.temperature;
        request.max_tokens = self.max_tokens;
        request.metadata = Some(Value::Object(self.metadata.clone()));
        request
    }
}

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

impl LlmctlClient {
    pub fn new(base_url: impl AsRef<str>, api_key: impl Into<Option<String>>) -> Result<Self> {
        let base_url = normalize_base_url(base_url.as_ref())?;
        Ok(Self {
            http: reqwest::Client::builder()
                .timeout(DEFAULT_CLIENT_TIMEOUT)
                .build()
                .map_err(|err| LlmctlError::BadRequest {
                    message: format!("failed to build HTTP client: {err}"),
                })?,
            base_url,
            api_key: api_key.into(),
        })
    }

    pub fn from_env() -> Result<Self> {
        let base_url = local_from_env_values(|name| env::var(name).ok())?;
        let api_key = first_env_value(&["LLMCTL_API_KEY", "RS_LLMCTL_API_KEY"], |name| {
            env::var(name).ok()
        });
        Self::new(base_url, api_key)
    }

    pub fn from_provider_env(provider: ProviderKind) -> Result<Self> {
        client_from_provider_env_values(provider, |name| env::var(name).ok())
    }

    pub async fn ask(&self, model: impl Into<String>, prompt: impl Into<String>) -> Result<String> {
        self.ask_question(AskConfig::new(model), Question::new(prompt))
            .await
    }

    pub async fn ask_question(&self, config: AskConfig, question: Question) -> Result<String> {
        let request = config.to_request(question)?;
        let response = self.chat(request).await?;
        response
            .text()
            .map(ToOwned::to_owned)
            .ok_or_else(|| LlmctlError::Decode {
                message: "chat response did not include assistant text".to_string(),
            })
    }

    pub async fn chat(&self, mut request: ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        request.stream = Some(false);
        self.post_json("v1/chat/completions", &request)
            .await
            .map(|response| response.body)
    }

    pub async fn chat_with_session(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<LlmctlResponse<ChatCompletionResponse>> {
        request.stream = Some(false);
        self.post_json("v1/chat/completions", &request).await
    }

    pub async fn chat_stream(&self, mut request: ChatCompletionRequest) -> Result<ChatStream> {
        request.stream = Some(true);
        let response = self
            .request(reqwest::Method::POST, "v1/chat/completions")?
            .json(&request)
            .send()
            .await
            .map_err(transport_error)?;
        let status = response.status();
        if !status.is_success() {
            return Err(error_from_response(status, response.text().await.ok()));
        }

        let stream = response.bytes_stream();
        Ok(Box::pin(async_stream::try_stream! {
            let mut buffer = String::new();
            futures_util::pin_mut!(stream);
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(transport_error)?;
                let text = std::str::from_utf8(&chunk).map_err(|err| LlmctlError::Decode {
                    message: format!("stream chunk is not utf-8: {err}"),
                })?;
                buffer.push_str(text);
                while let Some(index) = buffer.find("\n\n") {
                    let event = buffer[..index].to_string();
                    buffer.drain(..index + 2);
                    if let Some(chunk) = decode_sse_event(&event)? {
                        yield chunk;
                    }
                }
            }
            if !buffer.trim().is_empty() {
                if let Some(chunk) = decode_sse_event(&buffer)? {
                    yield chunk;
                }
            }
        }))
    }

    pub async fn models(&self) -> Result<ModelList> {
        self.get_json("v1/models")
            .await
            .map(|response| response.body)
    }

    pub async fn models_with_session(&self) -> Result<LlmctlResponse<ModelList>> {
        self.get_json("v1/models").await
    }

    pub async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        self.post_json("v1/embeddings", &request)
            .await
            .map(|response| response.body)
    }

    pub async fn embeddings_with_session(
        &self,
        request: EmbeddingRequest,
    ) -> Result<LlmctlResponse<EmbeddingResponse>> {
        self.post_json("v1/embeddings", &request).await
    }

    pub async fn local_search(&self, request: LocalSearchRequest) -> Result<LocalSearchResponse> {
        self.post_json("v1/local/search", &request)
            .await
            .map(|response| response.body)
    }

    pub async fn local_search_with_session(
        &self,
        request: LocalSearchRequest,
    ) -> Result<LlmctlResponse<LocalSearchResponse>> {
        self.post_json("v1/local/search", &request).await
    }

    pub async fn local_recommendations(
        &self,
        request: LocalRecommendationsRequest,
    ) -> Result<LocalRecommendationsResponse> {
        self.post_json("v1/local/recommendations", &request)
            .await
            .map(|response| response.body)
    }

    pub async fn local_recommendations_with_session(
        &self,
        request: LocalRecommendationsRequest,
    ) -> Result<LlmctlResponse<LocalRecommendationsResponse>> {
        self.post_json("v1/local/recommendations", &request).await
    }

    pub async fn chat_session(&self, session: &Session) -> Result<ChatCompletionResponse> {
        self.chat(session.to_request()).await
    }

    pub async fn chat_session_with_metadata(
        &self,
        session: &Session,
    ) -> Result<LlmctlResponse<ChatCompletionResponse>> {
        self.chat_with_session(session.to_request()).await
    }

    fn request(&self, method: reqwest::Method, path: &str) -> Result<reqwest::RequestBuilder> {
        let url = self
            .base_url
            .join(path)
            .map_err(|err| LlmctlError::BadRequest {
                message: format!("invalid request path {path}: {err}"),
            })?;
        let mut builder = self.http.request(method, url);
        if let Some(api_key) = &self.api_key {
            builder = builder.bearer_auth(api_key);
        }
        Ok(builder)
    }

    async fn get_json<T>(&self, path: &str) -> Result<LlmctlResponse<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let response = self
            .request(reqwest::Method::GET, path)?
            .send()
            .await
            .map_err(transport_error)?;
        decode_response(response).await
    }

    async fn post_json<T, B>(&self, path: &str, body: &B) -> Result<LlmctlResponse<T>>
    where
        T: for<'de> Deserialize<'de>,
        B: Serialize + ?Sized,
    {
        let response = self
            .request(reqwest::Method::POST, path)?
            .json(body)
            .send()
            .await
            .map_err(transport_error)?;
        decode_response(response).await
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl ChatCompletionRequest {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            temperature: None,
            max_tokens: None,
            stream: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            extra: Map::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatToolCall>>,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            name: None,
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatToolFunction,
}

impl ChatTool {
    pub fn function(function: ChatToolFunction) -> Self {
        Self {
            kind: "function".to_string(),
            function,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatToolFunction {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatToolCallFunction,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatToolCallFunction {
    pub name: String,
    pub arguments: String,
}

impl ChatToolCall {
    pub fn name(&self) -> &str {
        &self.function.name
    }

    pub fn arguments_json(&self) -> Result<Value> {
        serde_json::from_str(&self.function.arguments).map_err(|err| LlmctlError::Decode {
            message: format!("failed to decode tool arguments for {}: {err}", self.id),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    #[serde(default)]
    pub created: Option<i64>,
    pub model: String,
    #[serde(default)]
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub usage: Option<TokenUsage>,
}

impl ChatCompletionResponse {
    pub fn text(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|choice| choice.message.content.as_deref())
    }

    pub fn assistant_message(&self) -> Option<ChatMessage> {
        self.choices.first().map(|choice| choice.message.clone())
    }

    pub fn first_tool_call(&self) -> Option<&ChatToolCall> {
        self.choices
            .first()
            .and_then(|choice| choice.message.tool_calls.as_ref())
            .and_then(|calls| calls.first())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    #[serde(default)]
    pub created: Option<i64>,
    pub model: String,
    #[serde(default)]
    pub choices: Vec<ChatChunkChoice>,
    #[serde(default)]
    pub usage: Option<TokenUsage>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: ChatMessageDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessageDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ChatToolCall>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    pub total_tokens: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl EmbeddingRequest {
    pub fn new(model: impl Into<String>, input: impl Into<EmbeddingInput>) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            encoding_format: None,
            metadata: None,
            extra: Map::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingInput {
    String(String),
    StringArray(Vec<String>),
}

impl From<String> for EmbeddingInput {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for EmbeddingInput {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<Vec<String>> for EmbeddingInput {
    fn from(value: Vec<String>) -> Self {
        Self::StringArray(value)
    }
}

impl From<Vec<&str>> for EmbeddingInput {
    fn from(value: Vec<&str>) -> Self {
        Self::StringArray(value.into_iter().map(ToOwned::to_owned).collect())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingResponse {
    pub object: String,
    #[serde(default)]
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    #[serde(default)]
    pub usage: Option<TokenUsage>,
    #[serde(default)]
    pub llmctl: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingObject {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelObject>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LocalSearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    pub documents: Vec<SearchDocument>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LocalSearchResponse {
    pub object: String,
    pub query: String,
    pub data: Vec<SearchHit>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LocalRecommendationsRequest {
    pub task: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    pub documents: Vec<SearchDocument>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LocalRecommendationsResponse {
    pub object: String,
    pub task: String,
    pub data: Vec<SearchHit>,
    pub recommendations: Vec<Recommendation>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchDocument {
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SearchHit {
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    pub score: f64,
    pub snippet: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Recommendation {
    pub rank: String,
    pub document_id: String,
    pub title: String,
    pub reason: String,
}

fn default_limit() -> usize {
    10
}

async fn decode_response<T>(response: reqwest::Response) -> Result<LlmctlResponse<T>>
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

fn decode_sse_event(event: &str) -> Result<Option<ChatCompletionChunk>> {
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

fn error_from_response(status: StatusCode, body: Option<String>) -> LlmctlError {
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

fn transport_error(err: reqwest::Error) -> LlmctlError {
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

fn normalize_base_url(raw: &str) -> Result<Url> {
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

fn header_value(headers: &header::HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
}

fn local_from_env_values(mut get: impl FnMut(&str) -> Option<String>) -> Result<String> {
    first_env_value(&["LLMCTL_BASE_URL", "RS_LLMCTL_BASE_URL"], &mut get).ok_or_else(|| {
        LlmctlError::BadRequest {
            message: "LLMCTL_BASE_URL or RS_LLMCTL_BASE_URL must be set".to_string(),
        }
    })
}

fn client_from_provider_env_values(
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

fn first_provider_env_value(
    names: &[String],
    mut get: impl FnMut(&str) -> Option<String>,
) -> Option<String> {
    names
        .iter()
        .find_map(|name| get(name).filter(|value| !value.trim().is_empty()))
}

fn first_env_value(names: &[&str], mut get: impl FnMut(&str) -> Option<String>) -> Option<String> {
    names
        .iter()
        .find_map(|name| get(name).filter(|value| !value.trim().is_empty()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_resolution_uses_only_llmctl_names_for_local_client() {
        let value = local_from_env_values(|name| match name {
            "LLMCTL_BASE_URL" => Some("http://llmctl".to_string()),
            "RS_LLMCTL_BASE_URL" => Some("http://rs".to_string()),
            "OPENAI_BASE_URL" => Some("http://openai/v1".to_string()),
            _ => None,
        })
        .expect("base url");

        assert_eq!(value, "http://llmctl");

        let err = local_from_env_values(|name| match name {
            "OPENAI_BASE_URL" => Some("http://openai/v1".to_string()),
            _ => None,
        })
        .expect_err("OpenAI-compatible aliases are explicit provider-only inputs");

        assert!(err.to_string().contains("LLMCTL_BASE_URL"));
    }

    #[test]
    fn normalize_base_url_accepts_openai_v1_base_url() {
        let url = normalize_base_url("http://localhost:8765/v1").expect("url");
        assert_eq!(url.as_str(), "http://localhost:8765/");
    }

    #[test]
    fn ask_config_builds_chat_request_with_history_and_metadata() {
        let request = AskConfig::new("qwen")
            .system("answer briefly")
            .temperature(0.2)
            .max_tokens(64)
            .metadata(serde_json::json!({"session_id": "ask-1"}))
            .to_request(
                Question::new("continue").with_history(vec![ChatMessage::assistant("previous")]),
            )
            .expect("request");

        assert_eq!(request.model, "qwen");
        assert_eq!(request.messages[0].role, "system");
        assert_eq!(request.messages[1].role, "assistant");
        assert_eq!(request.messages[2].content.as_deref(), Some("continue"));
        assert_eq!(request.temperature, Some(0.2));
        assert_eq!(request.max_tokens, Some(64));
        assert_eq!(
            request.metadata,
            Some(serde_json::json!({"session_id": "ask-1"}))
        );
    }

    #[test]
    fn non_local_provider_kinds_are_contract_only_until_server_side_egress_exists() {
        let err = AskConfig {
            provider: ProviderKind::OpenAiCompatible,
            ..AskConfig::new("gpt-4o-mini")
        }
        .to_request(Question::new("hello"))
        .expect_err("external provider request is reserved metadata");

        assert!(err
            .to_string()
            .contains("contract-only metadata and cannot route traffic"));
    }

    #[test]
    fn provider_contract_preserves_local_default_and_marks_external_adapters() {
        let local = ProviderContract::local_llmctl();

        assert_eq!(local.kind, ProviderKind::LocalLlmctl);
        assert_eq!(local.routing, ProviderRouting::LocalOnly);
        assert_eq!(local.status, ProviderStatus::Implemented);
        assert!(local.local_first);
        assert!(!local.routes_external_provider_traffic);
        local.validate_routable().expect("local llmctl is routable");

        for provider in [
            ProviderKind::OpenAiCompatible,
            ProviderKind::VertexAi,
            ProviderKind::OpenRouter,
        ] {
            let contract = ProviderContract::for_kind(provider.clone());
            assert_eq!(contract.kind, provider);
            assert_eq!(contract.routing, ProviderRouting::ExternalReserved);
            assert_eq!(contract.status, ProviderStatus::ContractOnly);
            assert!(!contract.routes_external_provider_traffic);
            assert!(contract.base_url_env.is_empty());
            assert!(contract.api_key_env.is_empty());
            contract
                .validate_routable()
                .expect_err("external providers are reserved metadata in the native-only client");
        }
    }

    #[test]
    fn provider_client_rejects_external_provider_bypass() {
        let err = client_from_provider_env_values(ProviderKind::OpenRouter, |name| match name {
            "OPENROUTER_BASE_URL" => Some("https://openrouter.example/api/v1".to_string()),
            "OPENROUTER_API_KEY" => Some("provider-secret".to_string()),
            _ => None,
        })
        .expect_err("external provider bypass is rejected");

        assert!(err
            .to_string()
            .contains("contract-only metadata and cannot route traffic"));
    }

    #[test]
    fn scheduler_contract_serializes_fifo_runtime_with_metadata_only_batching_and_kv_cache() {
        let contract = SchedulerContract::fifo_runtime();
        let serialized = serde_json::to_value(&contract).expect("scheduler contract serializes");

        assert_eq!(serialized["contract_only"], false);
        assert_eq!(serialized["queue"]["implemented"], true);
        assert_eq!(serialized["batching"]["continuous_batching"], false);
        assert_eq!(serialized["batching"]["implemented"], false);
        assert_eq!(
            serialized["kv_cache"]["cache_budget_metadata_key"],
            "llmctl.scheduler.kv_cache_budget_bytes"
        );
        assert_eq!(serialized["kv_cache"]["implemented"], false);
        assert_eq!(serialized["cancellation"]["implemented"], false);
        contract
            .validate_runtime_contract()
            .expect("FIFO scheduler runtime contract validates");
    }
}
