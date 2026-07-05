use crate::*;
use futures_util::StreamExt;
use reqwest::{header, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::env;

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

    pub(crate) fn to_request(&self, question: Question) -> Result<ChatCompletionRequest> {
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
