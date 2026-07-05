//! Token accounting: chat-prompt canonicalization, prompt templating, and token counters.
use super::*;

pub trait NativeTokenCounter: Send + Sync {
    fn accounting_mode(&self) -> TokenAccountingMode {
        TokenAccountingMode::NativeExact
    }

    fn count_chat_input(&self, messages: &[NativeChatMessage]) -> Result<u64>;
    fn count_text(&self, text: &str) -> Result<u64>;
}

pub trait NativeTokenAccountingAdapter: NativeTokenCounter {}

impl<T> NativeTokenAccountingAdapter for T where T: NativeTokenCounter + ?Sized {}

pub fn canonical_native_chat_input(messages: &[NativeChatMessage]) -> String {
    let mut input = String::new();
    for message in messages {
        input.push_str("<|");
        input.push_str(&message.role);
        input.push_str("|>\n");
        input.push_str(&message_content_text(message));
        if message.tool_calls.is_some() {
            input.push_str("\n<|assistant_tool_calls|>");
        }
        if let Some(tool_call_id) = &message.tool_call_id {
            input.push_str("\n<|tool_call_id|>");
            input.push_str(tool_call_id);
        }
        input.push('\n');
    }
    input
}

/// Renders `messages` using Gemma 4's chat template turn format —
/// `<|turn>{role}\n{content}<turn|>\n` — as embedded in this model's GGUF
/// `tokenizer.chat_template`. The older Gemma 2/3
/// `<start_of_turn>{role}\n{content}<end_of_turn>\n` format is NOT used here:
/// those tokens are absent from this model's vocabulary and would otherwise
/// be split into garbage sub-tokens, corrupting the prompt.
///
/// `assistant` maps to the `model` role; `system` is passed through
/// unchanged; all other roles (`user`, `tool`, etc.) map to `user`. The
/// rendered prompt always ends with the generation cue
/// `<|turn>model\n<|channel>thought\n<channel|>` (no closing `<turn|>`).
#[must_use]
pub fn gemma_chat_input(messages: &[NativeChatMessage]) -> String {
    let mut input = String::new();

    for message in messages {
        let role = match message.role.as_str() {
            "assistant" => "model",
            "system" => "system",
            _ => "user",
        };

        input.push_str("<|turn>");
        input.push_str(role);
        input.push('\n');
        input.push_str(message_content_text(message).trim());
        input.push_str("<turn|>\n");
    }

    input.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    input
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
pub fn format_native_chat_prompt(
    family: CandleModelFamily,
    messages: &[NativeChatMessage],
) -> String {
    match family {
        // Mistral family (Devstral, Mistral Small, etc.) — classic
        // `<s>[INST] ... [/INST]` template. Devstral and other agentic-tuned
        // Mistral fine-tunes also accept this format; their own elaborate
        // chat_templates (in the GGUF metadata) are a strict superset.
        CandleModelFamily::Mistral => {
            let mut out = String::from("<s>");
            for msg in messages {
                let content = message_content_text(msg);
                let role = msg.role.as_str();
                match role {
                    "user" => {
                        out.push_str("[INST] ");
                        out.push_str(&content);
                        out.push_str(" [/INST]");
                    }
                    "assistant" => {
                        out.push(' ');
                        out.push_str(&content);
                        out.push_str("</s>");
                    }
                    "system" => {
                        // System prompts in Mistral land are normally folded
                        // into the first user turn — but a leading
                        // [INST] block alone is still parseable.
                        out.push_str("[INST] ");
                        out.push_str(&content);
                        out.push_str(" [/INST]");
                    }
                    _ => {
                        out.push_str("[INST] ");
                        out.push_str(&content);
                        out.push_str(" [/INST]");
                    }
                }
            }
            out
        }
        // Qwen3 (dense + MoE) — native ChatML format <|im_start|>role\n...<|im_end|>
        CandleModelFamily::Qwen3 | CandleModelFamily::Qwen3Moe => {
            let mut out = String::new();
            for msg in messages {
                let role = match msg.role.as_str() {
                    "assistant" => "assistant",
                    "system" => "system",
                    _ => "user",
                };
                let content = message_content_text(msg);
                out.push_str("<|im_start|>");
                out.push_str(role);
                out.push('\n');
                out.push_str(&content);
                out.push_str("<|im_end|>\n");
            }
            out.push_str("<|im_start|>assistant\n");
            out
        }
        CandleModelFamily::Gemma4 => gemma_chat_input(messages),
        _ => canonical_native_chat_input(messages),
    }
}

pub fn message_content_text(message: &NativeChatMessage) -> String {
    match &message.content {
        Some(Value::String(content)) => content.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(Value::as_str)
                    .or_else(|| part.as_str())
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EstimatedNativeTokenCounter;

impl EstimatedNativeTokenCounter {
    const CHARS_PER_TOKEN: u64 = 4;
    const MESSAGE_OVERHEAD_TOKENS: u64 = 4;

    pub(crate) fn estimate_text_tokens(text: &str) -> u64 {
        let normalized_chars = text.chars().filter(|ch| !ch.is_control()).count() as u64;
        if normalized_chars == 0 {
            return 0;
        }
        normalized_chars
            .saturating_add(Self::CHARS_PER_TOKEN - 1)
            .saturating_div(Self::CHARS_PER_TOKEN)
            .max(1)
    }
}

impl NativeTokenCounter for EstimatedNativeTokenCounter {
    fn accounting_mode(&self) -> TokenAccountingMode {
        TokenAccountingMode::Estimated
    }

    fn count_chat_input(&self, messages: &[NativeChatMessage]) -> Result<u64> {
        Ok(messages
            .iter()
            .map(|message| {
                Self::MESSAGE_OVERHEAD_TOKENS
                    .saturating_add(Self::estimate_text_tokens(&message.role))
                    .saturating_add(Self::estimate_text_tokens(&message_content_text(message)))
                    .saturating_add(if message.tool_calls.is_some() { 1 } else { 0 })
                    .saturating_add(
                        message
                            .tool_call_id
                            .as_deref()
                            .map(Self::estimate_text_tokens)
                            .unwrap_or(0),
                    )
            })
            .sum())
    }

    fn count_text(&self, text: &str) -> Result<u64> {
        Ok(Self::estimate_text_tokens(text))
    }
}

#[cfg(feature = "native-tokenizers")]
#[derive(Debug, Clone)]
pub struct TokenizersNativeTokenCounter {
    tokenizer: tokenizers::Tokenizer,
}

#[cfg(feature = "native-tokenizers")]
impl TokenizersNativeTokenCounter {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|err| anyhow::anyhow!("failed to load tokenizer json: {err}"))?;
        Ok(Self::from_tokenizer(tokenizer))
    }

    pub const fn from_tokenizer(tokenizer: tokenizers::Tokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    fn count_serialized_input(&self, input: &str) -> Result<u64> {
        let encoding = self
            .tokenizer
            .encode(input, false)
            .map_err(|err| anyhow::anyhow!("failed to tokenize native input: {err}"))?;
        Ok(encoding.len() as u64)
    }
}

#[cfg(feature = "native-tokenizers")]
impl NativeTokenCounter for TokenizersNativeTokenCounter {
    fn accounting_mode(&self) -> TokenAccountingMode {
        TokenAccountingMode::NativeExact
    }

    fn count_chat_input(&self, messages: &[NativeChatMessage]) -> Result<u64> {
        self.count_serialized_input(&canonical_native_chat_input(messages))
    }

    fn count_text(&self, text: &str) -> Result<u64> {
        self.count_serialized_input(text)
    }
}

/// Builds a [`NativeTokenUsage`] from the exact token counts observed during
/// native decoding. `prompt_tokens` is the number of tokens actually fed to the
/// model (the templated prompt plus any prepended BOS) and `completion_tokens`
/// is the number of tokens the decode loop actually produced. Because both are
/// real model-token counts — not a re-tokenization of the decoded string — the
/// usage is labeled [`TokenAccountingMode::NativeExact`] (Bug 12).
pub fn native_exact_usage(prompt_tokens: u64, completion_tokens: u64) -> NativeTokenUsage {
    NativeTokenUsage::with_mode(
        prompt_tokens,
        completion_tokens,
        TokenAccountingMode::NativeExact,
    )
}

pub fn usage_from_native_tokens(
    counter: &dyn NativeTokenAccountingAdapter,
    request: &NativeChatRequest,
    response_text: &str,
) -> Result<NativeTokenUsage> {
    Ok(NativeTokenUsage::with_mode(
        counter.count_chat_input(&request.messages)?,
        counter.count_text(response_text)?,
        counter.accounting_mode(),
    ))
}
