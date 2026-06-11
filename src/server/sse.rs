//! Parsing of Server-Sent Events (SSE) chat-completion streams to recover
//! token usage counts for accounting, plus the equivalent helper for
//! non-streaming JSON responses.

use serde_json::Value;

pub(super) const MAX_SSE_USAGE_BUFFER_BYTES: usize = 1024 * 1024;

pub(super) fn usage_tokens(bytes: &[u8]) -> (u64, u64) {
    let Ok(value) = serde_json::from_slice::<Value>(bytes) else {
        return (0, 0);
    };
    usage_tokens_from_value(&value)
}

fn usage_tokens_from_value(value: &Value) -> (u64, u64) {
    let Some(usage) = value.get("usage") else {
        return (0, 0);
    };
    let input = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    (input, output)
}

#[derive(Debug, Default)]
pub(super) struct SseUsageParser {
    buffer: String,
}

impl SseUsageParser {
    pub(super) fn push(&mut self, bytes: &[u8]) -> std::result::Result<(u64, u64), &'static str> {
        let text = String::from_utf8_lossy(bytes);
        self.buffer.push_str(&text);
        if self.buffer.len() > MAX_SSE_USAGE_BUFFER_BYTES && !self.buffer.contains("\n\n") {
            return Err("SSE usage parser buffer exceeded maximum frame size");
        }
        let mut input_tokens = 0u64;
        let mut output_tokens = 0u64;

        while let Some(frame_end) = self.buffer.find("\n\n") {
            if frame_end > MAX_SSE_USAGE_BUFFER_BYTES {
                return Err("SSE usage parser buffer exceeded maximum frame size");
            }
            let frame = self.buffer[..frame_end].to_string();
            self.buffer.drain(..frame_end + 2);
            let (input, output) = sse_frame_usage_tokens(&frame);
            input_tokens = input_tokens.saturating_add(input);
            output_tokens = output_tokens.saturating_add(output);
        }

        Ok((input_tokens, output_tokens))
    }
}

#[cfg(test)]
fn sse_usage_tokens(bytes: &[u8]) -> (u64, u64) {
    SseUsageParser::default().push(bytes).expect("valid SSE")
}

fn sse_frame_usage_tokens(frame: &str) -> (u64, u64) {
    frame
        .lines()
        .filter_map(|line| line.trim_start().strip_prefix("data:"))
        .map(str::trim)
        .filter(|data| !data.is_empty() && *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .map(|value| usage_tokens_from_value(&value))
        .fold(
            (0u64, 0u64),
            |(total_input, total_output), (input, output)| {
                (
                    total_input.saturating_add(input),
                    total_output.saturating_add(output),
                )
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::stream_status;

    #[test]
    fn extracts_openai_usage_tokens() {
        let body = br#"{"usage":{"prompt_tokens":11,"completion_tokens":13}}"#;
        assert_eq!(usage_tokens(body), (11, 13));
    }

    #[test]
    fn extracts_streaming_sse_usage_tokens() {
        let chunk = br#"event: completion
data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":9}}

data: [DONE]
"#;

        assert_eq!(sse_usage_tokens(chunk), (7, 9));
        assert_eq!(stream_status(0, 0), "stream_unmetered");
        assert_eq!(stream_status(7, 9), "ok");
    }

    #[test]
    fn extracts_split_streaming_sse_usage_tokens() {
        let mut parser = SseUsageParser::default();

        assert_eq!(
            parser.push(br#"data: {"choices":[],"usage":{"prompt_tokens":7"#),
            Ok((0, 0))
        );
        assert_eq!(
            parser.push(
                br#","completion_tokens":9}}

"#
            ),
            Ok((7, 9))
        );
    }

    #[test]
    fn sse_usage_parser_rejects_unbounded_partial_frames() {
        let mut parser = SseUsageParser::default();
        let oversized = vec![b'a'; MAX_SSE_USAGE_BUFFER_BYTES + 1];
        assert!(parser.push(&oversized).is_err());
    }

    #[test]
    fn sse_usage_parser_rejects_oversized_complete_frames() {
        let mut parser = SseUsageParser::default();
        let mut oversized = vec![b'a'; MAX_SSE_USAGE_BUFFER_BYTES + 1];
        oversized.extend_from_slice(b"\n\n");
        assert!(parser.push(&oversized).is_err());
    }
}
