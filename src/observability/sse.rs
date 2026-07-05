use super::*;

/// Maximum bytes buffered per SSE frame before the parser gives up.
const MAX_SSE_CONTENT_BUFFER_BYTES: usize = 512 * 1024;

/// Tail length kept for cross-chunk `<think>` / `</think>` tag detection.
const TAG_TAIL_LEN: usize = 8;

/// Streaming phase tracked by [`SseContentParser`].
#[derive(Debug, Default, PartialEq, Eq)]
enum ContentPhase {
    #[default]
    Output,
    Thinking,
}

/// Parses a Server-Sent Events stream and counts output versus thinking-phase
/// content deltas.
///
/// Thinking content is any delta that arrives between a `<think>` and a
/// `</think>` tag, even when the tags are split across consecutive SSE frames.
/// Delta frames that carry only the tag itself are not counted in either
/// category.
#[derive(Debug, Default)]
pub struct SseContentParser {
    buffer: String,
    phase: ContentPhase,
    thinking_deltas: u64,
    output_deltas: u64,
    /// Last [`TAG_TAIL_LEN`] chars of previously-seen content, used to detect
    /// tags that are split across frame boundaries.
    tag_tail: String,
}

impl SseContentParser {
    /// Feeds raw SSE bytes into the parser, updating internal delta counters.
    pub fn push(&mut self, bytes: &[u8]) {
        let text = String::from_utf8_lossy(bytes);
        self.buffer.push_str(&text);
        while let Some(frame_end) = self.buffer.find("\n\n") {
            if frame_end > MAX_SSE_CONTENT_BUFFER_BYTES {
                self.buffer.drain(..frame_end + 2);
                continue;
            }
            let frame = self.buffer[..frame_end].to_string();
            self.buffer.drain(..frame_end + 2);
            self.process_frame(&frame);
        }
    }

    /// Returns the number of thinking-phase content deltas seen so far.
    pub fn thinking_deltas(&self) -> u64 {
        self.thinking_deltas
    }

    /// Returns the number of output-phase content deltas seen so far.
    pub fn output_deltas(&self) -> u64 {
        self.output_deltas
    }

    fn process_frame(&mut self, frame: &str) {
        for line in frame.lines() {
            let Some(data) = line.trim_start().strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let Ok(value) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            self.process_delta_value(&value);
        }
    }

    fn process_delta_value(&mut self, value: &Value) {
        let Some(choices) = value.get("choices").and_then(Value::as_array) else {
            return;
        };
        for choice in choices {
            let Some(content) = choice
                .get("delta")
                .and_then(|d| d.get("content"))
                .and_then(Value::as_str)
            else {
                continue;
            };
            self.classify_delta(content);
        }
    }

    fn classify_delta(&mut self, delta: &str) {
        let combined = format!("{}{}", self.tag_tail, delta);
        // Only look for a transition into thinking when we're currently in output phase.
        let entered_thinking = self.phase == ContentPhase::Output && combined.contains("<think>");
        // Only look for a transition out of thinking when we're currently in thinking phase.
        let left_thinking = self.phase == ContentPhase::Thinking && combined.contains("</think>");

        // Update tail with just the last TAG_TAIL_LEN chars of this delta
        // (not combined) so it remains the suffix of actual content seen.
        self.tag_tail = tail_str(&combined, TAG_TAIL_LEN);

        // A delta that carries a phase-transition tag is not counted in either bucket.
        if entered_thinking {
            self.phase = ContentPhase::Thinking;
            return;
        }
        if left_thinking {
            self.phase = ContentPhase::Output;
            return;
        }

        match self.phase {
            ContentPhase::Thinking => self.thinking_deltas += 1,
            ContentPhase::Output => self.output_deltas += 1,
        }
    }
}

/// Returns the last `n` chars of `s` as a new `String`.
fn tail_str(s: &str, n: usize) -> String {
    s.chars()
        .rev()
        .take(n)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}
