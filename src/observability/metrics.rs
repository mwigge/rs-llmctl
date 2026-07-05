use super::*;

pub fn genai_input_tokens_counter() -> &'static Counter<u64> {
    static COUNTER: OnceLock<Counter<u64>> = OnceLock::new();
    COUNTER.get_or_init(|| {
        global::meter(crate::SERVICE_NAME)
            .u64_counter("gen_ai.usage.input_tokens")
            .with_description("Input tokens per GenAI semantic conventions")
            .build()
    })
}

pub fn genai_output_tokens_counter() -> &'static Counter<u64> {
    static COUNTER: OnceLock<Counter<u64>> = OnceLock::new();
    COUNTER.get_or_init(|| {
        global::meter(crate::SERVICE_NAME)
            .u64_counter("gen_ai.usage.output_tokens")
            .with_description("Output tokens per GenAI semantic conventions")
            .build()
    })
}

/// Histogram of wall-clock time spent constructing a native model from a GGUF
/// or safetensors artifact. Attributes: `model.family`, `model.quant`,
/// `gpu.backend`. Recorded once per model load.
pub fn native_model_load_duration_ms() -> &'static Histogram<f64> {
    static H: OnceLock<Histogram<f64>> = OnceLock::new();
    H.get_or_init(|| {
        global::meter(crate::SERVICE_NAME)
            .f64_histogram("native.model.load.duration_ms")
            .with_description("Wall-clock time to load a native model artifact, in milliseconds")
            .with_unit("ms")
            .build()
    })
}

/// Histogram of throughput in tokens-per-second for native inference, split
/// by phase. Attributes: `model.family`, `phase` (`"prefill"` or `"generation"`).
pub fn native_model_tokens_per_second() -> &'static Histogram<f64> {
    static H: OnceLock<Histogram<f64>> = OnceLock::new();
    H.get_or_init(|| {
        global::meter(crate::SERVICE_NAME)
            .f64_histogram("native.model.tokens_per_second")
            .with_description("Native model throughput in tokens per second, split by phase")
            .with_unit("token/s")
            .build()
    })
}

/// Gauge of peak resident memory observed after a native model load completes.
/// Attribute: `model.family`. Sampled via `getrusage(RUSAGE_SELF).ru_maxrss`
/// (macOS reports bytes; Linux reports KiB).
pub fn native_model_peak_resident_mb() -> &'static Gauge<f64> {
    static G: OnceLock<Gauge<f64>> = OnceLock::new();
    G.get_or_init(|| {
        global::meter(crate::SERVICE_NAME)
            .f64_gauge("native.model.peak_resident_mb")
            .with_description("Peak resident memory after native model load, in MB")
            .with_unit("MB")
            .build()
    })
}

/// Sample the current process's peak resident memory in MB.
/// Returns `None` if the syscall is unavailable or fails.
///
/// macOS `ru_maxrss` is in bytes; Linux `ru_maxrss` is in kilobytes (KiB).
/// Other Unix platforms report bytes per BSD convention.
#[must_use]
pub fn process_peak_resident_mb() -> Option<f64> {
    #[cfg(unix)]
    {
        // SAFETY: getrusage is async-signal-safe; the rusage struct is POD;
        // we zero-init it before the call.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        if rc != 0 {
            return None;
        }
        let maxrss = usage.ru_maxrss as f64;
        // Linux reports `ru_maxrss` in KiB; macOS and most BSDs report it in
        // bytes. Both cases normalise to megabytes here.
        #[cfg(target_os = "linux")]
        let mb = maxrss / 1024.0;
        #[cfg(not(target_os = "linux"))]
        let mb = maxrss / (1024.0 * 1024.0);
        Some(mb)
    }
    #[cfg(not(unix))]
    {
        None
    }
}

/// Returns the histogram that tracks thinking-phase token counts per model.
pub fn gen_ai_thinking_tokens_histogram() -> &'static Histogram<u64> {
    static HIST: OnceLock<Histogram<u64>> = OnceLock::new();
    HIST.get_or_init(|| {
        global::meter(crate::SERVICE_NAME)
            .u64_histogram("gen_ai.thinking_tokens")
            .with_description("Number of thinking-phase content deltas per inference request")
            .build()
    })
}

macro_rules! static_f64_gauge {
    ($name:expr, $desc:expr) => {{
        static GAUGE: OnceLock<Gauge<f64>> = OnceLock::new();
        GAUGE.get_or_init(|| {
            global::meter(crate::SERVICE_NAME)
                .f64_gauge($name)
                .with_description($desc)
                .build()
        })
    }};
}

/// Returns the gauge that tracks the fraction of output that was thinking content.
pub fn gen_ai_thinking_ratio_gauge() -> &'static Gauge<f64> {
    static_f64_gauge!(
        "gen_ai.thinking_ratio",
        "Fraction of content deltas that were thinking-phase (0.0–1.0) per request"
    )
}

/// Returns the gauge that tracks the KV-cache occupancy ratio for one model worker.
///
/// Records values in the range `[0.0, 1.0]` where `1.0` means the cache is
/// completely full.  Tagged with `gen_ai.request.model`.
pub fn gen_ai_kv_cache_usage_ratio_gauge() -> &'static Gauge<f64> {
    static_f64_gauge!(
        "gen_ai.kv_cache.usage_ratio",
        "KV-cache occupancy ratio per model worker (0.0 = empty, 1.0 = full)"
    )
}

fn add_thinking_phase_event(name: &'static str, attrs: Vec<KeyValue>) {
    use opentelemetry::trace::get_active_span;
    get_active_span(|span| span.add_event(name, attrs));
}

/// Adds a `gen_ai.thinking.started` event to the current span.
pub fn emit_gen_ai_thinking_phase_started(model: &str, position: u64) {
    add_thinking_phase_event(
        "gen_ai.thinking.started",
        vec![
            KeyValue::new("gen_ai.request.model", model.to_string()),
            KeyValue::new(
                "gen_ai.token.position",
                i64::try_from(position).unwrap_or(i64::MAX),
            ),
        ],
    );
}

/// Adds a `gen_ai.thinking.ended` event to the current span.
pub fn emit_gen_ai_thinking_phase_ended(model: &str, thinking_tokens: u64, duration_seconds: f64) {
    add_thinking_phase_event(
        "gen_ai.thinking.ended",
        vec![
            KeyValue::new("gen_ai.request.model", model.to_string()),
            KeyValue::new(
                "gen_ai.thinking.tokens",
                i64::try_from(thinking_tokens).unwrap_or(i64::MAX),
            ),
            KeyValue::new("gen_ai.thinking.duration_s", duration_seconds),
        ],
    );
}

/// Emits `gen_ai.thinking_tokens` and `gen_ai.thinking_ratio` metrics for one
/// completed inference request, tagged with the serving model name.
pub fn emit_gen_ai_thinking_metrics(model: &str, thinking_deltas: u64, output_deltas: u64) {
    let attrs = [KeyValue::new("gen_ai.request.model", model.to_string())];
    gen_ai_thinking_tokens_histogram().record(thinking_deltas, &attrs);
    let total = thinking_deltas + output_deltas;
    let ratio = if total > 0 {
        thinking_deltas as f64 / total as f64
    } else {
        0.0
    };
    gen_ai_thinking_ratio_gauge().record(ratio, &attrs);
}
