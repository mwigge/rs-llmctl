#[cfg(test)]
mod kv_cache_guard_tests {
    use crate::native::{admit_fresh_kv_session, KvCacheReset};
    use std::sync::atomic::AtomicBool;

    // Regression for KV-cache cross-request contamination: a model family whose
    // cache cannot be cleared (Qwen3 MoE / Mistral GGUF) must not serve a second
    // request off the retained state of the first. `generate()` runs exactly
    // this sequence — `reset_kv_cache()` then `admit_fresh_kv_session()` against
    // the decoder's `served` flag — so driving the guard twice on one flag
    // mirrors two `generate()` calls on the same engine.
    #[test]
    fn retained_cache_refuses_second_request_on_shared_session() {
        let served = AtomicBool::new(false);
        // First request runs on a fresh, empty cache — allowed.
        assert!(admit_fresh_kv_session(KvCacheReset::Retained, &served).is_ok());
        // Second request would reuse the prior request's retained KV state —
        // must be refused (fail-closed) instead of serving contaminated output.
        let err = admit_fresh_kv_session(KvCacheReset::Retained, &served)
            .expect_err("second retained-cache generation must be refused");
        assert!(
            err.to_string().contains("cross-request contamination"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn clearable_cache_allows_repeated_requests() {
        let served = AtomicBool::new(false);
        // A family whose cache is genuinely cleared may serve many requests.
        assert!(admit_fresh_kv_session(KvCacheReset::Cleared, &served).is_ok());
        assert!(admit_fresh_kv_session(KvCacheReset::Cleared, &served).is_ok());
    }
}

#[cfg(test)]
mod sampling_and_accounting_tests {
    use crate::native::*;
    use std::collections::BTreeMap;

    fn request_with(
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        seed: Option<u64>,
    ) -> NativeChatRequest {
        NativeChatRequest {
            model: "test".to_string(),
            messages: Vec::new(),
            temperature,
            max_tokens: Some(16),
            top_p,
            top_k,
            seed,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: None,
            tool_choice: None,
            metadata: BTreeMap::new(),
        }
    }

    // Bug 12: `native_exact_usage` reports the exact loop-observed token counts
    // (and the `NativeExact` label), not a re-tokenization of the decoded text.
    #[test]
    fn native_exact_usage_uses_loop_counts_not_retokenization() {
        // What the decode loop actually produced.
        let generated_token_count = 3u64;
        // A re-tokenization of the decoded string would give a *different*
        // number — that mismatch is exactly the Bug 12 defect.
        let retokenized = EstimatedNativeTokenCounter
            .count_text("internationalization")
            .expect("estimate");
        assert_ne!(
            retokenized, generated_token_count,
            "test precondition: re-tokenization must differ from the loop count"
        );

        let usage = native_exact_usage(5, generated_token_count);
        assert_eq!(usage.input_tokens, 5);
        assert_eq!(usage.output_tokens, generated_token_count);
        assert_ne!(usage.output_tokens, retokenized);
        assert_eq!(usage.accounting_mode, TokenAccountingMode::NativeExact);
    }

    // Bug 11: temperature 0/None stays deterministic greedy (ArgMax), preserving
    // the pre-sampling behavior.
    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    #[test]
    fn zero_or_absent_temperature_maps_to_greedy_argmax() {
        use candle_transformers::generation::Sampling;
        assert_eq!(
            sampling_from_request(&request_with(None, None, None, None)),
            Sampling::ArgMax
        );
        assert_eq!(
            sampling_from_request(&request_with(Some(0.0), Some(0.9), Some(40), None)),
            Sampling::ArgMax,
            "temperature 0 must stay greedy even if top-p/top-k are set"
        );
    }

    // Bug 11: nonzero temperature builds a sampling strategy configured from the
    // request's top-p/top-k parameters.
    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    #[test]
    fn nonzero_temperature_selects_sampling_strategy_from_request() {
        use candle_transformers::generation::Sampling;
        match sampling_from_request(&request_with(Some(0.7), None, None, None)) {
            Sampling::All { temperature } => {
                assert!((temperature - f64::from(0.7f32)).abs() < 1e-6)
            }
            other => panic!("expected All sampling, got {other:?}"),
        }
        assert!(matches!(
            sampling_from_request(&request_with(Some(0.7), Some(0.9), None, None)),
            Sampling::TopP { .. }
        ));
        assert!(matches!(
            sampling_from_request(&request_with(Some(0.7), None, Some(40), None)),
            Sampling::TopK { .. }
        ));
        assert!(matches!(
            sampling_from_request(&request_with(Some(0.7), Some(0.9), Some(40), None)),
            Sampling::TopKThenTopP { .. }
        ));
    }

    // Bug 11: a fixed seed makes nonzero-temperature sampling reproducible.
    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    #[test]
    fn fixed_seed_makes_sampling_reproducible() {
        use candle_transformers::generation::LogitsProcessor;
        let request = request_with(Some(0.8), Some(0.95), None, Some(1234));
        let seed = request.seed.unwrap_or(DEFAULT_SAMPLING_SEED);
        // A skewed logit vector so multinomial sampling has a real distribution.
        let logits = candle_core::Tensor::new(
            &[0.1f32, 2.5, 0.3, 1.7, 0.9, 3.1, 0.2, 1.1],
            &candle_core::Device::Cpu,
        )
        .expect("logits tensor");

        let mut first = LogitsProcessor::from_sampling(seed, sampling_from_request(&request));
        let mut second = LogitsProcessor::from_sampling(seed, sampling_from_request(&request));
        let a = first.sample(&logits).expect("sample a");
        let b = second.sample(&logits).expect("sample b");
        assert_eq!(a, b, "same seed + params must reproduce the same token");
    }
}
