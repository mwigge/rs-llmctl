use super::*;

#[test]
fn peak_resident_mb_returns_a_positive_value_on_unix() {
    // Sanity check the cross-platform sampler. The actual value depends on
    // the test process; we just assert it's positive and plausible (< 32 GB
    // for a unit-test process is a very loose upper bound).
    #[cfg(unix)]
    {
        let mb = process_peak_resident_mb().expect("getrusage should succeed on Unix");
        assert!(mb > 0.0, "expected positive RSS, got {mb}");
        assert!(
            mb < 32_768.0,
            "implausibly large RSS for a test process: {mb} MB"
        );
    }
    #[cfg(not(unix))]
    {
        assert_eq!(process_peak_resident_mb(), None);
    }
}

#[test]
fn native_metric_instruments_share_a_single_global_meter() {
    // The OnceLock initialisation pattern means calling each accessor
    // twice should hand back the same instance — guards against accidental
    // duplicate metric registration that would silently fan-out exports.
    let load_a = native_model_load_duration_ms() as *const _;
    let load_b = native_model_load_duration_ms() as *const _;
    assert_eq!(
        load_a, load_b,
        "load duration histogram must be a OnceLock-cached singleton"
    );

    let tps_a = native_model_tokens_per_second() as *const _;
    let tps_b = native_model_tokens_per_second() as *const _;
    assert_eq!(
        tps_a, tps_b,
        "tokens/s histogram must be a OnceLock-cached singleton"
    );

    let rss_a = native_model_peak_resident_mb() as *const _;
    let rss_b = native_model_peak_resident_mb() as *const _;
    assert_eq!(
        rss_a, rss_b,
        "peak resident gauge must be a OnceLock-cached singleton"
    );
}
