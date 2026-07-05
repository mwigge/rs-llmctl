use super::*;
use axum::http::HeaderValue;

#[test]
fn dev_cors_allows_only_loopback_origins() {
    // The dev-default CORS policy must accept loopback origins (so the bundled
    // playground works) but reject arbitrary external websites — otherwise any
    // site in the operator's browser could drive this local, potentially
    // authenticated endpoint. Before the fix this branch used `Any`.
    for allowed in [
        "http://localhost",
        "http://localhost:8080",
        "https://localhost:3000",
        "http://127.0.0.1:8080",
        "http://[::1]:8080",
        "http://LocalHost:1234",
    ] {
        assert!(
            is_loopback_origin(&HeaderValue::from_static(allowed)),
            "{allowed} should be treated as loopback"
        );
    }

    for denied in [
        "https://evil.example.com",
        "http://attacker.test:8080",
        "http://127.0.0.1.evil.com",
        "http://notlocalhost",
        "null",
    ] {
        assert!(
            !is_loopback_origin(&HeaderValue::from_static(denied)),
            "{denied} must not be treated as loopback"
        );
    }
}

#[test]
fn circuit_breaker_opens_after_threshold_and_half_opens_after_reset() {
    let breakers = CircuitBreakers::default();
    let upstream = "http://127.0.0.1:18765";
    assert!(breakers.allow_request(upstream, Duration::from_secs(30)));
    breakers.record_failure(upstream, 2);
    assert!(breakers.allow_request(upstream, Duration::from_secs(30)));
    breakers.record_failure(upstream, 2);
    assert!(!breakers.allow_request(upstream, Duration::from_secs(30)));
    assert!(breakers.allow_request(upstream, Duration::from_secs(0)));
    assert!(!breakers.allow_request(upstream, Duration::from_secs(0)));
    breakers.record_success(upstream);
    assert!(breakers.allow_request(upstream, Duration::from_secs(30)));
}

#[test]
fn serving_limits_default_to_internal_admission_limit_without_quota_config() {
    let cfg = Config::default();

    let limits = ServingLimits::from_config(&cfg);

    assert_eq!(limits.max_in_flight, DEFAULT_MAX_IN_FLIGHT);
    assert_eq!(limits.upstream_timeout(), DEFAULT_UPSTREAM_TIMEOUT);
}

#[test]
fn serving_limits_use_configured_quota_concurrency_when_available() {
    let cfg = Config {
        quotas: vec![
            crate::config::QuotaConfig {
                subject: "alice".to_string(),
                team: "".to_string(),
                requests_per_minute: 10,
                tokens_per_day: 100,
                max_concurrency: 2,
                allowed_models: vec!["llama".to_string()],
            },
            crate::config::QuotaConfig {
                subject: "bob".to_string(),
                team: "".to_string(),
                requests_per_minute: 10,
                tokens_per_day: 100,
                max_concurrency: 3,
                allowed_models: vec!["llama".to_string()],
            },
        ],
        ..Default::default()
    };

    let limits = ServingLimits::from_config(&cfg);

    assert_eq!(limits.max_in_flight, 5);
}

#[test]
fn quota_admission_scopes_include_subject_and_team_limits() {
    let cfg = Config {
        quotas: vec![
            crate::config::QuotaConfig {
                subject: "alice".to_string(),
                team: "platform".to_string(),
                requests_per_minute: 10,
                tokens_per_day: 100,
                max_concurrency: 2,
                allowed_models: vec!["llama".to_string()],
            },
            crate::config::QuotaConfig {
                subject: "team-default".to_string(),
                team: "platform".to_string(),
                requests_per_minute: 10,
                tokens_per_day: 100,
                max_concurrency: 1,
                allowed_models: vec!["llama".to_string()],
            },
        ],
        ..Default::default()
    };
    let principal = Principal {
        subject: "alice".to_string(),
        team: "platform".to_string(),
        scopes: vec!["chat".to_string()],
        key_id: Some("alice-key".to_string()),
        key_owner: None,
        key_purpose: None,
        key_status: Some("active".to_string()),
    };

    assert_eq!(
        quota_admission_scopes(&cfg, &principal),
        vec![
            ("subject:alice".to_string(), 2),
            ("team:platform".to_string(), 1)
        ]
    );
}

#[test]
fn admission_controller_rejects_when_in_flight_limit_is_full() {
    let controller = AdmissionController::new(1);
    let first = controller.try_acquire_for(None).expect("first permit");

    assert_eq!(
        controller.try_acquire_for(None).unwrap_err(),
        AdmissionError::Busy
    );

    drop(first);
    assert!(controller.try_acquire_for(None).is_ok());
}

#[test]
fn admission_controller_applies_all_scoped_limits() {
    let controller = AdmissionController::new(8);
    let first = controller
        .try_acquire_for_all(vec![
            ("subject:alice".to_string(), 2),
            ("team:platform".to_string(), 1),
        ])
        .expect("scoped permit");

    assert_eq!(
        controller
            .try_acquire_for_all(vec![
                ("subject:alice".to_string(), 2),
                ("team:platform".to_string(), 1),
            ])
            .unwrap_err(),
        AdmissionError::Busy
    );
    assert!(controller
        .try_acquire_for_all(vec![
            ("subject:alice".to_string(), 2),
            ("team:research".to_string(), 1),
        ])
        .is_ok());

    drop(first);
    assert!(controller
        .try_acquire_for_all(vec![
            ("subject:alice".to_string(), 2),
            ("team:platform".to_string(), 1),
        ])
        .is_ok());
}

#[test]
fn admission_controller_applies_scoped_limits() {
    let controller = AdmissionController::new(8);
    let first = controller
        .try_acquire_for(Some(("subject:alice".to_string(), 1)))
        .expect("scoped permit");

    assert_eq!(
        controller
            .try_acquire_for(Some(("subject:alice".to_string(), 1)))
            .unwrap_err(),
        AdmissionError::Busy
    );
    assert!(controller
        .try_acquire_for(Some(("subject:bob".to_string(), 1)))
        .is_ok());

    drop(first);
    assert!(controller
        .try_acquire_for(Some(("subject:alice".to_string(), 1)))
        .is_ok());
}
