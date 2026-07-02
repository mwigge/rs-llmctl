use super::*;

#[test]
fn auth_failure_limiter_blocks_after_configured_window_limit() {
    let limiter = AuthFailureLimiter::default();
    assert!(!limiter.is_limited("bad-token", 2));
    limiter.record_failure("bad-token", 2);
    assert!(!limiter.is_limited("bad-token", 2));
    limiter.record_failure("bad-token", 2);
    assert!(limiter.is_limited("bad-token", 2));
    limiter.record_success("bad-token");
    assert!(!limiter.is_limited("bad-token", 2));
}

#[test]
fn trusted_proxy_forwarded_chain_uses_rightmost_untrusted_client() {
    let mut cfg = Config::default();
    cfg.security.trusted_proxies = vec!["10.0.0.0/8".to_string()];
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-forwarded-for",
        HeaderValue::from_static("203.0.113.10, 198.51.100.20, 10.0.0.5"),
    );

    assert_eq!(
        forwarded_client_ip(&cfg, &headers).as_deref(),
        Some("198.51.100.20")
    );
}

#[test]
fn trusted_proxy_forwarded_chain_reads_duplicate_headers() {
    let mut cfg = Config::default();
    cfg.security.trusted_proxies = vec!["10.0.0.0/8".to_string()];
    let mut headers = HeaderMap::new();
    headers.append("x-forwarded-for", HeaderValue::from_static("203.0.113.10"));
    headers.append(
        "x-forwarded-for",
        HeaderValue::from_static("198.51.100.20, 10.0.0.5"),
    );

    assert_eq!(
        forwarded_client_ip(&cfg, &headers).as_deref(),
        Some("198.51.100.20")
    );
}

#[test]
fn trusted_proxy_matching_rejects_wildcard_runtime_entries() {
    let mut cfg = Config::default();
    cfg.security.trusted_proxies = vec!["*".to_string()];

    assert!(!is_trusted_proxy(
        &cfg,
        "203.0.113.1".parse::<IpAddr>().unwrap()
    ));
    cfg.security.trusted_proxies = vec!["0.0.0.0/0".to_string(), "::/0".to_string()];
    assert!(!is_trusted_proxy(
        &cfg,
        "203.0.113.1".parse::<IpAddr>().unwrap()
    ));
    assert!(!is_trusted_proxy(
        &cfg,
        "2001:db8::1".parse::<IpAddr>().unwrap()
    ));
}

#[test]
fn trusted_proxy_matching_normalizes_ipv4_mapped_ipv6() {
    let mut cfg = Config::default();
    cfg.security.trusted_proxies = vec!["10.0.0.0/8".to_string()];

    // A dual-stack listener may report an IPv4 peer as ::ffff:10.x.x.x.
    assert!(is_trusted_proxy(
        &cfg,
        "::ffff:10.1.2.3".parse::<IpAddr>().unwrap()
    ));
    assert!(is_trusted_proxy(
        &cfg,
        "10.1.2.3".parse::<IpAddr>().unwrap()
    ));
    assert!(!is_trusted_proxy(
        &cfg,
        "::ffff:192.0.2.1".parse::<IpAddr>().unwrap()
    ));

    // Exact-match form should normalize too.
    cfg.security.trusted_proxies = vec!["10.1.2.3".to_string()];
    assert!(is_trusted_proxy(
        &cfg,
        "::ffff:10.1.2.3".parse::<IpAddr>().unwrap()
    ));
}

#[test]
fn trusted_proxy_matching_rejects_out_of_range_prefix() {
    let mut cfg = Config::default();
    cfg.security.trusted_proxies = vec!["10.0.0.0/33".to_string()];
    assert!(!is_trusted_proxy(
        &cfg,
        "10.0.0.1".parse::<IpAddr>().unwrap()
    ));

    cfg.security.trusted_proxies = vec!["2001:db8::/129".to_string()];
    assert!(!is_trusted_proxy(
        &cfg,
        "2001:db8::1".parse::<IpAddr>().unwrap()
    ));
}

#[test]
fn bearer_auth_uses_configured_sha256_keys() {
    let token = "secret";
    let cfg = Config {
        server: ServerConfig::default(),
        security: SecurityConfig {
            production: false,
            require_auth: true,
            bind_external: false,
            api_keys: vec![ApiKeyConfig {
                id: "dev".to_string(),
                sha256: hex::encode(Sha256::digest(token.as_bytes())),
                subject: "alice".to_string(),
                team: "platform".to_string(),
                scopes: vec!["chat".to_string()],
                created_at: None,
                expires_at: None,
                rotated_at: None,
                owner: None,
                purpose: None,
                last_four: None,
                fingerprint: None,
                status: "active".to_string(),
            }],
            ..SecurityConfig::default()
        },
        ..Default::default()
    };
    let mut headers = HeaderMap::new();
    headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer secret"));

    let principal = authenticate(&cfg, &headers).expect("auth should pass");
    assert_eq!(principal.subject, "alice");
    assert_eq!(principal.team, "platform");
    assert!(principal.has_scope("chat"));
}
