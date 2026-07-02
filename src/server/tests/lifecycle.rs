use super::*;

#[tokio::test]
async fn serve_with_storage_and_shutdown_exits_when_shutdown_future_completes() {
    let storage = Storage::in_memory().await.expect("storage");
    let mut cfg = Config::default();
    cfg.server.port = 0;
    cfg.server.graceful_drain_seconds = 0;
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

    let server = tokio::spawn(serve_with_storage_and_shutdown(cfg, storage, async move {
        let _ = shutdown_rx.await;
    }));

    shutdown_tx.send(()).expect("send shutdown signal");
    let result = tokio::time::timeout(std::time::Duration::from_secs(2), server)
        .await
        .expect("server exits after shutdown")
        .expect("server task joins");

    result.expect("server result");
}

#[tokio::test]
async fn tls_enabled_without_cert_or_key_fails_before_serving() {
    let storage = Storage::in_memory().await.expect("storage");
    let mut cfg = Config::default();
    cfg.server.port = 0;
    cfg.server.tls.enabled = true;
    let (_shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

    let err = serve_with_storage_and_shutdown(cfg, storage, async move {
        let _ = shutdown_rx.await;
    })
    .await
    .expect_err("missing cert/key should fail before serving");

    assert!(
        err.to_string().contains("server.tls.cert-path"),
        "unexpected error: {err}"
    );
}
