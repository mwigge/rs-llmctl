use crate::config::Config;
use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::Router;
use futures_util::{pin_mut, FutureExt};
use hyper::body::Incoming;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as HyperBuilder;
use hyper_util::service::TowerToHyperService;
use std::future::Future;
use std::sync::Arc;
use tokio::time::timeout;
use tokio_rustls::TlsAcceptor;
use tower::ServiceExt;

use super::TLS_HANDSHAKE_TIMEOUT;

pub(super) async fn build_tls_acceptor(cfg: &Config) -> Result<TlsAcceptor> {
    let cert_path = cfg
        .server
        .tls
        .cert_path
        .as_ref()
        .context("server.tls.cert-path is required when server TLS is enabled")?;
    let key_path = cfg
        .server
        .tls
        .key_path
        .as_ref()
        .context("server.tls.key-path is required when server TLS is enabled")?;
    anyhow::ensure!(
        !cfg.server.tls.require_client_cert,
        "server.tls.require-client-cert is not supported without client CA configuration"
    );

    let cert_bytes = tokio::fs::read(cert_path)
        .await
        .with_context(|| format!("read TLS certificate {}", cert_path.display()))?;
    let key_bytes = tokio::fs::read(key_path)
        .await
        .with_context(|| format!("read TLS private key {}", key_path.display()))?;

    use rustls::pki_types::pem::PemObject;
    let certs = rustls::pki_types::CertificateDer::pem_slice_iter(&cert_bytes)
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("parse TLS certificate {}", cert_path.display()))?;
    anyhow::ensure!(
        !certs.is_empty(),
        "TLS certificate {} did not contain any certificates",
        cert_path.display()
    );
    let key = rustls::pki_types::PrivateKeyDer::from_pem_slice(&key_bytes)
        .with_context(|| format!("parse TLS private key {}", key_path.display()))?;
    let tls_config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .context("build TLS server config")?;

    Ok(TlsAcceptor::from(Arc::new(tls_config)))
}

pub(super) async fn serve_tls<S>(
    listener: tokio::net::TcpListener,
    tls_acceptor: TlsAcceptor,
    app: Router,
    shutdown: S,
) -> Result<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    let (signal_tx, signal_rx) = tokio::sync::watch::channel(());
    let signal_tx = Arc::new(signal_tx);
    tokio::spawn(async move {
        shutdown.await;
        drop(signal_rx);
    });

    let (close_tx, close_rx) = tokio::sync::watch::channel(());
    loop {
        let (tcp_stream, remote_addr) = tokio::select! {
            result = listener.accept() => result.context("accept TLS connection")?,
            _ = signal_tx.closed() => break,
        };

        let tls_acceptor = tls_acceptor.clone();
        let app = app.clone();
        let signal_tx = Arc::clone(&signal_tx);
        let close_rx = close_rx.clone();

        tokio::spawn(async move {
            let tls_stream = tokio::select! {
                result = timeout(TLS_HANDSHAKE_TIMEOUT, tls_acceptor.accept(tcp_stream)) => {
                    match result {
                        Ok(Ok(stream)) => stream,
                        Ok(Err(err)) => {
                            tracing::debug!(%remote_addr, error = ?err, "TLS handshake failed");
                            drop(close_rx);
                            return;
                        }
                        Err(_) => {
                            tracing::debug!(%remote_addr, "TLS handshake timed out");
                            drop(close_rx);
                            return;
                        }
                    }
                }
                _ = signal_tx.closed() => {
                    drop(close_rx);
                    return;
                }
            };
            let io = TokioIo::new(tls_stream);
            let tower_service = app.map_request(move |mut req: axum::http::Request<Incoming>| {
                req.extensions_mut().insert(ConnectInfo(remote_addr));
                req.map(Body::new)
            });
            let hyper_service = TowerToHyperService::new(tower_service);
            let builder = HyperBuilder::new(TokioExecutor::new());
            let conn = builder.serve_connection_with_upgrades(io, hyper_service);
            pin_mut!(conn);

            let signal_closed = signal_tx.closed().fuse();
            pin_mut!(signal_closed);

            loop {
                tokio::select! {
                    result = conn.as_mut() => {
                        if let Err(err) = result {
                            tracing::debug!(%remote_addr, error = ?err, "failed to serve TLS connection");
                        }
                        break;
                    }
                    _ = &mut signal_closed => {
                        conn.as_mut().graceful_shutdown();
                    }
                }
            }

            drop(close_rx);
        });
    }

    drop(close_rx);
    drop(listener);
    close_tx.closed().await;

    Ok(())
}
