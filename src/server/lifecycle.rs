use super::{
    build_tls_acceptor, router_with_worker_control_native_engine_and_drain, serve_tls,
    NativeEngineRegistry, ServingLimits,
};
use crate::config::Config;
use crate::native;
use crate::observability::{
    emit_runtime_telemetry, RuntimeTelemetryEvent, TelemetryEventName, TelemetrySignal,
};
use crate::storage::Storage;
use crate::worker::{TokioWorkerRunner, WorkerSupervisor};
use anyhow::{Context, Result};
use chrono::Utc;
use serde_json::json;
use std::collections::BTreeMap;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;

pub async fn serve(cfg: Config) -> Result<()> {
    // Log hardware tier and recommended model once at startup. Advisory only —
    // operator-selected models are honoured regardless of tier.
    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    crate::tier::log_startup_recommendation();
    serve_with_shutdown(cfg, shutdown_signal()).await
}

pub async fn serve_with_shutdown<S>(cfg: Config, shutdown: S) -> Result<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    let storage = Storage::connect_config(&cfg.storage).await?;
    serve_with_storage_and_shutdown(cfg, storage, shutdown).await
}

pub async fn serve_with_storage(cfg: Config, storage: Storage) -> Result<()> {
    serve_with_storage_and_shutdown(cfg, storage, shutdown_signal()).await
}

pub async fn serve_with_storage_and_native_engine<S>(
    cfg: Config,
    storage: Storage,
    native_engine: Arc<dyn native::NativeEngine>,
    shutdown: S,
) -> Result<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    let mut native_engines = NativeEngineRegistry::new();
    native_engines.insert(native_engine.model_alias().to_string(), native_engine);
    serve_with_storage_and_native_engines(cfg, storage, native_engines, shutdown).await
}

pub async fn serve_with_storage_and_native_engines<S>(
    cfg: Config,
    storage: Storage,
    native_engines: NativeEngineRegistry,
    shutdown: S,
) -> Result<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    serve_with_storage_worker_control_native_engine_and_shutdown(
        cfg,
        storage,
        None,
        native_engines,
        shutdown,
    )
    .await
}

pub async fn serve_with_storage_and_shutdown<S>(
    cfg: Config,
    storage: Storage,
    shutdown: S,
) -> Result<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    serve_with_storage_worker_control_native_engine_and_shutdown(
        cfg,
        storage,
        None,
        NativeEngineRegistry::new(),
        shutdown,
    )
    .await
}

pub async fn serve_with_storage_worker_control_and_shutdown<S>(
    cfg: Config,
    storage: Storage,
    worker_control: Option<Arc<AsyncMutex<WorkerSupervisor<TokioWorkerRunner>>>>,
    shutdown: S,
) -> Result<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    serve_with_storage_worker_control_native_engine_and_shutdown(
        cfg,
        storage,
        worker_control,
        NativeEngineRegistry::new(),
        shutdown,
    )
    .await
}

async fn serve_with_storage_worker_control_native_engine_and_shutdown<S>(
    cfg: Config,
    storage: Storage,
    worker_control: Option<Arc<AsyncMutex<WorkerSupervisor<TokioWorkerRunner>>>>,
    native_engines: NativeEngineRegistry,
    shutdown: S,
) -> Result<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    if cfg.server.tls.enabled {
        return serve_https_with_storage_worker_control_native_engine_and_shutdown(
            cfg,
            storage,
            worker_control,
            native_engines,
            shutdown,
        )
        .await;
    }

    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    let limits = ServingLimits::from_config(&cfg);
    let draining = Arc::new(AtomicBool::new(false));
    let shutdown = drain_before_shutdown(
        shutdown,
        draining.clone(),
        cfg.server.graceful_drain_seconds,
    );
    let heartbeat = spawn_runtime_heartbeat(&cfg, draining.clone());
    let result = axum::serve(
        listener,
        router_with_worker_control_native_engine_and_drain(
            cfg,
            storage,
            limits,
            worker_control,
            native_engines,
            draining,
        )
        .into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown)
    .await
    .context("serve HTTP API");
    if let Some(handle) = heartbeat {
        handle.abort();
    }
    result
}

async fn serve_https_with_storage_worker_control_native_engine_and_shutdown<S>(
    cfg: Config,
    storage: Storage,
    worker_control: Option<Arc<AsyncMutex<WorkerSupervisor<TokioWorkerRunner>>>>,
    native_engines: NativeEngineRegistry,
    shutdown: S,
) -> Result<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    let tls_acceptor = build_tls_acceptor(&cfg).await?;
    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    let limits = ServingLimits::from_config(&cfg);
    let draining = Arc::new(AtomicBool::new(false));
    let shutdown = drain_before_shutdown(
        shutdown,
        draining.clone(),
        cfg.server.graceful_drain_seconds,
    );
    let app = router_with_worker_control_native_engine_and_drain(
        cfg.clone(),
        storage,
        limits,
        worker_control,
        native_engines,
        draining.clone(),
    );
    let heartbeat = spawn_runtime_heartbeat(&cfg, draining.clone());
    let result = serve_tls(listener, tls_acceptor, app, shutdown)
        .await
        .context("serve HTTPS API");
    if let Some(handle) = heartbeat {
        handle.abort();
    }
    result
}

async fn drain_before_shutdown<S>(shutdown: S, draining: Arc<AtomicBool>, drain_seconds: u64)
where
    S: Future<Output = ()>,
{
    shutdown.await;
    draining.store(true, Ordering::SeqCst);
    emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
        TelemetrySignal::Metric,
        TelemetryEventName::RuntimeHeartbeat,
        Utc::now(),
        BTreeMap::from([
            ("llmctl.server.draining".to_string(), json!(true)),
            (
                "llmctl.server.graceful_drain_seconds".to_string(),
                json!(drain_seconds),
            ),
        ]),
    ));
    if drain_seconds > 0 {
        tokio::time::sleep(Duration::from_secs(drain_seconds)).await;
    }
}

fn spawn_runtime_heartbeat(cfg: &Config, draining: Arc<AtomicBool>) -> Option<JoinHandle<()>> {
    let interval_seconds = cfg.runtime.heartbeat_interval_seconds;
    if interval_seconds == 0 {
        return None;
    }

    let cfg = cfg.clone();
    Some(tokio::spawn(async move {
        emit_runtime_heartbeat(&cfg, draining.load(Ordering::SeqCst));
        loop {
            tokio::time::sleep(Duration::from_secs(interval_seconds)).await;
            emit_runtime_heartbeat(&cfg, draining.load(Ordering::SeqCst));
        }
    }))
}

fn emit_runtime_heartbeat(cfg: &Config, draining: bool) {
    let heartbeat = native::heartbeat_from_config(cfg);
    let mut attributes = heartbeat.safe_telemetry_attributes();
    attributes.insert("llmctl.server.draining".to_string(), json!(draining));
    emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
        TelemetrySignal::Metric,
        TelemetryEventName::RuntimeHeartbeat,
        Utc::now(),
        attributes,
    ));
}

pub async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::warn!(%error, "failed to install Ctrl-C signal handler");
        }
    };

    #[cfg(unix)]
    {
        let terminate = async {
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(mut signal) => {
                    signal.recv().await;
                }
                Err(error) => {
                    tracing::warn!(%error, "failed to install terminate signal handler");
                    std::future::pending::<()>().await;
                }
            }
        };

        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }
    }

    #[cfg(not(unix))]
    ctrl_c.await;
}
