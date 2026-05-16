use anyhow::{Context, Result};
use clap::Parser;
use rs_llmctl::config::{self, StorageConfig};
use rs_llmctl::storage::{Storage, StorageBackend};
use rs_llmctl::worker::{StartupPlan, TokioWorkerRunner, WorkerState, WorkerSupervisor};
use std::path::PathBuf;
use tokio::fs;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(name = "llmctld", version, about = "Run the rs-llmctl daemon")]
struct Cli {
    #[arg(long, env = "LLMCTL_CONFIG")]
    config: Option<PathBuf>,
    #[arg(long)]
    json_logs: bool,
    #[arg(long)]
    dry_run: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.json_logs);

    let config_path = cli.config.unwrap_or_else(config::default_config_path);
    let cfg = config::load(&config_path)
        .await
        .with_context(|| format!("load config {}", config_path.display()))?;

    config::validate_production_security(&cfg)?;
    init_storage(&cfg.storage).await?;
    let worker_startup_plan = StartupPlan::from_config(&cfg);
    for planned in &worker_startup_plan.workers {
        let env_names = planned
            .command
            .env
            .iter()
            .map(|(key, _value)| key.as_str())
            .collect::<Vec<_>>();
        tracing::debug!(
            worker = planned.worker.id.as_str(),
            model = planned.worker.model.alias.as_str(),
            upstream = %planned.worker.upstream(),
            program = %planned.command.program.display(),
            args = ?planned.command.args,
            env = ?env_names,
            "planned worker command"
        );
    }

    if cli.dry_run {
        println!("{}", serde_json::to_string_pretty(&worker_startup_plan)?);
        return Ok(());
    }

    tracing::info!(
        service = rs_llmctl::SERVICE_NAME,
        config = %config_path.display(),
        bind = %format!("{}:{}", cfg.server.host, cfg.server.port),
        workers = worker_startup_plan.workers.len(),
        "starting daemon"
    );

    let mut worker_supervisor = WorkerSupervisor::new(TokioWorkerRunner::new());
    let worker_statuses = worker_supervisor.start_all(&worker_startup_plan).await;
    if let Some(failed) = worker_statuses
        .iter()
        .find(|status| status.state == WorkerState::Failed)
    {
        anyhow::bail!(
            "failed to start worker {}: {}",
            failed.worker_id.as_str(),
            failed
                .last_error
                .as_deref()
                .unwrap_or("unknown worker error")
        );
    }

    rs_llmctl::server::serve_with_shutdown(cfg, rs_llmctl::server::shutdown_signal()).await
}

fn init_tracing(json_logs: bool) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let subscriber = tracing_subscriber::fmt().with_env_filter(filter);
    if json_logs {
        subscriber.json().init();
    } else {
        subscriber.init();
    }
}

async fn init_storage(storage: &StorageConfig) -> Result<Storage> {
    let plan = storage.connection_plan()?;
    if plan.backend == StorageBackend::Sqlite {
        if let Some(parent) = storage.db_path.parent() {
            fs::create_dir_all(parent).await?;
        }
    }
    fs::create_dir_all(&storage.model_dir).await?;
    Storage::connect_config(storage).await
}
