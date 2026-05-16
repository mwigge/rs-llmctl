use anyhow::{bail, Context, Result};
use chrono::{Datelike, Duration, Utc};
use clap::{Args, Parser, Subcommand, ValueEnum};
use rs_llmctl::audit::{AuditEvent, ObservationEvent};
use rs_llmctl::config::{self, Config, Mode, ModelConfig, QuotaConfig, StorageConfig};
use rs_llmctl::model::{self, ModelInstallRequest, ModelSource};
use rs_llmctl::observability::{Exporter, ObservabilityPlan};
use rs_llmctl::quota::{self, Principal};
use rs_llmctl::reporting;
use rs_llmctl::storage::Storage;
use serde::Serialize;
use serde_json::json;
use std::path::{Path, PathBuf};
use tokio::fs;
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(name = "llmctl", version, about = "Control rs-llmctl model serving")]
struct Cli {
    #[arg(long, env = "LLMCTL_CONFIG", global = true)]
    config: Option<PathBuf>,
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Init(InitArgs),
    Server {
        #[command(subcommand)]
        command: ServerCommand,
    },
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    Swap {
        #[command(subcommand)]
        command: SwapCommand,
    },
    Quota {
        #[command(subcommand)]
        command: QuotaCommand,
    },
    Security {
        #[command(subcommand)]
        command: SecurityCommand,
    },
    Observe {
        #[command(subcommand)]
        command: ObserveCommand,
    },
    Audit {
        #[command(subcommand)]
        command: AuditCommand,
    },
    Usage {
        #[command(subcommand)]
        command: UsageCommand,
    },
    Data {
        #[command(subcommand)]
        command: DataCommand,
    },
}

#[derive(Debug, Args)]
struct InitArgs {
    #[arg(long)]
    force: bool,
    #[arg(long)]
    production: bool,
    #[arg(long)]
    bind: Option<String>,
}

#[derive(Debug, Subcommand)]
enum ServerCommand {
    Run,
    Check,
    SecurityCheck,
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    Install(ModelInstallArgs),
    ImportManifest(ModelImportManifestArgs),
    List,
}

#[derive(Debug, Subcommand)]
enum SwapCommand {
    Set(SwapSetArgs),
    Show,
}

#[derive(Debug, Args)]
struct SwapSetArgs {
    #[arg(long, value_enum)]
    mode: SwapMode,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum SwapMode {
    ColdSwap,
    HotSwap,
}

#[derive(Debug, Args)]
struct ModelInstallArgs {
    source: String,
    #[arg(long)]
    alias: String,
    #[arg(long, default_value = "chat")]
    role: String,
    #[arg(long, default_value_t = 1)]
    weight: u32,
    #[arg(long)]
    copy: bool,
    #[arg(long)]
    sha256: Option<String>,
}

#[derive(Debug, Args)]
struct ModelImportManifestArgs {
    manifest: PathBuf,
}

#[derive(Debug, Subcommand)]
enum QuotaCommand {
    Set(QuotaSetArgs),
    Status(QuotaStatusArgs),
    Report(ObserveWindowArgs),
    List,
}

#[derive(Debug, Args)]
struct QuotaSetArgs {
    #[arg(long)]
    subject: String,
    #[arg(long, default_value = "default")]
    team: String,
    #[arg(long, default_value_t = 60)]
    requests_per_minute: u32,
    #[arg(long, default_value_t = 100_000)]
    tokens_per_day: u64,
    #[arg(long, default_value_t = 4)]
    max_concurrency: u32,
    #[arg(long = "model")]
    allowed_models: Vec<String>,
}

#[derive(Debug, Args)]
struct QuotaStatusArgs {
    #[arg(long)]
    subject: String,
    #[arg(long)]
    model: String,
    #[arg(long)]
    team: Option<String>,
}

#[derive(Debug, Subcommand)]
enum SecurityCommand {
    Check,
}

#[derive(Debug, Subcommand)]
enum ObserveCommand {
    Snapshot,
    Plan,
    Drift(ObserveWindowArgs),
    Usage(ObserveWindowArgs),
    Show(ObserveShowArgs),
}

#[derive(Debug, Args)]
struct ObserveWindowArgs {
    #[arg(long, default_value_t = 24)]
    hours: i64,
}

#[derive(Debug, Args)]
struct ObserveShowArgs {
    #[arg(long, default_value_t = 20)]
    limit: i64,
    #[arg(long)]
    kind: Option<String>,
}

#[derive(Debug, Subcommand)]
enum AuditCommand {
    Report {
        #[command(subcommand)]
        command: AuditReportCommand,
    },
    Request(AuditRequestArgs),
}

#[derive(Debug, Subcommand)]
enum AuditReportCommand {
    Monthly(AuditReportMonthlyArgs),
    Request(AuditReportRequestArgs),
}

#[derive(Debug, Args)]
struct AuditReportMonthlyArgs {
    #[arg(long)]
    year: Option<i32>,
    #[arg(long)]
    month: Option<u32>,
    #[arg(long)]
    envelope: bool,
}

#[derive(Debug, Args)]
struct AuditReportRequestArgs {
    request_id: Uuid,
    #[arg(long)]
    envelope: bool,
}

#[derive(Debug, Args)]
struct AuditRequestArgs {
    #[arg(long)]
    actor: String,
    #[arg(long, default_value = "default")]
    team: String,
    #[arg(long)]
    action: String,
    #[arg(long)]
    resource: String,
    #[arg(long, default_value = "requested")]
    outcome: String,
}

#[derive(Debug, Subcommand)]
enum UsageCommand {
    Report(ObserveWindowArgs),
}

#[derive(Debug, Subcommand)]
enum DataCommand {
    Export(DataExportArgs),
}

#[derive(Debug, Args)]
struct DataExportArgs {
    #[arg(long, default_value_t = 24)]
    hours: i64,
    #[arg(long)]
    envelope: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config_path = cli.config.unwrap_or_else(config::default_config_path);

    match cli.command {
        Command::Init(args) => init(&config_path, args, cli.json).await,
        Command::Server { command } => server_command(&config_path, command, cli.json).await,
        Command::Model { command } => model_command(&config_path, command, cli.json).await,
        Command::Swap { command } => swap_command(&config_path, command, cli.json).await,
        Command::Quota { command } => quota_command(&config_path, command, cli.json).await,
        Command::Security { command } => security_command(&config_path, command, cli.json).await,
        Command::Observe { command } => observe_command(&config_path, command, cli.json).await,
        Command::Audit { command } => audit_command(&config_path, command, cli.json).await,
        Command::Usage { command } => usage_command(&config_path, command, cli.json).await,
        Command::Data { command } => data_command(&config_path, command, cli.json).await,
    }
}

async fn init(path: &Path, args: InitArgs, as_json: bool) -> Result<()> {
    if path.exists() && !args.force {
        bail!(
            "config already exists at {}; use --force to overwrite",
            path.display()
        );
    }

    let mut cfg = Config::default();
    cfg.security.production = args.production;
    if let Some(bind) = args.bind {
        cfg.server.host = bind;
        cfg.security.bind_external =
            cfg.server.host != "127.0.0.1" && cfg.server.host != "localhost";
    }

    create_storage_dirs(&cfg.storage).await?;
    config::save(path, &cfg).await?;
    init_storage(&cfg.storage).await?;
    emit(
        as_json,
        &json!({ "config": path, "database": cfg.storage.db_path, "model_dir": cfg.storage.model_dir }),
    )
}

async fn server_command(path: &Path, command: ServerCommand, as_json: bool) -> Result<()> {
    let cfg = load_config(path).await?;
    match command {
        ServerCommand::Run => {
            config::validate_production_security(&cfg)?;
            init_storage(&cfg.storage).await?;
            emit(
                as_json,
                &json!({ "status": "ready", "bind": format!("{}:{}", cfg.server.host, cfg.server.port) }),
            )?;
            rs_llmctl::server::serve(cfg).await
        }
        ServerCommand::Check => {
            create_storage_dirs(&cfg.storage).await?;
            init_storage(&cfg.storage).await?;
            emit(
                as_json,
                &json!({ "status": "ok", "config": path, "models": cfg.models.len(), "quotas": cfg.quotas.len() }),
            )
        }
        ServerCommand::SecurityCheck => {
            config::validate_production_security(&cfg)?;
            emit(
                as_json,
                &json!({ "status": "ok", "production": cfg.security.production, "require_auth": cfg.security.require_auth }),
            )
        }
    }
}

async fn model_command(path: &Path, command: ModelCommand, as_json: bool) -> Result<()> {
    let mut cfg = load_config(path).await?;
    match command {
        ModelCommand::Install(args) => {
            create_storage_dirs(&cfg.storage).await?;
            let installed = model::install_model(&ModelInstallRequest {
                alias: args.alias.clone(),
                source: model_source(&args.source),
                cache_dir: cfg.storage.model_dir.clone(),
                copy_to_cache: args.copy,
                expected_sha256: args.sha256,
                role: args.role.clone(),
                weight: args.weight,
            })
            .await?;

            upsert_model(&mut cfg.models, installed.config.clone());
            config::save(path, &cfg).await?;

            let storage = init_storage(&cfg.storage).await?;
            for model in &cfg.models {
                storage.upsert_model(model).await?;
            }

            emit(
                as_json,
                &json!({ "status": "installed", "model": installed, "models": cfg.models }),
            )
        }
        ModelCommand::ImportManifest(args) => {
            create_storage_dirs(&cfg.storage).await?;
            let installed = model::import_offline_manifest(&args.manifest).await?;

            for model in &installed {
                upsert_model(&mut cfg.models, model.config.clone());
            }
            config::save(path, &cfg).await?;

            let storage = init_storage(&cfg.storage).await?;
            for model in &cfg.models {
                storage.upsert_model(model).await?;
            }

            emit(
                as_json,
                &json!({ "status": "imported", "imported": installed, "models": cfg.models }),
            )
        }
        ModelCommand::List => emit(as_json, &cfg.models),
    }
}

async fn swap_command(path: &Path, command: SwapCommand, as_json: bool) -> Result<()> {
    let mut cfg = load_config(path).await?;
    match command {
        SwapCommand::Set(args) => {
            cfg.mode = args.mode.into();
            config::save(path, &cfg).await?;
            emit(
                as_json,
                &json!({ "status": "set", "mode": cfg.mode, "models": cfg.models.len() }),
            )
        }
        SwapCommand::Show => emit(
            as_json,
            &json!({ "mode": cfg.mode, "models": cfg.models.len(), "model_aliases": cfg.models.iter().map(|model| &model.alias).collect::<Vec<_>>() }),
        ),
    }
}

async fn quota_command(path: &Path, command: QuotaCommand, as_json: bool) -> Result<()> {
    let mut cfg = load_config(path).await?;
    match command {
        QuotaCommand::Set(args) => {
            upsert_quota(
                &mut cfg.quotas,
                QuotaConfig {
                    subject: args.subject,
                    team: args.team,
                    requests_per_minute: args.requests_per_minute,
                    tokens_per_day: args.tokens_per_day,
                    max_concurrency: args.max_concurrency,
                    allowed_models: args.allowed_models,
                },
            );
            config::save(path, &cfg).await?;
            emit(as_json, &json!({ "status": "set", "quotas": cfg.quotas }))
        }
        QuotaCommand::Status(args) => {
            let storage = init_storage(&cfg.storage).await?;
            let principal = quota_status_principal(&cfg.quotas, &args);
            let policy = matching_quota(&cfg.quotas, &principal);
            let subject_scoped = policy.is_some_and(|policy| policy.subject == principal.subject);
            let decision =
                quota::check_quota(&storage, &cfg.quotas, &principal, &args.model).await?;
            let now = Utc::now();
            let day_start = now.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc();
            let requests_last_minute = storage
                .allowed_quota_decision_count(
                    &principal,
                    subject_scoped,
                    now - Duration::minutes(1),
                    now,
                )
                .await?;
            let tokens_today = storage
                .usage_tokens_total(&principal, subject_scoped, day_start, now)
                .await?;

            emit(
                as_json,
                &json!({
                    "subject": principal.subject,
                    "team": principal.team,
                    "model": args.model,
                    "allowed": decision.allowed,
                    "reason": decision.reason,
                    "policy": policy,
                    "usage": {
                        "requests_last_minute": requests_last_minute,
                        "tokens_today": tokens_today
                    }
                }),
            )
        }
        QuotaCommand::Report(args) => {
            let storage = init_storage(&cfg.storage).await?;
            let (from, to) = window(args.hours);
            let usage_summary = reporting::usage_summary(&storage, from, to).await?;
            let decisions = storage.quota_decisions_between(from, to).await?;
            emit(
                as_json,
                &json!({
                    "hours": args.hours,
                    "from": from,
                    "to": to,
                    "generated_at": Utc::now(),
                    "policies": cfg.quotas,
                    "decisions": decisions,
                    "usage_summary": usage_summary
                }),
            )
        }
        QuotaCommand::List => emit(as_json, &cfg.quotas),
    }
}

async fn security_command(path: &Path, command: SecurityCommand, as_json: bool) -> Result<()> {
    let cfg = load_config(path).await?;
    match command {
        SecurityCommand::Check => {
            config::validate_production_security(&cfg)?;
            emit(
                as_json,
                &json!({
                    "status": "ok",
                    "production": cfg.security.production,
                    "require_auth": cfg.security.require_auth,
                    "bind_external": cfg.security.bind_external,
                    "host": cfg.server.host,
                    "api_keys": cfg.security.api_keys.len()
                }),
            )
        }
    }
}

async fn observe_command(path: &Path, command: ObserveCommand, as_json: bool) -> Result<()> {
    let cfg = load_config(path).await?;
    match command {
        ObserveCommand::Snapshot => {
            let storage = init_storage(&cfg.storage).await?;
            let (snapshot, plan) = rs_llmctl::resources::snapshot_and_plan(&cfg.resources);
            let value = if snapshot.total_memory_bytes == 0 {
                0.0
            } else {
                (snapshot.total_memory_bytes - snapshot.available_memory_bytes) as f64
                    / snapshot.total_memory_bytes as f64
            };
            let event = ObservationEvent {
                id: Uuid::new_v4(),
                request_id: None,
                at: Utc::now(),
                kind: "resource.snapshot".to_string(),
                model: "system".to_string(),
                source: "llmctl".to_string(),
                value,
                unit: "ratio".to_string(),
                attributes_json: json!({ "snapshot": snapshot, "budget_plan": plan }),
            };
            storage.insert_observation_event(&event).await?;
            emit(as_json, &event)
        }
        ObserveCommand::Plan => {
            let plan = ObservabilityPlan::from_config(&cfg)?;
            emit(as_json, &observability_plan_json(plan))
        }
        ObserveCommand::Drift(args) => {
            let storage = init_storage(&cfg.storage).await?;
            report_observations(&storage, "drift", args.hours, as_json).await
        }
        ObserveCommand::Usage(args) => {
            let storage = init_storage(&cfg.storage).await?;
            report_usage(&storage, args.hours, as_json).await
        }
        ObserveCommand::Show(args) => {
            let storage = init_storage(&cfg.storage).await?;
            show_observations(&storage, args, as_json).await
        }
    }
}

async fn audit_command(path: &Path, command: AuditCommand, as_json: bool) -> Result<()> {
    let cfg = load_config(path).await?;
    let storage = init_storage(&cfg.storage).await?;
    match command {
        AuditCommand::Report { command } => match command {
            AuditReportCommand::Monthly(args) => {
                let now = Utc::now();
                let year = args.year.unwrap_or_else(|| now.year());
                let month = args.month.unwrap_or_else(|| now.month());
                if args.envelope {
                    let report =
                        reporting::monthly_audit_report_envelope(&storage, year, month).await?;
                    emit(as_json, &report)
                } else {
                    let report = reporting::monthly_audit_report(&storage, year, month).await?;
                    emit(as_json, &report)
                }
            }
            AuditReportCommand::Request(args) => {
                if args.envelope {
                    let report =
                        reporting::per_request_audit_report_envelope(&storage, args.request_id)
                            .await?;
                    emit(as_json, &report)
                } else {
                    let report =
                        reporting::per_request_audit_report(&storage, args.request_id).await?;
                    emit(as_json, &report)
                }
            }
        },
        AuditCommand::Request(args) => {
            let event = AuditEvent::new(
                None,
                args.actor,
                args.team,
                args.action,
                args.resource,
                args.outcome,
                json!({ "source": "llmctl audit request" }),
            );
            storage.insert_audit_event(&event).await?;
            emit(as_json, &event)
        }
    }
}

async fn usage_command(path: &Path, command: UsageCommand, as_json: bool) -> Result<()> {
    let cfg = load_config(path).await?;
    let storage = init_storage(&cfg.storage).await?;
    match command {
        UsageCommand::Report(args) => report_usage(&storage, args.hours, as_json).await,
    }
}

async fn data_command(path: &Path, command: DataCommand, as_json: bool) -> Result<()> {
    let cfg = load_config(path).await?;
    let storage = init_storage(&cfg.storage).await?;
    match command {
        DataCommand::Export(args) => {
            let (from, to) = window(args.hours);
            if args.envelope {
                let report = reporting::data_export_envelope(&storage, from, to).await?;
                emit(as_json, &report)
            } else {
                let report = reporting::data_export(&storage, from, to).await?;
                emit(as_json, &report)
            }
        }
    }
}

async fn load_config(path: &Path) -> Result<Config> {
    config::load(path)
        .await
        .with_context(|| format!("load {}", path.display()))
}

async fn create_storage_dirs(storage: &StorageConfig) -> Result<()> {
    if let Some(parent) = storage.db_path.parent() {
        fs::create_dir_all(parent).await?;
    }
    fs::create_dir_all(&storage.model_dir).await?;
    Ok(())
}

async fn init_storage(storage: &StorageConfig) -> Result<Storage> {
    create_storage_dirs(storage).await?;
    Storage::connect(&storage.db_path).await
}

fn upsert_model(models: &mut Vec<ModelConfig>, model: ModelConfig) {
    if let Some(existing) = models.iter_mut().find(|m| m.alias == model.alias) {
        *existing = model;
    } else {
        models.push(model);
    }
}

fn upsert_quota(quotas: &mut Vec<QuotaConfig>, quota: QuotaConfig) {
    if let Some(existing) = quotas.iter_mut().find(|q| q.subject == quota.subject) {
        *existing = quota;
    } else {
        quotas.push(quota);
    }
}

fn quota_status_principal(quotas: &[QuotaConfig], args: &QuotaStatusArgs) -> Principal {
    let team = args.team.clone().unwrap_or_else(|| {
        quotas
            .iter()
            .find(|quota| quota.subject == args.subject)
            .map(|quota| quota.team.clone())
            .filter(|team| !team.is_empty())
            .unwrap_or_else(|| "public".to_string())
    });
    Principal {
        subject: args.subject.clone(),
        team,
        scopes: vec![],
    }
}

fn matching_quota<'a>(quotas: &'a [QuotaConfig], principal: &Principal) -> Option<&'a QuotaConfig> {
    quotas
        .iter()
        .find(|quota| quota.subject == principal.subject)
        .or_else(|| {
            quotas
                .iter()
                .find(|quota| !quota.team.is_empty() && quota.team == principal.team)
        })
}

fn observability_plan_json(plan: ObservabilityPlan) -> serde_json::Value {
    let exporter = match plan.exporter {
        Exporter::None => json!({ "type": "none" }),
        Exporter::Otlp {
            endpoint,
            protocol,
            headers,
            timeout_ms,
        } => json!({
            "type": "otlp",
            "endpoint": endpoint,
            "protocol": protocol,
            "headers": headers,
            "timeout_ms": timeout_ms
        }),
    };

    json!({
        "service_name": plan.service_name,
        "service_version": plan.service_version,
        "environment": plan.environment,
        "traces_enabled": plan.traces_enabled,
        "metrics_enabled": plan.metrics_enabled,
        "logs_enabled": plan.logs_enabled,
        "resource_attributes": plan.resource_attributes,
        "exporter": exporter
    })
}

fn model_source(source: &str) -> ModelSource {
    if let Some(model) = model::catalog_model(source) {
        return ModelSource::HuggingFace {
            repo: model.repo.to_string(),
            filename: model.filename.to_string(),
            revision: model.revision.to_string(),
        };
    }
    if source.starts_with("http://") || source.starts_with("https://") {
        ModelSource::DirectUrl {
            url: source.to_string(),
        }
    } else {
        ModelSource::LocalPath {
            path: PathBuf::from(source),
        }
    }
}

impl From<SwapMode> for Mode {
    fn from(mode: SwapMode) -> Self {
        match mode {
            SwapMode::ColdSwap => Mode::ColdSwap,
            SwapMode::HotSwap => Mode::HotSwap,
        }
    }
}

async fn report_observations(
    storage: &Storage,
    kind: &str,
    hours: i64,
    as_json: bool,
) -> Result<()> {
    let (from, to) = window(hours);
    let events = storage.observation_events_between(from, to).await?;
    let values: Vec<f64> = events
        .iter()
        .filter(|event| event.kind.contains(kind))
        .map(|event| event.value)
        .collect();
    let count = values.len();
    let avg_value = if count == 0 {
        None
    } else {
        Some(values.iter().sum::<f64>() / count as f64)
    };
    let max_value = values.iter().copied().reduce(f64::max);
    emit(
        as_json,
        &json!({ "hours": hours, "count": count, "avg_value": avg_value, "max_value": max_value }),
    )
}

async fn report_usage(storage: &Storage, hours: i64, as_json: bool) -> Result<()> {
    let (from, to) = window(hours);
    let summary = reporting::usage_summary(storage, from, to).await?;
    emit(as_json, &json!({ "hours": hours, "summary": summary }))
}

async fn show_observations(storage: &Storage, args: ObserveShowArgs, as_json: bool) -> Result<()> {
    let from = Utc::now() - Duration::days(3650);
    let mut events = storage.observation_events_between(from, Utc::now()).await?;
    if let Some(kind) = args.kind {
        events.retain(|event| event.kind == kind);
    }
    events.sort_by_key(|event| std::cmp::Reverse(event.at));
    events.truncate(args.limit.max(0) as usize);
    emit(as_json, &events)
}

fn window(hours: i64) -> (chrono::DateTime<Utc>, chrono::DateTime<Utc>) {
    let to = Utc::now();
    let from = if hours <= 0 {
        to - Duration::hours(24)
    } else {
        to - Duration::hours(hours)
    };
    (from, to)
}

fn emit<T: Serialize>(_: bool, value: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}
