use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use chrono::{Datelike, Duration, Utc};
use clap::Parser;
use ed25519_dalek::{SigningKey, VerifyingKey};
use rand_core::{OsRng, RngCore};
use regex::Regex;
use rs_llmctl::audit::{AuditEvent, ObservationEvent};
use rs_llmctl::config::{
    self, ApiKeyConfig, Config, DataFabricFormat, EventFormat, LogFormat, Mode, ModelConfig,
    NativeEmbeddingMode, QuotaConfig, StorageConfig,
};
use rs_llmctl::contracts::{self, DatasetKind};
use rs_llmctl::integrations;
use rs_llmctl::model::{self, ModelInstallRequest, ModelSource};
use rs_llmctl::native;
use rs_llmctl::observability::{
    emit_runtime_telemetry, Exporter, ObservabilityPlan, RuntimeTelemetryEvent, TelemetryEventName,
    TelemetryRuntime, TelemetrySignal,
};
use rs_llmctl::policy_sign::{
    self, encode_b64, policy_log_entry_hash, require_algorithm, required_str, sha256_hex,
    verify_policy_log_values,
};
use rs_llmctl::quota::{self, Principal};
use rs_llmctl::reporting;
use rs_llmctl::runtime;
use rs_llmctl::storage::Storage;
use rs_llmctl::worker::{
    StartupPlan, SwapPlan, TokioWorkerRunner, WorkerId, WorkerLaunchPlan, WorkerState,
    WorkerSupervisor,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex as AsyncMutex;
use uuid::Uuid;

const DEFAULT_SERVICE_NAME: &str = "llmctld.service";

#[path = "llmctl/aiops.rs"]
mod aiops;
#[path = "llmctl/cli/mod.rs"]
mod cli;
#[path = "llmctl/first_run.rs"]
mod first_run;
#[path = "llmctl/service_lifecycle.rs"]
mod service_lifecycle;
use aiops::*;
use cli::*;
use first_run::*;
use service_lifecycle::*;
#[path = "llmctl/amd.rs"]
mod amd;
use amd::*;
#[path = "llmctl/integration.rs"]
mod integration;
use integration::*;
#[path = "llmctl/eval.rs"]
mod eval;
use eval::*;
#[path = "llmctl/policy.rs"]
mod policy;
use policy::*;
#[path = "llmctl/compliance.rs"]
mod compliance;
use compliance::*;
#[path = "llmctl/lineage.rs"]
mod lineage;
use lineage::*;
#[path = "llmctl/usage.rs"]
mod usage;
use usage::*;
#[path = "llmctl/swap.rs"]
mod swap;
use swap::*;
#[path = "llmctl/quota_cmd.rs"]
mod quota_cmd;
use quota_cmd::*;
#[path = "llmctl/runtime_cmd.rs"]
mod runtime_cmd;
use runtime_cmd::*;
#[path = "llmctl/service.rs"]
mod service;
use service::*;
#[path = "llmctl/data.rs"]
mod data;
use data::*;
#[path = "llmctl/observe.rs"]
mod observe;
use observe::*;
#[path = "llmctl/audit.rs"]
mod audit;
use audit::*;
#[path = "llmctl/security.rs"]
mod security;
use security::*;
#[path = "llmctl/model_cmd.rs"]
mod model_cmd;
use model_cmd::*;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config_path = cli.config.unwrap_or_else(config::default_config_path);

    match cli.command {
        Command::Init(args) => init(&config_path, args, cli.json).await,
        Command::FirstRun(args) => first_run(&config_path, args, cli.json).await,
        Command::Server { command } => server_command(&config_path, command, cli.json).await,
        Command::Model { command } => model_command(&config_path, command, cli.json).await,
        Command::Service { command } => service_command(command, cli.json).await,
        Command::Runtime { command } => runtime_command(&config_path, command, cli.json).await,
        Command::Swap { command } => swap_command(&config_path, command, cli.json).await,
        Command::Quota { command } => quota_command(&config_path, command, cli.json).await,
        Command::Security { command } => security_command(&config_path, command, cli.json).await,
        Command::Observe { command } => observe_command(&config_path, command, cli.json).await,
        Command::Audit { command } => audit_command(&config_path, command, cli.json).await,
        Command::Usage { command } => usage_command(&config_path, command, cli.json).await,
        Command::Data { command } => data_command(&config_path, command, cli.json).await,
        Command::Aiops { command } => aiops_command(command, cli.json).await,
        Command::Eval { command } => eval_command(&config_path, command, cli.json).await,
        Command::Lineage { command } => lineage_command(&config_path, command, cli.json).await,
        Command::Policy { command } => policy_command(command, cli.json).await,
        Command::Compliance { command } => {
            compliance_command(&config_path, command, cli.json).await
        }
        Command::Integration { command } => {
            integration_command(&config_path, command, cli.json).await
        }
        Command::Amd { command } => amd_command(command, cli.json).await,
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
    apply_init_profile(&mut cfg, &args);
    if let Some(bind) = args.bind {
        cfg.server.host = bind;
        cfg.security.bind_external =
            cfg.server.host != "127.0.0.1" && cfg.server.host != "localhost";
    }
    if args.production {
        cfg.security.production = true;
    }
    if let Some(endpoint) = args.otel_endpoint {
        cfg.observability.exporter.endpoint = Some(endpoint);
    }
    if let Some(format) = args.log_format {
        cfg.log.format = format.into();
    }
    if let Some(format) = args.event_format {
        cfg.events.format = format.into();
    }
    if let Some(format) = args.data_format {
        cfg.data_fabric.format = format.into();
        cfg.data_fabric.enabled = true;
    }
    if args.disable_sse {
        cfg.sse.enabled = false;
    }
    if args.tls_provider.is_some() || args.tls_evidence.is_some() || args.mtls {
        cfg.security.tls_termination.enabled = true;
        cfg.security.tls_termination.provider = args.tls_provider;
        cfg.security.tls_termination.evidence = args.tls_evidence;
        cfg.security.tls_termination.m_tls = args.mtls;
        if cfg.security.trusted_proxies.is_empty() {
            cfg.security.trusted_proxies = vec!["127.0.0.1".to_string()];
        }
    }

    create_storage_dirs(&cfg.storage).await?;
    config::save(path, &cfg).await?;
    init_storage(&cfg.storage).await?;
    emit(
        as_json,
        &json!({ "config": path, "database": cfg.storage.db_path, "model_dir": cfg.storage.model_dir }),
    )
}

fn apply_init_profile(cfg: &mut Config, args: &InitArgs) {
    match args.profile {
        InitProfile::LocalDev => {
            cfg.security.production = args.production;
        }
        InitProfile::CpuOnly => {
            cfg.resources.cpu_only = true;
            cfg.security.production = args.production;
        }
        InitProfile::ProductionAiops => {
            cfg.security.production = true;
            cfg.security.bind_external = true;
            cfg.security.require_auth = true;
            if cfg.security.trusted_proxies.is_empty() {
                cfg.security.trusted_proxies = vec!["127.0.0.1".to_string()];
            }
            cfg.audit.monthly_reports = true;
            cfg.audit.retention_days = 365;
            if cfg.audit.report_directory.is_none() {
                cfg.audit.report_directory = Some(
                    cfg.storage
                        .db_path
                        .parent()
                        .unwrap_or_else(|| Path::new("."))
                        .join("reports"),
                );
            }
            cfg.observability.traces_enabled = true;
            cfg.observability.metrics_enabled = true;
            cfg.observability.logs_enabled = true;
            cfg.log.format = LogFormat::Json;
            cfg.events.format = EventFormat::Jsonl;
            cfg.data_fabric.enabled = true;
            cfg.data_fabric.format = DataFabricFormat::ArrowJson;
        }
    }
}

async fn server_command(path: &Path, command: ServerCommand, as_json: bool) -> Result<()> {
    if let ServerCommand::PlanDiff(args) = command {
        let old_plan = read_startup_plan(&args.old_plan).await?;
        let new_plan = read_startup_plan(&args.new_plan).await?;
        return emit(as_json, &old_plan.diff(&new_plan));
    }

    let cfg = load_config(path).await?;
    match command {
        ServerCommand::Run => {
            config::validate_production_security(&cfg)?;
            let storage = init_storage(&cfg.storage).await?;
            let plan = StartupPlan::from_config(&cfg);
            let has_subprocess = plan
                .workers
                .iter()
                .any(|w| matches!(w.launch, WorkerLaunchPlan::LlamaServerSubprocess { .. }));

            let telemetry = TelemetryRuntime::install(&cfg, cfg.log.format == LogFormat::Json)?;

            let result = if has_subprocess {
                // AMD HIP path: llama-server subprocess handles all inference.
                // Skip Candle engine loading entirely — attempting to Candle-load
                // a 14B GGUF on CPU while the subprocess is serving it via GPU
                // would waste RAM and time.
                let mut supervisor = WorkerSupervisor::new(TokioWorkerRunner::new());
                let statuses = supervisor.start_all(&plan).await;
                // Boot readiness must reflect live workers: a worker that failed
                // to spawn or never became ready must not count toward readiness.
                let planned_count = statuses.len();
                let ready_count = statuses
                    .iter()
                    .filter(|status| status.state == WorkerState::Ready)
                    .count();
                let worker_control = Arc::new(AsyncMutex::new(supervisor));
                emit(
                    as_json,
                    &json!({
                        "status": if ready_count > 0 { "ready" } else { "no_ready_workers" },
                        "bind": format!("{}:{}", cfg.server.host, cfg.server.port),
                        "backend": "llama-server-subprocess",
                        "workers": planned_count,
                        "ready_workers": ready_count,
                        "native_engines": 0
                    }),
                )?;
                // Supervision loop: detect crashed workers and mark them
                // not-ready so routing avoids the dead port.
                let reaper = rs_llmctl::server::spawn_worker_reaper(worker_control.clone());
                let serve_result =
                    rs_llmctl::server::serve_with_storage_worker_control_and_shutdown(
                        cfg,
                        storage,
                        Some(worker_control),
                        rs_llmctl::server::shutdown_signal(),
                    )
                    .await;
                reaper.abort();
                serve_result
            } else {
                let engines = load_native_engines_from_config(&cfg)?;
                #[cfg(feature = "llama-cpp-native")]
                let backend_label = if plan
                    .workers
                    .iter()
                    .any(|w| matches!(w.launch, WorkerLaunchPlan::LlamaCppNative { .. }))
                {
                    "llama-cpp-native"
                } else {
                    "candle-native"
                };
                #[cfg(not(feature = "llama-cpp-native"))]
                let backend_label = "candle-native";
                emit(
                    as_json,
                    &json!({
                        "status": if engines.is_empty() { "no_models" } else { "ready" },
                        "bind": format!("{}:{}", cfg.server.host, cfg.server.port),
                        "backend": backend_label,
                        "native_engines": engines.len()
                    }),
                )?;
                rs_llmctl::server::serve_with_storage_and_native_engines(
                    cfg,
                    storage,
                    engines,
                    rs_llmctl::server::shutdown_signal(),
                )
                .await
            };
            let shutdown = telemetry.shutdown();
            result.and(shutdown)
        }
        ServerCommand::Check => {
            create_storage_dirs(&cfg.storage).await?;
            init_storage(&cfg.storage).await?;
            let placement = native::placement_plan_from_config(&cfg);
            native::validate_placement_plan(&placement)?;
            emit(
                as_json,
                &json!({ "status": "ok", "config": path, "models": cfg.models.len(), "quotas": cfg.quotas.len() }),
            )
        }
        ServerCommand::Plan => {
            let plan = StartupPlan::from_config(&cfg);
            emit(as_json, &plan)
        }
        ServerCommand::PlanDiff(_) => unreachable!("handled before config load"),
        ServerCommand::Status => {
            create_storage_dirs(&cfg.storage).await?;
            let storage = init_storage(&cfg.storage).await?;
            let status = rs_llmctl::server::readiness_status(&cfg, &storage).await;
            emit(as_json, &status)
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

fn load_native_engines_from_config(
    cfg: &Config,
) -> Result<rs_llmctl::server::NativeEngineRegistry> {
    let local_aliases = local_native_model_aliases(cfg)?;
    let models = cfg
        .models
        .iter()
        .filter(|model| model.weight > 0 && local_aliases.contains(&model.alias))
        .collect::<Vec<_>>();

    // When the llama-cpp-native feature is compiled in, check whether the startup
    // plan selected LlamaCppNative for any local model.  If it did, load those
    // models via LlamaCppNativeEngine and return early — the Candle factory is not
    // used on this code path.
    #[cfg(feature = "llama-cpp-native")]
    {
        let startup_plan = StartupPlan::from_config(cfg);
        let llama_cpp_by_alias: BTreeMap<String, u32> = startup_plan
            .workers
            .iter()
            .filter_map(|pw| {
                if let WorkerLaunchPlan::LlamaCppNative { gpu_layers } = &pw.launch {
                    Some((pw.worker.id.as_str().to_string(), *gpu_layers))
                } else {
                    None
                }
            })
            .collect();

        if !llama_cpp_by_alias.is_empty() {
            let mut engines = rs_llmctl::server::NativeEngineRegistry::new();
            for model in &models {
                if let Some(&gpu_layers) = llama_cpp_by_alias.get(&model.alias) {
                    let engine: Box<dyn native::NativeEngine> =
                        Box::new(native::LlamaCppNativeEngine::load(
                            model.alias.clone(),
                            &model.path,
                            gpu_layers,
                        )?);
                    engines.insert(model.alias.clone(), std::sync::Arc::from(engine));
                }
            }
            return Ok(engines);
        }
    }

    let factory = native::NativeCandleEngineFactory::default();
    let mut engines = rs_llmctl::server::NativeEngineRegistry::new();
    for model in models {
        let engine: Box<dyn native::NativeEngine> =
            if should_load_native_embedding_engine(cfg, model) {
                Box::new(native::NativeBertEmbeddingEngine::load(
                    model.alias.clone(),
                    &model.path,
                )?)
            } else {
                let family = configured_candle_family(model)?;
                let plan = factory.plan(family, model, &cfg.resources)?;
                factory.load(&plan)?
            };
        engines.insert(model.alias.clone(), std::sync::Arc::from(engine));
    }
    Ok(engines)
}

fn local_native_model_aliases(cfg: &Config) -> Result<BTreeSet<String>> {
    let placement = native::placement_plan_from_config(cfg);
    native::validate_placement_plan(&placement)?;
    let local = placement
        .nodes
        .iter()
        .find(|node| node.id == cfg.cluster.node_id)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "cluster.node-id `{}` is not present in cluster.nodes",
                cfg.cluster.node_id
            )
        })?;
    Ok(local.model_aliases.iter().cloned().collect())
}

fn should_load_native_embedding_engine(cfg: &Config, model: &ModelConfig) -> bool {
    if cfg.runtime.embeddings.mode != NativeEmbeddingMode::Semantic {
        return false;
    }
    cfg.runtime
        .embeddings
        .model_alias
        .as_deref()
        .map(|alias| alias == model.alias)
        .unwrap_or_else(|| model.role.eq_ignore_ascii_case("embedding"))
}

fn configured_candle_family(model: &ModelConfig) -> Result<native::CandleModelFamily> {
    let family = model.family.as_deref().map(str::trim).filter(|value| !value.is_empty()).ok_or_else(|| {
        anyhow::anyhow!(
            "model {} must set family for native Candle loading; supported families are qwen3, gemma4, deepseek, mistral, kimi, minimax",
            model.alias
        )
    })?;
    match family.to_ascii_lowercase().as_str() {
        "qwen3" | "qwen" => Ok(native::CandleModelFamily::Qwen3),
        "gemma4" | "gemma3" | "gemma" => Ok(native::CandleModelFamily::Gemma4),
        "deepseek" | "deepseek2" => Ok(native::CandleModelFamily::DeepSeek),
        "mistral" => Ok(native::CandleModelFamily::Mistral),
        "kimi" => Ok(native::CandleModelFamily::Kimi),
        "minimax" | "mini-max" => Ok(native::CandleModelFamily::MiniMax),
        other => bail!(
            "model {} has unsupported native Candle family {other}; supported families are qwen3, gemma4, deepseek, mistral, kimi, minimax",
            model.alias
        ),
    }
}

pub(crate) fn generate_api_key_secret(prefix: &str) -> String {
    let cleaned = prefix
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '-' || *ch == '_')
        .collect::<String>();
    let prefix = if cleaned.is_empty() {
        "llmctl"
    } else {
        cleaned.as_str()
    };
    let mut bytes = [0u8; 32];
    OsRng.fill_bytes(&mut bytes);
    format!("{prefix}_{}", URL_SAFE_NO_PAD.encode(bytes))
}

pub(crate) async fn write_secret_file(path: &Path, secret: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .await
            .with_context(|| format!("create secret directory {}", parent.display()))?;
    }
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .await
        .with_context(|| format!("create api key secret file {}", path.display()))?;
    file.write_all(secret.as_bytes())
        .await
        .with_context(|| format!("write api key secret file {}", path.display()))?;
    file.write_all(b"\n")
        .await
        .with_context(|| format!("write api key secret file {}", path.display()))?;
    Ok(())
}

#[path = "llmctl/helpers.rs"]
mod helpers;
pub(crate) use helpers::*;

pub(crate) fn emit<T: Serialize>(_: bool, value: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}
