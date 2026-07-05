use crate::DEFAULT_SERVICE_NAME;
use chrono::Utc;
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

mod model;
pub(crate) use model::*;
mod ops;
pub(crate) use ops::*;
mod extras;
pub(crate) use extras::*;

#[derive(Debug, Parser)]
#[command(name = "llmctl", version, about = "Control rs-llmctl model serving")]
pub(crate) struct Cli {
    #[arg(long, env = "LLMCTL_CONFIG", global = true)]
    pub(crate) config: Option<PathBuf>,
    #[arg(long, global = true)]
    pub(crate) json: bool,
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    Init(InitArgs),
    FirstRun(FirstRunArgs),
    Server {
        #[command(subcommand)]
        command: ServerCommand,
    },
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    Service {
        #[command(subcommand)]
        command: ServiceCommand,
    },
    Runtime {
        #[command(subcommand)]
        command: RuntimeCommand,
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
    Aiops {
        #[command(subcommand)]
        command: AiopsCommand,
    },
    Eval {
        #[command(subcommand)]
        command: EvalCommand,
    },
    Lineage {
        #[command(subcommand)]
        command: LineageCommand,
    },
    Policy {
        #[command(subcommand)]
        command: PolicyCommand,
    },
    Compliance {
        #[command(subcommand)]
        command: ComplianceCommand,
    },
    Integration {
        #[command(subcommand)]
        command: IntegrationCommand,
    },
    Amd {
        #[command(subcommand)]
        command: AmdCommand,
    },
}

#[derive(Debug, Args)]
pub(crate) struct InitArgs {
    #[arg(long)]
    pub(crate) force: bool,
    #[arg(long)]
    pub(crate) production: bool,
    #[arg(long, value_enum, default_value_t = InitProfile::LocalDev)]
    pub(crate) profile: InitProfile,
    #[arg(long)]
    pub(crate) bind: Option<String>,
    #[arg(long)]
    pub(crate) otel_endpoint: Option<String>,
    #[arg(long, value_enum)]
    pub(crate) log_format: Option<CliLogFormat>,
    #[arg(long, value_enum)]
    pub(crate) event_format: Option<CliEventFormat>,
    #[arg(long, value_enum)]
    pub(crate) data_format: Option<CliDataFormat>,
    #[arg(long)]
    pub(crate) disable_sse: bool,
    #[arg(long)]
    pub(crate) tls_provider: Option<String>,
    #[arg(long)]
    pub(crate) tls_evidence: Option<String>,
    #[arg(long)]
    pub(crate) mtls: bool,
}

#[derive(Debug, Args)]
pub(crate) struct FirstRunArgs {
    #[arg(long)]
    pub(crate) apply: bool,
    #[arg(long)]
    pub(crate) secret_output: Option<PathBuf>,
    #[arg(long, default_value = "llmctl")]
    pub(crate) key_prefix: String,
    #[arg(long, default_value = "operator-first-run")]
    pub(crate) api_key_id: String,
    #[arg(long, default_value = "operator")]
    pub(crate) subject: String,
    #[arg(long, default_value = "platform")]
    pub(crate) team: String,
    #[arg(long = "scope")]
    pub(crate) scopes: Vec<String>,
    #[arg(long)]
    pub(crate) owner: Option<String>,
    #[arg(long)]
    pub(crate) purpose: Option<String>,
    #[arg(long)]
    pub(crate) data_dir: Option<PathBuf>,
    #[arg(long)]
    pub(crate) starter_model_path: Option<PathBuf>,
    #[arg(long, default_value = "qwen")]
    pub(crate) starter_model_alias: String,
    #[arg(long, default_value = "chat")]
    pub(crate) starter_model_role: String,
    #[arg(long, default_value = "qwen3")]
    pub(crate) starter_model_family: String,
    #[arg(long, default_value_t = 1)]
    pub(crate) starter_model_weight: u32,
    #[arg(long)]
    pub(crate) base_url: Option<String>,
    #[arg(long, default_value = "LLMCTL_API_KEY")]
    pub(crate) api_key_env: String,
    #[arg(long)]
    pub(crate) smoke_question: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum InitProfile {
    LocalDev,
    ProductionAiops,
    CpuOnly,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum CliLogFormat {
    Pretty,
    Json,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum CliEventFormat {
    Json,
    Jsonl,
    CloudEvents,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum CliDataFormat {
    Json,
    Jsonl,
    ArrowJson,
    ArrowIpc,
    Parquet,
}

#[derive(Debug, Subcommand)]
pub(crate) enum ServerCommand {
    Run,
    Check,
    Plan,
    PlanDiff(ServerPlanDiffArgs),
    Status,
    SecurityCheck,
}

#[derive(Debug, Args)]
pub(crate) struct ServerPlanDiffArgs {
    pub(crate) old_plan: PathBuf,
    pub(crate) new_plan: PathBuf,
}
