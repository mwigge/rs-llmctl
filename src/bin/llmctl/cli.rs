use crate::DEFAULT_SERVICE_NAME;
use chrono::Utc;
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

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

#[derive(Debug, Subcommand)]
pub(crate) enum ModelCommand {
    Install(ModelInstallArgs),
    Status(ModelStatusArgs),
    Start(ModelStartArgs),
    Stop(ModelStopArgs),
    Update(ModelReplaceArgs),
    Upgrade(ModelReplaceArgs),
    Downgrade(ModelReplaceArgs),
    Drift(ObserveWindowArgs),
    ImportManifest(ModelImportManifestArgs),
    Inventory,
    List,
    Profile {
        #[command(subcommand)]
        command: ModelProfileCommand,
    },
}

#[derive(Debug, Subcommand)]
pub(crate) enum ModelProfileCommand {
    List,
    Inspect(ModelProfileAliasArgs),
    ImportLocal(ModelProfileImportLocalArgs),
    ImportCatalog(ModelProfileImportCatalogArgs),
    Qualify(ModelProfileQualifyArgs),
    Quarantine(ModelProfileQuarantineArgs),
    Remove(ModelProfileAliasArgs),
    Adapters,
}

#[derive(Debug, Args)]
pub(crate) struct ModelProfileAliasArgs {
    pub(crate) alias: String,
}

#[derive(Debug, Args)]
pub(crate) struct ModelProfileImportLocalArgs {
    pub(crate) path: PathBuf,
    #[arg(long)]
    pub(crate) alias: String,
}

#[derive(Debug, Args)]
pub(crate) struct ModelProfileImportCatalogArgs {
    pub(crate) id: String,
}

#[derive(Debug, Args)]
pub(crate) struct ModelProfileQualifyArgs {
    pub(crate) alias: String,
    #[arg(long)]
    pub(crate) available_vram_bytes: Option<u64>,
}

#[derive(Debug, Args)]
pub(crate) struct ModelProfileQuarantineArgs {
    pub(crate) alias: String,
    #[arg(long)]
    pub(crate) reason: String,
}

#[derive(Debug, Subcommand)]
pub(crate) enum SwapCommand {
    Set(SwapSetArgs),
    Plan(SwapPlanArgs),
    Show,
}

#[derive(Debug, Args)]
pub(crate) struct SwapSetArgs {
    #[arg(long, value_enum)]
    pub(crate) mode: SwapMode,
}

#[derive(Debug, Args)]
pub(crate) struct SwapPlanArgs {
    #[arg(long)]
    pub(crate) active: String,
    #[arg(long)]
    pub(crate) replacement: String,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum SwapMode {
    ColdSwap,
    HotSwap,
}

#[derive(Debug, Subcommand)]
pub(crate) enum ServiceCommand {
    Status(ServiceLifecycleArgs),
    Start(ServiceLifecycleArgs),
    Stop(ServiceLifecycleArgs),
    Restart(ServiceLifecycleArgs),
    Upgrade(ServiceLifecycleArgs),
    Downgrade(ServiceLifecycleArgs),
}

#[derive(Debug, Subcommand)]
pub(crate) enum RuntimeCommand {
    Status,
    Heartbeat,
    Placement,
    Route(RuntimeRouteArgs),
    AmdQualification(RuntimeAmdQualificationArgs),
    Gemma4Readiness(RuntimeGemma4ReadinessArgs),
    ValidationPlan(RuntimeValidationPlanArgs),
    ValidationRun(RuntimeValidationRunArgs),
    Validate,
}

#[derive(Debug, Args)]
pub(crate) struct RuntimeRouteArgs {
    #[arg(long, conflicts_with = "role")]
    pub(crate) model: Option<String>,
    #[arg(long, conflicts_with = "model")]
    pub(crate) role: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct RuntimeValidationPlanArgs {
    #[arg(long, default_value_t = 240)]
    pub(crate) soak_minutes: u64,
    #[arg(long, default_value_t = 8)]
    pub(crate) streaming_concurrency: u32,
    #[arg(long, default_value_t = 3)]
    pub(crate) rotation_keys: u32,
    #[arg(long, default_value_t = 16)]
    pub(crate) quota_concurrency: u32,
}

#[derive(Debug, Args)]
pub(crate) struct RuntimeValidationRunArgs {
    #[arg(long)]
    pub(crate) evidence_output: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct RuntimeGemma4ReadinessArgs {
    #[arg(long)]
    pub(crate) model_path: PathBuf,
    #[arg(long, default_value = "gemma4")]
    pub(crate) alias: String,
    #[arg(long)]
    pub(crate) evidence_output: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct RuntimeAmdQualificationArgs {
    #[arg(long)]
    pub(crate) preview: bool,
    #[arg(long, alias = "community-opt-in")]
    pub(crate) arch_opt_in: bool,
    #[arg(long)]
    pub(crate) evidence: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct ServiceLifecycleArgs {
    #[arg(long, default_value = DEFAULT_SERVICE_NAME)]
    pub(crate) service_name: String,
    #[arg(long, conflicts_with = "system")]
    pub(crate) user: bool,
    #[arg(long, conflicts_with = "user")]
    pub(crate) system: bool,
    #[arg(long)]
    pub(crate) dry_run: bool,
}

#[derive(Debug, Args)]
pub(crate) struct ModelInstallArgs {
    pub(crate) source: String,
    #[arg(long)]
    pub(crate) alias: String,
    #[arg(long, default_value = "chat")]
    pub(crate) role: String,
    #[arg(long, default_value = "qwen3")]
    pub(crate) family: String,
    #[arg(long, default_value_t = 1)]
    pub(crate) weight: u32,
    #[arg(long)]
    pub(crate) copy: bool,
    #[arg(long)]
    pub(crate) sha256: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct ModelStatusArgs {
    pub(crate) alias: String,
}

#[derive(Debug, Args)]
pub(crate) struct ModelStartArgs {
    pub(crate) alias: String,
    #[arg(long)]
    pub(crate) weight: Option<u32>,
    #[arg(long)]
    pub(crate) dry_run: bool,
}

#[derive(Debug, Args)]
pub(crate) struct ModelStopArgs {
    pub(crate) alias: String,
    #[arg(long)]
    pub(crate) dry_run: bool,
}

#[derive(Debug, Args)]
pub(crate) struct ModelReplaceArgs {
    pub(crate) source: String,
    #[arg(long)]
    pub(crate) alias: String,
    #[arg(long)]
    pub(crate) new_alias: Option<String>,
    #[arg(long)]
    pub(crate) role: Option<String>,
    #[arg(long)]
    pub(crate) family: Option<String>,
    #[arg(long)]
    pub(crate) weight: Option<u32>,
    #[arg(long)]
    pub(crate) copy: bool,
    #[arg(long)]
    pub(crate) sha256: Option<String>,
    #[arg(long)]
    pub(crate) dry_run: bool,
}

#[derive(Debug, Args)]
pub(crate) struct ModelImportManifestArgs {
    pub(crate) manifest: PathBuf,
}

#[derive(Debug, Subcommand)]
pub(crate) enum QuotaCommand {
    Set(QuotaSetArgs),
    Status(QuotaStatusArgs),
    Report(ObserveWindowArgs),
    Export,
    Import(QuotaImportArgs),
    List,
}

#[derive(Debug, Args)]
pub(crate) struct QuotaSetArgs {
    #[arg(long)]
    pub(crate) subject: String,
    #[arg(long, default_value = "default")]
    pub(crate) team: String,
    #[arg(long, default_value_t = 60)]
    pub(crate) requests_per_minute: u32,
    #[arg(long, default_value_t = 100_000)]
    pub(crate) tokens_per_day: u64,
    #[arg(long, default_value_t = 4)]
    pub(crate) max_concurrency: u32,
    #[arg(long = "model")]
    pub(crate) allowed_models: Vec<String>,
}

#[derive(Debug, Args)]
pub(crate) struct QuotaStatusArgs {
    #[arg(long)]
    pub(crate) subject: String,
    #[arg(long)]
    pub(crate) model: String,
    #[arg(long)]
    pub(crate) team: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct QuotaImportArgs {
    pub(crate) path: PathBuf,
}

#[derive(Debug, Subcommand)]
pub(crate) enum SecurityCommand {
    Check,
    GenerateKey(SecurityGenerateKeyArgs),
    HashKey(SecurityHashKeyArgs),
    AddKey(SecurityAddKeyArgs),
    ListKeys,
    RotateKey(SecurityRotateKeyArgs),
    RevokeKey(SecurityRevokeKeyArgs),
    KeyUsage(SecurityKeyUsageArgs),
    AuditConfig(SecurityAuditConfigArgs),
}

#[derive(Debug, Args)]
pub(crate) struct SecurityGenerateKeyArgs {
    #[arg(long, default_value = "llmctl")]
    pub(crate) prefix: String,
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct SecurityHashKeyArgs {
    #[arg(long, conflicts_with = "env")]
    pub(crate) stdin: bool,
    #[arg(long)]
    pub(crate) env: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct SecurityAddKeyArgs {
    #[arg(long)]
    pub(crate) id: String,
    #[arg(long)]
    pub(crate) sha256: String,
    #[arg(long)]
    pub(crate) subject: String,
    #[arg(long)]
    pub(crate) team: String,
    #[arg(long = "scope")]
    pub(crate) scopes: Vec<String>,
    #[arg(long)]
    pub(crate) owner: Option<String>,
    #[arg(long)]
    pub(crate) purpose: Option<String>,
    #[arg(long)]
    pub(crate) expires_at: Option<chrono::DateTime<Utc>>,
    #[arg(long)]
    pub(crate) last_four: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct SecurityRotateKeyArgs {
    #[arg(long)]
    pub(crate) id: String,
    #[arg(long)]
    pub(crate) new_id: Option<String>,
    #[arg(long)]
    pub(crate) sha256: String,
    #[arg(long)]
    pub(crate) expires_at: Option<chrono::DateTime<Utc>>,
    #[arg(long)]
    pub(crate) last_four: Option<String>,
    #[arg(long)]
    pub(crate) reason: Option<String>,
    #[arg(long)]
    pub(crate) replace: bool,
}

#[derive(Debug, Args)]
pub(crate) struct SecurityRevokeKeyArgs {
    #[arg(long)]
    pub(crate) id: String,
    #[arg(long)]
    pub(crate) reason: Option<String>,
    #[arg(long)]
    pub(crate) remove: bool,
}

#[derive(Debug, Args)]
pub(crate) struct SecurityKeyUsageArgs {
    #[arg(long)]
    pub(crate) id: Option<String>,
    #[arg(long, default_value_t = 24)]
    pub(crate) hours: i64,
}

#[derive(Debug, Args)]
pub(crate) struct SecurityAuditConfigArgs {
    #[arg(long)]
    pub(crate) systemd_unit: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum ObserveCommand {
    Snapshot,
    Plan,
    Drift(ObserveWindowArgs),
    Usage(ObserveWindowArgs),
    Show(ObserveShowArgs),
}

#[derive(Debug, Args)]
pub(crate) struct ObserveWindowArgs {
    #[arg(long, default_value_t = 24)]
    pub(crate) hours: i64,
}

#[derive(Debug, Args)]
pub(crate) struct ObserveShowArgs {
    #[arg(long, default_value_t = 20)]
    pub(crate) limit: i64,
    #[arg(long)]
    pub(crate) kind: Option<String>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum AuditCommand {
    Report {
        #[command(subcommand)]
        command: AuditReportCommand,
    },
    Retention {
        #[command(subcommand)]
        command: AuditRetentionCommand,
    },
    Request(AuditRequestArgs),
}

#[derive(Debug, Subcommand)]
pub(crate) enum AuditReportCommand {
    Monthly(AuditReportMonthlyArgs),
    Request(AuditReportRequestArgs),
}

#[derive(Debug, Subcommand)]
pub(crate) enum AuditRetentionCommand {
    Plan(AuditRetentionPlanArgs),
    Apply(AuditRetentionApplyArgs),
}

#[derive(Debug, Args)]
pub(crate) struct AuditReportMonthlyArgs {
    #[arg(long)]
    pub(crate) year: Option<i32>,
    #[arg(long)]
    pub(crate) month: Option<u32>,
    #[arg(long)]
    pub(crate) envelope: bool,
    #[arg(long)]
    pub(crate) write: bool,
}

#[derive(Debug, Args)]
pub(crate) struct AuditReportRequestArgs {
    pub(crate) request_id: Uuid,
    #[arg(long)]
    pub(crate) envelope: bool,
}

#[derive(Debug, Args)]
pub(crate) struct AuditRetentionPlanArgs {
    #[arg(long)]
    pub(crate) envelope: bool,
}

#[derive(Debug, Args)]
pub(crate) struct AuditRetentionApplyArgs {
    #[arg(long)]
    pub(crate) yes: bool,
    #[arg(long)]
    pub(crate) envelope: bool,
}

#[derive(Debug, Args)]
pub(crate) struct AuditRequestArgs {
    #[arg(long)]
    pub(crate) actor: String,
    #[arg(long, default_value = "default")]
    pub(crate) team: String,
    #[arg(long)]
    pub(crate) action: String,
    #[arg(long)]
    pub(crate) resource: String,
    #[arg(long, default_value = "requested")]
    pub(crate) outcome: String,
}

#[derive(Debug, Subcommand)]
pub(crate) enum UsageCommand {
    Report(ObserveWindowArgs),
    Chargeback(UsageChargebackArgs),
}

#[derive(Debug, Args)]
pub(crate) struct UsageChargebackArgs {
    #[arg(long)]
    pub(crate) hours: i64,
    #[arg(long)]
    pub(crate) team: Option<String>,
    #[arg(long)]
    pub(crate) actor: Option<String>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum DataCommand {
    Export(DataExportArgs),
    Contracts(DataContractsArgs),
    VerifyEnvelope(DataVerifyEnvelopeArgs),
}

#[derive(Debug, Subcommand)]
pub(crate) enum AiopsCommand {
    Gaps,
    SloPlan(AiopsSloPlanArgs),
    IncidentTemplate(AiopsIncidentTemplateArgs),
}

#[derive(Debug, Subcommand)]
pub(crate) enum EvalCommand {
    Run(EvalRunArgs),
    RunSuite(EvalRunSuiteArgs),
    List,
    Report,
}

#[derive(Debug, Subcommand)]
pub(crate) enum LineageCommand {
    Record(LineageRecordArgs),
    List,
}

#[derive(Debug, Subcommand)]
pub(crate) enum PolicyCommand {
    Bundle(PolicyBundleArgs),
    VerifyBundle(PolicyVerifyBundleArgs),
    Keygen(PolicyKeygenArgs),
    Sign(PolicySignArgs),
    Verify(PolicyVerifyArgs),
    Log {
        #[command(subcommand)]
        command: PolicyLogCommand,
    },
    LegalHoldPlan(PolicyLegalHoldPlanArgs),
}

#[derive(Debug, Subcommand)]
pub(crate) enum PolicyLogCommand {
    Append(PolicyLogAppendArgs),
    Verify(PolicyLogVerifyArgs),
}

#[derive(Debug, Args)]
pub(crate) struct AiopsSloPlanArgs {
    #[arg(long, default_value_t = 99.0)]
    pub(crate) availability_percent: f64,
    #[arg(long, default_value_t = 2_000)]
    pub(crate) latency_p95_ms: u64,
    #[arg(long, default_value_t = 1.0)]
    pub(crate) error_rate_percent: f64,
    #[arg(long, value_enum, default_value_t = AiopsSloPlanFormat::Plan)]
    pub(crate) format: AiopsSloPlanFormat,
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum AiopsSloPlanFormat {
    Plan,
    Prometheus,
    Grafana,
}

#[derive(Debug, Args)]
pub(crate) struct AiopsIncidentTemplateArgs {
    #[arg(long, default_value = "undetermined")]
    pub(crate) severity: String,
    #[arg(long, default_value = "operations")]
    pub(crate) team: String,
}

#[derive(Debug, Args)]
pub(crate) struct EvalRunArgs {
    #[arg(long)]
    pub(crate) model: String,
    #[arg(long)]
    pub(crate) suite: String,
    #[arg(long)]
    pub(crate) score: f64,
    #[arg(long)]
    pub(crate) baseline: Option<f64>,
    #[arg(long)]
    pub(crate) notes: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct EvalRunSuiteArgs {
    #[arg(long)]
    pub(crate) manifest: PathBuf,
    #[arg(long)]
    pub(crate) base_url: Option<String>,
    #[arg(long)]
    pub(crate) api_key_env: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EvalSuiteManifest {
    pub(crate) suite: String,
    pub(crate) model: String,
    #[serde(default)]
    pub(crate) system: Option<String>,
    #[serde(default)]
    pub(crate) temperature: Option<f64>,
    #[serde(default)]
    pub(crate) max_tokens: Option<u64>,
    pub(crate) cases: Vec<EvalCaseManifest>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EvalCaseManifest {
    pub(crate) id: String,
    pub(crate) prompt: String,
    pub(crate) expect: EvalExpectation,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EvalExpectation {
    #[serde(default)]
    pub(crate) exact: Option<String>,
    #[serde(default)]
    pub(crate) contains: Vec<String>,
    #[serde(default)]
    pub(crate) regex: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct LineageRecordArgs {
    #[arg(long, value_enum)]
    pub(crate) kind: LineageKind,
    #[arg(long)]
    pub(crate) id: String,
    #[arg(long = "parent")]
    pub(crate) parents: Vec<String>,
    #[arg(long)]
    pub(crate) sha256: Option<String>,
    #[arg(long)]
    pub(crate) source: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[value(rename_all = "kebab-case")]
#[serde(rename_all = "kebab-case")]
pub(crate) enum LineageKind {
    Prompt,
    Corpus,
    EmbeddingIndex,
    Model,
    Release,
}

#[derive(Debug, Args)]
pub(crate) struct PolicyBundleArgs {
    #[arg(long)]
    pub(crate) name: String,
    #[arg(long)]
    pub(crate) input: PathBuf,
    #[arg(long)]
    pub(crate) output: PathBuf,
    #[arg(long)]
    pub(crate) signing_key_env: String,
}

#[derive(Debug, Args)]
pub(crate) struct PolicyVerifyBundleArgs {
    pub(crate) path: PathBuf,
    #[arg(long)]
    pub(crate) signing_key_env: String,
}

#[derive(Debug, Args)]
pub(crate) struct PolicyKeygenArgs {
    #[arg(long)]
    pub(crate) private_key: PathBuf,
    #[arg(long)]
    pub(crate) public_key: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct PolicySignArgs {
    #[arg(long)]
    pub(crate) input: PathBuf,
    #[arg(long)]
    pub(crate) signature: PathBuf,
    #[arg(long)]
    pub(crate) private_key: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct PolicyVerifyArgs {
    #[arg(long)]
    pub(crate) input: PathBuf,
    #[arg(long)]
    pub(crate) signature: PathBuf,
    #[arg(long)]
    pub(crate) public_key: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct PolicyLogAppendArgs {
    #[arg(long = "log")]
    pub(crate) log_path: PathBuf,
    #[arg(long)]
    pub(crate) artifact: PathBuf,
    #[arg(long)]
    pub(crate) signature: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct PolicyLogVerifyArgs {
    #[arg(long = "log")]
    pub(crate) log_path: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct PolicyLegalHoldPlanArgs {
    #[arg(long, value_enum)]
    pub(crate) dataset: DataContractDataset,
    #[arg(long)]
    pub(crate) case_id: String,
    #[arg(long)]
    pub(crate) reason: String,
}

#[derive(Debug, Subcommand)]
pub(crate) enum ComplianceCommand {
    Evidence,
    CraArticle14,
    PciDss,
    ReleaseChecklist,
}

#[derive(Debug, Subcommand)]
pub(crate) enum IntegrationCommand {
    AqeContract,
}

#[derive(Debug, Subcommand)]
pub(crate) enum AmdCommand {
    Qualify(AmdQualifyArgs),
    InstallServer(AmdInstallServerArgs),
}

#[derive(Debug, Args)]
pub(crate) struct AmdQualifyArgs {
    #[arg(long)]
    pub(crate) preview: bool,
    #[arg(long, alias = "community-opt-in")]
    pub(crate) arch_opt_in: bool,
    #[arg(long)]
    pub(crate) evidence: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct AmdInstallServerArgs {
    #[arg(long)]
    pub(crate) dry_run: bool,
    #[arg(
        long,
        help = "Path to install-amd-hip.sh (defaults to scripts/install-amd-hip.sh in cwd)"
    )]
    pub(crate) script: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct DataExportArgs {
    #[arg(long, default_value_t = 24)]
    pub(crate) hours: i64,
    #[arg(long, value_enum, default_value_t = DataDataset::All)]
    pub(crate) dataset: DataDataset,
    #[arg(long, value_enum, default_value_t = DataExportFormat::Json)]
    pub(crate) format: DataExportFormat,
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,
    #[arg(long)]
    pub(crate) envelope: bool,
    #[arg(long, default_value_t = 1_000_000)]
    pub(crate) max_rows: usize,
}

#[derive(Debug, Args)]
pub(crate) struct DataContractsArgs {
    #[arg(long, value_enum)]
    pub(crate) dataset: Option<DataContractDataset>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum DataDataset {
    All,
    Security,
    Observability,
    Usage,
    User,
    Finops,
    Models,
    Drift,
    Audit,
    Lineage,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum DataContractDataset {
    Security,
    Observability,
    Usage,
    User,
    Finops,
    Models,
    Drift,
    Audit,
    Lineage,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum DataExportFormat {
    Json,
    Jsonl,
    ArrowJson,
    ArrowIpc,
    Parquet,
}

#[derive(Debug, Args)]
pub(crate) struct DataVerifyEnvelopeArgs {
    pub(crate) path: PathBuf,
}
