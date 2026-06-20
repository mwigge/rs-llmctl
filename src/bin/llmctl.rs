use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use chrono::{Datelike, Duration, Utc};
use clap::{Args, Parser, Subcommand, ValueEnum};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use hmac::{Hmac, Mac};
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
use rs_llmctl::quota::{self, Principal};
use rs_llmctl::reporting;
use rs_llmctl::runtime;
use rs_llmctl::storage::Storage;
use rs_llmctl::worker::{
    StartupPlan, SwapPlan, TokioWorkerRunner, WorkerId, WorkerLaunchPlan, WorkerSupervisor,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs as stdfs;
use std::io::Read;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::process::Command as TokioCommand;
use tokio::sync::Mutex as AsyncMutex;
use uuid::Uuid;

const DEFAULT_SERVICE_NAME: &str = "llmctld.service";

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
struct InitArgs {
    #[arg(long)]
    force: bool,
    #[arg(long)]
    production: bool,
    #[arg(long, value_enum, default_value_t = InitProfile::LocalDev)]
    profile: InitProfile,
    #[arg(long)]
    bind: Option<String>,
    #[arg(long)]
    otel_endpoint: Option<String>,
    #[arg(long, value_enum)]
    log_format: Option<CliLogFormat>,
    #[arg(long, value_enum)]
    event_format: Option<CliEventFormat>,
    #[arg(long, value_enum)]
    data_format: Option<CliDataFormat>,
    #[arg(long)]
    disable_sse: bool,
    #[arg(long)]
    tls_provider: Option<String>,
    #[arg(long)]
    tls_evidence: Option<String>,
    #[arg(long)]
    mtls: bool,
}

#[derive(Debug, Args)]
struct FirstRunArgs {
    #[arg(long)]
    apply: bool,
    #[arg(long)]
    secret_output: Option<PathBuf>,
    #[arg(long, default_value = "llmctl")]
    key_prefix: String,
    #[arg(long, default_value = "operator-first-run")]
    api_key_id: String,
    #[arg(long, default_value = "operator")]
    subject: String,
    #[arg(long, default_value = "platform")]
    team: String,
    #[arg(long = "scope")]
    scopes: Vec<String>,
    #[arg(long)]
    owner: Option<String>,
    #[arg(long)]
    purpose: Option<String>,
    #[arg(long)]
    data_dir: Option<PathBuf>,
    #[arg(long)]
    starter_model_path: Option<PathBuf>,
    #[arg(long, default_value = "qwen")]
    starter_model_alias: String,
    #[arg(long, default_value = "chat")]
    starter_model_role: String,
    #[arg(long, default_value = "qwen3")]
    starter_model_family: String,
    #[arg(long, default_value_t = 1)]
    starter_model_weight: u32,
    #[arg(long)]
    base_url: Option<String>,
    #[arg(long, default_value = "LLMCTL_API_KEY")]
    api_key_env: String,
    #[arg(long)]
    smoke_question: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum InitProfile {
    LocalDev,
    ProductionAiops,
    CpuOnly,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum CliLogFormat {
    Pretty,
    Json,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum CliEventFormat {
    Json,
    Jsonl,
    CloudEvents,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum CliDataFormat {
    Json,
    Jsonl,
    ArrowJson,
    ArrowIpc,
    Parquet,
}

#[derive(Debug, Subcommand)]
enum ServerCommand {
    Run,
    Check,
    Plan,
    PlanDiff(ServerPlanDiffArgs),
    Status,
    SecurityCheck,
}

#[derive(Debug, Args)]
struct ServerPlanDiffArgs {
    old_plan: PathBuf,
    new_plan: PathBuf,
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
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
enum ModelProfileCommand {
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
struct ModelProfileAliasArgs {
    alias: String,
}

#[derive(Debug, Args)]
struct ModelProfileImportLocalArgs {
    path: PathBuf,
    #[arg(long)]
    alias: String,
}

#[derive(Debug, Args)]
struct ModelProfileImportCatalogArgs {
    id: String,
}

#[derive(Debug, Args)]
struct ModelProfileQualifyArgs {
    alias: String,
    #[arg(long)]
    available_vram_bytes: Option<u64>,
}

#[derive(Debug, Args)]
struct ModelProfileQuarantineArgs {
    alias: String,
    #[arg(long)]
    reason: String,
}

#[derive(Debug, Subcommand)]
enum SwapCommand {
    Set(SwapSetArgs),
    Plan(SwapPlanArgs),
    Show,
}

#[derive(Debug, Args)]
struct SwapSetArgs {
    #[arg(long, value_enum)]
    mode: SwapMode,
}

#[derive(Debug, Args)]
struct SwapPlanArgs {
    #[arg(long)]
    active: String,
    #[arg(long)]
    replacement: String,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum SwapMode {
    ColdSwap,
    HotSwap,
}

#[derive(Debug, Subcommand)]
enum ServiceCommand {
    Status(ServiceLifecycleArgs),
    Start(ServiceLifecycleArgs),
    Stop(ServiceLifecycleArgs),
    Restart(ServiceLifecycleArgs),
    Upgrade(ServiceLifecycleArgs),
    Downgrade(ServiceLifecycleArgs),
}

#[derive(Debug, Subcommand)]
enum RuntimeCommand {
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
struct RuntimeRouteArgs {
    #[arg(long, conflicts_with = "role")]
    model: Option<String>,
    #[arg(long, conflicts_with = "model")]
    role: Option<String>,
}

#[derive(Debug, Args)]
struct RuntimeValidationPlanArgs {
    #[arg(long, default_value_t = 240)]
    soak_minutes: u64,
    #[arg(long, default_value_t = 8)]
    streaming_concurrency: u32,
    #[arg(long, default_value_t = 3)]
    rotation_keys: u32,
    #[arg(long, default_value_t = 16)]
    quota_concurrency: u32,
}

#[derive(Debug, Args)]
struct RuntimeValidationRunArgs {
    #[arg(long)]
    evidence_output: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct RuntimeGemma4ReadinessArgs {
    #[arg(long)]
    model_path: PathBuf,
    #[arg(long, default_value = "gemma4")]
    alias: String,
    #[arg(long)]
    evidence_output: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct RuntimeAmdQualificationArgs {
    #[arg(long)]
    preview: bool,
    #[arg(long, alias = "community-opt-in")]
    arch_opt_in: bool,
    #[arg(long)]
    evidence: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ServiceLifecycleArgs {
    #[arg(long, default_value = DEFAULT_SERVICE_NAME)]
    service_name: String,
    #[arg(long, conflicts_with = "system")]
    user: bool,
    #[arg(long, conflicts_with = "user")]
    system: bool,
    #[arg(long)]
    dry_run: bool,
}

#[derive(Debug, Args)]
struct ModelInstallArgs {
    source: String,
    #[arg(long)]
    alias: String,
    #[arg(long, default_value = "chat")]
    role: String,
    #[arg(long, default_value = "qwen3")]
    family: String,
    #[arg(long, default_value_t = 1)]
    weight: u32,
    #[arg(long)]
    copy: bool,
    #[arg(long)]
    sha256: Option<String>,
}

#[derive(Debug, Args)]
struct ModelStatusArgs {
    alias: String,
}

#[derive(Debug, Args)]
struct ModelStartArgs {
    alias: String,
    #[arg(long)]
    weight: Option<u32>,
    #[arg(long)]
    dry_run: bool,
}

#[derive(Debug, Args)]
struct ModelStopArgs {
    alias: String,
    #[arg(long)]
    dry_run: bool,
}

#[derive(Debug, Args)]
struct ModelReplaceArgs {
    source: String,
    #[arg(long)]
    alias: String,
    #[arg(long)]
    new_alias: Option<String>,
    #[arg(long)]
    role: Option<String>,
    #[arg(long)]
    family: Option<String>,
    #[arg(long)]
    weight: Option<u32>,
    #[arg(long)]
    copy: bool,
    #[arg(long)]
    sha256: Option<String>,
    #[arg(long)]
    dry_run: bool,
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
    Export,
    Import(QuotaImportArgs),
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

#[derive(Debug, Args)]
struct QuotaImportArgs {
    path: PathBuf,
}

#[derive(Debug, Subcommand)]
enum SecurityCommand {
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
struct SecurityGenerateKeyArgs {
    #[arg(long, default_value = "llmctl")]
    prefix: String,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct SecurityHashKeyArgs {
    #[arg(long, conflicts_with = "env")]
    stdin: bool,
    #[arg(long)]
    env: Option<String>,
}

#[derive(Debug, Args)]
struct SecurityAddKeyArgs {
    #[arg(long)]
    id: String,
    #[arg(long)]
    sha256: String,
    #[arg(long)]
    subject: String,
    #[arg(long)]
    team: String,
    #[arg(long = "scope")]
    scopes: Vec<String>,
    #[arg(long)]
    owner: Option<String>,
    #[arg(long)]
    purpose: Option<String>,
    #[arg(long)]
    expires_at: Option<chrono::DateTime<Utc>>,
    #[arg(long)]
    last_four: Option<String>,
}

#[derive(Debug, Args)]
struct SecurityRotateKeyArgs {
    #[arg(long)]
    id: String,
    #[arg(long)]
    new_id: Option<String>,
    #[arg(long)]
    sha256: String,
    #[arg(long)]
    expires_at: Option<chrono::DateTime<Utc>>,
    #[arg(long)]
    last_four: Option<String>,
    #[arg(long)]
    reason: Option<String>,
    #[arg(long)]
    replace: bool,
}

#[derive(Debug, Args)]
struct SecurityRevokeKeyArgs {
    #[arg(long)]
    id: String,
    #[arg(long)]
    reason: Option<String>,
    #[arg(long)]
    remove: bool,
}

#[derive(Debug, Args)]
struct SecurityKeyUsageArgs {
    #[arg(long)]
    id: Option<String>,
    #[arg(long, default_value_t = 24)]
    hours: i64,
}

#[derive(Debug, Args)]
struct SecurityAuditConfigArgs {
    #[arg(long)]
    systemd_unit: Option<PathBuf>,
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
    Retention {
        #[command(subcommand)]
        command: AuditRetentionCommand,
    },
    Request(AuditRequestArgs),
}

#[derive(Debug, Subcommand)]
enum AuditReportCommand {
    Monthly(AuditReportMonthlyArgs),
    Request(AuditReportRequestArgs),
}

#[derive(Debug, Subcommand)]
enum AuditRetentionCommand {
    Plan(AuditRetentionPlanArgs),
    Apply(AuditRetentionApplyArgs),
}

#[derive(Debug, Args)]
struct AuditReportMonthlyArgs {
    #[arg(long)]
    year: Option<i32>,
    #[arg(long)]
    month: Option<u32>,
    #[arg(long)]
    envelope: bool,
    #[arg(long)]
    write: bool,
}

#[derive(Debug, Args)]
struct AuditReportRequestArgs {
    request_id: Uuid,
    #[arg(long)]
    envelope: bool,
}

#[derive(Debug, Args)]
struct AuditRetentionPlanArgs {
    #[arg(long)]
    envelope: bool,
}

#[derive(Debug, Args)]
struct AuditRetentionApplyArgs {
    #[arg(long)]
    yes: bool,
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
    Chargeback(UsageChargebackArgs),
}

#[derive(Debug, Args)]
struct UsageChargebackArgs {
    #[arg(long)]
    hours: i64,
    #[arg(long)]
    team: Option<String>,
    #[arg(long)]
    actor: Option<String>,
}

#[derive(Debug, Subcommand)]
enum DataCommand {
    Export(DataExportArgs),
    Contracts(DataContractsArgs),
    VerifyEnvelope(DataVerifyEnvelopeArgs),
}

#[derive(Debug, Subcommand)]
enum AiopsCommand {
    Gaps,
    SloPlan(AiopsSloPlanArgs),
    IncidentTemplate(AiopsIncidentTemplateArgs),
}

#[derive(Debug, Subcommand)]
enum EvalCommand {
    Run(EvalRunArgs),
    RunSuite(EvalRunSuiteArgs),
    List,
    Report,
}

#[derive(Debug, Subcommand)]
enum LineageCommand {
    Record(LineageRecordArgs),
    List,
}

#[derive(Debug, Subcommand)]
enum PolicyCommand {
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
enum PolicyLogCommand {
    Append(PolicyLogAppendArgs),
    Verify(PolicyLogVerifyArgs),
}

#[derive(Debug, Args)]
struct AiopsSloPlanArgs {
    #[arg(long, default_value_t = 99.0)]
    availability_percent: f64,
    #[arg(long, default_value_t = 2_000)]
    latency_p95_ms: u64,
    #[arg(long, default_value_t = 1.0)]
    error_rate_percent: f64,
    #[arg(long, value_enum, default_value_t = AiopsSloPlanFormat::Plan)]
    format: AiopsSloPlanFormat,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum AiopsSloPlanFormat {
    Plan,
    Prometheus,
    Grafana,
}

#[derive(Debug, Args)]
struct AiopsIncidentTemplateArgs {
    #[arg(long, default_value = "undetermined")]
    severity: String,
    #[arg(long, default_value = "operations")]
    team: String,
}

#[derive(Debug, Args)]
struct EvalRunArgs {
    #[arg(long)]
    model: String,
    #[arg(long)]
    suite: String,
    #[arg(long)]
    score: f64,
    #[arg(long)]
    baseline: Option<f64>,
    #[arg(long)]
    notes: Option<String>,
}

#[derive(Debug, Args)]
struct EvalRunSuiteArgs {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    base_url: Option<String>,
    #[arg(long)]
    api_key_env: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EvalSuiteManifest {
    suite: String,
    model: String,
    #[serde(default)]
    system: Option<String>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    max_tokens: Option<u64>,
    cases: Vec<EvalCaseManifest>,
}

#[derive(Debug, Deserialize)]
struct EvalCaseManifest {
    id: String,
    prompt: String,
    expect: EvalExpectation,
}

#[derive(Debug, Deserialize)]
struct EvalExpectation {
    #[serde(default)]
    exact: Option<String>,
    #[serde(default)]
    contains: Vec<String>,
    #[serde(default)]
    regex: Option<String>,
}

#[derive(Debug, Args)]
struct LineageRecordArgs {
    #[arg(long, value_enum)]
    kind: LineageKind,
    #[arg(long)]
    id: String,
    #[arg(long = "parent")]
    parents: Vec<String>,
    #[arg(long)]
    sha256: Option<String>,
    #[arg(long)]
    source: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[value(rename_all = "kebab-case")]
#[serde(rename_all = "kebab-case")]
enum LineageKind {
    Prompt,
    Corpus,
    EmbeddingIndex,
    Model,
    Release,
}

#[derive(Debug, Args)]
struct PolicyBundleArgs {
    #[arg(long)]
    name: String,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    signing_key_env: String,
}

#[derive(Debug, Args)]
struct PolicyVerifyBundleArgs {
    path: PathBuf,
    #[arg(long)]
    signing_key_env: String,
}

#[derive(Debug, Args)]
struct PolicyKeygenArgs {
    #[arg(long)]
    private_key: PathBuf,
    #[arg(long)]
    public_key: PathBuf,
}

#[derive(Debug, Args)]
struct PolicySignArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    signature: PathBuf,
    #[arg(long)]
    private_key: PathBuf,
}

#[derive(Debug, Args)]
struct PolicyVerifyArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    signature: PathBuf,
    #[arg(long)]
    public_key: PathBuf,
}

#[derive(Debug, Args)]
struct PolicyLogAppendArgs {
    #[arg(long = "log")]
    log_path: PathBuf,
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long)]
    signature: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct PolicyLogVerifyArgs {
    #[arg(long = "log")]
    log_path: PathBuf,
}

#[derive(Debug, Args)]
struct PolicyLegalHoldPlanArgs {
    #[arg(long, value_enum)]
    dataset: DataContractDataset,
    #[arg(long)]
    case_id: String,
    #[arg(long)]
    reason: String,
}

#[derive(Debug, Subcommand)]
enum ComplianceCommand {
    Evidence,
    CraArticle14,
    PciDss,
    ReleaseChecklist,
}

#[derive(Debug, Subcommand)]
enum IntegrationCommand {
    AqeContract,
}

#[derive(Debug, Subcommand)]
enum AmdCommand {
    Qualify(AmdQualifyArgs),
    InstallServer(AmdInstallServerArgs),
}

#[derive(Debug, Args)]
struct AmdQualifyArgs {
    #[arg(long)]
    preview: bool,
    #[arg(long, alias = "community-opt-in")]
    arch_opt_in: bool,
    #[arg(long)]
    evidence: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct AmdInstallServerArgs {
    #[arg(long)]
    dry_run: bool,
    #[arg(
        long,
        help = "Path to install-amd-hip.sh (defaults to scripts/install-amd-hip.sh in cwd)"
    )]
    script: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct DataExportArgs {
    #[arg(long, default_value_t = 24)]
    hours: i64,
    #[arg(long, value_enum, default_value_t = DataDataset::All)]
    dataset: DataDataset,
    #[arg(long, value_enum, default_value_t = DataExportFormat::Json)]
    format: DataExportFormat,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    envelope: bool,
    #[arg(long, default_value_t = 1_000_000)]
    max_rows: usize,
}

#[derive(Debug, Args)]
struct DataContractsArgs {
    #[arg(long, value_enum)]
    dataset: Option<DataContractDataset>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum DataDataset {
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
enum DataContractDataset {
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
enum DataExportFormat {
    Json,
    Jsonl,
    ArrowJson,
    ArrowIpc,
    Parquet,
}

#[derive(Debug, Args)]
struct DataVerifyEnvelopeArgs {
    path: PathBuf,
}

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

async fn first_run(path: &Path, args: FirstRunArgs, as_json: bool) -> Result<()> {
    validate_api_key_id(&args.api_key_id)?;
    validate_first_run_identity(&args)?;
    let scopes = first_run_scopes(&args);
    let config_exists = path.exists();
    let mut cfg = if config_exists {
        load_config(path).await?
    } else {
        first_run_default_config(path, args.data_dir.as_deref())
    };
    if let Some(data_dir) = args.data_dir.as_deref() {
        cfg.storage.db_path = data_dir.join("llmctl.db");
        cfg.storage.model_dir = data_dir.join("models");
    }

    let base_url = args
        .base_url
        .clone()
        .unwrap_or_else(|| first_run_base_url(&cfg));
    let smoke_model = first_run_smoke_model(&cfg, &args);
    let smoke_question = args
        .smoke_question
        .as_deref()
        .unwrap_or("Reply with only: llmctl smoke ok");
    let plan = first_run_plan_json(&FirstRunRenderContext {
        path,
        cfg: &cfg,
        args: &args,
        scopes: &scopes,
        config_existed: config_exists,
        base_url: &base_url,
        smoke_model: &smoke_model,
        smoke_question,
    });

    if !args.apply {
        return emit(as_json, &plan);
    }

    let secret_output = args
        .secret_output
        .as_deref()
        .context("first-run --apply requires --secret-output so the raw API key is written once outside config")?;
    if cfg
        .security
        .api_keys
        .iter()
        .any(|key| key.id == args.api_key_id)
    {
        bail!("api key id `{}` already exists", args.api_key_id);
    }

    let secret = generate_api_key_secret(&args.key_prefix);
    let sha256 = hex::encode(Sha256::digest(secret.as_bytes()));
    let last_four = last_four(&secret);
    let key = ApiKeyConfig {
        id: args.api_key_id.clone(),
        sha256,
        subject: args.subject.clone(),
        team: args.team.clone(),
        scopes: scopes.clone(),
        created_at: Some(Utc::now()),
        expires_at: None,
        rotated_at: None,
        owner: args.owner.clone(),
        purpose: args
            .purpose
            .clone()
            .or_else(|| Some("first-run operator access".to_string())),
        last_four: Some(last_four.clone()),
        fingerprint: None,
        status: "active".to_string(),
    };
    cfg.security.require_auth = true;
    cfg.security.api_keys.push(key);

    let installed_model = if let Some(model_path) = args.starter_model_path.as_ref() {
        create_storage_dirs(&cfg.storage).await?;
        let installed = model::install_model(&ModelInstallRequest {
            alias: args.starter_model_alias.clone(),
            source: ModelSource::LocalPath {
                path: model_path.clone(),
            },
            cache_dir: cfg.storage.model_dir.clone(),
            copy_to_cache: false,
            expected_sha256: None,
            role: args.starter_model_role.clone(),
            family: Some(args.starter_model_family.clone()),
            weight: args.starter_model_weight,
        })
        .await?;
        upsert_model(&mut cfg.models, installed.config.clone());
        Some(installed)
    } else {
        None
    };

    create_storage_dirs(&cfg.storage).await?;
    write_secret_file(secret_output, &secret).await?;
    config::save(path, &cfg).await?;
    let storage = init_storage(&cfg.storage).await?;
    for model in &cfg.models {
        storage.upsert_model(model).await?;
    }

    emit(
        as_json,
        &first_run_applied_json(
            &FirstRunRenderContext {
                path,
                cfg: &cfg,
                args: &args,
                scopes: &scopes,
                config_existed: config_exists,
                base_url: &base_url,
                smoke_model: &smoke_model,
                smoke_question,
            },
            secret_output,
            &last_four,
            installed_model.as_ref(),
        ),
    )
}

struct FirstRunRenderContext<'a> {
    path: &'a Path,
    cfg: &'a Config,
    args: &'a FirstRunArgs,
    scopes: &'a [String],
    config_existed: bool,
    base_url: &'a str,
    smoke_model: &'a str,
    smoke_question: &'a str,
}

fn first_run_default_config(path: &Path, data_dir: Option<&Path>) -> Config {
    let mut cfg = Config::default();
    let state_dir = data_dir.map(Path::to_path_buf).unwrap_or_else(|| {
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .join("rs-llmctl-state")
    });
    cfg.storage.db_path = state_dir.join("llmctl.db");
    cfg.storage.model_dir = state_dir.join("models");
    cfg.resources.cpu_only = true;
    cfg
}

fn validate_first_run_identity(args: &FirstRunArgs) -> Result<()> {
    if args.subject.trim().is_empty() {
        bail!("subject must not be empty");
    }
    if args.team.trim().is_empty() {
        bail!("team must not be empty");
    }
    if args.key_prefix.trim().is_empty() {
        bail!("key-prefix must not be empty");
    }
    if args.api_key_env.trim().is_empty() {
        bail!("api-key-env must not be empty");
    }
    if args.starter_model_alias.trim().is_empty() {
        bail!("starter-model-alias must not be empty");
    }
    if args.starter_model_role.trim().is_empty() {
        bail!("starter-model-role must not be empty");
    }
    Ok(())
}

fn first_run_scopes(args: &FirstRunArgs) -> Vec<String> {
    if args.scopes.is_empty() {
        vec!["chat".to_string(), "models.read".to_string()]
    } else {
        args.scopes.clone()
    }
}

fn first_run_base_url(cfg: &Config) -> String {
    format!("http://{}:{}/v1", cfg.server.host, cfg.server.port)
}

fn first_run_smoke_model(cfg: &Config, args: &FirstRunArgs) -> String {
    if args.starter_model_path.is_some() {
        return args.starter_model_alias.clone();
    }
    cfg.models
        .iter()
        .find(|model| model.weight > 0)
        .or_else(|| cfg.models.first())
        .map(|model| model.alias.clone())
        .unwrap_or_else(|| args.starter_model_alias.clone())
}

fn first_run_plan_json(context: &FirstRunRenderContext<'_>) -> Value {
    let next_command = first_run_apply_command(context);
    json!({
        "status": "planned",
        "mode": "dry-run",
        "side_effects": false,
        "config": context.path,
        "config_exists": context.config_existed,
        "api_key": first_run_api_key_json(context.args, context.scopes, None, None),
        "starter_model": first_run_starter_model_plan(context.args),
        "config_changes": {
            "write_config": true,
            "require_auth": true,
            "storage_db_path": context.cfg.storage.db_path,
            "model_dir": context.cfg.storage.model_dir
        },
        "smoke": first_run_smoke_json(
            context.base_url,
            context.smoke_model,
            context.smoke_question,
            &context.args.api_key_env
        ),
        "next_command": shell_join(&next_command),
        "next_command_argv": next_command
    })
}

fn first_run_apply_command(context: &FirstRunRenderContext<'_>) -> Vec<String> {
    let args = context.args;
    let mut command = vec![
        "llmctl".to_string(),
        "--config".to_string(),
        context.path.display().to_string(),
        "first-run".to_string(),
        "--apply".to_string(),
        "--secret-output".to_string(),
        args.secret_output
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<secret-file>".to_string()),
        "--key-prefix".to_string(),
        args.key_prefix.clone(),
        "--api-key-id".to_string(),
        args.api_key_id.clone(),
        "--subject".to_string(),
        args.subject.clone(),
        "--team".to_string(),
        args.team.clone(),
    ];
    for scope in &args.scopes {
        command.push("--scope".to_string());
        command.push(scope.clone());
    }
    if let Some(owner) = &args.owner {
        command.push("--owner".to_string());
        command.push(owner.clone());
    }
    if let Some(purpose) = &args.purpose {
        command.push("--purpose".to_string());
        command.push(purpose.clone());
    }
    if let Some(data_dir) = &args.data_dir {
        command.push("--data-dir".to_string());
        command.push(data_dir.display().to_string());
    }
    if let Some(path) = &args.starter_model_path {
        command.push("--starter-model-path".to_string());
        command.push(path.display().to_string());
    }
    command.push("--starter-model-alias".to_string());
    command.push(args.starter_model_alias.clone());
    command.push("--starter-model-role".to_string());
    command.push(args.starter_model_role.clone());
    command.push("--starter-model-family".to_string());
    command.push(args.starter_model_family.clone());
    command.push("--starter-model-weight".to_string());
    command.push(args.starter_model_weight.to_string());
    if let Some(base_url) = &args.base_url {
        command.push("--base-url".to_string());
        command.push(base_url.clone());
    }
    command.push("--api-key-env".to_string());
    command.push(args.api_key_env.clone());
    if let Some(question) = &args.smoke_question {
        command.push("--smoke-question".to_string());
        command.push(question.clone());
    }
    command
}

fn shell_join(args: &[String]) -> String {
    args.iter()
        .map(|arg| {
            if arg.chars().all(|ch| {
                ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '/' | ':' | '<' | '>')
            }) {
                arg.clone()
            } else {
                format!("'{}'", arg.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn first_run_applied_json(
    context: &FirstRunRenderContext<'_>,
    secret_output: &Path,
    last_four: &str,
    installed_model: Option<&model::InstalledModel>,
) -> Value {
    json!({
        "status": "applied",
        "mode": "apply",
        "side_effects": true,
        "config": context.path,
        "config_existed": context.config_existed,
        "api_key": first_run_api_key_json(
            context.args,
            context.scopes,
            Some(secret_output),
            Some(last_four)
        ),
        "starter_model": first_run_starter_model_applied(context.args, installed_model),
        "config_changes": {
            "wrote_config": true,
            "require_auth": context.cfg.security.require_auth,
            "api_keys": context.cfg.security.api_keys.len(),
            "models": context.cfg.models.len(),
            "storage_db_path": context.cfg.storage.db_path,
            "model_dir": context.cfg.storage.model_dir
        },
        "smoke": first_run_smoke_json(
            context.base_url,
            context.smoke_model,
            context.smoke_question,
            &context.args.api_key_env
        )
    })
}

fn first_run_api_key_json(
    args: &FirstRunArgs,
    scopes: &[String],
    secret_output: Option<&Path>,
    last_four: Option<&str>,
) -> Value {
    json!({
        "action": "generate",
        "id": args.api_key_id,
        "subject": args.subject,
        "team": args.team,
        "scopes": scopes,
        "owner": args.owner,
        "purpose": args.purpose.as_deref().unwrap_or("first-run operator access"),
        "secret_output": args.secret_output,
        "secret_written": secret_output.map(|path| path.display().to_string()),
        "last_four": last_four,
        "sha256_present": secret_output.is_some(),
        "config_storage": "sha256-only",
        "plaintext_secret_storage": false,
        "print_secret": false
    })
}

fn first_run_starter_model_plan(args: &FirstRunArgs) -> Value {
    match args.starter_model_path.as_ref() {
        Some(path) => json!({
            "action": "configure-local",
            "alias": args.starter_model_alias,
            "role": args.starter_model_role,
            "family": args.starter_model_family,
            "weight": args.starter_model_weight,
            "path": path,
            "source_kind": "local",
            "network": false,
            "exists": path.exists()
        }),
        None => json!({
            "action": "recommend",
            "alias": args.starter_model_alias,
            "role": args.starter_model_role,
            "family": args.starter_model_family,
            "weight": args.starter_model_weight,
            "source_kind": "none",
            "network": false,
            "recommendation": "provide --starter-model-path /path/to/model.gguf, /path/to/model.safetensors with sibling config.json/tokenizer.json, or a safetensors directory containing config.json, tokenizer.json, and weights; offline manifests are also supported"
        }),
    }
}

fn first_run_starter_model_applied(
    args: &FirstRunArgs,
    installed_model: Option<&model::InstalledModel>,
) -> Value {
    if let Some(installed) = installed_model {
        json!({
            "action": "configured",
            "alias": installed.alias,
            "role": installed.config.role,
            "family": installed.config.family,
            "weight": installed.config.weight,
            "path": installed.path,
            "source_kind": "local",
            "network": false,
            "sha256": installed.sha256,
            "bytes": installed.bytes,
            "verified": installed.verification.verified
        })
    } else {
        first_run_starter_model_plan(args)
    }
}

fn first_run_smoke_json(base_url: &str, model: &str, question: &str, api_key_env: &str) -> Value {
    json!({
        "action": "plan",
        "base_url": base_url,
        "model": model,
        "question": question,
        "api_key_env": api_key_env,
        "ask_question": {
            "helper": "ask_question",
            "crate": "rs-llmctl-client",
            "environment": {
                "LLMCTL_BASE_URL": base_url,
                "api_key_env": api_key_env
            },
            "metadata": {
                "session_id": "first-run-smoke",
                "purpose": "operator-smoke"
            }
        },
        "openai_compatible": {
            "method": "POST",
            "endpoint": "/v1/chat/completions",
            "url": format!("{}/chat/completions", base_url.trim_end_matches('/')),
            "headers": [
                format!("Authorization: Bearer ${api_key_env}"),
                "Content-Type: application/json"
            ],
            "body": {
                "model": model,
                "messages": [
                    { "role": "user", "content": question }
                ],
                "metadata": {
                    "session_id": "first-run-smoke",
                    "purpose": "operator-smoke"
                }
            }
        }
    })
}

fn last_four(value: &str) -> String {
    value
        .chars()
        .rev()
        .take(4)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>()
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
                let worker_count = statuses.len();
                let worker_control = Arc::new(AsyncMutex::new(supervisor));
                emit(
                    as_json,
                    &json!({
                        "status": "ready",
                        "bind": format!("{}:{}", cfg.server.host, cfg.server.port),
                        "backend": "llama-server-subprocess",
                        "workers": worker_count,
                        "native_engines": 0
                    }),
                )?;
                rs_llmctl::server::serve_with_storage_worker_control_and_shutdown(
                    cfg,
                    storage,
                    Some(worker_control),
                    rs_llmctl::server::shutdown_signal(),
                )
                .await
            } else {
                let engines = load_native_engines_from_config(&cfg)?;
                emit(
                    as_json,
                    &json!({
                        "status": if engines.is_empty() { "no_models" } else { "ready" },
                        "bind": format!("{}:{}", cfg.server.host, cfg.server.port),
                        "backend": "candle-native",
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
                family: Some(args.family.clone()),
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
        ModelCommand::Status(args) => {
            let model = cfg
                .models
                .iter()
                .find(|model| model.alias == args.alias)
                .with_context(|| format!("model alias '{}' is not configured", args.alias))?;
            let status = if model.weight == 0 {
                "stopped"
            } else {
                "running"
            };
            let readiness = model
                .family
                .as_deref()
                .filter(|family| family.eq_ignore_ascii_case("gemma4"))
                .map(|_| {
                    rs_llmctl::readiness::read_state(&rs_llmctl::readiness::evidence_path(
                        &cfg.storage.model_dir,
                        &model.alias,
                    ))
                });

            emit(
                as_json,
                &json!({
                    "status": status,
                    "alias": &model.alias,
                    "weight": model.weight,
                    "restart_required": false,
                    "restart_hint": default_restart_hint(),
                    "runtime_backend": &cfg.runtime.backend,
                    "one_binary": true,
                    "entrypoint": one_binary_entrypoint(),
                    "readiness": readiness,
                    "model": model,
                }),
            )
        }
        ModelCommand::Start(args) => {
            let model = cfg
                .models
                .iter_mut()
                .find(|model| model.alias == args.alias)
                .with_context(|| format!("model alias '{}' is not configured", args.alias))?;
            let previous_weight = model.weight;
            let weight = args.weight.unwrap_or_else(|| previous_weight.max(1));
            if args.dry_run {
                return emit(
                    as_json,
                    &json!({
                        "status": "planned",
                        "action": "start",
                        "alias": &model.alias,
                        "previous_weight": previous_weight,
                        "weight": weight,
                        "restart_required": true,
                        "restart_hint": default_restart_hint(),
                        "runtime_backend": &cfg.runtime.backend,
                        "one_binary": true,
                        "entrypoint": one_binary_entrypoint(),
                        "model": model,
                    }),
                );
            }
            model.weight = weight;
            let model_config = model.clone();
            persist_models(path, &cfg).await?;

            emit(
                as_json,
                &json!({
                    "status": "started",
                    "action": "start",
                    "alias": &model_config.alias,
                    "previous_weight": previous_weight,
                    "weight": model_config.weight,
                    "restart_required": true,
                    "restart_hint": default_restart_hint(),
                    "runtime_backend": &cfg.runtime.backend,
                    "one_binary": true,
                    "entrypoint": one_binary_entrypoint(),
                    "model": &model_config,
                    "models": &cfg.models,
                }),
            )
        }
        ModelCommand::Stop(args) => {
            let model = cfg
                .models
                .iter_mut()
                .find(|model| model.alias == args.alias)
                .with_context(|| format!("model alias '{}' is not configured", args.alias))?;
            let previous_weight = model.weight;
            if args.dry_run {
                return emit(
                    as_json,
                    &json!({
                        "status": "planned",
                        "action": "stop",
                        "alias": &model.alias,
                        "previous_weight": previous_weight,
                        "weight": 0,
                        "restart_required": true,
                        "restart_hint": default_restart_hint(),
                        "runtime_backend": &cfg.runtime.backend,
                        "one_binary": true,
                        "entrypoint": one_binary_entrypoint(),
                        "model": model,
                    }),
                );
            }
            model.weight = 0;
            let model_config = model.clone();
            persist_models(path, &cfg).await?;

            emit(
                as_json,
                &json!({
                    "status": "stopped",
                    "action": "stop",
                    "alias": &model_config.alias,
                    "previous_weight": previous_weight,
                    "weight": model_config.weight,
                    "restart_required": true,
                    "restart_hint": default_restart_hint(),
                    "runtime_backend": &cfg.runtime.backend,
                    "one_binary": true,
                    "entrypoint": one_binary_entrypoint(),
                    "model": &model_config,
                    "models": &cfg.models,
                }),
            )
        }
        ModelCommand::Update(args) => {
            replace_model(path, &mut cfg, args, "update", "updated", as_json).await
        }
        ModelCommand::Upgrade(args) => {
            replace_model(path, &mut cfg, args, "upgrade", "upgraded", as_json).await
        }
        ModelCommand::Downgrade(args) => {
            replace_model(path, &mut cfg, args, "downgrade", "downgraded", as_json).await
        }
        ModelCommand::Drift(args) => {
            let storage = init_storage(&cfg.storage).await?;
            record_latency_drift_observations(&storage, args.hours).await?;
            report_observations(&storage, "drift", args.hours, as_json).await
        }
        ModelCommand::ImportManifest(args) => {
            create_storage_dirs(&cfg.storage).await?;
            let installed = model::import_offline_manifest(&args.manifest).await?;

            for model in &installed {
                upsert_model(&mut cfg.models, model.config.clone());
            }
            persist_models(path, &cfg).await?;

            emit(
                as_json,
                &json!({ "status": "imported", "imported": installed, "models": cfg.models }),
            )
        }
        ModelCommand::Inventory => {
            let storage = init_storage(&cfg.storage).await?;
            let inventory = model_inventory(&cfg, &storage).await?;
            emit(as_json, &inventory)
        }
        ModelCommand::List => emit(as_json, &cfg.models),
        ModelCommand::Profile { command } => {
            model_profile_command(&cfg.storage.model_dir, command, as_json).await
        }
    }
}

async fn model_profile_command(
    model_dir: &Path,
    command: ModelProfileCommand,
    as_json: bool,
) -> Result<()> {
    match command {
        ModelProfileCommand::List => emit(as_json, &rs_llmctl::profiles::list_profiles(model_dir)?),
        ModelProfileCommand::Inspect(args) => emit(
            as_json,
            &rs_llmctl::profiles::read_profile(model_dir, &args.alias)?,
        ),
        ModelProfileCommand::ImportLocal(args) => {
            let profile =
                rs_llmctl::profiles::import_local_candidate(&args.path, &args.alias).await?;
            let path = rs_llmctl::profiles::write_profile(model_dir, &profile)?;
            emit(
                as_json,
                &json!({"status": "candidate", "path": path, "profile": profile}),
            )
        }
        ModelProfileCommand::ImportCatalog(args) => {
            let profile = rs_llmctl::profiles::import_catalog_candidate(&args.id)?;
            let path = rs_llmctl::profiles::write_profile(model_dir, &profile)?;
            emit(
                as_json,
                &json!({"status": "candidate", "path": path, "profile": profile}),
            )
        }
        ModelProfileCommand::Qualify(args) => {
            let profile = rs_llmctl::profiles::read_profile(model_dir, &args.alias)?;
            let (profile, policy) =
                rs_llmctl::profiles::qualify_profile(profile, args.available_vram_bytes);
            let path = rs_llmctl::profiles::write_profile(model_dir, &profile)?;
            emit(
                as_json,
                &json!({"status": profile.qualification, "path": path, "policy": policy, "profile": profile}),
            )
        }
        ModelProfileCommand::Quarantine(args) => {
            let mut profile = rs_llmctl::profiles::read_profile(model_dir, &args.alias)?;
            profile.qualification = rs_llmctl::profiles::QualificationStatus::Quarantined;
            profile.quarantine_reason = Some(args.reason);
            let path = rs_llmctl::profiles::write_profile(model_dir, &profile)?;
            emit(
                as_json,
                &json!({"status": "quarantined", "path": path, "profile": profile}),
            )
        }
        ModelProfileCommand::Remove(args) => {
            rs_llmctl::profiles::remove_profile(model_dir, &args.alias)?;
            emit(as_json, &json!({"status": "removed", "alias": args.alias}))
        }
        ModelProfileCommand::Adapters => emit(as_json, &rs_llmctl::profiles::backend_catalog()),
    }
}

async fn service_command(command: ServiceCommand, as_json: bool) -> Result<()> {
    let (action, args) = match command {
        ServiceCommand::Status(args) => (ServiceLifecycleAction::Status, args),
        ServiceCommand::Start(args) => (ServiceLifecycleAction::Start, args),
        ServiceCommand::Stop(args) => (ServiceLifecycleAction::Stop, args),
        ServiceCommand::Restart(args) => (ServiceLifecycleAction::Restart, args),
        ServiceCommand::Upgrade(args) => (ServiceLifecycleAction::Upgrade, args),
        ServiceCommand::Downgrade(args) => (ServiceLifecycleAction::Downgrade, args),
    };
    if args.service_name.trim().is_empty() {
        bail!("--service-name must not be empty");
    }
    let plan = plan_service_lifecycle(action, &args);
    if args.dry_run {
        return emit(as_json, &plan);
    }

    let result = execute_service_lifecycle(plan).await?;
    emit(as_json, &result)
}

async fn runtime_command(path: &Path, command: RuntimeCommand, as_json: bool) -> Result<()> {
    let cfg = load_config(path).await?;
    match command {
        RuntimeCommand::Status => {
            let status = runtime::status_from_config(&cfg);
            let mut attributes = BTreeMap::new();
            attributes.insert("runtime.backend".to_string(), json!(status.backend));
            attributes.insert("runtime.engine".to_string(), json!(status.engine));
            attributes.insert("runtime.primary".to_string(), json!(status.primary));
            attributes.insert("runtime.implemented".to_string(), json!(status.implemented));
            attributes.insert(
                "runtime.resource.budget_fraction".to_string(),
                json!(status.resource_policy.budget_fraction),
            );
            emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
                TelemetrySignal::Log,
                TelemetryEventName::NativeRuntimeStatus,
                Utc::now(),
                attributes,
            ));
            emit(as_json, &status)
        }
        RuntimeCommand::Heartbeat => {
            let heartbeat = native::heartbeat_from_config(&cfg);
            emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
                TelemetrySignal::Metric,
                TelemetryEventName::RuntimeHeartbeat,
                Utc::now(),
                heartbeat.safe_telemetry_attributes(),
            ));
            emit(as_json, &heartbeat)
        }
        RuntimeCommand::Placement => emit(as_json, &native::placement_plan_from_config(&cfg)),
        RuntimeCommand::Route(args) => {
            let placement = native::placement_plan_from_config(&cfg);
            native::validate_placement_plan(&placement)?;
            let selection = match (args.model.as_deref(), args.role.as_deref()) {
                (Some(model), None) => native::route_selection_for_model(&placement, model)?,
                (None, Some(role)) => native::route_selection_for_role(&placement, role)?,
                (None, None) => bail!("runtime route requires --model or --role"),
                (Some(_), Some(_)) => unreachable!("clap enforces conflicts_with"),
            };
            emit(as_json, &selection)
        }
        RuntimeCommand::AmdQualification(args) => emit(
            as_json,
            &rs_llmctl::amd::qualification_report_with_evidence(
                args.preview,
                args.arch_opt_in,
                args.evidence.as_deref(),
            ),
        ),
        RuntimeCommand::Gemma4Readiness(args) => {
            let evidence =
                rs_llmctl::readiness::run_gemma4_readiness(&args.model_path, &args.alias).await?;
            let evidence_path = args.evidence_output.unwrap_or_else(|| {
                rs_llmctl::readiness::evidence_path(&cfg.storage.model_dir, &args.alias)
            });
            rs_llmctl::readiness::write_evidence(&evidence_path, &evidence)?;
            let result = rs_llmctl::readiness::ensure_qualified(&evidence);
            emit(as_json, &evidence)?;
            result
        }
        RuntimeCommand::ValidationPlan(args) => emit(
            as_json,
            &runtime::native_validation_plan(
                &cfg,
                runtime::NativeRuntimeValidationOptions {
                    soak_minutes: args.soak_minutes,
                    streaming_concurrency: args.streaming_concurrency,
                    rotation_keys: args.rotation_keys,
                    quota_concurrency: args.quota_concurrency,
                },
            ),
        ),
        RuntimeCommand::ValidationRun(args) => {
            let evidence = runtime_validation_run(&cfg).await;
            if let Some(path) = args.evidence_output.as_ref() {
                write_json_file(path, &evidence).await?;
            }
            let failed = evidence["checks"]
                .as_array()
                .map(|checks| checks.iter().any(|check| check["status"] != "ok"))
                .unwrap_or(true);
            emit(as_json, &evidence)?;
            if failed {
                bail!("native runtime validation failed; inspect validation-run evidence");
            }
            Ok(())
        }
        RuntimeCommand::Validate => {
            let placement = native::placement_plan_from_config(&cfg);
            native::validate_placement_plan(&placement)?;
            emit(
                as_json,
                &json!({
                    "status": "ok",
                    "routing_mode": placement.routing_mode,
                    "nodes": placement.nodes.len(),
                    "unassigned_models": placement.unassigned_models,
                }),
            )
        }
    }
}

async fn runtime_validation_run(cfg: &Config) -> Value {
    let mut checks = Vec::new();
    let placement = native::placement_plan_from_config(cfg);
    checks.push(validation_check(
        "placement",
        native::validate_placement_plan(&placement).map(|_| ()),
    ));

    let runnable_models = cfg.models.iter().filter(|model| model.weight > 0);
    let mut runnable_count = 0usize;
    for model in runnable_models {
        runnable_count += 1;
        let result = configured_candle_family(model)
            .and_then(|family| native::validate_candle_model_artifacts(family, model).map(|_| ()));
        checks.push(validation_check(
            &format!("artifact:{}", model.alias),
            result,
        ));
    }
    if runnable_count == 0 {
        checks.push(json!({
            "name": "artifacts",
            "status": "failed",
            "error": "no positive-weight native models are configured",
        }));
    }

    let failed = checks.iter().any(|check| check["status"] != "ok");
    json!({
        "status": if failed { "failed" } else { "ok" },
        "runtime_backend": rs_llmctl::runtime::RuntimeBackend::CandleNative,
        "executable": true,
        "models_checked": runnable_count,
        "checks": checks,
    })
}

fn validation_check(name: &str, result: Result<()>) -> Value {
    match result {
        Ok(()) => json!({ "name": name, "status": "ok" }),
        Err(err) => json!({ "name": name, "status": "failed", "error": err.to_string() }),
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
        SwapCommand::Plan(args) => {
            let active = WorkerId::new(args.active);
            let replacement = WorkerId::new(args.replacement);
            let plan = match cfg.mode {
                Mode::ColdSwap => SwapPlan::cold(active, replacement),
                Mode::HotSwap => SwapPlan::hot(active, replacement),
                Mode::Single | Mode::Weighted | Mode::Fallback => {
                    bail!(
                        "swap plan is only supported for cold-swap or hot-swap modes; current mode is {}",
                        mode_name(&cfg.mode)
                    );
                }
            };
            emit(as_json, &plan)
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
            let policy_summary = quota::summarize_quota_policies(&cfg.quotas);
            emit(
                as_json,
                &json!({
                    "hours": args.hours,
                    "from": from,
                    "to": to,
                    "generated_at": Utc::now(),
                    "policies": cfg.quotas,
                    "policy_summary": policy_summary,
                    "decisions": decisions,
                    "usage_summary": usage_summary
                }),
            )
        }
        QuotaCommand::Export => emit(
            as_json,
            &json!({
                "status": "exported",
                "format": "json",
                "count": cfg.quotas.len(),
                "quotas": cfg.quotas
            }),
        ),
        QuotaCommand::Import(args) => {
            let imported = load_quota_policy(&args.path).await?;
            validate_quota_policies(&imported.quotas)?;
            cfg.quotas = imported.quotas;
            config::save(path, &cfg).await?;
            emit(
                as_json,
                &json!({
                    "status": "imported",
                    "format": imported.format,
                    "path": args.path,
                    "count": cfg.quotas.len(),
                    "quotas": cfg.quotas
                }),
            )
        }
        QuotaCommand::List => emit(as_json, &cfg.quotas),
    }
}

async fn security_command(path: &Path, command: SecurityCommand, as_json: bool) -> Result<()> {
    match command {
        SecurityCommand::Check => {
            let cfg = load_config(path).await?;
            config::validate_production_security(&cfg)?;
            emit(
                as_json,
                &json!({
                    "status": "ok",
                    "production": cfg.security.production,
                    "require_auth": cfg.security.require_auth,
                    "bind_external": cfg.security.bind_external,
                    "host": cfg.server.host,
                    "tls_termination": cfg.security.tls_termination,
                    "api_keys": cfg.security.api_keys.len()
                }),
            )
        }
        SecurityCommand::GenerateKey(args) => {
            let secret = generate_api_key_secret(&args.prefix);
            let sha256 = hex::encode(Sha256::digest(secret.as_bytes()));
            let last_four = secret
                .chars()
                .rev()
                .take(4)
                .collect::<String>()
                .chars()
                .rev()
                .collect::<String>();
            if let Some(output) = args.output.as_ref() {
                write_secret_file(output, &secret).await?;
            }
            emit(
                as_json,
                &json!({
                    "status": "generated",
                    "secret": if args.output.is_none() { Some(secret.as_str()) } else { None },
                    "secret_written": args.output.as_ref().map(|path| path.display().to_string()),
                    "sha256": sha256,
                    "last_four": last_four,
                    "metadata": {
                        "purpose": "api-key",
                        "algorithm": "sha256",
                        "encoding": "hex",
                        "store_secret_once": true,
                        "next": "llmctl security add-key --id <id> --sha256 <sha256> --subject <subject> --team <team> --scope chat"
                    }
                }),
            )
        }
        SecurityCommand::HashKey(args) => {
            let (secret, input) = read_api_key_secret(args).await?;
            rs_llmctl::security::validate_api_secret_material(&secret)?;
            let sha256 = hex::encode(Sha256::digest(secret.as_bytes()));
            emit(
                as_json,
                &json!({
                    "sha256": sha256,
                    "metadata": {
                        "algorithm": "sha256",
                        "encoding": "hex",
                        "input": input,
                        "purpose": "api-key"
                    }
                }),
            )
        }
        SecurityCommand::ListKeys => {
            let cfg = load_config(path).await?;
            emit(as_json, &api_key_inventory_report(&cfg))
        }
        SecurityCommand::RotateKey(args) => {
            let mut cfg = load_config(path).await?;
            let sha256 = args.sha256.to_ascii_lowercase();
            validate_sha256_digest(&sha256)?;
            let Some(position) = cfg
                .security
                .api_keys
                .iter()
                .position(|key| key.id == args.id)
            else {
                bail!("api key id `{}` was not found", args.id);
            };
            if let Some(new_id) = args.new_id.as_ref() {
                validate_api_key_id(new_id)?;
                if cfg.security.api_keys.iter().any(|key| key.id == *new_id) {
                    bail!("api key id `{new_id}` already exists");
                }
                let now = Utc::now();
                let mut retiring = cfg.security.api_keys[position].clone();
                retiring.status = "retiring".to_string();
                retiring.rotated_at = Some(now);
                cfg.security.api_keys[position] = retiring.clone();
                let replacement = ApiKeyConfig {
                    id: new_id.clone(),
                    sha256,
                    subject: retiring.subject,
                    team: retiring.team,
                    scopes: retiring.scopes,
                    created_at: Some(now),
                    expires_at: args.expires_at,
                    rotated_at: None,
                    owner: retiring.owner,
                    purpose: retiring.purpose,
                    last_four: args.last_four,
                    fingerprint: None,
                    status: "active".to_string(),
                };
                cfg.security.api_keys.push(replacement);
                config::save(path, &cfg).await?;
                record_security_key_event(
                    &cfg,
                    "security.api_key.rotate",
                    new_id,
                    "rotated",
                    json!({
                        "api_key_id": args.id,
                        "new_api_key_id": new_id,
                        "mode": "overlap",
                        "reason": args.reason,
                        "old_status": "retiring"
                    }),
                )
                .await?;
                emit(
                    as_json,
                    &json!({
                        "status": "rotated",
                        "mode": "overlap",
                        "retiring_id": args.id,
                        "active_id": new_id,
                        "restart_required": true,
                        "restart_hint": default_restart_hint()
                    }),
                )?;
                return Ok(());
            }
            if !args.replace {
                bail!(
                    "rotate-key requires --new-id for overlap rotation or --replace for in-place replacement"
                );
            }
            let key = &mut cfg.security.api_keys[position];
            key.sha256 = sha256;
            key.rotated_at = Some(Utc::now());
            key.expires_at = args.expires_at.or(key.expires_at);
            key.last_four = args.last_four.or_else(|| key.last_four.clone());
            key.status = "active".to_string();
            config::save(path, &cfg).await?;
            record_security_key_event(
                &cfg,
                "security.api_key.rotate",
                &args.id,
                "rotated",
                json!({
                    "api_key_id": args.id,
                    "mode": "replace",
                    "reason": args.reason
                }),
            )
            .await?;
            emit(
                as_json,
                &json!({
                    "status": "rotated",
                    "mode": "replace",
                    "id": args.id,
                    "sha256_present": true,
                    "restart_required": true,
                    "restart_hint": default_restart_hint()
                }),
            )
        }
        SecurityCommand::RevokeKey(args) => {
            let mut cfg = load_config(path).await?;
            let Some(position) = cfg
                .security
                .api_keys
                .iter()
                .position(|key| key.id == args.id)
            else {
                bail!("api key id `{}` was not found", args.id);
            };
            let removed = cfg.security.api_keys.remove(position);
            config::save(path, &cfg).await?;
            record_security_key_event(
                &cfg,
                "security.api_key.revoke",
                &args.id,
                "revoked",
                json!({
                    "api_key_id": args.id,
                    "reason": args.reason,
                    "removed": true,
                    "remove_requested": args.remove,
                    "subject": removed.subject,
                    "team": removed.team,
                    "owner": removed.owner,
                    "purpose": removed.purpose,
                    "previous_status": removed.status
                }),
            )
            .await?;
            emit(
                as_json,
                &json!({
                    "status": "revoked",
                    "id": args.id,
                    "api_keys": cfg.security.api_keys.len(),
                    "restart_required": true,
                    "restart_hint": default_restart_hint()
                }),
            )
        }
        SecurityCommand::KeyUsage(args) => {
            let cfg = load_config(path).await?;
            let storage = init_storage(&cfg.storage).await?;
            let report = api_key_usage_report(&storage, args.id.as_deref(), args.hours).await?;
            emit(as_json, &report)
        }
        SecurityCommand::AddKey(args) => {
            let mut cfg = load_config(path).await?;
            let sha256 = args.sha256.to_ascii_lowercase();
            validate_add_key_args(&args.id, &sha256, &args.subject, &args.team)?;
            let key = ApiKeyConfig {
                id: args.id,
                sha256,
                subject: args.subject,
                team: args.team,
                scopes: args.scopes,
                created_at: Some(Utc::now()),
                expires_at: args.expires_at,
                rotated_at: None,
                owner: args.owner,
                purpose: args.purpose,
                last_four: args.last_four,
                fingerprint: None,
                status: "active".to_string(),
            };
            let action = upsert_api_key(&mut cfg.security.api_keys, key.clone());
            config::save(path, &cfg).await?;
            emit(
                as_json,
                &json!({
                    "status": "saved",
                    "action": action,
                    "api_keys": cfg.security.api_keys.len(),
                    "key": {
                        "id": key.id,
                        "subject": key.subject,
                        "team": key.team,
                        "scopes": key.scopes,
                        "owner": key.owner,
                        "purpose": key.purpose,
                        "created_at": key.created_at,
                        "expires_at": key.expires_at,
                        "last_four": key.last_four,
                        "status": key.status,
                        "sha256_present": true
                    }
                }),
            )
        }
        SecurityCommand::AuditConfig(args) => {
            let cfg = load_config(path).await?;
            let report = audit_config_report(path, &cfg, args.systemd_unit.as_deref()).await?;
            emit(as_json, &report)
        }
    }
}

fn generate_api_key_secret(prefix: &str) -> String {
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

async fn write_secret_file(path: &Path, secret: &str) -> Result<()> {
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

fn api_key_inventory_report(cfg: &Config) -> Value {
    json!({
        "status": "ok",
        "require_auth": cfg.security.require_auth,
        "api_keys": cfg.security.api_keys.iter().map(|key| {
            json!({
                "id": key.id,
                "subject": key.subject,
                "team": key.team,
                "scopes": key.scopes,
                "owner": key.owner,
                "purpose": key.purpose,
                "created_at": key.created_at,
                "expires_at": key.expires_at,
                "rotated_at": key.rotated_at,
                "last_four": key.last_four,
                "fingerprint": key.fingerprint,
                "status": key.status,
                "sha256_present": !key.sha256.trim().is_empty()
            })
        }).collect::<Vec<_>>()
    })
}

async fn api_key_usage_report(storage: &Storage, id: Option<&str>, hours: i64) -> Result<Value> {
    let now = Utc::now();
    let from = now - Duration::hours(hours.max(1));
    let key_usage = storage.api_key_usage_between(from, now).await?;
    let mut by_key: BTreeMap<String, ApiKeyUsageSummary> = BTreeMap::new();
    for record in key_usage {
        if id.is_some_and(|expected| expected != record.api_key_id) {
            continue;
        }
        let summary = by_key.entry(record.api_key_id.clone()).or_default();
        summary.request_count = summary.request_count.saturating_add(1);
        if record.audit_outcome != "ok" && record.audit_outcome != "allowed" {
            summary.error_count = summary.error_count.saturating_add(1);
        }
        summary.last_seen = Some(
            summary
                .last_seen
                .map_or(record.usage_at, |last| last.max(record.usage_at)),
        );
        summary.input_tokens = summary.input_tokens.saturating_add(record.input_tokens);
        summary.output_tokens = summary.output_tokens.saturating_add(record.output_tokens);
        summary.total_tokens = summary.total_tokens.saturating_add(record.total_tokens);
        summary.latency_ms = summary.latency_ms.saturating_add(record.latency_ms);
        summary.actors.insert(record.actor);
        summary.teams.insert(record.team);
        summary.models.insert(record.model);
        summary.statuses.insert(record.status);
    }

    let audit_events = storage.audit_events_between(from, now).await?;
    for event in audit_events {
        let Some(key_id) = event.detail_json.get("api_key_id").and_then(Value::as_str) else {
            continue;
        };
        if id.is_some_and(|expected| expected != key_id) {
            continue;
        }
        let summary = by_key.entry(key_id.to_string()).or_default();
        summary.audit_event_count = summary.audit_event_count.saturating_add(1);
        summary.actions.insert(event.action);
        summary.resources.insert(event.resource);
        summary.actors.insert(event.actor);
        summary.teams.insert(event.team);
        summary.last_seen = Some(
            summary
                .last_seen
                .map_or(event.at, |last| last.max(event.at)),
        );
    }

    Ok(json!({
        "status": "ok",
        "from": from,
        "to": now,
        "filter": { "id": id },
        "keys": by_key.into_iter().map(|(key_id, summary)| {
            json!({
                "id": key_id,
                "request_count": summary.request_count,
                "audit_event_count": summary.audit_event_count,
                "error_count": summary.error_count,
                "input_tokens": summary.input_tokens,
                "output_tokens": summary.output_tokens,
                "total_tokens": summary.total_tokens,
                "latency_ms": summary.latency_ms,
                "last_seen": summary.last_seen,
                "actors": summary.actors.into_iter().collect::<Vec<_>>(),
                "teams": summary.teams.into_iter().collect::<Vec<_>>(),
                "models": summary.models.into_iter().collect::<Vec<_>>(),
                "statuses": summary.statuses.into_iter().collect::<Vec<_>>(),
                "actions": summary.actions.into_iter().collect::<Vec<_>>(),
                "resources": summary.resources.into_iter().collect::<Vec<_>>()
            })
        }).collect::<Vec<_>>()
    }))
}

#[derive(Debug, Default)]
struct ApiKeyUsageSummary {
    request_count: u64,
    audit_event_count: u64,
    error_count: u64,
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
    latency_ms: u64,
    last_seen: Option<chrono::DateTime<Utc>>,
    actors: std::collections::BTreeSet<String>,
    teams: std::collections::BTreeSet<String>,
    models: std::collections::BTreeSet<String>,
    statuses: std::collections::BTreeSet<String>,
    actions: std::collections::BTreeSet<String>,
    resources: std::collections::BTreeSet<String>,
}

async fn read_api_key_secret(args: SecurityHashKeyArgs) -> Result<(String, &'static str)> {
    if let Some(name) = args.env {
        let secret =
            std::env::var(&name).with_context(|| format!("read secret from env {name}"))?;
        return Ok((secret, "env"));
    }

    if args.stdin {
        let mut secret = String::new();
        std::io::stdin()
            .read_to_string(&mut secret)
            .context("read secret from stdin")?;
        return Ok((secret.trim_end_matches(['\r', '\n']).to_string(), "stdin"));
    }

    bail!(
        "security hash-key requires --stdin or --env NAME so secrets are not exposed in process arguments"
    )
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

async fn record_latency_drift_observations(storage: &Storage, hours: i64) -> Result<usize> {
    let hours = hours.max(1);
    let now = Utc::now();
    let current_from = now - Duration::hours(hours);
    let previous_from = current_from - Duration::hours(hours);
    let current = storage.usage_events_between(current_from, now).await?;
    let previous = storage
        .usage_events_between(previous_from, current_from)
        .await?;
    let current_avg = average_latency_by_model(&current);
    let previous_avg = average_latency_by_model(&previous);
    let mut inserted = 0usize;
    for (model, current_ms) in current_avg {
        let Some(previous_ms) = previous_avg.get(&model).copied() else {
            continue;
        };
        if previous_ms <= 0.0 {
            continue;
        }
        let ratio = (current_ms - previous_ms) / previous_ms;
        if ratio.abs() >= 0.25 {
            let event = ObservationEvent {
                id: Uuid::new_v4(),
                request_id: None,
                at: now,
                kind: "model.drift.latency".to_string(),
                model: model.clone(),
                source: "llmctl-model-drift".to_string(),
                value: ratio,
                unit: "ratio".to_string(),
                attributes_json: json!({
                    "current_avg_latency_ms": current_ms,
                    "previous_avg_latency_ms": previous_ms,
                    "window_hours": hours
                }),
            };
            storage.insert_observation_event(&event).await?;
            emit_runtime_telemetry(&RuntimeTelemetryEvent::new(
                TelemetrySignal::Metric,
                TelemetryEventName::DriftObservation,
                Utc::now(),
                BTreeMap::from([
                    ("llmctl.model".to_string(), json!(model)),
                    ("llmctl.drift.kind".to_string(), json!("latency")),
                    ("llmctl.drift.value".to_string(), json!(ratio)),
                ]),
            ));
            inserted += 1;
        }
    }
    Ok(inserted)
}

fn average_latency_by_model(events: &[rs_llmctl::audit::UsageEvent]) -> BTreeMap<String, f64> {
    let mut totals = BTreeMap::<String, (u64, u64)>::new();
    for event in events {
        let entry = totals.entry(event.model.clone()).or_default();
        entry.0 = entry.0.saturating_add(event.latency_ms);
        entry.1 = entry.1.saturating_add(1);
    }
    totals
        .into_iter()
        .filter_map(|(model, (latency, count))| {
            (count > 0).then_some((model, latency as f64 / count as f64))
        })
        .collect()
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
                    if args.write {
                        let path =
                            write_audit_report(&cfg, year, month, "envelope", &report).await?;
                        return emit(
                            as_json,
                            &json!({"status": "written", "path_redacted": redact_display_path(&path)}),
                        );
                    }
                    emit(as_json, &report)
                } else {
                    let report = reporting::monthly_audit_report(&storage, year, month).await?;
                    if args.write {
                        let path = write_audit_report(&cfg, year, month, "report", &report).await?;
                        return emit(
                            as_json,
                            &json!({"status": "written", "path_redacted": redact_display_path(&path)}),
                        );
                    }
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
        AuditCommand::Retention { command } => match command {
            AuditRetentionCommand::Plan(args) => {
                let report = audit_retention_plan(&cfg, &storage).await?;
                if args.envelope {
                    let envelope =
                        reporting::report_envelope(reporting::ReportKind::RetentionPlan, report)?;
                    emit(as_json, &envelope)
                } else {
                    emit(as_json, &report)
                }
            }
            AuditRetentionCommand::Apply(args) => {
                anyhow::ensure!(
                    args.yes,
                    "audit retention apply requires --yes; run audit retention plan first"
                );
                let report = audit_retention_apply(&cfg, &storage).await?;
                if args.envelope {
                    let envelope =
                        reporting::report_envelope(reporting::ReportKind::RetentionPlan, report)?;
                    emit(as_json, &envelope)
                } else {
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

async fn audit_retention_plan(cfg: &Config, storage: &Storage) -> Result<serde_json::Value> {
    let generated_at = Utc::now();
    let cutoff = generated_at - Duration::days(i64::from(cfg.audit.retention_days));
    let counts = storage.audit_retention_counts(cutoff).await?;

    Ok(json!({
        "status": "planned",
        "operation": "audit_retention",
        "dry_run": true,
        "deletes": false,
        "generated_at": generated_at,
        "retention": {
            "days": cfg.audit.retention_days,
            "cutoff": cutoff,
            "report_directory": cfg.audit.report_directory,
            "report_formats": cfg.audit.report_formats,
            "monthly_reports": cfg.audit.monthly_reports
        },
        "counts": counts
    }))
}

async fn audit_retention_apply(cfg: &Config, storage: &Storage) -> Result<serde_json::Value> {
    let generated_at = Utc::now();
    let cutoff = generated_at - Duration::days(i64::from(cfg.audit.retention_days));
    let before = storage.audit_retention_counts(cutoff).await?;
    let deleted = storage.delete_audit_events_before(cutoff).await?;
    let after = storage.audit_retention_counts(cutoff).await?;

    Ok(json!({
        "status": "applied",
        "operation": "audit_retention",
        "dry_run": false,
        "deletes": true,
        "generated_at": generated_at,
        "retention": {
            "days": cfg.audit.retention_days,
            "cutoff": cutoff,
            "report_directory": cfg.audit.report_directory,
            "report_formats": cfg.audit.report_formats,
            "monthly_reports": cfg.audit.monthly_reports
        },
        "deleted": deleted,
        "counts_before": before,
        "counts_after": after
    }))
}

async fn usage_command(path: &Path, command: UsageCommand, as_json: bool) -> Result<()> {
    let cfg = load_config(path).await?;
    let storage = init_storage(&cfg.storage).await?;
    match command {
        UsageCommand::Report(args) => report_usage(&storage, args.hours, as_json).await,
        UsageCommand::Chargeback(args) => report_chargeback(&storage, args, as_json).await,
    }
}

async fn data_command(path: &Path, command: DataCommand, as_json: bool) -> Result<()> {
    match command {
        DataCommand::Export(args) => {
            let cfg = load_config(path).await?;
            let storage = init_storage(&cfg.storage).await?;
            let (from, to) = window(args.hours);
            if args.envelope {
                anyhow::ensure!(
                    matches!(args.dataset, DataDataset::All)
                        && matches!(args.format, DataExportFormat::Json),
                    "data export --envelope currently wraps the canonical all/json export"
                );
                let report = reporting::data_export_envelope_limited(
                    &storage,
                    from,
                    to,
                    Some(args.max_rows),
                )
                .await?;
                emit(as_json, &report)
            } else {
                let report = if matches!(args.dataset, DataDataset::All) {
                    reporting::data_export_limited(&storage, from, to, Some(args.max_rows)).await?
                } else {
                    let dataset = args
                        .dataset
                        .contract_kind()
                        .context("data export requires a concrete dataset")?;
                    reporting::data_export_dataset_limited(
                        &storage,
                        from,
                        to,
                        dataset,
                        Some(args.max_rows),
                    )
                    .await?
                };
                let output = format_data_export(
                    report,
                    args.dataset,
                    args.format,
                    args.output.as_deref(),
                    args.max_rows,
                )?;
                emit(as_json, &output)
            }
        }
        DataCommand::Contracts(args) => {
            let contracts = if let Some(dataset) = args.dataset {
                vec![contracts::contract_for(dataset.into())]
            } else {
                contracts::all_contracts()
            };
            emit(
                as_json,
                &json!({
                    "schema_version": contracts::CONTRACT_SCHEMA_VERSION,
                    "contracts": contracts
                }),
            )
        }
        DataCommand::VerifyEnvelope(args) => {
            let envelope_bytes = fs::read(&args.path)
                .await
                .with_context(|| format!("read {}", args.path.display()))?;
            let envelope: serde_json::Value = serde_json::from_slice(&envelope_bytes)
                .with_context(|| format!("parse {}", args.path.display()))?;
            let verification = reporting::verify_envelope_value(&envelope)?;
            let mut output = serde_json::to_value(verification)?;
            if let Some(object) = output.as_object_mut() {
                object.insert(
                    "artifact".to_string(),
                    serde_json::Value::String(
                        args.path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("envelope.json")
                            .to_string(),
                    ),
                );
                object.insert(
                    "path_redacted".to_string(),
                    serde_json::Value::String(redact_display_path(&args.path)),
                );
            }
            emit(as_json, &output)
        }
    }
}

async fn write_audit_report<T: Serialize>(
    cfg: &Config,
    year: i32,
    month: u32,
    suffix: &str,
    report: &T,
) -> Result<PathBuf> {
    let dir = cfg
        .audit
        .report_directory
        .as_ref()
        .context("audit report monthly --write requires audit.report-directory")?;
    fs::create_dir_all(dir)
        .await
        .with_context(|| format!("create audit report directory {}", dir.display()))?;
    #[cfg(unix)]
    fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
        .await
        .with_context(|| format!("secure audit report directory {}", dir.display()))?;
    let path = dir.join(format!("monthly-audit-{year:04}-{month:02}-{suffix}.json"));
    let body = serde_json::to_vec_pretty(report)?;
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options
        .open(&path)
        .await
        .with_context(|| format!("write audit report {}", path.display()))?;
    file.write_all(&body)
        .await
        .with_context(|| format!("write audit report {}", path.display()))?;
    Ok(path)
}

fn format_data_export(
    report: reporting::DataExport,
    dataset: DataDataset,
    format: DataExportFormat,
    output: Option<&Path>,
    max_rows: usize,
) -> Result<serde_json::Value> {
    let rows = dataset_rows(&report, dataset)?;
    if rows.len() > max_rows {
        bail!(
            "data export for dataset `{}` produced {} rows, exceeding --max-rows {}; narrow --hours or raise --max-rows",
            dataset.as_str(),
            rows.len(),
            max_rows
        );
    }
    let dataset_name = dataset.as_str();
    let contract = dataset.contract_kind().map(contracts::contract_for);

    match format {
        DataExportFormat::Json if matches!(dataset, DataDataset::All) => {
            Ok(serde_json::to_value(report)?)
        }
        DataExportFormat::Json => Ok(json!({
            "format": "json",
            "schema_version": contracts::CONTRACT_SCHEMA_VERSION,
            "dataset": dataset_name,
            "from": report.from,
            "to": report.to,
            "report_summary": report.report_summary,
            "rows": rows
        })),
        DataExportFormat::Jsonl => Ok(json!({
            "format": "jsonl",
            "schema_version": contracts::CONTRACT_SCHEMA_VERSION,
            "dataset": dataset_name,
            "from": report.from,
            "to": report.to,
            "lines": rows.into_iter().map(|row| serde_json::to_string(&row)).collect::<Result<Vec<_>, _>>()?
        })),
        DataExportFormat::ArrowJson => Ok(json!({
            "format": "arrow-json",
            "schema_version": contracts::CONTRACT_SCHEMA_VERSION,
            "dataset": dataset_name,
            "from": report.from,
            "to": report.to,
            "arrow_schema": contract.map(|contract| contract.arrow_schema).unwrap_or_else(|| json!({
                "format": "arrow-json-schema",
                "name": "rs_llmctl_all_v1",
                "fields": []
            })),
            "rows": rows
        })),
        DataExportFormat::ArrowIpc => {
            let path = output.ok_or_else(|| {
                anyhow::anyhow!("data export --format arrow-ipc requires --output")
            })?;
            let contract = contract.ok_or_else(|| {
                anyhow::anyhow!("data export --format arrow-ipc requires a concrete --dataset")
            })?;
            let row_count = rs_llmctl::data_fabric::write_arrow_ipc(path, &contract, &rows)?;
            let output_path = redact_display_path(path);
            Ok(json!({
                "format": "arrow-ipc",
                "schema_version": contracts::CONTRACT_SCHEMA_VERSION,
                "dataset": dataset_name,
                "artifact": path.file_name().and_then(|name| name.to_str()).unwrap_or("data.arrow"),
                "output_path_redacted": output_path,
                "rows": row_count,
                "arrow_schema": contract.arrow_schema
            }))
        }
        DataExportFormat::Parquet => {
            let path = output
                .ok_or_else(|| anyhow::anyhow!("data export --format parquet requires --output"))?;
            let contract = contract.ok_or_else(|| {
                anyhow::anyhow!("data export --format parquet requires a concrete --dataset")
            })?;
            let row_count = rs_llmctl::data_fabric::write_parquet(path, &contract, &rows)?;
            let output_path = redact_display_path(path);
            Ok(json!({
                "format": "parquet",
                "schema_version": contracts::CONTRACT_SCHEMA_VERSION,
                "dataset": dataset_name,
                "artifact": path.file_name().and_then(|name| name.to_str()).unwrap_or("data.parquet"),
                "output_path_redacted": output_path,
                "rows": row_count,
                "arrow_schema": contract.arrow_schema
            }))
        }
    }
}

fn dataset_rows(
    report: &reporting::DataExport,
    dataset: DataDataset,
) -> Result<Vec<serde_json::Value>> {
    match dataset {
        DataDataset::All => Ok(vec![serde_json::to_value(report)?]),
        DataDataset::Security => Ok(report
            .audit_events
            .iter()
            .map(|event| {
                json!({
                    "at": event.at,
                    "kind": "audit",
                    "actor": event.actor,
                    "team": event.team,
                    "resource": event.resource,
                    "outcome": event.outcome,
                    "request_id": event.request_id
                })
            })
            .chain(report.quota_decisions.iter().map(|decision| {
                json!({
                    "at": decision.at,
                    "kind": "quota-decision",
                    "actor": decision.actor,
                    "team": decision.team,
                    "resource": decision.model,
                    "outcome": if decision.allowed { "allowed" } else { "denied" },
                    "request_id": decision.request_id
                })
            }))
            .collect()),
        DataDataset::Observability => Ok(report
            .observation_events
            .iter()
            .map(|event| {
                json!({
                    "at": event.at,
                    "kind": event.kind,
                    "source": event.source,
                    "model": event.model,
                    "value": event.value,
                    "unit": event.unit,
                    "request_id": event.request_id
                })
            })
            .collect()),
        DataDataset::Usage => Ok(report
            .usage_events
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()?),
        DataDataset::User => Ok(report
            .usage_summary
            .by_actor
            .iter()
            .map(|actor| {
                let team = report
                    .usage_events
                    .iter()
                    .find(|event| event.actor == actor.key)
                    .map(|event| event.team.as_str())
                    .unwrap_or("unknown");
                json!({
                    "actor": actor.key,
                    "team": team,
                    "request_count": actor.request_count,
                    "input_tokens": actor.input_tokens,
                    "output_tokens": actor.output_tokens,
                    "total_tokens": actor.total_tokens
                })
            })
            .collect()),
        DataDataset::Finops => {
            let mut rows = Vec::new();
            rows.extend(report.usage_summary.by_team.iter().map(|team| {
                json!({
                    "team": team.key,
                    "actor": null,
                    "model": null,
                    "request_count": team.request_count,
                    "total_tokens": team.total_tokens,
                    "total_latency_ms": team.total_latency_ms
                })
            }));
            rows.extend(report.usage_summary.by_actor.iter().map(|actor| {
                json!({
                    "team": null,
                    "actor": actor.key,
                    "model": null,
                    "request_count": actor.request_count,
                    "total_tokens": actor.total_tokens,
                    "total_latency_ms": actor.total_latency_ms
                })
            }));
            rows.extend(report.usage_summary.by_model.iter().map(|model| {
                json!({
                    "team": null,
                    "actor": null,
                    "model": model.key,
                    "request_count": model.request_count,
                    "total_tokens": model.total_tokens,
                    "total_latency_ms": model.total_latency_ms
                })
            }));
            Ok(rows)
        }
        DataDataset::Models => Ok(report
            .models
            .iter()
            .map(|model| {
                json!({
                    "alias": model.alias,
                    "role": model.role,
                    "weight": model.weight,
                    "updated_at": model.updated_at
                })
            })
            .collect()),
        DataDataset::Drift => Ok(report
            .observation_events
            .iter()
            .filter(|event| event.kind.contains("drift"))
            .map(|event| {
                json!({
                    "at": event.at,
                    "kind": event.kind,
                    "model": event.model,
                    "value": event.value,
                    "unit": event.unit,
                    "request_id": event.request_id
                })
            })
            .collect()),
        DataDataset::Lineage => Ok(report
            .lineage
            .iter()
            .map(|join| {
                json!({
                    "at": join.at,
                    "request_id": join.request_id,
                    "lineage_id": join.lineage_id,
                    "model": join.model,
                    "corpus": join.corpus,
                    "source": join.source
                })
            })
            .collect()),
        DataDataset::Audit => Ok(report
            .audit_events
            .iter()
            .map(|event| {
                json!({
                    "at": event.at,
                    "action": event.action,
                    "actor": event.actor,
                    "team": event.team,
                    "resource": event.resource,
                    "outcome": event.outcome,
                    "request_id": event.request_id
                })
            })
            .collect()),
    }
}

fn redact_display_path(path: &Path) -> String {
    let rendered = path.display().to_string();
    std::env::var("HOME")
        .ok()
        .filter(|home| !home.is_empty())
        .and_then(|home| rendered.strip_prefix(&home).map(|tail| format!("~{tail}")))
        .unwrap_or(rendered)
}

async fn aiops_command(command: AiopsCommand, as_json: bool) -> Result<()> {
    match command {
        AiopsCommand::Gaps => emit(as_json, &aiops_gaps_report()),
        AiopsCommand::SloPlan(args) => emit_slo_plan(args),
        AiopsCommand::IncidentTemplate(args) => emit(as_json, &incident_template(args)),
    }
}

fn aiops_gaps_report() -> serde_json::Value {
    json!({
        "status": "tracked",
        "delivered": [
            "typed production/local config profiles",
            "SSE, log, event, OTel, and data-fabric config fields",
            "schema-versioned contracts for security, observability, usage, user, finops, model, drift, and audit datasets",
            "domain-filtered JSON, JSONL, Arrow-schema JSON, Arrow IPC, and Parquet exports",
            "CRA Article 14 active-control evidence and PCI DSS aligned reporting commands",
            "OpenAI-compatible model and chat serving, local search, recommendations, quotas, and worker lifecycle controls",
            "manifest-driven eval suites that execute golden prompts against OpenAI-compatible endpoints",
            "runtime request-to-lineage joins for chat, local search, and recommendations",
            "Prometheus/Alertmanager rules and Grafana dashboard renderers for SLOs",
            "HMAC policy bundles plus Ed25519 policy signatures and hash-chained transparency logs",
            "Candle-native greedy autoregressive decoding for Qwen3, Gemma-family, and Mistral safetensors paths where Candle exposes model support"
        ],
        "gaps": [
            {
                "area": "native-inference",
                "gap": "DeepSeek, Kimi, and MiniMax remain tracked native backend targets; DeepSeek metadata exists in Candle but is not wired and verified, while Kimi and MiniMax do not expose reviewed Candle architecture modules to instantiate",
                "next_control": "wire DeepSeek first if Candle deepseek2 maps cleanly to the target artifacts, then upgrade Candle or vendor reviewed Kimi and MiniMax model implementations behind the NativeCandleDecoder"
            },
            {
                "area": "observability",
                "gap": "RED metrics, upstream circuit-breaker state metrics, heartbeat, admission rejection metrics, and worker lifecycle telemetry are emitted; deeper burn-rate deployment sync remains operator-managed",
                "next_control": "add optional push/apply helpers for Prometheus and Grafana provisioning"
            },
            {
                "area": "model-quality",
                "gap": "eval suites execute configured prompts, but advanced judges and rubric scoring are not bundled",
                "next_control": "add optional LLM-as-judge and rubric evaluators with deterministic evidence output"
            },
            {
                "area": "lineage",
                "gap": "runtime joins are recorded when clients provide lineage IDs; automatic corpus/model lineage inference is not complete",
                "next_control": "derive lineage IDs from configured model manifests and managed RAG indexes"
            },
            {
                "area": "operations",
                "gap": "SLO plans include Prometheus/Alertmanager rules and Grafana dashboards; live deployment sync is operator-managed",
                "next_control": "add optional push/apply helpers for Prometheus rule files and Grafana dashboard provisioning"
            },
            {
                "area": "governance",
                "gap": "Ed25519 signing and a local transparency log exist; Sigstore/Rekor publication is not bundled",
                "next_control": "add optional Sigstore/Rekor publication for organizations that want public transparency"
            }
        ]
    })
}

fn emit_slo_plan(args: AiopsSloPlanArgs) -> Result<()> {
    let rendered = match args.format {
        AiopsSloPlanFormat::Plan => serde_json::to_string_pretty(&slo_plan(&args))?,
        AiopsSloPlanFormat::Prometheus => prometheus_slo_rules(&args),
        AiopsSloPlanFormat::Grafana => serde_json::to_string_pretty(&grafana_slo_dashboard(&args))?,
    };

    if let Some(output) = args.output {
        stdfs::write(&output, rendered).with_context(|| format!("write {}", output.display()))?;
    } else {
        println!("{rendered}");
    }
    Ok(())
}

fn slo_plan(args: &AiopsSloPlanArgs) -> serde_json::Value {
    json!({
        "schema_version": 1,
        "kind": "slo-plan",
        "generated_at": Utc::now(),
        "slos": {
            "availability_percent": args.availability_percent,
            "latency_p95_ms": args.latency_p95_ms,
            "error_rate_percent": args.error_rate_percent
        },
        "alert_rules": [
            {
                "name": "llmctl_availability_below_slo",
                "expr": format!("100 * (1 - (sum(rate(llmctl_request_errors_total[5m])) / sum(rate(llmctl_requests_total[5m])))) < {}", args.availability_percent),
                "for": "10m",
                "severity": "page"
            },
            {
                "name": "llmctl_high_error_rate",
                "expr": format!("sum(rate(llmctl_request_errors_total[5m])) / sum(rate(llmctl_requests_total[5m])) > {}", args.error_rate_percent / 100.0),
                "for": "10m",
                "severity": "page"
            },
            {
                "name": "llmctl_fast_burn_error_budget",
                "expr": format!("(sum(rate(llmctl_slo_violations_total[5m])) / sum(rate(llmctl_requests_total[5m])) > {fast_burn}) and (sum(rate(llmctl_slo_violations_total[1h])) / sum(rate(llmctl_requests_total[1h])) > {fast_burn})", fast_burn = (100.0 - args.availability_percent) / 100.0 * 14.4),
                "for": "2m",
                "severity": "page"
            },
            {
                "name": "llmctl_slow_burn_error_budget",
                "expr": format!("(sum(rate(llmctl_slo_violations_total[30m])) / sum(rate(llmctl_requests_total[30m])) > {slow_burn}) and (sum(rate(llmctl_slo_violations_total[6h])) / sum(rate(llmctl_requests_total[6h])) > {slow_burn})", slow_burn = (100.0 - args.availability_percent) / 100.0 * 6.0),
                "for": "15m",
                "severity": "ticket"
            },
            {
                "name": "llmctl_high_latency_p95",
                "expr": format!("histogram_quantile(0.95, rate(llmctl_request_latency_ms_bucket[5m])) > {}", args.latency_p95_ms),
                "for": "15m",
                "severity": "ticket"
            }
        ],
        "evidence_commands": [
            "llmctl observe plan",
            "llmctl usage report --hours 24",
            "llmctl compliance evidence"
        ]
    })
}

fn prometheus_slo_rules(args: &AiopsSloPlanArgs) -> String {
    format!(
        r#"groups:
  - name: llmctl_slo_alerts
    rules:
      - alert: LlmctlAvailabilityBelowSlo
        expr: 100 * (1 - (sum(rate(llmctl_request_errors_total[5m])) / sum(rate(llmctl_requests_total[5m])))) < {availability_percent}
        for: 10m
        labels:
          severity: page
          service: llmctl
        annotations:
          summary: rs-llmctl availability is below SLO
          description: Availability over 5m is below {availability_percent}%.
      - alert: LlmctlHighErrorRate
        expr: sum(rate(llmctl_request_errors_total[5m])) / sum(rate(llmctl_requests_total[5m])) > {error_rate}
        for: 10m
        labels:
          severity: page
          service: llmctl
        annotations:
          summary: rs-llmctl error rate exceeds SLO
          description: Error rate over 5m is above {error_rate_percent}%.
      - alert: LlmctlFastBurnErrorBudget
        expr: (sum(rate(llmctl_slo_violations_total[5m])) / sum(rate(llmctl_requests_total[5m])) > {fast_burn}) and (sum(rate(llmctl_slo_violations_total[1h])) / sum(rate(llmctl_requests_total[1h])) > {fast_burn})
        for: 2m
        labels:
          severity: page
          service: llmctl
        annotations:
          summary: rs-llmctl is burning error budget quickly
          description: 5m and 1h burn-rate windows both exceed the fast-burn threshold.
      - alert: LlmctlSlowBurnErrorBudget
        expr: (sum(rate(llmctl_slo_violations_total[30m])) / sum(rate(llmctl_requests_total[30m])) > {slow_burn}) and (sum(rate(llmctl_slo_violations_total[6h])) / sum(rate(llmctl_requests_total[6h])) > {slow_burn})
        for: 15m
        labels:
          severity: ticket
          service: llmctl
        annotations:
          summary: rs-llmctl is steadily burning error budget
          description: 30m and 6h burn-rate windows both exceed the slow-burn threshold.
      - alert: LlmctlHighLatencyP95
        expr: histogram_quantile(0.95, sum(rate(llmctl_request_latency_ms_bucket[5m])) by (le)) > {latency_p95_ms}
        for: 15m
        labels:
          severity: ticket
          service: llmctl
        annotations:
          summary: rs-llmctl p95 latency exceeds SLO
          description: Request latency p95 is above {latency_p95_ms}ms.
"#,
        availability_percent = args.availability_percent,
        error_rate = args.error_rate_percent / 100.0,
        error_rate_percent = args.error_rate_percent,
        fast_burn = (100.0 - args.availability_percent) / 100.0 * 14.4,
        slow_burn = (100.0 - args.availability_percent) / 100.0 * 6.0,
        latency_p95_ms = args.latency_p95_ms,
    )
}

fn grafana_slo_dashboard(args: &AiopsSloPlanArgs) -> serde_json::Value {
    json!({
        "uid": "llmctl-slos",
        "title": "rs-llmctl SLOs",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "30s",
        "tags": ["llmctl", "slo", "aiops"],
        "time": {
            "from": "now-6h",
            "to": "now"
        },
        "templating": {
            "list": [
                {
                    "name": "datasource",
                    "type": "datasource",
                    "query": "prometheus",
                    "current": {
                        "text": "Prometheus",
                        "value": "Prometheus"
                    }
                }
            ]
        },
        "panels": [
            grafana_timeseries_panel(
                1,
                "Availability",
                0,
                0,
                "percent",
                "100 * sum(rate(llmctl_requests_total{status!=\"error\"}[5m])) / sum(rate(llmctl_requests_total[5m]))".to_string(),
                Some(args.availability_percent),
            ),
            grafana_timeseries_panel(
                2,
                "Error Rate",
                12,
                0,
                "percentunit",
                "sum(rate(llmctl_requests_total{status=\"error\"}[5m])) / sum(rate(llmctl_requests_total[5m]))".to_string(),
                Some(args.error_rate_percent / 100.0),
            ),
            grafana_timeseries_panel(
                3,
                "Latency p95",
                0,
                8,
                "ms",
                "histogram_quantile(0.95, sum(rate(llmctl_request_latency_ms_bucket[5m])) by (le))".to_string(),
                Some(args.latency_p95_ms as f64),
            ),
        ]
    })
}

fn grafana_timeseries_panel(
    id: u64,
    title: &str,
    x: u64,
    y: u64,
    unit: &str,
    expr: String,
    threshold: Option<f64>,
) -> serde_json::Value {
    json!({
        "id": id,
        "type": "timeseries",
        "title": title,
        "datasource": {
            "type": "prometheus",
            "uid": "${datasource}"
        },
        "gridPos": {
            "h": 8,
            "w": 12,
            "x": x,
            "y": y
        },
        "targets": [
            {
                "refId": "A",
                "expr": expr,
                "legendFormat": title
            }
        ],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {
                            "color": "green",
                            "value": null
                        },
                        {
                            "color": "red",
                            "value": threshold
                        }
                    ]
                }
            },
            "overrides": []
        },
        "options": {
            "legend": {
                "displayMode": "list",
                "placement": "bottom"
            },
            "tooltip": {
                "mode": "single",
                "sort": "none"
            }
        }
    })
}

async fn eval_command(path: &Path, command: EvalCommand, as_json: bool) -> Result<()> {
    let cfg = load_config(path).await?;
    let path = state_file(&cfg, "eval-runs.jsonl")?;
    match command {
        EvalCommand::Run(args) => {
            let record = json!({
                "schema_version": 1,
                "id": Uuid::new_v4(),
                "at": Utc::now(),
                "model": args.model,
                "suite": args.suite,
                "score": args.score,
                "baseline": args.baseline,
                "delta": args.baseline.map(|baseline| args.score - baseline),
                "notes": args.notes
            });
            append_jsonl(&path, &record).await?;
            emit(as_json, &record)
        }
        EvalCommand::RunSuite(args) => {
            let record = run_eval_suite(&cfg, args).await?;
            append_jsonl(&path, &record).await?;
            emit(as_json, &record)
        }
        EvalCommand::List => emit(
            as_json,
            &json!({
                "schema_version": 1,
                "path": path,
                "runs": read_jsonl(&path).await?
            }),
        ),
        EvalCommand::Report => {
            let runs = read_jsonl(&path).await?;
            emit(as_json, &eval_report(&runs))
        }
    }
}

async fn run_eval_suite(cfg: &Config, args: EvalRunSuiteArgs) -> Result<serde_json::Value> {
    let manifest = read_eval_manifest(&args.manifest).await?;
    if manifest.cases.is_empty() {
        bail!("eval manifest {} has no cases", args.manifest.display());
    }

    let base_url = args
        .base_url
        .unwrap_or_else(|| format!("http://{}:{}", cfg.server.host, cfg.server.port));
    let endpoint = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
    let api_key = match args.api_key_env.as_deref() {
        Some(env) => Some(std::env::var(env).with_context(|| format!("read API key env {env}"))?),
        None => None,
    };
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .context("build eval HTTP client")?;
    let mut cases = Vec::with_capacity(manifest.cases.len());

    for case in &manifest.cases {
        validate_expectation(&case.expect)
            .with_context(|| format!("validate eval case {}", case.id))?;
        let output = if base_url.starts_with("mock://") {
            mock_eval_output(&case.expect, &case.prompt)
        } else {
            execute_eval_case(&client, &endpoint, api_key.as_deref(), &manifest, case)
                .await
                .with_context(|| format!("execute eval case {}", case.id))?
        };
        let checks = score_eval_case(&case.expect, &output)
            .with_context(|| format!("score eval case {}", case.id))?;
        let passed = checks.values().all(|passed| *passed);
        cases.push(json!({
            "id": case.id,
            "passed": passed,
            "checks": checks,
            "output": output
        }));
    }

    let passed = cases
        .iter()
        .filter(|case| case.get("passed").and_then(serde_json::Value::as_bool) == Some(true))
        .count();
    let total = cases.len();
    let score = passed as f64 / total as f64;
    Ok(json!({
        "schema_version": 1,
        "kind": "eval-suite-run",
        "id": Uuid::new_v4(),
        "at": Utc::now(),
        "manifest": args.manifest,
        "base_url": base_url,
        "model": manifest.model,
        "suite": manifest.suite,
        "score": score,
        "passed": passed,
        "failed": total - passed,
        "total": total,
        "cases": cases
    }))
}

fn mock_eval_output(expect: &EvalExpectation, prompt: &str) -> String {
    let mut output = expect
        .exact
        .clone()
        .or_else(|| {
            if expect.contains.is_empty() {
                None
            } else {
                Some(expect.contains.join(" "))
            }
        })
        .unwrap_or_else(|| prompt.to_string());
    if expect.regex.is_some() && !output.contains("score=") {
        output.push_str(" score=1");
    }
    output
}

async fn read_eval_manifest(path: &Path) -> Result<EvalSuiteManifest> {
    let body = fs::read_to_string(path)
        .await
        .with_context(|| format!("read eval manifest {}", path.display()))?;
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("toml") => toml::from_str(&body)
            .with_context(|| format!("parse TOML eval manifest {}", path.display())),
        _ => serde_json::from_str(&body)
            .with_context(|| format!("parse JSON eval manifest {}", path.display())),
    }
}

fn validate_expectation(expect: &EvalExpectation) -> Result<()> {
    if expect.exact.is_none() && expect.contains.is_empty() && expect.regex.is_none() {
        bail!("expectation must set exact, contains, or regex");
    }
    if let Some(pattern) = &expect.regex {
        Regex::new(pattern).with_context(|| format!("compile regex {pattern:?}"))?;
    }
    Ok(())
}

async fn execute_eval_case(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: Option<&str>,
    manifest: &EvalSuiteManifest,
    case: &EvalCaseManifest,
) -> Result<String> {
    let mut messages = Vec::new();
    if let Some(system) = &manifest.system {
        messages.push(json!({"role": "system", "content": system}));
    }
    messages.push(json!({"role": "user", "content": &case.prompt}));

    let mut request = json!({
        "model": &manifest.model,
        "messages": messages,
        "stream": false
    });
    if let Some(temperature) = manifest.temperature {
        request["temperature"] = json!(temperature);
    }
    if let Some(max_tokens) = manifest.max_tokens {
        request["max_tokens"] = json!(max_tokens);
    }

    let mut builder = client.post(endpoint).json(&request);
    if let Some(api_key) = api_key {
        builder = builder.bearer_auth(api_key);
    }
    let response = builder
        .send()
        .await
        .with_context(|| format!("POST {endpoint}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .with_context(|| format!("read response from {endpoint}"))?;
    if !status.is_success() {
        bail!("endpoint returned {status}: {body}");
    }
    let value: serde_json::Value =
        serde_json::from_str(&body).context("parse chat completion response")?;
    value
        .pointer("/choices/0/message/content")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            anyhow::anyhow!("chat completion response missing choices[0].message.content")
        })
}

fn score_eval_case(expect: &EvalExpectation, output: &str) -> Result<BTreeMap<String, bool>> {
    let mut checks = BTreeMap::new();
    if let Some(exact) = &expect.exact {
        checks.insert("exact".to_string(), output == exact);
    }
    if !expect.contains.is_empty() {
        checks.insert(
            "contains".to_string(),
            expect.contains.iter().all(|needle| output.contains(needle)),
        );
    }
    if let Some(pattern) = &expect.regex {
        checks.insert("regex".to_string(), Regex::new(pattern)?.is_match(output));
    }
    Ok(checks)
}

async fn lineage_command(path: &Path, command: LineageCommand, as_json: bool) -> Result<()> {
    let cfg = load_config(path).await?;
    let path = state_file(&cfg, "lineage-records.jsonl")?;
    match command {
        LineageCommand::Record(args) => {
            let record = json!({
                "schema_version": 1,
                "id": args.id,
                "kind": args.kind,
                "parents": args.parents,
                "sha256": args.sha256,
                "source": args.source,
                "recorded_at": Utc::now()
            });
            append_jsonl(&path, &record).await?;
            emit(as_json, &record)
        }
        LineageCommand::List => emit(
            as_json,
            &json!({
                "schema_version": 1,
                "path": path,
                "records": read_jsonl(&path).await?,
                "joins": Storage::connect_config(&cfg.storage)
                    .await?
                    .request_lineage_joins()
                    .await?
            }),
        ),
    }
}

async fn policy_command(command: PolicyCommand, as_json: bool) -> Result<()> {
    match command {
        PolicyCommand::Bundle(args) => {
            let policy = fs::read_to_string(&args.input)
                .await
                .with_context(|| format!("read policy {}", args.input.display()))?;
            let policy_value: serde_json::Value =
                if args.input.extension().and_then(|ext| ext.to_str()) == Some("toml") {
                    let value: toml::Value = toml::from_str(&policy)
                        .with_context(|| format!("parse TOML {}", args.input.display()))?;
                    serde_json::to_value(value)?
                } else {
                    serde_json::from_str(&policy)
                        .with_context(|| format!("parse JSON {}", args.input.display()))?
                };
            let payload = json!({
                "schema_version": 1,
                "kind": "policy-bundle",
                "name": args.name,
                "created_at": Utc::now(),
                "policy": policy_value
            });
            let signature = hmac_signature(&args.signing_key_env, &payload)?;
            let bundle = json!({
                "metadata": {
                    "algorithm": "hmac-sha256",
                    "key_source": format!("env:{}", args.signing_key_env),
                    "signature": signature
                },
                "payload": payload
            });
            fs::write(&args.output, serde_json::to_vec_pretty(&bundle)?)
                .await
                .with_context(|| format!("write {}", args.output.display()))?;
            emit(
                as_json,
                &json!({
                    "status": "created",
                    "path": args.output,
                    "algorithm": "hmac-sha256",
                    "signature": signature
                }),
            )
        }
        PolicyCommand::VerifyBundle(args) => {
            let bytes = fs::read(&args.path)
                .await
                .with_context(|| format!("read {}", args.path.display()))?;
            let bundle: serde_json::Value = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse {}", args.path.display()))?;
            let payload = bundle
                .get("payload")
                .ok_or_else(|| anyhow::anyhow!("policy bundle missing payload"))?;
            let expected = bundle
                .pointer("/metadata/signature")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("policy bundle missing signature"))?;
            let actual = hmac_signature(&args.signing_key_env, payload)?;
            emit(
                as_json,
                &json!({
                    "status": if expected.eq_ignore_ascii_case(&actual) { "valid" } else { "invalid" },
                    "valid": expected.eq_ignore_ascii_case(&actual),
                    "path": args.path,
                    "algorithm": "hmac-sha256"
                }),
            )
        }
        PolicyCommand::Keygen(args) => {
            let mut rng = rand_core::OsRng;
            let signing_key = SigningKey::generate(&mut rng);
            let verifying_key = signing_key.verifying_key();
            let public_key = encode_b64(&verifying_key.to_bytes());
            let private_key = encode_b64(&signing_key.to_bytes());
            let private_doc = json!({
                "schema_version": 1,
                "kind": "policy-signing-private-key",
                "algorithm": "ed25519",
                "private_key": private_key,
                "public_key": public_key
            });
            let public_doc = json!({
                "schema_version": 1,
                "kind": "policy-signing-public-key",
                "algorithm": "ed25519",
                "public_key": public_key
            });
            write_json_file(&args.private_key, &private_doc).await?;
            restrict_private_key_file(&args.private_key).await?;
            write_json_file(&args.public_key, &public_doc).await?;
            emit(
                as_json,
                &json!({
                    "status": "created",
                    "algorithm": "ed25519",
                    "private_key": args.private_key,
                    "public_key": args.public_key
                }),
            )
        }
        PolicyCommand::Sign(args) => {
            let input = fs::read(&args.input)
                .await
                .with_context(|| format!("read {}", args.input.display()))?;
            let signing_key = read_policy_signing_key(&args.private_key).await?;
            let signature = signing_key.sign(&input);
            let payload_sha256 = sha256_hex(&input);
            let signature_doc = json!({
                "schema_version": 1,
                "kind": "policy-signature",
                "algorithm": "ed25519",
                "signed_at": Utc::now(),
                "payload_sha256": payload_sha256,
                "public_key": encode_b64(&signing_key.verifying_key().to_bytes()),
                "signature": encode_b64(&signature.to_bytes())
            });
            write_json_file(&args.signature, &signature_doc).await?;
            emit(
                as_json,
                &json!({
                    "status": "signed",
                    "algorithm": "ed25519",
                    "input": args.input,
                    "signature": args.signature,
                    "payload_sha256": payload_sha256
                }),
            )
        }
        PolicyCommand::Verify(args) => {
            let input = fs::read(&args.input)
                .await
                .with_context(|| format!("read {}", args.input.display()))?;
            let verifying_key = read_policy_verifying_key(&args.public_key).await?;
            let signature_doc = read_json_file(&args.signature).await?;
            require_algorithm(&signature_doc)?;
            let expected_hash = required_str(&signature_doc, "payload_sha256")?;
            let actual_hash = sha256_hex(&input);
            let signature_bytes = decode_b64(required_str(&signature_doc, "signature")?)?;
            let signature = Signature::from_slice(&signature_bytes).context("parse signature")?;
            let signature_valid = verifying_key.verify(&input, &signature).is_ok();
            let hash_valid = expected_hash.eq_ignore_ascii_case(&actual_hash);
            emit(
                as_json,
                &json!({
                    "status": if signature_valid && hash_valid { "valid" } else { "invalid" },
                    "valid": signature_valid && hash_valid,
                    "algorithm": "ed25519",
                    "input": args.input,
                    "signature": args.signature,
                    "payload_sha256": actual_hash
                }),
            )
        }
        PolicyCommand::Log { command } => match command {
            PolicyLogCommand::Append(args) => {
                let entry = append_policy_log_entry(&args).await?;
                emit(as_json, &entry)
            }
            PolicyLogCommand::Verify(args) => {
                let result = verify_policy_log(&args.log_path).await?;
                emit(as_json, &result)
            }
        },
        PolicyCommand::LegalHoldPlan(args) => emit(
            as_json,
            &json!({
                "schema_version": 1,
                "kind": "legal-hold-plan",
                "generated_at": Utc::now(),
                "dataset": DatasetKind::from(args.dataset).as_str(),
                "case_id": args.case_id,
                "reason": args.reason,
                "retention": {
                    "override": "hold_until_released",
                    "applies_to_dataset": true
                },
                "operator_steps": [
                    "attach this plan to the case record",
                    "exclude dataset scope from automated retention pruning",
                    "generate monthly audit and data export envelopes while hold is active",
                    "record signed release of hold before retention resumes"
                ]
            }),
        ),
    }
}

fn eval_report(runs: &[serde_json::Value]) -> serde_json::Value {
    let mut by_model: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for run in runs {
        if let (Some(model), Some(score)) = (
            run.get("model").and_then(serde_json::Value::as_str),
            run.get("score").and_then(serde_json::Value::as_f64),
        ) {
            by_model.entry(model.to_string()).or_default().push(score);
        }
    }
    let models = by_model
        .into_iter()
        .map(|(model, scores)| {
            let count = scores.len() as f64;
            let average_score = if count == 0.0 {
                None
            } else {
                Some(scores.iter().sum::<f64>() / count)
            };
            json!({
                "model": model,
                "runs": scores.len(),
                "average_score": average_score
            })
        })
        .collect::<Vec<_>>();
    json!({
        "schema_version": 1,
        "kind": "eval-report",
        "generated_at": Utc::now(),
        "run_count": runs.len(),
        "models": models
    })
}

async fn append_jsonl(path: &Path, value: &serde_json::Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }
    let mut existing = match fs::read_to_string(path).await {
        Ok(body) => body,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(err) => return Err(err).with_context(|| format!("read {}", path.display())),
    };
    existing.push_str(&serde_json::to_string(value)?);
    existing.push('\n');
    fs::write(path, existing)
        .await
        .with_context(|| format!("write {}", path.display()))
}

async fn read_jsonl(path: &Path) -> Result<Vec<serde_json::Value>> {
    let body = match fs::read_to_string(path).await {
        Ok(body) => body,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err).with_context(|| format!("read {}", path.display())),
    };
    body.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("parse jsonl record"))
        .collect()
}

fn state_file(cfg: &Config, name: &str) -> Result<PathBuf> {
    let dir = cfg
        .storage
        .db_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("storage db_path has no parent directory"))?;
    Ok(dir.join(name))
}

type HmacSha256 = Hmac<Sha256>;

fn hmac_signature(key_env: &str, payload: &serde_json::Value) -> Result<String> {
    let key = std::env::var(key_env).with_context(|| format!("read signing key env {key_env}"))?;
    let canonical = reporting::canonical_json(payload)?;
    let mut mac = HmacSha256::new_from_slice(key.as_bytes())?;
    mac.update(canonical.as_bytes());
    Ok(hex::encode(mac.finalize().into_bytes()))
}

async fn write_json_file(path: &Path, value: &serde_json::Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .await
        .with_context(|| format!("write {}", path.display()))
}

async fn read_json_file(path: &Path) -> Result<serde_json::Value> {
    let bytes = fs::read(path)
        .await
        .with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

async fn restrict_private_key_file(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .await
            .with_context(|| format!("set private key permissions {}", path.display()))?;
    }
    Ok(())
}

async fn read_policy_signing_key(path: &Path) -> Result<SigningKey> {
    let doc = read_json_file(path).await?;
    require_algorithm(&doc)?;
    let bytes = decode_b64(required_str(&doc, "private_key")?)?;
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("ed25519 private key must be 32 bytes"))?;
    Ok(SigningKey::from_bytes(&bytes))
}

async fn read_policy_verifying_key(path: &Path) -> Result<VerifyingKey> {
    let doc = read_json_file(path).await?;
    require_algorithm(&doc)?;
    let bytes = decode_b64(required_str(&doc, "public_key")?)?;
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("ed25519 public key must be 32 bytes"))?;
    VerifyingKey::from_bytes(&bytes).context("parse ed25519 public key")
}

fn require_algorithm(doc: &serde_json::Value) -> Result<()> {
    let algorithm = required_str(doc, "algorithm")?;
    if algorithm != "ed25519" {
        bail!("unsupported policy signing algorithm {algorithm}");
    }
    Ok(())
}

fn required_str<'a>(doc: &'a serde_json::Value, field: &str) -> Result<&'a str> {
    doc.get(field)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("missing string field {field}"))
}

fn encode_b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn decode_b64(value: &str) -> Result<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(value)
        .context("decode base64")
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

async fn append_policy_log_entry(args: &PolicyLogAppendArgs) -> Result<serde_json::Value> {
    let current = read_jsonl(&args.log_path).await?;
    let verification = verify_policy_log_values(&current)?;
    if !verification
        .get("valid")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        bail!("policy transparency log hash chain is invalid");
    }

    let artifact = fs::read(&args.artifact)
        .await
        .with_context(|| format!("read {}", args.artifact.display()))?;
    let signature_sha256 = if let Some(path) = &args.signature {
        let bytes = fs::read(path)
            .await
            .with_context(|| format!("read {}", path.display()))?;
        Some(sha256_hex(&bytes))
    } else {
        None
    };
    let previous_hash = current
        .last()
        .and_then(|entry| entry.get("entry_hash"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let body = json!({
        "schema_version": 1,
        "kind": "policy-transparency-log-entry",
        "index": current.len(),
        "logged_at": Utc::now(),
        "artifact_sha256": sha256_hex(&artifact),
        "signature_sha256": signature_sha256,
        "previous_hash": previous_hash
    });
    let entry_hash = policy_log_entry_hash(&body)?;
    let mut entry = body;
    entry["entry_hash"] = json!(entry_hash);
    append_jsonl(&args.log_path, &entry).await?;
    Ok(entry)
}

async fn verify_policy_log(path: &Path) -> Result<serde_json::Value> {
    let entries = read_jsonl(path).await?;
    verify_policy_log_values(&entries)
}

fn verify_policy_log_values(entries: &[serde_json::Value]) -> Result<serde_json::Value> {
    let mut previous_hash: Option<String> = None;
    for (index, entry) in entries.iter().enumerate() {
        let Some(actual_hash) = entry.get("entry_hash").and_then(serde_json::Value::as_str) else {
            return Ok(policy_log_verification(
                false,
                entries.len(),
                index,
                "missing entry_hash",
            ));
        };
        if entry.get("index").and_then(serde_json::Value::as_u64) != Some(index as u64) {
            return Ok(policy_log_verification(
                false,
                entries.len(),
                index,
                "index mismatch",
            ));
        }
        if entry
            .get("previous_hash")
            .and_then(serde_json::Value::as_str)
            != previous_hash.as_deref()
        {
            return Ok(policy_log_verification(
                false,
                entries.len(),
                index,
                "previous_hash mismatch",
            ));
        }
        let expected_hash = policy_log_entry_hash(entry)?;
        if actual_hash != expected_hash {
            return Ok(policy_log_verification(
                false,
                entries.len(),
                index,
                "entry_hash mismatch",
            ));
        }
        previous_hash = Some(actual_hash.to_string());
    }
    Ok(json!({
        "status": "valid",
        "valid": true,
        "entries": entries.len(),
        "head": previous_hash
    }))
}

fn policy_log_verification(
    valid: bool,
    entries: usize,
    failed_index: usize,
    reason: &str,
) -> serde_json::Value {
    json!({
        "status": if valid { "valid" } else { "invalid" },
        "valid": valid,
        "entries": entries,
        "failed_index": failed_index,
        "reason": reason
    })
}

fn policy_log_entry_hash(entry: &serde_json::Value) -> Result<String> {
    let mut body = entry.clone();
    if let Some(object) = body.as_object_mut() {
        object.remove("entry_hash");
    }
    let canonical = reporting::canonical_json(&body)?;
    Ok(sha256_hex(canonical.as_bytes()))
}

fn incident_template(args: AiopsIncidentTemplateArgs) -> serde_json::Value {
    json!({
        "schema_version": 1,
        "kind": "incident-evidence-template",
        "generated_at": Utc::now(),
        "severity": args.severity,
        "team": args.team,
        "cra_article_14": {
            "operational_status": "active_control",
            "early_warning_due": "within_24_hours",
            "vulnerability_notification_due": "within_72_hours",
            "final_vulnerability_report_due": "within_14_days_after_mitigation"
        },
        "sections": [
            "summary",
            "timeline",
            "affected_models",
            "affected_users_or_teams",
            "security_impact",
            "data_impact",
            "mitigation",
            "evidence"
        ],
        "evidence_commands": [
            "llmctl security audit-config",
            "llmctl audit report monthly --envelope",
            "llmctl data export --envelope",
            "llmctl lineage list",
            "llmctl eval report"
        ]
    })
}

async fn compliance_command(path: &Path, command: ComplianceCommand, as_json: bool) -> Result<()> {
    let cfg = load_config(path).await?;
    let evidence = compliance_evidence(&cfg);
    match command {
        ComplianceCommand::Evidence => emit(as_json, &evidence),
        ComplianceCommand::CraArticle14 => emit(as_json, &evidence["cra_article_14"]),
        ComplianceCommand::PciDss => emit(as_json, &evidence["pci_dss"]),
        ComplianceCommand::ReleaseChecklist => emit(as_json, &evidence["release_integrity"]),
    }
}

fn compliance_evidence(cfg: &Config) -> serde_json::Value {
    json!({
        "generated_at": Utc::now(),
        "status": "operator_evidence_ready",
        "security_posture": {
            "production": cfg.security.production,
            "require_auth": cfg.security.require_auth,
            "bind_external": cfg.security.bind_external || config::is_external_host(&cfg.server.host),
            "hashed_api_keys": cfg.security.api_keys.iter().all(|key| key.sha256.len() == 64),
            "api_key_count": cfg.security.api_keys.len(),
            "tls_termination": {
                "enabled": cfg.security.tls_termination.enabled,
                "provider": cfg.security.tls_termination.provider.as_deref(),
                "evidence": cfg.security.tls_termination.evidence.as_deref(),
                "m_tls": cfg.security.tls_termination.m_tls
            },
            "audit_retention_days": cfg.audit.retention_days,
            "monthly_reports": cfg.audit.monthly_reports
        },
        "evidence_completeness": {
            "production_security_validation": cfg.security.production || cfg.security.bind_external || config::is_external_host(&cfg.server.host),
            "hashed_api_keys": cfg.security.api_keys.iter().all(|key| key.sha256.len() == 64),
            "tls_termination_documented": cfg.security.tls_termination.enabled
                && cfg.security.tls_termination.provider.as_deref().is_some_and(|value| !value.trim().is_empty())
                && cfg.security.tls_termination.evidence.as_deref().is_some_and(|value| !value.trim().is_empty()),
            "audit_reports_enabled": cfg.audit.monthly_reports && cfg.audit.retention_days > 0,
            "otel_enabled": cfg.observability.traces_enabled && cfg.observability.metrics_enabled && cfg.observability.logs_enabled,
            "otel_exporter_configured": cfg.observability.exporter.endpoint.is_some() || cfg.observability.otlp_endpoint.is_some(),
            "cra_article_14_active_control": cfg.audit.monthly_reports
                && cfg.audit.retention_days > 0
                && cfg.observability.traces_enabled
                && cfg.observability.metrics_enabled
                && cfg.observability.logs_enabled
                && (cfg.observability.exporter.endpoint.is_some() || cfg.observability.otlp_endpoint.is_some()),
            "release_integrity_scripts": [
                "packaging/generate-sbom.sh",
                "packaging/generate-checksums.sh",
                "packaging/sign-release.sh"
            ]
        },
        "cra_article_14": {
            "regulation": "Regulation (EU) 2024/2847",
            "operational_status": "active_control",
            "control_assumption": "treat CRA Article 14 obligations as live for all production operations",
            "early_warning_due": "within_24_hours",
            "vulnerability_notification_due": "within_72_hours",
            "final_vulnerability_report_due": "within_14_days_after_mitigation",
            "severe_incident_notification_due": "without_undue_delay",
            "evidence_commands": [
                "llmctl security audit-config",
                "llmctl audit report monthly --envelope",
                "llmctl data export --envelope",
                "llmctl compliance evidence"
            ],
            "workflow": [
                "classify vulnerability or severe incident",
                "open incident record with impacted versions and mitigations",
                "submit regulatory notification in the required window",
                "attach signed release artifacts, SBOM, audit envelope, and data export",
                "close with final report after mitigation verification"
            ]
        },
        "pci_dss": {
            "baseline": "pci_dss_v4_0_1_aligned_controls",
            "controls": [
                { "area": "access_control", "evidence": "hashed API keys, scopes, auth-required production validation" },
                { "area": "audit_logging", "evidence": "audit events, request IDs, report envelopes, retention plan/apply" },
                { "area": "vulnerability_management", "evidence": "cargo audit, SBOM generation, signed release checksums" },
                { "area": "secure_configuration", "evidence": "security check, audit-config, documented TLS termination or mTLS, least privilege systemd unit" },
                { "area": "monitoring", "evidence": "OTel traces, metrics, logs, resource snapshots, drift observations" }
            ],
            "regular_reports": [
                "monthly audit report",
                "per-request audit report",
                "data export envelope",
                "usage chargeback report",
                "quota report"
            ]
        },
        "release_integrity": {
            "sbom": "packaging/generate-sbom.sh",
            "checksums": "packaging/generate-checksums.sh",
            "signing": "packaging/sign-release.sh",
            "provenance": "CI run URL, git commit, signed tag, SBOM, checksums, signature",
            "required_before_release": [
                "cargo fmt --check",
                "cargo clippy --all-targets --all-features -- -D warnings",
                "cargo test --all-targets --all-features",
                "cargo audit",
                "packaging/generate-sbom.sh",
                "packaging/generate-checksums.sh",
                "packaging/sign-release.sh dist"
            ]
        }
    })
}

async fn integration_command(
    path: &Path,
    command: IntegrationCommand,
    as_json: bool,
) -> Result<()> {
    let cfg = load_config(path).await?;
    match command {
        IntegrationCommand::AqeContract => {
            emit(as_json, &integrations::aqe_governance_contract(&cfg))
        }
    }
}

async fn amd_command(command: AmdCommand, as_json: bool) -> Result<()> {
    match command {
        AmdCommand::Qualify(args) => emit(
            as_json,
            &rs_llmctl::amd::qualification_report_with_evidence(
                args.preview,
                args.arch_opt_in,
                args.evidence.as_deref(),
            ),
        ),
        AmdCommand::InstallServer(args) => {
            let script = args.script.unwrap_or_else(|| {
                std::env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .join("scripts")
                    .join("install-amd-hip.sh")
            });
            if !script.exists() {
                bail!(
                    "install script not found: {}  (set --script or run from the rs-llmctl repo root)",
                    script.display()
                );
            }
            let mut cmd = std::process::Command::new("bash");
            cmd.arg(&script);
            if args.dry_run {
                cmd.env("DRY_RUN", "1");
            }
            let status = cmd
                .status()
                .with_context(|| format!("failed to run {}", script.display()))?;
            if !status.success() {
                bail!("install script exited with status {status}");
            }
            Ok(())
        }
    }
}

async fn load_config(path: &Path) -> Result<Config> {
    config::load(path)
        .await
        .with_context(|| format!("load {}", path.display()))
}

async fn read_startup_plan(path: &Path) -> Result<StartupPlan> {
    let bytes = fs::read(path)
        .await
        .with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse startup plan {}", path.display()))
}

async fn create_storage_dirs(storage: &StorageConfig) -> Result<()> {
    let plan = storage.connection_plan()?;
    if plan.backend == rs_llmctl::storage::StorageBackend::Sqlite {
        if let Some(parent) = storage.db_path.parent() {
            fs::create_dir_all(parent).await?;
        }
    }
    fs::create_dir_all(&storage.model_dir).await?;
    Ok(())
}

async fn init_storage(storage: &StorageConfig) -> Result<Storage> {
    create_storage_dirs(storage).await?;
    Storage::connect_config(storage).await
}

async fn record_security_key_event(
    cfg: &Config,
    action: &str,
    resource: &str,
    outcome: &str,
    detail_json: Value,
) -> Result<()> {
    let storage = init_storage(&cfg.storage).await?;
    let event = AuditEvent::new(
        None,
        "llmctl-cli",
        "security",
        action,
        resource,
        outcome,
        detail_json,
    );
    storage.insert_audit_event(&event).await
}

async fn persist_models(path: &Path, cfg: &Config) -> Result<()> {
    config::save(path, cfg).await?;

    let storage = init_storage(&cfg.storage).await?;
    for model in &cfg.models {
        storage.upsert_model(model).await?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum ServiceLifecycleAction {
    Status,
    Start,
    Stop,
    Restart,
    Upgrade,
    Downgrade,
}

impl ServiceLifecycleAction {
    fn as_str(self) -> &'static str {
        match self {
            Self::Status => "status",
            Self::Start => "start",
            Self::Stop => "stop",
            Self::Restart => "restart",
            Self::Upgrade => "upgrade",
            Self::Downgrade => "downgrade",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct ServiceCommandPlan {
    program: String,
    args: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct OneBinaryEntrypoint {
    program: String,
    args: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ServiceLifecyclePlan {
    status: String,
    action: String,
    service_name: String,
    scope: String,
    dry_run: bool,
    one_binary: bool,
    runtime_backend: rs_llmctl::runtime::RuntimeBackend,
    entrypoint: OneBinaryEntrypoint,
    commands: Vec<ServiceCommandPlan>,
    restart_hint: String,
    artifact_action_supported: bool,
    artifact_action_note: Option<String>,
}

#[derive(Debug, Serialize)]
struct ServiceCommandResult {
    command: ServiceCommandPlan,
    success: bool,
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
}

#[derive(Debug, Serialize)]
struct ServiceLifecycleResult {
    status: String,
    action: String,
    service_name: String,
    scope: String,
    dry_run: bool,
    one_binary: bool,
    runtime_backend: rs_llmctl::runtime::RuntimeBackend,
    entrypoint: OneBinaryEntrypoint,
    commands: Vec<ServiceCommandPlan>,
    restart_hint: String,
    results: Vec<ServiceCommandResult>,
}

fn plan_service_lifecycle(
    action: ServiceLifecycleAction,
    args: &ServiceLifecycleArgs,
) -> ServiceLifecyclePlan {
    let scope = if args.user { "user" } else { "system" };
    let service_name = normalize_service_name(&args.service_name);
    let systemctl_scope = if args.user { Some("--user") } else { None };
    let commands = service_systemctl_verbs(action)
        .into_iter()
        .map(|verb| {
            let mut command_args = Vec::new();
            if let Some(scope_arg) = systemctl_scope {
                command_args.push(scope_arg.to_string());
            }
            command_args.push(verb.to_string());
            if verb != "daemon-reload" {
                command_args.push(service_name.clone());
            }
            ServiceCommandPlan {
                program: "systemctl".to_string(),
                args: command_args,
            }
        })
        .collect();
    let artifact_action_supported = !matches!(
        action,
        ServiceLifecycleAction::Upgrade | ServiceLifecycleAction::Downgrade
    );
    let artifact_action_note = if artifact_action_supported {
        None
    } else {
        Some(
            "service upgrade/downgrade is a planning guard only; install a verified release artifact with install.sh or the system package manager, then restart the service"
                .to_string(),
        )
    };

    ServiceLifecyclePlan {
        status: "planned".to_string(),
        action: action.as_str().to_string(),
        service_name: service_name.clone(),
        scope: scope.to_string(),
        dry_run: args.dry_run,
        one_binary: true,
        runtime_backend: rs_llmctl::runtime::RuntimeBackend::CandleNative,
        entrypoint: one_binary_entrypoint(),
        commands,
        restart_hint: restart_hint(scope, &service_name),
        artifact_action_supported,
        artifact_action_note,
    }
}

fn service_systemctl_verbs(action: ServiceLifecycleAction) -> Vec<&'static str> {
    match action {
        ServiceLifecycleAction::Status => vec!["status"],
        ServiceLifecycleAction::Start => vec!["start"],
        ServiceLifecycleAction::Stop => vec!["stop"],
        ServiceLifecycleAction::Restart => vec!["restart"],
        ServiceLifecycleAction::Upgrade | ServiceLifecycleAction::Downgrade => Vec::new(),
    }
}

async fn execute_service_lifecycle(plan: ServiceLifecyclePlan) -> Result<ServiceLifecycleResult> {
    ensure_service_lifecycle_allowed(&plan)?;
    if !plan.artifact_action_supported {
        bail!(
            "{}",
            plan.artifact_action_note
                .as_deref()
                .unwrap_or("service artifact action is not executable")
        );
    }
    let mut results = Vec::new();
    for command in &plan.commands {
        let output = TokioCommand::new(&command.program)
            .args(&command.args)
            .output()
            .await
            .with_context(|| format!("run {}", shell_words(command)))?;
        results.push(ServiceCommandResult {
            command: command.clone(),
            success: output.status.success(),
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        });
    }

    let success = results.iter().all(|result| result.success);
    Ok(ServiceLifecycleResult {
        status: if success { "ok" } else { "failed" }.to_string(),
        action: plan.action,
        service_name: plan.service_name,
        scope: plan.scope,
        dry_run: false,
        one_binary: plan.one_binary,
        runtime_backend: plan.runtime_backend,
        entrypoint: plan.entrypoint,
        commands: plan.commands,
        restart_hint: plan.restart_hint,
        results,
    })
}

fn ensure_service_lifecycle_allowed(plan: &ServiceLifecyclePlan) -> Result<()> {
    if plan.dry_run || plan.scope != "system" || current_uid().unwrap_or(0) == 0 {
        return Ok(());
    }
    bail!(
        "system service scope requires root or polkit authorization; rerun with sudo or pass --user for a user-scoped service"
    )
}

fn current_uid() -> Option<u32> {
    let status = stdfs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("Uid:")
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|uid| uid.parse::<u32>().ok())
    })
}

fn normalize_service_name(service_name: &str) -> String {
    let trimmed = service_name.trim();
    if trimmed.ends_with(".service") {
        trimmed.to_string()
    } else {
        format!("{trimmed}.service")
    }
}

fn default_restart_hint() -> String {
    restart_hint("system", DEFAULT_SERVICE_NAME)
}

fn one_binary_entrypoint() -> OneBinaryEntrypoint {
    OneBinaryEntrypoint {
        program: "llmctl".to_string(),
        args: vec!["server".to_string(), "run".to_string()],
    }
}

fn restart_hint(scope: &str, service_name: &str) -> String {
    match scope {
        "system" => format!("systemctl restart {service_name}"),
        _ => format!("systemctl --user restart {service_name}"),
    }
}

fn shell_words(command: &ServiceCommandPlan) -> String {
    std::iter::once(command.program.as_str())
        .chain(command.args.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join(" ")
}

async fn replace_model(
    path: &Path,
    cfg: &mut Config,
    args: ModelReplaceArgs,
    action: &str,
    status: &str,
    as_json: bool,
) -> Result<()> {
    let previous = cfg
        .models
        .iter()
        .find(|model| model.alias == args.alias)
        .cloned()
        .with_context(|| format!("model alias '{}' is not configured", args.alias))?;
    let target_alias = args.new_alias.unwrap_or_else(|| previous.alias.clone());
    let role = args.role.unwrap_or_else(|| previous.role.clone());
    let family = args.family.or_else(|| previous.family.clone());
    let weight = args.weight.unwrap_or(previous.weight);

    if args.dry_run {
        return emit(
            as_json,
            &json!({
                "status": "planned",
                "action": action,
                "alias": &previous.alias,
                "new_alias": &target_alias,
                "previous_weight": previous.weight,
                "weight": weight,
                "role": role,
                "source": &args.source,
                "copy": args.copy,
                "sha256": args.sha256,
                "restart_required": true,
                "restart_hint": default_restart_hint(),
                "runtime_backend": &cfg.runtime.backend,
                "one_binary": true,
                "entrypoint": one_binary_entrypoint(),
                "previous_model": &previous,
            }),
        );
    }

    create_storage_dirs(&cfg.storage).await?;
    let installed = model::install_model(&ModelInstallRequest {
        alias: target_alias.clone(),
        source: model_source(&args.source),
        cache_dir: cfg.storage.model_dir.clone(),
        copy_to_cache: args.copy,
        expected_sha256: args.sha256,
        role,
        family,
        weight,
    })
    .await?;

    if target_alias != previous.alias {
        cfg.models.retain(|model| model.alias != previous.alias);
    }
    upsert_model(&mut cfg.models, installed.config.clone());
    persist_models(path, cfg).await?;

    emit(
        as_json,
        &json!({
            "status": status,
            "action": action,
            "alias": &previous.alias,
            "new_alias": &target_alias,
            "previous_model": &previous,
            "model": &installed,
            "restart_required": true,
            "restart_hint": default_restart_hint(),
            "runtime_backend": &cfg.runtime.backend,
            "one_binary": true,
            "entrypoint": one_binary_entrypoint(),
            "models": &cfg.models,
        }),
    )
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

fn mode_name(mode: &Mode) -> &'static str {
    match mode {
        Mode::Single => "single",
        Mode::ColdSwap => "cold-swap",
        Mode::HotSwap => "hot-swap",
        Mode::Weighted => "weighted",
        Mode::Fallback => "fallback",
    }
}

#[derive(Debug, Deserialize)]
struct ImportedQuotaPolicy {
    #[serde(default = "default_quota_policy_format")]
    format: String,
    #[serde(default)]
    quotas: Vec<QuotaConfig>,
}

fn default_quota_policy_format() -> String {
    "json".to_string()
}

async fn load_quota_policy(path: &Path) -> Result<ImportedQuotaPolicy> {
    let body = fs::read_to_string(path)
        .await
        .with_context(|| format!("read quota policy {}", path.display()))?;
    if path.extension().and_then(|ext| ext.to_str()) == Some("toml") {
        let mut imported: ImportedQuotaPolicy =
            toml::from_str(&body).with_context(|| format!("parse TOML {}", path.display()))?;
        imported.format = "toml".to_string();
        return Ok(imported);
    }

    let value: serde_json::Value =
        serde_json::from_str(&body).with_context(|| format!("parse JSON {}", path.display()))?;
    if value.is_array() {
        let quotas = serde_json::from_value(value)
            .with_context(|| format!("parse quotas {}", path.display()))?;
        Ok(ImportedQuotaPolicy {
            format: "json".to_string(),
            quotas,
        })
    } else {
        let mut imported: ImportedQuotaPolicy = serde_json::from_value(value)
            .with_context(|| format!("parse quotas {}", path.display()))?;
        imported.format = "json".to_string();
        Ok(imported)
    }
}

fn validate_quota_policies(quotas: &[QuotaConfig]) -> Result<()> {
    let mut subjects = BTreeMap::new();
    let mut teams = BTreeMap::new();
    for (index, quota) in quotas.iter().enumerate() {
        if quota.subject.trim().is_empty() {
            bail!("quotas[{index}].subject must not be empty");
        }
        if quota.team.trim().is_empty() {
            bail!("quotas[{index}].team must not be empty");
        }
        if quota.requests_per_minute == 0 {
            bail!("quotas[{index}].requests_per_minute must be greater than zero");
        }
        if quota.tokens_per_day == 0 {
            bail!("quotas[{index}].tokens_per_day must be greater than zero");
        }
        if quota.max_concurrency == 0 {
            bail!("quotas[{index}].max_concurrency must be greater than zero");
        }
        if quota
            .allowed_models
            .iter()
            .any(|model| model.trim().is_empty())
        {
            bail!("quotas[{index}].allowed_models must not contain empty model aliases");
        }
        if let Some(first_index) = subjects.insert(quota.subject.as_str(), index) {
            bail!(
                "quotas[{index}].subject duplicates quotas[{first_index}].subject: duplicate subject {:?}",
                quota.subject
            );
        }
        if let Some(first_index) = teams.insert(quota.team.as_str(), index) {
            bail!(
                "quotas[{index}].team duplicates quotas[{first_index}].team: duplicate team {:?}",
                quota.team
            );
        }
    }
    Ok(())
}

fn upsert_api_key(keys: &mut Vec<ApiKeyConfig>, key: ApiKeyConfig) -> &'static str {
    if let Some(existing) = keys.iter_mut().find(|existing| existing.id == key.id) {
        *existing = key;
        "updated"
    } else {
        keys.push(key);
        "inserted"
    }
}

fn validate_add_key_args(id: &str, sha256: &str, subject: &str, team: &str) -> Result<()> {
    validate_api_key_id(id)?;
    validate_sha256_digest(sha256)?;
    if subject.trim().is_empty() {
        bail!("subject must not be empty");
    }
    if team.trim().is_empty() {
        bail!("team must not be empty");
    }
    Ok(())
}

fn validate_api_key_id(id: &str) -> Result<()> {
    if id.trim().is_empty() {
        bail!("api key id must not be empty");
    }
    if !id
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
    {
        bail!("api key id must contain only ASCII letters, digits, dash, underscore, or dot");
    }
    Ok(())
}

fn validate_sha256_digest(sha256: &str) -> Result<()> {
    if !is_sha256_hex(sha256) {
        bail!("sha256 must be 64 hexadecimal characters");
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct ModelInventoryOutput {
    configured: usize,
    models: Vec<ModelInventoryItem>,
}

#[derive(Debug, Serialize)]
struct ModelInventoryItem {
    alias: String,
    role: String,
    weight: u32,
    path: String,
    updated_at: Option<chrono::DateTime<Utc>>,
    readiness: Option<rs_llmctl::readiness::ReadinessState>,
}

async fn model_inventory(cfg: &Config, storage: &Storage) -> Result<ModelInventoryOutput> {
    let persisted = storage
        .list_models()
        .await?
        .into_iter()
        .map(|record| (record.alias.clone(), record))
        .collect::<BTreeMap<_, _>>();

    let models = cfg
        .models
        .iter()
        .map(|model| {
            let persisted = persisted.get(&model.alias);
            ModelInventoryItem {
                alias: model.alias.clone(),
                role: model.role.clone(),
                weight: model.weight,
                path: path_basename(&model.path),
                updated_at: persisted.map(|record| record.updated_at),
                readiness: model
                    .family
                    .as_deref()
                    .filter(|family| family.eq_ignore_ascii_case("gemma4"))
                    .map(|_| {
                        rs_llmctl::readiness::read_state(&rs_llmctl::readiness::evidence_path(
                            &cfg.storage.model_dir,
                            &model.alias,
                        ))
                    }),
            }
        })
        .collect();

    Ok(ModelInventoryOutput {
        configured: cfg.models.len(),
        models,
    })
}

fn path_basename(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.display().to_string())
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
        key_id: None,
        key_owner: None,
        key_purpose: None,
        key_status: None,
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
            "headers": headers
                .into_iter()
                .map(|(key, value)| {
                    let rendered = if value.starts_with("env:") {
                        value
                    } else {
                        "[REDACTED]".to_string()
                    };
                    (key, rendered)
                })
                .collect::<BTreeMap<_, _>>(),
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

async fn audit_config_report(
    path: &Path,
    cfg: &Config,
    systemd_unit: Option<&Path>,
) -> Result<serde_json::Value> {
    let external_bind = cfg.security.bind_external || config::is_external_host(&cfg.server.host);
    let key_reports: Vec<_> = cfg
        .security
        .api_keys
        .iter()
        .map(|key| {
            json!({
                "id": key.id,
                "subject": key.subject,
                "team": key.team,
                "scopes": key.scopes,
                "sha256_present": !key.sha256.is_empty(),
                "sha256_valid": is_sha256_hex(&key.sha256)
            })
        })
        .collect();
    let hashed_api_keys = cfg
        .security
        .api_keys
        .iter()
        .all(|key| is_sha256_hex(&key.sha256));
    let secret_headers: Vec<_> = cfg
        .observability
        .exporter
        .headers
        .iter()
        .filter(|(name, _)| is_sensitive_name(name))
        .map(|(name, value)| {
            let value_source = if value.starts_with("env:") {
                "env"
            } else {
                "plaintext"
            };
            json!({
                "name": name,
                "value_source": value_source,
                "reference": if value.starts_with("env:") { Some(value.as_str()) } else { None }
            })
        })
        .collect();
    let systemd = systemd_audit(systemd_unit).await?;
    let trusted_proxy_reports: Vec<_> = cfg
        .security
        .trusted_proxies
        .iter()
        .map(|proxy| {
            let valid = trusted_proxy_is_explicit(proxy);
            json!({
                "value": proxy,
                "valid": valid,
                "wildcard": proxy.trim() == "*"
            })
        })
        .collect();
    let trusted_proxies_valid = trusted_proxy_reports
        .iter()
        .all(|proxy| proxy["valid"].as_bool().unwrap_or(false));

    let mut findings = Vec::new();
    if !hashed_api_keys {
        findings.push("api keys must be stored as sha256 hex digests".to_string());
    }
    if (cfg.security.production || external_bind)
        && (!cfg.security.require_auth || cfg.security.api_keys.is_empty())
    {
        findings.push("external/production serving requires authentication".to_string());
    }
    if cfg.security.production || external_bind {
        let native_tls = cfg.server.tls.enabled
            && cfg
                .server
                .tls
                .cert_path
                .as_ref()
                .is_some_and(|path| !path.as_os_str().is_empty())
            && cfg
                .server
                .tls
                .key_path
                .as_ref()
                .is_some_and(|path| !path.as_os_str().is_empty());
        if !native_tls && !cfg.security.tls_termination.enabled {
            findings.push(
                "external/production serving requires native TLS or documented TLS termination or mTLS"
                    .to_string(),
            );
        }
        if !native_tls
            && cfg
                .security
                .tls_termination
                .provider
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
        {
            findings.push("TLS termination must declare a provider".to_string());
        }
        if !native_tls
            && cfg
                .security
                .tls_termination
                .evidence
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
        {
            findings.push("TLS termination must declare evidence".to_string());
        }
        if !native_tls && cfg.security.trusted_proxies.is_empty() {
            findings.push("TLS termination must declare trusted-proxies".to_string());
        }
        if !native_tls && !trusted_proxies_valid {
            findings.push(
                "trusted-proxies must list explicit IP addresses or CIDR ranges; wildcard is not allowed"
                    .to_string(),
            );
        }
    }
    if cfg
        .observability
        .exporter
        .headers
        .iter()
        .any(|(name, value)| is_sensitive_name(name) && !value.starts_with("env:"))
    {
        findings.push("observability secret headers must use environment references".to_string());
    }
    if cfg.audit.retention_days == 0 {
        findings.push("audit retention must be greater than zero days".to_string());
    }
    if cfg.security.production || external_bind {
        if !cfg.audit.monthly_reports {
            findings
                .push("CRA Article 14 active control requires monthly audit reports".to_string());
        }
        if cfg
            .audit
            .report_directory
            .as_ref()
            .is_none_or(|path| path.as_os_str().is_empty())
        {
            findings
                .push("CRA Article 14 active control requires audit.report-directory".to_string());
        }
        if !(cfg.observability.traces_enabled
            && cfg.observability.metrics_enabled
            && cfg.observability.logs_enabled)
        {
            findings.push(
                "CRA Article 14 active control requires OTel traces, metrics, and logs".to_string(),
            );
        }
        if cfg
            .observability
            .exporter
            .endpoint
            .as_deref()
            .or(cfg.observability.otlp_endpoint.as_deref())
            .is_none_or(|endpoint| endpoint.trim().is_empty())
        {
            findings.push(
                "CRA Article 14 active control requires an OTel exporter endpoint".to_string(),
            );
        }
    }
    if systemd_unit.is_some()
        && (!systemd["present"].as_bool().unwrap_or(false)
            || !systemd["has_exec_start"].as_bool().unwrap_or(false))
    {
        findings.push("systemd unit template is missing or incomplete".to_string());
    }

    Ok(json!({
        "status": if findings.is_empty() { "ok" } else { "warning" },
        "config": path.file_name().and_then(|name| name.to_str()).unwrap_or("config.toml"),
        "config_path_redacted": redact_evidence_path(path),
        "external_bind": {
            "enabled": external_bind,
            "host": cfg.server.host,
            "port": cfg.server.port,
            "declared": cfg.security.bind_external
        },
        "auth": {
            "production": cfg.security.production,
            "require_auth": cfg.security.require_auth,
            "api_key_count": cfg.security.api_keys.len(),
            "hashed_api_keys": hashed_api_keys,
            "keys": key_reports
        },
        "tls_termination": {
            "enabled": cfg.security.tls_termination.enabled,
            "provider": cfg.security.tls_termination.provider.as_deref(),
            "evidence": cfg.security.tls_termination.evidence.as_deref(),
            "m_tls": cfg.security.tls_termination.m_tls,
            "trusted_proxies": {
                "count": cfg.security.trusted_proxies.len(),
                "valid": trusted_proxies_valid,
                "entries": trusted_proxy_reports
            }
        },
        "observability": {
            "endpoint_configured": cfg.observability.exporter.endpoint.is_some() || cfg.observability.otlp_endpoint.is_some(),
            "secret_header_count": secret_headers.len(),
            "secret_headers": secret_headers
        },
        "audit": {
            "retention_days": cfg.audit.retention_days,
            "report_directory": cfg.audit.report_directory.as_ref().and_then(|path| path.file_name()).and_then(|name| name.to_str()),
            "report_directory_redacted": cfg.audit.report_directory.as_ref().map(|path| redact_evidence_path(path)),
            "report_formats": cfg.audit.report_formats,
            "monthly_reports": cfg.audit.monthly_reports
        },
        "cra_article_14": {
            "regulation": "Regulation (EU) 2024/2847",
            "operational_status": "active_control",
            "monthly_reports": cfg.audit.monthly_reports,
            "otel_exporter_configured": cfg.observability.exporter.endpoint.is_some() || cfg.observability.otlp_endpoint.is_some(),
            "required_evidence": [
                "security audit-config",
                "audit report monthly --envelope",
                "audit report request --envelope",
                "data export --envelope",
                "compliance evidence"
            ]
        },
        "systemd": systemd,
        "findings": findings
    }))
}

async fn systemd_audit(systemd_unit: Option<&Path>) -> Result<serde_json::Value> {
    let Some(path) = systemd_unit else {
        return Ok(json!({
            "checked": false,
            "present": false,
            "artifact": null,
            "path_redacted": null,
            "has_service_section": false,
            "has_exec_start": false,
            "mentions_llmctld": false
        }));
    };

    match fs::read_to_string(path).await {
        Ok(body) => Ok(json!({
            "checked": true,
            "present": true,
            "artifact": path.file_name().and_then(|name| name.to_str()).unwrap_or("systemd-unit"),
            "path_redacted": redact_evidence_path(path),
            "has_service_section": body.contains("[Service]"),
            "has_exec_start": body.lines().any(|line| line.trim_start().starts_with("ExecStart=")),
            "mentions_llmctld": body.contains("llmctld")
        })),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(json!({
            "checked": true,
            "present": false,
            "artifact": path.file_name().and_then(|name| name.to_str()).unwrap_or("systemd-unit"),
            "path_redacted": redact_evidence_path(path),
            "has_service_section": false,
            "has_exec_start": false,
            "mentions_llmctld": false
        })),
        Err(err) => Err(err).with_context(|| format!("read systemd unit {}", path.display())),
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn trusted_proxy_is_explicit(value: &str) -> bool {
    let value = value.trim();
    if value.is_empty() || value == "*" {
        return false;
    }
    if let Some((addr, prefix)) = value.split_once('/') {
        let Ok(ip) = addr.parse::<std::net::IpAddr>() else {
            return false;
        };
        let Ok(prefix) = prefix.parse::<u8>() else {
            return false;
        };
        return prefix > 0 && prefix <= if ip.is_ipv4() { 32 } else { 128 };
    }
    value.parse::<std::net::IpAddr>().is_ok()
}

fn redact_evidence_path(path: &Path) -> String {
    format!(
        "<redacted>/{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("artifact")
    )
}

fn is_sensitive_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("authorization")
        || name.contains("api-key")
        || name.contains("apikey")
        || name.contains("token")
        || name.contains("secret")
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

impl From<CliLogFormat> for LogFormat {
    fn from(format: CliLogFormat) -> Self {
        match format {
            CliLogFormat::Pretty => LogFormat::Pretty,
            CliLogFormat::Json => LogFormat::Json,
        }
    }
}

impl From<CliEventFormat> for EventFormat {
    fn from(format: CliEventFormat) -> Self {
        match format {
            CliEventFormat::Json => EventFormat::Json,
            CliEventFormat::Jsonl => EventFormat::Jsonl,
            CliEventFormat::CloudEvents => EventFormat::CloudEvents,
        }
    }
}

impl From<CliDataFormat> for DataFabricFormat {
    fn from(format: CliDataFormat) -> Self {
        match format {
            CliDataFormat::Json => DataFabricFormat::Json,
            CliDataFormat::Jsonl => DataFabricFormat::Jsonl,
            CliDataFormat::ArrowJson => DataFabricFormat::ArrowJson,
            CliDataFormat::ArrowIpc => DataFabricFormat::ArrowIpc,
            CliDataFormat::Parquet => DataFabricFormat::Parquet,
        }
    }
}

impl DataDataset {
    fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Security => "security",
            Self::Observability => "observability",
            Self::Usage => "usage",
            Self::User => "user",
            Self::Finops => "finops",
            Self::Models => "models",
            Self::Drift => "drift",
            Self::Audit => "audit",
            Self::Lineage => "lineage",
        }
    }

    fn contract_kind(self) -> Option<DatasetKind> {
        match self {
            Self::All => None,
            Self::Security => Some(DatasetKind::Security),
            Self::Observability => Some(DatasetKind::Observability),
            Self::Usage => Some(DatasetKind::Usage),
            Self::User => Some(DatasetKind::User),
            Self::Finops => Some(DatasetKind::Finops),
            Self::Models => Some(DatasetKind::Models),
            Self::Drift => Some(DatasetKind::Drift),
            Self::Audit => Some(DatasetKind::Audit),
            Self::Lineage => Some(DatasetKind::Lineage),
        }
    }
}

impl From<DataContractDataset> for DatasetKind {
    fn from(dataset: DataContractDataset) -> Self {
        match dataset {
            DataContractDataset::Security => DatasetKind::Security,
            DataContractDataset::Observability => DatasetKind::Observability,
            DataContractDataset::Usage => DatasetKind::Usage,
            DataContractDataset::User => DatasetKind::User,
            DataContractDataset::Finops => DatasetKind::Finops,
            DataContractDataset::Models => DatasetKind::Models,
            DataContractDataset::Drift => DatasetKind::Drift,
            DataContractDataset::Audit => DatasetKind::Audit,
            DataContractDataset::Lineage => DatasetKind::Lineage,
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
    let matching_events = events
        .into_iter()
        .filter(|event| event.kind.contains(kind))
        .collect::<Vec<_>>();
    let values: Vec<f64> = matching_events.iter().map(|event| event.value).collect();
    let count = values.len();
    let avg_value = if count == 0 {
        None
    } else {
        Some(values.iter().sum::<f64>() / count as f64)
    };
    let max_value = values.iter().copied().reduce(f64::max);
    emit(
        as_json,
        &json!({ "kind": kind, "hours": hours, "count": count, "avg_value": avg_value, "max_value": max_value, "events": matching_events }),
    )
}

async fn report_usage(storage: &Storage, hours: i64, as_json: bool) -> Result<()> {
    let (from, to) = window(hours);
    let summary = reporting::usage_summary(storage, from, to).await?;
    emit(as_json, &json!({ "hours": hours, "summary": summary }))
}

async fn report_chargeback(
    storage: &Storage,
    args: UsageChargebackArgs,
    as_json: bool,
) -> Result<()> {
    let (from, to) = window(args.hours);
    let report = reporting::chargeback_report_filtered(
        storage,
        from,
        to,
        args.team.as_deref(),
        args.actor.as_deref(),
    )
    .await?;
    emit(
        as_json,
        &json!({
            "hours": args.hours,
            "from": report.from,
            "to": report.to,
            "generated_at": Utc::now(),
            "filters": {
                "team": report.team,
                "actor": report.actor
            },
            "usage_summary": report.usage_summary
        }),
    )
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
