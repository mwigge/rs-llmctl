//! Usage, data, aiops, eval, lineage, and policy command definitions.
use super::*;

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
