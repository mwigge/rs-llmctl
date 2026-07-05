//! Quota, security, observe, and audit command definitions.
use super::*;

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
