//! Model, swap, service, and runtime command argument definitions.
use super::*;

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
