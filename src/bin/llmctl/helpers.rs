//! CLI helper utilities: JSONL/JSON file IO, key-file handling, redaction, and output DTOs.
use super::*;

#[derive(Debug, Default)]
pub(crate) struct ApiKeyUsageSummary {
    pub(crate) request_count: u64,
    pub(crate) audit_event_count: u64,
    pub(crate) error_count: u64,
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) total_tokens: u64,
    pub(crate) latency_ms: u64,
    pub(crate) last_seen: Option<chrono::DateTime<Utc>>,
    pub(crate) actors: std::collections::BTreeSet<String>,
    pub(crate) teams: std::collections::BTreeSet<String>,
    pub(crate) models: std::collections::BTreeSet<String>,
    pub(crate) statuses: std::collections::BTreeSet<String>,
    pub(crate) actions: std::collections::BTreeSet<String>,
    pub(crate) resources: std::collections::BTreeSet<String>,
}

pub(crate) fn redact_display_path(path: &Path) -> String {
    let rendered = path.display().to_string();
    std::env::var("HOME")
        .ok()
        .filter(|home| !home.is_empty())
        .and_then(|home| rendered.strip_prefix(&home).map(|tail| format!("~{tail}")))
        .unwrap_or(rendered)
}

pub(crate) async fn append_jsonl(path: &Path, value: &serde_json::Value) -> Result<()> {
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

pub(crate) async fn read_jsonl(path: &Path) -> Result<Vec<serde_json::Value>> {
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

pub(crate) fn state_file(cfg: &Config, name: &str) -> Result<PathBuf> {
    let dir = cfg
        .storage
        .db_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("storage db_path has no parent directory"))?;
    Ok(dir.join(name))
}

pub(crate) async fn write_json_file(path: &Path, value: &serde_json::Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .await
        .with_context(|| format!("write {}", path.display()))
}

pub(crate) async fn read_json_file(path: &Path) -> Result<serde_json::Value> {
    let bytes = fs::read(path)
        .await
        .with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

pub(crate) async fn restrict_private_key_file(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .await
            .with_context(|| format!("set private key permissions {}", path.display()))?;
    }
    Ok(())
}

pub(crate) async fn load_config(path: &Path) -> Result<Config> {
    config::load(path)
        .await
        .with_context(|| format!("load {}", path.display()))
}

pub(crate) async fn read_startup_plan(path: &Path) -> Result<StartupPlan> {
    let bytes = fs::read(path)
        .await
        .with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse startup plan {}", path.display()))
}

pub(crate) async fn create_storage_dirs(storage: &StorageConfig) -> Result<()> {
    let plan = storage.connection_plan()?;
    if plan.backend == rs_llmctl::storage::StorageBackend::Sqlite {
        if let Some(parent) = storage.db_path.parent() {
            fs::create_dir_all(parent).await?;
        }
    }
    fs::create_dir_all(&storage.model_dir).await?;
    Ok(())
}

pub(crate) async fn init_storage(storage: &StorageConfig) -> Result<Storage> {
    create_storage_dirs(storage).await?;
    Storage::connect_config(storage).await
}

pub(crate) async fn record_security_key_event(
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

pub(crate) fn upsert_model(models: &mut Vec<ModelConfig>, model: ModelConfig) {
    if let Some(existing) = models.iter_mut().find(|m| m.alias == model.alias) {
        *existing = model;
    } else {
        models.push(model);
    }
}

pub(crate) fn mode_name(mode: &Mode) -> &'static str {
    match mode {
        Mode::Single => "single",
        Mode::ColdSwap => "cold-swap",
        Mode::HotSwap => "hot-swap",
        Mode::Weighted => "weighted",
        Mode::Fallback => "fallback",
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ImportedQuotaPolicy {
    #[serde(default = "default_quota_policy_format")]
    pub(crate) format: String,
    #[serde(default)]
    pub(crate) quotas: Vec<QuotaConfig>,
}

pub(crate) fn validate_api_key_id(id: &str) -> Result<()> {
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

pub(crate) fn validate_sha256_digest(sha256: &str) -> Result<()> {
    if !is_sha256_hex(sha256) {
        bail!("sha256 must be 64 hexadecimal characters");
    }
    Ok(())
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelInventoryOutput {
    pub(crate) configured: usize,
    pub(crate) models: Vec<ModelInventoryItem>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelInventoryItem {
    pub(crate) alias: String,
    pub(crate) role: String,
    pub(crate) weight: u32,
    pub(crate) path: String,
    pub(crate) updated_at: Option<chrono::DateTime<Utc>>,
    pub(crate) readiness: Option<rs_llmctl::readiness::ReadinessState>,
}

pub(crate) fn observability_plan_json(plan: ObservabilityPlan) -> serde_json::Value {
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

pub(crate) fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub(crate) fn trusted_proxy_is_explicit(value: &str) -> bool {
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

pub(crate) fn redact_evidence_path(path: &Path) -> String {
    format!(
        "<redacted>/{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("artifact")
    )
}

pub(crate) fn is_sensitive_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("authorization")
        || name.contains("api-key")
        || name.contains("apikey")
        || name.contains("token")
        || name.contains("secret")
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
    pub(crate) fn as_str(self) -> &'static str {
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

    pub(crate) fn contract_kind(self) -> Option<DatasetKind> {
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
