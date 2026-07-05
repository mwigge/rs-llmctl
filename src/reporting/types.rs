//! Report envelope, metadata, and per-report-type data structures.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportKind {
    MonthlyAudit,
    PerRequestAudit,
    PerRequestData,
    DataExport,
    Chargeback,
    RetentionPlan,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub report_kind: ReportKind,
    pub generated_at: DateTime<Utc>,
    pub schema_version: u32,
    pub producer: String,
    pub contract_schema_version: u32,
    pub contract_hashes: BTreeMap<String, String>,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportEnvelope<T> {
    pub metadata: ReportMetadata,
    pub payload: T,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EnvelopeVerification {
    pub status: String,
    pub valid: bool,
    pub expected_sha256: Option<String>,
    pub actual_sha256: Option<String>,
    pub report_kind: Option<String>,
    pub schema_version: Option<u64>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthlyAuditReport {
    pub year: i32,
    pub month: u32,
    pub from: DateTime<Utc>,
    pub to: DateTime<Utc>,
    #[serde(default)]
    pub report_summary: ReportSummary,
    pub audit_events: Vec<AuditEvent>,
    pub usage_events: Vec<UsageEvent>,
    pub usage_summary: UsageSummary,
    pub quota_decisions: Vec<QuotaDecisionRecord>,
    pub observations: Vec<ObservationEvent>,
    pub models: Vec<ExportModelInventoryRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExportModelInventoryRecord {
    pub alias: String,
    pub role: String,
    pub weight: u32,
    pub updated_at: DateTime<Utc>,
}

impl From<ModelInventoryRecord> for ExportModelInventoryRecord {
    fn from(record: ModelInventoryRecord) -> Self {
        Self {
            alias: record.alias,
            role: record.role,
            weight: record.weight,
            updated_at: record.updated_at,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerRequestAuditReport {
    pub request_id: Uuid,
    pub audit_events: Vec<AuditEvent>,
    pub usage_events: Vec<UsageEvent>,
    pub quota_decisions: Vec<QuotaDecisionRecord>,
    pub observations: Vec<ObservationEvent>,
    pub usage_summary: UsageSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerRequestDataReport {
    pub request_id: Uuid,
    pub audit_events: Vec<AuditEvent>,
    pub usage_events: Vec<UsageEvent>,
    pub quota_decisions: Vec<QuotaDecisionRecord>,
    pub observations: Vec<ObservationEvent>,
    pub usage_summary: UsageSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChargebackReport {
    pub from: DateTime<Utc>,
    pub to: DateTime<Utc>,
    pub team: Option<String>,
    pub actor: Option<String>,
    pub usage_summary: UsageSummary,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct UsageSummary {
    pub request_count: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub total_latency_ms: u64,
    pub average_latency_ms: Option<f64>,
    pub by_model: Vec<UsageBreakdown>,
    pub by_team: Vec<UsageBreakdown>,
    pub by_actor: Vec<UsageBreakdown>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct UsageBreakdown {
    pub key: String,
    pub request_count: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub total_latency_ms: u64,
    pub average_latency_ms: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ReportSummary {
    pub audit_event_count: u64,
    pub usage_event_count: u64,
    pub observation_event_count: u64,
    pub model_record_count: u64,
    pub quota_decision_count: u64,
    #[serde(default)]
    pub lineage_join_count: u64,
    pub usage: UsageSummary,
}

impl ReportSummary {
    pub fn new(
        audit_event_count: u64,
        usage_event_count: u64,
        observation_event_count: u64,
        model_record_count: u64,
        quota_decision_count: u64,
        lineage_join_count: u64,
        usage: UsageSummary,
    ) -> Self {
        Self {
            audit_event_count,
            usage_event_count,
            observation_event_count,
            model_record_count,
            quota_decision_count,
            lineage_join_count,
            usage,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataExport {
    pub from: DateTime<Utc>,
    pub to: DateTime<Utc>,
    #[serde(default)]
    pub report_summary: ReportSummary,
    pub audit_events: Vec<AuditEvent>,
    pub usage_events: Vec<UsageEvent>,
    pub usage_summary: UsageSummary,
    pub observation_events: Vec<ObservationEvent>,
    pub quota_decisions: Vec<QuotaDecisionRecord>,
    pub models: Vec<ExportModelInventoryRecord>,
    #[serde(default)]
    pub lineage: Vec<RequestLineageJoinRecord>,
}
