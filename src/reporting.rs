use crate::audit::{AuditEvent, ObservationEvent, UsageEvent};
use crate::storage::{ModelInventoryRecord, QuotaDecisionRecord, Storage};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Datelike, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use uuid::Uuid;

const REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportKind {
    MonthlyAudit,
    PerRequestAudit,
    PerRequestData,
    DataExport,
    Chargeback,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub report_kind: ReportKind,
    pub generated_at: DateTime<Utc>,
    pub schema_version: u32,
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
    pub audit_events: Vec<AuditEvent>,
    pub usage_events: Vec<UsageEvent>,
    pub usage_summary: UsageSummary,
    pub quota_decisions: Vec<QuotaDecisionRecord>,
    pub observations: Vec<ObservationEvent>,
    pub models: Vec<ModelInventoryRecord>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataExport {
    pub from: DateTime<Utc>,
    pub to: DateTime<Utc>,
    pub audit_events: Vec<AuditEvent>,
    pub usage_events: Vec<UsageEvent>,
    pub usage_summary: UsageSummary,
    pub observation_events: Vec<ObservationEvent>,
    pub quota_decisions: Vec<QuotaDecisionRecord>,
    pub models: Vec<ModelInventoryRecord>,
}

pub async fn monthly_audit_report(
    storage: &Storage,
    year: i32,
    month: u32,
) -> Result<MonthlyAuditReport> {
    let (from, to) = month_bounds(year, month)?;
    let audit_events = storage.audit_events_between(from, to).await?;
    let usage_events = storage.usage_events_between(from, to).await?;
    let quota_decisions = storage.quota_decisions_between(from, to).await?;
    let observations = storage.observation_events_between(from, to).await?;
    let models = storage.list_models().await?;
    let usage_summary = summarize_usage(&usage_events);

    Ok(MonthlyAuditReport {
        year,
        month,
        from,
        to,
        audit_events,
        usage_events,
        usage_summary,
        quota_decisions,
        observations,
        models,
    })
}

pub async fn per_request_audit_report(
    storage: &Storage,
    request_id: Uuid,
) -> Result<PerRequestAuditReport> {
    let usage_events = storage.usage_events_for_request(request_id).await?;
    Ok(PerRequestAuditReport {
        request_id,
        audit_events: storage.audit_events_for_request(request_id).await?,
        usage_summary: summarize_usage(&usage_events),
        usage_events,
        quota_decisions: storage.quota_decisions_for_request(request_id).await?,
        observations: storage.observation_events_for_request(request_id).await?,
    })
}

pub async fn per_request_data_report(
    storage: &Storage,
    request_id: Uuid,
) -> Result<PerRequestDataReport> {
    let usage_events = storage.usage_events_for_request(request_id).await?;
    Ok(PerRequestDataReport {
        request_id,
        audit_events: storage.audit_events_for_request(request_id).await?,
        usage_summary: summarize_usage(&usage_events),
        usage_events,
        quota_decisions: storage.quota_decisions_for_request(request_id).await?,
        observations: storage.observation_events_for_request(request_id).await?,
    })
}

pub async fn usage_summary(
    storage: &Storage,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> Result<UsageSummary> {
    let usage_events = storage.usage_events_between(from, to).await?;
    Ok(summarize_usage(&usage_events))
}

pub async fn chargeback_report(
    storage: &Storage,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> Result<ChargebackReport> {
    chargeback_report_filtered(storage, from, to, None, None).await
}

pub async fn chargeback_report_filtered(
    storage: &Storage,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    team: Option<&str>,
    actor: Option<&str>,
) -> Result<ChargebackReport> {
    let mut usage_events = storage.usage_events_between(from, to).await?;
    if let Some(team) = team {
        usage_events.retain(|event| event.team == team);
    }
    if let Some(actor) = actor {
        usage_events.retain(|event| event.actor == actor);
    }

    Ok(ChargebackReport {
        from,
        to,
        team: team.map(str::to_string),
        actor: actor.map(str::to_string),
        usage_summary: summarize_usage(&usage_events),
    })
}

pub async fn data_export(
    storage: &Storage,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> Result<DataExport> {
    let usage_events = storage.usage_events_between(from, to).await?;
    Ok(DataExport {
        from,
        to,
        audit_events: storage.audit_events_between(from, to).await?,
        usage_summary: summarize_usage(&usage_events),
        usage_events,
        observation_events: storage.observation_events_between(from, to).await?,
        quota_decisions: storage.quota_decisions_between(from, to).await?,
        models: storage.list_models().await?,
    })
}

pub async fn monthly_audit_report_envelope(
    storage: &Storage,
    year: i32,
    month: u32,
) -> Result<ReportEnvelope<MonthlyAuditReport>> {
    let report = monthly_audit_report(storage, year, month).await?;
    report_envelope(ReportKind::MonthlyAudit, report)
}

pub async fn per_request_audit_report_envelope(
    storage: &Storage,
    request_id: Uuid,
) -> Result<ReportEnvelope<PerRequestAuditReport>> {
    let report = per_request_audit_report(storage, request_id).await?;
    report_envelope(ReportKind::PerRequestAudit, report)
}

pub async fn per_request_data_report_envelope(
    storage: &Storage,
    request_id: Uuid,
) -> Result<ReportEnvelope<PerRequestDataReport>> {
    let report = per_request_data_report(storage, request_id).await?;
    report_envelope(ReportKind::PerRequestData, report)
}

pub async fn data_export_envelope(
    storage: &Storage,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> Result<ReportEnvelope<DataExport>> {
    let export = data_export(storage, from, to).await?;
    report_envelope(ReportKind::DataExport, export)
}

pub async fn chargeback_report_envelope(
    storage: &Storage,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> Result<ReportEnvelope<ChargebackReport>> {
    let report = chargeback_report(storage, from, to).await?;
    report_envelope(ReportKind::Chargeback, report)
}

pub fn report_envelope<T>(report_kind: ReportKind, payload: T) -> Result<ReportEnvelope<T>>
where
    T: Serialize,
{
    report_envelope_at(report_kind, payload, Utc::now())
}

pub fn canonical_sha256<T>(payload: &T) -> Result<String>
where
    T: Serialize,
{
    let canonical = canonical_json(payload)?;
    Ok(hex::encode(Sha256::digest(canonical.as_bytes())))
}

pub fn canonical_json<T>(payload: &T) -> Result<String>
where
    T: Serialize,
{
    let value = serde_json::to_value(payload)?;
    Ok(serde_json::to_string(&canonicalize_value(value))?)
}

pub fn verify_envelope_value(envelope: &Value) -> Result<EnvelopeVerification> {
    let metadata = match envelope.get("metadata").and_then(Value::as_object) {
        Some(metadata) => metadata,
        None => {
            return Ok(invalid_envelope(
                None,
                None,
                None,
                "missing metadata object",
            ))
        }
    };
    let payload = match envelope.get("payload") {
        Some(payload) => payload,
        None => {
            return Ok(invalid_envelope(
                expected_sha256(metadata),
                report_kind(metadata),
                schema_version(metadata),
                "missing payload",
            ))
        }
    };
    let expected = match expected_sha256(metadata) {
        Some(expected) => expected,
        None => {
            return Ok(invalid_envelope(
                None,
                report_kind(metadata),
                schema_version(metadata),
                "missing metadata sha256",
            ))
        }
    };
    let actual = canonical_sha256(payload)?;
    let valid = expected.eq_ignore_ascii_case(&actual);

    Ok(EnvelopeVerification {
        status: if valid { "valid" } else { "invalid" }.to_string(),
        valid,
        expected_sha256: Some(expected),
        actual_sha256: Some(actual),
        report_kind: report_kind(metadata),
        schema_version: schema_version(metadata),
        reason: if valid {
            None
        } else {
            Some("payload sha256 mismatch".to_string())
        },
    })
}

pub fn summarize_usage(events: &[UsageEvent]) -> UsageSummary {
    let mut summary = UsageSummary::default();
    let mut by_model = BTreeMap::new();
    let mut by_team = BTreeMap::new();
    let mut by_actor = BTreeMap::new();

    for event in events {
        accumulate_summary(&mut summary, event);
        accumulate_breakdown(&mut by_model, &event.model, event);
        accumulate_breakdown(&mut by_team, &event.team, event);
        accumulate_breakdown(&mut by_actor, &event.actor, event);
    }

    summary.average_latency_ms = average(summary.total_latency_ms, summary.request_count);
    summary.by_model = finish_breakdowns(by_model);
    summary.by_team = finish_breakdowns(by_team);
    summary.by_actor = finish_breakdowns(by_actor);
    summary
}

pub fn month_bounds(year: i32, month: u32) -> Result<(DateTime<Utc>, DateTime<Utc>)> {
    if !(1..=12).contains(&month) {
        return Err(anyhow!("month must be in 1..=12, got {month}"));
    }

    let from = Utc
        .with_ymd_and_hms(year, month, 1, 0, 0, 0)
        .single()
        .ok_or_else(|| anyhow!("invalid month {year:04}-{month:02}"))?;
    let (next_year, next_month) = if month == 12 {
        (year + 1, 1)
    } else {
        (year, month + 1)
    };
    let to = Utc
        .with_ymd_and_hms(next_year, next_month, 1, 0, 0, 0)
        .single()
        .ok_or_else(|| anyhow!("invalid next month {next_year:04}-{next_month:02}"))?;
    Ok((from, to))
}

pub fn current_month_bounds() -> Result<(DateTime<Utc>, DateTime<Utc>)> {
    let now = Utc::now();
    month_bounds(now.year(), now.month())
}

fn accumulate_summary(summary: &mut UsageSummary, event: &UsageEvent) {
    summary.request_count += 1;
    summary.input_tokens = summary.input_tokens.saturating_add(event.input_tokens);
    summary.output_tokens = summary.output_tokens.saturating_add(event.output_tokens);
    summary.total_tokens = summary
        .total_tokens
        .saturating_add(event.input_tokens.saturating_add(event.output_tokens));
    summary.total_latency_ms = summary.total_latency_ms.saturating_add(event.latency_ms);
}

fn accumulate_breakdown(
    breakdowns: &mut BTreeMap<String, UsageBreakdown>,
    key: &str,
    event: &UsageEvent,
) {
    let breakdown = breakdowns
        .entry(key.to_string())
        .or_insert_with(|| UsageBreakdown {
            key: key.to_string(),
            ..UsageBreakdown::default()
        });
    breakdown.request_count += 1;
    breakdown.input_tokens = breakdown.input_tokens.saturating_add(event.input_tokens);
    breakdown.output_tokens = breakdown.output_tokens.saturating_add(event.output_tokens);
    breakdown.total_tokens = breakdown
        .total_tokens
        .saturating_add(event.input_tokens.saturating_add(event.output_tokens));
    breakdown.total_latency_ms = breakdown.total_latency_ms.saturating_add(event.latency_ms);
}

fn finish_breakdowns(breakdowns: BTreeMap<String, UsageBreakdown>) -> Vec<UsageBreakdown> {
    breakdowns
        .into_values()
        .map(|mut breakdown| {
            breakdown.average_latency_ms =
                average(breakdown.total_latency_ms, breakdown.request_count);
            breakdown
        })
        .collect()
}

fn average(total: u64, count: u64) -> Option<f64> {
    if count == 0 {
        None
    } else {
        Some(total as f64 / count as f64)
    }
}

fn invalid_envelope(
    expected_sha256: Option<String>,
    report_kind: Option<String>,
    schema_version: Option<u64>,
    reason: &str,
) -> EnvelopeVerification {
    EnvelopeVerification {
        status: "invalid".to_string(),
        valid: false,
        expected_sha256,
        actual_sha256: None,
        report_kind,
        schema_version,
        reason: Some(reason.to_string()),
    }
}

fn expected_sha256(metadata: &Map<String, Value>) -> Option<String> {
    metadata
        .get("sha256")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn report_kind(metadata: &Map<String, Value>) -> Option<String> {
    metadata
        .get("report_kind")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn schema_version(metadata: &Map<String, Value>) -> Option<u64> {
    metadata.get("schema_version").and_then(Value::as_u64)
}

fn report_envelope_at<T>(
    report_kind: ReportKind,
    payload: T,
    generated_at: DateTime<Utc>,
) -> Result<ReportEnvelope<T>>
where
    T: Serialize,
{
    Ok(ReportEnvelope {
        metadata: ReportMetadata {
            report_kind,
            generated_at,
            schema_version: REPORT_SCHEMA_VERSION,
            sha256: canonical_sha256(&payload)?,
        },
        payload,
    })
}

fn canonicalize_value(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize_value).collect()),
        Value::Object(values) => {
            let mut sorted = Map::new();
            let mut entries: Vec<_> = values.into_iter().collect();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            for (key, value) in entries {
                sorted.insert(key, canonicalize_value(value));
            }
            Value::Object(sorted)
        }
        scalar => scalar,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::{AuditEvent, ObservationEvent, UsageEvent};
    use crate::storage::{ModelInventoryRecord, QuotaDecisionRecord};
    use chrono::TimeZone;
    use serde_json::{json, Value};

    #[test]
    fn summarizes_usage_by_total_model_team_and_actor() {
        let request_id = Uuid::new_v4();
        let events = vec![
            usage_event(request_id, "llama", "alice", "platform", 10, 20, 100),
            usage_event(request_id, "llama", "bob", "platform", 1, 2, 50),
            usage_event(request_id, "mistral", "alice", "research", 4, 5, 150),
        ];

        let summary = summarize_usage(&events);

        assert_eq!(summary.request_count, 3);
        assert_eq!(summary.input_tokens, 15);
        assert_eq!(summary.output_tokens, 27);
        assert_eq!(summary.total_tokens, 42);
        assert_eq!(summary.average_latency_ms, Some(100.0));
        assert_eq!(summary.by_model[0].key, "llama");
        assert_eq!(summary.by_model[0].request_count, 2);
        assert_eq!(summary.by_team[0].key, "platform");
        assert_eq!(summary.by_actor[0].key, "alice");
    }

    #[test]
    fn builds_month_bounds() -> Result<()> {
        let (from, to) = month_bounds(2026, 2)?;
        assert_eq!(from, Utc.with_ymd_and_hms(2026, 2, 1, 0, 0, 0).unwrap());
        assert_eq!(to, Utc.with_ymd_and_hms(2026, 3, 1, 0, 0, 0).unwrap());
        assert!(month_bounds(2026, 13).is_err());
        Ok(())
    }

    #[test]
    fn canonical_json_hash_is_stable_for_object_key_order() -> Result<()> {
        let left: Value = serde_json::from_str(r#"{"b":2,"a":{"d":4,"c":3}}"#)?;
        let right: Value = serde_json::from_str(r#"{"a":{"c":3,"d":4},"b":2}"#)?;

        assert_eq!(canonical_json(&left)?, canonical_json(&right)?);
        assert_eq!(canonical_sha256(&left)?, canonical_sha256(&right)?);
        Ok(())
    }

    #[test]
    fn canonical_json_hash_changes_when_payload_changes() -> Result<()> {
        let original = json!({"request_count": 1, "status": "ok"});
        let changed = json!({"request_count": 2, "status": "ok"});

        assert_ne!(canonical_sha256(&original)?, canonical_sha256(&changed)?);
        Ok(())
    }

    #[test]
    fn report_envelope_hashes_payload_and_keeps_generated_metadata() -> Result<()> {
        let generated_at = Utc.with_ymd_and_hms(2026, 5, 16, 12, 0, 0).unwrap();
        let payload = json!({"month": 5, "year": 2026});

        let envelope = report_envelope_at(ReportKind::MonthlyAudit, payload.clone(), generated_at)?;

        assert_eq!(envelope.metadata.report_kind, ReportKind::MonthlyAudit);
        assert_eq!(envelope.metadata.generated_at, generated_at);
        assert_eq!(envelope.metadata.schema_version, 1);
        assert_eq!(envelope.metadata.sha256, canonical_sha256(&payload)?);
        Ok(())
    }

    #[test]
    fn report_envelope_hash_is_stable_across_generation_times() -> Result<()> {
        let payload = json!({"request_id": "018f9c40-1a2b-7320-bc4f-111111111111"});
        let first = report_envelope_at(
            ReportKind::PerRequestData,
            payload.clone(),
            Utc.with_ymd_and_hms(2026, 5, 16, 12, 0, 0).unwrap(),
        )?;
        let second = report_envelope_at(
            ReportKind::PerRequestData,
            payload,
            Utc.with_ymd_and_hms(2026, 5, 16, 13, 0, 0).unwrap(),
        )?;

        assert_ne!(first.metadata.generated_at, second.metadata.generated_at);
        assert_eq!(first.metadata.sha256, second.metadata.sha256);
        Ok(())
    }

    #[test]
    fn verify_envelope_value_accepts_matching_payload_hash() -> Result<()> {
        let payload = json!({"month": 5, "year": 2026});
        let envelope = json!({
            "metadata": {
                "report_kind": "monthly_audit",
                "generated_at": "2026-05-16T12:00:00Z",
                "schema_version": 1,
                "sha256": canonical_sha256(&payload)?
            },
            "payload": payload
        });

        let verified = verify_envelope_value(&envelope)?;

        assert!(verified.valid);
        assert_eq!(verified.status, "valid");
        assert_eq!(verified.report_kind.as_deref(), Some("monthly_audit"));
        assert_eq!(verified.schema_version, Some(1));
        assert_eq!(verified.reason, None);
        Ok(())
    }

    #[test]
    fn verify_envelope_value_rejects_tampered_payload_hash() -> Result<()> {
        let original = json!({"request_count": 1});
        let envelope = json!({
            "metadata": {
                "report_kind": "data_export",
                "generated_at": "2026-05-16T12:00:00Z",
                "schema_version": 1,
                "sha256": canonical_sha256(&original)?
            },
            "payload": {"request_count": 2}
        });

        let verified = verify_envelope_value(&envelope)?;

        assert!(!verified.valid);
        assert_eq!(verified.status, "invalid");
        assert_eq!(verified.reason.as_deref(), Some("payload sha256 mismatch"));
        assert_ne!(verified.expected_sha256, verified.actual_sha256);
        Ok(())
    }

    #[tokio::test]
    async fn builds_reports_and_export_from_storage() -> Result<()> {
        let storage = Storage::in_memory().await?;
        let request_id = Uuid::new_v4();
        storage
            .insert_audit_event(&AuditEvent::new(
                Some(request_id),
                "alice",
                "platform",
                "chat.create",
                "model/llama",
                "allow",
                json!({}),
            ))
            .await?;
        storage
            .insert_usage_event(&usage_event(
                request_id, "llama", "alice", "platform", 10, 20, 100,
            ))
            .await?;
        storage
            .insert_observation_event(&ObservationEvent {
                id: Uuid::new_v4(),
                request_id: Some(request_id),
                at: Utc::now(),
                kind: "latency".to_string(),
                model: "llama".to_string(),
                source: "worker".to_string(),
                value: 100.0,
                unit: "ms".to_string(),
                attributes_json: json!({"trace": "abc"}),
            })
            .await?;
        storage
            .upsert_model_record(&ModelInventoryRecord {
                alias: "llama".to_string(),
                path: "/models/llama.gguf".to_string(),
                role: "chat".to_string(),
                weight: 1,
                updated_at: Utc::now(),
            })
            .await?;
        storage
            .insert_quota_decision(&QuotaDecisionRecord {
                id: Uuid::new_v4(),
                request_id: Some(request_id),
                at: Utc::now(),
                actor: "alice".to_string(),
                team: "platform".to_string(),
                model: "llama".to_string(),
                allowed: true,
                reason: "ok".to_string(),
                policy_json: json!({}),
            })
            .await?;

        let now = Utc::now();
        let report = monthly_audit_report(&storage, now.year(), now.month()).await?;
        assert_eq!(report.audit_events.len(), 1);
        assert_eq!(report.usage_events.len(), 1);
        assert_eq!(report.usage_summary.request_count, 1);
        assert_eq!(report.quota_decisions.len(), 1);
        assert_eq!(report.observations.len(), 1);
        assert_eq!(report.models.len(), 1);

        let report_envelope =
            monthly_audit_report_envelope(&storage, now.year(), now.month()).await?;
        assert_eq!(
            report_envelope.metadata.report_kind,
            ReportKind::MonthlyAudit
        );
        assert_eq!(
            report_envelope.metadata.sha256,
            canonical_sha256(&report_envelope.payload)?
        );

        let request_report = per_request_audit_report(&storage, request_id).await?;
        assert_eq!(request_report.audit_events.len(), 1);
        assert_eq!(request_report.usage_events.len(), 1);
        assert_eq!(request_report.usage_summary.total_tokens, 30);
        assert_eq!(request_report.quota_decisions.len(), 1);
        assert_eq!(request_report.observations.len(), 1);

        let request_envelope = per_request_audit_report_envelope(&storage, request_id).await?;
        assert_eq!(
            request_envelope.metadata.report_kind,
            ReportKind::PerRequestAudit
        );
        assert_eq!(
            request_envelope.metadata.sha256,
            canonical_sha256(&request_envelope.payload)?
        );

        let data_report = per_request_data_report(&storage, request_id).await?;
        assert_eq!(data_report.request_id, request_id);
        assert_eq!(data_report.usage_summary.average_latency_ms, Some(100.0));
        assert_eq!(data_report.observations[0].request_id, Some(request_id));

        let data_report_envelope = per_request_data_report_envelope(&storage, request_id).await?;
        assert_eq!(
            data_report_envelope.metadata.report_kind,
            ReportKind::PerRequestData
        );
        assert_eq!(
            data_report_envelope.metadata.sha256,
            canonical_sha256(&data_report_envelope.payload)?
        );

        let (from, to) = current_month_bounds()?;
        let export = data_export(&storage, from, to).await?;
        assert_eq!(export.audit_events.len(), 1);
        assert_eq!(export.usage_events.len(), 1);
        assert_eq!(export.usage_summary.request_count, 1);
        assert_eq!(export.observation_events.len(), 1);
        assert_eq!(export.quota_decisions.len(), 1);

        let export_envelope = data_export_envelope(&storage, from, to).await?;
        assert_eq!(export_envelope.metadata.report_kind, ReportKind::DataExport);
        assert_eq!(
            export_envelope.metadata.sha256,
            canonical_sha256(&export_envelope.payload)?
        );
        Ok(())
    }

    #[tokio::test]
    async fn builds_chargeback_report_for_usage_window() -> Result<()> {
        let storage = Storage::in_memory().await?;
        let from = Utc.with_ymd_and_hms(2026, 5, 1, 0, 0, 0).unwrap();
        let to = Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap();

        let mut platform_llama =
            usage_event(Uuid::new_v4(), "llama", "alice", "platform", 10, 20, 100);
        platform_llama.at = from;
        storage.insert_usage_event(&platform_llama).await?;

        let mut research_mistral =
            usage_event(Uuid::new_v4(), "mistral", "bob", "research", 5, 5, 200);
        research_mistral.at = from + chrono::Duration::days(1);
        storage.insert_usage_event(&research_mistral).await?;

        let mut outside_window =
            usage_event(Uuid::new_v4(), "llama", "carol", "platform", 100, 100, 500);
        outside_window.at = to;
        storage.insert_usage_event(&outside_window).await?;

        let report = chargeback_report(&storage, from, to).await?;

        assert_eq!(report.from, from);
        assert_eq!(report.to, to);
        assert_eq!(report.usage_summary.request_count, 2);
        assert_eq!(report.usage_summary.total_tokens, 40);
        assert_eq!(report.usage_summary.average_latency_ms, Some(150.0));
        assert_eq!(report.usage_summary.by_team[0].key, "platform");
        assert_eq!(report.usage_summary.by_team[0].total_tokens, 30);
        assert_eq!(report.usage_summary.by_actor[0].key, "alice");
        assert_eq!(report.usage_summary.by_model[1].key, "mistral");

        let envelope = chargeback_report_envelope(&storage, from, to).await?;
        assert_eq!(envelope.metadata.report_kind, ReportKind::Chargeback);
        assert_eq!(envelope.payload, report);
        assert_eq!(
            envelope.metadata.sha256,
            canonical_sha256(&envelope.payload)?
        );
        Ok(())
    }

    fn usage_event(
        request_id: Uuid,
        model: &str,
        actor: &str,
        team: &str,
        input_tokens: u64,
        output_tokens: u64,
        latency_ms: u64,
    ) -> UsageEvent {
        UsageEvent {
            id: Uuid::new_v4(),
            request_id,
            at: Utc::now(),
            model: model.to_string(),
            actor: actor.to_string(),
            team: team.to_string(),
            input_tokens,
            output_tokens,
            latency_ms,
            status: "ok".to_string(),
        }
    }
}
