use crate::audit::{AuditEvent, ObservationEvent, UsageEvent};
use crate::storage::{ModelInventoryRecord, QuotaDecisionRecord, Storage};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Datelike, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthlyAuditReport {
    pub year: i32,
    pub month: u32,
    pub from: DateTime<Utc>,
    pub to: DateTime<Utc>,
    pub generated_at: DateTime<Utc>,
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
    pub generated_at: DateTime<Utc>,
    pub audit_events: Vec<AuditEvent>,
    pub usage_events: Vec<UsageEvent>,
    pub quota_decisions: Vec<QuotaDecisionRecord>,
    pub observations: Vec<ObservationEvent>,
    pub usage_summary: UsageSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerRequestDataReport {
    pub request_id: Uuid,
    pub generated_at: DateTime<Utc>,
    pub audit_events: Vec<AuditEvent>,
    pub usage_events: Vec<UsageEvent>,
    pub quota_decisions: Vec<QuotaDecisionRecord>,
    pub observations: Vec<ObservationEvent>,
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
    pub generated_at: DateTime<Utc>,
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
        generated_at: Utc::now(),
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
        generated_at: Utc::now(),
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
        generated_at: Utc::now(),
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

pub async fn data_export(
    storage: &Storage,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> Result<DataExport> {
    let usage_events = storage.usage_events_between(from, to).await?;
    Ok(DataExport {
        from,
        to,
        generated_at: Utc::now(),
        audit_events: storage.audit_events_between(from, to).await?,
        usage_summary: summarize_usage(&usage_events),
        usage_events,
        observation_events: storage.observation_events_between(from, to).await?,
        quota_decisions: storage.quota_decisions_between(from, to).await?,
        models: storage.list_models().await?,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::{AuditEvent, ObservationEvent, UsageEvent};
    use crate::storage::{ModelInventoryRecord, QuotaDecisionRecord};
    use chrono::TimeZone;
    use serde_json::json;

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

        let request_report = per_request_audit_report(&storage, request_id).await?;
        assert_eq!(request_report.audit_events.len(), 1);
        assert_eq!(request_report.usage_events.len(), 1);
        assert_eq!(request_report.usage_summary.total_tokens, 30);
        assert_eq!(request_report.quota_decisions.len(), 1);
        assert_eq!(request_report.observations.len(), 1);

        let data_report = per_request_data_report(&storage, request_id).await?;
        assert_eq!(data_report.request_id, request_id);
        assert_eq!(data_report.usage_summary.average_latency_ms, Some(100.0));
        assert_eq!(data_report.observations[0].request_id, Some(request_id));

        let (from, to) = current_month_bounds()?;
        let export = data_export(&storage, from, to).await?;
        assert_eq!(export.audit_events.len(), 1);
        assert_eq!(export.usage_events.len(), 1);
        assert_eq!(export.usage_summary.request_count, 1);
        assert_eq!(export.observation_events.len(), 1);
        assert_eq!(export.quota_decisions.len(), 1);
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
