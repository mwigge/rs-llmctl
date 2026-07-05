use crate::audit::{AuditEvent, ObservationEvent, UsageEvent};
use crate::contracts;
use crate::storage::{
    ModelInventoryRecord, QuotaDecisionRecord, RequestLineageJoinRecord, Storage,
};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Datelike, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use uuid::Uuid;

mod types;
pub use types::*;
mod summary;
pub use summary::*;
mod envelope;
pub use envelope::*;

#[cfg(test)]
mod tests;

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
    let models = storage
        .list_models()
        .await?
        .into_iter()
        .map(ExportModelInventoryRecord::from)
        .collect::<Vec<_>>();
    let usage_summary = summarize_usage(&usage_events);
    let report_summary = summarize_report(
        audit_events.len(),
        usage_events.len(),
        observations.len(),
        models.len(),
        quota_decisions.len(),
        0,
        usage_summary.clone(),
    );

    Ok(MonthlyAuditReport {
        year,
        month,
        from,
        to,
        report_summary,
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
    data_export_limited(storage, from, to, None).await
}

pub async fn data_export_limited(
    storage: &Storage,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    max_rows: Option<usize>,
) -> Result<DataExport> {
    let query_limit = max_rows.map(|limit| limit.saturating_add(1));
    let audit_events = storage
        .audit_events_between_limited(from, to, query_limit)
        .await?;
    let usage_events = storage
        .usage_events_between_limited(from, to, query_limit)
        .await?;
    let observation_events = storage
        .observation_events_between_limited(from, to, query_limit)
        .await?;
    let quota_decisions = storage
        .quota_decisions_between_limited(from, to, query_limit)
        .await?;
    let lineage = storage
        .request_lineage_joins_between_limited(from, to, query_limit)
        .await?;
    if let Some(max_rows) = max_rows {
        ensure_limited("audit", audit_events.len(), max_rows)?;
        ensure_limited("usage", usage_events.len(), max_rows)?;
        ensure_limited("observability", observation_events.len(), max_rows)?;
        ensure_limited("quota", quota_decisions.len(), max_rows)?;
        ensure_limited("lineage", lineage.len(), max_rows)?;
    }
    let models = storage
        .list_models()
        .await?
        .into_iter()
        .map(ExportModelInventoryRecord::from)
        .collect::<Vec<_>>();
    let usage_summary = summarize_usage(&usage_events);
    let report_summary = summarize_report(
        audit_events.len(),
        usage_events.len(),
        observation_events.len(),
        models.len(),
        quota_decisions.len(),
        lineage.len(),
        usage_summary.clone(),
    );

    Ok(DataExport {
        from,
        to,
        report_summary,
        audit_events,
        usage_summary,
        usage_events,
        observation_events,
        quota_decisions,
        models,
        lineage,
    })
}

pub async fn data_export_dataset_limited(
    storage: &Storage,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    dataset: contracts::DatasetKind,
    max_rows: Option<usize>,
) -> Result<DataExport> {
    let query_limit = max_rows.map(|limit| limit.saturating_add(1));
    let mut audit_events = Vec::new();
    let mut usage_events = Vec::new();
    let mut observation_events = Vec::new();
    let mut quota_decisions = Vec::new();
    let mut lineage = Vec::new();
    let mut models = Vec::new();

    match dataset {
        contracts::DatasetKind::Security => {
            audit_events = storage
                .audit_events_between_limited(from, to, query_limit)
                .await?;
            quota_decisions = storage
                .quota_decisions_between_limited(from, to, query_limit)
                .await?;
            if let Some(max_rows) = max_rows {
                ensure_limited("audit", audit_events.len(), max_rows)?;
                ensure_limited("quota", quota_decisions.len(), max_rows)?;
            }
        }
        contracts::DatasetKind::Observability | contracts::DatasetKind::Drift => {
            observation_events = storage
                .observation_events_between_limited(from, to, query_limit)
                .await?;
            if let Some(max_rows) = max_rows {
                ensure_limited("observability", observation_events.len(), max_rows)?;
            }
        }
        contracts::DatasetKind::Usage
        | contracts::DatasetKind::User
        | contracts::DatasetKind::Finops => {
            usage_events = storage
                .usage_events_between_limited(from, to, query_limit)
                .await?;
            if let Some(max_rows) = max_rows {
                ensure_limited("usage", usage_events.len(), max_rows)?;
            }
        }
        contracts::DatasetKind::Models => {
            models = storage
                .list_models()
                .await?
                .into_iter()
                .map(ExportModelInventoryRecord::from)
                .collect();
            if let Some(max_rows) = max_rows {
                ensure_limited("models", models.len(), max_rows)?;
            }
        }
        contracts::DatasetKind::Lineage => {
            lineage = storage
                .request_lineage_joins_between_limited(from, to, query_limit)
                .await?;
            if let Some(max_rows) = max_rows {
                ensure_limited("lineage", lineage.len(), max_rows)?;
            }
        }
        contracts::DatasetKind::Audit => {
            audit_events = storage
                .audit_events_between_limited(from, to, query_limit)
                .await?;
            if let Some(max_rows) = max_rows {
                ensure_limited("audit", audit_events.len(), max_rows)?;
            }
        }
    }

    let usage_summary = summarize_usage(&usage_events);
    let report_summary = summarize_report(
        audit_events.len(),
        usage_events.len(),
        observation_events.len(),
        models.len(),
        quota_decisions.len(),
        lineage.len(),
        usage_summary.clone(),
    );

    Ok(DataExport {
        from,
        to,
        report_summary,
        audit_events,
        usage_summary,
        usage_events,
        observation_events,
        quota_decisions,
        models,
        lineage,
    })
}

fn ensure_limited(dataset: &str, count: usize, max_rows: usize) -> Result<()> {
    anyhow::ensure!(
        count <= max_rows,
        "data export dataset `{dataset}` exceeds max_rows {max_rows}; narrow the time window or raise --max-rows"
    );
    Ok(())
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

pub async fn data_export_envelope_limited(
    storage: &Storage,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    max_rows: Option<usize>,
) -> Result<ReportEnvelope<DataExport>> {
    let export = data_export_limited(storage, from, to, max_rows).await?;
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
