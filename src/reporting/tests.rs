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
    assert_eq!(report.report_summary.audit_event_count, 1);
    assert_eq!(report.report_summary.usage_event_count, 1);
    assert_eq!(report.report_summary.observation_event_count, 1);
    assert_eq!(report.report_summary.model_record_count, 1);
    assert_eq!(report.report_summary.quota_decision_count, 1);
    assert_eq!(report.report_summary.usage.request_count, 1);
    assert_eq!(report.report_summary.usage.total_tokens, 30);

    let report_envelope = monthly_audit_report_envelope(&storage, now.year(), now.month()).await?;
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
    assert_eq!(export.report_summary.audit_event_count, 1);
    assert_eq!(export.report_summary.usage_event_count, 1);
    assert_eq!(export.report_summary.observation_event_count, 1);
    assert_eq!(export.report_summary.model_record_count, 1);
    assert_eq!(export.report_summary.quota_decision_count, 1);
    assert_eq!(export.report_summary.usage.by_model[0].key, "llama");
    assert_eq!(export.report_summary.usage.by_model[0].total_tokens, 30);

    let export_envelope = data_export_envelope(&storage, from, to).await?;
    assert_eq!(export_envelope.metadata.report_kind, ReportKind::DataExport);
    assert_eq!(
        export_envelope.metadata.sha256,
        canonical_sha256(&export_envelope.payload)?
    );
    Ok(())
}

#[test]
fn report_summary_serializes_only_aggregate_reporting_fields() -> Result<()> {
    let usage = summarize_usage(&[usage_event(
        Uuid::new_v4(),
        "llama",
        "alice",
        "platform",
        10,
        20,
        100,
    )]);
    let summary = ReportSummary::new(2, 1, 3, 4, 5, 6, usage);

    let value = serde_json::to_value(&summary)?;

    assert_eq!(value["audit_event_count"], 2);
    assert_eq!(value["usage_event_count"], 1);
    assert_eq!(value["observation_event_count"], 3);
    assert_eq!(value["lineage_join_count"], 6);
    assert_eq!(value["model_record_count"], 4);
    assert_eq!(value["quota_decision_count"], 5);
    assert_eq!(value["usage"]["total_tokens"], 30);
    assert!(value.get("audit_events").is_none());
    assert!(value.get("quota_decisions").is_none());
    assert!(value.get("detail_json").is_none());
    assert!(value.get("policy_json").is_none());
    Ok(())
}

#[tokio::test]
async fn builds_chargeback_report_for_usage_window() -> Result<()> {
    let storage = Storage::in_memory().await?;
    let from = Utc.with_ymd_and_hms(2026, 5, 1, 0, 0, 0).unwrap();
    let to = Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap();

    let mut platform_llama = usage_event(Uuid::new_v4(), "llama", "alice", "platform", 10, 20, 100);
    platform_llama.at = from;
    storage.insert_usage_event(&platform_llama).await?;

    let mut research_mistral = usage_event(Uuid::new_v4(), "mistral", "bob", "research", 5, 5, 200);
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

#[tokio::test]
async fn dataset_limited_export_queries_only_requested_dataset() -> Result<()> {
    let storage = Storage::in_memory().await?;
    let from = Utc::now() - chrono::Duration::hours(1);
    let to = Utc::now() + chrono::Duration::hours(1);
    for index in 0..3 {
        storage
            .insert_audit_event(&AuditEvent {
                id: Uuid::new_v4(),
                request_id: None,
                at: Utc::now(),
                actor: format!("actor-{index}"),
                team: "platform".to_string(),
                action: "test".to_string(),
                resource: "model/test".to_string(),
                outcome: "allowed".to_string(),
                detail_json: json!({}),
            })
            .await?;
    }
    storage
        .upsert_model_record(&ModelInventoryRecord {
            alias: "qwen".to_string(),
            path: "/models/qwen".to_string(),
            role: "chat".to_string(),
            weight: 1,
            updated_at: Utc::now(),
        })
        .await?;

    assert!(data_export_limited(&storage, from, to, Some(1))
        .await
        .is_err());
    let models =
        data_export_dataset_limited(&storage, from, to, contracts::DatasetKind::Models, Some(1))
            .await?;

    assert_eq!(models.models.len(), 1);
    assert!(models.audit_events.is_empty());
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
