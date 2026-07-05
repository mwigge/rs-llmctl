use crate::*;

pub(crate) async fn observe_command(
    path: &Path,
    command: ObserveCommand,
    as_json: bool,
) -> Result<()> {
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

pub(crate) async fn record_latency_drift_observations(
    storage: &Storage,
    hours: i64,
) -> Result<usize> {
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

pub(crate) async fn report_observations(
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

pub(crate) async fn report_usage(storage: &Storage, hours: i64, as_json: bool) -> Result<()> {
    let (from, to) = window(hours);
    let summary = reporting::usage_summary(storage, from, to).await?;
    emit(as_json, &json!({ "hours": hours, "summary": summary }))
}

pub(crate) async fn report_chargeback(
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

pub(crate) async fn show_observations(
    storage: &Storage,
    args: ObserveShowArgs,
    as_json: bool,
) -> Result<()> {
    let from = Utc::now() - Duration::days(3650);
    let mut events = storage.observation_events_between(from, Utc::now()).await?;
    if let Some(kind) = args.kind {
        events.retain(|event| event.kind == kind);
    }
    events.sort_by_key(|event| std::cmp::Reverse(event.at));
    events.truncate(args.limit.max(0) as usize);
    emit(as_json, &events)
}

pub(crate) fn window(hours: i64) -> (chrono::DateTime<Utc>, chrono::DateTime<Utc>) {
    let to = Utc::now();
    let from = if hours <= 0 {
        to - Duration::hours(24)
    } else {
        to - Duration::hours(hours)
    };
    (from, to)
}
