//! Usage aggregation and month-window helpers.
use super::*;

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

pub fn summarize_report(
    audit_event_count: usize,
    usage_event_count: usize,
    observation_event_count: usize,
    model_record_count: usize,
    quota_decision_count: usize,
    lineage_join_count: usize,
    usage: UsageSummary,
) -> ReportSummary {
    ReportSummary::new(
        audit_event_count as u64,
        usage_event_count as u64,
        observation_event_count as u64,
        model_record_count as u64,
        quota_decision_count as u64,
        lineage_join_count as u64,
        usage,
    )
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
