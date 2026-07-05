use crate::config::QuotaConfig;
use crate::storage::{QuotaDecisionRecord, Storage};
use anyhow::Result;
use chrono::{Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Principal {
    pub subject: String,
    pub team: String,
    pub scopes: Vec<String>,
    pub key_id: Option<String>,
    pub key_owner: Option<String>,
    pub key_purpose: Option<String>,
    pub key_status: Option<String>,
}

impl Principal {
    pub fn anonymous() -> Self {
        Self {
            subject: "anonymous".to_string(),
            team: "public".to_string(),
            scopes: vec!["chat".to_string(), "models.read".to_string()],
            key_id: None,
            key_owner: None,
            key_purpose: None,
            key_status: None,
        }
    }

    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.iter().any(|s| {
            s == scope
                || s == "admin"
                || (scope.starts_with("models.") && s == "models")
                || (scope.starts_with("chat.") && s == "chat")
        })
    }
}

#[derive(Debug, Clone)]
pub struct QuotaDecision {
    pub allowed: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuotaPolicyReport {
    pub policy_count: usize,
    pub total_requests_per_minute: u64,
    pub total_tokens_per_day: u64,
    pub total_max_concurrency: u64,
    pub by_team: Vec<QuotaTeamPolicySummary>,
    pub by_subject: Vec<QuotaSubjectPolicySummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuotaTeamPolicySummary {
    pub team: String,
    pub policy_count: usize,
    pub subjects: Vec<String>,
    pub allowed_models: Vec<String>,
    pub total_requests_per_minute: u64,
    pub total_tokens_per_day: u64,
    pub total_max_concurrency: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuotaSubjectPolicySummary {
    pub subject: String,
    pub team: String,
    pub policy_count: usize,
    pub allowed_models: Vec<String>,
    pub total_requests_per_minute: u64,
    pub total_tokens_per_day: u64,
    pub total_max_concurrency: u64,
}

pub fn summarize_quota_policies(quotas: &[QuotaConfig]) -> QuotaPolicyReport {
    let mut report = QuotaPolicyReport {
        policy_count: quotas.len(),
        total_requests_per_minute: 0,
        total_tokens_per_day: 0,
        total_max_concurrency: 0,
        by_team: Vec::new(),
        by_subject: Vec::new(),
    };
    let mut teams = BTreeMap::<String, QuotaPolicyAccumulator>::new();
    let mut subjects = BTreeMap::<(String, String), QuotaPolicyAccumulator>::new();

    for quota in quotas {
        report.total_requests_per_minute = report
            .total_requests_per_minute
            .saturating_add(u64::from(quota.requests_per_minute));
        report.total_tokens_per_day = report
            .total_tokens_per_day
            .saturating_add(quota.tokens_per_day);
        report.total_max_concurrency = report
            .total_max_concurrency
            .saturating_add(u64::from(quota.max_concurrency));

        let team = teams.entry(quota.team.clone()).or_default();
        team.add_quota(quota);
        team.subjects.insert(quota.subject.clone());

        subjects
            .entry((quota.subject.clone(), quota.team.clone()))
            .or_default()
            .add_quota(quota);
    }

    report.by_team = teams
        .into_iter()
        .map(|(team, accumulator)| QuotaTeamPolicySummary {
            team,
            policy_count: accumulator.policy_count,
            subjects: accumulator.subjects.into_iter().collect(),
            allowed_models: accumulator.allowed_models.into_iter().collect(),
            total_requests_per_minute: accumulator.total_requests_per_minute,
            total_tokens_per_day: accumulator.total_tokens_per_day,
            total_max_concurrency: accumulator.total_max_concurrency,
        })
        .collect();
    report.by_subject = subjects
        .into_iter()
        .map(|((subject, team), accumulator)| QuotaSubjectPolicySummary {
            subject,
            team,
            policy_count: accumulator.policy_count,
            allowed_models: accumulator.allowed_models.into_iter().collect(),
            total_requests_per_minute: accumulator.total_requests_per_minute,
            total_tokens_per_day: accumulator.total_tokens_per_day,
            total_max_concurrency: accumulator.total_max_concurrency,
        })
        .collect();
    report
}

#[derive(Debug, Default)]
struct QuotaPolicyAccumulator {
    policy_count: usize,
    subjects: BTreeSet<String>,
    allowed_models: BTreeSet<String>,
    total_requests_per_minute: u64,
    total_tokens_per_day: u64,
    total_max_concurrency: u64,
}

impl QuotaPolicyAccumulator {
    fn add_quota(&mut self, quota: &QuotaConfig) {
        self.policy_count += 1;
        self.total_requests_per_minute = self
            .total_requests_per_minute
            .saturating_add(u64::from(quota.requests_per_minute));
        self.total_tokens_per_day = self
            .total_tokens_per_day
            .saturating_add(quota.tokens_per_day);
        self.total_max_concurrency = self
            .total_max_concurrency
            .saturating_add(u64::from(quota.max_concurrency));
        self.allowed_models
            .extend(quota.allowed_models.iter().cloned());
    }
}

pub async fn check_quota(
    storage: &Storage,
    quotas: &[QuotaConfig],
    principal: &Principal,
    model: &str,
) -> Result<QuotaDecision> {
    let matching = matching_quota_policies(quotas, principal);
    if matching.is_empty() {
        return Ok(QuotaDecision {
            allowed: true,
            reason: "no quota configured".to_string(),
        });
    }

    for q in &matching {
        if !q.allowed_models.is_empty() && !q.allowed_models.iter().any(|m| m == model) {
            return Ok(QuotaDecision {
                allowed: false,
                reason: format!(
                    "model {model} is not allowed for {}",
                    quota_scope_label(q, principal)
                ),
            });
        }
    }

    let now = Utc::now();
    for q in &matching {
        if q.requests_per_minute > 0 {
            let admitted = storage
                .allowed_quota_decision_count(
                    principal,
                    quota_is_subject_scoped(q, principal),
                    now - Duration::minutes(1),
                    now,
                )
                .await?;
            if admitted >= u64::from(q.requests_per_minute) {
                return Ok(QuotaDecision {
                    allowed: false,
                    reason: format!(
                        "requests_per_minute exhausted for {}: {admitted}/{}",
                        quota_scope_label(q, principal),
                        q.requests_per_minute
                    ),
                });
            }
        }
    }

    let day_start = now.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc();
    for q in &matching {
        if q.tokens_per_day > 0 {
            let used = storage
                .usage_tokens_total(
                    principal,
                    quota_is_subject_scoped(q, principal),
                    day_start,
                    now,
                )
                .await?;
            if used >= q.tokens_per_day {
                return Ok(QuotaDecision {
                    allowed: false,
                    reason: format!(
                        "tokens_per_day exhausted for {}: {used}/{}",
                        quota_scope_label(q, principal),
                        q.tokens_per_day
                    ),
                });
            }
        }
    }

    Ok(QuotaDecision {
        allowed: true,
        reason: "quota policy allowed".to_string(),
    })
}

/// Atomically checks the quota for `principal`/`model` and records the
/// admission decision while holding the per-scope admission lock.
///
/// The requests-per-minute check counts previously recorded quota-decision
/// rows. If the counting row were written after releasing the lock, several
/// concurrent same-scope requests could all read the same stale count and each
/// pass a limit they should collectively exhaust. Inserting the counting row
/// inside the locked section makes check-and-record atomic: a request only
/// releases the lock once its decision is visible to the next one.
///
/// The insert fails closed — if the counting row cannot be persisted the whole
/// admission returns an error rather than admitting an unrecorded request,
/// because a dropped row would under-count RPM and let later requests slip past
/// the limit.
pub async fn admit_request(
    storage: &Storage,
    quotas: &[QuotaConfig],
    principal: &Principal,
    model: &str,
    request_id: Option<Uuid>,
    policy_json: serde_json::Value,
) -> Result<QuotaDecision> {
    let scope = quota_admission_scope(principal);
    storage
        .with_quota_admission(&scope, || async {
            let decision = check_quota(storage, quotas, principal, model).await?;
            let record = QuotaDecisionRecord::new(
                request_id,
                principal,
                model,
                &decision,
                policy_json.clone(),
            );
            storage.insert_quota_decision(&record).await?;
            Ok(decision)
        })
        .await
}

pub fn matching_quota_policy<'a>(
    quotas: &'a [QuotaConfig],
    principal: &Principal,
) -> Option<&'a QuotaConfig> {
    matching_quota_policies(quotas, principal)
        .into_iter()
        .next()
}

pub fn matching_quota_policies<'a>(
    quotas: &'a [QuotaConfig],
    principal: &Principal,
) -> Vec<&'a QuotaConfig> {
    quotas
        .iter()
        .filter(|q| {
            q.subject == principal.subject || (!q.team.is_empty() && q.team == principal.team)
        })
        .collect()
}

pub fn quota_is_subject_scoped(q: &QuotaConfig, principal: &Principal) -> bool {
    q.subject == principal.subject
}

/// The `Storage::with_quota_admission` lock scope for `principal`.
///
/// `check_quota` reads counts scoped to either `principal.team` or
/// `principal.subject` depending on the matching policy. Keying the
/// admission lock by team (falling back to subject when there is no team)
/// keeps team-wide quotas correct under concurrent requests from different
/// subjects in the same team, while letting unrelated teams admit requests
/// fully in parallel instead of serializing on one global lock.
pub fn quota_admission_scope(principal: &Principal) -> String {
    if principal.team.is_empty() {
        format!("subject:{}", principal.subject)
    } else {
        format!("team:{}", principal.team)
    }
}

fn quota_scope_label(q: &QuotaConfig, principal: &Principal) -> String {
    if quota_is_subject_scoped(q, principal) {
        format!("subject {}", principal.subject)
    } else {
        format!("team {}", principal.team)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::UsageEvent;
    use crate::storage::{QuotaDecisionRecord, Storage};
    use chrono::Utc;
    use serde_json::json;
    use uuid::Uuid;

    #[tokio::test]
    async fn allows_when_policy_and_usage_are_inside_limits() -> Result<()> {
        let storage = Storage::in_memory().await?;
        let principal = principal("alice", "platform");
        let quotas = vec![quota("alice", "", 2, 100, &["llama"])];

        let decision = check_quota(&storage, &quotas, &principal, "llama").await?;

        assert!(decision.allowed, "{}", decision.reason);
        Ok(())
    }

    #[tokio::test]
    async fn allows_when_no_quota_matches() -> Result<()> {
        let storage = Storage::in_memory().await?;
        let principal = principal("alice", "platform");
        let quotas = vec![quota("bob", "research", 1, 10, &["mistral"])];

        let decision = check_quota(&storage, &quotas, &principal, "llama").await?;

        assert!(decision.allowed, "{}", decision.reason);
        assert_eq!(decision.reason, "no quota configured");
        Ok(())
    }

    #[tokio::test]
    async fn denies_models_outside_policy() -> Result<()> {
        let storage = Storage::in_memory().await?;
        let principal = principal("alice", "platform");
        let quotas = vec![quota("alice", "", 2, 100, &["llama"])];

        let decision = check_quota(&storage, &quotas, &principal, "mistral").await?;

        assert!(!decision.allowed);
        assert!(decision.reason.contains("model mistral is not allowed"));
        Ok(())
    }

    #[tokio::test]
    async fn denies_when_overlapping_team_policy_disallows_subject_model() -> Result<()> {
        let storage = Storage::in_memory().await?;
        let principal = principal("alice", "platform");
        let quotas = vec![
            quota("alice", "platform", 10, 100, &["llama", "mistral"]),
            quota("team-default", "platform", 10, 100, &["llama"]),
        ];

        let decision = check_quota(&storage, &quotas, &principal, "mistral").await?;

        assert!(!decision.allowed);
        assert!(decision.reason.contains("model mistral is not allowed"));
        assert!(decision.reason.contains("team platform"));
        Ok(())
    }

    #[tokio::test]
    async fn denies_when_requests_per_minute_is_exhausted() -> Result<()> {
        let storage = Storage::in_memory().await?;
        let principal = principal("alice", "platform");
        let quotas = vec![quota("alice", "", 1, 100, &["llama"])];
        let allowed = QuotaDecision {
            allowed: true,
            reason: "quota policy allowed".to_string(),
        };
        storage
            .insert_quota_decision(&QuotaDecisionRecord::new(
                Some(Uuid::new_v4()),
                &principal,
                "llama",
                &allowed,
                json!({}),
            ))
            .await?;

        let decision = check_quota(&storage, &quotas, &principal, "llama").await?;

        assert!(!decision.allowed);
        assert!(decision.reason.contains("requests_per_minute"));
        Ok(())
    }

    // Regression for the RPM quota bypass under concurrency: several
    // same-scope requests admitted at once must not all read a stale count of
    // zero and pass a limit of one. `admit_request` records the counting row
    // inside the admission lock, so exactly one is admitted.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_admissions_respect_requests_per_minute_limit_of_one() -> Result<()> {
        use std::sync::Arc;

        let storage = Storage::in_memory().await?;
        let quotas = Arc::new(vec![quota("alice", "", 1, 100, &["llama"])]);
        let principal = Arc::new(principal("alice", "platform"));

        let concurrency = 8;
        let mut handles = Vec::with_capacity(concurrency);
        for _ in 0..concurrency {
            let storage = storage.clone();
            let quotas = Arc::clone(&quotas);
            let principal = Arc::clone(&principal);
            handles.push(tokio::spawn(async move {
                admit_request(
                    &storage,
                    &quotas,
                    &principal,
                    "llama",
                    Some(Uuid::new_v4()),
                    json!({}),
                )
                .await
            }));
        }

        let mut admitted = 0;
        for handle in handles {
            if handle.await.expect("admission task panicked")?.allowed {
                admitted += 1;
            }
        }

        assert_eq!(
            admitted, 1,
            "requests_per_minute limit of 1 must admit exactly one concurrent request"
        );
        Ok(())
    }

    #[tokio::test]
    async fn denies_when_overlapping_team_request_limit_is_more_restrictive() -> Result<()> {
        let storage = Storage::in_memory().await?;
        let principal = principal("alice", "platform");
        let quotas = vec![
            quota("alice", "platform", 2, 100, &["llama"]),
            quota("team-default", "platform", 1, 100, &["llama"]),
        ];
        let allowed = QuotaDecision {
            allowed: true,
            reason: "quota policy allowed".to_string(),
        };
        storage
            .insert_quota_decision(&QuotaDecisionRecord::new(
                Some(Uuid::new_v4()),
                &principal,
                "llama",
                &allowed,
                json!({}),
            ))
            .await?;

        let decision = check_quota(&storage, &quotas, &principal, "llama").await?;

        assert!(!decision.allowed);
        assert!(decision.reason.contains("requests_per_minute"));
        assert!(decision.reason.contains("team platform"));
        Ok(())
    }

    #[tokio::test]
    async fn denies_when_daily_token_budget_is_exhausted() -> Result<()> {
        let storage = Storage::in_memory().await?;
        let principal = principal("alice", "platform");
        let quotas = vec![quota("alice", "", 2, 30, &["llama"])];
        storage
            .insert_usage_event(&UsageEvent {
                id: Uuid::new_v4(),
                request_id: Uuid::new_v4(),
                at: Utc::now(),
                model: "llama".to_string(),
                actor: principal.subject.clone(),
                team: principal.team.clone(),
                input_tokens: 10,
                output_tokens: 20,
                latency_ms: 5,
                status: "ok".to_string(),
            })
            .await?;

        let decision = check_quota(&storage, &quotas, &principal, "llama").await?;

        assert!(!decision.allowed);
        assert!(decision.reason.contains("tokens_per_day"));
        Ok(())
    }

    #[test]
    fn summarizes_quota_policies_by_team() {
        let quotas = vec![
            quota_with_concurrency("alice", "platform", 10, 1_000, 2, &["llama", "gpt"]),
            quota_with_concurrency("bob", "platform", 20, 2_000, 3, &["llama", "mistral"]),
            quota_with_concurrency("carol", "research", 30, 3_000, 4, &["gpt"]),
        ];

        let report = summarize_quota_policies(&quotas);

        assert_eq!(report.policy_count, 3);
        assert_eq!(report.total_requests_per_minute, 60);
        assert_eq!(report.total_tokens_per_day, 6_000);
        assert_eq!(report.total_max_concurrency, 9);

        assert_eq!(report.by_team.len(), 2);
        assert_eq!(report.by_team[0].team, "platform");
        assert_eq!(report.by_team[0].policy_count, 2);
        assert_eq!(report.by_team[0].subjects, vec!["alice", "bob"]);
        assert_eq!(
            report.by_team[0].allowed_models,
            vec!["gpt", "llama", "mistral"]
        );
        assert_eq!(report.by_team[0].total_requests_per_minute, 30);
        assert_eq!(report.by_team[0].total_tokens_per_day, 3_000);
        assert_eq!(report.by_team[0].total_max_concurrency, 5);

        assert_eq!(report.by_team[1].team, "research");
        assert_eq!(report.by_team[1].allowed_models, vec!["gpt"]);
    }

    #[test]
    fn summarizes_quota_policies_by_subject() {
        let quotas = vec![
            quota_with_concurrency("bob", "platform", 20, 2_000, 3, &["llama", "mistral"]),
            quota_with_concurrency("alice", "platform", 10, 1_000, 2, &["llama", "gpt"]),
        ];

        let report = summarize_quota_policies(&quotas);

        assert_eq!(report.by_subject.len(), 2);
        assert_eq!(report.by_subject[0].subject, "alice");
        assert_eq!(report.by_subject[0].team, "platform");
        assert_eq!(report.by_subject[0].policy_count, 1);
        assert_eq!(report.by_subject[0].allowed_models, vec!["gpt", "llama"]);
        assert_eq!(report.by_subject[0].total_requests_per_minute, 10);
        assert_eq!(report.by_subject[0].total_tokens_per_day, 1_000);
        assert_eq!(report.by_subject[0].total_max_concurrency, 2);

        assert_eq!(report.by_subject[1].subject, "bob");
        assert_eq!(
            report.by_subject[1].allowed_models,
            vec!["llama", "mistral"]
        );
    }

    fn principal(subject: &str, team: &str) -> Principal {
        Principal {
            subject: subject.to_string(),
            team: team.to_string(),
            scopes: vec!["chat".to_string()],
            key_id: Some(format!("{subject}-key")),
            key_owner: None,
            key_purpose: None,
            key_status: Some("active".to_string()),
        }
    }

    fn quota(
        subject: &str,
        team: &str,
        requests_per_minute: u32,
        tokens_per_day: u64,
        allowed_models: &[&str],
    ) -> QuotaConfig {
        quota_with_concurrency(
            subject,
            team,
            requests_per_minute,
            tokens_per_day,
            0,
            allowed_models,
        )
    }

    fn quota_with_concurrency(
        subject: &str,
        team: &str,
        requests_per_minute: u32,
        tokens_per_day: u64,
        max_concurrency: u32,
        allowed_models: &[&str],
    ) -> QuotaConfig {
        QuotaConfig {
            subject: subject.to_string(),
            team: team.to_string(),
            requests_per_minute,
            tokens_per_day,
            max_concurrency,
            allowed_models: allowed_models
                .iter()
                .map(|model| model.to_string())
                .collect(),
        }
    }
}
