use crate::config::QuotaConfig;
use crate::storage::Storage;
use anyhow::Result;
use chrono::{Duration, Utc};

#[derive(Debug, Clone)]
pub struct Principal {
    pub subject: String,
    pub team: String,
    pub scopes: Vec<String>,
}

impl Principal {
    pub fn anonymous() -> Self {
        Self {
            subject: "anonymous".to_string(),
            team: "public".to_string(),
            scopes: vec!["chat".to_string(), "models.read".to_string()],
        }
    }

    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.iter().any(|s| s == scope || s == "admin")
    }
}

#[derive(Debug, Clone)]
pub struct QuotaDecision {
    pub allowed: bool,
    pub reason: String,
}

pub async fn check_quota(
    storage: &Storage,
    quotas: &[QuotaConfig],
    principal: &Principal,
    model: &str,
) -> Result<QuotaDecision> {
    let Some(q) = quotas.iter().find(|q| {
        q.subject == principal.subject || (!q.team.is_empty() && q.team == principal.team)
    }) else {
        return Ok(QuotaDecision {
            allowed: true,
            reason: "no quota configured".to_string(),
        });
    };
    if !q.allowed_models.is_empty() && !q.allowed_models.iter().any(|m| m == model) {
        return Ok(QuotaDecision {
            allowed: false,
            reason: format!("model {model} is not allowed for {}", principal.subject),
        });
    }

    let subject_scoped = q.subject == principal.subject;
    let now = Utc::now();
    if q.requests_per_minute > 0 {
        let admitted = storage
            .allowed_quota_decision_count(
                principal,
                subject_scoped,
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

    if q.tokens_per_day > 0 {
        let day_start = now.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc();
        let used = storage
            .usage_tokens_total(principal, subject_scoped, day_start, now)
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

    Ok(QuotaDecision {
        allowed: true,
        reason: "quota policy allowed".to_string(),
    })
}

fn quota_scope_label(q: &QuotaConfig, principal: &Principal) -> String {
    if q.subject == principal.subject {
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

    fn principal(subject: &str, team: &str) -> Principal {
        Principal {
            subject: subject.to_string(),
            team: team.to_string(),
            scopes: vec!["chat".to_string()],
        }
    }

    fn quota(
        subject: &str,
        team: &str,
        requests_per_minute: u32,
        tokens_per_day: u64,
        allowed_models: &[&str],
    ) -> QuotaConfig {
        QuotaConfig {
            subject: subject.to_string(),
            team: team.to_string(),
            requests_per_minute,
            tokens_per_day,
            max_concurrency: 0,
            allowed_models: allowed_models
                .iter()
                .map(|model| model.to_string())
                .collect(),
        }
    }
}
