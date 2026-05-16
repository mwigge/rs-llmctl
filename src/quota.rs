use crate::config::QuotaConfig;
use anyhow::Result;

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

pub fn check_quota(
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
    Ok(QuotaDecision {
        allowed: true,
        reason: "quota policy allowed".to_string(),
    })
}
