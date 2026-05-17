use crate::config::{Config, ModelConfig};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AqeGovernanceContract {
    pub kind: String,
    pub endpoint: IntegrationEndpointContract,
    pub auth: IntegrationAuthContract,
    pub response_headers: Vec<String>,
    pub quota_reporting: IntegrationReportingContract,
    pub team_reporting: IntegrationTeamReportingContract,
    pub model_aliases: Vec<IntegrationModelAlias>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IntegrationEndpointContract {
    pub base_url: String,
    pub openai_paths: OpenAiPathContract,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAiPathContract {
    pub models: String,
    pub chat_completions: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IntegrationAuthContract {
    pub scheme: String,
    pub required: bool,
    pub required_scopes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IntegrationReportingContract {
    pub fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IntegrationTeamReportingContract {
    pub fields: Vec<String>,
    pub teams: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IntegrationModelAlias {
    pub alias: String,
    pub role: String,
    pub weight: u32,
}

pub fn aqe_governance_contract(cfg: &Config) -> AqeGovernanceContract {
    AqeGovernanceContract {
        kind: "aqe-openai-governance-contract".to_string(),
        endpoint: IntegrationEndpointContract {
            base_url: base_url(&cfg.server.host, cfg.server.port),
            openai_paths: OpenAiPathContract {
                models: "/v1/models".to_string(),
                chat_completions: "/v1/chat/completions".to_string(),
            },
        },
        auth: IntegrationAuthContract {
            scheme: "bearer".to_string(),
            required: cfg.security.require_auth,
            required_scopes: vec!["chat".to_string(), "models.read".to_string()],
        },
        response_headers: vec![
            "x-request-id".to_string(),
            "x-llmctl-model-count".to_string(),
            "x-llmctl-model".to_string(),
            "x-llmctl-upstream-model".to_string(),
            "x-llmctl-quota-decision".to_string(),
        ],
        quota_reporting: IntegrationReportingContract {
            fields: vec![
                "team".to_string(),
                "subject".to_string(),
                "requests_per_minute".to_string(),
                "tokens_per_day".to_string(),
                "max_concurrency".to_string(),
                "allowed_models".to_string(),
            ],
        },
        team_reporting: IntegrationTeamReportingContract {
            fields: vec![
                "team".to_string(),
                "subjects".to_string(),
                "allowed_models".to_string(),
            ],
            teams: cfg
                .quotas
                .iter()
                .map(|quota| quota.team.clone())
                .filter(|team| !team.trim().is_empty())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
        },
        model_aliases: model_aliases(&cfg.models),
    }
}

fn base_url(host: &str, port: u16) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("http://[{host}]:{port}")
    } else {
        format!("http://{host}:{port}")
    }
}

fn model_aliases(models: &[ModelConfig]) -> Vec<IntegrationModelAlias> {
    let mut aliases = models
        .iter()
        .map(|model| IntegrationModelAlias {
            alias: model.alias.clone(),
            role: model.role.clone(),
            weight: model.weight,
        })
        .collect::<Vec<_>>();
    aliases.sort_by(|left, right| left.alias.cmp(&right.alias));
    aliases
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ApiKeyConfig, QuotaConfig};
    use std::path::PathBuf;

    #[test]
    fn aqe_contract_is_serializable_and_redacted_by_design() {
        let mut cfg = Config::default();
        cfg.server.host = "0.0.0.0".to_string();
        cfg.server.port = 8765;
        cfg.security.require_auth = true;
        cfg.security.api_keys = vec![ApiKeyConfig {
            id: "operator".to_string(),
            sha256: "a".repeat(64),
            subject: "alice".to_string(),
            team: "platform".to_string(),
            scopes: vec!["admin".to_string()],
            ..Default::default()
        }];
        cfg.models = vec![ModelConfig {
            alias: "chat".to_string(),
            path: PathBuf::from("/private/models/chat.gguf"),
            role: "chat".to_string(),
            family: Some("qwen3".to_string()),
            weight: 1,
        }];
        cfg.quotas = vec![QuotaConfig {
            subject: "alice".to_string(),
            team: "platform".to_string(),
            requests_per_minute: 30,
            tokens_per_day: 100_000,
            max_concurrency: 4,
            allowed_models: vec!["chat".to_string()],
        }];

        let raw = serde_json::to_string(&aqe_governance_contract(&cfg)).expect("serialize");

        assert!(raw.contains("aqe-openai-governance-contract"));
        assert!(!raw.contains("operator"));
        assert!(!raw.contains(&"a".repeat(64)));
        assert!(!raw.contains("/private/models/chat.gguf"));
        assert!(!raw.contains("upstream.internal"));
        assert!(!raw.contains("prompt"));
    }
}
