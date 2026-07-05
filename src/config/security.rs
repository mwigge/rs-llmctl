//! Security posture, TLS termination, and API-key configuration.
use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct SecurityConfig {
    pub production: bool,
    #[serde(alias = "require_auth")]
    pub require_auth: bool,
    #[serde(alias = "bind_external")]
    pub bind_external: bool,
    #[serde(default = "default_auth_failure_limit_per_minute")]
    pub auth_failure_limit_per_minute: u32,
    #[serde(default, alias = "tls_termination")]
    pub tls_termination: TlsTerminationConfig,
    #[serde(default)]
    pub trusted_proxies: Vec<String>,
    #[serde(alias = "api_keys")]
    pub api_keys: Vec<ApiKeyConfig>,
}

fn default_auth_failure_limit_per_minute() -> u32 {
    60
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            production: false,
            require_auth: false,
            bind_external: false,
            auth_failure_limit_per_minute: default_auth_failure_limit_per_minute(),
            tls_termination: TlsTerminationConfig::default(),
            trusted_proxies: Vec::new(),
            api_keys: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct TlsTerminationConfig {
    pub enabled: bool,
    pub provider: Option<String>,
    pub evidence: Option<String>,
    #[serde(alias = "mtls")]
    pub m_tls: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct ApiKeyConfig {
    pub id: String,
    pub sha256: String,
    pub subject: String,
    pub team: String,
    pub scopes: Vec<String>,
    #[serde(alias = "created_at")]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(alias = "expires_at")]
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(alias = "rotated_at")]
    pub rotated_at: Option<chrono::DateTime<chrono::Utc>>,
    pub owner: Option<String>,
    pub purpose: Option<String>,
    #[serde(alias = "last_four")]
    pub last_four: Option<String>,
    pub fingerprint: Option<String>,
    #[serde(default = "default_api_key_status")]
    pub status: String,
}

impl Default for ApiKeyConfig {
    fn default() -> Self {
        Self {
            id: String::new(),
            sha256: String::new(),
            subject: String::new(),
            team: String::new(),
            scopes: Vec::new(),
            created_at: None,
            expires_at: None,
            rotated_at: None,
            owner: None,
            purpose: None,
            last_four: None,
            fingerprint: None,
            status: default_api_key_status(),
        }
    }
}

fn default_api_key_status() -> String {
    "active".to_string()
}
