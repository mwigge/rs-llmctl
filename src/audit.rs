use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub id: Uuid,
    pub request_id: Option<Uuid>,
    pub at: DateTime<Utc>,
    pub actor: String,
    pub team: String,
    pub action: String,
    pub resource: String,
    pub outcome: String,
    pub detail_json: serde_json::Value,
}

impl AuditEvent {
    pub fn new(
        request_id: Option<Uuid>,
        actor: impl Into<String>,
        team: impl Into<String>,
        action: impl Into<String>,
        resource: impl Into<String>,
        outcome: impl Into<String>,
        detail_json: serde_json::Value,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            request_id,
            at: Utc::now(),
            actor: actor.into(),
            team: team.into(),
            action: action.into(),
            resource: resource.into(),
            outcome: outcome.into(),
            detail_json,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEvent {
    pub id: Uuid,
    pub request_id: Uuid,
    pub at: DateTime<Utc>,
    pub model: String,
    pub actor: String,
    pub team: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub latency_ms: u64,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationEvent {
    pub id: Uuid,
    pub request_id: Option<Uuid>,
    pub at: DateTime<Utc>,
    pub kind: String,
    pub model: String,
    pub source: String,
    pub value: f64,
    pub unit: String,
    pub attributes_json: serde_json::Value,
}
