use crate::audit::{AuditEvent, ObservationEvent, UsageEvent};
use crate::config::{ModelConfig, StorageConfig};
use crate::quota::{Principal, QuotaDecision};
use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};
use std::fmt;
use std::path::Path;
use std::str::FromStr;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum StorageBackend {
    Sqlite,
    Postgres,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum SqlDialect {
    Sqlite,
    Postgres,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageConnectionPlan {
    pub backend: StorageBackend,
    target: String,
    display_target: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageMigrationPlan {
    dialect: SqlDialect,
    statements: Vec<String>,
}

impl StorageConnectionPlan {
    pub fn from_config(config: &StorageConfig) -> Result<Self> {
        if let Some(database_url) = config.database_url.as_deref() {
            let backend = backend_for_url(database_url)?;
            return Ok(Self {
                backend,
                target: database_url.to_string(),
                display_target: redact_database_url(database_url),
            });
        }

        let backend = config.backend.unwrap_or(StorageBackend::Sqlite);
        match backend {
            StorageBackend::Sqlite => Ok(Self {
                backend,
                target: config.db_path.display().to_string(),
                display_target: config.db_path.display().to_string(),
            }),
            StorageBackend::Postgres => {
                bail!("postgres storage requires storage.database-url")
            }
        }
    }

    pub fn dialect(&self) -> SqlDialect {
        match self.backend {
            StorageBackend::Sqlite => SqlDialect::Sqlite,
            StorageBackend::Postgres => SqlDialect::Postgres,
        }
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn display_target(&self) -> &str {
        &self.display_target
    }
}

impl fmt::Display for StorageConnectionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.backend {
            StorageBackend::Sqlite => write!(f, "sqlite database {}", self.display_target),
            StorageBackend::Postgres => write!(f, "postgres database {}", self.display_target),
        }
    }
}

impl StorageMigrationPlan {
    pub fn new(dialect: SqlDialect) -> Self {
        Self {
            dialect,
            statements: migration_statements(dialect),
        }
    }

    pub fn dialect(&self) -> SqlDialect {
        self.dialect
    }

    pub fn statements(&self) -> &[String] {
        &self.statements
    }
}

fn backend_for_url(database_url: &str) -> Result<StorageBackend> {
    let Some((scheme, _)) = database_url.split_once(':') else {
        bail!("storage.database-url must include a URL scheme");
    };

    match scheme {
        "sqlite" => Ok(StorageBackend::Sqlite),
        "postgres" | "postgresql" => Ok(StorageBackend::Postgres),
        other => bail!("unsupported storage database-url scheme `{other}`"),
    }
}

fn redact_database_url(database_url: &str) -> String {
    let Some((scheme, rest)) = database_url.split_once("://") else {
        return database_url.to_string();
    };
    let Some((authority, tail)) = rest.split_once('/') else {
        return redact_authority_url(scheme, rest, "");
    };
    redact_authority_url(scheme, authority, &format!("/{tail}"))
}

fn redact_authority_url(scheme: &str, authority: &str, tail: &str) -> String {
    let Some((userinfo, host)) = authority.rsplit_once('@') else {
        return format!("{scheme}://{authority}{tail}");
    };
    let redacted_userinfo = match userinfo.split_once(':') {
        Some((user, _password)) => format!("{user}:[REDACTED]"),
        None => userinfo.to_string(),
    };
    format!("{scheme}://{redacted_userinfo}@{host}{tail}")
}

fn migration_statements(dialect: SqlDialect) -> Vec<String> {
    let bool_type = match dialect {
        SqlDialect::Sqlite => "INTEGER",
        SqlDialect::Postgres => "BOOLEAN",
    };
    let float_type = match dialect {
        SqlDialect::Sqlite => "REAL",
        SqlDialect::Postgres => "DOUBLE PRECISION",
    };
    let id_type = match dialect {
        SqlDialect::Sqlite => "TEXT",
        SqlDialect::Postgres => "UUID",
    };
    let time_type = match dialect {
        SqlDialect::Sqlite => "TEXT",
        SqlDialect::Postgres => "TIMESTAMPTZ",
    };
    let json_type = match dialect {
        SqlDialect::Sqlite => "TEXT",
        SqlDialect::Postgres => "JSONB",
    };
    let bool_check = match dialect {
        SqlDialect::Sqlite => " CHECK (allowed IN (0, 1))",
        SqlDialect::Postgres => "",
    };

    vec![
        format!(
            r#"
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id TEXT PRIMARY KEY NOT NULL,
                checksum TEXT NOT NULL,
                applied_at {time_type} NOT NULL
            )
            "#
        ),
        format!(
            r#"
            CREATE TABLE IF NOT EXISTS audit_events (
                id {id_type} PRIMARY KEY NOT NULL,
                request_id {id_type},
                at {time_type} NOT NULL,
                actor TEXT NOT NULL,
                team TEXT NOT NULL,
                action TEXT NOT NULL,
                resource TEXT NOT NULL,
                outcome TEXT NOT NULL,
                detail_json {json_type} NOT NULL
            )
            "#
        ),
        "CREATE INDEX IF NOT EXISTS idx_audit_events_at ON audit_events(at)".to_string(),
        "CREATE INDEX IF NOT EXISTS idx_audit_events_request_id ON audit_events(request_id)"
            .to_string(),
        format!(
            r#"
            CREATE TABLE IF NOT EXISTS usage_events (
                id {id_type} PRIMARY KEY NOT NULL,
                request_id {id_type} NOT NULL,
                at {time_type} NOT NULL,
                model TEXT NOT NULL,
                actor TEXT NOT NULL,
                team TEXT NOT NULL,
                input_tokens INTEGER NOT NULL CHECK (input_tokens >= 0),
                output_tokens INTEGER NOT NULL CHECK (output_tokens >= 0),
                latency_ms INTEGER NOT NULL CHECK (latency_ms >= 0),
                status TEXT NOT NULL
            )
            "#
        ),
        "CREATE INDEX IF NOT EXISTS idx_usage_events_at ON usage_events(at)".to_string(),
        "CREATE INDEX IF NOT EXISTS idx_usage_events_request_id ON usage_events(request_id)"
            .to_string(),
        "CREATE INDEX IF NOT EXISTS idx_usage_events_actor_at ON usage_events(actor, at)"
            .to_string(),
        "CREATE INDEX IF NOT EXISTS idx_usage_events_team_at ON usage_events(team, at)"
            .to_string(),
        format!(
            r#"
            CREATE TABLE IF NOT EXISTS observation_events (
                id {id_type} PRIMARY KEY NOT NULL,
                request_id {id_type},
                at {time_type} NOT NULL,
                kind TEXT NOT NULL,
                model TEXT NOT NULL,
                source TEXT NOT NULL,
                value {float_type} NOT NULL,
                unit TEXT NOT NULL,
                attributes_json {json_type} NOT NULL
            )
            "#
        ),
        "CREATE INDEX IF NOT EXISTS idx_observation_events_at ON observation_events(at)"
            .to_string(),
        "CREATE INDEX IF NOT EXISTS idx_observation_events_request_id ON observation_events(request_id)"
            .to_string(),
        r#"
            CREATE TABLE IF NOT EXISTS model_inventory (
                alias TEXT PRIMARY KEY NOT NULL,
                path TEXT NOT NULL,
                role TEXT NOT NULL,
                weight INTEGER NOT NULL CHECK (weight >= 0),
                updated_at TEXT NOT NULL
            )
            "#
        .to_string(),
        format!(
            r#"
            CREATE TABLE IF NOT EXISTS quota_decisions (
                id {id_type} PRIMARY KEY NOT NULL,
                request_id {id_type},
                at {time_type} NOT NULL,
                actor TEXT NOT NULL,
                team TEXT NOT NULL,
                model TEXT NOT NULL,
                allowed {bool_type} NOT NULL{bool_check},
                reason TEXT NOT NULL,
                policy_json {json_type} NOT NULL
            )
            "#
        ),
        "CREATE INDEX IF NOT EXISTS idx_quota_decisions_at ON quota_decisions(at)".to_string(),
        "CREATE INDEX IF NOT EXISTS idx_quota_decisions_request_id ON quota_decisions(request_id)"
            .to_string(),
        "CREATE INDEX IF NOT EXISTS idx_quota_decisions_actor_allowed_at ON quota_decisions(actor, allowed, at)"
            .to_string(),
        "CREATE INDEX IF NOT EXISTS idx_quota_decisions_team_allowed_at ON quota_decisions(team, allowed, at)"
            .to_string(),
    ]
}

#[derive(Debug, Clone)]
pub struct Storage {
    pool: SqlitePool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInventoryRecord {
    pub alias: String,
    pub path: String,
    pub role: String,
    pub weight: u32,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaDecisionRecord {
    pub id: Uuid,
    pub request_id: Option<Uuid>,
    pub at: DateTime<Utc>,
    pub actor: String,
    pub team: String,
    pub model: String,
    pub allowed: bool,
    pub reason: String,
    pub policy_json: serde_json::Value,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditRetentionCounts {
    pub total: u64,
    pub in_retention_window: u64,
    pub outside_retention_window: u64,
}

impl QuotaDecisionRecord {
    pub fn new(
        request_id: Option<Uuid>,
        principal: &Principal,
        model: impl Into<String>,
        decision: &QuotaDecision,
        policy_json: serde_json::Value,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            request_id,
            at: Utc::now(),
            actor: principal.subject.clone(),
            team: principal.team.clone(),
            model: model.into(),
            allowed: decision.allowed,
            reason: decision.reason.clone(),
            policy_json,
        }
    }
}

impl Storage {
    pub async fn connect_config(config: &StorageConfig) -> Result<Self> {
        let plan = config.connection_plan()?;
        match plan.backend {
            StorageBackend::Sqlite => Self::connect_sqlite_target(plan.target()).await,
            StorageBackend::Postgres => {
                bail!(
                    "postgres runtime storage is not enabled in this build; use sqlite storage or run storage planning commands only"
                )
            }
        }
    }

    pub async fn connect(db_path: impl AsRef<Path>) -> Result<Self> {
        let db_path = db_path.as_ref();
        if let Some(parent) = db_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("create storage directory {}", parent.display()))?;
        }

        let options = SqliteConnectOptions::from_str(&format!("sqlite://{}", db_path.display()))?
            .create_if_missing(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await
            .with_context(|| format!("open sqlite database {}", db_path.display()))?;
        let storage = Self { pool };
        storage.migrate().await?;
        Ok(storage)
    }

    async fn connect_sqlite_target(target: &str) -> Result<Self> {
        if let Some(path) = target.strip_prefix("sqlite://") {
            Self::connect(path).await
        } else {
            Self::connect(target).await
        }
    }

    pub async fn in_memory() -> Result<Self> {
        let options = SqliteConnectOptions::from_str("sqlite::memory:")?;
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
            .context("open in-memory sqlite database")?;
        let storage = Self { pool };
        storage.migrate().await?;
        Ok(storage)
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    pub async fn migrate(&self) -> Result<()> {
        for statement in StorageMigrationPlan::new(SqlDialect::Sqlite).statements() {
            sqlx::query(statement).execute(&self.pool).await?;
        }
        add_column_if_missing(&self.pool, "observation_events", "request_id TEXT").await?;

        Ok(())
    }

    pub async fn insert_audit_event(&self, event: &AuditEvent) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO audit_events
                (id, request_id, at, actor, team, action, resource, outcome, detail_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(event.id.to_string())
        .bind(event.request_id.map(|id| id.to_string()))
        .bind(encode_time(event.at))
        .bind(&event.actor)
        .bind(&event.team)
        .bind(&event.action)
        .bind(&event.resource)
        .bind(&event.outcome)
        .bind(event.detail_json.to_string())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn audit_events_between(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<AuditEvent>> {
        let rows = sqlx::query(
            r#"
            SELECT id, request_id, at, actor, team, action, resource, outcome, detail_json
            FROM audit_events
            WHERE at >= ? AND at < ?
            ORDER BY at ASC, id ASC
            "#,
        )
        .bind(encode_time(from))
        .bind(encode_time(to))
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(row_to_audit_event).collect()
    }

    pub async fn audit_events_for_request(&self, request_id: Uuid) -> Result<Vec<AuditEvent>> {
        let rows = sqlx::query(
            r#"
            SELECT id, request_id, at, actor, team, action, resource, outcome, detail_json
            FROM audit_events
            WHERE request_id = ?
            ORDER BY at ASC, id ASC
            "#,
        )
        .bind(request_id.to_string())
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(row_to_audit_event).collect()
    }

    pub async fn audit_retention_counts(
        &self,
        cutoff: DateTime<Utc>,
    ) -> Result<AuditRetentionCounts> {
        let row = sqlx::query(
            r#"
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(CASE WHEN at >= ? THEN 1 ELSE 0 END), 0) AS in_retention_window,
                COALESCE(SUM(CASE WHEN at < ? THEN 1 ELSE 0 END), 0) AS outside_retention_window
            FROM audit_events
            "#,
        )
        .bind(encode_time(cutoff))
        .bind(encode_time(cutoff))
        .fetch_one(&self.pool)
        .await?;

        Ok(AuditRetentionCounts {
            total: i64_to_u64(row.try_get("total")?, "total")?,
            in_retention_window: i64_to_u64(
                row.try_get("in_retention_window")?,
                "in_retention_window",
            )?,
            outside_retention_window: i64_to_u64(
                row.try_get("outside_retention_window")?,
                "outside_retention_window",
            )?,
        })
    }

    pub async fn delete_audit_events_before(&self, cutoff: DateTime<Utc>) -> Result<u64> {
        let result = sqlx::query("DELETE FROM audit_events WHERE at < ?")
            .bind(encode_time(cutoff))
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }

    pub async fn insert_usage_event(&self, event: &UsageEvent) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO usage_events
                (id, request_id, at, model, actor, team, input_tokens, output_tokens, latency_ms, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(event.id.to_string())
        .bind(event.request_id.to_string())
        .bind(encode_time(event.at))
        .bind(&event.model)
        .bind(&event.actor)
        .bind(&event.team)
        .bind(i64::try_from(event.input_tokens)?)
        .bind(i64::try_from(event.output_tokens)?)
        .bind(i64::try_from(event.latency_ms)?)
        .bind(&event.status)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn usage_events_between(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<UsageEvent>> {
        let rows = sqlx::query(
            r#"
            SELECT id, request_id, at, model, actor, team, input_tokens, output_tokens, latency_ms, status
            FROM usage_events
            WHERE at >= ? AND at < ?
            ORDER BY at ASC, id ASC
            "#,
        )
        .bind(encode_time(from))
        .bind(encode_time(to))
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(row_to_usage_event).collect()
    }

    pub async fn usage_events_for_request(&self, request_id: Uuid) -> Result<Vec<UsageEvent>> {
        let rows = sqlx::query(
            r#"
            SELECT id, request_id, at, model, actor, team, input_tokens, output_tokens, latency_ms, status
            FROM usage_events
            WHERE request_id = ?
            ORDER BY at ASC, id ASC
            "#,
        )
        .bind(request_id.to_string())
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(row_to_usage_event).collect()
    }

    pub async fn insert_observation_event(&self, event: &ObservationEvent) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO observation_events
                (id, request_id, at, kind, model, source, value, unit, attributes_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(event.id.to_string())
        .bind(event.request_id.map(|id| id.to_string()))
        .bind(encode_time(event.at))
        .bind(&event.kind)
        .bind(&event.model)
        .bind(&event.source)
        .bind(event.value)
        .bind(&event.unit)
        .bind(event.attributes_json.to_string())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn observation_events_between(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<ObservationEvent>> {
        let rows = sqlx::query(
            r#"
            SELECT id, request_id, at, kind, model, source, value, unit, attributes_json
            FROM observation_events
            WHERE at >= ? AND at < ?
            ORDER BY at ASC, id ASC
            "#,
        )
        .bind(encode_time(from))
        .bind(encode_time(to))
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(row_to_observation_event).collect()
    }

    pub async fn observation_events_for_request(
        &self,
        request_id: Uuid,
    ) -> Result<Vec<ObservationEvent>> {
        let rows = sqlx::query(
            r#"
            SELECT id, request_id, at, kind, model, source, value, unit, attributes_json
            FROM observation_events
            WHERE request_id = ?
            ORDER BY at ASC, id ASC
            "#,
        )
        .bind(request_id.to_string())
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(row_to_observation_event).collect()
    }

    pub async fn upsert_model(&self, model: &ModelConfig) -> Result<()> {
        let record = ModelInventoryRecord {
            alias: model.alias.clone(),
            path: model.path.display().to_string(),
            role: model.role.clone(),
            weight: model.weight,
            updated_at: Utc::now(),
        };
        self.upsert_model_record(&record).await
    }

    pub async fn upsert_model_record(&self, model: &ModelInventoryRecord) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO model_inventory (alias, path, role, weight, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(alias) DO UPDATE SET
                path = excluded.path,
                role = excluded.role,
                weight = excluded.weight,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(&model.alias)
        .bind(&model.path)
        .bind(&model.role)
        .bind(i64::from(model.weight))
        .bind(encode_time(model.updated_at))
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInventoryRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT alias, path, role, weight, updated_at
            FROM model_inventory
            ORDER BY alias ASC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(row_to_model_inventory_record)
            .collect()
    }

    pub async fn insert_quota_decision(&self, decision: &QuotaDecisionRecord) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO quota_decisions
                (id, request_id, at, actor, team, model, allowed, reason, policy_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(decision.id.to_string())
        .bind(decision.request_id.map(|id| id.to_string()))
        .bind(encode_time(decision.at))
        .bind(&decision.actor)
        .bind(&decision.team)
        .bind(&decision.model)
        .bind(if decision.allowed { 1_i64 } else { 0_i64 })
        .bind(&decision.reason)
        .bind(decision.policy_json.to_string())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn quota_decisions_between(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<QuotaDecisionRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, request_id, at, actor, team, model, allowed, reason, policy_json
            FROM quota_decisions
            WHERE at >= ? AND at < ?
            ORDER BY at ASC, id ASC
            "#,
        )
        .bind(encode_time(from))
        .bind(encode_time(to))
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(row_to_quota_decision_record).collect()
    }

    pub async fn quota_decisions_for_request(
        &self,
        request_id: Uuid,
    ) -> Result<Vec<QuotaDecisionRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, request_id, at, actor, team, model, allowed, reason, policy_json
            FROM quota_decisions
            WHERE request_id = ?
            ORDER BY at ASC, id ASC
            "#,
        )
        .bind(request_id.to_string())
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(row_to_quota_decision_record).collect()
    }

    pub async fn allowed_quota_decision_count(
        &self,
        principal: &Principal,
        subject_scoped: bool,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<u64> {
        let scope_value = if subject_scoped {
            &principal.subject
        } else {
            &principal.team
        };
        let scope_column = if subject_scoped { "actor" } else { "team" };
        let query = format!(
            r#"
            SELECT COUNT(*) AS count
            FROM quota_decisions
            WHERE allowed = 1
              AND at >= ?
              AND at < ?
              AND {scope_column} = ?
            "#
        );
        let row = sqlx::query(&query)
            .bind(encode_time(from))
            .bind(encode_time(to))
            .bind(scope_value)
            .fetch_one(&self.pool)
            .await?;
        i64_to_u64(row.try_get("count")?, "count")
    }

    pub async fn usage_tokens_total(
        &self,
        principal: &Principal,
        subject_scoped: bool,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<u64> {
        let scope_value = if subject_scoped {
            &principal.subject
        } else {
            &principal.team
        };
        let scope_column = if subject_scoped { "actor" } else { "team" };
        let query = format!(
            r#"
            SELECT COALESCE(SUM(input_tokens + output_tokens), 0) AS tokens
            FROM usage_events
            WHERE at >= ?
              AND at < ?
              AND {scope_column} = ?
            "#
        );
        let row = sqlx::query(&query)
            .bind(encode_time(from))
            .bind(encode_time(to))
            .bind(scope_value)
            .fetch_one(&self.pool)
            .await?;
        i64_to_u64(row.try_get("tokens")?, "tokens")
    }
}

fn encode_time(at: DateTime<Utc>) -> String {
    at.to_rfc3339()
}

fn parse_time(value: &str) -> Result<DateTime<Utc>> {
    Ok(DateTime::parse_from_rfc3339(value)?.with_timezone(&Utc))
}

fn parse_uuid(value: &str) -> Result<Uuid> {
    Uuid::parse_str(value).with_context(|| format!("parse uuid {value}"))
}

fn parse_optional_uuid(value: Option<String>) -> Result<Option<Uuid>> {
    value.as_deref().map(parse_uuid).transpose()
}

fn parse_json(value: String) -> Result<serde_json::Value> {
    serde_json::from_str(&value).context("parse sqlite json payload")
}

fn i64_to_u64(value: i64, field: &str) -> Result<u64> {
    u64::try_from(value).with_context(|| format!("read non-negative integer field {field}"))
}

fn i64_to_u32(value: i64, field: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("read u32 integer field {field}"))
}

fn row_to_audit_event(row: sqlx::sqlite::SqliteRow) -> Result<AuditEvent> {
    Ok(AuditEvent {
        id: parse_uuid(&row.try_get::<String, _>("id")?)?,
        request_id: parse_optional_uuid(row.try_get::<Option<String>, _>("request_id")?)?,
        at: parse_time(&row.try_get::<String, _>("at")?)?,
        actor: row.try_get("actor")?,
        team: row.try_get("team")?,
        action: row.try_get("action")?,
        resource: row.try_get("resource")?,
        outcome: row.try_get("outcome")?,
        detail_json: parse_json(row.try_get("detail_json")?)?,
    })
}

fn row_to_usage_event(row: sqlx::sqlite::SqliteRow) -> Result<UsageEvent> {
    Ok(UsageEvent {
        id: parse_uuid(&row.try_get::<String, _>("id")?)?,
        request_id: parse_uuid(&row.try_get::<String, _>("request_id")?)?,
        at: parse_time(&row.try_get::<String, _>("at")?)?,
        model: row.try_get("model")?,
        actor: row.try_get("actor")?,
        team: row.try_get("team")?,
        input_tokens: i64_to_u64(row.try_get("input_tokens")?, "input_tokens")?,
        output_tokens: i64_to_u64(row.try_get("output_tokens")?, "output_tokens")?,
        latency_ms: i64_to_u64(row.try_get("latency_ms")?, "latency_ms")?,
        status: row.try_get("status")?,
    })
}

fn row_to_observation_event(row: sqlx::sqlite::SqliteRow) -> Result<ObservationEvent> {
    Ok(ObservationEvent {
        id: parse_uuid(&row.try_get::<String, _>("id")?)?,
        request_id: parse_optional_uuid(row.try_get::<Option<String>, _>("request_id")?)?,
        at: parse_time(&row.try_get::<String, _>("at")?)?,
        kind: row.try_get("kind")?,
        model: row.try_get("model")?,
        source: row.try_get("source")?,
        value: row.try_get("value")?,
        unit: row.try_get("unit")?,
        attributes_json: parse_json(row.try_get("attributes_json")?)?,
    })
}

async fn add_column_if_missing(pool: &SqlitePool, table: &str, column_sql: &str) -> Result<()> {
    let column = column_sql
        .split_whitespace()
        .next()
        .context("column definition must include a name")?;
    let pragma = format!("PRAGMA table_info({table})");
    let rows = sqlx::query(&pragma).fetch_all(pool).await?;
    let exists = rows.iter().any(|row| {
        row.try_get::<String, _>("name")
            .is_ok_and(|name| name == column)
    });
    if !exists {
        let alter = format!("ALTER TABLE {table} ADD COLUMN {column_sql}");
        sqlx::query(&alter).execute(pool).await?;
    }
    Ok(())
}

fn row_to_model_inventory_record(row: sqlx::sqlite::SqliteRow) -> Result<ModelInventoryRecord> {
    Ok(ModelInventoryRecord {
        alias: row.try_get("alias")?,
        path: row.try_get("path")?,
        role: row.try_get("role")?,
        weight: i64_to_u32(row.try_get("weight")?, "weight")?,
        updated_at: parse_time(&row.try_get::<String, _>("updated_at")?)?,
    })
}

fn row_to_quota_decision_record(row: sqlx::sqlite::SqliteRow) -> Result<QuotaDecisionRecord> {
    let allowed: i64 = row.try_get("allowed")?;
    Ok(QuotaDecisionRecord {
        id: parse_uuid(&row.try_get::<String, _>("id")?)?,
        request_id: parse_optional_uuid(row.try_get::<Option<String>, _>("request_id")?)?,
        at: parse_time(&row.try_get::<String, _>("at")?)?,
        actor: row.try_get("actor")?,
        team: row.try_get("team")?,
        model: row.try_get("model")?,
        allowed: allowed != 0,
        reason: row.try_get("reason")?,
        policy_json: parse_json(row.try_get("policy_json")?)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::{AuditEvent, ObservationEvent, UsageEvent};
    use crate::config::Config;
    use serde_json::json;

    #[test]
    fn legacy_db_path_config_plans_sqlite_storage() -> Result<()> {
        let cfg: Config = toml::from_str(
            r#"
            [storage]
            db_path = "/var/lib/llmctl/state.db"
            model_dir = "/var/lib/llmctl/models"
            "#,
        )?;

        let plan = cfg.storage.connection_plan()?;

        assert_eq!(plan.backend, StorageBackend::Sqlite);
        assert_eq!(plan.dialect(), SqlDialect::Sqlite);
        assert_eq!(plan.display_target(), "/var/lib/llmctl/state.db");
        assert_eq!(plan.to_string(), "sqlite database /var/lib/llmctl/state.db");
        Ok(())
    }

    #[test]
    fn postgres_url_config_plans_postgres_storage_and_redacts_credentials() -> Result<()> {
        let cfg: Config = toml::from_str(
            r#"
            [storage]
            database-url = "postgres://llmctl:secret-token@db.internal:5432/llmctl?sslmode=require"
            "#,
        )?;

        let plan = cfg.storage.connection_plan()?;

        assert_eq!(plan.backend, StorageBackend::Postgres);
        assert_eq!(plan.dialect(), SqlDialect::Postgres);
        assert_eq!(
            plan.display_target(),
            "postgres://llmctl:[REDACTED]@db.internal:5432/llmctl?sslmode=require"
        );
        assert!(!plan.to_string().contains("secret-token"));
        assert!(plan.to_string().contains("[REDACTED]"));
        Ok(())
    }

    #[tokio::test]
    async fn connect_config_rejects_postgres_runtime_storage_without_sqlite_fallback() -> Result<()>
    {
        let cfg: Config = toml::from_str(
            r#"
            [storage]
            database-url = "postgres://llmctl:secret-token@db.internal:5432/llmctl"
            "#,
        )?;

        let err = Storage::connect_config(&cfg.storage)
            .await
            .unwrap_err()
            .to_string();

        assert!(
            err.contains("postgres runtime storage is not enabled"),
            "{err}"
        );
        assert!(!err.contains("secret-token"));
        Ok(())
    }

    #[test]
    fn migration_plan_renders_postgres_compatible_ddl() {
        let plan = StorageMigrationPlan::new(SqlDialect::Postgres);
        let ddl = plan.statements().join("\n");

        assert!(ddl.contains("CREATE TABLE IF NOT EXISTS audit_events"));
        assert!(ddl.contains("allowed BOOLEAN NOT NULL"));
        assert!(ddl.contains("CREATE INDEX IF NOT EXISTS idx_quota_decisions_request_id"));
        assert!(!ddl.contains("PRAGMA"));
    }

    #[tokio::test]
    async fn persists_audit_usage_observation_model_and_quota_records() -> Result<()> {
        let storage = Storage::in_memory().await?;
        let request_id = Uuid::new_v4();
        let audit = AuditEvent::new(
            Some(request_id),
            "alice",
            "platform",
            "chat.create",
            "model/llama",
            "allow",
            json!({"ip": "127.0.0.1"}),
        );
        storage.insert_audit_event(&audit).await?;

        let usage = UsageEvent {
            id: Uuid::new_v4(),
            request_id,
            at: Utc::now(),
            model: "llama".to_string(),
            actor: "alice".to_string(),
            team: "platform".to_string(),
            input_tokens: 10,
            output_tokens: 20,
            latency_ms: 30,
            status: "ok".to_string(),
        };
        storage.insert_usage_event(&usage).await?;

        let observation = ObservationEvent {
            id: Uuid::new_v4(),
            request_id: Some(request_id),
            at: Utc::now(),
            kind: "latency".to_string(),
            model: "llama".to_string(),
            source: "worker".to_string(),
            value: 42.0,
            unit: "ms".to_string(),
            attributes_json: json!({"worker": 1}),
        };
        storage.insert_observation_event(&observation).await?;

        let model = ModelInventoryRecord {
            alias: "llama".to_string(),
            path: "/models/llama.gguf".to_string(),
            role: "chat".to_string(),
            weight: 10,
            updated_at: Utc::now(),
        };
        storage.upsert_model_record(&model).await?;

        let quota = QuotaDecisionRecord {
            id: Uuid::new_v4(),
            request_id: Some(request_id),
            at: Utc::now(),
            actor: "alice".to_string(),
            team: "platform".to_string(),
            model: "llama".to_string(),
            allowed: true,
            reason: "ok".to_string(),
            policy_json: json!({"limit": 100}),
        };
        storage.insert_quota_decision(&quota).await?;

        let from = Utc::now() - chrono::Duration::minutes(1);
        let to = Utc::now() + chrono::Duration::minutes(1);
        assert_eq!(storage.audit_events_between(from, to).await?.len(), 1);
        assert_eq!(storage.audit_events_for_request(request_id).await?.len(), 1);
        assert_eq!(storage.usage_events_between(from, to).await?.len(), 1);
        assert_eq!(storage.usage_events_for_request(request_id).await?.len(), 1);
        assert_eq!(storage.observation_events_between(from, to).await?.len(), 1);
        assert_eq!(
            storage
                .observation_events_for_request(request_id)
                .await?
                .len(),
            1
        );
        assert_eq!(storage.list_models().await?, vec![model]);
        assert_eq!(storage.quota_decisions_between(from, to).await?.len(), 1);
        assert_eq!(
            storage.quota_decisions_for_request(request_id).await?.len(),
            1
        );
        Ok(())
    }

    #[tokio::test]
    async fn counts_audit_events_inside_and_outside_retention_cutoff() -> Result<()> {
        let storage = Storage::in_memory().await?;
        let cutoff = Utc::now() - chrono::Duration::days(30);

        let mut old = AuditEvent::new(
            None,
            "alice",
            "platform",
            "old.action",
            "audit",
            "ok",
            json!({}),
        );
        old.at = cutoff - chrono::Duration::seconds(1);
        storage.insert_audit_event(&old).await?;

        let mut current = AuditEvent::new(
            None,
            "bob",
            "platform",
            "current.action",
            "audit",
            "ok",
            json!({}),
        );
        current.at = cutoff;
        storage.insert_audit_event(&current).await?;

        let counts = storage.audit_retention_counts(cutoff).await?;
        assert_eq!(counts.total, 2);
        assert_eq!(counts.in_retention_window, 1);
        assert_eq!(counts.outside_retention_window, 1);
        Ok(())
    }
}
