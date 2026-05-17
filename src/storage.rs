use crate::audit::{AuditEvent, ObservationEvent, UsageEvent};
use crate::config::{ModelConfig, StorageConfig};
use crate::quota::{Principal, QuotaDecision};
use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::any::{install_default_drivers, AnyPoolOptions};
use sqlx::pool::PoolConnection;
use sqlx::{Any, AnyPool, Row};
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};
use uuid::Uuid;

const QUOTA_ADVISORY_LOCK_KEY: i64 = 0x6c6c_6d63_746c;

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

fn dialect_name(dialect: SqlDialect) -> &'static str {
    match dialect {
        SqlDialect::Sqlite => "sqlite",
        SqlDialect::Postgres => "postgres",
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
        format!(
            r#"
            CREATE TABLE IF NOT EXISTS request_lineage_joins (
                id {id_type} PRIMARY KEY NOT NULL,
                request_id {id_type} NOT NULL,
                at {time_type} NOT NULL,
                lineage_id TEXT NOT NULL,
                model TEXT,
                corpus TEXT,
                source TEXT NOT NULL
            )
            "#
        ),
        "CREATE INDEX IF NOT EXISTS idx_request_lineage_joins_request_id ON request_lineage_joins(request_id)"
            .to_string(),
        "CREATE INDEX IF NOT EXISTS idx_request_lineage_joins_lineage_id ON request_lineage_joins(lineage_id)"
            .to_string(),
        "CREATE INDEX IF NOT EXISTS idx_request_lineage_joins_model ON request_lineage_joins(model)"
            .to_string(),
        "CREATE INDEX IF NOT EXISTS idx_request_lineage_joins_corpus ON request_lineage_joins(corpus)"
            .to_string(),
    ]
}

#[derive(Debug, Clone)]
pub struct Storage {
    pool: AnyPool,
    dialect: SqlDialect,
    quota_admission_lock: Arc<AsyncMutex<()>>,
}

pub struct QuotaAdmissionGuard {
    _local: OwnedMutexGuard<()>,
    postgres_connection: Option<PoolConnection<Any>>,
}

impl QuotaAdmissionGuard {
    pub async fn release(mut self) -> Result<()> {
        if let Some(mut connection) = self.postgres_connection.take() {
            sqlx::query("SELECT pg_advisory_unlock(?)")
                .bind(QUOTA_ADVISORY_LOCK_KEY)
                .execute(&mut *connection)
                .await?;
        }
        Ok(())
    }
}

impl Drop for QuotaAdmissionGuard {
    fn drop(&mut self) {
        if let Some(mut connection) = self.postgres_connection.take() {
            tokio::spawn(async move {
                let _ = sqlx::query("SELECT pg_advisory_unlock(?)")
                    .bind(QUOTA_ADVISORY_LOCK_KEY)
                    .execute(&mut *connection)
                    .await;
            });
        }
    }
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RequestLineageJoinRecord {
    pub id: Uuid,
    pub request_id: Uuid,
    pub at: DateTime<Utc>,
    pub lineage_id: String,
    pub model: Option<String>,
    pub corpus: Option<String>,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApiKeyUsageRecord {
    pub request_id: Uuid,
    pub api_key_id: String,
    pub actor: String,
    pub team: String,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub latency_ms: u64,
    pub status: String,
    pub usage_at: DateTime<Utc>,
    pub first_audit_at: DateTime<Utc>,
    pub audit_outcome: String,
}

#[derive(Debug, Clone)]
struct ApiKeyAuditMetadata {
    api_key_id: String,
    first_audit_at: DateTime<Utc>,
    audit_outcome: String,
}

impl RequestLineageJoinRecord {
    pub fn new(
        request_id: Uuid,
        lineage_id: impl Into<String>,
        model: Option<String>,
        corpus: Option<String>,
        source: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            request_id,
            at: Utc::now(),
            lineage_id: lineage_id.into(),
            model,
            corpus,
            source: source.into(),
        }
    }
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
        let max_connections = config.max_connections.max(1);
        match plan.backend {
            StorageBackend::Sqlite => {
                Self::connect_sqlite_target(plan.target(), max_connections).await
            }
            StorageBackend::Postgres => {
                if max_connections < 2 {
                    anyhow::bail!(
                        "postgres storage requires max_connections >= 2 for atomic quota admission"
                    );
                }
                Self::connect_any(plan.target(), SqlDialect::Postgres, max_connections).await
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

        Self::connect_any(
            &format!("sqlite://{}?mode=rwc", db_path.display()),
            SqlDialect::Sqlite,
            5,
        )
        .await
    }

    async fn connect_any(
        database_url: &str,
        dialect: SqlDialect,
        max_connections: u32,
    ) -> Result<Self> {
        install_default_drivers();
        let pool = AnyPoolOptions::new()
            .max_connections(max_connections)
            .connect(database_url)
            .await
            .with_context(|| format!("open {} database", dialect_name(dialect)))?;
        let storage = Self {
            pool,
            dialect,
            quota_admission_lock: Arc::new(AsyncMutex::new(())),
        };
        storage.migrate().await?;
        Ok(storage)
    }

    async fn connect_sqlite_target(target: &str, max_connections: u32) -> Result<Self> {
        if let Some(path) = target.strip_prefix("sqlite://") {
            if path != ":memory:" {
                let path_without_query = path.split_once('?').map(|(path, _)| path).unwrap_or(path);
                if let Some(parent) = Path::new(path_without_query).parent() {
                    tokio::fs::create_dir_all(parent).await.with_context(|| {
                        format!("create storage directory {}", parent.display())
                    })?;
                }
            }
            if path.contains("?mode=") || path == ":memory:" {
                Self::connect_any(target, SqlDialect::Sqlite, max_connections).await
            } else {
                Self::connect_any(
                    &format!("{target}?mode=rwc"),
                    SqlDialect::Sqlite,
                    max_connections,
                )
                .await
            }
        } else {
            Self::connect(target).await
        }
    }

    pub async fn in_memory() -> Result<Self> {
        install_default_drivers();
        let pool = AnyPoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .context("open in-memory sqlite database")?;
        let storage = Self {
            pool,
            dialect: SqlDialect::Sqlite,
            quota_admission_lock: Arc::new(AsyncMutex::new(())),
        };
        storage.migrate().await?;
        Ok(storage)
    }

    pub fn pool(&self) -> &AnyPool {
        &self.pool
    }

    pub async fn quota_admission_guard(&self) -> Result<QuotaAdmissionGuard> {
        let local = self.quota_admission_lock.clone().lock_owned().await;
        let postgres_connection = if self.dialect == SqlDialect::Postgres {
            let mut connection = self.pool.acquire().await?;
            sqlx::query("SELECT pg_advisory_lock(?)")
                .bind(QUOTA_ADVISORY_LOCK_KEY)
                .execute(&mut *connection)
                .await?;
            Some(connection)
        } else {
            None
        };
        Ok(QuotaAdmissionGuard {
            _local: local,
            postgres_connection,
        })
    }

    pub async fn migrate(&self) -> Result<()> {
        for statement in StorageMigrationPlan::new(self.dialect).statements() {
            sqlx::query(statement).execute(&self.pool).await?;
        }
        if self.dialect == SqlDialect::Sqlite {
            add_column_if_missing(&self.pool, "observation_events", "request_id TEXT").await?;
        }

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
        self.audit_events_between_limited(from, to, None).await
    }

    pub async fn audit_events_between_limited(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
        limit: Option<usize>,
    ) -> Result<Vec<AuditEvent>> {
        let limit_clause = limit_clause(limit);
        let rows = sqlx::query(&format!(
            r#"
            SELECT id, request_id, at, actor, team, action, resource, outcome, detail_json
            FROM audit_events
            WHERE at >= ? AND at < ?
            ORDER BY at ASC, id ASC
            {limit_clause}
            "#
        ))
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
        self.usage_events_between_limited(from, to, None).await
    }

    pub async fn usage_events_between_limited(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
        limit: Option<usize>,
    ) -> Result<Vec<UsageEvent>> {
        let limit_clause = limit_clause(limit);
        let rows = sqlx::query(
            &format!(
                r#"
            SELECT id, request_id, at, model, actor, team, input_tokens, output_tokens, latency_ms, status
            FROM usage_events
            WHERE at >= ? AND at < ?
            ORDER BY at ASC, id ASC
            {limit_clause}
            "#
            ),
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

    pub async fn api_key_usage_between(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<ApiKeyUsageRecord>> {
        let audit_events = self.audit_events_between(from, to).await?;
        let usage_events = self.usage_events_between(from, to).await?;
        Ok(join_api_key_usage(audit_events, usage_events))
    }

    pub async fn api_key_usage_for_request(
        &self,
        request_id: Uuid,
    ) -> Result<Vec<ApiKeyUsageRecord>> {
        let audit_events = self.audit_events_for_request(request_id).await?;
        let usage_events = self.usage_events_for_request(request_id).await?;
        Ok(join_api_key_usage(audit_events, usage_events))
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
        self.observation_events_between_limited(from, to, None)
            .await
    }

    pub async fn observation_events_between_limited(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
        limit: Option<usize>,
    ) -> Result<Vec<ObservationEvent>> {
        let limit_clause = limit_clause(limit);
        let rows = sqlx::query(&format!(
            r#"
            SELECT id, request_id, at, kind, model, source, value, unit, attributes_json
            FROM observation_events
            WHERE at >= ? AND at < ?
            ORDER BY at ASC, id ASC
            {limit_clause}
            "#
        ))
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
        .bind(decision.allowed)
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
        self.quota_decisions_between_limited(from, to, None).await
    }

    pub async fn quota_decisions_between_limited(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
        limit: Option<usize>,
    ) -> Result<Vec<QuotaDecisionRecord>> {
        let limit_clause = limit_clause(limit);
        let rows = sqlx::query(&format!(
            r#"
            SELECT id, request_id, at, actor, team, model, allowed, reason, policy_json
            FROM quota_decisions
            WHERE at >= ? AND at < ?
            ORDER BY at ASC, id ASC
            {limit_clause}
            "#
        ))
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

    pub async fn insert_request_lineage_join(
        &self,
        record: &RequestLineageJoinRecord,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO request_lineage_joins
                (id, request_id, at, lineage_id, model, corpus, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(record.id.to_string())
        .bind(record.request_id.to_string())
        .bind(encode_time(record.at))
        .bind(&record.lineage_id)
        .bind(&record.model)
        .bind(&record.corpus)
        .bind(&record.source)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn request_lineage_joins(&self) -> Result<Vec<RequestLineageJoinRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, request_id, at, lineage_id, model, corpus, source
            FROM request_lineage_joins
            ORDER BY at ASC, id ASC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(row_to_request_lineage_join_record)
            .collect()
    }

    pub async fn request_lineage_joins_between(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<RequestLineageJoinRecord>> {
        self.request_lineage_joins_between_limited(from, to, None)
            .await
    }

    pub async fn request_lineage_joins_between_limited(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
        limit: Option<usize>,
    ) -> Result<Vec<RequestLineageJoinRecord>> {
        let limit_clause = limit_clause(limit);
        let rows = sqlx::query(&format!(
            r#"
            SELECT id, request_id, at, lineage_id, model, corpus, source
            FROM request_lineage_joins
            WHERE at >= ? AND at < ?
            ORDER BY at ASC, id ASC
            {limit_clause}
            "#
        ))
        .bind(encode_time(from))
        .bind(encode_time(to))
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(row_to_request_lineage_join_record)
            .collect()
    }

    pub async fn request_lineage_joins_for_request(
        &self,
        request_id: Uuid,
    ) -> Result<Vec<RequestLineageJoinRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, request_id, at, lineage_id, model, corpus, source
            FROM request_lineage_joins
            WHERE request_id = ?
            ORDER BY at ASC, id ASC
            "#,
        )
        .bind(request_id.to_string())
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(row_to_request_lineage_join_record)
            .collect()
    }

    pub async fn allowed_quota_decision_count(
        &self,
        principal: &Principal,
        subject_scoped: bool,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<u64> {
        let (query, scope_value) = match (self.dialect, subject_scoped) {
            (SqlDialect::Sqlite, true) => (
                r#"
                SELECT COUNT(*) AS count
                FROM quota_decisions
                WHERE allowed = 1
                  AND at >= ?
                  AND at < ?
                  AND actor = ?
                "#,
                &principal.subject,
            ),
            (SqlDialect::Sqlite, false) => (
                r#"
                SELECT COUNT(*) AS count
                FROM quota_decisions
                WHERE allowed = 1
                  AND at >= ?
                  AND at < ?
                  AND team = ?
                "#,
                &principal.team,
            ),
            (SqlDialect::Postgres, true) => (
                r#"
                SELECT COUNT(*) AS count
                FROM quota_decisions
                WHERE allowed = TRUE
                  AND at >= ?
                  AND at < ?
                  AND actor = ?
                "#,
                &principal.subject,
            ),
            (SqlDialect::Postgres, false) => (
                r#"
                SELECT COUNT(*) AS count
                FROM quota_decisions
                WHERE allowed = TRUE
                  AND at >= ?
                  AND at < ?
                  AND team = ?
                "#,
                &principal.team,
            ),
        };
        let row = sqlx::query(query)
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

fn limit_clause(limit: Option<usize>) -> String {
    limit
        .map(|limit| format!("LIMIT {}", limit.max(1)))
        .unwrap_or_default()
}

fn i64_to_u64(value: i64, field: &str) -> Result<u64> {
    u64::try_from(value).with_context(|| format!("read non-negative integer field {field}"))
}

fn i64_to_u32(value: i64, field: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("read u32 integer field {field}"))
}

fn row_to_audit_event(row: sqlx::any::AnyRow) -> Result<AuditEvent> {
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

fn join_api_key_usage(
    audit_events: Vec<AuditEvent>,
    usage_events: Vec<UsageEvent>,
) -> Vec<ApiKeyUsageRecord> {
    let mut audit_by_request = BTreeMap::<Uuid, ApiKeyAuditMetadata>::new();
    for event in audit_events {
        let Some(request_id) = event.request_id else {
            continue;
        };
        let Some(api_key_id) = event
            .detail_json
            .get("api_key_id")
            .and_then(|value| value.as_str())
        else {
            continue;
        };
        audit_by_request
            .entry(request_id)
            .and_modify(|metadata| {
                metadata.audit_outcome = event.outcome.clone();
            })
            .or_insert_with(|| ApiKeyAuditMetadata {
                api_key_id: api_key_id.to_string(),
                first_audit_at: event.at,
                audit_outcome: event.outcome,
            });
    }

    usage_events
        .into_iter()
        .filter_map(|event| {
            let metadata = audit_by_request.get(&event.request_id)?;
            Some(ApiKeyUsageRecord {
                request_id: event.request_id,
                api_key_id: metadata.api_key_id.clone(),
                actor: event.actor,
                team: event.team,
                model: event.model,
                input_tokens: event.input_tokens,
                output_tokens: event.output_tokens,
                total_tokens: event.input_tokens.saturating_add(event.output_tokens),
                latency_ms: event.latency_ms,
                status: event.status,
                usage_at: event.at,
                first_audit_at: metadata.first_audit_at,
                audit_outcome: metadata.audit_outcome.clone(),
            })
        })
        .collect()
}

fn row_to_usage_event(row: sqlx::any::AnyRow) -> Result<UsageEvent> {
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

fn row_to_observation_event(row: sqlx::any::AnyRow) -> Result<ObservationEvent> {
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

async fn add_column_if_missing(pool: &AnyPool, table: &str, column_sql: &str) -> Result<()> {
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

fn row_to_model_inventory_record(row: sqlx::any::AnyRow) -> Result<ModelInventoryRecord> {
    Ok(ModelInventoryRecord {
        alias: row.try_get("alias")?,
        path: row.try_get("path")?,
        role: row.try_get("role")?,
        weight: i64_to_u32(row.try_get("weight")?, "weight")?,
        updated_at: parse_time(&row.try_get::<String, _>("updated_at")?)?,
    })
}

fn row_to_quota_decision_record(row: sqlx::any::AnyRow) -> Result<QuotaDecisionRecord> {
    let allowed = row
        .try_get::<bool, _>("allowed")
        .or_else(|_| row.try_get::<i64, _>("allowed").map(|value| value != 0))?;
    Ok(QuotaDecisionRecord {
        id: parse_uuid(&row.try_get::<String, _>("id")?)?,
        request_id: parse_optional_uuid(row.try_get::<Option<String>, _>("request_id")?)?,
        at: parse_time(&row.try_get::<String, _>("at")?)?,
        actor: row.try_get("actor")?,
        team: row.try_get("team")?,
        model: row.try_get("model")?,
        allowed,
        reason: row.try_get("reason")?,
        policy_json: parse_json(row.try_get("policy_json")?)?,
    })
}

fn row_to_request_lineage_join_record(row: sqlx::any::AnyRow) -> Result<RequestLineageJoinRecord> {
    Ok(RequestLineageJoinRecord {
        id: parse_uuid(&row.try_get::<String, _>("id")?)?,
        request_id: parse_uuid(&row.try_get::<String, _>("request_id")?)?,
        at: parse_time(&row.try_get::<String, _>("at")?)?,
        lineage_id: row.try_get("lineage_id")?,
        model: row.try_get("model")?,
        corpus: row.try_get("corpus")?,
        source: row.try_get("source")?,
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

    #[test]
    fn connect_config_accepts_postgres_runtime_plan_without_exposing_passwords() -> Result<()> {
        let cfg: Config = toml::from_str(
            r#"
            [storage]
            database-url = "postgres://llmctl:secret-token@db.internal:5432/llmctl"
            "#,
        )?;

        let plan = cfg.storage.connection_plan()?;

        assert_eq!(plan.backend, StorageBackend::Postgres);
        assert_eq!(plan.dialect(), SqlDialect::Postgres);
        assert!(!plan.display_target().contains("secret-token"));
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
            json!({
                "api_key_id": "alice-key",
                "api_key_subject": "alice",
                "api_key_team": "platform",
                "ip": "127.0.0.1"
            }),
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

        let lineage = RequestLineageJoinRecord::new(
            request_id,
            "corpus:internal-docs",
            Some("llama".to_string()),
            Some("internal-docs".to_string()),
            "chat.completions",
        );
        storage.insert_request_lineage_join(&lineage).await?;

        let from = Utc::now() - chrono::Duration::minutes(1);
        let to = Utc::now() + chrono::Duration::minutes(1);
        let principal = Principal {
            subject: "alice".to_string(),
            team: "platform".to_string(),
            scopes: vec!["chat".to_string()],
            key_id: Some("alice-key".to_string()),
            key_owner: None,
            key_purpose: None,
            key_status: Some("active".to_string()),
        };
        assert_eq!(storage.audit_events_between(from, to).await?.len(), 1);
        assert_eq!(storage.audit_events_for_request(request_id).await?.len(), 1);
        assert_eq!(storage.usage_events_between(from, to).await?.len(), 1);
        assert_eq!(storage.usage_events_for_request(request_id).await?.len(), 1);
        assert_eq!(
            storage.api_key_usage_for_request(request_id).await?,
            vec![ApiKeyUsageRecord {
                request_id,
                api_key_id: "alice-key".to_string(),
                actor: "alice".to_string(),
                team: "platform".to_string(),
                model: "llama".to_string(),
                input_tokens: 10,
                output_tokens: 20,
                total_tokens: 30,
                latency_ms: 30,
                status: "ok".to_string(),
                usage_at: usage.at,
                first_audit_at: audit.at,
                audit_outcome: "allow".to_string(),
            }]
        );
        assert_eq!(storage.api_key_usage_between(from, to).await?.len(), 1);
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
        assert_eq!(
            storage
                .allowed_quota_decision_count(&principal, true, from, to)
                .await?,
            1
        );
        assert_eq!(
            storage
                .allowed_quota_decision_count(&principal, false, from, to)
                .await?,
            1
        );
        assert_eq!(
            storage.request_lineage_joins().await?,
            vec![lineage.clone()]
        );
        assert_eq!(
            storage
                .request_lineage_joins_for_request(request_id)
                .await?,
            vec![lineage]
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
