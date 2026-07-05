use crate::audit::{AuditEvent, ObservationEvent, UsageEvent};
use crate::config::{ModelConfig, StorageConfig};
use crate::quota::{Principal, QuotaDecision};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::any::{install_default_drivers, AnyPoolOptions};
use sqlx::pool::PoolConnection;
use sqlx::{Any, AnyPool, Connection, Row};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};
use uuid::Uuid;

mod schema;
pub use schema::*;
use schema::{dialect_name, is_in_memory_sqlite};

mod audit;
mod decisions;
mod lineage;
mod models;
mod observation;
mod usage;

#[derive(Debug, Clone)]
pub struct Storage {
    pool: AnyPool,
    dialect: SqlDialect,
    /// Per-scope admission locks (e.g. one per team), so that requests for
    /// unrelated tenants don't serialize on a single global mutex. Created
    /// lazily and never removed; the number of distinct scopes is bounded by
    /// the number of configured teams/subjects, so this does not grow
    /// unboundedly in practice.
    quota_admission_locks: Arc<StdMutex<HashMap<String, Arc<AsyncMutex<()>>>>>,
}

pub struct QuotaAdmissionGuard {
    _local: OwnedMutexGuard<()>,
    postgres_connection: Option<PoolConnection<Any>>,
    advisory_lock_key: i64,
}

impl QuotaAdmissionGuard {
    /// Releases the admission lock. Always call this when done with the
    /// guard — prefer `Storage::with_quota_admission`, which calls this for
    /// you on every code path. The `Drop` impl is only a safety net.
    pub async fn release(mut self) -> Result<()> {
        if let Some(mut connection) = self.postgres_connection.take() {
            match sqlx::query("SELECT pg_advisory_unlock($1)")
                .bind(self.advisory_lock_key)
                .execute(&mut *connection)
                .await
            {
                Ok(_) => {}
                Err(err) => {
                    // The unlock failed, so this connection's session STILL
                    // holds the advisory lock. Returning it to the pool (which
                    // is what dropping a `PoolConnection` does) would hand the
                    // next borrower a locked connection and stall it. Instead,
                    // detach it from the pool and close it: ending the Postgres
                    // session releases every session-level advisory lock it
                    // held server-side. The pool simply opens a fresh
                    // connection in its place.
                    if let Err(close_err) = connection.detach().close().await {
                        tracing::warn!(
                            error = %close_err,
                            "failed to close detached connection after advisory-unlock error"
                        );
                    }
                    return Err(err).context("release quota admission advisory lock");
                }
            }
        }
        Ok(())
    }
}

impl Drop for QuotaAdmissionGuard {
    fn drop(&mut self) {
        if let Some(connection) = self.postgres_connection.take() {
            // Reaching here means `release()` was never called, which is a
            // bug. We cannot run the explicit unlock here: `Drop` can't await,
            // and spawning onto the Tokio runtime panics when the guard is
            // dropped off a runtime thread. Instead, detach the connection from
            // the pool and drop it. Closing the connection ends its Postgres
            // session, which releases every session-level advisory lock it held
            // server-side. Critically, the still-locked connection is never
            // returned to the pool where it could stall the next borrower.
            tracing::warn!(
                "quota admission guard dropped without release(); \
                 closing its connection to release the advisory lock"
            );
            record_quota_admission_release_failure_metric("dropped_without_release");
            drop(connection.detach());
        }
    }
}

/// Cap on how many distinct admission scopes keep a resident lock before idle
/// entries are pruned (see [`Storage::scoped_admission_lock`]).
const QUOTA_ADMISSION_LOCK_MAX_TRACKED: usize = 4096;

fn record_quota_admission_release_failure_metric(reason: &'static str) {
    opentelemetry::global::meter(crate::SERVICE_NAME)
        .u64_counter("llmctl_quota_admission_release_failures_total")
        .with_description("Failures (or missed calls) releasing the quota admission lock")
        .build()
        .add(1, &[opentelemetry::KeyValue::new("reason", reason)]);
}

/// Derives a stable Postgres advisory lock key for an admission scope. A
/// hash collision between two scopes only causes them to share an advisory
/// lock (extra serialization), not a correctness problem.
fn advisory_lock_key_for_scope(scope: &str) -> i64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    scope.hash(&mut hasher);
    hasher.finish() as i64
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
        // Every connection to an in-memory SQLite database is a DISTINCT,
        // initially empty database. `migrate()` runs on a single pooled
        // connection, so any *other* pooled connection would see "no such
        // table" at runtime. Pin the pool to one connection so all queries
        // share the single migrated database. (A file-backed SQLite path or
        // Postgres is required for real connection concurrency.)
        let effective_max = if dialect == SqlDialect::Sqlite && is_in_memory_sqlite(database_url) {
            1
        } else {
            max_connections
        };
        let pool = AnyPoolOptions::new()
            .max_connections(effective_max)
            .connect(database_url)
            .await
            .with_context(|| format!("open {} database", dialect_name(dialect)))?;
        let storage = Self {
            pool,
            dialect,
            quota_admission_locks: Arc::new(StdMutex::new(HashMap::new())),
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
            quota_admission_locks: Arc::new(StdMutex::new(HashMap::new())),
        };
        storage.migrate().await?;
        Ok(storage)
    }

    pub fn pool(&self) -> &AnyPool {
        &self.pool
    }

    /// Acquires the quota admission lock for `scope` (e.g. a team name).
    /// Prefer `with_quota_admission`, which guarantees the lock is released.
    pub async fn quota_admission_guard(&self, scope: &str) -> Result<QuotaAdmissionGuard> {
        let local = self.scoped_admission_lock(scope).lock_owned().await;
        let advisory_lock_key = advisory_lock_key_for_scope(scope);
        let postgres_connection = if self.dialect == SqlDialect::Postgres {
            let mut connection = self.pool.acquire().await?;
            sqlx::query("SELECT pg_advisory_lock($1)")
                .bind(advisory_lock_key)
                .execute(&mut *connection)
                .await?;
            Some(connection)
        } else {
            None
        };
        Ok(QuotaAdmissionGuard {
            _local: local,
            postgres_connection,
            advisory_lock_key,
        })
    }

    /// Runs `f` while holding the quota admission lock for `scope`,
    /// releasing the lock on every path (success, error, or panic-via-Drop
    /// fallback) so callers never need to remember to release it manually.
    pub async fn with_quota_admission<T, F, Fut>(&self, scope: &str, f: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let guard = self.quota_admission_guard(scope).await?;
        let result = f().await;
        if let Err(err) = guard.release().await {
            tracing::warn!(error = %err, scope, "failed to release quota admission lock");
            record_quota_admission_release_failure_metric("release_error");
        }
        result
    }

    fn scoped_admission_lock(&self, scope: &str) -> Arc<AsyncMutex<()>> {
        let mut locks = self
            .quota_admission_locks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Bound growth from many distinct scopes: when the map grows large,
        // drop entries no admission guard currently holds. An outstanding
        // `QuotaAdmissionGuard` keeps an extra `Arc` clone alive (via its
        // `OwnedMutexGuard`), so `strong_count == 1` means only the map
        // references the lock and it can be recreated on demand.
        if locks.len() >= QUOTA_ADMISSION_LOCK_MAX_TRACKED {
            locks.retain(|_, lock| Arc::strong_count(lock) > 1);
        }
        locks
            .entry(scope.to_string())
            .or_insert_with(|| Arc::new(AsyncMutex::new(())))
            .clone()
    }

    /// Rewrite `?` bind placeholders to Postgres's `$1, $2, ...` syntax when
    /// running against Postgres. The sqlx `Any` driver passes SQL through
    /// verbatim, so `?` placeholders only work as-is on SQLite.
    fn sql<'a>(&self, query: &'a str) -> Cow<'a, str> {
        rewrite_placeholders(self.dialect, query)
    }

    pub async fn migrate(&self) -> Result<()> {
        for statement in StorageMigrationPlan::new(self.dialect).statements() {
            sqlx::query(statement).execute(&self.pool).await?;
        }
        add_column_if_missing(
            &self.pool,
            self.dialect,
            "observation_events",
            "request_id TEXT",
        )
        .await?;

        Ok(())
    }
}

/// Rewrite `?` bind placeholders to Postgres's `$1, $2, ...` syntax for the
/// Postgres dialect; left unchanged for SQLite. None of the queries in this
/// module embed `?` inside string literals, so a simple sequential
/// left-to-right replace is sufficient.
fn rewrite_placeholders(dialect: SqlDialect, query: &str) -> Cow<'_, str> {
    if dialect != SqlDialect::Postgres {
        return Cow::Borrowed(query);
    }
    let mut rewritten = String::with_capacity(query.len() + 8);
    let mut placeholder = 0u32;
    for ch in query.chars() {
        if ch == '?' {
            placeholder += 1;
            rewritten.push('$');
            rewritten.push_str(&placeholder.to_string());
        } else {
            rewritten.push(ch);
        }
    }
    Cow::Owned(rewritten)
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

async fn add_column_if_missing(
    pool: &AnyPool,
    dialect: SqlDialect,
    table: &str,
    column_sql: &str,
) -> Result<()> {
    match dialect {
        // Postgres has supported `ADD COLUMN IF NOT EXISTS` since 9.6, so no
        // separate existence check is needed.
        SqlDialect::Postgres => {
            let alter = format!("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column_sql}");
            sqlx::query(&alter).execute(pool).await?;
        }
        SqlDialect::Sqlite => {
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
                if let Err(err) = sqlx::query(&alter).execute(pool).await {
                    // TOCTOU guard: another process/connection may have added
                    // the column between the PRAGMA check above and this ALTER
                    // (SQLite has no `ADD COLUMN IF NOT EXISTS`). It reports the
                    // race as a "duplicate column name" error, which is benign
                    // here — the column now exists, which is the desired
                    // postcondition. Any other error is a real failure.
                    let message = err.to_string().to_ascii_lowercase();
                    if !message.contains("duplicate column") {
                        return Err(err)
                            .with_context(|| format!("add column `{column_sql}` to {table}"));
                    }
                }
            }
        }
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
mod tests;
