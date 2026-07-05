use crate::config::StorageConfig;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

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

/// True when `database_url` targets an in-memory SQLite database, whose pooled
/// connections are each a distinct empty database (see [`Storage::connect_any`]).
pub(super) fn is_in_memory_sqlite(database_url: &str) -> bool {
    database_url.contains(":memory:") || database_url.contains("mode=memory")
}

pub(super) fn dialect_name(dialect: SqlDialect) -> &'static str {
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
    // IDs, timestamps, and JSON payloads are all serialized to/from strings
    // by the application (`encode_time`, `parse_time`, `.to_string()` on
    // `Uuid`, `parse_json`/`Value::to_string()`), so every dialect stores
    // them as TEXT. Using native UUID/TIMESTAMPTZ/JSONB on Postgres would
    // require the bound parameters to carry those types (or per-query
    // casts), which the sqlx `Any` driver does not do for us.
    let id_type = "TEXT";
    let time_type = "TEXT";
    let json_type = "TEXT";
    let bool_check = match dialect {
        SqlDialect::Sqlite => " CHECK (allowed IN (0, 1))",
        SqlDialect::Postgres => "",
    };

    // NOTE: a `schema_migrations` table used to be created here, but nothing
    // ever wrote to or read from it — it was dead versioning scaffolding.
    // Removing it (rather than wiring a real migration ledger) is the
    // lower-risk cleanup: the schema is applied idempotently via
    // `CREATE TABLE IF NOT EXISTS` / `add_column_if_missing`, so no version
    // ledger is required for correctness, and no code depends on the table.
    vec![
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
