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
    assert_storage_round_trip(&storage).await
}

/// Exercises a full read/write round trip across every record type and
/// query variant (including the dialect-specific `allowed_quota_decision_count`
/// branches). Shared by the SQLite in-memory test and the Postgres
/// integration test so both dialects are checked against the same
/// assertions.
async fn assert_storage_round_trip(storage: &Storage) -> Result<()> {
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
    assert_eq!(
        storage
            .usage_tokens_total(&principal, true, from, to)
            .await?,
        30
    );
    assert_eq!(
        storage
            .usage_tokens_total(&principal, false, from, to)
            .await?,
        30
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

#[test]
fn rewrite_placeholders_leaves_sqlite_queries_unchanged() {
    let query = "INSERT INTO t (a, b, c) VALUES (?, ?, ?)";
    assert_eq!(
        rewrite_placeholders(SqlDialect::Sqlite, query),
        Cow::Borrowed(query)
    );
}

#[test]
fn rewrite_placeholders_numbers_postgres_placeholders_sequentially() {
    let query = "INSERT INTO t (a, b, c) VALUES (?, ?, ?)";
    assert_eq!(
        rewrite_placeholders(SqlDialect::Postgres, query),
        "INSERT INTO t (a, b, c) VALUES ($1, $2, $3)"
    );
}

#[test]
fn rewrite_placeholders_handles_repeated_where_clauses() {
    let query = "SELECT * FROM t WHERE at >= ? AND at < ? AND actor = ?";
    assert_eq!(
        rewrite_placeholders(SqlDialect::Postgres, query),
        "SELECT * FROM t WHERE at >= $1 AND at < $2 AND actor = $3"
    );
}

#[test]
fn rewrite_placeholders_handles_advisory_lock_calls() {
    assert_eq!(
        rewrite_placeholders(SqlDialect::Postgres, "SELECT pg_advisory_lock(?)"),
        "SELECT pg_advisory_lock($1)"
    );
}

#[test]
fn detects_in_memory_sqlite_targets() {
    assert!(is_in_memory_sqlite("sqlite://:memory:"));
    assert!(is_in_memory_sqlite("sqlite::memory:"));
    assert!(is_in_memory_sqlite(
        "sqlite://file:foo?mode=memory&cache=shared"
    ));
    assert!(!is_in_memory_sqlite(
        "sqlite:///var/lib/rs-llmctl/state.db?mode=rwc"
    ));
}

/// Bug 15 regression: an in-memory sqlite store opened while requesting a
/// multi-connection pool must be pinned to a single connection and still be
/// able to query a migrated table.
///
/// Fail-before: without the clamp, `connect_any` honored `max_connections =
/// 5`. `migrate()` created the schema on one pooled connection, but every
/// other connection to an in-memory sqlite database is a DISTINCT empty
/// database, so a query routed to a fresh connection failed at runtime with
/// "no such table: audit_events". The clamp forces one shared connection.
#[tokio::test]
async fn in_memory_sqlite_pool_is_pinned_to_single_connection() -> Result<()> {
    // Request 5 connections deliberately; the fix must clamp this to 1 so
    // migration and every subsequent query share the one in-memory DB.
    let storage = Storage::connect_any("sqlite::memory:", SqlDialect::Sqlite, 5).await?;
    assert_eq!(
        storage.pool.options().get_max_connections(),
        1,
        "in-memory sqlite pool must be clamped to a single connection"
    );

    // The single shared connection sees the migrated schema.
    let row = sqlx::query("SELECT COUNT(*) AS count FROM audit_events")
        .fetch_one(&storage.pool)
        .await
        .context("query migrated audit_events table")?;
    let count: i64 = row.try_get("count")?;
    assert_eq!(count, 0);
    Ok(())
}

/// A file-backed sqlite target must NOT be clamped — real connection
/// concurrency is preserved there (only in-memory needs a single owner).
#[tokio::test]
async fn file_backed_sqlite_pool_is_not_clamped() -> Result<()> {
    let dir = tempfile::tempdir().context("tempdir")?;
    let db_path = dir.path().join("state.db");
    let url = format!("sqlite://{}?mode=rwc", db_path.display());
    let storage = Storage::connect_any(&url, SqlDialect::Sqlite, 5).await?;
    assert_eq!(storage.pool.options().get_max_connections(), 5);
    Ok(())
}

/// Exercises the storage layer against a real Postgres instance.
/// Run with: `TEST_DATABASE_URL=postgres://user:pass@host/db cargo test --lib -- --ignored storage_works_against_real_postgres`
#[tokio::test]
#[ignore = "requires a running Postgres instance via TEST_DATABASE_URL"]
async fn storage_works_against_real_postgres() -> Result<()> {
    let database_url =
        std::env::var("TEST_DATABASE_URL").context("set TEST_DATABASE_URL to run this test")?;
    let storage = Storage::connect_any(&database_url, SqlDialect::Postgres, 2).await?;
    assert_eq!(storage.dialect, SqlDialect::Postgres);

    // Make this test repeatable against a persistent (non-ephemeral)
    // Postgres instance by clearing out rows left over from a previous run.
    for table in [
        "audit_events",
        "usage_events",
        "observation_events",
        "model_inventory",
        "quota_decisions",
        "request_lineage_joins",
    ] {
        sqlx::query(&format!("TRUNCATE TABLE {table}"))
            .execute(&storage.pool)
            .await?;
    }

    // Exercises the advisory-lock based quota admission path, which is
    // only taken on Postgres.
    let guard = storage.quota_admission_guard("team:platform").await?;
    guard.release().await?;

    // Same assertions as the SQLite in-memory test, including the
    // `allowed_quota_decision_count` branches that use `allowed = TRUE`
    // on Postgres vs `allowed = 1` on SQLite.
    assert_storage_round_trip(&storage).await?;

    let cutoff = Utc::now() + chrono::Duration::seconds(1);
    let counts = storage.audit_retention_counts(cutoff).await?;
    assert_eq!(counts.total, counts.outside_retention_window);
    let deleted = storage.delete_audit_events_before(cutoff).await?;
    assert_eq!(deleted, counts.total);

    Ok(())
}
