use super::*;

impl Storage {
    pub async fn insert_audit_event(&self, event: &AuditEvent) -> Result<()> {
        sqlx::query(&self.sql(
            r#"
            INSERT INTO audit_events
                (id, request_id, at, actor, team, action, resource, outcome, detail_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        ))
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
        let rows = sqlx::query(&self.sql(&format!(
            r#"
            SELECT id, request_id, at, actor, team, action, resource, outcome, detail_json
            FROM audit_events
            WHERE at >= ? AND at < ?
            ORDER BY at ASC, id ASC
            {limit_clause}
            "#
        )))
        .bind(encode_time(from))
        .bind(encode_time(to))
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(row_to_audit_event).collect()
    }

    pub async fn audit_events_for_request(&self, request_id: Uuid) -> Result<Vec<AuditEvent>> {
        let rows = sqlx::query(&self.sql(
            r#"
            SELECT id, request_id, at, actor, team, action, resource, outcome, detail_json
            FROM audit_events
            WHERE request_id = ?
            ORDER BY at ASC, id ASC
            "#,
        ))
        .bind(request_id.to_string())
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(row_to_audit_event).collect()
    }

    pub async fn audit_retention_counts(
        &self,
        cutoff: DateTime<Utc>,
    ) -> Result<AuditRetentionCounts> {
        let row = sqlx::query(&self.sql(
            r#"
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(CASE WHEN at >= ? THEN 1 ELSE 0 END), 0) AS in_retention_window,
                COALESCE(SUM(CASE WHEN at < ? THEN 1 ELSE 0 END), 0) AS outside_retention_window
            FROM audit_events
            "#,
        ))
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
        let result = sqlx::query(&self.sql("DELETE FROM audit_events WHERE at < ?"))
            .bind(encode_time(cutoff))
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }
}
