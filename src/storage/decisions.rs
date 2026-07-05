use super::*;

impl Storage {
    pub async fn insert_quota_decision(&self, decision: &QuotaDecisionRecord) -> Result<()> {
        sqlx::query(&self.sql(
            r#"
            INSERT INTO quota_decisions
                (id, request_id, at, actor, team, model, allowed, reason, policy_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        ))
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
        let rows = sqlx::query(&self.sql(&format!(
            r#"
            SELECT id, request_id, at, actor, team, model, allowed, reason, policy_json
            FROM quota_decisions
            WHERE at >= ? AND at < ?
            ORDER BY at ASC, id ASC
            {limit_clause}
            "#
        )))
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
        let rows = sqlx::query(&self.sql(
            r#"
            SELECT id, request_id, at, actor, team, model, allowed, reason, policy_json
            FROM quota_decisions
            WHERE request_id = ?
            ORDER BY at ASC, id ASC
            "#,
        ))
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
        let row = sqlx::query(&self.sql(query))
            .bind(encode_time(from))
            .bind(encode_time(to))
            .bind(scope_value)
            .fetch_one(&self.pool)
            .await?;
        i64_to_u64(row.try_get("count")?, "count")
    }
}
