use super::*;

impl Storage {
    pub async fn insert_usage_event(&self, event: &UsageEvent) -> Result<()> {
        sqlx::query(&self.sql(
            r#"
            INSERT INTO usage_events
                (id, request_id, at, model, actor, team, input_tokens, output_tokens, latency_ms, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        ))
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
            &self.sql(&format!(
                r#"
            SELECT id, request_id, at, model, actor, team, input_tokens, output_tokens, latency_ms, status
            FROM usage_events
            WHERE at >= ? AND at < ?
            ORDER BY at ASC, id ASC
            {limit_clause}
            "#
            )),
        )
        .bind(encode_time(from))
        .bind(encode_time(to))
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(row_to_usage_event).collect()
    }

    pub async fn usage_events_for_request(&self, request_id: Uuid) -> Result<Vec<UsageEvent>> {
        let rows = sqlx::query(&self.sql(
            r#"
            SELECT id, request_id, at, model, actor, team, input_tokens, output_tokens, latency_ms, status
            FROM usage_events
            WHERE request_id = ?
            ORDER BY at ASC, id ASC
            "#,
        ))
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
        let row = sqlx::query(&self.sql(&query))
            .bind(encode_time(from))
            .bind(encode_time(to))
            .bind(scope_value)
            .fetch_one(&self.pool)
            .await?;
        i64_to_u64(row.try_get("tokens")?, "tokens")
    }
}
