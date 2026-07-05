use super::*;

impl Storage {
    pub async fn insert_observation_event(&self, event: &ObservationEvent) -> Result<()> {
        sqlx::query(&self.sql(
            r#"
            INSERT INTO observation_events
                (id, request_id, at, kind, model, source, value, unit, attributes_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        ))
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
        let rows = sqlx::query(&self.sql(&format!(
            r#"
            SELECT id, request_id, at, kind, model, source, value, unit, attributes_json
            FROM observation_events
            WHERE at >= ? AND at < ?
            ORDER BY at ASC, id ASC
            {limit_clause}
            "#
        )))
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
        let rows = sqlx::query(&self.sql(
            r#"
            SELECT id, request_id, at, kind, model, source, value, unit, attributes_json
            FROM observation_events
            WHERE request_id = ?
            ORDER BY at ASC, id ASC
            "#,
        ))
        .bind(request_id.to_string())
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(row_to_observation_event).collect()
    }
}
