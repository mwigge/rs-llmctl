use super::*;

impl Storage {
    pub async fn insert_request_lineage_join(
        &self,
        record: &RequestLineageJoinRecord,
    ) -> Result<()> {
        sqlx::query(&self.sql(
            r#"
            INSERT INTO request_lineage_joins
                (id, request_id, at, lineage_id, model, corpus, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
        ))
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
        let rows = sqlx::query(&self.sql(&format!(
            r#"
            SELECT id, request_id, at, lineage_id, model, corpus, source
            FROM request_lineage_joins
            WHERE at >= ? AND at < ?
            ORDER BY at ASC, id ASC
            {limit_clause}
            "#
        )))
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
        let rows = sqlx::query(&self.sql(
            r#"
            SELECT id, request_id, at, lineage_id, model, corpus, source
            FROM request_lineage_joins
            WHERE request_id = ?
            ORDER BY at ASC, id ASC
            "#,
        ))
        .bind(request_id.to_string())
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(row_to_request_lineage_join_record)
            .collect()
    }
}
