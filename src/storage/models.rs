use super::*;

impl Storage {
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
        sqlx::query(&self.sql(
            r#"
            INSERT INTO model_inventory (alias, path, role, weight, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(alias) DO UPDATE SET
                path = excluded.path,
                role = excluded.role,
                weight = excluded.weight,
                updated_at = excluded.updated_at
            "#,
        ))
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
}
