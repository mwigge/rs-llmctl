use crate::contracts::{DataContract, FieldContract};
use anyhow::{anyhow, Context, Result};
use arrow::array::{
    ArrayRef, Float64Array, StringArray, TimestampMillisecondArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Utc};
use parquet::arrow::ArrowWriter;
use serde_json::Value;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub const DEFAULT_ARROW_BATCH_ROWS: usize = 8192;

pub fn write_arrow_ipc(path: &Path, contract: &DataContract, rows: &[Value]) -> Result<u64> {
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let schema = arrow_schema(contract);
    let mut writer = FileWriter::try_new(file, schema.as_ref())?;
    if rows.is_empty() {
        writer.write(&record_batch_with_schema(contract, &schema, rows)?)?;
    } else {
        for chunk in rows.chunks(DEFAULT_ARROW_BATCH_ROWS) {
            writer.write(&record_batch_with_schema(contract, &schema, chunk)?)?;
        }
    }
    writer.finish()?;
    Ok(rows.len() as u64)
}

pub fn write_parquet(path: &Path, contract: &DataContract, rows: &[Value]) -> Result<u64> {
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let schema = arrow_schema(contract);
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;
    if rows.is_empty() {
        writer.write(&record_batch_with_schema(contract, &schema, rows)?)?;
    } else {
        for chunk in rows.chunks(DEFAULT_ARROW_BATCH_ROWS) {
            writer.write(&record_batch_with_schema(contract, &schema, chunk)?)?;
        }
    }
    writer.close()?;
    Ok(rows.len() as u64)
}

fn arrow_schema(contract: &DataContract) -> Arc<Schema> {
    let fields = contract
        .fields
        .iter()
        .map(|field| Field::new(field.name, arrow_type(field.data_type), field.nullable))
        .collect::<Vec<_>>();
    Arc::new(Schema::new(fields))
}

fn record_batch_with_schema(
    contract: &DataContract,
    schema: &Arc<Schema>,
    rows: &[Value],
) -> Result<RecordBatch> {
    let columns = contract
        .fields
        .iter()
        .map(|field| array_for_field(field, rows))
        .collect::<Result<Vec<_>>>()?;
    Ok(RecordBatch::try_new(schema.clone(), columns)?)
}

fn arrow_type(data_type: &str) -> DataType {
    if data_type.starts_with("timestamp") {
        DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into()))
    } else if data_type == "uint64" {
        DataType::UInt64
    } else if data_type == "uint32" {
        DataType::UInt32
    } else if data_type == "float64" {
        DataType::Float64
    } else {
        DataType::Utf8
    }
}

fn array_for_field(field: &FieldContract, rows: &[Value]) -> Result<ArrayRef> {
    validate_field_values(field, rows)?;
    let name = field.name;
    let data_type = field.data_type;
    if data_type.starts_with("timestamp") {
        let values = rows
            .iter()
            .map(|row| timestamp_millis(row.get(name)))
            .collect::<Result<Vec<_>>>()?;
        return Ok(Arc::new(TimestampMillisecondArray::from(values)));
    }

    match data_type {
        "uint64" => Ok(Arc::new(UInt64Array::from(
            rows.iter()
                .map(|row| optional_u64(row.get(name)))
                .collect::<Result<Vec<_>>>()?,
        ))),
        "uint32" => Ok(Arc::new(UInt32Array::from(
            rows.iter()
                .map(|row| optional_u32(row.get(name)))
                .collect::<Result<Vec<_>>>()?,
        ))),
        "float64" => Ok(Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| optional_f64(row.get(name)))
                .collect::<Result<Vec<_>>>()?,
        ))),
        _ => Ok(Arc::new(StringArray::from(
            rows.iter()
                .map(|row| optional_string(row.get(name)))
                .collect::<Vec<_>>(),
        ))),
    }
}

fn validate_field_values(field: &FieldContract, rows: &[Value]) -> Result<()> {
    for (index, row) in rows.iter().enumerate() {
        let value = row.get(field.name);
        if !field.nullable && matches!(value, None | Some(Value::Null)) {
            return Err(anyhow!(
                "row {index} missing required field `{}` for data contract",
                field.name
            ));
        }

        let Some(value) = value else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        let valid = if field.data_type.starts_with("timestamp") {
            value
                .as_str()
                .and_then(|text| DateTime::parse_from_rfc3339(text).ok())
                .is_some()
        } else {
            match field.data_type {
                "uint64" => value.as_u64().is_some(),
                "uint32" => value
                    .as_u64()
                    .is_some_and(|value| u32::try_from(value).is_ok()),
                "float64" => value.as_f64().is_some(),
                "utf8" => value.as_str().is_some(),
                _ => true,
            }
        };
        if !valid {
            return Err(anyhow!(
                "row {index} field `{}` does not match data contract type `{}`",
                field.name,
                field.data_type
            ));
        }
    }
    Ok(())
}

fn optional_string(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(value)) => Some(value.clone()),
        Some(Value::Null) | None => None,
        Some(value) => Some(value.to_string()),
    }
}

fn optional_u64(value: Option<&Value>) -> Result<Option<u64>> {
    match value {
        Some(Value::Number(value)) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| anyhow!("expected unsigned integer")),
        Some(Value::Null) | None => Ok(None),
        Some(value) => Err(anyhow!("expected unsigned integer, got {value}")),
    }
}

fn optional_u32(value: Option<&Value>) -> Result<Option<u32>> {
    optional_u64(value).and_then(|value| {
        value
            .map(u32::try_from)
            .transpose()
            .map_err(|_| anyhow!("uint32 value is out of range"))
    })
}

fn optional_f64(value: Option<&Value>) -> Result<Option<f64>> {
    match value {
        Some(Value::Number(value)) => value
            .as_f64()
            .map(Some)
            .ok_or_else(|| anyhow!("expected float")),
        Some(Value::Null) | None => Ok(None),
        Some(value) => Err(anyhow!("expected float, got {value}")),
    }
}

fn timestamp_millis(value: Option<&Value>) -> Result<Option<i64>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let Some(value) = value.as_str() else {
        return Ok(None);
    };
    let parsed = DateTime::parse_from_rfc3339(value)
        .with_context(|| format!("parse timestamp {value:?}"))?
        .with_timezone(&Utc);
    Ok(Some(parsed.timestamp_millis()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{contract_for, DatasetKind};
    use serde_json::json;

    #[test]
    fn validates_required_fields_before_arrow_conversion() {
        let contract = contract_for(DatasetKind::Usage);
        let err = record_batch_with_schema(
            &contract,
            &arrow_schema(&contract),
            &[json!({
                "at": "2026-05-17T00:00:00Z",
                "model": "qwen",
                "actor": "alice",
                "team": "platform",
                "input_tokens": 1,
                "output_tokens": 1,
                "latency_ms": 10,
                "status": "ok"
            })],
        )
        .expect_err("missing request_id should fail");

        assert!(err.to_string().contains("request_id"));
    }

    #[test]
    fn validates_timestamp_field_types_before_arrow_conversion() {
        let contract = contract_for(DatasetKind::Usage);
        let err = record_batch_with_schema(
            &contract,
            &arrow_schema(&contract),
            &[json!({
                "at": "not-a-date",
                "request_id": "req",
                "model": "qwen",
                "actor": "alice",
                "team": "platform",
                "input_tokens": 1,
                "output_tokens": 1,
                "latency_ms": 10,
                "status": "ok"
            })],
        )
        .expect_err("bad timestamp should fail");

        assert!(err.to_string().contains("at"));
    }
}
