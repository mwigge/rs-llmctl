use crate::contracts::DataContract;
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

pub fn write_arrow_ipc(path: &Path, contract: &DataContract, rows: &[Value]) -> Result<u64> {
    let batch = record_batch(contract, rows)?;
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = FileWriter::try_new(file, batch.schema().as_ref())?;
    writer.write(&batch)?;
    writer.finish()?;
    Ok(rows.len() as u64)
}

pub fn write_parquet(path: &Path, contract: &DataContract, rows: &[Value]) -> Result<u64> {
    let batch = record_batch(contract, rows)?;
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(rows.len() as u64)
}

fn record_batch(contract: &DataContract, rows: &[Value]) -> Result<RecordBatch> {
    let fields = contract
        .fields
        .iter()
        .map(|field| Field::new(field.name, arrow_type(field.data_type), field.nullable))
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    let columns = contract
        .fields
        .iter()
        .map(|field| array_for_field(field.name, field.data_type, rows))
        .collect::<Result<Vec<_>>>()?;
    Ok(RecordBatch::try_new(schema, columns)?)
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

fn array_for_field(name: &str, data_type: &str, rows: &[Value]) -> Result<ArrayRef> {
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
