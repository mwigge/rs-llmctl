use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const CONTRACT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DatasetKind {
    Security,
    Observability,
    Usage,
    User,
    Finops,
    Models,
    Drift,
    Audit,
}

impl DatasetKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Security => "security",
            Self::Observability => "observability",
            Self::Usage => "usage",
            Self::User => "user",
            Self::Finops => "finops",
            Self::Models => "models",
            Self::Drift => "drift",
            Self::Audit => "audit",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldContract {
    pub name: &'static str,
    pub data_type: &'static str,
    pub nullable: bool,
    pub description: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataContract {
    pub dataset: &'static str,
    pub schema_version: u32,
    pub event_format: &'static str,
    pub primary_time_field: &'static str,
    pub fields: Vec<FieldContract>,
    pub json_schema: Value,
    pub arrow_schema: Value,
}

pub fn all_contracts() -> Vec<DataContract> {
    [
        DatasetKind::Security,
        DatasetKind::Observability,
        DatasetKind::Usage,
        DatasetKind::User,
        DatasetKind::Finops,
        DatasetKind::Models,
        DatasetKind::Drift,
        DatasetKind::Audit,
    ]
    .into_iter()
    .map(contract_for)
    .collect()
}

pub fn contract_for(dataset: DatasetKind) -> DataContract {
    let fields = fields_for(dataset);
    let dataset_name = dataset.as_str();
    DataContract {
        dataset: dataset_name,
        schema_version: CONTRACT_SCHEMA_VERSION,
        event_format: "rs-llmctl.data.v1",
        primary_time_field: "at",
        json_schema: json_schema(dataset_name, &fields),
        arrow_schema: arrow_schema(dataset_name, &fields),
        fields,
    }
}

fn fields_for(dataset: DatasetKind) -> Vec<FieldContract> {
    match dataset {
        DatasetKind::Security => vec![
            field("at", "timestamp[ms, tz=UTC]", false, "event timestamp"),
            field("kind", "utf8", false, "security event kind"),
            field("actor", "utf8", true, "authenticated subject or operator"),
            field("team", "utf8", true, "owning team"),
            field("resource", "utf8", true, "resource affected by the event"),
            field("outcome", "utf8", true, "security decision or result"),
            field("request_id", "utf8", true, "request correlation id"),
        ],
        DatasetKind::Observability => vec![
            field("at", "timestamp[ms, tz=UTC]", false, "event timestamp"),
            field("kind", "utf8", false, "observation kind"),
            field("source", "utf8", false, "producer"),
            field("model", "utf8", true, "model alias"),
            field("value", "float64", false, "measured value"),
            field("unit", "utf8", false, "measurement unit"),
            field("request_id", "utf8", true, "request correlation id"),
        ],
        DatasetKind::Usage => vec![
            field("at", "timestamp[ms, tz=UTC]", false, "event timestamp"),
            field("request_id", "utf8", false, "request correlation id"),
            field("model", "utf8", false, "model alias"),
            field("actor", "utf8", false, "authenticated subject"),
            field("team", "utf8", false, "owning team"),
            field("input_tokens", "uint64", false, "prompt/input token count"),
            field(
                "output_tokens",
                "uint64",
                false,
                "completion/output token count",
            ),
            field("latency_ms", "uint64", false, "model request latency"),
            field("status", "utf8", false, "request status"),
        ],
        DatasetKind::User => vec![
            field("actor", "utf8", false, "authenticated subject"),
            field("team", "utf8", false, "owning team"),
            field("request_count", "uint64", false, "requests in window"),
            field("input_tokens", "uint64", false, "input tokens in window"),
            field("output_tokens", "uint64", false, "output tokens in window"),
            field("total_tokens", "uint64", false, "total tokens in window"),
        ],
        DatasetKind::Finops => vec![
            field("team", "utf8", true, "owning team"),
            field("actor", "utf8", true, "authenticated subject"),
            field("model", "utf8", true, "model alias"),
            field("request_count", "uint64", false, "requests in window"),
            field("total_tokens", "uint64", false, "billable token volume"),
            field("total_latency_ms", "uint64", false, "aggregate latency"),
        ],
        DatasetKind::Models => vec![
            field("alias", "utf8", false, "model alias"),
            field("role", "utf8", false, "serving role"),
            field("path", "utf8", false, "configured model path"),
            field("weight", "uint32", false, "routing weight"),
            field(
                "updated_at",
                "timestamp[ms, tz=UTC]",
                true,
                "inventory update timestamp",
            ),
        ],
        DatasetKind::Drift => vec![
            field("at", "timestamp[ms, tz=UTC]", false, "event timestamp"),
            field("kind", "utf8", false, "drift signal kind"),
            field("model", "utf8", true, "model alias"),
            field("value", "float64", false, "drift score"),
            field("unit", "utf8", false, "measurement unit"),
            field("request_id", "utf8", true, "request correlation id"),
        ],
        DatasetKind::Audit => vec![
            field("at", "timestamp[ms, tz=UTC]", false, "event timestamp"),
            field("action", "utf8", false, "audited action"),
            field("actor", "utf8", false, "authenticated subject or operator"),
            field("team", "utf8", false, "owning team"),
            field("resource", "utf8", false, "audited resource"),
            field("outcome", "utf8", false, "action outcome"),
            field("request_id", "utf8", true, "request correlation id"),
        ],
    }
}

fn field(
    name: &'static str,
    data_type: &'static str,
    nullable: bool,
    description: &'static str,
) -> FieldContract {
    FieldContract {
        name,
        data_type,
        nullable,
        description,
    }
}

fn json_schema(dataset: &str, fields: &[FieldContract]) -> Value {
    let properties = fields
        .iter()
        .map(|field| {
            (
                field.name.to_string(),
                json!({
                    "type": json_type(field.data_type, field.nullable),
                    "description": field.description
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let required = fields
        .iter()
        .filter(|field| !field.nullable)
        .map(|field| Value::String(field.name.to_string()))
        .collect::<Vec<_>>();

    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": format!("https://schemas.rs-llmctl.local/{dataset}/v1.json"),
        "title": format!("rs-llmctl {dataset} data contract"),
        "type": "object",
        "additionalProperties": true,
        "required": required,
        "properties": properties
    })
}

fn json_type(data_type: &str, nullable: bool) -> Value {
    let base = if data_type.starts_with("uint") {
        "integer"
    } else if data_type == "float64" {
        "number"
    } else {
        "string"
    };
    if nullable {
        json!([base, "null"])
    } else {
        json!(base)
    }
}

fn arrow_schema(dataset: &str, fields: &[FieldContract]) -> Value {
    json!({
        "format": "arrow-json-schema",
        "name": format!("rs_llmctl_{dataset}_v1"),
        "fields": fields.iter().map(|field| {
            json!({
                "name": field.name,
                "data_type": field.data_type,
                "nullable": field.nullable,
                "metadata": {
                    "description": field.description
                }
            })
        }).collect::<Vec<_>>()
    })
}
