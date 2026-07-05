//! Signed report envelopes: canonical JSON, SHA-256, and verification.
use super::*;

const REPORT_SCHEMA_VERSION: u32 = 1;

pub fn report_envelope<T>(report_kind: ReportKind, payload: T) -> Result<ReportEnvelope<T>>
where
    T: Serialize,
{
    report_envelope_at(report_kind, payload, Utc::now())
}

pub fn canonical_sha256<T>(payload: &T) -> Result<String>
where
    T: Serialize,
{
    let canonical = canonical_json(payload)?;
    Ok(hex::encode(Sha256::digest(canonical.as_bytes())))
}

pub fn canonical_json<T>(payload: &T) -> Result<String>
where
    T: Serialize,
{
    let value = serde_json::to_value(payload)?;
    Ok(serde_json::to_string(&canonicalize_value(value))?)
}

pub fn verify_envelope_value(envelope: &Value) -> Result<EnvelopeVerification> {
    let metadata = match envelope.get("metadata").and_then(Value::as_object) {
        Some(metadata) => metadata,
        None => {
            return Ok(invalid_envelope(
                None,
                None,
                None,
                "missing metadata object",
            ))
        }
    };
    let payload = match envelope.get("payload") {
        Some(payload) => payload,
        None => {
            return Ok(invalid_envelope(
                expected_sha256(metadata),
                report_kind(metadata),
                schema_version(metadata),
                "missing payload",
            ))
        }
    };
    let expected = match expected_sha256(metadata) {
        Some(expected) => expected,
        None => {
            return Ok(invalid_envelope(
                None,
                report_kind(metadata),
                schema_version(metadata),
                "missing metadata sha256",
            ))
        }
    };
    if schema_version(metadata) != Some(u64::from(REPORT_SCHEMA_VERSION)) {
        return Ok(invalid_envelope(
            Some(expected),
            report_kind(metadata),
            schema_version(metadata),
            "unsupported report envelope schema_version",
        ));
    }
    if report_kind(metadata).is_none() {
        return Ok(invalid_envelope(
            Some(expected),
            None,
            schema_version(metadata),
            "missing or unknown report_kind",
        ));
    }
    let actual = canonical_sha256(payload)?;
    let valid = expected.eq_ignore_ascii_case(&actual);

    Ok(EnvelopeVerification {
        status: if valid { "valid" } else { "invalid" }.to_string(),
        valid,
        expected_sha256: Some(expected),
        actual_sha256: Some(actual),
        report_kind: report_kind(metadata),
        schema_version: schema_version(metadata),
        reason: if valid {
            None
        } else {
            Some("payload sha256 mismatch".to_string())
        },
    })
}

fn invalid_envelope(
    expected_sha256: Option<String>,
    report_kind: Option<String>,
    schema_version: Option<u64>,
    reason: &str,
) -> EnvelopeVerification {
    EnvelopeVerification {
        status: "invalid".to_string(),
        valid: false,
        expected_sha256,
        actual_sha256: None,
        report_kind,
        schema_version,
        reason: Some(reason.to_string()),
    }
}

pub(crate) fn expected_sha256(metadata: &Map<String, Value>) -> Option<String> {
    metadata
        .get("sha256")
        .and_then(Value::as_str)
        .map(str::to_string)
}

pub(crate) fn report_kind(metadata: &Map<String, Value>) -> Option<String> {
    metadata
        .get("report_kind")
        .and_then(Value::as_str)
        .map(str::to_string)
}

pub(crate) fn schema_version(metadata: &Map<String, Value>) -> Option<u64> {
    metadata.get("schema_version").and_then(Value::as_u64)
}

pub(crate) fn report_envelope_at<T>(
    report_kind: ReportKind,
    payload: T,
    generated_at: DateTime<Utc>,
) -> Result<ReportEnvelope<T>>
where
    T: Serialize,
{
    Ok(ReportEnvelope {
        metadata: ReportMetadata {
            report_kind,
            generated_at,
            schema_version: REPORT_SCHEMA_VERSION,
            producer: crate::SERVICE_NAME.to_string(),
            contract_schema_version: contracts::CONTRACT_SCHEMA_VERSION,
            contract_hashes: contract_hashes()?,
            sha256: canonical_sha256(&payload)?,
        },
        payload,
    })
}

fn contract_hashes() -> Result<BTreeMap<String, String>> {
    contracts::all_contracts()
        .into_iter()
        .map(|contract| {
            let hash = canonical_sha256(&contract)?;
            Ok((contract.dataset.to_string(), hash))
        })
        .collect()
}

fn canonicalize_value(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize_value).collect()),
        Value::Object(values) => {
            let mut sorted = Map::new();
            let mut entries: Vec<_> = values.into_iter().collect();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            for (key, value) in entries {
                sorted.insert(key, canonicalize_value(value));
            }
            Value::Object(sorted)
        }
        scalar => scalar,
    }
}
