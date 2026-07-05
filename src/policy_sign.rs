//! Security-sensitive policy signing and verification primitives.
//!
//! Extracted from the `llmctl` binary so the ed25519 sign/verify, HMAC bundle
//! signing, base64/SHA-256 helpers, and the hash-chained policy transparency
//! log verification live in the library crate where they are reusable and
//! unit-testable. Behaviour is identical to the previous in-binary
//! implementations: the same canonical-JSON encoding, algorithms, and JSON
//! document shapes are preserved.

use anyhow::{bail, Context, Result};
use base64::Engine;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use hmac::{Hmac, Mac};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::reporting;

type HmacSha256 = Hmac<Sha256>;

/// Standard base64 encoding of `bytes`.
#[must_use]
pub fn encode_b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Decode standard base64 `value`.
///
/// # Errors
///
/// Returns an error if `value` is not valid base64.
pub fn decode_b64(value: &str) -> Result<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(value)
        .context("decode base64")
}

/// Lowercase hex-encoded SHA-256 digest of `bytes`.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// HMAC-SHA256 over the canonical JSON encoding of `payload`, hex-encoded.
///
/// # Errors
///
/// Returns an error if `payload` cannot be canonicalised or the key is
/// rejected by the MAC.
pub fn hmac_signature_with_key(key: &[u8], payload: &Value) -> Result<String> {
    let canonical = reporting::canonical_json(payload)?;
    let mut mac = HmacSha256::new_from_slice(key)?;
    mac.update(canonical.as_bytes());
    Ok(hex::encode(mac.finalize().into_bytes()))
}

/// Ensures `doc.algorithm == "ed25519"`.
///
/// # Errors
///
/// Returns an error if the field is missing or names a different algorithm.
pub fn require_algorithm(doc: &Value) -> Result<()> {
    let algorithm = required_str(doc, "algorithm")?;
    if algorithm != "ed25519" {
        bail!("unsupported policy signing algorithm {algorithm}");
    }
    Ok(())
}

/// Reads a required string `field` from `doc`.
///
/// # Errors
///
/// Returns an error if the field is missing or not a string.
pub fn required_str<'a>(doc: &'a Value, field: &str) -> Result<&'a str> {
    doc.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("missing string field {field}"))
}

/// Parses an ed25519 [`SigningKey`] from a policy signing-key document.
///
/// # Errors
///
/// Returns an error if the document is not an `ed25519` key document or the
/// embedded private key is not 32 base64-encoded bytes.
pub fn signing_key_from_doc(doc: &Value) -> Result<SigningKey> {
    require_algorithm(doc)?;
    let bytes = decode_b64(required_str(doc, "private_key")?)?;
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("ed25519 private key must be 32 bytes"))?;
    Ok(SigningKey::from_bytes(&bytes))
}

/// Parses an ed25519 [`VerifyingKey`] from a policy public-key document.
///
/// # Errors
///
/// Returns an error if the document is not an `ed25519` key document or the
/// embedded public key is not 32 base64-encoded bytes.
pub fn verifying_key_from_doc(doc: &Value) -> Result<VerifyingKey> {
    require_algorithm(doc)?;
    let bytes = decode_b64(required_str(doc, "public_key")?)?;
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("ed25519 public key must be 32 bytes"))?;
    VerifyingKey::from_bytes(&bytes).context("parse ed25519 public key")
}

/// Signs `input` with `key`, returning the base64-encoded detached signature.
#[must_use]
pub fn sign_ed25519(key: &SigningKey, input: &[u8]) -> String {
    encode_b64(&key.sign(input).to_bytes())
}

/// Verifies a base64-encoded ed25519 `signature_b64` over `input` with `key`.
///
/// # Errors
///
/// Returns an error if `signature_b64` is not valid base64 or not a
/// well-formed ed25519 signature. A cryptographically invalid (but well-formed)
/// signature yields `Ok(false)`.
pub fn verify_ed25519(key: &VerifyingKey, input: &[u8], signature_b64: &str) -> Result<bool> {
    let signature_bytes = decode_b64(signature_b64)?;
    let signature = Signature::from_slice(&signature_bytes).context("parse signature")?;
    Ok(key.verify(input, &signature).is_ok())
}

/// Computes the canonical `entry_hash` for a policy transparency-log entry:
/// the SHA-256 (hex) of the canonical JSON of the entry with any existing
/// `entry_hash` field removed.
///
/// # Errors
///
/// Returns an error if the entry cannot be canonicalised.
pub fn policy_log_entry_hash(entry: &Value) -> Result<String> {
    let mut body = entry.clone();
    if let Some(object) = body.as_object_mut() {
        object.remove("entry_hash");
    }
    let canonical = reporting::canonical_json(&body)?;
    Ok(sha256_hex(canonical.as_bytes()))
}

/// Builds the JSON result value describing a policy-log verification outcome.
#[must_use]
pub fn policy_log_verification(
    valid: bool,
    entries: usize,
    failed_index: usize,
    reason: &str,
) -> Value {
    json!({
        "status": if valid { "valid" } else { "invalid" },
        "valid": valid,
        "entries": entries,
        "failed_index": failed_index,
        "reason": reason
    })
}

/// Verifies the hash chain of a policy transparency log.
///
/// Checks per-entry `entry_hash`, monotonic `index`, and `previous_hash`
/// linkage. Returns a JSON result value; a chain break yields a `valid: false`
/// document rather than an error.
///
/// # Errors
///
/// Returns an error only if an entry cannot be canonicalised while recomputing
/// its expected hash.
pub fn verify_policy_log_values(entries: &[Value]) -> Result<Value> {
    let mut previous_hash: Option<String> = None;
    for (index, entry) in entries.iter().enumerate() {
        let Some(actual_hash) = entry.get("entry_hash").and_then(Value::as_str) else {
            return Ok(policy_log_verification(
                false,
                entries.len(),
                index,
                "missing entry_hash",
            ));
        };
        if entry.get("index").and_then(Value::as_u64) != Some(index as u64) {
            return Ok(policy_log_verification(
                false,
                entries.len(),
                index,
                "index mismatch",
            ));
        }
        if entry.get("previous_hash").and_then(Value::as_str) != previous_hash.as_deref() {
            return Ok(policy_log_verification(
                false,
                entries.len(),
                index,
                "previous_hash mismatch",
            ));
        }
        let expected_hash = policy_log_entry_hash(entry)?;
        if actual_hash != expected_hash {
            return Ok(policy_log_verification(
                false,
                entries.len(),
                index,
                "entry_hash mismatch",
            ));
        }
        previous_hash = Some(actual_hash.to_string());
    }
    Ok(json!({
        "status": "valid",
        "valid": true,
        "entries": entries.len(),
        "head": previous_hash
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn test_key() -> SigningKey {
        // Deterministic key material keeps the round-trip test reproducible.
        SigningKey::from_bytes(&[7u8; 32])
    }

    #[test]
    fn ed25519_sign_verify_round_trip() {
        let signing_key = test_key();
        let verifying_key = signing_key.verifying_key();
        let payload = b"policy-bundle-contents";

        let signature_b64 = sign_ed25519(&signing_key, payload);
        assert!(
            verify_ed25519(&verifying_key, payload, &signature_b64).unwrap(),
            "valid signature over the signed payload must verify"
        );
        assert!(
            !verify_ed25519(&verifying_key, b"tampered", &signature_b64).unwrap(),
            "signature must not verify against a different payload"
        );
    }

    #[test]
    fn ed25519_key_docs_round_trip() {
        let signing_key = test_key();
        let verifying_key = signing_key.verifying_key();
        let private_doc = json!({
            "algorithm": "ed25519",
            "private_key": encode_b64(&signing_key.to_bytes()),
            "public_key": encode_b64(&verifying_key.to_bytes()),
        });
        let public_doc = json!({
            "algorithm": "ed25519",
            "public_key": encode_b64(&verifying_key.to_bytes()),
        });

        let parsed_signing = signing_key_from_doc(&private_doc).unwrap();
        let parsed_verifying = verifying_key_from_doc(&public_doc).unwrap();

        let input = b"artifact-bytes";
        let signature_b64 = sign_ed25519(&parsed_signing, input);
        assert!(verify_ed25519(&parsed_verifying, input, &signature_b64).unwrap());

        let wrong_algorithm = json!({ "algorithm": "hmac-sha256", "public_key": "" });
        assert!(verifying_key_from_doc(&wrong_algorithm).is_err());
    }

    #[test]
    fn hmac_signature_is_stable_and_key_sensitive() {
        let payload = json!({ "kind": "policy-bundle", "name": "demo" });
        let a = hmac_signature_with_key(b"secret-key", &payload).unwrap();
        let b = hmac_signature_with_key(b"secret-key", &payload).unwrap();
        let c = hmac_signature_with_key(b"other-key", &payload).unwrap();
        assert_eq!(a, b, "HMAC over identical key+payload must be stable");
        assert_ne!(a, c, "HMAC must change with the key");
    }

    #[test]
    fn policy_log_hash_chain_verifies_and_detects_tampering() {
        let mut first = json!({
            "schema_version": 1,
            "kind": "policy-transparency-log-entry",
            "index": 0,
            "artifact_sha256": sha256_hex(b"artifact-0"),
            "previous_hash": Value::Null,
        });
        first["entry_hash"] = json!(policy_log_entry_hash(&first).unwrap());

        let mut second = json!({
            "schema_version": 1,
            "kind": "policy-transparency-log-entry",
            "index": 1,
            "artifact_sha256": sha256_hex(b"artifact-1"),
            "previous_hash": first["entry_hash"].clone(),
        });
        second["entry_hash"] = json!(policy_log_entry_hash(&second).unwrap());

        let chain = vec![first, second.clone()];
        let ok = verify_policy_log_values(&chain).unwrap();
        assert_eq!(ok["valid"], json!(true));

        let mut tampered = second;
        tampered["artifact_sha256"] = json!(sha256_hex(b"tampered"));
        let broken = vec![chain[0].clone(), tampered];
        let result = verify_policy_log_values(&broken).unwrap();
        assert_eq!(result["valid"], json!(false));
    }
}
