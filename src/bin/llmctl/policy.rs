use crate::*;

pub(crate) async fn policy_command(command: PolicyCommand, as_json: bool) -> Result<()> {
    match command {
        PolicyCommand::Bundle(args) => {
            let policy = fs::read_to_string(&args.input)
                .await
                .with_context(|| format!("read policy {}", args.input.display()))?;
            let policy_value: serde_json::Value =
                if args.input.extension().and_then(|ext| ext.to_str()) == Some("toml") {
                    let value: toml::Value = toml::from_str(&policy)
                        .with_context(|| format!("parse TOML {}", args.input.display()))?;
                    serde_json::to_value(value)?
                } else {
                    serde_json::from_str(&policy)
                        .with_context(|| format!("parse JSON {}", args.input.display()))?
                };
            let payload = json!({
                "schema_version": 1,
                "kind": "policy-bundle",
                "name": args.name,
                "created_at": Utc::now(),
                "policy": policy_value
            });
            let signature = hmac_signature(&args.signing_key_env, &payload)?;
            let bundle = json!({
                "metadata": {
                    "algorithm": "hmac-sha256",
                    "key_source": format!("env:{}", args.signing_key_env),
                    "signature": signature
                },
                "payload": payload
            });
            fs::write(&args.output, serde_json::to_vec_pretty(&bundle)?)
                .await
                .with_context(|| format!("write {}", args.output.display()))?;
            emit(
                as_json,
                &json!({
                    "status": "created",
                    "path": args.output,
                    "algorithm": "hmac-sha256",
                    "signature": signature
                }),
            )
        }
        PolicyCommand::VerifyBundle(args) => {
            let bytes = fs::read(&args.path)
                .await
                .with_context(|| format!("read {}", args.path.display()))?;
            let bundle: serde_json::Value = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse {}", args.path.display()))?;
            let payload = bundle
                .get("payload")
                .ok_or_else(|| anyhow::anyhow!("policy bundle missing payload"))?;
            let expected = bundle
                .pointer("/metadata/signature")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("policy bundle missing signature"))?;
            let actual = hmac_signature(&args.signing_key_env, payload)?;
            emit(
                as_json,
                &json!({
                    "status": if expected.eq_ignore_ascii_case(&actual) { "valid" } else { "invalid" },
                    "valid": expected.eq_ignore_ascii_case(&actual),
                    "path": args.path,
                    "algorithm": "hmac-sha256"
                }),
            )
        }
        PolicyCommand::Keygen(args) => {
            let mut rng = rand_core::OsRng;
            let signing_key = SigningKey::generate(&mut rng);
            let verifying_key = signing_key.verifying_key();
            let public_key = encode_b64(&verifying_key.to_bytes());
            let private_key = encode_b64(&signing_key.to_bytes());
            let private_doc = json!({
                "schema_version": 1,
                "kind": "policy-signing-private-key",
                "algorithm": "ed25519",
                "private_key": private_key,
                "public_key": public_key
            });
            let public_doc = json!({
                "schema_version": 1,
                "kind": "policy-signing-public-key",
                "algorithm": "ed25519",
                "public_key": public_key
            });
            write_json_file(&args.private_key, &private_doc).await?;
            restrict_private_key_file(&args.private_key).await?;
            write_json_file(&args.public_key, &public_doc).await?;
            emit(
                as_json,
                &json!({
                    "status": "created",
                    "algorithm": "ed25519",
                    "private_key": args.private_key,
                    "public_key": args.public_key
                }),
            )
        }
        PolicyCommand::Sign(args) => {
            let input = fs::read(&args.input)
                .await
                .with_context(|| format!("read {}", args.input.display()))?;
            let signing_key = read_policy_signing_key(&args.private_key).await?;
            let signature = policy_sign::sign_ed25519(&signing_key, &input);
            let payload_sha256 = sha256_hex(&input);
            let signature_doc = json!({
                "schema_version": 1,
                "kind": "policy-signature",
                "algorithm": "ed25519",
                "signed_at": Utc::now(),
                "payload_sha256": payload_sha256,
                "public_key": encode_b64(&signing_key.verifying_key().to_bytes()),
                "signature": signature
            });
            write_json_file(&args.signature, &signature_doc).await?;
            emit(
                as_json,
                &json!({
                    "status": "signed",
                    "algorithm": "ed25519",
                    "input": args.input,
                    "signature": args.signature,
                    "payload_sha256": payload_sha256
                }),
            )
        }
        PolicyCommand::Verify(args) => {
            let input = fs::read(&args.input)
                .await
                .with_context(|| format!("read {}", args.input.display()))?;
            let verifying_key = read_policy_verifying_key(&args.public_key).await?;
            let signature_doc = read_json_file(&args.signature).await?;
            require_algorithm(&signature_doc)?;
            let expected_hash = required_str(&signature_doc, "payload_sha256")?;
            let actual_hash = sha256_hex(&input);
            let signature_valid = policy_sign::verify_ed25519(
                &verifying_key,
                &input,
                required_str(&signature_doc, "signature")?,
            )?;
            let hash_valid = expected_hash.eq_ignore_ascii_case(&actual_hash);
            emit(
                as_json,
                &json!({
                    "status": if signature_valid && hash_valid { "valid" } else { "invalid" },
                    "valid": signature_valid && hash_valid,
                    "algorithm": "ed25519",
                    "input": args.input,
                    "signature": args.signature,
                    "payload_sha256": actual_hash
                }),
            )
        }
        PolicyCommand::Log { command } => match command {
            PolicyLogCommand::Append(args) => {
                let entry = append_policy_log_entry(&args).await?;
                emit(as_json, &entry)
            }
            PolicyLogCommand::Verify(args) => {
                let result = verify_policy_log(&args.log_path).await?;
                emit(as_json, &result)
            }
        },
        PolicyCommand::LegalHoldPlan(args) => emit(
            as_json,
            &json!({
                "schema_version": 1,
                "kind": "legal-hold-plan",
                "generated_at": Utc::now(),
                "dataset": DatasetKind::from(args.dataset).as_str(),
                "case_id": args.case_id,
                "reason": args.reason,
                "retention": {
                    "override": "hold_until_released",
                    "applies_to_dataset": true
                },
                "operator_steps": [
                    "attach this plan to the case record",
                    "exclude dataset scope from automated retention pruning",
                    "generate monthly audit and data export envelopes while hold is active",
                    "record signed release of hold before retention resumes"
                ]
            }),
        ),
    }
}

async fn append_policy_log_entry(args: &PolicyLogAppendArgs) -> Result<serde_json::Value> {
    let current = read_jsonl(&args.log_path).await?;
    let verification = verify_policy_log_values(&current)?;
    if !verification
        .get("valid")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        bail!("policy transparency log hash chain is invalid");
    }

    let artifact = fs::read(&args.artifact)
        .await
        .with_context(|| format!("read {}", args.artifact.display()))?;
    let signature_sha256 = if let Some(path) = &args.signature {
        let bytes = fs::read(path)
            .await
            .with_context(|| format!("read {}", path.display()))?;
        Some(sha256_hex(&bytes))
    } else {
        None
    };
    let previous_hash = current
        .last()
        .and_then(|entry| entry.get("entry_hash"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let body = json!({
        "schema_version": 1,
        "kind": "policy-transparency-log-entry",
        "index": current.len(),
        "logged_at": Utc::now(),
        "artifact_sha256": sha256_hex(&artifact),
        "signature_sha256": signature_sha256,
        "previous_hash": previous_hash
    });
    let entry_hash = policy_log_entry_hash(&body)?;
    let mut entry = body;
    entry["entry_hash"] = json!(entry_hash);
    append_jsonl(&args.log_path, &entry).await?;
    Ok(entry)
}

async fn verify_policy_log(path: &Path) -> Result<serde_json::Value> {
    let entries = read_jsonl(path).await?;
    verify_policy_log_values(&entries)
}

async fn read_policy_signing_key(path: &Path) -> Result<SigningKey> {
    let doc = read_json_file(path).await?;
    policy_sign::signing_key_from_doc(&doc)
}

async fn read_policy_verifying_key(path: &Path) -> Result<VerifyingKey> {
    let doc = read_json_file(path).await?;
    policy_sign::verifying_key_from_doc(&doc)
}

fn hmac_signature(key_env: &str, payload: &serde_json::Value) -> Result<String> {
    let key = std::env::var(key_env).with_context(|| format!("read signing key env {key_env}"))?;
    policy_sign::hmac_signature_with_key(key.as_bytes(), payload)
}
