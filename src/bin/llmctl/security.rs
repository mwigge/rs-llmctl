use crate::*;

pub(crate) async fn security_command(
    path: &Path,
    command: SecurityCommand,
    as_json: bool,
) -> Result<()> {
    match command {
        SecurityCommand::Check => {
            let cfg = load_config(path).await?;
            config::validate_production_security(&cfg)?;
            emit(
                as_json,
                &json!({
                    "status": "ok",
                    "production": cfg.security.production,
                    "require_auth": cfg.security.require_auth,
                    "bind_external": cfg.security.bind_external,
                    "host": cfg.server.host,
                    "tls_termination": cfg.security.tls_termination,
                    "api_keys": cfg.security.api_keys.len()
                }),
            )
        }
        SecurityCommand::GenerateKey(args) => {
            let secret = generate_api_key_secret(&args.prefix);
            let sha256 = hex::encode(Sha256::digest(secret.as_bytes()));
            let last_four = secret
                .chars()
                .rev()
                .take(4)
                .collect::<String>()
                .chars()
                .rev()
                .collect::<String>();
            if let Some(output) = args.output.as_ref() {
                write_secret_file(output, &secret).await?;
            }
            emit(
                as_json,
                &json!({
                    "status": "generated",
                    "secret": if args.output.is_none() { Some(secret.as_str()) } else { None },
                    "secret_written": args.output.as_ref().map(|path| path.display().to_string()),
                    "sha256": sha256,
                    "last_four": last_four,
                    "metadata": {
                        "purpose": "api-key",
                        "algorithm": "sha256",
                        "encoding": "hex",
                        "store_secret_once": true,
                        "next": "llmctl security add-key --id <id> --sha256 <sha256> --subject <subject> --team <team> --scope chat"
                    }
                }),
            )
        }
        SecurityCommand::HashKey(args) => {
            let (secret, input) = read_api_key_secret(args).await?;
            rs_llmctl::security::validate_api_secret_material(&secret)?;
            let sha256 = hex::encode(Sha256::digest(secret.as_bytes()));
            emit(
                as_json,
                &json!({
                    "sha256": sha256,
                    "metadata": {
                        "algorithm": "sha256",
                        "encoding": "hex",
                        "input": input,
                        "purpose": "api-key"
                    }
                }),
            )
        }
        SecurityCommand::ListKeys => {
            let cfg = load_config(path).await?;
            emit(as_json, &api_key_inventory_report(&cfg))
        }
        SecurityCommand::RotateKey(args) => {
            let mut cfg = load_config(path).await?;
            let sha256 = args.sha256.to_ascii_lowercase();
            validate_sha256_digest(&sha256)?;
            let Some(position) = cfg
                .security
                .api_keys
                .iter()
                .position(|key| key.id == args.id)
            else {
                bail!("api key id `{}` was not found", args.id);
            };
            if let Some(new_id) = args.new_id.as_ref() {
                validate_api_key_id(new_id)?;
                if cfg.security.api_keys.iter().any(|key| key.id == *new_id) {
                    bail!("api key id `{new_id}` already exists");
                }
                let now = Utc::now();
                let mut retiring = cfg.security.api_keys[position].clone();
                retiring.status = "retiring".to_string();
                retiring.rotated_at = Some(now);
                cfg.security.api_keys[position] = retiring.clone();
                let replacement = ApiKeyConfig {
                    id: new_id.clone(),
                    sha256,
                    subject: retiring.subject,
                    team: retiring.team,
                    scopes: retiring.scopes,
                    created_at: Some(now),
                    expires_at: args.expires_at,
                    rotated_at: None,
                    owner: retiring.owner,
                    purpose: retiring.purpose,
                    last_four: args.last_four,
                    fingerprint: None,
                    status: "active".to_string(),
                };
                cfg.security.api_keys.push(replacement);
                config::save(path, &cfg).await?;
                record_security_key_event(
                    &cfg,
                    "security.api_key.rotate",
                    new_id,
                    "rotated",
                    json!({
                        "api_key_id": args.id,
                        "new_api_key_id": new_id,
                        "mode": "overlap",
                        "reason": args.reason,
                        "old_status": "retiring"
                    }),
                )
                .await?;
                emit(
                    as_json,
                    &json!({
                        "status": "rotated",
                        "mode": "overlap",
                        "retiring_id": args.id,
                        "active_id": new_id,
                        "restart_required": true,
                        "restart_hint": default_restart_hint()
                    }),
                )?;
                return Ok(());
            }
            if !args.replace {
                bail!(
                    "rotate-key requires --new-id for overlap rotation or --replace for in-place replacement"
                );
            }
            let key = &mut cfg.security.api_keys[position];
            key.sha256 = sha256;
            key.rotated_at = Some(Utc::now());
            key.expires_at = args.expires_at.or(key.expires_at);
            key.last_four = args.last_four.or_else(|| key.last_four.clone());
            key.status = "active".to_string();
            config::save(path, &cfg).await?;
            record_security_key_event(
                &cfg,
                "security.api_key.rotate",
                &args.id,
                "rotated",
                json!({
                    "api_key_id": args.id,
                    "mode": "replace",
                    "reason": args.reason
                }),
            )
            .await?;
            emit(
                as_json,
                &json!({
                    "status": "rotated",
                    "mode": "replace",
                    "id": args.id,
                    "sha256_present": true,
                    "restart_required": true,
                    "restart_hint": default_restart_hint()
                }),
            )
        }
        SecurityCommand::RevokeKey(args) => {
            let mut cfg = load_config(path).await?;
            let Some(position) = cfg
                .security
                .api_keys
                .iter()
                .position(|key| key.id == args.id)
            else {
                bail!("api key id `{}` was not found", args.id);
            };
            let removed = cfg.security.api_keys.remove(position);
            config::save(path, &cfg).await?;
            record_security_key_event(
                &cfg,
                "security.api_key.revoke",
                &args.id,
                "revoked",
                json!({
                    "api_key_id": args.id,
                    "reason": args.reason,
                    "removed": true,
                    "remove_requested": args.remove,
                    "subject": removed.subject,
                    "team": removed.team,
                    "owner": removed.owner,
                    "purpose": removed.purpose,
                    "previous_status": removed.status
                }),
            )
            .await?;
            emit(
                as_json,
                &json!({
                    "status": "revoked",
                    "id": args.id,
                    "api_keys": cfg.security.api_keys.len(),
                    "restart_required": true,
                    "restart_hint": default_restart_hint()
                }),
            )
        }
        SecurityCommand::KeyUsage(args) => {
            let cfg = load_config(path).await?;
            let storage = init_storage(&cfg.storage).await?;
            let report = api_key_usage_report(&storage, args.id.as_deref(), args.hours).await?;
            emit(as_json, &report)
        }
        SecurityCommand::AddKey(args) => {
            let mut cfg = load_config(path).await?;
            let sha256 = args.sha256.to_ascii_lowercase();
            validate_add_key_args(&args.id, &sha256, &args.subject, &args.team)?;
            let key = ApiKeyConfig {
                id: args.id,
                sha256,
                subject: args.subject,
                team: args.team,
                scopes: args.scopes,
                created_at: Some(Utc::now()),
                expires_at: args.expires_at,
                rotated_at: None,
                owner: args.owner,
                purpose: args.purpose,
                last_four: args.last_four,
                fingerprint: None,
                status: "active".to_string(),
            };
            let action = upsert_api_key(&mut cfg.security.api_keys, key.clone());
            config::save(path, &cfg).await?;
            emit(
                as_json,
                &json!({
                    "status": "saved",
                    "action": action,
                    "api_keys": cfg.security.api_keys.len(),
                    "key": {
                        "id": key.id,
                        "subject": key.subject,
                        "team": key.team,
                        "scopes": key.scopes,
                        "owner": key.owner,
                        "purpose": key.purpose,
                        "created_at": key.created_at,
                        "expires_at": key.expires_at,
                        "last_four": key.last_four,
                        "status": key.status,
                        "sha256_present": true
                    }
                }),
            )
        }
        SecurityCommand::AuditConfig(args) => {
            let cfg = load_config(path).await?;
            let report = audit_config_report(path, &cfg, args.systemd_unit.as_deref()).await?;
            emit(as_json, &report)
        }
    }
}

fn api_key_inventory_report(cfg: &Config) -> Value {
    json!({
        "status": "ok",
        "require_auth": cfg.security.require_auth,
        "api_keys": cfg.security.api_keys.iter().map(|key| {
            json!({
                "id": key.id,
                "subject": key.subject,
                "team": key.team,
                "scopes": key.scopes,
                "owner": key.owner,
                "purpose": key.purpose,
                "created_at": key.created_at,
                "expires_at": key.expires_at,
                "rotated_at": key.rotated_at,
                "last_four": key.last_four,
                "fingerprint": key.fingerprint,
                "status": key.status,
                "sha256_present": !key.sha256.trim().is_empty()
            })
        }).collect::<Vec<_>>()
    })
}

async fn api_key_usage_report(storage: &Storage, id: Option<&str>, hours: i64) -> Result<Value> {
    let now = Utc::now();
    let from = now - Duration::hours(hours.max(1));
    let key_usage = storage.api_key_usage_between(from, now).await?;
    let mut by_key: BTreeMap<String, ApiKeyUsageSummary> = BTreeMap::new();
    for record in key_usage {
        if id.is_some_and(|expected| expected != record.api_key_id) {
            continue;
        }
        let summary = by_key.entry(record.api_key_id.clone()).or_default();
        summary.request_count = summary.request_count.saturating_add(1);
        if record.audit_outcome != "ok" && record.audit_outcome != "allowed" {
            summary.error_count = summary.error_count.saturating_add(1);
        }
        summary.last_seen = Some(
            summary
                .last_seen
                .map_or(record.usage_at, |last| last.max(record.usage_at)),
        );
        summary.input_tokens = summary.input_tokens.saturating_add(record.input_tokens);
        summary.output_tokens = summary.output_tokens.saturating_add(record.output_tokens);
        summary.total_tokens = summary.total_tokens.saturating_add(record.total_tokens);
        summary.latency_ms = summary.latency_ms.saturating_add(record.latency_ms);
        summary.actors.insert(record.actor);
        summary.teams.insert(record.team);
        summary.models.insert(record.model);
        summary.statuses.insert(record.status);
    }

    let audit_events = storage.audit_events_between(from, now).await?;
    for event in audit_events {
        let Some(key_id) = event.detail_json.get("api_key_id").and_then(Value::as_str) else {
            continue;
        };
        if id.is_some_and(|expected| expected != key_id) {
            continue;
        }
        let summary = by_key.entry(key_id.to_string()).or_default();
        summary.audit_event_count = summary.audit_event_count.saturating_add(1);
        summary.actions.insert(event.action);
        summary.resources.insert(event.resource);
        summary.actors.insert(event.actor);
        summary.teams.insert(event.team);
        summary.last_seen = Some(
            summary
                .last_seen
                .map_or(event.at, |last| last.max(event.at)),
        );
    }

    Ok(json!({
        "status": "ok",
        "from": from,
        "to": now,
        "filter": { "id": id },
        "keys": by_key.into_iter().map(|(key_id, summary)| {
            json!({
                "id": key_id,
                "request_count": summary.request_count,
                "audit_event_count": summary.audit_event_count,
                "error_count": summary.error_count,
                "input_tokens": summary.input_tokens,
                "output_tokens": summary.output_tokens,
                "total_tokens": summary.total_tokens,
                "latency_ms": summary.latency_ms,
                "last_seen": summary.last_seen,
                "actors": summary.actors.into_iter().collect::<Vec<_>>(),
                "teams": summary.teams.into_iter().collect::<Vec<_>>(),
                "models": summary.models.into_iter().collect::<Vec<_>>(),
                "statuses": summary.statuses.into_iter().collect::<Vec<_>>(),
                "actions": summary.actions.into_iter().collect::<Vec<_>>(),
                "resources": summary.resources.into_iter().collect::<Vec<_>>()
            })
        }).collect::<Vec<_>>()
    }))
}

async fn read_api_key_secret(args: SecurityHashKeyArgs) -> Result<(String, &'static str)> {
    if let Some(name) = args.env {
        let secret =
            std::env::var(&name).with_context(|| format!("read secret from env {name}"))?;
        return Ok((secret, "env"));
    }

    if args.stdin {
        let mut secret = String::new();
        std::io::stdin()
            .read_to_string(&mut secret)
            .context("read secret from stdin")?;
        return Ok((secret.trim_end_matches(['\r', '\n']).to_string(), "stdin"));
    }

    bail!(
        "security hash-key requires --stdin or --env NAME so secrets are not exposed in process arguments"
    )
}

fn upsert_api_key(keys: &mut Vec<ApiKeyConfig>, key: ApiKeyConfig) -> &'static str {
    if let Some(existing) = keys.iter_mut().find(|existing| existing.id == key.id) {
        *existing = key;
        "updated"
    } else {
        keys.push(key);
        "inserted"
    }
}

fn validate_add_key_args(id: &str, sha256: &str, subject: &str, team: &str) -> Result<()> {
    validate_api_key_id(id)?;
    validate_sha256_digest(sha256)?;
    if subject.trim().is_empty() {
        bail!("subject must not be empty");
    }
    if team.trim().is_empty() {
        bail!("team must not be empty");
    }
    Ok(())
}
