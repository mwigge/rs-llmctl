use crate::*;

pub(crate) async fn quota_command(path: &Path, command: QuotaCommand, as_json: bool) -> Result<()> {
    let mut cfg = load_config(path).await?;
    match command {
        QuotaCommand::Set(args) => {
            upsert_quota(
                &mut cfg.quotas,
                QuotaConfig {
                    subject: args.subject,
                    team: args.team,
                    requests_per_minute: args.requests_per_minute,
                    tokens_per_day: args.tokens_per_day,
                    max_concurrency: args.max_concurrency,
                    allowed_models: args.allowed_models,
                },
            );
            config::save(path, &cfg).await?;
            emit(as_json, &json!({ "status": "set", "quotas": cfg.quotas }))
        }
        QuotaCommand::Status(args) => {
            let storage = init_storage(&cfg.storage).await?;
            let principal = quota_status_principal(&cfg.quotas, &args);
            let policy = matching_quota(&cfg.quotas, &principal);
            let subject_scoped = policy.is_some_and(|policy| policy.subject == principal.subject);
            let decision =
                quota::check_quota(&storage, &cfg.quotas, &principal, &args.model).await?;
            let now = Utc::now();
            let day_start = now.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc();
            let requests_last_minute = storage
                .allowed_quota_decision_count(
                    &principal,
                    subject_scoped,
                    now - Duration::minutes(1),
                    now,
                )
                .await?;
            let tokens_today = storage
                .usage_tokens_total(&principal, subject_scoped, day_start, now)
                .await?;

            emit(
                as_json,
                &json!({
                    "subject": principal.subject,
                    "team": principal.team,
                    "model": args.model,
                    "allowed": decision.allowed,
                    "reason": decision.reason,
                    "policy": policy,
                    "usage": {
                        "requests_last_minute": requests_last_minute,
                        "tokens_today": tokens_today
                    }
                }),
            )
        }
        QuotaCommand::Report(args) => {
            let storage = init_storage(&cfg.storage).await?;
            let (from, to) = window(args.hours);
            let usage_summary = reporting::usage_summary(&storage, from, to).await?;
            let decisions = storage.quota_decisions_between(from, to).await?;
            let policy_summary = quota::summarize_quota_policies(&cfg.quotas);
            emit(
                as_json,
                &json!({
                    "hours": args.hours,
                    "from": from,
                    "to": to,
                    "generated_at": Utc::now(),
                    "policies": cfg.quotas,
                    "policy_summary": policy_summary,
                    "decisions": decisions,
                    "usage_summary": usage_summary
                }),
            )
        }
        QuotaCommand::Export => emit(
            as_json,
            &json!({
                "status": "exported",
                "format": "json",
                "count": cfg.quotas.len(),
                "quotas": cfg.quotas
            }),
        ),
        QuotaCommand::Import(args) => {
            let imported = load_quota_policy(&args.path).await?;
            validate_quota_policies(&imported.quotas)?;
            cfg.quotas = imported.quotas;
            config::save(path, &cfg).await?;
            emit(
                as_json,
                &json!({
                    "status": "imported",
                    "format": imported.format,
                    "path": args.path,
                    "count": cfg.quotas.len(),
                    "quotas": cfg.quotas
                }),
            )
        }
        QuotaCommand::List => emit(as_json, &cfg.quotas),
    }
}

fn upsert_quota(quotas: &mut Vec<QuotaConfig>, quota: QuotaConfig) {
    if let Some(existing) = quotas.iter_mut().find(|q| q.subject == quota.subject) {
        *existing = quota;
    } else {
        quotas.push(quota);
    }
}

pub(crate) fn default_quota_policy_format() -> String {
    "json".to_string()
}

async fn load_quota_policy(path: &Path) -> Result<ImportedQuotaPolicy> {
    let body = fs::read_to_string(path)
        .await
        .with_context(|| format!("read quota policy {}", path.display()))?;
    if path.extension().and_then(|ext| ext.to_str()) == Some("toml") {
        let mut imported: ImportedQuotaPolicy =
            toml::from_str(&body).with_context(|| format!("parse TOML {}", path.display()))?;
        imported.format = "toml".to_string();
        return Ok(imported);
    }

    let value: serde_json::Value =
        serde_json::from_str(&body).with_context(|| format!("parse JSON {}", path.display()))?;
    if value.is_array() {
        let quotas = serde_json::from_value(value)
            .with_context(|| format!("parse quotas {}", path.display()))?;
        Ok(ImportedQuotaPolicy {
            format: "json".to_string(),
            quotas,
        })
    } else {
        let mut imported: ImportedQuotaPolicy = serde_json::from_value(value)
            .with_context(|| format!("parse quotas {}", path.display()))?;
        imported.format = "json".to_string();
        Ok(imported)
    }
}

fn validate_quota_policies(quotas: &[QuotaConfig]) -> Result<()> {
    let mut subjects = BTreeMap::new();
    let mut teams = BTreeMap::new();
    for (index, quota) in quotas.iter().enumerate() {
        if quota.subject.trim().is_empty() {
            bail!("quotas[{index}].subject must not be empty");
        }
        if quota.team.trim().is_empty() {
            bail!("quotas[{index}].team must not be empty");
        }
        if quota.requests_per_minute == 0 {
            bail!("quotas[{index}].requests_per_minute must be greater than zero");
        }
        if quota.tokens_per_day == 0 {
            bail!("quotas[{index}].tokens_per_day must be greater than zero");
        }
        if quota.max_concurrency == 0 {
            bail!("quotas[{index}].max_concurrency must be greater than zero");
        }
        if quota
            .allowed_models
            .iter()
            .any(|model| model.trim().is_empty())
        {
            bail!("quotas[{index}].allowed_models must not contain empty model aliases");
        }
        if let Some(first_index) = subjects.insert(quota.subject.as_str(), index) {
            bail!(
                "quotas[{index}].subject duplicates quotas[{first_index}].subject: duplicate subject {:?}",
                quota.subject
            );
        }
        if let Some(first_index) = teams.insert(quota.team.as_str(), index) {
            bail!(
                "quotas[{index}].team duplicates quotas[{first_index}].team: duplicate team {:?}",
                quota.team
            );
        }
    }
    Ok(())
}

fn quota_status_principal(quotas: &[QuotaConfig], args: &QuotaStatusArgs) -> Principal {
    let team = args.team.clone().unwrap_or_else(|| {
        quotas
            .iter()
            .find(|quota| quota.subject == args.subject)
            .map(|quota| quota.team.clone())
            .filter(|team| !team.is_empty())
            .unwrap_or_else(|| "public".to_string())
    });
    Principal {
        subject: args.subject.clone(),
        team,
        scopes: vec![],
        key_id: None,
        key_owner: None,
        key_purpose: None,
        key_status: None,
    }
}

fn matching_quota<'a>(quotas: &'a [QuotaConfig], principal: &Principal) -> Option<&'a QuotaConfig> {
    quotas
        .iter()
        .find(|quota| quota.subject == principal.subject)
        .or_else(|| {
            quotas
                .iter()
                .find(|quota| !quota.team.is_empty() && quota.team == principal.team)
        })
}
