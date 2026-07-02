use crate::cli::FirstRunArgs;
use crate::{
    create_storage_dirs, emit, generate_api_key_secret, init_storage, load_config, upsert_model,
    validate_api_key_id, write_secret_file,
};
use anyhow::{bail, Context, Result};
use chrono::Utc;
use rs_llmctl::config::{self, ApiKeyConfig, Config};
use rs_llmctl::model::{self, ModelInstallRequest, ModelSource};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

pub(crate) async fn first_run(path: &Path, args: FirstRunArgs, as_json: bool) -> Result<()> {
    validate_api_key_id(&args.api_key_id)?;
    validate_first_run_identity(&args)?;
    let scopes = first_run_scopes(&args);
    let config_exists = path.exists();
    let mut cfg = if config_exists {
        load_config(path).await?
    } else {
        first_run_default_config(path, args.data_dir.as_deref())
    };
    if let Some(data_dir) = args.data_dir.as_deref() {
        cfg.storage.db_path = data_dir.join("llmctl.db");
        cfg.storage.model_dir = data_dir.join("models");
    }

    let base_url = args
        .base_url
        .clone()
        .unwrap_or_else(|| first_run_base_url(&cfg));
    let smoke_model = first_run_smoke_model(&cfg, &args);
    let smoke_question = args
        .smoke_question
        .as_deref()
        .unwrap_or("Reply with only: llmctl smoke ok");
    let plan = first_run_plan_json(&FirstRunRenderContext {
        path,
        cfg: &cfg,
        args: &args,
        scopes: &scopes,
        config_existed: config_exists,
        base_url: &base_url,
        smoke_model: &smoke_model,
        smoke_question,
    });

    if !args.apply {
        return emit(as_json, &plan);
    }

    let secret_output = args
        .secret_output
        .as_deref()
        .context("first-run --apply requires --secret-output so the raw API key is written once outside config")?;
    if cfg
        .security
        .api_keys
        .iter()
        .any(|key| key.id == args.api_key_id)
    {
        bail!("api key id `{}` already exists", args.api_key_id);
    }

    let secret = generate_api_key_secret(&args.key_prefix);
    let sha256 = hex::encode(Sha256::digest(secret.as_bytes()));
    let last_four = last_four(&secret);
    let key = ApiKeyConfig {
        id: args.api_key_id.clone(),
        sha256,
        subject: args.subject.clone(),
        team: args.team.clone(),
        scopes: scopes.clone(),
        created_at: Some(Utc::now()),
        expires_at: None,
        rotated_at: None,
        owner: args.owner.clone(),
        purpose: args
            .purpose
            .clone()
            .or_else(|| Some("first-run operator access".to_string())),
        last_four: Some(last_four.clone()),
        fingerprint: None,
        status: "active".to_string(),
    };
    cfg.security.require_auth = true;
    cfg.security.api_keys.push(key);

    let installed_model = if let Some(model_path) = args.starter_model_path.as_ref() {
        create_storage_dirs(&cfg.storage).await?;
        let installed = model::install_model(&ModelInstallRequest {
            alias: args.starter_model_alias.clone(),
            source: ModelSource::LocalPath {
                path: model_path.clone(),
            },
            cache_dir: cfg.storage.model_dir.clone(),
            copy_to_cache: false,
            expected_sha256: None,
            role: args.starter_model_role.clone(),
            family: Some(args.starter_model_family.clone()),
            weight: args.starter_model_weight,
        })
        .await?;
        upsert_model(&mut cfg.models, installed.config.clone());
        Some(installed)
    } else {
        None
    };

    create_storage_dirs(&cfg.storage).await?;
    write_secret_file(secret_output, &secret).await?;
    config::save(path, &cfg).await?;
    let storage = init_storage(&cfg.storage).await?;
    for model in &cfg.models {
        storage.upsert_model(model).await?;
    }

    emit(
        as_json,
        &first_run_applied_json(
            &FirstRunRenderContext {
                path,
                cfg: &cfg,
                args: &args,
                scopes: &scopes,
                config_existed: config_exists,
                base_url: &base_url,
                smoke_model: &smoke_model,
                smoke_question,
            },
            secret_output,
            &last_four,
            installed_model.as_ref(),
        ),
    )
}

struct FirstRunRenderContext<'a> {
    path: &'a Path,
    cfg: &'a Config,
    args: &'a FirstRunArgs,
    scopes: &'a [String],
    config_existed: bool,
    base_url: &'a str,
    smoke_model: &'a str,
    smoke_question: &'a str,
}

fn first_run_default_config(path: &Path, data_dir: Option<&Path>) -> Config {
    let mut cfg = Config::default();
    let state_dir = data_dir.map(Path::to_path_buf).unwrap_or_else(|| {
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .join("rs-llmctl-state")
    });
    cfg.storage.db_path = state_dir.join("llmctl.db");
    cfg.storage.model_dir = state_dir.join("models");
    cfg.resources.cpu_only = true;
    cfg
}

fn validate_first_run_identity(args: &FirstRunArgs) -> Result<()> {
    if args.subject.trim().is_empty() {
        bail!("subject must not be empty");
    }
    if args.team.trim().is_empty() {
        bail!("team must not be empty");
    }
    if args.key_prefix.trim().is_empty() {
        bail!("key-prefix must not be empty");
    }
    if args.api_key_env.trim().is_empty() {
        bail!("api-key-env must not be empty");
    }
    if args.starter_model_alias.trim().is_empty() {
        bail!("starter-model-alias must not be empty");
    }
    if args.starter_model_role.trim().is_empty() {
        bail!("starter-model-role must not be empty");
    }
    Ok(())
}

fn first_run_scopes(args: &FirstRunArgs) -> Vec<String> {
    if args.scopes.is_empty() {
        vec!["chat".to_string(), "models.read".to_string()]
    } else {
        args.scopes.clone()
    }
}

fn first_run_base_url(cfg: &Config) -> String {
    format!("http://{}:{}/v1", cfg.server.host, cfg.server.port)
}

fn first_run_smoke_model(cfg: &Config, args: &FirstRunArgs) -> String {
    if args.starter_model_path.is_some() {
        return args.starter_model_alias.clone();
    }
    cfg.models
        .iter()
        .find(|model| model.weight > 0)
        .or_else(|| cfg.models.first())
        .map(|model| model.alias.clone())
        .unwrap_or_else(|| args.starter_model_alias.clone())
}

fn first_run_plan_json(context: &FirstRunRenderContext<'_>) -> Value {
    let next_command = first_run_apply_command(context);
    json!({
        "status": "planned",
        "mode": "dry-run",
        "side_effects": false,
        "config": context.path,
        "config_exists": context.config_existed,
        "api_key": first_run_api_key_json(context.args, context.scopes, None, None),
        "starter_model": first_run_starter_model_plan(context.args),
        "config_changes": {
            "write_config": true,
            "require_auth": true,
            "storage_db_path": context.cfg.storage.db_path,
            "model_dir": context.cfg.storage.model_dir
        },
        "smoke": first_run_smoke_json(
            context.base_url,
            context.smoke_model,
            context.smoke_question,
            &context.args.api_key_env
        ),
        "next_command": shell_join(&next_command),
        "next_command_argv": next_command
    })
}

fn first_run_apply_command(context: &FirstRunRenderContext<'_>) -> Vec<String> {
    let args = context.args;
    let mut command = vec![
        "llmctl".to_string(),
        "--config".to_string(),
        context.path.display().to_string(),
        "first-run".to_string(),
        "--apply".to_string(),
        "--secret-output".to_string(),
        args.secret_output
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<secret-file>".to_string()),
        "--key-prefix".to_string(),
        args.key_prefix.clone(),
        "--api-key-id".to_string(),
        args.api_key_id.clone(),
        "--subject".to_string(),
        args.subject.clone(),
        "--team".to_string(),
        args.team.clone(),
    ];
    for scope in &args.scopes {
        command.push("--scope".to_string());
        command.push(scope.clone());
    }
    if let Some(owner) = &args.owner {
        command.push("--owner".to_string());
        command.push(owner.clone());
    }
    if let Some(purpose) = &args.purpose {
        command.push("--purpose".to_string());
        command.push(purpose.clone());
    }
    if let Some(data_dir) = &args.data_dir {
        command.push("--data-dir".to_string());
        command.push(data_dir.display().to_string());
    }
    if let Some(path) = &args.starter_model_path {
        command.push("--starter-model-path".to_string());
        command.push(path.display().to_string());
    }
    command.push("--starter-model-alias".to_string());
    command.push(args.starter_model_alias.clone());
    command.push("--starter-model-role".to_string());
    command.push(args.starter_model_role.clone());
    command.push("--starter-model-family".to_string());
    command.push(args.starter_model_family.clone());
    command.push("--starter-model-weight".to_string());
    command.push(args.starter_model_weight.to_string());
    if let Some(base_url) = &args.base_url {
        command.push("--base-url".to_string());
        command.push(base_url.clone());
    }
    command.push("--api-key-env".to_string());
    command.push(args.api_key_env.clone());
    if let Some(question) = &args.smoke_question {
        command.push("--smoke-question".to_string());
        command.push(question.clone());
    }
    command
}

fn shell_join(args: &[String]) -> String {
    args.iter()
        .map(|arg| {
            if arg.chars().all(|ch| {
                ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '/' | ':' | '<' | '>')
            }) {
                arg.clone()
            } else {
                format!("'{}'", arg.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn first_run_applied_json(
    context: &FirstRunRenderContext<'_>,
    secret_output: &Path,
    last_four: &str,
    installed_model: Option<&model::InstalledModel>,
) -> Value {
    json!({
        "status": "applied",
        "mode": "apply",
        "side_effects": true,
        "config": context.path,
        "config_existed": context.config_existed,
        "api_key": first_run_api_key_json(
            context.args,
            context.scopes,
            Some(secret_output),
            Some(last_four)
        ),
        "starter_model": first_run_starter_model_applied(context.args, installed_model),
        "config_changes": {
            "wrote_config": true,
            "require_auth": context.cfg.security.require_auth,
            "api_keys": context.cfg.security.api_keys.len(),
            "models": context.cfg.models.len(),
            "storage_db_path": context.cfg.storage.db_path,
            "model_dir": context.cfg.storage.model_dir
        },
        "smoke": first_run_smoke_json(
            context.base_url,
            context.smoke_model,
            context.smoke_question,
            &context.args.api_key_env
        )
    })
}

fn first_run_api_key_json(
    args: &FirstRunArgs,
    scopes: &[String],
    secret_output: Option<&Path>,
    last_four: Option<&str>,
) -> Value {
    json!({
        "action": "generate",
        "id": args.api_key_id,
        "subject": args.subject,
        "team": args.team,
        "scopes": scopes,
        "owner": args.owner,
        "purpose": args.purpose.as_deref().unwrap_or("first-run operator access"),
        "secret_output": args.secret_output,
        "secret_written": secret_output.map(|path| path.display().to_string()),
        "last_four": last_four,
        "sha256_present": secret_output.is_some(),
        "config_storage": "sha256-only",
        "plaintext_secret_storage": false,
        "print_secret": false
    })
}

fn first_run_starter_model_plan(args: &FirstRunArgs) -> Value {
    match args.starter_model_path.as_ref() {
        Some(path) => json!({
            "action": "configure-local",
            "alias": args.starter_model_alias,
            "role": args.starter_model_role,
            "family": args.starter_model_family,
            "weight": args.starter_model_weight,
            "path": path,
            "source_kind": "local",
            "network": false,
            "exists": path.exists()
        }),
        None => json!({
            "action": "recommend",
            "alias": args.starter_model_alias,
            "role": args.starter_model_role,
            "family": args.starter_model_family,
            "weight": args.starter_model_weight,
            "source_kind": "none",
            "network": false,
            "recommendation": "provide --starter-model-path /path/to/model.gguf, /path/to/model.safetensors with sibling config.json/tokenizer.json, or a safetensors directory containing config.json, tokenizer.json, and weights; offline manifests are also supported"
        }),
    }
}

fn first_run_starter_model_applied(
    args: &FirstRunArgs,
    installed_model: Option<&model::InstalledModel>,
) -> Value {
    if let Some(installed) = installed_model {
        json!({
            "action": "configured",
            "alias": installed.alias,
            "role": installed.config.role,
            "family": installed.config.family,
            "weight": installed.config.weight,
            "path": installed.path,
            "source_kind": "local",
            "network": false,
            "sha256": installed.sha256,
            "bytes": installed.bytes,
            "verified": installed.verification.verified
        })
    } else {
        first_run_starter_model_plan(args)
    }
}

fn first_run_smoke_json(base_url: &str, model: &str, question: &str, api_key_env: &str) -> Value {
    json!({
        "action": "plan",
        "base_url": base_url,
        "model": model,
        "question": question,
        "api_key_env": api_key_env,
        "ask_question": {
            "helper": "ask_question",
            "crate": "rs-llmctl-client",
            "environment": {
                "LLMCTL_BASE_URL": base_url,
                "api_key_env": api_key_env
            },
            "metadata": {
                "session_id": "first-run-smoke",
                "purpose": "operator-smoke"
            }
        },
        "openai_compatible": {
            "method": "POST",
            "endpoint": "/v1/chat/completions",
            "url": format!("{}/chat/completions", base_url.trim_end_matches('/')),
            "headers": [
                format!("Authorization: Bearer ${api_key_env}"),
                "Content-Type: application/json"
            ],
            "body": {
                "model": model,
                "messages": [
                    { "role": "user", "content": question }
                ],
                "metadata": {
                    "session_id": "first-run-smoke",
                    "purpose": "operator-smoke"
                }
            }
        }
    })
}

fn last_four(value: &str) -> String {
    value
        .chars()
        .rev()
        .take(4)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>()
}
