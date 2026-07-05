use crate::*;

pub(crate) async fn model_command(path: &Path, command: ModelCommand, as_json: bool) -> Result<()> {
    let mut cfg = load_config(path).await?;
    match command {
        ModelCommand::Install(args) => {
            create_storage_dirs(&cfg.storage).await?;
            let installed = model::install_model(&ModelInstallRequest {
                alias: args.alias.clone(),
                source: model_source(&args.source),
                cache_dir: cfg.storage.model_dir.clone(),
                copy_to_cache: args.copy,
                expected_sha256: args.sha256,
                role: args.role.clone(),
                family: Some(args.family.clone()),
                weight: args.weight,
            })
            .await?;

            upsert_model(&mut cfg.models, installed.config.clone());
            config::save(path, &cfg).await?;

            let storage = init_storage(&cfg.storage).await?;
            for model in &cfg.models {
                storage.upsert_model(model).await?;
            }

            emit(
                as_json,
                &json!({ "status": "installed", "model": installed, "models": cfg.models }),
            )
        }
        ModelCommand::Status(args) => {
            let model = cfg
                .models
                .iter()
                .find(|model| model.alias == args.alias)
                .with_context(|| format!("model alias '{}' is not configured", args.alias))?;
            let status = if model.weight == 0 {
                "stopped"
            } else {
                "running"
            };
            let readiness = model
                .family
                .as_deref()
                .filter(|family| family.eq_ignore_ascii_case("gemma4"))
                .map(|_| {
                    rs_llmctl::readiness::read_state(&rs_llmctl::readiness::evidence_path(
                        &cfg.storage.model_dir,
                        &model.alias,
                    ))
                });

            emit(
                as_json,
                &json!({
                    "status": status,
                    "alias": &model.alias,
                    "weight": model.weight,
                    "restart_required": false,
                    "restart_hint": default_restart_hint(),
                    "runtime_backend": &cfg.runtime.backend,
                    "one_binary": true,
                    "entrypoint": one_binary_entrypoint(),
                    "readiness": readiness,
                    "model": model,
                }),
            )
        }
        ModelCommand::Start(args) => {
            let model = cfg
                .models
                .iter_mut()
                .find(|model| model.alias == args.alias)
                .with_context(|| format!("model alias '{}' is not configured", args.alias))?;
            let previous_weight = model.weight;
            let weight = args.weight.unwrap_or_else(|| previous_weight.max(1));
            if args.dry_run {
                return emit(
                    as_json,
                    &json!({
                        "status": "planned",
                        "action": "start",
                        "alias": &model.alias,
                        "previous_weight": previous_weight,
                        "weight": weight,
                        "restart_required": true,
                        "restart_hint": default_restart_hint(),
                        "runtime_backend": &cfg.runtime.backend,
                        "one_binary": true,
                        "entrypoint": one_binary_entrypoint(),
                        "model": model,
                    }),
                );
            }
            model.weight = weight;
            let model_config = model.clone();
            persist_models(path, &cfg).await?;

            emit(
                as_json,
                &json!({
                    "status": "started",
                    "action": "start",
                    "alias": &model_config.alias,
                    "previous_weight": previous_weight,
                    "weight": model_config.weight,
                    "restart_required": true,
                    "restart_hint": default_restart_hint(),
                    "runtime_backend": &cfg.runtime.backend,
                    "one_binary": true,
                    "entrypoint": one_binary_entrypoint(),
                    "model": &model_config,
                    "models": &cfg.models,
                }),
            )
        }
        ModelCommand::Stop(args) => {
            let model = cfg
                .models
                .iter_mut()
                .find(|model| model.alias == args.alias)
                .with_context(|| format!("model alias '{}' is not configured", args.alias))?;
            let previous_weight = model.weight;
            if args.dry_run {
                return emit(
                    as_json,
                    &json!({
                        "status": "planned",
                        "action": "stop",
                        "alias": &model.alias,
                        "previous_weight": previous_weight,
                        "weight": 0,
                        "restart_required": true,
                        "restart_hint": default_restart_hint(),
                        "runtime_backend": &cfg.runtime.backend,
                        "one_binary": true,
                        "entrypoint": one_binary_entrypoint(),
                        "model": model,
                    }),
                );
            }
            model.weight = 0;
            let model_config = model.clone();
            persist_models(path, &cfg).await?;

            emit(
                as_json,
                &json!({
                    "status": "stopped",
                    "action": "stop",
                    "alias": &model_config.alias,
                    "previous_weight": previous_weight,
                    "weight": model_config.weight,
                    "restart_required": true,
                    "restart_hint": default_restart_hint(),
                    "runtime_backend": &cfg.runtime.backend,
                    "one_binary": true,
                    "entrypoint": one_binary_entrypoint(),
                    "model": &model_config,
                    "models": &cfg.models,
                }),
            )
        }
        ModelCommand::Update(args) => {
            replace_model(path, &mut cfg, args, "update", "updated", as_json).await
        }
        ModelCommand::Upgrade(args) => {
            replace_model(path, &mut cfg, args, "upgrade", "upgraded", as_json).await
        }
        ModelCommand::Downgrade(args) => {
            replace_model(path, &mut cfg, args, "downgrade", "downgraded", as_json).await
        }
        ModelCommand::Drift(args) => {
            let storage = init_storage(&cfg.storage).await?;
            record_latency_drift_observations(&storage, args.hours).await?;
            report_observations(&storage, "drift", args.hours, as_json).await
        }
        ModelCommand::ImportManifest(args) => {
            create_storage_dirs(&cfg.storage).await?;
            let installed = model::import_offline_manifest(&args.manifest).await?;

            for model in &installed {
                upsert_model(&mut cfg.models, model.config.clone());
            }
            persist_models(path, &cfg).await?;

            emit(
                as_json,
                &json!({ "status": "imported", "imported": installed, "models": cfg.models }),
            )
        }
        ModelCommand::Inventory => {
            let storage = init_storage(&cfg.storage).await?;
            let inventory = model_inventory(&cfg, &storage).await?;
            emit(as_json, &inventory)
        }
        ModelCommand::List => emit(as_json, &cfg.models),
        ModelCommand::Profile { command } => {
            model_profile_command(&cfg.storage.model_dir, command, as_json).await
        }
    }
}

pub(crate) async fn model_profile_command(
    model_dir: &Path,
    command: ModelProfileCommand,
    as_json: bool,
) -> Result<()> {
    match command {
        ModelProfileCommand::List => emit(as_json, &rs_llmctl::profiles::list_profiles(model_dir)?),
        ModelProfileCommand::Inspect(args) => emit(
            as_json,
            &rs_llmctl::profiles::read_profile(model_dir, &args.alias)?,
        ),
        ModelProfileCommand::ImportLocal(args) => {
            let profile =
                rs_llmctl::profiles::import_local_candidate(&args.path, &args.alias).await?;
            let path = rs_llmctl::profiles::write_profile(model_dir, &profile)?;
            emit(
                as_json,
                &json!({"status": "candidate", "path": path, "profile": profile}),
            )
        }
        ModelProfileCommand::ImportCatalog(args) => {
            let profile = rs_llmctl::profiles::import_catalog_candidate(&args.id)?;
            let path = rs_llmctl::profiles::write_profile(model_dir, &profile)?;
            emit(
                as_json,
                &json!({"status": "candidate", "path": path, "profile": profile}),
            )
        }
        ModelProfileCommand::Qualify(args) => {
            let profile = rs_llmctl::profiles::read_profile(model_dir, &args.alias)?;
            let (profile, policy) =
                rs_llmctl::profiles::qualify_profile(profile, args.available_vram_bytes);
            let path = rs_llmctl::profiles::write_profile(model_dir, &profile)?;
            emit(
                as_json,
                &json!({"status": profile.qualification, "path": path, "policy": policy, "profile": profile}),
            )
        }
        ModelProfileCommand::Quarantine(args) => {
            let mut profile = rs_llmctl::profiles::read_profile(model_dir, &args.alias)?;
            profile.qualification = rs_llmctl::profiles::QualificationStatus::Quarantined;
            profile.quarantine_reason = Some(args.reason);
            let path = rs_llmctl::profiles::write_profile(model_dir, &profile)?;
            emit(
                as_json,
                &json!({"status": "quarantined", "path": path, "profile": profile}),
            )
        }
        ModelProfileCommand::Remove(args) => {
            rs_llmctl::profiles::remove_profile(model_dir, &args.alias)?;
            emit(as_json, &json!({"status": "removed", "alias": args.alias}))
        }
        ModelProfileCommand::Adapters => emit(as_json, &rs_llmctl::profiles::backend_catalog()),
    }
}

async fn model_inventory(cfg: &Config, storage: &Storage) -> Result<ModelInventoryOutput> {
    let persisted = storage
        .list_models()
        .await?
        .into_iter()
        .map(|record| (record.alias.clone(), record))
        .collect::<BTreeMap<_, _>>();

    let models = cfg
        .models
        .iter()
        .map(|model| {
            let persisted = persisted.get(&model.alias);
            ModelInventoryItem {
                alias: model.alias.clone(),
                role: model.role.clone(),
                weight: model.weight,
                path: path_basename(&model.path),
                updated_at: persisted.map(|record| record.updated_at),
                readiness: model
                    .family
                    .as_deref()
                    .filter(|family| family.eq_ignore_ascii_case("gemma4"))
                    .map(|_| {
                        rs_llmctl::readiness::read_state(&rs_llmctl::readiness::evidence_path(
                            &cfg.storage.model_dir,
                            &model.alias,
                        ))
                    }),
            }
        })
        .collect();

    Ok(ModelInventoryOutput {
        configured: cfg.models.len(),
        models,
    })
}

fn model_source(source: &str) -> ModelSource {
    if let Some(model) = model::catalog_model(source) {
        return ModelSource::HuggingFace {
            repo: model.repo.to_string(),
            filename: model.filename.to_string(),
            revision: model.revision.to_string(),
        };
    }
    if source.starts_with("http://") || source.starts_with("https://") {
        ModelSource::DirectUrl {
            url: source.to_string(),
        }
    } else {
        ModelSource::LocalPath {
            path: PathBuf::from(source),
        }
    }
}

fn path_basename(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.display().to_string())
}

async fn persist_models(path: &Path, cfg: &Config) -> Result<()> {
    config::save(path, cfg).await?;

    let storage = init_storage(&cfg.storage).await?;
    for model in &cfg.models {
        storage.upsert_model(model).await?;
    }
    Ok(())
}

async fn replace_model(
    path: &Path,
    cfg: &mut Config,
    args: ModelReplaceArgs,
    action: &str,
    status: &str,
    as_json: bool,
) -> Result<()> {
    let previous = cfg
        .models
        .iter()
        .find(|model| model.alias == args.alias)
        .cloned()
        .with_context(|| format!("model alias '{}' is not configured", args.alias))?;
    let target_alias = args.new_alias.unwrap_or_else(|| previous.alias.clone());
    let role = args.role.unwrap_or_else(|| previous.role.clone());
    let family = args.family.or_else(|| previous.family.clone());
    let weight = args.weight.unwrap_or(previous.weight);

    if args.dry_run {
        return emit(
            as_json,
            &json!({
                "status": "planned",
                "action": action,
                "alias": &previous.alias,
                "new_alias": &target_alias,
                "previous_weight": previous.weight,
                "weight": weight,
                "role": role,
                "source": &args.source,
                "copy": args.copy,
                "sha256": args.sha256,
                "restart_required": true,
                "restart_hint": default_restart_hint(),
                "runtime_backend": &cfg.runtime.backend,
                "one_binary": true,
                "entrypoint": one_binary_entrypoint(),
                "previous_model": &previous,
            }),
        );
    }

    create_storage_dirs(&cfg.storage).await?;
    let installed = model::install_model(&ModelInstallRequest {
        alias: target_alias.clone(),
        source: model_source(&args.source),
        cache_dir: cfg.storage.model_dir.clone(),
        copy_to_cache: args.copy,
        expected_sha256: args.sha256,
        role,
        family,
        weight,
    })
    .await?;

    if target_alias != previous.alias {
        cfg.models.retain(|model| model.alias != previous.alias);
    }
    upsert_model(&mut cfg.models, installed.config.clone());
    persist_models(path, cfg).await?;

    emit(
        as_json,
        &json!({
            "status": status,
            "action": action,
            "alias": &previous.alias,
            "new_alias": &target_alias,
            "previous_model": &previous,
            "model": &installed,
            "restart_required": true,
            "restart_hint": default_restart_hint(),
            "runtime_backend": &cfg.runtime.backend,
            "one_binary": true,
            "entrypoint": one_binary_entrypoint(),
            "models": &cfg.models,
        }),
    )
}
