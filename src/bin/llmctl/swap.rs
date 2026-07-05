use crate::*;

pub(crate) async fn swap_command(path: &Path, command: SwapCommand, as_json: bool) -> Result<()> {
    let mut cfg = load_config(path).await?;
    match command {
        SwapCommand::Set(args) => {
            cfg.mode = args.mode.into();
            config::save(path, &cfg).await?;
            emit(
                as_json,
                &json!({ "status": "set", "mode": cfg.mode, "models": cfg.models.len() }),
            )
        }
        SwapCommand::Plan(args) => {
            let active = WorkerId::new(args.active);
            let replacement = WorkerId::new(args.replacement);
            let plan = match cfg.mode {
                Mode::ColdSwap => SwapPlan::cold(active, replacement),
                Mode::HotSwap => SwapPlan::hot(active, replacement),
                Mode::Single | Mode::Weighted | Mode::Fallback => {
                    bail!(
                        "swap plan is only supported for cold-swap or hot-swap modes; current mode is {}",
                        mode_name(&cfg.mode)
                    );
                }
            };
            emit(as_json, &plan)
        }
        SwapCommand::Show => emit(
            as_json,
            &json!({ "mode": cfg.mode, "models": cfg.models.len(), "model_aliases": cfg.models.iter().map(|model| &model.alias).collect::<Vec<_>>() }),
        ),
    }
}
