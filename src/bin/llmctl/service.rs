use crate::*;

pub(crate) async fn service_command(command: ServiceCommand, as_json: bool) -> Result<()> {
    let (action, args) = match command {
        ServiceCommand::Status(args) => (ServiceLifecycleAction::Status, args),
        ServiceCommand::Start(args) => (ServiceLifecycleAction::Start, args),
        ServiceCommand::Stop(args) => (ServiceLifecycleAction::Stop, args),
        ServiceCommand::Restart(args) => (ServiceLifecycleAction::Restart, args),
        ServiceCommand::Upgrade(args) => (ServiceLifecycleAction::Upgrade, args),
        ServiceCommand::Downgrade(args) => (ServiceLifecycleAction::Downgrade, args),
    };
    if args.service_name.trim().is_empty() {
        bail!("--service-name must not be empty");
    }
    let plan = plan_service_lifecycle(action, &args);
    if args.dry_run {
        return emit(as_json, &plan);
    }

    let result = execute_service_lifecycle(plan).await?;
    emit(as_json, &result)
}
