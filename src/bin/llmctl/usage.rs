use crate::*;

pub(crate) async fn usage_command(path: &Path, command: UsageCommand, as_json: bool) -> Result<()> {
    let cfg = load_config(path).await?;
    let storage = init_storage(&cfg.storage).await?;
    match command {
        UsageCommand::Report(args) => report_usage(&storage, args.hours, as_json).await,
        UsageCommand::Chargeback(args) => report_chargeback(&storage, args, as_json).await,
    }
}
