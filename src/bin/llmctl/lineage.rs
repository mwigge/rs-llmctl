use crate::*;

pub(crate) async fn lineage_command(
    path: &Path,
    command: LineageCommand,
    as_json: bool,
) -> Result<()> {
    let cfg = load_config(path).await?;
    let path = state_file(&cfg, "lineage-records.jsonl")?;
    match command {
        LineageCommand::Record(args) => {
            let record = json!({
                "schema_version": 1,
                "id": args.id,
                "kind": args.kind,
                "parents": args.parents,
                "sha256": args.sha256,
                "source": args.source,
                "recorded_at": Utc::now()
            });
            append_jsonl(&path, &record).await?;
            emit(as_json, &record)
        }
        LineageCommand::List => emit(
            as_json,
            &json!({
                "schema_version": 1,
                "path": path,
                "records": read_jsonl(&path).await?,
                "joins": Storage::connect_config(&cfg.storage)
                    .await?
                    .request_lineage_joins()
                    .await?
            }),
        ),
    }
}
