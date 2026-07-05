use crate::*;

pub(crate) async fn integration_command(
    path: &Path,
    command: IntegrationCommand,
    as_json: bool,
) -> Result<()> {
    let cfg = load_config(path).await?;
    match command {
        IntegrationCommand::AqeContract => {
            emit(as_json, &integrations::aqe_governance_contract(&cfg))
        }
    }
}
