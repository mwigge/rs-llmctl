use crate::cli::AmdCommand;
use crate::emit;
use anyhow::{bail, Context, Result};
use std::path::PathBuf;

pub(crate) async fn amd_command(command: AmdCommand, as_json: bool) -> Result<()> {
    match command {
        AmdCommand::Qualify(args) => emit(
            as_json,
            &rs_llmctl::amd::qualification_report_with_evidence(
                args.preview,
                args.arch_opt_in,
                args.evidence.as_deref(),
            ),
        ),
        AmdCommand::InstallServer(args) => {
            let script = args.script.unwrap_or_else(|| {
                std::env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .join("scripts")
                    .join("install-amd-hip.sh")
            });
            if !script.exists() {
                bail!(
                    "install script not found: {}  (set --script or run from the rs-llmctl repo root)",
                    script.display()
                );
            }
            let mut cmd = std::process::Command::new("bash");
            cmd.arg(&script);
            if args.dry_run {
                cmd.env("DRY_RUN", "1");
            }
            let status = cmd
                .status()
                .with_context(|| format!("failed to run {}", script.display()))?;
            if !status.success() {
                bail!("install script exited with status {status}");
            }
            Ok(())
        }
    }
}
