use crate::cli::ServiceLifecycleArgs;
use crate::DEFAULT_SERVICE_NAME;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::fs as stdfs;
use tokio::process::Command as TokioCommand;

#[derive(Debug, Clone, Copy)]
pub(crate) enum ServiceLifecycleAction {
    Status,
    Start,
    Stop,
    Restart,
    Upgrade,
    Downgrade,
}

impl ServiceLifecycleAction {
    fn as_str(self) -> &'static str {
        match self {
            Self::Status => "status",
            Self::Start => "start",
            Self::Stop => "stop",
            Self::Restart => "restart",
            Self::Upgrade => "upgrade",
            Self::Downgrade => "downgrade",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ServiceCommandPlan {
    program: String,
    args: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct OneBinaryEntrypoint {
    program: String,
    args: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ServiceLifecyclePlan {
    status: String,
    action: String,
    service_name: String,
    scope: String,
    dry_run: bool,
    one_binary: bool,
    runtime_backend: rs_llmctl::runtime::RuntimeBackend,
    entrypoint: OneBinaryEntrypoint,
    commands: Vec<ServiceCommandPlan>,
    restart_hint: String,
    artifact_action_supported: bool,
    artifact_action_note: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ServiceCommandResult {
    command: ServiceCommandPlan,
    success: bool,
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct ServiceLifecycleResult {
    status: String,
    action: String,
    service_name: String,
    scope: String,
    dry_run: bool,
    one_binary: bool,
    runtime_backend: rs_llmctl::runtime::RuntimeBackend,
    entrypoint: OneBinaryEntrypoint,
    commands: Vec<ServiceCommandPlan>,
    restart_hint: String,
    results: Vec<ServiceCommandResult>,
}

pub(crate) fn plan_service_lifecycle(
    action: ServiceLifecycleAction,
    args: &ServiceLifecycleArgs,
) -> ServiceLifecyclePlan {
    let scope = if args.user { "user" } else { "system" };
    let service_name = normalize_service_name(&args.service_name);
    let systemctl_scope = if args.user { Some("--user") } else { None };
    let commands = service_systemctl_verbs(action)
        .into_iter()
        .map(|verb| {
            let mut command_args = Vec::new();
            if let Some(scope_arg) = systemctl_scope {
                command_args.push(scope_arg.to_string());
            }
            command_args.push(verb.to_string());
            if verb != "daemon-reload" {
                command_args.push(service_name.clone());
            }
            ServiceCommandPlan {
                program: "systemctl".to_string(),
                args: command_args,
            }
        })
        .collect();
    let artifact_action_supported = !matches!(
        action,
        ServiceLifecycleAction::Upgrade | ServiceLifecycleAction::Downgrade
    );
    let artifact_action_note = if artifact_action_supported {
        None
    } else {
        Some(
            "service upgrade/downgrade is a planning guard only; install a verified release artifact with install.sh or the system package manager, then restart the service"
                .to_string(),
        )
    };

    ServiceLifecyclePlan {
        status: "planned".to_string(),
        action: action.as_str().to_string(),
        service_name: service_name.clone(),
        scope: scope.to_string(),
        dry_run: args.dry_run,
        one_binary: true,
        runtime_backend: rs_llmctl::runtime::RuntimeBackend::CandleNative,
        entrypoint: one_binary_entrypoint(),
        commands,
        restart_hint: restart_hint(scope, &service_name),
        artifact_action_supported,
        artifact_action_note,
    }
}

fn service_systemctl_verbs(action: ServiceLifecycleAction) -> Vec<&'static str> {
    match action {
        ServiceLifecycleAction::Status => vec!["status"],
        ServiceLifecycleAction::Start => vec!["start"],
        ServiceLifecycleAction::Stop => vec!["stop"],
        ServiceLifecycleAction::Restart => vec!["restart"],
        ServiceLifecycleAction::Upgrade | ServiceLifecycleAction::Downgrade => Vec::new(),
    }
}

pub(crate) async fn execute_service_lifecycle(
    plan: ServiceLifecyclePlan,
) -> Result<ServiceLifecycleResult> {
    ensure_service_lifecycle_allowed(&plan)?;
    if !plan.artifact_action_supported {
        bail!(
            "{}",
            plan.artifact_action_note
                .as_deref()
                .unwrap_or("service artifact action is not executable")
        );
    }
    let mut results = Vec::new();
    for command in &plan.commands {
        let output = TokioCommand::new(&command.program)
            .args(&command.args)
            .output()
            .await
            .with_context(|| format!("run {}", shell_words(command)))?;
        results.push(ServiceCommandResult {
            command: command.clone(),
            success: output.status.success(),
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        });
    }

    let success = results.iter().all(|result| result.success);
    Ok(ServiceLifecycleResult {
        status: if success { "ok" } else { "failed" }.to_string(),
        action: plan.action,
        service_name: plan.service_name,
        scope: plan.scope,
        dry_run: false,
        one_binary: plan.one_binary,
        runtime_backend: plan.runtime_backend,
        entrypoint: plan.entrypoint,
        commands: plan.commands,
        restart_hint: plan.restart_hint,
        results,
    })
}

fn ensure_service_lifecycle_allowed(plan: &ServiceLifecyclePlan) -> Result<()> {
    if plan.dry_run || plan.scope != "system" || current_uid().unwrap_or(0) == 0 {
        return Ok(());
    }
    bail!(
        "system service scope requires root or polkit authorization; rerun with sudo or pass --user for a user-scoped service"
    )
}

fn current_uid() -> Option<u32> {
    let status = stdfs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("Uid:")
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|uid| uid.parse::<u32>().ok())
    })
}

fn normalize_service_name(service_name: &str) -> String {
    let trimmed = service_name.trim();
    if trimmed.ends_with(".service") {
        trimmed.to_string()
    } else {
        format!("{trimmed}.service")
    }
}

pub(crate) fn default_restart_hint() -> String {
    restart_hint("system", DEFAULT_SERVICE_NAME)
}

pub(crate) fn one_binary_entrypoint() -> OneBinaryEntrypoint {
    OneBinaryEntrypoint {
        program: "llmctl".to_string(),
        args: vec!["server".to_string(), "run".to_string()],
    }
}

fn restart_hint(scope: &str, service_name: &str) -> String {
    match scope {
        "system" => format!("systemctl restart {service_name}"),
        _ => format!("systemctl --user restart {service_name}"),
    }
}

fn shell_words(command: &ServiceCommandPlan) -> String {
    std::iter::once(command.program.as_str())
        .chain(command.args.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join(" ")
}
