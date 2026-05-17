#!/usr/bin/env sh
set -eu

repo="${RS_LLMCTL_REPO:-mwigge/rs-llmctl}"
version="${RS_LLMCTL_VERSION:-latest}"
tarball="${RS_LLMCTL_TARBALL:-}"
prefix="${PREFIX:-/usr/local}"
bin_dir="${prefix}/bin"
config_dir="${LLMCTL_CONFIG_DIR:-/etc/rs-llmctl}"
config_file="${LLMCTL_CONFIG:-${config_dir}/config.toml}"
state_dir="${LLMCTL_STATE_DIR:-/var/lib/rs-llmctl}"
log_dir="${LLMCTL_LOG_DIR:-/var/log/rs-llmctl}"
service_name="${LLMCTL_SERVICE_NAME:-llmctld}"
install_systemd="${LLMCTL_INSTALL_SYSTEMD:-auto}"
start_service="${LLMCTL_START_SERVICE:-0}"
enable_audit_timer="${LLMCTL_ENABLE_AUDIT_TIMER:-0}"
systemd_dir="${LLMCTL_SYSTEMD_DIR:-/etc/systemd/system}"
os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"

info() { printf '  -> %s\n' "$*"; }
ok() { printf '  ok %s\n' "$*"; }
warn() { printf '  ! %s\n' "$*" >&2; }
fatal() { printf '  error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fatal "$1 not found; $2"; }
is_root() { [ "$(id -u)" -eq 0 ]; }

can_write_path() {
  path=$1
  if [ -e "${path}" ]; then
    [ -w "${path}" ]
    return
  fi

  parent=${path}
  while :; do
    parent=$(dirname "${parent}")
    if [ -d "${parent}" ]; then
      [ -w "${parent}" ] && [ -x "${parent}" ]
      return
    fi
    [ "${parent}" = "/" ] && return 1
  done
}

run_root() {
  if is_root; then
    "$@"
  elif [ -n "${sudo_cmd}" ]; then
    sudo "$@"
  else
    fatal "root privileges are required for: $* (rerun as root or install sudo)"
  fi
}

run_for_path() {
  path=$1
  shift
  if is_root || can_write_path "${path}"; then
    "$@"
  else
    run_root "$@"
  fi
}

group_exists() {
  if command -v getent >/dev/null 2>&1; then
    getent group "$1" >/dev/null 2>&1
  else
    grep -q "^$1:" /etc/group 2>/dev/null
  fi
}

user_exists() {
  id -u "$1" >/dev/null 2>&1
}

nologin_shell() {
  for shell_path in /usr/sbin/nologin /sbin/nologin /bin/false; do
    if [ -x "${shell_path}" ]; then
      printf '%s\n' "${shell_path}"
      return
    fi
  done
  printf '/bin/false\n'
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    fatal "sha256sum or shasum not found; install coreutils or perl digest tools"
  fi
}

verify_archive_checksum() {
  archive=$1
  checksums=$2
  name=$(basename "${archive}")
  expected=$(awk -v name="${name}" '$2 == name || $2 == "./" name { print $1; found=1 } END { if (!found) exit 1 }' "${checksums}") \
    || fatal "SHA256SUMS does not contain ${name}"
  actual=$(sha256_file "${archive}")
  [ "${expected}" = "${actual}" ] || fatal "checksum mismatch for ${name}"
}

safe_extract_archive() {
  archive=$1
  destination=$2
  mkdir -p "${destination}"
  tar -tzf "${archive}" | while IFS= read -r member; do
    case "${member}" in
      ""|/*|*"/../"*|../*|*"/.."|..)
        fatal "unsafe archive member path: ${member}"
        ;;
    esac
  done
  tar -tvzf "${archive}" | while IFS= read -r line; do
    mode=$(printf '%s\n' "${line}" | awk '{print substr($1,1,1)}')
    case "${mode}" in
      -|d) ;;
      *) fatal "unsafe archive member type in: ${line}" ;;
    esac
  done
  tar -xzf "${archive}" -C "${destination}"
}

find_llmctl_binary() {
  root=$1
  candidate=$(find "${root}" -type f -name llmctl -perm -111 | head -n 1)
  [ -n "${candidate}" ] || fatal "release archive is missing executable llmctl"
  printf '%s\n' "${candidate}"
}

case "${arch}" in
  x86_64|amd64) arch="x86_64" ;;
  aarch64|arm64) arch="aarch64" ;;
  *) fatal "unsupported architecture: ${arch}" ;;
esac

case "${os}" in
  linux|darwin) ;;
  *) fatal "unsupported OS: ${os}" ;;
esac

if [ "${os}" != "linux" ]; then
  install_systemd="0"
elif [ "${install_systemd}" = "auto" ]; then
  if command -v systemctl >/dev/null 2>&1; then
    install_systemd="1"
  else
    install_systemd="0"
  fi
fi

case "${install_systemd}" in
  0|1) ;;
  *) fatal "LLMCTL_INSTALL_SYSTEMD must be auto, 0, or 1" ;;
esac

if [ "${install_systemd}" = "1" ]; then
  case "${prefix}" in
    /home/*)
      fatal "system service installs cannot use a home-directory PREFIX; use PREFIX=/usr/local or set LLMCTL_INSTALL_SYSTEMD=0"
      ;;
  esac
  if [ -n "${HOME:-}" ]; then
    case "${prefix}" in
      "${HOME}"/*)
        fatal "system service installs cannot use a home-directory PREFIX; use PREFIX=/usr/local or set LLMCTL_INSTALL_SYSTEMD=0"
        ;;
    esac
  fi
  case "${service_name}" in
    *[!abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.@-]*|""|*/*)
      fatal "LLMCTL_SERVICE_NAME must be a systemd unit basename, not: ${service_name}"
      ;;
  esac
  case "${bin_dir}${config_file}${state_dir}${log_dir}${systemd_dir}" in
    *[[:space:]]*) fatal "system service install paths cannot contain whitespace" ;;
  esac
fi

if is_root; then
  sudo_cmd=""
elif command -v sudo >/dev/null 2>&1; then
  sudo_cmd="sudo"
else
  sudo_cmd=""
fi

if [ -z "${tarball}" ]; then
  need curl "install curl"
fi
need tar "install tar"
need install "install coreutils"
need mktemp "install mktemp"
need awk "install awk"
need find "install find"
need head "install coreutils"
need grep "install grep"

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
extract_dir="${tmp}/extract"

if [ -n "${tarball}" ]; then
  [ -f "${tarball}" ] || fatal "RS_LLMCTL_TARBALL does not exist: ${tarball}"
  info "Using local tarball ${tarball}"
  if [ -n "${RS_LLMCTL_SHA256SUMS:-}" ]; then
    [ -f "${RS_LLMCTL_SHA256SUMS}" ] || fatal "RS_LLMCTL_SHA256SUMS does not exist: ${RS_LLMCTL_SHA256SUMS}"
    verify_archive_checksum "${tarball}" "${RS_LLMCTL_SHA256SUMS}"
  else
    warn "local tarball install has no RS_LLMCTL_SHA256SUMS verification file"
  fi
  safe_extract_archive "${tarball}" "${extract_dir}"
else
  if [ "${version}" = "latest" ]; then
    url="https://github.com/${repo}/releases/latest/download/rs-llmctl-${os}-${arch}.tar.gz"
    sums_url="https://github.com/${repo}/releases/latest/download/SHA256SUMS"
  else
    url="https://github.com/${repo}/releases/download/${version}/rs-llmctl-${os}-${arch}.tar.gz"
    sums_url="https://github.com/${repo}/releases/download/${version}/SHA256SUMS"
  fi

  info "Downloading ${url}"
  curl -fsSL "${url}" -o "${tmp}/rs-llmctl.tar.gz"
  info "Downloading ${sums_url}"
  curl -fsSL "${sums_url}" -o "${tmp}/SHA256SUMS"
  verify_archive_checksum "${tmp}/rs-llmctl.tar.gz" "${tmp}/SHA256SUMS"
  warn "verified archive checksum only; verify SHA256SUMS signature from the GitHub release before production installs when publisher authentication is required"
  safe_extract_archive "${tmp}/rs-llmctl.tar.gz" "${extract_dir}"
fi

llmctl_bin=$(find_llmctl_binary "${extract_dir}")
if find "${extract_dir}" -type f -name llmctld -perm -111 | grep . >/dev/null 2>&1; then
  warn "release archive includes legacy llmctld; default install uses llmctl only"
fi

info "Installing binaries to ${bin_dir}"
run_for_path "${bin_dir}" mkdir -p "${bin_dir}"
run_for_path "${bin_dir}/llmctl" install -m 0755 "${llmctl_bin}" "${bin_dir}/llmctl"
ok "Installed llmctl"

if [ "${install_systemd}" = "1" ]; then
  need systemctl "install systemd or set LLMCTL_INSTALL_SYSTEMD=0"
  need id "install coreutils"
  need grep "install grep"

  info "Creating system account and data directories"
  if ! group_exists llmctl; then
    need groupadd "install shadow-utils or passwd"
    run_root groupadd --system llmctl
  fi
  if ! user_exists llmctl; then
    need useradd "install shadow-utils or passwd"
    run_root useradd --system --gid llmctl --home-dir "${state_dir}" --shell "$(nologin_shell)" llmctl
  fi
  run_root install -d -m 0750 -o llmctl -g llmctl "${state_dir}" "${state_dir}/models" "${state_dir}/reports" "${log_dir}"
  run_root install -d -m 0750 -o root -g llmctl "${config_dir}"

  if [ ! -f "${config_file}" ]; then
    info "Writing default loopback config to ${config_file}"
    config_tmp="${tmp}/config.toml"
    cat > "${config_tmp}" <<EOF
mode = "single"

[server]
host = "127.0.0.1"
port = 8765
worker_base_port = 18765
context_size = 8192

[security]
production = false
require_auth = false
bind_external = false
api_keys = []

[resources]
budget = 0.80
cpu_only = true
gpu_vendor = "none"

[storage]
db_path = "${state_dir}/llmctl.db"
model_dir = "${state_dir}/models"

[observability]
service_name = "rs-llmctl"
environment = "local"
traces_enabled = true
metrics_enabled = true
logs_enabled = true

[audit]
retention-days = 30
report-directory = "${state_dir}/reports"
report-formats = ["json"]
monthly-reports = false
EOF
    run_root install -m 0640 -o root -g llmctl "${config_tmp}" "${config_file}"
    ok "Installed default config"
  else
    ok "Keeping existing config at ${config_file}"
  fi

  info "Installing systemd service"
  unit_tmp="${tmp}/${service_name}.service"
  cpu_quota_percent=80
  if command -v nproc >/dev/null 2>&1; then
    cpu_quota_percent=$(( $(nproc) * 80 ))
  fi
  cat > "${unit_tmp}" <<EOF
[Unit]
Description=rs-llmctl OpenAI-compatible model serving daemon
Documentation=https://github.com/${repo}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=llmctl
Group=llmctl
Environment=LLMCTL_CONFIG=${config_file}
ExecStart=${bin_dir}/llmctl --config \${LLMCTL_CONFIG} server run
Restart=on-failure
RestartSec=5s

CPUAccounting=true
MemoryAccounting=true
CPUQuota=${cpu_quota_percent}%
MemoryMax=80%

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=${state_dir} ${log_dir}

[Install]
WantedBy=multi-user.target
EOF
  run_root install -d -m 0755 "${systemd_dir}"
  run_root install -m 0644 "${unit_tmp}" "${systemd_dir}/${service_name}.service"
  if [ -f "${extract_dir}/packaging/systemd/llmctl-monthly-audit.timer" ]; then
    audit_unit_tmp="${tmp}/llmctl-monthly-audit.service"
    cat > "${audit_unit_tmp}" <<EOF
[Unit]
Description=Write rs-llmctl monthly audit evidence
Documentation=https://github.com/${repo}

[Service]
Type=oneshot
User=llmctl
Group=llmctl
Environment=LLMCTL_CONFIG=${config_file}
ExecStart=${bin_dir}/llmctl --config \${LLMCTL_CONFIG} audit report monthly --envelope --write
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=${state_dir} ${log_dir}
EOF
    run_root install -m 0644 "${audit_unit_tmp}" "${systemd_dir}/llmctl-monthly-audit.service"
    run_root install -m 0644 "${extract_dir}/packaging/systemd/llmctl-monthly-audit.timer" "${systemd_dir}/llmctl-monthly-audit.timer"
  fi
  run_root systemctl daemon-reload
  run_root systemctl reset-failed "${service_name}.service" >/dev/null 2>&1 || true
  audit_timer_installed=0
  if [ -f "${systemd_dir}/llmctl-monthly-audit.timer" ]; then
    audit_timer_installed=1
  fi
  if [ "${audit_timer_installed}" = "1" ] && [ "${enable_audit_timer}" = "1" ]; then
    run_root systemctl enable --now llmctl-monthly-audit.timer >/dev/null 2>&1 || warn "Installed monthly audit timer but could not enable it automatically"
  fi

  if [ "${start_service}" = "1" ]; then
    info "Enabling and starting ${service_name}.service"
    if run_root systemctl enable --now "${service_name}.service"; then
      ok "Started ${service_name}.service"
    else
      warn "Installed ${service_name}.service, but systemd could not start it automatically"
      warn "Check status with: sudo systemctl status ${service_name}.service"
    fi
  else
    ok "Installed ${service_name}.service without starting it"
    warn "Run first-run with a model and API key before starting the service:"
    warn "  sudo ${bin_dir}/llmctl --config ${config_file} first-run --apply --secret-output /root/llmctl-api-key.txt --starter-model-path /path/to/safetensors-model-dir"
    warn "Then start it with: sudo systemctl enable --now ${service_name}.service"
  fi
else
  case ":${PATH}:" in
    *":${bin_dir}:"*) ;;
    *) warn "Add ${bin_dir} to PATH before running llmctl." ;;
  esac
fi

printf '\n'
ok "rs-llmctl installed"
if [ "${install_systemd}" = "1" ]; then
  if [ "${start_service}" != "1" ]; then
    printf '  Next:    sudo %s/llmctl --config %s first-run --apply --secret-output /root/llmctl-api-key.txt --starter-model-path /path/to/safetensors-model-dir\n' "${bin_dir}" "${config_file}"
  fi
  printf '  Service: sudo systemctl status %s.service\n' "${service_name}"
  printf '  Start:   sudo systemctl enable --now %s.service\n' "${service_name}"
  if [ "${audit_timer_installed:-0}" = "1" ]; then
    if [ "${enable_audit_timer}" = "1" ]; then
      printf '  Audit:   sudo systemctl status llmctl-monthly-audit.timer\n'
    else
      printf '  Audit:   sudo systemctl enable --now llmctl-monthly-audit.timer\n'
    fi
  fi
  printf '  API:     http://127.0.0.1:8765/v1\n'
  printf '  Config:  %s\n' "${config_file}"
else
  printf '  Next: %s/llmctl init --profile local-dev\n' "${bin_dir}"
fi
