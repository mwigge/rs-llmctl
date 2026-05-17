# Changelog

All notable release-facing changes are recorded here. Keep entries focused on
operator behavior, packaging contents, service lifecycle, and verification.

## Unreleased

- Native-first packaging: the default archive publishes the single `llmctl`
  runtime binary, README, changelog, license, and `llmctld.service` systemd
  template.
- Stable service name: Linux installs continue to use `llmctld.service` for
  runbooks and monitoring, while `ExecStart` runs `llmctl --config
  /etc/rs-llmctl/config.toml server run`.
- Install validation: `packaging/validate-install.sh` remains passive and offline,
  checking the installed binary, config, state/log directories,
  service unit, CLI readiness commands, and `systemd-analyze verify` when
  available.
- Release integrity: `packaging/generate-checksums.sh` writes
  `dist/rs-llmctl-<os>-<arch>.tar.gz` and `dist/SHA256SUMS`; use
  `packaging/sign-release.sh dist` for optional `cosign` or `minisign`
  signatures.
