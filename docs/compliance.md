# Compliance Evidence

`rs-llmctl` gives operators evidence they can attach to security reviews,
change records, incident records, and monthly service reports.

## Operator Commands

```bash
llmctl --config /etc/rs-llmctl/config.toml compliance evidence
llmctl --config /etc/rs-llmctl/config.toml compliance cra-article14
llmctl --config /etc/rs-llmctl/config.toml compliance pci-dss
llmctl --config /etc/rs-llmctl/config.toml compliance release-checklist
```

The output is JSON by default and can be wrapped with the existing audit/data
envelope commands when it needs payload hashing.

## CRA Article 14 Process

`rs-llmctl` treats EU Cyber Resilience Act Article 14 as an active production
control for all releases and operations. The built-in evidence command records
the operating timelines:

- early warning within 24 hours;
- vulnerability notification within 72 hours;
- final vulnerability report within 14 days after mitigation;
- severe incident notification without undue delay.

The operating workflow is:

1. Classify the vulnerability or severe incident.
2. Open an incident record with impacted versions, systems, users, mitigations,
   and current status.
3. Generate `compliance evidence`, `security audit-config`, `data export
   --envelope`, and the relevant `audit report`.
4. Attach SBOM, checksums, signatures, git commit, signed tag, and CI run URL.
5. Submit the regulatory notification in the required window.
6. Close with the final report after mitigation and verification.

## PCI DSS Evidence Posture

The current posture is PCI DSS v4.0.1-aligned rather than a substitute for a
formal assessor report. Evidence is grouped around access control, audit
logging, vulnerability management, secure configuration, and monitoring.

Monthly evidence should include:

- `audit report monthly --envelope`;
- `usage chargeback`;
- `quota report`;
- `data export --envelope`;
- `security audit-config`;
- `compliance evidence`;
- TLS termination or mTLS evidence from `security audit-config`.
- SBOM, checksums, and release signature for deployed binaries.

## SBOM, Provenance, And Signing

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo audit
packaging/generate-sbom.sh dist
packaging/generate-checksums.sh dist
packaging/sign-release.sh dist
```

`packaging/generate-sbom.sh` prefers CycloneDX through `cargo-cyclonedx` and
falls back to `cargo metadata` so air-gapped environments still get dependency
evidence. `packaging/sign-release.sh` supports `cosign` or `minisign`.

## TLS

Production external bind must sit behind TLS termination or an mTLS-capable
service mesh/load balancer. Keep `security.require-auth = true`, pin allowed
CORS origins, and document the termination point in the deployment change
record. `security check` rejects production external bind when
`security.tls-termination.enabled`, `provider`, or `evidence` is missing.
