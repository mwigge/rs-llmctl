# Data Contracts

`rs-llmctl` treats operational data as a small data fabric. The CLI exposes
schema-versioned contracts and domain exports so security, observability,
usage, finops, model lifecycle, drift, and audit data can move into external
reporting systems without guessing field names.

## Contracts

```bash
llmctl data contracts
llmctl data contracts --dataset security
llmctl data contracts --dataset observability
llmctl data contracts --dataset user
llmctl data contracts --dataset finops
```

Every contract includes:

- `schema_version`, currently `1`;
- a stable dataset name;
- required and nullable fields;
- a JSON Schema view;
- an `arrow-json-schema` view for Arrow-oriented pipelines.

The current datasets are `security`, `observability`, `usage`, `user`,
`finops`, `models`, `drift`, and `audit`.

## Exports

```bash
llmctl data export --hours 24 --dataset security --format json
llmctl data export --hours 24 --dataset observability --format jsonl
llmctl data export --hours 24 --dataset finops --format arrow-json
llmctl data export --hours 24 --dataset drift --format arrow-json
llmctl data export --hours 24 --dataset finops --format arrow-ipc --output finops.arrow
llmctl data export --hours 24 --dataset finops --format parquet --output finops.parquet
```

Formats:

- `json`: one JSON object with metadata and `rows`;
- `jsonl`: one JSON object with pre-rendered JSONL `lines`;
- `arrow-json`: rows plus the dataset `arrow_schema`.
- `arrow-ipc`: a native Arrow IPC file written to `--output`;
- `parquet`: a native Parquet file written to `--output`.

Binary exports require a concrete `--dataset` so the writer can apply a stable
schema. `--dataset all` remains JSON-only because it is an envelope-style
object, not one rectangular table.

## Canonical Evidence

Report envelopes are still the canonical evidence mechanism for signed audit
and compliance artifacts:

```bash
llmctl audit report monthly --envelope
llmctl audit report request <request-id> --envelope
llmctl data export --envelope
llmctl data verify-envelope ./data-export-envelope.json
```

Envelope export currently wraps the full canonical JSON data export. Domain
filtered exports are meant for data pipelines and dashboards.
