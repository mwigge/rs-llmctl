# AI Developer Workflows

`rs-llmctl` is useful as a local and server-side base for code assistance when
the caller wants private material to stay under explicit control.

## Local Code And Document Search

Use `/v1/local/search` when an assistant already has a set of local snippets,
files, tickets, or runbook excerpts and needs ranked context:

```bash
curl -sS \
  -H "Authorization: Bearer $LLMCTL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"worker readiness","documents":[{"id":"ops","title":"Operations","content":"Worker readiness probes and restart backoff"}]}' \
  http://127.0.0.1:8765/v1/local/search
```

The endpoint does not crawl the host filesystem. The caller chooses which
documents are sent, which keeps code, notes, and internal material scoped to the
current workflow.

## Local Recommendations

Use `/v1/local/recommendations` when the assistant needs a short recommendation
set grounded in the same caller-provided local material:

```bash
curl -sS \
  -H "Authorization: Bearer $LLMCTL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"task":"review local model operations","documents":[{"id":"ops","title":"Operations","content":"hot swap readiness and audit reporting"}]}' \
  http://127.0.0.1:8765/v1/local/recommendations
```

The response includes ranked hits plus recommendation metadata that can be fed
back into an OpenAI-compatible chat completion call.

## Model Calls

Use `/v1/chat/completions` and `/v1/embeddings` for OpenAI-compatible model
access. Requests are audited, quota-checked, and tied to request IDs so a code
assistant can be reviewed after the fact without exposing prompts in telemetry
attributes.
