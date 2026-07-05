# Running Local Models Like Real Infrastructure

![rs-llmctl operations overview](images/rs-llmctl-operations-hero.png)

I like local models most when they feel uneventful. The model is there, the
endpoint is predictable, the machine still has breathing room, and the audit
record tells me what happened later.

That is the space `rs-llmctl` is aiming for. It is not a chat UI. It is a small
operations layer for serving local or private models to real clients.

## The Job

At a high level, `rs-llmctl` does four things:

1. It registers approved model files.
2. It plans CPU/GPU workers with a resource budget.
3. It exposes an OpenAI-compatible API.
4. It records audit, usage, quota, and observation data.

That combination matters because model serving quickly becomes more than "run
this binary with a model path." Once teams depend on the endpoint, you need
quotas, reporting, swap plans, and predictable failure behavior.

## Offline First Feels Better

![rs-llmctl model lifecycle](images/rs-llmctl-lifecycle.png)

The cleanest production path is offline-first. Put the model files in a bundle,
ship a manifest with hashes, import it, and review the server plan before
traffic moves.

That gives you a paper trail without making the process heavy. The operator can
point to the manifest, the SHA-256 values, the dry-run server plan, and the
retention/report envelope.

## GPU Choice Should Not Be A Surprise

Different hosts have different accelerators. Some are NVIDIA, some are
AMD/Vulkan, some are Apple Metal, and some should be CPU-only. `rs-llmctl`
keeps that choice in config and planning output instead of hiding it in a
one-off shell session.

The default resource budget is 80%. It is conservative on purpose. A model
server that leaves no room for the operating system or neighboring services is
not really production-ready.

## Hot Swap, Cold Swap, And Routing That Reflects Reality

Model swap support is useful only if routing reflects reality. The selected model must
go to the selected worker. Weighted, fallback, hot-swap, and cold-swap modes all
need visible plans and predictable behavior.

The daemon starts planned workers, the API routes by resolved model alias, and
usage/audit records keep the requested model and upstream model visible.

## What Operators Get Back

The most useful output is not always another dashboard. Sometimes it is a
simple envelope you can attach to a change record:

- which API key subject acted;
- which team it belonged to;
- which model was used;
- whether quota allowed it;
- how many tokens were reported;
- which request ID ties the records together.

That is the kind of boring evidence that makes an internal model service easier
to trust.

## Where This Is Going

The near-term shape is a practical standalone model operations tool: one daemon,
one CLI, OpenAI-compatible clients, local-first storage, and clear production
checks.

The longer-term shape is deeper enterprise delivery: stronger worker health
supervision, live Postgres storage, richer OTel exporters, and stronger policy
hooks. The important part is that these are product capabilities, not migration
stories. `rs-llmctl` stands on its own.
