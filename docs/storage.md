# Storage Notes

SQLite is the default runtime database. It keeps a single-node install simple
and works well for development, edge deployments, and small internal model
services.

## Runtime SQLite

Runtime SQLite stores:

- audit events;
- usage events;
- observation events;
- quota decisions;
- model inventory;
- schema migration records.

The schema includes indexes for request lookup, retention windows, quota checks,
and usage totals. Quota paths use composite indexes for actor/team and time
range scans.

## Postgres Planning

External database storage with Postgres is available as a planning surface for
enterprise review. The plan uses a redacted database URL and emits dialect-specific migration SQL with UUID,
TIMESTAMPTZ, JSONB, boolean, and numeric types where appropriate.

Runtime Postgres is not silently enabled. If a production config sets a
Postgres `database-url`, runtime startup fails fast until the live backend is
implemented and tested.

## Migration Hygiene

New schema changes should be forward-only and represented in the shared
migration statement list. Avoid destructive schema changes in the same step as
data movement.
