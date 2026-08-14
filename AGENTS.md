# AGENTS.md — W4H API

Agent instructions for the Wearables for Health (W4H) multi-repo workspace.

## Workspace map

| Repo | Role |
|------|------|
| **w4h-docs** | Product docs, ADRs, MVP, architecture, guides |
| **w4h-app** | Vue 3 + Vite web UI |
| **w4h-api** (this repo) | Express API + PostgreSQL schema / ACL |
| **w4h-utils** | Scripts, CLI helpers, utility notebooks |
| **w4h-datasets** | Sample / synthetic CSV datasets |
| **w4h-notebooks** | Example Jupyter workflows |

Open these together via `w4h-repos.code-workspace` when changing cross-cutting behavior.

## Source of truth

- Requirements: sibling `w4h-docs/dev/mvp.md`
- Roles / ACL: `w4h-docs/dev/ADR-004-acl-roles.md`
- Dataset linking: `w4h-docs/dev/ADR-012-project-dataset-linking.md`
- Target table model: `w4h-docs/dev/ADR-003-dataset-table-model.md` (may ahead of current schema)

## Product model (locked)

- **Team** = tenancy. **Project** = primary access / work unit.
- Platform role Super admin is account-wide; team/project authority comes from memberships.
- Project membership: `owner` / `editor` / `viewer` (UI: Admin / User).
- Users are discontinued, not hard-deleted.
- Never log or return plaintext API keys except intentional step-up reveal flows; prefer encrypted wrap + audit.

## This repository

- Stack: Node (ESM), Express, `pg`. Entry: `app.js`. Routes: `routes/`. Helpers: `lib/`, `db.js`.
- Schema: `data/schema.sql` (and ACL-related SQL under `data/`). Boot/`ensureSchema` should match what routes assume — add DDL when introducing tables (e.g. `w4h.tables`).
- Env: `.env.example` → `.env`. Unset `DB_DATABASE` to re-run initial setup.
- Scripts: `npm start`; CLI via `bin/w4h`; `npm run index:api` regenerates [`API_INDEX.md`](./API_INDEX.md).
- Structural API map: [`API_INDEX.md`](./API_INDEX.md) (generated; mounts, endpoints, lib, schema tables — not feature docs). Re-run when routes or schema DDL change. Do not regenerate on every start.
- Prefer catalog/`w4h.tables` + `information_schema` over legacy `getQueryMap` paths when adding dataset features.
- Dataset `summary` / `summary_updated_at`: cache estimates; do not block list endpoints with live full-table `COUNT(*)` scans.

## Cross-repo changes

New/changed response fields should be consumed in **w4h-app** and reflected in **w4h-docs** when they are product-visible.
