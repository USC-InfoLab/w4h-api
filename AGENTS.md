# AGENTS.md — W4H API

Agent instructions for the Wearables for Health (W4H) multi-repo workspace.

## Workspace map

| Repo | Role |
|------|------|
| **w4h-docs** | Product docs, ADRs, MVP, architecture, guides |
| **w4h-app** | Vue 3 + Vite web UI |
| **w4h-api** (this repo) | Express API + PostgreSQL schema / ACL |
| **w4h-utils** | Scripts, CLI helpers, utility notebooks |
| **w4h-datasets** | Sample / synthetic CSV and JSON |
| **w4h-notebooks** | Index of legacy notebooks; new tools are one repo per contribution |
| **w4h-sample-fitbit-csv** / **-importer** | ADR-016 Fitbit CSV sample + importer (Contributions catalog) |
| **w4h-load-json** | First-party JSON ingest notebook (Contributions catalog) |
| **w4h-sql-compute-mets** | YAML-bound SQL METS job (ADR-023); sibling checkout next to this repo; notebook/CLI call signals + bind/execute APIs |
| **w4h-fitbit-wear-compliance** | Wear-compliance notebook + CLI (ICDE `ComplianceAnalyzer` port); its protocol is the one `lib/dataset-wear-summary.js` computes for the Analytics cohort tile |
| **w4h-demo-approximate-queries** | Notebook demo of the `fft_approximate` extension (connects to Postgres directly) |
| **w4h-hr-anomaly** | Example model contribution (MVP M4); reads `POST /datasets/:id/analytics/series`, writes a derived table |
| **w4h-docker** | Docker Compose packaging; `api/Dockerfile` and entrypoint live there |

Open these together via `w4h-repos.code-workspace` when changing cross-cutting behavior.

## Source of truth

- Requirements: sibling `w4h-docs/dev/mvp.md`
- Roles / ACL: `w4h-docs/dev/ADR-004-acl-roles.md`
- Dataset linking: `w4h-docs/dev/ADR-012-project-dataset-linking.md`
- Cross-org projects: `w4h-docs/dev/ADR-027-cross-org-project-collaboration.md` · platform access requests: `ADR-028-platform-access-requests.md` · charts and published reports: `ADR-029-visualization-component-and-reports.md`
- Target table model: `w4h-docs/dev/ADR-003-dataset-table-model.md` — physical tables + **`w4h.tables` mapping** (GeoMTS roles); do not treat in-schema `dataset_tables` as SoT
- GeoMTS roles: `w4h-docs/dev/ADR-008-geomts-data-model.md` (mapping, not forced `subject_id`/`ts`/`lat`/`lon`)

## Product model (locked)

- **Team** = tenancy. **Project** = primary access / work unit.
- Platform role Super admin is account-wide; team/project authority comes from memberships.
- **Team admin is `team_members.role = 'team_admin'` per team — never `users.role`** (R-09). `users.role` is `super_admin` or `project_user`; `team_admin` is accepted only as an *input alias* when creating/updating a user. Use `canManageTeamAsync` / `isTeamAdminOnTeam` / `isTeamAdminOfTeam` (membership reads), not role strings. A Super admin is in no team and **does not create or approve teams**: `POST /teams` rejects them. Teams are created by an Admin, including an approved one with `users.can_create_teams` and no team yet.
- **Access requests** (ADR-028): public `POST /access-requests` + `POST /access-requests/verify` (unauthenticated: USC-domain allowlist `ACCESS_REQUEST_ALLOWED_DOMAINS`, exact match, mailbox verification, honeypot, rate limits, no enumeration). The Super admin approves in `/access-requests` (Platform Admin); approval sends an Admin invitation and creates **no team**.
- Project membership: `owner` / `editor` / `viewer` (UI: Admin / User). A project has one owning team and may have **collaborating teams** (`w4h.project_teams`, ADR-027): their people join as `editor` only; a project owner is always from the owning team.
- **Dataset data follows project membership** (ADR-027): reads go through `assertCanViewDataset` (member of a project that links the dataset, or the owning team's Admin, or Super admin). Team membership or a team-wide share alone grants nothing. Ingest (`assertCanIngestDataset`) is the owning team's Admin, or a project owner **of the dataset's own team**. A team member who is not on a project sees its overview only (`capabilities.is_participant`).
- W4H does not register project-owned notebooks. Browse **Contributions** (`/contributions`, ADR-018).
- Catalog schema: `w4h.contributions` (publisher, kind, language, repo, **version pin**). Do not reintroduce `project_notebooks`.
- SQL analytics (sibling `w4h-docs/dev/ADR-023-sql-analytics-contributions.md`): bind Super-admin-vetted YAML templates via `lib/sql-contribution-engine.js`; **never** `POST` arbitrary SQL. In-app: `POST /contributions/:slug/run { dataset_id }` (same entry as importers); notebooks: `POST /datasets/:id/contributions/:slug/sql` (sync). Package lives in sibling `w4h-sql-compute-mets`. `kind: sql` is a job; `postgres_extension` is `CREATE EXTENSION`. Do not add new `lib/dataset-*.js` modules that own a formula.
- **Charts and reports** (ADR-029): `GET /datasets/:id/analytics/catalog` and `POST /datasets/:id/analytics/series` (kinds `series`, `distribution`, `coverage`, `scatter`, `events`) in `lib/viz-queries.js`. They read the mapping (`w4h.tables`) + `information_schema`, never a vendor name; identifiers are validated, every value is a bind parameter, queries run `READ ONLY` with a timeout, results are capped, never raw rows. Contributions may declare a `display:` block (`lib/viz-display.js`) from the fixed chart set — they ship no UI code. Published notebook **reports** (`w4h.project_reports`, `lib/reports.js`) are unvetted HTML: always served with the injected CSP, only to project members, only for datasets the publisher's own team owns, deleted when a source dataset is unlinked. Never add `allow-same-origin` or relax the CSP.
- **Timestamps leave SQL as text.** A `timestamp without time zone` or a `DATE` read by node-pg becomes a JS `Date` in the *server's* timezone and shifts days (this caused real bugs). Select dates/times as text (`::date::text`, `to_char(... 'YYYY-MM-DD"T"HH24:MI:SS"Z"')`) or via `to_timestamp(EXTRACT(EPOCH …))`, and compare with strings.
- Cached cohort cards (ADR-019): `datasets.summary.mets_summary` and `.wear_compliance` are recomputed by `POST /datasets/:id/summary/refresh` (Admin; optional `wear_protocol`) and after every successful in-app contribution run (`lib/dataset-summary-refresh.js`). `lib/dataset-wear-summary.js` ports the `w4h-fitbit-wear-compliance` protocol and is an **interim exception** to the no-formula-in-`lib/dataset-*.js` rule above (like `dataset-mets-summary.js`): keep it bounded and in step with that repo's defaults, and move it into a `kind: sql` manifest once the engine supports optional requires and overridable scalars.
- Cohort comparison (ADR-021): `GET`/`POST /datasets/:id/analytics/cohort` in `lib/cohort-analytics.js` aggregates mapped HR/calories/METs. It does **not** own the MET formula. Filter-based subject IDs only — no free SQL, no attribute matching until demographics exist.
- Never log or return plaintext API keys except intentional step-up reveal flows; prefer encrypted wrap + audit.

## This repository

- Stack: Node (ESM), Express, `pg`. Entry: `app.js`. Routes: `routes/`. Helpers: `lib/`, `db.js`.
- Schema: `data/schema.sql` (and ACL-related SQL under `data/`). Boot/`ensureSchema` should match what routes assume — add DDL when introducing tables (e.g. `w4h.tables`).
- Env: `.env.example` → `.env`. Unset `DB_DATABASE` to re-run initial setup.
- Scripts: `npm start`; CLI via `bin/w4h`; `node bin/w4h seed-docs` creates a separate **`w4h-sample`** database (does not write `DB_DATABASE`). Serve it with `npm run start:sample`. See sibling `w4h-docs/dev/docs-fixtures.md`. `npm run index:api` regenerates [`API_INDEX.md`](./API_INDEX.md).
- Structural API map: [`API_INDEX.md`](./API_INDEX.md) (generated; mounts, endpoints, lib, schema tables — not feature docs). Re-run when routes or schema DDL change. Do not regenerate on every start.
- Env additions: `ACCESS_REQUEST_ALLOWED_DOMAINS`, `ACCESS_REQUEST_MAX_PER_IP` / `_PER_EMAIL`, `REPORT_MAX_BYTES` (see `.env.example`). The request form needs SMTP + `APP_URL`.
- No automated test suite yet. Verify API work against a **scratch database** (`CREATE DATABASE …`, `DB_DATABASE=… PORT=… node app.js`, mint a JWT with `TOKEN_SECRET`, assert over HTTP), drop it afterwards, and stop test servers **by PID** (`lsof -iTCP:<port>`), never `pkill -f "node app.js"` — a developer's own API may be running.
- Prefer catalog/`w4h.tables` (mapping SoT) + `information_schema` over legacy `getQueryMap` paths when adding dataset features.
- Dataset `summary` / `summary_updated_at`: cache estimates; do not block list endpoints with live full-table `COUNT(*)` scans.

## Cross-repo changes

New/changed response fields should be consumed in **w4h-app** and reflected in **w4h-docs** when they are product-visible.
