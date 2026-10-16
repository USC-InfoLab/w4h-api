# API Index

Structural map of `w4h-api` for agents and humans. **Index only** — not API docs.

- Regenerate: `npm run index:api`
- Generated: 2026-10-15
- Do not hand-edit the lists below; change code, then re-run the script.

## Orientation

| Area | Path |
|------|------|
| Entry | `app.js` |
| Auth | `auth.js` |
| DB helpers | `db.js`, `lib/db-init.js` |
| Routes | `routes/` |
| Lib | `lib/` |
| Schema | `data/schema.sql`, `data/projects_acl.sql` |
| CLI | `bin/w4h` |

## Mounts

| Mount | File |
|-------|------|
| `/users` | `routes/users.js` |
| `/datasets` | `routes/datasets.js` |
| `/docs` | `routes/docs.js` |
| `/config` | `routes/config.js` |
| `/platform` | `routes/platform.js` |
| `/projects` | `routes/projects.js` |
| `/teams` | `routes/teams.js` |
| `/contributions` | `routes/contributions.js` |
| `/packages` | `routes/contributions.js` |
| `/ai` | `routes/ai.js` |
| `/access-requests` | `routes/access-requests.js` |

Unmounted route modules: `routes/data.js`, `routes/subjects.js`

## Endpoints (168)

- **Root:** GET `/`
- **`/users`:** POST `/users/login` · POST `/users/send-code` · POST `/users/verify-code` · POST `/users/token-valid` · GET `/users/invite/:token` · POST `/users/invite/accept` · POST `/users/invite/resend` · POST `/users/invite/revoke` · GET `/users/me/session` · GET `/users/me/api-key` · POST `/users/me/api-key/reveal` · POST `/users/me/api-key` · DELETE `/users/me/api-key` · DELETE `/users/user/:email/api-key` · GET `/users/` · GET `/users/user/:email` · POST `/users/profile` · POST `/users/add` · POST `/users/set` · POST `/users/discontinue` · POST `/users/resume` · DELETE `/users/:email` · GET `/users/roles`
- **`/datasets`:** GET `/datasets/projects` · GET `/datasets/projects-admin` · GET `/datasets/projects-admin/:id` · POST `/datasets/projects-admin` · PUT `/datasets/projects-admin/:id` · GET `/datasets/tableinfo` · GET `/datasets/` · POST `/datasets/` · GET `/datasets/shares` · GET `/datasets/share-targets` · DELETE `/datasets/shares/:shareId` · GET `/datasets/:id/shares` · POST `/datasets/:id/shares` · DELETE `/datasets/:id` · PATCH `/datasets/:id` · POST `/datasets/:id/import/csv-batch` · POST `/datasets/:id/summary/refresh` · POST `/datasets/:id/contributions/:slug/sql` · GET `/datasets/:id/dependents` · GET `/datasets/:id/signals` · GET `/datasets/:id/analytics/cohort` · POST `/datasets/:id/analytics/cohort` · GET `/datasets/:id/analytics/catalog` · POST `/datasets/:id/analytics/series` · GET `/datasets/:id`
- **`/docs`:** GET `/docs/` · GET `/docs/search` · GET `/docs/cards/feature` · GET `/docs/export` · GET `/docs/refresh/preview` · POST `/docs/refresh` · GET `/docs/types` · POST `/docs/types` · GET `/docs/:slug` · PUT `/docs/:slug` · DELETE `/docs/:slug`
- **`/config`:** GET `/config/status` · GET `/config/admin-home` · GET `/config/team-home` · GET `/config/usage` · GET `/config/api-usage` · POST `/config/client-usage` · POST `/config/report-issue` · GET `/config/audit-log` · GET `/config/database-settings` · GET `/config/mailer-settings` · POST `/config/mailer-settings` · POST `/config/mailer-test` · GET `/config/app-settings` · POST `/config/app-settings` · POST `/config/init-database` · POST `/config/admin-setup` · GET `/config/test` · POST `/config/reset`
- **`/platform`:** GET `/platform/hub-principles`
- **`/projects`:** GET `/projects/` · POST `/projects/` · GET `/projects/collaboration-invites` · GET `/projects/:id/db-tokens` · DELETE `/projects/:id/db-tokens/purge-inactive` · DELETE `/projects/:id/db-tokens/:tokenId` · GET `/projects/:id/acl` · PUT `/projects/:id/acl` · GET `/projects/:id/linkable-datasets` · GET `/projects/:id` · PATCH `/projects/:id` · POST `/projects/:id/members` · PATCH `/projects/:id/members/:userId` · DELETE `/projects/:id/members/:userId` · PUT `/projects/:id/members` · PUT `/projects/:id/organizations` · PUT `/projects/:id/datasets` · POST `/projects/:id/datasets/:datasetId/link` · DELETE `/projects/:id/datasets/:datasetId` · GET `/projects/:id/invitable-teams` · POST `/projects/:id/teams` · POST `/projects/:id/teams/:teamId/accept` · DELETE `/projects/:id/teams/:teamId` · PUT `/projects/:id/docs` · GET `/projects/:id/pages` · POST `/projects/:id/pages` · GET `/projects/:id/pages/:slug` · PUT `/projects/:id/pages/:slug` · DELETE `/projects/:id/pages/:slug` · POST `/projects/:id/db-token` · GET `/projects/:id/reports` · POST `/projects/:id/reports` · GET `/projects/:id/reports/:reportId/content` · DELETE `/projects/:id/reports/:reportId`
- **`/teams`:** GET `/teams/` · POST `/teams/` · GET `/teams/:id` · PATCH `/teams/:id` · POST `/teams/:id/members` · PATCH `/teams/:id/members/:userId` · DELETE `/teams/:id/members/:userId`
- **`/contributions`:** GET `/contributions/` · GET `/contributions/runs/:id` · POST `/contributions/inspect` · POST `/contributions/check-updates` · POST `/contributions/:slug/update-version` · POST `/contributions/:slug/uploads` · GET `/contributions/uploads/:uploadId` · PUT `/contributions/uploads/:uploadId/files/:name` · DELETE `/contributions/uploads/:uploadId/files/:name` · DELETE `/contributions/uploads/:uploadId` · GET `/contributions/:slug` · POST `/contributions/` · POST `/contributions/:slug/run` · PATCH `/contributions/:slug` · DELETE `/contributions/:slug`
- **`/packages`:** GET `/packages/` · GET `/packages/runs/:id` · POST `/packages/inspect` · POST `/packages/check-updates` · POST `/packages/:slug/update-version` · POST `/packages/:slug/uploads` · GET `/packages/uploads/:uploadId` · PUT `/packages/uploads/:uploadId/files/:name` · DELETE `/packages/uploads/:uploadId/files/:name` · DELETE `/packages/uploads/:uploadId` · GET `/packages/:slug` · POST `/packages/` · POST `/packages/:slug/run` · PATCH `/packages/:slug` · DELETE `/packages/:slug`
- **`/ai`:** GET `/ai/status` · GET `/ai/settings/platform` · PATCH `/ai/settings/platform` · GET `/ai/settings/team/:teamId` · PATCH `/ai/settings/team/:teamId` · GET `/ai/conversations` · POST `/ai/conversations` · GET `/ai/conversations/:id` · PATCH `/ai/conversations/:id` · DELETE `/ai/conversations/:id` · POST `/ai/chat`
- **`/access-requests`:** POST `/access-requests/` · POST `/access-requests/verify` · GET `/access-requests/count` · GET `/access-requests/` · POST `/access-requests/:id/fulfill` · POST `/access-requests/:id/decline` · POST `/access-requests/:id/spam`

## Lib (42)

- `lib/`: `access-requests.js`, `cohort-analytics.js`, `config.js`, `contribution-runner.js`, `contribution-uploads.js`, `contributions-manifest.js`, `contributions-versions.js`, `contributions.js`, `dataset-metadata-summary.js`, `dataset-mets-summary.js`, `dataset-summary-refresh.js`, `dataset-wear-summary.js`, `db-init.js`, `docs-mermaid-render.js`, `env-path.js`, `ingest-csv.js`, `invitations.js`, `mailer.js`, `reports.js`, `seed-docs-fixtures.js`, `sql-contribution-engine.js`, `sql-ident.js`, `sql-ident.test.js`, `viz-display.js`, `viz-queries.js`
- `lib/ai/`: `claude.js`, `conversations.js`, `keys.js`, `phi.js`, `prompt.js`
- `lib/middleware/`: `projectDbTokenAuth.js`
- `lib/utils/`: `apiKeyCrypto.js`, `apiKeys.js`, `audit.js`, `datasetProjectLinkColumn.js`, `defs.js`, `docs.js`, `md-paths.js`, `md-render.js`, `password-validator.js`, `usage.js`, `user.js`

## Schema tables (31)

From `CREATE TABLE` in `data/*.sql` (may lag live DB):

`w4h.access_requests` · `w4h.ai_conversations` · `w4h.ai_messages` · `w4h.audit_log` · `w4h.contribution_runs` · `w4h.contributions` · `w4h.dataset_shares` · `w4h.datasets` · `w4h.doc_types` · `w4h.docs` · `w4h.invitations` · `w4h.login_codes` · `w4h.platform_ai_settings` · `w4h.project_acl_rules` · `w4h.project_datasets` · `w4h.project_db_tokens` · `w4h.project_docs` · `w4h.project_members` · `w4h.project_organizations` · `w4h.project_pages` · `w4h.project_reports` · `w4h.project_teams` · `w4h.projects` · `w4h.survey_responses` · `w4h.tables` · `w4h.team_ai_settings` · `w4h.team_members` · `w4h.teams` · `w4h.usage_events` · `w4h.user_projects` · `w4h.users`
