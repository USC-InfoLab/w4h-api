# Ansel Tasks: Token Auth, ACL Tables, Project Services (deliverable index)

This file ties together the three assigned tasks with **where to look in the repo** and how **`GET /projects/:id`** maps to the Vue project hub.

---

## Task 1 — Research: token-based authentication for database access in Python

**Delivered as documentation + example code**

| Resource | Purpose |
|----------|---------|
| [PYTHON_PROJECT_DB_TOKEN.md](./PYTHON_PROJECT_DB_TOKEN.md) | Patterns: JWT → mint token → `X-W4H-Project-DB-Token` on `GET /projects/:id`; hygiene; future Postgres broker |
| [examples/python/project_db_token_client.py](../examples/python/project_db_token_client.py) | Runnable `requests` example |

**Runtime behavior**

- `POST /projects/:id/db-token` — requires **JWT**; returns opaque token; stores **SHA-256 hash** only (`w4h.project_db_tokens`).
- `GET /projects/:id` — accepts **either** JWT **or** `X-W4H-Project-DB-Token` (see `lib/middleware/projectDbTokenAuth.js`).
- Mutations (`PATCH`, `PUT`, `POST` other than db-token) require **JWT** only.

---

## Task 2 — Design: database table structure for ACL and projects

**Delivered as SQL**

| File | Contents |
|------|----------|
| [data/projects_acl.sql](../data/projects_acl.sql) | `w4h.projects`, `project_members`, `project_datasets`, `project_notebooks`, `project_docs`, `project_acl_rules`, `project_db_tokens` + `COMMENT ON` for metadata |

Apply after core schema (`data/schema.sql`). Safe to re-run (`IF NOT EXISTS`).

---

## Task 3 — Implement: backend services + mapping to frontend

**Backend** — [routes/projects.js](../routes/projects.js)

| Method | Path | Notes |
|--------|------|--------|
| GET | `/projects` | Projects visible to user (membership; admins see all) |
| POST | `/projects` | Create; creator becomes **owner** |
| GET | `/projects/:id` | **JWT or project DB token**; aggregate payload below |
| PATCH | `/projects/:id` | Metadata; managers |
| PUT | `/projects/:id/members` | Replace roster; body `{ members: [{ user_id \| email, role }] }`; **≥1 owner** |
| PUT | `/projects/:id/datasets` | Replace links |
| PUT | `/projects/:id/notebooks` | Replace notebook metadata |
| PUT | `/projects/:id/docs` | Replace doc slugs (must exist in `w4h.docs`) |
| POST | `/projects/:id/db-token` | Mint token (JWT only) |
| GET | `/projects/:id/db-tokens` | List rows (managers) |
| DELETE | `/projects/:id/db-tokens/:tokenId` | Revoke (managers) |
| GET | `/projects/:id/acl` | List `project_acl_rules` (managers) |

**`GET /projects/:id` JSON → frontend `ProjectWorkspaceView`**

| API path | UI usage |
|----------|----------|
| `project` | Title, status, visibility, IRB, description |
| `members[]` | Members sidebar (`user_id`, `name`, `email`, `role`, `user_role`) |
| `datasets[]` | Linked datasets → `linkedDatasetIds` + detail from `GET /datasets` |
| `notebooks[]` | Notebook cards (`repo_url`, `repo_ref`, `notebook_path`, `launchers`, …) |
| `docs[]` | Doc links (`slug`, `title`, `updated_at`) |
| `organizations[]` | Collaborating orgs (from linked studies) |
| `capabilities.can_manage` | Show edit members, dataset linking, revoke tokens |
| `capabilities.can_issue_db_token` | Show “Issue project token” |
| `capabilities.auth_via_project_token` | Hide mutating actions (read-only session) |

Launcher URLs (Jupyter/RStudio/VSCode) remain **client-side preferences** in local storage until a project-level column is added.

---

## Open questions (team)

1. ACL precedence: explicit deny vs role allow.
2. Postgres broker: temporary role per token vs API-only SQL.
3. Persist launcher base URLs on `w4h.projects` (JSONB) vs global config.

---

## Rollout checklist

1. Run `projects_acl.sql` on dev DB.
2. Smoke-test `/projects` with JWT.
3. Run `examples/python/project_db_token_client.py` against dev API.
4. Use project hub UI: create project → open → link datasets → issue token.
