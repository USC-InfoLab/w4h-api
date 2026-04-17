# Ansel Tasks: Token Auth, ACL Tables, Project Services

This document captures the implementation blueprint and practical guidance for:

1. token-based database authentication from Python,
2. database schema for ACL/projects,
3. backend services that map project data to the frontend.

---

## 1) Token-Based Auth for Database Access in Python

### Goal

Avoid storing raw database credentials inside notebooks/scripts. Instead:

1. user authenticates to W4H backend (normal JWT login),
2. script asks backend for a short-lived project DB token,
3. backend validates membership/permissions,
4. backend returns a short-lived token,
5. Python code uses token for subsequent API-mediated DB work (or brokered DB creds in a later phase).

### Current implementation in this repository

- New endpoint: `POST /projects/:id/db-token`
- Requires app JWT auth header.
- Checks user can read the project.
- Returns a short-lived token payload and stores only a hash in `w4h.project_db_tokens`.

### Why this is safer

- Notebook does not hardcode DB user/password.
- Tokens expire quickly (default 2 days, max 7 days).
- Tokens can be revoked centrally.

### Python example (requests + psycopg placeholder)

```python
import requests

API_BASE = "http://localhost:2026"
APP_JWT = "..."   # from W4H login
PROJECT_ID = "..."

res = requests.post(
    f"{API_BASE}/projects/{PROJECT_ID}/db-token",
    headers={"authorization": APP_JWT},
    json={"days": 2},
    timeout=30,
)
res.raise_for_status()
db_token_payload = res.json()
print("token expires:", db_token_payload["expires_at"])

# Phase 1 (current): token used for backend API calls
# Phase 2 (future): backend exchanges token for short-lived pg credentials
```

### Recommended next step (research deliverable)

Decide one of:

- **User-scoped token** (better per-person audit granularity),
- **Project-scoped token** (easier automation/shared jobs),

and add revocation + rotation policy (e.g. 48h default, forced refresh, admin revoke endpoint).

---

## 2) Database Table Structure for ACL and Projects

Implemented SQL file:

- `w4h-api/data/projects_acl.sql`

### Tables added

- `w4h.projects`: project identity and IRB metadata
- `w4h.project_members`: per-project users with role (`owner/editor/viewer/admin`)
- `w4h.project_datasets`: dataset links per project with access type (`read/read_write`)
- `w4h.project_notebooks`: notebook/repo metadata only (no notebook file content)
- `w4h.project_docs`: project-to-doc slug links
- `w4h.project_acl_rules`: fine-grained ACL rules by subject/resource/permission/effect
- `w4h.project_db_tokens`: hashed short-lived token records

### Design rationale

- Keeps admin-managed datasets and project collaboration separate.
- Supports "project shell" concept: linked datasets + members + notebook metadata + docs.
- Adds a path for IRB-aware, auditable access decisions.

---

## 3) Backend Services for Project Information Mapping to Frontend

Implemented route:

- `w4h-api/routes/projects.js`
- registered in `w4h-api/app.js` as `app.use('/projects', projects)`.

### Endpoints

- `GET /projects` – list projects visible to current user
- `GET /projects/:id` – aggregate payload for project page
- `POST /projects` – create project and auto-add creator as owner
- `PATCH /projects/:id` – update project metadata
- `PUT /projects/:id/members` – replace project members
- `PUT /projects/:id/datasets` – replace linked datasets
- `PUT /projects/:id/notebooks` – replace notebook metadata list
- `PUT /projects/:id/docs` – replace linked docs
- `POST /projects/:id/db-token` – issue short-lived project DB token

### Frontend mapping target

`GET /projects/:id` returns:

- `project`: identity + status + visibility + IRB
- `members`: collaborators + roles
- `datasets`: linked datasets + study context
- `notebooks`: metadata for repo/path/launchers
- `docs`: linked documentation pages
- `organizations`: currently derived from linked study records

This supports the project-hub UI direction directly.

---

## Open Questions for Team Review

1. Token scope: user vs project?
2. Final ACL precedence: explicit deny wins vs role allow?
3. Should notebook launchers be global config or per-project?
4. Should dataset write access be explicit ACL only, or tied to project role?
5. Should project docs link by slug only, or by doc UUID?

---

## Minimal rollout order

1. Run SQL migration (`projects_acl.sql`) in dev DB.
2. Verify `/projects` endpoints with authenticated requests.
3. Point project frontend to `GET /projects/:id` instead of local storage.
4. Add admin screens for project membership/dataset linking.
5. Add token revoke/list endpoints and audit log events.
