# Token-based database access from Python (research notes)

This document explains how W4H issues **project DB tokens**, how to use them from Python, and how they fit into a HIPAA-conscious architecture where **raw Postgres passwords never live in notebooks**.

## What you get today

1. Log in to the web app and obtain the normal **JWT** (`Authorization` header on API calls).
2. Call `POST /projects/:projectId/db-token` with that JWT.
3. The API returns a **short-lived opaque token** (plus `expires_at`). Only a **SHA-256 hash** of that token is stored in `w4h.project_db_tokens`.
4. For **read-only project overview** data, call `GET /projects/:projectId` with header **`X-W4H-Project-DB-Token: <token>`** instead of the JWT.

This lets automation (notebooks, batch jobs) refresh project metadata **without embedding the user password** and without keeping the long-lived JWT in every cell.

## What is intentionally not in scope yet

- Direct `psycopg2.connect(user=..., password=...)` using this token. Postgres does not know about W4H tokens; bridging requires a **broker** (vault, temporary role, or API-mediated queries).
- Row-level security policies in Postgres keyed off token identity — possible future work once the broker exists.

## Recommended Python pattern

### 1) Mint a token (JWT session)

Use `requests` or `httpx`. Keep the JWT out of source control: read from environment (e.g. `W4H_JWT`).

```python
import os
import requests

API = os.environ["W4H_API_BASE"].rstrip("/")  # e.g. http://localhost:2026
JWT = os.environ["W4H_JWT"]
PROJECT_ID = os.environ["W4H_PROJECT_ID"]

r = requests.post(
    f"{API}/projects/{PROJECT_ID}/db-token",
    headers={"authorization": JWT},
    json={"days": 2},
    timeout=30,
)
r.raise_for_status()
payload = r.json()
token = payload["token"]
expires_at = payload["expires_at"]
```

### 2) Call APIs with the project token

Pass the opaque token on **read** routes that support it (currently `GET /projects/:id`):

```python
r2 = requests.get(
    f"{API}/projects/{PROJECT_ID}",
    headers={"x-w4h-project-db-token": token},
    timeout=30,
)
r2.raise_for_status()
overview = r2.json()
# overview includes: project, members, datasets, notebooks, docs, organizations, acl_rules, capabilities
```

### 3) Operational hygiene

| Practice | Why |
|----------|-----|
| Export token to env for a single run | Limits exposure in process listings and notebooks |
| Short TTL (`days` 1–2) | Reduces blast radius if leaked |
| Never commit tokens or JWTs | Treat like passwords |
| Revoke via API when job ends | `DELETE /projects/:id/db-tokens/:tokenId` (managers) |

### 4) Using `httpx` with a small client class

```python
class W4HProjectClient:
    def __init__(self, base_url: str, project_id: str, jwt: str):
        self.base = base_url.rstrip("/")
        self.project_id = project_id
        self._jwt = jwt
        self._project_token: str | None = None

    def mint_project_token(self, days: int = 2) -> dict:
        import httpx
        r = httpx.post(
            f"{self.base}/projects/{self.project_id}/db-token",
            headers={"authorization": self._jwt},
            json={"days": days},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        self._project_token = data["token"]
        return data

    def get_project_overview(self) -> dict:
        import httpx
        headers = {}
        if self._project_token:
            headers["x-w4h-project-db-token"] = self._project_token
        else:
            headers["authorization"] = self._jwt
        r = httpx.get(
            f"{self.base}/projects/{self.project_id}",
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
```

## Future: connecting to Postgres from Python

Two common patterns:

1. **API-mediated SQL** — Notebook calls only W4H HTTP APIs; the API enforces ACL and audit. No direct DB string in Python.
2. **Credential broker** — Short-lived Postgres role/password issued after validating `X-W4H-Project-DB-Token`, then `psycopg2` connects with those credentials until expiry.

Both require server-side work beyond this milestone; the token table and hash storage are the audit-friendly foundation.

## See also

- `w4h-api/examples/python/project_db_token_client.py` — runnable outline.
- `w4h-api/docs/ANSEL_TASKS_PROJECTS_ACL_TOKEN_PLAN.md` — schema and endpoint index.
