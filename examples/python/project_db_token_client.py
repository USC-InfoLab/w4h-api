#!/usr/bin/env python3
"""
Example: mint a project DB token with a W4H JWT, then fetch project overview
using X-W4H-Project-DB-Token (no password in notebook).

Usage:
  export W4H_API_BASE=http://localhost:2026
  export W4H_JWT='<paste token from browser localStorage w4h.user.token>'
  export W4H_PROJECT_ID='<uuid>'
  python project_db_token_client.py

After this works, reuse the same X-W4H-Project-DB-Token on read-only project routes; notebook rows appear in GET /projects/<id> under "notebooks".

Requires: pip install requests
"""

from __future__ import annotations

import os
import sys

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base = os.environ.get("W4H_API_BASE", "http://localhost:2026").rstrip("/")
    jwt = os.environ.get("W4H_JWT", "").strip()
    project_id = os.environ.get("W4H_PROJECT_ID", "").strip()
    if not jwt or not project_id:
        print("Set W4H_JWT and W4H_PROJECT_ID", file=sys.stderr)
        sys.exit(1)

    r1 = requests.post(
        f"{base}/projects/{project_id}/db-token",
        headers={"authorization": jwt},
        json={"days": 2},
        timeout=30,
    )
    r1.raise_for_status()
    mint = r1.json()
    token = mint["token"]
    print("Issued project token, expires_at:", mint.get("expires_at"))

    r2 = requests.get(
        f"{base}/projects/{project_id}",
        headers={"x-w4h-project-db-token": token},
        timeout=30,
    )
    r2.raise_for_status()
    overview = r2.json()
    name = overview.get("project", {}).get("name", "?")
    print("Project overview OK:", name)
    caps = overview.get("capabilities", {})
    print("capabilities:", caps)


if __name__ == "__main__":
    main()
