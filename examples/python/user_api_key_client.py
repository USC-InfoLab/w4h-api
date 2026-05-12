#!/usr/bin/env python3

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
    api_key = os.environ.get("W4H_API_KEY", "").strip()
    if not api_key:
        print("Set W4H_API_KEY (from POST /users/me/api-key in the portal)", file=sys.stderr)
        sys.exit(1)

    headers = {"x-w4h-api-key": api_key}

    r1 = requests.get(f"{base}/projects", headers=headers, timeout=30)
    r1.raise_for_status()
    projects = r1.json()
    print("Projects:", len(projects) if isinstance(projects, list) else projects)

    r2 = requests.get(f"{base}/datasets", headers=headers, timeout=60)
    r2.raise_for_status()
    data = r2.json()
    n = len(data) if isinstance(data, list) else "?"
    print("Datasets:", n)


if __name__ == "__main__":
    main()
