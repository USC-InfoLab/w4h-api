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
        print("Set W4H_API_KEY to a key from POST /users/me/api-key (browser session)", file=sys.stderr)
        sys.exit(1)

    headers = {"X-W4H-API-Key": api_key}

    r = requests.get(f"{base}/datasets", headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    n = len(data) if isinstance(data, list) else 0
    print("GET /datasets OK, rows:", n)


if __name__ == "__main__":
    main()
