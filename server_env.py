"""Server credentials loader for Extraction Ops remote scripts.

Priority:
1. Environment variables (GPU_SERVER_*) — for remote/server usage.
2. `.env` file next to this module — for local usage.

No third-party dependencies. `.env` is gitignored; commit `.env.example`
instead as a template.

Usage:
    from server_env import SERVER_HOST, SERVER_PORT, SERVER_USER, SERVER_PWD
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULTS = {
    "GPU_SERVER_HOST": "connect.nmb2.seetacloud.com",
    "GPU_SERVER_PORT": "14970",
    "GPU_SERVER_USER": "root",
}


def _load_dotenv() -> dict[str, str]:
    """Parse KEY=VALUE pairs from the `.env` file next to this module."""
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


_DOTENV = _load_dotenv()


def _get(key: str) -> str:
    # 1. process env (highest priority, e.g. exported on the remote box)
    if key in os.environ:
        return os.environ[key]
    # 2. .env file (local dev)
    if key in _DOTENV:
        return _DOTENV[key]
    # 3. non-secret defaults
    if key in _DEFAULTS:
        return _DEFAULTS[key]
    raise RuntimeError(
        f"Missing {key}: set it in .env (local) or export it (remote). "
        f"See .env.example for the template."
    )


SERVER_HOST = _get("GPU_SERVER_HOST")
SERVER_PORT = int(_get("GPU_SERVER_PORT"))
SERVER_USER = _get("GPU_SERVER_USER")
SERVER_PWD = _get("GPU_SERVER_PWD")
