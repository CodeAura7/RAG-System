"""API keys live OUTSIDE the code. Lookup order, first hit wins:
  1. an environment variable,
  2. a local `.env` file next to config.toml (git-ignored, never shipped in the ZIP),
  3. the OS credential store (Windows Credential Manager) via the `keyring` library.

The `.env` file is read on demand and is never copied into os.environ, so keys are not
inherited by any child process this program starts."""
from __future__ import annotations

import os
from pathlib import Path

SERVICE = "hackerone-rag"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MissingKeyError(RuntimeError):
    pass


def _dotenv_path() -> Path:
    return PROJECT_ROOT / ".env"


def _read_dotenv() -> dict[str, str]:
    """Tiny .env reader (no extra dependency): KEY=value, optional `export `, quotes, # comments, BOM, CRLF."""
    try:
        text = _dotenv_path().read_text(encoding="utf-8-sig")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not sep or not name:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]                       # quoted: '=' and '#' inside are kept as they are
        else:
            value = value.split(" #", 1)[0].rstrip()  # unquoted: a ' #' starts a comment
        if value:
            values[name] = value
    return values


def _keyring():
    try:
        import keyring  # noqa: PLC0415 - optional at import time so tests/CLI help work without it
        return keyring
    except ImportError:  # pragma: no cover
        return None


def _from_keyring(name: str) -> str | None:
    kr = _keyring()
    if kr is None:
        return None
    try:
        return kr.get_password(SERVICE, name) or None
    except Exception:  # no usable keyring backend on this machine
        return None


def get_secret(name: str) -> str | None:
    return os.environ.get(name) or _read_dotenv().get(name) or _from_keyring(name)


def require_secret(name: str) -> str:
    value = get_secret(name)
    if not value:
        raise MissingKeyError(
            f"No API key found for {name}. Put a line  {name}=...  in the .env file next to config.toml "
            f"(see .env.example), or set the {name} environment variable, "
            f"or store it with:  python -m hrag keys set {name}\n"
            f"Keys are never written into the code or the ZIP."
        )
    return value


def set_secret(name: str, value: str) -> None:
    kr = _keyring()
    if kr is None:
        raise MissingKeyError("the 'keyring' package is not installed; use the .env file or an environment variable instead")
    try:
        kr.set_password(SERVICE, name, value)
    except Exception as exc:
        raise MissingKeyError(
            f"could not write to the OS credential store ({type(exc).__name__}). "
            f"Use the .env file or set the {name} environment variable instead."
        ) from exc


def secret_source(name: str) -> str:
    """'env', '.env', 'keyring' or 'missing' - never returns the value."""
    if os.environ.get(name):
        return "env"
    if _read_dotenv().get(name):
        return ".env"
    return "keyring" if _from_keyring(name) else "missing"
