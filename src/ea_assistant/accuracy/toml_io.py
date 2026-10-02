from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_toml(path: str | Path) -> dict[str, Any]:
    """Read one TOML fixture with the shared binary-open/decode path."""
    with Path(path).open("rb") as handle:
        value = tomllib.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"TOML file {Path(path).name!r} must contain a table")
    return value
