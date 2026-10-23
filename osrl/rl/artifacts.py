"""Checkpoint naming and publication shared with the lightweight dashboard."""

from __future__ import annotations

import json
from pathlib import Path


def is_validated(path: str | Path) -> bool:
    # UNVALIDATED contains VALIDATED: substring matching is unsafe here.
    return Path(path).resolve().stem.endswith("_VALIDATED")


def normalization_path(checkpoint: str | Path) -> Path:
    return Path(checkpoint).resolve().with_suffix(".vecnorm.pkl")


def read_manifest(checkpoint: str | Path) -> dict:
    try:
        result = json.loads(Path(checkpoint).resolve().with_suffix(".json").read_text())
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError):
        return {}


def publish_link(directory: Path, name: str, checkpoint: Path) -> None:
    """Readers see either complete old artifacts or complete new artifacts."""
    temporary = directory / f".{name}.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(checkpoint)
    temporary.replace(directory / name)
