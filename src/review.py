from __future__ import annotations

from pathlib import Path

from src.edits import default_edit_log_path


def resolve_edit_log_path(image_path: str | Path, explicit_path: str | None = None) -> Path | None:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None
    default_path = default_edit_log_path(image_path)
    if default_path.exists():
        return default_path
    return None
