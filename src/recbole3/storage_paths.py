"""Portable DATA_ROOT and CHECKPOINT_ROOT configuration for this repository."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_NAME = 'RecBole3.0'
PROJECT_ROOT = Path(__file__).absolute().parents[2]


def configured_root(name: str, default: Path) -> Path:
    value = os.environ.get(name, str(default))
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty absolute path")
    path = Path(os.path.expandvars(value)).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path: {value}")
    return Path(os.path.abspath(path))


def data_root() -> Path:
    return configured_root("DATA_ROOT", Path.home() / "data")


def checkpoint_root() -> Path:
    return configured_root("CHECKPOINT_ROOT", Path.home() / "scratchDirectory")


def _project_path(path: str | Path, root: Path, project: str | None) -> Path:
    value = Path(os.path.expandvars(str(path))).expanduser()
    if value.is_absolute():
        value = Path(os.path.abspath(value))
        if value.is_relative_to(PROJECT_ROOT):
            return root / (project or PROJECT_NAME) / value.relative_to(PROJECT_ROOT)
        return value  # An explicitly supplied external destination.
    # Legacy cider/portrait entrypoints run from src and use ../rec_datasets.
    if value.parts[:2] == ("..", "rec_datasets"):
        value = Path(*value.parts[1:])
    if ".." in value.parts:
        return Path(os.path.abspath(value))
    return root / (project or PROJECT_NAME) / value


def data_path(path: str | Path = "data", *, project: str | None = None) -> Path:
    """Resolve a project-relative dataset path, preserving absolute external paths."""
    return Path(path).expanduser() if Path(path).expanduser().is_absolute() else PROJECT_ROOT / path


def checkpoint_path(path: str | Path = "", *, project: str | None = None) -> Path:
    """Resolve a project-relative checkpoint path under CHECKPOINT_ROOT/<project>."""
    return _project_path(path, checkpoint_root(), project)
