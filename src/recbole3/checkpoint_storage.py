"""Scratch-backed checkpoint writes with compatible project-local read paths.

RecBole checkpoint paths are mirrored relative to the repository. Explicit
paths outside the repository are respected. No fallback to home storage is used.
"""
from __future__ import annotations
import os
from pathlib import Path
import tempfile
from typing import Any
from recbole3.storage_paths import checkpoint_root, configured_root, data_root
PROJECT_ROOT = Path(__file__).absolute().parents[2]

def scratch_root() -> Path:
    return PROJECT_ROOT

def checkpoint_path(path: str | Path) -> Path:
    """Map a lexical project path to scratch without following existing symlinks."""
    local = Path(os.path.abspath(os.path.expanduser(str(path))))
    root = scratch_root()
    if local.is_relative_to(root):
        return local
    if local.is_relative_to(PROJECT_ROOT):
        return root / local.relative_to(PROJECT_ROOT)
    for shared_data in (data_root() / PROJECT_ROOT.name, PROJECT_ROOT.parent / 'data' / PROJECT_ROOT.name):
        if local.is_relative_to(shared_data):
            return root / local.relative_to(shared_data)
    return local

def _publish_link(local: Path, target: Path) -> None:
    if local.resolve() == target.resolve():
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    (fd, temporary) = tempfile.mkstemp(prefix='.checkpoint-link-', dir=local.parent)
    os.close(fd)
    os.unlink(temporary)
    try:
        os.symlink(str(target), temporary)
        os.replace(temporary, local)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)

def checkpoint_directory(path: str | Path) -> Path:
    """Create a scratch directory; retain existing local logs and old checkpoints."""
    local = Path(os.path.abspath(os.path.expanduser(str(path))))
    target = checkpoint_path(local)
    target.mkdir(parents=True, exist_ok=True)
    if local == target:
        return target
    if local.is_symlink():
        if local.resolve() != target.resolve():
            raise ValueError(f'Checkpoint directory links to a different location: {local}')
        return target
    if local.is_dir():
        try:
            local.rmdir()
        except OSError:
            return target
    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        local.symlink_to(target, target_is_directory=True)
    except FileExistsError:
        if not local.is_dir():
            raise
    return target

def publish_checkpoint_directory(path: str | Path) -> None:
    """Expose HF outputs through local paths after a successful training save."""
    local = Path(os.path.abspath(os.path.expanduser(str(path))))
    target = checkpoint_path(local)
    if local.resolve() == target.resolve():
        return
    if local.is_dir():
        for existing in local.iterdir():
            if existing.is_symlink() and (not existing.exists()):
                linked = Path(os.readlink(existing))
                if not linked.is_absolute():
                    linked = existing.parent / linked
                if linked.absolute().is_relative_to(target):
                    existing.unlink()
    for child in target.iterdir():
        destination = local / child.name
        if child.is_dir() and destination.is_dir() and (not destination.is_symlink()):
            publish_checkpoint_directory(destination)
        else:
            _publish_link(destination, child)

def save_checkpoint(obj: Any, path: str | Path, **kwargs: Any) -> None:
    """Atomically save a PyTorch checkpoint on scratch, then publish its local link."""
    import torch
    local = Path(os.path.abspath(os.path.expanduser(str(path))))
    target = checkpoint_path(local)
    target.parent.mkdir(parents=True, exist_ok=True)
    (fd, temporary) = tempfile.mkstemp(prefix='checkpoint-save-', suffix='.pt', dir=target.parent)
    os.close(fd)
    try:
        torch.save(obj, temporary, **kwargs)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    _publish_link(local, target)

def remove_checkpoint(path: str | Path) -> None:
    """Prune both the scratch checkpoint and its compatibility link."""
    local = Path(os.path.abspath(os.path.expanduser(str(path))))
    target = checkpoint_path(local)
    target.unlink(missing_ok=True)
    if local != target and local.is_symlink():
        local.unlink(missing_ok=True)
