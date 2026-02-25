"""Utilities for discovering experiment checkpoints on disk."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


def find_experiment_dirs(
    search_root: Optional[Path] = None,
    task_filter: Optional[str] = None,
) -> list[Path]:
    """Find experiment directories containing config.pkl and checkpoints/.

    Searches up to 3 levels deep under search_root (default: cwd).
    Returns directories sorted by modification time (newest first).
    """
    if search_root is None:
        search_root = Path.cwd()

    candidates = []
    for dirpath, dirnames, filenames in os.walk(search_root):
        depth = len(Path(dirpath).relative_to(search_root).parts)
        if depth > 3:
            dirnames.clear()
            continue
        if "config.pkl" in filenames and "checkpoints" in dirnames:
            exp_dir = Path(dirpath)
            if task_filter and task_filter not in exp_dir.name:
                continue
            candidates.append(exp_dir)

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def find_latest_checkpoint(experiment_dir: Path) -> Optional[Path]:
    """Find the checkpoint with the highest frame number in an experiment dir."""
    ckpt_dir = experiment_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None

    checkpoints = sorted(
        ckpt_dir.glob("checkpoint_*.pt"),
        key=_checkpoint_frame_number,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def find_latest_experiment_checkpoint(
    search_root: Optional[Path] = None,
    task_filter: Optional[str] = None,
) -> Optional[Path]:
    """Find the latest checkpoint from the most recent experiment.

    Iterates experiments newest-first until one with checkpoints is found.
    """
    for exp_dir in find_experiment_dirs(search_root, task_filter):
        ckpt = find_latest_checkpoint(exp_dir)
        if ckpt is not None:
            return ckpt
    return None


def _checkpoint_frame_number(path: Path) -> int:
    """Extract the frame number from checkpoint_54000.pt -> 54000."""
    match = re.search(r"checkpoint_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else 0


def resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu'. Pass through other values."""
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def print_device_info(device: str) -> None:
    """Print which device will be used for training/evaluation."""
    import torch

    resolved = resolve_device(device)
    if resolved == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Device] Using GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("[Device] Using CPU")
