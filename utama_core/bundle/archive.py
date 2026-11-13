"""A `.strategy` file on disk: standard library only, so fingerprinting doesn't import a strategy."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

SUFFIX = ".strategy"
_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "utama" / "strategies"


def unpack(path: Path) -> Path:
    """The file's unpacked directory, shared by every process that plays the same bytes."""
    root = _CACHE / hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    if root.exists():
        return root
    _CACHE.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(dir=_CACHE))
    with zipfile.ZipFile(path) as archive:
        archive.extractall(partial)
    try:
        partial.rename(root)
    except OSError:  # another match unpacked it first
        shutil.rmtree(partial)
    return root


def read_manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text())
