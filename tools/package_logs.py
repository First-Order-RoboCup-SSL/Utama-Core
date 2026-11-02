"""Package tournament results and field logs into lightweight, shareable archives.

Modes:
  1. Tournament Results (Tier 1): packages summary.json, *.stats.json, *.timeline.json,
     ledger entries, and reports while stripping large columnar *.npz arrays (reducing
     ~1 GB to ~3 MB).
  2. Real Field Logs (Tier 2): compresses raw .log files to .log.gz and bundles them
     with calibration files and hardware configurations for sharing.

Usage:
    pixi run python tools/package_logs.py tournament [DIR_OR_ID] [--out OUT] [--with-intentions] [--dry-run]
    pixi run python tools/package_logs.py field [DIR] [--out OUT] [--compress-in-place] [--dry-run]
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_latest_tournament(replays_dir: Path) -> Optional[Path]:
    """Find the most recent tournament_* directory in replays_dir."""
    runs = sorted(replays_dir.glob("tournament_*"))
    return runs[-1] if runs else None


def collect_tournament_files(run_dir: Path, with_intentions: bool = False) -> list[Path]:
    """Collect lightweight tournament files, skipping heavy .npz and .pkl dumps."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Tournament directory not found: {run_dir}")

    files: list[Path] = []
    # Core summaries and stats
    for pattern in ("summary.json", "*.stats.json", "*.timeline.json", "*.md", "ledger.jsonl"):
        files.extend(sorted(run_dir.glob(pattern)))

    if with_intentions:
        files.extend(sorted(run_dir.glob("*.intentions.jsonl*")))

    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f.is_file() and f not in seen:
            seen.add(f)
            unique_files.append(f)
    return unique_files


def collect_field_files(field_dir: Path, include_replays: bool = False) -> list[Path]:
    """Collect field logs, calibration files, and robot hardware descriptions."""
    field_dir = Path(field_dir)
    if not field_dir.is_dir():
        raise FileNotFoundError(f"Field directory not found: {field_dir}")

    files: list[Path] = []
    patterns = ["*.log", "*.log.gz", "*.calibration.json", "*.tag.json", "robots_found.json", "*.json"]
    if include_replays:
        patterns.extend(["*.npz", "*.sparse_referee.pkl"])

    for pattern in patterns:
        files.extend(sorted(field_dir.glob(pattern)))

    # Filter out empty files, temporary files, and .tmp.npz / .tmp.pkl
    valid = []
    seen = set()
    for f in files:
        if not f.is_file() or f in seen:
            continue
        if ".tmp" in f.name or f.stat().st_size == 0:
            continue
        seen.add(f)
        valid.append(f)
    return valid


def compress_log_file(src: Path, compresslevel: int = 6) -> Path:
    """Gzip an individual .log file into .log.gz and remove the uncompressed source."""
    dest = src.with_suffix(".log.gz")
    with open(src, "rb") as f_in, gzip.open(dest, "wb", compresslevel=compresslevel) as f_out:
        shutil.copyfileobj(f_in, f_out)
    src.unlink()
    return dest


def compress_field_logs_in_place(field_dir: Path) -> list[tuple[Path, Path]]:
    """Compress all uncompressed .log files in field_dir in-place."""
    field_dir = Path(field_dir)
    results = []
    for log_file in sorted(field_dir.glob("*.log")):
        if log_file.is_file() and log_file.stat().st_size > 0:
            gz_path = compress_log_file(log_file)
            results.append((log_file, gz_path))
    return results


def package_to_tar_gz(files: Sequence[Path], base_dir: Path, out_tar_gz: Path) -> Path:
    """Pack files into a .tar.gz archive with paths relative to base_dir."""
    out_tar_gz = Path(out_tar_gz)
    out_tar_gz.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar_gz, "w:gz") as tar:
        for file in files:
            arcname = file.relative_to(base_dir)
            tar.add(file, arcname=str(arcname))
    return out_tar_gz


def package_tournament(
    target: Optional[str] = None,
    out: Optional[Path] = None,
    with_intentions: bool = False,
    dry_run: bool = False,
    replays_dir: Path = REPO_ROOT / "replays",
) -> Optional[Path]:
    if target:
        run_dir = Path(target) if Path(target).is_dir() else replays_dir / target
    else:
        run_dir = find_latest_tournament(replays_dir)
        if not run_dir:
            print("No tournament directory found in replays/", file=sys.stderr)
            return None

    files = collect_tournament_files(run_dir, with_intentions=with_intentions)
    if not files:
        print(f"No lightweight tournament files found in {run_dir}", file=sys.stderr)
        return None

    total_bytes = sum(f.stat().st_size for f in files)
    out_path = out or replays_dir / f"{run_dir.name}_results.tar.gz"

    print(f"Tournament: {run_dir.name}")
    print(f"Files to package ({len(files)} files, {total_bytes / (1024 * 1024):.2f} MB uncompressed):")
    for f in files[:8]:
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    if len(files) > 8:
        print(f"  ... and {len(files) - 8} more files")

    if dry_run:
        print(f"\n[dry-run] Would write archive to {out_path}")
        return out_path

    package_to_tar_gz(files, base_dir=run_dir.parent, out_tar_gz=out_path)
    arch_size = out_path.stat().st_size
    print(f"\nWrote {out_path} ({arch_size / (1024 * 1024):.2f} MB)")
    return out_path


def package_field(
    field_dir: Optional[Path] = None,
    out: Optional[Path] = None,
    compress_in_place: bool = False,
    include_replays: bool = False,
    dry_run: bool = False,
) -> Optional[Path]:
    dir_path = Path(field_dir) if field_dir else REPO_ROOT / "replays" / "field"
    if not dir_path.is_dir():
        print(f"Field directory not found: {dir_path}", file=sys.stderr)
        return None

    if compress_in_place and not dry_run:
        compressed = compress_field_logs_in_place(dir_path)
        if compressed:
            print(f"Compressed {len(compressed)} .log files to .log.gz in-place.")

    files = collect_field_files(dir_path, include_replays=include_replays)
    if not files:
        print(f"No field files found in {dir_path}", file=sys.stderr)
        return None

    total_bytes = sum(f.stat().st_size for f in files)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out or dir_path.parent / f"field_logs_{timestamp}.tar.gz"

    print(f"Field logs directory: {dir_path}")
    print(f"Files to package ({len(files)} files, {total_bytes / (1024 * 1024):.2f} MB):")
    for f in files[:8]:
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    if len(files) > 8:
        print(f"  ... and {len(files) - 8} more files")

    if dry_run:
        print(f"\n[dry-run] Would write archive to {out_path}")
        return out_path

    package_to_tar_gz(files, base_dir=dir_path.parent, out_tar_gz=out_path)
    arch_size = out_path.stat().st_size
    print(f"\nWrote {out_path} ({arch_size / (1024 * 1024):.2f} MB)")
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Package tournament results and field logs into shareable archives.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Tournament subparser
    t_parser = subparsers.add_parser("tournament", help="Package lightweight tournament summary & statistics.")
    t_parser.add_argument("target", nargs="?", default=None, help="Tournament dir or name (default: latest).")
    t_parser.add_argument("--out", type=Path, default=None, help="Output .tar.gz archive path.")
    t_parser.add_argument("--with-intentions", action="store_true", help="Include intentions logs.")
    t_parser.add_argument("--dry-run", action="store_true", help="List files without creating archive.")

    # Field subparser
    f_parser = subparsers.add_parser("field", help="Package real-robot field logs & calibration files.")
    f_parser.add_argument("dir", nargs="?", type=Path, default=None, help="Field directory (default: replays/field).")
    f_parser.add_argument("--out", type=Path, default=None, help="Output .tar.gz archive path.")
    f_parser.add_argument("--compress-in-place", action="store_true", help="Gzip uncompressed .log files first.")
    f_parser.add_argument("--include-replays", action="store_true", help="Also include .npz and .pkl replays.")
    f_parser.add_argument("--dry-run", action="store_true", help="List files without creating archive.")

    args = parser.parse_args(argv)

    if args.command == "tournament":
        package_tournament(
            target=args.target,
            out=args.out,
            with_intentions=args.with_intentions,
            dry_run=args.dry_run,
        )
    elif args.command == "field":
        package_field(
            field_dir=args.dir,
            out=args.out,
            compress_in_place=args.compress_in_place,
            include_replays=args.include_replays,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
