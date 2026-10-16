"""Official RoboCup SSL game logs as replays in our format (`utama_core/replay/ssl_log.py`).

    pixi run python tools/convert_ssl_log.py LOG_OR_DIR ... --out DIR [--raw-vision]
    pixi run python tools/convert_ssl_log.py --corpus 2022-2026 --out DIR [--division div-b] [--jobs 4]
        [--raw-vision] [--backfill-extras]

The first form converts game logs (`*.log.gz`; a directory: every log under it). The second
lists the logs the league publishes (https://ssl.robocup.org/game-logs/, the public share
`<year>/<division>/`) and, per log, downloads it into `--raw-dir`, converts it into
`DIR/<year>/` and deletes the download: a raw log is 50-350 MB, its replay about a fifth of
that, so only `--jobs` raw logs are on disk at any time. A log whose replay exists is skipped,
so an interrupted run resumes; `DIR/corpus.json` lists every log with its outcome.

Each replay is `<stem>.npz` + `<stem>.sparse_referee.pkl` + `<stem>.source.json` (teams, score,
tracker source, field geometry), `<stem>` the log's name without `.log.gz`. Yellow is the
friendly team. Downloads are parsed as data only. Do not write them into `replays/`.

Sidecars for measuring physics (see `ssl_log`): `<stem>.kicks.json` (the tracker's kicks) and
`<stem>.tracker.zip` (visibility, angular velocity) always, `<stem>.vision.zip` (every raw
camera detection, about the replay's size again) with `--raw-vision`. `--backfill-extras`
downloads a log whose replay exists but lacks a sidecar asked for, and adds the sidecars
without rewriting the replay; without it such a log is skipped. `--keep-raw` keeps each
download in `--raw-dir/<year>/` instead of deleting it (about 15–20 GB for 2022–2026 Division
B), reuses a kept download instead of fetching it again, and downloads the raw log of a match
that is otherwise complete.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.replay.ssl_log import convert_log, log_stem  # noqa: E402

SHARE_HOST = "https://cloud.robocup.org"
SHARE = f"{SHARE_HOST}/public.php/dav/files/2mkwfxZjPf6NBPS"
# Nextcloud answers a public-share WebDAV request without this header with 401
HEADERS = {"X-Requested-With": "XMLHttpRequest"}


def list_logs(year: int, division: str) -> list[dict]:
    """The published logs of one year and division: url, file name and size in bytes."""
    request = urllib.request.Request(
        f"{SHARE}/{year}/{division}/", method="PROPFIND", headers={**HEADERS, "Depth": "1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        root = ET.fromstring(response.read())
    logs = []
    for item in root.iter("{DAV:}response"):
        href = item.findtext("{DAV:}href") or ""
        if not href.endswith(".log.gz"):
            continue
        size = item.findtext(".//{DAV:}getcontentlength")
        name = urllib.parse.unquote(href.rsplit("/", 1)[1])
        logs.append({"year": year, "url": SHARE_HOST + href, "name": name, "bytes": int(size or 0)})
    return sorted(logs, key=lambda log: log["name"])


def _download(url: str, path: Path, attempts: int = 3) -> None:
    part = path.with_name(path.name + ".part")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as response:
                with open(part, "wb") as f:
                    shutil.copyfileobj(response, f, length=1 << 20)
            part.replace(path)
            return
        except OSError:
            part.unlink(missing_ok=True)
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))


def _summary_line(stem: str, summary: dict) -> str:
    score = summary["score"]
    return (
        f"{stem}: {summary['yellow']} {score['yellow']}-{score['blue']} {summary['blue']}, "
        f"{summary['duration_s'] / 60:.0f} min, world from {summary['world_source']}"
    )


def has_extras(out_dir: Path, stem: str, raw_vision: bool) -> bool:
    """Whether a converted replay has the sidecars asked for (per its `source.json`)."""
    extras = json.loads((out_dir / f"{stem}.source.json").read_text()).get("extras")
    return extras is not None and (extras["raw_vision"] or not raw_vision)


def _corpus_job(job: dict) -> dict:
    """Download, convert (or only add sidecars to) and delete (unless kept) one published log."""
    stem = log_stem(job["name"])
    keep_raw = job.get("keep_raw", False)
    out_dir = Path(job["out"]) / str(job["year"])
    raw_dir = Path(job["raw_dir"]) / str(job["year"]) if keep_raw else Path(job["raw_dir"])
    raw = raw_dir / f"{stem}.log.gz"
    row = {"year": job["year"], "log": job["name"], "stem": stem, "bytes": job["bytes"]}
    exists = (out_dir / f"{stem}.npz").exists() and (out_dir / f"{stem}.source.json").exists()
    complete = exists and (not job["backfill_extras"] or has_extras(out_dir, stem, job["raw_vision"]))
    if complete and (not keep_raw or raw.exists()):
        return {**row, "status": "exists"}
    started = time.time()
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        if not raw.exists():  # `_download` writes `.part` and renames it once complete
            _download(job["url"], raw)
        if complete:
            return {**row, "status": "raw kept", "seconds": round(time.time() - started)}
        summary = convert_log(raw, out_dir, stem=stem, raw_vision=job["raw_vision"], write_replay=not exists)
        return {
            **row,
            "status": "extras added" if exists else "converted",
            "seconds": round(time.time() - started),
            "line": _summary_line(stem, summary),
        }
    except Exception as exc:  # one bad log must not stop the corpus
        return {**row, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
    finally:
        if not keep_raw:
            raw.unlink(missing_ok=True)
        raw.with_name(raw.name + ".part").unlink(missing_ok=True)


def run_corpus(
    years: list[int],
    division: str,
    out: Path,
    raw_dir: Path,
    jobs: int,
    raw_vision: bool = False,
    backfill_extras: bool = False,
    keep_raw: bool = False,
) -> list[dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    work = []
    for year in years:
        logs = list_logs(year, division)
        print(f"{year}/{division}: {len(logs)} logs, {sum(log['bytes'] for log in logs) / 1e9:.1f} GB", flush=True)
        work += [
            {
                **log,
                "out": str(out),
                "raw_dir": str(raw_dir),
                "raw_vision": raw_vision,
                "backfill_extras": backfill_extras,
                "keep_raw": keep_raw,
            }
            for log in logs
        ]
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for row in pool.map(_corpus_job, work):
            rows.append(row)
            print(
                f"[{len(rows)}/{len(work)}] {row['status']}: {row.get('line') or row.get('error') or row['stem']}",
                flush=True,
            )
    (out / "corpus.json").write_text(json.dumps(rows, indent=1) + "\n")
    return rows


def _years(spec: str) -> list[int]:
    first, _, last = spec.partition("-")
    return list(range(int(first), int(last or first) + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path, help="game logs, or directories of them, to convert")
    parser.add_argument("--out", type=Path, required=True, help="output directory (not replays/)")
    parser.add_argument("--corpus", metavar="YEARS", help="download and convert the published logs, e.g. 2022-2026")
    parser.add_argument("--division", default="div-b", help="share directory per year (default div-b)")
    parser.add_argument("--raw-dir", type=Path, help="where downloads go while converted (default OUT/raw)")
    parser.add_argument("--jobs", type=int, default=1, help="logs downloaded and converted at once")
    parser.add_argument("--raw-vision", action="store_true", help="also keep every raw camera detection")
    parser.add_argument(
        "--backfill-extras",
        action="store_true",
        help="add missing sidecars to replays already converted, without rewriting them",
    )
    parser.add_argument(
        "--keep-raw", action="store_true", help="keep each download in RAW_DIR/<year>/ instead of deleting it"
    )
    args = parser.parse_args()

    if args.corpus:
        rows = run_corpus(
            _years(args.corpus),
            args.division,
            args.out,
            args.raw_dir or args.out / "raw",
            args.jobs,
            raw_vision=args.raw_vision,
            backfill_extras=args.backfill_extras,
            keep_raw=args.keep_raw,
        )
        failed = [row for row in rows if row["status"] == "failed"]
        print(f"{len(rows) - len(failed)}/{len(rows)} logs converted or already there; failed: {len(failed)}")
        sys.exit(1 if failed else 0)

    logs = []
    for path in args.paths:
        logs += sorted([*path.rglob("*.log.gz"), *path.rglob("*.log")]) if path.is_dir() else [path]
    if not logs:
        parser.error("no game logs given")
    for log in logs:
        stem = log_stem(log.name)
        exists = (args.out / f"{stem}.npz").exists() and (args.out / f"{stem}.source.json").exists()
        if exists and args.backfill_extras and has_extras(args.out, stem, args.raw_vision):
            print(f"{stem}: sidecars already there", flush=True)
            continue
        replay = not (exists and args.backfill_extras)
        summary = convert_log(log, args.out, raw_vision=args.raw_vision, write_replay=replay)
        print(_summary_line(stem, summary) + ("" if replay else " (sidecars added)"), flush=True)


if __name__ == "__main__":
    main()
