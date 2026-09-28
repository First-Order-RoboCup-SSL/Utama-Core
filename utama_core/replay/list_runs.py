"""List the tournament runs under ./replays (`pixi run runs`).

One line per `replays/tournament_*` directory, oldest first: when it started, the
git commit it ran on (`*` if the tree was dirty), how many matches finished, how
many stalled, and its command-line arguments -- all read from the run's
`summary.json`. A run still in progress, or from before a field existed, shows `-`
for what its summary doesn't have.
"""

import json
from pathlib import Path

from utama_core.config.settings import REPLAY_BASE_PATH


def run_rows(replays_dir: Path) -> list[dict]:
    rows = []
    for run_dir in sorted(replays_dir.glob("tournament_*")):
        if not run_dir.is_dir():
            continue
        try:
            summary = json.loads((run_dir / "summary.json").read_text())
        except (OSError, ValueError):
            summary = {}
        run = summary.get("run") or {}
        commit = run.get("git_commit")
        results = summary.get("results")
        argv = run.get("argv")
        rows.append(
            {
                "run": run_dir.name,
                "started": run.get("started_utc", "-"),
                "commit": (commit[:8] + ("*" if run.get("git_dirty") else "")) if commit else "-",
                "matches": len(results) if isinstance(results, list) else "-",
                "stalled": summary.get("stalled_match_count", "-"),
                "argv": " ".join(argv) if argv else "-",
            }
        )
    return rows


def main() -> None:
    rows = run_rows(REPLAY_BASE_PATH)
    if not rows:
        print(f"No tournament runs in {REPLAY_BASE_PATH}")
        return
    columns = ["run", "started", "commit", "matches", "stalled", "argv"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns[:-1]}
    print("  ".join(c.ljust(widths[c]) for c in columns[:-1]) + "  argv")
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in columns[:-1]) + "  " + r["argv"])


if __name__ == "__main__":
    main()
