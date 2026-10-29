"""One shared format for the problems the picture tools flag (`--findings PATH` on each tool).

A picture is for understanding a flagged problem, never the only evidence for it: each tool also emits
its flags as JSON lines, one `Finding` per line, so an agent (or `tools/triage.py`) can filter on the
JSON and open the picture only to see what it means.

    {"tool": "match_timeline", "kind": "tactic_thrash", "replay": "replays/.../a_vs_b.npz", "side": "friendly",
     "t0": 0.0, "t1": 480.0, "robots": [4], "value": 78.98, "threshold": 4.0, "unit": "per live min",
     "message": "R4 switched tactic 78.98 times per live minute (threshold 4.00).", "image": null}

`value` and `threshold` are in `unit`; `threshold` is None for descriptive facts with no limit.
`image` is the PNG drawn alongside, or None when the finding was computed without drawing.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Union


@dataclass(frozen=True)
class Finding:
    tool: str  # the tool that computed it, e.g. "match_timeline"
    kind: str  # e.g. "tactic_thrash", "parked", "pushed_out", "target_in_keep_out"
    replay: Optional[str]  # the replay path, or None for a run-level fact
    side: Optional[str]  # "friendly", "enemy" or None
    t0: Optional[float]  # seconds on the replay's clock
    t1: Optional[float]
    robots: tuple[int, ...]
    value: float
    threshold: Optional[float]
    unit: str
    message: str  # one plain sentence with the numbers
    image: Optional[str] = None  # the PNG drawn for it, when one was drawn


def to_dict(finding: Finding) -> dict:
    d = asdict(finding)
    d["robots"] = list(finding.robots)
    return d


def from_dict(d: dict) -> Finding:
    return Finding(
        tool=d["tool"],
        kind=d["kind"],
        replay=d.get("replay"),
        side=d.get("side"),
        t0=d.get("t0"),
        t1=d.get("t1"),
        robots=tuple(int(r) for r in d.get("robots") or ()),
        value=float(d["value"]),
        threshold=d.get("threshold"),
        unit=d["unit"],
        message=d["message"],
        image=d.get("image"),
    )


def write_jsonl(findings: Iterable[Finding], path: Union[str, Path]) -> None:
    """One JSON object per line. `path` "-" writes to stdout."""
    lines = "".join(json.dumps(to_dict(f)) + "\n" for f in findings)
    if str(path) == "-":
        sys.stdout.write(lines)
        sys.stdout.flush()
        return
    Path(path).write_text(lines)


def read_jsonl(path: Union[str, Path]) -> list[Finding]:
    with open(path) as f:
        return [from_dict(json.loads(line)) for line in f if line.strip()]
