"""`MatchLog` — structured trace of tactic-assignment decisions, for post-match analysis.

Not a debug log (see `logging.basicConfig(level=logging.CRITICAL)` in
`strategy_runner.py`, which would silently swallow `logger.info` calls
anyway) — a separate, deliberately structured event stream an agent can read
back to answer "why did this match go the way it did" without inferring
intent from raw robot coordinates. One event per tactic-assignment change,
not per tick, so a robot holding the same tactic for seconds produces one
line, not thousands.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

from utama_core.kernel.tactic import RobotId, TacticId, TacticTag


@dataclass(frozen=True)
class IntentionEvent:
    tick: int
    sim_time: float
    tactic_id: TacticId
    robot_ids: tuple[RobotId, ...]
    tag: TacticTag
    note: Optional[str] = None


class MatchLog:
    """Accumulates `IntentionEvent`s during a match; flush once via `to_jsonl()`."""

    def __init__(self) -> None:
        self._events: list[IntentionEvent] = []

    def intention(
        self,
        tick: int,
        sim_time: float,
        tactic_id: TacticId,
        robot_ids: tuple[RobotId, ...],
        tag: TacticTag,
        note: Optional[str] = None,
    ) -> None:
        self._events.append(
            IntentionEvent(
                tick=tick,
                sim_time=sim_time,
                tactic_id=tactic_id,
                robot_ids=tuple(sorted(robot_ids)),
                tag=tag,
                note=note,
            )
        )

    def events(self) -> list[IntentionEvent]:
        return list(self._events)

    def to_jsonl(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            for event in self._events:
                row = asdict(event)
                row["tag"] = event.tag.value
                f.write(json.dumps(row) + "\n")
