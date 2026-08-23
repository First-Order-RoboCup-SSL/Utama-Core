"""`MatchLog` — structured trace of tactic-assignment decisions, for post-match analysis.

Not a debug log (see `logging.basicConfig(level=logging.CRITICAL)` in
`strategy_runner.py`, which would silently swallow `logger.info` calls
anyway) — a separate, deliberately structured event stream an agent can read
back to answer "why did this match go the way it did" without inferring
intent from raw robot coordinates. One event per tactic-assignment change,
not per tick, so a robot holding the same tactic for seconds produces one
line, not thousands.

`TraceEvent`/`trace()` extend this to arbitrary scalar facts a tactic or
skill wants to record mid-`tick()` — e.g. "which branch did `go_to_ball` take
this tick", "was a shot lane open" — the exact things that used to get
answered with a hand-added `os.environ`-gated `print()`, run, read stdout,
then revert before committing. `KernelContext.match_log` (set by `Strategy`
from the same instance passed to `to_jsonl()`) is how a tactic/skill reaches
this without every call site threading a separate logger through. Same file,
same reader (`load_jsonl` returns both event kinds in tick order) — a second
parallel logging path was considered and rejected as unnecessary duplication
of what this module already does for `IntentionEvent`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

from utama_core.engine.tactic import RobotId, TacticId, TacticTag

_UNSET = object()


@dataclass(frozen=True)
class IntentionEvent:
    tick: int
    sim_time: float
    tactic_id: TacticId
    robot_ids: tuple[RobotId, ...]
    tag: TacticTag
    note: Optional[str] = None


@dataclass(frozen=True)
class TraceEvent:
    tick: int
    sim_time: float
    key: str
    value: Any


@dataclass(frozen=True)
class RefereeEvent:
    """One row per referee-state *change* (score/command/stage/designated
    position) — not per tick. "Changed" is decided by the caller comparing
    consecutive `RefereeData` via its own `__eq__` (which already excludes
    noisy fields like timestamps/game_events); this dataclass just records
    what changed to.
    """

    tick: int
    sim_time: float
    command: str
    stage: str
    yellow_score: int
    blue_score: int
    designated: Optional[tuple] = None
    note: Optional[str] = None


class MatchLog:
    """Accumulates `IntentionEvent`s/`TraceEvent`s/`RefereeEvent`s during a match; flush once via `to_jsonl()`."""

    def __init__(self) -> None:
        self._events: list[Union[IntentionEvent, TraceEvent, RefereeEvent]] = []
        self._last_trace_value: dict[str, Any] = {}

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

    def trace(self, tick: int, sim_time: float, key: str, value: Any) -> None:
        """Record one arbitrary scalar fact for this tick (JSON-serializable `value`)."""
        self._events.append(TraceEvent(tick=tick, sim_time=sim_time, key=key, value=value))

    def trace_if_changed(self, tick: int, sim_time: float, key: str, value: Any) -> None:
        """Like `trace()`, but only records when `value` differs from the last value logged for `key`."""
        if self._last_trace_value.get(key, _UNSET) != value:
            self._last_trace_value[key] = value
            self.trace(tick, sim_time, key, value)

    def referee(
        self,
        tick: int,
        sim_time: float,
        command: str,
        stage: str,
        yellow_score: int,
        blue_score: int,
        designated: Optional[tuple] = None,
        note: Optional[str] = None,
    ) -> None:
        """Record a referee-state change. Caller decides "changed" (see `RefereeEvent`)."""
        self._events.append(
            RefereeEvent(
                tick=tick,
                sim_time=sim_time,
                command=command,
                stage=stage,
                yellow_score=yellow_score,
                blue_score=blue_score,
                designated=tuple(designated) if designated is not None else None,
                note=note,
            )
        )

    def events(self) -> list[Union[IntentionEvent, TraceEvent, RefereeEvent]]:
        return list(self._events)

    def to_jsonl(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            for event in self._events:
                row = asdict(event)
                if isinstance(event, TraceEvent):
                    row["event"] = "trace"
                elif isinstance(event, RefereeEvent):
                    row["event"] = "referee"
                else:
                    row["event"] = "intention"
                    row["tag"] = event.tag.value
                f.write(json.dumps(row) + "\n")


def load_jsonl(path: Union[str, Path]) -> list[Union[IntentionEvent, TraceEvent, RefereeEvent]]:
    """Read back a `MatchLog.to_jsonl()` file as `IntentionEvent`/`TraceEvent`/`RefereeEvent`s, in order."""
    events: list[Union[IntentionEvent, TraceEvent, RefereeEvent]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            kind = row.get("event")
            if kind == "trace":
                events.append(
                    TraceEvent(tick=row["tick"], sim_time=row["sim_time"], key=row["key"], value=row["value"])
                )
            elif kind == "referee":
                events.append(
                    RefereeEvent(
                        tick=row["tick"],
                        sim_time=row["sim_time"],
                        command=row["command"],
                        stage=row["stage"],
                        yellow_score=row["yellow_score"],
                        blue_score=row["blue_score"],
                        designated=tuple(row["designated"]) if row.get("designated") is not None else None,
                        note=row.get("note"),
                    )
                )
            else:
                events.append(
                    IntentionEvent(
                        tick=row["tick"],
                        sim_time=row["sim_time"],
                        tactic_id=row["tactic_id"],
                        robot_ids=tuple(row["robot_ids"]),
                        tag=TacticTag(row["tag"]),
                        note=row["note"],
                    )
                )
    return events
