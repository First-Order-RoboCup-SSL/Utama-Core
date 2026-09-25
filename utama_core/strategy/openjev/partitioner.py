"""OpenJev as a kernel `Partitioner`, plus the `build_*_kernel_strategy`-shaped factory.

Every `1 / decision_hz` seconds of *sim* time the partitioner encodes the game,
asks OpenJev which split to use, and holds that split until the next decision
(`decision_hz=60` = a fresh decision on every control tick). The sim is
stepped synchronously, so inference latency never costs the team game time —
it only makes the match slower to run.

If the server errors, the tick falls back to `tiki_taka`'s picker and the
error is logged; the match keeps going.

Each decision appends one JSON line to `log_path` (if set): sim time, the live
state block, the full probability vector, choice, confidence, latency, cache
hit, fallback reason and the resulting partition.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import IO, Any, Callable, Optional

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.engine.tactic import RobotId
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.kernel_strategy import _tiki_taka_picker
from utama_core.strategy.openjev import encoder
from utama_core.strategy.openjev.client import OpenJevClient
from utama_core.strategy.openjev.splits import SPLIT_KEYS, allocate, make_tactics

Partition = dict[str, frozenset[RobotId]]


class OpenJevPartitioner:
    def __init__(
        self,
        client: OpenJevClient,
        decision_hz: float = 60.0,
        log_path: Optional[str] = None,
        fallback: Callable[..., Partition] = _tiki_taka_picker,
    ):
        self.client = client
        self.period = 1.0 / decision_hz
        self.fallback = fallback
        self._log: Optional[IO[str]] = (
            open(log_path, "w", buffering=1) if log_path else None
        )
        self.current_split: Optional[str] = None
        self._split_since: float = 0.0
        self._last_decision_t: Optional[float] = None
        self._recent: deque[tuple[float, str]] = deque(maxlen=3)
        self._last_possession: Optional[str] = None
        self._last_score: Optional[tuple[int, int]] = None
        self.n_decisions = self.n_cached = self.n_fallback = 0
        self.latency_ms_total = 0.0

    # --- events for the state's `recent` list -------------------------------------------------

    def _note(self, t: float, event: str) -> None:
        self._recent.appendleft((t, event))

    def _track_events(self, game: Game, t: float, possession: str) -> None:
        if self._last_possession is not None and possession != self._last_possession:
            if possession == "us":
                self._note(t, "we won possession")
            elif possession == "them":
                self._note(t, "we lost possession")
        self._last_possession = possession
        ref = game.referee
        if ref is not None:
            us, them = (
                (ref.yellow_team, ref.blue_team)
                if game.my_team_is_yellow
                else (ref.blue_team, ref.yellow_team)
            )
            score = (us.score, them.score)
            if self._last_score is not None and score != self._last_score:
                self._note(
                    t, "we scored" if score[0] > self._last_score[0] else "they scored"
                )
            self._last_score = score

    def _recent_text(self, t: float) -> list[str]:
        return [f"{round(t - et, 1)}s ago: {e}" for et, e in self._recent]

    # --- Partitioner --------------------------------------------------------------------------

    def __call__(
        self,
        game: Game,
        free_robots: frozenset[RobotId],
        prev_partition: Optional[Partition],
        applicable_tactic_ids: frozenset[str],
    ) -> Partition:
        t = game.ts
        possession, _, _ = encoder.possession(game)
        self._track_events(game, t, possession)

        due = (
            self._last_decision_t is None
            or t - self._last_decision_t >= self.period - 1e-9
        )
        if due:
            self._last_decision_t = t
            fallback_partition = self._decide(
                game, t, free_robots, prev_partition, applicable_tactic_ids
            )
            if fallback_partition is not None:
                return fallback_partition
        if self.current_split is None:
            # No successful decision yet (server failing): stay on the scripted picker between
            # attempts instead of retrying the server every tick.
            return self.fallback(
                game, free_robots, prev_partition, applicable_tactic_ids
            )
        return allocate(self.current_split, free_robots, applicable_tactic_ids)

    def _decide(
        self,
        game: Game,
        t: float,
        free_robots: frozenset[RobotId],
        prev_partition: Optional[Partition],
        applicable: frozenset[str],
    ) -> Optional[Partition]:
        robot_tactic = {
            rid: tid for tid, rids in (prev_partition or {}).items() for rid in rids
        }
        live = encoder.live_state(
            game,
            current_split=self.current_split,
            split_held_s=t - self._split_since if self.current_split else 0.0,
            robot_tactic=robot_tactic,
            applicable=applicable,
            recent=self._recent_text(t),
        )
        row: dict[str, Any] = {"t": round(t, 3), "live": live}
        self.n_decisions += 1
        try:
            d = self.client.decide(encoder.render(live), encoder.question())
        except (
            Exception
        ) as e:  # server down / bad response: keep playing on the scripted picker
            self.n_fallback += 1
            partition = self.fallback(game, free_robots, prev_partition, applicable)
            row.update(
                fallback=f"{type(e).__name__}: {e}"[:300],
                partition=_jsonable(partition),
            )
            self._write(row)
            return partition

        ans = d.answers[encoder.QUESTION_KEY]
        choice = ans["choice"]
        if (
            choice not in SPLIT_KEYS
        ):  # defensive: server answered with a key we didn't offer
            choice = self.current_split or SPLIT_KEYS[0]
        if choice != self.current_split:
            if self.current_split is not None:
                self._note(t, f"switched split to {choice}")
            self.current_split = choice
            self._split_since = t
        self.n_cached += int(d.cached)
        self.latency_ms_total += d.latency_ms
        partition = allocate(choice, free_robots, applicable)
        row.update(
            choice=choice,
            probabilities=ans.get("probabilities"),
            confidence=ans.get("confidence"),
            latency_ms=round(d.latency_ms, 1),
            cached=d.cached,
            fallback=None,
            partition=_jsonable(partition),
        )
        self._write(row)
        return None

    def _write(self, row: dict[str, Any]) -> None:
        if self._log is not None:
            self._log.write(json.dumps(row, ensure_ascii=False) + "\n")

    def summary(self) -> dict[str, Any]:
        live = self.n_decisions - self.n_cached - self.n_fallback
        return {
            "decisions": self.n_decisions,
            "cached": self.n_cached,
            "fallback": self.n_fallback,
            "mean_latency_ms": round(self.latency_ms_total / live, 1) if live else None,
        }

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None


def _jsonable(p: Partition) -> dict[str, list[int]]:
    return {k: sorted(v) for k, v in p.items()}


def build_openjev_kernel_strategy(
    outfield_robot_ids: tuple[int, ...],
    *,
    url: str = "http://127.0.0.1:3000",
    decision_hz: float = 60.0,
    log_path: Optional[str | Path] = None,
    client: Optional[OpenJevClient] = None,
    on_build: Optional[Callable[[OpenJevPartitioner], None]] = None,
):
    """OpenJev picks the split; the existing tactics play it. Same shape as the
    `build_*_kernel_strategy` factories, so `AbstractStrategy` takes it unchanged.

    `on_build` receives the partitioner once the strategy is built, so a match
    driver can read its counters and close its log afterwards.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        jev_client = client or OpenJevClient(url)
        partitioner = OpenJevPartitioner(
            jev_client,
            decision_hz=decision_hz,
            log_path=str(log_path) if log_path else None,
        )
        if on_build is not None:
            on_build(partitioner)
        return KernelSchedulerStrategy(
            tactics=make_tactics(),
            partitioner=partitioner,
            outfield_robot_ids=outfield_robot_ids,
            ctx=TickContext(motion_controller=motion_controller),
        )

    return _build
