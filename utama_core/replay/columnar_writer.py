"""Columnar replay writer — an alternative to `ReplayWriter`'s one-pickle-
per-frame format.

Why this exists: `find_stuck_windows`/dashboards/any bulk analysis over a
replay only ever wants numeric arrays (positions, velocities, referee
command) over the whole match's time axis — never one Python object per
frame. Loading the pickle format means reconstructing ~36,000 `GameFrame`s
(each nesting ~12 `Robot` + a `Ball` + a `RefereeData`) just to immediately
throw the objects away and pull out a handful of floats. Profiling a real
600s replay showed this reconstruction cost dominates load time (84% of it
in `pickle.load` alone). Storing the same data as flat numpy arrays from the start removes
that reconstruction cost entirely for bulk/vectorized consumers.

Design:
- Per-tick scalar/vector fields (robot positions, ball state, the three
  referee fields every real caller actually reads — see the grep audit in
  the commit that introduced this) are buffered per-frame as plain Python
  values during the match (cheap, same cost class as today's per-frame
  dict access) and only converted to fixed-width numpy arrays at flush
  time, when the complete robot-id roster for the whole match is known.
  This matters because a robot's id set is not guaranteed to be the same
  every tick (`test_non_sequential_robot_ids` exercises exactly this: a
  robot present at tick 0 can be absent by tick 1, and a new id can appear
  later) — locking the roster at the first tick would silently drop any
  robot that only appears later. A tick where a given id is absent gets
  NaN position/velocity/acceleration and `has_ball=False` for that id's
  column once everything is assembled.
- Flushing periodically (every `checkpoint_every_s` seconds of match time)
  means a mid-match crash still leaves a valid, loadable (if truncated)
  replay — mirroring the durability the old pickle-per-frame format got
  for free from flushing every dump, which matters when debugging a
  robosim crash. Each flush rebuilds arrays from every tick
  buffered so far (not just the newest ones) — this is the one op that's
  O(n_ticks) instead of O(1) per tick, deliberately paid only a few times
  per match rather than continuously.
- The dense/sparse split is by *shape*, not by how often a field changes.
  `referee_command`/`stage`/`designated_position` stay dense even though
  they're constant for long stretches (a command can hold for tens of
  seconds), because they're the three `RefereeData` fields real callers
  actually read in bulk across a whole match (grep audit: 29 uses of
  `.referee_command`, 5 of `.stage`, 3 of `designated_position`, 0 of
  everything else in this codebase) — e.g. `stuck_detector.py`'s
  `live_frac` check, which wants "what fraction of this window was
  `NORMAL_START`/`FORCE_START`" as a vectorized boolean-array reduction,
  not a Python loop reconstructing a `RefereeData` object per tick just to
  read one enum off it. Both also happen to fit a fixed-width numeric
  column trivially (small int enums; two floats with NaN for "absent").
  `game_events`/`status_message`/`source_identifier`/`next_command`/team
  info don't fit a column at all without truncation (variable-length
  lists, strings, nested objects) — those go to a small sparse sidecar
  list: the full `RefereeData` at each tick where it differs from the last
  stored one other than by its clocks running on (`columnar_reader.
  advance_clocks`), which the reader carries forward. A live referee sets
  `source_identifier` and team info on every message, so "any rare field
  set" meant every tick: 230 MB per round-robin of identical messages.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
import pickle
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Optional

import numpy as np

from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game import GameFrame
from utama_core.replay.columnar_reader import advance_clocks
from utama_core.replay.entities import ReplayMetadata

_NAN = float("nan")


@dataclass(kw_only=True)
class ColumnarReplayWriterConfig:
    """Same fields as `ReplayWriterConfig` — see that class for descriptions."""

    replay_name: str
    is_my_perspective: bool = True
    overwrite_existing: bool = False
    checkpoint_every_s: float = 10.0


@dataclass
class _TickRecord:
    ts: float
    friendly: dict  # id -> (p, v, a, orientation, has_ball)
    enemy: dict
    ball_p: tuple
    ball_v: tuple
    ball_a: tuple
    has_referee: bool
    referee_command: int
    stage: int
    designated_position: tuple


class ColumnarReplayWriter:
    """Drop-in replacement for `ReplayWriter` with the same `write_frame`/
    `close` interface, backed by columnar array buffers instead of one
    `pickle.dump` per frame. See module docstring for the format design.
    """

    def __init__(
        self,
        replay_configs: ColumnarReplayWriterConfig,
        my_team_is_yellow: bool,
        exp_friendly: int,
        exp_enemy: int,
        path: Optional[Path] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.replay_configs = replay_configs
        self.metadata = ReplayMetadata(
            my_team_is_yellow=my_team_is_yellow,
            exp_friendly=exp_friendly,
            exp_enemy=exp_enemy,
        )
        # `path` lets a caller (namely tests) place the file somewhere
        # other than `REPLAY_BASE_PATH` without exercising the real
        # directory/collision-avoidance logic in `_resolve_path`.
        self.path = path if path is not None else self._resolve_path(replay_configs)

        self._my_team_is_right: Optional[bool] = None
        self._ticks: list[_TickRecord] = []
        self._sparse_referee: list[tuple[int, RefereeData]] = []
        self._last_stored_ts = 0.0
        self._last_checkpoint_ts: Optional[float] = None

    def _resolve_path(self, replay_configs: ColumnarReplayWriterConfig) -> Path:
        path = REPLAY_BASE_PATH / f"{replay_configs.replay_name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not replay_configs.overwrite_existing:
            for i in count(1):
                candidate = REPLAY_BASE_PATH / f"{replay_configs.replay_name}_{i}.npz"
                if not candidate.exists():
                    return candidate
        return path

    def write_frame(self, frame: GameFrame) -> None:
        if self._my_team_is_right is None:
            self._my_team_is_right = frame.my_team_is_right
        friendly = {
            rid: (r.p.x, r.p.y, r.v.x, r.v.y, r.a.x, r.a.y, r.orientation, r.has_ball)
            for rid, r in frame.friendly_robots.items()
            if r is not None
        }
        enemy = {
            rid: (r.p.x, r.p.y, r.v.x, r.v.y, r.a.x, r.a.y, r.orientation, r.has_ball)
            for rid, r in frame.enemy_robots.items()
            if r is not None
        }
        if frame.ball is not None:
            ball_p = (frame.ball.p.x, frame.ball.p.y, frame.ball.p.z)
            ball_v = (frame.ball.v.x, frame.ball.v.y, frame.ball.v.z)
            ball_a = (frame.ball.a.x, frame.ball.a.y, frame.ball.a.z)
        else:
            ball_p = ball_v = ball_a = (_NAN, _NAN, _NAN)

        referee = frame.referee
        tick_index = len(self._ticks)
        if referee is not None:
            has_referee = True
            referee_command = referee.referee_command.value
            stage = referee.stage.value
            designated_position = (
                referee.designated_position if referee.designated_position is not None else (_NAN, _NAN)
            )
            if self._changed(referee, frame.ts):
                # a copy: a referee may update its team info in place
                self._sparse_referee.append((tick_index, copy.deepcopy(referee)))
                self._last_stored_ts = frame.ts
        else:
            has_referee = False
            referee_command = -1
            stage = -1
            designated_position = (_NAN, _NAN)

        self._ticks.append(
            _TickRecord(
                ts=frame.ts,
                friendly=friendly,
                enemy=enemy,
                ball_p=ball_p,
                ball_v=ball_v,
                ball_a=ball_a,
                has_referee=has_referee,
                referee_command=referee_command,
                stage=stage,
                designated_position=designated_position,
            )
        )

        if self.replay_configs.checkpoint_every_s > 0:
            if self._last_checkpoint_ts is None:
                self._last_checkpoint_ts = frame.ts
            elif frame.ts - self._last_checkpoint_ts >= self.replay_configs.checkpoint_every_s:
                self._flush()
                self._last_checkpoint_ts = frame.ts

    def _changed(self, referee: RefereeData, ts: float) -> bool:
        """Whether `referee` differs from the last stored message carried forward to `ts`."""
        if not self._sparse_referee:
            return True
        expected = advance_clocks(self._sparse_referee[-1][1], ts - self._last_stored_ts)
        return not _same_referee(referee, expected)

    def _build_arrays(self) -> dict:
        n = len(self._ticks)
        friendly_ids = sorted({rid for t in self._ticks for rid in t.friendly})
        enemy_ids = sorted({rid for t in self._ticks for rid in t.enemy})

        friendly_p = np.full((n, len(friendly_ids), 2), _NAN)
        friendly_v = np.full((n, len(friendly_ids), 2), _NAN)
        friendly_a = np.full((n, len(friendly_ids), 2), _NAN)
        friendly_orientation = np.full((n, len(friendly_ids)), _NAN)
        friendly_has_ball = np.zeros((n, len(friendly_ids)), dtype=bool)

        enemy_p = np.full((n, len(enemy_ids), 2), _NAN)
        enemy_v = np.full((n, len(enemy_ids), 2), _NAN)
        enemy_a = np.full((n, len(enemy_ids), 2), _NAN)
        enemy_orientation = np.full((n, len(enemy_ids)), _NAN)
        enemy_has_ball = np.zeros((n, len(enemy_ids)), dtype=bool)

        ts = np.empty(n, dtype=np.float64)
        ball_p = np.empty((n, 3), dtype=np.float64)
        ball_v = np.empty((n, 3), dtype=np.float64)
        ball_a = np.empty((n, 3), dtype=np.float64)
        has_referee = np.empty(n, dtype=bool)
        referee_command = np.empty(n, dtype=np.int8)
        stage = np.empty(n, dtype=np.int8)
        designated_position = np.empty((n, 2), dtype=np.float64)

        friendly_slot = {rid: i for i, rid in enumerate(friendly_ids)}
        enemy_slot = {rid: i for i, rid in enumerate(enemy_ids)}

        for tick_i, t in enumerate(self._ticks):
            for rid, (px, py, vx, vy, ax, ay, orientation, has_ball) in t.friendly.items():
                slot = friendly_slot[rid]
                friendly_p[tick_i, slot] = (px, py)
                friendly_v[tick_i, slot] = (vx, vy)
                friendly_a[tick_i, slot] = (ax, ay)
                friendly_orientation[tick_i, slot] = orientation
                friendly_has_ball[tick_i, slot] = has_ball
            for rid, (px, py, vx, vy, ax, ay, orientation, has_ball) in t.enemy.items():
                slot = enemy_slot[rid]
                enemy_p[tick_i, slot] = (px, py)
                enemy_v[tick_i, slot] = (vx, vy)
                enemy_a[tick_i, slot] = (ax, ay)
                enemy_orientation[tick_i, slot] = orientation
                enemy_has_ball[tick_i, slot] = has_ball
            ts[tick_i] = t.ts
            ball_p[tick_i] = t.ball_p
            ball_v[tick_i] = t.ball_v
            ball_a[tick_i] = t.ball_a
            has_referee[tick_i] = t.has_referee
            referee_command[tick_i] = t.referee_command
            stage[tick_i] = t.stage
            designated_position[tick_i] = t.designated_position

        return {
            "my_team_is_yellow": np.array(self.metadata.my_team_is_yellow),
            "my_team_is_right": np.array(bool(self._my_team_is_right)),
            "friendly_ids": np.array(friendly_ids, dtype=np.int32),
            "enemy_ids": np.array(enemy_ids, dtype=np.int32),
            "ts": ts,
            "friendly_p": friendly_p,
            "friendly_v": friendly_v,
            "friendly_a": friendly_a,
            "friendly_orientation": friendly_orientation,
            "friendly_has_ball": friendly_has_ball,
            "enemy_p": enemy_p,
            "enemy_v": enemy_v,
            "enemy_a": enemy_a,
            "enemy_orientation": enemy_orientation,
            "enemy_has_ball": enemy_has_ball,
            "ball_p": ball_p,
            "ball_v": ball_v,
            "ball_a": ball_a,
            "has_referee": has_referee,
            "referee_command": referee_command,
            "stage": stage,
            "designated_position": designated_position,
        }

    def _flush(self) -> None:
        arrays = self._build_arrays()
        # `np.savez_compressed` always appends a literal ".npz" to whatever
        # path it's given (even if the path already ends in ".npz") — so the
        # tmp file is written under a name one level removed from `.npz`
        # entirely, then that exact file is renamed into place.
        tmp_stem = self.path.with_suffix("").as_posix() + ".tmp"
        np.savez_compressed(tmp_stem, **arrays)
        Path(tmp_stem + ".npz").replace(self.path)
        if self._sparse_referee:
            sidecar_path = self.path.with_suffix(".sparse_referee.pkl")
            tmp_sidecar = sidecar_path.with_suffix(".pkl.tmp")
            with open(tmp_sidecar, "wb") as f:
                pickle.dump(self._sparse_referee, f)
            tmp_sidecar.replace(sidecar_path)

    def close(self) -> None:
        self._flush()


_CLOCK_TOLERANCE_S = 1e-6


def _team_fields(team) -> Optional[dict]:
    return None if team is None else vars(team)


def _same_referee(a: RefereeData, b: RefereeData) -> bool:
    """Every field equal, clocks to within `_CLOCK_TOLERANCE_S`."""
    for f in dataclasses.fields(RefereeData):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if f.name in ("blue_team", "yellow_team"):
            x, y = _team_fields(x), _team_fields(y)
        if f.name in ("time_sent", "time_received", "stage_time_left"):
            if not math.isclose(x, y, abs_tol=_CLOCK_TOLERANCE_S):
                return False
        elif f.name == "current_action_time_remaining" and x is not None and y is not None:
            if abs(x - y) > 1:  # microseconds
                return False
        elif x != y:
            return False
    return True
