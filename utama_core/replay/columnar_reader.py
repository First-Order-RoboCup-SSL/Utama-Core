"""Reader for the columnar replay format written by `ColumnarReplayWriter`.

Loading is a single `np.load` — no per-frame object reconstruction. Bulk/
vectorized consumers (the stuck detector, dashboards, any stats pass)
should work directly against the returned `ColumnarReplay`'s arrays rather
than building `GameFrame`s at all. `frame_at`/`iter_frames` exist only for
callers that genuinely need `GameFrame`-shaped objects (e.g. `render_window`
drawing a specific tick) and build them lazily, one at a time, so the cost
scales with how many ticks are actually inspected rather than the whole
replay.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np

from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, Robot
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage


@dataclass
class ColumnarReplay:
    my_team_is_yellow: bool
    my_team_is_right: bool
    friendly_ids: np.ndarray
    enemy_ids: np.ndarray
    ts: np.ndarray
    friendly_p: np.ndarray  # (n_ticks, n_friendly, 2)
    friendly_v: np.ndarray
    friendly_a: np.ndarray
    friendly_orientation: np.ndarray  # (n_ticks, n_friendly)
    friendly_has_ball: np.ndarray
    enemy_p: np.ndarray
    enemy_v: np.ndarray
    enemy_a: np.ndarray
    enemy_orientation: np.ndarray
    enemy_has_ball: np.ndarray
    ball_p: np.ndarray  # (n_ticks, 3)
    ball_v: np.ndarray
    ball_a: np.ndarray
    has_referee: np.ndarray  # (n_ticks,) bool
    referee_command: np.ndarray  # (n_ticks,) int8, -1 == no referee data
    stage: np.ndarray
    designated_position: np.ndarray  # (n_ticks, 2), NaN when absent
    sparse_referee: dict[int, RefereeData]  # tick index -> full RefereeData, rare ticks only

    @property
    def n_ticks(self) -> int:
        return len(self.ts)

    def frame_at(self, tick: int) -> GameFrame:
        """Reconstruct the `GameFrame` for a single tick, on demand."""
        friendly_robots = _robots_at(
            self.friendly_ids,
            True,
            self.friendly_p[tick],
            self.friendly_v[tick],
            self.friendly_a[tick],
            self.friendly_orientation[tick],
            self.friendly_has_ball[tick],
        )
        enemy_robots = _robots_at(
            self.enemy_ids,
            False,
            self.enemy_p[tick],
            self.enemy_v[tick],
            self.enemy_a[tick],
            self.enemy_orientation[tick],
            self.enemy_has_ball[tick],
        )
        ball = Ball(
            p=Vector3D(*self.ball_p[tick]),
            v=Vector3D(*self.ball_v[tick]),
            a=Vector3D(*self.ball_a[tick]),
        )
        referee = self._referee_at(tick)
        return GameFrame(
            ts=float(self.ts[tick]),
            my_team_is_yellow=self.my_team_is_yellow,
            my_team_is_right=self.my_team_is_right,
            friendly_robots=friendly_robots,
            enemy_robots=enemy_robots,
            ball=ball,
            referee=referee,
        )

    def _referee_at(self, tick: int) -> Optional[RefereeData]:
        if not self.has_referee[tick]:
            return None
        if tick in self.sparse_referee:
            return self.sparse_referee[tick]
        designated = self.designated_position[tick]
        return RefereeData(
            source_identifier=None,
            time_sent=0.0,
            time_received=0.0,
            referee_command=RefereeCommand.from_id(int(self.referee_command[tick])),
            referee_command_timestamp=0.0,
            stage=Stage(int(self.stage[tick])),
            stage_time_left=0.0,
            blue_team=None,
            yellow_team=None,
            designated_position=None if np.isnan(designated[0]) else tuple(designated),
        )

    def iter_frames(self) -> Iterator[GameFrame]:
        for tick in range(self.n_ticks):
            yield self.frame_at(tick)

    def frames_in_range(self, t_start: float, t_end: float) -> list[GameFrame]:
        """Equivalent of `replay_player.load_frames_in_range`, without
        materializing every tick outside the window."""
        mask = (self.ts >= t_start) & (self.ts <= t_end)
        return [self.frame_at(tick) for tick in np.flatnonzero(mask)]


def _robots_at(ids, is_friendly, p_row, v_row, a_row, orientation_row, has_ball_row) -> dict[int, Robot]:
    robots = {}
    for i, rid in enumerate(ids):
        if np.isnan(p_row[i, 0]):
            continue
        robots[int(rid)] = Robot(
            id=int(rid),
            is_friendly=is_friendly,
            has_ball=bool(has_ball_row[i]),
            p=Vector2D(*p_row[i]),
            v=Vector2D(*v_row[i]),
            a=Vector2D(*a_row[i]),
            orientation=float(orientation_row[i]),
        )
    return robots


def load_columnar_replay(path: Union[str, Path]) -> ColumnarReplay:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    sparse_referee: dict[int, RefereeData] = {}
    sidecar_path = path.with_suffix(".sparse_referee.pkl")
    if sidecar_path.exists():
        with open(sidecar_path, "rb") as f:
            sparse_referee = dict(pickle.load(f))

    return ColumnarReplay(
        my_team_is_yellow=bool(arrays["my_team_is_yellow"]),
        my_team_is_right=bool(arrays["my_team_is_right"]),
        friendly_ids=arrays["friendly_ids"],
        enemy_ids=arrays["enemy_ids"],
        ts=arrays["ts"],
        friendly_p=arrays["friendly_p"],
        friendly_v=arrays["friendly_v"],
        friendly_a=arrays["friendly_a"],
        friendly_orientation=arrays["friendly_orientation"],
        friendly_has_ball=arrays["friendly_has_ball"],
        enemy_p=arrays["enemy_p"],
        enemy_v=arrays["enemy_v"],
        enemy_a=arrays["enemy_a"],
        enemy_orientation=arrays["enemy_orientation"],
        enemy_has_ball=arrays["enemy_has_ball"],
        ball_p=arrays["ball_p"],
        ball_v=arrays["ball_v"],
        ball_a=arrays["ball_a"],
        has_referee=arrays["has_referee"],
        referee_command=arrays["referee_command"],
        stage=arrays["stage"],
        designated_position=arrays["designated_position"],
        sparse_referee=sparse_referee,
    )
