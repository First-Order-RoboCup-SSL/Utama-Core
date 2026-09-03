"""Build a real (non-Mock) `Game`/`GameFrame` by hand, for pure-logic tests that
need actual entity objects but no rsim simulator.

`robot_with_ball`/`has_ball` semantics: the real pipeline (see
`CurrentGameFrame._set_robot_with_ball`, `utama_core/entities/game/
current_game_frame.py`) derives `game.robot_with_ball` purely from
`Robot.has_ball` on whichever robots are passed in -- it scans friendly
robots first, then enemy, and returns the first one with `has_ball=True`
(`None` if none). This module does not reimplement that rule; it just sets
`Robot.has_ball` per the `has_ball_robot` parameter below and lets the real
`Game`/`CurrentGameFrame` constructor derive `robot_with_ball` exactly as
production code would. There is no separate "set robot_with_ball directly"
path -- match the real pipeline by setting `has_ball_robot` instead.

`random_game`'s "legal" random state means: every robot (friendly + enemy)
lies inside the field bounds (with a small margin) and no two robots
(regardless of team) are closer than one robot diameter apart. The ball is
placed uniformly at random inside the field. Nothing here validates
referee-legality (defense-area occupancy etc.) -- callers needing that
should filter/adjust after the fact.
"""

from __future__ import annotations

import random as random_module
from typing import Dict, Optional

from utama_core.config.field_params import STANDARD_FIELD_DIMS, FieldDimensions
from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.field import Field
from utama_core.entities.game.game import Game
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.game_history import GameHistory
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage

_ZERO2 = Vector2D(0.0, 0.0)
_ZERO3 = Vector3D(0.0, 0.0, 0.0)
_ROBOT_DIAMETER = 2.0 * ROBOT_RADIUS


def make_robot(
    robot_id: int,
    *,
    is_friendly: bool,
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    orientation: float = 0.0,
    has_ball: bool = False,
) -> Robot:
    """A real `Robot` dataclass instance -- position/velocity/orientation as given."""
    return Robot(
        id=robot_id,
        is_friendly=is_friendly,
        has_ball=has_ball,
        p=Vector2D(x, y),
        v=Vector2D(vx, vy),
        a=_ZERO2,
        orientation=orientation,
    )


def make_ball(x: float = 0.0, y: float = 0.0, vx: float = 0.0, vy: float = 0.0) -> Ball:
    return Ball(p=Vector3D(x, y, 0.0), v=Vector3D(vx, vy, 0.0), a=_ZERO3)


def _make_referee_data(command: RefereeCommand) -> RefereeData:
    from utama_core.entities.game.team_info import TeamInfo

    empty_team = TeamInfo(name="", score=0, goalkeeper=0)
    return RefereeData(
        source_identifier=None,
        time_sent=0.0,
        time_received=0.0,
        referee_command=command,
        referee_command_timestamp=0.0,
        stage=Stage.NORMAL_FIRST_HALF,
        stage_time_left=0.0,
        blue_team=empty_team,
        yellow_team=empty_team,
    )


def build_game(
    friendly_robots: Dict[int, Robot],
    enemy_robots: Dict[int, Robot],
    ball: Optional[Ball] = None,
    *,
    my_team_is_right: bool = True,
    my_team_is_yellow: bool = True,
    referee_command: Optional[RefereeCommand] = None,
    has_ball_robot: Optional[tuple] = None,
    field_dims: FieldDimensions = STANDARD_FIELD_DIMS,
    ts: float = 0.0,
    history_len: int = 10,
) -> Game:
    """Build a real `Game` wrapping a real `GameFrame`.

    `has_ball_robot`, if given, is `("friendly" | "enemy", robot_id)` --
    convenience for setting exactly one robot's `has_ball=True` (mirroring
    the "only one robot can have the ball" assumption `CurrentGameFrame`
    itself documents) without hand-editing the input dicts. `game.
    robot_with_ball` is then derived from that by the real `CurrentGameFrame`
    constructor, not set directly.
    """
    friendly_robots = dict(friendly_robots)
    enemy_robots = dict(enemy_robots)
    if has_ball_robot is not None:
        team, robot_id = has_ball_robot
        target = friendly_robots if team == "friendly" else enemy_robots
        target[robot_id] = _with_has_ball(target[robot_id], True)

    if ball is None:
        ball = make_ball()

    referee = _make_referee_data(referee_command) if referee_command is not None else None

    frame = GameFrame(
        ts=ts,
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        friendly_robots=friendly_robots,
        enemy_robots=enemy_robots,
        ball=ball,
        referee=referee,
    )
    field = Field(my_team_is_right=my_team_is_right, field_dims=field_dims, field_bounds=field_dims.full_field_bounds)
    return Game(past=GameHistory(history_len), current=frame, field=field)


def _with_has_ball(robot: Robot, has_ball: bool) -> Robot:
    import dataclasses

    return dataclasses.replace(robot, has_ball=has_ball)


def random_game(
    seed: int,
    n_friendly: int,
    n_enemy: int,
    field_dims: FieldDimensions = STANDARD_FIELD_DIMS,
    *,
    my_team_is_right: bool = True,
    my_team_is_yellow: bool = True,
    margin: float = 0.2,
) -> Game:
    """A legal random `Game`: robots inside the field (minus `margin`), no two
    robots (friendly or enemy) closer than one robot diameter, ball placed
    uniformly at random inside the field. Deterministic for a given `seed`.

    Robot ids are `0..n_friendly-1` (friendly) and `0..n_enemy-1` (enemy) --
    matching the real pipeline's per-team id space (`current_game_frame.py`'s
    `ObjectKey` scoping is per-`TeamType`, so ids may collide across teams).
    """
    rng = random_module.Random(seed)
    half_l = field_dims.full_field_half_length - margin
    half_w = field_dims.full_field_half_width - margin

    placed: list[Vector2D] = []

    def _sample_free_point() -> Vector2D:
        for _ in range(200):
            candidate = Vector2D(rng.uniform(-half_l, half_l), rng.uniform(-half_w, half_w))
            if all(candidate.distance_to(p) >= _ROBOT_DIAMETER for p in placed):
                return candidate
        # Field too crowded for the requested robot count at this margin --
        # fall back to the candidate anyway rather than looping forever.
        return candidate

    friendly_robots: Dict[int, Robot] = {}
    for rid in range(n_friendly):
        p = _sample_free_point()
        placed.append(p)
        friendly_robots[rid] = make_robot(
            rid,
            is_friendly=True,
            x=p.x,
            y=p.y,
            vx=rng.uniform(-1.0, 1.0),
            vy=rng.uniform(-1.0, 1.0),
            orientation=rng.uniform(-3.14159, 3.14159),
        )

    enemy_robots: Dict[int, Robot] = {}
    for rid in range(n_enemy):
        p = _sample_free_point()
        placed.append(p)
        enemy_robots[rid] = make_robot(
            rid,
            is_friendly=False,
            x=p.x,
            y=p.y,
            vx=rng.uniform(-1.0, 1.0),
            vy=rng.uniform(-1.0, 1.0),
            orientation=rng.uniform(-3.14159, 3.14159),
        )

    ball = make_ball(
        x=rng.uniform(-half_l, half_l),
        y=rng.uniform(-half_w, half_w),
        vx=rng.uniform(-2.0, 2.0),
        vy=rng.uniform(-2.0, 2.0),
    )

    return build_game(
        friendly_robots,
        enemy_robots,
        ball,
        my_team_is_right=my_team_is_right,
        my_team_is_yellow=my_team_is_yellow,
        field_dims=field_dims,
    )
