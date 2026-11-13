"""Ball, robot and frame builders shared by the custom-referee tests."""

from __future__ import annotations

from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot


def ball(x: float, y: float, z: float = 0.0, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0) -> Ball:
    return Ball(p=Vector3D(x, y, z), v=Vector3D(vx, vy, vz), a=Vector3D(0, 0, 0))


def robot(robot_id: int, x: float, y: float, is_friendly: bool, has_ball: bool = False) -> Robot:
    return Robot(
        id=robot_id,
        is_friendly=is_friendly,
        has_ball=has_ball,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=0.0,
    )


def frame(
    ball: Ball,
    friendly_robots: dict | None = None,
    enemy_robots: dict | None = None,
    my_team_is_yellow: bool = True,
    my_team_is_right: bool = False,
    ts: float = 10.0,
) -> GameFrame:
    return GameFrame(
        ts=ts,
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        friendly_robots=friendly_robots or {},
        enemy_robots=enemy_robots or {},
        ball=ball,
        referee=None,
    )
