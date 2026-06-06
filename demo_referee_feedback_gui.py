"""Preview the referee web UI controller-feedback panel without hardware.

Run:
    pixi run python demo_referee_feedback_gui.py
    # open http://localhost:8080

This starts the CustomReferee browser UI and injects fake raw controller-port
feedback rows so the "Controller Feedback" section can be inspected without
real robots, serial hardware, grSim, or RSim.
"""

from __future__ import annotations

import argparse
import math
import time

from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.profiles.profile_loader import load_profile
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand


def _robot(
    robot_id: int,
    *,
    is_friendly: bool,
    has_ball: bool,
    x: float,
    y: float,
    orientation: float,
) -> Robot:
    zero = Vector2D(0.0, 0.0)
    return Robot(
        id=robot_id,
        is_friendly=is_friendly,
        has_ball=has_ball,
        p=Vector2D(x, y),
        v=zero,
        a=zero,
        orientation=orientation,
    )


def _ball(x: float, y: float) -> Ball:
    return Ball(
        p=Vector3D(x, y, 0.0),
        v=Vector3D(0.0, 0.0, 0.0),
        a=Vector3D(0.0, 0.0, 0.0),
    )


def _preview_frame(now: float, elapsed: float) -> GameFrame:
    ball_x = 0.25 * math.sin(elapsed * 0.8)
    ball_y = 0.15 * math.cos(elapsed * 0.8)
    friendly_has_ball = elapsed % 8.0 < 4.0

    return GameFrame(
        ts=now,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={
            0: _robot(0, is_friendly=True, has_ball=friendly_has_ball, x=0.6, y=0.0, orientation=math.pi),
            1: _robot(1, is_friendly=True, has_ball=False, x=1.2, y=0.7, orientation=-math.pi / 2),
        },
        enemy_robots={
            0: _robot(0, is_friendly=False, has_ball=False, x=-0.8, y=-0.5, orientation=0.0),
        },
        ball=_ball(ball_x, ball_y),
        referee=None,
    )


def _preview_feedback(elapsed: float) -> list[dict]:
    robot_zero_has_ball = elapsed % 8.0 < 4.0
    robot_one_connected = elapsed % 10.0 < 7.0
    stale_age = 0.18 if robot_one_connected else 1.25

    return [
        {
            "port_id": 0,
            "has_ball": robot_zero_has_ball,
            "connected": True,
            "age_seconds": 0.03,
            "team": "friendly",
            "team_color": "yellow",
            "vision_id": 0,
        },
        {
            "port_id": 1,
            "has_ball": False,
            "connected": robot_one_connected,
            "age_seconds": stale_age,
            "team": "friendly",
            "team_color": "yellow",
            "vision_id": 1,
        },
        {
            "port_id": 12,
            "has_ball": False,
            "connected": False,
            "age_seconds": 1.6,
            "team": "enemy",
            "team_color": "blue",
            "vision_id": 0,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the referee GUI controller-feedback panel.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port for the referee GUI.")
    args = parser.parse_args()

    referee = CustomReferee(
        load_profile("human"),
        n_robots_yellow=2,
        n_robots_blue=1,
        enable_gui=True,
        gui_port=args.port,
    )

    start = time.time()
    referee.seed_clock(start, RefereeCommand.FORCE_START)
    print(f"Preview running. Open http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            now = time.time()
            elapsed = now - start
            referee.set_robot_feedback_data(_preview_feedback(elapsed))
            referee.step(_preview_frame(now, elapsed), now)
            time.sleep(1 / 15)
    except KeyboardInterrupt:
        print("\nPreview stopped.")


if __name__ == "__main__":
    main()
