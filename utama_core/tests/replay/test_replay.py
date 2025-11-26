from unittest.mock import patch

import pytest

from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, Robot
from utama_core.replay.replay_player import ReplayStandardSSL

# Example frame with non-sequential IDs
frame = GameFrame(
    ts=0.0,
    my_team_is_yellow=True,
    my_team_is_right=True,
    friendly_robots={
        1: Robot(
            id=1,
            is_friendly=True,
            has_ball=False,
            p=Vector2D(1, 1),
            v=Vector2D(0, 0),
            a=Vector2D(0, 0),
            orientation=0.0,
        )
    },
    enemy_robots={},
    ball=Ball(p=Vector3D(0, 0, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
)

second_frame = GameFrame(
    ts=0.1,
    my_team_is_yellow=True,
    my_team_is_right=True,
    friendly_robots={
        3: Robot(
            id=3,
            is_friendly=True,
            has_ball=False,
            p=Vector2D(1.1, 1.1),
            v=Vector2D(0, 0),
            a=Vector2D(0, 0),
            orientation=0.0,
        ),
        5: Robot(
            id=5,
            is_friendly=True,
            has_ball=False,
            p=Vector2D(2, 2),
            v=Vector2D(0, 0),
            a=Vector2D(0, 0),
            orientation=0.0,
        ),
    },
    enemy_robots={
        2: Robot(
            id=2,
            is_friendly=False,
            has_ball=False,
            p=Vector2D(-1, -1),
            v=Vector2D(0, 0),
            a=Vector2D(0, 0),
            orientation=0.0,
        )
    },
    ball=Ball(p=Vector3D(0, 0, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
)


@pytest.mark.parametrize("test_frame", [frame, second_frame])
def test_non_sequential_robot_ids(test_frame):
    n_yellow = len(test_frame.friendly_robots)
    n_blue = len(test_frame.enemy_robots)

    replay_env = ReplayStandardSSL(n_robots_yellow=n_yellow, n_robots_blue=n_blue)

    # Patch render to do nothing so we don't open a pygame window
    with patch.object(replay_env, "render", return_value=None):
        try:
            replay_env.step_replay(test_frame)
        except Exception as e:
            pytest.fail(f"Replay failed with non-sequential robot IDs: {e}")
