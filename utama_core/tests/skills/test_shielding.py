"""Tests for `utama_core.skills.src.shielding`, extracted out of `go_to_ball.py`."""

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Field, Game, GameFrame, GameHistory, Robot
from utama_core.skills.src.shielding import (
    _RELEASE_RANGE,
    COMMIT_RANGE,
    CONTEST_RANGE,
    nearest_contesting_enemy,
    reset_shield_state,
    shielded_approach_angle,
)


@pytest.fixture(autouse=True)
def _clear_shield_commit_state():
    """`shielded_approach_angle`'s commit/release hysteresis (see
    `shielding.COMMIT_RANGE`'s docstring) is per-robot-id module state, so a
    stale commitment from one test could otherwise leak into the next test
    reusing the same robot id (every test here uses `robot_id=0`)."""
    reset_shield_state(0)
    reset_shield_state(1)
    yield
    reset_shield_state(0)
    reset_shield_state(1)


def _robot(rid: int, x: float, y: float, is_friendly: bool) -> Robot:
    return Robot(
        id=rid,
        is_friendly=is_friendly,
        has_ball=False,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=0.0,
    )


def _game(friendly: dict, enemy: dict, ball_xy: tuple, my_team_is_yellow: bool = True) -> Game:
    zv = Vector3D(0, 0, 0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=True,
        friendly_robots=friendly,
        enemy_robots=enemy,
        ball=Ball(p=Vector3D(ball_xy[0], ball_xy[1], 0), v=zv, a=zv),
    )
    return Game(
        past=GameHistory(10),
        current=frame,
        field=Field(
            my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
        ),
    )


def test_no_contesting_enemy_returns_none():
    game = _game({0: _robot(0, 0.0, 0.0, True)}, {}, (1.0, 0.0))
    assert nearest_contesting_enemy(game, Vector2D(1.0, 0.0)) is None


def test_enemy_outside_contest_range_is_ignored():
    game = _game(
        {0: _robot(0, 0.0, 0.0, True)},
        {0: _robot(0, 1.0 + CONTEST_RANGE + 1.0, 0.0, False)},
        (1.0, 0.0),
    )
    assert nearest_contesting_enemy(game, Vector2D(1.0, 0.0)) is None


def test_enemy_inside_contest_range_is_returned():
    enemy_pos = Vector2D(1.0 + CONTEST_RANGE / 2, 0.0)
    game = _game({0: _robot(0, 0.0, 0.0, True)}, {0: _robot(0, enemy_pos.x, enemy_pos.y, False)}, (1.0, 0.0))
    result = nearest_contesting_enemy(game, Vector2D(1.0, 0.0))
    assert result == enemy_pos


def test_nearest_contesting_enemy_picks_closest_of_several():
    ball = Vector2D(0.0, 0.0)
    far = Vector2D(0.4, 0.0)
    near = Vector2D(0.1, 0.0)
    game = _game(
        {0: _robot(0, 5.0, 5.0, True)},
        {0: _robot(0, far.x, far.y, False), 1: _robot(1, near.x, near.y, False)},
        (0.0, 0.0),
    )
    assert nearest_contesting_enemy(game, ball) == near


def test_shielded_approach_angle_shields_when_enemy_contesting_and_far_from_ball():
    ball = Vector2D(0.0, 0.0)
    robot = Vector2D(-5.0, 0.0)  # far from ball, beyond COMMIT_RANGE
    enemy = Vector2D(0.3, 0.0)  # within CONTEST_RANGE
    game = _game({0: _robot(0, robot.x, robot.y, True)}, {0: _robot(0, enemy.x, enemy.y, False)}, (0.0, 0.0))

    angle, shielding = shielded_approach_angle(game, robot, ball, robot_id=0)

    assert shielding is True
    assert angle == pytest.approx(enemy.angle_to(ball))


def test_shielded_approach_angle_commits_direct_once_close_to_ball():
    ball = Vector2D(0.0, 0.0)
    robot = Vector2D(COMMIT_RANGE / 2, 0.0)  # inside COMMIT_RANGE
    enemy = Vector2D(0.3, 0.3)  # still within CONTEST_RANGE
    game = _game({0: _robot(0, robot.x, robot.y, True)}, {0: _robot(0, enemy.x, enemy.y, False)}, (0.0, 0.0))

    angle, shielding = shielded_approach_angle(game, robot, ball, robot_id=0)

    assert shielding is False
    assert angle == pytest.approx(robot.angle_to(ball))


def test_shielded_approach_angle_direct_when_no_contesting_enemy():
    ball = Vector2D(0.0, 0.0)
    robot = Vector2D(-5.0, 0.0)
    game = _game({0: _robot(0, robot.x, robot.y, True)}, {}, (0.0, 0.0))

    angle, shielding = shielded_approach_angle(game, robot, ball, robot_id=0)

    assert shielding is False
    assert angle == pytest.approx(robot.angle_to(ball))


def test_shielded_approach_angle_does_not_rechatter_near_commit_boundary():
    """Regression test for
    `docs/investigation_ball_contact_orientation_divergence.md`: a robot
    whose chassis distance to the ball wobbles back and forth across
    `COMMIT_RANGE` (routine near the physical contact radius — see
    `COMMIT_RANGE`'s docstring) must not flip back to shielding until it has
    backed off past `_RELEASE_RANGE`. Without hysteresis, each re-crossing
    re-triggers a 100+ degree `target_oren` jump, which is what the
    investigation traced as the divergence's proximate cause."""
    ball = Vector2D(0.0, 0.0)
    enemy = Vector2D(0.3, 0.3)  # within CONTEST_RANGE, contesting throughout

    def _shielding_at(dist: float) -> bool:
        robot = Vector2D(dist, 0.0)
        game = _game({0: _robot(0, robot.x, robot.y, True)}, {0: _robot(0, enemy.x, enemy.y, False)}, (0.0, 0.0))
        _, shielding = shielded_approach_angle(game, robot, ball, robot_id=0)
        return shielding

    # Approach from far away: shielding, as before.
    assert _shielding_at(CONTEST_RANGE - 0.01) is True
    # Cross into COMMIT_RANGE: commits to direct, as before.
    assert _shielding_at(COMMIT_RANGE - 0.005) is False
    # Wobble back out just past COMMIT_RANGE (routine near the contact
    # radius) — must NOT re-engage shielding; that's the bug.
    assert _shielding_at(COMMIT_RANGE + 0.01) is False
    assert _shielding_at(COMMIT_RANGE + 0.03) is False
    # Only once genuinely clear of the ball (past _RELEASE_RANGE) does
    # shielding re-engage.
    assert _shielding_at(_RELEASE_RANGE + 0.01) is True


def test_shielded_approach_angle_survives_a_transient_enemy_exit_from_contest_range():
    """Regression test for a live 2026-09-02 tournament finding
    (`counter_flow_vs_tiki_taka.pkl`, t=14.7-15.4s): a contesting enemy
    routinely steps in and out of `CONTEST_RANGE` on a contested midfield
    loose ball, not just once. Committing to a direct approach, then having
    the enemy step out and back in, must not restart the hysteresis from
    scratch (which would re-run the tight `dist > COMMIT_RANGE` check
    instead of the wide `_RELEASE_RANGE` check on re-entry) -- that
    reproduced the exact approach/retreat oscillation `_RELEASE_RANGE` was
    added to prevent in the first place, just gated by the enemy's in/out
    timing instead of our own distance wobble."""
    ball = Vector2D(0.0, 0.0)

    def _shielding_at(dist: float, enemy: Vector2D | None) -> bool:
        robot = Vector2D(dist, 0.0)
        enemy_robots = {0: _robot(0, enemy.x, enemy.y, False)} if enemy is not None else {}
        game = _game({0: _robot(0, robot.x, robot.y, True)}, enemy_robots, (0.0, 0.0))
        _, shielding = shielded_approach_angle(game, robot, ball, robot_id=0)
        return shielding

    # Commit to direct while still contested.
    assert _shielding_at(COMMIT_RANGE - 0.005, Vector2D(0.3, 0.3)) is False
    # Robot itself stays well within _RELEASE_RANGE, but the contesting
    # enemy transiently steps outside CONTEST_RANGE (no enemy at all here).
    assert _shielding_at(COMMIT_RANGE + 0.05, None) is False
    # Enemy re-enters contest range -- must still honor the earlier
    # commitment (release check), not restart at the tight commit check.
    assert _shielding_at(COMMIT_RANGE + 0.05, Vector2D(0.3, 0.3)) is False


def test_shielded_approach_angle_forgets_commitment_once_genuinely_clear_of_the_ball():
    """The no-contesting-enemy branch must still eventually drop a stale
    commitment -- once this robot is far past `_RELEASE_RANGE` from the
    ball (a wholly separate, later approach), it must not silently inherit
    an old commitment from a previous, unrelated ball chase. Nothing in the
    production codebase calls `reset_shield_state` today, so this distance-
    based fallback is the only thing preventing that leak."""
    ball = Vector2D(0.0, 0.0)
    enemy = Vector2D(0.3, 0.3)

    def _shielding_at(dist: float, contested: bool) -> bool:
        robot = Vector2D(dist, 0.0)
        enemy_robots = {0: _robot(0, enemy.x, enemy.y, False)} if contested else {}
        game = _game({0: _robot(0, robot.x, robot.y, True)}, enemy_robots, (0.0, 0.0))
        _, shielding = shielded_approach_angle(game, robot, ball, robot_id=0)
        return shielding

    # Commit to direct while contested.
    assert _shielding_at(COMMIT_RANGE - 0.005, contested=True) is False
    # Robot moves well clear of the ball with no enemy contesting -- a
    # genuinely separate later approach, not a transient enemy blip.
    assert _shielding_at(_RELEASE_RANGE + 1.0, contested=False) is False
    # A fresh contest from far away should shield again, not stay
    # incorrectly committed to direct from the earlier, unrelated approach.
    assert _shielding_at(CONTEST_RANGE - 0.01, contested=True) is True


def test_shielded_approach_angle_commit_state_is_per_robot():
    """Two robots' hysteresis states must not interfere with each other."""
    ball = Vector2D(0.0, 0.0)
    enemy = Vector2D(0.3, 0.3)

    def _shielding_for(robot_id: int, dist: float) -> bool:
        robot = Vector2D(dist, 0.0)
        game = _game(
            {robot_id: _robot(robot_id, robot.x, robot.y, True)}, {0: _robot(0, enemy.x, enemy.y, False)}, (0.0, 0.0)
        )
        _, shielding = shielded_approach_angle(game, robot, ball, robot_id=robot_id)
        return shielding

    # Robot 0 commits direct.
    assert _shielding_for(0, COMMIT_RANGE - 0.005) is False
    # Robot 1, still far away, should independently still shield.
    assert _shielding_for(1, CONTEST_RANGE - 0.01) is True


def _shielding(dist: float, my_team_is_yellow: bool = True) -> bool:
    """Robot 0 at `dist` from the ball, an enemy contesting it throughout."""
    game = _game(
        {0: _robot(0, dist, 0.0, True)},
        {0: _robot(0, 0.3, 0.3, False)},
        (0.0, 0.0),
        my_team_is_yellow=my_team_is_yellow,
    )
    return shielded_approach_angle(game, Vector2D(dist, 0.0), Vector2D(0.0, 0.0), robot_id=0)[1]


def test_commitment_is_per_team():
    """Both teams run in one process with the same robot ids: yellow robot 0 committing to a
    direct approach must not stop blue robot 0 shielding."""
    assert _shielding(COMMIT_RANGE - 0.005) is False  # yellow commits

    assert _shielding(COMMIT_RANGE + 0.01, my_team_is_yellow=False) is True
    assert _shielding(COMMIT_RANGE + 0.01) is False


def test_reset_shield_state_with_no_robot_forgets_every_robot():
    """A new match starts uncommitted (the runner calls this; a round-robin worker process plays
    many matches)."""
    assert _shielding(COMMIT_RANGE - 0.005) is False

    reset_shield_state()

    assert _shielding(COMMIT_RANGE + 0.01) is True
