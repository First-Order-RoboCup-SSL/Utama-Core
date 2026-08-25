"""Regression test for the traced ball-contest deadlock (`docs/roadmap.md`,
Open item 11): a friendly and enemy robot pinned body-to-body around a ball
that neither dribbler ever registers (`has_ball=False` on both throughout —
see `robot_contact.py`'s docstring), each pushing into the other with
similar force, drifting in place rather than resolving.

That symptom was root-caused at the `FastPathPlanner`/`go_to_ball` level and
explicitly left unfixed there, then reframed via the SSL rulebook's
"Pushing" rule (§8.4.1: "if both robots are pushing each other with similar
force, no team is at fault") as a legitimate no-fault physical state that a
*referee*, not a planner, should resolve — by stopping play and restarting
with `FORCE_START` at the ball's position, same as the rulebook prescribes.
`PushingRule` (`rules/pushing_rule.py`) was built for exactly this. Its own
unit tests (`test_pushing_crashing.py`) confirm the rule *fires* correctly
in isolation, but per `docs/testing_gaps.md` gap #1/#6, nothing before this
file actually drove that firing through `CustomReferee.step()` — the real
call path production code uses — nor confirmed the resulting `STOP` command
actually causes the pinned pair to be physically separated by
`RefereeOverride`'s `StopStep`, rather than just being counted as a foul
with no effect on the still-pinned robots. This file closes both gaps for
this specific scenario.
"""

from __future__ import annotations

import math

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.referee_constants import BALL_KEEP_OUT_DISTANCE
from utama_core.custom_referee.custom_referee import CustomReferee
from utama_core.engine.referee_override import RefereeOverride
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand

# Robots placed exactly at PushingRule's contact distance (2*ROBOT_RADIUS +
# 0.03 = 0.18m) along the x-axis, ball trapped exactly between their fronts —
# the traced deadlock's geometry (t=40.23-46.17s of the original replay: a
# carrier held 0.11-0.34m from a stationary-ish ball, orbiting rather than
# closing, while an opponent pinned it from the other side).
_CONTACT_X = 0.18


def _ball(x: float = 0.0, y: float = 0.0) -> Ball:
    return Ball(p=Vector3D(x, y, 0.0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0))


def _robot(robot_id: int, x: float, y: float, is_friendly: bool, vx: float, vy: float = 0.0) -> Robot:
    return Robot(
        id=robot_id,
        is_friendly=is_friendly,
        has_ball=False,  # never true throughout — see robot_contact.py's docstring
        p=Vector2D(x, y),
        v=Vector2D(vx, vy),
        a=Vector2D(0, 0),
        orientation=0.0,
    )


def _pinned_pair_frame(ts: float) -> GameFrame:
    """Friendly and enemy robot pinned around a centred ball, each driving
    into the other at equal speed (symmetric-force, no-fault push), neither
    dribbler in contact — the exact shape `find_robot_pair_contacts` and
    `PushingRule` are built to recognise without `has_ball`."""
    friendly = _robot(0, 0.0, 0.0, is_friendly=True, vx=1.0)
    enemy = _robot(0, _CONTACT_X, 0.0, is_friendly=False, vx=-1.0)
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=False,
        friendly_robots={0: friendly},
        enemy_robots={0: enemy},
        ball=_ball(_CONTACT_X / 2.0, 0.0),
        referee=None,
    )


def test_custom_referee_step_detects_sustained_push_and_issues_stop():
    """Driven through the real `CustomReferee.step()` call path (not
    `PushingRule.check()` directly) with the "simulation" profile's actual
    configured thresholds (persistence_frames=15, min_closing_speed_mps=0.05,
    similar_force_margin_mps=0.15 — see `profiles/simulation.yaml`), the
    pinned-pair scenario must flip `referee_command` to `STOP` once contact
    has been sustained long enough, exactly as `PushingRule`'s no-fault
    "similar force" branch prescribes."""
    referee = CustomReferee.from_profile_name("simulation", n_robots_yellow=1, n_robots_blue=1)
    referee.seed_clock(0.0, initial_command=RefereeCommand.NORMAL_START)

    result = None
    for tick in range(20):
        ts = tick * (1.0 / 60.0)
        result = referee.step(_pinned_pair_frame(ts), current_time=ts)
        if result.referee_command == RefereeCommand.STOP:
            break

    assert result is not None
    assert result.referee_command == RefereeCommand.STOP, (
        "sustained symmetric-force contact must be detected via CustomReferee.step() "
        "and issue STOP, not merely be detectable via PushingRule.check() in isolation"
    )
    assert referee.last_violation is not None
    assert referee.last_violation.rule_name == "pushing"
    assert referee.last_violation.offending_teams == (), "similar-force push must not charge either team's foul counter"
    assert referee.last_violation.next_command == RefereeCommand.FORCE_START


def test_stop_step_physically_separates_the_pinned_pair():
    """Once STOP is in effect, `RefereeOverride`'s `StopStep` must actually
    drive the encroaching friendly robot outside the ball keep-out radius —
    proving the deadlock is physically broken, not just logged as a foul
    with the two robots left exactly where they were pinned."""
    from dataclasses import dataclass
    from dataclasses import field as dc_field

    from utama_core.entities.data.command import RobotCommand
    from utama_core.entities.game.field import FieldBounds

    @dataclass
    class _FakeField:
        top_left: tuple = (-4.5, 3.0)
        bottom_right: tuple = (4.5, -3.0)
        half_defense_area_depth: float = 0.9
        half_defense_area_width: float = 1.8

        @property
        def half_length(self) -> float:
            return 4.5

        @property
        def half_width(self) -> float:
            return 3.0

        @property
        def field_bounds(self) -> FieldBounds:
            return FieldBounds(top_left=self.top_left, bottom_right=self.bottom_right)

    @dataclass
    class _FakeReferee:
        designated_position: tuple | None = None

    @dataclass
    class _FakeGame:
        friendly_robots: dict
        my_team_is_yellow: bool = True
        my_team_is_right: bool = False
        ball: Ball = dc_field(default_factory=lambda: _ball(_CONTACT_X / 2.0, 0.0))
        referee: _FakeReferee = dc_field(default_factory=_FakeReferee)
        field: _FakeField = dc_field(default_factory=_FakeField)

    friendly = _robot(0, 0.0, 0.0, is_friendly=True, vx=1.0)
    game = _FakeGame(friendly_robots={0: friendly})

    captured_targets: dict[int, tuple] = {}

    def _fake_move(game, motion_controller, robot_id, target_coords, target_oren, dribbling=False) -> RobotCommand:
        captured_targets[robot_id] = (target_coords.x, target_coords.y)
        return RobotCommand(
            local_forward_vel=0.0, local_left_vel=0.0, angular_vel=0.0, kick=False, chip=False, dribble=False
        )

    import utama_core.custom_referee.actions as referee_actions

    original_move = referee_actions.move
    referee_actions.move = _fake_move
    try:
        override = RefereeOverride()
        cmd_map = override.tick(game, motion_controller=None, command=RefereeCommand.STOP)
    finally:
        referee_actions.move = original_move

    # The friendly robot started at distance CONTACT_X/2 (0.09m) from the
    # ball — well inside BALL_KEEP_OUT_DISTANCE (0.8m) — so StopStep must
    # actively drive it clear rather than leave it pinned in place.
    assert 0 in cmd_map and cmd_map[0] is not None
    assert 0 in captured_targets, "StopStep must issue a move command, not freeze the encroaching robot in place"
    tx, ty = captured_targets[0]
    ball = game.ball
    dist_to_ball = math.hypot(tx - ball.p.x, ty - ball.p.y)
    assert dist_to_ball >= BALL_KEEP_OUT_DISTANCE - 1e-6, (
        f"target {(tx, ty)} still inside the ball keep-out radius ({dist_to_ball:.3f}m < {BALL_KEEP_OUT_DISTANCE}m) "
        "— pinned robot was not actually driven clear"
    )
