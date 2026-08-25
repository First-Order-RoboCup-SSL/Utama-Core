"""Integration-shaped tests for the 6 remaining §8.4 rules (Pushing already
covered by `test_ball_contest_deadlock.py`), driven through the real
`CustomReferee.step()` call path rather than `SomeRule().check(...)`
directly — see `docs/testing_gaps.md` gap #1: a rule's own unit tests all
called `check()` with the args that rule's author wrote against, so a
signature drift in `BaseRule.check()` (a 4th `designated_position` param
added mid-session) passed every rule's own suite and was only caught by an
unrelated full-suite run. One test per rule below proves it is correctly
wired into `CustomReferee.step()` — the actual call path production code
uses — using each rule's real "simulation" profile thresholds
(`profiles/simulation.yaml`), not made-up ones.
"""

from __future__ import annotations

from utama_core.custom_referee.custom_referee import CustomReferee
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand


def _ball(x: float = 0.0, y: float = 0.0, vx: float = 0.0, vy: float = 0.0) -> Ball:
    return Ball(p=Vector3D(x, y, 0.0), v=Vector3D(vx, vy, 0.0), a=Vector3D(0, 0, 0))


def _robot(
    robot_id: int,
    x: float,
    y: float,
    is_friendly: bool,
    vx: float = 0.0,
    vy: float = 0.0,
    has_ball: bool = False,
) -> Robot:
    return Robot(
        id=robot_id,
        is_friendly=is_friendly,
        has_ball=has_ball,
        p=Vector2D(x, y),
        v=Vector2D(vx, vy),
        a=Vector2D(0, 0),
        orientation=0.0,
    )


def _frame(
    ts: float,
    ball: Ball,
    friendly_robots: dict | None = None,
    enemy_robots: dict | None = None,
    my_team_is_yellow: bool = True,
    my_team_is_right: bool = False,
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


# ---------------------------------------------------------------------------
# CrashingRule — §8.4.2, non-stopping, edge-triggered on the tick contact
# first begins. fault_speed_threshold_mps=1.5 (simulation.yaml).
# ---------------------------------------------------------------------------


def test_crashing_rule_fires_through_custom_referee_step():
    referee = CustomReferee.from_profile_name("simulation", n_robots_yellow=1, n_robots_blue=1)
    referee.seed_clock(0.0, initial_command=RefereeCommand.NORMAL_START)

    # Tick 0: robots far apart, no contact yet.
    friendly = _robot(0, 0.0, 0.0, is_friendly=True, vx=2.0)
    enemy = _robot(0, 1.0, 0.0, is_friendly=False, vx=0.0)
    result = referee.step(
        _frame(0.0, _ball(0.5, 0.0), {0: friendly}, {0: enemy}),
        current_time=0.0,
    )
    assert referee.last_violation is None

    # Tick 1: friendly has closed to contact distance (0.18m) at 2.0 m/s
    # closing, stationary enemy — closing-speed difference magnitude 2.0
    # m/s > 1.5 threshold, so friendly (the closing/faster robot) fouls.
    friendly = _robot(0, 0.0, 0.0, is_friendly=True, vx=2.0)
    enemy = _robot(0, 0.18, 0.0, is_friendly=False, vx=0.0)
    result = referee.step(
        _frame(1.0 / 60.0, _ball(0.09, 0.0), {0: friendly}, {0: enemy}),
        current_time=1.0 / 60.0,
    )

    assert referee.last_violation is not None
    assert referee.last_violation.rule_name == "crashing"
    assert referee.last_violation.is_stopping is False
    assert referee.last_violation.offending_teams == (True,)  # friendly=yellow is the faster/closing robot
    # Non-stopping foul: command must NOT change out of NORMAL_START.
    assert result.referee_command == RefereeCommand.NORMAL_START


# ---------------------------------------------------------------------------
# KeeperHeldBallRule — §8.4.1. max_hold_seconds=10.0 (simulation.yaml).
# ---------------------------------------------------------------------------


def test_keeper_held_ball_rule_fires_through_custom_referee_step():
    referee = CustomReferee.from_profile_name("simulation", n_robots_yellow=1, n_robots_blue=1)
    referee.seed_clock(0.0, initial_command=RefereeCommand.NORMAL_START)

    # Ball resting inside yellow's own defense area the whole time (friendly
    # is yellow, my_team_is_right=False -> own defense area is on the left).
    # No robot needs to be present; the rule is purely geometric on the ball.
    # Checked at the tick it fires, not after — `last_violation` reverts to
    # None on the very next clean tick, so asserting only after the whole
    # loop would just observe stale state.
    ball_pos = (-4.0, 0.0)
    violation = None
    result = None
    for ts in (0.0, 3.0, 6.0, 9.0, 10.1, 10.5):
        result = referee.step(_frame(ts, _ball(*ball_pos)), current_time=ts)
        if referee.last_violation is not None:
            violation = referee.last_violation
            break

    assert violation is not None
    assert violation.rule_name == "keeper_held_ball"
    assert violation.offending_teams == (True,)  # yellow's own area held too long
    assert violation.next_command == RefereeCommand.DIRECT_FREE_BLUE
    # designated_position is set (ball's own position), so the foul routes
    # through ball placement before the free kick, and with no robot present
    # to keep clear-checking pending, STOP auto-advances to BALL_PLACEMENT_BLUE
    # within this same tick — see GameStateMachine._handle_foul/_all_robots_clear.
    assert result.referee_command == RefereeCommand.BALL_PLACEMENT_BLUE


# ---------------------------------------------------------------------------
# ExcessiveDribblingRule — §8.4.1. max_dribble_meters=1.0 (simulation.yaml).
# ---------------------------------------------------------------------------


def test_excessive_dribbling_rule_fires_through_custom_referee_step():
    referee = CustomReferee.from_profile_name("simulation", n_robots_yellow=1, n_robots_blue=1)
    referee.seed_clock(0.0, initial_command=RefereeCommand.NORMAL_START)

    # Friendly robot 0 keeps has_ball=True while the ball moves >1m from
    # where the dribble streak began.
    result = None
    for ts, bx in ((0.0, 0.0), (0.5, 0.3), (1.0, 0.6), (1.5, 1.2)):
        friendly = _robot(0, bx, 0.0, is_friendly=True, has_ball=True)
        result = referee.step(_frame(ts, _ball(bx, 0.0), {0: friendly}), current_time=ts)

    assert referee.last_violation is not None
    assert referee.last_violation.rule_name == "excessive_dribbling"
    assert referee.last_violation.offending_teams == (True,)
    assert referee.last_violation.next_command == RefereeCommand.DIRECT_FREE_BLUE
    assert result.referee_command == RefereeCommand.STOP


# ---------------------------------------------------------------------------
# RobotStopSpeedRule — §8.4.3, non-stopping. max_speed_mps=1.5,
# grace_seconds=2.0 (simulation.yaml).
# ---------------------------------------------------------------------------


def test_robot_stop_speed_rule_fires_through_custom_referee_step():
    referee = CustomReferee.from_profile_name("simulation", n_robots_yellow=1, n_robots_blue=1)
    referee.seed_clock(0.0, initial_command=RefereeCommand.NORMAL_START)
    referee.force_command(RefereeCommand.STOP, 0.0)

    violation = None
    result = None
    # RobotStopSpeedRule starts its own grace clock from the first tick it
    # observes STOP (game_frame.ts of the first step() call below, not
    # force_command's timestamp) — so the grace window here runs 0.5s to
    # 2.5s, not 0.0s to 2.0s. Past that, friendly robot still moving at
    # 2.0 m/s > 1.5 m/s max_speed_mps.
    for ts in (0.5, 1.0, 1.5, 2.1, 2.6, 2.7):
        friendly = _robot(0, ts, 0.0, is_friendly=True, vx=2.0)
        result = referee.step(_frame(ts, _ball(5.0, 5.0), {0: friendly}), current_time=ts)
        if referee.last_violation is not None:
            violation = referee.last_violation
            break

    assert violation is not None
    assert violation.rule_name == "robot_stop_speed"
    assert violation.is_stopping is False
    assert violation.offending_teams == (True,)
    # Non-stopping foul: command must remain STOP, not transition.
    assert result.referee_command == RefereeCommand.STOP


# ---------------------------------------------------------------------------
# BallPlacementInterferenceRule — §8.4.3, non-stopping.
# stadium_radius_meters=0.5, grace_seconds=2.0 (simulation.yaml).
# ---------------------------------------------------------------------------


def test_ball_placement_interference_rule_fires_through_custom_referee_step():
    referee = CustomReferee.from_profile_name("simulation", n_robots_yellow=1, n_robots_blue=1)
    referee.seed_clock(0.0, initial_command=RefereeCommand.NORMAL_START)

    # Yellow (friendly) is placing; ball sits far from the designated
    # target so ball_placement_to_next auto-advance never fires mid-test.
    # Blue (enemy) robot loiters on the placement line the whole time.
    ball_pos = (0.0, 0.0)
    designated = (2.0, 0.0)
    referee.force_command(RefereeCommand.BALL_PLACEMENT_YELLOW, 0.0, ball_placement_target=designated)

    violation = None
    result = None
    for ts in (0.1, 1.0, 2.1, 2.2):
        enemy = _robot(0, 1.0, 0.0, is_friendly=False)  # on the ball->designated line, well inside 0.5m stadium
        result = referee.step(_frame(ts, _ball(*ball_pos), enemy_robots={0: enemy}), current_time=ts)
        if referee.last_violation is not None:
            violation = referee.last_violation
            break

    assert violation is not None
    assert violation.rule_name == "ball_placement_interference"
    assert violation.is_stopping is False
    assert violation.offending_teams == (False,)  # enemy (blue) is the non-placing/interfering team
    # Non-stopping: command must remain the active BALL_PLACEMENT_YELLOW.
    assert result.referee_command == RefereeCommand.BALL_PLACEMENT_YELLOW


# ---------------------------------------------------------------------------
# DefenseAreaStoppageRule — §8.4.1, first offense is non-stopping.
# min_distance_meters=0.2, grace_seconds=2.0 (simulation.yaml).
# ---------------------------------------------------------------------------


def test_defense_area_stoppage_rule_fires_through_custom_referee_step():
    referee = CustomReferee.from_profile_name("simulation", n_robots_yellow=1, n_robots_blue=1)
    referee.seed_clock(0.0, initial_command=RefereeCommand.NORMAL_START)
    referee.force_command(RefereeCommand.STOP, 0.0)

    # Friendly is yellow, my_team_is_right=False -> yellow's own area is on
    # the left, blue's (the opponent's, for yellow) is on the right. Put a
    # friendly (yellow) robot inside blue's defense area the whole time.
    violation = None
    result = None
    for ts in (0.1, 1.0, 2.1, 2.2):
        friendly = _robot(0, 4.4, 0.0, is_friendly=True)  # deep inside the right (blue/opponent) defense area
        result = referee.step(_frame(ts, _ball(0.0, 0.0), friendly_robots={0: friendly}), current_time=ts)
        if referee.last_violation is not None:
            violation = referee.last_violation
            break

    assert violation is not None
    assert violation.rule_name == "defense_area_stoppage"
    assert violation.offending_teams == (True,)  # yellow (friendly) encroaching
    # First offense: game stays stopped regularly (non-stopping), not HALT.
    assert violation.is_stopping is False
    assert result.referee_command == RefereeCommand.STOP
