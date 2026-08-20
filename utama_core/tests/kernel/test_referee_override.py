"""Integration tests for the kernel model's referee-restart override.

Uses a real rsim `Game` (via `AbstractStrategy`/`StrategyRunner`) rather than a
hand-built fake, since the override delegates to `strategy/referee/actions.py`
Step classes that read a wide slice of `Game` (field dims, friendly_robots,
ball, referee.designated_position, ...) — reproducing that surface with a
fake would just be a worse copy of `Game` itself. Pure-scheduling behavior
(does `Strategy.tick()` even route to the override) is covered in
`test_strategy.py`; this file checks the override actually repositions robots
legally for the two easiest-to-verify restarts.

Robots start each test placed *inside* the relevant keep-out zone
deliberately (teleported there via `sim_controller`), not left wherever the
split-shape tactics happen to put them — the split-shape formation is
naturally spread out, so a version of this test that only checked "robots
ended up far from the ball/centre" passed even when the override was not
actually engaging (caught during development: see the
`side.current_game_frame` note on `_set_referee_command` below). Starting
adversarially inside the zone means a passing test actually demonstrates the
override moved something, not that nothing needed to move.
"""

from __future__ import annotations

import math

import pytest

from utama_core.config.referee_constants import BALL_KEEP_OUT_DISTANCE
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.kernel.abstract_strategy import AbstractStrategy
from utama_core.strategy.kernel_strategy import build_split_shape_kernel_strategy


@pytest.fixture
def split_shape_runner():
    from utama_core.run.strategy_runner import StrategyRunner

    strategy = AbstractStrategy(build_kernel_strategy=build_split_shape_kernel_strategy((1, 2, 3, 4, 5)))
    runner = StrategyRunner(
        strategy=strategy,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=6,
        exp_enemy=3,
        exp_ball=True,
    )
    yield runner
    runner.close()


def _set_referee_command(runner, command: RefereeCommand, designated_position=None):
    """Inject a referee command that actually persists across ticks.

    `Game.referee` is a read-only property over `game.current` (a frozen
    `GameFrame`), so setting it needs `object.__setattr__` the same way
    `CurrentGameFrame.__init__` itself bypasses immutability. That alone is
    NOT enough, though: `StrategyRunner._run_step` rebuilds each tick's
    `GameFrame` via `dataclasses.replace(side.current_game_frame, ...)` — a
    *different* attribute than `game.current` — so a mutation applied only to
    `game.current` is silently discarded on the very next tick, reverting to
    whatever `side.current_game_frame.referee` still holds (found by tracing
    `referee_command` tick-by-tick while developing this test: it reverted to
    `None` after exactly one tick). Both attributes must be set for an
    injected command to actually hold across a multi-tick loop.
    """
    referee_data = RefereeData(
        source_identifier=None,
        time_sent=0.0,
        time_received=0.0,
        referee_command=command,
        referee_command_timestamp=0.0,
        stage=Stage.NORMAL_FIRST_HALF,
        stage_time_left=0.0,
        blue_team=TeamInfo(name="blue"),
        yellow_team=TeamInfo(name="yellow"),
        designated_position=designated_position,
    )
    object.__setattr__(runner.my.game.current, "referee", referee_data)
    object.__setattr__(runner.my.current_game_frame, "referee", referee_data)


def test_their_ball_placement_clears_our_robots_from_keep_out_zone(split_shape_runner):
    """During the opponent's ball placement, a robot starting inside the
    keep-out zone around the ball moves outside it — a plain tactic tick has
    no notion of this rule and would just leave it there or drive it closer."""
    game = split_shape_runner.my.game
    split_shape_runner.step_once()

    ball_x, ball_y = game.ball.p.x, game.ball.p.y
    split_shape_runner.sim_controller.teleport_robot(True, 1, ball_x + 0.1, ball_y, 0.0)
    split_shape_runner.step_once()  # let the teleport land in game state

    designated = (1.5, 0.0)
    _set_referee_command(split_shape_runner, RefereeCommand.BALL_PLACEMENT_BLUE, designated_position=designated)

    for _ in range(200):
        split_shape_runner.step_once()

    ball = game.ball.p
    for robot_id, robot in game.friendly_robots.items():
        dist_to_ball = math.hypot(robot.p.x - ball.x, robot.p.y - ball.y)
        dist_to_spot = math.hypot(robot.p.x - designated[0], robot.p.y - designated[1])
        assert dist_to_ball >= BALL_KEEP_OUT_DISTANCE - 0.05, f"robot {robot_id} too close to ball"
        assert dist_to_spot >= BALL_KEEP_OUT_DISTANCE - 0.05, f"robot {robot_id} too close to designated spot"


def test_their_kickoff_clears_our_robots_outside_center_circle(split_shape_runner):
    """A robot starting at the centre spot for the opponent's kickoff moves
    outside the keep-out radius."""
    game = split_shape_runner.my.game
    split_shape_runner.step_once()

    split_shape_runner.sim_controller.teleport_robot(True, 1, 0.05, 0.0, 0.0)
    split_shape_runner.step_once()

    _set_referee_command(split_shape_runner, RefereeCommand.PREPARE_KICKOFF_BLUE)

    for _ in range(200):
        split_shape_runner.step_once()

    ball = game.ball.p
    for robot_id, robot in game.friendly_robots.items():
        dist_to_ball = math.hypot(robot.p.x - ball.x, robot.p.y - ball.y)
        assert dist_to_ball >= BALL_KEEP_OUT_DISTANCE - 0.05, f"robot {robot_id} inside centre keep-out zone"


def test_override_ends_and_tactics_resume_on_normal_start(split_shape_runner):
    """Once NORMAL_START arrives after a restart, tactics tick again (mem was
    reset by the barrier transition on the way in) instead of the override
    continuing to hold every robot in place forever."""
    strategy = split_shape_runner.my.strategy

    _set_referee_command(split_shape_runner, RefereeCommand.PREPARE_KICKOFF_BLUE)
    for _ in range(50):
        split_shape_runner.step_once()

    _set_referee_command(split_shape_runner, RefereeCommand.NORMAL_START)
    split_shape_runner.step_once()

    partition = strategy._kernel_strategy.active_partition
    assert partition, "tactics never resumed after the restart ended"


def test_back_to_back_restarts_dispatch_a_fresh_step_each_time(split_shape_runner):
    """A second restart arriving without an intervening NORMAL_START (e.g. a
    kickoff foul immediately becoming a ball placement) must dispatch to the
    *new* command's Step, not keep re-running the first command's Step just
    because `Strategy` never saw a barrier-clearing NORMAL_START in between.

    Checked at the `RefereeOverride`/`Strategy.tick()` mechanism level
    (returned command dict differs) rather than by watching rsim physics
    settle to a final position: `_clear_to_legal_positions` recomputes its
    target from the live ball position every tick, and the ball itself can
    drift for several seconds after a restart (a known rsim convergence
    quirk unrelated to this override, see `dribble.py`'s KNOWN ISSUE note) —
    that makes "did the robot's position stop changing" an unreliable signal
    for "did the override switch commands," even though the switch itself is
    correct and instant.
    """
    game = split_shape_runner.my.game
    strategy = split_shape_runner.my.strategy._kernel_strategy
    split_shape_runner.step_once()

    _set_referee_command(split_shape_runner, RefereeCommand.PREPARE_KICKOFF_BLUE)
    split_shape_runner.step_once()
    kickoff_commands = strategy.tick(game)

    # No NORMAL_START in between — straight into a second, different restart.
    _set_referee_command(split_shape_runner, RefereeCommand.BALL_PLACEMENT_BLUE, designated_position=(1.0, 1.0))
    split_shape_runner.step_once()
    placement_commands = strategy.tick(game)

    assert kickoff_commands != placement_commands, "override did not re-dispatch for the second restart command"


def test_goalkeeper_stops_during_halt(split_shape_runner):
    """The goalkeeper must stop issuing motion commands during HALT, same as
    the outfield pool (`Strategy.tick()`'s `is_paused` check) — it must not
    keep chasing the ball just because it's ticked outside `Strategy`."""
    game = split_shape_runner.my.game
    split_shape_runner.step_once()

    _set_referee_command(split_shape_runner, RefereeCommand.HALT)
    for _ in range(30):  # let residual velocity from before HALT decay first
        split_shape_runner.step_once()

    gk_pos_before = (game.friendly_robots[0].p.x, game.friendly_robots[0].p.y)
    for _ in range(20):
        split_shape_runner.step_once()
    gk_pos_after = (game.friendly_robots[0].p.x, game.friendly_robots[0].p.y)

    dist_moved = math.hypot(gk_pos_after[0] - gk_pos_before[0], gk_pos_after[1] - gk_pos_before[1])
    assert dist_moved < 0.05, f"goalkeeper moved {dist_moved:.3f}m while HALT was active"


# ---------------------------------------------------------------------------
# Penalty and direct-free routing — RefereeOverride._step_for's "ours"/"theirs"
# dispatch for these four command families had zero test coverage until now:
# the only place it was ever exercised was the BT path's
# TestDispatcherRouting (utama_core/tests/referee/test_referee_unit.py),
# deleted alongside strategy/referee/tree.py since it tested the BT
# dispatcher classes directly, not the kernel's _step_for. Ball placement and
# kickoff (both directions) are covered above; these four fill the gap for
# penalty and direct-free.
# ---------------------------------------------------------------------------


def test_their_penalty_clears_our_robots_from_the_penalty_area(split_shape_runner):
    """During the opponent's penalty, PreparePenaltyTheirsStep's own two-part
    formation is actually reached: our goalkeeper moves to our goal line
    centre, and a non-keeper robot moves onto the behind-the-line formation
    in our own half — not just "moved somewhere far away" (there is no
    ball-keep-out-distance rule here at all; it's a fixed formation)."""
    game = split_shape_runner.my.game
    split_shape_runner.step_once()

    # Robot 0 (goalkeeper) and robot 1 both start on the wrong side of the
    # field entirely, so a passing test demonstrates the override actually
    # repositioned them rather than them already sitting somewhere plausible
    # — but close enough to their targets to cover the distance within the
    # 200-tick (~3.3s) budget this file's other tests use.
    split_shape_runner.sim_controller.teleport_robot(True, 0, 2.0, 0.0, 0.0)
    split_shape_runner.sim_controller.teleport_robot(True, 1, 0.5, 2.0, 0.0)
    split_shape_runner.step_once()

    _set_referee_command(split_shape_runner, RefereeCommand.PREPARE_PENALTY_BLUE)

    for _ in range(200):
        split_shape_runner.step_once()

    field = game.field
    # PreparePenaltyTheirsStep: our_goal_x = +half_length when my_team_is_right
    # (True here); the opponent's penalty mark (and therefore the behind-line
    # formation) sits between there and centre.
    our_goal_x = field.full_field_half_length
    penalty_mark_x = 0.5 * our_goal_x  # PENALTY_MARK_HALF_FIELD_RATIO

    keeper = game.friendly_robots[0]
    dist_keeper_to_goal = math.hypot(keeper.p.x - our_goal_x, keeper.p.y)
    assert dist_keeper_to_goal < 0.3, f"goalkeeper did not reach our goal line: dist={dist_keeper_to_goal:.3f}m"

    non_keeper = game.friendly_robots[1]
    dist_to_line = abs(non_keeper.p.x - penalty_mark_x)
    assert dist_to_line < 0.5, f"robot 1 did not reach the behind-the-line formation: dist_x={dist_to_line:.3f}m"


def test_our_penalty_moves_kicker_to_the_penalty_mark(split_shape_runner):
    """During our own penalty, the (non-goalkeeper) kicker drives to the
    opponent's penalty mark — PreparePenaltyOursStep's actual target, not
    just "moved somewhere"."""
    game = split_shape_runner.my.game
    split_shape_runner.step_once()

    # Robot 1 starts far from the opponent's penalty mark.
    split_shape_runner.sim_controller.teleport_robot(True, 1, -1.0, 2.0, 0.0)
    split_shape_runner.step_once()

    _set_referee_command(split_shape_runner, RefereeCommand.PREPARE_PENALTY_YELLOW)

    for _ in range(200):
        split_shape_runner.step_once()

    field = game.field
    # PreparePenaltyOursStep: opp_goal_x = +half_length if NOT my_team_is_right
    # else -half_length. split_shape_runner has my_team_is_right=True, so the
    # opponent's goal (and penalty mark) is on the NEGATIVE-x side.
    penalty_x = -0.5 * field.full_field_half_length
    non_goalkeeper_ids = [rid for rid in game.friendly_robots if rid != 0]
    kicker_id = min(non_goalkeeper_ids)  # PreparePenaltyOursStep: sorted(friendly_robots), first non-keeper
    kicker = game.friendly_robots[kicker_id]
    dist_to_mark = math.hypot(kicker.p.x - penalty_x, kicker.p.y)
    assert dist_to_mark < 0.3, f"kicker (robot {kicker_id}) did not reach the penalty mark: dist={dist_to_mark:.3f}m"


def test_their_direct_free_clears_our_robots_from_ball_keep_out_zone(split_shape_runner):
    """During the opponent's direct free kick, a robot starting on the ball
    clears the keep-out radius — same observable contract as ball placement's
    "theirs" test above, different command family."""
    game = split_shape_runner.my.game
    split_shape_runner.step_once()

    ball_x, ball_y = game.ball.p.x, game.ball.p.y
    split_shape_runner.sim_controller.teleport_robot(True, 1, ball_x, ball_y, 0.0)
    split_shape_runner.step_once()

    _set_referee_command(split_shape_runner, RefereeCommand.DIRECT_FREE_BLUE)

    for _ in range(200):
        split_shape_runner.step_once()

    ball = game.ball.p
    for robot_id, robot in game.friendly_robots.items():
        dist_to_ball = math.hypot(robot.p.x - ball.x, robot.p.y - ball.y)
        assert dist_to_ball >= BALL_KEEP_OUT_DISTANCE - 0.05, f"robot {robot_id} too close to ball"


def test_our_direct_free_moves_kicker_to_the_ball(split_shape_runner):
    """During our own direct free kick, the closest robot to the ball drives
    to an approach position near it (DirectFreeOursStep) — the same contract
    `_ApproachBallManager` verifies on the BT-replacement ball-placement path,
    for a different command family."""
    game = split_shape_runner.my.game
    split_shape_runner.step_once()

    ball_x, ball_y = game.ball.p.x, game.ball.p.y
    split_shape_runner.sim_controller.teleport_robot(True, 1, ball_x + 1.5, ball_y, 0.0)
    split_shape_runner.step_once()

    _set_referee_command(split_shape_runner, RefereeCommand.DIRECT_FREE_YELLOW)

    for _ in range(200):
        split_shape_runner.step_once()

    ball = game.ball.p
    kicker = min(
        game.friendly_robots.values(),
        key=lambda robot: math.hypot(robot.p.x - ball.x, robot.p.y - ball.y),
    )
    dist_to_ball = math.hypot(kicker.p.x - ball.x, kicker.p.y - ball.y)
    assert (
        dist_to_ball < 0.3
    ), f"no robot approached the ball for our direct free kick: closest dist={dist_to_ball:.3f}m"
