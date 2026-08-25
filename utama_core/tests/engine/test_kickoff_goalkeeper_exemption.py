"""Tests for the goalkeeper-exemption and defense-area-legality fixes to
`PrepareKickoffOursStep`/`PrepareKickoffTheirsStep`.

Before this fix, both kickoff Steps hardcoded "goalkeeper = robot id 0" and
pulled that robot into the outfield kickoff formation (as a candidate kicker
on "ours", and unconditionally on "theirs") — unlike `PreparePenaltyOursStep`/
`PreparePenaltyTheirsStep`, which already read the real keeper ID from the
referee packet (`ref.yellow_team.goalkeeper` / `ref.blue_team.goalkeeper`).
Neither kickoff Step passed `clear_own_defense_area`/`clear_opp_defense_area`
to `_clear_to_legal_positions` either, so a formation position that happened
to land inside a defense area had no live correction — unlike `StopStep`,
which already used both flags.

Follows the same rsim-integration convention as `test_referee_override.py`
(same file, same fixture style) rather than hand-building a fake `Game` —
`actions.py`'s Step classes call `move()`, which calls into a real
`MotionController.calculate(...)`, so a bare fake `Game` would need to fake
that too; rsim is the lighter-weight correct fixture here.
"""

from __future__ import annotations

import math

import pytest

from utama_core.config.referee_constants import BALL_KEEP_OUT_DISTANCE
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.strategy.kernel_strategy import build_split_shape_kernel_strategy


def _set_referee_command(
    runner,
    command: RefereeCommand,
    designated_position=None,
    yellow_goalkeeper: int = 0,
    blue_goalkeeper: int = 0,
):
    """Same helper as `test_referee_override.py`'s, extended with goalkeeper IDs
    so a test can put the real keeper somewhere other than the default 0."""
    referee_data = RefereeData(
        source_identifier=None,
        time_sent=0.0,
        time_received=0.0,
        referee_command=command,
        referee_command_timestamp=0.0,
        stage=Stage.NORMAL_FIRST_HALF,
        stage_time_left=0.0,
        blue_team=TeamInfo(name="blue", goalkeeper=blue_goalkeeper),
        yellow_team=TeamInfo(name="yellow", goalkeeper=yellow_goalkeeper),
        designated_position=designated_position,
    )
    object.__setattr__(runner.my.game.current, "referee", referee_data)
    object.__setattr__(runner.my.current_game_frame, "referee", referee_data)


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


@pytest.fixture
def nonzero_keeper_runner():
    """Robot 3, not 0, is the real goalkeeper — outfield pool is (0,1,2,4,5).

    Exercises the "keeper ID isn't 0" path both `PrepareKickoffOursStep` and
    `PrepareKickoffTheirsStep` mishandled before this fix (kicker selection /
    formation-slot assignment both used to just check `rid != 0`).
    """
    from utama_core.run.strategy_runner import StrategyRunner

    strategy = AbstractStrategy(
        build_kernel_strategy=build_split_shape_kernel_strategy((0, 1, 2, 4, 5)),
        goalkeeper_id=3,
    )
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


def test_our_kickoff_nonzero_keeper_is_never_kicker_or_in_formation(nonzero_keeper_runner):
    """With the real keeper at robot 3 (not 0), our kickoff must never pick
    robot 3 as kicker or give it a formation command — it should be left for
    `GoalkeeperTactic`, which tracks/blocks near our own goal line (see
    `goalkeep()`), not frozen. So the observable contract isn't "the keeper
    doesn't move" (it legitimately does, chasing the ball toward our goal),
    it's "the keeper ends up near our goal line, nowhere near the centre-spot
    kicker position or the kickoff-formation targets" — while every other
    non-keeper robot converges outside the centre keep-out zone as usual."""
    game = nonzero_keeper_runner.my.game
    nonzero_keeper_runner.step_once()

    # Scatter everyone off their spawn positions so a pass only demonstrates
    # something if the override actually moved them.
    for rid in (0, 1, 2, 4, 5):
        nonzero_keeper_runner.sim_controller.teleport_robot(True, rid, 0.1 * rid, -2.5 + 0.2 * rid, 0.0)
    nonzero_keeper_runner.sim_controller.teleport_robot(True, 3, -3.0, 2.0, 0.0)
    nonzero_keeper_runner.step_once()

    _set_referee_command(nonzero_keeper_runner, RefereeCommand.PREPARE_KICKOFF_YELLOW, yellow_goalkeeper=3)

    for _ in range(200):
        nonzero_keeper_runner.step_once()

    # my_team_is_right=True -> our goal/half is on the +x side (see
    # PreparePenalty tests above for the same convention). `goalkeep()`
    # converges toward the goal but isn't pinned to an exact x, so the robust
    # check is "moved into our own half, nowhere near the centre spot" — not
    # an exact goal-line distance, which is `goalkeep()`'s own concern.
    keeper = game.friendly_robots[3]
    dist_keeper_to_ball = math.hypot(keeper.p.x - game.ball.p.x, keeper.p.y - game.ball.p.y)
    assert (
        keeper.p.x > 0
    ), f"keeper (robot 3) did not move into our own half during PREPARE_KICKOFF_YELLOW: x={keeper.p.x:.3f}"
    assert (
        dist_keeper_to_ball >= BALL_KEEP_OUT_DISTANCE - 0.05
    ), "keeper (robot 3) ended up at the centre spot — it must never be picked as kicker"

    # Robot 0 is the lowest-ID non-keeper robot, so it's the kicker here and
    # is *supposed* to approach the ball closely — only the support robots
    # (1, 2, 4, 5) are checked against the keep-out radius.
    ball = game.ball.p
    for rid in (1, 2, 4, 5):
        robot = game.friendly_robots[rid]
        dist_to_ball = math.hypot(robot.p.x - ball.x, robot.p.y - ball.y)
        assert dist_to_ball >= BALL_KEEP_OUT_DISTANCE - 0.05, f"support robot {rid} inside centre keep-out zone"


def test_their_kickoff_nonzero_keeper_is_never_in_defensive_wall(nonzero_keeper_runner):
    """Same non-zero-keeper setup, opponent's kickoff: robot 3 is exempt from
    the defensive-wall formation and left to `GoalkeeperTactic` (so it ends up
    near our own goal line, not at one of the fixed defence-formation spots),
    while every other friendly robot moves into the defensive formation
    outside the centre circle."""
    game = nonzero_keeper_runner.my.game
    nonzero_keeper_runner.step_once()

    nonzero_keeper_runner.sim_controller.teleport_robot(True, 3, -3.0, 2.0, 0.0)
    nonzero_keeper_runner.sim_controller.teleport_robot(True, 0, 0.05, 0.0, 0.0)
    nonzero_keeper_runner.step_once()

    _set_referee_command(nonzero_keeper_runner, RefereeCommand.PREPARE_KICKOFF_BLUE, yellow_goalkeeper=3)

    for _ in range(200):
        nonzero_keeper_runner.step_once()

    keeper = game.friendly_robots[3]
    assert (
        keeper.p.x > 0
    ), f"keeper (robot 3) did not move into our own half during PREPARE_KICKOFF_BLUE: x={keeper.p.x:.3f}"

    ball = game.ball.p
    for rid in (0, 1, 2, 4, 5):
        robot = game.friendly_robots[rid]
        dist_to_ball = math.hypot(robot.p.x - ball.x, robot.p.y - ball.y)
        assert dist_to_ball >= BALL_KEEP_OUT_DISTANCE - 0.05, f"non-keeper robot {rid} inside centre keep-out zone"


def test_our_kickoff_default_keeper_id_zero_still_exempt(split_shape_runner):
    """Regression check for the common case: with the default `goalkeeper_id=0`
    (matching `TeamInfo`'s default `goalkeeper=0`), robot 0 must still be
    excluded from both kicker-selection and the support formation — this
    isn't relying on "any non-zero ID," it must hold for the original
    id-0 case too. As with the non-zero-keeper tests above, "exempt" means
    "goes to GoalkeeperTactic's own-goal tracking," not "frozen.\" """
    game = split_shape_runner.my.game
    split_shape_runner.step_once()

    split_shape_runner.sim_controller.teleport_robot(True, 0, -3.0, 2.0, 0.0)
    split_shape_runner.step_once()

    _set_referee_command(split_shape_runner, RefereeCommand.PREPARE_KICKOFF_YELLOW, yellow_goalkeeper=0)

    for _ in range(200):
        split_shape_runner.step_once()

    keeper = game.friendly_robots[0]
    dist_keeper_to_ball = math.hypot(keeper.p.x - game.ball.p.x, keeper.p.y - game.ball.p.y)
    assert (
        keeper.p.x > 0
    ), f"keeper (robot 0) did not move into our own half during PREPARE_KICKOFF_YELLOW: x={keeper.p.x:.3f}"
    assert (
        dist_keeper_to_ball >= BALL_KEEP_OUT_DISTANCE - 0.05
    ), "keeper (robot 0) ended up at the centre spot — it must never be picked as kicker"


def test_our_kickoff_evicts_a_formation_robot_from_own_defense_area(nonzero_keeper_runner):
    """`clear_own_defense_area=True` is now passed by `PrepareKickoffOursStep`
    (previously it passed neither defense-area flag at all, unlike `StopStep`)
    — a non-keeper, non-kicker robot starting deep inside our own defense area
    must be evicted to just outside the front edge rather than left there or
    driven toward a formation spot that happens to route through the box.
    """
    game = nonzero_keeper_runner.my.game
    field = game.field
    nonzero_keeper_runner.step_once()

    # my_team_is_right=True -> our goal/defense area is on the +x side.
    our_goal_x = field.full_field_half_length
    deep_in_own_box = (our_goal_x - 0.3, 0.2)
    nonzero_keeper_runner.sim_controller.teleport_robot(True, 4, deep_in_own_box[0], deep_in_own_box[1], 0.0)
    nonzero_keeper_runner.step_once()

    _set_referee_command(nonzero_keeper_runner, RefereeCommand.PREPARE_KICKOFF_YELLOW, yellow_goalkeeper=3)

    for _ in range(200):
        nonzero_keeper_runner.step_once()

    depth = field.half_defense_area_depth
    width = field.half_defense_area_width
    inner_x = our_goal_x - 2.0 * depth

    robot = game.friendly_robots[4]
    still_in_box = robot.p.x >= inner_x and abs(robot.p.y) <= width
    assert not still_in_box, (
        f"robot 4 was left inside our own defense area (x={robot.p.x:.3f}, y={robot.p.y:.3f}) "
        "after PREPARE_KICKOFF_YELLOW settled — clear_own_defense_area should have evicted it"
    )


def test_their_kickoff_evicts_a_formation_robot_from_own_defense_area(nonzero_keeper_runner):
    """Same eviction check for `PrepareKickoffTheirsStep`, which also gained
    `clear_own_defense_area=True` in this fix."""
    game = nonzero_keeper_runner.my.game
    field = game.field
    nonzero_keeper_runner.step_once()

    our_goal_x = field.full_field_half_length
    deep_in_own_box = (our_goal_x - 0.3, -0.2)
    nonzero_keeper_runner.sim_controller.teleport_robot(True, 5, deep_in_own_box[0], deep_in_own_box[1], 0.0)
    nonzero_keeper_runner.step_once()

    _set_referee_command(nonzero_keeper_runner, RefereeCommand.PREPARE_KICKOFF_BLUE, yellow_goalkeeper=3)

    for _ in range(200):
        nonzero_keeper_runner.step_once()

    depth = field.half_defense_area_depth
    width = field.half_defense_area_width
    inner_x = our_goal_x - 2.0 * depth

    robot = game.friendly_robots[5]
    still_in_box = robot.p.x >= inner_x and abs(robot.p.y) <= width
    assert not still_in_box, (
        f"robot 5 was left inside our own defense area (x={robot.p.x:.3f}, y={robot.p.y:.3f}) "
        "after PREPARE_KICKOFF_BLUE settled — clear_own_defense_area should have evicted it"
    )


def test_our_kickoff_with_only_keeper_present_does_not_raise(nonzero_keeper_runner):
    """Edge case: if every non-keeper robot is (hypothetically) absent from
    `game.friendly_robots`, `PrepareKickoffOursStep.update()` must return
    `None`/no-op rather than raising (empty `robot_ids` after removing the
    keeper). Exercised directly against the Step class with a minimal object,
    since reproducing "rsim spawned only one robot" isn't practical through
    the fixture — this only needs to prove the early-return guard fires."""
    from utama_core.custom_referee.actions import PrepareKickoffOursStep

    class _FakeRef:
        designated_position = None

        class yellow_team:
            goalkeeper = 3

        class blue_team:
            goalkeeper = 0

    class _FakeGame:
        my_team_is_yellow = True
        my_team_is_right = True
        friendly_robots = {3: object()}
        referee = _FakeRef()

    class _Blackboard:
        game = _FakeGame()
        motion_controller = None
        cmd_map: dict = {}

    step = PrepareKickoffOursStep()
    step.blackboard = _Blackboard()
    result = step.update()

    assert result is None
    assert step.blackboard.cmd_map == {}
