from unittest.mock import patch

from utama_core.config.formations import get_formations
from utama_core.entities.game.field import FieldBounds
from utama_core.global_utils.mapping_utils import (
    map_friendly_enemy_to_colors,
    map_left_right_to_colors,
)
from utama_core.run.strategy_runner import StrategyRunner
from utama_core.tests.strategy_runner.strat_runner_test_utils import DummyStrategy


class _FakeGRSimController:
    def __init__(self, field_bounds, exp_ball):
        self.field_bounds = field_bounds
        self.exp_ball = exp_ball
        self.set_robot_presence_calls: list[tuple[int, bool, bool]] = []
        self.teleport_robot_calls: list[tuple[bool, int, float, float, float | None]] = []
        self.teleport_ball_calls: list[tuple[float, float]] = []
        self.remove_ball_calls = 0

    def set_robot_presence(self, robot_id, is_team_yellow, should_robot_be_present):
        self.set_robot_presence_calls.append((robot_id, is_team_yellow, should_robot_be_present))

    def teleport_robot(self, is_team_yellow, robot_id, x, y, theta=None):
        self.teleport_robot_calls.append((is_team_yellow, robot_id, x, y, theta))

    def teleport_ball(self, x, y):
        self.teleport_ball_calls.append((x, y))

    def remove_ball(self):
        self.remove_ball_calls += 1


class _FakeGRSimRobotController:
    def __init__(self, is_team_yellow, n_friendly):
        self.is_team_yellow = is_team_yellow
        self.n_friendly = n_friendly


def test_grsim_spawn_positions_and_ball_use_field_bounds_center():
    bounds = FieldBounds(top_left=(-2.0, 2.4), bottom_right=(4.0, -1.6))
    my_team_is_yellow = True
    my_team_is_right = False
    exp_friendly = 2
    exp_enemy = 1

    with (
        patch(
            "utama_core.run.strategy_runner.GRSimController",
            _FakeGRSimController,
        ),
        patch(
            "utama_core.run.strategy_runner.GRSimRobotController",
            _FakeGRSimRobotController,
        ),
        patch.object(
            StrategyRunner,
            "start_threads",
            lambda self, vision_receiver: None,
        ),
        patch.object(
            StrategyRunner,
            "_load_game",
            lambda self: None,
        ),
        patch.object(
            StrategyRunner,
            "_assert_exp_goals",
            lambda self: None,
        ),
    ):
        runner = StrategyRunner(
            strategy=DummyStrategy(),
            my_team_is_yellow=my_team_is_yellow,
            my_team_is_right=my_team_is_right,
            mode="grsim",
            exp_friendly=exp_friendly,
            exp_enemy=exp_enemy,
            field_bounds=bounds,
            exp_ball=True,
        )

    assert isinstance(runner.sim_controller, _FakeGRSimController)

    right_start, left_start = get_formations(
        bounds=bounds,
        n_right=exp_enemy,
        n_left=exp_friendly,
    )
    expected_yellow, expected_blue = map_left_right_to_colors(
        my_team_is_yellow,
        my_team_is_right,
        right_start,
        left_start,
    )

    n_yellow, n_blue = map_friendly_enemy_to_colors(
        my_team_is_yellow,
        exp_friendly,
        exp_enemy,
    )

    expected_calls: dict[tuple[bool, int], tuple[float, float, float]] = {}
    for y in range(n_yellow):
        e = expected_yellow[y]
        expected_calls[(True, y)] = (e.x, e.y, e.theta)
    for b in range(n_blue):
        e = expected_blue[b]
        expected_calls[(False, b)] = (e.x, e.y, e.theta)

    actual_calls = {
        (is_yellow, robot_id): (x, y, theta)
        for is_yellow, robot_id, x, y, theta in runner.sim_controller.teleport_robot_calls
    }

    assert actual_calls == expected_calls
    assert runner.sim_controller.teleport_ball_calls == [bounds.center]


def test_grsim_exp_ball_false_removes_ball_not_center_teleport():
    bounds = FieldBounds(top_left=(-2.0, 2.4), bottom_right=(4.0, -1.6))
    strategy = DummyStrategy()
    strategy.exp_ball = False

    with (
        patch(
            "utama_core.run.strategy_runner.GRSimController",
            _FakeGRSimController,
        ),
        patch(
            "utama_core.run.strategy_runner.GRSimRobotController",
            _FakeGRSimRobotController,
        ),
        patch.object(
            StrategyRunner,
            "start_threads",
            lambda self, vision_receiver: None,
        ),
        patch.object(
            StrategyRunner,
            "_load_game",
            lambda self: None,
        ),
        patch.object(
            StrategyRunner,
            "_assert_exp_goals",
            lambda self: None,
        ),
    ):
        runner = StrategyRunner(
            strategy=strategy,
            my_team_is_yellow=True,
            my_team_is_right=False,
            mode="grsim",
            exp_friendly=2,
            exp_enemy=1,
            field_bounds=bounds,
            exp_ball=False,
        )

    assert runner.sim_controller.remove_ball_calls == 1
    assert runner.sim_controller.teleport_ball_calls == []
