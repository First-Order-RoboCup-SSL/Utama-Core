from types import SimpleNamespace

import py_trees
import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.game.field import Field, FieldBounds
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.strategy.common import (
    AbstractBehaviour,
    AbstractStrategy,
    SpaceRequirements,
)
from utama_core.strategy.common.abstract_strategy import (
    _REFEREE_STOPPAGE_COMMANDS,
    _ResetStrategyOnRefereeStoppage,
)


# Dummy blackboard helper
def make_dummy_blackboard(actual_field_bounds):
    bb = SimpleNamespace()
    bb.game = SimpleNamespace()
    bb.game.field = Field(
        my_team_is_right=True,
        field_dims=STANDARD_FIELD_DIMS,
        field_bounds=actual_field_bounds,
    )
    return bb


# Dummy game object
def make_dummy_game(field_bounds):
    game = SimpleNamespace()
    game.field = Field(my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=field_bounds)
    return game


# Helper strategy
class DummyStrategy(AbstractStrategy):
    exp_ball = True  # Not relevant for these tests

    def create_behaviour_tree(self):
        return AbstractBehaviour()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        return True

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool):
        return True

    def __init__(self, min_bb=None):
        super().__init__()
        self._min_bb = min_bb

    def get_min_bounding_req(self):
        return self._min_bb


# --- Test cases ---


def test_normal_case():
    actual_field = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
    min_bb = FieldBounds(top_left=(-4.0, 2.5), bottom_right=(4.0, -2.5))
    strategy = DummyStrategy(min_bb=min_bb)
    strategy.blackboard = make_dummy_blackboard(actual_field)
    game = make_dummy_game(actual_field)
    strategy.assert_field_requirements(game)  # should pass


def test_min_bb_none():
    actual_field = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
    strategy = DummyStrategy(min_bb=None)
    strategy.blackboard = make_dummy_blackboard(actual_field)
    game = make_dummy_game(actual_field)
    strategy.assert_field_requirements(game)  # should pass


def test_min_bb_outside_field():
    actual_field = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
    min_bb = FieldBounds(top_left=(-5.0, 3.5), bottom_right=(4.0, -2.5))
    strategy = DummyStrategy(min_bb=min_bb)
    strategy.blackboard = make_dummy_blackboard(actual_field)
    game = make_dummy_game(actual_field)
    with pytest.raises(ValueError):
        strategy.assert_field_requirements(game)


def test_crossed_bounding_box():
    actual_field = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
    min_bb = FieldBounds(top_left=(1.0, -1.0), bottom_right=(-1.0, 1.0))  # crossed
    strategy = DummyStrategy(min_bb=min_bb)
    strategy.blackboard = make_dummy_blackboard(actual_field)
    game = make_dummy_game(actual_field)
    with pytest.raises(ValueError):
        strategy.assert_field_requirements(game)


def test_min_bb_exceeds_full_field():
    actual_field = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
    min_bb = FieldBounds(top_left=(-5.0, 4.0), bottom_right=(5.0, -4.0))
    strategy = DummyStrategy(min_bb=min_bb)
    strategy.blackboard = make_dummy_blackboard(actual_field)
    game = make_dummy_game(actual_field)
    with pytest.raises(ValueError):
        strategy.assert_field_requirements(game)


def test_min_bb_not_contained_in_actual_field():
    actual_field = FieldBounds(top_left=(-2.0, 2.0), bottom_right=(2.0, -2.0))
    min_bb = FieldBounds(top_left=(-3.0, 1.5), bottom_right=(1.0, -1.5))
    strategy = DummyStrategy(min_bb=min_bb)
    strategy.blackboard = make_dummy_blackboard(actual_field)
    game = make_dummy_game(actual_field)
    with pytest.raises(ValueError, match="does not contain"):
        strategy.assert_field_requirements(game)


def test_space_requirements_rejected_when_field_too_small():
    actual_field = FieldBounds(top_left=(-2.0, 1.0), bottom_right=(2.0, -1.0))
    strategy = DummyStrategy(min_bb=SpaceRequirements(min_length=4.5, min_width=2.5))
    strategy.blackboard = make_dummy_blackboard(actual_field)
    game = make_dummy_game(actual_field)
    with pytest.raises(ValueError, match="too small for strategy"):
        strategy.assert_field_requirements(game)


def test_strategy_subtree_resets_once_when_referee_enters_stoppage():
    strategy = ResetRecorderStrategy()
    strategy.setup_strategy_blackboard(is_opp_strat=False)
    strategy.setup_behaviour_tree(is_opp_strat=False)
    strategy.blackboard.game = make_referee_game(RefereeCommand.FORCE_START, timestamp=0.0)
    strategy.blackboard.cmd_map = {}

    strategy.behaviour_tree.tick()
    assert strategy.recorder.initialise_count == 1
    assert strategy.recorder.status == py_trees.common.Status.RUNNING

    strategy.blackboard.game.referee = make_referee_data(RefereeCommand.HALT, timestamp=1.0)
    strategy.behaviour_tree.tick()

    assert strategy.recorder.invalidated_count == 1
    assert strategy.recorder.status == py_trees.common.Status.INVALID

    strategy.behaviour_tree.tick()
    assert strategy.recorder.invalidated_count == 1

    strategy.blackboard.game.referee = make_referee_data(RefereeCommand.FORCE_START, timestamp=2.0)
    strategy.behaviour_tree.tick()

    assert strategy.recorder.initialise_count == 2
    assert strategy.recorder.status == py_trees.common.Status.RUNNING


def test_active_referee_start_commands_do_not_reset_strategy_subtree():
    strategy = ResetRecorderStrategy()
    strategy.setup_strategy_blackboard(is_opp_strat=False)
    strategy.setup_behaviour_tree(is_opp_strat=False)
    strategy.blackboard.game = make_referee_game(RefereeCommand.NORMAL_START, timestamp=0.0)
    strategy.blackboard.cmd_map = {}

    strategy.behaviour_tree.tick()
    strategy.blackboard.game.referee = make_referee_data(RefereeCommand.FORCE_START, timestamp=1.0)
    strategy.behaviour_tree.tick()

    assert strategy.recorder.initialise_count == 1
    assert strategy.recorder.invalidated_count == 0


@pytest.mark.parametrize("command", sorted(_REFEREE_STOPPAGE_COMMANDS, key=lambda item: item.value))
def test_reset_guard_invalidates_for_every_stoppage_command(command):
    target = RunningRecorder(name="Target")
    target.status = py_trees.common.Status.RUNNING
    guard = _ResetStrategyOnRefereeStoppage(target=target)
    guard.blackboard = SimpleNamespace(game=make_referee_game(command, timestamp=1.0))

    assert guard.update() == py_trees.common.Status.SUCCESS
    assert target.invalidated_count == 1


def test_reset_guard_ignores_active_start_commands():
    target = RunningRecorder(name="Target")
    target.status = py_trees.common.Status.RUNNING
    guard = _ResetStrategyOnRefereeStoppage(target=target)
    guard.blackboard = SimpleNamespace(game=make_referee_game(RefereeCommand.NORMAL_START, timestamp=1.0))

    assert guard.update() == py_trees.common.Status.SUCCESS
    assert target.invalidated_count == 0


class ResetRecorderStrategy(AbstractStrategy):
    exp_ball = True

    def __init__(self):
        self.recorder = RunningRecorder(name="Recorder")
        super().__init__()

    def create_behaviour_tree(self):
        return self.recorder

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        return True

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool):
        return True

    def get_min_bounding_req(self):
        return None


class RunningRecorder(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name=name)
        self.initialise_count = 0
        self.invalidated_count = 0

    def initialise(self):
        self.initialise_count += 1

    def update(self):
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if new_status == py_trees.common.Status.INVALID:
            self.invalidated_count += 1


def make_referee_game(command, timestamp):
    return SimpleNamespace(
        referee=make_referee_data(command, timestamp),
        friendly_robots={},
        my_team_is_yellow=True,
        my_team_is_right=False,
    )


def make_referee_data(command, timestamp):
    return SimpleNamespace(
        referee_command=command,
        referee_command_timestamp=timestamp,
    )
