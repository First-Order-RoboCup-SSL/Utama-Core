from types import SimpleNamespace

import pytest

from utama_core.entities.game.field import FieldBounds
from utama_core.strategy.common import AbstractBehaviour, AbstractStrategy


# Dummy blackboard helper
def make_dummy_blackboard(actual_field_bounds):
    bb = SimpleNamespace()
    bb.game = SimpleNamespace()
    bb.game.field = SimpleNamespace()
    bb.game.field.field_bounds = actual_field_bounds
    return bb


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

    def get_min_bounding_zone(self):
        return self._min_bb


# --- Test cases ---


def test_normal_case():
    actual_field = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
    min_bb = FieldBounds(top_left=(-4.0, 2.5), bottom_right=(4.0, -2.5))
    strategy = DummyStrategy(min_bb=min_bb)
    strategy.blackboard = make_dummy_blackboard(actual_field)
    strategy.assert_field_requirements()  # should pass


def test_min_bb_none():
    actual_field = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
    strategy = DummyStrategy(min_bb=None)
    strategy.blackboard = make_dummy_blackboard(actual_field)
    strategy.assert_field_requirements()  # should pass


def test_min_bb_outside_field():
    actual_field = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
    min_bb = FieldBounds(top_left=(-5.0, 3.5), bottom_right=(4.0, -2.5))
    strategy = DummyStrategy(min_bb=min_bb)
    strategy.blackboard = make_dummy_blackboard(actual_field)
    with pytest.raises(AssertionError):
        strategy.assert_field_requirements()


def test_crossed_bounding_box():
    actual_field = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
    min_bb = FieldBounds(top_left=(1.0, -1.0), bottom_right=(-1.0, 1.0))  # crossed
    strategy = DummyStrategy(min_bb=min_bb)
    strategy.blackboard = make_dummy_blackboard(actual_field)
    with pytest.raises(AssertionError):
        strategy.assert_field_requirements()


def test_min_bb_exceeds_full_field():
    actual_field = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
    min_bb = FieldBounds(top_left=(-5.0, 4.0), bottom_right=(5.0, -4.0))
    strategy = DummyStrategy(min_bb=min_bb)
    strategy.blackboard = make_dummy_blackboard(actual_field)
    with pytest.raises(AssertionError):
        strategy.assert_field_requirements()
