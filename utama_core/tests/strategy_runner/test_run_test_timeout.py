"""`run_test`'s episode timeout in rsim is game time, not wall-clock time.

rsim isn't throttled to real time (about 5x on an idle machine), so a wall-clock
timeout gave a test however many simulated seconds the machine's load allowed that
run, and the same test could pass on a quiet machine and fail on a busy one."""

import pytest

from utama_core.run import StrategyRunner
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)
from utama_core.tests.motion_planning._kernel_test_strategies import (
    go_to_point_strategy,
)

_TIMEOUT_S = 1.0


class _NeverFinishes(AbstractTestManager):
    n_episodes = 1

    def __init__(self):
        super().__init__()
        self.first_ts = None
        self.last_ts = None

    def reset_field(self, sim_controller, game):
        pass

    def eval_status(self, game):
        if self.first_ts is None:
            self.first_ts = game.ts
        self.last_ts = game.ts
        return TestingStatus.IN_PROGRESS


def test_an_rsim_episode_times_out_after_its_timeout_in_game_time(headless):
    runner = StrategyRunner(
        strategy=go_to_point_strategy(robot_targets={0: (0.0, 0.0)}),
        my_team_is_yellow=True,
        my_team_is_right=False,
        mode="rsim",
        exp_friendly=1,
        exp_enemy=0,
        exp_ball=False,
    )
    manager = _NeverFinishes()

    assert runner.run_test(test_manager=manager, episode_timeout=_TIMEOUT_S, rsim_headless=headless) is False
    assert manager.last_ts - manager.first_ts == pytest.approx(_TIMEOUT_S, abs=0.05)
