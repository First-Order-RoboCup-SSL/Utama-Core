"""FastPathPlanner must route around the opponent's defense area.

Found via a live grsim run of the split-shape kernel strategy
(`demo_split_shape_match.py`): an outfield attacker chasing the ball drove
straight into the opponent's defense area — an SSL rule violation flagged by
the referee GUI. `FastPathPlanner._get_obstacles` treated only robots and the
field boundary as obstacles; nothing represented the opponent's defense area
at all, so any tactic driving a robot toward a point inside it (directly, or
indirectly via a moving ball) got no avoidance whatsoever. Fixed at the
planner level rather than per-tactic since the rule applies to every
non-goalkeeper robot unconditionally, with no tactic-specific exception.
"""

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.game import Game
from utama_core.run import StrategyRunner
from utama_core.strategy.examples import SimpleNavigationStrategy
from utama_core.team_controller.src.controllers import AbstractSimController
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)

# my_team_is_right=True in the test below means our attacking (enemy) goal is
# on the LEFT — the opponent's defense area sits at the field's negative-x end.
_ENEMY_DEFENSE_AREA_CENTER = (
    -(STANDARD_FIELD_DIMS.full_field_half_length - STANDARD_FIELD_DIMS.half_defense_area_depth),
    0.0,
)


class _StaysOutOfEnemyDefenseAreaManager(AbstractTestManager):
    """Fails immediately if the robot ever enters the enemy defense area.

    The target sits dead-center of the defense area, which is permanently
    unreachable once the area is a respected obstacle — so there is no
    "arrived, stop" signal to succeed on (confirmed experimentally: with the
    fix applied, the robot keeps sliding along the obstacle boundary for the
    entire episode, endlessly re-seeking a closer subgoal to the unreachable
    target, never settling to zero velocity). Success is therefore just
    "survived `_TICKS_REQUIRED` ticks without ever crossing in" — this is a
    negative property (absence of violation), not a destination to reach.
    """

    n_episodes = 1
    _TICKS_REQUIRED = 600

    def __init__(self, robot_id: int):
        super().__init__()
        self.robot_id = robot_id
        self.entered_defense_area = False
        self.survived_full_duration = False
        self._ticks_seen = 0

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        sim_controller.teleport_robot(game.my_team_is_yellow, self.robot_id, 1.5, 0.0, 0.0)

    def eval_status(self, game: Game) -> TestingStatus:
        robot = game.friendly_robots[self.robot_id]
        corners = game.field.enemy_defense_area
        min_x = min(c[0] for c in corners)
        max_x = max(c[0] for c in corners)
        min_y = min(c[1] for c in corners)
        max_y = max(c[1] for c in corners)

        if min_x <= robot.p.x <= max_x and min_y <= robot.p.y <= max_y:
            self.entered_defense_area = True
            return TestingStatus.FAILURE

        self._ticks_seen += 1
        if self._ticks_seen >= self._TICKS_REQUIRED:
            self.survived_full_duration = True
            return TestingStatus.SUCCESS

        return TestingStatus.IN_PROGRESS


def test_robot_targeting_enemy_defense_area_center_is_routed_around_it(headless):
    """A target set dead-center of the opponent's defense area must never be reached directly."""
    my_team_is_yellow = True
    my_team_is_right = True
    robot_id = 0

    runner = StrategyRunner(
        strategy=SimpleNavigationStrategy(
            robot_id=robot_id,
            target_position=_ENEMY_DEFENSE_AREA_CENTER,
            target_orientation=0.0,
        ),
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        mode="rsim",
        exp_friendly=1,
        exp_enemy=0,
        exp_ball=False,
    )

    test_manager = _StaysOutOfEnemyDefenseAreaManager(robot_id=robot_id)
    runner.run_test(test_manager=test_manager, episode_timeout=60.0, rsim_headless=headless)

    # `run_test`'s own return value is `not entered_defense_area` in disguise
    # (FAILURE -> passed=False, SUCCESS -> passed=True) — assert directly on
    # the test manager's own flags instead of that boolean, so a genuine
    # violation reads as "robot entered the defense area" rather than an
    # opaque "test_passed is False".
    assert not test_manager.entered_defense_area, "robot entered the opponent's defense area"
    assert test_manager.survived_full_duration, "episode ended before completing the required tick count"
