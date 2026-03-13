"""WanderingStrategy — base strategy for referee visualisation.

Each robot cycles through its own list of waypoints on the field indefinitely.
When a referee command fires, the RefereeOverride tree (built into AbstractStrategy)
intercepts before this strategy runs, so you can clearly see robots interrupted
and repositioned by the referee.
"""

import math

import py_trees

from utama_core.entities.data.vector import Vector2D
from utama_core.skills.src.utils.move_utils import move
from utama_core.strategy.common import AbstractBehaviour, AbstractStrategy

# One waypoint list per robot (by index into sorted robot IDs).
# Robots on the right half defend the right goal, so positions are spread
# across both halves to make motion easy to see.
_WAYPOINT_SETS = [
    # Robot 0 — large figure-8 across the field
    [
        Vector2D(-3.0, 1.5),
        Vector2D(0.0, 0.0),
        Vector2D(3.0, -1.5),
        Vector2D(0.0, 0.0),
    ],
    # Robot 1 — diagonal patrol
    [
        Vector2D(-2.0, -2.0),
        Vector2D(2.0, 2.0),
    ],
    # Robot 2 — wide horizontal sweep
    [
        Vector2D(-3.5, 0.5),
        Vector2D(3.5, 0.5),
        Vector2D(3.5, -0.5),
        Vector2D(-3.5, -0.5),
    ],
    # Robot 3 — small loop near centre
    [
        Vector2D(1.0, 1.0),
        Vector2D(-1.0, 1.0),
        Vector2D(-1.0, -1.0),
        Vector2D(1.0, -1.0),
    ],
    # Robot 4 — left-half patrol
    [
        Vector2D(-3.0, 0.0),
        Vector2D(-1.0, 2.0),
        Vector2D(-1.0, -2.0),
    ],
    # Robot 5 — right-half patrol
    [
        Vector2D(3.0, 0.0),
        Vector2D(1.0, 2.0),
        Vector2D(1.0, -2.0),
    ],
]

_ARRIVAL_THRESHOLD = 0.15  # metres — how close counts as "reached"


class WanderingStep(AbstractBehaviour):
    """Moves each robot through its waypoint list, advancing when it arrives."""

    def initialise(self):
        # Track waypoint index per robot ID
        self._wp_index: dict[int, int] = {}

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        motion_controller = self.blackboard.motion_controller

        robot_ids = sorted(game.friendly_robots.keys())

        for slot, robot_id in enumerate(robot_ids):
            waypoints = _WAYPOINT_SETS[slot % len(_WAYPOINT_SETS)]

            if robot_id not in self._wp_index:
                self._wp_index[robot_id] = 0

            wp_idx = self._wp_index[robot_id]
            target = waypoints[wp_idx]

            robot = game.friendly_robots[robot_id]
            dist = robot.p.distance_to(target)

            if dist < _ARRIVAL_THRESHOLD:
                # Advance to next waypoint
                self._wp_index[robot_id] = (wp_idx + 1) % len(waypoints)
                target = waypoints[self._wp_index[robot_id]]

            oren = robot.p.angle_to(target)
            self.blackboard.cmd_map[robot_id] = move(game, motion_controller, robot_id, target, oren)

        return py_trees.common.Status.RUNNING


class WanderingStrategy(AbstractStrategy):
    """Strategy where every robot continuously patrols a set of waypoints.

    Intended for use with the referee visualisation simulation so that referee
    commands visibly interrupt robot motion.
    """

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int) -> bool:
        return True

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool) -> bool:
        return True

    def get_min_bounding_zone(self):
        return None

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        root = py_trees.composites.Sequence(name="WanderingRoot", memory=False)
        root.add_child(WanderingStep())
        return root
