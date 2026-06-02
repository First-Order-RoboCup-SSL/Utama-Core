"""WanderingStrategy — base strategy for referee visualisation.

Each robot cycles through its own list of waypoints on the field indefinitely.
When a referee command fires, the RefereeOverride tree (built into AbstractStrategy)
intercepts before this strategy runs, so you can clearly see robots interrupted
and repositioned by the referee.
"""

import py_trees

from utama_core.config.field_params import STANDARD_FIELD_DIMS, FieldDimensions
from utama_core.entities.data.vector import Vector2D
from utama_core.skills.src.utils.move_utils import move
from utama_core.strategy.common import AbstractBehaviour, AbstractStrategy

# Waypoints defined as fractions of the standard field half-dimensions
# (half_length=4.5, half_width=3.0) so they scale correctly to any field.
# Values are in the range (-1, 1) relative to each half-axis.
_WAYPOINT_SETS_NORMALISED = [
    # Robot 0 — large figure-8 across the field
    [(-0.67, 0.50), (0.0, 0.0), (0.67, -0.50), (0.0, 0.0)],
    # Robot 1 — diagonal patrol
    [(-0.44, -0.67), (0.44, 0.67)],
    # Robot 2 — wide horizontal sweep
    [(-0.78, 0.17), (0.78, 0.17), (0.78, -0.17), (-0.78, -0.17)],
    # Robot 3 — small loop near centre
    [(0.22, 0.33), (-0.22, 0.33), (-0.22, -0.33), (0.22, -0.33)],
    # Robot 4 — left-half patrol
    [(-0.67, 0.0), (-0.22, 0.67), (-0.22, -0.67)],
    # Robot 5 — right-half patrol
    [(0.67, 0.0), (0.22, 0.67), (0.22, -0.67)],
]


def _scale_waypoints(
    field_dims: FieldDimensions,
) -> list[list[Vector2D]]:
    """Return waypoint lists scaled to *field_dims*."""
    L = field_dims.full_field_half_length
    W = field_dims.full_field_half_width
    return [[Vector2D(fx * L, fy * W) for fx, fy in pattern] for pattern in _WAYPOINT_SETS_NORMALISED]


_ARRIVAL_THRESHOLD = 0.15  # metres — how close counts as "reached"


class WanderingStep(AbstractBehaviour):
    """Moves each robot through its waypoint list, advancing when it arrives."""

    def __init__(self, waypoints: list[list[Vector2D]], name: str = "WanderingStep"):
        super().__init__(name=name)
        self._waypoints = waypoints

    def initialise(self):
        # Track waypoint index per robot ID
        self._wp_index: dict[int, int] = {}

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        motion_controller = self.blackboard.motion_controller

        robot_ids = sorted(game.friendly_robots.keys())

        for slot, robot_id in enumerate(robot_ids):
            waypoints = self._waypoints[slot % len(self._waypoints)]

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

    Waypoints are scaled to *field_dims* so the strategy works correctly on
    any field size.  Defaults to STANDARD_FIELD_DIMS when omitted, which
    preserves the original behaviour for existing callers.

    Intended for use with the referee visualisation simulation so that referee
    commands visibly interrupt robot motion.
    """

    def __init__(self, field_dims: FieldDimensions | None = None) -> None:
        self._waypoints = _scale_waypoints(field_dims or STANDARD_FIELD_DIMS)
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int) -> bool:
        return True

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool) -> bool:
        return True

    def get_min_bounding_req(self):
        return None

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        root = py_trees.composites.Sequence(name="WanderingRoot", memory=False)
        root.add_child(WanderingStep(self._waypoints))
        return root
