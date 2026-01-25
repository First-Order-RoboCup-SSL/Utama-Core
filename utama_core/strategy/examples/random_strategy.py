import random
from typing import Dict, List, Optional, Tuple

import py_trees
from py_trees.composites import Sequence

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game.field import FieldBounds
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.strategy.common import AbstractBehaviour, AbstractStrategy


def generate_preset_roads(
    field_bounds: FieldBounds,
    robot_ids: list[int],
    margin: float,
    seed: Optional[int] = None,
    shuffle: bool = False,
) -> Dict[int, List[Tuple[float, float]]]:
    min_x = field_bounds.top_left[0] + margin
    max_x = field_bounds.bottom_right[0] - margin
    min_y = field_bounds.bottom_right[1] + margin
    max_y = field_bounds.top_left[1] - margin

    if min_x > max_x or min_y > max_y:
        min_x = field_bounds.top_left[0]
        max_x = field_bounds.bottom_right[0]
        min_y = field_bounds.bottom_right[1]
        max_y = field_bounds.top_left[1]

    base_road = [
        (min_x, min_y),
        (min_x, max_y),
        (max_x, max_y),
        (max_x, min_y),
    ]
    if shuffle and seed is not None:
        shuffled = base_road[:]
        random.Random(seed).shuffle(shuffled)
        base_road = shuffled

    roads: Dict[int, List[Tuple[float, float]]] = {}
    for robot_id in sorted(robot_ids):
        shift = robot_id % len(base_road)
        roads[robot_id] = base_road[shift:] + base_road[:shift]
    return roads


class PresetRoadStep(AbstractBehaviour):
    """
    A behaviour that commands all robots to follow pre-set looped roads.

    **Returns:**
        - `py_trees.common.Status.RUNNING`: The behaviour is actively commanding the robot to move.
    """

    def __init__(
        self,
        margin: float = 0.2,
        waypoint_tolerance: float = 0.15,
        seed: Optional[int] = None,
        shuffle: bool = False,
    ):
        super().__init__()
        self.margin = margin
        self.waypoint_tolerance = waypoint_tolerance
        self.seed = seed
        self.shuffle = shuffle
        self._roads: Dict[int, List[Tuple[float, float]]] = {}
        self._indices: Dict[int, int] = {}

    def initialise(self) -> None:
        game = self.blackboard.game
        field_bounds = game.field.field_bounds
        robot_ids = list(game.friendly_robots.keys())

        self._roads = generate_preset_roads(
            field_bounds,
            robot_ids,
            self.margin,
            self.seed,
            shuffle=self.shuffle,
        )
        self._indices = {robot_id: 0 for robot_id in robot_ids}

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        motion_controller = self.blackboard.motion_controller

        for robot_id in game.friendly_robots.keys():
            if robot_id not in self._roads:
                field_bounds = game.field.field_bounds
                self._roads.update(
                    generate_preset_roads(
                        field_bounds,
                        [robot_id],
                        self.margin,
                        self.seed,
                        shuffle=self.shuffle,
                    )
                )
                self._indices[robot_id] = 0

            road = self._roads[robot_id]
            if not road:
                continue

            target_index = self._indices.get(robot_id, 0)
            tx, ty = road[target_index]

            robot = game.friendly_robots.get(robot_id)
            if robot is not None:
                robot_pos = Vector2D(robot.p.x, robot.p.y)
                if robot_pos.distance_to(Vector2D(tx, ty)) <= self.waypoint_tolerance:
                    self._indices[robot_id] = (target_index + 1) % len(road)
                    tx, ty = road[self._indices[robot_id]]

            command = go_to_point(
                game,
                motion_controller,
                robot_id,
                Vector2D(tx, ty),
                False,
            )
            self.blackboard.cmd_map[robot_id] = command

        return py_trees.common.Status.RUNNING


class RandomStrategy(AbstractStrategy):
    def __init__(self, seed: Optional[int] = None, shuffle_roads: bool = False):
        self.seed = seed
        self.shuffle_roads = shuffle_roads
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        return True  # No specific robot count requirements

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool):
        return True  # No specific goal line requirements

    def get_min_bounding_zone(self) -> FieldBounds:
        return self.blackboard.game.field.field_bounds

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Factory function to create a complete behaviour tree."""

        coach_root = Sequence(name="CoachRoot", memory=False)

        ### Assemble the tree ###

        coach_root.add_children([PresetRoadStep(seed=self.seed, shuffle=self.shuffle_roads)])

        return coach_root
