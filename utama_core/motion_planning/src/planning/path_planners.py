import logging
import random
from math import dist, pi
from typing import Generator, List, Optional, Tuple, Union

import numpy as np

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.game import Game
from utama_core.motion_planning.src.planning.geometry import (
    AxisAlignedRectangle,
    point_segment_distance,
)
from utama_core.motion_planning.src.planning.obstacles import (
    ObstacleRegion,
    to_rectangles,
)

logger = logging.getLogger(__name__)

ROBOT_DIAMETER = 2 * ROBOT_RADIUS

"""
TODO -
Edge cases:
    target inside obstacle
    target starts within / too close to obstacle 

Drift
Cleanup so that it takes a robot for the path every time
Magic numbers

Motion controller stateful with waypoints - take robot index or separate one for each robot
Inner stateless one does the path planning
 -> Take local planning if possible
 -> Otherwise make a new global tree and go towards next waypoint until you get there and then use local planning
Fix clearance
Make Field stuff static and fix the dimensions (2.15)
Test with motion
Slow motion and never gets there.
"""

# ---------- Basic geometry helpers ----------


def point_to_tuple(point: np.ndarray) -> Tuple[float, float]:
    return (float(point[0]), float(point[1]))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)


def interpolate_segment(p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
    """Interpolate along the line from p1→p2 by fraction t (0–1)."""
    return p1 + t * (p2 - p1)


def segment_length(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p2 - p1)


def point_to_segment_distance(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
    """Compute the shortest distance between a point and a line segment."""
    seg_vec = seg_end - seg_start
    t = np.clip(np.dot(point - seg_start, seg_vec) / np.dot(seg_vec, seg_vec), 0, 1)
    proj = seg_start + t * seg_vec
    return np.linalg.norm(point - proj)


def segment_to_segment_distance(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> float:
    """Shortest distance between two line segments in 2D."""


    u = a2 - a1
    v = b2 - b1
    w = a1 - b1
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)
    D = a * c - b * b
    sc, sN, sD = D, D, D
    tc, tN, tD = D, D, D
    SMALL_NUM = 1e-9

    if D < SMALL_NUM:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = b * e - c * d
        tN = a * e - b * d
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = -d + b
            sD = a

    sc = 0.0 if abs(sN) < SMALL_NUM else sN / sD
    tc = 0.0 if abs(tN) < SMALL_NUM else tN / tD
    dP = w + (sc * u) - (tc * v)
    return np.linalg.norm(dP)


def rotate_vector(vec: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return vec @ rot.T


# ---------- Collision helpers ----------


def intersects_any_polygon(seg_start: np.ndarray, seg_end: np.ndarray, obstacles: List[np.ndarray]) -> bool:
    """Return True if segment intersects or comes within ROBOT_RADIUS of any polygon."""
    for poly in obstacles:
        edges = np.stack([poly, np.roll(poly, -1, axis=0)], axis=1)
        for a, b in edges:
            if segment_to_segment_distance(seg_start, seg_end, a, b) < ROBOT_RADIUS:
                return True
    return False


# ---------- Core planners ----------


class RRTPlanner:
    SAFE_OBSTACLES_RADIUS = 2 * ROBOT_RADIUS + 0.08
    STOPPING_DISTANCE = 0.2
    EXPLORE_BIAS = 0.1
    STEP_SIZE = 0.15
    GOOD_ENOUGH_REL = 1.2
    GOOD_ENOUGH_ABS = 1

    def __init__(self, game: Game):
        self._game = game
        self.waypoints = []
        self.par = dict()

    def _get_obstacles(self, robot_id: int) -> List[np.ndarray]:
        robots = (
            self._game.friendly_robots[:robot_id] + self._game.friendly_robots[robot_id + 1 :] + self._game.enemy_robots
        )
        return [np.array([r.x, r.y]) for r in robots]

    def _closest_obstacle(self, robot_id: int, seg_start: np.ndarray, seg_end: Optional[np.ndarray] = None) -> float:
        """Return minimum distance to any robot obstacle."""
        obs = self._get_obstacles(robot_id)
        if not obs:
            return float('inf')
        if seg_end is None:
            return min(distance(o, seg_start) for o in obs)
        return min(point_to_segment_distance(o, seg_start, seg_end) for o in obs)

    def path_to(
        self,
        friendly_robot_id: int,
        target: Tuple[float, float],
        max_iterations: int = 3000,
    ) -> Optional[List[Tuple[float, float]]]:
        robot = self._game.friendly_robots[friendly_robot_id]
        start = np.array([robot.x, robot.y])
        goal = np.array(target)

        if self._closest_obstacle(friendly_robot_id, goal) < ROBOT_DIAMETER / 2:
            return [tuple(start)]

        if distance(start, goal) < ROBOT_DIAMETER / 2:
            return [tuple(goal)]

        if self._closest_obstacle(friendly_robot_id, start, goal) > 3 * self.SAFE_OBSTACLES_RADIUS:
            return [tuple(goal)]

        nodes = [start]
        self.par = {tuple(start): None}
        cost_map = {tuple(start): 0}
        path_found = False

        for its in range(max_iterations):
            rand_point = (
                np.array(
                    [
                        random.uniform(-self._game.field.half_length, self._game.field.half_length),
                        random.uniform(-self._game.field.half_width, self._game.field.half_width),
                    ]
                )
                if random.random() < self.EXPLORE_BIAS
                else goal
            )

            closest = min(nodes, key=lambda n: distance(n, rand_point))
            direction = rand_point - closest
            if np.linalg.norm(direction) == 0:
                continue
            direction /= np.linalg.norm(direction)
            new_point = closest + self.STEP_SIZE * direction

            if self._closest_obstacle(friendly_robot_id, closest, new_point) > self.SAFE_OBSTACLES_RADIUS:
                nodes.append(new_point)
                self.par[tuple(new_point)] = tuple(closest)
                cost_map[tuple(new_point)] = cost_map[tuple(closest)] + distance(closest, new_point)

                if distance(new_point, goal) < self.STOPPING_DISTANCE:
                    self.par[tuple(goal)] = tuple(new_point)
                    path_found = True
                    break

        if not path_found:
            return None

        # reconstruct path
        path = []
        cur = tuple(goal)
        while cur is not None:
            path.append(cur)
            cur = self.par.get(cur)
        path.reverse()
        return path


class BisectorPlanner:
    OBSTACLE_CLEARANCE = ROBOT_DIAMETER
    CLOSE_LIMIT = 0.5
    SAMPLE_SIZE = 0.10

    def __init__(self, game, friendly_colour, env):
        self._game = game
        self._friendly_colour = friendly_colour
        self._env = env

    def _get_obstacles(self, robot_id: int) -> List[np.ndarray]:
        robots = (
            self._game.friendly_robots[:robot_id] + self._game.friendly_robots[robot_id + 1 :] + self._game.enemy_robots
        )
        return [np.array([r.x, r.y]) for r in robots]

    def perpendicular_bisector(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        mid = (start + end) / 2
        direction = end - start
        perp_dir = rotate_vector(direction, 90)
        normed = perp_dir / np.linalg.norm(perp_dir)
        large = 1000
        return np.stack([mid - large * normed, mid + large * normed])

    def path_to(
        self,
        robot_id: int,
        target: Tuple[float, float],
        temporary_obstacles: List[np.ndarray] = [],
    ) -> Tuple[float, float]:
        robot = self._game.friendly_robots[robot_id]
        our_pos = np.array([robot.x, robot.y])
        target = np.array(target)

        seg_len = distance(our_pos, target)
        if seg_len < self.CLOSE_LIMIT:
            return tuple(target)

        perp_line = self.perpendicular_bisector(our_pos, target)
        mid = (perp_line[0] + perp_line[1]) / 2

        if distance(mid, target) < ROBOT_RADIUS:
            return tuple(target)

        halves = [np.stack([mid, perp_line[0]]), np.stack([mid, perp_line[1]])]
        obstacles = self._get_obstacles(robot_id)

        got = None
        max_dist = max(self._game.field.half_length * 2, self._game.field.half_width * 2)
        samples = int(max_dist / self.SAMPLE_SIZE)

        for s in range(samples):
            offset = s * self.SAMPLE_SIZE
            for h in halves:
                direction = h[1] - h[0]
                direction /= np.linalg.norm(direction)
                p1 = h[0] + offset * direction

                seg1_start, seg1_end = our_pos, p1
                seg2_start, seg2_end = p1, target

                if all(
                    point_to_segment_distance(o, seg1_start, seg1_end) > self.OBSTACLE_CLEARANCE for o in obstacles
                ) and all(
                    point_to_segment_distance(o, seg2_start, seg2_end) > self.OBSTACLE_CLEARANCE for o in obstacles
                ):
                    if not intersects_any_polygon(
                        seg1_start, seg1_end, temporary_obstacles
                    ) and not intersects_any_polygon(seg2_start, seg2_end, temporary_obstacles):
                        got = p1
                        break
            if got is not None:
                break

        return tuple(got if got is not None else mid)
