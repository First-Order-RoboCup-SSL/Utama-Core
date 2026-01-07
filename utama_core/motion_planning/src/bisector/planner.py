import logging
from math import dist, exp, pi
from typing import List, Tuple

import numpy as np  # type: ignore

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.game import Game
from utama_core.global_utils.math_utils import normalise_heading
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv

logger = logging.getLogger(__name__)

from utama_core.motion_planning.src.bisector.config import bisectorplannerconfig

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
global counter


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)


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


def intersects_any_polygon(seg_start: np.ndarray, seg_end: np.ndarray, obstacles: List[np.ndarray]) -> bool:
    """Return True if segment intersects or comes within ROBOT_RADIUS of any polygon."""
    for poly in obstacles:
        edges = np.stack([poly, np.roll(poly, -1, axis=0)], axis=1)
        for a, b in edges:
            if segment_to_segment_distance(seg_start, seg_end, a, b) < ROBOT_RADIUS:
                return True
    return False


class BisectorPlanner:

    def __init__(self, env: SSLStandardEnv):
        self._env = env
        self.config = bisectorplannerconfig
        self.CLOSE_LIMIT = self.config.CLOSE_LIMIT
        self.OBSTACLE_CLEARANCE = self.config.ROBOT_DIAMETER
        self.SAMPLE_SIZE = self.config.SAMPLE_SIZE
        self.MAX_VEL = self.config.MAX_VEL

    def _get_obstacles(self, game: Game, robot_id: int) -> List[np.ndarray]:
        robots = (
            list(game.friendly_robots.values())[:robot_id]
            + list(game.friendly_robots.values())[robot_id + 1 :]
            + list(game.enemy_robots.values())
        )
        return [np.array([r.p.x, r.p.y]) for r in robots]

    def perpendicular_bisector(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        mid = (start + end) / 2
        direction = end - start
        perp_dir = rotate_vector(direction, 90)
        normed = perp_dir / np.linalg.norm(perp_dir)
        large = 1000
        return np.stack([mid - large * normed, mid + large * normed])

    def path_to(
        self,
        game: Game,
        robot_id: int,
        target: Tuple[float, float],
        temporary_obstacles: List[np.ndarray] = [],
    ) -> Tuple[float, float]:
        robot = game.friendly_robots[robot_id]
        our_pos = np.array([robot.p.x, robot.p.y])
        target = np.array(target)

        seg_len = distance(our_pos, target)
        if seg_len < self.CLOSE_LIMIT:
            self._env.draw_point(target[0], target[1], width=10)
            return tuple(target)

        perp_line = self.perpendicular_bisector(our_pos, target)
        mid = (perp_line[0] + perp_line[1]) / 2

        if distance(mid, target) < ROBOT_RADIUS:
            self._env.draw_point(target[0], target[1], width=10)
            return tuple(target)

        halves = [np.stack([mid, perp_line[0]]), np.stack([mid, perp_line[1]])]
        obstacles = self._get_obstacles(game, robot_id)

        got = None
        max_dist = max(game.field.half_length * 2, game.field.half_width * 2)
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
        if got is not None:
            self._env.draw_point(got[0], got[1], width=10)
        else:
            self._env.draw_point(mid[0], mid[1], width=10)
        return tuple(got if got is not None else mid)
