import math
from typing import List, Tuple

import numpy as np  # type: ignore

from utama_core.config.referee_constants import OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
from utama_core.config.settings import CONTROL_FREQUENCY
from utama_core.entities.game import Game
from utama_core.entities.game.field import FieldBounds
from utama_core.global_utils.math_utils import (
    closest_point_on_segment,
    distance,
    distance_between_line_segments,
    distance_point_to_segment,
    find_intersection,
    rotate_vector,
)
from utama_core.motion_planning.src.fastpathplanning.config import (
    fastpathplanningconfig as config,
)
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv


class FastPathPlanner:
    def __init__(self, env: SSLStandardEnv | None):
        self._env = env
        self.config = config
        self.OBSTACLE_CLEARANCE = self.config.OBSTACLE_CLEARANCE
        self.LOOK_AHEAD_RANGE = self.config.LOOK_AHEAD_RANGE
        self.SUBGOAL_DISTANCE = self.config.SUBGOAL_DISTANCE
        self.MAXRECURSIONLENGTH = self.config.MAXRECURSION_LENGTH
        self.PROJECTEDFRAMES = self.config.PROJECTEDFRAMES
        self.PROJECTION_DISTANCE = self.config.PROJECTION_DISTANCE

        # Initialize collision cache dictionary
        self._collision_cache = {}

    def is_point_in_field(self, point, field_bounds: FieldBounds) -> bool:
        x, y = float(point[0]), float(point[1])
        min_x = min(field_bounds.top_left[0], field_bounds.bottom_right[0])
        max_x = max(field_bounds.top_left[0], field_bounds.bottom_right[0])
        min_y = min(field_bounds.top_left[1], field_bounds.bottom_right[1])
        max_y = max(field_bounds.top_left[1], field_bounds.bottom_right[1])
        return min_x <= x <= max_x and min_y <= y <= max_y

    def _enemy_defense_rect(self, game: Game, margin: float) -> Tuple[float, float, float, float]:
        """(min_x, max_x, min_y, max_y) of the opponent's defense area, inflated
        outward by `margin`.

        Every call site currently passes `OPPONENT_DEFENSE_AREA_KEEP_DISTANCE`
        (the same standoff `actions.py` uses during restarts) — a smaller or
        zero margin was tried for the target-legality checks specifically
        (letting a target sit right at the bare rule boundary) but measured to
        still let a fast, head-on approach cross several centimetres into the
        real defense area under PID tracking overshoot; preventing the actual
        SSL violation took priority. KNOWN TRADE-OFF: this also redirects a
        target placed deliberately at/near the exact boundary line even when
        that target isn't itself a violation — e.g. `test_mirror_swap`'s
        formation spots at (3.5, ±0.75), which sit precisely on a
        standard-field defense area's edge, now fail that (synthetic, not
        real-gameplay) test. `margin` is kept as a parameter rather than
        hardcoded so a future, better-tuned fix (e.g. a velocity-aware margin)
        can revisit this per call site without re-deriving the rectangle logic.

        Kept as an explicit rectangle rather than only as obstacle line
        segments: `sanitize_target` pushes a target away from a *nearby*
        segment, but a target deep in the *interior* of an enclosed rectangle
        can be farther than `OBSTACLE_CLEARANCE` from every one of its four
        edges, so the segment-only check never triggers for it at all (found
        via a target set at the defense area's exact center — it passed
        through sanitize_target completely untouched). The rectangle form
        lets `_path_to` check "is this point inside?" directly and project it
        to the nearest edge before anything else runs.
        """
        corners = game.field.enemy_defense_area
        min_x = min(c[0] for c in corners) - margin
        max_x = max(c[0] for c in corners) + margin
        min_y = min(c[1] for c in corners) - margin
        max_y = max(c[1] for c in corners) + margin
        return min_x, max_x, min_y, max_y

    def _project_outside_rect(self, point: np.ndarray, rect: Tuple[float, float, float, float]) -> np.ndarray:
        """Push `point` to the nearest edge of `rect` if it lies inside; otherwise return it unchanged."""
        min_x, max_x, min_y, max_y = rect
        x, y = point[0], point[1]
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return point

        # Distance to each of the 4 edges; move along whichever is nearest.
        dist_to_edge = {
            "left": x - min_x,
            "right": max_x - x,
            "bottom": y - min_y,
            "top": max_y - y,
        }
        nearest = min(dist_to_edge, key=dist_to_edge.get)
        if nearest == "left":
            return np.array([min_x, y])
        if nearest == "right":
            return np.array([max_x, y])
        if nearest == "bottom":
            return np.array([x, min_y])
        return np.array([x, max_y])

    def _get_obstacles(
        self, game: Game, robot_id: int, our_pos: np.ndarray, field_bounds: FieldBounds
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Compiles obstacles and draws projected velocity lines in Red.
        """
        friendly_obstacles = [robot for robot in game.friendly_robots.values() if robot.id != robot_id]
        robots = friendly_obstacles + list(game.enemy_robots.values())
        obstacle_list = []

        for r in robots:
            robot_pos = np.array([r.p.x, r.p.y])
            if distance(our_pos, robot_pos) < self.LOOK_AHEAD_RANGE:
                velocity = np.array([r.v.x, r.v.y])
                # Project the "Ghost Wall" based on current velocity
                point = robot_pos + velocity * (self.PROJECTEDFRAMES / CONTROL_FREQUENCY)
                obstacle_segment = (robot_pos, point)

                obstacle_list.append(obstacle_segment)

                # DRAWING: Show the projected velocity line in Red when an RSim renderer is available.
                if self._env is not None:
                    self._env.draw_line(obstacle_segment, color="Red")

        # Field bounds as obstacles (static, usually not drawn to keep screen clean)
        tl, br = np.array(field_bounds.top_left), np.array(field_bounds.bottom_right)
        tr = np.array([field_bounds.bottom_right[0], field_bounds.top_left[1]])
        bl = np.array([field_bounds.top_left[0], field_bounds.bottom_right[1]])

        obstacle_list.extend([(tl, tr), (tr, br), (br, bl), (bl, tl)])

        # Opponent's defense area is off-limits to every non-goalkeeper robot
        # at all times during live play (SSL rules) — no per-tactic exception
        # exists, so this belongs at the planner level rather than something
        # every tactic has to remember to avoid itself. Our own defense area
        # is deliberately NOT added here: our own goalkeeper (and, briefly,
        # any outfield robot legally retrieving the ball from it) needs to be
        # able to enter it, and this planner has no notion of "except the
        # goalkeeper" to carve that out safely.
        min_x, max_x, min_y, max_y = self._enemy_defense_rect(game, margin=OPPONENT_DEFENSE_AREA_KEEP_DISTANCE)
        c0 = np.array([min_x, max_y])
        c1 = np.array([max_x, max_y])
        c2 = np.array([max_x, min_y])
        c3 = np.array([min_x, min_y])
        obstacle_list.extend([(c0, c1), (c1, c2), (c2, c3), (c3, c0)])

        return obstacle_list

    def _find_subgoal(
        self,
        robot_pos: np.ndarray,
        target: np.ndarray,
        obstacle_pos: np.ndarray,
        obstacles: List,
        subgoal_direction: int,
        multiple: int,
    ) -> np.ndarray:

        # Failsafe to prevent infinite loops if completely trapped
        if multiple > 10:
            return obstacle_pos

        direction = target - robot_pos
        perp_dir = rotate_vector(direction[0], direction[1], math.pi * (subgoal_direction + 0.5))
        unitvec = perp_dir / np.linalg.norm(perp_dir)
        subgoal = obstacle_pos + self.SUBGOAL_DISTANCE * unitvec * multiple

        for o in obstacles:
            # OPTIMIZATION: Removed np.isclose, ensuring strictly less-than for clearance
            if distance_point_to_segment(subgoal, o[0], o[1]) < self.OBSTACLE_CLEARANCE:
                return self._find_subgoal(
                    robot_pos,
                    target,
                    obstacle_pos,
                    obstacles,
                    subgoal_direction,
                    multiple + 1,
                )
        return subgoal

    def collides(self, segment: Tuple, obstacles: List):
        # OPTIMIZATION: Cache collision results (convert numpy arrays to tuples for hashability)
        seg_key = (tuple(segment[0]), tuple(segment[1]))
        if seg_key in self._collision_cache:
            return self._collision_cache[seg_key]

        closest_obstacle = None
        min_dist_to_robot = float("inf")

        for o in obstacles:
            # OPTIMIZATION: Removed double distance call
            dist_between_segs = distance_between_line_segments(o[0], o[1], segment[0], segment[1])

            if dist_between_segs < self.OBSTACLE_CLEARANCE:
                # We want the obstacle closest to the START of the segment (the robot)
                dist_to_robot = distance_point_to_segment(segment[0], o[0], o[1])
                if dist_to_robot < min_dist_to_robot:
                    min_dist_to_robot = dist_to_robot
                    closest_obstacle = o

        obstacle_pos = None
        if closest_obstacle is not None:
            obstacle_pos = find_intersection(segment, closest_obstacle)
            if obstacle_pos is None:
                # Fallback to closest physical point if lines don't strictly intersect
                dists = [
                    distance_point_to_segment(closest_obstacle[0], segment[0], segment[1]),
                    distance_point_to_segment(closest_obstacle[1], segment[0], segment[1]),
                ]
                point_c = closest_point_on_segment(segment[0], closest_obstacle[0], closest_obstacle[1])
                point_d = closest_point_on_segment(segment[1], closest_obstacle[0], closest_obstacle[1])

                dists.extend([distance(segment[0], point_c), distance(segment[1], point_d)])
                points = [closest_obstacle[0], closest_obstacle[1], point_c, point_d]
                obstacle_pos = points[dists.index(min(dists))]

        # Save to cache
        self._collision_cache[seg_key] = obstacle_pos
        return obstacle_pos

    def _trajectory_length(self, trajectory):
        return sum(distance(seg[0], seg[1]) for seg in trajectory)

    def check_segment(
        self,
        segment: Tuple[np.ndarray, np.ndarray],
        obstacles: List[Tuple[np.ndarray, np.ndarray]],
        recursion_length: int,
        target: np.ndarray,
        field_bounds: FieldBounds,
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], float]:
        """
        Recursively checks a segment for collisions and generates subgoals with
        a hysteresis bias to prevent path-switching jitter (indecisiveness).
        """
        closest_obstacle = self.collides(segment, obstacles)
        segment_length = distance(segment[0], segment[1])

        # Base case: Path is clear or maximum detour complexity reached
        if closest_obstacle is None or recursion_length >= self.MAXRECURSIONLENGTH:
            return [segment], segment_length

        # Generate left and right detours
        subgoal_left = self._find_subgoal(segment[0], segment[1], closest_obstacle, obstacles, 1, 1)
        subgoal_right = self._find_subgoal(segment[0], segment[1], closest_obstacle, obstacles, 0, 1)

        left_valid = self.is_point_in_field(subgoal_left, field_bounds)
        right_valid = self.is_point_in_field(subgoal_right, field_bounds)

        best_subgoal = None

        # Heuristic: Pick the valid subgoal closest to the ultimate destination
        if left_valid and right_valid:
            if distance(subgoal_left, target) < distance(subgoal_right, target):
                best_subgoal = subgoal_left
            else:
                best_subgoal = subgoal_right
        elif left_valid:
            best_subgoal = subgoal_left
        elif right_valid:
            best_subgoal = subgoal_right
        else:
            return [segment], segment_length

        # Recursively check the two halves of the selected detour
        seg1, len1 = self.check_segment(
            (segment[0], best_subgoal),
            obstacles,
            recursion_length + 1,
            target,
            field_bounds,
        )
        seg2, len2 = self.check_segment(
            (best_subgoal, segment[1]),
            obstacles,
            recursion_length + 1,
            target,
            field_bounds,
        )

        return seg1 + seg2, len1 + len2

    def _clamp_to_obstacle_clearance(
        self, origin: np.ndarray, unit_vec: np.ndarray, max_distance: float, obstacles: List
    ) -> float:
        """Shrink `max_distance` so the point `origin + unit_vec * distance` never lands
        inside `OBSTACLE_CLEARANCE` of any obstacle segment.

        `smooth_path`'s projected carrot is otherwise unchecked — it is derived fresh from
        `robot_position` and `PROJECTION_DISTANCE` alone, not from `check_segment`'s already
        obstacle-aware trajectory, so nothing stops it from overshooting past a nearby
        obstacle when the robot is already close to one (found via the opponent-defense-area
        obstacle: a robot approaching it head-on crossed the boundary by several centimetres
        even though the underlying trajectory correctly routed around it).
        """
        clamped = max_distance
        for o in obstacles:
            end_point = origin + unit_vec * clamped
            if distance_point_to_segment(end_point, o[0], o[1]) >= self.OBSTACLE_CLEARANCE:
                continue
            # Binary search along the ray for the furthest distance that still
            # keeps clearance — cheap, bounded, and avoids deriving a closed-form
            # ray/segment-clearance solution for what is a rare correction path.
            lo, hi = 0.0, clamped
            for _ in range(12):
                mid = (lo + hi) / 2.0
                point = origin + unit_vec * mid
                if distance_point_to_segment(point, o[0], o[1]) >= self.OBSTACLE_CLEARANCE:
                    lo = mid
                else:
                    hi = mid
            clamped = min(clamped, lo)
        return clamped

    def smooth_path(self, trajectory, target, robot_position, obstacles: List) -> np.ndarray:
        if len(trajectory) == 1:
            return target

        direction = trajectory[0][1] - robot_position
        unit_vec = direction / np.linalg.norm(direction)
        safe_distance = self._clamp_to_obstacle_clearance(robot_position, unit_vec, self.PROJECTION_DISTANCE, obstacles)
        new_target = robot_position + unit_vec * safe_distance

        # Removed redundant math ops by caching distance calls here too
        dist_new_target = distance(new_target, robot_position)
        dist_trajectory = distance(robot_position, trajectory[0][1])

        if dist_new_target < dist_trajectory:
            return trajectory[0][1]
        else:
            point = closest_point_on_segment(new_target, trajectory[1][0], trajectory[1][1])
            return (point + new_target) / 2.0

    def sanitize_target(
        self, target: np.ndarray, obstacles: List, robot_pos: np.ndarray, field_bounds: FieldBounds | None = None
    ) -> np.ndarray:
        """
        Ensures the target isn't inside a velocity-line obstacle.

        Field boundary walls are intentionally excluded here: the ball can
        legally be near the touchline and the robot must be able to reach it.
        Boundary walls are still used in path collision detection so the planner
        never routes *through* the wall — they just don't repel the target.
        """
        if field_bounds is not None:
            # Build the set of field-boundary segments so we can skip them below.
            tl = np.array(field_bounds.top_left)
            br = np.array(field_bounds.bottom_right)
            tr = np.array([br[0], tl[1]])
            bl = np.array([tl[0], br[1]])
            boundary_segments = {
                (tuple(tl), tuple(tr)),
                (tuple(tr), tuple(br)),
                (tuple(br), tuple(bl)),
                (tuple(bl), tuple(tl)),
            }
        else:
            boundary_segments = set()

        safe_target = np.copy(target)
        for _ in range(5):
            collision_found = False
            for o in obstacles:
                if (tuple(o[0]), tuple(o[1])) in boundary_segments:
                    continue
                if distance_point_to_segment(safe_target, o[0], o[1]) < self.OBSTACLE_CLEARANCE:
                    closest_pt = closest_point_on_segment(safe_target, o[0], o[1])
                    push_dir = safe_target - closest_pt
                    if np.linalg.norm(push_dir) == 0:
                        push_dir = robot_pos - closest_pt
                    unit_push = push_dir / np.linalg.norm(push_dir)
                    safe_target = closest_pt + unit_push * (self.OBSTACLE_CLEARANCE * 1.05)
                    collision_found = True
            if not collision_found:
                break
        return safe_target

    def _path_to(
        self,
        game: Game,
        robot_id: int,
        target: Tuple[float, float],
        field_bounds: FieldBounds,
    ):
        """
        Main entry point. Clears cache, sanitizes target, and plans path.
        """
        self._collision_cache.clear()

        robot = game.friendly_robots[robot_id]
        our_pos = np.array([robot.p.x, robot.p.y])
        raw_target = np.array(target)

        # 1. Get obstacles and draw Red velocity lines
        obstacles = self._get_obstacles(game, robot_id, our_pos, field_bounds)

        # 2. A target inside the (enclosed) opponent defense area needs its own
        # check: `sanitize_target` below only reacts to a target close to an
        # obstacle *line*, so a point deep in a rectangle's interior — farther
        # than OBSTACLE_CLEARANCE from all four edges — passes through it
        # completely untouched (confirmed with a target at the rectangle's
        # exact center). Project it to the nearest edge first so the rest of
        # the pipeline only ever has to reason about a target near a boundary.
        # Uses the inflated margin, not the bare boundary: KNOWN TRADE-OFF —
        # this also redirects a target placed deliberately at/near the exact
        # boundary line (e.g. `test_mirror_swap`'s formation spots at
        # (3.5, ±0.75), which sit precisely on a standard-field defense area's
        # edge) even though that target isn't itself a rule violation. Chosen
        # anyway because the smaller, boundary-exact margin was tried and
        # measured to still let a fast, head-on approach cross several
        # centimetres into the real defense area (see git history / design
        # doc) — preventing the actual SSL violation took priority over this
        # synthetic test's exact-boundary formation targets.
        raw_target = self._project_outside_rect(
            raw_target, self._enemy_defense_rect(game, OPPONENT_DEFENSE_AREA_KEEP_DISTANCE)
        )

        # 3. Sanitize target — skip boundary walls so robots can reach the ball near touchlines
        safe_target = self.sanitize_target(raw_target, obstacles, our_pos, field_bounds)

        # 4. Plan geometric path
        final_trajectory, _ = self.check_segment((our_pos, safe_target), obstacles, 0, safe_target, field_bounds)

        # 5. Draw the resulting safe path segments when an RSim renderer is available.
        if self._env is not None:
            for i in final_trajectory:
                self._env.draw_line(i)

        # 6. Smooth the path and draw the final "Carrot" target in Blue
        new_target = self.smooth_path(final_trajectory, safe_target, our_pos, obstacles)

        # 7. Last-line safety net, specific to the defense-area rectangle: the
        # smoothing/blending steps above (subgoal search, carrot projection,
        # final subgoal/carrot averaging) each place intermediate points right
        # at OBSTACLE_CLEARANCE from a line obstacle by design, and none of
        # them are individually checked against being *inside* an enclosed
        # rectangle afterward — found by tracing a real crossing where the
        # final blended waypoint landed a few centimetres inside the rule
        # boundary despite every upstream step "correctly" respecting
        # clearance from the lines it was reasoning about individually.
        # Projecting the final result away from the rectangle is cheap and
        # certain, versus continuing to chase which specific blend step needs
        # its own clearance check. Same inflated-margin trade-off as the
        # interior-target check above (see its comment) — this is a waypoint
        # the robot is actively driving toward, not a stationary destination,
        # so a smaller margin was measured to still let the robot cross the
        # real boundary under momentum.
        new_target = self._project_outside_rect(
            new_target, self._enemy_defense_rect(game, OPPONENT_DEFENSE_AREA_KEEP_DISTANCE)
        )

        if self._env is not None:
            self._env.draw_line((our_pos, new_target), color="Blue")

        return new_target
