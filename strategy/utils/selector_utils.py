import py_trees
import numpy as np
from strategy.common.abstract_behaviour import AbstractBehaviour

from entities.game import Game
from entities.data.vector import Vector2D


class HasBall(AbstractBehaviour):
    """
    Checks if the specified robot currently has possession of the ball.

    This behavior is a condition that reads the `has_ball` attribute of a
    robot from the game state. It's used to verify if a robot has
    successfully collected the ball.

    **Blackboard Interaction:**
        Reads:
            - `robot_id` (int): The ID of the robot to check for ball possession. Typically from the `SetBlackboardVariable` node.

    **Returns:**
        - `py_trees.common.Status.SUCCESS`: If the robot has the ball.
        - `py_trees.common.Status.FAILURE`: Otherwise.
    """
        
    def __init__(self, visual: bool = False, ball_capture_dist: float = 0.15):
        self.visual = visual
        self.ball_capture_dist = float(ball_capture_dist)
        super().__init__()

    def setup_(self):
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)

    def update(self):
        # print(f"Checking if robot {self.blackboard.robot_id} has the ball")
        game = self.blackboard.game
        robot_id = self.blackboard.robot_id
        
        if self.visual:
            return self._has_ball_visual(game, robot_id)
        else:
            return self._has_ball_from_state(game, robot_id)
    
    def _has_ball_visual(self, game: Game, robot_id: int) -> py_trees.common.Status:
        """
        Visual possession: success if the robot is within `ball_capture_radius` of the ball.
        Uses squared distance (no sqrt) for speed.
        """
        robot = game.friendly_robots[robot_id]
        ball = game.ball
        
        r_pos = Vector2D(robot.p.x, robot.p.y)
        b_pos = Vector2D(ball.p.x, ball.p.y)

        dist_sq = r_pos.distance_to(b_pos)

        return (
            py_trees.common.Status.SUCCESS
            if dist_sq < self.ball_capture_dist
            else py_trees.common.Status.FAILURE
        )
    
    def _has_ball_from_state(self, game: Game, robot_id: int) -> py_trees.common.Status:
        """
        State possession: success if the game state's `has_ball` flag is true.
        """
        has_ball = game.current.friendly_robots[robot_id].has_ball

        return py_trees.common.Status.SUCCESS if has_ball else py_trees.common.Status.FAILURE


class GoalScored(AbstractBehaviour):
    """
    Checks if a goal has been scored by either team.

    This behavior is a condition that checks if the ball has crossed either
    side of the pitch by examining its x-coordinate. It provides a simple
    way to detect a score and trigger a change in strategy.

    **Returns:**
        - `py_trees.common.Status.SUCCESS`: If the ball's absolute x-coordinate > 4.5.
        - `py_trees.common.Status.FAILURE`: Otherwise.
    """

    def update(self):
        if abs(self.blackboard.game.current.ball.p.x) > 4.5:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class AtDribbleToTarget(AbstractBehaviour):
    """
    Checks if the controlled robot is within a tolerance of a target point.

    **Blackboard Interaction:**
        Reads:
            - ``robot_id`` (int): ID of the robot to check.
            - ``target_coords`` (Vector2D): Desired target location.

    Args:
        tolerance (float): Maximum allowed distance to consider the robot at the target.
    """

    def __init__(self, tolerance: float, name: str = "AtTarget"):
        super().__init__(name=name)
        self.tolerance = tolerance

    def setup_(self):
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="target_coords", access=py_trees.common.Access.WRITE)

    def initialise(self):
        self.target = self._dribble_to_target_decision_maker()
        return super().initialise()
    
    def _dribble_to_target_decision_maker(
        self,
        safe_distance: float = 0.5,
        wall_margin: float = 0.35,
        wall_influence: float = 1.2,
        corner_push_gain: float = 0.9,
        step_size: float = 0.6,
    ) -> Vector2D:
        """
        Decide a dribble target that progresses to goal, avoids the nearest enemy,
        and never gets trapped in the corners on a 6x9 m field (assumed centered at (0,0)).

        - Blends goal-seeking, enemy-evade (perpendicular + away), and wall/corner repulsion.
        - Applies a center bias that increases near corners.
        """

        game = self.blackboard.game.current
        robot = game.friendly_robots[self.blackboard.robot_id]

        # --- Field geometry (6 x 9 m). Many SSL sims center the field at (0,0).
        # If your sim uses different origin, adapt these two lines accordingly.
        FIELD_HALF_Y = 3.0   # total height 6 m  -> y in [-3, +3]
        FIELD_HALF_X = 4.5   # total length 9 m  -> x in [-4.5, +4.5]

        # Enemy goal line midpoint (aim at center of enemy goal mouth by default)
        target_goal_line = game.field.enemy_goal_line
        goal_x = target_goal_line.coords[0][0]
        goal_y = 0.0

        rx, ry = robot.p.x, robot.p.y

        # --- Vector helpers
        def vec(x, y):
            return np.array([x, y], dtype=float)

        def nrm(v, eps=1e-9):
            n = np.linalg.norm(v)
            return v / (n + eps)

        def clamp_to_field(p):
            # keep inside a safe interior rectangle
            cx = np.clip(p[0], -FIELD_HALF_X + wall_margin, FIELD_HALF_X - wall_margin)
            cy = np.clip(p[1], -FIELD_HALF_Y + wall_margin, FIELD_HALF_Y - wall_margin)
            return np.array([cx, cy], dtype=float)

        rpos = vec(rx, ry)

        # --- Goal-seeking
        g_dir = nrm(vec(goal_x, goal_y) - rpos)

        # --- Nearest enemy + evade vector
        nearest_enemy = None
        nearest_d = float('inf')
        for e in game.enemy_robots.values():
            d = np.hypot(rx - e.p.x, ry - e.p.y)
            if d < nearest_d:
                nearest_d = d
                nearest_enemy = e

        # If no enemies are seen, just push to goal while respecting walls/corners.
        if nearest_enemy is None:
            enemy_away = np.zeros(2)
            enemy_perp = np.zeros(2)
            d_enemy = float('inf')
        else:
            ex, ey = nearest_enemy.p.x, nearest_enemy.p.y
            enemy_vec = rpos - vec(ex, ey)          # from enemy to us
            d_enemy = max(np.linalg.norm(enemy_vec), 1e-6)
            enemy_away = nrm(enemy_vec)
            # Perpendicular direction (choose the sign that slightly aligns with goal progress)
            perp1 = nrm(np.array([-enemy_vec[1], enemy_vec[0]]))
            perp2 = -perp1
            # pick the perpendicular that has better dot with g_dir
            enemy_perp = perp1 if float(perp1 @ g_dir) >= float(perp2 @ g_dir) else perp2

        # --- Wall & corner repulsion (soft potentials within wall_influence)
        def wall_repulsion(p):
            x, y = p
            rep = np.zeros(2)

            # distances to each wall (inside-positive)
            d_left   = (x + FIELD_HALF_X)         # distance to left boundary x=-FHX
            d_right  = (FIELD_HALF_X - x)         # to right boundary x=+FHX
            d_bottom = (y + FIELD_HALF_Y)         # to bottom boundary y=-FHY
            d_top    = (FIELD_HALF_Y - y)         # to top boundary y=+FHY

            def push(mag, dir_vec):
                # inverse-square inside influence radius
                return (1.0 / (mag*mag + 1e-6)) * dir_vec

            # Only apply if within influence distance (beyond margin)
            # Push direction points back toward field center from that wall.
            if d_left   < wall_influence: rep += push(max(d_left, 0.05),  vec(+1,  0))
            if d_right  < wall_influence: rep += push(max(d_right, 0.05), vec(-1,  0))
            if d_bottom < wall_influence: rep += push(max(d_bottom, 0.05),vec( 0, +1))
            if d_top    < wall_influence: rep += push(max(d_top, 0.05),   vec( 0, -1))

            # Corner boost: if simultaneously close to a vertical and a horizontal wall,
            # add an extra pull toward the field center.
            near_vert = (d_left < wall_influence) or (d_right < wall_influence)
            near_horz = (d_bottom < wall_influence) or (d_top < wall_influence)
            if near_vert and near_horz:
                rep += corner_push_gain * nrm(vec(0, 0) - p)

            return rep

        w_rep = wall_repulsion(rpos)

        # --- Blending weights
        # Enemy very close -> emphasize perpendicular dodge; far -> emphasize goal
        # Use a smooth falloff around safe_distance
        t = np.clip((d_enemy - safe_distance) / max(safe_distance, 1e-6), 0.0, 1.0)
        # t=0   : enemy on top of us -> evade heavy
        # t=1   : enemy far           -> goal heavy

        w_goal  = 0.45 * t + 0.15          # 0.15 .. 0.60
        w_perp  = 0.55 * (1.0 - t)         # 0.55 .. 0.00
        w_away  = 0.20 * (1.0 - t)         # 0.20 .. 0.00
        w_wall  = 0.35                     # constant wall/corner awareness

        # --- Compose desired direction
        desired_dir = (
            w_goal * g_dir +
            w_perp * enemy_perp +
            w_away * enemy_away +
            w_wall * nrm(w_rep)  # normalize so scale ≈ weight
        )

        # If we’ve accidentally nullified direction, fall back to goal
        if np.linalg.norm(desired_dir) < 1e-6:
            desired_dir = g_dir

        # --- Take a step and clamp inside the safe interior
        target = rpos + nrm(desired_dir) * step_size
        target = clamp_to_field(target)

        # --- Extra "never trap" safeguard:
        # If y is still very close to top/bottom after clamping, nudge target toward center line (y=0).
        near_top_or_bottom = (FIELD_HALF_Y - abs(target[1])) < (wall_margin + 0.05)
        if near_top_or_bottom:
            target[1] *= 0.7  # pull toward center to open the angle

        # If close to both a vertical and horizontal wall, bias x slightly toward center too.
        near_left_or_right = (FIELD_HALF_X - abs(target[0])) < (wall_margin + 0.05)
        if near_top_or_bottom and near_left_or_right:
            target *= 0.6  # stronger pull to the middle when right in a corner pocket

        # --- Stop condition near goal (keep your original behavior)
        goal_dist = np.hypot(goal_x - rx, goal_y - ry)
        if goal_dist < 2.0:
            return None

        return Vector2D(float(target[0]), float(target[1]))

        
    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game.current
        robot = game.friendly_robots[self.blackboard.robot_id]
        self.blackboard.set("target_coords", self.target, overwrite=True)
        current_point = Vector2D(robot.p.x, robot.p.y)
        
        if self.target is not None:
            distance_to_target = current_point.distance_to(self.target)
            if distance_to_target <= self.tolerance:
                return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class DribbledEnough(AbstractBehaviour):
    """
    Checks if the robot has dribbled the ball beyond a specified distance.

    **Blackboard Interaction:**
        Reads:
            - ``dribbled_distance`` (float): Distance the robot has dribbled the ball.

    Args:
        limit (float): Distance threshold for dribbling.
    """

    def __init__(self, limit: float, name: str = "DribbledEnough"):
        super().__init__(name=name)
        self.limit = limit

    def setup_(self):
        self.blackboard.register_key(
            key="dribbled_distance", access=py_trees.common.Access.WRITE
        )

    def update(self) -> py_trees.common.Status:
        if self.blackboard.dribbled_distance >= self.limit * 0.9:
            # print(f"Dribbled Enough: {self.blackboard.dribbled_distance} >= {self.limit}")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE
