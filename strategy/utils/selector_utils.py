import py_trees
import numpy as np
from strategy.common.abstract_behaviour import AbstractBehaviour

from entities.game import Game
from entities.data.vector import Vector2D


class HasBall(AbstractBehaviour):
    """Checks if the specified robot currently has possession of the ball.

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
    """Checks if a goal has been scored by either team.

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


class AtDribbleTarget(AbstractBehaviour):
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
        # persistence for anti-stuck & hysteresis
        self.prev_target = None       # (x,y)
        self.prev_target_dist = None  # float
        self.prev_rpos = None         # (x,y)
        self.stuck_ticks = 0
        # new: perpendicular side hysteresis
        self.perp_sign = 0            # +1/-1 for which perpendicular, 0 = unset
        self.perp_cooldown = 0        # ticks before allowing another flip
        # cache latest target each tick
        self.target = None
        self.lock_ticks_left = 0      # dwell on current target
        self.escape_ticks_left = 0    # hold an escape target for N ticks
        self.escape_target = None     # (x,y) escape point to hold
        self.prev_move_dir = None     # for a tiny motion "momentum"


    def setup_(self):
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="target_coords", access=py_trees.common.Access.WRITE)

    def initialise(self):
        # compute in update() each tick (so hysteresis can act)
        self.target = None
        return super().initialise()
        
    def update(self) -> py_trees.common.Status:
        # recompute each tick with hysteresis
        self.target = self._dribble_to_target_decision_maker()
        self.blackboard.set("target_coords", self.target, overwrite=True)

        game = self.blackboard.game.current
        robot = game.friendly_robots[self.blackboard.robot_id]
        print(f"target: {self.target}, robot: {robot.p}")

        if self.target is not None:
            distance_to_target = Vector2D(robot.p.x, robot.p.y).distance_to(self.target)
            if distance_to_target <= self.tolerance:
                return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE
    
    def _dribble_to_target_decision_maker(
        self,
        safe_distance: float = 0.7,
        wall_margin: float = 0.4,
        wall_influence: float = 1.2,
        corner_push_gain: float = 0.8,
        step_size: float = 0.6,
    ) -> Vector2D:
        game = self.blackboard.game.current
        robot = game.friendly_robots[self.blackboard.robot_id]

        FIELD_HALF_Y = 3.0
        FIELD_HALF_X = 4.5

        target_goal_line = game.field.enemy_goal_line
        goal_x = target_goal_line.coords[0][0]
        goal_y = 0.0

        rx, ry = robot.p.x, robot.p.y

        # --- helpers
        def vec(x, y): return np.array([x, y], dtype=float)

        def nrm(v, eps=1e-9):
            n = np.linalg.norm(v)
            return v / (n + eps)

        def rot90(v): return np.array([-v[1], v[0]], dtype=float)

        def clamp_to_field(p):
            cx = np.clip(p[0], -FIELD_HALF_X + wall_margin, FIELD_HALF_X - wall_margin)
            cy = np.clip(p[1], -FIELD_HALF_Y + wall_margin, FIELD_HALF_Y - wall_margin)
            return np.array([cx, cy], dtype=float)

        rpos = vec(rx, ry)
        g_dir = nrm(vec(goal_x, goal_y) - rpos)

        # --- nearest enemy & evade
        nearest_enemy, nearest_d = None, float('inf')
        for e in game.enemy_robots.values():
            d = np.hypot(rx - e.p.x, ry - e.p.y)
            if d < nearest_d:
                nearest_d, nearest_enemy = d, e

        if nearest_enemy is None:
            enemy_away = np.zeros(2)
            enemy_perp = np.zeros(2)
            d_enemy = float('inf')
        else:
            ex, ey = nearest_enemy.p.x, nearest_enemy.p.y
            enemy_vec = rpos - vec(ex, ey)
            d_enemy = max(np.linalg.norm(enemy_vec), 1e-6)
            enemy_away = nrm(enemy_vec)

            # two perpendiculars
            perp1 = nrm(np.array([-enemy_vec[1], enemy_vec[0]]))
            perp2 = -perp1
            dot1 = float(perp1 @ g_dir)
            dot2 = float(perp2 @ g_dir)
            pref_sign = +1 if dot1 >= dot2 else -1
            # --- hysteresis: keep previous side unless clear advantage AND cooldown elapsed
            FLIP_MARGIN = 0.12   # need at least this dot advantage to flip
            if self.perp_sign == 0:
                # first time: choose the preferred one
                self.perp_sign = pref_sign
            else:
                if self.perp_cooldown > 0:
                    pref_sign = self.perp_sign  # locked for now
                else:
                    adv = abs(dot1 - dot2)
                    if pref_sign != self.perp_sign and adv < FLIP_MARGIN:
                        pref_sign = self.perp_sign  # not enough advantage -> stick
                    elif pref_sign != self.perp_sign:
                        # allow flip but start cooldown so we don't bounce back
                        self.perp_sign = pref_sign
                        self.perp_cooldown = 8  # ticks
            enemy_perp = perp1 if self.perp_sign > 0 else perp2

        # cooldown countdown
        if self.perp_cooldown > 0:
            self.perp_cooldown -= 1

        # --- wall repulsion & proximity
        def wall_distances(p):
            x, y = p
            d_left   = (x + FIELD_HALF_X)
            d_right  = (FIELD_HALF_X - x)
            d_bottom = (y + FIELD_HALF_Y)
            d_top    = (FIELD_HALF_Y - y)
            return d_left, d_right, d_bottom, d_top

        def wall_proximity(p):
            d_left, d_right, d_bottom, d_top = wall_distances(p)
            return float(np.clip((wall_influence - min(d_left, d_right, d_bottom, d_top))
                                 / max(wall_influence, 1e-6), 0.0, 1.0))

        def wall_repulsion(p):
            x, y = p
            rep = np.zeros(2)
            d_left, d_right, d_bottom, d_top = wall_distances(p)

            def push(mag, dir_vec):
                return (1.0 / (mag * mag + 1e-6)) * dir_vec

            if d_left < wall_influence:
                rep += push(max(d_left, 0.05), vec(+1, 0))
            if d_right < wall_influence:
                rep += push(max(d_right, 0.05), vec(-1, 0))
            if d_bottom < wall_influence:
                rep += push(max(d_bottom, 0.05), vec(0, +1))
            if d_top < wall_influence:
                rep += push(max(d_top, 0.05), vec(0, -1))

            near_vert = (d_left < wall_influence) or (d_right < wall_influence)
            near_horz = (d_bottom < wall_influence) or (d_top < wall_influence)
            if near_vert and near_horz:
                rep += corner_push_gain * nrm(vec(0, 0) - p)
            return rep

        w_rep = wall_repulsion(rpos)

        # --- weights
        prox = wall_proximity(rpos)
        t = np.clip((d_enemy - safe_distance) / max(safe_distance, 1e-6), 0.0, 1.0)
        w_goal  = 0.45 * t + 0.15
        w_perp  = 0.55 * (1.0 - t)
        w_away  = 0.20 * (1.0 - t)
        w_wall  = 0.15 + 0.55 * (prox ** 2)

        # --- compose desired direction
        desired_dir = (
            w_goal * g_dir +
            w_perp * enemy_perp +
            w_away * enemy_away +
            w_wall * nrm(w_rep)
        )
        desired_dir += 0.12 * prox * nrm(vec(0, 0) - rpos)

        eps = 1e-6
        if np.linalg.norm(desired_dir) < eps:
            desired_dir = g_dir

        # avoid pushing into walls if already on margin
        at_left_margin   = rpos[0] <= (-FIELD_HALF_X + wall_margin + eps)
        at_right_margin  = rpos[0] >= ( FIELD_HALF_X - wall_margin - eps)
        at_bottom_margin = rpos[1] <= (-FIELD_HALF_Y + wall_margin + eps)
        at_top_margin    = rpos[1] >= ( FIELD_HALF_Y - wall_margin - eps)
        if at_left_margin and desired_dir[0] < 0.0:
            desired_dir[0] = 0.0
        if at_right_margin and desired_dir[0] > 0.0:
            desired_dir[0] = 0.0
        if at_bottom_margin and desired_dir[1] < 0.0:
            desired_dir[1] = 0.0
        if at_top_margin and desired_dir[1] > 0.0:
            desired_dir[1] = 0.0

        move_dir = nrm(desired_dir)
        if self.prev_move_dir is not None:
            move_dir = nrm(0.85 * move_dir + 0.15 * np.array(self.prev_move_dir))
        self.prev_move_dir = (float(move_dir[0]), float(move_dir[1]))
        proposed = rpos + move_dir * step_size
        proposed = clamp_to_field(proposed)

        # nudges away from pockets
        near_top_or_bottom = (FIELD_HALF_Y - abs(proposed[1])) < (wall_margin + 0.05)
        if near_top_or_bottom:
            proposed[1] *= 0.7
        near_left_or_right = (FIELD_HALF_X - abs(proposed[0])) < (wall_margin + 0.05)
        if near_top_or_bottom and near_left_or_right:
            proposed *= 0.6

        # --------- stagnation & target hysteresis ---------
        # ---------- scoring helpers ----------
        def min_wall_distance(p):
            x, y = p
            return min(
                (x + FIELD_HALF_X),
                (FIELD_HALF_X - x),
                (y + FIELD_HALF_Y),
                (FIELD_HALF_Y - y),
            )

        def enemy_clearance(p):
            if nearest_enemy is None:
                return 5.0
            ex, ey = nearest_enemy.p.x, nearest_enemy.p.y
            return float(np.linalg.norm(p - vec(ex, ey)))

        def heading_gain(p):
            v = p - rpos
            nv = nrm(v)
            return float(max(0.0, nv @ g_dir))  # prefer heading toward goal

        def score(p):
            # weights: safety first, then enemies, then heading
            return 0.6 * min_wall_distance(p) + 0.3 * enemy_clearance(p) + 0.1 * heading_gain(p)

        # proposed target from current field
        proposed = proposed  # from your code above
        proposed_score = score(proposed)

        # ---------- stagnation detection ----------
        same_as_prev = (
            self.prev_target is not None
            and float(np.linalg.norm(proposed - np.array(self.prev_target))) < 0.05
        )
        progressed = (self.prev_target_dist is None) or \
                    (float(np.linalg.norm(np.array(self.prev_target) - rpos)) < self.prev_target_dist - 1e-3)
        self.stuck_ticks = (self.stuck_ticks + 1) if (not progressed or same_as_prev) else 0

        # ---------- parameters ----------
        LOCK_MIN_TICKS  = 10     # dwell time
        SWITCH_DEADBAND = 0.25   # ignore tiny changes (< 25cm)
        IMPROVE_MARGIN  = 0.15   # require clear score improvement to break lock early
        SMOOTH_BETA     = 0.35   # low-pass when switching
        STUCK_TICKS_THRESH = 6

        # ---------- escape mode (hold a lateral/center target for a few ticks) ----------
        if self.escape_ticks_left > 0:
            target_np = np.array(self.escape_target, dtype=float)
            self.escape_ticks_left -= 1
        else:
            # if stuck while locked on previous target, generate an escape target and HOLD it
            if self.stuck_ticks >= STUCK_TICKS_THRESH and self.prev_target is not None:
                side_len = 0.5 * step_size
                left_dir   = nrm(np.array([-move_dir[1],  move_dir[0]]))
                right_dir  = nrm(np.array([ move_dir[1], -move_dir[0]]))
                center_dir = nrm(vec(0, 0) - rpos)
                cands = [
                    clamp_to_field(rpos + left_dir   * side_len),
                    clamp_to_field(rpos + right_dir  * side_len),
                    clamp_to_field(rpos + center_dir * side_len),
                ]
                target_np = cands[int(np.argmax([score(c) for c in cands]))]
                self.escape_target = (float(target_np[0]), float(target_np[1]))
                self.escape_ticks_left = 8                   # hold for a short burst
                self.lock_ticks_left = LOCK_MIN_TICKS       # also lock it
                self.stuck_ticks = 0
            else:
                # ---------- target lock + Schmitt trigger ----------
                if self.prev_target is None:
                    target_np = proposed
                    self.lock_ticks_left = LOCK_MIN_TICKS
                else:
                    prev = np.array(self.prev_target)
                    prev_score = score(prev)
                    delta = float(np.linalg.norm(proposed - prev))

                    if self.lock_ticks_left > 0:
                        # while locked: keep prev unless new is clearly better AND different enough
                        if (delta > SWITCH_DEADBAND) and (proposed_score > prev_score + IMPROVE_MARGIN):
                            target_np = (1.0 - SMOOTH_BETA) * prev + SMOOTH_BETA * proposed
                            self.lock_ticks_left = LOCK_MIN_TICKS
                        else:
                            target_np = prev
                            self.lock_ticks_left -= 1
                    else:
                        # lock expired: allow switch if far enough OR clearly better
                        if (delta > SWITCH_DEADBAND) or (proposed_score > prev_score + IMPROVE_MARGIN):
                            target_np = (1.0 - SMOOTH_BETA) * prev + SMOOTH_BETA * proposed
                        else:
                            target_np = prev
                        self.lock_ticks_left = LOCK_MIN_TICKS

        # near-goal special: pick recenter/backtrack
        goal_dist = np.hypot(goal_x - rx, goal_y - ry)
        if goal_dist < 1.7:
            center_dir = nrm(vec(0, 0) - rpos)
            back_dir   = -g_dir
            cand_center = clamp_to_field(rpos + center_dir * step_size)
            cand_back   = clamp_to_field(rpos + back_dir   * step_size)

            def min_wall_distance(p):
                x, y = p
                return min((x + FIELD_HALF_X), (FIELD_HALF_X - x), (y + FIELD_HALF_Y), (FIELD_HALF_Y - y))
            def enemy_clearance(p):
                if nearest_enemy is None:
                    return 5.0
                ex, ey = nearest_enemy.p.x, nearest_enemy.p.y
                return float(np.linalg.norm(p - vec(ex, ey)))

            def score(p):
                return 0.7 * min_wall_distance(p) + 0.3 * enemy_clearance(p)

            target_np = cand_center if score(cand_center) >= score(cand_back) else cand_back

        # persist and return
        self.prev_target = (float(target_np[0]), float(target_np[1]))
        self.prev_target_dist = float(np.linalg.norm(target_np - rpos))
        self.prev_rpos = (float(rx), float(ry))
        return Vector2D(self.prev_target[0], self.prev_target[1])

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
