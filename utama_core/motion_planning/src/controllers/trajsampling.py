"""`MotionController` wrapper around `TrajectorySamplingPlanner` -- see that
module's docstring for the algorithm. Outputs velocity directly from the
accepted bang-bang trajectory evaluated one control period ahead, rather
than handing a position "carrot" to a PID translator the way
`FastPathPlanningController` does: the trajectory already encodes a
velocity-limit-respecting profile, so re-deriving velocity from a position
error would throw that away and reintroduce the same braking-distance
class of bug `TwoDPID.set_final_target` works around for FPP's carrot.

Also implements a residual emergency-brake layer, but this is NOT the
primary fix for two robots colliding -- that turned out to be a teammate
PRIORITY ordering in the planner itself (see `TrajectorySamplingPlanner`'s
module docstring and `_has_priority`), ported after checking TIGERs'
actual published source (Sumatra) rather than inferring from the paper:
`MovingObstacleResultAcceptor.accept` rejects a candidate outright when it
collides with a higher-priority teammate, no braking-distance carve-out at
all. Four successive versions of this brake layer alone (own-speed check,
own-radius accounting, closing-speed instead of own-speed) never stopped
the mirror_swap 6v6 test's two robots from grazing each other at
~0.176-0.179m (collision threshold 0.18m) -- because a per-tick "is MY
trajectory currently safe" check has no way to make one of two
symmetrically-reasonable robots yield to the other; only a decided-in-
advance ordering does that (see the planner's docstring for why).

What's left here is the genuine fallback case: when NO candidate is
priority-clear (the lower-priority robot truly has nowhere to go right
now), the planner's `best_fallback` may still return a trajectory that
survives only briefly before a non-priority obstacle. This brake, comparing
`PlanResult.closing_speed` (how fast the gap to the nearest obstacle is
shrinking, accounting for the obstacle's own motion, not just this robot's
speed) against a safe threshold, keeps that residual case from running
into something at speed while the robot waits for room to open up.
"""

import math

from utama_core.config.enums import Mode
from utama_core.config.robot_params import GRSIM_PARAMS, REAL_PARAMS, RSIM_PARAMS
from utama_core.config.settings import CONTROL_FREQUENCY
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.motion_planning.src.pid.pid import get_pids
from utama_core.motion_planning.src.trajsampling.config import (
    trajsamplingconfig as config,
)
from utama_core.motion_planning.src.trajsampling.planner import (
    TrajectorySamplingPlanner,
)
from utama_core.rsoccer_simulator.src.ssl.envs import SSLStandardEnv

_PARAMS_BY_MODE = {Mode.RSIM: RSIM_PARAMS, Mode.GRSIM: GRSIM_PARAMS, Mode.REAL: REAL_PARAMS}


class TrajectorySamplingController(MotionController):
    def __init__(self, mode: Mode, rsim_env: SSLStandardEnv | None = None):
        super().__init__(mode, rsim_env)
        params = _PARAMS_BY_MODE[mode]
        self.planner = TrajectorySamplingPlanner(v_max=params.MAX_VEL, a_max=params.MAX_ACCELERATION)
        self._brake_acceleration = params.MAX_ACCELERATION * config.BRAKE_ACCELERATION_MULTIPLIER
        # Orientation is a separate concern from translation in this
        # codebase (see FastPathPlanningController) -- reused as-is rather
        # than reinvented, since bang-bang trajectory sampling only replaces
        # 2D translation planning.
        self.pid_oren, _ = get_pids(mode)
        self._dt = 1.0 / CONTROL_FREQUENCY

    def calculate(
        self,
        game: Game,
        robot_id: int,
        target_pos: Vector2D,
        target_oren: float,
    ) -> tuple[Vector2D, float]:
        field_bounds = game.field.field_bounds
        robot = game.friendly_robots[robot_id]

        oren = self.pid_oren.calculate(target_oren, robot.orientation, robot_id)

        result = self.planner.plan(game, robot_id, (target_pos.x, target_pos.y), field_bounds)
        # `result.elapsed` is 0.0 for a freshly-computed trajectory and >0
        # when `plan()` is still executing an already-committed one (see
        # `PlanResult.elapsed`'s docstring) -- always offset by it, never
        # read the trajectory from its own t=0 directly, or a reused
        # trajectory looks identical every tick and the robot never
        # actually progresses along it.
        lookahead = min(result.elapsed + self._dt, result.trajectory.duration)
        _, (vx, vy) = result.trajectory.state_at(lookahead)

        current_speed = math.hypot(robot.v.x, robot.v.y)
        if result.nearest_obstacle_distance is not None and current_speed > 1e-6:
            # v = sqrt(2*a*d) is the fastest CLOSING speed that can still be
            # braked to zero within remaining gap `d` under deceleration `a`
            # -- same formula `TwoDPID._calculate`'s braking-distance cap
            # uses, but compared against `closing_speed` (how fast the gap
            # to the nearest obstacle is actually shrinking, accounting for
            # the obstacle's own motion too -- see `PlanResult.closing_speed`
            # docstring), not `current_speed` alone. Checking own speed only
            # was found live to miss real danger: two robots each
            # individually "within their own braking distance" of a
            # stationary threat can still be closing on EACH OTHER fast
            # enough to collide, because neither one's own speed reflects
            # the other's contribution to how fast the gap is shrinking.
            closing_speed = max(result.closing_speed or 0.0, 0.0)
            max_safe_closing_speed = math.sqrt(
                2 * self._brake_acceleration * max(result.nearest_obstacle_distance, 0.0)
            )
            if closing_speed > 1e-6 and closing_speed > max_safe_closing_speed:
                # Scale the robot's own velocity down by the same ratio the
                # closing speed needs to shrink by. Not an exact decomposition
                # of "how much of the closing speed is this robot's own
                # contribution" (that would need the obstacle's velocity
                # vector, not just a scalar rate) -- but it's a safe,
                # monotonic response: it always reduces this robot's speed
                # when the gap is closing too fast, and never increases it.
                brake_scale = max_safe_closing_speed / closing_speed
                return Vector2D(robot.v.x * brake_scale, robot.v.y * brake_scale), oren

        return Vector2D(vx, vy), oren

    def reset(self, robot_id):
        self.pid_oren.reset(robot_id)
        self.planner._committed.pop(robot_id, None)
        self.planner._last_intermediate_target.pop(robot_id, None)
