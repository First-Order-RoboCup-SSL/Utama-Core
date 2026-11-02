import math
import warnings
from dataclasses import replace
from typing import FrozenSet, List, Optional

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.data_processing.refiners.base_refiner import BaseRefiner
from utama_core.entities.data.command import RobotResponse
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot

# Distance threshold for vision-based has_ball inference: robot centre + small buffer.
_BALL_CAPTURE_DIST = ROBOT_RADIUS + 0.04  # ~0.13 m

# Real robots' ball sensors drop out with the ball still on the dribbler: in the 2026-10-27 field
# runs one went off mid-carry for 0.5 s with the ball 0.09-0.10 m ahead, and the placer braked
# from 1 m/s and lost it. The default `contact_grace_s`, in every mode: the simulators' contact
# does not flicker like that, but one code path means sim matches and tests exercise it too.
CONTACT_GRACE_S = 0.3


def _with_has_ball(robot: Robot, has_ball: bool) -> Robot:
    """`dataclasses.replace(robot, has_ball=has_ball)` without its per-call overhead
    (this runs for every robot every frame)."""
    return Robot(robot.id, robot.is_friendly, has_ball, robot.p, robot.v, robot.a, robot.orientation)


class RobotInfoRefiner(BaseRefiner):
    """Merges IR-sensor robot responses into the game frame.

    Args:
        trusted_ir_robots: Set of robot IDs (vision IDs) whose IR sensor is known to be
            working.  Robots in this set use the raw sensor reading directly.  Any robot
            ID **not** in this set has its has_ball inferred from vision proximity
            (~0.13 m threshold) instead.  Pass ``None`` (default) to trust every
            robot's IR sensor — this is the normal stable-hardware behaviour and can
            be restored by simply removing the argument.
        contact_grace_s: A trusted sensor that reads off within this many seconds of its last
            reading on still counts as on while vision shows the ball in front of the robot,
            within capture distance (`CONTACT_GRACE_S` by default; 0 takes every reading as it comes).

    Enemy robots' ``has_ball`` is applied from ``enemy_robot_responses`` when the
    caller provides them — in rsim PVP ``StrategyRunner`` feeds both teams'
    contact data into each frame (the sim physics reports contact for both
    sides), left untouched otherwise. Previously enemy ``has_ball`` was never
    filled here (always False in the referee's view), which made touch-dependent
    referee rules — e.g. ``DoubleTouchRule``, whose restart window only closes
    when *any* robot touches the ball — blind to the opponent's touches and able
    to flag phantom "second touches" long after a legal intervening touch.
    (Real-mode PVP does not yet feed opponent responses here.)
    """

    def __init__(self, trusted_ir_robots: Optional[FrozenSet[int]] = None, contact_grace_s: float = CONTACT_GRACE_S):
        self._trusted_ir_robots = trusted_ir_robots
        self._contact_grace_s = contact_grace_s
        self._last_contact_ts: dict[int, float] = {}

    def refine(
        self,
        game_frame: GameFrame,
        robot_responses: List[RobotResponse],
        enemy_robot_responses: List[RobotResponse] = None,
    ):
        friendly_robots = game_frame.friendly_robots.copy()
        enemy_robots = game_frame.enemy_robots.copy()

        # When an allowlist is active, first infer has_ball for every untrusted
        # robot from vision proximity.  This covers frames where the robot drops
        # its serial response entirely — without this pass, has_ball would stay
        # frozen at the previous frame's value instead of being inferred.
        if self._trusted_ir_robots is not None:
            for robot_id, robot in friendly_robots.items():
                if robot_id not in self._trusted_ir_robots:
                    friendly_robots[robot_id] = _with_has_ball(robot, self._infer_has_ball(game_frame, robot))

        # Then overlay IR sensor readings for robots that sent a response.
        if robot_responses:
            for robot_response in robot_responses:
                rid = robot_response.id
                if rid not in friendly_robots:
                    warnings.warn(f"Robot ID {rid} in robot responses not found in friendly robots. ")
                    continue

                robot = friendly_robots[rid]
                if self._trusted_ir_robots is None or rid in self._trusted_ir_robots:
                    # Trusted (or trust-all mode): the IR reading, over `contact_grace_s`'s dropouts
                    friendly_robots[rid] = _with_has_ball(robot, self._contact(game_frame, robot, robot_response))
                # Untrusted robots were already handled by the vision-proximity pass above

        # Enemy has_ball: provided team-tagged by the caller (StrategyRunner
        # pulls both teams' responses once per tick in PVP). No IR allowlist
        # concept applies — the raw response (sim contact physics, or the
        # shared-transmitter payload in real PVP) is the only source.
        if enemy_robot_responses:
            for robot_response in enemy_robot_responses:
                rid = robot_response.id
                if rid in enemy_robots:
                    enemy_robots[rid] = _with_has_ball(enemy_robots[rid], robot_response.has_ball)

        if friendly_robots == game_frame.friendly_robots and enemy_robots == game_frame.enemy_robots:
            return game_frame
        return replace(game_frame, friendly_robots=friendly_robots, enemy_robots=enemy_robots)

    def _contact(self, game_frame: GameFrame, robot: Robot, response: RobotResponse) -> bool:
        if response.has_ball:
            self._last_contact_ts[robot.id] = game_frame.ts
            return True
        last = self._last_contact_ts.get(robot.id)
        if last is None or game_frame.ts - last > self._contact_grace_s or game_frame.ball is None:
            return False
        dx, dy = game_frame.ball.p.x - robot.p.x, game_frame.ball.p.y - robot.p.y
        ahead = dx * math.cos(robot.orientation) + dy * math.sin(robot.orientation)
        return ahead > 0.0 and math.hypot(dx, dy) < _BALL_CAPTURE_DIST

    @staticmethod
    def _infer_has_ball(game_frame: GameFrame, robot) -> bool:
        """Proximity-based has_ball: True when robot is within capture distance of ball."""
        if game_frame.ball is None:
            return False
        ball_2d = game_frame.ball.p.to_2d()
        return robot.p.distance_to(ball_2d) < _BALL_CAPTURE_DIST
