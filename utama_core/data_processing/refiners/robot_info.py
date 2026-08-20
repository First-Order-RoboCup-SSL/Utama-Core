import warnings
from dataclasses import replace
from typing import FrozenSet, List, Optional

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.data_processing.refiners.base_refiner import BaseRefiner
from utama_core.entities.data.command import RobotResponse
from utama_core.entities.game.game_frame import GameFrame

# Distance threshold for vision-based has_ball inference: robot centre + small buffer.
_BALL_CAPTURE_DIST = ROBOT_RADIUS + 0.04  # ~0.13 m


class RobotInfoRefiner(BaseRefiner):
    """Merges IR-sensor robot responses into the game frame.

    Args:
        trusted_ir_robots: Set of robot IDs (vision IDs) whose IR sensor is known to be
            working.  Robots in this set use the raw sensor reading directly.  Any robot
            ID **not** in this set has its has_ball inferred from vision proximity
            (~0.13 m threshold) instead.  Pass ``None`` (default) to trust every
            robot's IR sensor — this is the normal stable-hardware behaviour and can
            be restored by simply removing the argument.

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

    def __init__(self, trusted_ir_robots: Optional[FrozenSet[int]] = None):
        self._trusted_ir_robots = trusted_ir_robots

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
                    friendly_robots[robot_id] = replace(robot, has_ball=self._infer_has_ball(game_frame, robot))

        # Then overlay IR sensor readings for robots that sent a response.
        if robot_responses:
            for robot_response in robot_responses:
                rid = robot_response.id
                if rid not in friendly_robots:
                    warnings.warn(f"Robot ID {rid} in robot responses not found in friendly robots. ")
                    continue

                robot = friendly_robots[rid]
                if self._trusted_ir_robots is None or rid in self._trusted_ir_robots:
                    # Trusted (or trust-all mode): use raw IR reading
                    friendly_robots[rid] = replace(robot, has_ball=robot_response.has_ball)
                # Untrusted robots were already handled by the vision-proximity pass above

        # Enemy has_ball: provided team-tagged by the caller (StrategyRunner
        # pulls both teams' responses once per tick in PVP). No IR allowlist
        # concept applies — the raw response (sim contact physics, or the
        # shared-transmitter payload in real PVP) is the only source.
        if enemy_robot_responses:
            for robot_response in enemy_robot_responses:
                rid = robot_response.id
                if rid in enemy_robots:
                    enemy_robots[rid] = replace(enemy_robots[rid], has_ball=robot_response.has_ball)

        if friendly_robots == game_frame.friendly_robots and enemy_robots == game_frame.enemy_robots:
            return game_frame
        return replace(game_frame, friendly_robots=friendly_robots, enemy_robots=enemy_robots)

    @staticmethod
    def _infer_has_ball(game_frame: GameFrame, robot) -> bool:
        """Proximity-based has_ball: True when robot is within capture distance of ball."""
        if game_frame.ball is None:
            return False
        ball_2d = game_frame.ball.p.to_2d()
        return robot.p.distance_to(ball_2d) < _BALL_CAPTURE_DIST
