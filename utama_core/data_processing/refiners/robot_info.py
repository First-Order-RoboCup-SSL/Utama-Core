import warnings
from dataclasses import replace
from typing import FrozenSet, List, Optional

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.data_processing.refiners.base_refiner import BaseRefiner
from utama_core.entities.data.command import RobotResponse
from utama_core.entities.game.game_frame import GameFrame

# TODO: current doesn't handle has_ball for enemy robots. In future, implement using vision data

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
    """

    def __init__(self, trusted_ir_robots: Optional[FrozenSet[int]] = None):
        self._trusted_ir_robots = trusted_ir_robots

    def refine(self, game_frame: GameFrame, robot_responses: List[RobotResponse]):
        if self._trusted_ir_robots is None and (robot_responses is None or len(robot_responses) == 0):
            return game_frame

        friendly_robots = game_frame.friendly_robots.copy()

        # When a trust list is active, first apply vision-proximity inference for every
        # untrusted robot.  This ensures that robots which did not return a response this
        # tick (non-blocking real-mode polling can skip robots) do not keep stale values.
        if self._trusted_ir_robots is not None:
            for robot_id, robot in friendly_robots.items():
                if robot_id not in self._trusted_ir_robots:
                    friendly_robots[robot_id] = replace(robot, has_ball=self._infer_has_ball(game_frame, robot))

        # Overlay IR readings for trusted robots that responded this tick.
        if robot_responses:
            for robot_response in robot_responses:
                rid = robot_response.id
                if rid not in friendly_robots:
                    warnings.warn(f"Robot ID {rid} in robot responses not found in friendly robots. ")
                    continue

                if self._trusted_ir_robots is None or rid in self._trusted_ir_robots:
                    friendly_robots[rid] = replace(friendly_robots[rid], has_ball=robot_response.has_ball)

        new_game_frame = replace(game_frame, friendly_robots=friendly_robots)
        return new_game_frame

    @staticmethod
    def _infer_has_ball(game_frame: GameFrame, robot) -> bool:
        """Proximity-based has_ball: True when robot is within capture distance of ball."""
        if game_frame.ball is None:
            return False
        ball_2d = game_frame.ball.p.to_2d()
        return robot.p.distance_to(ball_2d) < _BALL_CAPTURE_DIST
