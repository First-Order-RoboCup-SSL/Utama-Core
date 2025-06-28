from run.refiners.base_refiner import BaseRefiner
from entities.data.command import RobotResponse
from entities.game.game_frame import GameFrame
from entities.data.object import ObjectKey, TeamType, ObjectType
import warnings

from dataclasses import replace

from typing import List, Dict


# TODO: current doesn't handle has_ball for enemy robots. In future, implement using vision data


class RobotInfoRefiner(BaseRefiner):

    def refine(self, game_frame: GameFrame, robot_responses: List[RobotResponse]):
        robot_with_ball = None
        if robot_responses is None or len(robot_responses) == 0:
            return game_frame

        friendly_robots = game_frame.friendly_robots.copy()
        for robot_response in robot_responses:
            id = robot_response.id
            if id in friendly_robots:
                robot = friendly_robots[id]
                friendly_robots[id] = replace(robot, has_ball=robot_response.has_ball)
            else:
                warnings.warn(
                    f"Robot ID {id} in robot responses not found in friendly robots. "
                )
        new_game = replace(game_frame, friendly_robots=friendly_robots)
        return new_game
