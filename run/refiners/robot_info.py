from run.refiners.base_refiner import BaseRefiner
from entities.data.command import RobotResponse
from entities.game.game import Game
from entities.data.object import ObjectKey, TeamType, ObjectType
import warnings

from dataclasses import replace

from typing import List, Dict


# TODO: current doesn't handle has_ball for enemy robots. In future, implement using vision data


class RobotInfoRefiner(BaseRefiner):

    def refine(self, game: Game, robot_responses: List[RobotResponse]):
        robot_with_ball = None
        if robot_responses is None or len(robot_responses) == 0:
            return game

        friendly_robots = game.friendly_robots.copy()
        for robot_response in robot_responses:
            id = robot_response.id
            if id in friendly_robots:
                robot = friendly_robots[id]
                friendly_robots[id] = replace(robot, has_ball=robot_response.has_ball)
                # TODO: assumes there is only one robot with the ball
                if robot_response.has_ball:
                    robot_with_ball = ObjectKey(TeamType.FRIENDLY, ObjectType.ROBOT, id)
            else:
                warnings.warn(
                    f"Robot ID {id} in robot responses not found in friendly robots. "
                )
        new_game = replace(
            game, friendly_robots=friendly_robots, robot_with_ball=robot_with_ball
        )
        return new_game
