from refiners.base_refiner import BaseRefiner
from entities.data.command import RobotResponse
from entities.game.game import Game
from entities.game.robot import Robot

from dataclasses import replace

from typing import List, Dict


class RobotInfoRefiner(BaseRefiner):

    def refine(self, game: Game, robot_responses: List[RobotResponse]):
        if robot_responses is None or len(robot_responses) == 0:
            return game

        friendly_robots = game.friendly_robots.copy()
        for robot_response in robot_responses:
            id = robot_response.id
            if id in friendly_robots:
                robot = friendly_robots[id]
                friendly_robots[id] = replace(robot, has_ball=robot_response.has_ball)
                
        return replace(game, friendly_robots=friendly_robots)

