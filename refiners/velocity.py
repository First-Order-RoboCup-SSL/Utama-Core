from dataclasses import replace
from typing import List, Union, Callable
from entities.game.ball import Ball
from entities.game.game import Game
from entities.game.past_game import PastGame
from entities.game.robot import Robot
from refiners.base_refiner import BaseRefiner
from vector import VectorObject2D
import logging

logger = logging.getLogger(__name__)

class VelocityRefiner(BaseRefiner):
    ACCELERATION_WINDOW_SIZE = 5
    ACCELERATION_N_WINDOWS = 3

    def refine(self, past_game: PastGame, game: Game, data):
        game.ball.v = self._get_object_v(lambda game: game.ball)
        game.ball.a = self._get_object_a(lambda game: game.ball)

        game.friendly_robots = self.update_robots(game.friendly_robots)
        game.enemy_robots = self.update_robots(game.enemy_robots)

        return game
    
    def update_robots(self, robots: List[Robot]):
        select_robot = lambda id: (lambda game: game.friendly_robots[id])
        return [
            replace(robot, v=self._get_object_v(select_robot(robot.id)), a=self._get_object_a(select_robot(robot.id)))
            for robot in robots
        ]

    def _get_object_v(past_game: PastGame, game: Game, get_object_out_of_game: Callable[[Game], Union[Robot, Ball]]):
        previous_game = past_game.n_steps_ago(1)
        current_game = game

        try:  
            previous_pos = get_object_out_of_game(previous_game).p
        except:
            return VectorObject2D(0, 0) # Assume not moving if we can't get the object
        
        current_pos = get_object_out_of_game(current_game).p

        previous_time_received = previous_game.ts
        time_received = current_game.ts

        dt_secs = time_received - previous_time_received

        return (current_pos - previous_pos) / dt_secs

    def _get_object_a(self, past_game: PastGame, game: Game, get_object_out_of_game: Callable[[Game], Union[Robot, Ball]]):
        try:
            pairs = self._extract_time_velocity_pairs(past_game, game) # Assume no acceleration if we can't get all the velocities we need
        except:
            return VectorObject2D(0, 0)
        return self._calculate_acceleration_from_pairs(pairs)

    def _extract_time_velocity_pairs(past_game: PastGame, game: Game, get_object_out_of_game: Callable[[Game], Union[Robot, Ball]]):
        time_velocity_pairs = []
        for i in range(VelocityRefiner.ACCELERATION_N_WINDOWS * VelocityRefiner.ACCELERATION_WINDOW_SIZE):
            old_game = get_object_out_of_game(past_game.n_steps_ago(i + 1))
            time_velocity_pairs.append((old_game.ts, get_object_out_of_game(old_game).v))
        return time_velocity_pairs

    def _calculate_acceleration_from_pairs(time_velocity_pairs: tuple[float, float]):
        iter = 0

        for i in range(VelocityRefiner.ACCELERATION_N_WINDOWS):
            averageVelocity = VectorObject2D(x=0, y=0)
            windowStart = 1 + (i * VelocityRefiner.ACCELERATION_WINDOW_SIZE)
            windowEnd = windowStart + VelocityRefiner.ACCELERATION_WINDOW_SIZE  # Excluded
            windowMiddle = (windowStart + windowEnd) // 2

            for j in range(windowStart, windowEnd):
                object_at_frame_j = time_velocity_pairs[j]                
                curr_vel = object_at_frame_j[1]
                averageVelocity += curr_vel

            total = VectorObject2D(x=0, y=0)

            averageVelocity /= VelocityRefiner.ACCELERATION_WINDOW_SIZE

            if i != 0:
                dt = (
                    time_velocity_pairs[windowMiddle - VelocityRefiner.ACCELERATION_WINDOW_SIZE][0] # ts
                    - time_velocity_pairs[windowMiddle][0] # ts
                )
                accel = (futureAverageVelocity - averageVelocity) / dt
                total += accel
                iter += 1

            futureAverageVelocity = averageVelocity

        return total / iter
