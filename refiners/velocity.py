from collections.abc import Callable
from typing import Union
from entities.game.ball import Ball
from entities.game.game import Game
from entities.game.past_game import PastGame
from entities.game.robot import Robot
from refiners.base_refiner import BaseRefiner
from vector import VectorObject2D
import logging

logger = logging.getLogger(__name__)

class VelocityRefiner(BaseRefiner):
    
    def refine(self, past_game: PastGame, game: Game, data):
        game.ball.v = self._get_object_v(lambda game: game.ball)
        game.ball.a = self._get_object_a(lambda frame: past_game.n_steps_ago(frame).ball)

        return game
    
    def _get_object_v(past_game: PastGame, game: Game, get_object_out_of_game: Callable[int, Union[Robot, Ball]]):
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

    def _get_object_a(past_game: PastGame, game: Game, get_object_out_of_game: Callable[int, Union[Robot, Ball]]):
        WINDOW = 5
        N_WINDOWS = 3

        iter = 0

        for i in range(N_WINDOWS):
            averageVelocity = VectorObject2D(x=0, y=0)
            windowStart = 1 + (i * WINDOW)
            windowEnd = windowStart + WINDOW  # Excluded
            windowMiddle = (windowStart + windowEnd) // 2

            for j in range(windowStart, windowEnd):
                try:
                    object_at_frame_j = get_object_out_of_game(past_game.n_steps_ago(j))
                except:
                    logger.warning("No velocity data to calculate acceleration, assuming 0")
                    return VectorObject2D(x=0, y=0)
                
                curr_vel = object_at_frame_j.v 
                averageVelocity += curr_vel

            total = VectorObject2D(x=0, y=0)

            averageVelocity /= WINDOW

            if i != 0:
                dt = (
                    past_game.n_steps_ago(windowMiddle - WINDOW).ts
                    - past_game.n_steps_ago(windowMiddle).ts
                )
                accel = (futureAverageVelocity - averageVelocity) / dt
                total += accel
                iter += 1

            futureAverageVelocity = averageVelocity

        return total / iter
