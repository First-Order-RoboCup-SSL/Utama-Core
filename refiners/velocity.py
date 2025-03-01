from dataclasses import replace
from entities.game.game import Game
from entities.game.past_game import PastGame
from refiners.base_refiner import BaseRefiner
from vector import VectorObject2D, VectorObject3D
from lenses import UnboundLens, lens
import logging

logger = logging.getLogger(__name__)

def zero_vector(twod:bool):
    return VectorObject2D(x=0, y=0) if twod else VectorObject3D(x=0, y=0, z=0)

class VelocityRefiner(BaseRefiner):
    ACCELERATION_WINDOW_SIZE = 5
    ACCELERATION_N_WINDOWS = 3

    def refine(self, past_game: PastGame, game: Game):
        """Adds velocity data to the robot and ball objects inside the game, using position data and IMU data
           Because game is immutable use lenses to update it. This means it creates a new version of game
           with the updated values (but sharing all the data that can be shared so no unnecessary copying)
           https://python-lenses.readthedocs.io/ """

        if game.ball is not None:
            game &= lens.ball.v.set(self._get_object_v(past_game, game, lens.ball, twod=False))
            game &= lens.ball.a.set(self._get_object_a(past_game, game, lens.ball, twod=False))

        game &= self.update_velocity_lens(past_game, game, lens.friendly_robots)
        game &= self.update_velocity_lens(past_game, game, lens.enemy_robots)

        game &= self.update_acceleration_lens(past_game, game, lens.friendly_robots)
        game &= self.update_acceleration_lens(past_game, game, lens.enemy_robots)

        return game
    
    def update_velocity_lens(self, past_game: PastGame, game: Game, robots_lens: UnboundLens):
        return robots_lens.Values().modify(
            lambda robot : robot & lens.v.set(self._get_object_v(past_game, game, robots_lens[robot.id], twod=True))
        )

    def update_acceleration_lens(self, past_game: PastGame, game: Game, robots_lens: UnboundLens):
        return robots_lens.Values().modify(
            lambda robot : robot & lens.a.set(self._get_object_a(past_game, game, robots_lens[robot.id], twod=True))
        )

    def _get_object_v(self, past_game: PastGame, game: Game, object_lens: UnboundLens, twod: bool):
        current_game = game

        try:
            previous_game = past_game.n_steps_ago(1)
            previous_pos = previous_game & object_lens.p.get()
        except Exception as e:
            logger.warning(f"Position data not available for velocity calculation; assuming 0 - {e}")
            return zero_vector(twod) # Assume not moving if we can't get the object
        
        current_pos = current_game & object_lens.p.get()

        previous_time_received = previous_game.ts
        time_received = current_game.ts

        dt_secs = time_received - previous_time_received

        return (current_pos - previous_pos) / dt_secs

    def _get_object_a(self, past_game: PastGame, game: Game, object_lens: UnboundLens, twod=False):
        try:
            pairs = self._extract_time_velocity_pairs(past_game, game, object_lens) # Assume no acceleration if we can't get all the velocities we need
        except Exception as e:
            logger.warning(f"Velocity data not available for acceleration calculation; assuming 0 - {e}")
            return zero_vector(twod)
        return self._calculate_acceleration_from_pairs(pairs)

    def _extract_time_velocity_pairs(self, past_game: PastGame, game: Game, object_lens: UnboundLens):
        time_velocity_pairs = []
        for i in range(VelocityRefiner.ACCELERATION_N_WINDOWS * VelocityRefiner.ACCELERATION_WINDOW_SIZE):
            old_game = past_game.n_steps_ago(i + 1)
            time_velocity_pairs.append((old_game.ts, old_game & object_lens.v.get()))
        return time_velocity_pairs

    def _calculate_acceleration_from_pairs(self, time_velocity_pairs: tuple[float, float], twod: bool):
        iter = 0

        for i in range(VelocityRefiner.ACCELERATION_N_WINDOWS):
            averageVelocity = zero_vector(twod)
            windowStart = 1 + (i * VelocityRefiner.ACCELERATION_WINDOW_SIZE)
            windowEnd = windowStart + VelocityRefiner.ACCELERATION_WINDOW_SIZE  # Excluded
            windowMiddle = (windowStart + windowEnd) // 2

            for j in range(windowStart, windowEnd):
                object_at_frame_j = time_velocity_pairs[j]                
                curr_vel = object_at_frame_j[1]
                averageVelocity += curr_vel

            total = zero_vector(twod)

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
