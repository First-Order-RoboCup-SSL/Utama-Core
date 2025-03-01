from collections.abc import Callable
from typing import Union
from entities.game.ball import Ball
from entities.game.game import Game
from entities.game.robot import Robot
from refiners.base_refiner import BaseRefiner
from vector import VectorObject2D, Vector
import logging

logger = logging.getLogger(__name__)

class VelocityRefiner(BaseRefiner):
    
    def refine(self, past_game: PastGame, game: Game, data):
        game.ball.v = self._get_object_v(lambda game: game.ball)
        game.ball.a = self._get_object_a(lambda frame: past_game.n_frames_ago(frame).ball)

        return game
    
    def _get_object_v(past_game: PastGame, game: Game, get_object_out_of_game: Callable[int, Union[Robot, Ball]]):
        previous_game = past_game.n_frames_ago(1)
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
        missing_velocities = 0

        if len(self._records) < WINDOW * N_WINDOWS + 1:
            return None

        for i in range(N_WINDOWS):
            averageVelocity = VectorObject2D(x=0, y=0)
            windowStart = 1 + (i * WINDOW)
            windowEnd = windowStart + WINDOW  # Excluded
            windowMiddle = (windowStart + windowEnd) // 2

            for j in range(windowStart, windowEnd):
                try:
                    object_at_frame_j = get_object_out_of_game(past_game.n_frames_ago(j))
                except:
                    logger.warning("No velocity data to calculate acceleration, assuming 0")
                    return VectorObject2D(x=0, y=0)
                
                curr_vel = object_at_frame_j.v 
                averageVelocity += curr_vel

            total = VectorObject2D(x=0, y=0)

            averageVelocity /= WINDOW

            if i != 0:
                dt = (
                    past_game.n_frames_ago(windowMiddle - WINDOW).ts
                    - past_game.n_frames_ago(windowMiddle).ts
                )
                accel = (futureAverageVelocity - averageVelocity) / dt
                total += accel
                iter += 1

            futureAverageVelocity = averageVelocity

        return total / iter



    def get_robots_velocity(self, is_yellow: bool) -> List[tuple]:
        if len(self._records) <= 1:
            return None
        if is_yellow:
            return [
                self.get_object_velocity(RobotEntity(Colour.YELLOW, i))
                for i in range(
                    len(self.get_robots_pos(True))
                )  # TODO: This is a bit of a hack, we should be able to get the number of robots from the field
            ]
        else:
            return [
                self.get_object_velocity(RobotEntity(Colour.BLUE, i))
                for i in range(len(self.get_robots_pos(False)))
            ]

    def get_ball_velocity(self) -> Optional[tuple]:
        return self.get_object_velocity(Ball)

    def predict_frame_after(self, t: float) -> VisionData:
        yellow_pos = [
            self.predict_object_pos_after(t, RobotEntity(Colour.YELLOW, i))
            for i in range(len(self.get_robots_pos(True)))
        ]
        blue_pos = [
            self.predict_object_pos_after(t, RobotEntity(Colour.BLUE, i))
            for i in range(len(self.get_robots_pos(False)))
        ]
        ball_pos = self.predict_object_pos_after(t, Ball)
        if ball_pos is None or None in yellow_pos or None in blue_pos:
            return None
        else:
            return VisionData(
                self._records[-1].ts + t,
                list(map(lambda pos: VisionRobotData(pos[0], pos[1], 0), yellow_pos)),
                list(map(lambda pos: VisionRobotData(pos[0], pos[1], 0), blue_pos)),
                [VisionBallData(ball_pos[0], ball_pos[1], 0, 1)],  # TODO : Support z axis
            )

    ### General Object Position Prediction ###
    def predict_object_pos_after(self, t: float, object: GameObject) -> Optional[tuple]:
        # If t is after the object has stopped we return the position at which object stopped.
        sx = 0
        sy = 0

        acceleration = self.get_object_acceleration(object)

        if acceleration is None:
            return None

        ax, ay = acceleration
        vels = self.get_object_velocity(object)

        if vels is None:
            ux, uy = None, None
        else:
            ux, uy = vels

        if object is Ball:
            ball = self.get_ball_pos()
            start_x, start_y = ball[0].x, ball[0].y
        else:
            posn = self._get_object_position_at_frame(len(self._records) - 1, object)
            start_x, start_y = posn.x, posn.y

        if ax and ux:
            sx = self._calculate_displacement(ux, ax, t)

        if ay and uy:
            sy = self._calculate_displacement(uy, ay, t)

        return (
            start_x + sx,
            start_y + sy,
        )  # TODO: Doesn't take into account spin / angular vel

    def _calculate_displacement(self, u, a, t) -> float:
        if a == 0:  # Handle zero acceleration case
            return u * t
        else:
            stop_time = -u / a
            if stop_time < 0:
                stop_time = float("inf")
            effective_time = min(t, stop_time)
            displacement = (u * effective_time) + (0.5 * a * effective_time**2)
            logger.debug(
                f"Displacement: {displacement} for time: {effective_time}, stop time: {stop_time}"
            )
            return displacement

    def predict_ball_pos_at_x(self, x: float) -> Optional[tuple]:
        vel = self.get_ball_velocity()

        if not vel or not vel[0] or not vel[0]:
            return None

        ux, uy = vel
        pos = self.get_ball_pos()[0]
        bx = pos.x
        by = pos.y

        if uy == 0:
            return (bx, by)

        t = (x - bx) / ux
        y = by + uy * t
        return (x, y)

    def get_object_velocity(self, object: GameObject) -> Optional[tuple]:
        return self._get_object_velocity_at_frame(len(self._records) - 1, object)

    def _get_object_position_at_frame(self, frame: int, object: GameObject):
        if object == Ball:
            return self._records[frame].ball[0]
        elif isinstance(object, RobotEntity):
            return (
                self._records[frame].yellow_robots[object.id]
                if object.colour == Colour.YELLOW
                else self._records[frame].blue_robots[object.id]
            )

