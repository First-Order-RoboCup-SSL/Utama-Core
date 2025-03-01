

from refiners.base_refiner import BaseRefiner


class VelocityRefiner(BaseRefiner):
    
    def refine(self, game, data):
        return game



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

    def _get_object_velocity_at_frame(
        self, frame: int, object: GameObject
    ) -> Optional[tuple]:
        if frame >= len(self._records) or frame == 0:
            logger.warning("Cannot provide velocity at a frame that does not exist")
            return None

        previous_frame = self._records[frame - 1]
        current_frame = self._records[frame]

        previous_pos = self._get_object_position_at_frame(frame - 1, object)
        current_pos = self._get_object_position_at_frame(frame, object)

        if current_pos is None or previous_pos is None:
            logger.warning("No position data to calculate velocity for frame %d", frame)
            return None

        previous_time_received = previous_frame.ts
        time_received = current_frame.ts

        if time_received < previous_time_received:
            logger.warning(
                "Timestamps out of order for vision data %f should be after %f",
                time_received,
                previous_time_received,
            )
            return None

        dt_secs = time_received - previous_time_received

        vx = (current_pos.x - previous_pos.x) / dt_secs
        vy = (current_pos.y - previous_pos.y) / dt_secs

        return (vx, vy)

    def get_object_acceleration(self, object: GameObject) -> Optional[tuple]:
        totalX = 0
        totalY = 0
        WINDOW = 5
        N_WINDOWS = 3
        iter = 0
        missing_velocities = 0

        if len(self._records) < WINDOW * N_WINDOWS + 1:
            return None

        for i in range(N_WINDOWS):
            missing_velocities = 0
            averageVelocity = [0, 0]
            windowStart = 1 + (i * WINDOW)
            windowEnd = windowStart + WINDOW  # Excluded
            windowMiddle = (windowStart + windowEnd) // 2

            for j in range(windowStart, windowEnd):
                curr_vel = self._get_object_velocity_at_frame(
                    len(self._records) - j, object
                )
                if curr_vel:
                    averageVelocity[0] += curr_vel[0]
                    averageVelocity[1] += curr_vel[1]
                elif missing_velocities == WINDOW - 1:
                    logging.warning(
                        f"No velocity data to calculate acceleration for frame {len(self._records) - j}"
                    )
                    return None
                else:
                    missing_velocities += 1

            averageVelocity[0] /= WINDOW - missing_velocities
            averageVelocity[1] /= WINDOW - missing_velocities

            if i != 0:
                dt = (
                    self._records[-windowMiddle + WINDOW].ts
                    - self._records[-windowMiddle].ts
                )
                accelX = (
                    futureAverageVelocity[0] - averageVelocity[0]
                ) / dt  # TODO vec
                accelY = (futureAverageVelocity[1] - averageVelocity[1]) / dt
                totalX += accelX
                totalY += accelY
                iter += 1

            futureAverageVelocity = tuple(averageVelocity)

        return (totalX / iter, totalY / iter)