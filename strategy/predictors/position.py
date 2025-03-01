

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
