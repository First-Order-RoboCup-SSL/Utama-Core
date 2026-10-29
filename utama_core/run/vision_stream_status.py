"""Browser vision stream for `StrategyRunner`, mixed in as `VisionStreamStatusMixin`: starting it,
and the status it shows (scoreboard, commentary, footballer names and roster) built from the referee
and the current game frame."""

import time

from utama_core.run.vision_stream import GameFrameRenderer, RSimVisionStreamServer

_VS_KICK_THRESHOLD = 0.5  # m/s — ball speed above this triggers kick commentary


class VisionStreamStatusMixin:
    """`StrategyRunner` methods for the browser vision stream; they read runner state set in `__init__`."""

    def _start_vision_stream(self, http_port: int) -> None:
        """Start the browser stream that mirrors refined game frames."""
        try:
            self._vision_stream_renderer = GameFrameRenderer(self.full_field_dims, scale=300.0)
            self.vision_stream = RSimVisionStreamServer(
                http_port=http_port,
            )
            self.vision_stream.start()
            self._publish_vision_stream_frame()
            self.logger.info("Vision stream available at %s", self.vision_stream.url)
            print(f"Vision stream available at {self.vision_stream.url}")
        except Exception:
            self.vision_stream = None
            self._vision_stream_renderer = None
            self.logger.warning("Vision stream could not be started; continuing without browser video.", exc_info=True)

    def _publish_vision_stream_frame(self) -> None:
        """Publish the latest refined game frame to the browser stream."""
        if self.vision_stream is None or self._vision_stream_renderer is None or self.my.current_game_frame is None:
            return
        if not self.vision_stream.is_due():
            return
        try:
            self.vision_stream.publish_status(self._vision_stream_status())
            frame = self._vision_stream_renderer.render(self.my.current_game_frame)
            self.vision_stream.publish_rgb_frame(frame)
        except Exception:
            # The stream only shows the match: a frame it cannot draw (a robot at an absurd
            # position) is skipped, never allowed to stop the robots.
            if not getattr(self, "_vision_stream_failed", False):
                self._vision_stream_failed = True
                self.logger.warning("Vision stream could not draw a frame; skipping such frames.", exc_info=True)

    _VS_TEAM_NAMES = [
        "FC Recursion",
        "Real Segfault",
        "Borussia Debugmund",
        "Manchester Bytecode",
        "Inter Malloc",
        "Atletico del Stack",
        "Null Pointer United",
        "Schalke 0x04",
        "Deportivo Kernel",
        "Racing Club de Runtime",
        "Galactic Overhead",
        "Ajax Exception",
        "Off-by-One City",
        "SV Deadlock",
        "Infinite Loop FC",
    ]

    _VS_IDLE_LINES = [
        "The robots are thinking...",
        "Calculating optimal trajectory",
        "Both sides plotting their next move",
        "The crowd holds its breath",
        "Pure silicon determination out there",
        "No human reflexes required",
        "Running at full clock speed",
        "Algorithms at war",
        "404: defence not found",
        "This is peak robot football",
    ]

    _VS_KICK_LINES = [
        "What a strike!",
        "They've let it fly!",
        "Big boot from the robot!",
        "The ball is moving!",
        "Powerful kick!",
        "Sending it downfield!",
    ]

    _VS_GOAL_LINES_YELLOW = [
        "GOAL! Yellow draws blood!",
        "Yellow scores! Unbelievable!",
        "The yellow machine delivers!",
        "Yellow puts it in the net!",
    ]

    _VS_GOAL_LINES_BLUE = [
        "GOAL! Blue strikes back!",
        "Blue finds the net!",
        "Brilliant from the blue side!",
        "Blue pulls one back!",
    ]

    _VS_FOOTBALLER_NAMES = [
        "Martin",
        "Fred",
        "Joel",
        "Louis",
    ]

    @staticmethod
    def _assign_team_names() -> tuple[str, str]:
        import random

        pool = list(VisionStreamStatusMixin._VS_TEAM_NAMES)
        random.shuffle(pool)
        return pool[0], pool[1]

    def _update_vs_commentary(self, ball_speed: float | None, score_blue: int, score_yellow: int) -> None:
        import random

        now = time.monotonic()
        score = (score_blue, score_yellow)

        # Goal scored — highest priority
        if score != self._vs_prev_score:
            if score[1] > self._vs_prev_score[1]:
                line = random.choice(self._VS_GOAL_LINES_YELLOW)
            else:
                line = random.choice(self._VS_GOAL_LINES_BLUE)
            self._vs_commentary = line
            self._vs_commentary_until = now + 5.0
            self._vs_prev_score = score
            self._vs_prev_ball_speed = ball_speed or 0.0
            return

        self._vs_prev_score = score

        # Kick detected
        prev_spd = self._vs_prev_ball_speed
        cur_spd = ball_speed or 0.0
        self._vs_prev_ball_speed = cur_spd
        if cur_spd > _VS_KICK_THRESHOLD and prev_spd <= _VS_KICK_THRESHOLD:
            if now >= self._vs_commentary_until:
                self._vs_commentary = random.choice(self._VS_KICK_LINES)
                self._vs_commentary_until = now + 2.5

        # Idle rotation
        if now >= self._vs_idle_next:
            if now >= self._vs_commentary_until:
                self._vs_commentary = self._VS_IDLE_LINES[self._vs_idle_index % len(self._VS_IDLE_LINES)]
                self._vs_idle_index += 1
            self._vs_idle_next = now + 6.0

    def _vision_stream_status(self) -> dict[str, object]:
        """Build status metadata shown above the browser stream."""
        stage_secs = max(0.0, self.referee_refiner.stage_time_left)
        stage_min = int(stage_secs // 60)
        stage_sec = int(stage_secs % 60)

        blue = self.referee_refiner.blue_team
        yellow = self.referee_refiner.yellow_team

        ball = self.my.current_game_frame.ball if self.my.current_game_frame else None
        ball_speed = (ball.v.x**2 + ball.v.y**2) ** 0.5 if ball is not None else None

        name_yellow, name_blue = (
            self._vs_team_names if self.my_team_is_yellow else (self._vs_team_names[1], self._vs_team_names[0])
        )
        self._update_vs_commentary(ball_speed, blue.score, yellow.score)

        return {
            "time_left": f"{stage_min}:{stage_sec:02d}",
            "score_blue": blue.score,
            "score_yellow": yellow.score,
            "yellow_cards_blue": blue.yellow_cards,
            "yellow_cards_yellow": yellow.yellow_cards,
            "red_cards_blue": blue.red_cards,
            "red_cards_yellow": yellow.red_cards,
            "team_blue": name_blue,
            "team_yellow": name_yellow,
            "mode": self.mode.value,
            "ball_speed": round(ball_speed, 2) if ball_speed is not None else None,
            "commentary": self._vs_commentary,
            "annotations": self._vision_stream_annotations(),
            "roster": self._vision_stream_roster(),
        }

    def _get_robot_name(self, is_friendly: bool, robot_id: int) -> str:
        key = (is_friendly, robot_id)
        if key not in self._vs_robot_names:
            used = set(self._vs_robot_names.values())
            pool = [n for n in self._VS_FOOTBALLER_NAMES if n not in used]
            if not pool:
                pool = self._VS_FOOTBALLER_NAMES
            import random

            self._vs_robot_names[key] = random.choice(pool)
        return self._vs_robot_names[key]

    def _vision_stream_annotations(self) -> list[dict]:
        """Build per-robot label annotations (footballer name) for the overlay canvas."""
        renderer = self._vision_stream_renderer
        game_frame = self.my.current_game_frame
        if renderer is None or game_frame is None:
            return []

        annotations = []
        for robot in game_frame.friendly_robots.values():
            name = self._get_robot_name(True, robot.id)
            px, py = renderer._pos_transform(robot.p.x, -robot.p.y)
            annotations.append({"id": robot.id, "team": "friendly", "px": px, "py": py, "label": name})

        for robot in game_frame.enemy_robots.values():
            name = self._get_robot_name(False, robot.id)
            px, py = renderer._pos_transform(robot.p.x, -robot.p.y)
            annotations.append({"id": robot.id, "team": "enemy", "px": px, "py": py, "label": name})

        return annotations

    def _vision_stream_roster(self) -> list[dict]:
        """Build the player roster list shown below the scoreboard."""
        game_frame = self.my.current_game_frame
        if game_frame is None:
            return []

        _ROLE_LABELS = {
            "GOALKEEPER": "Goalkeeper",
            "DEFENDER": "Defender",
            "STRIKER": "Striker",
            "MIDFIELDER": "Midfielder",
            "UNASSIGNED": "On field",
        }

        role_map: dict = {}
        passer_id: int | None = None
        receiver_id: int | None = None
        try:
            bb = self.my.strategy.blackboard
            if bb is not None:
                role_map = bb.role_map or {}
                passer_id = bb.get("passer_id")
                receiver_id = bb.get("receiver_id")
        except Exception:
            pass

        roster = []
        for robot in game_frame.friendly_robots.values():
            name = self._get_robot_name(True, robot.id)
            if robot.id == passer_id:
                status = "Passing"
            elif robot.id == receiver_id:
                status = "Receiving"
            else:
                role = role_map.get(robot.id)
                status = _ROLE_LABELS.get(role.name, "On field") if role else "On field"
            roster.append(
                {"name": name, "team": "yellow" if game_frame.my_team_is_yellow else "blue", "status": status}
            )

        for robot in game_frame.enemy_robots.values():
            name = self._get_robot_name(False, robot.id)
            role = role_map.get(robot.id)
            status = _ROLE_LABELS.get(role.name, "On field") if role else "On field"
            roster.append(
                {"name": name, "team": "blue" if game_frame.my_team_is_yellow else "yellow", "status": status}
            )

        return roster
