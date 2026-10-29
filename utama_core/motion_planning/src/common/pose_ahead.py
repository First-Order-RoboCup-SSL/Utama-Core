"""Our robots where they will be when a command sent now takes effect."""

import dataclasses
import math

from utama_core.config.enums import Mode
from utama_core.config.robot_params import GRSIM_PARAMS, REAL_PARAMS, RSIM_PARAMS
from utama_core.entities.game import Game
from utama_core.entities.game.robot import Robot

_PARAMS_BY_MODE = {Mode.RSIM: RSIM_PARAMS, Mode.GRSIM: GRSIM_PARAMS, Mode.REAL: REAL_PARAMS}

# A turn rate is measured between consecutive frames no further apart than this; across a longer
# gap (the robot was not seen, or not moved for a while) it is taken as zero.
_MAX_FRAME_GAP_S = 0.1


def control_latency_s(mode) -> float:
    return _PARAMS_BY_MODE[Mode(mode)].CONTROL_LATENCY_S


class PoseAhead:
    """Our robots `latency_s` ahead of vision, at their current velocity and turn rate.

    Vision shows a robot where it was, and a command sent now acts later still: on the real
    robots about 79 ms in all (`RobotParams.CONTROL_LATENCY_S`). Planning from the seen pose, a
    robot brakes and swerves that late, and its world velocity is turned into robot-frame
    commands with a heading that is out of date while it turns. In rsim with that delay (2026-10-28):
    driving 2.2 m while turning half a turn, the robot left its straight line by 30 cm, 7 cm with
    this (4 cm with no delay); the motion benchmark's collisions went from 7 to 3.

    Only our robots move ahead, and only for planning their motion (`move()`): the game itself,
    its history and every other decision keep what vision saw. With no latency (the simulators)
    `game()` returns the game unchanged.
    """

    def __init__(self, latency_s: float):
        self.latency_s = latency_s
        self._source = None
        self._ahead = None
        self._headings: dict[int, tuple[float, float]] = {}  # robot id -> (ts, orientation) last seen

    def game(self, game: Game) -> Game:
        if self.latency_s <= 0:
            return game
        if game is self._source and self._ahead.ts == game.ts:  # one tick, several robots
            return self._ahead
        robots = {rid: self._ahead_of(robot, game.ts) for rid, robot in game.friendly_robots.items()}
        self._source, self._ahead = game, game.with_friendly_robots(robots)
        return self._ahead

    def _ahead_of(self, robot: Robot, ts: float) -> Robot:
        turn_rate = 0.0
        last = self._headings.get(robot.id)
        if last is not None and 0 < ts - last[0] <= _MAX_FRAME_GAP_S:
            turned = (robot.orientation - last[1] + math.pi) % (2 * math.pi) - math.pi
            turn_rate = turned / (ts - last[0])
        if last is None or ts != last[0]:
            self._headings[robot.id] = (ts, robot.orientation)
        t = self.latency_s
        orientation = (robot.orientation + turn_rate * t + math.pi) % (2 * math.pi) - math.pi
        return dataclasses.replace(robot, p=robot.p + robot.v * t, orientation=orientation)
