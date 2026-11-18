"""Keeps each real robot's kicker charged, so its first kick after a quiet spell is a full one.

The kicker recharges about 1 s after each kick, then loses its charge if it does not kick for
about 30 s: a robot that has not kicked for `KEEP_S` fires an empty kick, when the ball is not on
it (its ball sensor) or near it (vision, `CLEAR_M`), and never in a HALT (people reach in to move
robots). A robot not yet known to have kicked fires at the first chance, so play starts charged.
"""

from __future__ import annotations

import math

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.entities.referee.referee_command import RefereeCommand

KEEP_S = 25.0  # the charge holds about 30 s
CLEAR_M = 0.3  # nearer the ball than this an empty kick could hit it


class KickerCharge:
    def __init__(self):
        self.last_kick: dict[int, float] = {}  # robot id -> game time of its last kick

    def apply(self, game: Game, commands: dict[int, RobotCommand]) -> dict[int, RobotCommand]:
        """`commands` with an empty kick added for each robot whose charge is going stale."""
        ts = game.ts
        halted = game.referee is not None and game.referee.referee_command == RefereeCommand.HALT
        ball = game.ball
        out = dict(commands)
        for rid, cmd in commands.items():
            if cmd.kick or cmd.chip:
                self.last_kick[rid] = ts
                continue
            robot = game.friendly_robots.get(rid)
            if halted or ball is None or robot is None or robot.has_ball:
                continue
            if robot.p.distance_to(ball.p.to_2d()) < CLEAR_M:
                continue
            if ts - self.last_kick.get(rid, -math.inf) >= KEEP_S:
                out[rid] = cmd._replace(kick=True)
                self.last_kick[rid] = ts
        return out
