"""Aim and force a kick when holding the ball on any longer would cost it.

Tactics decide when to shoot or pass; this is the backstop behind all of them, run on the commands
they return. A robot that finds no shot keeps sliding for one with the ball on its dribbler, with
nothing to end that. In the 2026-11-09 round-robin that gave 387 solo excessive-dribbling fouls,
and with an enemy on the same ball a jam of 90 s per match, 230 s at worst, the ball creeping just
enough for the no-progress rule never to fire. Kept here and not in tactics' memory so a hand-over
between tactics in the middle of a carry does not restart the count, as the referee's does not.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.shared.pass_and_score_geometry import (
    ANNOUNCE_PASS_RAD,
    RELEASE_LANE_CLEARANCE,
    open_teammate,
    oriented_towards,
    reaches_own_goal,
    release_heading,
)
from utama_core.skills.utils.move_utils import turn_on_spot

if TYPE_CHECKING:
    from utama_core.engine.context import TickContext
    from utama_core.entities.game import Game

# Past tactics' own CARRY_LIMIT_M (0.8): their carries are not cut short, this catches the rest.
CARRY_RELEASE_M = 0.96
# Aim from here: a carry of 0.8 turning to shoot has stopped, and is left to its tactic.
CARRY_AIM_M = 0.85
# Ball travel before a kick sent now leaves the robot: the command delay and a tick.
_RELEASE_LAG_S = 0.12
_CARRY_BRAKING_MPS2 = 4.0
PINNED_RELEASE_S = 1.5
# With a teammate to kick to there is nothing to wait for: an enemy on the ball only gets stronger.
PINNED_WITH_LANE_S = 0.3
# Stood on the ball this long with it not moving. Whatever a tactic is waiting for, this is what the
# stalls of the 2026-11-09 round-robin looked like: nothing else on the pitch moves it.
HELD_STILL_S = 2.0
_HELD_STILL_M = 0.05
# Kicked blind along the heading, 82% of pinned releases and 41% of carry ones went out of play
# (12 matches, 2026-11-09). A turn gets most of that back, without letting a release wait.
AIM_LIMIT_S = 0.6
_AIM_TOLERANCE_RAD = 0.15
CONTACT_M = 0.13  # robot centre to ball with both touching: 0.09 + 0.0215 and some slack
_LIVE_COMMANDS = frozenset({RefereeCommand.NORMAL_START, RefereeCommand.FORCE_START})
_KICK = RobotCommand(local_forward_vel=0, local_left_vel=0, angular_vel=0, kick=1, chip=0, dribble=0)


class BallRelease:
    def __init__(self) -> None:
        self._carry_origin: dict[int, tuple[float, float]] = {}
        self._pinned_since: dict[int, float] = {}
        self._aiming_since: dict[int, float] = {}
        self._still_at: dict[int, tuple[tuple[float, float], float]] = {}

    def reset(self) -> None:
        self._still_at.clear()
        self._carry_origin.clear()
        self._pinned_since.clear()
        self._aiming_since.clear()

    def apply(self, game: "Game", ctx: "TickContext", commands: dict[int, RobotCommand]) -> dict[int, RobotCommand]:
        """`commands` with a kick for any robot that must release the ball now. Live play only."""
        referee = getattr(game, "referee", None)
        ball = getattr(game, "ball", None)
        if ball is None or referee is None or referee.referee_command not in _LIVE_COMMANDS:
            self.reset()
            return commands

        here = (ball.p.x, ball.p.y)
        enemy_on_ball = any(math.dist((e.p.x, e.p.y), here) <= CONTACT_M for e in game.enemy_robots.values())
        for robot_id, robot in game.friendly_robots.items():
            if enemy_on_ball and math.dist((robot.p.x, robot.p.y), here) <= CONTACT_M:
                self._pinned_since.setdefault(robot_id, game.ts)
            else:
                self._pinned_since.pop(robot_id, None)
            if not robot.has_ball:
                self._carry_origin.pop(robot_id, None)
                self._aiming_since.pop(robot_id, None)
                self._still_at.pop(robot_id, None)
                continue
            origin = self._carry_origin.setdefault(robot_id, here)
            spot, still_since = self._still_at.setdefault(robot_id, (here, game.ts))
            if math.dist(here, spot) >= _HELD_STILL_M:
                self._still_at[robot_id], still_since = (here, game.ts), game.ts

            command: Optional[RobotCommand] = commands.get(robot_id)
            if command is None or command.kick or command.chip:
                continue
            carried = math.dist(here, origin)
            speed = math.hypot(ball.v.x, ball.v.y)
            pinned_since = self._pinned_since.get(robot_id)
            teammate = open_teammate(game, robot_id, RELEASE_LANE_CLEARANCE)
            if carried + speed * _RELEASE_LAG_S >= CARRY_RELEASE_M:
                if not reaches_own_goal(game, robot.p, robot.orientation):
                    ctx.trace(game, f"release[{robot_id}]", "carry limit")
                    commands[robot_id] = _KICK
                continue
            if carried + speed * _RELEASE_LAG_S + speed**2 / (2.0 * _CARRY_BRAKING_MPS2) >= CARRY_AIM_M:
                reason = "carry limit"
            elif pinned_since is not None and game.ts - pinned_since >= PINNED_RELEASE_S:
                reason = "pinned"
            elif pinned_since is not None and game.ts - pinned_since >= PINNED_WITH_LANE_S and teammate is not None:
                reason = "pinned, teammate open"
            elif game.ts - still_since >= HELD_STILL_S:
                reason = "held still"
            else:
                self._aiming_since.pop(robot_id, None)
                continue
            aiming_for = game.ts - self._aiming_since.setdefault(robot_id, game.ts)
            heading = release_heading(game, robot_id)
            if teammate is not None and oriented_towards(game, robot_id, heading, ANNOUNCE_PASS_RAD):
                ctx.expected_receivers.add(teammate)
            aimed = oriented_towards(game, robot_id, heading, _AIM_TOLERANCE_RAD)
            if (aimed or aiming_for >= AIM_LIMIT_S) and not reaches_own_goal(game, robot.p, robot.orientation):
                ctx.trace(game, f"release[{robot_id}]", reason)
                commands[robot_id] = _KICK
            else:
                ctx.trace(game, f"release[{robot_id}]", f"aiming · {reason}")
                commands[robot_id] = turn_on_spot(
                    game=game,
                    motion_controller=ctx.motion_controller,
                    robot_id=robot_id,
                    target_oren=heading,
                    dribbling=True,
                )
        return commands
