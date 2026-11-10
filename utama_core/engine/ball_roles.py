"""Who on the team deals with the ball this tick, whichever tactics each robot is in.

Tactics each decide for their own robots, so two of them (an attack slot and a press slot, say)
send one robot each at the same ball. In the 2026-11-09 round-robin a side with a robot on the ball
had a teammate within 0.6 m of it 46% of the time. Run on the commands every tactic returned:

- the robot a free ball is rolling to meets it on its path, whether or not a pass was announced,
  and stays on it until the ball arrives or goes elsewhere;
- a robot a teammate is turning to pass to stops and faces the ball;
- of the robots driving at a loose ball, the nearest carries on and the others cover it. With the
  ball ours they keep clear of it where they are: covering pulled attackers back to our side of it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable, Optional

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.shared.pass_and_score_geometry import (
    ball_line_receive_point,
    clamp_outside_enemy_defense_area,
    clamp_outside_own_defense_area,
    clamp_to_field,
    receiver_of_rolling_ball,
    still_receiving,
)
from utama_core.skills.src.utils.move_utils import move, turn_on_spot

if TYPE_CHECKING:
    from utama_core.engine.context import TickContext
    from utama_core.entities.game import Game

COVER_M = 1.0  # from the ball, on the line to our goal
_COVER_ANGLES_RAD = (0.0, 0.7, -0.7, 1.4, -1.4)
ON_BALL_SWITCH_MARGIN_M = 0.2  # nearer than the robot already on the ball by this, or it stays on it
_LIVE_COMMANDS = frozenset({RefereeCommand.NORMAL_START, RefereeCommand.FORCE_START})


class BallRoles:
    def __init__(self) -> None:
        self._on_ball: Optional[int] = None
        self._receiving: Optional[int] = None

    def reset(self) -> None:
        self._on_ball = None
        self._receiving = None

    def apply(
        self, game: "Game", ctx: "TickContext", commands: dict[int, RobotCommand], outfield: Iterable[int]
    ) -> dict[int, RobotCommand]:
        """`commands` with the robots above changed. Live play only."""
        referee = getattr(game, "referee", None)
        if getattr(game, "ball", None) is None or referee is None or referee.referee_command not in _LIVE_COMMANDS:
            self.reset()
            return commands

        candidates = [rid for rid in outfield if rid in commands and rid in game.friendly_robots]
        # A new pick replaces the last one: the ball's velocity is unreliable on the first tick of a kick,
        # and a robot chosen on it kept the job. The hold is for ticks when the reach check finds nobody.
        pick = receiver_of_rolling_ball(game, candidates)
        if pick is not None or self._receiving not in candidates or not still_receiving(game, self._receiving):
            self._receiving = pick
        receiver = self._receiving
        if receiver is not None and not game.friendly_robots[receiver].has_ball:
            self._receive(game, ctx, commands, receiver)
        for robot_id in ctx.expected_receivers:
            if robot_id in commands and robot_id != receiver and not game.friendly_robots[robot_id].has_ball:
                self._get_ready(game, ctx, commands, robot_id)
        self._one_on_the_ball(game, ctx, commands, receiver)
        return commands

    def _receive(self, game: "Game", ctx: "TickContext", commands: dict[int, RobotCommand], robot_id: int) -> None:
        spot = ball_line_receive_point(game, robot_id)
        if spot is None:
            return
        ctx.trace(game, f"ball[{robot_id}]", "receiving")
        robot = game.friendly_robots[robot_id]
        commands[robot_id] = move(
            game=game,
            motion_controller=ctx.motion_controller,
            robot_id=robot_id,
            target_coords=clamp_outside_enemy_defense_area(game, clamp_outside_own_defense_area(game, spot)),
            target_oren=robot.p.angle_to(game.ball.p.to_2d()),
            dribbling=True,
            # A 5 m/s pass 2 m away arrives in 0.4 s. In 20 free-kick drills sprinting, with the pick
            # above, took passes received from 9 to 14 and balls out of play from 13 to 8 (2026-11-10).
            sprint=True,
        )

    def _get_ready(self, game: "Game", ctx: "TickContext", commands: dict[int, RobotCommand], robot_id: int) -> None:
        ctx.trace(game, f"ball[{robot_id}]", "waiting for the pass")
        robot = game.friendly_robots[robot_id]
        commands[robot_id] = turn_on_spot(
            game=game,
            motion_controller=ctx.motion_controller,
            robot_id=robot_id,
            target_oren=robot.p.angle_to(game.ball.p.to_2d()),
            dribbling=True,
        )

    def _one_on_the_ball(
        self, game: "Game", ctx: "TickContext", commands: dict[int, RobotCommand], receiver: Optional[int]
    ) -> None:
        claims = sorted(rid for rid in ctx.ball_claims if rid in commands and rid in game.friendly_robots)
        holder = next((rid for rid, robot in game.friendly_robots.items() if robot.has_ball), None)
        ours = receiver if receiver is not None else holder
        if ours is not None:
            self._on_ball = ours
            for robot_id in claims:
                if robot_id != ours:
                    ctx.trace(game, f"ball[{robot_id}]", "keeping clear")
                    self._send(game, ctx, commands, robot_id, _clear_of_ball(game, robot_id))
            return
        if not claims:
            self._on_ball = None
            return
        self._on_ball = self._nearest(game, claims)
        closest = min(claims, key=lambda rid: game.friendly_robots[rid].p.distance_to(game.ball.p.to_2d()))
        for rank, robot_id in enumerate(rid for rid in claims if rid not in (self._on_ball, closest)):
            ctx.trace(game, f"ball[{robot_id}]", "covering")
            self._send(game, ctx, commands, robot_id, _cover_point(game, rank))

    def _send(
        self, game: "Game", ctx: "TickContext", commands: dict[int, RobotCommand], robot_id: int, target: Vector2D
    ) -> None:
        commands[robot_id] = move(
            game=game,
            motion_controller=ctx.motion_controller,
            robot_id=robot_id,
            target_coords=target,
            target_oren=game.friendly_robots[robot_id].p.angle_to(game.ball.p.to_2d()),
        )

    def _nearest(self, game: "Game", claims: list[int]) -> int:
        ball = game.ball.p.to_2d()
        distance = {rid: game.friendly_robots[rid].p.distance_to(ball) for rid in claims}
        nearest = min(claims, key=lambda rid: distance[rid])
        current = self._on_ball
        if current in distance and distance[nearest] > distance[current] - ON_BALL_SWITCH_MARGIN_M:
            return current
        return nearest


def _clear_of_ball(game: "Game", robot_id: int) -> Vector2D:
    """Where the robot stands, or straight out from the ball to `COVER_M` if it is nearer than that."""
    ball = game.ball.p.to_2d()
    position = game.friendly_robots[robot_id].p
    offset = position - ball
    distance = offset.mag()
    if distance >= COVER_M or distance == 0.0:
        return position
    point = ball + offset * (COVER_M / distance)
    return clamp_outside_enemy_defense_area(game, clamp_outside_own_defense_area(game, clamp_to_field(point, game)))


def _cover_point(game: "Game", rank: int) -> Vector2D:
    """`COVER_M` from the ball towards our goal; the further covers spread either side of that line."""
    ball = game.ball.p.to_2d()
    to_goal = ball.angle_to(Vector2D(float(game.field.my_goal_line[0][0]), 0.0))
    angle = to_goal + _COVER_ANGLES_RAD[rank % len(_COVER_ANGLES_RAD)]
    point = ball + Vector2D(math.cos(angle), math.sin(angle)) * COVER_M
    return clamp_outside_enemy_defense_area(game, clamp_outside_own_defense_area(game, clamp_to_field(point, game)))
