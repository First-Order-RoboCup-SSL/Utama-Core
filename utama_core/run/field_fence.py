"""Keeps our robots inside the area in play (`field_bounds`, what the camera sees) on camera runs.

A tactic chasing a ball that left the camera's half drove its robot after it, out of sight,
where nothing tracks it (a 2026-10-22 grSim test of `examples/shooting_drill.py`: the
attacker 0.9 m past the half-way line). The field check only halts once a robot is already
0.3 m out. `FencedRobotController` sits under the strategy, like `ProfiledRobotController`, and
caps each robot's speed towards each edge of the area to what it can still stop from before the
line; speed away from an edge is never limited, so a robot outside drives back in. It knows the
robots' poses, nothing of the tactics, so it holds whichever tactic or skill made the command.

Only edges inside the field are fenced (the half-way line on a half-field run): a robot must
cross the touch and goal lines to fetch a ball out of play, or a restart never completes
(`custom_referee/actions.py`, `_clamp_to_field_or_ball`), and the cameras see past them.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Union

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.entities.game.field import FieldBounds

BRAKE = 2.0  # m/s^2: half the robots' MAX_ACCELERATION, for a floor that grips less
DELAY = 0.1  # s from a command to the robot acting on it (radio, firmware), at full speed


def _max_speed_towards(distance: float) -> float:
    """The fastest a robot `distance` m from a line may move towards it and still stop before it."""
    if distance <= 0:
        return 0.0
    return BRAKE * (-DELAY + math.sqrt(DELAY * DELAY + 2 * distance / BRAKE))


def inner_edges(bounds: FieldBounds, full: FieldBounds) -> Optional[FieldBounds]:
    """`bounds` with each edge on `full`'s outer lines moved out to infinity; None when every
    edge is on them (the cameras see the whole field: nothing to fence)."""
    (left, top), (right, bottom) = bounds.top_left, bounds.bottom_right
    (f_left, f_top), (f_right, f_bottom) = full.top_left, full.bottom_right
    opened = (
        -math.inf if left <= f_left else left,
        math.inf if top >= f_top else top,
        math.inf if right >= f_right else right,
        -math.inf if bottom <= f_bottom else bottom,
    )
    if all(math.isinf(edge) for edge in opened):
        return None
    return FieldBounds(top_left=(opened[0], opened[1]), bottom_right=(opened[2], opened[3]))


def fence(x: float, y: float, orientation: float, command: RobotCommand, bounds: FieldBounds) -> RobotCommand:
    """`command` for a robot at (`x`, `y`) facing `orientation`, its speed towards each edge of
    `bounds` capped; unchanged when no cap applies."""
    c, s = math.cos(orientation), math.sin(orientation)
    vx = command.local_forward_vel * c - command.local_left_vel * s
    vy = command.local_forward_vel * s + command.local_left_vel * c
    (left, top), (right, bottom) = bounds.top_left, bounds.bottom_right
    fx = min(vx, _max_speed_towards(right - x)) if vx > 0 else max(vx, -_max_speed_towards(x - left))
    fy = min(vy, _max_speed_towards(top - y)) if vy > 0 else max(vy, -_max_speed_towards(y - bottom))
    if fx == vx and fy == vy:
        return command
    return command._replace(local_forward_vel=fx * c + fy * s, local_left_vel=-fx * s + fy * c)


class FencedRobotController:
    """A team's robot controller whose robots' commands are fenced to `bounds` (see module
    docstring). `game` gives the side's current `Game`, whose poses are in vision coordinates
    like `bounds`; a robot it does not see is sent its command unchanged."""

    def __init__(self, controller, bounds: FieldBounds, game: Callable[[], Optional[Game]]):
        self._controller = controller
        self._bounds = bounds
        self._game = game

    def __getattr__(self, name):
        return getattr(self._controller, name)

    def add_robot_commands(
        self,
        robot_commands: Union[RobotCommand, Dict[int, RobotCommand]],
        robot_id: Optional[int] = None,
    ) -> None:
        if isinstance(robot_commands, RobotCommand):
            if robot_id is None:
                raise ValueError("robot_id cannot be None type!")
            robot_commands = {robot_id: robot_commands}
        game = self._game()
        robots = game.friendly_robots if game is not None else {}
        fenced = {}
        for rid, command in robot_commands.items():
            robot = robots.get(rid)
            fenced[rid] = (
                command if robot is None else fence(robot.p.x, robot.p.y, robot.orientation, command, self._bounds)
            )
        self._controller.add_robot_commands(fenced)
