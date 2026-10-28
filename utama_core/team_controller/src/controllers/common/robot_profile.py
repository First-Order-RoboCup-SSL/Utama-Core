"""What each physical robot can do (kicker, dribbler, ball sensor), and the stand-ins for what it cannot.

Robots go to a test with a broken kicker or dribbler and are tested anyway. Rather than
teach every tactic about broken hardware, `ProfiledRobotController` wraps a team's robot
controller and rewrites each command for what that robot has:

- **No kicker:** a kick (or chip) becomes a push: drive straight forward at `PUSH_SPEED`
  for `PUSH_SECONDS`, so the robot's front strikes the ball the way it was aiming. The
  tactic aimed by turning before it kicked, so the push goes where the kick would have,
  only slower. A stop command (all zero, as HALT and `close()` send) ends a push at once.
- **No dribbler:** while the tactic asks to dribble, the dribbler stays off and the robot
  moves as one can without it and keep the ball in front: never backwards (the ball
  stays behind) and no faster than `CARRY_SPEED` / `CARRY_TURN` (scaled together, so a
  pivot around the ball keeps its shape). It pushes the ball along instead of holding it.

It sits under the strategy in every mode, so a test with a broken robot can be rehearsed
in rsim or grSim first. It knows nothing of the game, only each robot's commands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Union

from utama_core.config.physical_constants import CARRY_SPEED, CARRY_TURN
from utama_core.config.settings import CONTROL_FREQUENCY
from utama_core.entities.data.command import RobotCommand
from utama_core.team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)

PUSH_SPEED = 1.0  # m/s forward for a kick's push; REAL_PARAMS.MAX_VEL, the real robots' cap
PUSH_SECONDS = 0.3  # about 0.3 m of travel: the ball is struck, the robot does not chase it

_PUSH_TICKS = round(PUSH_SECONDS * CONTROL_FREQUENCY)


@dataclass(frozen=True)
class RobotProfile:
    """The hardware one robot has working.

    `ir`: its ball sensor. Without it `has_ball` is judged from vision (within ~13 cm of
    the ball, whatever way it faces; `RobotInfoRefiner`) instead of the sensor. In a
    simulator this ignores the sim's contact sensor the same way.
    """

    kicker: bool = True
    dribbler: bool = True
    ir: bool = True


def _is_stop(c: RobotCommand) -> bool:
    return not (c.local_forward_vel or c.local_left_vel or c.angular_vel or c.kick or c.chip or c.dribble)


def _carry(c: RobotCommand) -> RobotCommand:
    """`c` as a dribbler-less robot keeping the ball in front can follow it."""
    forward = max(c.local_forward_vel, 0.0)
    speed = math.hypot(forward, c.local_left_vel)
    scale = min(1.0, CARRY_SPEED / speed if speed else 1.0, CARRY_TURN / abs(c.angular_vel) if c.angular_vel else 1.0)
    return c._replace(
        local_forward_vel=forward * scale,
        local_left_vel=c.local_left_vel * scale,
        angular_vel=c.angular_vel * scale,
        dribble=False,
    )


class ProfiledRobotController:
    """A team's robot controller with each robot's commands fitted to its `RobotProfile`.

    `profiles` is keyed by the robot ID the strategy commands (the vision ID on real
    robots); a robot without a profile is sent its commands unchanged. Everything other
    than adding commands goes straight to the wrapped controller.
    """

    def __init__(self, controller: AbstractRobotController, profiles: Dict[int, RobotProfile]):
        self._controller = controller
        self._profiles = profiles
        self._push_ticks_left: Dict[int, int] = {}

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
        self._controller.add_robot_commands({rid: self.fit(rid, c) for rid, c in robot_commands.items()})

    def fit(self, robot_id: int, command: RobotCommand) -> RobotCommand:
        """`command` as robot `robot_id` can carry it out; advances its push by one tick."""
        profile = self._profiles.get(robot_id)
        if profile is None:
            return command
        if _is_stop(command):
            self._push_ticks_left.pop(robot_id, None)
            return command
        if not profile.kicker:
            if self._push_ticks_left.get(robot_id, 0) == 0 and (command.kick or command.chip):
                self._push_ticks_left[robot_id] = _PUSH_TICKS
            if self._push_ticks_left.get(robot_id, 0) > 0:
                self._push_ticks_left[robot_id] -= 1
                return RobotCommand(PUSH_SPEED, 0.0, 0.0, kick=False, chip=False, dribble=False)
        if not profile.dribbler and command.dribble:
            command = _carry(command)
        return command
