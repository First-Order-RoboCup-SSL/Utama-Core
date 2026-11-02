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

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Union

from utama_core.config.settings import CONTROL_FREQUENCY
from utama_core.entities.data.command import RobotCommand
from utama_core.team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)

PUSH_SPEED = 1.0  # m/s forward for a kick's push; the real robots' cap until 2026-10-28
PUSH_SECONDS = 0.3  # about 0.3 m of travel: the ball is struck, the robot does not chase it
CARRY_SPEED = 0.5  # m/s, the fastest a dribbler-less robot carries the ball
CARRY_TURN = 2.0  # rad/s, the fastest it turns with the ball

_PUSH_TICKS = round(PUSH_SECONDS * CONTROL_FREQUENCY)


@dataclass(frozen=True)
class RobotProfile:
    """The hardware one robot has working.

    `ir`: its ball sensor. Without it `has_ball` is judged from vision (within ~13 cm of
    the ball, whatever way it faces; `RobotInfoRefiner`) instead of the sensor. In a
    simulator this ignores the sim's contact sensor the same way.

    `kick_bend_deg_per_cm`: how far its kick leaves the heading towards the side the ball
    sits on the dribbler, per cm the ball is off-centre. Passes and shots aim against it
    (`kick_bend`), on real robots and in rsim, which kicks a profiled robot so (`real_kick.py`);
    grSim kicks along the heading.

    `kick_bias_deg`: how far its kick leaves the heading vision reports with the ball centred,
    anticlockwise + (a kicker, or a tag stuck on turned); aimed against with the bend. Fitted
    from kicks, so a tag's heading needs no measuring of its own.

    `tag_forward_m`, `tag_left_m`: where its vision tag's centre sits ahead of and left of the
    robot's centre (from its turns on the spot). Vision is corrected by them on real robots only
    (`PositionRefiner`); a tag is a sticker, measured once per lid.

    `kick_speed`: its kick's ball speed (m/s, the median of its kicks); `kick_spread_deg`: how far
    its kicks scatter about bias and bend (rms of the fit). rsim kicks a profiled robot so. None:
    not measured, the sim's speed and no scatter.

    `carry_speed`, `carry_turn`: the fastest it carries the ball, m/s and rad/s, from its carry
    ladder; `move()` holds it to them. None: the team's (`RobotParams.CARRY_*`).

    `pixi run calibrate` measures all but the first three and writes them to the robot's profile
    file (`load_profile`).
    """

    kicker: bool = True
    dribbler: bool = True
    ir: bool = True
    kick_bend_deg_per_cm: float = 0.0
    kick_bias_deg: float = 0.0
    tag_forward_m: float = 0.0
    tag_left_m: float = 0.0
    carry_speed: Optional[float] = None
    carry_turn: Optional[float] = None
    kick_speed: Optional[float] = None
    kick_spread_deg: Optional[float] = None


# One file per robot, by radio ID: `pixi run calibrate` writes it, every script reads it.
PROFILE_DIR = Path(__file__).resolve().parents[4] / "config" / "robots"


def profile_path(radio_id: int, directory: Path = PROFILE_DIR) -> Path:
    return directory / f"radio_{radio_id}.json"


def load_profile(radio_id: int, directory: Path = PROFILE_DIR) -> RobotProfile:
    """The profile of our robot on radio `radio_id`; a blank one if it has no file."""
    path = profile_path(radio_id, directory)
    if not path.exists():
        return RobotProfile()
    return RobotProfile(**json.loads(path.read_text())["profile"])


def save_profile(radio_id: int, profile: RobotProfile, calibrated: dict, directory: Path = PROFILE_DIR) -> Path:
    """Writes `profile` to the robot's file, with `calibrated`: where its numbers came from."""
    path = profile_path(radio_id, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"radio_id": radio_id, "profile": asdict(profile), "calibrated": calibrated}
    path.write_text(json.dumps(body, indent=1) + "\n")
    return path


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
