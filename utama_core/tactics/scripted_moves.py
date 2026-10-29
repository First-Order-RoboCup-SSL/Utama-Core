"""Fixed moves for measuring a robot: the commands are a script, not a decision.

`ScriptedMovesTactic` sends `CALIBRATION_SCRIPT`'s raw velocity commands, step by step, to the
one robot it is given, then stands still. Run with a game log being recorded
(`StrategyRunner(game_log_path=...)`), the log holds each command as sent and the robot's
motion as the camera saw it; `tools/robot_calibration.py` turns that into the robot's
numbers (vision noise, delay, speed and turn tracking, acceleration). The ball part of a
calibration fetches the ball itself instead (`examples/ball_trials.py`).

Each move goes and comes back (forward then backward, left then right), so the robot
ends near where it started; the longest is 0.8 m forward. Pauses (HALT, STOP) do not
advance the script, and the step a pause interrupted starts again, so no measurement is
split. A pause within a there-and-back (or a turn and its return) first drives the robot back
to where the pair began, then replays the pair: restarted where it stood, a half-done "back"
drove it on past its start, and each pause added more (2026-10-29: past the goal line). The
drive back varies its command every tick, so the analysis, which takes constant commands
(`analysis/robot_calibration.py`), leaves it out.

A move with a `prompt` holds the robot still until `proceed()`: the operator sets something up
(a ball on the dribbler) and says so. `examples/calibrate_robot.py` asks in the terminal.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game


class Move(NamedTuple):
    name: str
    seconds: float
    forward: float = 0.0  # m/s, robot frame (kicker forward)
    left: float = 0.0  # m/s
    turn: float = 0.0  # rad/s, anticlockwise
    dribble: bool = False
    kick: bool = False
    prompt: Optional[str] = None  # stand still until `proceed()`; `seconds` is unused
    home: bool = False  # a pause from here to the next `home` move returns the robot to where this one began


_REST = 1.5  # s still between moves, so each response settles
_GAP_S = 0.1  # a tick this long after the last one means play was paused
# Driving back to where a pair began after a pause: speed and turn caps (the carry caps, slow
# enough to stop on the spot), floors that still move the robot (REAL_PARAMS.MIN_TURN 0.3), and
# how near counts as there.
_RETURN_SPEED, _RETURN_TURN = 0.3, 1.0
_RETURN_MIN_SPEED, _RETURN_MIN_TURN = 0.08, 0.4
_RETURN_POS_M, _RETURN_HEADING_RAD = 0.03, 0.05


def _there_and_back(name: str, speed: float, distance: float, sideways: bool = False) -> list[Move]:
    t = distance / speed
    axis = "left" if sideways else "forward"
    go, back = Move(name, t, home=True, **{axis: speed}), Move(f"{name} back", t, **{axis: -speed})
    return [go, Move("rest", _REST), back, Move("rest", _REST)]


def _turns(rate: float, seconds: float) -> list[Move]:
    name = f"turn {rate:g} rad/s"
    return [
        Move(name, seconds, turn=rate, home=True),
        Move("rest", _REST),
        Move(f"{name} back", seconds, turn=-rate),
        Move("rest", _REST),
    ]


def _towards(pose: tuple[float, float, float], home: tuple[float, float, float]) -> Optional[RobotCommand]:
    """A robot-frame command from `pose` (x, y, heading) towards `home`; None once there."""
    dx, dy = home[0] - pose[0], home[1] - pose[1]
    turn_err = math.atan2(math.sin(home[2] - pose[2]), math.cos(home[2] - pose[2]))
    dist = math.hypot(dx, dy)
    if dist < _RETURN_POS_M and abs(turn_err) < _RETURN_HEADING_RAD:
        return None
    speed = 0.0 if dist < _RETURN_POS_M else max(_RETURN_MIN_SPEED, min(_RETURN_SPEED, 1.5 * dist))
    turn = 0.0
    if abs(turn_err) >= _RETURN_HEADING_RAD:
        turn = math.copysign(max(_RETURN_MIN_TURN, min(_RETURN_TURN, 2.0 * abs(turn_err))), turn_err)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    ux, uy = (dx / dist, dy / dist) if dist > 0 else (0.0, 0.0)
    return RobotCommand(
        speed * (c * ux + s * uy), speed * (-s * ux + c * uy), turn, kick=False, chip=False, dribble=False
    )


# About 70 s. Speeds stay within the real robots' caps (REAL_PARAMS: 1 m/s, 4 rad/s).
CALIBRATION_SCRIPT: tuple[Move, ...] = (
    Move("still", 10.0),  # vision noise: robot (and a ball placed in view) at rest
    *_there_and_back("forward 0.3 m/s", 0.3, 0.6),
    *_there_and_back("forward 0.6 m/s", 0.6, 0.8),
    *_there_and_back("forward 1 m/s", 1.0, 0.8),
    *_there_and_back("sideways 0.5 m/s", 0.5, 0.6, sideways=True),
    *_turns(2.0, 3.0),
    *_turns(4.0, 1.6),
    Move("still", 3.0),
)


class ScriptedMovesTactic(BaseTactic[None]):
    """Plays `script` on its first robot; others (and the first, once done) stand still."""

    tag = TacticTag.MIXED

    def __init__(self, script: tuple[Move, ...] = CALIBRATION_SCRIPT):
        self.script = script
        # Progress lives here, not in `mem`: the kernel may reset `mem` when play resumes.
        self._step = 0
        self._step_elapsed = 0.0
        self._last_ts: Optional[float] = None
        self._proceed = False
        self._home: Optional[tuple[float, float, float]] = None  # where the latest `home` move began
        self._home_step: Optional[int] = None
        self._returning = False  # after a pause: driving back to `_home`, then resting `_REST`
        self._rested = 0.0

    def initial_mem(self) -> None:
        return None

    @property
    def done(self) -> bool:
        return self._step >= len(self.script)

    @property
    def step(self) -> int:
        return self._step

    @property
    def waiting_for(self) -> Optional[str]:
        """The current move's prompt, while it holds for `proceed()`."""
        return None if self.done else self.script[self._step].prompt

    def proceed(self) -> None:
        """The operator has done what the current prompt asked: carry on next tick."""
        self._proceed = True

    @property
    def status(self) -> str:
        """The current move, for the operator; changes only when the move does."""
        if self.done:
            return "done"
        m = self.script[self._step]
        if self._returning:
            return f"[{self._step + 1:>2}/{len(self.script)}] back to where {m.name} began, then again"
        return f"[{self._step + 1:>2}/{len(self.script)}] {m.name} ({m.seconds:g} s)"

    def _home_before(self, step: int) -> Optional[int]:
        """The `home` move a pause at `step` returns to: the latest at or before it, none past a prompt."""
        for i in range(step, -1, -1):
            if self.script[i].home:
                return i
            if self.script[i].prompt is not None:
                return None
        return None

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: None
    ) -> tuple[dict[RobotId, RobotCommand], None]:
        still = RobotCommand(0.0, 0.0, 0.0, kick=False, chip=False, dribble=False)
        commands = {rid: still for rid in robot_ids}
        if not robot_ids:
            return commands, mem
        dt = 0.0 if self._last_ts is None else game.ts - self._last_ts
        self._last_ts = game.ts
        robot = getattr(game, "friendly_robots", {}).get(robot_ids[0])
        pose = None if robot is None else (robot.p.x, robot.p.y, robot.orientation)
        if dt > _GAP_S:
            self._step_elapsed = 0.0  # resumed after a pause: this step again, from its start
            home = None if self.done else self._home_before(self._step)
            if home is not None and home == self._home_step:
                self._step, self._returning, self._rested = home, True, 0.0
        else:
            self._step_elapsed += dt
        if self._returning:
            back = None if pose is None else _towards(pose, self._home)
            if back is not None or pose is None:
                self._rested = 0.0
                if back is not None:
                    commands[robot_ids[0]] = back
                return commands, mem
            self._rested += dt
            if self._rested < _REST:
                return commands, mem
            self._returning, self._step_elapsed = False, 0.0
        while not self.done:
            m = self.script[self._step]
            if m.prompt is not None:
                if not self._proceed:
                    break
                self._proceed, self._step_elapsed = False, 0.0
            elif self._step_elapsed < m.seconds:
                break
            else:
                self._step_elapsed -= m.seconds
            self._step += 1
        if not self.done and self.script[self._step].home and self._home_step != self._step and pose is not None:
            self._home, self._home_step = pose, self._step
        if not self.done and self.script[self._step].prompt is None:
            m = self.script[self._step]
            commands[robot_ids[0]] = RobotCommand(m.forward, m.left, m.turn, kick=m.kick, chip=False, dribble=m.dribble)
        return commands, mem
