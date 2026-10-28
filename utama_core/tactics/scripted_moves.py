"""Fixed moves for measuring a robot: the commands are a script, not a decision.

`ScriptedMovesTactic` sends `CALIBRATION_SCRIPT`'s raw velocity commands, step by step, to the
one robot it is given, then stands still. Run with a game log being recorded
(`StrategyRunner(game_log_path=...)`), the log holds each command as sent and the robot's
motion as the camera saw it; `tools/robot_calibration.py` turns that into the robot's
numbers (vision noise, delay, speed and turn tracking, acceleration, and with `BALL_SCRIPT` the
dribbler and kicker).

Each move goes and comes back (forward then backward, left then right), so the robot
ends near where it started; the longest is 0.8 m forward. Pauses (HALT, STOP) do not
advance the script, and the step a pause interrupted starts again, so no measurement is
split: halt to put the robot back, then Force Start to carry on.

A move with a `prompt` holds the robot still until `proceed()`: the operator sets something up
(a ball on the dribbler) and says so. `examples/calibrate_robot.py` asks in the terminal.
"""

from __future__ import annotations

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


_REST = 1.5  # s still between moves, so each response settles
_GAP_S = 0.1  # a tick this long after the last one means play was paused


def _there_and_back(name: str, speed: float, distance: float, sideways: bool = False) -> list[Move]:
    t = distance / speed
    axis = "left" if sideways else "forward"
    go, back = Move(name, t, **{axis: speed}), Move(name, t, **{axis: -speed})
    return [go, Move("rest", _REST), back, Move("rest", _REST)]


def _turns(rate: float, seconds: float) -> list[Move]:
    name = f"turn {rate:g} rad/s"
    return [Move(name, seconds, turn=rate), Move("rest", _REST), Move(name, seconds, turn=-rate), Move("rest", _REST)]


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

_BALL_ON = "Put the ball against the dribbler, robot facing +x with 1.5 m clear ahead"


def _kick(i: int, n: int) -> list[Move]:
    return [
        Move(f"place ball for kick {i}/{n}", 0.0, prompt=_BALL_ON),
        Move("dribble still", 1.0, dribble=True),
        Move(f"kick {i}/{n}", 0.3, dribble=True, kick=True),  # one kick: the kicker's cooldown is 1 s
        Move("rest", 2.0),
    ]


def _dribble(name: str, seconds: float, **axis: float) -> list[Move]:
    return [Move(name, seconds, dribble=True, **axis), Move("dribble still", 1.0, dribble=True)]


# About 45 s plus the operator's set-ups: how the ball sits on the dribbler and how the sensor reads
# it at rest, turning and driving with it, then 3 kicks for the ball's speed and direction.
BALL_SCRIPT: tuple[Move, ...] = (
    Move("place ball", 0.0, prompt=_BALL_ON),
    Move("dribble still", 3.0, dribble=True),
    *_dribble("dribble turn 1 rad/s", 3.0, turn=1.0),
    *_dribble("dribble turn -1 rad/s", 3.0, turn=-1.0),
    *_dribble("dribble forward 0.3 m/s", 1.5, forward=0.3),
    *_dribble("dribble backward 0.3 m/s", 1.5, forward=-0.3),
    *(m for i in range(1, 4) for m in _kick(i, 3)),
)

_START = "Robot near the middle of the camera's view facing +x, 0.8 m clear ahead and 0.6 m to its left, a ball at rest in view"
# Motion, then the ball: the whole profile of one robot, run by `examples/calibrate_robot.py`.
FULL_SCRIPT: tuple[Move, ...] = (Move("start", 0.0, prompt=_START), *CALIBRATION_SCRIPT, *BALL_SCRIPT)


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

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: None
    ) -> tuple[dict[RobotId, RobotCommand], None]:
        still = RobotCommand(0.0, 0.0, 0.0, kick=False, chip=False, dribble=False)
        commands = {rid: still for rid in robot_ids}
        if not robot_ids:
            return commands, mem
        dt = 0.0 if self._last_ts is None else game.ts - self._last_ts
        self._last_ts = game.ts
        if dt > _GAP_S:
            self._step_elapsed = 0.0  # resumed after a pause: this step again, from its start
        else:
            self._step_elapsed += dt
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
        if not self.done and self.script[self._step].prompt is None:
            m = self.script[self._step]
            commands[robot_ids[0]] = RobotCommand(m.forward, m.left, m.turn, kick=m.kick, chip=False, dribble=m.dribble)
        return commands, mem
