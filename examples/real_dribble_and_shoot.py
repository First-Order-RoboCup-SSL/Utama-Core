# ruff: noqa: E402
"""examples/real_dribble_and_shoot.py — one robot dribbles a lap of the field, then scores.

Run:
    pixi run python examples/real_dribble_and_shoot.py            # real robot
    pixi run python examples/real_dribble_and_shoot.py --mode rsim
    # the dashboard is at http://localhost:8080: press Force Start to play, Halt to freeze

Everything else is `real_score_goal.py`: the robot set there (`SHOOTER`), the field and camera
view from `real_attacker_defender.py`, HALT at start, the field check. The robot fetches the
ball and dribbles it corner to corner round a rectangle at 55% of the field (`DribbleTactic`,
letting go and taking the ball back every 0.8 m, inside the 1 m dribbling rule), and after
`LAPS` laps shoots at the goal at +x (`LeadAndSupportTactic`). For another go: Halt, put the
ball back, and Run in the Live tab's Strategy panel restarts the lap.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import dataclass, field

from real_score_goal import build_runner

from utama_core.engine.tactic import BaseTactic, TacticTag
from utama_core.tactics.dribble import DribbleMem, DribbleTactic
from utama_core.tactics.lead_and_support import LeadAndSupportMem, LeadAndSupportTactic

LAPS = 1


@dataclass
class DribbleThenShootMem:
    dribble: DribbleMem
    shoot: LeadAndSupportMem = field(default_factory=LeadAndSupportMem)


class DribbleThenShootTactic(BaseTactic[DribbleThenShootMem]):
    """`DribbleTactic` until it has reached `LAPS` x 4 corners, then `LeadAndSupportTactic`."""

    tag = TacticTag.ATTACK

    def __init__(self):
        self._dribble, self._shoot = DribbleTactic(), LeadAndSupportTactic()

    def initial_mem(self) -> DribbleThenShootMem:
        return DribbleThenShootMem(dribble=self._dribble.initial_mem())

    def _shooting(self, mem: DribbleThenShootMem) -> bool:
        return mem.dribble.corner_index >= 4 * LAPS

    def is_committed(self, game, mem: DribbleThenShootMem) -> bool:
        return self._shooting(mem) and self._shoot.is_committed(game, mem.shoot)

    def tick(self, game, ctx, robot_ids, mem: DribbleThenShootMem):
        if self._shooting(mem):
            commands, mem.shoot = self._shoot.tick(game, ctx, robot_ids, mem.shoot)
        else:
            commands, mem.dribble = self._dribble.tick(game, ctx, robot_ids, mem.dribble)
        return commands, mem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("real", "rsim", "grsim"), default="real")
    tactics = {"dribble_then_shoot": DribbleThenShootTactic, "dribble": DribbleTactic, "shoot": LeadAndSupportTactic}
    build_runner(parser.parse_args().mode, tactics, name="dribble_and_shoot").run()


if __name__ == "__main__":
    main()
