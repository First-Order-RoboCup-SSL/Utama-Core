"""Defense tactic — shadows the ball-to-goal shot line with 1-2 outfield defenders.

Ported from `utama_strategy.examples.defense_strategy.DefenceStrategy`. That
BT strategy was almost entirely role-assignment plumbing (`SetRoles` pinning
robot 4 as goalkeeper, `execute_default_action` dispatching by `Role`) around
a single already-self-contained skill function,
`utama_core.skills.src.defend_parameter.defend_parameter` — the actual
positioning logic (choosing which post to cover, shadowing the shot line,
picking sides dynamically for a 2-defender team) already lives in Core and
needed no porting itself. This tactic is that same dispatch, expressed as a
`tick()` instead of `execute_default_action`.

No cross-tick state: `defend_parameter` recomputes its target from scratch
every tick, including the 2-defender side-selection, so `mem` is empty.
"""

from __future__ import annotations

from dataclasses import dataclass

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.kernel.context import KernelContext
from utama_core.kernel.tactic import BaseTactic, RobotId, TacticTag
from utama_core.skills.src.defend_parameter import defend_parameter


@dataclass
class DefenseMem:
    """No cross-tick state: `defend_parameter` recomputes its target every tick."""


class DefenseTactic(BaseTactic[DefenseMem]):
    """One or two robots, shadowing the ball-to-goal shot line.

    Passes its own `robot_ids` to `defend_parameter` as `defender_group`, so
    the dynamic 2-defender side-selection triggers on how many robots *this
    tactic* was handed, not on the whole team's robot count — and the
    near/far-post parity fallback is keyed off position within that group,
    not the global `robot_id == 1` convention. Fixes a real bug where a team
    with more than 2 outfield robots (e.g. 2 defenders + 3 attackers
    elsewhere) could assign both defenders to the same post: neither one's
    `robot_id` needed to be 1, so both hit the `else` branch and picked the
    same side, ending up on top of each other and tripping the "too many
    defenders in own area" foul.
    """

    tag = TacticTag.DEFENSE

    def initial_mem(self) -> DefenseMem:
        return DefenseMem()

    def tick(
        self, game: Game, ctx: KernelContext, robot_ids: tuple[RobotId, ...], mem: DefenseMem
    ) -> tuple[dict[RobotId, RobotCommand], DefenseMem]:
        commands = {
            robot_id: defend_parameter(game, ctx.motion_controller, robot_id, defender_group=robot_ids)
            for robot_id in robot_ids
        }
        return commands, mem
