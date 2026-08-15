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
every tick, including the 2-defender side-selection (it reads
`len(game.friendly_robots)` and `robot_id` directly), so `mem` is empty.
"""

from __future__ import annotations

from dataclasses import dataclass

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.kernel.context import KernelContext
from utama_core.kernel.tactic import BaseTactic, RobotId
from utama_core.skills.src.defend_parameter import defend_parameter


@dataclass
class DefenseMem:
    """No cross-tick state: `defend_parameter` recomputes its target every tick."""


class DefenseTactic(BaseTactic[DefenseMem]):
    """One or two robots, shadowing the ball-to-goal shot line.

    Carried over as-is from the source: `defend_parameter`'s dynamic
    2-defender side-selection triggers on `len(game.friendly_robots) == 2`
    (whole-team robot count), not on how many robots this tactic was handed.
    That was correct for the original dedicated 2-robot defense strategy it
    came from; if this tactic is ever run with 2 defenders on a team with
    more than 2 robots total (e.g. 2 defenders + 1 attacker elsewhere), each
    defender falls back to the fixed near-post-by-parity assignment instead
    of the dynamic side choice. Not fixed here since it wasn't a forcing
    case yet — flagging so it isn't mistaken for new behaviour.
    """

    def initial_mem(self) -> DefenseMem:
        return DefenseMem()

    def tick(
        self, game: Game, ctx: KernelContext, robot_ids: tuple[RobotId, ...], mem: DefenseMem
    ) -> tuple[dict[RobotId, RobotCommand], DefenseMem]:
        commands = {robot_id: defend_parameter(game, ctx.motion_controller, robot_id) for robot_id in robot_ids}
        return commands, mem
