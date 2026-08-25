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

Exception: a loose ball with no enemy contesting it (see `ball_is_loose`)
never gets shadowed at all — shadowing only ever reacts to the ball's
current position, so an abandoned ball just sits there forever with a
defender parked a shot-shadow's distance away. Found live: a `clear_danger`
vs `low_block` match pinned 28 straight seconds this way, ball dead in a
corner near `low_block`'s own goal line, three defenders standing
1.1-1.5m away all still shadowing a shot no one was taking. The nearest
assigned defender breaks off to fetch it instead (see `tick()` below); the
rest keep shadowing normally via `defend_parameter`.

No other cross-tick state: `defend_parameter` recomputes its target from
scratch every tick, including the 2-defender side-selection, so `mem` is
empty.
"""

from __future__ import annotations

from dataclasses import dataclass

from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.shared.pass_and_score_geometry import (
    ball_in_own_defense_area,
    ball_is_loose,
    own_defense_area_exit_point,
)
from utama_core.skills.src.defend_parameter import defend_parameter
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.skills.src.go_to_point import go_to_point

_LOOSE_BALL_CLAIM_RANGE = 1.5  # metres — matches ball_is_loose's own contest range


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
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: DefenseMem
    ) -> tuple[dict[RobotId, RobotCommand], DefenseMem]:
        retriever_id = None
        if ball_is_loose(game):
            ball_pos = game.ball.p.to_2d()
            nearest_id = min(robot_ids, key=lambda rid: game.friendly_robots[rid].p.distance_to(ball_pos))
            if game.friendly_robots[nearest_id].p.distance_to(ball_pos) <= _LOOSE_BALL_CLAIM_RANGE:
                retriever_id = nearest_id

        commands: dict[RobotId, RobotCommand] = {}
        for robot_id in robot_ids:
            if robot_id == retriever_id:
                if ball_in_own_defense_area(game):
                    # Only the keeper may enter our own box (DefenseAreaRule)
                    # — hold the nearest legal edge instead; goalkeep.py's own
                    # retrieval branch (see that module) is what actually
                    # fetches a loose ball once it's this deep.
                    commands[robot_id] = go_to_point(
                        game=game,
                        motion_controller=ctx.motion_controller,
                        robot_id=robot_id,
                        target_coords=own_defense_area_exit_point(game, game.ball.p.to_2d().y),
                    )
                else:
                    commands[robot_id] = go_to_ball(
                        game=game, motion_controller=ctx.motion_controller, robot_id=robot_id, ctx=ctx
                    )
            else:
                commands[robot_id] = defend_parameter(game, ctx.motion_controller, robot_id, defender_group=robot_ids)
        return commands, mem
