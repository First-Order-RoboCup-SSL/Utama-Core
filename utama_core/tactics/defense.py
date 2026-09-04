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

Second exception, same family: once that retriever actually reaches the
ball, `ball_is_loose` flips False on the very next tick (a friendly robot
now `has_ball`), so `retriever_id` resets to `None` and the ball-carrying
robot falls straight back into `defend_parameter` — pure shot-shadow
positioning with no ball awareness at all — and just drags the ball along
with it for the rest of the match. Found live (roadmap item 15,
2026-09-04): a `low_block` defender sliding along its own back line with
the ball glued to its dribbler, a permanent `COMMITTED_FROZEN` liveness
stall. Fixed by tracking `carrier_id` (the last tick's retriever, if it
now genuinely has the ball) across ticks and, while set, driving that
robot through the same chase-can't-happen/aim/kick clearance sequence
`ClearBallTactic` uses (reusing its shared geometry helpers, not its
lane-scoring — this tactic stays the deliberately simple baseline it was
designed to be, so it aims at a single fixed upfield-and-away-from-goal
target rather than scoring multiple candidate lanes).

Only `carrier_id` is cross-tick state now — `defend_parameter` still
recomputes its own target from scratch every tick, including the
2-defender side-selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.shared.pass_and_score_geometry import (
    ball_in_own_defense_area,
    ball_is_loose,
    has_ball,
    oriented_towards,
    own_defense_area_exit_point,
)
from utama_core.skills.src.defend_parameter import defend_parameter
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.skills.src.utils.move_utils import kick, turn_on_spot

_LOOSE_BALL_CLAIM_RANGE = 1.5  # metres — matches ball_is_loose's own contest range
# How far upfield (away from our own goal) the clearance aims, along the
# ball's current y — simpler than `ClearBallTactic`'s multi-lane scoring,
# appropriate for this tactic's "minimal baseline" role (see class docstring).
_CLEAR_DISTANCE = 4.5


@dataclass
class DefenseMem:
    """`shadow_ids`/`retriever_id` are last-tick's role split, kept only so
    `highlights()` (called with the *previous* tick's mem, before `tick()`
    runs again) has something to report — `defend_parameter` itself still
    recomputes its target every tick from scratch.

    `carrier_id` is real cross-tick decision state: once a retriever
    actually gets the ball, it stays the clearer across ticks (aim/kick is
    itself a multi-tick sequence) even though `retriever_id` immediately
    resets to `None` next tick (`ball_is_loose` goes False the instant a
    friendly robot has it — see module docstring's second exception)."""

    shadow_ids: tuple[RobotId, ...] = ()
    retriever_id: Optional[RobotId] = None
    carrier_id: Optional[RobotId] = None


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

    def highlights(self, mem: DefenseMem) -> dict[RobotId, str]:
        return {rid: "shadow" for rid in mem.shadow_ids} | (
            {mem.retriever_id: "retriever"} if mem.retriever_id is not None else {}
        )

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: DefenseMem
    ) -> tuple[dict[RobotId, RobotCommand], DefenseMem]:
        # A robot already mid-clearance stays the carrier regardless of
        # ball_is_loose/robot_ids membership this tick, until it actually
        # fires the kick (see module docstring's second exception) or is no
        # longer this tactic's robot / no longer actually has the ball.
        carrier_id = mem.carrier_id
        if carrier_id is not None and (carrier_id not in robot_ids or not has_ball(game, carrier_id)):
            carrier_id = None

        # A fresh carrier: some assigned robot already has the ball this
        # tick (e.g. last tick's retriever, whose own approach just landed
        # it). Checked directly against `has_ball`, not `ball_is_loose` +
        # `retriever_id` — the instant a robot's IR sensor goes True,
        # `ball_is_loose` is *already* False (any friendly `has_ball` makes
        # it so, same tick), so a retriever-gated check can never actually
        # observe the acquisition; it would only ever see the ball already
        # gone.
        if carrier_id is None:
            for rid in robot_ids:
                if has_ball(game, rid):
                    carrier_id = rid
                    break

        retriever_id = None
        if carrier_id is None and ball_is_loose(game):
            ball_pos = game.ball.p.to_2d()
            nearest_id = min(robot_ids, key=lambda rid: game.friendly_robots[rid].p.distance_to(ball_pos))
            if game.friendly_robots[nearest_id].p.distance_to(ball_pos) <= _LOOSE_BALL_CLAIM_RANGE:
                retriever_id = nearest_id

        active_id = carrier_id if carrier_id is not None else retriever_id
        mem.shadow_ids = tuple(rid for rid in robot_ids if rid != active_id)
        mem.retriever_id = retriever_id
        mem.carrier_id = carrier_id

        if ctx.match_log is not None:
            goal_x = game.field.my_goal_line[0][0]
            ctx.match_log.trace_if_changed(
                tick=0,
                sim_time=getattr(game, "ts", 0.0),
                key="defense.shadow_post",
                value={"x": goal_x, "y": game.ball.p.to_2d().y},
            )

        commands: dict[RobotId, RobotCommand] = {}
        for robot_id in robot_ids:
            if robot_id == carrier_id:
                commands[robot_id] = self._clear(game, ctx, robot_id)
            elif robot_id == retriever_id:
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

    def _clear(self, game: Game, ctx: TickContext, robot_id: RobotId) -> RobotCommand:
        """Aim upfield, away from our own goal, along the ball's current y,
        and kick — same chase-can't-happen/aim/kick shape as
        `ClearBallTactic`, minus its multi-lane scoring (see class
        docstring). `robot_id` is only ever called here once it already has
        the ball (`carrier_id`'s definition in `tick()`), so there is no
        chase branch."""
        ball_pos = game.ball.p.to_2d()
        attack_dir = -1.0 if game.my_team_is_right else 1.0
        half_length = game.field.half_length
        target_x = max(-half_length + 0.5, min(half_length - 0.5, ball_pos.x + attack_dir * _CLEAR_DISTANCE))
        target = Vector2D(target_x, ball_pos.y)
        target_oren = game.friendly_robots[robot_id].p.angle_to(target)
        if oriented_towards(game, robot_id, target_oren) and has_ball(game, robot_id):
            return kick()
        return turn_on_spot(
            game=game,
            motion_controller=ctx.motion_controller,
            robot_id=robot_id,
            target_oren=target_oren,
            dribbling=True,
        )
