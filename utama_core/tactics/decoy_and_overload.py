"""Decoy-and-overload attack tactic — deliberate misdirection, not a straight-on drive.

New tactical logic, not ported from Utama-Strategy and not a variation on an
existing Core tactic (see `docs/roadmap.md`'s "More tactics" entry, which
asks for genuinely new football vocabulary rather than growing the catalog
by variations on what's already there). Every attacking tactic so far drives
straight at the danger: `PassAndShootTactic` sets up a fixed pass-then-shoot,
`LeadAndSupportTactic`'s leader dribbles toward goal and looks for a lane,
`GiveAndGoTactic` cycles hops but each hop is still "whoever is open now,
shoot or pass." None of them deliberately manufacture space by first *luring*
a defender out of it. Real football uses exactly this pattern: a decoy run
into a low-value area drags a marker with it, and the space that marker
vacates is what actually gets attacked — either by the decoy's own shot once
the lane clears behind the marker, or by a teammate who runs into the space
the marker just left.

Two roles, both re-evaluated only outside a commitment window (mirrors
`is_committed()` in every other attack tactic here — see `pass_and_shoot`'s
docstring for the original rationale): a **decoy** (the ball carrier) and an
**overloader** (the teammate who exploits the vacated space). Phases:

1. `"lure"` — decoy dribbles toward the near touchline on whichever flank its
   own nearest marker is currently covering, deliberately *away* from goal.
   This is the "lure" itself: the decoy's own nearest defender must follow
   (an SSL defender that stays goal-side of a ball carrier heading toward
   the touchline concedes the wide space it just left uncontested), tracked
   via `_marker_dragged` comparing the marker's distance from the central
   shot lane at lure-start vs now. The overloader simultaneously drifts into
   the central lane the decoy vacated by leaving it.
2. `"finish"` — once the marker is confirmed dragged (or a lure-duration cap
   is hit, so a marker that refuses to bite doesn't stall the tactic
   forever), the decoy re-evaluates: shoot immediately if its own lane to
   goal is now clear (the marker chased it out wide, so nobody is left
   centrally to block *this* robot specifically — checked the same way
   `GiveAndGoTactic` checks a hop-ending shot), otherwise pass to the
   overloader, who is already sitting in the space the marker gave up, and
   `_score_goal` from there.

Reuses `_pass_and_score`'s `_pass_exec`/`_score_goal` for the mechanics of
the pass/shot (generic two-robot ball transfer and shot-taking, not specific
to any one tactic's phase sequence — same reuse rationale `GiveAndGoTactic`
gives), and `go_to_point`/`shared.pass_and_score_geometry` for movement and
shot-lane checks. What's new is only the lure decision and the dragged-marker
detection layered on top.

Two-robot minimum (a decoy needs a marker to drag and a beneficiary to feed);
any robots beyond the two assigned just hold at `LeadAndSupportTactic`-style
support points rather than inventing a third role no design forcing case has
asked for yet (per the minimalism discipline in `AGENTS.md`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utama_core.config.settings import CONTROL_FREQUENCY
from utama_core.engine.context import KernelContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.object import TeamType
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.shared.pass_and_score_geometry import (
    clamp_outside_enemy_defense_area,
    enemy_goal_line,
    enemy_positions,
    find_best_shot,
    has_ball,
    segment_blocked,
)
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.tactics._pass_and_score import _pass_exec, _score_goal

_LURE_DRAG_THRESHOLD = 0.8  # metres — marker must be pulled at least this far off the central shot lane's y
_LURE_MAX_TIME = 1.5  # seconds — cap so a marker that doesn't bite can't stall the tactic forever
_LURE_MAX_TICKS = round(_LURE_MAX_TIME * CONTROL_FREQUENCY)
_LURE_TOUCHLINE_MARGIN = 0.5  # metres in from the touchline — how close the decoy's lure run goes
_OVERLOAD_STANDOFF = 0.4  # metres — how far past the marker's original shadow the overloader sits


def _nearest_marker(game: Game, decoy_id: int) -> Optional[int]:
    decoy_pos = game.friendly_robots[decoy_id].p
    if not game.enemy_robots:
        return None
    return min(game.enemy_robots, key=lambda eid: game.enemy_robots[eid].p.distance_to(decoy_pos))


def _central_lane_y(game: Game) -> float:
    """The y the decoy's straight-on shot lane would use: the ball's current y,
    same reference point `_fallback_hold_target`-style tactics use for "where
    the action currently is," since there is no fixed shot target until
    `find_best_shot` is actually queried at finish time."""
    return game.ball.p.to_2d().y


def _lure_target(game: Game, decoy_id: int, marker_id: Optional[int]) -> Vector2D:
    """A point toward the near touchline, on whichever side the marker already
    sits — dribbling toward where the marker already leans is what makes the
    marker's follow a *cost* (abandoning central cover) rather than a free
    lateral shuffle."""
    decoy_pos = game.friendly_robots[decoy_id].p
    half_width = game.field.half_width
    marker_y = game.enemy_robots[marker_id].p.y if marker_id is not None else decoy_pos.y
    side = 1.0 if marker_y >= 0 else -1.0
    target_y = side * (half_width - _LURE_TOUCHLINE_MARGIN)
    # Advance toward goal only modestly while luring — the point is lateral
    # drag, not progress; going flat-out for the corner just runs out of
    # pitch without ever threatening the goal enough to justify a marker
    # abandoning it.
    goal_x = game.field.enemy_goal_line[0][0]
    target_x = decoy_pos.x + 0.4 * (goal_x - decoy_pos.x)
    # Recomputed from the decoy's *current* position every tick, this target
    # converges toward goal_x itself as the decoy chases it tick over tick —
    # nothing here ever stops short of the goal line, let alone the defense
    # area 1 m in front of it. This was the actual source of repeated
    # attacker-infringement `defense_area` fouls (confirmed via replay
    # inspection: the decoy sat inside the enemy box for 200+ consecutive
    # ticks in one match) — `_overload_target`'s clamp alone did not cover
    # this second forward-converging target in the same tactic.
    return clamp_outside_enemy_defense_area(game, Vector2D(target_x, target_y))


def _overload_target(game: Game, marker_start_y: float) -> Vector2D:
    """Where the overloader runs into: the central lane the marker vacated,
    slightly past the marker's original shadow position so the overloader
    is genuinely inside the space that opened, not just back where the
    decoy started."""
    goal_x, _goal_y1, _goal_y2 = enemy_goal_line(game)
    # Sit inside attacking-third depth, on the lane the dragged marker used
    # to cover.
    target_x = goal_x - (goal_x / abs(goal_x)) * 1.5 if goal_x != 0 else 0.0
    offset = _OVERLOAD_STANDOFF if marker_start_y >= 0 else -_OVERLOAD_STANDOFF
    target = Vector2D(target_x, marker_start_y - offset)
    # The nominal 1.5 m standoff only clears the enemy defense area's own
    # 1 m depth by 0.5 m — thin enough that controller overshoot chasing a
    # moving marker regularly lands the overloader inside the box, an
    # attacker-infringement foul (`DefenseAreaRule`'s `attacker_infringement`
    # default). Found via repeated `defense_area` fouls (20 in one match)
    # once this tactic ran against a live opponent that could actually drag
    # its marker deep. Clamp with the same margin every other tactic's
    # box-adjacent target uses.
    return clamp_outside_enemy_defense_area(game, target)


def _decoy_shot_open(game: Game, decoy_id: int) -> bool:
    decoy_pos = game.friendly_robots[decoy_id].p
    goal_x, goal_y1, goal_y2 = enemy_goal_line(game)
    best_shot_y, _gap = find_best_shot(decoy_pos, list(game.enemy_robots.values()), goal_x, goal_y1, goal_y2)
    if best_shot_y is None:
        return False
    return not segment_blocked(decoy_pos, Vector2D(goal_x, best_shot_y), enemy_positions(game))


@dataclass
class DecoyOverloadMem:
    phase: str = "lure"  # "lure" -> "finish" -> (goal_scored)
    decoy_id: Optional[int] = None
    overloader_id: Optional[int] = None
    marker_id: Optional[int] = None
    marker_start_y: Optional[float] = None
    lure_ticks: int = 0
    goal_scored: bool = False


def _support_hold_point(game: Game, robot_id: int, index: int) -> Vector2D:
    """Non-role-assigned extra robots hold in open space on the attacking
    half, spaced out by index — same spirit as `ShadowAndMarkTactic`'s
    fallback holders, just on the attacking side. No third role invented for
    these; see module docstring."""
    goal_x = game.field.enemy_goal_line[0][0]
    hold_x = goal_x * 0.4
    hold_y = (game.field.half_width * 0.5) * (1 if index % 2 == 0 else -1) * ((index // 2) + 1) * 0.5
    return Vector2D(hold_x, hold_y)


class DecoyOverloadTactic(BaseTactic[DecoyOverloadMem]):
    """2+ attackers: one drags a marker wide as a decoy, another overloads the
    central space that marker vacates. See module docstring for the full
    lure -> finish phase sequence and rationale.
    """

    tag = TacticTag.ATTACK

    def initial_mem(self) -> DecoyOverloadMem:
        return DecoyOverloadMem()

    def is_committed(self, game: Game, mem: DecoyOverloadMem) -> bool:
        # Once roles are locked and the lure/finish sequence has started,
        # reassigning robots mid-sequence would strand a decoy run or a pass
        # in flight — same protection window every other attack tactic here
        # gives its in-progress action.
        return mem.decoy_id is not None and not mem.goal_scored

    def tick(
        self, game: Game, ctx: KernelContext, robot_ids: tuple[RobotId, ...], mem: DecoyOverloadMem
    ) -> tuple[dict[RobotId, RobotCommand], DecoyOverloadMem]:
        if len(robot_ids) < 2:
            # No marker to drag *and* nobody to feed — degrade to a plain
            # ball chase rather than crash; a single-robot allocation to an
            # ATTACK-tagged, two-role tactic is a Partitioner misconfiguration
            # (see `_fixed_ratio_picker`'s `min_attack` handling of the same
            # class of problem for `PassAndShootTactic`), not something
            # this tactic should silently invent a role split for.
            robot_id = robot_ids[0]
            return {
                robot_id: go_to_ball(game=game, motion_controller=ctx.motion_controller, robot_id=robot_id, ctx=ctx)
            }, mem

        if mem.decoy_id is None or mem.decoy_id not in robot_ids or mem.overloader_id not in robot_ids:
            ordered = sorted(robot_ids, key=lambda rid: game.friendly_robots[rid].p.distance_to(game.ball.p.to_2d()))
            mem = DecoyOverloadMem(decoy_id=ordered[0], overloader_id=ordered[1])
            mem.marker_id = _nearest_marker(game, mem.decoy_id)
            mem.marker_start_y = (
                game.enemy_robots[mem.marker_id].p.y if mem.marker_id is not None else _central_lane_y(game)
            )

        commands: dict[RobotId, RobotCommand] = {}
        extra_ids = [rid for rid in robot_ids if rid not in (mem.decoy_id, mem.overloader_id)]
        for i, rid in enumerate(extra_ids):
            commands[rid] = go_to_point(
                game=game,
                motion_controller=ctx.motion_controller,
                robot_id=rid,
                target_coords=_support_hold_point(game, rid, i),
            )

        if mem.phase == "lure":
            if not has_ball(game, mem.decoy_id):
                commands[mem.decoy_id] = go_to_ball(
                    game=game, motion_controller=ctx.motion_controller, robot_id=mem.decoy_id, ctx=ctx
                )
            else:
                target = _lure_target(game, mem.decoy_id, mem.marker_id)
                commands[mem.decoy_id] = go_to_point(
                    game=game,
                    motion_controller=ctx.motion_controller,
                    robot_id=mem.decoy_id,
                    target_coords=target,
                    dribbling=True,
                )

            overload_target = _overload_target(game, mem.marker_start_y)
            commands[mem.overloader_id] = go_to_point(
                game=game,
                motion_controller=ctx.motion_controller,
                robot_id=mem.overloader_id,
                target_coords=overload_target,
            )

            mem.lure_ticks += 1
            dragged = False
            if mem.marker_id is not None and mem.marker_id in game.enemy_robots:
                marker_now_y = game.enemy_robots[mem.marker_id].p.y
                dragged = abs(marker_now_y - mem.marker_start_y) >= _LURE_DRAG_THRESHOLD
            if dragged or mem.lure_ticks >= _LURE_MAX_TICKS:
                mem.phase = "finish"

            return commands, mem

        # phase == "finish": decoy shoots if its own lane is now open,
        # otherwise passes to the overloader sitting in the vacated lane.
        if has_ball(game, mem.decoy_id, visual=True) and _decoy_shot_open(game, mem.decoy_id):
            shot_cmd, scored = _score_goal(game, ctx, mem.decoy_id)
            commands[mem.decoy_id] = shot_cmd
            commands.setdefault(
                mem.overloader_id,
                go_to_point(
                    game=game,
                    motion_controller=ctx.motion_controller,
                    robot_id=mem.overloader_id,
                    target_coords=_overload_target(game, mem.marker_start_y),
                ),
            )
            mem.goal_scored = scored
            return commands, mem

        pass_cmds, pass_complete = _pass_exec(game, ctx, mem.decoy_id, mem.overloader_id)
        commands.update(pass_cmds)
        if pass_complete:
            shot_cmd, scored = _score_goal(game, ctx, mem.overloader_id)
            commands[mem.overloader_id] = shot_cmd
            mem.goal_scored = scored

        return commands, mem
