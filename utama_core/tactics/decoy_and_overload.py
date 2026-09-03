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
from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.object import ObjectType, TeamType
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
_FINISH_TIMEOUT_TIME = 12.0  # seconds — same pass+shoot budget as pass_and_shoot._PHASE_TIMEOUT_TIME
_FINISH_TIMEOUT_TICKS = round(_FINISH_TIMEOUT_TIME * CONTROL_FREQUENCY)
_LURE_TOUCHLINE_MARGIN = 0.5  # metres in from the touchline — how close the decoy's lure run goes
_OVERLOAD_STANDOFF = 0.4  # metres — how far past the marker's original shadow the overloader sits
_LOOSE_BALL_SPEED = 0.3  # m/s — matches ball_is_loose's own threshold; see _teammate_already_has_ball


def _nearest_marker(game: Game, decoy_id: int) -> Optional[int]:
    decoy_pos = game.friendly_robots[decoy_id].p
    if not game.enemy_robots:
        return None
    return min(game.enemy_robots, key=lambda eid: game.enemy_robots[eid].p.distance_to(decoy_pos))


def _teammate_already_has_ball(game: Game, excluding_id: int) -> bool:
    """True when the ball already belongs to our own team's play and isn't
    this tactic's problem to fetch: either a FRIENDLY robot other than
    `excluding_id` currently has it, or it's moving fast enough that it's
    almost certainly a pass in flight between two other teammates rather
    than a genuinely abandoned ball (mirrors `ball_is_loose`'s own
    `_LOOSE_BALL_SPEED` reasoning, but deliberately doesn't reuse that
    function itself -- its enemy-contest-range logic answers "should a
    DEFENSIVE tactic break formation to retrieve this," a different
    question from "is a robot in a completely different ATTACK-tagged
    tactic already using this ball").

    `has_ball(game, mem.decoy_id)` (used elsewhere in this file) only ever
    answers "does the decoy itself have it" -- it has no way to notice a
    different teammate, in a different tactic slot, already holding or
    mid-pass with it. `KernelSchedulerStrategy` can legitimately run this
    tactic's "overload" allocation concurrently with `GiveAndGoTactic`'s
    "attack" slot on a different robot subset (both ATTACK-tagged, both
    eligible when we have the ball), so nothing upstream of this tactic
    guarantees its own decoy is the only robot on the team that might go
    fetch the ball. Found live, 2026-09-03: the decoy (nearest-to-ball of
    this tactic's own two assigned robots, picked with no knowledge of the
    rest of the team) drove straight at a ball a `GiveAndGoTactic` carrier
    had already collected one tick earlier, physically colliding with it --
    a same-team scrum, both robots then reading has_ball=True while the
    ball itself went nowhere. The possession-only version of this check
    still let a second scrum through mid-pass (the ball is legitimately
    held by nobody for the handful of ticks it's in flight between a
    passer and receiver), hence the added speed check.
    """
    with_ball = game.robot_with_ball
    if (
        with_ball is not None
        and with_ball.team_type == TeamType.FRIENDLY
        and with_ball.object_type == ObjectType.ROBOT
        and with_ball.id != excluding_id
    ):
        return True
    ball_speed = (game.ball.v.x**2 + game.ball.v.y**2) ** 0.5
    return ball_speed >= _LOOSE_BALL_SPEED


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
    finish_ticks: int = 0
    goal_scored: bool = False
    prev_best_shot_y: Optional[float] = None  # feeds _score_goal's switch-margin hysteresis; see _pass_and_score.py


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

    def highlights(self, mem: DecoyOverloadMem) -> dict[RobotId, str]:
        highlights: dict[RobotId, str] = {}
        if mem.decoy_id is not None:
            highlights[mem.decoy_id] = "decoy"
        if mem.overloader_id is not None:
            highlights[mem.overloader_id] = "overload"
        return highlights

    def is_committed(self, game: Game, mem: DecoyOverloadMem) -> bool:
        # Once roles are locked and the lure/finish sequence has started,
        # reassigning robots mid-sequence would strand a decoy run or a pass
        # in flight — same protection window every other attack tactic here
        # gives its in-progress action.
        return mem.decoy_id is not None and not mem.goal_scored

    def suggest_next(self, game: Game, mem: DecoyOverloadMem) -> Optional[TacticId]:
        """Purely advisory (see `Tactic.suggest_next`'s contract) —
        `is_committed()` already releases this tactic once `goal_scored` is
        True, so a `TacticGraph` would reassign it regardless via normal
        eviction-and-fallback-to-start. This exists only so a graph can hand
        off to a specific next pattern (give-and-go, to build the next
        possession up rather than immediately re-luring) instead of whatever
        the graph's fallback happens to pick. No existing strategy calls
        this (nothing consulted `suggest_next` anywhere until
        `strategy/tactic_graph.py`), so this has no effect on any
        already-tuned `build_*_kernel_strategy` config.
        """
        if mem.goal_scored:
            return "givego"
        return None

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: DecoyOverloadMem
    ) -> tuple[dict[RobotId, RobotCommand], DecoyOverloadMem]:
        if len(robot_ids) < 2:
            # No marker to drag *and* nobody to feed — degrade to a plain
            # ball chase rather than crash; a single-robot allocation to an
            # ATTACK-tagged, two-role tactic is a Partitioner misconfiguration
            # (see `_fixed_ratio_picker`'s `min_attack` handling of the same
            # class of problem for `PassAndShootTactic`), not something
            # this tactic should silently invent a role split for.
            robot_id = robot_ids[0]
            if _teammate_already_has_ball(game, excluding_id=robot_id):
                # A different tactic's carrier already has it -- see
                # `_teammate_already_has_ball`'s docstring. Hold instead of
                # chasing a ball that's already legally possessed.
                return {
                    robot_id: go_to_point(
                        game=game,
                        motion_controller=ctx.motion_controller,
                        robot_id=robot_id,
                        target_coords=_support_hold_point(game, robot_id, 0),
                    )
                }, mem
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
            if not has_ball(game, mem.decoy_id) and _teammate_already_has_ball(game, excluding_id=mem.decoy_id):
                # A different tactic's carrier already has it -- see
                # `_teammate_already_has_ball`'s docstring. Hold at the
                # support point rather than driving into an already-claimed
                # ball; the picker will re-evaluate this tactic's allocation
                # next time it's not commitment-pinned.
                commands[mem.decoy_id] = go_to_point(
                    game=game,
                    motion_controller=ctx.motion_controller,
                    robot_id=mem.decoy_id,
                    target_coords=_support_hold_point(game, mem.decoy_id, 0),
                )
            elif not has_ball(game, mem.decoy_id):
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
                if ctx.match_log is not None:
                    ctx.match_log.trace_if_changed(
                        tick=0,
                        sim_time=getattr(game, "ts", 0.0),
                        key="decoy_and_overload.lure_target",
                        value={"x": target.x, "y": target.y},
                    )

            overload_target = _overload_target(game, mem.marker_start_y)
            commands[mem.overloader_id] = go_to_point(
                game=game,
                motion_controller=ctx.motion_controller,
                robot_id=mem.overloader_id,
                target_coords=overload_target,
            )
            if ctx.match_log is not None:
                ctx.match_log.trace_if_changed(
                    tick=0,
                    sim_time=getattr(game, "ts", 0.0),
                    key="decoy_and_overload.overload_target",
                    value={"x": overload_target.x, "y": overload_target.y},
                )

            mem.lure_ticks += 1
            dragged = False
            if mem.marker_id is not None and mem.marker_id in game.enemy_robots:
                marker_now_y = game.enemy_robots[mem.marker_id].p.y
                dragged = abs(marker_now_y - mem.marker_start_y) >= _LURE_DRAG_THRESHOLD
            # Only advance to "finish" once the decoy has actually collected
            # the ball -- that phase immediately treats `mem.decoy_id` as the
            # PASSER in `_pass_exec`, which chases the ball itself
            # (go_to_ball) the instant it doesn't already have it, with no
            # awareness that a different tactic's carrier might already be
            # using it (see `_teammate_already_has_ball`'s docstring for the
            # live-found scrum this caused). A lure that timed out
            # (`_LURE_MAX_TICKS`) without ever fetching the ball -- because a
            # teammate elsewhere already had/was passing it -- has nothing
            # to hand off; keep holding instead of transitioning into a
            # phase that assumes otherwise.
            if (dragged or mem.lure_ticks >= _LURE_MAX_TICKS) and has_ball(game, mem.decoy_id):
                mem.phase = "finish"

            return commands, mem

        # phase == "finish": decoy shoots if its own lane is now open,
        # otherwise passes to the overloader sitting in the vacated lane.
        # Found live 2026-09-02 (full-length tournament,
        # clear_press_plus_vs_shadow_switch_LK.pkl, t=205.8s): a referee
        # restart (STOP -> FORCE_START, ball placed far from both decoy and
        # overloader) can strand this phase with the ball nowhere near
        # either robot and no path back to "setup" -- is_committed() only
        # releases on goal_scored, so with no goal ever scored this held the
        # slot's two robots for the remaining ~394s of the match while a
        # different tactic's robot converged on the same displaced ball with
        # no cross-tactic awareness of the other, and both stalled at
        # FastPathPlanner.OBSTACLE_CLEARANCE apart. Same stalled-phase-with-
        # no-timeout bug class pass_and_shoot.py's own _PHASE_TIMEOUT_TICKS
        # was added to fix; apply the identical budget here.
        mem.finish_ticks += 1
        if mem.finish_ticks > _FINISH_TIMEOUT_TICKS:
            return {}, DecoyOverloadMem()

        if has_ball(game, mem.decoy_id, visual=True) and _decoy_shot_open(game, mem.decoy_id):
            shot_cmd, scored, mem.prev_best_shot_y = _score_goal(game, ctx, mem.decoy_id, mem.prev_best_shot_y)
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

        pass_cmds, pass_complete, _lane_blocked = _pass_exec(game, ctx, mem.decoy_id, mem.overloader_id)
        commands.update(pass_cmds)
        if pass_complete:
            shot_cmd, scored, mem.prev_best_shot_y = _score_goal(game, ctx, mem.overloader_id, mem.prev_best_shot_y)
            commands[mem.overloader_id] = shot_cmd
            mem.goal_scored = scored

        return commands, mem
