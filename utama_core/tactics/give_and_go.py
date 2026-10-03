"""Give-and-go attack tactic — carrier passes and relocates, receiver becomes
the next carrier, repeat until a shot lane opens.

New tactical logic, not ported from Utama-Strategy. `PassAndShootTactic`
runs one pass then always shoots; `LeadAndSupportTactic` never passes at
all — the leader dribbles in alone while supports just hold space. Neither
captures the actual "wall pass" / one-two pattern: a carrier under pressure
passes to a moving teammate and immediately relocates to a *new* open
position to receive the ball straight back, cycling as many times as it
takes for a lane to open, rather than committing to a single scripted
setup -> pass -> score sequence. This tactic is that cycle, generalized past
two robots — any assigned robot can become "next receiver," chosen fresh
each hop by who currently offers the best `score_pass_setup` (the same
scoring function `pass_and_shoot`'s dynamic setup already uses, reused
here as a per-hop receiver choice instead of a one-time setup optimization).

Reuses `_pass_and_score`'s `_pass_exec` (aim, intercept-position the
receiver, kick, confirm catch) as-is for the mechanics of one hop — that
machinery is generic two-robot ball transfer, not specific to
`PassAndShootTactic`'s fixed-pair phase sequence, so it was imported
rather than re-derived. What is new here is the *decision* layered on top:
after each catch, the new carrier either shoots immediately (if
`segment_blocked` says its lane to goal is clear) or picks the best-scoring
teammate and passes again, while the just-passed-from robot relocates to a
fresh support point (via `LeadAndSupportTactic`'s support-scoring approach,
reused at k=1) instead of standing still waiting for a return pass that may
never come.

`is_committed()` covers the same window `pass_and_shoot` protects: once a
carrier has the ball and is aiming (or already selected a target
receiver for this hop), reassigning this tactic's robots mid-hop would
strand a pass in flight. Between hops (ball not yet caught, no receiver
locked) there is no commitment, matching the "setup" window in the older
tactic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utama_core.config.settings import CONTROL_FREQUENCY
from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.shared.pass_and_score_geometry import (
    ball_in_own_defense_area,
    ball_line_receive_point,
    carry_exhausted,
    carry_origin,
    clamp_outside_enemy_defense_area,
    clamp_outside_own_defense_area,
    enemy_goal_line,
    enemy_positions,
    find_best_shot,
    has_ball,
    in_own_defense_area,
    intercept_point,
    no_shot_reposition_target,
    oriented_towards,
    own_defense_area_exit_point,
    score_pass_setup,
    segment_blocked,
)
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.skills.src.utils.move_utils import kick, move, turn_on_spot
from utama_core.tactics._pass_and_score import _pass_exec

_MAX_HOPS_PER_POSSESSION = 6  # safety valve — force a shot attempt rather than passing forever
_SUGGEST_HANDOFF_TIME = 2.0  # seconds — see suggest_next below
_SUGGEST_HANDOFF_TICKS = round(_SUGGEST_HANDOFF_TIME * CONTROL_FREQUENCY)
# Safety valve for a single in-flight hop, mirroring _MAX_HOPS_PER_POSSESSION's
# role for the whole possession: _pass_exec's synchronized passer/receiver
# handshake (both must reach position + orientation before either kicks) has
# no timeout of its own. Found live (stuck-match investigation,
# 2026-08-26, docs/testing_gaps.md gap #11): a carrier trapped in an extreme
# field corner locked a receiver_id and then held the ball motionless for
# the rest of a 600s match — `is_committed()` returns True whenever
# receiver_id is not None, so nothing ever reassigns these robots either,
# and no custom_referee rule catches a legally-dribbling, barely-moving
# ball outside a defense area (keeper_held_ball only watches the defense
# area; excessive_dribbling only watches distance carried, not hold time).
# Abandoning a hop that's taken too long and forcing the fallback shoot/
# reposition path (same one used when no pass lane exists at all) breaks
# the deadlock regardless of which specific geometry caused the receiver to
# never arrive.
_MAX_HOP_TICKS = round(4.0 * CONTROL_FREQUENCY)  # 4s — generous vs. a real catch, short vs. a match
# An enemy walking onto the direct carrier-receiver line mid-hop makes the
# pass interceptable, but is a much more urgent signal than "receiver is
# slow to arrive" — bail well before _MAX_HOP_TICKS's 4s once it's clearly
# not a one-tick noise blip (an enemy grazing the clearance boundary for a
# single frame while running through, not actually planted in the lane).
_LANE_BLOCKED_ABANDON_TICKS = round(0.5 * CONTROL_FREQUENCY)
# Safety valve for the first-touch receiver *search* itself, distinct from
# _MAX_HOP_TICKS (which only bounds a single locked-in hop): in congested
# play, `_nearest_safe_receiver` can keep finding a fresh candidate every
# time the previous one gets lane-blocked and abandoned (_LANE_BLOCKED_
# ABANDON_TICKS, ~0.5s each) without a single hop ever completing --
# `hop_count` never leaves 0, so `_MAX_HOPS_PER_POSSESSION`'s force_shot
# never triggers either. Found live (tournament stall investigation,
# 2026-09-01): 10 distinct receiver-lock attempts over 12.5s, hop_count
# stuck at 0 throughout. `ticks_held` already counts ticks since this exact
# first-touch situation began (reset only where mem itself resets, never by
# a completed hop while hop_count is still 0) making it the right budget to
# reuse here. Deliberately routes to the no-open-lane dribble/reposition
# fallback, never `kick()`: DoubleTouchRule only re-arms right after a real
# restart, but GiveAndGoTactic can't see referee state here, so a stuck
# first-touch situation must still be treated as though a solo shot might
# be an illegal second touch on the ball.
_MAX_FIRST_TOUCH_TICKS = round(8.0 * CONTROL_FREQUENCY)  # 8s — generous vs. real congested play
# `first_touch_stuck` above is a *caution*, not a life sentence: it exists so
# a solo shot can't be an illegal second touch on a restart kick, but
# DoubleTouchRule's own window closes within a couple of seconds of
# NORMAL_START in every real case (any other robot's touch, or leaving
# NORMAL_START entirely, disarms it immediately — see double_touch_rule.py).
# `hop_count` staying stuck at 0 for an entire possession (the same
# already-documented gap noted above) means `first_touch_stuck` can now
# persist far past any restart window that could possibly still be live —
# found live (tournament stall investigation, 2026-09-03): a carrier held a
# wide-open shot lane for 50+ seconds and strafed the whole time instead of
# shooting, because `force_shot` only checks `hop_count`, which never moves.
# Once stuck for this much longer, any restart-kick window has certainly
# closed regardless of match congestion, so treat it the same as
# `_MAX_HOPS_PER_POSSESSION` and force the shot rather than repositioning
# forever.
_FIRST_TOUCH_FORCE_SHOT_TICKS = round(15.0 * CONTROL_FREQUENCY)  # 15s — well past any real restart window
_RELOCATE_MIN_SEPARATION = 0.9  # metres — a relocating support point must clear the carrier and other supports
# Retreat standoff from our own area front edge while the ball is in our own
# half: support robots hold this far off the box line instead of packing it.
_RELOCATE_BOX_RETREAT = 1.0


def _best_receiver(game: Game, carrier_id: int, candidate_ids: tuple[int, ...]) -> Optional[int]:
    """Highest-`score_pass_setup` teammate, using each candidate's live position (no repositioning)."""
    carrier_pos = game.friendly_robots[carrier_id].p
    best_id, best_score = None, None
    for candidate_id in candidate_ids:
        candidate_pos = game.friendly_robots[candidate_id].p
        result = score_pass_setup(game, carrier_pos, candidate_pos)
        if result is None:
            continue
        if best_score is None or result.score > best_score:
            best_id, best_score = candidate_id, result.score
    return best_id


_MIN_SAFE_PASS_DISTANCE = 0.7  # metres — matches score_pass_setup's own min_pass_distance


def _nearest_safe_receiver(game: Game, carrier_id: int, candidate_ids: tuple[int, ...]) -> Optional[int]:
    """Nearest teammate with a clear passing lane — no scoring-setup bar.

    `_best_receiver`/`score_pass_setup` require the *receiver* to already
    have an open shot from its current position — a fine bar for choosing
    the best of several plausible receivers mid-possession, but far too
    strict for the very first touch of a possession (kickoff, restart):
    right then, every teammate is typically still in its kickoff/defensive
    formation spot, nowhere near a scoring position, so `_best_receiver`
    legitimately returns None every tick. This picks *any* reachable
    teammate with an unblocked lane instead, purely to get a second robot's
    touch on the ball before the carrier could possibly touch it again
    (SSL's double-touch rule) — not to set up a good shot.
    """
    carrier_pos = game.friendly_robots[carrier_id].p
    enemies = enemy_positions(game)
    best_id, best_dist = None, None
    for candidate_id in candidate_ids:
        candidate_pos = game.friendly_robots[candidate_id].p
        dist = carrier_pos.distance_to(candidate_pos)
        if dist < _MIN_SAFE_PASS_DISTANCE:
            continue
        if segment_blocked(carrier_pos, candidate_pos, enemies):
            continue
        if best_dist is None or dist < best_dist:
            best_id, best_dist = candidate_id, dist
    return best_id


def _has_open_shot(game: Game, robot_id: int) -> bool:
    robot_pos = game.friendly_robots[robot_id].p
    goal_x, goal_y1, goal_y2 = enemy_goal_line(game)
    best_shot_y, gap = find_best_shot(robot_pos, list(game.enemy_robots.values()), goal_x, goal_y1, goal_y2)
    if best_shot_y is None or gap is None:
        return False
    return not segment_blocked(robot_pos, Vector2D(goal_x, best_shot_y), enemy_positions(game))


def _relocate_target(game: Game, robot_id: int, avoid: list[Vector2D]) -> Vector2D:
    """An open point ahead of the current carrier, clear of `avoid`, biased
    toward genuine forward progress rather than merely "away from where this
    robot already stands".

    `dx` used to be hardcoded positive (`ball_x + dx`), silently assuming the
    team always attacks toward `+x` — for `my_team_is_right=False` (attacking
    `-x`), every candidate this produced sat *behind* the ball, toward this
    team's own goal, not ahead of it. Found live investigating a real "these
    strategies never shoot" bug (`high_press`/`overload_flow`/
    `score_aware_zone_flow`, 2026-09-05, `docs/roadmap.md`): traced replay
    showed `give_and_go.shot_lane.open=True` 132 times in one match, yet the
    carrier's x-position never advanced past -1.10 toward a goal at -4.5 —
    support runs were never actually offering forward options, so
    `_best_receiver`/`score_pass_setup` had nothing genuinely advanced to
    pick from even when a lane was open. `attack_sign` below (derived from
    `enemy_goal_line`, the same source every shot/pass helper in this file
    already trusts) fixes the direction; the widened `dx` range (previously
    capped at 1.8m) fixes the separate depth problem — a support run that
    never offers more than ~2m of advance can't consistently reach the shot
    detector's attacking-third gate (`_SHOT_ATTACKING_THIRD_M`, 1.5m past the
    *far* goal line — the final ~3m of a 9m-long field) over repeated short
    hops.
    """
    half_length = game.field.half_length
    half_width = game.field.half_width
    ball_x = game.ball.p.to_2d().x
    current = game.friendly_robots[robot_id].p
    goal_x, _goal_y1, _goal_y2 = enemy_goal_line(game)
    attack_sign = 1.0 if goal_x > 0 else -1.0

    # When the ball is in our own half, support points ahead of the ball
    # (`ball_x + attack_sign * dx`) sit on the clamped defense-area edge,
    # packing the whole trio onto the box line — a scrum that gets shoved
    # across it in loose-ball scrambles (defense-area fouls). Hold a retreat
    # line instead: well off the area front edge.
    own_goal_sign = -attack_sign
    own_half_edge = own_goal_sign * 0.0  # midfield in own-goal-sign coords
    area_front_x = float(game.field.my_defense_area[1][0])
    if (ball_x - own_half_edge) * own_goal_sign > 0.0:
        # Ball is in our own half: cap candidates short of the box.
        forward_cap = area_front_x - own_goal_sign * _RELOCATE_BOX_RETREAT
    else:
        forward_cap = attack_sign * (half_length - 0.5)
    # `forward_cap` clamps toward whichever end is nearer the attacking goal
    # — `min`/`max` needs to match `attack_sign` so this clamp still limits
    # *overshoot past the cap*, not silently no-op or clamp the wrong way.
    clamp = min if attack_sign > 0 else max

    candidates = [
        Vector2D(
            clamp(ball_x + attack_sign * dx, forward_cap), max(-half_width + 0.6, min(half_width - 0.6, current.y + dy))
        )
        for dx in (0.5, 1.0, 1.8, 2.8, 4.0)
        for dy in (-1.2, 1.2, -2.2, 2.2)
    ]
    best, best_progress = None, None
    for point in candidates:
        if any(point.distance_to(other) < _RELOCATE_MIN_SEPARATION for other in avoid):
            continue
        # Prefer real progress toward the attacking goal first (so a deep,
        # reachable run beats a short lateral shuffle); among similarly
        # advanced options, prefer whichever is farther from this robot's
        # current spot (the tactic's own existing tie-break, kept as-is —
        # spreads support robots apart rather than clustering them).
        progress = point.x * attack_sign
        key = (progress, point.distance_to(current))
        if best_progress is None or key > best_progress:
            best, best_progress = point, key
    return best if best is not None else current


@dataclass
class GiveAndGoMem:
    carrier_id: Optional[int] = None
    receiver_id: Optional[int] = None  # locked target for the in-flight hop, None while deciding
    hop_count: int = 0
    ticks_held: int = 0  # ticks since the current carrier was assigned; feeds suggest_next's timeout below
    hop_ticks: int = 0  # ticks since receiver_id was locked for the current hop; feeds _MAX_HOP_TICKS below
    lane_blocked_ticks: int = 0  # consecutive ticks _pass_exec reported the lane blocked; feeds early hop abandon
    carry_origin: Optional[Vector2D] = None  # see shared carry_origin; caps the no-lane strafe
    first_touch_hold_spent: bool = False  # the no-receiver hold ran its _MAX_HOP_TICKS once; never hold again


class GiveAndGoTactic(BaseTactic[GiveAndGoMem]):
    """2+ robots cycling carrier/receiver roles via repeated one-two passes.

    Works with any `robot_ids` count >= 2 (fewer than 2 falls back to
    dribble-and-shoot with no passing, same as a single-robot
    `LeadAndSupportTactic`). Uncommitted, non-carrying robots relocate to a
    fresh support point every tick rather than holding a fixed position, so
    the pool of pass targets keeps changing as the carrier's situation does
    — the mechanism this tactic is meant to exercise.
    """

    tag = TacticTag.ATTACK

    def initial_mem(self) -> GiveAndGoMem:
        return GiveAndGoMem()

    def is_committed(self, game: Game, mem: GiveAndGoMem) -> bool:
        if mem.carrier_id is None:
            return False
        # Mid-hop: a receiver has been picked for this hop and the ball hasn't
        # landed with them yet. Reassignment here would strand a pass in flight.
        return mem.receiver_id is not None

    def suggest_next(self, game: Game, mem: GiveAndGoMem) -> Optional[TacticId]:
        """Purely advisory (see `Tactic.suggest_next`'s contract) — this
        tactic's own `tick()` already force-shoots once `hop_count` hits
        `_MAX_HOPS_PER_POSSESSION` rather than cycling forever, so this isn't
        needed for correctness. It exists only for a `TacticGraph`-driven
        repertoire that wants a chance to try a different attacking pattern
        rather than staying on this one indefinitely. Triggers on wall-clock
        ticks held (`_SUGGEST_HANDOFF_TICKS`), not `hop_count` — observed in
        a real match against a low-possession opponent that this tactic can
        hold the carrier role for the entire match without ever completing a
        single hop (falls back to solo dribble-and-shoot when it can't find
        a good pass, per the class docstring), so a hop-count-based trigger
        would simply never fire. No existing strategy calls this (nothing
        consulted `suggest_next` anywhere until `strategy/tactic_graph.py`),
        so this has no effect on any already-tuned `build_*_kernel_strategy`
        config.
        """
        if mem.ticks_held >= _SUGGEST_HANDOFF_TICKS:
            return "switch"
        return None

    def highlights(self, mem: GiveAndGoMem) -> dict[RobotId, str]:
        highlights: dict[RobotId, str] = {}
        if mem.carrier_id is not None:
            highlights[mem.carrier_id] = "carrier"
        if mem.receiver_id is not None:
            highlights[mem.receiver_id] = "receiver"
        return highlights

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: GiveAndGoMem
    ) -> tuple[dict[RobotId, RobotCommand], GiveAndGoMem]:
        if mem.carrier_id is None or mem.carrier_id not in robot_ids:
            # `robot_ids` arrives numerically sorted by the scheduler (see
            # `Strategy._run_step`'s `tuple(sorted(robot_ids))`), not ordered
            # by proximity -- `robot_ids[0]` here previously meant "whichever
            # assigned robot has the lowest id", not "whichever is closest to
            # the ball". Self-correcting (the wrong pick just fetches the
            # ball via `go_to_ball` below) rather than a permanent lockout
            # like the equivalent bug found in `PressAndContainTactic`
            # (2026-09-01), but still wastes time sending a farther robot
            # after the ball while a closer teammate relocates instead. Pick
            # the actually-closest assigned robot as the initial carrier.
            ball_pos = game.ball.p.to_2d()
            initial_carrier = min(robot_ids, key=lambda rid: game.friendly_robots[rid].p.distance_to(ball_pos))
            mem.carrier_id, mem.receiver_id, mem.hop_count, mem.ticks_held, mem.hop_ticks = (
                initial_carrier,
                None,
                0,
                0,
                0,
            )

        mem.ticks_held += 1
        carrier_id = mem.carrier_id
        commands: dict[RobotId, RobotCommand] = {}
        carrier_has_ball = has_ball(game, carrier_id)
        mem.carry_origin = carry_origin(game, carrier_id, mem.carry_origin)

        if ctx.match_log is not None:
            ctx.match_log.trace_if_changed(
                tick=0, sim_time=getattr(game, "ts", 0.0), key="give_and_go.carrier_has_ball", value=carrier_has_ball
            )

        # Just after the kick the carrier no longer has the ball, but the pass is
        # still ours: keep the receiver on it (see `_pass_exec`) rather than
        # relocating it with the others while the ball rolls at it.
        pass_rolling = (
            mem.receiver_id is not None
            and mem.receiver_id in robot_ids
            and ball_line_receive_point(game, mem.receiver_id) is not None
        )
        if not carrier_has_ball and not pass_rolling:
            if ball_in_own_defense_area(game):
                # The ball is inside our own box — an outfield robot may not
                # enter it (DefenseAreaRule: the keeper owns the area). Hold
                # the edge nearest the ball instead of chasing it in.
                commands[carrier_id] = go_to_point(
                    game=game,
                    motion_controller=ctx.motion_controller,
                    robot_id=carrier_id,
                    target_coords=own_defense_area_exit_point(game, game.ball.p.to_2d().y),
                )
            else:
                commands[carrier_id] = go_to_ball(
                    game=game, motion_controller=ctx.motion_controller, robot_id=carrier_id, ctx=ctx
                )
            self._relocate_others(game, ctx, robot_ids, carrier_id, commands)
            return commands, mem

        carrier_pos = game.friendly_robots[carrier_id].p
        if in_own_defense_area(game, carrier_pos):
            # Carried the ball into our own box (e.g. a rebound scramble):
            # holding it inside makes 2 robots in the area (keeper + carrier)
            # and draws the same foul — dribble straight out to the edge.
            commands[carrier_id] = go_to_point(
                game=game,
                motion_controller=ctx.motion_controller,
                robot_id=carrier_id,
                target_coords=own_defense_area_exit_point(game, carrier_pos.y),
                dribbling=True,
            )
            self._relocate_others(game, ctx, robot_ids, carrier_id, commands)
            return commands, mem

        others = tuple(rid for rid in robot_ids if rid != carrier_id)

        force_shot = (
            mem.hop_count >= _MAX_HOPS_PER_POSSESSION or not others or mem.ticks_held >= _FIRST_TOUCH_FORCE_SHOT_TICKS
        )
        # Never let the *first* touch of a possession be a solo shot when a
        # teammate is available, even if `_has_open_shot` says the lane is
        # clear — an unblocked lane right after a kickoff/restart is not
        # evidence the shot will land, only that nobody has moved into the
        # way yet, and every one of this tactic's shots is a single
        # uncontested kick with no in-flight recovery. If it doesn't score,
        # the carrier is the only robot near the dead ball afterward and
        # ends up touching it again itself: SSL's double-touch rule forbids
        # exactly this for the entire window from NORMAL_START until some
        # *other* robot touches the ball first (see DoubleTouchRule) — found
        # live at kickoff, where the goal is wide open by construction and
        # `_has_open_shot` fires on hop 0 with two idle teammates standing
        # by. hop_count >= 1 keeps today's "shoot when open" behavior for
        # every later touch in the same possession, once a genuine
        # intervening touch has already happened.
        first_touch_of_possession = mem.hop_count == 0
        # Cumulative give-up: unlike `hop_ticks` (which only bounds a single
        # locked-in hop or a single no-candidate hold and gets reset on every
        # fresh attempt), `ticks_held` never resets while `hop_count` is
        # still 0 — see `_MAX_FIRST_TOUCH_TICKS`'s definition for why a
        # per-attempt budget alone lets this loop run forever in congested
        # play. Not yet treated as `force_shot` at this threshold: a
        # stuck-since-restart first touch must still avoid an immediate solo
        # shot, so this disables further receiver search and drops to the
        # no-open-lane dribble/reposition path first, same as a stationary
        # keeper leaves no lane at all. `force_shot`'s own, longer
        # `_FIRST_TOUCH_FORCE_SHOT_TICKS` threshold (see its definition)
        # eventually overrides this and takes the open shot rather than
        # repositioning forever — see that constant's comment for why.
        first_touch_stuck = first_touch_of_possession and mem.ticks_held >= _MAX_FIRST_TOUCH_TICKS
        if not force_shot and not first_touch_stuck and mem.receiver_id is None and first_touch_of_possession:
            # The very first touch of a possession must never be a solo shot
            # (see the block comment above `first_touch_of_possession`'s
            # definition, moved here) — use the relaxed selector, not
            # `_best_receiver`, since a kickoff/restart formation has no
            # teammate anywhere near a scoring position yet and
            # `score_pass_setup` would reject all of them.
            mem.receiver_id = _nearest_safe_receiver(game, carrier_id, others)
            if mem.receiver_id is not None:
                mem.hop_ticks = 0
                mem.lane_blocked_ticks = 0
        elif (
            not force_shot
            and not first_touch_stuck
            and mem.receiver_id is None
            and not _has_open_shot(game, carrier_id)
        ):
            mem.receiver_id = _best_receiver(game, carrier_id, others)
            mem.hop_ticks = 0
            mem.lane_blocked_ticks = 0

        if (
            first_touch_of_possession
            and not first_touch_stuck
            and not mem.first_touch_hold_spent
            and mem.receiver_id is None
            and not force_shot
        ):
            # No teammate was even reachable/unblocked (e.g. boxed in by
            # opponents right at kickoff) — hold rather than fall through to
            # the shoot branch below, which would reintroduce the double-
            # touch bug this whole block exists to prevent. Retry next tick
            # as `_relocate_others` keeps repositioning teammates; give up
            # and force a solo shot only after `_MAX_HOP_TICKS`, the same
            # timeout budget the in-flight pass handshake already uses.
            #
            # The give-up is once per possession (`first_touch_hold_spent`):
            # it used to fall through for one tick and then hold again for
            # another `_MAX_HOP_TICKS`, so a pressed carrier with no open
            # teammate stood still with the ball until `first_touch_stuck`
            # (8 s of `ticks_held`). A repartition or a referee reset renews
            # `mem` before that (overload_flow's picker every ~6 s, and the
            # referee's own no_progress restart at 10 s), so it never came:
            # carriers froze 15-20 s while both teams' other robots milled
            # around them (tournament_20261003_102921,
            # clear_danger_vs_overload_flow_RK t=171-192 s).
            mem.hop_ticks += 1
            if mem.hop_ticks < _MAX_HOP_TICKS:
                carrier_pos = game.friendly_robots[carrier_id].p
                goal_x, goal_y1, goal_y2 = enemy_goal_line(game)
                commands[carrier_id] = move(
                    game=game,
                    motion_controller=ctx.motion_controller,
                    robot_id=carrier_id,
                    target_coords=carrier_pos,
                    target_oren=carrier_pos.angle_to(Vector2D(goal_x, (goal_y1 + goal_y2) / 2.0)),
                    dribbling=True,
                )
                self._relocate_others(game, ctx, robot_ids, carrier_id, commands)
                return commands, mem
            mem.hop_ticks = 0
            mem.first_touch_hold_spent = True

        if mem.receiver_id is not None:
            if ctx.match_log is not None:
                intercept_pos, _intercept_oren = intercept_point(game, carrier_id, mem.receiver_id)
                ctx.match_log.trace_if_changed(
                    tick=0,
                    sim_time=getattr(game, "ts", 0.0),
                    key="give_and_go.pass_target",
                    value={"receiver_id": mem.receiver_id, "x": intercept_pos.x, "y": intercept_pos.y},
                )

            mem.hop_ticks += 1
            if mem.hop_ticks >= _MAX_HOP_TICKS or mem.lane_blocked_ticks >= _LANE_BLOCKED_ABANDON_TICKS:
                # Receiver never became ready (unreachable intercept point,
                # stuck itself, or any other reason _pass_exec's handshake
                # never resolves), or an enemy has settled onto the direct
                # line to this receiver — abandon this hop rather than
                # holding the ball (or aiming through the enemy) forever.
                # Falls through to the shoot-or-reposition branch below on
                # this same tick.
                mem.receiver_id = None
                mem.hop_ticks = 0
                mem.lane_blocked_ticks = 0
            else:
                hop_commands, pass_complete, lane_blocked = _pass_exec(game, ctx, carrier_id, mem.receiver_id)
                mem.lane_blocked_ticks = mem.lane_blocked_ticks + 1 if lane_blocked else 0
                commands.update(hop_commands)
                self._relocate_others(game, ctx, robot_ids, carrier_id, commands, also_exclude=mem.receiver_id)
                if pass_complete:
                    mem.carrier_id, mem.receiver_id = mem.receiver_id, None
                    mem.carry_origin = None
                    mem.hop_count += 1
                    mem.hop_ticks = 0
                    mem.lane_blocked_ticks = 0
                return commands, mem

        # No pass in flight: either we have an open shot, or we've hit the hop
        # cap and are forcing one regardless of lane quality.
        goal_x, goal_y1, goal_y2 = enemy_goal_line(game)
        carrier_pos = game.friendly_robots[carrier_id].p
        best_shot_y, _gap = find_best_shot(carrier_pos, list(game.enemy_robots.values()), goal_x, goal_y1, goal_y2)

        if ctx.match_log is not None:
            ctx.match_log.trace_if_changed(
                tick=0,
                sim_time=getattr(game, "ts", 0.0),
                key="give_and_go.shot_lane",
                value={
                    "from": {"x": carrier_pos.x, "y": carrier_pos.y},
                    "to": {"x": goal_x, "y": best_shot_y} if best_shot_y is not None else None,
                    "open": best_shot_y is not None,
                },
            )

        # The strafe below carries the ball sideways with no end of its own;
        # past 1.0m that is an excessive-dribbling foul (51 in the 2026-09-23
        # round-robin). Once the carry is spent, shoot at the goal centre
        # instead: a blocked shot keeps the ball live, a foul hands it over.
        carry_spent = carry_exhausted(game, mem.carry_origin)
        if carry_spent and best_shot_y is None:
            best_shot_y = (goal_y1 + goal_y2) / 2.0
        if not carry_spent and (best_shot_y is None or (first_touch_stuck and not force_shot)):
            # No open lane at all — freezing here (the old behaviour) never
            # resolves against a stationary blocker (e.g. a keeper at the
            # goal mouth): nothing about the position changes, so the shot
            # search comes back empty forever. Strafe instead; see
            # `no_shot_reposition_target`'s docstring.
            reposition = no_shot_reposition_target(
                carrier_pos, enemy_positions(game), goal_x, goal_y1, goal_y2, game.field.half_width
            )
            commands[carrier_id] = move(
                game=game,
                motion_controller=ctx.motion_controller,
                robot_id=carrier_id,
                target_coords=reposition,
                target_oren=carrier_pos.angle_to(Vector2D(goal_x, (goal_y1 + goal_y2) / 2.0)),
                dribbling=True,
            )
        else:
            target_oren = carrier_pos.angle_to(Vector2D(goal_x, best_shot_y))
            if oriented_towards(game, carrier_id, target_oren):
                commands[carrier_id] = kick()
            else:
                commands[carrier_id] = turn_on_spot(
                    game=game,
                    motion_controller=ctx.motion_controller,
                    robot_id=carrier_id,
                    target_oren=target_oren,
                    dribbling=True,
                )
        self._relocate_others(game, ctx, robot_ids, carrier_id, commands)
        return commands, mem

    def _relocate_others(
        self,
        game: Game,
        ctx: TickContext,
        robot_ids: tuple[RobotId, ...],
        carrier_id: RobotId,
        commands: dict[RobotId, RobotCommand],
        also_exclude: Optional[RobotId] = None,
    ) -> None:
        occupied = [game.friendly_robots[carrier_id].p]
        for robot_id in robot_ids:
            if robot_id == carrier_id or robot_id == also_exclude or robot_id in commands:
                continue
            target = _relocate_target(game, robot_id, occupied)
            occupied.append(target)
            # Both clamps, not just our own box: `_relocate_target`'s deeper
            # candidates (see its own docstring) now reach within 0.5m of the
            # enemy goal line, close enough to land inside the enemy defense
            # area on some `dy` offsets — an outfield attacker loitering
            # there is `attacker_infringement`, the same class of foul
            # `_pass_exec` already guards its own receive-point target
            # against (see `clamp_outside_enemy_defense_area`'s docstring).
            target = clamp_outside_own_defense_area(game, target)
            target = clamp_outside_enemy_defense_area(game, target)
            commands[robot_id] = go_to_point(
                game=game,
                motion_controller=ctx.motion_controller,
                robot_id=robot_id,
                target_coords=target,
            )
