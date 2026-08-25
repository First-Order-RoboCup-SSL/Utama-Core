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
    clamp_outside_own_defense_area,
    enemy_goal_line,
    enemy_positions,
    find_best_shot,
    has_ball,
    in_own_defense_area,
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


def _has_open_shot(game: Game, robot_id: int) -> bool:
    robot_pos = game.friendly_robots[robot_id].p
    goal_x, goal_y1, goal_y2 = enemy_goal_line(game)
    best_shot_y, gap = find_best_shot(robot_pos, list(game.enemy_robots.values()), goal_x, goal_y1, goal_y2)
    if best_shot_y is None or gap is None:
        return False
    return not segment_blocked(robot_pos, Vector2D(goal_x, best_shot_y), enemy_positions(game))


def _relocate_target(game: Game, robot_id: int, avoid: list[Vector2D]) -> Vector2D:
    """A simple open point ahead of the current carrier, clear of `avoid`."""
    half_length = game.field.half_length
    half_width = game.field.half_width
    ball_x = game.ball.p.to_2d().x
    current = game.friendly_robots[robot_id].p

    # When the ball is in our own half, support points ahead of the ball
    # (`ball_x + dx`) sit on the clamped defense-area edge, packing the
    # whole trio onto the box line — a scrum that gets shoved across it in
    # loose-ball scrambles (defense-area fouls). Hold a retreat line instead:
    # well off the area front edge.
    own_goal_sign = 1.0 if game.my_team_is_right else -1.0
    own_half_edge = own_goal_sign * 0.0  # midfield in own-goal-sign coords
    area_front_x = float(game.field.my_defense_area[1][0])
    if (ball_x - own_half_edge) * own_goal_sign > 0.0:
        # Ball is in our own half: cap candidates short of the box.
        forward_cap = area_front_x - own_goal_sign * _RELOCATE_BOX_RETREAT
    else:
        forward_cap = half_length - 0.5

    candidates = [
        Vector2D(min(ball_x + dx, forward_cap), max(-half_width + 0.6, min(half_width - 0.6, current.y + dy)))
        for dx in (1.0, 1.8, 0.5)
        for dy in (-1.2, 1.2, -2.2, 2.2)
    ]
    best, best_dist = None, -1.0
    for point in candidates:
        if any(point.distance_to(other) < _RELOCATE_MIN_SEPARATION for other in avoid):
            continue
        dist = point.distance_to(current)
        if dist > best_dist:
            best, best_dist = point, dist
    return best if best is not None else current


@dataclass
class GiveAndGoMem:
    carrier_id: Optional[int] = None
    receiver_id: Optional[int] = None  # locked target for the in-flight hop, None while deciding
    hop_count: int = 0
    ticks_held: int = 0  # ticks since the current carrier was assigned; feeds suggest_next's timeout below


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

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: GiveAndGoMem
    ) -> tuple[dict[RobotId, RobotCommand], GiveAndGoMem]:
        if mem.carrier_id is None or mem.carrier_id not in robot_ids:
            mem.carrier_id, mem.receiver_id, mem.hop_count, mem.ticks_held = robot_ids[0], None, 0, 0

        mem.ticks_held += 1
        carrier_id = mem.carrier_id
        commands: dict[RobotId, RobotCommand] = {}
        carrier_has_ball = has_ball(game, carrier_id)

        if ctx.match_log is not None:
            ctx.match_log.trace_if_changed(
                tick=0, sim_time=getattr(game, "ts", 0.0), key="give_and_go.carrier_has_ball", value=carrier_has_ball
            )

        if not carrier_has_ball:
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

        force_shot = mem.hop_count >= _MAX_HOPS_PER_POSSESSION or not others
        if not force_shot and mem.receiver_id is None and not _has_open_shot(game, carrier_id):
            mem.receiver_id = _best_receiver(game, carrier_id, others)

        if mem.receiver_id is not None:
            hop_commands, pass_complete = _pass_exec(game, ctx, carrier_id, mem.receiver_id)
            commands.update(hop_commands)
            self._relocate_others(game, ctx, robot_ids, carrier_id, commands, also_exclude=mem.receiver_id)
            if pass_complete:
                mem.carrier_id, mem.receiver_id = mem.receiver_id, None
                mem.hop_count += 1
            return commands, mem

        # No pass in flight: either we have an open shot, or we've hit the hop
        # cap and are forcing one regardless of lane quality.
        goal_x, goal_y1, goal_y2 = enemy_goal_line(game)
        carrier_pos = game.friendly_robots[carrier_id].p
        best_shot_y, _gap = find_best_shot(carrier_pos, list(game.enemy_robots.values()), goal_x, goal_y1, goal_y2)
        if best_shot_y is None:
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
            commands[robot_id] = go_to_point(
                game=game,
                motion_controller=ctx.motion_controller,
                robot_id=robot_id,
                target_coords=clamp_outside_own_defense_area(game, target),
            )
