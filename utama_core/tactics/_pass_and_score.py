"""Shared pass-and-score phase machinery used by `pass_and_shoot`.

Ported from `utama_strategy.functional.strategies.pass_and_score` — only the
pieces `pass_and_shoot` actually calls (`PassAndScoreMem`, `_setup_positions`,
`run_setup_phase`, `_pass_exec`, `_score_goal`). The original module's own
`PassAndScoreStrategy` class (a standalone fixed-assignment tactic) is not
ported in this pass — out of scope; port it if/when a caller needs a
fixed-pair pass tactic distinct from `pass_and_shoot`'s dynamic assignment.

Leading underscore: this is `pass_and_shoot`'s private implementation
detail, not a tactic of its own and not meant to be imported elsewhere.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from utama_core.engine.context import KernelContext
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.shared.field_scaling import scale_point_from_standard_field
from utama_core.shared.pass_and_score_geometry import (
    at_target,
    clamp_outside_enemy_defense_area,
    clamp_outside_own_defense_area,
    clamp_to_field,
    enemy_goal_line,
    enemy_positions,
    find_best_shot,
    has_ball,
    intercept_point,
    no_shot_reposition_target,
    oriented_towards,
    score_pass_setup,
)
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.skills.src.utils.move_utils import (
    empty_command,
    kick,
    move,
    turn_on_spot,
)

_MIN_SETUP_CLEARANCE = 0.35  # metres — must be > FPP's SAFE_OBSTACLES_RADIUS (0.26 m)
_SETUP_SAMPLE_COUNT = 48
_SETUP_SAMPLE_RADIUS_PASSER = 0.6
_SETUP_SAMPLE_RADIUS_RECEIVER = 0.8


def _scaled_setup_positions(passer_pos: Vector2D, receiver_pos: Vector2D, game: Game) -> tuple[Vector2D, Vector2D]:
    scaled_passer = scale_point_from_standard_field(passer_pos, game.field)
    scaled_receiver = scale_point_from_standard_field(receiver_pos, game.field)
    if game.my_team_is_right:
        return Vector2D(-scaled_passer.x, scaled_passer.y), Vector2D(-scaled_receiver.x, scaled_receiver.y)
    return scaled_passer, scaled_receiver


def _sample_near(base: Vector2D, radius: float, rng: random.Random) -> Vector2D:
    angle = rng.uniform(0.0, 2.0 * math.pi)
    distance = radius * (rng.random() ** 0.5)
    return Vector2D(base.x + distance * math.cos(angle), base.y + distance * math.sin(angle))


def choose_setup_positions(
    game: Game,
    base_passer: Vector2D,
    base_receiver: Vector2D,
    rng: random.Random,
    sample_count: int = _SETUP_SAMPLE_COUNT,
) -> tuple[Vector2D, Vector2D]:
    best: Optional[tuple[Vector2D, Vector2D, float]] = None
    for passer, receiver in _candidate_pairs(game, base_passer, base_receiver, rng, sample_count):
        result = score_pass_setup(game, passer, receiver)
        if result is not None and (best is None or result.score > best[2]):
            best = (passer, receiver, result.score)
    if best is None:
        return base_passer, base_receiver
    return best[0], best[1]


def _candidate_pairs(game, base_passer, base_receiver, rng, sample_count):
    yield base_passer, base_receiver
    for _ in range(sample_count):
        passer = clamp_to_field(_sample_near(base_passer, _SETUP_SAMPLE_RADIUS_PASSER, rng), game)
        receiver = clamp_to_field(_sample_near(base_receiver, _SETUP_SAMPLE_RADIUS_RECEIVER, rng), game)
        yield passer, receiver


@dataclass
class PassAndScoreMem:
    phase: str = "setup"  # "setup" -> "pass_then_score" -> "score" -> (goal_scored)
    passer_position: Optional[Vector2D] = None
    receiver_position: Optional[Vector2D] = None
    locked_assignment: Optional[tuple[int, int]] = None
    rng: random.Random = field(default_factory=random.Random)
    goal_scored: bool = False
    phase_ticks: int = 0  # ticks spent in the current non-setup phase; drives the timeout reset
    setup_ticks_without_ball: int = 0  # consecutive setup ticks the passer has read has_ball=False
    prev_best_shot_y: Optional[float] = None  # feeds _score_goal's switch-margin hysteresis; see _score_goal below


def _setup_positions(
    game: Game,
    mem: PassAndScoreMem,
    passer_id: int,
    receiver_id: int,
    base_passer_pos: Vector2D,
    base_receiver_pos: Vector2D,
    dynamic_setup: bool,
) -> PassAndScoreMem:
    assignment = (passer_id, receiver_id)
    if mem.locked_assignment == assignment and mem.passer_position is not None:
        return mem

    base_passer, base_receiver = _scaled_setup_positions(base_passer_pos, base_receiver_pos, game)
    if dynamic_setup:
        # Seed deterministically from the assignment rather than trusting
        # mem.rng's default (unseeded) state. Observed in practice: the
        # kernel slot holding this tactic's mem can hand back a *fresh*
        # PassAndScoreMem mid-setup (its own bookkeeping, outside this
        # tactic's control) — with an unseeded rng that meant every reset
        # sampled a brand-new random setup position, so a robot chasing a
        # position that gets discarded and re-rolled before it arrives
        # never converges on anything. Seeding on (passer_id, receiver_id)
        # makes repeated re-initialization re-derive the *same* candidate
        # position instead of a new random one each time.
        rng = random.Random(hash(assignment))
        passer_pos, receiver_pos = choose_setup_positions(game, base_passer, base_receiver, rng)
    else:
        passer_pos, receiver_pos = base_passer, base_receiver

    mem.locked_assignment = assignment
    mem.passer_position = passer_pos
    mem.receiver_position = receiver_pos
    return mem


def _move_to(game: Game, ctx: KernelContext, robot_id: int, target: Vector2D) -> tuple[RobotCommand, bool]:
    robot = game.friendly_robots[robot_id]
    target_oren = robot.p.angle_to(game.ball.p.to_2d())
    arrived = at_target(game, robot_id, target)
    command = move(
        game=game,
        motion_controller=ctx.motion_controller,
        robot_id=robot_id,
        target_coords=target,
        target_oren=target_oren,
    )
    return command, arrived


def _hold_or_acquire_ball(game: Game, ctx: KernelContext, robot_id: int) -> RobotCommand:
    if has_ball(game, robot_id, visual=True):
        return empty_command(dribbler_on=True)
    return go_to_ball(game=game, motion_controller=ctx.motion_controller, robot_id=robot_id, ctx=ctx)


_SETUP_BALL_LOSS_GRACE_TICKS = 10  # ~0.17s at 60Hz — see comment below


def run_setup_phase(
    game: Game,
    ctx: KernelContext,
    passer_id: int,
    receiver_id: int,
    mem: "PassAndScoreMem",
) -> tuple[dict[int, RobotCommand], bool]:
    """Move passer (carrying the ball) and receiver to their setup positions.

    Returns (commands, phase_complete).
    """
    # The strict IR/contact `has_ball` sensor can stay False even while the
    # robot is visually touching the ball (observed: robot parked on the
    # ball for the rest of a match, has_ball_visual=True, has_ball=False,
    # go_to_ball looping forever with nowhere left to go). Use the visual
    # fallback here, same as _pass_exec's receiver-catch check below, so a
    # flaky sensor reading can't permanently strand the passer at the ball.
    #
    # Grace period on top of that: driving toward passer_position while
    # dribbling can cause has_ball(visual=True) to flicker False for a tick
    # or two even when the ball is still under control (observed: distance
    # to target oscillating 0.3m-2.5m for 30s straight, has_ball toggling
    # every few ticks, never converging) — every flicker was previously
    # switching straight to go_to_ball, which re-approaches the ball from
    # scratch and discards all progress toward passer_position, so the
    # passer never made net progress. Only fall back to re-acquisition once
    # the ball has actually been away for several consecutive ticks.
    if has_ball(game, passer_id, visual=True):
        mem.setup_ticks_without_ball = 0
        passer_cmd, passer_arrived = _move_to(game, ctx, passer_id, mem.passer_position)
    else:
        mem.setup_ticks_without_ball += 1
        if mem.setup_ticks_without_ball <= _SETUP_BALL_LOSS_GRACE_TICKS:
            passer_cmd, _ = _move_to(game, ctx, passer_id, mem.passer_position)
        else:
            passer_cmd = _hold_or_acquire_ball(game, ctx, passer_id)
        passer_arrived = False

    receiver_cmd, receiver_arrived = _move_to(game, ctx, receiver_id, mem.receiver_position)

    return {passer_id: passer_cmd, receiver_id: receiver_cmd}, passer_arrived and receiver_arrived


def _pass_exec(
    game: Game,
    ctx: KernelContext,
    passer_id: int,
    receiver_id: int,
) -> tuple[dict[int, RobotCommand], bool]:
    """Synchronized aiming, intercept positioning, and kick. Returns (commands, pass_complete)."""
    intercept_pos_raw, intercept_oren = intercept_point(game, passer_id, receiver_id)
    # A receiver (or the passer's aim) may never enter our own defense area —
    # the keeper owns the box ("too many defenders in own area" fouls trip on
    # any second robot inside). Clamp the whole pass target to the edge so a
    # defensive scramble around our box cannot walk either robot in.
    intercept_pos = clamp_outside_own_defense_area(game, intercept_pos_raw)
    # The same is true of the *enemy's* box (attacker-infringement fouls),
    # which this function never clamped: an attacking pass's receive point is
    # naturally near the enemy goal by design (that's the whole point of a
    # finishing pass), and `intercept_point` has no notion of the box at all.
    # Found via direct replay/kernel-state inspection: `DecoyOverloadTactic`'s
    # "finish" phase calls this shared helper (decoy -> overloader) and its
    # decoy sat inside the enemy box for 200+ consecutive ticks, driven
    # entirely from here — the tactic-level target clamps added to
    # `decoy_and_overload.py`'s "lure" phase never touched this later phase
    # at all, since it hands off to this shared primitive instead.
    intercept_pos = clamp_outside_enemy_defense_area(game, intercept_pos)

    passer_target_oren = game.friendly_robots[passer_id].p.angle_to(intercept_pos)
    passer_aimed = oriented_towards(game, passer_id, passer_target_oren)
    # visual=True: see run_setup_phase's comment — the strict sensor can
    # stay False while the robot is visually on the ball, which would
    # otherwise strand the passer in go_to_ball indefinitely.
    passer_has_ball = has_ball(game, passer_id, visual=True)

    commands: dict[int, RobotCommand] = {}

    if not passer_has_ball:
        commands[passer_id] = go_to_ball(
            game=game, motion_controller=ctx.motion_controller, robot_id=passer_id, ctx=ctx
        )
    elif not passer_aimed:
        commands[passer_id] = turn_on_spot(
            game=game,
            motion_controller=ctx.motion_controller,
            robot_id=passer_id,
            target_oren=passer_target_oren,
            dribbling=True,
        )
    else:
        commands[passer_id] = empty_command(dribbler_on=True)

    receiver_at_intercept = at_target(game, receiver_id, intercept_pos)
    receiver_facing_pass = oriented_towards(game, receiver_id, intercept_oren)
    receiver_ready = receiver_at_intercept and receiver_facing_pass

    if not receiver_at_intercept:
        commands[receiver_id] = move(
            game=game,
            motion_controller=ctx.motion_controller,
            robot_id=receiver_id,
            target_coords=intercept_pos,
            target_oren=intercept_oren,
        )
    elif not receiver_facing_pass:
        commands[receiver_id] = turn_on_spot(
            game=game, motion_controller=ctx.motion_controller, robot_id=receiver_id, target_oren=intercept_oren
        )
    else:
        commands[receiver_id] = empty_command(dribbler_on=True)

    ready_to_kick = passer_has_ball and passer_aimed and receiver_ready
    if ready_to_kick:
        commands[passer_id] = kick()

    receiver_has_ball = has_ball(game, receiver_id, visual=True)
    pass_complete = receiver_has_ball
    return commands, pass_complete


_SHOT_SWITCH_MARGIN = (
    0.15  # metres of extra clearance a new gap must beat the previous one by; see find_best_shot's own docstring
)


def _score_goal(
    game: Game, ctx: KernelContext, robot_id: int, prev_best_shot_y: Optional[float] = None
) -> tuple[RobotCommand, bool, Optional[float]]:
    goal_x, goal_y1, goal_y2 = enemy_goal_line(game)
    robot = game.friendly_robots[robot_id]
    best_shot_y, _gap = find_best_shot(
        robot.p,
        list(game.enemy_robots.values()),
        goal_x,
        goal_y1,
        goal_y2,
        prev_best_shot_y=prev_best_shot_y,
        switch_margin=_SHOT_SWITCH_MARGIN,
    )
    if best_shot_y is None:
        # See `no_shot_reposition_target`'s docstring: freezing here never
        # resolves against a stationary blocker (e.g. a keeper at the goal
        # mouth) — nothing about a stationary position changes the shot
        # search's inputs, so it never comes back non-None on its own.
        reposition = no_shot_reposition_target(
            robot.p, enemy_positions(game), goal_x, goal_y1, goal_y2, game.field.half_width
        )
        cmd = move(
            game=game,
            motion_controller=ctx.motion_controller,
            robot_id=robot_id,
            target_coords=reposition,
            target_oren=robot.p.angle_to(Vector2D(goal_x, (goal_y1 + goal_y2) / 2.0)),
            dribbling=True,
        )
        return cmd, False, prev_best_shot_y

    target_oren = robot.p.angle_to(Vector2D(goal_x, best_shot_y))
    # visual=True: see run_setup_phase's comment on the strict sensor's
    # unreliability — without this the shooter can stall on the ball
    # forever if the IR/contact flag never fires.
    if not has_ball(game, robot_id, visual=True):
        return (
            go_to_ball(game=game, motion_controller=ctx.motion_controller, robot_id=robot_id, ctx=ctx),
            False,
            best_shot_y,
        )
    if not oriented_towards(game, robot_id, target_oren):
        return (
            turn_on_spot(
                game=game,
                motion_controller=ctx.motion_controller,
                robot_id=robot_id,
                target_oren=target_oren,
                dribbling=True,
            ),
            False,
            best_shot_y,
        )
    return kick(), True, best_shot_y
