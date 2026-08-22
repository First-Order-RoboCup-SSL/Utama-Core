"""Danger clearance defense tactic — when pinned deep under pressure, kick the
ball out of danger instead of trying to build through it.

Every attack tactic in the catalog is possession-building (`GiveAndGoTactic`'s
hops, `SwitchOfPlayTactic`'s relay, `PassAndShootTactic`'s scripted setup), and
every existing defensive tactic is positioning-only: `DefenseTactic`/
`ShadowAndMarkTactic` shadow the shot line, `BlockShapeTactic` holds a zone
screen, `PressAndContainTactic` contains but never removes the ball. The only
`kick()` call sites anywhere aim at the *enemy goal* (via `find_best_shot`) or
at a teammate (`_pass_exec`). No tactic ever deliberately kicks the ball out of
danger — so a team pinned deep in its own third under pressure has no relief
valve: it can only shadow/mark/block and wait, which is the shape behind the
catalog's worst conceded-under-pressure records (e.g. `low_block`, GF0-GA6
with an 80% defense split).

This tactic is that valve. It reads game state nothing else composes:
*danger* = ball deep in our own defensive third AND an enemy contesting the
ball. `kernel_strategy._ball_zone()`'s thirds vocabulary exists but only picks
between attack patterns; `PressAndContainTactic` reads enemy proximity to the
ball but never ball depth; nothing combines both. When danger fires, the
assigned robot nearest the ball wins it and kicks long toward open space —
the clearance direction scored on enemy-clearance along the kick lane — while
the rest hold a spread line ahead of the ball as next-phase cover.

Mechanically this leans on two already-fixed pieces: rsim's kick-direction fix
(a clearance now goes where aimed) and the PID-level orientation-discontinuity
auto-reset (no manual motion-controller resets needed across the chase→aim→kick
transitions). The aim-and-fire pattern mirrors `GiveAndGoTactic`'s shot branch
exactly (`turn_on_spot(dribbling=True)` until `oriented_towards`, then `kick()`).

No commitment: like `BlockShapeTactic`, roles are recomputed every tick — a
clearance takes about a second, and if the kernel reassigns mid-aim the new
robot simply re-chases; there is no in-flight pass to strand. `applicable()`
is a real §15 precondition: with the ball out of our defensive third (or no
enemy contesting it) there is nothing to clear, so the slot evaporates and the
Partitioner folds its robots elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utama_core.engine.context import KernelContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.shared.pass_and_score_geometry import (
    ball_in_own_defense_area,
    clamp_outside_own_defense_area,
    enemy_positions,
    has_ball,
    oriented_towards,
    own_defense_area_exit_point,
    segment_clearance,
)
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.skills.src.utils.move_utils import kick, turn_on_spot

# Ball deeper than this (distance from our own goal line) counts as "danger
# zone" for applicability. Slightly tighter than the whole own third
# (2*half_length/3 ≈ 3.0 m): the last metre of buildup in front of the box is
# normal possession play, not yet an emergency.
_DANGER_DEPTH = 2.6
# An enemy within this range of the ball makes the danger "contested" — without
# this gate the tactic would fire on every harmless back-pass to a free keeper
# situation even with no opponent anywhere near.
_PRESSURE_RANGE = 1.2
# Clearance-lane openness beyond 1.0 m is all equally "clear"; capping keeps a
# fully-open lane from dominating the score via raw distance.
_LANE_OPENNESS_CAP = 1.0
# Hysteresis margin between clearance-direction candidates (same shape as
# `find_best_shot`'s switch_margin / `go_to_ball`'s commit-range): recomputing
# the best lane fresh every tick lets ordinary enemy jitter flip which candidate
# wins between near-equal lanes, swinging the aim target tick to tick.
_SWITCH_MARGIN = 0.15
# How far upfield the candidate landing points sit from the ball.
_CLEAR_DISTANCE = 4.5


def _own_goal_x(game: Game) -> float:
    return (1.0 if game.my_team_is_right else -1.0) * game.field.half_length


def _progress_from_own_goal(game: Game, x: float) -> float:
    """Signed distance from our goal line toward the enemy goal (see block_shape)."""
    sign = 1.0 if game.my_team_is_right else -1.0
    return (x - _own_goal_x(game)) * -sign


def _nearest_enemy_distance_to_ball(game: Game) -> Optional[float]:
    ball_p = game.ball.p.to_2d()
    best: Optional[float] = None
    for enemy in game.enemy_robots.values():
        if enemy is None:
            continue
        dist = enemy.p.distance_to(ball_p)
        if best is None or dist < best:
            best = dist
    return best


def in_danger(game: Game) -> bool:
    """The tactic's danger read: ball deep in our own third, contested by an enemy."""
    if game.ball is None:
        return False
    if _progress_from_own_goal(game, game.ball.p.to_2d().x) > _DANGER_DEPTH:
        return False
    enemy_dist = _nearest_enemy_distance_to_ball(game)
    return enemy_dist is not None and enemy_dist <= _PRESSURE_RANGE


def _clearance_candidates(game: Game, ball_p: Vector2D) -> list[Vector2D]:
    """Upfield landing points at spread y offsets, clamped inside the field."""
    half_length = game.field.half_length
    half_width = game.field.half_width
    attack_dir = -1.0 if game.my_team_is_right else 1.0
    land_x = max(-half_length + 0.5, min(half_length - 0.5, ball_p.x + attack_dir * _CLEAR_DISTANCE))
    return [Vector2D(land_x, y) for y in (-half_width * 0.6, 0.0, half_width * 0.6)]


def _best_clear_target(game: Game, ball_p: Vector2D, prev_target: Optional[Vector2D]) -> Vector2D:
    """Openest upfield lane, with hysteresis toward the previous choice."""
    enemies = enemy_positions(game)

    def _score(point: Vector2D) -> float:
        return min(segment_clearance(ball_p, point, enemies), _LANE_OPENNESS_CAP)

    candidates = _clearance_candidates(game, ball_p)
    if prev_target is not None:
        prev_score = min(segment_clearance(ball_p, prev_target, enemies), _LANE_OPENNESS_CAP)
        best_point, best_score = prev_target, prev_score
        for candidate in candidates:
            # A challenger must clearly beat the standing choice, not tie it.
            if _score(candidate) > best_score + _SWITCH_MARGIN:
                best_point, best_score = candidate, _score(candidate)
        return best_point

    return max(candidates, key=_score)


@dataclass
class ClearBallMem:
    """Last clearance target, kept only for direction-choice hysteresis."""

    prev_clear_target: Optional[Vector2D] = None


class ClearBallTactic(BaseTactic[ClearBallMem]):
    """Win the ball in the danger zone and kick it out (see module docstring).

    Robot-count-agnostic: 1 robot clears solo; 2+ also hold the cover line.
    """

    tag = TacticTag.DEFENSE

    def initial_mem(self) -> ClearBallMem:
        return ClearBallMem()

    def applicable(self, game: Game) -> bool:
        return in_danger(game)

    def tick(
        self, game: Game, ctx: KernelContext, robot_ids: tuple[RobotId, ...], mem: ClearBallMem
    ) -> tuple[dict[RobotId, RobotCommand], ClearBallMem]:
        commands: dict[RobotId, RobotCommand] = {}
        if not robot_ids or game.ball is None:
            return commands, mem

        ball_p = game.ball.p.to_2d()
        clearer_id = min(robot_ids, key=lambda rid: game.friendly_robots[rid].p.distance_to(ball_p))
        others = [rid for rid in sorted(robot_ids) if rid != clearer_id]

        phase = self._command_clearer(game, ctx, clearer_id, ball_p, mem, commands)
        self._trace(ctx, game, clearer_id, phase, commands.get(clearer_id))
        self._hold_cover_line(game, ctx, others, ball_p, commands)
        return commands, mem

    def _command_clearer(
        self,
        game: Game,
        ctx: KernelContext,
        clearer_id: RobotId,
        ball_p: Vector2D,
        mem: ClearBallMem,
        commands: dict[RobotId, RobotCommand],
    ) -> str:
        if ball_in_own_defense_area(game):
            # Outfield robots may not enter our own box (DefenseAreaRule — the
            # keeper owns it). Hold the edge nearest the ball; the keeper will
            # distribute, or the ball will roll out to us.
            commands[clearer_id] = go_to_point(
                game=game,
                motion_controller=ctx.motion_controller,
                robot_id=clearer_id,
                target_coords=own_defense_area_exit_point(game, ball_p.y),
            )
            return "hold_exit"

        if not has_ball(game, clearer_id, visual=True):
            commands[clearer_id] = go_to_ball(
                game=game, motion_controller=ctx.motion_controller, robot_id=clearer_id, ctx=ctx
            )
            return "chase"

        target = _best_clear_target(game, ball_p, mem.prev_clear_target)
        target_oren = game.friendly_robots[clearer_id].p.angle_to(target)
        if oriented_towards(game, clearer_id, target_oren):
            commands[clearer_id] = kick()
            mem.prev_clear_target = None
            return "kick"
        commands[clearer_id] = turn_on_spot(
            game=game,
            motion_controller=ctx.motion_controller,
            robot_id=clearer_id,
            target_oren=target_oren,
            dribbling=True,
        )
        mem.prev_clear_target = target
        return "aim"

    def _hold_cover_line(
        self,
        game: Game,
        ctx: KernelContext,
        other_ids: list[RobotId],
        ball_p: Vector2D,
        commands: dict[RobotId, RobotCommand],
    ) -> None:
        """Cover robots hold a spread line a little upfield of the ball — out of
        the danger pocket, positioned for whatever follows the clearance."""
        half_width = game.field.half_width
        attack_dir = -1.0 if game.my_team_is_right else 1.0
        line_x = ball_p.x + attack_dir * 1.8
        lane_limit = half_width - 0.7
        spreads = ((-0.9, 0.9), (-1.4, 0.0, 1.4), (-1.8, -0.6, 0.6, 1.8))
        offsets = spreads[min(len(other_ids), len(spreads)) - 1]
        for rid, offset in zip(other_ids, offsets):
            target_y = max(-lane_limit, min(lane_limit, ball_p.y + offset))
            commands[rid] = go_to_point(
                game=game,
                motion_controller=ctx.motion_controller,
                robot_id=rid,
                target_coords=clamp_outside_own_defense_area(game, Vector2D(line_x, target_y)),
            )

    def _trace(
        self,
        ctx: KernelContext,
        game: Game,
        clearer_id: RobotId,
        phase: str,
        command: Optional[RobotCommand],
    ) -> None:
        """Per-tick mechanism trace (see STRATEGY_DEVELOPMENT.md's observability
        section). `match_log` is None on tournament/CI runs, so this is a no-op
        there and stays in permanently."""
        if ctx.match_log is None:
            return
        value: dict[str, object] = {"phase": phase}
        if phase == "aim" and command is not None:
            robot = game.friendly_robots[clearer_id]
            value["oren"] = round(float(robot.orientation), 3)
        ctx.match_log.trace(tick=0, sim_time=getattr(game, "ts", 0.0), key=f"clear_ball[{clearer_id}]", value=value)
