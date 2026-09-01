"""Press-and-contain defense tactic — active ball pressure, not shot-line shadowing.

New tactical logic, not ported from Utama-Strategy. Every existing defensive
tactic (`DefenseTactic`, `ShadowAndMarkTactic`) is reactive: it shadows a
fixed shot line or marks a fixed opponent, regardless of how urgent the ball
is. This tactic makes the opposite bet — the robot closest to the ball
actively closes it down and denies the immediate shot/pass lane (via
`skills.block_attacker`, which already exists in Core but had no tactic
calling it), while every other assigned robot marks the nearest still-open
opponent (via `skills.man_mark`, also previously unused by any tactic) to
deny outlet passes. There is no fixed "first two shadow, rest mark" split
like `ShadowAndMarkTactic` — the presser role follows the ball-nearest enemy
every tick, since who is most dangerous right now is exactly what pressing
is supposed to react to.

`applicable()` is the interesting part of this one: pressing only makes
sense when the opponent actually has the ball (or is clearly about to), so
this tactic declares itself inapplicable when no enemy is within pressing
range of the ball — see `_PRESS_RANGE`. That is a real, non-trivial use of
the `applicable()` precondition (design doc §15), not just a default-True
stub: it lets a `Partitioner` include this tactic in a defense-tagged pool
without needing its own game-state logic to know when pressing is sensible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.shared.pass_and_score_geometry import (
    ball_in_own_defense_area,
    own_defense_area_exit_point,
)
from utama_core.skills.src.block import block_attacker
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.skills.src.man_mark import man_mark

_PRESS_RANGE = 1.5  # metres — ball must be within this of an enemy for pressing to be applicable


def _enemy_nearest_ball(game: Game) -> tuple[int, float] | tuple[None, None]:
    ball_pos = game.ball.p.to_2d()
    best_id, best_dist = None, None
    for enemy_id, enemy in game.enemy_robots.items():
        dist = enemy.p.distance_to(ball_pos)
        if best_dist is None or dist < best_dist:
            best_id, best_dist = enemy_id, dist
    return best_id, best_dist


def _assign_markers(game: Game, marker_ids: tuple[int, ...], exclude_enemy_id: int) -> dict[int, int]:
    """Nearest-unmarked-opponent-first greedy assignment, excluding the pressed enemy."""
    remaining_enemies = {eid for eid in game.enemy_robots if eid != exclude_enemy_id}
    assignment: dict[int, int] = {}
    for marker_id in marker_ids:
        if not remaining_enemies:
            break
        marker_pos = game.friendly_robots[marker_id].p
        closest_enemy = min(remaining_enemies, key=lambda eid: marker_pos.distance_to(game.enemy_robots[eid].p))
        assignment[marker_id] = closest_enemy
        remaining_enemies.discard(closest_enemy)
    return assignment


@dataclass
class PressAndContainMem:
    """No cross-tick decision state — presser/marker choice is recomputed
    every tick. `presser_id`/`marks` are last-tick's picks, kept only so
    `highlights()` (called with the *previous* tick's mem) has something to
    report."""

    presser_id: Optional[RobotId] = None
    marks: dict = field(default_factory=dict)


class PressAndContainTactic(BaseTactic[PressAndContainMem]):
    """One presser closing down the ball-nearest enemy, the rest marking outlets.

    Presser assignment is not sticky (re-picked every tick by ball-nearest
    enemy), matching `ShadowAndMarkTactic`'s stance on marking: no
    phase/commitment state exists here to disrupt by switching targets
    tick to tick. Revisit only if presser-flapping is observed to be a real
    problem, not in anticipation of one.
    """

    tag = TacticTag.DEFENSE

    def initial_mem(self) -> PressAndContainMem:
        return PressAndContainMem()

    def applicable(self, game: Game) -> bool:
        _enemy_id, distance = _enemy_nearest_ball(game)
        return distance is not None and distance <= _PRESS_RANGE

    def highlights(self, mem: PressAndContainMem) -> dict[RobotId, str]:
        highlights: dict[RobotId, str] = {rid: "marker" for rid in mem.marks}
        if mem.presser_id is not None:
            highlights[mem.presser_id] = "presser"
        return highlights

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: PressAndContainMem
    ) -> tuple[dict[RobotId, RobotCommand], PressAndContainMem]:
        pressed_enemy_id, _distance = _enemy_nearest_ball(game)

        commands: dict[RobotId, RobotCommand] = {}

        if pressed_enemy_id is None:
            # No enemies on the field at all — nothing to press or mark.
            mem.presser_id, mem.marks = None, {}
            return commands, mem

        # `robot_ids` arrives numerically sorted by the scheduler (see
        # `Strategy._run_step`'s `tuple(sorted(robot_ids))`), not ordered by
        # proximity -- picking `robot_ids[0]` here silently meant "whichever
        # assigned robot has the lowest id", not "whichever is closest to
        # the ball". Confirmed live, 2026-09-01 (full_match_tournament.py
        # stuck-window sweep): a fixed low-id robot sat 0.8m from a parked
        # ball-holder (outside `_STEAL_RANGE`, so it only ever shadowed the
        # shot line) while a different assigned robot sat 0.47m away --
        # inside steal range, close enough to win the ball -- and was
        # demoted to marking duty and never engaged. Nobody ever contested
        # the ball again for the remaining 500s of that match. Pick the
        # actually-closest assigned robot as presser instead.
        pressed_enemy = game.enemy_robots[pressed_enemy_id]
        presser_id = min(robot_ids, key=lambda rid: game.friendly_robots[rid].p.distance_to(pressed_enemy.p))
        marker_ids = tuple(rid for rid in robot_ids if rid != presser_id)
        mem.presser_id = presser_id

        if ctx.match_log is not None:
            ctx.match_log.trace_if_changed(
                tick=0,
                sim_time=getattr(game, "ts", 0.0),
                key="press_and_contain.press",
                value={"presser_id": presser_id, "pressed_enemy_id": pressed_enemy_id},
            )

        if ball_in_own_defense_area(game):
            # The ball is inside our own box — pressing there means an
            # outfield robot in the keeper's area (DefenseAreaRule foul).
            # Hold the edge nearest the ball, like `GiveAndGoTactic`'s
            # carrier; the keeper owns the box.
            commands[presser_id] = go_to_point(
                game=game,
                motion_controller=ctx.motion_controller,
                robot_id=presser_id,
                target_coords=own_defense_area_exit_point(game, game.ball.p.to_2d().y),
            )
        elif game.robot_with_ball is None:
            # Ball is fully loose — nobody on either team currently has it.
            # block_attacker's "attacker doesn't have ball" branch computes
            # the presser's target relative to the tracked enemy's *own*
            # position (a shot-line-style standoff, 70% of the way from
            # that enemy toward the ball), not straight at the ball itself.
            # Found live (stuck-match investigation, 2026-08-26,
            # docs/testing_gaps.md gap #11): when the tracked enemy is
            # itself stationary and far from the ball (e.g. its own team is
            # locked into an all-defense posture with nobody assigned to
            # fetch a loose ball — a separate bug of its own), the computed
            # target never converges on the ball, so the presser parks
            # nearby and never actually closes in to take possession. Two
            # tactics that are each individually reasonable in isolation
            # (contain a threat; don't chase a ball the opponent is closer
            # to) combined into a ball that nobody ever collects, for the
            # rest of a 600s match. When the ball is genuinely unclaimed,
            # go straight for it instead of shadowing a non-threat.
            commands[presser_id] = go_to_ball(
                game=game, motion_controller=ctx.motion_controller, robot_id=presser_id, ctx=ctx
            )
        else:
            commands[presser_id] = block_attacker(
                game=game,
                motion_controller=ctx.motion_controller,
                friendly_robot_id=presser_id,
                enemy_robot_id=pressed_enemy_id,
                attacker_has_ball=game.enemy_robots[pressed_enemy_id].has_ball,
            )

        marks = _assign_markers(game, marker_ids, exclude_enemy_id=pressed_enemy_id)
        mem.marks = marks
        for marker_id in marker_ids:
            opponent_id = marks.get(marker_id)
            if opponent_id is None:
                continue
            commands[marker_id] = man_mark(game, ctx.motion_controller, marker_id, opponent_id)

        return commands, mem
