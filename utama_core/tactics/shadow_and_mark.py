"""Shadow-and-mark defense tactic — shot-line shadowing plus opponent marking.

New tactical logic for defenders beyond the first two, not ported from
Utama-Strategy. `defend_parameter` (already in Core, reused here as-is)
already handles shot-line shadowing for 1-2 defenders, including its own
dynamic side-selection for exactly 2 — see `tactics/defense.py`, which wraps
that function directly and is left unchanged for the 1-2 robot case. That
function has no concept of a 3rd, 4th, or 5th defender, so it cannot simply
be called per-robot for larger defensive groups: nothing stops it from
sending every robot to the same post. This tactic makes the missing decision
explicit — the first two assigned robots shadow via `defend_parameter`
exactly like `DefenseTactic`; any further robots each mark the nearest
not-yet-marked enemy robot (denying that opponent space/a passing lane)
instead of clustering on the same shot line. Markers with no opponent left to
mark hold in open space well clear of our own defense area, via
`_fallback_hold_target` — NOT `defend_parameter` again, which was tried first
and found to converge multiple robots on the same shadow post, tripping the
"too many defenders in own area" rule (see that function's docstring for the
mechanism, found via a live grsim run of the split-shape strategy).

Reuses `defend_parameter`, `go_to_point`, and `proximity_lookup`/plain
distance comparisons — all existing motion/geometry primitives, not
tactics.

Shares `DefenseTactic`'s loose-ball gap for its own shadow pair (both wrap
the same `defend_parameter`, which only ever reacts to the ball's current
position and never approaches it) — see that tactic's docstring for the
live-match pin this was found from. Fixed the same way: the shadow-pair
member nearest an abandoned ball (see `ball_is_loose`) breaks off to fetch
it instead of shadowing a shot no one is taking; the other shadow robot and
all markers are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass

from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
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

_MARK_STANDOFF = 0.6  # metres — mark from this distance on the goal side of the opponent, not on top of them

# How far in front of our own defense area an unmatched marker holds, as a
# fraction of the way from the defense area's front edge to the centre line —
# clearly outside the area (unlike defend_parameter's shadow post, which sits
# only ROBOT_RADIUS outside the front edge on purpose) since nothing here is
# meant to shadow the shot line.
_FALLBACK_HOLD_FRACTION = 0.35
_FALLBACK_Y_SPACING = 0.8  # metres between stacked fallback holders


def _mark_target(game: Game, opponent_id: int) -> Vector2D:
    goal_x = game.field.my_goal_line[0][0]
    opponent = game.enemy_robots[opponent_id]
    direction = 1.0 if goal_x < opponent.p.x else -1.0
    return Vector2D(opponent.p.x + direction * _MARK_STANDOFF, opponent.p.y)


def _fallback_hold_target(game: Game, index: int) -> Vector2D:
    """Open-space holding point for a marker with no opponent left to mark.

    Must not reuse `defend_parameter` (the shadow-post skill): that function
    only ever returns one of two fixed y-values keyed off `robot_id`'s raw
    value once the team has more than 2 robots (see `defend_parameter`'s
    `else` branch) — every fallback marker calling it collides with whichever
    real shadow-defender already owns that post, converging several robots on
    the same spot right at the edge of our own defense area and tripping the
    "too many defenders in own area" rule (confirmed live via the referee GUI
    during a grsim run of the split-shape strategy).
    """
    goal_x = game.field.my_goal_line[0][0]
    defense_front_x = float(game.field.my_defense_area[1][0])
    hold_x = defense_front_x + (goal_x - defense_front_x) * -_FALLBACK_HOLD_FRACTION
    # Stack fallback holders alternately above/below the ball's y so multiple
    # unmatched markers don't stand on top of each other either.
    ball_y = game.ball.p.to_2d().y
    offset = _FALLBACK_Y_SPACING * ((index + 1) // 2) * (1 if index % 2 == 0 else -1)
    return Vector2D(hold_x, ball_y + offset)


def _assign_marks(game: Game, marker_ids: tuple[int, ...]) -> dict[int, int]:
    """Nearest-unmarked-opponent-first greedy assignment, marker by marker."""
    enemy_ids = list(game.enemy_robots.keys())
    assignment: dict[int, int] = {}
    remaining_enemies = set(enemy_ids)

    for marker_id in marker_ids:
        if not remaining_enemies:
            break
        marker_pos = game.friendly_robots[marker_id].p
        closest_enemy = min(remaining_enemies, key=lambda eid: marker_pos.distance_to(game.enemy_robots[eid].p))
        assignment[marker_id] = closest_enemy
        remaining_enemies.discard(closest_enemy)

    return assignment


@dataclass
class ShadowAndMarkMem:
    """No cross-tick state: shadow assignment and marks are recomputed every tick."""


class ShadowAndMarkTactic(BaseTactic[ShadowAndMarkMem]):
    """Up to 5 defenders: first two shadow the shot line, the rest mark opponents.

    Marking assignment is greedy nearest-first per tick, not sticky — a
    marker can switch targets tick to tick if a closer unmarked opponent
    appears. Acceptable for now since marking has no phase/commitment state
    to disrupt (unlike `LeadAndSupportTactic`'s leader role); revisit only if
    mark-flapping is observed to be a real problem.
    """

    tag = TacticTag.DEFENSE

    def initial_mem(self) -> ShadowAndMarkMem:
        return ShadowAndMarkMem()

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: ShadowAndMarkMem
    ) -> tuple[dict[RobotId, RobotCommand], ShadowAndMarkMem]:
        shadow_ids = robot_ids[:2]
        marker_ids = robot_ids[2:]

        retriever_id = None
        if shadow_ids and ball_is_loose(game):
            ball_pos = game.ball.p.to_2d()
            nearest_id = min(shadow_ids, key=lambda rid: game.friendly_robots[rid].p.distance_to(ball_pos))
            if game.friendly_robots[nearest_id].p.distance_to(ball_pos) <= _LOOSE_BALL_CLAIM_RANGE:
                retriever_id = nearest_id

        commands: dict[RobotId, RobotCommand] = {}
        for robot_id in shadow_ids:
            if robot_id == retriever_id:
                if ball_in_own_defense_area(game):
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
                commands[robot_id] = defend_parameter(game, ctx.motion_controller, robot_id, defender_group=shadow_ids)

        marks = _assign_marks(game, marker_ids)
        fallback_index = 0
        for marker_id in marker_ids:
            opponent_id = marks.get(marker_id)
            if opponent_id is None:
                # More markers than opponents to mark — hold in open space
                # (see `_fallback_hold_target`'s docstring for why this can't
                # just call `defend_parameter` again).
                target = _fallback_hold_target(game, fallback_index)
                fallback_index += 1
                commands[marker_id] = go_to_point(
                    game=game, motion_controller=ctx.motion_controller, robot_id=marker_id, target_coords=target
                )
                continue
            target = _mark_target(game, opponent_id)
            commands[marker_id] = go_to_point(
                game=game, motion_controller=ctx.motion_controller, robot_id=marker_id, target_coords=target
            )

        return commands, mem
