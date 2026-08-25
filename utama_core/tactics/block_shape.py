"""Block-shape defense tactic — a shifting zone screen in front of our own area.

Every existing defensive tactic is man-based: `DefenseTactic`/`ShadowAndMarkTactic`
shadow the ball-to-goal shot line or mark specific opponents, `PressAndContainTactic`
presses the ball-carrier and man-marks everyone else. None of them defend *space*:
real low blocks hold a horizontal screen line at a fixed depth, shift the whole line
sideways with the ball, and let exactly one robot step out of the line to close the
ball-carrier down — the rest stay in the line, never chasing across the pitch and
never collapsing onto the keeper's post (the clustering that trips the "too many
defenders in own area" rule when several shadowing robots converge on one shadow
post).

Roles, recomputed every tick (defense commitments are not needed — a screen is a
continuous shape, not an action):

- **first defender** (the assigned robot nearest the ball): steps OUT of the line
  onto the ball-to-goal axis, a fixed standoff ahead of the ball, so the carrier is
  slowed at a predictable distance from the line. Clamped to stay outside our own
  defense area even when the ball is deep in it — the screen never turns into
  another defender stacking inside the box. Exception: if the ball is loose (see
  `ball_is_loose`) and outside our own box, the first defender drives straight to
  it instead of the fixed lead-offset point — the lead-offset point only makes
  sense as a "slow the carrier down" standoff, which is meaningless once there is
  no carrier at all; without this a loose ball near (but wide of) our own box just
  sits there with the first defender parked on an axis nobody is threatening.
  Found live: a `clear_danger` vs `low_block`-shaped match pinned 28 seconds this
  way (see `tactics/defense.py`'s matching fix for the sibling case in
  `DefenseTactic`).
- **screen** (everyone else): holds the line at `_SCREEN_OFFSET` in front of our
  defense area, each robot on its own lane, the whole line shifting sideways with
  the ball (`_SHIFT_FACTOR`), lanes clamped inside the field width. Never enters
  the defense area, by construction — the line's depth is fixed.

Robot-count-agnostic: 1 robot presses (guards the axis); 2+ maintain the line
behind it.
"""

from __future__ import annotations

from dataclasses import dataclass

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.shared.pass_and_score_geometry import (
    ball_in_own_defense_area,
    ball_is_loose,
)
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.skills.src.go_to_point import go_to_point

# Depth of our own defense area (standard SSL penalty box front edge).
_DEFENSE_AREA_DEPTH = 1.3
# How far in front of the area front edge the screen line sits.
_SCREEN_OFFSET = 0.6
# How far the stepping-out first defender stands in front of the ball.
_FIRST_DEFENDER_LEAD = 0.45
# Screen robots must clear the area front edge by at least this much (radius + margin).
_AREA_CLEARANCE = ROBOT_RADIUS + 0.12
# How strongly the screen line follows the ball in y (0 = fixed center, 1 = glued).
_SHIFT_FACTOR = 0.9
# Lane spread for 1, 2, 3, ... screen robots (y offsets from the shifted center).
_LANE_SPREADS = ((0.0,), (-0.9, 0.9), (-1.4, 0.0, 1.4), (-1.8, -0.6, 0.6, 1.8), (-2.1, -1.05, 0.0, 1.05, 2.1))


@dataclass
class BlockShapeMem:
    """No cross-tick state: the screen is re-derived from the ball every tick."""


class BlockShapeTactic(BaseTactic[BlockShapeMem]):
    """Holds a shifting zone screen in front of our own area (see module docstring)."""

    tag = TacticTag.DEFENSE

    def initial_mem(self) -> BlockShapeMem:
        return BlockShapeMem()

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: BlockShapeMem
    ) -> tuple[dict[RobotId, RobotCommand], BlockShapeMem]:
        commands: dict[RobotId, RobotCommand] = {}
        if not robot_ids:
            return commands, mem

        ball_p = game.ball.p.to_2d()
        own_goal_sign = 1.0 if game.my_team_is_right else -1.0
        half_length = game.field.half_length
        half_width = game.field.half_width
        # Own goal line x and the deepest x an outfield robot may legally stand.
        own_goal_x = own_goal_sign * half_length
        # Limit along the attack axis, as distance *from our own goal line*:
        # everything must be at least the area depth + clearance away.
        min_progress_from_goal = _DEFENSE_AREA_DEPTH + _AREA_CLEARANCE

        def _progress_from_own_goal(x: float) -> float:
            """Signed distance from our goal line toward the enemy goal."""
            return (x - own_goal_x) * -own_goal_sign

        # --- step 1: nearest robot steps out to press the ball on the axis ---
        presser_id = min(robot_ids, key=lambda rid: game.friendly_robots[rid].p.distance_to(ball_p))

        if ball_is_loose(game) and not ball_in_own_defense_area(game):
            # No carrier to stand off from — go get it instead of guarding an
            # axis nobody is threatening (see class docstring).
            commands[presser_id] = go_to_ball(
                game=game, motion_controller=ctx.motion_controller, robot_id=presser_id, ctx=ctx
            )
        else:
            goal_x = own_goal_sign * half_length  # our own goal — press toward it keeps us between ball and goal
            axis = (goal_x - ball_p.x, 0.0 - ball_p.y)
            axis_len = (axis[0] ** 2 + axis[1] ** 2) ** 0.5
            if axis_len > 1e-6:
                lead_x = ball_p.x + axis[0] / axis_len * _FIRST_DEFENDER_LEAD
                lead_y = ball_p.y + axis[1] / axis_len * _FIRST_DEFENDER_LEAD
            else:
                lead_x, lead_y = ball_p.x, ball_p.y
            # Clamp the presser to stay clear of our own defense area.
            if _progress_from_own_goal(lead_x) < min_progress_from_goal:
                lead_x = own_goal_x - own_goal_sign * min_progress_from_goal
            lead_y = max(-half_width + 0.5, min(half_width - 0.5, lead_y))
            commands[presser_id] = go_to_point(game, ctx.motion_controller, presser_id, (lead_x, lead_y))

        # --- the rest hold the shifting screen line ---
        screen_ids = [rid for rid in robot_ids if rid != presser_id]
        screen_x = own_goal_x - own_goal_sign * (_DEFENSE_AREA_DEPTH + _SCREEN_OFFSET)
        center_y = ball_p.y * _SHIFT_FACTOR
        lane_limit = half_width - 0.7
        spreads = _LANE_SPREADS[min(len(screen_ids), len(_LANE_SPREADS) - 1)]
        for rid, offset in zip(sorted(screen_ids), spreads):
            target_y = max(-lane_limit, min(lane_limit, center_y + offset))
            commands[rid] = go_to_point(game, ctx.motion_controller, rid, (screen_x, target_y))

        return commands, mem
