"""The `split_outlet_wall` kernel strategy: `split_outlet`'s slots plus `split_wall`'s free-kick wall.

Combines the two best ideas of the 2026-10-11 search. Copied, not imported, from the
strategy modules (strategy modules do not import each other); keep the copies identical:

- `_split_outlet_picker` and the tactic slots (attack `LeadAndSupportTactic`, outlet
  `OutletTactic`, defense `ShadowAndMarkTactic`) are copied from `split_outlet.py`.
- The wall geometry (`WALL_*`, `MARK_GOAL_SIDE_M`, `LEFTOVER_SPACING_M`, `_unit`,
  `_spread`, `wall_points`, `marker_point`, `leftover_points`, `_legal_target`,
  `_own_goal_centre`, `opponent_free_kick_targets`) and the referee override
  (`_restart_override`) are copied from `split_wall.py`.

Design point: at the OPPONENT's direct free kick the wall override takes over every
outfield robot, the outlet robot included, so the outlet is one of the wall or markers
for that restart. Once the restart is released the picker takes over again as in
`split_outlet`. Our own direct free kick runs the built-in step unchanged.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional

from utama_core.config.referee_constants import BALL_KEEP_OUT_DISTANCE
from utama_core.custom_referee.actions import (
    DirectFreeOursStep,
    _clear_to_legal_positions,
)
from utama_core.engine.context import TickContext
from utama_core.engine.referee_override import RefereeActionOverride
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.engine.tactic import RobotId
from utama_core.entities.data.object import TeamType
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.shared.pass_and_score_geometry import (
    clamp_outside_own_defense_area,
    clamp_to_field,
)
from utama_core.skills.src.utils.move_utils import move
from utama_core.strategy.pickers import carrier_first
from utama_core.tactics.lead_and_support import LeadAndSupportTactic
from utama_core.tactics.outlet import OutletTactic
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic

# ---------------------------------------------------------------------------
# Possession split with an outlet slot (copied from split_outlet.py).
# ---------------------------------------------------------------------------


def _split_outlet_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    available_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """`split_shape`'s split, with an outlet slot taken from the defence when we lack the ball edge.

    Counts (n = robots): with the ball edge, or with fewer than 3 robots, the split is
    `split_shape`'s exactly and the outlet slot is never used. Otherwise attack 1,
    outlet 1, defence n - 2. A slot missing from `available_tactic_ids` hands its
    robots to defence, then attack, then outlet.

    Only emits a key for a slot it assigns robots to, never a zero-robot entry (see
    `split_shape`'s picker for why).
    """
    ordered = carrier_first(game, free_robots)
    if not ordered:
        return {}
    n = len(ordered)

    _friendly_closest, friendly_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.FRIENDLY)
    _enemy_closest, enemy_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.ENEMY)
    friendly_has_ball_edge = friendly_dist < enemy_dist

    if friendly_has_ball_edge or n < 3:
        attack_count = (n + 1) // 2 + 1 if friendly_has_ball_edge else n // 2 - 1
        attack_count = max(0, min(n, attack_count))
        counts = {"attack": attack_count, "outlet": 0, "defense": n - attack_count}
    else:
        counts = {"attack": 1, "outlet": 1, "defense": n - 2}

    available = {slot: count for slot, count in counts.items() if slot in available_tactic_ids and count > 0}
    spill = sum(count for slot, count in counts.items() if slot not in available_tactic_ids)
    if spill:
        target = next((slot for slot in ("defense", "attack", "outlet") if slot in available), None)
        if target is None:
            return {}
        available[target] = available.get(target, 0) + spill

    partition: dict[str, frozenset[RobotId]] = {}
    start = 0
    for slot in ("attack", "outlet", "defense"):
        count = available.get(slot, 0)
        if count > 0:
            partition[slot] = frozenset(ordered[start : start + count])
            start += count
    return partition


# ---------------------------------------------------------------------------
# Opponent direct free kick: wall + marking geometry (copied from split_wall.py).
# ---------------------------------------------------------------------------

# Wall sits 0.1 m beyond the built-in keep-out radius (0.8 m), well clear of the 0.5 m
# foul line, and still close enough to block a dribble straight at goal.
WALL_DISTANCE_M = BALL_KEEP_OUT_DISTANCE + 0.1
WALL_SPACING_M = 0.2
MARK_GOAL_SIDE_M = 0.5
LEFTOVER_SPACING_M = 0.4
# Every restart target stays at least this far from the ball (see `_legal_target`).
TARGET_KEEP_OUT_M = BALL_KEEP_OUT_DISTANCE
_RING_STEPS = 360


def _own_goal_centre(game: Game) -> Vector2D:
    return Vector2D(float(game.field.my_goal_line[0][0]), 0.0)


def _unit(start: Vector2D, end: Vector2D) -> tuple[float, float]:
    dx, dy = end.x - start.x, end.y - start.y
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 1.0, 0.0
    return dx / length, dy / length


def _spread(count: int, spacing: float) -> list[float]:
    """Offsets centred on zero, ascending, `spacing` apart."""
    return [(i - (count - 1) / 2.0) * spacing for i in range(count)]


def wall_points(ball: Vector2D, goal: Vector2D, count: int) -> list[Vector2D]:
    """`count` points side by side, WALL_DISTANCE_M from the ball on the ball->goal line.

    Ordered by ascending offset along the perpendicular to that line.
    """
    ux, uy = _unit(ball, goal)
    cx, cy = ball.x + ux * WALL_DISTANCE_M, ball.y + uy * WALL_DISTANCE_M
    px, py = -uy, ux
    return [Vector2D(cx + px * off, cy + py * off) for off in _spread(count, WALL_SPACING_M)]


def marker_point(man: Vector2D, goal: Vector2D) -> Vector2D:
    """MARK_GOAL_SIDE_M from the marked enemy, on the enemy->our-goal line."""
    ux, uy = _unit(man, goal)
    return Vector2D(man.x + ux * MARK_GOAL_SIDE_M, man.y + uy * MARK_GOAL_SIDE_M)


def leftover_points(ball: Vector2D, goal: Vector2D, front_x: float, count: int) -> list[Vector2D]:
    """`count` points on the defense-area front edge where the ball->goal line crosses it.

    Spread along the edge (perpendicular to the line) LEFTOVER_SPACING_M apart.
    """
    dx = goal.x - ball.x
    if abs(dx) < 1e-9:
        base = ball
    else:
        t = min(1.0, max(0.0, (front_x - ball.x) / dx))
        base = Vector2D(ball.x + t * dx, ball.y + t * (goal.y - ball.y))
    ux, uy = _unit(ball, goal)
    px, py = -uy, ux
    return [Vector2D(base.x + px * off, base.y + py * off) for off in _spread(count, LEFTOVER_SPACING_M)]


def _is_legal_spot(game: Game, point: Vector2D) -> bool:
    """True when `point` is already inside the field and outside our defense area."""
    return clamp_outside_own_defense_area(game, clamp_to_field(point, game)).distance_to(point) < 1e-9


def _legal_target(game: Game, point: Vector2D, ball: Vector2D, taken: tuple[Vector2D, ...] = ()) -> Vector2D:
    """Inside the field, outside our defense area (DefenseAreaRule), and TARGET_KEEP_OUT_M from the ball.

    The box clamp can pull a point toward the ball. With the ball near the box front edge the
    wall (0.9 m out on the goal line) lands behind the edge and is clamped to ~0.43 m from the
    ball, inside the 0.5 m keep-out foul radius. So a target still inside the keep-out circle
    moves to the nearest legal spot on that circle that no earlier target in `taken` claims
    (WALL_SPACING_M apart). Several targets crowd the same stretch of box edge, so sliding
    them all onto one line would stack robots on one spot.
    """
    legal = clamp_outside_own_defense_area(game, clamp_to_field(point, game))
    if legal.distance_to(ball) >= TARGET_KEEP_OUT_M:
        return legal
    best, best_d = legal, math.inf
    for step in range(_RING_STEPS):
        angle = 2.0 * math.pi * step / _RING_STEPS
        spot = Vector2D(ball.x + TARGET_KEEP_OUT_M * math.cos(angle), ball.y + TARGET_KEEP_OUT_M * math.sin(angle))
        if not _is_legal_spot(game, spot) or any(spot.distance_to(t) < WALL_SPACING_M for t in taken):
            continue
        d = spot.distance_to(legal)
        if d < best_d:
            best, best_d = spot, d
    return best


def opponent_free_kick_targets(game: Game, outfield_ids) -> dict[RobotId, tuple[Vector2D, float]]:
    """Target position and orientation for each present outfield robot at an opponent direct free kick.

    Two robots nearest the ball form the wall. The kicker is the enemy nearest the ball;
    the other enemies, nearest our goal first, each get one marker (nearest free robot to
    its mark point). Robots left over go to the defense-area edge on the ball->goal line.
    Requires `game.ball`.
    """
    ball = game.ball.p.to_2d()
    goal = _own_goal_centre(game)
    robots = [rid for rid in outfield_ids if rid in game.friendly_robots]
    if not robots:
        return {}

    by_ball = sorted(robots, key=lambda rid: game.friendly_robots[rid].p.distance_to(ball))
    wall_ids, free = by_ball[:2], by_ball[2:]
    targets: dict[RobotId, tuple[Vector2D, float]] = {}

    # Wall: robots ordered by their side of the ball so the two do not cross.
    px, py = -_unit(ball, goal)[1], _unit(ball, goal)[0]
    wall_ids = sorted(
        wall_ids,
        key=lambda rid: (game.friendly_robots[rid].p.x - ball.x) * px + (game.friendly_robots[rid].p.y - ball.y) * py,
    )
    for rid, point in zip(wall_ids, wall_points(ball, goal, len(wall_ids))):
        legal = _legal_target(game, point, ball, tuple(p for p, _ in targets.values()))
        targets[rid] = (legal, legal.angle_to(ball))

    enemies = game.enemy_robots
    if enemies:
        kicker = min(enemies, key=lambda eid: enemies[eid].p.distance_to(ball))
        markable = sorted(
            (eid for eid in enemies if eid != kicker),
            key=lambda eid: enemies[eid].p.distance_to(goal),
        )
        for eid in markable:
            if not free:
                break
            man = enemies[eid].p
            mark = _legal_target(game, marker_point(man, goal), ball, tuple(p for p, _ in targets.values()))
            rid = min(free, key=lambda r: game.friendly_robots[r].p.distance_to(mark))
            free.remove(rid)
            targets[rid] = (mark, mark.angle_to(man))

    front_x = float(game.field.my_defense_area[1][0])
    for rid, point in zip(free, leftover_points(ball, goal, front_x, len(free))):
        legal = _legal_target(game, point, ball, tuple(p for p, _ in targets.values()))
        targets[rid] = (legal, legal.angle_to(ball))

    return targets


def _restart_override(command_is_yellow: bool, outfield_ids, own_step: DirectFreeOursStep) -> RefereeActionOverride:
    """Override for the raw command `DIRECT_FREE_{YELLOW,BLUE}`, which fires for both teams' kicks.

    The lookup is by raw colour, so this callable decides by `game.my_team_is_yellow`
    which team the kick belongs to. Our own kick runs the built-in `DirectFreeOursStep`
    unchanged. The opponent's kick gets the wall for outfield robots and the built-in
    keep-out clearance for the goalkeeper.
    """

    def override(game: Game, motion_controller: MotionController):
        shim = SimpleNamespace(game=game, motion_controller=motion_controller, cmd_map={})
        if game.my_team_is_yellow == command_is_yellow:
            own_step.blackboard = shim
            own_step.update()
            return dict(shim.cmd_map)

        if game.ball is None:
            _clear_to_legal_positions(shim, ball_keep_dist=BALL_KEEP_OUT_DISTANCE)
            return dict(shim.cmd_map)

        for rid, (target, oren) in opponent_free_kick_targets(game, outfield_ids).items():
            shim.cmd_map[rid] = move(game, motion_controller, rid, target, oren)
        _clear_to_legal_positions(
            shim,
            ball_keep_dist=BALL_KEEP_OUT_DISTANCE,
            exempt_robot_ids=set(outfield_ids),
        )
        return dict(shim.cmd_map)

    return override


def build_split_outlet_wall_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) split-outlet-wall `Strategy` factory.

    `split_outlet`'s tactics and picker, plus `split_wall`'s wall-and-mark referee
    override at the opponent's direct free kick (see module docstring).

    Returns a `build_kernel_strategy(motion_controller)` callable suitable for
    `AbstractStrategy`'s constructor argument of the same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        own_step = DirectFreeOursStep()
        overrides = {
            RefereeCommand.DIRECT_FREE_YELLOW: _restart_override(True, outfield_robot_ids, own_step),
            RefereeCommand.DIRECT_FREE_BLUE: _restart_override(False, outfield_robot_ids, own_step),
        }
        return KernelSchedulerStrategy(
            tactics={
                "attack": LeadAndSupportTactic(),
                "outlet": OutletTactic(),
                "defense": ShadowAndMarkTactic(),
            },
            partitioner=_split_outlet_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
            referee_overrides=overrides,
        )

    return _build
