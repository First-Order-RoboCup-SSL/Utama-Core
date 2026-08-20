import math
from typing import Optional

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.skills.src.utils.move_utils import move

# Overshoot past ball center so the DWA keeps driving until the robot makes
# contact. The dribbler-on approach can push further; the dribbler-off approach
# just nudges past the ball center instead of settling short.
_APPROACH_OVERSHOOT_M = ROBOT_RADIUS * 0
_DRIBBLE_OVERSHOOT_M = ROBOT_RADIUS * (1 / 10)

# An enemy within this range of the ball is treated as "contesting" it —
# close enough that a straight-line approach from our own current position
# would converge on the enemy's body rather than open ball, the mechanism
# behind the default_vs_lowblock investigation's pin (two robots converging
# on the same point from opposite sides wedge at this rough distance apart,
# never reaching the ball itself). See `docs/investigation_default_vs_lowblock_stalemate.md`,
# fix candidate #1.
_CONTEST_RANGE = 0.5


def _target_past_ball(ball: Vector2D, approach_oren: float, overshoot_distance: float) -> Vector2D:
    if overshoot_distance <= 0.0:
        return ball

    dx = math.cos(approach_oren)
    dy = math.sin(approach_oren)
    return Vector2D(ball.x + dx * overshoot_distance, ball.y + dy * overshoot_distance)


def _nearest_contesting_enemy(game: Game, ball: Vector2D) -> Optional[Vector2D]:
    """Position of the closest enemy within `_CONTEST_RANGE` of the ball, if any."""
    nearest_pos, nearest_dist = None, None
    for enemy in game.enemy_robots.values():
        if enemy is None:
            continue
        dist = enemy.p.distance_to(ball)
        if dist <= _CONTEST_RANGE and (nearest_dist is None or dist < nearest_dist):
            nearest_pos, nearest_dist = enemy.p, dist
    return nearest_pos


def go_to_ball(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    dribble_when_near: bool = True,
    dribble_threshold: float = 0.5,
) -> RobotCommand:
    ball = game.ball.p.to_2d()
    robot = game.friendly_robots[robot_id].p

    contesting_enemy = _nearest_contesting_enemy(game, ball)
    if contesting_enemy is not None:
        # Approach from the far side of the ball relative to the contesting
        # enemy — our body ends up between the enemy and the ball (a shield),
        # instead of a straight line from our own position that, against an
        # enemy also converging on the ball, wedges both robots a fixed
        # distance short of it and never actually reaches the ball (the
        # default_vs_lowblock pin).
        approach_oren = contesting_enemy.angle_to(ball)
    else:
        approach_oren = robot.angle_to(ball)

    # Kicker/dribbler is on the back of the robot; approach with back facing ball.
    target_oren = (approach_oren + math.pi) % (2 * math.pi) - math.pi

    # Dribbler runs the whole approach so it is already spinning at contact.
    dribbling = dribble_when_near

    overshoot_distance = _DRIBBLE_OVERSHOOT_M if dribbling else _APPROACH_OVERSHOOT_M
    target = _target_past_ball(ball, approach_oren, overshoot_distance)

    return move(
        game=game,
        motion_controller=motion_controller,
        robot_id=robot_id,
        target_coords=target,
        target_oren=target_oren,
        dribbling=dribbling,
    )
