"""`man_mark` — stand goal-side of one enemy robot, facing the ball.

The marker stands `MARK_STANDOFF` metres from the opponent toward our own goal, the same
spot `ShadowAndMarkTactic` marks from (`mark_target`). It used to stand 0.5 m beside the
ball-opponent line, which no other marker agreed with: when two markers marked each other
(one `man_mark`, one `ShadowAndMarkTactic`) each target moved with the other robot and the
pair walked to the wall. With one rule on both sides, opposite teams marking each other
have a fixed point (`A = B + 0.6 toward A's goal`, `B = A + 0.6 toward B's goal`).
Called by `PressAndContainTactic` on every marker robot (`robot_ids[1:]`, one enemy each via
`_assign_markers`) while `robot_ids[0]` presses the ball carrier via `block_attacker` — see
`press_and_contain.py`.
"""

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.skills.src.utils.move_utils import face_ball, move

MARK_STANDOFF = 0.6  # metres — mark from this distance on the goal side of the opponent, not on top of them


def mark_target(game: Game, opponent_id: int) -> Vector2D:
    """The point `MARK_STANDOFF` from the opponent, toward our own goal."""
    goal_x = game.field.my_goal_line[0][0]
    opponent = game.enemy_robots[opponent_id]
    # Toward our own goal. The sign was reversed until 2026-10-05: markers stood on the
    # far side, leaving the opponent a clear run at goal.
    direction = -1.0 if goal_x < opponent.p.x else 1.0
    return Vector2D(opponent.p.x + direction * MARK_STANDOFF, opponent.p.y)


def man_mark(game: Game, motion_controller: MotionController, robot_id: int, target_id: int):
    """Move `robot_id` to `mark_target` for `target_id`, facing the ball."""
    robot = game.friendly_robots[robot_id]
    return move(
        game,
        motion_controller,
        robot_id,
        mark_target(game, target_id),
        face_ball(robot.p, game.ball.p.to_2d()),
    )
