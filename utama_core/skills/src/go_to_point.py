"""`go_to_point` — drive a robot to an arbitrary field position, always facing the ball.

The generic "move here" primitive for when the target isn't the ball itself
(holding a formation spot, a support position, a defensive post). Orientation
is not a parameter: the robot always faces the ball via `face_ball` — a robot
that needs to hold a specific orientation while stationary (e.g. facing a
pass target, not the ball) must call `move()` directly with an explicit
`target_oren` instead. This has bitten more than one tactic (see
`docs/STRATEGY_DEVELOPMENT.md`'s "Writing a Tactic" section) — grep every
`go_to_point(` call in a new tactic and check whether orientation actually
matters at that call site.
"""

from typing import Tuple

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.skills.src.utils.move_utils import face_ball, move


def go_to_point(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    target_coords: Vector2D,
    dribbling: bool = False,
) -> RobotCommand:
    """Move `robot_id` to `target_coords`, facing the ball throughout (see module docstring)."""
    return move(
        game=game,
        motion_controller=motion_controller,
        robot_id=robot_id,
        target_coords=target_coords,
        target_oren=face_ball(game.friendly_robots[robot_id].p, game.ball.p),
        dribbling=dribbling,
    )
