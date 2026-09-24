"""`kick_upfield` — turn on the spot and kick the held ball straight upfield.

The minimal clearance for a robot that has the ball where holding it is the
wrong thing to do (the keeper in its own box, a defender on the box edge): no
landing-lane scoring (that's `ClearBallTactic`), just get it moving away from
our goal before a carry or hold foul. Shared by `goalkeep` and
`ShadowAndMarkTactic`.
"""

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.shared.pass_and_score_geometry import oriented_towards
from utama_core.skills.src.utils.move_utils import kick, turn_on_spot


def kick_upfield(game: Game, motion_controller: MotionController, robot_id: int):
    """Kick once facing straight upfield (away from our own goal, along the
    field's long axis); until then, turn on the spot with the dribbler on.
    Assumes the ball is on `robot_id`'s dribbler — check contact `has_ball`
    first, a kick without it does nothing."""
    robot = game.friendly_robots[robot_id]
    upfield_sign = -1.0 if game.my_team_is_right else 1.0
    target_oren = robot.p.angle_to(Vector2D(robot.p.x + upfield_sign * 2.0, robot.p.y))
    if oriented_towards(game, robot_id, target_oren):
        return kick()
    return turn_on_spot(
        game=game,
        motion_controller=motion_controller,
        robot_id=robot_id,
        target_oren=target_oren,
        dribbling=True,
    )
