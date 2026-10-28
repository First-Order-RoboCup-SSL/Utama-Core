# ruff: noqa: E402
"""examples/save_and_counter.py — a 1v1 game: a keeper that saves takes the ball and counterattacks.

Each robot keeps its own goal and attacks the other's.

Run:
    pixi run python examples/save_and_counter.py            # real robots (MODE below)
    pixi run python examples/save_and_counter.py --mode rsim
    pixi run python examples/save_and_counter.py --mode grsim   # grSim must be running
    # the dashboard is at http://localhost:8080 (--port to change): Force Start to play, Halt to freeze

What it does:
  - Two robots, both ours on one radio, one per team, a goal each: yellow defends -x and attacks
    +x, blue the other way. Neither is "the attacker": each reads from the world, every tick,
    whether it attacks or keeps (`play_role`), and both read the same world, so they agree.
  - The robot with the ball, or nearer a still ball, attacks: it carries the ball upfield to
    `SHOOT_RANGE_M` from the goal at `CARRY_SPEED_MPS`, then shoots as `LeadAndSupportTactic`
    does (turns to the open corner, or moves for a better shot when the keeper covers it). The
    other keeps its goal (`GoalkeeperTactic`). A save the keeper holds, or a ball it gets to
    first, makes it the attacker, and the robot that shot runs back to keep its own goal.
  - A ball rolling towards a robot's goal (a shot, a clearance) has that robot keeping and the
    other attacking, so nobody leaves the goal to chase a shot. A ball outside the field (or
    unseen) holds both robots until the operator puts it back.
  - The field, the robot hardware type and the referee rules are `shooting_drill.py`'s
    (`FIELD`, `TestRobot`, `RULES`): only goals are detected. After a goal play stops; put the
    ball back and Force Start. Both goals must be in view (`CAMERA_HALF = "full"`).
  - Every mode starts in HALT; Ctrl+C sends stop commands to the robots before exiting. The Live
    tab's Strategy panel can stand either robot still or restart it, while halted. The field
    check must be all green or amber before Force Start (docs/real_robot_setup.md).
"""

import sys
from pathlib import Path

# Run as `pixi run python examples/<this>.py`: utama_core is not installed, so put the repo on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import math
import time
from typing import Optional

from shooting_drill import (
    BLUE_0,
    FIELD,
    YELLOW_1,
    TestRobot,
    _field_bounds,
    _one_robot_strategy,
    _referee_profile,
)

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.custom_referee import CustomReferee
from utama_core.dashboard import attach_dashboard
from utama_core.dashboard.views import referee as referee_view
from utama_core.dashboard.views import strategy as strategy_view
from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run import StrategyRunner
from utama_core.shared.pass_and_score_geometry import (
    enemy_goal_line,
    has_ball,
    oriented_towards,
)
from utama_core.skills.src.utils.move_utils import empty_command, move, turn_on_spot
from utama_core.tactics._pass_and_score import FACE_GOAL_RAD
from utama_core.tactics.goalkeeper import GoalkeeperTactic
from utama_core.tactics.lead_and_support import LeadAndSupportTactic
from utama_core.tactics.scripted_moves import ScriptedMovesTactic

# ---------------------------------------------------------------------------
# Test-day configuration
# ---------------------------------------------------------------------------

MODE = "real"  # "real", "rsim" or "grsim"; --mode overrides it

# Swapping a robot is a change here only. They must wear different colours (one per team).
ROBOT_A = YELLOW_1
ROBOT_B = BLUE_0
ROBOT_A_COLOUR = "yellow"  # robot B is the other colour

# The attacker carries the ball to this far from the goal line before it looks for a shot. The
# robot's aim error is an angle, so the goal it can hit past a keeper grows as it gets closer:
# from 1.2 m a keeper on the shot bisector left no gap a robot 0.05 rad off could count on, and
# the attacker moved round for one until the round timed out (rsim, 5 rounds in 45). From 0.7 m:
# 63 goals in 64 rounds.
SHOOT_RANGE_M = 0.7
# Carrying speed. At the real cap (1 m/s) a placer slowing down lost the ball off its dribbler
# (2026-10-27, `BallPlacementOursStep._CARRY_SPEED_MPS`).
CARRY_SPEED_MPS = 0.5

# With a camera (real, grsim), record every run as an SSL game log in replays/field/.
RECORD = True

DASHBOARD_PORT = 8080  # --port overrides it

# ---------------------------------------------------------------------------

_ROLLING_MPS = 0.5  # a ball faster than this is a shot or clearance under way
_TIE_M = 0.05  # nearer the still ball by less than this is a tie, won by the yellow robot
_CONTROL_M = ROBOT_RADIUS + 0.06  # an opponent's centre this near the ball has it, or is about to
_CARRY_INTO_RANGE_M = 0.1  # the carry ends this far inside `SHOOT_RANGE_M`


def _ball_in_field(game: Game) -> bool:
    ball = game.ball.p
    return abs(ball.x) <= game.field.half_length and abs(ball.y) <= game.field.half_width


def play_role(game: Game, robot_id: int) -> str:
    """What `robot_id` does this tick; both robots compute it from the same world, so they agree.

    "hold": no ball, no opponent, or the ball out of the field (the operator puts it back).
    "attack": it has the ball; or the ball rolls away from its goal; or it is nearer the still ball.
    "keep": the opponent has the ball; or the ball rolls towards its goal; or the opponent is
    nearer the still ball.
    """
    if game.ball is None or not game.enemy_robots or not _ball_in_field(game):
        return "hold"
    me = game.friendly_robots[robot_id]
    ball = game.ball.p.to_2d()
    opponent = min(game.enemy_robots.values(), key=lambda r: r.p.distance_to(ball))
    if has_ball(game, robot_id, visual=True):
        return "attack"
    if opponent.p.distance_to(ball) < _CONTROL_M:
        return "keep"
    velocity = Vector2D(game.ball.v.x, game.ball.v.y)
    if velocity.mag() >= _ROLLING_MPS:
        own_goal_sign = 1.0 if game.my_team_is_right else -1.0
        return "keep" if velocity.x * own_goal_sign > 0 else "attack"
    margin = me.p.distance_to(ball) - opponent.p.distance_to(ball)
    if margin < -_TIE_M or (abs(margin) <= _TIE_M and game.my_team_is_yellow):
        return "attack"
    return "keep"


def _capped(command: RobotCommand, max_speed: float) -> RobotCommand:
    """`command` with its driving speed scaled down to at most `max_speed`, direction kept."""
    speed = math.hypot(command.local_forward_vel, command.local_left_vel)
    if speed <= max_speed:
        return command
    scale = max_speed / speed
    return command._replace(
        local_forward_vel=command.local_forward_vel * scale, local_left_vel=command.local_left_vel * scale
    )


class KeepOrAttackTactic(BaseTactic[None]):
    """One robot of the 1v1: keeps its goal or attacks the other, as `play_role` says each tick.
    Its keeper and its shooter are the existing tactics; each restarts when the robot takes it up."""

    tag = TacticTag.ATTACK

    def __init__(self):
        self._role: Optional[str] = None
        self._keeper: Optional[GoalkeeperTactic] = None
        self._keeper_mem = None
        self._shooter = LeadAndSupportTactic()
        self._shooter_mem = None

    def initial_mem(self) -> None:
        return None

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: None
    ) -> tuple[dict[RobotId, RobotCommand], None]:
        rid = robot_ids[0]
        role = play_role(game, rid)
        ctx.trace(game, f"save_and_counter[{rid}].role", role)
        taken_up = role != self._role
        self._role = role
        if role == "hold":
            return {rid: empty_command()}, mem
        if role == "keep":
            if taken_up or self._keeper is None or self._keeper.robot_id != rid:
                self._keeper = GoalkeeperTactic(rid)
                self._keeper_mem = self._keeper.initial_mem()
            commands, self._keeper_mem = self._keeper.tick(game, ctx, (rid,), self._keeper_mem)
            return commands, mem
        if taken_up:
            self._shooter_mem = self._shooter.initial_mem()
        carry = self._carry(game, ctx, rid)
        if carry is not None:
            return {rid: carry}, mem
        commands, self._shooter_mem = self._shooter.tick(game, ctx, (rid,), self._shooter_mem)
        return commands, mem

    def _carry(self, game: Game, ctx: TickContext, rid: int) -> Optional[RobotCommand]:
        """With the ball further than `SHOOT_RANGE_M` upfield of the goal line: carry it up,
        facing the goal. None once in range, or without the ball: the shooter takes over.

        Measured along the field, not straight to the goal: the shooter's move for a better shot
        is sideways, and with a straight-line range that move took the robot back out of range
        and the carry brought it back in, in turn. The target is `_CARRY_INTO_RANGE_M` inside the
        range: aimed at the range itself, the robot stopped 1 mm short and stood there (rsim).

        A robot that took the ball facing elsewhere (a keeper's save) pivots on it to face the
        goal first: turning while it drove off sideways lost the ball on the real robots
        (2026-10-28), as in `move_for_a_shot`."""
        if not has_ball(game, rid, visual=True):
            return None
        goal_x, goal_y1, goal_y2 = enemy_goal_line(game)
        goal = Vector2D(goal_x, (goal_y1 + goal_y2) / 2.0)
        me = game.friendly_robots[rid].p
        if abs(goal_x - me.x) <= SHOOT_RANGE_M:
            return None
        goal_oren = me.angle_to(goal)
        if not oriented_towards(game, rid, goal_oren, FACE_GOAL_RAD):
            ctx.trace(game, f"shoot[{rid}]", "facing the goal")
            return turn_on_spot(game, ctx.motion_controller, rid, goal_oren, dribbling=True)
        upfield = math.copysign(1.0, goal_x - me.x)
        target = Vector2D(goal_x - upfield * (SHOOT_RANGE_M - _CARRY_INTO_RANGE_M), me.y)
        command = move(game, ctx.motion_controller, rid, target, goal_oren, dribbling=True)
        ctx.trace(game, f"shoot[{rid}]", "carrying it up")
        return _capped(command, CARRY_SPEED_MPS)


def _still():
    return ScriptedMovesTactic(script=())


TACTICS = {"save_and_counter": KeepOrAttackTactic, "still": _still}


def build_runner(mode: str, port: int = DASHBOARD_PORT) -> StrategyRunner:
    real = mode == "real"
    if ROBOT_A_COLOUR not in ("yellow", "blue"):
        raise ValueError(f"ROBOT_A_COLOUR must be 'yellow' or 'blue', got {ROBOT_A_COLOUR!r}")
    a_is_yellow = ROBOT_A_COLOUR == "yellow"
    # In the simulators each team's robots are numbered from 0.
    a_id = ROBOT_A.vision_id if real else 0
    b_id = ROBOT_B.vision_id if real else 0
    a_map = {ROBOT_A.vision_id: ROBOT_A.firmware_id}
    b_map = {ROBOT_B.vision_id: ROBOT_B.firmware_id}
    a_profiles, b_profiles = {a_id: ROBOT_A.profile}, {b_id: ROBOT_B.profile}

    profile = _referee_profile()
    referee = CustomReferee(profile, n_robots_yellow=1, n_robots_blue=1)
    server = attach_dashboard(port=port)
    referee_view.attach(server, referee, profile)

    game_log = (
        f"replays/field/{time.strftime('%Y%m%d_%H%M%S')}_{mode}_save_and_counter.log"
        if RECORD and mode != "rsim"
        else None
    )
    if game_log:
        print(f"Recording to {game_log}")
    runner = StrategyRunner(
        game_log_path=game_log,
        strategy=_one_robot_strategy(a_id, KeepOrAttackTactic),
        opp_strategy=_one_robot_strategy(b_id, KeepOrAttackTactic),
        my_team_is_yellow=a_is_yellow,
        my_team_is_right=not a_is_yellow,  # yellow defends -x and attacks +x
        mode=mode,
        exp_friendly=1,
        exp_enemy=1,
        full_field_dims=FIELD,
        field_bounds=_field_bounds(FIELD, "full"),  # both goals are in play
        referee=referee,
        referee_initial_command=RefereeCommand.HALT,
        show_live_status=not real,
        yellow_vision_to_cmd_mapping=(a_map if a_is_yellow else b_map) if real else None,
        blue_vision_to_cmd_mapping=(b_map if a_is_yellow else a_map) if real else None,
        yellow_robot_profiles=a_profiles if a_is_yellow else b_profiles,
        blue_robot_profiles=b_profiles if a_is_yellow else a_profiles,
    )
    # The Live tab's Strategy panel: stand either robot still, or restart it, while halted.
    strategy_view.attach(
        server,
        runner,
        choices={
            "my": {n: (lambda t=t: _one_robot_strategy(a_id, t)) for n, t in TACTICS.items()},
            "opp": {n: (lambda t=t: _one_robot_strategy(b_id, t)) for n, t in TACTICS.items()},
        },
        running={"my": "save_and_counter", "opp": "save_and_counter"},
    )
    return runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("real", "rsim", "grsim"), default=MODE)
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT, help="dashboard port (default %(default)s)")
    args = parser.parse_args()
    build_runner(args.mode, args.port).run()


if __name__ == "__main__":
    main()
