"""demo_ball_placement.py — Entry point for the ball placement feature.

Run:
    pixi run python demo_ball_placement.py
    # RSim window opens; open http://localhost:8080 in a browser

What this sets up
-----------------
- Exhibition Road field (4 m × 3 m, ``GREAT_EXHIBITION_FIELD_DIMS``)
- 2v2 format: two yellow robots (your team) vs two blue robots that
  deliberately push the ball out of bounds in a top/bottom/left/right cycle
- CustomReferee pre-configured with:
    - Goal detection enabled (issues PREPARE_KICKOFF → NORMAL_START cycle)
    - Out-of-bounds enabled (issues STOP → BALL_PLACEMENT_YELLOW → DIRECT_FREE)
    - Defense area and keep-out rules OFF (less noise during development)
    - Auto-advance fully enabled so state transitions fire automatically —
      you can watch the full placement → free-kick → restart cycle without
      touching the GUI
- Browser GUI at http://localhost:8080 — use "Manual Commands" to fire
  BALL_PLACEMENT_YELLOW at any time and set the target position

Your task
---------
Open ``utama_core/strategy/examples/ball_placement_strategy.py`` and implement
``BallPlacementStep.update()``.  Everything else here is already wired up.

Workflow
--------
1. Run this script and open http://localhost:8080.
2. Let the blue robots push the ball out of bounds, or use the GUI "Manual
   Commands" panel to issue BALL_PLACEMENT_YELLOW.
3. Watch one of your robots (yellow) drive to the ball, capture it with the
   dribbler, and carry it to the target circle shown in the GUI.
4. Iterate until the test suite passes:
       pixi run pytest utama_core/tests/strategy_runner/test_ball_placement_rsim.py -v
"""

import math

import py_trees

from utama_core.config.field_params import GREAT_EXHIBITION_FIELD_DIMS
from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.profiles.profile_loader import (
    AutoAdvanceConfig,
    DefenseAreaConfig,
    GameConfig,
    GoalDetectionConfig,
    KeepOutConfig,
    OutOfBoundsConfig,
    RefereeProfile,
    RulesConfig,
)
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run import StrategyRunner
from utama_core.skills.src.utils.move_utils import move
from utama_core.strategy.common import AbstractBehaviour
from utama_core.strategy.examples.ball_placement_strategy import BallPlacementStrategy
from utama_core.strategy.examples.deliberate_out_of_bounds_strategy import (
    DeliberateOutOfBoundsStrategy,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GUI_PORT = 8080
N_ROBOTS = 2
MY_TEAM_IS_YELLOW = True
MY_TEAM_IS_RIGHT = True


def _angle_delta(target: float, current: float) -> float:
    return math.atan2(math.sin(target - current), math.cos(target - current))


class KickTowardOpponentStep(AbstractBehaviour):
    """Stay idle except for one kick after a referee-managed restart."""

    _KICK_ALIGNMENT_RAD = 0.25

    def setup_(self):
        self._seen_first_live_restart = False
        self._active_restart_timestamp: float | None = None
        self._kick_sent_for_timestamp: float | None = None

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        motion_controller = self.blackboard.motion_controller
        ref = game.referee

        if ref is None or ref.referee_command != RefereeCommand.NORMAL_START:
            return py_trees.common.Status.RUNNING

        restart_timestamp = ref.referee_command_timestamp
        if not self._seen_first_live_restart:
            self._seen_first_live_restart = True
            self._active_restart_timestamp = restart_timestamp
            return py_trees.common.Status.RUNNING

        if restart_timestamp != self._active_restart_timestamp:
            self._active_restart_timestamp = restart_timestamp
            self._kick_sent_for_timestamp = None

        if self._kick_sent_for_timestamp == restart_timestamp:
            return py_trees.common.Status.RUNNING

        if game.ball is None or not game.friendly_robots:
            return py_trees.common.Status.RUNNING

        kicker_id = min(
            game.friendly_robots,
            key=lambda rid: game.friendly_robots[rid].p.distance_to(game.ball.p),
        )
        kicker = game.friendly_robots[kicker_id]
        ball_pos = Vector2D(game.ball.p.x, game.ball.p.y)
        target_pos = self._opponent_target(game, ball_pos)
        kick_oren = ball_pos.angle_to(target_pos)

        for robot_id in game.friendly_robots:
            if robot_id != kicker_id:
                continue

            if abs(_angle_delta(kick_oren, kicker.orientation)) <= self._KICK_ALIGNMENT_RAD:
                self.blackboard.cmd_map[robot_id] = RobotCommand(
                    local_forward_vel=0,
                    local_left_vel=0,
                    angular_vel=0,
                    kick=1,
                    chip=0,
                    dribble=0,
                )
                self._kick_sent_for_timestamp = restart_timestamp
                continue

            self.blackboard.cmd_map[robot_id] = move(
                game,
                motion_controller,
                robot_id,
                kicker.p,
                kick_oren,
                dribbling=False,
            )

        return py_trees.common.Status.RUNNING

    def _opponent_target(self, game, ball_pos: Vector2D) -> Vector2D:
        if game.enemy_robots:
            target_robot = min(
                game.enemy_robots.values(),
                key=lambda robot: robot.p.distance_to(ball_pos),
            )
            return Vector2D(target_robot.p.x, target_robot.p.y)

        goal_x = -game.field.half_length if game.my_team_is_right else game.field.half_length
        return Vector2D(goal_x, 0.0)


class BallPlacementAndKickStrategy(BallPlacementStrategy):
    """Use the built-in referee ball-placement override, then kick toward blue."""

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        root = py_trees.composites.Sequence(name="PlacementRestartKickRoot", memory=False)
        root.add_child(KickTowardOpponentStep(name="KickTowardOpponent"))
        return root


# ---------------------------------------------------------------------------
# Referee profile
#
# Out-of-bounds is the primary trigger for ball placement in a real game.
# Defense area and keep-out rules are disabled to reduce noise while you're
# developing the placement skill itself.
# ---------------------------------------------------------------------------

_BALL_PLACEMENT_PROFILE = RefereeProfile(
    profile_name="ball_placement_dev",
    rules=RulesConfig(
        goal_detection=GoalDetectionConfig(
            enabled=True,
            cooldown_seconds=1.0,
        ),
        out_of_bounds=OutOfBoundsConfig(
            enabled=True,
            free_kick_assigner="last_touch",
        ),
        defense_area=DefenseAreaConfig(
            enabled=False,
            max_defenders=1,
            attacker_infringement=False,
        ),
        keep_out=KeepOutConfig(
            enabled=False,
            radius_meters=0.3,
            violation_persistence_frames=30,
        ),
    ),
    game=GameConfig(
        half_duration_seconds=300.0,
        kickoff_team="yellow",
        force_start_after_goal=False,
        stop_duration_seconds=2.0,
        prepare_duration_seconds=3.0,
        kickoff_timeout_seconds=10.0,
        auto_advance=AutoAdvanceConfig(
            # All auto-advance enabled: state machine drives itself so you can
            # observe the full placement → free-kick → normal-start cycle.
            stop_to_next_command=True,
            prepare_kickoff_to_normal=True,
            prepare_penalty_to_normal=True,
            direct_free_to_normal=True,
            ball_placement_to_next=True,
            normal_start_to_force=True,
        ),
    ),
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    referee = CustomReferee(
        _BALL_PLACEMENT_PROFILE,
        n_robots_yellow=N_ROBOTS,
        n_robots_blue=N_ROBOTS,
        enable_gui=True,
        gui_port=GUI_PORT,
    )

    runner = StrategyRunner(
        strategy=BallPlacementAndKickStrategy(),
        # Opponents deliberately create out-of-bounds events in a repeatable
        # top -> bottom -> left -> right cycle.
        opp_strategy=DeliberateOutOfBoundsStrategy(field_dims=GREAT_EXHIBITION_FIELD_DIMS),
        my_team_is_yellow=MY_TEAM_IS_YELLOW,
        my_team_is_right=MY_TEAM_IS_RIGHT,
        mode="rsim",
        control_scheme="pid",
        exp_friendly=N_ROBOTS,
        exp_enemy=N_ROBOTS,
        full_field_dims=GREAT_EXHIBITION_FIELD_DIMS,
        referee=referee,
        show_live_status=True,
    )

    runner.run()


if __name__ == "__main__":
    main()
