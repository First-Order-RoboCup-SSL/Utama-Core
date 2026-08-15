"""`AbstractStrategy` — base class for kernel-tactic strategies run by `StrategyRunner`.

`StrategyRunner` drives a strategy through a fixed contract: `load_rsim_env`,
`load_robot_controller`, `load_motion_controller`, `load_game`, `assert_exp_robots`,
`assert_exp_goals`, and once per tick, `step()`. A concrete `AbstractStrategy` wraps a
`kernel.Strategy` (see `utama_core.kernel.strategy`): `step()` ticks that `Strategy` (plus
the goalkeeper, pinned outside the kernel scheduler) directly every frame.

Robot 0 is always the goalkeeper, ticked directly and never handed to the kernel
`Strategy` — see `tactics/goalkeeper.py` and the "Tactics as Processes" design note,
section 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utama_core.config.enums import Role
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.entities.game.field import FieldBounds
from utama_core.global_utils.math_utils import (
    assert_contains,
    assert_valid_bounding_box,
)
from utama_core.kernel.referee_override import is_override_command
from utama_core.kernel.referee_reset import is_paused
from utama_core.kernel.strategy import Strategy as KernelSchedulerStrategy
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
from utama_core.skills.src.utils.move_utils import empty_command
from utama_core.tactics.goalkeeper import GoalkeeperTactic
from utama_core.team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)


@dataclass(slots=True, frozen=True)
class SpaceRequirements:
    """
    Represents minimum space requirements for a strategy.

    Attributes:
        min_length (float): Minimum required length of the field region.
        min_width (float): Minimum required width of the field region.
    """

    min_length: float
    min_width: float


class AbstractStrategy:
    """Kernel-tactic strategy base class, driven by `StrategyRunner`.

    Args:
        build_kernel_strategy: called once, from `load_motion_controller`, as
            `build_kernel_strategy(motion_controller) -> kernel.Strategy`.
            Deferred to a factory (rather than passed pre-built) only because
            `AbstractStrategy.__init__` itself runs before `StrategyRunner` has
            injected anything — `Strategy.__init__` never reads `game`, only
            `motion_controller`, so the `Strategy` can be built as soon as
            `load_motion_controller` fires, without waiting for `load_game`.
        goalkeeper_id: robot ID pinned to the goalkeeper tactic, outside the
            kernel scheduler. Defaults to 0 per SSL/team convention.
        exp_ball: whether the strategy expects the ball to be present on the
            field. If True, and StrategyRunner's exp_ball is False, StrategyRunner
            will raise an error and not run the strategy. If False, but the ball
            exists, we will just rock on.
    """

    def __init__(
        self,
        build_kernel_strategy,
        goalkeeper_id: int = 0,
        exp_ball: bool = True,
    ):
        self.exp_ball = exp_ball
        self._build_kernel_strategy = build_kernel_strategy
        self._kernel_strategy: Optional[KernelSchedulerStrategy] = None
        self._goalkeeper = GoalkeeperTactic(robot_id=goalkeeper_id)
        self._goalkeeper_id = goalkeeper_id
        self._goalkeeper_mem = self._goalkeeper.initial_mem()

        ### These attributes are set by the StrategyRunner before the strategy is run. ###
        self.robot_controller: Optional[AbstractRobotController] = None
        self.motion_controller: Optional[MotionController] = None
        self.rsim_env: Optional[SSLBaseEnv] = None
        self.game: Optional[Game] = None

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int) -> bool:
        """Validate that the number of friendly and enemy robots matches the strategy's
        expectations. Kernel strategies accept any robot count by default — override to
        enforce a specific constraint."""
        return True

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool) -> bool:
        """Validate that the field configuration includes the expected goals. Kernel
        strategies accept any goal configuration by default — override to enforce a
        specific constraint."""
        return True

    def get_min_bounding_req(self) -> Optional[FieldBounds | SpaceRequirements]:
        """Return the minimum field region required by the strategy.

        If the strategy only operates within a subset of the full field, return
        a `FieldBounds` object defining that region. Otherwise, return `None`
        to indicate no restriction (the default).

        This method is called during `load_game()`, when `self.game` is already
        populated.

        Note:
            The bounding zone should be defined in field coordinates (i.e., absolute
            positions).
        """
        return None

    def execute_default_action(self, game: Game, role: Role, robot_id: int) -> RobotCommand:
        """
        Provide a fallback command for robots without a tactic-assigned command.

        Invoked once per tick for any robot `step()` didn't otherwise cover.
        Override to implement a safer or more appropriate default behaviour.

        Args:
            game: Current game state snapshot.
            role: Role assigned to the robot, or `Role.UNASSIGNED`.
            robot_id: Identifier of the robot.

        Returns:
            A `RobotCommand` to send to the controller.
        """
        return empty_command(False)

    def load_rsim_env(self, env: SSLBaseEnv):
        """Called by StrategyRunner: store the RSim environment."""
        self.rsim_env = env

    def load_robot_controller(self, robot_controller: AbstractRobotController):
        """Called by StrategyRunner: store the robot controller."""
        self.robot_controller = robot_controller

    def load_motion_controller(self, motion_controller: MotionController):
        """Called by StrategyRunner: store the motion controller and build the
        kernel `Strategy` (see class docstring for why this, not `load_game`,
        is the right timing)."""
        self.motion_controller = motion_controller
        self._kernel_strategy = self._build_kernel_strategy(motion_controller)

    def assert_field_requirements(self, game: Game):
        """
        Assert that the actual field size meets the strategy's requirements,
        that both actual field and min_bounding_zone are within the full field,
        and that bounding boxes are well-formed (top-left above/left of bottom-right).
        """
        actual_field_bounds = game.field.field_bounds
        min_bounding_req = self.get_min_bounding_req()

        # --- Validate min bounding zone ---
        if min_bounding_req is not None:
            # Validate required zone
            if isinstance(min_bounding_req, FieldBounds):
                assert_valid_bounding_box(
                    min_bounding_req,
                    game.field.full_field_half_length,
                    game.field.full_field_half_width,
                )
                # Check containment
                assert_contains(actual_field_bounds, min_bounding_req)
            elif isinstance(min_bounding_req, SpaceRequirements):
                # Check if the actual field is large enough
                actual_length = actual_field_bounds.bottom_right[0] - actual_field_bounds.top_left[0]
                actual_width = actual_field_bounds.top_left[1] - actual_field_bounds.bottom_right[1]
                if actual_length < min_bounding_req.min_length:
                    raise ValueError(
                        "Field bound length too small for strategy. "
                        f"Actual length: {actual_length}, required minimum length: {min_bounding_req.min_length}."
                    )

                if actual_width < min_bounding_req.min_width:
                    raise ValueError(
                        "Field bound width too small for strategy. "
                        f"Actual width: {actual_width}, required minimum width: {min_bounding_req.min_width}."
                    )

    def load_game(self, game: Game):
        """Called by StrategyRunner: store the game object.

        TestManager may reset the game object for a new episode, so this is a
        plain attribute set, re-called every episode, not a one-time init.
        """
        self.game = game
        self.assert_field_requirements(game)

    def step(self):
        game = self.game

        outfield_commands = self._kernel_strategy.tick(game)

        cmd_map: dict[int, RobotCommand] = {}
        cmd_map.update(outfield_commands)

        # During a referee-restart override, `outfield_commands` already covers
        # every friendly robot including the goalkeeper (the override's Step
        # classes compute for all of `game.friendly_robots`, not just the
        # outfield pool) — ticking GoalkeeperTactic on top would overwrite that
        # with normal ball-tracking logic mid-restart.
        #
        # During HALT/STOP, the goalkeeper must stop issuing motion commands
        # for the same reason `Strategy.tick()` freezes the outfield pool via
        # `is_paused` — skipping the tick here falls through to
        # `execute_default_action` below, which returns `empty_command(False)`,
        # the correct "stop" command.
        referee = getattr(game, "referee", None)
        current_command = getattr(referee, "referee_command", None) if referee is not None else None
        if (
            not is_override_command(current_command)
            and not is_paused(current_command)
            and self._goalkeeper_id not in cmd_map
        ):
            gk_commands, self._goalkeeper_mem = self._goalkeeper.tick(
                game, self._kernel_strategy._ctx, (self._goalkeeper_id,), self._goalkeeper_mem
            )
            cmd_map.update(gk_commands)

        for robot_id in game.friendly_robots:
            if robot_id in cmd_map:
                self.robot_controller.add_robot_commands(cmd_map[robot_id], robot_id)
            else:
                role = Role.GOALKEEPER if robot_id == self._goalkeeper_id else Role.UNASSIGNED
                self.robot_controller.add_robot_commands(self.execute_default_action(game, role, robot_id), robot_id)

        self.robot_controller.send_robot_commands()

    def debug_status(self) -> dict[int, list[str]]:
        """Per-robot `["<tactic>", "committed"?]` for GUI display.

        Reports which tactic slot each robot currently belongs to and whether
        that slot is `is_committed()` (the actual "why won't this reassign"
        signal in this model) — used by `StrategyRunner._push_bt_nodes_to_referee`
        for the debug GUI panel.
        """
        game = self.game
        status: dict[int, list[str]] = {self._goalkeeper_id: ["goalkeeper"]}
        for tactic_id, info in self._kernel_strategy.slot_status(game).items():
            label = tactic_id if not info["committed"] else f"{tactic_id} (committed)"
            for robot_id in info["robots"]:
                status[robot_id] = [label]
        return status
