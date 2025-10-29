from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from utama_core.config.settings import TIMESTEP
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.dwa.config import DynamicWindowConfig
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv

if TYPE_CHECKING:
    from utama_core.motion_planning.src.dwa.planner import DynamicWindowPlanner


class AbstractDynamicWindowController(ABC):
    """Base class for Dynamic Window controllers."""

    def __init__(self, config: DynamicWindowConfig, env: SSLStandardEnv | None = None):
        self._planner_config = config
        self.env: SSLStandardEnv | None = env
        self._control_period = TIMESTEP
        self._planner: Optional["DynamicWindowPlanner"] = None

    @abstractmethod
    def calculate(
        self,
        game: Game,
        target: Vector2D,
        robot_id: int,
    ) -> Vector2D:
        """Generate the controller output for the given robot."""

    @abstractmethod
    def reset(self, robot_id: int):
        """Reset controller state for the specified robot."""

    def set_debug_env(self, env):
        self.env = env
        if self._planner is not None:
            self._planner.env = env

    def _ensure_planner(self, game: Game) -> "DynamicWindowPlanner":
        if not isinstance(game, Game):
            raise TypeError(f"DWA planner requires a Game instance. {type(game)} given.")

        if self._planner is None:
            self._planner = self._create_planner(game)
        return self._planner

    @abstractmethod
    def _create_planner(self, game: Game) -> "DynamicWindowPlanner":
        """Factory hook for subclasses to build their planner."""
