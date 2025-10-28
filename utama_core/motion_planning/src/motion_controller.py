"""Factory for per-robot motion controllers.

The original implementation returned PID controllers. To support alternative
controllers (e.g. Dynamic Window Approach), this module now builds the
requested control scheme while keeping the public API backwards compatible.
"""

from typing import Tuple

from utama_core.entities.game import Game
from utama_core.motion_planning.src.dwa import (
    DWATranslationController,
    get_grsim_dwa_controllers,
    get_real_dwa_controllers,
    get_rsim_dwa_controllers,
)
from utama_core.motion_planning.src.pid.pid import (
    PID,
    TwoDPID,
    get_grsim_pids,
    get_real_pids,
    get_rsim_pids,
)
from utama_core.motion_planning.src.pid.pid_abstract import AbstractPID

ControllerPair = Tuple[AbstractPID, AbstractPID]


class MotionController:
    """Container for the orientation and translation controllers.

    Args:
        mode: Deployment mode ("rsim", "grsim", or "real").
        control_scheme: Controller family to use. Supported values are
            ``"dwa"`` (default) and ``"pid"``.
    """

    def __init__(self, mode: str, control_scheme: str = "dwa", debug_env=None):
        self._mode = mode
        self._control_scheme = control_scheme.lower()
        self._debug_env = debug_env
        self._orientation, self._translation = self._initialise_controllers(mode)
        self._attach_debug_env(self._translation)

    def _initialise_controllers(self, mode: str) -> ControllerPair:
        if self._control_scheme == "dwa":
            return self._initialise_dwa(mode)
        if self._control_scheme == "pid":
            return self._initialise_pid(mode)
        raise ValueError(f"Unknown control scheme '{self._control_scheme}'. Choose from 'dwa' or 'pid'.")

    def _initialise_pid(self, mode: str) -> ControllerPair:
        if mode == "rsim":
            return get_rsim_pids()
        if mode == "grsim":
            return get_grsim_pids()
        if mode == "real":
            return get_real_pids()
        raise ValueError(f"Unknown mode: {mode}. Choose from 'rsim', 'grsim', or 'real'.")

    def _initialise_dwa(self, mode: str) -> ControllerPair:
        if mode == "rsim":
            return get_rsim_dwa_controllers()
        if mode == "grsim":
            return get_grsim_dwa_controllers()
        if mode == "real":
            return get_real_dwa_controllers()
        raise ValueError(f"Unknown mode: {mode}. Choose from 'rsim', 'grsim', or 'real'.")

    def update_game(self, game: Game) -> None:
        """Push the latest game snapshot to controllers that require context."""

        if isinstance(self._translation, DWATranslationController):
            self._translation.update_game(game)

    def _attach_debug_env(self, translation):
        if isinstance(translation, DWATranslationController) and self._debug_env is not None:
            translation.set_debug_env(self._debug_env)

    @property
    def orientation(self) -> AbstractPID:
        return self._orientation

    @property
    def translation(self) -> AbstractPID:
        return self._translation
