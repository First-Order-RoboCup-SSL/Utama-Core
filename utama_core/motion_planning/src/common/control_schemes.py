from enum import Enum
from typing import Type

from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.motion_planning.src.controllers import (
    DWAController,
    MPCCppController,
    PIDController,
)


class ControlScheme(Enum):
    """Available motion control schemes."""

    PID = "pid"
    DWA = "dwa"
    MPC = "mpc"
    MPC_CPP = "mpc-cpp"

    def get_controller(self) -> Type[MotionController]:
        """Get the controller class for this scheme."""
        if self == ControlScheme.PID:
            return PIDController
        elif self == ControlScheme.DWA:
            return DWAController
        elif self == ControlScheme.MPC:
            from utama_core.motion_planning.src.controllers.mpc_controller import (
                MPCController,
            )

            return MPCController
        elif self == ControlScheme.MPC_CPP:
            return MPCCppController

        raise ValueError(f"Unknown control scheme: {self}")
