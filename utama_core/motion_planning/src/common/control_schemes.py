from typing import Type

from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.motion_planning.src.controllers import (
    DWAController,
    MPCCppController,
    PIDController,
)


def get_control_scheme(scheme_name: str) -> Type[MotionController]:
    """
    Get the control scheme class based on the scheme name.
    """
    scheme = scheme_name.lower()

    if scheme == "pid":
        return PIDController
    elif scheme == "dwa":
        return DWAController
    elif scheme == "mpc":
        # Local import for Python MPC
        from utama_core.motion_planning.src.controllers.mpc_controller import (
            MPCController,
        )

        return MPCController
    elif scheme == "mpc-cpp":
        # The C++ Wrapper we just made
        return MPCCppController

    raise ValueError(f"Unknown control scheme: {scheme_name}")
