from typing import Type

from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.motion_planning.src.controllers import DWAController, PIDController


CONTROL_SCHEME_MAP = {"pid": PIDController, "dwa": DWAController}

def get_control_scheme(scheme_name: str) -> Type[MotionController]:
    """Get the control scheme class based on the scheme name."""
    scheme_class = CONTROL_SCHEME_MAP.get(scheme_name.lower())
    if not scheme_class:
        raise ValueError(f"Unknown control scheme: {scheme_name}")
    return scheme_class
