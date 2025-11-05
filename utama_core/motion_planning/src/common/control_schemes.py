from typing import Type

from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.motion_planning.src.controllers import DWAController, PIDController


CONTROL_SCHEME_MAP = {"pid": PIDController, "dwa": DWAController}

def get_control_scheme(scheme_name: str) -> Type[MotionController]:
    """
    Get the control scheme class based on the scheme name.

    Args:
        scheme_name: The name of the control scheme ('pid' or 'dwa').

    Returns:
        The control scheme class (PIDController or DWAController).

    Raises:
        ValueError: If the scheme_name is not recognized.
    """
    scheme_class = CONTROL_SCHEME_MAP.get(scheme_name.lower())
    if not scheme_class:
        raise ValueError(f"Unknown control scheme: {scheme_name}")
    return scheme_class
