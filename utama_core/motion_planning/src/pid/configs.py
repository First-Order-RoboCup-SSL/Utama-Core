from dataclasses import dataclass
from typing import Union

from utama_core.config.enums import Mode
from utama_core.config.robot_params import GRSIM_PARAMS, REAL_PARAMS, RSIM_PARAMS
from utama_core.config.settings import TIMESTEP


@dataclass(slots=True)
class OrientationPIDConfigs:
    dt: float = TIMESTEP
    max_output: Union[float, None] = None
    min_output: Union[float, None] = None
    kp: float = 0.0
    kd: float = 0.0
    ki: float = 0.0
    integral_min: Union[float, None] = None
    integral_max: Union[float, None] = None
    max_acceleration: float = 0.0


@dataclass(slots=True)
class TranslationPIDConfigs:
    dt: float = TIMESTEP
    max_velocity: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    ki: float = 0.0
    integral_min: Union[float, None] = None
    integral_max: Union[float, None] = None
    max_acceleration: float = 0.0


@dataclass(slots=True)
class PIDConfigs:
    orientation: OrientationPIDConfigs
    translation: TranslationPIDConfigs


def get_pid_configs(mode: Mode) -> PIDConfigs:
    """Return the PID configuration for a given simulator mode."""
    if mode == Mode.RSIM:
        return PIDConfigs(
            orientation=OrientationPIDConfigs(
                max_output=RSIM_PARAMS.MAX_ANGULAR_VEL,
                min_output=-RSIM_PARAMS.MAX_ANGULAR_VEL,
                kp=4.5,
                kd=0.02,
                ki=0.0,
                integral_min=-10,
                integral_max=10,
                max_acceleration=RSIM_PARAMS.MAX_ANGULAR_ACCELERATION,
            ),
            translation=TranslationPIDConfigs(
                max_velocity=RSIM_PARAMS.MAX_VEL,
                kp=1.8,
                kd=0.025,
                ki=0.0,
                integral_min=-5,
                integral_max=5,
                max_acceleration=RSIM_PARAMS.MAX_ACCELERATION,
            ),
        )
    if mode == Mode.GRSIM:
        return PIDConfigs(
            orientation=OrientationPIDConfigs(
                max_output=GRSIM_PARAMS.MAX_ANGULAR_VEL,
                min_output=-GRSIM_PARAMS.MAX_ANGULAR_VEL,
                kp=4.5,
                kd=0.02,
                ki=0.0,
                integral_min=-10,
                integral_max=10,
                max_acceleration=GRSIM_PARAMS.MAX_ANGULAR_ACCELERATION,
            ),
            translation=TranslationPIDConfigs(
                max_velocity=GRSIM_PARAMS.MAX_VEL,
                kp=1.8,
                kd=0.025,
                ki=0.0,
                integral_min=-5,
                integral_max=5,
                max_acceleration=GRSIM_PARAMS.MAX_ACCELERATION,
            ),
        )
    if mode == Mode.REAL:
        return PIDConfigs(
            orientation=OrientationPIDConfigs(
                max_output=REAL_PARAMS.MAX_ANGULAR_VEL,
                min_output=-REAL_PARAMS.MAX_ANGULAR_VEL,
                kp=0.5,
                kd=0.075,
                ki=0.0,
                max_acceleration=REAL_PARAMS.MAX_ANGULAR_ACCELERATION,
            ),
            translation=TranslationPIDConfigs(
                max_velocity=REAL_PARAMS.MAX_VEL,
                kp=1.8,
                kd=0.025,
                ki=0.0,
                integral_min=-5,
                integral_max=5,
                max_acceleration=REAL_PARAMS.MAX_ACCELERATION,
            ),
        )
    raise ValueError(f"Unknown mode enum: {mode}.")
