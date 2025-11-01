from dataclasses import dataclass
from typing import Union

from utama_core.config.enums import Mode
from utama_core.config.robot_params.grsim import MAX_ANGULAR_VEL, MAX_VEL
from utama_core.config.robot_params.real import MAX_ANGULAR_VEL as REAL_MAX_ANG
from utama_core.config.robot_params.real import MAX_VEL as REAL_MAX_VEL
from utama_core.config.robot_params.rsim import MAX_ANGULAR_VEL as RSIM_MAX_ANG
from utama_core.config.robot_params.rsim import MAX_VEL as RSIM_MAX_VEL
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
                max_output=RSIM_MAX_ANG,
                min_output=-RSIM_MAX_ANG,
                kp=4.5,
                kd=0.02,
                ki=0.0,
                integral_min=-10,
                integral_max=10,
                max_acceleration=50,
            ),
            translation=TranslationPIDConfigs(
                max_velocity=RSIM_MAX_VEL,
                kp=1.8,
                kd=0.025,
                ki=0.0,
                integral_min=-5,
                integral_max=5,
                max_acceleration=2,
            ),
        )
    if mode == Mode.GRSIM:
        return PIDConfigs(
            orientation=OrientationPIDConfigs(
                max_output=MAX_ANGULAR_VEL,
                min_output=-MAX_ANGULAR_VEL,
                kp=4.5,
                kd=0.02,
                ki=0.0,
                integral_min=-10,
                integral_max=10,
                max_acceleration=50,
            ),
            translation=TranslationPIDConfigs(
                max_velocity=MAX_VEL,
                kp=1.8,
                kd=0.025,
                ki=0.0,
                integral_min=-5,
                integral_max=5,
                max_acceleration=2,
            ),
        )
    if mode == Mode.REAL:
        return PIDConfigs(
            orientation=OrientationPIDConfigs(
                max_output=REAL_MAX_ANG,
                min_output=-REAL_MAX_ANG,
                kp=0.5,
                kd=0.075,
                ki=0.0,
                max_acceleration=0.2,
            ),
            translation=TranslationPIDConfigs(
                max_velocity=REAL_MAX_VEL,
                kp=1.8,
                kd=0.025,
                ki=0.0,
                integral_min=-5,
                integral_max=5,
                max_acceleration=2,
            ),
        )
    raise ValueError(f"Unknown mode enum: {mode}.")
