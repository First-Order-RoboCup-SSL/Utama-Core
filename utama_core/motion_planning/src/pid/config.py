from utama_core.config.enums import Mode
from utama_core.config.robot_params.grsim import MAX_ANGULAR_VEL, MAX_VEL
from utama_core.config.robot_params.real import MAX_ANGULAR_VEL as REAL_MAX_ANG
from utama_core.config.robot_params.real import MAX_VEL as REAL_MAX_VEL
from utama_core.config.robot_params.rsim import MAX_ANGULAR_VEL as RSIM_MAX_ANG
from utama_core.config.robot_params.rsim import MAX_VEL as RSIM_MAX_VEL
from utama_core.config.settings import TIMESTEP
from utama_core.motion_planning.src.pid.pid import PID, TwoDPID
from utama_core.motion_planning.src.pid.pid_acceleration_limiter import (
    PIDAccelerationLimiterWrapper,
)


# Helper functions to create PID controllers.
def get_pids(mode: Mode) -> tuple[PID, TwoDPID]:
    """
    Get PID controllers for orientation and translation based on the specified mode.
    """
    if mode == Mode.RSIM:
        pid_oren = PID(
            TIMESTEP,
            RSIM_MAX_ANG,
            -RSIM_MAX_ANG,
            4.5,
            0.02,
            0,
            integral_min=-10,
            integral_max=10,
        )
        pid_trans = TwoDPID(
            TIMESTEP,
            RSIM_MAX_VEL,
            1.8,
            0.025,
            0.0,
            integral_min=-5,
            integral_max=5,
        )
        return PIDAccelerationLimiterWrapper(pid_oren, max_acceleration=50, dt=TIMESTEP), PIDAccelerationLimiterWrapper(
            pid_trans, max_acceleration=2, dt=TIMESTEP
        )
    elif mode == Mode.GRSIM:
        pid_oren = PID(
            TIMESTEP,
            MAX_ANGULAR_VEL,
            -MAX_ANGULAR_VEL,
            4.5,
            0.02,
            0,
            integral_min=-10,
            integral_max=10,
        )
        pid_trans = TwoDPID(
            TIMESTEP,
            MAX_VEL,
            1.8,
            0.025,
            0.0,
            integral_min=-5,
            integral_max=5,
        )
        return PIDAccelerationLimiterWrapper(pid_oren, max_acceleration=50, dt=TIMESTEP), PIDAccelerationLimiterWrapper(
            pid_trans, max_acceleration=2, dt=TIMESTEP
        )
    elif mode == Mode.REAL:
        pid_oren = PID(
            TIMESTEP,
            REAL_MAX_ANG,
            -REAL_MAX_ANG,
            0.5,
            0.075,
            0,
        )
        pid_trans = TwoDPID(
            TIMESTEP,
            REAL_MAX_VEL,
            0,
            0,
            0.0,
        )
        return PIDAccelerationLimiterWrapper(pid_oren, max_acceleration=0.2), PIDAccelerationLimiterWrapper(
            pid_trans, max_acceleration=0.05
        )
    else:
        raise ValueError(f"Unknown mode enum: {mode}.")


def get_real_pids_goalie():
    pid_oren = PID(
        TIMESTEP,
        REAL_MAX_ANG,
        -REAL_MAX_ANG,
        1.5,
        0,
        0,
    )
    pid_trans = TwoDPID(TIMESTEP, 2, 8.5, 0.025, 1)
    return PIDAccelerationLimiterWrapper(pid_oren, max_acceleration=2), PIDAccelerationLimiterWrapper(
        pid_trans, max_acceleration=1
    )
