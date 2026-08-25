import math
import time

from utama_core.config.enums import Mode
from utama_core.entities.data.vector import Vector2D
from utama_core.global_utils.math_utils import normalise_heading
from utama_core.motion_planning.src.pid.configs import (
    OrientationPIDConfigs,
    TranslationPIDConfigs,
    get_pid_configs,
)
from utama_core.motion_planning.src.pid.pid_abstract import AbstractPID

# A phase hand-off / reassignment re-aiming a robot swings the target by a
# large angle in one tick (observed 100+ degrees in `switch_of_play`'s
# relay->finish transition); ordinary tracking of a moving target (e.g.
# facing the ball while it drifts) never jumps this far in a single 1/60s
# tick. See `AbstractPID._target_jumped`'s docstring for why this matters.
_ORIENTATION_JUMP_THRESHOLD = 0.5  # radians (~28.6 degrees)


class PID(AbstractPID[float]):
    """A PID controller that control the Orientation of the robot.

    Args:
        dt (float): Time step for each update.
        max_output (Optional[float]): Maximum output value (None for no limit).
        min_output (Optional[float]): Minimum output value (None for no limit).
        Kp (float): Proportional gain.
        Kd (float): Derivative gain.
        Ki (float): Integral gain.
        integral_min (Optional[float]): Minimum allowed integral value.
        integral_max (Optional[float]): Maximum allowed integral value.

    Note:
        The delay used by the Smith predictor is internally set using the
        :data:`SENDING_DELAY` configuration value.
    """

    def __init__(
        self,
        config: OrientationPIDConfigs,
    ):
        super().__init__(config)
        self.max_output = config.max_output
        self.min_output = config.min_output

    def _target_jumped(self, target: float, last_target: float) -> bool:
        return abs(normalise_heading(target - last_target)) > _ORIENTATION_JUMP_THRESHOLD

    def _calculate(
        self,
        target: float,
        current: float,
        robot_id: int,
    ) -> float:
        """Compute the PID output to move a robot towards a target with delay compensation."""
        # Compute the basic (instantaneous) error
        raw_error = target - current
        # For angular measurements adjust error
        error = normalise_heading(raw_error)
        # For very small errors, return zero
        if abs(error) < 0.001:
            return 0.0

        # Compute derivative term using the previous stored error
        if not self.first_pass[robot_id]:
            derivative = (error - self.pre_errors[robot_id]) / self.dt
        else:
            derivative = 0.0
            self.first_pass[robot_id] = False

        # --- Delay Compensation (Smith predictor approach) ---
        # Predict the "current" value (what it will be after the delay)
        # Here we assume a simple linear extrapolation: x_predicted = current + derivative * delay
        if self.delay > 0:
            predicted_current = current + derivative * self.delay
            # Then the predicted error is:
            predicted_error = normalise_heading(target - predicted_current)
            # Optionally, you might replace 'error' with 'predicted_error' in the PID computation.
            effective_error = predicted_error
        else:
            effective_error = error

        # Proportional term
        Pout = self.Kp * effective_error if self.Kp != 0 else 0.0

        # Integral term with anti-windup using effective_error
        if self.Ki != 0:
            self.integrals[robot_id] += effective_error * self.dt
            if self.integral_max is not None:
                self.integrals[robot_id] = min(self.integrals[robot_id], self.integral_max)
            if self.integral_min is not None:
                self.integrals[robot_id] = max(self.integrals[robot_id], self.integral_min)
            Iout = self.Ki * self.integrals[robot_id]
        else:
            Iout = 0.0

        # Derivative term based on effective error (already computed above)
        Dout = self.Kd * derivative

        # Combine the PID outputs
        output = Pout + Iout + Dout

        # Clamp the output for consistency
        if self.max_output is not None:
            output = min(self.max_output, output)
        if self.min_output is not None:
            output = max(self.min_output, output)

        # Store the error and update the time for the next iteration
        self.pre_errors[robot_id] = error
        # print(f"oren PID: {robot_id}, current:{current}, target: {target}, error: {error}, output: {output}")

        return output


class TwoDPID(AbstractPID[Vector2D]):
    """A 2D PID controller that controls the X and Y dimensions and scales the resulting velocity vector to a maximum
    speed if needed."""

    def __init__(
        self,
        config: TranslationPIDConfigs,
    ):
        super().__init__(config)
        self.max_velocity = config.max_velocity
        self.max_acceleration = config.max_acceleration
        # Set by a caller (`FastPathPlanningController`) right before
        # `calculate()` when `target` is a lookahead "carrot" rather than the
        # robot's actual destination — see `set_final_target`'s docstring for
        # why the braking-distance cap needs this distinction.
        self._final_targets: dict[int, Vector2D] = {}

    def set_final_target(self, robot_id: int, final_target: Vector2D | None) -> None:
        """Tell the braking-distance cap the robot's true destination, when
        `target` passed to `calculate()` is a path-planning waypoint instead.

        `FastPathPlanner._path_to`'s `smooth_path` deliberately projects a
        "carrot" up to `PROJECTION_DISTANCE` (1m) ahead of the robot, not at
        the real destination, so `check_segment`'s obstacle-aware routing has
        somewhere useful to aim while still far away. But `_calculate`'s
        braking-distance cap (`max_brake_vel = sqrt(2*a*error)`) needs the
        *true* remaining distance to know when to start slowing down — fed
        the carrot's distance instead, `error` stays pinned near 1m for the
        entire approach (the carrot recedes at the same rate the robot closes
        on it), so the cap never engages until the carrot snaps to the real
        target on the final ~1m, by which point the robot may already be at
        full speed with too little runway left to stop. Confirmed via a live
        tournament replay (`counter_flow_vs_tiki_taka_plus_Lk.pkl`,
        t=13-17.5s): the keeper's own target (`stop_y`) was smoothly settled
        at -0.5 well before t=16.5s, yet its measured `y` velocity kept
        flipping sign every few ticks — 6+ reversals — as it repeatedly shot
        past the target and corrected, until the ball rolled out of play
        during the oscillation window. Reproduced synthetically too
        (`debug_match.py` trace): carrot pinned at `carrot_err≈1.0` while
        `true_err` closed from >3m to <0.15m at 1.5-2.0 m/s, no deceleration
        signal reaching the PID until far too late.

        Direction still comes from `target` (the carrot) — only the braking
        cap's distance uses `final_target`, so obstacle-aware path-following
        is unaffected; this only changes when the robot starts slowing down
        for its own eventual stop. Cleared (`None`) by callers that don't use
        a carrot (e.g. `PIDController`, which never sets this) so `_calculate`
        falls back to the carrot-derived `error` — the direct-target case
        where they're the same value anyway.
        """
        if final_target is None:
            self._final_targets.pop(robot_id, None)
        else:
            self._final_targets[robot_id] = final_target

    def _target_jumped(self, target: Vector2D, last_target: Vector2D) -> bool:
        # Unlike orientation, translation targets legitimately move by
        # metre-scale distances tick-to-tick under ordinary operation — a
        # moving formation reference point, a live-recomputed setup
        # position, a path-planner waypoint (`FastPathPlanningController`
        # feeds these straight from `FastPathPlanner._path_to`) all shift
        # substantially without representing a discontinuity. Auto-reset
        # on a position jump was tried and measured to do more harm than
        # good: it fired ~14 times over 200 ticks tracking a single
        # formation target whose reference point itself moves as the robot
        # approaches (`test_referee_override.py`'s penalty-formation test),
        # each reset discarding real acceleration-limiter/derivative
        # progress and costing enough convergence time to fail the test's
        # tolerance. Every actual bug this session traced was the
        # *orientation* PID's stale derivative from a target re-aim
        # (`PID._target_jumped` below) — translation was never the
        # culprit, so it isn't given this behavior.
        del target, last_target
        return False

    def _calculate(self, target: Vector2D, current: Vector2D, robot_id: int) -> Vector2D:
        dx = target[0] - current[0]
        dy = target[1] - current[1]

        error = math.hypot(dx, dy)

        if abs(error) < 3 / 1000:
            return Vector2D(0.0, 0.0)

        # Compute derivative term using the previous stored error
        if not self.first_pass[robot_id]:
            derivative = (error - self.pre_errors[robot_id]) / self.dt
        else:
            derivative = 0.0
            self.first_pass[robot_id] = False

        # --- Delay Compensation (Smith predictor approach) ---
        # Then the predicted error is:
        if self.delay > 0:
            predicted_error = error + derivative * self.delay
            # Optionally, you might replace 'error' with 'predicted_error' in the PID computation.
            effective_error = predicted_error
        else:
            effective_error = error

        # Proportional term
        Pout = self.Kp * effective_error if self.Kp != 0 else 0.0

        # Integral term with anti-windup using effective_error
        if self.Ki != 0:
            self.integrals[robot_id] += effective_error * self.dt
            if self.integral_max is not None:
                self.integrals[robot_id] = min(self.integrals[robot_id], self.integral_max)
            if self.integral_min is not None:
                self.integrals[robot_id] = max(self.integrals[robot_id], self.integral_min)
            Iout = self.Ki * self.integrals[robot_id]
        else:
            Iout = 0.0

        # Derivative term based on effective error (already computed above)
        Dout = self.Kd * derivative

        # Combine the PID outputs
        output = Pout + Iout + Dout

        # Store the error and update the time for the next iteration
        self.pre_errors[robot_id] = error

        # print(f"x-y PID: {robot_id}, current:{current}, target: {target}, error: {error}, output: {output}")
        if error == 0.0:
            return Vector2D(0.0, 0.0)
        x_vel = output * (dx / error)
        y_vel = output * (dy / error)

        # Braking-distance cap: Kp*error alone doesn't account for the robot's
        # own deceleration limit, so on a fast approach the commanded speed
        # can exceed what's stoppable within the remaining distance, and the
        # rate-limited ramp-down (accel_limiter, applied after this returns)
        # isn't fast enough to prevent overshoot. v = sqrt(2*a*d) is the max
        # speed that can still be braked to zero by the time distance d closes.
        #
        # `error` (the carrot's distance) is the wrong `d` whenever `target`
        # is a path-planning lookahead point rather than the robot's actual
        # destination — see `set_final_target`'s docstring. Use the true
        # remaining distance for this cap when the caller has provided one;
        # direction (`dx/dy` above) still follows the carrot regardless.
        final_target = self._final_targets.get(robot_id)
        brake_distance = (
            error if final_target is None else math.hypot(final_target[0] - current[0], final_target[1] - current[1])
        )
        max_brake_vel = (
            math.sqrt(2 * self.max_acceleration * brake_distance) if self.max_acceleration > 0 else self.max_velocity
        )
        return self._apply_speed_limits(x_vel, y_vel, min(self.max_velocity, max_brake_vel))

    def _apply_speed_limits(self, x_vel: float, y_vel: float, max_vel: float) -> Vector2D:
        current_vel = math.hypot(x_vel, y_vel)
        if current_vel > max_vel:
            scaling_factor = max_vel / current_vel
            x_vel *= scaling_factor
            y_vel *= scaling_factor
        return Vector2D(x_vel, y_vel)

    def reset(self, robot_id: int):
        super().reset(robot_id)
        self._final_targets.pop(robot_id, None)


def get_pids(
    mode: Mode,
) -> tuple[PID, TwoDPID]:
    """Instantiate PID controllers from a configuration."""
    config = get_pid_configs(mode)
    pid_oren = PID(config.orientation)
    pid_trans = TwoDPID(config.translation)
    return pid_oren, pid_trans
