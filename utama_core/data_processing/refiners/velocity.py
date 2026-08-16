import logging
from dataclasses import replace
from typing import Dict, Tuple, Union  # Added List for type hinting

import numpy as np  # Import NumPy

from utama_core.data_processing.refiners.base_refiner import BaseRefiner
from utama_core.entities.data.object import ObjectKey, TeamType
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import GameFrame, Robot
from utama_core.entities.game.game_history import (
    AttributeType,
    GameHistory,
    get_structured_object_key,
)

logger = logging.getLogger(__name__)


def zero_vector(twod: bool) -> Union[Vector2D, Vector3D]:
    return Vector2D(0, 0) if twod else Vector3D(0, 0, 0)


class VelocityRefiner(BaseRefiner):
    # Acceleration uses windowed-average finite differencing: split the last
    # N_WINDOWS*WINDOW_SIZE velocity samples into N_WINDOWS windows, average each window, then
    # diff consecutive window-averages. This trades responsiveness for noise rejection, which is
    # fine for acceleration (a slower-changing, already-smoothed-once-removed quantity feeding
    # mostly planning rather than a tight feedback loop).
    #
    # Velocity intentionally does NOT use this scheme, even at a small window size: any window
    # or regression smoothing of velocity introduces lag under acceleration (a step change in
    # true velocity, e.g. a robot starting/stopping), because the estimate is necessarily
    # centered on a timestamp behind the current tick. This was tried and measured to break
    # time-critical control loops (a robot failed to clear a keep-out zone in time and failed to
    # intercept a moving ball, both by a wide margin, even with a minimal 4-point window). Position
    # is already Kalman-smoothed upstream (see PositionRefiner); velocity stays a plain 1-step
    # finite difference so it stays maximally responsive for control.
    ACCELERATION_WINDOW_SIZE = 5
    ACCELERATION_N_WINDOWS = 3

    def refine(self, game_history: GameHistory, game_frame: GameFrame) -> GameFrame:
        current_game_ts = game_frame.ts

        # Process Ball (Keep this commented if you want to focus on robots first)
        if game_frame.ball:  # Ensure ball processing is guarded
            game_frame = self._refine_ball_kinematics(game_history, game_frame, current_game_ts)

        # Process Friendly Robots
        game_frame = self._refine_robot_group(
            game_history,
            game_frame,
            current_game_ts,
            game_frame.friendly_robots,
            TeamType.FRIENDLY,
            "friendly_robots",
            twod=True,
        )

        # Process Enemy Robots
        game_frame = self._refine_robot_group(
            game_history,
            game_frame,
            current_game_ts,
            game_frame.enemy_robots,
            TeamType.ENEMY,
            "enemy_robots",
            twod=True,
        )
        return game_frame

    def _refine_robot_group(
        self,
        game_history: GameHistory,
        game_state: GameFrame,
        current_ts: float,
        robots_to_process_dict: Dict[int, Robot],
        team_type: TeamType,
        field_name: str,
        twod: bool,
    ) -> GameFrame:
        updated_robots_dict = {}
        for robot_instance in robots_to_process_dict.values():
            robot_id = getattr(robot_instance, "id", None)
            if robot_id is None:
                logger.error(f"{team_type.name} robot instance encountered without an ID. Skipping.")
                continue

            if robot_instance.p is None:
                logger.warning(f"{team_type.name} robot {robot_id} has no position. Setting zero v/a.")
                updated_robot = replace(robot_instance, v=zero_vector(twod), a=zero_vector(twod))
                updated_robots_dict[robot_id] = updated_robot
                continue

            robot_obj_key = get_structured_object_key(robot_instance, team_type)
            if not robot_obj_key:
                logger.error(f"Could not get ObjectKey for {team_type.name} robot {robot_id}. Adding original to dict.")
                updated_robots_dict[robot_id] = robot_instance
                continue

            new_v = self._calculate_object_velocity(game_history, robot_instance.p, robot_obj_key, current_ts, twod)

            new_a = zero_vector(twod)  # Default to zero
            try:
                new_a = self._calculate_object_acceleration(game_history, robot_obj_key, twod)
            except Exception as e:
                logger.warning(
                    f"Could not calculate acceleration for {team_type.name} robot {robot_id} (key: {robot_obj_key}), setting to zero: {e}"
                )

            updated_robot = replace(robot_instance, v=new_v, a=new_a)
            updated_robots_dict[robot_id] = updated_robot

        return replace(game_state, **{field_name: updated_robots_dict})

    def _refine_ball_kinematics(self, game_history: GameHistory, game_state: GameFrame, current_ts: float) -> GameFrame:
        if not game_state.ball:
            return game_state

        if game_state.ball.p is None:
            logger.warning("Ball exists but has no position data; setting zero velocity and acceleration.")
            new_ball = replace(game_state.ball, v=zero_vector(twod=False), a=zero_vector(twod=False))
            return replace(game_state, ball=new_ball)

        ball_obj_key = get_structured_object_key(game_state.ball, TeamType.NEUTRAL)
        if not ball_obj_key:
            logger.error("Could not get ObjectKey for ball. Skipping ball refinement.")
            return game_state

        new_ball_v = self._calculate_object_velocity(
            game_history, game_state.ball.p, ball_obj_key, current_ts, twod=False
        )

        new_ball_a = zero_vector(twod=False)  # Default to zero
        try:
            new_ball_a = self._calculate_object_acceleration(game_history, ball_obj_key, twod=False)
        except Exception as e:
            logger.warning(f"Could not calculate acceleration for ball (key: {ball_obj_key}), setting to zero: {e}")

        new_ball = replace(game_state.ball, v=new_ball_v, a=new_ball_a)
        return replace(game_state, ball=new_ball)

    def _calculate_object_velocity(
        self,
        game_history: GameHistory,
        current_pos: Union[Vector2D, Vector3D],
        object_key: ObjectKey,
        current_ts: float,
        twod: bool,
    ) -> Union[Vector2D, Vector3D]:
        """Estimates velocity as a plain 1-step finite difference against the most recent
        historical position. Deliberately unsmoothed — see the class docstring comment on
        ACCELERATION_WINDOW_SIZE for why: any windowing here adds control-loop-breaking lag.
        """
        timestamps_np, positions_np = game_history.get_historical_attribute_series(
            object_key, AttributeType.POSITION, 1
        )

        if not timestamps_np.size or not positions_np.size:
            return zero_vector(twod)

        previous_time_received = timestamps_np[0]
        previous_pos_np = positions_np[0]

        dt_secs = current_ts - previous_time_received
        if dt_secs <= 1e-9:
            return zero_vector(twod)

        if twod:
            previous_pos = Vector2D(previous_pos_np[0], previous_pos_np[1])
        else:
            previous_pos = Vector3D(previous_pos_np[0], previous_pos_np[1], previous_pos_np[2])

        return (current_pos - previous_pos) / dt_secs

    def _calculate_object_acceleration(
        self, game_history: GameHistory, object_key: ObjectKey, twod: bool
    ) -> Union[Vector2D, Vector3D]:
        try:
            num_points_needed = self.ACCELERATION_N_WINDOWS * self.ACCELERATION_WINDOW_SIZE
            timestamps_np, velocities_np = self._extract_time_velocity_np_arrays(
                game_history, object_key, num_points_needed
            )

            if timestamps_np.shape[0] < num_points_needed:  # Check if enough points were returned
                logger.debug(
                    f"Not enough velocity points from GameHistory for {object_key}. Have {timestamps_np.shape[0]}, need {num_points_needed}"
                )
                return zero_vector(twod)
        except Exception as e:
            raise ValueError(f"Velocity data not available for acceleration for {object_key}: {e}") from e

        return self._windowed_average_derivative(
            timestamps_np,
            velocities_np,
            self.ACCELERATION_N_WINDOWS,
            self.ACCELERATION_WINDOW_SIZE,
            twod,
            log_prefix="ACCEL",
        )

    def _extract_time_velocity_np_arrays(
        self, game_history: GameHistory, object_key: ObjectKey, num_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        timestamps_np, velocities_np = game_history.get_historical_attribute_series(
            object_key, AttributeType.VELOCITY, num_points
        )
        return timestamps_np, velocities_np

    def _windowed_average_derivative(
        self,
        timestamps_np: np.ndarray,
        values_np: np.ndarray,
        n_windows: int,
        window_size: int,
        twod: bool,
        log_prefix: str,
    ) -> Union[Vector2D, Vector3D]:
        """Estimates d(values)/dt by averaging consecutive windows and differencing the averages.

        Splits the last n_windows*window_size samples into n_windows equal windows, averages
        each window's value and timestamp, then diffs consecutive window-averages and divides
        to get a per-segment derivative, averaged across all valid (non-degenerate dt) segments.
        """
        if n_windows < 2 or window_size < 1:
            logger.warning(
                f"{log_prefix}: n_windows={n_windows}, window_size={window_size} is degenerate. Returning zero."
            )
            return zero_vector(twod)

        min_total_points_needed = n_windows * window_size
        num_dimensions = 2 if twod else 3

        if (
            timestamps_np.shape[0] < min_total_points_needed
            or values_np.shape[0] < min_total_points_needed
            or values_np.shape[1] != num_dimensions
        ):
            logger.warning(
                f"{log_prefix}: Insufficient/malformed data. timestamps={timestamps_np.shape}, "
                f"values={values_np.shape}, need {min_total_points_needed} points x {num_dimensions} dims. Returning zero."
            )
            return zero_vector(twod)

        # Use the most recent min_total_points_needed points.
        # Plain-float implementation, not numpy: this is called ~46k times per match with a
        # small fixed shape (n_windows=3, window_size=5, 2-3 dims), where np.reshape/np.mean/
        # np.diff's dispatch overhead dwarfs the actual arithmetic at this size — same pattern
        # as distance_point_to_segment's hot-path float rewrite (see that function's docstring).
        start = timestamps_np.shape[0] - min_total_points_needed
        active_timestamps = timestamps_np[start:].tolist()
        active_values = values_np[start:].tolist()

        avg_ts_per_window = [0.0] * n_windows
        avg_values_per_window = [[0.0] * num_dimensions for _ in range(n_windows)]
        for w in range(n_windows):
            ts_sum = 0.0
            val_sums = [0.0] * num_dimensions
            base = w * window_size
            for i in range(window_size):
                ts_sum += active_timestamps[base + i]
                row = active_values[base + i]
                for d in range(num_dimensions):
                    val_sums[d] += row[d]
            avg_ts_per_window[w] = ts_sum / window_size
            avg_values_per_window[w] = [s / window_size for s in val_sums]

        sum_derivative = [0.0] * num_dimensions
        valid_segments = 0
        for w in range(1, n_windows):
            d_ts = avg_ts_per_window[w] - avg_ts_per_window[w - 1]
            if d_ts <= 1e-9:
                continue
            valid_segments += 1
            for d in range(num_dimensions):
                sum_derivative[d] += (avg_values_per_window[w][d] - avg_values_per_window[w - 1][d]) / d_ts

        if valid_segments == 0:
            logger.warning(f"{log_prefix}: All dt segments too small or zero. Returning zero.")
            return zero_vector(twod)

        final_derivative = [s / valid_segments for s in sum_derivative]

        if twod:
            return Vector2D(final_derivative[0], final_derivative[1])
        else:
            return Vector3D(final_derivative[0], final_derivative[1], final_derivative[2])
