from dataclasses import replace
from entities.game import Game, Robot
from entities.data.vector import Vector2D, Vector3D

from typing import Dict, List, Union, Tuple  # Added List for type hinting

# Assuming your new PastGame is in entities.game.past_game or accessible
from entities.game.past_game import (
    PastGame,
    AttributeType,
    TeamType,
    ObjectClass,
    ObjectKey,
    get_structured_object_key,
)
from run.refiners.base_refiner import BaseRefiner
from lenses import UnboundLens, lens
import logging
import numpy as np  # Import NumPy

logger = logging.getLogger(__name__)


def zero_vector(twod: bool) -> Union[Vector2D, Vector3D]:
    return Vector2D(0, 0) if twod else Vector3D(0, 0, 0)


class VelocityRefiner(BaseRefiner):
    ACCELERATION_WINDOW_SIZE = 5
    ACCELERATION_N_WINDOWS = 3

    def refine(self, past_game: PastGame, game: Game) -> Game:
        current_game_ts = game.ts

        # Process Ball (Keep this commented if you want to focus on robots first)
        if game.ball:  # Ensure ball processing is guarded
            game = self._refine_ball_kinematics(past_game, game, current_game_ts)

        # Process Friendly Robots
        game = self._refine_robot_group(
            past_game,
            game,
            current_game_ts,
            game.friendly_robots,
            TeamType.FRIENDLY,
            lens.friendly_robots,
            twod=True,
        )

        # Process Enemy Robots
        game = self._refine_robot_group(
            past_game,
            game,
            current_game_ts,
            game.enemy_robots,
            TeamType.ENEMY,
            lens.enemy_robots,
            twod=True,
        )
        return game

    def _refine_robot_group(
        self,
        past_game: PastGame,
        game_state: Game,
        current_ts: float,
        robots_to_process_dict: Dict[int, Robot],
        team_type: TeamType,
        group_lens: UnboundLens,
        twod: bool,
    ) -> Game:
        updated_robots_dict = {}
        for robot_instance in robots_to_process_dict.values():
            robot_id = getattr(robot_instance, "id", None)
            if robot_id is None:
                logger.error(
                    f"{team_type.name} robot instance encountered without an ID. Skipping."
                )
                continue

            if robot_instance.p is None:
                logger.warning(
                    f"{team_type.name} robot {robot_id} has no position. Setting zero v/a."
                )
                updated_robot = (
                    robot_instance
                    & lens.v.set(zero_vector(twod))
                    & lens.a.set(zero_vector(twod))
                )
                updated_robots_dict[robot_id] = updated_robot
                continue

            robot_obj_key = get_structured_object_key(robot_instance, team_type)
            if not robot_obj_key:
                logger.error(
                    f"Could not get ObjectKey for {team_type.name} robot {robot_id}. Adding original to dict."
                )
                updated_robots_dict[robot_id] = robot_instance
                continue

            new_v = self._calculate_object_velocity(
                past_game, robot_instance.p, robot_obj_key, current_ts, twod
            )

            new_a = zero_vector(twod)  # Default to zero
            try:
                new_a = self._calculate_object_acceleration(
                    past_game, robot_obj_key, twod
                )
            except Exception as e:
                logger.warning(
                    f"Could not calculate acceleration for {team_type.name} robot {robot_id} (key: {robot_obj_key}), setting to zero: {e}"
                )

            updated_robot = robot_instance & lens.v.set(new_v) & lens.a.set(new_a)
            updated_robots_dict[robot_id] = updated_robot

        return game_state & group_lens.set(updated_robots_dict)

    def _refine_ball_kinematics(
        self, past_game: PastGame, game_state: Game, current_ts: float
    ) -> Game:
        if not game_state.ball:
            return game_state

        if game_state.ball.p is None:
            logger.warning(
                "Ball exists but has no position data; setting zero velocity and acceleration."
            )
            return (
                game_state
                & lens.ball.v.set(zero_vector(twod=False))
                & lens.ball.a.set(zero_vector(twod=False))
            )

        ball_obj_key = get_structured_object_key(game_state.ball, TeamType.NEUTRAL)
        if not ball_obj_key:
            logger.error("Could not get ObjectKey for ball. Skipping ball refinement.")
            return game_state

        new_ball_v = self._calculate_object_velocity(
            past_game, game_state.ball.p, ball_obj_key, current_ts, twod=False
        )
        game_state &= lens.ball.v.set(new_ball_v)

        new_ball_a = zero_vector(twod=False)  # Default to zero
        try:
            new_ball_a = self._calculate_object_acceleration(
                past_game, ball_obj_key, twod=False
            )
        except Exception as e:
            logger.warning(
                f"Could not calculate acceleration for ball (key: {ball_obj_key}), setting to zero: {e}"
            )
        game_state &= lens.ball.a.set(new_ball_a)

        return game_state

    def _calculate_object_velocity(
        self,
        past_game: PastGame,
        current_pos: Union[Vector2D, Vector3D],
        object_key: ObjectKey,
        current_ts: float,
        twod: bool,
    ) -> Union[Vector2D, Vector3D]:
        try:
            timestamps_np, positions_np = past_game.get_historical_attribute_series(
                object_key, AttributeType.POSITION, 1
            )
            # logger.debug(f"VELOCITY_CALC: obj_key={object_key}, current_ts={current_ts}, current_pos={current_pos}")
            # logger.debug(f"VELOCITY_CALC: historical timestamps_np={timestamps_np}, positions_np={positions_np.tolist() if positions_np.size > 0 else 'EMPTY'}")

            if not timestamps_np.size or not positions_np.size:
                logger.warning(
                    f"VELOCITY_CALC: No historical position for {object_key}. PastGame returned empty arrays. Setting zero velocity."
                )
                return zero_vector(twod)

            # Assign these *after* confirming timestamps_np and positions_np are not empty
            previous_time_received = timestamps_np[0]
            previous_pos_np = positions_np[
                0
            ]  # This is a 1D NumPy array [x, y] or [x, y, z]

            # Now calculate dt_secs using the defined previous_time_received
            dt_secs = current_ts - previous_time_received
            # logger.debug(f"VELOCITY_CALC: obj_key={object_key}, dt_secs={dt_secs}, prev_ts={previous_time_received}, prev_pos_np={previous_pos_np}")

            if dt_secs <= 1e-9:
                logger.warning(
                    f"VELOCITY_CALC: Object {object_key} velocity dt_secs is too small or zero ({dt_secs}). Using zero velocity."
                )
                return zero_vector(twod)

            # Convert previous_pos_np back to VectorObject for subtraction
            if twod:
                previous_pos = Vector2D(previous_pos_np[0], y=previous_pos_np[1])
            else:
                previous_pos = Vector3D(
                    x=previous_pos_np[0], y=previous_pos_np[1], z=previous_pos_np[2]
                )

            velocity = (current_pos - previous_pos) / dt_secs
            # logger.debug(f"VELOCITY_CALC: obj_key={object_key}, calculated_v={velocity}")
            return velocity

        except IndexError:
            logger.error(
                f"VELOCITY_CALC: IndexError for {object_key}. This indicates an issue with historical data structure despite size checks. Setting zero velocity.",
                exc_info=True,
            )
            return zero_vector(twod)
        except Exception as e:
            logger.error(
                f"VELOCITY_CALC: Unexpected error calculating velocity for {object_key}, setting to zero: {e}",
                exc_info=True,
            )
            return zero_vector(twod)

    def _calculate_object_acceleration(
        self, past_game: PastGame, object_key: ObjectKey, twod: bool
    ) -> Union[Vector2D, Vector3D]:
        try:
            num_points_needed = (
                self.ACCELERATION_N_WINDOWS * self.ACCELERATION_WINDOW_SIZE
            )
            timestamps_np, velocities_np = self._extract_time_velocity_np_arrays(
                past_game, object_key, num_points_needed
            )

            if (
                timestamps_np.shape[0] < num_points_needed
            ):  # Check if enough points were returned
                raise ValueError(
                    f"Not enough velocity points from PastGame for {object_key}. Have {timestamps_np.shape[0]}, need {num_points_needed}"
                )
        except Exception as e:
            raise ValueError(
                f"Velocity data not available for acceleration for {object_key}: {e}"
            ) from e

        # Pass NumPy arrays directly
        return self._calculate_acceleration_from_pairs(
            timestamps_np, velocities_np, twod
        )

    def _extract_time_velocity_np_arrays(
        self, past_game: PastGame, object_key: ObjectKey, num_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        timestamps_np, velocities_np = past_game.get_historical_attribute_series(
            object_key, AttributeType.VELOCITY, num_points
        )
        return timestamps_np, velocities_np

    def _calculate_acceleration_from_pairs(
        self,
        timestamps_np: np.ndarray,
        velocities_np: np.ndarray,  # This is now a 2D NumPy array
        twod: bool,
    ) -> Union[Vector2D, Vector3D]:
        """
        Estimates an object's acceleration using NumPy arrays as input.
        """
        # logger.debug(f"ACCEL_PAIRS_INPUT: timestamps_np shape={timestamps_np.shape}, velocities_np shape={velocities_np.shape}, twod={twod}")

        if self.ACCELERATION_N_WINDOWS == 0 or self.ACCELERATION_WINDOW_SIZE == 0:
            # logger.info("ACCEL_PAIRS: N_WINDOWS or WINDOW_SIZE is 0, returning zero vector.")
            return zero_vector(twod)

        # Minimum points needed for the entire calculation based on N_WINDOWS
        # If N_WINDOWS < 2, we can't form any dv/dt segments with the current logic.
        if self.ACCELERATION_N_WINDOWS < 2:
            logger.warning(
                f"ACCEL_PAIRS: ACCELERATION_N_WINDOWS is {self.ACCELERATION_N_WINDOWS}. "
                "Need at least 2 windows to calculate acceleration with current logic. Returning zero."
            )
            return zero_vector(twod)

        min_total_points_needed = (
            self.ACCELERATION_N_WINDOWS * self.ACCELERATION_WINDOW_SIZE
        )

        if timestamps_np.shape[0] < min_total_points_needed:
            logger.warning(
                f"ACCEL_PAIRS: Not enough data points. Have {timestamps_np.shape[0]}, "
                f"need {min_total_points_needed} for {self.ACCELERATION_N_WINDOWS} windows "
                f"of size {self.ACCELERATION_WINDOW_SIZE}. Returning zero."
            )
            return zero_vector(twod)

        # Ensure velocities_np also has enough points (should match timestamps_np)
        if velocities_np.shape[0] < min_total_points_needed:
            logger.warning(
                f"ACCEL_PAIRS: Velocities_np has insufficient data points. Have {velocities_np.shape[0]}, "
                f"need {min_total_points_needed}. Returning zero."
            )
            return zero_vector(twod)

        num_dimensions = 2 if twod else 3
        if velocities_np.shape[1] != num_dimensions:
            logger.error(
                f"ACCEL_PAIRS: velocities_np has incorrect number of dimensions. "
                f"Expected {num_dimensions} (twod={twod}), got {velocities_np.shape[1]}. Returning zero."
            )
            return zero_vector(twod)

        # Trim excess data if more points were provided than needed for the configured windows.
        # This should ideally be handled by the caller (_calculate_object_acceleration) ensuring
        # that `timestamps_np` and `velocities_np` are already correctly sized.
        # For robustness, we slice here if they are larger than needed.

        active_timestamps_np = timestamps_np[:min_total_points_needed]
        active_velocities_np = velocities_np[:min_total_points_needed]

        try:
            windowed_velocities = active_velocities_np.reshape(
                self.ACCELERATION_N_WINDOWS,
                self.ACCELERATION_WINDOW_SIZE,
                num_dimensions,
            )
            windowed_timestamps = active_timestamps_np.reshape(
                self.ACCELERATION_N_WINDOWS, self.ACCELERATION_WINDOW_SIZE
            )
        except ValueError as e:
            logger.error(
                f"ACCEL_PAIRS: Cannot reshape arrays. "
                f"active_timestamps_np shape: {active_timestamps_np.shape}, "
                f"active_velocities_np shape: {active_velocities_np.shape}, "
                f"N_WINDOWS: {self.ACCELERATION_N_WINDOWS}, WINDOW_SIZE: {self.ACCELERATION_WINDOW_SIZE}. Error: {e}",
                exc_info=True,
            )
            return zero_vector(twod)

        avg_velocities_per_window = np.mean(
            windowed_velocities, axis=1
        )  # Shape: (N_WINDOWS, num_dimensions)
        middle_ts_per_window = np.mean(
            windowed_timestamps, axis=1
        )  # Shape: (N_WINDOWS)

        # Calculate differences between consecutive window averages
        dv_segments = np.diff(
            avg_velocities_per_window, axis=0
        )  # Shape: (N_WINDOWS-1, num_dimensions)
        dt_segments = np.diff(middle_ts_per_window, axis=0)  # Shape: (N_WINDOWS-1)

        # Avoid division by zero or very small dt
        valid_dt_mask = dt_segments > 1e-9

        if not np.any(valid_dt_mask):
            logger.warning(
                "ACCEL_PAIRS: All dt for acceleration segments are too small or zero. Returning zero."
            )
            return zero_vector(twod)

        # Initialize accelerations_segments with zeros
        acceleration_segments_np = np.zeros_like(dv_segments)

        # Perform division only where dt is valid
        # dt_segments needs to be broadcastable: (N_WINDOWS-1,) -> (N_WINDOWS-1, 1) for division
        acceleration_segments_np[valid_dt_mask] = (
            dv_segments[valid_dt_mask] / dt_segments[valid_dt_mask, np.newaxis]
        )

        # Average the valid acceleration segments
        if (
            np.sum(valid_dt_mask) == 0
        ):  # Should be caught by np.any above, but as a safeguard
            logger.warning(
                "ACCEL_PAIRS: No valid dt segments after filtering. Returning zero."
            )
            return zero_vector(twod)

        final_accel_np = np.sum(
            acceleration_segments_np[valid_dt_mask], axis=0
        ) / np.sum(valid_dt_mask)

        # logger.debug(f"ACCEL_PAIRS_RESULT: final_accel_np={final_accel_np}")

        # Convert final NumPy array back to VectorObject
        if twod:
            return Vector2D(final_accel_np[0], final_accel_np[1])
        else:
            return Vector3D(final_accel_np[0], final_accel_np[1], final_accel_np[2])
