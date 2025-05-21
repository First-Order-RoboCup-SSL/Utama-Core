from dataclasses import replace
from entities.game.game import Game
from entities.game.robot import Robot

from typing import Dict, List # Added List for type hinting
# Assuming your new PastGame is in entities.game.past_game or accessible
from entities.game.past_game import PastGame, AttributeType, TeamType, ObjectClass, ObjectKey, get_structured_object_key
from refiners.base_refiner import BaseRefiner
from vector import VectorObject2D, VectorObject3D # Your custom vector classes
from lenses import UnboundLens, lens
import logging
import numpy as np # Import NumPy

logger = logging.getLogger(__name__)

def zero_vector(twod: bool) -> VectorObject2D | VectorObject3D: # Added return type hint
    return VectorObject2D(x=0, y=0) if twod else VectorObject3D(x=0, y=0, z=0)

class VelocityRefiner(BaseRefiner):
    ACCELERATION_WINDOW_SIZE = 5
    ACCELERATION_N_WINDOWS = 3

    def refine(self, past_game: PastGame, game: Game) -> Game:
        current_game_ts = game.ts

        # Process Ball (Keep this commented if you want to focus on robots first)
        if game.ball: # Ensure ball processing is guarded
            game = self._refine_ball_kinematics(past_game, game, current_game_ts)

        # Process Friendly Robots
        game = self._refine_robot_group(past_game, game, current_game_ts,
                                        game.friendly_robots, TeamType.FRIENDLY,
                                        lens.friendly_robots, twod=True)

        # Process Enemy Robots
        game = self._refine_robot_group(past_game, game, current_game_ts,
                                    game.enemy_robots, TeamType.ENEMY,
                                    lens.enemy_robots, twod=True)
        return game

    def _refine_robot_group(self,
        past_game: PastGame,
        game_state: Game,
        current_ts: float,
        robots_to_process_dict: Dict[int, Robot],
        team_type: TeamType,
        group_lens: UnboundLens,
        twod: bool) -> Game:
        updated_robots_dict = {}
        for robot_instance in robots_to_process_dict.values():
            robot_id = getattr(robot_instance, 'id', None)
            if robot_id is None:
                logger.error(f"{team_type.name} robot instance encountered without an ID. Skipping.")
                continue

            if robot_instance.p is None:
                logger.warning(f"{team_type.name} robot {robot_id} has no position. Setting zero v/a.")
                updated_robot = robot_instance & lens.v.set(zero_vector(twod)) & lens.a.set(zero_vector(twod))
                updated_robots_dict[robot_id] = updated_robot
                continue

            robot_obj_key = get_structured_object_key(robot_instance, team_type)
            if not robot_obj_key:
                logger.error(f"Could not get ObjectKey for {team_type.name} robot {robot_id}. Adding original to dict.")
                updated_robots_dict[robot_id] = robot_instance
                continue
            
            new_v = self._calculate_object_velocity(past_game, robot_instance.p, robot_obj_key, current_ts, twod)
            
            new_a = zero_vector(twod) # Default to zero
            try:
                new_a = self._calculate_object_acceleration(past_game, robot_obj_key, twod)
            except Exception as e:
                logger.warning(f"Could not calculate acceleration for {team_type.name} robot {robot_id} (key: {robot_obj_key}), setting to zero: {e}")
                
            updated_robot = robot_instance & lens.v.set(new_v) & lens.a.set(new_a)
            updated_robots_dict[robot_id] = updated_robot
            
        return game_state & group_lens.set(updated_robots_dict)

    def _refine_ball_kinematics(self,
                                past_game: PastGame,
                                game_state: Game,
                                current_ts: float) -> Game:
        if not game_state.ball:
            return game_state
        
        if game_state.ball.p is None:
            logger.warning("Ball exists but has no position data; setting zero velocity and acceleration.")
            return game_state & lens.ball.v.set(zero_vector(twod=False)) & lens.ball.a.set(zero_vector(twod=False))

        ball_obj_key = get_structured_object_key(game_state.ball, TeamType.NEUTRAL)
        if not ball_obj_key:
            logger.error("Could not get ObjectKey for ball. Skipping ball refinement.")
            return game_state

        new_ball_v = self._calculate_object_velocity(past_game, game_state.ball.p, ball_obj_key, current_ts, twod=False)
        game_state &= lens.ball.v.set(new_ball_v)

        new_ball_a = zero_vector(twod=False) # Default to zero
        try:
            new_ball_a = self._calculate_object_acceleration(past_game, ball_obj_key, twod=False)
        except Exception as e:
            logger.warning(f"Could not calculate acceleration for ball (key: {ball_obj_key}), setting to zero: {e}")
        game_state &= lens.ball.a.set(new_ball_a)
        
        return game_state
    
    def _calculate_object_velocity(self,
        past_game: PastGame,
        current_pos: VectorObject2D | VectorObject3D,
        object_key: ObjectKey,
        current_ts: float,
        twod: bool) -> VectorObject2D | VectorObject3D:
        try:
            past_pos_data = past_game.get_historical_attribute_series(
                object_key, AttributeType.POSITION, 1
            )
            if not past_pos_data:
                raise ValueError(f"No historical position for {object_key}.")

            previous_time_received, previous_pos = past_pos_data[0]
            dt_secs = current_ts - previous_time_received

            if dt_secs <= 1e-9:
                logger.warning(f"Object {object_key} velocity dt_secs is too small or zero ({dt_secs}). Using zero velocity.")
                return zero_vector(twod)
            
            # Assuming your VectorObject already supports subtraction and division by scalar
            return (current_pos - previous_pos) / dt_secs
        except Exception as e:
            logger.warning(f"Could not calculate velocity for {object_key}, setting to zero: {e}")
            return zero_vector(twod)
    
    def _calculate_object_acceleration(self, past_game: PastGame, object_key: ObjectKey, twod: bool) -> VectorObject2D | VectorObject3D:
        try:
            pairs = self._extract_time_velocity_pairs(past_game, object_key)
            num_points_needed = self.ACCELERATION_N_WINDOWS * self.ACCELERATION_WINDOW_SIZE
            if not pairs or len(pairs) < num_points_needed: # Check against total points needed
                raise ValueError(f"Not enough velocity pairs for {object_key}. Have {len(pairs)}, need {num_points_needed}")
        except Exception as e:
            raise ValueError(f"Velocity data not available for acceleration for {object_key}: {e}") from e
        
        return self._calculate_acceleration_from_pairs_np(pairs, twod) # Call NumPy version

    def _extract_time_velocity_pairs(self, past_game: PastGame, object_key: ObjectKey) -> List[tuple[float, VectorObject2D | VectorObject3D]]:
        num_points_needed = self.ACCELERATION_N_WINDOWS * self.ACCELERATION_WINDOW_SIZE
        
        time_velocity_pairs = past_game.get_historical_attribute_series(
            object_key,
            AttributeType.VELOCITY,
            num_points_needed
        )
        return time_velocity_pairs # type: ignore

    def _calculate_acceleration_from_pairs_np(self, time_velocity_pairs: List[tuple[float, VectorObject2D | VectorObject3D]], twod: bool) -> VectorObject2D | VectorObject3D:
        """
        Estimates an object's acceleration using NumPy for calculations.
        """
        min_points_for_one_window = self.ACCELERATION_WINDOW_SIZE
        if not time_velocity_pairs or len(time_velocity_pairs) < min_points_for_one_window:
             raise ValueError(f"Not enough data points ({len(time_velocity_pairs)}) for acceleration calculation (need at least {min_points_for_one_window}).")

        if self.ACCELERATION_N_WINDOWS == 0:
            return zero_vector(twod)

        # Convert list of (timestamp, VectorObject) to NumPy arrays
        num_dimensions = 2 if twod else 3
        timestamps_np = np.array([pair[0] for pair in time_velocity_pairs], dtype=np.float64)
        
        velocities_list_of_lists = []
        if twod:
            for _, vel_obj in time_velocity_pairs:
                velocities_list_of_lists.append([vel_obj.x, vel_obj.y])
        else:
            for _, vel_obj in time_velocity_pairs:
                velocities_list_of_lists.append([vel_obj.x, vel_obj.y, vel_obj.z]) # type: ignore
        velocities_np = np.array(velocities_list_of_lists, dtype=np.float64)

        if velocities_np.shape[0] < self.ACCELERATION_N_WINDOWS * self.ACCELERATION_WINDOW_SIZE:
            # This check is important if _extract_time_velocity_pairs might return fewer than absolutely required
            # which it shouldn't if the check in _calculate_object_acceleration is correct.
            raise ValueError(f"Converted NumPy arrays have insufficient velocity points for all windows for acceleration calculation.")


        total_accel_np = np.zeros(num_dimensions, dtype=np.float64)
        num_accel_calcs = 0
        
        prev_window_avg_velocity_np = None
        prev_window_middle_ts_np = None

        for i in range(self.ACCELERATION_N_WINDOWS):
            window_start_index = i * self.ACCELERATION_WINDOW_SIZE
            window_end_index = window_start_index + self.ACCELERATION_WINDOW_SIZE
            
            if window_end_index > velocities_np.shape[0]:
                logger.warning(f"Not enough data points in NumPy array for acceleration window {i}. Skipping.")
                continue

            current_window_velocities_np = velocities_np[window_start_index:window_end_index]
            current_window_timestamps_np = timestamps_np[window_start_index:window_end_index]

            if current_window_velocities_np.shape[0] == 0: # Should be caught by above check too
                continue

            current_window_avg_velocity_np = np.mean(current_window_velocities_np, axis=0)
            current_window_middle_ts_np = np.mean(current_window_timestamps_np)

            if prev_window_avg_velocity_np is not None and prev_window_middle_ts_np is not None:
                dt = current_window_middle_ts_np - prev_window_middle_ts_np
                
                if dt <= 1e-9:
                    logger.warning(f"Acceleration dt (NumPy) is too small or zero ({dt}). Skipping this accel calculation.")
                else:
                    accel_segment_np = (current_window_avg_velocity_np - prev_window_avg_velocity_np) / dt
                    total_accel_np += accel_segment_np
                    num_accel_calcs += 1
            
            prev_window_avg_velocity_np = current_window_avg_velocity_np
            prev_window_middle_ts_np = current_window_middle_ts_np

        if num_accel_calcs == 0:
            if self.ACCELERATION_N_WINDOWS > 1:
                logger.warning("No acceleration segments could be calculated (NumPy).")
            return zero_vector(twod)
        
        final_accel_np = total_accel_np / num_accel_calcs
        
        # Convert final NumPy array back to VectorObject
        if twod:
            return VectorObject2D(x=final_accel_np[0], y=final_accel_np[1])
        else:
            return VectorObject3D(x=final_accel_np[0], y=final_accel_np[1], z=final_accel_np[2])

    # Keep the original _calculate_acceleration_from_pairs if needed for comparison or fallback,
    # otherwise, it can be removed if _calculate_acceleration_from_pairs_np is the definitive version.
    # For this example, I've renamed the NumPy version and changed the call in _calculate_object_acceleration.