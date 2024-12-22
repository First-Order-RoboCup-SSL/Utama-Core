import os
import sys
import time
import numpy as np
from typing import Tuple, List, Union, Dict, Optional

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
print(project_root)
sys.path.insert(0, project_root)

from team_controller.src.controllers import RSimRobotController
from entities.data.command import RobotCommand
from entities.game import Game

from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv

from entities.game import Field
from entities.data.command import RobotCommand
from entities.data.vision import BallData, RobotData
from global_utils.math_utils import rotate_vector
from motion_planning.src.pid.pid import PID
from team_controller.src.controllers import RSimRobotController

# Constants
ROBOT_RADIUS = 0.18


def ball_to_robot_dist(
    ball_x: float, ball_y: float, robot_x: float, robot_y: float
) -> float:
    return np.sqrt((ball_y - robot_y) ** 2 + (ball_x - robot_x) ** 2)


def angle_to_robot(
    ball_x: float, ball_y: float, robot_x: float, robot_y: float
) -> float:
    return np.arctan((robot_y - ball_y) / (robot_x - ball_x))


# Calculates the intersection of 2 rays with the goal
def shadow(
    start_x: float, start_y: float, angle1: float, angle2: float, goal_x: float
) -> Tuple[float, float]:
    slope1: float = np.tan(angle1)
    slope2: float = np.tan(angle2)
    shadow_start: float = start_y + slope1 * (goal_x - start_x)
    shadow_end: float = start_y + slope2 * (goal_x - start_x)
    return tuple(sorted((shadow_start, shadow_end)))


# Filters the shadows to only keep the ones relevant to the shot and merges overlapping shadows
def filter_and_merge_shadows(
    shadows: List[Tuple[float, float]], goal_y1: float, goal_y2: float
) -> List[Tuple[float, float]]:
    valid_shadows: List[Tuple[float, float]] = []

    for start, end in shadows:
        if start > goal_y1 and start < goal_y2:
            if end > goal_y2:
                end = goal_y2
            valid_shadows.append((start, end))
        elif end > goal_y1 and end < goal_y2:
            if start < goal_y1:
                start = goal_y1
            valid_shadows.append((start, end))

    valid_shadows.sort()

    merged_shadows: List[Tuple[float, float]] = []
    for start, end in valid_shadows:
        if not merged_shadows or merged_shadows[-1][1] < start:
            merged_shadows.append((start, end))
        else:
            merged_shadows[-1] = (
                merged_shadows[-1][0],
                max(merged_shadows[-1][1], end),
            )

    return merged_shadows


# Casts a ray along the 2 tangents to each enemy robot, and calls filter_and_merge_shadows
def ray_casting(
    ball: Tuple[float, float],
    enemy_robots: List[Tuple[float, float]],
    goal_x: float,
    goal_y1: float,
    goal_y2: float,
) -> List[Tuple[float, float]]:
    shadows: List[Tuple[float, float]] = []
    for enemy in enemy_robots:
        if enemy != None:
            if enemy.x > ball.x:
                dist: float = ball_to_robot_dist(ball.x, ball.y, enemy.x, enemy.y)
                angle_to_robot_: float = angle_to_robot(
                    ball.x, ball.y, enemy.x, enemy.y
                )
                alpha: float = np.arcsin(ROBOT_RADIUS / dist)
                shadows.append(
                    shadow(
                        ball.x,
                        ball.y,
                        angle_to_robot_ + alpha,
                        angle_to_robot_ - alpha,
                        goal_x,
                    )
                )
                shadows = filter_and_merge_shadows(shadows, goal_y1, goal_y2)
    return shadows


# Finds the biggest area of the goal that doesn't have a shadow (the biggest gap) and finds its midpoint for best shot
# TODO: could add heuristics to prefer shots closer to the goalpost
def find_best_shot(
    shadows: List[Tuple[float, float]], goal_y1: float, goal_y2: float
) -> float:
    if not shadows:
        return (goal_y2 + goal_y1) / 2

    open_spaces: List[Tuple[float, float]] = []

    if shadows[0][0] > goal_y1:
        open_spaces.append((goal_y1, shadows[0][0]))

    for i in range(1, len(shadows)):
        if shadows[i][0] > shadows[i - 1][1]:
            open_spaces.append((shadows[i - 1][1], shadows[i][0]))

    if shadows[-1][1] < goal_y2:
        open_spaces.append((shadows[-1][1], goal_y2))

    largest_gap: Tuple[float, float] = max(open_spaces, key=lambda x: x[1] - x[0])
    best_shot: float = (largest_gap[0] + largest_gap[1]) / 2

    return best_shot


class ShootingController:
    def __init__(
        self,
        shooter_id,
        goal_x,
        goal_y1,
        goal_y2,
        game_obj: Game,
        robot_controller: RSimRobotController,
        debug=False,
    ):
        self.game_obj = game_obj
        self.robot_controller = robot_controller

        self.first_action = False

        self.robot_command = RobotCommand(
            local_forward_vel=0,
            local_left_vel=0,
            angular_vel=0,
            kick_spd=0,
            kick_angle=0,
            dribbler_spd=0,
        )

        self.goal_x = goal_x
        self.goal_y1 = goal_y1
        self.goal_y2 = goal_y2

        # TODO: Tune PID parameters further when going from sim to real(it works for Grsim)
        # potentially have a set tunig parameters for each robot
        self.pid_oren = PID(0.0167, 8, -8, 4.5, 0, 0.045, num_robots=6)
        self.pid_trans = PID(0.0167, 1.5, -1.5, 4.5, 0, 0.035, num_robots=6)

        self.debug = debug
        self.shooter_id = shooter_id

    # Added this function
    def kick_ball(self, current_oren: float = None, target_oren: float = None):
        if np.round(target_oren, 1) and np.round(current_oren, 1):
            # print(f"{np.round(target_oren, 2) - np.round(current_oren, 2)}")
            if abs(np.round(target_oren, 2) - np.round(current_oren, 2)) <= 0.02:
                self.robot_command = self.robot_command._replace(
                    kick_spd=3, kick_angle=0, dribbler_spd=0
                )
                print("Kicking ball\n")
                return True
            else:
                self.robot_command = self.robot_command._replace(
                    kick_spd=0, kick_angle=0, dribbler_spd=1
                )
                print("Dribbling ball\n")
                return False

    # makes all the descisions for the robot
    def approach_ball(self):
        robots, enemy_robots, balls = self._get_positions()

        if robots and balls:
            shadows = ray_casting(
                balls[0], enemy_robots, self.goal_x, self.goal_y1, self.goal_y2
            )
            best_shot = find_best_shot(shadows, self.goal_y1, self.goal_y2)
            # Changed to atan2 to get the correct angle
            shot_orientation = np.atan2(
                (best_shot - balls[0].y), (self.goal_x - balls[0].x)
            )

            robot_data = (
                robots[self.shooter_id] if self.shooter_id < len(robots) else None
            )

            # Lost of changed here, added a lot of print statements to debug
            if balls[0] != None and robot_data != None:
                target_oren = np.atan2(
                    balls[0].y - robot_data.y, balls[0].x - robot_data.x
                )
                if robot_data is not None:
                    if self.first_action or (
                        abs(
                            np.round(target_oren, 1)
                            - np.round(robot_data.orientation, 1)
                        )
                        >= 0.3
                        and self.robot_controller.robot_has_ball(self.shooter_id)
                    ):
                        print("first action")
                        target_coords = (None, None, None)
                        face_ball = True
                        self.robot_command = self._calculate_robot_velocities(
                            self.shooter_id,
                            target_coords,
                            robots,
                            balls,
                            face_ball=face_ball,
                        )
                        self.first_action = False
                    elif self.robot_controller.robot_has_ball(self.shooter_id):
                        print("robot has ball")
                        current_oren = robots[self.shooter_id].orientation
                        face_ball = False
                        target_coords = (None, None, shot_orientation)

                        self.robot_command = self._calculate_robot_velocities(
                            self.shooter_id,
                            target_coords,
                            robots,
                            balls,
                            face_ball=face_ball,
                        )
                        self.first_action = self.kick_ball(
                            current_oren, shot_orientation
                        )
                    else:
                        print("approaching ball")
                        face_ball = True
                        target_coords = (balls[0].x, balls[0].y, None)
                        self.robot_command = self._calculate_robot_velocities(
                            self.shooter_id,
                            target_coords,
                            robots,
                            balls,
                            face_ball=face_ball,
                        )

                # print(self.robot_command, "\n")
                self.robot_controller.add_robot_commands(
                    self.robot_command, robot_id=self.shooter_id
                )
                # print(self.robot_controller.out_packet)
                self.robot_controller.send_robot_commands()

    def _get_positions(self) -> tuple:
        # Fetch the latest positions of robots and balls with thread locking.
        robots = self.game_obj.get_robots_pos(is_yellow=True)
        enemy_robots = self.game_obj.get_robots_pos(is_yellow=False)
        balls = self.game_obj.get_ball_pos()
        return robots, enemy_robots, balls

    def _calculate_robot_velocities(
        self,
        robot_id: int,
        target_coords: Union[Tuple[float, float], Tuple[float, float, float]],
        robots: List[RobotData],
        balls: List[BallData],
        face_ball=False,
    ) -> RobotCommand:
        """
        Calculates the linear and angular velocities required for a robot to move towards a specified target position
        and orientation.

        Args:
            robot_id (int): Unique identifier for the robot.
            target_coords (Tuple[float, float] | Tuple[float, float, float]): Target coordinates the robot should move towards.
                Can be a (x, y) or (x, y, orientation) tuple. If `face_ball` is True, the robot will face the ball instead of
                using the orientation value in target_coords.
            robots (Dict[int, Optional[Tuple[float, float, float]]]): All the Current coordinates of the robots sepateated
                by thier robot_id which containts a tuple (x, y, orientation).
            balls (Dict[int, Tuple[float, float, float]]): All the Coordinates of the detected balls (int) , typically (x, y, z/height in 3D space).            face_ball (bool, optional): If True, the robot will orient itself to face the ball's position. Defaults to False.

        Returns:
            Dict[str, float]: A dictionary containing the following velocity components:
                - "id" (int): Robot identifier.
                - "xvel" (float): X-axis velocity to move towards the target.
                - "yvel" (float): Y-axis velocity to move towards the target.
                - "wvel" (float): Angular velocity to adjust the robot's orientation.

        The method uses PID controllers to calculate velocities for linear and angular movement. If `face_ball` is set,
        the robot will calculate the angular velocity to face the ball. The resulting x and y velocities are rotated to align
        with the robot's current orientation.
        """

        # Get current positions
        if balls[0] and robots[robot_id]:
            ball_x, ball_y, ball_z = balls[0]
            current_x, current_y, current_oren = robots[robot_id]

        target_x, target_y = target_coords[:2]

        if face_ball:
            target_oren = np.atan2(ball_y - current_y, ball_x - current_x)
        elif not face_ball and len(target_coords) == 3:
            target_oren = target_coords[2]

        # print(f"\nRobot {robot_id} current position: ({current_x:.3f}, {current_y:.3f}, {current_oren:.3f})")
        # print(f"Robot {robot_id} target position: ({target_x:.3f}, {target_y:.3f}, {target_oren:.3f})")

        if target_oren != None:
            angular_vel = self.pid_oren.calculate(
                target_oren, current_oren, robot_id, oren=True
            )
        else:
            angular_vel = 0

        if target_x != None and target_y != None:
            left_vel = self.pid_trans.calculate(
                target_y, current_y, robot_id, normalize_range=3
            )
            forward_vel = self.pid_trans.calculate(
                target_x, current_x, robot_id, normalize_range=4.5
            )

            forward_vel, left_vel = rotate_vector(forward_vel, left_vel, current_oren)
        else:
            forward_vel = 0
            left_vel = 0
        # print(f"Output: {forward_vel}, {left_vel}, {angular_vel}")
        return RobotCommand(
            local_forward_vel=forward_vel,
            local_left_vel=left_vel,
            angular_vel=angular_vel,
            kick_spd=0,
            kick_angle=0,
            dribbler_spd=0,
        )


field = Field()

shooter_id = 3
goal_x = -field.half_length
goal_y1 = -field.half_goal_width
goal_y2 = field.half_goal_width

if __name__ == "__main__":
    game = Game()

    # making environment
    env = SSLStandardEnv(n_robots_blue=3)
    env.reset()
    env.teleport_robot(False, 0, x=1, y=1)
    env.teleport_ball(0, 0)
    # Note we don't need a vision receiver for rsim
    sim_robot_controller = RSimRobotController(
        is_team_yellow=True, env=env, game_obj=game, debug=False
    )
    decision_maker = ShootingController(
        shooter_id,
        goal_x,
        goal_y1,
        goal_y2,
        game,
        sim_robot_controller,
        debug=True,
    )

    try:
        while True:
            decision_maker.approach_ball()
    except KeyboardInterrupt:
        print("Exiting...")
