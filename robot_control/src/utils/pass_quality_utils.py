import numpy as np
from entities.game.robot import RobotData
from global_utils.math_utils import distance

ROBOT_RADIUS = 0.09


class PointOnField:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def x(self) -> float:
        self.x

    def y(self) -> float:
        self.y


def ball_position(t, x0, v0, a):
    x0 = np.array(x0)
    v0 = np.array(v0)
    a = np.array(a)
    return x0[0:2] + v0 * t + 0.5 * a * (t**2)


def interception_chance(
    passer, receiver, opponent, robot_speed, ball_v0_magnitude, ball_a_magnitude
):

    # assert type(passer) != tuple
    # assert type(receiver) != tuple
    # robot_speed = np.linalg.norm(robot_velocity)
    passer_vector = np.array([passer.x, passer.y])
    receiver_vector = np.array([receiver.x, receiver.y])
    opp_vector = np.array([opponent.x, opponent.y])
    passer_to_receiver_vec = receiver_vector - passer_vector
    pr_unit = passer_to_receiver_vec / np.linalg.norm(passer_to_receiver_vec)
    ball_v0 = pr_unit * ball_v0_magnitude
    ball_a = pr_unit * ball_a_magnitude

    passer_to_opp_vec = opp_vector - passer_vector
    projection = np.dot(passer_to_opp_vec, passer_to_receiver_vec) / np.dot(
        passer_to_receiver_vec, passer_to_receiver_vec
    )
    if projection < 0 or projection > 1:
        return 0, None, None
    closest_point = passer_vector + projection * passer_to_opp_vec
    ball_distance = np.linalg.norm(closest_point - passer_vector)
    ball_speed = np.linalg.norm(ball_v0)
    ball_time_roots = np.roots(
        [0.5 * np.linalg.norm(ball_a), ball_speed, -ball_distance]
    )
    ball_time = (
        np.max(ball_time_roots[ball_time_roots > 0])
        if np.any(ball_time_roots > 0)
        else float("inf")
    )

    opp_dist_to_pass = np.linalg.norm(closest_point - opp_vector) - ROBOT_RADIUS
    if opp_dist_to_pass > 0:
        opp_to_pass_time = (
            opp_dist_to_pass / robot_speed if robot_speed != 0 else float("inf")
        )
    else:
        opp_to_pass_time = 0

    if opp_to_pass_time <= ball_time:
        ball_pos = ball_position(opp_to_pass_time, passer, ball_v0, ball_a)
        opp_to_ball_dist = np.linalg.norm(ball_pos - closest_point)
        chance = np.log(1 + opp_to_ball_dist)
    else:
        chance = 0
        return 0, None, None

    return chance, closest_point, ball_pos


def find_pass_quality(
    passer,
    receiver,
    enemy_positions,
    enemy_speeds,
    ball_v0_magnitude,
    ball_a_magnitude,
    goal_x,
    goal_y1,
    goal_y2,
    shoot_in_left_goal,
):
    from robot_control.src.utils.shooting_utils import find_shot_quality

    total_interception_chance = 0
    for enemy_pos, enemy_speed in zip(enemy_positions, enemy_speeds):
        interception, _, _ = interception_chance(
            passer,
            receiver,
            enemy_pos,
            enemy_speed,
            ball_v0_magnitude,
            ball_a_magnitude,
        )
        total_interception_chance += interception
    goal_chance = find_shot_quality(
        receiver, enemy_positions, goal_x, goal_y1, goal_y2, shoot_in_left_goal
    )

    distance_to_goal_ratio = (np.absolute(receiver.x - goal_x)) / np.absolute(
        2 * goal_x
    )

    distance_to_passer = distance((passer.x, passer.y), (receiver.x, receiver.y))

    # these will be adjusted
    interception_chance_weight = 3
    goal_chance_weight = 0.5
    distance_to_goal_weight = 0.2

    if distance_to_passer >= 0.7:
        pass_quality = (
            1
            - interception_chance_weight * total_interception_chance
            + goal_chance_weight * goal_chance
            - distance_to_goal_weight * distance_to_goal_ratio
        )  # pass quality metric
    else:
        pass_quality = float("-inf")

    return pass_quality


def find_best_pass(
    passer,
    friendly_robots,
    enemy_positions,
    enemy_speeds,
    ball_v0_magnitude,
    ball_a_magnitude,
    goal_x,
    goal_y1,
    goal_y2,
    shoot_in_left_goal,
):
    best_quality = -float("inf")
    best_receiver = None
    pass_qualities = []

    for receiver in friendly_robots:
        pass_quality = find_pass_quality(
            passer,
            receiver,
            enemy_positions,
            enemy_speeds,
            ball_v0_magnitude,
            ball_a_magnitude,
            goal_x,
            goal_y1,
            goal_y2,
            shoot_in_left_goal,
        )
        pass_qualities.append(pass_quality)
        if pass_quality > best_quality:
            best_quality = pass_quality
            best_receiver = receiver

    return best_receiver, pass_qualities


def find_best_receiver_position(
    receiver_position,
    passer,
    enemy_positions,
    enemy_speeds,
    ball_v0_magnitude,
    ball_a_magnitude,
    goal_x,
    goal_y1,
    goal_y2,
    shoot_in_left_goal,
    field_limits=[(-4.5, 4.5), (-3.0, 3.0)],
    sample_radius=0.7,
    num_samples=10,
):
    """
    Finds the best nearby receiver position that maximizes pass quality.

    Returns:
        - best_position: (x, y) coordinates of the best receiver position
        - sampled_positions: list of (x, y) all sampled points including the original receiver position
        - pass_qualities: list of pass quality values corresponding to sampled positions
    """

    x_min, x_max = field_limits[0]
    y_min, y_max = field_limits[1]

    best_position = receiver_position
    best_quality = -float("inf")

    sampled_positions = [
        PointOnField(receiver_position.x, receiver_position.y)
    ]  # Include the current position
    pass_qualities = []  # Store pass quality for each sampled point

    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)

    for angle in angles:
        new_x = receiver_position.x + sample_radius * np.cos(angle)
        new_y = receiver_position.y + sample_radius * np.sin(angle)

        # Ensure the sampled position is within the field
        if x_min <= new_x <= x_max and y_min <= new_y <= y_max:
            sampled_positions.append(PointOnField(new_x, new_y))

    for candidate in sampled_positions:
        pass_quality = find_pass_quality(
            passer,
            candidate,
            enemy_positions,
            enemy_speeds,
            ball_v0_magnitude,
            ball_a_magnitude,
            goal_x,
            goal_y1,
            goal_y2,
            shoot_in_left_goal,
        )
        pass_qualities.append(pass_quality)
        # pass_quality += 100 * squared_distance(receiver_position, passer)
        if pass_quality > best_quality:
            best_quality = pass_quality
            best_position = candidate

    return best_position, sampled_positions, pass_qualities
