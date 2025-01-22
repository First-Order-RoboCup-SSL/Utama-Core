import logging
import random
import math

from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game

from robot_control.src.tests.utils import setup_pvp
from motion_planning.src.pid.pid import get_rsim_pids
from robot_control.src.skills import face_ball, go_to_point
from robot_control.src.intent import score_goal

logger = logging.getLogger(__name__)

ITERS = 2000


def predict_ball_position(ball, steps=10, ball_velocity=(0, 0)):
    """
    Predict the future position of the ball based on its current velocity.

    :param ball: Current position of the ball (ball.x, ball.y)
    :param steps: Number of prediction steps
    :param ball_velocity: Current velocity of the ball (vx, vy)
    :return: (predicted_x, predicted_y) - The future position of the ball
    """
    predicted_x = ball.x + ball_velocity[0] * steps
    predicted_y = ball.y + ball_velocity[1] * steps
    return predicted_x, predicted_y


def dynamic_block_ratio(attacker, ball, goal_pos, distance_factor=1.0):
    """
    Dynamically adjust block ratio based on the distance between the attacker, the ball, and the goal.
    The 'distance_factor' parameter allows you to increase or decrease how aggressively the defender
    positions itself between the attacker and the goal.

    :param attacker: Attacker position (attacker.x, attacker.y)
    :param ball: Ball position (ball.x, ball.y)
    :param goal_pos: Goal position (x, y)
    :param distance_factor: A multiplier to adjust how close the defender stays to the attacker.
                           > 1.0 will position the defender closer to the attacker,
                           < 1.0 will position it farther away.
    :return: Adjusted block ratio
    """
    attacker_to_goal = math.hypot(attacker.x - goal_pos[0], attacker.y - goal_pos[1])
    ball_to_goal = math.hypot(ball.x - goal_pos[0], ball.y - goal_pos[1])

    # Basic ratio based on relative distances
    ratio = min(0.5, max(0.2, ball_to_goal / (attacker_to_goal + 1e-6)))

    # Multiply by 'distance_factor' so we can fine-tune how aggressively we want to block
    ratio *= distance_factor

    return ratio


def improved_block_goal_and_attacker(
    robot, attacker, ball, goal_pos, 
    pid_oren, pid_trans,
    attacker_has_ball: bool,
    max_ball_follow_dist: float = 1.0,
    ball_velocity=(0, 0),
    distance_factor=1.0
):
    """
    Enhanced defensive strategy using prediction and dynamic adjustments.
    Now includes 'distance_factor' to dynamically adjust how close the defender
    stands between the attacker and the goal.

    :param robot: Current state of the defender robot (robot.x, robot.y)
    :param attacker: Attacker robot (attacker.x, attacker.y)
    :param ball: Ball (ball.x, ball.y)
    :param goal_pos: Center position of the goal being defended (x, y)
    :param pid_oren: PID controller (orientation)
    :param pid_trans: PID controller (translation)
    :param attacker_has_ball: Whether the attacker has possession of the ball
    :param max_ball_follow_dist: Maximum allowed distance from the defender to the ball when following it
    :param ball_velocity: Current velocity of the ball (vx, vy)
    :param distance_factor: A multiplier to adjust how close the defender stays to the attacker.
                           > 1.0 -> More aggressive (closer)
                           < 1.0 -> Less aggressive (farther)
    :return: Commands to be used with sim_robot_controller_defender.add_robot_commands(cmd, 0)
    """
    if attacker_has_ball:
        # --- Predict attacker trajectory and block the route --- #
        predicted_ball_x, predicted_ball_y = predict_ball_position(ball, steps=10, ball_velocity=ball_velocity)
        block_ratio = dynamic_block_ratio(attacker, ball, goal_pos, distance_factor=distance_factor)

        ax, ay = attacker.x, attacker.y
        gx, gy = goal_pos[0], goal_pos[1]
        
        agx, agy = gx - ax, gy - ay
        dist_ag = math.hypot(agx, agy)
        if dist_ag < 1e-6:
            target_x, target_y = gx, gy
        else:
            target_x = ax + block_ratio * agx
            target_y = ay + block_ratio * agy
        
        # Orient the defender to face the attacker
        face_theta = math.atan2((ay - robot.y), (ax - robot.x))
    else:
        # --- Protect the ball while considering future positions --- #
        predicted_ball_x, predicted_ball_y = predict_ball_position(ball, steps=10, ball_velocity=ball_velocity)
        ax, ay = attacker.x, attacker.y
        bx, by = predicted_ball_x, predicted_ball_y

        abx, aby = bx - ax, by - ay
        target_x = ax + 0.6 * abx
        target_y = ay + 0.6 * aby

        dist_def_to_ball = math.hypot(robot.x - bx, robot.y - by)
        if dist_def_to_ball > max_ball_follow_dist:
            ratio = max_ball_follow_dist / dist_def_to_ball
            target_x = robot.x + (bx - robot.x) * ratio
            target_y = robot.y + (by - robot.y) * ratio

        # Orient the defender to face the ball
        face_theta = math.atan2((by - robot.y), (bx - robot.x))

    # Generate movement commands
    cmd = go_to_point(
        pid_oren,
        pid_trans,
        robot,
        0,  # robot_id
        (target_x, target_y),
        face_theta
    )
    return cmd


def test_ultimate_one_on_one(defender_is_yellow: bool, headless: bool, distance_factor=1.0):
    """
    A "stronger" one-on-one attack-defense test example, combining attacking, shooting, defending, and intercepting.

    :param defender_is_yellow: If True, the yellow team is the defender, and the blue team is the attacker; otherwise, the roles are reversed.
    :param headless: If True, the simulation will not render images; if False, the simulation will display the process in a window.
    :param distance_factor: A multiplier to dynamically adjust how close the defender positions itself 
                            in relation to the attacker. (> 1.0 = closer, < 1.0 = farther)
    """
    # 1. Initialize the game and environment: each team has one robot
    game = Game()
    env = SSLStandardEnv(
        n_robots_blue=1 if defender_is_yellow else 1,
        n_robots_yellow=1 if not defender_is_yellow else 1,
        render_mode="ansi" if headless else "human",
    )
    env.reset()
    
    # 2. Randomly place the ball
    ball_x = random.uniform(-2.5, 2.5)
    ball_y = random.uniform(-1.5, 1.5)
    env.teleport_ball(ball_x, ball_y)

    # 3. Set up PID controllers
    pid_oren_y, pid_2d_y = get_rsim_pids(1)  # Yellow team PID
    pid_oren_b, pid_2d_b = get_rsim_pids(1)  # Blue team PID

    # 4. Initialize PVP mode controllers (sim_robot_controller)
    sim_robot_controller_yellow, sim_robot_controller_blue, pvp_manager = setup_pvp(
        env, game, n_robots_blue=1, n_robots_yellow=1
    )

    # Assign PIDs to the attacker and defender
    if defender_is_yellow:
        sim_robot_controller_defender = sim_robot_controller_yellow
        sim_robot_controller_attacker = sim_robot_controller_blue

        pid_oren_d, pid_2d_d = pid_oren_y, pid_2d_y
        pid_oren_a, pid_2d_a = pid_oren_b, pid_2d_b
    else:
        sim_robot_controller_defender = sim_robot_controller_blue
        sim_robot_controller_attacker = sim_robot_controller_yellow

        pid_oren_d, pid_2d_d = pid_oren_b, pid_2d_b
        pid_oren_a, pid_2d_a = pid_oren_y, pid_2d_y

    # 5. Define attack and defense targets (assuming the attacker aims for the opponent's goal)
    goal_pos = (4.5, 0) if defender_is_yellow else (-4.5, 0)

    goal_scored = False

    for i in range(ITERS):
        if goal_scored:
            break

        # Get the latest frame information for both teams (from the defender's perspective)
        friendly, enemy, balls = game.get_my_latest_frame(my_team_is_yellow=defender_is_yellow)
        ball = balls[0]

        # ========== Attacker logic: shoot or move toward the ball ========== 
        attacker = enemy[0]  # Assumes there is only 1 enemy
        has_ball = sim_robot_controller_attacker.robot_has_ball(0)

        # If the attacker has the ball, use the "score_goal" function to shoot
        if has_ball:
            cmd_attacker = score_goal(
                game,
                True,
                0,
                pid_oren_a,
                pid_2d_a,
                (not defender_is_yellow),  # Attacker's team color
                (not defender_is_yellow)   # Target left or right goal
            )
        else:
            # If the attacker doesn't have the ball, move toward it
            cmd_attacker = go_to_point(
                pid_oren_a,
                pid_2d_a,
                attacker,
                0,
                (ball.x, ball.y),
                face_ball((attacker.x, attacker.y), (ball.x, ball.y)),
            )

        sim_robot_controller_attacker.add_robot_commands(cmd_attacker, 0)
        sim_robot_controller_attacker.send_robot_commands()

        # ========== Defender logic: protect the ball and block the attacker ========== 
        defender = friendly[0]

        cmd_defender = improved_block_goal_and_attacker(
            defender,
            attacker,
            ball,
            goal_pos,
            pid_oren_d,
            pid_2d_d,
            has_ball,
            1.0,
            (0, 0),
            distance_factor=distance_factor  # Pass the new distance factor parameter here
        )
        sim_robot_controller_defender.add_robot_commands(cmd_defender, 0)
        sim_robot_controller_defender.send_robot_commands()

        # ========== Check if a goal was scored ========== 
        if game.is_ball_in_goal(our_side=(defender_is_yellow)):
            a = f"[Iteration {i}] The attacker scored successfully! Goal position: {game.get_ball_pos()}"
            logger.info(a)
            goal_scored = True

        # ========== Visualization: draw lines and points in the environment ========== 
        # Draw a line between the defender and the attacker
        env.draw_line(
            [(defender.x, defender.y), (attacker.x, attacker.y)],
            width=2,
            color="BLUE",
        )
        # Draw a dashed line from the attacker toward the goal
        env.draw_line(
            [(attacker.x, attacker.y), goal_pos],
            width=2,
            color="RED",
        )

    if not goal_scored:
        logger.info("The defender successfully prevented a goal throughout all iterations!")


if __name__ == "__main__":
    try:
        # You can adjust the 'distance_factor' below to see the defender position more/less aggressively:
        # e.g. test_ultimate_one_on_one(defender_is_yellow=True, headless=False, distance_factor=1.2)
        test_ultimate_one_on_one(defender_is_yellow=True, headless=False, distance_factor=1.0)
    except KeyboardInterrupt:
        print("Exiting...")
