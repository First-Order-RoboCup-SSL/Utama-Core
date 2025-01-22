import logging
import random
import math

from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game

# Imports from other scripts or modules within the same project
from robot_control.src.tests.utils import setup_pvp
from motion_planning.src.pid.pid import get_rsim_pids
from robot_control.src.skills import face_ball, go_to_point
from robot_control.src.intent import score_goal

logger = logging.getLogger(__name__)

ITERS = 2000


def improved_block_goal_and_attacker(
    robot, attacker, ball, goal_pos,
    pid_oren, pid_trans,
    attacker_has_ball: bool,
    block_ratio: float = 0.3,
    max_ball_follow_dist: float = 1.0
):
    """
    Intelligent defense strategy:
    1) If the attacker has the ball, block on the attacker-goal line.
    2) Otherwise, stay closer to the ball while still considering the attacker's possible shot.

    :param robot: Defender's Robot object (robot.x, robot.y, robot.orientation)
    :param attacker: Attacker's Robot object
    :param ball: Ball object
    :param goal_pos: The (x, y) location of the goal to defend
    :param pid_oren: PID controller for orientation
    :param pid_trans: PID controller for translation
    :param attacker_has_ball: Whether the attacker currently has ball possession
    :param block_ratio: 0~1 ratio to position ourselves between attacker & goal
    :param max_ball_follow_dist: if attacker doesn't have ball, how close we stay near the ball
    :return: The command dict for the defender robot
    """
    if attacker_has_ball:
        # ========== Prioritize blocking the shot line ==========
        ax, ay = attacker.x, attacker.y
        gx, gy = goal_pos[0], goal_pos[1]
        agx, agy = (gx - ax), (gy - ay)
        dist_ag = math.hypot(agx, agy)

        if dist_ag < 1e-6:
            # Extreme edge case if attacker and goal are basically the same
            target_x, target_y = gx, gy
        else:
            target_x = ax + block_ratio * agx
            target_y = ay + block_ratio * agy

        # Face the attacker
        face_theta = math.atan2((ay - robot.y), (ax - robot.x))
    else:
        # ========== Attacker doesn't have ball; defend ball more closely ==========
        ax, ay = attacker.x, attacker.y
        bx, by = ball.x, ball.y

        # Move ~70% of the way toward the ball from the attacker
        abx, aby = (bx - ax), (by - ay)
        target_x = ax + 0.7 * abx
        target_y = ay + 0.7 * aby

        # If we are too far from the ball, move closer
        dist_def_to_ball = math.hypot(robot.x - bx, robot.y - by)
        if dist_def_to_ball > max_ball_follow_dist:
            ratio = max_ball_follow_dist / dist_def_to_ball
            target_x = robot.x + (bx - robot.x) * ratio
            target_y = robot.y + (by - robot.y) * ratio

        # Face the ball
        face_theta = math.atan2((by - robot.y), (bx - robot.x))

    # === IMPORTANT CHANGE HERE ===
    # Instead of passing the Robot object directly to go_to_point(),
    # we pass (x, y, orientation) as a tuple.
    current_pose = (robot.x, robot.y, robot.orientation if hasattr(robot, "orientation") else 0.0)

    cmd = go_to_point(
        pid_oren,
        pid_trans,
        current_pose,  # (x, y, orientation) tuple
        0,
        (target_x, target_y),
        face_theta
    )
    return cmd


def test_ultimate_one_on_one(defender_is_yellow: bool, headless: bool):
    """
    A 1v1 scenario with dynamic switching of attacker/defender roles:
    - If a team picks up the ball, it becomes the attacker.
    - The other team defends.
    """
    # 1) Initialize Game and environment
    game = Game()
    env = SSLStandardEnv(
        n_robots_blue=1 if defender_is_yellow else 1,
        n_robots_yellow=1 if not defender_is_yellow else 1,
        render_mode="ansi" if headless else "human",
    )
    env.reset()

    # Random ball placement
    ball_x = random.uniform(-2.5, 2.5)
    ball_y = random.uniform(-1.5, 1.5)
    env.teleport_ball(ball_x, ball_y)

    # 2) PID controllers
    pid_oren_y, pid_2d_y = get_rsim_pids(1)  # Yellow
    pid_oren_b, pid_2d_b = get_rsim_pids(1)  # Blue

    # 3) Setup PVP robot controllers
    sim_robot_controller_yellow, sim_robot_controller_blue, pvp_manager = setup_pvp(
        env, game, n_robots_blue=1, n_robots_yellow=1
    )

    # 4) Define goal positions
    yellow_goal_pos = (4.5, 0)
    blue_goal_pos = (-4.5, 0)

    goal_scored = False

    for i in range(ITERS):
        if goal_scored:
            break

        # *** DeprecationWarning: we keep using get_my_latest_frame for now ***
        friendly, enemy, balls = game.get_my_latest_frame(my_team_is_yellow=defender_is_yellow)
        ball = balls[0]

        # Robot references
        defender_robot = friendly[0]
        attacker_robot = enemy[0]

        # Also direct access by color
        yellow_robots = game.get_yellow_robots()
        blue_robots = game.get_blue_robots()
        yellow_robot = yellow_robots[0]
        blue_robot = blue_robots[0]

        # Who has the ball?
        yellow_has_ball = sim_robot_controller_yellow.robot_has_ball(0)
        blue_has_ball   = sim_robot_controller_blue.robot_has_ball(0)

        # A) If Yellow has ball => Yellow attacks, Blue defends
        if yellow_has_ball and not blue_has_ball:
            # Y: Attack
            cmd_attacker = score_goal(
                game,
                True,
                0,
                pid_oren_y,
                pid_2d_y,
                True,   # is_yellow
                False   # target_left_goal = false => shoot to left (blue side)
            )
            sim_robot_controller_yellow.add_robot_commands(cmd_attacker, 0)
            sim_robot_controller_yellow.send_robot_commands()

            # B: Defend
            cmd_defender = improved_block_goal_and_attacker(
                blue_robot,  # defender
                yellow_robot,# attacker
                ball,
                blue_goal_pos,
                pid_oren_b,
                pid_2d_b,
                attacker_has_ball=True,
                block_ratio=0.4,
                max_ball_follow_dist=1.0
            )
            sim_robot_controller_blue.add_robot_commands(cmd_defender, 0)
            sim_robot_controller_blue.send_robot_commands()

        # B) If Blue has ball => Blue attacks, Yellow defends
        elif blue_has_ball and not yellow_has_ball:
            # B: Attack
            cmd_attacker = score_goal(
                game,
                True,
                0,
                pid_oren_b,
                pid_2d_b,
                False,  # is_blue
                False   # target_left_goal = false => shoot to right (yellow side)
            )
            sim_robot_controller_blue.add_robot_commands(cmd_attacker, 0)
            sim_robot_controller_blue.send_robot_commands()

            # Y: Defend
            cmd_defender = improved_block_goal_and_attacker(
                yellow_robot,
                blue_robot,
                ball,
                yellow_goal_pos,
                pid_oren_y,
                pid_2d_y,
                attacker_has_ball=True,
                block_ratio=0.4,
                max_ball_follow_dist=1.0
            )
            sim_robot_controller_yellow.add_robot_commands(cmd_defender, 0)
            sim_robot_controller_yellow.send_robot_commands()

        # C) Neither side has the ball => both try to get it
        else:
            # Convert Robot -> (x, y, orientation) before calling go_to_point
            current_pose_yellow = (
                yellow_robot.x,
                yellow_robot.y,
                getattr(yellow_robot, "orientation", 0.0)
            )
            cmd_yellow = go_to_point(
                pid_oren_y,
                pid_2d_y,
                current_pose_yellow,
                0,
                (ball.x, ball.y),
                face_ball((yellow_robot.x, yellow_robot.y), (ball.x, ball.y))
            )
            sim_robot_controller_yellow.add_robot_commands(cmd_yellow, 0)
            sim_robot_controller_yellow.send_robot_commands()

            current_pose_blue = (
                blue_robot.x,
                blue_robot.y,
                getattr(blue_robot, "orientation", 0.0)
            )
            cmd_blue = go_to_point(
                pid_oren_b,
                pid_2d_b,
                current_pose_blue,
                0,
                (ball.x, ball.y),
                face_ball((blue_robot.x, blue_robot.y), (ball.x, ball.y))
            )
            sim_robot_controller_blue.add_robot_commands(cmd_blue, 0)
            sim_robot_controller_blue.send_robot_commands()

        # Check if goal was scored
        if game.is_ball_in_goal(our_side=(defender_is_yellow)):
            a = f"[Iteration {i}] The attacker scored successfully! Ball pos: {game.get_ball_pos()}"
            logger.info(a)
            goal_scored = True
        
        if game.is_ball_in_goal(our_side=(not defender_is_yellow)):
            a = f"[Iteration {i}] The attacker scored successfully! Ball pos: {game.get_ball_pos()}"
            logger.info(a)
            goal_scored = True

        # (Optional) Drawing lines
        env.draw_line(
            [(yellow_robot.x, yellow_robot.y), (blue_robot.x, blue_robot.y)],
            width=2,
            color="BLUE",
        )
        env.draw_line(
            [(ball.x, ball.y), (yellow_goal_pos if defender_is_yellow else blue_goal_pos)],
            width=2,
            color="RED",
        )

    if not goal_scored:
        logger.info("No goal was scored! Defender(s) successfully prevented scoring for all iterations.")


if __name__ == "__main__":
    try:
        # Toggle defender_is_yellow=True or False as needed
        test_ultimate_one_on_one(defender_is_yellow=True, headless=False)
    except KeyboardInterrupt:
        print("Exiting...")
