from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from robot_control.src.intent import score_goal
from motion_planning.src.pid import PID


if __name__ == "__main__":
    game = Game()

    # making environment
    env = SSLStandardEnv(n_robots_blue=3)
    env.reset()
    shooter_id = 3
    # env.teleport_robot(False, 0, x=1, y=1)
    env.teleport_ball(1, 1)

    pid_oren = PID(0.0167, 8, -8, 4.5, 0, 0.045, num_robots=6)
    pid_trans = PID(0.0167, 1.5, -1.5, 4.5, 0, 0.035, num_robots=6)

    sim_robot_controller = RSimRobotController(
        is_team_yellow=True, env=env, game_obj=game, debug=False
    )

    try:
        while True:
            # TODO: We shoud move robot_has_ball within game obj as well
            # This will do for now.
            cmd = score_goal(
                game,
                sim_robot_controller.robot_has_ball(shooter_id),
                shooter_id=shooter_id,
                pid_oren=pid_oren,
                pid_trans=pid_trans,
            )
            sim_robot_controller.add_robot_commands(cmd, shooter_id)
            sim_robot_controller.send_robot_commands()
    except KeyboardInterrupt:
        print("Exiting...")
