import sys
import os

print(sys.path)
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
print(project_root)
sys.path.insert(0, project_root)

from motion_planning.src.pid.pid import TwoDPID
from robot_control.src.skills import go_to_ball, go_to_point
from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from robot_control.src.intent import score_goal
from motion_planning.src.pid import PID
from team_controller.src.controllers.sim.rsim_robot_controller import PVPManager
from team_controller.src.config.settings import TIMESTEP


if __name__ == "__main__":
    IS_YELLOW = False

    game = Game()

    env = SSLStandardEnv(n_robots_blue=3)
    env.reset()
    defender_ids = [1]
    shooter_id = 3
    env.teleport_ball(1, 1)
    pid_oren = PID(TIMESTEP, 8, -8, 3, 3, 0.1, num_robots=6)
    pid_2d = TwoDPID(TIMESTEP, 1.5, -1.5, 3, 0.1, 0.0, num_robots=6)


    sim_robot_controller_yellow = RSimRobotController(
        is_team_yellow=IS_YELLOW, env=env, game_obj=game, debug=True
    )
    try:
        done = False
        LEADER = 0
        ty = 1.5
        tx = 0

        defender_y_targets = {defender_id:ty for defender_id in defender_ids}
        last_sent = 0
        while True:
            import numpy as np
            import math
            env.draw_line([(0,0), (2,2)], color="RED", width=10)

            latest_frame = game.get_my_latest_frame(IS_YELLOW)
            if latest_frame:
                friendly_robots, enemy_robots, balls = latest_frame
            
                cx, cy, co = friendly_robots[defender_ids[0]]
                target_oren = math.pi+np.arctan2(
                    ty - cy, tx - cx
                )
                if math.dist((tx, ty), (cx, cy))  < 0.002:
                    ty *= -1
                from entities.data.command import RobotCommand
                import math
                oren =  math.pi / 2 if ty > 0 else - math.pi / 2

                cmd = go_to_point(pid_oren, pid_2d, friendly_robots[defender_ids[0]], defender_ids[0], (tx, ty), oren)
                sim_robot_controller_yellow.add_robot_commands(cmd, defender_ids[0])
                sim_robot_controller_yellow.send_robot_commands()

    except KeyboardInterrupt:
        print("Exiting...")
