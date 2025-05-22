from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from strategy.one_robot_placement_strategy import RobotPlacementStrategy
from run import StrategyRunner

if __name__ == "__main__":
    runner = StrategyRunner(
        strategy=RobotPlacementStrategy(id=4),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="grsim",
        exp_friendly=6,
        exp_enemy=6,
        exp_ball=True,
    )
    runner.run()
