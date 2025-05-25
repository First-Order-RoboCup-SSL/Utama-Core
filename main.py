from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from strategy.examples.go_to_ball import GoToBallStrategy
from strategy.examples.test.go_to_ball_test import GoToBallTestManager
from run import StrategyRunner

if __name__ == "__main__":
    runner = StrategyRunner(
        strategy=GoToBallStrategy(target_id=0),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="grsim",
        exp_friendly=1,
        exp_enemy=1,
        exp_ball=True,
    )
    test = runner.run_test(GoToBallTestManager(), 1000)
    print(test)
