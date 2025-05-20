from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from strategy.startup_strategy import StartupStrategy
from run import run_strategy

if __name__ == "__main__":
    env = SSLStandardEnv()
    run_strategy(StartupStrategy(), True, True, "rsim", 6, 6, True, env)
