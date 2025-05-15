from team_controller.src.controllers import GRSimRobotController
from strategy.startup_strategy import StartupStrategy
from motion_planning.src.pid.pid import get_grsim_pids
from run import run

if __name__ == "__main__":
    sim_robot_controller = GRSimRobotController(is_team_yellow=True)
    # bt = DummyBehaviour()
    # main(BehaviourTreeStrategy(sim_robot_controller, bt), sim_robot_controller)
    run(StartupStrategy(sim_robot_controller, get_grsim_pids))
