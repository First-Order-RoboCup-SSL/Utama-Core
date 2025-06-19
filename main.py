from strategy.skills.go_to_ball import GoToBallStrategy
from run import StrategyRunner
from strategy.behaviour_trees.behaviour_tree_strategy import BehaviourTreeStrategy
from strategy.behaviour_trees.behaviours.skills_behaviours import create_go_to_ball_behaviour

if __name__ == "__main__":
    # The robot we want to control
    target_robot_id = 1

    # Create the go-to-ball behaviour for the target robot
    go_to_ball_behaviour = create_go_to_ball_behaviour(robot_id=target_robot_id)

    # Set up the runner
    runner = StrategyRunner(
        strategy=BehaviourTreeStrategy(behaviour=go_to_ball_behaviour),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="grsim",
        exp_friendly=3,
        exp_enemy=3,
    )
    
    # Run the simulation
    test = runner.run()
