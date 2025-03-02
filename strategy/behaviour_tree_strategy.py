import py_trees
from entities.game.present_future_game import PresentFutureGame
from strategy.strategy import Strategy
from team_controller.src.controllers.common.robot_controller_abstract import AbstractRobotController


class BehaviourTreeStrategy(Strategy):
    def __init__(self, robot_controller: AbstractRobotController, behaviour_tree: py_trees.behaviour.Behaviour):
        self.blackboard = py_trees.blackboard.Client(name="GlobalConfig")
        self.blackboard.register_key(key="robot_controller", access=py_trees.common.Access.EXCLUSIVE_WRITE)
        self.blackboard.register_key(key="past_future_game", access=py_trees.common.Access.EXCLUSIVE_WRITE)
        self.blackboard.robot_controller = robot_controller
        self.behaviour_tree = behaviour_tree

    def step(self, present_future_game: PresentFutureGame):
        self.blackboard.present_future_game = present_future_game
        self.behaviour_tree.tick_once()
