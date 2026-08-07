import py_trees


class DummyStrategy:
    exp_ball: bool = True  # Not relevant for these tests

    def __init__(self):
        # StrategyRunner.__init__ unconditionally does
        # `self.my.strategy.behaviour_tree.add_visitor(...)` for BT
        # visualization — a real (if trivial) BehaviourTree is needed here,
        # not just duck-typing the rest of AbstractStrategy's interface.
        self.behaviour_tree = py_trees.trees.BehaviourTree(
            py_trees.composites.Selector(name="DummyStrategyUnusedRoot", memory=False)
        )

    def assert_exp_robots(self, exp_friendly, exp_enemy):
        return True

    def assert_exp_goals(self, my_goal, opp_goal):
        return True

    def get_min_bounding_req(self):
        return None

    def setup_strategy_blackboard(self, is_opp_strat):
        pass

    def setup_behaviour_tree(self, is_opp_strat):
        pass

    def load_rsim_env(self, env):
        pass

    def load_robot_controller(self, controller):
        pass

    def load_motion_controller(self, controller):
        pass

    def load_game(self, game):
        pass

    def step(self):
        pass
