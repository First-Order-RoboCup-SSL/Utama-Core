class DummyStrategy:
    exp_ball: bool = True  # Not relevant for these tests

    def assert_exp_robots(self, exp_friendly, exp_enemy):
        return True

    def assert_exp_goals(self, my_goal, opp_goal):
        return True

    def get_min_bounding_zone(self):
        return None

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
