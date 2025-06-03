import math
import time

from strategy.skills.solo_defender import SoloDefenderStrategy
from test.common.abstract_test_manager import AbstractTestManager, TestingStatus
from team_controller.src.controllers import AbstractSimController
from config.starting_formation import LEFT_START_ONE, RIGHT_START_ONE
from entities.game.game import Game
from global_utils.mapping_utils import map_left_right_to_colors

from run import StrategyRunner

# Constants for test evaluation
POSITION_TOLERANCE_DEF = (
    0.35  # Maximum allowed distance from the defender to its target position.
)
EVAL_DURATION_SEC = 5  # Duration after which the defender's position is evaluated.
DEFENDER_TEST_TIMEOUT = EVAL_DURATION_SEC + 3  # Total timeout for each test episode.

TEST_ENEMY_ATTACKER_ID = 0  # Designated ID for the enemy robot acting as an attacker.


class SoloDefenderTestManager(AbstractTestManager):
    """
    Test manager for the SoloDefenderStrategy.
    It sets up specific scenarios to verify the defender's behavior under different conditions.
    The core idea is to check if the defender positions itself correctly based on the
    attacker's and ball's positions, and whether the attacker is perceived to have the ball.
    """

    def __init__(self, defender_id: int, strategy_params: dict):
        super().__init__()
        self.defender_id = defender_id
        self.strategy_params = (
            strategy_params  # Contains 'block_ratio' and 'max_ball_follow_dist'
        )

        # Scenarios are defined assuming OUR GOAL is on the NEGATIVE X side of the field.
        # Positions will be mirrored if our goal is on the positive X side during the test.
        # Structure: (def_init_x, def_init_y, enemy_att_init_x, enemy_att_init_y,
        #             ball_init_x, ball_init_y, enemy_simulated_has_ball_flag, description_string)
        self.scenarios = [
            (
                # Scenario 1: Attacker is set up with the ball, threatening a direct shot.
                # Defender initial position: (-1.5, 0.5)
                # Enemy attacker initial position: (2.5, 0.0)
                # Ball initial position: (2.4, 0.0) - Placed very close to the attacker (dist 0.1m)
                #                              to ensure the strategy's 'enemy_has_ball_guess' is True.
                # Enemy simulated_has_ball flag: True - This flag in the test dictates the evaluation logic.
                # Expected Behavior: The SoloDefenderStrategy should identify the attacker has the ball
                #                    and position the defender robot to block the direct shot line
                #                    between the attacker and our goal. The exact blocking position
                #                    is determined by the 'block_ratio' parameter.
                # Evaluation: Check if the defender is at the calculated blocking point on the
                #             attacker-goal line, within POSITION_TOLERANCE_DEF.
                -1.5,
                0.5,
                2.5,
                0.0,
                2.4,
                0.0,  # Ball close to attacker
                True,
                "Block direct shot: Attacker has ball (ball at attacker)",
            ),
            (
                # Scenario 2: Ball is loose, and the attacker is relatively distant from it.
                # Defender initial position: (-1.0, -1.0)
                # Enemy attacker initial position: (1.5, 1.5) - Further from the ball.
                # Ball initial position: (0.0, 1.0) - Loose in midfield.
                # Enemy simulated_has_ball flag: False - This flag in the test dictates the evaluation logic.
                # Expected Behavior: The SoloDefenderStrategy should identify the ball is loose.
                #                    If the defender is further from the ball than 'max_ball_follow_dist',
                #                    it should move towards the ball to cover it. The strategy aims
                #                    to position the defender such that its distance to the ball is
                #                    approximately 'max_ball_follow_dist' or closer.
                # Evaluation: Check if the defender's final distance to the ball is less than
                #             'max_ball_follow_dist' + POSITION_TOLERANCE_DEF.
                -1.0,
                -1.0,
                1.5,
                1.5,
                0.0,
                1.0,
                False,
                "Cover loose ball: Attacker distant, defender moves towards ball",
            ),
        ]
        self.n_episodes = len(self.scenarios)
        self.episode_start_time = 0
        self.our_goal_display_x = 0
        self.current_scenario_data = {}

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        self.episode_start_time = time.time()
        # Determine the X-coordinate of our goal line for calculations and display
        self.our_goal_display_x = game.field.my_goal_line.coords[0][0]

        (
            def_x_orig,
            def_y_orig,
            enemy_att_x_orig,
            enemy_att_y_orig,
            ball_x_orig,
            ball_y_orig,
            enemy_has_ball_setup,
            desc,
        ) = self.scenarios[self.episode_i]

        self.current_scenario_data = {
            "enemy_att_x_setup": enemy_att_x_orig,
            "enemy_att_y_setup": enemy_att_y_orig,
            "ball_x_setup": ball_x_orig,
            "ball_y_setup": ball_y_orig,
            "enemy_has_ball_setup": enemy_has_ball_setup,
            "description": desc,
        }

        def_x, def_y = def_x_orig, def_y_orig
        enemy_att_x, enemy_att_y = enemy_att_x_orig, enemy_att_y_orig
        ball_x, ball_y = ball_x_orig, ball_y_orig

        # Mirror scenario X-coordinates if our goal is on the positive X side.
        # This ensures scenarios behave consistently regardless of initial team assignment.
        if game.field.my_goal_line.coords[0][0] > 0:
            def_x *= -1
            enemy_att_x *= -1
            ball_x *= -1

        self.current_scenario_data.update(
            {
                "actual_enemy_att_x": enemy_att_x,
                "actual_enemy_att_y": enemy_att_y,
                "actual_ball_x": ball_x,
                "actual_ball_y": ball_y,
            }
        )

        # Teleport other robots (not involved in the specific scenario) to default positions.
        ini_yellow, ini_blue = map_left_right_to_colors(
            game.my_team_is_yellow,
            game.my_team_is_right,
            RIGHT_START_ONE,
            LEFT_START_ONE,
        )

        for r_id in game.friendly_robots:
            if r_id != self.defender_id:
                pos_array = (
                    ini_yellow[r_id] if game.my_team_is_yellow else ini_blue[r_id]
                )
                sim_controller.teleport_robot(
                    game.my_team_is_yellow,
                    r_id,
                    pos_array[0],
                    pos_array[1],
                    pos_array[2],
                )

        for r_id in game.enemy_robots:
            if r_id != TEST_ENEMY_ATTACKER_ID:
                pos_array = (
                    ini_blue[r_id] if game.my_team_is_yellow else ini_yellow[r_id]
                )
                sim_controller.teleport_robot(
                    not game.my_team_is_yellow,
                    r_id,
                    pos_array[0],
                    pos_array[1],
                    pos_array[2],
                )

        # Teleport the designated defender to its initial scenario position, facing the ball.
        sim_controller.teleport_robot(
            is_team_yellow=game.my_team_is_yellow,
            robot_id=self.defender_id,
            x=def_x,
            y=def_y,
            theta=math.atan2(ball_y - def_y, ball_x - def_x),
        )

        # Teleport the designated enemy attacker to its initial scenario position.
        # It faces our goal if it's set up to "have the ball", otherwise it faces the ball.
        sim_controller.teleport_robot(
            is_team_yellow=not game.my_team_is_yellow,
            robot_id=TEST_ENEMY_ATTACKER_ID,
            x=enemy_att_x,
            y=enemy_att_y,
            theta=(
                math.atan2(0 - enemy_att_y, self.our_goal_display_x - enemy_att_x)
                if enemy_has_ball_setup
                else math.atan2(ball_y - enemy_att_y, ball_x - enemy_att_x)
            ),
        )
        # Teleport the ball to its scenario-defined initial position.
        sim_controller.teleport_ball(ball_x, ball_y)

        print(f"Episode {self.episode_i + 1}/{self.n_episodes}: {desc}")
        print(f"  Defender ({self.defender_id}) initial: ({def_x:.2f}, {def_y:.2f})")
        print(
            f"  Enemy Attacker ({TEST_ENEMY_ATTACKER_ID}) initial: ({enemy_att_x:.2f}, {enemy_att_y:.2f})"
        )
        print(
            f"  Ball initial: ({ball_x:.2f}, {ball_y:.2f}), Enemy_setup_has_ball: {enemy_has_ball_setup}"
        )
        print(f"  Defending goal at x={self.our_goal_display_x:.2f}")

    def eval_status(self, game: Game) -> TestingStatus:
        if game.ball is None:
            print("Failure: Ball object is None.")
            return TestingStatus.FAILURE

        # Immediate failure if an own goal occurs.
        if game.is_ball_in_goal(right_goal=(game.field.my_goal_line.coords[0][0] > 0)):
            print(
                f"Failure: Ball entered our goal. Ball at ({game.ball.p.x:.2f}, {game.ball.p.y:.2f})."
            )
            return TestingStatus.FAILURE

        # Allow EVAL_DURATION_SEC for the strategy to position the robot.
        if time.time() - self.episode_start_time < EVAL_DURATION_SEC:
            return TestingStatus.IN_PROGRESS

        defender = game.friendly_robots.get(self.defender_id)
        if not defender:
            print("Failure: Defender robot not found in game state.")
            return TestingStatus.FAILURE

        enemy_attacker_for_eval = game.enemy_robots.get(TEST_ENEMY_ATTACKER_ID)
        if not enemy_attacker_for_eval:
            print(
                f"Warning: Test enemy attacker {TEST_ENEMY_ATTACKER_ID} not found. Using setup positions."
            )
            ax, ay = (
                self.current_scenario_data["actual_enemy_att_x"],
                self.current_scenario_data["actual_enemy_att_y"],
            )
        else:
            ax, ay = enemy_attacker_for_eval.p.x, enemy_attacker_for_eval.p.y

        bx, by = game.ball.p.x, game.ball.p.y

        scenario_instructs_enemy_has_ball = self.current_scenario_data[
            "enemy_has_ball_setup"
        ]

        if scenario_instructs_enemy_has_ball:
            # Expected Outcome: Defender blocks the shot.
            # The defender should be positioned on the line segment connecting the attacker (ax, ay)
            # and our goal's center (gx, gy). The 'block_ratio' determines how far along this
            # line (from the attacker) the defender should be.
            # gx is our goal line's x-coordinate, gy is assumed to be 0 (center of goal height).
            gx, gy = game.field.my_goal_line.coords[0][0], 0
            block_ratio = self.strategy_params.get("block_ratio", 0.1)

            agx, agy = (gx - ax), (gy - ay)
            dist_ag = math.hypot(agx, agy)
            if dist_ag < 1e-6:  # Attacker is effectively at our goal center.
                expected_x, expected_y = gx, gy
            else:
                expected_x = ax + block_ratio * agx
                expected_y = ay + block_ratio * agy

            dist_to_expected = math.hypot(
                defender.p.x - expected_x, defender.p.y - expected_y
            )
            if dist_to_expected < POSITION_TOLERANCE_DEF:
                print(
                    f"Success (Scenario: Attacker has ball): Defender at ({defender.p.x:.2f}, {defender.p.y:.2f}), "
                    f"expected near ({expected_x:.2f}, {expected_y:.2f}). Dist: {dist_to_expected:.2f}"
                )
                return TestingStatus.SUCCESS
            else:
                print(
                    f"Failure (Scenario: Attacker has ball): Defender at ({defender.p.x:.2f}, {defender.p.y:.2f}), "
                    f"expected near ({expected_x:.2f}, {expected_y:.2f}). Dist: {dist_to_expected:.2f}"
                )
                print(
                    f"  Attacker: ({ax:.2f}, {ay:.2f}), Goal: ({gx:.2f}, {gy:.2f}), BlockRatio: {block_ratio:.2f}"
                )
                return TestingStatus.FAILURE
        else:
            # Expected Outcome: Defender covers the loose ball.
            # The defender should move towards the ball if it's further than 'max_ball_follow_dist'.
            # The test checks if the final distance from the defender to the ball is within
            # 'max_ball_follow_dist' (plus a tolerance).
            dist_def_to_ball = math.hypot(defender.p.x - bx, defender.p.y - by)
            max_follow = self.strategy_params.get("max_ball_follow_dist", 1.0)

            if dist_def_to_ball < max_follow + POSITION_TOLERANCE_DEF:
                print(
                    f"Success (Scenario: Attacker no ball): Defender at ({defender.p.x:.2f}, {defender.p.y:.2f}) "
                    f"is {dist_def_to_ball:.2f}m from ball ({bx:.2f}, {by:.2f}) (target_max_dist_plus_tol: {max_follow + POSITION_TOLERANCE_DEF:.2f}m)."
                )
                return TestingStatus.SUCCESS
            else:
                print(
                    f"Failure (Scenario: Attacker no ball): Defender at ({defender.p.x:.2f}, {defender.p.y:.2f}) "
                    f"is {dist_def_to_ball:.2f}m from ball ({bx:.2f}, {by:.2f}) (target_max_dist_plus_tol: {max_follow + POSITION_TOLERANCE_DEF:.2f}m)."
                )
                return TestingStatus.FAILURE

        print(
            "Failure: Defender position conditions not met after evaluation duration (reached end of eval_status)."
        )
        return TestingStatus.FAILURE

    def get_n_episodes(self) -> int:
        return self.n_episodes


def test_solo_defender(
    my_team_is_yellow: bool,
    my_team_is_right: bool,
    target_id: int,
    headless: bool,
    mode: str = "rsim",
):
    """
    Main test execution function for the SoloDefenderStrategy.
    This function is typically called by a test runner like pytest.
    It initializes the StrategyRunner with the SoloDefenderStrategy and the SoloDefenderTestManager.
    """
    strategy_params = {
        "block_ratio": 0.4,  # Ratio for blocking position on attacker-goal line.
        "max_ball_follow_dist": 1.0,  # Max distance defender maintains from ball when loose.
    }

    runner = StrategyRunner(
        strategy=SoloDefenderStrategy(
            target_id=target_id,
            block_ratio=strategy_params["block_ratio"],
            max_ball_follow_dist=strategy_params["max_ball_follow_dist"],
        ),
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        mode=mode,
        # exp_friendly: Number of friendly robots expected. SoloDefenderStrategy needs at least 1.
        # exp_enemy: Number of enemy robots expected. Test needs at least 1 for the attacker role.
        # The specific values (e.g., 3) can accommodate default robot setups in the simulator,
        # as long as the target_id and TEST_ENEMY_ATTACKER_ID are present.
        exp_friendly=3,
        exp_enemy=3,
    )
    test_manager = SoloDefenderTestManager(
        defender_id=target_id, strategy_params=strategy_params
    )
    test_result = runner.run_test(
        testManager=test_manager,
        episode_timeout=DEFENDER_TEST_TIMEOUT,
        rsim_headless=headless,
    )
    assert test_result, f"SoloDefenderStrategy test failed for target_id {target_id}."


if __name__ == "__main__":
    print("Running SoloDefenderStrategy Test Manually...")
    test_solo_defender(
        my_team_is_yellow=True,
        my_team_is_right=False,  # Our team defends the goal on the negative X side.
        target_id=0,  # ID of the friendly robot acting as the defender.
        mode="rsim",
        headless=False,
    )
