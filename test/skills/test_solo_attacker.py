import math
import time

from strategy.skills.solo_attacker import SoloAttackerStrategy
from test.common.abstract_test_manager import AbstractTestManager, TestingStatus
from team_controller.src.controllers import AbstractSimController
from config.starting_formation import LEFT_START_ONE, RIGHT_START_ONE
from entities.game.game import Game
from global_utils.mapping_utils import map_left_right_to_colors  # Corrected import

from run import StrategyRunner

# Constants for test evaluation
GOAL_SCORE_TIMEOUT = 15  # Maximum time allowed for the attacker to score a goal.


class SoloAttackerTestManager(AbstractTestManager):
    """
    Test manager for the SoloAttackerStrategy.
    It sets up scenarios to verify the attacker's ability to take the ball and score a goal.
    The primary success condition for all scenarios is the ball entering the enemy's goal.
    """

    def __init__(self, target_id: int):
        super().__init__()
        self.target_id = target_id
        # Scenarios are defined assuming the team is attacking the GOAL ON THE POSITIVE X side.
        # Positions will be mirrored if the enemy goal is on the negative X side during the test.
        # Field dimensions (typical): length=9m, width=6m. Goals at x = +/-4.5m.
        # Structure: (robot_init_x, robot_init_y, ball_init_x, ball_init_y, description_string)
        self.scenarios = [
            (
                # Scenario 1: Robot starts away from the ball.
                # Robot initial position: (-2.0, 1.0)
                # Ball initial position: (0.0, 0.0) - At center field.
                # Expected Behavior: The SoloAttackerStrategy should command the robot to navigate
                #                    to the ball, gain possession (implicitly or explicitly), and
                #                    then move towards the enemy goal to score.
                # Evaluation: Success if the ball enters the enemy goal within GOAL_SCORE_TIMEOUT.
                -2.0,
                1.0,
                0.0,
                0.0,
                "Go to ball and score",
            ),
            (
                # Scenario 2: Robot starts with the ball, relatively far from the enemy goal.
                # Robot initial position: (1.0, 0.5)
                # Ball initial position: (1.0, 0.5) - At the robot's location.
                # Expected Behavior: The robot should dribble the ball towards the enemy goal
                #                    and attempt to score.
                # Evaluation: Success if the ball enters the enemy goal within GOAL_SCORE_TIMEOUT.
                -1.0,
                0.5,
                0.0,
                0.5,
                "Dribble and score",
            ),
            (
                # Scenario 3: Robot starts with the ball, close to the enemy goal.
                # Robot initial position: (3.5, 0.5) - Close to the positive X goal line (at 4.5).
                # Ball initial position: (3.5, 0.5) - At the robot's location.
                # Expected Behavior: The robot should execute a shot or a short dribble followed
                #                    by a shot to score into the enemy goal.
                # Evaluation: Success if the ball enters the enemy goal within GOAL_SCORE_TIMEOUT.
                3.5,
                0.5,
                3.5,
                0.5,
                "Shoot and score",
            ),
        ]
        self.n_episodes = len(self.scenarios)
        self.episode_start_time = 0
        self.enemy_goal_display_x = 0

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        self.episode_start_time = time.time()
        # Determine the X-coordinate of the enemy goal line for display.
        self.enemy_goal_display_x = game.field.enemy_goal_line.coords[0][0]

        # Teleport other robots (not involved in the specific scenario) to default positions.
        ini_yellow, ini_blue = map_left_right_to_colors(
            game.my_team_is_yellow,
            game.my_team_is_right,
            RIGHT_START_ONE,
            LEFT_START_ONE,
        )

        for r_id in game.friendly_robots:
            if r_id != self.target_id:
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
        for (
            r_id
        ) in (
            game.enemy_robots
        ):  # Usually exp_enemy=0 for open goal, but this handles them if present.
            pos_array = ini_blue[r_id] if game.my_team_is_yellow else ini_yellow[r_id]
            sim_controller.teleport_robot(
                not game.my_team_is_yellow,
                r_id,
                pos_array[0],
                pos_array[1],
                pos_array[2],
            )

        robot_x_orig, robot_y_orig, ball_x_orig, ball_y_orig, description = (
            self.scenarios[self.episode_i]
        )

        robot_x, robot_y = robot_x_orig, robot_y_orig
        ball_x, ball_y = ball_x_orig, ball_y_orig

        # Mirror scenario X-coordinates if the enemy goal is on the negative X side.
        # This ensures scenarios behave consistently regardless of initial team assignment.
        if game.field.enemy_goal_line.coords[0][0] < 0:
            robot_x *= -1
            ball_x *= -1

        # Teleport the designated attacker to its initial scenario position, facing the ball.
        sim_controller.teleport_robot(
            is_team_yellow=game.my_team_is_yellow,
            robot_id=self.target_id,
            x=robot_x,
            y=robot_y,
            theta=math.atan2(ball_y - robot_y, ball_x - robot_x),
        )
        # Teleport the ball to its scenario-defined initial position.
        sim_controller.teleport_ball(ball_x, ball_y)  # Corrected from (0,0)

        print(f"Episode {self.episode_i + 1}/{self.n_episodes}: {description}")
        print(
            f"  Attacker ({self.target_id}) initial: ({robot_x:.2f}, {robot_y:.2f}), Ball initial: ({ball_x:.2f}, {ball_y:.2f})"
        )
        print(f"  Attacking goal at x={self.enemy_goal_display_x:.2f}")

    def eval_status(self, game: Game) -> TestingStatus:
        if game.ball is None:
            print("Failure: Ball object is None.")
            return TestingStatus.FAILURE

        # Expected Outcome: Ball enters the enemy goal.
        # 'is_enemy_goal_on_right_side' determines which goal (left or right side of field)
        # is the enemy's goal based on our team's orientation.
        # game.is_ball_in_goal() checks if the ball's current position is within that goal area.
        is_enemy_goal_on_right_side = not game.my_team_is_right

        if game.is_ball_in_goal(right_goal=is_enemy_goal_on_right_side):
            print(
                f"Success! Goal scored. Ball at ({game.ball.p.x:.2f}, {game.ball.p.y:.2f})"
            )
            return TestingStatus.SUCCESS

        # Failure if the goal is not scored within the GOAL_SCORE_TIMEOUT.
        if time.time() - self.episode_start_time > GOAL_SCORE_TIMEOUT:
            print(
                f"Failure: Timeout. Ball at ({game.ball.p.x:.2f}, {game.ball.p.y:.2f})"
            )
            return TestingStatus.FAILURE

        return TestingStatus.IN_PROGRESS

    def get_n_episodes(self) -> int:
        return self.n_episodes


def test_solo_attacker(
    my_team_is_yellow: bool,
    my_team_is_right: bool,
    target_id: int,
    headless: bool,
    mode: str = "rsim",
):
    """
    Main test execution function for the SoloAttackerStrategy.
    This function is typically called by a test runner like pytest.
    It initializes the StrategyRunner with the SoloAttackerStrategy and SoloAttackerTestManager.
    """
    runner = StrategyRunner(
        strategy=SoloAttackerStrategy(target_id=target_id),
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        mode=mode,
        exp_friendly=3,
        exp_enemy=3,
    )
    test_manager = SoloAttackerTestManager(target_id=target_id)
    test_result = runner.run_test(
        testManager=test_manager,
        episode_timeout=GOAL_SCORE_TIMEOUT,
        rsim_headless=headless,
    )
    assert test_result, f"SoloAttackerStrategy test failed for target_id {target_id}."


if __name__ == "__main__":
    print("Running SoloAttackerStrategy Test Manually...")
    test_solo_attacker(
        my_team_is_yellow=True,
        my_team_is_right=False,  # Our team attacks the goal on the positive X side (enemy goal on right).
        target_id=0,  # ID of the friendly robot acting as the attacker.
        mode="rsim",
        headless=False,
    )
