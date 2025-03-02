
from typing import Dict, Tuple
from config.starting_formation import LEFT_START_ONE, RIGHT_START_ONE
from entities.data.command import RobotCommand
from entities.game.present_future_game import PresentFutureGame
from motion_planning.src.pid.pid import PID, TwoDPID
from robot_control.src.skills import face_ball, go_to_point
from strategy.behaviour_tree_strategy import BehaviourTreeStrategy
from strategy.strategy import Strategy
import numpy as np

class StartupBehaviourTree(py_trees.behaviour.Behaviour):
    """Demonstrates the at-a-distance style action behaviour.

    This behaviour connects to a separately running process
    (initiated in setup()) and proceeeds to work with that subprocess to
    initiate a task and monitor the progress of that task at each tick
    until completed. While the task is running the behaviour returns
    :data:`~py_trees.common.Status.RUNNING`.

    On completion, the the behaviour returns with success or failure
    (depending on success or failure of the task itself).

    Key point - this behaviour itself should not be doing any work!
    """

    def __init__(self, name: str):
        """Configure the name of the behaviour."""
        super(Action, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def setup(self, **kwargs: int) -> None:
        """Kickstart the separate process this behaviour will work with.

        Ordinarily this process will be already running. In this case,
        setup is usually just responsible for verifying it exists.
        """
        self.logger.debug(
            "%s.setup()->connections to an external process" % (self.__class__.__name__)
        )
        self.parent_connection, self.child_connection = multiprocessing.Pipe()
        self.planning = multiprocessing.Process(
            target=planning, args=(self.child_connection,)
        )
        atexit.register(self.planning.terminate)
        self.planning.start()

    def initialise(self) -> None:
        """Reset a counter variable."""
        self.logger.debug(
            "%s.initialise()->sending new goal" % (self.__class__.__name__)
        )
        self.parent_connection.send(["new goal"])
        self.percentage_completion = 0

    def update(self) -> py_trees.common.Status:
        """Increment the counter, monitor and decide on a new status."""
        new_status = py_trees.common.Status.RUNNING
        if self.parent_connection.poll():
            self.percentage_completion = self.parent_connection.recv().pop()
            if self.percentage_completion == 100:
                new_status = py_trees.common.Status.SUCCESS
        if new_status == py_trees.common.Status.SUCCESS:
            self.feedback_message = "Processing finished"
            self.logger.debug(
                "%s.update()[%s->%s][%s]"
                % (
                    self.__class__.__name__,
                    self.status,
                    new_status,
                    self.feedback_message,
                )
            )
        else:
            self.feedback_message = "{0}%".format(self.percentage_completion)
            self.logger.debug(
                "%s.update()[%s][%s]"
                % (self.__class__.__name__, self.status, self.feedback_message)
            )
        return new_status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """Nothing to clean up in this example."""
        self.logger.debug(
            "%s.terminate()[%s->%s]"
            % (self.__class__.__name__, self.status, new_status)
        )

class (BehaviourTreeStrategy):
    def __int
    bt_strategy = BehaviourTreeStrategy(sim_robot_controller, Behaviour)

    # def __init__(self, pid_oren: PID, pid_trans: TwoDPID):
    #     self.pid_oren = pid_oren
    #     self.pid_trans = pid_trans

    # def done():
        

    # def step(self, present_future_game: PresentFutureGame) -> Dict[int, RobotCommand]:
    #     START_FORMATION = RIGHT_START_ONE if present_future_game.current.my_team_is_right else LEFT_START_ONE 
        
    #     commands = {}
    #     for robot_id, robot_data in enumerate(present_future_game.current.friendly_robots.items()):
    #         target_coords = START_FORMATION[robot_id]
    #         commands[robot_id] = self._calculate_robot_velocities(
    #             robot_id, target_coords, present_future_game, face_ball=True
    #         )

    #     return commands

    # def _calculate_robot_velocities(
    #     self,
    #     robot_id: int,
    #     target_coords: Tuple[float, float],
    #     present_future_game: PresentFutureGame
    #     face_ball=False,
    # ) -> RobotCommand:
        
    #     ball_p = present_future_game.current.ball.p
    #     current_p = present_future_game.current.friendly_robots[robot_id].p
    #     target_oren=(ball_p - current_p).phi if face_ball else None
    #     return go_to_point(
    #         present_future_game.current,
    #         self.pid_oren,
    #         self.pid_trans,
    #         robot_id,
    #         target_coords,
    #         target_oren
    #     )
