import py_trees
from py_trees.composites import Sequence, Selector
from py_trees.decorators import Inverter, SuccessIsRunning
from strategy.common import AbstractStrategy
from strategy.utils.blackboard_utils import SetBlackboardVariable
from strategy.utils.selector_utils import HasBall, AtDribbleTarget, DribbledEnough
from strategy.utils.action_nodes import (
    GoToBallStep,
    DribbleMoveStep,
    StopStep,
    UpdateDribbleDistance,
)


class DribbleStrategy(AbstractStrategy):
    def __init__(self, robot_id: int, tolerance: float = 0.1, dribble_limit: float = 1):
        """Strategy to dribble the ball towards a specified target location."""
        self.robot_id = robot_id
        self.tolerance = tolerance
        self.dribble_limit = dribble_limit
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        return 1 <= n_runtime_friendly <= 3 and 1 <= n_runtime_enemy <= 3
    
    def _dribble_logic(self) -> py_trees.behaviour.Behaviour:
        at_target = Sequence(
                name="AtDribbleToTarget",
                memory=False,
                children=[AtDribbleTarget(self.tolerance)]
            )

        # Distinct 'not has ball' checks for each parent that needs one
        not_has_ball_wait = Inverter(
            name="NotHasBall?ForWait",
            child=HasBall(visual=True),
        )
        not_has_ball_get = Inverter(
            name="NotHasBall?ForGet",
            child=HasBall(),
        )

        # Keep stopping until the visual check says the ball has left.
        wait_for_release = Selector(
            name="WaitForBallLeave",
            memory=False,
            children=[
                # Immediately succeed once the ball is visually gone.
                not_has_ball_wait,
                # While the ball is still there, keep StopStep "running".
                SuccessIsRunning(
                    name="HoldStopUntilReleased",
                    child=StopStep(name="StopWhileBallHeld")
                ),
            ],
        )

        # Once dribbled enough, wait until the ball has left (visually), then reset distance.
        release_ball = Sequence(
            name="ReleaseBall",
            memory=False,
            children=[
                DribbledEnough(self.dribble_limit),
                wait_for_release,
                SetBlackboardVariable("ResetDribbleDistance", "dribbled_distance", 0.0),
            ],
        )

        with_ball = Selector(
            name="WithBallSelector",
            memory=False,
            children=[
                release_ball,                               # priority: release first if we've dribbled enough
                DribbleMoveStep(self.dribble_limit),        # otherwise keep dribbling forward
            ],
        )

        without_ball = Sequence(
            name="GetBallSeq",
            memory=True,
            children=[
                not_has_ball_get,   # we don't have the ball
                GoToBallStep(),
            ]
        )

        dribble_selector = Selector(
            name="DribbleSelector",
            memory=False,
            children=[
                at_target,
                Sequence(
                    name="HasBallSeq",
                    memory=False,
                    children=[HasBall(), with_ball]
                ),
                without_ball,
            ],
        )

        return Sequence(
            name="DribbleLogic",
            memory=False,
            children=[UpdateDribbleDistance(), dribble_selector]
        )
      
    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        root = Sequence(
            name="DribbleModule",
            memory=True,
            children=[
                SetBlackboardVariable("SetRobotID", "robot_id", self.robot_id),
                self._dribble_logic(),
            ],
        )
        return root

    def create_module(self) -> py_trees.behaviour.Behaviour:
        return self._dribble_logic()
