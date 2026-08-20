"""demo_dribbler_test.py — Test the dribbler with one robot on the Exhibition Road field.

Run:
    pixi run python demo_dribbler_test.py

What this does
--------------
- 1 yellow robot, 0 opponents, Exhibition Road field (3 m × 2.25 m)
- DribbleTactic: robot fetches the ball, then loops corner-to-corner around a
  rectangle, releasing and reacquiring the ball each segment
"""

from utama_core.config.field_params import GREAT_EXHIBITION_FIELD_DIMS
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.context import KernelContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.run import StrategyRunner
from utama_core.tactics.dribble import DribbleTactic

N_FRIENDLY = 1
MY_TEAM_IS_YELLOW = True
MY_TEAM_IS_RIGHT = True


def _build_dribbler_kernel_strategy(motion_controller: MotionController) -> KernelSchedulerStrategy:
    ctx = KernelContext(motion_controller=motion_controller)
    return KernelSchedulerStrategy(
        tactics={"dribble": DribbleTactic()},
        partitioner=KernelSchedulerStrategy.single_tactic_picker(lambda game, active: "dribble"),
        outfield_robot_ids=(0,),
        ctx=ctx,
    )


def main() -> None:
    runner = StrategyRunner(
        strategy=AbstractStrategy(build_kernel_strategy=_build_dribbler_kernel_strategy),
        my_team_is_yellow=MY_TEAM_IS_YELLOW,
        my_team_is_right=MY_TEAM_IS_RIGHT,
        mode="rsim",
        exp_friendly=N_FRIENDLY,
        exp_enemy=0,
        full_field_dims=GREAT_EXHIBITION_FIELD_DIMS,
        show_live_status=True,
    )

    runner.run()


if __name__ == "__main__":
    main()
