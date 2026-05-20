"""ball_placement_strategy.py — Skeleton for the ball placement feature.

Your task
---------
Implement ``BallPlacementStep.update()`` so that, when the referee issues
BALL_PLACEMENT_YELLOW (our team places), one robot picks up the ball with its
dribbler and carries it to ``game.referee.designated_position``, while the
other robot waits clear of the ball.

The referee integration (deciding *when* to use this behaviour) is already
wired up in the referee override tree — you do not need to touch the referee
config.  Your work is entirely inside this file's strategy and behaviour tree.

How ball placement works
------------------------
1. Referee issues BALL_PLACEMENT_YELLOW with a ``designated_position`` (x, y).
2. One robot (the "placer") drives to the ball with its dribbler on to capture it.
3. Once it has the ball (``robot.has_ball`` is True), it carries it to
   ``designated_position`` and stops.
4. When the ball is within ``BALL_PLACEMENT_DONE_DISTANCE`` of the target the
   referee auto-advances to DIRECT_FREE_YELLOW → NORMAL_START.
5. All other robots must stay at least ``BALL_KEEP_OUT_DISTANCE`` from the ball.

Useful references
-----------------
- ``game.referee.designated_position`` — (x, y) tuple, the placement target.
- ``game.ball.p``                       — current ball position (Vector2D-like).
- ``robot.has_ball``                    — True when the robot's IR dribbler sensor
                                          detects the ball.
- ``move(game, motion_controller, robot_id, target, orientation, dribbling=True)``
  — send a motion command; pass ``dribbling=True`` to activate the dribbler.
- ``empty_command(False)``              — stop a robot in place.
- ``BALL_PLACEMENT_DONE_DISTANCE``      — 0.15 m; ball must be this close to target.
- ``BALL_KEEP_OUT_DISTANCE``            — 0.8 m; clearance for non-placer robots.

Running the demo
----------------
    pixi run python demo_ball_placement.py

The RSim window opens and the browser GUI at http://localhost:8080 lets you
issue BALL_PLACEMENT_YELLOW commands from the "Manual Commands" panel and watch
your strategy respond.

Running the tests
-----------------
    pixi run pytest utama_core/tests/strategy_runner/test_ball_placement_rsim.py -v

There are three tests that must pass before the feature is considered complete:
  1. placer robot drives toward the ball after the command is issued.
  2. placer carries the ball toward the designated position (dribbler on).
  3. non-placer robot stays outside the keep-out radius throughout.
"""

from typing import Optional

import py_trees

from utama_core.config.referee_constants import (
    BALL_KEEP_OUT_DISTANCE,
    BALL_PLACEMENT_DONE_DISTANCE,
)
from utama_core.entities.data.vector import Vector2D
from utama_core.skills.src.utils.move_utils import empty_command, move
from utama_core.strategy.common import AbstractBehaviour, AbstractStrategy
from utama_core.strategy.referee.actions import _clear_to_legal_positions


class BallPlacementStep(AbstractBehaviour):
    """Move one robot to place the ball at the referee's designated position.

    Implementation guide
    --------------------
    The ``update()`` method is called every simulation tick while the referee
    command is BALL_PLACEMENT_YELLOW.  It must:

    1. Read ``game.referee.designated_position`` — the (x, y) target tuple.
    2. If the ball is already at the target (within ``BALL_PLACEMENT_DONE_DISTANCE``),
       stop all robots and return RUNNING (the referee will advance the state).
    3. Pick the robot closest to the ball as the placer.
    4. If the placer does not yet have the ball (``robot.has_ball`` is False),
       drive it to the ball with the dribbler on.
    5. Once the placer has the ball, drive it to ``designated_position`` with
       the dribbler still on.
    6. Keep all non-placer robots at least ``BALL_KEEP_OUT_DISTANCE`` away from
       the ball by calling ``_clear_to_legal_positions``.

    Return ``py_trees.common.Status.RUNNING`` on every tick (the referee
    override tree holds here until the command changes).

    Tips
    ----
    - Use ``Vector2D(target[0], target[1])`` to convert the tuple to a Vector2D.
    - Orientation: ``robot.p.angle_to(target_pos)`` gives the heading angle.
    - Call ``move(..., dribbling=True)`` to activate the dribbler.
    - Write commands to ``self.blackboard.cmd_map[robot_id]`` for each robot.
    """

    def update(self) -> py_trees.common.Status:
        # TODO: implement ball placement logic here.
        #
        # Skeleton to get you started:
        #
        #   game = self.blackboard.game
        #   ref = game.referee
        #   motion_controller = self.blackboard.motion_controller
        #
        #   target = ref.designated_position   # (x, y) or None
        #   ball   = game.ball                 # Ball | None
        #
        #   # Pick placer (closest robot to ball)
        #   placer_id = min(
        #       game.friendly_robots,
        #       key=lambda rid: game.friendly_robots[rid].p.distance_to(ball.p),
        #   )
        #
        #   # Drive placer; clear everyone else
        #   ...
        #
        # Remove this line once you have a real implementation:
        game = self.blackboard.game
        for robot_id in game.friendly_robots:
            self.blackboard.cmd_map[robot_id] = empty_command(False)
        return py_trees.common.Status.RUNNING


class BallPlacementStrategy(AbstractStrategy):
    """2v2 ball placement strategy for the Exhibition Road field.

    The referee override tree (built into AbstractStrategy) automatically handles
    HALT, STOP, kickoff, and penalty commands — you only need to implement what
    happens during BALL_PLACEMENT_YELLOW (i.e., in ``BallPlacementStep``).

    During NORMAL_START / FORCE_START the ``IdleRoot`` sequence at the bottom of
    the tree runs — replace it with your regular game strategy once placement works.
    """

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        root = py_trees.composites.Sequence(name="BallPlacementRoot", memory=False)
        root.add_child(BallPlacementStep(name="PlaceBall"))
        return root

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int) -> bool:
        return n_runtime_friendly >= 1

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool) -> bool:
        return True

    def get_min_bounding_req(self) -> Optional[object]:
        return None
