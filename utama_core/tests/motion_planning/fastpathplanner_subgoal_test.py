"""Direct unit tests for `FastPathPlanner._find_subgoal`/`check_segment`'s
detour-recursion failsafe -- no rsim/StrategyRunner needed, since the bug and
fix live entirely in this pure-geometry recursion.

Found live in a full_match_tournament.py replay
(counter_press_vs_tiki_taka_RK.pkl, 2026-09-01): a robot standing right at
the field's boundary wall, routing toward a target on the other side of the
field, got a carrot that flip-flopped between the real route and a bogus
point sitting exactly on the wall -- because `_find_subgoal`'s recursion
failsafe (`multiple > 10`) used to hand back the raw obstacle-intersection
point as if it were a valid subgoal whenever it gave up, and stepping
perpendicular to the robot->target line from a point on a long, nearly-
parallel obstacle (the boundary wall) can take 130+ steps to actually clear
it -- far more than the 10-step budget. `check_segment` then treated that
on-the-wall point as a legitimate detour, steering the robot straight at the
obstacle it was supposed to route around.

The fix only rejects the failsafe fallback when the *same* obstacle
`check_segment` originally asked to route around is still the one blocking
at the cutoff (a genuine dead end) -- not when some other obstacle is
blocking (a busy field, where the old "take the point anyway" answer is
still more useful than giving up entirely). See `_find_subgoal`'s docstring.
"""

import numpy as np

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.motion_planning.src.fastpathplanning.planner import FastPathPlanner


def _planner() -> FastPathPlanner:
    return FastPathPlanner(env=None)


class TestFindSubgoalDeadEnd:
    def test_stepping_along_a_long_parallel_obstacle_returns_none_at_the_wall(self):
        """Robot standing right at the right boundary wall (x=4.5), target
        far to the left -- the perpendicular-stepping direction that runs
        along the wall (not away from it) must give up rather than hand back
        a point still sitting on the wall."""
        planner = _planner()
        robot_pos = np.array([4.4857, -0.8647])
        target = np.array([3.27, -0.8647])
        wall = (np.array([4.5, 3.0]), np.array([4.5, -3.0]))
        obstacle_pos = np.array([4.5, -0.8647])  # robot's intersection with the wall

        # subgoal_direction=0 is the side that runs along the wall without
        # ever clearing it within a sane number of steps (confirmed: needs
        # 130+ steps one way, never clears going the other way at all).
        result = planner._find_subgoal(
            robot_pos,
            target,
            obstacle_pos,
            obstacles=[wall],
            subgoal_direction=0,
            multiple=1,
            origin_obstacle=wall,
        )
        assert result is None

    def test_check_segment_falls_through_to_direct_line_at_the_wall(self):
        """End-to-end through check_segment: with only the boundary wall in
        play and no way to detour around it within the step budget, the
        planner must give up on detouring entirely and route straight to the
        target -- not steer at the wall (the original bug) and not crash."""
        planner = _planner()
        fb = STANDARD_FIELD_DIMS.full_field_bounds
        robot_pos = np.array([4.4857, -0.8647])
        target = np.array([3.27, -0.8647])
        wall = (np.array([4.5, 3.0]), np.array([4.5, -3.0]))

        trajectory, length = planner.check_segment(
            (robot_pos, target), obstacles=[wall], recursion_length=0, target=target, field_bounds=fb, robot_id=1
        )
        assert len(trajectory) == 1
        assert np.allclose(trajectory[0][0], robot_pos)
        assert np.allclose(trajectory[0][1], target)

    def test_dead_end_only_applies_to_the_same_obstacle_still_blocking(self):
        """A failsafe timeout caused by a *different* obstacle than the one
        this call started out avoiding must still fall back to the old
        on-obstacle point -- rejecting the fallback is only correct for a
        genuine dead end against the original obstacle, not merely "search
        exhausted its budget in a crowded field" (confirmed live:
        `test_our_kickoff_nonzero_keeper_is_never_kicker_or_in_formation`
        needs exactly this in a busy 6-robot kickoff scrum -- rejecting the
        fallback unconditionally there regressed that test)."""
        planner = _planner()
        robot_pos = np.array([1.0, -1.83])
        target = np.array([0.12, 0.0])
        obstacle_pos = np.array([0.977, -1.845])
        origin = (np.array([0.977, -1.845]), np.array([0.977, -1.845]))
        # A different, very long obstacle (not `origin`) that keeps blocking
        # every stepped-away point within the search budget.
        other_far_obstacle = (np.array([2.0, -2.0]), np.array([-40.0, -25.0]))

        result = planner._find_subgoal(
            robot_pos,
            target,
            obstacle_pos,
            obstacles=[other_far_obstacle],
            subgoal_direction=0,
            multiple=11,  # already past the step budget on entry
            origin_obstacle=origin,
            blocked_by_origin=False,  # last blocker was NOT the origin obstacle
        )
        assert result is not None
        assert np.array_equal(result, obstacle_pos)
