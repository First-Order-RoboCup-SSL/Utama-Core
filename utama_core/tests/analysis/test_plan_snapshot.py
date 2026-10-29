"""tools/plan_snapshot.py's drawing function, on the kick-off state of
test_fpp_inside_ball_keep_out.py (no simulator)."""

import math

import numpy as np

from utama_core.analysis.plan_snapshot import plan_snapshot
from utama_core.config.referee_constants import BALL_KEEP_OUT_DISTANCE
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot


def test_snapshot_draws_the_planners_carrot_and_flags_a_robot_inside_keep_out(tmp_path):
    us = (0.762, -0.225)
    round_the_circle = (0.8 * math.cos(math.radians(-76)), 0.8 * math.sin(math.radians(-76)))
    game = build_game(
        {4: make_robot(4, is_friendly=True, x=us[0], y=us[1])},
        {0: make_robot(0, is_friendly=False, x=0.12, y=0.0)},
        make_ball(0.0, 0.0),
        my_team_is_right=False,
        my_team_is_yellow=False,
        referee_command=RefereeCommand.PREPARE_KICKOFF_YELLOW,
    )
    out = tmp_path / "kickoff.png"

    snap = plan_snapshot(game, 4, round_the_circle, out, title="kick-off")

    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert snap.path == out
    assert snap.inside_keep_out is True
    assert snap.keeps_off_ball is True
    assert snap.carrot.shape == (2,) and np.all(np.isfinite(snap.carrot))
    assert snap.robot_to_ball < BALL_KEEP_OUT_DISTANCE
