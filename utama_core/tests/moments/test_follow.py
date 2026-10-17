"""Pure checks (no rsim) of `play --follow`: which recorded pose a play time reads, when the
opponents hand back to their strategy, the servo's command, and the `follow_s` field."""

import dataclasses
import math

import numpy as np
import pytest

from tools import moments
from utama_core.replay.scenario import recorded_index

TIMES = np.array([0.0, 1 / 60, 2 / 60, 3 / 60])


@pytest.mark.parametrize(
    "tau, index",
    [
        (-0.5, 0),  # before the path's start: the first frame
        (0.0, 0),
        (0.01, 0),  # between frames 0 and 1: the earlier one
        (1 / 60, 1),  # exactly on a frame: that frame
        (2.5 / 60, 2),
        (3 / 60, 3),
        (9.0, 3),  # past the end: the last frame
    ],
)
def test_recorded_index_reads_the_last_frame_at_or_before_the_play_time(tau, index):
    assert recorded_index(TIMES, tau) == index


@pytest.mark.parametrize(
    "tau, active",
    [(0.0, True), (0.99, True), (1.0, False), (2.5, False)],
)
def test_follow_hands_back_to_the_strategy_at_follow_s(tau, active):
    assert moments.follow_active(tau, 1.0) is active


def test_follow_zero_never_steers():
    assert not any(moments.follow_active(tau, 0.0) for tau in (0.0, 0.5, 10.0))


def test_follow_command_is_zero_on_the_recorded_pose_at_rest():
    cmd = moments.follow_command((1.0, -2.0, 0.3), (1.0, -2.0, 0.3), (0.0, 0.0))
    assert (cmd.local_forward_vel, cmd.local_left_vel, cmd.angular_vel) == (0.0, 0.0, 0.0)
    assert not (cmd.kick or cmd.chip or cmd.dribble)


def test_follow_command_drives_toward_the_target_in_the_robot_frame():
    # Facing +x: a target 1 m ahead is straight forward; 1 m to its left (+y) is pure left.
    ahead = moments.follow_command((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0))
    assert ahead.local_forward_vel > 0 and abs(ahead.local_left_vel) < 1e-9
    left = moments.follow_command((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0))
    assert abs(left.local_forward_vel) < 1e-9 and left.local_left_vel > 0
    # Facing +y: a target at +x is on its right, so the left velocity is negative.
    right = moments.follow_command((0.0, 0.0, math.pi / 2), (1.0, 0.0, math.pi / 2), (0.0, 0.0))
    assert right.local_left_vel < 0


def test_follow_command_feeds_the_recorded_velocity_forward():
    cmd = moments.follow_command((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.5, 0.0))
    assert cmd.local_forward_vel == pytest.approx(1.5)


def test_follow_command_turns_the_short_way_and_caps_the_turn():
    assert moments.follow_command((0, 0, 0.0), (0, 0, 0.5), (0, 0)).angular_vel > 0
    assert moments.follow_command((0, 0, 0.0), (0, 0, -0.5), (0, 0)).angular_vel < 0
    # heading 3.0 to -3.0 is 0.28 rad the short way (through pi), not 6 rad the long way
    short = moments.follow_command((0, 0, 3.0), (0, 0, -3.0), (0, 0))
    assert abs(short.angular_vel) == pytest.approx(moments.FOLLOW_KTHETA * (2 * math.pi - 6.0), rel=1e-6)
    assert moments.follow_command((0, 0, 0.0), (0, 0, 3.0), (0, 0)).angular_vel == moments.FOLLOW_MAX_TURN_RADPS


def test_follow_command_caps_the_speed():
    cmd = moments.follow_command((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0))
    assert math.hypot(cmd.local_forward_vel, cmd.local_left_vel) == pytest.approx(moments.FOLLOW_MAX_SPEED_MPS)


def _captured_moment(**fields):
    m = moments.template("follow_test", "note", "NORMAL_START", ["keeps_clear"])
    return dataclasses.replace(m, **fields)


def test_follow_s_needs_a_captured_source_and_is_not_negative():
    with pytest.raises(moments.MomentError, match="follow_s needs"):
        moments.validate(_captured_moment(follow_s=2.0))
    with pytest.raises(moments.MomentError, match="zero or more"):
        moments.validate(_captured_moment(follow_s=-1.0))
    moments.validate(_captured_moment(follow_s=0.0))  # the default: nothing to check


def test_follow_s_round_trips_and_is_left_out_of_moments_that_do_not_use_it():
    source = {"replay": "/x/a_vs_b.npz", "t": 3.0}
    m = _captured_moment(source=source, follow_s=2.0)
    d = moments.to_dict(m)
    assert d["follow_s"] == 2.0
    assert moments.from_dict(d).follow_s == 2.0
    assert "follow_s" not in moments.to_dict(dataclasses.replace(m, follow_s=0.0))
    assert moments.from_dict(moments.to_dict(dataclasses.replace(m, follow_s=0.0))).follow_s == 0.0
