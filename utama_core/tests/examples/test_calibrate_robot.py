"""`examples/calibrate_robot.py` writes what a run measured into the robot's profile, keeping
what it did not measure; the profile file reads back as written."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from utama_core.team_controller.src.controllers.common.robot_profile import (
    RobotProfile,
    load_profile,
    save_profile,
)

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
sys.path.insert(0, str(EXAMPLES))
try:
    import calibrate_robot as ex
finally:
    sys.path.remove(str(EXAMPLES))

KEPT = {
    "forward": 1.0,
    "backward": 0.5,
    "left": 0.7,
    "right": 0.7,
    "turn anticlockwise": 3.0,
    "turn clockwise": 2.0,
}


def test_the_carry_caps_are_the_slowest_directions():
    assert ex.carry_caps(KEPT) == (0.5, 2.0)


def test_a_direction_lost_at_its_slowest_or_untried_sets_no_cap():
    assert ex.carry_caps({**KEPT, "left": None}) == (None, 2.0)
    assert ex.carry_caps({}) == (None, None)


def test_a_run_writes_what_it_measured_and_keeps_the_rest():
    old = RobotProfile(dribbler=False, kick_bend_deg_per_cm=4.2, kick_bias_deg=2.7, tag_left_m=-0.002, carry_turn=1.0)
    tag = {"tag_forward_m": 0.0114, "tag_left_m": -0.0031, "spins": [{}, {}]}

    new, notes = ex.measured_profile(old, tag, None, {**KEPT, "turn clockwise": None})

    assert new == RobotProfile(
        dribbler=False,
        kick_bend_deg_per_cm=4.2,
        kick_bias_deg=2.7,
        tag_forward_m=0.011,
        tag_left_m=-0.003,
        carry_speed=0.5,
        carry_turn=1.0,
    )
    assert set(notes) == {"tag", "carry"}


def test_a_kick_fit_sets_bend_bias_speed_and_spread():
    bend = {
        "deg_per_cm": 4.19,
        "deg_per_cm_se": 0.56,
        "bias_deg": 2.66,
        "bias_se_deg": 0.9,
        "rms_deg": 5.9,
        "speed": 2.19,
        "kicks": 55,
    }
    new, notes = ex.measured_profile(RobotProfile(), None, bend, {})
    assert (new.kick_bend_deg_per_cm, new.kick_bias_deg) == (4.2, 2.7)
    assert (new.kick_speed, new.kick_spread_deg) == (2.19, 5.9)
    assert "55 kicks" in notes["kick"]


def test_a_profile_file_reads_back_as_written(tmp_path):
    profile = RobotProfile(kick_bend_deg_per_cm=4.2, kick_bias_deg=2.7, tag_left_m=-0.002, carry_speed=0.5)
    save_profile(3, profile, {"lid": "yellow 3"}, tmp_path)
    assert load_profile(3, tmp_path) == profile
    assert load_profile(4, tmp_path) == RobotProfile()  # no file: a blank profile


def test_the_committed_profiles_load():
    assert load_profile(0).kick_bend_deg_per_cm > 0 and load_profile(1).kick_bend_deg_per_cm > 0


LIDS = {0: ("yellow", 0), 1: ("blue", 0)}


def test_the_robot_to_calibrate_is_the_radio_given_the_lid_given_or_the_only_one():
    assert ex.pick(LIDS, 1, None) == 1
    assert ex.pick(LIDS, None, ("blue", 0)) == 1
    assert ex.pick({4: ("yellow", 2)}, None, None) == 4


def test_with_several_robots_found_the_operator_picks_one_until_it_is_one_found():
    answers = iter(["7", "x", "1"])
    assert ex.pick(LIDS, None, None, ask=lambda _: next(answers)) == 1


@pytest.mark.parametrize("radio, robot", [(5, None), (None, ("yellow", 9))])
def test_a_robot_not_found_stops_it(radio, robot):
    with pytest.raises(SystemExit):
        ex.pick(LIDS, radio, robot)
    with pytest.raises(SystemExit):
        ex.pick({}, None, None)
