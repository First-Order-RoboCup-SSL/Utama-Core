"""`examples/calibrate_tag.py` prints the `RobotProfile` fields for a measured tag, which correct
vision on real robots, so they must be the fields and signs `RobotProfile` and `on_robot` read."""

import sys
from pathlib import Path

from utama_core.team_controller.src.controllers.common.robot_profile import RobotProfile

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
sys.path.insert(0, str(EXAMPLES))
try:
    import calibrate_tag as ex
finally:
    sys.path.remove(str(EXAMPLES))


def test_the_printed_fields_make_the_measured_profile():
    result = {"tag_forward_m": -0.0003, "tag_left_m": -0.0017}

    fields = ex.profile_fields(result, -3.51)

    assert fields == "tag_forward_m=-0.000, tag_left_m=-0.002, tag_heading_deg=-3.5"
    assert eval(f"RobotProfile({fields})") == RobotProfile(tag_left_m=-0.002, tag_heading_deg=-3.5)


def test_a_part_not_measured_is_left_out():
    assert ex.profile_fields({"tag_forward_m": None}, -3.51) == "tag_heading_deg=-3.5"
    assert ex.profile_fields({"tag_forward_m": None}, None) == "nothing measured"
