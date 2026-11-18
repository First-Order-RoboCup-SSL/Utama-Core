"""Pure checks (no rsim) of moments in `tools/debug/moments.py`: loading, saving, the validation errors and
`play --seed`. (Hand-made templates, `tools/debug/moments.py new`, were removed on 2026-11-11: moments are captured
from replays.)"""

import dataclasses
import re

import pytest

from tools.debug import moments

PROBE = moments.MOMENTS_DIR / "free_kick_defender_dodges_into_ball.json"


def _template():
    """A captured moment (seed 2), renamed, checking `keeps_clear` only."""
    return dataclasses.replace(moments.load(PROBE), name="probe", note="a note", checks=["keeps_clear"])


def _with_friendly(m, robots):
    return dataclasses.replace(m, scenario=dataclasses.replace(m.scenario, friendly_robots=tuple(robots)))


def _robot(robots, rid):
    return next(r for r in robots if r.id == rid)


def _replace_robot(robots, rid, **changes):
    return tuple(dataclasses.replace(r, **changes) if r.id == rid else r for r in robots)


def test_a_moment_round_trips_through_save_and_load(tmp_path):
    m = _template()
    path = moments.save(m, tmp_path / "probe.json")
    again = moments.load(path)
    assert moments.to_dict(again) == moments.to_dict(m)
    assert sorted(r.id for r in again.scenario.friendly_robots) == list(range(6))
    assert sorted(r.id for r in again.scenario.enemy_robots) == list(range(6))


def test_existing_moments_still_load():
    for path in sorted(moments.MOMENTS_DIR.glob("*.json")):
        moments.load(path)


def test_referee_rule_names_come_from_the_code():
    names = moments.referee_rule_names()
    assert {"keep_out", "double_touch", "excessive_dribbling", "goal"} <= names


# --- validation ---------------------------------------------------------------------------------


def test_each_side_needs_ids_0_to_5_exactly_once():
    m = _template()
    friendly = _replace_robot(m.scenario.friendly_robots, 5, id=3)
    with pytest.raises(moments.MomentError, match=r"friendly robots must have ids 0-5 exactly once"):
        moments.validate(_with_friendly(m, friendly))


def test_robots_must_be_on_the_pitch():
    m = _with_friendly(_template(), _replace_robot(_template().scenario.friendly_robots, 2, x=5.0))
    with pytest.raises(moments.MomentError, match=r"friendly robot 2 at \(5.00, -?\d+\.\d\d\) is off the pitch"):
        moments.validate(m)


def test_the_ball_must_be_on_the_pitch():
    m = dataclasses.replace(_template(), scenario=dataclasses.replace(_template().scenario, ball_y=3.5))
    with pytest.raises(moments.MomentError, match=r"ball at \(-?\d+\.\d\d, 3.50\) is off the pitch"):
        moments.validate(m)


def test_an_unknown_check_is_named():
    m = dataclasses.replace(_template(), checks=["keeps_clera"])
    with pytest.raises(moments.MomentError, match=r"unknown check 'keeps_clera'"):
        moments.validate(m)


def test_checks_must_not_be_empty():
    with pytest.raises(moments.MomentError, match=r"checks is empty"):
        moments.validate(dataclasses.replace(_template(), checks=[]))


def test_a_real_rule_name_and_star_are_valid_checks():
    moments.validate(dataclasses.replace(_template(), checks=["double_touch", "*"]))


def test_unknown_strategy_and_opponent_are_named():
    with pytest.raises(moments.MomentError, match=r"unknown strategy 'nope'"):
        moments.validate(dataclasses.replace(_template(), strategy="nope"))
    with pytest.raises(moments.MomentError, match=r"unknown opponent 'nope'"):
        moments.validate(dataclasses.replace(_template(), opponent="nope"))


def test_seconds_must_be_positive():
    with pytest.raises(moments.MomentError, match=r"seconds must be positive, got 0"):
        moments.validate(dataclasses.replace(_template(), seconds=0))


def test_every_problem_is_reported_together():
    m = dataclasses.replace(_template(), checks=["nope"], seconds=-1)
    with pytest.raises(moments.MomentError) as err:
        moments.validate(m)
    assert "unknown check 'nope'" in str(err.value) and "seconds must be positive" in str(err.value)


# --- `play --seed` and `--against` ---------------------------------------------------------------


def test_with_seed_replaces_only_the_seed_and_does_not_mutate_the_moment():
    m = _template()
    seeded = moments.with_seed(m, 7)
    assert seeded.sim_noise["seed"] == 7
    assert {k: v for k, v in seeded.sim_noise.items() if k != "seed"} == {
        k: v for k, v in m.sim_noise.items() if k != "seed"
    }
    assert m.sim_noise["seed"] == 2


def test_with_seed_needs_noise_to_seed():
    with pytest.raises(moments.MomentError, match=r"no sim_noise to seed"):
        moments.with_seed(dataclasses.replace(_template(), sim_noise=None), 3)


def test_against_worktrees_sit_beside_the_repo_in_an_undated_folder():
    """Until 2026-11-11 `--against` used `moments-work-20261017/against`, a dated folder from one session."""
    assert moments.AGAINST_DIR.parent == moments.REPO_ROOT.parent
    assert moments.AGAINST_DIR.name == f"{moments.REPO_ROOT.name}-moments-against"
    assert not re.search(r"\d{8}", str(moments.AGAINST_DIR.relative_to(moments.REPO_ROOT.parent)))
