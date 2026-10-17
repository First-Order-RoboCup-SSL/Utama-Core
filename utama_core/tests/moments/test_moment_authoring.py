"""Pure checks (no rsim) of hand-made moments in `tools/moments.py`: the template and the
validation errors."""

import dataclasses

import pytest

from tools import moments


def _template():
    return moments.template("probe", "a note", "NORMAL_START", ["keeps_clear"])


def _with_friendly(m, robots):
    return dataclasses.replace(m, scenario=dataclasses.replace(m.scenario, friendly_robots=tuple(robots)))


def _robot(robots, rid):
    return next(r for r in robots if r.id == rid)


def _replace_robot(robots, rid, **changes):
    return tuple(dataclasses.replace(r, **changes) if r.id == rid else r for r in robots)


def test_template_round_trips_through_load(tmp_path):
    m = _template()
    path = moments.save(m, tmp_path / "probe.json")
    again = moments.load(path)
    assert moments.to_dict(again) == moments.to_dict(m)
    assert again.sim_noise["seed"] == 1
    assert sorted(r.id for r in again.scenario.friendly_robots) == list(range(6))
    assert sorted(r.id for r in again.scenario.enemy_robots) == list(range(6))


def test_template_is_our_side_on_the_right_facing_left():
    m = _template()
    ours = {r.id: r for r in m.scenario.friendly_robots}
    theirs = {r.id: r for r in m.scenario.enemy_robots}
    assert ours[0].x > 0 and theirs[0].x < 0
    assert all(ours[i].x > 0 and theirs[i].x < 0 for i in range(6))


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
    with pytest.raises(moments.MomentError, match=r"friendly robot 2 at \(5.00, -1.20\) is off the pitch"):
        moments.validate(m)


def test_the_ball_must_be_on_the_pitch():
    m = dataclasses.replace(_template(), scenario=dataclasses.replace(_template().scenario, ball_y=3.5))
    with pytest.raises(moments.MomentError, match=r"ball at \(0.00, 3.50\) is off the pitch"):
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
        moments.validate(dataclasses.replace(_template(), strategy="build_nope_kernel_strategy"))
    with pytest.raises(moments.MomentError, match=r"unknown opponent 'nope'"):
        moments.validate(dataclasses.replace(_template(), opponent="build_nope_kernel_strategy"))


def test_seconds_must_be_positive():
    with pytest.raises(moments.MomentError, match=r"seconds must be positive, got 0"):
        moments.validate(dataclasses.replace(_template(), seconds=0))


def test_every_problem_is_reported_together():
    m = dataclasses.replace(_template(), checks=["nope"], seconds=-1)
    with pytest.raises(moments.MomentError) as err:
        moments.validate(m)
    assert "unknown check 'nope'" in str(err.value) and "seconds must be positive" in str(err.value)


# --- `play --seed`, and the foul listing's restart side ---------------------------------------


def test_with_seed_replaces_only_the_seed_and_does_not_mutate_the_moment():
    m = _template()
    seeded = moments.with_seed(m, 7)
    assert seeded.sim_noise["seed"] == 7
    assert {k: v for k, v in seeded.sim_noise.items() if k != "seed"} == {
        k: v for k, v in m.sim_noise.items() if k != "seed"
    }
    assert m.sim_noise["seed"] == 1


def test_with_seed_needs_noise_to_seed():
    with pytest.raises(moments.MomentError, match=r"no sim_noise to seed"):
        moments.with_seed(dataclasses.replace(_template(), sim_noise=None), 3)


@pytest.mark.parametrize(
    "command, we_are_yellow, side",
    [
        ("DIRECT_FREE_YELLOW", True, "ours"),
        ("BALL_PLACEMENT_YELLOW", True, "ours"),
        ("DIRECT_FREE_BLUE", True, "theirs"),
        ("DIRECT_FREE_BLUE", False, "ours"),
        ("NORMAL_START", True, None),
        ("STOP", True, None),
        ("TIMEOUT_YELLOW", True, None),
    ],
)
def test_restart_side_is_ours_theirs_or_neither(command, we_are_yellow, side):
    from utama_core.entities.referee.referee_command import RefereeCommand

    assert moments.restart_side(RefereeCommand[command], we_are_yellow) == side
    assert moments.restart_side(None, we_are_yellow) is None


def test_our_fouls_names_the_command_active_and_the_restart_side(tmp_path):
    import json

    def foul(t, side="friendly", rule="keep_out"):
        return {"rule": rule, "sim_time": t, "side": side, "robot_id": 1, "tactic": "X", "inferred": False}

    results = [
        {
            "config_a": "build_a_kernel_strategy",
            "config_b": "build_b_kernel_strategy",
            "stats": {"fouls": [foul(1.0), foul(5.0), foul(12.0), foul(20.0), foul(6.0, side="enemy")]},
        }
    ]
    (tmp_path / "summary.json").write_text(json.dumps({"results": results}))
    rows = [
        {"event": "referee", "sim_time": 0.0, "command": "NORMAL_START"},
        {"event": "referee", "sim_time": 4.0, "command": "DIRECT_FREE_BLUE"},
        {"event": "referee", "sim_time": 10.0, "command": "DIRECT_FREE_YELLOW"},
    ]
    (tmp_path / "a_vs_b.intentions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    fouls = moments.our_fouls(tmp_path)
    assert [(f["sim_time"], f["command"], f["restart"]) for f in fouls] == [
        (1.0, "NORMAL_START", None),
        (5.0, "DIRECT_FREE_BLUE", "theirs"),
        (12.0, "DIRECT_FREE_YELLOW", "ours"),
        (20.0, "DIRECT_FREE_YELLOW", "ours"),
    ]
    kept = moments.without_own_restarts(fouls)
    assert [f["sim_time"] for f in kept] == [1.0, 5.0]


def test_our_fouls_without_a_match_log_has_no_command(tmp_path):
    import json

    row = {"rule": "keep_out", "sim_time": 3.0, "side": "friendly", "robot_id": 1, "tactic": "X"}
    results = [
        {"config_a": "build_a_kernel_strategy", "config_b": "build_b_kernel_strategy", "stats": {"fouls": [row]}}
    ]
    (tmp_path / "summary.json").write_text(json.dumps({"results": results}))
    (fouls,) = moments.our_fouls(tmp_path)
    assert fouls["command"] is None and fouls["restart"] is None
    assert moments.without_own_restarts([fouls]) == [fouls]
