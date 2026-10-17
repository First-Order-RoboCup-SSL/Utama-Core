"""Pure checks (no rsim) of hand-made and tactic moments in `tools/moments.py`: the template, the
validation errors, the mem decoding and the one-off tactic strategy."""

import dataclasses

import pytest

from tools import moments
from utama_core.entities.data.vector import Vector2D
from utama_core.tactics.defense import DefenseTactic
from utama_core.tactics.dribble import DribbleTactic
from utama_core.tactics.give_and_go import GiveAndGoMem, GiveAndGoTactic
from utama_core.tactics.pass_and_shoot import PassAndShootTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic


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
    assert again.tactic is None
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


def test_tactic_classes_are_the_public_tactics():
    classes = moments.tactic_classes()
    assert {"GiveAndGoTactic", "DefenseTactic", "GoalkeeperTactic"} <= set(classes)
    assert all(not name.startswith("_") for name in classes)
    assert "BaseTactic" not in classes


def test_the_example_tactic_moment_is_played_by_test_moments():
    from utama_core.tests.moments import test_moments

    ids = [p.id for p in test_moments._cases()]
    assert "give_and_go_carrier_mid_pass-GiveAndGoTactic" in ids


def test_strategies_for_a_tactic_moment_is_its_class_name():
    m = dataclasses.replace(_template(), tactic={"class": "GiveAndGoTactic", "robots": [1, 2], "mem": None})
    assert moments.strategies_for(m) == ["GiveAndGoTactic"]
    assert moments.short_name("GiveAndGoTactic") == "GiveAndGoTactic"


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


def _tactic_moment(**tactic):
    return dataclasses.replace(
        _template(), tactic={"class": "GiveAndGoTactic", "robots": [1, 2], "mem": None, **tactic}
    )


def test_a_tactic_moment_is_valid():
    moments.validate(_tactic_moment())


def test_a_tactic_class_must_be_a_public_tactic():
    with pytest.raises(moments.MomentError, match=r"tactic class 'NoSuchTactic' is not a public Tactic"):
        moments.validate(_tactic_moment(**{"class": "NoSuchTactic"}))


def test_tactic_robots_are_outfield_and_distinct():
    with pytest.raises(moments.MomentError, match=r"tactic robots must be distinct outfield ids"):
        moments.validate(_tactic_moment(robots=[0, 1]))
    with pytest.raises(moments.MomentError, match=r"tactic robots must be distinct outfield ids"):
        moments.validate(_tactic_moment(robots=[1, 1]))
    with pytest.raises(moments.MomentError, match=r"tactic robots must be distinct outfield ids"):
        moments.validate(_tactic_moment(robots=[]))


def test_a_tactic_moment_must_not_name_a_strategy():
    m = dataclasses.replace(_tactic_moment(), strategy="build_clear_danger_kernel_strategy")
    with pytest.raises(moments.MomentError, match=r'drop "strategy"'):
        moments.validate(m)


def test_unknown_tactic_keys_are_named():
    with pytest.raises(moments.MomentError, match=r"tactic has unknown keys \['nope'\]"):
        moments.validate(_tactic_moment(nope=1))


def test_a_bad_mem_field_is_reported_as_a_moment_error():
    with pytest.raises(moments.MomentError, match=r"GiveAndGoMem has no mem field 'nope'"):
        moments.validate(_tactic_moment(mem={"nope": 1}))


# --- mem decoding -------------------------------------------------------------------------------


def test_mem_fields_are_set_by_name_and_the_rest_keep_their_initial_value():
    mem = moments.decode_mem(GiveAndGoTactic, {"carrier_id": 1, "carry_origin": [1.0, 2.0], "hop_count": 2})
    assert isinstance(mem, GiveAndGoMem)
    assert mem.carrier_id == 1 and mem.hop_count == 2
    assert isinstance(mem.carry_origin, Vector2D)
    assert (mem.carry_origin.x, mem.carry_origin.y) == (1.0, 2.0)
    assert mem.receiver_id is None and mem.first_touch_hold_spent is False


def test_a_null_mem_is_the_initial_mem():
    assert moments.decode_mem(GiveAndGoTactic, None) == GiveAndGoTactic().initial_mem()


def test_a_null_optional_field_decodes_to_none():
    mem = moments.decode_mem(GiveAndGoTactic, {"carrier_id": None, "carry_origin": None})
    assert mem.carrier_id is None and mem.carry_origin is None


def test_json_lists_decode_to_tuples():
    assert moments.decode_mem(DefenseTactic, {"shadow_ids": [1, 2]}).shadow_ids == (1, 2)


def test_a_fixed_length_tuple_decodes_and_checks_its_length():
    assert moments.decode_mem(PassAndShootTactic, {"assigned_pair": [1, 2]}).assigned_pair == (1, 2)
    with pytest.raises(moments.MomentError, match=r"PassAndShootMem.assigned_pair must have 2 items"):
        moments.decode_mem(PassAndShootTactic, {"assigned_pair": [1, 2, 3]})


def test_a_vector2d_is_an_xy_pair():
    with pytest.raises(moments.MomentError, match=r"GiveAndGoMem.carry_origin must be Vector2D"):
        moments.decode_mem(GiveAndGoTactic, {"carry_origin": [1.0]})
    with pytest.raises(moments.MomentError, match=r"GiveAndGoMem.carry_origin must be Vector2D"):
        moments.decode_mem(GiveAndGoTactic, {"carry_origin": "here"})


def test_an_unsupported_field_type_is_named():
    with pytest.raises(moments.MomentError, match=r"PressAndContainMem.marks has type dict, which a moment cannot set"):
        moments.decode_mem(PressAndContainTactic, {"marks": {}})
    with pytest.raises(
        moments.MomentError, match=r"PassAndShootMem.pass_and_score has type PassAndScoreMem, which a moment cannot set"
    ):
        moments.decode_mem(PassAndShootTactic, {"pass_and_score": {}})


def test_a_value_of_the_wrong_type_is_rejected():
    with pytest.raises(moments.MomentError, match=r"GiveAndGoMem.hop_count must be int, got '2'"):
        moments.decode_mem(GiveAndGoTactic, {"hop_count": "2"})
    with pytest.raises(moments.MomentError, match=r"GiveAndGoMem.hop_count must be int"):
        moments.decode_mem(GiveAndGoTactic, {"hop_count": True})
    with pytest.raises(moments.MomentError, match=r"GiveAndGoMem.first_touch_hold_spent must be bool"):
        moments.decode_mem(GiveAndGoTactic, {"first_touch_hold_spent": 1})
    with pytest.raises(moments.MomentError, match=r"GiveAndGoMem.carrier_id must be int"):
        moments.decode_mem(GiveAndGoTactic, {"carrier_id": 1.5})


def test_an_int_is_accepted_for_a_float_field():
    value = moments.decode_mem(DribbleTactic, {"dribbled_distance": 2}).dribbled_distance
    assert value == 2.0 and isinstance(value, float)


def test_a_mem_must_be_an_object():
    with pytest.raises(moments.MomentError, match=r"GiveAndGoMem mem must be an object"):
        moments.decode_mem(GiveAndGoTactic, [1])


# --- the one-off tactic strategy ----------------------------------------------------------------


def test_the_tactic_starts_from_the_decoded_mem_not_the_initial_one():
    mem = moments.decode_mem(GiveAndGoTactic, {"carrier_id": 1, "receiver_id": 2, "hop_count": 1})
    started = moments._started(GiveAndGoTactic, mem)()
    assert started.initial_mem() == mem
    assert started.initial_mem() is not started.initial_mem()  # a copy each time, never the stored one
    assert type(started).__name__ == "GiveAndGoTactic"


def test_the_built_strategy_runs_the_tactic_from_its_mem_and_holds_the_rest():
    tactic = {"class": "GiveAndGoTactic", "robots": [1, 2], "mem": {"carrier_id": 1, "hop_count": 1}}
    kernel = moments.build_tactic_kernel_strategy(moments.OUTFIELD_ROBOT_IDS, tactic)(None)
    assert kernel._tactics["GiveAndGoTactic"].initial_mem().carrier_id == 1
    assert kernel._tactics["GiveAndGoTactic"].initial_mem().hop_count == 1

    free = frozenset(moments.OUTFIELD_ROBOT_IDS)
    split = kernel._partitioner(None, free, None, frozenset({"GiveAndGoTactic", moments.HOLD_ID}))
    assert split == {"GiveAndGoTactic": frozenset({1, 2}), moments.HOLD_ID: frozenset({3, 4, 5})}
    # An inapplicable or committed tactic is not offered: its robots hold instead.
    held = kernel._partitioner(None, free - {1, 2}, None, frozenset({moments.HOLD_ID}))
    assert held == {moments.HOLD_ID: frozenset({3, 4, 5})}


# --- the tactic that never runs, `play --seed`, and the foul listing's restart side ---------------


def test_a_tactic_moment_fails_for_the_robots_its_tactic_never_ticked_with():
    from utama_core.entities.referee.referee_command import RefereeCommand

    m = _tactic_moment()
    never = [moments.Tick(t=0.0, command=RefereeCommand.NORMAL_START, ball=None, ours={}, our_speeds={}, our_foul=None)]
    assert "GiveAndGoTactic never ran on robots [1, 2]" in moments.judge(m, never)
    ran_one = [dataclasses.replace(never[0], tactic_robots=frozenset({1}))]
    assert "GiveAndGoTactic never ran on robots [2]" in moments.judge(m, ran_one)
    ran_both = [dataclasses.replace(never[0], tactic_robots=frozenset({1, 2}))]
    assert not any("never ran" in f for f in moments.judge(m, ran_both))


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
