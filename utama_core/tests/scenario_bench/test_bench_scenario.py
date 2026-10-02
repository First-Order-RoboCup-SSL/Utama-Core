"""Tests for `utama_core.scenario_bench.bench_scenario` (bench scenario schema,
provenance, lifecycle, and the static validity screen).

See that module's docstring for the design constraints these tests pin:
policy-agnostic field state only, conservative lifecycle default, and a
static screen that catches malformed snapshots before any dynamic (play-
forward) screening or scoring happens.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.scenario import RobotState, Scenario
from utama_core.scenario_bench.bench_scenario import (
    _DUPLICATE_BALL_M,
    _DUPLICATE_ROBOT_M,
    _JITTER_POS_M,
    BenchScenario,
    ScenarioFamily,
    ScenarioLifecycle,
    ScenarioProvenance,
    ScenarioTrigger,
    drop_near_duplicates,
    is_near_duplicate,
    jittered,
    load_bank,
    save_bank,
    static_screen,
)


def _rs(id_: int, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> RobotState:
    return RobotState(id=id_, x=x, y=y, orientation=0.0, vx=vx, vy=vy)


def _valid_scenario(**overrides) -> Scenario:
    defaults = dict(
        sim_time=0.0,
        ball_x=0.0,
        ball_y=0.0,
        ball_vx=0.0,
        ball_vy=0.0,
        friendly_robots=(_rs(0, -4.0, 0.0), _rs(1, -1.0, 0.5)),
        enemy_robots=(_rs(0, 4.0, 0.0), _rs(1, 1.0, -0.5)),
        referee_command=RefereeCommand.NORMAL_START,
        source_replay=Path("test"),
        config_a_name=None,
        config_b_name=None,
        frame_ts=0.0,
    )
    defaults.update(overrides)
    return Scenario(**defaults)


def test_static_screen_accepts_well_formed_scenario():
    result = static_screen(_valid_scenario())
    assert result.ok
    assert result.violations == ()


def test_static_screen_rejects_ball_out_of_bounds():
    result = static_screen(_valid_scenario(ball_x=100.0))
    assert not result.ok
    assert any("ball out of bounds" in v for v in result.violations)


def test_static_screen_rejects_robot_out_of_bounds():
    result = static_screen(_valid_scenario(friendly_robots=(_rs(0, -4.0, 0.0), _rs(1, 50.0, 0.0))))
    assert not result.ok
    assert any("out of bounds" in v for v in result.violations)


def test_static_screen_rejects_a_robot_the_sim_cannot_be_teleported_to():
    # the sim refuses to place a robot past the field lines (full_field_bounds), so a
    # start with one there fails to set up; just inside is fine
    half = STANDARD_FIELD_DIMS.full_field_half_length
    inside = static_screen(_valid_scenario(friendly_robots=(_rs(0, -(half - 1e-3), 0.0), _rs(1, -1.0, 0.5))))
    outside = static_screen(_valid_scenario(friendly_robots=(_rs(0, -(half + 1e-3), 0.0), _rs(1, -1.0, 0.5))))
    assert inside.ok
    assert not outside.ok and any("out of bounds" in v for v in outside.violations)


def test_a_jittered_start_keeps_every_robot_inside_the_field_lines():
    half = STANDARD_FIELD_DIMS.full_field_half_length
    base = dataclasses.replace(
        _sample_bench_scenario(),
        scenario=_valid_scenario(friendly_robots=(_rs(0, -(half - 1e-3), 0.0), _rs(1, -1.0, 0.5))),
    )
    for seed in range(1, 30):
        sc = jittered(base, seed).scenario
        assert all(abs(r.x) <= half for r in (*sc.friendly_robots, *sc.enemy_robots))


def test_static_screen_rejects_overlapping_robots():
    result = static_screen(
        _valid_scenario(
            friendly_robots=(_rs(0, -4.0, 0.0), _rs(1, 0.0, 0.0)),
            enemy_robots=(_rs(0, 4.0, 0.0), _rs(1, 0.001, 0.0)),
        )
    )
    assert not result.ok
    assert any("overlaps" in v for v in result.violations)


def test_static_screen_rejects_implausible_robot_speed():
    result = static_screen(_valid_scenario(friendly_robots=(_rs(0, -4.0, 0.0), _rs(1, -1.0, 0.5, vx=50.0))))
    assert not result.ok
    assert any("implausible speed" in v for v in result.violations)


def test_static_screen_rejects_missing_team():
    result = static_screen(_valid_scenario(enemy_robots=()))
    assert not result.ok
    assert any("no enemy robots" in v for v in result.violations)


def test_bench_scenario_defaults_to_candidate_lifecycle():
    scenario = _valid_scenario()
    provenance = ScenarioProvenance(
        source_run_id="hand_authored",
        evaluator_version="abc123",
        trigger=ScenarioTrigger.HAND_AUTHORED,
        family=ScenarioFamily.KICKOFF,
    )
    bench_scenario = BenchScenario(scenario_id="test_v1", scenario=scenario, provenance=provenance)

    assert bench_scenario.lifecycle is ScenarioLifecycle.CANDIDATE
    assert bench_scenario.to_scenario() is scenario


def _sample_bench_scenario(scenario_id: str = "test_v1", **provenance_overrides) -> BenchScenario:
    provenance_defaults = dict(
        source_run_id="tournament_20260904_221937",
        evaluator_version="c7d45a3",
        trigger=ScenarioTrigger.RESTART,
        family=ScenarioFamily.DIRECT_FREE_ATTACKING,
        anchor_tick=42.5,
        source_replay=Path("replays/tournament_20260904_221937/a_vs_b.npz"),
        perspective="candidate_kicking",
    )
    provenance_defaults.update(provenance_overrides)
    scenario = _valid_scenario(
        sim_time=42.5,
        ball_x=-4.0,
        ball_y=0.75,
        friendly_robots=(_rs(0, -4.2, 0.7), _rs(1, -1.0, 0.5)),
    )
    return BenchScenario(
        scenario_id=scenario_id,
        scenario=scenario,
        provenance=ScenarioProvenance(**provenance_defaults),
        lifecycle=ScenarioLifecycle.ACTIVE,
        lead_in_s=1.5,
    )


def test_bench_scenario_to_dict_from_dict_round_trips():
    original = _sample_bench_scenario()
    restored = BenchScenario.from_dict(original.to_dict())

    assert restored.scenario_id == original.scenario_id
    assert restored.lifecycle == original.lifecycle
    assert restored.lead_in_s == original.lead_in_s
    assert restored.provenance == original.provenance
    assert restored.scenario.sim_time == original.scenario.sim_time
    assert restored.scenario.ball_x == original.scenario.ball_x
    assert restored.scenario.ball_y == original.scenario.ball_y
    assert restored.scenario.referee_command == original.scenario.referee_command
    assert restored.scenario.friendly_robots == original.scenario.friendly_robots
    assert restored.scenario.enemy_robots == original.scenario.enemy_robots
    assert restored.scenario.config_a_name == original.scenario.config_a_name
    assert restored.scenario.config_b_name == original.scenario.config_b_name


def test_bench_scenario_to_dict_from_dict_round_trips_none_referee_command():
    base = _sample_bench_scenario(scenario_id="no_referee")
    original = dataclasses.replace(base, scenario=dataclasses.replace(base.scenario, referee_command=None))

    restored = BenchScenario.from_dict(original.to_dict())

    assert restored.scenario.referee_command is None


def test_save_bank_and_load_bank_round_trips(tmp_path):
    scenarios = [_sample_bench_scenario(scenario_id="s1"), _sample_bench_scenario(scenario_id="s2", anchor_tick=10.0)]
    bank_path = tmp_path / "bank_v1.json"

    save_bank(scenarios, bank_path, bank_id="v1")
    bank_id, restored = load_bank(bank_path)

    assert bank_id == "v1"
    assert [s.scenario_id for s in restored] == ["s1", "s2"]
    assert restored[0].provenance == scenarios[0].provenance
    assert restored[1].provenance.anchor_tick == 10.0


def test_load_bank_rejects_mismatched_schema_version(tmp_path):
    scenarios = [_sample_bench_scenario()]
    bank_path = tmp_path / "bank.json"
    save_bank(scenarios, bank_path, bank_id="v1")

    payload = json.loads(bank_path.read_text())
    payload["schema_version"] = 999
    bank_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="schema_version"):
        load_bank(bank_path)


def test_save_bank_creates_parent_directories(tmp_path):
    scenarios = [_sample_bench_scenario()]
    bank_path = tmp_path / "nested" / "dir" / "bank.json"

    save_bank(scenarios, bank_path, bank_id="v1")

    assert bank_path.exists()


def test_jittered_is_reproducible_small_and_keeps_the_robot_on_the_ball():
    """rsim is deterministic, so repeats need different starts. Seed 0 is the scenario
    as authored; other seeds nudge robots away from the ball by at most
    `_JITTER_POS_M` per axis, the same way every time; the robot on the ball stays put."""
    on_ball = _rs(2, 0.09, 0.0)
    bs = BenchScenario(
        scenario_id="x",
        scenario=_valid_scenario(friendly_robots=(_rs(0, -4.0, 0.0), _rs(1, -1.0, 0.5), on_ball)),
        provenance=ScenarioProvenance(
            source_run_id="hand_authored",
            evaluator_version="abc123",
            trigger=ScenarioTrigger.HAND_AUTHORED,
            family=ScenarioFamily.KICKOFF,
        ),
    )

    assert jittered(bs, 0) is bs
    a, b = jittered(bs, 1), jittered(bs, 1)
    assert a == b
    assert a != jittered(bs, 2)
    assert static_screen(a.scenario).ok
    assert a.scenario.friendly_robots[2] == on_ball
    for old, new in zip(
        bs.scenario.friendly_robots[:2] + bs.scenario.enemy_robots,
        a.scenario.friendly_robots[:2] + a.scenario.enemy_robots,
    ):
        assert (new.x, new.y) != (old.x, old.y)
        assert abs(new.x - old.x) <= _JITTER_POS_M and abs(new.y - old.y) <= _JITTER_POS_M


def _at(scenario_id: str, *, ball_dx: float = 0.0, robot_dx: float = 0.0, **provenance_overrides) -> BenchScenario:
    """`_sample_bench_scenario` with the ball and friendly robot 1 shifted along x."""
    base = _sample_bench_scenario(scenario_id, **provenance_overrides)
    s = base.scenario
    friendly = (s.friendly_robots[0], dataclasses.replace(s.friendly_robots[1], x=s.friendly_robots[1].x + robot_dx))
    return dataclasses.replace(
        base, scenario=dataclasses.replace(s, ball_x=s.ball_x + ball_dx, friendly_robots=friendly)
    )


@pytest.mark.parametrize(
    "ball_dx, robot_dx, duplicate",
    [
        (0.0, 0.0, True),
        (_DUPLICATE_BALL_M - 1e-9, 0.0, True),
        (_DUPLICATE_BALL_M + 1e-3, 0.0, False),
        (0.0, _DUPLICATE_ROBOT_M - 1e-9, True),
        (0.0, _DUPLICATE_ROBOT_M + 1e-3, False),
    ],
)
def test_near_duplicate_boundaries(ball_dx, robot_dx, duplicate):
    assert is_near_duplicate(_at("a"), _at("b", ball_dx=ball_dx, robot_dx=robot_dx)) is duplicate


def test_same_positions_in_a_different_situation_are_not_duplicates():
    assert not is_near_duplicate(_at("a"), _at("b", family=ScenarioFamily.DIRECT_FREE_DEFENDING))
    assert not is_near_duplicate(_at("a"), _at("b", perspective="candidate_defending"))


def test_drop_near_duplicates_keeps_the_existing_bank_and_first_of_new():
    existing = [_at("old")]
    new = [_at("copy_of_old", robot_dx=0.05), _at("fresh", ball_dx=1.0), _at("copy_of_fresh", ball_dx=1.05)]

    assert [s.scenario_id for s in drop_near_duplicates(new, keep=existing)] == ["fresh"]


def test_drop_near_duplicates_renames_a_colliding_id():
    kept = drop_near_duplicates([_at("same_id", ball_dx=1.0)], keep=[_at("same_id")])

    assert [s.scenario_id for s in kept] == ["same_id_tournament_20260904_221937"]
