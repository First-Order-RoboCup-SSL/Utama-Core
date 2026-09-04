"""Tests for `utama_core.replay.bench_scenario` (bench scenario schema,
provenance, lifecycle, and the static validity screen).

See that module's docstring for the design constraints these tests pin:
policy-agnostic field state only, conservative lifecycle default, and a
static screen that catches malformed snapshots before any dynamic (play-
forward) screening or scoring happens.
"""

from __future__ import annotations

from pathlib import Path

from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.bench_scenario import (
    BenchScenario,
    ScenarioFamily,
    ScenarioLifecycle,
    ScenarioProvenance,
    ScenarioTrigger,
    static_screen,
)
from utama_core.replay.scenario import RobotState, Scenario


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
