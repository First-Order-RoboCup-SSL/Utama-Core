"""Pure tests for the benchmark's `--figures` output: no rsim run, a synthetic 20-tick record."""

import math
from types import SimpleNamespace

from tools.motion_planning_benchmark import (
    SCHEMES,
    BenchmarkManager,
    Scenario,
    write_figures,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
TICKS = 20


def _trace(friendly_speed: float, opponent_dy: float) -> dict:
    """Friendly robot 0 runs along +x; an opponent stands still at (0, 0.3) and another runs
    across at `opponent_dy` per tick, so one scheme's trace has clearance and the other's has none."""
    t = [k / 60.0 for k in range(TICKS)]
    friendly = [(-2.0 + friendly_speed * tk, 0.0) for tk in t]
    opponent = [(0.0, 0.3 + opponent_dy * k) for k in range(TICKS)]
    static = [(1.0, -0.5)] * TICKS
    return {
        "t": t,
        "robots": {
            "friendly:0": {"xy": friendly, "speed": [friendly_speed] * TICKS},
            "enemy:0": {"xy": opponent, "speed": [abs(opponent_dy) * 60.0] * TICKS},
            "enemy:1": {"xy": static, "speed": [0.0] * TICKS},
        },
        "clearance": [math.dist(friendly[k], opponent[k]) - 0.18 for k in range(TICKS)],
    }


def _aggregate(scenario: str, scheme: str) -> dict:
    return {
        "scenario": scenario,
        "scheme": scheme,
        "passed": scheme == "fpp",
        "sim_time_mean_s": 3.0,
        "path_length_ratio_mean": 1.02,
        "min_clearance_worst_m": 0.1,
        "collision_events_total": 0,
        "final_error_max_worst_m": 0.05,
        "stopped_while_unreached_ratio_mean": 0.01,
        "controller_mean_ms": 0.2,
        "controller_p95_ms_mean": 0.4,
        "max_speed_worst_mps": 1.5,
    }


def _scenario(name: str, friendly_targets, enemy_starts) -> Scenario:
    return Scenario(
        name=name,
        description="Synthetic scenario for the figure test.",
        friendly_starts=((-2.0, 0.0),),
        friendly_targets=friendly_targets,
        enemy_starts=enemy_starts,
    )


def test_figures_are_written_as_png(tmp_path):
    schemes = [SCHEMES[0], SCHEMES[1]]
    scenario = _scenario("synthetic_crossing", ((2.0, 0.0),), ((0.0, 0.3), (1.0, -0.5)))
    payload = {
        "schemes": schemes,
        "scenarios": [scenario.name],
        "aggregates": [_aggregate(scenario.name, s) for s in schemes],
    }
    traces = {
        (scenario.name, schemes[0]): _trace(friendly_speed=0.2, opponent_dy=0.01),
        (scenario.name, schemes[1]): _trace(friendly_speed=0.1, opponent_dy=0.0),
    }

    written = write_figures(payload, traces, tmp_path, {scenario.name: scenario})

    assert [p.name for p in written] == ["trajectories_synthetic_crossing.png", "summary.png"]
    for path in written:
        data = path.read_bytes()
        assert data[:8] == PNG_MAGIC, path
        assert len(data) > 1000, path


def test_single_robot_scenario_without_obstacles_still_writes_figures(tmp_path):
    scheme = SCHEMES[0]
    scenario = _scenario("synthetic_direct", ((2.0, 0.0),), ())
    trace = _trace(friendly_speed=0.2, opponent_dy=0.0)
    trace["robots"] = {"friendly:0": trace["robots"]["friendly:0"]}
    trace["clearance"] = [None] * TICKS
    payload = {"schemes": [scheme], "scenarios": [scenario.name], "aggregates": [_aggregate(scenario.name, scheme)]}

    written = write_figures(payload, {(scenario.name, scheme): trace}, tmp_path, {scenario.name: scenario})

    assert (tmp_path / "trajectories_synthetic_direct.png").read_bytes()[:8] == PNG_MAGIC
    assert written[-1] == tmp_path / "summary.png"


def test_default_manager_records_nothing_and_recording_captures_each_tick():
    scenario = _scenario("synthetic_default", ((2.0, 0.0),), ((0.0, 0.5),))
    assert BenchmarkManager(scenario).trace is None

    manager = BenchmarkManager(scenario, record_trace=True)
    robots = {
        ("friendly", 0): SimpleNamespace(p=SimpleNamespace(x=0.0, y=0.0), v=SimpleNamespace(x=0.3, y=0.4)),
        ("enemy", 0): SimpleNamespace(p=SimpleNamespace(x=0.0, y=0.5), v=SimpleNamespace(x=0.0, y=0.0)),
    }
    manager._record_tick(0.0, robots, min_distance=0.5)

    assert manager.trace["t"] == [0.0]
    assert manager.trace["robots"]["friendly:0"] == {"xy": [(0.0, 0.0)], "speed": [0.5]}
    assert manager.trace["robots"]["enemy:0"]["speed"] == [0.0]
    # Clearance is the nearest centre distance minus both radii, the same as `min_clearance_m`.
    assert math.isclose(manager.trace["clearance"][0], 0.5 - 0.18)
