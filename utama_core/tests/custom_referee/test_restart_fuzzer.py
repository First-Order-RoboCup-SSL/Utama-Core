"""Tests for `RestartFuzzingReferee` (utama_core/custom_referee/restart_fuzzer.py).

Covers:
  - Seed determinism: same seed → identical injection schedule/kinds/positions.
  - Injections only fire during live play (never mid-restart, never right
    after a goal before the queued kickoff resolves).
  - Every drawn `designated_position` is legal (inside field with margin,
    outside both defense areas with margin) — checked over several hundred
    seeded draws.
  - The `kinds` filter is respected.

No rsim match: how many injections a match gets depends on how play unfolds, which differs by CPU,
so such a test failed on CI while passing locally.
"""

from __future__ import annotations

import math

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.restart_fuzzer import (
    ALL_KINDS,
    KIND_BALL_PLACEMENT_DIRECT_FREE,
    KIND_FORCE_START,
    KIND_PREPARE_KICKOFF,
    RestartFuzzingReferee,
)
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run.strategy_runner import StrategyRunner
from utama_core.strategy.kernel_strategy import build_default_kernel_strategy

GEO = RefereeGeometry.from_field_dims(STANDARD_FIELD_DIMS)

# ---------------------------------------------------------------------------
# Helpers (mirrors utama_core/tests/custom_referee/test_custom_referee.py)
# ---------------------------------------------------------------------------


def _ball(x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> Ball:
    return Ball(p=Vector3D(x, y, 0.0), v=Vector3D(vx, vy, 0.0), a=Vector3D(0, 0, 0))


def _robot(robot_id: int, x: float, y: float, is_friendly: bool) -> Robot:
    return Robot(
        id=robot_id,
        is_friendly=is_friendly,
        has_ball=False,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=0.0,
    )


def _frame(
    ts: float,
    ball_xy: tuple[float, float] = (0.0, 0.0),
    friendly: dict | None = None,
    enemy: dict | None = None,
) -> GameFrame:
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=False,
        friendly_robots=friendly or {},
        enemy_robots=enemy or {},
        ball=_ball(*ball_xy),
        referee=None,
    )


def _far_from_ball_robots() -> dict[int, Robot]:
    """A handful of robots far from the ball/centre so `_all_robots_clear()`
    is satisfied immediately and STOP auto-advances without needing to wait
    out `_STOP_CLEAR_TIMEOUT_SECONDS`."""
    return {
        0: _robot(0, -3.0, 2.0, True),
        1: _robot(1, -3.0, -2.0, True),
        2: _robot(2, 3.0, 2.0, True),
    }


def _run_ticks(referee: RestartFuzzingReferee, n_ticks: int, dt: float = 1 / 60) -> None:
    """Drive `referee.step()` for `n_ticks` synthetic ticks, holding the ball
    at the origin and robots far away throughout (so any STOP auto-advances
    immediately) — enough to exercise scheduling/injection logic without a
    real simulator.
    """
    referee.seed_clock(0.0, RefereeCommand.NORMAL_START)
    friendly = _far_from_ball_robots()
    enemy = {10: _robot(10, 3.0, -2.0, False)}
    t = 0.0
    for _ in range(n_ticks):
        frame = _frame(t, ball_xy=(0.0, 0.0), friendly=friendly, enemy=enemy)
        referee.step(frame, t)
        t += dt


# ---------------------------------------------------------------------------
# Seed determinism
# ---------------------------------------------------------------------------


def test_seed_determinism_same_schedule_and_positions():
    ref_a = RestartFuzzingReferee.from_profile_name(
        "simulation", seed=42, interval_s=(0.5, 1.0), n_robots_yellow=3, n_robots_blue=3
    )
    ref_b = RestartFuzzingReferee.from_profile_name(
        "simulation", seed=42, interval_s=(0.5, 1.0), n_robots_yellow=3, n_robots_blue=3
    )

    _run_ticks(ref_a, 600)  # 10s @ 60Hz
    _run_ticks(ref_b, 600)

    assert len(ref_a.injections) > 0, "test setup produced no injections to compare"
    assert ref_a.injections == ref_b.injections


def test_different_seed_different_schedule():
    ref_a = RestartFuzzingReferee.from_profile_name(
        "simulation", seed=1, interval_s=(0.5, 1.0), n_robots_yellow=3, n_robots_blue=3
    )
    ref_b = RestartFuzzingReferee.from_profile_name(
        "simulation", seed=2, interval_s=(0.5, 1.0), n_robots_yellow=3, n_robots_blue=3
    )

    _run_ticks(ref_a, 600)
    _run_ticks(ref_b, 600)

    assert ref_a.injections != ref_b.injections


# ---------------------------------------------------------------------------
# Injections only during live play
# ---------------------------------------------------------------------------


def test_injection_never_starts_mid_restart():
    """Force the referee into a non-live-play command right when an
    injection would be due, and confirm the fuzzer waits rather than
    injecting on top of an in-flight restart."""
    referee = RestartFuzzingReferee.from_profile_name(
        "simulation", seed=7, interval_s=(0.1, 0.1), n_robots_yellow=3, n_robots_blue=3
    )
    referee.seed_clock(0.0, RefereeCommand.NORMAL_START)

    friendly = _far_from_ball_robots()
    enemy = {10: _robot(10, 3.0, -2.0, False)}

    # Put the referee into a real, in-flight restart (DIRECT_FREE_YELLOW)
    # that will not auto-advance immediately (kicker/defenders not in
    # position), then drive ticks through the scheduled injection time and
    # confirm no injection fires while command != live play.
    referee.force_command(RefereeCommand.DIRECT_FREE_YELLOW, 0.0)

    t = 0.0
    for _ in range(30):  # 0.5s @ 60Hz — well past the 0.1s interval
        frame = _frame(t, ball_xy=(0.0, 0.0), friendly=friendly, enemy=enemy)
        result = referee.step(frame, t)
        if result.referee_command in (RefereeCommand.NORMAL_START, RefereeCommand.FORCE_START):
            # Auto-advanced back to live play — from here on injections are
            # legitimately allowed again, so stop asserting.
            break
        assert len(referee.injections) == 0, "fuzzer injected while a restart was already in flight"
        t += 1 / 60


def test_all_injections_recorded_during_live_play_command_at_time_of_injection():
    """Every recorded injection's `sim_time` must correspond to a tick where
    the referee's command-before-injection was live play (NORMAL_START or
    FORCE_START) with no restart already queued — i.e. `_is_live_play` held
    at the moment `set_command` was first called for that injection."""
    referee = RestartFuzzingReferee.from_profile_name(
        "simulation", seed=3, interval_s=(0.2, 0.4), n_robots_yellow=3, n_robots_blue=3
    )
    referee.seed_clock(0.0, RefereeCommand.NORMAL_START)

    friendly = _far_from_ball_robots()
    enemy = {10: _robot(10, 3.0, -2.0, False)}
    t = 0.0
    live_play_ticks: set[float] = set()
    for _ in range(1200):  # 20s @ 60Hz
        frame = _frame(t, ball_xy=(0.0, 0.0), friendly=friendly, enemy=enemy)
        result_before = (referee._state.command, referee._state.next_command)
        referee.step(frame, t)
        if result_before[0] in (RefereeCommand.NORMAL_START, RefereeCommand.FORCE_START) and result_before[1] is None:
            live_play_ticks.add(round(t, 6))
        t += 1 / 60

    assert len(referee.injections) > 0
    for inj in referee.injections:
        assert round(inj.sim_time, 6) in live_play_ticks, (
            f"injection at sim_time={inj.sim_time} did not occur while the referee was in live play "
            "with no restart queued"
        )


# ---------------------------------------------------------------------------
# Legal positions
# ---------------------------------------------------------------------------


def _assert_legal(x: float, y: float) -> None:
    assert GEO.is_in_field(x, y), f"({x}, {y}) is not inside the field"
    # Field-line margin: strictly inside the shrunk rectangle used for sampling.
    assert abs(x) <= GEO.half_length - 0.25 + 1e-9
    assert abs(y) <= GEO.half_width - 0.25 + 1e-9
    assert GEO.distance_to_left_defense_area(x, y) >= 0.2 - 1e-9, f"({x}, {y}) too close to left defense area"
    assert GEO.distance_to_right_defense_area(x, y) >= 0.2 - 1e-9, f"({x}, {y}) too close to right defense area"


def test_random_legal_position_always_legal_many_seeded_draws():
    for seed in range(300):
        referee = RestartFuzzingReferee.from_profile_name(
            "simulation", seed=seed, interval_s=(1.0, 1.0), n_robots_yellow=3, n_robots_blue=3
        )
        x, y = referee._random_legal_position()
        _assert_legal(x, y)


def test_ball_placement_injections_use_legal_positions_end_to_end():
    referee = RestartFuzzingReferee.from_profile_name(
        "simulation",
        seed=11,
        interval_s=(0.2, 0.3),
        kinds=(KIND_BALL_PLACEMENT_DIRECT_FREE,),
        n_robots_yellow=3,
        n_robots_blue=3,
    )
    _run_ticks(referee, 1200)  # 20s @ 60Hz

    ball_placement_injections = [inj for inj in referee.injections if inj.kind == KIND_BALL_PLACEMENT_DIRECT_FREE]
    assert len(ball_placement_injections) > 0
    for inj in ball_placement_injections:
        assert inj.position is not None
        _assert_legal(*inj.position)


# ---------------------------------------------------------------------------
# kinds filter
# ---------------------------------------------------------------------------


def test_kinds_filter_restricts_injected_kinds():
    for only_kind in ALL_KINDS:
        referee = RestartFuzzingReferee.from_profile_name(
            "simulation",
            seed=5,
            interval_s=(0.1, 0.2),
            kinds=(only_kind,),
            n_robots_yellow=3,
            n_robots_blue=3,
        )
        _run_ticks(referee, 1200)  # 20s @ 60Hz

        assert len(referee.injections) > 0, f"no injections at all for kinds=({only_kind},)"
        assert all(inj.kind == only_kind for inj in referee.injections)


def test_kinds_filter_rejects_unknown_kind():
    import pytest

    with pytest.raises(ValueError):
        RestartFuzzingReferee.from_profile_name("simulation", seed=1, kinds=("not_a_real_kind",))


def test_invalid_interval_raises():
    import pytest

    with pytest.raises(ValueError):
        RestartFuzzingReferee.from_profile_name("simulation", seed=1, interval_s=(5.0, 1.0))
