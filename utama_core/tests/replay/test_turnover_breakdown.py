import math
from types import SimpleNamespace

import pytest

from utama_core.replay.turnover_breakdown import (
    _REACH_M,
    _PassTracker,
    breakdown,
    receptions,
)


def _turnover(kind: str, regained_after_s, tactic: str = "GiveAndGoTactic") -> dict:
    return {"kind": kind, "t": 0.0, "tactic": tactic, "just_restarted": False, "regained_after_s": regained_after_s}


def test_breakdown_counts_only_real_losses():
    """A turnover won back within 1.0s is two robots on one ball (the nearest robot
    flips), and a `during_stoppage` handover is already counted as the foul/ball-out
    restart that caused it — neither is a real loss. Everything else is, including
    restarts given away."""
    results = [
        {
            "match": "a_vs_b",
            "acc_turnovers": 5,
            "turnovers": [
                _turnover("tackled", 1.0),  # flicker: excluded
                _turnover("tackled", 1.01, tactic="PressAndContainTactic"),
                _turnover("tackled", None, tactic="PressAndContainTactic"),  # never won back
                _turnover("pass_intercepted", 4.0),
                _turnover("during_stoppage", None, tactic="restart override"),  # excluded
            ],
            "restarts": [
                {"kind": "foul", "t": 0.0, "rule": "Excessive dribbling", "tactic": "GiveAndGoTactic"},
                {"kind": "ball_out_after_kick", "t": 0.0, "rule": "unknown", "tactic": "GiveAndGoTactic"},
            ],
        },
        {"match": "c_vs_d", "acc_turnovers": 0, "turnovers": [], "restarts": []},
    ]

    b = breakdown(results)

    assert b["matches"] == 2
    assert b["raw_turnovers"] == 5
    assert b["real_losses"] == 5
    assert b["real_losses_per_match"] == 2.5
    assert b["by_kind"] == {"tackled": 2, "pass_intercepted": 1, "foul": 1, "ball_out_after_kick": 1}
    assert b["fouls_by_rule"] == {"Excessive dribbling": 1}
    assert b["by_tactic"] == {"GiveAndGoTactic": 3, "PressAndContainTactic": 2}


def _frame(ts, ball_xy, ball_v, robots):
    """robots: {id: (x, y, orientation_rad, has_ball)}"""
    v2 = lambda x, y: SimpleNamespace(x=x, y=y)  # noqa: E731
    return SimpleNamespace(
        ts=ts,
        ball=SimpleNamespace(p=v2(*ball_xy), v=v2(*ball_v)),
        friendly_robots={
            rid: SimpleNamespace(p=v2(x, y), v=v2(0.0, 0.0), orientation=o, has_ball=hb)
            for rid, (x, y, o, hb) in robots.items()
        },
    )


def _run_pass(receiver_y: float, caught: bool) -> dict:
    """Passer 0 at the origin kicks along +x at 3 m/s toward receiver 1 at (1, receiver_y),
    which faces straight back at the ball. The ball ends past the receiver and goes loose."""
    tracker = _PassTracker(0.0, passer=0, tactic="GiveAndGoTactic", release_xy=(0.0, 0.0))
    finished = None
    for i in range(1, 240):
        t = i / 60
        x = 3.0 * t
        has = caught and abs(x - 1.0) < 0.05
        finished = tracker.step(
            _frame(t, (x, 0.0), (3.0, 0.0), {0: (0.0, 0.0, 0.0, False), 1: (1.0, receiver_y, math.pi, has)}),
            live=True,
            enemy_has_it=False,
        )
        if finished is not None:
            break
    return finished


def test_pass_reaching_a_teammate_is_received_or_missed_by_reach():
    """The reception boundary is `_REACH_M` from the receiver's centre: a pass that never
    touched the dribbler but passed within it is a missed reception, one just outside is
    off target; contact is received regardless."""
    assert _run_pass(0.0, caught=True)["outcome"] == "received"
    missed = _run_pass(_REACH_M - 0.01, caught=False)
    assert missed["outcome"] == "missed_reception"
    assert missed["receiver"] == 1
    assert missed["facing_off_deg"] == pytest.approx(0.0, abs=1e-6)
    assert _run_pass(_REACH_M + 0.01, caught=False)["outcome"] == "off_target"


def test_receptions_catch_rate_by_facing():
    passes = [
        {"outcome": "received", "tactic": "A", "facing_off_deg": 5.0},
        {
            "outcome": "missed_reception",
            "tactic": "A",
            "facing_off_deg": 15.0,
            "ball_speed": 4.0,
            "receiver_speed": 0.0,
            "distance_m": 0.1,
        },
        {"outcome": "off_target", "tactic": "A", "facing_off_deg": 90.0},
    ]
    rec = receptions([{"passes": passes}])
    assert rec["catch_rate"] == 0.5  # off-target passes are not reachable
    assert rec["received_by_facing"]["<10deg"] == [1, 1]
    assert rec["received_by_facing"]["10-20deg"] == [0, 1]
    assert rec["received_by_facing"][">=45deg"] == [0, 0]
