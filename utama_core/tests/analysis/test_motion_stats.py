"""Tests for `utama_core.analysis.motion_stats`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from utama_core.analysis.motion_stats import (
    compute_motion_metrics_from_arrays,
    compute_motion_metrics_from_replay,
    enrich_tournament_directory,
)


def test_compute_motion_metrics_from_arrays():
    # 3 ticks, 2 friendly robots, 1 enemy robot
    # Tick 0: friendly overlap (dist=0.10, two_r=0.18 -> clearance=-0.08m)
    # Tick 1: near miss friendly vs enemy (dist=0.20 -> clearance=0.02m)
    # Tick 2: completely safe (dist >= 1.0m)
    fp = np.zeros((3, 2, 2), dtype=np.float32)
    ep = np.zeros((3, 1, 2), dtype=np.float32)

    fp[0, 0] = [0.0, 0.0]
    fp[0, 1] = [0.10, 0.0]
    ep[0, 0] = [5.0, 5.0]

    fp[1, 0] = [0.0, 0.0]
    fp[1, 1] = [2.0, 2.0]
    ep[1, 0] = [0.20, 0.0]

    fp[2, 0] = [0.0, 0.0]
    fp[2, 1] = [2.0, 2.0]
    ep[2, 0] = [5.0, 5.0]

    res = compute_motion_metrics_from_arrays(fp, ep)
    assert res["robot_overlap_ticks"] == 1
    assert res["robot_overlap_incidents"] == 1
    assert res["near_miss_ticks"] == 1
    assert res["teammate_overlap_ticks"] == {"friendly": 1, "enemy": 0}
    assert res["teammate_overlap_incidents"] == {"friendly": 1, "enemy": 0}
    assert res["opponent_overlap_ticks"] == 0
    assert res["opponent_overlap_incidents"] == 0
    assert res["worst_clearance_m"] == pytest.approx(-0.08, abs=1e-3)
    assert res["total_ticks"] == 3


def test_enrich_tournament_directory(tmp_path: Path):
    fp = np.zeros((2, 2, 2), dtype=np.float32)
    ep = np.zeros((2, 1, 2), dtype=np.float32)
    fp[0, 0] = [0.0, 0.0]
    fp[0, 1] = [0.10, 0.0]
    ep[0, 0] = [2.0, 2.0]
    fp[1, 0] = [0.0, 0.0]
    fp[1, 1] = [2.0, 2.0]
    ep[1, 0] = [5.0, 5.0]

    npz_path = tmp_path / "strat_a_vs_strat_b.npz"
    np.savez(npz_path, friendly_p=fp, enemy_p=ep)

    stats_path = tmp_path / "strat_a_vs_strat_b.stats.json"
    with open(stats_path, "w") as f:
        json.dump({"rule_event_counts": {}}, f)

    enrich_tournament_directory(tmp_path)

    with open(stats_path) as f:
        data = json.load(f)
    assert data["robot_overlap_ticks"] == 1
    assert data["robot_overlap_incidents"] == 1
    assert data["teammate_overlap_incidents"] == {"friendly": 1, "enemy": 0}
    assert data["worst_clearance_m"] == pytest.approx(-0.08, abs=1e-3)
