import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utama_core.entities.referee.referee_command import RefereeCommand as C

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "replay_trace.py"
_spec = importlib.util.spec_from_file_location("replay_trace", _TOOL)
replay_trace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay_trace)


def _replay(tmp_path, n=60):
    commands = np.array([C.NORMAL_START.value] * 30 + [C.STOP.value] * 30, dtype=np.int8)
    friendly = np.zeros((n, 2, 2))
    friendly[:, 0] = (1.3, 0.0)
    friendly[:, 1] = (3.0, 0.0)
    enemy = np.full((n, 1, 2), (0.0, 2.0))
    path = tmp_path / "a_vs_b.npz"
    np.savez(
        path,
        ts=50.0 + np.arange(n) / 60.0,
        has_referee=np.ones(n, bool),
        referee_command=commands,
        ball_p=np.tile([1.0, 0.0, 0.0], (n, 1)),
        ball_v=np.zeros((n, 3)),
        friendly_p=friendly,
        friendly_ids=np.array([4, 5]),
        friendly_has_ball=np.zeros((n, 2), bool),
        enemy_p=enemy,
        enemy_ids=np.array([2]),
        enemy_has_ball=np.zeros((n, 1), bool),
    )
    statuses = [(i, SimpleNamespace(status_message="Keep-out circle violation" if i >= 30 else None)) for i in range(n)]
    with open(tmp_path / "a_vs_b.sparse_referee.pkl", "wb") as f:
        pickle.dump(statuses, f)
    return path


def test_trace_prints_each_referee_change_with_its_reason_and_the_nearest_robots(tmp_path):
    lines = replay_trace.trace(_replay(tmp_path), 0.0, 1.0, every=1000)

    assert len(lines) == 2
    assert lines[0].split()[:2] == ["0.00", "NORMAL_START"]
    assert "near F 4@0.30" in lines[0] and "E 2@2.24" in lines[0]
    assert lines[1].split()[:2] == ["0.50", "STOP"]
    assert lines[1].endswith("| Keep-out circle violation")


def test_trace_keeps_to_the_window(tmp_path):
    lines = replay_trace.trace(_replay(tmp_path), 0.6, 0.8, every=6)
    assert [line.split()[0] for line in lines] == ["0.60", "0.70", "0.80"]
