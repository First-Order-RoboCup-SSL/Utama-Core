"""Where `tools/evaluation/match.run_match` writes a match's replay: in its `run_dir`, beside the
match's stats and intentions, however deep `run_dir` sits under REPLAY_BASE_PATH.

Before 2026-11-12 the replay was named `<run_dir.name>/<tag>`, right only when run_dir sat directly
under replays/: an evaluate.py round folder (eval_<id>/round_<k>/) put every run's replays in
replays/round_<k>/, overwriting each other, and the round folders had none.
"""

from __future__ import annotations

import pytest

from tools.evaluation import match


class _Stop(Exception):
    pass


@pytest.mark.parametrize("depth", [("eval_20261112_120000", "round_1"), ("tournament_x",)])
def test_the_replay_lands_in_the_run_dir_beside_the_stats(tmp_path, monkeypatch, depth):
    base = tmp_path / "replays"
    run_dir = base.joinpath(*depth)
    seen = {}

    def fake_runner(a, b, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(match, "REPLAY_BASE_PATH", base)
    monkeypatch.setattr(match, "load_side", lambda name: object())
    monkeypatch.setattr(match, "sim_runner", fake_runner)
    monkeypatch.setattr(match, "enable_sim_reuse", lambda: None)
    with pytest.raises(_Stop):
        match.run_match("pendulum", "breakwater", duration_seconds=1.0, run_dir=run_dir)

    replay = base / f"{seen['replay_writer_config'].replay_name}.npz"
    stats = seen["stats_path"]
    assert replay == run_dir / "pendulum_vs_breakwater.npz"
    assert str(replay.with_suffix("")) + ".stats.json" == stats
