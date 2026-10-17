"""`tools/evaluation/explain.py` on a synthetic run dir of real columnar replays: the side assignment
(a candidate's numbers come from its own matches, against any opponent), the goals from the final
referee state, the paired goal difference, the four files, and `evaluate.py --explain` off by default."""

import json

import pytest

from tools import pitch_maps as pm
from tools.evaluation import evaluate, explain
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, Robot
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.replay.columnar_writer import (
    ColumnarReplayWriter,
    ColumnarReplayWriterConfig,
)

CAND, CHAMP, OPP1, OPP2 = (
    "build_cand_kernel_strategy",
    "build_champ_kernel_strategy",
    "build_opp1_kernel_strategy",
    "build_opp2_kernel_strategy",
)


def _robot(rid, x, y, friendly):
    return Robot(
        id=rid,
        is_friendly=friendly,
        has_ball=False,
        p=Vector2D(x, y),
        v=Vector2D(0.0, 0.0),
        a=Vector2D(0.0, 0.0),
        orientation=0.0,
    )


def _referee(score_yellow, score_blue):
    return RefereeData(
        source_identifier=None,
        time_sent=0.0,
        time_received=0.0,
        referee_command=RefereeCommand.NORMAL_START,
        referee_command_timestamp=0.0,
        stage=Stage.NORMAL_FIRST_HALF,
        stage_time_left=100.0,
        blue_team=TeamInfo("blue", score=score_blue),
        yellow_team=TeamInfo("yellow", score=score_yellow),
    )


def _write_match(run_dir, a, b, score_a, score_b, seed=None):
    """A three-tick match, config_a (yellow, friendly) scoring `score_a`; the team with the ball is
    always the friendly one. Written the way `evaluate.py`/`round_robin.py` name their files."""
    stem = f"{pm.sr.short(a)}_vs_{pm.sr.short(b)}" + ("" if seed is None else f"_s{seed}")
    path = run_dir / f"{stem}.npz"
    cfg = ColumnarReplayWriterConfig(replay_name=stem, overwrite_existing=True, checkpoint_every_s=0)
    writer = ColumnarReplayWriter(cfg, my_team_is_yellow=True, exp_friendly=3, exp_enemy=3, path=path)
    for i in range(3):
        writer.write_frame(
            GameFrame(
                ts=i * 0.1,
                my_team_is_yellow=True,
                my_team_is_right=True,
                friendly_robots={
                    0: _robot(0, 4.3, 0.0, True),
                    1: _robot(1, 1.0, 0.0, True),
                    2: _robot(2, -1.0, 2.0, True),
                },
                enemy_robots={
                    10: _robot(10, -1.5, 0.0, False),
                    11: _robot(11, 2.0, -2.0, False),
                    12: _robot(12, 0.0, 3.0, False),
                },
                ball=Ball(p=Vector3D(1.0, 0.0, 0.0), v=Vector3D(0.0, 0.0, 0.0), a=Vector3D(0.0, 0.0, 0.0)),
                referee=_referee(score_a, score_b),
            )
        )
    writer.close()
    (run_dir / f"{stem}.stats.json").write_text(
        json.dumps({"fouls": [{"rule": "crashing", "sim_time": 0.0, "side": "friendly", "ball_xy": [1.0, 0.0]}]})
    )
    return path


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    _write_match(d, CAND, OPP1, 2, 0, seed=1)  # candidate friendly: 2-0
    _write_match(d, OPP1, CAND, 1, 3, seed=1)  # candidate enemy: 3-1
    _write_match(d, CHAMP, OPP1, 0, 1, seed=1)  # champion friendly: 0-1
    _write_match(d, OPP1, CHAMP, 1, 1, seed=1)  # champion enemy: 1-1
    _write_match(d, CAND, CHAMP, 1, 1, seed=3)  # head to head: 1-1
    _write_match(d, OPP1, OPP2, 5, 0, seed=1)  # neither side: must not count for either
    return d


def test_explain_writes_the_four_files(run_dir, tmp_path):
    out = tmp_path / "out"
    explain.explain(run_dir, CAND, CHAMP, out, workers=1)
    for name in ("positions.png", "shots.png", "fouls.png"):
        data = (out / name).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) < 250_000
    assert json.loads((out / "summary.json").read_text())["candidate"] == "cand"


def test_each_side_counts_its_own_matches_and_goals(run_dir, tmp_path):
    s = explain.explain(run_dir, CAND, CHAMP, tmp_path / "out", workers=1)
    cand, champ = s["sides"]["cand"], s["sides"]["champ"]
    assert cand["matches"] == 3 and champ["matches"] == 3  # the OPP1-OPP2 match is in neither
    # candidate: 2-0, 3-1, 1-1 -> for (2+3+1)/3, against (0+1+1)/3
    assert cand["goals_for_per_match"] == pytest.approx(2.0)
    assert cand["goals_against_per_match"] == pytest.approx(2 / 3)
    # champion: 0-1, 1-1, 1-1 -> for (0+1+1)/3, against (1+1+1)/3
    assert champ["goals_for_per_match"] == pytest.approx(2 / 3)
    assert champ["goals_against_per_match"] == pytest.approx(1.0)


def test_paired_difference_uses_same_opponent_seed_and_colour(run_dir, tmp_path):
    s = explain.explain(run_dir, CAND, CHAMP, tmp_path / "out", workers=1)
    paired = s["paired"]
    assert set(paired) == {"opp1", "head-to-head"}  # opp2 never met either side
    # both pairs (seed 1, each colour): candidate +2 and +2, champion -1 and 0
    assert paired["opp1"] == {"candidate": 4, "champion": -1, "matches": 2}
    assert paired["head-to-head"] == {"candidate": 0, "champion": 0, "matches": 1}


def test_explain_refuses_when_a_side_has_no_replay(run_dir, tmp_path):
    with pytest.raises(FileNotFoundError):
        explain.explain(run_dir, "build_nobody_kernel_strategy", CHAMP, tmp_path / "out", workers=1)


def test_evaluate_explain_is_off_by_default():
    assert evaluate.build_parser().parse_args(["build_x_kernel_strategy"]).explain is False
    assert evaluate.build_parser().parse_args(["build_x_kernel_strategy", "--explain"]).explain is True


def test_an_explain_failure_never_escapes(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("no replays")

    monkeypatch.setattr(explain, "explain", boom)
    evaluate._explain(tmp_path, "eval_test", [CAND], CHAMP)  # must print and return, not raise
