"""Tests for `utama_core.replay.render_window` and `replay_player.load_frames_in_range`."""

from __future__ import annotations

import pickle

import pytest

from utama_core.engine.match_log import MatchLog
from utama_core.engine.tactic import TacticTag
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, Robot
from utama_core.replay.entities import ReplayMetadata
from utama_core.replay.render_window import render_around_event, render_window
from utama_core.replay.replay_player import load_frames_in_range


def _robot(rid: int, x: float, y: float, is_friendly: bool = True) -> Robot:
    return Robot(
        id=rid,
        is_friendly=is_friendly,
        has_ball=False,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=0.0,
    )


def _frame(ts: float, robot_x: float) -> GameFrame:
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={1: _robot(1, robot_x, 0.0)},
        enemy_robots={2: _robot(2, -robot_x, 1.0, is_friendly=False)},
        ball=Ball(p=Vector3D(robot_x / 2, 0, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
    )


@pytest.fixture
def replay_file(tmp_path):
    """A .pkl replay with one frame per 0.1s from t=0.0 to t=0.9, robot drifting in x."""
    path = tmp_path / "match.pkl"
    with open(path, "wb") as f:
        pickle.dump(ReplayMetadata(my_team_is_yellow=True, exp_friendly=1, exp_enemy=1), f)
        for i in range(10):
            pickle.dump(_frame(ts=i * 0.1, robot_x=i * 0.2), f)
    return path


def test_load_frames_in_range_filters_by_timestamp(replay_file):
    frames = load_frames_in_range(replay_file, 0.2, 0.5)
    assert [round(f.ts, 2) for f in frames] == [0.2, 0.3, 0.4, 0.5]


def test_load_frames_in_range_empty_window(replay_file):
    assert load_frames_in_range(replay_file, 5.0, 6.0) == []


def test_load_frames_in_range_narrower_window_returns_fewer_frames(replay_file):
    wide = load_frames_in_range(replay_file, 0.0, 0.9)
    narrow = load_frames_in_range(replay_file, 0.3, 0.5)
    assert len(narrow) < len(wide)


def test_render_window_produces_a_png(replay_file, tmp_path):
    out = tmp_path / "window.png"
    result = render_window(replay_file, 0.0, 0.9, out)

    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 1000  # a real rendered PNG, not an empty/corrupt file
    with open(out, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def test_render_window_raises_on_empty_window(replay_file, tmp_path):
    with pytest.raises(ValueError):
        render_window(replay_file, 5.0, 6.0, tmp_path / "empty.png")


def test_render_around_event_resolves_timestamp_from_match_log(replay_file, tmp_path):
    log = MatchLog()
    log.intention(tick=5, sim_time=0.5, tactic_id="steal_ball", robot_ids=(1,), tag=TacticTag.DEFENSE)
    log_path = tmp_path / "match_log.jsonl"
    log.to_jsonl(log_path)

    out = tmp_path / "around.png"
    result = render_around_event(replay_file, log_path, event_index=0, pad_seconds=0.15, out_path=out)

    assert result == str(out)
    assert out.exists()


def test_render_around_event_bad_index_raises(replay_file, tmp_path):
    log = MatchLog()
    log.intention(tick=1, sim_time=0.1, tactic_id="a", robot_ids=(1,), tag=TacticTag.MIXED)
    log_path = tmp_path / "match_log.jsonl"
    log.to_jsonl(log_path)

    with pytest.raises(IndexError):
        render_around_event(replay_file, log_path, event_index=5, pad_seconds=0.1, out_path=tmp_path / "x.png")
