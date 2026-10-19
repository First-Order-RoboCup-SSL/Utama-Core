"""jev_tournament's guard against recording a match the scripted fallback played (server down)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import jev_tournament


class _Played:
    def __init__(self, questions: int, fallback: int):
        self.questions, self.fallback = questions, fallback

    def summary(self) -> dict:
        return {"policy": "openjev", "questions": self.questions, "fallback": self.fallback}

    def close(self) -> None:
        pass


def _play(monkeypatch, tmp_path, questions: int, fallback: int) -> None:
    def fake_run_match(*args, factory_a, **kwargs):
        factory_a.keywords["on_build"](_Played(questions, fallback))
        return SimpleNamespace(score_a=1, score_b=0, stats={})

    monkeypatch.setattr(jev_tournament, "run_match", fake_run_match)
    jev_tournament.play(
        "split_shape",
        ("_Rk", True, True),
        tmp_path,
        engine="flat",
        url="http://unused",
        duration=1.0,
        decision_hz=1.0,
        render=False,
    )


def test_match_with_fallback_at_the_limit_is_recorded(monkeypatch, tmp_path):
    _play(monkeypatch, tmp_path, questions=1000, fallback=10)
    assert (tmp_path / "jevflat_vs_split_shape_Rk.result.json").exists()


def test_match_with_fallback_over_the_limit_stops_the_run_unrecorded(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="11 of 1000 questions fell back"):
        _play(monkeypatch, tmp_path, questions=1000, fallback=11)
    assert not (tmp_path / "jevflat_vs_split_shape_Rk.result.json").exists()
    assert (tmp_path / "jevflat_vs_split_shape_Rk.invalid.json").exists()
